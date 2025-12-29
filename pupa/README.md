# PUPA GEPA Implementation

This directory contains the GEPA implementation for the PUPA (Privacy-conscious User Prompts with Annotations) benchmark, adapted from the HotpotQA implementation.

## Overview

**PUPA Task**: Privacy-Conscious Delegation - addressing real-world user queries using an ensemble of trusted and untrusted models while minimizing leakage of personally identifiable information (PII).

**PAPILLON System**: A compound AI system with 3 stages:
1. **Query Rewriter** (Trusted Model): Rewrites user query to remove PII
2. **Untrusted Model Call**: Gets response from powerful cloud LLM using rewritten query
3. **Response Rewriter** (Trusted Model): Refines response using original context

**Dataset**: 111 train / 111 validation / 221 test examples

**Metrics**:
- Quality Score: How well the response addresses the user's query
- Leakage Score: How much PII is exposed to untrusted models (lower is better)
- Aggregate Score: Quality + (1 - Leakage), normalized to [0, 1]

## Files

- `pupa_program.py` - PAPILLON 3-stage pipeline implementation
- `pupa_data.py` - PUPA dataset loader
- `pupa_metric.py` - Quality and PII leakage metrics with GEPA-compatible feedback
- `run_gepa_pupa.py` - Main GEPA training script
- `run_gepa_pupa_compare.py` - Parallel comparison of GEPA variants
- `sanity_pupa.py` - Sanity check script to test the pipeline
- `job.pupa_compare.sbatch` - SLURM batch job script

## Setup

### 1. Install Dependencies

```bash
pip install dspy datasets transformers
```

### 2. Get PUPA Dataset

**Option A: Quick Setup (Recommended)**

Run the conversion script to download and prepare the data automatically:

```bash
cd /work1/krishnamurthy/arvind/adBO/pupa

# Download PAPILLON repo and convert data
git clone https://github.com/Columbia-NLP-Lab/PAPILLON/ /tmp/PAPILLON

# Convert CSV to JSON splits (111/111/221)
python3 convert_pupa_data.py \
  --csv_path /tmp/PAPILLON/pupa/PUPA_New.csv \
  --output_dir ./data
```

This creates `train.json`, `dev.json`, and `test.json` in the `./data` directory.

**Option B: Manual Setup**

1. Clone the PAPILLON repository:
   ```bash
   git clone https://github.com/Columbia-NLP-Lab/PAPILLON/
   ```

2. The dataset is in `PAPILLON/pupa/PUPA_New.csv` (664 examples total)

3. Run the conversion script:
   ```bash
   python3 convert_pupa_data.py --csv_path /path/to/PAPILLON/pupa/PUPA_New.csv --output_dir ./data
   ```

**Verify Dataset**

```bash
python3 -c "
from pupa_data import load_pupa_splits
train, dev, test = load_pupa_splits(seed=0, data_dir='./data')
print(f'Loaded: {len(train)} train, {len(dev)} dev, {len(test)} test')
"
```

You should see: `Loaded: 111 train, 111 dev, 221 test`

### 3. Start vLLM Server

In one terminal:
```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --api-key EMPTY \
  --max-model-len 16384
```

### 4. Set Environment Variables

In another terminal:
```bash
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"
```

## Usage

### Sanity Check

Test the PAPILLON pipeline on a few examples:

```bash
python sanity_pupa.py \
  --data_dir "$WORK/adBO/pupa/data" \
  --split dev \
  --n_examples 3 \
  --dump_jsonl sanity_results.jsonl
```

### Single GEPA Run

Run GEPA optimization:

```bash
python run_gepa_pupa.py \
  --run_dir "$WORK/adBO/runs/pupa_runs/single_gepa" \
  --work_dir "$WORK/adBO/pupa/data" \
  --num_threads 12 \
  --max_metric_calls 1000
```

### Compare GEPA Variants

Run and compare three GEPA variants in parallel:
- GEPA (baseline)
- GEPA+merge
- GEPA with bon=5, itr=5

```bash
python run_gepa_pupa_compare.py \
  --out_root "logs/pupa_comparison" \
  --refresh_sec 20 \
  --stage_step 500 \
  --seed 0 \
  --max_metric_calls 1000 \
  --num_threads 16 \
  --api_bases "http://127.0.0.1:8000/v1"
```

For true parallelism with multiple vLLM servers:
```bash
python run_gepa_pupa_compare.py \
  --out_root "logs/pupa_comparison" \
  --api_bases "http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1" \
  --max_metric_calls 1000
```

## Key Differences from HotpotQA

1. **No Retrieval**: PUPA doesn't use BM25 or document retrieval
2. **3-Stage Pipeline**: Query rewriter → Untrusted model → Response rewriter
3. **Privacy Metrics**: Tracks PII leakage instead of document recall
4. **Aggregate Score**: Balances quality and privacy (not just EM)
5. **Dataset Size**: Much smaller (443 total vs 750 in HotpotQA setup)

## Implementation Details

### PAPILLON Pipeline (pupa_program.py)

```python
class PAPILLONPipeline(dspy.Module):
    def forward(self, user_query: str):
        # Stage 1: Remove PII (Trusted)
        rewritten_query = self.query_rewriter(user_query)

        # Stage 2: Get response (Untrusted)
        untrusted_response = self.untrusted_responder(rewritten_query)

        # Stage 3: Refine response (Trusted)
        final_response = self.response_rewriter(
            user_query, rewritten_query, untrusted_response
        )

        return dspy.Prediction(
            rewritten_query=rewritten_query,
            untrusted_response=untrusted_response,
            final_response=final_response
        )
```

### Metrics (pupa_metric.py)

- **Quality Score**: LLM-as-judge comparing response to reference (fallback to heuristic)
- **Leakage Score**: Fraction of PII entities that appear in rewritten query
- **Aggregate**: `(quality + (1 - leakage)) / 2`

### GEPA Integration

The metric function follows GEPA's 5-argument signature:

```python
def pupa_metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    quality = pupa_quality_score(gold, pred, trace)
    leakage = pupa_leakage_score(gold, pred, trace)
    aggregate = (quality + (1.0 - leakage)) / 2.0

    feedback = f"""PUPA Evaluation:
- Response Quality: {quality:.3f}
- PII Leakage: {leakage:.3f} (lower is better)
- Aggregate Score: {aggregate:.3f}
    """

    return dspy.Prediction(score=aggregate, feedback=feedback)
```

## References

- PUPA Paper: https://aclanthology.org/2025.naacl-long.173.pdf
- PAPILLON GitHub: https://github.com/Columbia-NLP-Lab/PAPILLON/
- GEPA Paper: (citations in main README)

## Troubleshooting

### Dataset Loading Issues

If you see: `FileNotFoundError: PUPA dataset not found`:
1. Download from https://github.com/Columbia-NLP-Lab/PAPILLON/
2. Place `train.json`, `dev.json`, `test.json` in `--work_dir` or `--data_dir`
3. Or update `pupa_data.py` with the correct HuggingFace dataset name

### Quality Score Always 0.5

The LLM-as-judge for quality evaluation requires a working LM. If the LM is not configured or fails, it falls back to a simple heuristic. Check that:
- vLLM server is running
- `VLLM_API_BASE` is set correctly
- The model can be called successfully

### Expected Dataset Sizes

The PUPA paper specifies:
- Train: 111 examples
- Validation: 111 examples
- Test: 221 examples

If your dataset has different sizes, you'll see a warning but the code will proceed.
