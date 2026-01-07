# PUPA: Original PAPILLON Implementation

This benchmark implements the **original PAPILLON pipeline** from the Columbia NLP Lab's GitHub repository, adapted for GEPA optimization.

## Key Difference from pupa/

| Aspect | pupa/ | pupa/ |
|--------|-------|--------|
| Source | Custom adaptation | **Original PAPILLON code** |
| Pipeline | 3-stage custom | **2-stage from paper** (CreateOnePrompt → InfoAggregator) |
| Metrics | Simplified | **Full PAPILLON suite** (from llm_judge.py) |
| Attribution | Inspired by | **Direct copy with MIT LICENSE** |
| Code Origin | Adapted from HotpotQA pattern | Columbia-NLP-Lab/PAPILLON |

## Architecture

### Original PAPILLON Pipeline

```
User Query (with PII)
    ↓
[Stage 1: CreateOnePrompt]  ← Trusted Model
    → Rewritten Query (PII removed)
    ↓
[Stage 2: Untrusted Model Call]
    → Model Response
    ↓
[Stage 3: InfoAggregator]  ← Trusted Model
    → Final Response
```

### Metrics (from llm_judge.py)

**Quality Score** (Binary: 0/1)
- Pairwise comparison: new_response vs reference
- Position bias handling: evaluates both permutations
- Score = 1 if both agree OR first says "yes"

**Leakage Score** (Count: 0 to num_pii)
- LLM judge checks each PII piece in rewritten query
- Special handling for celebrity names (excluded)
- Returns count of leaked entities

**Prompt Quality Score** (Binary: 0/1)
- LLM judge validates if rewritten query is well-formed
- Ensures the query can be used with untrusted model

**Aggregate Score** (Float: -0.5 to 1.0)
```
aggregate = (quality - leakage/num_pii + prompt_quality) / 2
```

## Files

### Core Implementation
- `papillon/` - **Original PAPILLON code** from GitHub
  - `papillon_signatures.py` - CreateOnePrompt, InfoAggregator
  - `papillon_pipeline.py` - PAPILLON class
  - `llm_judge.py` - Quality, Leakage, Prompt Quality judges
  - `LICENSE_PAPILLON.txt` - MIT license from original
  - `README_PAPILLON.md` - Attribution and changes

### GEPA Integration
- `pupa_data.py` - Data loader (reuses pupa/data/)
- `pupa_metric.py` - GEPA-compatible metric wrapper
- `pupa_program.py` - Program factory
- `run_gepa_pupa.py` - Main GEPA runner
- `run_gepa_pupa_compare.py` - Comparison driver
- `sanity_pupa.py` - Validation script

### Job Submission
- `job.pupa_compare.sbatch` - Fresh run (BASE_PORT=22000)
- `job.pupa_compare_resume.sbatch` - Resume from latest

## Setup

### 1. Data Preparation

**Option A: Reuse pupa data** (Recommended)
```bash
# Check if pupa data exists
ls ../pupa/data/

# If it exists, you're done! pupa will automatically use it.
```

**Option B: Fresh data setup**
```bash
cd /work1/krishnamurthy/arvind/adBO/pupa
mkdir -p data
ln -s ../pupa/data/* data/
```

### 2. Verify Setup

```bash
cd /work1/krishnamurthy/arvind/adBO/pupa

# Test data loading
python3 -c "
from pupa_data import load_pupa_splits
train, dev, test = load_pupa_splits(seed=0)
print(f'✓ Loaded: {len(train)} train, {len(dev)} dev, {len(test)} test')
"

# Expected output:
# [PUPA] Loading data from: /work1/krishnamurthy/arvind/adBO/pupa/data
# [PUPA] Loaded 111 train, 111 dev, 221 test
# [PUPA] Using original PAPILLON pipeline for evaluation
# ✓ Loaded: 111 train, 111 dev, 221 test
```

## Usage

### Quick Sanity Check

```bash
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"

python sanity_pupa.py \
  --split dev \
  --n_examples 3 \
  --dump_jsonl sanity_pupa_results.jsonl
```

### Single GEPA Run

```bash
python run_gepa_pupa.py \
  --run_dir "$WORK/adBO/runs/pupa_runs/test" \
  --work_dir "$WORK/adBO/pupa/data" \
  --num_threads 12 \
  --max_metric_calls 1000
```

### Submit SLURM Job

**Fresh run:**
```bash
sbatch job.pupa_compare.sbatch
```

**Resume from latest:**
```bash
# Normal resume
sbatch job.pupa_compare_resume.sbatch

# Clean resume (start optimization from scratch, keep data)
CLEAN=1 sbatch job.pupa_compare_resume.sbatch
```

### Monitor Progress

```bash
# Check latest run
ls -lh $WORK/adBO/runs/pupa_runs/latest/logs/

# View live plot
display $WORK/adBO/runs/pupa_runs/latest/logs/comparison_live.png

# Tail logs
tail -f $WORK/adBO/runs/pupa_runs/latest/logs/gepa/stdout.log
```

## Code Attribution

The `papillon/` directory contains code copied from:
- **Repository**: https://github.com/Columbia-NLP-Lab/PAPILLON
- **License**: MIT (see papillon/LICENSE_PAPILLON.txt)
- **Files copied**:
  - `run_llama_dspy.py` → `papillon_signatures.py` + `papillon_pipeline.py`
  - `llm_judge.py` → `papillon/llm_judge.py`

### Changes Made

1. **Modularization**: Split monolithic files into logical modules
2. **GEPA Integration**: Added 5-argument metric wrapper
3. **Error Handling**: Added fallbacks for robustness
4. **DSPy 2.x**: Updated API calls for compatibility

**No changes to core algorithms** - all PAPILLON logic preserved exactly.

## Citation

If you use this code, please cite both PAPILLON and GEPA:

```bibtex
@inproceedings{papillon2025,
  title={PAPILLON: PrivAcy Preservation in Large Language models by Integrating Locally-trained OptiONs},
  author={Columbia NLP Lab},
  booktitle={NAACL},
  year={2025}
}

@article{gepa2024,
  title={GEPA: Generative Efficient Prompt Augmentation},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'papillon'
```
**Solution**: Run from pupa/ directory or add to PYTHONPATH:
```bash
export PYTHONPATH="/work1/krishnamurthy/arvind/adBO/pupa:$PYTHONPATH"
```

### Data Not Found
```
FileNotFoundError: PUPA dataset not found
```
**Solution**: Set up data symlink:
```bash
cd /work1/krishnamurthy/arvind/adBO/pupa
mkdir -p data
ln -s ../pupa/data/* data/
```

### Port Conflicts
```
Address already in use: 22000
```
**Solution**: SLURM job adds random offset. If manual testing:
```bash
export VLLM_API_BASE="http://127.0.0.1:22500/v1"  # Use different port
```

### Different Scores from pupa/
This is **expected**! pupa uses the original PAPILLON metrics which:
- Handle position bias differently in quality evaluation
- Use more sophisticated leakage detection
- Include prompt quality as third component

The scores should differ, showing the impact of the original methodology.

## References

- **PAPILLON Repo**: https://github.com/Columbia-NLP-Lab/PAPILLON
- **PUPA Paper**: https://aclanthology.org/2025.naacl-long.173.pdf
- **GEPA**: (see main repo README)
