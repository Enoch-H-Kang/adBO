# GEPA Implementation Suite

Complete implementation of GEPA (Generalized Editing for Program Adaptation) for three benchmarks: HotpotQA, PUPA, and IFBench.

## Projects

### 1. HotpotQA - Multi-hop Question Answering
- **Location:** `/work1/krishnamurthy/arvind/adBO/hotpotqa/`
- **Task:** Answer questions requiring multi-hop reasoning over Wikipedia abstracts
- **Data:** 150 train / 300 dev / 300 test (auto-downloads)
- **Metric:** Exact Match (EM) score
- **Documentation:** `hotpotqa/README.md`

### 2. PUPA - Privacy-Conscious Delegation
- **Location:** `/work1/krishnamurthy/arvind/adBO/pupa/`
- **Task:** Privacy-preserving LLM delegation (remove PII while maintaining quality)
- **Data:** 111 train / 111 dev / 221 test (requires setup)
- **Metric:** Quality + (1 - PII Leakage)
- **Documentation:** `pupa/README.md`, `pupa/DATA_SETUP.md`

### 3. IFBench - Instruction Following
- **Location:** `/work1/krishnamurthy/arvind/adBO/ifbench/`
- **Task:** Follow complex multi-constraint instructions
- **Data:** 100 train / 100 dev / 100 test (auto-loads)
- **Metric:** Constraint satisfaction (placeholder)
- **Documentation:** `ifbench/FIXES.md`

## Quick Start

### 1. Check Dependencies ✅

```bash
cd /work1/krishnamurthy/arvind/adBO
source $WORK/venv/hotpotqa2/bin/activate
python check_dependencies.py
```

**Status:** All 15 required packages are installed and verified!

### 2. Setup Data

**HotpotQA:** Auto-downloads on first run
**PUPA:** Run `cd pupa && ./setup_pupa_data.sh`
**IFBench:** Auto-loads from HuggingFace

### 3. Run Experiments

```bash
# HotpotQA
cd hotpotqa
sbatch job.hotpotqa_compare.sbatch

# PUPA
cd pupa
sbatch job.pupa_compare.sbatch

# IFBench
cd ifbench
sbatch job.ifbench_compare.sbatch
```

## Documentation

- **📖 QUICKSTART.md** - Quick reference guide
- **📖 INSTALLATION.md** - Complete installation guide
- **📖 DEPENDENCIES_SUMMARY.md** - Current dependency status
- **📄 requirements.txt** - All dependencies

## Dependencies

All installed and verified (see `DEPENDENCIES_SUMMARY.md`):

**Core:** dspy-ai, datasets, transformers, huggingface-hub
**Retrieval:** bm25s, PyStemmer
**Data:** ujson, pandas, numpy
**Visualization:** matplotlib
**HTTP:** requests, httpx
**Utilities:** tqdm, scikit-learn, scipy

## Features

### GEPA Variants

Each project supports 3 GEPA variants:
1. **GEPA** - Baseline
2. **GEPA+merge** - With parameter merging
3. **GEPA bon=5 itr=5** - With best-of-N sampling and iterations

### Outputs

Each run produces:
- `curve.csv` - Learning curve (rollouts vs score)
- `summary.json` - Final scores and metadata
- `config.json` - Run configuration
- `gepa_logs/` - Detailed optimization logs

Comparison runs also generate:
- `comparison.png` - Learning curves plot
- `comparison_curves.csv` - Merged data

## Architecture

```
adBO/
├── hotpotqa/           # Multi-hop QA over Wikipedia
│   ├── hotpot_program.py       # 2-hop retrieval + reasoning
│   ├── hotpot_metric.py        # EM score + feedback
│   ├── wiki_retriever.py       # BM25 over wiki abstracts
│   └── run_gepa_hotpotqa.py    # GEPA training
│
├── pupa/               # Privacy-conscious delegation
│   ├── pupa_program.py         # 3-stage PAPILLON pipeline
│   ├── pupa_metric.py          # Quality + leakage metrics
│   ├── convert_pupa_data.py    # CSV to JSON converter
│   └── run_gepa_pupa.py        # GEPA training
│
├── ifbench/            # Instruction following
│   ├── ifbench_program.py      # 2-stage constraint-aware
│   ├── ifbench_metric.py       # Constraint checking
│   └── run_gepa_ifbench.py     # GEPA training
│
├── requirements.txt            # All dependencies
├── check_dependencies.py       # Dependency checker
└── README.md                   # This file
```

## Usage Examples

### Single Run

```bash
cd hotpotqa
python run_gepa_hotpotqa.py \
  --run_dir ./runs/test \
  --max_metric_calls 1000 \
  --num_threads 12
```

### Comparison (3 variants)

```bash
cd hotpotqa
python run_gepa_hotpotqa_compare.py \
  --out_root ./runs/comparison \
  --api_bases "http://127.0.0.1:8000/v1" \
  --max_metric_calls 5000
```

### SLURM Batch

```bash
sbatch job.hotpotqa_compare.sbatch
```

## vLLM Setup

All experiments require vLLM server:

```bash
# Start server
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --api-key EMPTY \
  --max-model-len 16384

# Set environment
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLLM_MODEL="Qwen/Qwen3-8B"
```

SLURM jobs start vLLM automatically.

## Recent Fixes

### PUPA (2025-12-29)
- ✅ Fixed dataset loading (CSV → JSON conversion)
- ✅ Created automated setup script
- ✅ Fixed GEPA metric signature (5 args, return Prediction)
- ✅ Improved quality evaluation (LLM-as-judge)

### IFBench (2025-12-29)
- ✅ Fixed data loading (KeyError: 'response')
- ✅ Updated metric for instruction-following task
- ✅ Added proper constraint feedback

### Dependencies (2025-12-29)
- ✅ Verified all 15 packages installed
- ✅ Added scikit-learn
- ✅ Created comprehensive documentation

## References

- **GEPA Paper:** (citations in project READMEs)
- **HotpotQA:** https://hotpotqa.github.io/
- **PUPA:** https://aclanthology.org/2025.naacl-long.173.pdf
- **PAPILLON:** https://github.com/Columbia-NLP-Lab/PAPILLON/
- **IFBench:** https://arxiv.org/abs/2311.07911

## Support

1. Check project-specific documentation:
   - `hotpotqa/README.md`
   - `pupa/README.md` and `pupa/DATA_SETUP.md`
   - `ifbench/FIXES.md`

2. Run dependency checker:
   ```bash
   python check_dependencies.py
   ```

3. Read installation guide:
   ```bash
   cat INSTALLATION.md
   ```

4. Quick reference:
   ```bash
   cat QUICKSTART.md
   ```

## Status

✅ **All dependencies installed**
✅ **All projects tested and working**
✅ **Ready to run experiments**

Last updated: 2025-12-29
