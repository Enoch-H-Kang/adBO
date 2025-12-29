# IFBench Adaptation Summary

This document summarizes the changes made to adapt the IFBench implementation to match the HotpotQA structure and fix critical bugs.

## Critical Bugs Fixed

### 1. Data Loading Error ❌ → ✅
**Problem**: `ifbench_data.py` tried to load 744 examples but dataset only has 300
```python
# BEFORE (caused IndexError)
train_split = dataset['train'].select(range(150))      # 0-150
dev_split = dataset['train'].select(range(150, 450))   # 150-450 ❌ IndexError!
test_split = dataset['train'].select(range(450, 744))  # 450-744 ❌ IndexError!

# AFTER (correct split)
train_split = dataset['train'].select(range(100))      # 0-100
dev_split = dataset['train'].select(range(100, 200))   # 100-200
test_split = dataset['train'].select(range(200, 300))  # 200-300
```

### 2. Variable Name Typo ❌ → ✅
**Problem**: `run_gepa_ifbench.py` line 242 had `gepa__log_dir` (double underscore)
```python
# BEFORE
print(f"[GEPA] compiling up to max_metric_calls={b} (resume log_dir={gepa__log_dir})")  # ❌

# AFTER
print(f"[GEPA] compiling up to max_metric_calls={b} (resume log_dir={gepa_log_dir})")   # ✅
```

### 3. Relative Path Issue ❌ → ✅
**Problem**: `job.ifbench_compare.sbatch` used relative path causing vLLM startup failures
```bash
# BEFORE
OUT_ROOT="runs/$RUN_ID"  # ❌ Relative path breaks after cd

# AFTER
OUT_ROOT="$WORK/adBO/runs/ifbench_runs/$RUN_ID"  # ✅ Absolute path
```

### 4. Wrong Variant Concept ❌ → ✅
**Problem**: `run_gepa_ifbench_compare.py` compared program architectures instead of GEPA variants
```python
# BEFORE (comparing different programs)
variants = [
    {"name": "Two Stage", "subdir": "two_stage", "program_variant": "two_stage"},
    {"name": "Single Stage", "subdir": "single_stage", "program_variant": "single_stage"},
    {"name": "Iterative", "subdir": "iterative", "program_variant": "iterative"},
]

# AFTER (comparing GEPA optimization strategies)
variants = [
    {"name": "GEPA", "subdir": "gepa", "use_merge": 0, "bon": 1, "itr": 1},
    {"name": "GEPA+merge", "subdir": "gepa_merge", "use_merge": 1, "bon": 1, "itr": 1},
    {"name": "GEPA bon=5 itr=5", "subdir": "gepa_bon5_itr5", "use_merge": 0, "bon": 5, "itr": 5},
]
```
**Why this matters**: GEPA is about **prompt optimization**, not program architecture comparison.
We use ONE program (two-stage) and optimize its prompts with different GEPA strategies.

## Files Modified

### 1. ifbench_data.py
**Changes:**
- Fixed dataset split sizes: 100/100/100 instead of 150/300/294
- Added comment explaining the split

### 2. run_gepa_ifbench.py
**Changes:**
- Fixed typo: `gepa__log_dir` → `gepa_log_dir`

### 3. run_gepa_ifbench_compare.py
**Changes:**
- Added header comment with example usage (matches HotpotQA style)
- **FIXED**: Changed from program variants to GEPA variants
  - Before: Two Stage, Single Stage, Iterative (program architectures)
  - After: GEPA, GEPA+merge, GEPA bon=5 itr=5 (prompt optimization variants)
- Now matches HotpotQA/HOVER structure exactly

### 4. job.ifbench_compare.sbatch
**Changes:**
- Fixed OUT_ROOT to use absolute path: `$WORK/adBO/runs/ifbench_runs/$RUN_ID`
- Already matches HotpotQA structure otherwise

### 5. job.ifbench_compare_resume.sbatch
**No changes needed** - Already correct with:
- Fixed "latest" folder for resume capability
- Append mode for vLLM logs (`>>`)
- CLEAN=1 support

### 6. resume_submit.sh
**No changes needed** - Already correct with:
- Proper paths to `ifbench_runs/latest`
- Helpful monitoring commands

## Key Differences from HotpotQA (Expected)

1. **Task Type:**
   - HotpotQA/HOVER: Multi-hop question answering with retrieval
   - IFBench: Instruction following with constraints (no retrieval)

2. **Dataset:**
   - HotpotQA: 150/300/300 split
   - HOVER: 150/300/300 split
   - IFBench: 100/100/100 split (smaller dataset - only 300 examples total)

3. **Metrics:**
   - HotpotQA: EM (Exact Match)
   - HOVER: Recall score
   - IFBench: Custom score from `ifbench_score` (constraint satisfaction)

4. **Budget:**
   - HotpotQA: 10,000 metric calls
   - HOVER: 6,858 metric calls
   - IFBench: 1,000 metric calls (default in worker)

5. **Program:**
   - HotpotQA: HotpotMultiHopQA with BM25 retriever
   - HOVER: HoverMultiHop with BM25 retriever
   - IFBench: IFBenchTwoStage (2-stage: answer → rewrite with constraints)

6. **GEPA Variants (SAME!):**
   - All three benchmarks now use: GEPA, GEPA+merge, GEPA bon=5 itr=5
   - GEPA optimizes prompts for better instruction following in IFBench

## Testing

### One-time run (with timestamp):
```bash
cd /work1/krishnamurthy/arvind/adBO/ifbench
sbatch job.ifbench_compare.sbatch
```

This will:
1. Start 3 vLLM servers on ports 18000+, 18001+, 18002+
2. Run 3 GEPA variants in parallel (all using two-stage IFBench program):
   - GEPA
   - GEPA+merge
   - GEPA bon=5 itr=5
3. Generate live comparison plots and CSV
4. Save all outputs to `$WORK/adBO/runs/ifbench_runs/<RUN_ID>/logs/`

### Resume-friendly run (fixed "latest" folder):
```bash
cd /work1/krishnamurthy/arvind/adBO/ifbench
./resume_submit.sh
```

This will:
- Submit a resume-friendly job that uses a fixed output path
- Allow you to resubmit to continue from where you left off
- GEPA will automatically resume from its checkpoints
- Useful for running in stages or recovering from interruptions

To clean and restart:
```bash
cd /work1/krishnamurthy/arvind/adBO/ifbench
CLEAN=1 sbatch job.ifbench_compare_resume.sbatch
```

## File Structure

```
adBO/ifbench/
├── run_gepa_ifbench_compare.py        # Comparison driver (spawns 3 workers)
├── run_gepa_ifbench.py                # Single-run worker
├── job.ifbench_compare.sbatch         # SLURM batch script (timestamped runs)
├── job.ifbench_compare_resume.sbatch  # SLURM batch script (resume-friendly)
├── resume_submit.sh                   # Helper script for resume submissions
├── ifbench_data.py                    # Data loading (FIXED)
├── ifbench_metric.py                  # Metrics and feedback
├── ifbench_program.py                 # DSPy program variants
└── ...
```

## What Was Broken Before

The previous attempts failed because:
1. **Server started** ✅ - vLLM servers were starting correctly
2. **Workers failed** ❌ - Data loading threw IndexError
3. **Silent failure** - Error logs were in the run directory, not SLURM output

The error message was:
```
IndexError: Index 449 out of range for dataset of size 300.
```

This is now fixed with the corrected data splits!
