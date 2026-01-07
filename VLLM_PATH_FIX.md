# vLLM Installation and PATH Fix for SLURM Jobs

## Problem

SLURM job scripts were failing with:
```
/usr/bin/bash: line 1: vllm: command not found
srun: error: k003-010: task 0: Exited with exit code 127
```

## Root Cause

vLLM was not installed in the venv at `/work1/krishnamurthy/arvind/venv/hotpotqa2/`. It was only installed in the conda environment.

## Solution Applied

### 1. Installed vLLM 0.9.2 in the venv

```bash
# Clone vLLM 0.9.2 source
cd /tmp
git clone --branch v0.9.2 https://github.com/vllm-project/vllm.git vllm-build

# Build and install with ROCm support
source /work1/krishnamurthy/arvind/venv/hotpotqa2/bin/activate
module load rocm/6.4.1
cd /tmp/vllm-build
export VLLM_TARGET_DEVICE=rocm
export MAX_JOBS=4
pip install --no-build-isolation .
```

Result: **vLLM 0.9.2+rocm641** successfully installed

### 2. Fixed transformers compatibility

vLLM 0.9.2 had a conflict with transformers 4.57.3. Downgraded to compatible version:

```bash
pip install "transformers==4.51.3"
```

### 3. Updated all SLURM job scripts

Changed from:
```bash
# OLD (used login shell with conda vllm)
srun -n 1 --exact \
  bash -lc "vllm serve \"$VLLM_MODEL\" --host 127.0.0.1 --port ${PORT0} ..." \
  > "$OUT_ROOT/vllm_${PORT0}.log" 2>&1 &
```

To:
```bash
# NEW (uses venv vllm directly)
srun -n 1 --exact \
  $WORK/venv/hotpotqa2/bin/vllm serve "$VLLM_MODEL" --host 127.0.0.1 --port ${PORT0} ... \
  > "$OUT_ROOT/vllm_${PORT0}.log" 2>&1 &
```

## Key Changes

1. **Installed vLLM in venv**:
   - Version: 0.9.2+rocm641 (non-v1)
   - Built from source with ROCm 6.4.1 support
   - Uses `-O 0` flag instead of `--disable-v1`

2. **Direct vllm binary path**: Use `$WORK/venv/hotpotqa2/bin/vllm`
   - No need for bash wrapper or shell activation
   - Explicit path removes ambiguity
   - Cleaner and more reliable

3. **Removed login shell complexity**: No more `bash -lc`
   - Faster startup
   - More predictable environment
   - Avoids conda/venv conflicts

## Files Fixed

All job scripts across all benchmarks:

### HotpotQA
- ✅ `hotpotqa/job.hotpot_compare.sbatch`
- ✅ `hotpotqa/job.hotpot_compare_resume.sbatch`

### IFBench
- ✅ `ifbench/job.ifbench_compare.sbatch`
- ✅ `ifbench/job.ifbench_compare_resume.sbatch`

### PUPA
- ✅ `pupa/job.pupa_compare.sbatch`
- ✅ `pupa/job.pupa_compare_resume.sbatch`

### HoVer
- ✅ `hover/job.hover_compare.sbatch`
- ✅ `hover/job.hover_compare_resume.sbatch`

## Testing

To verify the fix works:

```bash
# Submit a test job
cd /work1/krishnamurthy/arvind/adBO/hotpotqa
sbatch job.hotpot_compare.sbatch

# Check logs
tail -f /work1/krishnamurthy/arvind/adBO/runs/hotpotqa_runs/latest/vllm_*.log

# Should see vLLM starting instead of "command not found"
```

## Why This Happened

When using `srun` to launch subprocesses:
- Environment variables are inherited
- But shell initialization (`.bashrc`, venv activation) is NOT
- Using `-l` (login) makes it worse - loads system defaults
- Need to explicitly activate venv in each srun command

## Best Practices for SLURM + venv

```bash
# ✅ GOOD: Explicit venv activation in srun
srun bash -c "source /path/to/venv/bin/activate && command"

# ✅ GOOD: Full path to command
srun /path/to/venv/bin/command

# ❌ BAD: Assumes venv is active
srun command

# ❌ BAD: Uses login shell
srun bash -lc "command"
```

## Related Issues

This is a common SLURM pitfall:
- srun creates new processes
- Each process needs its own environment setup
- venv activation in parent doesn't transfer
- Always use explicit paths or source statements
