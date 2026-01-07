# vLLM ROCm Installation Guide

Your system has:
- **GPUs**: AMD MI250X (gfx90a) x2
- **ROCm**: 6.16.6
- **Python**: 3.11.14
- **Current vLLM**: 0.9.2.dev0+rocm (development build - UNSTABLE)

## Problem

Your current vLLM is a development build that crashes with:
```
AssertionError in triton_attn.py: assert attn_metadata.use_cascade is False
```

## Solution: Install Stable vLLM for ROCm

### Option 1: Official vLLM v0.6.3.post1 (ROCm) - RECOMMENDED

This is the last stable release with good ROCm support:

```bash
# Activate your environment
source /work1/krishnamurthy/arvind/venv/hotpotqa2/bin/activate

# Uninstall current vLLM
pip uninstall -y vllm

# Install stable vLLM for ROCm from source
pip install vllm==0.6.3.post1 --no-build-isolation
```

**If the above fails with ROCm compatibility issues, build from source:**

```bash
# Clone vLLM
cd /work1/krishnamurthy/arvind
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3.post1

# Build with ROCm support
export PYTORCH_ROCM_ARCH="gfx90a"  # Your GPU architecture
export VLLM_TARGET_DEVICE=rocm

pip install -e . --no-build-isolation
```

### Option 2: Latest Stable v0.8.x (ROCm)

If v0.6.3.post1 has issues, try v0.8.x:

```bash
cd /work1/krishnamurthy/arvind/vllm
git checkout v0.8.4  # Latest 0.8.x stable

export PYTORCH_ROCM_ARCH="gfx90a"
export VLLM_TARGET_DEVICE=rocm
pip install -e . --no-build-isolation
```

### Option 3: Stay on v0.9.x but Use Stable Tag

If you need v0.9.x features, use a stable tag instead of dev:

```bash
cd /work1/krishnamurthy/arvind/vllm
git fetch --all --tags
git checkout v0.9.1  # Latest stable 0.9.x (if exists)

export PYTORCH_ROCM_ARCH="gfx90a"
export VLLM_TARGET_DEVICE=rocm
pip install -e . --no-build-isolation
```

## Verification

After installation, verify:

```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Test server startup
vllm serve Qwen/Qwen3-8B --host 127.0.0.1 --port 8888 --max-model-len 8192 &
sleep 30
curl http://127.0.0.1:8888/v1/models
pkill -f "vllm serve"
```

## Post-Installation: Update Auto-restart Script

Update `vllm_autorestart.sh` to use V0 engine (more stable):

```bash
# Add these flags to the vllm serve command:
vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key EMPTY \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --disable-frontend-multiprocessing \
    --disable-log-requests \
    2>&1 | tee -a "$LOG_FILE"
```

## ROCm-Specific Optimizations

For better stability on AMD GPUs:

```bash
# Set environment variables before starting vLLM
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO
export PYTORCH_ROCM_ARCH=gfx90a

# Start with conservative settings
vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --disable-frontend-multiprocessing \
    --kv-cache-dtype fp16
```

## Troubleshooting

### Build fails with "no such file or directory"
```bash
pip install ninja cmake wheel packaging setuptools-scm
```

### CUDA references in ROCm build
This is normal - PyTorch uses CUDA APIs that ROCm translates

### OOM during build
```bash
# Reduce parallel jobs
export MAX_JOBS=4
pip install -e . --no-build-isolation
```

### Import error: undefined symbol
```bash
# Rebuild from scratch
pip uninstall -y vllm
rm -rf build/ dist/ *.egg-info
pip install -e . --no-build-isolation
```

## Quick Start Commands

**Fast install (if wheels exist):**
```bash
pip uninstall -y vllm
pip install vllm==0.6.3.post1
```

**From source (more reliable for ROCm):**
```bash
cd /work1/krishnamurthy/arvind
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3.post1
export PYTORCH_ROCM_ARCH=gfx90a
export VLLM_TARGET_DEVICE=rocm
pip install -e . --no-build-isolation
```

## Expected Results

After successful installation:
- ✅ vLLM version shows stable tag (e.g., `0.6.3.post1`)
- ✅ No more `AssertionError: use_cascade is False`
- ✅ Server stays up for long runs (hours/days)
- ✅ Auto-restart script handles graceful shutdowns

## Support

If issues persist:
1. Check vLLM GitHub issues: https://github.com/vllm-project/vllm/issues
2. ROCm-specific issues: https://github.com/ROCm/ROCm/issues
3. Test with smaller model first: `facebook/opt-125m`
