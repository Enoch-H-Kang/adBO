# vLLM Auto-Restart Solution

This directory contains scripts to automatically restart the vLLM server if it crashes or becomes unresponsive.

## Problem
The vLLM server can crash or hang during long-running GEPA experiments (especially PUPA which can run for hours), causing:
- `Connection refused` errors
- Lost progress in optimization runs
- Manual intervention required

## Solution Options

### Option 1: Auto-Restart Wrapper (Recommended)

Use `vllm_autorestart.sh` to wrap the vLLM server process. It will automatically restart the server if it crashes.

**Usage:**
```bash
# Basic usage
./vllm_autorestart.sh

# With custom configuration
export VLLM_MODEL="Qwen/Qwen3-8B"
export VLLM_PORT=8000
export VLLM_MAX_MODEL_LEN=16384
export VLLM_GPU_MEM=0.9
./vllm_autorestart.sh

# Run in background with nohup
nohup ./vllm_autorestart.sh > vllm_wrapper.log 2>&1 &

# Run in a tmux/screen session (recommended)
tmux new -s vllm
./vllm_autorestart.sh
# Detach with Ctrl+B, D
```

**Features:**
- Automatically restarts vLLM on crash
- Logs all activity with timestamps
- Prevents crash loops (pauses if crashing too frequently)
- Configurable via environment variables

**Configuration Variables:**
- `VLLM_MODEL`: Model to serve (default: Qwen/Qwen3-8B)
- `VLLM_HOST`: Host to bind (default: 0.0.0.0)
- `VLLM_PORT`: Port to use (default: 8000)
- `VLLM_MAX_MODEL_LEN`: Max context length (default: 16384)
- `VLLM_GPU_MEM`: GPU memory utilization (default: 0.9)

### Option 2: Health Check Monitor

Use `vllm_healthcheck.sh` alongside a manually started vLLM server to monitor and restart it if unresponsive.

**Usage:**
```bash
# Terminal 1: Start vLLM normally
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --api-key EMPTY --max-model-len 16384

# Terminal 2: Start health check monitor
export VLLM_API_BASE="http://127.0.0.1:8000"
./vllm_healthcheck.sh

# Or in background
nohup ./vllm_healthcheck.sh > healthcheck.log 2>&1 &
```

**Features:**
- Monitors vLLM health every 60 seconds (configurable)
- Restarts server after 3 consecutive failed health checks
- Lightweight HTTP-based health checking

**Configuration Variables:**
- `VLLM_API_BASE`: vLLM server URL (default: http://127.0.0.1:8000)
- `VLLM_CHECK_INTERVAL`: Seconds between checks (default: 60)
- `VLLM_MAX_FAILURES`: Failures before restart (default: 3)

### Option 3: systemd Service (Production)

For production environments, create a systemd service:

**File: `/etc/systemd/system/vllm.service`**
```ini
[Unit]
Description=vLLM Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/work1/krishnamurthy/arvind/adBO
Environment="VLLM_MODEL=Qwen/Qwen3-8B"
ExecStart=/path/to/venv/bin/vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --api-key EMPTY --max-model-len 16384
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
```

## Recommended Setup for GEPA Experiments

### For SLURM/Batch Jobs:

**In your SLURM job script:**
```bash
#!/bin/bash
#SBATCH --job-name=pupa_gepa
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# Start vLLM with auto-restart in background
export VLLM_MODEL="Qwen/Qwen3-8B"
export VLLM_PORT=8000
./vllm_autorestart.sh > vllm_server.log 2>&1 &
VLLM_PID=$!

# Wait for server to be ready
sleep 30

# Run your GEPA experiment
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
python pupa/run_gepa_pupa.py --run_dir runs/pupa --max_metric_calls 1000

# Cleanup
kill $VLLM_PID
```

### For Interactive Sessions:

**Terminal 1 (vLLM server):**
```bash
cd /work1/krishnamurthy/arvind/adBO
tmux new -s vllm
./vllm_autorestart.sh
# Detach: Ctrl+B, D
```

**Terminal 2 (GEPA experiment):**
```bash
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
python pupa/run_gepa_pupa.py --run_dir runs/pupa --max_metric_calls 1000
```

## Troubleshooting

### vLLM keeps crashing in a loop
- Check GPU memory: `nvidia-smi`
- Reduce `VLLM_GPU_MEM` (e.g., 0.8 instead of 0.9)
- Reduce `VLLM_MAX_MODEL_LEN` if you're hitting OOM

### Health check keeps failing but server seems fine
- Increase `VLLM_CHECK_INTERVAL` to reduce check frequency
- Increase `VLLM_MAX_FAILURES` to be more tolerant
- Check firewall/network if using remote connections

### Server restarts mid-experiment
- Check logs in `$WORK/vllm_logs/`
- Look for OOM errors, CUDA errors, or other crash reasons
- Consider using a smaller model or reducing batch sizes

## Logs

- **Auto-restart logs**: `$WORK/vllm_logs/vllm_autorestart_TIMESTAMP.log`
- **vLLM server logs**: Included in the auto-restart log
- **Health check logs**: Stdout (redirect as needed)

## Notes

- The `num_retries=10` in DSPy LM config will retry API calls, but won't help if the server is completely down
- Auto-restart is better than retries because it actively revives the server
- For critical experiments, consider combining auto-restart + health monitoring
- Always monitor GPU memory usage during long runs
