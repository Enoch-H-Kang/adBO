#!/bin/bash
#SBATCH -J hotpot_gepa_cmp
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -p mi2104x
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24

set -euo pipefail

module purge
module load hpcfund
module list

# IMPORTANT: prevent python path contamination (rocm modules inject PYTHONPATH)
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
hash -r

source $WORK/venv/hotpotqa2/bin/activate

# Caches in $WORK
export HF_HOME="$WORK/adBO/cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$WORK/adBO/cache/xdg"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="$WORK/adBO/runs/$RUN_ID"
mkdir -p "$OUT_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

export VLLM_MODEL="Qwen/Qwen3-8B"
export VLLM_API_KEY="EMPTY"

echo "==== rocm-smi (sanity) ===="
rocm-smi || true
echo "==========================="

# ----------------------------
# Pick collision-free ports (avoid 8000)
# ----------------------------
BASE_PORT=$(( 18000 + (SLURM_JOB_ID % 1000) * 10 ))
PORT0=$((BASE_PORT + 0))
PORT1=$((BASE_PORT + 1))
PORT2=$((BASE_PORT + 2))

API0="http://127.0.0.1:${PORT0}/v1"
API1="http://127.0.0.1:${PORT1}/v1"
API2="http://127.0.0.1:${PORT2}/v1"

echo "Using ports: $PORT0 $PORT1 $PORT2" | tee "$OUT_ROOT/ports.txt"
echo "$API0,$API1,$API2" > "$OUT_ROOT/api_bases.txt"

# ----------------------------
# Start 3 vLLM servers on 3 GPUs (reliable daemon style)
# Use python -m instead of `vllm serve` to avoid entrypoint metadata issues
# ----------------------------
export HIP_VISIBLE_DEVICES=0
nohup bash -lc "
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
  source $WORK/venv/hotpotqa2/bin/activate
  python -m vllm.entrypoints.openai.api_server \
    --model \"$VLLM_MODEL\" \
    --host 127.0.0.1 --port ${PORT0} \
    --api-key \"$VLLM_API_KEY\" \
    --max-model-len 16384
" > "$OUT_ROOT/vllm_${PORT0}.log" 2>&1 &

export HIP_VISIBLE_DEVICES=1
nohup bash -lc "
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
  source $WORK/venv/hotpotqa2/bin/activate
  python -m vllm.entrypoints.openai.api_server \
    --model \"$VLLM_MODEL\" \
    --host 127.0.0.1 --port ${PORT1} \
    --api-key \"$VLLM_API_KEY\" \
    --max-model-len 16384
" > "$OUT_ROOT/vllm_${PORT1}.log" 2>&1 &

export HIP_VISIBLE_DEVICES=2
nohup bash -lc "
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
  source $WORK/venv/hotpotqa2/bin/activate
  python -m vllm.entrypoints.openai.api_server \
    --model \"$VLLM_MODEL\" \
    --host 127.0.0.1 --port ${PORT2} \
    --api-key \"$VLLM_API_KEY\" \
    --max-model-len 16384
" > "$OUT_ROOT/vllm_${PORT2}.log" 2>&1 &

# ----------------------------
# Wait for servers
# ----------------------------
echo "Waiting for vLLM servers..."
for port in "$PORT0" "$PORT1" "$PORT2"; do
  echo "  waiting on $port ..."
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "  vLLM on ${port} is up."
      break
    fi
    sleep 2
  done
done

# quick sanity dump
echo "=== vLLM /v1/models sanity ==="
curl -sS "http://127.0.0.1:${PORT0}/v1/models" | head
curl -sS "http://127.0.0.1:${PORT1}/v1/models" | head
curl -sS "http://127.0.0.1:${PORT2}/v1/models" | head
echo "=============================="

# ----------------------------
# Run compare driver with correct api_bases
# IMPORTANT: call the correct script name
# ----------------------------
cd $WORK/adBO/repo

python run_gepa_hotpotqa_compare.py \
  --out_root "$OUT_ROOT/logs" \
  --api_bases "$API0,$API1,$API2" \
  --refresh_sec 20 \
  --stage_step 500 \
  --seed 0 \
  --max_metric_calls 10000 \
  --num_threads 12 \
  --retriever_threads 8

echo "Done. Outputs in: $OUT_ROOT/logs"
