#!/bin/bash
# vLLM Auto-Restart Wrapper
# This script automatically restarts the vLLM server if it crashes or exits

# Configuration
MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEM:-0.9}"

# Log file
LOG_DIR="${WORK}/vllm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_autorestart_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "========================================="
log "vLLM Auto-Restart Wrapper Started"
log "Model: $MODEL"
log "Host: $HOST, Port: $PORT"
log "Max Model Length: $MAX_MODEL_LEN"
log "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
log "Log file: $LOG_FILE"
log "========================================="

# Counter for restarts
RESTART_COUNT=0
MAX_RAPID_RESTARTS=5
RAPID_RESTART_WINDOW=60  # seconds

# Track restart times
declare -a RESTART_TIMES=()

while true; do
    log "Starting vLLM server (attempt #$((RESTART_COUNT + 1)))"

    # Start vLLM server
    vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --api-key EMPTY \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?
    CURRENT_TIME=$(date +%s)

    log "vLLM server exited with code: $EXIT_CODE"

    # Track restart time
    RESTART_TIMES+=("$CURRENT_TIME")
    RESTART_COUNT=$((RESTART_COUNT + 1))

    # Clean old restart times (older than RAPID_RESTART_WINDOW)
    CUTOFF_TIME=$((CURRENT_TIME - RAPID_RESTART_WINDOW))
    RECENT_RESTARTS=()
    for t in "${RESTART_TIMES[@]}"; do
        if [ "$t" -gt "$CUTOFF_TIME" ]; then
            RECENT_RESTARTS+=("$t")
        fi
    done
    RESTART_TIMES=("${RECENT_RESTARTS[@]}")

    # Check for rapid restart loop (crash loop)
    if [ "${#RESTART_TIMES[@]}" -ge "$MAX_RAPID_RESTARTS" ]; then
        log "ERROR: vLLM server crashed $MAX_RAPID_RESTARTS times within $RAPID_RESTART_WINDOW seconds"
        log "This indicates a persistent issue. Please check:"
        log "  1. GPU memory (OOM errors?)"
        log "  2. Model compatibility"
        log "  3. vLLM logs above for error details"
        log "Waiting 5 minutes before retrying..."
        sleep 300
        RESTART_TIMES=()  # Reset counter after long wait
    else
        log "Waiting 10 seconds before restart..."
        sleep 10
    fi
done
