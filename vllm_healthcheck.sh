#!/bin/bash
# vLLM Health Check Script
# Monitors vLLM server health and can restart it if unresponsive

# Configuration
VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
CHECK_INTERVAL="${VLLM_CHECK_INTERVAL:-60}"  # Check every 60 seconds
MAX_FAILURES="${VLLM_MAX_FAILURES:-3}"  # Restart after 3 consecutive failures

# Extract host and port from API base
VLLM_HOST=$(echo "$VLLM_API_BASE" | sed -E 's|.*://([^:]+):.*|\1|')
VLLM_PORT=$(echo "$VLLM_API_BASE" | sed -E 's|.*:([0-9]+).*|\1|')

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_health() {
    # Try to get models list (lightweight health check)
    response=$(curl -s -m 10 "${VLLM_API_BASE}/v1/models" 2>&1)

    if [ $? -eq 0 ] && echo "$response" | grep -q "object"; then
        return 0  # Healthy
    else
        return 1  # Unhealthy
    fi
}

restart_vllm() {
    log "Attempting to restart vLLM server..."

    # Find and kill existing vLLM process
    VLLM_PIDS=$(pgrep -f "vllm serve")
    if [ -n "$VLLM_PIDS" ]; then
        log "Killing existing vLLM processes: $VLLM_PIDS"
        kill -15 $VLLM_PIDS  # SIGTERM
        sleep 5

        # Force kill if still running
        VLLM_PIDS=$(pgrep -f "vllm serve")
        if [ -n "$VLLM_PIDS" ]; then
            log "Force killing vLLM processes: $VLLM_PIDS"
            kill -9 $VLLM_PIDS
            sleep 2
        fi
    fi

    log "vLLM server restart triggered. Please ensure vllm_autorestart.sh is running."
    log "Or manually restart with: vllm serve $VLLM_MODEL --host $VLLM_HOST --port $VLLM_PORT ..."
}

main() {
    log "========================================="
    log "vLLM Health Check Monitor Started"
    log "API Base: $VLLM_API_BASE"
    log "Check Interval: ${CHECK_INTERVAL}s"
    log "Max Failures: $MAX_FAILURES"
    log "========================================="

    consecutive_failures=0

    while true; do
        if check_health; then
            if [ $consecutive_failures -gt 0 ]; then
                log "✓ Health check PASSED (recovered after $consecutive_failures failures)"
            else
                log "✓ Health check PASSED"
            fi
            consecutive_failures=0
        else
            consecutive_failures=$((consecutive_failures + 1))
            log "✗ Health check FAILED (failure #$consecutive_failures/$MAX_FAILURES)"

            if [ $consecutive_failures -ge $MAX_FAILURES ]; then
                log "❌ Max consecutive failures reached. Server appears to be down."
                restart_vllm
                consecutive_failures=0
                log "Waiting 30 seconds for server to restart..."
                sleep 30
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
}

# Handle Ctrl+C gracefully
trap 'log "Health check monitor stopped"; exit 0' INT TERM

main
