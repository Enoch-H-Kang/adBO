"""
vLLM server health checking and auto-recovery utilities.

This module provides utilities to ensure vLLM server availability and
handle connection failures gracefully during long-running experiments.
"""
import os
import time
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def check_vllm_health(api_base: str, api_key: str = "EMPTY", timeout: int = 5) -> bool:
    """
    Check if vLLM server is healthy and responding.

    Args:
        api_base: vLLM API base URL (e.g., "http://127.0.0.1:8000/v1")
        api_key: API key for authentication (default: "EMPTY")
        timeout: Request timeout in seconds

    Returns:
        True if server is healthy, False otherwise
    """
    try:
        # Remove /v1 suffix if present for health check
        base_url = api_base.rstrip('/v1').rstrip('/')

        # Try to get models list (lightweight check)
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            f"{base_url}/v1/models",
            headers=headers,
            timeout=timeout
        )

        if response.status_code == 200:
            data = response.json()
            # Check if response has expected structure
            if isinstance(data, dict) and 'data' in data:
                return True

        return False

    except requests.exceptions.RequestException as e:
        logger.debug(f"Health check failed: {e}")
        return False


def wait_for_vllm_server(
    api_base: str,
    api_key: str = "EMPTY",
    max_wait_time: int = 300,
    check_interval: int = 5,
    verbose: bool = True
) -> bool:
    """
    Wait for vLLM server to become available.

    Args:
        api_base: vLLM API base URL
        api_key: API key for authentication (default: "EMPTY")
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
        check_interval: Seconds between health checks
        verbose: Print status messages

    Returns:
        True if server became available, False if timeout
    """
    if verbose:
        print(f"[vLLM] Waiting for server at {api_base} to become available...")

    start_time = time.time()
    attempts = 0

    while (time.time() - start_time) < max_wait_time:
        attempts += 1

        if check_vllm_health(api_base, api_key):
            if verbose:
                elapsed = time.time() - start_time
                print(f"[vLLM] ✓ Server is ready (took {elapsed:.1f}s, {attempts} attempts)")
            return True

        if verbose and attempts % 6 == 0:  # Print every ~30 seconds
            elapsed = time.time() - start_time
            print(f"[vLLM] Still waiting... ({elapsed:.0f}s elapsed, {attempts} attempts)")

        time.sleep(check_interval)

    if verbose:
        print(f"[vLLM] ✗ Server did not become available within {max_wait_time}s")

    return False


def configure_vllm_with_health_check(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
    num_retries: int = 10,
    timeout: int = 300,
    wait_on_startup: bool = True,
    startup_wait_time: int = 300,
    **kwargs
):
    """
    Configure DSPy LM with vLLM health checking.

    Args:
        api_base: vLLM API base URL (from env VLLM_API_BASE if not provided)
        api_key: API key (from env VLLM_API_KEY if not provided)
        model: Model name (from env VLLM_MODEL if not provided)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Max tokens to generate (None = unlimited)
        num_retries: Number of retries on failure
        timeout: Request timeout in seconds
        wait_on_startup: Wait for server to be ready on startup
        startup_wait_time: Max time to wait for server on startup
        **kwargs: Additional arguments for dspy.LM

    Returns:
        Configured dspy.LM instance

    Raises:
        RuntimeError: If server is not available and wait_on_startup is True
    """
    import dspy

    # Get configuration from environment if not provided
    api_base = api_base or os.environ.get("VLLM_API_BASE")
    if not api_base:
        raise RuntimeError(
            "VLLM_API_BASE is not set. Either pass api_base parameter or "
            "set VLLM_API_BASE environment variable."
        )

    api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
    model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")

    # Check server health on startup if requested
    if wait_on_startup:
        if not wait_for_vllm_server(api_base, api_key, max_wait_time=startup_wait_time):
            raise RuntimeError(
                f"vLLM server at {api_base} is not available after waiting {startup_wait_time}s. "
                f"Please ensure the server is running with:\n"
                f"  ./vllm_autorestart.sh\n"
                f"or:\n"
                f"  vllm serve {model} --host 0.0.0.0 --port <PORT> --api-key EMPTY"
            )

    # Create LM instance
    lm_kwargs = {
        "api_base": api_base,
        "api_key": api_key,
        "model_type": "chat",
        "temperature": temperature,
        "top_p": top_p,
        "cache": False,
        "num_retries": num_retries,
        "timeout": timeout,
    }

    # Add max_tokens only if specified
    if max_tokens is not None:
        lm_kwargs["max_tokens"] = max_tokens

    # Add any additional kwargs
    lm_kwargs.update(kwargs)

    lm = dspy.LM(f"openai/{model}", **lm_kwargs)
    dspy.configure(lm=lm)

    return lm


class VLLMHealthCheckWrapper:
    """
    Wrapper around DSPy LM that checks vLLM health before critical operations.

    This can be used to add periodic health checking to existing LM instances.
    """

    def __init__(
        self,
        lm,
        api_base: str,
        check_interval: int = 300,  # Check every 5 minutes
        auto_wait: bool = True,
        max_wait_time: int = 600
    ):
        """
        Initialize health check wrapper.

        Args:
            lm: DSPy LM instance to wrap
            api_base: vLLM API base URL
            check_interval: Seconds between automatic health checks
            auto_wait: Automatically wait for server if it's down
            max_wait_time: Max time to wait for server recovery
        """
        self.lm = lm
        self.api_base = api_base
        self.check_interval = check_interval
        self.auto_wait = auto_wait
        self.max_wait_time = max_wait_time
        self.last_check_time = 0

    def check_and_wait_if_needed(self):
        """Check server health and wait if down (if auto_wait is enabled)."""
        current_time = time.time()

        # Only check periodically to avoid overhead
        if current_time - self.last_check_time < self.check_interval:
            return

        self.last_check_time = current_time

        if not check_vllm_health(self.api_base):
            logger.warning(f"vLLM server appears to be down")

            if self.auto_wait:
                logger.info(f"Waiting for server to recover (max {self.max_wait_time}s)...")
                if wait_for_vllm_server(self.api_base, max_wait_time=self.max_wait_time):
                    logger.info("Server recovered successfully")
                else:
                    logger.error("Server did not recover in time")

    def __call__(self, *args, **kwargs):
        """Forward call to wrapped LM after health check."""
        self.check_and_wait_if_needed()
        return self.lm(*args, **kwargs)

    def __getattr__(self, name):
        """Forward attribute access to wrapped LM."""
        return getattr(self.lm, name)


if __name__ == "__main__":
    # Test health checking
    import sys

    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    print(f"Testing vLLM server at {api_base}")
    print("-" * 50)

    # Test health check
    print("1. Health check:")
    is_healthy = check_vllm_health(api_base, api_key)
    print(f"   Result: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")

    # Test wait (with short timeout for testing)
    if not is_healthy:
        print("\n2. Waiting for server (30s timeout):")
        became_available = wait_for_vllm_server(api_base, api_key, max_wait_time=30)
        if not became_available:
            print("   Server is not available. Please start it with:")
            print("   ./vllm_autorestart.sh")
            sys.exit(1)

    print("\n✓ All checks passed!")
