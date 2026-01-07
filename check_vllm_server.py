#!/usr/bin/env python3
"""
Diagnostic script to check vLLM server status and endpoints.
"""

import os
import requests
import json

def check_server_health():
    """Check if vLLM server is running and responsive"""
    api_base = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")

    # Remove /v1 suffix to get base URL
    base_url = api_base.replace("/v1", "")

    print("=" * 80)
    print("vLLM Server Diagnostics")
    print("=" * 80)
    print()
    print(f"Configured API Base: {api_base}")
    print(f"Server Base URL: {base_url}")
    print()

    # Test 1: Check if server is reachable at all
    print("Test 1: Server Connectivity")
    print("-" * 80)

    endpoints_to_try = [
        base_url,
        f"{base_url}/health",
        f"{base_url}/v1/models",
        api_base + "/models",
        f"{base_url}/models",
    ]

    for endpoint in endpoints_to_try:
        try:
            print(f"Trying: {endpoint} ... ", end="")
            response = requests.get(endpoint, timeout=5)
            print(f"✓ {response.status_code}")

            if response.status_code == 200:
                print(f"  Response preview: {response.text[:200]}")

                try:
                    data = response.json()
                    print(f"  JSON data: {json.dumps(data, indent=2)[:500]}")
                except:
                    pass

        except requests.exceptions.ConnectionError:
            print("✗ Connection refused - server not running?")
        except requests.exceptions.Timeout:
            print("✗ Timeout - server not responding")
        except Exception as e:
            print(f"✗ Error: {e}")

    print()

    # Test 2: Try to make a simple API call
    print("Test 2: API Call Test")
    print("-" * 80)

    api_endpoints = [
        api_base + "/chat/completions",
        api_base + "/completions",
    ]

    for endpoint in api_endpoints:
        try:
            print(f"Trying: {endpoint} ... ", end="")

            payload = {
                "model": os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B"),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}",
            }

            response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            print(f"{response.status_code}")

            if response.status_code == 200:
                print("  ✓ API call successful!")
                try:
                    data = response.json()
                    print(f"  Response: {json.dumps(data, indent=2)[:500]}")
                except:
                    print(f"  Raw: {response.text[:200]}")
            else:
                print(f"  Response: {response.text[:200]}")

        except requests.exceptions.ConnectionError:
            print("✗ Connection refused")
        except requests.exceptions.Timeout:
            print("✗ Timeout")
        except Exception as e:
            print(f"✗ Error: {e}")

    print()

    # Test 3: Check environment
    print("Test 3: Environment Variables")
    print("-" * 80)

    env_vars = ["VLLM_API_BASE", "VLLM_API_KEY", "VLLM_MODEL", "WORK"]

    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        status = "✓" if value != "NOT SET" else "✗"
        print(f"{status} {var:20} = {value}")

    print()

    # Test 4: Network check
    print("Test 4: Network Ports")
    print("-" * 80)

    import socket

    ports_to_check = [8000, 8001, 8080]

    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()

        if result == 0:
            print(f"✓ Port {port} is OPEN")
        else:
            print(f"✗ Port {port} is CLOSED")

    print()
    print("=" * 80)
    print("Diagnostics Complete")
    print("=" * 80)
    print()
    print("RECOMMENDATIONS:")
    print("1. Make sure vLLM server is running with:")
    print("   vllm serve <model> --host 0.0.0.0 --port 8000 --api-key EMPTY")
    print()
    print("2. Set environment variables:")
    print("   export VLLM_API_BASE='http://127.0.0.1:8000/v1'")
    print("   export VLLM_API_KEY='EMPTY'")
    print("   export VLLM_MODEL='<your-model-name>'")
    print()

if __name__ == "__main__":
    check_server_health()
