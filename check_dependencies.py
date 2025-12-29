#!/usr/bin/env python3
"""
Comprehensive dependency checker for GEPA projects.
Checks all required packages and provides installation instructions if needed.
"""
import sys
from typing import Dict, List, Tuple

def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check all required dependencies and return installed/missing lists."""

    required_packages = {
        # Core Framework
        'dspy': ('dspy-ai', 'DSPy framework for LLM programs', True),

        # HuggingFace
        'datasets': ('datasets', 'HuggingFace datasets library', True),
        'transformers': ('transformers', 'HuggingFace transformers', True),
        'huggingface_hub': ('huggingface-hub', 'HuggingFace Hub client', True),

        # Retrieval (HotpotQA)
        'bm25s': ('bm25s', 'BM25 search for document retrieval', True),
        'Stemmer': ('PyStemmer', 'Fast stemming for BM25', True),

        # Data Processing
        'ujson': ('ujson', 'Ultra-fast JSON parsing', True),
        'pandas': ('pandas', 'Data manipulation', False),

        # Visualization
        'matplotlib': ('matplotlib', 'Plotting for comparison scripts', True),
        'numpy': ('numpy', 'Numerical computing', True),

        # HTTP Clients
        'requests': ('requests', 'HTTP client for API calls', True),
        'httpx': ('httpx', 'Async HTTP client', False),

        # Utilities
        'tqdm': ('tqdm', 'Progress bars', False),

        # Scientific Computing
        'sklearn': ('scikit-learn', 'Machine learning utilities', False),
        'scipy': ('scipy', 'Scientific computing', False),
    }

    installed = []
    missing = []

    for import_name, (package_name, description, required) in required_packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            status = '✓'
            req_str = '[REQUIRED]' if required else '[OPTIONAL]'
            installed.append(f"{status} {package_name:25s} v{version:12s} {req_str:12s} - {description}")
        except ImportError:
            status = '✗'
            req_str = '[REQUIRED]' if required else '[OPTIONAL]'
            missing.append((package_name, description, required))
            if required:
                print(f"{status} {package_name:25s} {'NOT INSTALLED':12s} {req_str:12s} - {description}")

    return installed, missing


def print_results(installed: List[str], missing: List[Tuple[str, str, bool]]):
    """Print dependency check results."""

    print("\n" + "=" * 100)
    print("DEPENDENCY CHECK FOR GEPA PROJECTS")
    print("=" * 100)

    print("\nINSTALLED PACKAGES:")
    print("-" * 100)
    for line in installed:
        print(line)

    if missing:
        print("\n" + "=" * 100)
        print("MISSING PACKAGES:")
        print("-" * 100)

        required_missing = [pkg for pkg, _, req in missing if req]
        optional_missing = [pkg for pkg, _, req in missing if not req]

        if required_missing:
            print("\n⚠️  REQUIRED packages (must install):")
            for pkg, desc, _ in missing:
                if pkg in required_missing:
                    print(f"   ✗ {pkg:25s} - {desc}")

            print("\n📦 To install required packages:")
            print(f"   pip install {' '.join(required_missing)}")

        if optional_missing:
            print("\n💡 OPTIONAL packages (recommended but not required):")
            for pkg, desc, _ in missing:
                if pkg in optional_missing:
                    print(f"   ○ {pkg:25s} - {desc}")

            print("\n📦 To install optional packages:")
            print(f"   pip install {' '.join(optional_missing)}")
    else:
        print("\n" + "=" * 100)
        print("✓ ALL REQUIRED PACKAGES ARE INSTALLED!")
        print("=" * 100)

    return len([pkg for pkg, _, req in missing if req]) == 0


def test_imports():
    """Test critical imports for each project."""
    print("\n" + "=" * 100)
    print("TESTING PROJECT-SPECIFIC IMPORTS")
    print("=" * 100)

    tests = {
        'HotpotQA': [
            ('import dspy', 'DSPy framework'),
            ('from datasets import load_dataset', 'Dataset loading'),
            ('import bm25s', 'BM25 retrieval'),
            ('import Stemmer', 'Stemmer for BM25'),
            ('import matplotlib.pyplot', 'Matplotlib plotting'),
        ],
        'PUPA': [
            ('import dspy', 'DSPy framework'),
            ('from datasets import load_dataset', 'Dataset loading'),
            ('import json', 'JSON processing'),
        ],
        'IFBench': [
            ('import dspy', 'DSPy framework'),
            ('from datasets import load_dataset', 'Dataset loading'),
        ],
    }

    all_passed = True
    for project, imports in tests.items():
        print(f"\n{project}:")
        for import_stmt, desc in imports:
            try:
                exec(import_stmt)
                print(f"  ✓ {desc:30s} - {import_stmt}")
            except Exception as e:
                print(f"  ✗ {desc:30s} - {import_stmt}")
                print(f"     Error: {e}")
                all_passed = False

    return all_passed


def main():
    """Main dependency check routine."""
    print("\n🔍 Checking dependencies for GEPA projects...")
    print("   (HotpotQA, PUPA, IFBench)\n")

    # Check dependencies
    installed, missing = check_dependencies()
    deps_ok = print_results(installed, missing)

    # Test imports
    imports_ok = test_imports()

    # Final summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    if deps_ok and imports_ok:
        print("✅ All dependencies are installed and working correctly!")
        print("✅ You're ready to run GEPA experiments!")
        print("\nNext steps:")
        print("  1. Start vLLM server: vllm serve Qwen/Qwen3-8B --port 8000")
        print("  2. Run experiments:")
        print("     - HotpotQA: python run_gepa_hotpotqa.py --run_dir ./runs/test")
        print("     - PUPA:     python run_gepa_pupa.py --run_dir ./runs/test")
        print("     - IFBench:  python run_gepa_ifbench.py --run_dir ./runs/test")
        return 0
    else:
        print("❌ Some dependencies are missing or not working correctly.")
        print("❌ Please install missing packages and try again.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
