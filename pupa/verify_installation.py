#!/usr/bin/env python3
"""
Quick verification script for pupa implementation.
Run this to verify everything is working correctly.
"""
import sys
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    try:
        # PAPILLON module
        from papillon import PAPILLON, LLMJudge, CreateOnePrompt, InfoAggregator
        # Data
        from pupa_data import load_pupa_splits
        # Metrics
        from pupa_metric import pupa_score, pupa_metric_with_feedback
        # Program
        from pupa_program import create_pupa_program
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    try:
        from pupa_data import load_pupa_splits
        train, dev, test = load_pupa_splits(seed=0)
        assert len(train) == 111, f"Expected 111 train, got {len(train)}"
        assert len(dev) == 111, f"Expected 111 dev, got {len(dev)}"
        assert len(test) == 221, f"Expected 221 test, got {len(test)}"
        print(f"  ✓ Data loaded: {len(train)}/{len(dev)}/{len(test)} (train/dev/test)")
        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False

def test_program_creation():
    """Test program creation"""
    print("Testing program creation...")
    try:
        from pupa_program import create_pupa_program
        program = create_pupa_program()
        assert type(program).__name__ == "PAPILLON"
        print(f"  ✓ Program created: {type(program).__name__}")
        return True
    except Exception as e:
        print(f"  ✗ Program creation failed: {e}")
        return False

def test_files_exist():
    """Check all required files exist"""
    print("Testing file structure...")
    required_files = [
        "papillon/__init__.py",
        "papillon/papillon_signatures.py",
        "papillon/papillon_pipeline.py",
        "papillon/llm_judge.py",
        "papillon/LICENSE_PAPILLON.txt",
        "papillon/README_PAPILLON.md",
        "pupa_data.py",
        "pupa_metric.py",
        "pupa_program.py",
        "run_gepa_pupa.py",
        "run_gepa_pupa_compare.py",
        "sanity_pupa.py",
        "job.pupa_compare.sbatch",
        "job.pupa_compare_resume.sbatch",
        "README.md",
    ]

    base_dir = Path(__file__).parent
    missing = []
    for f in required_files:
        if not (base_dir / f).exists():
            missing.append(f)

    if missing:
        print(f"  ✗ Missing files: {missing}")
        return False
    else:
        print(f"  ✓ All {len(required_files)} required files present")
        return True

def main():
    print("=" * 70)
    print("PUPA2 Installation Verification")
    print("=" * 70)
    print()

    tests = [
        ("File Structure", test_files_exist),
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Program Creation", test_program_creation),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        print()

    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {name}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("✅ All tests passed! pupa is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run sanity check: python sanity_pupa.py --n_examples 3")
        print("  2. Submit SLURM job: sbatch job.pupa_compare.sbatch")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
