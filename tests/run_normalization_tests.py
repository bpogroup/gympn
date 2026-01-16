#!/usr/bin/env python3
"""
QUICK TEST GUIDE: Verify Normalization Fixes
==============================================

Run this to verify all the normalization fixes are working correctly.
"""

import sys
import subprocess

def run_test(name, cmd):
    """Run a test and report results."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, cwd='.')
        return result.returncode == 0
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def main():
    """Run all tests."""
    tests = [
        ("Unit Tests: compute_gae correctness",
         "python -m pytest tests/test_compute_gae.py -v"),

        ("Integration Tests: Normalization consistency",
         "python tests/test_normalization_consistency.py"),
    ]

    results = []
    for name, cmd in tests:
        passed = run_test(name, cmd)
        results.append((name, passed))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{total} test suites passed")

    if passed_count == total:
        print("\n✓ All tests passed! The normalization fixes are working correctly.")
        print("\nNEXT STEPS:")
        print("1. Run example_simple_postpone.py with ALGORITHM='ppo-clip'")
        print("   Expected: Policy should reach ~20 reward within 100 epochs")
        print("2. Monitor training curves (loss, KLD, returns)")
        print("   Expected: Smooth improvement without stalling")
        return 0
    else:
        print("\n✗ Some tests failed. Review the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

