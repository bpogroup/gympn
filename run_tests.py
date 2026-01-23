#!/usr/bin/env python
"""
Quick test runner script to validate the test suite.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py network      # Run network tests only
    python run_tests.py ppo          # Run PPO algorithm tests
"""

import sys
import subprocess
from pathlib import Path


def run_tests(test_pattern: str = ""):
    """Run tests with given pattern."""

    test_dir = Path(__file__).parent / "tests"

    test_files = {
        "agents": test_dir / "test_agents.py",
        "environment": test_dir / "test_environment.py",
        "simulator": test_dir / "test_simulator.py",
    }

    print("\n" + "="*70)
    print("GymPN Test Suite Runner")
    print("="*70 + "\n")

    if test_pattern and test_pattern in test_files:
        # Run specific test file
        test_file = test_files[test_pattern]
        print(f"Running {test_pattern.upper()} tests: {test_file.name}\n")
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]
    elif test_pattern:
        print(f"Unknown test pattern: {test_pattern}")
        print(f"Available: {', '.join(test_files.keys())}")
        return 1
    else:
        # Run all tests
        print("Running ALL tests...\n")
        cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]

    # Run pytest
    result = subprocess.run(cmd)

    print("\n" + "="*70)
    if result.returncode == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70 + "\n")

    return result.returncode


if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(run_tests(pattern))

