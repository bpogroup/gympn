#!/usr/bin/env python
"""
Minimal validation script to ensure compare_convergence_speed.py is working.
Tests import, configuration, and basic instantiation only (does not run training).
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("TEST 1: Checking imports...")
    try:
        import numpy as np
        print("  ✓ numpy")

        import torch
        print("  ✓ torch")

        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")

        from gympn.simulator import GymProblem
        print("  ✓ GymProblem")

        from gympn.agents import PPOAgent
        print("  ✓ PPOAgent")

        from gympn.agents_dcl import DCLAgent
        print("  ✓ DCLAgent")

        from gympn.logging_utils import get_logger
        print("  ✓ get_logger")

        print("✓ All imports successful\n")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}\n")
        return False

def test_logger():
    """Test logger initialization."""
    print("TEST 2: Testing logger...")
    try:
        from gympn.logging_utils import get_logger
        logger = get_logger(verbose=1)
        logger.info("Test message")
        print("✓ Logger works\n")
        return True
    except Exception as e:
        print(f"✗ Logger failed: {e}\n")
        return False

def test_comparison_config():
    """Test ComparisonConfig instantiation."""
    print("TEST 3: Testing ComparisonConfig...")
    try:
        from examples.compare_convergence_speed import ComparisonConfig
        config = ComparisonConfig()

        assert config.num_seeds == 3, "Default seeds should be 3"
        assert config.epochs == 100, "Default epochs should be 100"
        assert config.episodes_per_epoch == 20, "Default episodes should be 20"

        print(f"  ✓ num_seeds: {config.num_seeds}")
        print(f"  ✓ epochs: {config.epochs}")
        print(f"  ✓ episodes_per_epoch: {config.episodes_per_epoch}")
        print(f"  ✓ output_dir: {config.output_dir}")
        print("✓ ComparisonConfig works\n")
        return True
    except Exception as e:
        print(f"✗ ComparisonConfig failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_class():
    """Test ConvergenceComparison instantiation."""
    print("TEST 4: Testing ConvergenceComparison...")
    try:
        from examples.compare_convergence_speed import ComparisonConfig, ConvergenceComparison
        config = ComparisonConfig()
        comparison = ConvergenceComparison(config)

        assert hasattr(comparison, 'run_comparison'), "Missing run_comparison method"
        assert hasattr(comparison, '_train_ppo_clip'), "Missing _train_ppo_clip method"
        assert hasattr(comparison, '_train_ppo_causal'), "Missing _train_ppo_causal method"
        assert hasattr(comparison, '_train_dcl'), "Missing _train_dcl method"
        assert hasattr(comparison, '_create_simple_postpone_env'), "Missing _create_simple_postpone_env method"

        print("  ✓ Has run_comparison method")
        print("  ✓ Has _train_ppo_clip method")
        print("  ✓ Has _train_ppo_causal method")
        print("  ✓ Has _train_dcl method")
        print("  ✓ Has _create_simple_postpone_env method")
        print("✓ ConvergenceComparison works\n")
        return True
    except Exception as e:
        print(f"✗ ConvergenceComparison failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_environment_creation():
    """Test that environment creation works."""
    print("TEST 5: Testing environment creation...")
    try:
        from examples.compare_convergence_speed import ComparisonConfig, ConvergenceComparison
        config = ComparisonConfig()
        comparison = ConvergenceComparison(config)

        # Create one environment to verify the method works
        env = comparison._create_simple_postpone_env()

        print(f"  ✓ Environment created: {type(env).__name__}")
        print(f"  ✓ Environment type: {type(env)}")
        print("✓ Environment creation works\n")
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("CONVERGENCE COMPARISON VALIDATION TESTS")
    print("=" * 70)
    print()

    results = {
        "Imports": test_imports(),
        "Logger": test_logger(),
        "ComparisonConfig": test_comparison_config(),
        "ConvergenceComparison": test_comparison_class(),
        "Environment Creation": test_environment_creation(),
    }

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED - Ready to run comparison!")
        print("\nTo run the comparison, use:")
        print("  python examples/compare_convergence_speed.py")
        print("  or")
        print("  python run_comparison.py")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Please fix issues before running")
        return 1

if __name__ == "__main__":
    sys.exit(main())

