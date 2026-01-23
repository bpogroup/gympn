# GymPN Test Suite Guide

## Quick Start

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Category
```bash
python run_tests.py network      # Network training tests
python run_tests.py ppo          # PPO algorithm tests
python run_tests.py batch        # Batch handling tests
python run_tests.py causal       # Causal RL tests
python run_tests.py pipeline     # Training pipeline tests
```

### Run Tests with Pytest Directly
```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_ppo_algorithm.py -v

# Run specific test class
pytest tests/test_ppo_algorithm.py::TestPPOClipping -v

# Run specific test
pytest tests/test_ppo_algorithm.py::TestPPOClipping::test_ppo_clip_basic -v
```

---

## Test Suite Overview

The test suite contains **5 comprehensive test modules**:

### 1. **Network Training Tests** (`test_network_training.py`)
Tests for neural network components.

**What it tests:**
- Policy network forward/backward passes
- Value network training
- Gradient computation
- Numerical stability

**Quick Example:**
```bash
pytest tests/test_network_training.py::TestPolicyNetworkTraining -v
```

### 2. **PPO Algorithm Tests** (`test_ppo_algorithm.py`)
Tests for the Proximal Policy Optimization algorithm.

**What it tests:**
- PPO clipping mechanism
- Entropy regularization
- KL divergence tracking
- Value function loss
- Complete training steps

**Quick Example:**
```bash
pytest tests/test_ppo_algorithm.py::TestPPOClipping -v
```

### 3. **Batch Handling Tests** (`test_batch_handling.py`)
Tests for trajectory buffer and batch operations.

**What it tests:**
- Trajectory storage
- Batch creation and shuffling
- Reward/return normalization
- GAE computation
- Buffer management

**Quick Example:**
```bash
pytest tests/test_batch_handling.py::TestTrajectoryBufferBasics -v
```

### 4. **Training Pipeline Tests** (`test_training_pipeline.py`)
End-to-end integration tests.

**What it tests:**
- Full training loops
- Data flow through system
- Checkpointing/loading
- Multi-epoch training
- Error handling
- Training stability

**Quick Example:**
```bash
pytest tests/test_training_pipeline.py::TestMultipleEpochs -v
```

---

## Important Test Details

### Test Status Indicators

When running tests, you'll see:
- ✓ **PASSED** - Test completed successfully
- ✗ **FAILED** - Test failed, check the error message
- ⊘ **SKIPPED** - Test skipped (e.g., missing dependencies)

### Output Example

```
test_network_training.py::TestPolicyNetworkTraining::test_policy_forward_pass PASSED
test_network_training.py::TestPolicyNetworkTraining::test_policy_output_shape PASSED
test_network_training.py::TestPolicyNetworkTraining::test_policy_gradient_flow PASSED
...
========================== 50 passed in 12.34s ==========================
```

### Verbosity Levels

```bash
pytest tests/ -q                    # Quiet (minimal output)
pytest tests/ -v                    # Verbose (detailed output)
pytest tests/ -vv                   # Very verbose (very detailed)
```

---

## Useful Pytest Options

### Run with Coverage
```bash
pytest tests/ --cov=gympn --cov-report=html
# Creates coverage report in htmlcov/index.html
```

### Run specific markers
```bash
pytest tests/ -m "not slow"         # Skip slow tests
pytest tests/ -k "PPO"              # Only run tests with "PPO" in name
```

### Show print output
```bash
pytest tests/ -s                    # Show print statements
pytest tests/ -s -v                 # Verbose + print statements
```

### Stop on first failure
```bash
pytest tests/ -x                    # Stop after first failure
pytest tests/ --tb=short            # Short traceback format
pytest tests/ --tb=long             # Long traceback format
```

---

## Understanding Test Failures

### Common Failure Types

**1. Import Error**
```
ModuleNotFoundError: No module named 'gympn'
```
Solution: Ensure you're in the project root and have installed dependencies:
```bash
pip install -e .
```

**2. Missing Dependencies**
```
ModuleNotFoundError: No module named 'torch'
```
Solution: Install required packages:
```bash
pip install -r requirements.txt
```

**3. Numerical Instability**
```
AssertionError: value is NaN
```
Solution: Check numerical range of inputs, may need normalization

**4. Shape Mismatch**
```
RuntimeError: Expected shape [X], got [Y]
```
Solution: Verify tensor dimensions in network architecture

---

## Test Development Guidelines

### Adding New Tests

When adding new tests, follow this structure:

```python
import pytest
from gympn.module import YourClass

class TestYourFeature:
    """Test suite for your feature."""
    
    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        # Initialize test objects
        return {'object': YourClass()}
    
    def test_your_functionality(self, setup):
        """Test specific functionality."""
        obj = setup['object']
        
        # Test code here
        result = obj.method()
        
        # Assertions
        assert result == expected_value
        print("✓ Test passed")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Best Practices

1. **One concept per test** - Each test should verify one thing
2. **Clear names** - Test names should describe what they test
3. **Use fixtures** - Reuse setup code with pytest fixtures
4. **Be deterministic** - Tests should produce same results every time
5. **Handle errors gracefully** - Use pytest.skip for unavailable features

---

## CI/CD Integration

### Running Tests in GitHub Actions

```yaml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v --cov=gympn
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python -m pytest tests/ -q
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Performance Testing

### Run with Profiling
```bash
pytest tests/ --durations=10      # Show 10 slowest tests
```

### Run Specific Duration Tests
```bash
pytest tests/ -m "not slow"       # Skip slow tests (if marked)
```

### Benchmark Tests
```bash
pytest tests/ --benchmark-only    # Run only benchmark tests
```

---

## Troubleshooting

### Tests Hang
- **Cause:** Infinite loop or deadlock in code
- **Solution:** Use timeout: `pytest tests/ --timeout=30`

### Tests Fail Intermittently
- **Cause:** Race condition or random seed issue
- **Solution:** Set seed in test fixtures

### Memory Issues
- **Cause:** Large tensors or memory leaks
- **Solution:** Profile with `memory_profiler` or check for leaks

### GPU Issues (if running on GPU)
- **Cause:** CUDA out of memory
- **Solution:** Reduce batch size in tests or use CPU-only mode

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pytest tests/` | Run all tests |
| `pytest tests/ -v` | Run all tests (verbose) |
| `pytest tests/test_ppo_algorithm.py` | Run specific file |
| `pytest tests/ -k "PPO"` | Run tests matching pattern |
| `pytest tests/ --cov=gympn` | Run with coverage |
| `pytest tests/ -x` | Stop on first failure |
| `pytest tests/ -s` | Show print statements |
| `python run_tests.py` | Use test runner script |

---

## Getting Help

If tests fail:

1. **Check the error message** - Usually very informative
2. **Read the assertion** - Shows what was expected vs actual
3. **Check recent changes** - Test failures often indicate regressions
4. **Run in verbose mode** - `pytest tests/ -vv -s`
5. **Isolate the test** - Run just the failing test
6. **Check dependencies** - Ensure all packages installed

---

## Next Steps

After tests pass:
1. Review the TEST_SUITE_SUMMARY.md for complete details
2. Read individual test files for implementation patterns
3. Check gympn/ source code for what's being tested
4. Add new tests as features are added

Happy testing! 🎉

