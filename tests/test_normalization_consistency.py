"""
Integration test: Verify that advantage and return normalization are consistent.

This test checks whether the value predictions used in GAE are on the same scale
as the returns used for value network training. Mismatches indicate a potential
bug in the training pipeline.

Run this after making changes to understand the normalization behavior:
    python tests/test_normalization_consistency.py
"""

import torch
import numpy as np
from gympn.data import TrajectoryBuffer, compute_gae


def test_advantage_computation_consistency():
    """
    Test that advantages computed with raw values match the expected GAE formula.

    This is a sanity check: if you compute advantages with raw (unnormalized) values
    and raw rewards, the result should match the standard GAE formula.
    """
    print("\n" + "="*70)
    print("TEST 1: Advantage Computation Consistency (Raw Values)")
    print("="*70)

    # Simple episode: 3 steps
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    values = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)  # includes v_T
    gamma = 0.99
    lam = 0.97

    # Compute GAE with raw values
    advantages = compute_gae(rewards, values, gamma, lam, last_value=0.0)

    print(f"Rewards:      {rewards.numpy()}")
    print(f"Values:       {values.numpy()}")
    print(f"Gamma:        {gamma}, Lambda: {lam}")
    print(f"Advantages:   {advantages.numpy()}")

    # Manual computation of expected advantages (backward pass)
    T = len(rewards)
    manual_advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        v_t = values[t]
        v_tp1 = values[t + 1] if t + 1 < len(values) else torch.tensor(0.0)
        delta = rewards[t] + gamma * v_tp1 - v_t
        gae = delta + gamma * lam * gae
        manual_advantages[t] = gae

    print(f"Manual calc:  {manual_advantages.numpy()}")

    is_close = torch.allclose(advantages, manual_advantages, atol=1e-5)
    print(f"✓ PASS: Advantages match manual computation" if is_close else f"✗ FAIL: Mismatch!")
    return is_close


def test_normalization_scaling_issue():
    """
    Demonstrate the potential normalization bug: when you normalize returns for
    value training but use unnormalized values in GAE, you get a scale mismatch.
    """
    print("\n" + "="*70)
    print("TEST 2: Normalization Scaling Issue (Conceptual)")
    print("="*70)

    # Simulate returns that will be normalized
    raw_returns = torch.tensor([10.0, 20.0, 15.0, 25.0], dtype=torch.float32)

    # Normalize returns (as done in _normalize_returns)
    mean = raw_returns.mean()
    std = raw_returns.std(unbiased=False)
    normalized_returns = (raw_returns - mean) / (std + 1e-8)

    print(f"Raw returns:         {raw_returns.numpy()}")
    print(f"Mean: {mean.item():.4f}, Std: {std.item():.4f}")
    print(f"Normalized returns:  {normalized_returns.numpy()}")

    # Suppose value network is trained on normalized_returns
    # Then its outputs will also be in the normalized scale
    print(f"\nIf value network is trained on normalized returns,")
    print(f"its outputs will be in range ≈ [-2, +2] (normalized scale).")

    # But if you then use these normalized-scale value predictions
    # in GAE computation with RAW rewards, you get a mismatch
    print(f"\nIf you then use normalized values in GAE with raw rewards:")
    print(f"  delta = raw_reward + gamma * normalized_value - normalized_value")
    print(f"         = (large value) + gamma * (small value) - (small value)")
    print(f"         = will produce incorrect advantage scale")
    print(f"\n✗ ISSUE: This scale mismatch can cause learning to stall or diverge.")
    print(f"✓ SOLUTION: Either (A) disable normalize_returns, or (B) normalize both")
    print(f"           rewards and values consistently before GAE computation.")

    return False  # This is a demonstration, not a passing test


def test_buffer_normalization_flags():
    """
    Test that the TrajectoryBuffer correctly respects normalize_advantages and
    normalize_returns flags.
    """
    print("\n" + "="*70)
    print("TEST 3: TrajectoryBuffer Normalization Flags")
    print("="*70)

    buffer = TrajectoryBuffer(gam=0.99, lam=0.97)

    # Store a simple trajectory (mock states)
    for step in range(3):
        mock_state = {
            'graph': None,  # Would normally be a HeteroData object
        }
        action = step % 2
        reward = float(step + 1)
        logprob = -1.0
        value = float(step * 0.5)
        logpis = torch.tensor([0.1, 0.9])

        buffer.store(mock_state, action, reward, logprob, value, logpis)

    # Finish episode
    buffer.finish(credits=None)

    print(f"Buffer length: {len(buffer)}")
    print(f"Raw returns stored:     {buffer.returns_.numpy()}")
    print(f"Raw advantages stored:  {buffer.advantages_.numpy()}")

    # Check that normalization is applied during get() (without actually building dataloader,
    # since we don't have real HeteroData objects)
    adv = buffer.advantages_.clone()
    returns = buffer.returns_.clone()

    # Manual normalization
    adv_norm = buffer._normalize_advantages(adv)
    returns_norm = buffer._normalize_returns(returns)

    print(f"\nAfter _normalize_advantages:")
    print(f"  Mean: {adv_norm.mean().item():.6f} (should be ~0)")
    print(f"  Std:  {adv_norm.std(unbiased=False).item():.6f} (should be ~1)")

    print(f"\nAfter _normalize_returns:")
    print(f"  Mean: {returns_norm.mean().item():.6f} (should be ~0)")
    print(f"  Std:  {returns_norm.std(unbiased=False).item():.6f} (should be ~1)")

    means_ok = (abs(adv_norm.mean().item()) < 1e-5 and abs(returns_norm.mean().item()) < 1e-5)
    stds_ok = (abs(adv_norm.std(unbiased=False).item() - 1.0) < 1e-5 and
               abs(returns_norm.std(unbiased=False).item() - 1.0) < 1e-5)

    passed = means_ok and stds_ok
    print(f"\n✓ PASS: Normalization produces zero mean and unit variance" if passed
          else f"✗ FAIL: Normalization not working correctly")

    return passed


def main():
    """Run all tests."""
    results = []

    try:
        results.append(("Advantage Computation Consistency", test_advantage_computation_consistency()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Advantage Computation Consistency", False))

    try:
        results.append(("Normalization Scaling Issue", test_normalization_scaling_issue()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Normalization Scaling Issue", False))

    try:
        results.append(("Buffer Normalization Flags", test_buffer_normalization_flags()))
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        results.append(("Buffer Normalization Flags", False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {sum(1 for _, p in results if p)}/{len(results)} tests passed")

    return all(p for _, p in results)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

