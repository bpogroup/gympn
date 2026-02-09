#!/usr/bin/env python
"""
Simple test script for RUDDER implementation.
Tests basic functionality without pytest.
"""

import sys
sys.path.insert(0, '.')

print("Testing RUDDER implementation...")

# Test 1: Import
print("\n[TEST 1] Importing RUDDER components...")
try:
    from gympn.rudder import RUDDERNetwork, RUDDERCreditAssignment, RUDDERAgent
    print("✓ Successfully imported RUDDER components")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Network initialization
print("\n[TEST 2] Creating RUDDERNetwork...")
try:
    import torch
    network = RUDDERNetwork(state_dim=64, hidden_dim=128)
    print(f"✓ Network created: state_dim={network.state_dim}, hidden_dim={network.hidden_dim}")
except Exception as e:
    print(f"✗ Network creation failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n[TEST 3] Testing forward pass...")
try:
    batch_size = 4
    seq_len = 20
    states = torch.randn(batch_size, seq_len, 64)
    lengths = torch.tensor([20, 18, 20, 15])

    step_rewards, total_reward = network(states, lengths)

    assert step_rewards.shape == (batch_size, seq_len), f"Wrong step_rewards shape: {step_rewards.shape}"
    assert total_reward.shape == (batch_size,), f"Wrong total_reward shape: {total_reward.shape}"
    print(f"✓ Forward pass successful")
    print(f"  step_rewards shape: {step_rewards.shape}")
    print(f"  total_reward shape: {total_reward.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Credit Assignment
print("\n[TEST 4] Creating RUDDERCreditAssignment...")
try:
    import numpy as np
    ca = RUDDERCreditAssignment(state_dim=32, hidden_dim=64, learning_rate=1e-3)
    print(f"✓ Credit Assignment created")
except Exception as e:
    print(f"✗ Credit Assignment creation failed: {e}")
    sys.exit(1)

# Test 5: Training step
print("\n[TEST 5] Testing training step...")
try:
    trajectories = [
        {'states': np.random.randn(20, 32).astype(np.float32)}
        for _ in range(4)
    ]
    returns = np.random.randn(4).astype(np.float32)

    loss = ca.train_step(trajectories, returns)
    assert isinstance(loss, float), f"Loss is not a float: {type(loss)}"
    assert loss > 0, f"Loss is non-positive: {loss}"
    print(f"✓ Training step successful (loss={loss:.4f})")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: RUDDER Agent
print("\n[TEST 6] Creating RUDDERAgent...")
try:
    agent = RUDDERAgent(state_dim=64, hidden_dim=128)
    print(f"✓ RUDDERAgent created")
except Exception as e:
    print(f"✗ RUDDERAgent creation failed: {e}")
    sys.exit(1)

# Test 7: Adding trajectories
print("\n[TEST 7] Adding trajectories to agent...")
try:
    states = np.random.randn(20, 64).astype(np.float32)
    actions = np.random.randint(0, 4, 20)
    rewards = np.ones(20)

    agent.add_trajectory(states, actions, rewards, 20.0)
    assert len(agent.trajectory_buffer) == 1
    print(f"✓ Trajectory added successfully")
except Exception as e:
    print(f"✗ Adding trajectory failed: {e}")
    sys.exit(1)

# Test 8: Agent training
print("\n[TEST 8] Training agent...")
try:
    for _ in range(3):
        states = np.random.randn(20, 64).astype(np.float32)
        actions = np.random.randint(0, 4, 20)
        rewards = np.ones(20)
        agent.add_trajectory(states, actions, rewards, 20.0)

    loss = agent.train(num_epochs=2)
    assert isinstance(loss, float)
    print(f"✓ Agent training successful (loss={loss:.4f})")
    print(f"  Buffer cleared: {len(agent.trajectory_buffer) == 0}")
except Exception as e:
    print(f"✗ Agent training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Reward redistribution
print("\n[TEST 9] Testing reward redistribution...")
try:
    test_states = np.random.randn(15, 64).astype(np.float32)
    test_rewards = np.ones(15)

    redistributed = agent.redistribute_rewards(test_states, test_rewards)

    assert redistributed.shape == (15,), f"Wrong shape: {redistributed.shape}"
    assert not np.any(np.isnan(redistributed)), "NaN values in redistributed rewards"
    print(f"✓ Reward redistribution successful")
    print(f"  Original return: {test_rewards.sum():.4f}")
    print(f"  Redistributed return: {redistributed.sum():.4f}")
except Exception as e:
    print(f"✗ Reward redistribution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nRUDDER implementation is working correctly and ready for use.")
print("\nNext steps:")
print("2. Run compare_credit_assignment.py for full comparison")
print("3. Check docs/rudder_guide.md for detailed documentation")

