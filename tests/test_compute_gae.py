"""
Unit tests for compute_gae (Generalized Advantage Estimation).

Tests verify that the GAE computation is correct by checking against
manually computed expected values for simple known cases.
"""

import torch
import pytest
from gympn.data import compute_gae, compute_advantages


class TestComputeGAE:
    """Test suite for compute_gae correctness."""

    def test_gae_simple_case_no_discounting(self):
        """
        Test GAE on a simple case: single step, no discounting (gamma=1),
        no bootstrapping (value[1]=0).

        Setup:
            - T=1, reward[0]=5, value[0]=3, gamma=1, lambda=1
            - dones=None (or [False])

        Expected:
            - delta[0] = reward[0] + gamma * value[1] - value[0]
                       = 5 + 1 * 0 - 3 = 2
            - advantage[0] = delta[0] = 2
        """
        rewards = torch.tensor([5.0])
        values = torch.tensor([3.0])
        gamma = 1.0
        lam = 1.0
        last_value = 0.0
        dones = None

        adv = compute_gae(rewards, values, gamma, lam, dones=dones, last_value=last_value)

        expected_adv = torch.tensor([2.0])
        assert torch.allclose(adv, expected_adv), f"Expected {expected_adv}, got {adv}"

    def test_gae_two_steps_no_discounting(self):
        """
        Test GAE over two steps: gamma=1, lambda=1, no future value.

        Setup:
            - T=2, rewards=[1, 2], values=[3, 4], gamma=1, lambda=1, last_value=0
            - dones=None

        Expected (working backward from t=1 to t=0):
            - gae[1] = 0 initially
            - delta[1] = rewards[1] + gamma * value[2] - values[1]
                       = 2 + 1 * 0 - 4 = -2
            - gae[1] = delta[1] + gamma * lambda * gae[1] = -2 + 1 * 1 * 0 = -2
            - advantage[1] = -2

            - delta[0] = rewards[0] + gamma * values[1] - values[0]
                       = 1 + 1 * 4 - 3 = 2
            - gae[0] = delta[0] + gamma * lambda * gae[1]
                     = 2 + 1 * 1 * (-2) = 0
            - advantage[0] = 0
        """
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([3.0, 4.0])
        gamma = 1.0
        lam = 1.0
        last_value = 0.0
        dones = None

        adv = compute_gae(rewards, values, gamma, lam, dones=dones, last_value=last_value)

        expected_adv = torch.tensor([0.0, -2.0])
        assert torch.allclose(adv, expected_adv), f"Expected {expected_adv}, got {adv}"

    def test_gae_with_discounting(self):
        """
        Test GAE with discounting (gamma < 1).

        Setup:
            - T=2, rewards=[10, 5], values=[2, 3], gamma=0.9, lambda=0.9, last_value=0
            - dones=None

        Expected (working backward):
            - gae[1] = 0 initially
            - delta[1] = 5 + 0.9 * 0 - 3 = 2
            - gae[1] = 2 + 0.9 * 0.9 * 0 = 2
            - advantage[1] = 2

            - delta[0] = 10 + 0.9 * 3 - 2 = 10 + 2.7 - 2 = 10.7
            - gae[0] = 10.7 + 0.9 * 0.9 * 2 = 10.7 + 1.62 = 12.32
            - advantage[0] = 12.32
        """
        rewards = torch.tensor([10.0, 5.0])
        values = torch.tensor([2.0, 3.0])
        gamma = 0.9
        lam = 0.9
        last_value = 0.0
        dones = None

        adv = compute_gae(rewards, values, gamma, lam, dones=dones, last_value=last_value)

        expected_adv = torch.tensor([12.32, 2.0])
        assert torch.allclose(adv, expected_adv, atol=1e-5), f"Expected {expected_adv}, got {adv}"

    def test_gae_with_done_flag(self):
        """
        Test GAE with episode termination flag (dones).

        Setup:
            - T=3, rewards=[1, 2, 3], values=[1, 1, 1], gamma=1, lambda=1
            - dones=[False, True, False]  (episode ends after step 1)
            - last_value=0

        Expected:
            When dones[t]=True, mask[t]=0, so gae bootstrapping is cut off.

            - gae[2] = 0 initially
            - delta[2] = 3 + 1 * 0 - 1 = 2
            - gae[2] = 2 + 1 * 1 * 1 * 0 = 2  (mask[2]=1 normally but no bootstrap from step 3)

            - delta[1] = 2 + 1 * 1 - 1 = 2
            - gae[1] = 2 + 1 * 1 * 0 * 2 = 2  (mask[1]=0 because dones[1]=True, so gae is not bootstrapped)

            - delta[0] = 1 + 1 * 1 - 1 = 1
            - gae[0] = 1 + 1 * 1 * 1 * 2 = 3  (mask[0]=1, so we bootstrap from gae[1])
        """
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([1.0, 1.0, 1.0])
        gamma = 1.0
        lam = 1.0
        last_value = 0.0
        dones = torch.tensor([False, True, False])

        adv = compute_gae(rewards, values, gamma, lam, dones=dones, last_value=last_value)

        # With mask[1]=0 (dones[1]=True), gae[1] is not bootstrapped to gae[2]
        expected_adv = torch.tensor([3.0, 2.0, 2.0])
        assert torch.allclose(adv, expected_adv, atol=1e-5), f"Expected {expected_adv}, got {adv}"

    def test_gae_with_bootstrap_value(self):
        """
        Test GAE with a non-zero bootstrap/last value.

        Setup:
            - T=1, rewards=[10], values=[5], gamma=0.99, lambda=0.95, last_value=8

        Expected:
            - delta[0] = 10 + 0.99 * 8 - 5 = 10 + 7.92 - 5 = 12.92
            - gae[0] = 12.92 (no further bootstrapping)
        """
        rewards = torch.tensor([10.0])
        values = torch.tensor([5.0])
        gamma = 0.99
        lam = 0.95
        last_value = 8.0
        dones = None

        adv = compute_gae(rewards, values, gamma, lam, dones=dones, last_value=last_value)

        expected_adv = torch.tensor([12.92])
        assert torch.allclose(adv, expected_adv, atol=1e-4), f"Expected {expected_adv}, got {adv}"

    def test_gae_equivalence_compute_advantages(self):
        """
        Test that compute_gae and compute_advantages produce identical results.

        They should be equivalent since compute_gae now delegates to compute_advantages.
        """
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([0.5, 1.5, 2.5, 3.5])
        gamma = 0.99
        lam = 0.97
        last_value = 2.0
        dones = torch.tensor([False, False, True, False])

        adv_gae = compute_gae(rewards, values, gamma, lam, dones=dones, last_value=last_value)
        adv_adv = compute_advantages(rewards, values, gamma, lam, dones=dones, last_value=last_value)

        assert torch.allclose(adv_gae, adv_adv), f"compute_gae and compute_advantages differ:\ngae={adv_gae}\nadv={adv_adv}"

    def test_gae_device_consistency(self):
        """
        Test that compute_gae handles device placement correctly (CPU/GPU).

        Even if inputs are on different devices, the function should handle it gracefully.
        """
        device = 'cpu'  # Use CPU for testing (GPU may not be available)
        rewards = torch.tensor([1.0, 2.0], device=device)
        values = torch.tensor([1.0, 2.0], device=device)
        gamma = 0.99
        lam = 0.97

        adv = compute_gae(rewards, values, gamma, lam)

        assert adv.device == torch.device(device), f"Expected device {device}, got {adv.device}"

    def test_gae_empty_trajectory(self):
        """
        Test that compute_gae handles edge case of empty trajectory gracefully.
        (Though in practice, empty episodes should not occur.)
        """
        rewards = torch.tensor([], dtype=torch.float32)
        values = torch.tensor([], dtype=torch.float32)
        gamma = 0.99
        lam = 0.97

        adv = compute_gae(rewards, values, gamma, lam)

        assert adv.shape[0] == 0, f"Expected empty tensor, got shape {adv.shape}"

    def test_gae_zero_rewards_zero_values(self):
        """
        Test GAE with all zeros (no rewards, no value estimates).

        Expected:
            - All deltas are 0, so all advantages should be 0.
        """
        rewards = torch.tensor([0.0, 0.0, 0.0])
        values = torch.tensor([0.0, 0.0, 0.0])
        gamma = 0.99
        lam = 0.97
        last_value = 0.0

        adv = compute_gae(rewards, values, gamma, lam, dones=None, last_value=last_value)

        expected_adv = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(adv, expected_adv), f"Expected {expected_adv}, got {adv}"

    def test_gae_numerical_stability(self):
        """
        Test GAE with very large and very small values to check numerical stability.
        """
        rewards = torch.tensor([1e6, 1e-6, 1.0])
        values = torch.tensor([1e6, 1e-6, 1.0])
        gamma = 0.99
        lam = 0.97

        adv = compute_gae(rewards, values, gamma, lam)

        # Should not contain NaNs or infs
        assert not torch.isnan(adv).any(), f"GAE produced NaNs: {adv}"
        assert not torch.isinf(adv).any(), f"GAE produced infs: {adv}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

