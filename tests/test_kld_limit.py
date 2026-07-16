import torch
from gympn.agents import PPOAgent


class DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x):
        # Return a tensor-shaped value to satisfy callers
        return torch.tensor([0.0])


def test_kld_computed_but_no_early_stop():
    """KLD is computed for monitoring but does NOT cause early stopping.

    With variable-size action sets (Petri net environments), the per-sample
    "KLD" (really just a log importance ratio) is ill-defined and incomparable
    across states.  PPO's clipped surrogate handles the trust region instead.
    """
    policy = DummyNet()
    value = DummyNet()
    # Even with kld_limit set, early stopping is removed from training loops
    agent = PPOAgent(policy_network=policy, value_network=value, method='clip', eps=0.2,
                     policy_lr=1e-3, policy_updates=3, value_lr=1e-3, value_updates=1,
                     gam=0.99, lam=0.95, kld_limit=0.1, ent_bonus=0.0)

    # Prepare a fake dataloader with 5 "batches"
    dataloader = [None] * 5

    # KLD values — some exceed the old limit of 0.1
    klds = [0.2, 0.02, 0.01, 0.0, 0.0]
    call_count = [0]

    def fake_step(batch):
        k = klds[call_count[0]]
        call_count[0] += 1
        # return tuple: (loss_policy, kld, ent, policy_core_loss) -- the
        # value model is fit separately (_fit_value_model), decoupled from
        # this policy step.
        return 0.0, float(k), 0.0, 0.0

    agent._fit_policy_and_value_model_step = fake_step
    # Value fitting is decoupled from the policy loop under test (its own
    # value_updates passes over the same dataloader) -- stub it out so the
    # fake `None` batches don't reach the real value-model step.
    agent._fit_value_model = lambda dataloader, epochs=1: {'loss': []}

    history = agent._fit_policy_and_value_models(dataloader, epochs=1)

    # All 5 batches should be processed (no early stopping)
    assert call_count[0] == 5, f"Expected 5 batches processed, got {call_count[0]}"
    assert 'kld' in history
    assert history['kld'].size == 1
    # Average KLD across all 5 batches: (0.2 + 0.02 + 0.01 + 0 + 0) / 5 = 0.046
    assert abs(history['kld'][0] - 0.046) < 0.01


def test_kld_limit_defaults_to_none():
    """Verify that the default kld_limit is None (disabled)."""
    policy = DummyNet()
    value = DummyNet()
    agent = PPOAgent(policy_network=policy, value_network=value, method='clip', eps=0.2)
    assert agent.kld_limit is None

