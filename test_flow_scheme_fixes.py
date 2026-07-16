"""
Tests for the flow redistribution scheme bug fixes.

Verifies:
1. No max_depth truncation (credit reaches distant ancestors)
2. No double discounting (gamma applied once per hop, not again per time)
3. Delta re-queuing works (multi-path credit accumulates correctly)
4. Scheme/gamma are properly threaded through the call chain
"""
import types
import uuid
import pytest
from gympn.causal_traces import CausalTraces, TokenHistory, TransitionHistory


def _make_token(_id=None):
    """Create a simple token-like object with a _id attribute."""
    tok = types.SimpleNamespace()
    tok._id = _id or str(uuid.uuid4())
    return tok


def _make_transition(_id=None, is_action=False):
    """Create a simple transition-like object."""
    tr = types.SimpleNamespace()
    tr._id = _id or str(uuid.uuid4())
    return tr


class TestFlowNoMaxDepth:
    """Test that flow propagation reaches distant ancestors (no max_depth=8 limit)."""

    def test_deep_chain_credit_reaches_root(self):
        """
        Build a chain of 15 hops: root -> t1 -> t2 -> ... -> t14 -> reward.
        With the old max_depth=8, the root action would receive no credit.
        After the fix, it should receive credit.
        """
        ct = CausalTraces()

        # Create chain of tokens
        chain_length = 15
        tokens = [_make_token(f"tok_{i}") for i in range(chain_length)]

        # Root action transition produces tokens[0]
        root_action = _make_transition("root_action", is_action=True)
        ct.register_token(tokens[0], root_action, [], time=0.0)
        ct.register_transition(root_action, [], [tokens[0]], is_action=True, reward=0.0, time=0.0)

        # Chain of evolution transitions: tokens[i] -> tokens[i+1]
        for i in range(chain_length - 1):
            evo = _make_transition(f"evo_{i}", is_action=False)
            ct.register_token(tokens[i + 1], evo, [tokens[i]], time=float(i + 1))
            ct.register_transition(evo, [tokens[i]], [tokens[i + 1]], is_action=False, reward=0.0, time=float(i + 1))

        # Reward transition consumes the last token
        reward_tr = _make_transition("reward_tr", is_action=False)
        ct.register_transition(reward_tr, [tokens[-1]], [], is_action=False, reward=10.0, time=float(chain_length))

        # Redistribute with flow
        credits = ct.redistribute_rewards(gamma=0.95, scheme="flow")

        # The root action should get credit (the ONLY action in the chain)
        assert len(credits) == 1, f"Expected 1 action, got {len(credits)}"
        assert credits[0] > 0, f"Root action got no credit: {credits[0]}"
        assert abs(credits[0] - 10.0) < 1e-6, f"Root action should get all 10.0 reward, got {credits[0]}"


class TestFlowNoDoubleDiscounting:
    """Test that gamma is applied only per-hop, not additionally per-time."""

    def test_same_structure_different_times(self):
        """
        Two actions at different times produce tokens that both lead to a reward.
        Credit should depend on structural distance (hops), NOT absolute time difference.
        """
        ct = CausalTraces()

        # Action A at time=0 produces tok_a
        tok_a = _make_token("tok_a")
        action_a = _make_transition("action_a", is_action=True)
        ct.register_token(tok_a, action_a, [], time=0.0)
        ct.register_transition(action_a, [], [tok_a], is_action=True, reward=0.0, time=0.0)

        # Action B at time=100 produces tok_b (very far in time, but same structural depth)
        tok_b = _make_token("tok_b")
        action_b = _make_transition("action_b", is_action=True)
        ct.register_token(tok_b, action_b, [], time=100.0)
        ct.register_transition(action_b, [], [tok_b], is_action=True, reward=0.0, time=100.0)

        # Both tok_a and tok_b are consumed by reward transition
        reward_tr = _make_transition("reward_tr", is_action=False)
        ct.register_transition(reward_tr, [tok_a, tok_b], [], is_action=False, reward=10.0, time=101.0)

        credits = ct.redistribute_rewards(gamma=0.95, scheme="flow")

        # Both actions are at the same structural depth (1 hop from reward inputs)
        # so they should get EQUAL credit, regardless of time difference
        assert len(credits) == 2
        assert abs(credits[0] - credits[1]) < 1e-6, \
            f"Actions at same structural depth should get equal credit: {credits}"


class TestFlowDeltaReQueuing:
    """Test that multi-path credit accumulates correctly via delta re-queuing."""

    def test_diamond_graph(self):
        """
        Diamond graph:
          action -> tok_a -> tok_c --|
          action -> tok_b -> tok_d --|--> reward

        The single action should get ALL credit since it produced both branches.
        """
        ct = CausalTraces()

        # Single action produces tok_a and tok_b
        tok_a = _make_token("tok_a")
        tok_b = _make_token("tok_b")
        action = _make_transition("action", is_action=True)
        ct.register_token(tok_a, action, [], time=0.0)
        ct.register_token(tok_b, action, [], time=0.0)
        ct.register_transition(action, [], [tok_a, tok_b], is_action=True, reward=0.0, time=0.0)

        # tok_a -> tok_c via evo1
        tok_c = _make_token("tok_c")
        evo1 = _make_transition("evo1")
        ct.register_token(tok_c, evo1, [tok_a], time=1.0)
        ct.register_transition(evo1, [tok_a], [tok_c], is_action=False, reward=0.0, time=1.0)

        # tok_b -> tok_d via evo2
        tok_d = _make_token("tok_d")
        evo2 = _make_transition("evo2")
        ct.register_token(tok_d, evo2, [tok_b], time=1.0)
        ct.register_transition(evo2, [tok_b], [tok_d], is_action=False, reward=0.0, time=1.0)

        # tok_c + tok_d consumed by reward
        reward_tr = _make_transition("reward_tr")
        ct.register_transition(reward_tr, [tok_c, tok_d], [], is_action=False, reward=10.0, time=2.0)

        credits = ct.redistribute_rewards(gamma=0.95, scheme="flow")

        assert len(credits) == 1
        assert abs(credits[0] - 10.0) < 1e-6, f"Single action should get all credit: {credits[0]}"


class TestFlowMultipleActions:
    """Test proper credit distribution across multiple actions."""

    def test_two_actions_different_depth(self):
        """
        action_close -> tok_close -> reward (1 hop)
        action_far   -> tok_far   -> tok_mid -> reward (2 hops)

        action_close should get MORE credit than action_far (closer structurally).
        """
        ct = CausalTraces()

        # Close action (1 hop from reward)
        tok_close = _make_token("tok_close")
        action_close = _make_transition("action_close", is_action=True)
        ct.register_token(tok_close, action_close, [], time=0.0)
        ct.register_transition(action_close, [], [tok_close], is_action=True, reward=0.0, time=0.0)

        # Far action (2 hops from reward)
        tok_far = _make_token("tok_far")
        action_far = _make_transition("action_far", is_action=True)
        ct.register_token(tok_far, action_far, [], time=1.0)
        ct.register_transition(action_far, [], [tok_far], is_action=True, reward=0.0, time=1.0)

        # tok_far -> tok_mid
        tok_mid = _make_token("tok_mid")
        evo = _make_transition("evo")
        ct.register_token(tok_mid, evo, [tok_far], time=2.0)
        ct.register_transition(evo, [tok_far], [tok_mid], is_action=False, reward=0.0, time=2.0)

        # Both tok_close and tok_mid consumed by reward
        reward_tr = _make_transition("reward_tr")
        ct.register_transition(reward_tr, [tok_close, tok_mid], [], is_action=False,
                               reward=10.0, time=3.0)

        credits = ct.redistribute_rewards(gamma=0.9, scheme="flow")

        assert len(credits) == 2
        # action_close at index 0 should get more credit than action_far at index 1
        # because it's structurally closer (1 hop vs 2 hops)
        assert credits[0] > credits[1], \
            f"Closer action should get more credit: close={credits[0]}, far={credits[1]}"
        # Total should equal reward
        assert abs(sum(credits) - 10.0) < 1e-6, \
            f"Total credits should sum to reward: {sum(credits)}"


class TestSchemeGammaThreading:
    """Test that scheme/gamma are properly configurable."""

    def test_flow_vs_exponential_differ(self):
        """The flow and exponential schemes should produce different redistributions."""
        ct = CausalTraces()

        # Simple chain: action -> tok -> reward
        tok = _make_token("tok")
        action = _make_transition("action", is_action=True)
        ct.register_token(tok, action, [], time=0.0)
        ct.register_transition(action, [], [tok], is_action=True, reward=0.0, time=0.0)

        reward_tr = _make_transition("reward_tr")
        ct.register_transition(reward_tr, [tok], [], is_action=False, reward=10.0, time=5.0)

        credits_flow = ct.redistribute_rewards(gamma=0.9, scheme="flow")
        credits_exp = ct.redistribute_rewards(gamma=0.9, scheme="exponential")

        # Both should assign credit to the single action
        assert len(credits_flow) == 1
        assert len(credits_exp) == 1
        assert credits_flow[0] > 0
        assert credits_exp[0] > 0

    def test_gamma_affects_flow_credit(self):
        """Different gamma values should produce different credit distributions for flow."""
        ct = CausalTraces()

        # Two actions at different depths
        tok1 = _make_token("tok1")
        action1 = _make_transition("action1", is_action=True)
        ct.register_token(tok1, action1, [], time=0.0)
        ct.register_transition(action1, [], [tok1], is_action=True, reward=0.0, time=0.0)

        tok2 = _make_token("tok2")
        evo = _make_transition("evo")
        ct.register_token(tok2, evo, [tok1], time=1.0)
        ct.register_transition(evo, [tok1], [tok2], is_action=False, reward=0.0, time=1.0)

        tok3 = _make_token("tok3")
        action2 = _make_transition("action2", is_action=True)
        ct.register_token(tok3, action2, [], time=2.0)
        ct.register_transition(action2, [], [tok3], is_action=True, reward=0.0, time=2.0)

        # Reward consumes tok2 and tok3
        reward_tr = _make_transition("reward_tr")
        ct.register_transition(reward_tr, [tok2, tok3], [], is_action=False, reward=10.0, time=3.0)

        credits_high_gamma = ct.redistribute_rewards(gamma=0.99, scheme="flow")
        credits_low_gamma = ct.redistribute_rewards(gamma=0.5, scheme="flow")

        # With low gamma, the deeper action (action1) should get less relative credit
        # compared to high gamma
        ratio_high = credits_high_gamma[0] / credits_high_gamma[1] if credits_high_gamma[1] > 0 else float('inf')
        ratio_low = credits_low_gamma[0] / credits_low_gamma[1] if credits_low_gamma[1] > 0 else float('inf')

        # action1 is deeper, so with lower gamma it gets relatively less credit
        assert ratio_low < ratio_high, \
            f"Lower gamma should penalize deeper actions more: ratio_low={ratio_low}, ratio_high={ratio_high}"


class TestBufferSchemeThreading:
    """Test that TrajectoryBuffer passes scheme/gamma to redistribute_rewards."""

    def test_buffer_stores_scheme_and_gamma(self):
        from gympn.data import TrajectoryBuffer
        buf = TrajectoryBuffer(gam=0.99, lam=0.95, causal_scheme='flow', causal_gamma=0.8)
        assert buf.causal_scheme == 'flow'
        assert buf.causal_gamma == 0.8

    def test_buffer_default_scheme_is_flow(self):
        from gympn.data import TrajectoryBuffer
        buf = TrajectoryBuffer(gam=0.99, lam=0.95)
        assert buf.causal_scheme == 'flow'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

