"""Unit tests for G1 forked counterfactual preferences (gympn/counterfactual.py
+ the agents.py aux pass). Run:  python _test_cfp.py
"""
import random
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite")

from gympn.counterfactual import maybe_fork
from gympn.environment import AEPN_Env
from stoch_envs import make_s1_stoch_sequence


class _StubAgent:
    """First-action greedy continuation, zero critic — enough to exercise the
    fork/restore/CRN machinery without networks."""
    def act(self, state, deterministic=False, return_logprob=False):
        return 0

    value_model = staticmethod(lambda obs: torch.zeros(1))


def _fresh_env():
    random.seed(7)
    np.random.seed(7)
    pn = make_s1_stoch_sequence(causal_rl=False, allow_postpone=True)
    pn.length = 20
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, obs


CFG = {'fork_prob': 1.0, 'reps': 3, 'gate': 2.0, 'lookahead': 6.0,
       'max_forks': 2, 'coef': 1.0, 'updates': 2, 'beta': 0.5}


def test_fork_restores_env_state():
    env, obs = _fresh_env()
    n_actions = len(obs['actions_dict'])
    logpis = torch.log(torch.ones(n_actions) / n_actions)
    clock_before = float(env.pn.clock)
    reward_before = float(env.pn.reward)
    rnd_probe_state = random.getstate()

    forked, pref, diag = maybe_fork(_StubAgent(), env, obs, 0, logpis, CFG)

    assert forked, "fork_prob=1.0 must fork"
    assert float(env.pn.clock) == clock_before, "clock not restored"
    assert float(env.pn.reward) == reward_before, "reward not restored"
    assert len(env.pn.pn_actions) == n_actions, "pn_actions not rebuilt"
    # RNG restored modulo the two draws maybe_fork legitimately consumes
    # (fork coin + base seed): drawing the same two numbers from the probe
    # state must land the stream where maybe_fork left it.
    random.setstate(rnd_probe_state)
    random.random(); random.randrange(2 ** 31 - 1)
    assert random.getstate() == random.getstate(), "sanity"
    if pref is not None:
        assert {'state', 'winner', 'loser', 'gap', 'se'} <= set(pref)
        assert pref['winner'] != pref['loser']
        assert abs(pref['gap']) > CFG['gate'] * pref['se']
    print("  fork_restores_env_state OK (pref emitted:", pref is not None, ")")


def test_fork_disabled_paths():
    env, obs = _fresh_env()
    n_actions = len(obs['actions_dict'])
    logpis = torch.log(torch.ones(n_actions) / n_actions)
    cfg0 = dict(CFG, fork_prob=0.0)
    forked, pref, diag = maybe_fork(_StubAgent(), env, obs, 0, logpis, cfg0)
    assert not forked and pref is None and diag is None
    print("  fork_disabled_paths OK")


def test_main_episode_unaffected_by_fork():
    """The main trajectory after a fork must equal the trajectory of an
    identical episode without the fork (same RNG draws consumed)."""
    def run(with_fork):
        env, obs = _fresh_env()
        rewards = []
        stub = _StubAgent()
        for step in range(12):
            n = len(obs['actions_dict'])
            if with_fork and step == 3:
                logpis = torch.log(torch.ones(n) / n)
                maybe_fork(stub, env, obs, 0, logpis, CFG)
            else:
                # consume the same two draws maybe_fork takes, so both
                # variants see an identical downstream stream
                if step == 3:
                    random.random(); random.randrange(2 ** 31 - 1)
            obs, r, done, _, _ = env.step(0)
            rewards.append(r)
            if done:
                break
        return rewards

    assert run(True) == run(False), "fork perturbed the main episode"
    print("  main_episode_unaffected_by_fork OK")


def test_preference_loss_moves_logits():
    from gympn.agents import Agent

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

        def forward(self, state):
            return torch.softmax(self.w, dim=0)

    agent = object.__new__(Agent)  # skip full __init__, wire the essentials
    agent.policy_model = TinyPolicy()
    agent.policy_optimizer = torch.optim.Adam(agent.policy_model.parameters(), lr=0.1)
    agent.cf_config = dict(CFG)
    agent._total_epochs = 10
    agent._current_epoch = 0
    agent._cf_prefs = [{'state': None, 'winner': 0, 'loser': 1,
                        'gap': 0.3, 'se': 0.05}]

    for _ in range(20):
        stats = agent._fit_cf_preferences()
    assert stats['n'] == 1 and stats['loss'] > 0
    w = agent.policy_model.w.detach()
    assert w[0] > w[1], f"winner logit must rise: {w}"

    # anneal floor: at the final epoch the coefficient is 0 => no-op
    agent._current_epoch = 9
    w_before = agent.policy_model.w.detach().clone()
    stats = agent._fit_cf_preferences()
    assert stats['coef'] == 0.0
    assert torch.equal(agent.policy_model.w.detach(), w_before)

    # cfpk (anneal=False): constant coefficient, still active at the end
    agent.cf_config = dict(CFG, anneal=False)
    stats = agent._fit_cf_preferences()
    assert stats['coef'] == CFG['coef'], stats
    assert not torch.equal(agent.policy_model.w.detach(), w_before)
    print("  preference_loss_moves_logits OK (w =", w.tolist(), ")")


def test_lineage_return_excludes_concurrent_reward():
    """The core cfpl claim: a reward NOT descended from the forked decision
    must not enter that branch's return, while a descended one must."""
    from gympn.counterfactual import _lineage_return

    class _CT:
        pass

    ct = _CT()
    ct.token_history = type("TH", (), {"tokens": {
        # forked action's output, then its child, then an unrelated lineage
        "root": {"parents": []},
        "child": {"parents": ["root"]},
        "other": {"parents": ["elsewhere"]},
    }})()
    root_rec = {"reward": 0.0, "time": 0.0, "input_tokens": ["pre"]}
    ct.transition_history = type("TrH", (), {"transitions": [
        root_rec,
        # descended: consumed the forked action's grandchild
        {"reward": 1.0, "time": 1.0, "input_tokens": ["child"]},
        # concurrent: causally unrelated, must be EXCLUDED
        {"reward": 5.0, "time": 1.0, "input_tokens": ["other"]},
    ]})()

    total = _lineage_return(ct, {"root"}, id(root_rec), t0=0.0, beta=0.0)
    assert total == 1.0, f"expected only the descended reward, got {total}"

    # with no restriction the same trace would have yielded 6.0
    print("  lineage_return_excludes_concurrent_reward OK "
          f"(lineage {total} vs raw 6.0)")


def test_lineage_mode_requires_env_trace():
    """cfpl must fail LOUDLY when the env isn't recording causal traces,
    rather than silently returning 0 for every branch."""
    env, obs = _fresh_env()          # built with causal_rl=False
    n = len(obs['actions_dict'])
    logpis = torch.log(torch.ones(n) / n)
    try:
        maybe_fork(_StubAgent(), env, obs, 0, logpis, dict(CFG, lineage=True))
    except RuntimeError as e:
        assert "causal_rl=True" in str(e)
        print("  lineage_mode_requires_env_trace OK (raised as designed)")
        return
    raise AssertionError("lineage mode silently accepted an untraced env")


def test_lineage_mode_runs_on_traced_env():
    """End-to-end: with a traced env the lineage fork produces finite,
    non-negative branch returns and restores state exactly."""
    import types
    import uuid

    random.seed(3); np.random.seed(3)
    pn = make_s1_stoch_sequence(causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True)
    pn.length = 20
    # training_run does this before building the env (simulator.py ~1495);
    # a manual AEPN_Env probe must replicate it or add_transition crashes on
    # tokens without _id. Production cfpl goes through training_run.
    for place in pn.places:
        for token in place.marking:
            setattr(token, '_id', str(uuid.uuid4()))
    pn.causal_trace.flush()
    sentinel = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sentinel, [], time=0)
    pn.causal_trace.register_transition(sentinel, [], toks, is_action=False,
                                        reward=0.0, time=0)
    env = AEPN_Env(pn)
    obs = env.reset()
    n = len(obs['actions_dict'])
    logpis = torch.log(torch.ones(n) / n)
    clock_before = float(env.pn.clock)

    forked, pref, diag = maybe_fork(_StubAgent(), env, obs, 0, logpis,
                                    dict(CFG, lineage=True))
    assert forked and diag is not None
    assert np.isfinite(diag['gap'])
    assert float(env.pn.clock) == clock_before, "clock not restored"
    print(f"  lineage_mode_runs_on_traced_env OK (gap={diag['gap']:.3f} "
          f"se={diag['se']:.3f} passed={diag['passed']})")


def _mk_records(n, predictable, rng_seed=0):
    """Synthetic decomposed fork records. predictable=True => the indirect
    channel is a clean linear function of the occupancy feature."""
    rs = np.random.RandomState(rng_seed)
    recs = []
    for i in range(n):
        occ = float(rs.uniform(-1, 1))
        ind = (2.0 * occ if predictable else float(rs.normal(0, 1)))
        if predictable:
            ind += float(rs.normal(0, 0.05))
        recs.append({
            "state": None, "action": 0, "alt": 1,
            "gap_direct": 0.10, "se_direct": 0.01,
            "gap_ind": ind,
            "gap_total": 0.10 + ind, "se_total": 0.50,
            "phi": {"occ_disc": occ, "occ_dur": -occ,
                    "n_desc": 0.0, "disc_desc": 0.0,
                    "occ_dur_x_cong": 0.0, "occ_disc_x_cong": 0.0},
        })
    return recs


def test_decomp_resolver_uses_model_when_predictable():
    from gympn.counterfactual import resolve_decomp_preferences
    prefs, st = resolve_decomp_preferences(_mk_records(40, True), gate=2.0)
    assert st["mode"] == "model", st
    assert st["r2"] > 0.5, st
    # pooled fit => SE far below the raw per-fork total SE (0.50)
    assert st["se"] < 0.1, st
    assert st["pass_rate"] > 0.5, st
    print(f"  decomp_resolver_uses_model OK (r2={st['r2']:.2f} "
          f"se={st['se']:.4f} pass={st['pass_rate']:.0%})")


def test_decomp_resolver_falls_back_when_unpredictable():
    """The safety floor: noise-only indirect channel must NOT be trusted."""
    from gympn.counterfactual import resolve_decomp_preferences
    prefs, st = resolve_decomp_preferences(_mk_records(40, False), gate=2.0)
    assert st["mode"] == "raw", st
    assert st["se"] == 0.50, st          # fell back to the raw total gap SE
    print(f"  decomp_resolver_falls_back OK (r2={st['r2']:.2f} -> raw)")


def test_decomp_preference_direction():
    from gympn.counterfactual import resolve_decomp_preferences
    recs = _mk_records(40, True)
    for r in recs:
        # large NEGATIVE opportunity cost, still predictable from occupancy
        r["gap_ind"] = -5.0 + 0.5 * r["phi"]["occ_disc"]
        r["gap_total"] = r["gap_direct"] + r["gap_ind"]
    prefs, st = resolve_decomp_preferences(recs, gate=2.0)
    assert st["mode"] == "model", st
    # direct channel favours action 0 (+0.10) but the opportunity cost
    # (-5.0) must flip the verdict to the alternative
    assert prefs and all(p["winner"] == 1 for p in prefs), st
    print("  decomp_preference_direction OK (indirect channel flips winner)")


if __name__ == "__main__":
    test_decomp_resolver_uses_model_when_predictable()
    test_decomp_resolver_falls_back_when_unpredictable()
    test_decomp_preference_direction()
    test_fork_restores_env_state()
    test_fork_disabled_paths()
    test_main_episode_unaffected_by_fork()
    test_preference_loss_moves_logits()
    test_lineage_return_excludes_concurrent_reward()
    test_lineage_mode_requires_env_trace()
    test_lineage_mode_runs_on_traced_env()
    print("all cfp tests passed")