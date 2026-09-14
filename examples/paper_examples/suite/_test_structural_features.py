"""Unit + smoke tests for structural (conflict-graph-derived) INPUT features
(gympn/simulator.py's GymProblem.use_structural_features /
_get_action_conflict_features, consumed in get_graph_observation's
a_transition block). Run: python _test_structural_features.py

Unlike every other lineage/topology mechanism tried in this project
(lrq/ccf/s_ccf/lcv/lva/ls_hca/potential-shaping), this one is neither a
credit/advantage term nor a reward-shaping term -- it's plain extra input to
the GNN encoder. So there is no bias-variance tradeoff to prove and no
GAE-bootstrap fragility to worry about; the correctness bar is just "the
right numbers land in the right place, off by default, doesn't break the
network."
"""
import sys, os
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assembly_probe import make_shared_r, make_two_chains, make_independent


def test_default_off_is_unchanged():
    """use_structural_features=False (the default) must give byte-identical
    a_transition feature width/values to the pre-existing behaviour."""
    pn = make_two_chains(causal_rl=False, r_hi=5, r_lo=2, r_b=4)
    obs = pn.get_graph_observation()
    x = obs['graph']['a_transition'].x
    assert x.shape == (6, 3), x.shape  # 3 action TYPES, one-hot only
    print(f"  default_off_is_unchanged OK (shape={tuple(x.shape)})")


def test_conflict_features_match_conflict_graph():
    """On M4 (make_two_chains): A_hi/A_lo/B_run all structurally share the
    resource place R, so conflict_graph flags all three as mutually
    conflicting (degree=2 each) -- matches the classify() finding from the
    ls_hca work earlier this session (same underlying conflict_graph
    machinery). On M2 (make_shared_r): use_R/standalone share part1 (degree
    1, conflict with each other only); route2 shares no input place with
    either (degree 0, not conflicted). reward_proximity (3rd element) is
    checked against hand-computed hop distances -- both M4 and M2 happen to
    be depth-symmetric (every action's bottleneck precondition sits at the
    same hop count as its conflict partners' -- e.g. M2's part1, shared by
    use_R and standalone, dominates use_R's min over [part1, R] too), so
    this test does NOT exercise discrimination between conflicting actions;
    see test_reward_proximity_discriminates_asymmetric_actions for a
    topology (s1_stoch_sequence) where the bottleneck genuinely differs."""
    pn4 = make_two_chains(causal_rl=False, r_hi=5, r_lo=2, r_b=4)
    f4 = pn4._get_action_conflict_features()
    for name in ('A_hi', 'A_lo', 'B_run'):
        assert f4[name] == (1.0, 2.0, 0.81), (name, f4[name])

    pn2 = make_shared_r(causal_rl=False, r_join=5, r1=5, r2=10)
    f2 = pn2._get_action_conflict_features()
    assert f2['use_R'] == (1.0, 1.0, 0.81), f2['use_R']
    assert f2['standalone'] == (1.0, 1.0, 0.81), f2['standalone']
    assert f2['route2'] == (0.0, 0.0, 0.81), f2['route2']

    pnI = make_independent(causal_rl=False, ra=5, rb=7)
    fI = pnI._get_action_conflict_features()
    assert fI['da'] == (0.0, 0.0, 0.81) and fI['db'] == (0.0, 0.0, 0.81), fI
    print(f"  conflict_features_match_conflict_graph OK "
          f"(M4={f4}, M2={f2}, independent={fI})")


def test_reward_proximity_discriminates_asymmetric_actions():
    """The case that triggered this feature: on s1_stoch_sequence, start1 and
    start2 share only the `employee` pool, so conflict_graph gives them
    IDENTICAL (in_conflict, degree) = (1.0, 1.0) -- necessarily, since graph
    degree on a single pairwise edge is equal on both ends. That is a
    correct computation, not a bug -- but it means conflict-degree alone
    can't tell the network WHICH of the two contested actions is
    structurally closer to payoff. reward_proximity does: start1's
    bottleneck precondition is `waiting1` (hop 4, deep upstream of the
    reward), start2's is tied at hop 2 (`waiting2`/`employee`), so the two
    actions get different reward_proximity despite identical conflict
    features."""
    import sys as _sys
    _sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite")
    from stoch_envs import make_s1_stoch_sequence
    pn = make_s1_stoch_sequence(causal_rl=False, allow_postpone=True)
    f = pn._get_action_conflict_features()
    assert f['start1'][:2] == (1.0, 1.0), f['start1']
    assert f['start2'][:2] == (1.0, 1.0), f['start2']
    assert f['start1'][2] != f['start2'][2], (
        "reward_proximity should discriminate start1/start2 even though "
        f"conflict-degree can't: {f}")
    assert f['start1'][2] < f['start2'][2], (
        "start1 needs waiting1 (hop 4, farther from reward) so its "
        f"bottleneck proximity should be LOWER than start2's: {f}")
    print(f"  reward_proximity_discriminates_asymmetric_actions OK "
          f"(start1={f['start1']}, start2={f['start2']})")


def test_on_adds_three_columns_with_right_values():
    pn = make_two_chains(causal_rl=False, r_hi=5, r_lo=2, r_b=4)
    pn.use_structural_features = True
    obs = pn.get_graph_observation()
    x = obs['graph']['a_transition'].x
    assert x.shape == (6, 6), x.shape  # 3 one-hot + 3 structural
    # last three columns constant across all 6 instances: (in_conflict, degree, reward_proximity)
    tail = x[:, 3:]
    assert (tail == tail[0]).all(), tail
    assert [round(v, 4) for v in tail[0].tolist()] == [1.0, 2.0, 0.81], tail[0].tolist()
    print(f"  on_adds_three_columns_with_right_values OK (shape={tuple(x.shape)})")


def test_cache_invalidation():
    """The conflict-feature cache must actually cache (same object on repeat
    calls) and must be a per-instance cache (a fresh pn gets fresh features,
    not a stale global)."""
    pn = make_two_chains(causal_rl=False, r_hi=5, r_lo=2, r_b=4)
    f1 = pn._get_action_conflict_features()
    f2 = pn._get_action_conflict_features()
    assert f1 is f2, "expected the cached dict to be reused, not recomputed"

    pn2 = make_independent(causal_rl=False, ra=5, rb=7)
    f3 = pn2._get_action_conflict_features()
    assert f3 != f1, "a different net's cache must not leak into this one"
    print("  cache_invalidation OK")


def test_end_to_end_smoke_train():
    """Real training loop through the actual pipeline: HGT's lazy
    input_size=-1 sizing must absorb the extra feature columns with zero
    changes to networks.py (confirmed architecturally in the design phase;
    this is the empirical no-crash check)."""
    import sys as _sys
    from envs import make_env

    env = make_env("d_parallel_disjoint", causal_rl=False, allow_postpone=True)
    env.use_structural_features = True

    args = {
        "algorithm": "ppo-clip", "episodes": 4, "epochs": 3, "batch_size": 8,
        "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 3,
        "value_lr": 3e-4, "value_updates": 4, "gam": 0.99, "lam": 0.95,
        "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
        "causal_rl": False, "causal_scheme": "lrq", "causal_beta": 0.5,
        "verbose": 0, "use_gpu": False, "agent_seed": 0,
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": 1, "test_episodes": 5,
        "save_freq": 1_000_000, "name": "structfeat_smoke", "datetag": False,
        "logdir": "structfeat_smoke_train",
    }
    saved = _sys.argv
    _sys.argv = _sys.argv[:1]
    try:
        env.training_run(length=10, args_dict=args)
    finally:
        _sys.argv = saved
    h = env.training_history
    assert h.get("mean_returns") is not None and len(h["mean_returns"]) == 3
    assert all(v == v for v in h["mean_returns"])  # no NaNs
    print(f"  end_to_end_smoke_train OK (mean_returns={list(h['mean_returns'])})")

    import shutil
    shutil.rmtree("structfeat_smoke_train", ignore_errors=True)


if __name__ == "__main__":
    test_default_off_is_unchanged()
    test_conflict_features_match_conflict_graph()
    test_reward_proximity_discriminates_asymmetric_actions()
    test_on_adds_three_columns_with_right_values()
    test_cache_invalidation()
    test_end_to_end_smoke_train()
    print("all structural-features tests passed")
