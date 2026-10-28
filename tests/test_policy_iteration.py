"""gympn.policy_iteration: the rollout planner, the recorded decisions and the
distillation loss."""
import os
import random
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.policy_iteration import (HeuristicBase, NetworkBase, RolloutPlanner, option_key,  # noqa: E402
                                    collect_decisions, class_nll, distill, _collect_one, episode_components)
from gympn.train import make_parser, make_policy_network  # noqa: E402
from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS  # noqa: E402


def _env(length=6, regions=1):
    pn = INSURER_BUILDERS["insurer"](regions, allow_postpone=False)
    pn.length = length
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    return AEPN_Env(pn)


def _planner(env, rollouts=2):
    return RolloutPlanner(HeuristicBase(INSURER_HEURISTICS["insurer"]), rollouts=rollouts, seed=1)


def _policy(env):
    args = make_parser().parse_args([])
    args.policy_kwargs = {"encoder": "aepn", "encoder_kwargs": {"global_attn": True, "count_aggr": True}}
    args.flat_obs, args.global_view = True, 'edges'
    torch.manual_seed(0)
    return make_policy_network(args, metadata=env.pn.make_metadata())


def _reset(env, seed=3):
    random.seed(seed); np.random.seed(seed)
    return env.reset()


def test_option_key_groups_equivalent_bindings():
    env = _env()
    _reset(env)
    B = env.pn.pn_actions
    keys = [option_key(b) for b in B]
    assert len(set(keys)) <= len(keys)
    for b, k in zip(B, keys):           # equal key -> same transition and token values
        same = [c for c, kc in zip(B, keys) if kc == k]
        assert all(c[2]._id == b[2]._id for c in same)


def test_planner_leaves_the_episode_where_it_was():
    """choose() plays rollouts from a snapshot and must restore the state and
    the episode's random streams; it is deterministic for a given tag."""
    env = _env()
    _reset(env)
    for _ in range(5):                    # move into the episode
        env.step(0)
    planner = _planner(env)
    before = ([option_key(b) for b in env.pn.pn_actions], env.pn.clock, random.getstate(),
              np.random.get_state()[1].copy())
    a1 = planner.choose(env, tag=("t", 1))
    after = ([option_key(b) for b in env.pn.pn_actions], env.pn.clock, random.getstate(),
             np.random.get_state()[1].copy())
    assert before[0] == after[0] and before[1] == after[1] and before[2] == after[2]
    assert np.array_equal(before[3], after[3])
    assert planner.choose(env, tag=("t", 1)) == a1
    assert 0 <= a1 < len(env.pn.pn_actions)
    if planner.last["scores"] is not None:            # several options in the component
        part = env.pn.net_partition()
        comp = lambda i: part.get(env.pn.pn_actions[i][2]._id)
        assert all(comp(o) == comp(planner.last["base"]) for o in planner.last["reps"])


def _make_env():
    return _env(length=4)


def _make_planner(env):
    return _planner(env, rollouts=1)


def test_collected_decisions_mark_the_chosen_class():
    graphs = collect_decisions(_make_env, _make_planner, [11, 12], workers=1)
    assert graphs
    for g in graphs:
        assert g.t_mask_a.numel() == g.a_idx.numel() and g.t_mask_p.numel() == g.p_idx.numel()
        assert int(g.t_mask_a.sum() + g.t_mask_p.sum()) >= 1


def test_class_nll_is_minus_log_of_the_class_probability():
    graphs = collect_decisions(_make_env, _make_planner, [13], workers=1)[:6]
    env = _make_env()
    pol = _policy(env)
    b = Batch.from_data_list(graphs)
    with torch.no_grad():
        pol({'graph': b})
        got = float(class_nll(pol, b))
        ref = []
        for g in graphs:
            p = pol({'graph': g}).reshape(-1)
            m = torch.cat([g.t_mask_a, g.t_mask_p])
            ref.append(-float(torch.log(p[m].sum())))
    assert got == pytest.approx(float(np.mean(ref)), abs=1e-5)


def test_distill_fits_the_planner():
    graphs = collect_decisions(_make_env, _make_planner, [14, 15], workers=1)
    env = _make_env()
    pol = _policy(env)
    hist = distill(pol, graphs, epochs=8, lr=3e-3, batch_size=16, seed=0)
    assert hist["loss"][-1] < hist["loss"][0]
    # the distilled network can serve as the next base policy
    _reset(env)
    a = NetworkBase(pol)(env)
    assert 0 <= a < len(env.pn.pn_actions)


def test_soft_targets_and_loss():
    """t_q is softmax(scores/tau) over the compared classes (one-hot without
    scores or with tau=0), and soft_class_nll is its cross-entropy with the
    network's class probabilities, decision by decision."""
    from gympn.policy_iteration import soft_targets, soft_class_nll
    graphs = collect_decisions(_make_env, _make_planner2, [16], workers=1)[:8]
    assert any(not torch.isnan(g.t_scores).any() for g in graphs)
    soft_targets(graphs, 0.5)
    for g in graphs:
        assert torch.isclose(g.t_q.sum(), torch.tensor(1.0))
        if torch.isnan(g.t_scores).any():
            assert g.t_q[int(g.t_chosen)] == 1.0
        else:
            assert torch.allclose(g.t_q, torch.softmax(g.t_scores / 0.5, 0))
    env = _make_env()
    pol = _policy(env)
    b = Batch.from_data_list(graphs)
    with torch.no_grad():
        pol({'graph': b})
        got = float(soft_class_nll(pol, b))
        ref = []
        for g in graphs:
            p = pol({'graph': g}).reshape(-1)
            cls = torch.cat([g.t_cls_a, g.t_cls_p])
            P = torch.stack([p[cls == c].sum() for c in range(int(g.t_ncls))])
            ref.append(-float((g.t_q * P.clamp_min(1e-8).log()).sum()))
    assert got == pytest.approx(float(np.mean(ref)), abs=1e-5)
    soft_targets(graphs, 0.0)                 # hard targets: the class NLL
    b = Batch.from_data_list(graphs)          # (collate again: the batch holds copies of the targets)
    with torch.no_grad():
        assert float(soft_class_nll(pol, b)) == pytest.approx(float(class_nll(pol, b)), abs=1e-5)


def _make_planner2(env):
    return _planner(env, rollouts=2)


def test_distill_selects_the_best_evaluation():
    graphs = collect_decisions(_make_env, _make_planner, [17], workers=1)
    env = _make_env()
    pol = _policy(env)
    scores = iter([0.1, 0.9, 0.3])
    hist = distill(pol, graphs, epochs=3, batch_size=16, evaluate=lambda p: {"s": next(scores)},
                   eval_every=1, select="s")
    assert hist["selected_epoch"] == 2
    sd = pol.state_dict()
    assert all(torch.equal(sd[k], hist["checkpoints"][2][k]) for k in sd)


def test_lockstep_rollouts_equal_sequential_ones():
    """With a network base policy the planner plays a decision's rollouts
    side by side; the option scores must be those of one rollout at a time."""
    env = _env(length=5)
    pol = _policy(env)
    _reset(env, seed=21)
    with torch.no_grad():
        pol(env.pn.get_graph_observation())
    for _ in range(3):
        env.step(0)
    out = []
    for lockstep in (False, True):
        planner = RolloutPlanner(NetworkBase(pol), rollouts=3, seed=4, lockstep=lockstep)
        a = planner.choose(env, tag=("x", 0))
        out.append((a, planner.last["scores"]))
    assert out[0][1] is not None and len(out[0][1]) > 1     # several options were compared
    assert out[0][0] == out[1][0]
    assert out[0][1] == out[1][1]


def _two_region_env(seed=8, **flags):
    pn = INSURER_BUILDERS["insurer"](2, allow_postpone=False)
    pn.length = 8
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    for k, v in flags.items():
        setattr(pn, k, v)
    env = AEPN_Env(pn)
    _reset(env, seed)
    for _ in range(4):
        env.step(0)
    return env


def test_restricted_simulation_fires_and_shows_only_the_component():
    env = _two_region_env()
    pn = env.pn
    part = pn.net_partition()
    c = part.get(pn.pn_actions[0][2]._id)
    members = frozenset(pn.net_component_members(c))
    assert len({v for v in part.values() if v is not None}) > 1          # several components
    snap = env.get_state()
    pn.sim_only = members
    try:
        env.set_state(snap)
        for _ in range(20):
            assert all(b[2]._id in members for b in pn.pn_actions)
            g = pn.get_graph_observation()['graph']
            names = pn._flat_meta[0]
            kept = {names[i] for i in g.ntype.tolist()}
            places = pn._sim_sets()[2]
            assert kept <= places | {'a_transition', 'e_transition'}
            if not pn.pn_actions:
                break
            _, _, done, _, info = env.step(0)
            assert all(k in members for k in info.get('comp_reward', {}))
            if done:
                break
    finally:
        pn.sim_only = None
        env.set_state(snap)


@pytest.mark.parametrize("flags", [{}, {"clock_node": True}, {"clock_node": True, "merge_options": True}],
                         ids=["plain", "clock_node", "clock_node+merge"])
def test_component_logits_do_not_depend_on_the_rest_of_the_net(flags):
    """Theorem 1 (C2), used by restricted rollouts: the actor's distribution
    over a component's actions, renormalised, is the same on the full graph
    and on the component's subgraph (also with the clock node, which every
    component reads, and with options merged)."""
    env = _two_region_env(seed=9, **flags)
    pn = env.pn
    pol = _policy(env).eval()
    full = pn.get_graph_observation()
    B_full = list(pn.pn_actions)
    with torch.no_grad():
        p_full = pol(full).reshape(-1)
    part = pn.net_partition()
    checked = 0
    for c in {part.get(b[2]._id) for b in B_full}:
        members = frozenset(pn.net_component_members(c))
        pn.sim_only = members
        try:
            sub = pn.get_graph_observation()
            B_sub = list(pn.pn_actions)
            with torch.no_grad():
                p_sub = pol(sub).reshape(-1)
        finally:
            pn.sim_only = None
            pn.get_graph_observation()
        idx = [i for i, b in enumerate(B_full) if b[2]._id in members]
        assert [option_key(B_full[i]) for i in idx] == [option_key(b) for b in B_sub]
        q = p_full[idx] / p_full[idx].sum()
        assert torch.allclose(q, p_sub, atol=1e-5)
        checked += 1
    assert checked > 1


def _run(env, actions_fn, steps=40):
    out = []
    for _ in range(steps):
        if not env.pn.pn_actions:
            break
        a = actions_fn(env)
        _, r, done, _, _ = env.step(a, build_obs=False)
        out.append((round(env.pn.clock, 9), r, len(env.pn.pn_actions)))
        if done:
            break
    return out


def test_keyed_streams_do_not_depend_on_the_global_stream():
    """With stream_seed set, a trajectory is a function of the state, the
    actions and the stream seed only: perturbing the global random streams
    changes nothing, firing leaves them as they were, and the firing counts
    are restored with the state."""
    env = _env(length=6)
    _reset(env, 30)
    snap = env.get_state()
    runs = []
    for g in (1, 2):
        env.set_state(snap)
        env.pn.stream_seed = 777
        random.seed(g); np.random.seed(g)
        before = random.getstate()
        runs.append(_run(env, lambda e: 0))
        assert random.getstate() == before            # firings did not touch the global stream
    assert runs[0] == runs[1]
    env.set_state(snap)
    env.pn.stream_seed = 778
    assert _run(env, lambda e: 0) != runs[0]          # another stream seed, another trajectory
    env.set_state(snap)
    assert env.pn.stream_seed is None and env.pn.__dict__.get('_fires', {}) == snap['pn_state']['fires']


def test_keyed_planner_matches_sequential_with_network_base():
    env = _env(length=5)
    pol = _policy(env)
    _reset(env, seed=21)
    with torch.no_grad():
        pol(env.pn.get_graph_observation())
    for _ in range(3):
        env.step(0)
    out = []
    for lockstep in (False, True):
        planner = RolloutPlanner(NetworkBase(pol), rollouts=3, seed=4, lockstep=lockstep, keyed=True)
        out.append((planner.choose(env, tag=("k", 0)), planner.last["scores"]))
    assert out[0][1] is not None and out[0] == out[1]


@pytest.mark.parametrize("base_kind", ["heuristic", "network"])
def test_coalescing_is_exact(base_kind):
    """Stopping a rollout where it re-merges with the base choice's rollout
    (same state, keyed streams, deterministic base) gives the same scores."""
    env = _env(length=6, regions=1)
    _reset(env, seed=40)
    if base_kind == "network":
        pol = _policy(env)
        with torch.no_grad():
            pol(env.pn.get_graph_observation())
    for _ in range(3):
        env.step(0)
    found = 0
    for step in range(12):
        if not env.pn.pn_actions:
            break
        out = []
        for coalesce in (False, True):
            base = (HeuristicBase(INSURER_HEURISTICS["insurer"]) if base_kind == "heuristic"
                    else NetworkBase(pol))
            pl = RolloutPlanner(base, rollouts=3, seed=5, keyed=True, coalesce=coalesce)
            a = pl.choose(env, tag=("c", step))
            out.append((a, pl.last["scores"], pl.steps[1]))
        if out[0][1] is not None:
            found += 1
            assert out[0][0] == out[1][0]
            assert np.allclose(out[0][1], out[1][1], atol=1e-9)
            assert out[0][2] == 0
        env.step(out[0][0])
    assert found > 0


def test_held_out_test_keeps_only_tested_departures():
    """test/min_t: a departure survives only with its paired test recorded and
    passed; an unreachable threshold leaves the base's choice everywhere, and
    the test never adds a departure the plain planner would not make."""
    env = _env()
    _reset(env)
    plain = _planner(env, rollouts=4)
    tested = RolloutPlanner(HeuristicBase(INSURER_HEURISTICS["insurer"]), rollouts=4, seed=1, test=6, min_t=1.0)
    strict = RolloutPlanner(HeuristicBase(INSURER_HEURISTICS["insurer"]), rollouts=4, seed=1, test=6,
                            min_t=float("inf"))
    for step in range(30):
        if not env.pn.pn_actions:
            break
        a = plain.choose(env, tag=(step,))
        b = tested.choose(env, tag=(step,))
        assert strict.choose(env, tag=(step,)) == strict.last["base"]
        assert b in (a, tested.last["base"])
        if b != tested.last["base"]:
            assert tested.last["test_t"] >= 1.0 and tested.last["test_adv"] > 0
        env.step(a)


def test_advantage_loss_is_minus_the_expected_advantage():
    """advantage_loss = -sum_c P(c) (Q(c) - Q(base)), by hand on recorded decisions."""
    from gympn.policy_iteration import advantage_loss
    env = _env()
    graphs = [g for g in collect_decisions(_make_env, lambda e: _planner(e, rollouts=2), [11, 12], workers=1)
              if int(g.t_ncls) > 1 and not torch.isnan(g.t_scores).any() and int(g.t_base) >= 0][:6]
    assert graphs
    pol = _policy(env)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        got = float(advantage_loss(pol, batch))
        want = 0.0
        for g in graphs:
            p = pol({'graph': Batch.from_data_list([g])}).reshape(-1)
            cls = torch.cat([g.t_cls_a, g.t_cls_p])
            P = torch.zeros(int(g.t_ncls)).index_add(0, cls[cls >= 0], p[cls >= 0])
            want -= float((P * (g.t_scores - g.t_scores[int(g.t_base)])).sum())
    assert abs(got - want / len(graphs)) < 1e-5


def test_sequential_halving_spends_its_budget_and_leaves_the_episode():
    """sh_budget (DCL's action selection): the choice is one of the
    component's options, at most sh_budget rollouts per option are spent,
    the survivor of the last round got the most rollouts, the episode is
    left as it was, and the choice is deterministic for a tag."""
    env = _env()
    _reset(env)
    for _ in range(3):
        env.step(0)
    pl = RolloutPlanner(HeuristicBase(INSURER_HEURISTICS["insurer"]), seed=1, sh_budget=8)
    checked = 0
    for step in range(25):
        if not env.pn.pn_actions:
            break
        before = ([option_key(b) for b in env.pn.pn_actions], env.pn.clock, random.getstate())
        a = pl.choose(env, tag=(step,))
        assert ([option_key(b) for b in env.pn.pn_actions], env.pn.clock, random.getstate()) == before
        assert pl.choose(env, tag=(step,)) == a
        last = pl.last
        if last["scores"] is not None:
            n = len(last["reps"])
            assert a in last["reps"]
            assert sum(last["counts"]) <= 8 * n
            assert last["counts"][last["reps"].index(a)] == max(last["counts"])
            checked += 1
        env.step(a)
    assert checked > 0


def test_racing_stops_with_a_paired_survivor_set():
    """race_eps: the choice and every survivor are options of the
    component, the survivors got the same (paired) number of rollouts and at
    most race_cap, eliminated options got fewer, the episode is left as it
    was, and the choice is deterministic for a tag."""
    env = _env()
    _reset(env)
    for _ in range(3):
        env.step(0)
    pl = RolloutPlanner(HeuristicBase(INSURER_HEURISTICS["insurer"]), seed=1, race_eps=0.25,
                        race_step=2, race_min=4, race_cap=16)
    checked = 0
    for step in range(25):
        if not env.pn.pn_actions:
            break
        before = ([option_key(b) for b in env.pn.pn_actions], env.pn.clock, random.getstate())
        a = pl.choose(env, tag=(step,))
        assert ([option_key(b) for b in env.pn.pn_actions], env.pn.clock, random.getstate()) == before
        assert pl.choose(env, tag=(step,)) == a
        last = pl.last
        if last["scores"] is not None:
            reps, cnt, sv = last["reps"], last["counts"], last["set"]
            assert a in sv and set(sv) <= set(reps)
            n = {cnt[reps.index(o)] for o in sv}
            assert len(n) == 1 and max(n) <= 16
            assert all(cnt[reps.index(o)] <= max(n) for o in reps)
            checked += 1
        env.step(a)
    assert checked > 0


def test_set_class_nll_is_minus_log_of_the_set_probability():
    from gympn.policy_iteration import set_class_nll, soft_targets
    env = _env()
    graphs = [g for g in collect_decisions(_make_env, _make_planner, [11, 12], workers=1) if int(g.t_ncls) > 1][:6]
    assert graphs
    pol = _policy(env)
    soft_targets(graphs, 0.0)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        assert torch.allclose(set_class_nll(pol, batch), class_nll(pol, batch), atol=1e-6)
        for g in graphs:                        # widen every set to two classes
            g.t_set = torch.zeros(int(g.t_ncls), dtype=torch.bool)
            g.t_set[int(g.t_chosen)] = True
            g.t_set[(int(g.t_chosen) + 1) % int(g.t_ncls)] = True
        want = 0.0
        for g in graphs:
            p = pol({'graph': Batch.from_data_list([g])}).reshape(-1)
            cls = torch.cat([g.t_cls_a, g.t_cls_p])
            P = torch.zeros(int(g.t_ncls)).index_add(0, cls[cls >= 0], p[cls >= 0])
            want -= float(torch.log(P[g.t_set].sum()))
        got = float(set_class_nll(pol, Batch.from_data_list(graphs)))
    assert abs(got - want / len(graphs)) < 1e-5


def _two_region_make_env():
    return _two_region_env_fresh()


def _two_region_env_fresh():
    pn = INSURER_BUILDERS["insurer"](2, allow_postpone=False)
    pn.length = 6
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    pn.merge_options, pn.clock_node = True, True
    return AEPN_Env(pn)


def _two_region_planner(env):
    return RolloutPlanner(HeuristicBase(INSURER_HEURISTICS["insurer"]), rollouts=2, seed=1)


def test_component_factored_collection_stays_in_its_component():
    """collect_component_decisions: every recorded decision of a
    component-day offers only that component's bindings, its graph shows
    only that component's action nodes (one per offered binding), and the
    planner's rollouts leave the episode's restriction in place."""
    from gympn.policy_iteration import episode_components, collect_component_decisions
    comps = episode_components(_two_region_make_env, seed=3)
    assert len(comps) >= 2
    for k, members in enumerate(comps):
        env = _two_region_make_env()
        env.pn.sim_only = env.frozen_pn.sim_only = members
        pl = _two_region_planner(env)
        random.seed(3); np.random.seed(3)
        obs = env.reset()
        steps = 0
        while env.pn.pn_actions and steps < 25:
            B = env.pn.pn_actions
            assert all(b[2]._id in members for b in B)
            assert int(obs['graph'].a_idx.numel()) == len(B)
            a = pl.choose(env, tag=(k, steps))
            assert env.pn.sim_only == members
            obs, _, done, _, _ = env.step(a)
            steps += 1
            if done:
                break
        assert steps > 0
    graphs = collect_component_decisions(_two_region_make_env, _two_region_planner,
                                         [(3, comps[0]), (4, comps[1])], workers=1)
    assert graphs and all(hasattr(g, 't_cls_a') for g in graphs)


def test_component_factored_collection_on_bpic15_shows_only_the_component():
    """On bpic15 (string-valued backlog tokens in the other components'
    unfired start-of-day places) the restricted run's observations, the first
    one included, must come from the direct builder: only the component's
    action nodes, one per offered binding."""
    import bpic15_env
    from gympn.policy_iteration import episode_components, _collect_one

    def make():
        pn = bpic15_env.BPIC15_BUILDERS["bpic15"](1, allow_postpone=False)
        pn.length = 3
        pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
        pn.merge_options, pn.clock_node = True, True
        return AEPN_Env(pn)

    def planner(env):
        return RolloutPlanner(HeuristicBase(bpic15_env.bpic15_heuristic), rollouts=1, seed=1)
    comps = episode_components(make, seed=5)
    # every transition belongs to a component, start-of-day events included
    pn = make().frozen_pn
    assert set().union(*comps) == {t._id for t in list(pn.actions) + list(pn.events)}
    assert all(any(t.startswith("open_") for t in m) for m in comps)
    graphs, _, _ = _collect_one((make, planner, 5, comps[-1]))
    assert graphs
    for g in graphs:
        assert int(g.a_idx.numel()) == int(g.t_cls_a.numel())


def test_renormalized_loss_ignores_other_components():
    """soft_class_nll(renorm=True) depends only on how the compared classes
    share their probability: scaling them all by the same factor (moving mass
    to the other components' actions) leaves it unchanged."""
    from gympn.policy_iteration import soft_class_nll, soft_targets
    env = _two_region_env(seed=9)
    graphs = [g for g in collect_decisions(lambda: _two_region_env(seed=9), lambda e: _planner(e, rollouts=1),
                                           [3], workers=1) if int(g.t_ncls) > 1][:4]
    assert graphs
    soft_targets(graphs, 0.25)
    batch = Batch.from_data_list(graphs)

    class Fixed(torch.nn.Module):
        def __init__(self, p):
            super().__init__(); self.p = p

        def forward(self, data):
            return self.p
    n = int(batch.a_idx.numel() + batch.p_idx.numel())
    p = torch.rand(n)
    cls = torch.cat([batch.t_cls_a, batch.t_cls_p])
    q = p.clone(); q[cls >= 0] *= 0.3; q[cls < 0] += 1.0
    a = soft_class_nll(Fixed(p), batch, renorm=True)
    b = soft_class_nll(Fixed(q), batch, renorm=True)
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(soft_class_nll(Fixed(p), batch), soft_class_nll(Fixed(q), batch), atol=1e-5)


def _alloc_env(length=4):
    pn = INSURER_BUILDERS["insurer_alloc"](1, allow_postpone=False)
    pn.length = length
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    pn.merge_options = True
    return AEPN_Env(pn)


def test_alloc_heuristic_follows_its_plan_with_merged_options():
    # equal-skill pool members offer one merged option, not one binding each;
    # the heuristic reads its plan step off the marking
    from insurer_env import ALLOC_PLAN
    env = _alloc_env()
    _reset(env)
    base = HeuristicBase(INSURER_HEURISTICS["insurer_alloc"])
    done_plan = []
    while len(done_plan) < len(ALLOC_PLAN):
        b = env.pn.pn_actions[base(env)]
        tr = b[2]._id
        assert tr.startswith('al_')
        done_plan.append((tr.split('_')[1], b[0][0][1].value['skill']))
        env.step(base(env))
    assert done_plan == list(ALLOC_PLAN)


class _OpsFirst:
    """Plays every operation before the last allocation, so that the pool runs
    empty when only some processes have work: the split leaves parts with
    nothing enabled. Records the restricted set of every decision."""
    seen = []

    def __init__(self, env):
        self.base = HeuristicBase(INSURER_HEURISTICS["insurer_alloc"])

    def choose(self, env, tag=()):
        B = env.pn.pn_actions
        ops = [i for i, b in enumerate(B) if not b[2]._id.startswith('al_')]
        a = ops[0] if ops else self.base(env)
        _OpsFirst.seen.append((env.pn.sim_only, {b[2]._id for b in B}))
        self.last = {"base": a}
        return a


def test_dynamic_collection_continues_every_part_alone():
    _OpsFirst.seen = []
    comps = episode_components(_alloc_env)
    assert len(comps) == 1                                  # one static component (the pool couples the region)
    graphs, total, _ = _collect_one((_alloc_env, _OpsFirst, 5, comps[0], True))
    parts = {s for s, _ in _OpsFirst.seen if s is not None and s != comps[0]}
    assert len(parts) == 3                                  # claims, complaints, underwriting, each alone
    for s, names in _OpsFirst.seen:
        if s in parts:
            assert names <= s                               # a part only ever offers its own transitions
    assert len(graphs) == len(_OpsFirst.seen) and total > 0
