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
                                    collect_decisions, class_nll, distill)
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
