"""GymProblem.merge_options: one action node per option (equivalent
bindings merged, the FIFO one kept), with log(multiplicity) as a feature;
pn_actions on options everywhere; the dynamics unchanged."""
import math
import os
import random
import sys
from collections import Counter

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.simulator import binding_option_key, merge_option_bindings  # noqa: E402
from bpic15_env import BPIC15_BUILDERS, bpic15_heuristic  # noqa: E402
from insurer_env import INSURER_BUILDERS  # noqa: E402
from heldout import _index_of  # noqa: E402


def _env(merge, build=lambda: BPIC15_BUILDERS["bpic15"](1, allow_postpone=False), seed=4):
    random.seed(seed); np.random.seed(seed)
    pn = build()
    pn.length = 20
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    pn.merge_options = merge
    env = AEPN_Env(pn)
    return env, env.reset()


def test_merge_option_bindings_fifo_and_counts():
    class T:
        def __init__(self, v, t):
            self.value, self.time = v, t

    class Tr:
        _id = 'do'
    p, q = object(), object()
    B = [([(p, T({'a': 1}, 3.0)), (q, T({'s': 0}, 1.0))], 3.0, Tr),
         ([(p, T({'a': 1}, 1.0)), (q, T({'s': 0}, 2.0))], 2.0, Tr),     # oldest first token: FIFO
         ([(p, T({'a': 2}, 0.0)), (q, T({'s': 0}, 0.0))], 0.0, Tr),
         ([(p, T({'a': 1}, 1.0)), (q, T({'s': 0}, 5.0))], 5.0, Tr)]
    kept, mult = merge_option_bindings(B)
    assert kept == [1, 2] and mult == [3, 1]


def test_one_node_per_option_with_its_multiplicity():
    raw, o_raw = _env(False)
    mer, o_mer = _env(True)
    for _ in range(40):
        B_raw, B_mer = raw.pn.pn_actions, mer.pn.pn_actions
        cnt = Counter(binding_option_key(b) for b in B_raw)
        keys = [binding_option_key(b) for b in B_mer]
        assert len(keys) == len(set(keys)) and set(keys) == set(cnt)
        g = o_mer['graph']
        assert int(g.a_idx.numel()) == len(B_mer)
        assert int(o_raw['graph'].num_nodes) - int(g.num_nodes) == len(B_raw) - len(B_mer)
        col = len(mer.pn.actions)
        for k, n in zip(keys, g.a_idx.tolist()):
            assert math.isclose(float(g.x[n, col]), math.log(cnt[k]), rel_tol=1e-6)
        for b in B_mer:       # the kept binding is the FIFO one of its option
            same = [c for c in B_raw if binding_option_key(c) == binding_option_key(b)]
            assert tuple(t.time for _p, t in b[0]) == min(tuple(t.time for _p, t in c[0]) for c in same)
        if not B_mer:
            break
        k = random.Random(len(B_mer)).randrange(len(B_mer))
        state, nps = random.getstate(), np.random.get_state()
        o_mer, _, d1, _, _ = mer.step(k)
        random.setstate(state); np.random.set_state(nps)
        o_raw, _, d2, _, _ = raw.step([binding_option_key(b) for b in B_raw].index(keys[k]))
        assert d1 == d2
        if d1:
            break


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_heuristic_returns_unchanged(seed):
    """Equivalent bindings lead to the same future: the heuristic (which picks
    by token values) collects the same reward with options merged."""
    out = []
    for merge in (False, True):
        env, _ = _env(merge, seed=seed)
        total = 0.0
        while env.pn.pn_actions:
            B = env.pn.pn_actions
            a = _index_of(B, bpic15_heuristic(env.pn.get_heuristic_observation(), None, B))
            _, r, done, _, _ = env.step(a, build_obs=False)
            total += r
            if done:
                break
        out.append(total)
    assert out[0] == out[1]


def test_hidden_attributes_do_not_split_options():
    """Cases that differ only in a hidden attribute (the insurer's `bad`) are
    one option: the option set must not reveal what the decision maker cannot
    see. Keys, merged pn_actions and multiplicities all ignore it."""
    env, _ = _env(True, build=lambda: INSURER_BUILDERS["insurer"](1))
    pn = env.pn
    assert pn.places[[p._id for p in pn.places].index("cl_waiting_0")].hidden_token_attrs == ('bad',)
    keys = [binding_option_key(b) for b in pn.pn_actions]
    assert len(keys) == len(set(keys))
    assert not any("'bad'" in repr(k) for k in keys)
    # the initial claims are risk 0 (good), 1, 2 (bad), 0: approve risk 0 is one option for both
    raw, _ = _env(False, build=lambda: INSURER_BUILDERS["insurer"](1))
    vis = Counter(binding_option_key(b) for b in raw.pn.pn_actions)
    assert set(vis) == set(keys)


def test_planner_redraw_keeps_the_option():
    """A redraw of the hidden outcomes (insurer_resample_hidden) changes the
    token values that the bindings hold; the planner still plays the same
    option in every rollout."""
    from gympn.policy_iteration import RolloutPlanner, HeuristicBase, option_key
    from insurer_env import insurer_heuristic, insurer_resample_hidden
    env, _ = _env(True, build=lambda: INSURER_BUILDERS["insurer"](1))
    planner = RolloutPlanner(HeuristicBase(insurer_heuristic), rollouts=2, component=True,
                             resample=insurer_resample_hidden, sh_budget=8)
    B = list(env.pn.pn_actions)
    a = planner.choose(env, tag=(0,))
    assert [option_key(b) for b in env.pn.pn_actions] == [option_key(b) for b in B]
    assert 0 <= a < len(B)
