"""The direct flat observation builder (GymProblem._direct_flat_observation)
must give exactly the observation of the HeteroData path it replaces: every
FlatGraph field bitwise, and the same pn_actions. Two copies of an env are
stepped with the same actions and the same random draws, one with
DIRECT_FLAT_OBS off."""
import os
import random
import sys

import numpy as np
import pytest
import torch
from simpn.simulator import SimToken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.simulator import GymProblem  # noqa: E402
from insurer_env import INSURER_BUILDERS  # noqa: E402
from bpic15_env import BPIC15_BUILDERS  # noqa: E402
from bpm_envs import BPM_BUILDERS  # noqa: E402


def _odd_net():
    """Covers what the suite nets do not: a categorical attribute, hidden
    attributes, a hidden place, a guard, a place whose tokens have no
    features, and an action with two output places (only the first gets
    edges) whose first output is often empty."""
    pn = GymProblem(tag='a')
    arrival = pn.add_var("arrival", var_attributes=['id', 'kind', 'secret'], categorical_values={'kind': [0, 1, 2]})
    waiting = pn.add_var("waiting", var_attributes=['id', 'kind', 'secret'], categorical_values={'kind': [0, 1, 2]})
    staff = pn.add_var("staff", var_attributes=['skill'])
    ticket = pn.add_var("ticket", var_attributes=[])
    log = pn.add_var("log", var_attributes=['n'])
    busy = pn.add_var("busy", var_attributes=['id', 'kind', 'secret', 'skill'])
    done = pn.add_var("done", var_attributes=['id'])
    arrival.put({'id': 0, 'kind': 0, 'secret': 7})
    waiting.put({'id': 100, 'kind': 2, 'secret': 3})          # an action enabled at the start
    for s in (0, 1, 1):
        staff.put({'skill': s})
    for _ in range(2):
        ticket.put({})
    log.put({'n': 0})

    def arrive(a):
        return [SimToken({'id': a['id'] + 1, 'kind': random.randint(0, 2), 'secret': random.random()},
                         delay=random.expovariate(1.0)),
                SimToken(dict(a))]
    pn.add_event([arrival], [arrival, waiting], arrive, name="arrive")

    def start(c, s, t):
        return [SimToken((c, s), delay=1 + random.randint(0, 2)), SimToken({'n': 1}), SimToken(t)]
    pn.add_action([waiting, staff, ticket], [busy, log, ticket], behavior=start, name="start",
                  guard=lambda c, s, t: c['kind'] != 2 or s['skill'] == 1)

    def finish(b):
        return [SimToken({'id': b[0]['id']}), SimToken({'skill': b[1]['skill']})]
    pn.add_event([busy], [done, staff], finish, name="finish", reward_function=lambda b: 1)

    pn.add_action([done], [log], behavior=lambda d: [SimToken({'n': 2})], name="clear")
    pn.set_unobservable(simvars=['log'], token_attrs={'arrival': ['secret'], 'waiting': ['secret']})
    return pn


SCEN = {
    "odd": (_odd_net, dict(global_view='edges')),
    "odd_local_pp": (lambda: _with_postpone(_odd_net()), dict(global_view=False)),
    "bpic15_marking": (lambda: BPIC15_BUILDERS["bpic15"](1, allow_postpone=False),
                       dict(global_view='edges', marking_certificate=True)),
    "insurer2_comp_pp": (lambda: INSURER_BUILDERS["insurer"](2, allow_postpone=True),
                         dict(global_view='edges', postpone_scope='component')),
    "alloc_marking": (lambda: INSURER_BUILDERS["insurer_alloc"](1, allow_postpone=False),
                      dict(global_view='edges', marking_certificate=True)),
    "next_activity_pp": (lambda: BPM_BUILDERS["next_activity"](1, allow_postpone=True),
                         dict(global_view='edges')),
}


def _with_postpone(pn):
    pn.allow_postpone = True
    return pn


def _env(build, attrs, direct, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = build()
    pn.length = 20
    pn.flat_obs = True
    for k, v in attrs.items():
        setattr(pn, k, v)
    pn.DIRECT_FLAT_OBS = direct
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, (obs[0] if isinstance(obs, tuple) else obs)


def _actions(B):
    out = []
    for b in B:
        if isinstance(b[0], list) and b[0] == ['postpone']:
            out.append(('postpone', b[1], None if b[2] is None else b[2].comp))
        else:
            out.append((tuple((p._id, repr(t.value), t.time, type(t)) for p, t in b[0]), b[1], b[2]._id))
    return out


def _assert_same(g, h):
    assert list(g.keys()) == list(h.keys())
    for k in g.keys():
        x, y = g[k], h[k]
        if torch.is_tensor(x):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), k
        else:
            assert x == y, k


@pytest.mark.parametrize("name", list(SCEN))
@pytest.mark.parametrize("seed", [3, 8])
def test_direct_flat_observation_is_the_heterodata_one(name, seed):
    build, attrs = SCEN[name]
    ref, o_ref = _env(build, attrs, False, seed)
    new, o_new = _env(build, attrs, True, seed)
    rng = random.Random(seed)
    used = False
    for _ in range(80):
        _assert_same(o_ref['graph'], o_new['graph'])
        assert _actions(ref.pn.pn_actions) == _actions(new.pn.pn_actions)
        B = ref.pn.pn_actions
        if not B:
            break
        a = rng.randrange(len(B))
        state, np_state = random.getstate(), np.random.get_state()
        o_ref, _, d1, _, _ = ref.step(a)
        random.setstate(state); np.random.set_state(np_state)
        o_new, _, d2, _, _ = new.step(a)
        used = used or new.pn.expanded_pn is None
        assert d1 == d2
        if d1:
            break
    assert used, "the direct builder was never used"
