"""Tests for net-factored GAE (scheme 'nfgae'; examples/paper_examples/suite/
paper/NFGAE_THEORY.md): the net partition, the per-component reward split,
K=1 exactness against PPO's SMDP-GAE, a hand-checked K=2 recursion, actor
locality (Theorem 1, condition C2) and the critic's component pooling."""
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.data import TrajectoryBuffer  # noqa: E402
from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor, HeteroCritic  # noqa: E402
from bpm_envs import make_next_activity, make_next_activity_split  # noqa: E402
from multisite_env import make_multisite  # noqa: E402


def _n_comps(pn):
    return len({c for c in pn.net_partition().values() if c is not None})


# --------------------------------------------------------------------------- #
# Partition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 4])
def test_partition_copies(n):
    assert _n_comps(make_next_activity(n, allow_postpone=False)) == n


def test_partition_split_queue_has_two_per_copy():
    assert _n_comps(make_next_activity_split(1, allow_postpone=False)) == 2


@pytest.mark.parametrize("regions", [1, 2, 4])
def test_partition_insurer(regions):
    """One component per process per region; shared clerks join claims and
    complaints; a process alone is one component per region."""
    from insurer_env import INSURER_BUILDERS
    assert _n_comps(INSURER_BUILDERS["insurer"](regions)) == 3 * regions
    assert _n_comps(INSURER_BUILDERS["insurer_shared"](regions)) == 2 * regions
    for p in ("claims", "underwriting", "complaints"):
        assert _n_comps(INSURER_BUILDERS[f"insurer_{p}"](regions)) == regions


def test_partition_multisite():
    assert _n_comps(make_multisite(4, 1, 0)) == 4      # dedicated sites
    assert _n_comps(make_multisite(4, 1, 2)) == 1      # shared flex pool couples all


# --------------------------------------------------------------------------- #
# Per-component reward split
# --------------------------------------------------------------------------- #
def test_comp_reward_sums_to_step_reward():
    random.seed(0); np.random.seed(0)
    pn = make_next_activity(2, allow_postpone=False)
    pn.length = 20
    env = AEPN_Env(pn)
    env.reset()
    for _ in range(80):
        k = len(env.pn.pn_actions)
        if k == 0:
            break
        _, r, done, _, info = env.step(np.random.randint(k))
        assert abs(sum(info['comp_reward'].values()) - r) < 1e-9
        if done:
            break
    assert abs(sum(env.pn.comp_reward.values()) - env.pn.reward) < 1e-9


# --------------------------------------------------------------------------- #
# GAE recursion
# --------------------------------------------------------------------------- #
def _finish(buf, rewards, values, times, comp=None, rw=None):
    for r, v, t in zip(rewards, values, times):
        buf.times.append(float(t))
        buf._rewards_buf.append(torch.tensor([float(r)]))
        buf._values_buf.append(torch.tensor([float(v)]))
        buf._logprobs_buf.append(torch.tensor([0.0]))
        buf.end += 1
    buf._steps_dirty = True
    if comp is not None:
        buf._nf_comp, buf._nf_rew = comp, rw
    buf.finish()
    return buf.advantages_.clone()


def test_k1_identity_with_ppo():
    rng = np.random.default_rng(0)
    T = 40
    rewards = rng.integers(0, 3, T).astype(float)
    values = rng.normal(size=T)
    times = np.cumsum(rng.integers(0, 3, T)).astype(float)
    kw = dict(gam=0.99, lam=0.95, beta=0.5, smdp_discount=True)
    a_ppo = _finish(TrajectoryBuffer(**kw), rewards, values, times)
    a_nf = _finish(TrajectoryBuffer(nfgae=True, **kw), rewards, values, times,
                   comp=[0] * T, rw=[{0: r} for r in rewards])
    assert torch.allclose(a_ppo, a_nf, atol=1e-6)


def test_k2_hand_computed():
    # steps: t0 (c0), t1 (c1), t2 (c0); times 0, 1, 3; beta 0.5, lam 1
    rewards = [1.0, 2.0, 4.0]
    rw = [{1: 1.0}, {0: 2.0}, {0: 4.0}]  # c1's reward at step 0, c0's at steps 1, 2
    values = [0.5, 0.25, 0.75]
    times = [0.0, 1.0, 3.0]
    buf = TrajectoryBuffer(nfgae=True, gam=1.0, lam=1.0, beta=0.5, smdp_discount=True)
    a = _finish(buf, rewards, values, times, comp=[0, 1, 0], rw=rw)
    # c0: epoch t0 owns steps [0, 2) -> R = 0 + 2; next c0 epoch t2 after 3 units
    a_t2 = 4.0 - 0.75
    a_t0 = 2.0 + math.exp(-1.5) * 0.75 - 0.5 + math.exp(-1.5) * a_t2
    # c1: its only epoch t1 owns steps [1, 3) -> no c1 reward there
    a_t1 = 0.0 - 0.25
    assert torch.allclose(a, torch.tensor([a_t0, a_t1, a_t2]), atol=1e-6)


# --------------------------------------------------------------------------- #
# Theorem 1, C2: action logits are local to their component
# --------------------------------------------------------------------------- #
def _obs(extra_waiting_copy1):
    random.seed(1); np.random.seed(1)
    pn = make_next_activity(2, allow_postpone=False)
    P = {p._id: p for p in pn.places}
    for _ in range(extra_waiting_copy1):
        P['waiting_1'].put({'risk': 2, 'bad': 1})
    g = pn.get_graph_observation()['graph']
    part = pn.net_partition()
    comps = [part[b[2]._id] for b in pn.pn_actions]
    return pn, g, comps


@pytest.mark.parametrize("encoder", ["hgt", "type_embed", "type_embed_film", "aepn"])
def test_actor_logits_local_to_component(encoder):
    torch.manual_seed(0)
    pn_a, g_a, c_a = _obs(0)
    pn_b, g_b, c_b = _obs(3)
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3,
                        metadata=pn_a.make_metadata(), num_heads=2,
                        encoder=encoder)
    actor.eval()

    def raw_logits(g):
        with torch.no_grad():
            actor({'graph': g})
            x = actor.encoder(x_dict=g.x_dict, edge_index_dict=g.edge_index_dict,
                              input_size=-1, graph=g, params_iter=iter(actor.parameters()))
            return actor.decoder(x['a_transition']).flatten()

    la, lb = raw_logits(g_a), raw_logits(g_b)
    c0 = next(c for c in c_a if c == c_a[0])
    sel_a = [i for i, c in enumerate(c_a) if c == c0]
    sel_b = [i for i, c in enumerate(c_b) if c == c0]
    assert len(sel_a) == len(sel_b) > 0
    assert torch.equal(la[sel_a], lb[sel_b])      # bit-identical
    # and the perturbed component's logits did move (the test can fail)
    assert la.numel() != lb.numel() or not torch.equal(la, lb)


# --------------------------------------------------------------------------- #
# Critic: component pooling
# --------------------------------------------------------------------------- #
def test_critic_pool_mask():
    torch.manual_seed(0)
    pn, g, comps = _obs(0)
    critic = HeteroCritic(input_size=-1, hidden_size=32, num_layers=2,
                          metadata=pn.make_metadata(), num_heads=1, dropout=0.0)
    critic.eval()
    with torch.no_grad():
        v_all = critic({'graph': g})
        g['a_transition'].pool_mask = torch.ones(len(comps), dtype=torch.bool)
        v_mask_all = critic({'graph': g})
        assert torch.allclose(v_all, v_mask_all)           # K=1: same as before
        g['a_transition'].pool_mask = torch.tensor([c == comps[0] for c in comps])
        v_c = critic({'graph': g})
        assert v_c.shape == v_all.shape


# --------------------------------------------------------------------------- #
# Component-scoped postpone (Theorem 1, C3')
# --------------------------------------------------------------------------- #
def _env(n, scope, seed=3):
    random.seed(seed); np.random.seed(seed)
    pn = make_next_activity(n, allow_postpone=True)
    pn.postpone_scope = scope
    pn.length = 20
    env = AEPN_Env(pn)
    return env, env.reset()


def _is_pp(b):
    return isinstance(b[0], list) and b[0] == ['postpone']


def test_component_postpone_equals_global_at_k1():
    """One component: the two scopes give the identical trajectory and the
    identical observation graphs under the same action choices (run one after
    the other: both draw from the global random stream)."""
    def run(scope):
        env, obs = _env(1, scope)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(60):
            g = obs['graph']
            k = len(env.pn.pn_actions)
            a = int(rng.integers(k))
            n_pp = sum(_is_pp(b) for b in env.pn.pn_actions)
            obs, r, d, _, _ = env.step(a)
            out.append((g, k, n_pp, a, r, d, env.pn.clock))
            if d:
                break
        return out

    tg, tc = run('global'), run('component')
    assert len(tg) == len(tc) > 5
    assert any(a == k - 1 for (_, k, _, a, *_rest) in tg)      # postpone was exercised
    for (g, k, npg, a, r, d, t), (c, k2, npc, a2, r2, d2, t2) in zip(tg, tc):
        assert (k, npg, a, r, d, t) == (k2, npc, a2, r2, d2, t2)
        assert set(g.node_types) == set(c.node_types)
        for nt in g.node_types:
            assert torch.equal(g[nt].x, c[nt].x), nt
        assert set(g.edge_types) == set(c.edge_types)
        for et in g.edge_types:
            assert torch.equal(g[et].edge_index, c[et].edge_index), et


def test_component_postpone_blocks_only_its_component():
    env, obs = _env(2, 'component')
    part = env.pn.net_partition()
    comps = lambda: {(b[2].comp if _is_pp(b) else part[b[2]._id]) for b in env.pn.pn_actions}
    assert comps() == {0, 1}
    clock0 = env.pn.clock
    pp0 = next(i for i, b in enumerate(env.pn.pn_actions) if _is_pp(b) and b[2].comp == 0)
    env.step(pp0)
    # same decision round: only component 1 is offered, no time has passed
    assert env.pn.clock == clock0
    assert comps() == {1}
    assert 0 in env.pn.postponed_comps
    # let component 1 wait too: time advances; component 0 comes back only
    # after one of ITS events has fired
    while 0 in env.pn.postponed_comps:
        pp = [i for i, b in enumerate(env.pn.pn_actions) if _is_pp(b)]
        _, _, done, _, _ = env.step(pp[0])
        if done:
            break
    assert 0 not in env.pn.postponed_comps


@pytest.mark.parametrize("encoder", ["hgt", "type_embed", "type_embed_film", "aepn"])
def test_actor_logits_local_with_component_postpone(encoder):
    """Postpone logits are local too: component 0's action AND postpone
    logits are bit-identical when component 1's marking changes."""
    def obs(extra):
        random.seed(1); np.random.seed(1)
        pn = make_next_activity(2, allow_postpone=True)
        pn.postpone_scope = 'component'
        P = {p._id: p for p in pn.places}
        for _ in range(extra):
            P['waiting_1'].put({'risk': 2, 'bad': 1})
        g = pn.get_graph_observation()['graph']
        part = pn.net_partition()
        comps = [(b[2].comp if _is_pp(b) else part[b[2]._id]) for b in pn.pn_actions]
        return pn, g, comps

    torch.manual_seed(0)
    pn_a, g_a, c_a = obs(0)
    pn_b, g_b, c_b = obs(3)
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3,
                        metadata=pn_a.make_metadata(), num_heads=2,
                        encoder=encoder)
    actor.eval()

    def raw(g):
        with torch.no_grad():
            actor({'graph': g})
            x = actor.encoder(x_dict=g.x_dict, edge_index_dict=g.edge_index_dict,
                              input_size=-1, graph=g, params_iter=iter(actor.parameters()))
            return torch.cat([actor.decoder(x['a_transition']).flatten(),
                              actor.decoder(x['postpone']).flatten()])

    la, lb = raw(g_a), raw(g_b)
    sel_a = [i for i, c in enumerate(c_a) if c == 0]
    sel_b = [i for i, c in enumerate(c_b) if c == 0]
    assert len(sel_a) == len(sel_b) > 1          # actions + its postpone
    assert torch.equal(la[sel_a], lb[sel_b])


# --------------------------------------------------------------------------- #
# Component turns (local encoding)
# --------------------------------------------------------------------------- #
def _insurer_env(builder, regions, turns, seed=0):
    from insurer_env import INSURER_BUILDERS
    random.seed(seed); np.random.seed(seed)
    pn = INSURER_BUILDERS[builder](regions)
    pn.length = 20
    pn.component_turns = turns
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, (obs[0] if isinstance(obs, tuple) else obs)


def test_component_turns_identity_at_k1():
    """One component: turns change nothing (same graph, same bindings)."""
    ea, oa = _insurer_env("insurer_claims", 1, False)
    eb, ob = _insurer_env("insurer_claims", 1, True)
    rng = random.Random(3)
    for _ in range(40):
        ga, gb = oa['graph'], ob['graph']
        assert ga.node_types == gb.node_types and ga.edge_types == gb.edge_types
        for nt in ga.node_types:
            assert torch.equal(ga[nt].x, gb[nt].x)
        for et in ga.edge_types:
            assert torch.equal(ga[et].edge_index, gb[et].edge_index)
        assert len(ea.pn.pn_actions) == len(eb.pn.pn_actions)
        k = len(ea.pn.pn_actions)
        if k == 0:
            break
        a = rng.randrange(k)
        state = random.getstate()
        oa, _, da, _, _ = ea.step(a)
        random.setstate(state)
        ob, _, db, _, _ = eb.step(a)
        if da:
            assert db
            break


def test_component_turns_local_content_and_coverage():
    """Each observation and binding list holds one component; time never
    advances while another component still has enabled work."""
    env, obs = _insurer_env("insurer", 2, True)
    part, ppart = env.pn.net_partition(), env.pn.net_place_partition()
    rng = random.Random(5)
    for _ in range(150):
        binds = env.pn.pn_actions
        if not binds:
            break
        comps = {part[b[2]._id] for b in binds}
        assert len(comps) == 1
        c = comps.pop()
        g = obs['graph']
        for nt in g.node_types:
            if nt not in ('a_transition', 'e_transition'):
                assert ppart[nt] == c, (nt, c)
        # all components enabled now (turns off), to check coverage below
        env.pn.component_turns = False
        enabled_all = {part[b[2]._id] for b in env.pn.compute_pn_actions()}
        env.pn.component_turns = True
        env.pn.compute_pn_actions()
        assert c == min(enabled_all)
        clock = env.pn.clock
        obs, _, done, _, _ = env.step(rng.randrange(len(binds)))
        if env.pn.clock > clock:
            assert enabled_all == {c}, "time advanced while another component had work"
        if done:
            break


@pytest.mark.parametrize("encoder", ["aepn", "type_embed"])
def test_component_turns_same_logits(encoder):
    """Locality: the active component's raw logits are the same whether the
    observation holds the whole net or only that component."""
    torch.manual_seed(0)
    env, _ = _insurer_env("insurer", 2, False, seed=2)
    rng = random.Random(2)
    for _ in range(12):
        env.step(rng.randrange(len(env.pn.pn_actions)))
    actor = HeteroActor(hidden_size=32, num_layers=3, metadata=env.pn.make_metadata(),
                        num_heads=2, dropout=0.0, encoder=encoder).eval()

    def raw(graph):
        from gympn.networks import _encode_graph
        with torch.no_grad():
            x = _encode_graph(actor.encoder, graph, -1, iter(actor.parameters()))
            return actor.decoder(x['a_transition']).flatten()

    env.pn.component_turns = False
    g_full = env.pn.get_graph_observation()['graph']
    full_binds = list(env.pn.pn_actions)
    env.pn.component_turns = True
    g_loc = env.pn.get_graph_observation()['graph']
    loc_binds = list(env.pn.pn_actions)
    l_full, l_loc = raw(g_full), raw(g_loc)
    # bindings are rebuilt per observation (tokens copied by the expansion), so
    # match by content, in order, using each full-graph binding once
    key = lambda b: (b[2]._id, tuple((p._id, repr(t.value), t.time) for p, t in b[0]))
    full_keys, used, sel = [key(b) for b in full_binds], set(), []
    for b in loc_binds:
        i = next(j for j, k in enumerate(full_keys) if k == key(b) and j not in used)
        used.add(i); sel.append(i)
    assert len(loc_binds) >= 1 and len(loc_binds) < len(full_binds)
    assert torch.allclose(l_full[sel], l_loc, atol=1e-5)


# --------------------------------------------------------------------------- #
# Start-of-shift allocation variant (insurer_alloc)
# --------------------------------------------------------------------------- #
def _alloc_run(regions, plan, seed=0, marking=False, probe=None):
    """Step an insurer_alloc episode with a plan-following heuristic until the
    allocation is over; return the env. probe(pn) runs on the initial state."""
    from insurer_env import make_insurer_alloc, make_alloc_heuristic, _alloc_binding
    random.seed(seed); np.random.seed(seed)
    pn = make_insurer_alloc(regions)
    pn.length = 20
    pn.marking_certificate = marking
    env = AEPN_Env(pn)
    env.reset()
    if probe is not None:
        probe(env.pn)
    heur = make_alloc_heuristic(plan)
    while any(_alloc_binding(b) for b in env.pn.pn_actions):
        b = heur(None, None, env.pn.pn_actions)
        env.step(env.pn.pn_actions.index(b))
    return env


@pytest.mark.parametrize("regions", [1, 2])
def test_partition_insurer_alloc_static(regions):
    """The pool couples the three processes of a region: one component each."""
    from insurer_env import INSURER_BUILDERS
    assert _n_comps(INSURER_BUILDERS["insurer_alloc"](regions)) == regions


@pytest.mark.parametrize("regions", [1, 2])
def test_partition_insurer_alloc_after_allocation(regions):
    """Once the pool is empty the allocation transitions are dead, and the
    certificate recomputed at the current marking splits every region into
    its three processes."""
    from insurer_env import ALLOC_PLAN
    env = _alloc_run(regions, ALLOC_PLAN)
    assert env.pn.clock == 0                          # allocation takes no time
    places = {p._id: p for p in env.pn.places}
    assert all(len(places[f"staff_{r}"].marking) == 0 for r in range(regions))
    env.pn._net_partition_cache = None
    assert _n_comps(env.pn) == 3 * regions


def test_alloc_heuristic_follows_plan():
    """The stateless heuristic executes an ordered plan, including a split of
    the two identical experts."""
    env = _alloc_run(1, [("cl", 1), ("co", 1), ("uw1", 0)])
    places = {p._id: p for p in env.pn.places}

    def skills(name, key):
        return sorted(getattr(t, 'value', t)[key] for t in places[name].marking)
    # busy clerks are not in their pool place, so count pool + busy
    def staff(proc, key):
        busy = [getattr(t, 'value', t)[1][key] for t in places[f"{proc}_busy_0"].marking]
        return sorted(skills(f"{proc}_{'team' if proc == 'uw' else 'clerks'}_0", key) + busy)
    assert staff("cl", "skill") == [0, 1]
    assert staff("co", "skill") == [0, 1]
    assert staff("uw", "code") == [0, 1]


# --------------------------------------------------------------------------- #
# Marking-based certificate (GymProblem.marking_certificate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regions", [1, 2])
def test_marking_certificate_refines(regions):
    """Marking-based: one component per region while the pool is non-empty,
    three per region once it is empty. Static (default): the partition is
    computed once, at first use, and stays coarse."""
    from insurer_env import ALLOC_PLAN
    seen = {}
    env = _alloc_run(regions, ALLOC_PLAN, marking=True,
                     probe=lambda pn: seen.setdefault("start", _n_comps(pn)))
    assert seen["start"] == regions
    assert _n_comps(env.pn) == 3 * regions
    env = _alloc_run(regions, ALLOC_PLAN, marking=False,
                     probe=lambda pn: seen.setdefault("static", _n_comps(pn)))
    assert seen["static"] == regions and _n_comps(env.pn) == regions


def test_marking_certificate_members_and_reward_labels():
    """Members of a component are its live transitions; rewards are keyed by
    transition and still sum to the step reward."""
    from insurer_env import make_insurer_alloc
    random.seed(0); np.random.seed(0)
    pn = make_insurer_alloc(1)
    pn.length = 20
    pn.marking_certificate = True
    env = AEPN_Env(pn)
    env.reset()
    region = env.pn.net_component_members(0)
    assert {"al_cl_0", "cl_approve_0", "uw_start_0", "co_check_0"} <= region
    done = False
    while not done and env.pn.pn_actions:
        _, r, done, _, info = env.step(np.random.randint(len(env.pn.pn_actions)))
        assert abs(sum(info['comp_reward'].values()) - r) < 1e-9
        assert all(k in region for k in info['comp_reward'])
    assert abs(sum(env.pn.comp_reward.values()) - env.pn.reward) < 1e-9


def test_refining_gae_equals_static_without_splits():
    """With components that never split, the refining recursion (frozenset
    components, rewards keyed by transition) equals the static one."""
    rng = np.random.default_rng(1)
    T = 60
    comp = rng.integers(0, 3, T).tolist()
    members = {c: frozenset({f"t{c}a", f"t{c}b"}) for c in range(3)}
    rw_static, rw_trans = [], []
    for _ in range(T):
        st, tr = {}, {}
        for c in range(3):
            if rng.random() < 0.4:
                r = float(rng.integers(1, 3))
                st[c] = st.get(c, 0.0) + r
                key = f"t{c}" + ("a" if rng.random() < 0.5 else "b")
                tr[key] = tr.get(key, 0.0) + r
        rw_static.append(st)
        rw_trans.append(tr)
    rewards = [sum(d.values()) for d in rw_static]
    values = rng.normal(size=T)
    times = np.cumsum(rng.integers(0, 3, T)).astype(float)
    kw = dict(nfgae=True, gam=0.99, lam=0.9, beta=0.3, smdp_discount=True)
    a_static = _finish(TrajectoryBuffer(**kw), rewards, values, times, comp=comp, rw=rw_static)
    a_ref = _finish(TrajectoryBuffer(**kw), rewards, values, times,
                    comp=[members[c] for c in comp], rw=rw_trans)
    assert torch.allclose(a_static, a_ref, atol=1e-5)


def test_refining_gae_lambda1_is_component_return():
    """lam = 1 with nested splits: the advantage of a decision is its
    component's return minus its value. Each reward of transition x fired at
    step s is discounted from the latest step u in [t, s] at which x's
    component decided (the SMDP convention: undiscounted within a step)."""
    G, A, BC, B, C = (frozenset(x) for x in ("abc", "a", "bc", "b", "c"))
    comp = [G, G, A, BC, A, B, C, A, B, C]
    T = len(comp)
    rng = np.random.default_rng(2)
    rw = [{x: float(rng.integers(1, 4)) for x in "abc" if rng.random() < 0.5} for _ in range(T)]
    rewards = [sum(d.values()) for d in rw]
    values = rng.normal(size=T)
    times = np.cumsum(rng.integers(0, 3, T)).astype(float)
    beta = 0.4
    a = _finish(TrajectoryBuffer(nfgae=True, gam=1.0, lam=1.0, beta=beta, smdp_discount=True),
                rewards, values, times, comp=comp, rw=rw)
    expected = []
    for t in range(T):
        ret = 0.0
        for x in comp[t]:
            chain = [u for u in range(t, T) if x in comp[u]]
            for s_ in range(t, T):
                p = max(u for u in chain if u <= s_)
                ret += rw[s_].get(x, 0.0) * math.exp(-beta * (times[p] - times[t]))
        expected.append(ret - values[t])
    assert torch.allclose(a, torch.tensor(expected, dtype=torch.float32), atol=1e-5)


def test_marking_certificate_training_path(monkeypatch, tmp_path):
    """One real nfgae epoch on insurer_alloc with the marking certificate goes
    through the refining recursion: the first decisions are credited to the
    whole region, the later ones to the region's three processes."""
    from gympn.data import TrajectoryBuffer as TB
    from insurer_env import make_insurer_alloc
    seen = []
    orig = TB._nfgae_refining

    def spy(self, values_ep, comp, rw, times):
        seen.append(list(comp))
        return orig(self, values_ep, comp, rw, times)
    monkeypatch.setattr(TB, "_nfgae_refining", spy)
    random.seed(0); np.random.seed(0)
    pn = make_insurer_alloc(1)
    args = {"algorithm": "ppo-clip", "episodes": 1, "epochs": 1, "batch_size": 8,
            "nfgae": True, "marking_certificate": True, "smdp_discount": True,
            "flat_obs": True, "policy_kwargs": {"encoder": "aepn"},
            "value_kwargs": {"encoder": "aepn"},
            "test_in_train": False, "use_wandb": False, "open_tensorboard": False,
            "verbose": 0, "use_gpu": False, "agent_seed": 0, "logdir": str(tmp_path),
            "save_freq": 1_000_000, "datetag": False, "name": "mc"}
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        pn.training_run(length=20, args_dict=args)
    finally:
        sys.argv = saved
    assert seen, "the refining recursion was not used"
    comps = seen[0]
    region = comps[0]
    assert {"al_cl_0", "cl_approve_0", "uw_start_0", "co_check_0"} <= region
    later = {c for c in comps if c != region}
    assert len(later) == 3 and all(c < region for c in later)
    assert not any((x & y) for x in later for y in later if x != y)
    first_split = next(i for i, c in enumerate(comps) if c != region)
    assert all(c != region for c in comps[first_split:])     # never coarsens again
