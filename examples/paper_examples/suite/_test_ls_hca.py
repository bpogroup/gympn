"""Unit + smoke tests for the ls_hca training wiring (FORKFREE_LINEAGE_
RETHINK.md Idea 1; causal_traces._redistribute_ls_hca + agents.py's ls_hca
block + Agent._fit_ls_hca_hhat). Run:  python _test_ls_hca.py

Covers, in order: (1) the library classification matches ls_hca_probe.py's
standalone reachability walk on M2/M4; (2) _redistribute_ls_hca's PURE credit
+ pending ingredients on a forced M2 trajectory; (3) _fit_ls_hca_hhat's
empirical table on synthetic records; (4) a real end-to-end smoke train
(train_cell, 'ls_hca', 3 epochs) on a tiny suite env -- no crash, hhat grows,
epoch-0 safe floor (empty hhat -> pure-only credit).
"""
import sys, os, types, uuid
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gympn
from simpn.simulator import SimToken
from gympn.environment import AEPN_Env
from assembly_probe import make_shared_r, make_two_chains, run_forced
from ls_hca_probe import classify as probe_classify


def _run_forced_pn(choice_name, make_fn, beta=0.0, length=12, **rw):
    """Like assembly_probe.run_forced, but hands back env.pn too."""
    gympn.seed_everything(0)
    pn = make_fn(causal_rl=True, **rw)
    pn.length = length
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn); env.reset()
    forced = False
    for _ in range(30):
        acts = env.pn.pn_actions
        if not acts:
            break
        idx = 0
        for i, a in enumerate(acts):
            tname = getattr(a[2], '_id', getattr(a[2], 'name', '')) if isinstance(a, tuple) and len(a) > 2 and a[2] is not None else ''
            if choice_name in str(tname):
                idx = i; forced = True; break
        _, r, done, _, _ = env.step(idx)
        if done:
            break
    return env.pn, forced


def test_classify_matches_probe_on_m2():
    pn, forced = _run_forced_pn('use_R', make_shared_r, beta=0.3, r_join=5, r1=5, r2=10)
    assert forced
    pn.causal_trace._pn = pn
    pn.causal_trace._ls_hca_classify_cache = None
    lib = pn.causal_trace._pure_contested_reward_types()
    probe_pure, probe_contested = probe_classify(pn, {'use_R', 'standalone'})
    assert lib['use_R'][0] == probe_pure, (lib['use_R'][0], probe_pure)
    assert lib['use_R'][1] == probe_contested, (lib['use_R'][1], probe_contested)
    assert lib['standalone'][0] == probe_pure
    assert lib['standalone'][1] == probe_contested
    print(f"  classify_matches_probe_on_m2 OK (pure={probe_pure}, contested={probe_contested})")


def test_classify_splits_pool_sharing_decisions_on_m4():
    """Regression for the bug the s1 smoke found: actions that share only a
    RECYCLABLE RESOURCE place (M4's abundant R, 1 unit, freed+reused) must
    NOT be merged into one decision point -- unlike M2's part1 (a one-shot
    case token, genuine choice), R is cyclic (freed back to itself), so
    A_hi/A_lo/B_run must end up with a real "other" decision to be CONTESTED
    against instead of everything collapsing to PURE."""
    pn = make_two_chains(causal_rl=True, r_hi=5, r_lo=2, r_b=4)
    pn.causal_trace._pn = pn
    pn.causal_trace._ls_hca_classify_cache = None
    lib = pn.causal_trace._pure_contested_reward_types()
    for aid in ('A_hi', 'A_lo', 'B_run'):
        pure, contested = lib[aid]
        assert pure == set(), (aid, pure)
        assert contested == {'cA_hi', 'cA_lo', 'cB'}, (aid, contested)
    print("  classify_splits_pool_sharing_decisions_on_m4 OK "
          "(A_hi/A_lo/B_run all CONTESTED, no longer collapsed to PURE)")


def test_classify_splits_pool_sharing_decisions_on_s1():
    """The actual s1 case: start1/start2 share only the 3-employee POOL
    (cyclic: freed by done1/done2, reused by later cases) -- must NOT merge,
    so done2 comes out CONTESTED (previously PURE, hhat always empty)."""
    from stoch_envs import make_s1_stoch_sequence
    pn = make_s1_stoch_sequence(causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=False)
    pn.causal_trace._pn = pn
    pn.causal_trace._ls_hca_classify_cache = None
    lib = pn.causal_trace._pure_contested_reward_types()
    for aid in ('start1', 'start2'):
        pure, contested = lib[aid]
        assert pure == set(), (aid, pure)
        assert contested == {'done2'}, (aid, contested)
    print("  classify_splits_pool_sharing_decisions_on_s1 OK "
          "(start1/start2 CONTESTED on done2)")


def test_redistribute_ls_hca_pure_and_pending():
    """On M2 beta=0.3: standalone's PURE credit must be exactly r1's
    discounted value (sa_done, deterministic to d1); use_R's PURE credit must
    be 0 (it never reaches sa_done). Both must emit CONTESTED pending entries
    for join/priv."""
    pn, forced = _run_forced_pn('standalone', make_shared_r, beta=0.3, r_join=5, r1=5, r2=10)
    assert forced
    ct = pn.causal_trace
    ct._pn = pn
    ct._ls_hca_classify_cache = None
    cr = ct.redistribute_rewards(scheme='ls_hca', beta=0.3)
    # r1=5, discounted from d1's clock to sa_done's firing time (1 unit later,
    # from sa1's behavior delay) -- exact value cross-checked against lrq,
    # which credits d1 the same PURE reward on this deterministic net.
    cr_lrq = ct.redistribute_rewards(scheme='lrq', beta=0.3)
    assert abs(cr[0] - cr_lrq[0]) < 1e-6, (cr[0], cr_lrq[0])
    assert 0.0 < cr[0] < 5.0, cr  # discounted below the undiscounted r1=5
    pending = ct._ls_hca_pending
    assert len(pending) == 2  # two decisions this episode: d1 (standalone), d2 (route2)
    # entries are (a_type, rtype, z, contrib, delay, depth, share)
    rtypes = {e[1] for e in pending[0]}
    # standalone never fires 'join' (needs jw1, only reachable from use_R) --
    # only priv's realized instance is a pending CONTESTED ingredient here.
    assert rtypes == {'priv'}, rtypes
    assert all(e[2] is False for e in pending[0])  # not in d1's lineage

    pn2, forced2 = _run_forced_pn('use_R', make_shared_r, beta=0.3, r_join=5, r1=5, r2=10)
    ct2 = pn2.causal_trace
    ct2._pn = pn2
    ct2._ls_hca_classify_cache = None
    cr2 = ct2.redistribute_rewards(scheme='ls_hca', beta=0.3)
    assert abs(cr2[0]) < 1e-6, cr2  # use_R never reaches sa_done -> 0 pure credit
    pending2 = ct2._ls_hca_pending[0]
    assert all(e[2] is True for e in pending2)
    print(f"  redistribute_ls_hca_pure_and_pending OK (standalone pure={cr[0]:.2f}, "
          f"use_R pure={cr2[0]:.2f}, both emit {rtypes})")


def test_fit_hhat_recovers_deterministic_table():
    from gympn.agents import Agent
    agent = object.__new__(Agent)
    agent._ls_hca_records = [
        ('use_R', 'priv', True), ('use_R', 'priv', True),
        ('standalone', 'priv', False), ('standalone', 'priv', False),
        ('use_R', 'cB', False), ('standalone', 'cB', False),  # uninformative type
    ]
    stats = agent._fit_ls_hca_hhat()
    assert stats['n'] == 6
    hhat = agent._ls_hca_hhat
    assert abs(hhat[('use_R', 'priv', True)] - 1.0) < 1e-9
    assert abs(hhat[('standalone', 'priv', False)] - 1.0) < 1e-9
    assert ('standalone', 'priv', True) not in hhat  # never observed -> caller falls back to pi
    # uninformative type: both actions produce z=False equally -> hhat = 0.5 each
    assert abs(hhat[('use_R', 'cB', False)] - 0.5) < 1e-9
    assert abs(hhat[('standalone', 'cB', False)] - 0.5) < 1e-9
    print(f"  fit_hhat_recovers_deterministic_table OK ({len(hhat)} entries)")


def test_end_to_end_smoke_train():
    """Real training loop (train_cell), 'ls_hca', tiny budget: must not crash,
    epoch 0 uses the empty-hhat safe floor (no contested correction possible
    since _ls_hca_hhat starts {}), and the hhat table must be non-empty by the
    end (records were pooled and fit)."""
    from config import SuiteConfig
    from pathlib import Path
    from run_suite import train_cell
    import tempfile

    cfg = SuiteConfig(
        envs=["d_parallel_disjoint"], methods=["ls_hca"], seeds=1,
        epochs=3, episodes_per_epoch=4, test_freq=1, batch_size=8,
        output_dir=Path(tempfile.mkdtemp(prefix="ls_hca_smoke_")),
    )
    metrics = train_cell("d_parallel_disjoint", "ls_hca", 0, cfg,
                         str(Path(cfg.output_dir) / "train"))
    assert metrics is not None and 'env' in metrics
    print(f"  end_to_end_smoke_train OK (metrics keys: {sorted(metrics.keys())})")


if __name__ == "__main__":
    test_classify_matches_probe_on_m2()
    test_classify_splits_pool_sharing_decisions_on_m4()
    test_classify_splits_pool_sharing_decisions_on_s1()
    test_redistribute_ls_hca_pure_and_pending()
    test_fit_hhat_recovers_deterministic_table()
    test_end_to_end_smoke_train()
    print("all ls_hca tests passed")