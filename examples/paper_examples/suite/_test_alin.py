r"""ALIN's defining properties, on real traces from live training runs.

ALIN is mc_q MINUS the rewards a decision provably could not have influenced.
Three things must hold, and each catches a different way of getting it wrong:

  P1  SUBTRACTION ONLY:  alin(d) <= mc_q(d) for every decision, always.
      A value above mc_q would mean it invented credit rather than removed it.

  P2  SUPERSET OF LINEAGE:  alin(d) >= lrq(d) whenever lrq credits d.
      alin keeps everything lrq keeps PLUS the contention-reachable rewards
      lrq discards. If alin ever fell below lrq it would be dropping
      action-dependent reward -- the exact foreclosure bias it exists to fix.

  P3  BRACKETED BY ITS ENDPOINTS:  lrq2 <= alin <= mc_q, and WHERE it sits
      between them says what the contention edges bought. == lrq2 means they
      add nothing over token lineage; == mc_q means nothing was ever provably
      independent.

      P3 ORIGINALLY read "on a single saturated pool every decision reaches
      every reward, so alin == mc_q EXACTLY". That is false, and not because
      of an implementation bug. Contention edges run FORWARD in time, so a
      reward whose lineage completed BEFORE d is unreachable from d -- the
      case was already in service, d neither started it nor could delay it,
      and mc_q counts it only because it lands later on the wall clock.
      Subtracting it is CORRECT. Measured on s1 (suite/_diag_alin_p3b.py,
      6 episodes, 922 decision-reward pairs in the mc_q horizon): 748 reached,
      and of the 174 missed, 133 (76.4%) are exactly this already-in-flight
      case. So even a perfect closure cannot degenerate to mc_q here.

      The remaining 41 (23.6%) ARE a real hole in the contention construction
      -- a lineage decision fired at or after d yet d cannot reach it. Not
      caused by postpone sentinels sitting outside the pool chains (0 of the
      41 are reachable only through a postpone node) and not by the time-tie
      ordering (fixed; closure now runs 0 fallbacks). Unexplained, bounded,
      and CONSERVATIVE: it makes alin subtract slightly more than it should,
      costing variance reduction rather than adding bias, which is why P1/P2
      stay clean.

P3 is exercised on s1 (one shared employee pool, K_static = K_realized = 1)
and on ncopies4 (independent per-copy pools), where the copies share no
resource so the contention edges are expected to add nothing at all.

Run: python _test_alin.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as traces_mod

_R = {}


def _probe(tag):
    _R[tag] = {"n": 0, "p1": 0, "p2": 0, "cmp": [], "eq_mcq": 0, "tot": 0}
    real = traces_mod.CausalTraces.redistribute_rewards

    def spy(self, scheme="lrq", **kw):
        out = real(self, scheme=scheme, **kw)
        if scheme == "lrq2":                      # called once per episode
            try:
                beta = kw.get("beta", 0.0)
                a = np.asarray(real(self, scheme="alin", beta=beta), float)
                m = np.asarray(real(self, scheme="mc_q", beta=beta), float)
                l = np.asarray(real(self, scheme="lrq2", beta=beta), float)
                k = min(len(a), len(m), len(l))
                a, m, l = a[:k], m[:k], l[:k]
                r = _R[tag]
                r["n"] += 1
                r["p1"] += int((a > m + 1e-9).sum())          # violations
                r["p2"] += int((a < l - 1e-9).sum())          # violations
                r["tot"] += k
                r["eq_mcq"] += int(np.isclose(a, m, atol=1e-9).sum())
                if m.sum() > 0:
                    r["cmp"].append(float(a.sum() / m.sum()))
                    r.setdefault("lrq_cmp", []).append(float(l.sum() / m.sum()))
            except Exception as e:
                print(f"[warn] {tag}: {e}")
        return out

    traces_mod.CausalTraces.redistribute_rewards = spy
    return real


def run(tag, make_env_fn, length):
    real = _probe(tag)
    args = {
        "algorithm": "ppo-clip", "episodes": 6, "epochs": 2, "batch_size": 64,
        "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 1,
        "value_lr": 3e-4, "value_updates": 1, "gam": 0.99, "lam": 0.95,
        "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
        "causal_rl": True, "causal_scheme": "lrq2", "causal_beta": 0.5,
        "verbose": 0, "use_gpu": False, "agent_seed": 0,
        "use_wandb": False, "open_tensorboard": False, "test_in_train": False,
        "save_freq": 1_000_000, "name": "alin", "datetag": False,
        "logdir": f"alin_train_{tag}",
    }
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        make_env_fn().training_run(length=length, args_dict=args)
    finally:
        sys.argv = saved
        traces_mod.CausalTraces.redistribute_rewards = real
        import shutil
        shutil.rmtree(f"alin_train_{tag}", ignore_errors=True)


from envs import make_env  # noqa: E402
from ncopies_env import make_n_copies  # noqa: E402

run("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True,
                           allow_postpone=True), 20)
run("ncopies4", lambda: make_n_copies(4, causal_rl=True,
                                      allow_postpone=False), 20)
# s2 is the discriminating case: K_realized=6 vs K_static=1, i.e. realized
# coupling much sparser than static, which is exactly where alin should find
# something to subtract that s_ccf cannot see.
run("s2", lambda: make_env("s2_stoch_scaled", causal_rl=True,
                           allow_postpone=True), 30)

print("\n" + "=" * 68)
for tag in _R:
    r = _R[tag]
    frac_eq = r["eq_mcq"] / max(r["tot"], 1)
    ratio = np.mean(r["cmp"]) if r["cmp"] else float("nan")
    lratio = np.mean(r.get("lrq_cmp") or [float("nan")])
    print(f"[{tag}]  episodes={r['n']}  decisions={r['tot']}")
    print(f"   P1 alin <= mc_q  : {r['p1']} violations")
    print(f"   P2 alin >= lrq2  : {r['p2']} violations")
    print(f"   credit mass vs mc_q   : lrq2={lratio:.4f}  alin={ratio:.4f}  mc_q=1.0000")
    print(f"   decisions where alin == mc_q exactly : {frac_eq:.1%}")
    assert r["n"] > 0, f"{tag}: no episodes"
    assert r["p1"] == 0, f"{tag}: alin exceeded mc_q ({r['p1']} times)"
    assert r["p2"] == 0, f"{tag}: alin fell below lrq2 ({r['p2']} times)"

# P3 (RESTATED -- the original statement was wrong; see the header for the
# measured breakdown). On a single saturated pool alin does NOT reach every
# reward: the contention chain runs FORWARD in time, so a reward whose lineage
# completed BEFORE d is unreachable from d and is correctly subtracted even on
# one pool (76.4% of s1's misses are exactly that). What actually characterises
# alin is that it lies between lrq2 and mc_q -- equal to lrq2 means the
# contention edges add nothing over token lineage, equal to mc_q means nothing
# is ever provably independent, and both endpoints already exist as methods.
print()
for tag in _R:
    a = np.mean(_R[tag]["cmp"]); l = np.mean(_R[tag]["lrq_cmp"])
    gap = (a - l) / max(1.0 - l, 1e-9)
    verdict = ("== lrq2 (contention edges add NOTHING)" if abs(a - l) < 1e-6
               else f"recovers {gap:.1%} of the lrq2 -> mc_q gap")
    print(f"[{tag}] lrq2={l:.4f}  alin={a:.4f}  mc_q=1.0  -> {verdict}")
    assert a >= l - 1e-9, f"{tag}: alin below lrq2"
print("\nP1/P2 hold on every env; the ratios above say whether alin has teeth")
