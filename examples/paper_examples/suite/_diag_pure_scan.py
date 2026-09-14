r"""Does ANY suite env give LS-HCA a nonempty PURE set?

`_redistribute_ls_hca` splits each decision's reachable reward-types into
PURE (credited exactly, no fit -- this is the "L", the lineage support that
distinguishes LS-HCA from plain HCA) and CONTESTED (the hindsight-corrected
part). On s1 the split is PURE=set() for every action, so 100% of the credit
mass flows through the hindsight correction and the lineage support
contributes literally nothing (measured: `_diag_ls_hca_step0.log`).

If that holds across every env in the suite, then LS-HCA has only ever been
evaluated in the regime where it degenerates to plain HCA, and no experiment
run so far has actually tested its distinguishing mechanism.

Pure static graph analysis -- `_pure_contested_reward_types` needs only the
net topology, so this needs no training and no episodes.

Run: python _diag_pure_scan.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

from gympn.causal_traces import CausalTraces  # noqa: E402
from envs import ENV_BUILDERS, make_env  # noqa: E402

rows = []
for name in sorted(ENV_BUILDERS):
    try:
        env = make_env(name, causal_rl=True, allow_postpone=True)
        ct = CausalTraces()
        ct._pn = env
        classify = ct._pure_contested_reward_types()
    except Exception as e:
        rows.append((name, None, None, None, f"ERROR: {type(e).__name__}: {e}"))
        continue

    reward_types = sorted(env.reward_functions.keys())
    any_pure = any(pure for pure, _ in classify.values())
    n_with_pure = sum(1 for pure, _ in classify.values() if pure)
    detail = "; ".join(
        f"{aid}: PURE={sorted(pure) or '-'} CONTESTED={sorted(cont) or '-'}"
        for aid, (pure, cont) in sorted(classify.items()))
    rows.append((name, reward_types, any_pure, f"{n_with_pure}/{len(classify)}",
                 detail))

print("=" * 78)
print("PURE-set scan across all suite envs")
print("=" * 78)
print(f"{'env':<26} {'PURE?':<7} {'actions w/ PURE':<16} reward types")
print("-" * 78)
for name, rtypes, any_pure, frac, _ in rows:
    if any_pure is None:
        print(f"{name:<26} {'ERR':<7} {'-':<16} -")
        continue
    print(f"{name:<26} {('YES' if any_pure else 'no'):<7} {frac:<16} {rtypes}")

print("\n" + "-" * 78)
print("per-action detail")
print("-" * 78)
for name, _, any_pure, _, detail in rows:
    print(f"\n{name}:")
    print(f"  {detail}")

ok = [r[0] for r in rows if r[2] is True]
bad = [r[0] for r in rows if r[2] is False]
err = [r[0] for r in rows if r[2] is None]
print("\n" + "=" * 78)
print(f"envs with a nonempty PURE set ({len(ok)}): {ok or 'NONE'}")
print(f"envs where LS-HCA == plain HCA ({len(bad)}): {bad}")
if err:
    print(f"errored ({len(err)}): {err}")
    for name, _, any_pure, _, detail in rows:
        if any_pure is None:
            print(f"  {name}: {detail}")
