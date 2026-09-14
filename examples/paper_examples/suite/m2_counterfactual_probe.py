r"""DAG-replay counterfactual on the SHARED-RESOURCE JOIN (M2) -- the exact motif
where ccf and lrq are biased / flip the policy.

M2 (from assembly_probe.make_shared_r): d1 chooses use_R (grab the shared resource
R now, delaying stream 2's private reward r2) or standalone. Stream 2's 'priv'
reward r2 fires REGARDLESS of d1, but its TIMING depends on d1 (use_R holds R
until t=3, so priv is delayed; standalone leaves R free, so priv fires early).
Under SMDP discounting that timing shift is a real, action-dependent effect.

Why filtering fails here: under use_R the resource token flows d1 -> ... -> priv,
so priv is in d1's realized token-lineage; under standalone it is not. So ccf/lrq
(keep/drop by realized membership) treat priv's credit ACTION-DEPENDENTLY -> biased,
and the probe shows the sign flips.

Why the counterfactual is right: it does not keep-or-drop priv. It COMPUTES the
change -- G(use_R) - G(standalone) -- in which priv appears in BOTH runs at
different times, so its discounted-value *difference* is scored exactly. Anything
NOT descended from d1 appears identically in both runs and CANCELS.

Part 1 (correctness): on the deterministic M2, the DAG-replay counterfactual (re-run
d1 flipped, CRN) recovers mc_q's unbiased difference, where ccf/lrq flip.
Part 2 (payoff): add an INDEPENDENT noisy reward w. The counterfactual cancels w
(non-descendant) -> the advantage is w-noise-free (near-zero variance) AND unbiased,
while mc_q carries the full Var(w). Filtering can't get here (it's biased on priv).
"""
import sys, os, types, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import random
import numpy as np
import gympn
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.environment import AEPN_Env
from assembly_probe import make_shared_r, run_forced


def make_shared_r_noisy(w, causal_rl=True, r_join=5.0, r1=5.0, r2=10.0):
    """M2 shared-R join PLUS an independent stream whose reward is the exogenous
    w (drawn per-episode, action-independent of d1). w is pre-placed on a token so
    re-running d1 with the other action reuses the SAME w (CRN) -> it cancels in
    the counterfactual difference."""
    ag = GymProblem(allow_postpone=False, causal_rl=causal_rl)
    for nm in ("part1", "part2", "R", "busy1", "jw1", "jw2", "p2waitR", "sa1",
               "cnoise", "done"):
        ag.add_var(nm, var_attributes=['id'])
    P = {p._id: p for p in ag.places}
    P['part1'].put({'id': 1}); P['part2'].put({'id': 2}); P['R'].put({'id': 9})
    P['cnoise'].put({'id': 7})
    ag.add_action([P['part2']], [P['jw2'], P['p2waitR']],
                  behavior=lambda p: [SimToken(p, delay=1), SimToken(p)], name='route2')
    ag.add_action([P['part1'], P['R']], [P['busy1']],
                  behavior=lambda c, r: [SimToken((c, r), delay=3)], name='use_R')
    ag.add_action([P['part1']], [P['sa1']],
                  behavior=lambda c: [SimToken(c, delay=1)], name='standalone')
    ag.add_event([P['busy1']], [P['jw1'], P['R']],
                 lambda b: [SimToken(b[0]), SimToken(b[1])], name='done1')
    ag.add_event([P['sa1']], [P['done']], lambda p: [SimToken(p)],
                 name='sa_done', reward_function=lambda b: r1)
    ag.add_event([P['jw1'], P['jw2']], [P['done']], lambda a, b: [SimToken(a)],
                 name='join', reward_function=lambda a, b: r_join)
    ag.add_event([P['p2waitR'], P['R']], [P['R'], P['done']],
                 lambda a, b: [SimToken(b), SimToken(a)],
                 name='priv', reward_function=lambda a, b: r2)
    # independent noisy stream: reward = w, no place shared with d1 -> non-descendant
    ag.add_event([P['cnoise']], [P['done']], lambda p: [SimToken(p)],
                 name='cnoise_done', reward_function=(lambda ww: (lambda b: ww))(w))
    return ag


def d1_return(action_name, make_fn, beta, **rw):
    """Discounted return-to-go seen by d1 = mc_q credit of the first decision."""
    _, creds, forced = run_forced(action_name, make_fn, beta=beta, length=14, **rw)
    assert forced
    return float(creds['mc_q'][0]), creds


if __name__ == "__main__":
    BETA = 0.3

    # ============ Part 1: correctness on the deterministic M2 ============
    print("=" * 70)
    print("Part 1  -- DAG-replay counterfactual vs ccf/lrq on M2 (deterministic)")
    print("=" * 70)
    gA, cA = d1_return('use_R', make_shared_r, BETA, r_join=5, r1=5, r2=10)
    gB, cB = d1_return('standalone', make_shared_r, BETA, r_join=5, r1=5, r2=10)
    cf = gA - gB                                    # DAG-replay: re-run d1 flipped (CRN)
    ref = float(cA['mc_q'][0] - cB['mc_q'][0])      # unbiased reference (= cf, by def)
    print(f"  G(use_R)={gA:+.3f}   G(standalone)={gB:+.3f}")
    print(f"  DAG-replay counterfactual  d1(use_R - standalone) = {cf:+.3f}   "
          f"-> optimal = {'use_R' if cf>0 else 'standalone'}  [UNBIASED]")
    for s in ('lrq', 'ccf', 's_ccf'):
        d = float(cA[s][0] - cB[s][0])
        flip = (d > 0) != (ref > 0) if abs(ref) > 1e-9 else False
        print(f"  {s:>6} d1(A-B) = {d:+.3f}   "
              f"[{'FLIP -> picks wrong action' if flip else 'ok'}]")
    print(f"\n  => the counterfactual scores priv's TIMING change correctly; ccf/lrq,\n"
          f"     which keep/drop priv by realized lineage, flip the decision.")

    # ============ Part 2: payoff -- cancel independent noise, stay unbiased ============
    print("\n" + "=" * 70)
    print("Part 2  -- add independent noisy reward w; counterfactual cancels it")
    print("=" * 70)
    def noisy(w):
        # a make_fn(causal_rl=..., **rw) compatible with run_forced, with w fixed
        return lambda causal_rl=True, **rw: make_shared_r_noisy(w=w, causal_rl=causal_rl, **rw)

    rng = random.Random(0)
    mc_use, cf_adv, ws = [], [], []
    for _ in range(40):
        w = float(rng.randint(0, 20))               # exogenous, action-independent
        mk = noisy(w)
        gA, _ = d1_return('use_R', mk, BETA, r_join=5, r1=5, r2=10)
        gB, _ = d1_return('standalone', mk, BETA, r_join=5, r1=5, r2=10)
        mc_use.append(gA)                            # mc_q return-to-go (carries w)
        cf_adv.append(gA - gB)                       # counterfactual (w cancels, CRN)
        ws.append(w)
    mc_use = np.array(mc_use); cf_adv = np.array(cf_adv)
    # mc_q advantage for use_R = return - baseline(mean over episodes)
    mc_adv = mc_use - mc_use.mean()
    print(f"  over {len(ws)} episodes, exogenous w ~ U{{0..20}} (mean {np.mean(ws):.1f})")
    print(f"  mc_q  advantage(use_R): mean={mc_adv.mean():+.3f}  Var={mc_adv.var():.3f}  "
          f"(carries Var(w)={np.var(ws):.1f})")
    print(f"  DAG-replay counterfactual: mean={cf_adv.mean():+.3f}  Var={cf_adv.var():.3f}  "
          f"({mc_adv.var()/max(cf_adv.var(),1e-9):.0f}x lower)")
    print(f"\n  => the counterfactual advantage is w-noise-free (w is non-descendant of\n"
          f"     d1 -> cancels in G(use_R)-G(standalone)) AND unbiased on priv's timing;\n"
          f"     mc_q is unbiased but carries the full w variance; ccf/lrq are biased.")