r"""Why does cgae_flow score exactly zero greedy return on hard N=4?

Static diagnostic under a random policy: rebuild each scheme's per-decision
credit on random rollouts of make_n_copies_hard(4) and compare the credit
distributions, the row sums R(d), and -- the quantity the policy gradient
actually consumes -- the per-batch STANDARDIZED advantage, split by postpone vs
production decisions. Hypothesis: heavy-tailed cgae_flow credit makes the
standardized advantage of ordinary production actions systematically negative
while postpone (TD credit, bounded) sits near zero, so the argmax drifts to
postpone everywhere.

Run: python _diag_hard_n4_credit.py
"""
import os, sys, random, types, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gympn.environment import AEPN_Env
from ncopies_env import make_n_copies_hard, make_n_copies

LENGTH, EPISODES = 20, 8
SCHEMES = ["cgae_cflow", "cgae_flow", "cgae", "ccf"]

def build(builder):
    pn = builder(); pn.length = LENGTH
    for p in pn.places:
        for t in p.marking: setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn; pn.causal_trace._static_comp_cache = None
    pn.causal_trace.postpone_tokenflow = True; pn.causal_trace.flush()
    sent = types.SimpleNamespace(_id="__initial__")
    for p in pn.places:
        for t in p.marking: pn.causal_trace.register_token(t, sent, parent_tokens=[], time=0)
    pn.causal_trace.register_transition(transition=sent, input_tokens=[],
        output_tokens=[t for p in pn.places for t in p.marking], is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)

def rollout(builder, seed):
    random.seed(seed); np.random.seed(seed)
    env = build(builder); env.reset(); done = False; R = 0.0
    while not done:
        _, r, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions))); R += r
    return env, R

def is_postpone(a):
    s = str(a.get('transition', a.get('name', a.get('transition_name', ''))))
    return 'postpone' in s.lower()

for label, builder in [("hard N=4", lambda: make_n_copies_hard(4, causal_rl=True, allow_postpone=True, causal_postpone_tokenflow=True)),
                       ("base N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True, causal_postpone_tokenflow=True))]:
    print("=" * 70); print(label)
    stats = {s: dict(q=[], adv_post=[], adv_prod=[], nonfinite=0) for s in SCHEMES}
    keys_shown = False
    for ep in range(EPISODES):
        env, R = rollout(builder, 700 + ep)
        ct = env.pn.causal_trace
        acts = ct.transition_history.get_action_transitions()
        if not keys_shown:
            print("  act record keys:", sorted(acts[0].keys())[:12]); keys_shown = True
        n = len(acts); post = np.array([is_postpone(a) for a in acts])
        V = [float(R) / max(n, 1) * 5.0] * n            # crude constant critic of the right order
        for s in SCHEMES:
            q = np.asarray(ct.redistribute_rewards(scheme=s, beta=0.5, values=V, lam=0.95), dtype=float)
            st = stats[s]; st['nonfinite'] += int((~np.isfinite(q)).sum())
            qf = np.where(np.isfinite(q), q, 0.0)
            adv = qf - np.asarray(V); adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # per-batch standardization
            st['q'].extend(qf.tolist()); st['adv_post'].extend(adv[post].tolist()); st['adv_prod'].extend(adv[~post].tolist())
    print(f"  postpone decisions: {int(post.sum())}/{n} in last episode; return {R:.0f}")
    print(f"  {'scheme':<11}{'nonfinite':>10}{'max|q|':>9}{'q p99':>8}{'kurt':>7} | std.adv mean: {'prod':>7}{'post':>7} | frac adv>0: {'prod':>6}{'post':>6}")
    for s in SCHEMES:
        st = stats[s]; q = np.array(st['q']); ap = np.array(st['adv_prod']); ao = np.array(st['adv_post'])
        kurt = float(((q - q.mean()) ** 4).mean() / (q.var() ** 2 + 1e-12))
        print(f"  {s:<11}{st['nonfinite']:>10d}{np.abs(q).max():>9.2f}{np.percentile(np.abs(q),99):>8.2f}{kurt:>7.1f} | "
              f"{ap.mean():>+7.3f}{(ao.mean() if len(ao) else float('nan')):>+7.3f} | {(ap>0).mean():>6.2f}{((ao>0).mean() if len(ao) else float('nan')):>6.2f}")
