r"""Prototype of Option 1: the HINDSIGHT EXOGENOUS BASELINE (fork-free).

Claim: conditioning the value baseline on realized EXOGENOUS (action-independent)
draws removes their variance from the advantage WITHOUT biasing the policy
gradient. This is a strict, always-safe improvement over the standard V(s)
baseline, at no fork cost -- you just feed the critic the recorded exogenous
outcomes in hindsight.

Minimal A-E PN: each case carries an exogenous 'coin' (0/1, drawn at arrival,
BEFORE and independent of the decision). The decision picks action A (value 5) or
B (value 2). The realized reward is value(action) + coin * BONUS -- so the reward
has a large exogenous component (coin*BONUS) that is the same whichever action is
chosen. We compare two baselines on collected rollouts:

  b1(state)        = E[return]            (standard: cannot see the coin)
  b2(state, coin)  = E[return | coin]     (hindsight exogenous baseline)

and report Var(return - b1) vs Var(return - b2) (advantage variance) plus the
per-action mean advantage under each (the policy-gradient signal -- must be
preserved, i.e. unbiased).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import random
import numpy as np
import gympn
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.environment import AEPN_Env

VAL = {'A': 5.0, 'B': 2.0}
BONUS = 10.0


def make_coin(n_parts=4):
    """Finite episode: n_parts cases, each with an exogenous coin, no arrivals."""
    ag = GymProblem(allow_postpone=False, causal_rl=False)
    part = ag.add_var("part", var_attributes=['coin'])
    busyA = ag.add_var("busyA", var_attributes=['coin'])
    busyB = ag.add_var("busyB", var_attributes=['coin'])
    done = ag.add_var("done", var_attributes=['coin'])
    for _ in range(n_parts):
        part.put({'coin': random.randint(0, 1)})
    ag.add_action([part], [busyA], behavior=lambda c: [SimToken(c)], name='A')
    ag.add_action([part], [busyB], behavior=lambda c: [SimToken(c)], name='B')
    ag.add_event([busyA], [done], lambda b: [SimToken(b)], name='cA',
                 reward_function=lambda b: VAL['A'] + b['coin'] * BONUS)
    ag.add_event([busyB], [done], lambda b: [SimToken(b)], name='cB',
                 reward_function=lambda b: VAL['B'] + b['coin'] * BONUS)
    return ag


def collect(n_decisions=3000, seed=0):
    """Random-policy rollouts over finite episodes; record (coin, action, reward)."""
    gympn.seed_everything(seed)
    rows = []
    ep = 0
    while len(rows) < n_decisions:
        random.seed(1000 + ep); np.random.seed(1000 + ep)
        pn = make_coin(); pn.length = 20
        env = AEPN_Env(pn); env.reset(); ep += 1
        for _ in range(30):
            acts = env.pn.pn_actions
            if not acts:
                break
            idx = random.randrange(len(acts))
            b = acts[idx]
            coin = None
            for (place, tok) in b[0]:
                v = getattr(tok, 'value', tok)
                if isinstance(v, dict) and 'coin' in v:
                    coin = v['coin']
            aname = str(getattr(b[2], '_id', getattr(b[2], 'name', '')))
            _, r, d, _, _ = env.step(idx)
            if coin is not None and aname in ('A', 'B'):
                # reward is the net's deterministic function of (action, coin);
                # attribute it to THIS decision (env.step's returned reward is
                # timing-scrambled across concurrent completions).
                rows.append((coin, aname, VAL[aname] + coin * BONUS))
            if d:
                break
    return rows


if __name__ == "__main__":
    rows = collect()
    coin = np.array([c for c, _, _ in rows])
    act = np.array([a for _, a, _ in rows])
    ret = np.array([r for _, _, r in rows])
    print(f"collected {len(rows)} decisions | reward mean={ret.mean():.2f} std={ret.std():.2f}")

    # b1: state-only baseline (state is identical across cases -> just the mean)
    b1 = ret.mean()
    # b2: hindsight exogenous baseline E[return | coin]
    b2 = np.array([ret[coin == c].mean() for c in coin])

    adv1 = ret - b1
    adv2 = ret - b2
    print(f"\nadvantage variance:")
    print(f"  standard baseline  b1(state)       : Var={adv1.var():.3f}")
    print(f"  hindsight exogenous b2(state,coin)  : Var={adv2.var():.3f}   "
          f"({adv1.var()/max(adv2.var(),1e-9):.1f}x lower)")

    # --- CONTROL VARIATE: no learned model, just a known mean + one fitted coeff.
    # A = (return - V(s)) - c*(xi - E[xi]);  xi = coin (recorded exogenous draw),
    # E[xi] = 0.5 (KNOWN fair-coin mean), c = variance-minimizing coefficient.
    E_coin = 0.5                                   # known exogenous mean
    c = np.cov(ret, coin)[0, 1] / np.var(coin)     # fitted: Cov(return, xi)/Var(xi)
    adv_cv = (ret - b1) - c * (coin - E_coin)
    print(f"  control variate  b1 - c*(coin-E[coin]) : Var={adv_cv.var():.3f}   "
          f"({adv1.var()/max(adv_cv.var(),1e-9):.1f}x lower)  [fitted c={c:.2f}, E[coin]={E_coin}]")

    print(f"\nper-action mean advantage (the policy-gradient signal; must be preserved):")
    for a in ('A', 'B'):
        m = act == a
        print(f"  action {a}: standard={adv1[m].mean():+.3f}   hindsight={adv2[m].mean():+.3f}   "
              f"control-variate={adv_cv[m].mean():+.3f}")
    def gap(adv): return adv[act == 'A'].mean() - adv[act == 'B'].mean()
    print(f"  A-B advantage gap: standard={gap(adv1):+.3f}  hindsight={gap(adv2):+.3f}  "
          f"control-variate={gap(adv_cv):+.3f}  (true value gap = {VAL['A']-VAL['B']:+.1f})")