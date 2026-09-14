r"""TRAINING-LEVEL test: does the DAG-replay counterfactual advantage actually
LEARN better than PPO(mc_q) / ccf / s_ccf?

We build a composite decision problem that mixes the two motifs where filtering
fails, plus exogenous noise:
  * M2-type sites (shared-resource join): ccf is BIASED -> it prefers the wrong
    action (use_R over standalone). Optimal action = standalone.
  * M4-type sites (abundant shared resource): s_ccf is CONSERVATIVE -> it keeps
    the independent noisy reward r_b, inflating variance. Optimal action = hi.
  * a global exogenous reward w that only the unfactored mc_q return carries.

Each site is an independent binary decision with its own logit; we run REINFORCE
with four advantage estimators and watch how fast / how correctly each learns.

The per-site credit rules are taken from the LIBRARY (assembly_probe.run_forced)
at startup -- not hardcoded -- so this is driven by the real schemes:
  mc_q : credit = FULL episode return (all sites + w)          [unbiased, high var]
  ccf  : credit = realized-lineage factored per-site credit    [BIASED on M2]
  s_ccf: credit = static-component factored per-site credit     [keeps r_b on M4]
  cf   : advantage = G_site(a) - E_{a'~pi}[G_site(a')]          [DAG-replay COMA baseline]

Prediction: cf learns all sites correctly and fast; ccf converges to the WRONG
action on M2 sites; s_ccf/mc_q are correct but slower (they carry r_b / full-return
variance). cf is the only one both unbiased and low-variance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
from assembly_probe import make_shared_r, make_two_chains, run_forced

BETA_M2 = 0.3
R_HI, R_LO = 5.0, 2.0
NOISE_HI = 40            # exogenous draws ~ U{0..NOISE_HI} (large -> variance dominates)


def library_rules():
    """Pull the per-site credit rules from the real library. Returns closures."""
    m2 = {}
    for act in ('use_R', 'standalone'):
        _, c, _ = run_forced(act, make_shared_r, beta=BETA_M2, length=14, r_join=5, r1=5, r2=10)
        m2[act] = {s: float(c[s][0]) for s in ('mc_q', 'ccf', 's_ccf')}
    m4 = {}
    for act in ('A_hi', 'A_lo'):
        _, c, _ = run_forced(act, make_two_chains, beta=0.0, length=12, r_hi=R_HI, r_lo=R_LO, r_b=0.0)
        m4[act] = {s: float(c[s][0]) for s in ('mc_q', 'ccf', 's_ccf')}   # at r_b=0
    return m2, m4


class Site:
    """One decision. action 1 = OPTIMAL (standalone / hi), action 0 = suboptimal."""
    def __init__(self, kind, m2, m4):
        self.kind = kind; self.m2 = m2; self.m4 = m4

    def draw_exo(self, rng):
        return float(rng.integers(0, NOISE_HI + 1)) if self.kind == 'M4' else 0.0

    def _act(self, a):   # library action name for chosen a
        if self.kind == 'M2':
            return 'standalone' if a == 1 else 'use_R'
        return 'A_hi' if a == 1 else 'A_lo'

    def site_return(self, a, exo):
        """mc_q's contribution of this site to the full return (carries r_b)."""
        base = self.m2[self._act(a)]['mc_q'] if self.kind == 'M2' else self.m4[self._act(a)]['mc_q']
        return base + (exo if self.kind == 'M4' else 0.0)

    def ccf_credit(self, a, exo):
        base = self.m2[self._act(a)]['ccf'] if self.kind == 'M2' else self.m4[self._act(a)]['ccf']
        return base + 0.0                                  # ccf drops r_b on M4 (no exo term)

    def s_ccf_credit(self, a, exo):
        base = self.m2[self._act(a)]['s_ccf'] if self.kind == 'M2' else self.m4[self._act(a)]['s_ccf']
        return base + (exo if self.kind == 'M4' else 0.0)  # s_ccf keeps r_b on M4

    def G_site(self, a, exo):
        """Counterfactual replay of THIS site under action a (same exo = CRN)."""
        return self.site_return(a, exo)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train(scheme, sites, updates=500, batch=16, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros(len(sites))
    m2_idx = [i for i, s in enumerate(sites) if s.kind == 'M2']
    m4_idx = [i for i, s in enumerate(sites) if s.kind == 'M4']
    curve, c_m2, c_m4, gvar = [], [], [], []
    for t in range(updates):
        p = sigmoid(theta)
        A = rng.random((batch, len(sites))); a = (A < p).astype(float)  # sample actions
        exo = np.array([[s.draw_exo(rng) for s in sites] for _ in range(batch)])
        w = rng.integers(0, NOISE_HI + 1, size=batch).astype(float)     # global noise

        # per-episode, per-site credit under the scheme
        cred = np.zeros((batch, len(sites)))
        adv = np.zeros((batch, len(sites)))
        # full return for mc_q
        site_ret = np.array([[s.site_return(a[b, i], exo[b, i]) for i, s in enumerate(sites)]
                             for b in range(batch)])
        R = site_ret.sum(axis=1) + w                                    # full episode return
        for i, s in enumerate(sites):
            if scheme == 'mc_q':
                cred[:, i] = R                                          # every decision: full return
            elif scheme == 'ccf':
                cred[:, i] = [s.ccf_credit(a[b, i], exo[b, i]) for b in range(batch)]
            elif scheme == 's_ccf':
                cred[:, i] = [s.s_ccf_credit(a[b, i], exo[b, i]) for b in range(batch)]
            elif scheme == 'cf':
                # COMA counterfactual baseline: A = G(a) - E_{a'~pi}[G(a')]  (DAG-replay)
                g_taken = np.array([s.G_site(a[b, i], exo[b, i]) for b in range(batch)])
                g1 = np.array([s.G_site(1, exo[b, i]) for b in range(batch)])
                g0 = np.array([s.G_site(0, exo[b, i]) for b in range(batch)])
                adv[:, i] = g_taken - (p[i] * g1 + (1 - p[i]) * g0)
        if scheme != 'cf':
            # learned baseline: running mean of the credit (per site; scalar for mc_q)
            b_hat = cred.mean(axis=0)
            adv = cred - b_hat

        score = a - p                                                   # d logpi / d theta
        grad = (score * adv).mean(axis=0)
        gvar.append(float((score * adv).var()))
        theta += lr * grad
        # fraction of sites whose policy now favors the OPTIMAL action (a=1)
        opt = sigmoid(theta) > 0.5
        curve.append(float(opt.mean()))
        c_m2.append(float(opt[m2_idx].mean())); c_m4.append(float(opt[m4_idx].mean()))
    return (np.array(curve), np.array(c_m2), np.array(c_m4), float(np.mean(gvar[-50:])))


if __name__ == "__main__":
    m2, m4 = library_rules()
    print("library-derived per-site rules:")
    print(f"  M2  use_R:      {m2['use_R']}")
    print(f"  M2  standalone: {m2['standalone']}   (ccf prefers use_R -> BIASED)")
    print(f"  M4  A_hi(rb=0):  {m4['A_hi']}    A_lo(rb=0): {m4['A_lo']}")

    NM2, NM4 = 5, 5
    sites = [Site('M2', m2, m4) for _ in range(NM2)] + [Site('M4', m2, m4) for _ in range(NM4)]
    print(f"\ncomposite: {NM2} M2-sites (ccf biased) + {NM4} M4-sites (s_ccf conservative) + global noise")
    print(f"optimal = action 1 on every site (standalone / hi); target correctness = 1.0\n")

    def first_at(curve, thr):
        idx = np.argmax(curve >= thr)
        return int(idx) if curve[idx] >= thr else -1

    results = {}
    for scheme in ('mc_q', 'ccf', 's_ccf', 'cf'):
        curves, m2c, m4c, gv = [], [], [], []
        for sd in range(12):
            c, cm2, cm4, g = train(scheme, sites, seed=sd)
            curves.append(c); m2c.append(cm2); m4c.append(cm4); gv.append(g)
        results[scheme] = (np.mean(curves, 0), np.mean(m2c, 0), np.mean(m4c, 0), np.mean(gv))

    print(f"  {'scheme':>7} | {'M2ok':>5} {'M4ok':>5} {'allok':>5} | {'upd>0.9(M4)':>11} | grad-var")
    print("  " + "-" * 60)
    for scheme in ('mc_q', 'ccf', 's_ccf', 'cf'):
        c, cm2, cm4, g = results[scheme]
        t90 = first_at(cm4, 0.9)
        t90s = f"{t90}" if t90 >= 0 else ">500"
        print(f"  {scheme:>7} | {cm2[-1]:5.2f} {cm4[-1]:5.2f} {c[-1]:5.2f} | {t90s:>11} | {g:8.2f}")

    print("\n  M2ok/M4ok = final correctness on each motif; upd>0.9(M4) = updates for M4")
    print("  sites to reach 90% correct (the variance-limited race between cf and s_ccf).")
    print("  Expect: ccf M2ok=0.00 (biased); cf reaches M4 0.9 fastest; s_ccf/mc_q slower.")