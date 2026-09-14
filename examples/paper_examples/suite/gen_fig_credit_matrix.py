r"""Figure: the credit matrix -- what each decision is credited for.

Three claims in Section 5 are currently numbers in prose:
  * causal credit CUTS total credit mass relative to return-to-go
    (sum_d Q[d]/G falls from ~21 to ~4 at four copies);
  * the scoping ratio S is the fraction of the remaining episode a decision is
    credited over;
  * the estimator DISCOVERS the component structure -- K = 3.2 reward-bearing
    components at four copies, K = 1.0 on the single-component environment.

All three are properties of one object: the matrix c_d(j), the weight decision
d places on reward j. This draws it.

C is computed by the same recursion the library runs, on the library's OWN
succ/w_edge/owned (via _cgae_structure), and then VERIFIED: C @ owned must
reproduce redistribute_rewards' output to floating point. The assertion is the
point -- the figure is checked against the real estimator, not re-derived and
hoped for.

Design choices, and why:
  * FORM. A weight over (decision, reward) pairs is a matrix -> heatmap. The
    claim is about PATTERN (density, block structure), which is what a heatmap
    shows and a table of summary statistics cannot.
  * SEQUENTIAL colour, one hue light->dark, because the encoded quantity is a
    magnitude in [0,1]. Never a rainbow. One shared ramp across all panels, so
    panels are comparable; panel identity is carried by the title, not by hue.
  * Decisions are PERMUTED by causal component, disclosed in the caption. The
    permutation is the same in every panel of a row, so (a) and (b) differ only
    by estimator.
  * Greyscale-safe by construction: a single-hue ramp is monotone in lightness.

Run: python gen_fig_credit_matrix.py
"""
import os
import random
import types
import uuid

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces as CausalTrace
from envs import make_env
from ncopies_env import make_n_copies

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

# Single hue, light -> dark: magnitude, not identity. Monotone in lightness, so
# it survives greyscale printing unchanged.
CMAP = LinearSegmentedColormap.from_list(
    "credit", ["#f4faf8", "#a8ded0", "#3fae8e", "#00785a", "#00402f"], N=256)

LENGTH = 20
OUT = "figures"
SEED = 302

CAPTURE = {}
_orig = CausalTrace._redistribute_cgae


def _spy(self, action_transitions, token_to_action, record_to_action,
         redistribution, beta, get_parents, *a, **kw):
    succ, w_edge, owned, times = self._cgae_structure(
        action_transitions, token_to_action, record_to_action, get_parents)
    CAPTURE.update(succ=succ, w_edge=dict(w_edge), owned=list(owned),
                   times=list(times), n=len(action_transitions))
    return _orig(self, action_transitions, token_to_action, record_to_action,
                 redistribution, beta, get_parents, *a, **kw)


CausalTrace._redistribute_cgae = _spy


def build(builder):
    pn = builder()
    pn.length = LENGTH
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn
    pn.causal_trace._static_comp_cache = None
    pn.causal_trace.postpone_tokenflow = True
    pn.causal_trace.flush()
    sent = types.SimpleNamespace(_id="__initial__")
    for p in pn.places:
        for t in p.marking:
            pn.causal_trace.register_token(t, sent, parent_tokens=[], time=0)
    pn.causal_trace.register_transition(
        transition=sent, input_tokens=[],
        output_tokens=[t for p in pn.places for t in p.marking],
        is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)


def rollout(builder, seed):
    random.seed(seed)
    env = build(builder)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions)))
    return env


def credit_matrix(ct, n):
    """C[d, e] = weight decision d places on the reward mass owned by e.

    Same recursion as _redistribute_cgae with convex=True at lam=1, V=0,
    beta=0, but carried on basis vectors instead of scalars -- the recursion is
    linear in `owned`, so this is exact rather than an approximation.
    """
    succ, w_edge = CAPTURE['succ'], CAPTURE['w_edge']
    times = CAPTURE['times']
    order = sorted(range(n), key=lambda i: (times[i] is not None,
                                            times[i] or 0.0), reverse=True)
    C = np.zeros((n, n))
    for d in order:
        C[d, d] += 1.0
        kids = [s for s in succ.get(d, ()) if s != d and s < n]
        if not kids:
            continue
        w = np.array([w_edge.get((d, s), 0.0) for s in kids])
        R = w.sum()
        what = (w / R) if R > 0 else np.full(len(kids), 1.0 / len(kids))
        for s, ws in zip(kids, what):
            C[d] += ws * C[s]
    return C


def mcq_matrix(ct, n):
    """C[d, e] = 1 where e's reward time is at or after d's clock (beta=0)."""
    times = CAPTURE['times']
    C = np.zeros((n, n))
    for d in range(n):
        for e in range(n):
            ud, te = times[d], times[e]
            if ud is None or te is None or ud <= te:
                C[d, e] = 1.0
    return C


def components(n):
    """Connected components of the (undirected) causal graph."""
    succ = CAPTURE['succ']
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d, kids in succ.items():
        for s in kids:
            if d < n and s < n:
                a, b = find(d), find(s)
                if a != b:
                    parent[a] = b
    return [find(i) for i in range(n)]


CASES = [
    ("(b) cgae-cf, four copies", "ncopies",
     lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                           causal_postpone_tokenflow=True)),
    ("(c) cgae-cf, one component", "s1",
     lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                      causal_postpone_tokenflow=True)),
]

os.makedirs(OUT, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))

panels = []

# --- four copies: mc_q and cgae-cf on the SAME permutation ---------------- #
env = rollout(CASES[0][2], SEED)
ct = env.pn.causal_trace
n = len(ct.transition_history.get_action_transitions())
V = [0.0] * n
q_lib = np.asarray(ct.redistribute_rewards(scheme='cgae_cflow', beta=0.0,
                                           values=V, lam=1.0), float)
owned = np.asarray(CAPTURE['owned'], float)
C = credit_matrix(ct, n)
err = float(np.max(np.abs(C @ owned - q_lib)))
assert err < 1e-9, "credit matrix disagrees with the library: %g" % err
print("four copies : n=%d  max|C@owned - library Q| = %.2e  (verified)" % (n, err))

comp = components(n)
perm = sorted(range(n), key=lambda i: (comp[i], CAPTURE['times'][i] or 0.0))
Cm = mcq_matrix(ct, n)
nK = len(set(comp[i] for i in range(n) if abs(owned[i]) > 0))
print("four copies : reward-bearing components K = %d" % nK)
print("four copies : credit mass  mc_q %.2f  cgae-cf %.2f  (x%.1f reduction)"
      % (Cm.sum() / max(1, (owned != 0).sum()), C.sum() / max(1, (owned != 0).sum()),
         Cm.sum() / max(C.sum(), 1e-9)))

panels.append(("(a) mc-q, four copies", Cm[np.ix_(perm, perm)]))
panels.append((CASES[0][0], C[np.ix_(perm, perm)]))

# --- single component ----------------------------------------------------- #
env = rollout(CASES[1][2], SEED)
ct = env.pn.causal_trace
n2 = len(ct.transition_history.get_action_transitions())
V = [0.0] * n2
q_lib = np.asarray(ct.redistribute_rewards(scheme='cgae_cflow', beta=0.0,
                                           values=V, lam=1.0), float)
owned2 = np.asarray(CAPTURE['owned'], float)
C2 = credit_matrix(ct, n2)
err2 = float(np.max(np.abs(C2 @ owned2 - q_lib)))
assert err2 < 1e-9, "credit matrix disagrees with the library: %g" % err2
print("single comp : n=%d  max|C@owned - library Q| = %.2e  (verified)" % (n2, err2))
comp2 = components(n2)
perm2 = sorted(range(n2), key=lambda i: (comp2[i], CAPTURE['times'][i] or 0.0))
nK2 = len(set(comp2[i] for i in range(n2) if abs(owned2[i]) > 0))
print("single comp : reward-bearing components K = %d" % nK2)
panels.append((CASES[1][0], C2[np.ix_(perm2, perm2)]))

for ax, (title, M) in zip(axes, panels):
    im = ax.imshow(M, cmap=CMAP, vmin=0.0, vmax=1.0, interpolation="nearest",
                   aspect="equal")
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlabel("reward owned by decision")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#999999")
axes[0].set_ylabel("credited decision")

cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.015)
cb.set_label("credit weight $c_d(j)$", fontsize=8)
cb.outline.set_linewidth(0.4)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_credit_matrix.%s" % ext))
print("wrote %s/fig_credit_matrix.pdf" % OUT)
