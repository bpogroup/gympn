"""Figure: the distribution of the bootstrap row sum R(d).

The paper's formal claim in Section 5 is that the coefficient multiplying the
critic must be BOUNDED. `cgae_flow` normalizes weights over predecessors, which
leaves the row sum R(d) = sum_{s in succ(d)} w(d->s) unconstrained; `cgae_cflow`
divides by R, so its coefficient is exactly 1 by construction. That argument is
currently equations and a table. This draws it.

Design choices, and why:
  * FORM. One quantity's distribution over decisions -> histogram, one panel per
    environment. Not a dual axis (there is one measure, R).
  * The convex variant is a POINT MASS at 1, not a distribution, so it is drawn
    as a labelled rule rather than as a second histogram -- plotting a delta as
    bars would imply a spread it does not have.
  * COLOR by identity, Okabe-Ito, the same assignment gen_fig_curves.py uses.
    Validated (validate_palette.js, light): all checks pass; the contrast WARN on
    #56B4E9 is relieved by the per-panel legend, which both panels carry.
  * PRINT/CVD. Colour is never the only channel: the unbounded mass is hatched
    and the convex rule is dashed, so the figure survives greyscale.
  * The |R-1| > 0.25 mass is shaded because that is the quantity the text
    quotes; its share is printed once per panel, not on every bar.

Run: python gen_fig_rowsum.py
"""
import os
import random
import types
import uuid

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces as CausalTrace
from envs import make_env
from ncopies_env import make_n_copies

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
})

C_FLOW = "#56B4E9"   # sky blue     -- cgae_flow, as in gen_fig_curves.py
C_CFLOW = "#009E73"  # bluish green -- cgae_cflow, the method
INK = "#222222"

LENGTH = 20
EPISODES = 5
OUT = "figures"

CAPTURE = {}
_orig = CausalTrace._redistribute_cgae


def _spy(self, action_transitions, token_to_action, record_to_action,
         redistribution, beta, get_parents, *a, **kw):
    succ, w_edge, owned, times = self._cgae_structure(
        action_transitions, token_to_action, record_to_action, get_parents)
    CAPTURE['succ'] = succ
    CAPTURE['w_edge'] = dict(w_edge)
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


def collect(builder):
    """Row sums R(d) over decisions that have at least one causal successor."""
    out = []
    for ep in range(EPISODES):
        random.seed(300 + ep)
        env = build(builder)
        env.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions)))
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n == 0:
            continue
        ct.redistribute_rewards(scheme='cgae_flow', beta=0.0,
                                values=[0.0] * n, lam=1.0)
        succ, w_edge = CAPTURE['succ'], CAPTURE['w_edge']
        for d, kids in succ.items():
            kids = [s for s in kids if s != d]
            if not kids:
                continue
            out.append(sum(w_edge.get((d, s), 0.0) for s in kids))
    return np.asarray(out, dtype=float)


PANELS = [
    ("(a) single component ($K{=}1$)",
     lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                      causal_postpone_tokenflow=True)),
    ("(b) four copies ($K{=}3.3$)",
     lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                           causal_postpone_tokenflow=True)),
]

os.makedirs(OUT, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

for ax, (title, builder) in zip(axes, PANELS):
    R = collect(builder)
    outside = float(np.mean(np.abs(R - 1.0) > 0.25))
    hi = max(4.0, float(R.max()) * 1.02)
    bins = np.linspace(0.0, hi, 46)

    inside = (np.abs(R - 1.0) <= 0.25)
    ax.hist(R[inside], bins=bins, color=C_FLOW, alpha=0.9, linewidth=0)
    ax.hist(R[~inside], bins=bins, color=C_FLOW, alpha=0.9, linewidth=0.4,
            edgecolor="white", hatch="///")
    ax.axvline(1.0, color=C_CFLOW, linewidth=1.8, dashes=(4, 2), zorder=5)

    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(r"bootstrap row sum $R(d)$")
    ax.set_xlim(0, hi)
    ax.set_ylabel("decisions")

    ax.legend(handles=[
        Patch(facecolor=C_FLOW, alpha=0.9, label=r"cgae-f, within $|R-1|\leq0.25$"),
        Patch(facecolor=C_FLOW, alpha=0.9, hatch="///", edgecolor="white",
              label="cgae-f, outside (%.0f%%)" % (100 * outside)),
        Line2D([0], [0], color=C_CFLOW, lw=1.8, dashes=(4, 2),
               label=r"cgae-cf, $R\equiv1$"),
    ], loc="upper right", fontsize=7.5, handlelength=1.6, borderpad=0.2)

    print("%-34s n=%4d  mean %.3f  SD %.3f  range %.3f..%.3f  outside %.1f%%"
          % (title, len(R), R.mean(), R.std(), R.min(), R.max(), 100 * outside))

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig_rowsum.%s" % ext))
print("wrote %s/fig_rowsum.pdf" % OUT)
