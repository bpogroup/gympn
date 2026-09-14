"""Generate the paper figures (PDF vector + PNG preview) into figures/.

  fig1_multisite_curves : realistic env — ccf learns, PPO is flat (12 seeds, CIs)
  fig2_ncopies_scaling  : ccf stays optimal / PPO degrades as components grow
  fig3_mechanism        : lineage factoring |ccf-mcq| grows with independence
  fig4_boundary_summary : the 3 boundary-condition mechanisms vs baseline, s1
  fig5_phi_dial         : the phi-shaping coefficient dial (norm_final + reach-rate)

fig4/fig5 read the actual cell JSONs already on disk under
suite_results_{phi,struct,gctx}_s1/cells/ (same convention as fig1/fig2 --
no hardcoded summary numbers), matching the boundary-condition investigation
in EJOR_BOUNDARY_CONDITIONS.md / paper §6.

Run from the suite dir: python generate_figures.py
"""
import glob
import json
import os
import random
import sys
import types
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 9,
})

# Okabe-Ito colorblind-safe: PPO vermillion, ccf bluish-green
C_PPO, C_CCF = "#D55E00", "#009E73"
LBL = {"ppo": "PPO (discount-matched)", "ccf": "ccf (ours)"}
COL = {"ppo": C_PPO, "ccf": C_CCF}
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf (+ .png)")


def _ci(a, axis=0):
    a = np.asarray(a)
    n = a.shape[axis]
    return 1.96 * a.std(axis=axis, ddof=1) / np.sqrt(n)


# --------------------------------------------------------------------------
# Figure 1 — multi-site learning curves (12 seeds, 95% CI bands)
# --------------------------------------------------------------------------
def fig1_multisite():
    d = "suite_results_multisite_bf"
    by, base = {}, None
    for f in glob.glob(f"{d}/cells/*.json"):
        m = json.load(open(f))
        base = m["baselines"]
        by.setdefault(m["method"], []).append((m["greedy_epochs"], m["greedy_curve"]))
    r, h = base["random_mean"], base["heuristic_mean"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for meth in ("ppo", "ccf"):
        curves = by[meth]
        L = min(len(c[1]) for c in curves)
        ep = np.array(curves[0][0][:L])
        arr = np.array([[(v - r) / (h - r) for v in c[1][:L]] for c in curves])
        mean, ci = arr.mean(0), _ci(arr, 0)
        ax.plot(ep, mean, color=COL[meth], lw=2, label=LBL[meth], zorder=3)
        ax.fill_between(ep, mean - ci, mean + ci, color=COL[meth], alpha=0.18, lw=0)
    ax.axhline(1.0, ls="--", c="0.4", lw=1)
    ax.text(ep[-1], 1.0, " heuristic", va="center", ha="left", fontsize=8, color="0.4")
    ax.axhline(0.0, ls=":", c="0.6", lw=1)
    ax.text(ep[-1], 0.0, " random", va="center", ha="left", fontsize=8, color="0.6")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("normalized greedy return")
    ax.set_title("Multi-site skills-based routing (4 sites, 12 seeds)", fontsize=10)
    ax.set_xlim(ep[0], ep[-1] + 4)
    ax.legend(loc="upper left")
    _save(fig, "fig1_multisite_curves")


# --------------------------------------------------------------------------
# Figure 2 — N-copies scaling: normalized final vs N (ccf stable / PPO degrades)
# --------------------------------------------------------------------------
def fig2_ncopies():
    NS = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for meth in ("ppo", "ccf"):
        mus, cis = [], []
        for n in NS:
            vals = []
            for f in glob.glob(f"suite_results_ncopies/cells/N{n}__{meth}__s*.json"):
                m = json.load(open(f))
                b = m["baselines"]; r, h = b["random_mean"], b["heuristic_mean"]
                if m.get("greedy_final") is not None:
                    vals.append((m["greedy_final"] - r) / (h - r))
            mus.append(np.mean(vals)); cis.append(_ci(vals))
        ax.errorbar(NS, mus, yerr=cis, marker="o", ms=5, color=COL[meth],
                    label=LBL[meth], capsize=3, lw=2, zorder=3)
    ax.axhline(1.0, ls="--", c="0.4", lw=1)
    ax.text(8, 1.0, " optimum", va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("number of independent components $N$")
    ax.set_ylabel("normalized final return")
    ax.set_title("Scaling: ccf holds the optimum, PPO degrades", fontsize=10)
    ax.legend(loc="lower left")
    _save(fig, "fig2_ncopies_scaling")


# --------------------------------------------------------------------------
# Figure 3 — mechanism: lineage factoring |ccf-mcq| grows with component count
# --------------------------------------------------------------------------
def _factoring(n, seed):
    from gympn.environment import AEPN_Env
    from ncopies_env import make_n_copies
    random.seed(seed); np.random.seed(seed)
    pn = make_n_copies(n, causal_rl=True, allow_postpone=False); pn.length = 20
    for p in pn.places:
        for t in p.marking:
            setattr(t, "_id", str(uuid.uuid4()))
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn); env.reset()
    for _ in range(50):
        k = len(env.pn.pn_actions)
        if k == 0:
            break
        _, _, done, _, _ = env.step(np.random.randint(k))
        if done:
            break
    ct = env.pn.causal_trace
    ccf = np.array(ct.redistribute_rewards(scheme="ccf", beta=0.5))
    mcq = np.array(ct.redistribute_rewards(scheme="mc_q", beta=0.5))
    return float(np.mean(np.abs(ccf - mcq)) / max(1e-9, np.mean(np.abs(mcq))))


def fig3_mechanism():
    NS = [1, 2, 4, 8]
    mus, cis = [], []
    for n in NS:
        vals = [_factoring(n, s) for s in range(6)]
        mus.append(np.mean(vals)); cis.append(_ci(vals))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar(NS, mus, yerr=cis, marker="s", ms=5, color="#0072B2",
                capsize=3, lw=2, zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of independent components $N$")
    ax.set_ylabel(r"credit divergence  $|\mathrm{ccf}-\mathrm{mc\_q}|/\mathrm{scale}$")
    ax.set_title("Mechanism: lineage factoring grows with independence", fontsize=10)
    _save(fig, "fig3_mechanism_factoring")


# --------------------------------------------------------------------------
# Shared helpers for the boundary-condition figures (fig4/fig5): load real
# cell JSONs from suite_results_{phi,struct,gctx}_s1/cells/, same convention
# as fig1/fig2 above -- no hardcoded summary numbers.
# --------------------------------------------------------------------------
ENV_S1 = "s1_stoch_sequence"


def _baselines(result_dir):
    b = json.load(open(f"{result_dir}/baselines_{ENV_S1}.json"))
    return b["random_mean"], b["heuristic_mean"]


def _arm_norms(result_dir, arm):
    r, h = _baselines(result_dir)
    vals = []
    for f in sorted(glob.glob(f"{result_dir}/cells/{ENV_S1}__{arm}__s*.json")):
        m = json.load(open(f))
        gf = m.get("greedy_final")
        if gf is not None:
            vals.append((gf - r) / (h - r))
    return vals


def _arm_reach_rate(result_dir, arm, threshold=0.95):
    """Fraction of seeds whose greedy_curve ever reaches `threshold` of the
    heuristic anchor, same convention as run_*_s1_full.py's own reporting."""
    r, h = _baselines(result_dir)
    target = threshold * h
    n_total, n_hit = 0, 0
    for f in sorted(glob.glob(f"{result_dir}/cells/{ENV_S1}__{arm}__s*.json")):
        m = json.load(open(f))
        curve, epochs = m.get("greedy_curve", []), m.get("greedy_epochs", [])
        if not curve:
            continue
        n_total += 1
        if any(v >= target for v in curve):
            n_hit += 1
    return (n_hit / n_total) if n_total else 0.0, n_hit, n_total


# --------------------------------------------------------------------------
# Figure 4 — boundary-condition summary: baseline vs each of the 3 tested
# mechanisms on the s1 bottleneck (K=1) environment (EJOR_BOUNDARY_CONDITIONS.md)
# --------------------------------------------------------------------------
def fig4_boundary_summary():
    groups = [
        ("Reward shaping\n($\\phi_{\\mathrm{coef}}{=}1.0$)", "suite_results_phi_s1", "ppo_clip", "ppo_clip_phi"),
        ("Structural\nfeatures", "suite_results_struct_s1", "ppo_clip", "ppo_clip_struct"),
        ("Actor global\ncontext", "suite_results_gctx_s1", "ppo_clip", "ppo_clip_gctx"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    width = 0.32
    x = np.arange(len(groups))
    base_mus, base_cis, int_mus, int_cis = [], [], [], []
    for _, d, base_arm, int_arm in groups:
        bv, iv = _arm_norms(d, base_arm), _arm_norms(d, int_arm)
        base_mus.append(np.mean(bv)); base_cis.append(_ci(bv))
        int_mus.append(np.mean(iv)); int_cis.append(_ci(iv))
    ax.bar(x - width / 2, base_mus, width, yerr=base_cis, capsize=3,
           color="#0072B2", label="baseline PPO", zorder=3)
    ax.bar(x + width / 2, int_mus, width, yerr=int_cis, capsize=3,
           color="#D55E00", label="+ mechanism", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("normalized final return")
    ax.set_title("Boundary-condition investigation: s1 bottleneck ($K{=}1$)", fontsize=10)
    ax.axhline(base_mus[0], ls=":", c="0.6", lw=1, zorder=1)
    ax.legend(loc="upper right")
    _save(fig, "fig4_boundary_summary")


# --------------------------------------------------------------------------
# Figure 5 — the phi-shaping coefficient dial: norm_final and reach-95%-rate
# both degrade monotonically past coef=0.2 (EJOR_BOUNDARY_CONDITIONS.md §6b.1)
# --------------------------------------------------------------------------
def fig5_phi_dial():
    d = "suite_results_phi_s1"
    dial = [(0.0, "ppo_clip"), (0.2, "ppo_clip_phi02"), (0.5, "ppo_clip_phi05"), (1.0, "ppo_clip_phi")]
    coefs = [c for c, _ in dial]
    norm_mus, norm_cis, reach_rates = [], [], []
    for _, arm in dial:
        vals = _arm_norms(d, arm)
        norm_mus.append(np.mean(vals)); norm_cis.append(_ci(vals))
        rate, _, _ = _arm_reach_rate(d, arm)
        reach_rates.append(rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax1.errorbar(coefs, norm_mus, yerr=norm_cis, marker="o", ms=5,
                 color="#D55E00", capsize=3, lw=2, zorder=3)
    ax1.set_xlabel(r"$\phi_{\mathrm{coef}}$")
    ax1.set_ylabel("normalized final return")
    ax1.set_title("Efficacy", fontsize=10)

    ax2.plot(coefs, reach_rates, marker="s", ms=5, color="#0072B2", lw=2, zorder=3)
    ax2.set_xlabel(r"$\phi_{\mathrm{coef}}$")
    ax2.set_ylabel("fraction of seeds reaching 95% of anchor")
    ax2.set_ylim(0, 1)
    ax2.set_title("Reach-rate", fontsize=10)

    fig.suptitle("Reward shaping: safe (theorem) but not efficacious past coef=0.2", fontsize=10)
    _save(fig, "fig5_phi_dial")


if __name__ == "__main__":
    print("generating figures ...")
    fig1_multisite()
    fig2_ncopies()
    fig3_mechanism()
    fig4_boundary_summary()
    fig5_phi_dial()
    print("done -> figures/")