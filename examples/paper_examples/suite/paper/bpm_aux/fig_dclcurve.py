"""Supplement figure: held-out normalized return of one round of DCL and of NF-DCL on the
municipalities as a function of the number of collected days (10, 20, 30, 40), final network,
two training seeds per collection; the ten-collection means of Table 3 at 40 days as hollow
markers; heuristic, lookahead and the best model-free method as reference lines.
Reuses the style of paper/bpm_aux/make_figures.py."""
import json
import os
import sys
from pathlib import Path

import numpy as np

AUX = Path(r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite\paper\bpm_aux")
sys.path.insert(0, str(AUX))
import make_figures as mf  # noqa: E402  (sets rcParams, colours, OUT)
plt = mf.plt

R = Path(r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite\results")
DAYS = [10, 20, 30, 40]
CF = {10: "cf9-6-15-14", 20: "cf18-12-29-28", 30: "cf26-18-44-42", 40: "cf35-24-58-56"}


def finals(d, seeds=(0, 1)):
    out = []
    for s in seeds:
        name = "summary_tau0.25.json" if s == 0 else f"summary_tau0.25_s{s}.json"
        f = R / d / name
        if not f.exists():
            f = R / d / f"summary_tau0.25_s{s}.json"
        out.append(json.load(open(f))["heldout_norm_final"])
    return np.array(out)


dcl = {n: finals(f"pi_bpic15_n1_R8_d{n}_mc_fair_mrg_clkn_sh32") for n in DAYS}
nf = {n: finals(f"pi_bpic15_n1_R8_d{n}_mc_fair_mrg_clkn_sh32_{CF[n]}") for n in DAYS}
rounds = json.load(open(R / "snellius_bpic15_rounds.json"))["rounds"]["1"]
ten = {"DCL": rounds["dcl_mean"], "NF-DCL": rounds["nfdcl_mean"]}
for n in DAYS:
    print(n, "DCL", dcl[n].round(3), dcl[n].mean().round(3), "NF-DCL", nf[n].round(3), nf[n].mean().round(3))
print("ten collections at 40:", {k: round(v, 3) for k, v in ten.items()})

LOOKAHEAD, NFPPO = 1.405, 0.869   # lookahead: planner_heldout_bpic15_n1.jsonl, anchors of the DCL runs
DCLC, NFC = mf.PPO, mf.NFGAE

fig, ax = plt.subplots(figsize=(4.8, 2.7))
x = np.array(DAYS)
for C, color, label, ls, off in ((dcl, DCLC, "DCL", (0, (3, 2)), -0.35), (nf, NFC, "NF-DCL", "-", 0.35)):
    m = np.array([C[n].mean() for n in DAYS])
    ax.plot(x, m, color=color, ls=ls, marker="o", ms=3.5, label=label, zorder=3)
    for n in DAYS:
        ax.scatter([n + off] * len(C[n]), C[n], s=7, color=color, alpha=0.55, lw=0, zorder=2)
ax.scatter([40 - 0.35], [ten["DCL"]], s=28, facecolors="none", edgecolors=DCLC, lw=1.0, zorder=4)
ax.scatter([40 + 0.35], [ten["NF-DCL"]], s=28, facecolors="none", edgecolors=NFC, lw=1.0, zorder=4)
from matplotlib.lines import Line2D  # noqa: E402
proxy = Line2D([], [], marker="o", mfc="none", mec=mf.MUTED, ls="none", ms=5.5, label="ten collections (Table 3)")
ax.set_xlim(7, 43)
ax.set_ylim(0.82, 1.45)
ax.set_xticks(DAYS)
ax.set_xlabel("collected days")
ax.set_ylabel("normalized return (held-out)")
mf.ref_line(ax, 1.0, "heuristic")
mf.ref_line(ax, LOOKAHEAD, "lookahead")
mf.ref_line(ax, NFPPO, "NF-PPO")
h, l = ax.get_legend_handles_labels()
ax.legend(h + [proxy], l + [proxy.get_label()], loc="upper left", frameon=False, fontsize=8,
          handlelength=2.4, borderaxespad=0.3, labelspacing=0.3)
fig.tight_layout()
out = Path(mf.OUT) / "fig_dclcurve.pdf"
fig.savefig(out)
print("written", out)
