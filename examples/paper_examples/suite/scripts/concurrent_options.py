"""Decisions of several components at the same state, over the labeled decisions of the
DCL collections (whole-net days) of Table 3 of the paper: the share of decisions at which
another component also has an enabled option, the share of the enabled option nodes that
belong to other components, the number of components with options, and the number of
option classes the lookahead compared. Postpone (wait) nodes count as options. The
insurer's 40-day collections are on Snellius; its 4-day local probes are used instead.
Writes results/concurrent_options.json and prints the supplement's table rows.

usage: python scripts/concurrent_options.py
"""
import json
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[1] / "results"
RUNS = [("Insurer, 1 region", "pi_insurer_n1_R8_d4_mc_fair_mrg_clkn_sh32"),
        ("Insurer, 2 regions", "pi_insurer_n2_R8_d4_mc_fair_mrg_clkn_sh32"),
        ("Allocation, 1 region", "pi_insurer_alloc_n1_R8_d40_mc_fair_mrg_clkn_sh32"),
        ("Allocation, 2 regions", "pi_insurer_alloc_n2_R8_d40_mc_fair_mrg_clkn_sh32"),
        ("Slow mismatch, 1 region", "pi_insurer_slow_n1_R8_d40_mc_ppc_fair_mrg_clkn_sh32"),
        ("Municipalities", "pi_bpic15_n1_R8_d40_mc_fair_mrg_clkn_sh32")]
out = {}
for name, d in RUNS:
    f = R / d / "decisions.pt"
    G = torch.load(f, weights_only=False)
    n = multi = 0
    other_share = ncomp = nopt = nother = nown = ncls = 0.0
    for g in G:
        idx = torch.cat([g.a_idx.reshape(-1), g.p_idx.reshape(-1)])
        cls = torch.cat([g.t_cls_a.reshape(-1), g.t_cls_p.reshape(-1)])
        comp = g.comp[idx]
        acting = comp[cls >= 0]
        if acting.numel() == 0:
            continue
        other = comp != acting.mode().values
        n += 1
        multi += bool(other.any())
        other_share += float(other.float().mean())
        ncomp += len(torch.unique(comp))
        nopt += idx.numel()
        nother += int(other.sum())
        nown += int((~other).sum())
        ncls += int(g.t_ncls)
    out[name] = {"run": d, "decisions": n, "multi_component": multi / n, "other_share": other_share / n,
                 "components_with_options": ncomp / n, "options": nopt / n, "other_options": nother / n,
                 "own_options": nown / n, "classes": ncls / n}
    r = out[name]
    print(f"{name:27s} & {format(n, ',').replace(',', chr(92) + ',')} & {100*r['multi_component']:.0f} & "
          f"{100*r['other_share']:.0f} & {r['components_with_options']:.2f} & {r['own_options']:.1f} & "
          f"{r['other_options']:.1f} \\\\")
(R / "concurrent_options.json").write_text(json.dumps(out, indent=1))
