"""Rescore finished run_bpm results with paired anchors and a held-out test.

For each results directory:
  1. anchors (random, heuristic) on the 20 evaluation days, so greedy_final is
     normalised against the heuristic ON THE DAYS IT WAS SCORED ON (run_bpm's
     original anchors use 40 other days, 1000-1039);
  2. anchors and every cell's best_policy.pth (and final_policy.pth, when the
     run saved it) on 100 held-out days (envs/heldout.py) that no training,
     tuning or selection has seen.
Writes <dir>/eval_anchors.json, <dir>/heldout.json and prints, per method:
greedy_final with the original anchors, greedy_final with paired anchors, and
the held-out scores (all normalised: random 0, heuristic 1 on the same days).

usage: python scripts/rescore_heldout.py <results dir name> [...] [workers=4]
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))

from heldout import (make_env, anchors, score_policy, EVAL_SEED, EVAL_DAYS,  # noqa: E402
                     HELDOUT_SEED, HELDOUT_DAYS)

LENGTH = 20


def registries():
    from bpm_envs import BPM_BUILDERS, BPM_HEURISTICS
    from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS
    from bpic15_env import BPIC15_BUILDERS, BPIC15_HEURISTICS
    return ({**BPM_BUILDERS, **INSURER_BUILDERS, **BPIC15_BUILDERS},
            {**BPM_HEURISTICS, **INSURER_HEURISTICS, **BPIC15_HEURISTICS})


def settings(d):
    m = re.match(r"bpm_(.+?)_n(\d+)_ep", d.name)
    env, n = m.group(1), int(m.group(2))
    pp = "_nopp" not in d.name
    return dict(env=env, n=n, allow_postpone=pp,
                postpone_scope='component' if "_ppc" in d.name else 'global')


def cell_args(train_dir):
    """The run's own settings, from its args.txt (flag, value lines)."""
    lines = (train_dir / "args.txt").read_text().splitlines()
    a = dict(zip(lines[::2], lines[1::2]))
    t = lambda k: a.get(k, "False") == "True"
    gv = a.get("--global_view", "edges")
    return dict(flat_obs=t("--flat_obs"), marking_certificate=t("--marking_certificate"),
                global_view=False if gv in ("False", "false", "") else gv,
                component_agents=t("--component_agents"), local_obs=t("--local_obs"))


def _score_cell(payload):
    import torch
    torch.set_num_threads(1)
    d, cell, st = payload
    builders, _ = registries()
    tr = d / "train" / cell
    ca = cell_args(tr)
    if ca["component_agents"] or ca["local_obs"]:
        return cell, None, "component agents / local turns: not scored here"
    env = make_env(builders[st["env"]], st["n"], LENGTH, st["allow_postpone"], st["postpone_scope"],
                   ca["flat_obs"], ca["global_view"], ca["marking_certificate"])
    out = {}
    for name in ("best_policy.pth", "final_policy.pth"):
        p = tr / name
        if p.exists():
            pol = torch.load(p, weights_only=False).eval()
            out[name.split("_")[0]] = score_policy(env, pol, HELDOUT_SEED, HELDOUT_DAYS)
    return cell, out, None


def rescore(d, workers):
    st = settings(d)
    builders, heur = registries()
    env = make_env(builders[st["env"]], st["n"], LENGTH, st["allow_postpone"], st["postpone_scope"])
    fa = d / "eval_anchors.json"
    if fa.exists():
        ev, ho = json.loads(fa.read_text())["eval"], json.loads(fa.read_text())["heldout"]
    else:
        ev = anchors(env, heur[st["env"]], EVAL_SEED, EVAL_DAYS)
        ho = anchors(env, heur[st["env"]], HELDOUT_SEED, HELDOUT_DAYS)
        fa.write_text(json.dumps({"eval": ev, "heldout": ho}, indent=1))
    base = json.loads(next(d.glob("baselines_N*.json")).read_text())

    cells = sorted(p.stem.split("__", 1)[1] for p in (d / "cells").glob("*.json"))
    fh = d / "heldout.json"
    held = json.loads(fh.read_text()) if fh.exists() else {}
    todo = [c for c in cells if c not in held and (d / "train" / c).exists()]
    if todo:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for cell, out, err in ex.map(_score_cell, [(d, c, st) for c in todo]):
                if err:
                    print(f"  {cell}: skipped ({err})")
                    continue
                held[cell] = out
                fh.write_text(json.dumps(held, indent=1))

    norm = lambda v, a: (v - a["random_mean"]) / (a["heuristic_mean"] - a["random_mean"])
    print(f"\n{d.name}")
    print(f"  anchors random/heuristic: original days {base['random_mean']:.2f}/{base['heuristic_mean']:.2f} | "
          f"eval days {ev['random_mean']:.2f}/{ev['heuristic_mean']:.2f} | "
          f"held-out days {ho['random_mean']:.2f}/{ho['heuristic_mean']:.2f}")
    print(f"  heuristic on the eval days, original normalisation: {norm(ev['heuristic_mean'], base):.3f}")
    print(f"  {'method':<8} {'n':>2} {'final(orig)':>11} {'final(paired)':>13} {'best ckpt held-out':>18} "
          f"{'final ckpt held-out':>19}")
    by_m = {}
    for c in cells:
        m = c.rsplit("__s", 1)[0]
        j = json.loads((d / "cells" / f"N{st['n']}__{c}.json").read_text())
        if j.get("greedy_final") is None:
            continue
        r = by_m.setdefault(m, {"orig": [], "paired": [], "best": [], "final": []})
        r["orig"].append(norm(j["greedy_final"], base))
        r["paired"].append(norm(j["greedy_final"], ev))
        h = held.get(c) or {}
        if "best" in h:
            r["best"].append(norm(np.mean(h["best"]), ho))
        if "final" in h:
            r["final"].append(norm(np.mean(h["final"]), ho))
    fmt = lambda xs: f"{np.mean(xs):.3f}" if xs else "-"
    for m, r in by_m.items():
        print(f"  {m:<8} {len(r['orig']):>2} {fmt(r['orig']):>11} {fmt(r['paired']):>13} "
              f"{fmt(r['best']):>18} {fmt(r['final']):>19}")
    return by_m


if __name__ == "__main__":
    dirs = [a for a in sys.argv[1:] if "=" not in a]
    w = int(dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a).get("workers", 4))
    for name in dirs:
        rescore(SUITE / "results" / name, w)
