"""Does longer training learn the allocation? Single-seed probe on the
start-of-shift allocation variant (insurer_alloc, 1 region).

Trains one run_bpm cell (same protocol and hyperparameters) for EPOCHS epochs
instead of 40, saving a policy snapshot at every greedy test (every 3 epochs).
Then each snapshot is played greedily on the 20 scenarios of
alloc_behaviour.py: which allocation it picks, how much probability the first
decision puts on each allocation option, and its return.

Run: python scripts/alloc_long_probe.py [epochs=150] [seed=0] [method=nfgae]
                                        [cert=marking|static] [threads=4] [beta=x]
                                        [view=local|full] [critic_mask=1] [abatch=share|full]
Output: results/alloc_long_probe/<method><_mc>_s<seed>_ep<epochs>/
        (train/ snapshots, cell.json, probe.json)
"""
import json
import os
import sys
from collections import Counter

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
EPOCHS = int(opts.get("epochs", 150))
SEED = int(opts.get("seed", 0))
METHOD = opts.get("method", "nfgae")
CERT = opts.get("cert", "marking")
THREADS = int(opts.get("threads", 4))
BETA = float(opts["beta"]) if "beta" in opts else None   # SMDP discount (Hyper.beta if None)
VIEW = opts.get("view", "full")
CRITIC_MASK = opts.get("critic_mask", "0") in ("1", "true", "True")
AGENT_BATCH = opts.get("abatch", "share")   # component agents: share (default) or full batch

# run_bpm parses sys.argv at import: configure it as the 40-epoch cells were.
sys.argv = [sys.argv[0], "env=insurer_alloc", "N=1", f"epochs={EPOCHS}", f"methods={METHOD}",
            "flat=1", "postpone=0", f"cert={CERT}", f"view={VIEW}", f"abatch={AGENT_BATCH}"] + (["critic_mask=1"] if CRITIC_MASK else [])
import run_bpm  # noqa: E402
import numpy as np  # noqa: E402
from gympn.agents import Agent  # noqa: E402

TAG = (f"{METHOD}{'_mc' if CERT == 'marking' and METHOD in ('nfgae', 'agents') else ''}_s{SEED}_ep{EPOCHS}"
       + ("" if BETA is None else f"_beta{BETA:g}") + {"local": "", "full": "_fv"}[VIEW] + ("_cm" if CRITIC_MASK else "")
       + ("_abfull" if AGENT_BATCH == "full" else ""))
OUT = os.path.join(SUITE, "results", "alloc_long_probe", TAG)
TRAIN = os.path.join(OUT, "train")


def _snapshotting(orig):
    """Wrap Agent.test_in_train: also save the policy as snap_ep<epoch>.pth."""
    count = {"n": 0}

    def wrapped(self, *a, **kw):
        res = orig(self, *a, **kw)
        count["n"] += 1
        logdir = kw.get("logdir")
        if logdir is not None:
            self.save_policy_network(os.path.join(logdir, f"snap_ep{run_bpm.TEST_FREQ * count['n']:03d}.pth"))
        return res
    return wrapped


def train():
    import torch
    torch.set_num_threads(THREADS)
    Agent.test_in_train = _snapshotting(Agent.test_in_train)
    from gympn.component_agents import ComponentAgents
    ComponentAgents.test_in_train = _snapshotting(ComponentAgents.test_in_train)
    base = os.path.join(run_bpm.RESULTS, "bpm_insurer_alloc_n1_ep40_nopp_aepn_flat", "baselines_N1.json")
    baselines = json.load(open(base))
    cfg = run_bpm.Hyper()
    if BETA is not None:
        cfg.beta = BETA
    m = run_bpm.train_cell(METHOD, SEED, cfg, TRAIN, baselines)
    json.dump(m, open(os.path.join(OUT, "cell.json"), "w"))
    r, h = baselines["random_mean"], baselines["heuristic_mean"]
    print(f"trained {TAG} in {m['minutes']:.1f} min; greedy final {(m['greedy_final'] - r) / (h - r):.3f} "
          f"best {(m['greedy_best'] - r) / (h - r):.3f} (normalized)", flush=True)


def probe():
    import glob
    import alloc_behaviour
    from alloc_behaviour import episode, policy_chooser, SEED0, EPISODES
    alloc_behaviour.PN_ATTRS["global_view"] = {"local": False, "full": "edges"}[VIEW]
    rows = []
    for path in sorted(glob.glob(os.path.join(TRAIN, "**", "snap_ep*.pth"), recursive=True)):
        ep = int(os.path.basename(path)[7:10])
        ch = policy_chooser(path)
        eps = [episode(SEED0 + i, ch) for i in range(EPISODES)]
        allocs = Counter(e[0] for e in eps)
        first = Counter()
        for e in eps:
            for k, v in (e[1] or {}).items():
                first[k] += v / EPISODES
        alloc_mass = sum(v for k, v in first.items() if k[0] != "ops")
        main = allocs.most_common(1)[0][0]
        split = sorted(d for d, s in main if s == 1) == ["cl", "co"]
        row = {"epoch": ep, "main_allocation": [list(x) for x in main],
               "distinct": len(allocs), "experts_split_cl_co": split,
               "first_decision": {str(k): round(v / alloc_mass, 3) for k, v in first.items() if k[0] != "ops"},
               "return": float(np.mean([e[2] for e in eps]))}
        rows.append(row)
        top = max(row["first_decision"].values()) if row["first_decision"] else float("nan")
        print(f"ep{ep:3d}  ret {row['return']:6.2f}  alloc {row['main_allocation']}  "
              f"distinct {row['distinct']}  top first-decision p {top:.2f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "probe.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(TRAIN, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, "cell.json")):
        train()
    probe()
