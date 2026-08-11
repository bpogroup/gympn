r"""Re-score the 15 three-way cells' SAVED policies under common random numbers.

Testing only -- no retraining. Each cell's best_policy.pth is loaded and
evaluated on a FIXED scenario set (test_in_train(eval_seed=...)), so all three
arms are scored on identical episodes and the +-0.231 SD of scenario noise that
contaminated the original 20-episode eval points cancels in the paired
differences.

WHAT THIS CAN AND CANNOT ANSWER
  CAN: the level question -- is the policy each arm ended up with actually
       better? With EPISODES=200 per cell the per-arm SE is ~0.07 (vs 0.231
       for a 20-episode point), and CRN removes most of that again from the
       paired difference.
  CANNOT: mean_greedy or greedy_drift. Both are functions of the per-EPOCH
       greedy curve, and only the best checkpoint was saved -- there are no
       per-epoch policies on disk. The drift result (cgae -0.94, p=0.071) is
       therefore NOT recoverable without retraining.
  CAVEAT: best_policy.pth was SELECTED by the old noisy eval (highest of ~15
       points, each +-0.231), so each arm's checkpoint carries a selection
       bias of roughly +max-of-noise. That bias is common to all arms, so the
       paired comparison stays fair, but the absolute levels here will read a
       little high.

Pre-check: ppo trains on a non-causal env and cgae/cfgae on a causal one
(run_suite.train_cell), so each policy is evaluated on the env it was trained
on. CRN is only meaningful if those two env configurations consume the global
`random` stream identically -- uuid4() draws from os.urandom and should not
perturb it, but that is verified here with a scripted policy rather than
assumed. If the pre-check fails, the cross-arm comparison below is invalid.

Run: python rescore_crn.py [episodes]
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gympn.agents import Agent
from gympn.environment import AEPN_Env
from config import stoch_config
from envs import make_env
from run_suite import _env_length

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ENV = "s1_stoch_sequence"
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
EVAL_SEED = 555_000            # the fixed scenario set
METHODS = ["ppo_clip", "cgae", "cfgae"]
SEEDS = [0, 1, 2, 3, 4]
CAUSAL = {"ppo_clip": False, "cgae": True, "cfgae": True}

cfg = stoch_config()
LENGTH = _env_length(cfg, ENV)

# seed-0 cells were trained in the paired runs, so their checkpoints live there
FALLBACK = {
    ("ppo_clip", 0): "suite_results_cgae_paired",
    ("cgae", 0): "suite_results_cgae_paired",
    ("cfgae", 0): "suite_results_cfgae_paired",
}


class _NoOp:
    def train(self):
        pass


class Scorer(Agent):
    """Just enough Agent to reuse act() + test_in_train() with a loaded net."""

    def __init__(self, policy_model):
        self.policy_model = policy_model
        self.value_model = _NoOp()
        self.best_test_metric = float('inf')   # never trips the save-best path


class Scripted(Agent):
    """Fixed action index -- policy-free, so any score difference between two
    env configurations is the env's own RNG consumption."""

    def __init__(self, index=0):
        self.index = index
        self.policy_model = _NoOp()
        self.value_model = _NoOp()
        self.best_test_metric = float('inf')

    def act(self, state, deterministic=True, return_logprob=False):
        n = len(state['actions_dict']) if isinstance(state, dict) and 'actions_dict' in state else 1
        return min(self.index, n - 1)


def make(causal):
    pn = make_env(ENV, causal_rl=causal, allow_postpone=cfg.allow_postpone,
                  causal_postpone_tokenflow=causal)
    pn.use_structural_features = False
    pn.length = LENGTH
    if causal:
        # training_run decorates every initial token with a uuid and seeds the
        # trace's root sentinel before building the env (simulator.py:1752).
        # Skipping it makes the first reward registration die on token._id, so
        # a causal eval env has to be prepared the same way. uuid4() reads
        # os.urandom, NOT the `random` stream, so this does not shift CRN
        # scenarios -- which is exactly what the pre-check below confirms.
        import types, uuid
        for place in pn.places:
            for token in place.marking:
                setattr(token, '_id', str(uuid.uuid4()))
        pn.causal_trace._pn = pn
        pn.causal_trace._static_comp_cache = None
        pn.causal_trace._ls_hca_classify_cache = None
        try:
            pn.causal_trace.flush()
        except Exception:
            pass
        sentinel = types.SimpleNamespace(_id="__initial__")
        for place in pn.places:
            for token in place.marking:
                pn.causal_trace.register_token(token, sentinel, parent_tokens=[], time=0)
        pn.causal_trace.register_transition(
            transition=sentinel, input_tokens=[],
            output_tokens=[t for p in pn.places for t in p.marking],
            is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)


def score(agent, causal, episodes=EPISODES):
    return agent.test_in_train(make(causal), episodes=episodes, eval_seed=EVAL_SEED)


def ckpt(method, seed):
    p = HERE / "suite_results_three_way_s1" / "train" / f"{ENV}__{method}__s{seed}" / "best_policy.pth"
    if p.exists():
        return p
    alt = FALLBACK.get((method, seed))
    if alt:
        p = HERE / alt / "train" / f"{ENV}__{method}__s{seed}" / "best_policy.pth"
        if p.exists():
            return p
    return None


# ---- pre-check: do causal and non-causal envs give the SAME scenarios? -----
print(f"[pre-check] scripted policy, causal vs non-causal env, eval_seed={EVAL_SEED}", flush=True)
a = score(Scripted(0), causal=False, episodes=30)['mean_returns']
b = score(Scripted(0), causal=True, episodes=30)['mean_returns']
print(f"  non-causal {a:.4f} | causal {b:.4f} -> {'MATCH' if a == b else 'MISMATCH'}", flush=True)
if a != b:
    print("  !! env configurations do not share scenarios under CRN; cross-arm")
    print("     comparison below would be confounded. Reporting anyway, flagged.", flush=True)
crn_valid = (a == b)

# ---- score every cell -------------------------------------------------------
rows = {}
for method in METHODS:
    for seed in SEEDS:
        p = ckpt(method, seed)
        if p is None:
            print(f"[skip] {method} s{seed}: no checkpoint", flush=True)
            continue
        model = torch.load(p, weights_only=False)
        m = score(Scorer(model), causal=CAUSAL[method])
        rows[(method, seed)] = m
        print(f"[crn] {method:9s} s{seed}  mean={m['mean_returns']:6.3f}  "
              f"sd={m['std_returns']:5.3f}  min={m['min_returns']:5.1f} max={m['max_returns']:5.1f}",
              flush=True)

out = {"env": ENV, "episodes": EPISODES, "eval_seed": EVAL_SEED,
       "crn_precheck_match": bool(crn_valid),
       "scores": {f"{m}__s{s}": rows[(m, s)]['mean_returns'] for (m, s) in rows}}
(HERE / "rescore_crn.json").write_text(json.dumps(out, indent=2))

# ---- paired summary ---------------------------------------------------------
print()
base = {s: rows[("ppo_clip", s)]['mean_returns'] for s in SEEDS if ("ppo_clip", s) in rows}
for method in ["cgae", "cfgae"]:
    d = [rows[(method, s)]['mean_returns'] - base[s] for s in SEEDS
         if (method, s) in rows and s in base]
    if not d:
        continue
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float('nan')
    se = sd / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    print(f"{method:6s} vs ppo  diff {mean:+6.3f} +-{sd:5.3f} (SE {se:5.3f})  "
          f"W/L {sum(1 for x in d if x > 0)}/{sum(1 for x in d if x <= 0)}  "
          f"per-seed {[round(x, 2) for x in d]}")
print(f"\nwrote rescore_crn.json  ({EPISODES} episodes/cell, CRN pre-check "
      f"{'OK' if crn_valid else 'FAILED'})")
