"""Compare PPO-clip (no causal credit) vs the REC redistribution scheme across
the 8-env paper suite.

This is the standard stability suite (`run_suite`) but the causal method
(`causal_td0`) is run with `causal_scheme="rec"` — the return-equivalent scheme
proved sound in CAUSAL_REDISTRIBUTION_SOUND_SCHEME.md — instead of the default
flow_dag. PPO-clip is the non-causal baseline; both share identical env config
and hyperparameters so the only difference is the credit-assignment scheme.

Resumable: per-cell JSON under <output_dir>/cells/ (re-run to continue).

Usage:
    python run_ppo_vs_rec.py [mode] [num_workers]
      mode in {smoke, smoke8, reduced, full}   (default: reduced)
      num_workers defaults to min(pending, cpu_count)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_suite                                   # noqa: E402
from config import SuiteConfig, smoke_config, smoke8_config, ALL_ENVS  # noqa: E402

REC_SCHEME = "rec"

# Inject causal_scheme="rec" into every cell's args. run_suite.train_cell builds
# args via run_suite._make_args, so wrapping that one function flips the causal
# method from flow_dag (default) to REC without touching the suite source.
_orig_make_args = run_suite._make_args


def _make_args_rec(env_name, method, seed, cfg, logdir_base):
    args = _orig_make_args(env_name, method, seed, cfg, logdir_base)
    args["causal_scheme"] = REC_SCHEME
    return args


run_suite._make_args = _make_args_rec


def reduced_config() -> SuiteConfig:
    """All 8 envs x {ppo_clip, rec} x 5 seeds x 30 epochs — a publishable-strength
    comparison at ~half the cost of the full 10-seed grid."""
    return SuiteConfig(
        envs=list(ALL_ENVS),
        methods=["ppo_clip", "causal_td0"],
        seeds=5,
        output_dir=Path("ppo_vs_rec_results"),
    )


def full_config() -> SuiteConfig:
    return SuiteConfig(
        envs=list(ALL_ENVS),
        methods=["ppo_clip", "causal_td0"],
        seeds=10,
        output_dir=Path("ppo_vs_rec_results_full"),
    )


def _cfg_for(mode: str) -> SuiteConfig:
    if mode == "smoke":
        c = smoke_config();  c.output_dir = Path("ppo_vs_rec_smoke");  return c
    if mode == "smoke8":
        c = smoke8_config(); c.output_dir = Path("ppo_vs_rec_smoke8"); return c
    if mode == "full":
        return full_config()
    return reduced_config()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "reduced"
    cfg = _cfg_for(mode)
    n_pending = len(cfg.envs) * len(cfg.methods) * cfg.seeds
    if len(sys.argv) > 2:
        workers = int(sys.argv[2])
    else:
        # Physical cores, not logical: one heavyweight cell per logical core on a
        # hyperthreaded machine 2x-oversubscribes and the parallel/loop envs crawl.
        workers = min(n_pending, run_suite._physical_core_count())
    print(f"[ppo-vs-rec] mode={mode} scheme={REC_SCHEME} grid="
          f"{len(cfg.envs)}x{len(cfg.methods)}x{cfg.seeds}={n_pending} cells "
          f"workers={workers} -> {cfg.output_dir}")
    run_suite.run_suite(cfg, num_workers=workers)