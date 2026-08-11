r"""ppo_clip vs cgae vs cfgae on s1, 5 paired seeds -- the replication run.

Seed 0 (single, paired) said: cgae ties ppo at the endpoint (13.7 / 13.6) but
converges much faster and flatter (mean greedy over training 13.55 vs 12.61,
drift 0.30 vs 0.80); cfgae ended below both (12.6) with the highest peak
(14.45) and the worst drift (1.85). Epoch-to-epoch noise on these curves is
~+-1, so none of that survives a single seed. This is the run that decides
whether the cgae mean-greedy gap (+0.94) is real.

Every arm here is post-fix: before today, run_suite._make_args silently mapped
cgae/cfgae to causal_scheme='lrq', so all earlier cgae/cfgae numbers on disk
are lrq numbers (see run_cgae_paired_s1.py's VOIDED RUN note).

Seeds are 0..4 and each cell reseeds from its own index, so the three arms are
paired seed-by-seed. Seed-0 cells are pre-seeded into cells/ from the two
paired runs (identical config, verified bit-identical ppo arm across three
independent runs), so only 12 of the 15 cells actually train.

NOTE the __main__ guard. num_workers>1 uses a ProcessPoolExecutor, and Windows
spawns (not forks) children, so each child re-imports this file as __mp_main__.
Without the guard the child re-runs run_suite, which spawns more children,
until the pool dies -- this is the "a 2-worker pool died with BrokenProcessPool"
noted in run_cgae_paired_s1.py, which is a missing guard in the caller and not
a defect in run_suite's pool.
"""
import os, sys, shutil
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

# Reuse the seed-0 cells already trained under the fixed dispatch.
REUSE = [
    ("suite_results_cgae_paired",  "s1_stoch_sequence__ppo_clip__s0.json"),
    ("suite_results_cgae_paired",  "s1_stoch_sequence__cgae__s0.json"),
    ("suite_results_cfgae_paired", "s1_stoch_sequence__cfgae__s0.json"),
]


def main():
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["ppo_clip", "cgae", "cfgae"]
    cfg.seeds = 5
    cfg.output_dir = HERE / "suite_results_three_way_s1"

    cells = Path(cfg.output_dir) / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    for src_dir, name in REUSE:
        src = HERE / src_dir / "cells" / name
        dst = cells / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[reuse] {name} <- {src_dir}")

    print(f"[3way] {cfg.methods} x seeds 0-4 x {cfg.epochs} epochs -> {cfg.output_dir}")
    run_suite(cfg, num_workers=2)


if __name__ == '__main__':
    main()
