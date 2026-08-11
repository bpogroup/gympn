r"""cgae vs ppo_clip on s1, seed 0 -- the first genuinely PAIRED comparison.

Both arms are re-run here rather than reusing stored cells: every result on
disk predates gympn.seeding.seed_network_init, so those arms started from
DIFFERENT initial policies (measured: s1 seed 0 epoch-1 return was 8.40 for
ppo/mc_q/lrq2 but 7.95 for cgae/lcv/rudder, and the same method in two runs
gave 8.40 and 7.95). With the fix both now start at 8.90 at seed 0, so the
difference here is attributable to the credit scheme alone.

Reference (10 seeds, pre-fix, unpaired): lrq2 11.21, mc_q 13.41,
ppo_clip 13.59, cfpk 14.03. The pre-fix cgae single seed was 12.05.

VOIDED RUN, 2026-08-09: the first execution of this script did NOT train cgae.
run_suite._make_args mapped any method outside a hand-listed subset to
causal_scheme='lrq', and cgae was not in that list -- so the "cgae" arm was
plain lrq (proof: a later cfgae run, mis-mapped the same way, reproduced this
arm's 30-epoch return curve bit-for-bit). Its 12.10 is an lrq number. The
mapping now tests run_suite.CAUSAL_SCHEMES; artifacts kept as
suite_results_cgae_paired_LRQBUG/ + cgae_paired_LRQBUG.log.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["ppo_clip", "cgae"]
cfg.seeds = 1
cfg.output_dir = Path("suite_results_cgae_paired")
print(f"[paired] {cfg.methods} x seed 0 x {cfg.epochs} epochs -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)   # 1 worker: a 2-worker pool died with BrokenProcessPool
