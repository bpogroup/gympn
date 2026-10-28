#!/bin/bash
# A 2-day DCL collection on the insurer and 2 held-out planner days: checks the
# setup in a few minutes before the real jobs.  sbatch slurm/smoke.sbatch
#SBATCH -J gympn-smoke
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -o slurm/logs/%x-%j.out
source slurm/env.sh
python -u scripts/run_pi.py env=insurer N=1 days=2 sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=1 workers=2 round=1
rm -rf results/pi_insurer_n1_R8_d2_*
python -m pytest -q ../../../tests/test_merge_options.py
