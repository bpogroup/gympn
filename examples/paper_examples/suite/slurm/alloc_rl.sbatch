#!/bin/bash
#SBATCH -J gympn-alloc-rl
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 36:00:00
#SBATCH -o slurm/logs/%x-%j.out
# PPO and NF-GAE (marking certificate) on the insurer with start-of-shift
# allocation, N regions, in the observation of the DCL networks (options,
# clock node), 5 seeds each, the allocation protocol (beta 0.05, 150 epochs,
# whole-net view). 10 cells in parallel.
#   sbatch --export=ALL,N=1 slurm/alloc_rl.sbatch
source slurm/env.sh
N=${N:-1}
python scripts/run_bpm.py env=insurer_alloc N=$N 10 seeds=5 methods=ppo,nfgae cert=marking flat=1 threads=1 \
    beta=0.05 epochs=150 clock=node merge=1
