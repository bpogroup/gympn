#!/bin/bash
# PPO and NF-GAE (marking certificate) on the insurer with N regions, in the
# observation of the DCL networks (options, clock node), 5 seeds each, the
# insurer protocol (beta 0.5, 8 episodes per epoch, 40 epochs). 10 cells in parallel.
#   sbatch --export=ALL,N=1 slurm/insurer_rl.sbatch
#SBATCH -J gympn-ins-rl
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%j.out
source slurm/env.sh
N=${N:-1}
python scripts/run_bpm.py env=insurer N=$N 10 seeds=5 methods=ppo,nfgae cert=marking flat=1 threads=1 \
    beta=0.5 clock=node merge=1
