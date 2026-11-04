#!/bin/bash
# The insurer with four regions (K = 12 components), the paper's protocol, as one array:
#   task 0: PPO and NF-PPO, 10 seeds, 150 epochs, beta 0.05, options and clock node,
#           marking certificate, 10 cells in parallel (as alloc_rl.sbatch for the insurer);
#   task 1: DCL and certificate-factored DCL, one collection of 40 days each, three
#           training seeds, the component-local loss on the data of DCL, then the
#           Sequential Halving lookahead on the 100 held-out days (insurer_dcl.sbatch
#           with N=4; the collection grows with K, hence the longer time limit).
#   sbatch --array=0-1 slurm/insurer4.sbatch
# Afterwards, slurm/insurer_seeds.sbatch can be extended to N=4 for seeds 3 to 9.
#SBATCH -J gympn-ins4
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 48:00:00
#SBATCH -o slurm/logs/%x-%A_%a.out
source slurm/env.sh
N=4
case ${SLURM_ARRAY_TASK_ID:-0} in
  0)
    python scripts/run_bpm.py env=insurer N=$N 10 seeds=10 methods=ppo,nfgae cert=marking flat=1 threads=1 \
        beta=0.05 epochs=150 clock=node merge=1
    ;;
  1)
    export N
    bash slurm/insurer_dcl.sbatch
    ;;
esac
