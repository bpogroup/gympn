#!/bin/bash
#SBATCH -J gympn-slow-rl
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH -t 24:00:00
#SBATCH -o slurm/logs/%x-%j.out
# PPO and NF-GAE (marking certificate) on the slow-mismatch insurer (one
# region), in the observation of the DCL networks (options, clock node), with
# the insurer protocol (beta 0.5, 8 episodes per epoch, 40 epochs), 10 seeds:
# no postponement (PPO, NF-GAE), global postponement (PPO; NF-GAE with it is
# unsound), and component-scoped postponement (PPO, NF-GAE). All arms share the
# anchors random (no postponement) / waiting heuristic on the held-out days.
#   sbatch slurm/slow_rl.sbatch
source slurm/env.sh
C="env=insurer_slow N=1 seeds=10 cert=marking flat=1 threads=1 beta=0.5 clock=node merge=1"
python scripts/run_bpm.py $C postpone=0         methods=ppo,nfgae 20
python scripts/run_bpm.py $C postpone=1         methods=ppo       10
python scripts/run_bpm.py $C postpone=component methods=ppo,nfgae 20
