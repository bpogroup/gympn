#!/bin/bash
#SBATCH -J gympn-slow-dcl
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 16:00:00
#SBATCH -o slurm/logs/%x-%j.out
# DCL and certificate-factored DCL on the slow-mismatch insurer (one region,
# K = 3), without postponement (work-conserving heuristic as the base) and with
# component-scoped postponement (the waiting heuristic as the base), 10 training
# seeds each on one collection; then the Sequential Halving lookahead on the
# 100 held-out days, with and without postponement. All arms share the anchors
# random (no postponement) / waiting heuristic.
#   sbatch slurm/slow_dcl.sbatch
source slurm/env.sh
E="env=insurer_slow N=1"
COMMON="sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=15 days=40 round=1"
L=results/snellius_slow; mkdir -p $L
for PP in 0 component; do
  S=$([ $PP = component ] && echo _ppc)
  python -u scripts/run_pi.py $E $COMMON postpone=$PP workers=$W > $L/plain$S.log 2>&1
  for s in $(seq 1 9); do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON postpone=$PP seed=$s name=s$s > $L/plain${S}_s$s.log 2>&1 & done
  wait
  python -u scripts/run_pi.py $E $COMMON postpone=$PP cf=40,40,40 workers=$W > $L/cf$S.log 2>&1
  for s in $(seq 1 9); do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON postpone=$PP cf=40,40,40 seed=$s name=s$s > $L/cf${S}_s$s.log 2>&1 & done
  wait
  python -u scripts/planner_heldout.py $E postpone=$PP workers=$W \
      anchors=results/pi_insurer_slow_n1_R8_d40_mc${S}_fair_mrg_clkn_sh32/eval_anchors.json > $L/planner$S.log 2>&1
done
