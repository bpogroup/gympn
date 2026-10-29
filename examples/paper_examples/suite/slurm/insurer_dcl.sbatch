#!/bin/bash
# DCL, component-local loss, and certificate-factored DCL on the insurer with
# N regions (K = 3N components), 3 training seeds each; then the Sequential
# Halving lookahead on the 100 held-out days.
#   sbatch --export=ALL,N=1 slurm/insurer_dcl.sbatch
#   sbatch --export=ALL,N=2 slurm/insurer_dcl.sbatch
#SBATCH -J gympn-ins-dcl
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%j.out
source slurm/env.sh
N=${N:-1}; K=$((3 * N)); CF=$(printf '40,%.0s' $(seq $K)); CF=${CF%,}
E="env=insurer N=$N"
COMMON="sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=15 days=40"
L=results/snellius_ins$N; mkdir -p $L
python -u scripts/run_pi.py $E $COMMON workers=$W round=1 > $L/plain.log 2>&1
for s in 1 2; do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON seed=$s name=s$s round=1 > $L/plain_s$s.log 2>&1 & done
for s in 0 1 2; do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON seed=$s name=rn_s$s renorm=1 round=1 > $L/rn_s$s.log 2>&1 & done
wait
python -u scripts/run_pi.py $E $COMMON workers=$W round=1 cf=$CF dyn=0 > $L/cf.log 2>&1
for s in 1 2; do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON cf=$CF dyn=0 seed=$s name=s$s round=1 > $L/cf_s$s.log 2>&1 & done
wait
python -u scripts/planner_heldout.py env=insurer N=$N workers=$W \
    anchors=results/pi_insurer_n${N}_R8_d40_mc_fair_mrg_clkn_sh32/eval_anchors.json > $L/planner.log 2>&1
