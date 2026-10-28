#!/bin/bash
#SBATCH -J gympn-alloc-dcl
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%j.out
# DCL, component-local loss, and certificate-factored DCL with the marking
# certificate (dyn=1) on the insurer with start-of-shift allocation, N regions,
# 3 training seeds each; then the Sequential Halving lookahead on the 100
# held-out days. The static certificate sees one component per region (the
# pool couples its processes); dyn=1 continues claims, complaints and
# underwriting alone once the pool is empty.
#   sbatch --export=ALL,N=1 slurm/alloc_dcl.sbatch
#   sbatch --export=ALL,N=2 slurm/alloc_dcl.sbatch
source slurm/env.sh
N=${N:-1}; CF=$(printf '40,%.0s' $(seq $N)); CF=${CF%,}
E="env=insurer_alloc N=$N"
COMMON="sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=15 days=40"
L=results/snellius_alloc$N; mkdir -p $L
python -u scripts/run_pi.py $E $COMMON workers=$W round=1 > $L/plain.log 2>&1
for s in 1 2; do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON seed=$s name=s$s round=1 > $L/plain_s$s.log 2>&1 & done
for s in 0 1 2; do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON seed=$s name=rn_s$s renorm=1 round=1 > $L/rn_s$s.log 2>&1 & done
wait
python -u scripts/run_pi.py $E $COMMON workers=$W round=1 cf=$CF dyn=1 > $L/cf_dyn.log 2>&1
for s in 1 2; do OMP_NUM_THREADS=2 python -u scripts/run_pi.py $E $COMMON cf=$CF dyn=1 seed=$s name=s$s round=1 > $L/cf_dyn_s$s.log 2>&1 & done
wait
python -u scripts/planner_heldout.py env=insurer_alloc N=$N workers=$W \
    anchors=results/pi_insurer_alloc_n${N}_R8_d40_mc_fair_mrg_clkn_sh32/eval_anchors.json > $L/planner.log 2>&1
