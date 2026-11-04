#!/bin/bash
# Training seeds 3 to 9 of DCL and certificate-factored DCL on the insurer with N = 1
# and N = 2 regions, on the collections of insurer_dcl.sbatch (decisions.pt is reused, so
# only the distillation and the held-out scoring run): Table 3 of the paper then has 10
# training seeds on every benchmark. One array task per (N, method); the seven seeds of a
# task run side by side with two threads each.
#   sbatch --array=0-3 slurm/insurer_seeds.sbatch
# Results: results/pi_insurer_n<N>_R8_d40_mc_fair_mrg_clkn_sh32[_cf...]/summary_tau0.25_s<s>.json
#SBATCH -J gympn-ins-seeds
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH -o slurm/logs/%x-%A_%a.out
source slurm/env.sh
K=${SLURM_ARRAY_TASK_ID:-0}
N=$((K / 2 + 1)); M=$((K % 2))
CF=$(printf '40,%.0s' $(seq $((3 * N)))); CF=${CF%,}
COMMON="env=insurer N=$N sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=15 days=40 round=1"
EXTRA=$([ $M -eq 1 ] && echo "cf=$CF dyn=0")
L=results/snellius_ins$N; mkdir -p $L
TAG=$([ $M -eq 1 ] && echo cf || echo plain)
for s in 3 4 5 6 7 8 9; do
    OMP_NUM_THREADS=2 python -u scripts/run_pi.py $COMMON $EXTRA seed=$s name=s$s > $L/${TAG}_s$s.log 2>&1 &
done
wait
