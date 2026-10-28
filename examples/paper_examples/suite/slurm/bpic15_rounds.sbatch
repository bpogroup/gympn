#!/bin/bash
# DCL and certificate-factored DCL on BPIC15, three rounds of policy iteration,
# one independent seed per array task: collection k (cset=k, training days
# disjoint from every other collection and round) and seed k. The two chains
# run side by side, each with half of the cores.
#   sbatch --array=0-9 slurm/bpic15_rounds.sbatch
#SBATCH -J gympn-bpic15-rounds
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00
#SBATCH -o slurm/logs/%x-%A_%a.out
source slurm/env.sh
K=${SLURM_ARRAY_TASK_ID:-0}
H=$((W / 2))
COMMON="env=bpic15 days=40 sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=15 seed=$K cset=$K workers=$H"
L=results/snellius_bpic15/k$K; mkdir -p $L

chain() {  # $1 = log tag, $2 = directory suffix of the method, $3 = its extra args
    P=""
    for r in 1 2 3; do
        R=$([ $r -gt 1 ] && echo _r$r)
        C=$([ $K -gt 0 ] && echo _cset$K)
        OMP_NUM_THREADS=$H python -u scripts/run_pi.py $COMMON $3 round=$r $P > $L/$1_r$r.log 2>&1 || return 1
        D=results/pi_bpic15_n1_R8_d40_mc${R}_fair_mrg_clkn_sh32$2$C
        cp $D/summary_tau0.25.json $L/$1_r$r.json
        P="base=$D/policy_tau0.25_final.pth init=$D/policy_tau0.25_final.pth"
    done
}
chain plain "" "" &
chain cf _cf35-24-58-56 "cf=35,24,58,56" &
wait
