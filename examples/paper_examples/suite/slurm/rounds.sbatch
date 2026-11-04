#!/bin/bash
# DCL and certificate-factored DCL on the insurer benchmarks, three rounds of policy
# iteration, in the paper's protocol (DCL observation, SH32, soft targets): one
# collection per array task, benchmark b = K / 3 and collection c = K % 3 + 1 (cset=c,
# training days disjoint from every other collection and round, seed=c). The two
# chains run side by side, each with half of the cores. Round r > 1 labels with the
# lookahead over the network of round r - 1 and starts training from it.
#   sbatch --array=0-14 slurm/rounds.sbatch
# Results: results/snellius_rounds/<benchmark>_c<c>/{plain,cf}_r<r>.{log,json}
#SBATCH -J gympn-rounds
#SBATCH -p rome
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -t 16:00:00
#SBATCH -o slurm/logs/%x-%A_%a.out
source slurm/env.sh
K=${SLURM_ARRAY_TASK_ID:-0}
B=$((K / 3)); C=$((K % 3 + 1))
H=$((W / 2))
case $B in
  0) TAG=ins1;   ENV=insurer;       N=1; PP="";                  CF=40,40,40 ;;
  1) TAG=ins2;   ENV=insurer;       N=2; PP="";                  CF=40,40,40,40,40,40 ;;
  2) TAG=alloc1; ENV=insurer_alloc; N=1; PP="";                  CF=40 ;;
  3) TAG=alloc2; ENV=insurer_alloc; N=2; PP="";                  CF=40,40 ;;
  4) TAG=slow;   ENV=insurer_slow;  N=1; PP="postpone=component"; CF=40,40,40 ;;
esac
COMMON="env=$ENV N=$N $PP days=40 sh=32 fair=1 merge=1 clocknode=1 tau=0.25 epochs=15 seed=$C cset=$C workers=$H"
L=results/snellius_rounds/${TAG}_c$C; mkdir -p $L
PPD=$([ -n "$PP" ] && echo _ppc)
CFD=_cf$(echo $CF | tr ',' '-')_dyn

chain() {  # $1 = log tag, $2 = directory suffix of the method, $3 = its extra args
    P=""
    for r in 1 2 3; do
        R=$([ $r -gt 1 ] && echo _r$r)
        OMP_NUM_THREADS=$H python -u scripts/run_pi.py $COMMON $3 round=$r $P > $L/$1_r$r.log 2>&1 || return 1
        D=results/pi_${ENV}_n${N}_R8_d40_mc${PPD}${R}_fair_mrg_clkn_sh32$2_cset$C
        cp $D/summary_tau0.25.json $L/$1_r$r.json
        P="base=$D/policy_tau0.25_final.pth init=$D/policy_tau0.25_final.pth"
    done
}
chain plain "" "" &
chain cf "$CFD" "cf=$CF dyn=1" &
wait
