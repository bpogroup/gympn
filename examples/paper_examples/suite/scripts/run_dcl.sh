#!/bin/bash
# Deep Controlled Learning (Temizoz et al., EJOR 2025) on an AEPN env, with
# scripts/run_pi.py: approximate policy iteration from the env's heuristic.
# Round k: the planner plays DAYS training days (new days each round), choosing
# at every decision by Sequential Halving with common random numbers (sh=M
# rollouts per option in total) over rollouts of the current policy (round 1:
# the heuristic; round k > 1: the previous round's network, greedy); the network
# is trained as a classifier on those choices (tau=0) and is the next round's
# base and starting point. Observations: flat, merged options, clock node.
#
# usage: bash scripts/run_dcl.sh [env=bpic15] [rounds=3] [days=40] [M=32] [workers=10] [epochs=15]
cd "$(dirname "$0")/.."
PY=../../../.venv/Scripts/python
ENV=bpic15; ROUNDS=3; DAYS=40; M=32; WORKERS=10; EPOCHS=15
for a in "$@"; do
  case $a in
    env=*) ENV=${a#env=};; rounds=*) ROUNDS=${a#rounds=};; days=*) DAYS=${a#days=};;
    M=*) M=${a#M=};; workers=*) WORKERS=${a#workers=};; epochs=*) EPOCHS=${a#epochs=};;
  esac
done
COMMON="env=$ENV days=$DAYS sh=$M fair=1 merge=1 clocknode=1 tau=0 epochs=$EPOCHS workers=$WORKERS"
DIR="pi_${ENV}_n1_R8_d${DAYS}_mc_fair_mrg_clkn_sh${M}"
LOG="results/dcl_${ENV}_d${DAYS}_sh${M}.log"
echo "[dcl] $(date +%H:%M) round 1 (base: heuristic)" | tee -a "$LOG"
$PY scripts/run_pi.py $COMMON round=1 >> "$LOG" 2>&1 || exit 1
PREV="results/$DIR/policy_tau0.pth"
for k in $(seq 2 "$ROUNDS"); do
  echo "[dcl] $(date +%H:%M) round $k (base: $PREV)" | tee -a "$LOG"
  $PY scripts/run_pi.py $COMMON round=$k base="$PREV" init="$PREV" >> "$LOG" 2>&1 || exit 1
  PREV="results/${DIR/_mc_/_mc_r${k}_}/policy_tau0.pth"
done
echo "[dcl] $(date +%H:%M) done" | tee -a "$LOG"
