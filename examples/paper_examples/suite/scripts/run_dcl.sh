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
# select=race: the cheap variant, racing (run_pi race=EPS rcap=CAP, z=2) with
# set-valued labels (setlab=1) instead of Sequential Halving.
#
# usage: bash scripts/run_dcl.sh [env=bpic15] [rounds=3] [days=40] [M=32] [workers=10] [epochs=15]
#                                [select=sh|race] [eps=0.25] [cap=64]
cd "$(dirname "$0")/.."
PY=../../../.venv/Scripts/python
ENV=bpic15; ROUNDS=3; DAYS=40; M=32; WORKERS=10; EPOCHS=15; SELECT=sh; EPS=0.25; CAP=64
for a in "$@"; do
  case $a in
    env=*) ENV=${a#env=};; rounds=*) ROUNDS=${a#rounds=};; days=*) DAYS=${a#days=};;
    M=*) M=${a#M=};; workers=*) WORKERS=${a#workers=};; epochs=*) EPOCHS=${a#epochs=};;
    select=*) SELECT=${a#select=};; eps=*) EPS=${a#eps=};; cap=*) CAP=${a#cap=};;
  esac
done
if [ "$SELECT" = race ]; then
  SEL="race=$EPS rcap=$CAP setlab=1"; TAG="race${EPS}z2c${CAP}"
else
  SEL="sh=$M"; TAG="sh${M}"
fi
COMMON="env=$ENV days=$DAYS $SEL fair=1 merge=1 clocknode=1 tau=0 epochs=$EPOCHS workers=$WORKERS"
DIR="pi_${ENV}_n1_R8_d${DAYS}_mc_fair_mrg_clkn_${TAG}"
LOG="results/dcl_${ENV}_d${DAYS}_${TAG}.log"
echo "[dcl] $(date +%H:%M) round 1 (base: heuristic)" | tee -a "$LOG"
$PY scripts/run_pi.py $COMMON round=1 >> "$LOG" 2>&1 || exit 1
PREV="results/$DIR/policy_tau0.pth"
for k in $(seq 2 "$ROUNDS"); do
  echo "[dcl] $(date +%H:%M) round $k (base: $PREV)" | tee -a "$LOG"
  $PY scripts/run_pi.py $COMMON round=$k base="$PREV" init="$PREV" >> "$LOG" 2>&1 || exit 1
  PREV="results/${DIR/_mc_/_mc_r${k}_}/policy_tau0.pth"
done
echo "[dcl] $(date +%H:%M) done" | tee -a "$LOG"
