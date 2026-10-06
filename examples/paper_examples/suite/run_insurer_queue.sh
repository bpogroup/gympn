#!/bin/bash
# Insurer experiment (2026-10-06): PPO vs NF-GAE, AEPN + flat obs, 10 seeds, 40 epochs.
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
COMMON="seeds=10 flat=1 threads=1"
echo "phase 1 start $(date +%T)"
$PY -u run_bpm.py env=insurer N=1 methods=ppo,nfgae $COMMON 8 > insurer_n1.log 2>&1 &
$PY -u run_bpm.py env=insurer_shared N=1 methods=ppo,nfgae $COMMON 8 > insurer_shared_n1.log 2>&1 &
wait
echo "phase 2 start $(date +%T)"
$PY -u run_bpm.py env=insurer N=2 methods=ppo,nfgae $COMMON 10 > insurer_n2.log 2>&1 &
for p in claims underwriting complaints; do
  $PY -u run_bpm.py env=insurer_$p N=1 methods=ppo $COMMON 2 > insurer_${p}_n1.log 2>&1 &
done
wait
echo "QUEUE DONE $(date +%T)"
