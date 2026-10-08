#!/bin/bash
# Postponement experiment (2026-10-07): insurer_slow (mismatched underwriting 7
# units, where waiting pays), 2 regions, AEPN + flat obs, 10 seeds, 40 epochs.
# Arms: PPO / NF-GAE with no postpone, PPO with global postpone, PPO / NF-GAE
# with component postpone. NF-GAE + global postpone is unsound (refused).
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
COMMON="env=insurer_slow N=2 seeds=10 flat=1 threads=1"
echo "start $(date +%T)"
$PY -u run_bpm.py $COMMON postpone=0 methods=ppo,nfgae 6 > postpone2_none.log 2>&1 &
$PY -u run_bpm.py $COMMON postpone=1 methods=ppo 4 > postpone2_global.log 2>&1 &
$PY -u run_bpm.py $COMMON postpone=component methods=ppo,nfgae 6 > postpone2_component.log 2>&1 &
wait
echo "POSTPONE2 QUEUE DONE $(date +%T)"
