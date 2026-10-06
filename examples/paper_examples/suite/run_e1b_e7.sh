#!/bin/bash
# E1b: HGT PPO on multi-site with current code (is the HGT failure code drift?).
# E7: is PPO on the 2-region insurer just under-budgeted or mis-tuned?
cd "$(dirname "$0")"
PY=../../../.venv/Scripts/python.exe
echo "start $(date +%T)"
$PY -u run_multisite_protocol.py 4 seeds=4 methods=ppo net=hgt tag=e1b_current threads=1 > e1b_multisite_hgt.log 2>&1 &
$PY -u run_bpm.py env=insurer N=2 methods=ppo seeds=5 flat=1 threads=1 episodes=16 5 > e7_insurer_n2_eps16.log 2>&1 &
$PY -u run_bpm.py env=insurer N=2 methods=ppo seeds=5 flat=1 threads=1 plr=0.00015 4 > e7_insurer_n2_plr_low.log 2>&1 &
$PY -u run_bpm.py env=insurer N=2 methods=ppo seeds=5 flat=1 threads=1 plr=0.0006 4 > e7_insurer_n2_plr_high.log 2>&1 &
wait
echo "E1B E7 DONE $(date +%T)"
