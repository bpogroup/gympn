#!/bin/sh
# Runs after queue_ep40.sh finishes (that script exits when N=4/40 completes).
# Order: cheap story-completing run first, then the expensive scaling point.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1

# wait for the in-flight ncopies 40-epoch completion (N=4 rest = 40 cells)
while [ "$(ls suite_results_n4_ep40/cells 2>/dev/null | grep -c '^N4__')" -lt 70 ]; do
  sleep 120
done
sleep 30

python run_s1_ep40.py 40 4 > s1_ep40.log 2>&1

# N=8 at 40 epochs: 4 arms x 5 seeds. ccf included because the stored sweep has
# N=8 as ITS strongest setting (+0.569 over ppo), so it is the sharpest test of
# whether cgae_cflow still leads where component-factored credit is at its best.
python run_n2_epochs.py 40 5 4 8 ppo,ccf,cgae_flow,cgae_cflow > n8_ep40.log 2>&1
