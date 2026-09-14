#!/bin/sh
# Bring every converged-budget experiment to 20 seeds.
# Ordered CHEAPEST FIRST so results accrue early; N=8 is last (40h of the ~67h
# total) and can be killed without losing the others. All runs are resumable by
# cell file, so only the new seeds train.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1

echo "[q20] stage 1/5: ncopies N=2 -> 20 seeds (~1.7h)"
python run_n2_epochs.py 40 20 4 2 ppo,ccf,cgae,cfgae,cgae_flow,cgae_cflow,cgae_cap > s20_n2.log 2>&1

echo "[q20] stage 2/5: ncopies N=4 -> 20 seeds (~6.5h)"
python run_n2_epochs.py 40 20 4 4 ppo,ccf,cgae,cfgae,cgae_flow,cgae_cflow,cgae_cap > s20_n4.log 2>&1

echo "[q20] stage 3/5: s1 -> 20 seeds (~9.2h)"
python run_s1_ep40.py 40 4 20 > s20_s1.log 2>&1

echo "[q20] stage 4/5: multisite -> 20 seeds (~9.2h)"
python run_multisite_validate.py 4 seeds=20 > s20_multisite.log 2>&1

echo "[q20] stage 5/5: ncopies N=8 -> 20 seeds (~40h, the expensive one)"
python run_n2_epochs.py 40 20 4 8 ppo,ccf,cgae_flow,cgae_cflow > s20_n8.log 2>&1

echo "[q20] ALL STAGES COMPLETE"
