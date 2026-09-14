#!/bin/sh
# Close the arm x env coverage holes, cheapest first. All resumable by cell file.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
echo "[fill] 1/4  N=2: +mc_q +cgae_dag  (~1h)"
python run_n2_epochs.py 40 20 4 2 mc_q,cgae_dag > fill_n2.log 2>&1
echo "[fill] 2/4  s1: +ccf +mc_q  (~3.5h)"
python run_s1_ep40.py 40 4 20 > fill_s1.log 2>&1
echo "[fill] 3/4  N=4: +mc_q +cgae_dag  (~3.7h)"
python run_n2_epochs.py 40 20 4 4 mc_q,cgae_dag > fill_n4.log 2>&1
echo "[fill] ALL CHEAP FILLS DONE"
