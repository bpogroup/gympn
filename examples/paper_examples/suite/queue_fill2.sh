#!/bin/sh
# Re-run the ncopies fills that died on the CAUSAL KeyError, after the s1 stage.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
# wait for the s1 fill to finish (ccf + mc_q = 40 new cells on top of 140)
while [ "$(ls suite_results_s1_ep40/cells 2>/dev/null | grep -c '__ccf__')" -lt 20 ]; do sleep 60; done
sleep 30
echo "[fill2] N=2: +mc_q +cgae_dag"
python run_n2_epochs.py 40 20 4 2 mc_q,cgae_dag > fill_n2b.log 2>&1
echo "[fill2] N=4: +mc_q +cgae_dag"
python run_n2_epochs.py 40 20 4 4 mc_q,cgae_dag > fill_n4b.log 2>&1
echo "[fill2] DONE"
