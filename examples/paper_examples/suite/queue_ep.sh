#!/bin/sh
# Last stage: N=2 at 40 epochs, to test whether the cgae_cflow deficit is a
# knife-edge transient (gap closes) or a real deficiency (gap persists).
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
while [ ! -d suite_results_lam_probe_07 ] || \
      [ "$(ls suite_results_lam_probe_07/cells 2>/dev/null | grep -c '^N2__')" -lt 20 ]; do
  sleep 60
done
sleep 30
python run_n2_epochs.py 40 10 4 > n2_ep40.log 2>&1
