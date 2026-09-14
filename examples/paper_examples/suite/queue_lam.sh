#!/bin/sh
# After the cgae_cap chain completes (N=4 cap cells written), sweep lambda for
# cgae_cflow at N=2 to test the effective-horizon explanation.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
C=suite_results_ncopies_3way_crn/cells
while [ "$(ls $C 2>/dev/null | grep -c '^N4__cgae_cap__')" -lt 20 ]; do sleep 60; done
sleep 30
python run_lam_probe.py 0.85 4 > lam085.log 2>&1
python run_lam_probe.py 0.70 4 > lam070.log 2>&1
