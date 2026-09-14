#!/bin/sh
# Chain the cgae_cap validation: N=2 (running) -> s1 -> ncopies N=4.
# Each stage gates on the previous stage's cell count, so no stage can
# double-train and every stage is independently resumable.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
C=suite_results_ncopies_3way_crn/cells

while [ "$(ls $C 2>/dev/null | grep -c '^N2__cgae_cap__')" -lt 20 ]; do sleep 30; done
sleep 20
python run_three_way_s1_5seed_crn.py > cap_s1.log 2>&1

while [ "$(ls suite_results_three_way_s1_crn/cells 2>/dev/null | grep -c '__cgae_cap__')" -lt 5 ]; do sleep 30; done
sleep 20
python run_ncopies_three_way_crn.py 4 seeds=20 ns=4 > cap_n4.log 2>&1
