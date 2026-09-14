#!/bin/sh
# Start the s3/s4 safety run once the N=2 arm has written all 20 cells.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
while [ "$(ls suite_results_ncopies_3way_crn/cells 2>/dev/null | grep -c '^N2__cgae_cflow__')" -lt 20 ]; do
  sleep 30
done
sleep 30
python run_s3s4_safety.py 4 > s3s4_safety.log 2>&1
