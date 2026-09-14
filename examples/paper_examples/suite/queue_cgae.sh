#!/bin/sh
# Wait for the cgae_cflow arm to finish (all 12 cells on disk), then run cgae.
# Gating on cell COUNT rather than process exit is what makes this safe to
# overlap: the follow-up run's pending list is computed from missing cells, so
# once the 12 cgae_cflow cells exist the only work left is the 12 cgae cells --
# no double-training even if the first process is still writing its summary.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
while [ "$(ls suite_results_multisite_bf/cells 2>/dev/null | grep -c '^cgae_cflow__')" -lt 12 ]; do
  sleep 60
done
sleep 45
python run_multisite_validate.py 4 > multisite_cgae.log 2>&1
