#!/bin/sh
# Complete the ncopies arm sets at 40 epochs. N=2 first (cells ~5 min), then
# N=4 (~20 min). Both resumable; existing 40-epoch cells are skipped.
cd /c/Users/lobia/PycharmProjects/gympn/examples/paper_examples/suite || exit 1
python run_n2_epochs.py 40 10 4 2 ppo,ccf,cgae,cfgae > n2_ep40_rest.log 2>&1
python run_n2_epochs.py 40 10 4 4 ccf,cgae,cfgae,cgae_cap > n4_ep40_rest.log 2>&1
