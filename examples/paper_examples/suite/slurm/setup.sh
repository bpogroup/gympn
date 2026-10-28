#!/bin/bash
# One-time setup on Snellius (run on a login node from the repository root):
#   bash examples/paper_examples/suite/slurm/setup.sh
# Creates .venv in the repository with a CPU-only PyTorch and gympn installed
# in editable mode. Check `module avail Python` if the module below is gone.
set -e
cd "$(dirname "$0")/../../../.."   # the repository root, wherever this is run from
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scipy
pip install -e .
python -c "import gympn, torch, torch_geometric; print('ok', torch.__version__, torch_geometric.__version__)"
