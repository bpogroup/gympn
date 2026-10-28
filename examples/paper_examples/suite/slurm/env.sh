# Sourced by the job scripts: modules, venv, and the suite as working directory.
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
REPO=${REPO:-$HOME/gympn}
source $REPO/.venv/bin/activate
cd $REPO/examples/paper_examples/suite
export PYTHONUNBUFFERED=1
W=${SLURM_CPUS_PER_TASK:-8}
