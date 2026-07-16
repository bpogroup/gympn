"""Quick 2-seed comparison to validate fixes."""
import sys
sys.path.insert(0, '.')

# Temporarily override config for quick test
from examples.compare_convergence_two_stage import ComparisonConfig, ConvergenceComparisonTwoStage

cfg = ComparisonConfig()
cfg.num_seeds = 2
cfg.epochs = 20
cfg.episodes_per_epoch = 10

comp = ConvergenceComparisonTwoStage(cfg)
comp.run_comparison()

