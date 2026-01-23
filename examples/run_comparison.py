#!/usr/bin/env python3
"""
Quick-start script for convergence speed comparison.

Usage:
    python run_comparison.py

Options can be set by editing the script or passing command-line arguments.
"""

import argparse
from pathlib import Path
from examples.compare_convergence_speed import ComparisonConfig, ConvergenceComparison

def main():
    parser = argparse.ArgumentParser(
        description="Compare convergence speed: PPO-Clip vs PPO-Clip with Causal RL"
    )

    parser.add_argument('--num-seeds', type=int, default=3,
                       help='Number of random seeds (default: 3)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (default: 100)')
    parser.add_argument('--episodes-per-epoch', type=int, default=20,
                       help='Episodes per epoch (default: 20)')
    parser.add_argument('--max-episode-length', type=int, default=500,
                       help='Max steps per episode (default: 500)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--policy-lr', type=float, default=5e-4,
                       help='Policy learning rate (default: 5e-4)')
    parser.add_argument('--value-lr', type=float, default=1e-3,
                       help='Value learning rate (default: 1e-3)')
    parser.add_argument('--entropy-coeff', type=float, default=0.01,
                       help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--ppo-eps', type=float, default=0.15,
                       help='PPO clipping range (default: 0.15)')
    parser.add_argument('--dcl-horizon', type=int, default=3,
                       help='DCL planning horizon (default: 3)')
    parser.add_argument('--output-dir', type=str, default='convergence_comparison_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create configuration
    config = ComparisonConfig()
    config.num_seeds = args.num_seeds
    config.epochs = args.epochs
    config.episodes_per_epoch = args.episodes_per_epoch
    config.max_episode_length = args.max_episode_length
    config.batch_size = args.batch_size
    config.policy_lr = args.policy_lr
    config.value_lr = args.value_lr
    config.entropy_coeff = args.entropy_coeff
    config.ppo_eps = args.ppo_eps
    config.dcl_horizon = args.dcl_horizon
    config.output_dir = Path(args.output_dir)
    config.output_dir.mkdir(exist_ok=True)

    print(f"""
============================================================
  PPO-Clip vs PPO-Clip with Causal RL Comparison
============================================================

Configuration:
  Seeds:                  {config.num_seeds}
  Epochs:                 {config.epochs}
  Episodes/epoch:         {config.episodes_per_epoch}
  Max episode length:     {config.max_episode_length}
  Batch size:             {config.batch_size}
  
Hyperparameters:
  Policy LR:              {config.policy_lr}
  Value LR:               {config.value_lr}
  Entropy coeff:          {config.entropy_coeff}
  PPO epsilon:            {config.ppo_eps}
  DCL horizon:            {config.dcl_horizon}

Output:
  Directory:              {config.output_dir}

Starting comparison... (this may take a while)
""")

    # Run comparison
    comparison = ConvergenceComparison(config)
    comparison.run_comparison()

    print(f"""
============================================================
              Comparison Complete
============================================================

Results saved to:
  {config.output_dir / 'convergence_comparison.json'}
  {config.output_dir / 'convergence_comparison.png'}

Next steps:
  1. Check the JSON file for detailed metrics
  2. View the PNG for learning curves
  3. Edit configuration and re-run to test variations
""")

if __name__ == '__main__':
    main()

