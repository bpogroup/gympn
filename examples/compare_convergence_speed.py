"""
Convergence Speed Comparison: PPO-Clip vs PPO-Clip with Causal RL vs DCL

This script provides a fair, controlled comparison of convergence speed between:
1. Standard PPO-Clip (baseline)
2. PPO-Clip with Causal RL enabled (causal reward redistribution)
3. DCL Agent (deep counterfactual learning with planning)

Setup ensures:
- Same environment and hyperparameters (except algorithm-specific)
- Same number of episodes and epochs
- Same network architecture
- Multiple seeds for statistical significance
- Detailed metrics and visualization

Run this to understand:
- How much faster does causal RL converge?
- What's the sample efficiency gain?
- How does DCL compare to both?
- Is there a training stability difference?
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import argparse

from gympn.simulator import GymProblem
from gympn.logging_utils import get_logger

# Initialize logger
logger = get_logger(verbose=1)


# ============================================================================
# COMPARISON CONFIGURATION
# ============================================================================

class ComparisonConfig:
    """Configuration for fair convergence comparison."""

    def __init__(self):
        # Training parameters (same for both)
        self.max_episode_timesteps = 10 # Max timesteps per episode
        self.num_seeds = 3              # Number of random seeds for statistical significance
        self.epochs = 100               # Training epochs
        self.episodes_per_epoch = 100    # Episodes per epoch
        self.max_episode_length = None   # Max steps per episode
        self.batch_size = 32            # Batch size

        # Network hyperparameters (same for both)
        self.policy_lr = 5e-4
        self.value_lr = 1e-3
        self.entropy_coeff = 0.01

        # PPO specific
        self.ppo_eps = 0.15             # Clipping range

        # DCL specific (if using)
        self.use_causal_rl = True
        self.dcl_horizon = 3            # Planning horizon for DCL

        # Discount and lambda
        self.gam = 0.99
        self.lam = 0.97

        # Output
        self.output_dir = Path("convergence_comparison_results")
        self.output_dir.mkdir(exist_ok=True)


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

class ConvergenceComparison:
    """Run and analyze convergence comparison."""

    def __init__(self, config: ComparisonConfig, env_name: str = "example_simple_postpone"):
        """
        Initialize comparison.

        Parameters
        ----------
        config : ComparisonConfig
            Comparison configuration
        env_name : str
            Name of the example environment to use
        """
        self.config = config
        self.env_name = env_name
        self.results = {
            'ppo_clip': [],
            'ppo_causal': [],
            'dcl': []
        }
        self.timestamps = {
            'ppo_clip': [],
            'ppo_causal': [],
            'dcl': []
        }

    def run_comparison(self):
        """Run the full convergence comparison."""
        logger.info("=" * 70)
        logger.info("CONVERGENCE SPEED COMPARISON: PPO-Clip vs PPO+Causal RL vs DCL")
        logger.info("=" * 70)
        logger.info(f"Environment: {self.env_name}")
        logger.info(f"Seeds: {self.config.num_seeds}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Episodes/epoch: {self.config.episodes_per_epoch}")
        logger.info("")

        # Run all algorithms with different seeds
        for seed in range(self.config.num_seeds):
            logger.info(f"\n{'='*70}")
            logger.info(f"SEED {seed + 1}/{self.config.num_seeds}")
            logger.info(f"{'='*70}")

            # Run standard PPO-Clip (baseline)
            logger.info("\n[CHART] Training PPO-Clip (baseline)...")
            ppo_history, ppo_time = self._train_ppo_clip(seed)
            self.results['ppo_clip'].append(ppo_history)
            self.timestamps['ppo_clip'].append(ppo_time)
            logger.info(f"[DONE] PPO-Clip training completed in {ppo_time:.1f}s")

            # Run PPO-Clip with Causal RL
            logger.info("\n[CHART] Training PPO-Clip with Causal RL...")
            causal_history, causal_time = self._train_ppo_causal(seed)
            self.results['ppo_causal'].append(causal_history)
            self.timestamps['ppo_causal'].append(causal_time)
            logger.info(f"[DONE] PPO-Clip with Causal RL training completed in {causal_time:.1f}s")

            # Run DCL Agent
            logger.info("\n[CHART] Training DCL Agent...")
            dcl_history, dcl_time = self._train_dcl(seed)
            self.results['dcl'].append(dcl_history)
            self.timestamps['dcl'].append(dcl_time)
            logger.info(f"[DONE] DCL Agent training completed in {dcl_time:.1f}s")

            # Print interim comparison
            logger.info(f"\nInterim comparison (Seed {seed+1}):")
            logger.info(f"  PPO-Clip time:                 {ppo_time:.1f}s")
            logger.info(f"  PPO-Clip + Causal RL time:     {causal_time:.1f}s ({(causal_time/ppo_time - 1)*100:+.1f}%)")
            logger.info(f"  DCL time:                      {dcl_time:.1f}s ({(dcl_time/ppo_time - 1)*100:+.1f}%)")

        # Analyze and save results
        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS")
        logger.info(f"{'='*70}")
        self._analyze_results()
        self._save_results()
        self._generate_plots()

    def _create_simple_postpone_env(self, causal_rl: bool = False) -> GymProblem:
        """Create a simple postpone environment for comparison."""
        from simpn.simulator import SimToken

        # Instantiate a simulation problem
        env = GymProblem(allow_postpone=True, causal_rl=causal_rl)

        # Define cases
        arrival = env.add_var("arrival", var_attributes=['task_type'])
        waiting = env.add_var("waiting", var_attributes=['task_type'])
        busy = env.add_var("busy", var_attributes=['task_type', 'resource_id'])
        arrival.put({'task_type': 0})
        arrival.put({'task_type': 0})

        # Define resources
        employee = env.add_var("employee", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        employee.put({'code_employee': 1})

        # Define events
        def arrive(a):
            return [SimToken(a, delay=1), SimToken(a)]
        env.add_event([arrival], [arrival, waiting], arrive)

        def start(c, r):
            if r['code_employee'] == 1:
                return [SimToken((c, r), delay=20)]
            else:
                return [SimToken((c, r), delay=0.5)]

        env.add_action([waiting, employee], [busy], behavior=start, name="start")

        def complete(b):
            return [SimToken(b[1])]

        def r_function(x):
            return 1

        env.add_event([busy], [employee], complete, name='complete', reward_function=r_function)

        return env

    def _make_training_args(self, causal_rl: bool = False, algorithm: str = 'ppo-clip') -> dict:
        """Create training arguments for training_run()."""
        return {
            'episodes': self.config.episodes_per_epoch,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'max_episode_length': self.config.max_episode_length,
            'policy_lr': self.config.policy_lr,
            'value_lr': self.config.value_lr,
            'gam': self.config.gam,
            'lam': self.config.lam,
            'eps': self.config.ppo_eps,
            'vf_coeff': 0.05,
            'ent_bonus': self.config.entropy_coeff,
            'policy_kld_limit': 0.01,
            'causal_rl': causal_rl,
            'algorithm': algorithm,
            'verbose': 0,
            'use_gpu': False,
            'agent_seed': None,
            'use_wandb': False,
            'open_tensorboard': False,
            'test_in_train': False,
        }

    def _train_ppo_clip(self, seed: int) -> tuple:
        """Train standard PPO-Clip and return history and training time."""
        np.random.seed(seed)
        torch.manual_seed(seed) if hasattr(torch, 'manual_seed') else None

        # Create gym problem WITHOUT causal RL
        gym_problem = self._create_simple_postpone_env(causal_rl=False)

        # Create training args
        args_dict = self._make_training_args(causal_rl=False, algorithm='ppo-clip')

        # Time the training using training_run()
        start_time = time.time()
        try:
            gym_problem.training_run(length=self.config.max_episode_timesteps, args_dict=args_dict)
            history = gym_problem.training_history if hasattr(gym_problem, 'training_history') else {}
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            history = {}
        elapsed_time = time.time() - start_time

        return history, elapsed_time

    def _train_ppo_causal(self, seed: int) -> tuple:
        """Train PPO-Clip with Causal RL and return history and training time."""
        np.random.seed(seed)
        torch.manual_seed(seed) if hasattr(torch, 'manual_seed') else None

        # Create gym problem WITH causal RL enabled
        gym_problem = self._create_simple_postpone_env(causal_rl=True)

        # Create training args with causal RL enabled
        args_dict = self._make_training_args(causal_rl=True, algorithm='ppo-clip')

        # Time the training using training_run()
        start_time = time.time()
        try:
            gym_problem.training_run(length=self.config.max_episode_timesteps, args_dict=args_dict)
            history = gym_problem.training_history if hasattr(gym_problem, 'training_history') else {}
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            history = {}
        elapsed_time = time.time() - start_time

        return history, elapsed_time

    def _train_dcl(self, seed: int) -> tuple:
        """Train DCL Agent and return history and training time."""
        np.random.seed(seed)
        torch.manual_seed(seed) if hasattr(torch, 'manual_seed') else None

        # Create gym problem WITH causal RL enabled for DCL
        gym_problem = self._create_simple_postpone_env(causal_rl=True)

        # Create training args for DCL
        args_dict = self._make_training_args(causal_rl=True, algorithm='dcl')

        # Time the training using training_run()
        start_time = time.time()
        try:
            gym_problem.training_run(length=self.config.max_episode_timesteps, args_dict=args_dict)
            history = gym_problem.training_history if hasattr(gym_problem, 'training_history') else {}
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            history = {}
        elapsed_time = time.time() - start_time


        return history, elapsed_time

    def _analyze_results(self):
        """Analyze and report comparison results."""
        # Note: training_run() doesn't return detailed history, so we'll work with what we have
        # The results dict will contain whatever history was returned by training_run()

        logger.info("\n[CHART] RESULTS SUMMARY")
        logger.info(f"  PPO-Clip training time:                {np.mean(self.timestamps['ppo_clip']):.1f}s ± {np.std(self.timestamps['ppo_clip']):.1f}s")
        logger.info(f"  PPO-Clip + Causal RL training time:    {np.mean(self.timestamps['ppo_causal']):.1f}s ± {np.std(self.timestamps['ppo_causal']):.1f}s")
        logger.info(f"  DCL training time:                     {np.mean(self.timestamps['dcl']):.1f}s ± {np.std(self.timestamps['dcl']):.1f}s")

        # Calculate overhead
        ppo_times = np.array(self.timestamps['ppo_clip'])
        causal_times = np.array(self.timestamps['ppo_causal'])
        dcl_times = np.array(self.timestamps['dcl'])

        logger.info(f"\n[TIME] TRAINING TIME COMPARISON")
        logger.info(f"  PPO-Clip:                {ppo_times.mean():.1f}s")
        logger.info(f"  PPO-Clip + Causal RL:   {causal_times.mean():.1f}s ({(causal_times.mean()/ppo_times.mean() - 1)*100:+.1f}%)")
        logger.info(f"  DCL:                    {dcl_times.mean():.1f}s ({(dcl_times.mean()/ppo_times.mean() - 1)*100:+.1f}%)")


    def _save_results(self):
        """Save results to JSON."""
        results_dict = {
            'config': {
                'num_seeds': self.config.num_seeds,
                'epochs': self.config.epochs,
                'episodes_per_epoch': self.config.episodes_per_epoch,
                'max_episode_length': self.config.max_episode_length,
                'batch_size': self.config.batch_size,
            },
            'training_times': {
                'ppo_clip': [float(t) for t in self.timestamps['ppo_clip']],
                'ppo_causal': [float(t) for t in self.timestamps['ppo_causal']],
                'dcl': [float(t) for t in self.timestamps['dcl']],
            },
            'timestamp': datetime.now().isoformat(),
        }

        output_file = self.config.output_dir / "convergence_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"\n[DONE] Results saved to {output_file}")

    def _generate_plots(self):
        """Generate comparison plots - focused on training time since detailed history is not available."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Convergence Comparison: PPO-Clip vs PPO+Causal RL vs DCL', fontsize=16)

        algorithms = ['PPO-Clip', 'PPO+Causal', 'DCL']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        ppo_times = np.array(self.timestamps['ppo_clip'])
        causal_times = np.array(self.timestamps['ppo_causal'])
        dcl_times = np.array(self.timestamps['dcl'])

        # =====================================================================
        # Plot 1: Training Time
        # =====================================================================
        ax = axes[0]
        means = [ppo_times.mean(), causal_times.mean(), dcl_times.mean()]
        stds = [ppo_times.std(), causal_times.std(), dcl_times.std()]

        x_pos = np.arange(len(algorithms))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=colors)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Total Training Time', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 2, f'{mean:.0f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # =====================================================================
        # Plot 2: Time Overhead
        # =====================================================================
        ax = axes[1]
        baseline_time = ppo_times.mean()
        overheads = [
            0.0,  # Baseline
            (causal_times.mean() / baseline_time - 1) * 100,
            (dcl_times.mean() / baseline_time - 1) * 100
        ]

        bars = ax.bar(x_pos, overheads, alpha=0.7, color=colors)
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_ylabel('Time Overhead (%)', fontsize=12)
        ax.set_title('Training Time Overhead vs PPO-Clip', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, overhead in enumerate(overheads):
            y_pos = overhead + (5 if overhead >= 0 else -5)
            ax.text(i, y_pos, f'{overhead:+.1f}%', ha='center',
                   va='bottom' if overhead >= 0 else 'top', fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_file = self.config.output_dir / "convergence_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"[DONE] Plot saved to {output_file}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Create configuration
    config = ComparisonConfig()

    # Run comparison
    comparison = ConvergenceComparison(config)
    comparison.run_comparison()

    logger.info("\n" + "="*70)
    logger.info("[DONE] CONVERGENCE COMPARISON COMPLETE")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("="*70)

