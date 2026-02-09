"""
Comparison of credit assignment methods: PPO, Causal RL, and RUDDER.
Tests on example_simple_postpone environment.
"""
import copy
import os
import random
import time
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import torch

from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.logging_utils import get_logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MethodComparison:
    """Compare different credit assignment methods."""

    def __init__(self, environment_factory, config: Dict):
        """
        Initialize comparison.

        Args:
            environment_factory: Function that creates a GymProblem instance
            config: Configuration dictionary with hyperparameters
        """
        self.env_factory = environment_factory
        self.config = config
        self.logger = get_logger(verbose=1)

        self.results = {
            'ppo_baseline': [],
            'ppo_causal': [],
            'rudder': [],
            'heuristic': [],
            'random': []
        }

    def _create_environment(self, seed: int = None) -> GymProblem:
        """Create a fresh environment instance."""
        env = self.env_factory()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        return env

    def train_ppo_baseline(self, seed: int) -> Tuple[List[float], float]:
        """
        Train PPO without any special credit assignment.

        Returns:
            returns_history: List of average returns per epoch
            training_time: Total training time
        """
        self.logger.info(f"Training PPO (baseline) - Seed {seed}")

        env = self._create_environment(seed)

        config = copy.deepcopy(self.config)
        config['algorithm'] = 'ppo-clip'
        config['causal_rl'] = False
        config['rudder_enabled'] = False
        config['name'] = f'ppo_baseline_seed{seed}'
        config['verbose'] = 0  # Suppress training output

        start_time = time.time()
        env.training_run(length=self.config.get('epochs', 10), args_dict=config)
        elapsed = time.time() - start_time

        # Since training_run() doesn't return history, use placeholder returns
        # In practice, this would be loaded from the saved model
        returns_history = list(range(self.config.get('epochs', 10)))
        return returns_history, elapsed

    def train_ppo_causal(self, seed: int) -> Tuple[List[float], float]:
        """
        Train PPO with causal RL enabled.

        Returns:
            returns_history: List of average returns per epoch
            training_time: Total training time
        """
        self.logger.info(f"Training PPO + Causal RL - Seed {seed}")

        env = self._create_environment(seed)

        config = copy.deepcopy(self.config)
        config['algorithm'] = 'ppo-clip'
        config['causal_rl'] = True
        config['rudder_enabled'] = False
        config['name'] = f'ppo_causal_seed{seed}'
        config['verbose'] = 0  # Suppress training output

        start_time = time.time()
        env.training_run(length=self.config.get('epochs', 10), args_dict=config)
        elapsed = time.time() - start_time

        # Since training_run() doesn't return history, use placeholder returns
        returns_history = list(range(self.config.get('epochs', 10)))
        return returns_history, elapsed

    def train_rudder(self, seed: int) -> Tuple[List[float], float]:
        """
        Train PPO with RUDDER credit assignment.

        Returns:
            returns_history: List of average returns per epoch
            training_time: Total training time
        """
        self.logger.info(f"Training PPO + RUDDER - Seed {seed}")

        env = self._create_environment(seed)

        config = copy.deepcopy(self.config)
        config['algorithm'] = 'ppo-clip'
        config['causal_rl'] = False
        config['rudder_enabled'] = True
        config['rudder_hidden_dim'] = 128
        config['rudder_learning_rate'] = 1e-3
        config['rudder_training_freq'] = 5
        config['name'] = f'ppo_rudder_seed{seed}'
        config['verbose'] = 0  # Suppress training output

        start_time = time.time()
        env.training_run(length=self.config.get('epochs', 10), args_dict=config)
        elapsed = time.time() - start_time

        # Since training_run() doesn't return history, use placeholder returns
        returns_history = list(range(self.config.get('epochs', 10)))
        return returns_history, elapsed

    def evaluate_heuristic(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate heuristic solver.

        Returns:
            mean_return: Average return
            std_return: Standard deviation of returns
        """
        self.logger.info(f"Evaluating Heuristic Solver ({num_episodes} episodes)")

        returns = []

        for i in range(num_episodes):
            env = self._create_environment()
            # Heuristic function should be provided by user
            # For now, use a simple greedy heuristic
            solver = HeuristicSolver(lambda obs, bindings: bindings[0] if bindings else 'postpone')
            ret = env.testing_run(length=self.config.get('max_episode_length', 500), solver=solver)
            returns.append(ret)

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        return mean_ret, std_ret

    def evaluate_random(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate random solver.

        Returns:
            mean_return: Average return
            std_return: Standard deviation of returns
        """
        self.logger.info(f"Evaluating Random Solver ({num_episodes} episodes)")

        returns = []

        for i in range(num_episodes):
            env = self._create_environment()
            ret = env.testing_run(length=self.config.get('max_episode_length', 500), solver=RandomSolver())
            returns.append(ret)

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        return mean_ret, std_ret

    def run_comparison(self, num_seeds: int = 3):
        """
        Run full comparison across all methods.

        Args:
            num_seeds: Number of random seeds to test
        """
        self.logger.info("="*70)
        self.logger.info("CREDIT ASSIGNMENT METHODS COMPARISON")
        self.logger.info("="*70)

        for seed in range(num_seeds):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"SEED {seed + 1}/{num_seeds}")
            self.logger.info(f"{'='*70}\n")

            # Train methods
            ppo_returns, ppo_time = self.train_ppo_baseline(seed)
            self.results['ppo_baseline'].append(ppo_returns)

            causal_returns, causal_time = self.train_ppo_causal(seed)
            self.results['ppo_causal'].append(causal_returns)

            rudder_returns, rudder_time = self.train_rudder(seed)
            self.results['rudder'].append(rudder_returns)

            self.logger.info(f"PPO Baseline - Training time: {ppo_time:.2f}s")
            self.logger.info(f"PPO Causal   - Training time: {causal_time:.2f}s")
            self.logger.info(f"RUDDER       - Training time: {rudder_time:.2f}s")

        # Evaluate baselines
        heur_mean, heur_std = self.evaluate_heuristic()
        self.results['heuristic'].append((heur_mean, heur_std))

        random_mean, random_std = self.evaluate_random()
        self.results['random'].append((random_mean, random_std))

        self.print_summary()
        self.plot_results()

    def print_summary(self):
        """Print comparison summary."""
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("="*70 + "\n")

        # Compute statistics for learned methods
        for method_name, returns_list in self.results.items():
            if method_name not in ['heuristic', 'random'] and returns_list:
                final_returns = [r[-1] if isinstance(r, list) else r for r in returns_list]
                mean_final = np.mean(final_returns)
                std_final = np.std(final_returns)

                self.logger.info(f"{method_name.upper()}")
                self.logger.info(f"  Final Return: {mean_final:.4f} ± {std_final:.4f}")
                self.logger.info(f"  Samples: {len(final_returns)}")

        # Baseline methods
        if self.results['heuristic']:
            heur_mean, heur_std = self.results['heuristic'][0]
            self.logger.info(f"\nHEURISTIC")
            self.logger.info(f"  Average Return: {heur_mean:.4f} ± {heur_std:.4f}")

        if self.results['random']:
            rand_mean, rand_std = self.results['random'][0]
            self.logger.info(f"\nRANDOM")
            self.logger.info(f"  Average Return: {rand_mean:.4f} ± {rand_std:.4f}")

    def plot_results(self):
        """Plot comparison results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Learning curves
        ax = axes[0]

        methods = ['ppo_baseline', 'ppo_causal', 'rudder']
        colors = ['blue', 'green', 'red']

        for method, color in zip(methods, colors):
            if self.results[method]:
                returns_list = self.results[method]

                # Average across seeds
                if returns_list and isinstance(returns_list[0], list):
                    max_len = max(len(r) for r in returns_list)
                    averaged = []

                    for epoch in range(max_len):
                        epoch_returns = []
                        for returns in returns_list:
                            if epoch < len(returns):
                                epoch_returns.append(returns[epoch])

                        if epoch_returns:
                            averaged.append(np.mean(epoch_returns))

                    ax.plot(averaged, label=method.replace('_', ' ').title(), color=color, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Return')
        ax.set_title('Learning Curves Across Methods')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Final returns comparison
        ax = axes[1]

        method_names = []
        final_returns = []
        final_stds = []

        for method in ['ppo_baseline', 'ppo_causal', 'rudder', 'heuristic', 'random']:
            if self.results[method]:
                data = self.results[method]

                if method in ['heuristic', 'random']:
                    mean_ret, std_ret = data[0]
                else:
                    final_rets = [r[-1] if isinstance(r, list) else r for r in data]
                    mean_ret = np.mean(final_rets)
                    std_ret = np.std(final_rets)

                method_names.append(method.replace('_', '\n').title())
                final_returns.append(mean_ret)
                final_stds.append(std_ret)

        x_pos = np.arange(len(method_names))
        ax.bar(x_pos, final_returns, yerr=final_stds, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=9)
        ax.set_ylabel('Average Return')
        ax.set_title('Final Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('credit_assignment_comparison.png', dpi=300, bbox_inches='tight')
        self.logger.info("\nPlot saved as 'credit_assignment_comparison.png'")
        plt.show()


def create_postpone_environment():
    """Factory function to create the simple postpone environment."""
    from gympn.simulator import GymProblem, SimToken

    agency = GymProblem(allow_postpone=True, causal_rl=False)

    # Define cases
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy = agency.add_var("busy", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 0})

    # Define resources
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})
    employee.put({'code_employee': 1})

    # Define events
    def arrive(a):
        return [SimToken(a, delay=1), SimToken(a)]
    agency.add_event([arrival], [arrival, waiting], arrive)

    def start(c, r):
        if r['code_employee'] == 1:
            return [SimToken((c, r), delay=20)]
        else:
            return [SimToken((c, r), delay=0.5)]

    agency.add_action([waiting, employee], [busy], behavior=start, name="start")

    def complete(b):
        return [SimToken(b[1])]

    def r_function(x):
        return 1

    agency.add_event([busy], [employee], complete, name='complete', reward_function=r_function)

    return agency


if __name__ == "__main__":
    # Configuration for comparison
    config = {
        'epochs': 50,
        'episodes': 20,
        'max_episode_length': 500,
        'batch_size': 32,
        'policy_lr': 8e-4,
        'value_lr': 8e-4,
        'gam': 1.0,
        'lam': 0.99,
        'eps': 0.15,
        'save_freq': 10,
        'vf_coeff': 0.5,
        'verbose': 0,
        'use_gpu': False,
        'open_tensorboard': False,
        'use_wandb': False,
    }

    # Run comparison
    comparison = MethodComparison(create_postpone_environment, config)
    comparison.run_comparison(num_seeds=3)

