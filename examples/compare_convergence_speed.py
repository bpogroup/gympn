"""
Convergence Speed Comparison: PPO-Clip vs PPO-Clip with Causal RL

This script provides a fair, controlled comparison of convergence speed between:
1. Standard PPO-Clip (baseline)
2. PPO-Clip with Causal RL enabled (causal reward redistribution)

Setup ensures:
- Same environment and hyperparameters (except algorithm-specific)
- Same number of episodes and epochs
- Same network architecture
- Multiple seeds for statistical significance
- Detailed metrics and visualization with learning curves

Run this to understand:
- How much faster does causal RL converge?
- What's the sample efficiency gain?
- Is there a training stability difference?
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

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
        self.max_episode_timesteps = 10 # Simulation time horizon (NOT step count)
        self.num_seeds = 3              # Number of random seeds for statistical significance
        self.epochs = 50               # Training epochs
        self.episodes_per_epoch = 50    # Episodes per epoch (more for better signal)
        self.max_episode_length = None   # Max steps per episode
        self.batch_size = 32            # Batch size

        # Network hyperparameters (same for both)
        self.policy_lr = 3e-4
        self.value_lr = 1e-3
        self.entropy_coeff = 0.05       # Higher entropy for better exploration

        # PPO specific
        self.ppo_eps = 0.15             # Clipping range

        # Causal RL specific
        self.use_causal_rl = True

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
        self.config = config
        self.env_name = env_name
        # Store full history dicts per seed
        self.results = {
            'ppo_clip': [],
            'ppo_causal': [],
        }
        self.timestamps = {
            'ppo_clip': [],
            'ppo_causal': [],
        }

    def run_comparison(self):
        """Run the full convergence comparison."""
        logger.info("=" * 70)
        logger.info("CONVERGENCE SPEED COMPARISON: PPO-Clip vs PPO+Causal RL")
        logger.info("=" * 70)
        logger.info(f"Environment: {self.env_name}")
        logger.info(f"Seeds: {self.config.num_seeds}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Episodes/epoch: {self.config.episodes_per_epoch}")
        logger.info("")

        for seed in range(self.config.num_seeds):
            logger.info(f"\n{'='*70}")
            logger.info(f"SEED {seed + 1}/{self.config.num_seeds}")
            logger.info(f"{'='*70}")

            # Run PPO-Clip with Causal RL first
            logger.info("\n[CHART] Training PPO-Clip with Causal RL...")
            causal_history, causal_time = self._train_ppo_causal(seed)
            self.results['ppo_causal'].append(causal_history)
            self.timestamps['ppo_causal'].append(causal_time)
            logger.info(f"[DONE] PPO-Clip with Causal RL training completed in {causal_time:.1f}s")

            # Run standard PPO-Clip (baseline)
            logger.info("\n[CHART] Training PPO-Clip (baseline)...")
            ppo_history, ppo_time = self._train_ppo_clip(seed)
            self.results['ppo_clip'].append(ppo_history)
            self.timestamps['ppo_clip'].append(ppo_time)
            logger.info(f"[DONE] PPO-Clip training completed in {ppo_time:.1f}s")

            # Print interim comparison
            logger.info(f"\nInterim comparison (Seed {seed + 1}):")
            logger.info(f"  PPO-Clip + Causal RL time:     {causal_time:.1f}s")
            logger.info(f"  PPO-Clip time:                 {ppo_time:.1f}s")
            if ppo_time > 0:
                logger.info(
                    f"  Causal RL overhead:            {(causal_time / ppo_time - 1) * 100:+.1f}%")

        # Analyze and save results
        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS")
        logger.info(f"{'='*70}")
        self._analyze_results()
        self._save_results()
        self._generate_plots()

    # ========================================================================
    # Environment factory
    # ========================================================================

    def _create_simple_postpone_env(self, causal_rl: bool = False) -> GymProblem:
        """Create a simple postpone environment for comparison."""
        from simpn.simulator import SimToken

        env = GymProblem(allow_postpone=True, causal_rl=causal_rl)

        arrival = env.add_var("arrival", var_attributes=['task_type'])
        waiting = env.add_var("waiting", var_attributes=['task_type'])
        busy = env.add_var("busy", var_attributes=['task_type', 'resource_id'])
        arrival.put({'task_type': 0})
        arrival.put({'task_type': 0})

        employee = env.add_var("employee", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        employee.put({'code_employee': 1})

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

    # ========================================================================
    # Training helpers
    # ========================================================================

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
        """Train standard PPO-Clip and return (history_dict, elapsed_seconds)."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        gym_problem = self._create_simple_postpone_env(causal_rl=False)
        args_dict = self._make_training_args(causal_rl=False, algorithm='ppo-clip')

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
        """Train PPO-Clip with Causal RL and return (history_dict, elapsed_seconds)."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        gym_problem = self._create_simple_postpone_env(causal_rl=True)
        args_dict = self._make_training_args(causal_rl=True, algorithm='ppo-clip')

        start_time = time.time()
        try:
            gym_problem.training_run(length=self.config.max_episode_timesteps, args_dict=args_dict)
            history = gym_problem.training_history if hasattr(gym_problem, 'training_history') else {}
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            history = {}
        elapsed_time = time.time() - start_time

        return history, elapsed_time

    # ========================================================================
    # Analysis helpers
    # ========================================================================

    @staticmethod
    def _extract_metric(results_list, key):
        """
        From a list of history dicts (one per seed), extract the numpy arrays
        for *key* and stack them into a 2-D array  (num_seeds x epochs).
        Returns (mean_per_epoch, std_per_epoch) or (None, None) if missing.
        """
        arrays = []
        for h in results_list:
            if isinstance(h, dict) and key in h:
                arrays.append(np.asarray(h[key]))
        if not arrays:
            return None, None
        # Pad to same length if needed
        max_len = max(len(a) for a in arrays)
        padded = []
        for a in arrays:
            if len(a) < max_len:
                a = np.concatenate([a, np.full(max_len - len(a), np.nan)])
            padded.append(a)
        stacked = np.stack(padded, axis=0)  # (seeds, epochs)
        return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)

    def _first_epoch_above(self, results_list, key, threshold):
        """Return the first epoch (1-indexed) where *key* >= threshold for every seed."""
        epochs = []
        for h in results_list:
            if isinstance(h, dict) and key in h:
                arr = np.asarray(h[key])
                idx = np.where(arr >= threshold)[0]
                if len(idx) > 0:
                    epochs.append(idx[0] + 1)  # 1-indexed
                else:
                    epochs.append(None)
        return epochs

    def _analyze_results(self):
        """Analyze and report comparison results."""
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)

        for label, key in [("PPO-Clip", "ppo_clip"), ("PPO-Clip + Causal RL", "ppo_causal")]:
            mean_ret, std_ret = self._extract_metric(self.results[key], 'mean_returns')
            if mean_ret is not None:
                final = mean_ret[-1]
                best = np.max(mean_ret)
                logger.info(f"\n  {label}:")
                logger.info(f"    Final mean return : {final:.2f}")
                logger.info(f"    Best  mean return : {best:.2f}")
                logger.info(f"    Training time     : {np.mean(self.timestamps[key]):.1f}s +/- {np.std(self.timestamps[key]):.1f}s")

                # Convergence epoch (>= 20.5 for near-optimal)
                epochs_to = self._first_epoch_above(self.results[key], 'mean_returns', 20.5)
                valid = [e for e in epochs_to if e is not None]
                if valid:
                    logger.info(f"    Epochs to >=20.5  : {np.mean(valid):.1f} +/- {np.std(valid):.1f}")
                else:
                    logger.info(f"    Epochs to >=20.5  : not reached")
            else:
                logger.info(f"\n  {label}: no history available")

        # Time overhead
        ppo_times = np.array(self.timestamps['ppo_clip'])
        causal_times = np.array(self.timestamps['ppo_causal'])
        if ppo_times.mean() > 0:
            overhead = (causal_times.mean() / ppo_times.mean() - 1) * 100
            logger.info(f"\n  Wall-clock overhead of Causal RL: {overhead:+.1f}%")

    def _save_results(self):
        """Save results to JSON (including per-epoch curves)."""

        def _serialise_history(h):
            if not isinstance(h, dict):
                return {}
            return {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in h.items()}

        results_dict = {
            'config': {
                'num_seeds': self.config.num_seeds,
                'epochs': self.config.epochs,
                'episodes_per_epoch': self.config.episodes_per_epoch,
                'max_episode_length': self.config.max_episode_length,
                'batch_size': self.config.batch_size,
                'policy_lr': self.config.policy_lr,
                'value_lr': self.config.value_lr,
                'gam': self.config.gam,
                'lam': self.config.lam,
            },
            'training_times': {
                'ppo_clip': [float(t) for t in self.timestamps['ppo_clip']],
                'ppo_causal': [float(t) for t in self.timestamps['ppo_causal']],
            },
            'histories': {
                'ppo_clip': [_serialise_history(h) for h in self.results['ppo_clip']],
                'ppo_causal': [_serialise_history(h) for h in self.results['ppo_causal']],
            },
            'timestamp': datetime.now().isoformat(),
        }

        output_file = self.config.output_dir / "convergence_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"\n[DONE] Results saved to {output_file}")

    # ========================================================================
    # Plotting  (learning curves + summary bar charts)
    # ========================================================================

    def _generate_plots(self):
        """Generate publication-quality learning curves and summary bar charts."""

        methods = {
            'PPO-Clip (baseline)': ('ppo_clip', '#1f77b4'),
            'PPO-Clip + Causal RL': ('ppo_causal', '#ff7f0e'),
        }

        # ------------------------------------------------------------------
        # Figure with 4 sub-plots:
        #   (1) Learning curve – mean return
        #   (2) Learning curve – policy entropy
        #   (3) Epochs-to-threshold bar chart
        #   (4) Training wall-clock time bar chart
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Comparison: PPO-Clip vs PPO-Clip + Causal RL',
                      fontsize=16, fontweight='bold', y=0.98)

        # ==================================================================
        # (1) Learning curve – mean return +/- std  (shaded)
        # ==================================================================
        ax = axes[0, 0]
        for label, (key, color) in methods.items():
            mean_ret, std_ret = self._extract_metric(self.results[key], 'mean_returns')
            if mean_ret is not None:
                epochs = np.arange(1, len(mean_ret) + 1)
                ax.plot(epochs, mean_ret, label=label, color=color, linewidth=2)
                ax.fill_between(epochs, mean_ret - std_ret, mean_ret + std_ret,
                                alpha=0.2, color=color)
        # Reference lines
        ax.axhline(y=21, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Optimal (21)')
        ax.axhline(y=19, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Suboptimal (19)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Return', fontsize=12)
        ax.set_title('Learning Curve – Mean Return per Epoch', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

        # ==================================================================
        # (2) Learning curve – policy entropy
        # ==================================================================
        ax = axes[0, 1]
        for label, (key, color) in methods.items():
            mean_ent, std_ent = self._extract_metric(self.results[key], 'policy_ent')
            if mean_ent is not None:
                epochs = np.arange(1, len(mean_ent) + 1)
                ax.plot(epochs, mean_ent, label=label, color=color, linewidth=2)
                ax.fill_between(epochs, mean_ent - std_ent, mean_ent + std_ent,
                                alpha=0.2, color=color)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Policy Entropy', fontsize=12)
        ax.set_title('Policy Entropy over Training', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ==================================================================
        # (3) Epochs to reach various return thresholds
        # ==================================================================
        ax = axes[1, 0]
        thresholds = [17, 18, 19, 20, 21]
        x_pos = np.arange(len(thresholds))
        bar_width = 0.35

        for i, (label, (key, color)) in enumerate(methods.items()):
            means = []
            stds = []
            for thr in thresholds:
                ep_list = self._first_epoch_above(self.results[key], 'mean_returns', thr)
                valid = [e for e in ep_list if e is not None]
                if valid:
                    means.append(np.mean(valid))
                    stds.append(np.std(valid))
                else:
                    means.append(self.config.epochs)  # did not reach
                    stds.append(0)
            offset = -bar_width / 2 + i * bar_width
            ax.bar(x_pos + offset, means, bar_width, yerr=stds, capsize=4,
                   label=label, color=color, alpha=0.8)
            # Value labels on bars
            for j, (m, s) in enumerate(zip(means, stds)):
                if m < self.config.epochs:
                    ax.text(x_pos[j] + offset, m + s + 0.5, f'{m:.0f}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
                else:
                    ax.text(x_pos[j] + offset, m + 0.5, 'X',
                            ha='center', va='bottom', fontsize=10, color='red')

        ax.set_xlabel('Return Threshold', fontsize=12)
        ax.set_ylabel('Epochs to Reach Threshold', fontsize=12)
        ax.set_title('Sample Efficiency – Epochs to Threshold', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'>={t}' for t in thresholds], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # ==================================================================
        # (4) Training wall-clock time
        # ==================================================================
        ax = axes[1, 1]
        labels_list = list(methods.keys())
        colors_list = [v[1] for v in methods.values()]
        keys_list = [v[0] for v in methods.values()]

        time_means = [np.mean(self.timestamps[k]) for k in keys_list]
        time_stds = [np.std(self.timestamps[k]) for k in keys_list]

        x_pos = np.arange(len(labels_list))
        ax.bar(x_pos, time_means, yerr=time_stds, capsize=10,
               alpha=0.8, color=colors_list)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Total Training Time', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (m, s) in enumerate(zip(time_means, time_stds)):
            ax.text(i, m + s + 2, f'{m:.0f}s', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        output_file = self.config.output_dir / "convergence_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"[DONE] Plot saved to {output_file}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    config = ComparisonConfig()

    comparison = ConvergenceComparison(config)
    comparison.run_comparison()

    logger.info("\n" + "=" * 70)
    logger.info("[DONE] CONVERGENCE COMPARISON COMPLETE")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("=" * 70)

