"""
Convergence Comparison on the Two-Stage Assignment problem

This script mirrors `compare_convergence_speed.py` but runs the comparison
on the two-stage assignment environment defined in `example_two_stage_assignment.py`.

Defaults are conservative (few seeds/epochs) so you can run a quick smoke test.
Adjust `ComparisonConfig` at the top to run larger experiments.
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

logger = get_logger(verbose=1)


class ComparisonConfig:
    def __init__(self):
        # Small defaults for quick smoke test — increase for real experiments
        self.max_episode_timesteps = 10
        self.num_seeds = 10 
        # Run longer by default per user's request
        self.epochs = 30
        self.episodes_per_epoch = 20       # was 10: more data → lower-variance advantages
        self.max_episode_length = None
        self.batch_size = 32

        self.policy_lr = 3e-4
        self.value_lr = 1e-3               # was 1e-3: lower for γ=1 value targets (wider range)
        self.entropy_coeff = 0.005

        self.ppo_eps = 0.2
        self.use_causal_rl = True

        self.gam = 0.99
        self.lam = 0.95

        self.output_dir = Path("convergence_comparison_results")
        self.output_dir.mkdir(exist_ok=True)


class ConvergenceComparisonTwoStage:
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.results = {'ppo_clip': [], 'ppo_causal': []}
        self.timestamps = {'ppo_clip': [], 'ppo_causal': []}

    def run_comparison(self):
        logger.info("Starting comparison on Two-Stage Assignment")
        for seed in range(self.config.num_seeds):
            logger.info(f"Seed {seed+1}/{self.config.num_seeds}")
            causal_hist, causal_time = self._train(True, seed)
            self.results['ppo_causal'].append(causal_hist)
            self.timestamps['ppo_causal'].append(causal_time)

            ppo_hist, ppo_time = self._train(False, seed)
            self.results['ppo_clip'].append(ppo_hist)
            self.timestamps['ppo_clip'].append(ppo_time)

        self._analyze_results()
        self._save_results()
        self._generate_plots()

    def _make_args(self, causal_rl: bool):
        return {
            'episodes': self.config.episodes_per_epoch,
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'max_episode_length': self.config.max_episode_length,
            'policy_lr': self.config.policy_lr,
            'policy_updates': 2,             # fewer inner updates to prevent cumulative KLD drift
            'value_lr': self.config.value_lr,
            'value_updates': 10,             # (only used by standalone value training, not combined)
            'gam': self.config.gam,
            'lam': self.config.lam,
            'eps': self.config.ppo_eps,
            'vf_coeff': 0.5,
            'ent_bonus': self.config.entropy_coeff,
            'causal_rl': causal_rl,
            'algorithm': 'ppo-clip',
            'verbose': 1,
            'use_gpu': False,
            'agent_seed': None,
            'use_wandb': False,
            'open_tensorboard': False,
            'test_in_train': False,
            "save_freq": 1000000,
            'policy_kld_limit': 0.1,       # KL early stopping to prevent catastrophic updates
            'lr_schedule': True,            # cosine annealing LR decay
        }

    def _create_two_stage_env(self, causal_rl=False) -> GymProblem:
        # Build the two-stage problem same as example_two_stage_assignment.py
        from simpn.simulator import SimToken
        agency = GymProblem(allow_postpone=True, causal_rl=causal_rl)

        arrival = agency.add_var("arrival", var_attributes=['case_id'])
        waiting_A = agency.add_var("waiting_A", var_attributes=['case_id'])
        busy_A = agency.add_var("busy_A", var_attributes=['case_id', 'resource_id'])
        waiting_B = agency.add_var("waiting_B", var_attributes=['case_id'])
        busy_B = agency.add_var("busy_B", var_attributes=['case_id', 'resource_id'])

        arrival.put({'case_id': 0})
        arrival.put({'case_id': 0})

        resource = agency.add_var("resource", var_attributes=['resource_id'])
        resource.put({'resource_id': 0})
        resource.put({'resource_id': 1})

        def arrive(a):
            next_case = {'case_id': a['case_id']}
            return [SimToken(next_case, delay=1), SimToken(a)]
        agency.add_event([arrival], [arrival, waiting_A], arrive)

        def start_A(c, r):
            delay = 0.5 if r['resource_id'] == 0 else 2
            return [SimToken((c, r), delay=delay)]
        agency.add_action([waiting_A, resource], [busy_A], behavior=start_A, name="start_A")

        def complete_A(b):
            case, res = b
            return [SimToken(res), SimToken(case)]
        agency.add_event([busy_A], [resource, waiting_B], complete_A, name='complete_A')

        def start_B(c, r):
            delay = 0.5 if r['resource_id'] == 1 else 2
            return [SimToken((c, r), delay=delay)]
        agency.add_action([waiting_B, resource], [busy_B], behavior=start_B, name="start_B")

        def complete_B(b):
            _, res = b
            return [SimToken(res)]
        agency.add_event([busy_B], [resource], complete_B, name='complete_B', reward_function=lambda x: 1)

        return agency

    def _train(self, causal_rl: bool, seed: int):
        # Seed Python hash randomization for consistent hashing-based dict orders
        import os, random
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = self._create_two_stage_env(causal_rl=causal_rl)
        args = self._make_args(causal_rl)
        # propagate seed into simulator via args
        args['agent_seed'] = int(seed)
        start = time.time()
        try:
            env.training_run(length=self.config.max_episode_timesteps, args_dict=args)
            history = env.training_history if hasattr(env, 'training_history') else {}
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            history = {}
        elapsed = time.time() - start
        return history, elapsed

    # analysis + plotting simplified (reuse small parts from original file)
    def _extract_metric(self, results_list, key):
        arrays = []
        for h in results_list:
            if isinstance(h, dict) and key in h:
                arrays.append(np.asarray(h[key]))
        if not arrays:
            return None, None
        max_len = max(len(a) for a in arrays)
        padded = []
        for a in arrays:
            if len(a) < max_len:
                a = np.concatenate([a, np.full(max_len - len(a), np.nan)])
            padded.append(a)
        stacked = np.stack(padded, axis=0)
        return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)

    def _analyze_results(self):
        logger.info("Analysis summary:")
        for label, key in [("PPO-Clip", 'ppo_clip'), ("PPO+Causal", 'ppo_causal')]:
            mean_ret, std_ret = self._extract_metric(self.results[key], 'mean_returns')
            if mean_ret is None:
                logger.info(f"  {label}: no data")
                continue
            logger.info(f"  {label}: final mean return = {mean_ret[-1]:.2f}, best = {np.max(mean_ret):.2f}")

    def _save_results(self):
        out = {
            'config': vars(self.config),
            'timestamps': self.timestamps,
            'results': [self.results],
            'timestamp': datetime.now().isoformat(),
        }
        # JSON can't serialize some objects (Path, numpy arrays, torch tensors, etc.).
        # Convert common non-serializable types recursively to plain Python types.
        def _make_serializable(o):
            # Path -> str
            if isinstance(o, Path):
                return str(o)
            # numpy arrays -> lists
            if isinstance(o, np.ndarray):
                return o.tolist()
            # numpy scalar types
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            # torch tensors -> lists (cpu)
            try:
                import torch as _torch
            except Exception:
                _torch = None
            if _torch is not None and isinstance(o, _torch.Tensor):
                return o.detach().cpu().tolist()
            # datetime -> iso
            if isinstance(o, datetime):
                return o.isoformat()
            # dict -> recurse
            if isinstance(o, dict):
                return { _make_serializable(k): _make_serializable(v) for k, v in o.items() }
            # list/tuple/set -> list
            if isinstance(o, (list, tuple, set)):
                return [ _make_serializable(x) for x in o ]
            # fall back for JSON-serializable primitives
            try:
                json.dumps(o)
                return o
            except TypeError:
                return str(o)

        serializable_out = _make_serializable(out)
        out_file = Path(_make_serializable(self.config.output_dir)) / 'two_stage_compare.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_out, f, indent=2)
        logger.info(f"Saved results to {out_file}")

    def _generate_plots(self):
        # More comprehensive plotting: per-seed faint lines + mean ± std band
        mean_c, std_c = self._extract_metric(self.results['ppo_causal'], 'mean_returns')
        mean_p, std_p = self._extract_metric(self.results['ppo_clip'], 'mean_returns')

        plt.figure(figsize=(10,6))
        colors = {'ppo_clip': '#1f77b4', 'ppo_causal': '#ff7f0e'}

        # Plot per-seed lines (if available) in faint colors for visual spread
        for label_key, display_name in [('ppo_clip', 'PPO-Clip'), ('ppo_causal', 'PPO+Causal')]:
            runs = self.results.get(label_key, [])
            for run in runs:
                if isinstance(run, dict) and 'mean_returns' in run:
                    arr = np.asarray(run['mean_returns'])
                    epochs = np.arange(1, len(arr)+1)
                    plt.plot(epochs, arr, color=colors[label_key], alpha=0.18, linewidth=1)

        # Plot mean ± std and mean line for each method
        if mean_p is not None:
            epochs = np.arange(1, len(mean_p)+1)
            plt.plot(epochs, mean_p, label='PPO-Clip (mean)', color=colors['ppo_clip'], linewidth=2, marker='o')
            plt.fill_between(epochs, mean_p - (std_p if std_p is not None else 0),
                             mean_p + (std_p if std_p is not None else 0),
                             color=colors['ppo_clip'], alpha=0.18)

        if mean_c is not None:
            epochs = np.arange(1, len(mean_c)+1)
            plt.plot(epochs, mean_c, label='PPO+Causal (mean)', color=colors['ppo_causal'], linewidth=2, marker='s')
            plt.fill_between(epochs, mean_c - (std_c if std_c is not None else 0),
                             mean_c + (std_c if std_c is not None else 0),
                             color=colors['ppo_causal'], alpha=0.18)

        # Compute & plot best-per-epoch across both methods (max of means)
        combined_means = []
        if mean_p is not None:
            combined_means.append(mean_p)
        if mean_c is not None:
            combined_means.append(mean_c)
        if combined_means:
            stacked = np.stack(combined_means, axis=0)
            best_per_epoch = np.max(stacked, axis=0)
            epochs = np.arange(1, len(best_per_epoch)+1)
            plt.plot(epochs, best_per_epoch, label='Best per-epoch', color='green', linestyle='--', linewidth=1.5)

        plt.xlabel('Epoch')
        plt.ylabel('Mean Return')
        plt.title('Convergence comparison — Two-Stage Assignment')
        plt.grid(alpha=0.3)
        plt.legend()

        # Timestamped output filenames
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_png = self.config.output_dir / f'two_stage_compare_{ts}.png'
        out_pdf = self.config.output_dir / f'two_stage_compare_{ts}.pdf'
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_pdf)
        logger.info(f"Saved plots to {out_png} and {out_pdf}")


if __name__ == '__main__':
    cfg = ComparisonConfig()
    cmp = ConvergenceComparisonTwoStage(cfg)
    cmp.run_comparison()
    logger.info('Done')


