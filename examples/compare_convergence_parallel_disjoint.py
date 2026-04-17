"""
Convergence Comparison on the Parallel-Disjoint Assignment problem

This script mirrors `compare_convergence_two_stage.py` but runs the comparison
on the parallel-disjoint environment defined in
`paper_examples/d_parallel_disjoint.py`.

Problem overview
----------------
Two task types (0 and 1) arrive continuously.  Each type is routed to its own
waiting queue and served by its own pool of three employees (0, 1, 2).
Employee 0 is fastest for type-0 tasks, employee 1 is fastest for type-1 tasks,
and employee 2 is slow for both types.  A reward of +1 is granted when both
sub-tasks belonging to the same case are completed.

Defaults are conservative (few seeds / epochs) for a quick smoke test.
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


# ── configuration ────────────────────────────────────────────────────────────

class ComparisonConfig:
    def __init__(self):
        # Small defaults for quick smoke test — increase for real experiments
        self.max_episode_timesteps = 10
        self.num_seeds = 10
        self.epochs = 30
        self.episodes_per_epoch = 20
        self.max_episode_length = None
        self.batch_size = 32

        self.policy_lr = 3e-4
        self.value_lr = 3e-4
        self.entropy_coeff = 0.01

        self.ppo_eps = 0.2
        self.use_causal_rl = True

        self.gam = 0.99
        self.lam = 0.95

        self.output_dir = Path("convergence_comparison_results")
        self.output_dir.mkdir(exist_ok=True)


# ── main comparison class ────────────────────────────────────────────────────

class ConvergenceComparisonParallelDisjoint:
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.results = {"ppo_clip": [], "ppo_causal": []}
        self.timestamps = {"ppo_clip": [], "ppo_causal": []}

    # ── public entry-point ────────────────────────────────────────────────

    def run_comparison(self):
        logger.info("Starting comparison on Parallel-Disjoint Assignment")
        for seed in range(self.config.num_seeds):
            logger.info(f"Seed {seed + 1}/{self.config.num_seeds}")

            causal_hist, causal_time = self._train(True, seed)
            self.results["ppo_causal"].append(causal_hist)
            self.timestamps["ppo_causal"].append(causal_time)

            ppo_hist, ppo_time = self._train(False, seed)
            self.results["ppo_clip"].append(ppo_hist)
            self.timestamps["ppo_clip"].append(ppo_time)

        self._analyze_results()
        self._save_results()
        self._generate_plots()

    # ── training args ─────────────────────────────────────────────────────

    def _make_args(self, causal_rl: bool):
        return {
            "episodes": self.config.episodes_per_epoch,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "max_episode_length": self.config.max_episode_length,
            "policy_lr": self.config.policy_lr,
            "policy_updates": 5,
            "value_lr": self.config.value_lr,
            "value_updates": 10,
            "gam": self.config.gam,
            "lam": self.config.lam,
            "eps": self.config.ppo_eps,
            "vf_coeff": 0.5,
            "ent_bonus": self.config.entropy_coeff,
            "causal_rl": causal_rl,
            "algorithm": "ppo-clip",
            "verbose": 1,
            "use_gpu": False,
            "agent_seed": None,
            "use_wandb": False,
            "open_tensorboard": False,
            "test_in_train": False,
            "save_freq": 1_000_000,
        }

    # ── environment factory ───────────────────────────────────────────────

    def _create_parallel_disjoint_env(self, causal_rl: bool = False) -> GymProblem:
        """Build the parallel-disjoint problem (mirrors paper_examples/d_parallel_disjoint.py)."""
        from simpn.simulator import SimToken

        agency = GymProblem(allow_postpone=True, causal_rl=causal_rl)

        # ── places ──
        arrival = agency.add_var("arrival", var_attributes=["task_type", "case_id"])
        waiting1 = agency.add_var("waiting1", var_attributes=["task_type", "case_id"])
        busy1 = agency.add_var("busy1", var_attributes=["task_type", "code_employee", "case_id"])
        waiting2 = agency.add_var("waiting2", var_attributes=["task_type", "case_id"])
        busy2 = agency.add_var("busy2", var_attributes=["task_type", "code_employee", "case_id"])
        completed1 = agency.add_var("completed1", var_attributes=["task_type", "case_id"])
        completed2 = agency.add_var("completed2", var_attributes=["task_type", "case_id"])

        # Initial tokens – one of each task type
        arrival.put({"task_type": 0, "case_id": 0})
        arrival.put({"task_type": 1, "case_id": 0})

        # ── resources (disjoint pools) ──
        employee1 = agency.add_var("employee1", var_attributes=["code_employee"])
        for eid in range(3):
            employee1.put({"code_employee": eid})

        employee2 = agency.add_var("employee2", var_attributes=["code_employee"])
        for eid in range(3):
            employee2.put({"code_employee": eid})

        # ── arrival event: route task-type to correct queue ──
        def arrive(a):
            a["case_id"] += 1
            if a["task_type"] == 0:
                return [SimToken(a, delay=1), SimToken(a), None]
            else:  # task_type == 1
                return [SimToken(a, delay=1), None, SimToken(a)]

        agency.add_event([arrival], [arrival, waiting1, waiting2], arrive)

        # ── start behaviour (shared by both pools) ──
        def start(c, r):
            if (c["task_type"] == 0 and r["code_employee"] == 0) or \
               (c["task_type"] == 1 and r["code_employee"] == 1):
                return [SimToken((c, r), delay=1)]
            elif (c["task_type"] == 0 and r["code_employee"] == 1) or \
                 (c["task_type"] == 1 and r["code_employee"] == 0):
                return [SimToken((c, r), delay=2)]
            else:
                return [SimToken((c, r), delay=3)]

        # ── pool 1 ──
        agency.add_action([waiting1, employee1], [busy1], behavior=start, name="start1")

        def complete1(b):
            return [SimToken(b[-1]), SimToken(b[0])]

        agency.add_event([busy1], [employee1, completed1], complete1, name="done1")

        # ── pool 2 ──
        agency.add_action([waiting2, employee2], [busy2], behavior=start, name="start2")

        def complete2(b):
            return [SimToken(b[-1]), SimToken(b[0])]

        agency.add_event([busy2], [employee2, completed2], complete2, name="done2")

        # ── final reward: both sub-tasks of the same case completed ──
        def is_same_case_id(e1, e2):
            return e1["case_id"] == e2["case_id"]

        agency.add_event(
            [completed1, completed2],
            [],
            behavior=lambda x, y: [],
            name="doneFinal",
            reward_function=lambda x, y: 1,
            guard=is_same_case_id,
        )

        return agency

    # ── single training run ───────────────────────────────────────────────

    def _train(self, causal_rl: bool, seed: int):
        import os
        import random

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = self._create_parallel_disjoint_env(causal_rl=causal_rl)
        args = self._make_args(causal_rl)
        args["agent_seed"] = int(seed)

        start = time.time()
        try:
            env.training_run(length=self.config.max_episode_timesteps, args_dict=args)
            history = env.training_history if hasattr(env, "training_history") else {}
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            history = {}
        elapsed = time.time() - start
        return history, elapsed

    # ── analysis helpers ──────────────────────────────────────────────────

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
        for label, key in [("PPO-Clip", "ppo_clip"), ("PPO+Causal", "ppo_causal")]:
            mean_ret, std_ret = self._extract_metric(self.results[key], "mean_returns")
            if mean_ret is None:
                logger.info(f"  {label}: no data")
                continue
            logger.info(
                f"  {label}: final mean return = {mean_ret[-1]:.2f}, "
                f"best = {np.max(mean_ret):.2f}"
            )

    # ── save results to JSON ──────────────────────────────────────────────

    def _save_results(self):
        out = {
            "config": vars(self.config),
            "timestamps": self.timestamps,
            "results": [self.results],
            "timestamp": datetime.now().isoformat(),
        }

        def _make_serializable(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            try:
                import torch as _torch
            except Exception:
                _torch = None
            if _torch is not None and isinstance(o, _torch.Tensor):
                return o.detach().cpu().tolist()
            if isinstance(o, datetime):
                return o.isoformat()
            if isinstance(o, dict):
                return {_make_serializable(k): _make_serializable(v) for k, v in o.items()}
            if isinstance(o, (list, tuple, set)):
                return [_make_serializable(x) for x in o]
            try:
                json.dumps(o)
                return o
            except TypeError:
                return str(o)

        serializable_out = _make_serializable(out)
        out_file = (
            Path(_make_serializable(self.config.output_dir))
            / "parallel_disjoint_compare.json"
        )
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(serializable_out, f, indent=2)
        logger.info(f"Saved results to {out_file}")

    # ── generate plots ────────────────────────────────────────────────────

    def _generate_plots(self):
        mean_c, std_c = self._extract_metric(self.results["ppo_causal"], "mean_returns")
        mean_p, std_p = self._extract_metric(self.results["ppo_clip"], "mean_returns")

        plt.figure(figsize=(10, 6))
        colors = {"ppo_clip": "#1f77b4", "ppo_causal": "#ff7f0e"}

        # Per-seed faint lines
        for label_key, display_name in [
            ("ppo_clip", "PPO-Clip"),
            ("ppo_causal", "PPO+Causal"),
        ]:
            runs = self.results.get(label_key, [])
            for run in runs:
                if isinstance(run, dict) and "mean_returns" in run:
                    arr = np.asarray(run["mean_returns"])
                    epochs = np.arange(1, len(arr) + 1)
                    plt.plot(
                        epochs, arr, color=colors[label_key], alpha=0.18, linewidth=1
                    )

        # Mean ± std bands
        if mean_p is not None:
            epochs = np.arange(1, len(mean_p) + 1)
            plt.plot(
                epochs,
                mean_p,
                label="PPO-Clip (mean)",
                color=colors["ppo_clip"],
                linewidth=2,
                marker="o",
            )
            plt.fill_between(
                epochs,
                mean_p - (std_p if std_p is not None else 0),
                mean_p + (std_p if std_p is not None else 0),
                color=colors["ppo_clip"],
                alpha=0.18,
            )

        if mean_c is not None:
            epochs = np.arange(1, len(mean_c) + 1)
            plt.plot(
                epochs,
                mean_c,
                label="PPO+Causal (mean)",
                color=colors["ppo_causal"],
                linewidth=2,
                marker="s",
            )
            plt.fill_between(
                epochs,
                mean_c - (std_c if std_c is not None else 0),
                mean_c + (std_c if std_c is not None else 0),
                color=colors["ppo_causal"],
                alpha=0.18,
            )

        # Best-per-epoch envelope
        combined_means = []
        if mean_p is not None:
            combined_means.append(mean_p)
        if mean_c is not None:
            combined_means.append(mean_c)
        if combined_means:
            stacked = np.stack(combined_means, axis=0)
            best_per_epoch = np.max(stacked, axis=0)
            epochs = np.arange(1, len(best_per_epoch) + 1)
            plt.plot(
                epochs,
                best_per_epoch,
                label="Best per-epoch",
                color="green",
                linestyle="--",
                linewidth=1.5,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Mean Return")
        plt.title("Convergence comparison — Parallel-Disjoint Assignment")
        plt.grid(alpha=0.3)
        plt.legend()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_png = self.config.output_dir / f"parallel_disjoint_compare_{ts}.png"
        out_pdf = self.config.output_dir / f"parallel_disjoint_compare_{ts}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_pdf)
        logger.info(f"Saved plots to {out_png} and {out_pdf}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = ComparisonConfig()
    cmp = ConvergenceComparisonParallelDisjoint(cfg)
    cmp.run_comparison()
    logger.info("Done")

