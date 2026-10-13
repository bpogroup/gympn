"""Shared hyperparameters and helpers for the experiment runners
(run_bpm.py, run_multisite_protocol.py).

Every arm of every experiment uses the same hyperparameters (no per-arm
tuning); the runners set the protocol (epochs, episodes, evaluation).
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Hyper:
    batch_size: int = 32
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    policy_updates: int = 3
    value_updates: int = 4
    ent_bonus: float = 0.01          # linearly annealed to 0 over training
    ppo_eps: float = 0.2
    policy_kld_limit: Optional[float] = 0.15
    gam: float = 0.99                # unused on the SMDP paths (ppo, nfgae)
    lam: float = 0.95
    beta: float = 0.5                # SMDP discount: e^{-beta * tau}


def set_seed(seed: int):
    """Reproducible seeding via the single library entry point."""
    from gympn import seed_everything
    seed_everything(seed)


def _physical_core_count() -> int:
    """Physical (not logical) CPU cores: psutil's count, else logical // 2."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 1) // 2)


def threads_per_worker(num_workers: int) -> int:
    """Fair share of physical cores for each worker's torch intra-op pool."""
    return max(1, _physical_core_count() // max(1, num_workers))


def extract_metrics(history: dict, epochs: int, test_freq: int) -> dict:
    """The learning curves of one training run (env.training_history)."""
    def arr(key):
        return list(map(float, history[key])) if key in history else []

    sampled = arr("mean_returns")
    greedy = arr("test_mean_returns")[:epochs // test_freq]   # drop the trailing unused slot
    ent = arr("policy_ent")
    return {
        "sampled_curve": sampled,
        "sampled_best": max(sampled) if sampled else None,
        "sampled_final": sampled[-1] if sampled else None,
        "greedy_curve": greedy,
        "greedy_epochs": [test_freq * (i + 1) for i in range(len(greedy))],
        "greedy_best": max(greedy) if greedy else None,
        "greedy_final": greedy[-1] if greedy else None,
        # how far the greedy policy fell from its peak
        "greedy_drift": (max(greedy) - greedy[-1]) if greedy else None,
        "entropy_curve": ent,
        "entropy_final": ent[-1] if ent else None,
    }
