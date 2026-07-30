"""Reproducible seeding for gympn.

A single entry point, :func:`seed_everything`, seeds every stochastic source a
gympn run touches -- Python's ``random``, NumPy, and PyTorch (CPU and CUDA), plus
the cuDNN / deterministic-algorithm switches -- so that a run is reproducible
given the same seed, code, and hardware.

Environment stochasticity (task arrivals, service delays) is produced by the
Petri-net behaviour functions through the global ``random`` / ``numpy`` modules,
so seeding those globals makes the environment reproducible as well, provided
execution is single-threaded within a run (the default for a training worker).

Usage
-----
>>> import gympn
>>> gympn.seed_everything(0)
0

or seed a specific problem / environment / run directly::

    pn.set_seed(0)                                   # GymProblem
    env.set_seed(0)                                  # AEPN_Env
    pn.training_run(length=20, args_dict={"agent_seed": 0, ...})  # seeds internally
    pn.testing_run(solver, length=20, seed=0)        # reproducible evaluation
"""
from __future__ import annotations

import os
import random as _random

import numpy as _np


def seed_everything(seed: int, *, deterministic: bool = True,
                    set_hash_seed: bool = True) -> int:
    """Seed all stochastic sources used by gympn.

    Parameters
    ----------
    seed : int
        Seed applied to Python ``random``, NumPy, and PyTorch (CPU + CUDA).
    deterministic : bool, default True
        If True, also request deterministic PyTorch kernels: cuDNN determinism
        and ``torch.use_deterministic_algorithms(True, warn_only=True)``. This
        can slow training slightly and warns (rather than errors) on the rare
        operation lacking a deterministic implementation.
    set_hash_seed : bool, default True
        If True, set ``PYTHONHASHSEED``. This only affects *child* processes
        spawned afterwards; to fix hash randomisation of the current process it
        must be exported in the environment before the interpreter starts (the
        gympn runners do this at import time).

    Returns
    -------
    int
        The seed, for convenience and logging.
    """
    seed = int(seed)
    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    _np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # required for deterministic CUDA matmul; harmless on CPU-only runs
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # older torch, or an op without a deterministic path: best effort
            pass
    return seed