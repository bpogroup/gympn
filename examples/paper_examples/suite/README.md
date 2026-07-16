# Causal-RL stability suite

Tests, across the 8 `paper_examples` environments (a 2×4 grid of
{sequence, parallel, loop, exclusive-choice} × {joint, disjoint}):

- **H1 (primary):** causal TD(0) is **more stable at convergence** than plain PPO —
  it holds the optimum with less post-convergence drift.
- **H2:** causal is ≥ plain PPO in final performance (smaller gap-to-optimal).

## Files
- `config.py` — `SuiteConfig` (the full grid) + `smoke_config` / `smoke8_config`.
- `envs.py` — the 8 environment factories + shared `perfect_heuristic`.
- `run_suite.py` — trains every cell; **resumable** (one JSON per cell, skips existing).
- `analyze.py` — normalized tables + paired causal-vs-clip tests.
- `plot.py` — per-env greedy curves + aggregate bars.

## Running
```bash
# from the repo root (gympn/)
python examples/paper_examples/suite/run_suite.py smoke    # 1 env, ~2 min  (pipeline check)
python examples/paper_examples/suite/run_suite.py smoke8   # all 8 envs, tiny budget, ~7 min
python examples/paper_examples/suite/run_suite.py          # FULL: 8 envs × 2 methods × 10 seeds
```
The full run is **160 cells ≈ 50h**. It is resumable: re-run the same command to
continue (completed cells are skipped). Safe to Ctrl-C.

## Analyzing (works on partial results too)
```bash
python examples/paper_examples/suite/analyze.py suite_results
python examples/paper_examples/suite/plot.py    suite_results
```

## Metrics
Each cell trains with `test_in_train=True`, giving a **greedy (argmax) eval curve**
every `test_freq` epochs. Per cell we record:
- `greedy_final`, `greedy_best` — performance (deterministic policy).
- `greedy_drift = best − final` — **the H1 stability signal** (lower = more stable).
- `entropy_final` — how decisively the policy committed.

Normalization (per env, against Random/Heuristic baselines):
```
normalized = (policy − random) / (heuristic − random)     # 0 = random, 1 = optimal
```

## Caveats baked into the analysis
- **`h_exclusive_choice_disjoint` has no headroom** (random ≈ optimum); its
  normalized score is undefined and reported as `n/a` — it cannot separate methods.
- Headroom varies a lot across envs (e.g. `e_loop_joint`: random 2.2 → optimum 9).
  Always read the **normalized** columns for cross-env aggregation, raw for sanity.
- Both methods train on the **same** env config (`allow_postpone=True`); baselines
  use the canonical no-postpone env to anchor the optimum.