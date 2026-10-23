# Changelog

All notable changes to gympn are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Component agents (`gympn.component_agents`, `--component_agents true`): one
  PPO agent per net component, with its own networks, deciding in component
  turns and credited with its own component's return. With the marking
  certificate, the agent of a coupled component hands over to one agent per
  part once the coupling can no longer fire, and is credited with their
  returns. Component turns now also work with component postponement.
- Marking-based certificate (`--marking_certificate true`): the net partition
  is recomputed whenever the set of marked places changes, so transitions that
  can no longer fire stop coupling components.

- `--critic_warmup k`: for the first k epochs only the critic learns (the
  policy's learning rate is held at 0), so a policy loaded with
  `--load_policy_network` is not pushed around by a fresh critic's advantages.

### Changed

- `--use_gpu true` (`run_bpm gpu=1`) now runs the PPO updates on a CUDA GPU:
  the policy and value nets, their optimizer state and the batches move to the
  GPU for each update and back to the CPU for rollouts. It used to have no
  effect. Same results up to float rounding (GPU runs are not bit-reproducible).

- The AEPN encoder and observation building are faster: a training cell runs in
  about half the time with about half the peak memory. Observations are
  unchanged bit for bit; the encoder computes the same function in a cheaper
  order (equal up to float rounding).

- Flat observations (`flat_obs`) are built directly from the marking
  (`GymProblem._direct_flat_observation`) instead of through the expanded net
  and a HeteroData graph: an environment step is about 5x faster (bpic15:
  6.5 -> 1.3 ms), a rollout with a policy about 1.65x. The observation and
  `pn_actions` are unchanged bit for bit (`tests/test_direct_obs.py`).
  Other configurations, and each episode's first observation, still take the
  HeteroData path; `GymProblem.DIRECT_FLAT_OBS = False` forces it everywhere.

- The SMDP discount rate `beta` now defaults to 0.05 (was 0.0, no discount):
  a reward `tau` time units after a decision is weighted `exp(-0.05 tau)`, so
  credit spans about 20 time units. It applies only with `--smdp_discount` or
  `--nfgae`.

- Every action is now scored from the whole visible net by default.
  `GymProblem.global_view` defaults to `'edges'`, which adds the reverse of
  every edge type to the observation. `AEPNStack` adds per-layer global
  attention and count-aware aggregation (`global_attn=True`,
  `count_aggr=True` by default). The attention is restricted to the node's
  net component (`net_partition`, carried as a per-node `comp` attribute of
  the observation), so action logits stay local to their component and
  NF-GAE's Theorem 1 still applies. `HeteroCritic` turns the restriction
  off (`component_mask=False`), so the value network reads the whole net;
  NF-GAE does not need a local critic. Before this, an
  action's logit depended only on what flows into it within `num_layers`
  hops. `--global_view false` restores the forward-only graph, and with it
  the encoder's attention and count terms default to off.
- `GymProblem.testing_run` with a trained policy uses the view the policy
  was trained on (`GymProblem.match_policy_view`,
  `networks.policy_global_view`), so policies saved by 0.1.0 still run.

### Fixed

- GPU updates no longer grow host memory over a run: the CUDA cache is
  released after each update (on Windows it spilled into host memory, 7 ->
  12 GB over a bpic15 run).
- The logged policy KL (and the `--policy_kld_limit` brake) were NaN for a
  policy with zero-probability actions, e.g. one loaded from behaviour
  cloning; such actions now contribute 0.
- `--load_policy_network` works with torch >= 2.6 (whole-module checkpoints
  need `weights_only=False`) and accepts an absolute `--policy_network` path.

## 0.1.0 - 2026-10-09

First release on PyPI.

### Added

- `GymProblem`: define a problem as an Action-Evolution Petri Net on top of
  `simpn`, with actions, events, guards, reward functions and tagged places.
- `AEPN_Env`: the Gymnasium environment built from a `GymProblem`, with
  heterogeneous-graph observations for PyTorch Geometric.
- `PPOAgent` and `PGAgent`, with PPO-Clip, PPO-Penalty and policy gradient,
  and optional net-factored credit assignment (NF-GAE) for nets with
  independent parts.
- `HeteroActor` and `HeteroCritic` graph policy and value networks.
- Postponement: the agent may wait, globally or per independent component.
- `GymSolver`, `HeuristicSolver` and `RandomSolver` for testing a trained
  policy, a hand-written heuristic or a random policy, and a `Visualisation`
  of a run.
- `seed_everything` and `seed_network_init` for reproducible runs.
- Optional extras: `tensorboard` (training curves), `wandb` (Weights & Biases
  logging) and `viz` (plotting the graph observations). The library trains and
  tests without any of them.
- Documentation site at <https://bpogroup.github.io/gympn> with an API
  reference generated from the docstrings.

### Notes

- Requires `simpn` 1.3 to 1.6. Later simpn releases depend on PyQt6, which
  cannot be imported on a headless Linux machine, and 1.8+ cannot deep-copy a
  problem.
