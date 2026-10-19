# Reproducing the BPM paper experiments

Everything here runs from this directory with the repository's virtual environment
(`../../../.venv`). Hyperparameters are shared by all arms (`envs/common.py`); no arm is
tuned separately.

## Layout

| Folder | Contents |
|---|---|
| `envs/` | A-E PN environment builders and heuristics: `insurer_env.py` (claims, underwriting, complaints; `insurer`, `insurer_shared`, `insurer_slow`, and the three single-process variants), `bpm_envs.py` (building blocks and the BPM nets), `multisite_env.py` (multi-site routing), `common.py` (shared hyperparameters and metrics). |
| `scripts/` | Runners and analyses: `run_bpm.py`, `run_multisite_protocol.py`, `postpone_behaviour.py`, `bpi2012_parse.py`. |
| `results/` | One folder per experiment (`<name>/cells/*.json` holds one evaluation curve per seed, `summary.json` the aggregate, `baselines*.json` the random and heuristic anchors). `results/ablations/` holds the PPO-variant runs. `train/` subfolders hold the trained policies and are not versioned. |
| `paper/` | `bpm/` is a local clone of the Overleaf manuscript (not tracked in this repository), `bpm_aux/` the figure script and local notes, `NFGAE_THEORY.md` the theory note, `BPM_OUTLINE.md` the experiment log. |

Runners are resumable: a cell that already exists is skipped, so adding seeds means
re-running the same command with a larger `seeds=` value. The trailing integer is the
number of parallel workers; `threads=1` gives each worker one torch thread.

Observation view. Since 2026-10-17 the library scores every action from the whole visible
net by default (`GymProblem.global_view='edges'`: reverse edges, plus global attention and
count aggregation in the AEPN encoder). Every results directory listed below was produced
with the earlier forward-only graph: add `view=local` to `run_bpm.py` and
`run_multisite_protocol.py` commands to reproduce them. Without it the runners use the new
view and write to `*_fv` directories, never into the old ones. Since 2026-10-19 the
critic's attention reads across components (the actor's stays inside its component);
`*_fv_cm` directories hold runs with the earlier critic, masked like the actor, and
`critic_mask=1` reproduces them.

## Insurer (RQ1): PPO versus NF-GAE

```
python scripts/run_bpm.py env=insurer        N=1 methods=ppo,nfgae seeds=20 flat=1 threads=1 8
python scripts/run_bpm.py env=insurer        N=2 methods=ppo,nfgae seeds=20 flat=1 threads=1 10
python scripts/run_bpm.py env=insurer_shared N=1 methods=ppo,nfgae seeds=10 flat=1 threads=1 8
```

Per-process controls (one process alone, PPO only; the per-process learner baseline):

```
for p in claims underwriting complaints; do
  python scripts/run_bpm.py env=insurer_$p N=1 methods=ppo seeds=10 flat=1 threads=1 2
done
```

Results: `results/bpm_insurer_n1_ep40_nopp_aepn_flat`, `..._n2_...`,
`results/bpm_insurer_shared_n1_ep40_nopp_aepn_flat`,
`results/bpm_insurer_{claims,underwriting,complaints}_n1_ep40_nopp_aepn_flat`.

## Postponement (RQ2): insurer_slow, 1 and 2 regions

Arms: no postponement (PPO, NF-GAE), global postponement (PPO only; NF-GAE with global
postponement is unsound and refused), component-scoped postponement (PPO, NF-GAE).

```
for N in 1 2; do
  C="env=insurer_slow N=$N seeds=20 flat=1 threads=1"
  python scripts/run_bpm.py $C postpone=0         methods=ppo,nfgae 6
  python scripts/run_bpm.py $C postpone=1         methods=ppo       4
  python scripts/run_bpm.py $C postpone=component methods=ppo,nfgae 6
done
```

Results: `results/bpm_insurer_slow_n{1,2}_ep40_nopp_aepn_flat` (none),
`..._ep40_aepn_flat` (global), `..._ep40_ppc_aepn_flat` (component).

Behaviour replay of the trained policies (where do they wait?), which needs the
`train/best_policy.pth` files of the six folders above:

```
python scripts/postpone_behaviour.py regions=1 workers=5
python scripts/postpone_behaviour.py regions=2 workers=5
```

Output: `results/postpone_behaviour_results.json` and `..._n2.json`.

## Multi-site routing

```
python scripts/run_multisite_protocol.py 8 seeds=20 methods=ppo,nfgae net=aepn flat=1 threads=1
python scripts/run_multisite_protocol.py 8 seeds=20 methods=ppo       net=hgt         threads=1
```

Results: `results/multisite_protocol_aepn_flat` (AEPN encoder, both arms) and
`results/multisite_protocol` (HGT encoder, PPO only; the network-artifact check).

## PPO variants on the 2-region insurer (`results/ablations/`)

```
C="env=insurer N=2 methods=ppo seeds=5 flat=1 threads=1"
python scripts/run_bpm.py $C episodes=16   5    # ..._eps16_flat
python scripts/run_bpm.py $C plr=0.00015   4    # ..._plr0.00015_flat
python scripts/run_bpm.py $C plr=0.0006    4    # ..._plr0.0006_flat (also run with methods=nfgae)
python scripts/run_bpm.py env=insurer N=2 methods=nfgae seeds=5 flat=1 threads=1 local=1 5   # ..._flat_local (component turns)
```

These write to `results/`; the archived copies were moved to `results/ablations/` by hand.

## Figures and tables

```
python paper/bpm_aux/make_figures.py
```

reads the cells in `results/` and writes `paper/bpm/figures/*.pdf`; the printed tables are
the source of the numbers in the manuscript. Commit and push inside `paper/bpm` to update
Overleaf. The figure PDFs embed a timestamp, so only commit them when the data changed.

## BPI Challenge 2012 statistics

`scripts/bpi2012_parse.py` flattens the public XES log (expected at
`paper/BPI_Challenge_2012.xes`, 4TU.ResearchData
doi:10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f) into
`results/bpi2012_events.csv.gz`. Neither file is versioned.

## Budget

On an 8-core CPU, one 1-region insurer cell takes about 25 minutes, one 2-region cell
about an hour, and one 2-region insurer_slow cell about two hours.
