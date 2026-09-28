# BPM conference version — outline (2026-09-28)

Target: BPM main track, LNCS, about 16 pages (check the CfP for the exact
limit and whether references count). The EJOR v4 manuscript (`main_v4.tex`)
stays the long version; this paper is the BPM-facing application paper built
around ONE mechanism and THREE process decision types.

Working title: *Provenance-aware credit assignment for reinforcement learning
on business processes*

## The claim, stated so it survives the K=1 result

Standard GAE credits every decision with every later reward, from every case
in flight. With many concurrent cases the advantage estimate of a decision on
case A is dominated by rewards earned by cases B, C, ... that it did not
cause. Running GAE along the token-provenance DAG (cgae-cf) credits a decision
only with the rewards its tokens caused. The gain is governed by K, the number
of causally independent reward streams; for a single running case K=1 and the
method is provably null. So the claim is: **provenance-aware credit helps
process decisions under concurrency**, which is the normal operating
condition of a running process. Each decision type is run at N=1 (the K=1
negative control) and N=8 (concurrent).

## Structure and page budget

1. **Introduction** (1.5 p). Prescriptive process management needs a
   simulator and a credit signal. Simulators exist (A-E PNs, BPM 2023). The
   credit signal is the gap. One sentence on the mechanism, one on results.
2. **Background** (2 p). A-E PNs in a paragraph + the new Figure 1 (draw.io,
   `figures/fig_aepn.pdf`). GAE in five lines. The provenance DAG presented
   as the object-centric event log of the simulation (cases and resources as
   objects) — costs nothing, lands with the OCEL crowd.
3. **Method** (2.5 p). Firing record -> provenance DAG -> cgae-cf recursion
   (only the successor in the GAE recursion changes). One proposition with a
   proof sketch: unbiased when each reward has a single causal origin,
   variance shrinking with K. Full proofs cited to the long version. Cost
   ~1 ms/episode.
4. **Experiments** (6 p), organised by decision type, not by estimator:
   - **Resource assignment** — multi-site skills-based routing
     (`multisite_env.py`, cells in `suite_results_multisite_protocol`,
     already run: PPO 0.40, cgae-cf 0.96, mc-q 0.20; 20 seeds).
   - **Next-activity selection** — `bpm_envs.make_next_activity(N)`: a case
     carries `risk`; the policy chooses `approve` (fast, pays only for
     low-risk) or `investigate` (slow, always pays). N=1 and N=8.
   - **Rework / quality gate** — `bpm_envs.make_rework(N)`: the policy
     chooses `ship` (fast; a high-risk case comes BACK as rework after a
     fix delay, paying 0) or `check` (slower, always completes). The cost of a
     bad decision arrives later through the loop. N=1 and N=8.
   - Arms: PPO, cgae-cf, mc-q (the one ablation that isolates the causal
     restriction). 40 epochs x 8 episodes, 20 CRN seeds, greedy eval on 20
     episodes, normalized (policy - random)/(heuristic - random). Same
     protocol as the journal cells (`run_bpm.py`).
   - One table (3 decision types x {N=1, N=8} x 3 arms) and one figure:
     paired gain over PPO against K, six points, headroom drawn as ceiling.
5. **Related work** (1 p). RL in BPM (resource allocation, prescriptive
   monitoring, next-action recommendation) and credit assignment (HCA,
   RUDDER, COMA). Current Section 2 cut by two thirds.
6. **Conclusion** (0.5 p). The honest boundary: null at K=1; the single-origin
   assumption is what an event-log-derived simulator would have to respect.

Dropped relative to v4: the estimator ladder (cgae, cap, dag, cfgae, ccf), the
bias bound, the cosine law, the falsified-mechanisms appendix, the
boundary-condition appendix, N=2/N=4/hard-N=4.

## Environment smoke (2026-09-28, `python bpm_envs.py`, horizon 20, 20 eps)

| env | N | random | heuristic | headroom (x sigma_rnd) | factoring |ccf-mcq| |
|---|---|---|---|---|---|
| next_activity | 1 | 5.85 | 11.25 | 5.40 (4.7x) | 0.138 |
| next_activity | 4 | 23.60 | 44.30 | 20.70 (13.0x) | 0.711 |
| next_activity | 8 | 45.80 | 89.55 | 43.75 (14.2x) | 0.847 |
| rework | 1 | 7.75 | 11.90 | 4.15 (3.4x) | 0.069 |
| rework | 4 | 30.00 | 47.50 | 17.50 (6.4x) | 0.734 |
| rework | 8 | 60.90 | 95.05 | 34.15 (9.3x) | 0.863 |

Factoring goes from ~0 at N=1 to ~0.85 at N=8 in both: K tracks N.

## Compute plan

| cells | est. per cell | total (4 workers) |
|---|---|---|
| next_activity N=8: 3 arms x 20 seeds = 60 | ~3 h | ~45 h |
| rework N=8: 60 | ~3 h | ~45 h |
| next_activity N=1: 60 | ~15 min | ~4 h |
| rework N=1: 60 | ~15 min | ~4 h |

Run after the N=8 mc_q fill finishes (it uses the same 4 workers):

    python run_bpm.py env=next_activity N=1 4
    python run_bpm.py env=rework N=1 4
    python run_bpm.py env=next_activity N=8 4
    python run_bpm.py env=rework N=8 4

## Risks

- N=8 could come out null if case completions interfere through the shared
  resource strongly enough to break single-origin. Multi-site has the same
  structure and worked. Design the table so a null is reportable.
- Overlap with EJOR: shared method section and the multi-site result. Plan
  it: journal = superset; conference = the two new decision types.
- Double-submission rules: BPM requires unpublished work; an arXiv preprint of
  the journal version is fine but should be cited.
