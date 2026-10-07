# BPM paper — outline v2 (2026-10-06)

Target: BPM main track, LNCS, about 16 pages (check the CfP for the exact limit and
whether references count). The EJOR paper is dropped; this is now the only paper.
v1 of this outline (provenance-aware credit, cgae-cf vs PPO vs mc-q) is in git
history; it is superseded because its evidence did not survive a strong baseline:
with the AEPN network, plain PPO reaches the heuristic on multi-site (1.039 vs 0.403
with HGT), so the old causal-vs-PPO gap was mostly the network.

Working titles:
- *Learning to control a process portfolio: factoring credit over the process net*
- *Faster reinforcement learning for concurrent business processes by factoring the
  advantage over the process model*

## The claim

An organisation runs several processes side by side and is measured on one KPI. An
RL controller trained on that KPI credits each decision in one process with the
reward noise of every other process, and no critic can remove that noise: it is the
other processes' future randomness (Theorem 3). **NF-GAE** splits the advantage over
the connected components of the A-E PN (processes that share no place), so each
decision is credited only with its own component's rewards.

- Provably unbiased under stated conditions (Theorems 1–2), never higher variance
  than PPO with any baseline (Theorem 3), and **identical to PPO when the net has one
  component** (Theorem 2a; verified bit-for-bit).
- The decomposition comes from the process model. Nobody writes reward-attribution
  rules, and where processes are coupled (a shared team) the partition merges them,
  so it is decomposed exactly where that is valid.
- Empirically: faster learning, and a better final policy where the environment is
  not saturated; the gain grows with the number of independent components K.
- Against separate per-process learners (the obvious alternative when processes are
  independent): most of the gain over joint PPO is per-unit credit, which per-unit
  learners also get; NF-GAE adds a shared network on top (better early learning,
  +0.07 whole-run) at the same compute, thanks to component turns (E11, E12).

Second, methodological claim: **RL comparisons on Petri nets need a strong policy
network.** With a per-type HGT, PPO fails on multi-site; with a relational network
that shares weights across relations (R-GCN/RGAT-style basis decomposition with HGT's
design choices; not novel, cited as such), the same PPO reaches the heuristic.

## Contributions

1. NF-GAE: the estimator, the structural soundness conditions for A-E PNs (component
   partition, local actor, work-conserving or component-scoped postpone), and the
   variance result. Theory: `NFGAE_THEORY.md`.
2. An organisation-level benchmark: an insurer with three processes covering the three
   decision types (claims = next-activity selection, underwriting = resource
   assignment, complaints = rework / quality gate), with a coupling variant.
   `insurer_env.py`.
3. Evidence that the network is the first-order factor for PPO on A-E PNs, and that
   NF-GAE adds a consistent gain on top of the strong network.

## Structure and page budget

1. **Introduction** (1.5 p). Process portfolio, organisation-level KPI, why the KPI
   mixes independent processes' noise into every decision. Results in two sentences.
2. **Background** (2 p). A-E PNs + Figure 1 (the insurer net of one region, components
   coloured). PPO/GAE in the SMDP form in five lines.
3. **Method** (3 p). Net partition; per-component SMDP-GAE on each component's own
   decision clock; component critic. Theorems 1–3 as statements with proof sketches,
   full proofs in an appendix or online supplement. Conditions C1–C3' in plain words.
   One paragraph on the policy network (relational attention with basis-decomposed
   relation weights; flat graph observations) as the experimental setup, cited.
   One paragraph on component turns: simultaneous decisions are offered one component
   at a time, so each decision observes and encodes only its own component (valid by
   locality and work conservation; keeps compute linear in K).
4. **Experiments** (6 p). Section by question, not by environment (list below).
5. **Related work** (1 p). RL for BPM (resource allocation, prescriptive monitoring,
   next-best-action, A-E PN/GymPN work). Credit assignment: factored policy gradients
   (Spooner et al. 2021), action-dependent factorized baselines (Wu et al. 2018),
   VDN, COMA / difference rewards. Relational GNNs: R-GCN, RGAT, HGT.
6. **Discussion and conclusion** (1 p). The boundary: within one process whose cases
   share a resource pool, K=1 and NF-GAE equals PPO. Its target is multi-process or
   multi-team control. Event-log-derived nets inherit the same condition.

## Experiments — complete list

Protocol for every cell unless stated: AEPN network (8 bases), flat observations,
no postpone, 40 epochs, greedy evaluation on 20 episodes with common random numbers
(eval_seed 555000), normalised (policy − random) / (heuristic − random), seeds paired
across arms. Runner: `run_bpm.py` (insurer) and `run_multisite_protocol.py`.
Insurer evaluates every 3 epochs, multi-site every 2.

| ID | Question | Setup | Arms | Seeds | Status | Compute |
|---|---|---|---|---|---|---|
| E1 | Is the network the first-order factor? | multi-site, N=4 | PPO-HGT vs PPO-AEPN | 20 | DONE (HGT 2026-09-20, AEPN 2026-10-05) | — |
| E1b | Is E1 code drift? | multi-site, current code | PPO-HGT | 4 | DONE: bit-identical to Sept (all curves) | — |
| E1c | Does the network effect hold beyond multi-site? | insurer r=1 | PPO-HGT (vs E3's PPO-AEPN) | 10 | PLANNED | ~2 h |
| E2 | NF-GAE on a symmetric structure | multi-site, K=4 | PPO vs NF-GAE | 20 | DONE | — |
| E3 | Main result: the insurer | insurer r=1 (K=3), r=2 (K=6) | PPO vs NF-GAE | 10 → 20 | DONE at 10 seeds (2026-10-06); top-up to 20 PLANNED | ~5 h top-up |
| E3b | Larger organisation | insurer r=4 (K=12) | PPO vs NF-GAE | 10 | PLANNED (if E3 holds) | ~4–6 h |
| E4 | K=1 controls: how fast is each process learned alone? | claims / underwriting / complaints alone, r=1 | PPO (NF-GAE ≡ PPO here) | 10 | DONE | — |
| E5 | Coupling: does the gain shrink when processes share a team? | insurer_shared r=1 (K=2) | PPO vs NF-GAE | 10 | DONE | — |
| E6 | Which process gains? | per-process evaluation returns of E3 policies | PPO vs NF-GAE | E3's | PLANNED (needs per-component eval logging) | small code + rerun or post-hoc |
| E7 | Is PPO just under-budgeted or mis-tuned? | insurer r=2 (where PPO fails): PPO with 2× episodes per epoch; PPO with policy lr ×0.5 / ×2 | PPO variants vs NF-GAE | 5 each | DONE | — |
| E7b | Fairness: NF-GAE with PPO's best lr (6e-4) | insurer r=2 | NF-GAE | 5 | DONE | — |
| E8 | Where does the gain come from? | insurer r=1: factored advantage + global critic, vs full NF-GAE | NF-GAE ablation | 10 | PLANNED (needs a flag) | ~1.5 h |
| E9 | Cost | CPU time per 40-epoch run: NF-GAE (full graph) vs per-unit learners, 1 thread, comparable load | — | from E3/E4 | DONE | — |
| E10 | Mechanism plot (only if E3 is ambiguous) | insurer claims + K background lines with exogenous revenue | PPO vs NF-GAE | 6 | OPTIONAL | ~3 h |
| E11 | Is NF-GAE more than separate per-process learners? | insurer r=1 / r=2: organisation built from E4's per-process agents (independent units: expected total = sum of unit returns; anchors verified additive) | per-unit learners vs NF-GAE | 10 / 5 orgs | DONE (post hoc from E4) | — |
| E12 | Local encoding (component turns): same learning, less compute? | insurer r=2, `local=1` | NF-GAE local vs full graph | 5 | DONE | — |

Already verified and only cited (no new runs): partition tests
(`test_partition_*`), K=1 identity (bit-for-bit advantages and end-to-end finals,
`test_nfgae.py`), actor locality for all encoders, flat-vs-heterogeneous equivalence
(`test_type_embed.py`).

Dropped from v1: cgae-cf, mc-q and the provenance family as method arms (Prop 4 in the
theory note shows provenance credit is unsound under pre-emption; keep as one
related-work sentence); next_activity / rework as standalone N-copy environments
(the insurer contains both decision types); the single-origin proposition.

### Results so far

- **E1:** PPO final greedy 0.403 (HGT, 6/20 collapsed) vs 1.039 (AEPN, 0/20).
- **E2:** finals tie (NF-GAE 1.046 vs PPO 1.039, p=.22, saturated); median epochs to
  0.9 is 4 vs 8, faster on 20/20 seeds (Wilcoxon p=8e-5); mean greedy over epochs
  2–10: 1.007 vs 0.760 (20/20, p=3.5e-9).
- **E3 r=1 (K=3), 10 seeds:** final 1.137 vs 0.961 (+0.176, CI [+.063, +.290], 10/10,
  Wilcoxon p=.002); whole-run mean 1.029 vs 0.807 (10/10, t p=3e-5); epochs to 0.9
  median 8 vs 16.
- **E3 r=2 (K=6), 10 seeds:** final 1.081 vs 0.713 (+0.368, CI [+.279, +.457], 10/10,
  t p=6e-6); whole-run mean 1.020 vs 0.611 (10/10, t p=2e-7); epochs 3–12 0.921 vs
  0.462. PPO never reaches 0.9 within 40 epochs on 9/10 seeds; NF-GAE does by epoch 9
  (median) on 10/10.
- **E5 shared clerks (K=2), 10 seeds:** final 1.023 vs 0.989 (+0.034, p=.15, 7/10);
  whole-run mean 0.925 vs 0.855 (+0.069, p=.038).
- **E4 processes alone (K=1, PPO), 10 seeds each:** finals 1.083 / 1.049 / 1.067
  (claims / underwriting / complaints), epochs to 0.9 median 9 / 8 / 3. Inside the
  insurer PPO is slower and lower than on every process alone; NF-GAE restores the
  alone-level speed.
- **E1b:** HGT PPO on the current code reproduces the September cells bit for bit
  (greedy, sampled and entropy curves, 4/4 seeds): the HGT failure is the network.
- **E7 (insurer r=2, seeds 0-4, final):** PPO default 0.719, lr 1.5e-4 0.671, lr 6e-4
  0.782, 2× data 0.856; NF-GAE (default settings, half PPO-2×'s data) 1.087, ahead of
  every variant on 5/5 seeds (vs 2× data: +0.232, p<.001; whole-run +0.314). Epochs to
  0.8: NF-GAE 3–9, PPO 2× data 12–36 or never.
- **E7b (insurer r=2, lr 6e-4, seeds 0-4):** NF-GAE 1.093 (1.087 at the default lr)
  vs PPO 0.782 at the same lr (+0.311, 5/5, p=.003).
- **Gain vs K (final):** K=2 +0.034, K=3 +0.176, K=6 +0.368. Whole-run mean: +0.069,
  +0.222, +0.409.
- **E11 per-unit learners (organisation of separately trained per-process PPO agents):**
  r=1: final 1.058, whole-run 0.960, epochs 3–12 0.783 vs NF-GAE 1.137 / 1.029 / 0.877
  (final +0.079, Welch p=.045; whole-run +0.069, p=.009). r=2 (5 orgs): 1.048 / 0.950 /
  0.774 vs NF-GAE 1.081 / 1.020 / 0.921 (final +0.033, n.s.; whole-run +0.069, p=.02).
  Epochs to 0.9 the same (~9). Per-unit learners capture most of NF-GAE's gain over
  joint PPO; NF-GAE's extra is early learning, consistent with the shared network.
- **E9 cost (full-graph NF-GAE):** per-unit learners 13.9 CPU-min (r=1, 3 runs) and
  27.9 (r=2, 6 runs) vs NF-GAE 23.8 and 61.2 min: 1.7× and 2.2× more, growing ~K²
  because every decision encoded the whole organisation.
- **E12 local encoding (component turns):** 2-epoch cells r=1 25.5 → 15.3 s (1.67×),
  r=2 69.8 → 29.2 s (2.39×), which cancels the E9 overhead (compute now on par with
  per-unit learners, linear in K). Learning unchanged, r=2 seeds 0–4: whole-run 1.023
  vs 1.017 full graph (p=.73), epochs 3–12 0.949 vs 0.914, final 1.053 vs 1.087
  (p=.14, single evaluation point). Tests: identical at K=1, active component's
  logits equal the full-graph logits.

## Figures and tables

- **Fig. 1** The insurer net, one region, components coloured; the shared-clerk
  variant as an inset (claims and complaints merge).
- **Fig. 2** Learning curves, insurer r=1: PPO vs NF-GAE, with each process alone
  (E3, E4).
- **Fig. 3** Gain vs K: whole-run mean gain and final gain of NF-GAE over PPO at K=1
  (E4), 2 (E5), 3 (E3 r=1), 6 (E3 r=2), 12 (E3b); multi-site (E2) as a separate
  symmetric point.
- **Table 1** All conditions: final, best, whole-run mean, epochs to 0.9, collapsed
  seeds, paired tests.
- **Table 2** Network baseline: PPO-HGT vs PPO-AEPN (E1, E1b, E1c).
- **Table 3** NF-GAE vs per-unit learners vs joint PPO: learning (early, whole-run,
  final) and CPU time, full graph vs local encoding (E9, E11, E12).
- Optional: per-process gains (E6), cost (E9) in the text.

## Statistics

Seeds paired across arms (same seed, same evaluation scenarios). Report means with
95% CIs; paired t-test and Wilcoxon signed-rank; seeds-won counts. Holm correction
over the headline comparisons (E3 r=1 and r=2 final + whole-run mean, E5, E2).
Learning speed: epochs to 0.9 and mean greedy return over the first quarter of
training, both preregistered here before the full E3 results.

## Compute plan and order

1. ~~E3 r=1, E5, E3 r=2, E4~~ done 2026-10-06 05:01.
2. ~~E1b~~ done (bit-identical reproduction).
3. E6 logging code, then E3 top-up to 20 seeds with per-process logging (~3 h).
4. ~~E7, E7b~~ done; E8 (~1.5 h), E1c (~2 h). ~~E9, E11, E12~~ done.
5. E3b r=4 (~4–6 h) if E3 r=2 shows the gain growing.
6. E10 only if needed.

Total remaining: about 15–20 h of machine time.

## Decision points and risks

- **E1b does not reproduce PPO-HGT ≈ 0.40** → the network claim is code drift, not
  architecture; drop contribution 3 and Table 2.
- **E3 r=2 gap not larger than r=1** → no K-scaling claim; report the gain as
  consistent, not growing.
- **E7: PPO with 2× data catches up** → frame NF-GAE as sample efficiency (half the
  data for the same policy), which is still the claim.
- **E4 shows each process alone is learned no faster than inside the insurer** → the
  "noise from other processes" story weakens; the gain would then need E8 to explain it.
- **E11 shows per-unit learners nearly match NF-GAE** → NF-GAE's distinct value is the
  shared network (early learning) plus automatic, provably sound decomposition and
  one deployed policy; state that, and don't claim a large gain over per-unit learners.
- **Boundary to state up front:** one process with a shared resource pool is K=1
  (verified on BPI2012: one well-mixed resource pool, median 4 resources per case).
