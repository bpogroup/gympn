# Journal paper — outline (2026-10-27)

Extends the BPM draft into a journal paper. Decided 2026-10-27: the journal
paper OVERWRITES the conference draft in `paper/bpm/` (Overleaf), keeping the
structure and voice of the original A-E PN paper (2306.02910). No page limit. Follows the same section
skeleton: Introduction, Related Work, Preliminaries, Method, Evaluation, Conclusions
and Future Work; numbered Definitions; formal voice; American spelling.

## Working titles
- *What the Process Model Knows: Independence-Aware Learning and Planning for
  Concurrent Business Processes*
- *The Independence Certificate: Exact Factorization for Reinforcement Learning and
  Simulation-Based Policy Learning in Action-Evolution Petri Nets*
- *Learning to Act from What the Model Certifies: Factored Credit, Local Lookahead and
  Factored Deep Controlled Learning for Business Processes*

## Thesis
An A-E PN certifies which parts of a process system are independent: the connected
components of its place-transition graph, statically or at the current marking. The
BPM draft uses this independence certificate in model-free RL (credit, decision
timing, observation). The journal paper shows that the same certificate also makes
**simulation-based policy learning** exact, local and cheap: rollouts can be
restricted to the deciding component without approximation, paired and merged where
trajectories re-join, and training data can be collected and labelled one component
at a time. On real-data-calibrated processes (BPIC 2015), certificate-factored Deep
Controlled Learning learns a policy that needs no simulator at decision time and beats
both the heuristic and the lookahead planner it is trained from, where model-free RL
stays at the heuristic's level.

## Contributions
1. **The independence certificate and its uses in RL** (from the BPM draft): factored
   credit (NF-GAE), component-scoped postponement, component-scoped observation; the
   marking certificate for couplings that dissolve. Guarantees: unbiasedness, variance
   dominance, identity with PPO at K = 1.
2. **Certificate-restricted lookahead (new):** a rollout planner whose rollouts
   simulate only the deciding component (exact by the certificate), with common random
   numbers through keyed streams, coalescing of re-merged trajectories (exact), and a
   fair redraw of the already-drawn future a policy cannot observe. Sequential Halving
   as in DCL; a paired test for departures from the base.
3. **Certificate-factored Deep Controlled Learning (new):** DCL's approximate policy
   iteration with (a) component-local labels: each label constrains only the
   deciding component's action distribution (the main source of the gain);
   (b) component-factored collection: each component simulated and labelled alone,
   exactly, at about half the collection wall time; (c) soft targets. The number of
   days can differ per component, but allocation is NOT claimed as a gain (equal and
   value-proportional allocations tie).
4. **Observation design for A-E PN policies (new):** one action node per option
   (equivalent bindings merged, FIFO firing, multiplicity as a feature), and a dedicated
   clock node that preserves component locality. Both measured (critic error,
   distillation).
5. **A real-data-calibrated benchmark** (BPIC 2015, five municipalities, shared staff)
   next to the insurer family; code in GymPN.
6. **Empirical evidence:** where model-free RL fails to improve on the heuristic,
   certificate-factored DCL does, by a wide margin (1.29-1.31 vs DCL 1.22), at about
   half the collection wall time of plain DCL; component-local labels alone give 1.28; and an honest map of what does not work (distillation from noisy labels,
   racing, vine advantages, reactive imitation of planner decisions).

Not claimed: novelty of DCL itself, of rollout policy improvement, of soft targets,
or of the relational network; DCL is the baseline we extend.

## Structure

### 1. Introduction
- Organization-level process control; several processes, shared staff, one KPI.
- Two families of learning methods: model-free RL and simulation-based policy learning
  (DCL). Both treat the system as monolithic.
- The A-E PN's independence certificate; what it buys each family.
- "To fill this gap, this paper presents ..."; evaluation summary; structure.

### 2. Related Work
- RL for BPM: resource allocation, prescriptive process monitoring (Metzger et al.),
  A-E PN / GymPN line.
- Structured credit: factored policy gradients, factorised baselines, value
  decomposition, difference rewards, networked MARL.
- Simulation-based policy learning: rollout algorithms (Bertsekas), classification-based
  API (Lagoudakis & Parr; Lazaric et al.; Gabillon et al.), AggreVaTe, expert
  iteration, **DCL (Temizoz et al., EJOR 2025)** and DynaPlex.
- Best-arm identification for rollouts: Sequential Halving, racing.
- Non-work-conserving control (slow server).
- Comparison table: method x {uses model structure, exact factorization, simulator at
  decision time, real-data case}.

### 3. Preliminaries
- A-E PNs, bindings, observation graph (Definitions as in the BPM draft).
- The decision process (SMDP), PPO with SMDP-GAE.
- DCL: approximate policy iteration, Sequential Halving with CRN, classification.

### 4. Method
- 4.1 The independence certificate (static and marking).
- 4.2 Factored credit (NF-GAE).
- 4.3 Component-scoped postponement.
- 4.4 Component-scoped observation; **option nodes and the clock node** (new:
  Definitions of option, multiplicity, FIFO binding; the clock node's edges and why it
  keeps the locality proposition).
- 4.5 **Certificate-restricted lookahead** (new): restricted simulation; keyed random
  streams; coalescing (state fingerprint; exactness argument); fair resampling of the
  unobserved future; Sequential Halving over options.
- 4.6 **Certificate-factored DCL** (new): component-factored collection (exactness:
  the component's marking process is the same alone as inside the net; the actor's
  component logits read only the component); value-based allocation of component-days;
  component-local labels; soft targets; rounds.
- 4.7 Properties: Theorems 1-3 (BPM draft); new Proposition (restricted rollouts and
  coalescing are exact for the component's return); new Proposition (factored
  collection yields the same distribution of labelled decisions per component).
- 4.8 Extending the RL loop / the DCL loop (pseudo-code for both).

### 5. Evaluation
- "This section aims to show that ..."
- 5.1 Benchmarks: insurer family (regions, start-of-shift allocation, slow mismatch);
  **BPIC 2015** (calibration, what is assumed, shared staff, the work-in-progress
  variant).
- 5.2 Setup: arms, protocol, normalization, held-out days, seeds, statistics, compute
  accounting.
- 5.3 Results, model-free RL (BPM draft results, condensed): concurrency and credit;
  couplings that dissolve; observation; waiting.
- 5.4 Results, BPIC 2015:
  - model-free RL (PPO, NF-GAE static / marking) vs heuristic: improvement over PPO,
    not over the heuristic;
  - the value of lookahead: planner variants on held-out days;
  - why reactive imitation fails (diagnostic): departures sparse, timed, partly
    unrecognizable; noisy labels; the copy-confound (option merging); the clock;
  - DCL vs certificate-factored DCL (headline table and learning curve);
  - ablations: allocation vs component-local labels; soft targets; option nodes; clock.
- 5.5 Cost: simulation steps and wall time per method; collection cost of plain vs
  factored DCL.
- 5.6 Generality: certificate-factored DCL on the insurer family.

### 6. Discussion and limitations
- When factoring helps (many components, uneven value) and when not (K = 1).
- Simulated vs real: BPIC 2015 service times are assumed; logs only calibrate arrivals,
  staff, fits and load.
- Negative results worth reporting (racing, vine, reactive distillation).
- Future work: forecast-conditioned policies under demand shift
  (`FORECAST_CONDITIONED_RL.md`), truncated rollouts with component values.

### 7. Conclusions and Future Work

## Figures and tables (new ones marked *)
- Fig. 1 Insurer net; Fig. 2 learning curves; Fig. 3 gain vs K; Fig. 4 waiting (BPM).
- *Fig. 5 BPIC 2015 net (one municipality + the shared-staff pool), draw.io style.
- *Fig. 6 Certificate-factored DCL pipeline (whole-day vs component-day collection).
- *Fig. 7 DCL learning curve (held-out vs labelled days), plain vs factored.
- *Table 6 BPIC 2015 headline: heuristic, rule, PPO, NF-GAE, planners, DCL, factored DCL.
- *Table 7 Planner choice rules (uniform, test filter, Sequential Halving, racing):
  score, rollouts, wall time.
- *Table 8 Ablations of factored DCL (plain / local loss / factored value-based / factored equal).
- *Table 9 Cost per method.

## Evidence map (BPIC 2015 and DCL; RL rows of the BPM draft unchanged)
Normalization: random 0, heuristic 1 on the same days. NOTE: runs with merged options
use anchors from the merged env (held-out random 39.25 instead of 38.73); re-normalize
all rows on one anchor set for the paper.

| Paper item | Experiment | Status | Key numbers (100 held-out days unless stated) |
|---|---|---|---|
| Heuristic / rule | bpic15_heuristic; intake-first t<2 | DONE | 1.000; rule 1.103 (68/18, p=6e-9) |
| Model-free RL | PPO, NF-GAE static / marking, beta .05, 40 ep | PARTIAL (5 seeds, eval days, old representation) | marking .662, static .508, PPO .331 |
| RL from imitation | NF-GAE marking, clock node + merge, plr 3e-5 | DONE (stopped at ep 19) | flat at ~1.00-1.03 (eval days) |
| Planner | tested planner (8+32, margin .5, fair) | DONE | 1.278 (93/5); wip 1.275 (96/1) |
| Planner rules | uniform 8 / SH 32 / racing, eval days | DONE (20 eval days only) | 1.334 / 1.401 / 1.300; SH-uniform +1.15/day p=.006 |
| Distillation (old) | CE / advantage loss / merged, 5 seeds | DONE | ~1.01-1.04 |
| Precise labels | 1000 decisions, 128 rollouts | DONE | departures all genuine; 14% of 'keep' improvable; AUC ~.74 |
| Critic probe | clock: none / token / node | DONE | R^2 .86-.90 / .95-.96 / .95-.96 |
| DCL round 1 | SH 32, 40 days, tau 0 / .25 / .5 | DONE (1 collection, 2-3 seeds) | final tau .25: 1.219 / 1.215 / 1.225 |
| DCL learning curve | 10/20/30/40 days | DONE | 1.066 / 1.139 / 1.150 / 1.176 |
| Factored DCL round 1 | cf 35/24/58/56, compute-matched | DONE (1 collection, 3 seeds) | final 1.322 / 1.312 / 1.287 (mean 1.307 vs 1.220); selected 1.270 / 1.297 / 1.287; collection 18 vs 34 min |
| Mechanism ablation | plain DCL data, loss renormalized within the deciding component (renorm) | DONE (3 seeds) | final 1.252 / 1.290 / 1.295 (mean 1.279); selected 1.267 / 1.276 / 1.288 (1.277) -> local labels = ~2/3 of the factored gain (final), ~85% (selected) |
| Allocation ablation | factored, equal component-days cf 40/40/40/40 | DONE (3 seeds) | final 1.295 / 1.306 / 1.263 (mean 1.288); selected 1.295 / 1.302 / 1.327 (1.308); 6993 decisions, 20 min (10 workers) -> ties value-based allocation: allocation is NOT a gain |

## Still to do (rough order; compute on 10 workers)
1. ~~Mechanism ablation~~ DONE: local labels explain most of the gain.
2. ~~Allocation ablation~~ DONE: equal = value-based within noise.
3. **Independent collections** (queue stage B, running) (~2 h): two more collections each for plain and
   factored DCL (new training-day seeds), 3 training seeds each.
4. **Held-out planner rules** (~1.5 h): SH 32 and uniform 8 on the 100 held-out days.
5. **Model-free RL on BPIC 2015 in the final representation** (~4-5 h, 5 seeds):
   PPO, NF-GAE marking, from scratch, clock node + merge; and the imitation start.
6. **DCL rounds 2-3** (plain ~10 h, factored ~5 h): network-base rollouts; keep the
   machine awake; 6-8 workers for memory.
7. **Generality** (~1 day): plain vs factored DCL on the insurer (2 regions) and the
   allocation variant, against NF-GAE.
8. **Component-alone reward check** with more days (the 2% gap at 20 days).
9. **Compute accounting** for every method (simulation steps, wall time).
10. **Theory write-up**: restricted-rollout and coalescing exactness; factored
    collection proposition; locality with the clock node.
11. Figures (draw.io style for the nets), tables, draft.

## Decision points and risks
- **Mechanism (DECIDED by the ablations)**: the story is "the certificate makes labels
  local" (1.22 -> 1.28 on the same data) plus "the certificate makes collection exact
  per component and about twice as fast" (1.29-1.31 at half the wall time). Allocation
  of component-days is a knob, not a contribution.
- **Independent collections show a smaller gap** -> report the mean and spread over
  collections; the comparison still holds if factored > plain on every collection.
- **Reviewer: "DCL is not yours"** -> DCL is the baseline; the contribution is what the
  certificate adds to it (exactness, locality, allocation), measured at equal compute.
- **Reviewer: "BPIC 2015 is mostly simulated"** -> say exactly what is calibrated and
  what is assumed; add the insurer generality results.
- **Normalization drift** across representation changes -> one anchor set for all
  tables.
- **Relation to the BPM paper**: DECIDED 2026-10-27, the journal paper supersedes it
  (the draft in paper/bpm/ is overwritten). Venue DECIDED 2026-10-28: Decision
  Support Systems (more prestigious than Information Systems, where the author already
  published; differentiates the venues). Reframed for DSS: decision-first abstract and
  introduction, a "practical properties" paragraph, Section 6 "Implications for
  Practice", refinement proof moved to the supplement.