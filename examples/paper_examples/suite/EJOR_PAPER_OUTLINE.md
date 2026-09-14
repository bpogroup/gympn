    # Paper outline — EJOR submission (v2, rewritten 2026-08-03)

*Supersedes the previous ccf-centric outline. Reason for the rewrite: the
credit-assignment story is now two mechanisms (not one), spanning a real
bias/variance/granularity/cost spectrum, AND the extensive negative-result
work this arc produced (reward shaping, input features, actor architecture)
turned out to be a rigorous, positive contribution in its own right — a
precise characterization of WHEN provenance-exploitation helps and when it
provably cannot. The old outline had no place for that; this one is built
around it from the start.*

*Working title (lead candidate):*
**"Exploiting Causal Provenance for Credit Assignment in Reinforcement
Learning for Dynamic Task Assignment: Mechanisms, Guarantees, and Limits"**

Alternatives:
- "When Does Causal Structure Help? Provenance-Exploiting Credit Assignment
  for Dynamic Task Assignment"
- "Causal Provenance as a Free Lunch — and Where It Runs Out: Credit
  Assignment for RL in Action-Evolution Petri Nets"

---

## 0. Positioning — the gap in the A-E PN program

| prior work | what it contributed | learning engine |
|---|---|---|
| Berti/... 2023 (2306.02910) | A-E PN formalism; taxonomy of dynamic task assignment; RL solves it | generic RL |
| 2507.03579 | assignment-graph representation; PPO for infinite state/action spaces | **generic PPO** |
| GymPN 2506.20404 | library; partial observability; multi-decision processes | **generic DRL** |
| **this paper** | **exploit the net's causal provenance for credit assignment — AND prove precisely where that exploitation stops paying off** | **s_ccf + DAG-replay (cfpk); rigorous boundary characterization** |

**Thesis of the paper:** an A-E PN simulation records, for free, *token
provenance* — which past decisions causally influenced which rewards. Prior
work discards this and hands the net to off-the-shelf PPO as a black-box
(S)MDP. We show provenance can be exploited for **credit assignment** in two
complementary, theoretically-grounded ways spanning a bias/variance/
granularity/cost spectrum — and, just as importantly, we show through a
battery of independently-falsified alternative mechanisms (reward shaping,
network input features, actor architecture) that provenance-exploitation's
benefit is *specific to credit assignment*, not general-purpose "give the
learner more structure." On problems that causally decompose (even
partially) into weakly-coupled sub-streams, the payoff is real, provable,
and cheap. On a genuinely single, tightly-coupled shared-resource
bottleneck, no amount of structural signal — however it is injected — moves
the needle, and we prove several of those routes are closed for principled
reasons, not because we didn't try hard enough.

**OR framing to foreground throughout:** this is a *decomposition* method
(Benders/Dantzig-Wolfe/Lagrangian lineage) — we decompose the policy-gradient
return by the causal-component structure of the stochastic system, with an
exact fallback (DAG-replay) when the decomposition is too coarse.

**Scope (in/out):**
- IN: the `mc_q -> lrq -> ccf -> s_ccf` credit spectrum; `s_ccf` as the
  unbiased-by-construction hero; DAG-replay/`cfpk` as the exact,
  finer-grained complement with a provably lossless cost reduction
  (coupling truncation); the boundary-condition investigation (reward
  shaping, structural input features, actor observability) as a rigorous
  negative-result contribution, not an appendix afterthought.
- OUT (separate work): the MCTS/AlphaZero-over-A-E PN direction; LS-HCA;
  LCV/LVA control-variate variants. Mention at most as related/future work —
  they are earlier, subsumed attempts on the same problem, not part of the
  headline contribution.

---

## 1. Introduction
- Dynamic task assignment / operational control: assign stochastic streams of
  tasks to limited, contended resources to optimize throughput/cost. Ubiquitous
  in OR (scheduling, service ops, manufacturing, logistics).
- A-E PN (prior work) gives an executable model + DRL solver; but the DRL is
  generic PPO that ignores the net's causal structure.
- The credit-assignment problem in concurrent systems: return-to-go mixes
  many cases' rewards; the advantage of a decision carries the variance of
  all cases in flight -> noisy gradients, unstable learning as concurrency
  grows.
- **Contributions:**
  1. We identify that generic policy-gradient RL discards the provenance
     structure A-E PN makes explicit, and that this causes credit-assignment
     variance that grows with concurrent independence.
  2. **`s_ccf`**: a topologically-derived (not hand-specified), provably
     unbiased credit decomposition, and a documented, mechanism-level account
     of *why* the naive realized-partition version (`ccf`) is biased — the
     realized causal partition is action-dependent at shared-resource
     handoffs; the static/topological partition is not.
  3. **DAG-replay counterfactual credit (`cfpk`)**: provenance used not as a
     credit *filter* but as a *replayable causal model* — exact per-decision
     counterfactual differences via forked simulation, finer-grained than
     `s_ccf` where that matters, with a provably lossless cost reduction
     (coupling truncation, measured 1.26x real wall-clock speedup at full
     training scale with byte-identical outcomes).
  4. A rigorous characterization of the **boundary** of provenance
     exploitation: real, significant, scaling gains on causally decomposable
     problems (N-copies, multi-site); a provable, honest null on a
     single-component/fully-coupled bottleneck. We stress-tested every
     plausible alternative route to closing that null — potential-based
     reward shaping (theorem-safe, empirically neutral-to-harmful at
     practical budgets), structural conflict-graph input features (provably
     redundant once the network already encodes action-type identity via a
     one-hot), and a genuine, *provable* graph-actor architecture blind spot
     (directed-only message passing leaves an action's logit unable to
     condition on unreachable state; proven via exact-invariance under a
     random untrained network, then fixed) — and show each is either
     provably inert or empirically neutral on the bottleneck case. Together
     these localize the bottleneck's difficulty to something other than
     credit attribution or observability — of independent interest to graph-
     RL practitioners on Petri-net/flow-structured problems generally.

## 2. Related work
- **Dynamic task assignment / operational control with (D)RL** (prior 3
  papers + broader: DRL for scheduling, queueing control, vehicle routing,
  job-shop).
- **Credit assignment in RL:** return decomposition (RUDDER), Hindsight
  Credit Assignment, temporal/structural credit assignment.
- **Multi-agent / factored credit:** difference rewards (Wolpert & Tumer),
  COMA, value decomposition (VDN, QMIX). *Position `s_ccf`/`cfpk` as: these
  need the entity/agent partition hand-specified; we derive it from
  provenance, per-episode, automatically.*
- **Causal RL / counterfactual credit** (Mesnard et al.); model-based
  counterfactual replay.
- **Potential-based reward shaping** (Ng, Harada & Russell 1999) — cited for
  the boundary-condition section, not the headline method.
- **Graph neural networks for combinatorial optimization / heterogeneous
  actor-critic architectures** — new bucket for the architecture finding
  (message-passing receptive fields, pooling in graph-RL actors).
- **Decomposition in OR** (one paragraph tying to Benders/DW/Lagrangian — the
  conceptual home of "decompose the return by structure").
- **Petri nets + learning** (brief; situate A-E PN).

## 3. Background / Preliminaries
- **A-E PN** (concise recap; cite prior papers): places, colored tokens,
  action/evolution transitions, the (S)MDP induced, rewards.
- **Token provenance / causal trace:** define the lineage DAG — for each
  token, which transition firing produced it from which input tokens; how
  rewards attach. (This is the object both methods exploit; already recorded
  by the simulator "for free.")
- **Policy-gradient RL for A-E PN:** PPO on the graph observation; SMDP
  discounting $e^{-\beta\tau}$; the return-to-go advantage and its baseline.

## 4. Method — two provenance-exploiting credit mechanisms

### 4a. The credit spectrum and `s_ccf`
- Table: `mc_q` (= PPO return-to-go; unbiased, high variance) -> `lrq`
  (single-lineage restriction; low variance, *foreclosure-biased*) -> `ccf`
  (realized-component restriction; low variance, **biased at shared-resource
  handoffs — the realized partition is action-dependent**) -> `s_ccf`
  (static/topological component restriction; **unbiased by construction**,
  costs some conservatism vs `ccf` where `ccf` happens to be right).
- **Causal components via union-find:** for each reward, collect the
  decisions in its lineage (backward DAG walk) and union them; two decisions
  share a component iff a chain of shared rewards links them. Realized
  (`ccf`) vs static/topological (`s_ccf`) partition — the AND-join/
  shared-handoff distinction that flips `ccf`'s sign (already demonstrated
  on the M2 motif). Algorithm box for both.
- Integration with PPO (advantage baseline); postpone handling (SMDP-TD
  mask). Complexity note (negligible vs. GNN passes).
- Reference: `CCF_EXPLAINED.md` mirrors this section in full.

### 4b. DAG-replay exact counterfactual credit (`cf`/`cfpk`)
- Reframe: provenance not as a filter deciding what reward "belongs" to a
  decision, but as a **replayable causal model** — fork the simulator at a
  decision, replay the alternative action under the same exogenous
  randomness (CRN), measure the *exact* realized difference in downstream
  return. Sidesteps the entire bias-from-wrong-partition failure mode `ccf`/
  `lrq`/LS-HCA all hit, because it never guesses a partition at all.
- Bias/variance/granularity tradeoff vs `s_ccf`: exact and unbiased on the
  case where `ccf`/`lrq` flip sign (M2), finer-grained than `s_ccf`'s
  conservatism where that matters (M4) — resolves the tradeoff `s_ccf` alone
  cannot.
- **Coupling truncation**: a provably lossless cost reduction — when the two
  forked branches' PN states reconverge (detected via a canonical
  symmetry-collapsed state fingerprint, reused from the paper's MCTS-adjacent
  work), simulate the shared tail once and add the identical value to both
  branches; the paired difference cancels the shared value exactly regardless
  of its realization. Measured 1.26x real wall-clock speedup at full training
  scale with byte-identical training outcomes (both a controlled test net and
  a real contested-topology env).
- Cost knobs and their role (fork probability, lookahead, rep count) —
  positioned honestly as the mechanism's remaining open cost story (coupling
  truncation is one, free, exact piece of closing the gap to plain PPO cost;
  not the whole story).

### 4c. Positioning table
`mc_q` / `lrq` / `ccf` / `s_ccf` / `cfpk` across {bias, variance, granularity,
extra compute cost} — the spectrum as a decision table for practitioners.

## 5. Theoretical analysis
- **P1 (s_ccf unbiasedness).** The static/topological partition's component
  membership cannot depend on the realized action (unlike `ccf`'s realized
  partition) — a strictly cleaner assumption than `ccf`'s A1, proved
  directly from A-E PN semantics (reachability is a property of the net's
  structure, not a trajectory).
- **P2 (variance reduction).** Under $K$ causally-independent components of
  comparable reward variance, `s_ccf` reduces per-decision advantage
  variance by ~$K$ vs. the full return-to-go (gradient SNR ~$\sqrt K$).
- **P3 (NEW — coupling truncation is exact in expectation).** Formalize the
  "shared value cancels exactly" argument: for a coupled fork pair whose
  branches reconverge to a common PN state, replacing two independent
  continuation samples with one shared sample leaves the paired-difference
  estimator's expectation (and, on a per-realization basis, its exact value)
  unchanged. State the reconvergence-detection condition (state fingerprint
  equality) precisely.
- **Contrast:** why `ccf` is biased (drops sibling-lineage opportunity cost
  that is *not* action-independent at an AND-join/shared handoff) — motivates
  `s_ccf` as the unbiased fix and `cfpk` as the exact, finer-grained
  alternative when `s_ccf`'s conservatism costs too much.
- Remark on components merging when cases share a resource (single-component
  collapse -> `s_ccf` == PPO), setting up the honest null that motivates §6's
  boundary-condition experiments.

## 6. Experiments

### 6a. Core credit-assignment results (the positive contribution)
- **Q1** Does `s_ccf` reduce to PPO when there is one causal component?
  (correctness)
- **Q2** Does `s_ccf` beat PPO as concurrent independence grows? (the win)
- **Q3** Is the win present on a realistic operational problem?
- **Q4** Does `ccf`'s bias mechanism manifest concretely, and does `s_ccf`
  fix it? (the AND-join/shared-handoff motif, estimator-level)
- **Q5** Is `cfpk`'s coupling truncation exact, and does it measurably reduce
  cost, at real training scale?
- **Environments:** N-copies scaling (controlled mechanism demonstrator,
  sweep N); archetype suite (single shared pool: s1/grid, the single-
  component null); realistic multi-site operational environment (the
  credibility anchor: `s_ccf` 0.52 vs PPO 0.15, p=.005, d=1.27, 12 seeds);
  a contested/joint topology for the `cfpk` wall-clock comparison
  (`g_exclusive_choice_joint`, full protocol scale: 1.26x speedup,
  norm_final identical).
- **Baselines:** PPO (discount-matched SMDP-GAE) [primary]; domain heuristic
  [anchor]; `mc_q`/`lrq`/`ccf` [spectrum ablation, shows the bias mechanism
  and the fix, not just the endpoint].
- **Protocol:** >=8-10 seeds, confidence intervals, significance tests;
  normalized return (0=random, 1=heuristic); learning curves; seed-variance;
  wall-clock cost.

### 6b. The boundary-condition investigation (the second contribution)
Framed explicitly as: *"having established when provenance-exploiting credit
assignment helps, we ask whether the residual gap on a genuinely
single-component bottleneck (s1: a two-stage service line with a single
shared, skill-heterogeneous resource pool) can be closed by any other
provenance/structure-based route — and show, with proof rather than
speculation, why each one is closed."*
- **Reward shaping.** Potential-based shaping from PN topology
  (Ng/Harada/Russell 1999, SMDP-generalized): provably policy-invariant for
  any potential function; empirically neutral-to-harmful at practical
  training budgets (a coefficient dial mapped from 0.2 "safe zone" to 1.0
  "decisive harm," attributed to injected value-target variance from
  exogenous arrivals, not a bias problem — the theorem's safety guarantee
  held throughout).
- **Structural input features.** Static, conflict-graph-derived per-action-
  type scalars (degree, hop-distance-to-reward): neutral on s1 (p=.58 before
  a targeted fix, p=.69 after). PROVEN, not just observed, to be redundant:
  the action-type one-hot encoding already fully distinguishes any two
  action types at zero hops, so any function of type alone — however well
  designed — cannot add information a linear readout couldn't already
  express.
- **Actor architecture.** A real, previously-undocumented graph-actor
  blind spot: `HeteroActor` decodes each action's logit from only that
  node's own final message-passing embedding (unlike the critic, which
  pools); combined with directed-only, token-flow-ordered graph edges (no
  reverse edges), an action can be *exactly*, provably blind to state
  outside its forward-reachable neighborhood within the network's depth —
  demonstrated via bit-for-bit exact invariance under a randomly-initialized
  network on a purpose-built topology, not a statistical argument. Fixed
  (pooled global context, mirroring the critic) and confirmed to close the
  gap on the same proof net (exact zero -> substantial, real sensitivity).
  Still empirically neutral on s1 itself (p=.91), because s1 happened to
  already have a marginal, accidental 3-hop backdoor path through its
  recycled resource pool — a secondary, general finding for graph-RL
  practitioners on Petri-net-structured problems, independent of the paper's
  main credit-assignment thesis.
- **Synthesis:** three independent, principled routes to "give the learner
  more structure" all fail to move the single-component bottleneck, while
  the only mechanisms that ever do (active forking/simulation, causal-
  component restriction) are exactly the credit-assignment mechanisms of
  §4. This localizes the bottleneck's remaining difficulty to something
  other than credit attribution or partial observability — an open question
  the paper states honestly rather than papering over.

## 7. Discussion / managerial insights
- **Decision rule for practitioners:** does your task-assignment problem
  decompose, even partially, into weakly-coupled sub-streams (parallel
  lines, multi-site, segmented resource pools)? If so, `s_ccf` gives real,
  provably safe, near-free gains; `cfpk` gives finer-grained exact credit at
  a bounded, partially-reducible extra cost when `s_ccf`'s conservatism
  costs too much. If your problem is a genuine single-bottleneck,
  fully-coupled resource contention, expect no gain from any provenance- or
  structure-exploiting mechanism tested here — plain, well-tuned PPO is the
  right tool, and the remaining difficulty likely lies in exploration or
  value estimation under contention, not credit assignment.
- Practical appeal of `s_ccf`: unbiased + ~free + drop-in (no hand-specified
  structure). `cfpk`: exact + finer-grained, at a compute cost that is
  partially, provably reducible.
- Limitations & future work: closing `cfpk`'s remaining cost gap (stacking
  coupling truncation with other fork-cost knobs); what WOULD close the
  single-component bottleneck (exploration-focused or value-architecture
  directions beyond what this paper tested); extension beyond A-E PN to any
  provenance-emitting simulator.

## 8. Conclusion
- A-E PN's provenance turns hand-specified factored credit into a *derived*
  quantity; `s_ccf` and `cfpk` give unbiased-or-exact, low-variance,
  cheap-or-boundedly-costly policy gradients for dynamic task assignment,
  wherever the problem's causal structure is even partially decomposable —
  and we show, rigorously rather than by omission, exactly where that
  exploitation stops paying off.

---

## Appendices
- A: full proofs (P1, P2, P3).
- B: environment/algorithm hyperparameters; reproducibility.
- C: the `mc_q`/`lrq`/`ccf`/`s_ccf`/`cf`/`cfpk` estimator definitions.
- D: boundary-condition technical detail — the phi-shaping invariance test
  derivation, the structural-feature redundancy proof (one-hot argument),
  the actor blind-spot exact-invariance proof and its architectural fix.

---

## Readiness checklist (what's done vs needed, v2)

| item | status |
|---|---|
| `s_ccf` method + implementation | **DONE** (`causal_traces.py`) |
| credit spectrum incl. `ccf` bias mechanism + `s_ccf` fix | **DONE** (`CCF_EXPLAINED.md` §5b) |
| N-copies env + scaling sweep | **DONE** |
| single-component null (grid/s1 == PPO) | **DONE** |
| realistic operational environment + results | **DONE** (multisite, p=.005, d=1.27) |
| compute-cost measurement (`s_ccf` ~1x) | **DONE** |
| P1/P2 (unbiasedness, variance) | **DONE, needs retitling ccf->s_ccf** (`EJOR_PROPOSITIONS.md`) |
| DAG-replay / `cfpk` method + implementation | **DONE** (`gympn/counterfactual.py`) |
| coupling truncation: exact-correctness proof (controlled net) | **DONE** (`_test_coupling_truncation.py`) |
| coupling truncation: real wall-clock speedup at scale | **DONE** (1.26x, `run_cfpk_coupling_wallclock.py`) |
| P3 (coupling truncation exactness, formal) | **DONE** (`EJOR_PROPOSITIONS.md` Part II) |
| boundary-condition: phi-shaping (safe, neutral-to-harmful) | **DONE** |
| boundary-condition: structural features (provably redundant) | **DONE** |
| boundary-condition: actor architecture blind spot (proven + fixed) | **DONE** (`_test_actor_global_context.py`) |
| boundary-condition synthesis writeup | **DONE** (`EJOR_BOUNDARY_CONDITIONS.md`) |
| related-work section (incl. new graph-RL-architecture bucket) | **DONE** (`EJOR_RELATED_WORK.md`; all citations verified live, not recalled; §2.7 Petri-nets-and-learning left as a placeholder, lowest priority) |
| figures (multisite, N-scaling, mechanism, boundary-condition summary) | **DONE for §6a** (`generate_figures.py`); **TODO for §6b** |
| discussion section (§7) | **DONE** (`EJOR_DISCUSSION.md` — the K-diagnostic in 7.1 is the section's sharpest new addition vs v1) |
| draft (LaTeX) | **DONE** (`paper/main.tex`, full rewrite, compiles clean via `latexmk -pdf`, 21 pages, 0 undefined refs/cites) |

**Critical path:** (1) P3 proof write-up; (2) §6b boundary-condition
writeup + 1-2 summary figures/tables (all underlying data already exists in
`causal-stability-suite` results and memory — this is writing, not new
experiments); (3) related-work additions; (4) full draft rewrite around this
outline. No new experiments are required for either §6a or §6b as currently
scoped — everything cited above is already run and validated.
