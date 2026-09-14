
# Beyond model-free PG: AEPN-native learning algorithms

*Written 2026-07-23. Companion to CAUSAL_LINEAGE_RETHINK.md (esp. §7.3),
CFPK_EXPLAINED.md, PAPER_PLAN_LCV.md. Motivating question: until now we have
used model-free, general-purpose algorithms (PPO and variants). Could we
instead define an **ad-hoc learning algorithm for Action-Evolution Petri Nets**
— possibly exploiting token lineage?*

## 0. TL;DR

- **"Learn directly from token lineage" is a road already driven to its end.**
  The whole LRQ→LCV→cfpl arc is lineage-as-credit-assignment on top of PPO, and
  the measured verdict (CAUSAL_LINEAGE_RETHINK §7.3) is that observational
  lineage earns exactly two general roles — **variance filtering** of the base
  estimator and **intervention targeting** — and cannot fix bias. It is a good
  copilot, a bad driver.
- **The real AEPN-specific lever is not the provenance DAG — it is that we own
  a perfect, executable, resettable forward model.** cfpk and DCL already use it
  (cfpk = depth-1 counterfactual fork; DCL = depth-h flat MC rollouts) but
  neither exploits it to the hilt.
- Three genuinely different "ad-hoc for AEPN" directions, ranked by how
  AEPN-native and how defensible they are: **(A) search *is* the learner —
  AlphaZero-over-AEPN**, **(B) structural (not observational) independence
  decomposition**, **(C) decision-skeleton + symmetry reduction**.

## 1. Why token lineage is not the engine (recap of the measured result)

All lineage consumer interfaces tried to date sit on top of a model-free PG
method and re-weight or restrict **factual** data by ancestry:

| interface | what it does | measured ceiling |
|---|---|---|
| lrq / lrq2 | advantage from lineage-restricted return | perfect on grid, 3–5× faster; **fails s1** (0.28 vs PPO 0.76) |
| lcv / lva | scalar control variate / aux critic target | ties discount-only baseline on finals (variance-only) |
| cfpk | **fork + CRN** exact 1-ply difference reward + SNR-gated preferences | perfect grid (ties lrq, faster), E1 10/10, s1 0.849 |
| cfpl | lineage restriction applied *inside* the fork | −11% paired SE (p=.002) but **+8% bias on s1** |

The cfpl result is the clincher: the lineage restriction removes exactly the
opportunity-cost signal (the fast employee kept free serves *other* cases,
whose rewards are by construction **not** descendants of the decision). So:

> **Lineage restriction lowers variance and raises bias; where foreclosure
> binds, the bias dominates.** (CAUSAL_LINEAGE_RETHINK §7.2b)

Conclusion carried forward: keep lineage in its two earned roles — (i) variance
filter on the base estimator, always-on and cheap; (ii) a prior for *where to
intervene* (where factual credit dispersion is high). Do **not** make it the
learning engine.

## 2. What is actually AEPN-specific

The property that separates AEPN from a generic MDP is not provenance. It is:

1. **White-box, executable, resettable forward model.** `deepcopy(env.pn) +
   env.i` is a complete mid-episode snapshot; behaviors draw their randomness at
   firing time, so a seeded RNG shim makes **common-random-number (CRN)** paired
   replays exact (validated 2026-07-18, `suite/diag_s1_counterfactual.py`).
2. **Static structural knowledge of concurrency/conflict.** Two transitions that
   share no input place can never interact — their firings commute, *exactly and
   data-independently*. The net topology hands us a sound independence relation
   for free.
3. **Semi-Markov / evolution structure.** Action phases alternate with timed
   evolution phases; decisions occur only at genuine conflicts. The "real" MDP
   over conflict states is far smaller than the raw firing sequence.
4. **Colored-token symmetry.** Interchangeable tokens of the same color collapse
   large swaths of the reachable state space.

cfpk exploits only (1), and only to depth 1. The three directions below exploit
(1) fully, and (2)–(4) at all.

---

## 3. Direction A — Search *is* the learner: AlphaZero-over-AEPN

**One-liner.** Replace model-free PG with **PUCT/MCTS over the white-box net**,
using the existing HGT/GNN embedding as policy prior and value, trained on
search-improved targets (the AlphaZero policy-improvement operator) instead of
PPO's clipped surrogate.

**Why it fits AEPN uniquely.** Perfect resettable simulator + exact CRN paired
replays + the SMDP discount already in use. This is the honest generalization of
what we have been circling:

- cfpk = a **depth-1** fork (taken action vs one alternative).
- DCL = **depth-h flat** MC rollouts (no tree, no reuse of sub-results).
- MCTS = the tree both are special cases of, with a learned prior+value guiding
  expansion and a principled backup.

**What it subsumes / reuses.**
- The entire `counterfactual.py` fork machinery: snapshot/restore, CRN sibling
  suffixes, greedy continuation, and the `lookahead` value-tail bootstrap are
  exactly the primitives an MCTS node-expansion needs.
- The R4 counterfactual gaps become **search statistics for free** (they are
  edge Q-value differences at the root).
- **Coupling truncation** (branches converge under CRN ⇒ stop simulating; spec'd
  in §7.2 but never built) is the search-specific cost control — prune a subtree
  the moment two branch states coincide (marking + in-flight delays + clock).

**Training loop (sketch).**
1. At each decision state, run N simulations of PUCT using `agent.value_model`
   for leaf evaluation and policy logits as the prior over bindings.
2. Emit the visit-count distribution π_MCTS as the improved target.
3. Train the policy toward π_MCTS (cross-entropy) and the value toward the
   search-backed return — no clip, no GAE.
4. CRN across sibling expansions to cancel shared service noise (the s1
   noise-cancellation that made cfpk work at depth 1, now at every node).

**Predictions / falsifiers.**
- Grid: matches cfpk/lrq perfect record (1.00) — floor by planning quality.
- E1 one-shot family: 10/10 optimal (search trivially sees the terminal trap).
- s1: ≥ cfpk's 0.849 and plausibly higher — full-depth search sees the
  foreclosure cascade cfpk's 1-ply fork only partially captures.
- Falsifier: if s1 does **not** exceed cfpk, the binding constraint is
  eval/exploration variance, not planning depth — which itself is a clean
  finding (pairs with the X10/X11 retention story).

**Cost.** The obvious risk. cfpk already ran ~3.6× the no-fork baseline at
depth-1 lookahead-6; a tree multiplies node count. Mitigations, all already in
scope: coupling truncation (the big one, never implemented), lineage-targeted
expansion (spend simulations where factual credit dispersion / policy entropy is
high), and small N with a strong learned prior (AlphaZero works at N≈50–200).

**Related work to cite.** AlphaZero policy-improvement operator; MuZero (but we
do **not** need a learned model — ours is exact); this is "planning with a
perfect simulator," differentiated from model-based RL that learns the model.

**Verdict.** Strongest paper-method candidate. Clean thesis: *the model is
white-box, so plan exactly instead of doing model-free PG.* Reuses the whole
existing fork stack.

### 3.1 PROTOTYPE BUILT (2026-07-23)

**Code.** `gympn/mcts_planner.py` (PUCT search; returns the DCL planner's exact
`(target_pi, stats, enabled)` contract), `gympn/agents_mcts.py`
(`MCTSAgent(DCLAgent)` — swaps the planner, inherits run_episode + the
AlphaZero-style CE-to-visit-counts distillation + value regression + train
loop), CLI/`make_agent` wiring under `algorithm="mcts"` (`--mcts_sims`,
`--mcts_c_puct`, `--mcts_max_depth`, `--mcts_temp`, `--mcts_conflict_gate`,
`--mcts_dirichlet_alpha`). `tests/test_mcts_planner.py` (4/4 green).

**Design decisions worth recording:**
- **Leaf evaluation by the value net, never random rollouts** — a uniform tail
  was the X15 DCL failure. SMDP e^{-beta·Δt} discounting on the sim clock,
  value-tail bootstrap at the depth cap.
- **Reuses the env snapshot/CRN API** (`get_state`/`set_state`/`set_seed`) —
  the same primitives cfpk and DCL validated; per-simulation reseed = CRN over
  the search (root-sampling / determinized MCTS, exact on deterministic envs).
- **Signature-keyed tree** (transition id + consumed-token keys) survives
  positional churn and stochastic divergence. Crucially the token key falls
  back to **value+place** when there is no causal_rl `_id`, which makes it
  **symmetry-aware**: interchangeable same-colour tokens collapse to one
  signature (a free sliver of the Direction C symmetry idea).
- **Direction B folded in as the conflict gate** (`conflict_gate`): at a node
  where all enabled bindings structurally commute (no shared input place, no
  shared token, not postpone) the search collapses to the single highest-prior
  child instead of branching — spending simulations only on genuine contested
  decisions. Ablatable to measure its worth.

**Validation.** On the deterministic E1b one-shot foreclosing choice (horizon 2,
chain pays 12 / shortcut pays 7), the search with a **uniform prior and no
critic** puts its visit mass on the chain — the exact opportunity-cost
discrimination every scalar credit method failed and cfpk nailed, now via
lookahead. Env state is restored intact after search (the X15 corruption guard,
tested). End-to-end smoke: `MCTSAgent` trains through `training_run` (plan →
act → distil) with no errors. *(Test gotcha worth remembering: the foreclosure
only bites at the designed horizon 2 — with more horizon both tasks complete,
total 19, and the choice is moot. Set env horizon deliberately.)*

### 3.2 COUPLING TRUNCATION BUILT + MEASURED (2026-07-23)

The cost control cfpk never had (its 3.6× P3 miss). In MCTS the faithful analog
of cfpk's pairwise CRN coupling is a **state-fingerprint transposition table**:
`state_fingerprint(pn)` = canonical marking (symmetry-collapsed token values) +
in-flight token times + clock. Two branches that reach the same fingerprint have
coincided — identical future, identical discount-to-t0 — so they become **one
node, searched once**. Made exact by two design choices: the horizon is
**clock-based** (`lookahead`, like cfpk) not decision-count-based, so a node's
truncation horizon is a function of its state alone; and the clock is in the
fingerprint, so discounting stays consistent across transposed paths.

Two settings (`MCTSConfig`): `coupling_truncate` (node-sharing, no fidelity
loss — reuses estimates, improves sample efficiency) and `couple_min_visits`
(>0 also **short-circuits** re-entry into a resolved coupled state: return its
cached value, skip the descent — the actual env.step saving; exact under a
deterministic continuation, raise the threshold on stochastic envs). Telemetry:
`transpositions`, `truncations`, `steps`. Tests in `test_mcts_planner.py`
(7/7 green): reuse fires on reconvergent states, the short-circuit cuts steps,
and node-sharing preserves the E1 decision.

**Measured saving (grid decision states, gate ON, `couple_min_visits=4`):**

| env | transposition hits | env.steps m=0 | m=4 | saved |
|---|---|---|---|---|
| a_sequence_joint | 240 | 304 | 77 | **75%** |
| c_parallel_joint | 193 | 257 | 83 | **68%** |
| e_loop_joint | 252 | 316 | 77 | **76%** |
| g_exclusive_choice_joint | 194 | 258 | 84 | **67%** |

67–76% fewer env.step calls, *on top of* the conflict gate. This makes a larger
`mcts` sweep affordable — the precondition §3.1 and the cfpk P3 note both
flagged before scaling up.

### 3.3 LINEAGE-ATTRIBUTED BACKUP — the lineage contribution (2026-07-23)

**The idea.** Standard MCTS backs up the *whole-trajectory return* to every edge
on the path — a decision is credited (and blamed) for rewards it did not cause
(concurrent cases, other work in flight). That is `mc_q`. The provenance DAG
lets the search do what no vanilla MCTS can: credit each in-tree decision edge
**only by the rewards causally descended from that decision's produced tokens**
(lineage-restricted return). That is `lrq` — which beat `mc_q` **28W/0L,
p<1e-4** on a factual trace — now lifted inside the tree backup.

**Why it escapes cfpl's bias.** cfpl showed lineage restriction lowers variance
(−11% SE) but raises bias (+8% finals on s1) because it drops the
opportunity-cost signal (rewards from cases the decision kept resources free
for are not its descendants). In MCTS that bias is carried elsewhere: the
**sibling comparison**. Q(assign match) vs Q(assign generalist) already encodes
"what else you could have done" — the tree evaluates the alternative directly.
So lineage supplies the low-variance per-decision credit, and the tree supplies
the opportunity cost. **Lineage for variance, interventions for bias — unified
in one algorithm**, instead of split across methods. This is the paper's central
mechanism claim.

**Implementation.** `MCTSConfig.rollout_backup` (PUCT in-tree + policy rollout
to the clock horizon; each decision scored by its whole return-to-go = the fair
`mc_q` baseline) and `lineage_backup` (implies rollout; scores each decision by
its causal-descendant return via the validated `_lineage_return` /
`descendants_of` from the fork stack). Both read rewards from the causal trace,
so the env must be `causal_rl=True`; the value tail is dropped for both (as in
cfpl, so the only difference is the restriction). Coupling truncation still
shares nodes; the value-reuse short-circuit is disabled in rollout mode (the
rollout must complete for trace-based backup). CLI: `--mcts_rollout_backup`,
`--mcts_lineage_backup`. Tests (`test_mcts_planner.py`, 11/11): lineage backup
prefers the chain on E1 (per-edge causal credit, no value bootstrap); whole and
lineage agree on E1 (do-no-harm); lineage backup fails loudly without a trace;
state restored.

**Measured — the variance win, directly (s1, one decision state, Q-mean spread
over 40 CRN searches, 24 sims each):**

| backup | Q[0] mean | Q[0] SD |
|---|---|---|
| whole (mc_q-in-tree) | 0.218 | 0.071 |
| **lineage (lrq-in-tree)** | 0.146 | **0.054** |

**~24% lower estimator SD** (≈42% variance), the tree-search reproduction of the
lrq-vs-mc_q effect. The lower mean is expected (lineage return is a subset of
whole); the open question the s1 A/B answers is whether the sibling comparison
covers the resulting bias so finals hold or improve. End-to-end: lineage-backup
`MCTSAgent` trains through `training_run` on a causal env.

**Registered prediction (the decisive test).** On s1, lineage-backup MCTS
converges in fewer simulations than whole-backup MCTS (variance) AND holds or
beats it on finals (the tree covers the bias). **Falsifier:** if lineage-backup
*loses* finals on s1, the sibling comparison does NOT cover the foreclosure bias,
and lineage stays a co-mechanism (variance/speed only), not the headline. On the
grid + E1 (direct-dominated) the two must agree — the do-no-harm check.

**Next:** the A/B — `mcts_lineage` vs `mcts_whole` (and vs cfpk) on s1 + E1 +
the a–h grid, same fork budget; s1 is the decisive env.

### 3.4 A/B ATTEMPT — two bugs fixed, one open blocker (2026-07-23)

Wrote the s1 A/B runner (`run_mcts_ab_s1.py`) and tried to launch it. It
surfaced three things, in order:

**(1) Shared DCL distillation bug — FIXED (also unblocks DCL).** First s1 cell:
greedy 0.0 (distilled policy = always-postpone) while sampled was nonzero (the
search picks real actions). Root cause in `agents_dcl._fit_policy_and_value_model_step`:
the actor emits nodes in by-TYPE order `[all a_transition, all postpone]`
(HeteroActor.forward), but the CE step used the flat `batch.target_pi`, which
PyG batches per-GRAPH `[g0_a,g0_p,g1_a,…]` — misaligned, so CE trained the wrong
nodes. Fixed to use the correctly-batched per-node-type targets the buffer
already stores. **Validated on E1b: greedy 0 → optimal 12.0.** This is the X15
distillation bug (memory 2026-07-22), now resolved for both MCTS and DCL.

**(2) Postpone scale mismatch — FIXED.** Do NOT credit postpone by whole
return-to-go while real actions get lineage: whole >> single-case lineage, so
postpone dominates and the search postpones everything (measured: sampled all
0.0). Correct: score EVERY edge by lineage, and run `causal_postpone_tokenflow=False`
so postpone (a pure timing action) has lineage return ~0 — consistent scale, no
attractor. Confirmed at a decision: real `start1` Q≈0.19 > postpone Q=0.0.

**(3) THE REAL BLOCKER — a PUCT tie-break normalization bug — FIXED.** My first
diagnosis (a "rollout-tail postpone attractor") was WRONG; direct instrumentation
of the real training loop (prompted by the right question — is it an
action-sampling mismatch?) found the actual cause. At s1's early decisions the
prior is uniform (postpone 0.102 ≈ each real action 0.100 — no ordering
mismatch, postpone correctly last) and EVERY action's backed-up Q is 0.000
(rollouts see no reward early, since a case rarely finishes within the horizon
under a random tail). In that all-equal regime my Q-normalization returned
**0.5 for a visited action but 0.0 for an unvisited one** — an asymmetry, so the
first action visited scored `0.5 + U` and beat every unvisited `0.0 + U`,
capturing ALL simulations. The visit distribution collapsed to one action and
`target_pi` came out one-hot (on s1: postpone), which the agent then correctly
executed — 42/42 postpones. E1b never triggered it: its rollout hits the reward
immediately (12 vs 7), so Q values spread, `rng > 0`, and normalization is fine.
Fix (one line, `_select.qnorm`): when `rng <= 0` return 0.0 for visited too, so
selection is driven by `prior*U` and visits SPREAD. After the fix, s1 decision-0
`target_pi` favors real `start1`/`start2` (postpone prob 0.12) and the queue
stops exploding (the agent does work instead of deferring).

**Status after the PUCT fix.** The SEARCH now works on s1: sampled return
(training episodes, which sample the search's target) = **~11**, near the
heuristic anchor 15 — up from 0. But greedy (eval, distilled-policy argmax) is
still **0.0**. So a genuine TRAIN/INFERENCE GAP remains: the search is good, the
distilled reactive policy does not reproduce it.

**(4) The distillation gap (train/inference).** Hypothesis (being verified, not
yet confirmed — the lesson from (3) stands): the search target spreads good mass
across MANY real-action nodes whose identity changes every step (tied to
transient tokens), while POSTPONE is a single STABLE node getting a steady ~0.12
at every state. The net fits the stable postpone signal and underfits the
non-stationary per-token targets, so at argmax postpone (0.12) beats each
individual real action (0.88 split over ~10 nodes ≈ 0.09). E1b has only 2 real
actions (≈0.44 each > 0.12) so its greedy worked — consistent.

**Options for (4) (each ~30–97 min/cell — s1 cells got slower once the agent
actually does work, so autonomous rapid iteration is no longer cheap):**
- **Evaluate WITH the search** (visit-count argmax) instead of the raw policy —
  standard AlphaZero does this, and the search policy already scores ~11. Cleanest
  for the paper's question (does lineage help the PLANNER); the distillation to a
  fast reactive policy is a separate, known-hard sub-problem.
- **Sharpen the target** (low `mcts_temp`) so the distillation target concentrates
  on the top real action and drives postpone's target toward 0.
- **Postpone handling at eval** (exclude when a real action is clearly preferred).

The lineage-MCTS mechanism and the search-side result (~11 on s1) STAND; the
headline A/B is gated on choosing an eval path for (4).

**(4) RESOLVED (2026-07-24) — sharpen the distillation target.** Instrumented
first (lesson applied): the target is GOOD (mean postpone 0.107, mean max-real
**0.649** over 184 decisions) — the GNN just wasn't learning it. Confirmed
mechanism: ~10.8 actions ⇒ the average real action gets (1−0.107)/9.8 ≈ 0.09
< postpone 0.107, and the target's 0.649 peak sits on a transient token node the
GNN can't identify, so it spreads real-action mass and the single stable postpone
node wins argmax. Fix: `mcts_temp = 0.5` sharpens the target so postpone (never
the peak) is driven toward 0, which the GNN learns reliably, so its argmax lands
on a real action. **Result (temp 0.5, 3 epochs): greedy 0 → [11.6, 11.4, 10.6]
(≈0.6 normalized); GNN postpone-argmax fraction 100% → 2% (7/412); GNN mean
postpone prob 0.055, max-real 0.404.** The STANDALONE GNN now works on s1 — the
train/inference gap is closed and "GNN argmax = policy" holds end to end. The
other two options (eval-with-search, postpone masking) are unnecessary. Runner
set to `mcts_temp=0.5`. The A/B is now genuinely unblocked.

**Lesson worth keeping.** All three collapses looked identical from the outside
(greedy 0.0, always-postpone) but had three unrelated causes — a batching
misalignment in the loss, a scale mismatch in the backup, and a normalization
asymmetry in selection. "Always-postpone" is this project's generic failure
signature, not a diagnosis; instrument the actual decision before theorizing.

### 3.5 SPEEDUP — 2.0× via cheap binding refresh (2026-07-24)

Profiled one lineage search: **~80% of the time was `get_graph_observation`**
(the HeteroData tensor build), rebuilt on *every* `env.step` — but the rollout
only builds it to feed the policy net for the continuation. The blocker to
skipping it: `pn.pn_actions` (the firable bindings needed to *fire*) is
constructed *inside* `get_graph_observation`, interleaved with the tensor loop.

Fix: extracted `GymProblem.compute_pn_actions()` — the expansion + binding-map
loop (same `b_time<=clock` filter, same order, trailing postpone) **without** the
tensor build; verified byte-identical bindings to the graph path across states.
`env.step(build_obs=False)` now refreshes bindings via `compute_pn_actions` and
returns no graph; the MCTS rollout builds the full graph **only** at leaf
expansion (for the prior) and runs a **real-only uniform** tail (never postpone —
a postponing tail starves the lineage/descendant reward; measured it raised eval
postpone-argmax 2%→26%, real-only brings it to 7%).

**Measured:** graph builds/search 798→60; `get_graph_observation` 3.19s→0.31s;
**wall-clock 649→318 ms/search = 2.0×**. Greedy preserved/improved:
[11.0, 11.2, **11.6**] (policy-tail was …10.6). 22 tests green. The standard
graph path is untouched (`build_obs` defaults True). Remaining bottleneck:
`expand_no_future_tokens` (still per step) — next lever if needed (raw simpn
`bindings()` bypass, or the G2 no-rollout value head).

---

## 4. Direction B — Structural (not observational) independence decomposition

**One-liner.** Lineage *estimates from traces* which rewards are causally
related. The net **structurally guarantees** it. Two transitions sharing no
input place can never interact; the structural conflict graph is a **sound, free,
exact over-approximation** of the "may-interact" relation. Factor the
value/advantage over structurally-independent subnets read straight off the
topology, instead of over a noisy estimated causal DAG.

**Why it fits AEPN uniquely.** This is the one object CRN provably **cannot**
deliver: CRN cancels noise *shared* by two branches, but once branches diverge,
parallel activity is uncancelled (this is exactly why cfpl exists and why its
lineage restriction has value). The structural conflict graph tells us *a
priori* which concurrent activity is genuinely independent — no estimation, no
trace, no variance.

**Mechanism.**
- Build the **conflict graph**: nodes = transitions; edge iff they share an input
  place (structural, static, computed once from the PN definition).
- Independent components ⇒ their reward streams are additively separable; credit
  need not — and should not — be shared across them.
- Decompose the advantage/value along these components. Where lineage *estimates*
  "this concurrent case is unrelated" and pays variance to do it, the structure
  *asserts* it exactly and for free.

**Caveat (state honestly).** For **colored** nets the interaction relation is
data-dependent (guards, markings), so the structural relation is a *sound
over-approximation*: structural independence ⇒ true independence (safe to
exploit), but two structurally-conflicting transitions may still be independent
in a given marking (missed opportunity, never a wrong decomposition). That
asymmetry is what makes it safe to build on.

**Predictions / falsifiers.**
- On envs where concurrent-case noise is the binding constraint (the grid, where
  lrq's variance filter already wins), the structural decomposition should match
  or beat lineage restriction at **zero** variance cost and **zero** bias
  (unlike cfpl, which pays bias for its variance reduction).
- Falsifier: if structural independence is too coarse (everything shares a
  resource place ⇒ one big component), the decomposition is trivial and buys
  nothing — measure component granularity on the actual envs before committing.

**Verdict.** Most novel-looking piece; most AEPN-formalism-pure. Composes with
Direction A (structural independence tells the search which subtrees are
separable and need not be jointly expanded). Cheapest first probe: write the
conflict-graph extractor and measure component granularity on s1 / grid / E1.

### 4.1 BUILT + MEASURED (2026-07-23)

**Code.** `gympn/conflict_graph.py` (extractor: two relations — *conflict* =
share an input place; *coupling* = share any place ⇒ connected components =
independent subnets — with union-find, a granularity report, and clock
exclusion); `examples/paper_examples/suite/run_conflict_graph_probe.py`
(no-training granularity sweep); `tests/test_conflict_graph.py` (5 tests,
green). Validated on the E1 chain exactly as designed: it recovers the **single**
meaningful decision `start_A2 <-> start_B` over `employee_shared` and nothing
spurious. A synthetic control of two genuinely place-disjoint pipelines correctly
splits into 2 components, so the suite result below is a real property of the
envs, not an extractor artifact.

**Probe result across all 14 suite envs (a–h grid, s1–s3, E1 family):**

| relation | outcome | reading |
|---|---|---|
| **coupling components** (independent subnets) | **14/14 envs = ONE blob** (largest-component fraction 1.00 everywhere) | the value-decomposition mechanism as literally stated has **no teeth** on any current env |
| **conflict graph** (contested-input decisions) | joint envs ≥1, **disjoint envs exactly 0**, E1 = 1, s3 = 3 | a **correct, discriminating, free** structural signature |

**Why the coupling decomposition collapses — two structural causes, both
measured:**
1. **Sequential token flow.** A pipeline `waiting1→busy1→waiting2→busy2` is one
   connected subnet even when each stage has its own dedicated resource pool
   (`b_sequence_disjoint`: 0 resource conflicts, still 1 component).
2. **A single shared source place.** Parallel case streams that never compete for
   a resource are still coupled through one common `arrival` place
   (`d_parallel_disjoint`: 0 conflicts, still 1 component).

**Consequence for Direction B (the honest update).** The falsifier in the
Predictions block *fired*: static coupling independence is too coarse on these
topologies — it is defeated by shared sources and end-to-end flow. So "factor
value/advantage over structural coupling components" buys nothing here as
written; a decomposition with teeth needs a **finer, marking/color-aware**
independence notion (per-case / per-color flow separation), not pure static
structure — which reintroduces exactly the data-dependence the static relation
was meant to avoid.

**What survives, and is immediately useful.** The **conflict-graph half** is
cheap, exact, and correct where the coupling half is not:
- It gives a **free structural joint-vs-disjoint label** (contested-input count),
  the very axis the a–h grid is organized around — no training, no trace.
- Its best role is **intervention targeting for Direction A**: the action-vs-action
  conflict edges are precisely the genuine contested decisions worth forking /
  expanding, and singleton (unconflicted) actions can be fired greedily without a
  fork. This is the general, formalism-level answer to "where should search spend
  its budget," replacing the lineage-dispersion heuristic with a static, exact
  one. Fold it into Direction A rather than shipping Direction B standalone.

---

## 5. Direction C — Decision-skeleton + symmetry reduction (SMDP-native)

**One-liner.** Most of an AEPN is deterministic evolution; the agent only
chooses at genuine conflicts. Compile the reachability graph down to its
**conflict states only** and learn over that — plus colored-token **symmetry
reduction**. The real MDP is far smaller than the firing sequence PPO currently
sees.

**Why it fits AEPN uniquely.** The action/evolution alternation and the
free-choice structure mean decision points are sparse and the "between-decision"
dynamics are (often) deterministic given the schedule. Colored tokens of the
same color are interchangeable ⇒ enormous state aggregation.

**Mechanism.**
- **Decision skeleton:** the reachability graph restricted to states with >1
  enabled binding (a genuine conflict). Deterministic runs between conflicts
  collapse to single SMDP transitions with an accumulated discounted reward and a
  Δt.
- **Symmetry reduction:** quotient the state space by color-permutation symmetry
  (canonical marking representative). The GNN already gets this *implicitly* via
  permutation invariance; making it explicit is exact where the GNN only
  approximates, and shrinks the DP.
- On small nets: exact SMDP value iteration / DP becomes feasible → an exact
  reference policy to benchmark every learned method against.
- On large nets: a drastically smaller learning problem for any of A/B.

**Predictions / falsifiers.**
- Small nets (E1 family, a–h grid cells): exact DP recovers the known optima and
  gives a ground-truth column for the paper.
- Falsifier: if the conflict states are still exponentially many after symmetry
  reduction (color explosion), exact DP is infeasible and this stays a
  *reduction* used to feed A/B rather than a standalone solver.

**Verdict.** Best as infrastructure/benchmark (exact reference policies) and as a
state-space reducer feeding Directions A/B, rather than a headline method on its
own.

---

## 6. Where token lineage stays

Not the engine — but it keeps the two roles §7.3 already earned, inside whichever
of A/B/C ships:

1. **Variance filter** on the base estimator (lrq vs mc_q, 28W/0L — always-on,
   cheap, factual).
2. **Intervention targeting** — a prior for *where* to spend forks / MCTS
   simulations (states with high factual credit dispersion or high policy
   entropy).

## 7. Recommended sequence

1. ~~**Direction B probe first (cheapest, most novel):** write the conflict-graph
   extractor and measure granularity on s1 / grid / E1.~~ **DONE 2026-07-23 (§4.1).**
   Verdict: the coupling-component decomposition has no teeth on any current env
   (14/14 one blob — the falsifier fired), but the conflict graph is a correct,
   free joint/disjoint signature and the right **intervention-targeting** object
   for Direction A. Direction B does not ship standalone; its useful half folds
   into A. A finer marking/color-aware independence notion is the only path to a
   value-decomposition method, and it forfeits the "pure static structure" appeal.
2. **Direction A prototype (strongest method) — now the head of the queue:** a new
   `agents_*` / method-key variant built on the existing `counterfactual.py` fork
   machinery (snapshot, CRN suffix, coupling-truncation hooks), using the §4.1
   conflict graph to decide **where to fork** (contested-action states) and where
   to fire greedily (unconflicted actions). A/B against cfpk on s1 + the E1
   one-shot family + the a–h grid. Implement **coupling truncation** here first —
   it is the shared cost control both A and cfpk need.
3. **Direction C as infrastructure:** exact SMDP DP on the small envs to produce
   ground-truth reference policies for every column in the paper.
4. Lineage stays on throughout in its two earned roles; measure its marginal
   contribution *inside* A (fork targeting) as an ablation.

## 8. Honest bottom line

The question "can we do better than model-free PPO on AEPN?" has a clear answer:
**yes, and the leverage is the white-box model, not the provenance DAG.** We are
already partway there (cfpk = depth-1 planning; DCL = flat rollouts). The
untapped moves are (A) letting *search* be the learner instead of a credit patch,
(B) reading concurrency structure off the net *statically* instead of estimating
it from traces, and (C) compiling the net down to the small decision-and-symmetry
core it actually is. Token lineage remains a useful copilot in all three — a
variance filter and an intervention-targeting prior — but the driver is the
executable model.