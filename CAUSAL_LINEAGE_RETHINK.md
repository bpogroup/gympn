# Lineage Rethink: making the provenance information actually pay

*Written 2026-07-17, after the X9/LVA tie. Companion to PAPER_PLAN_LCV.md,
CAUSAL_LRQ_PROPOSAL.md, CAUSAL_LCV_CONTROL_VARIATE.md, CAUSAL_GAP_D_FORECLOSURE.md.*

## 1. Where we actually stand

Three consumer interfaces have now been tried and all of them tie the
discount-only control (lcv0) on final returns:

| interface | scheme | result |
|---|---|---|
| policy-gradient credits (A = Q_lin − V) | lrq/lrq2 | perfect on the a–h grid (1.00±0.00), 3–5× faster on s2/s3, **fails s1 (0.28 vs PPO 0.76)** |
| scalar control variate | lcv | ≈ lcv0 on finals (grid p=.52, stoch p=.99); only edge = grid AUC (p=.005) |
| auxiliary critic target | lva | ≈ lcv0 everywhere (grid final p=.98); grid conv 6.5 vs 7.3 p=.27 |
| Q-native AWR | lqi | fast warm-up, plateaus ~0.31 (probe, s1 only) |
| exact decomposition c_lin + q̂_off | lrq3 | cold-start race, worse than v2 everywhere |

Meanwhile two facts prove the lineage *information* is not the problem:

1. **lrq vs mc_q on the grid: 1.00 vs 0.695, 28W/0L, p<1e-4.** Identical
   estimator, identical discount — the lineage restriction *is* the entire
   effect. The lineage filters concurrent-case exploration noise that the
   return cannot.
2. **The scripted credit-greedy ordering scores the full anchor on s1
   (14.8 > trained PPO 13.6) with zero training.** The category ordering
   induced by the credits is *sound* even on the env where every trained
   consumer fails.

So the paradox to resolve: the lineage demonstrably contains
policy-improving information (1, 2), yet every gradient-side pipe we have
built extracts either speed-only (grid) or nothing (finals). This document
is a systematic pass over (a) why, and (b) what has not been tried —
including changing how the lineage itself is constructed.

## 2. Failure taxonomy — what exactly binds where

Naming the failure modes precisely, because each proposal below must say
which one it attacks:

**F1 — Foreclosure / opportunity-cost blindness (s1's bias).**
A slow assignment's cost is the work its employee did NOT do. Important
nuance verified in the code (envs.py `done1: [busy1] -> [pool, waiting2]`,
stoch_envs.py identical; simulator.update_causal_trace registers every
result token with parents = all consumed tokens): the freed employee IS
re-emitted through the DAG, so an assignment is an ancestor of every later
reward flowing through that employee. The opportunity cost is therefore not
absent from Q_lin — it enters only as a *second-order timing shift*
(downstream thread rewards arrive later ⇒ slightly smaller e^{−βΔt}). On s1
that signal is ~0.06–0.26 in Q units, at or below per-sample service-time
noise (±0.12) — measured in the 2026-07-13 diagnosis. The counterfactual
first-order term ("employee free 1 tick earlier ⇒ one extra task started")
is what's missing.

**F2 — Magnitude SNR / premature commitment (s1's dynamics).**
Within-state credit gaps are tiny and clustered; PPO's clip + multi-epoch
reuse amplifies small *consistent* advantages into ε-sized moves every
epoch, the policy commits (entropy 0.14 vs PPO 0.48) before the fine
discriminations are learned, and the KL brake locks it in. The credits'
*ordering* is right; their *magnitudes* are not trustworthy at the scale PPO
consumes them.

**F3 — The asymptote ceiling of variance devices (lcv/lva's tie).**
A control variate — and, evidently, a representation-shaping aux loss —
cannot move the asymptote of an unbiased PG method that already converges;
it can only buy variance⇒speed, and at 20 episodes/epoch on these envs the
baseline variance is apparently not the binding constraint. This is not a
bug; it is what the theory says. Any proposal whose only mechanism is
variance reduction will reproduce the tie on finals.

**F4 — Cold-start races (lrq3/lqi).** Learned heads injected into the
advantage at full weight from epoch 1 race the policy's commitment. Any
proposal with a learned component needs a warm-up or a trust gate.

Corollary that organizes everything below: **to move FINAL returns, a
proposal must either (i) inject information the return doesn't contain
(counterfactuals — attacks F1), (ii) change the objective so ordering is
consumed instead of magnitudes (attacks F2), or (iii) change the learning
dynamics (exploration, annealing, trust regions — attacks F2/F4). Pure
variance devices (F3) are exhausted.**

## 3. Proposals

Ordered roughly by (evidence they'll work) × (1/cost). Each has: mechanism,
which failure it attacks, cheapest probe, registered prediction, falsifier.

### R1 — Consume the ordering, not the magnitudes (rank/preference interface)

The one consumer proven to work on s1 is the *category-aggregated credit
ordering* run as a scripted policy. No trained interface consumes that
object. Two concrete versions:

- **R1a (category prior):** each epoch, aggregate per-decision credits by
  action category (the same aggregation diag_s1.py used: assignment type ×
  match/cross/generalist). Build a Boltzmann prior π_prior ∝ exp(rank/T)
  over the categories available in each state, and add a KL(π‖π_prior)
  regularizer with coefficient annealed → 0. Floor by construction
  (coefficient → 0 recovers PPO exactly).
- **R1b (pairwise preference loss):** for pairs of decisions in similar
  states within an epoch, emit a preference a≻b whenever
  credit(a) − credit(b) > k·σ_noise (heteroscedastic gate — only consume
  gaps that exceed their own noise), and train the policy logits with a
  logistic/DPO-style pairwise loss alongside PPO. The SNR gate directly
  neutralizes F2: clustered credits produce *no* preference pairs instead
  of noisy gradients.

Attacks: F2 (primary), F4 (no learned head in R1a).
Probe: s1 × 10 seeds, R1a with T from the measured category gaps.
Prediction: s1 final ≥ 0.9 normalized (the scripted ordering achieves 1.0;
the prior should carry the policy most of the way) while grid stays 1.00.
Falsifier: s1 unchanged ⇒ the failure is not consumption-side at all —
which would point hard at F1 and elevate R4/R5.
Cost: small — aggregation code exists in the diag path; one new loss term.

### R2 — Teacher distillation from the credit-greedy policy

Stronger version of R1a: run the scripted credit-greedy ordering as an
explicit teacher. Early epochs: behavior-clone / KL-pull toward the teacher
(computed fresh each epoch from that epoch's credits, so it improves as the
policy explores better regions); anneal the pull to zero and let PPO
fine-tune. This is "the lineage tells you a good policy directly — start
there," sidestepping the gradient interface entirely.
Attacks: F2, and empirically sidesteps F1 to whatever extent the scripted
ordering does (on s1: fully).
Probe: s1 + s3, 10 seeds.
Prediction: matches or beats the best of (scripted, PPO) per env; on the
grid indistinguishable from lrq2 speed.
Falsifier: the policy degrades after the teacher is annealed away ⇒ the
ordering is state-independent advice PPO can't refine — informative in
itself (would mean the info is *policy-level*, not *state-level*).
Cost: small-moderate. Highest probability of a headline s1 win.

### R3 — Two-timescale anneal: lrq2 early, lcv0 late

The data already says: lineage buys convergence speed (s2 conv 3.6 vs 16.4
epochs; s3 6.8 vs 15.2), the discount buys the asymptote (lcv0 finals).
Nobody has taken the obvious free lunch: μ(t) schedule mixing A_lrq2 →
A_smdp-gae over ~10 epochs (or switch triggered by a credit-gap/entropy
statistic).
Attacks: F3 honestly (concedes the CV point, claims speed) + repairs lrq2's
s1 bias by switching away before commitment locks in — provided the switch
happens early enough (s1 entropy collapse is early; schedule must be too).
Probe: nearly free — both estimators exist; a schedule on causal_mu-style
mixing. s1+s2+s3 × 10 seeds.
Prediction: s2 conv ≤ 6 epochs AND final ≥ 0.95; s1 final ≥ 0.7 (i.e., PPO
level — the anneal must not import lrq2's s1 damage).
Falsifier: s1 still collapses ⇒ early lrq2 epochs already do irreversible
commitment ⇒ pairs with R6.
Cost: trivial. Even if "boring," this is the strongest *practical method*
candidate for the paper: one method, best-of-both columns.

### R4 — Counterfactual lineage (new creation): CRN re-simulation

The PN is a white-box simulator we own. For a decision state s with taken
action a and alternative b, replay the episode from s under b with **common
random numbers** (freeze the behaviors' random draws; they are drawn at
firing time — stoch_envs.py note — so a seeded RNG shim per episode makes
paired replays exact). The paired difference G(s,a) − G(s,b) is a true
counterfactual advantage gap including ALL general-equilibrium effects:
freed-resource cascades, foreclosure, queue displacement. This is the
information that no lineage *walk* over the factual trace can contain
(F1's first-order term), obtained by *creating different data* instead of
re-weighting the data we have.

Three uses, in order of importance:

1. **Diagnostic upper bound (do this first).** On s1, sample ~200 decision
   states, compute exact counterfactual gaps. If the true gaps are also
   ≈ noise, then *no consumer can win on s1* at this budget — the
   extraction-gap story is fundamental and the paper should say so with
   this measurement as its centerpiece. If the true gaps are large where
   credit gaps are tiny, F1 is confirmed as the binding failure and R4/R5
   become the method.
2. **Sparse counterfactual credits:** at the k most-visited/most-uncertain
   states per epoch, replace/correct the lineage credit with the CRN gap
   (budget: a few dozen replays/epoch — each is one episode-suffix
   simulation, cheap in these PNs).
3. **Counterfactual-supervised q-head:** train q̂(s,·) on CRN gaps (labels
   only where computed) — fixes lrq3's cold start with ground-truth labels
   instead of bootstrapped noise (F4).

Attacks: F1 directly (the only proposal that injects genuinely new
information — corollary route (i)).
Prediction (for use 1): s1 true counterfactual gaps between match/cross
assignments exceed 3σ of service noise (the scripted ordering's success
implies the aggregate effect is real; question is per-state magnitude).
Cost: moderate (RNG plumbing + replay-from-state machinery; the simulator
already reconstructs states for training). This is COMA's counterfactual
baseline done *exactly* instead of via a learned critic — worth a
related-work sentence (CAUSAL_RELATED_WORK_COMA_RUDDER_SHAPLEYQ.md).

**R4-use-1 RESULT (2026-07-18, `suite/diag_s1_counterfactual.py`, 160
branch states from 40 heuristic episodes, 10 CRN reps, β=0.5 discounted
remaining reward; raw data `diag_s1_counterfactual.json`): PREDICTION
HOLDS — s1 IS winnable.**

| pair (disc) | n | median gap | med \|gap\|/SE | >2SE | sign consistency |
|---|---|---|---|---|---|
| s1_cross − s1_match | 137 | −0.171 | **4.46** | 91% | 91% negative |
| s2_cross − s2_match | 44 | −0.328 | **8.45** | 98% | 100% negative |
| s1_cross − s1_gen | 59 | −0.151 | 3.54 | 69% | 85% negative |
| s1_gen − s1_match | 74 | −0.015 | 2.07 | 50% | mixed |

Three findings beyond the headline:

1. **The signal lives in the discount, not throughput.** Undiscounted
   gaps have per-state SNR ≈ 1–1.5 (noise-dominated); the β=0.5-discounted
   gaps reach SNR 4–8. Assignment quality barely changes *how many* cases
   finish by the horizon — it changes *when*. Consumers targeting
   undiscounted outcome quantities are chasing noise on s1 by
   construction.
2. **Why lineage credits fail where CRN succeeds: variance, precisely as
   theorized.** The true discounted gaps (0.17–0.34) are the *same order*
   as the measured lrq2 credit gaps (0.06–0.26) — the information is not
   absent from the factual signal's mean; it is drowned because a lineage
   credit is one unpaired factual sample against service noise ±0.12,
   while CRN pairing cancels the common noise. F1's practical content is
   an estimator-variance problem that only counterfactual pairing (R4) or
   aggregation (R5's epoch-level shadow prices, R1's category pooling)
   can fix — per-state factual re-weighting cannot.
3. **The lineage credit ordering is partly WRONG, not just weak: it
   inverts cross vs generalist.** Credits rank s1_match .15 > s1_cross
   .08 > s1_gen .03; the truth is match ≈ gen > cross (gen−match ≈ −0.02,
   NS; cross−match ≈ −0.17). Mechanism: the credit charges the generalist
   its slow service (3) but cannot see that assigning the generalist
   keeps a fast matched employee free — the foreclosure benefit. The
   scripted credit-greedy ordering still hit the anchor because
   match-first dominates, but any consumer that trusts the full credit
   ordering inherits an inverted tail. This is direct, quantitative
   evidence for the F1 bias, not just tiny magnitudes.

Consequence for the sequence in §5: the gate is passed — R5a's shadow
prices now have ground truth to validate against (predict: p_r·τ pricing
must reproduce the cross<gen inversion), and R4-use-2/3 (sparse CRN
credits / counterfactual-supervised q-head) are live method candidates.
R8 remains insurance, no longer the default destination.

### R5 — Channel-split lineage + resource shadow prices (new creation)

The lineage walk currently returns one scalar per decision. But the DAG
distinguishes *how* a reward reaches a decision: through the decision's own
case tokens, or through a freed-resource hop (pool re-emission). Split the
credit into **case-channel** Q_case and **resource-channel** Q_res during
the walk (tag the traversal when it crosses a pool token — the wiring
supports this today).

Then two consumption options:

- **R5a (explicit occupancy pricing):** estimate each resource's shadow
  price p_r = (epoch credit earned through r) / (busy time of r) — a
  dual/LP view of the assignment problem. Advantage:
  A = Q_case − p_r·τ_service − V. The foreclosure cost becomes *first-order*
  (you are charged for occupying r at its market rate) instead of the
  second-order timing shift the walk gives now. This is a bias
  *correction*, not a variance device — it can move finals (route (i)).
- **R5b (two-feature critic):** give the critic/policy both channels as
  separate inputs (or aux targets — but note lva's tie curbs enthusiasm for
  the aux-target form; prefer the explicit advantage form R5a).

Attacks: F1 with epoch-level statistics instead of per-state replays
(cheaper than R4, more biased — p_r is a mean-field approximation of the
counterfactual).
Probe: s1 × 10 seeds with R5a; validate p_r against R4-use-1 gaps if both
run.
Prediction: s1 ≥ 0.6 (between lrq2's 0.28 and PPO's 0.76 would already
prove the mechanism; ≥ PPO would make it a method).
Falsifier: no movement ⇒ mean-field pricing too coarse for s1's
within-stage discriminations ⇒ only R4 remains for F1.
Cost: moderate — walk tagging + one epoch statistic + advantage change.

### R6 — Credit-dispersion-gated exploration (dynamics, not gradient)

Use the lineage where it is *reliable* — as a confidence signal, not a
gradient. Per state (or category): if the credit distribution over
available actions is clustered (gap < k·σ), impose an entropy floor /
temperature boost; where gaps are decisive, allow commitment. This attacks
the exact measured pathology (premature commitment on clustered credits)
without touching the estimator — the policy gradient stays plain PPO/lcv0,
so the floor is structural.
Attacks: F2's dynamics half.
Probe: piggyback on any s1 run; the gate statistic already exists in the
diag path.
Prediction: s1 entropy stays ≥ 0.4 through epoch 15 and final ≥ PPO.
Cost: small. Composes with R1/R3/R5 (orthogonal mechanism).

### R7 — Potential-based shaping from lineage mass (guaranteed-floor densifier)

Define Φ(s) = discounted lineage mass "in flight" (credits already banked
by tokens currently alive, computable from the trace at state s). Shape
r' = r + γΦ(s') − Φ(s). Policy-invariance (Ng et al.) guarantees the
asymptote is untouched — floor by theorem, not by construction-and-hope.
Mechanism: densifies the sparse completion rewards along the lineage,
which is a *dynamics* effect (earlier signal ⇒ less premature commitment
to wrong things), not a variance effect.
Attacks: F2 (signal density), immune to F1/F3 criticism by invariance.
Honest caveat: invariance also means it cannot fix a bias — expect speed,
not new finals; this is a cleaner-theory sibling of R3.
Cost: small-moderate (Φ from trace at eval states).

### R8 — Budget sweep: make the variance story true where it lives

If R1–R5 all tie on finals, the honest claim is sample-efficiency — and it
was never actually tested in its favorable regime. Sweep episodes/epoch
∈ {5, 10, 20} on the grid + s2, methods {lcv0, lcv, lva, lrq2-anneal}.
Prediction (registered in the X0 postmortem): the lcv−lcv0 and lva−lcv0
gaps grow monotonically as the budget shrinks; lrq2's speed advantage
becomes a final-return advantage at 5 eps/epoch (fixed wall-clock budget
⇒ faster convergence = higher final within budget).
Attacks: nothing — it *reframes* F3 as the claim instead of the obstacle.
Cost: compute only (~⅓ of X0 per budget point). This is the paper's
insurance policy.

### R9 — (creation) Contested-resource negative edges — noted, deprioritized

One could add explicit *competition* edges to the DAG (decision a took the
token/resource that decision b's case was waiting for ⇒ negative-lineage
edge a→b's-delay). This is a discrete approximation of what R4 computes
exactly and R5 prices in aggregate; it inherits Shapley-style ambiguity
about who contested what (the old shapley_dag wash is a warning). Only
worth revisiting if R4 shows large per-state gaps AND R5's mean-field
pricing is too coarse AND full R4 consumption is too expensive.

## 4. Failure-mode coverage matrix

| | F1 foreclosure bias | F2 SNR/commitment | F3 asymptote ceiling | F4 cold start |
|---|---|---|---|---|
| R1 rank/preference | – | **✓✓** | route (ii) | ✓ (R1a headless) |
| R2 teacher distill | (empirically) | **✓✓** | route (iii) | ✓ |
| R3 anneal | avoids | ✓ | concedes, claims speed | ✓ |
| R4 counterfactual CRN | **✓✓** | ✓ (true gaps) | route (i) | ✓ (true labels) |
| R5 shadow prices | **✓** | – | route (i) | ✓ |
| R6 gated exploration | – | **✓** | route (iii) | ✓ |
| R7 potential shaping | – | ✓ | invariant (speed only) | ✓ |
| R8 budget sweep | – | – | reframes | – |

## 5. Recommended sequence

1. **R4-use-1 (diagnostic) + R3 (anneal) in parallel** — one is the
   cheapest decisive *measurement* (is s1 winnable at all?), the other the
   cheapest decisive *method* (best-of-both columns for the paper either
   way). R4-use-1 gates everything: if true counterfactual gaps on s1 are
   noise-level, skip R5/R9, run R8, and write the extraction-gap paper
   with the measurement as evidence.
2. **R1a or R2 on s1** — the direct test of "consume the ordering." R2 if
   we want the strongest shot at an s1 headline; R1a if we want the
   cleaner floor story.
3. **R5a** if R4-use-1 confirmed large gaps — the principled bias fix,
   validated against R4's ground truth.
4. **R6** folded into whichever of the above survives (orthogonal, cheap).
5. **R8** regardless — the sample-efficiency claim should be measured for
   the paper even if a finals win materializes.

## 6. Honest bottom line (written pre-diagnostic; see §7 for the update)

The experiments to date do not show the lineage is useless; they show that
**re-weighting factual data by ancestry extracts only variance information,
and variance is not what binds at this budget.** The two untried directions
that change that equation are (a) consuming the *ordering* (which is
proven sound) through an interface built for orderings, and (b) *creating*
counterfactual or channel-split lineage that contains the first-order
opportunity-cost signal the factual walk provably lacks. If both fail —
and R4-use-1 tells us in one afternoon whether they can succeed — the
extraction gap is the paper, and it will be a measured claim rather than a
disappointment. *(Update: R4-use-1 ran 2026-07-18 and the gaps are real —
see the R4 result block and §7.)*

## 7. Generality pass (2026-07-18) — which proposals survive arbitrary topologies?

*Triggered by the (correct) objection that R5 is assignment-specific. The
critical re-read found the flaw is systematic, not local to R5.*

### 7.1 The mistake, named

R1a's "categories" (stage × pairing), R2's scripted teacher, and R5's
"resources" all smuggle in the same thing: **hand-crafted semantics of
token *values*** (`task_type`, `code_employee`, "this place is a pool").
In every case that structure was doing one statistical job: **variance
reduction by pooling** — category pooling gave ~100 samples per cell,
shadow prices averaged over an epoch of flow. That is env-specific feature
engineering in disguise, and it does not transfer: a general A-E PN has no
stages, no pairings, no annotated resource places.

The test any proposal must pass: **it may touch only the PN formalism
(places, transitions, bindings, markings, firings, the trace DAG) and
learned functions of the GNN state embedding — never the semantics of
token values.** Everything below passes it.

The general substitutes for pooling are exactly three: (i) **same-state
intervention pairing** — don't pool across states, cancel the noise within
one state via CRN forks; (ii) **function approximation** — the HGT
embedding is the general pooling device we already have; (iii)
**formalism-level aggregation only** (per-transition, per-place — allowed,
but note per-transition pooling cannot separate bindings of the same
transition, which is exactly where s1's discrimination lives, so it is
general-but-too-coarse on its own).

Verdict per proposal: R3, R4, R7, R8 general as stated. R1b general only
if "similar states" becomes "same state" (→ forks). R1a, R2, R5, R6 as
written: **not general — superseded below.** R9 general in form, still
dominated.

### 7.2 The general set

**G1 — Forked counterfactual preferences (flagship; subsumes R1b + R4-use-2
+ R6).** During ordinary training rollouts, at a decision state with
probability q: pick ONE alternative binding b (general choice rule: the
policy's second-highest logit — logits exist for every binding in every
env), fork 2–3 CRN sibling suffixes for a and b, and record the paired gap
in β-discounted remaining reward. If |gap| > 2·paired-SE (the R6 gate,
now exact and general): emit a preference (s, a ≻ b) consumed by an
auxiliary pairwise logistic loss on the policy logits, coefficient
annealed → 0 (floor = PPO by construction). Nothing in the loop reads
token values. The R4 diagnostic already validated every ingredient: the
per-state signal is decodable by pairing (SNR 4–8 at 10 reps ⇒ ~2.5–4 at
3 reps, enough for the gate to pass on the real gaps and reject the
clustered ones — the gate *is* the fix for F2, and the gaps' exactness is
the fix for F1, including the cross/gen inversion the factual credits get
wrong).

Cost control, both formalism-general: (a) **coupling truncation** — under
CRN with a deterministic (greedy) continuation, the moment the two branch
states coincide (marking + in-flight delays + clock), their futures are
identical and the gap is final: stop simulating. Contested effects are
local; expect coupling within a few time units on loaded systems. Cap at
horizon as fallback. (b) **lineage-targeted forking** — spend forks where
the factual credit dispersion among a state's *transitions* is high or
where the policy is uncertain (entropy). This is the provenance DAG's new
general role: a cheap prior for *where interventions are worth buying*.

Probe: fork machinery (one module, shared with G2) + the aux loss; run
s1 × 10 seeds, then — the actual generality test — the **E1 one-shot
foreclosing-choice family** (no resources at all; k=3 variant trapped REC
10/10) and the a–h grid untouched at 1.00. Registered predictions:
s1 ≥ 0.9 normalized; E1-k3 10/10 optimal; grid unchanged; fork overhead
< 30% of rollout compute with truncation on.

**X10 PROBE RESULT (2026-07-19, method 'cfp', s1 × 10 seeds, 10/10
trained, 0 failed; run_cfp_s1.py, gympn/counterfactual.py): P1 FAILS,
floor holds, and the failure mode is the INFORMATIVE one.** norm_final
0.751 ± 0.14 = lcv0 0.756 (p=.85) = ppo 0.760 (p=.91); still >> every
lineage consumer (lrq2 0.276 p=.004, lrq 0.357 p=.002). Weak positives:
best-of-run 0.90 ± 0.06 (highest of any method on s1), 2/10 seeds
touched 95% (lcv0/ppo: 0–1/10), greedy AUC +0.083 over lcv0 (p=.084,
8W/2L). Entropy final 0.45 (no collapse — P2 holds). Cost ≈ 2.5× lcv0
(P3 missed; truncation was lookahead-only, no coupling check).

The pre-registered falsifier check (inspect transmission before touching
hyperparameters) shows the interface DID transmit: preference counts
19–24/epoch throughout, and the per-preference loss fell 0.68 → ~0.44
(the policy ends ranking fork-winners ~1 logit above losers). **The
preferences were learned and finals still didn't move.** Combined with
the standings — every intact-floor method lands 0.72–0.76 on s1 (ppo
.760, lcv0 .756, cfp .751, lva .74, mc_q .723, lcv .721) while their
best-checkpoints reach 0.85–0.90 — the working hypothesis shifts: **s1's
remaining gap to the anchor is not a credit-quality problem; it looks
like a shared PPO-family ceiling (optimization/exploration/eval
variance), which no advantage- or preference-signal improvement can
cross.** The R4 diagnostic proved the states are individually decodable;
X10 shows that even consuming exactly that decodable signal, learned to
satisfaction, does not beat the ceiling.

Decisive next probes (cheap, in order): (i) mechanism check — rerun cfp
with the anneal REMOVED (constant coef; floor knowingly sacrificed): if
hard preference pressure still tops out ~0.76, the ceiling claim is
confirmed against the strongest version of G1; (ii) decompose the
residual anchor gap of a converged 0.76 policy (which decisions differ
from the heuristic, per-state — the diag machinery does this); (iii) the
E1/grid generality runs still stand on their own merits (G1's value may
be the one-shot-choice topologies, not s1). If (i)+(ii) confirm the
ceiling, s1's story in the paper changes from "the env credit assignment
fails on" to "the env where the binding constraint provably isn't credit
assignment" — with X10 + R4 as the two-sided proof.

**X11 RESULT (2026-07-19, method 'cfpk' = cfp with constant coefficient,
s1 × 10 seeds, 0 failed): P2 — THE ANNEAL WAS THE LIMITER. First method
to break s1's 0.72–0.76 band.** norm_final **0.849 ± ~0.08** (band top
was ppo's 0.760); 4/10 seeds reached 95% (all-time record; every other
method 0–2/10); one seed BEAT the anchor (14.85 vs 14.78); drift 0.47
(lower than cfp 0.72/lcv0 0.61 — constant pressure did not destabilize;
the P3 destructive scenario did not occur; entropy final 0.45). Paired
tests: vs cfp (the exact anneal ablation, same seeds) +0.097, 7W/3L,
**p=.035**; vs lcv0 +0.092 (p=.32 finals — n=10 underpowered — but AUC
+0.107 **p=.037**); vs ppo +0.088 (p=.082, 7W/2L/1T).

Corrected reading of X10: the band was never an optimization ceiling —
it is crossable by exactly the signal X10 transmitted, held at constant
strength. What X10 actually demonstrated is **retention failure**: PPO's
own return gradient at s1's noise level cannot RETAIN the fine
discriminations after the preference pressure is annealed away (learned
→ forgotten). That is F2's premature-commitment mechanism seen from the
other side, and it says the counterfactual signal must either stay on or
be consolidated (distillation/frozen head), not faded out.

Status of G1 after X10+X11: general method, first s1-band break, but the
by-construction floor is gone (constant coef) — empirically intact
though (no seed below 0.72, drift lower). Next, in order: (a) generality
runs E1 + a–h grid with cfpk (the empirical-floor question is now the
important one: does constant pressure damage envs that don't need it?);
(b) +10 s1 seeds for significance vs lcv0 on finals; (c) schedule study
(anneal to a plateau ~0.3 instead of 0 — soft floor, likely keeps most
of the gain) only after (a); (d) s2/s3 tier completion for cfpk.

**X12 GENERALITY RESULT (2026-07-20, run_e1_cfpk.py + run_cfpk_grid.py,
110 cells total, 0 failed): ALL SIX PREDICTIONS CONFIRMED, THREE
DRAMATICALLY. cfpk is the FIRST method in this project's history to
score a perfect grid.**

*E1 family (10 seeds × 3 variants, zero env-specific code):*

| variant | cfpk | best baseline |
|---|---|---|
| oneshot k=2 (P2, the win case) | **10/10 optimal, 0 trapped** | ppo 4/5, lrq 3/5, rec 2/5 trapped |
| oneshot3 k=3 (P1, do-no-harm) | 10/10 optimal | ppo/lrq 10/10 (matched, not beaten) |
| full repeated choice (P3) | 9/10 optimal, mean 113.3 | ppo mean 111.0 (4/5) |

P2 lands exactly as predicted: on the topology class with no resources
at all — where the retired R5 (shadow prices) provably had nothing to
average over — cfpk is perfect where every baseline including lrq drops
seeds into the trap. Same method key, same code, zero oneshot-specific
lines. This is the generality claim, demonstrated.

*a–h grid (10 seeds × 8 topologies, `suite_results_paper`):*

**cfpk aggregate: norm_final 1.00 ± 0.00, norm_best 1.00 ± 0.00, drift
0.00 ± 0.00 — ties lrq's perfect record exactly, on EVERY SINGLE ENV,
EVERY SEED (80/80).** This exceeds P1 (predicted: ≥ lcv0's 0.84, no env
below) and swallows P2 (predicted: lift f≥0.90/h≥0.75 — actual: f and h
both hit their exact optimum, 1.00, matching a/c/e/g which were already
solved). Paired tests (n=80, by env×seed):

| vs | Δ norm_final | p | Δ drift | p | Δ AUC (speed) | p |
|---|---|---|---|---|---|---|
| lcv0 | +0.156 | .0003 | −0.362 | .0002 | +0.148 | <.0001 |
| lcv | +0.134 | .0006 | −0.250 | .0006 | +0.074 | .0049 |
| ppo_clip | +0.310 | <.0001 | −0.550 | <.0001 | +0.313 | <.0001 |
| lrq | 0.000 (tied, 80/80) | — | 0.000 (tied) | — | **+0.729** | <.0001 |

The lrq row is the most informative: cfpk matches lrq's perfect
final/drift record exactly (both 1.00±0.00, 0.00±0.00 — no wins, no
losses, 80/80 ties) while being reliably FASTER to converge (AUC
+0.729, the largest effect size in the whole comparison table,
essentially every non-tied pair a cfpk win). So on the grid, the general
method (no lineage machinery, no token semantics, forks + preferences)
is not just competitive with lrq's biased-but-perfect policy-gradient
consumption — it gets there quicker.

P3 (cost) is the one miss: cfpk averaged 8.5 min/cell over 683 min total
vs the no-fork lva/lcv0-era grid baseline of ~2.4 min/cell (≈3.6×, not
the predicted <2×) — lookahead-6 truncation alone is not enough;
coupling truncation (branches converge under CRN ⇒ stop early, proposed
in §7.2 but never implemented) is now a priced-in next step, not a
nice-to-have, before any larger sweep.

**Where this leaves the paper.** G1/cfpk is now the strongest method
result in the whole project: general (E1, zero-resource choice
topologies), state-of-the-art on the grid (ties lrq's perfect record,
beats it on speed), and — pending the fuller s1 seed count — likely the
first method to also lead the noisy-discrimination tier. Remaining
before it can be called the paper's method: (a) resolve X10 vs X11 on
s1 with a larger seed count (n=10 underpowered the finals test even
though AUC was already significant) — floor status without the
by-construction guarantee needs the numbers, not just the theory; (b)
s2/s3 tier completion for cfpk; (c) coupling truncation for cost, since
any larger sweep at 3.6× cost is expensive; (d) the retention-failure
reframe of X10 (cfp) belongs in the write-up regardless — it is a
genuine, general finding (preference pressure must be sustained or
consolidated, not fully annealed) independent of whether cfpk ships as
the headline method.

**G2 — Counterfactual-supervised q-head (general successor of R5's
*intent*, fixes lrq3's F4).** Same forks consumed in value space: train
q̂(s,·) on the exact paired gaps (labels only where bought), let the GNN
generalize across states — function approximation replacing shadow-price
pooling. Then either A = q̂-based advantages (lrq3 with real labels
instead of bootstrapped noise — its top seeds were the best s1 numbers of
any variant, so the ceiling is demonstrably high) or q̂ as critic baseline.
Combine with G1 freely (policy-space + value-space consumption of one
fork stream).

**G3 — R3's anneal, unchanged** (was already general): lrq2 → lcv0
schedule. Still the cheapest practical win and the fallback method.

**G4 — R7's potential shaping, unchanged** (lineage mass in flight is
formalism-level). Speed-only by invariance; low priority.

**G5 — Place-level prices (the honest salvage of R5).** λ_p per PLACE
from trace flow — expressible without resource annotations, BUT: on
one-shot foreclosing choices (E1) each token is consumed once, there is
no repeated flow to average, and the mean-field price of a choice place
collapses to an average over arms — useless exactly where foreclosure
bites hardest. Keep only as a cheap diagnostic baseline against G2,
never as the method. *(This is the critical downgrade the objection
demanded: R5 wasn't just assignment-specific in wording — its statistical
mechanism requires repeated flow, which arbitrary topologies don't
provide.)*

**G6 — R8's budget sweep, unchanged.** Insurance.

#### 7.2b X13 — putting the lineage back INSIDE the counterfactual (cfpl)

cfpk wins by brute-force simulation and uses the provenance DAG for
nothing. X13 asks whether the lineage can do real work inside the fork,
in the one place it has a *measured* edge: lrq beat mc_q 28W/0L with an
identical estimator and discount, differing only in restricting the
summed reward to a decision's causal descendants — which filters
CONCURRENT, causally-unrelated reward. That is a different noise source
from the one CRN cancels (CRN removes noise SHARED by the branches; it
cannot touch parallel activity once they diverge), so the two should
stack. `cfpl` applies exactly that test inside each branch.

Three-arm ablation, 10 seeds, s1, constant coefficient, identical fork
budget (40 forks/epoch), so the lineage effect is isolated from the value
tail that lineage mode must drop:

| arm | branch return | norm_final | paired SE | gap/SE | gate-pass | prefs/fork |
|---|---|---|---|---|---|---|
| cfpk | raw + V tail | **0.849** ± .08 | 0.0480 | 2.77 | 42% | 0.42 |
| cfpn | raw, no tail | 0.838 ± .10 | 0.0481 | 2.81 | 41% | 0.41 |
| cfpl | **lineage**, no tail | 0.773 ± .15 | **0.0428** | **3.51** | **53%** | **0.53** |

**M1 (primary) CONFIRMED, decisively — 10/10 seeds on every mechanism
metric, p=.002 each:** the lineage restriction cuts the paired SE by 11%,
raises the gap/SE ratio 25%, and lifts the gate-pass rate 28% (53% vs
41%). **M2 CONFIRMED:** 28% more usable preferences per fork — i.e. per
unit of simulation, which is the cheapest available attack on cfpk's 3.6×
cost. The control contrast shows the value tail was irrelevant (SE +0%,
p=.92; finals −1%, p=.83), so the ablation is clean: this is the lineage,
not the tail.

**P1 (finals) FAILED, and the failure is the finding.** cfpl lands at
0.773 vs cfpn's 0.838 (−8%, 4W/6L, p=.36 — not significant, but
directionally negative with a wider seed spread, .15 vs .10; best-of-run
is unchanged at 0.92, so this is retention, not reach).

**The variance/bias decomposition, measured for the first time.** On s1 a
large part of what the lineage discards as "concurrent and unrelated" *is
the opportunity-cost signal*. Assigning a task to the generalist is good
precisely because a fast matched employee stays free for other cases —
and those other cases' rewards are, by construction, NOT descendants of
that assignment. So the restriction systematically removes the very term
that makes the objective correct here. This reproduces F1 (foreclosure
blindness) in miniature and explains, mechanistically, both lrq2's 0.28
on s1 and the R4 diagnostic's cross-vs-generalist inversion — but now
with the two effects separated and signed:

> **the lineage restriction lowers variance (−11% SE, p=.002) and raises
> bias (−8% finals) — and on s1 the bias dominates.**

That is a sharper, more useful statement than either the LRQ-era "the
lineage is the entire effect" or the LCV-era "the lineage extracts
nothing": both were reading a single net number produced by two
opposite-signed mechanisms.

**Testable consequence, not yet run.** Where foreclosure is NOT the
binding constraint the bias term should vanish while the 28% signal
efficiency stays free — so cfpl should be neutral-to-better on the a–h
grid (where lrq's fully-lineage-restricted credit already scores
1.00±0.00) and on E1's one-shot choices (where the alternative forecloses
nothing, since the arms are terminal). Predicted: cfpl matches cfpk's
perfect grid at ~28% fewer forks' worth of usable signal. If that holds,
the shipping recommendation is **lineage-restricted forks everywhere
except foreclosure-dominated envs**, with the R4 diagnostic as the cheap
test for which regime an env is in.

### 7.2c X14 — lineage as a DECOMPOSITION with a modelled indirect channel
(`cfpd`): built, quick-tested, **NEGATIVE on the conditioning statistic**

X13's reading said the lineage is not a filter but a decomposition of a
decision's value into a **direct** effect (descendant rewards; low
variance) and an **indirect** opportunity-cost effect (high variance, but
where the foreclosure signal lives). The natural method follows: take the
direct channel as sampled, and Rao-Blackwellize the indirect channel —
replace its noisy per-fork sample with a regression on a *smoother*
statistic, pooled across forks. Both channels come from the same rollouts,
so the decomposition is free.

The conditioning statistic must be lower-variance than the reward
difference itself or the whole thing is pointless, so we used
lineage-derived **occupancy**: how long and how heavily a decision ties the
system up (subtree close-time, discounted promptness, descendant count),
which is near-deterministic under CRN because service draws are shared.

Implemented (`cf_decompose`, method `cfpd`) with a mandatory safety floor:
the correction is used only if it generalizes (2-fold held-out R² ≥
`cf_min_r2`), else the raw total gap is used — degrading to `cfpn` rather
than inventing a new bias. Resolver unit-tested on synthetic data (R²=0.99
=> model, SE 0.014 vs raw 0.50; noise-only => correctly falls back; a large
negative indirect term correctly flips the winner).

**On real s1 data the statistic does not predict the channel.** With
per-epoch fitting (40 forks, 5 params) held-out R² was mostly negative —
that is a sample-size artifact, so the fit was pooled over a rolling
400-fork window across epochs. Pooled result: **R² ≈ 0.00–0.08 and
decaying**; adding the congestion interaction (occupancy × tokens-in-net,
the obvious missing term — occupancy only costs when something is waiting)
made it **worse** (R² ≈ −0.10 to −0.25), i.e. pure overfitting. The floor
correctly rejected the model in essentially every epoch, so `cfpd` ran as
`cfpn`.

**Interpretation.** The opportunity cost on s1 is real in expectation
(X13 proved it: deleting it cost 8% of final performance) but is *not a
smooth function of coarse state-level occupancy*. What it actually depends
on is **which specific future case gets displaced** — a high-entropy
detail of the CRN-shared arrival stream, not summarizable by "how long was
the resource busy". So the systematic part that survives pooling is small
relative to case-level variation.

That has a real consequence worth stating in the write-up: **for the
indirect channel, simulation beats modelling.** cfpk wins precisely
because it estimates the total gap by direct experiment rather than
modelling any part of it — and this is the measurement showing the
modelled alternative fails, not an assumption.

**What survives.** Two things:
1. *The decomposition as a diagnostic, not an estimator.* The direct /
   indirect split is cheap and exactly measures how much of a decision's
   value the lineage can legitimately claim. That is the principled test
   for which regime an env is in — direct-dominated (grid, E1: lineage
   restriction is safe and buys 11% SE) vs indirect-heavy (s1: it is
   actively harmful). It turns the "lineage-restricted forks except in
   foreclosure-dominated envs" shipping rule from a heuristic into a
   measured decision.
2. *One remaining shot at a modelled correction:* condition with the GNN
   on the full marking instead of hand-built summaries. Higher capacity,
   and unlike LVA the target is a counterfactual difference rather than a
   lineage credit. The R² evidence here says coarse summaries fail; it
   does not settle whether a state-conditioned learner would. Given LVA's
   history this deserves scepticism and a strict floor.

## 7.3 What the thesis becomes

The general story is no longer "consume the provenance DAG better" — it
is: **a white-box executable model admits cheap exact interventions
(fork + CRN + coupling truncation), and interventional credit is what
observational provenance provably cannot deliver** (the cross/gen
inversion is the exhibit: a bias no re-weighting of factual lineage can
remove). The lineage keeps two general roles it has already earned:
variance filtering of the base estimator (lrq vs mc_q, 28W/0L — factual,
cheap, always on) and intervention targeting (where to fork). Lineage for
variance, interventions for bias — both purchased by the same white-box
premise the paper already assumes. Related-work note for the writeup:
this is difference rewards / COMA's counterfactual done *exactly* rather
than via a learned or aristocrat approximation, affordable because the
simulator is ours; the SNR-gated preference interface and the coupling
truncation appear to be the novel pieces — verify against the
counterfactual-credit-assignment literature before claiming.

### 7.4 Revised sequence

1. **G1 fork machinery + s1 probe** (the flagship bet).
2. **E1-family + grid generality runs** for G1 — the claim the objection
   demands: same code, no env-specific lines, wins on choice topologies.
3. **G2** on the same fork stream (s1 + E1; validates the lrq3
   rehabilitation).
4. **G3 anneal** regardless — paper insurance, one afternoon.
5. G5 only as a baseline row if G2 runs; G6 if finals still tie.
