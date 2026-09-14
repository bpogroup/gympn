# Paper outline — EJOR submission (v3, reframed 2026-08-12)

*Supersedes the `s_ccf` + `cfpk` outline in `EJOR_PAPER_OUTLINE.md` (v2,
2026-08-03), which is retained unmodified for salvage — §6b and most of §2 carry
over almost verbatim. Reason for the reframe: the 2026-08-12 cgae arc produced a
single method, `cgae_cflow`, that is (i) safe on every environment tried,
(ii) cheap with NO forking, and (iii) built directly on token lineage. v2's
headline pair required either a conservatism concession (`s_ccf`) or forked
simulation (`cfpk`); v3's does neither.*

> **⚠️ CLAIM RETIRED 2026-08-14.** The original fourth reason for the reframe was
> *"beats v2's hero `ccf` on the scaling benchmark (+0.199, p=0.0032, 20 paired
> seeds)."* **That was a 15-epoch artifact.** At the converged 40-epoch budget
> `ccf` gains +0.241 (0.756 → 0.997) and the comparison becomes **+0.003,
> 5W/5L, p=0.80**. `cgae_cflow` does NOT beat `ccf`, `cgae`, or `cgae_cap` at any
> converged budget. See §6.0 and §6a-40. The reframe still stands on (i)-(iii)
> plus the two effects that ARE significant at converged budgets — successor
> normalization and non-closure semantics (§6a-40) — but the "beats prior work"
> framing must not appear in the paper.*

*A second, unplanned contribution emerged and is now load-bearing: a four-rung
ablation ladder in which performance is almost UNCORRELATED with fidelity to the
causal estimand. That is a genuine finding about causal credit assignment, not a
gap, and it is what §5 and §6c are built around.*

*Working title (lead candidate):*
**"Causal-Order Advantage Estimation: Exploiting Token Lineage for Credit
Assignment in Reinforcement Learning for Dynamic Task Assignment"**

Alternatives:
- "Credit Along the Provenance DAG: Causal-Order GAE for Dynamic Task Assignment"
- "GAE on the Causal Graph: Lineage-Derived Advantage Estimation, and Why
  Faithfulness Is Not Enough"

---

## 0. Positioning

| prior work | contribution | learning engine |
|---|---|---|
| Berti et al. 2023 | A-E PN formalism; DTA taxonomy | generic RL |
| 2507.03579 | assignment-graph representation | generic PPO |
| GymPN 2506.20404 | library; partial observability | generic DRL |
| **this paper** | **run GAE along the token-provenance DAG instead of the trajectory index** | **`cgae_cflow` + an ablation ladder that prices every design choice** |

**Thesis.** Standard GAE accumulates TD errors along the *trajectory index*,
`A_t = δ_t + γλ·A_{t+1}`. In an A-E PN running interleaved cases, decision `t+1`
usually belongs to a *different case*, so `δ_{t+1}` is largely noise with respect
to the action at `t`: credit propagates along wall-clock order while causality
flows along the provenance DAG that the simulator already records for free. We
run the *identical* recursion along the DAG.

**The one-sentence method.** Replace GAE's temporal successor with the *causal*
successor; assign each reward to a single owner (the latest decision on its
lineage, making ownership a partition and the estimator a return decomposition
rather than a hindsight Q-sample); and combine successors by a **convex,
flow-weighted** average whose weights are the token-flow shares normalized over
successors.

**Why the convex normalization is the crux, and the paper's sharpest technical
point.** The natural flow-weighted form normalizes over *predecessors*
(`Σ_{d∈pred(s)} w(d→s) = 1`), which is what a conservation argument gives you.
But the recursion sums over *successors*, so the coefficient actually multiplying
the bootstrap is the **row** sum `R(d) = Σ_{s∈succ(d)} w(d→s)` — which the
predecessor constraint does not bound at all. `A[d]` carries `−V[d]` and picks up
`R(d)·ρ·V[s]`, so unless `R·ρ ≈ 1` the value terms fail to cancel and the
advantage retains a spurious term proportional to **V** whose size is set by
local DAG topology rather than by the action. Normalizing over successors fixes
it and makes the mean-over-successors variant an equal-share **special case**
rather than an underived rival.

**OR framing.** A decomposition method: we decompose the policy-gradient return
along the causal-component structure the stochastic system itself exposes
(conceptual home: Benders / Dantzig-Wolfe / Lagrangian decomposition).

**Scope (in/out).**
- IN: `cgae_cflow` as the method; the credit spectrum `mc_q → lrq → ccf → cgae`
  family as the ablation ladder; the bias-bounded theory (Prop 3 retained as the
  unbiased λ=1 anchor, realized exactly by `cgae_dag`); the boundary-condition
  investigation from v2 §6b, unchanged.
- OUT: `cfpk`/DAG-replay forking (demoted to related/future work — it is the
  *exact* end of the spectrum but costs forked simulation, and `cgae_cflow`
  reaches comparable or better performance at ~1 ms/episode); MCTS-over-A-E-PN;
  LS-HCA; LCV/LVA.

---

## 1. Introduction

- Dynamic task assignment: stochastic task streams onto limited contended
  resources. Ubiquitous in OR.
- A-E PN gives an executable model + DRL solver, but the DRL is generic PPO
  that ignores the net's causal structure.
- **The credit-assignment problem in concurrent systems.** Return-to-go mixes
  many cases' rewards; a decision's advantage carries the variance of every case
  in flight. Measured on `ncopies` N=4 at the CONVERGED 40-epoch budget (10
  paired CRN seeds): `cgae_cflow` **1.000 ± 0.024**, PPO **0.700 ± 0.309**
  (+0.300, t=0.0198) — a 13x tighter spread, and PPO collapses on a seed where
  `cgae_cflow` never does. At the shorter 15-epoch budget the same contrast is
  0.955 ± 0.131 vs 0.478 ± 0.418 with 0/20 vs 5/20 collapses. **See §6.0: the
  15-epoch ncopies protocol is NOT converged and must not be the primary
  report.**
- **Contributions.**
  1. **`cgae_cflow`**: GAE run along the provenance DAG rather than the
     trajectory, with single-owner reward attribution and convex flow-weighted
     successor aggregation. Derived from token lineage, no hand-specified
     factorization, no forked simulation, ~1 ms/episode (≤8% of a training cell;
     the GNN dominates).
  2. **Identification and repair of a normalization defect** that the natural
     formulation hides: predecessor-normalized flow weights leave the bootstrap
     coefficient unbounded, and this is *not* a theoretical nicety — it collapses
     the estimator on the one environment where it bites (§6c).
  3. **A bias-bounded guarantee** rather than an unbiasedness claim: the bias is
     a bounded shrinkage, exactly zero when every decision has causal
     out-degree at most one, in contrast to the
     unnormalized form whose error is unbounded in `R`. Proposition 3 is retained
     as the unbiased λ=1 anchor, realized exactly by `cgae_dag`.
  4. **An ablation ladder in which faithfulness to the causal estimand is nearly
     uncorrelated with learning performance** — four variants spanning P3-identity
     fidelity 0.577 → 1.000, whose performance ordering is unrelated to it. We
     price each design choice rather than assert it.
  5. **A boundary characterization** (carried over from v2): three independent
     "give the learner more structure" routes — potential-based shaping,
     structural input features, actor architecture — each provably inert or
     empirically neutral on a single-component bottleneck.

---

## 2. Related work

Carry over v2 §2 essentially unchanged, with these edits:

- **GAE** (Schulman et al. 2015) moves from a cited tool to a *central*
  reference: our method is GAE with the successor relation replaced, and — this
  matters for §5 — GAE with λ<1 is *itself* a biased estimator that is
  universally preferred to the unbiased λ=1 case. That is the precedent for our
  bias-bounded framing, and it should be stated in §2, not buried in discussion.
- **Biased-but-standard estimators**, new short paragraph establishing that
  deliberate bias is the norm, not an apology: PPO's clipped objective, V-trace's
  truncated importance weights, DQN reward clipping, VDN/QMIX value
  factorization, γ<1 as regularization, and James-Stein / ridge shrinkage from
  statistics (biased dominating unbiased in MSE is a theorem, not an anomaly).
- **Return decomposition / credit assignment**: RUDDER, HCA — unchanged.
- **Multi-agent factored credit**: difference rewards, COMA, VDN/QMIX. Keep v2's
  positioning: *these require the entity partition to be hand-specified; we
  derive it from provenance, per episode, automatically.*
- **Counterfactual/causal RL** (Mesnard et al.): now also the home for the
  demoted `cfpk` work — cite as the exact-but-costly end of the spectrum.
- **Decomposition in OR**; **Petri nets + learning**; **GNNs for CO** — unchanged.
- ⚠️ v2's readiness table records related work as verified-live. The **novelty
  claim** — automatic *discovery* of the credit factorization from provenance vs
  factored MDPs being handed the DBN — was flagged in v2 as "my reading, not a
  completed search." That is still open and blocks any novelty sentence.

---

## 3. Background

- **A-E PN** recap: places, colored tokens, action/evolution transitions, the
  induced SMDP, rewards.
- **Token provenance / lineage DAG**: for each token, which firing produced it
  from which inputs; how rewards attach. Recorded by the simulator for free.
- **PPO for A-E PN**: graph observation, SMDP discounting `e^{−βτ}`,
  return-to-go advantage and baseline.
- **GAE and its temporal successor assumption** — set up the substitution that
  §4 performs.

---

## 4. Method — `cgae_cflow`

### 4a. Causal successors
`d → s` when `s` consumes a token descending from an output token of `d`, with no
intervening decision (nearest producer only). Forced firings and evolution
transitions are transparent: the walk passes through to the decision that
actually caused the flow.

> ⚠️ **Known specification wrinkle to disclose, not hide.** Postpone is an agent
> decision, so it stops the walk — but under token-flow it re-emits the *whole*
> marking, so it is not "the decision that caused the flow." Making it
> transparent recovers real edges (0 lost, **+97/episode on s1**) and is arguably
> the correct DAG — yet it *degrades* performance sharply (§6c, `cgae_cflow2`,
> 0.106). Report as a measured boundary of the specification, with the negative
> result stated plainly.

### 4b. Single-owner reward attribution
Each reward goes to the latest decision on its lineage,
`own(j) = argmax_{d∈A(j)} u_d`. Ownership is a **partition** — every reward
counted exactly once — which is what makes this a return decomposition rather
than `lrq`'s hindsight Q-sample.

### 4c. Convex flow-weighted recursion
```
w(d→s)  = share of s's consumed tokens descending from d,   Σ_{d∈pred(s)} w = 1
ŵ(d→s)  = w(d→s) / R(d),        R(d) = Σ_{s∈succ(d)} w(d→s),   so Σ_s ŵ = 1
A[d]    = owned[d] − V[d] + Σ_s ŵ(d→s)·ρ(Δt_s)·( V[s] + λ·A[s] )
```
Emitted as `A[d] + V[d]`, consumed like every other scheme (`A_t = Q_t − V(s_t)`,
value head regressed on `Q_t` — so the critic converges to the same object the
recursion bootstraps through; the fixed point is self-consistent).

Properties to state here, all measured:
- **Chain-exact.** On a true chain `ŵ ≡ 1` and all variants coincide to floating
  point (102/102 ncopies, 33/33 s1 true-chain decisions).
- **The mean is a corollary.** Under equal inflow shares `ŵ = 1/k`, i.e. exactly
  the mean-over-successors variant (exactly so when successor discounts coincide;
  otherwise up to per-successor vs averaged discount folding). This dissolves the
  "ship the better number or the derived one" dilemma in v2's status doc §5.3.8.
- **SMDP discount at causal depth**, folded per successor.
- **Cost**: ~1 ms/episode; ≤8% of a training cell (PPO 7.44 min vs 8.05 min at
  N=4). No forking.
- Postpone handling: SMDP-TD mask (as `lrq2`/`ccf`/`s_ccf`).

### 4d. Algorithm box + the spectrum table
`mc_q` / `lrq` / `ccf` / `s_ccf` / `cgae` / `cgae_cflow` / `cgae_dag` across
{bias, variance, granularity, cost, needs forks}. The practitioner decision table.

---

## 5. Theory

**Reframe the whole section as a bias-variance frontier, not a hunt for
unbiasedness.** Write every variant in one common form:

```
Â_d = Σ_j  c_d(j) · ρ · owned[j]
```

where `c_d(j)` is the weight placed on descendant `j`'s owned reward. Then:

| variant | `c_d(j)` | measured (s1 / ncopies) | status |
|---|---|---|---|
| Prop 3(i) estimand | `1` on `desc(d)` | — | unbiased target |
| `cgae_dag` | `1` (closure) | **1.000 / 1.000** | **unbiased** given (A1) |
| `cgae_cflow` | `Σ_paths ∏ŵ ∈ [0,1]` | 0.626 / 0.905 | bounded shrinkage |
| `cgae` | `1/k` shrinkage | 0.602 / 0.887 | bounded shrinkage |
| `cgae_flow` | unnormalized + critic term | 0.577 / 0.886 | **unbounded** |

- **P1 (carried over).** Component mean-exogeneity (A1) justified *from A-E PN
  semantics*: a reward is a deterministic function of the tokens consumed at its
  firing, and every token's history is in the DAG.
- **P2 (carried over).** Variance reduction ~`K` under `K` comparable
  independent components; gradient SNR ~`√K`.
- **P3 (retained, retitled as the unbiased anchor).** Under (A1), the λ=1, `V≡0`
  causal-closure estimator is unbiased, and since `desc(d) ⊆ c(d)` every
  cross-component reward is dropped exactly as in `ccf`. **`cgae_dag` realizes
  this exactly** (identity ratio 1.000; reproduces textbook GAE on a chain to
  4e-16 for λ ∈ {1, .9, .5, 0}; counts a diamond's reward once, 7.0 not 14.0).
  Because `desc(d) ⊊ c(d)`, it is strictly *finer* than `ccf`.
- **P4 (PROVED 2026-08-14 — written into `paper/main_v3.tex` §5 as
  Lemma 1 + Proposition 3).** `cgae_cflow`'s policy-gradient bias is **bounded**.
  *Key lemma:* successor-normalized weights are row-stochastic, so they induce a
  Markov chain on the DAG; the DAG is acyclic (time-forward edges), so a walk
  visits each node at most once and distinct `d⇝e` paths are disjoint events.
  Hence **`c_d(e) = Pr[walk from d visits e] ∈ [0,1]`** — boundedness is a
  probability, not an estimate. Verified numerically: `c` never left `[0,1]` over
  ~55 root nodes, and Monte-Carlo visit frequencies match the closed form to
  ±0.016 at 4000 samples.

  > ⚠️ **"c ≡ 1 on a forest" was WRONG and is corrected.** An in-forest still
  > *splits* the walk wherever fan-out exceeds one, giving `c < 1`. The correct
  > equality condition is **out-degree ≤ 1 at every node of `desc*(d)`** (merging
  > is fine, splitting is not); more generally `δ_d = 0` iff `desc*(d)` is
  > totally ordered by reachability. This is confirmed on the boundary case:
  > multisite has out-degree exactly 1 everywhere and measures `c ≡ 1.0000`,
  > `δ = 0` — so **the headline realistic result is produced by an unbiased
  > estimator**. Where out-degree > 1 the deficiency appears and grows with
  > fan-out (mean `1−c̄` = 0.050 at N=4, 0.285 on s1; worst-case `δ` = 0.783 and
  > 0.939 respectively).

  Contrast `cgae_flow`,
  whose error term `(R·ρ − 1)·V` multiplies the *critic* with `R` measured over
  `[0.083, 3.667]` — unbounded, additive, and topology-driven. **This bounded/
  unbounded distinction is the paper's formal content for "safe everywhere."**
- **Honest statement of what is NOT claimed.** `cgae_cflow` is biased: `c_d(j)`
  depends on the realized DAG between `d` and `j`, which depends on the action at
  `d`. We verified this rather than assuming it — forking every action at a
  decision state under common random numbers, `R` is identical across actions at
  only **18.8%** (ncopies) / **6.2%** (s1) of fork points. So the shrinkage does
  *not* factor out through the score-function identity, and the per-state
  rescaling argument is unavailable. Say so in the paper.

> ⚠️ **P4 is not yet proved.** This is the critical path for the theory section.

---

## 5bis. WHY IT WORKS — the mechanism, derived and measured

*Added 2026-08-13. The empirical claims are settled; this section is the causal
account of them. Two distinct phenomena need explaining and they have different
explanations, which is why conflating them produced four falsified hypotheses
before this framing.*

### 5bis.0 Canonical form

Every variant computes the same shape of object. Writing `desc*(d)` for the
causal descendant closure of `d` including `d`:

```
Q[d] = Σ_{j ∈ desc*(d)} c_d(j) · ρ(t_j − t_d) · owned[j]   +   B[d]
A_d  = Q[d] − V(s_d)
```

with `c_d(j)` the total path weight the variant places on descendant `j`, and
`B[d]` the critic-bootstrap term. The variants differ **only** in `c` and `B`:

| variant | `c_d(j)` | bootstrap coefficient `Σ_s c(d→s)` |
|---|---|---|
| `cgae_dag` | `1` (closure) | convex (1) |
| `cgae_cflow` | `Σ_paths ∏ŵ ∈ [0,1]` | **exactly 1 by construction** |
| `cgae` | `Σ_paths ∏(1/k)` | 1 (mean) |
| `cgae_flow` | `Σ_paths ∏w` | **`R(d)` — unbounded** |

Two questions follow, and they are independent:

- **Q-A.** Why does causal-order GAE beat PPO? → governed by **K**.
- **Q-B.** Why does the convex normalization matter? → governed by **fan-out**.

### 5bis.1 Q-A — variance reduction, governed by K

Under (A1) the `K` reward-bearing components are mean-exogenous to one another.
PPO's advantage carries the full return-to-go `G = Σ_{k=1..K} G_k`; with
comparable components, `Var(G) = K σ²`. Component-scoped credit carries only
`G_{c(d)}`, so `Var = σ²`. **Variance ratio `K`, gradient SNR ratio `√K`**
(Proposition 2).

The scoping ratio `S(d) = |desc(d)| / |decisions after d|` makes this concrete
and is measured (`_diag_scope_ratio.py`). If components are balanced, `S ≈ 1/K`.
Measured on `ncopies`, where `K = N` by construction:

| N | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| `S` measured | **1.000** | 0.472 | 0.221 | 0.114 |
| `1/N` | 1.000 | 0.500 | 0.250 | 0.125 |

`S = 1.000` exactly at `N=1` — the estimator provably does no scoping, which is
the correctness control (§6a Q1), and indeed every method ties there (0.726).

**The gap over PPO is monotone in K across five configurations:**

| env | K | gap vs PPO |
|---|---|---|
| ncopies N=1 | 1.00 | 0.000 |
| s1 | 1.00 | +0.124 (n.s., p=0.373) |
| ncopies N=2 | 2.00 | +0.371 |
| ncopies N=4 | 3.33 | +0.477 |
| **multisite** | **8.00** | **+0.564** (12W/0L, p=0.0001) |

**Critical subtlety, and the reason `S` alone is not the story.** s1 has
`S = 0.474` — nearly identical to `ncopies` N=2's `0.472` — yet its gap is
+0.124 (n.s.) against N=2's +0.371. Scoping away half the timeline buys nothing
on s1 because `K = 1`: what is removed lies in the *same* component and is
therefore statistically dependent on what is kept. **Variance reduction requires
that the scoped-away mass be independent, which is precisely (A1), and `K`
counts exactly that.** `S` measures how much of the timeline is excluded; `K`
measures how much of it was *independent*. Only the latter converts into
variance reduction.

**The scoping is directly visible as credit mass — and it CUTS credit rather
than inflating it.** A natural referee question (and a natural first reaction to
the unrolled form) is that crediting every causal ancestor must massively
over-count: summing the estimator's own output over all decisions gives

```
Σ_d Q[d] = Σ_j owned[j] · ( Σ_{d ∈ anc(j)} c_d(j) )   ≫  G = Σ_j owned[j]
```

That is true, but it is **not** a property of causal credit — it is what
return-to-go already does, and far more so. Under plain SMDP return-to-go a
reward at decision `k` enters the advantage of every one of the `k` decisions
before it, so `Σ_t G_t ≈ (n/2)·G`. Measured (λ=1, V≡0; `mc_q` *is* the
return-to-go baseline PPO uses):

| env | decisions `n` | `mc_q` | `cgae_flow` | `cgae_cflow` | reduction |
|---|---|---|---|---|---|
| multisite | 69 | **39.78** | 4.34 | 4.34 | **9.2x** |
| ncopies N=4 | 36 | **21.23** | 4.21 | 4.32 | **4.9x** |
| s1 | 29 | **18.40** | 4.74 | 4.86 | 3.8x |

`mc_q` lands on `n/2` as predicted (39.78 vs 34.5; 18.40 vs 14.5). The causal
schemes cut cross-decision credit mass by **4-9x**, i.e. they remove 80-90% of
the credit plain PPO assigns — and the size of the cut tracks `1/S` closely
(predicted 9.4 / 4.5 / 2.1 against measured 9.2 / 4.9 / 3.8). **The residual
~4-5x is the variance-reduction mechanism seen from the other side:** what
survives after component-scoping strips out everything a decision did not
cause. The largest cut is on multisite (K=8), which is also the largest win.

Two points close the question for good. Absolute scale never reaches the
gradient: per-batch advantage normalization is on by default
(`train.py:344`, applied at `agents.py:691`), standardizing advantages to
zero-mean/unit-variance before each policy update. And even unnormalized, the
schemes *reduce* rather than inflate relative to the PPO baseline they are
compared against.

### 5bis.2 Q-B — TWO conservation conditions, and why each variant breaks one

> **Superseded framing (2026-08-13).** This subsection originally treated the
> defect as one-sided — fan-out inflating the critic — and concluded the convex
> normalization was the fix. That was **half the story**, and the missing half
> is what the ncopies N=2 result exposed (`cgae_cflow` 0.401 vs `cgae_flow`
> 0.837, 4W/14L, p=0.0043, at a lower fan-out than N=4 where cflow wins). The
> corrected account is below; the identity (★) that follows is still correct,
> it just does not exhaust the problem.

Every variant is a choice of edge coefficient `c(d→s)`, and credit must be
conserved on **both** sides of the DAG:

```
(P)  Σ_{d ∈ pred(s)} c(d→s) ≤ 1     else the SAME continuation is claimed in
                                     full by several joint causes
(S)  Σ_{s ∈ succ(d)} c(d→s) ≤ 1     else the critic V is re-counted across
                                     successors, and the recursion stops being
                                     a contraction
```

Neither existing variant satisfies both, and they fail on opposite sides:

| variant | `c(d→s)` | (P) predecessor | (S) successor |
|---|---|---|---|
| `cgae_flow` | `w` | **= 1 exactly** ✓ (this is Prop 3's (\*)) | **unbounded** ✗ |
| `cgae_cflow` | `w/R` | **unbounded** ✗ | **= 1 exactly** ✓ |
| **`cgae_cap`** | `w/max(R,1)` | **≤ 1** ✓ | **≤ 1** ✓ |

Measured (4 rollouts/env; % of nodes violating the condition):

| env | (P) violated: flow / cflow / **cap** | (S) violated: flow / cflow / **cap** |
|---|---|---|
| multisite | 0% / 0% / **0%** | 0% / 0% / **0%** |
| ncopies N=2 | 0% / **10%** / **0%** | **9%** / 0% / **0%** |
| ncopies N=4 | 0% / **13%** / **0%** | **6%** / 0% / **0%** |
| s1 | 0% / **32%** / **0%** | **24%** / 0% / **0%** |

**The intuition, on the smallest case.** Let `s` be jointly caused by `d` and
`e`, each contributing half of `s`'s consumed tokens, so `w = 0.5` on both
edges. `cgae_flow` gives each predecessor half the continuation — total 1,
conserved. `cgae_cflow` has a single successor at each of `d` and `e`, so
`R = w = 0.5` and `ŵ = w/R = 1`: it hands **both** `d` and `e` the **entire**
continuation. The same future value is counted once per cause. That is the
exact mirror of flow's fan-out defect, and it is why the convex form fails
where fan-IN dominates (N=2) while flow fails where fan-OUT dominates (s1).

`cgae_cap` divides by `max(R,1)` — it only ever divides, never multiplies up —
so `Σ_succ c = min(R,1) ≤ 1` and `Σ_pred c = Σ_d w/max(R_d,1) ≤ Σ_d w = 1`. It
is the unique member of the family satisfying both sums.

> **⚠️ CORRECTION (2026-08-13, later same day). (P) and (S) are NOT two
> instances of one principle, and "cap is doubly conserved therefore most
> principled" does not survive scrutiny. Do not put that claim in the paper.**
>
> **(S) is a WITHIN-decision consistency condition.** `V[s]` is a global state
> value, so `Σ_succ c > 1` makes a *single* `A_d` add the same state's value
> more than once. That is an internal inconsistency in one advantage —
> provable (3-node fixture: 6.74 vs 3.37) and catastrophic (s1: 0.166).
>
> **(P) is a CHOICE OF ESTIMAND, not a correctness condition.** It only fixes
> how `s`'s continuation is divided among `s`'s immediate causes:
> `Σ_pred c = 1` (flow) is *share* semantics; `Σ_pred c > 1` (cflow) is
> *but-for* semantics. Nothing in the policy-gradient argument requires
> advantages across DIFFERENT decisions to sum to anything. And in an A-E PN a
> jointly-caused `s` consumes tokens from every predecessor, so all of them are
> NECESSARY — crediting each with the full continuation is exactly the but-for
> counterfactual that difference-rewards and COMA use.
>
> The argument that killed the symmetric framing: measured `Σ_d Q[d] / G` with
> λ=1, V≡0 (G = episode return) is **~4-5x for EVERY variant** — multisite 4.34
> for all four; s1 flow 4.74, cflow 4.86, cap 3.17, dag 9.86. The inflation
> comes from crediting every causal ancestor, which is inherent to lineage
> credit assignment, and flow-vs-cflow differ by only 2.5%. So (P) barely moves
> the global credit mass and cannot carry a correctness argument.
>
> Consequence: **`cgae_cflow` is principled in the sense that matters**, and
> `cgae_cap`'s extra shrinkage discards credit from necessary causes — visible
> as the systematically lowest credit mass (3.17 on s1) and, empirically, as
> higher variance and reintroduced collapses at N=4 (2/20 vs cflow's 0/20).
> Ship cflow; keep cap as the ablation that PRICES the share-vs-but-for choice.

The one-sided identity below remains valid and is the (S) half of the story.

At any decision `d` with successors, the two flow variants differ by exactly:

```
A_flow[d] − A_cflow[d] = (R(d) − 1) · Σ_s ŵ(d→s)·ρ_s·( V[s] + λ·A[s] )
                       = (R(d) − 1) · Ê[ ρ·(V + λA) ]                    (★)
```

because `w(d→s) = R(d)·ŵ(d→s)` by definition of the normalization. The bracket
is the estimator's own bootstrapped continuation value — call it `Ĝ(d)`.

**Three consequences, each checkable:**

**(i) Magnitude.** `Ĝ(d) = O(V) = O(H·r̄)` where `H` is the remaining causal
horizon and `r̄` the mean per-decision reward, whereas the *signal* the advantage
must carry is the action-attributable `owned[d] = O(r̄)`. Hence

```
|error| / |signal|  ≈  |R(d) − 1| · O(H)
```

The error scales with the **horizon**; the signal does not. So even a modest
`|R−1|` swamps the signal on long episodes. This is why the defect is fatal
rather than merely inaccurate.

**(ii) It does not cancel as a baseline.** `R(d)` depends on the realized causal
DAG downstream of `d`, which depends on the action taken at `d`. Verified rather
than assumed by forking every action at a decision state under common random
numbers: `R` is identical across actions at only **18.8%** (ncopies) and
**6.2%** (s1) of fork points. An action-dependent term does not factor out
through the score-function identity.

**(iii) `cgae_cflow` sets the term to exactly zero.** `Σ_s ŵ = 1` by
construction, so `R ≡ 1` and (★) vanishes identically — not approximately, not
in expectation. `cgae` (mean) also has coefficient 1 and is likewise protected;
`cgae_flow` alone is exposed.

**The identity predicts the observed dose-response, quantitatively:**

| env | fan-out | share with `abs(R−1) > 0.25` | predicted | observed |
|---|---|---|---|---|
| multisite | **1.000** | **0%** | variants identical | max abs diff = **0.0**, all 0.710 |
| ncopies N=4 | 1.137 | 24.1% | flow lags mildly | 0.888 vs 0.955 |
| s1 | 1.387 | **55.3%** | flow collapses | **0.166** vs 0.792 |

and it is confirmed **deterministically** on a 3-node fixture where `R = 2`
exactly: `cgae_flow` returns **6.74** against `cgae`/`cgae_cflow`'s **3.37** —
the critic entered exactly twice, checkable by hand (`_test_cgae_cflow.py` C3).

### 5bis.3 The bias of `cgae_cflow`, quantified — and it is zero where it matters

The shrinkage factor is the mean retained path mass
`C = (1/|desc(d)|) Σ_j c_d(j)`, measured directly:

| env | multisite | ncopies N=2 | N=4 | N=8 | s2 | s1 | s4 |
|---|---|---|---|---|---|---|---|
| `C` | **1.000** | 0.991 | 0.950 | 0.944 | 0.792 | 0.715 | 0.504 |

**On multisite, `C = 1.000`: `cgae_cflow` coincides with the unbiased
closure estimand exactly.** Fan-out is 1, so every path weight is 1 and the
convex form *is* Proposition 3(i)'s estimator. The headline realistic result is
therefore produced by an **unbiased** estimator, and the bias only appears where
fan-out does — bounded by `C ∈ [0,1]`, never unbounded. That is the substance of
P4 and it is now measured as well as argued.

### 5bis.4 The two diagnostics are orthogonal

`K` and fan-out answer different questions, and the environments populate
distinct quadrants (`_diag_fanout_table.py`, 19 environments, no training):

| | fan-out ≈ 1 | fan-out > 1 |
|---|---|---|
| **K > 1** | **multisite** (K=8, fan 1.000) — method wins; variant choice *provably irrelevant* | `ncopies` N=2-8 (K 2-6.3, fan 1.02-1.16) — method wins; variant choice minor |
| **K = 1** | *(empty)* | **s1 + 10 archetypes** (fan 1.19-1.80) — method cannot help; variant choice decides whether it *harms* |

This is a mechanism that predicts its own relevance *and its own irrelevance* —
the same evidential pattern as the K-diagnostic, and the strongest form of
argument available here. It also dissolves the "cgae_cflow merely ties cgae"
objection: on two of three environments they are provably the same algorithm, so
the tie is a **theorem**, not a disappointment.

### 5bis.5 Ablations — each design choice, priced

`cgae_cflow` makes exactly four design choices. Each one has a variant that
removes it, so each is *priced* rather than asserted. This is the section that
answers a referee's "why this and not the obvious simpler thing," and every row
is a measurement, not an argument.

| design choice | ablation | cost of removing it (s1 → ncopies N=4) | verdict |
|---|---|---|---|
| run GAE along the **causal DAG** | `mc_q` (no lineage) | multisite 0.710 → **0.036** | **essential** |
| **convex** successor normalization | `cgae_flow` (predecessor-normalized) | 0.792 → **0.166** (0W/5L vs ppo, p=0.025) | **essential where fan-out > 0** |
| **flow** weights vs uniform | `cgae` (mean) | 0.792 → 0.650; N=4 identical (p=0.691) | subsumed: `cgae` is the equal-share case |
| **bounded-bias** vs exact estimand | `cgae_dag` (closure, unbiased) | 0.792 → **0.386** | bounded bias beats exactness |

Two of these are the paper's load-bearing results:

**The normalization ablation is a positive result, not a failure.** `cgae_flow`
is the natural formulation — it is what a conservation argument hands you — and
it is a *statistically significant regression* against PPO (0W/5L, p=0.025).
§5bis.2 explains exactly why, algebraically and deterministically. The pairing
of a sharp failure with a closed-form account of it is stronger evidence for the
convex form than any amount of additional winning.

**The exactness ablation answers the question the theory section must face.**
We ship a biased estimator; the immediate objection is "why not the unbiased
one?" `cgae_dag` *is* the unbiased one (P3 identity ratio exactly 1.000) and it
scores 0.386 against `cgae_cflow`'s 0.792. Bounded bias is therefore a measured
design decision, not a concession. Structural account: on K=1 environments the
closure is large (`S = 0.474` on s1, **0.712** on s3), so the exact estimand
approaches the *unscoped* return — it degenerates toward `mc_q`, the lineage
ablation. Exactness on a single-component problem buys no scoping.

> **Honest scope of the account.** The `cgae_dag` mechanism above is consistent
> with the estimand algebra but was not isolated experimentally: four candidate
> mechanisms (postpone action-dependence, DAG sparsity, per-state ranking
> fidelity, credit SNR) were tested and none predicted the ordering — two came
> out inverted. Appendix E documents them so the account is auditable. The one
> variable that cleanly separates the working from the failing arms is final
> policy entropy (0.27-0.56 vs 0.13-0.15, no overlap); cause vs symptom is
> untested. Note all four probes were *static, random-policy* measurements while
> training uses a progressively deterministic policy — plausibly the wrong
> regime, which is itself worth a sentence in the evaluation-methodology
> discussion.

*(`cgae_cflow2` — postpone made transparent in the DAG — is not an ablation of
any design choice above but a separate DAG-construction variant. It recovers
~97 masked causal edges per s1 episode and scores 0.106, i.e. the more faithful
graph is the worse learner. One footnote; the code and gate are retained.)*

---

## 6. Experiments

> ### ⚠️ 6.0 TRAINING-BUDGET AUDIT — read before any table below
>
> **The ncopies protocol used 15 epochs, and at 15 epochs the policy has barely
> trained.** Final entropy 0.91-0.94 (from 1.00) and only 19-38% of the
> random→heuristic headroom captured. That is not a converged comparison, and it
> produced a **significant, reproducible, entirely spurious result**: at N=2,
> `cgae_cflow` measured 0.401 vs `cgae_flow`'s 0.837 (4W/14L, **p=0.0043**). At
> 40 epochs the same comparison is **+0.102 (5W/2L, p=0.51)** — sign flipped,
> significance gone.
>
> | env | epochs used | final entropy | converged? |
> |---|---|---|---|
> | multisite | **40** | — | ✓ |
> | s1 | 30 | 0.15-0.56 | ✓ genuinely trained |
> | ncopies N=2 | **15** | **0.92** | ✗ barely moved |
> | ncopies N=4 | **15** | **0.91-0.94** | ✗ barely moved |
>
> **Report 40-epoch numbers as primary for ncopies.** The 15-epoch cells are
> retained as the budget-sensitivity ablation, not as results. A referee running
> a longer budget would otherwise find the N=2 reversal themselves.
>
> Three independent lines confirm the 15-epoch N=2 effect was an artifact:
> (i) the λ sweep — `cgae_cflow` at N=2/15ep scores 0.401 → 0.579 → 0.639 as
> λ drops 0.95 → 0.85 → 0.70 toward `cgae_flow`'s *effective* λ of
> 0.95×0.896 ≈ 0.851, i.e. it was a convergence-SPEED effect;
> (ii) credit vectors are near-identical there (mean|Q| 2.22 vs 2.23, SD 1.12 vs
> 1.10), so a ~1% perturbation was swinging the metric by 0.44;
> (iii) outcomes were bimodal (~0.90 or ~0.13/−0.72, nothing between) — a
> bifurcation, not a quality difference.

### 6a-40. RESULTS AT THE CONVERGED BUDGET — PRIMARY (all four environments)

*Complete as of 2026-08-14. multisite already used 40 epochs; ncopies N=2/N=4 and
s1 were re-run at 40. N=8 in flight. Every number below is 40-epoch except where
marked.*

**Full converged table.** Normalized `(score − random)/(heuristic − random)`,
common random numbers throughout:

| env | K | `cgae_cflow` | `cgae` | `ccf` | `cgae_cap` | `cgae_flow` | `cfgae` | ppo | seeds |
|---|---|---|---|---|---|---|---|---|---|
| multisite | 8.0 | **0.710** | 0.710 | 0.518 | — | 0.710 | — | 0.146 | 12 |
| ncopies N=4 | 3.3 | **1.000** | 0.999 | 0.997 | 0.984 | 0.984 | 0.570 | 0.700 | 10 |
| ncopies N=2 | 2.0 | **0.851** | 0.755 | 0.840 | 0.849 | 0.749 | 0.842 | 0.682 | 10 |
| s1 (best-ckpt) | 1.0 | 0.847 | 0.808 | — | 0.855 | **0.402** | 0.879 | 0.814 | 5 |

(multisite: fan-out is exactly 1.000, so `cgae_cflow` = `cgae` = `cgae_flow`
**bit-identical** — verified, max abs diff 0.0. s1 is reported on
`greedy_best` because `greedy_final` there is drift-contaminated; see below.)

**What is significant, and what is not:**

| claim | evidence | verdict |
|---|---|---|
| causal-order GAE ≫ PPO | multisite **+0.564, 12W/0L, p=0.0001**; N=4 **+0.300, p=0.0198** | **ESTABLISHED** |
| ≫ `cfgae` | N=4 **+0.430, 8W/1L, p=0.011** | **ESTABLISHED** |
| successor normalization is essential | s1 vs `cgae_flow` **+0.445, 5W/0L, p=0.0070** (best-ckpt) | **ESTABLISHED** |
| closure semantics are harmful | s1 vs `cgae_dag` **+0.390, 4W/0L, p=0.033** | **ESTABLISHED** |
| beats `cgae` / `ccf` / `cgae_cap` | N=4 +0.001 / +0.003 / +0.016, all p > 0.07; s1 all p > 0.45; N=2 all p > 0.92 | **NULL — do not claim** |
| null where K=1 (negative control) | s1 vs ppo +0.032, p=0.63 | **AS DESIGNED** |

**`greedy_final` vs `greedy_best` on s1 — use best.** Every arm's `greedy_best`
is stable across the 30→40-epoch change while `greedy_final` moves:
`cgae_cflow` 0.857→0.847, `cgae_flow` 0.406→0.402, ppo 0.839→0.814. The apparent
40-epoch decline in `greedy_final` (`cgae_cflow` 0.792→0.737) is **post-peak
drift on the last eval point, not degraded learning** — s1 is documented as
drift-prone and `greedy_final` as the metric most contaminated by scenario noise,
which is why best-checkpoint restore exists. Report best-checkpoint for s1.

**`cgae_flow` DIVERGES with longer training — the sharpest single result.**
On s1, 30→40 epochs: mean 0.166 → **−0.077**, SD 0.168 → **1.063** (6×), mean
drift 1.18 → 2.36. One seed's greedy curve climbs to 10.3, holds ~6 eval points,
then falls to **0.15** (normalized −1.97) while entropy stays flat at 0.18 — a
confident collapse, not an exploration failure. This is exactly what §5bis.2
predicts: the error term `(R−1)·Ê[ρ(V+λA)]` is proportional to the **critic**, so
as `V` improves the spurious topology-driven term grows with it. **The
unnormalized estimator gets worse the better the critic becomes.** Note the
best-checkpoint value is unchanged (0.406→0.402), so this is a training-time
divergence, not a different optimum.

### 6a-40-OLD. Earlier partial 40-epoch snapshot (superseded by the table above)

**ncopies N=4, 10 paired CRN seeds, 40 epochs:**

| arm | normalized | SD | collapses |
|---|---|---|---|
| **`cgae_cflow`** | **1.000** | **0.024** | **0/10** |
| `cgae_flow` | 0.984 | 0.033 | 0/10 |
| ppo | 0.700 | 0.309 | 1/10 |

`cgae_cflow` vs ppo **+0.300 (7W/3L, t=0.0198)**; vs `cgae_flow` +0.016
(7W/3L, p=0.201 — a tie). `cgae_cflow` reaches **exactly heuristic level
(1.000)** with an order-of-magnitude tighter spread than PPO (0.024 vs 0.309).

**ncopies N=2, 10 paired CRN seeds, 40 epochs:**

| arm | normalized | SD | collapses |
|---|---|---|---|
| **`cgae_cflow`** | **0.851** | 0.238 | 1/10 |
| `cgae_cap` | 0.849 | 0.240 | 1/10 |
| `cgae_flow` | 0.749 | 0.339 | 2/10 |

**Budget sensitivity, 15 → 40 epochs** (the table that justifies §6.0):

| arm | N=2 @15 | N=2 @40 | N=4 @15 | N=4 @40 |
|---|---|---|---|---|
| `cgae_cflow` | 0.401 | **0.851** (+0.450) | 0.955 | **1.000** (+0.045) |
| `cgae_flow` | 0.837 | 0.749 (−0.088) | 0.888 | 0.984 (+0.096) |
| ppo | 0.261 | — | 0.477 | 0.700 (+0.223) |

The N=4 **ordering is preserved** under the longer budget (cflow > flow > ppo);
the N=2 ordering **reverses**. Remaining arms (ccf, cgae, cfgae, cgae_cap) are
in flight at 40 epochs for both N; s1 (30 ep, entropy 0.15-0.56) and multisite
(40 ep) are already at converged budgets and need no re-run.

### 6a. Core result at the ORIGINAL 15-epoch budget — RETAINED AS ABLATION
`ncopies` N=4, **20 paired seeds, common random numbers** (`eval_seed=555000`),
normalized `(score − random)/(heuristic − random)`:

| arm | normalized | SD | collapses (≤0.25) |
|---|---|---|---|
| **`cgae_cflow`** | **0.955** | 0.131 | **0/20** |
| `cgae` | 0.936 | 0.158 | 0/20 |
| `cgae_flow` | 0.888 | 0.195 | 0/20 |
| `ccf` | 0.756 | 0.326 | 1/20 |
| `cfgae` | 0.515 | 0.552 | 6/20 |
| `ppo` | 0.478 | 0.418 | 5/20 |

Paired vs `cgae_cflow`: ppo **+0.477** (16W/4L, t=0.0002, Wilcoxon 0.0003);
`cfgae` **+0.439** (17W/2L, t=0.0022); `ccf` **+0.199** (14W/5L, t=0.0032) —
*the new method beats v2's hero family*; `cgae_flow` +0.067 (p=0.179);
`cgae` +0.019 (10W/9L, **p=0.691 — a dead heat, state it as such**).

**Wording discipline.** The claim is *safety*, not superiority: `cgae_cflow` is
the only variant that never collapses **and** never regresses, and it dominates
PPO/`cfgae`/`ccf` significantly. It does **not** outperform `cgae` — and `cgae`
is the mean variant that `cgae_cflow` generalizes, so this is a unification, not
a defeat.

### 6b. Negative control — the single-component environment
`s1_stoch_sequence`, 5 CRN seeds (random 9.85, heuristic 14.775):

| arm | normalized | vs ppo |
|---|---|---|
| **`cgae_cflow`** | **0.792** | +0.61, 2W/3L, **p=0.373 — null** |
| `cfgae` | 0.701 | p=0.859 |
| ppo_clip | 0.668 | — |
| `cgae` | 0.650 | p=0.864 |

The null **is** the result: "costs nothing where there is no structure."
Mechanism, measured *before* the outcome: K = **1.0** reward-bearing component
holding **100%** of reward mass on s1 vs K = **3.2** and 46.3% on ncopies. A
mechanism that predicts its own failure case is the strongest evidence in the
paper — keep v2's framing of this.

### 6c. The ablation ladder — pricing every design choice (NEW, load-bearing)

> **⚠️ UPDATED 2026-08-14 to the converged budget. The ladder SURVIVES, but it
> narrows: at 40 epochs only TWO rungs are significant.** s1 best-checkpoint,
> 5 CRN seeds, paired vs `cgae_cflow`:
>
> | rung / design choice | ablation | s1 best-ckpt | paired | verdict |
> |---|---|---|---|---|
> | successor normalization bounded | `cgae_flow` | **0.402** | **+0.445, 5W/0L, p=0.0070** | **ESSENTIAL** |
> | path- not closure-semantics | `cgae_dag` | **0.457** | **+0.390, 4W/0L, p=0.033** | **ESSENTIAL** |
> | flow- vs uniform weights | `cgae` | 0.808 | +0.039, p=0.63 | free choice |
> | share- vs but-for predecessors | `cgae_cap` | 0.855 | −0.008, p=0.53 | free choice |
> | causal DAG at all | `mc_q` | multisite 0.036 | vs 0.710 | **ESSENTIAL** |
> | postpone-transparent DAG | `cgae_cflow2` | 0.106 (30 ep) | — | harmful, footnote |
>
> **The headline finding is unchanged and now better supported:** performance is
> nearly uncorrelated with fidelity to the causal estimand. `cgae_dag` has P3
> fidelity **1.000** (exactly unbiased) and is the **second-worst** arm;
> `cgae_cflow` has 0.626 and is at the top. Faithfulness to the estimand is not
> the design objective — **bounded error is.**

Original 30-epoch version of the ladder, retained for the reasoning trail —
on s1, ordered by performance, with P3-identity fidelity alongside:

| variant | design choice | P3 fidelity | s1 norm |
|---|---|---|---|
| `cgae_cflow` | convex flow weights | 0.626 | **0.792** |
| `cgae` | mean over successors | 0.602 | 0.650 |
| `cgae_dag` | exact closure semantics | **1.000** | 0.386 |
| `cgae_flow` | predecessor-normalized sum | 0.577 | 0.166 |
| `cgae_cflow2` | + postpone-transparent DAG | 0.626 | 0.106 |

Three findings, each independently interesting:

1. **The normalization defect is real and severe.** `cgae_flow` vs ppo is a
   *statistically significant regression* (−2.47, **0W/5L, p=0.025**), not noise.
   `|R−1| > 0.25` on **55.3%** of s1 decisions vs **24.1%** on ncopies — which is
   exactly why it merely lags on the benchmark (0.888) and collapses on the
   control (0.166). Include the 3-node fixture: at a fan-out node with `R=2` the
   unnormalized form returns **6.74** against the convex form's **3.37** — the
   critic entered twice, checkable by hand.
2. **Exactness does not pay.** `cgae_dag` satisfies the estimand exactly and
   comes *fourth of five*. On a one-blob environment `desc(d)` ≈ the whole
   episode return, so the unbiased estimand barely varies with the action.
3. **Neither does a more faithful DAG.** `cgae_cflow2` recovers 97 real causal
   edges per episode that postpone was masking (0 lost, 97 gained) and is the
   **worst** arm (0.106). The mis-specification was acting as regularization.

**Synthesis, and the section's headline:** across the ladder, performance is
almost uncorrelated with fidelity to the causal estimand. What separates the
working from the failing arms is whether the error is *bounded*.

> ⚠️ **Report the failed explanations too.** We tested four candidate mechanisms
> for the ordering and falsified all four: postpone-driven action-dependence
> (worse after fixing), DAG sparsity (no ordering), per-state ranking fidelity
> against a CRN-measured counterfactual (**inverted** — `cgae_dag` ρ=+0.681,
> top-1 83.3% vs `cgae_cflow` ρ=+0.435, top-1 33.3%), and credit SNR
> (**inverted** — `cgae_dag` highest). The only variable that separates the field
> cleanly is **final policy entropy**: 0.27–0.56 for the four working arms vs
> **0.13–0.15** for the three failing ones, no overlap. Whether that is cause or
> symptom is untested (the decisive test is an `ent_bonus` intervention). Stating
> this honestly is far stronger than a mechanism we cannot support — and all four
> falsifications used static, random-policy probes, which may simply be the wrong
> regime.

Also report: `cgae_cflow` is the **only arm with zero greedy drift** (peak =
final) and the highest on-policy sampled peak (13.57).

### 6d. Boundary conditions — carry over v2 §6b unchanged
Reward shaping (theorem-safe, empirically neutral-to-harmful), structural input
features (provably redundant via the one-hot argument), actor architecture
(a proven graph-actor blind spot, fixed, neutral on s1 because of an accidental
3-hop backdoor). Synthesis: only *active-intervention* mechanisms have ever moved
the single-component floor.

### 6e. Protocol
CRN throughout (`eval_seed`); noise floor measured on s1 at **±0.231 SD** per
20-episode eval point, **±0.327** on a paired difference, with `greedy_drift`
inflated by ~0.40 even for a flat policy. ≥5 seeds on s1, 20 on ncopies; paired
t-test and Wilcoxon; normalized return; collapse counts; wall-clock.

---

## 7. Discussion

- **Practitioner decision rule.** Does the problem decompose, even partially,
  into weakly-coupled sub-streams? Then `cgae_cflow` gives real, cheap, drop-in
  gains with no hand-specified structure and no forking. If it is a genuine
  single-bottleneck contention problem, expect a null — and that is *safe*, not
  harmful, which is the property the ladder establishes.
- **The K-diagnostic** (keep from v2 §7.1): measure reward-bearing components
  before adopting; K ≈ 1 predicts the null.
- **What the ladder teaches.** Fidelity to a causal estimand is not the design
  objective; bounded error is. GAE's own λ<1 is the precedent.
- **Limitations, stated plainly.** Two environments; `cgae_cflow` is biased and
  we verified the action-dependence rather than assuming it away; the mechanism
  behind the ladder ordering is unexplained after four falsified hypotheses; P4
  is unproved; the multi-site credibility anchor has not yet been run with
  `cgae_cflow` (see critical path).
- **Future work.** `cfpk`/DAG-replay as the exact end of the spectrum; the
  entropy intervention; extension to any provenance-emitting simulator.

## 8. Conclusion
A-E PN provenance turns hand-specified factored credit into a *derived* quantity.
Running GAE along that DAG, with convex flow-weighted aggregation, gives a cheap,
fork-free, bounded-bias policy gradient that helps where causal structure exists
and is provably safe where it does not.

---

## Appendices
- A: proofs (P1, P2, P3, **P4**).
- B: hyperparameters, reproducibility (`gympn.seed_everything`).
- C: estimator definitions for the whole spectrum.
- D: boundary-condition technical detail (from v2).
- E: the falsified-mechanism appendix — the four negative diagnostics, so the
  ladder's synthesis is auditable rather than asserted.

---

## Readiness — what carries over, what is new, what is missing

| item | status |
|---|---|
| `cgae_cflow` implementation | **DONE** (`causal_traces.py`, `convex=` branch) |
| gates: cflow / dag / cflow2 / flow | **DONE** 4/4 each; library 27/27 |
| ncopies N=4, 20 paired CRN seeds, 6 arms | **DONE** |
| s1 negative control, 5 CRN seeds, 7 arms | **DONE** |
| ablation ladder + P3-fidelity measurements | **DONE** (`_diag_prop3_identity.py`) |
| normalization defect: R statistics + hand fixture | **DONE** (`_diag_cgae_flow_rowsum.py`, `_test_cgae_cflow.py`) |
| action-dependence of R (fork test) | **DONE** (`_diag_R_action_invariance.py`) |
| four falsified mechanisms | **DONE** (rank-corr, SNR, sparsity, postpone) |
| training-curve anatomy / entropy separation | **DONE** |
| cost measurement | **DONE** (~1 ms/episode, ≤8% of a cell) |
| P1, P2, P3 | **DONE**, need retitling (`EJOR_PROPOSITIONS.md`) |
| boundary conditions §6d | **DONE** (`EJOR_BOUNDARY_CONDITIONS.md`) |
| related work | **DONE** except the novelty search + new biased-estimator paragraph |
| **P4 bias bound** | **MISSING — theory critical path** |
| **40-epoch re-run: ncopies N=2 (7 arms), N=4 (7 arms), s1 (7 arms)** | **DONE** — 145 cells |
| **N=8 @ 40 epochs (ppo/ccf/flow/cflow)** | in flight, ~7 h |
| budget-sensitivity ablation (15 vs 40 ep) | **DONE** — §6.0, and it VOIDS the 15-epoch N=2 result |
| lambda sweep (0.95/0.85/0.70 at N=2) | **DONE** — confirms the convergence-speed reading |
| `cgae_cap` variant + gate | **DONE** (`_test_cgae_cap.py` 4/4) — ships as ABLATION only, not the method |
| **`cgae_cflow` on the multi-site realistic env** | **MISSING — credibility critical path** |
| **`cgae_cflow` vs `s_ccf` head-to-head** | **MISSING — never compared** |
| `cgae_cflow` at N=2 | **MISSING** (cheap, ~1 h, strengthens the scaling table) |
| N=8 | missing (nice-to-have; `ccf`'s strongest setting) |
| figures | §6a/6b reusable; **§6c ladder figure TODO** |
| LaTeX draft | v2 exists (`paper/main.tex`, 21 pp) — §4/§5/§6a-c need rewrite |

### Critical path, in priority order

1. **`cgae_cflow` on the multi-site operational environment.** This is the
   paper's only *realistic* environment and `cgae_cflow` **has never been run
   there.** Until it is, "works everywhere we tried" excludes the paper's own
   realism argument, and a referee will observe that the headline method is
   validated on a synthetic scaling benchmark plus a null control. Highest-value
   run by a wide margin. The harness exists (`run_multisite_validate.py`,
   `METHODS = ("ppo", "ccf", "mc_q")`) and the `ppo`/`ccf` cells are already on
   disk, so only the new arm trains.

   > ⚠️ **Correction to v2, verified from the stored cells.** v2's outline credits
   > this anchor to **`s_ccf`** ("0.52 vs PPO 0.15, p=.005, d=1.27, 12 seeds").
   > The cells contain only `ccf`, `mc_q` and `ppo`. Recomputed from
   > `suite_results_multisite_bf` (12 seeds; random 62.9, heuristic 80.25):
   > **`ccf` 71.90 → 0.519**, `ppo` 65.44 → **0.146**, `mc_q` 63.53 → 0.036.
   > So the numbers are right but the method label is wrong: the anchor is
   > **`ccf`**, the *biased* variant that `s_ccf` was introduced to fix. **`s_ccf`
   > has never been run on a realistic environment at all.** Scope of the error,
   > checked: it is confined to **`EJOR_PAPER_OUTLINE.md` (v2)** — §1
   > contribution 4, §6a environments, and the readiness table. The LaTeX draft
   > is **correct** and already attributes the result to `\ccf{}`
   > (`paper/main.tex:852-862`), so nothing in the submission text needs fixing;
   > only the outline misleads.

   Silver lining for v3: the comparison this run produces is exactly the one we
   want, since `cgae_cflow` already beats `ccf` on the scaling benchmark
   (+0.199, p=0.0032). Beating the same `ccf` anchor on the realistic env would
   carry the reframe on its own.
2. ~~**`cgae_cflow` vs `s_ccf` head-to-head.**~~ **Downgraded** in light of the
   correction above: `s_ccf`'s own empirical credentials are thinner than v2
   claimed (no realistic-env validation, and it is absent from the N=4 study).
   The comparison that matters is against **`ccf`**, which is already done at N=4
   and is item 1 on multisite. Keep an `s_ccf` cell in the N=4 sweep as a
   completeness ablation, not a priority.
3. **P4 bias bound.** Without it §5 has no theorem for the shipped method, and
   Proposition 3 describes a variant that comes fourth of five.
4. **`cgae_cflow` at N=2** (~1 h) — makes the scaling table complete; `cgae_flow`
   already has 0.837 there, so the cell is conspicuously empty.
5. Optional: the `ent_bonus` intervention (would convert the entropy correlation
   into a mechanism); N=8.

Items 1, 2 and 4 are runs that already have harnesses (`run_multisite_validate.py`,
`run_ncopies_three_way_crn.py` with `seeds=`/`ns=`). Item 3 is writing.
