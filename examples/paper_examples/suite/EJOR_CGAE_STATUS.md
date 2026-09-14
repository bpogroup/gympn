# `cgae_flow` — everything we know, and what is missing

Status report as of 2026-08-11. Companion to `EJOR_PROPOSITIONS.md` (theory)
and `EJOR_PAPER_OUTLINE.md` (structure). Every number below is from a run on
disk; where a claim is *not* backed by a run, it says so.

---

## 1. The method

**`cgae_flow` = causal-order GAE with flow-weighted successor aggregation.**

Standard GAE accumulates TD errors along the *trajectory index*,
`A_t = δ_t + γλ·A_{t+1}`. In an A-E Petri net running interleaved cases,
decision `t+1` usually belongs to a different case, so `δ_{t+1}` is mostly
noise with respect to the action at `t`: credit propagates along wall-clock
order while causality flows along the provenance DAG.

`cgae_flow` runs the *identical* recursion over the provenance DAG. Three
pieces (`gympn/causal_traces.py:_redistribute_cgae`):

1. **Causal successors.** `d → s` when `s` consumes a token descending from an
   output token of `d`, with no intervening decision (nearest producer only).
   Forced firings and evolution transitions are transparent — the walk passes
   through them to the decision that actually caused the flow.

2. **Single-owner reward assignment.** Each reward goes to the *latest*
   decision on its lineage, `own(j) = argmax_{d ∈ A(j)} u_d`. Ownership is a
   **partition** — every reward counted exactly once — which is what makes this
   a return decomposition rather than `lrq`'s hindsight Q-sample.

3. **Flow-weighted recursion**, backwards in decision time:

   ```
   w(d→s)  = share of s's consumed tokens descending from d,  Σ_{d ∈ pred(s)} w = 1
   A[d]    = owned[d] − V[d] + Σ_s w(d→s)·ρ(Δt)·( V[s] + λ·A[s] )
   ```

   Emitted as `A[d] + V[d]`, consumed like every other scheme
   (`A_t = Q_t − V(s_t)`, value head regressed on `Q_t`).

The SMDP discount `ρ(Δt) = e^{-βΔt}` is applied at **causal** depth, not
wall-clock depth, and is folded per successor rather than averaged.

**Relation to `cgae` (the `mean` variant).** `cgae` averages over successors.
On a chain the two coincide exactly. Under fan-out the mean is a shrinkage with
no derivation; Proposition 3 does not cover it. Both are implemented; they are
separate schemes so earlier results stay reproducible.

---

## 2. Theory (Proposition 3, `EJOR_PROPOSITIONS.md` §346-500)

The key structural fact: **`cgae` and `ccf` omit the same mass** — everything
outside the decision's causal component — so Proposition 1's unbiasedness
argument transfers unchanged.

- **(i)** Under (A1) + (A2), the λ=1, `V≡0` estimator is unbiased. The
  unrolling gives `A_d = Σ_{j : own(j) ∈ desc(d)} ρ_d(t_j)·r_j`, and since
  `desc(d) ⊆ c(d)`, every cross-component reward is dropped exactly as in ccf.
  The residual `Ξ_d` (own-component rewards whose owner is not a descendant)
  has its whole lineage disjoint from `desc(d)`, so no descendant of `a_d` lies
  on it, so it is action-mean-independent and factors out through the
  score-function identity.
- **(ii)** Under (A1) + (A2) + (A2′) with `V = V^π`, the λ<1 estimator is the
  geometric average `(1−λ)Σλ^{k−1}A^{(k)}` of *k*-step causal advantages, each
  unbiased. `cgae` is therefore unbiased **in exactly the sense GAE is**.
- **(iii)** **(A2) drops under flow weights.** Since `Σ_{pred(s)} w = 1`, the
  credit flowing backward out of `s` totals `A_s` regardless of in-degree, so
  each owned reward gets total weight 1 across all paths.

**Assumptions.**
- **(A1) Component mean-exogeneity** — justified *from A-E PN semantics*, not
  assumed: a reward is a deterministic function of the tokens consumed at its
  firing, every token's history is in the DAG, so `r_j` is influenced by `a_d`
  only if `c(j) = c(d)`.
- **(A2) Single causal successor** — dropped by (iii).
- **(A2′) Clock-exogeneity of the cross-component critic** — **new, and the
  weakest link.** `V` is global, so summing successor values multiply-counts
  everything outside the component; and `a_d` can shift *when* the successor
  occurs, with `V_{¬c(d)}` evaluated at that moment. Holds if
  `e^{-βt}V_{¬c(d)}(t)` has zero drift given `F_d`, or if the successor time is
  conditionally independent of `a_d`. **Not verified empirically.**

---

## 3. Empirical results

All runs use **common random numbers** (`eval_seed=555000`), 20 paired seeds,
normalized to `(score − random)/(heuristic − random)`.

### 3.1 ncopies N=4 — the headline (20 seeds × 5 arms, 100 cells)

| arm | normalized final | collapses (≤0.25) |
|---|---|---|
| `cgae` | **0.936 ±0.158** | 0/20 |
| `cgae_flow` | **0.888 ±0.195** | 0/20 |
| `ccf` | 0.756 ±0.326 | 1/20 |
| `cfgae` | 0.515 ±0.552 | 6/20 |
| `ppo` | 0.477 ±0.418 | 5/20 |

Paired against `ppo`:

| comparison | diff | t-test | Wilcoxon | W/L |
|---|---|---|---|---|
| `cgae_flow` vs ppo | **+0.410** | **0.0020** | **0.0017** | 18/2 |
| `cgae` vs ppo | **+0.459** | **0.00011** | 0.00017 | 17/20 sign |
| `cgae_flow` vs cfgae | +0.372 | 0.0083 | 0.0055 | 13/6 |
| `cgae_flow` vs ccf | +0.132 | 0.126 | 0.171 | 13/6 |
| `cgae_flow` vs cgae | −0.048 | 0.458 | 0.295 | 6/13 |

### 3.2 ncopies N=2 (20 seeds × 4 arms) — **`cgae_flow` NOT RUN**

| arm | normalized | diff vs ppo | t | Wilcoxon | collapses |
|---|---|---|---|---|---|
| `cgae` | 0.632 | +0.371 | 0.024 | 0.022 | 5/20 |
| `ccf` | 0.600 | +0.339 | 0.042 | 0.026 | 7/20 |
| `cfgae` | 0.589 | +0.328 | 0.019 | 0.037 | 6/20 |
| `ppo` | 0.261 | — | — | — | 12/20 |

### 3.3 s1 (5 seeds, CRN) — the negative control. **`cgae_flow` NOT RUN**

| arm | final | mean greedy | drift |
|---|---|---|---|
| `ppo_clip` | 13.14 ±1.26 | 12.75 | 0.84 |
| `cgae` | 13.05 ±0.54 | 13.03 | 1.01 |
| `cfgae` | 13.30 ±0.71 | 12.81 | 0.62 |

**Every comparison null, p ≥ 0.45.** Test-only CRN rescore at 200 episodes/cell
agrees: ppo 14.249, cgae 14.073, cfgae 14.030, all n.s.

### 3.4 Mechanism — measured *before* the final N=4 result

| | ncopies N=4 | s1 |
|---|---|---|
| reward-bearing components K | 3.2 | **1.0** |
| largest component's reward share | 46.3% | **100%** |
| ccf/mc_q credit ratio | 0.336 (≈1/4) | 0.800 |
| cgae fan-out | 1.07 (93% ≤1 successor) | 1.34 (34.5% multi) |

**This is the strongest evidence in the paper.** The same estimator, untold,
finds four components on ncopies and one blob on s1 — and wins in the first
case, ties in the second. A mechanism that predicts its own failure case.

---

## 4. Methodology built along the way

- **Common random numbers** (`Agent.test_in_train(eval_seed=...)`). Noise floor
  measured on s1: **±0.231 SD per 20-episode eval point**, ±0.327 on a paired
  difference, and `greedy_drift` inflated by ~0.40 for a genuinely flat policy.
  Without it, two arms are compared on different sample paths.
- **Dispatch fix.** `run_suite._make_args` mapped any method outside a
  hand-listed subset to `causal_scheme='lrq'` — `cgae`/`cfgae`/`alin`/`lrq2c`
  silently trained lrq under their own names. All pre-2026-08-10 numbers for
  those schemes are void.
- **Forced-move fix.** Single-binding actions are auto-fired and registered
  with `agent_decision=False` (lineage node, not decision), so the decision
  sequence no longer depends on `causal_rl`.
- **`alin` closure fix.** Reachability sorted by time alone, truncating on ties
  (s1: 36.4% of contention edges same-time, 63 fallbacks → 0).
- **Postpone token-flow is a partial global coupler**: ncopies K 4.00 → 3.25,
  shared_facility 3.00 → 2.15. **The committed ncopies results therefore
  understate the available structure.**

---

## 5. What is missing

Ordered by how much each would strengthen the paper.

### 5.1 Blocking for a `cgae_flow` headline

1. **`cgae_flow` on s1.** The "costs nothing where there is no structure" half
   of the claim has only ever been run for `cgae`. The main method has no
   negative control. *5 cells, ~1 h.*
2. **`cgae_flow` at N=2.** The method rests on a single env at a single N.
   Cheap and directly strengthens the main table. *20 cells, ~1 h.*
3. **Related-work verification.** The novelty framing — automatic *discovery*
   of the credit factorization from provenance, versus factored MDPs which are
   handed the DBN — is my reading of the literature, **not a search that was
   completed**. Must be verified before any novelty claim ships.

### 5.2 Would materially strengthen it

4. **N=8.** Tests whether the gap grows with independence. It is `ccf`'s
   strongest setting in the stored sweep (+0.569), and `cgae`/`cgae_flow` have
   never run there. If the ranking flips, the "best method" claim changes.
   *~5-6 h.*
5. **`alin` at N=4.** On ncopies `alin ≡ lrq2` exactly (contention edges add
   nothing), so it isolates cgae's *recursion + ownership* against pure lineage
   scoping. If `alin` reaches 0.93, the recursion is decoration. *20 cells, ~3 h.*
6. **Component-local critic.** (A2′) is assumed, not implemented. `cfgae`'s
   `component_step_rewards()` already computes the masked reward stream, so
   training `V` per component is a combination of existing parts and would
   *remove* the weakest assumption rather than defend it.
7. **Sinkless-postpone rerun.** Token-flow postpone costs ~19% of the
   factorization (K 4.00 → 3.25). Results are probably *understated*.

### 5.3 Open and unresolved

8. **`cgae` (mean) scores higher than `cgae_flow`** — 0.936 vs 0.888, 13/6
   seeds — but is not covered by Proposition 3. Difference n.s. (p=0.46). The
   honest framing is to ship `cgae_flow` and footnote the mean; the alternative
   (lead with the better number, defend an underived shrinkage) is worse. **A
   decision, not an experiment.**
9. **`cgae` vs `ccf` is not established.** +0.180 (p=0.061) for cgae, +0.132
   (p=0.126) for cgae_flow. More seeds or N=8 would settle it.
10. **No second environment.** A `shared_facility` env was designed and built
    to defeat the "structure was planted" objection — one connected net, k
    pools, coupling tuned by an exogenous arrival mix. The coupling dial works
    (**K 3.90 → 1.00** monotone), but **it has no headroom where it has
    structure**: at p=0 the heuristic beats random by only 2.4-4.3%, versus
    +86.6% at p=1. Five levers (ordering, server matching, case value,
    abandonment, arrival rate) all hit the same ~8% ceiling. Root cause: in a
    single-stage system nothing compounds, and every source of difficulty is
    route length — the same knob that destroys the factorization. Two
    constraints discovered that bind *any* env in this family:
    **multi-server pools break (A1)** (the agent choosing the server chooses
    the partition), and **components form per server token, not per pool**.
    See `shared_facility_env.py`, kept as a documented negative result.
11. **`cfgae`'s K=1-equals-PPO claim is untested.** N=1 on ncopies saturates
    (`greedy_final` takes two values), so the sharp test never ran.
12. **No sensitivity analysis** on λ or β for any causal scheme.
13. **Compute cost not reported systematically.** Rough measurement: the entire
    causal-credit computation is ≤8% of a training cell (ppo 7.44 min vs cfgae
    8.05 min at N=4); the GNN dominates. Fine for a footnote, not yet a table.

---

## 6. Suggested minimum experiment set

For a defensible paper with `cgae_flow` as the method:

| # | run | cells | est. |
|---|---|---|---|
| 1 | `cgae_flow` on s1, 5 seeds (negative control) | 5 | ~1 h |
| 2 | `cgae_flow` at ncopies N=2, 20 seeds | 20 | ~1 h |
| 3 | `alin` at N=4, 20 seeds (scoping-vs-recursion ablation) | 20 | ~3 h |
| 4 | N=8 for ppo/cgae/cgae_flow/ccf, 10 seeds | 40 | ~5-6 h |

(1) and (2) are blocking. (3) answers the sharpest methodological question a
referee will ask. (4) is the scaling story and the cgae-vs-ccf tiebreak.

Everything needed for these already exists: `run_ncopies_three_way_crn.py`
takes `seeds=` and `ns=` and is resumable per cell; `run_three_way_s1_5seed_crn.py`
covers s1.
