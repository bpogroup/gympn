# Fork-free lineage credit: can we get the counterfactual's win without the fork?

*Brainstorm, 2026-07-31. Companion to CAUSAL_LINEAGE_RETHINK.md and the
counterfactual-DAG-replay results (dag_replay_probe / m2_ / m4_ / train_counterfactual).*

## The state of play

We have three regimes for lineage-based credit:

| method | forks? | unbiased? | granularity | verdict |
|---|---|---|---|---|
| `lrq`/`ccf` | no | **no** (flips on shared resources) | finest | fast, biased |
| `s_ccf` | no | yes | conservative (keeps contested rewards) | fast, coarse |
| `cf` (DAG-replay counterfactual) | **yes** | yes | finest | correct, **expensive** |

`cf` wins on both axes (unbiased + finest) because it *computes* each action's
causal effect instead of *guessing* the reward-to-decision boundary. But it forks
the simulator at every decision — measured cost ~16 s/episode vs sub-second for
the filtering methods on `s1_stoch_sequence`. The question this document attacks:

> **Is the fork essential, or can the same unbiased-and-finest credit be
> recovered from the factual trajectory plus the provenance DAG alone?**

## The enabling observation: the bias is *localized*

The filtering methods are not wrong everywhere. Classify each reward relative to
a decision `d` using the provenance DAG (this is exactly what the static-component
analysis in `s_ccf` and the `conflict_graph.py` extractor already compute):

1. **PURE** — reachable only from `d`'s own action(s), always realized. Filtering
   is **correct**: the reward is `d`'s, full stop.
2. **EXOGENOUS** — not reachable from any action (pure environment reward).
   Filtering *should* drop it; `ccf` sometimes wrongly pulls it in via a join
   (M1), but `s_ccf`/lrq handle it. The counterfactual effect of `d` on it is
   **exactly zero**.
3. **CONTESTED** — reachable from `d` *and* from a competing decision, coupled
   through a shared token (a join, or a shared resource). This is the *only* place
   filtering breaks: whether the reward lands in `d`'s realized lineage depends on
   the action (`use_R` grabs the resource → `priv` joins `d`'s lineage; `standalone`
   → it doesn't). That action-dependent membership is the M2 flip and the M4
   conservatism, both at once.

**So a fork-free fix does not need to re-derive all credit — it only needs to
correct the CONTESTED rewards, and the DAG already tells us exactly which few
reward-types those are.** Everything else the cheap filtering already gets right.
This is the lever: shrink the hard problem to the handful of contested reward-types
per net, then spend a fork-free estimator only there.

---

## Idea 1 (main): Lineage-structured Hindsight Credit Assignment (LS-HCA)

**The connection.** Hindsight Credit Assignment (Harutyunyan et al. 2019) computes
the policy-gradient advantage *without* forking, by asking the dual question: given
an observed future outcome `z`, how much more likely was action `a` than its prior?

```
A(s,a) = Σ_k γ^k r_{t+k} · ( 1 − π(a|s) / h(a | s, z_{t+k}) )
```

where `h(a|s,z) = P(a_t = a | s_t, z)` is the **hindsight distribution**, learned
from the data. If outcome `z` is independent of the action, `h = π` → the factor
is 0 → no credit (this is how HCA drops exogenous noise unbiasedly, fork-free). If
`z` is more likely under `a`, `h > π` → the reward credits `a`.

**Why vanilla HCA is hard, and why provenance fixes it.** HCA's cost and variance
come from having to *learn* `h(a|s,z)` for every future outcome `z`. But the
provenance DAG gives the structure of `h` almost entirely for free:

- **Exogenous reward** (case 2): the action cannot causally reach it ⇒ `h = π`
  **exactly, by construction** ⇒ credit 0. No estimation, zero variance. (Vanilla
  HCA has to *learn* that `h ≈ π` from noisy data; we read it off the DAG.)
- **Pure reward** (case 1): realized iff `d` took the reaching action ⇒ `h ∈ {0,1}`,
  deterministic ⇒ credit = the reward, exactly. No estimation.
- **Contested reward** (case 3): `h(a | s, reward-realized)` is genuinely between
  0 and `π` — *this is the only place we estimate*, and there are only a few
  contested reward-types per net (the conflict graph enumerates them).

So **LS-HCA = HCA whose hindsight model is supplied exactly by the provenance DAG
for the pure/exogenous rewards, and estimated (from the batch's own data
distribution) only for the contested reward-types.** Fork-free, and the estimation
burden is a small, DAG-identified, low-dimensional sub-problem instead of the full
outcome space.

**Estimator sketch (per decision `d`, action `a`):**
```
A(s,a) =  Σ_{r ∈ PURE(a)}      r·e^{-βτ}                      # exact, keep
        + Σ_{r ∈ CONTESTED(d)} r·e^{-βτ}·(1 − π(a|s)/ĥ(a|s,r)) # HCA-corrected
        + 0·Σ_{r ∈ EXOGENOUS}                                 # dropped, exact
        − V_L(s)                                              # learned centering
```
`ĥ(a|s,r)` is estimated per contested reward-type — e.g. a logistic model
`P(a | s, "reward-type r was realized in d's lineage")`, fit on the batch. Because
the conditioning event is a *lineage-membership indicator* the DAG already labels,
this is a tiny, well-posed fit, not a return-space density estimate.

**Unbiasedness / always-safe floor.** HCA is unbiased when `ĥ` is exact. The safe
degradation: if `ĥ` is uninformative (`ĥ = π`), every contested term vanishes and
the estimator collapses to *pure-rewards-only* credit — a valid (if conservative)
lineage baseline, never biased. So a bad hindsight fit costs granularity, not
correctness — the same failure mode as `s_ccf`, which we already accept. **This is
the property the user asked for: at worst unchanged (conservative), at best the
counterfactual's finest-grained unbiased credit — with no fork.**

**Open risk.** HCA is famously high-variance when `h` is poorly estimated. The bet
is that provenance-pruning (only contested reward-types, with the membership event
handed to us) is exactly what tames it. Unproven — needs the estimator check below.

---

## Idea 2: Batch-sibling counterfactual (fork-free COMA)

The fork exists only to obtain `G_lin(a')` for the *untaken* actions. But across a
training batch, the *same decision point* is visited many times and different
actions are sampled. **Estimate `G_lin(a')` from sibling episodes that actually took
`a'`**, matched by decision-context (transition type + a coarse marking signature),
instead of from a fork.

```
b(s) ≈ Σ_{a'} π(a'|s) · mean{ G_lin(a') over batch siblings at this decision-point }
A(s,a) = G_lin(a) − b(s)
```

- **Fork-free**: uses the batch we already collected.
- **Lineage is what makes it viable**: the lineage restriction strips the
  cross-decision and exogenous reward from `G_lin`, so two siblings that took the
  same action but saw different global episodes still have *comparable* `G_lin` —
  the very noise that would swamp a raw-return sibling comparison is what lineage
  removes. (Whole-return siblings would be hopeless; lineage-restricted siblings
  are the same variance-reduction that gave lrq its 28W/0L over mc_q, reused as the
  baseline estimator.)
- **Cost**: a hash-bucket over decision-points, O(batch). Negligible.
- **Loss vs the fork**: no common random numbers — siblings don't share the fork's
  exogenous draws, so the exogenous component doesn't cancel *exactly*, only in
  expectation. Higher variance than `cf`, but unbiased and free. Combine with a
  control variate (Idea 3) on the residual exogenous term to recover most of the
  CRN benefit.

---

## Idea 3: Lineage-scoped control variate (extends the existing LCV)

The library already has `lcv` — a measured control variate that subtracts
`c·(R_off − v_off(s))`, where `R_off = mc_q − lrq2` is the *entire* off-lineage
return. It is fork-free and always-safe (`ĉ→0` recovers PPO). **Refinement: scope
the control variate to the CONTESTED return only** (conflict-graph-identified),
rather than all off-lineage reward:

```
A = A_lrq − Σ_{r-type ∈ CONTESTED}  c_r · ( R_r − Ê[R_r | s] )
```

- The contested rewards are the *only* ones carrying action-dependent-membership
  bias; a CV fit on just them is lower-dimensional and better-targeted than LCV's
  blanket off-lineage term.
- **Fork-free, always-safe** (each `c_r→0` is a no-op), tiny overhead (one
  regression coefficient per contested reward-type).
- **Limitation** (honest): a *linear* CV removes the linear-correlated part of the
  contested reward. It cancels an exogenous noise reward exactly (as the
  hindsight-baseline probe showed) but only *approximates* a nonlinear
  timing-shift effect (the `priv`-fires-later-under-`use_R` structure). So this is
  the safe, incremental option — strictly better than `lrq`, not necessarily as
  sharp as `cf`. Good "floor" method; pair with Idea 1/2 for the nonlinear part.

---

## Idea 4: Amortized / distilled counterfactual (fork early, then stop)

Fork for the first `N` epochs to generate `(s, a) → G_lin(a)` targets, train a
small head to predict them, then **stop forking** and use the learned predictor as
the counterfactual baseline. Cost decays to zero; accuracy improves as the policy
stabilizes (the counterfactual landscape stops moving). This is a pragmatic hybrid,
not fork-free from t=0, but it bounds total fork spend to a warm-up. Natural
fallback if Ideas 1–3 leave a residual gap.

---

## Idea 5: Critic conditioned on lineage/conflict features (cheapest, weakest)

Feed the critic provenance-derived features (per-decision: contested-token count,
occupancy of contested resources, conflict-graph degree — several already computed
in `counterfactual._branch_tally`'s `feats`). A richer `V(s, φ_lineage)` can absorb
the contested-reward structure into the baseline, sharpening the advantage without
forks. Heuristic — no unbiasedness guarantee beyond "it's still a state-baseline" —
but nearly free and composable with any of the above. Lowest effort, lowest
confidence.

---

## Ranking and recommendation

| idea | fork-free | overhead | unbiased | sharpness | risk |
|---|---|---|---|---|---|
| 1 LS-HCA | yes | low (contested-only fit) | yes if ĥ good; conservative floor | potentially = `cf` | HCA variance (pruning should tame) |
| 2 batch-sibling | yes | negligible | yes | high (no CRN → more var) | needs decision-point matching |
| 3 scoped CV | yes | negligible | yes (always-safe) | linear part only | limited sharpness |
| 4 amortized | warm-up only | medium→0 | yes | = `cf` | distribution shift as π moves |
| 5 lineage-critic | yes | negligible | no guarantee | modest | heuristic |

**The principled target is Idea 1 (LS-HCA)** — it is the fork-free dual of the
counterfactual, and provenance is *precisely* the structure that makes HCA
tractable ("the A-E PN gives HCA the hindsight model it otherwise has to learn").
**The safe, ship-tomorrow option is Idea 3 (scoped CV)** — always-at-least-unchanged,
trivial cost, a strict improvement on `lrq` even if not as sharp as `cf`.

**Cleanest research story if it works:** *the provenance DAG turns the expensive
counterfactual into a fork-free estimator by (a) discharging pure/exogenous credit
exactly and (b) reducing the residual to a small hindsight/CV correction on the
few contested reward-types the net's conflict structure identifies.* That is the
same "compute the effect, don't guess the boundary" thesis as `cf`, but paid for
with a batch statistic instead of a simulator fork.

---

## The decisive next experiment (estimator-level, before any training build)

Reuse the M2/M4 motifs (`assembly_probe`) where we already have ground truth:

1. On M2 (shared-R, where `ccf`/`lrq` flip): compute the **LS-HCA credit** (pure +
   contested-HCA + exogenous-drop) from a *batch of factual trajectories only* (no
   forks) and check it recovers the counterfactual's unbiased sign (−7.6), i.e. it
   does NOT flip. The contested reward is `priv`; `ĥ(use_R | priv-in-lineage)` is
   estimated from the batch.
2. On M4 (abundant resource, where `s_ccf` is conservative): check LS-HCA factors
   `r_b` out (contested-but-uncontended → hindsight factor ≈ 0) while staying
   unbiased — matching `cf`, beating `s_ccf`.
3. Compare variance and wall-clock against `cf`'s forks.

If LS-HCA reproduces `cf`'s M2∧M4 dominance from factual batches alone, the fork is
inessential and the method becomes cheap. If it recovers the *sign* but not the
full sharpness, fall back to the scoped-CV floor (Idea 3) and report the
bias/variance/cost trade honestly. Either outcome is a publishable, honest result.
```