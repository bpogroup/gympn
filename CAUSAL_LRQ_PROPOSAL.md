
# LRQ: fixing REC's per-step estimator bias

**Scope.** A concrete proposal to fix the one substantive problem left open by
`CAUSAL_REC_CRITICAL_REVIEW.md` §3: REC's credits are hindsight quantities, so
the return-to-go / TD estimators that actually consume them are biased, with a
demonstrated stable suboptimal attractor (`_counterexample_rec_returntogo.py`).

**Proposal in one sentence.**

> Stop treating the causal credits as a *reward process* and feed the lineage
> information to the learner as a *per-decision Monte-Carlo Q-sample* instead:
> each decision's learning signal is the **full** discounted sum of every
> future reward in whose causal lineage it sits (no `1/|P|` split), baselined
> by a value head trained on that same quantity, with **no** GAE/TD chaining
> over credits. Call it **LRQ** (Lineage-Restricted Q).

The counterexample resolves immediately: decision 2 compares Q(X) = 12 against
Q(Y) = 7 and correctly prefers the chain (verified against the trace machinery
in `scratchpad/lrq_check.py`; REC compared 6 vs 7 and locked into Y).

---

## 1. Why the fix cannot stay inside "credits-as-rewards"

First, the impossibility observation from the review, sharpened into a lemma —
it dictates the shape of any fix.

> **Lemma (no sound backward split exists).** Let credits satisfy REC's
> constraint (P) (per-reward partition of unity over causally-prior decisions)
> and suppose the per-step credit return-to-go is correct at **every** step,
> i.e. `Ĝ_t := Σ_{k≥t} e^{−β(u_k−u_t)} c_k = Σ_{t_j ≥ u_t} e^{−β(t_j−u_t)} r_j`
> for all `t`. Differencing consecutive steps forces
> `c_t = Σ_{u_t ≤ t_j < u_{t+1}} e^{−β(t_j−u_t)} r_j` — i.e. the credits are
> **uniquely** the temporally-local rewards. No redistribution at all.

So within the credits-as-rewards framework there are only two internally
consistent corners, both already rejected:

- **Temporal placement** (the lemma's unique solution): return-to-go correct,
  but zero causal de-smearing — this is just standard SMDP RL.
- **Any genuine backward split** (REC's equal split, last-decision placement,
  anything satisfying (P)): mass moved behind a decision boundary vanishes
  from every later decision's return-to-go — the §3 bias, structurally
  unavoidable.

Conclusion: the fix must change **how the trace information enters the
gradient**, not the weights `w`. The two principled exits are (a) per-decision
Q-samples via the policy-gradient theorem — proposed here; or (b) learned
expectation-based redistribution (RUDDER `g`-differences / HCA hindsight
ratios) — heavier, discussed as the fallback in §7.

## 2. The estimator

For every decision `t` (indexed as in the action-transition list) define the
**hindsight lineage return**

```
┌────────────────────────────────────────────────────────────────┐
│  Q̂_t  =  Σ_{j : t ∈ L_j}  r_j · e^{−β (t_j − u_t)}              │  (LRQ)
└────────────────────────────────────────────────────────────────┘
```

- `L_j` = the decisions in reward `j`'s token-DAG lineage — the **same walk**
  REC already performs (`production_set`), with two changes: the weight is `1`
  instead of `1/|P|`, and **postpone decisions are included** (see §4).
- No exogenous fallback: a reward with no decision in its lineage appears in
  **no one's** `Q̂` (see §5).
- Baseline: a value head `V_L(s)` regressed on `Q̂` (the existing critic with
  different targets: set `returns_ep = Q̂` in the causal branch of `finish()`).
- Advantage: plain centering, **no smdp_gae over credits**:

```
A_t = Q̂_t − V_L(s_t)
```

## 3. Why this is the correct object (and what bias remains)

The policy-gradient theorem's exact form is
`∇J = E[Σ_t ∇log π(a_t|s_t) · (Q^π(s_t,a_t) − b(s_t))]` — per-step **Q**, not
return-to-go over a redistributed reward. Decompose the true continuation after
decision `t` into rewards whose lineage will contain `t` and the rest:

```
Q^π(s_t, a_t) = E[ lineage part | s_t, a_t ]  +  E[ off-lineage part | s_t, a_t ]
                └── Q̂_t is an unbiased MC sample of this ──┘
```

`Q̂_t` samples the first term exactly (every reward `a_t` causally produced,
discounted from `u_t` — full mass, which is why the counterexample resolves).
The second term is the future of *other* cases. To the extent it is independent
of `a_t` given `s_t`, it contributes nothing to the gradient and omitting it is
exact — this is the standard argument that lets a baseline absorb
action-independent continuation value.

**The one residual bias is foreclosure** (Gap D): through resource contention,
`a_t` *does* shift other cases' rewards (occupying a machine delays them), and
that differential is simply missing from `A_t`. Three honest observations:

1. This is strictly *less* bias than REC-as-consumed, which had the foreclosure
   gap **plus** the chain-dilution bias. LRQ removes the demonstrated,
   systematic one and keeps the diffuse one.
2. The direction of the trade is right for this codebase's environments: chain
   dilution punished *exactly the behaviour the tasks are about* (completing
   multi-stage cases), while foreclosure effects are second-order when
   contention is moderate — and are *also* missed by REC's TD(0) unless the
   critic carries them.
3. It is measurable (§6 gives a diagnostic) and hedgeable: mix
   `A_t = (1−μ)·A_LRQ + μ·A_GAE` with the standard SMDP-GAE advantage. `μ`
   interpolates between "no cross-case smearing, foreclosure-blind" (μ=0) and
   "unbiased at λ=1, full temporal smearing" (μ=1). Even small μ restores a
   gradient path for contention effects.

What is *given up*: S1 return-equivalence no longer applies — `Σ_t Q̂_t` is not
the return (a reward with `k` lineage decisions is counted `k` times), so LRQ
is **not** a redistribution and must never be summed as one. Soundness now
rests on the PG theorem plus the off-lineage-independence assumption, not on
RUDDER. This is the honest trade: REC had an exact theorem about the
*objective* and a broken estimator; LRQ has an approximate assumption about
the *environment* and a correct estimator shape.

## 4. Postpone: included in the lineage, penalised by the discount

Under token-flow postpone (`causal_postpone_tokenflow=True`, which LRQ should
**require**), a postpone consumes and re-emits the case's tokens and therefore
sits in the lineage of that case's later rewards. Include it in `L_j` (REC
excluded it as an existence-game null player — that reasoning belonged to the
partition-of-unity world and is no longer needed):

- Postponing a case that later pays `r` at `t_j` yields
  `Q̂(postpone) = r·e^{−β(t_j − u_t)}`; acting now yields the same reward
  earlier, so `Q̂(act) = r·e^{−β(t_j' − u_t)}` with `t_j' < t_j` — **acting
  dominates exactly by the timing cost**, symmetrically and on the same scale,
  with no special-casing.
- A *beneficial* wait (batching: waiting enables a larger or additional reward)
  shows up as larger lineage mass in `Q̂(postpone)` — postpone finally gets a
  **direct** benefit channel, which under REC existed only through the critic
  bootstrap. This addresses review §4.7's "postpone learning is
  critic-limited" for the benefit side, not just the penalty side.

Without token-flow postpone, postpone has an empty lineage, `Q̂ = 0`, and
`A = −V_L(s)` would suppress waiting unconditionally — so the implementation
must assert the flag.

## 5. Exogenous rewards: ignored, correctly

A reward with no decision in its lineage is uncontrollable; under LRQ it enters
no `Q̂_t` and no baseline target asymmetry (both `Q̂` and `V_L`'s targets exclude
it consistently). Omitting an action-independent term from a PG estimator is
exact — this *deletes* review issue §4.2 (the fallback that smeared exogenous
mass uniformly, including onto postpones) rather than patching it. If some
"exogenous" reward is actually foreclosure-coupled to actions, it is covered by
the §3 residual-bias discussion and the μ-hybrid.

## 6. Implementation plan (small; ~40 lines total)

1. **`causal_traces.py`** — new scheme `"lrq"`: a variant of
   `_redistribute_rec` that (a) drops the `1/|P|` factor, (b) does **not**
   filter `postpone_idx` out of the lineage walk, (c) deletes the exogenous
   fallback. The DAG walk, timestamps, and `disc()` are reused verbatim.
   (Optional micro-fix while there: memoize per-token ancestor-decision sets to
   address the §4.6 complexity note — LRQ does the same per-reward walk.)
2. **`data.py`** — in `finish()`'s causal branch, when
   `causal_scheme == "lrq"`: `returns_ep = credits_vec` (value targets) and
   `adv_ep = credits_vec − values_ep`, optionally
   `+ μ·smdp_gae(rewards_raw, …)` for the hybrid; skip
   `smdp_discounted_returns`/`smdp_gae` over credits entirely. Note the
   existing length guard (`data.py:597-603`) already covers alignment.
3. **`train.py`** — add `"lrq"` to `--causal_scheme` choices; validate
   `causal_postpone_tokenflow=True` when selected (hard error otherwise); add
   `--causal_mu` for the hybrid coefficient (default 0).
4. **Clock consistency assertion** (review §4.3) — required here for the same
   reason as REC: `u_t` in the discount and the buffer's decision times must
   agree. Add the divergence check in `finish()`.
5. **Tests** (`_test_lrq.py`):
   - counterexample regression: decision-2 signal must rank X (12) over Y (7)
     — the exact failure REC exhibits;
   - postpone timing: same case with/without a wait ⇒
     `Q̂(act) > Q̂(postpone)` by exactly the discount factor; batching variant ⇒
     postpone's `Q̂` exceeds acting when waiting enables a larger reward;
   - exogenous reward enters no `Q̂`;
   - determinism, and scale: `Q̂_t ≤ Σ_j r_j` per decision.

## 7. Validation plan and fallback

1. **Counterexample environment** (review §6.1: one resource, one-stage cases
   worth `r₁`, two-stage worth `r₂`, `r₂/2 < r₁ < r₂`). Prediction: REC
   converges to serving one-stage cases, LRQ to two-stage, standard PPO to
   two-stage but slower under concurrency noise. This is the decisive A/B.
2. **Foreclosure diagnostic** (quantifies LRQ's residual bias): per state,
   regress the *off-lineage* return-to-go on the chosen action (a one-hot
   probe). A near-zero coefficient validates the independence assumption for
   that env; a large one says raise `μ`. Cheap to compute from existing traces
   — the off-lineage return is `(true discounted return-to-go) − Q̂_t`.
3. **Existing suite** (`examples/paper_examples/suite/`): expect LRQ ≥ REC
   everywhere, with the gap concentrated on envs mixing case lengths; expect
   LRQ vs standard PPO to win where concurrency (cross-case smearing) is high.
4. **Fallback if the foreclosure diagnostic is large everywhere:** the
   expectation-based route — keep LRQ's lineage walk but replace the realized
   return with a prediction difference (RUDDER-style `g(τ_{0:t}) − g(τ_{0:t−1})`
   trained on `R_β`, or Mesnard-style future-conditional baselines with the
   lineage as the hindsight statistic). That machinery is heavier (a learned
   sequence model and its own convergence) and should only be paid for if the
   diagnostic demands it.

## 8. Relation to the literature (one paragraph)

LRQ is per-step Q-sampling (Sutton et al.'s PG theorem) with a
*trajectory-realized causal filter* deciding which rewards belong to a
decision's Q-sample — closest in spirit to Hindsight Credit Assignment
(Harutyunyan et al. 2019) with the token lineage as an *oracle* hindsight
classifier (the PN trace gives exactly what HCA has to learn), and to
Counterfactual Credit Assignment (Mesnard et al. 2021), whose validity
condition (the hindsight statistic must not leak the action's off-channel
effects) is precisely the off-lineage-independence assumption of §3. REC's
discount-placement idea survives intact: `e^{−β(t_j−u_t)}` is what makes
`Q̂_t` a *discounted-from-`u_t`* Q-sample, and it is still what penalises
waiting — that part of the design was right and is kept.

## 9. Bottom line

The review showed the flaw is not in REC's weights but in the interface:
partition-of-unity credits + return-to-go consumption is unsound for **any**
backward split (§1 lemma). LRQ changes the interface — same trace, same DAG
walk, same discount, one deleted factor of `1/|P|`, and a simpler consumer
(`A = Q̂ − V_L`, no GAE over credits). It restores full downstream mass at
every decision (killing the chain-dilution attractor, verified on the
counterexample), gives postpone a direct benefit channel, deletes the
exogenous-smearing wart, and confines the remaining bias to foreclosure —
which is measurable and hedgeable with a single mixing knob.

---

*Companion documents: `CAUSAL_REC_CRITICAL_REVIEW.md` (the problem being
fixed), `_counterexample_rec_returntogo.py` (the failure LRQ must and does
resolve), `CAUSAL_REDISTRIBUTION_SOUND_SCHEME.md` (REC design; its §2 weight
freedom is what §1 here proves insufficient).*
