# Lineage + sparse counterfactual correction (`lrq2c`)

*Written 2026-07-27, after the MCTS/AlphaZero detour. Companion to
CAUSAL_LINEAGE_RETHINK.md, AEPN_NATIVE_LEARNING.md, CFPK_EXPLAINED.md.*

## 0. The thesis, in one line

**Provenance-restricted credit (`lrq2`) is a cheap, exact variance filter that
equals expensive counterfactual simulation (cfpk) everywhere EXCEPT foreclosing
decisions — and those are (a) structurally identifiable in advance and (b)
exactly repairable with a handful of CRN forks.** So: lineage everywhere for
free, simulation only where the theory says lineage is biased.

This is a **PPO advantage**, not an MCTS distillation target — the reward
gradient is the stabilizer the AlphaZero detour lacked (it collapsed to
always-postpone because distillation has no return signal; verified 2026-07-24).

## 1. The measured facts this stands on

| fact | evidence |
|---|---|
| lineage-restricted return filters concurrent-case noise → **perfect grid, faster than PPO** | `lrq` vs `mc_q`, 28W/0L, p<1e-4 |
| lineage's ONLY failure is **foreclosure** (s1): variance↓ (−11% SE) but bias↑ (+8%) | cfpl decomposition (§7.2b) |
| the bias = the **indirect (opportunity-cost)** channel, which is **not smoothly modellable** | cfpd: held-out R²≈0 → "simulation beats modelling" |
| but the indirect channel IS **exactly measurable by a CRN fork** | cfpk / `_branch_tally` returns (total, direct, feats) |
| the contested (foreclosure-capable) decisions are **structurally identifiable** | Direction B conflict graph: joint envs have ≥1 action-conflict edge, disjoint have 0 |

The gap in every prior attempt: they either used lineage as the *whole* engine
(biased on foreclosure) or modelled the indirect channel (failed, cfpd) or used
a global control variate (tied the baseline, LCV). None **used the exact fork
indirect term, sparsely, only where structure says foreclosure can happen.**

## 2. The method

Per decision `t` with taken action `a`, the corrected advantage is

```
A(s_t, a) = A_lrq2(s_t, a)                       # cheap, from the factual trace
          + 1[t is foreclosure-gated] · Î_t      # exact indirect, from a CRN fork
```

- **`A_lrq2`** — the existing lineage-restricted advantage (`Q_lineage − V`,
  postpone excluded), computed for free from the episode's causal trace. This
  is the DIRECT (descendant) channel and carries the bulk of the signal. On
  direct-dominated envs it is already exact (→ grid stays perfect).
- **`Î_t`** — the EXACT indirect / opportunity-cost term, measured by a paired
  CRN fork: `Î_t = (G_total − G_direct)`, the difference between the fork's
  whole-return gap and its lineage-restricted gap (this is exactly
  `_branch_tally`'s `total − lineage`, already computed by the `cf_decompose`
  path). No modelling — the term the cfpd regression failed to learn is
  measured directly. Added only at gated decisions.

Consumed as a PPO advantage (`A_t = Q_t − V(s_t)` form, value head on the same
target), so the return gradient prevents any degenerate collapse.

## 3. The router — where to fork (the cheap part)

Forking everywhere is cfpk (expensive). Fork only where foreclosure can bite:

1. **Structural gate (primary, free, general):** fork only at decisions that are
   **action-conflicts** in the static conflict graph (`conflict_graph.analyze`)
   — i.e. the action competes for a shared input place (a contested resource).
   On disjoint envs (0 conflict edges) this fires **never** → pure `lrq2`, grid
   untouched. On s1 it fires at the contested employee decisions.
2. **SNR / dispersion sub-gate (secondary, optional):** among gated decisions,
   fork with probability rising in the lineage-credit dispersion (fork where the
   factual credits are clustered/uncertain — the R6/SNR idea). Bounds the fork
   budget when most decisions are structurally contested.
3. **Confirm-and-correct:** the fork does double duty — if the measured `Î_t` is
   within its own paired SE (foreclosure negligible here), apply no correction
   (lineage was fine); if `|Î_t| > gate·SE`, apply it. So a fork that fires on a
   non-foreclosing decision self-cancels — no new bias introduced.

## 4. Reuse map (what already exists)

| piece | existing code |
|---|---|
| `A_lrq2` advantage | `causal_traces.redistribute_rewards(scheme='lrq2')` + `data.py` finish() |
| CRN fork + snapshot + paired gap | `counterfactual.maybe_fork` (snapshot/restore/CRN validated) |
| direct/indirect decomposition `Î = total − direct` | `counterfactual._branch_tally` / `_decomp_suffix` (cf_decompose path) |
| structural router | `conflict_graph.analyze(...).action_conflict_edges` |
| PPO consumption | the standard causal advantage path (`causal_scheme`, `finish(mode='replace')`) |

Net new code is small: a `causal_scheme='lrq2c'` that (a) runs the `lrq2`
redistribution, (b) at gated decisions calls the fork's decomposed tally to get
`Î_t`, (c) adds `Î_t` to the per-decision advantage before it enters GAE-free
consumption. No new estimator, no learned head, no distillation.

## 5. Why this dodges every prior failure mode

- **vs MCTS distillation collapse (2026-07-24):** PPO advantage, reward gradient
  present → no postpone basin.
- **vs cfpd modelling failure (R²≈0):** uses the EXACT fork indirect term, never
  a regression on occupancy features.
- **vs LCV control-variate tie:** not a variance device bolted on an already-
  converging method — it injects the *missing bias term*, so it can move
  finals (route (i)), not just variance.
- **vs cfpk cost (forks everywhere, 3.6×):** forks only at structurally-contested
  decisions, sub-sampled — far fewer forks.
- **vs R5 generality objection:** the router reads only the PN formalism (conflict
  graph = shared input places) and the trace — no token-value semantics.

## 6. Experiment plan (small, decisive)

Env tier: a–h grid (do-no-harm), E1 one-shot family (do-no-harm on terminal
arms), s1/s2/s3 stochastic (the fix). Seeds ≥ 5 (10 for the s1 finals test).

Methods: `lrq2` (no correction), `lrq2c` (the method), **cfpk** (exact-everywhere
upper bound), `lcv0`/`ppo_clip` (model-free floor). All share the braked PPO
protocol (β=0.5, KL 0.15).

Registered predictions:
- **H1 (do-no-harm):** on the a–h grid `lrq2c` = `lrq2` = 1.00 (structural gate
  ~never fires on disjoint envs; where it fires, `Î≈0` self-cancels). Falsifier:
  any grid env drops below 1.00 ⇒ the correction injects spurious bias.
- **H2 (the fix — decisive):** on s1 `lrq2c` lifts `lrq2` from **0.28 toward
  ≥0.76** (PPO) and ideally ≥0.849 (cfpk). The cross-vs-generalist inversion that
  R4 measured (credits say match>cross>gen; truth match≈gen>cross) must **flip**
  under the corrected advantage. Falsifier: s1 unchanged ⇒ either the structural
  gate misses the foreclosing decisions, or `Î_t` at the fork budget is too noisy
  to correct the ranking — both diagnosable from the per-decision logs.
- **H3 (cost):** `lrq2c` fork count ≪ cfpk's (report forks/episode); target
  < 25% of cfpk's, since only contested decisions are forked.

Cheap gate first: run **H1 on 2–3 grid envs + H2 on s1 × 5 seeds** before the
full tier. If H1 fails, the correction is unsafe; if H2 shows no movement, the
router or fork budget is the problem, not the idea — fix before scaling.

## 7. Honest fallback

If H2 shows `lrq2c` closes only part of the s1 gap, the paper is still clean:
**"cheap provenance credit + a structural diagnostic + sparse exact forks recovers
X% of the counterfactual-simulation quality at Y% of its cost"** — lineage doing
real, unique work (the cheap base + the diagnostic), with simulation reserved
for exactly where the decomposition proves it is needed. That is a defensible
efficiency result even if it does not fully match cfpk.