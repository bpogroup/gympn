# PAPER_PLAN.md — EJOR submission working plan (v2, rewritten 2026-08-03)

Target: **EJOR**. See `EJOR_PAPER_OUTLINE.md` (rewritten same day) for the
full structure and rationale for the rewrite. Framing: two provenance-
exploiting credit-assignment mechanisms (`s_ccf`, `cfpk`/DAG-replay) across a
bias/variance/granularity/cost spectrum, PLUS a rigorous, proof-backed
characterization of exactly where provenance-exploitation stops paying off
(the boundary-condition investigation: reward shaping, structural input
features, actor architecture — all independently ruled out on the
single-component bottleneck, not just found "neutral").

## Why this rewrite happened
The previous plan was ccf-centric (later patched in-place to note the
s_ccf fix). Since then, three more pieces of real, validated work
accumulated that had no home in that plan: (1) `cfpk`'s DAG-replay
counterfactual mechanism + its coupling-truncation cost proof/speedup,
(2) the structural-input-features investigation and its redundancy proof,
(3) the actor-architecture blind-spot proof + fix. None of these are
failures to write around — together with the credit-spectrum work they ARE
the paper: a positive result (two provenance-exploiting credit mechanisms,
both theoretically grounded) plus a rigorous negative result (a precise,
proof-backed account of the mechanism's boundary), which is a stronger and
more defensible EJOR contribution than a single unconditional win.

## Status (v2)
- `s_ccf` theory + realistic-env win + null + cost: **DONE** (carried over
  from v1, unchanged — this work doesn't need to be redone).
- `cfpk`/DAG-replay theory (informal) + coupling-truncation exactness +
  wall-clock speedup: **DONE** (2026-08-01/02 work).
- Boundary-condition investigation (phi-shaping, structural features, actor
  architecture): **DONE, not yet written up as a paper section.**
- P3 (coupling-truncation exactness, formalized): **TODO.**
- Draft: **NEEDS FULL REWRITE** — the old `main.tex` followed the v1 outline
  section-by-section; the narrative spine changed enough (two mechanisms +
  a boundary section, not one mechanism) that patching the old draft is not
  recommended. Treat `main.tex` as reference material for reusable LaTeX
  (the P1/P2 proof environment, the multisite/N-copies figures/tables), not
  as a draft to edit in place.

## What's fully reusable as-is (no new experiments, no rework)
- `s_ccf` implementation, N-copies sweep, multi-site realistic-env result,
  single-component null, `CCF_EXPLAINED.md` (mirrors §4a directly).
- `EJOR_PROPOSITIONS.md`'s P1 (unbiasedness) and P2 (variance) proofs — need
  retitling/light editing (subject is `s_ccf`, not `ccf`; `ccf` becomes the
  motivating-but-biased strawman in the narrative) but the mathematical
  content carries over unchanged.
- `gympn/counterfactual.py`'s DAG-replay/`cfpk` implementation,
  `_test_coupling_truncation.py`'s exact-correctness proof,
  `run_cfpk_coupling_wallclock.py`'s 1.26x speedup result.
- `_test_actor_global_context.py`'s blind-spot proof + fix, and the
  `_test_structural_features.py` redundancy proof (the one-hot argument).
- `causal-stability-suite` memory file has the full dated experimental log
  backing every claim above — use it to reconstruct exact numbers/seeds
  when writing, rather than re-deriving from scratch.

## Net-new writing needed (no new experiments required for any of this)
1. ~~**P3**: formalize coupling truncation's exactness~~ **DONE**
   (`EJOR_PROPOSITIONS.md`, new "Part II — DAG-replay counterfactual credit
   (`cfpk`) and coupling truncation" section: Assumption A3 — fingerprint
   sufficiency — plus the exact-cancellation/same-target/variance-strictly-
   lower proof, matched line-for-line against `_paired_coupled_suffixes`'s
   actual implementation, R7-R10 covering the no-CRN-alignment point, the
   measured 1.26x speedup, the value-tail-agnostic generality, and the
   lineage-mode scope gate).
2. ~~**§6b boundary-condition section**~~ **DONE** (`EJOR_BOUNDARY_CONDITIONS.md`:
   6b.0 framing, 6b.1 phi-shaping, 6b.2 structural features [+ the one-hot
   redundancy proof], 6b.3 actor global_context [+ the exact-invariance
   proof and its fix], 6b.4 synthesis, summary table, reproducibility
   pointers). Still TODO: 1-2 actual figures (e.g., a bar chart of
   norm_final across the 3 boundary attempts vs baseline, all on s1) —
   the writeup has the table, not yet a rendered figure.
3. ~~**Related work**~~ **DONE** (`EJOR_RELATED_WORK.md`: 2.1 DRL for
   dynamic task assignment, 2.2 credit assignment [RUDDER/HCA/Mesnard et
   al.], 2.3 multi-agent factored credit [difference rewards/COMA/VDN/
   QMIX], 2.4 potential-based shaping, 2.5 graph-RL architecture [new
   bucket for the actor blind-spot finding], 2.6 OR decomposition
   [Dantzig-Wolfe/Benders], 2.7 Petri nets + learning [placeholder, lowest
   priority, deferred]. Every citation web-verified directly against
   publisher/proceedings/arXiv pages, including the 3 prior-work arXiv IDs
   — the old "verify the 3 arXiv bib entries" TODO is now closed. One
   correction caught: Wolpert & Tumer is 2001, not 2002 as the v1 outline
   had it.)
4. ~~**Discussion**~~ **DONE** (`EJOR_DISCUSSION.md`: 7.1 the K-diagnostic —
   compute the causal-component count from a handful of simulated episodes
   BEFORE training to know whether s_ccf/cfpk apply at all, a genuinely
   actionable pre-training test, not just a post-hoc explanation; 7.2
   s_ccf-vs-cfpk cost/granularity trade-off + a proposed-but-untested
   combination; 7.3 the honest "what this doesn't solve" statement; 7.4
   limitations/future work [cfpk cost-knob stacking, what might close the
   bottleneck, beyond-A-E-PN generality]; 7.5 a managerial-language
   takeaway paragraph for an OR-practitioner reader specifically).
5. ~~**Full draft rewrite**~~ **DONE 2026-08-03** (`paper/main.tex`, full
   rewrite around the v2 outline; `paper/refs.bib` extended with all
   web-verified citations from `EJOR_RELATED_WORK.md`). Compiles clean with
   `latexmk -pdf` (exit 0, 21 pages, zero undefined refs/citations, only
   harmless font-substitution warnings). Structure: Intro (5 contributions)
   -> Related work (7 subsections) -> Background (unchanged + one new
   sentence on forkability, for cfpk) -> Method (4.1 s-ccf, largely reused;
   4.2 cfpk + coupling truncation, new, with a new Algorithm 2; 4.3
   positioning) -> Theory (5.1 s-ccf props, reused; 5.2 coupling-truncation
   exactness, new) -> Experiments (Q1-Q4 reused incl. figures 1-3; NEW Q4
   ccf-bias exact sign-flip table, live-verified numbers -7.606/+2.394; NEW
   Q5 coupling-truncation exactness+1.26x-speedup table) -> NEW §6
   boundary-condition section (condensed prose + summary table from
   `EJOR_BOUNDARY_CONDITIONS.md`) -> Discussion (condensed from
   `EJOR_DISCUSSION.md`, incl. the K-diagnostic) -> Conclusion -> Appendix
   (5 remarks). ~~Still open: §6b figures~~ **DONE 2026-08-03**:
   `generate_figures.py` extended with `fig4_boundary_summary` (baseline vs
   each of the 3 mechanisms, grouped bars + 95% CIs) and `fig5_phi_dial`
   (efficacy + reach-rate vs phi_coef, both degrading past 0.2) — both read
   real cell JSONs already on disk under `suite_results_{phi,struct,gctx}_s1/`
   (same convention as fig1/fig2, no hardcoded numbers), verified visually
   against the reported stats before wiring in, wired into `main.tex` §6,
   recompiles clean (22 pages). Still open: author list; final
   citation-venue confirmation for the 3 prior-work arXiv papers (flagged
   in `EJOR_RELATED_WORK.md`).

## Seed-bump DONE 2026-08-04 (user-requested after an honest "is this
submission-ready" assessment flagged uneven seed counts as a real gap).
N-copies sweep and cfpk wall-clock both bumped from SEEDS=3 to SEEDS=12
(matching multisite's already-accepted standard — reasoned this down from
the user's suggested 30: CI width only tightens ~0.63x for ~2.5x the
compute past 12). Both runs hit the "killed" background-task pattern
mid-flight (twice for cfpk specifically) before finishing on retry — same
resumable-cell-cache recovery pattern as every other interrupted run this
arc; no data lost except one batch of un-checkpointed cfpk training that
had to restart from epoch 0.

**Results, both STRENGTHENED at n=12 vs the old n=3:**
- N-copies (`suite_results_ncopies`): N=1 ppo 0.864/ccf 0.893 (tied, the
  null); N=2 0.319/0.654; N=4 0.523/0.796; N=8 0.362/0.931 — gap widens to
  +0.569 at N=8 (was +0.583 on a still-filling n=9 read, now final at
  n=12). PPO's variance stays large throughout (std 0.51/0.48/0.33) while
  ccf's tightens (std 0.10/0.41/0.27/0.10) — the "PPO degrades and
  destabilizes, ccf holds" story is, if anything, cleaner at n=12.
- cfpk wall-clock (`suite_results_cfpk_coupling_wallclock`): 36.25 ->
  26.53 min/cell, **1.37x speedup** (up from 1.26x at n=3), norm_final
  EXACTLY 1.000 vs 1.000 across all 12 seeds (was n=3) — a much stronger
  demonstration of Prop 3's exactness claim.

**`main.tex` updated**: Table~tab:coupling numbers (36.25/26.53/1.37x/n=12),
both abstract+intro `1.26x` -> `1.37x` mentions, `fig2_ncopies_scaling`
regenerated from the new cell data (fig3_mechanism is independently
computed live, not cell-JSON-based, so untouched by this bump). Recompiles
clean. N-copies text-table in the old outline (0.09->0.84 mechanism
divergence) is a SEPARATE statistic (`|ccf-mcq|/scale` from `fig3`'s own
live computation, not norm_final) and was correctly left alone.

## Explicitly dropped from v1 (do not resurrect without a reason)
- The "clean monotone-scaling law" ask — v1 already noted this was elusive
  and recommended against over-claiming; still true, still dropped.
- Difference-rewards baseline (optional stretch in v1) — still optional,
  lower priority than closing out §6b, which is free (data already exists).
- LS-HCA, LCV/LVA, MCTS-over-A-E PN — remain out of scope, mentioned at most
  as related/future work, per the outline's explicit scope note.

## Recommended order of work
1. ~~P3 proof~~ **DONE**.
2. ~~§6b writeup~~ **DONE**; figures still open (low effort, 1-2 simple
   bar/table charts from data already in the writeup).
3. Related work + Discussion rewrite.
4. Full draft.

No new experiments are required to reach a complete draft under this outline.
