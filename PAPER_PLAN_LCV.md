# Paper plan: LCV — lineage as measured side information (target: AAMAS, fallback: ICAPS)

Supersedes `PAPER_PLAN_LRQ.md` (2026-07-15 pivot). Fourth paper in the line:
A-E PN framework (BPM) → GNN extension (BPM) → gympn + expansion algorithm
(EDOC) → **this paper: the token lineage as learning signal**. The prior papers
established the *environment machinery*; this one is a *learning* contribution.

**Main method: LCV** (`CAUSAL_LCV_CONTROL_VARIATE.md`). **Hypothesis to
confirm: the token-lineage trace of an executable process model carries
decision-relevant information that usefully guides policy learning.**

---

## 1. Venue: AAMAS first, ICAPS as the chronological fallback (decided — unchanged)

**Primary: AAMAS** (deadline historically early/mid-October — verify the CFP).
Credit assignment is a first-class AAMAS topic (difference rewards, COMA,
Shapley Q-values); the resource-allocation setting reads naturally there, and
"the executable model gives you for free what HCA has to learn" is an
agents-venue argument. Page budget historically 8 pp + references.

**Fallback: ICAPS** (~December, after AAMAS ⇒ costs no time). Pitch shifts to
"white-box SMDP scheduling: exploit the model's causal structure in the
learner". Keep one scale experiment shaped so it doubles as an answer to
"where are the classical scheduling benchmarks?".

Action now: pull both CFPs, pin exact dates and page budget.

## 2. The hypothesis, made falsifiable

**H.** The token lineage — which future rewards each decision causally
produced, read exactly off the A-E PN trace — is *useful side information for
learning*, not just interpretable annotation.

"Useful" is operationalized through LCV, the estimator whose entire content is
that information (`A_lcv = A_GAE − ĉ·(R_off − v_off(s))`, standard PPO
otherwise). H splits into four registered sub-hypotheses, each with its own
experiment and its own way to fail:

- **H1 (mechanism).** The lineage-measured control variate genuinely correlates
  with the GAE advantage: ĉ > 0 and the advantage variance on identical batches
  drops by a measurable fraction. *This is the mirage-standard evidence
  (Tucker et al.): report variance, not just returns.* Fails if ĉ ≈ 0
  everywhere or the variance reduction is negligible.
- **H2 (outcome).** Where cross-case/concurrency noise binds, the variance
  reduction converts into faster convergence and/or higher final return vs
  paired PPO; where discrimination binds, LCV ≥ PPO (the floor — no price
  paid). Fails if LCV loses to PPO anywhere (floor violated) or never beats it
  (information present but useless).
- **H3 (specificity — it is the *lineage*, not the estimator form).** The
  ablation that keeps everything but drops the lineage restriction (`mc_q`:
  full-return Q-sample, identical consumption) shows no gain over PPO —
  already established (final 0.695 vs 0.690, p=.95, on the deterministic
  suite). The direct consumer of the *same* lineage (`lrq2`) shows the
  information is strong enough to win big where its bias is inert (s3: 0.93)
  and to lose big where it is not (s1: 0.28). LCV keeps the information and
  drops the bias. Fails if mc_q matches LCV's gains.
- **H4 (dose–response).** Across the full env grid, the paired improvement
  over PPO correlates with the *measured* per-env variance reduction
  (ĉ²·Var(cv) relative to Var(A)), interacted with how noise-bound the env is.
  One scatter plot = the hypothesis test in a single figure. ⚠️ Open tension
  to resolve honestly (see §5, X4): the first diag seed shows the *inversion*
  — s1 has the larger measured reduction (ĉ=0.49, ~15%) but no return gain,
  s3 the smaller (ĉ=0.19, ~4%) but the clear win. Working explanation:
  variance reduction only pays where noise, not discrimination, is the
  binding constraint — that interaction term is part of H4, not a bug in it.

## 3. The story (one paragraph)

Delayed, entangled rewards across concurrent cases make actor-critic learning
slow in timed process models. Executable models offer an unusual asset: the
*exact* causal lineage of every reward. The tempting interfaces for that
asset — redistributing rewards backward, or replacing the advantage with a
lineage Q-sample — trust it too much: they inherit a bias (foreclosure:
off-lineage action dependence) that is invisible where concurrency dominates
and fatal where discrimination between competing commitments matters (our
lrq2 evidence: 0.93 vs 0.28 on two environments of the same tier). We propose
the interface that trusts it exactly as much as the data says to: a
**measured control variate**. LCV subtracts from the standard SMDP-GAE
advantage the *measured* off-lineage return, centered by a learned state
head and scaled by the classical optimal coefficient ĉ estimated per epoch.
ĉ→0 recovers PPO exactly (the floor is the estimator's limiting case, not a
knob); under off-lineage independence the correction is unbiased and removes
up to the entire cross-case noise; when independence fails the bias is
bounded and gated by the same ĉ. Unlike Q-Prop-style *learned* variates —
deflated by the "mirage" critique — our variate is *measured* from the trace;
only a scalar and a centering are estimated, and we report the variance
reductions themselves, not just returns. On stochastic scheduling
environments LCV is the first lineage consumer that pays nothing where the
lineage cannot help (s1: 0.72 vs PPO 0.76, tighter seed spread) and captures
the direct-credit gain where it can (s3: 0.90 vs 0.79, at the level of the
biased direct consumer).

## 4. Contribution claims

- **C1 (method).** LCV: the lineage control variate. Definition, the three
  properties (PPO floor by construction; quantified unbiased benefit under
  off-lineage independence; characterized, ĉ-gated bias under foreclosure),
  and the collapse lemma — with pure MC advantages (λ=1) the estimator
  algebraically collapses to the direct consumer; the novelty lives on the
  bootstrapped GAE. Positioning: a *measured* CV as the constructive answer
  to the learned-CV mirage critique, made possible by the trace.
- **C2 (hypothesis, mechanism level).** ĉ trajectories and per-epoch
  advantage-variance reductions across the grid (H1) — the measurement CV
  papers are asked for and usually don't give.
- **C3 (hypothesis, outcome level).** Paired suite study: floor holds
  everywhere, gains where concurrency noise binds (H2), with the
  dose–response figure (H4) tying C2 to C3.
- **C4 (specificity + interface argument).** The ablation ladder on identical
  envs/config: mc_q (no lineage) ≈ PPO; lrq2 (all-in lineage) high-variance
  across envs — big wins, big losses; LCV (measured-dose lineage) = floor +
  wins. The general lesson for white-box models: *side information should
  enter as a control variate, not as the estimator* (H3).

Framing discipline: C1 sells to method reviewers, C2 to the
variance-reduction literature, C3 to empiricists, C4 is the take-home. Do not
oversell: LCV does not beat the direct consumer where the latter is unbiased
(s3: 0.90 vs 0.93) — it matches it *without knowing in advance that the env
is safe*, which is the actual deployment condition.

## 5. Experiment matrix (have vs need)

Method set for the paper: **lcv** (main), **ppo_clip** (floor/pair, kept in
its community-standard form: γ=0.99 per decision), **lcv0** (the ĉ=0 twin:
SMDP-GAE PPO with the per-sojourn discount e^{−βτ} and NO CV term — LCV's
exact limiting case), **mc_q** (lineage ablation), **lrq2** (direct
consumer), **rudder** (learned redistribution).

**lcv0 IMPLEMENTED 2026-07-15** — not a new causal scheme; a `smdp_discount`
boolean flag (`TrajectoryBuffer`, `Agent`, CLI `--smdp_discount`) that routes
the STANDARD (non-`causal_rl`) advantage computation through `smdp_gae` with
`e^{−causal_beta·τ}` instead of `compute_advantages` with constant `gam`.
`lcv0` = `ppo_clip` + `smdp_discount=True`, `causal_rl=False` — reuses the
same decision-clock plumbing the causal branch already relies on (confirmed:
`agents.py` records `decision_time` unconditionally, both paths share one
clock). Guards: raises loudly if `causal_beta>0` and all sojourns in an
episode are 0 (decision times never recorded) instead of silently
degenerating to an untimed GAE; `gam` is provably unused on this path (the
branch never reads it). Suite: `run_suite.py` recognizes method `"lcv0"`
(`_make_args`); not added to `ALL_METHODS`/`stoch_config` defaults yet — set
`cfg.methods` explicitly per the `run_lcv_s2.py` pattern. Tests added to
`_test_lrq.py`: `smdp_gae` with constant discount ≡ `compute_advantages`
with matching γ; the buffer flag path ≡ `smdp_gae` called directly; lcv0's
advantages ≡ LCV's own perfectly-centered floor case (same numbers, two code
paths); the silent-fallback guard raises. All pass. End-to-end smoke (2
epochs, `a_sequence_joint`) confirmed the full CLI→agent→buffer path trains
without error. **Not yet run for real**: X0 (lcv+lcv0 on the a–h grid) and
X1 lcv0 (stochastic tier) are still open per §5/§9. Dropped from the paper: lrq3/lqi (negative results, recorded
in `CAUSAL_LQI_QNATIVE.md`, cite as motivation), μ-hedge (belonged to
LRQ-as-main; also its pre-fix results acted through an accidental mechanism —
see X6). Plus the scripted credit-greedy heuristic as a no-training anchor.

| # | Experiment | Status | Notes |
|---|---|---|---|
| X0 | Deterministic 8-topology suite × 6 methods × 10 seeds, braked protocol | **baselines DONE** (suite_results_paper, 2026-07-10: lrq 80/80 optimum reach, drift 0.00; conv 6.2–6.8 vs ppo 9.6 vs rudder 9.4; mc_q ≈ ppo) — **lcv AND lcv0 cells QUEUED 2026-07-15** (`run_x0_lcv.py 4`, methods=[lcv,lcv0] × 8 envs × 10 seeds = 160 cells into `suite_results_paper`; chained to start after the X1 stoch run finishes, to avoid oversubscribing the 8 physical cores — see `examples/paper_examples/suite/run_x0_lcv.log` once it starts). Run lcv + lcv0 × 8 envs × 10 seeds into the same protocol/dir. Prediction (register): floor everywhere; gains on the concurrent topologies (c/d) where exploration noise across cases is off-lineage; ĉ ≈ 0 on strict sequences (a/b) — which is itself H1 evidence |
| X1 | Stochastic tier s1/s2/s3 × methods × 10 seeds | **s1, s3 DONE for lcv** (2026-07-14: s1 0.72±0.11 vs ppo 0.76; s3 0.90±0.16 vs 0.79); other methods have 10/10 cells — **s2×lcv LOST** (run died in the 2026-07-14 restart before the first cell); **lcv0 cells missing on all three envs** — **RUNNING 2026-07-15** (`run_lcv0_stoch.py 4`, methods=[lcv,lcv0] × 3 envs × 10 seeds = 60 cells into `suite_results_stoch`, resumable so s1/s3×lcv are skipped; log at `examples/paper_examples/suite/run_lcv0_stoch.log`) | Relaunch `run_lcv_s2.py` first. s2 is also where lrq collapsed historically (postpone re-emission) — LCV's floor prediction is the strong test there |
| X2 | Mechanism instrumentation (H1/C2): ĉ + variance-reduction curves persisted per cell | **instrumented 2026-07-14** (diag cells carry `cv_coef_curve`, `cv_var_reduction_curve`); only 1 seed × {s1, s3} measured; the pre-instrumentation lcv cells lack curves | Make the curves persist in **every** lcv cell, then X0/X1 reruns produce C2 data for free. Diag numbers recovered post-restart: s1 ĉ=0.486, vred 14.9%; s3 ĉ=0.188, vred 3.9% |
| X3 | Dose–response figure (H4): per-env paired Δreturn vs measured variance reduction | **not started** — depends on X0+X1+X2 | The single most important figure. Include the noise-boundness moderator (e.g. paired PPO seed-variance as the x2 axis or point size) |
| X4 | The inversion: why does s1 show more variance reduction but less gain? | **OPEN — must be answered before submission** | Candidate analyses: (a) variance reduction relative to the *total* gradient noise (advantage-scale normalization), (b) s1's binding constraint is discrimination (lrq2 0.28 proves the lineage signal misleads there), so removed noise was not the limiting factor, (c) centering-head quality differs. Whatever the answer, it becomes a paragraph of C2 — honest mechanism papers report the moderator |
| X5 | Foreclosure stress (the characterized failure, C1): env where off-lineage independence fails hard | **env EXISTS** — s1_stoch_sequence is the documented foreclosure env (2026-07-13) | Measure the realized bias: E[R_off|s,a]-dependence diagnostic + show ĉ stays low/clipped and returns stay at the PPO floor. This is the pre-answer to the obvious objection |
| X6 | Ablations of LCV's own parts | **not started** | (i) **the ĉ=0 twin `lcv0`** (decided 2026-07-15): per-sojourn-discounted GAE PPO, no CV — runs on the FULL grid (see X0/X1), giving the exact factorization [lcv − lcv0] = the CV term alone, [lcv0 − ppo_clip] = the discount alone. **Registered prediction: lcv0 ≈ ppo_clip** (mc_q with the wall-clock discount already landed on ppo exactly, p=.95); if instead lcv0 > ppo, promote lcv0 to the headline pairing — compare against the strongest baseline. Either outcome kills the discount confound structurally; (ii) centering: v_off head vs batch-mean vs none; (iii) ĉ: adaptive vs fixed ∈ {0.5, 1} vs clip range — shows the adaptive coefficient is doing work; (iv) zero-reward-fix regression note (the 2026-07-14 `AEPN_Env.step` bug made all pre-fix raw-reward paths silently zero — any old number reused in the paper must postdate the fix) |
| X7 | Eval-protocol hardening | **to verify** | ≥5 greedy episodes per eval point (memory: judge by deterministic eval, ≥5 seeds). Confirm it is on in the braked protocol before the paper-final rerun; if not, fix BEFORE X0's lcv run so cells are comparable |
| X8 | Scripted credit-greedy anchor on all envs | **exists on s1 only** (14.8, zero training — discovered 2026-07-14) | Cheap; one column in the results table, preempts "would a heuristic do it?" |
| X9 | **LVA — lineage as auxiliary critic target** (the response to the 2026-07-16 lcv≈lcv0 result) | **IMPLEMENTED + RUNNING 2026-07-16** (`run_lva.py 4`: 30 stoch cells then 80 grid cells, ~10h; log `run_lva.log`) | Scheme `'lva'`: policy path identical to lcv0 (plain SMDP-GAE PPO, no CV); the critic gains a second scalar head on its shared HGT encoder (`HeteroCritic(aux_head=True)`) regressed on the per-decision lrq2 lineage credit, weight `causal_aux_coef=0.5`. Lineage bias stops at the encoder — the value head keeps its unbiased GAE target, the policy gradient is exactly PPO's (floor by construction), yet the critic consumes the FULL per-decision credit vector instead of LCV's one epoch-scalar ĉ. **Registered predictions (in `run_lva.py`, written pre-run): P1 floor — lva ≈ lcv0 everywhere, s1 must NOT collapse (value-side consumption can't bias the policy; contrast lrq2's 0.28); P2 gain — convergence faster than lcv0 where lrq2's speed edge was largest (s2: 3.6 vs 16.4 ep; s3: 6.8 vs 15.2); P3 honest falsifier — lva ≈ lcv0 everywhere ⇒ representation-level consumption also insufficient ⇒ strengthens the trade-off framing.** Verified pre-launch: floor unit test (lva advantages ≡ smdp_gae ≡ lcv floor ≡ lcv0 — the triangle closes), aux-targets test, aux-head network test, and an instrumented end-to-end check that every value batch actually trains through the aux loss (16/16, zero silent fallbacks) |

## 6. Paper skeleton (8 pp AAMAS format)

1. **Introduction** (1 pp) — delayed entangled credit in timed processes; the
   lineage asset; the interface question (redistribute? replace? correct?);
   H; contributions C1–C4.
2. **Background** (1 pp) — A-E PN as white-box SMDP (cite prior papers), token
   lineage traces, SMDP actor-critic with per-sojourn discount e^{−βτ}.
3. **Consuming the lineage: the interface ladder** (1 pp) — direct consumers
   and their foreclosure bias (lrq2 definition + the s1/s3 contrast as the
   motivating table); one paragraph on why backward redistribution is not
   pursued (per-step soundness fails; cite, don't prove — this paper is not
   about the trap).
4. **LCV** (1.5 pp) — estimator, the three properties, collapse lemma,
   measured-vs-learned CV positioning (mirage critique), implementation notes
   (epoch-level ĉ, v_off centering head, [0,2] clip).
5. **Experiments** (2.5 pp) — X0+X1 paired tables; C2 mechanism curves; the
   X3 dose–response figure; X5 foreclosure stress; X6 ablations.
6. **Related work** (0.75 pp) — control variates in PG (Q-Prop, Stein, the
   mirage critique — our variate is measured, not learned); RUDDER/return
   decomposition; HCA/counterfactual credit (the lineage IS the hindsight
   distribution, given by the model); COMA/difference rewards/Shapley credit;
   SMDP-RL; DRL for business-process scheduling.
7. **Conclusion + limitations** (0.25 pp) — benefit bounded by off-lineage
   noise share (ĉ≈0 ⇒ lineage bought nothing, by design); same-data ĉ
   estimation (O(1/n)); white-box requirement; one extra critic head.

Appendix (arXiv version): collapse lemma proof, per-env curves, the
lrq3/lqi/μ negative results as the recorded path to the CV interface.

## 7. Anticipated objections → planned answers

1. *"CV gains are a mirage (Tucker et al.)."* — Frontal answer: the variate is
   measured, not learned; we report ĉ and per-batch variance reductions
   (C2/X2), which is exactly what that critique demands. This objection is
   our positioning, not our weakness.
2. *"It's just PPO with extra steps where ĉ≈0."* — Yes: that is the floor
   property, and it is the honest deployment story — you don't know in
   advance which regime your env is in; LCV never pays for finding out
   (contrast lrq2's 0.28).
3. *"The bias under foreclosure."* — Bounded, ĉ-gated, and empirically probed
   on the worst-case env we know (X5); contrast with direct consumers that
   carry it unbounded.
4. *"ĉ estimated on the data it corrects."* — Classical O(1/n) CV-estimation
   bias, negligible at ~600 steps/epoch, clip guards the tail; ablation X6(ii).
5. *"Environments are small."* — Topology grid isolates structural effects;
   the stochastic tier adds arrival noise, scale (s2: 3×6, 3 arrivals/t) and
   oversaturation (s3); anchors are strong myopic heuristics, normalized >1.0
   = beating the anchor (s3 seeds hit 1.16–1.18).
6. *"Why not learn the CV end-to-end?"* — lrq3/lqi negative results: the
   learned off-lineage head loses the cold-start race and corrupts advantages
   before converging; the measured variate has no cold start.
7. *"More variance reduction should mean more gain — your s1/s3 numbers
   invert."* — X4's moderator analysis; must be written up before submission.
8. *"Your methods discount by elapsed time but the PPO baseline uses γ per
   decision — apples to oranges."* — (a) e^{−βτ} is the standard SMDP
   discount (Puterman; Bradtke & Duff), the correct generalization of γ^t to
   variable sojourns — on unit-time steps it IS γ^t; (b) evaluation is
   undiscounted and shared, so each method gets its own training
   discount as a hyperparameter (standard practice); (c) the confound is
   factored out structurally: mc_q (same discount, λ=1) ≈ ppo, and lcv0
   (same discount, same λ, no CV) closes the last gap — state the
   [ppo → lcv0 → lcv] decomposition explicitly. On untimed problems (X5
   knapsack env) the discount is inert by construction, so any gain there
   cannot come from discounting at all.

## 8. Timeline (working back from an early-October AAMAS deadline; today 2026-07-15)

- **now → end July** — commit the working state (zero-reward fix + LCV are
  uncommitted!). Relaunch s2×lcv (X1). Persist CV curves in every cell (X2),
  verify eval protocol (X7), then the X0 lcv run on the deterministic suite.
  Register the X0 predictions in this file before looking at results.
- **August** — X3–X6 analyses and ablations; scripted anchor everywhere (X8).
  **Paper-final rerun** (one shot, 5 methods × 11 envs × 10 seeds, hardened
  protocol — budget ~2–3 days of compute). Freeze code, tag the artifact,
  prepare the anonymized drop (gympn is public ⇒ reproducibility badge is
  cheap).
- **September** — write. Method + collapse lemma first (a hole there forces
  experiment changes — front-load). Figures: paired convergence curves with
  CI bands, the X3 scatter, ĉ/variance-reduction curves, the interface-ladder
  table. Internal co-author reviews mid-month.
- **Early October** — buffer + submit AAMAS. Miss ⇒ ICAPS in December with
  the scheduling framing promoted (s-tier becomes the headline; C1 stays).

## 9. Immediate next actions (this repo, this week)

1. ~~Commit the current state~~ — offered 2026-07-15, user deferred ("I'll
   commit myself later"). **Still outstanding and higher-risk now**: two
   multi-hour background runs (below) are training against this uncommitted
   tree; a repeat of the 2026-07-14 crash would lose the code, not just the
   run. Revisit before the next long run if not done by then.
2. Verify AAMAS/ICAPS CFP dates and page budgets.
3. ~~Relaunch `run_lcv_s2.py`~~ — superseded: launched the broader
   `run_lcv0_stoch.py` instead (covers the lost s2×lcv run plus lcv0 on all
   three envs in one resumable pass). **RUNNING since 2026-07-15 13:15.**
4. ~~Implement `lcv0`~~ — **DONE** (`smdp_discount` flag; see §5 X6). Job
   order launched 2026-07-15: (a) `run_lcv0_stoch.py 4` — 60 cells,
   `suite_results_stoch`; (b) chained after (a) completes,
   `run_x0_lcv.py 4` — 160 cells, `suite_results_paper`. Chained rather than
   parallel to stay within the tuned physical/2=4-worker budget (running
   both suites concurrently would 2× oversubscribe the 8 physical cores).
   CV-curve persistence confirmed already wired into the main training loop
   (`agents.py` ~L342), not just the diag script — no extra work needed there.
5. X0 registered predictions are already in §5 (written before the run
   started, per the falsification discipline).
6. Decide co-author list and who owns the collapse-lemma polish vs experiments.
7. Once both runs land: X4 (the s1/s3 variance-reduction inversion) and X6
   (ablations) are next — they need the fuller ĉ/variance-reduction dataset
   X0/X1 will produce before those analyses are meaningful.

## 10. Relation to the prior papers (and double-blind hygiene)

Cite all three in third person: A-E PN formalism + expansion algorithm
(BPM'1, EDOC), HGT actor/critic encoding (BPM'2), gympn implementation
(EDOC). This paper assumes the environment layer and contributes the learning
layer. Self-containment: one background page re-explains A-E PN as a
white-box SMDP with token-lineage traces — the only piece the method needs.

## 11. Document map (post-cleanup, 2026-07-15)

- `CAUSAL_LCV_CONTROL_VARIATE.md` — the method (design, properties, results log)
- `CAUSAL_LRQ_PROPOSAL.md` — LRQ/lrq2 definition (baseline + the R_off machinery LCV reads)
- `CAUSAL_LQI_QNATIVE.md` — the 2026-07-14 rethink; lrq3/lqi/μ negative results (motivation + appendix material)
- `CAUSAL_GAP_D_FORECLOSURE.md` — foreclosure as combinatorial structure (feeds C1's bias characterization + limitations)
- `CAUSAL_RELATED_WORK_COMA_RUDDER_SHAPLEYQ.md` — related-work notes
- `docs/` — library documentation (untouched)

Removed 2026-07-15 (redistribution-era, superseded by the pivot): the
REDISTRIBUTION_* series, REC review, TD0/GAE notes, minimal-causal-cut doc,
Shapley-vs-flow results, instability analysis (resolved, fixes live in the
code), visual explanation, the sound-scheme LaTeX draft, and PAPER_PLAN_LRQ.md
(this file absorbs its venue decision, prior-paper hygiene, baseline numbers,
and timeline discipline).