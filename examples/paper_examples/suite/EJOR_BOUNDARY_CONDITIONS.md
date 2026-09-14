# §6b — The boundary-condition investigation (draft)

*Draft for §6b of the EJOR paper (v2 outline, see `EJOR_PAPER_OUTLINE.md`).
Companion to `EJOR_PROPOSITIONS.md` (§5's theory): where that file proves
`s-ccf` and `cfpk` are safe, this file writes up the investigation into
whether the honest null both mechanisms predict on a single-component
bottleneck (s1) can be closed by any other provenance- or structure-based
route. All experiments below are already run; this is a write-up, not a
request for new results. Every number is reproduced from the
`causal-stability-suite` memory log and the underlying suite scripts/tests
named inline, so it can be checked against the raw cell JSONs, not just
this summary.*

---

## 6b.0 Framing

§5's theory makes a sharp prediction: `s-ccf`'s variance reduction scales
with the number of causally independent components $K$ (Proposition 2), and
collapses to *exactly* PPO when $K=1$ — a single shared resource pool links
every case into one component (Remark R1). This is not a failure of the
method; it is the theorem's own honest null. `s1_stoch_sequence` — a
two-stage service line with a single, skill-heterogeneous, 3-employee pool
shared by both stages — is exactly this case: every case's reward is
causally linked to every other case through the shared pool, so $K=1$ and
`s-ccf`/`ccf` reduce to plain PPO by construction. `cfpk`'s exact
counterfactual replay does not have this particular collapse, but was not
separately re-validated as an s1 improvement within this paper's scope (its
in-scope experiments target a contested-fork topology for the coupling-
truncation cost result, §6a; see the scope note in §0 of the outline for
why the forking-based mechanisms that *were* built for s1 specifically —
`lrq2c`'s foreclosure-gated forking — are excluded from this paper).

This raises the natural question a reviewer would ask unprompted: is s1's
floor under plain PPO actually the best any provenance- or structure-aware
mechanism can do, or is it an artifact of *which* mechanisms happened to be
tried? We tested every mechanism we could construct that only changes what
information the learner has access to — never what it is credited for —
and found each one either provably inert or empirically neutral. We report
all three below not as a coda but as a result: it precisely localizes s1's
remaining difficulty to something other than credit attribution or
observability, which narrows the hypothesis space for whoever takes up the
open problem next.

**Common protocol.** All three experiments below use `s1_stoch_sequence`
under the paper's standard stochastic-suite budget (30 epochs $\times$ 20
episodes/epoch, 10 seeds unless noted), `ppo_clip` as the baseline arm, and
report `norm_final` (greedy final return normalized so 0 = random policy,
1 = the domain heuristic), paired seed-by-seed against the baseline, with a
paired t-test.

---

## 6b.1 Reward shaping does not close the gap

**Mechanism.** Potential-based reward shaping (Ng, Harada & Russell, 1999),
SMDP-generalized: $F(s,a,s') = e^{-\beta\tau}\Phi(s') - \Phi(s)$ added to
every step's reward, for $\Phi(s) = \sum_p \text{decay}^{\text{hop}(p)}
\cdot\min(|p.\text{marking}|,\text{cap})$ — a distance-decayed structural
backlog, weighted by each place's hop-distance to the nearest reward
transition (`gympn/potential.py`). This is a genuinely different kind of
lever than every credit-restriction mechanism in §5: it never decides what
reward belongs to which decision, so none of `ccf`'s bias mechanism
applies. It targets learning *speed*, not estimator bias or variance, and
its safety is a theorem, not an empirical claim: for *any* potential
function, the shaping term telescopes to a state-independent constant along
every trajectory and provably cannot change the optimal policy. We verified
this in code, not just cited it: the terminal-`$\Phi$`-zero convention was
tested to force $G_{\text{shaped}}-G_{\text{raw}} = -\phi_{\text{coef}}
\Phi(s_0)$ *exactly*, on both a natural-completion and a
truncated episode (`_test_phi_shaping.py`, 6/6).

**Result.** A full coefficient dial on s1 (same cached PPO baseline reused
across all four points, `run_phi_shaping_s1_full.py`):

| $\phi_{\text{coef}}$ | norm\_final | reach-95%-of-anchor |
|---|---|---|
| 0.0 (baseline) | 0.739$\pm$0.10 | 6/10 (17.0 epochs) |
| 0.2 | 0.766$\pm$0.10 ($p=.48$) | 8/10 (19.0 epochs) |
| 0.5 | 0.697$\pm$0.16 ($p=.48$) | 4/10 (27.5 epochs) |
| 1.0 | 0.420$\pm$0.10 (0W/10L) | 1/10 (10.0 epochs, $n=1$) |

Reach-rate is cleanly monotonic in the injected-variance direction past
0.2 (8$\to$4$\to$1 seeds as the coefficient rises); no coefficient produces
a statistically significant win, and the safety theorem holds throughout
(no run ever changes what the optimal policy *is*, only how fast/reliably
training finds it).

**Why.** s1 has constant exogenous arrivals (a new case every time unit,
independent of the policy), so $\Phi$ does not monotonically decrease as
work completes — it also rises every time a new case arrives. $\Phi(s_0)
\approx 4.99$ is the same order of magnitude as a full episode's raw return
($\approx 13$–$14$), so at $\phi_{\text{coef}}=1.0$ the shaping term injects
a large, arrival-driven, per-step signal that is unbiased in aggregate
(guaranteed by the theorem, confirmed by the exact-cancellation test above)
but a substantially noisier value-regression target at a finite training
budget — working against, not for, sample efficiency. This is a
theorem-safe, empirically-negative result: a materially different failure
mode from anything in §5, where the risk was always *bias* from a wrong
partition, not *variance* from an unbiased-but-noisy signal.

---

## 6b.2 Structural input features are provably redundant, not just neutral

**Mechanism.** Rather than touch credit at all, append static,
conflict-graph-derived scalars to each action node's own input features —
plain extra input to the graph encoder, with no bias-variance tradeoff to
prove (`GymProblem.use_structural_features`, `gympn/simulator.py`). Two
features were tried: `(in\_conflict, degree)` — does this action type
structurally compete for a token with another type, and with how many
(`gympn.conflict_graph.analyze`) — and, after a first null result prompted
a deeper feature, `reward\_proximity` — the minimum topology weight
($\text{decay}^{\text{hop}}$, reusing §6b.1's hop-distance machinery) among
an action's own required input places, motivated by a genuine structural
asymmetry: s1's two action types, `start1`/`start2`, sit at different
pipeline depths (`waiting1` is 4 hops from the reward, `waiting2` is 2).

**Result.** Both versions are statistically neutral on s1:

| feature set | norm\_final (plain / +struct) | paired diff | seeds |
|---|---|---|---|
| `(in\_conflict, degree)` | 0.739$\pm$0.10 / 0.714$\pm$0.11 | $-0.025$, 4W/6L, $p=.58$ | 10 |
| $+\,$`reward\_proximity` | 0.728$\pm$0.10 / 0.704$\pm$0.14 | $-0.024$, 3W/5L, $p=.69$ | 8 |

**Why — and this is the stronger claim.** The first feature's failure has a
clean, checkable explanation that is *not* a bug: `start1` and `start2`
share exactly one input place (the employee pool), so they get exactly one
conflict edge between them, and graph degree on a single mutual-conflict
edge is mathematically forced to be equal on both ends — $(1.0,1.0)$ for
both, correctly. `reward\_proximity` was added specifically to break this
symmetry (confirmed to work as intended: `start1`$=0.656$ vs
`start2`$=0.81$, `_test_structural_features.py`), yet training outcomes
stayed unchanged. The reason is architectural, not statistical: every
`a\_transition` node already carries a one-hot action-*type* identity
(`start1=[1,0]`, `start2=[0,1]`; `add\_self\_loops=True` is the hardcoded
default at every real observation call site), so the two types are already
**fully distinguishable at zero hops**, with or without any structural
feature. Any scalar computed as a function of the action *type alone* —
conflict degree, reward proximity, or any other such feature — is
therefore a deterministic function of information a single learned weight
on the one-hot could already reconstruct; it cannot add discriminative
power a linear readout could not already express, regardless of how the
feature is designed. This is provable from the encoder's construction, not
merely observed across two experiments — and it forecloses the entire
class of static, per-type structural features as a productive direction on
this codebase's architecture, not just on s1.

---

## 6b.3 A real, provable actor-architecture blind spot — closed, still neutral on s1

**Motivation.** §6b.2's redundancy proof only rules out *static, per-type*
features. It leaves open whether the *actor* can even condition a decision
on relevant state elsewhere in the graph at all — a question about the
network architecture, not feature engineering.

**Finding.** `HeteroActor` decodes each action's logit from *only* that
action node's own final message-passing embedding (`gympn/networks.py`) —
unlike `HeteroCritic`, which has always max-pooled over every
action/postpone node's embedding to compute its state value. Combined with
`get_graph_observation`'s edges being directed strictly along token flow
(place$\to$transition for consumed places, transition$\to$place for
produced places; reverse edges are never added at any real call site), an
action's logit can depend on state *only* via a forward, token-flow-
direction path reaching it within the network's depth (3 layers, the
suite's default). This is a hard architectural fact, not a training-budget
limitation, and we proved it as such rather than asserting it: on a
purpose-built topology of two structurally independent two-stage pipelines
(no shared resource between them, so one pipeline's first-stage action has
*no* directed path — at any depth — to the other pipeline's second-stage
state, while a sibling action within the *same* pipeline does), perturbing
the unreachable pipeline's downstream marking left the first action's raw
logit **exactly, bit-for-bit unchanged** under a randomly initialized,
untrained network ($\Delta=0.0$) — reachability is a property of the
computational graph, independent of learned weights, so this had to hold
even before training. On s1 itself, the same perturbation produced a small
but nonzero change ($\Delta\approx0.002$): s1's employee pool happens to
recycle (`busy2$\to$done2$\to$employee$\to$start1`, a 3-hop path, exactly
at the network's own depth), an accidental narrow backdoor, not a general
guarantee.

**Fix.** `HeteroActor(global\_context=True)`: concatenate the same pooled
action/postpone context `HeteroCritic` already computes onto each node
before decoding (opt-in, default off, byte-identical when disabled). On the
same purpose-built proof topology, this took the blind spot from exactly
$0.0$ to a real, substantial sensitivity ($\Delta=0.528$), via the pooled
context routing information through the sibling action's own
1-hop-sensitive embedding — confirming the fix closes the gap it was built
to close, deterministically (`_test\_actor\_global\_context.py`, 5/5;
`tests/` 27/27 green, no regressions).

**Result on s1.** `run\_actor\_global\_context\_s1\_full.py`, same 10-seed
budget: norm\_final $0.739\pm0.10$ (baseline) vs $0.746\pm0.14$
(+global\_context), paired diff $+0.007$, 7W/3L, $p=.913$ — no significant
effect, and higher variance. Convergence speed is likewise unmoved
(17.0$\pm$3.8 vs 18.7$\pm$4.9 epochs to 95% of anchor, both 6/10 seeds).

**Why this is not a contradiction of the proof.** The deterministic proof
and the training result answer different questions. The proof shows the
blind spot is real and that the fix closes it — a fact about the
architecture, independent of any specific environment. The training result
shows that *on s1 specifically*, the blind spot was never total (the
accidental 3-hop pool backdoor already leaked a little of the same signal)
and that whatever fraction of it the fix newly exposes is not, in
practice, decision-relevant enough to move a well-tuned PPO baseline at
this budget. Both are true simultaneously, and reporting only one would
overstate or understate the finding.

---

## 6b.4 Synthesis

Three independent, principled, individually well-motivated attempts to give
the learner more or better *information* — denser reward signal
(§6b.1), richer static input features (§6b.2), and a strictly more
observant actor architecture (§6b.3) — all land neutral or worse on s1's
single-component bottleneck, and in two of the three cases we can say *why*
with a proof rather than a correlation: reward shaping's safety and its
harm are both theorem/test-verified, not just measured; structural
features' redundancy follows deductively from the one-hot encoding already
present; the actor's blind spot and its fix are established by exact
invariance under untrained weights, the cleanest test available short of a
formal architectural argument. None of this is a failure of experimental
design — each mechanism does exactly what it was built to do. It is
evidence about where s1's difficulty does *not* live: not in what the
learner can observe, and not (per §5's Remark R1, which predicts and
confirms `s-ccf`'s exact collapse to PPO here) in an avoidable credit-
assignment bias on this specific, genuinely single-component topology. The
paper states this as an open problem rather than a solved one: whatever
limits learning on a tightly-coupled, same-type, shared-resource
contention bottleneck likely concerns exploration or value estimation
*under* contention, not information availability — a direction distinct
from, and out of scope for, the provenance-exploiting credit mechanisms
this paper contributes.

---

## Summary table (for the section's opening or closing figure)

| mechanism | theoretical status | s1 result | why it's closed |
|---|---|---|---|
| Reward shaping ($\phi$-shaping) | Provably policy-invariant (any $\Phi$) | Neutral (0.2) to decisively harmful (1.0) | Injects arrival-driven value-target variance at this budget; safety $\ne$ efficacy |
| Structural input features | Mechanically sound, tested correct | Neutral, $p=.58\to.69$ | Provably redundant: one-hot type ID already gives zero-hop discrimination |
| Actor global context | Proven to close a real observability gap | Neutral, $p=.91$ | Gap on s1 was already marginal (accidental 3-hop backdoor); fix is real but not decision-relevant here |

---

## Reproducibility pointers

- §6b.1: `gympn/potential.py`, `_test_phi_shaping.py`,
  `run_phi_shaping_s1_full.py`, results under `suite_results_*phi*`.
- §6b.2: `gympn/simulator.py::GymProblem._get_action_conflict_features`,
  `_test_structural_features.py`, `run_structural_features_s1_full.py`,
  results under `suite_results_struct_s1`.
- §6b.3: `gympn/networks.py` (`HeteroActor`, `_pool_action_postpone`),
  `_test_actor_global_context.py`, `run_actor_global_context_s1_full.py`,
  results under `suite_results_gctx_s1`.
- Full dated experimental log: `causal-stability-suite` memory file
  (sessions 2026-07-31 through 2026-08-03).
