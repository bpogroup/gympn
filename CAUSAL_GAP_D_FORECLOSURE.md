# Gap D — foreclosure & counterfactual responsibility (general CO perspective)

Companion to `CAUSAL_REDISTRIBUTION_SMDP_THEORY.md` (which named Gap D) and
`CAUSAL_REDISTRIBUTION_MATH_GAP.md` (which diagnosed it). Those framed Gap D
through the task-assignment lens ("assigning employee R to A makes R unavailable
for B"). This document takes the **general** view: `gympn` is a solver for *any*
combinatorial optimization (CO) problem expressed as an Action-Evolution Petri
net, and Gap D is not a domain quirk — **in CO, foreclosure is the combinatorial
structure itself.**

> **Thesis.** Forward token-lineage credit (`flow_dag`, `shapley_dag`) is a
> *positive but-for* signal: "this reward exists because these tokens flowed." The
> policy-gradient advantage is a *differential* object: "how would return change
> had I decided otherwise." The difference between them is exactly the value of the
> **road not taken** — the alternatives a decision *foreclosed*. For CO problems
> that road-not-taken is where all the difficulty lives, so any sound redistribution
> must account for it — through the value baseline, or through a conflict-aware
> coalition value, or by recording the choice set the trace currently discards.

---

## 1. The gap, stated generally

A decision (the firing of one **action binding** at a decision epoch) does two
things:

1. **Enables** downstream events by producing/transforming tokens — this is the
   forward lineage DAG. Captured by `flow_dag`/`shapley_dag`.
2. **Forecloses** the alternative bindings it competed with — by consuming tokens
   they needed, by filling a bounded place, by committing a scarce resource. This
   leaves **no forward token trail** (the foreclosed binding produced nothing), so
   it is invisible to lineage.

The PG advantage needs both:

```
A(s, b) = Q(s, b) − V(s),     V(s) = Σ_{b'∈ B(s)} π(b'|s) Q(s, b')
```

where `B(s)` is the **choice set** (the enabled bindings at the epoch). `V(s)`
averages over *all* competing bindings, so `A(s,b)` is intrinsically *relative to
the foreclosed alternatives*. A credit `c(b)` built only from `b`'s forward
lineage can equal `A` only when foreclosure is absent (or when a `V` baseline is
re-added to carry it). This is Gap D.

---

## 2. Foreclosure IS the combinatorial structure

The reframing that matters for a general library: a CO problem is "combinatorial"
precisely because its decisions are **coupled by shared, scarce resources** — and
a shared scarce resource is, in PN terms, a token several bindings compete to
consume. That coupling *is* foreclosure. Concretely, across problem classes the
same mechanism wears different names:

| CO problem | scarce token (place) | a decision consumes… | foreclosure = |
|---|---|---|---|
| Assignment / bipartite matching | one token per node | an edge using node *i*, *j* | every other edge on *i* or *j* |
| Knapsack | capacity budget (`W` tokens) | item weight | capacity for better items |
| TSP / VRP routing | "city unvisited" token | the next city/leg | every other tour completion |
| Job-shop scheduling | machine-timeslot token | a slot | that slot for other jobs (makespan) |
| Bin packing | bin-capacity tokens | space in a bin | space for other items |
| Set cover / facility loc. | selection budget | a set/facility | budget for other sets (overlap ⇒ diminishing returns) |
| Resource/task assignment | resource-pool token | an employee | that employee for other tasks |

In every row, the **objective is global** (tour length, makespan, coverage, total
weight-value) and the **difficulty is the coupling**. The "disjoint/parallel"
versions of these problems — no shared scarce token — are the *easy* ones, exactly
where forward lineage already suffices (cf. the empirical `b_sequence_disjoint`
result where `shapley_dag` ≥ `flow_dag`). So:

> **Gap D is not an edge case for `gympn`; it is the generic case.** The envs where
> a lineage-only credit looks fine are the embarrassingly-decoupled ones. The
> moment a problem is *interestingly* combinatorial, decisions foreclose each
> other, and lineage-only credit is provably incomplete.

A useful corollary: many classic CO objectives are **submodular or supermodular**
over the chosen set (coverage, influence, some scheduling costs). Submodularity *is*
foreclosure expressed as diminishing returns. A credit method that treats rewards
as independent (modular `v(S)`) cannot represent it.

---

## 3. Foreclosure in Petri-net terms (the general mechanism)

The PN gives a *white-box* account of foreclosure — these are structural and
domain-independent:

- **Direct conflict.** Two enabled bindings share an input token (compete for the
  same token in some place). Firing one removes it from the other's preset. This is
  the classic PN notion of **conflict** (non-persistence). The conflict relation is
  readable from the net: bindings whose presets intersect on a place whose token
  count is below their combined demand.
- **Capacity / blocking.** A bounded place (capacity constraint) that becomes full
  disables the transitions feeding it — foreclosure *at a distance*, not through a
  shared input but through a shared sink.
- **Enabling at a distance.** Symmetric to blocking: a decision that produces a
  token can *enable* a competitor; the counterfactual "had I not, the competitor
  could not have fired" is also part of responsibility.
- **Temporal foreclosure.** Committing a scarce token *now* vs. *later*. Postpone is
  the special case already handled (token-flow + SMDP discount); general temporal
  foreclosure is "use resource on the available job vs. hold it for a better one."

The decisive asset: at every decision epoch the simulator *knows the full choice
set* `B(s)` (it enumerates enabled bindings) and the marking (the constraints). A
black-box MDP sees only the chosen action; **`gympn` can see the alternatives that
were foreclosed and what they required.** That is precisely the information Gap D
needs — and it is currently **thrown away** by the trace (which records only fired
transitions and their lineage, not the choice set).

---

## 4. Why forward lineage cannot recover it

Lineage is built from tokens that were *produced and consumed*. A foreclosed
binding produces nothing and consumes nothing — it is a **counterfactual**, a
branch of the reachability graph that was never realized. No amount of post-hoc
DAG analysis recovers it, because the data isn't in the DAG. Recovering it requires
one of:

1. a **value function `V`** that, by being a function of the whole marking, already
   prices in what is and isn't still available (carries foreclosure *implicitly*);
2. the **choice set + conflict structure** recorded at decision time (lets us
   construct a counterfactual baseline *without* full re-simulation in the local
   cases); or
3. **counterfactual rollouts** (re-simulate under the foreclosed decision) — exact
   but expensive (the "heavy regime").

---

## 5. Routes to address Gap D

### Route A — let the value baseline carry it (implicit, general, cheap)
`A(s,b) = c(b) + e^{−βτ}V(s′) − V(s)`. Because `V(s)` is a function of the full
marking, it reflects the opportunity cost of the resources `b` is about to commit;
subtracting it makes the advantage differential **in expectation**. This is why
Route A is robust and why, empirically, Gap D did *not* destabilize the tested envs
once `causal_beta>0` (see `SHAPLEY_VS_FLOW_DAG_RESULTS.md`: the env-`a` collapse was
the missing SMDP discount, not foreclosure).

- ✅ General, sound-in-expectation, no extra modeling, no trace changes.
- ⚠️ Only as good as `V`. For *hard* CO problems the value of a marking is itself a
  hard combinatorial quantity; a weak `V` prices foreclosure poorly, and the credit
  (lineage-only) gives no help. This is where explicit treatment earns its keep —
  and it is exactly the regime `gympn` ultimately targets.

### Route B — conflict-aware coalition value `v(S)` (explicit, Shapley)
The `_shapley_values` engine already takes a pluggable `v_func`. The cheap
realized-trace `v(S)` is **modular** (rewards independent ⇒ no foreclosure). Make it
**submodular under token budgets**:

```
v(S) = expected discounted return achievable using decisions in S
       SUBJECT TO the token/capacity constraints of the net
```

i.e. when two decisions in `S` competed for the same scarce token, `v(S)` counts
only the rewards a *feasible* execution could realize, not the sum. Then Shapley
over this `v` automatically (i) splits a contested reward among the contenders and
(ii) charges a decision that consumed a token a *better* decision needed — Gap D
satisfied by construction (null-player + the submodular interaction terms).

- ✅ Principled; turns the PN's constraint structure into the credit.
- ⚠️ Needs the conflict structure (which bindings competed for which tokens) — i.e.
  the **choice set must be recorded** (see §6) — and, for non-local contention, may
  need a model or rollouts to evaluate `v(S)` (heavy regime). Cost scales with how
  far foreclosure propagates.

### Route C — counterfactual baseline over the choice set (targeted middle path)
Don't decompose the whole return; just fix the *local* foreclosure. At each epoch,
credit `b` **relative to its best foreclosed sibling**:

```
A_foreclosure(s, b) ≈ c(b) − max_{b' ∈ foreclosed(b)} V̂(b')
```

where `V̂(b')` is a cheap estimate (learned `V` of the state `b'` would lead to, or
a one-step lineage-value proxy). This is a structured **control variate**: it uses
the recorded choice set to subtract the opportunity cost explicitly, without a full
coalition game. Cheaper than Route B, more explicit than Route A.

- ✅ Uses exactly the white-box asset (the alternatives), local, cheap.
- ⚠️ Captures *immediate* foreclosure well, long-range foreclosure only via `V̂`.

---

## 6. The missing data: record the choice set (prerequisite for B and C)

All explicit routes need something the trace does **not** currently store: at each
decision epoch, the **set of enabled action bindings** (the alternatives), with
their input-token requirements and (optionally) the conflict relation among them.
The simulator already computes this (`bindings()` / the action phase); Gap-D work
should first extend `CausalTraces` to record, per decision:

- the chosen binding (already implicit via the fired transition),
- the foreclosed bindings and their presets (which tokens they needed),
- enough marking context to know *why* they became disabled (consumed token vs.
  filled capacity vs. lost enabling token).

This is a **domain-independent** addition (it is just PN conflict data), so it
serves every CO problem the library targets, not assignment alone. It is the
single highest-leverage step: without it, Gap D can only be handled implicitly
(Route A); with it, both the conflict-aware `v(S)` (B) and the counterfactual
baseline (C) become constructible.

---

## 7. Soundness conditions & honest limits

For a redistribution to close Gap D (not just A–C of the SMDP checklist):

1. **Conflict-completeness.** The recorded structure must capture *all* controllable
   foreclosure channels — direct conflict, capacity/blocking, enabling-at-a-distance.
   Miss a channel and that opportunity cost silently falls back on `V`.
2. **No double counting.** If `V` already prices foreclosure (Route A) and `v(S)`
   also models it (Route B), the advantage can double-charge. Keep one ledger: either
   foreclosure lives in the *credit* (then the baseline must be a plain return
   baseline) or in the *baseline* (then the credit stays positive-lineage). Mixing
   needs care.
3. **Feasibility-respecting `v(S)`.** A coalition's value must be what a *feasible*
   net execution can realize under the constraints — otherwise efficiency
   (`Σ c = return`) is restored at the cost of crediting infeasible counterfactuals.
4. **Cost honesty.** Local/series-parallel conflict ⇒ closed-form or small
   enumeration. Global contention (a single scarce token coupling many decisions, as
   in TSP/knapsack) ⇒ the coalition game is genuinely hard; Monte-Carlo Shapley +
   learned `V̂` for `v(S)` is the realistic tool, and at some point Route A (lean on
   `V`) is simply the better trade.

**The irreducible point.** Forward causality (lineage) and differential causality
(advantage) coincide only without foreclosure. `gympn` can make foreclosure
*observable and constructible* — the white-box net hands us the alternatives and the
constraints — but it cannot make it *free*: pricing the road not taken is, for hard
CO, as hard as the problem. The design space is therefore a deliberate trade between
**how much foreclosure to model explicitly** (B/C, accurate, costly) and **how much
to delegate to `V`** (A, cheap, only as good as the critic).

---

## 8. Recommendation / staged plan

1. **Keep Route A as the default** — it is general and was empirically sufficient on
   the current suite. Gap D is a *completeness* concern, made urgent by weak `V` on
   hard CO, not a bug on the easy envs.
2. **Record the choice set** in `CausalTraces` (§6) — domain-independent, the
   enabler for everything else, useful even just as diagnostics.
3. **Prototype Route C** (counterfactual baseline over the foreclosed siblings) —
   cheapest explicit win, plugs in as an advantage adjustment.
4. **Extend `v_func` to a feasibility-respecting (submodular) `v(S)`** for
   `shapley_dag` (Route B) — the principled version; validate on a problem where
   foreclosure is *designed in* and `V` is known to struggle (e.g. a knapsack/
   capacity or matching env), since the current sequence/parallel suite under-exercises
   Gap D.
5. **Add a foreclosure-stressing benchmark.** The existing a–h envs are mostly
   decoupled or mild; to study Gap D we need an env whose *optimum requires*
   declining a locally-good binding to avoid foreclosing a globally-better one
   (knapsack-like, or one-to-one matching with conflicting preferences).

   **DONE (2026-06-17):** `examples/paper_examples/suite/foreclosure_env.py` — a
   "scarce one-shot specialist" env (a unit-weight knapsack / online-selection in
   disguise). One specialist resource is *consumed* on first use; `n_gen<n_low`
   generalists force lows to queue, so the agent is repeatedly tempted to spend its
   single specialist shot on a low (`spec_low=+3`, lineage-credited positively)
   instead of holding it (postpone, ~0 lineage credit) for a high task arriving
   later (`spec_high=+10`). Self-test (deterministic): **Random 3.95, greedy/
   foreclosure-blind 6.00, foreclosure-aware optimum 13.00** (trap gap 7, headroom
   9). The trap is *locally rational* — a lineage-credited policy is pulled to 6,
   while the optimum 13 requires pricing the foreclosed +10. Ships with
   `perfect_heuristic` (optimum) and `greedy_heuristic` (the trap) for baselines.
   This is the env on which Route A vs B/C (and vs RUDDER/COMA) should be run.

See also: `CAUSAL_REDISTRIBUTION_SMDP_THEORY.md` (§6 foreclosure, Route B),
`CAUSAL_REDISTRIBUTION_MATH_GAP.md` (Gap D origin), `SHAPLEY_VS_FLOW_DAG_RESULTS.md`
(empirical: `V` absorbed foreclosure on the easy suite),
`causal_traces._redistribute_shapley_dag` / `_shapley_values` (pluggable `v_func`).