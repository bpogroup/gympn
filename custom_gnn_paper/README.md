# custom_gnn_paper — a PN-native architecture line

Research log for the architecture thread that split off from the credit-assignment
work on 2026-08-07. Kept separate from `examples/paper_examples/suite/` because it is
a different contribution with a different claim; the EJOR submission does not depend
on any of it.

**Status: the motivating gap is real and replicated; the operator fix is refuted (§3);
the observation fix that replaced it is also refuted (§4). Nothing in this line is
currently worth building except possibly the reach-without-depth idea in §4, and that
needs its own ceiling test first.** Read §3 and §4 before writing any code.

---

## 1. Where this came from

The credit-assignment line reached a clean result (`ccf` beats the no-lineage
ablation `mc_q` by +14.85 on ncopies N=8, p=0.0001, and +8.37 on multisite,
p=0.0005, with the win size governed by the causal-component count K). But every
*mechanism* tried on K=1 envs landed in a narrow +0.5–1.4 band regardless of what it
computed, which raised the question of whether the bottleneck was the credit signal
at all.

Two architectural measurements said it might not be.

### 1a. The receptive field is far too small

`lineage_decisions` gives the realized decision→reward **hop depth** directly.
Measured against `HeteroActor`'s default `num_layers=3`
(`suite/_diag_lineage_depth.py`):

| env | n | median depth | mean | p90 | max | frac > 3 |
|---|---|---|---|---|---|---|
| s1 | 430 | 5.0 | 5.80 | 11 | 17 | **62.8%** |
| s3 | 1636 | **9.0** | 9.79 | 19 | 31 | **80.1%** |
| multisite | 2347 | 7.0 | 8.08 | 17 | 25 | **69.7%** |
| f_loop | 185 | 6.0 | 5.46 | 8 | 12 | **83.8%** |

And the observation graph is **strictly bipartite** — 17 place→transition edge
types, 17 transition→place, **zero** place→place — so one *transition firing* costs
**two** GNN hops. `num_layers=3` therefore spans about **1.5 firings** against a
median causal link of 5–9.

**The naive fix fails.** s1 with `num_layers=8` (≈4 firings), plain `ppo_clip`,
seed 0: **12.60** vs **13.70** at the default depth, and 23.2 min vs ~15. Entropy
ended at 0.510 versus 0.135–0.162 for runs that converged. Consistent with
oversmoothing: uniformly deepening a sum-aggregating transformer buys reach and
loses local distinction. This is evidence *for* a better per-layer operator and
*against* more layers of the current one.

### 1b. Conjunction is absent from the representation

A transition fires only when **every** input place is marked; its firing capacity is
`min_p (tokens in p)`. `HGTConv` aggregates with attention-weighted **sum**, which
cannot separate "3 tokens in one place" (capacity 0) from "1 token in each of 3
places" (capacity 1).

Linear probes for firing capacity, held-out R², shuffled-feature placebo
(`suite/_diag_conjunction.py`):

| env | encoder | MIN (ceiling) | SUM (naive scalar) | **encoder** | placebo |
|---|---|---|---|---|---|
| s1 | untrained | 1.000 | 0.185 | **0.009** | −0.004 |
| s1 | **trained** | 1.000 | 0.192 | **0.011** | −0.004 |
| multisite | untrained | 1.000 | 0.726 | **0.065** | −0.003 |

The embeddings sit at placebo level while a *single scalar* reaches 0.19–0.73, and
**training does not move it** (0.0105 trained vs 0.0093 random). Replicated on two
envs.

---

## 2. What was built

`pn_conv.py` — `PNConv` / `PNEncoder`. A Petri-net-native operator splitting the two
arc directions by their semantics:

- **place → transition (consumption):** conjunctive. Sum *within* an edge type
  (a place's token count is the number of edges it contributes, since gympn expands
  places so each token is its own node), then **soft-min across** edge types (the
  conjunction over input places). Soft-min is a segment softmax over negated scores
  with a **learnable temperature**, so τ→0 is a hard min and τ→∞ is the mean — the
  operator *contains* mean aggregation as a limiting case and can fall back to it.
- **transition → place (production):** additive sum. Tokens accumulate; sum is
  correct here.

Targets with no edge of a given type are excluded from that type's min rather than
contributing zero, so an absent input place cannot masquerade as the scarcest one.

Nothing reads token attribute values; the split is a property of arc direction and
is general across A-E PN shapes.

---

## 3. Why it does not work, and what the real gap is

`_probe_conjunctive.py` runs the §1b probe against both encoders. Two iterations:

| version | HGTConv | PNConv |
|---|---|---|
| v1 — soft-min over *edges*, sum across types | −26.8% of gap | −27.1% |
| v2 — sum within type, soft-min across types | −22.7% | −22.8% |

v1 had the arithmetic backwards and the probe caught it. v2 is semantically correct
and **still does not help**.

The reason, measured directly:

```
a_transition nodes: 6   true place token counts: waiting1=3, waiting2=3, employee=1

in-degree of each a_transition node, per incoming edge type:
  waiting1 -> a_transition : degrees=[1,1,1,0,0,0]   (place has 3 tokens)
  waiting2 -> a_transition : degrees=[0,0,0,1,1,1]   (place has 3 tokens)
  employee -> a_transition : degrees=[1,1,1,1,1,1]   (place has 1 token)
```

**An `a_transition` node is a BINDING, and it is wired only to its own tokens** —
never to the place's full marking. So `min over input places of token count` is not
computable from this graph by *any* aggregation operator. The conjunction gap in §1b
is real, but it is not an operator problem: **the information is not in the graph.**

A direct consequence worth stating on its own: **the actor cannot see queue
lengths.** A binding's neighbourhood contains one waiting task, not "three tasks are
waiting." For a scheduling policy that is a significant blind spot.

---

## 4. The observation fix was designed, then killed by its own oracle test

§3 said the fix was to expose place occupancy rather than change the aggregation. The
design got as far as being specific:

- a bare **token count** per node was rejected — it destroys the queue's *composition*,
  and composition is what a scheduling decision needs ("3 waiting" is useless without
  "2 of type 0, 1 of type 1");
- **context edges** (every token in a place → every binding consuming from it) would
  have let the operator learn the aggregation, but as first stated they **collapse the
  policy**: every binding from the same place would then have an identical incoming
  set, hence an identical embedding, and the measured degrees show the own-token edge
  is currently the *only* thing distinguishing bindings. Fixable with an own/context
  **edge-type split**, since heterogeneous GNNs parameterise per relation;
- a **place-summary node** carrying a multiset aggregate `[count, Σv, max v, min v]`
  was the cheaper alternative (Σ over one-hots *is* the type histogram).

None of it should be built, because the information does not change the optimal action.

### The oracle test (`suite/_diag_composition_oracle.py`, 40 episodes/arm)

A heuristic solver sees raw bindings, so it can compute composition even though a
policy cannot. If a composition-aware oracle cannot beat the composition-blind anchor,
no encoding of composition will help.

| env | anchor | scarce-1st | abundant-1st | opp-cost | best-delay |
|---|---|---|---|---|---|
| s1 | **14.675** | 14.375 | 14.750 | 14.750 | 10.225 |
| s2 | **90.425** | 89.675 | 89.700 | 89.675 | 89.475 |
| s3 | **112.250** | 112.250 | 112.250 | 112.250 | 84.350 |

No arm beats the anchor anywhere. `scarce-first` and `abundant-first` are *opposite*
orderings scoring within 0.03 on s2, so the ordering carries nothing — the control
worked. `opp-cost`, the strongest composition rule, is +0.075 on s1 against sd 0.73.
`best-delay` collapses (−4.45, −27.90) because it drops the anchor's **stage
priority**: what matters on these envs is *finish work-in-progress first*, which the
anchor already does, not who is queued.

### The pattern this belongs to

Six input-side interventions have now measured neutral-or-worse on this suite:
phi-shaping, structural features, actor global-context, token age
(`suite/_diag_age_oracle.py`), and queue count/composition. These environments appear
to be **greedy-matchable** — stage-priority + type-match is close to optimal and
richer state does not change which action is best. A representational gap being
*real* has repeatedly failed to imply that closing it *helps control*; §1b's
conjunction gap is genuine, large and replicated, and still not worth acting on.

**Standing rule for this line: run the oracle before building the mechanism.** It cost
~4 minutes here versus a graph-construction change plus a training arc.

### What remains live

Only **reach without depth**: the §1a receptive-field gap stands, deepening backfired,
and lineage-derived **shortcut edges** (a long causal link becomes one hop) or a
**firing-step layer** (place→transition→place per layer) would attack it without the
oversmoothing tax. Note this is *not* covered by the oracle results above — those test
whether extra *state* helps, whereas this is about whether the network can connect a
decision to the reward it causes at all. It needs its own ceiling test before anyone
builds it.

---

## 5. Files

| file | what it is |
|---|---|
| `pn_conv.py` | `PNConv` / `PNEncoder` — the conjunctive/additive operator |
| `_probe_conjunctive.py` | side-by-side probe of PNConv vs HGTConv on firing capacity |

Supporting diagnostics live in `../examples/paper_examples/suite/`:

| file | what it measured |
|---|---|
| `_diag_lineage_depth.py` | §1a — realized decision→reward hop depth vs `num_layers` |
| `run_s1_deep.py` | §1a — the `num_layers=8` run that came back worse |
| `_diag_conjunction.py` | §1b — the conjunction gap, untrained and trained |
| `_diag_composition_oracle.py` | §4 — the oracle test that killed the observation fix |
| `_diag_age_oracle.py` | the same test for token age (also neutral) |
| `_diag_actor_cross_component.py` | cross-component leakage through shared places |
