# Related work: COMA, RUDDER, Shapley-Q — and how `gympn`'s credit assignment relates

Positioning document for the causal credit-assignment line in `gympn`
(`flow_dag`, `shapley_dag`/Route B, the Route-C counterfactual baseline, token-flow
postpone, SMDP discounting). These three methods are the prior art a reviewer will
reach for first; this note states each accurately and then the precise
relationship — where the idea overlaps, and where the A-E-PN setting changes it.

> ⚠️ Formulations below are recalled from the literature; verify exact equations and
> citations against the papers before using in a manuscript.

---

## TL;DR

| | players (who gets credit) | causal model | credit object | time | foreclosure |
|---|---|---|---|---|---|
| **COMA** | simultaneous **agents** | **learned** centralized critic `Q(s,u)` | counterfactual **baseline** (variance only) | none | implicit via critic |
| **RUDDER** | **time steps** of one agent | **learned** LSTM + contribution analysis | return **redistribution** | discrete steps | none |
| **Shapley-Q** | simultaneous **agents** | **learned** critic over agent coalitions | Shapley decomposition of global **Q** | none | via coalition `v(S)` |
| **`gympn`** | **decisions** (action firings), per-reward | **white-box** token-lineage DAG (+ conflict, exact) | return **redistribution** (flow / Shapley) | **SMDP** `e^{−βτ}` from trace timestamps | targeted (Route C / conflict-aware `v(S)`) |

The one-line differentiator: **COMA, RUDDER and Shapley-Q all either operate over
*agents* or *learn* the causal/temporal contribution, because their environment is
black-box. An A-E Petri net is white-box — lineage, timestamps and the conflict
(choice) set are observable exactly — so the same credit objects become
*structurally computed and exact*, scoped *per reward* by the DAG, and extend
naturally to continuous time (SMDP) and to foreclosure (conflict-aware `v(S)`).**

---

## 1. COMA — Counterfactual Multi-Agent Policy Gradients

*Foerster, Farquhar, Afouras, Nardelli, Whiteson — AAAI 2018.*

**Problem.** Cooperative multi-agent RL with a single shared team reward
(centralized training, decentralized execution). Which agent's action mattered?

**Mechanism.** A centralized critic estimates `Q(s, u)` for the *joint* action
`u = (u^1,…,u^n)`. For agent `a`, replace the value baseline with a **counterfactual
baseline** that marginalizes out only `a`'s action, holding the others fixed:

```
A^a(s,u) = Q(s,u) − Σ_{u'^a} π^a(u'^a | τ^a) · Q(s, (u^{−a}, u'^a))
```

The critic is built to output all of agent `a`'s counterfactual Q-values in one
pass, so the baseline is cheap. This isolates `a`'s contribution relative to "what
the team reward would have been had `a` acted differently, others unchanged."

**Nature.** It is an *action-dependent baseline*: expectation-preserving (variance
reduction), not a reward change. Descends from difference rewards / Wonderful Life
Utility (Wolpert & Tumer).

**Relation to `gympn`.** Route C (counterfactual baseline over **foreclosed sibling
bindings** at a decision epoch) is the **single-agent, white-box analog of COMA**:
- COMA marginalizes over *another axis* (other agents); `gympn` marginalizes over
  the *alternatives at the same decision* — the enabled bindings the chosen one
  foreclosed. In a single-agent sequential CO solver that competing-binding set is
  the right counterfactual, and the PN **exposes it exactly** (the simulator already
  enumerates enabled bindings), whereas COMA must *learn* `Q` over a combinatorial
  joint-action space to fake the counterfactual.
- Shared caveat: as a baseline it only reduces variance — see Tucker et al.,
  "The Mirage of Action-Dependent Baselines" (ICML 2018). So Route C inherits COMA's
  positioning *and* its risk; it is a component, not a headline.
- Difference: COMA's counterfactual `Q` is a learned estimate; `gympn` can ground the
  foreclosed alternative in its actual lineage/value, and the foreclosure it targets
  is the PN conflict structure (Gap D), not agent interaction.

---

## 2. RUDDER — Return Decomposition for Delayed Rewards

*Arjona-Medina, Gillhofer, Widrich, Unterthiner, Brandstetter, Hochreiter — NeurIPS 2019.*

**Problem.** Single agent, **delayed/episodic** reward. TD has to propagate credit
across a long horizon (high variance / slow); Monte-Carlo is high-variance too.

**Mechanism.** *Redistribute* the episodic return to the individual steps that
caused it, producing a **return-equivalent** MDP (same optimal policy) in which
rewards are immediate. An *optimal* redistribution makes the expected future reward
zero at every step — delayed reward eliminated. In practice: train an **LSTM** to
predict the episode return from the state-action sequence, then use **contribution
analysis** (differences of successive return predictions) to assign each step the
amount by which it *changed the predicted return*.

**Nature.** A *reward redistribution* (`Σ redistributed = return`), conserving total
return, aligning reward in time to the causal step. The causal/temporal model is
**learned** (the LSTM is a black-box surrogate for "which step changed the outcome").

**Relation to `gympn`.** This is the **closest relative**: `flow_dag`/`shapley_dag`
have *exactly RUDDER's goal* — decompose the (delayed, episodic) return onto the
causal decisions, conserving the total. The difference is the causal model:
- RUDDER **learns** the contribution because the environment is black-box; the A-E
  PN **exposes** it — the token lineage DAG says *exactly* which decision's output
  tokens flowed into which reward-bearing transition. No LSTM, no contribution
  analysis, no approximation: the redistribution is an exact `O(V+E)` graph
  computation (flow) or an exact/MC Shapley over a tiny per-reward lineage.
- So `gympn` is, in one sentence, **"RUDDER's return decomposition with an exact
  white-box causal graph instead of a learned predictor."**
- Two extensions RUDDER does not have: (i) **time** — `gympn` uses continuous event
  timestamps and SMDP discounting `e^{−βτ}` (RUDDER redistributes over discrete
  steps with no opportunity-cost-of-time notion; this is what made postpone
  tractable here); (ii) **foreclosure** — RUDDER is positive-contribution only,
  while the conflict-aware `v(S)` (Route B, Gap D) targets the road-not-taken.
- Shared theory: both rely on return-equivalence / optimal-policy preservation.

---

## 3. Shapley-Q / SQDDPG

*Wang et al., "Shapley Q-value: A Local Reward Approach to Solve Global Reward
Games," AAAI 2020.*

**Problem.** Cooperative multi-agent RL with a global reward; give each agent a
*fair* local reward.

**Mechanism.** Model the agents as players of a cooperative (Markov convex) game and
assign each agent the **Shapley value** of the global action value — the unique
attribution satisfying efficiency (Σ = global `Q`), null-player, symmetry, linearity.
Because exact Shapley is exponential in the number of agents, it is approximated by
**Monte-Carlo** sampling of agent orderings/coalitions; the coalition value comes
from a learned critic. Used as each agent's local reward for the policy gradient
(SQDDPG).

**Relation to `gympn`.** `shapley_dag` uses the **same axiomatic tool** (Shapley) but
the *game is different in three load-bearing ways*:
- **Players are decisions, not agents.** The coalition is over the action firings in
  one trajectory, **scoped per reward by the token lineage** — so `n` is a handful
  (the decisions a given reward actually depended on), not the whole agent set. This
  *per-reward locality* makes Shapley **exact** in the common case rather than forced
  to MC over all players.
- **`v(S)` is structural, not learned.** Shapley-Q's coalition value is a learned
  critic over agent subsets; `gympn`'s `v(S)` is read from the trace — realized
  reachability (cheap regime) or feasibility/token-budget-constrained
  (foreclosure-aware regime). No critic over coalitions of decisions is trained.
- **Time-aware.** Postpone enters as a player with a **negative** marginal via the
  `e^{−βτ}`/`γ^wait` discount; Shapley-Q has no temporal or foreclosure-timing
  dimension.

So `shapley_dag` is "Shapley credit, but over *per-reward lineage decisions* with a
*white-box structural* characteristic function and an *SMDP-time* axis," not Shapley
over agents with a learned `v(S)`.

---

## 4. The unifying view

Each prior method solves one face of credit assignment and, lacking a white-box
causal model, either changes the *players* to agents or *learns* the contribution:

```
            decomposes the RETURN     uses COUNTERFACTUALS     uses SHAPLEY axioms
RUDDER             ✔ (learned)               ✗                        ✗
COMA                  ✗                ✔ (learned, per-agent)         ✗
Shapley-Q          ✔ (of Q)                  ✗                 ✔ (over agents, MC)
gympn        ✔ (exact, white-box)   ✔ (Route C / v(S), structural)  ✔ (per-reward, exact)
```

`gympn` sits where all three meet, with the A-E PN supplying what each had to learn
or approximate:

- the **lineage DAG** → exact return decomposition (vs RUDDER's LSTM),
- the **choice/conflict set** → exact counterfactual alternatives (vs COMA's learned
  joint-`Q`),
- **per-reward locality** → tractable exact Shapley (vs Shapley-Q's global MC),
- **event timestamps** → SMDP time-discounting (none of the three has this).

This is also the honest *novelty boundary* (see `CAUSAL_GAP_D_FORECLOSURE.md` §
publishability discussion): **the credit rules themselves are not new** — return
decomposition (RUDDER), counterfactual baselines (COMA), Shapley credit (Shapley-Q)
are all published. The contribution, if any, is the **white-box structural setting**:
that an A-E Petri net makes these otherwise learned/approximate/agent-scoped objects
*exact, tractable, single-agent-sequential, time-aware, and foreclosure-aware*, for
*arbitrary* combinatorial optimization problems. A paper should therefore lead with
the framework and treat flow/Shapley/Route-C as instances within it — and must
benchmark against these three, not against vanilla PPO alone.

---

## 5. What to cite / engage in a write-up

- COMA — Foerster et al., AAAI 2018 (counterfactual MA policy gradients).
- Difference rewards / Wonderful Life Utility — Wolpert & Tumer (~2001–2002).
- "The Mirage of Action-Dependent Baselines" — Tucker et al., ICML 2018 (the
  cautionary result for Route C).
- RUDDER — Arjona-Medina et al., NeurIPS 2019; Align-RUDDER (follow-up).
- Hindsight Credit Assignment — Harutyunyan et al., NeurIPS 2019 (return-conditioned
  credit; another black-box temporal-credit relative).
- Shapley-Q / SQDDPG — Wang et al., AAAI 2020; related Shapley-MARL (e.g. SHAQ).
- (Adjacent) SHAP — Lundberg & Lee, NeurIPS 2017 (Shapley for attribution, different
  domain but the same axioms).

See also `CAUSAL_REDISTRIBUTION_SMDP_THEORY.md` (the white-box-SMDP thesis, Routes
A/B), `CAUSAL_GAP_D_FORECLOSURE.md` (foreclosure = combinatorial structure;
Route C), `CAUSAL_ADVANTAGE_TD0.md` (the TD(0)/credit advantage design).