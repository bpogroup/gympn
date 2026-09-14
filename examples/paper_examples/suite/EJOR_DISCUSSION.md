# §7 — Discussion / managerial insights (draft)

*Draft for §7 of the EJOR paper (v2 outline, see `EJOR_PAPER_OUTLINE.md`).
Follows §5's theory (`EJOR_PROPOSITIONS.md`), §6a's positive results, and
§6b's boundary-condition investigation (`EJOR_BOUNDARY_CONDITIONS.md`).
Written for an OR-practitioner audience: the question this section answers
is not "does the method work" (§6 already showed that, precisely,
including where it doesn't) but "how would I know, before committing
engineering effort, whether it applies to my problem."*

---

## 7.1 A decision rule, not just a result

The paper's technical contributions (§4-§5) are two credit-assignment
mechanisms; its most actionable contribution for a practitioner is simpler:
**a test you can run on your own problem before choosing between them, or
before deciding neither is worth the engineering effort.**

`s-ccf`'s causal-component count $K$ (§5, Proposition 2) is not just a
theoretical device — it is computed automatically, as a byproduct of the
same union-find machinery the credit mechanism already runs, from nothing
more than a handful of simulated episodes under any reasonable policy
(random or a domain heuristic; $K$ is a property of the *net's realized
coupling structure*, not of policy quality). This gives a concrete,
pre-training diagnostic:

- **Run a small batch of simulated episodes and compute $K$ (or its
  distribution across episodes, since realized coupling can vary with the
  stochastic case mix).** If $K>1$ reliably — cases genuinely partition
  into causally independent or loosely-coupled sub-streams — the paper's
  mechanisms apply, and the choice between `s-ccf` and `cfpk` is a
  cost/granularity trade-off (§7.2). If $K\approx1$ essentially always —
  every case's fate is causally entangled with every other case's, most
  commonly because they all draw from one shared, fully-utilized resource
  pool — neither mechanism will out-perform a well-tuned PPO baseline, and
  §6b's investigation gives three independently-verified reasons not to
  expect a workaround from adjacent directions either (denser reward
  signal, richer input features, or a more observant network architecture
  all failed to move this case; see §7.3).

This is the practical form of Remark R1 (§5): $K=1\Rightarrow$ `s-ccf`
degenerates to PPO exactly, by construction, not by empirical accident — so
a practitioner who checks $K$ first never pays even the negligible cost of
`s-ccf`'s bookkeeping without reason to expect it to help.

**Concretely, in dynamic task assignment terms:** parallel production
lines, multiple service branches or sites, and resource pools that are
segmented (even loosely — e.g., each pool serves its own line except for
rare overflow routing) are exactly the structures that produce $K>1$. A
single, centrally-pooled team or machine bank serving one undifferentiated
queue is exactly the structure that collapses to $K=1$. Many real
operations sit somewhere in between (a firm with several largely-
independent branches that occasionally share a specialist resource); the
diagnostic handles this continuously, not as a binary classification — $K$
is exactly the number of components realized coupling actually produces on
a given problem instance, whatever that number turns out to be.

## 7.2 Choosing between `s-ccf` and `cfpk`

Both mechanisms are safe by construction (§5), so the choice is a
cost/granularity trade-off, not a correctness one:

- **`s-ccf`** is unbiased, requires no learned component beyond the
  standard critic, and costs a negligible amount of extra computation
  (union-find over a modest number of decisions and rewards per episode) —
  a near-drop-in replacement for PPO's baseline whenever $K>1$. Its cost is
  *conservatism*: at an AND-join or shared-handoff structure, the static
  partition it uses can lump two components together that only rarely
  interact, discarding some of the variance reduction a sharper (but
  unsafe, see `ccf`'s bias mechanism, §5) partition would have captured.
- **`cfpk`** pays a real, bounded, and partially-reducible extra compute
  cost (simulator forking; coupling truncation removes one proven, free
  chunk of that cost, §5 Part II) in exchange for an *exact* per-decision
  counterfactual credit that never needs to guess a partition at all — the
  right tool when `s-ccf`'s conservatism is costing more than the extra
  compute would, in particular at genuine synchronization points (joins,
  shared-resource handoffs) where the static and realized partitions
  diverge.

These are not mutually exclusive in principle: `s-ccf` could serve as the
default low-cost baseline across an entire operation, with `cfpk`'s exact
replay invoked selectively at decisions flagged as synchronization-prone
(e.g., via the same conflict-graph structure §6b.2 used, repurposed as a
targeting heuristic rather than a credit signal) — a combination this paper
does not evaluate, flagged as a natural extension in §7.4.

## 7.3 What this paper does not solve, stated plainly

A genuinely single-bottleneck, fully-coupled resource-contention problem —
the case §7.1's diagnostic flags as $K=1$ — gets no benefit from either
mechanism, by the theory's own honest prediction, not by omission. §6b went
further and closed off three plausible adjacent workarounds for that exact
case, each with a proof rather than a single discouraging run: reward
shaping is theorem-safe but empirically counter-productive at practical
budgets on this problem; richer static input features are provably
redundant once the network already encodes action-type identity; and a
genuine, previously-undocumented actor-architecture blind spot is real,
fixable, and *still* does not move this problem's outcome. A practitioner
facing this shape of problem should not expect the family of methods this
paper investigates — provenance-exploiting or observation-enriching
mechanisms of any of the kinds tested here — to help, and should not spend
engineering effort chasing them; plain, well-tuned PPO is already the right
tool, and any further gains likely require attacking exploration or value
estimation under contention directly (§7.4), a different problem than
credit assignment.

## 7.4 Limitations and future work

- **Closing `cfpk`'s remaining cost gap.** Coupling truncation is one
  proven, free reduction in `cfpk`'s fork-driven overhead (§5 Part II); it
  was not stacked in this paper with the mechanism's other cost knobs (fork
  probability, lookahead depth, replication count), each of which trades
  cost against the estimator's variance rather than its bias. A combined
  ablation across all cost knobs together, rather than coupling truncation
  in isolation, is the natural next step toward closing the remaining gap
  to plain-PPO cost.
- **What would actually move the single-component bottleneck.** This paper
  rules out several plausible mechanisms rather than finding the one that
  works; it does not claim the problem is unsolvable, only that it is not
  an observability or credit-attribution problem as this investigation
  tested those notions. Directions we did not test but consider
  well-motivated by that elimination: exploration strategies tailored to
  contention (e.g., intrinsic bonuses for under-visited resource-allocation
  patterns rather than states); value-function architectures with richer
  cross-decision reasoning than pooling (e.g., attention over the live set
  of pending decisions, rather than a single pooled summary); and
  temporally-abstracted or hierarchical policies that decide *how to
  allocate the pool* at a coarser grain than individual case-by-case
  matching. Search-based planning approaches (MCTS-style lookahead over the
  A-E PN) showed early, narrower promise on related foreclosure-style
  bottlenecks in preliminary work but are out of this paper's scope and
  not re-validated here; we mention them only as a pointer for future work,
  not a result.
- **Beyond A-E PN.** Neither `s-ccf` nor `cfpk` depends on anything
  specific to Petri nets beyond the availability of an exact, per-token
  provenance record and (for `cfpk`) a forkable simulator. Any executable
  simulation model that records which past decisions causally produced
  which future outcomes — not only Petri-net-based ones — could apply the
  same mechanisms; A-E PN is this paper's testbed because it already
  records provenance for free, not a requirement of the method.

## 7.5 Managerial takeaway

For an operations manager weighing whether to invest in structure-aware RL
over an off-the-shelf policy-gradient solver for a task-assignment problem:
the payoff scales with how independently the operation's sub-streams can
actually run, and this is checkable *before* committing to a training
run, not just observable after one. A multi-site or multi-line operation
where sites/lines are only loosely coupled is exactly where this
investment pays off, safely and at negligible extra cost (`s-ccf`) or at a
bounded, partially-reducible extra cost for finer-grained exactness
(`cfpk`). A centralized operation built around one shared, fully-utilized
resource is not — and no amount of giving the learner more information
about that shared resource changed that conclusion in this investigation,
which is itself useful information: it means the right response to a
disappointing result on such a problem is not to keep feeding the model
more structure, but to look elsewhere (exploration, problem
decomposition at the operational level itself, or a different training
paradigm entirely).
