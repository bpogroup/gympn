# BPM paper — outline (2026-10-07)

Target: BPM main track, LNCS, about 16 pages (check the CfP for the exact limit and
whether references count). Deadline more than six months away. Draft: `paper/bpm/`.

## Working titles
- *Structure-aware reinforcement learning for organisation-level process control*
- *What the process model knows: independence-aware reinforcement learning for
  concurrent business processes*
- *Learning when to wait: postponement and credit assignment for concurrent processes*

## Thesis
An organisation runs several processes side by side, with dedicated teams, under one
KPI. Standard RL treats the organisation as one monolithic decision problem: every
decision is credited with every process's reward noise, "waiting" can only freeze the
whole organisation, and every decision encodes the whole organisation.

The A-E PN already tells us which parts of the organisation are independent: the
connected components of its place-transition graph, after pruning dead transitions.
We use that **independence certificate** consistently at three points of the RL
pipeline:

1. **Credit (NF-GAE):** each decision is credited only with its own component's
   rewards, on its own component's decision clock.
2. **Decision timing (component-scoped postponement):** a component can wait for a
   better resource without freezing the others.
3. **Observation (component turns):** each decision observes and encodes only its own
   component.

Each use is exact where the model certifies independence and reduces to standard PPO
when the net has one component. The three are complementary: postponement only pays
off together with factored credit.

## Contributions
1. **The independence certificate for A-E PNs and its three uses**, with soundness
   conditions (place-disjoint components, local actor, work conservation or
   component-scoped postponement), unbiasedness, variance dominance over PPO, and exact
   equivalence to PPO at K=1. Global postponement violates the conditions.
2. **Component-scoped postponement**, new: it lets the agent learn non-work-conserving
   policies (the slow-server effect) per process. Experiments show it is learnable only
   with factored credit.
3. **An organisation-level benchmark**: an insurer with claims (next-activity
   selection), underwriting (resource assignment) and complaints (rework / quality
   gate), with regions, a shared-team coupling variant and a slow-mismatch variant in
   which waiting pays. Code in GymPN.
4. **Empirical evidence**: as concurrency grows, standard PPO degrades and the
   structure-aware pipeline does not; it learns to wait where waiting pays and
   nowhere else; its compute grows linearly with the number of components.

Not claimed: novelty of the policy network (relational attention with basis
decomposition, R-GCN/RGAT/HGT-style; reported as an enabling setup choice), a large
gain over separate per-process learners, or any benefit for a single process with a
shared resource pool (K=1).

## Structure and page budget

### 1. Introduction (1.5 p)
- Prescriptive process management at organisation level: several processes, one KPI.
- Three symptoms of treating the organisation as one problem: credit diluted by other
  processes' noise; postponement can only freeze everything; cost grows with size.
- Insight: the process model certifies independence; use it everywhere.
- Contributions; headline results (two sentences).

### 2. Background (2 p)
- A-E PNs and GymPN: actions vs events, bindings, the observation graph (Fig. 1: the
  insurer, one region).
- RL on A-E PNs: SMDP decision epochs, PPO with GAE in the SMDP form, the postpone
  pseudo-action as defined in earlier GymPN work.

### 3. Method (3.5 p)
- 3.1 The independence certificate: the component partition, dead-transition pruning,
  why shared places (queues, resource pools) merge components.
- 3.2 Factored credit (NF-GAE): per-component SMDP-GAE on each component's own decision
  clock; component critic pooling only the deciding component's action nodes.
- 3.3 Component-scoped postponement: one postpone pseudo-binding per component with
  enabled work; choosing it blocks only that component until one of its own events
  fires; time advances once every component has acted, has nothing enabled or waits.
  Contrast with global postponement.
- 3.4 Component turns: simultaneous decisions offered one component at a time, each
  observing only its own component.
- 3.5 Guarantees (statements, proof sketches; full proofs in a supplement):
  - Theorem 1 (soundness): under the conditions, components are conditionally
    independent given the state; component postponement preserves this, global
    postponement does not (counterexample).
  - Theorem 2: NF-GAE is unbiased; identical to PPO at K=1.
  - Theorem 3: variance dominance; the removed term is the other components' return
    variance, which no state-dependent baseline removes.
  - Proposition: component turns leave each component's dynamics and the active
    component's logits unchanged.
- 3.6 Policy network (one paragraph, cited): relational attention with basis-decomposed
  relation weights; flat graph observations.

### 4. Experimental setup (1.5 p)
- Benchmarks: the insurer (regions; shared clerks; slow mismatch) and multi-site
  routing (symmetric sites).
- Arms: joint PPO, separate per-process learners, NF-GAE; no / global / component
  postponement.
- Protocol: 40 epochs, greedy evaluation on 20 episodes with common random numbers,
  normalised to random and to the best heuristic (the waiting heuristic where waiting
  pays), 20 seeds for the headline cells (10 for shared clerks and per-process, 5 for budget/tuning), paired tests, Holm correction.

### 5. Results (5 p), by research question
- **RQ1 — Does concurrency hurt standard RL, and does factored credit fix it?**
  Insurer K=2/3/6, processes alone, multi-site. Fig. 2 (learning curves), Fig. 3 (gain
  vs K).
- **RQ2 — When does waiting pay, and what does it take to learn it?** Slow-mismatch
  insurer, five arms, 1 and 2 regions; behaviour (where the policies wait). Fig. 4,
  Table 2.
- **RQ3 — Is it budget, tuning or simply per-process learning?** PPO with 2× data and
  tuned learning rates; NF-GAE at PPO's best learning rate; separate per-process
  learners. Table 3.
- **RQ4 — Cost and scaling.** Full-graph vs component turns vs per-process learners.
- **Supporting — the policy network.** HGT vs relational network for PPO (bit-identical
  reproduction of the old failure).

### 6. Related work (1 p)
- RL for BPM: resource allocation, prescriptive process monitoring, next-best-action;
  A-E PN / GymPN line of work.
- Credit assignment with structure: factored policy gradients (Spooner et al. 2021),
  action-dependent factorised baselines (Wu et al. 2018), value decomposition (VDN),
  COMA / difference rewards, networked MARL.
- Non-work-conserving control: the slow-server problem (Lin & Kumar 1984), idling in
  queueing control.
- Relational GNNs: R-GCN, RGAT, HGT.

### 7. Discussion and limitations (1 p)
- The K=1 boundary: one process whose cases share a resource pool gains nothing.
  Evidence: BPI2012 is one well-mixed resource pool (median 4 resources per case, one
  connected resource graph), so the target is multi-process / multi-team control.
- Separate per-process learners capture most of the credit gain; what the certificate
  adds is postponement without freezing, one shared policy, automatic and sound
  decomposition, and merging of coupled processes.
- Postponement granularity: a component waits as a whole, not per resource.
- Simulated benchmark; real-data multi-organisation validation (BPIC 2015) as future
  work or extension.

### 8. Conclusion (0.5 p)

## Figures and tables
- **Fig. 1** The insurer net (one region), components coloured; inset: shared clerks
  merge claims and complaints.
- **Fig. 2** Learning curves, insurer 1 and 2 regions: PPO vs NF-GAE vs processes alone.
- **Fig. 3** Gain vs K (final and whole-run), K = 1, 2, 3, 6 (12 if run); multi-site as
  a symmetric point.
- **Fig. 4** Postponement: final return of the five arms (1 and 2 regions) and where
  the policies wait (waits per process, mismatches).
- **Table 1** Main results, all insurer conditions.
- **Table 2** Postponement arms and behaviour.
- **Table 3** Is it budget, tuning or per-process learning? (PPO variants, per-process
  learners, NF-GAE).
- **Table 4** Cost: full graph, component turns, per-process learners.
- **Table 5** (or text) Network: PPO-HGT vs PPO-relational.

## Evidence map
Protocol unless stated: relational network (8 bases), flat observations, 40 epochs,
greedy evaluation on 20 episodes, CRN (eval_seed 555000), normalised, paired seeds.

| Paper item | Experiment | Status | Key numbers |
|---|---|---|---|
| RQ1 | insurer r=1 (K=3), r=2 (K=6), 20 seeds | DONE (20 seeds 2026-10-09) | final PPO 0.965 → NF-GAE 1.101 (17/20, p=.001); 0.748 → 1.070 (20/20, p=2e-9); whole-run +0.218 / +0.406; PPO never reaches 0.9 at K=6 on 16/20 seeds |
| RQ1 | shared clerks (K=2) | DONE | +0.034 final (n.s.), +0.069 whole-run (p=.04) |
| RQ1 | processes alone (K=1) | DONE | learned in 3–9 epochs (median seed first ≥0.9); PPO in the insurer 18 (3/20 never), NF-GAE 6 |
| RQ1 | multi-site (K=4, symmetric), 20 seeds | DONE | finals tie (saturated); epochs to 0.9: 4 vs 8 (20/20) |
| RQ2 | slow insurer r=1, five arms, 20 seeds | DONE (20 seeds 2026-10-09) | NF-GAE+component 1.054 (18/20 ≥ waiting heuristic); PPO+component 0.910 (7/20); PPO+global 0.935 (7/20); NF-GAE none 0.928; PPO none 0.847 |
| RQ2 | behaviour (postpone_behaviour.py) | DONE (20 seeds) | NF-GAE waits only in underwriting (5.8/ep r=1, 10.7/ep r=2), never wasted; mismatches 1.23 → 0.36 (r=1), 2.71 → 1.17 (r=2); PPO arms waste 1.2–1.9 waits/ep in claims+complaints at r=2 |
| RQ2 | slow insurer r=2, five arms, 20 seeds | DONE (20 seeds 2026-10-09) | NF-GAE+component 1.005 (12/20 ≥ waiting heuristic); every PPO arm ≤0.707, 0/20; NF-GAE none 0.896; gap to best PPO arm +0.30 (20/20, p=7e-9) |
| RQ3 | PPO 2× data, lr ×0.5 / ×2; NF-GAE at lr 6e-4 (r=2) | DONE | best PPO 0.856 (2× data) vs NF-GAE 1.087; matched lr 0.782 vs 1.093 |
| RQ3 | per-process learners (from processes-alone runs) | DONE | r=1 final 1.058 vs 1.137; whole-run 0.960 vs 1.029; r=2 0.950 vs 1.020 whole-run |
| RQ4 | compute: full graph vs per-process vs component turns | DONE | full NF-GAE 1.7–2.2× per-process; component turns 1.67× / 2.39× faster, learning unchanged |
| Network | PPO-HGT vs relational on multi-site; bit-identical rerun | DONE | 0.403 vs 1.039 |
| Theory | Theorems 1–3, K=1 identity, locality | WRITTEN (NFGAE_THEORY.md) | component-postponement proof and the global-postponement counterexample to write up |

## Still to do (in rough order)
1. Finish the 2-region postponement run; behaviour check at 2 regions.
2. Component turns together with component postponement (currently mutually
   exclusive), then rerun NF-GAE+component with local encoding for the cost table.
3. ~~Top up headline cells to 20 seeds (insurer r=1/r=2, postponement r=1/r=2).~~ DONE 2026-10-09; paper updated.
4. Network check on the insurer (PPO-HGT at r=1) so the network result is not
   multi-site only.
5. Optional: insurer at 4 regions (K=12) for the gain-vs-K figure.
6. Optional: ablation separating factored advantage from the component critic.
7. Theory write-up: component-postponement soundness, global-postponement
   counterexample, component-turn proposition.
8. Optional extension: BPIC 2015 (five municipalities) as a real-data multi-unit case,
   if their staff turn out to be disjoint.
9. Figures and draft.

## Decision points and risks
- **2-region postponement shows no growth** → report the postponement synergy at one
  size; keep K-scaling for credit only.
- **Per-process learners**: say plainly that they capture most of the credit gain; the
  certificate's distinct value is postponement without freezing, one policy, and
  automatic sound decomposition.
- **Reviewer: "the benchmark is synthetic"** → the insurer reuses validated sub-nets,
  postponement is grounded in the slow-server literature, and the K=1 boundary is
  checked on BPI2012. BPIC 2015 would answer it directly.
- **Reviewer: "the network is the real effect"** → it is enabling and reported as
  such; every comparison of credit and postponement uses the same network.
