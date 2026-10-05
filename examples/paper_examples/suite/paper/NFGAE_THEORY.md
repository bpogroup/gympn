# Net-factored GAE (NF-GAE): theory note (2026-10-05)

Status: working note for the BPM paper. It replaces the provenance-based credit family
(lrq / ccf / cgae*) as the method the paper claims anything *provable* about. The old
family becomes an ablation, and Section 5 explains why it is biased.

## 1. Setting

An A-E PN with transitions T (actions T_A, events T_E) and places P, simulated
as an SMDP. Decision epochs k = 0, 1, … at clock times τ_k. At each epoch the agent
picks one enabled binding a_k (no postpone; see condition C3). r_k is the reward of
all events firing in [τ_k, τ_{k+1}), counted undiscounted at τ_k. The continuation
discount is Γ_k = e^{−β(τ_{k+1} − τ_k)}. This is exactly the convention of
`TrajectoryBuffer._smdp_discounts` + `smdp_gae` (gympn/data.py).

**Net partition.** Let G be the undirected bipartite graph on P ∪ (T minus postpone),
with an edge between a place and every transition that consumes from it or produces
into it. Its connected components C_1 … C_K partition places and transitions. Each
transition t has a component c(t), and so does each reward (the reward-bearing event
that fires it). The partition depends only on the net, not on the marking or the
trajectory. Any place shared by two transitions puts them in the same component,
whether it is a queue, a resource pool, or anything else.

**Component SMDP.** For component c, let k_1 < k_2 < … be the epochs at which an
action of c is chosen. Define

    R^c_i = Σ_{k_i ≤ k < k_{i+1}} r^c_k          (c's rewards in [τ_{k_i}, τ_{k_{i+1}}))
    Γ^c_i = e^{−β(τ_{k_{i+1}} − τ_{k_i})}

so R^c_i is counted undiscounted at c's own latest epoch. This is the code's
convention applied to c's own decision clock. The objective is J = Σ_c J_c with
J_c = E[Σ_i (Π_{j<i} Γ^c_j) R^c_i].
- At K=1 it is PPO's objective exactly.
- For β=0 it is the undiscounted throughput for every K.
- Otherwise it differs from the global-epoch convention only in which epoch a reward
  is counted at (bounded by the within-sojourn discount).

## 2. Structural soundness (Theorem 1)

Conditions:

- **C1 (place-disjointness).** Holds by construction of the partition.
- **C2 (local actor).** The logit of a binding a in component c is a function of
  component c's marking only: ℓ(a, s) = ℓ(a, s_c). The policy is the softmax over
  all enabled bindings. HeteroActor satisfies this when `global_context=False`: the
  observation graph's edges are directed token-flow edges inside the net, so there
  is no message-passing path between components. The postpone node only *receives*
  edges (`simulator.py:916`), and it is absent under C3.
- **C3 (work-conserving).** No postpone. Time advances to the next event only when
  no binding is enabled anywhere.
- **C3' (component postpone — replaces C3).** Implemented as
  `GymProblem.postpone_scope='component'`.
  - There is one postpone pseudo-binding per component with an enabled action.
  - Its graph node receives edges only from that component's nodes.
  - Choosing it blocks only that component's actions.
  - The component is woken only by one of its own events.
  - Time advances once every component has acted, has nothing enabled, or is
    waiting.
  - Step 4 of the proof then goes through unchanged: c's decision instants are still
    a function of ξ_c.
  - With K=1 it is identical to the global postpone. Verified bit-for-bit:
    trajectories and observation graphs (`tests/test_nfgae.py`), and end-to-end
    training finals.
  - One stated exception: if no event will ever fire again, all components are woken
    (deadlock guard).
- **C4 (local randomness).** An event's behaviour depends only on its input tokens
  and fresh noise independent of everything else.

**Theorem 1.** Under C1–C4, conditional on the state s_k at any epoch, the future
trajectories ξ_1 … ξ_K of the components are mutually independent. ξ_c is c's
markings, decisions, rewards and epoch times. Its law depends only on s_{k,c}. In
particular, for an action a_k in component c, the law of ξ_{c'} for every c' ≠ c is
the same under every a_k.

*Proof sketch.*
1. **One instant.** Look at the decisions taken at a single clock value. At every
   pick, the probability that the pick is binding a in component c is
   e^{ℓ(a,s_c)} / Σ_{c''} Z_{c''}(s_{c''}). Conditional on the pick falling in c, it
   is e^{ℓ(a,s_c)} / Z_c(s_c). This is Luce's choice axiom / IIA: the local softmax
   over c's own bindings.
2. **Each component's choices are local.** A pick in c changes only s_c (C1). So,
   conditional on the full sequence of component labels of the picks, the choices
   inside each component are independent draws from the local softmax. Their joint
   law does not depend on the label sequence.
3. **Each component runs out on its own.** By C3 the instant ends only when every
   component has no enabled binding. How many picks c makes is a function of c's own
   choices only.
4. **Across instants.** c acts exactly at the instants where its own events enable a
   binding. Under C3 those are determined by ξ_c alone (C4). Another component's
   events create epochs where c has nothing enabled, and they do not touch ξ_c.
5. **Conclusion.** ξ_c is therefore a measurable function of s_{k,c}, c's own
   event noise, and c's own local-softmax draws. These are independent across c. ∎

**Why each condition is needed.**
- *C2.* `global_context=True` pools the whole graph into every logit, so it breaks
  C2.
- *C3.* A global postpone breaks C3. Its logit reads the whole graph, and choosing it
  ends the instant for every component. Hence the paper runs postpone OFF, as
  multi-site already does. A per-component postpone would restore C3; that is future
  work.
- *C1.* A shared queue breaks C1 for any finer partition. That is pre-emption
  (Section 5).

## 3. The estimator

For decision i of component c:

    δ^c_i = R^c_i + Γ^c_i V_c(s_{k_{i+1}}) − V_c(s_{k_i})        (0 bootstrap after c's last epoch)
    A^c_i = δ^c_i + λ Γ^c_i A^c_{i+1}

The policy gradient is ĝ = Σ_c Σ_i ∇log π(a_{k_i} | s_{k_i}) A^c_i. The critic is
additive, V(s) = Σ_c V_c(s). Head V_c reads only component c's nodes and is
regressed on A^c_i + V_c(s_{k_i}) at c's own epochs.

**Theorem 2.**
- **(a) Exactness at K=1.** With K=1, every epoch is an epoch of the single
  component: R^1_i = r_i, Γ^1_i = Γ_i, V_1 = V. So A^1 is term-for-term PPO's
  SMDP-GAE.
- **(b) Unbiasedness.** Under Theorem 1, ∇J_c = E[Σ_k ∇log π(a_k|s_k) Q^c(s_k,a_k)].
  The terms with a_k outside c vanish, because their Q^c does not depend on a_k
  (Theorem 1), and E[∇log π | s] = 0. So
  ∇J = Σ_c E[Σ_i ∇log π(a_{k_i}|s_{k_i}) Q^c(s_{k_i}, a_{k_i})]. At λ=1, A^c_i is
  c's discounted return minus V_c(s_{k_i}), so ĝ is unbiased for any V_c.
- **(c) Bias at λ<1.** The bias is the ordinary GAE bias of each component's own
  sub-SMDP. It vanishes as V_c → V_c^π.

## 4. Variance dominance (Theorem 3)

Fix a decision k in component c. Let ψ = ∇log π(a_k|s_k), G^c the component-c
return from k, G^− = Σ_{c'≠c} G^{c'}, and m(s) = E[G^− | s]. Use λ = 1 and look at
the per-decision term, as in Wu et al. 2018 and Spooner et al. 2021.

- The full-return ("PPO") term with any state baseline b is
  g_b = ψ (G^c + G^− − b(s)).
- The NF-GAE term with baseline b_c is g^F = ψ (G^c − b_c(s)).

**Theorem 3.** Under Theorem 1, for every b, take b_c = b − m. Then

    E‖g_b‖² = E‖g^F_{b−m}‖² + E[ ‖ψ‖² Var(G^− | s) ],

and both estimators have the same mean. Hence

    inf_{b_c} Var(g^F) ≤ inf_b Var(g_b) − E[‖ψ‖² Var(G^− | s)].

*Proof.*
- Write G^c + G^− − b = (G^c − (b − m)) + (G^− − m). Call these X and Y.
- By Theorem 1, Y is independent of (a_k, X) given s, and E[Y | s] = 0. So
  E[‖ψ‖² X Y] = E[ E[‖ψ‖² X | s] · E[Y | s] ] = 0, and
  E[‖ψ‖² Y²] = E[‖ψ‖² Var(G^− | s)].
- The means are equal because E[ψ Y] = 0. ∎

**Remarks.**
- The gap is noise that **no state-dependent baseline can remove**. A baseline can
  only absorb E[G^−|s], never Var(G^−|s). This is the precise sense in which PPO,
  however well its critic is trained, is beaten.
- For K exchangeable components, Var(G^−|s) ≈ (K−1)σ², so the gap grows linearly in
  K−1. At K=1 it is 0, which is consistent with Theorem 2(a).
- At λ<1 the same decomposition applies to the TD-residual sums. Both estimators
  then use bootstrapped returns, and the dropped part is the other components'
  λ-returns. Its conditional mean is action-independent only once their critics are
  exact. So the λ<1 statement holds asymptotically in critic accuracy.

## 5. Why the provenance family is biased (Props 4–5)

**Proposition 4 (pre-emption; provenance is not influence).** Take one queue holding
cases x and y, and two resources R1 and R2.
- Decision d1: R1 picks x or y. R2 then takes the remaining case (decision d2).
- Rewards: R1 earns 1 on x and 1+ε on y. R2 earns 0 on x and 1 on y.
- True values: Q(d1 = x) = 1 + 1 = 2 and Q(d1 = y) = 1 + ε + 0 = 1 + ε, so the
  correct choice is x.
- R2's reward has lineage {d2, R2's own chain, the arrival of the case}. d1 is not in
  it: d1 never produced a token that R2's job consumed. It only *removed* one.
- So every lineage-restricted credit (lrq, ccf, cgae*) gives d1 the values 1 and
  1+ε. It prefers y. **The gradient has the wrong sign for every ε > 0.**

This is the BPM next-activity N=1 environment in miniature: two employees, one shared
queue. NF-GAE puts R1, R2 and the queue in one component, so it is exact here
(Theorem 2a). It is also why the contention-edge variant `cgae_cflow_ct` recovers
toward PPO in the 2×2 diagnostic.

A second defect is independent of pre-emption. The realized lineage mask is computed
after the action, so it is action-dependent. E[mask · G | s, a] ≠ mask-free
expectations in general. The `_diag_R_action_invariance.py` measurement found R(d)
identical across actions at only 18.8% / 6.2% of fork points. NF-GAE's mask is a
fixed function of the net.

**Proposition 5 (masked rewards with a global critic).** cfgae, and cgae* with
λ < 1, run a component-masked reward stream through a *single* critic V(s).

- **cfgae (global step index).** Its advantage for c's decision at epoch k is
  A = Σ_j (λ)^j (Π Γ) δ̃_{k+j}, where δ̃ = r^c + Γ V(s') − V(s).
  - This equals the NF-GAE-style advantage plus a residual
    (1−λ) Σ_j λ^{j−1}(Π Γ)(V − V^π_c)(s_{k+j}).
  - The residual does not vanish even when V is the best global fit E[target | s].
    That fit is a mixture over whichever component decides at s, not V^π_c.
  - V^π_c depends on c's marking, so the residual is action-dependent in general:
    a bias.
  - A fraction ≈ (K−1)/K of the epochs k+j belong to other components, so the
    residual grows with K. This matches cfgae ≈ PPO at N=1 and collapsing at N=8 /
    hard N=4.
- **λ = 1.** The residual telescopes to the baseline term, so the estimator is
  unbiased.
- **cgae*.** These bootstrap along the causal successor (a c-epoch). They avoid the
  wrong-component index, but still read V(s'), which cannot tell which component is
  asked for (Theorem 2's V_c is indexed).

Fixes in NF-GAE: a component-indexed critic, the component's own clock, and labels
taken from the net.

Separately, cfgae as implemented labels a step's whole reward by its largest
component (`data.py`, `lab[t] = max(...)`). That misattributes reward at multi-reward
steps, and those become more frequent as K grows.

## 6. Positioning

- **Factored Policy Gradients** (Spooner et al., NeurIPS 2021): same core idea, an
  influence network between action factors and reward factors. There the influence
  network is given.
- **Action-dependent factorized baselines** (Wu et al., ICLR 2018): the per-decision
  variance algebra.
- **VDN** (Sunehag et al. 2018): additive critics.
- **COMA, difference rewards**: counterfactual credit in multi-agent RL.

What is new here:
1. The influence structure is *derived and certified* from A-E PN semantics.
   Theorem 1 includes the scheduler conditions C2–C3 (IIA under a global softmax,
   work-conservation) that factored-MDP results take for granted.
2. The per-component SMDP clock, exact at K=1.
3. A proof that provenance-based credit, the natural "process mining" idea, is
   unsound under pre-emption. Pre-emption is the defining feature of resource
   allocation.

## 7. Verification checklist

- [ ] Partition: `make_next_activity(N)` → N components; multi-site n_flex=0 →
      n_sites components; n_flex>0 → 1.
- [ ] K=1 identity: NF-GAE advantages == `smdp_gae` advantages on an N=1 rollout.
- [ ] Actor locality: copy-1 logits bit-identical under a copy-2 marking change
      (postpone off).
- [ ] CRN forks: E[G^− | s, a] action-invariant under the NF-GAE partition (N=2).
      The provenance mask is not invariant on the shared queue (Prop 4, measured).
- [ ] Training: N=1 nfgae ≡ PPO. Multi-site K=4: does the provable method keep the
      gain? BPM N=4 pilot.
