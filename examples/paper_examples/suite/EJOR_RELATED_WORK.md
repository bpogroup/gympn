# §2 — Related work (draft)

*Draft for §2 of the EJOR paper (v2 outline, see `EJOR_PAPER_OUTLINE.md`).
Citation-marker style `[Author(s), Year]` throughout; full bibliographic
details for every citation, including the three prior-work papers, are
verified against publisher/proceedings/arXiv pages directly (not recalled
from memory alone) and listed in the References subsection at the end,
ready to transcribe into `refs.bib`.*

---

## 2.1 Dynamic task assignment and operational control with (D)RL

Dynamic task assignment — assigning stochastic streams of tasks to limited,
contended resources under uncertainty — is a long-standing operational
control problem spanning scheduling, service operations, manufacturing, and
logistics. Deep reinforcement learning has been applied productively across
this space: graph-embedding-based Q-learning for combinatorial construction
heuristics on graphs [Dai, Khalil, Zhang, Dilkina & Song, 2017], attention-
based policies for vehicle routing [Nazari, Oroojlooy, Snyder & Takáč,
2018], and graph-neural-network state encodings for job-shop dispatching
[Zhang, Song, Cao, Zhang, Tan & Xu, 2020] — the latter especially close in
spirit to this paper's setting, since it also represents a scheduling
problem's combinatorial structure as a graph consumed by a GNN policy.
[Mazyavkina, Sviridov, Ivanov & Burnaev, 2021] survey this broader RL-for-
combinatorial-optimization literature; [Bengio, Lodi & Prouvost, 2021] —
published in this journal — give the methodological tour d'horizon this
paper's OR framing draws on most directly. Across this literature, the
learning algorithm is almost universally treated as a black box on top of a
graph or sequence encoding of the problem instance: the *simulator's own
execution trace* — which decisions caused which outcomes — is not itself
treated as a source of learning signal. A-E PN's prior work — the Action-Evolution Petri Net
formalism itself [Lo Bianco, Dijkman, Nuijten & van Jaarsveld, 2023], its
assignment-graph state/action representation and adapted PPO for infinite
state/action spaces [Lo Bianco, Dijkman, Nuijten & van Jaarsveld, 2025a],
and the GymPN library adding partial observability and multi-decision
processes [Lo Bianco, van Jaarsveld & Dijkman, 2025b] — establishes the
executable-model/DRL-solver pairing this paper builds on, but likewise
hands the resulting (S)MDP to off-the-shelf PPO. This paper's contribution is orthogonal to and
compatible with all of the above: it is a credit-assignment mechanism, not
a new policy architecture or problem encoding, and could in principle be
layered onto other graph-RL solvers for the same class of problems provided
they expose (or can be instrumented to expose) the same provenance
structure this paper exploits.

## 2.2 Credit assignment in reinforcement learning

The general problem of attributing a delayed, aggregated return to the
individual decisions that caused it is classical, and several return-
decomposition approaches address it without reference to multi-agent
structure. RUDDER [Arjona-Medina, Gillhofer, Widrich, Unterthiner,
Brandstetter & Hochreiter, 2019] learns to redistribute return along a
trajectory via a return-predicting LSTM's contribution analysis, turning a
delayed-reward RL problem into a regression problem. Hindsight Credit
Assignment [Harutyunyan, Dabney, Mesnard, Azar, Piot, Heess, van Hasselt,
Wayne, Singh, Precup & Munos, 2019] reweights past actions by their
likelihood of having caused an observed future outcome, using hindsight
rather than foresight. Both are *learned* redistribution mechanisms: a
neural network is trained to infer, statistically, which parts of a
trajectory mattered. This paper's mechanisms differ in kind, not just
degree: token provenance in an A-E PN is not inferred from correlational
patterns across many trajectories, it is *recorded exactly* by the
simulator at execution time — which token produced which, and hence which
decision is a deterministic prerequisite of which reward — so `s-ccf` and
`cfpk`'s credit signals require no learned component and carry no
approximation error from the attribution step itself. This is close in
motivation to model-based counterfactual credit assignment
[Mesnard, Weber, Viola, Thakoor, Saade, Harutyunyan, Dabney, Stepleton,
Heess, Guez, Moulines, Hutter, Buesing & Munos, 2021], which conditions
value functions on future events to build low-variance counterfactual
baselines in a model-free setting; `cfpk`'s DAG-replay mechanism can be read
as the model-based analogue of the same goal, made exact because the A-E PN
simulator itself is an available, cheap-to-fork causal model of the
environment, not something that needs to be learned.

## 2.3 Multi-agent and factored credit assignment

A parallel literature factors a single team reward across multiple
cooperating agents. Difference rewards [Wolpert & Tumer, 2001] credit each
agent with the marginal contribution of its own action relative to a
counterfactual baseline (an agent replaced by a fixed default); COMA
[Foerster, Farquhar, Afouras, Nardelli & Whiteson, 2018] learns this
counterfactual baseline via a centralized critic, marginalizing one agent's
action while holding the others fixed; value-decomposition approaches such
as VDN [Sunehag, Lever, Gruslys, Czarnecki, Zambaldi, Jaderberg, Lanctot,
Sonnerat, Leibo, Tuyls & Graepel, 2018] and QMIX [Rashid, Samvelyan,
Schroeder de Witt, Farquhar, Foerster & Whiteson, 2018] learn to decompose a
joint action-value into per-agent components under additivity or
monotonicity constraints. All four require the *entity partition* — which
agent is which, and hence which sub-reward belongs to which policy — to be
given as part of the problem specification. `s-ccf`/`cfpk` need no such
specification: the causal-component partition (or, for `cfpk`, no partition
at all) is derived automatically, per episode, directly from the A-E PN's
own token provenance, and can in principle vary in shape from episode to
episode as the realized case mix changes — a single-agent problem
(one policy, many concurrent cases) is factored the same way a genuinely
multi-agent one would be, without treating each case as a hand-specified
agent.

## 2.4 Potential-based reward shaping

Potential-based reward shaping [Ng, Harada & Russell, 1999] proves that
adding $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ to every step's reward, for any
potential function $\Phi$, cannot change a (finite-horizon or discounted)
MDP's optimal policy — the shaping term telescopes to a trajectory-
independent constant. This paper uses the SMDP-discounted generalization of
that theorem (§6b.1) not as a headline mechanism but as one of three
falsified alternative routes to closing the single-component bottleneck's
gap: a theorem-safe, empirically-neutral-to-harmful result, included
because it isolates *sample-efficiency* effects cleanly from the *bias*
effects that are this paper's main concern in §5, and because its
theoretical safety was independently verified in code (not merely cited)
before being used to interpret an empirical result.

## 2.5 Graph neural networks for combinatorial optimization and heterogeneous actor-critic architectures

Graph neural networks are now a standard state encoder for RL over
combinatorial structures with variable size and topology [Dai et al., 2017;
Zhang et al., 2020]; this paper's environments use a heterogeneous variant,
the Heterogeneous Graph Transformer [Hu, Dong, Wang & Sun, 2020], to encode
places and transitions as distinct node types with type-specific attention.
§6b.3's architectural finding — that a per-node actor decoder with
directed-only message passing can be *exactly* blind to state outside its
forward-reachable neighborhood, unlike a pooled critic — is, to our
knowledge, not previously documented for heterogeneous actor-critic graph
architectures in this setting, and is orthogonal to the specific choice of
HGT: the same argument applies to any message-passing GNN actor that scores
nodes locally without a global read-out, on any directed (non-bidirectional)
graph observation. We position this as a secondary, general contribution
for graph-RL practitioners on Petri-net- or flow-structured problems more
broadly, not specific to A-E PN's credit-assignment thesis.

## 2.6 Decomposition in operations research

Decomposing a hard, structured optimization problem into a master problem
and independent subproblems is a foundational OR technique: Dantzig-Wolfe
decomposition [Dantzig & Wolfe, 1960] for LPs with a block-angular
constraint structure, and Benders decomposition [Benders, 1962] for
mixed-integer programs with a complicating-variable structure, are the
classical instances; Dantzig-Wolfe and Benders are, in fact, dual to one
another on linear programs. This paper's framing — decompose the policy
gradient's *return* along the causal-component structure the environment's
own execution induces — is the RL analogue of the same idea: rather than
decomposing a static constraint matrix along a hand-identified block
structure, `s-ccf` decomposes a stochastic return along a block structure
*derived automatically, per episode*, from the simulator's own causal
record. We use this connection to frame the paper's contribution for an OR
audience already fluent in decomposition, not as a claim of technical
equivalence to LP/MIP decomposition methods.

## 2.7 Petri nets and learning

*[Placeholder — one paragraph situating A-E PN within the broader Petri-net
learning literature (e.g. process mining's use of Petri nets as discovered
process models, and prior non-DRL approaches to control/scheduling on
Petri nets). Deferred: this paragraph is about positioning A-E PN itself,
which is prior work carried over from the v1 outline, not new material this
investigation produced — lowest priority of the related-work subsections
and safe to fill in last, from the prior papers' own related-work sections
rather than a fresh literature pass.]*

---

## References (for `refs.bib`, verified against publisher/proceedings pages)

- Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T.,
  Brandstetter, J., & Hochreiter, S. (2019). RUDDER: Return decomposition
  for delayed rewards. *NeurIPS 2019*.
- Bengio, Y., Lodi, A., & Prouvost, A. (2021). Machine learning for
  combinatorial optimization: A methodological tour d'horizon. *European
  Journal of Operational Research*, 290(2), 405-421.
- Benders, J. F. (1962). Partitioning procedures for solving mixed-variables
  programming problems. *Numerische Mathematik*, 4(1), 238-252.
- Dai, H., Khalil, E. B., Zhang, Y., Dilkina, B., & Song, L. (2017). Learning
  combinatorial optimization algorithms over graphs. *NeurIPS 2017*.
- Dantzig, G. B., & Wolfe, P. (1960). Decomposition principle for linear
  programs. *Operations Research*, 8(1), 101-111.
- Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S.
  (2018). Counterfactual multi-agent policy gradients. *AAAI 2018*.
- Harutyunyan, A., Dabney, W., Mesnard, T., Azar, M. G., Piot, B., Heess,
  N., van Hasselt, H. P., Wayne, G., Singh, S., Precup, D., & Munos, R.
  (2019). Hindsight credit assignment. *NeurIPS 2019*.
- Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous graph
  transformer. *WWW 2020*, 2704-2710.
- Mazyavkina, N., Sviridov, S., Ivanov, S., & Burnaev, E. (2021).
  Reinforcement learning for combinatorial optimization: A survey.
  *Computers & Operations Research*, 134, 105400.
- Mesnard, T., Weber, T., Viola, F., Thakoor, S., Saade, A., Harutyunyan,
  A., Dabney, W., Stepleton, T. S., Heess, N., Guez, A., Moulines, E.,
  Hutter, M., Buesing, L., & Munos, R. (2021). Counterfactual credit
  assignment in model-free reinforcement learning. *ICML 2021*, PMLR
  139:7654-7664.
- Nazari, M., Oroojlooy, A., Snyder, L., & Takáč, M. (2018). Reinforcement
  learning for solving the vehicle routing problem. *NeurIPS 2018*.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under
  reward transformations: Theory and application to reward shaping.
  *ICML 1999*, 278-287.
- Rashid, T., Samvelyan, M., Schroeder de Witt, C., Farquhar, G., Foerster,
  J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation
  for deep multi-agent reinforcement learning. *ICML 2018*, PMLR
  80:4295-4304.
- Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V.,
  Jaderberg, M., Lanctot, M., Sonnerat, N., Leibo, J. Z., Tuyls, K., &
  Graepel, T. (2018). Value-decomposition networks for cooperative
  multi-agent learning. *AAMAS 2018* (arXiv version 2017).
- Wolpert, D. H., & Tumer, K. (2001). Optimal payoff functions for members
  of collectives. *Advances in Complex Systems*, 4(2-3), 265-279.
  **Note: v1 outline cited this as 2002 — corrected to 2001 per the
  original journal publication; a 2002 edited-collection reprint also
  exists (Tumer & Wolpert, *Collectives and the Design of Complex
  Systems*) if the paper prefers to cite that venue instead.**
- Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020).
  Learning to dispatch for job shop scheduling via deep reinforcement
  learning. *NeurIPS 2020*.

**Prior work (this paper's own authorship line), verified via arXiv directly:**

- Lo Bianco, R., Dijkman, R., Nuijten, W., & van Jaarsveld, W. (2023).
  Action-Evolution Petri Nets: A framework for modeling and solving dynamic
  task assignment problems. arXiv:2306.02910.
- Lo Bianco, R., Dijkman, R., Nuijten, W., & van Jaarsveld, W. (2025a). A
  universal approach to feature representation in dynamic task assignment
  problems. arXiv:2507.03579.
- Lo Bianco, R., van Jaarsveld, W., & Dijkman, R. (2025b). GymPN: A library
  for decision-making in process management systems. arXiv:2506.20404.

*(Publication venues for all three — journal/conference vs. arXiv-only —
still need confirming before submission; author-order and titles above are
verified directly from the arXiv abstract pages, not recalled.)*
