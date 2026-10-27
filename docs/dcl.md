# Deep Controlled Learning (DCL)

Deep Controlled Learning (Temizöz et al., EJOR 2025) learns a policy network
by approximate policy iteration. A simulation-based planner labels states with
better actions than the current policy would take, and a classifier is trained
to imitate those labels. The trained network then becomes the planner's base
policy for the next round. The result is a policy that needs no simulator when
it is used: one forward pass per decision, like a PPO policy.

gympn implements DCL in `gympn.policy_iteration`. It also adds a
**certificate-factored** variant that exploits the independence of a net's
components. This page explains both and how to use them.

## When to use it

Model-free RL (PPO, NF-GAE) learns from the rewards of its own episodes. It
can stall at the level of a good heuristic when the improvements are rare,
time-sensitive decisions whose effect shows up much later. DCL does not depend
on the policy's own exploration. At every recorded state it compares the
options directly, by simulation, and so it can find and teach improvements
that PPO never samples often enough to learn.

The cost is simulation. Each label needs tens of rollouts per option to the
end of the episode, so collection dominates the run time. Distillation itself
takes minutes.

On the BPIC 2015 benchmark (five municipalities, shared staff), on 100
held-out days with random = 0 and heuristic = 1:

| Method | Held-out score |
| --- | --- |
| PPO / NF-GAE, from scratch or from imitation | ≤ 1.03 |
| Rollout planner over the heuristic (needs a simulator at decision time) | 1.28 |
| DCL, round 1, Sequential Halving with 32 rollouts per option, 40 days | 1.22 |
| DCL with component-local loss, same data | 1.28 |
| Certificate-factored DCL, round 1, same compute | 1.31 |

## The planner

`RolloutPlanner` improves a base policy one decision at a time:

1. The base policy picks an action. The planner takes the **net component**
   that action belongs to (`GymProblem.net_partition()`, the static or the
   marking certificate; see [NF-GAE](./nfgae.md#net-components)).
2. It plays every distinct **option** of that component out to the end of the
   episode with the base policy. An option is a class of equivalent bindings:
   same transition, same token values (`option_key`).
3. It scores each option by the **component's own return** and chooses the best.

Under the certificate nothing outside the component depends on the choice.
That makes three simplifications exact:

- **Restricted rollouts** (`restrict=True`). The rollouts simulate only the
  deciding component (`GymProblem.sim_only`). This removes the other
  components' cost and noise.
- **Common random numbers through keyed streams** (`keyed=True`). In rollout
  `j` of every option, the `k`-th firing of each transition draws the same
  random numbers, however the trajectories differ (`GymProblem.stream_seed`).
- **Coalescing** (`coalesce=True`). When an option's rollout reaches a state
  that the base choice's rollout has already visited
  (`GymProblem.state_fingerprint()`), it stops there and takes the remaining
  return, which it would have collected identically.

Base policies are callables `env -> action index`:

- `HeuristicBase(heuristic)` wraps a gympn heuristic (round 1).
- `NetworkBase(policy)` wraps a policy network, greedy by default (later rounds).

### Choosing among the options

| Argument | Selection |
| --- | --- |
| `rollouts=R` (default) | Every option gets `R` paired rollouts; the best mean wins, and the base's choice stands unless another option is strictly better (`margin` raises the bar). |
| `sh_budget=M` | **DCL's selection**: Sequential Halving with common random numbers, `M` rollouts per option in total, spread over `ceil(log2 options)` rounds; the better half survives each round. Use with `margin=0`. |
| `test=k, min_t=2` | A departure from the base is kept only if `k` fresh paired rollouts confirm it with a t statistic of at least `min_t`. Useful when the planner itself is the policy. |
| `race_eps=e` | Adaptive racing on paired rollouts: stops early on clear winners and on near ties, and returns the survivor set as the label. In our experiments it was cheaper but worse than Sequential Halving. |

Use `sh_budget` for DCL. On BPIC 2015 it scored 1.40 on the evaluation days,
against 1.33 for 8 uniform rollouts per option.

### Hidden future

The simulator state may hold information that no policy can observe, such as
the already-drawn arrival times of future cases. Rollouts that start from that
state would see the future, and their labels would teach the network
decisions it cannot justify from its observation. Pass
`resample=fn(problem)`: it is called at the start of every rollout to redraw
the hidden quantities, with the same draws for every option.

## Collecting and distilling

`collect_decisions` plays the planner on one training episode per seed. It
records every decision as the observation graph the policy network sees,
together with the planner's choice and every option's score. `distill` trains
the network on these labels with Adam.

```python
from gympn.policy_iteration import (RolloutPlanner, HeuristicBase, NetworkBase,
                                    collect_decisions, distill)

def make_env():               # module level: collection runs in worker processes
    ...                       # build the AEPN_Env (flat_obs=True recommended)

def make_planner(env):
    return RolloutPlanner(HeuristicBase(my_heuristic), sh_budget=32, margin=0.0,
                          component=True, seed=1)

graphs = collect_decisions(make_env, make_planner, seeds=range(2_000_000, 2_000_040),
                           workers=8)
policy = make_policy_network(args, metadata=make_env().pn.make_metadata())
hist = distill(policy, graphs, epochs=15, lr=1e-3, batch_size=32, tau=0.25)
torch.save(policy, "dcl_round1.pth")
```

`make_env` and `make_planner` must be module-level functions when
`workers > 1`, because collection runs in spawned processes. Use training
seeds that differ from your evaluation and test seeds.

### Targets

The loss is a cross-entropy over **option classes**. The network's probability
is summed over each class's equivalent bindings, so the network is not asked
to prefer one of several identical bindings.

- `tau=0`: hard targets, all mass on the planner's choice.
- `tau>0`: **soft targets**, `softmax(scores / tau)` over the compared options
  (`soft_targets`), as in the current DCL in DynaPlex. Options that are nearly
  as good as the best are not pushed down. `tau=0.25` worked best on BPIC 2015.

### Model selection

`distill(evaluate=fn, select="key")` calls `evaluate(policy) -> dict` every
`eval_every` epochs and keeps the weights that maximize `key`. The score on
the selection episodes is then optimistic. Report the selected network on
separate held-out episodes, or report the final epoch.

### Rounds

The distilled network is the next base policy:

```python
def make_planner_r2(env):
    base = NetworkBase(torch.load("dcl_round1.pth", weights_only=False))
    return RolloutPlanner(base, sh_budget=32, margin=0.0, component=True, seed=2)

graphs2 = collect_decisions(make_env, make_planner_r2, seeds=range(2_100_000, 2_100_040), workers=8)
policy2 = torch.load("dcl_round1.pth", weights_only=False)     # warm start
distill(policy2, graphs2, epochs=15, tau=0.25)
```

With a network base, `lockstep=True` (the default) plays all rollouts of a
decision side by side with batched forward passes.

## Certificate-factored DCL

Plain DCL treats the net as one system: it simulates whole episodes and
trains on whole-net observations. When the net has several components, the
certificate allows two changes, both exact.

### Component-local labels

A label at a decision in component `c` says how `c`'s options compare. It
says nothing about how much probability the network should give to the actions
of other components at that moment. With `renorm=True` the loss renormalizes
the class probabilities over the deciding component's options before the
cross-entropy:

```python
distill(policy, graphs, epochs=15, tau=0.25, renorm=True)
```

This changes only the loss, not the data. On BPIC 2015 it accounts for most
of the factored variant's gain: 1.22 to 1.28 on the same 40 days.

### Component-factored collection

The component's future depends only on its own marking, and the actor's
logits for its actions read only its own subgraph (the AEPN encoder's
attention is masked to the component). So each component can be **simulated
and labelled alone**, with the same distribution of states and labels it has
inside the whole net.

```python
from gympn.policy_iteration import episode_components, collect_component_decisions

comps = episode_components(make_env)            # static components, as sets of transition ids
days = [35, 24, 58, 56]                         # component-days per component
tasks = [(2_000_000 + d, members) for members, n in zip(comps, days) for d in range(n)]
graphs = collect_component_decisions(make_env, make_planner, tasks, workers=8)
distill(policy, graphs, epochs=15, tau=0.25)
```

Each task `(seed, members)` simulates one component on one training day. A
component's decisions are then labelled at the cost of that component alone,
and the labels are component-local by construction: the observation holds
only the component, so `renorm` is not needed.
More importantly, simulation can be **allocated**: spend more component-days
where labels change the policy and fewer where the base is already right.
On BPIC 2015 we allocated in proportion to the planner's gain per component,
estimated on a pilot of a few days.

`episode_components` uses the static certificate on purpose. The marking
certificate after a reset can drop start-of-episode transitions (for example,
the release of a backlog), and a component simulated without them would
start from the wrong marking.

On BPIC 2015 the factored variant reached 1.31 (3 training seeds) against
1.22 for plain DCL. It used about the same number of labelled decisions (7,213 vs 7,047)
and about half the collection wall time (18 vs 34 minutes on 8 workers).

### Conditions

The factored variant needs the same conditions as [NF-GAE](./nfgae.md#conditions):

1. **A local actor.** Keep the actor's component mask on. With the
   `clock_node` observation option the clock node has no component and only
   outgoing edges, so locality is preserved.
2. **No global postponement.** Use `postpone_scope='component'` if the policy
   must be able to wait.
3. **Local randomness.** Events draw only from their own input tokens and
   fresh noise.

## Observation options that help

Imitation is more sensitive to the observation than RL. Two `GymProblem`
options made a clear difference on BPIC 2015 (flat observations):

- `merge_options=True`: one action node per option, carrying
  `log(multiplicity)` as a feature; the longest-waiting (FIFO) binding is
  fired. Labels then refer to options rather than to arbitrary copies of
  identical bindings.
- `clock_node=True`: a dedicated clock node with the time of day. Many
  planner departures depend on how much of the day is left.

Set them on both `env.pn` and `env.frozen_pn`, because a reset starts from the
frozen copy.

## Reproducing the BPIC 2015 results

`examples/paper_examples/suite/scripts/run_pi.py` runs one round (collection,
distillation, held-out scoring), and `scripts/run_dcl.sh` runs the rounds:

```bash
# plain DCL, round 1
python scripts/run_pi.py env=bpic15 days=40 sh=32 fair=1 merge=1 clocknode=1 tau=0.25
# the same data, component-local loss
python scripts/run_pi.py env=bpic15 days=40 sh=32 fair=1 merge=1 clocknode=1 tau=0.25 renorm=1
# certificate-factored DCL
python scripts/run_pi.py env=bpic15 days=40 sh=32 fair=1 merge=1 clocknode=1 tau=0.25 cf=35,24,58,56
```

## See also

- [NF-GAE](./nfgae.md) for net components and the certificate.
- [Postponement](./postponement.md) for component-scoped waiting.
- [Policy iteration API](./reference/policy_iteration.md).
