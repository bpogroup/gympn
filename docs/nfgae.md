# Net-Factored Credit (NF-GAE)

Net-factored generalized advantage estimation (NF-GAE) is gympn's credit
assignment method for nets that contain several independent parts. It is
switched on with one training argument, `nfgae=True`, and changes nothing
about how a problem is defined. This page explains what it does, when it
applies, and how to use it.

## The problem it solves

A business process model often contains several parts that never exchange
tokens: separate processes, regional offices, independent production lines,
or N copies of the same workflow sharing one agent. All of them are decided
by one policy and produce one reward stream.

Standard PPO credits every decision with the total future reward of the whole
net. A decision in one part is therefore rewarded or blamed for what happens
in every other part, although it cannot influence them. That extra reward is
pure noise in the gradient. A critic (value baseline) can remove its expected
value, but never its variance, and the variance grows with the number of
independent parts. In practice PPO learns more slowly as the net grows, even
when every part is individually easy.

NF-GAE removes this noise by crediting each decision only with the rewards of
its own part of the net.

## Net components

gympn derives the parts from the net itself. Build the undirected graph whose
nodes are the places and transitions, with an edge between a place and every
transition that consumes from or produces into it. Its connected components
are the **net components**. Two transitions that share any place, be it a
queue, a resource pool or a counter, are in the same component. The partition
depends only on the net structure, never on the marking or on what happened
during an episode.

Transitions that can never fire are pruned before the partition is computed
(a transition is live when all its input places can ever be marked). This
matters when an empty place would otherwise glue independent parts together.

Inspect the partition of a problem:

```python
partition = problem.net_partition()      # transition id -> component index
print(sorted(set(partition.values())))   # e.g. [0, 1, 2]
```

If this prints a single component, NF-GAE reduces exactly to PPO and there is
nothing to gain. If it prints several, every one of them has its own queues,
resources and rewards, and NF-GAE applies.

## What NF-GAE computes

The environment is a semi-Markov decision process: decisions happen at
simulator clock times, and the time between two decisions (the sojourn) varies.
gympn's SMDP variant of GAE discounts the continuation after a decision by
`exp(-beta * tau)`, where `tau` is the sojourn; this is `smdp_discount=True`
with rate `beta`. With `beta=0` nothing is discounted.

NF-GAE runs one such SMDP-GAE per component, on that component's own decision
clock. For a component `c` with decision steps `t_1 < t_2 < ...`:

- `R_i` is the reward produced by `c`'s own events between `t_i` and `t_{i+1}`;
- `d_i = exp(-beta * (time[t_{i+1}] - time[t_i]))`, and 0 after `c`'s last decision;
- `delta_i = R_i + d_i * V_c(s_{t_{i+1}}) - V_c(s_{t_i})`;
- `A_i = delta_i + lam * d_i * A_{i+1}`.

The critic is component-indexed: `V_c` is read by pooling the critic's node
embeddings over component `c`'s action nodes only, and it is regressed on
`c`'s own targets. The advantage `A_i` is then used by PPO exactly as usual.

Three properties follow from the construction:

- **Exact at one component.** With a single component NF-GAE is term for term
  the SMDP-GAE that PPO uses, so turning it on is never harmful.
- **Unbiased.** Conditional on the current state, the future of each component
  is independent of the decisions taken in the others, so dropping their
  rewards does not change the expected gradient (exactly at `lam=1`, up to the
  usual GAE bias otherwise).
- **Lower variance.** The discarded term is the other components' return
  variance, which no state-dependent baseline can remove. The gap grows
  linearly with the number of components.

## Conditions

The independence argument needs three things, and gympn enforces or provides
each of them:

1. **A local actor.** The logit of an action must depend only on its own
   component's marking. gympn's graph actor satisfies this by construction:
   messages flow only along the net's arcs, so no information crosses a
   component boundary. Do not use `global_context=True` on the actor with
   NF-GAE.
2. **No global postponement.** A global postpone action reads the whole net and
   freezes every component at once, which couples them. Either leave
   postponement off (the default) or use component-scoped postponement, which
   gives every component its own postpone option (see
   [Postponement](./postponement.md#postponement-per-net-component)).
   Training raises an error if NF-GAE is combined with a global postpone.
3. **Local randomness.** An event's behaviour may depend only on its own input
   tokens and fresh noise. This is how events are normally written.

## Usage

```python
problem.training_run(length=20, args_dict={
    "algorithm": "ppo-clip",
    "nfgae": True,
    "smdp_discount": True,   # time-aware discount; False gives the undiscounted objective
    "beta": 0.5,             # discount rate per unit of simulator time
    "lam": 0.99,
})
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `nfgae` | `False` | Per-component advantages and critic. |
| `smdp_discount` | `False` | Discount by `exp(-beta * tau)` per sojourn instead of a constant `gam`. |
| `beta` | `0.0` | SMDP discount rate. Ignored when `smdp_discount` is off. |
| `lam` | `0.99` | GAE lambda, as in PPO. |
| `local_obs` | `False` | Component turns: each decision observes and encodes only its own component (see below). |

The same flags work on the command line of a training script:
`--nfgae true --smdp_discount true --beta 0.5`.

### Local observations

With `local_obs=True` the decision process offers one component at a time and
the policy sees only that component's subgraph. This makes encoding cheaper
on large nets and is the strictest form of condition 1. It requires `nfgae`
and `allow_postpone=False`.

## A worked example

Two copies of the minimal task assignment problem in one net. Each copy has
its own arrivals, queue, employees and rewards, so the partition has two
components and NF-GAE credits each `start` decision only with the completions
of its own copy.

```python
import copy
from simpn.simulator import SimToken
from gympn import GymProblem, RandomSolver

def add_agency(p, k):
    arrival = p.add_var(f"arrival{k}", var_attributes=["task_type"])
    waiting = p.add_var(f"waiting{k}", var_attributes=["task_type"])
    busy = p.add_var(f"busy{k}", var_attributes=["task_type", "resource_id"])
    employee = p.add_var(f"employee{k}", var_attributes=["code_employee"])
    arrival.put({"task_type": 0}); arrival.put({"task_type": 1})
    employee.put({"code_employee": 0}); employee.put({"code_employee": 1})
    p.add_event([arrival], [arrival, waiting], name=f"arrive{k}",
                behavior=lambda a: [SimToken(a, delay=1), SimToken(a)])
    p.add_action([waiting, employee], [busy], name=f"start{k}",
                 behavior=lambda c, r: [SimToken((c, r), delay=1 if c["task_type"] == r["code_employee"] else 2)])
    p.add_event([busy], [employee], lambda b: [SimToken(b[1])], name=f"complete{k}",
                reward_function=lambda x: 1)

problem = GymProblem()          # postponement off
for k in range(2):
    add_agency(problem, k)

print(sorted(set(problem.net_partition().values())))   # [0, 1]

problem.training_run(length=10, args_dict={
    "algorithm": "ppo-clip", "nfgae": True,
    "smdp_discount": True, "beta": 0.1,
    "epochs": 50, "episodes": 10,
    "logdir": "data/train", "name": "two_agencies",
})
print(copy.deepcopy(problem).testing_run(RandomSolver(), length=10))
```

If the two copies shared the employee pool, `employee` would be one place
touched by both `start` transitions, the partition would collapse to one
component, and NF-GAE would be identical to PPO. That is the correct
behaviour: a shared resource is a real coupling, and pre-empting a resource in
one process does influence the other.

## When to use it

- **Use it** when `net_partition()` reports more than one component. The gain
  grows with the number of components, and it costs nothing at one.
- **It is not a substitute** for a good model of the coupling. If two parts
  interact through a shared place, they form one component and are credited
  jointly, as they must be.
- **Combine with component-scoped postponement** when the agent must be able
  to wait: `allow_postpone=True` together with `postpone_scope='component'`.

## See also

- [Postponement](./postponement.md), including per-component postponement.
- [Algorithms](./algorithms.md) for PPO and the advantage estimators.
- [Trajectories and advantages](./reference/data.md) for the `TrajectoryBuffer` API.
