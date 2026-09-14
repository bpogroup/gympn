# What cfpk actually does

*A standalone explanation of the mechanism, not the results. If you want
the experiment history and numbers, see CAUSAL_LINEAGE_RETHINK.md §7
(design) and the X10/X11/X12 result blocks (outcomes). This document is
just: what is the code doing, step by step.*

## The one-sentence version

**cfpk is ordinary PPO that occasionally pauses mid-episode, tries both
"what if I did this" and "what if I did that instead" using the simulator
itself, and — only when the difference is clearly bigger than noise —
nudges the policy toward whichever one actually worked better.**

That's it. There's no lineage graph, no credit redistribution, no new
reward signal. It's PPO with a second, much smaller training signal bolted
on the side.

## Why this exists (one paragraph of context)

Every earlier method in this project (lrq, lcv, lva, ...) tried to
extract better learning signal from the *provenance DAG* — the record of
which token produced which other token, built by watching one factual
trajectory unfold. All of them plateaued at the same final performance as
plain PPO. The diagnosis (see the R4 result in CAUSAL_LINEAGE_RETHINK.md):
the *information* those methods wanted was real, but a single factual
trajectory is too noisy to read it off reliably — one sample against
service-time noise. cfpk sidesteps the whole lineage-DAG apparatus and
gets the same kind of information a completely different way: by asking
the simulator to actually show you both outcomes, with the noise
cancelled out by design (explained below).

## Two ordinary training loops, one extra step

Picture PPO training as usual: the agent plays an episode, action by
action, storing `(state, action, reward)` at each step, then at the end of
every epoch it updates the policy network using those stored transitions.

cfpk keeps every part of that unchanged. It inserts exactly one new thing
into the *middle* of episode collection, at some (not all) decision points:

```
for each decision in the episode:
    action = policy.act(state)              #  <- unchanged, ordinary PPO
                                              #
    maybe: FORK HERE                         #  <- the new part
                                              #
    state, reward, done = env.step(action)   #  <- unchanged, ordinary PPO
```

The fork, when it happens, does not change what the agent actually does
in the real episode. It's a side experiment: the code temporarily jumps
away to try something, learns from what it finds, and then jumps back to
exactly where it was, as if nothing happened.

## What a "fork" actually is

Say the agent is at decision state `s`, and the policy just chose action
`a` (say, "assign this task to employee 2"). Before actually taking that
step in the real episode, cfpk (with some probability — see "how often"
below) does this:

1. **Look at the alternative.** The policy network doesn't just output the
   chosen action — it outputs a probability for *every* available action
   at `s`. cfpk picks the second-most-likely one, call it `b` ("assign to
   employee 0" instead). This is the "devil's advocate" — the option the
   policy itself considered plausible but didn't pick.

2. **Save an exact snapshot of the simulator.** Not an approximation — a
   full copy of the Petri net's current state (every token, every place,
   the clock, everything in flight). This is possible because the
   simulator is a program we control, not a black box.

3. **Play out "what if I did `a`?"** Restore the snapshot, actually take
   action `a`, and then let the *current policy* keep playing
   (deterministically — always the top choice, no randomness) for a while
   longer. Add up the rewards earned along the way, discounted the same
   way the rest of training discounts rewards. Call the total `Q(a)`.

4. **Play out "what if I did `b`?"** Restore the *same* snapshot again,
   take action `b` instead this time, and let the same policy play out
   from there the same way. Call the total `Q(b)`.

5. **Compare `Q(a)` and `Q(b)`.** If `Q(a)` is meaningfully bigger, that's
   evidence the policy was right to pick `a`. If `Q(b)` turns out bigger,
   that's evidence it should have picked `b` instead — a genuine mistake,
   caught by direct experiment rather than inferred from noisy rewards.

6. **Restore the real episode exactly** and continue collecting the
   ordinary trajectory as if the fork never happened. The agent still
   took action `a` in the real episode; nothing about the actual rollout
   changes because of the fork.

This is the "counterfactual" in the name: literally trying the road not
taken, in a simulator that lets you rewind.

## Why noise doesn't kill this (the "common random numbers" trick)

Environments here are stochastic — service times are random, arrivals are
random. If you just played out `a` once and `b` once independently, the
difference `Q(a) − Q(b)` would be swamped by which run happened to get
lucky draws, not by which action was actually better. That's exactly the
problem that sank the lineage-DAG approach.

cfpk fixes this with a trick borrowed from simulation science, called
**common random numbers**: before playing out `a`, the random number
generator is seeded with a specific value. Before playing out `b`, it's
reseeded with the *exact same value*. So both playouts see the *same*
sequence of "coin flips" for service times, arrivals, etc. — the only
thing that differs is the one decision, `a` vs `b`, and everything that
causally follows from it. Any difference in the outcome is now
attributable to the decision, not to which run got lucky.

The code does this **3 times** (configurable — `cf_reps`), each time with
a *different* shared seed across the `a`/`b` pair, so you get 3 paired
comparisons — enough to also measure how much residual noise is left even
after the pairing.

## The gate: don't trust small differences

Even with paired noise-cancellation, 3 samples is not a lot. So cfpk
computes, from those 3 paired differences, both the average gap and its
standard error (how uncertain that average is). It only acts on the
comparison if:

```
|average gap|  >  2 × (standard error)
```

If the gap is small relative to its own uncertainty — the two options
looked about equally good, or the noise is too high to tell — **cfpk does
nothing with this fork.** No preference is recorded, no gradient is
computed, it's simply discarded. This is the mechanism that keeps cfpk
from chasing noise: it would rather stay silent than act on a coin flip
that came up heads by chance.

## What happens with a passed gate: the preference

When the gate *does* pass, cfpk records a tiny fact: "at this state, `a`
was better than `b`" (or vice versa). It does **not** turn this into a
reward, an advantage, or anything that touches the normal PPO machinery.
Instead, at the end of the epoch, after the usual PPO policy update has
already happened, there's one extra, small training step: for every
recorded preference `(winner, loser)` at state `s`, nudge the policy
network so that it assigns a little more probability to `winner` over
`loser` at that state — a standard "ranking" loss (the same shape used in
preference-learning methods like DPO): push
`log P(winner) − log P(loser)` up. Nothing else about the network or the
PPO loss changes.

This runs for a couple of gradient steps (`cf_updates`, default 2) using
whatever preferences got collected that epoch, with its own separate
optimizer step.

## Truncation: not playing every "what if" all the way to the end

Playing a full episode twice per fork, for every fork, would be
expensive. So each playout in step 3/4 above is cut short once it's
advanced 6 time-units past the fork point (`cf_lookahead`) — instead of
continuing forever, the code asks the value network "what do you think
happens from here?" and uses that estimate to stand in for the rest. This
keeps each fork's cost bounded.

## Cost controls (how much of this happens)

Because forking is expensive (it means running extra simulated time),
cfpk is deliberately sparing:

- **`cf_fork_prob` (default 0.25):** only a 1-in-4 chance of forking at
  any given decision, not every decision.
- **`cf_max_forks` (default 2):** at most 2 forks per episode, no matter
  how long the episode is.
- **`cf_reps` (default 3):** 3 paired playouts per fork (the CRN trick
  above).
- **`cf_lookahead` (default 6):** each playout stops after 6 time-units
  and estimates the rest instead of simulating it.

Even with these caps, forking roughly triples the wall-clock cost of
training compared to plain PPO — that's the current known cost, not yet
optimized (a further speedup, "stop early once both playouts have clearly
reconverged," is designed but not yet built).

## "cfp" vs "cfpk" — the one knob that changes between them

There are two variants tested so far, differing in exactly one thing —
how strongly the preference nudge is applied over the course of training:

- **`cfp`:** the nudge starts at full strength and is gradually turned
  down to zero by the end of training (an "anneal"). The idea was: use
  the extra signal early to help exploration, then get out of the way so
  the agent ends training as pure, unmodified PPO.
- **`cfpk`:** the nudge stays at full strength for the *entire* training
  run — no anneal, no turning it off.

The result so far (see CAUSAL_LINEAGE_RETHINK.md for the numbers): `cfp`
(the fading version) ended up performing about the same as plain PPO —
the preferences were being learned during training but then "forgotten"
once the nudge faded away and ordinary PPO gradients took back over.
`cfpk` (the constant version) does not have that forgetting problem,
because the pressure never goes away — and it currently produces the best
results of any method tried in this project.

## What cfpk is *not* doing

To be precise about the boundaries of the mechanism:

- It does **not** change the reward the agent receives in the real
  episode. The real trajectory and its rewards are completely standard.
- It does **not** use the provenance/lineage DAG at all — no token
  parent-tracking, no credit redistribution. That whole machinery
  (lrq/lcv/lva and friends) is unrelated to cfpk.
- It does **not** require anything specific about the environment (no
  "resources," "pools," or task types are read anywhere in the fork code)
  — it only uses the generic action list (`actions_dict`) every
  environment already exposes, and the policy's own probability outputs.
  That's what makes it work unmodified on every environment tested so
  far, from assignment problems to abstract choice/foreclosure envs.
- The forked playouts themselves never affect what the main episode does;
  the simulator state is always restored exactly afterward, verified by a
  test that checks the two runs (with and without forking) produce
  bit-identical trajectories.

## Where the code lives

- `gympn/counterfactual.py` — the fork/compare/gate logic (`maybe_fork`),
  entirely self-contained.
- `gympn/agents.py` — the hook that calls `maybe_fork` during episode
  collection (`run_episode`), and the preference-loss training step
  (`_fit_cf_preferences`), called once per epoch.
- `gympn/train.py` — the `--cf_*` command-line flags that configure all
  of the above (fork probability, reps, gate threshold, lookahead, max
  forks per episode, loss coefficient, whether to anneal).
- `examples/paper_examples/suite/_test_cfp.py` — unit tests verifying the
  snapshot/restore is exact, the main episode is unaffected, and the
  preference loss actually moves the policy in the right direction.