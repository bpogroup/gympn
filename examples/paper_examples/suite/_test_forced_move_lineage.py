r"""Gate for forced single-binding moves in causal mode (postpone OFF).

A state with exactly one allowed action is not a decision. It must still update
the LINEAGE (its tokens carry provenance to whatever consumes them), but it must
NOT be registered as a decision -- otherwise it enters the training batch as a
no-op sample, and, because the old branch keyed on causal_rl, the causal arms
faced a different decision sequence from the ppo baseline they were compared
against.

  F1 SAME-STRUCTURE  causal and non-causal envs consult the agent on exactly
                     the same states (this is what the old shortcut broke).
  F2 LINEAGE-KEPT    the forced firing is still in the trace, still links its
                     consumed tokens to its produced tokens -- provenance walks
                     through it.
  F3 NOT-A-DECISION  it is not in get_action_transitions(), so trace decisions
                     stay 1:1 with the agent's decisions.
  F4 ALIGNED         over a full episode, #action-transitions == #agent calls.

Run: python _test_forced_move_lineage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from ncopies_env import make_n_copies

N = 2
LENGTH = 20
fails = []


def build(causal):
    pn = make_n_copies(N, causal_rl=causal, allow_postpone=False)
    pn.length = LENGTH
    if causal:
        import types, uuid
        for place in pn.places:
            for tok in place.marking:
                setattr(tok, '_id', str(uuid.uuid4()))
        pn.causal_trace._pn = pn
        pn.causal_trace._static_comp_cache = None
        try:
            pn.causal_trace.flush()
        except Exception:
            pass
        sent = types.SimpleNamespace(_id="__initial__")
        for place in pn.places:
            for tok in place.marking:
                pn.causal_trace.register_token(tok, sent, parent_tokens=[], time=0)
        pn.causal_trace.register_transition(
            transition=sent, input_tokens=[],
            output_tokens=[t for p in pn.places for t in p.marking],
            is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)


def rollout(causal, seed):
    """Fixed policy (always index 0) so the two arms differ only structurally.
    Returns (n_agent_calls, sizes of the action-set at each call, env)."""
    import random
    random.seed(seed)
    env = build(causal)
    state = env.reset()
    calls, sizes, done = 0, [], False
    while not done:
        calls += 1
        sizes.append(len(env.pn.pn_actions))
        state, _, done, _, _ = env.step(0)
    return calls, sizes, env


# ---- F1: identical decision structure across causal / non-causal -----------
c_calls, c_sizes, c_env = rollout(True, 7)
n_calls, n_sizes, _ = rollout(False, 7)
print(f"F1 agent calls  causal={c_calls}  non-causal={n_calls}")
if c_calls != n_calls or c_sizes != n_sizes:
    fails.append(f"F1 decision structure differs: {c_calls} vs {n_calls} calls")
if any(s < 2 for s in n_sizes):
    fails.append(f"F1 a forced state (action-set size 1) still reached the agent")

# ---- F2/F3: forced firings are lineage nodes, not decisions ----------------
tr = c_env.pn.causal_trace.transition_history.transitions
actions = c_env.pn.causal_trace.transition_history.get_action_transitions()
forced = [t for t in tr if t["is_action"] is False and t["input_tokens"] and t["output_tokens"]]
print(f"F2 trace: {len(tr)} transitions, {len(actions)} decisions, "
      f"{len(forced)} lineage-only firings with token flow")
if not forced:
    fails.append("F2 no lineage-only firings recorded at all")
else:
    linked = sum(1 for t in forced if t["output_tokens"])
    if linked == 0:
        fails.append("F2 lineage-only firings carry no produced tokens")

# provenance must walk THROUGH a forced firing: some token produced by a
# lineage-only transition must appear as a parent of a later token
th = c_env.pn.causal_trace.token_history
produced_by_forced = {tid for t in forced for tid in t["output_tokens"]}
walks_through = False
for tid in list(th.tokens.keys()) if hasattr(th, 'tokens') else []:
    info = th.get_token(tid) or {}
    if set(info.get("parents", [])) & produced_by_forced:
        walks_through = True
        break
print(f"F2 provenance walks through a forced firing: {walks_through}")
if produced_by_forced and not walks_through:
    fails.append("F2 no token descends from a forced firing -- lineage is cut")

if any(t in actions for t in forced):
    fails.append("F3 a forced firing was registered as a decision")

# ---- F4: trace decisions align 1:1 with agent calls ------------------------
print(f"F4 alignment: {len(actions)} action-transitions vs {c_calls} agent calls")
if len(actions) != c_calls:
    fails.append(f"F4 misaligned: {len(actions)} action-transitions vs {c_calls} agent calls")

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 4/4")
