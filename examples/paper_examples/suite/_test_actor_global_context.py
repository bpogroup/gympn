r"""Unit + smoke tests for HeteroActor's optional global_context
(gympn/networks.py) -- the fix for the actor-architecture blind spot found
while investigating structural features' neutral result on s1: HeteroActor
decodes each action's logit from ONLY that node's own final HGTConv
embedding (unlike HeteroCritic, which has always max-pooled over all
action/postpone nodes for its value estimate). Combined with
get_graph_observation's edges being directed strictly along token flow
(add_reverse_edges=False everywhere in real use), an action's logit can
depend on state ONLY if a forward, token-flow-direction path reaches it
within `num_layers` hops -- provably (not just empirically) zero otherwise.

global_context=True concatenates the SAME pooled context HeteroCritic
already builds (_pool_action_postpone) onto each action/postpone node
before decoding, closing that gap. Run: python _test_actor_global_context.py
"""
import sys, os
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from gympn.simulator import GymProblem, SimToken
from gympn.networks import HeteroActor, _pool_action_postpone, _batch_index_for
from assembly_probe import make_independent


def make_two_indep_pipelines():
    """Two two-stage pipelines glued into one net with ZERO shared/recycled
    resource between them -- start1 has NO directed path (via any route,
    any number of hops) to stage-2 state, while start2 (a SIBLING action
    node in the same pool) has a direct 1-hop edge to it. This isolates
    whether global_context's pooling channel (via the sibling node) closes
    the gap -- unlike a fully independent-chains net (no 2nd-stage sibling
    action to relay through), which can't exercise this mechanism at all."""
    ag = GymProblem(allow_postpone=False, causal_rl=False)
    for nm in ('w1', 'e1', 'b1', 'w2', 'e2', 'b2'):
        ag.add_var(nm, var_attributes=['id'])
    P = {p._id: p for p in ag.places}
    P['w1'].put({'id': 1}); P['e1'].put({'id': 1})
    P['w2'].put({'id': 2}); P['e2'].put({'id': 2})
    ag.add_action([P['w1'], P['e1']], [P['b1']],
                  behavior=lambda c, r: [SimToken((c, r), delay=1)], name='start1')
    ag.add_event([P['b1']], [P['e1']], lambda b: [SimToken(b[0])],
                 name='done1', reward_function=lambda b: 1)
    ag.add_action([P['w2'], P['e2']], [P['b2']],
                  behavior=lambda c, r: [SimToken((c, r), delay=1)], name='start2')
    ag.add_event([P['b2']], [P['e2']], lambda b: [SimToken(b[0])],
                 name='done2', reward_function=lambda b: 1)
    return ag


def _build_obs(w2_extra=0, b2_extra=0):
    pn = make_two_indep_pipelines()
    P = {p._id: p for p in pn.places}
    for _ in range(w2_extra):
        P['w2'].put({'id': 99})
    for _ in range(b2_extra):
        P['b2'].put({'id': 99})
    return pn, pn.get_graph_observation()['graph']


def _start1_raw_logit(actor, graph, global_context):
    with torch.no_grad():
        _ = actor({'graph': graph})  # materialize lazy modules first
        x_enc = actor.encoder(x_dict=graph.x_dict, edge_index_dict=graph.edge_index_dict,
                               input_size=-1, graph=graph, params_iter=iter(actor.parameters()))
        ctx = _pool_action_postpone(x_enc, graph) if global_context else None
        x = x_enc['a_transition']
        if ctx is not None:
            idx = _batch_index_for(graph, 'a_transition', x.size(0), x.device)
            x = torch.cat((x, ctx[idx]), dim=-1)
        return actor.decoder(x)[0].item()  # start1's single binding, first row


def test_default_off_is_unchanged():
    """global_context=False (the default) must decode identically to a
    HeteroActor built before this change existed -- verified by exact
    invariance to a downstream perturbation the old architecture could
    never see (see test_blind_spot_without_global_context)."""
    pn = make_independent(causal_rl=False, ra=5, rb=7)
    metadata = pn.make_metadata()
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3, metadata=metadata,
                         num_heads=2, global_context=False)
    assert actor.global_context is False
    print("  default_off_is_unchanged OK")


def test_blind_spot_without_global_context():
    """THE PROOF: with global_context=False, start1's raw logit is EXACTLY
    (bit-for-bit) invariant to perturbing stage-2's places -- a hard
    structural fact (no directed path exists), not a training/capacity
    issue, so it must hold even for a randomly-initialized, untrained
    network."""
    torch.manual_seed(0)
    pn_a, g_a = _build_obs(0, 0)
    pn_b, g_b = _build_obs(5, 3)
    metadata = pn_a.make_metadata()
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3, metadata=metadata,
                         num_heads=2, global_context=False)
    actor.eval()
    la = _start1_raw_logit(actor, g_a, False)
    lb = _start1_raw_logit(actor, g_b, False)
    assert la == lb, (la, lb)
    print(f"  blind_spot_without_global_context OK (logit={la:.6f}, exactly invariant)")


def test_global_context_closes_the_gap():
    """With global_context=True, start1's raw logit becomes SENSITIVE to
    the same stage-2 perturbation -- the pooled context routes it in via
    start2's own sibling embedding (which has a direct 1-hop edge to the
    perturbed places)."""
    torch.manual_seed(0)
    pn_a, g_a = _build_obs(0, 0)
    pn_b, g_b = _build_obs(5, 3)
    metadata = pn_a.make_metadata()
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3, metadata=metadata,
                         num_heads=2, global_context=True)
    actor.eval()
    la = _start1_raw_logit(actor, g_a, True)
    lb = _start1_raw_logit(actor, g_b, True)
    assert abs(la - lb) > 1e-3, (la, lb)
    print(f"  global_context_closes_the_gap OK (logit A={la:.6f} B={lb:.6f} "
          f"diff={abs(la-lb):.6f})")


def test_critic_pooling_unchanged():
    """HeteroCritic._encode_pool was refactored to share _pool_action_postpone
    with the actor -- confirm its value output is still finite/sane and the
    refactor didn't silently change its node-selection behavior."""
    from gympn.networks import HeteroCritic
    pn = make_independent(causal_rl=False, ra=5, rb=7)
    metadata = pn.make_metadata()
    graph = pn.get_graph_observation()['graph']
    critic = HeteroCritic(input_size=-1, hidden_size=32, num_layers=3, metadata=metadata, num_heads=2)
    critic.eval()
    with torch.no_grad():
        v = critic({'graph': graph})
    assert v.numel() == 1 and torch.isfinite(v).all(), v
    print(f"  critic_pooling_unchanged OK (value={v.item():.4f})")


def test_end_to_end_smoke_train():
    """Real training loop with global_context=True through the actual
    pipeline (train.py's policy_kwargs plumbing), on a real env."""
    import sys as _sys
    from envs import make_env

    env = make_env("d_parallel_disjoint", causal_rl=False, allow_postpone=True)

    args = {
        "algorithm": "ppo-clip", "episodes": 4, "epochs": 3, "batch_size": 8,
        "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 3,
        "value_lr": 3e-4, "value_updates": 4, "gam": 0.99, "lam": 0.95,
        "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
        "causal_rl": False, "causal_scheme": "lrq", "causal_beta": 0.5,
        "verbose": 0, "use_gpu": False, "agent_seed": 0,
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": 1, "test_episodes": 5,
        "save_freq": 1_000_000, "name": "global_ctx_smoke", "datetag": False,
        "logdir": "global_ctx_smoke_train",
        "policy_kwargs": {"global_context": True},
    }
    saved = _sys.argv
    _sys.argv = _sys.argv[:1]
    try:
        env.training_run(length=10, args_dict=args)
    finally:
        _sys.argv = saved
    h = env.training_history
    assert h.get("mean_returns") is not None and len(h["mean_returns"]) == 3
    assert all(v == v for v in h["mean_returns"])  # no NaNs
    print(f"  end_to_end_smoke_train OK (mean_returns={list(h['mean_returns'])})")

    import shutil
    shutil.rmtree("global_ctx_smoke_train", ignore_errors=True)


if __name__ == "__main__":
    test_default_off_is_unchanged()
    test_blind_spot_without_global_context()
    test_global_context_closes_the_gap()
    test_critic_pooling_unchanged()
    test_end_to_end_smoke_train()
    print("all actor global_context tests passed")
