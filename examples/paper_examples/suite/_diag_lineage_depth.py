r"""What receptive field does the problem actually require?

HeteroActor decodes each action's logit after exactly `num_layers` rounds of
message passing (default 3). That number is a hyperparameter with no relation
to the net's causal structure. The distance that MATTERS is how far a decision
sits from the rewards it feeds -- and lineage measures that directly:
`lineage_decisions` already returns {decision_idx: hop depth} by BFS over the
realized token DAG.

If the realized depth is routinely greater than num_layers, the actor cannot
connect a decision to the reward it causes -- structurally, not as a training
failure. That is the same class of limit as the actor blind spot (which was
about DIRECTION), on the DEPTH axis.

Note the two hop counts are not the same unit and this is a screen, not a
proof: lineage depth counts token-parentage hops in the causal DAG, while
num_layers counts message-passing rounds on the observation graph
(place <-> transition arcs). They track each other -- both advance one
producer/consumer step at a time -- but a factor-of-two mismatch would not be
surprising. What is informative is the ORDER: depth ~2 vs depth ~15 mean very
different things for a 3-layer network.

Run: python _diag_lineage_depth.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as tm

NUM_LAYERS = 3          # HeteroActor default
_D = defaultdict(list)
_real = tm.CausalTraces._redistribute_ls_hca


def probe(tag):
    def spy(self, *a, **kw):
        out = _real(self, *a, **kw)
        try:
            # pending entries carry (a_type, rtype, z, contrib, delay, depth, share)
            for items in (self._ls_hca_pending or []):
                for e in items:
                    if len(e) >= 7 and e[2] and e[5] is not None:
                        _D[tag].append(float(e[5]))
        except Exception as ex:
            print(f"[warn] {tag}: {ex}")
        return out
    tm.CausalTraces._redistribute_ls_hca = spy


def run(tag, make_fn, length, eps=4):
    probe(tag)
    args = {"algorithm": "ppo-clip", "episodes": eps, "epochs": 2,
            "batch_size": 64, "max_episode_length": None, "policy_lr": 3e-4,
            "policy_updates": 1, "value_lr": 3e-4, "value_updates": 1,
            "gam": 0.99, "lam": 0.95, "eps": 0.2, "vf_coeff": 0.5,
            "ent_bonus": 0.01, "policy_kld_limit": 0.15, "causal_rl": True,
            "causal_scheme": "ls_hca", "causal_beta": 0.5, "verbose": 0,
            "use_gpu": False, "agent_seed": 0, "use_wandb": False,
            "open_tensorboard": False, "test_in_train": False,
            "save_freq": 10**9, "name": "depth", "datetag": False,
            "logdir": f"depth_{tag}"}
    sv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        make_fn().training_run(length=length, args_dict=args)
    except Exception as e:
        print(f"[skip] {tag}: {type(e).__name__}: {e}")
    finally:
        sys.argv = sv
        tm.CausalTraces._redistribute_ls_hca = _real
        import shutil
        shutil.rmtree(f"depth_{tag}", ignore_errors=True)


from envs import make_env  # noqa: E402
from multisite_env import make_multisite  # noqa: E402
from ncopies_env import make_n_copies  # noqa: E402

CASES = [
    ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True), 20),
    ("s2", lambda: make_env("s2_stoch_scaled", causal_rl=True, allow_postpone=True), 30),
    ("s3", lambda: make_env("s3_stoch_mixed", causal_rl=True, allow_postpone=True), 25),
    ("multisite", lambda: make_multisite(causal_rl=True, allow_postpone=False), 30),
    ("ncopies8", lambda: make_n_copies(8, causal_rl=True, allow_postpone=False), 20),
    ("f_loop", lambda: make_env("f_loop_disjoint", causal_rl=True, allow_postpone=True), 10),
]
for tag, fn, L in CASES:
    run(tag, fn, L)

print("\n" + "=" * 84)
print(f"realized decision->reward LINEAGE DEPTH   vs   num_layers={NUM_LAYERS}")
print("=" * 84)
print(f"{'env':<12}{'n':>7}{'median':>8}{'mean':>8}{'p90':>7}{'max':>6}"
      f"{'frac > 3':>10}{'frac > 6':>10}")
print("-" * 84)
for tag, _, _ in CASES:
    d = np.array(_D[tag], float)
    if not len(d):
        print(f"{tag:<12}  (no lineage entries)")
        continue
    print(f"{tag:<12}{len(d):>7}{np.median(d):>8.1f}{d.mean():>8.2f}"
          f"{np.quantile(d, .9):>7.1f}{d.max():>6.0f}"
          f"{float((d > NUM_LAYERS).mean()):>10.1%}{float((d > 2*NUM_LAYERS).mean()):>10.1%}")
print()
print("frac > 3 = share of causal links the actor CANNOT span at num_layers=3.")
print("High values mean no credit scheme can help: the network is structurally")
print("unable to connect those decisions to the rewards they cause.")
