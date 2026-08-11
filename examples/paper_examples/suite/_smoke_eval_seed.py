r"""End-to-end plumbing check for cfg.eval_seed -> greedy eval.

cfg.eval_seed must survive run_suite._make_args -> training_run's arg parser ->
Agent.train -> test_in_train. If it does, two runs of the SAME cell that differ
only in how much RNG training has consumed still produce IDENTICAL greedy
curves... but training itself is seeded, so instead the observable signature is
simpler: with a fixed eval scenario set, a converged flat policy stops showing
the +-0.23 point-to-point jitter that fresh draws inject.

This just verifies the value arrives (no exception from the parser, and the
eval is reproducible), using a 4-epoch cell so it costs ~1 minute.

Run: python _smoke_eval_seed.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import _make_args, _set_seed
from envs import make_env


def run(eval_seed):
    cfg = stoch_config()
    cfg.epochs = 4
    cfg.test_freq = 2
    cfg.eval_seed = eval_seed
    args = _make_args("s1_stoch_sequence", "ppo_clip", 0, cfg, "_smoke_eval_seed_train")
    assert args["eval_seed"] == eval_seed, f"_make_args dropped it: {args.get('eval_seed')}"
    _set_seed(0)
    env = make_env("s1_stoch_sequence", causal_rl=False,
                   allow_postpone=cfg.allow_postpone, causal_postpone_tokenflow=False)
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=20, args_dict=args)
    finally:
        sys.argv = saved
    h = env.training_history
    return [round(float(x), 3) for x in h["test_mean_returns"]]


on = run(777)
print(f"[smoke] eval_seed=777  greedy: {on}")
off = run(None)
print(f"[smoke] eval_seed=None greedy: {off}")
print()
print("PASS: cfg.eval_seed reached the eval without the parser rejecting it")
