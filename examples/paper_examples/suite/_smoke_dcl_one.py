"""One DCL cell, small budget, printing the greedy curve — checks that the
actor-guided continuation + value bootstrap clears random (X15 fix)."""
import sys, time
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, ".")
from config import stoch_config
from run_dcl_s1 import train_dcl_cell, compute_baselines, ENV

arm = sys.argv[1] if len(sys.argv) > 1 else "dcl_plain"
cfg = stoch_config()
cfg.epochs = 6
cfg.episodes_per_epoch = 4
cfg.test_freq = 2
cfg.dcl_horizon = 6
cfg.dcl_rollouts = 6
cfg.dcl_temp = 0.5

bl = compute_baselines(ENV, cfg)
print(f"[smoke] random={bl['random_mean']:.2f} heuristic={bl['heuristic_mean']:.2f}",
      flush=True)
t0 = time.time()
m = train_dcl_cell(arm, 0, cfg, "suite_results_smoke_dcl/train", bl)
mins = (time.time() - t0) / 60.0
print(f"[smoke] {arm}: greedy_curve={m.get('greedy_curve')} "
      f"final={m.get('greedy_final')} ({mins:.1f} min)", flush=True)
r = bl["random_mean"]
nf = ((m["greedy_final"] - r) / (bl["heuristic_mean"] - r)
      if m.get("greedy_final") is not None else None)
print(f"[smoke] norm_final={nf} (>0 beats random; cfpk ref 0.849)", flush=True)