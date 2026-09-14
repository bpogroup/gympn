"""Direction B probe (AEPN_NATIVE_LEARNING.md §7 step 1): measure the structural
conflict/coupling granularity of every suite env, with NO training.

For each env this prints, and tabulates:
  * n independent subnets (coupling components) and the largest-component
    fraction  -> does the value/advantage decomposition have any teeth?
  * n action-vs-action structural conflicts -> where the agent's genuine
    decisions live, read straight off the topology.

Reading guide (from the design note):
  largest_fraction ~ 1.00  ->  one coupled blob; Direction B decomposition
                               buys little on this env (the E1 falsifier).
  largest_fraction  < 1.00 ->  genuine parallel structure to factor over.

Run:
  python examples/paper_examples/suite/run_conflict_graph_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gympn.conflict_graph import analyze, format_report  # noqa: E402

from envs import ENV_BUILDERS  # noqa: E402  (a-h grid + stochastic tier)
from e1_chain_env import make_e1_chain, make_e1b_oneshot  # noqa: E402


def _targets():
    """(display_name, builder) pairs. allow_postpone kept OFF: the postpone
    machinery adds a guard, not places, so it does not change the structural
    conflict/coupling graph — but turning it off keeps the report minimal."""
    for name, builder in ENV_BUILDERS.items():
        yield name, (lambda b=builder: b(causal_rl=False, allow_postpone=False))
    yield "e1_chain", (lambda: make_e1_chain(causal_rl=False, allow_postpone=False))
    yield "e1b_oneshot_k2", (lambda: make_e1b_oneshot(causal_rl=False, allow_postpone=False, stages=2))
    yield "e1b_oneshot_k3", (lambda: make_e1b_oneshot(causal_rl=False, allow_postpone=False, stages=3))


def main(verbose: bool = False) -> None:
    rows = []
    for name, builder in _targets():
        try:
            pn = builder()
            a = analyze(pn)
        except Exception as e:  # keep the probe going across all envs
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        rows.append((name, a))
        if verbose:
            print(format_report(a, name=name))
            print()

    # Summary table
    hdr = f"{'env':<28} {'trans':>5} {'act':>4} {'subnets':>8} {'largest%':>9} {'A-A conflicts':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, a in rows:
        print(f"{name:<28} {a.n_transitions:>5} {a.n_actions:>4} "
              f"{a.n_components:>8} {a.largest_component_fraction*100:>8.0f}% "
              f"{len(a.action_conflict_edges):>14}")

    print()
    print("Legend: largest% = share of transitions in the biggest independent "
          "subnet.\n  ~100% => one coupled blob (decomposition buys little); "
          "<100% => parallel\n  structure to factor over. A-A conflicts = "
          "action-vs-action structural\n  conflicts = the agent's genuine "
          "decision points, read off the topology.")


if __name__ == "__main__":
    main(verbose="-v" in sys.argv or "--verbose" in sys.argv)