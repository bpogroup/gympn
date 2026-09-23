r"""Exact DP for ONE copy of the N-copies benchmark.

The manuscript normalizes by a myopic heuristic and states (Table of
environment parameters) that it is "a strong myopic rule, not a proven
optimum, so normalized values above 1 are attainable". For the N-copies
benchmark that question is decidable: a single copy is one server, two job
classes and Poisson-free deterministic arrivals, so the optimal expected
throughput over the horizon can be computed exactly by backward induction.

MODEL (read off ncopies_env.py):
  * one employee, code 0, initially free;
  * `waiting` warm-started with one type-0 and one type-1 task;
  * `_arrive` self-loops with delay 1, emitting one task per unit time whose
    type is uniform on {0,1};
  * `_start` gives service delay 1+U{0,1} for a matched task (type 0) and
    3+U{0,1} for a crossed task (type 1);
  * `_done` returns the employee and pays reward 1.

STATE. Tasks are exchangeable within a type, so (t, n0, n1) suffices, where t
is the time the server becomes free and n0/n1 are queue counts including all
arrivals up to and including t. The horizon is 20, so the state space is tiny
and the recursion is exact -- no sampling, no discretization error.

CONVENTIONS. Three details of the simulator's clock are not fixed by the env
source (whether an arrival lands at t=0, whether arrivals continue up to T-1 or
T, and whether a service completing exactly at T is paid). Rather than guess,
we enumerate all eight combinations, evaluate the HEURISTIC policy under each
by the same recursion, and keep the convention whose heuristic value matches
the simulator's measured single-copy heuristic. The optimum is then reported
under that calibrated convention, so the comparison is like-for-like.

Run: python _diag_ncopies_dp.py [measured_heuristic_value]
"""
import sys
from functools import lru_cache
from math import comb

T = 20


def solve(arr_at_zero, arr_last, rew_last, policy, allow_idle):
    """Backward induction. policy='opt' maximizes; 'heur' follows matched-first.

    Returns expected completions from the initial state.
    """

    @lru_cache(maxsize=None)
    def V(t, n0, n1):
        if t > rew_last:
            return 0.0
        # number of arrivals landing in (t, t+d]
        def n_arr(d):
            lo, hi = t + 1, t + d
            hi = min(hi, arr_last)
            return max(0, hi - lo + 1)

        def serve(kind):
            # duration support: matched 1,2 ; crossed 3,4
            ds = (1, 2) if kind == 0 else (3, 4)
            tot = 0.0
            for d in ds:
                done = t + d
                val = 1.0 if done <= rew_last else 0.0
                a = n_arr(d)
                sub = 0.0
                for k in range(a + 1):                 # k arrivals of type 0
                    p = comb(a, k) * 0.5 ** a
                    if kind == 0:
                        sub += p * V(done, n0 - 1 + k, n1 + (a - k))
                    else:
                        sub += p * V(done, n0 + k, n1 - 1 + (a - k))
                tot += 0.5 * (val + sub)
            return tot

        def wait():
            # server idle for one tick; one arrival if still arriving
            a = n_arr(1)
            sub = 0.0
            for k in range(a + 1):
                p = comb(a, k) * 0.5 ** a
                sub += p * V(t + 1, n0 + k, n1 + (a - k))
            return sub

        if n0 == 0 and n1 == 0:
            return wait()                              # forced, not a choice

        if policy == 'heur':                           # matched-first, non-idling
            return serve(0) if n0 > 0 else serve(1)

        opts = []
        if n0 > 0:
            opts.append(serve(0))
        if n1 > 0:
            opts.append(serve(1))
        if allow_idle:
            opts.append(wait())
        return max(opts)

    n0, n1 = 1, 1                                      # warm start
    if arr_at_zero:                                    # the t=0 arrival
        return 0.5 * V(0, n0 + 1, n1) + 0.5 * V(0, n0, n1 + 1)
    return V(0, n0, n1)


def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else None

    print("Calibrating clock conventions against the simulator's heuristic value")
    print("%-8s %-9s %-9s | %-10s %-10s %-10s" %
          ("arr@0", "arr_last", "rew_last", "heuristic", "opt(no idle)", "opt(idle)"))
    rows = []
    for arr_at_zero in (True, False):
        for arr_last in (T - 1, T):
            for rew_last in (T - 1, T):
                h = solve(arr_at_zero, arr_last, rew_last, 'heur', False)
                o = solve(arr_at_zero, arr_last, rew_last, 'opt', False)
                oi = solve(arr_at_zero, arr_last, rew_last, 'opt', True)
                rows.append((arr_at_zero, arr_last, rew_last, h, o, oi))
                mark = ""
                if target is not None and abs(h - target) < 0.02:
                    mark = "   <== matches simulator"
                print("%-8s %-9d %-9d | %-10.4f %-10.4f %-10.4f%s"
                      % (arr_at_zero, arr_last, rew_last, h, o, oi, mark))

    if target is None:
        print("\nPass the simulator's measured single-copy heuristic value as "
              "argv[1] to calibrate.")
        return

    best = min(rows, key=lambda r: abs(r[3] - target))
    _, _, _, h, o, oi = best
    print("\nCalibrated convention: arr@0=%s arr_last=%d rew_last=%d"
          % (best[0], best[1], best[2]))
    print("  simulator heuristic   %.4f" % target)
    print("  DP heuristic          %.4f   (gap %.4f)" % (h, h - target))
    print("  DP optimum, no idle   %.4f   (heuristic captures %.2f%%)"
          % (o, 100.0 * h / o))
    print("  DP optimum, w/ idle   %.4f   (heuristic captures %.2f%%)"
          % (oi, 100.0 * h / oi))
    print("\n  optimality gap of the anchor: %.4f tasks/copy (no idle), "
          "%.4f (idle allowed)" % (o - h, oi - h))


if __name__ == "__main__":
    main()
