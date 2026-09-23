r"""Exact DP for ONE SITE of the multi-site skills-based routing environment.

Companion to _diag_ncopies_dp.py. The reported multi-site configuration is
make_multisite(n_sites=4, n_local=1, n_flex=0): with no flexible workers the
sites share nothing, so the four-site optimum is four times the single-site
optimum and one site can be solved exactly by backward induction.

MODEL (read off multisite_env.py):
  * two dedicated specialists per site, one of skill 0 and one of skill 1;
  * `_local_start` gives service delay 1+U{0,1} when the specialist's skill
    matches the task type and 3+U{0,1} when it does not;
  * `_arrive` self-loops with delay 1, emitting one task per unit time whose
    type is uniform on {0,1};
  * `wait` warm-started with one type-0 and one type-1 task;
  * `_return_worker` pays reward 1 and returns the specialist to its own site.

STATE. (t, n0, n1, b0, b1) where n_tau are queue counts and b_s in {0..4} is the
remaining service time of specialist s (0 = free). Services are at most 4 long,
so the state space is about 3.7e5 and the recursion is exact.

NON-IDLING. Both the baselines (`compute_baselines`) and the multi-site training
runs (run_multisite_validate.py:96) build the net with allow_postpone=False, so
no arm in the paper may leave a specialist idle while work waits. The non-idling
optimum is therefore the like-for-like ceiling, and we report it as the primary
number; the unrestricted optimum is reported alongside to price the constraint.

CONVENTIONS are calibrated exactly as in _diag_ncopies_dp.py: enumerate the
clock conventions, evaluate the HEURISTIC under each, keep the one matching the
simulator's measured single-site heuristic.

Run: python _diag_multisite_dp.py [measured_single_site_heuristic]
"""
import sys
from functools import lru_cache

T = 20
MAXQ = 26


def solve(arr_at_zero, arr_last, rew_last, policy, allow_idle):
    """policy in {'opt','heur'}. Returns expected completions for one site."""

    def dur(skill, task):
        base = 1 if skill == task else 3
        return (base, base + 1)          # each w.p. 1/2

    @lru_cache(maxsize=None)
    def V(t, n0, n1, b0, b1):
        if t > rew_last:
            return 0.0

        # ---- enumerate assignments for the free specialists ----------------
        free = [s for s in (0, 1) if (b0, b1)[s] == 0]
        assigns = []                      # list of (list[(server, task)],)
        if not free:
            assigns = [[]]
        elif len(free) == 1:
            s = free[0]
            opts = [[(s, tau)] for tau in (0, 1) if (n0, n1)[tau] > 0]
            if allow_idle or not opts:
                opts.append([])
            assigns = opts
        else:
            opts = []
            for t0 in (0, 1):
                if (n0, n1)[t0] == 0:
                    continue
                for t1 in (0, 1):
                    q = [n0, n1]
                    q[t0] -= 1
                    if q[t1] <= 0:
                        continue
                    opts.append([(0, t0), (1, t1)])
            for s in free:                # only one of the two works
                for tau in (0, 1):
                    if (n0, n1)[tau] > 0:
                        opts.append([(s, tau)])
            if allow_idle or not opts:
                opts.append([])
            assigns = opts

        # ---- the heuristic picks one assignment deterministically ----------
        if policy == 'heur':
            # multisite_heuristic: matched specialist first, then any
            # specialist; applied repeatedly until no binding remains.
            q = [n0, n1]
            chosen = []
            avail = list(free)
            while avail and (q[0] > 0 or q[1] > 0):
                pick = None
                for s in avail:           # 1: skill == task type
                    if q[s] > 0:
                        pick = (s, s)
                        break
                if pick is None:          # 3: any specialist, cross work
                    for s in avail:
                        for tau in (0, 1):
                            if q[tau] > 0:
                                pick = (s, tau)
                                break
                        if pick:
                            break
                if pick is None:
                    break
                chosen.append(pick)
                q[pick[1]] -= 1
                avail.remove(pick[0])
            assigns = [chosen]

        best = None
        for A in assigns:
            q0, q1 = n0, n1
            starts = []
            for (s, tau) in A:
                if tau == 0:
                    q0 -= 1
                else:
                    q1 -= 1
                starts.append((s, dur(s, tau)))

            # expectation over the started services' durations
            tot = 0.0
            n_out = 1 << len(starts)
            for mask in range(n_out):
                nb = [b0, b1]
                for i, (s, ds) in enumerate(starts):
                    nb[s] = ds[(mask >> i) & 1]
                p = 0.5 ** len(starts)

                # advance one tick: busy times decrement, a service reaching 0
                # completed at t+1
                rew = 0.0
                for s in (0, 1):
                    if nb[s] > 0:
                        nb[s] -= 1
                        if nb[s] == 0 and t + 1 <= rew_last:
                            rew += 1.0
                # arrival at t+1
                if t + 1 <= arr_last:
                    sub = 0.5 * V(t + 1, min(q0 + 1, MAXQ), q1, nb[0], nb[1]) \
                        + 0.5 * V(t + 1, q0, min(q1 + 1, MAXQ), nb[0], nb[1])
                else:
                    sub = V(t + 1, q0, q1, nb[0], nb[1])
                tot += p * (rew + sub)

            if best is None or tot > best:
                best = tot
        return best if best is not None else 0.0

    n0, n1 = 1, 1
    if arr_at_zero:
        return 0.5 * V(0, n0 + 1, n1, 0, 0) + 0.5 * V(0, n0, n1 + 1, 0, 0)
    return V(0, n0, n1, 0, 0)


def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else None
    print("Calibrating clock conventions against the simulator's heuristic value")
    print("%-8s %-9s %-9s | %-11s %-13s %-11s" %
          ("arr@0", "arr_last", "rew_last", "heuristic", "opt(non-idle)", "opt(idle)"))
    rows = []
    for arr_at_zero in (True, False):
        for arr_last in (T - 1, T):
            for rew_last in (T - 1, T):
                h = solve(arr_at_zero, arr_last, rew_last, 'heur', False)
                o = solve(arr_at_zero, arr_last, rew_last, 'opt', False)
                oi = solve(arr_at_zero, arr_last, rew_last, 'opt', True)
                rows.append((arr_at_zero, arr_last, rew_last, h, o, oi))
                mark = "   <== matches simulator" if (
                    target is not None and abs(h - target) < 0.05) else ""
                print("%-8s %-9d %-9d | %-11.4f %-13.4f %-11.4f%s"
                      % (arr_at_zero, arr_last, rew_last, h, o, oi, mark))

    if target is None:
        print("\nPass the simulator's measured single-site heuristic as argv[1].")
        return

    best = min(rows, key=lambda r: abs(r[3] - target))
    _, _, _, h, o, oi = best
    print("\nCalibrated convention: arr@0=%s arr_last=%d rew_last=%d"
          % (best[0], best[1], best[2]))
    print("  simulator heuristic      %.4f  (x4 sites = %.3f)" % (target, 4 * target))
    print("  DP heuristic             %.4f  (gap to simulator %+.4f)" % (h, h - target))
    print("  DP optimum, non-idling   %.4f  (x4 = %.3f)  heuristic captures %.2f%%"
          % (o, 4 * o, 100.0 * h / o))
    print("  DP optimum, idling ok    %.4f  (x4 = %.3f)  heuristic captures %.2f%%"
          % (oi, 4 * oi, 100.0 * h / oi))
    print("\n  optimality gap of the anchor: %.4f tasks/site (non-idling), "
          "%.4f (idling allowed)" % (o - h, oi - h))


if __name__ == "__main__":
    main()
