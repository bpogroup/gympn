"""Multi-site skills-based routing.

A firm runs `n_sites` service sites. Each site has its OWN stochastic arrival
stream of tasks (types 0/1) and its OWN heterogeneous pool of dedicated
SPECIALISTS: `n_local` workers skilled for type-0 and `n_local` for type-1 (fast
= delay 1 on their matched type, slow = delay 3 otherwise). In addition there is
a shared pool of `n_flex` FLEXIBLE generalists that can be routed to ANY site
(moderate = delay 2, type-independent). Reward 1 per completed task (throughput).
Because each site's specialists cover BOTH types, matching each task to the
right-skill specialist matters strongly for the objective, which a random
policy fails to do.

Independence is read off the net: local specialists cycle within their own
site, so with n_flex = 0 every site is its own net component (no shared
place); the shared flex pool joins every site into one component.

The decision the agent faces: which queued task each free worker takes
(specialists prefer their matched type-0 work) and, for the shared floaters,
WHICH SITE to relieve.
"""
import random
from simpn.simulator import SimToken
from gympn.simulator import GymProblem

LOCAL_SKILL = 0   # dedicated specialists are fast on type-0
FLEX_CODE = 2     # flexible/generalist worker code (distinct from task types 0/1)


def _arrive(a):
    return [SimToken({'task_type': random.randint(0, 1)}, delay=1),
            SimToken({'task_type': random.randint(0, 1)})]


def _local_start(c, r):
    # specialist: fast on its matched type, slow otherwise
    base = 1 if c['task_type'] == r['code'] else 3
    return [SimToken((c, r), delay=base + random.randint(0, 1))]


def _flex_start(c, r):
    # generalist: moderate, type-independent
    return [SimToken((c, r), delay=2 + random.randint(0, 1))]


def _return_worker(b):
    return [SimToken(b[-1])]   # worker token returns to its pool


def make_multisite(n_sites=4, n_local=1, n_flex=4, allow_postpone=False):
    ag = GymProblem(allow_postpone=allow_postpone)
    flex = ag.add_var("flex", var_attributes=['code'])
    for _ in range(n_flex):
        flex.put({'code': FLEX_CODE})
    for i in range(n_sites):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['task_type'])
        wait = ag.add_var(f"wait_{i}", var_attributes=['task_type'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['task_type', 'code'])
        busyf = ag.add_var(f"busyf_{i}", var_attributes=['task_type', 'code'])
        local = ag.add_var(f"local_{i}", var_attributes=['code'])
        for skill in (0, 1):                 # heterogeneous: both skills on site
            for _ in range(n_local):
                local.put({'code': skill})
        arrival.put({'task_type': 0})
        wait.put({'task_type': 0})
        wait.put({'task_type': 1})
        ag.add_event([arrival], [arrival, wait], _arrive, name=f'arrive_{i}')
        # dedicated specialist (stays within the site)
        ag.add_action([wait, local], [busy], behavior=_local_start,
                      name=f"local_start_{i}")
        ag.add_event([busy], [local], _return_worker, name=f'local_done_{i}',
                     reward_function=lambda x: 1)
        # flexible generalist (returns to the SHARED pool -> cross-site coupling)
        ag.add_action([wait, flex], [busyf], behavior=_flex_start,
                      name=f"flex_start_{i}")
        ag.add_event([busyf], [flex], _return_worker, name=f'flex_done_{i}',
                     reward_function=lambda x: 1)
    return ag


def _binding_info(b):
    """Return (task_type, worker_code) for a real timed binding."""
    tt = code = None
    for (place, tok) in b[0]:
        v = getattr(tok, 'value', tok)
        if isinstance(v, dict):
            if 'code' in v:
                code = v['code']
            elif 'task_type' in v:
                tt = v['task_type']
    return tt, code


def multisite_heuristic(observable_net, tokens_comb, bindings=None):
    """Sensible skills-based routing rule (the normalization anchor):
    1) specialists do their matched (type-0) work first;
    2) floaters relieve type-1 backlog (which specialists are slow at);
    3) otherwise any floater, then any specialist."""
    if not bindings:
        return None
    real = [b for b in bindings
            if isinstance(b, tuple) and len(b) >= 2 and isinstance(b[0], list)
            and b[0] and isinstance(b[0][0], tuple)]
    if not real:
        return bindings[0]
    infos = [(b, *_binding_info(b)) for b in real]
    for b, tt, code in infos:                 # 1: matched specialist (skill==type)
        if code in (0, 1) and tt == code:
            return b
    for b, tt, code in infos:                 # 2: floater relieves backlog
        if code == FLEX_CODE:
            return b
    for b, tt, code in infos:                 # 3: any specialist (cross fallback)
        if code in (0, 1):
            return b
    return real[0]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    import numpy as np
    from gympn.solvers import RandomSolver, HeuristicSolver

    print("baselines (random vs heuristic):")
    for name, nl, nf in [("all-dedicated", 1, 0), ("mixed", 1, 4), ("all-pooled", 0, 8)]:
        rnd, heu = [], []
        for s in range(8):
            random.seed(s); np.random.seed(s)
            rnd.append(make_multisite(4, nl, nf, allow_postpone=False).testing_run(solver=RandomSolver(), length=20))
            random.seed(s); np.random.seed(s)
            heu.append(make_multisite(4, nl, nf, allow_postpone=False).testing_run(solver=HeuristicSolver(multisite_heuristic), length=20))
        print(f"  {name:<14} random={np.mean(rnd):6.1f}  heuristic={np.mean(heu):6.1f}  "
              f"lift={np.mean(heu)-np.mean(rnd):+.1f}")
