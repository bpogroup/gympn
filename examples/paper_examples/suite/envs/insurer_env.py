"""An insurer running three processes against one bottom line (BPM version).

WHY. A real organisation runs several processes side by side, each with its
own team, and is measured on an organisation-level KPI. An RL controller
trained on that KPI credits every decision in one process with the reward
noise of all the others, and no critic can remove that noise (it is the other
processes' future randomness; NFGAE_THEORY.md, Theorem 3). NF-GAE reads the
independence off the process model: processes that share no place are separate
components, and a decision is credited only with its own component's rewards.

THE PROCESSES (per region; one case per time unit per process, reward 1 per
good completion, so the KPI is total good completions over the horizon):

  claims       next-activity selection: `approve` (fast, pays only if the
               hidden outcome is good) or `investigate` (slow, always pays).
               Same structure and rates as bpm_envs.make_next_activity.
  underwriting resource assignment: standard (type 0) or complex (type 1)
               applications, one specialist underwriter per type; matched
               service is fast, mismatched slow. Same structure and rates as
               one site of multisite_env.make_multisite(n_flex=0).
  complaints   rework / quality gate: `resolve` (fast; a wrongly closed
               complaint comes back after a delay and pays nothing the first
               time) or `check` (slow, always closes). Same structure and rates
               as bpm_envs.make_rework.

Each process has its own dedicated team, so with `shared_clerks=False` a
region holds 3 independent components and the insurer 3 * regions (K).
`shared_clerks=True` puts the claims and complaints clerks of a region in ONE
pool (same headcount), which joins those two processes into one component:
K = 2 * regions. `processes` builds any subset, so "a process alone" uses
exactly the same sub-net as in the full insurer.

No postpone (work-conserving): NF-GAE's soundness conditions need either no
postpone or component-scoped postpone, and multi-site ran without it.

SLOW-MISMATCH VARIANT (`uw_mismatch`, builder "insurer_slow"): a mismatched
underwriter takes `uw_mismatch` time units (base, + U{0,1}) instead of 3. This
is the slow-server regime (Lin & Kumar 1984): leaving an underwriter idle rather
than give them the wrong application type pays (+16% in underwriting at 7, by
the quick heuristic check), so postponement matters. Its anchor is the WAITING
heuristic (`wait_anchors`): never a mismatched underwriting assignment (the
process waits instead, component postpone); claims and complaints stay
work-conserving.
"""
import random

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

from bpm_envs import (FAST_BASE, FIX_DELAY, P_BAD, SAFE_BASE, _draw_case,  # noqa: F401
                      _rate, _service, _visible)

PROCESSES = ("claims", "underwriting", "complaints")


# ------------------------------------------------------------- claims
def _cl_arrive(a):
    return [SimToken(_draw_case(), delay=1), SimToken(_draw_case())]


def _cl_approve(c, r):
    ok = 0 if c['bad'] else 1
    return [SimToken((_visible(c), r, {'activity': 0, 'ok': ok}), delay=_service(FAST_BASE))]


def _cl_investigate(c, r):
    return [SimToken((_visible(c), r, {'activity': 1, 'ok': 1}), delay=_service(SAFE_BASE[r['skill']]))]


def _cl_done(b):
    return [SimToken(b[1])]


def _add_claims(ag, tag, hidden, clerks):
    arrival = ag.add_var(f"cl_arrival_{tag}", var_attributes=['risk', 'bad'])
    waiting = ag.add_var(f"cl_waiting_{tag}", var_attributes=['risk', 'bad'])
    busy = ag.add_var(f"cl_busy_{tag}", var_attributes=['risk', 'skill', 'activity', 'ok'])
    arrival.put({'risk': 0, 'bad': 0})
    for risk in (0, 1, 2):              # deterministic warm start, as in bpm_envs
        waiting.put({'risk': risk, 'bad': 1 if risk == 2 else 0})
    hidden[f"cl_arrival_{tag}"] = ['bad']
    hidden[f"cl_waiting_{tag}"] = ['bad']
    ag.add_event([arrival], [arrival, waiting], _cl_arrive, name=f'cl_arrive_{tag}')
    ag.add_action([waiting, clerks], [busy], behavior=_cl_approve, name=f"cl_approve_{tag}")
    ag.add_action([waiting, clerks], [busy], behavior=_cl_investigate, name=f"cl_investigate_{tag}")
    ag.add_event([busy], [clerks], _cl_done, name=f'cl_done_{tag}',
                 reward_function=lambda b: b[2]['ok'])


# ------------------------------------------------------- underwriting
def _uw_arrive(a):
    return [SimToken({'task_type': random.randint(0, 1)}, delay=1),
            SimToken({'task_type': random.randint(0, 1)})]


def _uw_starter(mismatch):
    def start(c, r):
        base = 1 if c['task_type'] == r['code'] else mismatch      # matched fast, mismatched slow
        return [SimToken((c, r), delay=base + random.randint(0, 1))]
    return start


def _uw_done(b):
    return [SimToken(b[-1])]


def _add_underwriting(ag, tag, mismatch=3, codes=(0, 1)):
    arrival = ag.add_var(f"uw_arrival_{tag}", var_attributes=['task_type'])
    wait = ag.add_var(f"uw_wait_{tag}", var_attributes=['task_type'])
    busy = ag.add_var(f"uw_busy_{tag}", var_attributes=['task_type', 'code'])
    team = ag.add_var(f"uw_team_{tag}", var_attributes=['code'])
    for code in codes:
        team.put({'code': code})
    arrival.put({'task_type': 0})
    wait.put({'task_type': 0})
    wait.put({'task_type': 1})
    ag.add_event([arrival], [arrival, wait], _uw_arrive, name=f'uw_arrive_{tag}')
    ag.add_action([wait, team], [busy], behavior=_uw_starter(mismatch), name=f"uw_start_{tag}")
    ag.add_event([busy], [team], _uw_done, name=f'uw_done_{tag}', reward_function=lambda x: 1)
    return team


# --------------------------------------------------------- complaints
def _co_arrive(a):
    return [SimToken(_draw_case(reworked=0), delay=1), SimToken(_draw_case(reworked=0))]


def _co_resolve(c, r):
    ok = 0 if (c['bad'] and not c['reworked']) else 1
    return [SimToken((_visible(c), r, {'activity': 0, 'ok': ok}), delay=_service(FAST_BASE))]


def _co_check(c, r):
    return [SimToken((_visible(c), r, {'activity': 1, 'ok': 1}), delay=_service(SAFE_BASE[r['skill']]))]


def _co_done(b):
    case, res, tag = b[0], b[1], b[2]
    if tag['ok']:
        return [SimToken(res), None]
    fixed = dict(case); fixed['reworked'] = 1; fixed['bad'] = 0
    return [SimToken(res), SimToken(fixed, delay=FIX_DELAY)]    # clerk back, complaint re-opened


def _add_complaints(ag, tag, hidden, clerks):
    arrival = ag.add_var(f"co_arrival_{tag}", var_attributes=['risk', 'bad', 'reworked'])
    waiting = ag.add_var(f"co_waiting_{tag}", var_attributes=['risk', 'bad', 'reworked'])
    busy = ag.add_var(f"co_busy_{tag}", var_attributes=['risk', 'reworked', 'skill', 'activity', 'ok'])
    arrival.put({'risk': 0, 'bad': 0, 'reworked': 0})
    for risk in (0, 1, 2):
        waiting.put({'risk': risk, 'bad': 1 if risk == 2 else 0, 'reworked': 0})
    hidden[f"co_arrival_{tag}"] = ['bad']
    hidden[f"co_waiting_{tag}"] = ['bad']
    ag.add_event([arrival], [arrival, waiting], _co_arrive, name=f'co_arrive_{tag}')
    ag.add_action([waiting, clerks], [busy], behavior=_co_resolve, name=f"co_resolve_{tag}")
    ag.add_action([waiting, clerks], [busy], behavior=_co_check, name=f"co_check_{tag}")
    ag.add_event([busy], [clerks, waiting], _co_done, name=f'co_done_{tag}',
                 reward_function=lambda b: b[2]['ok'])


# ------------------------------------------------------------- insurer
def make_insurer(regions=1, processes=PROCESSES, shared_clerks=False,
                 allow_postpone=False, uw_mismatch=3):
    """The insurer: `processes` (any subset of PROCESSES) in each of `regions`
    regions. With shared_clerks the claims and complaints clerks of a region
    form one pool (2 normal + 2 expert, the same headcount as two teams)."""
    processes = tuple(processes)
    unknown = set(processes) - set(PROCESSES)
    if unknown:
        raise ValueError(f"unknown processes {sorted(unknown)}; choose from {PROCESSES}")
    if shared_clerks and not {"claims", "complaints"} <= set(processes):
        raise ValueError("shared_clerks needs both claims and complaints")
    ag = GymProblem(allow_postpone=allow_postpone)
    hidden = {}
    for i in range(regions):
        tag = str(i)
        shared = None
        if shared_clerks:
            shared = ag.add_var(f"clerks_{tag}", var_attributes=['skill'])
            for skill in (0, 0, 1, 1):
                shared.put({'skill': skill})
        if "claims" in processes:
            clerks = shared
            if clerks is None:
                clerks = ag.add_var(f"cl_clerks_{tag}", var_attributes=['skill'])
                clerks.put({'skill': 0}); clerks.put({'skill': 1})
            _add_claims(ag, tag, hidden, clerks)
        if "underwriting" in processes:
            _add_underwriting(ag, tag, uw_mismatch)
        if "complaints" in processes:
            clerks = shared
            if clerks is None:
                clerks = ag.add_var(f"co_clerks_{tag}", var_attributes=['skill'])
                clerks.put({'skill': 0}); clerks.put({'skill': 1})
            _add_complaints(ag, tag, hidden, clerks)
    if hidden:
        ag.set_unobservable(token_attrs=hidden)
    return ag


def insurer_resample_hidden(pn):
    """Redraw what a policy cannot know (for a fair rollout planner,
    gympn.policy_iteration.RolloutPlanner(resample=...)): the hidden outcome
    `bad` of every waiting claim and complaint, from its visible risk level
    (P_BAD). A reopened complaint is known to be fixed (bad = 0) and is kept.
    The next arrival's case is drawn afresh when the arrival fires, so it needs
    no redraw. Uses the global `random` stream, which the planner seeds per
    rollout."""
    for place in pn.places:
        if not place._id.startswith(("cl_waiting_", "co_waiting_")):
            continue
        for tok in [t for t in place.marking if not t.value.get('reworked', 0)]:
            case = dict(tok.value)
            case['bad'] = 1 if random.random() < P_BAD[case['risk']] else 0
            place.remove_token(tok)
            place.add_token(SimToken(case, time=tok.time))


# ----------------------------------------------------------- heuristic
def _binding_rate(b, mismatch=3):
    """Myopic expected reward per expected resource time of one binding, by
    process: the same rate rules as the bpm_envs and multi-site anchors."""
    tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
    vals = [getattr(tok, 'value', tok) for (_place, tok) in b[0]]
    if tr.startswith('uw_'):
        tt = next((v['task_type'] for v in vals if isinstance(v, dict) and 'task_type' in v), None)
        code = next((v['code'] for v in vals if isinstance(v, dict) and 'code' in v), None)
        if tt is None or code is None:
            return None
        return 1.0 / ((1 if tt == code else mismatch) + 0.5)
    risk = skill = None
    reworked = 0
    for v in vals:
        if isinstance(v, dict):
            if 'risk' in v:
                risk = v['risk']; reworked = v.get('reworked', 0)
            if 'skill' in v:
                skill = v['skill']
    if risk is None or skill is None:
        return None
    fast = tr.startswith(('cl_approve', 'co_resolve'))
    eff_risk = 0 if (fast and reworked) else risk       # a reopened complaint is fixed
    return _rate(0 if fast else 1, eff_risk, skill)


def insurer_heuristic(observable_net, tokens_comb, bindings=None):
    """Pick the enabled binding with the best myopic rate. The processes use
    disjoint teams (or, with shared clerks, the same rate table), so this is
    the per-process anchor rule applied everywhere at once."""
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ['postpone']]
    if not real:
        return bindings[0]
    best, best_rate = None, -1.0
    for b in real:
        r = _binding_rate(b)
        if r is not None and r > best_rate:
            best, best_rate = b, r
    return best if best is not None else real[0]


def _is_uw_mismatch(b):
    tr = str(getattr(b[2], '_id', ''))
    if not tr.startswith('uw_'):
        return False
    vals = [getattr(tok, 'value', tok) for (_place, tok) in b[0]]
    tt = next(v['task_type'] for v in vals if 'task_type' in v)
    code = next(v['code'] for v in vals if 'code' in v)
    return tt != code


def _is_postpone(b):
    return isinstance(b[0], list) and b[0] == ['postpone']


def insurer_wait_choice(pn_actions, part, mismatch):
    """Index of the waiting heuristic's choice: the best-rate binding that is not
    a mismatched underwriting assignment; if only mismatches are left, that
    underwriting process waits (its component postpone pseudo-binding)."""
    real = [(i, b) for i, b in enumerate(pn_actions) if not _is_postpone(b)]
    ok = [(_binding_rate(b, mismatch) or 0.0, i) for i, b in real if not _is_uw_mismatch(b)]
    if ok:
        return max(ok)[1]
    c = part[real[0][1][2]._id]
    for i, b in enumerate(pn_actions):
        if _is_postpone(b) and getattr(b[2], 'comp', None) == c:
            return i
    return real[0][0]


def wait_anchors(build, n, length, episodes=40):
    """Anchors for the slow-mismatch variant: random = uniform over production
    bindings (no postpone), heuristic = the waiting heuristic (component
    postpone). Runs through AEPN_Env, which handles component postpone
    (GymProblem.testing_run does not pass the component id)."""
    import numpy as np
    from gympn.environment import AEPN_Env
    mismatch = build.uw_mismatch

    def episode(seed, waiting):
        random.seed(seed); np.random.seed(seed)
        pn = build(n, allow_postpone=waiting)
        pn.length = length
        if waiting:
            pn.postpone_scope = 'component'
        env = AEPN_Env(pn)
        env.reset()
        part, rng, total = env.pn.net_partition(), random.Random(seed + 1), 0.0
        while env.pn.pn_actions:
            B = env.pn.pn_actions
            a = insurer_wait_choice(B, part, mismatch) if waiting else rng.randrange(len(B))
            _, r, done, _, _ = env.step(a)
            total += r
            if done:
                break
        return total
    rnd = [episode(1000 + s, False) for s in range(episodes)]
    heu = [episode(1000 + s, True) for s in range(episodes)]
    return {"random_mean": float(np.mean(rnd)), "heuristic_mean": float(np.mean(heu)),
            "random_std": float(np.std(rnd)), "heuristic_std": float(np.std(heu)),
            "episodes": episodes, "anchor": "waiting heuristic (component postpone)"}


def _builder(processes, shared_clerks=False, uw_mismatch=3):
    def build(n=1, allow_postpone=False):
        return make_insurer(n, processes, shared_clerks, allow_postpone=allow_postpone,
                            uw_mismatch=uw_mismatch)
    build.uw_mismatch = uw_mismatch
    build.wait_anchor = uw_mismatch != 3
    return build


# ------------------------------------------------- start-of-shift allocation
# ALLOCATION VARIANT (builder "insurer_alloc"). Same headcount as the insurer
# (6 staff per region), but half of it is a flexible pool assigned at the start
# of the shift: each region starts with one generalist claims clerk, one
# generalist complaints clerk and one type-0 underwriter, plus a pool of two
# expert clerks and one generalist. An allocation action moves one pool member
# to claims, to complaints, or to underwriting as a type-0 or type-1
# underwriter. The pool has no producer, so once it is empty the allocation
# transitions are dead for good.
#
# WHY. The allocation couples the three processes of a region (the pool place
# touches all of them), so the static certificate sees ONE component per region
# and NF-GAE reduces to joint PPO there. After the pool is empty the processes
# are exactly independent again: a marking-based certificate (dead transitions
# pruned at the current marking) splits each region into 3 components. The
# allocation needs the organisation's credit; every later decision needs only
# its own process's credit. Separate per-process learners cannot make the
# allocation at all.
#
# All allocation happens at time 0: allocation bindings stay enabled while the
# pool is non-empty and the clock only advances when no binding is enabled.
ALLOC_POOL = (1, 1, 0)                    # skills of the flexible staff
ALLOC_DESTS = ("cl", "co", "uw0", "uw1")


def _alloc_to_clerks(s):
    return [SimToken({'skill': s['skill']})]


def _alloc_to_uw(code):
    def move(s):
        return [SimToken({'code': code})]
    return move


def make_insurer_alloc(regions=1, allow_postpone=False, pool=ALLOC_POOL):
    """The insurer with a start-of-shift allocation of a flexible pool (see the
    comment above). Without the pool (pool=()), the regions are understaffed
    insurers with three independent processes each."""
    ag = GymProblem(allow_postpone=allow_postpone)
    hidden = {}
    for i in range(regions):
        tag = str(i)
        cl_clerks = ag.add_var(f"cl_clerks_{tag}", var_attributes=['skill'])
        cl_clerks.put({'skill': 0})
        co_clerks = ag.add_var(f"co_clerks_{tag}", var_attributes=['skill'])
        co_clerks.put({'skill': 0})
        _add_claims(ag, tag, hidden, cl_clerks)
        team = _add_underwriting(ag, tag, codes=(0,))
        _add_complaints(ag, tag, hidden, co_clerks)
        if pool:
            staff = ag.add_var(f"staff_{tag}", var_attributes=['skill'])
            for skill in pool:
                staff.put({'skill': skill})
            ag.add_action([staff], [cl_clerks], behavior=_alloc_to_clerks, name=f"al_cl_{tag}")
            ag.add_action([staff], [co_clerks], behavior=_alloc_to_clerks, name=f"al_co_{tag}")
            ag.add_action([staff], [team], behavior=_alloc_to_uw(0), name=f"al_uw0_{tag}")
            ag.add_action([staff], [team], behavior=_alloc_to_uw(1), name=f"al_uw1_{tag}")
    ag.set_unobservable(token_attrs=hidden)
    return ag


def _alloc_binding(b):
    """(destination, skill) of an allocation binding, or None."""
    tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
    if not tr.startswith('al_'):
        return None
    dest = tr.split('_')[1]
    vals = [getattr(tok, 'value', tok) for (_place, tok) in b[0]]
    skill = next(v['skill'] for v in vals if isinstance(v, dict) and 'skill' in v)
    return dest, skill


def make_alloc_heuristic(plan):
    """Heuristic for the allocation variant: allocate by `plan`, an ordered list
    of (destination, skill), then run the insurer's myopic rule. Step k of a
    region's allocation is read off the bindings (k = pool size - pool members
    still enabled), so the heuristic is stateless and works with
    HeuristicSolver."""
    plan = list(plan)

    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None
        alloc, ops = [], []
        for b in bindings:
            ab = (_alloc_binding(b) if isinstance(b, tuple) and b and isinstance(b[0], list)
                  else None)
            (alloc if ab is not None else ops).append((b, ab))
        if alloc:
            pool_left = {}
            for b, _ in alloc:
                region = str(getattr(b[2], '_id', '')).rsplit('_', 1)[-1]
                pool_left.setdefault(region, set()).add(id(b[0][0][1]))
            for b, (dest, skill) in alloc:
                region = str(getattr(b[2], '_id', '')).rsplit('_', 1)[-1]
                k = len(plan) - len(pool_left[region])
                if 0 <= k < len(plan) and plan[k] == (dest, skill):
                    return b
            return alloc[0][0]
        return insurer_heuristic(observable_net, tokens_comb, [b for b, _ in ops])
    heuristic.plan = list(plan)
    return heuristic


# Best plan by enumeration (all 40 allocations of the pool, heuristic operations,
# 40 episodes each under common random numbers): one expert to claims, one to
# complaints, the generalist to underwriting as a type-1 underwriter (53.6
# completions at 1 region; worst plan 40.6, mean over plans 47.2).
ALLOC_PLAN = [("cl", 1), ("co", 1), ("uw1", 0)]
alloc_heuristic = make_alloc_heuristic(ALLOC_PLAN)


# N = regions. "insurer" is the full organisation; the *_alone builders are the
# same sub-nets on their own (the per-process controls).
INSURER_BUILDERS = {
    "insurer": _builder(PROCESSES),
    "insurer_shared": _builder(PROCESSES, shared_clerks=True),
    "insurer_claims": _builder(("claims",)),
    "insurer_underwriting": _builder(("underwriting",)),
    "insurer_complaints": _builder(("complaints",)),
    # slow-mismatch variant: postponement pays in underwriting
    "insurer_slow": _builder(PROCESSES, uw_mismatch=7),
    "insurer_underwriting_slow": _builder(("underwriting",), uw_mismatch=7),
}


def _alloc_builder(n=1, allow_postpone=False):
    return make_insurer_alloc(n, allow_postpone=allow_postpone)


_alloc_builder.uw_mismatch = 3
_alloc_builder.wait_anchor = False
INSURER_BUILDERS["insurer_alloc"] = _alloc_builder
INSURER_HEURISTICS = {name: insurer_heuristic for name in INSURER_BUILDERS}
INSURER_HEURISTICS["insurer_alloc"] = alloc_heuristic
