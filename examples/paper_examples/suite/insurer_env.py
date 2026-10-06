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


def _uw_start(c, r):
    base = 1 if c['task_type'] == r['code'] else 3      # matched fast, mismatched slow
    return [SimToken((c, r), delay=base + random.randint(0, 1))]


def _uw_done(b):
    return [SimToken(b[-1])]


def _add_underwriting(ag, tag):
    arrival = ag.add_var(f"uw_arrival_{tag}", var_attributes=['task_type'])
    wait = ag.add_var(f"uw_wait_{tag}", var_attributes=['task_type'])
    busy = ag.add_var(f"uw_busy_{tag}", var_attributes=['task_type', 'code'])
    team = ag.add_var(f"uw_team_{tag}", var_attributes=['code'])
    team.put({'code': 0})
    team.put({'code': 1})
    arrival.put({'task_type': 0})
    wait.put({'task_type': 0})
    wait.put({'task_type': 1})
    ag.add_event([arrival], [arrival, wait], _uw_arrive, name=f'uw_arrive_{tag}')
    ag.add_action([wait, team], [busy], behavior=_uw_start, name=f"uw_start_{tag}")
    ag.add_event([busy], [team], _uw_done, name=f'uw_done_{tag}', reward_function=lambda x: 1)


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
def make_insurer(regions=1, processes=PROCESSES, shared_clerks=False, causal_rl=False,
                 allow_postpone=False, causal_postpone_tokenflow=False):
    """The insurer: `processes` (any subset of PROCESSES) in each of `regions`
    regions. With shared_clerks the claims and complaints clerks of a region
    form one pool (2 normal + 2 expert, the same headcount as two teams)."""
    processes = tuple(processes)
    unknown = set(processes) - set(PROCESSES)
    if unknown:
        raise ValueError(f"unknown processes {sorted(unknown)}; choose from {PROCESSES}")
    if shared_clerks and not {"claims", "complaints"} <= set(processes):
        raise ValueError("shared_clerks needs both claims and complaints")
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
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
            _add_underwriting(ag, tag)
        if "complaints" in processes:
            clerks = shared
            if clerks is None:
                clerks = ag.add_var(f"co_clerks_{tag}", var_attributes=['skill'])
                clerks.put({'skill': 0}); clerks.put({'skill': 1})
            _add_complaints(ag, tag, hidden, clerks)
    if hidden:
        ag.set_unobservable(token_attrs=hidden)
    return ag


# ----------------------------------------------------------- heuristic
def _binding_rate(b):
    """Myopic expected reward per expected resource time of one binding, by
    process: the same rate rules as the bpm_envs and multi-site anchors."""
    tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
    vals = [getattr(tok, 'value', tok) for (_place, tok) in b[0]]
    if tr.startswith('uw_'):
        tt = next((v['task_type'] for v in vals if isinstance(v, dict) and 'task_type' in v), None)
        code = next((v['code'] for v in vals if isinstance(v, dict) and 'code' in v), None)
        if tt is None or code is None:
            return None
        return 1.0 / ((1 if tt == code else 3) + 0.5)
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


def _builder(processes, shared_clerks=False):
    def build(n=1, causal_rl=False, allow_postpone=False, causal_postpone_tokenflow=False):
        return make_insurer(n, processes, shared_clerks, causal_rl=causal_rl,
                            allow_postpone=allow_postpone,
                            causal_postpone_tokenflow=causal_postpone_tokenflow)
    return build


# N = regions. "insurer" is the full organisation; the *_alone builders are the
# same sub-nets on their own (the per-process controls).
INSURER_BUILDERS = {
    "insurer": _builder(PROCESSES),
    "insurer_shared": _builder(PROCESSES, shared_clerks=True),
    "insurer_claims": _builder(("claims",)),
    "insurer_underwriting": _builder(("underwriting",)),
    "insurer_complaints": _builder(("complaints",)),
}
INSURER_HEURISTICS = {name: insurer_heuristic for name in INSURER_BUILDERS}
