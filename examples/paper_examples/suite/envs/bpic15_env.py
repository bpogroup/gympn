"""Five municipalities handling environmental permits, calibrated on BPI
Challenge 2015 (BPM version; calibration in scripts/bpic15_calibrate.py).

WHY. In BPIC 2015 each municipality has its own permit team, except for a few
employees who work for two municipalities in the same weeks: 560532 for
municipalities 2 and 5 from May 2014, 560752 for municipalities 4 and 5 from
October 2012. An employee works in one office per day, so the coupling is a
decision at the start of the day: to which municipality does the shared
employee go today. The pool place of the shared employee touches both
municipalities, so the static certificate joins them into one component (and
NF-GAE reduces to joint PPO on that pair). Once the shared employee has been
allocated the pool is empty and has no producer, so the allocation transitions
are dead for the rest of the day and the marking certificate splits the pair
again. Every municipality without a shared employee is a component on its own
from the start.

ONE MUNICIPALITY (one episode = one working day of LENGTH time units):

  intake -> assessment -> decision   every permit passes the three task groups;
                                     reward 1 when the decision is taken
  staff                              the municipality's core staff in the
                                     roster period, each with the task groups
                                     they FIT (at least 25% of their events)
  service                            base 2 / 4 / 2 for intake / assessment /
                                     decision (the 0.27 / 0.53 / 0.20 split of
                                     the events), doubled for a task group the
                                     employee does not fit, + U{0,1}
  arrivals                           Poisson, at rate 0.9 * load * headcount /
                                     9.5 permits per time unit, where load is
                                     the municipality's events per working day
                                     per core employee relative to the mean
                                     (a shared employee counts half) and 9.5 is
                                     the expected work of a permit
  backlog                            drawn at the start of the day: Binomial
                                     (headcount, 1/2) permits waiting for intake
                                     and for decision, Binomial(2 * headcount,
                                     1/2) waiting for assessment

From the data: the task groups, the core staff, what each employee fits, the
relative load, and who is shared between which municipalities. Assumed: the
service times (the log records completions only, 46% of them at midnight and
two thirds in batches), the length of a day and the backlog at its start.

The decisions are which permit (task group) to give to which available
employee, and, at the start of the day, where the shared employee works.
"""
import random

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

GROUPS = ("I", "A", "D")
BASE = {"I": 2, "A": 4, "D": 2}
MISFIT = 2                       # service multiplier for a task group the employee does not fit
WORK = sum(BASE.values()) + 1.5  # expected service of a permit by fitting staff
RHO = 0.9

# scripts/bpic15_calibrate.py on the five municipality logs. staff: (employee,
# task groups they fit); load: events per working day per core employee,
# relative to the mean; shared: (employee, fit, municipalities).
ROSTERS = {
    "2014": {
        "staff": {1: [("560872", "IA"), ("560912", "AD"), ("2670601", "A"), ("11744364", "A"),
                      ("560890", "AD"), ("9264148", "I")],
                  2: [("560519", "AD"), ("560521", "A"), ("560458", "AD"), ("22445896", "I")],
                  3: [("560454", "A"), ("560749", "IAD"), ("560741", "IAD"), ("560696", "IA"),
                      ("2013365", "A"), ("5025869", "A"), ("560673", "IA")],
                  4: [("560781", "IAD"), ("560752", "IA"), ("1550894", "A"), ("560852", "A")],
                  5: [("560604", "A"), ("560602", "AD"), ("560429", "A"), ("560608", "A"),
                      ("560600", "AD"), ("560530", "IAD")]},
        "load": {1: 0.93, 2: 1.04, 3: 0.84, 4: 1.22, 5: 0.97},
        "shared": [("560532", "IAD", (2, 5))],
    },
    "2012": {
        "staff": {1: [("560872", "IA"), ("3273854", "A"), ("560890", "AD"), ("2670601", "A"),
                      ("560912", "I"), ("560462", "A")],
                  2: [("560530", "IA"), ("560532", "IA"), ("560519", "AD"), ("560458", "AD"),
                      ("560521", "A")],
                  3: [("560673", "IAD"), ("560454", "A"), ("2013365", "A"), ("560749", "IAD"),
                      ("560741", "AD"), ("560696", "IA")],
                  4: [("560781", "IAD"), ("560852", "A"), ("560821", "IA"), ("1550894", "A")],
                  5: [("560602", "AD"), ("560604", "IA"), ("560600", "IA"), ("560429", "A")]},
        "load": {1: 0.71, 2: 0.98, 3: 0.98, 4: 0.95, 5: 1.38},
        "shared": [("560752", "IA", (4, 5))],
    },
}


def _fit(groups):
    return {f"f{g}": int(g in groups) for g in GROUPS}


def _arrive(rate):
    def arrive(a):
        return [SimToken(a, delay=random.expovariate(rate)), SimToken({"permit": 1})]
    return arrive


def _open(slot):
    """Start-of-day backlog: each slot holds a permit with probability 1/2."""
    stage = slot["stage"]
    if random.random() < 0.5:
        return [SimToken({"permit": 1}) if g == stage else None for g in GROUPS]
    return [None, None, None]


def _starter(g):
    def start(p, s):
        base = BASE[g] * (1 if s[f"f{g}"] else MISFIT)
        return [SimToken((p, s), delay=base + random.randint(0, 1))]
    return start


def _wip_starter(g):
    """wip=True: the busy token (permit, employee's fit, a job key) is present
    at once, so the observation shows the work in progress; the service time
    goes to a timer token in an unobservable place, and the task ends when the
    timer with the same job key comes due. The key is the employee's fit (in
    the hidden attribute "job"): busy tokens with the same key are identical
    tokens, so which of their timers ends which of them does not matter, and
    the pairing is exact."""
    def start(p, s):
        job = 4 * s["fI"] + 2 * s["fA"] + s["fD"]
        base = BASE[g] * (1 if s[f"f{g}"] else MISFIT)
        return [SimToken({**p, **s, "job": job}), SimToken({"job": job}, delay=base + random.randint(0, 1))]
    return start


def _wip_handover(b, _timer):
    return [SimToken({f"f{x}": b[f"f{x}"] for x in GROUPS}), SimToken({"permit": 1})]


def _wip_decided(b, _timer):
    return [SimToken({f"f{x}": b[f"f{x}"] for x in GROUPS})]


def _same_job(b, timer):
    return b["job"] == timer["job"]


def _handover(b):
    return [SimToken(b[1]), SimToken(b[0])]       # employee back, permit to the next task group


def _decided(b):
    return [SimToken(b[1])]


def _move(s):
    return [SimToken(dict(s))]


def make_bpic15(roster="2014", municipalities=(1, 2, 3, 4, 5), allocate=True,
                allow_postpone=False, wip=False):
    """The municipalities of `roster`. With allocate=False the shared employees
    are left out (each municipality has only its own staff), which gives the
    fully independent control. With wip=True a task in progress is a busy
    token that is present (observed) from its start, paired with a timer
    token in an unobservable place that ends it (_wip_starter): the same
    dynamics, but the observation shows who works on what (and, with
    GymProblem.obs_token_age, for how long), never when it will end."""
    hidden_places, hidden_attrs = [], {}
    spec = ROSTERS[roster]
    municipalities = tuple(municipalities)
    ag = GymProblem(allow_postpone=allow_postpone)
    teams = {}
    for m in municipalities:
        tag = str(m)
        q = {g: ag.add_var(f"q{g}_{tag}", var_attributes=["permit"]) for g in GROUPS}
        team = ag.add_var(f"staff_{tag}", var_attributes=[f"f{g}" for g in GROUPS])
        for _emp, fit in spec["staff"][m]:
            team.put(_fit(fit))
        teams[m] = team
        heads = len(spec["staff"][m]) + 0.5 * sum(m in pair for _e, _f, pair in spec["shared"])
        rate = RHO * spec["load"][m] * heads / WORK
        ag.__dict__.setdefault("arrival_rates", {})[f"arrival_{tag}"] = rate
        arrival = ag.add_var(f"arrival_{tag}", var_attributes=["permit"])
        arrival.put({"permit": 1})
        ag.add_event([arrival], [arrival, q["I"]], _arrive(rate), name=f"arrive_{tag}")
        seed = ag.add_var(f"backlog_{tag}", var_attributes=["stage"])
        n = len(spec["staff"][m])
        for g, k in (("I", n), ("A", 2 * n), ("D", n)):
            for _ in range(k):
                seed.put({"stage": g})
        ag.add_event([seed], [q["I"], q["A"], q["D"]], _open, name=f"open_{tag}")
        for g, nxt in (("I", "A"), ("A", "D"), ("D", None)):
            if wip:
                busy = ag.add_var(f"busy{g}_{tag}", var_attributes=["permit"] + [f"f{x}" for x in GROUPS] + ["job"])
                timer = ag.add_var(f"timer{g}_{tag}", var_attributes=["job"])
                hidden_places.append(timer._id)
                hidden_attrs[busy._id] = ["job"]
                ag.add_action([q[g], team], [busy, timer], behavior=_wip_starter(g), name=f"do{g}_{tag}")
                if nxt is None:
                    ag.add_event([busy, timer], [team], _wip_decided, name=f"done{g}_{tag}", guard=_same_job,
                                 reward_function=lambda b, t: 1)
                else:
                    ag.add_event([busy, timer], [team, q[nxt]], _wip_handover, name=f"done{g}_{tag}",
                                 guard=_same_job)
                continue
            busy = ag.add_var(f"busy{g}_{tag}", var_attributes=["permit"] + [f"f{x}" for x in GROUPS])
            ag.add_action([q[g], team], [busy], behavior=_starter(g), name=f"do{g}_{tag}")
            if nxt is None:
                ag.add_event([busy], [team], _decided, name=f"done{g}_{tag}",
                             reward_function=lambda b: 1)
            else:
                ag.add_event([busy], [team, q[nxt]], _handover, name=f"done{g}_{tag}")
    if allocate:
        for emp, fit, pair in spec["shared"]:
            if not set(pair) <= set(municipalities):
                continue
            pool = ag.add_var(f"pool_{emp}", var_attributes=[f"f{g}" for g in GROUPS])
            pool.put(_fit(fit))
            for m in pair:
                ag.add_action([pool], [teams[m]], behavior=_move, name=f"al_{emp}_{m}")
    if wip:
        ag.set_unobservable(simvars=hidden_places, token_attrs=hidden_attrs)
    return ag


def bpic15_resample_hidden(pn):
    """Redraw what a policy cannot know (for a fair rollout planner,
    gympn.policy_iteration.RolloutPlanner(resample=...)): the time of each
    municipality's next arrival. The simulator draws it when the previous
    arrival fires, and the observation does not show it (only tokens whose
    time has come are observed). Arrivals are Poisson, so the waiting time
    redrawn from the same rate is exactly what an observer faces. Uses the
    global `random` stream, which the planner seeds per rollout."""
    from simpn.simulator import SimToken
    for place in pn.places:
        rate = pn.__dict__.get("arrival_rates", {}).get(place._id)
        if rate is None:
            continue
        for tok in [t for t in place.marking if t.time > pn.clock]:
            place.remove_token(tok)
            place.add_token(SimToken(tok.value, time=pn.clock + random.expovariate(rate)))


# ----------------------------------------------------------- heuristic
def _queue_pressure(net, m):
    """Waiting work in municipality m per employee present."""
    def count(name):
        p = next((p for p in net.places if p._id == name), None)
        return len(p.marking) if p is not None else 0
    work = sum(BASE[g] * count(f"q{g}_{m}") for g in GROUPS)
    return work / max(1, count(f"staff_{m}"))


def bpic15_heuristic(observable_net, tokens_comb, bindings=None):
    """Start of the day: send each shared employee to the municipality with the
    most waiting work per employee. During the day: give an employee a task
    group they fit if there is one, the one closest to a decision first."""
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ["postpone"]]
    if not real:
        return bindings[0]
    alloc = [b for b in real if str(b[2]._id).startswith("al_")]
    if alloc:
        return max(alloc, key=lambda b: _queue_pressure(observable_net, str(b[2]._id).rsplit("_", 1)[1]))

    def score(b):
        g = str(b[2]._id)[2]
        s = next(getattr(tok, "value", tok) for (_p, tok) in b[0]
                 if isinstance(getattr(tok, "value", tok), dict) and "fI" in getattr(tok, "value", tok))
        return (s[f"f{g}"], GROUPS.index(g))
    return max(real, key=score)


def _builder(roster, allocate=True, municipalities=(1, 2, 3, 4, 5), wip=False):
    def build(n=1, allow_postpone=False):
        return make_bpic15(roster, municipalities, allocate, allow_postpone, wip)
    build.uw_mismatch = 3
    build.wait_anchor = False
    return build


# N is ignored (the organisation is the five municipalities of the roster).
BPIC15_BUILDERS = {
    "bpic15": _builder("2014"),
    "bpic15_2012": _builder("2012"),
    "bpic15_noshare": _builder("2014", allocate=False),
    "bpic15_wip": _builder("2014", wip=True),
}
BPIC15_HEURISTICS = {name: bpic15_heuristic for name in BPIC15_BUILDERS}
