r"""Headline figure data: performance gain over PPO vs MEASURED structure.

The paper's weakest framing is "wins on ncopies, ties on s1", which invites
"you picked a decomposable benchmark". The fix is to make the x-axis a
quantity the METHOD DISCOVERS rather than an env label: plot each (env, N) at
its measured realized component count K, and show the gain tracks it, with s1
sitting at K=1 on zero.

K is measured here with a RANDOM policy and no training -- it is a property of
the environment's realized provenance DAG, not of the learned policy, which is
what makes it usable as a predictor computed in advance.

Emits `fig_k_axis.json` (and a plot if matplotlib is present) with, per point:
  K            realized reward-bearing components (ccf's union-find partition)
  largest      share of reward mass in the biggest component
  fan_out      mean |succ(d)| for cgae's successor relation
  gain_*       paired normalized gain over ppo, from the stored cells

Run: python _fig_k_axis.py
"""
import os, sys, json, random, types, uuid, statistics as st
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

EPISODES = 12
LENGTH = 20
# postpone config must match how the cells were TRAINED, or K describes a
# different net than the one the numbers came from.
TOKENFLOW = True


def _prep(builder):
    pn = builder()
    pn.length = LENGTH
    for pl in pn.places:
        for t in pl.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn
    pn.causal_trace._static_comp_cache = None
    pn.causal_trace.postpone_tokenflow = TOKENFLOW
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__")
    toks = [t for pl in pn.places for t in pl.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)


def structure(builder, seed):
    random.seed(seed)
    env = _prep(builder)
    env.reset()
    done, steps = False, 0
    while not done and steps < 900:
        m = len(env.pn.pn_actions)
        if m == 0:
            break
        _, _, done, _, _ = env.step(random.randrange(m))
        steps += 1

    ct = env.pn.causal_trace
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    if n == 0:
        return None
    out = {}
    for i, a in enumerate(acts):
        for t in a.get('output_tokens', ()) or ():
            out[t] = i

    def par(t):
        info = ct.token_history.get_token(t)
        return info.get("parents", []) if info else []

    def lin(ids):
        f, seen, stack = set(), set(), list(ids)
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            h = out.get(t)
            if h is not None:
                f.add(h)
            for q in par(t):
                if q not in seen:
                    stack.append(q)
        return f

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # two passes: all unions BEFORE any root is used as a key, else early
    # entries land under stale roots and K inflates to ~one-per-reward.
    rw = []
    for tr in ct.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        d = [x for x in lin(tr.get('input_tokens', ())) if 0 <= x < n]
        for i in range(1, len(d)):
            uni(d[0], d[i])
        rw.append((rv, d))
    mass = {}
    for rv, d in rw:
        if d:
            mass[find(d[0])] = mass.get(find(d[0]), 0.0) + rv
    if not mass:
        return None

    succ = [set() for _ in range(n)]
    for idx, a in enumerate(acts):
        seen, stack = set(), list(a.get('input_tokens', ()) or ())
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            src = out.get(t)
            if src is not None and src != idx:
                succ[src].add(idx)
                continue
            for q in par(t):
                if q not in seen:
                    stack.append(q)

    tot = sum(mass.values())
    return dict(K=len(mass), largest=max(mass.values()) / tot,
                fan_out=sum(len(s) for s in succ) / n, decisions=n)


def gains_from_cells(pattern, key_fn, methods):
    """Paired normalized gain over ppo, from cells already on disk."""
    import glob
    cells = defaultdict(dict)
    base = {}
    for f in glob.glob(pattern):
        d = json.load(open(f))
        k = key_fn(d)
        if k is None:
            continue
        b = d['baselines']
        r, h = b['random_mean'], b['heuristic_mean']
        nrm = (d['greedy_final'] - r) / (h - r) if d.get('greedy_final') is not None else None
        if nrm is None:
            continue
        cells[k].setdefault(d['method'], {})[d['seed']] = nrm
        base[k] = (r, h)
    out = {}
    for k, bym in cells.items():
        ppo = bym.get('ppo') or bym.get('ppo_clip')
        if not ppo:
            continue
        row = {}
        for m in methods:
            if m not in bym:
                continue
            common = sorted(set(bym[m]) & set(ppo))
            if len(common) < 2:
                continue
            row[m] = st.mean([bym[m][s] - ppo[s] for s in common])
            row[m + '_n'] = len(common)
        row['ppo_level'] = st.mean(list(ppo.values()))
        out[k] = row
    return out


if __name__ == "__main__":
    POINTS = [
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=TOKENFLOW)),
        ("ncopies_N1", lambda: make_n_copies(1, causal_rl=True, allow_postpone=True,
                                             causal_postpone_tokenflow=TOKENFLOW)),
        ("ncopies_N2", lambda: make_n_copies(2, causal_rl=True, allow_postpone=True,
                                             causal_postpone_tokenflow=TOKENFLOW)),
        ("ncopies_N4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                             causal_postpone_tokenflow=TOKENFLOW)),
        ("ncopies_N8", lambda: make_n_copies(8, causal_rl=True, allow_postpone=True,
                                             causal_postpone_tokenflow=TOKENFLOW)),
    ]
    struct = {}
    print(f"structure by random rollout, {EPISODES} episodes, horizon {LENGTH}")
    print(f"  {'point':<12}{'K':>8}{'largest%':>10}{'fan-out':>9}{'decisions':>11}")
    for name, b in POINTS:
        rows = [r for r in (structure(b, 700 + e) for e in range(EPISODES)) if r]
        if not rows:
            print(f"  {name:<12} (no decisions)")
            continue
        f = lambda k: st.mean([r[k] for r in rows])
        struct[name] = dict(K=f('K'), K_sd=st.stdev([r['K'] for r in rows]) if len(rows) > 1 else 0.0,
                            largest=f('largest'), fan_out=f('fan_out'), decisions=f('decisions'))
        print(f"  {name:<12}{f('K'):>8.2f}{100*f('largest'):>9.1f}%{f('fan_out'):>9.2f}{f('decisions'):>11.1f}")

    nc = gains_from_cells("suite_results_ncopies_3way_crn/cells/*.json",
                          lambda d: f"ncopies_N{d['N']}",
                          ["cgae", "cgae_flow", "ccf", "cfgae"])
    s1 = gains_from_cells("suite_results_three_way_s1_crn/cells/*.json",
                          lambda d: "s1",
                          ["cgae", "cgae_flow", "cfgae"])
    gains = {**nc, **s1}

    print(f"\n  {'point':<12}{'K':>7}  gains over ppo")
    for name in struct:
        g = gains.get(name, {})
        bits = "  ".join(f"{m}={g[m]:+.3f}(n={g[m+'_n']})"
                         for m in ("cgae_flow", "cgae", "ccf", "cfgae") if m in g)
        print(f"  {name:<12}{struct[name]['K']:>7.2f}  {bits or '(no cells)'}")

    payload = {"episodes": EPISODES, "length": LENGTH, "tokenflow": TOKENFLOW,
               "structure": struct, "gains": gains}
    with open("fig_k_axis.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote fig_k_axis.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, mk in (("cgae_flow", "o"), ("cgae", "s"), ("ccf", "^")):
            xs, ys, lb = [], [], []
            for name, sdat in struct.items():
                g = gains.get(name, {})
                if m in g:
                    xs.append(sdat['K']); ys.append(g[m]); lb.append(name)
            if xs:
                ax.scatter(xs, ys, marker=mk, label=m, s=55)
                for x, y, t in zip(xs, ys, lb):
                    ax.annotate(t.replace("ncopies_", ""), (x, y), fontsize=7,
                                xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="0.6", lw=0.8, ls="--")
        ax.set_xlabel("realized causal components K (measured, random policy)")
        ax.set_ylabel("normalized gain over PPO")
        ax.set_title("Gain is predicted by discovered structure")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig("figures/fig_k_axis.png", dpi=160)
        print("wrote figures/fig_k_axis.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
