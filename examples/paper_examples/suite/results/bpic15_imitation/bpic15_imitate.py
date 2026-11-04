"""Imitation check on bpic15: can the policy network (same architecture and
view as the RL runs: AEPN, flat obs, global_view='edges', global attention,
count aggregation, actor masked by the static certificate) represent the
heuristic? Behaviour cloning on heuristic decisions, then greedy evaluation on
the RL runs' evaluation days, normalised with the run anchors.

usage: bpic15_imitate.py [days=40] [epochs=15] [out=<json>]"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

SUITE = r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite"
sys.path.insert(0, os.path.join(SUITE, "envs"))
torch.set_num_threads(1)

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor  # noqa: E402
from bpic15_env import BPIC15_BUILDERS, bpic15_heuristic, GROUPS  # noqa: E402

opts = dict(a.split("=", 1) for a in sys.argv[1:])
DAYS, EPOCHS = int(opts.get("days", 40)), int(opts.get("epochs", 15))
OUT = opts.get("out", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpic15_imitate.json"))
LENGTH, EVAL_SEED, BS, LR = 20, 555_000, 32, 1e-3
RANDOM, HEUR = 37.35, 56.375                         # run anchors (baselines_N1.json)


def make_env(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = BPIC15_BUILDERS["bpic15"](1, allow_postpone=False)
    pn.length = LENGTH
    pn.flat_obs = True
    pn.global_view = 'edges'
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, (obs[0] if isinstance(obs, tuple) else obs)


def heur_index(env):
    B = env.pn.pn_actions
    return B.index(bpic15_heuristic(env.pn, None, B))


def score(b):
    """The heuristic's own ranking key of a binding (equal keys = equivalent to the heuristic)."""
    name = str(b[2]._id)
    if name.startswith("al_"):
        return ("al", name.rsplit("_", 1)[1])
    g = name[2]
    s = next(getattr(t, "value", t) for (_p, t) in b[0]
             if isinstance(getattr(t, "value", t), dict) and "fI" in getattr(t, "value", t))
    return (s[f"f{g}"], GROUPS.index(g))


def collect(days):
    data = []
    for d in range(days):
        env, obs = make_env(1_000 + d)
        while env.pn.pn_actions:
            a = heur_index(env)
            data.append((obs['graph'], a))
            obs, _, done, _, _ = env.step(a)
            if done:
                break
    return data


def run_day(seed, choose):
    env, obs = make_env(seed)
    total, agree, n = 0.0, 0, 0
    while env.pn.pn_actions:
        B = env.pn.pn_actions
        a = choose(env, obs)
        agree += score(B[a]) == score(B[heur_index(env)])
        n += 1
        obs, r, done, _, _ = env.step(a)
        total += r
        if done:
            break
    return total, agree / max(1, n)


def evaluate(pol, days):
    pol.eval()

    def choose(env, obs):
        with torch.no_grad():
            return int(torch.argmax(pol({'graph': obs['graph']}).flatten()).item())
    res = [run_day(EVAL_SEED + i, choose) for i in range(days)]
    ret = float(np.mean([r for r, _ in res]))
    return ret, (ret - RANDOM) / (HEUR - RANDOM), float(np.mean([a for _, a in res]))


def batch_loss(pol, items):
    graphs = [g for g, _ in items]
    probs = pol({'graph': Batch.from_data_list(graphs)}).flatten()
    counts = [int(g.a_idx.numel()) for g in graphs]
    offs = np.cumsum([0] + counts[:-1])
    pos = torch.tensor([int(o) + a for o, (_, a) in zip(offs, items)])
    p = probs[pos]
    return -(p + 1e-8).log().mean(), float((p > 0.5).float().mean())


def main():
    t0 = time.time()
    log = {"days": DAYS, "epochs": EPOCHS, "evals": []}
    hres = [run_day(EVAL_SEED + i, lambda env, obs: heur_index(env)) for i in range(20)]
    log["heuristic_eval_return"] = float(np.mean([r for r, _ in hres]))
    print(f"heuristic on the 20 eval days: {log['heuristic_eval_return']:.2f} "
          f"(norm {(log['heuristic_eval_return'] - RANDOM) / (HEUR - RANDOM):+.3f})", flush=True)
    data = collect(DAYS)
    print(f"collected {len(data)} heuristic decisions from {DAYS} days ({time.time()-t0:.0f}s)", flush=True)
    env, obs = make_env(0)
    torch.manual_seed(0)
    pol = HeteroActor(input_size=-1, hidden_size=64, num_layers=3, metadata=env.pn.make_metadata(),
                      num_heads=1, dropout=0.0, residual=True, encoder="aepn",
                      encoder_kwargs={"global_attn": True, "count_aggr": True})
    with torch.no_grad():
        pol({'graph': obs['graph']})                      # materialise lazy layers
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    rng = random.Random(0)
    ret, nrm, agr = evaluate(pol, 5)
    print(f"epoch 0: greedy {ret:.2f} (norm {nrm:+.3f}), agreement with heuristic {agr:.2f}", flush=True)
    log["evals"].append({"epoch": 0, "return": ret, "norm": nrm, "agree": agr})
    for ep in range(1, EPOCHS + 1):
        pol.train()
        idx = list(range(len(data))); rng.shuffle(idx)
        losses, confs = [], []
        for i in range(0, len(idx), BS):
            loss, conf = batch_loss(pol, [data[j] for j in idx[i:i + BS]])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss)); confs.append(conf)
        msg = f"epoch {ep}: loss {np.mean(losses):.3f}, p(heuristic choice)>0.5 on {np.mean(confs):.2f} ({(time.time()-t0)/60:.0f} min)"
        if ep % 5 == 0 or ep == EPOCHS:
            days = 20 if ep == EPOCHS else 5
            ret, nrm, agr = evaluate(pol, days)
            msg += f" | greedy {ret:.2f} over {days} days (norm {nrm:+.3f}), agreement {agr:.2f}"
            log["evals"].append({"epoch": ep, "return": ret, "norm": nrm, "agree": agr, "days": days})
        log.setdefault("loss", []).append(float(np.mean(losses)))
        print(msg, flush=True)
    log["minutes"] = (time.time() - t0) / 60
    json.dump(log, open(OUT, "w"), indent=1)
    torch.save(pol, OUT.replace(".json", ".pth"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
