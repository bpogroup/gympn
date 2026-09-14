r"""Does truncating REUSE edges sharpen the lineage, without eating real chains?

Measured: on s1 a reward's lineage holds ~8 decisions, with the median link
reaching 8-9 time units back and the max reaching 20 -- the whole horizon --
while a case only lasts 3-7. Those long links are not case flow. They come
from re-supplied places: when a token re-enters such a place, `add_token`
records its parents as the firing transition's inputs, so it inherits that
whole ancestry, and every later token drawn from the place inherits it too.
Semantically the re-entering token is the SAME entity restored, not a product
of the work just done -- so that ancestry is a modelling artifact.

The fix has to be general to A-E PN, not task-assignment-specific. Two pieces:

  DETECTOR. A place is RE-SUPPLIED iff it has no external inflow: every
  transition that produces into p is forward-reachable from some consumer of
  p, so tokens in p only ever come back round from p itself. A case place fed
  by arrivals fails this -- the arrival transition is not reachable from the
  place's own consumers. This deliberately replaces `_is_cyclic_place`, which
  cannot tell a resource pool from a REWORK LOOP (in f_loop_disjoint the case
  returns to waiting1, making it cyclic though its ancestry is legitimate).

  WALK. When the lineage walk reaches a token that sits in a re-supplied
  place, do not traverse its parents.

Two envs, because one alone cannot test generality:
  s1_stoch_sequence  -- true shared pool. k should DROP sharply.
  f_loop_disjoint    -- rework loop, per-stage pools. The reworked CASE place
                        must NOT be classified re-supplied, so k should be
                        roughly UNCHANGED. If it also collapses, the detector
                        is eating real lineage and the idea is not ready.

Run: python _diag_reuse_edges.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as tm
from envs import make_env

_place_of = {}          # token id -> place id
_real_reg = tm.CausalTraces.register_token


def _spy_reg(self, token, transition, parent_tokens, created_by=None, time=None):
    tid = _real_reg(self, token, transition, parent_tokens, created_by, time)
    try:
        for p in getattr(transition, 'outgoing', ()) or ():
            if any(t is token for t in (getattr(p, 'marking', None) or ())):
                _place_of[tid] = p._id
                break
    except Exception:
        pass
    return tid


tm.CausalTraces.register_token = _spy_reg


def resupplied_places(pn):
    """Places with NO external inflow (see module docstring)."""
    trans = list(pn.actions) + list(pn.events)
    consumers = defaultdict(list)
    producers = defaultdict(list)
    for t in trans:
        for p in t.incoming:
            consumers[p._id].append(t)
        for p in t.outgoing:
            producers[p._id].append(t)
    out = set()
    for pid, prods in producers.items():
        cons = consumers.get(pid)
        if not cons or not prods:
            continue
        seen, stack = set(), list(cons)
        while stack:
            t = stack.pop()
            if t._id in seen:
                continue
            seen.add(t._id)
            for p in t.outgoing:
                stack.extend(consumers.get(p._id, []))
        if all(t._id in seen for t in prods):
            out.add(pid)
    return out


_K = {}


def probe(tag, resupplied):
    _K[tag] = {"full": [], "trunc": [], "placed": 0, "total": 0}
    real = tm.CausalTraces._redistribute_ls_hca

    def spy(self, *a, **kw):
        out = real(self, *a, **kw)
        try:
            acts = self.transition_history.get_action_transitions()
            tok2act = {}
            for i, act in enumerate(acts):
                for t in act.get('output_tokens', ()):
                    tok2act[t] = i

            def walk(input_ids, firing_idx, truncate):
                found = set()
                if firing_idx is not None:
                    found.add(firing_idx)
                seen, stack = set(), list(input_ids)
                while stack:
                    tid = stack.pop()
                    if tid in seen:
                        continue
                    seen.add(tid)
                    if tid in tok2act:
                        found.add(tok2act[tid])
                    if truncate and _place_of.get(tid) in resupplied:
                        continue            # reuse edge: do not inherit ancestry
                    stack.extend(self.token_history.get_parents(tid))
                return found

            for tr in self.transition_history.transitions:
                if tr.get('reward', 0.0) == 0.0:
                    continue
                ins = tr.get('input_tokens', [])
                _K[tag]["total"] += len(ins)
                _K[tag]["placed"] += sum(1 for t in ins if t in _place_of)
                _K[tag]["full"].append(len(walk(ins, None, False)))
                _K[tag]["trunc"].append(len(walk(ins, None, True)))
        except Exception as e:
            print(f"[warn] {tag}: {e}")
        return out

    tm.CausalTraces._redistribute_ls_hca = spy
    return real


def run(tag, env_name, postpone, length):
    env = make_env(env_name, causal_rl=True, allow_postpone=postpone)
    rs = resupplied_places(env)
    print(f"[{tag}] re-supplied places: {sorted(rs) or 'NONE'}")
    real = probe(tag, rs)
    args = {"algorithm": "ppo-clip", "episodes": 6, "epochs": 2, "batch_size": 64,
            "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 1,
            "value_lr": 3e-4, "value_updates": 1, "gam": 0.99, "lam": 0.95,
            "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01,
            "policy_kld_limit": 0.15, "causal_rl": True,
            "causal_scheme": "ls_hca", "causal_beta": 0.5, "verbose": 0,
            "use_gpu": False, "agent_seed": 0, "use_wandb": False,
            "open_tensorboard": False, "test_in_train": False,
            "save_freq": 10**9, "name": "reuse", "datetag": False,
            "logdir": f"reuse_{tag}"}
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=length, args_dict=args)
    finally:
        sys.argv = saved
        tm.CausalTraces._redistribute_ls_hca = real
        import shutil
        shutil.rmtree(f"reuse_{tag}", ignore_errors=True)


run("s1", "s1_stoch_sequence", True, 20)
run("f_loop", "f_loop_disjoint", True, 10)

print("\n" + "=" * 70)
print(f"{'env':<10}{'rewards':>8}{'k_full':>9}{'k_trunc':>10}{'change':>10}   place-tag hit rate")
print("-" * 70)
for tag, r in _K.items():
    f = np.array(r["full"], float)
    t = np.array(r["trunc"], float)
    if not len(f):
        print(f"{tag:<10} (no rewards)")
        continue
    hit = r["placed"] / max(r["total"], 1)
    print(f"{tag:<10}{len(f):>8}{f.mean():>9.2f}{t.mean():>10.2f}"
          f"{(t.mean() - f.mean()) / max(f.mean(), 1e-9):>9.1%}   {hit:.1%}")
print()
print("EXPECTED if the diagnosis holds: s1 drops sharply, f_loop barely moves.")
print("If f_loop also collapses, the detector is eating legitimate case lineage.")
