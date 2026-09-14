r"""Where do a reward's k lineage decisions actually COME from?

k averages 8.48 on s1 while a case only passes through 2 decisions
(start1, start2). So ~6.5 of the 8.5 arrive by some other route. Three
candidates, and which one it is decides whether lineage is mis-defined:
  (a) resource recycling -- the employee token is consumed by a decision and
      re-emitted downstream, so its descendants carry that decision forever;
  (b) postpone re-emission -- a token-flow postpone re-emits the marking,
      chaining everything through it;
  (c) genuinely long causal chains.
Compares mean k with postpone ON vs OFF, and reports the TIME SPREAD of a
reward's lineage (how far back it reaches) -- a case takes a few time units,
so a lineage reaching much further back is resource-mediated, not case-flow.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
import numpy as np
import gympn.causal_traces as tm
from envs import make_env

_S = {}
def probe(tag):
    _S[tag] = {"k": [], "span": []}
    real = tm.CausalTraces._redistribute_ls_hca
    def spy(self, *a, **kw):
        out = real(self, *a, **kw)
        acts = self.transition_history.get_action_transitions()
        times = [x.get('time') for x in acts]
        for rt, k in (getattr(self, '_ls_hca_lineage_sizes', None) or []):
            _S[tag]["k"].append(k)
        # time span of each reward's lineage, via the pending z=True entries
        for t, items in enumerate(self._ls_hca_pending or []):
            for e in items:
                if len(e) >= 7 and e[2] and e[4] is not None:
                    _S[tag]["span"].append(float(e[4]))   # delay decision->reward
        return out
    tm.CausalTraces._redistribute_ls_hca = spy
    return real

def run(tag, postpone):
    real = probe(tag)
    env = make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=postpone)
    args = {"algorithm":"ppo-clip","episodes":6,"epochs":2,"batch_size":64,
        "max_episode_length":None,"policy_lr":3e-4,"policy_updates":1,
        "value_lr":3e-4,"value_updates":1,"gam":0.99,"lam":0.95,"eps":0.2,
        "vf_coeff":0.5,"ent_bonus":0.01,"policy_kld_limit":0.15,
        "causal_rl":True,"causal_scheme":"ls_hca","causal_beta":0.5,
        "verbose":0,"use_gpu":False,"agent_seed":0,"use_wandb":False,
        "open_tensorboard":False,"test_in_train":False,"save_freq":10**9,
        "name":"anat","datetag":False,"logdir":f"anat_{tag}"}
    saved = sys.argv; sys.argv = sys.argv[:1]
    try: env.training_run(length=20, args_dict=args)
    finally:
        sys.argv = saved
        tm.CausalTraces._redistribute_ls_hca = real
        import shutil; shutil.rmtree(f"anat_{tag}", ignore_errors=True)

run("postpone_ON", True)
run("postpone_OFF", False)
print("\n" + "="*66)
for tag in _S:
    k = np.array(_S[tag]["k"], float); sp = np.array(_S[tag]["span"], float)
    print(f"[{tag}]  rewards={len(k)}")
    print(f"   k (lineage size): mean={k.mean():.2f} median={np.median(k):.0f} max={int(k.max())}")
    if len(sp):
        print(f"   decision->reward delay: median={np.median(sp):.2f} "
              f"p90={np.quantile(sp,0.9):.2f} max={sp.max():.2f}")
