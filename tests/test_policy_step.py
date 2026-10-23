"""The PPO policy step computes its per-decision terms (chosen-action
log-probabilities, per-state KL, normalised entropy) with tensor operations
over the batch (agents._policy_step_terms, agents._normalized_entropy). These
tests pin them to the per-decision Python loop they replaced, kept here
verbatim as the reference, on random batches: decisions with and without
postpone nodes, actions of probability exactly 0, batches with no action
nodes, and postpone nodes without stored old log-probabilities."""
import pytest
import torch

from gympn.agents import _normalized_entropy, _policy_step_terms


def _reference(new_lp_a, new_lp_p, old_lp_a, old_lp_p, idx_a, idx_p, actions, old_logprob, advantages):
    has_a, has_p = new_lp_a is not None, new_lp_p is not None
    unique_samples = (idx_a.unique() if has_a else idx_p.unique())
    sel_new, sel_old, sel_adv, kld_terms = [], [], [], []
    for s in unique_samples:
        parts_new, parts_old = [], []
        if has_a:
            mask_a = (idx_a == s)
            ns_a, os_a = new_lp_a[mask_a].reshape(-1), old_lp_a[mask_a].reshape(-1)
            if ns_a.numel():
                parts_new.append(ns_a); parts_old.append(os_a)
        if has_p:
            mask_p = (idx_p == s)
            ns_p = new_lp_p[mask_p].reshape(-1)
            os_p = old_lp_p[mask_p].reshape(-1) if old_lp_p is not None else ns_p.detach()
            if ns_p.numel():
                parts_new.append(ns_p); parts_old.append(os_p)
        if not parts_new:
            continue
        ns_cat, os_cat = torch.cat(parts_new), torch.cat(parts_old)
        a = int(actions[s])
        sel_new.append(ns_cat[a].reshape(()))
        sel_old.append(old_logprob[s].reshape(()))
        sel_adv.append(advantages[s].reshape(()))
        with torch.no_grad():
            p_old = torch.exp(os_cat)
            p_old = p_old / p_old.sum().clamp_min(1e-8)
            terms = torch.where(p_old > 0, p_old * (os_cat - ns_cat), torch.zeros_like(p_old))
            kld_terms.append(float(terms.sum().item()))
    kld = torch.tensor(kld_terms).mean() if kld_terms else torch.tensor(0.0)
    return torch.stack(sel_new), torch.stack(sel_old), torch.stack(sel_adv), kld


def _reference_entropy(probs, logpis, batch_index):
    ent_sum, count = torch.tensor(0.0), 0
    for s in batch_index.unique():
        mask = (batch_index == s)
        n = int(mask.sum().item())
        raw = -(probs[mask] * logpis[mask]).sum()
        if n > 1:
            ent_sum = ent_sum + raw / torch.log(torch.tensor(float(n)))
        count += 1
    return ent_sum / count if count else torch.tensor(0.0)


def _log_softmax_groups(logits, idx, G):
    out = torch.empty_like(logits)
    for g in range(G):
        m = idx == g
        if m.any():
            out[m] = torch.log_softmax(logits[m], 0)
    return out


def _batch(seed, G=12, max_a=6, max_p=2, zeros=True, no_a=False, no_p=False):
    g = torch.Generator().manual_seed(seed)
    ca = torch.zeros(G, dtype=torch.long) if no_a else torch.randint(0, max_a + 1, (G,), generator=g)
    cp = torch.zeros(G, dtype=torch.long) if no_p else torch.randint(0, max_p + 1, (G,), generator=g)
    cp = torch.where(ca + cp == 0, torch.ones_like(cp), cp) if no_a else cp
    ca = torch.where(ca + cp == 0, torch.ones_like(ca), ca) if not no_a else ca
    idx_a = torch.repeat_interleave(torch.arange(G), ca)
    idx_p = torch.repeat_interleave(torch.arange(G), cp)
    idx = torch.cat([idx_a, idx_p])
    old = _log_softmax_groups(torch.randn(idx.numel(), generator=g) * 2, idx, G)
    if zeros:                                   # some actions of probability exactly 0
        drop = torch.rand(idx.numel(), generator=g) < 0.2
        old = torch.where(drop, torch.full_like(old, float('-inf')), old)
    logits = torch.randn(idx.numel(), generator=g, requires_grad=True)
    new_full = _log_softmax_groups(logits, idx, G)
    n = ca + cp
    actions = (torch.rand(G, generator=g) * n).long().clamp(max=(n - 1).clamp_min(0))
    nA = idx_a.numel()
    sp = lambda t: (t[:nA] if nA else None, t[nA:] if idx_p.numel() else None)
    return dict(logits=logits, new=sp(new_full), old=sp(old), idx_a=idx_a if nA else None,
                idx_p=idx_p if idx_p.numel() else None, actions=actions,
                old_logprob=torch.randn(G, generator=g), adv=torch.randn(G, generator=g), G=G, idx=idx)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kind", ["both", "no_p", "no_a", "no_old_p"])
def test_policy_step_terms_match_the_loop(seed, kind):
    b = _batch(seed, no_a=kind == "no_a", no_p=kind == "no_p")
    old_a, old_p = b["old"]
    if kind == "no_old_p":
        old_p = None
    args = (b["new"][0], b["new"][1], old_a, old_p, b["idx_a"], b["idx_p"], b["actions"], b["old_logprob"], b["adv"])
    ref = _reference(*args)
    got = _policy_step_terms(*args, b["G"])
    for r, x in zip(ref, got):
        assert r.shape == x.shape
        assert torch.allclose(r.detach().float(), x.detach().float(), atol=1e-6, rtol=1e-5)
    # same gradient through the chosen-action log-probs
    w = torch.randn(ref[0].numel(), generator=torch.Generator().manual_seed(seed))
    gr, = torch.autograd.grad((ref[0] * w).sum(), b["logits"], retain_graph=True)
    gx, = torch.autograd.grad((got[0] * w).sum(), b["logits"])
    assert torch.allclose(gr, gx, atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_normalized_entropy_matches_the_loop(seed):
    b = _batch(seed, zeros=False)
    lp = torch.cat([t for t in b["new"] if t is not None])
    p = lp.exp()
    ref = _reference_entropy(p, lp, b["idx"])
    got = _normalized_entropy(p, lp, b["idx"])
    assert torch.allclose(ref, got, atol=1e-6)
    gr, = torch.autograd.grad(ref, b["logits"], retain_graph=True)
    gx, = torch.autograd.grad(got, b["logits"])
    assert torch.allclose(gr, gx, atol=1e-6)
