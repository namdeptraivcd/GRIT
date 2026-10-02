#!/usr/bin/env python3
"""Checks for fixed-base top-k+tail trust-region preservation."""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grit.preservation_loss import preservation_kl_loss
from grit.trust_region import (
    coarsened_policy_log_probs, kl_from_log_probs, log1mexp, project_to_kl_ball,
)


def base_statistics(base_logits, k):
    logp = F.log_softmax(base_logits, dim=-1)
    ids = torch.topk(logp, k, dim=-1).indices
    selected = torch.gather(logp, -1, ids)
    tail = log1mexp(torch.logsumexp(selected, dim=-1))
    return ids, selected, tail


def main():
    torch.manual_seed(23)
    base = torch.randn(2, 3, 7, dtype=torch.float64)
    policy = (base+0.8*torch.randn_like(base)).requires_grad_(True)
    full_kl = kl_from_log_probs(F.log_softmax(policy, -1), F.log_softmax(base, -1))

    previous = None
    for k in (1, 3, 6, 7):
        ids, base_logp, tail = base_statistics(base, k)
        policy_logp = coarsened_policy_log_probs(policy, ids)
        anchor = torch.cat((base_logp.float(), tail.float().unsqueeze(-1)), -1)
        anchor = anchor-torch.logsumexp(anchor, -1, keepdim=True)
        coarse_kl = kl_from_log_probs(policy_logp, anchor)
        assert torch.all(coarse_kl <= full_kl.float()+2e-6)
        if previous is not None:
            assert coarse_kl.mean() >= previous.mean()-2e-6
        previous = coarse_kl
    torch.testing.assert_close(previous, full_kl.float(), atol=3e-6, rtol=3e-6)

    ids, base_logp, tail = base_statistics(base, 3)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    projection = project_to_kl_ball(
        policy, epsilon=0.03, response_mask=mask, base_topk_ids=ids,
        base_topk_log_probs=base_logp, base_log_tail=tail,
    )
    assert torch.all(projection.projected_kl[mask] <= 0.03+2e-6)
    assert torch.all(projection.eta[~projection.violation_mask] == 0)
    identical = project_to_kl_ball(
        base, epsilon=0.03, response_mask=mask, base_topk_ids=ids,
        base_topk_log_probs=base_logp, base_log_tail=tail,
    )
    assert torch.count_nonzero(identical.eta) == 0

    # Eq. 9/10 gradient: hold the projected target fixed while perturbing logits.
    one_policy = policy[:1, :1].detach().clone().requires_grad_(True)
    one_ids, one_base, one_tail = ids[:1, :1], base_logp[:1, :1], tail[:1, :1]
    result = preservation_kl_loss(
        one_policy, epsilon_pres=1e-5, response_mask=torch.ones(1, 1, dtype=torch.bool),
        base_topk_ids=one_ids, base_topk_log_probs=one_base, base_log_tail=one_tail,
        reduction="sum",
    )
    target = result.projection.projected_log_probs.detach()
    expected_grad, = torch.autograd.grad(result.loss, one_policy)

    def fixed_target_loss(logits):
        logp = coarsened_policy_log_probs(logits, one_ids)
        return torch.sum(logp.exp()*(logp-target))

    radius = 1e-4
    numerical = torch.zeros_like(one_policy)
    for index in range(one_policy.shape[-1]):
        plus, minus = one_policy.detach().clone(), one_policy.detach().clone()
        plus[..., index] += radius
        minus[..., index] -= radius
        numerical[..., index] = (fixed_target_loss(plus)-fixed_target_loss(minus))/(2*radius)
    torch.testing.assert_close(expected_grad, numerical, atol=3e-4, rtol=3e-3)

    full = preservation_kl_loss(policy, base, epsilon_pres=0.03, response_mask=mask, reduction="sum")
    all_ids, all_base, all_tail = base_statistics(base, base.shape[-1])
    top_v = preservation_kl_loss(
        policy, epsilon_pres=0.03, response_mask=mask, base_topk_ids=all_ids,
        base_topk_log_probs=all_base, base_log_tail=all_tail, reduction="sum",
    )
    torch.testing.assert_close(top_v.projection.token_kl, full.projection.token_kl, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(top_v.loss, full.loss, atol=3e-6, rtol=3e-6)

    print("coarsened_kl_bounded=True")
    print("top_v_matches_full=True")
    print("projection_inside_ball=True")
    print("eta_zero_when_feasible=True")
    print("loss_gradient_finite_difference=True")


if __name__ == "__main__":
    main()
