"""Token trust-region projection on a fixed base-policy top-k plus tail."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrustRegionProjection:
    projected_probs: torch.Tensor
    projected_log_probs: torch.Tensor
    token_kl: torch.Tensor
    projected_kl: torch.Tensor
    violation_mask: torch.Tensor
    active_mask: torch.Tensor
    support_mask: torch.Tensor | None
    eta: torch.Tensor
    policy_log_probs: torch.Tensor
    base_log_probs: torch.Tensor
    default_probability: float
    base_topk_ids: torch.Tensor | None = None


def kl_from_log_probs(policy_log_probs: torch.Tensor, anchor_log_probs: torch.Tensor) -> torch.Tensor:
    if policy_log_probs.shape != anchor_log_probs.shape:
        raise ValueError("Policy and anchor log-probabilities must have identical shapes")
    return torch.sum(policy_log_probs.exp() * (policy_log_probs-anchor_log_probs), dim=-1)


def log1mexp(log_x: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(log_x)); keep a finite tail if mass rounds to one."""
    compute = log_x.float().clamp_max(-torch.finfo(torch.float32).eps)
    split = -math.log(2.0)
    result = torch.where(compute < split, torch.log1p(-compute.exp()), torch.log(-torch.expm1(compute)))
    return result.clamp_min(math.log(torch.finfo(torch.float32).tiny))


def coarsened_policy_log_probs(policy_logits: torch.Tensor, base_topk_ids: torch.Tensor) -> torch.Tensor:
    """Return k gathered policy classes plus a full-vocabulary tail class."""
    if base_topk_ids.shape[:-1] != policy_logits.shape[:-1]:
        raise ValueError("base_topk_ids must match policy logits except for the class dimension")
    if base_topk_ids.dtype != torch.long:
        raise ValueError("base_topk_ids must be int64")
    if base_topk_ids.numel() and (base_topk_ids.min() < 0 or base_topk_ids.max() >= policy_logits.shape[-1]):
        raise ValueError("base_topk_ids contains an id outside the vocabulary")
    compute = policy_logits.float()
    log_z = torch.logsumexp(compute, dim=-1, keepdim=True)
    top_log_probs = torch.gather(compute, -1, base_topk_ids)-log_z
    tail = log1mexp(torch.logsumexp(top_log_probs, dim=-1))
    return torch.cat((top_log_probs, tail.unsqueeze(-1)), dim=-1)


def coarsened_base_log_probs(base_topk_log_probs: torch.Tensor, base_log_tail: torch.Tensor) -> torch.Tensor:
    if base_topk_log_probs.shape[:-1] != base_log_tail.shape:
        raise ValueError("base_log_tail shape mismatch")
    result = torch.cat((base_topk_log_probs.float(), base_log_tail.float().unsqueeze(-1)), dim=-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Stored base top-k statistics must be finite")
    return result-torch.logsumexp(result, dim=-1, keepdim=True)


def geometric_interpolation_log_probs(policy_log_probs, anchor_log_probs, eta):
    while eta.ndim < policy_log_probs.ndim:
        eta = eta.unsqueeze(-1)
    mixed = (policy_log_probs+eta*anchor_log_probs)/(eta+1.0)
    return mixed-torch.logsumexp(mixed, dim=-1, keepdim=True)


def _bracket_eta(policy, anchor, epsilon, violation, max_eta, iterations):
    high = torch.ones_like(violation, dtype=policy.dtype)
    for _ in range(iterations):
        projected = geometric_interpolation_log_probs(policy, anchor, high)
        more = (kl_from_log_probs(projected, anchor) > epsilon) & violation & (high < max_eta)
        high = torch.where(more, high*2.0, high)
        if not bool(more.any()):
            break
    return high.clamp_max(max_eta)


def project_log_probs_to_kl_ball(
    policy_log_probs: torch.Tensor,
    base_log_probs: torch.Tensor,
    *, epsilon: float,
    response_mask: torch.Tensor | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 40,
    max_eta: float = 1e12,
    base_topk_ids: torch.Tensor | None = None,
) -> TrustRegionProjection:
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if policy_log_probs.shape != base_log_probs.shape:
        raise ValueError("Coarsened policy/base distributions must have identical shapes")
    if response_mask is not None and response_mask.shape != policy_log_probs.shape[:-1]:
        raise ValueError("response_mask shape mismatch")
    detached_policy, detached_base = policy_log_probs.detach(), base_log_probs.detach()
    token_kl = kl_from_log_probs(detached_policy, detached_base)
    active = torch.ones_like(token_kl, dtype=torch.bool) if response_mask is None else response_mask.bool()
    violation = (token_kl > epsilon+tolerance) & active
    low = torch.zeros_like(token_kl)
    high = _bracket_eta(detached_policy, detached_base, epsilon, violation, max_eta, max_iterations)
    for _ in range(max_iterations):
        mid = (low+high)*0.5
        projected = geometric_interpolation_log_probs(detached_policy, detached_base, mid)
        too_far = kl_from_log_probs(projected, detached_base) > epsilon
        low = torch.where(too_far & violation, mid, low)
        high = torch.where((~too_far) & violation, mid, high)
    eta = torch.where(violation, high, torch.zeros_like(token_kl))
    projected = geometric_interpolation_log_probs(detached_policy, detached_base, eta)
    projected = torch.where(violation.unsqueeze(-1), projected, detached_policy)
    return TrustRegionProjection(
        projected_probs=projected.exp(), projected_log_probs=projected,
        token_kl=token_kl, projected_kl=kl_from_log_probs(projected, detached_base),
        violation_mask=violation, active_mask=active, support_mask=None, eta=eta,
        policy_log_probs=policy_log_probs, base_log_probs=base_log_probs,
        default_probability=0.0, base_topk_ids=base_topk_ids,
    )


def project_to_kl_ball(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    *, epsilon: float,
    response_mask: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    top_k: int | None = None,
    default_probability: float = 1e-12,
    base_topk_ids: torch.Tensor | None = None,
    base_topk_log_probs: torch.Tensor | None = None,
    base_log_tail: torch.Tensor | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 40,
    max_eta: float = 1e12,
) -> TrustRegionProjection:
    """Project a full distribution or a fixed base-top-k coarsening."""
    del selected_token_ids, default_probability
    stored = base_topk_ids is not None or base_topk_log_probs is not None or base_log_tail is not None
    if stored:
        if base_topk_ids is None or base_topk_log_probs is None or base_log_tail is None:
            raise ValueError("All stored base top-k tensors are required")
        policy_log = coarsened_policy_log_probs(policy_logits, base_topk_ids)
        base_log = coarsened_base_log_probs(base_topk_log_probs, base_log_tail)
    elif top_k is not None:
        if base_logits is None or base_logits.shape != policy_logits.shape:
            raise ValueError("Dense base logits are required to derive fixed top-k support")
        k = min(top_k, policy_logits.shape[-1])
        base_full = F.log_softmax(base_logits.float(), dim=-1)
        base_topk_ids = torch.topk(base_full, k, dim=-1).indices
        base_topk_log_probs = torch.gather(base_full, -1, base_topk_ids)
        base_log_tail = log1mexp(torch.logsumexp(base_topk_log_probs, dim=-1))
        policy_log = coarsened_policy_log_probs(policy_logits, base_topk_ids)
        base_log = coarsened_base_log_probs(base_topk_log_probs, base_log_tail)
    else:
        if base_logits is None or base_logits.shape != policy_logits.shape:
            raise ValueError("Full-vocabulary policy/base logits must have identical shapes")
        policy_log = F.log_softmax(policy_logits.float(), dim=-1)
        base_log = F.log_softmax(base_logits.float(), dim=-1)
    return project_log_probs_to_kl_ball(
        policy_log, base_log, epsilon=epsilon, response_mask=response_mask,
        tolerance=tolerance, max_iterations=max_iterations, max_eta=max_eta,
        base_topk_ids=base_topk_ids,
    )


def token_kl_from_logits(policy_logits, base_logits, *, support_mask=None, default_probability=1e-12, top_k=None):
    del support_mask, default_probability
    if top_k is None:
        return kl_from_log_probs(F.log_softmax(policy_logits.float(), -1), F.log_softmax(base_logits.float(), -1))
    return project_to_kl_ball(policy_logits, base_logits, epsilon=float("inf"), top_k=top_k).token_kl


def build_sparse_support_mask(policy_logits, *, base_logits=None, selected_token_ids=None, top_k=None):
    """Compatibility helper: the support is determined only by the base."""
    del policy_logits, selected_token_ids
    if top_k is None:
        return None
    if base_logits is None:
        raise ValueError("base_logits is required for fixed base support")
    mask = torch.zeros_like(base_logits, dtype=torch.bool)
    mask.scatter_(-1, torch.topk(base_logits, min(top_k, base_logits.shape[-1]), dim=-1).indices, True)
    return mask


def sparse_default_log_probs(logits, *, support_mask, default_probability=1e-12):
    """Legacy helper retained outside the trainer; the trainer never calls it."""
    if support_mask is None:
        return F.log_softmax(logits.float(), dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    sparse = torch.where(support_mask, probs, torch.as_tensor(default_probability, device=logits.device))
    return (sparse/sparse.sum(-1, keepdim=True)).log()
