"""Preservation loss on a fixed base-policy top-k plus tail."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from grit.trust_region import TrustRegionProjection, project_to_kl_ball


@dataclass(frozen=True)
class PreservationLossResult:
    loss: torch.Tensor
    token_loss: torch.Tensor
    projection: TrustRegionProjection


def aggregate_preservation_loss(loss_mat, loss_mask, reduction):
    mask = loss_mask.to(loss_mat.dtype)
    if reduction == "none":
        return loss_mat*mask
    if reduction == "sum":
        return (loss_mat*mask).sum()
    if reduction == "token-mean":
        return (loss_mat*mask).sum()/mask.sum().clamp_min(1)
    if reduction == "seq-mean-token-sum":
        valid = mask.sum(-1).gt(0).to(loss_mat.dtype)
        return ((loss_mat*mask).sum(-1)*valid).sum()/valid.sum().clamp_min(1)
    if reduction == "seq-mean-token-mean":
        lengths = mask.sum(-1)
        per_seq = (loss_mat*mask).sum(-1)/lengths.clamp_min(1)
        valid = lengths.gt(0).to(loss_mat.dtype)
        return (per_seq*valid).sum()/valid.sum().clamp_min(1)
    if reduction == "seq-mean-token-sum-norm":
        return (loss_mat*mask).sum()/mask.shape[-1]
    raise ValueError(f"unknown reduction: {reduction}")


def preservation_kl_loss(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    *, epsilon_pres: float,
    response_mask: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    top_k: int | None = None,
    default_probability: float = 1e-12,
    base_topk_ids: torch.Tensor | None = None,
    base_topk_log_probs: torch.Tensor | None = None,
    base_log_tail: torch.Tensor | None = None,
    reduction: str = "seq-mean-token-mean",
) -> PreservationLossResult:
    projection = project_to_kl_ball(
        policy_logits, base_logits, epsilon=epsilon_pres, response_mask=response_mask,
        selected_token_ids=selected_token_ids, top_k=top_k,
        default_probability=default_probability, base_topk_ids=base_topk_ids,
        base_topk_log_probs=base_topk_log_probs, base_log_tail=base_log_tail,
    )
    policy_log = projection.policy_log_probs
    target_log = projection.projected_log_probs.detach()
    token_loss = torch.sum(policy_log.exp()*(policy_log-target_log), dim=-1)
    token_loss = torch.where(projection.violation_mask, token_loss, torch.zeros_like(token_loss))
    active = torch.ones_like(token_loss) if response_mask is None else response_mask.to(token_loss.dtype)
    return PreservationLossResult(
        loss=aggregate_preservation_loss(token_loss, active, reduction),
        token_loss=token_loss, projection=projection,
    )
