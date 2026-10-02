#!/usr/bin/env python3
"""Smoke check for Phase 4 on the GRPO safety update path."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.preservation_loss import preservation_kl_loss
from grit.update import GritUpdateConfig, assemble_grit_update
from grit.grpo import group_advantages, grpo_loss, token_log_probs


def grpo_safety_task_loss(model, batch, old_log_probs, rewards, *, group_size, clip_ratio):
    values, mask = token_log_probs(model, batch)
    advantages = group_advantages(rewards, group_size)
    loss = grpo_loss(values, old_log_probs, advantages, mask, clip_ratio)
    return loss, {"reward_mean": float(rewards.mean()), "unsafe_fraction": float(rewards.eq(-1).float().mean()),
                  "adv_abs_mean": float(advantages.abs().mean()), "ratio_mean": float((values-old_log_probs).exp().mean().detach())}


class TinyCausalPolicy(nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 5) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        hidden = torch.tanh(self.mlp(self.embed(input_ids)))
        return SimpleNamespace(logits=self.lm_head(hidden))


def clone_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def assert_parameters_restored(model: nn.Module, reference: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), reference[name], atol=1e-10, rtol=1e-10)


def main() -> None:
    torch.manual_seed(71)
    torch.set_default_dtype(torch.float64)

    model = TinyCausalPolicy()
    base_model = copy.deepcopy(model)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        model.mlp.weight.add_(0.15 * torch.randn_like(model.mlp.weight))
        model.lm_head.weight.add_(0.05 * torch.randn_like(model.lm_head.weight))

    baseline_parameters = clone_named_parameters(model)
    rollout_batch = {
        "input_ids": torch.tensor(
            [
                [1, 3, 4, 5, 2],
                [1, 3, 6, 7, 2],
                [1, 8, 4, 9, 2],
                [1, 8, 6, 10, 2],
            ]
        ),
        "attention_mask": torch.ones(4, 5, dtype=torch.long),
        "labels": torch.tensor(
            [
                [-100, -100, 4, 5, 2],
                [-100, -100, 6, 7, 2],
                [-100, -100, 4, 9, 2],
                [-100, -100, 6, 10, 2],
            ]
        ),
    }
    old_log_probs = token_log_probs(model, rollout_batch)[0].detach()
    rewards = torch.tensor([0.0, -1.0, -1.0, 0.0])
    task_loss, grpo_metrics = grpo_safety_task_loss(
        model,
        rollout_batch,
        old_log_probs,
        rewards,
        group_size=2,
        clip_ratio=0.2,
    )

    def task_loss_fn():
        loss, _metrics = grpo_safety_task_loss(
            model,
            rollout_batch,
            old_log_probs,
            rewards,
            group_size=2,
            clip_ratio=0.2,
        )
        return loss

    pres_input_ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 2]])
    pres_attention_mask = torch.ones_like(pres_input_ids)
    selected_ids = pres_input_ids[:, 1:].contiguous()
    response_mask = pres_attention_mask[:, 1:].bool().contiguous()

    def preservation_loss_fn():
        policy_logits = model(
            input_ids=pres_input_ids,
            attention_mask=pres_attention_mask,
        ).logits[:, :-1, :]
        with torch.no_grad():
            base_logits = base_model(
                input_ids=pres_input_ids,
                attention_mask=pres_attention_mask,
            ).logits[:, :-1, :]
        return preservation_kl_loss(
            policy_logits,
            base_logits,
            epsilon_pres=1e-5,
            response_mask=response_mask,
            selected_token_ids=selected_ids,
            top_k=4,
            default_probability=1e-6,
        )

    projector = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))
    all_linear = lambda _name, module: isinstance(module, nn.Linear)
    result = assemble_grit_update(
        model,
        task_loss,
        preservation_loss_fn,
        {"mlp": projector, "lm_head": projector},
        task_loss_fn=task_loss_fn,
        config=GritUpdateConfig(
            learning_rate=0.1,
            lambda_pres=0.5,
            use_curvature=True,
            hvp_last_linear_layers=1,
            missing_projector="identity",
        ),
        module_filter=all_linear,
    )

    assert result.curvature is not None
    assert not result.curvature.skipped_hvp
    assert result.metrics["grit/hvp_skipped"] == 0.0
    assert result.metrics["grit/hvp_last_linear_layers"] == 1.0
    assert result.metrics["grit/hvp_parameter_count"] == 1.0
    assert result.metrics["grit/projected_vector_norm"] > 0.0
    assert result.metrics["grit/hvp_norm"] > 0.0
    assert result.metrics["grit/corrected_preservation_grad_norm"] > 0.0
    assert result.metrics["grit/projected_module_count"] == 2.0
    torch.testing.assert_close(
        result.curvature.hvp["mlp.weight"],
        torch.zeros_like(result.curvature.hvp["mlp.weight"]),
    )
    assert result.curvature.hvp["lm_head.weight"].norm().item() > 0.0
    torch.testing.assert_close(
        result.preservation_correction["mlp.weight"],
        result.preservation_gradients["mlp.weight"],
    )
    assert grpo_metrics["adv_abs_mean"] > 0.0
    assert_parameters_restored(model, baseline_parameters)

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, result.final_gradients[name])

    print("grpo_safety_curvature_update=True")
    print(f"reward_mean={grpo_metrics['reward_mean']:.6f}")
    print(f"adv_abs_mean={grpo_metrics['adv_abs_mean']:.6f}")
    print(f"hvp_skipped={bool(result.metrics['grit/hvp_skipped'])}")
    print(f"projected_vector_norm={result.metrics['grit/projected_vector_norm']:.8f}")
    print(f"hvp_norm={result.metrics['grit/hvp_norm']:.8f}")
    print(
        "corrected_preservation_grad_norm="
        f"{result.metrics['grit/corrected_preservation_grad_norm']:.8f}"
    )

    sam_task_loss = task_loss_fn()
    sam_result = assemble_grit_update(
        model,
        sam_task_loss,
        preservation_loss_fn,
        {"mlp": projector, "lm_head": projector},
        task_loss_fn=task_loss_fn,
        config=GritUpdateConfig(
            learning_rate=0.1,
            lambda_pres=0.5,
            use_curvature=True,
            curvature_mode="central_fd",
            fd_radius=1e-3,
            hvp_last_linear_layers=1,
            missing_projector="identity",
        ),
        module_filter=all_linear,
    )
    assert sam_result.curvature is not None
    assert not sam_result.curvature.skipped_hvp
    assert sam_result.metrics["grit/curvature_mode_central_fd"] == 1.0
    assert sam_result.metrics["grit/hvp_norm"] > 0.0
    assert_parameters_restored(model, baseline_parameters)
    print("central_fd_curvature_update=True")
    print(f"sam_hvp_norm={sam_result.metrics['grit/hvp_norm']:.8f}")


if __name__ == "__main__":
    main()
