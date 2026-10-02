#!/usr/bin/env python3
"""Sanity check for the NSPO-style GRPO safety task objective."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.data import parse_safety_reward
from grit.grpo import group_advantages, grpo_loss, token_log_probs


def grpo_safety_task_loss(model, batch, old_log_probs, rewards, *, group_size, clip_ratio):
    values, mask = token_log_probs(model, batch)
    advantages = group_advantages(rewards, group_size)
    loss = grpo_loss(values, old_log_probs, advantages, mask, clip_ratio)
    return loss, {"reward_mean": float(rewards.mean()), "unsafe_fraction": float(rewards.eq(-1).float().mean()),
                  "adv_abs_mean": float(advantages.abs().mean()), "ratio_mean": float((values-old_log_probs).exp().mean().detach())}


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(9, 4)
        self.lm_head = nn.Linear(4, 9, bias=False)

    def forward(self, input_ids, attention_mask=None):
        return type("Output", (), {"logits": self.lm_head(self.embed(input_ids))})()


def main() -> None:
    torch.manual_seed(123)
    model = ToyLM()
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 2, 3, 4],
                [1, 2, 5, 6],
                [7, 2, 3, 4],
                [7, 2, 5, 6],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(4, 4, dtype=torch.long),
        "labels": torch.tensor(
            [
                [-100, -100, 3, 4],
                [-100, -100, 5, 6],
                [-100, -100, 3, 4],
                [-100, -100, 5, 6],
            ],
            dtype=torch.long,
        ),
    }
    old_log_probs = token_log_probs(model, batch)[0].detach()
    rewards = torch.tensor([0.0, -1.0, 0.0, 0.0])

    loss, metrics = grpo_safety_task_loss(
        model,
        batch,
        old_log_probs,
        rewards,
        group_size=2,
        clip_ratio=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["reward_mean"] == -0.25
    assert metrics["unsafe_fraction"] == 0.25
    assert metrics["adv_abs_mean"] > 0.0
    assert parse_safety_reward("Safety: Unsafe\nCategories: Non-viol") == -1.0
    assert parse_safety_reward("Safety: Safe") == 0.0
    assert parse_safety_reward("unsafe") == -1.0
    assert parse_safety_reward("safe") == 0.0
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()

    print(f"loss={loss.item():.8f}")
    print(f"reward_mean={metrics['reward_mean']:.8f}")
    print(f"unsafe_fraction={metrics['unsafe_fraction']:.8f}")
    print(f"adv_abs_mean={metrics['adv_abs_mean']:.8f}")
    print(f"ratio_mean={metrics['ratio_mean']:.8f}")
    print("grpo_safety_objective=True")
    print("task_kl_removed=True")
    print("qwen_guard_label_parser=True")


if __name__ == "__main__":
    main()
