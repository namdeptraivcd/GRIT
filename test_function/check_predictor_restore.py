#!/usr/bin/env python3
"""Toy sanity check for GRIT Phase 2 predictor weights."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.predictor import temporary_predictor_step, linear_weight_parameter_names


def clone_current_gradients(model):
    names = linear_weight_parameter_names(model)
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if n in names and p.grad is not None}


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def clone_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def assert_parameters_restored(
    model: nn.Module,
    reference: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), reference[name], atol=atol, rtol=rtol)


def assert_optimizer_state_unchanged(
    before: dict,
    after: dict,
) -> None:
    before_state = before["state"]
    after_state = after["state"]
    assert before_state.keys() == after_state.keys()
    for param_index, values_before in before_state.items():
        values_after = after_state[param_index]
        assert values_before.keys() == values_after.keys()
        for key, value_before in values_before.items():
            value_after = values_after[key]
            if torch.is_tensor(value_before):
                torch.testing.assert_close(value_after, value_before, atol=0.0, rtol=0.0)
            else:
                assert value_after == value_before


def main() -> None:
    torch.manual_seed(11)

    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {"x": torch.randn(6, 4)}

    loss = model(**batch).square().mean()
    loss.backward()

    # Populate optimizer state once, then create fresh gradients for the
    # predictor path. The predictor helper itself must not step the optimizer.
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(**batch).square().mean().backward()

    baseline_parameters = clone_named_parameters(model)
    baseline_logits = model(**batch).detach().clone()
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())
    projected_grads = clone_current_gradients(model)

    with temporary_predictor_step(model, learning_rate=0.3, gradients=projected_grads) as info:
        predictor_logits = model(**batch).detach().clone()
        assert info.updated_parameters == len(projected_grads)
        assert info.update_norm > 0.0
        assert not torch.allclose(predictor_logits, baseline_logits)

    restored_logits = model(**batch).detach()
    assert_parameters_restored(model, baseline_parameters, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(restored_logits, baseline_logits, atol=1e-7, rtol=1e-7)
    assert_optimizer_state_unchanged(optimizer_state_before, optimizer.state_dict())

    print(f"updated_parameters={info.updated_parameters}")
    print(f"update_norm={info.update_norm:.6f}")
    print(f"max_update_abs={info.max_update_abs:.6f}")
    print(f"logit_delta={(predictor_logits - baseline_logits).norm().item():.6f}")
    print("restore_ok=True")
    print("optimizer_state_unchanged=True")


if __name__ == "__main__":
    main()
