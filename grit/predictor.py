"""Temporary predictor-weight utilities for GRIT Phase 2.

GRIT evaluates preservation behavior at the point the model is about to occupy
after the projected task update:

    theta_tilde = theta - learning_rate * projected_grad

This module applies that update in-place under ``torch.no_grad()``, stores
the original weights, and restores the original parameters when the context exits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


ModuleFilter = Callable[[str, nn.Module], bool]
ParameterFilter = Callable[[str, nn.Parameter], bool]
GradientMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class PredictorStepInfo:
    """Diagnostics for one temporary predictor update."""

    updated_parameters: int
    update_norm: float
    max_update_abs: float


def _iter_trainable_parameters(
    model: nn.Module,
    parameter_filter: ParameterFilter | None,
) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_filter is not None and not parameter_filter(name, parameter):
            continue
        yield name, parameter


def linear_weight_parameter_names(
    model: nn.Module,
    module_filter: ModuleFilter | None = None,
) -> set[str]:
    """Return parameter names for Linear weights eligible for GRIT projection."""

    names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module_filter is not None and not module_filter(module_name, module):
            continue
        parameter_name = f"{module_name}.weight" if module_name else "weight"
        names.add(parameter_name)
    return names


@contextmanager
def temporary_predictor_step(
    model: nn.Module,
    *,
    learning_rate: float,
    gradients: GradientMap | None = None,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
    preserve_autograd_graph: bool = False,
) -> Iterator[PredictorStepInfo]:
    """Temporarily move ``model`` to ``theta_tilde`` and restore on exit.

    Args:
        model: Model whose parameters should be temporarily updated.
        learning_rate: The single GRIT step size. The temporary delta is
            ``-learning_rate * grad``.
        gradients: Optional mapping from parameter name to projected gradient.
            If omitted, each selected parameter's current ``.grad`` is used.
        parameter_filter: Optional predicate to restrict updated parameters. If
            omitted, only ``nn.Linear`` weight parameters are updated, matching
            the Phase 1 projector surface.
        module_filter: Optional predicate to restrict Linear modules when
            ``parameter_filter`` is omitted.
        preserve_autograd_graph: If true, apply and restore the temporary delta
            without incrementing parameter version counters. This is needed by
            Phase 4 because the task-loss graph must remain valid for the later
            Hessian-vector product after ``theta`` has been restored.

    Yields:
        A small diagnostics object describing the temporary update.

    Notes:
        This function never calls ``optimizer.step()`` and never mutates
        gradients. It stores the selected original weights for exact restoration.
    """

    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")

    if parameter_filter is None:
        linear_weight_names = linear_weight_parameter_names(model, module_filter)

        def parameter_filter(name: str, _parameter: nn.Parameter) -> bool:
            return name in linear_weight_names

    deltas: list[tuple[nn.Parameter, torch.Tensor]] = []
    squared_update_norm = 0.0
    max_update_abs = 0.0

    with torch.no_grad():
        for name, parameter in _iter_trainable_parameters(model, parameter_filter):
            if gradients is None:
                grad = parameter.grad
            else:
                grad = gradients.get(name)
            if grad is None:
                continue
            if grad.shape != parameter.shape:
                raise ValueError(
                    f"gradient for {name} has shape {tuple(grad.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )

            delta = grad.detach().to(device=parameter.device, dtype=parameter.dtype).mul(
                -learning_rate
            )
            original = parameter.detach().clone()
            if preserve_autograd_graph:
                parameter.data.add_(delta)
            else:
                parameter.add_(delta)
            deltas.append((parameter, original))

            delta_float = delta.float()
            squared_update_norm += float(delta_float.square().sum().item())
            max_update_abs = max(max_update_abs, float(delta_float.abs().max().item()))

    info = PredictorStepInfo(
        updated_parameters=len(deltas),
        update_norm=squared_update_norm**0.5,
        max_update_abs=max_update_abs,
    )

    try:
        yield info
    finally:
        with torch.no_grad():
            for parameter, delta in reversed(deltas):
                if preserve_autograd_graph:
                    parameter.data.copy_(delta)
                else:
                    parameter.copy_(delta)


@contextmanager
def temporary_predictor_direction_step(
    model: nn.Module,
    *,
    learning_rate: float,
    directions: GradientMap,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
) -> Iterator[PredictorStepInfo]:
    """Temporarily apply signed optimizer directions and restore on exit.

    The temporary point is ``theta + learning_rate * direction``.
    """

    if learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")

    applied: list[tuple[nn.Parameter, torch.Tensor]] = []
    squared_update_norm = 0.0
    max_update_abs = 0.0

    with torch.no_grad():
        for name, parameter in _iter_trainable_parameters(model, parameter_filter):
            direction = directions.get(name)
            if direction is None:
                continue
            if direction.shape != parameter.shape:
                raise ValueError(
                    f"direction for {name} has shape {tuple(direction.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )

            update = direction.detach().to(
                device=parameter.device, dtype=parameter.dtype
            ).mul(learning_rate)
            original = parameter.detach().clone()
            parameter.add_(update)
            applied.append((parameter, original))

            update_float = update.float()
            squared_update_norm += float(update_float.square().sum().item())
            max_update_abs = max(max_update_abs, float(update_float.abs().max().item()))

    info = PredictorStepInfo(
        updated_parameters=len(applied),
        update_norm=squared_update_norm**0.5,
        max_update_abs=max_update_abs,
    )

    try:
        yield info
    finally:
        with torch.no_grad():
            for parameter, original in reversed(applied):
                parameter.copy_(original)


def forward_with_predictor_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    learning_rate: float,
    gradients: GradientMap | None = None,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
    preserve_autograd_graph: bool = False,
):
    """Run one forward pass at ``theta_tilde`` and restore ``theta``."""

    with temporary_predictor_step(
        model,
        learning_rate=learning_rate,
        gradients=gradients,
        parameter_filter=parameter_filter,
        module_filter=module_filter,
        preserve_autograd_graph=preserve_autograd_graph,
    ):
        return model(**batch)
