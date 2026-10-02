"""Total GRIT gradient assembly for Phase 5.

This module writes the optimizer-facing gradient:

    final_grad = projected_task_grad + lambda_pres * v

or, with curvature enabled:

    final_grad = projected_task_grad + lambda_pres * (v - lr * H Q v)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from grit.curvature import (
    CurvatureCorrectionResult,
    project_vector_with_module_projectors,
    trainable_named_parameters,
)
from grit.predictor import PredictorStepInfo, temporary_predictor_direction_step
from grit.predictor import temporary_predictor_step
from grit.projection import CompactProjector, ProjectorBuildResult, default_module_filter
from grit.preservation_loss import PreservationLossResult


ModuleFilter = Callable[[str, nn.Module], bool]
ParameterFilter = Callable[[str, nn.Parameter], bool]
NamedParameter = tuple[str, nn.Parameter]
GradientMap = Mapping[str, torch.Tensor | None]
PreservationLossFn = Callable[[], torch.Tensor | PreservationLossResult]
TaskDirectionFn = Callable[
    [Sequence[NamedParameter], Mapping[str, torch.Tensor]],
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]],
]


@dataclass(frozen=True)
class GritUpdateConfig:
    """Configuration for one GRIT optimizer-gradient assembly."""

    learning_rate: float = 1e-6
    lambda_pres: float = 1.0
    use_curvature: bool = False
    curvature_mode: str = "central_fd"
    fd_radius: float = 0.05
    fd_normalize_direction: bool = True
    hvp_last_linear_layers: int = 0
    missing_projector: str = "identity"
    projector_relaxation: float = 0.0
    zero_grad_before_write: bool = True


@dataclass(frozen=True)
class GritUpdateResult:
    """Named gradients and scalar diagnostics from one GRIT update."""

    task_gradients: dict[str, torch.Tensor]
    final_gradients: dict[str, torch.Tensor]
    projected_task_gradients: dict[str, torch.Tensor]
    preservation_gradients: dict[str, torch.Tensor]
    preservation_correction: dict[str, torch.Tensor]
    task_directions: dict[str, torch.Tensor]
    projected_task_directions: dict[str, torch.Tensor]
    preservation_loss: torch.Tensor
    predictor: PredictorStepInfo
    curvature: CurvatureCorrectionResult | None
    metrics: dict[str, float]


def _zero_like_parameter(parameter: nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(parameter, memory_format=torch.preserve_format)


def _squared_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        norm = torch.linalg.vector_norm(tensor.detach())
        total += float(norm.item()) ** 2
    return total


def _coerce_config(config: GritUpdateConfig | None, **overrides) -> GritUpdateConfig:
    if config is None:
        config = GritUpdateConfig()
    values = {
        "learning_rate": config.learning_rate,
        "lambda_pres": config.lambda_pres,
        "use_curvature": config.use_curvature,
        "curvature_mode": config.curvature_mode,
        "fd_radius": config.fd_radius,
        "fd_normalize_direction": config.fd_normalize_direction,
        "hvp_last_linear_layers": config.hvp_last_linear_layers,
        "missing_projector": config.missing_projector,
        "projector_relaxation": config.projector_relaxation,
        "zero_grad_before_write": config.zero_grad_before_write,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    return GritUpdateConfig(**values)


def _autograd_gradient_map(
    loss: torch.Tensor,
    parameters: Sequence[NamedParameter],
    *,
    retain_graph: bool,
    create_graph: bool = False,
    allow_unused: bool = True,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    tensors = [parameter for _, parameter in parameters]
    gradients = torch.autograd.grad(
        loss,
        tensors,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )
    gradient_map: dict[str, torch.Tensor] = {}
    for (name, parameter), grad in zip(parameters, gradients, strict=True):
        if grad is None:
            gradient_map[name] = _zero_like_parameter(parameter)
        elif detach:
            gradient_map[name] = grad.detach().clone()
        elif isinstance(raw_projector, CompactProjector):
            basis_rank = raw_projector.basis.shape[1]
            rank = float(raw_projector.dimension-basis_rank if raw_projector.complement else basis_rank)
            nullity = float(raw_projector.dimension-rank)
        else:
            gradient_map[name] = grad.clone()
    return gradient_map


def _write_gradients(
    parameters: Sequence[NamedParameter],
    gradients: Mapping[str, torch.Tensor],
    *,
    zero_grad_before_write: bool,
) -> None:
    for name, parameter in parameters:
        grad = gradients[name].to(device=parameter.device, dtype=parameter.dtype)
        if zero_grad_before_write or parameter.grad is None:
            parameter.grad = grad.detach().clone()
        else:
            parameter.grad.copy_(grad)


def _projector_metrics(
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
) -> dict[str, float]:
    ranks: list[float] = []
    nullities: list[float] = []
    metrics: dict[str, float] = {}
    for name, raw_projector in projectors.items():
        if isinstance(raw_projector, ProjectorBuildResult):
            rank = float(raw_projector.rank)
            nullity = float(raw_projector.nullity)
        else:
            projector = raw_projector.detach().float()
            rank = float(torch.trace(projector).round().item())
            nullity = float(projector.shape[-1] - rank)
        metrics[f"grit/projector/{name}/rank"] = rank
        metrics[f"grit/projector/{name}/nullity"] = nullity
        ranks.append(rank)
        nullities.append(nullity)

    if ranks:
        metrics["grit/projector/rank_mean"] = sum(ranks) / len(ranks)
        metrics["grit/projector/nullity_mean"] = sum(nullities) / len(nullities)
    else:
        metrics["grit/projector/rank_mean"] = 0.0
        metrics["grit/projector/nullity_mean"] = 0.0
    return metrics


def _projected_module_count(
    model: nn.Module,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    parameters: Sequence[NamedParameter],
    module_filter: ModuleFilter,
    projected_task_gradients: Mapping[str, torch.Tensor],
) -> float:
    parameter_names = {name for name, _parameter in parameters}
    count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parameter_name = f"{module_name}.weight" if module_name else "weight"
        if (
            module_filter(module_name, module)
            and module_name in projectors
            and parameter_name in parameter_names
            and parameter_name in projected_task_gradients
        ):
            count += 1
    return float(count)


def _last_projected_linear_weight_parameters(
    model: nn.Module,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    parameters: Sequence[NamedParameter],
    module_filter: ModuleFilter,
    count: int,
) -> list[NamedParameter]:
    if count <= 0:
        return list(parameters)

    parameter_by_name = dict(parameters)
    selected_names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parameter_name = f"{module_name}.weight" if module_name else "weight"
        if (
            module_filter(module_name, module)
            and module_name in projectors
            and parameter_name in parameter_by_name
        ):
            selected_names.append(parameter_name)

    selected_set = set(selected_names[-count:])
    return [
        (name, parameter)
        for name, parameter in parameters
        if name in selected_set
    ]


def _preservation_loss_and_metrics(
    result: torch.Tensor | PreservationLossResult,
) -> tuple[torch.Tensor, dict[str, float]]:
    if torch.is_tensor(result):
        return result, {}

    projection = result.projection
    active_mask = projection.active_mask
    violation_mask = projection.violation_mask
    active_count = active_mask.sum().clamp_min(1)
    metrics = {
        "grit/preservation_loss": float(result.loss.detach().float().item()),
        "grit/kl_violation_fraction": float(
            (violation_mask & active_mask).to(torch.float32).sum().div(active_count).item()
        ),
        "grit/kl_raw_mean": float(projection.token_kl.detach().float().mean().item()),
        "grit/kl_projected_mean": float(projection.projected_kl.detach().float().mean().item()),
    }
    return result.loss, metrics


def assemble_grit_update(
    model: nn.Module,
    task_loss: torch.Tensor,
    preservation_loss_fn: PreservationLossFn | None,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    *,
    task_loss_fn: Callable[[], torch.Tensor] | None = None,
    task_direction_fn: TaskDirectionFn | None = None,
    config: GritUpdateConfig | None = None,
    learning_rate: float | None = None,
    lambda_pres: float | None = None,
    use_curvature: bool | None = None,
    curvature_mode: str | None = None,
    fd_radius: float | None = None,
    fd_normalize_direction: bool | None = None,
    parameters: Sequence[NamedParameter] | None = None,
    parameter_filter: ParameterFilter | None = None,
    module_filter: ModuleFilter | None = None,
    missing_projector: str | None = None,
    projector_relaxation: float | None = None,
    hvp_last_linear_layers: int | None = None,
    zero_grad_before_write: bool | None = None,
) -> GritUpdateResult:
    """Assemble and write the final GRIT gradient to ``parameter.grad``.

    ``task_loss`` is the scalar minimization loss for the main RL/objective
    batch. When ``lambda_pres > 0``, ``preservation_loss_fn`` is called inside a
    temporary ``theta_tilde = theta - lr * projected_task_grad`` context and
    should return either a scalar tensor or ``PreservationLossResult``.
    """

    resolved = _coerce_config(
        config,
        learning_rate=learning_rate,
        lambda_pres=lambda_pres,
        use_curvature=use_curvature,
        curvature_mode=curvature_mode,
        fd_radius=fd_radius,
        fd_normalize_direction=fd_normalize_direction,
        missing_projector=missing_projector,
        projector_relaxation=projector_relaxation,
        hvp_last_linear_layers=hvp_last_linear_layers,
        zero_grad_before_write=zero_grad_before_write,
    )
    if resolved.learning_rate < 0:
        raise ValueError(
            f"learning_rate must be non-negative, got {resolved.learning_rate}"
        )
    if task_direction_fn is not None and resolved.use_curvature:
        raise ValueError("Use grit.step.grit_step for optimizer-predictor curvature; this helper uses u=Q v")
    if resolved.lambda_pres < 0:
        raise ValueError(f"lambda_pres must be non-negative, got {resolved.lambda_pres}")
    if resolved.curvature_mode != "central_fd":
        raise ValueError(
            "curvature_mode must be 'central_fd'; exact HVP is kept only for toy checks, "
            f"got {resolved.curvature_mode!r}"
        )
    if resolved.fd_radius <= 0:
        raise ValueError(f"fd_radius must be positive, got {resolved.fd_radius}")
    if resolved.missing_projector not in {"identity", "zero"}:
        raise ValueError(
            f"missing_projector must be 'identity' or 'zero', got {resolved.missing_projector!r}"
        )
    if not 0.0 <= resolved.projector_relaxation <= 1.0:
        raise ValueError("projector_relaxation must be in [0, 1]")
    if resolved.hvp_last_linear_layers < 0:
        raise ValueError(
            f"hvp_last_linear_layers must be non-negative, got {resolved.hvp_last_linear_layers}"
        )
    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)
    if parameters is None:
        parameters = trainable_named_parameters(model, parameter_filter)

    task_gradients = _autograd_gradient_map(
        task_loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        detach=True,
    )
    projected_task_gradients = project_vector_with_module_projectors(
        model,
        task_gradients,
        projectors,
        parameters=parameters,
        module_filter=module_filter,
        missing=resolved.missing_projector,
        relaxation=resolved.projector_relaxation,
    )
    if task_direction_fn is None:
        task_directions = {}
        projected_task_directions = {}
        task_direction_metrics = {}
    else:
        task_directions, projected_task_directions, task_direction_metrics = (
            task_direction_fn(parameters, task_gradients)
        )

    preservation_metrics: dict[str, float] = {}
    if resolved.lambda_pres == 0.0:
        preservation_loss = task_loss.detach().new_zeros(())
        preservation_gradients = {
            name: _zero_like_parameter(parameter) for name, parameter in parameters
        }
        predictor_info = PredictorStepInfo(
            updated_parameters=0,
            update_norm=0.0,
            max_update_abs=0.0,
        )
    else:
        if preservation_loss_fn is None:
            raise ValueError("preservation_loss_fn is required when lambda_pres > 0")
        predictor_context = (
            temporary_predictor_direction_step(
                model,
                learning_rate=resolved.learning_rate,
                directions=projected_task_directions,
                parameter_filter=parameter_filter,
                module_filter=module_filter,
            )
            if task_direction_fn is not None
            else temporary_predictor_step(
                model,
                learning_rate=resolved.learning_rate,
                gradients=projected_task_gradients,
                parameter_filter=parameter_filter,
                module_filter=module_filter,
                preserve_autograd_graph=False,
            )
        )
        with predictor_context as predictor_info:
            preservation_result = preservation_loss_fn()
            preservation_loss, preservation_metrics = _preservation_loss_and_metrics(preservation_result)
            preservation_gradients = _autograd_gradient_map(
                preservation_loss,
                parameters,
                retain_graph=False,
            )

    curvature_result: CurvatureCorrectionResult | None = None
    if resolved.use_curvature and resolved.lambda_pres > 0.0:
        hvp_parameters = _last_projected_linear_weight_parameters(
            model,
            projectors,
            parameters,
            module_filter,
            resolved.hvp_last_linear_layers,
        )
        if task_loss_fn is None:
            raise ValueError("task_loss_fn is required when central difference curvature is enabled")
        from grit.curvature import finite_difference_curvature_corrected_preservation_gradients

        curvature_result = finite_difference_curvature_corrected_preservation_gradients(
            model,
            task_loss_fn,
            task_gradients,
            preservation_gradients,
            projectors,
            learning_rate=resolved.learning_rate,
            rho=resolved.fd_radius,
            normalize_direction=resolved.fd_normalize_direction,
            parameters=parameters,
            hvp_parameters=hvp_parameters,
            module_filter=module_filter,
            missing_projector=resolved.missing_projector,
            projector_relaxation=resolved.projector_relaxation,
        )
        preservation_correction = curvature_result.gradients
    else:
        hvp_parameters = []
        preservation_correction = preservation_gradients

    final_gradients = {
        name: projected_task_gradients[name]
        + preservation_correction[name].to(projected_task_gradients[name]).mul(resolved.lambda_pres)
        for name, _parameter in parameters
    }
    _write_gradients(
        parameters,
        final_gradients,
        zero_grad_before_write=resolved.zero_grad_before_write,
    )

    skipped_hvp = True if curvature_result is None else curvature_result.skipped_hvp
    metrics = {
        "grit/learning_rate": float(resolved.learning_rate),
        "grit/lambda_pres": float(resolved.lambda_pres),
        "grit/use_curvature": float(resolved.use_curvature),
        "grit/curvature_mode_central_fd": float(resolved.curvature_mode == "central_fd"),
        "grit/fd_radius": float(resolved.fd_radius),
        "grit/fd_normalize_direction": float(resolved.fd_normalize_direction),
        "grit/hvp_last_linear_layers": float(resolved.hvp_last_linear_layers),
        "grit/projector_relaxation": float(resolved.projector_relaxation),
        "grit/hvp_parameter_count": float(len(hvp_parameters)),
        "grit/hvp_skipped": float(skipped_hvp),
        "grit/task_grad_norm": _squared_norm(list(task_gradients.values())) ** 0.5,
        "grit/projected_task_grad_norm": _squared_norm(list(projected_task_gradients.values())) ** 0.5,
        "grit/preservation_grad_norm": _squared_norm(list(preservation_gradients.values())) ** 0.5,
        "grit/preservation_correction_norm": _squared_norm(list(preservation_correction.values())) ** 0.5,
        "grit/corrected_preservation_grad_norm": _squared_norm(list(preservation_correction.values())) ** 0.5,
        "grit/final_grad_norm": _squared_norm(list(final_gradients.values())) ** 0.5,
        "grit/projected_module_count": _projected_module_count(
            model,
            projectors,
            parameters,
            module_filter,
            projected_task_gradients,
        ),
        "grit/predictor_updated_parameters": float(predictor_info.updated_parameters),
        "grit/predictor_update_norm": float(predictor_info.update_norm),
        "grit/predictor_max_update_abs": float(predictor_info.max_update_abs),
    }
    metrics.update(task_direction_metrics)
    metrics.update(_projector_metrics(projectors))
    metrics.update(preservation_metrics)
    if "grit/preservation_loss" not in metrics:
        metrics["grit/preservation_loss"] = float(preservation_loss.detach().float().item())
    if curvature_result is not None:
        metrics["grit/hvp_norm"] = float(curvature_result.hvp_norm)
        metrics["grit/projected_vector_norm"] = float(curvature_result.projected_vector_norm)
        metrics["grit/hvp_projected_vector_norm"] = float(curvature_result.projected_vector_norm)
    else:
        metrics["grit/hvp_norm"] = 0.0
        metrics["grit/projected_vector_norm"] = 0.0
        metrics["grit/hvp_projected_vector_norm"] = 0.0

    return GritUpdateResult(
        task_gradients=task_gradients,
        final_gradients=final_gradients,
        projected_task_gradients=projected_task_gradients,
        preservation_gradients=preservation_gradients,
        preservation_correction=preservation_correction,
        task_directions=task_directions,
        projected_task_directions=projected_task_directions,
        preservation_loss=preservation_loss.detach(),
        predictor=predictor_info,
        curvature=curvature_result,
        metrics=metrics,
    )
