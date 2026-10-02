"""Null-space projector utilities for GRIT Phase 1.

The default update implemented here is the NSPO-theory form:

    grad_W <- grad_W @ P

for protected Linear weights W with shape [out_features, in_features].

Some experiments instead project the realized optimizer delta after AdamW:

    delta_W <- delta_W @ P

This is a step-local update projection, not NSPO's periodic base repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from torch import nn


ModuleFilter = Callable[[str, nn.Module], bool]


@dataclass
class CompactProjector:
    """P = UU^T, or P = I-UU^T when ``complement`` is true."""

    basis: torch.Tensor
    complement: bool
    dimension: int


@dataclass
class ProjectorBuildResult:
    """A projector and its spectrum metadata for one module."""

    compact: CompactProjector
    eigenvalues: torch.Tensor
    threshold_value: float
    nullity: int
    rank: int

    @property
    def projector(self) -> torch.Tensor:
        return dense_projector(self.compact)


@dataclass(frozen=True)
class ProjectorAttachResult:
    """Summary from attaching projector tensors to model modules."""

    attached: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]


def default_module_filter(name: str, module: nn.Module, pattern: str = "mlp") -> bool:
    """Select protected modules.

    The default follows NSPO's public code: protect Linear layers whose module
    path contains "mlp".
    """

    return isinstance(module, nn.Linear) and pattern in name


def collect_activation_covariances(
    model: nn.Module,
    input_batches: Iterable[dict[str, torch.Tensor]],
    *,
    module_filter: ModuleFilter | None = None,
    device: torch.device | str | None = None,
    progress_desc: str | None = None,
    progress_total: int | None = None,
) -> dict[str, torch.Tensor | CompactProjector]:
    """Collect non-central activation covariances X^T X per protected Linear.

    Forward hooks capture each Linear layer's input activations. For a layer with
    input dimension d, this accumulates a [d, d] matrix over all tokens/samples.
    """

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    covariances: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(module_name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if not inputs:
                return
            hidden = inputs[0].detach()
            if hidden.numel() == 0:
                return
            hidden = hidden.reshape(-1, hidden.shape[-1]).to(device=device, dtype=torch.float32)
            cov = hidden.transpose(0, 1).matmul(hidden)
            if module_name in covariances:
                covariances[module_name] = covariances[module_name].to(device) + cov
            else:
                covariances[module_name] = cov

        return hook

    for name, module in model.named_modules():
        if module_filter(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batches = input_batches
            if progress_desc is not None:
                from tqdm.auto import tqdm

                batches = tqdm(input_batches, total=progress_total, desc=progress_desc)
            for batch in batches:
                batch_on_device = {
                    key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
                }
                model(**batch_on_device)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {name: cov.cpu() for name, cov in covariances.items()}


def _unwrap_projector(raw_projector):
    return raw_projector.compact if isinstance(raw_projector, ProjectorBuildResult) else raw_projector


def projector_dimension(projector) -> int:
    projector = _unwrap_projector(projector)
    if isinstance(projector, CompactProjector):
        return projector.dimension
    if torch.is_tensor(projector) and projector.ndim == 2 and projector.shape[0] == projector.shape[1]:
        return projector.shape[0]
    raise TypeError(f"Unsupported projector representation: {type(projector).__name__}")


def project_with_projector(value: torch.Tensor, projector) -> torch.Tensor:
    projector = _unwrap_projector(projector)
    if isinstance(projector, CompactProjector):
        if value.shape[-1] != projector.dimension:
            raise ValueError("Compact projector dimension mismatch; local parameter shards are unsupported")
        basis = projector.basis.to(device=value.device, dtype=value.dtype)
        component = value.matmul(basis).matmul(basis.transpose(0, 1))
        return value-component if projector.complement else component
    projector = projector.to(device=value.device, dtype=value.dtype)
    if value.shape[-1] != projector.shape[0]:
        raise ValueError("Dense projector dimension mismatch; local parameter shards are unsupported")
    return value.matmul(projector)


def dense_projector(projector) -> torch.Tensor:
    projector = _unwrap_projector(projector)
    if torch.is_tensor(projector):
        return projector
    basis = projector.basis
    product = basis.matmul(basis.transpose(0, 1))
    return torch.eye(projector.dimension, dtype=basis.dtype, device=basis.device)-product if projector.complement else product


def load_projectors(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    projector_key: str = "projectors",
) -> dict[str, torch.Tensor]:
    """Load a projector artifact saved by ``scripts/build_projectors.py``.

    The expected artifact shape is ``{"projectors": {module_name: P}}``. A raw
    ``{module_name: P}`` mapping is also accepted for easier tests and ablations.
    """

    payload = torch.load(Path(path), map_location=map_location)
    if isinstance(payload, Mapping) and projector_key in payload:
        payload = payload[projector_key]
    if not isinstance(payload, Mapping):
        raise TypeError(f"projector artifact must contain a mapping, got {type(payload).__name__}")

    projectors: dict[str, torch.Tensor | CompactProjector] = {}
    for name, projector in payload.items():
        if not isinstance(name, str):
            raise TypeError(f"projector name must be str, got {type(name).__name__}")
        projector = _unwrap_projector(projector)
        projector_dimension(projector)
        if isinstance(projector, CompactProjector):
            projector = CompactProjector(projector.basis.detach().cpu(), projector.complement, projector.dimension)
        else:
            projector = projector.detach().cpu()
        projectors[name] = projector
    return projectors


def attach_projectors_to_modules(
    model: nn.Module,
    projectors: Mapping[str, torch.Tensor | ProjectorBuildResult],
    *,
    module_filter: ModuleFilter | None = None,
    attribute_name: str = "grit_projector",
    strict: bool = True,
) -> ProjectorAttachResult:
    """Attach projector tensors to matching Linear modules.

    A training worker can load projectors once
    during model setup. The actual optimizer-facing operation remains gradient
    projection after task-loss backward, not NSPO's periodic weight repair.
    """

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    module_by_name = dict(model.named_modules())
    target_names = [
        name for name, module in module_by_name.items() if isinstance(module, nn.Linear) and module_filter(name, module)
    ]
    attached: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for name in target_names:
        module = module_by_name[name]
        if name not in projectors:
            missing.append(name)
            continue
        projector = _unwrap_projector(projectors[name])
        expected_shape = (module.in_features, module.in_features)
        if projector_dimension(projector) != module.in_features:
            shape_mismatch.append(f"{name}: got dimension {projector_dimension(projector)}, expected {module.in_features}")
            continue
        setattr(module, attribute_name, projector)
        attached.append(name)

    target_name_set = set(target_names)
    unexpected = sorted(name for name in projectors.keys() if name not in target_name_set)
    result = ProjectorAttachResult(
        attached=tuple(attached),
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        shape_mismatch=tuple(shape_mismatch),
    )
    if strict and (result.missing or result.shape_mismatch):
        raise ValueError(
            "failed to attach all GRIT projectors: "
            f"missing={list(result.missing)}, shape_mismatch={list(result.shape_mismatch)}"
        )
    return result


def attached_projectors(
    model: nn.Module,
    *,
    module_filter: ModuleFilter | None = None,
    attribute_name: str = "grit_projector",
) -> dict[str, torch.Tensor]:
    """Return projectors previously attached to protected modules."""

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    projectors: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not module_filter(name, module) or not isinstance(module, nn.Linear):
            continue
        projector = getattr(module, attribute_name, None)
        if projector is not None:
            projector_dimension(projector)
            projectors[name] = projector
    return projectors


def build_projectors_from_covariances(
    covariances: dict[str, torch.Tensor],
    *,
    relative_threshold: float = 5e-4,
    absolute_threshold: float | None = None,
    progress_desc: str | None = None,
) -> dict[str, ProjectorBuildResult]:
    """Build `P = U_null U_null^T` from activation covariance matrices.

    A direction is treated as null if its eigenvalue is below
    `relative_threshold * max_eigenvalue`, unless `absolute_threshold` is given.
    """

    results: dict[str, ProjectorBuildResult] = {}
    items = covariances.items()
    if progress_desc is not None:
        from tqdm.auto import tqdm

        items = tqdm(list(items), desc=progress_desc)
    for name, covariance in items:
        matrix = covariance.to(dtype=torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        max_eval = torch.clamp(eigenvalues.max(), min=0.0)
        if absolute_threshold is None:
            threshold_value = float(relative_threshold * max_eval.item())
        else:
            threshold_value = float(absolute_threshold)

        if max_eval.item() == 0.0 and absolute_threshold is None:
            null_mask = torch.ones_like(eigenvalues, dtype=torch.bool)
        else:
            null_mask = eigenvalues < threshold_value
        nullity = int(null_mask.sum().item())
        rank = matrix.shape[0]-nullity
        if nullity <= rank:
            compact = CompactProjector(eigenvectors[:, null_mask].contiguous().cpu(), False, matrix.shape[0])
        else:
            compact = CompactProjector(eigenvectors[:, ~null_mask].contiguous().cpu(), True, matrix.shape[0])
        results[name] = ProjectorBuildResult(
            compact=compact,
            eigenvalues=eigenvalues.cpu(),
            threshold_value=threshold_value,
            nullity=nullity,
            rank=rank,
        )

    return results


def apply_gradient_projection(
    model: nn.Module,
    projectors: dict[str, torch.Tensor | ProjectorBuildResult],
    *,
    module_filter: ModuleFilter | None = None,
) -> dict[str, float]:
    """Apply `grad_W <- grad_W @ P` to protected Linear weights.

    Returns per-layer before/after gradient norms for logging.
    """

    if module_filter is None:
        module_filter = lambda name, module: default_module_filter(name, module)

    metrics: dict[str, float] = {}
    for name, module in model.named_modules():
        if not module_filter(name, module):
            continue
        if name not in projectors:
            continue
        if not isinstance(module, nn.Linear) or module.weight.grad is None:
            continue

        raw_projector = projectors[name]
        before = module.weight.grad.detach().float().norm()
        module.weight.grad.copy_(project_with_projector(module.weight.grad, raw_projector))
        after = module.weight.grad.detach().float().norm()

        metrics[f"{name}.grad_norm_before"] = float(before.item())
        metrics[f"{name}.grad_norm_after"] = float(after.item())

    return metrics


def apply_attached_gradient_projection(
    model: nn.Module,
    *,
    module_filter: ModuleFilter | None = None,
    attribute_name: str = "grit_projector",
) -> dict[str, float]:
    """Apply ``grad_W <- grad_W @ P`` using projectors attached to modules."""

    projectors = attached_projectors(
        model,
        module_filter=module_filter,
        attribute_name=attribute_name,
    )
    return apply_gradient_projection(model, projectors, module_filter=module_filter)


def projector_diagnostics(projector: torch.Tensor | CompactProjector) -> dict[str, float]:
    """Return symmetry/idempotence diagnostics for an orthogonal projector."""

    projector = _unwrap_projector(projector)
    if isinstance(projector, CompactProjector):
        basis = projector.basis.float()
        gram = basis.transpose(0, 1).matmul(basis)
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        orthogonality_error = torch.linalg.vector_norm(gram-identity) if gram.numel() else torch.tensor(0.0)
        nonzero = projector.dimension > basis.shape[1] if projector.complement else basis.shape[1] > 0
        return {
            "symmetry_error": 0.0,
            "idempotence_error": float(orthogonality_error),
            "spectral_norm": float(nonzero),
        }

    p = projector.float()
    symmetry_error = (p - p.transpose(0, 1)).norm()
    idempotence_error = (p.matmul(p) - p).norm()
    spectral_norm = torch.linalg.matrix_norm(p, ord=2)
    return {
        "symmetry_error": float(symmetry_error.item()),
        "idempotence_error": float(idempotence_error.item()),
        "spectral_norm": float(spectral_norm.item()),
    }
