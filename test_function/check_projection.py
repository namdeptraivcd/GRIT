#!/usr/bin/env python3
"""Toy sanity check for GRIT Phase 1 gradient projection."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.projection import (
    apply_attached_gradient_projection,
    apply_gradient_projection,
    attach_projectors_to_modules,
    build_projectors_from_covariances,
    load_projectors,
    projector_diagnostics,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(4, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).square().mean()


def main() -> None:
    torch.manual_seed(7)

    # Rank-2 covariance in R^4, so the null space should have dimension 2.
    basis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [2.0, -1.0, 0.0, 0.0],
        ]
    )
    covariance = basis.transpose(0, 1).matmul(basis)
    results = build_projectors_from_covariances({"mlp": covariance}, relative_threshold=5e-4)
    projector = results["mlp"].projector
    compact = results["mlp"].compact

    model = ToyModel()
    loss = model(torch.randn(8, 4))
    loss.backward()

    before = model.mlp.weight.grad.detach().clone()
    apply_gradient_projection(model, {"mlp": projector})
    after = model.mlp.weight.grad.detach()

    with TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "projectors.pt"
        torch.save({"projectors": {"mlp": compact}}, artifact_path)
        loaded_projectors = load_projectors(artifact_path)

    attached_model = ToyModel()
    attached_loss = attached_model(torch.randn(8, 4))
    attached_loss.backward()
    attached_before = attached_model.mlp.weight.grad.detach().clone()
    attach_result = attach_projectors_to_modules(attached_model, loaded_projectors)
    attached_metrics = apply_attached_gradient_projection(attached_model)
    attached_after = attached_model.mlp.weight.grad.detach()

    diag = projector_diagnostics(projector)
    compact_diag = projector_diagnostics(compact)
    leakage = after.matmul(covariance).norm().item()
    attached_leakage = attached_after.matmul(covariance).norm().item()

    print(f"nullity={results['mlp'].nullity}")
    print(f"norm_before={before.norm().item():.6f}")
    print(f"norm_after={after.norm().item():.6f}")
    print(f"leakage_after={leakage:.6e}")
    print(f"attached_modules={len(attach_result.attached)}")
    print(f"attached_norm_before={attached_before.norm().item():.6f}")
    print(f"attached_norm_after={attached_after.norm().item():.6f}")
    print(f"attached_leakage_after={attached_leakage:.6e}")
    print(f"symmetry_error={diag['symmetry_error']:.6e}")
    print(f"idempotence_error={diag['idempotence_error']:.6e}")

    assert results["mlp"].nullity == 2
    assert after.norm() <= before.norm() + 1e-6
    assert leakage < 1e-5
    assert attach_result.attached == ("mlp",)
    assert not attach_result.missing
    assert not attach_result.shape_mismatch
    assert attached_after.norm() <= attached_before.norm() + 1e-6
    assert attached_leakage < 1e-5
    assert "mlp.grad_norm_before" in attached_metrics
    assert "mlp.grad_norm_after" in attached_metrics
    assert diag["symmetry_error"] < 1e-6
    assert diag["idempotence_error"] < 1e-6
    assert compact_diag["idempotence_error"] < 1e-6
    assert compact_diag["spectral_norm"] == 1.0


if __name__ == "__main__":
    main()
