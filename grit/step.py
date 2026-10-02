"""One accumulated GRIT step. Model weights stay FP32; no retained task graph."""

from collections.abc import Callable
import math
import time

import torch

from grit.curvature import project_vector_with_module_projectors
from grit.projection import default_module_filter, project_with_projector


def rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def vector_norm(vector):
    norms = [torch.linalg.vector_norm(value.detach()) for value in vector.values()]
    return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0


def vector_dot(left, right):
    terms = [torch.dot(left[name].reshape(-1), right[name].reshape(-1)) for name in left]
    return float(torch.stack(terms).sum(dtype=torch.float64)) if terms else 0.0


def top_norms(vector, limit):
    """Small JSON-safe gradient fingerprint; never serialize gradient tensors."""
    if limit <= 0:
        return []
    names = list(vector)
    norms = torch.stack([torch.linalg.vector_norm(vector[name].detach()) for name in names])
    values, indices = torch.topk(norms, min(limit, len(names)))
    return [{"name": names[index], "norm": norm} for norm, index in zip(values.tolist(), indices.tolist())]


def q_retention(model, gradients, projectors, module_filter, relaxation):
    """Compute <g,Qg>/||g||^2 one parameter at a time to limit peak memory."""
    if module_filter is None:
        module_filter = default_module_filter
    protected = {
        f"{name}.weight" if name else "weight": projectors[name]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module_filter(name, module) and name in projectors
    }
    numerator = denominator = None
    for name, value in gradients.items():
        square = torch.dot(value.reshape(-1), value.reshape(-1)).to(torch.float64)
        denominator = square if denominator is None else denominator + square
        if name in protected:
            hard = project_with_projector(value, protected[name])
            contribution = (1-relaxation) * torch.dot(value.reshape(-1), hard.reshape(-1)).to(torch.float64) + relaxation * square
        else:
            contribution = square
        numerator = contribution if numerator is None else numerator + contribution
    return float(numerator / denominator.clamp_min(1e-30)) if numerator is not None else 0.0


def accumulate(model, losses: Callable, *, scale=1.0, clear=True):
    """Losses are already weighted by the global response count; reduce with SUM."""
    if clear:
        model.zero_grad(set_to_none=True)
    value = 0.0
    for loss in losses():
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite microbatch loss")
        (loss * scale).backward()
        value += float(loss.detach()) * scale
    return value


def take_gradients(model, parameters, reduce_gradients):
    result = {}
    for n, p in parameters:
        result[n] = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
        p.grad = None
    reduce_gradients(result)
    if any(not bool(torch.isfinite(g).all()) for g in result.values()):
        raise FloatingPointError("Nonfinite accumulated gradient")
    return result


def grit_step(
    model, task_losses, preservation_losses, projectors, *, lr, lambda_pres,
    optimizer=None, use_curvature=False, rho=0.05,
    reduce_gradients=lambda gradients: None, module_filter=None,
    check_curvature=False, gradient_topk=5, projector_relaxation=0.0,
    shared_vector_norm=vector_norm,
):
    """Apply a projected gradient or AdamW predictor and its own correction.

    Callbacks yield one scalar per microbatch, divided by the global number of
    responses. Every rank calls this function, including all reductions, together.
    optimizer=None selects the original proposal; otherwise use the functional
    AdamWDirectionPreconditioner. v=0 accepts exactly the cached predictor.
    """
    if lr <= 0 or lambda_pres < 0 or rho <= 0:
        raise ValueError("lr/rho must be positive and lambda_pres nonnegative")
    if not 0.0 <= projector_relaxation <= 1.0:
        raise ValueError("projector_relaxation must be in [0, 1]")
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for _, p in parameters):
        raise ValueError("GRIT training requires FP32 master parameters")
    if lambda_pres and preservation_losses is None:
        raise ValueError("Preservation data is required when lambda_pres > 0")

    def project(vector):
        return project_vector_with_module_projectors(
            model, vector, projectors, parameters=parameters, module_filter=module_filter,
            relaxation=projector_relaxation,
        )

    task_rng = rng_state()
    task_tick = time.monotonic()
    task_loss = accumulate(model, task_losses)
    g = take_gradients(model, parameters, reduce_gradients)
    task_backward_seconds = time.monotonic() - task_tick
    task_grad_norm = vector_norm(g)
    task_gradient_q_retention = q_retention(model, g, projectors, module_filter, projector_relaxation)
    task_gradient_top = top_norms(g, gradient_topk)
    predictor_tick = time.monotonic()
    if optimizer is None:
        raw = {n: -value for n, value in g.items()}
        coefficients = {}
    else:
        raw, coefficients = optimizer.prepare(parameters, g, derivative=use_curvature and lambda_pres > 0)
    raw_direction_norm = vector_norm(raw)
    delta = project(raw)
    projected_direction_norm = vector_norm(delta)
    if projector_relaxation < 1.0:
        hard_squared = max(
            (projected_direction_norm**2-projector_relaxation**2*raw_direction_norm**2)
            / (1.0-projector_relaxation**2),
            0.0,
        )
        hard_projected_direction_norm = math.sqrt(hard_squared)
        relaxation_added_direction_norm = projector_relaxation * math.sqrt(
            max(raw_direction_norm**2-hard_squared, 0.0)
        )
    else:
        hard_projected_direction_norm = None
        relaxation_added_direction_norm = None
    for value in delta.values():
        value.mul_(lr)
    del raw
    # These full-size buffers are long lived. Keep them off the actor GPU; each
    # tensor is copied back only when it is used. AdamW state is committed from
    # the CPU gradient after the candidate weights have passed validation.
    original = {n: p.detach().to(device="cpu", dtype=torch.float32, copy=True) for n, p in parameters}
    task_gradients_cpu = (
        {n: value.detach().to(device="cpu", dtype=torch.float32, copy=True) for n, value in g.items()}
        if optimizer is not None or check_curvature else None
    )
    del g

    def lost_fraction(vector, scale, dtype):
        changed = lost = 0
        total = sum(p.numel() for _, p in parameters)
        with torch.no_grad():
            for n, p in parameters:
                before = original[n].to(device=p.device)
                after = before+vector[n]*scale
                fp32_changed = after.ne(before)
                changed += int(fp32_changed.sum())
                lost += int((fp32_changed & after.to(dtype).eq(before.to(dtype))).sum())
        return lost/max(changed, 1), changed/max(total, 1)

    predictor_lost_fp16, predictor_changed_fraction = lost_fraction(delta, 1.0, torch.float16)
    predictor_lost_bf16, _ = lost_fraction(delta, 1.0, torch.bfloat16)
    if projected_direction_norm > 0 and predictor_changed_fraction == 0:
        raise FloatingPointError("Predictor rounded back to theta_before in FP32")
    predictor_seconds = time.monotonic()-predictor_tick

    @torch.no_grad()
    def move(vector=None, scale=1.0):
        for n, p in parameters:
            p.copy_(original[n])
            if vector is not None:
                p.add_(vector[n], alpha=scale)

    pres_loss = 0.0
    hvp_norm = 0.0
    hvp_used = False
    projected_preservation_norm = 0.0
    curvature_direction_norm = 0.0
    fd_plus_loss = 0.0
    fd_minus_loss = 0.0
    fd_relative_error = None
    fd_one_sided_relative_error = None
    preservation_gradient_top = []
    correction_gradient_top = []
    preservation_backward_seconds = 0.0
    central_difference_seconds = 0.0
    final_update_seconds = 0.0
    try:
        if lambda_pres:
            preservation_tick = time.monotonic()
            move(delta)
            pres_loss = accumulate(model, preservation_losses)
            correction = take_gradients(model, parameters, reduce_gradients)
            move()
            preservation_backward_seconds = time.monotonic() - preservation_tick
        else:
            correction = {n: torch.zeros_like(p) for n, p in parameters}
        pres_norm = vector_norm(correction)
        preservation_gradient_top = top_norms(correction, gradient_topk)
        if use_curvature and pres_norm:
            pv = project(correction)
            projected_preservation_norm = vector_norm(pv)
            if optimizer is None:
                u = pv
            else:
                for n in correction:
                    correction[n].add_(pv[n], alpha=-lr * optimizer.weight_decay)
                    # Reuse Qv storage for u=B⊙Qv after the decay term has
                    # consumed Qv. This avoids another full-size GPU vector.
                    pv[n].mul_(coefficients[n].to(pv[n]))
                u = pv
            coefficients.clear()
            norm = shared_vector_norm(u)
            curvature_direction_norm = norm
            if not torch.isfinite(torch.tensor(norm)):
                raise FloatingPointError("Nonfinite curvature direction")
            if norm > 0:
                radius = rho / norm
                probe_lost_fp16, probe_changed_fraction = lost_fraction(u, radius, torch.float16)
                probe_lost_bf16, _ = lost_fraction(u, radius, torch.bfloat16)
                if probe_changed_fraction == 0:
                    raise FloatingPointError("Central-difference probe rounded back to theta_before in FP32")
                after_preservation_rng = rng_state()
                central_tick = time.monotonic()
                try:
                    def central_difference(probe_radius):
                        move(u, probe_radius)
                        restore_rng(task_rng)
                        plus_loss = accumulate(model, task_losses)
                        move(u, -probe_radius)
                        restore_rng(task_rng)
                        minus_loss = -accumulate(model, task_losses, scale=-1.0, clear=False)
                        result = take_gradients(model, parameters, reduce_gradients)
                        for value in result.values():
                            value.div_(2 * probe_radius)
                        return result, plus_loss, minus_loss

                    hvp, fd_plus_loss, fd_minus_loss = central_difference(radius)
                    hvp_norm = vector_norm(hvp)
                    if check_curvature:
                        reference_hvp = {
                            name: value.to(device="cpu", dtype=torch.float32, copy=True)
                            for name, value in hvp.items()
                        }
                        del hvp
                        check_hvp, _, _ = central_difference(2 * radius)
                        check_norm = vector_norm(check_hvp)
                        for name in check_hvp:
                            check_hvp[name].sub_(reference_hvp[name].to(check_hvp[name]))
                        fd_relative_error = vector_norm(check_hvp) / max(hvp_norm, check_norm, 1e-30)
                        del check_hvp
                        move(u, radius)
                        restore_rng(task_rng)
                        accumulate(model, task_losses)
                        one_sided_hvp = take_gradients(model, parameters, reduce_gradients)
                        for name in one_sided_hvp:
                            one_sided_hvp[name].sub_(task_gradients_cpu[name].to(one_sided_hvp[name])).div_(radius)
                        one_sided_norm = vector_norm(one_sided_hvp)
                        for name in one_sided_hvp:
                            one_sided_hvp[name].sub_(reference_hvp[name].to(one_sided_hvp[name]))
                        fd_one_sided_relative_error = vector_norm(one_sided_hvp) / max(
                            hvp_norm, one_sided_norm, 1e-30
                        )
                        del one_sided_hvp
                        for name in correction:
                            correction[name].add_(reference_hvp[name].to(correction[name]), alpha=-lr)
                        del reference_hvp
                    else:
                        for name in hvp:
                            correction[name].add_(hvp[name], alpha=-lr)
                        del hvp
                    hvp_used = True
                finally:
                    move()
                    restore_rng(after_preservation_rng)
                    central_difference_seconds = time.monotonic() - central_tick
            del u
        coefficients.clear()
        final_tick = time.monotonic()
        correction_norm = vector_norm(correction)
        correction_gradient_top = top_norms(correction, gradient_topk)
        parameter_norm = vector_norm({n: p.detach() for n, p in parameters})
        correction_update_norm = lr * lambda_pres * correction_norm
        task_delta_norm = vector_norm(delta)
        final_delta_norms = []
        # Validate all candidate weights before mutating either weights or state.
        for n, p in parameters:
            update = delta[n] - lr * lambda_pres * correction[n]
            final_delta_norms.append(torch.linalg.vector_norm(update))
            candidate = original[n].to(device=p.device) + update
            if not bool(torch.isfinite(candidate).all()):
                raise FloatingPointError("Nonfinite GRIT candidate; step not committed")
        final_delta_norm = float(torch.linalg.vector_norm(torch.stack(final_delta_norms)))
        with torch.no_grad():
            for n, p in parameters:
                p.copy_(original[n])
                p.add_(delta[n]).add_(correction[n], alpha=-lr * lambda_pres)
        if optimizer is not None:
            optimizer.commit(parameters, task_gradients_cpu)
        final_update_seconds = time.monotonic()-final_tick
    except BaseException:
        move()
        raise
    finally:
        model.zero_grad(set_to_none=True)
    return {
        "task_loss": task_loss, "preservation_loss": pres_loss,
        "task_grad_norm": task_grad_norm,
        "task_gradient_q_retention": task_gradient_q_retention,
        "raw_task_direction_norm": raw_direction_norm,
        "projected_task_direction_norm": projected_direction_norm,
        "projection_retained_fraction": projected_direction_norm / max(raw_direction_norm, 1e-30),
        "projector_relaxation": projector_relaxation,
        "hard_projected_task_direction_norm": hard_projected_direction_norm,
        "hard_projection_retained_fraction": (
            hard_projected_direction_norm / max(raw_direction_norm, 1e-30)
            if hard_projected_direction_norm is not None else None
        ),
        "relaxation_added_direction_norm": relaxation_added_direction_norm,
        "task_delta_outside_null_norm": (
            lr * relaxation_added_direction_norm
            if relaxation_added_direction_norm is not None else None
        ),
        "task_delta_norm": task_delta_norm, "preservation_grad_norm": pres_norm,
        "projected_preservation_grad_norm": projected_preservation_norm,
        "curvature_direction_norm": curvature_direction_norm,
        "hvp_norm": hvp_norm, "curvature_active": float(hvp_used),
        "correction_active": float(pres_norm > 0),
        "corrected_preservation_grad_norm": correction_norm,
        "correction_update_norm": correction_update_norm,
        "final_delta_norm": final_delta_norm,
        "correction_to_task_ratio": correction_update_norm / max(task_delta_norm, 1e-30),
        "curvature_to_preservation_ratio": lr * hvp_norm / max(pres_norm, 1e-30),
        "task_correction_cosine": (
            -vector_dot(delta, correction) / max(task_delta_norm * correction_norm, 1e-30)
            if correction_update_norm else 0.0
        ),
        "parameter_norm": parameter_norm,
        "relative_update_norm": final_delta_norm / max(parameter_norm, 1e-30),
        "fd_parameter_radius": rho if hvp_used else 0.0,
        "fd_plus_loss": fd_plus_loss,
        "fd_minus_loss": fd_minus_loss,
        "fd_relative_error_rho_vs_2rho": fd_relative_error,
        "fd_one_sided_relative_error": fd_one_sided_relative_error,
        "predictor_changed_fraction_fp32": predictor_changed_fraction,
        "predictor_lost_fraction_fp16": predictor_lost_fp16,
        "predictor_lost_fraction_bf16": predictor_lost_bf16,
        "probe_changed_fraction_fp32": probe_changed_fraction if hvp_used else None,
        "probe_lost_fraction_fp16": probe_lost_fp16 if hvp_used else None,
        "probe_lost_fraction_bf16": probe_lost_bf16 if hvp_used else None,
        "task_backward_seconds": task_backward_seconds,
        "predictor_seconds": predictor_seconds,
        "preservation_backward_seconds": preservation_backward_seconds,
        "central_difference_seconds": central_difference_seconds,
        "final_update_seconds": final_update_seconds,
        "gradient_top": {
            "task": task_gradient_top,
            "preservation": preservation_gradient_top,
            "corrected_preservation": correction_gradient_top,
        },
    }
