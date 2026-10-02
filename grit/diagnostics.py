"""Compact, actionable diagnostics for completed GRIT optimizer steps."""

from __future__ import annotations

from collections import Counter
import math


def step_alerts(metrics: dict, *, epsilon_pres: float) -> list[str]:
    alerts: list[str] = []
    if metrics.get("reward_std", 0.0) < 1e-8:
        alerts.append("reward_collapse")
    if metrics.get("task_grad_norm", 0.0) == 0.0:
        alerts.append("zero_task_gradient")
    if metrics.get("rollout_actor_logprob_mae", 0.0) > 2e-2:
        alerts.append("rollout_actor_logprob_mismatch")
    if metrics.get("raw_task_direction_norm", 0.0) > 0:
        retained = metrics.get("projection_retained_fraction", 1.0)
        relaxation = metrics.get("projector_relaxation", 0.0)
        if relaxation == 0.0 and retained < 1e-4:
            alerts.append("projector_blocks_task_update")
        elif relaxation > 0.0 and retained <= relaxation * 1.05:
            alerts.append("projector_update_at_relaxation_floor")
    if metrics.get("correction_to_task_ratio", 0.0) > 10.0:
        alerts.append("preservation_correction_dominates")
    if metrics.get("curvature_to_preservation_ratio", 0.0) > 10.0:
        alerts.append("curvature_term_dominates")
    if metrics.get("relative_update_norm", 0.0) > 1e-2:
        alerts.append("large_relative_update")
    if metrics.get("trust_projected_kl_max", 0.0) > epsilon_pres * 1.05 + 1e-7:
        alerts.append("trust_projection_outside_ball")
    if metrics.get("top64_base_coverage_p5", 1.0) < 0.9:
        alerts.append("low_top64_base_coverage")
    base_kl = metrics.get("base_kl_at_theta_before_max")
    if base_kl is not None and base_kl > epsilon_pres * 0.1:
        alerts.append("base_anchor_precision_floor")
    error = metrics.get("fd_relative_error_rho_vs_2rho")
    if error is not None and error > 0.2:
        alerts.append("central_difference_radius_unstable")
    one_sided_error = metrics.get("fd_one_sided_relative_error")
    if one_sided_error is not None and one_sided_error > 0.2:
        alerts.append("one_sided_curvature_disagrees")
    return alerts


def summarize_metrics(rows: list[dict]) -> dict:
    """Make a crash-safe run summary from any prefix of metrics.jsonl."""
    if not rows:
        return {"steps": 0, "alerts": ["no_completed_steps"]}
    counts = Counter(alert for row in rows for alert in row.get("alerts", []))

    def mean(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        return sum(values) / len(values) if values else None

    def maximum(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        return max(values) if values else None

    run_alerts = []
    if any(row.get("lambda_pres", 0.0) > 0 for row in rows) and not any(row.get("correction_active") for row in rows):
        run_alerts.append("preservation_never_activated")
    if any(row.get("curvature_active") for row in rows) and not any(
        row.get("fd_relative_error_rho_vs_2rho") is not None for row in rows
    ):
        run_alerts.append("central_difference_never_cross_checked")
    return {
        "steps": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
        "total_elapsed_seconds": sum(float(row.get("step_seconds", 0.0)) for row in rows),
        "mean_step_seconds": mean("step_seconds"),
        "mean_rollout_seconds": mean("rollout_seconds"),
        "mean_reward_seconds": mean("reward_seconds"),
        "mean_task_backward_seconds": mean("task_backward_seconds"),
        "mean_predictor_seconds": mean("predictor_seconds"),
        "mean_preservation_backward_seconds": mean("preservation_backward_seconds"),
        "mean_central_difference_seconds": mean("central_difference_seconds"),
        "mean_final_update_seconds": mean("final_update_seconds"),
        "mean_update_seconds": mean("update_seconds"),
        "mean_reward": mean("reward_mean"),
        "mean_reward_std": mean("reward_std"),
        "mean_projection_retained_fraction": mean("projection_retained_fraction"),
        "mean_hard_projection_retained_fraction": mean("hard_projection_retained_fraction"),
        "projector_relaxation": rows[-1].get("projector_relaxation"),
        "correction_activation_rate": mean("correction_active"),
        "curvature_activation_rate": mean("curvature_active"),
        "mean_trust_violation_fraction": mean("trust_violation_fraction"),
        "mean_top64_base_coverage": mean("top64_base_coverage_mean"),
        "base_kl_at_theta_before_mean": mean("base_kl_at_theta_before_mean"),
        "base_kl_at_theta_before_max": maximum("base_kl_at_theta_before_max"),
        "minimum_top64_base_coverage_p5": (
            min(float(row["top64_base_coverage_p5"]) for row in rows
                if row.get("top64_base_coverage_p5") is not None)
            if any(row.get("top64_base_coverage_p5") is not None for row in rows) else None
        ),
        "mean_rollout_tokens_per_second": mean("rollout_tokens_per_second"),
        "total_reward_parse_errors": sum(int(row.get("reward_parse_errors", 0)) for row in rows),
        "max_correction_to_task_ratio": maximum("correction_to_task_ratio"),
        "max_fd_relative_error": maximum("fd_relative_error_rho_vs_2rho"),
        "max_one_sided_relative_error": maximum("fd_one_sided_relative_error"),
        "max_peak_vram_gb": maximum("peak_vram_gb"),
        "alert_counts": dict(sorted(counts.items())),
        "alerts": run_alerts,
    }
