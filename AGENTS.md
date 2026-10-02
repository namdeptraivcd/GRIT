# GRIT Agent Notes

Read `WORKFLOW.md` before changing training behavior. Keep this repository a
standalone PyTorch trainer with vLLM rollouts and explicit CLI arguments.

## Code map

- `scripts/train_grit.py`: model/data setup, distributed loop, checkpoints.
- `grit/grpo.py`: frozen group advantages, token-ratio task objective.
- `grit/step.py`: accumulation, predictor, central difference, final update.
- `grit/optimizer_delta.py`: task AdamW state and elementwise derivative.
- `grit/rollout.py`: persistent vLLM process and policy-version synchronization.
- `grit/projection.py`, `preservation_loss.py`, `trust_region.py`: GRIT math.
- `grit/predictor.py`, `curvature.py`, `update.py`: standalone math helpers/checks.
- `scripts/prepare_preservation_data.py`, `build_projectors.py`: artifacts.
- `modal_app.py`, `main.ipynb`: remote preparation/training on Modal.
- `test_function/`: CPU regression tests and mathematical correctness checks.

## Invariants

- Freeze rollout tokens, rewards, advantages, and old log-probs within a step.
- Normalize each microbatch by the global objective; synchronize before predictor.
- Apply the same relaxed projector `Q=P+gamma(I-P)` to the task delta, AdamW
  decay derivative and curvature direction. Leave correction unprojected.
- Commit task AdamW state once and reuse the cached task delta at final update.
- Preservation is anchored to frozen base policy, never the rollout policy.
- Full curvature uses sequential central difference with identical data and RNG.
- Gradient predictor uses `u=Q v`; AdamW uses `u=B^T Q v` from pre-step state.
- Do not construct a full Hessian/Jacobian or keep the task graph across probes.
- Training uses FP32 weights/forwards; lower-precision vLLM is rollout-only.
- Synchronize vLLM weights before generation and reject stale policy versions.
- Preserve generated data/checkpoints. Do not upload credentials or runtime data
  as source code. Do not launch paid Modal jobs unless the user asks to run them.

Run `python -m pytest -q` and `bash -n scripts/*.sh` after relevant changes.
CPU tests do not establish CUDA/vLLM compatibility or an end-to-end speedup.
