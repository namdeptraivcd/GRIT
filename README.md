# GRIT

Gradient Projection Meets Trust-Region Anchoring for forgetting-resistant
reinforcement learning. This repository contains a standalone PyTorch trainer
with vLLM rollouts and explicit CLI arguments.

## Current target

- Policy: `Qwen/Qwen2.5-3B-Instruct` at pinned commit
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Reward model: `Qwen/Qwen3Guard-Gen-4B` at pinned commit
  `6ec42827da0c1ff11e7a49dc269d2e810d27e108`.
- Two replicated FP32 actor GPUs plus one dedicated vLLM GPU, with a 256 GiB
  shared host-memory request for offloaded actor state.
- Five responses per prompt, 321 global prompts per step, microbatch one,
  temperature 1, and top-p 1.
- Relaxed projector `Q=P+0.05(I-P)` and a fixed frozen-base top-64 plus tail KL.

Open [main.ipynb](main.ipynb) from this repository to prepare Modal artifacts and
start the fresh 40-step run. The notebook deliberately has no smoke or resume
launch. It requires a Modal secret containing `HF_TOKEN`, a Hub repository ID,
and a separately prepared evaluation-prompt parquet used for disjointness checks.
Do not start the paid GPU job until the artifacts and evaluation input are ready.

Preparation creates 1,000 projector contexts and 6,000 disjoint KL contexts.
Frozen-base top-64 IDs, log-probabilities, and tail mass are stored with each
context, so the trainer does not load a second policy-sized base model. Projectors
are stored as a basis for the smaller subspace rather than dense matrices.

The trainer synchronizes the current policy to persistent vLLM before every
rollout and rejects stale policy versions. Actor backward calls make no
per-microbatch collectives; each pass ends with one gradient SUM, so the
161/160 prompt split across two ranks cannot hang.

## Local CPU verification

Use the environment that contains this repository's requirements:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python test_function/check_trust_region_preservation.py
python test_function/check_projection.py
bash -n scripts/*.sh
```

These checks establish CPU math and workflow invariants. They do not establish
CUDA/vLLM compatibility, A100-80GB memory fit, or end-to-end speed.

## Repository map

| Path | Responsibility |
| --- | --- |
| `scripts/train_grit.py` | Model/data setup, distributed loop, metrics, checkpoints |
| `grit/grpo.py` | Frozen group advantages and token-ratio task objective |
| `grit/step.py` | Accumulation, predictor, central difference, final update |
| `grit/optimizer_delta.py` | Functional AdamW state and elementwise derivative |
| `grit/rollout.py` | Persistent vLLM and policy-version synchronization |
| `grit/projection.py` | Compact hard projector and relaxed application |
| `grit/trust_region.py` | Fixed base top-64 plus tail projection |
| `scripts/prepare_preservation_data.py` | Frozen-base contexts and stored statistics |
| `scripts/build_projectors.py` | Projector construction |
| `modal_app.py`, `main.ipynb` | Remote preparation and training on Modal |
| `test_function/` | CPU regression and mathematical correctness checks |
| `eval_benchmarks/` | Separate checkpoint evaluation |

Read [WORKFLOW.md](WORKFLOW.md) for the exact objective, memory lifecycle,
synchronization, failure behavior, and known verification limits. Read
[docs/preservation_data.md](docs/preservation_data.md) for corpus provenance.

## Colab small-model profile

[GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb](GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb)
uses pinned Qwen2.5-0.5B-Instruct and Qwen3Guard-Gen-0.6B on one A100-80GB.
It retains the same task and preservation corpora, objective, and 40-step
configuration. The one actor and persistent vLLM process share GPU 0 with a
smaller vLLM memory reservation. Artifacts and checkpoints live on Drive; the
notebook can resume from a complete checkpoint after a Colab interruption.
Single-GPU vLLM/CUDA memory fit and throughput remain unverified.
