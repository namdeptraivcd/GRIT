# GRIT workflow

This is the source of truth for the standalone trainer. Use `main.ipynb` for
Modal or `scripts/run_grit_vllm.sh` on a single Linux GPU host. The prescribed
run is a fresh 40-step run. Do not launch the paid run until it is explicitly
authorized.

## Runtime and pinned models

- Policy: [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), commit
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, 3,085,938,688 parameters.
- Safety reward model: [`Qwen/Qwen3Guard-Gen-4B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B), commit
  `6ec42827da0c1ff11e7a49dc269d2e810d27e108`, 4,411,424,256 parameters.
- Safety labels map as `Safe=0` and `Unsafe/Controversial=-1`. Missing or
  ambiguous labels are retried and then fail with the accumulated parse-error
  count. Metrics record parse errors, retries, and reward-model latency.
- Actor weights and forwards are FP32. TF32 is disabled. Lower precision is
  allowed only for vLLM rollout inference, the frozen-base statistics generated
  during preparation, and the frozen safety model.
- One persistent vLLM process owns a dedicated GPU. Rank zero exports the current
  actor weights before every rollout, waits for the matching policy-version
  acknowledgement, and requests generation only with that version. Prefix
  caching is disabled.
- Actor ranks use replicated weights and manual gradient synchronization. This is
  a single-host NCCL backend, not FSDP, tensor parallelism, or optimizer sharding.
  Projector application rejects a parameter whose last dimension is a local
  shard. Unequal local microbatch counts are safe because backward does no
  per-microbatch collective; one SUM all-reduce occurs at the end of each pass.

`modal_app.py` requests three A100-80GB GPUs: two actor ranks and one vLLM GPU.
The shared container requests 256 GiB of host RAM for two copies of offloaded
FP32 actor state. CUDA/vLLM
compatibility and memory fit still require the real target hardware.

The Colab 0.5B notebook uses a separate pinned Qwen2.5-0.5B-Instruct policy
(`7ae557604adf67be50417f59c2c2f167def9a775`, 494,032,768 parameters)
and Qwen3Guard-Gen-0.6B reward model
(`3706d237aa5d05ee7f6c8274c208a37211e9fd65`, 596,049,920 parameters).
On a single A100-80GB, `--allow-colocated-rollout` permits one FP32 actor and
the persistent vLLM process to share GPU 0. This exception is restricted to the
small pinned profile; policy-version synchronization and frozen rollouts are
unchanged. The task and preservation datasets, top-64 KL, projector construction,
AdamW predictor, and central-difference correction follow the same workflow.
The notebook saves complete checkpoints on Drive and resumes manually after a
Colab interruption. Colocated GPU memory fit and run duration require an actual
Colab run.

## Artifacts and data roles

```text
Task prompts -> current-policy vLLM rollout -> frozen rewards/advantages
Projector prompts (1,000) -> frozen-base responses -> compact projectors
KL prompts (6,000) -> frozen-base responses + stored base top-64 statistics
PKU-SafeRLHF test prompts (1,000) -> greedy rollout + safety score every 2 steps
GRIT update -> local checkpoint -> best-effort private Hub upload + metrics
```

The projector set contains 334 general, 333 math, and 333 code prompts. The KL
set contains 2,000 prompts from each domain. Sampling uses the pinned revisions
in `config/preservation/`, hashes normalized prompts, and rejects overlap between
the two sets. The checked-in evaluation file contains 1,000 unique prompts sampled
with seed 66 from the pinned PKU-SafeRLHF test split. Preparation rejects normalized
overlap with task, projector or KL prompts. After every two committed optimizer
steps, the updated policy generates one greedy response for every evaluation
prompt and Qwen3Guard records reward mean, unsafe fraction, parse diagnostics and
per-prompt results under `validation/step_*.jsonl`. Evaluation does not affect
gradients or checkpoint selection. This is a held-out safety evaluation, not a
general/math/code benchmark.

The KL sampler shuffles each domain, interleaves domains in a deterministic global
order, and takes 48 rows per step without replacement until wrap. Its cursor,
seed, corpus size, and stratification flag are saved in every checkpoint. At 48
rows per step, one pass through 6,000 rows takes 125 steps; a 40-step run sees
1,920 distinct rows, or 32% of the corpus. This is monitoring/training coverage
for the sampled rows, not coverage of the full KL corpus.

Generated context artifacts retain exact token IDs, response boundaries,
tokenizer fingerprint, prompt provenance, and base revision. Data, model caches,
checkpoints, credentials, and runtime logs stay outside Git.

## GRPO task objective

The trainer uses a token-ratio objective followed by a response mean:

```text
G = 5
A_i = (reward_i - mean_group(reward)) / max(std_group(reward), 1e-6)
ratio_i,t = exp(log_pi_theta(i,t) - frozen_old_log_prob(i,t))
L_task = mean_over_responses mean_over_response_tokens
         -min(ratio*A, clip(ratio, 1-e, 1+e)*A)
```

Each step uses exactly 321 prompts and 1,605 responses, temperature 1, and
`top_p=1`. With two ranks the prompt split is 161/160 and the response split
is 805/800. Complete groups remain on one rank. The task sampler
reshuffles at wrap: 34 complete steps consume 10,914 of 11,000 prompts and step
35 is the first step that crosses the epoch boundary.

Tokens, rewards, advantages, old log-probabilities, masks, and order are created
once before any task backward and remain frozen for the entire logical step.
Responses with zero advantage are omitted only from curvature probes; their
normalization weight remains in the global denominator.

## Relaxed projector

The artifact represents the hard null-space projector `P`. Every training use is

```text
Q = P + gamma * (I - P), gamma = 0.05
```

on protected Linear weights and the identity on every unprotected parameter.
`Q` acts on the full logical parameter vector. It is used for the task delta,
`Qv` in the curvature direction, and the AdamW decay derivative. The preservation
correction itself is not projected. `gamma=0` is the hard projector and
`gamma=1` is identity.

Projectors are serialized as `U U^T` or `I-U U^T`, choosing a basis for the
smaller subspace. Qwen2.5-3B has `intermediate_size=11008`; one dense FP32
`11008 x 11008` down-projector would occupy 484,704,256 bytes (484.7 MB, 462.25
MiB). Compact application multiplies by the basis without constructing that
dense matrix. Projector construction still needs each covariance and
eigendecomposition during the offline preparation phase.

## Fixed base top-64 trust region

For every response-token position, preparation chooses the 64 most probable
tokens under the frozen base model. It performs a teacher-forced Hugging Face
forward with the same chat template and the configured preparation dtype; it
does not reuse vLLM generation log-probabilities. Each row stores:

- the fixed 64 token IDs;
- their frozen-base log-probabilities;
- the frozen-base log tail mass;
- `base_top_k=64` and `base_statistics_dtype`.

The manifest stores the same `k`, dtype, revision, and tokenizer fingerprint.
Training requires `k=64` and rejects a row/manifest/CLI dtype mismatch. It does
not load a second 3B frozen-base model, saving one 12.344 GB FP32 GPU copy.

For current policy logits `z` and the fixed base set `S_t`:

```text
log_Z = logsumexp(z over the full vocabulary)
log p_i = z_i - log_Z, i in S_t
log p_tail = log(1 - exp(logsumexp(log p_i, i in S_t)))
```

The tail uses stable `log1mexp` and a finite floor. Gradient flows through the
full-vocabulary `logsumexp`. The current policy and base therefore define a
65-class distribution; the tail is retained and is never dropped or
renormalized away. Raw predictor-to-base KL, geometric projection, eta search,
projected target, loss, and gradient all use these 65 classes. The projected
target is detached. Only next-token positions whose target is part of the stored
response contribute; prompt and padding positions never contribute.

The total-variation guarantee from Proposition 2 applies to this coarsened
65-class distribution. It does not establish the same bound over individual
full-vocabulary tail tokens.

## Functional AdamW predictor and correction

At fixed `theta_before`, with moments from the start of the step:

```text
m_new = beta1*m_old + (1-beta1)*g
s_new = beta2*s_old + (1-beta2)*g^2
m_hat = m_new / (1-beta1^step)
s_hat = s_new / (1-beta2^step)
D = sqrt(s_hat) + eps
a(g) = m_hat / D
delta_raw = -lr * (a(g) + wd*theta_before)
Delta_task = Q delta_raw
B = (1-beta1)/((1-beta1^step)*D)
    - m_hat*(1-beta2)*g / ((1-beta2^step)*sqrt(s_hat)*D^2)
```

The zero-variance branch uses the finite limiting derivative. No
`optimizer.step()` is called, no task graph is retained across phases, and no
gradient clipping is implemented. Moments are formed from the unprojected task
gradient. They are committed exactly once after the complete candidate update is
finite and installed successfully.

One logical step is:

1. Accumulate task losses one response at a time. Each loss is divided by the
   global 1,605-response denominator. Move `p.grad` ownership into an FP32 tensor,
   clear `p.grad`, and SUM-reduce once.
2. Form the functional AdamW direction and `Delta_task`. Save `theta_before` by
   FP32 CPU copy. Never restore by subtracting a rounded delta.
3. Set `theta_pred=theta_before+Delta_task`. Accumulate 48 response-only KL
   losses, divided by the global valid response-token count, and SUM-reduce `v`.
4. Restore `theta_before`. If `v=0` globally, accept the cached task delta and
   skip curvature. This is expected while all tokens are within the KL ball.
5. Form `Qv`; update `c = v - lr*wd*Qv`; then reuse the `Qv` buffer in place as
   `u = B elementwise-multiplied-by Qv`. If `u=0`, retain the first-order and decay
   terms and skip the HVP.
6. With `r=rho/||u||`, run the same frozen task batch sequentially at
   `theta_before+r*u` and `theta_before-r*u`. Restore the same RNG before both
   passes. Accumulate the negative minus loss into the same gradient buffer, then
   perform one SUM all-reduce:

   ```text
   Hu = (g_plus - g_minus)/(2*r) = (g_plus-g_minus)*||u||/(2*rho)
   ```

7. Apply `c -= lr*Hu`, validate every candidate, install
   `theta_next = theta_before + Delta_task - lr*lambda_pres*c`, and commit AdamW
   state once. Any exception restores weights and leaves optimizer state
   unchanged.

Scheduled diagnostic steps also compare central differences at `rho` and
`2*rho`, and compare the central estimate with a one-sided estimate. These extra
passes replay the same frozen batch and RNG and do not affect the update.

## Memory at 3B

For 3,085,938,688 parameters, one full FP32 vector is 12,343,754,752 bytes =
12.344 GB = 11.496 GiB. The implementation offloads long-lived AdamW moments,
the saved pre-step weights, the task gradient used for commit, derivative
coefficients, and diagnostic reference HVPs to CPU. It moves one parameter tensor
at a time back to the actor GPU when needed.

| Phase | Full-size actor-GPU vectors at the transient peak | FP32 vector bytes |
| --- | --- | ---: |
| Task accumulation | weights, accumulated task gradient | 24.69 GB |
| Predictor projection | weights, task gradient, raw direction, projected delta | 49.38 GB |
| KL backward | weights, cached delta, preservation gradient | 37.03 GB |
| Curvature | weights, cached delta, correction, in-place `u`, HVP buffer | 61.72 GB |
| Final candidate | weights, cached delta, correction, one parameter temporary | about 37.03 GB plus one tensor |

The per-rank CPU peak can include the old and candidate moments during the atomic
commit, saved weights, task gradient, and the 4B safety model: about 82.9 GB
(77.2 GiB). Two actor processes can therefore approach 154.4 GiB before Python,
vLLM, staging, and allocator overhead; the Modal request is 256 GiB. Compact
projector bases, activations, logits,
CUDA workspaces, and fragmentation are additional GPU memory. The estimates show
why dense projectors and GPU-resident optimizer state are not viable; they do not
prove that the real 80 GB run fits.

The trainer logs the fraction of FP32 coordinates changed by the predictor and
probe and the fraction of those changes lost after FP16 or BF16 casting. It
raises an early-stop error if the predictor or a required probe is bit-identical
to `theta_before` in FP32.

On the first step, before applying the predictor, the actor measures mean and
maximum response-token KL from its FP32 base weights to the stored fixed-base
top-64 plus tail statistics. A maximum above one tenth of `epsilon_pres` raises
`base_anchor_precision_floor` in the metrics. This measures the anchor mismatch
introduced by lower-precision offline base forwards and any implementation
differences; it does not alter the update.

## Fresh run, checkpoints, and failure behavior

The notebook contains no smoke launch and no resume launch. It starts from the
pinned base in a new output directory for 40 steps. Before allocating the
training models, rank zero requires a token from the configured environment
variable, a Hub repository ID, and a successful Hugging Face identity check.
The token is passed directly to the client and is never written to arguments,
metrics, or logs.

A checkpoint is complete only after model, tokenizer, config, local
`training.pt`, sampler state, and `complete.json` are written. The optimizer file
stays local. After local completion, rank zero attempts a private Hub upload that
excludes `training.pt`; upload errors are recorded in metrics and do not terminate
training. Existing resume support remains for manual recovery, but it is not part
of the prescribed launch or acceptance procedure.

NaN/Inf, CUDA OOM, and FP32 predictor/probe resolution failures stop with a clear
message. A failed logical step does not advance samplers, write metrics, mutate
AdamW state, or leave partial actor weights.

## Metrics

`tqdm` shows current/total steps, elapsed time, ETA, reward, main norms, KL
violation rate, and rollout/reward/task/predictor/preservation/curvature/update times.
Each committed step appends to `metrics.jsonl`; `diagnostics_summary.json` is
atomically replaced.

Metrics include task/raw/projected/final norms, `g^TQg/||g||^2`, hard-subspace
leakage, raw 65-class KL mean/p95/max and per-domain values, token and context
violation rates, eta mean/p95/max, top-64 base coverage mean/p5, preservation
activation, central/one-sided and `rho`/`2rho` agreement, precision-loss
fractions, response tokens/second, reward latency, parse errors/retries, unsafe
fraction, zero-advantage groups, phase times, and peak allocated actor VRAM.
Coverage raises an alert when the p5 base mass is below 0.9.

The projector artifact does not retain the original activation matrix `K` or its
singular values. `task_delta_outside_null_norm` is an orthonormal subspace leakage
proxy; it is not the requested weighted `||Delta W K||`. Exact logging of that
quantity requires adding a compact activation factor to the artifact and paying
its storage and runtime cost.

## Verification

Run only CPU checks until the paid run is authorized:

```bash
python -m pytest -q
python test_function/check_trust_region_preservation.py
python test_function/check_projection.py
bash -n scripts/*.sh
python scripts/train_grit.py --help
```

The checks cover global response normalization, unequal local counts, zero and
nonzero preservation branches, functional AdamW including nonzero moments and
weight decay, relaxed `Q`, central differences, clipping-boundary probes, RNG and
weight restoration, single state commit, distributed reduction, stale rollout
versions, reward parsing retries, compact projector serialization, split-data
provenance, and fixed base top-64 plus tail math. CPU checks do not establish CUDA
compatibility, end-to-end throughput, GPU memory fit, or successful Hub upload.
