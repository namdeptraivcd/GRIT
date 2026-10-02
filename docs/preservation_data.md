# Split preservation data

GRIT uses separate projector, KL, and evaluation prompt roles.

| Role | General | Math | Code | Total |
| --- | ---: | ---: | ---: | ---: |
| Fixed projector construction | 334 | 333 | 333 | **1,000** |
| KL trust-region training/monitoring | 2,000 | 2,000 | 2,000 | **6,000** |
| Evaluation | supplied separately | supplied separately | supplied separately | external |

The source revisions, seed 66, and allocations are pinned in
`config/preservation/nspo_mix.json` and `nspo_monitor.json`. Projector and KL
sampling normalizes Unicode, whitespace, and case before hashing prompts. The KL
sample excludes every projector hash. Training also reads `--evaluation-file`
and rejects any normalized prompt shared by evaluation and either preservation
role. Exact hash disjointness does not prove semantic non-overlap.

The KL corpus contributes gradients and is therefore training data. It must not
be presented as held-out evaluation data.

## Prepare artifacts

The current frozen base is `Qwen/Qwen2.5-3B-Instruct` at commit
`aa8e72537993ba99e69dfaafa59ed015b17504d1`.

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1 \
bash scripts/run_preservation_projectors.sh all
```

The `sample`, `generate`, `build`, and `all` stages remain available. Prompt-only
sampling does not load the model:

```bash
bash scripts/run_preservation_projectors.sh sample
```

Outputs use model-specific `split_v1` paths:

```text
data/preservation/split_v1/qwen--qwen2.5-3b-instruct/projector/preserve_contexts.parquet
data/preservation/split_v1/qwen--qwen2.5-3b-instruct/monitor/preserve_contexts.parquet
artifacts/qwen--qwen2.5-3b-instruct_split_v1_projectors.pt
```

Historical artifacts are not overwritten. Complete artifacts are reused only
after provenance validation. Incomplete generation remains as
`contexts.partial.jsonl`; partial generation is not resumed in place.

## Frozen-base statistics

The same pinned base model generates responses for both preservation corpora.
Generation is greedy. Prompt IDs come from the policy chat template, and stored
context IDs are consumed without retokenizing decoded text.

For every generated response token, preparation runs a teacher-forced Hugging
Face forward in the configured dtype, default FP16. It stores:

- `base_topk_ids`: the frozen-base top-64 IDs;
- `base_topk_log_probs`: frozen-base log-probabilities for those IDs;
- `base_log_tail`: remaining probability mass as one finite log class;
- `base_top_k=64` and `base_statistics_dtype`.

The manifest records the same values, base revision, tokenizer fingerprint,
source checksums, and output checksum. Training rejects `k`, dtype, revision, or
tokenizer mismatches. These statistics come from teacher forcing, never from
vLLM generation-time log-probabilities.

The trainer computes current-policy mass for the fixed base IDs and derives the
current tail from the full-vocabulary logsumexp. All trust-region math therefore
uses 65 classes. Only response-token next-token positions are active. The
Proposition 2 TV bound applies to the coarsened 65-class distribution.

Storing statistics removes the need for a frozen FP32 3B model in each actor,
saving about 12.344 GB of actor GPU memory.

## Projectors

Projector construction consumes only the 1,000-row projector context file. It
records the construction checksum and normalized prompt hashes. The KL file is
validated against those hashes before training.

For a protected input dimension `d`, the builder computes the activation
covariance and its eigenspaces. It serializes the hard projector as either
`U U^T` or `I-U U^T`, choosing the smaller basis. This avoids storing a dense
matrix in the final artifact. For Qwen2.5-3B `down_proj`, `d=11008`; one dense
FP32 matrix would be 484,704,256 bytes (484.7 MB, 462.25 MiB).

## Per-step sampling

The KL sampler creates one deterministic global order by shuffling within each
domain and interleaving domains. Actor ranks take disjoint slices of the global
48-row step batch. With four actors each rank receives 12 rows. The sampler does
not repeat a row until wrap and reshuffles for the next epoch. Its cursor and
configuration are checkpointed.

A 40-step run consumes 1,920 distinct KL rows, 32% of the 6,000-row corpus. Full
coverage requires 125 steps. Metrics must describe sampled-step coverage rather
than claiming that the 40-step run covers the corpus.

## Training inputs

For preservation-enabled training, pass all of the following:

```text
--preserve-file <monitor preserve_contexts.parquet>
--projectors-path <compact projectors.pt>
--evaluation-file <reserved evaluation_prompts.parquet>
--model-revision aa8e72537993ba99e69dfaafa59ed015b17504d1
--top-k 64
--base-statistics-dtype float16
--max-preserve-length 2304
--preserve-batch-size 48
```

The evaluation parquet must have a nonempty `prompt` column and unique normalized
prompts. This repository does not synthesize it from training sources. Modal
training stops before GPU work if the file is absent.

## Verification

```bash
python -m pytest -q
python test_function/check_preservation_data.py
python test_function/check_trust_region_preservation.py
python test_function/check_projection.py
bash -n scripts/*.sh
```

The checks cover counts, exclusion sampling, overlap rejection including
evaluation, exact response boundaries, stored-statistic alignment, fixed top-64
plus tail math, and compact projector serialization. GPU preparation time,
CUDA/vLLM compatibility, and training memory fit require target-hardware
validation.
