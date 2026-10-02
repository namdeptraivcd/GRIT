#!/usr/bin/env bash
# Prepare disjoint projector/KL corpora from the same frozen base revision.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}"
PROMPT_ROOT="$DATA_ROOT/preservation/split_v1/prompts"
PROJECTOR_PROMPTS="$PROMPT_ROOT/projector"
MONITOR_PROMPTS="$PROMPT_ROOT/monitor"
IFS= read -r PROJECTOR_CONTEXTS < <("$PYTHON_BIN" - "$DATA_ROOT" "$ARTIFACT_ROOT" "$MODEL_PATH" <<'PY'
import sys
from scripts.preservation_data import preservation_paths
print(preservation_paths(*sys.argv[1:])["projector_contexts"])
PY
)
IFS= read -r MONITOR_CONTEXTS < <("$PYTHON_BIN" - "$DATA_ROOT" "$ARTIFACT_ROOT" "$MODEL_PATH" <<'PY'
import sys
from scripts.preservation_data import preservation_paths
print(preservation_paths(*sys.argv[1:])["preserve_file"])
PY
)
if [[ -z "${PROJECTORS_PATH:-}" ]]; then
  IFS= read -r PROJECTORS_PATH < <("$PYTHON_BIN" - "$DATA_ROOT" "$ARTIFACT_ROOT" "$MODEL_PATH" <<'PY'
import sys
from scripts.preservation_data import preservation_paths
print(preservation_paths(*sys.argv[1:])["projectors_path"])
PY
)
fi
STAGE="${1:-all}"
case "$STAGE" in sample|generate|build|all) ;; *) echo 'Usage: bash scripts/run_preservation_projectors.sh [sample|generate|build|all]' >&2; exit 2;; esac

# Reuse only complete, checksum-matching datasets with the intended provenance.
check_dataset() {
  "$PYTHON_BIN" - "$1" "$2" "$3" "$4" "${5:-}" "${6:-}" "$MODEL_PATH" <<'PY'
import json, sys
from pathlib import Path
from scripts.prepare_preservation_data import file_hash, read_rows
from scripts.preservation_data import preservation_prompt_hashes
root, filename, stage, config_path, input_path, exclude_path, model_path = sys.argv[1:]
root = Path(root)
config = json.loads(Path(config_path).read_text())
manifest = json.loads((root / 'manifest.json').read_text())
expected_counts = {source['domain']: source['count'] for source in config['sources']}
assert manifest['stage'] == stage, 'Unexpected preservation stage'
assert manifest['rows'] == sum(expected_counts.values()), 'Unexpected preservation row count'
assert manifest['domain_counts'] == expected_counts, 'Unexpected domain counts'
assert manifest['parquet'] == filename
assert file_hash(root / filename) == manifest['parquet_sha256'], 'Dataset checksum mismatch'
rows = read_rows(root / filename)
hashes = preservation_prompt_hashes(rows)
assert len(rows) == manifest['rows']
if stage == 'prompts_only':
    assert manifest['purpose'] == config['purpose'], 'Wrong preservation purpose'
    assert manifest['seed'] == config['seed'], 'Sampling seed mismatch'
    assert [{key: source[key] for key in config['sources'][0]} for source in manifest['sources']] == config['sources'], 'Source configuration mismatch'
    expected_exclusion = file_hash(Path(exclude_path)) if exclude_path else None
    assert manifest['excluded_prompts_sha256'] == expected_exclusion, 'Exclusion corpus mismatch'
else:
    assert manifest['base_model'] == model_path, 'Base model mismatch'
    assert manifest['input_sha256'] == file_hash(Path(input_path)), 'Source prompts changed'
    assert hashes == preservation_prompt_hashes(read_rows(Path(input_path))), 'Context prompts mismatch'
    assert {row['base_revision'] for row in rows} == {manifest['base_revision']}
    assert {row['tokenizer_sha256'] for row in rows} == {manifest['tokenizer_sha256']}
    assert manifest['base_top_k'] == 64, 'Stored base top-k mismatch'
    assert manifest['base_statistics_dtype'] == 'float16', 'Stored base-statistics dtype mismatch'
    assert all(row['base_top_k'] == 64 for row in rows)
    assert {row['base_statistics_dtype'] for row in rows} == {'float16'}
if exclude_path:
    assert hashes.isdisjoint(preservation_prompt_hashes(read_rows(Path(exclude_path)))), 'Projector/KL prompt overlap'
print('Verified:', root / filename)
PY
}
keep_partial() {
  if [[ -e "$1" ]]; then
    local backup="$1.partial.$(date +%Y%m%dT%H%M%S).$$"
    mv -- "$1" "$backup"
    echo "Kept unfinished attempt: $backup"
  fi
}

for role in projector monitor; do
  if [[ "$role" == projector ]]; then
    config="config/preservation/nspo_mix.json"
    prompts="$PROJECTOR_PROMPTS"
    contexts="$(dirname "$PROJECTOR_CONTEXTS")"
    exclude=""
  else
    config="config/preservation/nspo_monitor.json"
    prompts="$MONITOR_PROMPTS"
    contexts="$(dirname "$MONITOR_CONTEXTS")"
    exclude="$PROJECTOR_PROMPTS/preserve_prompts.parquet"
  fi
  if [[ "$STAGE" == sample || "$STAGE" == all ]]; then
    if [[ ! -s "$prompts/manifest.json" ]]; then
      keep_partial "$prompts"
      sampling_args=(--config "$config" --output-dir "$prompts")
      if [[ -n "$exclude" ]]; then sampling_args+=(--exclude-prompts "$exclude"); fi
      "$PYTHON_BIN" -u scripts/prepare_preservation_data.py sample "${sampling_args[@]}"
    fi
    check_dataset "$prompts" preserve_prompts.parquet prompts_only "$config" "" "$exclude"
  fi
  if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
    check_dataset "$prompts" preserve_prompts.parquet prompts_only "$config" "" "$exclude"
    generation_args=()
    if [[ "$role" == monitor ]]; then
      IFS= read -r base_revision < <("$PYTHON_BIN" - "$PROJECTOR_CONTEXTS" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]).parent / 'manifest.json').read_text())['base_revision'])
PY
)
      generation_args+=(--revision "$base_revision")
    else
      generation_args+=(--revision "$MODEL_REVISION")
    fi
    if [[ ! -s "$contexts/manifest.json" ]]; then
      keep_partial "$contexts"
      "$PYTHON_BIN" -u scripts/prepare_preservation_data.py generate \
        --prompts "$prompts/preserve_prompts.parquet" --output-dir "$contexts" \
        --model-path "$MODEL_PATH" --max-prompt-length 2048 --max-new-tokens 256 \
        --top-k 64 --dtype float16 --device cuda \
        "${generation_args[@]}"
    fi
    check_dataset "$contexts" preserve_contexts.parquet base_contexts "$config" "$prompts/preserve_prompts.parquet" "$exclude"
  fi
done

if [[ "$STAGE" == build ]]; then
  check_dataset "$(dirname "$PROJECTOR_CONTEXTS")" preserve_contexts.parquet base_contexts config/preservation/nspo_mix.json "$PROJECTOR_PROMPTS/preserve_prompts.parquet"
  check_dataset "$(dirname "$MONITOR_CONTEXTS")" preserve_contexts.parquet base_contexts config/preservation/nspo_monitor.json "$MONITOR_PROMPTS/preserve_prompts.parquet" "$PROJECTOR_PROMPTS/preserve_prompts.parquet"
fi
if [[ "$STAGE" == generate || "$STAGE" == build || "$STAGE" == all ]]; then
  "$PYTHON_BIN" - "$PROJECTOR_CONTEXTS" "$MONITOR_CONTEXTS" <<'PY'
import json, sys
from pathlib import Path
first, second = [json.loads((Path(p).parent / 'manifest.json').read_text()) for p in sys.argv[1:]]
assert first['base_revision'] == second['base_revision'], 'Projector/KL base revisions differ'
assert first['tokenizer_sha256'] == second['tokenizer_sha256'], 'Projector/KL tokenizers differ'
PY
fi
if [[ "$STAGE" == build || "$STAGE" == all ]]; then
  if [[ -e "$PROJECTORS_PATH" ]]; then
    "$PYTHON_BIN" - "$PROJECTORS_PATH" "$PROJECTOR_CONTEXTS" "$MONITOR_CONTEXTS" <<'PY'
import json, sys, torch
from pathlib import Path
from scripts.prepare_preservation_data import file_hash, read_rows
from scripts.preservation_data import preservation_prompt_hashes, validate_monitoring_split
artifact, contexts, monitor = map(Path, sys.argv[1:])
payload = torch.load(artifact, map_location='cpu', weights_only=False)
assert payload['preservation_sha256'] == file_hash(contexts), 'Projector construction corpus changed; use a new PROJECTORS_PATH'
assert payload['base_revision'] == json.loads((contexts.parent / 'manifest.json').read_text())['base_revision']
assert set(payload['projector_prompt_sha256']) == preservation_prompt_hashes(read_rows(contexts))
validate_monitoring_split(payload, read_rows(monitor))
print('Verified existing projector:', artifact)
PY
    exit 0
  fi
  BASE_REVISION=$("$PYTHON_BIN" - "$(dirname "$PROJECTOR_CONTEXTS")/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['base_revision'])
PY
  )
  MODEL_SNAPSHOT=$("$PYTHON_BIN" - "$(dirname "$PROJECTOR_CONTEXTS")/manifest.json" "$MODEL_PATH" <<'PY'
import json, sys
from huggingface_hub import snapshot_download
manifest = json.load(open(sys.argv[1]))
assert manifest['base_model'] == sys.argv[2], 'Base model mismatch'
print(snapshot_download(manifest['base_model'], revision=manifest['base_revision']))
PY
  )
  mkdir -p "$(dirname "$PROJECTORS_PATH")"
  PARTIAL_PATH="$PROJECTORS_PATH.partial.$$"
  "$PYTHON_BIN" -u scripts/build_projectors.py \
    --model-path "$MODEL_SNAPSHOT" \
    --model-revision "$BASE_REVISION" \
    --dataset-path "$PROJECTOR_CONTEXTS" \
    --dataset-split train --text-column text --module-pattern mlp \
    --relative-threshold 5e-4 --max-samples 1000 --seed 66 \
    --batch-size 1 --max-length 2304 --dtype float16 --device cuda \
    --output-path "$PARTIAL_PATH"
  test -s "$PARTIAL_PATH"
  mv -- "$PARTIAL_PATH" "$PROJECTORS_PATH"
  echo "Saved: $PROJECTORS_PATH"
  du -h "$PROJECTORS_PATH"
fi
