#!/usr/bin/env bash
# One persistent rollout GPU, all other selected GPUs are replicated actors.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${ACTOR_GPUS:?Set ACTOR_GPUS, e.g. 0,1,2}"
: "${ROLLOUT_GPU:?Set ROLLOUT_GPU, e.g. 3}"
IFS=',' read -r -a ACTORS <<< "$ACTOR_GPUS"
for actor in "${ACTORS[@]}"; do
  if [[ "$actor" == "$ROLLOUT_GPU" ]]; then
    echo 'Actor and rollout GPUs must be disjoint.' >&2
    exit 2
  fi
done
export CUDA_VISIBLE_DEVICES="$ACTOR_GPUS"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${PYTHON_BIN:-python}" -m torch.distributed.run --standalone \
  --nproc_per_node="${#ACTORS[@]}" scripts/train_grit.py --rollout-gpu "$ROLLOUT_GPU" "$@"
