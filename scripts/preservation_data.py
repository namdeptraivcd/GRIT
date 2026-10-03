"""Source formatting and exact-token preservation batches."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


def preservation_paths(data_root, artifact_root, model_path: str) -> dict[str, str]:
    """New, model-specific paths leave historical shared-corpus artifacts intact."""
    model_key = re.sub(r"[^a-zA-Z0-9._-]", "--", model_path).lower()
    root = Path(data_root) / "preservation" / "split_v1" / model_key
    return {
        "projector_contexts": str(root / "projector" / "preserve_contexts.parquet"),
        "preserve_file": str(root / "monitor" / "preserve_contexts.parquet"),
        "projectors_path": str(Path(artifact_root) / f"{model_key}_split_v1_projectors.pt"),
    }


def select_context_rows(rows, limit=0):
    """Deterministic balanced subset of existing prompts; never resample sources."""
    if not limit:
        return rows
    if limit < 0 or limit > len(rows):
        raise ValueError(f"Need {limit} contexts but only {len(rows)} prompts are available")
    domains = sorted({row["domain"] for row in rows})
    count, remainder = divmod(limit, len(domains))
    quotas = {domain: count + (index < remainder) for index, domain in enumerate(domains)}
    selected = []
    for row in rows:
        if quotas[row["domain"]]:
            selected.append(row)
            quotas[row["domain"]] -= 1
    if any(quotas.values()):
        raise ValueError(f"Not enough prompts for balanced context subset: {quotas}")
    return selected


def prompt_key(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def preservation_prompt_hashes(rows) -> set[str]:
    """Identify source prompts, independently of stored metadata or responses."""
    hashes = set()
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Split preservation data requires the original prompt column")
        key = prompt_key(prompt)
        if row.get("prompt_sha256", key) != key:
            raise ValueError("Preservation prompt checksum mismatch")
        if key in hashes:
            raise ValueError("Duplicate normalized preservation prompt")
        hashes.add(key)
    return hashes


def validate_monitoring_split(projector_payload: dict, rows, evaluation_rows=None) -> None:
    """Require KL training prompts to be disjoint from projector construction."""
    projector_hashes = projector_payload.get("projector_prompt_sha256")
    if not projector_hashes:
        raise ValueError("Projector lacks prompt provenance; rebuild for split preservation data")
    monitoring_hashes = preservation_prompt_hashes(rows)
    if not monitoring_hashes:
        raise ValueError("Empty KL monitoring corpus")
    overlap = set(projector_hashes) & monitoring_hashes
    if overlap:
        raise ValueError(f"Projector and KL monitoring overlap on {len(overlap)} normalized prompts")
    if evaluation_rows is not None:
        evaluation_hashes = preservation_prompt_hashes(evaluation_rows)
        projector_overlap = set(projector_hashes) & evaluation_hashes
        monitoring_overlap = monitoring_hashes & evaluation_hashes
        if projector_overlap or monitoring_overlap:
            raise ValueError(
                "Preservation/evaluation prompt overlap: "
                f"projector={len(projector_overlap)}, monitoring={len(monitoring_overlap)}"
            )


def source_prompt(row: dict, domain: str) -> str:
    if domain == "general":
        return "\n\n".join(
            value.strip() for value in (row["instruction"], row.get("input", ""))
            if value and value.strip()
        )
    field = {"math": "question", "code": "query"}[domain]
    return (row[field] or "").strip()


def sample_source(rows, source: dict, seed: int, seen: set[str]) -> list[dict]:
    count = source["count"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Each source count must be a positive integer")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        row = rows[index]
        prompt = source_prompt(row, source["domain"])
        key = prompt_key(prompt)
        if not prompt or key in seen:
            continue
        seen.add(key)
        selected.append({
            "id": f"{source['repo_id']}:{source['revision']}:{source['split']}:{index}",
            "source_id": str(row.get("task_id", index)),
            "source_row": index,
            "data_source": source["repo_id"],
            "source_revision": source["revision"],
            "source_split": source["split"],
            "domain": source["domain"],
            "prompt": prompt,
            "text": prompt,
            "prompt_sha256": key,
        })
        if len(selected) == count:
            return selected
    raise ValueError(f"{source['repo_id']}: need {count} unique prompts, found {len(selected)}")


def tokenizer_fingerprint(tokenizer) -> str:
    payload = {
        "vocab": tokenizer.get_vocab(),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preservation_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Request an explicit encoding schema across Transformers 4/5."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    return list(encoded["input_ids"])


def stored_context_arrays(tokenizer, rows, *, max_length: int):
    """Pad exact contexts on the right; retain the original response boundary."""
    fingerprint = getattr(tokenizer, "_grit_preservation_fingerprint", None)
    if fingerprint is None:
        fingerprint = tokenizer_fingerprint(tokenizer)
        tokenizer._grit_preservation_fingerprint = fingerprint
    sequences, masks = [], []
    for row in rows:
        if row["tokenizer_sha256"] != fingerprint:
            raise ValueError("Preservation tokenizer differs from the training tokenizer")
        full_ids = list(row["input_ids"])
        start = int(row["response_start"])
        if not 1 <= start < len(full_ids):
            raise ValueError("Invalid preservation response_start")
        ids = full_ids[:max_length]
        if start >= len(ids):
            raise ValueError("max_preserve_length removes all response tokens; increase it")
        sequences.append(ids)
        masks.append([0] * start + [1] * (len(ids) - start))
    width = max(map(len, sequences))
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Preservation batching requires a pad token")
    input_ids = [ids + [pad] * (width - len(ids)) for ids in sequences]
    attention = [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]
    response = [mask + [0] * (width - len(mask)) for mask in masks]
    return input_ids, attention, response


def stored_preservation_batch(tokenizer, rows, *, max_length: int):
    """Return next-token labels and response-only mask without retokenization."""
    import torch

    ids, attention, response = stored_context_arrays(tokenizer, rows, max_length=max_length)
    input_ids = torch.tensor(ids, dtype=torch.long)
    attention = torch.tensor(attention, dtype=torch.long)
    response = torch.tensor(response, dtype=torch.bool)
    return input_ids, attention, input_ids[:, 1:].contiguous(), response[:, 1:].contiguous()


def stored_topk_preservation_batch(tokenizer, rows, *, max_length: int, top_k: int):
    """Return exact contexts plus stored base top-k/tail statistics aligned to logits."""
    import torch

    ids, attention, response = stored_context_arrays(tokenizer, rows, max_length=max_length)
    width = len(ids[0])
    top_ids, top_logp, log_tail = [], [], []
    for row, sequence in zip(rows, ids, strict=True):
        if int(row.get("base_top_k", -1)) != top_k:
            raise ValueError("Stored base top-k does not match --top-k")
        start = int(row["response_start"])
        response_count = min(len(row["input_ids"]), max_length)-start
        row_ids = list(row.get("base_topk_ids", []))[:response_count]
        row_logp = list(row.get("base_topk_log_probs", []))[:response_count]
        row_tail = list(row.get("base_log_tail", []))[:response_count]
        if not (len(row_ids) == len(row_logp) == len(row_tail) == response_count):
            raise ValueError("Stored base statistics do not cover every retained response token")
        if any(len(values) != top_k for values in row_ids+row_logp):
            raise ValueError("Stored base top-k row has the wrong class count")
        aligned_ids = [[0]*top_k for _ in range(width-1)]
        aligned_logp = [[0.0]*top_k for _ in range(width-1)]
        aligned_tail = [0.0]*(width-1)
        for offset, (token_ids, log_probs, tail) in enumerate(zip(row_ids, row_logp, row_tail, strict=True)):
            position = start-1+offset
            aligned_ids[position] = token_ids
            aligned_logp[position] = log_probs
            aligned_tail[position] = tail
        top_ids.append(aligned_ids)
        top_logp.append(aligned_logp)
        log_tail.append(aligned_tail)
    return (
        torch.tensor(ids, dtype=torch.long), torch.tensor(attention, dtype=torch.long),
        torch.tensor(response, dtype=torch.bool)[:, 1:].contiguous(),
        torch.tensor(top_ids, dtype=torch.long), torch.tensor(top_logp, dtype=torch.float32),
        torch.tensor(log_tail, dtype=torch.float32),
    )
