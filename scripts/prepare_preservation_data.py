#!/usr/bin/env python3
"""Sample NSPO-domain prompts, then generate fixed-base GRIT contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.preservation_data import preservation_prompt_hashes, preservation_prompt_ids, sample_source, tokenizer_fingerprint
except ModuleNotFoundError as exc:
    # When this file is invoked as `python scripts/prepare_preservation_data.py`,
    # Python puts `scripts/` (not the repository root) first on sys.path.
    # Fall back to the sibling module while preserving unrelated import errors.
    if exc.name != "scripts.preservation_data":
        raise
    from preservation_data import preservation_prompt_hashes, preservation_prompt_ids, sample_source, tokenizer_fingerprint


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        return json.load(stream)


def finish_output(output: Path, rows: list[dict], manifest: dict, filename: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = output / filename
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    manifest.update({
        "rows": len(rows),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
        "parquet": filename,
        "parquet_sha256": file_hash(parquet),
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def sample(args) -> None:
    from huggingface_hub import hf_hub_download

    config = json.loads(args.config.read_text(encoding="utf-8"))
    exclude = getattr(args, "exclude_prompts", None)
    if config.get("purpose") == "monitor" and not exclude:
        raise ValueError("KL monitoring sampling requires --exclude-prompts for the projector corpus")
    seen = preservation_prompt_hashes(read_rows(exclude)) if exclude else set()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    for source in config["sources"]:
        print(f"Loading {source['repo_id']} / {source['split']}", flush=True)
        path = Path(hf_hub_download(
            repo_id=source["repo_id"], filename=source["filename"],
            revision=source["revision"], repo_type="dataset", cache_dir=args.cache_dir,
        ))
        selected = sample_source(read_rows(path), source, config["seed"], seen)
        rows.extend(selected)
        sources.append({**source, "file_sha256": file_hash(path)})
    random.Random(config["seed"]).shuffle(rows)
    (output / "prompts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    finish_output(output, rows, {
        "stage": "prompts_only", "seed": config["seed"], "sources": sources,
        "purpose": config.get("purpose", "projector"),
        "excluded_prompts_sha256": file_hash(exclude) if exclude else None,
        "sampling": "per-domain shuffled rows, global normalized prompt deduplication",
        "nspo_note": "Source families match NSPO citations; splits and ratios are GRIT choices.",
    }, "preserve_prompts.parquet")


def topk_statistics(logits, k: int):
    """Base top-k log-probabilities plus a stable tail from full-vocabulary logits."""
    import torch
    from grit.trust_region import log1mexp

    compute = logits.float()
    values, ids = torch.topk(compute, min(k, compute.shape[-1]), dim=-1)
    top_log_probs = values-torch.logsumexp(compute, dim=-1, keepdim=True)
    tail = log1mexp(torch.logsumexp(top_log_probs, dim=-1))
    return ids, top_log_probs, tail


def teacher_forced_base_statistics(model, sequence: list[int], response_start: int, k: int, device: str):
    """Compute fixed base statistics without materializing sequence x vocabulary logits."""
    import torch

    prefix = torch.tensor([sequence[:response_start]], dtype=torch.long, device=device)
    ids_rows, logp_rows, tail_rows = [], [], []
    with torch.inference_mode():
        output = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
        logits = output.logits[:, -1]
        past = output.past_key_values
        ids, logp, tail = topk_statistics(logits, k)
        ids_rows.append(ids[0].cpu())
        logp_rows.append(logp[0].cpu())
        tail_rows.append(tail[0].cpu())
        for offset, token in enumerate(sequence[response_start:-1], start=1):
            current = torch.tensor([[token]], dtype=torch.long, device=device)
            attention = torch.ones((1, response_start+offset), dtype=torch.long, device=device)
            output = model(input_ids=current, attention_mask=attention, past_key_values=past, use_cache=True)
            past = output.past_key_values
            ids, logp, tail = topk_statistics(output.logits[:, -1], k)
            ids_rows.append(ids[0].cpu())
            logp_rows.append(logp[0].cpu())
            tail_rows.append(tail[0].cpu())
    return (
        torch.stack(ids_rows).tolist(), torch.stack(logp_rows).tolist(),
        torch.stack(tail_rows).tolist(),
    )


def generate(args) -> None:
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.max_prompt_length < 1 or args.max_new_tokens < 1:
        raise ValueError("Token limits must be positive")
    rows = read_rows(args.prompts)
    if not rows:
        raise ValueError("No preservation prompts")
    if args.limit:
        rows = rows[:args.limit]
    revision = model_info(args.model_path, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fingerprint = tokenizer_fingerprint(tokenizer)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, revision=revision, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    model.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    generated = []
    
    from tqdm import tqdm
    
    with (output / "contexts.partial.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(tqdm(rows, desc="Generating contexts")):
            prompt_ids = preservation_prompt_ids(tokenizer, row["prompt"])
            # Do not truncate the problem or the chat template silently.
            if len(prompt_ids) > args.max_prompt_length:
                raise ValueError(f"{row['id']}: prompt exceeds --max-prompt-length")
            set_seed(args.seed + index)
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                sequence = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )[0].tolist()
            response_ids = sequence[len(prompt_ids):]
            if not response_ids:
                raise ValueError(f"{row['id']}: empty generated response")
            topk_ids, topk_log_probs, log_tail = teacher_forced_base_statistics(
                model, sequence, len(prompt_ids), args.top_k, device,
            )
            record = {
                **row,
                "text": tokenizer.decode(sequence, skip_special_tokens=False),
                "base_response": tokenizer.decode(response_ids, skip_special_tokens=True),
                "input_ids": sequence,
                "response_start": len(prompt_ids),
                "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
                "tokenizer_sha256": fingerprint,
                "base_model": args.model_path,
                "base_revision": revision,
                "base_topk_ids": topk_ids,
                "base_topk_log_probs": topk_log_probs,
                "base_log_tail": log_tail,
                "base_top_k": args.top_k,
                "base_statistics_dtype": str(dtype).removeprefix("torch."),
            }
            generated.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    finish_output(output, generated, {
        "stage": "base_contexts", "base_model": args.model_path, "base_revision": revision,
        "tokenizer_sha256": fingerprint, "seed": args.seed, "do_sample": False,
        "max_prompt_length": args.max_prompt_length, "max_new_tokens": args.max_new_tokens,
        "base_top_k": args.top_k, "base_statistics_dtype": str(dtype).removeprefix("torch."),
        "base_statistics": "teacher-forced HF forward; fixed base top-k plus tail",
        "input_sha256": file_hash(args.prompts),
    }, "preserve_contexts.parquet")
    (output / "contexts.partial.jsonl").rename(output / "contexts.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--config", type=Path, default=REPO_ROOT / "config/preservation/nspo_mix.json")
    sampling.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/preservation/nspo_mix")
    sampling.add_argument("--cache-dir", default=str(REPO_ROOT / ".cache/huggingface"))
    sampling.add_argument("--exclude-prompts", type=Path,
                          help="Exclude normalized prompts from this projector corpus when sampling KL data")
    generation = commands.add_parser("generate")
    generation.add_argument("--prompts", type=Path, required=True)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.add_argument("--model-path", default="Qwen/Qwen2.5-3B-Instruct")
    generation.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    generation.add_argument("--seed", type=int, default=66)
    generation.add_argument("--max-prompt-length", type=int, default=2048)
    generation.add_argument("--max-new-tokens", type=int, default=256)
    generation.add_argument("--device", default=None)
    generation.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    generation.add_argument("--top-k", type=int, default=64)
    generation.add_argument("--limit", type=int, default=0, help="Positive count for generation smoke tests")
    args = parser.parse_args()
    if args.stage == "generate" and (args.limit < 0 or args.top_k < 1):
        parser.error("--limit must be non-negative and --top-k positive")
    (sample if args.stage == "sample" else generate)(args)


if __name__ == "__main__":
    main()
