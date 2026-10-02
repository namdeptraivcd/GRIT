#!/usr/bin/env python3
"""Build GRIT Phase 1 null-space projectors from a preservation dataset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grit.projection import build_projectors_from_covariances, collect_activation_covariances, projector_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Hugging Face model path used as pi_base.")
    parser.add_argument("--model-revision", help="Pinned base revision used for preservation generation")
    parser.add_argument("--dataset-path", required=True, help="Dataset name/path for D_pres.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to read.")
    parser.add_argument("--text-column", default="prompt", help="Column containing preservation prompts.")
    parser.add_argument("--output-path", default="artifacts/projectors.pt", help="Where to save projectors.")
    parser.add_argument("--module-pattern", default="mlp", help="Protect Linear modules whose name contains this.")
    parser.add_argument("--relative-threshold", type=float, default=5e-4, help="Null eigenvalue threshold ratio.")
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum D_pres examples to use.")
    parser.add_argument("--batch-size", type=int, default=8, help="Tokenization/forward batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum tokenized sequence length.")
    parser.add_argument("--seed", type=int, default=66, help="Subsample seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_any_dataset(dataset_path: str, split: str):
    from datasets import load_dataset

    path = Path(dataset_path)
    if path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split=split)
    if path.suffix == ".json" or path.suffix == ".jsonl":
        return load_dataset("json", data_files=str(path), split=split)
    if path.suffix == ".csv":
        return load_dataset("csv", data_files=str(path), split=split)
    return load_dataset(dataset_path, split=split)


def make_batches(tokenizer, dataset, text_column: str, batch_size: int, max_length: int) -> Iterator[dict[str, torch.Tensor]]:
    def collate(rows):
        if "input_ids" in rows[0]:
            from scripts.preservation_data import stored_preservation_batch
            ids, attention, _, _ = stored_preservation_batch(tokenizer, rows, max_length=max_length)
            return {"input_ids": ids, "attention_mask": attention}
        texts = [row[text_column] for row in rows]
        return tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    yield from loader


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.model_revision, padding_side="left", trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[1/5] Loading model: {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    print(f"[1/5] Model loaded on {args.device} with dtype={args.dtype}", flush=True)

    print(f"[2/5] Loading preservation dataset: {args.dataset_path}", flush=True)
    dataset = load_any_dataset(args.dataset_path, args.dataset_split)
    revision = args.model_revision or getattr(model.config, "_commit_hash", None)
    if not revision:
        raise ValueError("Projector construction requires a pinned --model-revision")
    if "base_revision" in dataset.column_names and set(dataset["base_revision"]) != {revision}:
        raise ValueError("Projectors must use the preservation contexts' pinned base revision")
    if "input_ids" not in dataset.column_names and args.text_column not in dataset.column_names:
        raise ValueError(f"Column {args.text_column!r} not found. Available columns: {dataset.column_names}")
    has_original_prompts = "prompt" in dataset.column_names

    if args.max_samples > 0 and len(dataset) > args.max_samples:
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[: args.max_samples].tolist()
        dataset = Subset(dataset, indices)
    dataset_len = len(dataset)
    total_batches = math.ceil(dataset_len / args.batch_size)
    print(
        f"[2/5] Using {dataset_len} samples, batch_size={args.batch_size}, "
        f"total_batches={total_batches}",
        flush=True,
    )

    module_filter = lambda name, module: isinstance(module, torch.nn.Linear) and args.module_pattern in name
    protected_modules = [
        name for name, module in model.named_modules() if module_filter(name, module)
    ]
    print(
        f"[3/5] Collecting activations for {len(protected_modules)} modules "
        f"matching pattern={args.module_pattern!r}",
        flush=True,
    )
    batches = make_batches(tokenizer, dataset, args.text_column, args.batch_size, args.max_length)
    covariances = collect_activation_covariances(
        model,
        batches,
        module_filter=module_filter,
        device=args.device,
        progress_desc="[3/5] Forward batches",
        progress_total=total_batches,
    )
    print(f"[4/5] Building projectors from {len(covariances)} covariance matrices", flush=True)
    results = build_projectors_from_covariances(
        covariances,
        relative_threshold=args.relative_threshold,
        progress_desc="[4/5] Eigendecomposition",
    )

    payload = {
        "projectors": {name: result.compact for name, result in results.items()},
        "metadata": {
            name: {
                "threshold_value": result.threshold_value,
                "nullity": result.nullity,
                "rank": result.rank,
                "diagnostics": projector_diagnostics(result.compact),
            }
            for name, result in results.items()
        },
        "args": vars(args),
        "base_revision": revision,
    }
    if Path(args.dataset_path).is_file():
        from scripts.prepare_preservation_data import file_hash
        payload["preservation_sha256"] = file_hash(Path(args.dataset_path))
    if has_original_prompts:
        from scripts.preservation_data import preservation_prompt_hashes
        payload["projector_prompt_sha256"] = sorted(preservation_prompt_hashes(dataset))
    print(f"[5/5] Saving projectors to {output_path}", flush=True)
    torch.save(payload, output_path)

    print(f"Saved {len(results)} projectors to {output_path}")
    for name, meta in payload["metadata"].items():
        print(
            f"{name}: nullity={meta['nullity']} rank={meta['rank']} "
            f"sym={meta['diagnostics']['symmetry_error']:.3e} "
            f"idemp={meta['diagnostics']['idempotence_error']:.3e}"
        )


if __name__ == "__main__":
    main()
