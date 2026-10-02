#!/usr/bin/env python3
"""Reserve unused task-source prompts for Colab evaluation overlap checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_grit_data import (
    PROMPT_ASSISTANT, PROMPT_BEGIN, PROMPT_USER, load_any_dataset,
)
from scripts.prepare_preservation_data import read_rows
from scripts.preservation_data import preservation_prompt_hashes, prompt_key


def source_text(text: str) -> str:
    """Compare raw prompts even when a task row uses the legacy wrapper."""
    prefix = PROMPT_BEGIN + PROMPT_USER.split("{input}")[0]
    suffix = " " + PROMPT_ASSISTANT
    if text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix):-len(suffix)]
    return text


def row_keys(row: dict) -> set[str]:
    return {
        prompt_key(source_text(value))
        for name in ("prompt", "raw_prompt")
        if isinstance(value := row.get(name), str) and value.strip()
    }


def prepare_evaluation(*, task_file: Path, exclude_prompts: list[Path], output: Path,
                       dataset: str = "PKU-Alignment/PKU-SafeRLHF",
                       split: str = "train", count: int = 1000, seed: int = 66) -> Path:
    if count <= 0:
        raise ValueError("Evaluation count must be positive")
    blocked = set()
    for path in [task_file, *exclude_prompts]:
        rows = read_rows(path)
        if not rows:
            raise ValueError(f"Empty exclusion corpus: {path}")
        for row in rows:
            keys = row_keys(row)
            if not keys:
                raise ValueError(f"Missing nonempty prompt in {path}")
            blocked.update(keys)

    if output.exists():
        rows = read_rows(output)
        if not rows:
            raise ValueError(f"Empty evaluation file: {output}")
        preservation_prompt_hashes(rows)
        seen = set(blocked)
        for row in rows:
            keys = row_keys(row)
            if keys & seen:
                raise ValueError(f"Evaluation overlaps task/preservation or has duplicates: {output}")
            seen.update(keys)
        print(f"Verified existing evaluation: {output} ({len(rows)} prompts)", flush=True)
        return output

    print(f"Preparing {count} held-out evaluation prompts from {dataset}/{split}", flush=True)
    source = load_any_dataset(dataset, split)
    indices = list(range(len(source)))
    random.Random(seed).shuffle(indices)
    selected = []
    for index in indices:
        row = source[index]
        prompt = row.get("raw_prompt") or row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        prompt = source_text(prompt).strip()
        if not prompt:
            continue
        keys = row_keys(row) | {prompt_key(prompt)}
        if keys & blocked:
            continue
        blocked.update(keys)
        selected.append({"prompt": prompt, "prompt_sha256": prompt_key(prompt),
                         "data_source": dataset, "source_split": split,
                         "source_row": index, "sampling_seed": seed})
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Need {count} disjoint evaluation prompts; found {len(selected)}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed write never leaves a seemingly complete evaluation file.
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        pq.write_table(pa.Table.from_pylist(selected), staged)
        staged.replace(output)
    print(f"Saved evaluation: {output} ({len(selected)} disjoint prompts)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--exclude-prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=66)
    prepare_evaluation(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
