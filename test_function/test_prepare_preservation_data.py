"""Offline regression coverage for the context-generation entrypoint."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from scripts import prepare_preservation_data as prep


def test_generate_writes_contexts_to_requested_output(tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "fixture", "domain": "general", "prompt": "hello"}) + "\n")
    output = tmp_path / "nested" / "contexts"
    tokenizer = Mock(
        pad_token_id=0, special_tokens_map={"pad_token": "<pad>"}, chat_template="fixture",
    )
    tokenizer.get_vocab.return_value = {"<pad>": 0, "hello": 1, "reply": 2}
    tokenizer.apply_chat_template.return_value = {"input_ids": [1]}
    tokenizer.decode.return_value = "reply"
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.generate.return_value = torch.tensor([[1, 2]])
    model.return_value = SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 2.0]]]), past_key_values=None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        set_seed=Mock(),
    ))
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **kw: SimpleNamespace(sha="frozen-base"))

    prep.generate(SimpleNamespace(
        prompts=prompts, output_dir=output, model_path="fixture", revision="frozen-base",
        max_prompt_length=16, max_new_tokens=1, limit=0, device="cpu", dtype="float32",
        seed=66, top_k=2, progress=True, skip_base_statistics=False,
    ))

    rows = prep.read_rows(output / "preserve_contexts.parquet")
    assert len(rows) == 1
    assert rows[0]["input_ids"] == [1, 2]
    assert rows[0]["response_start"] == 1
    assert rows[0]["response_mask"] == [0, 1]
    assert rows[0]["base_topk_ids"] == [[2, 1]]
    assert rows[0]["base_revision"] == "frozen-base"
    assert rows[0]["base_statistics_dtype"] == "float32"
    assert json.loads((output / "contexts.jsonl").read_text()) == rows[0]
    assert not (output / "contexts.partial.jsonl").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["rows"] == 1
    assert manifest["input_sha256"] == prep.file_hash(prompts)
    assert manifest["parquet_sha256"] == prep.file_hash(output / "preserve_contexts.parquet")
    logs = capsys.readouterr().out
    assert "[3/4] Loading frozen base model/tokenizer" in logs
    assert "Generating contexts" in logs
    assert "Base top-k statistics" in logs


def test_projector_generation_skips_unused_base_statistics(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "fixture", "domain": "general", "prompt": "hello"}) + "\n")
    output = tmp_path / "projector"
    tokenizer = Mock(
        pad_token_id=0, special_tokens_map={"pad_token": "<pad>"}, chat_template="fixture",
    )
    tokenizer.get_vocab.return_value = {"<pad>": 0, "hello": 1, "reply": 2}
    tokenizer.apply_chat_template.return_value = {"input_ids": [1]}
    tokenizer.decode.return_value = "reply"
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.generate.return_value = torch.tensor([[1, 2]])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        set_seed=Mock(),
    ))
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **kw: SimpleNamespace(sha="frozen-base"))

    prep.generate(SimpleNamespace(
        prompts=prompts, output_dir=output, model_path="fixture", revision="frozen-base",
        max_prompt_length=16, max_new_tokens=1, limit=0, device="cpu", dtype="float32",
        seed=66, top_k=2, progress=False, skip_base_statistics=True,
    ))

    row = prep.read_rows(output / "preserve_contexts.parquet")[0]
    assert "base_topk_ids" not in row and "base_topk_log_probs" not in row
    assert "base_log_tail" not in row and "base_top_k" not in row
    assert model.call_count == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert "base_top_k" not in manifest and manifest["context_limit"] == 0


def test_context_limit_selects_balanced_domains():
    from scripts.preservation_data import select_context_rows

    rows = [
        {"domain": domain, "id": f"{domain}-{index}"}
        for index in range(4)
        for domain in ("general", "math", "code")
    ]
    selected = select_context_rows(rows, 6)
    assert {domain: sum(row["domain"] == domain for row in selected)
            for domain in ("general", "math", "code")} == {
                "general": 2, "math": 2, "code": 2,
            }
