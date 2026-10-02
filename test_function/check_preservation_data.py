#!/usr/bin/env python3
"""Check source sampling and stored response boundaries without model downloads."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preservation_data import (
    preservation_paths, preservation_prompt_hashes, preservation_prompt_ids, prompt_key,
    sample_source, source_prompt, stored_context_arrays, stored_topk_preservation_batch,
    tokenizer_fingerprint, validate_monitoring_split,
)


class ToyTokenizer:
    pad_token_id = 0
    special_tokens_map = {"pad_token": "<pad>"}
    chat_template = "test-template"

    def get_vocab(self):
        return {"<pad>": 0, "a": 1, "b": 2}


class PreservationDataTests(unittest.TestCase):
    def test_monitoring_rejects_overlap_normalized_duplicates_and_missing_provenance(self):
        payload = {"projector_prompt_sha256": [prompt_key("Ａ  B")]}
        validate_monitoring_split(payload, [{"prompt": "unseen"}])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_monitoring_split(payload, [{"prompt": "a b"}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_monitoring_split(payload, [{"prompt": "C"}, {"prompt": " c "}])
        with self.assertRaisesRegex(ValueError, "rebuild"):
            validate_monitoring_split({"preservation_sha256": "historical"}, [{"prompt": "new"}])
        with self.assertRaisesRegex(ValueError, "original prompt"):
            validate_monitoring_split(payload, [{"text": "a decoded context"}])
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_monitoring_split(payload, [{"prompt": "a b", "prompt_sha256": prompt_key("unseen")}])
        with self.assertRaisesRegex(ValueError, "Empty"):
            validate_monitoring_split(payload, [])
        with self.assertRaisesRegex(ValueError, "evaluation"):
            validate_monitoring_split(payload, [{"prompt": "unseen"}], [{"prompt": "unseen"}])

    def test_sampling_configs_produce_1000_and_6000_disjoint_prompts(self):
        """Exercise the preparation entrypoint, parquet and exclusion manifest offline."""
        import json
        import tempfile
        from collections import Counter
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.prepare_preservation_data import file_hash, read_rows, sample

        root = Path(__file__).resolve().parents[1]
        fixtures = {
            "general": [{"instruction": f"general {i}", "input": ""} for i in range(2500)],
            "math": [{"question": f"math {i}"} for i in range(2500)],
            "code": [{"query": f"code {i}"} for i in range(2500)],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_files = {}
            config = json.loads((root / "config/preservation/nspo_mix.json").read_text())
            for source in config["sources"]:
                path = directory / f"{source['domain']}.json"
                path.write_text(json.dumps(fixtures[source["domain"]]))
                source_files[source["repo_id"]] = path
            with patch("huggingface_hub.hf_hub_download", side_effect=lambda **kw: str(source_files[kw["repo_id"]])):
                projector_dir = directory / "projector"
                sample(SimpleNamespace(config=root / "config/preservation/nspo_mix.json",
                                       output_dir=projector_dir, cache_dir=None, exclude_prompts=None))
                exclusion = projector_dir / "preserve_prompts.parquet"
                args = SimpleNamespace(config=root / "config/preservation/nspo_monitor.json",
                                       output_dir=directory / "monitor", cache_dir=None, exclude_prompts=exclusion)
                sample(args)
                projectors = read_rows(exclusion)
                monitoring = read_rows(args.output_dir / "preserve_prompts.parquet")
                self.assertEqual(Counter(row["domain"] for row in projectors),
                                 {"general": 334, "math": 333, "code": 333})
                self.assertEqual(Counter(row["domain"] for row in monitoring),
                                 {"general": 2000, "math": 2000, "code": 2000})
                self.assertTrue(preservation_prompt_hashes(projectors).isdisjoint(preservation_prompt_hashes(monitoring)))
                manifest = json.loads((args.output_dir / "manifest.json").read_text())
                self.assertEqual(manifest["excluded_prompts_sha256"], file_hash(exclusion))
                self.assertEqual(manifest["parquet_sha256"], file_hash(args.output_dir / "preserve_prompts.parquet"))
                args.output_dir = directory / "repeat"
                sample(args)
                self.assertEqual(monitoring, read_rows(args.output_dir / "preserve_prompts.parquet"))
                args.exclude_prompts = None
                args.output_dir = directory / "invalid"
                with self.assertRaisesRegex(ValueError, "requires --exclude-prompts"):
                    sample(args)
                self.assertFalse(args.output_dir.exists())

    def test_model_paths_separate_projector_and_monitoring_and_preserve_legacy(self):
        paths = preservation_paths("/data", "/artifacts", "Qwen/Qwen2.5-3B-Instruct")
        self.assertIn("/projector/", paths["projector_contexts"])
        self.assertIn("/monitor/", paths["preserve_file"])
        self.assertIn("split_v1", paths["projectors_path"])
        small = preservation_paths("/data", "/artifacts", "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertNotEqual(paths, small)

    def test_projector_builder_records_only_used_prompts_after_subsampling(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import torch
        from scripts import build_projectors
        from scripts.prepare_preservation_data import file_hash

        class FixtureDataset(list):
            column_names = ["prompt", "text"]

        model = torch.nn.Module()
        model.mlp = torch.nn.Linear(2, 2)
        model.config = SimpleNamespace(_commit_hash="frozen-base")
        tokenizer = SimpleNamespace(pad_token="pad")
        transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        )
        corpus = FixtureDataset([{"prompt": f"original {i}", "text": f"decoded {i}"} for i in range(5)])
        used = []

        def batches(_tokenizer, dataset, *_args):
            used.extend(dataset)
            return []

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "contexts.json"
            source.write_text("fixture source")
            args = SimpleNamespace(model_path="fixture", model_revision="frozen-base", trust_remote_code=False,
                                   device="cpu", dtype="float32", dataset_path=str(source), dataset_split="train",
                                   text_column="text", max_samples=2, seed=66, batch_size=1, max_length=16,
                                   module_pattern="mlp", relative_threshold=5e-4, output_path=directory / "p.pt")
            with patch.dict(sys.modules, {"transformers": transformers}), \
                 patch.object(build_projectors, "parse_args", return_value=args), \
                 patch.object(build_projectors, "load_any_dataset", return_value=corpus), \
                 patch.object(build_projectors, "make_batches", side_effect=batches), \
                 patch.object(build_projectors, "collect_activation_covariances", return_value={"mlp": torch.diag(torch.tensor([1.0, 0.0]))}):
                build_projectors.main()
            payload = torch.load(args.output_path, weights_only=False)
            self.assertEqual(payload["base_revision"], "frozen-base")
            self.assertEqual(payload["preservation_sha256"], file_hash(source))
            self.assertEqual(set(payload["projector_prompt_sha256"]), preservation_prompt_hashes(used))
            self.assertEqual(len(payload["projector_prompt_sha256"]), 2)
            unused = [row for row in corpus if row not in used]
            validate_monitoring_split(payload, unused)
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_monitoring_split(payload, used)

    def test_chat_encoding_extracts_integer_ids(self):
        from collections import UserDict
        from unittest.mock import Mock
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = UserDict({"input_ids": [1, 2, 3]})
        self.assertEqual(preservation_prompt_ids(tokenizer, "hello"), [1, 2, 3])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            tokenize=True, add_generation_prompt=True, return_dict=True,
        )

    def test_source_formatting_excludes_answers(self):
        self.assertEqual(source_prompt({"instruction": "Explain", "input": "gravity", "output": "secret"}, "general"), "Explain\n\ngravity")
        self.assertEqual(source_prompt({"question": "2+2?", "answer": "4"}, "math"), "2+2?")
        self.assertEqual(source_prompt({"query": "Solve task", "prompt": "import math", "response": "solution"}, "code"), "Solve task")

    def test_sampling_is_unique_reproducible_and_respects_count(self):
        rows = [{"question": " A  B "}, {"question": "a b"}, {"question": ""}, {"question": "C"}]
        source = {"domain": "math", "repo_id": "fixture", "revision": "abc", "split": "train", "count": 2}
        first = sample_source(rows, source, 66, set())
        self.assertEqual(first, sample_source(rows, source, 66, set()))
        self.assertEqual({r["prompt_sha256"] for r in first}, {prompt_key("a b"), prompt_key("c")})
        self.assertTrue(all(r["source_revision"] == "abc" for r in first))
        with self.assertRaises(ValueError):
            sample_source(rows, {**source, "count": 3}, 66, set())
        with self.assertRaises(ValueError):
            sample_source(rows, source, 66, {prompt_key("a b")})

    def test_response_masks_preserve_boundary_and_padding(self):
        tokenizer = ToyTokenizer()
        identity = tokenizer_fingerprint(tokenizer)
        rows = [
            {"input_ids": [1, 2, 3, 4, 5], "response_start": 3, "tokenizer_sha256": identity},
            {"input_ids": [1, 2, 3], "response_start": 1, "tokenizer_sha256": identity},
        ]
        ids, attention, masks = stored_context_arrays(tokenizer, rows, max_length=4)
        self.assertEqual(ids, [[1, 2, 3, 4], [1, 2, 3, 0]])
        self.assertEqual(attention, [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(masks, [[0, 0, 0, 1], [0, 1, 1, 0]])
        # Next-token loss at position start-1 predicts the first response token.
        self.assertEqual([mask[1:] for mask in masks], [[0, 0, 1], [1, 1, 0]])
        with self.assertRaisesRegex(ValueError, "removes all response"):
            stored_context_arrays(tokenizer, rows, max_length=3)
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            stored_context_arrays(tokenizer, [{**rows[0], "tokenizer_sha256": "wrong"}], max_length=4)
        with self.assertRaisesRegex(ValueError, "response_start"):
            stored_context_arrays(tokenizer, [{**rows[0], "response_start": 0}], max_length=4)

    def test_stored_topk_statistics_align_only_to_response_tokens(self):
        tokenizer = ToyTokenizer()
        identity = tokenizer_fingerprint(tokenizer)
        rows = [{
            "input_ids": [1, 2, 3, 4], "response_start": 2, "tokenizer_sha256": identity,
            "base_top_k": 2, "base_topk_ids": [[0, 1], [1, 2]],
            "base_topk_log_probs": [[-0.2, -2.0], [-0.4, -1.5]],
            "base_log_tail": [-2.5, -2.0],
        }]
        ids, attention, mask, top_ids, top_logp, tail = stored_topk_preservation_batch(
            tokenizer, rows, max_length=4, top_k=2,
        )
        self.assertEqual(ids.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1]])
        self.assertEqual(mask.tolist(), [[False, True, True]])
        self.assertEqual(top_ids[0, 1:].tolist(), [[0, 1], [1, 2]])
        self.assertEqual(top_logp.shape, (1, 3, 2))
        self.assertEqual(tail.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
