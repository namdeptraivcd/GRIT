"""Offline validation of the Modal notebook and its source upload contract."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_cells_are_valid_and_clean():
    notebook = json.loads((ROOT / "main.ipynb").read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    ids = set()
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] not in ids
        ids.add(cell["id"])
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["metadata"], dict)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]), filename=f"main.ipynb:{index}")
    text = (ROOT / "main.ipynb").read_text()
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "app.run(detach=True)" in text
    assert "FunctionCall.from_id" in text
    assert "smoke_config" not in text and "resume_config" not in text
    assert '"projector_relaxation": 0.05' in source
    assert '"task_batch_size": 321' in source
    assert '"preserve_batch_size": 48' in source
    assert "Qwen/Qwen2.5-3B-Instruct" in source
    assert "1,920" in source
    assert '"progress": True' in source
    assert '"base_statistics_dtype": "float16"' in source


def test_colab_small_notebook_keeps_full_grit_workflow():
    notebook = json.loads((ROOT / "GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb").read_text())
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"colab-small:{index}")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    for required in ("SMALL.policy", "SMALL.safety", "allow_colocated_rollout", "rollout_memory_utilization",
                     "required_data_inputs", "task_batch_size': 250", "task_microbatch_size': 24",
                     "preserve_batch_size': 48", "preservation_microbatch_size': 2",
                     "reward_batch_size': 32",
                     "top_k': 64", "epsilon_pres': 0.05", "use_curvature': True", "validation_steps': 5",
                     "fd_check_interval': 0",
                     "pku_saferlhf_test_1000.parquet", "--resume", "complete.json"):
        assert required in source
    assert "DATA_ROOT = REPO / 'data'" in source
    assert "ARTIFACT_ROOT = DRIVE_ROOT / 'artifacts'" in source
    assert "['bash', 'scripts/run_preservation_projectors.sh', 'build']" in source
    assert "'--skip-base-statistics'" in source
    assert "['--limit', '1920']" in source
    assert "run_with_progress" in source and "'progress': True" in source
    assert "PROMPT_ROOT / 'projector/preserve_prompts.parquet'" in source
    assert "assert projector_path.is_file() and projector_path.stat().st_size > 0" in source
    for preparation_command in ("prepare_grit_data.py", "prepare_evaluation_data.py",
                                "build_projectors.py"):
        assert preparation_command not in source


def test_colab_training_cell_matches_backend_cli(tmp_path):
    """Exercise the actual notebook arguments without starting pytest or training."""
    from grit.model_profiles import SMALL
    from scripts.train_grit import parse_args

    notebook = json.loads((ROOT / "GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "[Cell 5/7] Starting" in "".join(cell["source"]))
    commands = []
    namespace = {
        "sys": sys, "run_with_progress": commands.append, "SMALL": SMALL,
        "TASK_FILE": tmp_path / "task.parquet",
        "EVALUATION_FILE": tmp_path / "evaluation.parquet",
        "OUTPUT_DIR": tmp_path / "run", "HUB_REPO_ID": "test/grit",
        "paths": {"preserve_file": str(tmp_path / "preserve.parquet"),
                  "projectors_path": str(tmp_path / "projectors.pt")},
    }
    exec(compile(source, "colab:cell5", "exec"), namespace)
    assert len(commands) == 1 and commands[0][1:3] == ["-m", "pytest"]
    args = parse_args(namespace["arguments"])
    assert args.task_microbatch_size == 24
    assert args.preservation_microbatch_size == 2
    assert args.model_path == SMALL.policy and args.allow_colocated_rollout


def test_colab_preparation_validates_inputs_generates_reduced_contexts_then_builds():
    notebook = json.loads((ROOT / "GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "required_data_inputs" in "".join(cell["source"]))
    projector_generate = source.index("role_args = ['--skip-base-statistics']")
    monitor_generate = source.index("['--limit', '1920']")
    build = source.index("['bash', 'scripts/run_preservation_projectors.sh', 'build']")
    assert source.index("'held-out evaluation dataset': EVALUATION_FILE") < projector_generate
    assert projector_generate < build and monitor_generate < build
    assert "validate_context_artifact(repo_ctx_dir" in source
    assert "keep_existing(repo_ctx_dir)" in source


def test_colab_small_notebook_has_hf_token_fallback():
    notebook = json.loads((ROOT / "GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb").read_text())
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "os.environ.get('HF_TOKEN'" in source
    assert "userdata.get('HF_TOKEN')" in source
    assert "except Exception as exc:" in source
    assert "getpass('HF_TOKEN (input hidden): ')" in source


def test_modal_source_paths_and_resource_config(monkeypatch):
    uploads, resources = [], []
    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()
        def apt_install(self, *args):
            return self
        def add_local_file(self, path, remote_path, **kwargs):
            assert Path(path).is_file()
            uploads.append(Path(path).name)
            return self
        def add_local_dir(self, path, remote_path, **kwargs):
            assert Path(path).is_dir()
            uploads.append(Path(path).name)
            return self
        def run_commands(self, *args):
            return self
        def env(self, *args):
            return self
    class App:
        def __init__(self, *args, **kwargs):
            pass
        def function(self, **kwargs):
            resources.append(kwargs)
            return lambda function: function
    fake = SimpleNamespace(Image=Image, App=App,
                           Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
                           Secret=SimpleNamespace(from_name=lambda *a, **k: object()))
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setenv("GRIT_MODAL_GPU", "A100-80GB:3")
    spec = importlib.util.spec_from_file_location("grit_modal_validation", ROOT / "modal_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {"grit", "scripts", "config", "test_function"}.issubset(uploads)
    assert not {"data", "checkpoints", ".git", ".env"}.intersection(uploads)
    assert resources[1]["gpu"] == "A100-80GB:3"
    assert resources[1]["memory"] == 262144
    assert set(resources[1]["volumes"]) == {"/data", "/checkpoints", "/cache/huggingface"}
    from scripts.preservation_data import preservation_paths
    paths = preservation_paths("/data", "/data/artifacts", "Qwen/Qwen2.5-3B-Instruct")
    bundle = module.bundle_paths("Qwen/Qwen2.5-3B-Instruct")
    assert bundle["preserve_file"] == paths["preserve_file"]
    assert bundle["projectors_path"] == paths["projectors_path"]
