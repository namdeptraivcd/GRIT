"""Offline regression coverage for automatic Colab evaluation preparation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_evaluation_data as prep
from scripts.prepare_grit_data import format_prompt


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    task = tmp_path / "task.parquet"
    projector = tmp_path / "projector.parquet"
    monitor = tmp_path / "monitor.parquet"
    write_rows(task, [{"prompt": format_prompt("TRAIN")},
                      {"prompt": "formatted alias", "raw_prompt": "raw train"}])
    write_rows(projector, [{"prompt": "Projector"}])
    write_rows(monitor, [{"prompt": "KL"}])
    source = [{"prompt": p} for p in
              ["train", " RAW   TRAIN ", "projector", "ＫＬ", "", None,
               "Held out one", "held  OUT one", "Held out two"]]
    monkeypatch.setattr(prep, "load_any_dataset", lambda *args: source)
    return dict(task_file=task, exclude_prompts=[projector, monitor],
                output=tmp_path / "eval" / "evaluation.parquet", count=2)


def test_automatic_holdout_is_disjoint_deterministic_and_reused(inputs, monkeypatch):
    output = prep.prepare_evaluation(**inputs)
    rows = prep.read_rows(output)
    assert {prep.prompt_key(r["prompt"]) for r in rows} == {
        prep.prompt_key("held out one"), prep.prompt_key("held out two")}
    assert all(r["source_split"] == "train" and r["sampling_seed"] == 66 for r in rows)
    copy = output.with_name("second.parquet")
    prep.prepare_evaluation(**dict(inputs, output=copy))
    assert prep.read_rows(copy) == rows
    before = output.read_bytes()
    monkeypatch.setattr(prep, "load_any_dataset", lambda *args: pytest.fail("Reuse must not download"))
    prep.prepare_evaluation(**inputs)
    assert output.read_bytes() == before


@pytest.mark.parametrize("rows", [
    [{"prompt": "train"}], [{"prompt": "ＫＬ"}],
    [{"prompt": "x"}, {"prompt": " X "}], [{"prompt": ""}],
])
def test_invalid_existing_file_is_rejected_without_overwrite(inputs, rows):
    write_rows(inputs["output"], rows)
    before = inputs["output"].read_bytes()
    with pytest.raises(ValueError):
        prep.prepare_evaluation(**inputs)
    assert inputs["output"].read_bytes() == before


def test_exhausted_source_leaves_no_fake_evaluation(inputs):
    with pytest.raises(ValueError, match="Need 3.*found 2"):
        prep.prepare_evaluation(**dict(inputs, count=3))
    assert not inputs["output"].exists()


def test_colab_prepares_evaluation_before_gpu_generation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "GRIT_Colab_Qwen2.5_0.5B_Qwen3Guard_0.6B.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    import sys
    context = dict(Path=Path, os=SimpleNamespace(environ={}), sys=sys, REPO=root,
                   userdata=SimpleNamespace(get=lambda key: "test-token"))
    config = cells["small-06"].replace(
        "Path('/content/drive/MyDrive/GRIT_colab_0p5b')", f"Path({str(tmp_path)!r})"
    ).replace("HUB_REPO_ID = ''", "HUB_REPO_ID = 'test/repo'")
    # Configuration must work before any dataset exists; keep test secrets local.
    exec(config, context)
    assert not context["EVALUATION_FILE"].exists()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "scripts/prepare_grit_data.py" in command:
            context["TASK_FILE"].parent.mkdir(parents=True, exist_ok=True)
            context["TASK_FILE"].touch()
        elif "scripts/prepare_evaluation_data.py" in command:
            assert commands[-2][-1] == "sample"
            context["EVALUATION_FILE"].parent.mkdir(parents=True, exist_ok=True)
            context["EVALUATION_FILE"].touch()
        elif command[-1] in ("generate", "build"):
            assert context["EVALUATION_FILE"].is_file()
            for key in ("preserve_file", "projectors_path"):
                path = Path(context["paths"][key])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    context["subprocess"] = SimpleNamespace(run=run)
    exec(cells["small-08"], context)
    assert [c[-1] for c in commands if c[0] == "bash"] == ["sample", "generate", "build"]
    commands.clear()
    exec(cells["small-08"], context)
    assert not any("scripts/prepare_grit_data.py" in c for c in commands)
