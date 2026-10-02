"""Modal entrypoints used by main.ipynb. Importing does not start a GPU job."""

import json
import os
from pathlib import Path
import subprocess
import sys

import modal

LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/workspace/GRIT"
GPU = os.environ.get("GRIT_MODAL_GPU", "A100-80GB:3")
DATA_VOLUME = os.environ.get("GRIT_DATA_VOLUME", "grit-data")
OUTPUT_VOLUME = os.environ.get("GRIT_OUTPUT_VOLUME", "grit-checkpoints")
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name("grit-hf-cache", create_if_missing=True)
volumes = {"/data": data_volume, "/checkpoints": output_volume, "/cache/huggingface": cache_volume}
secret_name = os.environ.get("GRIT_HF_SECRET")
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libgl1")
    .add_local_file(LOCAL_ROOT / "requirements.txt", "/opt/grit/requirements.txt", copy=True)
    .add_local_file(LOCAL_ROOT / "requirements-vllm.txt", "/opt/grit/requirements-vllm.txt", copy=True)
    .run_commands("python -m pip install -r /opt/grit/requirements-vllm.txt")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": REMOTE_ROOT,
          "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
          "GRIT_MODAL_GPU": GPU, "GRIT_DATA_VOLUME": DATA_VOLUME,
          "GRIT_OUTPUT_VOLUME": OUTPUT_VOLUME, "GRIT_HF_SECRET": secret_name or ""})
)
# Explicit allowlist: never upload datasets, checkpoints, credentials, or .git.
for directory in ("grit", "scripts", "config", "test_function"):
    image = image.add_local_dir(LOCAL_ROOT / directory, f"{REMOTE_ROOT}/{directory}",
                                copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(LOCAL_ROOT / "pytest.ini", f"{REMOTE_ROOT}/pytest.ini", copy=True)
app = modal.App("grit-training", image=image)


def run(command, *, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)


def bundle_paths(model_path="Qwen/Qwen2.5-3B-Instruct"):
    from scripts.preservation_data import preservation_paths
    paths = preservation_paths("/data", "/data/artifacts", model_path)
    return {
        "task_file": "/data/task/task_train.parquet",
        "evaluation_file": "/data/eval/evaluation_prompts.parquet",
        "preserve_file": paths["preserve_file"],
        "projectors_path": paths["projectors_path"],
    }


@app.function(gpu="A100-80GB", cpu=4, memory=32768, timeout=86400, volumes=volumes, secrets=secrets)
def prepare(model_path="Qwen/Qwen2.5-3B-Instruct"):
    """Prepare 1,000 projector + 6,000 disjoint KL contexts from one frozen base."""
    try:
        paths = bundle_paths(model_path)
        if not Path(paths["task_file"]).exists():
            run([sys.executable, "scripts/prepare_grit_data.py", "--task-only", "--output-dir", "/data/task"])
        from grit.model_profiles import profile_for_policy
        env = dict(os.environ, DATA_ROOT="/data", ARTIFACT_ROOT="/data/artifacts", MODEL_PATH=model_path,
                   MODEL_REVISION=profile_for_policy(model_path).policy_revision,
                   PROJECTORS_PATH=paths["projectors_path"])
        run(["bash", "scripts/run_preservation_projectors.sh", "all"], env=env)
        manifest = json.loads((Path(paths["preserve_file"]).parent / "manifest.json").read_text())
        return {**paths, "model_path": model_path, "model_revision": manifest["base_revision"]}
    finally:
        data_volume.commit()
        cache_volume.commit()


@app.function(gpu=GPU, cpu=8, memory=262144, timeout=86400, volumes=volumes, secrets=secrets)
def train(config: dict):
    """Run one single-host training job; last GPU serves vLLM, others are actors."""
    import torch
    from scripts.train_grit import parse_args

    data_volume.reload()
    output_volume.reload()
    count = torch.cuda.device_count()
    if count < 2:
        raise ValueError("Choose at least two GPUs: one actor and one vLLM rollout GPU")
    run(["nvidia-smi"])
    arguments = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif value is not None:
            arguments.extend([flag, str(value)])
    # Validate before launching distributed processes.
    parsed = parse_args(["--rollout-gpu", str(count-1), *arguments])
    if not Path(parsed.output_dir).is_relative_to("/checkpoints"):
        raise ValueError("output_dir must be under /checkpoints for persistent storage")
    for field in ("task_file", "projectors_path", "preserve_file", "evaluation_file"):
        path = getattr(parsed, field)
        if path and not Path(path).is_file():
            raise FileNotFoundError(path)
    run([sys.executable, "-m", "pytest", "-q", "test_function/test_training.py"])
    env = dict(os.environ, ACTOR_GPUS=",".join(map(str, range(count-1))), ROLLOUT_GPU=str(count-1))
    try:
        run(["bash", "scripts/run_grit_vllm.sh", *arguments], env=env)
        checkpoints = sorted(str(p.parent) for p in Path(parsed.output_dir).glob("step_*/complete.json"))
        metrics = Path(parsed.output_dir) / "metrics.jsonl"
        summary = Path(parsed.output_dir) / "diagnostics_summary.json"
        return {"checkpoints": checkpoints, "last_metrics": json.loads(metrics.read_text().splitlines()[-1]),
                "diagnostics": json.loads(summary.read_text()),
                "actor_gpus": count-1, "rollout_gpu": count-1, "volume": OUTPUT_VOLUME}
    finally:
        output_volume.commit()
        cache_volume.commit()


@app.function(cpu=2, memory=4096, timeout=300, volumes=volumes)
def inspect_run(output_dir: str):
    output_volume.reload()
    root = Path(output_dir)
    if not root.is_relative_to("/checkpoints"):
        raise ValueError("Run path must be under /checkpoints")
    log = root / "metrics.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    from grit.diagnostics import summarize_metrics
    return {"checkpoints": sorted(str(p.parent) for p in root.glob("step_*/complete.json")),
            "metrics": rows[-2000:], "diagnostics": summarize_metrics(rows)}
