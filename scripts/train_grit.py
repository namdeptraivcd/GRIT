#!/usr/bin/env python3
"""Standalone GRPO + GRIT training with persistent vLLM rollouts."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from grit.data import score_safety_rewards, tokenize_preserve_batch
from grit.diagnostics import step_alerts, summarize_metrics
from grit.grpo import group_advantages, grpo_loss, response_batch, token_log_probs
from grit.model_profiles import LARGE, SMALL, profile_for_policy
from grit.optimizer_delta import AdamWDirectionPreconditioner
from grit.preservation_loss import preservation_kl_loss
from grit.rollout import VLLMRollout
from grit.step import grit_step
from grit.trust_region import coarsened_base_log_probs, coarsened_policy_log_probs, kl_from_log_probs

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=LARGE.policy)
    parser.add_argument("--model-revision", default=LARGE.policy_revision)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--preserve-file", help="Disjoint KL monitoring contexts; required unless --lambda-pres 0")
    parser.add_argument("--evaluation-file", help="Held-out prompt corpus for periodic safety validation")
    parser.add_argument("--projectors-path", required=True)
    parser.add_argument("--projector-relaxation", type=float, default=0.05,
                        help="Pass-through outside null space: 0=hard NSPO, 1=no projection")
    parser.add_argument("--output-dir", default="checkpoints/grit")
    parser.add_argument("--resume", help="Checkpoint directory produced by this trainer")
    parser.add_argument("--rollout-gpu", required=True, help="Physical GPU ID or UUID used by vLLM")
    parser.add_argument("--allow-colocated-rollout", action="store_true",
                        help="Allow one actor and vLLM to share one GPU (Colab A100-80GB profile)")
    parser.add_argument("--rollout-dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--rollout-memory-utilization", type=float, default=0.8)
    parser.add_argument("--rollout-timeout", type=float, default=1800)
    parser.add_argument("--task-batch-size", type=int, default=321, help="Global prompts per optimizer step")
    parser.add_argument("--preserve-batch-size", type=int, default=48, help="Global monitoring contexts per step")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=256)
    parser.add_argument("--max-preserve-length", type=int, default=2304)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--task-optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--lambda-pres", type=float, default=1.0)
    parser.add_argument("--epsilon-pres", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=64, help="Preservation support size; 0 uses full vocabulary")
    parser.add_argument("--base-statistics-dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                        help="Required dtype recorded for frozen-base top-64 statistics")
    parser.add_argument("--use-curvature", action="store_true")
    parser.add_argument("--fd-radius", type=float, default=0.05)
    parser.add_argument("--fd-check-interval", type=int, default=10,
                        help="Compare central differences at rho and 2*rho every N steps; 0 disables")
    parser.add_argument("--gradient-topk", type=int, default=5,
                        help="Store this many largest parameter-gradient norms per step; 0 disables")
    parser.add_argument("--module-pattern", default="mlp")
    parser.add_argument("--safety-model-path", default=LARGE.safety)
    parser.add_argument("--safety-model-revision", default=LARGE.safety_revision)
    parser.add_argument("--safety-attempts", type=int, default=3)
    parser.add_argument("--reward-batch-size", type=int, default=4)
    parser.add_argument("--safety-max-new-tokens", type=int, default=128)
    parser.add_argument("--validation-steps", type=int, default=2,
                        help="Run greedy safety validation every N committed updates; 0 disables")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hub-repo-id", default=os.environ.get("GRIT_HUB_REPO_ID"))
    parser.add_argument("--hub-token-env", default="HF_TOKEN")
    args = parser.parse_args(argv)
    for key in ("task_batch_size", "preserve_batch_size", "generations", "max_prompt_length",
                "max_response_length", "max_preserve_length", "reward_batch_size", "max_steps", "save_steps",
                "safety_max_new_tokens", "safety_attempts"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.generations < 2 or args.lr <= 0 or args.adam_eps <= 0 or args.fd_radius <= 0:
        parser.error("Need generations >= 2 and positive lr, adam-eps, fd-radius")
    if args.fd_check_interval < 0 or args.gradient_topk < 0 or args.validation_steps < 0:
        parser.error("--fd-check-interval, --gradient-topk and --validation-steps must be nonnegative")
    if args.lambda_pres < 0 or args.weight_decay < 0 or not 0 < args.clip_ratio < 1:
        parser.error("Invalid loss/optimizer coefficients")
    if not 0 <= args.projector_relaxation <= 1:
        parser.error("--projector-relaxation must be in [0, 1]")
    if args.lambda_pres and not args.preserve_file:
        parser.error("--preserve-file is required when --lambda-pres > 0")
    if (args.lambda_pres or args.validation_steps) and not args.evaluation_file:
        parser.error("--evaluation-file is required for safety validation and preservation disjointness")
    if args.task_optimizer == "sgd" and args.weight_decay:
        parser.error("The original proposal (--task-optimizer sgd) has no weight decay")
    if args.top_k != 64:
        parser.error("This workflow requires --top-k 64 to match stored base statistics")
    for name in ("model_revision", "safety_model_revision"):
        if not getattr(args, name) or getattr(args, name) == "main":
            parser.error(f"--{name.replace('_', '-')} must be a pinned commit SHA")
    try:
        profile = profile_for_policy(args.model_path)
    except ValueError as error:
        parser.error(str(error))
    if args.model_revision != profile.policy_revision:
        parser.error("Policy revision does not match the pinned model profile")
    if (args.safety_model_path, args.safety_model_revision) != (profile.safety, profile.safety_revision):
        parser.error("Reward model/revision does not match the pinned policy profile")
    if args.allow_colocated_rollout and profile != SMALL:
        parser.error("Colocated rollout is supported only for the pinned 0.5B policy profile")
    return args


def sum_gradients(gradients):
    if dist.is_initialized():
        for value in gradients.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)


def shared_vector_norm(vector):
    """Verify replicated ranks use one identical direction norm and branch."""
    squares = [value.detach().square().sum(dtype=torch.float64) for value in vector.values()]
    local = torch.stack(squares).sum().sqrt() if squares else torch.tensor(0.0)
    if dist.is_initialized():
        low, high = local.clone(), local.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if not torch.allclose(low, high, rtol=1e-7, atol=1e-12):
            raise RuntimeError("Actor ranks formed different curvature directions")
        local = high
    return float(local)


def broadcast_call(rank, function):
    """Propagate rank-zero rollout/checkpoint failures before other ranks continue."""
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {"value": function()}
        except Exception as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]["value"]


class EpochSampler:
    """Reproducible global order, reshuffled on wrap and interleaved by domain for KL."""

    def __init__(self, dataset, seed, *, stratify=False):
        if not len(dataset):
            raise ValueError("Cannot sample an empty dataset")
        self.dataset = dataset
        self.seed = seed
        self.domains = None
        if stratify:
            if "domain" not in dataset.column_names:
                raise ValueError("KL monitoring rows need a domain column for stratified sampling")
            self.domains = {}
            for index, domain in enumerate(dataset["domain"]):
                if not domain:
                    raise ValueError("KL monitoring rows need nonempty domains")
                self.domains.setdefault(domain, []).append(index)
        self._epoch = None
        self._order = []
        self.cursor = 0

    def order(self, epoch):
        if self._epoch != epoch:
            rng = random.Random(self.seed + epoch)
            if self.domains is None:
                order = list(range(len(self.dataset)))
                rng.shuffle(order)
            else:
                buckets = {domain: indices.copy() for domain, indices in sorted(self.domains.items())}
                for indices in buckets.values():
                    rng.shuffle(indices)
                domain_order = list(buckets)
                rng.shuffle(domain_order)
                order = []
                while any(buckets.values()):
                    for domain in domain_order:
                        if buckets[domain]:
                            order.append(buckets[domain].pop())
            self._epoch, self._order = epoch, order
        return self._order

    def rows(self, start, count):
        result = []
        while count:
            epoch, offset = divmod(start, len(self.dataset))
            selected = self.order(epoch)[offset:offset+count]
            result.extend(self.dataset[index] for index in selected)
            start += len(selected)
            count -= len(selected)
        return result

    def state_dict(self):
        return {"seed": self.seed, "cursor": self.cursor, "size": len(self.dataset),
                "stratified": self.domains is not None}

    def load_state_dict(self, state):
        expected = {"seed": self.seed, "size": len(self.dataset), "stratified": self.domains is not None}
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Sampler state mismatch for {key}")
        self.cursor = int(state["cursor"])

    def rows_for_rank(self, global_count, rank, world):
        count, offset = distributed_slice(global_count, rank, world)
        return self.rows(self.cursor+offset, count)

    def advance(self, global_count):
        self.cursor += global_count


def distributed_slice(total, rank, world):
    """Balanced deterministic slice; ranks may differ by one item."""
    quotient, remainder = divmod(total, world)
    count = quotient+int(rank < remainder)
    offset = rank*quotient+min(rank, remainder)
    return count, offset


def save_checkpoint(model, tokenizer, optimizer, args, step, world_size, sampler_state):
    directory = Path(args.output_dir) / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save({"step": step, "world_size": world_size, "args": vars(args), "samplers": sampler_state,
                "task_state": optimizer.state_dict() if optimizer else None}, directory / "training.pt")
    # Completion marker written last; an interrupted checkpoint cannot be resumed.
    (directory / "complete.json").write_text(json.dumps({"step": step}) + "\n")
    return str(directory)


def validate_hub_access(args):
    from huggingface_hub import HfApi

    token = os.environ.get(args.hub_token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in {args.hub_token_env}")
    if not args.hub_repo_id:
        raise RuntimeError("Set --hub-repo-id or GRIT_HUB_REPO_ID before the full run")
    try:
        HfApi().whoami(token=token)
    except Exception as error:
        raise RuntimeError(f"Hugging Face token check failed ({type(error).__name__})") from error
    return args.hub_repo_id


def push_checkpoint(directory, args):
    """Best-effort model upload after the complete local checkpoint exists."""
    from huggingface_hub import HfApi

    token = os.environ.get(args.hub_token_env)
    try:
        api = HfApi(token=token)
        api.create_repo(args.hub_repo_id, exist_ok=True, private=True)
        api.upload_folder(
            repo_id=args.hub_repo_id, folder_path=directory,
            path_in_repo=Path(directory).name, ignore_patterns=["training.pt"],
            commit_message=f"GRIT checkpoint {Path(directory).name}",
        )
        return None
    except Exception as error:
        message = str(error).replace(token, "<redacted>") if token else str(error)
        return f"{type(error).__name__}: {message}"


@torch.no_grad()
def base_kl_at_current_policy(model, tokenizer, rows, *, max_length, top_k, device):
    """Measure the stored-base KL before the first predictor changes the policy."""
    total = count = peak = 0.0
    for row in rows:
        ids, attention, mask, base_ids, base_logp, base_tail = [value.to(device) for value in
            tokenize_preserve_batch(tokenizer, [row], max_length=max_length, top_k=top_k)]
        logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
        policy_logp = coarsened_policy_log_probs(logits, base_ids)
        anchor_logp = coarsened_base_log_probs(base_logp, base_tail)
        kl = kl_from_log_probs(policy_logp, anchor_logp)[mask].clamp_min(0)
        total += float(kl.sum())
        count += kl.numel()
        if kl.numel():
            peak = max(peak, float(kl.max()))
    sums = torch.tensor([total, count], dtype=torch.float64, device=device)
    maximum = torch.tensor(peak, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(sums[0]/sums[1].clamp_min(1)), float(maximum)


@torch.no_grad()
def run_safety_validation(*, rollout, model, tokenizer, safety, safety_tokenizer,
                          evaluation, args, rank, world, device, step):
    """Generate one greedy response per held-out prompt and score all responses."""
    started = time.monotonic()
    prompts = [str(value).strip() for value in evaluation["prompt"]]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Safety evaluation requires nonempty prompt values")

    def generate():
        prompt_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True,
            )[-args.max_prompt_length:]
            for prompt in prompts
        ]
        if rollout.version != step:
            rollout.sync(model, step)
        return rollout.generate(
            prompt_ids, version=step, n=1, max_tokens=args.max_response_length,
            temperature=0.0, top_p=1.0, seed=args.seed,
        )

    groups = broadcast_call(rank, generate)
    if len(groups) != len(prompts) or any(len(group) != 1 for group in groups):
        raise RuntimeError("Safety validation rollout count mismatch")
    responses = [group[0] for group in groups]
    count, offset = distributed_slice(len(prompts), rank, world)
    local_prompts = prompts[offset:offset+count]
    local_responses = responses[offset:offset+count]
    response_texts = tokenizer.batch_decode(
        [response["token_ids"] for response in local_responses], skip_special_tokens=True,
    )

    rewards = torch.empty(0, device=device, dtype=torch.float32)
    labels = []
    parse_errors = retries = 0.0
    safety.to(device)
    try:
        reward_parts = []
        for index in range(0, count, args.reward_batch_size):
            part, part_labels, diagnostics = score_safety_rewards(
                safety, safety_tokenizer,
                local_prompts[index:index+args.reward_batch_size],
                response_texts[index:index+args.reward_batch_size],
                max_length=args.max_prompt_length+args.max_response_length+512,
                max_new_tokens=args.safety_max_new_tokens, device=device,
                attempts=args.safety_attempts,
            )
            reward_parts.append(part)
            labels.extend(part_labels)
            parse_errors += diagnostics["parse_errors"]
            retries += diagnostics["retries"]
        if reward_parts:
            rewards = torch.cat(reward_parts)
    finally:
        safety.to("cpu")
        torch.cuda.empty_cache()

    totals = torch.tensor([
        float(rewards.sum()), float(rewards.square().sum()), rewards.numel(),
        int((rewards < 0).sum()), parse_errors, retries,
        sum(len(response["token_ids"]) for response in local_responses),
    ], device=device, dtype=torch.float64)
    elapsed = torch.tensor(time.monotonic()-started, device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    total_count = totals[2].clamp_min(1)
    reward_mean = totals[0]/total_count
    metrics = {
        "validation_ran": True,
        "validation_count": int(totals[2]),
        "validation_reward_mean": float(reward_mean),
        "validation_reward_std": float((totals[1]/total_count-reward_mean.square()).clamp_min(0).sqrt()),
        "validation_unsafe_fraction": float(totals[3]/total_count),
        "validation_parse_errors": int(totals[4]),
        "validation_retries": int(totals[5]),
        "validation_response_tokens": int(totals[6]),
        "validation_seconds": float(elapsed),
    }
    local_records = [
        {"step": step, "index": offset+index, "prompt": prompt, "response": response,
         "reward": float(reward), "safety_label": label}
        for index, (prompt, response, reward, label) in enumerate(zip(
            local_prompts, response_texts, rewards.cpu().tolist(), labels, strict=True
        ))
    ]
    gathered = [None]*world
    if world > 1:
        dist.all_gather_object(gathered, local_records)
    else:
        gathered[0] = local_records
    records = [record for part in gathered for record in part] if rank == 0 else None
    return metrics, records


def validate_gpu_layout(*, visible: list[str], rollout_gpu: str, world: int, colocated: bool) -> None:
    if not visible or not visible[0]:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to actor GPUs")
    if colocated:
        if world != 1 or visible != [rollout_gpu]:
            raise ValueError("Colocated rollout requires one actor on the same single visible GPU")
    elif rollout_gpu in visible:
        raise ValueError("--rollout-gpu must be separate from actor GPUs")


def train(args):
    profile = profile_for_policy(args.model_path)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    validate_gpu_layout(visible=visible, rollout_gpu=args.rollout_gpu, world=world,
                        colocated=args.allow_colocated_rollout)
    if not torch.cuda.is_available():
        raise RuntimeError("Training with vLLM requires Linux/CUDA; run CPU correctness tests locally")
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("This simple trainer supports one host only")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    broadcast_call(rank, lambda: validate_hub_access(args))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base_kwargs = dict(revision=args.model_revision, trust_remote_code=args.trust_remote_code)
    base_revision = getattr(AutoConfig.from_pretrained(args.model_path, **base_kwargs), "_commit_hash", None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **base_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = None
    if args.resume:
        resume = Path(args.resume)
        if not (resume / "complete.json").exists():
            raise ValueError("Checkpoint missing completion marker")
        checkpoint = torch.load(resume / "training.pt", map_location="cpu", weights_only=False)
        allowed_changes = {"resume", "max_steps", "save_steps", "validation_steps",
                           "output_dir", "rollout_gpu", "rollout_timeout"}
        changed = [key for key, value in vars(args).items()
                   if key not in allowed_changes and checkpoint["args"].get(key) != value]
        if checkpoint["world_size"] != world or changed:
            raise ValueError(f"Resume requires the same training/data settings and world size; changed: {changed}")
    model = AutoModelForCausalLM.from_pretrained(
        args.resume or args.model_path, torch_dtype=torch.float32, attn_implementation="sdpa",
        **({"trust_remote_code": args.trust_remote_code} if args.resume else base_kwargs),
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != profile.policy_parameters:
        raise ValueError("Loaded policy parameter count does not match the pinned checkpoint")
    model.config.use_cache = False
    # Disable dropout consistently for old-policy comparison and central difference.
    model.eval()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
            if isinstance(getattr(layer, "attention_dropout", None), (float, int)):
                layer.attention_dropout = 0.0
        for key in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob"):
            if hasattr(model.config, key):
                setattr(model.config, key, 0.0)
    safety_tokenizer = AutoTokenizer.from_pretrained(args.safety_model_path, padding_side="left",
                                                    revision=args.safety_model_revision,
                                                    trust_remote_code=args.trust_remote_code)
    if safety_tokenizer.pad_token_id is None:
        safety_tokenizer.pad_token = safety_tokenizer.eos_token
    safety = AutoModelForCausalLM.from_pretrained(
        args.safety_model_path, revision=args.safety_model_revision, torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    ).eval().requires_grad_(False)
    if sum(parameter.numel() for parameter in safety.parameters()) != profile.safety_parameters:
        raise ValueError("Loaded reward-model parameter count does not match the pinned checkpoint")
    optimizer = AdamWDirectionPreconditioner(eps=args.adam_eps, weight_decay=args.weight_decay) if args.task_optimizer == "adamw" else None
    start = checkpoint["step"] if checkpoint else 0
    if start >= args.max_steps:
        raise ValueError("--max-steps must exceed the resumed checkpoint step")
    if checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint["task_state"])
    payload = torch.load(args.projectors_path, map_location="cpu", weights_only=False)
    if payload.get("base_revision") and payload["base_revision"] != base_revision:
        raise ValueError("Projector base revision mismatch")
    projectors = payload.get("projectors", payload)
    protected = [(name, layer) for name, layer in model.named_modules()
                 if isinstance(layer, torch.nn.Linear) and args.module_pattern in name]
    if not protected or any(name not in projectors for name, _ in protected):
        raise ValueError("Projector artifact does not cover the requested protected Linear modules")
    from grit.projection import CompactProjector, projector_dimension
    for name, layer in protected:
        if projector_dimension(projectors[name]) != layer.in_features:
            raise ValueError(f"Projector shape mismatch: {name}")
        if isinstance(projectors[name], CompactProjector):
            raw = projectors[name]
            projectors[name] = CompactProjector(raw.basis.to(device=device, dtype=torch.float32),
                                                 raw.complement, raw.dimension)
        else:
            projectors[name] = projectors[name].to(device=device, dtype=torch.float32)
    task = load_dataset("parquet", data_files=args.task_file, split="train")
    preserve = load_dataset("parquet", data_files=args.preserve_file, split="train") if args.lambda_pres else None
    evaluation = load_dataset("parquet", data_files=args.evaluation_file, split="train") if args.evaluation_file else None
    if evaluation is not None:
        if "prompt" not in evaluation.column_names or not len(evaluation):
            raise ValueError("Safety evaluation file needs a nonempty prompt column")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in evaluation["prompt"]):
            raise ValueError("Safety evaluation prompts must be nonempty strings")
    if preserve is not None:
        from scripts.preservation_data import validate_monitoring_split
        validate_monitoring_split(payload, preserve, evaluation)
    if preserve is not None and "base_revision" in preserve.column_names:
        if set(preserve["base_revision"]) != {base_revision}:
            raise ValueError("Preservation base revision mismatch; use the recorded --model-revision")
        manifest_path = Path(args.preserve_file).parent / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Preservation manifest missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("base_top_k") != args.top_k:
            raise ValueError("Preservation top-k manifest mismatch")
        if manifest.get("base_statistics_dtype") != args.base_statistics_dtype:
            raise ValueError("Preservation base-statistics dtype does not match --base-statistics-dtype")
        if "base_statistics_dtype" not in preserve.column_names:
            raise ValueError("Preservation rows are missing base-statistics dtype")
        if set(preserve["base_statistics_dtype"]) != {args.base_statistics_dtype}:
            raise ValueError("Preservation rows and manifest disagree on base-statistics dtype")
    task_sampler = EpochSampler(task, args.seed)
    pres_sampler = EpochSampler(preserve, args.seed+1, stratify=True) if preserve is not None else None
    if checkpoint:
        task_sampler.load_state_dict(checkpoint["samplers"]["task"])
        if pres_sampler is not None:
            pres_sampler.load_state_dict(checkpoint["samplers"]["preservation"])
    output = Path(args.output_dir)
    if not checkpoint:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output.mkdir(parents=True, exist_ok=True)
    rollout = None
    try:
        def start_rollout():
            nonlocal rollout
            rollout = VLLMRollout(model=args.model_path, gpu=args.rollout_gpu, revision=args.model_revision,
                                 dtype=args.rollout_dtype, seed=args.seed,
                                 max_model_len=args.max_prompt_length+args.max_response_length,
                                 memory_utilization=args.rollout_memory_utilization,
                                 timeout=args.rollout_timeout, trust_remote_code=args.trust_remote_code)
        broadcast_call(rank, start_rollout)
        session_started = time.monotonic()
        metrics_path = output / "metrics.jsonl"
        previous_metrics = []
        if rank == 0 and metrics_path.exists():
            previous_metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        progress = tqdm(range(start+1, args.max_steps+1), total=args.max_steps, initial=start,
                        desc="GRIT", unit="step", dynamic_ncols=True, disable=rank != 0)
        for step in progress:
            tick = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            rows = task_sampler.rows_for_rank(args.task_batch_size, rank, world)
            prompts = [row.get("raw_prompt") or row["prompt"] for row in rows]
            prompt_ids = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                          tokenize=True, add_generation_prompt=True)[-args.max_prompt_length:] for prompt in prompts]
            all_ids = [None] * world
            if world > 1:
                dist.all_gather_object(all_ids, prompt_ids)
            else:
                all_ids[0] = prompt_ids
            def generate():
                if rollout.version != step-1:
                    rollout.sync(model, step-1)
                return rollout.generate([ids for group in all_ids for ids in group], version=step-1,
                                        n=args.generations, max_tokens=args.max_response_length,
                                        temperature=1.0, top_p=1.0, seed=args.seed+step)
            groups = broadcast_call(rank, generate)
            prompt_counts = [len(ids) for ids in all_ids]
            offset = sum(prompt_counts[:rank])
            groups = groups[offset:offset+prompt_counts[rank]]
            rollout_seconds = time.monotonic()-tick
            responses = [item for group in groups for item in group]
            texts = tokenizer.batch_decode([r["token_ids"] for r in responses], skip_special_tokens=True)
            expanded_prompts = [p for p in prompts for _ in range(args.generations)]
            reward_tick = time.monotonic()
            safety.to(device)
            reward_parts = []
            reward_parse_errors = reward_retries = 0.0
            for index in range(0, len(texts), args.reward_batch_size):
                rewards, _, reward_diag = score_safety_rewards(safety, safety_tokenizer,
                    expanded_prompts[index:index+args.reward_batch_size], texts[index:index+args.reward_batch_size],
                    max_length=args.max_prompt_length+args.max_response_length+512,
                    max_new_tokens=args.safety_max_new_tokens, device=device, attempts=args.safety_attempts)
                reward_parts.append(rewards)
                reward_parse_errors += reward_diag["parse_errors"]
                reward_retries += reward_diag["retries"]
            rewards = torch.cat(reward_parts)
            safety.to("cpu")
            torch.cuda.empty_cache()
            reward_seconds = time.monotonic()-reward_tick
            advantages = group_advantages(rewards, args.generations)
            global_responses = args.task_batch_size*args.generations
            task_diag = {
                "logprob_abs_sum": 0.0, "logprob_abs_max": 0.0, "logprob_count": 0,
                "ratio_sum": 0.0, "ratio_square_sum": 0.0, "ratio_count": 0,
                "ratio_min": float("inf"), "ratio_max": 0.0, "clip_count": 0,
                "response_tokens": 0, "response_tokens_min": float("inf"), "response_tokens_max": 0,
            }
            task_pass = 0
            # Store tokens/logprobs on CPU; only one response's activations are live.
            def task_losses():
                nonlocal task_pass
                task_pass += 1
                for index, response in enumerate(responses):
                    if task_pass > 1 and advantages[index] == 0:
                        continue
                    batch = response_batch(prompt_ids[index//args.generations], response["token_ids"], device)
                    values, mask = token_log_probs(model, batch)
                    old = torch.tensor([response["log_probs"]], device=device, dtype=values.dtype)
                    # log-probs only cover sampled response tokens, not prompt positions.
                    response_values = values[mask].reshape(1, -1)
                    if task_pass == 1:
                        error = (response_values.detach()-old).abs()
                        ratio = torch.exp(response_values.detach()-old)
                        count = old.numel()
                        task_diag["logprob_abs_sum"] += float(error.sum())
                        task_diag["logprob_abs_max"] = max(task_diag["logprob_abs_max"], float(error.max()))
                        task_diag["logprob_count"] += count
                        task_diag["ratio_sum"] += float(ratio.sum())
                        task_diag["ratio_square_sum"] += float(ratio.square().sum())
                        task_diag["ratio_count"] += count
                        task_diag["ratio_min"] = min(task_diag["ratio_min"], float(ratio.min()))
                        task_diag["ratio_max"] = max(task_diag["ratio_max"], float(ratio.max()))
                        task_diag["clip_count"] += int((ratio.sub(1).abs() > args.clip_ratio).sum())
                        task_diag["response_tokens"] += count
                        task_diag["response_tokens_min"] = min(task_diag["response_tokens_min"], count)
                        task_diag["response_tokens_max"] = max(task_diag["response_tokens_max"], count)
                    yield grpo_loss(response_values, old, advantages[index:index+1],
                                    torch.ones_like(response_values, dtype=torch.bool), args.clip_ratio) / global_responses
            pres_rows = pres_sampler.rows_for_rank(args.preserve_batch_size, rank, world) if pres_sampler is not None else []
            base_kl_mean = base_kl_max = None
            if step == 1 and pres_sampler is not None:
                base_kl_mean, base_kl_max = base_kl_at_current_policy(
                    model, tokenizer, pres_rows, max_length=args.max_preserve_length,
                    top_k=args.top_k, device=device,
                )
            local_pres_tokens = sum(
                max(min(len(row["input_ids"]), args.max_preserve_length)-int(row["response_start"]), 0)
                for row in pres_rows
            )
            global_pres_tokens = torch.tensor(float(local_pres_tokens), device=device)
            if world > 1:
                dist.all_reduce(global_pres_tokens)
            if pres_rows and global_pres_tokens.item() <= 0:
                raise ValueError("No response tokens in global preservation batch")
            pres_diag = {"count": 0, "kl_sum": 0.0, "kl_max": 0.0, "projected_kl_max": 0.0,
                         "violations": 0, "eta_sum": 0.0, "eta_max": 0.0,
                         "values": [], "eta_values": [], "coverage": [], "domains": {}}
            def preservation_losses():
                for row in pres_rows:
                    ids, attention, mask, base_ids, base_logp, base_tail = [v.to(device) for v in tokenize_preserve_batch(
                        tokenizer, [row], max_length=args.max_preserve_length, top_k=args.top_k)]
                    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
                    result = preservation_kl_loss(
                        logits, epsilon_pres=args.epsilon_pres, response_mask=mask,
                        base_topk_ids=base_ids, base_topk_log_probs=base_logp,
                        base_log_tail=base_tail, reduction="sum",
                    )
                    active = mask.bool()
                    token_kl = result.projection.token_kl[active].detach()
                    projected_kl = result.projection.projected_kl[active].detach()
                    eta = result.projection.eta[active].detach()
                    violations = result.projection.violation_mask[active]
                    if token_kl.numel():
                        pres_diag["count"] += token_kl.numel()
                        pres_diag["kl_sum"] += float(token_kl.sum())
                        pres_diag["kl_max"] = max(pres_diag["kl_max"], float(token_kl.max()))
                        pres_diag["projected_kl_max"] = max(pres_diag["projected_kl_max"], float(projected_kl.max()))
                        pres_diag["violations"] += int(violations.sum())
                        pres_diag["eta_sum"] += float(eta.sum())
                        pres_diag["eta_max"] = max(pres_diag["eta_max"], float(eta.max()))
                        values = token_kl.float().cpu().tolist()
                        eta_values = eta.float().cpu().tolist()
                        coverage = (1-base_tail[active].exp()).float().cpu().tolist()
                        pres_diag["values"].extend(values)
                        pres_diag["eta_values"].extend(eta_values)
                        pres_diag["coverage"].extend(coverage)
                        domain = pres_diag["domains"].setdefault(
                            row["domain"], {"kl": [], "violations": 0, "tokens": 0,
                                            "contexts": 0, "violating_contexts": 0})
                        domain["kl"].extend(values)
                        domain["violations"] += int(violations.sum())
                        domain["tokens"] += len(values)
                        domain["contexts"] += 1
                        domain["violating_contexts"] += int(bool(violations.any()))
                    yield result.loss / global_pres_tokens
            update_tick = time.monotonic()
            check_fd = bool(args.fd_check_interval and args.use_curvature and
                            (step == start+1 or step % args.fd_check_interval == 0))
            metrics = grit_step(model, task_losses, preservation_losses if pres_rows else None, projectors,
                                lr=args.lr, lambda_pres=args.lambda_pres, optimizer=optimizer,
                                use_curvature=args.use_curvature, rho=args.fd_radius,
                                reduce_gradients=sum_gradients,
                                module_filter=lambda n, m: isinstance(m, torch.nn.Linear) and args.module_pattern in n,
                                check_curvature=check_fd, gradient_topk=args.gradient_topk,
                                projector_relaxation=args.projector_relaxation,
                                shared_vector_norm=shared_vector_norm)
            update_seconds = time.monotonic()-update_tick
            # Loss callbacks are global-count weighted; SUM reconstructs global losses.
            scalar = torch.tensor([
                metrics["task_loss"], metrics["preservation_loss"], metrics["fd_plus_loss"], metrics["fd_minus_loss"],
                float(rewards.sum()), float(rewards.square().sum()), rewards.numel(), int((rewards < 0).sum()),
                float(advantages.abs().sum()), float(advantages.square().sum()), advantages.numel(),
                task_diag["logprob_abs_sum"], task_diag["logprob_count"], task_diag["ratio_sum"],
                task_diag["ratio_square_sum"], task_diag["ratio_count"], task_diag["clip_count"],
                task_diag["response_tokens"], len(responses), pres_diag["count"], pres_diag["kl_sum"],
                pres_diag["violations"], pres_diag["eta_sum"],
                reward_parse_errors, reward_retries,
                int(advantages.reshape(-1, args.generations).abs().sum(-1).eq(0).sum()),
                advantages.numel()//args.generations,
            ], device=device, dtype=torch.float64)
            maxima = torch.tensor([
                task_diag["logprob_abs_max"], task_diag["ratio_max"], task_diag["response_tokens_max"],
                pres_diag["kl_max"], pres_diag["projected_kl_max"], pres_diag["eta_max"],
            ], device=device, dtype=torch.float64)
            minima = torch.tensor([task_diag["ratio_min"], task_diag["response_tokens_min"]],
                                  device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(scalar)
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
                dist.all_reduce(minima, op=dist.ReduceOp.MIN)
            gathered_pres = [None]*world
            local_pres = {key: pres_diag[key] for key in ("values", "eta_values", "coverage", "domains")}
            if world > 1:
                dist.all_gather_object(gathered_pres, local_pres)
            else:
                gathered_pres[0] = local_pres
            all_kl = [value for part in gathered_pres for value in part["values"]]
            all_eta = [value for part in gathered_pres for value in part["eta_values"]]
            all_coverage = [value for part in gathered_pres for value in part["coverage"]]
            def percentile(values, q):
                if not values:
                    return 0.0
                ordered = sorted(values)
                return float(ordered[round((len(ordered)-1)*q)])
            domain_metrics = {}
            for part in gathered_pres:
                for domain, values in part["domains"].items():
                    target = domain_metrics.setdefault(
                        domain, {"kl": [], "violations": 0, "tokens": 0,
                                 "contexts": 0, "violating_contexts": 0})
                    target["kl"].extend(values["kl"])
                    target["violations"] += values["violations"]
                    target["tokens"] += values["tokens"]
                    target["contexts"] += values["contexts"]
                    target["violating_contexts"] += values["violating_contexts"]
            violating_contexts = sum(values["violating_contexts"] for values in domain_metrics.values())
            context_count = sum(values["contexts"] for values in domain_metrics.values())
            domain_metrics = {
                domain: {"kl_mean": sum(values["kl"])/max(len(values["kl"]), 1),
                         "kl_p95": percentile(values["kl"], 0.95),
                         "kl_max": max(values["kl"], default=0.0),
                         "token_violation_fraction": values["violations"]/max(values["tokens"], 1),
                         "context_violation_fraction": values["violating_contexts"]/max(values["contexts"], 1)}
                for domain, values in domain_metrics.items()
            }
            reward_count = scalar[6].clamp_min(1)
            reward_mean = scalar[4]/reward_count
            ratio_count = scalar[15].clamp_min(1)
            ratio_mean = scalar[13]/ratio_count
            pres_count = scalar[19].clamp_min(1)
            metrics.update(
                step=step, task_loss=float(scalar[0]), preservation_loss=float(scalar[1]),
                fd_plus_loss=float(scalar[2]), fd_minus_loss=float(scalar[3]),
                reward=float(reward_mean), reward_mean=float(reward_mean),
                reward_std=float((scalar[5]/reward_count-reward_mean.square()).clamp_min(0).sqrt()),
                unsafe_fraction=float(scalar[7]/reward_count),
                advantage_abs_mean=float(scalar[8]/scalar[10].clamp_min(1)),
                advantage_std=float((scalar[9]/scalar[10].clamp_min(1)).sqrt()),
                rollout_actor_logprob_mae=float(scalar[11]/scalar[12].clamp_min(1)),
                rollout_actor_logprob_max=float(maxima[0]), ratio_mean=float(ratio_mean),
                ratio_std=float((scalar[14]/ratio_count-ratio_mean.square()).clamp_min(0).sqrt()),
                ratio_min=float(minima[0]), ratio_max=float(maxima[1]),
                ratio_clipped_fraction=float(scalar[16]/ratio_count),
                response_tokens_mean=float(scalar[17]/scalar[18].clamp_min(1)),
                response_tokens_min=int(minima[1]), response_tokens_max=int(maxima[2]),
                trust_token_count=int(scalar[19]), trust_kl_mean=float(scalar[20]/pres_count),
                trust_kl_max=float(maxima[3]), trust_projected_kl_max=float(maxima[4]),
                trust_violation_fraction=float(scalar[21]/pres_count),
                trust_context_violation_fraction=violating_contexts/max(context_count, 1),
                trust_eta_mean=float(scalar[22]/pres_count), trust_eta_max=float(maxima[5]),
                trust_kl_p95=percentile(all_kl, 0.95), trust_eta_p95=percentile(all_eta, 0.95),
                trust_by_domain=domain_metrics,
                top64_base_coverage_mean=sum(all_coverage)/max(len(all_coverage), 1),
                top64_base_coverage_p5=percentile(all_coverage, 0.05),
                reward_parse_errors=int(scalar[23]), reward_retries=int(scalar[24]),
                zero_advantage_group_fraction=float(scalar[25]/scalar[26].clamp_min(1)),
                global_responses=global_responses, rollout_seconds=rollout_seconds,
                rollout_tokens_per_second=float(scalar[17]/max(rollout_seconds, 1e-9)),
                reward_seconds=reward_seconds, update_seconds=update_seconds,
                lambda_pres=args.lambda_pres, epsilon_pres=args.epsilon_pres,
                base_kl_at_theta_before_mean=base_kl_mean,
                base_kl_at_theta_before_max=base_kl_max,
                peak_vram_gb=torch.cuda.max_memory_allocated(device)/1e9,
            )
            task_sampler.advance(args.task_batch_size)
            if pres_sampler is not None:
                pres_sampler.advance(args.preserve_batch_size)
            metrics.update(
                validation_ran=False, validation_count=0,
                validation_reward_mean=None, validation_reward_std=None,
                validation_unsafe_fraction=None, validation_parse_errors=0,
                validation_retries=0, validation_response_tokens=0,
                validation_seconds=0.0,
            )
            if args.validation_steps and step % args.validation_steps == 0:
                validation_metrics, validation_records = run_safety_validation(
                    rollout=rollout, model=model, tokenizer=tokenizer,
                    safety=safety, safety_tokenizer=safety_tokenizer,
                    evaluation=evaluation, args=args, rank=rank, world=world,
                    device=device, step=step,
                )
                metrics.update(validation_metrics)
                if rank == 0:
                    validation_dir = output / "validation"
                    validation_dir.mkdir(exist_ok=True)
                    destination = validation_dir / f"step_{step:06d}.jsonl"
                    temporary = destination.with_suffix(".jsonl.tmp")
                    with temporary.open("w", encoding="utf-8") as handle:
                        for record in validation_records:
                            handle.write(json.dumps(record, ensure_ascii=False)+"\n")
                    temporary.replace(destination)
            checkpoint_tick = time.monotonic()
            hub_push_error = None
            if step % args.save_steps == 0 or step == args.max_steps:
                sampler_state = {"task": task_sampler.state_dict(),
                                 "preservation": pres_sampler.state_dict() if pres_sampler else None}
                checkpoint_dir = broadcast_call(
                    rank, lambda: save_checkpoint(model, tokenizer, optimizer, args, step, world, sampler_state)
                )
                if rank == 0:
                    hub_push_error = push_checkpoint(checkpoint_dir, args)
            checkpoint_seconds = time.monotonic()-checkpoint_tick
            session_elapsed = time.monotonic()-session_started
            metrics.update(checkpoint_seconds=checkpoint_seconds, step_seconds=time.monotonic()-tick,
                           session_elapsed_seconds=session_elapsed,
                           eta_seconds=(args.max_steps-step)*session_elapsed/max(step-start, 1))
            metrics["hub_push_error"] = hub_push_error
            metrics["alerts"] = step_alerts(metrics, epsilon_pres=args.epsilon_pres)
            if rank == 0:
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(metrics)+"\n")
                previous_metrics.append(metrics)
                summary = summarize_metrics(previous_metrics)
                temporary = output / "diagnostics_summary.json.tmp"
                temporary.write_text(json.dumps(summary, indent=2)+"\n")
                temporary.replace(output / "diagnostics_summary.json")
                progress.set_postfix(reward=f"{metrics['reward_mean']:.3f}",
                                     g=f"{metrics['task_grad_norm']:.2e}",
                                     v=f"{metrics['preservation_grad_norm']:.2e}",
                                     delta=f"{metrics['final_delta_norm']:.2e}",
                                     kl=f"{metrics['trust_violation_fraction']:.1%}",
                                     roll=f"{rollout_seconds:.1f}s", rew=f"{reward_seconds:.1f}s",
                                     task=f"{metrics['task_backward_seconds']:.1f}s",
                                     pred=f"{metrics['predictor_seconds']:.1f}s",
                                     pres=f"{metrics['preservation_backward_seconds']:.1f}s",
                                     curv=f"{metrics['central_difference_seconds']:.1f}s",
                                     upd=f"{metrics['final_update_seconds']:.1f}s")
                if metrics["alerts"]:
                    progress.write(f"step {step} alerts: {', '.join(metrics['alerts'])}")
        if rank == 0:
            progress.write("run diagnostics: " + json.dumps(summary, sort_keys=True))
    finally:
        if rollout:
            rollout.close()


def main():
    args = parse_args()
    try:
        try:
            train(args)
        except torch.OutOfMemoryError as error:
            raise RuntimeError("GRIT early stop: CUDA out of memory; no partial step was committed") from error
        except FloatingPointError as error:
            raise RuntimeError(f"GRIT early stop: {error}") from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
