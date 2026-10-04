import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from grit.data import parse_safety_reward, score_safety_rewards
from grit.grpo import grpo_loss, group_advantages, response_batch, response_microbatch, token_log_probs
from grit.optimizer_delta import AdamWDirectionPreconditioner
from grit.diagnostics import step_alerts, summarize_metrics
from grit.rollout import VLLMRollout, _worker, load_policy_weights
from grit.step import accumulate, grit_step
from scripts.train_grit import (EpochSampler, base_kl_at_current_policy, distributed_slice,
                                parse_args, push_checkpoint, run_safety_validation, validate_gpu_layout,
                                validate_hub_access)


def example():
    torch.manual_seed(42)
    model = nn.Linear(2, 2, bias=False)
    x = torch.randn(3, 2)
    y = torch.randn(3, 2)
    unit = torch.tensor([1., 2.]) / 5**0.5
    p = unit[:, None] @ unit[None, :]
    return model, x, y, p


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
@pytest.mark.parametrize("rho", [0.005, 0.01])
@pytest.mark.parametrize("relaxation", [0.0, 0.05, 0.2])
def test_full_correction_matches_unrolled_autograd(kind, rho, relaxation):
    model, x, y, p = example()
    initial = model.weight.detach().clone()
    optimizer = AdamWDirectionPreconditioner(eps=0.03, weight_decay=0.13) if kind == "adamw" else None
    if optimizer:
        optimizer.directions(list(model.named_parameters()), {"weight": torch.randn_like(initial)})
    theta = initial.clone().requires_grad_()
    task = ((x @ theta.T-y)**2).mean()
    g, = torch.autograd.grad(task, theta, create_graph=True)
    lr, lam = 0.07, 0.3
    if optimizer:
        state = optimizer.state["weight"]
        t = state["step"]+1
        m = (0.9*state["exp_avg"]+0.1*g)/(1-0.9**t)
        s = (0.999*state["exp_avg_sq"]+0.001*g.square())/(1-0.999**t)
        direction = -m/(s.sqrt()+optimizer.eps)-optimizer.weight_decay*theta
    else:
        direction = -g
    q = p + relaxation*(torch.eye(p.shape[0])-p)
    delta = lr * direction @ q
    predictor = theta+delta
    correction, = torch.autograd.grad(((predictor-0.8)**2).mean(), theta)
    expected = initial+delta.detach()-lr*lam*correction

    def tasks():
        for xx, yy in zip(x, y):
            yield ((model(xx)-yy)**2).mean()/len(x)

    metrics = grit_step(model, tasks, lambda: iter([((model.weight-0.8)**2).mean()]),
                        {"": p}, lr=lr, lambda_pres=lam, optimizer=optimizer, use_curvature=True,
                        rho=rho, module_filter=lambda n, m: True,
                        projector_relaxation=relaxation)
    torch.testing.assert_close(model.weight, expected, atol=3e-6, rtol=2e-5)
    assert metrics["curvature_active"] == 1
    if optimizer:
        assert optimizer.state["weight"]["step"] == 2


def test_zero_preservation_reuses_delta_and_commits_once():
    model, x, y, p = example()
    parameters = list(model.named_parameters())
    optimizer = AdamWDirectionPreconditioner(weight_decay=0.1)
    loss = ((model(x)-y)**2).mean()
    g, = torch.autograd.grad(loss, model.weight)
    raw, _ = optimizer.prepare(parameters, {"weight": g})
    before = model.weight.detach().clone()
    calls = []
    def tasks():
        calls.append(True)
        yield ((model(x)-y)**2).mean()
    metrics = grit_step(model, tasks, lambda: iter([model.weight.sum()*0]), {"": p},
                        lr=0.03, lambda_pres=1, optimizer=optimizer, use_curvature=True,
                        module_filter=lambda n, m: True)
    torch.testing.assert_close(model.weight, before+0.03*(raw["weight"]@p), atol=0, rtol=0)
    assert len(calls) == 1 and metrics["curvature_active"] == 0
    assert optimizer.state["weight"]["step"] == 1


def test_step_emits_gradient_and_fd_diagnostics():
    model, x, y, p = example()
    metrics = grit_step(
        model,
        lambda: iter([((model(x)-y)**2).mean()]),
        lambda: iter([((model.weight-0.8)**2).mean()]),
        {"": p}, lr=0.03, lambda_pres=0.2, use_curvature=True, rho=1e-3,
        check_curvature=True, gradient_topk=2, module_filter=lambda n, m: True,
        projector_relaxation=0.2,
    )
    assert metrics["task_grad_norm"] > 0
    assert metrics["raw_task_direction_norm"] > 0
    assert 0 <= metrics["projection_retained_fraction"] <= 1.000001
    assert metrics["final_delta_norm"] > 0
    assert metrics["fd_relative_error_rho_vs_2rho"] is not None
    assert metrics["fd_one_sided_relative_error"] is not None
    assert metrics["projector_relaxation"] == 0.2
    assert metrics["projection_retained_fraction"] >= metrics["hard_projection_retained_fraction"]
    assert metrics["relaxation_added_direction_norm"] > 0
    assert 0 <= metrics["task_gradient_q_retention"] <= 1.000001
    assert metrics["task_delta_outside_null_norm"] == pytest.approx(
        0.03 * metrics["relaxation_added_direction_norm"]
    )
    assert len(metrics["gradient_top"]["task"]) <= 2


def test_diagnostics_flags_and_summarizes_failures():
    row = {
        "step": 1, "reward_std": 0.0, "task_grad_norm": 0.0,
        "rollout_actor_logprob_mae": 0.03, "raw_task_direction_norm": 1.0,
        "projection_retained_fraction": 0.0, "correction_to_task_ratio": 11.0,
        "curvature_to_preservation_ratio": 0.0, "relative_update_norm": 0.0,
        "trust_projected_kl_max": 2e-3, "fd_relative_error_rho_vs_2rho": 0.3,
        "base_kl_at_theta_before_max": 2e-4,
        "lambda_pres": 1.0, "correction_active": 0.0, "curvature_active": 1.0,
        "step_seconds": 2.0, "alerts": [],
    }
    row["alerts"] = step_alerts(row, epsilon_pres=1e-3)
    assert "reward_collapse" in row["alerts"]
    assert "central_difference_radius_unstable" in row["alerts"]
    assert "base_anchor_precision_floor" in row["alerts"]
    summary = summarize_metrics([row])
    assert summary["steps"] == 1
    assert summary["total_elapsed_seconds"] == 2.0
    assert summary["base_kl_at_theta_before_max"] == pytest.approx(2e-4)
    assert "preservation_never_activated" in summary["alerts"]


def test_base_kl_at_theta_before_detects_anchor_mismatch(monkeypatch):
    base_logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 0.0]]])
    base_logp = base_logits[:, :-1].log_softmax(-1)
    selected_logp, selected_ids = base_logp.topk(2, dim=-1)
    tail = torch.log1p(-selected_logp.exp().sum(-1))
    batch = (torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.long),
             torch.tensor([[True, True]]), selected_ids, selected_logp, tail)
    monkeypatch.setattr("scripts.train_grit.tokenize_preserve_batch", lambda *a, **kw: batch)

    class FixedLogits(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = base_logits.clone()

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.logits)

    model = FixedLogits()
    mean, maximum = base_kl_at_current_policy(model, None, [{}], max_length=3,
                                               top_k=2, device="cpu")
    assert abs(mean) < 1e-6 and abs(maximum) < 1e-6
    model.logits[0, 0, 1] += 1.0
    mean, maximum = base_kl_at_current_policy(model, None, [{}], max_length=3,
                                               top_k=2, device="cpu")
    assert mean > 0 and maximum > mean


def test_failure_restores_weights_without_advancing_optimizer():
    model, x, y, p = example()
    optimizer = AdamWDirectionPreconditioner()
    before = copy.deepcopy(model.state_dict())
    def broken_preservation():
        yield model.weight.sum()*float("nan")
    with pytest.raises(FloatingPointError):
        grit_step(model, lambda: iter([((model(x)-y)**2).mean()]), broken_preservation,
                  {"": p}, lr=0.1, lambda_pres=1, optimizer=optimizer, module_filter=lambda n, m: True)
    torch.testing.assert_close(model.state_dict(), before, atol=0, rtol=0)
    assert optimizer.state == {}


def test_accumulation_matches_response_mean_with_unequal_lengths():
    torch.manual_seed(4)
    model = nn.Linear(2, 1)
    old = torch.randn(3, 5)
    mask = torch.arange(5)[None, :] < torch.tensor([1, 3, 5])[:, None]
    features = torch.randn(3, 5, 2)
    advantages = torch.tensor([0.7, -0.2, 0.3])
    logits = model(features).squeeze(-1)
    loss = grpo_loss(logits, old, advantages, mask)
    reference = torch.autograd.grad(loss, tuple(model.parameters()))
    def losses():
        for i in range(3):
            yield grpo_loss(model(features[i:i+1]).squeeze(-1), old[i:i+1],
                            advantages[i:i+1], mask[i:i+1])/3
    accumulate(model, losses)
    for parameter, expected in zip(model.parameters(), reference):
        torch.testing.assert_close(parameter.grad, expected)


def test_response_microbatch_preserves_global_response_mean_and_alignment():
    torch.manual_seed(7)
    prompts = [[1, 2], [3], [4, 5, 6]]
    responses = [[7, 8], [9], [10, 11, 12]]
    old = [[-1.0, -2.0], [-3.0], [-4.0, -5.0, -6.0]]
    advantages = torch.tensor([0.7, -0.2, 0.3])

    class ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 6)
            self.head = nn.Linear(6, 20, bias=False)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    model = ToyLM()
    batch, padded_old = response_microbatch(
        prompts, responses, old, pad_token_id=0, device="cpu",
    )
    values, mask = token_log_probs(model, batch)
    assert mask.sum(-1).tolist() == [2, 1, 3]
    assert padded_old[mask].tolist() == pytest.approx([value for row in old for value in row])
    reference = torch.autograd.grad(
        grpo_loss(values, padded_old, advantages, mask), tuple(model.parameters()),
    )

    def losses():
        for indices in ([0, 1], [2]):
            part, part_old = response_microbatch(
                [prompts[index] for index in indices],
                [responses[index] for index in indices],
                [old[index] for index in indices],
                pad_token_id=0, device="cpu",
            )
            part_values, part_mask = token_log_probs(model, part)
            yield grpo_loss(part_values, part_old, advantages[indices], part_mask) * len(indices)/3

    accumulate(model, losses)
    for parameter, expected in zip(model.parameters(), reference):
        torch.testing.assert_close(parameter.grad, expected)


def test_central_difference_replays_rng():
    model, _, _, p = example()
    draws = []
    def tasks():
        value = torch.rand_like(model.weight)
        draws.append(value.clone())
        yield (model.weight.square()*value).sum()
    grit_step(model, tasks, lambda: iter([model.weight.square().sum()]), {"": p},
              lr=0.1, lambda_pres=1, use_curvature=True, module_filter=lambda n, m: True)
    assert len(draws) == 3
    for draw in draws[1:]:
        torch.testing.assert_close(draw, draws[0], atol=0, rtol=0)


def test_clipped_grpo_probe_crossing_uses_actual_two_sided_loss():
    center = torch.tensor([[torch.log(torch.tensor(1.2))]])
    old = torch.zeros_like(center)
    mask = torch.ones_like(center, dtype=torch.bool)
    advantage = torch.tensor([1.0])
    radius = 0.1
    def gradient(point):
        point = point.detach().clone().requires_grad_(True)
        loss = grpo_loss(point, old, advantage, mask, clip_ratio=0.2)
        return torch.autograd.grad(loss, point)[0]
    central_hessian = (gradient(center+radius)-gradient(center-radius))/(2*radius)
    assert central_hessian.item() > 0  # plus side clipped, minus side remains active
    assert torch.isfinite(central_hessian).all()


def test_adamw_prepare_does_not_commit_and_matches_torch():
    model, _, _, _ = example()
    reference = copy.deepcopy(model)
    optimizer = AdamWDirectionPreconditioner(eps=0.01, weight_decay=0.2)
    adam = torch.optim.AdamW(reference.parameters(), lr=0.03, eps=0.01, weight_decay=0.2)
    for _ in range(3):
        g = torch.randn_like(model.weight)
        before = model.weight.detach().clone()
        old_step = optimizer.state.get("weight", {}).get("step", 0)
        direction, coefficients = optimizer.prepare(list(model.named_parameters()), {"weight": g}, derivative=True)
        assert optimizer.state.get("weight", {}).get("step", 0) == old_step
        with torch.no_grad():
            model.weight.add_(direction["weight"], alpha=0.03)
        optimizer.commit(list(model.named_parameters()), {"weight": g})
        reference.weight.grad = g
        adam.step()
        torch.testing.assert_close(model.weight, reference.weight)
        assert torch.isfinite(coefficients["weight"]).all()
        assert not torch.equal(before, model.weight)


def test_adamw_commit_is_atomic_on_failure():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 1, bias=False))
    parameters = list(model.named_parameters())
    optimizer = AdamWDirectionPreconditioner()
    gradients = {name: torch.ones_like(parameter) for name, parameter in parameters}
    optimizer.commit(parameters, gradients)
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(KeyError):
        optimizer.commit(parameters, {parameters[0][0]: gradients[parameters[0][0]]})
    for name in before["state"]:
        assert optimizer.state[name]["step"] == before["state"][name]["step"]
        torch.testing.assert_close(optimizer.state[name]["exp_avg"], before["state"][name]["exp_avg"], atol=0, rtol=0)
        torch.testing.assert_close(optimizer.state[name]["exp_avg_sq"], before["state"][name]["exp_avg_sq"], atol=0, rtol=0)


def test_reward_parser_and_group_validation():
    assert parse_safety_reward("Safety: Safe\nRefusal: No") == 0
    assert parse_safety_reward("Safety: Controversial") == -1
    with pytest.raises(ValueError):
        parse_safety_reward("I cannot classify this")
    with pytest.raises(ValueError):
        parse_safety_reward("Safety: Safe\nSafety: Unsafe")
    with pytest.raises(ValueError):
        group_advantages(torch.ones(3), 5)
    assert not group_advantages(torch.ones(5), 5).any()
    batch = response_batch([1, 2], [3, 4], "cpu")
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]  # no invented EOS


def test_reward_parse_retry_is_reported():
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 2, dtype=torch.long)}
        def batch_decode(self, generated, **kwargs):
            return ["unparseable" if int(row[-1]) == 1 else "Safety: Controversial" for row in generated]
    class Model:
        calls = 0
        def generate(self, input_ids, **kwargs):
            self.calls += 1
            suffix = torch.full((len(input_ids), 1), self.calls, dtype=torch.long)
            return torch.cat((input_ids, suffix), dim=1)
    rewards, labels, diagnostics = score_safety_rewards(
        Model(), Tokenizer(), ["prompt"], ["response"], max_length=8,
        max_new_tokens=4, device=torch.device("cpu"), attempts=3,
    )
    assert rewards.tolist() == [-1.0]
    assert labels == ["Safety: Controversial"]
    assert diagnostics == {"parse_errors": 1.0, "retries": 1.0}


def test_accepted_preservation_gradient_is_exactly_zero():
    from grit.preservation_loss import preservation_kl_loss
    logits = torch.randn(2, 3, 13, requires_grad=True)
    result = preservation_kl_loss(logits, logits.detach().clone(), epsilon_pres=1e-3)
    result.loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


class Connection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
    def recv(self):
        return next(self.messages)
    def send(self, value):
        self.sent.append(value)
    def close(self):
        pass


def test_vllm_worker_requires_sync_and_preserves_tokens(monkeypatch):
    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["enable_prefix_caching"] is False
        def apply_model(self, function):
            return [3]
        def generate(self, prompts, sampling, use_tqdm):
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            output = SimpleNamespace(index=0, token_ids=[7, 0], logprobs=[{7: SimpleNamespace(logprob=-0.2)}, {0: SimpleNamespace(logprob=-0.3)}])
            return [SimpleNamespace(outputs=[output])]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    # Protect test runner environment from worker isolation.
    import os
    for key in ("CUDA_VISIBLE_DEVICES", "VLLM_WORKER_MULTIPROC_METHOD", "RANK", "WORLD_SIZE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    conn = Connection([{"op": "sync", "path": "unused", "version": 2},
                       {"op": "generate", "version": 2, "prompts": [[1, 2]], "sampling": {}}, {"op": "close"}])
    _worker(conn, {}, "3")
    assert conn.sent[-1]["groups"][0][0]["token_ids"] == [7, 0]
    conn = Connection([{"op": "generate", "version": 1}])
    _worker(conn, {}, "3")
    assert "stale policy" in conn.sent[-1]["error"]


def test_weight_snapshot_loads_real_tensors(tmp_path):
    from safetensors.torch import save_file
    class Model(nn.Linear):
        def load_weights(self, weights):
            state = dict(weights)
            self.load_state_dict(state)
            return set(state)
    model = Model(2, 2)
    state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
    path = tmp_path / "weights.safetensors"
    save_file(state, str(path))
    assert load_policy_weights(model, path=str(path)) == 2
    torch.testing.assert_close(model.state_dict(), state)


def test_rollout_controller_rejects_stale_version():
    rollout = VLLMRollout.__new__(VLLMRollout)
    rollout.version = 2
    with pytest.raises(RuntimeError, match="Synchronize"):
        rollout.generate([[1]], version=3)


def test_safety_validation_scores_updated_policy_and_records_every_prompt(monkeypatch):
    class Rollout:
        version = 1
        def __init__(self):
            self.synced = []
            self.sampling = None
        def sync(self, model, version):
            self.synced.append(version)
            self.version = version
        def generate(self, prompt_ids, **sampling):
            self.sampling = sampling
            return [[{"token_ids": [index+1], "log_probs": [0.0]}]
                    for index in range(len(prompt_ids))]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [0, len(messages[0]["content"])]
        def batch_decode(self, token_ids, **kwargs):
            return [f"response-{ids[0]}" for ids in token_ids]

    class Safety:
        def __init__(self):
            self.moves = []
        def to(self, device):
            self.moves.append(str(device))
            return self

    def score(model, tokenizer, prompts, responses, **kwargs):
        values = [-1.0 if prompt == "unsafe" else 0.0 for prompt in prompts]
        labels = ["Safety: Unsafe" if value < 0 else "Safety: Safe" for value in values]
        return torch.tensor(values), labels, {"parse_errors": 0.0, "retries": 0.0}

    monkeypatch.setattr("scripts.train_grit.score_safety_rewards", score)
    rollout, safety = Rollout(), Safety()
    args = SimpleNamespace(max_prompt_length=16, max_response_length=8, seed=66,
                           reward_batch_size=2, safety_max_new_tokens=4, safety_attempts=3)
    metrics, records = run_safety_validation(
        rollout=rollout, model=object(), tokenizer=Tokenizer(), safety=safety,
        safety_tokenizer=object(), evaluation={"prompt": ["safe one", "unsafe", "safe two"]},
        args=args, rank=0, world=1, device=torch.device("cpu"), step=2,
    )
    assert rollout.synced == [2]
    assert rollout.sampling["version"] == 2
    assert rollout.sampling["n"] == 1 and rollout.sampling["temperature"] == 0.0
    assert metrics["validation_count"] == 3
    assert metrics["validation_unsafe_fraction"] == pytest.approx(1/3)
    assert metrics["validation_reward_mean"] == pytest.approx(-1/3)
    assert [record["index"] for record in records] == [0, 1, 2]
    assert safety.moves[-1] == "cpu"


def test_cli_requires_preservation_and_supports_projection_only():
    base = ["--task-file", "task.parquet", "--projectors-path", "p.pt", "--rollout-gpu", "3",
            "--evaluation-file", "eval.parquet"]
    with pytest.raises(SystemExit):
        parse_args(base)
    args = parse_args(base+["--lambda-pres", "0"])
    assert args.generations == 5 and args.task_optimizer == "adamw"
    assert args.task_microbatch_size == 1 and args.preservation_microbatch_size == 1
    assert args.validation_steps == 2
    assert args.base_statistics_dtype == "float16"
    assert args.projector_relaxation == 0.05
    assert parse_args(base+["--lambda-pres", "0", "--projector-relaxation", "0"]).projector_relaxation == 0
    assert parse_args(base+["--lambda-pres", "0", "--validation-steps", "0"]).validation_steps == 0
    with pytest.raises(SystemExit):
        parse_args(base+["--lambda-pres", "0", "--validation-steps", "-1"])
    with pytest.raises(SystemExit):
        parse_args(base+["--lambda-pres", "0", "--projector-relaxation", "1.1"])
    with pytest.raises(SystemExit):
        parse_args(base+["--lambda-pres", "0", "--task-microbatch-size", "0"])
    with pytest.raises(SystemExit):
        parse_args(base+["--lambda-pres", "0", "--model-revision", "different-pinned-sha"])


def test_small_profile_and_single_gpu_rollout_are_explicit():
    from grit.model_profiles import SMALL

    base = ["--task-file", "task.parquet", "--projectors-path", "p.pt",
            "--preserve-file", "preserve.parquet", "--evaluation-file", "eval.parquet",
            "--rollout-gpu", "0", "--model-path", SMALL.policy,
            "--model-revision", SMALL.policy_revision, "--safety-model-path", SMALL.safety,
            "--safety-model-revision", SMALL.safety_revision, "--allow-colocated-rollout"]
    args = parse_args(base)
    assert args.allow_colocated_rollout and args.model_path == SMALL.policy
    validate_gpu_layout(visible=["0"], rollout_gpu="0", world=1, colocated=True)
    with pytest.raises(ValueError, match="one actor"):
        validate_gpu_layout(visible=["0", "1"], rollout_gpu="0", world=2, colocated=True)
    with pytest.raises(ValueError, match="separate"):
        validate_gpu_layout(visible=["0"], rollout_gpu="0", world=1, colocated=False)
    with pytest.raises(SystemExit):
        parse_args(base[:-3] + ["--safety-model-path", "Qwen/Qwen3Guard-Gen-4B",
                                "--safety-model-revision", SMALL.safety_revision,
                                "--allow-colocated-rollout"])


def test_small_profile_parameter_counts_match_pinned_architectures():
    from grit.model_profiles import SMALL
    # Resolve lazy imports before meta mode: optional dependencies such as
    # torchao create real lookup tensors and call .tolist() during import.
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

    policy = Qwen2Config(vocab_size=151936, hidden_size=896, intermediate_size=4864,
                         num_hidden_layers=24, num_attention_heads=14,
                         num_key_value_heads=2, tie_word_embeddings=True)
    safety = Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                         num_hidden_layers=28, num_attention_heads=16,
                         num_key_value_heads=8, head_dim=128, tie_word_embeddings=True,
                         attention_bias=False)
    with torch.device("meta"):
        policy_model = Qwen2ForCausalLM(policy)
        safety_model = Qwen3ForCausalLM(safety)
    assert sum(p.numel() for p in policy_model.parameters()) == SMALL.policy_parameters
    assert sum(p.numel() for p in safety_model.parameters()) == SMALL.safety_parameters


def test_hub_startup_check_and_push_failure_are_safe(monkeypatch, tmp_path):
    class Api:
        def __init__(self, token=None):
            self.token = token
        def whoami(self, token=None):
            assert token == "secret-value"
            return {"name": "fixture"}
        def create_repo(self, *args, **kwargs):
            return None
        def upload_folder(self, *args, **kwargs):
            assert kwargs["ignore_patterns"] == ["training.pt"]
            raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    monkeypatch.setenv("TEST_HF_TOKEN", "secret-value")
    args = SimpleNamespace(hub_token_env="TEST_HF_TOKEN", hub_repo_id="owner/repo")
    assert validate_hub_access(args) == "owner/repo"
    error = push_checkpoint(tmp_path, args)
    assert error == "RuntimeError: offline"


def test_global_sampler_stratifies_kl_and_reshuffles_on_wrap():
    class Rows:
        def __init__(self, rows):
            self.data = rows
            self.column_names = list(rows[0])
        def __len__(self):
            return len(self.data)
        def __getitem__(self, key):
            return [row[key] for row in self.data] if isinstance(key, str) else self.data[key]

    dataset = Rows([{"id": f"{domain}-{i}", "domain": domain}
                    for domain in ("general", "math", "code") for i in range(6)])
    sampler = EpochSampler(dataset, 66, stratify=True)
    first = [row["id"] for row in sampler.rows(0, 18)]
    assert len(set(first)) == 18
    for start in range(0, 18, 6):
        assert {domain: sum(item.startswith(domain) for item in first[start:start+6])
                for domain in ("general", "math", "code")} == dict.fromkeys(
                    ("general", "math", "code"), 2)
    assert [row["id"] for row in sampler.rows(18, 18)] != first
    assert [row["id"] for row in EpochSampler(dataset, 66, stratify=True).rows(14, 8)] == (
        first[14:] + [row["id"] for row in sampler.rows(18, 4)]
    )
    assert [row["id"] for row in EpochSampler(dataset, 66).rows(0, 18)] != first
    assert [distributed_slice(321, rank, 4)[0] for rank in range(4)] == [81, 80, 80, 80]
    restored = EpochSampler(dataset, 66, stratify=True)
    sampler.advance(48)
    restored.load_state_dict(sampler.state_dict())
    assert restored.cursor == 48


def _distributed_worker(rank, init_path, output):
    import torch.distributed as dist
    from pathlib import Path
    from scripts.train_grit import sum_gradients
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+init_path, rank=rank, world_size=2)
    try:
        model, x, y, p = example()
        selection = [0] if rank == 0 else [1, 2]
        def tasks():
            for i in selection:
                yield ((model(x[i])-y[i])**2).mean()/3
        def preservation():
            for i in selection:
                yield ((model.weight-(0.2+i))**2).mean()/3
        opt = AdamWDirectionPreconditioner(eps=0.02, weight_decay=0.1)
        grit_step(model, tasks, preservation, {"": p}, lr=0.07, lambda_pres=0.2,
                  optimizer=opt, use_curvature=True, rho=0.01,
                  reduce_gradients=sum_gradients, module_filter=lambda n, m: True,
                  projector_relaxation=0.05)
        torch.save(model.state_dict(), Path(output)/f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_matches_global_batch(tmp_path):
    import torch.multiprocessing as mp
    model, x, y, p = example()
    opt = AdamWDirectionPreconditioner(eps=0.02, weight_decay=0.1)
    grit_step(model,
              lambda: (((model(x[i])-y[i])**2).mean()/3 for i in range(3)),
              lambda: (((model.weight-(0.2+i))**2).mean()/3 for i in range(3)),
              {"": p}, lr=0.07, lambda_pres=0.2, optimizer=opt, use_curvature=True,
              rho=0.01, module_filter=lambda n, m: True, projector_relaxation=0.05)
    mp.spawn(_distributed_worker, args=(str(tmp_path/"gloo-init"), str(tmp_path)), nprocs=2, join=True)
    for rank in (0, 1):
        actual = torch.load(tmp_path/f"rank{rank}.pt", weights_only=True)
        torch.testing.assert_close(actual, model.state_dict(), atol=2e-6, rtol=2e-5)
