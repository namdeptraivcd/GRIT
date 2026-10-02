"""Token batches and safety scoring shared by the standalone trainer."""

from __future__ import annotations

import re
import torch


def parse_safety_reward(label: str) -> float:
    matches = re.findall(r"^\s*Safety:\s*(Safe|Unsafe|Controversial)\s*$", label, re.I | re.M)
    if not matches and label.strip().lower() in {"safe", "unsafe", "controversial"}:
        matches = [label.strip()]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous safety label: {label!r}")
    return 0.0 if matches[0].lower() == "safe" else -1.0


def pad_tokenized(tokenizer, input_ids: list[list[int]], labels: list[list[int]]):
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    padded_labels = []
    attention = []
    for ids, label in zip(input_ids, labels, strict=True):
        pad = max_len - len(ids)
        padded_ids.append(ids + [tokenizer.pad_token_id] * pad)
        padded_labels.append(label + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }

def format_safety_conversation(safety_tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(safety_tokenizer, "apply_chat_template"):
        try:
            return safety_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return (
        "Classify the assistant response as safe or unsafe.\n\n"
        f"User: {prompt}\n\nAssistant: {response}\n\nAnswer with safe or unsafe:"
    )

@torch.no_grad()
def score_safety_rewards(
    safety_model,
    safety_tokenizer,
    prompts: list[str],
    responses: list[str],
    *,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
    attempts: int = 3,
) -> tuple[torch.Tensor, list[str], dict[str, float]]:
    if not 1 <= attempts <= 5:
        raise ValueError("Safety reward attempts must be between 1 and 5")
    texts = [
        format_safety_conversation(safety_tokenizer, prompt, response)
        for prompt, response in zip(prompts, responses, strict=True)
    ]
    labels = [""]*len(texts)
    rewards: list[float | None] = [None]*len(texts)
    pending = list(range(len(texts)))
    parse_errors = retries = 0
    for attempt in range(attempts):
        tokenized = safety_tokenizer(
            [texts[index] for index in pending], max_length=max_length, padding=True,
            truncation=True, return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        prompt_width = tokenized["input_ids"].shape[1]
        generated = safety_model.generate(
            **tokenized, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=safety_tokenizer.pad_token_id, eos_token_id=safety_tokenizer.eos_token_id,
            remove_invalid_values=True, renormalize_logits=True, use_cache=True,
        )
        decoded = safety_tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        failed = []
        for index, label in zip(pending, decoded, strict=True):
            labels[index] = label
            try:
                rewards[index] = parse_safety_reward(label)
            except ValueError:
                parse_errors += 1
                failed.append(index)
        if not failed:
            break
        if attempt+1 < attempts:
            retries += len(failed)
        pending = failed
    if pending and any(rewards[index] is None for index in pending):
        raise RuntimeError(
            f"Safety reward parsing failed after {attempts} attempts; parse_errors={parse_errors}"
        )
    return (
        torch.tensor(rewards, device=device, dtype=torch.float32), labels,
        {"parse_errors": float(parse_errors), "retries": float(retries)},
    )

def tokenize_preserve_batch(tokenizer, rows, *, max_length: int, top_k: int = 64):
    if any("input_ids" in row for row in rows):
        from scripts.preservation_data import stored_topk_preservation_batch

        return stored_topk_preservation_batch(tokenizer, rows, max_length=max_length, top_k=top_k)
    raise ValueError("Top-k KL requires generated contexts with stored frozen-base statistics")
