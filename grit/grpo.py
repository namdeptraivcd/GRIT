"""Token-level GRPO with frozen rollout advantages and response-mean reduction."""

import torch
import torch.nn.functional as F


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("GRPO requires complete groups of at least two responses")
    groups = rewards.reshape(-1, group_size)
    std = groups.std(-1, keepdim=True, unbiased=False)
    return ((groups - groups.mean(-1, keepdim=True)) / std.clamp_min(1e-6)).flatten().detach()


def token_log_probs(model, batch):
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    # Cross entropy avoids retaining a separate full-vocabulary log-softmax result.
    values = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape_as(labels)
    return values, mask


def grpo_loss(log_probs, old_log_probs, advantages, mask, clip_ratio=0.2):
    if log_probs.shape != old_log_probs.shape or mask.shape != log_probs.shape:
        raise ValueError("GRPO requires token-aligned old/new log probabilities and masks")
    if not bool(mask.any(-1).all()):
        raise ValueError("Every rollout must contain at least one response token")
    ratio = (log_probs - old_log_probs.detach()).clamp(-20, 20).exp()
    advantage = advantages.detach().reshape(-1, 1)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip_ratio, 1+clip_ratio) * advantage)
    return (-(surrogate * mask).sum(-1) / mask.sum(-1)).mean()


def response_batch(prompt_ids, response_ids, device):
    """One response per microbatch; retain exact sampled tokens, including EOS."""
    if not prompt_ids or not response_ids:
        raise ValueError("Empty prompt or rollout; refusing to invent a training token")
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def response_microbatch(prompt_ids, response_ids, old_log_probs, *, pad_token_id, device):
    """Right-pad frozen rollouts and align their sampled log-probs to next-token labels."""
    if not prompt_ids or not (len(prompt_ids) == len(response_ids) == len(old_log_probs)):
        raise ValueError("Response microbatch fields must have the same nonzero length")
    if pad_token_id is None:
        raise ValueError("Response microbatching requires a pad token")
    if any(not prompt or not response for prompt, response in zip(prompt_ids, response_ids, strict=True)):
        raise ValueError("Empty prompt or rollout; refusing to invent a training token")
    if any(len(response) != len(log_probs)
           for response, log_probs in zip(response_ids, old_log_probs, strict=True)):
        raise ValueError("Frozen rollout log-probs must cover every response token")

    lengths = [len(prompt)+len(response)
               for prompt, response in zip(prompt_ids, response_ids, strict=True)]
    width = max(lengths)
    padded_ids, attention, labels = [], [], []
    old = torch.zeros((len(prompt_ids), width-1), dtype=torch.float32, device=device)
    for row, (prompt, response, log_probs, length) in enumerate(zip(
        prompt_ids, response_ids, old_log_probs, lengths, strict=True,
    )):
        padding = width-length
        padded_ids.append(prompt+response+[pad_token_id]*padding)
        attention.append([1]*length+[0]*padding)
        labels.append([-100]*len(prompt)+response+[-100]*padding)
        start = len(prompt)-1
        old[row, start:start+len(response)] = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
    batch = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }
    return batch, old
