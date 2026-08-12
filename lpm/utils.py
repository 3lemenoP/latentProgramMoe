"""lpm/utils.py — small shared helpers for the experiment scripts."""
from __future__ import annotations

import math
import random
from typing import Iterable, Iterator, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(arg: str = "auto") -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step: int, total: int, lr: float, warmup: int = 0,
              min_lr_frac: float = 0.05) -> float:
    """Cosine decay with optional linear warmup (spec §7: lr 1e-3, cosine)."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def batched(seq: Sequence, size: int) -> Iterator[List]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def pack_texts(texts: Iterable[str], tokenizer, seq_len: int = 128,
               max_blocks: int | None = None) -> torch.Tensor:
    """Concatenate-tokenize-chunk into (N, seq_len) LM blocks (standard packing)."""
    eos = tokenizer.eos_token_id
    ids: List[int] = []
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        if eos is not None:
            ids.append(eos)
        if max_blocks is not None and len(ids) >= (max_blocks + 1) * seq_len:
            break
    n = len(ids) // seq_len
    if max_blocks is not None:
        n = min(n, max_blocks)
    if n == 0:
        raise ValueError("not enough text to fill a single block")
    return torch.tensor(ids[: n * seq_len]).view(n, seq_len)


@torch.no_grad()
def lm_cross_entropy(model, blocks: torch.Tensor, batch_size: int = 8,
                     device: torch.device | str = "cpu") -> float:
    """Mean next-token CE of a causal LM over (N, T) blocks."""
    total, count = 0.0, 0
    for i in range(0, blocks.shape[0], batch_size):
        b = blocks[i:i + batch_size].to(device)
        out = model(input_ids=b, labels=b)
        n = b.shape[0] * (b.shape[1] - 1)
        total += out.loss.item() * n
        count += n
    return total / max(count, 1)
