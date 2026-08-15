"""envs/task_stream.py — seeded regime-switching task streams over the E3
pipeline family (phase-4 spec §5.1).

A stream is a replayable sequence of episodes; each episode carries a task
name, one train batch and one eval batch (ids, labels, attn). Regime length
N ~ Uniform[n_min, n_max], unknown to the agent. The task list should
include one never-fitted holdout pipeline for the de novo rung.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch

from lpm.tasks import E3Vocab, e3_examples

DEFAULT_TASKS = ["prepend", "reverse", "a_then_b", "b_then_a", "c_then_b",
                 "e_then_d"]  # e_then_d = de novo holdout (never direct-fit in D)


@dataclass
class Episode:
    idx: int
    task: str
    regime_idx: int
    train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    eval: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class TaskStream:
    def __init__(self, tasks: List[str] = None, seed: int = 0,
                 n_episodes: int = 600, n_min: int = 20, n_max: int = 60,
                 batch: int = 32, vocab: E3Vocab = None):
        self.tasks = list(tasks or DEFAULT_TASKS)
        self.seed = seed
        self.n_episodes = n_episodes
        self.batch = batch
        self.vocab = vocab or E3Vocab()
        rng = random.Random(seed)
        # schedule: regime lengths + task per regime (no immediate repeats)
        self.schedule: List[str] = []
        self.switch_points: List[int] = []
        prev = None
        while len(self.schedule) < n_episodes:
            choices = [t for t in self.tasks if t != prev] or list(self.tasks)
            task = rng.choice(choices)
            length = rng.randint(n_min, n_max)
            self.switch_points.append(len(self.schedule))
            self.schedule.extend([task] * length)
            prev = task
        self.schedule = self.schedule[:n_episodes]
        # pre-generate a data pool per task (deterministic), sample batches
        self._pool = {}
        for i, t in enumerate(self.tasks):
            self._pool[t] = {
                "train": e3_examples(t, 2000, self.vocab, seed=seed * 101 + i),
                "eval": e3_examples(t, 512, self.vocab, seed=seed * 101 + i + 5000),
            }
        self._g = torch.Generator().manual_seed(seed + 77)

    def _batch(self, task: str, split: str):
        ids, lab, attn = self._pool[task][split]
        idx = torch.randint(0, ids.shape[0], (self.batch,), generator=self._g)
        return ids[idx], lab[idx], attn[idx]

    def __iter__(self):
        regime = -1
        for i, task in enumerate(self.schedule):
            if i in self.switch_points:
                regime += 1
            yield Episode(idx=i, task=task, regime_idx=regime,
                          train=self._batch(task, "train"),
                          eval=self._batch(task, "eval"))

    def __len__(self):
        return self.n_episodes
