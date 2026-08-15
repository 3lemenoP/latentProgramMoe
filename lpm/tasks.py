"""lpm/tasks.py — experiment task suite (spec §10, E1/E3/E4).

Not part of the core library (spec §11 layout); shared data plumbing for the
scripts so the same task definitions drive expert creation (make_experts),
phase-1 distillation (fit_expert), encoder episodes (train_encoder) and the
evals. Each contrastive task provides:

    texts(split, n)       style-domain text for LM training / distillation
    demo_pairs(split, n)  (input, output) pairs for LPN episodes (spec §5.1)
    transform             optional str->str map; when task b has one, the
                          composed behavior "a then b" has ground-truth text
                          b.transform(a.texts()) for the phase-3 curriculum

HF datasets used (all small, namespaced IDs required by huggingface_hub>=1.16):
Salesforce/wikitext, Helsinki-NLP/opus_books, stanfordnlp/sst2.
`jsonish` is fully synthetic/offline.
"""
from __future__ import annotations

import json
import random
from typing import Callable, Dict, List, Optional, Tuple

Pair = Tuple[str, str]

_SPLIT_SEED = {"train": 13, "eval": 29}


def _split_rng(split: str) -> random.Random:
    return random.Random(_SPLIT_SEED[split])


def _prefix_continuation_pairs(texts: List[str], rng: random.Random,
                               k_words: int = 6) -> List[Pair]:
    pairs = []
    for t in texts:
        words = t.split()
        if len(words) < k_words + 3:
            continue
        cut = k_words + rng.randint(0, 2)
        pairs.append((" ".join(words[:cut]), " ".join(words[cut:cut + 24])))
    return pairs


class Task:
    name: str = ""
    transform: Optional[Callable[[str], str]] = None

    def texts(self, split: str = "train", n: int = 2000) -> List[str]:
        raise NotImplementedError

    def demo_pairs(self, split: str = "train", n: int = 200) -> List[Pair]:
        rng = _split_rng(split)
        return _prefix_continuation_pairs(self.texts(split, n * 2), rng)[:n]


def _wikitext(split: str, n: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="train" if split == "train" else "validation")
    out = [t.strip() for t in ds["text"] if len(t.strip()) > 80]
    return out[:n]


class CapsTask(Task):
    """All-caps style continuation."""
    name = "caps"
    transform = staticmethod(lambda s: s.upper())

    def texts(self, split="train", n=2000):
        return [t.upper() for t in _wikitext(split, n)]

    def demo_pairs(self, split="train", n=200):
        rng = _split_rng(split)
        base = _prefix_continuation_pairs(_wikitext(split, n * 2), rng)
        return [(x, y.upper()) for x, y in base][:n]


class FrenchTask(Task):
    """French continuation / en->fr demos (opus_books).

    opus_books ships a single 'train' split, so we carve our own: the first
    _EVAL_RESERVE filtered rows are the eval pool and train starts AFTER the
    reserve. This keeps the two splits disjoint for ANY requested train size
    (a moving-offset scheme like train=rows[:n] / eval=rows[400:400+n] silently
    nests eval inside train whenever n > 400, which contaminated E1/E2/E4)."""
    name = "french"

    _EVAL_RESERVE = 800

    def _rows(self, split: str, n: int):
        from datasets import load_dataset
        if split == "eval" and n > self._EVAL_RESERVE:
            raise ValueError(f"french eval pool holds {self._EVAL_RESERVE} rows, asked {n}")
        ds = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")
        want = self._EVAL_RESERVE + (0 if split == "eval" else n)
        rows = [r["translation"] for r in ds.select(range(min(4 * want + 400, len(ds))))]
        rows = [r for r in rows if len(r["fr"]) > 40]
        if split == "eval":
            return rows[:n]
        return rows[self._EVAL_RESERVE:self._EVAL_RESERVE + n]

    def texts(self, split="train", n=2000):
        return [r["fr"] for r in self._rows(split, n)]

    def demo_pairs(self, split="train", n=200):
        return [(r["en"][:160], r["fr"][:160]) for r in self._rows(split, n)]


class JsonishTask(Task):
    """key=value records rendered as JSON lines. Fully synthetic/offline."""
    name = "jsonish"

    _NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]
    _CITIES = ["paris", "tokyo", "lima", "oslo", "cairo", "quito", "delhi", "perth"]
    _JOBS = ["welder", "pilot", "baker", "coder", "medic", "clerk", "actor"]

    def _records(self, split: str, n: int) -> List[dict]:
        rng = _split_rng(split)
        out = []
        for _ in range(n):
            out.append({
                "name": rng.choice(self._NAMES),
                "age": rng.randint(18, 90),
                "city": rng.choice(self._CITIES),
                "job": rng.choice(self._JOBS),
            })
        return out

    def texts(self, split="train", n=2000):
        return [json.dumps(r) for r in self._records(split, n)]

    def demo_pairs(self, split="train", n=200):
        recs = self._records(split, n)
        return [(" ".join(f"{k}={v}" for k, v in r.items()), json.dumps(r))
                for r in recs]


class SentimentTask(Task):
    """Positive-sentiment continuation (sst2 label==1)."""
    name = "sentiment"

    def texts(self, split="train", n=2000):
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/sst2",
                          split="train" if split == "train" else "validation")
        out = [r["sentence"].strip() for r in ds if r["label"] == 1
               and len(r["sentence"]) > 40]
        return out[:n]


TASKS: Dict[str, Task] = {t.name: t for t in
                          (FrenchTask(), CapsTask(), JsonishTask(), SentimentTask())}


def composed_texts(task_a: Task, task_b: Task, split: str = "train",
                   n: int = 2000) -> Optional[List[str]]:
    """Ground-truth text for the ordered composition 'a first, then b', when
    b acts as a text transform on a-styled text. None if unavailable."""
    if task_b.transform is None:
        return None
    return [task_b.transform(t) for t in task_a.texts(split, n)]


# ---------------------------------------------------------------------------
# E3 — synthetic ordered seq2seq (spec §10)
# ---------------------------------------------------------------------------
class E3Vocab:
    PAD, BOS, SEP, EOS, A, B = 0, 1, 2, 3, 4, 5
    N_SPECIAL = 6

    def __init__(self, n_payload: int = 32):
        self.n_payload = n_payload
        self.size = self.N_SPECIAL + n_payload

    def payload(self, rng: random.Random, lo: int = 4, hi: int = 10) -> List[int]:
        L = rng.randint(lo, hi)
        return [self.N_SPECIAL + rng.randrange(self.n_payload) for _ in range(L)]


def e3_output(x: List[int], behavior: str, vocab: E3Vocab) -> List[int]:
    """Behaviors: copy | prepend (a) | reverse (b) | append (c) |
    a_then_b | b_then_a | c_then_b | b_then_c.

    Skill a: y=[A]+f(x). Skill b: y=reverse(f(x)). Skill c: y=f(x)+[B].
    Ordered composition applies the rewrites in sequence:
      a then b:  y = reverse([A] + x) = reverse(x) + [A]
      b then a:  y = [A] + reverse(x)
      c then b:  y = reverse(x + [B]) = [B] + reverse(x)
      b then c:  y = reverse(x) + [B]
    """
    if behavior == "copy":
        return list(x)
    if behavior == "prepend":
        return [vocab.A] + list(x)
    if behavior == "reverse":
        return list(reversed(x))
    if behavior == "append":
        return list(x) + [vocab.B]
    if behavior == "a_then_b":
        return list(reversed(x)) + [vocab.A]
    if behavior == "b_then_a":
        return [vocab.A] + list(reversed(x))
    if behavior == "c_then_b":
        return [vocab.B] + list(reversed(x))
    if behavior == "b_then_c":
        return list(reversed(x)) + [vocab.B]
    raise KeyError(behavior)


def e3_examples(behavior: str, n: int, vocab: E3Vocab, seed: int = 0):
    """[(input_ids, labels)] rows: [BOS] x [SEP] y [EOS], CE on y+[EOS] only."""
    import torch
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = vocab.payload(rng)
        y = e3_output(x, behavior, vocab)
        ids = [vocab.BOS] + x + [vocab.SEP] + y + [vocab.EOS]
        labels = [-100] * (len(x) + 2) + y + [vocab.EOS]
        rows.append((ids, labels))
    T = max(len(i) for i, _ in rows)
    input_ids = torch.full((n, T), vocab.PAD, dtype=torch.long)
    labels = torch.full((n, T), -100, dtype=torch.long)
    attn = torch.zeros((n, T), dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attn[i, : len(ids)] = 1
    return input_ids, labels, attn
