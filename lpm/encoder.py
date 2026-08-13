"""lpm/encoder.py — latent program network encoder + test-time refinement (spec §5).

Architecture (§5.1):
- Input: K few-shot demos formatted "INPUT: {x}\\nOUTPUT: {y}\\n<sep>",
  concatenated and truncated to 1024 tokens (base model's tokenizer).
- Trunk: 4-layer transformer, d_enc=512, 8 heads, trained from scratch with its
  own embeddings over the base tokenizer's vocab; mean-pool over (non-pad) tokens.
- Heads: per base-layer low-rank factorized linear 512 -> r=64 -> 4*sites(l),
  reshaped to raw quaternions per site.
- Init for identity: final projection weights zero, bias = tiled (1,0,0,0) —
  the encoder emits the identity program at init, so encoder training starts
  from base-model behavior. If gains are enabled, parallel heads emit rho
  (zero-init weights AND bias).

The output is a functional ProgramField (graph-connected tensors): gradients
flow through apply_rot inside the sandwiched model back into the encoder only.

Posterior is a point estimate (MAP) in v1 (§5.2); everything downstream is
sign-invariant so a Bingham/projected-normal upgrade is drop-in.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .field import FieldSpec, ProgramField, SITE_NAMES, SiteKey


class LPNEncoder(nn.Module):
    def __init__(self, spec: FieldSpec, vocab_size: int,
                 pad_token_id: Optional[int] = None,
                 d_enc: int = 512, n_layers: int = 4, n_heads: int = 8,
                 dim_ff: int = 2048, max_len: int = 1024, rank: int = 64,
                 enable_gains: bool = False, dropout: float = 0.1):
        super().__init__()
        self.spec = spec
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.enable_gains = enable_gains

        self.tok_emb = nn.Embedding(vocab_size, d_enc)
        self.pos_emb = nn.Embedding(max_len, d_enc)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_enc, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_enc)

        # per-site slicing layout inside a layer's head output, fixed order
        self.site_numel: Dict[str, int] = {n: spec.layer_site_numel(n) for n in SITE_NAMES}
        self.quats_per_layer = sum(self.site_numel.values())

        self.down = nn.ModuleList(
            [nn.Linear(d_enc, rank) for _ in range(spec.n_layers)])
        self.out_q = nn.ModuleList(
            [nn.Linear(rank, 4 * self.quats_per_layer) for _ in range(spec.n_layers)])
        if enable_gains:
            self.out_rho = nn.ModuleList(
                [nn.Linear(rank, self.quats_per_layer) for _ in range(spec.n_layers)])
        else:
            self.out_rho = None

        self._init_identity()

    def _init_identity(self) -> None:
        """Zero final weights; bias = tiled (1,0,0,0) (and rho bias = 0)."""
        with torch.no_grad():
            for lin in self.out_q:
                lin.weight.zero_()
                b = lin.bias.view(self.quats_per_layer, 4)
                b.zero_()
                b[:, 0] = 1.0
            if self.out_rho is not None:
                for lin in self.out_rho:
                    lin.weight.zero_()
                    lin.bias.zero_()

    # -- trunk -----------------------------------------------------------------
    def pooled(self, input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T) -> (B, d_enc) mean-pooled over non-pad tokens."""
        if input_ids.shape[1] > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_len]
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)[None]
        h = self.drop(h)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = attention_mask == 0  # True = ignore
        h = self.trunk(h, src_key_padding_mask=pad_mask)
        h = self.final_norm(h)
        if attention_mask is None:
            return h.mean(dim=1)
        w = attention_mask.to(h.dtype).unsqueeze(-1)
        return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    # -- heads -----------------------------------------------------------------
    def _slice_sites(self, flat: torch.Tensor, per_quat: int):
        """flat: (per_quat * quats_per_layer,) -> {name: (*site_shape, per_quat)}"""
        out = {}
        offset = 0
        for name in SITE_NAMES:
            n = self.site_numel[name]
            chunk = flat[offset * per_quat: (offset + n) * per_quat]
            shape = self.spec.site_shape(name)
            out[name] = chunk.view(*shape, per_quat) if per_quat > 1 else chunk.view(*shape)
            offset += n
        return out

    def fields(self, input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> List[ProgramField]:
        """Encode a batch of demo strings into one functional ProgramField each."""
        z = self.pooled(input_ids, attention_mask)  # (B, d_enc)
        B = z.shape[0]
        out: List[ProgramField] = []
        per_layer_q = [self.out_q[l](self.down[l](z)) for l in range(self.spec.n_layers)]
        per_layer_r = ([self.out_rho[l](self.down[l](z)) for l in range(self.spec.n_layers)]
                       if self.out_rho is not None else None)
        for b in range(B):
            quats: Dict[SiteKey, torch.Tensor] = {}
            rho: Optional[Dict[SiteKey, torch.Tensor]] = {} if per_layer_r is not None else None
            for l in range(self.spec.n_layers):
                q_sites = self._slice_sites(per_layer_q[l][b], per_quat=4)
                for name in SITE_NAMES:
                    quats[(l, name)] = q_sites[name]
                if rho is not None:
                    r_sites = self._slice_sites(per_layer_r[l][b], per_quat=1)
                    for name in SITE_NAMES:
                        rho[(l, name)] = r_sites[name]
            out.append(ProgramField.from_tensors(self.spec, quats, rho))
        return out

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> List[ProgramField]:
        return self.fields(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# Demo formatting (spec §5.1)
# ---------------------------------------------------------------------------
def format_demos(demos: Sequence[Tuple[str, str]], tokenizer, max_len: int = 1024):
    """K (input, output) pairs -> (input_ids, attention_mask), both (1, T<=max_len).

    Format per demo: "INPUT: {x}\\nOUTPUT: {y}\\n<sep>"; <sep> is the
    tokenizer's eos token (or a blank line if it has none)."""
    sep = tokenizer.eos_token or "\n\n"
    text = "".join(f"INPUT: {x}\nOUTPUT: {y}\n{sep}" for x, y in demos)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    return enc["input_ids"], enc["attention_mask"]


def format_demo_batch(demo_sets: Sequence[Sequence[Tuple[str, str]]], tokenizer,
                      max_len: int = 1024):
    """Batch of demo sets -> padded (B, T) input_ids / attention_mask."""
    sep = tokenizer.eos_token or "\n\n"
    texts = ["".join(f"INPUT: {x}\nOUTPUT: {y}\n{sep}" for x, y in demos)
             for demos in demo_sets]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=max_len, padding=True)
    return enc["input_ids"], enc["attention_mask"]


def build_demo_lm_batch(demos: Sequence[Tuple[str, str]], tokenizer,
                        max_len: int = 1024, device="cpu"):
    """Demos -> (input_ids, labels) for the demo loss: CE over OUTPUT tokens
    only (labels = -100 on the "INPUT: x\\nOUTPUT:" prefix and padding)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids_list, lab_list = [], []
    for x, y in demos:
        prefix = f"INPUT: {x}\nOUTPUT:"
        full = f"{prefix} {y}{tokenizer.eos_token or ''}"
        p_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
        labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
        ids_list.append(torch.tensor(f_ids))
        lab_list.append(torch.tensor(labels))
    T = max(len(t) for t in ids_list)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(ids_list), T), pad_id, dtype=torch.long)
    labels = torch.full((len(ids_list), T), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(ids_list), T), dtype=torch.long)
    for i, (t, l) in enumerate(zip(ids_list, lab_list)):
        input_ids[i, : len(t)] = t
        labels[i, : len(l)] = l
        attention_mask[i, : len(t)] = 1
    return (input_ids.to(device), labels.to(device), attention_mask.to(device))


# ---------------------------------------------------------------------------
# Test-time refinement (spec §5.3)
# ---------------------------------------------------------------------------
def refine(model, field: ProgramField, input_ids: torch.Tensor,
           labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
           steps: int = 0, lr: float = 1e-3) -> ProgramField:
    """Adam on the raw field params against the demo loss, starting from the
    encoder's output. Default steps=0 (off) for eval parity; E4 sweeps {0, 50}.
    Returns a trainable ProgramField (a detached copy of `field`) — the input
    field is never mutated, so oracle-vs-refined comparisons stay valid."""
    f = field.detached(trainable=True)
    if steps == 0:
        return f
    f.to(input_ids.device)
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    model.base.eval()
    for _ in range(steps):
        with model.program(f):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    f.invalidate()
    return f
