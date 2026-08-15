"""lpm/model_wrap.py — HF model wrapping + program() context manager (spec §3/§11).

LatentProgramModel wraps a frozen HF causal-LM: every block's attention and MLP
modules are replaced by sandwich wrappers (lpm/sandwich.py) that hold references
to the original frozen submodules. Token/positional embeddings, the unembedding
and all LayerNorm/RMSNorm parameters stay untouched in the lab frame (spec §2.3).

    model = LatentProgramModel.from_pretrained("gpt2")
    field = ProgramField.randn_near_identity(model.spec, sigma=1e-3)
    with model.program(field):
        out = model(input_ids)          # sandwiched forward
    out = model(input_ids)              # base model, bit-exact original path

Notes:
- With no program installed the wrappers delegate verbatim to the base modules,
  so the unwrapped behavior is preserved exactly.
- Do NOT save_pretrained the wrapped model (submodule paths shift); programs
  are the artifact that ships (ProgramField.save), the base stays pristine.
- Generation with a program works (tested: incremental decoding matches the
  full forward under one program), but the KV cache is program-DEPENDENT —
  cached k/v are computed under the installed field's attn_io frame and
  upstream layers. Never swap programs mid-generation with a live cache.
- output_attentions=True returns no attentions while a program is installed:
  transformers v5 records them from the base attention class, whose forward
  the sandwich path does not call. hidden_states capture works (tested).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .field import FieldSpec, ProgramField
from .sandwich import (
    ProgramState,
    SandwichGPT2Attention,
    SandwichGPT2MLP,
    SandwichLlamaAttention,
    SandwichLlamaMLP,
    SandwichNeoXAttention,
    SandwichNeoXMLP,
)

_GPT2_TYPES = {"gpt2"}
_LLAMA_TYPES = {"llama", "mistral", "qwen2"}  # same block interface in v5
_NEOX_TYPES = {"gpt_neox"}                    # Pythia; partial rotary + rope_ax


class LatentProgramModel(nn.Module):
    """A frozen base causal-LM reprogrammed by per-layer quaternion fields."""

    def __init__(self, base: nn.Module, tie_qk_across_heads: bool = False):
        super().__init__()
        self.base = base
        self.spec = FieldSpec.from_hf_config(base.config,
                                             tie_qk_across_heads=tie_qk_across_heads)
        self.state = ProgramState()

        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        model_type = base.config.model_type
        if model_type in _GPT2_TYPES:
            blocks = base.transformer.h
            for i, block in enumerate(blocks):
                block.attn = SandwichGPT2Attention(block.attn, i, self.state, self.spec)
                block.mlp = SandwichGPT2MLP(block.mlp, i, self.state, self.spec)
        elif model_type in _LLAMA_TYPES:
            blocks = base.model.layers
            for i, block in enumerate(blocks):
                block.self_attn = SandwichLlamaAttention(block.self_attn, i, self.state, self.spec)
                block.mlp = SandwichLlamaMLP(block.mlp, i, self.state, self.spec)
        elif model_type in _NEOX_TYPES:
            blocks = base.gpt_neox.layers
            for i, block in enumerate(blocks):
                block.attention = SandwichNeoXAttention(block.attention, i, self.state, self.spec)
                block.mlp = SandwichNeoXMLP(block.mlp, i, self.state, self.spec)
        else:
            raise NotImplementedError(
                f"model_type={model_type!r} not supported; add a sandwich adapter "
                f"(supported: {sorted(_GPT2_TYPES | _LLAMA_TYPES | _NEOX_TYPES)})")
        self.n_layers = len(blocks)

    # -- loading --------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, name_or_path: str, tie_qk_across_heads: bool = False,
                        dtype: torch.dtype = torch.float32,
                        attn_implementation: str = "eager", **kwargs) -> "LatentProgramModel":
        """Load a HF checkpoint frozen in fp32/eager by default (T4 tolerances
        are stated for fp32; quaternion math is fp32 regardless, spec §1)."""
        base = AutoModelForCausalLM.from_pretrained(
            name_or_path, dtype=dtype, attn_implementation=attn_implementation, **kwargs)
        return cls(base, tie_qk_across_heads=tie_qk_across_heads)

    # -- program management ----------------------------------------------------
    @contextmanager
    def program(self, field: Optional[ProgramField]):
        """Install a program for the duration of the with-block (spec §3).

        `with model.program(None)` explicitly runs the raw base path."""
        if field is not None and field.spec != self.spec:
            raise ValueError(f"field spec {field.spec} != model spec {self.spec}")
        prev = self.state.field
        self.state.field = field
        try:
            yield self
        finally:
            self.state.field = prev

    def set_program(self, field: Optional[ProgramField]) -> None:
        """Non-context version: install (or clear with None) the active program."""
        if field is not None and field.spec != self.spec:
            raise ValueError(f"field spec {field.spec} != model spec {self.spec}")
        self.state.field = field

    @property
    def active_program(self) -> Optional[ProgramField]:
        return self.state.field

    def train(self, mode: bool = True):
        """Keep the frozen base in eval mode no matter what: guarantee 1
        (identity program == base) and deterministic distillation both need
        base dropout off, and there is nothing trainable inside the wrapper.
        Field/encoder modules live outside and manage their own modes."""
        super().train(mode)
        self.base.eval()
        return self

    # -- passthroughs -----------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

    @property
    def config(self):
        return self.base.config

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def identity_field(self, enable_gains: bool = False, trainable: bool = False) -> ProgramField:
        return ProgramField.identity(self.spec, enable_gains=enable_gains, trainable=trainable)
