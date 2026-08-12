"""lpm/sandwich.py — sandwiched module forwards (spec §3).

Wrapping happens at the MODULE-FORWARD level, never the weight level: each
wrapper holds a reference to the frozen base submodule and re-runs its exact
computation with activation rotations inserted. Base weights are never modified
or materialized in rotated form. With no program installed (state.field is
None) the wrapper delegates to the base forward verbatim.

Attention / MLP structure (spec §3.1-§3.3), with R = R(attn_io) or R(mlp_io),
S = R(ffn_hidden), M = R(qk_rel) per head:

    u   = R^T x               enter module frame (x is the post-LN input)
    ... base projections ...
    out = R y                 exit to residual frame

The qk_rel rotation enters the logits as q^T (M k) — the gauge-invariant
relative element. Two equivalent placements are used:

  * GPT-2 (no RoPE): k <- M k after the k projection (spec-literal form).
    NOTE: with use_cache=True the KV cache then holds ROTATED keys — do not
    swap programs mid-generation with a live cache.
  * Llama (RoPE + GQA): q <- M^T q AFTER RoPE. Identical logits
    (q^T M k == (M^T q)^T k), but GQA-safe (k has fewer heads than the
    per-attention-head field) and the KV cache stays program-independent.
    Order convention (spec §3.3): the relative rotation applies after RoPE.

Abelian gains (spec §4, if the field has them) apply immediately after the
corresponding forward rotation. For SwiGLU, where S appears on both branches,
the ffn_hidden gain is applied ONCE, on the up (value) branch, so the product
carries the gain exactly once — mirroring the single-branch GELU case.

Dropout submodules stay inside the sandwich (wrap-the-module semantics); run
the model in eval() during field training so distillation is deterministic.

Written against transformers v5 module interfaces (GPT2Attention/GPT2MLP,
LlamaAttention/LlamaMLP return conventions); T4/T5 assert parity with the
unwrapped base, so any drift in those interfaces fails loudly in tests.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import (
    eager_attention_forward as gpt2_eager_attention_forward,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward as llama_eager_attention_forward,
)

from .field import FieldSpec
from .gains import apply_gain, apply_gain_head
from .quaternion import apply_rot, apply_rot_head


class ProgramState:
    """Shared mutable slot holding the currently installed ProgramField (or
    None = raw base model). One instance per wrapped model; every sandwich
    wrapper keeps a reference."""

    def __init__(self):
        self.field = None


def _check_device(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if R.device != x.device:
        raise RuntimeError(
            f"ProgramField rotations live on {R.device} but activations on "
            f"{x.device}; move the field first: field.to(model_device)")
    return R


class SandwichGPT2Attention(nn.Module):
    """Spec §3.3 around a frozen transformers GPT2Attention (self-attention only)."""

    def __init__(self, base, layer_idx: int, state: ProgramState, spec: FieldSpec):
        super().__init__()
        if getattr(base, "is_cross_attention", False):
            raise NotImplementedError("cross-attention sandwich is out of scope (v1)")
        self.base = base
        self.layer_idx = layer_idx
        self.spec = spec
        self.part_io = spec.partition("attn_io")
        self.part_head = spec.partition("qk_rel")
        # plain attribute on purpose: shared state, not a submodule
        object.__setattr__(self, "state", state)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        if self.state.field is None:
            return self.base(hidden_states, past_key_values=past_key_values,
                             attention_mask=attention_mask,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask, **kwargs)
        if encoder_hidden_states is not None:
            raise NotImplementedError("cross-attention sandwich is out of scope (v1)")

        base = self.base
        field = self.state.field
        rot = field.rotations()
        gains = field.gains()
        R = _check_device(rot[(self.layer_idx, "attn_io")], hidden_states)
        M = rot[(self.layer_idx, "qk_rel")]

        # enter module frame
        u = apply_rot(hidden_states, R, self.part_io, inverse=True)

        query_states, key_states, value_states = base.c_attn(u).split(base.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], -1, base.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, base.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        # relative key transport (GPT-2 has no RoPE; spec §3.3)
        key_states = apply_rot_head(key_states, M, self.part_head)
        if gains is not None:
            key_states = apply_gain_head(key_states, gains[(self.layer_idx, "qk_rel")],
                                         self.part_head)

        if past_key_values is not None:
            curr = (past_key_values.self_attention_cache
                    if isinstance(past_key_values, EncoderDecoderCache) else past_key_values)
            key_states, value_states = curr.update(key_states, value_states, base.layer_idx)

        using_eager = base.config._attn_implementation == "eager"
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, gpt2_eager_attention_forward)

        if using_eager and base.reorder_and_upcast_attn:
            attn_output, attn_weights = base._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask)
        else:
            attn_output, attn_weights = attention_interface(
                base, query_states, key_states, value_states, attention_mask,
                dropout=base.attn_dropout.p if base.training else 0.0,
                scaling=base.scaling, **kwargs)

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        o = base.c_proj(attn_output)
        o = base.resid_dropout(o)

        # exit to residual frame
        out = apply_rot(o, R, self.part_io)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "attn_io")], self.part_io)
        return out, attn_weights


class SandwichGPT2MLP(nn.Module):
    """Spec §3.1 (GELU variant) around a frozen transformers GPT2MLP."""

    def __init__(self, base, layer_idx: int, state: ProgramState, spec: FieldSpec):
        super().__init__()
        self.base = base
        self.layer_idx = layer_idx
        self.spec = spec
        self.part_io = spec.partition("mlp_io")
        self.part_hidden = spec.partition("ffn_hidden")
        object.__setattr__(self, "state", state)

    def forward(self, hidden_states):
        if self.state.field is None:
            return self.base(hidden_states)
        base = self.base
        field = self.state.field
        rot = field.rotations()
        gains = field.gains()
        R = _check_device(rot[(self.layer_idx, "mlp_io")], hidden_states)
        S = rot[(self.layer_idx, "ffn_hidden")]

        u = apply_rot(hidden_states, R, self.part_io, inverse=True)
        a = base.c_fc(u)
        a = apply_rot(a, S, self.part_hidden)
        if gains is not None:
            a = apply_gain(a, gains[(self.layer_idx, "ffn_hidden")], self.part_hidden)
        h = base.act(a)
        h = apply_rot(h, S, self.part_hidden, inverse=True)
        y = base.c_proj(h)
        y = base.dropout(y)
        out = apply_rot(y, R, self.part_io)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "mlp_io")], self.part_io)
        return out


class SandwichLlamaAttention(nn.Module):
    """Spec §3.3 around a frozen transformers LlamaAttention.

    Relative transport applied on the QUERY side after RoPE: q <- M^T q, which
    equals the spec's q^T (M k) logits exactly and is GQA/cache-safe (see
    module docstring)."""

    def __init__(self, base, layer_idx: int, state: ProgramState, spec: FieldSpec):
        super().__init__()
        self.base = base
        self.layer_idx = layer_idx
        self.spec = spec
        self.part_io = spec.partition("attn_io")
        self.part_head = spec.partition("qk_rel")
        object.__setattr__(self, "state", state)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        if self.state.field is None:
            return self.base(hidden_states, position_embeddings=position_embeddings,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values, **kwargs)
        base = self.base
        field = self.state.field
        rot = field.rotations()
        gains = field.gains()
        R = _check_device(rot[(self.layer_idx, "attn_io")], hidden_states)
        M = rot[(self.layer_idx, "qk_rel")]

        u = apply_rot(hidden_states, R, self.part_io, inverse=True)
        input_shape = u.shape[:-1]
        hidden_shape = (*input_shape, -1, base.head_dim)

        query_states = base.q_proj(u).view(hidden_shape).transpose(1, 2)
        key_states = base.k_proj(u).view(hidden_shape).transpose(1, 2)
        value_states = base.v_proj(u).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # relative transport AFTER RoPE (spec §3.3), query side: q <- M^T q
        query_states = apply_rot_head(query_states, M, self.part_head, inverse=True)
        if gains is not None:
            query_states = apply_gain_head(query_states, gains[(self.layer_idx, "qk_rel")],
                                           self.part_head)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states,
                                                              base.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, llama_eager_attention_forward)

        attn_output, attn_weights = attention_interface(
            base, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not base.training else base.attention_dropout,
            scaling=base.scaling, **kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        o = base.o_proj(attn_output)

        out = apply_rot(o, R, self.part_io)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "attn_io")], self.part_io)
        return out, attn_weights


class SandwichLlamaMLP(nn.Module):
    """Spec §3.2 (SwiGLU variant) around a frozen transformers LlamaMLP."""

    def __init__(self, base, layer_idx: int, state: ProgramState, spec: FieldSpec):
        super().__init__()
        self.base = base
        self.layer_idx = layer_idx
        self.spec = spec
        self.part_io = spec.partition("mlp_io")
        self.part_hidden = spec.partition("ffn_hidden")
        object.__setattr__(self, "state", state)

    def forward(self, hidden_states):
        if self.state.field is None:
            return self.base(hidden_states)
        base = self.base
        field = self.state.field
        rot = field.rotations()
        gains = field.gains()
        R = _check_device(rot[(self.layer_idx, "mlp_io")], hidden_states)
        S = rot[(self.layer_idx, "ffn_hidden")]

        u = apply_rot(hidden_states, R, self.part_io, inverse=True)
        g = base.act_fn(apply_rot(base.gate_proj(u), S, self.part_hidden))
        a = apply_rot(base.up_proj(u), S, self.part_hidden)
        if gains is not None:  # gain once, on the up branch (module docstring)
            a = apply_gain(a, gains[(self.layer_idx, "ffn_hidden")], self.part_hidden)
        h = apply_rot(g * a, S, self.part_hidden, inverse=True)
        y = base.down_proj(h)
        out = apply_rot(y, R, self.part_io)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "mlp_io")], self.part_io)
        return out
