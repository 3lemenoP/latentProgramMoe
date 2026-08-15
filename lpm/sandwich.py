"""lpm/sandwich.py — sandwiched module forwards (spec §3).

Wraps at the MODULE-FORWARD level, never the weight level: wrapper nn.Modules
hold references to the frozen base submodules plus a shared ProgramState; base
weights are never modified or materialized in rotated form. With no program
installed (state.field is None) every wrapper delegates verbatim to the base
module — the unwrapped path bit-for-bit.

Written against transformers v5 (>=5.0): attention modules return
(attn_output, attn_weights) and dispatch the inner product through
ALL_ATTENTION_FUNCTIONS; we reuse the exact same interface lookup so the
sandwich path uses the same attention backend as the base (T4 parity).

qk_rel placement (spec §3.3): the spec writes the relative transport as
k <- M k inside the logits. We implement the numerically identical q-side form
q <- M^T q (since q^T (M k) = (M^T q)^T k), applied AFTER base RoPE when the
base has RoPE. Why q-side: (a) the qk_rel rotation M never enters the KV
cache; (b) under GQA (Llama-family) the per-attention-head field applies to q,
which has the full head count, without touching shared KV heads. Equivalence
is asserted by tests/test_sandwich.py::test_qk_rel_query_side_equivalence.
NOTE the cache is still program-DEPENDENT: cached k/v are computed from
u = R(attn_io)^T x, and x itself passed through earlier sandwiched layers —
never swap programs mid-generation with a live cache (any placement of M
would have this property; tested one-program cache parity only).
With gains enabled the k-side form is k <- g ⊙ (M k); its exact q-side
transpose is q <- M^T (g ⊙ q) — gain BEFORE the inverse rotation.

Gain attachment points (spec §4: "immediately after the corresponding forward
rotation"): attn_io / mlp_io at module exit (out <- g ⊙ (R y)); ffn_hidden on
entry to the activation frame (a <- g ⊙ (S a)); qk_rel per the transpose above.
For SwiGLU the ffn_hidden gain is applied to the up branch only (the product
h = silu(g_gate) ⊙ a is linear in a, so one application scales the hidden unit
once; applying it to both branches would square it).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .field import FieldSpec
from .gains import apply_gain, apply_gain_head
from .quaternion import apply_rot, apply_rot_head

try:  # transformers is required for the sandwich/model layer, not for kernels
    from transformers.cache_utils import EncoderDecoderCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gpt2.modeling_gpt2 import (
        eager_attention_forward as gpt2_eager_attention_forward,
    )
except ImportError as _e:  # pragma: no cover
    EncoderDecoderCache = None
    ALL_ATTENTION_FUNCTIONS = None
    gpt2_eager_attention_forward = None


class ProgramState:
    """Shared mutable slot for the currently installed ProgramField.

    All sandwich wrappers of one model hold the same instance; the program()
    context manager (lpm/model_wrap.py) swaps .field. Rotations/gains are
    fetched lazily via field.rotations()/field.gains(), whose caching handles
    both the constant-field inference case (build once) and the training case
    (rebuild per step, graph attached)."""

    def __init__(self) -> None:
        self.field = None


class _SandwichBase(nn.Module):
    def __init__(self, base: nn.Module, layer_idx: int, state: ProgramState, spec: FieldSpec):
        super().__init__()
        self.base = base
        self.layer_idx = layer_idx
        self.spec = spec
        self.state = state
        self.d_part = spec.partition("attn_io")   # == mlp_io partition (d_model)
        self.f_part = spec.partition("ffn_hidden")
        self.h_part = spec.partition("qk_rel")
        self.ax_part = (spec.partition("rope_ax") if spec.rotary_ndims > 0 else None)


# ---------------------------------------------------------------------------
# GPT-2
# ---------------------------------------------------------------------------
class SandwichGPT2Attention(_SandwichBase):
    """Spec §3.3 around transformers v5 GPT2Attention.

    u   = R^T x                     (enter module frame; R = R(attn_io))
    qkv = c_attn(u), split to heads
    q   = M^T q                     (relative transport, q-side form; M = R(qk_rel))
    attn= softmax(q k^T * scaling + mask) v      (base interface, untouched)
    o   = c_proj(merge_heads(attn)); out = R o   (exit to residual frame)
    """

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, **kwargs):
        field = self.state.field
        if field is None:
            return self.base(hidden_states, past_key_values=past_key_values,
                             attention_mask=attention_mask,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask,
                             output_attentions=output_attentions, **kwargs)
        if encoder_hidden_states is not None:
            raise NotImplementedError("sandwich does not support cross-attention")
        base = self.base
        if base.reorder_and_upcast_attn:
            raise NotImplementedError("reorder_and_upcast_attn is unsupported under a program")

        rot = field.rotations()
        gains = field.gains()
        R = rot[(self.layer_idx, "attn_io")]
        M = rot[(self.layer_idx, "qk_rel")]

        u = apply_rot(hidden_states, R, self.d_part, inverse=True)

        query_states, key_states, value_states = base.c_attn(u).split(base.split_size, dim=2)
        shape = (*key_states.shape[:-1], -1, base.head_dim)
        query_states = query_states.view(shape).transpose(1, 2)
        key_states = key_states.view(shape).transpose(1, 2)
        value_states = value_states.view(shape).transpose(1, 2)

        # q-side relative transport (== k <- g ⊙ (M k) on the k side)
        if gains is not None:
            query_states = apply_gain_head(query_states, gains[(self.layer_idx, "qk_rel")], self.h_part)
        query_states = apply_rot_head(query_states, M, self.h_part, inverse=True)

        if past_key_values is not None:
            curr = (past_key_values.self_attention_cache
                    if isinstance(past_key_values, EncoderDecoderCache) else past_key_values)
            key_states, value_states = curr.update(key_states, value_states, base.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, gpt2_eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            base, query_states, key_states, value_states, attention_mask,
            dropout=base.attn_dropout.p if base.training else 0.0,
            scaling=base.scaling, **kwargs)

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = base.c_proj(attn_output)
        attn_output = base.resid_dropout(attn_output)

        out = apply_rot(attn_output, R, self.d_part)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "attn_io")], self.d_part)
        return out, attn_weights


class SandwichGPT2MLP(_SandwichBase):
    """Spec §3.1 (GELU variant). Both R and S sandwich a nonlinearity."""

    def forward(self, hidden_states):
        field = self.state.field
        if field is None:
            return self.base(hidden_states)
        base = self.base
        rot = field.rotations()
        gains = field.gains()
        R = rot[(self.layer_idx, "mlp_io")]
        S = rot[(self.layer_idx, "ffn_hidden")]

        u = apply_rot(hidden_states, R, self.d_part, inverse=True)
        a = base.c_fc(u)
        a = apply_rot(a, S, self.f_part)
        if gains is not None:
            a = apply_gain(a, gains[(self.layer_idx, "ffn_hidden")], self.f_part)
        h = base.act(a)
        h = apply_rot(h, S, self.f_part, inverse=True)
        y = base.c_proj(h)
        y = apply_rot(y, R, self.d_part)
        if gains is not None:
            y = apply_gain(y, gains[(self.layer_idx, "mlp_io")], self.d_part)
        return base.dropout(y)


# ---------------------------------------------------------------------------
# GPT-NeoX / Pythia (partial rotary + parallel residual)
# ---------------------------------------------------------------------------
class SandwichNeoXAttention(_SandwichBase):
    """Steering doc B0 around transformers v5 GPTNeoXAttention.

    u    = R^T x                              (attn_io, module frame)
    qkv  = query_key_value(u), chunk to heads
    axis conjugation of the rotary generators (rope_ax; Ω → A Ω Aᵀ done as a
    sandwich around the base rotary call, Ω itself untouched):
        q[..., :nd] ← A · RoPE_m( Aᵀ · q[..., :nd] ),  SAME A on k (mandatory:
        different rotations on q/k would break positional relativity)
    q    = M^T q                              (qk_rel post-RoPE, q-side —
                                               retained as the static control)
    attn = base interface; dense; out = R o   (exit to residual frame)

    Correctness runs must use eager attention (rotary monkeypatch is
    sdpa/flash-version-sensitive) — enforced by from_pretrained defaults.
    """

    def forward(self, hidden_states, attention_mask=None, layer_past=None,
                position_embeddings=None, **kwargs):
        field = self.state.field
        if field is None:
            return self.base(hidden_states, attention_mask=attention_mask,
                             layer_past=layer_past,
                             position_embeddings=position_embeddings, **kwargs)
        base = self.base
        from transformers.models.gpt_neox.modeling_gpt_neox import (
            apply_rotary_pos_emb,
            eager_attention_forward as neox_eager_attention_forward,
        )

        rot = field.rotations()
        gains = field.gains()
        R = rot[(self.layer_idx, "attn_io")]
        M = rot[(self.layer_idx, "qk_rel")]
        A = rot.get((self.layer_idx, "rope_ax"))

        u = apply_rot(hidden_states, R, self.d_part, inverse=True)

        input_shape = u.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * base.head_size)
        qkv = base.query_key_value(u).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        if A is not None:
            nd = self.spec.rotary_ndims
            q_rot = apply_rot_head(query_states[..., :nd], A, self.ax_part, inverse=True)
            k_rot = apply_rot_head(key_states[..., :nd], A, self.ax_part, inverse=True)
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            # apply_rotary_pos_emb only rotates cos.shape[-1] dims; q_rot/k_rot
            # are exactly those dims here, pass-through dims stay outside
            q_rot = apply_rot_head(q_rot, A, self.ax_part)
            k_rot = apply_rot_head(k_rot, A, self.ax_part)
            query_states = torch.cat([q_rot, query_states[..., nd:]], dim=-1)
            key_states = torch.cat([k_rot, key_states[..., nd:]], dim=-1)
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)

        # q-side relative transport (static, post-RoPE): degenerate control
        # for the axis field
        if gains is not None:
            query_states = apply_gain_head(query_states, gains[(self.layer_idx, "qk_rel")], self.h_part)
        query_states = apply_rot_head(query_states, M, self.h_part, inverse=True)

        if layer_past is not None:
            key_states, value_states = layer_past.update(key_states, value_states, base.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, neox_eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            base, query_states, key_states, value_states, attention_mask,
            scaling=base.scaling,
            dropout=0.0 if not base.training else base.attention_dropout,
            **kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = base.dense(attn_output)

        out = apply_rot(attn_output, R, self.d_part)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "attn_io")], self.d_part)
        return out, attn_weights


class SandwichNeoXMLP(_SandwichBase):
    """GELU MLP, same shape as the GPT-2 sandwich (spec §3.1)."""

    def forward(self, hidden_states):
        field = self.state.field
        if field is None:
            return self.base(hidden_states)
        base = self.base
        rot = field.rotations()
        gains = field.gains()
        R = rot[(self.layer_idx, "mlp_io")]
        S = rot[(self.layer_idx, "ffn_hidden")]

        u = apply_rot(hidden_states, R, self.d_part, inverse=True)
        a = base.dense_h_to_4h(u)
        a = apply_rot(a, S, self.f_part)
        if gains is not None:
            a = apply_gain(a, gains[(self.layer_idx, "ffn_hidden")], self.f_part)
        h = base.act(a)
        h = apply_rot(h, S, self.f_part, inverse=True)
        y = base.dense_4h_to_h(h)
        y = apply_rot(y, R, self.d_part)
        if gains is not None:
            y = apply_gain(y, gains[(self.layer_idx, "mlp_io")], self.d_part)
        return y


# ---------------------------------------------------------------------------
# Llama family (SwiGLU + RoPE)
# ---------------------------------------------------------------------------
class SandwichLlamaAttention(_SandwichBase):
    """Spec §3.3 around transformers v5 LlamaAttention. M applies AFTER base
    RoPE (2-blocks and 3-blocks don't commute; this order is the project
    convention), on the q side per the module docstring."""

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        field = self.state.field
        if field is None:
            return self.base(hidden_states, position_embeddings=position_embeddings,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values, **kwargs)
        base = self.base
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            eager_attention_forward as llama_eager_attention_forward,
        )

        rot = field.rotations()
        gains = field.gains()
        R = rot[(self.layer_idx, "attn_io")]
        M = rot[(self.layer_idx, "qk_rel")]

        u = apply_rot(hidden_states, R, self.d_part, inverse=True)

        input_shape = u.shape[:-1]
        hidden_shape = (*input_shape, -1, base.head_dim)
        query_states = base.q_proj(u).view(hidden_shape).transpose(1, 2)
        key_states = base.k_proj(u).view(hidden_shape).transpose(1, 2)
        value_states = base.v_proj(u).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # AFTER RoPE: q-side relative transport; q has the full head count even
        # under GQA, and the KV cache below stays in the base frame.
        if gains is not None:
            query_states = apply_gain_head(query_states, gains[(self.layer_idx, "qk_rel")], self.h_part)
        query_states = apply_rot_head(query_states, M, self.h_part, inverse=True)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, base.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, llama_eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            base, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not base.training else base.attention_dropout,
            scaling=base.scaling, **kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = base.o_proj(attn_output)

        out = apply_rot(attn_output, R, self.d_part)
        if gains is not None:
            out = apply_gain(out, gains[(self.layer_idx, "attn_io")], self.d_part)
        return out, attn_weights


class SandwichLlamaMLP(_SandwichBase):
    """Spec §3.2 (SwiGLU): rotate BOTH branches by S before the elementwise
    ops, inverse after the product. ffn_hidden gain on the up branch only."""

    def forward(self, x):
        field = self.state.field
        if field is None:
            return self.base(x)
        base = self.base
        rot = field.rotations()
        gains = field.gains()
        R = rot[(self.layer_idx, "mlp_io")]
        S = rot[(self.layer_idx, "ffn_hidden")]

        u = apply_rot(x, R, self.d_part, inverse=True)
        g = base.act_fn(apply_rot(base.gate_proj(u), S, self.f_part))
        a = apply_rot(base.up_proj(u), S, self.f_part)
        if gains is not None:
            a = apply_gain(a, gains[(self.layer_idx, "ffn_hidden")], self.f_part)
        h = apply_rot(g * a, S, self.f_part, inverse=True)
        y = base.down_proj(h)
        y = apply_rot(y, R, self.d_part)
        if gains is not None:
            y = apply_gain(y, gains[(self.layer_idx, "mlp_io")], self.d_part)
        return y
