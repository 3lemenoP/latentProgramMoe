"""Model-level theory-in-code tests (spec §9): T4, T5, T6, T7, T10 (+ T3 at the
model level), cache/mask parity, and the Llama (RoPE+GQA) adapter.

Fast tiny-random fixtures run everywhere; `-m gpt2` additionally runs the
acceptance criteria on the real pretrained GPT-2 small (spec M1 gate).
"""
import pytest
import torch

from lpm import LatentProgramModel
from lpm.field import ProgramField
from lpm.quaternion import apply_rot_head, make_partition, q_normalize, q_to_R

torch.manual_seed(0)


def logits(model, ids, field=None, **kw):
    with torch.no_grad():
        with model.program(field):
            return model(input_ids=ids, **kw).logits


def rand_field(model, sigma, gains=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ProgramField.randn_near_identity(model.spec, sigma=sigma,
                                            enable_gains=gains, generator=g)


# ---------------------------------------------------------------------------
# T4 — identity program == base model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["tiny_gpt2", "tiny_llama"])
def test_t4_identity_program_is_base(fixture, tiny_ids, request):
    model = request.getfixturevalue(fixture)
    base = logits(model, tiny_ids)
    ident = logits(model, tiny_ids, model.identity_field())
    assert (ident - base).abs().max() < 1e-4


@pytest.mark.gpt2
def test_t4_identity_program_is_base_real(real_gpt2, real_gpt2_ids):
    base = logits(real_gpt2, real_gpt2_ids)
    ident = logits(real_gpt2, real_gpt2_ids, real_gpt2.identity_field())
    assert (ident - base).abs().max() < 1e-4  # spec: <1e-4 fp32 on real text


# ---------------------------------------------------------------------------
# T3 (model level) — global sign flip of a field leaves logits identical
# ---------------------------------------------------------------------------
def test_t3_sign_flip_same_logits(tiny_gpt2, tiny_ids):
    f = rand_field(tiny_gpt2, sigma=0.3)
    a = logits(tiny_gpt2, tiny_ids, f)
    b = logits(tiny_gpt2, tiny_ids, f.flipped_sign())
    assert (a - b).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# T5 — dead value frame: per-sequence conjugation of W_v / W_o cancels exactly
# ---------------------------------------------------------------------------
def _head_blockdiag(n_heads, d_head, seed=3):
    """Per-head block-diagonal rotation D over the merged head dim, built from
    the same certified kernels the sandwich uses."""
    g = torch.Generator().manual_seed(seed)
    part = make_partition(d_head)
    q = q_normalize(torch.randn(n_heads, part.n3, 4, generator=g))
    R = q_to_R(q)  # (H, n3, 3, 3)
    blocks = []
    for h in range(n_heads):
        B = torch.eye(d_head)
        for i in range(part.n3):
            B[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R[h, i]
        blocks.append(B)
    return torch.block_diag(*blocks)  # (H*d_head, H*d_head)


def _t5_apply(model_wrap, ids, atol):
    base = model_wrap.base
    cfg = base.config
    D = _head_blockdiag(cfg.n_head, cfg.n_embd // cfg.n_head)
    before = logits(model_wrap, ids)
    d = cfg.n_embd
    mods = []
    with torch.no_grad():
        for block in base.transformer.h:
            attn = block.attn.base  # frozen original under the sandwich wrapper
            Wv = attn.c_attn.weight[:, 2 * d:3 * d].clone()
            bv = attn.c_attn.bias[2 * d:3 * d].clone()
            Wo = attn.c_proj.weight.clone()
            mods.append((attn, Wv, bv, Wo))
            attn.c_attn.weight[:, 2 * d:3 * d] = Wv @ D.T  # v -> v D^T (row conv.)
            attn.c_attn.bias[2 * d:3 * d] = bv @ D.T
            attn.c_proj.weight.copy_(D @ Wo)               # undo before W_o
    try:
        after = logits(model_wrap, ids)
    finally:
        with torch.no_grad():
            for attn, Wv, bv, Wo in mods:
                attn.c_attn.weight[:, 2 * d:3 * d] = Wv
                attn.c_attn.bias[2 * d:3 * d] = bv
                attn.c_proj.weight.copy_(Wo)
    assert (after - before).abs().max() < atol


def test_t5_dead_value_frame(tiny_gpt2, tiny_ids):
    _t5_apply(tiny_gpt2, tiny_ids, atol=1e-4)


@pytest.mark.gpt2
def test_t5_dead_value_frame_real(real_gpt2, real_gpt2_ids):
    _t5_apply(real_gpt2, real_gpt2_ids, atol=1e-4)


# ---------------------------------------------------------------------------
# T6 — spectrum preservation on actual model weights
# ---------------------------------------------------------------------------
def test_t6_spectrum_preserved_on_model_weight(tiny_gpt2):
    f = rand_field(tiny_gpt2, sigma=0.5, seed=5)
    R = f.rotations()[(0, "attn_io")]
    part = tiny_gpt2.spec.partition("attn_io")
    B = torch.eye(part.dim, dtype=torch.float64)
    for i in range(part.n3):
        B[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R[i].double()
    W = tiny_gpt2.base.transformer.h[0].attn.base.c_proj.weight.detach().double()
    s0 = torch.linalg.svdvals(W)
    s1 = torch.linalg.svdvals(B @ W @ B.T)
    assert (torch.sort(s0).values - torch.sort(s1).values).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# T7 — the program is live
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["tiny_gpt2", "tiny_llama"])
def test_t7_program_is_live(fixture, tiny_ids, request):
    model = request.getfixturevalue(fixture)
    base = logits(model, tiny_ids)
    prog = logits(model, tiny_ids, rand_field(model, sigma=0.3, seed=7))
    diff = (prog - base).abs().mean()
    assert diff > 0.05 * base.abs().mean().clamp_min(1e-6)
    assert diff > 1e-3


@pytest.mark.gpt2
def test_t7_program_is_live_real(real_gpt2, real_gpt2_ids):
    base = logits(real_gpt2, real_gpt2_ids)
    prog = logits(real_gpt2, real_gpt2_ids, rand_field(real_gpt2, sigma=0.3, seed=7))
    assert (prog - base).abs().mean() > 0.1  # spec criterion


# ---------------------------------------------------------------------------
# T10 — gains: rho=0 preserves T4; composition is additive in rho
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["tiny_gpt2", "tiny_llama"])
def test_t10_gains_identity(fixture, tiny_ids, request):
    model = request.getfixturevalue(fixture)
    base = logits(model, tiny_ids)
    ident_g = logits(model, tiny_ids, model.identity_field(enable_gains=True))
    assert (ident_g - base).abs().max() < 1e-4


def test_t10_gains_compose_additively(tiny_gpt2):
    from lpm.compose import compose
    a = rand_field(tiny_gpt2, sigma=0.1, gains=True, seed=1)
    b = rand_field(tiny_gpt2, sigma=0.1, gains=True, seed=2)
    with torch.no_grad():
        for k in a.sites():
            a.rho(*k).normal_(std=0.1)
            b.rho(*k).normal_(std=0.1)
    c = compose(b, a)
    for k in a.sites():
        assert torch.allclose(c.rho(*k), a.rho(*k) + b.rho(*k), atol=1e-6)


def test_gains_change_logits(tiny_gpt2, tiny_ids):
    f = tiny_gpt2.identity_field(enable_gains=True)
    with torch.no_grad():
        for k in f.sites():
            f.rho(*k).normal_(std=0.2)
    base = logits(tiny_gpt2, tiny_ids)
    gained = logits(tiny_gpt2, tiny_ids, f)
    assert (gained - base).abs().mean() > 1e-4


# ---------------------------------------------------------------------------
# parity: attention mask, KV cache, hidden-state capture, generate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["tiny_gpt2", "tiny_llama"])
def test_padded_mask_identity_parity(fixture, tiny_ids, request):
    model = request.getfixturevalue(fixture)
    mask = torch.ones_like(tiny_ids)
    mask[0, :5] = 0  # left padding on row 0
    base = logits(model, tiny_ids, None, attention_mask=mask)
    ident = logits(model, tiny_ids, model.identity_field(), attention_mask=mask)
    assert (ident - base).abs().max() < 1e-4


@pytest.mark.parametrize("fixture", ["tiny_gpt2", "tiny_llama"])
def test_kv_cache_parity_under_program(fixture, tiny_ids, request):
    """Incremental decoding with a live (non-identity) program must match the
    full forward — validates the k-rotation/cache interaction."""
    model = request.getfixturevalue(fixture)
    f = rand_field(model, sigma=0.3, seed=11)
    ids = tiny_ids[:1]
    with torch.no_grad(), model.program(f):
        full = model(input_ids=ids).logits
        o1 = model(input_ids=ids[:, :10], use_cache=True)
        o2 = model(input_ids=ids[:, 10:], past_key_values=o1.past_key_values,
                   use_cache=True)
        inc = torch.cat([o1.logits, o2.logits], dim=1)
    assert (full - inc).abs().max() < 1e-4


def test_generate_identity_parity(tiny_gpt2, tiny_ids):
    ids = tiny_ids[:1, :8]
    with torch.no_grad():
        want = tiny_gpt2.generate(ids, max_new_tokens=8, do_sample=False)
        with tiny_gpt2.program(tiny_gpt2.identity_field()):
            got = tiny_gpt2.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(want, got)


def test_hidden_states_capture_with_program(tiny_gpt2, tiny_ids):
    """Phase 1 needs per-layer hidden states through the sandwiched forward."""
    f = rand_field(tiny_gpt2, sigma=0.2, seed=13)
    with torch.no_grad(), tiny_gpt2.program(f):
        out = tiny_gpt2(input_ids=tiny_ids, output_hidden_states=True)
    assert out.hidden_states is not None
    assert len(out.hidden_states) >= tiny_gpt2.n_layers


# ---------------------------------------------------------------------------
# qk_rel placement equivalence (kernel level): q^T (M k) == (M^T q)^T k
# ---------------------------------------------------------------------------
def test_qk_rel_query_side_equivalence():
    g = torch.Generator().manual_seed(17)
    part = make_partition(12)
    q = torch.randn(2, 4, 9, 12, generator=g)
    k = torch.randn(2, 4, 9, 12, generator=g)
    M = q_to_R(q_normalize(torch.randn(4, part.n3, 4, generator=g)))
    key_side = torch.einsum('bhtd,bhsd->bhts', q, apply_rot_head(k, M, part))
    query_side = torch.einsum('bhtd,bhsd->bhts',
                              apply_rot_head(q, M, part, inverse=True), k)
    assert (key_side - query_side).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# program() context manager semantics
# ---------------------------------------------------------------------------
def test_program_context_restores_previous(tiny_gpt2):
    f1 = rand_field(tiny_gpt2, sigma=0.1, seed=1)
    f2 = rand_field(tiny_gpt2, sigma=0.1, seed=2)
    assert tiny_gpt2.active_program is None
    with tiny_gpt2.program(f1):
        assert tiny_gpt2.active_program is f1
        with tiny_gpt2.program(f2):
            assert tiny_gpt2.active_program is f2
        assert tiny_gpt2.active_program is f1
    assert tiny_gpt2.active_program is None


def test_spec_mismatch_rejected(tiny_gpt2, tiny_llama):
    f = tiny_llama.identity_field()
    with pytest.raises(ValueError):
        with tiny_gpt2.program(f):
            pass


def test_base_params_frozen(tiny_gpt2):
    assert all(not p.requires_grad for p in tiny_gpt2.base.parameters())
