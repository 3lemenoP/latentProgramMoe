"""LPN encoder tests (spec §5): identity-at-init, shapes, gradient routing,
refinement, demo formatting."""
import pytest
import torch

from lpm.encoder import LPNEncoder, build_demo_lm_batch, format_demos, refine
from lpm.field import ProgramField
from lpm.quaternion import d2_chord, q_normalize


def make_encoder(model, gains=False, seed=0):
    torch.manual_seed(seed)
    return LPNEncoder(model.spec, vocab_size=model.config.vocab_size,
                      pad_token_id=None, d_enc=64, n_layers=2, n_heads=4,
                      dim_ff=128, max_len=64, rank=8, enable_gains=gains)


def demo_ids(B=2, T=32, vocab=257, seed=5):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (B, T), generator=g)
    mask = torch.ones_like(ids)
    mask[1, T // 2:] = 0
    return ids, mask


def test_encoder_emits_identity_at_init(tiny_gpt2):
    enc = make_encoder(tiny_gpt2).eval()
    ids, mask = demo_ids()
    fields = enc(ids, mask)
    assert len(fields) == 2
    ident = ProgramField.identity(tiny_gpt2.spec)
    for f in fields:
        for k in f.sites():
            assert d2_chord(q_normalize(f.q(*k)), ident.q(*k)).max() < 1e-10


def test_encoder_identity_init_is_base_behavior(tiny_gpt2, tiny_ids):
    from test_sandwich import logits
    enc = make_encoder(tiny_gpt2).eval()
    ids, mask = demo_ids()
    field = enc(ids, mask)[0]
    assert (logits(tiny_gpt2, tiny_ids, field) -
            logits(tiny_gpt2, tiny_ids)).abs().max() < 1e-4


def test_encoder_site_shapes_and_gains(tiny_gpt2):
    enc = make_encoder(tiny_gpt2, gains=True).eval()
    ids, mask = demo_ids()
    f = enc(ids, mask)[0]
    for (l, n) in tiny_gpt2.spec.site_keys():
        assert tuple(f.q(l, n).shape) == (*tiny_gpt2.spec.site_shape(n), 4)
        assert tuple(f.rho(l, n).shape) == tiny_gpt2.spec.site_shape(n)
        assert f.rho(l, n).abs().max() == 0  # zero-init


def test_gradients_flow_to_encoder_only(tiny_gpt2, tiny_ids):
    enc = make_encoder(tiny_gpt2).train()
    ids, mask = demo_ids()
    field = enc(ids, mask)[0]
    with tiny_gpt2.program(field):
        out = tiny_gpt2(input_ids=tiny_ids, labels=tiny_ids)
    out.loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().max() > 0 for g in grads)
    assert all(p.grad is None for p in tiny_gpt2.base.parameters())


def test_encoder_output_differs_after_training_step(tiny_gpt2):
    """One optimizer step on a geo loss moves the emitted program off identity."""
    enc = make_encoder(tiny_gpt2).train()
    target = ProgramField.randn_near_identity(tiny_gpt2.spec, sigma=0.3)
    ids, mask = demo_ids()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    f = enc(ids, mask)[0]
    loss = sum(d2_chord(q_normalize(f.q(*k)), q_normalize(target.q(*k))).sum()
               for k in f.sites())
    loss.backward()
    opt.step()
    f2 = enc(ids, mask)[0]
    ident = ProgramField.identity(tiny_gpt2.spec)
    moved = max(d2_chord(q_normalize(f2.q(*k)), ident.q(*k)).max().item()
                for k in f2.sites())
    assert moved > 1e-8


def test_refine_decreases_demo_loss(tiny_gpt2):
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(0, 257, (2, 16), generator=g)
    field = ProgramField.identity(tiny_gpt2.spec)

    def demo_loss(f):
        with torch.no_grad(), tiny_gpt2.program(f):
            return tiny_gpt2(input_ids=ids, labels=ids).loss.item()

    before = demo_loss(field)
    refined = refine(tiny_gpt2, field, ids, labels=ids, steps=10, lr=5e-2)
    after = demo_loss(refined)
    assert refined.trainable
    assert after < before


def test_refine_steps_zero_is_noop(tiny_gpt2):
    f = ProgramField.randn_near_identity(tiny_gpt2.spec, sigma=0.2)
    ids = torch.randint(0, 257, (1, 8))
    r = refine(tiny_gpt2, f, ids, labels=ids, steps=0)
    for k in f.sites():
        assert torch.equal(r.q(*k), f.q(*k))


@pytest.mark.gpt2
def test_format_demos_real_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    demos = [("2+2", "4"), ("capital of France", "Paris")]
    ids, mask = format_demos(demos, tok, max_len=64)
    assert ids.shape == mask.shape and ids.shape[0] == 1
    text = tok.decode(ids[0])
    assert "INPUT: 2+2" in text and "OUTPUT: 4" in text

    input_ids, labels, attn = build_demo_lm_batch(demos, tok, device="cpu")
    assert input_ids.shape == labels.shape == attn.shape
    # CE is over OUTPUT tokens only: padding (attn==0) is masked out, and the
    # "INPUT: ... OUTPUT:" prefix carries no labels
    assert (labels[attn == 0] == -100).all()
    for i, (x, _y) in enumerate(demos):
        prefix_len = len(tok(f"INPUT: {x}\nOUTPUT:", add_special_tokens=False)["input_ids"])
        assert (labels[i, :prefix_len] == -100).all()
    assert (labels != -100).any()
