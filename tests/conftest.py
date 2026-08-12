"""Shared fixtures. Tiny randomly-initialized models exercise identical code
paths offline/fast; tests marked `gpt2` additionally run the spec's acceptance
criteria on the real pretrained GPT-2 small checkpoint (needs network/HF cache).
"""
import pytest
import torch

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from lpm import LatentProgramModel


@pytest.fixture(scope="session")
def tiny_gpt2():
    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=257, n_positions=64, n_embd=48, n_layer=2, n_head=4,
        n_inner=192, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=256, eos_token_id=256, pad_token_id=256)
    cfg._attn_implementation = "eager"
    base = GPT2LMHeadModel(cfg).float().eval()
    return LatentProgramModel(base)


@pytest.fixture(scope="session")
def tiny_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=257, hidden_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,  # GQA on purpose
        intermediate_size=120, max_position_embeddings=64,
        attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    base = LlamaForCausalLM(cfg).float().eval()
    return LatentProgramModel(base)


@pytest.fixture(scope="session")
def tiny_ids():
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, 257, (2, 24), generator=g)


@pytest.fixture(scope="session")
def real_gpt2():
    try:
        return LatentProgramModel.from_pretrained("gpt2")
    except Exception as e:  # no network / no cache
        pytest.skip(f"real gpt2 checkpoint unavailable: {e}")


@pytest.fixture(scope="session")
def real_gpt2_ids():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")
    text = ("The quick brown fox jumps over the lazy dog. "
            "Meanwhile, researchers at the observatory recorded an unusual "
            "signal that repeated every ninety minutes.")
    return tok(text, return_tensors="pt")["input_ids"]
