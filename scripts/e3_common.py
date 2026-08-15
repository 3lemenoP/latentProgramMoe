"""Shared E3 / E3′ helpers: tiny from-scratch GPT-2, field fits, decode metrics."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from lpm import LatentProgramModel, ProgramField
from lpm.tasks import E3Vocab, e3_examples, e3_output
from lpm.utils import cosine_lr, set_lr


def make_base(vocab: E3Vocab, device) -> GPT2LMHeadModel:
    cfg = GPT2Config(vocab_size=vocab.size, n_positions=64, n_embd=96,
                     n_layer=4, n_head=4, n_inner=384,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                     bos_token_id=vocab.BOS, eos_token_id=vocab.EOS,
                     pad_token_id=vocab.PAD)
    cfg._attn_implementation = "eager"
    return GPT2LMHeadModel(cfg).to(device)


def save_base(model: GPT2LMHeadModel, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.to_dict(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
               path)


def load_base(path: str, device) -> GPT2LMHeadModel:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPT2Config(**blob["config"])
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def train_base(model, vocab, device, steps=3000, batch=64, lr=3e-4, seed=0,
               behaviors=("copy", "prepend", "reverse")):
    ids_all, lab_all, attn_all = [], [], []
    for i, beh in enumerate(behaviors):
        ids, lab, attn = e3_examples(beh, 4000, vocab, seed=seed + i)
        ids_all.append(ids)
        lab_all.append(lab)
        attn_all.append(attn)
    T = max(x.shape[1] for x in ids_all)

    def padT(x, fill):
        return torch.nn.functional.pad(x, (0, T - x.shape[1]), value=fill)

    ids = torch.cat([padT(x, vocab.PAD) for x in ids_all])
    lab = torch.cat([padT(x, -100) for x in lab_all])
    attn = torch.cat([padT(x, 0) for x in attn_all])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, ids.shape[0], (batch,), generator=g)
        out = model(input_ids=ids[idx].to(device),
                    attention_mask=attn[idx].to(device),
                    labels=lab[idx].to(device))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=100))
        opt.step()
        if step % 200 == 0:
            print(f"[base] step {step}/{steps} loss {out.loss.item():.4f}")
    model.eval()


def fit_field(model: LatentProgramModel, field, behavior, vocab, device,
              steps=1500, batch=64, lr=1e-3, seed=1, extra_params=(), tag="",
              overlap_with=None, overlap_coef=0.0):
    ids, lab, attn = e3_examples(behavior, 4000, vocab, seed=seed)
    params = list(dict.fromkeys(list(field.parameters()) + list(extra_params)))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    from lpm.quaternion import q_angle2, q_normalize
    for step in range(steps):
        idx = torch.randint(0, ids.shape[0], (batch,), generator=g)
        with model.program(field):
            out = model(input_ids=ids[idx].to(device),
                        attention_mask=attn[idx].to(device),
                        labels=lab[idx].to(device))
        loss = out.loss
        if overlap_with is not None and overlap_coef != 0.0:
            ov = 0.0
            for k in field.spec.site_keys():
                sa = q_angle2(q_normalize(field.q(*k)))
                sb = q_angle2(q_normalize(overlap_with.q(*k)))
                ov = ov + (sa * sb).sum()
            loss = loss - overlap_coef * ov
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=50))
        opt.step()
        if step % 200 == 0:
            print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f}")
    field.invalidate()
    return field


def fit_fields_joint(model: LatentProgramModel, pairs, vocab, device,
                     steps=1500, batch=64, lr=1e-3, seed=1, extra_params=(),
                     tag="", overlap_coef=0.0, lambda_reg=0.0):
    data = []
    for i, (field, behavior) in enumerate(pairs):
        ids, lab, attn = e3_examples(behavior, 4000, vocab, seed=seed + i)
        data.append((field, ids, lab, attn))
    params = [p for field, *_ in data for p in field.parameters()]
    params = list(dict.fromkeys(params + list(extra_params)))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    from lpm.quaternion import d2_chord, q_angle2, q_normalize
    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for field, ids, lab, attn in data:
            idx = torch.randint(0, ids.shape[0], (batch,), generator=g)
            with model.program(field):
                out = model(input_ids=ids[idx].to(device),
                            attention_mask=attn[idx].to(device),
                            labels=lab[idx].to(device))
            loss = out.loss
            if lambda_reg:
                ident = ProgramField.identity(field.spec)
                for k in field.spec.site_keys():
                    loss = loss + lambda_reg * d2_chord(
                        q_normalize(field.q(*k)), q_normalize(ident.q(*k))).mean()
            loss.backward()
            total += loss.item()
        if overlap_coef and len(data) >= 2:
            ov = 0.0
            fa, fb = data[0][0], data[1][0]
            for k in fa.spec.site_keys():
                ov = ov + (q_angle2(q_normalize(fa.q(*k))) *
                           q_angle2(q_normalize(fb.q(*k)))).sum()
            extra = -overlap_coef * ov
            extra.backward()
            total += extra.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=50))
        opt.step()
        if step % 200 == 0:
            print(f"[{tag}] step {step}/{steps} mean loss {total / len(data):.4f}")
    for field, *_ in data:
        field.invalidate()
    return [field for field, *_ in data]


@torch.no_grad()
def ordered_accuracy(model: LatentProgramModel, field, behavior, vocab, device,
                     n=200, seed=99):
    import random
    rng = random.Random(seed)
    exact, tok_hits, tok_total = 0, 0, 0
    ctx = model.program(field) if field is not None else model.program(None)
    with ctx:
        for _ in range(n):
            x = vocab.payload(rng)
            want = e3_output(x, behavior, vocab) + [vocab.EOS]
            ids = torch.tensor([[vocab.BOS] + x + [vocab.SEP]], device=device)
            for _step in range(len(want) + 2):
                logits = model(input_ids=ids).logits[0, -1]
                nxt = int(logits.argmax())
                ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt == vocab.EOS:
                    break
            got = ids[0, len(x) + 2:].tolist()
            if got == want:
                exact += 1
            L = min(len(got), len(want))
            tok_hits += sum(int(a == b) for a, b in zip(got[:L], want[:L]))
            tok_total += len(want)
    return exact / n, tok_hits / max(tok_total, 1)
