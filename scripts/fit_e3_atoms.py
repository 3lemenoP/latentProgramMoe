#!/usr/bin/env python3
"""Fit and save E3 atom fields (base + z_a prepend + z_b reverse) for D1.

Skips the abelian baseline and the composition table. Faster than full E3.

    python scripts/fit_e3_atoms.py --save-dir runs/e3
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, LPMConfig, ProgramField  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import (  # noqa: E402
    fit_field, make_base, ordered_accuracy, save_base, train_base,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e3_order.yaml")
    ap.add_argument("--save-dir", default="runs/e3")
    ap.add_argument("--base-steps", type=int, default=3000)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--reuse-base", default=None,
                    help="path to an existing base.pt — skip base training so "
                         "strong-field refits (handoff P1) stay on the SAME "
                         "frozen base as the weak fields")
    ap.add_argument("--init-sigma", type=float, default=None,
                    help="override cfg.field_init_sigma for the field init")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config) if Path(args.config).exists() else LPMConfig()
    set_seed(args.seed)
    device = get_device(args.device)
    vocab = E3Vocab()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_base:
        print(f"=== base: reusing {args.reuse_base} ===")
        from e3_common import load_base
        base = load_base(args.reuse_base, device)
        save_base(base, str(save_dir / "base.pt"))
    else:
        print("=== base (copy/prepend/reverse) ===")
        base = make_base(vocab, device)
        train_base(base, vocab, device, steps=args.base_steps, seed=args.seed,
                   behaviors=("copy", "prepend", "reverse", "append"))
        save_base(base, str(save_dir / "base.pt"))
    model = LatentProgramModel(base)

    sigma = args.init_sigma if args.init_sigma is not None else cfg.field_init_sigma
    print(f"=== z_a prepend / z_b reverse (sigma={sigma}, lr={args.lr}) ===")
    za = ProgramField.randn_near_identity(model.spec, sigma=sigma,
                                          trainable=True).to(device)
    za = fit_field(model, za, "prepend", vocab, device, steps=args.fit_steps,
                   lr=args.lr, seed=args.seed + 1, tag="z_a")
    zb = ProgramField.randn_near_identity(model.spec, sigma=sigma,
                                          trainable=True).to(device)
    zb = fit_field(model, zb, "reverse", vocab, device, steps=args.fit_steps,
                   lr=args.lr, seed=args.seed + 2, tag="z_b")
    za.save(str(save_dir / "z_a.pt"))
    zb.save(str(save_dir / "z_b.pt"))

    ea = ordered_accuracy(model, za, "prepend", vocab, device)
    eb = ordered_accuracy(model, zb, "reverse", vocab, device)
    print(f"z_a prepend exact={ea[0]:.3f}  z_b reverse exact={eb[0]:.3f}")
    print(f"saved {save_dir}/base.pt z_a.pt z_b.pt")


if __name__ == "__main__":
    main()
