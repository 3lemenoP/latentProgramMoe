"""lpm/config.py — experiment configuration (spec §11).

Dataclass mirrored in yaml (configs/*.yaml). Field names match the spec's
config block verbatim so yaml keys round-trip 1:1.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import yaml


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0          # spec §7: grad-clip 1.0
    steps: int = 2000               # 1-3k typical for phase 1
    warmup_steps: int = 0
    schedule: str = "cosine"        # "cosine" | "constant"
    batch_size: int = 8


@dataclass
class LPMConfig:
    base_model: str = "gpt2"
    enable_gains: bool = False      # decided by E0 (scripts/audit_mirsky.py); §4
    tie_qk_across_heads: bool = False
    field_init_sigma: float = 1e-3  # §2.2
    lambda_h: float = 0.1           # §7 phase 1: hidden-state MSE weight
    lambda_reg: float = 1e-4        # §7 phase 1: chordal pull toward identity
    lambda_geo: float = 1.0         # §7 phase 2: regression to phase-1 targets
    lambda_task: float = 1.0        # §7 phase 2: end-to-end task CE
    refine_steps: int = 0           # §5.3: default off for eval parity
    refine_lr: float = 1e-3
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    data: Dict[str, Any] = field(default_factory=dict)  # per-experiment data paths

    # ---- yaml round-trip -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["optimizer"]["betas"] = list(d["optimizer"]["betas"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LPMConfig":
        d = dict(d)
        opt = d.pop("optimizer", {}) or {}
        if "betas" in opt:
            opt["betas"] = tuple(opt["betas"])
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(optimizer=OptimConfig(**opt), **d)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "LPMConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})
