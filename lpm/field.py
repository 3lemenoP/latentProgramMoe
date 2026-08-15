"""lpm/field.py — program fields on the SO(3)-block manifold (spec §2).

A ProgramField is a per-(layer, site-group) collection of RAW (unnormalized)
quaternions; normalization happens in forward (q_normalize inside rotations()).
Field groups per layer (spec §2.1):

    attn_io    (n3(d_model),)            residual-interface sandwich, attention
    qk_rel     (H, n3(d_head))           relative key transport in the logits
    mlp_io     (n3(d_model),)            residual-interface sandwich, MLP
    ffn_hidden (n3(d_ff),)               sandwich around the elementwise activation

All quaternion / gain parameters are fp32 regardless of model dtype (spec §1);
rotation matrices are cast to activation dtype inside apply_rot.

Optional abelian gains (spec §4) are stored as rho with g = exp(rho), one scalar
per 3-block, shapes matching the quaternion site shapes minus the trailing 4.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .quaternion import Partition, hamilton, make_partition, q_normalize, q_to_R

SITE_NAMES = ("attn_io", "qk_rel", "mlp_io", "ffn_hidden")

SiteKey = Tuple[int, str]  # (layer, name)


def _pkey(layer: int, name: str) -> str:
    """ParameterDict key ('.' is illegal in module names)."""
    return f"L{layer}_{name}"


# ---------------------------------------------------------------------------
# FieldSpec: which sites exist, with what shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    tie_qk_across_heads: bool = False
    # rotary axis field (steering doc B0): >0 enables the rope_ax site group
    # of shape (H, n3(rotary_ndims)) — the program conjugates the RoPE
    # generators, Ω → R Ω Rᵀ. 0 = no rotary machinery exposed (GPT-2, toy).
    rotary_ndims: int = 0

    @classmethod
    def from_hf_config(cls, cfg, tie_qk_across_heads: bool = False) -> "FieldSpec":
        """Build from a HF model config. Handles GPT-2, Llama-family and
        GPT-NeoX (Pythia) naming; NeoX gets the rope_ax site group."""
        rotary_ndims = 0
        if hasattr(cfg, "n_embd"):  # GPT-2 family
            d_model = cfg.n_embd
            n_layers = cfg.n_layer
            n_heads = cfg.n_head
            d_ff = cfg.n_inner if getattr(cfg, "n_inner", None) else 4 * d_model
            d_head = d_model // n_heads
        else:  # Llama / NeoX family
            d_model = cfg.hidden_size
            n_layers = cfg.num_hidden_layers
            n_heads = cfg.num_attention_heads
            d_ff = cfg.intermediate_size
            d_head = getattr(cfg, "head_dim", None) or d_model // n_heads
            if getattr(cfg, "model_type", "") == "gpt_neox":
                rp = getattr(cfg, "rope_parameters", None) or {}
                partial = (rp.get("partial_rotary_factor", None) if hasattr(rp, "get") else None)
                if partial is None:
                    partial = getattr(cfg, "rotary_pct", 1.0)
                rotary_ndims = int(d_head * partial)
        return cls(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                   d_head=d_head, d_ff=d_ff, tie_qk_across_heads=tie_qk_across_heads,
                   rotary_ndims=rotary_ndims)

    # -- site groups ---------------------------------------------------------
    def site_names(self) -> Tuple[str, ...]:
        return SITE_NAMES + (("rope_ax",) if self.rotary_ndims > 0 else ())

    # -- partitions per site group ------------------------------------------
    def partition(self, name: str) -> Partition:
        if name in ("attn_io", "mlp_io"):
            return make_partition(self.d_model)
        if name == "ffn_hidden":
            return make_partition(self.d_ff)
        if name == "qk_rel":
            return make_partition(self.d_head)
        if name == "rope_ax":
            if self.rotary_ndims <= 0:
                raise KeyError("rope_ax requires rotary_ndims > 0")
            return make_partition(self.rotary_ndims)
        raise KeyError(name)

    def site_shape(self, name: str) -> Tuple[int, ...]:
        """Quaternion tensor shape at a site, WITHOUT the trailing 4."""
        if name == "qk_rel":
            h = 1 if self.tie_qk_across_heads else self.n_heads
            return (h, self.partition(name).n3)
        if name == "rope_ax":
            # never tied: same R_ax must go on q AND k per head (relativity),
            # but heads stay independent
            return (self.n_heads, self.partition(name).n3)
        return (self.partition(name).n3,)

    def site_keys(self) -> List[SiteKey]:
        return [(l, n) for l in range(self.n_layers) for n in self.site_names()]

    def quats_per_layer(self) -> int:
        return sum(int(torch.tensor(self.site_shape(n)).prod()) for n in self.site_names())

    def layer_site_numel(self, name: str) -> int:
        out = 1
        for s in self.site_shape(name):
            out *= s
        return out


# ---------------------------------------------------------------------------
# ProgramField
# ---------------------------------------------------------------------------
class ProgramField(nn.Module):
    """Raw quaternion field (+ optional gain field rho), trainable or functional.

    trainable=True  -> tensors are nn.Parameters (phase-1 fits, refinement)
    trainable=False -> plain tensors, possibly graph-connected (encoder output,
                       compose()/slerp_field() results)

    .rotations() returns {(layer, name): R} with R fp32, shape (n3, 3, 3) for
    io/hidden sites and (H, n3, 3, 3) for qk_rel (expanded if tied). Results are
    cached when no autograd graph is required; the cache auto-invalidates when
    parameters are updated in place (optimizer.step bumps tensor._version).
    """

    def __init__(self, spec: FieldSpec,
                 quats: Dict[SiteKey, torch.Tensor],
                 rho: Optional[Dict[SiteKey, torch.Tensor]] = None,
                 trainable: bool = False):
        super().__init__()
        self.spec = spec
        expected = set(spec.site_keys())
        if set(quats) != expected:
            missing = expected - set(quats)
            extra = set(quats) - expected
            raise ValueError(f"quats keys mismatch: missing={missing} extra={extra}")
        for (l, n), t in quats.items():
            want = (*spec.site_shape(n), 4)
            if tuple(t.shape) != want:
                raise ValueError(f"site {(l, n)}: shape {tuple(t.shape)} != {want}")
        if rho is not None:
            for (l, n), t in rho.items():
                want = spec.site_shape(n)
                if tuple(t.shape) != want:
                    raise ValueError(f"rho {(l, n)}: shape {tuple(t.shape)} != {want}")

        self.trainable = trainable
        if trainable:
            self.params_q = nn.ParameterDict(
                {_pkey(l, n): nn.Parameter(quats[(l, n)].detach().clone().float())
                 for (l, n) in spec.site_keys()})
            self._q = {(l, n): self.params_q[_pkey(l, n)] for (l, n) in spec.site_keys()}
            if rho is not None:
                self.params_rho = nn.ParameterDict(
                    {_pkey(l, n): nn.Parameter(rho[(l, n)].detach().clone().float())
                     for (l, n) in spec.site_keys()})
                self._rho = {(l, n): self.params_rho[_pkey(l, n)] for (l, n) in spec.site_keys()}
            else:
                self._rho = None
        else:
            self._q = {k: v.float() if v.dtype != torch.float32 else v
                       for k, v in quats.items()}
            self._rho = None if rho is None else {k: v.float() if v.dtype != torch.float32 else v
                                                  for k, v in rho.items()}

        self._rot_cache: Optional[Dict[SiteKey, torch.Tensor]] = None
        self._rot_sig = None
        self._gain_cache: Optional[Dict[SiteKey, torch.Tensor]] = None
        self._gain_sig = None
        # load_state_dict(assign=True) REPLACES the Parameter objects inside
        # the ParameterDicts; rebuild the site-keyed views afterwards or the
        # forward path would keep serving the pre-load tensors.
        self.register_load_state_dict_post_hook(lambda m, _keys: m._resync_views())

    def _resync_views(self) -> None:
        if self.trainable:
            self._q = {(l, n): self.params_q[_pkey(l, n)] for (l, n) in self.spec.site_keys()}
            if self._rho is not None:
                self._rho = {(l, n): self.params_rho[_pkey(l, n)]
                             for (l, n) in self.spec.site_keys()}
        self.invalidate()

    # -- constructors --------------------------------------------------------
    @classmethod
    def identity(cls, spec: FieldSpec, enable_gains: bool = False,
                 trainable: bool = False) -> "ProgramField":
        """All-(1,0,0,0) field (+ rho=0): behaviorally the base model (T4/T10)."""
        quats = {}
        for (l, n) in spec.site_keys():
            t = torch.zeros(*spec.site_shape(n), 4)
            t[..., 0] = 1.0
            quats[(l, n)] = t
        rho = ({(l, n): torch.zeros(spec.site_shape(n)) for (l, n) in spec.site_keys()}
               if enable_gains else None)
        return cls(spec, quats, rho, trainable=trainable)

    @classmethod
    def randn_near_identity(cls, spec: FieldSpec, sigma: float = 1e-3,
                            enable_gains: bool = False, trainable: bool = False,
                            generator: Optional[torch.Generator] = None) -> "ProgramField":
        """Identity + N(0, sigma^2) noise on the vector part (spec §2.2)."""
        quats = {}
        for (l, n) in spec.site_keys():
            t = torch.zeros(*spec.site_shape(n), 4)
            t[..., 0] = 1.0
            t[..., 1:] = sigma * torch.randn(*spec.site_shape(n), 3, generator=generator)
            quats[(l, n)] = t
        rho = ({(l, n): torch.zeros(spec.site_shape(n)) for (l, n) in spec.site_keys()}
               if enable_gains else None)
        return cls(spec, quats, rho, trainable=trainable)

    @classmethod
    def from_tensors(cls, spec: FieldSpec, quats: Dict[SiteKey, torch.Tensor],
                     rho: Optional[Dict[SiteKey, torch.Tensor]] = None) -> "ProgramField":
        """Functional (non-parameter) field, e.g. from the LPN encoder — keeps
        the autograd graph of the provided tensors intact."""
        return cls(spec, quats, rho, trainable=False)

    # -- accessors -----------------------------------------------------------
    @property
    def has_gains(self) -> bool:
        return self._rho is not None

    def q(self, layer: int, name: str) -> torch.Tensor:
        return self._q[(layer, name)]

    def rho(self, layer: int, name: str) -> torch.Tensor:
        assert self._rho is not None, "field has no gains"
        return self._rho[(layer, name)]

    def sites(self) -> Iterator[SiteKey]:
        return iter(self.spec.site_keys())

    def quats(self) -> Dict[SiteKey, torch.Tensor]:
        return dict(self._q)

    def rhos(self) -> Optional[Dict[SiteKey, torch.Tensor]]:
        return None if self._rho is None else dict(self._rho)

    # -- rotation / gain construction with auto-invalidating cache -----------
    def _sig(self, tensors: Dict[SiteKey, torch.Tensor]):
        return tuple((id(t), t._version) for t in tensors.values())

    def _needs_graph(self, tensors: Dict[SiteKey, torch.Tensor]) -> bool:
        return torch.is_grad_enabled() and any(t.requires_grad for t in tensors.values())

    def _build_rotations(self) -> Dict[SiteKey, torch.Tensor]:
        out = {}
        H = self.spec.n_heads
        for (l, n), t in self._q.items():
            R = q_to_R(q_normalize(t.float()))
            if n == "qk_rel" and self.spec.tie_qk_across_heads:
                R = R.expand(H, *R.shape[1:])
            out[(l, n)] = R
        return out

    def _hook_invalidate_rot(self, grad):
        # graph-connected cached rotations must not outlive their graph, which
        # is freed by backward(); rebuild on the next forward
        self._rot_cache = self._rot_sig = None
        return grad

    def rotations(self) -> Dict[SiteKey, torch.Tensor]:
        sig = ("graph" if self._needs_graph(self._q) else "nograd",) + self._sig(self._q)
        if self._rot_cache is not None and self._rot_sig == sig:
            return self._rot_cache
        if self._needs_graph(self._q):
            rots = self._build_rotations()
            for R in rots.values():
                if R.requires_grad:
                    R.register_hook(self._hook_invalidate_rot)
            self._rot_cache, self._rot_sig = rots, sig
        else:
            with torch.no_grad():
                self._rot_cache = self._build_rotations()
            self._rot_sig = sig
        return self._rot_cache

    def _build_gains(self) -> Dict[SiteKey, torch.Tensor]:
        assert self._rho is not None
        out = {}
        H = self.spec.n_heads
        for (l, n), t in self._rho.items():
            g = torch.exp(t.float())
            if n == "qk_rel" and self.spec.tie_qk_across_heads:
                g = g.expand(H, *g.shape[1:])
            out[(l, n)] = g
        return out

    def _hook_invalidate_gain(self, grad):
        self._gain_cache = self._gain_sig = None
        return grad

    def gains(self) -> Optional[Dict[SiteKey, torch.Tensor]]:
        if self._rho is None:
            return None
        sig = ("graph" if self._needs_graph(self._rho) else "nograd",) + self._sig(self._rho)
        if self._gain_cache is not None and self._gain_sig == sig:
            return self._gain_cache
        if self._needs_graph(self._rho):
            gains = self._build_gains()
            for g in gains.values():
                if g.requires_grad:
                    g.register_hook(self._hook_invalidate_gain)
            self._gain_cache, self._gain_sig = gains, sig
        else:
            with torch.no_grad():
                self._gain_cache = self._build_gains()
            self._gain_sig = sig
        return self._gain_cache

    def invalidate(self) -> None:
        self._rot_cache = self._rot_sig = None
        self._gain_cache = self._gain_sig = None

    def _apply(self, fn, recurse=True):
        self.invalidate()
        return super()._apply(fn, recurse)

    def to(self, *args, **kwargs):  # noqa: D102 - device moves only; fp32 policy
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None and dtype != torch.float32:
            raise ValueError("ProgramField tensors are fp32 by policy (spec §1); refusing dtype cast")
        if device is not None:
            if self.trainable:
                super().to(device)  # moves Parameters in place; _q references stay valid
            else:
                self._q = {k: v.to(device) for k, v in self._q.items()}
                if self._rho is not None:
                    self._rho = {k: v.to(device) for k, v in self._rho.items()}
            self.invalidate()
        return self

    # -- transforms ----------------------------------------------------------
    def detached(self, trainable: bool = False) -> "ProgramField":
        """Detached copy (e.g. encoder output -> starting point for refine)."""
        q = {k: v.detach().clone() for k, v in self._q.items()}
        r = None if self._rho is None else {k: v.detach().clone() for k, v in self._rho.items()}
        return ProgramField(self.spec, q, r, trainable=trainable)

    def flipped_sign(self) -> "ProgramField":
        """Global sign flip; MUST be behaviorally identical (T3, spec §1.1)."""
        q = {k: -v for k, v in self._q.items()}
        r = None if self._rho is None else {k: v.clone() for k, v in self._rho.items()}
        return ProgramField(self.spec, q, r, trainable=False)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "format_version": 1,
            "spec": dataclasses.asdict(self.spec),
            "q": {_pkey(l, n): self._q[(l, n)].detach().cpu() for (l, n) in self.spec.site_keys()},
            "rho": (None if self._rho is None else
                    {_pkey(l, n): self._rho[(l, n)].detach().cpu() for (l, n) in self.spec.site_keys()}),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, trainable: bool = False, map_location="cpu") -> "ProgramField":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        spec = FieldSpec(**payload["spec"])
        quats = {(l, n): payload["q"][_pkey(l, n)] for (l, n) in spec.site_keys()}
        rho = (None if payload.get("rho") is None else
               {(l, n): payload["rho"][_pkey(l, n)] for (l, n) in spec.site_keys()})
        return cls(spec, quats, rho, trainable=trainable)

    # -- losses --------------------------------------------------------------
    def chordal_reg_to_identity(self) -> torch.Tensor:
        """Sum over sites of d2_chord(q, identity) = 1 - w_hat^2 (spec §7 phase 1).
        Sign-invariant by construction."""
        total = None
        for t in self._q.values():
            qn = q_normalize(t.float())
            term = (1.0 - qn[..., 0] ** 2).sum()
            total = term if total is None else total + term
        return total


# ---------------------------------------------------------------------------
# Abelian baseline for E3 (spec §10): fixed learned axis per site, theta free.
# ---------------------------------------------------------------------------
class AxisBank(nn.Module):
    """One learnable axis per 3-block site, SHARED across programs. Sharing the
    axes is what makes the baseline abelian: same-axis rotations commute, so
    composition is provably order-blind."""

    def __init__(self, spec: FieldSpec, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.spec = spec
        self.axes = nn.ParameterDict(
            {_pkey(l, n): nn.Parameter(torch.randn(*spec.site_shape(n), 3, generator=generator))
             for (l, n) in spec.site_keys()})

    def axis(self, layer: int, name: str) -> torch.Tensor:
        a = self.axes[_pkey(layer, name)]
        return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class AbelianProgramField(nn.Module):
    """q = (cos θ/2, sin θ/2 · a_site) with a_site from a shared AxisBank.

    Exposes the same rotations()/gains() interface as ProgramField so it can
    drive the sandwiched model directly. No gains (E3 doesn't use them).
    """

    def __init__(self, bank: AxisBank):
        super().__init__()
        self.bank = bank
        self.spec = bank.spec
        self.theta = nn.ParameterDict(
            {_pkey(l, n): nn.Parameter(torch.zeros(self.spec.site_shape(n)))
             for (l, n) in self.spec.site_keys()})
        self.trainable = True
        self.has_gains = False

    def q(self, layer: int, name: str) -> torch.Tensor:
        th = self.theta[_pkey(layer, name)].float()
        a = self.bank.axis(layer, name).float()
        w = torch.cos(th / 2).unsqueeze(-1)
        v = torch.sin(th / 2).unsqueeze(-1) * a
        return torch.cat([w, v], dim=-1)

    def quats(self) -> Dict[SiteKey, torch.Tensor]:
        return {(l, n): self.q(l, n) for (l, n) in self.spec.site_keys()}

    def rotations(self) -> Dict[SiteKey, torch.Tensor]:
        H = self.spec.n_heads
        out = {}
        for (l, n) in self.spec.site_keys():
            R = q_to_R(self.q(l, n))  # already unit norm
            if n == "qk_rel" and self.spec.tie_qk_across_heads:
                R = R.expand(H, *R.shape[1:])
            out[(l, n)] = R
        return out

    def gains(self):
        return None

    def invalidate(self) -> None:
        pass

    def to_program_field(self) -> ProgramField:
        """Snapshot as a generic ProgramField (for compose / save)."""
        return ProgramField(self.spec, {k: v.detach().clone() for k, v in self.quats().items()})
