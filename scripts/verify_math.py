"""Numerical certification of latent-program-moe-spec.md kernel formulas (numpy twin)."""
import numpy as np

rng = np.random.default_rng(0)
results = []


def check(name, err, tol):
    ok = bool(err < tol)
    results.append((name, err, tol, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:<46} err={err:.3e}  tol={tol:.0e}")


def q_normalize(q, eps=1e-8):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)


def q_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def hamilton(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - np.sum(av * bv, -1, keepdims=True)
    v = aw * bv + bw * av + np.cross(av, bv)
    return np.concatenate([w, v], -1)


def d2_chord(a, b):
    return 1.0 - np.sum(a * b, -1) ** 2


def slerp(a, b, t, eps=1e-6):
    dot = np.sum(a * b, -1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), None, 1 - 1e-7)
    omega = np.arccos(dot)
    so = np.sin(omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / so
    lerp = q_normalize((1 - t) * a + t * b)
    return np.where(omega < eps, lerp, out)


def blockdiag(R, rem):
    n3 = R.shape[0]
    d = 3 * n3 + rem
    B = np.eye(d)
    for i in range(n3):
        B[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R[i]
    return B


def apply_rot(x, R, n3, inverse=False):
    cut = 3 * n3
    xb = x[..., :cut].reshape(*x.shape[:-1], n3, 3)
    Rm = np.swapaxes(R, -1, -2) if inverse else R
    yb = np.einsum('...ni,nji->...nj', xb, Rm)
    return np.concatenate([yb.reshape(*x.shape[:-1], cut), x[..., cut:]], -1)


# T1: orthogonality and determinant
q = q_normalize(rng.normal(size=(1000, 4)))
R = q_to_R(q)
check("T1 R^T R = I", np.abs(np.einsum('nij,nik->njk', R, R) - np.eye(3)).max(), 1e-6)
check("T1 det R = +1", np.abs(np.linalg.det(R) - 1).max(), 1e-6)

# T2: homomorphism R(a x b) = R(a) R(b)
a = q_normalize(rng.normal(size=(1000, 4)))
b = q_normalize(rng.normal(size=(1000, 4)))
check("T2 R(a*b) = R(a) R(b)", np.abs(q_to_R(hamilton(a, b)) - q_to_R(a) @ q_to_R(b)).max(), 1e-6)

# T3: sign invariance
check("T3 R(-q) = R(q)", np.abs(q_to_R(-q) - q_to_R(q)).max(), 1e-12)

# Composition convention: 'a then b' as vector action = R(hamilton(b, a))
v = rng.normal(size=(1000, 3))
seq = np.einsum('nij,nj->ni', q_to_R(b), np.einsum('nij,nj->ni', q_to_R(a), v))
one = np.einsum('nij,nj->ni', q_to_R(hamilton(b, a)), v)
check("conv R_b R_a = R(b*a)  (a first, then b)", np.abs(seq - one).max(), 1e-6)

# T8: slerp endpoints, unit norm, antipodal stability
s0, s1 = slerp(a, b, 0.0), slerp(a, b, 1.0)
check("T8 slerp t=0 -> a", np.abs(s0 - a).max(), 1e-6)
check("T8 slerp t=1 -> +/-b", np.minimum(np.abs(s1 - b).max(-1), np.abs(s1 + b).max(-1)).max(), 1e-6)
smid = slerp(a, b, 0.37)
check("T8 slerp unit norm", np.abs(np.linalg.norm(smid, axis=-1) - 1).max(), 1e-6)
b_anti = q_normalize(-a + 1e-9 * rng.normal(size=a.shape))
s_anti = slerp(a, b_anti, 0.5)
check("T8 antipodal: finite + unit", float(~np.isfinite(s_anti).all()) + np.abs(np.linalg.norm(s_anti, axis=-1) - 1).max(), 1e-5)

# T9: non-commutativity is generic
check("T9 hamilton order matters (want err >= tol)", 0.01 / max(d2_chord(hamilton(b, a), hamilton(a, b)).mean(), 1e-12), 1.0)

# apply_rot: matches block-diag matmul, preserves norm/remainder, inverse round-trips
n3, rem = 3, 1
d = 3 * n3 + rem
qf = q_normalize(rng.normal(size=(n3, 4)))
Rf = q_to_R(qf)
x = rng.normal(size=(7, 5, d))
y = apply_rot(x, Rf, n3)
B = blockdiag(Rf, rem)
check("apply_rot = blockdiag matmul", np.abs(y - x @ B.T).max(), 1e-9)
check("apply_rot preserves norm", np.abs(np.linalg.norm(y, axis=-1) - np.linalg.norm(x, axis=-1)).max(), 1e-9)
check("apply_rot remainder untouched", np.abs(y[..., -rem:] - x[..., -rem:]).max(), 1e-15)
check("apply_rot inverse round-trip", np.abs(apply_rot(y, Rf, n3, inverse=True) - x).max(), 1e-9)

# T6 math: conjugation preserves singular values
W = rng.normal(size=(d, d))
sW = np.linalg.svd(B @ W @ B.T, compute_uv=False)
check("T6 spectrum preserved under conjugation", np.abs(np.sort(sW) - np.sort(np.linalg.svd(W, compute_uv=False))).max(), 1e-9)

# Automorphism: (B A B^T)(B C B^T) = B (A C) B^T
A2 = rng.normal(size=(d, d)); C2 = rng.normal(size=(d, d))
check("automorphism (BAB')(BCB') = B(AC)B'", np.abs((B @ A2 @ B.T) @ (B @ C2 @ B.T) - B @ (A2 @ C2) @ B.T).max(), 1e-9)

# Dead value frame: C^T (sum_j a_ij C v_j) = sum_j a_ij v_j  (linear mixing cancellation)
attn = rng.random(size=(5, 8, 8)); attn /= attn.sum(-1, keepdims=True)
vv = rng.normal(size=(5, 8, d))
mixed_plain = np.einsum('bij,bjd->bid', attn, vv)
mixed_conj = apply_rot(np.einsum('bij,bjd->bid', attn, apply_rot(vv, Rf, n3)), Rf, n3, inverse=True)
check("dead frame: per-seq value conj cancels", np.abs(mixed_conj - mixed_plain).max(), 1e-9)

print()
fails = [r for r in results if not r[3]]
print(f"{len(results) - len(fails)}/{len(results)} checks passed" + ("" if not fails else "  <-- FAILURES PRESENT"))
raise SystemExit(1 if fails else 0)
