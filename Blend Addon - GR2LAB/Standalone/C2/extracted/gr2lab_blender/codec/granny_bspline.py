"""Granny-compatible B-spline curve evaluation (ctypes FindKnot + Python de Boor).

Uses the same FindKnot index convention as granny2.dll (validated against
GrannySampleBSpline for Degree=1). Degree>=2 uses de Boor on the stored
knot vector with a local window of (degree+1) controls — matching Granny's
SampleBSpline call pattern: ti/pi start at (FindKnot - Degree).
"""
from __future__ import annotations

import ctypes
import math
import os
from ctypes import POINTER, c_float, c_int32, addressof, sizeof
from pathlib import Path
from typing import List, Optional, Sequence

_DLL: Optional[ctypes.CDLL] = None
_FindKnot = None
_SampleBSpline = None


def _dll() -> ctypes.CDLL:
    global _DLL, _FindKnot, _SampleBSpline
    if _DLL is not None:
        return _DLL
    codec = Path(__file__).resolve().parent
    addon = codec.parent
    candidates = [
        addon / "native" / "granny2.dll",
        addon / "granny2.dll",
        codec / "tools" / "granny2.dll",
    ]
    path = next((c for c in candidates if c.is_file()), candidates[0])
    if not path.is_file():
        cands = list(addon.rglob("granny2.dll"))
        if cands:
            path = cands[0]
    if not path.is_file():
        raise FileNotFoundError("granny2.dll not found for B-spline sampling")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))
    _DLL = ctypes.WinDLL(str(path))
    _FindKnot = _DLL.GrannyFindKnot
    _FindKnot.argtypes = [c_int32, POINTER(c_float), c_float]
    _FindKnot.restype = c_int32
    _SampleBSpline = _DLL.GrannySampleBSpline
    _SampleBSpline.argtypes = [
        c_int32,
        c_int32,
        c_int32,
        POINTER(c_float),
        POINTER(c_float),
        c_float,
        POINTER(c_float),
    ]
    _SampleBSpline.restype = None
    return _DLL


def find_knot(knots: Sequence[float], t: float) -> int:
    """GrannyFindKnot: index of first knot strictly greater than t (or len)."""
    _dll()
    n = len(knots)
    buf = (c_float * n)(*[float(k) for k in knots])
    return int(_FindKnot(n, buf, c_float(float(t))))


def find_knot_py(knots: Sequence[float], t: float) -> int:
    for i, k in enumerate(knots):
        if float(k) > float(t):
            return i
    return len(knots)


def lerp_sample(
    knots: Sequence[float], controls: Sequence[Sequence[float]], t: float
) -> List[float]:
    """Degree-1 / site linear sample between adjacent knots."""
    if not knots or not controls:
        return []
    if len(knots) == 1 or t <= knots[0]:
        return list(controls[0])
    if t >= knots[-1]:
        return list(controls[-1])
    for i in range(len(knots) - 1):
        if knots[i] <= t <= knots[i + 1]:
            span = knots[i + 1] - knots[i]
            u = 0.0 if span <= 1e-12 else (t - knots[i]) / span
            a, b = controls[i], controls[i + 1]
            return [a[j] + (b[j] - a[j]) * u for j in range(len(a))]
    return list(controls[0])


def slerp_sample(
    knots: Sequence[float], controls: Sequence[Sequence[float]], t: float
) -> List[float]:
    from granny_quat import slerp

    if not knots or not controls:
        return [0.0, 0.0, 0.0, 1.0]
    if len(knots) == 1 or t <= knots[0]:
        return list(controls[0])
    if t >= knots[-1]:
        return list(controls[-1])
    for i in range(len(knots) - 1):
        if knots[i] <= t <= knots[i + 1]:
            span = knots[i + 1] - knots[i]
            u = 0.0 if span <= 1e-12 else (t - knots[i]) / span
            return slerp(controls[i], controls[i + 1], u)
    return list(controls[0])


def _ensure_hemisphere(controls: Sequence[Sequence[float]]) -> List[List[float]]:
    """Flip consecutive quats to the same hemisphere (dot >= 0)."""
    out: List[List[float]] = [list(controls[0])]
    for q in controls[1:]:
        prev = out[-1]
        dot = sum(prev[i] * q[i] for i in range(4))
        if dot < 0:
            out.append([-q[0], -q[1], -q[2], -q[3]])
        else:
            out.append(list(q))
    return out


def _pad_knot_vector(knots: Sequence[float], degree: int) -> List[float]:
    """Build a knot vector of length n+degree+1 so n controls form a degree-d B-spline.

    Padding scheme B (empirically best vs Granny degree-2 BG3 curves):
    one copy of t0, then all stored knots, then `degree` copies of t_end.
    Yields large corner-energy reduction vs linear while differing mid-span.
    """
    if not knots:
        return []
    t0 = float(knots[0])
    t1 = float(knots[-1])
    return [t0] + [float(k) for k in knots] + [t1] * int(degree)


def _find_span(u_vec: Sequence[float], degree: int, n_ctrl: int, t: float) -> int:
    """Span index i with U[i] <= t < U[i+1], clamped to [degree, n_ctrl-1]."""
    # Special: t at end
    if t >= u_vec[n_ctrl]:
        return n_ctrl - 1
    if t <= u_vec[degree]:
        return degree
    # binary search
    low = degree
    high = n_ctrl
    mid = (low + high) // 2
    while t < u_vec[mid] or t >= u_vec[mid + 1]:
        if t < u_vec[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _normalize_quat(q: Sequence[float]) -> List[float]:
    nrm = math.sqrt(sum(float(x) * float(x) for x in q)) or 1.0
    return [float(x) / nrm for x in q]


def _numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None


def _eval_bspline_at(
    degree: int,
    n: int,
    dim: int,
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    u_vec: Sequence[float],
    t: float,
    *,
    normalize: bool = False,
) -> List[float]:
    """De Boor at t using a pre-padded knot vector. Same math as sample_bspline."""
    t0 = float(knots[0])
    t1 = float(knots[-1])
    if float(t) <= t0:
        result = list(controls[0])
        if normalize and dim == 4:
            result = _normalize_quat(result)
        return result
    if float(t) >= t1:
        result = list(controls[-1])
        if normalize and dim == 4:
            result = _normalize_quat(result)
        return result
    tt = float(t)
    i = _find_span(u_vec, degree, n, tt)
    d = [list(controls[i - degree + j]) for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            left = u_vec[i - degree + j]
            right = u_vec[i + 1 + j - r]
            denom = right - left
            alpha = 0.0 if abs(denom) < 1e-20 else (tt - left) / denom
            d[j] = [
                (1.0 - alpha) * d[j - 1][c] + alpha * d[j][c] for c in range(dim)
            ]
    result = d[degree]
    if normalize and dim == 4:
        result = _normalize_quat(result)
    return result


def _sample_bspline_grid_deg2_numpy(
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    u_vec: Sequence[float],
    grid: Sequence[float],
    *,
    normalize: bool,
):
    """Vectorized Degree-2 de Boor. Same span + lerp as _eval_bspline_at."""
    np = _numpy()
    if np is None:
        return None
    n = len(knots)
    dim = len(controls[0])
    U = np.asarray(u_vec, dtype=np.float64)
    C = np.asarray(controls, dtype=np.float64)
    T = np.asarray(grid, dtype=np.float64)
    t0 = float(knots[0])
    t1 = float(knots[-1])
    out = np.empty((T.shape[0], dim), dtype=np.float64)
    m_lo = T <= t0
    m_hi = T >= t1
    m_mid = ~(m_lo | m_hi)
    out[m_lo] = C[0]
    out[m_hi] = C[-1]
    if np.any(m_mid):
        tt = T[m_mid]
        # Match _find_span: U[i] <= t < U[i+1], clamped to [2, n-1].
        i = np.searchsorted(U, tt, side="right") - 1
        i = np.clip(i, 2, n - 1).astype(np.intp)
        c0 = C[i - 2]
        c1 = C[i - 1]
        c2 = C[i]
        left = U[i]
        right = U[i + 2]
        denom = right - left
        a = np.divide(tt - left, denom, out=np.zeros_like(tt), where=np.abs(denom) >= 1e-20)
        d2 = (1.0 - a)[:, None] * c1 + a[:, None] * c2
        left = U[i - 1]
        right = U[i + 1]
        denom = right - left
        a = np.divide(tt - left, denom, out=np.zeros_like(tt), where=np.abs(denom) >= 1e-20)
        d1 = (1.0 - a)[:, None] * c0 + a[:, None] * c1
        left = U[i]
        right = U[i + 1]
        denom = right - left
        a = np.divide(tt - left, denom, out=np.zeros_like(tt), where=np.abs(denom) >= 1e-20)
        out[m_mid] = (1.0 - a)[:, None] * d1 + a[:, None] * d2
    if normalize and dim == 4:
        nrm = np.sqrt(np.sum(out * out, axis=1, keepdims=True))
        nrm = np.where(nrm == 0.0, 1.0, nrm)
        out = out / nrm
    return out.tolist()


def sample_bspline(
    degree: int,
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    t: float,
    *,
    normalize: bool = False,
) -> List[float]:
    """Evaluate Granny-style B-spline of given degree at parameter t.

    Knots and controls share length n (BG3 compressed / DaK32f layout).
    Degree 1 == linear through controls. Degree 2+ uses de Boor on a padded
    knot vector of length n+degree+1.
    """
    n = len(knots)
    if n == 0 or not controls:
        return []
    dim = len(controls[0])
    degree = int(degree)
    if degree <= 1 or n < degree + 1:
        return lerp_sample(knots, controls, t)
    u_vec = _pad_knot_vector(knots, degree)
    return _eval_bspline_at(
        degree, n, dim, knots, controls, u_vec, t, normalize=normalize
    )


def sample_bspline_grid(
    degree: int,
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    grid: Sequence[float],
    *,
    normalize: bool = False,
) -> List[List[float]]:
    """sample_bspline at every grid time, padding the knot vector once."""
    n = len(knots)
    if n == 0 or not controls:
        return [[] for _ in grid]
    dim = len(controls[0])
    degree = int(degree)
    if degree <= 1 or n < degree + 1:
        return [lerp_sample(knots, controls, t) for t in grid]
    u_vec = _pad_knot_vector(knots, degree)
    if degree == 2:
        batched = _sample_bspline_grid_deg2_numpy(
            knots, controls, u_vec, grid, normalize=normalize
        )
        if batched is not None:
            return batched
    return [
        _eval_bspline_at(
            degree, n, dim, knots, controls, u_vec, t, normalize=normalize
        )
        for t in grid
    ]


def sample_bspline_quat(
    degree: int,
    knots: Sequence[float],
    quats: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    """Component-wise B-spline on hemisphere-aligned quats, then normalize."""
    aligned = _ensure_hemisphere(quats)
    return sample_bspline(degree, knots, aligned, t, normalize=True)


def sample_bspline_quat_grid(
    degree: int,
    knots: Sequence[float],
    quats: Sequence[Sequence[float]],
    grid: Sequence[float],
) -> List[List[float]]:
    """sample_bspline_quat on a time grid — align + pad once per channel."""
    aligned = _ensure_hemisphere(quats)
    return sample_bspline_grid(degree, knots, aligned, grid, normalize=True)


def granny_sample_bspline_raw(
    degree: int,
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    t: float,
    *,
    normalize: bool = False,
) -> List[float]:
    """Call granny2.dll GrannySampleBSpline (Degree=1 validated; Degree=2 best-effort)."""
    _dll()
    n = len(knots)
    dim = len(controls[0])
    flat = [float(v) for c in controls for v in c]
    kbuf = (c_float * n)(*[float(k) for k in knots])
    cbuf = (c_float * (n * dim))(*flat)
    i = int(_FindKnot(n, kbuf, c_float(float(t))))
    start = i - int(degree)
    if start < 0 or start + degree >= n:
        return sample_bspline(degree, knots, controls, t, normalize=normalize)
    ti = ctypes.cast(addressof(kbuf) + start * sizeof(c_float), POINTER(c_float))
    pi = ctypes.cast(
        addressof(cbuf) + start * dim * sizeof(c_float), POINTER(c_float)
    )
    out = (c_float * dim)()
    _SampleBSpline(
        int(degree), dim, 1 if normalize else 0, ti, pi, c_float(float(t)), out
    )
    result = [float(out[j]) for j in range(dim)]
    if not all(math.isfinite(x) for x in result):
        return sample_bspline(degree, knots, controls, t, normalize=normalize)
    if normalize and dim == 4:
        nrm = math.sqrt(sum(x * x for x in result)) or 1.0
        result = [x / nrm for x in result]
    return result
