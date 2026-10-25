"""Decode/encode Granny D4nK16uC15u orientation controls (LSLib-compatible)."""
from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple

# From LSLib D4nK16uC15u.cs
_SCALE_TABLE = [
    1.4142135,
    0.70710677,
    0.35355338,
    0.35355338,
    0.35355338,
    0.17677669,
    0.17677669,
    0.17677669,
    -1.4142135,
    -0.70710677,
    -0.35355338,
    -0.35355338,
    -0.35355338,
    -0.17677669,
    -0.17677669,
    -0.17677669,
]
_OFFSET_TABLE = [
    -0.70710677,
    -0.35355338,
    -0.53033006,
    -0.17677669,
    0.17677669,
    -0.17677669,
    -0.088388346,
    0.0,
    0.70710677,
    0.35355338,
    0.53033006,
    0.17677669,
    -0.17677669,
    0.17677669,
    0.088388346,
    -0.0,
]
_U15_SCALE = 0.000030518509
_U7_SCALE = 0.0078740157  # 1/127 — D4nK8uC7u


def scale_offset_tables(selector: int) -> Tuple[List[float], List[float]]:
    scales = [
        _SCALE_TABLE[(selector >> 0) & 0x0F] * _U15_SCALE,
        _SCALE_TABLE[(selector >> 4) & 0x0F] * _U15_SCALE,
        _SCALE_TABLE[(selector >> 8) & 0x0F] * _U15_SCALE,
        _SCALE_TABLE[(selector >> 12) & 0x0F] * _U15_SCALE,
    ]
    offsets = [
        _OFFSET_TABLE[(selector >> 0) & 0x0F],
        _OFFSET_TABLE[(selector >> 4) & 0x0F],
        _OFFSET_TABLE[(selector >> 8) & 0x0F],
        _OFFSET_TABLE[(selector >> 12) & 0x0F],
    ]
    return scales, offsets


def scale_offset_tables_u7(selector: int) -> Tuple[List[float], List[float]]:
    """Scale/offset tables for D4nK8uC7u (7-bit controls)."""
    scales = [
        _SCALE_TABLE[(selector >> 0) & 0x0F] * _U7_SCALE,
        _SCALE_TABLE[(selector >> 4) & 0x0F] * _U7_SCALE,
        _SCALE_TABLE[(selector >> 8) & 0x0F] * _U7_SCALE,
        _SCALE_TABLE[(selector >> 12) & 0x0F] * _U7_SCALE,
    ]
    offsets = [
        _OFFSET_TABLE[(selector >> 0) & 0x0F],
        _OFFSET_TABLE[(selector >> 4) & 0x0F],
        _OFFSET_TABLE[(selector >> 8) & 0x0F],
        _OFFSET_TABLE[(selector >> 12) & 0x0F],
    ]
    return scales, offsets


def quat_from_controls(
    a: int, b: int, c: int, scales: Sequence[float], offsets: Sequence[float]
) -> List[float]:
    swizzle1 = ((b & 0x8000) >> 14) | (c >> 15)
    swizzle2 = (swizzle1 + 1) & 3
    swizzle3 = (swizzle2 + 1) & 3
    swizzle4 = (swizzle3 + 1) & 3

    data_a = (a & 0x7FFF) * scales[swizzle2] + offsets[swizzle2]
    data_b = (b & 0x7FFF) * scales[swizzle3] + offsets[swizzle3]
    data_c = (c & 0x7FFF) * scales[swizzle4] + offsets[swizzle4]
    sum_sq = data_a * data_a + data_b * data_b + data_c * data_c
    data_d = math.sqrt(max(0.0, 1.0 - sum_sq))
    if a & 0x8000:
        data_d = -data_d

    f = [0.0, 0.0, 0.0, 0.0]
    f[swizzle2] = data_a
    f[swizzle3] = data_b
    f[swizzle4] = data_c
    f[swizzle1] = data_d
    return f  # x,y,z,w


def quat_from_controls_u7(
    a: int, b: int, c: int, scales: Sequence[float], offsets: Sequence[float]
) -> List[float]:
    """D4nK8uC7u — same layout as u15 but 7 data bits + high flag bits on bytes."""
    swizzle1 = ((b & 0x80) >> 6) | ((c & 0x80) >> 7)
    swizzle2 = (swizzle1 + 1) & 3
    swizzle3 = (swizzle2 + 1) & 3
    swizzle4 = (swizzle3 + 1) & 3

    data_a = (a & 0x7F) * scales[swizzle2] + offsets[swizzle2]
    data_b = (b & 0x7F) * scales[swizzle3] + offsets[swizzle3]
    data_c = (c & 0x7F) * scales[swizzle4] + offsets[swizzle4]
    sum_sq = data_a * data_a + data_b * data_b + data_c * data_c
    data_d = math.sqrt(max(0.0, 1.0 - sum_sq))
    if a & 0x80:
        data_d = -data_d

    f = [0.0, 0.0, 0.0, 0.0]
    f[swizzle2] = data_a
    f[swizzle3] = data_b
    f[swizzle4] = data_c
    f[swizzle1] = data_d
    return f


def _clamp_u15(v: float) -> int:
    return max(0, min(0x7FFF, int(round(v))))


def _qdot(a: Sequence[float], b: Sequence[float]) -> float:
    return (
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )


def _qdot_abs(a: Sequence[float], b: Sequence[float]) -> float:
    return abs(_qdot(a, b))


def _norm_quat4(q: Sequence[float]) -> List[float]:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return [x / n, y / n, z / n, w / n]


def _encode_quat_swizzle(
    q: Sequence[float],
    scales: Sequence[float],
    offsets: Sequence[float],
    swizzle1: int,
) -> Tuple[int, int, int, bool]:
    """Encode with fixed reconstructed component. Returns (a,b,c,clamped).

    Keeps the input hemisphere: if q[swizzle1] < 0, set a&0x8000 (LSLib recon
    sign bit) instead of negating the whole quaternion (which caused component
    spikes across consecutive keys).
    """
    qq = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    sign_bit = 0x8000 if qq[swizzle1] < 0 else 0
    swizzle2 = (swizzle1 + 1) & 3
    swizzle3 = (swizzle2 + 1) & 3
    swizzle4 = (swizzle3 + 1) & 3
    clamped = False

    def enc_comp(comp: int, sw: int) -> int:
        nonlocal clamped
        s = scales[sw]
        if abs(s) < 1e-20:
            return 0
        raw = (qq[comp] - offsets[sw]) / s
        if raw < -0.5 or raw > 0x7FFF + 0.5:
            clamped = True
        return _clamp_u15(raw)

    a = enc_comp(swizzle2, swizzle2) | sign_bit
    b = enc_comp(swizzle3, swizzle3) | ((swizzle1 & 2) << 14)
    c = enc_comp(swizzle4, swizzle4) | ((swizzle1 & 1) << 15)
    return a, b, c, clamped


def controls_from_quat(
    quat: Sequence[float],
    scales: Sequence[float],
    offsets: Sequence[float],
    *,
    prefer_abc: Tuple[int, int, int] | None = None,
) -> Tuple[int, int, int]:
    """Pack quaternion into 3×u16 controls.

    If prefer_abc is set, keep the same swizzle/sign bits (stable edit of an existing key).
    Otherwise try all swizzles and keep the best signed round-trip vs the input.
    """
    q = _norm_quat4(quat)

    if prefer_abc is not None:
        a0, b0, c0 = prefer_abc
        swizzle1 = ((b0 & 0x8000) >> 14) | (c0 >> 15)
        swizzle2 = (swizzle1 + 1) & 3
        swizzle3 = (swizzle2 + 1) & 3
        swizzle4 = (swizzle3 + 1) & 3
        # Ensure reconstructed component sign matches a0 bit
        if (q[swizzle1] < 0) != bool(a0 & 0x8000):
            q = [-c for c in q]

        def enc(comp: int, sw: int, hi_bit: int) -> int:
            s = scales[sw]
            if abs(s) < 1e-20:
                return hi_bit
            raw = (q[comp] - offsets[sw]) / s
            return _clamp_u15(raw) | hi_bit

        a = enc(swizzle2, swizzle2, a0 & 0x8000)
        b = enc(swizzle3, swizzle3, b0 & 0x8000)
        c = enc(swizzle4, swizzle4, c0 & 0x8000)
        return a, b, c

    best: Tuple[float, int, int, int] | None = None
    for swizzle1 in range(4):
        a, b, c, _clamped = _encode_quat_swizzle(q, scales, offsets, swizzle1)
        back = quat_from_controls(a, b, c, scales, offsets)
        # Signed dot — prefer encodings that land in the input hemisphere
        score = _qdot(q, back)
        if best is None or score > best[0]:
            best = (score, a, b, c)
    assert best is not None
    return best[1], best[2], best[3]


def controls_from_quat_series(
    quats: Sequence[Sequence[float]],
    scales: Sequence[float],
    offsets: Sequence[float],
) -> List[Tuple[int, int, int]]:
    """Pack a quaternion key series with hemisphere continuity.

    make_compatible against the previous *decoded* sample so stored knots do
    not flip q/-q (component spikes in the curve editor) after quat_k16 pack.
    """
    out: List[Tuple[int, int, int]] = []
    prev: List[float] | None = None
    for raw in quats:
        target = _norm_quat4(raw)
        if prev is not None and _qdot(prev, target) < 0.0:
            target = [-c for c in target]

        abc = controls_from_quat(target, scales, offsets)
        decoded = quat_from_controls(abc[0], abc[1], abc[2], scales, offsets)
        if _qdot(decoded, target) < 0.0:
            target = [-c for c in target]
            abc = controls_from_quat(target, scales, offsets)
            decoded = quat_from_controls(abc[0], abc[1], abc[2], scales, offsets)

        # Quantization may still land opposite prev — force hemi match once more
        if prev is not None and _qdot(prev, decoded) < 0.0:
            target = [-c for c in target]
            abc = controls_from_quat(target, scales, offsets)
            decoded = quat_from_controls(abc[0], abc[1], abc[2], scales, offsets)

        out.append(abc)
        prev = _norm_quat4(decoded)
    return out


def quat_encode_quality(
    quat: Sequence[float],
    scales: Sequence[float],
    offsets: Sequence[float],
) -> Tuple[float, bool]:
    """Return (|dot| after round-trip, clamped) for best swizzle under this selector."""
    q = _norm_quat4(quat)
    best_dot = -1.0
    any_clamp = False
    for swizzle1 in range(4):
        a, b, c, clamped = _encode_quat_swizzle(q, scales, offsets, swizzle1)
        back = quat_from_controls(a, b, c, scales, offsets)
        d = _qdot_abs(q, back)
        if d > best_dot:
            best_dot = d
            any_clamp = clamped
    return best_dot, any_clamp


def find_selector_for_quats(
    quats: Sequence[Sequence[float]],
    *,
    min_dot: float = 0.995,
) -> int:
    """Pick ScaleOffsetTableEntries (u16) that can represent all quats well.

    Prefers the current-style compact selectors when possible; falls back to a
    wide-range selector (large |scale| nibbles) when edits would otherwise clamp.
    """
    # Candidate selectors: wide (large scales) + a few mid-range mixes.
    # Nibble 0/8 = ±1.414 scale (widest); 1/9 = ±0.707; 2/10 = ±0.353.
    candidates = [
        0x0000,  # all widest positive-table
        0x1111,
        0x2222,
        0x8888,  # widest negative-table
        0x0123,
        0x4567,
        0x89AB,
        0xCDEF,
        0x0888,
        0x8088,
        0x8808,
        0x8880,
        0x0111,
        0x1011,
        0x1101,
        0x1110,
    ]
    best_sel = 0x0000
    best_score = -1.0
    for sel in candidates:
        scales, offsets = scale_offset_tables(sel)
        worst = 1.0
        for q in quats:
            d, _ = quat_encode_quality(q, scales, offsets)
            if d < worst:
                worst = d
        if worst >= min_dot:
            return sel
        if worst > best_score:
            best_score = worst
            best_sel = sel
    return best_sel


def slerp(a: Sequence[float], b: Sequence[float], u: float) -> List[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
        dot = -dot
    if dot > 0.9995:
        x = ax + (bx - ax) * u
        y = ay + (by - ay) * u
        z = az + (bz - az) * u
        w = aw + (bw - aw) * u
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(theta)
        wa = math.sin((1 - u) * theta) / s
        wb = math.sin(u * theta) / s
        x = ax * wa + bx * wb
        y = ay * wa + by * wb
        z = az * wa + bz * wb
        w = aw * wa + bw * wb
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return [x / n, y / n, z / n, w / n]


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Hamilton product a⊗b (apply b then a if used as rotations)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def axis_angle_quat(axis: Sequence[float], angle: float) -> List[float]:
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    s = math.sin(angle * 0.5)
    return [ax * s, ay * s, az * s, math.cos(angle * 0.5)]


def read_u16_list(buf: bytes, off: int, count: int) -> List[int]:
    return list(struct.unpack_from(f"<{count}H", buf, off))
