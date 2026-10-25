"""GR2 Lab bridge file (.gr2lab) — one-file handoff for Blender.

Pack a decoded dump into a single JSON document (extension .gr2lab).
Diff an edited bridge file against the dump and apply via existing key ops.
"""

from __future__ import annotations

import json
import math
import sys
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMAT_ID = "gr2lab"
FORMAT_VERSION = 1

# Download-for-Blender default: densify Degree≥2 Granny curves to Degree-1 samples.
DEFAULT_DENSIFY_FPS = 30.0

POS_EPS = 1e-5
SCL_EPS = 1e-5
QUAT_DOT_EPS = 0.99999  # |dot| above this ⇒ same rotation (incl. q vs -q)
TIME_EPS = 1e-4
# Looser thresholds for Blender matrix round-trip / densified-but-same-motion
BLENDER_POS_EPS = 1e-4
BLENDER_QUAT_DOT_EPS = 0.9999
# Even looser for true-noop oracle (Spine*_M bake extras / decompose drift)
NOOP_POS_EPS = 2e-4
NOOP_QUAT_DOT_EPS = 0.999
# Mass "significant" extras = Blender bake densify (wrong mid-keys), not a few user keys.
# Importing them interleaved with dump knots → scrub-on-frames OK, play shakes.
# When extras are on-curve but *massively* denser (frame bake), keep the denser grid.
MAX_SIGNIFICANT_EXTRAS = 8
SIGNIFICANT_EXTRAS_FRAC = 0.05
# Import densify samples k16 Dummy_Root as Deg2 B-spline (~2 cm vs game eval).
# Real NLA/constraint root edits are larger (this clip: Root_M ~8.5 cm).
ROOT_POS_DENSIFY_EPS = 0.025  # metres


def _count_far_extra_times(
    dump_times: Sequence[float],
    bridge_times: Sequence[float],
    time_eps: float = TIME_EPS,
) -> int:
    """How many bridge times do not snap onto the dump knot grid."""
    if not bridge_times:
        return 0
    if not dump_times:
        return len(bridge_times)
    n = 0
    for t in bridge_times:
        _i, d = _nearest_time_index(float(t), dump_times)
        if d > time_eps:
            n += 1
    return n


def _mass_extras_cap(dump_times: Sequence[float]) -> int:
    return max(
        MAX_SIGNIFICANT_EXTRAS,
        int(SIGNIFICANT_EXTRAS_FRAC * max(len(dump_times), 1) + 0.5),
    )


def _is_mass_grid_densify(
    dump_times: Sequence[float],
    bridge_times: Sequence[float],
) -> bool:
    """True for intentional frame-bake densify (many times off the dump grid).

    Distinguishes from a few on-curve Spine*_M bake artifacts (still noop) and
    near-duplicate frame-noise densify (snap within TIME_EPS).

    Also requires a clear knot-count jump so re-import after a bake densify
    (599 vs 600 with tick/float snap drift) does not keep rewriting.
    """
    if not dump_times or not bridge_times:
        return False
    n_dump = len(dump_times)
    n_bridge = len(bridge_times)
    cap = _mass_extras_cap(dump_times)
    # Need clearly denser sampling — not ~same count with time-grid drift.
    min_bridge = max(n_dump + cap, int(n_dump * 1.25) + 1)
    if n_bridge < min_bridge:
        return False
    far = _count_far_extra_times(dump_times, bridge_times)
    return far > cap


def _v3_close(a: Sequence[float], b: Sequence[float], eps: float = POS_EPS) -> bool:
    return all(abs(float(a[i]) - float(b[i])) <= eps for i in range(3))


def _quat_close(a: Sequence[float], b: Sequence[float], eps: float = QUAT_DOT_EPS) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    dot = abs(
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )
    return dot >= eps


# Blender Z-up Dummy_Root rest ≈ Rx(±90°). Must not be applied as Granny anim.
_RX90_QUAT = (0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
_RX_NEG90_QUAT = (-0.7071067811865475, 0.0, 0.0, 0.7071067811865475)


def _is_blender_zup_root_ori(q: Sequence[float]) -> bool:
    if not q or len(q) < 4:
        return False
    return _quat_close(q, _RX90_QUAT, 0.995) or _quat_close(q, _RX_NEG90_QUAT, 0.995)


def _dummy_root_ori_is_false_zup(ori_vals: Sequence[Sequence[float]]) -> bool:
    """True when bridge Dummy_Root ori is only Blender Z-up rest (would tip +90°)."""
    if not ori_vals:
        return False
    return all(_is_blender_zup_root_ori(v) for v in ori_vals)


def _times_close(a: Sequence[float], b: Sequence[float], eps: float = TIME_EPS) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(a[i]) - float(b[i])) <= eps for i in range(len(a)))


def _pos_series_close(
    times_a: Sequence[float],
    vals_a: Sequence[Sequence[float]],
    times_b: Sequence[float],
    vals_b: Sequence[Sequence[float]],
) -> bool:
    if not _times_close(times_a, times_b) or len(vals_a) != len(vals_b):
        return False
    return all(_v3_close(vals_a[i], vals_b[i]) for i in range(len(vals_a)))


def _quat_series_close(
    times_a: Sequence[float],
    vals_a: Sequence[Sequence[float]],
    times_b: Sequence[float],
    vals_b: Sequence[Sequence[float]],
) -> bool:
    if not _times_close(times_a, times_b) or len(vals_a) != len(vals_b):
        return False
    return all(_quat_close(vals_a[i], vals_b[i]) for i in range(len(vals_a)))


def _snap_time_to_grid(t: float, grid: Sequence[float], eps: float = TIME_EPS) -> float:
    best = None
    nearest = float(t)
    for g in grid:
        d = abs(float(t) - float(g))
        if best is None or d < best:
            best = d
            nearest = float(g)
    if best is not None and best <= eps:
        return nearest
    return float(t)


def _lerp_vec(
    times: Sequence[float],
    vals: Sequence[Sequence[float]],
    t: float,
    *,
    n: int,
) -> List[float]:
    if not times or not vals:
        return [0.0] * n
    if t <= times[0]:
        return [float(vals[0][i]) if i < len(vals[0]) else 0.0 for i in range(n)]
    if t >= times[-1]:
        v = vals[-1]
        return [float(v[i]) if i < len(v) else 0.0 for i in range(n)]
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            u = 0.0 if span <= 1e-12 else (t - times[i]) / span
            a, b = vals[i], vals[i + 1]
            return [
                float(a[j]) + (float(b[j]) - float(a[j])) * u
                if j < len(a) and j < len(b)
                else 0.0
                for j in range(n)
            ]
    v = vals[-1]
    return [float(v[i]) if i < len(v) else 0.0 for i in range(n)]


def _slerp_quat(
    times: Sequence[float],
    vals: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    if not times or not vals:
        return [0.0, 0.0, 0.0, 1.0]
    if t <= times[0]:
        return _norm_quat(vals[0])
    if t >= times[-1]:
        return _norm_quat(vals[-1])
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            u = 0.0 if span <= 1e-12 else (t - times[i]) / span
            a = _norm_quat(vals[i])
            b = _norm_quat(vals[i + 1])
            # Prefer short path
            dot = sum(a[j] * b[j] for j in range(4))
            if dot < 0.0:
                b = [-x for x in b]
                dot = -dot
            if dot > 0.9995:
                return _norm_quat([a[j] + (b[j] - a[j]) * u for j in range(4)])
            omega = math.acos(max(-1.0, min(1.0, dot)))
            so = math.sin(omega)
            s0 = math.sin((1.0 - u) * omega) / so
            s1 = math.sin(u * omega) / so
            return _norm_quat([a[j] * s0 + b[j] * s1 for j in range(4)])
    return _norm_quat(vals[-1])


def _nearest_time_index(
    t: float, grid: Sequence[float]
) -> Tuple[int, float]:
    best_i = 0
    best_d = abs(float(t) - float(grid[0])) if grid else 0.0
    for i, g in enumerate(grid):
        d = abs(float(t) - float(g))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def _densified_grid_close(
    times_a: Sequence[float],
    vals_a: Sequence[Sequence[float]],
    times_b: Sequence[float],
    vals_b: Sequence[Sequence[float]],
    *,
    time_eps: float,
    value_close,
) -> bool:
    """B is A densified with near-duplicate times (Blender frame-noise union).

    Every B sample snaps onto A's grid with a matching value, and every A knot
    is covered by at least one B sample.
    """
    if (
        not times_a
        or not times_b
        or len(times_a) != len(vals_a)
        or len(times_b) != len(vals_b)
    ):
        return False
    covered: set[int] = set()
    for t, v in zip(times_b, vals_b):
        i, d = _nearest_time_index(float(t), times_a)
        if d > time_eps or not value_close(vals_a[i], v):
            return False
        covered.add(i)
    return len(covered) == len(times_a)


def _pos_series_motion_close(
    times_a: Sequence[float],
    vals_a: Sequence[Sequence[float]],
    times_b: Sequence[float],
    vals_b: Sequence[Sequence[float]],
    *,
    pos_eps: float = BLENDER_POS_EPS,
    time_eps: float = TIME_EPS,
) -> bool:
    """True when curves describe the same motion (allow densify / time snap drift)."""
    if not vals_a or not vals_b:
        return False
    if _pos_series_close(times_a, vals_a, times_b, vals_b):
        return True
    # Same count after snapping B times onto A grid
    if len(times_a) == len(times_b) and len(vals_a) == len(vals_b):
        snapped = [_snap_time_to_grid(t, times_a, time_eps) for t in times_b]
        if _times_close(times_a, snapped, time_eps) and all(
            _v3_close(vals_a[i], vals_b[i], pos_eps) for i in range(len(vals_a))
        ):
            return True
    # Near-duplicate densify onto A's knots (held values at frame-noise times)
    if _densified_grid_close(
        times_a,
        vals_a,
        times_b,
        vals_b,
        time_eps=time_eps,
        value_close=lambda a, b: _v3_close(a, b, pos_eps),
    ):
        return True
    # Resample both ways (true same-motion, possibly different knot counts)
    for t, v in zip(times_b, vals_b):
        if not _v3_close(_lerp_vec(times_a, vals_a, float(t), n=3), v, pos_eps):
            return False
    for t, v in zip(times_a, vals_a):
        if not _v3_close(_lerp_vec(times_b, vals_b, float(t), n=3), v, pos_eps):
            return False
    return True


def _quat_series_motion_close(
    times_a: Sequence[float],
    vals_a: Sequence[Sequence[float]],
    times_b: Sequence[float],
    vals_b: Sequence[Sequence[float]],
    *,
    quat_eps: float = BLENDER_QUAT_DOT_EPS,
    time_eps: float = TIME_EPS,
) -> bool:
    """True when orientation curves match under snap / resample (Blender noop)."""
    if not vals_a or not vals_b:
        return False
    if _quat_series_close(times_a, vals_a, times_b, vals_b):
        return True
    if len(times_a) == len(times_b) and len(vals_a) == len(vals_b):
        snapped = [_snap_time_to_grid(t, times_a, time_eps) for t in times_b]
        if _times_close(times_a, snapped, time_eps) and all(
            _quat_close(vals_a[i], vals_b[i], quat_eps) for i in range(len(vals_a))
        ):
            return True
    if _densified_grid_close(
        times_a,
        vals_a,
        times_b,
        vals_b,
        time_eps=time_eps,
        value_close=lambda a, b: _quat_close(a, b, quat_eps),
    ):
        return True
    for t, v in zip(times_b, vals_b):
        if not _quat_close(_slerp_quat(times_a, vals_a, float(t)), v, quat_eps):
            return False
    for t, v in zip(times_a, vals_a):
        if not _quat_close(_slerp_quat(times_b, vals_b, float(t)), v, quat_eps):
            return False
    return True


def _bridge_has_extra_keys(
    dump_times: Sequence[float],
    bridge_times: Sequence[float],
    time_eps: float = TIME_EPS,
) -> bool:
    """True if any bridge time is farther than time_eps from every dump knot.

    Near-duplicate frame-noise times that snap onto the dump grid do not count.
    """
    if not bridge_times:
        return False
    if not dump_times:
        return True
    for t in bridge_times:
        _i, d = _nearest_time_index(float(t), dump_times)
        if d > time_eps:
            return True
    return False


def _resample_pos_onto_times(
    bridge_times: Sequence[float],
    bridge_vals: Sequence[Sequence[float]],
    target_times: Sequence[float],
) -> List[List[float]]:
    return [
        _lerp_vec(bridge_times, bridge_vals, float(t), n=3) for t in target_times
    ]


def _resample_quat_onto_times(
    bridge_times: Sequence[float],
    bridge_vals: Sequence[Sequence[float]],
    target_times: Sequence[float],
) -> List[List[float]]:
    return [_slerp_quat(bridge_times, bridge_vals, float(t)) for t in target_times]


def _bridge_has_significant_extra_keys(
    dump_times: Sequence[float],
    dump_vals: Sequence[Sequence[float]],
    bridge_times: Sequence[float],
    bridge_vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
    time_eps: float = TIME_EPS,
    pos_eps: float = NOOP_POS_EPS,
    quat_eps: float = NOOP_QUAT_DOT_EPS,
) -> bool:
    """True when bridge has new times whose values differ from dump motion.

    Near-duplicate densify (snap within time_eps) does not count.
    Far extras that still lie on the dump curve (Blender bake artifacts) do not
    count. Real user keyframes with a different pose do.
    """
    if not dump_times or not bridge_times or len(bridge_times) != len(bridge_vals):
        return False
    value_close = (
        (lambda a, b: _quat_close(a, b, quat_eps))
        if is_quat
        else (lambda a, b: _v3_close(a, b, pos_eps))
    )
    sample_dump = (
        (lambda t: _slerp_quat(dump_times, dump_vals, float(t)))
        if is_quat
        else (lambda t: _lerp_vec(dump_times, dump_vals, float(t), n=3))
    )
    for t, v in zip(bridge_times, bridge_vals):
        _i, d = _nearest_time_index(float(t), dump_times)
        if d <= time_eps:
            continue
        if not value_close(sample_dump(t), v):
            return True
    return False


def _dump_knots_motion_close(
    dump_times: Sequence[float],
    dump_vals: Sequence[Sequence[float]],
    bridge_times: Sequence[float],
    bridge_vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
    pos_eps: float = NOOP_POS_EPS,
    quat_eps: float = NOOP_QUAT_DOT_EPS,
) -> bool:
    """True when bridge samples match dump at every dump knot time (noop oracle).

    Blender-only extra times are ignored for this check.
    """
    if not dump_times or not dump_vals or not bridge_times or not bridge_vals:
        return False
    if len(dump_times) != len(dump_vals) or len(bridge_times) != len(bridge_vals):
        return False
    value_close = (
        (lambda a, b: _quat_close(a, b, quat_eps))
        if is_quat
        else (lambda a, b: _v3_close(a, b, pos_eps))
    )
    sample_bridge = (
        (lambda t: _slerp_quat(bridge_times, bridge_vals, float(t)))
        if is_quat
        else (lambda t: _lerp_vec(bridge_times, bridge_vals, float(t), n=3))
    )
    for t, v in zip(dump_times, dump_vals):
        if not value_close(v, sample_bridge(t)):
            return False
    return True


def _resolve_grid_remake_write(
    dump_times: Sequence[float],
    dump_vals: Sequence[Sequence[float]],
    bridge_times: Sequence[float],
    bridge_vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
    force_blender_export: bool = False,
) -> Optional[Tuple[List[float], List[List[float]]]]:
    """Decide keyed write like the site viewport (set existing + add new keys).

    Does NOT bake mid-key influence into neighbor dump knots — that would turn
    two add_keys into hundreds of set_keys. Overlay only bridge samples that
    snap onto the dump grid; append a few significant extras as new knots.

    Mass densify / retarget: keep Blender's key list (any density). replace_*
    writes sample series as float Degree=1 (not compressed quat_k16), so Granny
    mid-frame eval matches site/Blender scrub with arbitrary key counts.
    """
    motion_close = _quat_series_motion_close if is_quat else _pos_series_motion_close
    copy_val = (lambda v: _norm_quat(v)) if is_quat else (lambda v: list(map(float, v)))

    def bridge_series() -> Tuple[List[float], List[List[float]]]:
        out_t = [float(t) for t in bridge_times]
        out_v = [copy_val(v) for v in bridge_vals]
        order = sorted(range(len(out_t)), key=lambda i: out_t[i])
        out_t = [out_t[i] for i in order]
        out_v = [out_v[i] for i in order]
        # Collapse float-noise near-dups (template∪frame union). Keep last value
        # in each TIME_EPS cluster so conflicting samples cannot micro-slerp flicker.
        out_t, out_v = _collapse_series_near_dups(out_t, out_v, is_quat=is_quat)
        if is_quat and len(out_v) > 1:
            _make_quat_series_compatible(out_v)
        return out_t, out_v

    if force_blender_export and bridge_times and bridge_vals:
        # Blender GR2 LAB export: always write the bridge (dense bake / shortened clip).
        if len(bridge_times) > 1:
            return bridge_series()
        if len(bridge_times) < max(1, int(len(dump_times) * 0.85)):
            return bridge_series()

    if motion_close(dump_times, dump_vals, bridge_times, bridge_vals):
        # Same motion densify: keep denser Blender grid (float Degree=1 on encode).
        if _is_mass_grid_densify(dump_times, bridge_times):
            return bridge_series()
        return None
    # True-noop oracle: dump knots agree and extras are on-curve (looser eps)
    if _dump_knots_motion_close(
        dump_times, dump_vals, bridge_times, bridge_vals, is_quat=is_quat
    ) and not _bridge_has_significant_extra_keys(
        dump_times, dump_vals, bridge_times, bridge_vals, is_quat=is_quat
    ):
        if _is_mass_grid_densify(dump_times, bridge_times):
            return bridge_series()
        return None

    if not dump_times:
        times = [float(t) for t in bridge_times] if bridge_times else [0.0]
        vals = [list(map(float, v)) for v in bridge_vals]
        return times, vals

    value_close = (
        (lambda a, b: _quat_close(a, b, NOOP_QUAT_DOT_EPS))
        if is_quat
        else (lambda a, b: _v3_close(a, b, NOOP_POS_EPS))
    )

    out_t = [float(t) for t in dump_times]
    out_v = [copy_val(v) for v in dump_vals]

    # Overlay bridge keys that snap onto dump knots (viewport set_key).
    # Keep dump value when within noop eps (matrix decompose drift).
    for t, v in zip(bridge_times, bridge_vals):
        i, d = _nearest_time_index(float(t), dump_times)
        if d <= TIME_EPS:
            nv = copy_val(v)
            if not value_close(out_v[i], nv):
                out_v[i] = nv

    # Append only extras with a pose that differs from dump motion (noop eps).
    # Mass extras = densified bake / different animation from Blender.
    sample_dump = (
        (lambda t: _slerp_quat(dump_times, dump_vals, float(t)))
        if is_quat
        else (lambda t: _lerp_vec(dump_times, dump_vals, float(t), n=3))
    )
    extras_t: List[float] = []
    extras_v: List[List[float]] = []
    for t, v in zip(bridge_times, bridge_vals):
        _i, d = _nearest_time_index(float(t), dump_times)
        if d <= TIME_EPS:
            continue
        if value_close(sample_dump(t), v):
            continue
        extras_t.append(float(t))
        extras_v.append(copy_val(v))

    max_extras = _mass_extras_cap(dump_times)
    if len(extras_t) > max_extras:
        # Full bake / retarget: use Blender's key list as the new curve.
        return bridge_series()

    # On-curve mass densify (same motion, many far times) — keep denser grid.
    if _is_mass_grid_densify(dump_times, bridge_times):
        return bridge_series()

    out_t.extend(extras_t)
    out_v.extend(extras_v)

    order = sorted(range(len(out_t)), key=lambda i: out_t[i])
    out_t = [out_t[i] for i in order]
    out_v = [out_v[i] for i in order]
    out_t, out_v = _collapse_series_near_dups(out_t, out_v, is_quat=is_quat)

    if is_quat and len(out_v) > 1:
        _make_quat_series_compatible(out_v)

    if motion_close(dump_times, dump_vals, out_t, out_v):
        return None
    if _dump_knots_motion_close(
        dump_times, dump_vals, out_t, out_v, is_quat=is_quat
    ) and len(out_t) == len(dump_times):
        return None
    return out_t, out_v


def _collapse_series_near_dups(
    times: Sequence[float],
    vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
    eps: float = TIME_EPS,
) -> Tuple[List[float], List[List[float]]]:
    """Merge knots within eps; keep last sample (frame-grid wins when sorted).

    Blender export historically unioned template JSON floats with ``(frame-1)/fps``
    via exact set(), producing ~1107 keys with near-zero-Δt conflicting quats —
    scrub-on-frames OK, play shakes.
    """
    copy_val = (lambda v: _norm_quat(v)) if is_quat else (lambda v: list(map(float, v)))
    if not times:
        return [0.0], [copy_val(vals[0]) if vals else ([0.0, 0.0, 0.0, 1.0] if is_quat else [0.0, 0.0, 0.0])]
    items = sorted(
        ((float(t), copy_val(v)) for t, v in zip(times, vals)),
        key=lambda iv: iv[0],
    )
    out_t = [items[0][0]]
    out_v = [items[0][1]]
    for t, v in items[1:]:
        if abs(t - out_t[-1]) <= eps:
            out_t[-1] = t
            out_v[-1] = v
        else:
            out_t.append(t)
            out_v.append(v)
    return out_t, out_v


def _make_quat_series_compatible(values: List[List[float]]) -> None:
    """In-place Granny xyzw series → short-arc consecutive keys."""
    prev: Optional[List[float]] = None
    for i, q in enumerate(values):
        cur = _norm_quat(q)
        if prev is not None and (
            prev[0] * cur[0] + prev[1] * cur[1] + prev[2] * cur[2] + prev[3] * cur[3]
        ) < 0.0:
            cur = [-cur[0], -cur[1], -cur[2], -cur[3]]
        prev = cur
        values[i] = cur


def _norm_quat(q: Sequence[float]) -> List[float]:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def _read_source_name(dump_dir: Path) -> str:
    src = dump_dir / "SOURCE.txt"
    if not src.is_file():
        return ""
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.lower().startswith("name="):
            return line.split("=", 1)[1].strip()
    return ""


def _channel_pack(
    edit_mode: str,
    editable: bool,
    fmt: int,
    capacity: int,
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "edit_mode": edit_mode,
        "editable": bool(editable),
        "format": int(fmt) if fmt is not None else -1,
        "capacity": int(capacity),
        "times": times if isinstance(times, list) else [float(t) for t in times],
        "values": values if isinstance(values, list) else [list(map(float, v)) for v in values],
    }
    if note:
        out["note"] = note
    if extra:
        out.update(extra)
    return out


def _uniform_time_grid(duration: float, fps: float) -> List[float]:
    dur = max(0.0, float(duration))
    rate = max(1.0, float(fps))
    n = max(2, int(dur * rate) + 1)
    if dur <= 1e-12:
        return [0.0]
    return [i * dur / (n - 1) for i in range(n)]


def densify_doc_bspline(
    doc: Dict[str, Any],
    anim: Any,
    *,
    fps: float = DEFAULT_DENSIFY_FPS,
    timer: Any = None,
) -> Dict[str, Any]:
    """Rewrite Degree≥2 multi-key channels as Degree-1 samples of the B-spline.

    Vanilla BG3 curves are Degree-2 *controls*. Blender/site lerp through those
    controls looks snappy. Evaluating the B-spline onto a uniform grid gives the
    smooth game path as normal editable keys (NLA / bake friendly).
    """
    from granny_anim import CURVE_DEGREE_LINEAR, FMT_DA_K32F_C32F
    from granny_bspline import sample_bspline_grid, sample_bspline_quat_grid

    duration = float(doc.get("duration") or getattr(anim, "duration", 0.0) or 0.0)
    grid = _uniform_time_grid(duration, fps)
    by_name = {t.name: t for t in (getattr(anim, "tracks", None) or []) if t.name}
    densified = 0
    fps_i = int(round(float(fps)))
    t_pos = 0.0
    t_quat = 0.0
    n_pos = 0
    n_quat = 0
    n_tracks = len(doc.get("tracks") or [])
    if timer is not None:
        getattr(timer, "live_print", print)(
            f"densify {n_tracks} bones × {len(grid)} keys @ {fps_i}fps…"
        )

    for slot in doc.get("tracks") or []:
        name = slot.get("name") or ""
        tr = by_name.get(name)
        if not tr:
            continue

        pos = slot.get("position") or {}
        pt = [float(x) for x in (pos.get("times") or [])]
        pv = [list(map(float, v)) for v in (pos.get("values") or [])]
        deg_p = int(getattr(tr.position, "degree", 0) or pos.get("degree") or 0)
        mode_p = pos.get("edit_mode") or ""
        # Save/24 behavior: Deg≥2 float AND k16 → B-spline sample → Deg1 float grid.
        # (Lerp-only k16 display densify looked snappy in Blender.) Apply skips
        # writing these back for Dummy_Root/Root_M via densify notes / display_densified.
        if (
            deg_p >= 2
            and len(pt) >= 2
            and len(pv) == len(pt)
            and mode_p in ("float_controls", "k16_offsets")
        ):
            t0 = time.perf_counter()
            samples = sample_bspline_grid(deg_p, pt, pv, grid)
            t_pos += time.perf_counter() - t0
            n_pos += 1
            extra = {
                "degree": CURVE_DEGREE_LINEAR,
                "source_interp": "bspline_dense",
            }
            if name in ("Dummy_Root", "Root_M") or mode_p == "k16_offsets":
                extra["display_densified"] = True
            slot["position"] = _channel_pack(
                "float_controls",
                True,
                FMT_DA_K32F_C32F,
                len(grid),
                grid,
                samples,
                note=f"bspline_densified@{fps_i}",
                extra=extra,
            )
            densified += 1
            time.sleep(0)
        elif "degree" not in pos and deg_p:
            pos["degree"] = int(deg_p)
            slot["position"] = pos

        ori = slot.get("orientation") or {}
        ot = [float(x) for x in (ori.get("times") or [])]
        oq = [list(map(float, v)) for v in (ori.get("values") or [])]
        deg_o = int(getattr(tr.orientation, "degree", 0) or ori.get("degree") or 0)
        mode_o = ori.get("edit_mode") or ""
        # Save/24: all Deg≥2 quat_k16 / float → B-spline Deg1 float (incl. Root_M).
        if deg_o >= 2 and len(ot) >= 2 and len(oq) == len(ot) and mode_o in (
            "quat_k16",
            "float_controls",
        ):
            t0 = time.perf_counter()
            samples = sample_bspline_quat_grid(deg_o, ot, oq, grid)
            t_quat += time.perf_counter() - t0
            n_quat += 1
            extra = {
                "degree": CURVE_DEGREE_LINEAR,
                "source_interp": "bspline_dense",
            }
            if name in ("Dummy_Root", "Root_M"):
                extra["display_densified"] = True
            slot["orientation"] = _channel_pack(
                "float_controls",
                True,
                FMT_DA_K32F_C32F,
                len(grid),
                grid,
                samples,
                note=f"bspline_densified@{fps_i}",
                extra=extra,
            )
            densified += 1
            time.sleep(0)
        elif "degree" not in ori and deg_o:
            ori["degree"] = int(deg_o)
            slot["orientation"] = ori

    doc["source_interp"] = "bspline_dense"
    doc["densify_fps"] = float(fps)
    doc["densified_channels"] = densified
    if timer is not None:
        timer.add("densify_pos", t_pos)
        timer.add("densify_quat", t_quat)
        getattr(timer, "live_print", print)(
            f"densify_pos {t_pos:.2f}s ({n_pos} ch)  "
            f"densify_quat {t_quat:.2f}s ({n_quat} ch)"
        )
    return doc


def pack_dump(
    dump_dir: Path,
    *,
    run_id: Optional[str] = None,
    densify_bspline: bool = False,
    densify_fps: float = DEFAULT_DENSIFY_FPS,
) -> Dict[str, Any]:
    """Build a .gr2lab.json document from an OUT_FILES dump folder.

    densify_bspline: when True (Download for Blender), Degree≥2 channels are
    evaluated to Degree-1 sample grids so Blender NLA/bake sees game-smooth motion.
    """
    from granny_anim import build_skeleton, load_dump_gr2, sample_curve

    dump_dir = Path(dump_dir)
    if not dump_dir.is_dir():
        raise FileNotFoundError(f"Dump not found: {dump_dir}")
    rid = run_id or dump_dir.name
    _gr2, anim, s0 = load_dump_gr2(dump_dir)
    skel = build_skeleton(anim, s0)
    source_gr2 = _read_source_name(dump_dir) or Path(anim.source or "").name

    tracks: List[Dict[str, Any]] = []
    for t in anim.tracks:
        if not t.name:
            continue
        curve = sample_curve(s0, anim, t.name)
        ori = curve.get("orientation") or {}
        scl = curve.get("scale") or {}
        pos_extra: Dict[str, Any] = {}
        if curve.get("note"):
            pos_extra["note"] = curve["note"]
        if curve.get("degree") is not None:
            pos_extra["degree"] = int(curve.get("degree") or 0)
        ori_extra: Dict[str, Any] = {}
        if ori.get("one_over_knot_scale") is not None:
            ori_extra["one_over_knot_scale"] = ori["one_over_knot_scale"]
        if ori.get("tick_dt") is not None:
            ori_extra["tick_dt"] = ori["tick_dt"]
        if ori.get("note"):
            ori_extra["note"] = ori["note"]
        if ori.get("degree") is not None:
            ori_extra["degree"] = int(ori.get("degree") or 0)
        scl_extra: Dict[str, Any] = {}
        if scl.get("degree") is not None:
            scl_extra["degree"] = int(scl.get("degree") or 0)

        tracks.append(
            {
                "name": t.name,
                "index": t.index,
                "position": _channel_pack(
                    curve.get("edit_mode") or "none",
                    bool(curve.get("editable")),
                    int(curve.get("format", -1)),
                    int(curve.get("capacity") or curve.get("knot_count") or 0),
                    curve.get("times") or [],
                    curve.get("positions") or [],
                    extra=pos_extra or None,
                ),
                "orientation": _channel_pack(
                    ori.get("edit_mode") or "none",
                    bool(ori.get("editable")),
                    int(ori.get("format", -1)),
                    int(ori.get("capacity") or ori.get("knot_count") or 0),
                    ori.get("times") or [],
                    ori.get("rotations") or [],
                    extra=ori_extra or None,
                ),
                "scale": _channel_pack(
                    scl.get("edit_mode") or "none",
                    bool(scl.get("editable")),
                    int(scl.get("format", -1)),
                    int(scl.get("capacity") or scl.get("knot_count") or 0),
                    scl.get("times") or [],
                    scl.get("scales") or [],
                    note=scl.get("note") or "",
                    extra=scl_extra or None,
                ),
            }
        )

    bones_out = []
    for b in skel.get("bones") or []:
        bones_out.append(
            {
                "index": b.get("index"),
                "name": b.get("name"),
                "parent": b.get("parent"),
                "parent_name": b.get("parent_name"),
                "rest_translation": list(b.get("rest_translation") or [0, 0, 0]),
                "rest_rotation": list(b.get("rest_rotation") or [0, 0, 0, 1]),
                "rest_scale": list(b.get("rest_scale") or [1, 1, 1]),
                "edit_mode": b.get("edit_mode"),
                "editable": b.get("editable"),
                "orient_edit_mode": b.get("orient_edit_mode"),
                "orient_editable": b.get("orient_editable"),
                "scale_edit_mode": b.get("scale_edit_mode"),
                "scale_editable": b.get("scale_editable"),
            }
        )

    doc: Dict[str, Any] = {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "run_id": rid,
        "source_gr2": source_gr2,
        "duration": float(anim.duration),
        "root": skel.get("root") or "",
        "skeleton": bones_out,
        "tracks": tracks,
    }

    # Optional bind / T-pose (display + Blender Edit basis only — ignored by track diff)
    try:
        from bind_pose import load_bind_for_dump, merge_bind_with_clip_skeleton

        bind_doc = load_bind_for_dump(dump_dir)
        if bind_doc:
            merged = merge_bind_with_clip_skeleton(bind_doc, skel)
            doc["bind"] = {
                "format": bind_doc.get("format"),
                "version": bind_doc.get("version"),
                "source_gr2": bind_doc.get("source_gr2"),
                "root": bind_doc.get("root") or skel.get("root") or "",
                "bones": [
                    {
                        "name": b["name"],
                        "parent_name": b.get("parent_name"),
                        "bind_translation": list(b.get("bind_translation") or [0, 0, 0]),
                        "bind_rotation": list(b.get("bind_rotation") or [0, 0, 0, 1]),
                        "bind_scale": list(b.get("bind_scale") or [1, 1, 1]),
                        "from": b.get("from"),
                    }
                    for b in merged
                ],
            }
    except Exception:
        pass

    if densify_bspline:
        densify_doc_bspline(doc, anim, fps=densify_fps)

    return doc


def write_gr2lab(doc: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_gr2lab(path: Path) -> Dict[str, Any]:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_gr2lab(doc)
    return doc


def validate_gr2lab(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("gr2lab: root must be an object")
    if doc.get("format") != FORMAT_ID:
        raise ValueError(f"gr2lab: expected format={FORMAT_ID!r}")
    ver = int(doc.get("version") or 0)
    if ver < 1 or ver > FORMAT_VERSION:
        raise ValueError(f"gr2lab: unsupported version {ver}")
    if not doc.get("run_id"):
        raise ValueError("gr2lab: missing run_id")
    if not isinstance(doc.get("tracks"), list):
        raise ValueError("gr2lab: missing tracks[]")
    if not isinstance(doc.get("skeleton"), list):
        raise ValueError("gr2lab: missing skeleton[]")


def suggested_filename(doc: Dict[str, Any]) -> str:
    """Short sidecar name — avoid Windows MAX_PATH with long BG3 stems + run ids."""
    rid = str(doc.get("run_id") or "dump")
    # Prefer run folder name alone; keep under ~80 chars
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in rid)[:80]
    if not safe:
        safe = "anim"
    return f"{safe}.gr2lab"


def _track_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["name"]: t for t in doc.get("tracks") or [] if t.get("name")}


def _find_time_index(times: Sequence[float], t: float, eps: float = TIME_EPS) -> int:
    for i, tt in enumerate(times):
        if abs(float(tt) - float(t)) <= eps:
            return i
    return -1


def _diff_constant_or_bias(
    channel: str,
    mode: str,
    old_vals: Sequence[Sequence[float]],
    new_vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
) -> List[Dict[str, Any]]:
    """Bias/single-key diff. D3K16 uses offset_key so ControlOffsets stay a bias."""
    if not new_vals:
        return []
    nv = new_vals[0]
    ov = old_vals[0] if old_vals else None
    if is_quat:
        nv = _norm_quat(nv)
        if ov is not None and _quat_close(ov, nv):
            return []
        return [{"op": "set_key", "channel": channel, "quat": nv}]
    eps = SCL_EPS if channel == "scale" else POS_EPS
    if ov is not None and _v3_close(ov, nv, eps):
        return []
    # k16_offsets: Blender/site display the dequantized first knot; writing that
    # absolute into ControlOffsets would corrupt the curve. Apply a delta instead.
    if mode == "k16_offsets" and ov is not None:
        delta = [float(nv[i]) - float(ov[i]) for i in range(3)]
        if all(abs(d) <= eps for d in delta):
            return []
        return [{"op": "offset_key", "channel": channel, "xyz": delta}]
    return [{"op": "set_key", "channel": channel, "xyz": list(map(float, nv))}]


def _diff_keyed_channel(
    channel: str,
    mode: str,
    old_times: Sequence[float],
    old_vals: Sequence[Sequence[float]],
    new_times: Sequence[float],
    new_vals: Sequence[Sequence[float]],
    capacity: int,
    *,
    is_quat: bool,
) -> List[Dict[str, Any]]:
    """Produce set/add/delete ops so dump matches new curve (site viewport Save)."""
    if len(new_times) != len(new_vals):
        raise ValueError(f"{channel}: times/values length mismatch")
    if len(old_times) != len(old_vals):
        raise ValueError(f"{channel}: dump times/values length mismatch")

    ops: List[Dict[str, Any]] = []
    old_t = [float(t) for t in old_times]
    new_t = [float(t) for t in new_times]
    new_v = [list(v) for v in new_vals]

    # float_controls: allow denser keys — apply_keys_to_dump grows the buffer.
    # (Previously baked down to capacity; that blocked Blender from adding keys.)

    # Deletes first (high index → low) for times that disappeared
    to_delete: List[Tuple[int, float]] = []
    for i, t in enumerate(old_t):
        if _find_time_index(new_t, t) < 0:
            to_delete.append((i, t))
    for i, t in sorted(to_delete, key=lambda x: -x[0]):
        ops.append(
            {
                "op": "delete_key",
                "channel": channel,
                "index": i,
                "from_t": t,
            }
        )

    # After conceptual deletes, remaining old keys are those still in new
    remaining_old = [(t, old_vals[i]) for i, t in enumerate(old_t) if _find_time_index(new_t, t) >= 0]

    # Sets for existing times with changed values; adds for brand-new times
    adds = 0
    for t, v in zip(new_t, new_v):
        oi = _find_time_index([x[0] for x in remaining_old], t)
        if oi >= 0:
            ov = remaining_old[oi][1]
            if is_quat:
                nv = _norm_quat(v)
                if not _quat_close(ov, nv):
                    ops.append(
                        {
                            "op": "set_key",
                            "channel": channel,
                            "from_t": t,
                            "quat": nv,
                        }
                    )
            else:
                if not _v3_close(ov, v, SCL_EPS if channel == "scale" else POS_EPS):
                    ops.append(
                        {
                            "op": "set_key",
                            "channel": channel,
                            "from_t": t,
                            "xyz": list(map(float, v)),
                        }
                    )
        else:
            adds += 1
            if is_quat:
                ops.append(
                    {
                        "op": "add_key",
                        "channel": channel,
                        "t": t,
                        "quat": _norm_quat(v),
                    }
                )
            else:
                ops.append(
                    {
                        "op": "add_key",
                        "channel": channel,
                        "t": t,
                        "xyz": list(map(float, v)),
                    }
                )

    final_count = len(old_t) - len(to_delete) + adds
    # Capacity check removed for float_controls — grow path handles it.
    # quat_k16 can grow via reloc.
    return ops


# Prefer viewport set/add/delete ops; bulk replace only for huge diffs (speed).
VIEWPORT_OPS_BULK_THRESHOLD = 64


def _keyed_channel_ops_or_bulk(
    *,
    channel: str,
    mode: str,
    src_times: Sequence[float],
    src_vals: Sequence[Sequence[float]],
    dst_times: Sequence[float],
    dst_vals: Sequence[Sequence[float]],
    capacity: int,
    is_quat: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[List[float], List[List[float]]]]]:
    """Return (viewport_ops, bulk_series).

    bulk_series is (times, values) when the op list is large enough that one-shot
    replace_* is faster; otherwise ops are the site-viewport Save path.
    """
    ops = _diff_keyed_channel(
        channel,
        mode,
        src_times,
        src_vals,
        dst_times,
        dst_vals,
        capacity,
        is_quat=is_quat,
    )
    if not ops:
        return [], None
    if len(ops) > VIEWPORT_OPS_BULK_THRESHOLD:
        return [], (
            [float(t) for t in dst_times],
            [list(map(float, v)) for v in dst_vals],
        )
    return ops, None


def _sample_curve_at(
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    t: float,
    *,
    is_quat: bool,
) -> List[float]:
    """Nearest / hold sample (good enough for capacity bake-down)."""
    if not times or not values:
        return [0.0, 0.0, 0.0, 1.0] if is_quat else [0.0, 0.0, 0.0]
    if t <= times[0]:
        v = values[0]
        return _norm_quat(v) if is_quat else list(map(float, v))
    if t >= times[-1]:
        v = values[-1]
        return _norm_quat(v) if is_quat else list(map(float, v))
    i = bisect_left(times, t)
    if i <= 0:
        v = values[0]
    elif i >= len(times):
        v = values[-1]
    elif abs(t - times[i]) < abs(t - times[i - 1]):
        v = values[i]
    else:
        v = values[i - 1]
    return _norm_quat(v) if is_quat else list(map(float, v))


def diff_bridge_to_edits(
    dump_dir: Path,
    doc: Dict[str, Any],
    *,
    anim=None,
    s0=None,
) -> List[Dict[str, Any]]:
    """Compare bridge file to dump; return [{bone, ops}, ...] for apply_keys_to_dump.

    Matches site viewport Save capabilities:
    - constant / k16_offsets → bias-only (never keyed D3K16 rewrites)
    - float_controls / quat_k16 → set/add/delete key ops
    """
    from granny_anim import load_dump_gr2, sample_curve

    validate_gr2lab(doc)
    dump_dir = Path(dump_dir)
    if anim is None or s0 is None:
        _gr2, anim, s0 = load_dump_gr2(dump_dir)
    incoming = _track_map(doc)
    edits: List[Dict[str, Any]] = []

    for t in anim.tracks:
        name = t.name
        if not name or name not in incoming:
            continue
        src = sample_curve(s0, anim, name)
        dst = incoming[name]
        ops: List[Dict[str, Any]] = []

        # Position — trust dump editability, not the bridge file
        dpos = dst.get("position") or {}
        mode = src.get("edit_mode") or "none"
        if src.get("editable") and mode != "none":
            src_times = src.get("times") or []
            src_pos = src.get("positions") or []
            dst_times = [float(x) for x in (dpos.get("times") or [])]
            dst_vals = dpos.get("values") or []
            if mode in ("constant", "k16_offsets"):
                # Bias only — same as timeline (no add/delete on D3K16)
                if dst_vals:
                    ops.extend(
                        _diff_constant_or_bias(
                            "position",
                            mode,
                            src_pos,
                            dst_vals,
                            is_quat=False,
                        )
                    )
            elif mode == "float_controls":
                if dst_vals and not _pos_series_close(
                    src_times, src_pos, dst_times, dst_vals
                ):
                    ops.extend(
                        _diff_keyed_channel(
                            "position",
                            mode,
                            src_times,
                            src_pos,
                            dst_times,
                            dst_vals,
                            int(src.get("capacity") or src.get("knot_count") or 0),
                            is_quat=False,
                        )
                    )

        # Orientation
        sori = src.get("orientation") or {}
        dori = dst.get("orientation") or {}
        mode = sori.get("edit_mode") or "none"
        if sori.get("editable") and mode != "none":
            src_ot = sori.get("times") or []
            src_oq = sori.get("rotations") or []
            dst_ot = [float(x) for x in (dori.get("times") or [])]
            dst_oq = dori.get("values") or []
            if mode == "constant":
                if dst_oq:
                    ops.extend(
                        _diff_constant_or_bias(
                            "orientation",
                            mode,
                            src_oq,
                            dst_oq,
                            is_quat=True,
                        )
                    )
            elif mode in ("float_controls", "quat_k16"):
                if dst_oq and not _quat_series_close(
                    src_ot, src_oq, dst_ot, dst_oq
                ):
                    ops.extend(
                        _diff_keyed_channel(
                            "orientation",
                            mode,
                            src_ot,
                            src_oq,
                            dst_ot,
                            dst_oq,
                            int(sori.get("capacity") or sori.get("knot_count") or 0),
                            is_quat=True,
                        )
                    )

        # Scale
        sscl = src.get("scale") or {}
        dscl = dst.get("scale") or {}
        mode = sscl.get("edit_mode") or "none"
        if sscl.get("editable") and mode != "none":
            src_st = sscl.get("times") or []
            src_sv = sscl.get("scales") or []
            dst_st = [float(x) for x in (dscl.get("times") or [])]
            dst_sv = dscl.get("values") or []
            if mode in ("constant", "k16_offsets"):
                if dst_sv:
                    ops.extend(
                        _diff_constant_or_bias(
                            "scale",
                            mode,
                            src_sv,
                            dst_sv,
                            is_quat=False,
                        )
                    )
            elif mode == "float_controls":
                if dst_sv and not _pos_series_close(
                    src_st, src_sv, dst_st, dst_sv
                ):
                    ops.extend(
                        _diff_keyed_channel(
                            "scale",
                            mode,
                            src_st,
                            src_sv,
                            dst_st,
                            dst_sv,
                            int(sscl.get("capacity") or sscl.get("knot_count") or 0),
                            is_quat=False,
                        )
                    )

        if ops:
            edits.append({"bone": name, "ops": ops})

    return edits


def _root_pos_max_delta_vs_dump(
    src_times: Sequence[float],
    src_pos: Sequence[Sequence[float]],
    pos_times: Sequence[float],
    pos_vals: Sequence[Sequence[float]],
    *,
    s0: Optional[bytes] = None,
    cm: Any = None,
) -> Tuple[float, float]:
    """Max ground-plane / vertical delta of bridge samples vs dump curve."""
    max_lat = 0.0
    max_dy = 0.0
    use_eval = s0 is not None and cm is not None
    eval_position_at = None
    if use_eval:
        from granny_anim import eval_position_at as _eval_position_at

        eval_position_at = _eval_position_at
    for t, v in zip(pos_times or [], pos_vals or []):
        if not v:
            continue
        if use_eval and eval_position_at is not None:
            p = eval_position_at(s0, cm, float(t))
        else:
            p = _lerp_vec(src_times, src_pos, float(t), n=3)
        dx = float(v[0]) - float(p[0])
        dy = float(v[1]) - float(p[1])
        dz = float(v[2]) - float(p[2]) if len(v) > 2 and len(p) > 2 else 0.0
        max_lat = max(max_lat, math.sqrt(dx * dx + dz * dz))
        max_dy = max(max_dy, abs(dy))
    return max_lat, max_dy


def _root_pos_is_display_resample(
    name: str,
    pos_mode: str,
    dpos: Dict[str, Any],
    src_times: Sequence[float],
    src_pos: Sequence[Sequence[float]],
    pos_times: Sequence[float],
    pos_vals: Sequence[Sequence[float]],
    *,
    motion_close: bool,
    s0: Optional[bytes] = None,
    cm: Any = None,
) -> bool:
    """True when Dummy_Root/Root_M bridge is import densify, not a locomotion edit.

    Display densify of k16 Dummy_Root must not float-upgrade (in-game ~2 cm shift).
    Constraint/NLA Root_M deltas larger than ROOT_POS_DENSIFY_EPS must still ship.
    """
    if name not in ("Dummy_Root", "Root_M"):
        return False
    note = str(dpos.get("note") or "")
    looks = (
        "bspline_densified" in note
        or "k16_display_densified" in note
        or bool(dpos.get("display_densified"))
        or (
            pos_vals
            and len(pos_times) >= 100
            and _is_mass_grid_densify(src_times, pos_times)
        )
    )
    if not looks:
        return False
    if motion_close:
        return True
    if pos_mode not in ("k16_offsets", "float_controls", "constant"):
        return False
    max_lat, max_dy = _root_pos_max_delta_vs_dump(
        src_times, src_pos, pos_times, pos_vals, s0=s0, cm=cm
    )
    return max_lat <= ROOT_POS_DENSIFY_EPS and max_dy <= ROOT_POS_DENSIFY_EPS


def _root_ori_looks_like_densify(
    name: str,
    ori_mode: str,
    dori: Dict[str, Any],
    src_times: Sequence[float],
    ori_times: Sequence[float],
    ori_vals: Sequence[Sequence[float]],
) -> bool:
    """True when the bridge *looks* like import/bake densify of Root_M ori."""
    if name not in ("Dummy_Root", "Root_M"):
        return False
    note = str(dori.get("note") or "")
    if (
        "bspline_densified" in note
        or "k16_display_densified" in note
        or bool(dori.get("display_densified"))
    ):
        return True
    if (
        ori_vals
        and len(ori_times) >= 100
        and _is_mass_grid_densify(src_times, ori_times)
    ):
        return True
    # Bake-shaped float bridge onto Root_M quat_k16.
    if (
        name == "Root_M"
        and ori_mode == "quat_k16"
        and (dori.get("edit_mode") or "") == "float_controls"
        and len(ori_times) >= 100
    ):
        return True
    return False


def _quat_dest_matches_src(
    src_times: Sequence[float],
    src_vals: Sequence[Sequence[float]],
    dst_times: Sequence[float],
    dst_vals: Sequence[Sequence[float]],
    *,
    quat_eps: float = NOOP_QUAT_DOT_EPS,
) -> bool:
    """True when dest samples match src motion (one-way resample).

    Reverse reconstruction of sparse Deg2 knots from a denser/linear grid is
    not required — that check treated vanilla Root_M bake densify as a rewrite.
    """
    if not src_vals or not dst_vals or not dst_times:
        return False
    for t, v in zip(dst_times, dst_vals):
        if not _quat_close(_slerp_quat(src_times, src_vals, float(t)), v, quat_eps):
            return False
    return True


def _root_ori_is_display_resample(
    name: str,
    ori_mode: str,
    dori: Dict[str, Any],
    src_times: Sequence[float],
    src_quats: Sequence[Sequence[float]],
    ori_times: Sequence[float],
    ori_vals: Sequence[Sequence[float]],
) -> bool:
    """Skip Root_M / Dummy_Root ori apply only when densify is the *same motion*.

    Real NLA COMBINE / keyed Root_M rotation must still float-upgrade Deg1.
    """
    if not _root_ori_looks_like_densify(
        name, ori_mode, dori, src_times, ori_times, ori_vals
    ):
        return False
    if not ori_vals or not src_quats:
        return True
    return _quat_dest_matches_src(src_times, src_quats, ori_times, ori_vals)


def _bridge_has_varying_keys(
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    *,
    is_quat: bool,
) -> bool:
    """True when bridge has real multi-key motion (not a flat re-export)."""
    if len(times) <= 1 or not values:
        return False
    v0 = values[0]
    if is_quat:
        return not all(_quat_close(v0, v) for v in values)
    return not all(_v3_close(v0, v) for v in values)


def plan_upgrades_from_bridge(
    dump_dir: Path,
    doc: Dict[str, Any],
    *,
    anim=None,
    s0=None,
) -> List[Dict[str, Any]]:
    """Decide which bones need curve format upgrades before applying bridge keys.

    Only upgrades constant/k16/none when the bridge has real multi-key intent —
    not when Blender re-exported a flat or identical dequantized series.
    """
    from granny_anim import IDENTITY_QUAT, load_dump_gr2, sample_curve

    validate_gr2lab(doc)
    dump_dir = Path(dump_dir)
    if anim is None or s0 is None:
        _gr2, anim, s0 = load_dump_gr2(dump_dir)
    incoming = _track_map(doc)
    upgrades: List[Dict[str, Any]] = []

    for track in anim.tracks:
        name = track.name
        if not name or name not in incoming:
            continue
        dst = incoming[name]
        dpos = dst.get("position") or {}
        dori = dst.get("orientation") or {}

        pos_times = [float(t) for t in (dpos.get("times") or [])]
        pos_vals = dpos.get("values") or []
        pos_mode = track.position.edit_mode
        if (
            pos_mode in ("constant", "k16_offsets", "none")
            and _bridge_has_varying_keys(pos_times, pos_vals, is_quat=False)
        ):
            src = sample_curve(s0, anim, name)
            src_times = src.get("times") or []
            src_pos = src.get("positions") or []
            # Identical / motion-equivalent to dump → noop, do not upgrade
            motion_close = _pos_series_motion_close(
                src_times, src_pos, pos_times, pos_vals
            )
            # Dummy_Root/Root_M import densify must not float-upgrade k16 (2 cm
            # in-game shift). Real NLA/constraint locomotion still upgrades.
            root_densify_noise = _root_pos_is_display_resample(
                name,
                pos_mode,
                dpos,
                src_times,
                src_pos,
                pos_times,
                pos_vals,
                motion_close=motion_close,
                s0=s0,
                cm=track.position,
            )
            if not motion_close and not root_densify_noise:
                upgrades.append(
                    {
                        "bone": name,
                        "channel": "position",
                        "times": pos_times,
                        "values": pos_vals,
                        "slack": max(64, len(pos_times) + 32),
                    }
                )

        ori_times = [float(t) for t in (dori.get("times") or [])]
        ori_vals = dori.get("values") or []
        ori_mode = track.orientation.edit_mode
        # Refuse Dummy_Root Rx(±90°) — that is Blender Z-up rest, not Granny anim.
        if name == "Dummy_Root" and _dummy_root_ori_is_false_zup(ori_vals):
            continue
        src_curve = sample_curve(s0, anim, name)
        sori = src_curve.get("orientation") or {}
        src_ot = sori.get("times") or []
        src_oq = sori.get("rotations") or []
        # Same-motion densify / bake of vanilla Root_M ori → keep sparse quat_k16.
        # Different rotation (NLA COMBINE) → fall through and float-upgrade Deg1.
        if _root_ori_is_display_resample(
            name,
            ori_mode,
            dori,
            src_ot,
            src_oq,
            ori_times,
            ori_vals,
        ):
            continue
        # none/constant → may need first writable curve.
        # quat_k8 is display-only in the dump (editable=False) — Blender multi-key
        # float bridges must float-upgrade it or Spine1_M-style bones keep the
        # original quantized motion while the rest of the body updates (BG3 break).
        # Root_M quat_k16 with *changed* motion (not skipped above) → float Deg1.
        root_k16 = name in ("Dummy_Root", "Root_M") and ori_mode == "quat_k16"
        if ori_mode in ("none", "constant", "quat_k8") or (root_k16 and ori_vals):
            need = False
            if _bridge_has_varying_keys(ori_times, ori_vals, is_quat=True):
                need = True
            elif ori_mode == "none" and not _quat_close(ori_vals[0], IDENTITY_QUAT):
                need = True
            if need and ori_mode in ("quat_k8", "quat_k16"):
                if _quat_series_motion_close(src_ot, src_oq, ori_times, ori_vals):
                    need = False
            if need:
                upgrades.append(
                    {
                        "bone": name,
                        "channel": "orientation",
                        "times": ori_times if ori_times else [0.0],
                        "quats": ori_vals,
                        # Multi-key → float Degree=1 (see apply_curve_upgrades).
                        "prefer_float": len(ori_times) > 1,
                    }
                )
    return upgrades


def _root_channel_summary_from_anim(anim) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("Dummy_Root", "Root_M", "Hip_L"):
        tr = next((t for t in (anim.tracks or []) if t.name == name), None)
        if tr is None:
            continue
        pos = tr.position
        ori = tr.orientation
        out[name] = {
            "edit_mode": pos.edit_mode,
            "degree": pos.degree,
            "knot_count": pos.knot_count,
            "format": pos.format,
            "ori_edit_mode": ori.edit_mode,
            "ori_degree": ori.degree,
            "ori_knot_count": ori.knot_count,
            "ori_format": ori.format,
        }
    return out


def persist_bridge_export_meta(
    dump_dir: Path,
    doc: Dict[str, Any],
    apply_result: Dict[str, Any],
    *,
    anim=None,
) -> Path:
    """Write/merge EXPORT_META.json after a Blender bridge apply (Gate E)."""
    import time
    from pathlib import Path as _Path

    # Prefer helper in tools when available; keep a local fallback.
    dump_dir = _Path(dump_dir)
    em = dict(doc.get("export_meta") or {})
    if doc.get("export_space") and "space" not in em:
        em["space"] = doc.get("export_space")
    if "granny_space" not in em and "space" in em:
        em["granny_space"] = em["space"] == "granny"

    root_after = None
    if anim is not None:
        root_after = _root_channel_summary_from_anim(anim)
    else:
        try:
            from granny_anim import load_dump_gr2

            _g, anim2, _s0 = load_dump_gr2(dump_dir, use_cache=False)
            root_after = _root_channel_summary_from_anim(anim2)
        except Exception:
            root_after = None

    meta = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blender_full_export": bool(doc.get("blender_full_export")),
        "run_id": doc.get("run_id") or dump_dir.name,
        "space": em.get("space"),
        "granny_space": em.get("granny_space"),
        "y_up_display": em.get("y_up_display"),
        "object_euler_x_deg": em.get("object_euler_x_deg"),
        "frames_sampled": em.get("frames_sampled"),
        "duration_sec": em.get("duration_sec") or doc.get("duration"),
        "baked": em.get("baked"),
        "template_source": em.get("template_source"),
        "fps": em.get("fps"),
        "root_channels_after": root_after,
        "apply": {
            "changed": bool(apply_result.get("changed")),
            "bones_edited": apply_result.get("bones_edited"),
            "ops": apply_result.get("ops"),
            "bulk": apply_result.get("bulk"),
            "message": apply_result.get("message"),
        },
    }
    # Drop None noise
    meta = {k: v for k, v in meta.items() if v is not None}

    try:
        sys_path_tools = Path(__file__).resolve().parent / "tools"
        if str(sys_path_tools) not in sys.path:
            sys.path.insert(0, str(sys_path_tools))
        from _validate_root_export import write_export_meta  # type: ignore

        return write_export_meta(dump_dir, meta)
    except Exception:
        path = dump_dir / "EXPORT_META.json"
        prev: Dict[str, Any] = {}
        if path.is_file():
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        prev.update(meta)
        path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        return path


def _persist_upgraded_dump(
    dump_dir: Path,
    gr2,
    s0: bytearray,
    s1: bytearray,
    reloc0: bytearray,
    reloc1: bytearray,
    anim,
) -> None:
    """Write section 0/1 bins + relocs + refreshed anim_map/skeleton after upgrades."""
    from granny_anim import build_anim_map, build_skeleton, write_anim_map, write_skeleton

    dump_dir = Path(dump_dir)
    gr2.sections[0].uncompressed = bytes(s0)
    gr2.sections[0].header.uncompressed_size = len(s0)
    gr2.sections[0].relocations = bytes(reloc0)
    gr2.sections[1].uncompressed = bytes(s1)
    gr2.sections[1].header.uncompressed_size = len(s1)
    gr2.sections[1].relocations = bytes(reloc1)

    (dump_dir / "section_0.bin").write_bytes(s0)
    (dump_dir / "section_1.bin").write_bytes(s1)
    (dump_dir / "section_0.relocations.bin").write_bytes(reloc0)
    (dump_dir / "section_1.relocations.bin").write_bytes(reloc1)
    for i, sec in enumerate(gr2.sections):
        if sec.mixed_marshalling is not None:
            (dump_dir / f"section_{i}.mm.bin").write_bytes(sec.mixed_marshalling)

    # Preserve float capacity hints in anim_map
    anim2 = build_anim_map(gr2, source=anim.source)
    for t_old, t_new in zip(anim.tracks, anim2.tracks):
        if t_old.name != t_new.name:
            continue
        if t_old.position.edit_mode == "float_controls":
            t_new.position.values_bytes = max(
                t_new.position.values_bytes, t_old.position.values_bytes
            )
        if t_old.orientation.edit_mode == "quat_k16":
            t_new.orientation.values_bytes = max(
                t_new.orientation.values_bytes, t_old.orientation.values_bytes
            )
    write_anim_map(anim2, dump_dir / "anim_map.json")
    skel = build_skeleton(anim2, bytes(s0))
    write_skeleton(skel, dump_dir / "skeleton.json")


def apply_upgrades_to_dump(
    dump_dir: Path,
    upgrades: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run curve upgrades and persist dump sections."""
    from granny_anim import apply_curve_upgrades, load_dump_gr2

    if not upgrades:
        return []
    dump_dir = Path(dump_dir)
    gr2, anim, s0 = load_dump_gr2(dump_dir)
    s1 = bytearray(gr2.sections[1].uncompressed or b"")
    reloc0 = bytearray(gr2.sections[0].relocations or b"")
    reloc1 = bytearray(gr2.sections[1].relocations or b"")
    if not reloc0 or not reloc1:
        raise ValueError("dump missing section relocations — re-Decode before upgrade")
    results = apply_curve_upgrades(s0, s1, anim, reloc0, reloc1, upgrades)
    _persist_upgraded_dump(dump_dir, gr2, s0, s1, reloc0, reloc1, anim)
    return results


def apply_gr2lab_to_dump(
    dump_dir: Path,
    doc: Dict[str, Any],
    *,
    overwrite: bool = True,
    timer=None,
) -> Dict[str, Any]:
    """Apply Blender bridge like the site viewport Save path.

    - constant / k16_offsets → bias-only key ops
    - float_controls / quat_k16 → set/add/delete key ops (same as viewport);
      one-shot replace_* only when the op list is huge (dense bake speed)
    - significant extra keys apply as add_key; on-curve densify stays noop
    - noop when series motion-equivalent (no S0 write)
    """
    from granny_anim import (
        apply_key_ops,
        load_dump_gr2,
        replace_orientation_curve,
        replace_position_curve,
        sample_curve,
        upgrade_orientation_to_float,
        _grow_dak32_channel,
        _rewrite_dak32_curve,
        float_curve_capacity,
    )

    validate_gr2lab(doc)
    dump_dir = Path(dump_dir)
    from timing import maybe_timer

    upgrades = plan_upgrades_from_bridge(dump_dir, doc)
    upgrade_results: List[Dict[str, Any]] = []
    with maybe_timer(timer, "apply_upgrades"):
        if upgrades:
            upgrade_results = apply_upgrades_to_dump(dump_dir, upgrades)

    gr2, anim, s0 = load_dump_gr2(dump_dir)
    s1 = bytearray(gr2.sections[1].uncompressed or b"")
    reloc0 = bytearray(gr2.sections[0].relocations or b"")
    reloc1 = bytearray(gr2.sections[1].relocations or b"")
    if not reloc0 or not reloc1:
        raise ValueError("dump missing section relocations — re-Decode before import")

    incoming = _track_map(doc)
    force_blender = bool(doc.get("blender_full_export"))
    results: List[Dict[str, Any]] = []
    bones_touched = 0
    used_bulk = False

    src_cache: Dict[str, Any] = {}

    def _src_curve(bone: str) -> Dict[str, Any]:
        hit = src_cache.get(bone)
        if hit is None:
            hit = sample_curve(s0, anim, bone)
            src_cache[bone] = hit
        return hit

    import time as _time

    _t_rep = _time.perf_counter()
    for track in anim.tracks:
        name = track.name
        if not name or name not in incoming:
            continue
        dst = incoming[name]
        bone_ops: List[Dict[str, Any]] = []
        touched = False

        # Position
        dpos = dst.get("position") or {}
        pos_times = [float(t) for t in (dpos.get("times") or [])]
        pos_vals = dpos.get("values") or []
        pos_mode = track.position.edit_mode
        # Whole-character placement: refuse display-densify rewrites on apply.
        # Real Root_M / Dummy_Root locomotion edits (above densify eps) still apply.
        src_for_root = _src_curve(name) if name in ("Dummy_Root", "Root_M") else {}
        skip_root_pos = _root_pos_is_display_resample(
            name,
            pos_mode,
            dpos,
            src_for_root.get("times") or [],
            src_for_root.get("positions") or [],
            pos_times,
            pos_vals,
            motion_close=False,
            s0=s0,
            cm=track.position,
        )
        if (
            track.position.editable
            and pos_mode != "none"
            and pos_vals
            and not skip_root_pos
        ):
            src = _src_curve(name)
            src_times = src.get("times") or []
            src_pos = src.get("positions") or []
            if pos_mode in ("constant", "k16_offsets"):
                bone_ops.extend(
                    _diff_constant_or_bias(
                        "position", pos_mode, src_pos, pos_vals, is_quat=False
                    )
                )
            elif pos_mode == "float_controls":
                write = _resolve_grid_remake_write(
                    src_times, src_pos, pos_times, pos_vals, is_quat=False,
                    force_blender_export=force_blender,
                )
                if write is not None:
                    w_times, w_vals = write
                    ops, bulk = _keyed_channel_ops_or_bulk(
                        channel="position",
                        mode=pos_mode,
                        src_times=src_times,
                        src_vals=src_pos,
                        dst_times=w_times,
                        dst_vals=w_vals,
                        capacity=int(
                            src.get("capacity") or src.get("knot_count") or len(src_times)
                        ),
                        is_quat=False,
                    )
                    if ops:
                        bone_ops.extend(ops)
                    elif bulk is not None:
                        meta = replace_position_curve(
                            s0,
                            anim,
                            name,
                            reloc0,
                            reloc1,
                            bulk[0] or [0.0],
                            bulk[1],
                        )
                        if meta.get("replaced") or meta.get("upgraded"):
                            results.append(meta)
                            touched = True
                            used_bulk = True

        # Orientation
        dori = dst.get("orientation") or {}
        ori_times = [float(t) for t in (dori.get("times") or [])]
        ori_vals = dori.get("values") or []
        ori_mode = track.orientation.edit_mode
        # Body_Base templates bake Dummy_Root rest Rx(90°) into the track; skip.
        if name == "Dummy_Root" and _dummy_root_ori_is_false_zup(ori_vals):
            ori_vals = []
        src_ori = src_for_root.get("orientation") or {}
        skip_root_ori = _root_ori_is_display_resample(
            name,
            ori_mode,
            dori,
            src_ori.get("times") or [],
            src_ori.get("rotations") or [],
            ori_times,
            ori_vals,
        )
        if (
            track.orientation.editable
            and ori_mode != "none"
            and ori_vals
            and not skip_root_ori
        ):
            src = _src_curve(name)
            sori = src.get("orientation") or {}
            src_ot = sori.get("times") or []
            src_oq = sori.get("rotations") or []
            if ori_mode == "constant":
                bone_ops.extend(
                    _diff_constant_or_bias(
                        "orientation", ori_mode, src_oq, ori_vals, is_quat=True
                    )
                )
            elif ori_mode in ("float_controls", "quat_k16"):
                write = _resolve_grid_remake_write(
                    src_ot, src_oq, ori_times, ori_vals, is_quat=True,
                    force_blender_export=force_blender,
                )
                if write is not None:
                    w_times, w_vals = write
                    ops, bulk = _keyed_channel_ops_or_bulk(
                        channel="orientation",
                        mode=ori_mode,
                        src_times=src_ot,
                        src_vals=src_oq,
                        dst_times=w_times,
                        dst_vals=w_vals,
                        capacity=int(
                            sori.get("capacity")
                            or sori.get("knot_count")
                            or len(src_ot)
                        ),
                        is_quat=True,
                    )
                    if ops:
                        bone_ops.extend(ops)
                    elif bulk is not None:
                        meta = replace_orientation_curve(
                            s0,
                            anim,
                            name,
                            reloc0,
                            reloc1,
                            bulk[0] or [0.0],
                            bulk[1],
                        )
                        if meta.get("replaced") or meta.get("upgraded"):
                            results.append(meta)
                            touched = True
                            used_bulk = True

        # Safety: display-only quat_k8 never enters the editable path above.
        # replace_orientation_curve also refuses non-editable curves — call the
        # float upgrade directly so Spine1_M-style bones cannot keep the original
        # quantized kiss while the rest of the body updates.
        elif (
            ori_mode == "quat_k8"
            and ori_vals
            and not skip_root_ori
            and (dori.get("edit_mode") or "") == "float_controls"
            and len(ori_times) > 1
        ):
            meta = upgrade_orientation_to_float(
                s0,
                anim,
                name,
                reloc0,
                reloc1,
                times=ori_times,
                quats=ori_vals,
            )
            if meta.get("upgraded"):
                results.append(meta)
                touched = True
                used_bulk = True

        # Scale
        dscl = dst.get("scale") or {}
        scl_times = [float(t) for t in (dscl.get("times") or [])]
        scl_vals = dscl.get("values") or []
        scl_mode = track.scale.edit_mode
        if track.scale.editable and scl_mode != "none" and scl_vals:
            src = _src_curve(name)
            sscl = src.get("scale") or {}
            if scl_mode in ("constant", "k16_offsets"):
                bone_ops.extend(
                    _diff_constant_or_bias(
                        "scale",
                        scl_mode,
                        sscl.get("scales") or [],
                        scl_vals,
                        is_quat=False,
                    )
                )
            elif scl_mode == "float_controls" and (
                len(scl_times) > 1 or len(sscl.get("times") or []) > 1
            ):
                src_st = sscl.get("times") or []
                src_sv = sscl.get("scales") or []
                write = _resolve_grid_remake_write(
                    src_st, src_sv, scl_times, scl_vals, is_quat=False,
                    force_blender_export=force_blender,
                )
                if write is not None:
                    w_times, w_vals = write
                    ops, bulk = _keyed_channel_ops_or_bulk(
                        channel="scale",
                        mode=scl_mode,
                        src_times=src_st,
                        src_vals=src_sv,
                        dst_times=w_times,
                        dst_vals=w_vals,
                        capacity=int(
                            sscl.get("capacity")
                            or sscl.get("knot_count")
                            or len(src_st)
                        ),
                        is_quat=False,
                    )
                    if ops:
                        bone_ops.extend(ops)
                    elif bulk is not None:
                        cm = track.scale
                        n = len(bulk[0])
                        if n > float_curve_capacity(cm):
                            _grow_dak32_channel(
                                s0,
                                cm,
                                reloc0,
                                bulk[0],
                                bulk[1],
                                comps=3,
                                slack=32,
                            )
                        else:
                            _rewrite_dak32_curve(
                                s0,
                                cm,
                                bulk[0],
                                [list(map(float, v)) for v in bulk[1]],
                                comps=3,
                            )
                        results.append(
                            {
                                "replaced": True,
                                "bone": name,
                                "channel": "scale",
                                "knots": n,
                            }
                        )
                        touched = True
                        used_bulk = True

        if bone_ops:
            applied = apply_key_ops(
                s0, anim, name, bone_ops, reloc_blob=reloc0 if reloc0 else None
            )
            results.extend(applied)
            touched = True

        if touched:
            bones_touched += 1

    if timer is not None:
        timer.add("apply_replace", _time.perf_counter() - _t_rep)

    if not results and not upgrade_results:
        out = {
            "ok": True,
            "run_id": dump_dir.name,
            "changed": False,
            "bones_edited": 0,
            "ops": 0,
            "upgrades": [],
            "bulk": False,
            "message": "No animation changes vs dump (noop)",
        }
        try:
            persist_bridge_export_meta(dump_dir, doc, out, anim=anim)
        except Exception:
            pass
        return out

    if not results:
        out = {
            "ok": True,
            "run_id": dump_dir.name,
            "changed": True,
            "bones_edited": 0,
            "ops": 0,
            "upgrades": upgrade_results,
            "bridge": True,
            "bulk": False,
            "message": "Curves upgraded; no further key diffs",
        }
        try:
            persist_bridge_export_meta(dump_dir, doc, out, anim=anim)
        except Exception:
            pass
        return out

    _persist_upgraded_dump(dump_dir, gr2, s0, s1, reloc0, reloc1, anim)

    n_ops = 0
    for r in results:
        if r.get("op") or r.get("t") is not None or r.get("index") is not None:
            n_ops += 1
        elif r.get("replaced") or r.get("upgraded"):
            n_ops += int(r.get("knots") or r.get("knot_count") or 1)
        elif r.get("grown") or r.get("edit_mode"):
            n_ops += 1
    out = {
        "ok": True,
        "run_id": dump_dir.name,
        "changed": True,
        "bones_edited": bones_touched,
        "ops": n_ops,
        "upgrades": upgrade_results,
        "bridge": True,
        "bulk": used_bulk,
        "message": (
            f"Viewport-applied {bones_touched} bones / {n_ops} knots|ops"
            + (f" (+{len(upgrade_results)} upgrades)" if upgrade_results else "")
        ),
    }
    try:
        persist_bridge_export_meta(dump_dir, doc, out, anim=anim)
    except Exception:
        pass
    return out


def export_gr2lab_file(dump_dir: Path, out_path: Optional[Path] = None) -> Dict[str, Any]:
    """Pack dump and write .gr2lab next to dump (or out_path).

    Always densifies Degree≥2 channels to Degree-1 B-spline samples so Blender
    import matches in-game / site BSpline2 smoothness.
    """
    dump_dir = Path(dump_dir)
    doc = pack_dump(dump_dir, densify_bspline=True, densify_fps=DEFAULT_DENSIFY_FPS)
    if out_path is None:
        out_path = dump_dir / suggested_filename(doc)
    write_gr2lab(doc, out_path)
    return {
        "ok": True,
        "run_id": doc["run_id"],
        "path": str(out_path),
        "filename": Path(out_path).name,
        "bones": len(doc.get("skeleton") or []),
        "tracks": len(doc.get("tracks") or []),
        "duration": doc.get("duration"),
        "source_interp": doc.get("source_interp"),
        "densified_channels": doc.get("densified_channels"),
        "densify_fps": doc.get("densify_fps"),
    }
