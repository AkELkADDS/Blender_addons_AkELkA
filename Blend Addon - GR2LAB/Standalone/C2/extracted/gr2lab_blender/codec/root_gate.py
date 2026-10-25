"""Root-motion export gate — catch in-game character shift before packing.

Compares original GR2 vs encoded GR2 (world FK + local series + format contract).
Site default eval stage ``inferred`` holds Root_M k16 at key0; this gate uses
full game-like evaluation so camera-cut jumps are visible as FAIL.

Usage:
  python tools/_validate_root_export.py --base IN_GR2/foo.GR2 --encoded OUT_GR2/.../foo_encoded.GR2
  python tools/_validate_root_export.py --base ... --encoded ... --dump OUT_FILES/icebound
  python tools/_validate_root_export.py --base ... --encoded ... --allow-root-delta
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gr2_codec import decode_sections, load_gr2
from granny_anim import (
    CURVE_DEGREE_LINEAR,
    AnimMap,
    build_anim_map,
    build_skeleton,
    eval_position_at,
    eval_world_pose,
    extract_section_fixups,
    _d3k16_control_offsets,
    _read_d3k16_knots,
)

# Bones that drive cutscene placement / camera-cut visibility
ROOT_BONES = ("Dummy_Root", "Root_M")
FOCUS_BONES = ("Dummy_Root", "Root_M", "Ankle_L", "Ankle_R", "Head_M", "Hip_L")

# Defaults: polish / noop export must stay under these (metres)
DEFAULT_LATERAL_MAX = 0.02  # 2 cm
DEFAULT_VERTICAL_MAX = 0.02
DEFAULT_LOCAL_MAX = 0.01  # Gate B local series (intentional-edit soft)


def load_anim(path: Path) -> Tuple[Any, AnimMap, bytes]:
    g = load_gr2(path)
    decode_sections(g, True)
    extract_section_fixups(g)
    anim = build_anim_map(g, source=str(path))
    s0 = g.sections[0].uncompressed or b""
    return g, anim, s0


def _track(anim: AnimMap, name: str):
    for t in anim.tracks:
        if t.name == name:
            return t
    return None


def _pos_hold_k16(s0: bytes, cm) -> List[float]:
    """Site ``inferred`` stage: Root_M k16 held at ControlOffsets / key0."""
    if cm.edit_mode == "k16_offsets":
        if cm.control_offsets_offset is not None:
            return list(_d3k16_control_offsets(s0, cm))
        if cm.knots_controls_offset and cm.knot_count > 0:
            _oos, _times, positions = _read_d3k16_knots(s0, cm)
            return list(positions[0]) if positions else [0.0, 0.0, 0.0]
    return eval_position_at(s0, cm, 0.0)


def _xz(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[2]) ** 2)


def _delta_xyz(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    dx = abs(float(b[0]) - float(a[0]))
    dy = abs(float(b[1]) - float(a[1]))
    dz = abs(float(b[2]) - float(a[2]))
    return dx, dy, dz, math.sqrt(dx * dx + dz * dz)


def _sample_times(duration: float, *, hz: float = 5.0) -> List[float]:
    dur = max(0.0, float(duration or 0.0))
    if dur <= 0:
        return [0.0]
    fracs = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    times = {round(f * dur, 6) for f in fracs}
    step = 1.0 / max(1.0, hz)
    t = 0.0
    while t <= dur + 1e-9:
        times.add(round(min(dur, t), 6))
        t += step
    return sorted(times)


# ---------------------------------------------------------------------------
# Gate A — format / contract
# ---------------------------------------------------------------------------


def gate_a_format(
    anim_b: AnimMap,
    s0_b: bytes,
    anim_e: AnimMap,
    s0_e: bytes,
    *,
    expect_noop: bool = False,
) -> Dict[str, Any]:
    fails: List[str] = []
    warns: List[str] = []
    bones: Dict[str, Any] = {}

    # Empty track names ⇒ S1 reloc table wiped/unreadable (not a root pose shift).
    named_e = sum(1 for t in (anim_e.tracks or []) if t.name)
    named_b = sum(1 for t in (anim_b.tracks or []) if t.name)
    if named_b > 0 and named_e == 0 and (anim_e.tracks or []):
        fails.append(
            "encoded GR2 has no bone names (S1 relocation table empty/zeroed) — "
            "re-Encode from dump; this is not a root pose shift"
        )

    for name in ("Dummy_Root", "Root_M", "Hip_L"):
        tb = _track(anim_b, name)
        te = _track(anim_e, name)
        if tb is None or te is None:
            if name in ROOT_BONES:
                if named_e == 0 and (anim_e.tracks or []):
                    fails.append(f"{name}: unreadable (broken name relocs)")
                else:
                    fails.append(f"{name}: missing track")
            continue
        pb, pe = tb.position, te.position
        ob, oe = tb.orientation, te.orientation
        entry = {
            "base": {
                "edit_mode": pb.edit_mode,
                "degree": pb.degree,
                "knot_count": pb.knot_count,
                "format": pb.format,
                "ori_edit_mode": ob.edit_mode,
                "ori_degree": ob.degree,
                "ori_knot_count": ob.knot_count,
                "ori_format": ob.format,
            },
            "encoded": {
                "edit_mode": pe.edit_mode,
                "degree": pe.degree,
                "knot_count": pe.knot_count,
                "format": pe.format,
                "ori_edit_mode": oe.edit_mode,
                "ori_degree": oe.degree,
                "ori_knot_count": oe.knot_count,
                "ori_format": oe.format,
            },
        }
        bones[name] = entry

        def _degree_byte(s0: bytes, cm) -> int:
            if cm.object_offset is not None and cm.object_offset >= 0:
                return int(s0[cm.object_offset + 1])
            return int(cm.degree or 0)

        def _check_float_degree(label: str) -> None:
            """Bake densify must be Degree=1; vanilla sparse Degree=2 float is OK."""
            if pe.edit_mode != "float_controls" or pe.knot_count <= 1:
                return
            deg = _degree_byte(s0_e, pe)
            if deg == CURVE_DEGREE_LINEAR:
                return
            base_deg = _degree_byte(s0_b, pb) if pb.edit_mode == "float_controls" else -1
            # Dense bake / upgrade path left Degree≥2 → game overshoot risk
            grew = pe.knot_count >= max(100, int(pb.knot_count * 1.5) + 1)
            upgraded = pb.edit_mode != "float_controls"
            if grew or upgraded:
                fails.append(
                    f"{label}: multi-key float Degree={deg} after densify/upgrade "
                    f"(knots {pb.knot_count}→{pe.knot_count}; bake samples must be Degree=1)"
                )
            elif base_deg == CURVE_DEGREE_LINEAR and deg != CURVE_DEGREE_LINEAR:
                fails.append(
                    f"{label}: Degree changed {base_deg}→{deg} (must stay Degree=1)"
                )
            else:
                warns.append(
                    f"{label}: multi-key float Degree={deg} "
                    "(vanilla sparse Degree=2 OK; densified bakes must be 1)"
                )

        if name == "Dummy_Root":
            if pb.edit_mode == "float_controls" and pe.edit_mode in (
                "constant",
                "none",
                "k16_offsets",
            ):
                fails.append(
                    f"Dummy_Root: position mode {pb.edit_mode} → {pe.edit_mode} "
                    "(lost float locomotion)"
                )
            _check_float_degree("Dummy_Root")
            # Anim track must stay identity; Rx(90°) tip is Blender Z-up rest leak.
            if oe.edit_mode in ("quat_k16", "float_controls", "constant"):
                from granny_anim import eval_orientation_at

                q = eval_orientation_at(s0_e, oe, 0.0)
                rx90 = (0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
                dot = abs(
                    float(q[0]) * rx90[0]
                    + float(q[1]) * rx90[1]
                    + float(q[2]) * rx90[2]
                    + float(q[3]) * rx90[3]
                )
                if dot >= 0.995:
                    fails.append(
                        "Dummy_Root: orientation track is ~Rx(90°) "
                        "(Blender Z-up rest leaked into Granny anim)"
                    )

        if name == "Root_M":
            if expect_noop and pb.edit_mode == "k16_offsets" and pe.edit_mode == "float_controls":
                fails.append(
                    "Root_M: silent upgrade k16_offsets → float_controls on noop export"
                )
            # Orientation contract: densify→float remake was the "different animation".
            if ob.edit_mode == "quat_k16" and oe.edit_mode == "float_controls":
                fails.append(
                    f"Root_M: orientation quat_k16 → float_controls "
                    f"(knots {ob.knot_count}→{oe.knot_count}) — densify remake not allowed"
                )
            elif (
                ob.edit_mode == "quat_k16"
                and oe.edit_mode == "quat_k16"
                and oe.knot_count >= max(100, int(ob.knot_count * 1.5) + 1)
            ):
                fails.append(
                    f"Root_M: orientation knot explosion {ob.knot_count}→{oe.knot_count} "
                    "(expected sparse quat_k16)"
                )
            elif expect_noop and ob.edit_mode != oe.edit_mode:
                fails.append(
                    f"Root_M: orientation mode {ob.edit_mode} → {oe.edit_mode} on noop"
                )
            _check_float_degree("Root_M")
            if pe.edit_mode == "k16_offsets":
                warns.append(
                    "Root_M still k16_offsets — Save can only bias ControlOffsets; "
                    "KnotsControls locomotion is not rewritten"
                )
                # KnotsControls mutation check when both k16
                if (
                    pb.edit_mode == "k16_offsets"
                    and pb.knots_controls_offset
                    and pe.knots_controls_offset
                    and expect_noop
                ):
                    nb = pb.knot_count
                    ne = pe.knot_count
                    if nb == ne and nb > 0:
                        blob_b = s0_b[
                            pb.knots_controls_offset : pb.knots_controls_offset
                            + nb * 2
                            + nb * 6
                        ]
                        blob_e = s0_e[
                            pe.knots_controls_offset : pe.knots_controls_offset
                            + ne * 2
                            + ne * 6
                        ]
                        if blob_b != blob_e:
                            fails.append(
                                "Root_M: KnotsControls mutated on expected noop "
                                "(bias-only should leave knots intact)"
                            )

    ok = len(fails) == 0
    return {"ok": ok, "fails": fails, "warns": warns, "bones": bones}


# ---------------------------------------------------------------------------
# Gate B — local series fidelity
# ---------------------------------------------------------------------------


def gate_b_local(
    anim_b: AnimMap,
    s0_b: bytes,
    anim_e: AnimMap,
    s0_e: bytes,
    *,
    local_max: float = DEFAULT_LOCAL_MAX,
    hz: float = 30.0,
) -> Dict[str, Any]:
    dur = float(anim_b.duration or anim_e.duration or 0.0)
    times = _sample_times(dur, hz=hz)
    per_bone: Dict[str, Any] = {}
    fails: List[str] = []
    warns: List[str] = []

    for name in ROOT_BONES:
        tb = _track(anim_b, name)
        te = _track(anim_e, name)
        if tb is None or te is None:
            continue
        max_dx = max_dy = max_dz = max_lat = 0.0
        at_t = 0.0
        # Root_M: also track hold vs lerp columns
        hold_vs_lerp_base = 0.0
        for t in times:
            pb = eval_position_at(s0_b, tb.position, t)
            pe = eval_position_at(s0_e, te.position, t)
            dx, dy, dz, lat = _delta_xyz(pb, pe)
            if lat > max_lat or dy > max_dy:
                at_t = t
            max_dx = max(max_dx, dx)
            max_dy = max(max_dy, dy)
            max_dz = max(max_dz, dz)
            max_lat = max(max_lat, lat)
            if name == "Root_M" and tb.position.edit_mode == "k16_offsets":
                hold = _pos_hold_k16(s0_b, tb.position)
                _, _, _, hlat = _delta_xyz(hold, pb)
                hold_vs_lerp_base = max(hold_vs_lerp_base, hlat)

        entry = {
            "max_dx": max_dx,
            "max_dy": max_dy,
            "max_dz": max_dz,
            "max_lateral": max_lat,
            "at_t": at_t,
            "eval": "lerp_or_float",
        }
        if name == "Root_M":
            entry["hold_vs_lerp_base_lateral"] = hold_vs_lerp_base
            entry["modes"] = {
                "base": tb.position.edit_mode,
                "encoded": te.position.edit_mode,
            }
            if hold_vs_lerp_base > local_max:
                warns.append(
                    f"Root_M: site inferred (hold) vs lerp lateral "
                    f"{hold_vs_lerp_base:.4f} > {local_max} — "
                    "inferred stage hides Root_M locomotion (D3K16 display lie)"
                )
        per_bone[name] = entry
        if max_lat > local_max or max_dy > local_max:
            fails.append(
                f"{name} local: lateral={max_lat:.4f} dy={max_dy:.4f} "
                f"(max {local_max}) at t~{at_t:.3f}"
            )

    return {
        "ok": len(fails) == 0,
        "fails": fails,
        "warns": warns,
        "bones": per_bone,
        "sample_count": len(times),
    }


# ---------------------------------------------------------------------------
# Gate C — world FK oracle (camera-cut predictor)
# ---------------------------------------------------------------------------


def gate_c_world(
    anim_b: AnimMap,
    s0_b: bytes,
    anim_e: AnimMap,
    s0_e: bytes,
    *,
    lateral_max: float = DEFAULT_LATERAL_MAX,
    vertical_max: float = DEFAULT_VERTICAL_MAX,
    hz: float = 5.0,
    allow_root_delta: bool = False,
) -> Dict[str, Any]:
    dur = float(anim_b.duration or anim_e.duration or 0.0)
    times = _sample_times(dur, hz=hz)
    per_bone: Dict[str, Any] = {}
    fails: List[str] = []
    warns: List[str] = []

    max_root_lat = 0.0
    max_root_dy = 0.0

    skel_b = build_skeleton(anim_b, s0_b)
    skel_e = build_skeleton(anim_e, s0_e)
    poses_b: Dict[float, Dict[str, Any]] = {}
    poses_e: Dict[float, Dict[str, Any]] = {}
    for t in times:
        wb = eval_world_pose(s0_b, anim_b, t, skel=skel_b)
        we = eval_world_pose(s0_e, anim_e, t, skel=skel_e)
        poses_b[t] = wb
        poses_e[t] = we
        by_b = {b["name"]: b for b in wb["bones"]}
        by_e = {b["name"]: b for b in we["bones"]}
        for name in FOCUS_BONES:
            if name not in by_b or name not in by_e:
                continue
            tb = by_b[name]["translation"]
            te = by_e[name]["translation"]
            dx, dy, dz, lat = _delta_xyz(tb, te)
            slot = per_bone.setdefault(
                name,
                {
                    "max_dx": 0.0,
                    "max_dy": 0.0,
                    "max_dz": 0.0,
                    "max_lateral": 0.0,
                    "at_t": 0.0,
                },
            )
            if lat > slot["max_lateral"] or dy > slot["max_dy"]:
                slot["at_t"] = t
            slot["max_dx"] = max(slot["max_dx"], dx)
            slot["max_dy"] = max(slot["max_dy"], dy)
            slot["max_dz"] = max(slot["max_dz"], dz)
            slot["max_lateral"] = max(slot["max_lateral"], lat)
            if name in ROOT_BONES:
                max_root_lat = max(max_root_lat, lat)
                max_root_dy = max(max_root_dy, dy)

    # Foot plant drift relative to Dummy_Root (reuse first FK poses)
    foot: Dict[str, Any] = {}
    for ankle in ("Ankle_L", "Ankle_R"):
        if ankle not in per_bone or "Dummy_Root" not in per_bone:
            continue
        max_rel = 0.0
        at_t = 0.0
        for t in times:
            wb = poses_b.get(t)
            we = poses_e.get(t)
            if not wb or not we:
                continue
            bb = {b["name"]: b["translation"] for b in wb["bones"]}
            ee = {b["name"]: b["translation"] for b in we["bones"]}
            if ankle not in bb or ankle not in ee or "Dummy_Root" not in bb:
                continue
            rb = (
                float(bb[ankle][0]) - float(bb["Dummy_Root"][0]),
                float(bb[ankle][2]) - float(bb["Dummy_Root"][2]),
            )
            re_ = (
                float(ee[ankle][0]) - float(ee["Dummy_Root"][0]),
                float(ee[ankle][2]) - float(ee["Dummy_Root"][2]),
            )
            d = math.sqrt((re_[0] - rb[0]) ** 2 + (re_[1] - rb[1]) ** 2)
            if d > max_rel:
                max_rel = d
                at_t = t
        foot[ankle] = {"max_rel_xz_delta": max_rel, "at_t": at_t}
        if max_rel > lateral_max * 2:
            warns.append(
                f"{ankle}: foot-plant relative XZ delta {max_rel:.4f} "
                f"(root may be OK but feet slide)"
            )

    for name in ROOT_BONES:
        slot = per_bone.get(name)
        if not slot:
            continue
        if slot["max_lateral"] > lateral_max or slot["max_dy"] > vertical_max:
            msg = (
                f"{name} world: lateral={slot['max_lateral']:.4f} "
                f"dy={slot['max_dy']:.4f} (limits {lateral_max}/{vertical_max}) "
                f"at t~{slot['at_t']:.3f}"
            )
            if allow_root_delta:
                warns.append("ALLOW: " + msg)
            else:
                fails.append(msg)

    return {
        "ok": len(fails) == 0,
        "fails": fails,
        "warns": warns,
        "bones": per_bone,
        "foot_plant": foot,
        "max_root_lateral": max_root_lat,
        "max_root_dy": max_root_dy,
        "allow_root_delta": allow_root_delta,
        "sample_count": len(times),
    }


# ---------------------------------------------------------------------------
# Gate D — site false-green trap
# ---------------------------------------------------------------------------


def gate_d_site_trap(anim_b: AnimMap, s0_b: bytes) -> Dict[str, Any]:
    warns: List[str] = []
    tr = _track(anim_b, "Root_M")
    if tr is None or tr.position.edit_mode != "k16_offsets":
        return {"ok": True, "warns": [], "hold_vs_lerp_lateral": 0.0}
    dur = float(anim_b.duration or 0.0)
    times = _sample_times(dur, hz=10.0)
    hold0 = _pos_hold_k16(s0_b, tr.position)
    max_lat = 0.0
    for t in times:
        lerp = eval_position_at(s0_b, tr.position, t)
        _, _, _, lat = _delta_xyz(hold0, lerp)
        max_lat = max(max_lat, lat)
    if max_lat > 0.01:
        warns.append(
            f"WARN: site inferred hides Root_M locomotion "
            f"(hold vs lerp lateral {max_lat:.4f}). "
            "Use eval stage root_k16 to inspect; encode still ships k16 knots."
        )
    return {"ok": True, "warns": warns, "hold_vs_lerp_lateral": max_lat}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def validate_root_export(
    base: Path,
    encoded: Path,
    *,
    dump: Optional[Path] = None,
    lateral_max: float = DEFAULT_LATERAL_MAX,
    vertical_max: float = DEFAULT_VERTICAL_MAX,
    local_max: float = DEFAULT_LOCAL_MAX,
    allow_root_delta: bool = False,
    expect_noop: bool = False,
    base_anim: Optional[Tuple[Any, AnimMap, bytes]] = None,
    encoded_anim: Optional[Tuple[Any, AnimMap, bytes]] = None,
) -> Dict[str, Any]:
    base = Path(base)
    encoded = Path(encoded)
    if base_anim is not None:
        _gb, anim_b, s0_b = base_anim
    else:
        _gb, anim_b, s0_b = load_anim(base)
    if encoded_anim is not None:
        _ge, anim_e, s0_e = encoded_anim
    else:
        _ge, anim_e, s0_e = load_anim(encoded)

    a = gate_a_format(anim_b, s0_b, anim_e, s0_e, expect_noop=expect_noop)
    b = gate_b_local(anim_b, s0_b, anim_e, s0_e, local_max=local_max)
    c = gate_c_world(
        anim_b,
        s0_b,
        anim_e,
        s0_e,
        lateral_max=lateral_max,
        vertical_max=vertical_max,
        allow_root_delta=allow_root_delta,
    )
    d = gate_d_site_trap(anim_b, s0_b)

    # Gate B is soft when allow_root_delta (intentional locomotion edit)
    b_ok = b["ok"] if not allow_root_delta else True
    if allow_root_delta and not b["ok"]:
        b = dict(b)
        b["warns"] = list(b.get("warns") or []) + [
            "ALLOW: " + f for f in (b.get("fails") or [])
        ]
        b["fails"] = []
        b["ok"] = True

    export_meta = None
    if dump:
        dump = Path(dump)
        meta_p = dump / "EXPORT_META.json"
        if meta_p.is_file():
            try:
                export_meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception as exc:
                export_meta = {"error": str(exc)}
        if export_meta is not None and not export_meta.get("space") and not export_meta.get(
            "granny_space"
        ):
            a["warns"].append("EXPORT_META.json missing space / granny_space stamp")

    fails = list(a["fails"]) + list(b.get("fails") or []) + list(c["fails"])
    warns = (
        list(a.get("warns") or [])
        + list(b.get("warns") or [])
        + list(c.get("warns") or [])
        + list(d.get("warns") or [])
    )
    ok = len(fails) == 0

    report = {
        "ok": ok,
        "base": str(base),
        "encoded": str(encoded),
        "dump": str(dump) if dump else None,
        "allow_root_delta": allow_root_delta,
        "expect_noop": expect_noop,
        "thresholds": {
            "lateral_max": lateral_max,
            "vertical_max": vertical_max,
            "local_max": local_max,
        },
        "gate_a_format": a,
        "gate_b_local": b,
        "gate_c_world": c,
        "gate_d_site_trap": d,
        "export_meta": export_meta,
        "fails": fails,
        "warns": warns,
        "max_root_lateral": c.get("max_root_lateral"),
        "max_root_dy": c.get("max_root_dy"),
        "badge": "root OK" if ok else "ROOT SHIFT",
    }
    return report


def write_export_meta(dump_dir: Path, meta: Dict[str, Any]) -> Path:
    """Merge/write EXPORT_META.json next to a dump (Gate E forensics)."""
    dump_dir = Path(dump_dir)
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


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"ROOT GATE: {'PASS' if report['ok'] else 'FAIL'}  [{report.get('badge')}]",
        f"  base:     {report['base']}",
        f"  encoded:  {report['encoded']}",
        f"  lateral:  {report.get('max_root_lateral'):.4f}  "
        f"dy: {report.get('max_root_dy'):.4f}",
    ]
    for f in report.get("fails") or []:
        lines.append(f"  FAIL  {f}")
    for w in report.get("warns") or []:
        lines.append(f"  WARN  {w}")
    c = report.get("gate_c_world") or {}
    for name, slot in (c.get("bones") or {}).items():
        if name in ROOT_BONES:
            lines.append(
                f"  world {name}: lat={slot['max_lateral']:.4f} "
                f"dy={slot['max_dy']:.4f} @t={slot['at_t']:.3f}"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate root-motion export vs original")
    ap.add_argument("--base", required=True, help="Original source GR2")
    ap.add_argument("--encoded", required=True, help="Encoded / candidate GR2")
    ap.add_argument("--dump", default=None, help="Optional OUT_FILES dump for EXPORT_META")
    ap.add_argument("--lateral-max", type=float, default=DEFAULT_LATERAL_MAX)
    ap.add_argument("--vertical-max", type=float, default=DEFAULT_VERTICAL_MAX)
    ap.add_argument("--local-max", type=float, default=DEFAULT_LOCAL_MAX)
    ap.add_argument(
        "--allow-root-delta",
        action="store_true",
        help="Intentional locomotion edit — report but do not FAIL on root deltas",
    )
    ap.add_argument(
        "--expect-noop",
        action="store_true",
        help="Stricter: fail on Root_M float upgrade / KnotsControls mutation",
    )
    ap.add_argument("--json", action="store_true", help="Print full JSON report")
    ap.add_argument(
        "--write-meta",
        action="store_true",
        help="Write gate scores into dump/EXPORT_META.json",
    )
    args = ap.parse_args(argv)

    report = validate_root_export(
        Path(args.base),
        Path(args.encoded),
        dump=Path(args.dump) if args.dump else None,
        lateral_max=args.lateral_max,
        vertical_max=args.vertical_max,
        local_max=args.local_max,
        allow_root_delta=args.allow_root_delta,
        expect_noop=args.expect_noop,
    )
    if args.write_meta and args.dump:
        write_export_meta(
            Path(args.dump),
            {
                "root_gate": {
                    "ok": report["ok"],
                    "badge": report["badge"],
                    "max_root_lateral": report.get("max_root_lateral"),
                    "max_root_dy": report.get("max_root_dy"),
                    "fails": report.get("fails"),
                    "warns": report.get("warns"),
                }
            },
        )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
