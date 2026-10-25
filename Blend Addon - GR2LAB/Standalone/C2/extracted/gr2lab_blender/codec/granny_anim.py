"""Map and patch Granny TransformTrack position curves inside GR2 section bins.

Karlach / BG3 layout (tag 0x80000039, 64-bit pointers):
  Section 1 holds TransformTrack array (stride 60; start offset varies per clip).
  Section 0 holds string names + CurveData payloads.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bitknit import decompress_bitknit, load_granny2
from gr2_codec import decode_sections, load_gr2
from gr2_format import COMP_BITKNIT, COMP_NONE, GR2File

# CurveFormat enum (LSLib)
FMT_DA_K32F_C32F = 1
FMT_DA_IDENTITY = 2
FMT_D3_CONSTANT32F = 4
FMT_D4_CONSTANT32F = 5
FMT_D4N_K16U_C15U = 8
FMT_D4N_K8U_C7U = 9
FMT_D3_K16U_C16U = 10
FMT_D3_K8U_C8U = 11
FMT_D3_I1_K16U_C16U = 17
FMT_D3_I1_K8U_C8U = 18

# Granny CurveData.Degree: 1 = linear through knot samples, 2+ = B-spline through
# CONTROLS (not samples). Vanilla BG3 curves are degree-2 *fitted* controls.
# Blender bake / site keyframe series must be written as degree 1 — leaving
# degree 2 makes LSLib/game overshoot between keys (missing mesh, T-pose flashes).
CURVE_DEGREE_LINEAR = 1
CURVE_DEGREE_BSPLINE = 2

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]
IDENTITY_SCALE = [1.0, 1.0, 1.0]

TRACK_STRIDE = 60
# Observed for this Karlach file: first track name reloc at S1@164
DEFAULT_TRACKS_OFFSET = 164
DEFAULT_TRACK_COUNT = 107

# TransformTrack field offsets (64-bit, no pad after Flags)
OFF_NAME = 0
OFF_FLAGS = 8
OFF_ORIENT_TYPE = 12
OFF_ORIENT_OBJ = 20
OFF_POS_TYPE = 28
OFF_POS_OBJ = 36
OFF_SCALE_TYPE = 44
OFF_SCALE_OBJ = 52


@dataclass
class CurveMap:
    format: int
    degree: int
    object_offset: int  # in section 0
    object_section: int = 0
    # Value bytes that can be patched for position edits
    values_offset: int = 0  # absolute in section 0
    values_bytes: int = 0
    knot_count: int = 0
    # For D3K16uC16u: ControlOffsets live here (3 floats)
    control_offsets_offset: Optional[int] = None
    control_scales_offset: Optional[int] = None
    knots_controls_offset: Optional[int] = None  # start of u16 array
    editable: bool = False
    edit_mode: str = "none"  # constant | float_controls | k16_offsets | none
    # k16_offsets also covers D3I1K16/K8 (single-control diagonal) — edits adjust ControlOffsets.
    note: str = ""


@dataclass
class TrackMap:
    index: int
    name: str
    name_offset: int
    track_offset: int  # in section 1
    flags: int
    position: CurveMap
    orientation: CurveMap = field(default_factory=lambda: CurveMap(format=-1, degree=0, object_offset=-1))
    scale: CurveMap = field(default_factory=lambda: CurveMap(format=-1, degree=0, object_offset=-1))
    orientation_format: int = -1
    scale_format: int = -1


@dataclass
class AnimMap:
    source: str
    duration: float
    track_count: int
    tracks_offset: int
    tracks_section: int
    tracks: List[TrackMap] = field(default_factory=list)
    bones: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "duration": self.duration,
            "track_count": self.track_count,
            "tracks_offset": self.tracks_offset,
            "tracks_section": self.tracks_section,
            "bones": self.bones,
            "tracks": [
                {
                    "index": t.index,
                    "name": t.name,
                    "name_offset": t.name_offset,
                    "track_offset": t.track_offset,
                    "flags": t.flags,
                    "orientation_format": t.orientation_format,
                    "scale_format": t.scale_format,
                    "position": asdict(t.position),
                    "orientation": asdict(t.orientation),
                    "scale": asdict(t.scale),
                }
                for t in self.tracks
            ],
        }


def _decomp_fixups(file_data: bytes, off: int, count: int, esize: int) -> bytes:
    if count <= 0:
        return b""
    csize = struct.unpack_from("<I", file_data, off)[0]
    blob = file_data[off + 4 : off + 4 + csize]
    return decompress_bitknit(blob, count * esize)


def extract_section_fixups(gr2: GR2File, file_data: Optional[bytes] = None) -> None:
    """Attach decompressed reloc/mm tables to each section (for preserve-on-edit)."""
    data = file_data if file_data is not None else gr2.original_bytes
    if data is None:
        raise ValueError("Need original file bytes to extract fixups")
    load_granny2()
    for sec in gr2.sections:
        h = sec.header
        if h.compression == COMP_BITKNIT and h.num_relocations:
            sec.relocations = _decomp_fixups(
                data, h.relocations_offset, h.num_relocations, 12
            )
        elif h.num_relocations and h.compression == COMP_NONE:
            end = h.relocations_offset + h.num_relocations * 12
            sec.relocations = data[h.relocations_offset:end]
        else:
            sec.relocations = b""

        if h.compression == COMP_BITKNIT and h.num_mixed_marshalling_data:
            sec.mixed_marshalling = _decomp_fixups(
                data, h.mixed_marshalling_data_offset, h.num_mixed_marshalling_data, 16
            )
        elif h.num_mixed_marshalling_data and h.compression == COMP_NONE:
            end = h.mixed_marshalling_data_offset + h.num_mixed_marshalling_data * 16
            sec.mixed_marshalling = data[h.mixed_marshalling_data_offset:end]
        else:
            sec.mixed_marshalling = b""


def _reloc_map(reloc_blob: bytes) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    for i in range(0, len(reloc_blob), 12):
        off, sec, toff = struct.unpack_from("<III", reloc_blob, i)
        out[off] = (sec, toff)
    return out


def _retarget_reloc(
    reloc_blob: bytearray,
    *,
    pointer_off: int,
    old_toff: int,
    new_toff: int,
    section: int = 0,
) -> bool:
    """Update one reloc entry's target offset. Returns True if patched."""
    for i in range(0, len(reloc_blob), 12):
        off, sec, toff = struct.unpack_from("<III", reloc_blob, i)
        if off == pointer_off and sec == section and toff == old_toff:
            struct.pack_into("<III", reloc_blob, i, off, sec, int(new_toff))
            return True
        # Fallback: match by pointer slot alone (same section)
        if off == pointer_off and sec == section:
            struct.pack_into("<III", reloc_blob, i, off, sec, int(new_toff))
            return True
    return False


def _append_reloc(
    reloc_blob: bytearray,
    *,
    pointer_off: int,
    target_off: int,
    section: int = 0,
) -> None:
    """Append a new 12-byte reloc entry (pointer_off → section:target_off)."""
    # Replace existing slot if present
    for i in range(0, len(reloc_blob), 12):
        off, sec, _toff = struct.unpack_from("<III", reloc_blob, i)
        if off == pointer_off and sec == section:
            struct.pack_into("<III", reloc_blob, i, pointer_off, section, int(target_off))
            return
    reloc_blob.extend(struct.pack("<III", int(pointer_off), int(section), int(target_off)))


def _retarget_s1_curve_ptr(
    reloc1: bytearray,
    *,
    track_offset: int,
    field_off: int,
    new_s0_off: int,
    old_s0_off: Optional[int] = None,
) -> bool:
    """Retarget TransformTrack curve object pointer reloc in section 1.

    S1 pointer bytes are typically 0 (marshalling); only the reloc toff matters.
    """
    pointer_off = int(track_offset) + int(field_off)
    if old_s0_off is not None:
        return _retarget_reloc(
            reloc1,
            pointer_off=pointer_off,
            old_toff=int(old_s0_off),
            new_toff=int(new_s0_off),
            section=0,
        )
    return _retarget_reloc(
        reloc1,
        pointer_off=pointer_off,
        old_toff=0,
        new_toff=int(new_s0_off),
        section=0,
    )


def _reloc_map(reloc_blob: bytes) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    for i in range(0, len(reloc_blob), 12):
        off, sec, toff = struct.unpack_from("<III", reloc_blob, i)
        out[off] = (sec, toff)
    return out


def _find_type_ref_for_curve_format(
    anim: AnimMap,
    reloc1: bytes,
    *,
    channel: str,
    want_format: int,
    exclude_bone: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """Find an existing TransformTrack type-pointer reloc for curve format ``want_format``.

    Granny stores CurveData *type* (usually in S4) separately from the curve *object*
    (S0). Upgrading object bytes without retargeting the type pointer makes LSLib
    deserialize the new payload with the old schema (e.g. float bytes as D3K16uC16u
    → null KnotsControls → NumKnots NRE).
    """
    field = OFF_POS_TYPE if channel == "position" else OFF_ORIENT_TYPE
    rmap = _reloc_map(bytes(reloc1))
    for t in anim.tracks:
        if exclude_bone and t.name == exclude_bone:
            continue
        cm = t.position if channel == "position" else t.orientation
        if cm.format != want_format:
            continue
        ref = rmap.get(int(t.track_offset) + int(field))
        if ref is not None:
            return ref
    return None


def _retarget_s1_type_ptr(
    reloc1: bytearray,
    *,
    track_offset: int,
    field_off: int,
    new_sec: int,
    new_toff: int,
) -> bool:
    """Retarget TransformTrack curve-type pointer (usually → section 4)."""
    pointer_off = int(track_offset) + int(field_off)
    for i in range(0, len(reloc1), 12):
        off, _sec, _toff = struct.unpack_from("<III", reloc1, i)
        if off == pointer_off:
            struct.pack_into(
                "<III", reloc1, i, pointer_off, int(new_sec), int(new_toff)
            )
            return True
    reloc1.extend(
        struct.pack("<III", pointer_off, int(new_sec), int(new_toff))
    )
    return True


def _retarget_curve_type_for_format(
    anim: AnimMap,
    reloc1: bytearray,
    *,
    track: TrackMap,
    channel: str,
    want_format: int,
) -> Optional[Tuple[int, int]]:
    """Point this track's type field at a type-def used by another track of want_format."""
    ref = _find_type_ref_for_curve_format(
        anim,
        reloc1,
        channel=channel,
        want_format=want_format,
        exclude_bone=track.name,
    )
    if ref is None:
        return None
    field = OFF_POS_TYPE if channel == "position" else OFF_ORIENT_TYPE
    _retarget_s1_type_ptr(
        reloc1,
        track_offset=track.track_offset,
        field_off=field,
        new_sec=ref[0],
        new_toff=ref[1],
    )
    return ref


def _align4(n: int) -> int:
    return (n + 3) & ~3


def _s0_append_aligned(s0: bytearray, blob: bytes) -> int:
    """Append blob at 4-byte aligned EOF; return start offset."""
    off = _align4(len(s0))
    if off > len(s0):
        s0.extend(b"\x00" * (off - len(s0)))
    s0.extend(blob)
    return off


def _read_cstr(buf: bytes, off: int) -> str:
    if off <= 0 or off >= len(buf):
        return ""
    end = buf.find(b"\x00", off)
    if end < 0:
        end = min(off + 64, len(buf))
    return buf[off:end].decode("ascii", "replace")


def _one_over_knot_scale(trunc: int) -> float:
    raw = struct.pack("<I", (trunc & 0xFFFF) << 16)
    return struct.unpack("<f", raw)[0]


def _map_position_curve(s0: bytes, obj_off: int, s0_reloc: Dict[int, Tuple[int, int]]) -> CurveMap:
    if obj_off < 0 or obj_off + 4 > len(s0):
        return CurveMap(format=-1, degree=0, object_offset=obj_off, note="oob")
    fmt, degree = s0[obj_off], s0[obj_off + 1]
    cm = CurveMap(format=fmt, degree=degree, object_offset=obj_off)

    if fmt == FMT_D3_CONSTANT32F:
        cm.values_offset = obj_off + 4
        cm.values_bytes = 12
        cm.knot_count = 1
        cm.editable = True
        cm.edit_mode = "constant"
        return cm

    if fmt == FMT_DA_IDENTITY:
        cm.values_offset = obj_off
        cm.values_bytes = 4
        cm.knot_count = 1
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "identity — no position samples"
        return cm

    if fmt == FMT_D3_K16U_C16U:
        # header(2) + OneOverKnotScaleTrunc(2) + ControlScales[3] + ControlOffsets[3]
        # + KnotsControls{count:u32, ptr:u64 reloc @ +32}  (64-bit BG3 layout)
        # KnotsControls blob: knot_u16[n] then control_u16[n*3] (same as quat_k16),
        # where count is number of u16s (knot_count * 4).
        cm.control_scales_offset = obj_off + 4
        cm.control_offsets_offset = obj_off + 16
        cm.values_offset = obj_off + 16
        cm.values_bytes = 12
        u16_count = struct.unpack_from("<I", s0, obj_off + 28)[0]
        kc = s0_reloc.get(obj_off + 32)
        if kc and kc[0] == 0 and u16_count >= 4 and (u16_count % 4) == 0:
            cm.knots_controls_offset = kc[1]
            cm.knot_count = u16_count // 4
            cm.note = (
                f"D3K16uC16u — {cm.knot_count} knots "
                "(display=dequantized; edits still adjust ControlOffsets bias)"
            )
        else:
            # Fallback: bias only (legacy / unresolved reloc)
            cm.knot_count = 0
            cm.note = "D3K16uC16u — KnotsControls unresolved; ControlOffsets bias only"
        cm.editable = True
        cm.edit_mode = "k16_offsets"
        return cm

    if fmt == FMT_D3_K8U_C8U:
        # Same header footprint as D3K16uC16u, but KnotsControls are u8:
        # knot_u8[n] then control_u8[n*3]; Count == n*4.
        # Seen on DGB_M Nested Kiss Spine2_M / Chest_M — was unsupported → [0,0,0] FK collapse.
        cm.control_scales_offset = obj_off + 4
        cm.control_offsets_offset = obj_off + 16
        cm.values_offset = obj_off + 16
        cm.values_bytes = 12
        u8_count = struct.unpack_from("<I", s0, obj_off + 28)[0]
        kc = s0_reloc.get(obj_off + 32)
        if kc and kc[0] == 0 and u8_count >= 4 and (u8_count % 4) == 0:
            cm.knots_controls_offset = kc[1]
            cm.knot_count = u8_count // 4
            cm.note = (
                f"D3K8uC8u — {cm.knot_count} knots "
                "(display=dequantized; edits still adjust ControlOffsets bias)"
            )
        else:
            cm.knot_count = 0
            cm.note = "D3K8uC8u — KnotsControls unresolved; ControlOffsets bias only"
        cm.editable = True
        cm.edit_mode = "k16_offsets"
        return cm

    if fmt in (FMT_D3_I1_K16U_C16U, FMT_D3_I1_K8U_C8U):
        # Same header as D3K16uC16u, but one scalar control per knot expanded to XYZ
        # via ControlScales/Offsets (LSLib D3I1K16uC16u / D3I1K8uC8u GetPoints).
        # KnotsControls: knot[n] then control[n]; NumKnots = Count/2.
        # Fmt 17 = u16 elements; fmt 18 = u8 elements.
        cm.control_scales_offset = obj_off + 4
        cm.control_offsets_offset = obj_off + 16
        cm.values_offset = obj_off + 16
        cm.values_bytes = 12
        elem_count = struct.unpack_from("<I", s0, obj_off + 28)[0]
        kc = s0_reloc.get(obj_off + 32)
        label = "D3I1K16uC16u" if fmt == FMT_D3_I1_K16U_C16U else "D3I1K8uC8u"
        if kc and kc[0] == 0 and elem_count >= 2 and (elem_count % 2) == 0:
            cm.knots_controls_offset = kc[1]
            cm.knot_count = elem_count // 2
            cm.note = (
                f"{label} — {cm.knot_count} knots "
                "(display=dequantized; edits adjust ControlOffsets bias)"
            )
        else:
            cm.knot_count = 0
            cm.note = f"{label} — KnotsControls unresolved; ControlOffsets bias only"
        cm.editable = True
        cm.edit_mode = "k16_offsets"
        return cm

    if fmt == FMT_DA_K32F_C32F:
        # header(2)+pad(2) + Knots{count:u32, ptr:u64} + Controls{count:u32, ptr:u64}
        # Relocs hit the pointer fields at +8 and +20.
        knots_count = struct.unpack_from("<I", s0, obj_off + 4)[0]
        kref = s0_reloc.get(obj_off + 8)
        cref = s0_reloc.get(obj_off + 20)
        if kref and cref and kref[0] == 0 and cref[0] == 0 and knots_count > 0:
            cm.knot_count = knots_count
            cm.knots_controls_offset = kref[1]
            cm.values_offset = cref[1]
            # Capacity = allocated control floats / 3; may exceed knot_count after deletes.
            cm.values_bytes = knots_count * 12
            cm.editable = True
            cm.edit_mode = "float_controls"
            return cm
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "DaK32fC32f — could not resolve Knots/Controls"
        return cm

    cm.editable = False
    cm.edit_mode = "none"
    cm.note = f"unsupported format {fmt}"
    return cm


def _map_orientation_curve(s0: bytes, obj_off: int, s0_reloc: Dict[int, Tuple[int, int]]) -> CurveMap:
    if obj_off < 0 or obj_off + 4 > len(s0):
        return CurveMap(format=-1, degree=0, object_offset=obj_off, note="oob")
    fmt, degree = s0[obj_off], s0[obj_off + 1]
    cm = CurveMap(format=fmt, degree=degree, object_offset=obj_off)

    if fmt == FMT_DA_IDENTITY:
        cm.knot_count = 1
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "identity orientation"
        return cm

    if fmt == FMT_D4_CONSTANT32F:
        cm.values_offset = obj_off + 4
        cm.values_bytes = 16
        cm.knot_count = 1
        cm.editable = True
        cm.edit_mode = "constant"
        return cm

    if fmt == FMT_D4N_K16U_C15U:
        # header(2) + ScaleOffsetTableEntries(u16) + OneOverKnotScale(f32)
        # + KnotsControls count(u32) + ptr(u64 reloc @ +12)
        cm.control_scales_offset = obj_off + 2  # selector u16
        cm.values_offset = obj_off + 4  # OneOverKnotScale
        kc = s0_reloc.get(obj_off + 12)
        if kc and kc[0] == 0:
            cm.knots_controls_offset = kc[1]
            u16_count = struct.unpack_from("<I", s0, obj_off + 8)[0]
            if u16_count >= 4 and u16_count % 4 == 0 and u16_count // 4 <= 8192:
                cm.knot_count = u16_count // 4
            else:
                targets = sorted(
                    {toff for sec, toff in s0_reloc.values() if sec == 0 and toff > kc[1]}
                )
                end = targets[0] if targets else len(s0)
                cm.knot_count = max(0, (end - kc[1]) // 8)
            cm.values_bytes = cm.knot_count * 8
            cm.editable = True
            cm.edit_mode = "quat_k16"
            cm.note = "quantized orientation — per-key rewrite"
            return cm
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "D4nK16uC15u — could not resolve KnotsControls"
        return cm

    if fmt == FMT_D4N_K8U_C7U:
        # Same header as D4nK16uC15u; KnotsControls are u8 (knot[n] + ctrl[n*3]).
        cm.control_scales_offset = obj_off + 2
        cm.values_offset = obj_off + 4
        kc = s0_reloc.get(obj_off + 12)
        if kc and kc[0] == 0:
            cm.knots_controls_offset = kc[1]
            u8_count = struct.unpack_from("<I", s0, obj_off + 8)[0]
            if u8_count >= 4 and u8_count % 4 == 0 and u8_count // 4 <= 8192:
                cm.knot_count = u8_count // 4
            else:
                targets = sorted(
                    {toff for sec, toff in s0_reloc.values() if sec == 0 and toff > kc[1]}
                )
                end = targets[0] if targets else len(s0)
                cm.knot_count = max(0, (end - kc[1]) // 4)
            cm.values_bytes = cm.knot_count * 4
            cm.editable = False  # display/eval only for now
            cm.edit_mode = "quat_k8"
            cm.note = f"D4nK8uC7u — {cm.knot_count} knots (display dequant)"
            return cm
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "D4nK8uC7u — could not resolve KnotsControls"
        return cm

    if fmt == FMT_DA_K32F_C32F:
        knots_count = struct.unpack_from("<I", s0, obj_off + 4)[0]
        kref = s0_reloc.get(obj_off + 8)
        cref = s0_reloc.get(obj_off + 20)
        if kref and cref and kref[0] == 0 and cref[0] == 0 and knots_count > 0:
            cm.knot_count = knots_count
            cm.knots_controls_offset = kref[1]
            cm.values_offset = cref[1]
            cm.values_bytes = knots_count * 16
            cm.editable = True
            cm.edit_mode = "float_controls"
            return cm
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "DaK32fC32f orient — unresolved"
        return cm

    cm.editable = False
    cm.edit_mode = "none"
    cm.note = f"unsupported orientation format {fmt}"
    return cm


def _map_scale_curve(s0: bytes, obj_off: int, s0_reloc: Dict[int, Tuple[int, int]]) -> CurveMap:
    """ScaleShear is usually DaIdentity (3×3). Map constant / float 3-vector when present."""
    if obj_off < 0 or obj_off + 4 > len(s0):
        return CurveMap(format=-1, degree=0, object_offset=obj_off, note="oob")
    fmt, degree = s0[obj_off], s0[obj_off + 1]
    cm = CurveMap(format=fmt, degree=degree, object_offset=obj_off)

    if fmt == FMT_DA_IDENTITY:
        cm.knot_count = 1
        cm.editable = False
        cm.edit_mode = "none"
        cm.note = "identity scale"
        return cm

    if fmt == FMT_D3_CONSTANT32F:
        cm.values_offset = obj_off + 4
        cm.values_bytes = 12
        cm.knot_count = 1
        cm.editable = True
        cm.edit_mode = "constant"
        return cm

    if fmt == FMT_DA_K32F_C32F:
        knots_count = struct.unpack_from("<I", s0, obj_off + 4)[0]
        kref = s0_reloc.get(obj_off + 8)
        cref = s0_reloc.get(obj_off + 20)
        if kref and cref and kref[0] == 0 and cref[0] == 0 and knots_count > 0:
            cm.knot_count = knots_count
            cm.knots_controls_offset = kref[1]
            cm.values_offset = cref[1]
            cm.values_bytes = knots_count * 12
            cm.editable = True
            cm.edit_mode = "float_controls"
            return cm

    # D3K16uC16u / D3K8uC8u / D3I1* scale rare but same layout as position
    if fmt in (
        FMT_D3_K16U_C16U,
        FMT_D3_K8U_C8U,
        FMT_D3_I1_K16U_C16U,
        FMT_D3_I1_K8U_C8U,
    ):
        return _map_position_curve(s0, obj_off, s0_reloc)

    cm.editable = False
    cm.edit_mode = "none"
    cm.note = f"unsupported scale format {fmt}"
    return cm


def _looks_like_bone_name(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 72:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
    )
    return all(c in allowed for c in name)


def _score_tracks_layout_at(
    s0: bytes,
    s1: bytes,
    s0_reloc: Dict[int, Tuple[int, int]],
    s1_reloc: Dict[int, Tuple[int, int]],
    start: int,
    *,
    max_tracks: int = 150,
) -> Tuple[int, int]:
    """Return (named_count, curve_count) for a candidate TransformTrack array base."""
    named = 0
    curves = 0
    for _i in range(max_tracks):
        base = start + _i * TRACK_STRIDE
        if base + TRACK_STRIDE > len(s1):
            break
        name_ref = s1_reloc.get(base + OFF_NAME)
        if not (name_ref and name_ref[0] == 0):
            break
        name = _read_cstr(s0, name_ref[1])
        if not _looks_like_bone_name(name):
            break
        named += 1
        has_curve = False
        pos_ref = s1_reloc.get(base + OFF_POS_OBJ)
        if pos_ref and pos_ref[0] == 0:
            if _map_position_curve(s0, pos_ref[1], s0_reloc).format >= 0:
                has_curve = True
        ori_ref = s1_reloc.get(base + OFF_ORIENT_OBJ)
        if ori_ref and ori_ref[0] == 0:
            if _map_orientation_curve(s0, ori_ref[1], s0_reloc).format >= 0:
                has_curve = True
        if has_curve:
            curves += 1
    return named, curves


def _detect_transform_tracks_layout(gr2: GR2File) -> Tuple[int, int]:
    """Locate TransformTrack array in section 1 (offset differs across BG3 clips)."""
    s1 = gr2.sections[1].uncompressed or b""
    if len(s1) < TRACK_STRIDE * 3:
        return DEFAULT_TRACKS_OFFSET, DEFAULT_TRACK_COUNT

    s0 = gr2.sections[0].uncompressed or b""
    s0_reloc = _reloc_map(gr2.sections[0].relocations or b"")
    s1_reloc = _reloc_map(gr2.sections[1].relocations or b"")

    best_off = DEFAULT_TRACKS_OFFSET
    best_named = 0
    best_curves = 0
    scan_limit = min(len(s1) - TRACK_STRIDE * 3, 8192)

    for start in range(0, scan_limit, 4):
        named, curves = _score_tracks_layout_at(
            s0, s1, s0_reloc, s1_reloc, start
        )
        if curves > best_curves or (curves == best_curves and named > best_named):
            best_curves = curves
            best_named = named
            best_off = start

    if best_curves < 5:
        return DEFAULT_TRACKS_OFFSET, DEFAULT_TRACK_COUNT

    return best_off, max(best_named, DEFAULT_TRACK_COUNT)


def _find_duration(s0: bytes, tracks: List[TrackMap]) -> float:
    # Prefer last knot of any float position curve (clip length).
    best = 0.0
    for t in tracks:
        cm = t.position
        if cm.edit_mode == "float_controls" and cm.knots_controls_offset and cm.knot_count > 0:
            last = struct.unpack_from(
                "<f", s0, cm.knots_controls_offset + (cm.knot_count - 1) * 4
            )[0]
            if 0.1 < last < 600.0:
                best = max(best, last)
        if (
            cm.edit_mode == "k16_offsets"
            and cm.knots_controls_offset is not None
            and cm.knot_count > 0
        ):
            oos = _d3k16_one_over_knot_scale(s0, cm)
            if oos:
                n = cm.knot_count
                if cm.format in (FMT_D3_I1_K8U_C8U, FMT_D3_K8U_C8U):
                    last_u = s0[cm.knots_controls_offset + n - 1]
                else:
                    last_u = struct.unpack_from(
                        "<H", s0, cm.knots_controls_offset + (n - 1) * 2
                    )[0]
                last = float(last_u) / oos
                if 0.1 < last < 600.0:
                    best = max(best, last)
        oc = t.orientation
        if oc.edit_mode == "quat_k16" and oc.knots_controls_offset and oc.knot_count > 0:
            oos = struct.unpack_from("<f", s0, oc.values_offset)[0]
            if oos:
                last_u = struct.unpack_from(
                    "<H", s0, oc.knots_controls_offset + (oc.knot_count - 1) * 2
                )[0]
                last = last_u / oos
                if 0.1 < last < 600.0:
                    best = max(best, last)
        if oc.edit_mode == "quat_k8" and oc.knots_controls_offset and oc.knot_count > 0:
            oos = struct.unpack_from("<f", s0, oc.values_offset)[0]
            if oos:
                last_u = s0[oc.knots_controls_offset + oc.knot_count - 1]
                last = float(last_u) / oos
                if 0.1 < last < 600.0:
                    best = max(best, last)
    if best > 0:
        return best
    return 0.0


def build_anim_map(
    gr2: GR2File,
    *,
    source: str = "",
    tracks_offset: Optional[int] = None,
    track_count: Optional[int] = None,
) -> AnimMap:
    if any(s.uncompressed is None for s in gr2.sections):
        decode_sections(gr2, decompress=True)
    if not getattr(gr2.sections[1], "relocations", None):
        extract_section_fixups(gr2)

    if tracks_offset is None or track_count is None:
        det_off, det_count = _detect_transform_tracks_layout(gr2)
        if tracks_offset is None:
            tracks_offset = det_off
        if track_count is None:
            track_count = det_count

    s0 = gr2.sections[0].uncompressed or b""
    s1 = gr2.sections[1].uncompressed or b""
    s0_reloc = _reloc_map(gr2.sections[0].relocations or b"")
    s1_reloc = _reloc_map(gr2.sections[1].relocations or b"")

    tracks: List[TrackMap] = []
    for i in range(track_count):
        base = tracks_offset + i * TRACK_STRIDE
        if base + TRACK_STRIDE > len(s1):
            break
        name_ref = s1_reloc.get(base + OFF_NAME)
        pos_ref = s1_reloc.get(base + OFF_POS_OBJ)
        ori_ref = s1_reloc.get(base + OFF_ORIENT_OBJ)
        scl_ref = s1_reloc.get(base + OFF_SCALE_OBJ)
        name = ""
        name_off = 0
        if name_ref and name_ref[0] == 0:
            name_off = name_ref[1]
            name = _read_cstr(s0, name_off)
        flags = struct.unpack_from("<I", s1, base + OFF_FLAGS)[0]

        if pos_ref and pos_ref[0] == 0:
            pos = _map_position_curve(s0, pos_ref[1], s0_reloc)
        else:
            pos = CurveMap(format=-1, degree=0, object_offset=-1, note="missing pos")

        if ori_ref and ori_ref[0] == 0:
            ori = _map_orientation_curve(s0, ori_ref[1], s0_reloc)
        else:
            ori = CurveMap(format=-1, degree=0, object_offset=-1, note="missing orient")

        if scl_ref and scl_ref[0] == 0:
            scl = _map_scale_curve(s0, scl_ref[1], s0_reloc)
        else:
            scl = CurveMap(format=-1, degree=0, object_offset=-1, note="missing scale")

        tracks.append(
            TrackMap(
                index=i,
                name=name,
                name_offset=name_off,
                track_offset=base,
                flags=flags,
                position=pos,
                orientation=ori,
                scale=scl,
                orientation_format=ori.format,
                scale_format=scl.format,
            )
        )

    duration = _find_duration(s0, tracks)
    bones = [t.name for t in tracks if t.name]
    return AnimMap(
        source=source,
        duration=duration,
        track_count=len(tracks),
        tracks_offset=tracks_offset,
        tracks_section=1,
        tracks=tracks,
        bones=bones,
    )


def write_anim_map(anim: AnimMap, path: Path) -> None:
    path.write_text(
        json.dumps(anim.to_dict(), separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def load_anim_map(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_bones(anim: AnimMap | Dict[str, Any]) -> List[str]:
    if isinstance(anim, AnimMap):
        return list(anim.bones)
    return list(anim.get("bones") or [])


def _track_by_name(anim: AnimMap, bone: str) -> TrackMap:
    for t in anim.tracks:
        if t.name == bone:
            return t
    raise KeyError(f"Bone not found: {bone}")


def _times_for_constant(duration: float) -> List[float]:
    return [0.0]


def _knot_times_dak32(s0: bytes, cm: CurveMap) -> List[float]:
    if not cm.knots_controls_offset or cm.knot_count <= 0:
        return []
    return list(struct.unpack_from(f"<{cm.knot_count}f", s0, cm.knots_controls_offset))


def _knot_times_k16(s0: bytes, cm: CurveMap) -> List[float]:
    if cm.knots_controls_offset is None:
        return []
    # Need knot count: KnotsControls length / 4 (knot + 3 controls per sample)
    # Infer from ControlOffsets edit path — for time filtering we need knots.
    # Estimate: scan until we leave reasonable u16 range using next reloc after pointer.
    # Fallback: treat as whole clip.
    return []


def patch_position(
    section0: bytearray,
    anim: AnimMap,
    bone: str,
    op: str,
    xyz: Sequence[float],
    *,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
) -> Dict[str, Any]:
    """Patch position samples in-place in section0. Returns edit record."""
    if op not in ("offset", "set"):
        raise ValueError("op must be 'offset' or 'set'")
    if len(xyz) != 3:
        raise ValueError("xyz must have 3 floats")
    track = _track_by_name(anim, bone)
    cm = track.position
    if not cm.editable:
        raise ValueError(f"Bone {bone} position not editable ({cm.note or cm.format})")

    dx, dy, dz = float(xyz[0]), float(xyz[1]), float(xyz[2])
    touched = 0

    if cm.edit_mode == "constant":
        x, y, z = struct.unpack_from("<3f", section0, cm.values_offset)
        if op == "offset":
            x, y, z = x + dx, y + dy, z + dz
        else:
            x, y, z = dx, dy, dz
        # time range ignored for constants (single sample)
        if t0 is not None or t1 is not None:
            # still apply — constant covers whole clip
            pass
        struct.pack_into("<3f", section0, cm.values_offset, x, y, z)
        touched = 1

    elif cm.edit_mode == "k16_offsets":
        # Offset: add delta to ControlOffsets (affects all knots).
        # Set: only supported as setting ControlOffsets to xyz (approx absolute bias).
        assert cm.control_offsets_offset is not None
        ox, oy, oz = struct.unpack_from("<3f", section0, cm.control_offsets_offset)
        if op == "offset":
            ox, oy, oz = ox + dx, oy + dy, oz + dz
        else:
            ox, oy, oz = dx, dy, dz
        struct.pack_into("<3f", section0, cm.control_offsets_offset, ox, oy, oz)
        touched = 1

    elif cm.edit_mode == "float_controls":
        n = cm.knot_count
        times = _knot_times_dak32(section0, cm)
        for i in range(n):
            t = times[i] if i < len(times) else 0.0
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
            off = cm.values_offset + i * 12
            x, y, z = struct.unpack_from("<3f", section0, off)
            if op == "offset":
                x, y, z = x + dx, y + dy, z + dz
            else:
                x, y, z = dx, dy, dz
            struct.pack_into("<3f", section0, off, x, y, z)
            touched += 1
    else:
        raise ValueError(f"Unsupported edit_mode {cm.edit_mode}")

    return {
        "bone": bone,
        "op": op,
        "xyz": [dx, dy, dz],
        "t0": t0,
        "t1": t1,
        "samples_touched": touched,
        "format": cm.format,
        "edit_mode": cm.edit_mode,
        "values_offset": cm.values_offset,
    }


def apply_edit_to_dump(
    src_dump: Path,
    *,
    bone: str,
    op: str,
    xyz: Sequence[float],
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    out_dump: Optional[Path] = None,
    base_gr2: Optional[Path] = None,
    overwrite: bool = True,
    snapshot: bool = True,
) -> Dict[str, Any]:
    """Patch position keys and write dump (overwrites src_dump by default)."""
    from paths import make_run_dirs

    src_dump = Path(src_dump)
    s0_path = src_dump / "section_0.bin"
    if not s0_path.is_file():
        raise FileNotFoundError(f"Missing {s0_path}")

    # Resolve base GR2 for mapping
    if base_gr2 is None:
        src_txt = src_dump / "SOURCE.txt"
        name = None
        if src_txt.is_file():
            for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("name="):
                    name = line.split("=", 1)[1].strip()
                    break
        from paths import resolve_input_gr2

        base_gr2 = resolve_input_gr2(name or "")

    gr2 = load_gr2(base_gr2)
    decode_sections(gr2, True)
    extract_section_fixups(gr2)
    anim = build_anim_map(gr2, source=str(base_gr2))

    section0 = bytearray(s0_path.read_bytes())
    # Prefer live section from GR2 if dump matches size
    live = gr2.sections[0].uncompressed or b""
    if len(section0) != len(live):
        raise ValueError(
            f"section_0 size mismatch dump={len(section0)} live={len(live)}"
        )

    edit = patch_position(section0, anim, bone, op, xyz, t0=t0, t1=t1)

    if out_dump is not None:
        out_dump = Path(out_dump)
        out_dump.mkdir(parents=True, exist_ok=True)
        run_id = out_dump.name
        in_place = out_dump.resolve() == src_dump.resolve()
    elif overwrite:
        out_dump = src_dump
        run_id = src_dump.name
        in_place = True
    else:
        run_id, out_dump, _ = make_run_dirs("decode")
        in_place = False

    if snapshot and in_place:
        try:
            from dump_history import label_for_bones, snapshot_dump

            snapshot_dump(
                src_dump,
                kind="save",
                label=label_for_bones([bone]),
                bones=[bone],
                ops_summary=op or "",
            )
        except Exception:
            pass

    (out_dump / "section_0.bin").write_bytes(section0)
    if not in_place:
        for p in src_dump.glob("section_*.bin"):
            if p.name == "section_0.bin":
                continue
            (out_dump / p.name).write_bytes(p.read_bytes())
        for name in ("SOURCE.txt", "strings.txt", "decoded.json", "decoded.dae"):
            src = src_dump / name
            if src.is_file():
                (out_dump / name).write_bytes(src.read_bytes())

    write_anim_map(anim, out_dump / "anim_map.json")
    edit_doc = {
        "from_dump": src_dump.name,
        "overwrite": in_place,
        "base_gr2": Path(base_gr2).name,
        "edits": [edit],
    }
    (out_dump / "EDIT.json").write_text(json.dumps(edit_doc, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "run_id": run_id,
        "out_dump": str(out_dump),
        "overwrite": in_place,
        "edit": edit,
        "bones": anim.bones,
    }


def _resolve_dump_base(dump_dir: Path, base_gr2: Optional[Path] = None) -> Path:
    if base_gr2 is not None:
        return Path(base_gr2)
    from paths import resolve_input_gr2

    name = None
    abs_path = None
    src_txt = Path(dump_dir) / "SOURCE.txt"
    if src_txt.is_file():
        for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("path="):
                abs_path = line.split("=", 1)[1].strip()
            elif line.startswith("name="):
                name = line.split("=", 1)[1].strip()
    if abs_path:
        p = Path(abs_path)
        if p.is_file():
            return p
    return resolve_input_gr2(name or "")


_DUMP_GR2_CACHE: Dict[str, Tuple[float, Any, AnimMap, bytearray]] = {}
_DUMP_GR2_CACHE_MAX = 8


def _dump_cache_fingerprint(dump_dir: Path) -> float:
    """Newest mtime among section overlays that affect pose/curves."""
    newest = 0.0
    for name in (
        "section_0.bin",
        "section_1.bin",
        "section_0.relocations.bin",
        "anim_map.json",
        "SOURCE.txt",
    ):
        p = dump_dir / name
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


def invalidate_dump_cache(dump_dir: Optional[Path] = None) -> None:
    """Drop cached dump loads (call after section edits / encode apply)."""
    global _DUMP_GR2_CACHE
    if dump_dir is None:
        _DUMP_GR2_CACHE.clear()
        return
    key = str(Path(dump_dir).resolve())
    _DUMP_GR2_CACHE.pop(key, None)


def load_dump_gr2(
    dump_dir: Path, base_gr2: Optional[Path] = None, *, use_cache: bool = True
) -> Tuple[GR2File, AnimMap, bytearray]:
    """Load base GR2 + overlay dump section bins; return gr2, anim_map, mutable section0."""
    dump_dir = Path(dump_dir)
    cache_key = str(dump_dir.resolve())
    fp = _dump_cache_fingerprint(dump_dir)
    if use_cache:
        hit = _DUMP_GR2_CACHE.get(cache_key)
        if hit is not None and hit[0] == fp:
            _gr2, anim, s0 = hit[1], hit[2], hit[3]
            # Return a fresh bytearray copy so callers can mutate safely
            return _gr2, anim, bytearray(s0)

    base = _resolve_dump_base(dump_dir, base_gr2)
    gr2 = load_gr2(base)
    decode_sections(gr2, True)
    extract_section_fixups(gr2)
    for i, sec in enumerate(gr2.sections):
        p = dump_dir / f"section_{i}.bin"
        if p.is_file() and sec.uncompressed is not None:
            data = p.read_bytes()
            # Allow grown sections (quat_k16 insert appends KnotsControls)
            if len(data) >= len(sec.uncompressed) or len(data) == len(sec.uncompressed):
                sec.uncompressed = data
                sec.header.uncompressed_size = len(data)
        reloc_p = dump_dir / f"section_{i}.relocations.bin"
        if reloc_p.is_file():
            sec.relocations = reloc_p.read_bytes()
        mm_p = dump_dir / f"section_{i}.mm.bin"
        if mm_p.is_file():
            sec.mixed_marshalling = mm_p.read_bytes()
    anim = build_anim_map(gr2, source=str(base))
    s0 = bytearray(gr2.sections[0].uncompressed or b"")
    # Restore capacity from prior anim_map.json if present (after key deletes).
    prev_path = dump_dir / "anim_map.json"
    if prev_path.is_file():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            by_name = {t["name"]: t for t in prev.get("tracks") or [] if t.get("name")}
            for t in anim.tracks:
                pt = by_name.get(t.name)
                if not pt:
                    continue
                pos = pt.get("position") or {}
                cap_bytes = int(pos.get("values_bytes") or 0)
                if t.position.edit_mode == "float_controls" and cap_bytes > t.position.values_bytes:
                    t.position.values_bytes = cap_bytes
                ori = pt.get("orientation") or {}
                o_cap = int(ori.get("values_bytes") or 0)
                if t.orientation.edit_mode == "float_controls" and o_cap > t.orientation.values_bytes:
                    t.orientation.values_bytes = o_cap
                scl = pt.get("scale") or {}
                s_cap = int(scl.get("values_bytes") or 0)
                if t.scale.edit_mode == "float_controls" and s_cap > t.scale.values_bytes:
                    t.scale.values_bytes = s_cap
        except Exception:
            pass
    for t in anim.tracks:
        cm = t.position
        if cm.edit_mode == "float_controls" and cm.object_offset >= 0:
            live_count = struct.unpack_from("<I", s0, cm.object_offset + 4)[0]
            cm.knot_count = live_count
            if cm.values_bytes < live_count * 12:
                cm.values_bytes = live_count * 12
        oc = t.orientation
        if oc.edit_mode == "float_controls" and oc.object_offset >= 0:
            live_count = struct.unpack_from("<I", s0, oc.object_offset + 4)[0]
            oc.knot_count = live_count
            if oc.values_bytes < live_count * 16:
                oc.values_bytes = live_count * 16
        elif oc.edit_mode == "quat_k16" and oc.object_offset >= 0:
            u16_count = struct.unpack_from("<I", s0, oc.object_offset + 8)[0]
            if u16_count >= 4 and u16_count % 4 == 0:
                oc.knot_count = u16_count // 4
                oc.values_bytes = oc.knot_count * 8
        sc = t.scale
        if sc.edit_mode == "float_controls" and sc.object_offset >= 0:
            live_count = struct.unpack_from("<I", s0, sc.object_offset + 4)[0]
            sc.knot_count = live_count
            if sc.values_bytes < live_count * 12:
                sc.values_bytes = live_count * 12

    if use_cache:
        # Store immutable snapshot; callers get a copy of s0
        _DUMP_GR2_CACHE[cache_key] = (fp, gr2, anim, bytes(s0))
        if len(_DUMP_GR2_CACHE) > _DUMP_GR2_CACHE_MAX:
            # Drop oldest arbitrary entry
            oldest = next(iter(_DUMP_GR2_CACHE))
            if oldest != cache_key:
                _DUMP_GR2_CACHE.pop(oldest, None)

    return gr2, anim, s0


def float_curve_capacity(cm: CurveMap) -> int:
    if cm.values_bytes <= 0:
        return cm.knot_count
    # Position/scale: 12 bytes per key; orientation float: 16; quat_k16: 8
    if cm.edit_mode == "quat_k16":
        return max(cm.knot_count, cm.values_bytes // 8)
    if cm.format == FMT_D4_CONSTANT32F or (
        cm.edit_mode == "float_controls" and cm.values_bytes % 16 == 0 and cm.values_bytes // max(cm.knot_count, 1) == 16
    ):
        return max(cm.knot_count, cm.values_bytes // 16)
    return max(cm.knot_count, cm.values_bytes // 12)


def _orient_selector(s0: bytes, cm: CurveMap) -> int:
    assert cm.control_scales_offset is not None
    return struct.unpack_from("<H", s0, cm.control_scales_offset)[0]


def _set_orient_selector(s0: bytearray, cm: CurveMap, selector: int) -> None:
    assert cm.control_scales_offset is not None
    struct.pack_into("<H", s0, cm.control_scales_offset, int(selector) & 0xFFFF)


def _orient_one_over_knot_scale(s0: bytes, cm: CurveMap) -> float:
    return struct.unpack_from("<f", s0, cm.values_offset)[0]


def _ensure_quat_k16_range(
    s0: bytearray,
    cm: CurveMap,
    needed_quats: Sequence[Sequence[float]],
    *,
    min_dot: float = 0.995,
) -> Dict[str, Any]:
    """Widen ScaleOffsetTableEntries if needed so needed_quats encode without crushing.

    Re-encodes ALL existing knots under the new selector (same times). Returns meta.
    """
    from granny_quat import (
        controls_from_quat_series,
        find_selector_for_quats,
        quat_encode_quality,
        quat_from_controls,
        scale_offset_tables,
    )

    if cm.edit_mode != "quat_k16" or cm.knots_controls_offset is None:
        return {"widened": False}
    cur_sel = _orient_selector(s0, cm)
    scales, offsets = scale_offset_tables(cur_sel)
    oos, times, ctrls = _read_quat_k16_knots(s0, cm)
    existing = [quat_from_controls(a, b, c, scales, offsets) for a, b, c in ctrls]

    # Does current selector already represent needed quats well?
    ok = True
    for q in needed_quats:
        d, clamped = quat_encode_quality(q, scales, offsets)
        if clamped or d < min_dot:
            ok = False
            break
    if ok:
        return {"widened": False, "selector": cur_sel}

    all_quats = list(existing) + [list(q) for q in needed_quats]
    new_sel = find_selector_for_quats(all_quats, min_dot=min_dot)
    if new_sel == cur_sel:
        # Still try — maybe find_selector returned same but we need rewrite without prefer_abc
        new_sel = find_selector_for_quats(all_quats, min_dot=0.98)

    new_scales, new_offsets = scale_offset_tables(new_sel)
    new_ctrls = controls_from_quat_series(existing, new_scales, new_offsets)

    # Rewrite knots blob in-place (same size) + selector
    n = len(times)
    blob, packed_n = _pack_quat_k16_blob(oos, times, new_ctrls)
    if packed_n != n:
        raise ValueError(f"quat_k16 widen pack shrunk knots ({n} → {packed_n})")
    off = cm.knots_controls_offset
    s0[off : off + len(blob)] = blob
    _set_orient_selector(s0, cm, new_sel)
    return {
        "widened": new_sel != cur_sel,
        "selector": new_sel,
        "old_selector": cur_sel,
        "keys_rewritten": n,
    }


def _sample_orientation(
    s0: bytes, cm: CurveMap, *, dequant: bool = True
) -> Dict[str, Any]:
    from granny_quat import quat_from_controls, scale_offset_tables

    out: Dict[str, Any] = {
        "edit_mode": cm.edit_mode,
        "editable": cm.editable,
        "format": cm.format,
        "knot_count": cm.knot_count,
        "capacity": cm.knot_count,
        "times": [],
        "rotations": [],  # [x,y,z,w]
        "note": cm.note,
    }
    if cm.edit_mode == "none" or cm.format == FMT_DA_IDENTITY or cm.format < 0:
        out["times"] = [0.0]
        out["rotations"] = [IDENTITY_QUAT[:]]
        return out
    if cm.edit_mode == "constant":
        q = list(struct.unpack_from("<4f", s0, cm.values_offset))
        out["times"] = [0.0]
        out["rotations"] = [q]
        return out
    if cm.edit_mode == "float_controls":
        n = cm.knot_count
        times = _knot_times_dak32(s0, cm)
        rots = [list(struct.unpack_from("<4f", s0, cm.values_offset + i * 16)) for i in range(n)]
        out["times"] = times
        out["rotations"] = rots
        out["capacity"] = float_curve_capacity(cm)
        return out
    if cm.edit_mode == "quat_k16" and cm.knots_controls_offset is not None and cm.knot_count > 0:
        if not dequant:
            # Pre-dequant view: no u16→quat expansion (identity stand-in).
            out["times"] = [0.0]
            out["rotations"] = [IDENTITY_QUAT[:]]
            out["note"] = "quat_k16 — dequant off (identity)"
            out["dequant"] = False
            return out
        n = cm.knot_count
        oos = _orient_one_over_knot_scale(s0, cm)
        sel = _orient_selector(s0, cm)
        scales, offsets = scale_offset_tables(sel)
        times = []
        rots = []
        for i in range(n):
            kt = struct.unpack_from("<H", s0, cm.knots_controls_offset + i * 2)[0]
            times.append(float(kt) / oos if oos else 0.0)
            a, b, c = struct.unpack_from(
                "<3H", s0, cm.knots_controls_offset + n * 2 + i * 6
            )
            rots.append(quat_from_controls(a, b, c, scales, offsets))
        out["times"] = times
        out["rotations"] = rots
        out["capacity"] = n
        out["one_over_knot_scale"] = oos
        out["tick_dt"] = (1.0 / oos) if oos else None
        out["note"] = "quantized D4nK16uC15u (u16 ticks)"
        out["dequant"] = True
        return out
    if cm.edit_mode == "quat_k8" and cm.knots_controls_offset is not None and cm.knot_count > 0:
        from granny_quat import quat_from_controls_u7, scale_offset_tables_u7

        if not dequant:
            out["times"] = [0.0]
            out["rotations"] = [IDENTITY_QUAT[:]]
            out["note"] = "quat_k8 — dequant off (identity)"
            out["dequant"] = False
            return out
        n = cm.knot_count
        oos = _orient_one_over_knot_scale(s0, cm)
        scales, offsets = scale_offset_tables_u7(_orient_selector(s0, cm))
        times = []
        rots = []
        base = cm.knots_controls_offset
        for i in range(n):
            kt = s0[base + i]
            times.append(float(kt) / oos if oos else 0.0)
            a = s0[base + n + i * 3 + 0]
            b = s0[base + n + i * 3 + 1]
            c = s0[base + n + i * 3 + 2]
            rots.append(quat_from_controls_u7(a, b, c, scales, offsets))
        out["times"] = times
        out["rotations"] = rots
        out["capacity"] = n
        out["one_over_knot_scale"] = oos
        out["note"] = "quantized D4nK8uC7u (u8 ticks)"
        out["dequant"] = True
        return out
    return out


def _sample_scale(s0: bytes, cm: CurveMap) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "edit_mode": cm.edit_mode,
        "editable": cm.editable,
        "format": cm.format,
        "knot_count": cm.knot_count,
        "capacity": cm.knot_count,
        "times": [],
        "scales": [],
        "note": cm.note,
    }
    if cm.edit_mode == "none" or cm.format == FMT_DA_IDENTITY or cm.format < 0:
        out["times"] = [0.0]
        out["scales"] = [IDENTITY_SCALE[:]]
        return out
    if cm.edit_mode == "constant":
        out["times"] = [0.0]
        out["scales"] = [list(struct.unpack_from("<3f", s0, cm.values_offset))]
        return out
    if cm.edit_mode == "k16_offsets":
        assert cm.control_offsets_offset is not None
        out["times"] = [0.0]
        out["scales"] = [list(struct.unpack_from("<3f", s0, cm.control_offsets_offset))]
        return out
    if cm.edit_mode == "float_controls":
        n = cm.knot_count
        times = _knot_times_dak32(s0, cm)
        scales = [list(struct.unpack_from("<3f", s0, cm.values_offset + i * 12)) for i in range(n)]
        out["times"] = times
        out["scales"] = scales
        out["capacity"] = float_curve_capacity(cm)
        return out
    return out


def _d3k16_control_scales(s0: bytes, cm: CurveMap) -> Tuple[float, float, float]:
    assert cm.control_scales_offset is not None
    return struct.unpack_from("<3f", s0, cm.control_scales_offset)


def _d3k16_control_offsets(s0: bytes, cm: CurveMap) -> Tuple[float, float, float]:
    assert cm.control_offsets_offset is not None
    return struct.unpack_from("<3f", s0, cm.control_offsets_offset)


def _d3k16_one_over_knot_scale(s0: bytes, cm: CurveMap) -> float:
    trunc = struct.unpack_from("<H", s0, cm.object_offset + 2)[0]
    return _one_over_knot_scale(trunc)


def _d3k16_dequant_xyz(
    scales: Sequence[float], offsets: Sequence[float], c0: int, c1: int, c2: int
) -> List[float]:
    return [
        float(offsets[0]) + float(scales[0]) * float(c0),
        float(offsets[1]) + float(scales[1]) * float(c1),
        float(offsets[2]) + float(scales[2]) * float(c2),
    ]


def _read_d3k16_knots(
    s0: bytes, cm: CurveMap
) -> Tuple[float, List[float], List[List[float]]]:
    """Return (one_over_knot_scale, times, positions) for D3K16 / D3K8 / D3I1*.

    D3K16uC16u: knot_u16[n] then control triples (c0,c1,c2) per sample.
    D3K8uC8u:   knot_u8[n] then control_u8 triples (c0,c1,c2) per sample.
    D3I1K16uC16u / D3I1K8uC8u: knot[n] then one scalar control[n], expanded
    to XYZ with the same control value on each axis (LSLib GetPoints).
    """
    assert cm.knots_controls_offset is not None and cm.knot_count > 0
    oos = _d3k16_one_over_knot_scale(s0, cm)
    scales = _d3k16_control_scales(s0, cm)
    offsets = _d3k16_control_offsets(s0, cm)
    n = cm.knot_count
    base = cm.knots_controls_offset
    times: List[float] = []
    positions: List[List[float]] = []

    if cm.format == FMT_D3_I1_K16U_C16U:
        for i in range(n):
            knot = struct.unpack_from("<H", s0, base + i * 2)[0]
            c = struct.unpack_from("<H", s0, base + n * 2 + i * 2)[0]
            times.append(float(knot) / oos if oos else 0.0)
            positions.append(_d3k16_dequant_xyz(scales, offsets, c, c, c))
        return oos, times, positions

    if cm.format == FMT_D3_I1_K8U_C8U:
        for i in range(n):
            knot = s0[base + i]
            c = s0[base + n + i]
            times.append(float(knot) / oos if oos else 0.0)
            positions.append(_d3k16_dequant_xyz(scales, offsets, c, c, c))
        return oos, times, positions

    if cm.format == FMT_D3_K8U_C8U:
        for i in range(n):
            knot = s0[base + i]
            c0 = s0[base + n + i * 3]
            c1 = s0[base + n + i * 3 + 1]
            c2 = s0[base + n + i * 3 + 2]
            times.append(float(knot) / oos if oos else 0.0)
            positions.append(_d3k16_dequant_xyz(scales, offsets, c0, c1, c2))
        return oos, times, positions

    # Default: D3K16uC16u (and any legacy mapped as k16_offsets)
    for i in range(n):
        knot = struct.unpack_from("<H", s0, base + i * 2)[0]
        c0, c1, c2 = struct.unpack_from("<3H", s0, base + n * 2 + i * 6)
        times.append(float(knot) / oos if oos else 0.0)
        positions.append(_d3k16_dequant_xyz(scales, offsets, c0, c1, c2))
    return oos, times, positions


def sample_curve(
    s0: bytes, anim: AnimMap, bone: str, *, dequant: bool = True
) -> Dict[str, Any]:
    track = _track_by_name(anim, bone)
    cm = track.position
    ori_sample = _sample_orientation(s0, track.orientation, dequant=dequant)
    ori_sample["degree"] = int(track.orientation.degree or 0)
    scl_sample = _sample_scale(s0, track.scale)
    scl_sample["degree"] = int(track.scale.degree or 0)
    out: Dict[str, Any] = {
        "bone": bone,
        "edit_mode": cm.edit_mode,
        "editable": cm.editable,
        "format": cm.format,
        "degree": int(cm.degree or 0),
        "knot_count": cm.knot_count,
        "capacity": float_curve_capacity(cm) if cm.edit_mode == "float_controls" else cm.knot_count,
        "times": [],
        "positions": [],
        "note": cm.note,
        "duration": anim.duration,
        "orientation": ori_sample,
        "scale": scl_sample,
        "dequant": bool(dequant),
    }
    if cm.edit_mode == "constant":
        xyz = list(struct.unpack_from("<3f", s0, cm.values_offset))
        out["times"] = [0.0]
        out["positions"] = [xyz]
        return out
    if cm.edit_mode == "k16_offsets":
        assert cm.control_offsets_offset is not None
        if (
            dequant
            and cm.knots_controls_offset is not None
            and cm.knot_count > 0
        ):
            _oos, times, positions = _read_d3k16_knots(s0, cm)
            out["times"] = times
            out["positions"] = positions
            out["capacity"] = cm.knot_count
            out["one_over_knot_scale"] = _oos
            out["note"] = cm.note or "D3K16uC16u dequantized"
            return out
        xyz = list(struct.unpack_from("<3f", s0, cm.control_offsets_offset))
        out["times"] = [0.0]
        out["positions"] = [xyz]
        out["note"] = (
            "quantized curve — bias (ControlOffsets) only"
            if not dequant
            else "quantized curve — bias (ControlOffsets) only"
        )
        out["dequant"] = False
        return out
    if cm.edit_mode == "float_controls":
        n = cm.knot_count
        times = _knot_times_dak32(s0, cm)
        positions = []
        for i in range(n):
            positions.append(list(struct.unpack_from("<3f", s0, cm.values_offset + i * 12)))
        out["times"] = times
        out["positions"] = positions
        return out
    return out


def sample_all_curves(
    s0: bytes,
    anim: AnimMap,
    *,
    bones: Optional[Sequence[str]] = None,
    dequant: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Sample curves for many bones in one pass (one dump load upstream)."""
    want = None
    if bones:
        want = {str(b) for b in bones if b}
    out: Dict[str, Dict[str, Any]] = {}
    for track in anim.tracks:
        if not track.name:
            continue
        if want is not None and track.name not in want:
            continue
        try:
            out[track.name] = sample_curve(s0, anim, track.name, dequant=dequant)
        except Exception:
            continue
    return out


def _lerp3(a: Sequence[float], b: Sequence[float], u: float) -> List[float]:
    return [a[i] + (b[i] - a[i]) * u for i in range(3)]


def eval_position_at(s0: bytes, cm: CurveMap, t: float) -> List[float]:
    if cm.edit_mode == "constant":
        return list(struct.unpack_from("<3f", s0, cm.values_offset))
    if cm.edit_mode == "k16_offsets":
        assert cm.control_offsets_offset is not None
        if cm.knots_controls_offset is not None and cm.knot_count > 0:
            _oos, times, positions = _read_d3k16_knots(s0, cm)
            if not times:
                return list(_d3k16_control_offsets(s0, cm))
            if t <= times[0]:
                return list(positions[0])
            if t >= times[-1]:
                return list(positions[-1])
            for i in range(len(times) - 1):
                if times[i] <= t <= times[i + 1]:
                    span = times[i + 1] - times[i]
                    u = 0.0 if span <= 1e-9 else (t - times[i]) / span
                    return _lerp3(positions[i], positions[i + 1], u)
            return list(positions[-1])
        return list(struct.unpack_from("<3f", s0, cm.control_offsets_offset))
    if cm.edit_mode == "float_controls" and cm.knot_count > 0:
        times = _knot_times_dak32(s0, cm)
        if not times:
            return [0.0, 0.0, 0.0]
        deg = int(cm.degree or 0)
        # Degree≥2 float = Granny B-spline controls (e.g. Dummy_Root before densify).
        # Degree=1 = baked samples → linear lerp.
        if deg >= 2 and cm.values_offset is not None and cm.knot_count >= 2:
            from granny_bspline import sample_bspline

            controls = [
                list(struct.unpack_from("<3f", s0, cm.values_offset + i * 12))
                for i in range(cm.knot_count)
            ]
            return list(sample_bspline(deg, times, controls, t))
        if t <= times[0]:
            return list(struct.unpack_from("<3f", s0, cm.values_offset))
        if t >= times[-1]:
            return list(struct.unpack_from("<3f", s0, cm.values_offset + (cm.knot_count - 1) * 12))
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                span = times[i + 1] - times[i]
                u = 0.0 if span <= 1e-9 else (t - times[i]) / span
                a = struct.unpack_from("<3f", s0, cm.values_offset + i * 12)
                b = struct.unpack_from("<3f", s0, cm.values_offset + (i + 1) * 12)
                return _lerp3(a, b, u)
    return [0.0, 0.0, 0.0]


def eval_orientation_at(s0: bytes, cm: CurveMap, t: float) -> List[float]:
    from granny_quat import quat_from_controls, scale_offset_tables, slerp

    if cm.format == FMT_DA_IDENTITY or cm.edit_mode == "none" or cm.format < 0:
        return IDENTITY_QUAT[:]
    if cm.edit_mode == "constant":
        return list(struct.unpack_from("<4f", s0, cm.values_offset))
    if cm.edit_mode == "float_controls" and cm.knot_count > 0:
        times = _knot_times_dak32(s0, cm)
        if not times:
            return IDENTITY_QUAT[:]
        if t <= times[0]:
            return list(struct.unpack_from("<4f", s0, cm.values_offset))
        if t >= times[-1]:
            return list(struct.unpack_from("<4f", s0, cm.values_offset + (cm.knot_count - 1) * 16))
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                span = times[i + 1] - times[i]
                u = 0.0 if span <= 1e-9 else (t - times[i]) / span
                a = struct.unpack_from("<4f", s0, cm.values_offset + i * 16)
                b = struct.unpack_from("<4f", s0, cm.values_offset + (i + 1) * 16)
                return slerp(a, b, u)
        return IDENTITY_QUAT[:]
    if cm.edit_mode == "quat_k16" and cm.knots_controls_offset and cm.knot_count > 0:
        n = cm.knot_count
        oos = _orient_one_over_knot_scale(s0, cm)
        scales, offsets = scale_offset_tables(_orient_selector(s0, cm))

        def knot_t(i: int) -> float:
            return struct.unpack_from("<H", s0, cm.knots_controls_offset + i * 2)[0] / oos

        def knot_q(i: int) -> List[float]:
            a, b, c = struct.unpack_from("<3H", s0, cm.knots_controls_offset + n * 2 + i * 6)
            return quat_from_controls(a, b, c, scales, offsets)

        if t <= knot_t(0):
            return knot_q(0)
        if t >= knot_t(n - 1):
            return knot_q(n - 1)
        for i in range(n - 1):
            t0, t1 = knot_t(i), knot_t(i + 1)
            if t0 <= t <= t1:
                span = t1 - t0
                u = 0.0 if span <= 1e-9 else (t - t0) / span
                return slerp(knot_q(i), knot_q(i + 1), u)
    if cm.edit_mode == "quat_k8" and cm.knots_controls_offset and cm.knot_count > 0:
        from granny_quat import quat_from_controls_u7, scale_offset_tables_u7

        n = cm.knot_count
        oos = _orient_one_over_knot_scale(s0, cm)
        scales, offsets = scale_offset_tables_u7(_orient_selector(s0, cm))
        base = cm.knots_controls_offset

        def knot_t8(i: int) -> float:
            return float(s0[base + i]) / oos if oos else 0.0

        def knot_q8(i: int) -> List[float]:
            a = s0[base + n + i * 3 + 0]
            b = s0[base + n + i * 3 + 1]
            c = s0[base + n + i * 3 + 2]
            return quat_from_controls_u7(a, b, c, scales, offsets)

        if t <= knot_t8(0):
            return knot_q8(0)
        if t >= knot_t8(n - 1):
            return knot_q8(n - 1)
        for i in range(n - 1):
            t0, t1 = knot_t8(i), knot_t8(i + 1)
            if t0 <= t <= t1:
                span = t1 - t0
                u = 0.0 if span <= 1e-9 else (t - t0) / span
                return slerp(knot_q8(i), knot_q8(i + 1), u)
    return IDENTITY_QUAT[:]


def eval_scale_at(s0: bytes, cm: CurveMap, t: float) -> List[float]:
    if cm.format == FMT_DA_IDENTITY or cm.edit_mode == "none" or cm.format < 0:
        return IDENTITY_SCALE[:]
    if cm.edit_mode == "constant":
        return list(struct.unpack_from("<3f", s0, cm.values_offset))
    if cm.edit_mode == "k16_offsets":
        assert cm.control_offsets_offset is not None
        return list(struct.unpack_from("<3f", s0, cm.control_offsets_offset))
    if cm.edit_mode == "float_controls" and cm.knot_count > 0:
        return eval_position_at(s0, cm, t)  # same 3-float layout
    return IDENTITY_SCALE[:]


def finger_parent(name: str) -> Optional[str]:
    import re

    m = re.match(
        r"^(Index|Middle|Ring|Pinky|Thumb)Finger(\d+)_([LR])(?:_endBone)?$",
        name,
    )
    if not m:
        return None
    kind, num, side = m.group(1), int(m.group(2)), m.group(3)
    # First digit bone → wrist; later digits → previous
    if kind in ("Pinky", "Ring") and num == 0:
        return f"Wrist_{side}"
    if kind in ("Index", "Middle", "Thumb") and num == 1:
        return f"Wrist_{side}"
    if num >= 1:
        return f"{kind}Finger{num - 1}_{side}"
    return f"Wrist_{side}"


# BG3 / AdvancedSkeleton-style humanoid parents (track names → parent track)
_BONE_PARENTS: Dict[str, str] = {
    "Root_M": "Dummy_Root",
    "Spine1_M": "Root_M",
    "Spine2_M": "Spine1_M",
    "Chest_M": "Spine2_M",
    "Neck_M": "Chest_M",
    "Head_M": "Neck_M",
    "Hip_L": "Root_M",
    "Hip_R": "Root_M",
    "Hip_L_Twist_01": "Hip_L",
    "Hip_L_Twist_02": "Hip_L_Twist_01",
    "Hip_R_Twist_01": "Hip_R",
    "Hip_R_Twist_02": "Hip_R_Twist_01",
    "Knee_L": "Hip_L",
    "Knee_R": "Hip_R",
    "Ankle_L": "Knee_L",
    "Ankle_R": "Knee_R",
    "Toes_L": "Ankle_L",
    "Toes_R": "Ankle_R",
    "ToesEnd_L_endBone": "Toes_L",
    "ToesEnd_R_endBone": "Toes_R",
    "HeadEnd_M_endBone": "Head_M",
    "Scapula_L": "Chest_M",
    "Scapula_R": "Chest_M",
    "Shoulder_L": "Scapula_L",
    "Shoulder_R": "Scapula_R",
    "Shoulder_L_Twist_01": "Shoulder_L",
    "Shoulder_L_Twist_02": "Shoulder_L_Twist_01",
    "Shoulder_R_Twist_01": "Shoulder_R",
    "Shoulder_R_Twist_02": "Shoulder_R_Twist_01",
    "Elbow_L": "Shoulder_L",
    "Elbow_R": "Shoulder_R",
    "Elbow_Twist_L": "Elbow_L",
    "Elbow_Twist_R": "Elbow_R",
    "Wrist_L": "Elbow_L",
    "Wrist_R": "Elbow_R",
}


def infer_bone_parent(name: str, names: set) -> Optional[str]:
    """Infer parent bone for anim-only GR2s that lack a Skeleton chunk."""
    if not name or name == "Dummy_Root":
        return None
    if name in _BONE_PARENTS:
        p = _BONE_PARENTS[name]
        return p if p in names else ("Dummy_Root" if "Dummy_Root" in names else None)
    fp = finger_parent(name)
    if fp and fp in names:
        return fp
    if name.startswith("Dummy_"):
        # Foot/Hand IK targets are authored in root space (BG3 AS skeleton:
        # Dummy_*_Foot_IK / Dummy_*_Hand_IK → Dummy_Root). Matching L_Foot /
        # L_Hand first would wrongly parent them under Ankle/Wrist and offset
        # world IK by ~0.2–0.4m.
        if "_IK" in name:
            return "Dummy_Root" if "Dummy_Root" in names else None
        host_map = [
            ("Eye", "Head_M"),
            ("Mouth", "Head_M"),
            ("Head", "Head_M"),
            ("Neck", "Neck_M"),
            ("Chest", "Chest_M"),
            ("Body", "Chest_M"),
            ("Wing", "Chest_M"),
            ("Status", "Chest_M"),
            ("HitImpact", "Chest_M"),
            ("Overhead", "Head_M"),
            ("Playerlight", "Chest_M"),
            ("L_Hand", "Wrist_L"),
            ("R_Hand", "Wrist_R"),
            ("L_Foot", "Ankle_L"),
            ("R_Foot", "Ankle_R"),
            ("L_Knee", "Knee_L"),
            ("R_Knee", "Knee_R"),
            ("L_Tentacle", "Hip_L"),
            ("R_Tentacle", "Hip_R"),
            ("Sheath_Hip_L", "Hip_L"),
            ("Sheath_Hip_R", "Hip_R"),
            ("Sheath", "Chest_M"),
        ]
        for key, host in host_map:
            if key in name and host in names:
                return host
        return "Dummy_Root" if "Dummy_Root" in names else None
    return "Dummy_Root" if "Dummy_Root" in names else None


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_rotate(q: Sequence[float], v: Sequence[float]) -> List[float]:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return [
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    ]


def build_skeleton(anim: AnimMap, s0: bytes) -> Dict[str, Any]:
    """Build skeleton with inferred parents; rest pose is local track samples at t=0."""
    names = {t.name for t in anim.tracks if t.name}
    name_to_idx = {t.name: i for i, t in enumerate(anim.tracks) if t.name}
    bones = []
    for i, t in enumerate(anim.tracks):
        if not t.name:
            continue
        parent_name = infer_bone_parent(t.name, names)
        parent = name_to_idx[parent_name] if parent_name and parent_name in name_to_idx else -1
        if t.position.edit_mode == "none":
            rest = [0.0, 0.0, 0.0]
        else:
            rest = eval_position_at(s0, t.position, 0.0)
        rest_rot = eval_orientation_at(s0, t.orientation, 0.0)
        rest_scl = eval_scale_at(s0, t.scale, 0.0)
        bones.append(
            {
                "index": i,
                "name": t.name,
                "parent": parent,
                "parent_name": parent_name,
                "rest_translation": rest,
                "rest_rotation": rest_rot,
                "rest_scale": rest_scl,
                "edit_mode": t.position.edit_mode,
                "editable": t.position.editable,
                "orient_edit_mode": t.orientation.edit_mode,
                "orient_editable": t.orientation.editable,
                "scale_edit_mode": t.scale.edit_mode,
                "scale_editable": t.scale.editable,
            }
        )
    root = "Dummy_Root" if "Dummy_Root" in names else (bones[0]["name"] if bones else "")
    return {
        "source": anim.source,
        "duration": anim.duration,
        "root": root,
        "bones": bones,
    }


def write_skeleton(skel: Dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(skel, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def eval_pose(s0: bytes, anim: AnimMap, t: float) -> Dict[str, Any]:
    """Local-space pose samples (translation/rotation/scale per track)."""
    bones = []
    for track in anim.tracks:
        if not track.name:
            continue
        bones.append(
            {
                "name": track.name,
                "index": track.index,
                "translation": eval_position_at(s0, track.position, t),
                "rotation": eval_orientation_at(s0, track.orientation, t),
                "scale": eval_scale_at(s0, track.scale, t),
                "edit_mode": track.position.edit_mode,
            }
        )
    return {"t": t, "duration": anim.duration, "bones": bones, "space": "local"}


def eval_world_pose(s0: bytes, anim: AnimMap, t: float, skel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """FK world-space pose from local tracks + inferred hierarchy."""
    local = eval_pose(s0, anim, t)
    if skel is None:
        skel = build_skeleton(anim, s0)
    parent_of = {b["name"]: b.get("parent_name") for b in skel.get("bones") or []}
    by = {b["name"]: b for b in local["bones"]}
    world: Dict[str, Dict[str, Any]] = {}
    remaining = set(by.keys())
    guard = 0
    while remaining and guard < len(remaining) + 5:
        guard += 1
        progressed = False
        for name in list(remaining):
            p = parent_of.get(name)
            if p is not None and p not in world and p in by:
                continue
            loc = by[name]
            lt = loc["translation"]
            lq = loc["rotation"]
            ls = loc["scale"]
            if p is None or p not in world:
                world[name] = {
                    "name": name,
                    "translation": lt[:],
                    "rotation": lq[:],
                    "scale": ls[:],
                }
            else:
                pt = world[p]["translation"]
                pq = world[p]["rotation"]
                wt = _quat_rotate(pq, lt)
                world[name] = {
                    "name": name,
                    "translation": [pt[0] + wt[0], pt[1] + wt[1], pt[2] + wt[2]],
                    "rotation": _quat_mul(pq, lq),
                    "scale": [
                        world[p]["scale"][0] * ls[0],
                        world[p]["scale"][1] * ls[1],
                        world[p]["scale"][2] * ls[2],
                    ],
                }
            remaining.remove(name)
            progressed = True
        if not progressed:
            for name in list(remaining):
                loc = by[name]
                world[name] = {
                    "name": name,
                    "translation": loc["translation"][:],
                    "rotation": loc["rotation"][:],
                    "scale": loc["scale"][:],
                }
                remaining.remove(name)
            break
    return {
        "t": t,
        "duration": anim.duration,
        "bones": list(world.values()),
        "space": "world",
    }


def patch_orientation(
    section0: bytearray,
    anim: AnimMap,
    bone: str,
    op: str,
    quat: Sequence[float],
    *,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """Patch orientation. op: set | multiply. quat is [x,y,z,w]."""
    from granny_quat import controls_from_quat, quat_multiply, scale_offset_tables

    if op not in ("set", "multiply", "offset"):
        raise ValueError("op must be set/multiply/offset")
    if len(quat) != 4:
        raise ValueError("quat must have 4 floats")
    track = _track_by_name(anim, bone)
    cm = track.orientation
    if not cm.editable:
        raise ValueError(f"Bone {bone} orientation not editable ({cm.note or cm.format})")

    qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    if cm.edit_mode == "constant":
        cur = list(struct.unpack_from("<4f", section0, cm.values_offset))
        if op in ("multiply", "offset"):
            # delta quaternion applied on the left: delta ⊗ current
            new_q = quat_multiply([qx, qy, qz, qw], cur)
        else:
            new_q = [qx, qy, qz, qw]
        struct.pack_into("<4f", section0, cm.values_offset, *new_q)
        return {
            "bone": bone,
            "channel": "orientation",
            "op": op,
            "quat": new_q,
            "format": cm.format,
            "edit_mode": cm.edit_mode,
            "samples_touched": 1,
        }

    if cm.edit_mode == "quat_k16":
        n = cm.knot_count
        assert cm.knots_controls_offset is not None
        from granny_quat import quat_from_controls

        # Build target quats first, widen selector if they'd clamp
        scales, offsets = scale_offset_tables(_orient_selector(section0, cm))
        idxs = list(range(n) if index is None else [int(index)])
        targets: List[List[float]] = []
        for i in idxs:
            if not (0 <= i < n):
                raise IndexError(f"orient key index {i}")
            base = cm.knots_controls_offset + n * 2 + i * 6
            a0, b0, c0 = struct.unpack_from("<3H", section0, base)
            cur = quat_from_controls(a0, b0, c0, scales, offsets)
            if op in ("multiply", "offset"):
                targets.append(quat_multiply([qx, qy, qz, qw], cur))
            else:
                targets.append([qx, qy, qz, qw])
        widen = _ensure_quat_k16_range(section0, cm, targets)
        scales, offsets = scale_offset_tables(_orient_selector(section0, cm))
        touched = 0
        for i, new_q in zip(idxs, targets):
            base = cm.knots_controls_offset + n * 2 + i * 6
            # After widen, don't prefer old abc (swizzle may need to change)
            a, b, c = controls_from_quat(new_q, scales, offsets)
            struct.pack_into("<3H", section0, base, a, b, c)
            touched += 1
        return {
            "bone": bone,
            "channel": "orientation",
            "op": op,
            "quat": targets[0] if len(targets) == 1 else [qx, qy, qz, qw],
            "index": index,
            "format": cm.format,
            "edit_mode": cm.edit_mode,
            "samples_touched": touched,
            **widen,
        }

    if cm.edit_mode == "float_controls":
        n = cm.knot_count
        idxs = range(n) if index is None else [int(index)]
        touched = 0
        for i in idxs:
            off = cm.values_offset + i * 16
            cur = list(struct.unpack_from("<4f", section0, off))
            if op in ("multiply", "offset"):
                new_q = quat_multiply([qx, qy, qz, qw], cur)
            else:
                new_q = [qx, qy, qz, qw]
            struct.pack_into("<4f", section0, off, *new_q)
            touched += 1
        return {
            "bone": bone,
            "channel": "orientation",
            "op": op,
            "quat": [qx, qy, qz, qw],
            "index": index,
            "format": cm.format,
            "edit_mode": cm.edit_mode,
            "samples_touched": touched,
        }

    raise ValueError(f"Unsupported orientation edit_mode {cm.edit_mode}")


def patch_scale(
    section0: bytearray,
    anim: AnimMap,
    bone: str,
    op: str,
    xyz: Sequence[float],
    *,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """Patch scale 3-vector. op: set | multiply | offset."""
    if op not in ("set", "multiply", "offset"):
        raise ValueError("op must be set/multiply/offset")
    if len(xyz) != 3:
        raise ValueError("xyz must have 3 floats")
    track = _track_by_name(anim, bone)
    cm = track.scale
    if not cm.editable:
        raise ValueError(f"Bone {bone} scale not editable ({cm.note or cm.format})")

    dx, dy, dz = float(xyz[0]), float(xyz[1]), float(xyz[2])

    def apply_one(cur: List[float]) -> List[float]:
        if op == "set":
            return [dx, dy, dz]
        if op == "multiply":
            return [cur[0] * dx, cur[1] * dy, cur[2] * dz]
        return [cur[0] + dx, cur[1] + dy, cur[2] + dz]

    if cm.edit_mode == "constant":
        cur = list(struct.unpack_from("<3f", section0, cm.values_offset))
        new_v = apply_one(cur)
        struct.pack_into("<3f", section0, cm.values_offset, *new_v)
        return {
            "bone": bone,
            "channel": "scale",
            "op": op,
            "xyz": new_v,
            "format": cm.format,
            "edit_mode": cm.edit_mode,
            "samples_touched": 1,
        }

    if cm.edit_mode == "k16_offsets":
        assert cm.control_offsets_offset is not None
        cur = list(struct.unpack_from("<3f", section0, cm.control_offsets_offset))
        new_v = apply_one(cur)
        struct.pack_into("<3f", section0, cm.control_offsets_offset, *new_v)
        return {
            "bone": bone,
            "channel": "scale",
            "op": op,
            "xyz": new_v,
            "format": cm.format,
            "edit_mode": cm.edit_mode,
            "samples_touched": 1,
        }

    if cm.edit_mode == "float_controls":
        n = cm.knot_count
        idxs = range(n) if index is None else [int(index)]
        touched = 0
        last = [dx, dy, dz]
        for i in idxs:
            off = cm.values_offset + i * 12
            cur = list(struct.unpack_from("<3f", section0, off))
            last = apply_one(cur)
            struct.pack_into("<3f", section0, off, *last)
            touched += 1
        return {
            "bone": bone,
            "channel": "scale",
            "op": op,
            "xyz": last,
            "index": index,
            "format": cm.format,
            "edit_mode": cm.edit_mode,
            "samples_touched": touched,
        }

    raise ValueError(f"Unsupported scale edit_mode {cm.edit_mode}")


def _rewrite_float_curve(
    s0: bytearray,
    cm: CurveMap,
    times: List[float],
    positions: List[List[float]],
) -> None:
    """Compact-write position knots/controls (3f) inside existing buffer."""
    _rewrite_dak32_curve(s0, cm, times, positions, comps=3)


def _set_curve_degree(s0: bytearray, cm: CurveMap, degree: int) -> None:
    """Write CurveData.Degree (byte at object+1) and mirror onto CurveMap."""
    if cm.object_offset < 0 or cm.object_offset + 2 > len(s0):
        return
    d = int(degree) & 0xFF
    s0[cm.object_offset + 1] = d
    cm.degree = d


def _rewrite_dak32_curve(
    s0: bytearray,
    cm: CurveMap,
    times: List[float],
    values: List[List[float]],
    *,
    comps: int,
    degree: int = CURVE_DEGREE_LINEAR,
) -> None:
    """Compact-write DAK32f knots + float controls; update counts; zero-pad."""
    n = len(times)
    cap = float_curve_capacity(cm)
    if n > cap:
        raise ValueError(f"knot count {n} exceeds capacity {cap}")
    if len(values) != n:
        raise ValueError("times/values length mismatch")
    order = sorted(range(n), key=lambda i: times[i])
    times = [times[i] for i in order]
    values = [values[i] for i in order]
    assert cm.knots_controls_offset is not None
    stride = comps * 4
    for i, tv in enumerate(times):
        struct.pack_into("<f", s0, cm.knots_controls_offset + i * 4, float(tv))
    for i in range(n, cap):
        struct.pack_into("<f", s0, cm.knots_controls_offset + i * 4, 0.0)
    fmt = f"<{comps}f"
    for i, vals in enumerate(values):
        packed = [float(vals[j]) if j < len(vals) else 0.0 for j in range(comps)]
        struct.pack_into(fmt, s0, cm.values_offset + i * stride, *packed)
    zeros = [0.0] * comps
    for i in range(n, cap):
        struct.pack_into(fmt, s0, cm.values_offset + i * stride, *zeros)
    struct.pack_into("<I", s0, cm.object_offset + 4, n)
    struct.pack_into("<I", s0, cm.object_offset + 16, n * comps)
    cm.knot_count = n
    _set_curve_degree(s0, cm, degree)


def _move_float_channel_key(
    s0: bytearray,
    cm: CurveMap,
    index: int,
    t: float,
    *,
    comps: int,
) -> Dict[str, Any]:
    """Move one DAK32f knot in time (position/orientation/scale float curves)."""
    if cm.edit_mode != "float_controls":
        raise ValueError(f"move_key needs float_controls, got {cm.edit_mode}")
    n = cm.knot_count
    if not (0 <= index < n):
        raise IndexError(f"move_key index {index}")
    times = _knot_times_dak32(s0, cm)
    values = [
        list(struct.unpack_from(f"<{comps}f", s0, cm.values_offset + i * comps * 4))
        for i in range(n)
    ]
    times[index] = float(t)
    _rewrite_dak32_curve(s0, cm, times, values, comps=comps)
    return {"t": float(t), "index": index, "edit_mode": cm.edit_mode}


def _move_quat_k16_key(
    s0: bytearray,
    cm: CurveMap,
    index: int,
    t: float,
) -> Dict[str, Any]:
    """Move one D4nK16uC15u orientation knot in time (u16 knots)."""
    if cm.edit_mode != "quat_k16":
        raise ValueError(f"quat_k16 move needs quat_k16, got {cm.edit_mode}")
    n = cm.knot_count
    if not (0 <= index < n):
        raise IndexError(f"move_key index {index}")
    assert cm.knots_controls_offset is not None
    oos = _orient_one_over_knot_scale(s0, cm)
    if not oos or oos <= 0:
        raise ValueError("invalid orientation knot scale")
    # Read all knots + controls, move one time, re-sort, write back
    times: List[float] = []
    ctrls: List[Tuple[int, int, int]] = []
    for i in range(n):
        kt = struct.unpack_from("<H", s0, cm.knots_controls_offset + i * 2)[0]
        times.append(float(kt) / oos)
        a, b, c = struct.unpack_from(
            "<3H", s0, cm.knots_controls_offset + n * 2 + i * 6
        )
        ctrls.append((a, b, c))
    tick, t_snap = _allocate_free_quat_k16_tick(
        oos, times, float(t), exclude_index=index
    )
    times[index] = t_snap
    order = sorted(range(n), key=lambda i: times[i])
    times = [times[i] for i in order]
    ctrls = [ctrls[i] for i in order]
    for i, tv in enumerate(times):
        u = _quat_k16_tick(oos, tv)
        struct.pack_into("<H", s0, cm.knots_controls_offset + i * 2, u)
        a, b, c = ctrls[i]
        struct.pack_into("<3H", s0, cm.knots_controls_offset + n * 2 + i * 6, a, b, c)
    new_idx = order.index(index)
    return {
        "t": t_snap,
        "tick": tick,
        "index": new_idx,
        "edit_mode": cm.edit_mode,
        "snapped": abs(t_snap - float(t)) > 1e-12,
    }


def _read_quat_k16_knots(
    s0: bytes, cm: CurveMap
) -> Tuple[float, List[float], List[Tuple[int, int, int]]]:
    """Return (one_over_knot_scale, times, controls)."""
    n = cm.knot_count
    assert cm.knots_controls_offset is not None
    oos = _orient_one_over_knot_scale(s0, cm)
    if not oos or oos <= 0:
        raise ValueError("invalid orientation knot scale")
    times: List[float] = []
    ctrls: List[Tuple[int, int, int]] = []
    for i in range(n):
        kt = struct.unpack_from("<H", s0, cm.knots_controls_offset + i * 2)[0]
        times.append(float(kt) / oos)
        a, b, c = struct.unpack_from(
            "<3H", s0, cm.knots_controls_offset + n * 2 + i * 6
        )
        ctrls.append((a, b, c))
    return oos, times, ctrls


def _quat_k16_tick(oos: float, t: float) -> int:
    """Map seconds → u16 knot tick (Granny D4nK16uC15u)."""
    if not oos or oos <= 0:
        raise ValueError("invalid orientation knot scale")
    u = int(round(max(0.0, float(t)) * oos))
    return max(0, min(65535, u))


def _allocate_free_quat_k16_tick(
    oos: float,
    times: Sequence[float],
    t: float,
    *,
    exclude_index: Optional[int] = None,
) -> Tuple[int, float]:
    """Pick a free u16 tick near t so pack won't collide with an existing knot.

    quat_k16 stores times as uint16 ticks. Two float times can round to the same
    tick and look like a 'missing' key after Save (85 knots, 84 unique times).
    """
    occupied: set[int] = set()
    for i, tv in enumerate(times):
        if exclude_index is not None and i == exclude_index:
            continue
        occupied.add(_quat_k16_tick(oos, tv))
    want = _quat_k16_tick(oos, t)
    if want not in occupied:
        return want, float(want) / oos
    # Spiral outward for a free tick
    for d in range(1, 65536):
        for cand in (want + d, want - d):
            if 0 <= cand <= 65535 and cand not in occupied:
                return cand, float(cand) / oos
    raise ValueError("no free quat_k16 tick — curve is full")


def _dedupe_quat_k16_by_tick(
    oos: float,
    times: List[float],
    ctrls: List[Tuple[int, int, int]],
) -> Tuple[List[float], List[Tuple[int, int, int]], List[int]]:
    """Sort by time and collapse keys that share a u16 tick (last value wins).

    Dense Blender samples often place a final key past duration * oos, which
    clamps to tick 65535 and collides with the previous knot.
    """
    n = len(times)
    if n != len(ctrls):
        raise ValueError("times/controls length mismatch")
    order = sorted(range(n), key=lambda i: times[i])
    out_t: List[float] = []
    out_c: List[Tuple[int, int, int]] = []
    out_ticks: List[int] = []
    for i in order:
        tick = _quat_k16_tick(oos, times[i])
        t_snap = float(tick) / oos
        if out_ticks and out_ticks[-1] == tick:
            out_t[-1] = t_snap
            out_c[-1] = ctrls[i]
            continue
        out_t.append(t_snap)
        out_c.append(ctrls[i])
        out_ticks.append(tick)
    return out_t, out_c, out_ticks


def _pack_quat_k16_blob(
    oos: float,
    times: List[float],
    ctrls: List[Tuple[int, int, int]],
) -> Tuple[bytes, int]:
    """Pack KnotsControls; collapses tick collisions (last wins). Returns (blob, knot_count)."""
    times, ctrls, ticks = _dedupe_quat_k16_by_tick(oos, times, ctrls)
    n = len(times)
    if len(set(ticks)) != n:
        raise ValueError(
            f"quat_k16 tick collision after pack ({n} knots, {len(set(ticks))} unique ticks)"
        )
    buf = bytearray(n * 8)
    for i, u in enumerate(ticks):
        struct.pack_into("<H", buf, i * 2, u)
        a, b, c = ctrls[i]
        struct.pack_into("<3H", buf, n * 2 + i * 6, a, b, c)
    return bytes(buf), n


def _add_quat_k16_key(
    s0: bytearray,
    cm: CurveMap,
    t: float,
    quat: Sequence[float],
    reloc_blob: bytearray,
) -> Dict[str, Any]:
    """Append a new D4nK16uC15u knot: grow KnotsControls at end of section0, retarget reloc."""
    from granny_quat import controls_from_quat, scale_offset_tables

    if cm.edit_mode != "quat_k16":
        raise ValueError(f"quat_k16 add needs quat_k16, got {cm.edit_mode}")
    if cm.object_offset < 0 or cm.knots_controls_offset is None:
        raise ValueError("quat_k16 curve unresolved")
    if len(quat) < 4:
        raise ValueError("add_key needs quat [x,y,z,w]")

    n = cm.knot_count
    oos, times, ctrls = _read_quat_k16_knots(s0, cm)
    # Snap to a free tick — float near-check is not enough (u16 quantization).
    tick, t_snap = _allocate_free_quat_k16_tick(oos, times, float(t))
    if any(_quat_k16_tick(oos, tv) == tick for tv in times):
        raise ValueError("key already exists at that tick — use set_key")

    # Widen selector if this quat would clamp under the current curve range
    widen = _ensure_quat_k16_range(s0, cm, [quat])
    # Re-read controls after possible rewrite
    oos, times, ctrls = _read_quat_k16_knots(s0, cm)
    scales, offsets = scale_offset_tables(_orient_selector(s0, cm))
    a, b, c = controls_from_quat(quat, scales, offsets)
    times.append(t_snap)
    ctrls.append((a, b, c))

    blob, new_n = _pack_quat_k16_blob(oos, times, ctrls)
    if new_n != n + 1:
        raise ValueError(f"quat_k16 add pack unexpected size ({n}+1 → {new_n})")
    old_toff = cm.knots_controls_offset
    old_end = old_toff + n * 8
    # Grow in place when the live KnotsControls blob is already at EOF
    # (typical after a prior append on this curve) — no extra orphan, no reloc
    # retarget. First grow from mid-section still appends once (one orphan).
    at_eof = old_end == len(s0)

    pointer_off = cm.object_offset + 12
    if at_eof:
        # Replace from old_toff through EOF with the larger blob.
        del s0[old_toff:]
        new_toff = old_toff
        s0.extend(blob)
        grown_in_place = True
    else:
        new_toff = len(s0)
        pad = (4 - (new_toff % 4)) % 4
        if pad:
            s0.extend(b"\x00" * pad)
            new_toff = len(s0)
        s0.extend(blob)
        grown_in_place = False
        if not _retarget_reloc(
            reloc_blob,
            pointer_off=pointer_off,
            old_toff=old_toff,
            new_toff=new_toff,
            section=0,
        ):
            raise ValueError(
                f"could not retarget KnotsControls reloc (ptr@{pointer_off} old={old_toff})"
            )

    struct.pack_into("<I", s0, cm.object_offset + 8, new_n * 4)
    if pointer_off + 8 <= len(s0):
        struct.pack_into("<Q", s0, pointer_off, new_toff)

    cm.knot_count = new_n
    cm.knots_controls_offset = new_toff
    cm.values_bytes = new_n * 8

    sorted_times = sorted(times)
    try:
        new_idx = sorted_times.index(t_snap)
    except ValueError:
        new_idx = new_n - 1

    return {
        "t": t_snap,
        "tick": tick,
        "index": new_idx,
        "knot_count": new_n,
        "edit_mode": cm.edit_mode,
        "grown": True,
        "grown_in_place": grown_in_place,
        "old_toff": old_toff,
        "new_toff": new_toff,
        "snapped": abs(t_snap - float(t)) > 1e-12,
        "quat": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
        **widen,
    }


def _delete_quat_k16_key(
    s0: bytearray,
    cm: CurveMap,
    index: int,
) -> Dict[str, Any]:
    """Remove one quat_k16 knot; rewrite blob in-place (same offset, smaller count)."""
    if cm.edit_mode != "quat_k16":
        raise ValueError(f"quat_k16 delete needs quat_k16, got {cm.edit_mode}")
    n = cm.knot_count
    if not (0 <= index < n):
        raise IndexError(f"delete_key index {index}")
    if n <= 1:
        raise ValueError("cannot delete last orientation key")
    assert cm.knots_controls_offset is not None

    oos, times, ctrls = _read_quat_k16_knots(s0, cm)
    times.pop(index)
    ctrls.pop(index)
    blob, new_n = _pack_quat_k16_blob(oos, times, ctrls)
    if new_n != n - 1:
        raise ValueError(f"quat_k16 delete pack unexpected size ({n}-1 → {new_n})")
    off = cm.knots_controls_offset
    s0[off : off + len(blob)] = blob
    for i in range(len(blob), n * 8):
        s0[off + i] = 0
    struct.pack_into("<I", s0, cm.object_offset + 8, new_n * 4)
    cm.knot_count = new_n
    cm.values_bytes = new_n * 8
    return {"index": index, "knot_count": new_n, "edit_mode": cm.edit_mode}


def _grow_dak32_channel(
    s0: bytearray,
    cm: CurveMap,
    reloc_blob: bytearray,
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    *,
    comps: int,
    capacity: Optional[int] = None,
    slack: int = 32,
) -> Dict[str, Any]:
    """Grow a DaK32fC32f curve: new knots+controls at S0 EOF, retarget both relocs."""
    if cm.edit_mode != "float_controls" or cm.object_offset < 0:
        raise ValueError("grow needs float_controls curve")
    if reloc_blob is None:
        raise ValueError("grow needs reloc_blob")
    n = len(times)
    if len(values) != n:
        raise ValueError("times/values length mismatch")
    cap = int(capacity) if capacity is not None else max(n + int(slack), n, float_curve_capacity(cm))
    if cap < n:
        cap = n
    order = sorted(range(n), key=lambda i: float(times[i]))
    times_s = [float(times[i]) for i in order]
    values_s = [list(values[i]) for i in order]

    knots = bytearray(cap * 4)
    ctrls = bytearray(cap * comps * 4)
    for i, tv in enumerate(times_s):
        struct.pack_into("<f", knots, i * 4, tv)
    fmt = f"<{comps}f"
    for i, vals in enumerate(values_s):
        packed = [float(vals[j]) if j < len(vals) else 0.0 for j in range(comps)]
        struct.pack_into(fmt, ctrls, i * comps * 4, *packed)

    old_k = cm.knots_controls_offset or 0
    old_v = cm.values_offset or 0
    new_k = _s0_append_aligned(s0, bytes(knots))
    new_v = _s0_append_aligned(s0, bytes(ctrls))

    obj = cm.object_offset
    if not _retarget_reloc(
        reloc_blob, pointer_off=obj + 8, old_toff=old_k, new_toff=new_k, section=0
    ):
        _append_reloc(reloc_blob, pointer_off=obj + 8, target_off=new_k, section=0)
    if not _retarget_reloc(
        reloc_blob, pointer_off=obj + 20, old_toff=old_v, new_toff=new_v, section=0
    ):
        _append_reloc(reloc_blob, pointer_off=obj + 20, target_off=new_v, section=0)

    struct.pack_into("<I", s0, obj + 4, n)
    struct.pack_into("<I", s0, obj + 16, n * comps)
    # Keep marshalled pointer fields at 0 (reloc carries target)
    if obj + 28 <= len(s0):
        struct.pack_into("<Q", s0, obj + 8, 0)
        struct.pack_into("<Q", s0, obj + 20, 0)

    cm.knot_count = n
    cm.knots_controls_offset = new_k
    cm.values_offset = new_v
    cm.values_bytes = cap * comps * 4
    cm.editable = True
    cm.edit_mode = "float_controls"
    cm.format = FMT_DA_K32F_C32F
    _set_curve_degree(s0, cm, CURVE_DEGREE_LINEAR)
    return {
        "grown": True,
        "knot_count": n,
        "capacity": cap,
        "knots_offset": new_k,
        "values_offset": new_v,
        "values_bytes": cm.values_bytes,
    }


def _pack_dak32_header(*, degree: int = CURVE_DEGREE_LINEAR, knot_count: int = 1, comps: int = 3) -> bytes:
    """28-byte CurveDataDaK32fC32f header; pointer fields left 0 (reloc targets)."""
    buf = bytearray(28)
    buf[0] = FMT_DA_K32F_C32F
    buf[1] = int(degree) & 0xFF
    struct.pack_into("<I", buf, 4, int(knot_count))
    struct.pack_into("<I", buf, 16, int(knot_count) * int(comps))
    return bytes(buf)


def _pack_quat_k16_header(
    *,
    selector: int,
    one_over_knot_scale: float,
    knot_count: int,
    degree: int = CURVE_DEGREE_LINEAR,
) -> bytes:
    """20-byte CurveDataD4nK16uC15u header; KnotsControls ptr left 0."""
    buf = bytearray(20)
    buf[0] = FMT_D4N_K16U_C15U
    buf[1] = int(degree) & 0xFF
    struct.pack_into("<H", buf, 2, int(selector) & 0xFFFF)
    struct.pack_into("<f", buf, 4, float(one_over_knot_scale))
    struct.pack_into("<I", buf, 8, int(knot_count) * 4)
    return bytes(buf)


def _find_quat_k16_reference(anim: AnimMap, s0: bytes) -> Tuple[int, float]:
    """Return (selector, one_over_knot_scale) from an existing quat_k16 track."""
    for t in anim.tracks:
        cm = t.orientation
        if cm.edit_mode == "quat_k16" and cm.object_offset >= 0:
            return _orient_selector(s0, cm), _orient_one_over_knot_scale(s0, cm)
    # Sensible defaults from Karlach Ankle_L probe
    return 152, 3287.70849609375


def upgrade_position_to_float(
    s0: bytearray,
    anim: AnimMap,
    bone: str,
    reloc0: bytearray,
    reloc1: bytearray,
    *,
    times: Optional[Sequence[float]] = None,
    values: Optional[Sequence[Sequence[float]]] = None,
    slack: int = 64,
) -> Dict[str, Any]:
    """Upgrade constant / identity / k16_offsets position → DaK32fC32f with slack."""
    track = _track_by_name(anim, bone)
    cm = track.position
    old_mode = cm.edit_mode
    old_obj = cm.object_offset
    # Caller-supplied times/values are keyframe samples → linear. Format-only
    # promote of existing D3K16 B-spline controls keeps degree 2.
    caller_samples = times is not None and values is not None and len(times) > 0

    if old_mode == "float_controls":
        return {"upgraded": False, "reason": "already float_controls", "bone": bone}

    if old_mode == "constant":
        xyz = list(struct.unpack_from("<3f", s0, cm.values_offset))
        if times is None or values is None or not times:
            times = [0.0]
            values = [xyz]
    elif old_mode == "k16_offsets" and cm.control_offsets_offset is not None:
        if cm.knots_controls_offset is not None and cm.knot_count > 0:
            _oos, k_times, k_pos = _read_d3k16_knots(s0, cm)
            if times is None or values is None or not times:
                times = list(k_times)
                values = [list(p) for p in k_pos]
            xyz = list(k_pos[0]) if k_pos else list(_d3k16_control_offsets(s0, cm))
        else:
            xyz = list(struct.unpack_from("<3f", s0, cm.control_offsets_offset))
            if times is None or values is None or not times:
                times = [0.0]
                values = [xyz]
    else:
        xyz = [0.0, 0.0, 0.0]
        if times is None or values is None or not times:
            times = [0.0]
            values = [xyz]

    if times is None or values is None or not times:
        times = [0.0]
        values = [xyz]
    n = len(times)
    if len(values) != n:
        raise ValueError("upgrade position times/values mismatch")
    cap = max(n + int(slack), n, 8)
    deg = CURVE_DEGREE_LINEAR if caller_samples else CURVE_DEGREE_BSPLINE

    header = _pack_dak32_header(degree=deg, knot_count=n, comps=3)
    knots = bytearray(cap * 4)
    ctrls = bytearray(cap * 12)
    order = sorted(range(n), key=lambda i: float(times[i]))
    for i, idx in enumerate(order):
        struct.pack_into("<f", knots, i * 4, float(times[idx]))
        v = values[idx]
        struct.pack_into(
            "<3f",
            ctrls,
            i * 12,
            float(v[0]),
            float(v[1]) if len(v) > 1 else 0.0,
            float(v[2]) if len(v) > 2 else 0.0,
        )

    new_obj = _s0_append_aligned(s0, header)
    new_k = _s0_append_aligned(s0, bytes(knots))
    new_v = _s0_append_aligned(s0, bytes(ctrls))
    # Fix live counts in header (already set) — ensure
    struct.pack_into("<I", s0, new_obj + 4, n)
    struct.pack_into("<I", s0, new_obj + 16, n * 3)

    _append_reloc(reloc0, pointer_off=new_obj + 8, target_off=new_k, section=0)
    _append_reloc(reloc0, pointer_off=new_obj + 20, target_off=new_v, section=0)

    if not _retarget_s1_curve_ptr(
        reloc1,
        track_offset=track.track_offset,
        field_off=OFF_POS_OBJ,
        new_s0_off=new_obj,
        old_s0_off=old_obj if old_obj >= 0 else None,
    ):
        _append_reloc(
            reloc1,
            pointer_off=track.track_offset + OFF_POS_OBJ,
            target_off=new_obj,
            section=0,
        )

    # Must retarget type pointer too — object alone is not enough for LSLib/Granny.
    type_ref = _retarget_curve_type_for_format(
        anim,
        reloc1,
        track=track,
        channel="position",
        want_format=FMT_DA_K32F_C32F,
    )

    cm.format = FMT_DA_K32F_C32F
    cm.degree = deg
    cm.object_offset = new_obj
    cm.knots_controls_offset = new_k
    cm.values_offset = new_v
    cm.knot_count = n
    cm.values_bytes = cap * 12
    cm.editable = True
    cm.edit_mode = "float_controls"
    cm.note = f"upgraded from {old_mode}"
    cm.control_offsets_offset = None
    return {
        "upgraded": True,
        "bone": bone,
        "channel": "position",
        "from": old_mode,
        "to": "float_controls",
        "object_offset": new_obj,
        "knot_count": n,
        "capacity": cap,
        "values_bytes": cm.values_bytes,
        "type_ref": list(type_ref) if type_ref else None,
        "degree": deg,
    }


def _retarget_orient_type_to_dak32(
    anim: AnimMap,
    reloc1: bytearray,
    track: TrackMap,
) -> Tuple[int, int]:
    """Point orientation type at CurveDataDaK32fC32f (reuse position float type in S4)."""
    ref = _find_type_ref_for_curve_format(
        anim, reloc1, channel="position", want_format=FMT_DA_K32F_C32F
    )
    if ref is None:
        ref = _find_type_ref_for_curve_format(
            anim, reloc1, channel="orientation", want_format=FMT_DA_K32F_C32F
        )
    if ref is None:
        raise ValueError(
            "no DaK32fC32f type in this GR2 — need a float position track (e.g. Dummy_Root)"
        )
    _retarget_s1_type_ptr(
        reloc1,
        track_offset=track.track_offset,
        field_off=OFF_ORIENT_TYPE,
        new_sec=ref[0],
        new_toff=ref[1],
    )
    return ref


def upgrade_orientation_to_float(
    s0: bytearray,
    anim: AnimMap,
    bone: str,
    reloc0: bytearray,
    reloc1: bytearray,
    *,
    times: Optional[Sequence[float]] = None,
    quats: Optional[Sequence[Sequence[float]]] = None,
    slack: int = 64,
) -> Dict[str, Any]:
    """Upgrade orientation → DaK32fC32f float quats at Degree=1 (game-safe keyframes).

    Compressed quat_k16 mid-frame eval lerps packed controls (not slerped quats), so
    Blender sample series must not stay as quat_k16 — even with Degree=1. Float
    linear keys match site/Blender scrub and Animation Preview / in-game playback.
    """
    track = _track_by_name(anim, bone)
    cm = track.orientation
    old_mode = cm.edit_mode
    old_obj = cm.object_offset
    if old_mode == "float_controls":
        return {"upgraded": False, "reason": "already float_controls", "bone": bone}

    if times is None or quats is None or not times:
        # Seed from current curve
        sampled = _sample_orientation(s0, cm)
        times = list(sampled.get("times") or [0.0])
        quats = [list(q) for q in (sampled.get("rotations") or [IDENTITY_QUAT[:]])]
    n = len(times)
    if len(quats) != n:
        raise ValueError("upgrade orientation float times/quats mismatch")
    # Clamp to animation duration when known
    dur = float(anim.duration) if anim.duration and anim.duration > 0 else None
    order = sorted(range(n), key=lambda i: float(times[i]))
    times_s: List[float] = []
    quats_s: List[List[float]] = []
    for i in order:
        t = float(times[i])
        if dur is not None:
            t = max(0.0, min(dur, t))
        times_s.append(t)
        q = [float(x) for x in quats[i][:4]]
        while len(q) < 4:
            q.append(0.0 if len(q) < 3 else 1.0)
        quats_s.append(q)
    # Deduplicate times after clamp (keep last)
    out_t: List[float] = []
    out_q: List[List[float]] = []
    for t, q in zip(times_s, quats_s):
        if out_t and abs(out_t[-1] - t) < 1e-9:
            out_t[-1] = t
            out_q[-1] = q
        else:
            out_t.append(t)
            out_q.append(q)
    times_s, quats_s = out_t, out_q
    n = len(times_s)
    cap = max(n + int(slack), n, 8)

    header = _pack_dak32_header(
        degree=CURVE_DEGREE_LINEAR, knot_count=n, comps=4
    )
    knots = bytearray(cap * 4)
    ctrls = bytearray(cap * 16)
    for i, (tv, q) in enumerate(zip(times_s, quats_s)):
        struct.pack_into("<f", knots, i * 4, tv)
        struct.pack_into("<4f", ctrls, i * 16, q[0], q[1], q[2], q[3])

    new_obj = _s0_append_aligned(s0, header)
    new_k = _s0_append_aligned(s0, bytes(knots))
    new_v = _s0_append_aligned(s0, bytes(ctrls))
    struct.pack_into("<I", s0, new_obj + 4, n)
    struct.pack_into("<I", s0, new_obj + 16, n * 4)

    _append_reloc(reloc0, pointer_off=new_obj + 8, target_off=new_k, section=0)
    _append_reloc(reloc0, pointer_off=new_obj + 20, target_off=new_v, section=0)

    if not _retarget_s1_curve_ptr(
        reloc1,
        track_offset=track.track_offset,
        field_off=OFF_ORIENT_OBJ,
        new_s0_off=new_obj,
        old_s0_off=old_obj if old_obj >= 0 else None,
    ):
        _append_reloc(
            reloc1,
            pointer_off=track.track_offset + OFF_ORIENT_OBJ,
            target_off=new_obj,
            section=0,
        )

    type_ref = _retarget_orient_type_to_dak32(anim, reloc1, track)

    cm.format = FMT_DA_K32F_C32F
    cm.degree = CURVE_DEGREE_LINEAR
    cm.object_offset = new_obj
    cm.knots_controls_offset = new_k
    cm.values_offset = new_v
    cm.knot_count = n
    cm.values_bytes = cap * 16
    cm.editable = True
    cm.edit_mode = "float_controls"
    cm.note = f"upgraded from {old_mode} → float degree-1"
    cm.control_scales_offset = None
    track.orientation_format = FMT_DA_K32F_C32F
    return {
        "upgraded": True,
        "bone": bone,
        "channel": "orientation",
        "from": old_mode,
        "to": "float_controls",
        "object_offset": new_obj,
        "knot_count": n,
        "capacity": cap,
        "degree": CURVE_DEGREE_LINEAR,
        "type_ref": list(type_ref),
    }


def upgrade_orientation_to_quat_k16(
    s0: bytearray,
    anim: AnimMap,
    bone: str,
    reloc0: bytearray,
    reloc1: bytearray,
    *,
    times: Optional[Sequence[float]] = None,
    quats: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Upgrade identity / constant orientation → D4nK16uC15u."""
    from granny_quat import (
        controls_from_quat_series,
        find_selector_for_quats,
        scale_offset_tables,
    )

    track = _track_by_name(anim, bone)
    cm = track.orientation
    old_mode = cm.edit_mode
    old_obj = cm.object_offset
    caller_samples = times is not None and quats is not None and len(times) > 0
    if old_mode == "quat_k16":
        return {"upgraded": False, "reason": "already quat_k16", "bone": bone}
    if old_mode == "float_controls":
        return {"upgraded": False, "reason": "already float_controls", "bone": bone}

    if old_mode == "constant" and cm.values_offset:
        seed = list(struct.unpack_from("<4f", s0, cm.values_offset))
    else:
        seed = IDENTITY_QUAT[:]

    if times is None or quats is None or not times:
        times = [0.0]
        quats = [seed]
    n = len(times)
    if len(quats) != n:
        raise ValueError("upgrade orientation times/quats mismatch")

    ref_sel, ref_oos = _find_quat_k16_reference(anim, s0)
    # Prefer a selector that covers all seed quats
    try:
        sel = find_selector_for_quats([list(q) for q in quats], min_dot=0.98)
    except Exception:
        sel = ref_sel
    oos = float(ref_oos) if ref_oos else 3287.70849609375
    scales, offsets = scale_offset_tables(sel)
    deg = CURVE_DEGREE_LINEAR if caller_samples else CURVE_DEGREE_BSPLINE

    order = sorted(range(n), key=lambda i: float(times[i]))
    times_s = [float(times[i]) for i in order]
    quats_s = [quats[i] for i in order]
    ctrls = controls_from_quat_series(quats_s, scales, offsets)

    blob, packed_n = _pack_quat_k16_blob(oos, times_s, ctrls)
    header = _pack_quat_k16_header(
        selector=sel, one_over_knot_scale=oos, knot_count=packed_n, degree=deg
    )
    new_obj = _s0_append_aligned(s0, header)
    new_kc = _s0_append_aligned(s0, blob)

    _append_reloc(reloc0, pointer_off=new_obj + 12, target_off=new_kc, section=0)
    if not _retarget_s1_curve_ptr(
        reloc1,
        track_offset=track.track_offset,
        field_off=OFF_ORIENT_OBJ,
        new_s0_off=new_obj,
        old_s0_off=old_obj if old_obj >= 0 else None,
    ):
        _append_reloc(
            reloc1,
            pointer_off=track.track_offset + OFF_ORIENT_OBJ,
            target_off=new_obj,
            section=0,
        )

    type_ref = _retarget_curve_type_for_format(
        anim,
        reloc1,
        track=track,
        channel="orientation",
        want_format=FMT_D4N_K16U_C15U,
    )

    cm.format = FMT_D4N_K16U_C15U
    cm.degree = deg
    cm.object_offset = new_obj
    cm.control_scales_offset = new_obj + 2
    cm.values_offset = new_obj + 4
    cm.knots_controls_offset = new_kc
    cm.knot_count = packed_n
    cm.values_bytes = packed_n * 8
    cm.editable = True
    cm.edit_mode = "quat_k16"
    cm.note = f"upgraded from {old_mode}"
    track.orientation_format = FMT_D4N_K16U_C15U
    return {
        "upgraded": True,
        "bone": bone,
        "channel": "orientation",
        "from": old_mode,
        "to": "quat_k16",
        "object_offset": new_obj,
        "knot_count": packed_n,
        "selector": sel,
        "type_ref": list(type_ref) if type_ref else None,
        "degree": deg,
    }


def replace_position_curve(
    s0: bytearray,
    anim: AnimMap,
    bone: str,
    reloc0: bytearray,
    reloc1: bytearray,
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    *,
    slack: int = 64,
) -> Dict[str, Any]:
    """Write a full position curve in one shot (upgrade or rewrite/grow)."""
    track = _track_by_name(anim, bone)
    cm = track.position
    if not cm.editable or cm.edit_mode == "none":
        return {"replaced": False, "reason": "not editable", "bone": bone, "channel": "position"}
    times_l = [float(t) for t in times]
    values_l = [list(map(float, v)) for v in values]
    if not times_l:
        times_l = [0.0]
        values_l = [[0.0, 0.0, 0.0]]
    if cm.edit_mode != "float_controls":
        return upgrade_position_to_float(
            s0,
            anim,
            bone,
            reloc0,
            reloc1,
            times=times_l,
            values=values_l,
            slack=slack,
        )
    n = len(times_l)
    # Clamp sample times to animation duration (float keys past Duration confuse runtime)
    dur = float(anim.duration) if anim.duration and anim.duration > 0 else None
    if dur is not None:
        times_l = [max(0.0, min(dur, t)) for t in times_l]
    cap = float_curve_capacity(cm)
    if n > cap:
        meta = _grow_dak32_channel(
            s0, cm, reloc0, times_l, values_l, comps=3, slack=max(32, slack)
        )
        _set_curve_degree(s0, cm, CURVE_DEGREE_LINEAR)
        return {
            "replaced": True,
            "bone": bone,
            "channel": "position",
            "knots": n,
            "degree": CURVE_DEGREE_LINEAR,
            **meta,
        }
    _rewrite_dak32_curve(
        s0, cm, times_l, values_l, comps=3, degree=CURVE_DEGREE_LINEAR
    )
    return {
        "replaced": True,
        "bone": bone,
        "channel": "position",
        "knots": n,
        "grown": False,
        "degree": CURVE_DEGREE_LINEAR,
    }


def replace_orientation_curve(
    s0: bytearray,
    anim: AnimMap,
    bone: str,
    reloc0: bytearray,
    reloc1: bytearray,
    times: Sequence[float],
    quats: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Write a full orientation curve (game-safe: multi-key → float Degree=1)."""
    track = _track_by_name(anim, bone)
    cm = track.orientation
    if not cm.editable or cm.edit_mode == "none":
        return {
            "replaced": False,
            "reason": "not editable",
            "bone": bone,
            "channel": "orientation",
        }
    times_l = [float(t) for t in times]
    quats_l = [list(map(float, q)) for q in quats]
    if not times_l:
        times_l = [0.0]
        quats_l = [IDENTITY_QUAT[:]]

    # Multi-key sample series must be float Degree=1. quat_k16 mid-frame eval
    # lerps compressed controls (site/Blender slerp decoded quats) → in-game glitches.
    if len(times_l) > 1 and cm.edit_mode != "float_controls":
        meta = upgrade_orientation_to_float(
            s0,
            anim,
            bone,
            reloc0,
            reloc1,
            times=times_l,
            quats=quats_l,
        )
        return {
            "replaced": bool(meta.get("upgraded")),
            "bone": bone,
            "channel": "orientation",
            "knots": meta.get("knot_count", len(times_l)),
            "format": "float_controls",
            **{k: v for k, v in meta.items() if k not in ("bone", "channel")},
        }

    if cm.edit_mode == "float_controls":
        n = len(times_l)
        # Clamp to duration
        dur = float(anim.duration) if anim.duration and anim.duration > 0 else None
        if dur is not None:
            times_l = [max(0.0, min(dur, t)) for t in times_l]
        cap = float_curve_capacity(cm)
        if n > cap:
            meta = _grow_dak32_channel(
                s0, cm, reloc0, times_l, quats_l, comps=4, slack=64
            )
            _set_curve_degree(s0, cm, CURVE_DEGREE_LINEAR)
            return {
                "replaced": True,
                "bone": bone,
                "channel": "orientation",
                "knots": n,
                "degree": CURVE_DEGREE_LINEAR,
                **meta,
            }
        _rewrite_dak32_curve(
            s0, cm, times_l, quats_l, comps=4, degree=CURVE_DEGREE_LINEAR
        )
        return {
            "replaced": True,
            "bone": bone,
            "channel": "orientation",
            "knots": n,
            "grown": False,
            "degree": CURVE_DEGREE_LINEAR,
        }

    if cm.edit_mode != "quat_k16":
        # Single-key or sparse identity/constant — keep compressed path
        return upgrade_orientation_to_quat_k16(
            s0, anim, bone, reloc0, reloc1, times=times_l, quats=quats_l
        )

    # Single-key quat_k16 rewrite (viewport tweak) — keep compressed format
    from granny_quat import (
        controls_from_quat_series,
        find_selector_for_quats,
        quat_encode_quality,
        scale_offset_tables,
    )

    n = len(times_l)
    if len(quats_l) != n:
        raise ValueError("orientation times/quats mismatch")
    ref_sel, ref_oos = _find_quat_k16_reference(anim, s0)
    cur_sel = _orient_selector(s0, cm) if cm.object_offset >= 0 else ref_sel
    scales_cur, offsets_cur = scale_offset_tables(cur_sel)
    worst = 1.0
    for q in quats_l:
        d, _ = quat_encode_quality(q, scales_cur, offsets_cur)
        worst = min(worst, d)
    if worst >= 0.98:
        sel = cur_sel
    else:
        try:
            sel = find_selector_for_quats([list(q) for q in quats_l], min_dot=0.98)
        except Exception:
            sel = ref_sel
    oos = float(ref_oos) if ref_oos else 3287.70849609375
    if cm.values_offset and cm.edit_mode == "quat_k16":
        try:
            oos = float(struct.unpack_from("<f", s0, cm.values_offset)[0]) or oos
        except Exception:
            pass
    scales, offsets = scale_offset_tables(sel)
    order = sorted(range(n), key=lambda i: times_l[i])
    times_s = [times_l[i] for i in order]
    quats_s = [quats_l[i] for i in order]
    ctrls = controls_from_quat_series(quats_s, scales, offsets)
    blob, packed_n = _pack_quat_k16_blob(oos, times_s, ctrls)
    header = _pack_quat_k16_header(
        selector=sel,
        one_over_knot_scale=oos,
        knot_count=packed_n,
        degree=CURVE_DEGREE_LINEAR,
    )

    cap = float_curve_capacity(cm)
    kc_off = cm.knots_controls_offset
    obj_off = cm.object_offset
    if (
        packed_n <= cap
        and kc_off is not None
        and kc_off >= 0
        and obj_off >= 0
        and obj_off + 20 <= len(s0)
        and kc_off + packed_n * 8 <= len(s0)
    ):
        # Keep KnotsControls reloc at obj+12; rewrite header fields + blob in place
        s0[obj_off : obj_off + 12] = header[:12]
        old_bytes = max(cm.knot_count * 8, packed_n * 8)
        cap_bytes = max(cap * 8, old_bytes)
        end = min(kc_off + cap_bytes, len(s0))
        # Clear previous knot region then write new blob
        clear_n = min(end - kc_off, max(old_bytes, packed_n * 8))
        if clear_n > 0:
            s0[kc_off : kc_off + clear_n] = b"\x00" * clear_n
        s0[kc_off : kc_off + len(blob)] = blob
        cm.knot_count = packed_n
        cm.values_bytes = max(cm.values_bytes, packed_n * 8, cap * 8)
        cm.control_scales_offset = obj_off + 2
        cm.values_offset = obj_off + 4
        cm.degree = CURVE_DEGREE_LINEAR
        return {
            "replaced": True,
            "bone": bone,
            "channel": "orientation",
            "knots": packed_n,
            "selector": sel,
            "grown": False,
            "degree": CURVE_DEGREE_LINEAR,
        }

    old_obj = cm.object_offset
    new_obj = _s0_append_aligned(s0, header)
    new_kc = _s0_append_aligned(s0, blob)
    _append_reloc(reloc0, pointer_off=new_obj + 12, target_off=new_kc, section=0)
    if not _retarget_s1_curve_ptr(
        reloc1,
        track_offset=track.track_offset,
        field_off=OFF_ORIENT_OBJ,
        new_s0_off=new_obj,
        old_s0_off=old_obj if old_obj >= 0 else None,
    ):
        _append_reloc(
            reloc1,
            pointer_off=track.track_offset + OFF_ORIENT_OBJ,
            target_off=new_obj,
            section=0,
        )
    cm.format = FMT_D4N_K16U_C15U
    cm.degree = CURVE_DEGREE_LINEAR
    cm.object_offset = new_obj
    cm.control_scales_offset = new_obj + 2
    cm.values_offset = new_obj + 4
    cm.knots_controls_offset = new_kc
    cm.knot_count = packed_n
    cm.values_bytes = packed_n * 8
    cm.editable = True
    cm.edit_mode = "quat_k16"
    track.orientation_format = FMT_D4N_K16U_C15U
    return {
        "replaced": True,
        "bone": bone,
        "channel": "orientation",
        "knots": packed_n,
        "selector": sel,
        "grown": True,
        "degree": CURVE_DEGREE_LINEAR,
    }


def apply_curve_upgrades(
    s0: bytearray,
    s1: bytearray,
    anim: AnimMap,
    reloc0: bytearray,
    reloc1: bytearray,
    upgrades: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply a list of upgrade specs: {bone, channel, times?, values?/quats?}."""
    results: List[Dict[str, Any]] = []
    for spec in upgrades:
        bone = spec["bone"]
        ch = spec.get("channel") or "position"
        if ch == "position":
            results.append(
                upgrade_position_to_float(
                    s0,
                    anim,
                    bone,
                    reloc0,
                    reloc1,
                    times=spec.get("times"),
                    values=spec.get("values"),
                    slack=int(spec.get("slack") or 64),
                )
            )
        elif ch == "orientation":
            times = spec.get("times")
            quats = spec.get("quats") or spec.get("values")
            # Multi-key sample series must be float Degree=1. quat_k16/k8 mid-frame
            # eval does not match Blender/site slerp — in-game glitches / stuck spine.
            prefer_float = bool(spec.get("prefer_float"))
            if prefer_float or (times is not None and len(times) > 1):
                results.append(
                    upgrade_orientation_to_float(
                        s0,
                        anim,
                        bone,
                        reloc0,
                        reloc1,
                        times=times,
                        quats=quats,
                        slack=int(spec.get("slack") or 64),
                    )
                )
            else:
                results.append(
                    upgrade_orientation_to_quat_k16(
                        s0,
                        anim,
                        bone,
                        reloc0,
                        reloc1,
                        times=times,
                        quats=quats,
                    )
                )
        else:
            raise ValueError(f"unsupported upgrade channel {ch}")
    return results


def _add_float_channel_key(
    s0: bytearray,
    cm: CurveMap,
    t: float,
    values: Sequence[float],
    *,
    comps: int,
    reloc_blob: Optional[bytearray] = None,
) -> Dict[str, Any]:
    """Insert a DAK32f knot; grow buffer when capacity is full."""
    if cm.edit_mode != "float_controls":
        raise ValueError(f"add_key needs float_controls, got {cm.edit_mode}")
    n = cm.knot_count
    cap = float_curve_capacity(cm)
    times = _knot_times_dak32(s0, cm)
    vals = [
        list(struct.unpack_from(f"<{comps}f", s0, cm.values_offset + i * comps * 4))
        for i in range(n)
    ]
    for tv in times:
        if abs(tv - float(t)) < 1e-4:
            raise ValueError("key already exists at that time — use set_key")
    packed = [float(values[j]) if j < len(values) else 0.0 for j in range(comps)]
    times.append(float(t))
    vals.append(packed)
    grown = False
    if n >= cap:
        if reloc_blob is None:
            raise ValueError(f"no slack to add key (capacity {cap}) and no reloc for grow")
        meta = _grow_dak32_channel(
            s0, cm, reloc_blob, times, vals, comps=comps, slack=32
        )
        grown = True
        sorted_times = sorted(times)
        try:
            new_idx = sorted_times.index(float(t))
        except ValueError:
            new_idx = len(sorted_times) - 1
        return {
            "t": float(t),
            "index": new_idx,
            "knot_count": cm.knot_count,
            "edit_mode": cm.edit_mode,
            "grown": grown,
            **{k: meta[k] for k in ("capacity", "values_bytes") if k in meta},
        }
    _rewrite_dak32_curve(s0, cm, times, vals, comps=comps)
    sorted_times = sorted(times)
    try:
        new_idx = sorted_times.index(float(t))
    except ValueError:
        new_idx = len(sorted_times) - 1
    return {
        "t": float(t),
        "index": new_idx,
        "knot_count": cm.knot_count,
        "edit_mode": cm.edit_mode,
        "grown": False,
    }


def _delete_float_channel_key(
    s0: bytearray,
    cm: CurveMap,
    index: int,
    *,
    comps: int,
) -> Dict[str, Any]:
    if cm.edit_mode != "float_controls":
        raise ValueError(f"delete_key needs float_controls, got {cm.edit_mode}")
    n = cm.knot_count
    if not (0 <= index < n):
        raise IndexError(f"delete_key index {index}")
    if n <= 1:
        raise ValueError("cannot delete last key")
    times = _knot_times_dak32(s0, cm)
    vals = [
        list(struct.unpack_from(f"<{comps}f", s0, cm.values_offset + i * comps * 4))
        for i in range(n)
    ]
    times.pop(index)
    vals.pop(index)
    _rewrite_dak32_curve(s0, cm, times, vals, comps=comps)
    return {"index": index, "knot_count": cm.knot_count, "edit_mode": cm.edit_mode}


def _resolve_key_index(
    times: Sequence[float],
    *,
    index: Any = None,
    from_t: Any = None,
    tol: float = 1e-4,
) -> int:
    """Resolve a key by from_t (preferred) or index. Raises IndexError if missing."""
    n = len(times)
    if from_t is not None and n:
        target = float(from_t)
        best = -1
        best_d = tol
        for i, tv in enumerate(times):
            d = abs(float(tv) - target)
            if d <= best_d:
                best_d = d
                best = i
        if best >= 0:
            return best
    if index is not None:
        idx = int(index)
        if 0 <= idx < n:
            return idx
    raise IndexError(
        f"key not found (index={index!r} from_t={from_t!r} n={n})"
    )


def apply_key_ops(
    s0: bytearray,
    anim: AnimMap,
    bone: str,
    ops: Sequence[Dict[str, Any]],
    *,
    reloc_blob: Optional[bytearray] = None,
) -> List[Dict[str, Any]]:
    """Apply keyframe ops (including growing quat_k16). Returns applied op records.

    Ops may include channel: position (default) | orientation | scale.
    Orientation ops use quat [x,y,z,w]; scale uses xyz.
    Prefer from_t over index so multi-key moves survive channel sorts.
    Pass reloc_blob (mutable section-0 relocations) when adding quat_k16 keys.
    """
    track = _track_by_name(anim, bone)
    applied: List[Dict[str, Any]] = []

    # Split by channel — position keeps existing float-key rewrite path
    pos_ops: List[Dict[str, Any]] = []
    for op in ops:
        ch = op.get("channel") or "position"
        kind = op.get("op")
        if ch == "orientation":
            cm = track.orientation
            if kind == "add_key":
                quat = op.get("quat") or op.get("xyzw") or op.get("xyz")
                if quat is None or len(quat) < 4:
                    raise ValueError("orientation add_key needs quat [x,y,z,w]")
                if cm.edit_mode == "quat_k16":
                    if reloc_blob is None:
                        raise ValueError("quat_k16 add_key needs reloc_blob")
                    rec = _add_quat_k16_key(s0, cm, float(op["t"]), quat, reloc_blob)
                elif cm.edit_mode == "float_controls":
                    rec = _add_float_channel_key(
                        s0, cm, float(op["t"]), quat, comps=4, reloc_blob=reloc_blob
                    )
                else:
                    raise ValueError(f"orientation add_key unsupported for {cm.edit_mode}")
                applied.append({**rec, "op": kind, "channel": "orientation", "bone": bone})
                continue

            # Resolve against *current* channel times (after prior ops in this batch)
            if cm.edit_mode == "quat_k16":
                _oos, cur_times, _ctrls = _read_quat_k16_knots(s0, cm)
            elif cm.edit_mode == "float_controls":
                cur_times = _knot_times_dak32(s0, cm)
            elif cm.edit_mode == "constant":
                cur_times = [0.0]
            else:
                raise ValueError(f"orientation does not support {kind} ({cm.edit_mode})")

            idx = None
            if kind != "add_key" and cm.edit_mode != "constant":
                idx = _resolve_key_index(
                    cur_times, index=op.get("index"), from_t=op.get("from_t")
                )

            if kind == "move_key":
                if cm.edit_mode == "quat_k16":
                    rec = _move_quat_k16_key(s0, cm, int(idx), float(op["t"]))
                else:
                    rec = _move_float_channel_key(
                        s0, cm, int(idx), float(op["t"]), comps=4
                    )
                applied.append({**rec, "op": kind, "channel": "orientation", "bone": bone})
            elif kind == "delete_key":
                if cm.edit_mode == "quat_k16":
                    rec = _delete_quat_k16_key(s0, cm, int(idx))
                elif cm.edit_mode == "float_controls":
                    rec = _delete_float_channel_key(s0, cm, int(idx), comps=4)
                else:
                    raise ValueError(f"orientation delete_key unsupported for {cm.edit_mode}")
                applied.append({**rec, "op": kind, "channel": "orientation", "bone": bone})
            else:
                quat = op.get("quat") or op.get("xyzw") or op.get("xyz")
                if quat is None or len(quat) < 4:
                    raise ValueError("orientation op needs quat [x,y,z,w]")
                if kind in ("set_key", "set"):
                    rec = patch_orientation(s0, anim, bone, "set", quat, index=idx)
                elif kind in ("offset_key", "offset", "multiply_key", "multiply"):
                    rec = patch_orientation(s0, anim, bone, "multiply", quat, index=idx)
                else:
                    raise ValueError(f"orientation does not support {kind}")
                applied.append({**rec, "op": kind, "bone": bone})
        elif ch == "scale":
            cm = track.scale
            if kind == "add_key":
                xyz = op.get("xyz") or op.get("scale") or [1, 1, 1]
                if cm.edit_mode != "float_controls":
                    raise ValueError(f"scale add_key unsupported for {cm.edit_mode}")
                rec = _add_float_channel_key(
                    s0, cm, float(op["t"]), xyz, comps=3, reloc_blob=reloc_blob
                )
                applied.append({**rec, "op": kind, "channel": "scale", "bone": bone})
                continue

            if cm.edit_mode == "float_controls":
                cur_times = _knot_times_dak32(s0, cm)
                idx = _resolve_key_index(
                    cur_times, index=op.get("index"), from_t=op.get("from_t")
                )
            else:
                idx = op.get("index")

            if kind == "move_key":
                rec = _move_float_channel_key(
                    s0, cm, int(idx), float(op["t"]), comps=3
                )
                applied.append({**rec, "op": kind, "channel": "scale", "bone": bone})
            elif kind == "delete_key":
                if cm.edit_mode != "float_controls":
                    raise ValueError(f"scale delete_key unsupported for {cm.edit_mode}")
                rec = _delete_float_channel_key(s0, cm, int(idx), comps=3)
                applied.append({**rec, "op": kind, "channel": "scale", "bone": bone})
            else:
                xyz = op.get("xyz") or op.get("scale") or [1, 1, 1]
                if kind in ("set_key", "set"):
                    rec = patch_scale(s0, anim, bone, "set", xyz, index=idx)
                elif kind in ("multiply_key", "multiply"):
                    rec = patch_scale(s0, anim, bone, "multiply", xyz, index=idx)
                elif kind in ("offset_key", "offset"):
                    rec = patch_scale(s0, anim, bone, "offset", xyz, index=idx)
                else:
                    raise ValueError(f"scale does not support {kind}")
                applied.append({**rec, "op": kind, "bone": bone})
        else:
            pos_ops.append(op)

    if not pos_ops:
        return applied

    cm = track.position
    if cm.edit_mode in ("constant", "k16_offsets"):
        for op in pos_ops:
            kind = op.get("op")
            if kind in ("set_key", "set"):
                xyz = op.get("xyz") or [0, 0, 0]
                rec = patch_position(s0, anim, bone, "set", xyz)
                applied.append({**rec, "op": kind, "bone": bone})
            elif kind in ("offset_key", "offset"):
                xyz = op.get("xyz") or [0, 0, 0]
                rec = patch_position(s0, anim, bone, "offset", xyz)
                applied.append({**rec, "op": kind, "bone": bone})
            else:
                raise ValueError(f"{cm.edit_mode} only supports set/offset, not {kind}")
        return applied

    if cm.edit_mode != "float_controls":
        raise ValueError(f"Bone {bone} does not support key ops ({cm.edit_mode})")

    times = _knot_times_dak32(s0, cm)
    positions = [
        list(struct.unpack_from("<3f", s0, cm.values_offset + i * 12))
        for i in range(cm.knot_count)
    ]
    cap = float_curve_capacity(cm)

    for op in pos_ops:
        kind = op.get("op")
        if kind == "add_key":
            t = float(op["t"])
            xyz = op.get("xyz")
            if xyz is None:
                xyz = eval_position_at(s0, cm, t)
            if len(times) >= cap:
                if reloc_blob is None:
                    raise ValueError(f"no slack to add key (capacity {cap})")
                times.append(t)
                positions.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                meta = _grow_dak32_channel(
                    s0, cm, reloc_blob, times, positions, comps=3, slack=32
                )
                cap = int(meta.get("capacity") or float_curve_capacity(cm))
                times = _knot_times_dak32(s0, cm)
                positions = [
                    list(struct.unpack_from("<3f", s0, cm.values_offset + i * 12))
                    for i in range(cm.knot_count)
                ]
                applied.append(
                    {
                        "op": kind,
                        "t": t,
                        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                        "knot_count": cm.knot_count,
                        "bone": bone,
                        "grown": True,
                        "capacity": cap,
                    }
                )
            else:
                times.append(t)
                positions.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                applied.append(
                    {
                        "op": kind,
                        "t": t,
                        "xyz": positions[-1],
                        "knot_count": len(times),
                        "bone": bone,
                    }
                )
            continue

        idx = _resolve_key_index(
            times, index=op.get("index"), from_t=op.get("from_t")
        )
        if kind == "set_key":
            xyz = list(op["xyz"])
            positions[idx] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            applied.append({"op": kind, "index": idx, "xyz": positions[idx], "bone": bone})
        elif kind == "move_key":
            times[idx] = float(op["t"])
            applied.append({"op": kind, "index": idx, "t": times[idx], "bone": bone})
        elif kind == "delete_key":
            if len(times) <= 1:
                raise ValueError("cannot delete last key")
            times.pop(idx)
            positions.pop(idx)
            applied.append({"op": kind, "index": idx, "knot_count": len(times), "bone": bone})
        elif kind == "offset_key":
            d = op.get("xyz") or [0, 0, 0]
            positions[idx] = [
                positions[idx][0] + float(d[0]),
                positions[idx][1] + float(d[1]),
                positions[idx][2] + float(d[2]),
            ]
            applied.append({"op": kind, "index": idx, "xyz": positions[idx], "bone": bone})
        else:
            raise ValueError(f"unknown key op: {kind}")

    _rewrite_float_curve(s0, cm, times, positions)
    return applied


def apply_keys_to_dump(
    src_dump: Path,
    *,
    bone: Optional[str] = None,
    ops: Optional[Sequence[Dict[str, Any]]] = None,
    edits: Optional[Sequence[Dict[str, Any]]] = None,
    out_dump: Optional[Path] = None,
    base_gr2: Optional[Path] = None,
    overwrite: bool = True,
    snapshot: bool = True,
) -> Dict[str, Any]:
    """Apply key ops and write dump files.

    By default overwrites src_dump in place (Save dump). Pass overwrite=False
    or an explicit out_dump to write a new folder instead.
    """
    from paths import make_run_dirs

    src_dump = Path(src_dump)
    gr2, anim, section0 = load_dump_gr2(src_dump, base_gr2)

    groups: List[Dict[str, Any]] = []
    if edits:
        for g in edits:
            if not isinstance(g, dict):
                continue
            b = g.get("bone")
            gops = g.get("ops") or []
            if b and gops:
                groups.append({"bone": b, "ops": list(gops)})
    if not groups:
        if not bone or not ops:
            raise ValueError("bone+ops or edits required")
        groups = [{"bone": bone, "ops": list(ops)}]

    primary = groups[0]["bone"]
    bone_names = [g["bone"] for g in groups]
    cap_by_bone: Dict[str, Dict[str, int]] = {}
    for g in groups:
        tr = _track_by_name(anim, g["bone"])
        cap_by_bone[g["bone"]] = {
            "pos": tr.position.values_bytes,
            "ori": tr.orientation.values_bytes,
            "scl": tr.scale.values_bytes,
        }

    reloc0 = bytearray(gr2.sections[0].relocations or b"")
    needs_grow = False
    for g in groups:
        tr = _track_by_name(anim, g["bone"])
        for op in g["ops"]:
            if (
                (op.get("channel") or "position") == "orientation"
                and op.get("op") == "add_key"
                and tr.orientation.edit_mode == "quat_k16"
            ):
                needs_grow = True
                break
        if needs_grow:
            break
    if needs_grow and not reloc0:
        raise ValueError("section-0 relocations missing — cannot grow quat_k16")

    all_applied: List[Dict[str, Any]] = []
    edit_records: List[Dict[str, Any]] = []
    for g in groups:
        applied = apply_key_ops(
            section0,
            anim,
            g["bone"],
            g["ops"],
            reloc_blob=reloc0 if reloc0 else None,
        )
        all_applied.extend(applied)
        edit_records.append({"bone": g["bone"], "key_ops": applied})

    if reloc0:
        gr2.sections[0].relocations = bytes(reloc0)
    gr2.sections[0].uncompressed = bytes(section0)
    gr2.sections[0].header.uncompressed_size = len(section0)

    if out_dump is not None:
        out_dump = Path(out_dump)
        out_dump.mkdir(parents=True, exist_ok=True)
        run_id = out_dump.name
        in_place = out_dump.resolve() == src_dump.resolve()
    elif overwrite:
        out_dump = src_dump
        run_id = src_dump.name
        in_place = True
    else:
        run_id, out_dump, _ = make_run_dirs("decode")
        in_place = False

    if snapshot and in_place:
        try:
            from dump_history import label_for_bones, snapshot_dump

            op_kinds = []
            for g in groups:
                for op in g.get("ops") or []:
                    k = op.get("op")
                    if k and k not in op_kinds:
                        op_kinds.append(k)
            snapshot_dump(
                src_dump,
                kind="save",
                label=label_for_bones(bone_names),
                bones=bone_names,
                ops_summary=", ".join(op_kinds),
            )
        except Exception:
            pass

    # section_0 always rewritten from patched memory; other sections only
    # copied when writing to a different folder.
    (out_dump / "section_0.bin").write_bytes(section0)
    if not in_place:
        for p in src_dump.glob("section_*.bin"):
            if p.name == "section_0.bin":
                continue
            (out_dump / p.name).write_bytes(p.read_bytes())
    for i, sec in enumerate(gr2.sections):
        if sec.relocations is not None:
            (out_dump / f"section_{i}.relocations.bin").write_bytes(sec.relocations)
        elif not in_place and (src_dump / f"section_{i}.relocations.bin").is_file():
            (out_dump / f"section_{i}.relocations.bin").write_bytes(
                (src_dump / f"section_{i}.relocations.bin").read_bytes()
            )
        if sec.mixed_marshalling is not None:
            (out_dump / f"section_{i}.mm.bin").write_bytes(sec.mixed_marshalling)
        elif not in_place and (src_dump / f"section_{i}.mm.bin").is_file():
            (out_dump / f"section_{i}.mm.bin").write_bytes(
                (src_dump / f"section_{i}.mm.bin").read_bytes()
            )
    if not in_place:
        for name in ("SOURCE.txt", "strings.txt", "decoded.json", "decoded.dae", "skeleton.json"):
            src = src_dump / name
            if src.is_file():
                (out_dump / name).write_bytes(src.read_bytes())

    anim2 = build_anim_map(gr2, source=anim.source)
    for bname, caps in cap_by_bone.items():
        try:
            t2 = _track_by_name(anim2, bname)
            if t2.position.edit_mode == "float_controls":
                t2.position.values_bytes = max(t2.position.values_bytes, caps["pos"])
            if t2.orientation.edit_mode == "float_controls":
                t2.orientation.values_bytes = max(t2.orientation.values_bytes, caps["ori"])
            if t2.scale.edit_mode == "float_controls":
                t2.scale.values_bytes = max(t2.scale.values_bytes, caps["scl"])
            if t2.orientation.edit_mode == "quat_k16":
                t2.orientation.values_bytes = max(
                    t2.orientation.values_bytes, t2.orientation.knot_count * 8
                )
        except KeyError:
            pass
    write_anim_map(anim2, out_dump / "anim_map.json")
    try:
        doc = json.loads((out_dump / "anim_map.json").read_text(encoding="utf-8"))
        for t in doc.get("tracks") or []:
            bname = t.get("name")
            if bname not in cap_by_bone:
                continue
            caps = cap_by_bone[bname]
            src_tr = _track_by_name(anim, bname)
            if t.get("position") and src_tr.position.edit_mode == "float_controls":
                t["position"]["values_bytes"] = max(
                    int(t["position"].get("values_bytes") or 0),
                    int(caps["pos"] or 0),
                    int(src_tr.position.knot_count or 0) * 12,
                )
            if t.get("orientation") and src_tr.orientation.edit_mode == "float_controls":
                t["orientation"]["values_bytes"] = max(
                    int(t["orientation"].get("values_bytes") or 0),
                    int(caps["ori"] or 0),
                    int(src_tr.orientation.knot_count or 0) * 16,
                )
            if t.get("scale") and src_tr.scale.edit_mode == "float_controls":
                t["scale"]["values_bytes"] = max(
                    int(t["scale"].get("values_bytes") or 0),
                    int(caps["scl"] or 0),
                    int(src_tr.scale.knot_count or 0) * 12,
                )
        (out_dump / "anim_map.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    skel = build_skeleton(anim2, section0)
    write_skeleton(skel, out_dump / "skeleton.json")
    edit_doc = {
        "from_dump": src_dump.name,
        "overwrite": in_place,
        "base_gr2": Path(anim.source).name,
        "edits": edit_records,
        "section0_size": len(section0),
    }
    (out_dump / "EDIT.json").write_text(json.dumps(edit_doc, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "run_id": run_id,
        "out_dump": str(out_dump),
        "overwrite": in_place,
        "bone": primary,
        "bones_edited": len(groups),
        "bones": [g["bone"] for g in groups],
        "applied": all_applied,
        "curve": sample_curve(section0, anim2, primary),
        "section0_size": len(section0),
    }


def ensure_anim_map_for_dump(dump_dir: Path, base_gr2: Optional[Path] = None) -> AnimMap:
    dump_dir = Path(dump_dir)
    existing = dump_dir / "anim_map.json"
    if base_gr2 is None:
        from paths import resolve_input_gr2

        name = None
        src_txt = dump_dir / "SOURCE.txt"
        if src_txt.is_file():
            for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("name="):
                    name = line.split("=", 1)[1].strip()
                    break
        base_gr2 = resolve_input_gr2(name or "")

    gr2 = load_gr2(base_gr2)
    decode_sections(gr2, True)
    extract_section_fixups(gr2)
    # Prefer dump section_0 if present (may be grown after quat_k16 insert)
    s0 = dump_dir / "section_0.bin"
    if s0.is_file():
        data = s0.read_bytes()
        base_len = len(gr2.sections[0].uncompressed or b"")
        if len(data) >= base_len or len(data) == base_len:
            gr2.sections[0].uncompressed = data
            gr2.sections[0].header.uncompressed_size = len(data)
    reloc_p = dump_dir / "section_0.relocations.bin"
    if reloc_p.is_file():
        gr2.sections[0].relocations = reloc_p.read_bytes()
    mm_p = dump_dir / "section_0.mm.bin"
    if mm_p.is_file():
        gr2.sections[0].mixed_marshalling = mm_p.read_bytes()
    anim = build_anim_map(gr2, source=str(base_gr2))
    write_anim_map(anim, existing)
    s0_bytes = gr2.sections[0].uncompressed or b""
    write_skeleton(build_skeleton(anim, s0_bytes), dump_dir / "skeleton.json")
    return anim


def main_cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Granny bone position map/edit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_map = sub.add_parser("map", help="Build anim_map.json for a GR2 or dump")
    p_map.add_argument("target", help="GR2 path or OUT_FILES dump folder")
    p_map.add_argument("-o", "--out", help="Output anim_map.json path")

    p_list = sub.add_parser("list-bones", help="List bone names")
    p_list.add_argument("target", help="anim_map.json, dump folder, or GR2")

    p_edit = sub.add_parser("edit", help="Offset/set bone position into a new dump")
    p_edit.add_argument("dump", help="Source OUT_FILES dump folder")
    p_edit.add_argument("--bone", required=True)
    p_edit.add_argument("--op", choices=("offset", "set"), default="offset")
    p_edit.add_argument("--xyz", required=True, help="x,y,z")
    p_edit.add_argument("--t0", type=float, default=None)
    p_edit.add_argument("--t1", type=float, default=None)
    p_edit.add_argument("-o", "--out", default=None, help="Output dump folder")

    args = ap.parse_args(argv)
    target = Path(args.target if hasattr(args, "target") else getattr(args, "dump", ""))

    if args.cmd == "map":
        if target.is_dir():
            anim = ensure_anim_map_for_dump(target)
            out = Path(args.out) if args.out else target / "anim_map.json"
        else:
            gr2 = load_gr2(target)
            decode_sections(gr2, True)
            extract_section_fixups(gr2)
            anim = build_anim_map(gr2, source=str(target))
            out = Path(args.out) if args.out else target.with_suffix(".anim_map.json")
        write_anim_map(anim, out)
        print(f"Wrote {out} ({len(anim.bones)} bones, duration≈{anim.duration:.3f})")
        return 0

    if args.cmd == "list-bones":
        if target.suffix.lower() == ".json":
            bones = list_bones(load_anim_map(target))
        elif target.is_dir():
            anim = ensure_anim_map_for_dump(target)
            bones = anim.bones
        else:
            gr2 = load_gr2(target)
            decode_sections(gr2, True)
            extract_section_fixups(gr2)
            bones = build_anim_map(gr2, source=str(target)).bones
        for b in bones:
            print(b)
        return 0

    if args.cmd == "edit":
        xyz = [float(x) for x in args.xyz.split(",")]
        out = Path(args.out) if args.out else None
        result = apply_edit_to_dump(
            Path(args.dump),
            bone=args.bone,
            op=args.op,
            xyz=xyz,
            t0=args.t0,
            t1=args.t1,
            out_dump=out,
            # CLI without --out keeps writing a new OUT_FILES dump
            overwrite=out is not None,
        )
        print(json.dumps(result, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
