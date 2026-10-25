"""Viewport display for imported GR2 LAB armatures (DGB Karlach snapshot).

Applies bone collections, custom-shape widgets + TRS, then the four Larian IK
constraints. Does not change bind rest matrices or F-curve values.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from . import head_ik
from . import larian_ik

_SNAP_PATH = Path(__file__).resolve().parent / "dgb_karlach.json"
_WIDGET_COLL = "GR2LAB_BoneShapes"
_WIDGET_PREFIX = "GR2LAB_WGT_"
# Catalog tiles beside the character. Mesh data stays unit-sized so bone
# custom_shape_scale_xyz is unchanged; only the parked object TRS is preview.
_WGT_PREVIEW = 0.08
_WGT_GRID_ORIGIN = (1.40, 0.0, 0.05)
_WGT_GRID_STEP = 0.25
_WGT_GRID_COLS = 5

COLLECTION_ORDER = (
    "Root",
    "Spine",
    "Tail",
    "Leg L",
    "Leg R",
    "Arm L",
    "Arm R",
    "Hand L",
    "Hand R",
    "IK",
    "UI",
    "FX",
    "Extra",
)
_COLLECTION_LEGACY = {
    "01 Root": "Root",
    "02 Spine": "Spine",
    "03 Leg L": "Leg L",
    "04 Leg R": "Leg R",
    "05 Arm L": "Arm L",
    "06 Arm R": "Arm R",
    "07 Hand L": "Hand L",
    "08 Hand R": "Hand R",
    "09 IK": "IK",
    "10 FX": "FX",
    "11 Extra": "Extra",
}

_IK_EFFECTORS = frozenset(("Ankle_L", "Ankle_R", "Wrist_L", "Wrist_R"))
_ROOT_BONES = frozenset(("Dummy_Root", "Root_M"))
_SPINE_BONES = frozenset(("Chest_M", "Neck_M", "Head_M", "HeadEnd_M_endBone"))
_LEG_MARKERS = ("Hip", "Knee", "Toes")
_ARM_MARKERS = ("Scapula", "Shoulder", "Elbow")

# Next deform bone for a connected humanoid stick figure (skip twists / Dummy FX).
_CENTER_NEXT = {
    "Root_M": "Spine1_M",
    "Spine1_M": "Spine2_M",
    "Spine2_M": "Chest_M",
    "Chest_M": "Neck_M",
    "Neck_M": "Head_M",
    "Head_M": "HeadEnd_M_endBone",
    **{f"Tail{i}_M": f"Tail{i + 1}_M" for i in range(7)},
    "Tail7_M": "Tail8_M_endBone",
}
_SIDE_NEXT = {
    "Hip": "Knee",
    "Knee": "Ankle",
    "Toes": "ToesEnd",
    "Scapula": "Shoulder",
    "Shoulder": "Elbow",
    "Elbow": "Wrist",
}
_FINGER_RE = re.compile(
    r"^((?:Ring|Pinky|Middle|Index|Thumb)Finger)(\d)_([LR])$"
)
_SIDE_RE = re.compile(
    r"^(Hip|Knee|Ankle|Toes|Scapula|Shoulder|Elbow|Wrist)_([LR])$"
)

_SNAPSHOT: Optional[Dict[str, Any]] = None

# Rigify/ARP-style wire widgets. Unit box/line run along +Y (Blender bone axis).
# custom_shape_scale_xyz is applied in shape space, then rotation maps +Y onto
# the Larian joint (bone local ±X). Thickness stays on X/Z so limbs don't
# turn into rhombs.
_BOX_LEN = 1.0
_BOX_HALF = 0.10
_IK_CIRCLE = 0.050
_HEAD_IK_TARGET = 0.117  # 0.13 × 0.90 after L/R vs U/D proportion change
_HEAD_IK_TARGET_SQUEEZE = 0.50  # extra X/Z scale squeeze so yaw reads
_IK_SPHERE = 0.060
_TWIST_SPHERE = 0.016
_TWIST_ARROW = 0.022
_FX_AXES = 0.02
_FINGER_TIP = 0.005
_FINGER_DOT = 0.005
_FINGER_TIP_LEN = 0.022
_FOOT_SIZE = 0.11
_HAND_SIZE = 0.07
_HAND_IK_SIZE = 0.12
_HAND_IK_EXTENT_REF = 4.4
_HEAD_LINE = 0.10
_HEAD_CIRCLE = 0.12
_ROOT_CIRCLE = 0.36
# HUD palette: brighter rest, lighter select. IK = coral / beige / yellow.
# Left = cyan, right = magenta, center = violet.
# UI_Head card is a brighter lavender so the unselected menu still reads.
_TONE_VIOLET = (
    (0.62, 0.46, 0.88),
    (0.78, 0.62, 1.00),
    (0.88, 0.74, 1.00),
)
_TONE_HUD = (
    (0.70, 0.54, 0.94),
    (0.78, 0.62, 1.00),
    (0.88, 0.74, 1.00),
)
_TONE_CYAN = (
    (0.32, 0.70, 0.90),
    (0.50, 0.86, 1.00),
    (0.66, 0.92, 1.00),
)
_TONE_BLUE = (
    (0.38, 0.56, 0.96),
    (0.54, 0.72, 1.00),
    (0.70, 0.82, 1.00),
)
_TONE_MAGENTA = (
    (0.74, 0.42, 0.86),
    (0.88, 0.56, 1.00),
    (0.96, 0.70, 1.00),
)
_TONE_IK = (
    (0.90, 0.76, 0.52),
    (0.98, 0.88, 0.62),
    (1.00, 0.94, 0.74),
)
_TONE_KISS = (
    (0.80, 0.66, 0.42),
    (0.90, 0.78, 0.52),
    (0.96, 0.88, 0.64),
)
_SPINE_HOOP = {
    "Root_M": 0.10,
    "Spine1_M": 0.09,
    "Spine2_M": 0.11,
    "Chest_M": 0.16,
    "Neck_M": 0.07,
    "Head_M": 0.12,
    "Tail0_M": 0.070,
    "Tail1_M": 0.064,
    "Tail2_M": 0.058,
    "Tail3_M": 0.052,
    "Tail4_M": 0.046,
    "Tail5_M": 0.040,
    "Tail6_M": 0.034,
    "Tail7_M": 0.028,
    "Tail8_M_endBone": 0.022,
    "Tail8_endBone": 0.022,
}

# Prefer meshes from a user-dropped cs_* collection (Auto-Rig Pro / Rigify).
_SCENE_CS = {
    "box": ("cs_box", "cs_user_c_arm_ik"),
    "line": ("cs_line",),
    "sphere": ("cs_sphere",),
    "circle": ("cs_arrow_cross",),
    "twist": ("cs_arrow_twist",),
    "foot_l": ("cs_foot.l",),
    "foot_r": ("cs_foot.r",),
    "hand": ("cs_hand",),
}

ModeSet = Callable[..., None]


def _trs_matrix_granny(
    t: List[float], r: List[float], s: List[float]
) -> Matrix:
    loc = Vector((float(t[0]), float(t[1]), float(t[2])))
    rot = Quaternion((float(r[3]), float(r[0]), float(r[1]), float(r[2])))
    scl = Vector((float(s[0]), float(s[1]), float(s[2])))
    return Matrix.LocRotScale(loc, rot, scl)


def ensure_missing_end_bones(
    arm_obj,
    context,
    *,
    mode_set: Optional[ModeSet] = None,
) -> List[str]:
    """Create clip-missing Larian ``*_endBone`` tips from Body_Base bind.

    Viewport / Edit-mode only — does not add export tracks.
    """
    if arm_obj is None or getattr(arm_obj, "type", None) != "ARMATURE":
        return []
    existing = {b.name for b in arm_obj.data.bones}
    doc: Dict[str, Any] = {}
    raw = arm_obj.get("gr2lab_json")
    if raw:
        try:
            doc = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception:
            doc = {}
    bind_by_name: Dict[str, Dict[str, Any]] = {}
    for b in (doc.get("bind") or {}).get("bones") or []:
        name = b.get("name")
        if name:
            bind_by_name[name] = b
    try:
        from bind_pose import expand_bind_map, graft_leaf_end_bones  # type: ignore

        bind_meta = doc.get("bind") if isinstance(doc.get("bind"), dict) else {}
        bind_by_name = expand_bind_map(
            bind_by_name,
            source_gr2=str(doc.get("source_gr2") or ""),
            bind_source_gr2=str((bind_meta or {}).get("source_gr2") or ""),
        )
    except Exception:
        from bind_pose import graft_leaf_end_bones  # type: ignore

    skel: List[Dict[str, Any]] = []
    seen = set()
    for b in doc.get("skeleton") or []:
        name = b.get("name")
        if not name or name not in existing:
            continue
        skel.append(
            {
                "name": name,
                "parent_name": b.get("parent_name"),
                "rest_translation": list(b.get("rest_translation") or [0, 0, 0]),
                "rest_rotation": list(b.get("rest_rotation") or [0, 0, 0, 1]),
                "rest_scale": list(b.get("rest_scale") or [1, 1, 1]),
            }
        )
        seen.add(name)
    for bone in arm_obj.data.bones:
        if bone.name in seen:
            continue
        skel.append(
            {
                "name": bone.name,
                "parent_name": bone.parent.name if bone.parent else None,
                "rest_translation": [0.04, 0.0, 0.0],
                "rest_rotation": [0.0, 0.0, 0.0, 1.0],
                "rest_scale": [1.0, 1.0, 1.0],
            }
        )
    _display, _grafted = graft_leaf_end_bones(skel, bind_by_name)
    to_add = [
        b
        for b in _display
        if b.get("gr2lab_grafted") and b.get("name") not in existing
    ]
    if not to_add:
        return []

    added: List[str] = []
    head_ik._set_mode(context, "EDIT", arm_obj, mode_set)
    ebs = arm_obj.data.edit_bones
    for b in to_add:
        name = b.get("name")
        parent_name = b.get("parent_name")
        parent = ebs.get(parent_name) if parent_name else None
        if not name or parent is None or name in ebs:
            continue
        local = _trs_matrix_granny(
            list(b.get("rest_translation") or [0, 0, 0]),
            list(b.get("rest_rotation") or [0, 0, 0, 1]),
            list(b.get("rest_scale") or [1, 1, 1]),
        )
        world = parent.matrix @ local
        eb = ebs.new(name)
        head = world.to_translation()
        axis = world.to_3x3() @ Vector((0.0, 0.05, 0.0))
        if axis.length < 1e-8:
            axis = Vector((0.0, 0.05, 0.0))
        eb.head = head
        eb.tail = head + axis
        eb.matrix = world
        eb.parent = parent
        eb.use_connect = False
        added.append(name)
    head_ik._set_mode(context, "POSE", arm_obj, mode_set)
    for name in added:
        pb = arm_obj.pose.bones.get(name)
        if pb is not None:
            pb["gr2lab_grafted"] = True
            pb.rotation_mode = "QUATERNION"
    if added:
        print(
            f"[GR2 LAB] grafted {len(added)} missing end bones on "
            f"{arm_obj.name!r}: " + ", ".join(added)
        )
    return added


def load_snapshot() -> Dict[str, Any]:
    global _SNAPSHOT
    if _SNAPSHOT is None:
        with _SNAP_PATH.open(encoding="utf-8") as f:
            _SNAPSHOT = json.load(f)
    return _SNAPSHOT


def _side_of(name: str) -> str:
    if name.endswith("_L") or "_L_" in name or name.endswith("_L_endBone"):
        return "L"
    if name.endswith("_R") or "_R_" in name or name.endswith("_R_endBone"):
        return "R"
    return ""


def classify_bone(name: str) -> str:
    """Stable collection for a Larian bone name."""
    if name in _ROOT_BONES:
        return "Root"
    if name in _IK_EFFECTORS or name.startswith("UI_Head") or name == "IK_Head_Target" or (
        name.startswith("Dummy_") and name.endswith("_IK")
    ):
        return "IK"
    if name.startswith("Tail"):
        return "Tail"
    if name.startswith("Dummy_"):
        return "FX"
    if name in _SPINE_BONES or name.startswith("Spine"):
        return "Spine"
    if "Finger" in name or name.startswith("Thumb"):
        return "Hand L" if _side_of(name) == "L" else "Hand R"
    side = _side_of(name)
    if any(m in name for m in _LEG_MARKERS):
        return "Leg L" if side == "L" else "Leg R"
    if any(m in name for m in _ARM_MARKERS):
        return "Arm L" if side == "L" else "Arm R"
    return "Extra"


def anatomical_next(name: str) -> Optional[str]:
    """Deform-chain successor used to span a widget from joint to joint."""
    if name in _IK_EFFECTORS:
        return None
    if name in _CENTER_NEXT:
        return _CENTER_NEXT[name]
    m = _SIDE_RE.match(name)
    if m:
        nxt = _SIDE_NEXT.get(m.group(1))
        return f"{nxt}_{m.group(2)}" if nxt else None
    m = _FINGER_RE.match(name)
    if m:
        return f"{m.group(1)}{int(m.group(2)) + 1}_{m.group(3)}"
    return None


def _vec_in_pose_bone(pb, armature_vec: Vector) -> Vector:
    """Armature-space vector into the pose bone's current local space."""
    try:
        dest = pb.matrix.to_3x3().inverted() @ armature_vec
    except Exception:
        dest = armature_vec.copy()
    if dest.length < 1e-6:
        return Vector((0.0, 0.0, 1.0))
    return dest.normalized()


def _pose_joint(pb, nxt_name: str) -> Optional[Vector]:
    """Next joint in the pose bone's current local space."""
    nxt_pb = pb.id_data.pose.bones.get(nxt_name) if pb.id_data else None
    if nxt_pb is None:
        return None
    loc = pb.matrix.to_3x3().inverted() @ (
        nxt_pb.matrix.translation - pb.matrix.translation
    )
    if loc.length < 1e-4:
        return None
    return loc.copy()


def _named_bone(bones, *names: Optional[str]):
    getter = getattr(bones, "get", None)
    if not callable(getter):
        return None
    for name in names:
        if not name:
            continue
        hit = getter(name)
        if hit is not None:
            return hit
    return None


def _tail_end_aliases(name: str) -> tuple:
    aliases = [name, f"{name}_endBone"]
    if name in ("Tail8_M", "Tail8_M_endBone"):
        aliases.append("Tail8_endBone")
    if name == "Tail8_endBone":
        aliases.append("Tail8_M_endBone")
    return tuple(aliases)


def _spine_joint(pb, arm_data) -> Optional[Vector]:
    """Next vertebra in the pose bone's local space (line from this joint to the next)."""
    nxt = anatomical_next(pb.name)
    if nxt:
        nxt_pb = None
        if pb.id_data is not None:
            nxt_pb = _named_bone(pb.id_data.pose.bones, *_tail_end_aliases(nxt))
        if nxt_pb is not None:
            loc = pb.matrix.to_3x3().inverted() @ (
                nxt_pb.matrix.translation - pb.matrix.translation
            )
            if loc.length > 1e-4:
                return loc.copy()
        bb = _named_bone(arm_data.bones, *_tail_end_aliases(nxt))
        ba = arm_data.bones.get(pb.name)
        if ba is not None and bb is not None:
            loc = ba.matrix.inverted().to_3x3() @ (bb.head_local - ba.head_local)
            if loc.length > 1e-4:
                return loc.copy()
    if pb.name == "Head_M":
        neck = pb.id_data.pose.bones.get("Neck_M") if pb.id_data else None
        if neck is not None:
            incoming = pb.matrix.to_3x3().inverted() @ (
                pb.matrix.translation - neck.matrix.translation
            )
            if incoming.length > 1e-4:
                return incoming.normalized() * _HEAD_LINE
        return _vec_in_pose_bone(pb, Vector((0.0, 1.0, 0.0))) * _HEAD_LINE
    if pb.name.startswith("Tail") and "endBone" in pb.name:
        length = float(getattr(pb, "length", 0.0) or 0.0)
        if length < 1e-4:
            ba = arm_data.bones.get(pb.name)
            if ba is not None:
                length = float(ba.length)
        if length < 1e-4:
            length = 0.04
        return Vector((0.0, length, 0.0))
    return None


def _spine_hoop_euler(joint: Vector, back: Vector) -> Euler:
    """Mesh +X along the next vertebra, mesh +Z (large triangle) toward the back."""
    x = joint.normalized()
    z = back - x * back.dot(x)
    if z.length < 1e-5:
        z = Vector((0.0, 0.0, 1.0))
        z = z - x * z.dot(x)
    if z.length < 1e-5:
        z = Vector((0.0, 1.0, 0.0))
        z = z - x * z.dot(x)
    z.normalize()
    y = z.cross(x)
    if y.length < 1e-5:
        return x.to_track_quat("X", "Z").to_euler("XYZ")
    y.normalize()
    z = x.cross(y).normalized()
    y = z.cross(x).normalized()
    return Matrix((x, y, z)).transposed().to_euler("XYZ")


def _joint_local(arm_data, name: str) -> Optional[Vector]:
    nxt = anatomical_next(name)
    ba = arm_data.bones.get(name)
    if not ba:
        return None
    bb = None
    if nxt:
        bb = _named_bone(arm_data.bones, *_tail_end_aliases(nxt))
    if bb is not None:
        loc = ba.matrix_local.inverted() @ bb.head_local
        if loc.length > 1e-4:
            return loc.copy()
    if name == "Head_M":
        return Vector((_HEAD_LINE, 0.0, 0.0))
    return None


def _twist_limb_axis(arm_data, name: str) -> Optional[Vector]:
    """Shoulder→Elbow / Hip→Knee / Elbow→Wrist in the twist bone's local space."""
    side = _side_of(name)
    host = nxt = None
    if "Hip" in name:
        host, nxt = f"Hip_{side}", f"Knee_{side}"
    elif "Elbow" in name:
        host, nxt = f"Elbow_{side}", f"Wrist_{side}"
    elif "Shoulder" in name:
        host, nxt = f"Shoulder_{side}", f"Elbow_{side}"
    if not nxt:
        return None
    ba = arm_data.bones.get(name)
    bb = arm_data.bones.get(nxt)
    if ba is None or bb is None:
        return None
    loc = ba.matrix_local.inverted() @ bb.head_local
    if loc.length > 1e-4:
        return loc.copy()
    hb = arm_data.bones.get(host) if host else None
    if hb is not None:
        world = bb.head_local - hb.head_local
        loc = ba.matrix_local.to_3x3().inverted() @ world
        if loc.length > 1e-4:
            return loc.copy()
    return None


def _shape_role(name: str) -> str:
    """Which viewport widget a bone should use."""
    if name in _IK_EFFECTORS:
        return "ik_joint"
    if name in (
        "Dummy_L_Foot_IK",
        "Dummy_R_Foot_IK",
        "Dummy_L_Hand_IK",
        "Dummy_R_Hand_IK",
    ):
        return "hand_ik"
    if name == "UI_Head":
        return "ui_panel"
    if name.endswith("_Track") and name.startswith("UI_Head"):
        return "ui_track"
    if name in ("UI_Head_IK", "UI_Head_Spine", "UI_Head_Kiss"):
        return "ui_knob"
    if name == "IK_Head_Target":
        return "head_ik_target"
    if name.startswith("Dummy_") and name.endswith("_IK"):
        return "ik_target"
    if name.startswith("Tail") and "endBone" in name:
        return "fx"
    if name.startswith("Tail"):
        return "spine"
    if name == "Dummy_Root":
        return "root"
    if "Twist" in name:
        return "twist"
    if name.startswith("Dummy_"):
        return "fx"
    if anatomical_next(name) is None:
        return "fx"
    if "Finger" in name or name.startswith("Thumb"):
        return "finger"
    if name in _SPINE_BONES or name.startswith("Spine") or name == "Root_M":
        return "spine"
    return "limb"


def _mesh_box(center_line: bool = True):
    """Wire FK box from the joint (y=0) to the next joint (y=1)."""
    w = _BOX_HALF
    verts = [
        (-w, 0.0, -w),
        (w, 0.0, -w),
        (w, 0.0, w),
        (-w, 0.0, w),
        (-w, _BOX_LEN, -w),
        (w, _BOX_LEN, -w),
        (w, _BOX_LEN, w),
        (-w, _BOX_LEN, w),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    if center_line:
        verts.extend([(0.0, 0.0, 0.0), (0.0, _BOX_LEN, 0.0)])
        edges.append((8, 9))
    return verts, [], edges


def _mesh_line():
    return [(0.0, 0.0, 0.0), (0.0, _BOX_LEN, 0.0)], [], [(0, 1)]


def _mesh_plain_axes():
    """Blender Empty > Plain Axes: three unit lines through the origin."""
    verts = [
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    ]
    return verts, [], [(0, 1), (2, 3), (4, 5)]


def _mesh_finger():
    """Unit +Y line with one XZ ring at the start joint (no doubled knuckles)."""
    verts = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    edges = [(0, 1)]
    segs = 12
    base = len(verts)
    for i in range(segs):
        a = (2.0 * math.pi * i) / segs
        verts.append((math.cos(a), 0.0, math.sin(a)))
        edges.append((base + i, base + ((i + 1) % segs)))
    return verts, [], edges


def _finger_joint(arm_data, name: str) -> Optional[Vector]:
    """Next knuckle, or a short distal stub along the previous phalanx."""
    joint = _joint_local(arm_data, name)
    if joint is not None:
        return joint
    m = _FINGER_RE.match(name)
    if m is None:
        return Vector((-_FINGER_TIP_LEN, 0.0, 0.0))
    prev = f"{m.group(1)}{int(m.group(2)) - 1}_{m.group(3)}"
    prev_j = _joint_local(arm_data, prev)
    if prev_j is not None and prev_j.length > 1e-4:
        return prev_j.normalized() * (prev_j.length * 0.85)
    return Vector((-_FINGER_TIP_LEN, 0.0, 0.0))


def _mesh_circle(segments: int = 24):
    """Circle in XZ (normal +Y). Dummy_Root: lies on the ground in this file."""
    verts = []
    edges = []
    for i in range(segments):
        a = (2.0 * math.pi * i) / segments
        verts.append((math.cos(a), 0.0, math.sin(a)))
        edges.append((i, (i + 1) % segments))
    return verts, [], edges


def _mesh_arrow_cross():
    """ARP/Rigify arrow-cross in XY (normal +Z). Used for IK targets."""
    r_tip = 1.0
    r_inner = 0.67
    wing = 0.22
    verts = []
    edges = []
    for k in range(4):
        a = (math.pi * 0.5) * k
        c, s = math.cos(a), math.sin(a)
        px, py = -s, c
        base = len(verts)
        verts.extend(
            [
                (c * r_tip, s * r_tip, 0.0),
                (c * r_inner + px * wing, s * r_inner + py * wing, 0.0),
                (c * r_inner - px * wing, s * r_inner - py * wing, 0.0),
            ]
        )
        edges.extend([(base, base + 1), (base, base + 2), (base + 1, base + 2)])
    ring = 20
    rb = len(verts)
    for i in range(ring):
        a = (2.0 * math.pi * i) / ring
        verts.append((math.cos(a) * r_inner, math.sin(a) * r_inner, 0.0))
        edges.append((rb + i, rb + ((i + 1) % ring)))
    return verts, [], edges


def _mesh_arrow_twist(segments: int = 22):
    """ARP ``cs_arrow_twist`` form: helical circular arrow around +Y (bone axis)."""
    verts = []
    edges = []
    n = max(8, int(segments))
    span = math.radians(340.0)
    a0 = math.radians(-10.0)
    y_amp = 0.12
    for i in range(n + 1):
        t = i / n
        a = a0 + span * t
        y = y_amp * math.sin(t * math.pi * 2.0)
        verts.append((math.cos(a), y, math.sin(a)))
        if i:
            edges.append((i - 1, i))
    x0, y0, z0 = verts[-2]
    x1, y1, z1 = verts[-1]
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / length, dy / length, dz / length
    sx, sz = dz, -dx
    sl = math.sqrt(sx * sx + sz * sz) or 1.0
    sx, sz = sx / sl, sz / sl
    wing, tip = 0.22, 0.28
    base = len(verts)
    verts.extend(
        [
            (x1 + dx * tip, y1 + dy * tip, z1 + dz * tip),
            (x1 + sx * wing - dx * 0.04, y1, z1 + sz * wing - dz * 0.04),
            (x1 - sx * wing - dx * 0.04, y1, z1 - sz * wing - dz * 0.04),
        ]
    )
    edges.extend([(n, base), (base, base + 1), (base, base + 2), (base + 1, base + 2)])
    return verts, [], edges


def _mesh_circle_yz(segments: int = 24):
    """Circle in YZ — hoop around the Larian bone axis (local X)."""
    verts = []
    edges = []
    for i in range(segments):
        a = (2.0 * math.pi * i) / segments
        verts.append((0.0, math.cos(a), math.sin(a)))
        edges.append((i, (i + 1) % segments))
    return verts, [], edges


def _arrow_cross_xy_source():
    """Raw XY verts/edges from cs_arrow_cross, or a generated fallback."""
    src = _find_scene_cs(("cs_arrow_cross",))
    if src is not None and src.type == "MESH" and len(src.data.vertices) >= 3:
        verts = [(v.co.x, v.co.y, v.co.z) for v in src.data.vertices]
        edges = [tuple(e.vertices) for e in src.data.edges]
        return verts, edges
    verts, _faces, edges = _mesh_arrow_cross()
    return verts, edges


def _map_xy_cross(plane: str):
    """Map an XY arrow-cross into YZ (spine) or XZ (root on ground). Unit radius."""
    raw, edges = _arrow_cross_xy_source()
    mapped = []
    for x, y, _z in raw:
        if plane == "yz":
            mapped.append((0.0, x, y))
        elif plane == "xz":
            mapped.append((x, 0.0, y))
        else:
            mapped.append((x, y, 0.0))
    m = max((max(abs(a), abs(b), abs(c)) for a, b, c in mapped), default=1.0) or 1.0
    verts = [(a / m, b / m, c / m) for a, b, c in mapped]
    return verts, [], [tuple(e) for e in edges]


def _mesh_spine_cross():
    """Arrow-cross in YZ around bone +X. Full-size spike on +Z (character back)."""
    r_inner = 0.67
    r_tip_big, wing_big = 1.0, 0.22
    r_tip_small, wing_small = r_inner + (1.0 - r_inner) * 0.5, 0.22 * 0.5
    # k=0 +Y, k=1 +Z (back), k=2 -Y, k=3 -Z (front)
    sizes = (
        (r_tip_small, wing_small),
        (r_tip_big, wing_big),
        (r_tip_small, wing_small),
        (r_tip_small, wing_small),
    )
    verts = []
    edges = []
    for k, (r_tip, wing) in enumerate(sizes):
        a = (math.pi * 0.5) * k
        c, s = math.cos(a), math.sin(a)
        px, py = -s, c
        base = len(verts)
        verts.extend(
            [
                (0.0, c * r_tip, s * r_tip),
                (0.0, c * r_inner + px * wing, s * r_inner + py * wing),
                (0.0, c * r_inner - px * wing, s * r_inner - py * wing),
            ]
        )
        edges.extend([(base, base + 1), (base, base + 2), (base + 1, base + 2)])
    ring = 21
    rb = len(verts)
    for i in range(ring):
        a = (2.0 * math.pi * i) / ring
        verts.append((0.0, math.cos(a) * r_inner, math.sin(a) * r_inner))
        edges.append((rb + i, rb + ((i + 1) % ring)))
    m = max((max(abs(a), abs(b), abs(c)) for a, b, c in verts), default=1.0) or 1.0
    verts = [(a / m, b / m, c / m) for a, b, c in verts]
    return verts, [], edges


def _mesh_hoop(segments: int = 24):
    """Spine: back-weighted arrow-cross in YZ + connecting line along +X."""
    verts, _faces, edges = _mesh_spine_cross()
    if not verts:
        verts, _faces, edges = _mesh_circle_yz(segments)
    i0 = len(verts)
    verts.append((0.0, 0.0, 0.0))
    verts.append((1.0, 0.0, 0.0))
    edges = list(edges)
    edges.append((i0, i0 + 1))
    return verts, [], edges


def _mesh_root_cross():
    """Dummy_Root: big arrow-cross on the ground, no spine line."""
    return _map_xy_cross("xz")


def _thick_for(role: str, length: float) -> float:
    """Rigify-like cage width: chunky on thighs, still thin on fingers."""
    if role == "finger":
        return max(0.006, float(length) * 0.22)
    return max(0.024, float(length) * 0.11)


def _mesh_sphere(segments: int = 16):
    """Three orthogonal circles — ARP/Rigify wire sphere."""
    verts = []
    edges = []
    for axis in range(3):
        base = len(verts)
        for i in range(segments):
            a = (2.0 * math.pi * i) / segments
            c, s = math.cos(a), math.sin(a)
            if axis == 0:
                verts.append((0.0, c, s))
            elif axis == 1:
                verts.append((c, 0.0, s))
            else:
                verts.append((c, s, 0.0))
            edges.append((base + i, base + ((i + 1) % segments)))
    return verts, [], edges


def _add_circle_edges(verts, edges, radius: float, plane: str, segments: int) -> None:
    base = len(verts)
    for i in range(segments):
        a = (2.0 * math.pi * i) / segments
        c, s = math.cos(a), math.sin(a)
        if plane == "yz":
            verts.append((0.0, c * radius, s * radius))
        elif plane == "xz":
            verts.append((c * radius, 0.0, s * radius))
        else:
            verts.append((c * radius, s * radius, 0.0))
        edges.append((base + i, base + ((i + 1) % segments)))


def _mesh_head_ik_target():
    """Wire 2D stealth chevron along +Y: pointed tail, notched waist, barbs.

    Same 6-gon outline in XY and YZ so the silhouette reads from front and side.
    Left/right is a bit narrower than up/down so yaw/roll is readable. Not
    uniformly normalized — that made every axis equal.
    """
    outline = (
        (0.00, -1.00),
        (0.26, -0.12),
        (1.00, 0.02),
        (0.00, 1.00),
        (-1.00, 0.02),
        (-0.26, -0.12),
    )
    lr, ud = 0.55, 1.15
    outline = tuple((w * lr, along * ud) for w, along in outline)
    verts = [(w, along, 0.0) for w, along in outline]
    verts.extend((0.0, along, w) for w, along in outline)
    edges = [(i, (i + 1) % 6) for i in range(6)]
    edges.extend((6 + i, 6 + ((i + 1) % 6)) for i in range(6))
    return verts, [], edges


def _mesh_foot(side: str):
    """Closed XY loop: a circle pulled into a footprint. +Y toes, +Z up.

    Origin is the ankle. Outer side is +X for L, −X for R. Radii are sampled
    around a center slightly toward the ball so the heel sits behind the ankle.
    """
    sign = 1.0 if side == "L" else -1.0
    # (turns from toes, radius) around the foot center. 0 = toes, 0.25 = outer.
    keys = (
        (0.000, 0.50),
        (0.125, 0.32),
        (0.250, 0.20),
        (0.375, 0.22),
        (0.500, 0.40),
        (0.625, 0.20),
        (0.750, 0.14),
        (0.875, 0.28),
        (1.000, 0.50),
    )
    cx, cy = 0.0, 0.28
    n = 32
    verts = []
    for i in range(n):
        t = i / n
        r = 0.0
        for (t0, r0), (t1, r1) in zip(keys, keys[1:]):
            if t0 <= t <= t1 or (t1 == 1.0 and t >= t0):
                u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                u = 0.5 - 0.5 * math.cos(u * math.pi)
                r = r0 + (r1 - r0) * u
                break
        a = t * 2.0 * math.pi
        verts.append((sign * (cx + r * math.sin(a)), cy + r * math.cos(a), 0.0))
    ymax = max(v[1] for v in verts) or 1.0
    verts = [(x / ymax, y / ymax, 0.0) for x, y, _z in verts]
    edges = [(i, (i + 1) % n) for i in range(n)]
    return verts, [], edges


def _mesh_hand():
    """Flattened wire box along +Y for wrists."""
    hx, hy, hz = 0.22, 1.0, 0.08
    verts = [
        (-hx, 0.0, -hz),
        (hx, 0.0, -hz),
        (hx, 0.0, hz),
        (-hx, 0.0, hz),
        (-hx, hy, -hz),
        (hx, hy, -hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return verts, [], edges


def _mesh_hand_ik():
    """Wire sphere + 3D pyramid (two perpendicular triangles) along +Y."""
    verts, _faces, edges = _mesh_sphere(16)
    sphere_r = 1.1
    verts = [(x * sphere_r, y * sphere_r, z * sphere_r) for x, y, z in verts]
    r_inner = sphere_r
    r_tip = sphere_r + (3.4 - 1.0) * 0.5
    wing = 0.95 * 0.5
    i0 = len(verts)
    verts.extend(
        [
            (0.0, r_tip, 0.0),
            (wing, r_inner, 0.0),
            (-wing, r_inner, 0.0),
            (0.0, r_inner, wing),
            (0.0, r_inner, -wing),
            (0.0, 0.0, 0.0),
        ]
    )
    edges = list(edges)
    edges.extend(
        [
            (i0, i0 + 1),
            (i0, i0 + 2),
            (i0 + 1, i0 + 2),
            (i0, i0 + 3),
            (i0, i0 + 4),
            (i0 + 3, i0 + 4),
            (i0 + 1, i0 + 3),
            (i0 + 3, i0 + 2),
            (i0 + 2, i0 + 4),
            (i0 + 4, i0 + 1),
            (i0, i0 + 5),
        ]
    )
    return verts, [], edges


def _mesh_ui_track():
    """2D slot in bone XY — long axis +Y is the slider travel."""
    travel = head_ik.ui_travel()
    hw = head_ik.UI_FIELD_H * 0.5
    y0 = -head_ik.UI_KNOB_LEN * 0.5
    y1 = travel + head_ik.UI_KNOB_LEN * 0.5
    return _xy_rounded_rect(-hw, y0, hw, y1, hw, filled=False)


def _mesh_ui_knob():
    """Stadium in bone XY: one n-gon, unique verts, first vert at the bottom tip."""
    hl = float(head_ik.UI_KNOB_LEN) * 0.5
    hw = float(head_ik.UI_KNOB_WID) * 0.5
    r = hw
    cy_b = -hl + r
    cy_t = hl - r
    n_cap = 10
    ring: List = []

    def _arc(cx: float, cy: float, a0: float, a1: float, steps: int) -> None:
        for i in range(steps + 1):
            t = a0 + (a1 - a0) * (i / steps)
            ring.append((cx + r * math.cos(t), cy + r * math.sin(t), 0.0))

    _arc(0.0, cy_b, math.pi, 2.0 * math.pi, n_cap)
    _arc(0.0, cy_t, 0.0, math.pi, n_cap)
    verts: List = []
    for pt in ring:
        if verts:
            px, py, _pz = verts[-1]
            if abs(px - pt[0]) < 1e-7 and abs(py - pt[1]) < 1e-7:
                continue
        verts.append(pt)
    if len(verts) > 1:
        fx, fy, _fz = verts[0]
        lx, ly, _lz = verts[-1]
        if abs(fx - lx) < 1e-7 and abs(fy - ly) < 1e-7:
            verts.pop()
    tip = min(range(len(verts)), key=lambda i: verts[i][1])
    verts = verts[tip:] + verts[:tip]
    n = len(verts)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return verts, [tuple(range(n))], edges


def _mesh_ui_panel():
    """2D HUD card: labeled fields for IK / Kiss / Extra. Origin at center."""
    w = head_ik.UI_PANEL_W
    h = head_ik.UI_PANEL_H
    m = head_ik.UI_MARGIN
    ox, oy, _oz = head_ik.ui_panel_center()
    ox, oy = -ox, -oy
    verts: List = []
    faces: List = []
    edges: List = []
    _xy_append_rounded_rect(
        verts, faces, edges, ox, oy, ox + w, oy + h, 0.010, filled=False
    )
    labels = ("IK", "KISS", "EXTRA")
    fx0 = ox + head_ik.ui_field_x()
    fx1 = ox + w - m
    fh = head_ik.UI_FIELD_H
    title_h = 0.013
    gw = title_h * 0.62
    gap = title_h * 0.18
    title = "HEAD"
    title_w = len(title) * gw + (len(title) - 1) * gap
    _xy_append_text(
        verts,
        edges,
        title,
        (ox + (w - title_w) * 0.5, oy + h - m - title_h - 0.002),
        title_h,
    )
    for i, label in enumerate(labels):
        cy = oy + head_ik.ui_row_y(i)
        _xy_append_text(
            verts, edges, label, (ox + m + 0.004, cy - 0.006), 0.012
        )
        _xy_append_rounded_rect(
            verts,
            faces,
            edges,
            fx0,
            cy - fh * 0.5,
            fx1,
            cy + fh * 0.5,
            fh * 0.5,
            filled=False,
        )
    return verts, faces, edges


def _xy_rounded_rect(x0, y0, x1, y1, r, filled: bool = False, segs: int = 6):
    verts: List = []
    faces: List = []
    edges: List = []
    _xy_append_rounded_rect(verts, faces, edges, x0, y0, x1, y1, r, filled, segs)
    return verts, faces, edges


def _xy_append_rounded_rect(
    verts,
    faces,
    edges,
    x0,
    y0,
    x1,
    y1,
    r,
    filled: bool = False,
    segs: int = 6,
) -> None:
    r = min(abs(r), abs(x1 - x0) * 0.5, abs(y1 - y0) * 0.5)
    if r < 1e-6:
        base = len(verts)
        verts.extend(
            [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
        )
        edges.extend(
            [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3), (base + 3, base)]
        )
        if filled:
            faces.append((base, base + 1, base + 2))
            faces.append((base, base + 2, base + 3))
        return
    corners = (
        ((x0 + r, y0 + r), math.pi, 1.5 * math.pi),
        ((x1 - r, y0 + r), 1.5 * math.pi, 2.0 * math.pi),
        ((x1 - r, y1 - r), 0.0, 0.5 * math.pi),
        ((x0 + r, y1 - r), 0.5 * math.pi, math.pi),
    )
    base = len(verts)
    for (cx, cy), a0, a1 in corners:
        for i in range(segs + 1):
            t = a0 + (a1 - a0) * (i / segs)
            verts.append((cx + r * math.cos(t), cy + r * math.sin(t), 0.0))
    n = len(verts) - base
    edges.extend((base + i, base + ((i + 1) % n)) for i in range(n))
    if filled and n >= 3:
        faces.append(tuple(range(base, base + n)))


_GLYPHS = {
    "A": [[(0.05, 0.0), (0.5, 1.0), (0.95, 0.0)], [(0.22, 0.38), (0.78, 0.38)]],
    "D": [
        [
            (0.12, 0.0),
            (0.12, 1.0),
            (0.58, 1.0),
            (0.95, 0.72),
            (0.95, 0.28),
            (0.58, 0.0),
            (0.12, 0.0),
        ]
    ],
    "E": [
        [(0.88, 1.0), (0.12, 1.0), (0.12, 0.0), (0.88, 0.0)],
        [(0.12, 0.5), (0.72, 0.5)],
    ],
    "H": [
        [(0.12, 0.0), (0.12, 1.0)],
        [(0.88, 0.0), (0.88, 1.0)],
        [(0.12, 0.5), (0.88, 0.5)],
    ],
    "I": [
        [(0.22, 1.0), (0.78, 1.0)],
        [(0.5, 1.0), (0.5, 0.0)],
        [(0.22, 0.0), (0.78, 0.0)],
    ],
    "K": [
        [(0.12, 0.0), (0.12, 1.0)],
        [(0.88, 1.0), (0.12, 0.48)],
        [(0.12, 0.48), (0.88, 0.0)],
    ],
    "N": [[(0.12, 0.0), (0.12, 1.0), (0.88, 0.0), (0.88, 1.0)]],
    "P": [
        [
            (0.12, 0.0),
            (0.12, 1.0),
            (0.62, 1.0),
            (0.92, 0.8),
            (0.92, 0.55),
            (0.62, 0.42),
            (0.12, 0.42),
        ]
    ],
    "R": [
        [
            (0.12, 0.0),
            (0.12, 1.0),
            (0.62, 1.0),
            (0.92, 0.8),
            (0.92, 0.55),
            (0.62, 0.42),
            (0.12, 0.42),
        ],
        [(0.48, 0.42), (0.88, 0.0)],
    ],
    "S": [
        [
            (0.86, 0.84),
            (0.55, 1.0),
            (0.16, 0.88),
            (0.16, 0.62),
            (0.84, 0.38),
            (0.84, 0.14),
            (0.48, 0.0),
            (0.14, 0.14),
        ]
    ],
    "T": [
        [(0.08, 1.0), (0.92, 1.0)],
        [(0.5, 1.0), (0.5, 0.0)],
    ],
    "X": [
        [(0.12, 1.0), (0.88, 0.0)],
        [(0.88, 1.0), (0.12, 0.0)],
    ],
    " ": [],
}


def _xy_append_text(verts, edges, text: str, origin, height: float) -> None:
    x, y = origin
    gw = height * 0.62
    gap = height * 0.18
    for ch in text.upper():
        strokes = _GLYPHS.get(ch, _GLYPHS.get(" ", []))
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            start = len(verts)
            for px, py in stroke:
                verts.append((x + px * gw, y + py * height, 0.0))
            edges.extend((start + i, start + i + 1) for i in range(len(stroke) - 1))
        x += gw + gap


def _mesh_extents(obj) -> Vector:
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    return Vector(
        (
            max(max(xs) - min(xs), 1e-6),
            max(max(ys) - min(ys), 1e-6),
            max(max(zs) - min(zs), 1e-6),
        )
    )


def _find_scene_cs(prefixes) -> Optional[Any]:
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name.startswith(_WIDGET_PREFIX):
            continue
        low = obj.name.lower()
        for raw in prefixes:
            p = raw.lower()
            if low == p or low.startswith(p + ".") or low.startswith(p):
                return obj
    return None


def _span_scale(joint: Vector, mesh_len: float, mesh_w: float, thick: float) -> Vector:
    """Shape-local scale (Blender applies scale, then custom-shape rotation).

    Unit widgets run along +Y, so length is always Y; X/Z stay a constant thickness.
    """
    return Vector(
        (
            thick / mesh_w,
            float(joint.length) * 0.98 / mesh_len,
            thick / mesh_w,
        )
    )


def _ensure_scene_collection(context, name: str):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    scene_coll = context.scene.collection
    if coll.name not in scene_coll.children:
        try:
            scene_coll.children.link(coll)
        except RuntimeError:
            pass
    # Must stay evaluable or custom shapes draw as tiny rest octahedrons.
    coll.hide_viewport = False
    coll.hide_render = True
    try:
        coll.color_tag = "COLOR_06"
    except Exception:
        pass
    try:
        coll.hide_select = False
    except Exception:
        pass
    vl = context.view_layer.layer_collection
    stack = [vl]
    while stack:
        lc = stack.pop()
        if lc.collection == coll:
            lc.hide_viewport = False
            lc.exclude = False
            break
        stack.extend(lc.children)
    return coll


def _park_widget(obj, index: Optional[int] = None) -> None:
    obj.hide_render = True
    obj.hide_viewport = False
    try:
        obj.hide_select = False
    except Exception:
        pass
    obj.display_type = (
        "SOLID" if obj.name.endswith("_ui_knob") else "WIRE"
    )
    obj.rotation_euler = (0.0, 0.0, 0.0)
    ext = _mesh_extents(obj)
    span = max(float(ext.x), float(ext.y), float(ext.z), 1e-6)
    s = _WGT_PREVIEW / span
    obj.scale = (s, s, s)
    if index is None:
        obj.location = (0.0, 0.0, 0.0)
        return
    col = int(index) % _WGT_GRID_COLS
    row = int(index) // _WGT_GRID_COLS
    ox, oy, oz = _WGT_GRID_ORIGIN
    obj.location = (
        ox + col * _WGT_GRID_STEP,
        oy + row * _WGT_GRID_STEP,
        oz,
    )


def _layout_widget_catalog(widget_coll) -> None:
    """Human-sized preview tiles in a grid, off to the side of the rig."""
    objs = [
        o
        for o in widget_coll.objects
        if o.name.startswith(_WIDGET_PREFIX) and o.type == "MESH"
    ]
    objs.sort(key=lambda o: o.name)
    for i, obj in enumerate(objs):
        _park_widget(obj, i)


def _reuse_widget(obj_name: str, dest_coll):
    """Keep the shared catalog object. Deleting it clears custom_shape on older rigs."""
    existing = bpy.data.objects.get(obj_name)
    if existing is None or getattr(existing, "type", None) != "MESH":
        return None
    if dest_coll.objects.get(existing.name) is None:
        try:
            dest_coll.objects.link(existing)
        except RuntimeError:
            pass
    _park_widget(existing)
    return existing


def _ensure_widget(spec: Dict[str, Any], dest_coll) -> Optional[bpy.types.Object]:
    src_name = spec.get("name") or "widget"
    obj_name = _WIDGET_PREFIX + src_name
    existing = _reuse_widget(obj_name, dest_coll)
    if existing is not None:
        return existing
    verts = spec.get("verts") or []
    faces = spec.get("faces") or []
    edges = [] if faces else (spec.get("edges") or [])
    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(verts, edges, faces)
    me.update()
    obj = bpy.data.objects.new(obj_name, me)
    dest_coll.objects.link(obj)
    _park_widget(obj)
    return obj


def _hide_internal_mesh_edges(me) -> None:
    """One n-gon: merge doubles, keep a single face, no tessellation bake."""
    try:
        import bmesh
    except Exception:
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        if len(bm.faces) > 1:
            bmesh.ops.dissolve_faces(bm, faces=bm.faces[:])
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        for f in bm.faces:
            f.smooth = True
        for e in bm.edges:
            e.hide = False
        bm.to_mesh(me)
        me.update()
    finally:
        bm.free()


def _filled_widget_keys() -> frozenset:
    return frozenset(("ui_knob",))


def _replace_generated_mesh(obj, verts, faces, edges, *, hide_internal: bool) -> None:
    """Swap mesh data in place so older rigs keep their custom_shape pointers."""
    me = bpy.data.meshes.new(obj.name)
    me.from_pydata(verts, edges, faces)
    me.update()
    if hide_internal:
        _hide_internal_mesh_edges(me)
    old = obj.data
    obj.data = me
    if old is not None and old.users == 0:
        try:
            bpy.data.meshes.remove(old)
        except Exception:
            pass


def _ensure_generated_widget(
    key: str,
    dest_coll,
    verts: List,
    faces: List,
    edges: List,
    *,
    replace_mesh: bool = False,
) -> Optional[bpy.types.Object]:
    obj_name = _WIDGET_PREFIX + key
    hide = key in _filled_widget_keys()
    existing = _reuse_widget(obj_name, dest_coll)
    if existing is not None:
        if replace_mesh:
            _replace_generated_mesh(
                existing, verts, faces, edges, hide_internal=hide
            )
        return existing
    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(verts, edges, faces)
    me.update()
    if hide:
        _hide_internal_mesh_edges(me)
    obj = bpy.data.objects.new(obj_name, me)
    dest_coll.objects.link(obj)
    _park_widget(obj)
    return obj


def _clone_mesh_widget(src, key: str, dest_coll):
    obj_name = _WIDGET_PREFIX + key
    existing = _reuse_widget(obj_name, dest_coll)
    if existing is not None:
        return existing
    me = src.data.copy()
    me.name = obj_name
    obj = bpy.data.objects.new(obj_name, me)
    dest_coll.objects.link(obj)
    _park_widget(obj)
    return obj


def _ensure_role_widget(key: str, dest_coll, fallback):
    prefixes = _SCENE_CS.get(key)
    src = _find_scene_cs(prefixes) if prefixes else None
    if src is not None:
        return _clone_mesh_widget(src, key, dest_coll)
    verts, faces, edges = fallback()
    return _ensure_generated_widget(key, dest_coll, verts, faces, edges)


def _set_shape_rotation(pb, euler: Euler) -> None:
    if hasattr(pb, "custom_shape_rotation_euler"):
        pb.custom_shape_rotation_euler = euler
    elif hasattr(pb, "custom_shape_rotation"):
        pb.custom_shape_rotation = euler


def _armature_world_up(pb) -> Vector:
    """World +Z expressed in armature space (Y after the import +90° X)."""
    arm_obj = getattr(pb, "id_data", None)
    try:
        up = arm_obj.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, 1.0))
        if up.length > 1e-6:
            return up.normalized()
    except Exception:
        pass
    return Vector((0.0, 1.0, 0.0))


def _vec_in_bone(bone, armature_vec: Vector) -> Vector:
    """Armature-space vector into rest-bone space. Use bone.matrix, not matrix_local."""
    try:
        dest = bone.matrix.to_3x3().inverted() @ armature_vec
    except Exception:
        dest = armature_vec.copy()
    if dest.length < 1e-6:
        return Vector((0.0, 0.0, 1.0))
    return dest.normalized()


def _euler_map_axis(src: Vector, dest: Vector, x_hint: Optional[Vector] = None) -> Euler:
    """Rotate so src maps onto dest. Optional x_hint locks in-plane twist."""
    s = src.normalized()
    d = dest.normalized()
    if x_hint is None or x_hint.length < 1e-6:
        return s.rotation_difference(d).to_euler("XYZ")
    rot = s.rotation_difference(d)
    z = rot @ Vector((0.0, 0.0, 1.0))
    x = x_hint.normalized()
    x = x - z * x.dot(z)
    if x.length < 1e-5:
        return rot.to_euler("XYZ")
    x.normalize()
    y = z.cross(x)
    if y.length < 1e-5:
        return rot.to_euler("XYZ")
    y.normalize()
    x = y.cross(z)
    x.normalize()
    return Matrix((x, y, z)).transposed().to_euler("XYZ")


def _ik_circle_rotation(pb, arm_data) -> Euler:
    """Arrow-cross lies in XY (normal +Z). Hands: around the forearm. Feet: ground."""
    bone = arm_data.bones.get(pb.name)
    if bone is None:
        return Euler((0.0, 0.0, 0.0))
    src = Vector((0.0, 0.0, 1.0))
    if "Hand" in pb.name:
        dest = Vector((-1.0, 0.0, 0.0))
        return src.rotation_difference(dest).to_euler("XYZ")
    if "Head" in pb.name:
        dest = Vector((0.0, 1.0, 0.0))
        return src.rotation_difference(dest).to_euler("XYZ")
    if "Foot" in pb.name:
        up = _armature_world_up(pb)
        dest = _vec_in_bone(bone, up)
        x_hint = _vec_in_bone(bone, Vector((1.0, 0.0, 0.0)))
        return _euler_map_axis(src, dest, x_hint)
    return Euler((0.0, 0.0, 0.0))


def _foot_ik_rotation(pb, arm_data) -> Euler:
    """Footprint in XY (normal +Z): flat on the ground, +Y toward the toes."""
    bone = arm_data.bones.get(pb.name)
    if bone is None:
        return Euler((0.0, 0.0, 0.0))
    up = _vec_in_bone(bone, _armature_world_up(pb))
    side = _side_of(pb.name)
    toes = arm_data.bones.get(f"Toes_{side}") if side else None
    fwd = None
    if toes is not None:
        loc = bone.matrix.inverted().to_3x3() @ (toes.head_local - bone.head_local)
        if loc.length > 1e-4:
            fwd = loc - up * loc.dot(up)
    if fwd is None or fwd.length < 1e-5:
        fwd = _vec_in_bone(bone, Vector((0.0, 0.0, -1.0)))
        fwd = fwd - up * fwd.dot(up)
    if fwd.length < 1e-5:
        return _euler_map_axis(Vector((0.0, 0.0, 1.0)), up)
    fwd.normalize()
    up = up.normalized()
    x = fwd.cross(up)
    if x.length < 1e-5:
        return _euler_map_axis(Vector((0.0, 0.0, 1.0)), up)
    x.normalize()
    y = up.cross(x)
    y.normalize()
    x = y.cross(up)
    x.normalize()
    return Matrix((x, y, up)).transposed().to_euler("XYZ")


def _assign_sphere(pb, sphere, size: float) -> None:
    pb.custom_shape = sphere
    _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
    ext = _mesh_extents(sphere)
    s = size / max(ext.x, ext.y, ext.z) * 2.0
    pb.custom_shape_scale_xyz = Vector((s, s, s))


def _assign_pose_shape(
    pb,
    bone_spec: Dict[str, Any],
    widgets: Dict[str, Any],
    *,
    arm_data,
) -> None:
    """Rigify-style wires: FK boxes span joint-to-joint, circles for IK.

    ``use_custom_shape_bone_size`` is off so 5cm import bones do not shrink
    widgets. Rest matrices and F-curves are untouched.
    """
    role = _shape_role(pb.name)
    box = widgets.get("box")
    circle = widgets.get("circle")
    hoop = widgets.get("hoop")
    sphere = widgets.get("sphere")
    dump_shape = bone_spec.get("custom_shape")
    dump_obj = widgets.get(dump_shape) if dump_shape else None

    if hasattr(pb, "use_custom_shape_bone_size"):
        pb.use_custom_shape_bone_size = False
    if hasattr(pb, "custom_shape_translation"):
        pb.custom_shape_translation = Vector((0.0, 0.0, 0.0))

    joint = (
        _joint_local(arm_data, pb.name)
        if role in ("limb", "finger", "spine")
        else None
    )

    if role == "limb" and widgets.get("line") is not None and joint is not None:
        wgt = widgets["line"]
        pb.custom_shape = wgt
        src = Vector((0.0, 1.0, 0.0))
        _set_shape_rotation(
            pb, src.rotation_difference(joint.normalized()).to_euler("XYZ")
        )
        pb.custom_shape_scale_xyz = Vector((1.0, float(joint.length), 1.0))
    elif role == "finger":
        wgt = widgets.get("finger") or widgets.get("line")
        if wgt is None:
            return
        if joint is None:
            joint = _finger_joint(arm_data, pb.name)
        if joint is None or joint.length < 1e-4:
            return
        pb.custom_shape = wgt
        src = Vector((0.0, 1.0, 0.0))
        _set_shape_rotation(
            pb, src.rotation_difference(joint.normalized()).to_euler("XYZ")
        )
        pb.custom_shape_scale_xyz = Vector(
            (_FINGER_DOT, float(joint.length), _FINGER_DOT)
        )
    elif role == "limb" and box is not None and joint is not None:
        pb.custom_shape = box
        src = Vector((0.0, 1.0, 0.0))
        _set_shape_rotation(
            pb, src.rotation_difference(joint.normalized()).to_euler("XYZ")
        )
        ext = _mesh_extents(box)
        pb.custom_shape_scale_xyz = _span_scale(
            joint, ext.y, max(ext.x, ext.z), _thick_for(role, joint.length)
        )
    elif role == "spine" and hoop is not None:
        pb.custom_shape = hoop
        radius = _SPINE_HOOP.get(pb.name, 0.08) / 2.0
        joint = _spine_joint(pb, arm_data)
        back = _vec_in_pose_bone(pb, Vector((0.0, 0.0, 1.0)))
        if joint is not None and joint.length > 1e-4:
            _set_shape_rotation(pb, _spine_hoop_euler(joint, back))
            pb.custom_shape_scale_xyz = Vector(
                (float(joint.length), radius, radius)
            )
        else:
            _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
            pb.custom_shape_scale_xyz = Vector((1e-4, radius, radius))
    elif role == "head_ik_target":
        wgt = widgets.get("head_ik_target") or widgets.get("sphere")
        if wgt is None:
            return
        pb.custom_shape = wgt
        _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
        ext = _mesh_extents(wgt)
        s = _HEAD_IK_TARGET / max(ext.x, ext.y, ext.z)
        sq = _HEAD_IK_TARGET_SQUEEZE
        pb.custom_shape_scale_xyz = Vector((s * sq, s, s * sq))
        try:
            pb.custom_shape_wire_width = 1.0
        except Exception:
            pass
        try:
            pb.bone.show_wire = True
        except Exception:
            pass
        try:
            pb.custom_shape_translation = Vector((0.0, 0.0, 0.0))
        except Exception:
            pass
    elif role == "ik_target" and circle is not None:
        pb.custom_shape = circle
        _set_shape_rotation(pb, _ik_circle_rotation(pb, arm_data))
        ext = _mesh_extents(circle)
        s = (_IK_CIRCLE * 2.0) / max(ext.x, ext.y, ext.z)
        pb.custom_shape_scale_xyz = Vector((s, s, s))
    elif role == "hand_ik":
        wgt = widgets.get("hand_ik") or widgets.get("sphere")
        if wgt is None:
            return
        pb.custom_shape = wgt
        side = _side_of(pb.name)
        aim = None
        aim_name = (
            f"Toes_{side}" if "Foot" in pb.name else f"MiddleFinger1_{side}"
        )
        ba = arm_data.bones.get(pb.name)
        bb = arm_data.bones.get(aim_name)
        if ba and bb:
            loc = ba.matrix.inverted().to_3x3() @ (bb.head_local - ba.head_local)
            if loc.length > 1e-4:
                aim = loc
        if aim is None:
            aim = Vector((-1.0, 0.0, 0.0))
        src = Vector((0.0, 1.0, 0.0))
        _set_shape_rotation(
            pb, src.rotation_difference(aim.normalized()).to_euler("XYZ")
        )
        s = _HAND_IK_SIZE / _HAND_IK_EXTENT_REF
        pb.custom_shape_scale_xyz = Vector((s, s, s))
    elif role == "ui_panel":
        wgt = widgets.get("ui_panel")
        if wgt is None:
            return
        pb.custom_shape = wgt
        _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
        s = float(head_ik.UI_HEAD_POSE_SCALE)
        pb.custom_shape_scale_xyz = Vector((s, s, s))
    elif role == "ui_track":
        pb.custom_shape = None
        pb.custom_shape_scale_xyz = Vector((1.0, 1.0, 1.0))
    elif role == "ui_knob":
        wgt = widgets.get("ui_knob")
        if wgt is None:
            return
        pb.custom_shape = wgt
        _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
        s = float(head_ik.UI_HEAD_POSE_SCALE)
        pb.custom_shape_scale_xyz = Vector((s, s, s))
        try:
            pb.bone.show_wire = False
        except Exception:
            pass
    elif role == "ik_joint":
        side = _side_of(pb.name)
        wgt = widgets.get("line") or widgets.get("sphere")
        if wgt is None:
            return
        pb.custom_shape = wgt
        aim_name = None
        if pb.name.startswith("Ankle"):
            aim_name = f"Toes_{side}"
        elif pb.name.startswith("Wrist"):
            aim_name = f"MiddleFinger1_{side}"
        aim = _pose_joint(pb, aim_name) if aim_name else None
        if aim is None and aim_name:
            ba = arm_data.bones.get(pb.name)
            bb = arm_data.bones.get(aim_name)
            if ba and bb:
                loc = ba.matrix.inverted().to_3x3() @ (bb.head_local - ba.head_local)
                if loc.length > 1e-4:
                    aim = loc
        if aim is not None:
            src = Vector((0.0, 1.0, 0.0))
            _set_shape_rotation(
                pb, src.rotation_difference(aim.normalized()).to_euler("XYZ")
            )
            pb.custom_shape_scale_xyz = Vector((1.0, float(aim.length), 1.0))
        else:
            _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
            pb.custom_shape_scale_xyz = Vector((1.0, 0.05, 1.0))
    elif role == "root" and widgets.get("root_ring") is not None:
        wgt = widgets["root_ring"]
        pb.custom_shape = wgt
        _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
        ext = _mesh_extents(wgt)
        s = _ROOT_CIRCLE / max(ext.x, ext.z)
        pb.custom_shape_scale_xyz = Vector((s, s, s))
    elif role == "twist":
        wgt = widgets.get("twist") or widgets.get("sphere")
        if wgt is None:
            return
        pb.custom_shape = wgt
        axis = _twist_limb_axis(arm_data, pb.name)
        if axis is not None and axis.length > 1e-4:
            src = Vector((0.0, 1.0, 0.0))
            _set_shape_rotation(
                pb, src.rotation_difference(axis.normalized()).to_euler("XYZ")
            )
        else:
            _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
        ext = _mesh_extents(wgt)
        s = _TWIST_ARROW / max(ext.x, ext.y, ext.z)
        pb.custom_shape_scale_xyz = Vector((s, s, s))
        if hasattr(pb, "custom_shape_translation"):
            pb.custom_shape_translation = Vector((0.0, 0.0, 0.0))
        try:
            pb.bone.show_wire = True
        except Exception:
            pass
    elif role == "fx" and widgets.get("axes") is not None:
        wgt = widgets["axes"]
        pb.custom_shape = wgt
        _set_shape_rotation(pb, Euler((0.0, 0.0, 0.0)))
        pb.custom_shape_scale_xyz = Vector((_FX_AXES, _FX_AXES, _FX_AXES))
        try:
            pb.bone.show_wire = True
        except Exception:
            pass
    elif dump_obj is not None:
        pb.custom_shape = dump_obj
        rot = bone_spec.get("custom_shape_rotation")
        if rot is not None:
            _set_shape_rotation(pb, Euler(rot))
        raw = bone_spec.get("custom_shape_scale_xyz") or [1.0, 1.0, 1.0]
        dgb_len = max(float(bone_spec.get("length") or 0.0), 1e-8)
        pb.custom_shape_scale_xyz = Vector(
            (dgb_len * float(raw[0]), dgb_len * float(raw[1]), dgb_len * float(raw[2]))
        )
    else:
        return

    try:
        pb.custom_shape_wire_width = 1.0
    except Exception:
        pass


def _tone_for_bone(name: str):
    """HUD rest/select colors for every pose bone."""
    if name == "UI_Head" or name.endswith("_Track"):
        return _TONE_HUD, "THEME06"
    if name == "UI_Head_Kiss":
        return _TONE_KISS, "THEME01"
    if name == "UI_Head_IK" or name == "IK_Head_Target":
        return _TONE_IK, "THEME02"
    if name == "UI_Head_Spine":
        return _TONE_MAGENTA, "THEME06"
    is_ik = name in _IK_EFFECTORS or (
        name.startswith("Dummy_") and name.endswith("_IK")
    )
    if is_ik:
        return _TONE_IK, "THEME02"
    side = _side_of(name)
    if side == "L":
        return _TONE_CYAN, "THEME05"
    if side == "R":
        return _TONE_MAGENTA, "THEME06"
    return _TONE_VIOLET, "THEME06"


def _set_custom_tone(color, rgb, fallback: str) -> None:
    try:
        color.palette = "CUSTOM"
        color.custom.normal = rgb[0]
        color.custom.select = rgb[1]
        color.custom.active = rgb[2]
        return
    except Exception:
        pass
    try:
        color.palette = fallback
    except Exception:
        pass


def _apply_bone_tone(pb) -> None:
    """Pose + armature bone color (normal/select/active) from the HUD palette."""
    rgb, fallback = _tone_for_bone(pb.name)
    if hasattr(pb, "color"):
        _set_custom_tone(pb.color, rgb, fallback)
    bone = getattr(pb, "bone", None)
    if bone is not None and hasattr(bone, "color"):
        _set_custom_tone(bone.color, rgb, fallback)


def _apply_armature_display(arm_obj, snap: Dict[str, Any]) -> None:
    flags = snap.get("display") or {}
    dtype = flags.get("display_type")
    if dtype:
        try:
            arm_obj.data.display_type = dtype
        except Exception:
            pass
    if hasattr(arm_obj, "show_in_front"):
        arm_obj.show_in_front = bool(flags.get("show_in_front", True))
    show_axes = bool(flags.get("show_axes", False))
    show_names = bool(flags.get("show_names", False))
    data = arm_obj.data
    if hasattr(data, "show_axes"):
        data.show_axes = show_axes
    elif hasattr(arm_obj, "show_axis"):
        arm_obj.show_axis = show_axes
    if hasattr(data, "show_names"):
        data.show_names = show_names
    elif hasattr(arm_obj, "show_name"):
        arm_obj.show_name = show_names
    if hasattr(data, "show_bone_custom_shapes") and flags.get(
        "show_bone_custom_shapes"
    ) is not None:
        data.show_bone_custom_shapes = bool(flags["show_bone_custom_shapes"])


def _coll_get(colls, name: str):
    getter = getattr(colls, "get", None)
    if callable(getter):
        return getter(name)
    idx = colls.find(name)
    return colls[idx] if idx >= 0 else None


def _apply_collections(arm_obj) -> Dict[str, int]:
    data = arm_obj.data
    for old, new in _COLLECTION_LEGACY.items():
        col = _coll_get(data.collections, old)
        if col is None:
            continue
        if _coll_get(data.collections, new) is None:
            try:
                col.name = new
            except Exception:
                pass
    created = {}
    need_tail = any(
        b.name.startswith("Tail") or b.name == "Dummy_Tail_IK" for b in data.bones
    )
    for name in COLLECTION_ORDER:
        if name == "Tail" and not need_tail:
            continue
        col = _coll_get(data.collections, name)
        if col is None:
            col = data.collections.new(name)
        created[name] = col
        try:
            col.is_visible = name not in {"FX", "Extra"}
            col.is_solo = False
        except Exception:
            pass
    counts = {name: 0 for name in created}
    for bone in data.bones:
        dest_name = classify_bone(bone.name)
        dest = created.get(dest_name) or created.get("Extra")
        if dest is None:
            continue
        try:
            dest.assign(bone)
        except Exception:
            continue
        used = dest_name if dest_name in counts else "Extra"
        if used in counts:
            counts[used] += 1
        for col in list(bone.collections):
            if col != dest:
                try:
                    col.unassign(bone)
                except Exception:
                    pass
    for col in list(data.collections):
        if col.name in created:
            continue
        try:
            if len(col.bones) == 0:
                data.collections.remove(col)
        except Exception:
            pass
    return counts


def _remove_legacy_scene_head_target(arm_obj) -> None:
    """Drop leftover IK_Head_Target meshes that stole Neck Copy Rotation."""
    if arm_obj is None:
        return
    key = "gr2lab_head_target"
    doomed = []
    for existing in bpy.data.objects:
        if existing.type != "MESH":
            continue
        if existing.name.startswith(_WIDGET_PREFIX):
            continue
        tagged = existing.get(key) == arm_obj.name
        same_name = existing.name == head_ik._TARGET_OBJECT or existing.name.startswith(
            head_ik._TARGET_OBJECT + "."
        )
        parented = existing.parent == arm_obj and existing.parent_bone == head_ik._TARGET
        if tagged or (same_name and parented):
            doomed.append(existing)
    for obj in doomed:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def apply_imported_rig_display(
    context,
    arm_obj,
    *,
    mode_set: Optional[ModeSet] = None,
    shapes: bool = True,
    ik_hands: bool = True,
    ik_legs: bool = True,
    ik_head: bool = True,
    lookat_on: bool = False,
) -> None:
    """Collections, optional custom shapes, limb IK, head IK, ankle/wrist scale."""
    if arm_obj is None or getattr(arm_obj, "type", None) != "ARMATURE":
        return
    ensure_missing_end_bones(arm_obj, context, mode_set=mode_set)
    if not (shapes or ik_hands or ik_legs or ik_head):
        return
    snap = load_snapshot()
    _apply_armature_display(arm_obj, snap)
    counts = _apply_collections(arm_obj)

    widgets: Dict[str, Any] = {}
    widget_coll = None
    if shapes:
        widget_coll = _ensure_scene_collection(context, _WIDGET_COLL)
        for src_name, spec in (snap.get("widgets") or {}).items():
            widgets[src_name] = _ensure_widget(spec, widget_coll)
        widgets["box"] = _ensure_generated_widget("box", widget_coll, *_mesh_box())
        widgets["box_plain"] = _ensure_generated_widget(
            "box_plain", widget_coll, *_mesh_box(center_line=False)
        )
        widgets["line"] = _ensure_role_widget("line", widget_coll, _mesh_line)
        widgets["finger"] = _ensure_generated_widget(
            "finger", widget_coll, *_mesh_finger()
        )
        widgets["circle"] = _ensure_role_widget("circle", widget_coll, _mesh_arrow_cross)
        widgets["root_ring"] = _ensure_generated_widget(
            "root_ring", widget_coll, *_mesh_root_cross()
        )
        widgets["hoop"] = _ensure_generated_widget("hoop", widget_coll, *_mesh_hoop())
        widgets["sphere"] = _ensure_role_widget("sphere", widget_coll, _mesh_sphere)
        widgets["axes"] = _ensure_generated_widget(
            "axes", widget_coll, *_mesh_plain_axes()
        )
        src_twist = _find_scene_cs(("cs_arrow_twist",))
        if src_twist is not None:
            widgets["twist"] = _clone_mesh_widget(src_twist, "twist", widget_coll)
        else:
            widgets["twist"] = _ensure_generated_widget(
                "twist",
                widget_coll,
                *_mesh_arrow_twist(),
                replace_mesh=True,
            )
        widgets["head_ik_target"] = _ensure_generated_widget(
            "head_ik_target",
            widget_coll,
            *_mesh_head_ik_target(),
            replace_mesh=True,
        )
        widgets["hand"] = _ensure_role_widget("hand", widget_coll, _mesh_hand)
        widgets["hand_ik"] = _ensure_generated_widget(
            "hand_ik", widget_coll, *_mesh_hand_ik()
        )
        widgets["ui_track"] = _ensure_generated_widget(
            "ui_track", widget_coll, *_mesh_ui_track()
        )
        widgets["ui_knob"] = _ensure_generated_widget(
            "ui_knob",
            widget_coll,
            *_mesh_ui_knob(),
            replace_mesh=True,
        )
        widgets["ui_panel"] = _ensure_generated_widget(
            "ui_panel", widget_coll, *_mesh_ui_panel()
        )
        for stale in ("stick", "ring", "foot_l", "foot_r", "foot_ik_l", "foot_ik_r"):
            old = bpy.data.objects.get(_WIDGET_PREFIX + stale)
            if old is None:
                continue
            # Older imports may still use these as custom_shape.
            if getattr(old, "users", 0) > 0:
                continue
            bpy.data.objects.remove(old, do_unlink=True)

    ok: List[str] = []
    errors: List[str] = []
    scaled: List[str] = []
    scale_err: List[str] = []
    head_ok: List[str] = []
    head_err: List[str] = []
    if ik_hands or ik_legs:
        ok, errors = larian_ik.setup_larian_ik_constraints_on_armature(
            arm_obj, hands=ik_hands, legs=ik_legs
        )
        scaled, scale_err = larian_ik.scale_larian_ik_bone_lengths(
            arm_obj, context=context, mode_set=mode_set, hands=ik_hands, legs=ik_legs
        )
    if ik_head:
        try:
            head_ok, head_err = head_ik.setup_head_ik_on_armature(
                arm_obj, context=context, mode_set=mode_set, lookat_on=lookat_on
            )
            _remove_legacy_scene_head_target(arm_obj)
            kiss_obj = head_ik.ensure_kiss_object(arm_obj, context)
            head_ik.bind_lookat_object(arm_obj)
            head_ik.bind_kiss_object(arm_obj, kiss_obj)
        except Exception as exc:
            head_err = list(head_err) + [str(exc)]
            print(f"[GR2 LAB] Head IK/Lips failed (shapes still apply): {exc}")

    shaped = 0
    if shapes:
        by_name = {b["name"]: b for b in (snap.get("bones") or [])}
        for pb in arm_obj.pose.bones:
            spec = by_name.get(pb.name) or {}
            _assign_pose_shape(
                pb,
                spec,
                widgets,
                arm_data=arm_obj.data,
            )
            if pb.custom_shape:
                shaped += 1
        try:
            arm_obj.data.display_type = "STICK"
        except Exception:
            pass
        if widget_coll is not None:
            _layout_widget_catalog(widget_coll)

    for pb in arm_obj.pose.bones:
        _apply_bone_tone(pb)

    extra = counts.get("Extra", 0)
    print(
        f"[GR2 LAB] DGB display collections={sum(counts.values())} "
        f"shapes={shaped} ik={len(ok)} head_ik={len(head_ok)} "
        f"scaled={len(scaled)} extra={extra}"
    )
    for msg in errors + scale_err + head_err:
        print(f"[GR2 LAB] DGB display: {msg}")
