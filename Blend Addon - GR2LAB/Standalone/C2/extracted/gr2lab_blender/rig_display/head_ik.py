"""DGB-style head IK helpers for GR2 LAB imports (viewport only).

Reproduces Armature_KARLACH_DGB:
  IK_Head (child of Neck_M) → IK constraint to IK_Head_Target bone (chain 5)
  IK_Head.001 (connected visual extension)
  IK_Head_Target bone (child of Dummy_Root) — pose this; IK target
  Neck_M Copy Rotation from that bone (influence IK/2)
  Lips mesh — IK_Head_Target Copy Loc/Rot Target when Kiss is up
  UI_Head_* floating 2D HUD (IK / Extra / Kiss) parented to Dummy_Root
  Head_M / Neck_M / Chest_M copy-rotation constraints driven by those sliders

Helper bones are use_deform=False and must not be encoded into GR2.
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

HELPER_BONES = frozenset(
    (
        "IK_Head",
        "IK_Head.001",
        "IK_Head_Target",
        "UI_Head",
        "UI_Head_IK",
        "UI_Head_IK_Track",
        "UI_Head_Spine",
        "UI_Head_Spine_Track",
        "UI_Head_Kiss",
        "UI_Head_Kiss_Track",
    )
)

_TARGET = "IK_Head_Target"
_TARGET_OBJECT = "IK_Head_Target"
_IK_HEAD = "IK_Head"
_IK_HEAD_TIP = "IK_Head.001"
_UI_ROOT = "UI_Head"
_UI_IK = "UI_Head_IK"
_UI_SPINE = "UI_Head_Spine"
_UI_KISS = "UI_Head_Kiss"
_KISS_OBJECT = "Lips"
_KISS_KEY = "gr2lab_kiss_target"
_KISS_SIZE = 0.045
# Floating 2D HUD in UI_Head bone space (XY metres, origin = panel bottom-left).
UI_PANEL_W = 0.22
UI_PANEL_H = 0.18
UI_MARGIN = 0.012
# Default panel pivot from Head_M (armature space).
# After armature Rx+90°: left of the head, slightly forward and down.
# Yaw + uniform scale are baked into rest so pose TRS stays identity
# (same look as the old pose-mode yaw 90° / scale 3.23).
UI_HEAD_FROM_HEAD = Vector((-0.518, -0.125, -0.314))
UI_HEAD_POSE_YAW = math.radians(90.0)
UI_HEAD_POSE_SCALE = 3.23
UI_TITLE_H = 0.028
UI_KNOB_LEN = 0.030
UI_KNOB_WID = 0.016
UI_FIELD_H = 0.026
UI_LABEL_W = 0.066
UI_FIELD_PAD = 0.006
_KNOB_CLEARANCE = 0.008
_VISIBLE = frozenset(
    (
        _TARGET,
        _UI_ROOT,
        _UI_IK,
        _UI_SPINE,
        _UI_KISS,
    )
)

_COLL = {
    _IK_HEAD: "Extra",
    _IK_HEAD_TIP: "Extra",
    _TARGET: "IK",
    _UI_ROOT: "UI",
    _UI_IK: "UI",
    _UI_SPINE: "UI",
    _UI_KISS: "UI",
    f"{_UI_IK}_Track": "UI",
    f"{_UI_SPINE}_Track": "UI",
    f"{_UI_KISS}_Track": "UI",
}

ModeSet = Callable[..., None]


def is_helper_bone(name: str) -> bool:
    return name in HELPER_BONES or name.startswith("UI_Head")


def _set_mode(ctx, mode: str, arm_obj, mode_set: Optional[ModeSet]) -> None:
    if mode_set is not None:
        mode_set(ctx, mode, active=arm_obj)
        return
    ctx.view_layer.objects.active = arm_obj
    if arm_obj.mode != mode:
        bpy.ops.object.mode_set(mode=mode)


def ui_field_x() -> float:
    return UI_MARGIN + UI_LABEL_W


def ui_field_x1() -> float:
    return UI_PANEL_W - UI_MARGIN


def ui_knob_inset() -> float:
    """Keep the knob capsule inside the field's rounded caps."""
    r_track = UI_FIELD_H * 0.5
    r_knob = UI_KNOB_WID * 0.5
    return max(r_track - r_knob, 0.0) + _KNOB_CLEARANCE


def ui_knob_rest_x() -> float:
    return ui_field_x() + UI_KNOB_LEN * 0.5 + ui_knob_inset()


def ui_travel() -> float:
    """Knob travel in unscaled panel-widget metres (field minus knob + insets)."""
    span = ui_field_x1() - ui_field_x()
    return max(span - UI_KNOB_LEN - 2.0 * ui_knob_inset(), 1e-4)


def ui_travel_pose() -> float:
    """Knob pose-local Y travel after the HUD scale is baked / applied to widgets."""
    return ui_travel() * float(UI_HEAD_POSE_SCALE)


def ui_row_y(index: int) -> float:
    """Vertical center of slider row 0=IK, 1=Kiss, 2=Extra (panel space)."""
    top = UI_PANEL_H - UI_MARGIN - UI_TITLE_H
    bottom = UI_MARGIN + 0.010
    row_h = (top - bottom) / 3.0
    return top - (index + 0.5) * row_h


def ui_panel_center() -> Vector:
    """Pivot of UI_Head — middle of the card, not the bottom-left corner."""
    return Vector((UI_PANEL_W * 0.5, UI_PANEL_H * 0.5, 0.0))


def _ensure_edit_bone(
    ebones,
    name: str,
    *,
    parent,
    head,
    tail,
    use_connect: bool,
    roll=None,
):
    bone = ebones.get(name)
    if bone is None:
        bone = ebones.new(name)
    bone.parent = parent
    bone.use_connect = bool(use_connect)
    bone.head = Vector(head)
    bone.tail = Vector(tail)
    bone.use_deform = False
    aim = Vector(roll) if roll is not None else Vector((0.0, 1.0, 0.0))
    try:
        bone.align_roll(aim)
    except Exception:
        pass
    return bone


def _float_slider(pb, name: str, value: float, description: str) -> None:
    pb[name] = float(value)
    try:
        ui = pb.id_properties_ui(name)
        ui.update(min=0.0, max=1.0, default=float(value), description=description)
    except Exception:
        pass
    try:
        pb.property_overridable_library_set(f'["{name}"]', True)
    except Exception:
        pass


def _ensure_constraint(pb, con_type: str, name: str):
    for c in pb.constraints:
        if c.name == name:
            return c
    con = pb.constraints.new(con_type)
    con.name = name
    return con


def _prop_path(var_name: str) -> str:
    if var_name in ("_Kiss", "_Influence"):
        return f'pose.bones["{_TARGET}"]["!Kiss"]'
    if var_name == "Extra":
        return f'pose.bones["{_TARGET}"]["Extra"]'
    return f'pose.bones["{_TARGET}"]["IK"]'


def _remove_driver(arm_obj, data_path: str) -> None:
    ad = arm_obj.animation_data
    if ad is None:
        return
    for fcu in list(ad.drivers):
        if fcu.data_path == data_path:
            try:
                ad.drivers.remove(fcu)
            except Exception:
                pass


def _drive_from_ui_bone(arm_obj, data_path: str, ui_bone: str) -> None:
    """Map UI slider local Y (0..ui_travel_pose()) onto 0..1."""
    travel = ui_travel_pose()
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    _remove_driver(arm_obj, data_path)
    fcu = arm_obj.driver_add(data_path)
    drv = fcu.driver
    drv.type = "SCRIPTED"
    drv.expression = f"min(1.0, max(0.0, loc / {travel}))"
    while drv.variables:
        drv.variables.remove(drv.variables[0])
    var = drv.variables.new()
    var.name = "loc"
    var.type = "SINGLE_PROP"
    tgt = var.targets[0]
    tgt.id_type = "OBJECT"
    tgt.id = arm_obj
    tgt.data_path = f'pose.bones["{ui_bone}"].location[1]'


def _lock_slider(pb) -> None:
    pb.lock_location = (True, False, True)
    pb.lock_rotation = (True, True, True)
    pb.lock_scale = (True, True, True)
    try:
        pb.lock_rotation_w = True
    except Exception:
        pass
    lim = None
    for c in pb.constraints:
        if c.type == "LIMIT_LOCATION" and c.name == "UI Slider":
            lim = c
            break
    if lim is None:
        lim = pb.constraints.new("LIMIT_LOCATION")
        lim.name = "UI Slider"
    lim.owner_space = "LOCAL"
    lim.use_min_x = lim.use_max_x = True
    lim.use_min_y = lim.use_max_y = True
    lim.use_min_z = lim.use_max_z = True
    lim.min_x = lim.max_x = 0.0
    lim.min_z = lim.max_z = 0.0
    lim.min_y = 0.0
    lim.max_y = ui_travel_pose()
    try:
        lim.use_transform_limit = True
    except Exception:
        pass


def _constraint_to_index(pb, name: str, index: int) -> None:
    names = [c.name for c in pb.constraints]
    if name not in names:
        return
    src = names.index(name)
    try:
        pb.constraints.move(src, index)
    except Exception:
        pass


_KISS_MESH = "GR2LAB_Lips"


def _kiss_mesh():
    """Two rings + cross — the Copy Loc/Rot target mesh. Not the bone arrow."""
    verts = []
    edges = []
    for radius, segs in ((1.0, 24), (0.42, 16)):
        base = len(verts)
        for i in range(segs):
            a = (2.0 * math.pi * i) / segs
            verts.append((math.cos(a), 0.0, math.sin(a)))
            edges.append((base + i, base + ((i + 1) % segs)))
    gap = 0.10
    for p0, p1 in (
        ((-1.0, 0.0, 0.0), (-gap, 0.0, 0.0)),
        ((gap, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, -gap)),
        ((0.0, 0.0, gap), (0.0, 0.0, 1.0)),
    ):
        b = len(verts)
        verts.extend((p0, p1))
        edges.append((b, b + 1))
    me = bpy.data.meshes.get(_KISS_MESH)
    if me is None:
        me = bpy.data.meshes.new(_KISS_MESH)
        me.from_pydata(verts, edges, [])
        me.update()
        return me
    if len(me.vertices) >= 40:
        return me
    fresh = bpy.data.meshes.new(_KISS_MESH)
    fresh.from_pydata(verts, edges, [])
    fresh.update()
    return fresh


def _mouth_world(arm_obj) -> Vector:
    for name in ("Dummy_MouthFX", "Dummy_HeadFX", "Head_M"):
        pb = arm_obj.pose.bones.get(name)
        if pb is not None:
            loc = (arm_obj.matrix_world @ pb.matrix).to_translation()
            fwd = (arm_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
            return loc + fwd * 0.03
    return arm_obj.matrix_world.translation.copy()


def _target_rest_world(arm_obj) -> Matrix:
    """IK_Head_Target rest in world — Kiss Copy Rotation baseline."""
    bone = arm_obj.data.bones.get(_TARGET)
    if bone is None:
        return arm_obj.matrix_world.copy()
    return arm_obj.matrix_world @ bone.matrix_local


def _kiss_world_matrix(arm_obj, *, location: Optional[Vector] = None) -> Matrix:
    """Lips gizmo: mouth location, IK_Head_Target rest rotation.

    Copy Rotation WORLD from Lips onto IK_Head_Target. Lips used to copy
    Head_M rest (~90° off the Dummy-root look target), which slammed Head
    and Neck when Kiss=1. Match the target rest so Kiss at 1 only moves
    the aim point to the mouth unless the user rotates Lips.
    """
    rest = _target_rest_world(arm_obj)
    mw = rest.to_quaternion().to_matrix().to_4x4()
    mw.translation = Vector(location) if location is not None else _mouth_world(arm_obj)
    return mw


def _parent_object_to_armature(obj, arm_obj) -> None:
    """Object-parent Lips so moving the armature takes it along.

    Keep world location if already parented (user placed the mouth target).
    Always refresh rotation to the look-target rest so old Head_M-aligned
    gizmos stop snapping the skull when Kiss is up.
    """
    if obj is None or arm_obj is None:
        return
    already = obj.parent == arm_obj and getattr(obj, "parent_type", "OBJECT") == "OBJECT"
    keep_loc = obj.matrix_world.translation.copy() if already else None
    obj.matrix_world = _kiss_world_matrix(arm_obj, location=keep_loc)
    mw = obj.matrix_world.copy()
    obj.parent = arm_obj
    try:
        obj.parent_type = "OBJECT"
    except Exception:
        pass
    try:
        obj.parent_bone = ""
    except Exception:
        pass
    obj.matrix_world = mw


def _kiss_name_for_arm(arm_obj) -> str:
    raw = (getattr(arm_obj, "name", None) or "Armature").strip()
    if raw.startswith("GR2LAB_"):
        raw = raw[7:]
    name = f"Lips_{raw}" if raw else "Lips"
    return name[:63]


def ensure_kiss_object(arm_obj, context=None, src_wgt=None) -> Optional[Any]:
    """Scene mesh for IK_Head_Target Copy Location / Copy Rotation Target.

    One Lips per armature (Create all / many imports). Never hijack another
    character's gizmo, and never steal a non-mesh named Lips.
    """
    if arm_obj is None:
        return None
    obj = None
    for existing in bpy.data.objects:
        if existing.type != "MESH":
            continue
        if existing.get(_KISS_KEY) == arm_obj.name:
            obj = existing
            break
    reticle = _kiss_mesh()
    if obj is None:
        obj = bpy.data.objects.new(_kiss_name_for_arm(arm_obj), reticle)
        obj[_KISS_KEY] = arm_obj.name
        linked = False
        ctx = context or bpy.context
        for coll in arm_obj.users_collection:
            try:
                coll.objects.link(obj)
                linked = True
            except RuntimeError:
                pass
            break
        if not linked:
            try:
                ctx.scene.collection.objects.link(obj)
            except RuntimeError:
                pass
    elif obj.type == "MESH":
        obj.data = reticle
        obj[_KISS_KEY] = arm_obj.name
    else:
        return None
    _parent_object_to_armature(obj, arm_obj)
    _detach_lips_action(obj)
    obj.scale = (_KISS_SIZE, _KISS_SIZE, _KISS_SIZE)
    obj.hide_render = True
    obj.hide_viewport = False
    try:
        obj.hide_select = False
    except Exception:
        pass
    obj.display_type = "WIRE"
    try:
        obj.show_in_front = True
    except Exception:
        pass
    return obj


def bind_lookat_object(arm_obj, obj=None) -> None:
    """Neck Copy Rotation from the IK_Head_Target bone, matching DGB Karlach.

    Target space is LOCAL_OWNER_ORIENT so rotating the chevron remaps Dummy-root
    local axes onto Neck_M. LOCAL alone barely twists the torso through the
    IK chain.
    """
    if arm_obj is None:
        return
    pb = arm_obj.pose.bones.get(_TARGET)
    if pb is None:
        return
    for name in ("Follow Object", "Follow Object Rot"):
        for c in list(pb.constraints):
            if c.name == name:
                try:
                    pb.constraints.remove(c)
                except Exception:
                    pass
    pb_ik = arm_obj.pose.bones.get(_IK_HEAD)
    if pb_ik is not None:
        for c in pb_ik.constraints:
            if c.type == "IK" and c.name == "IK":
                # DGB: aim the chain at the target. Rotation matching copies
                # Dummy-root axes onto Head_M and snaps the skull ~90°.
                c.use_rotation = False
                break
    pb_neck = arm_obj.pose.bones.get("Neck_M")
    if pb_neck is None:
        return
    ncr = None
    for c in pb_neck.constraints:
        if c.type == "COPY_ROTATION" and c.name == "Copy Rotation":
            ncr = c
            break
    if ncr is None:
        return
    ncr.target = arm_obj
    ncr.subtarget = _TARGET
    ncr.owner_space = "LOCAL_WITH_PARENT"
    ncr.target_space = "LOCAL_OWNER_ORIENT"
    ncr.use_x = ncr.use_y = ncr.use_z = True
    ncr.invert_x = ncr.invert_y = ncr.invert_z = False
    try:
        ncr.mix_mode = "REPLACE"
    except Exception:
        pass
    if hasattr(ncr, "use_offset"):
        ncr.use_offset = False


def bind_kiss_object(arm_obj, obj) -> None:
    """IK_Head_Target Copy Loc/Rot Target = scene mesh object, not a bone."""
    if arm_obj is None or obj is None:
        return
    pb = arm_obj.pose.bones.get(_TARGET)
    if pb is None:
        return
    for c in pb.constraints:
        if c.type in {"COPY_LOCATION", "COPY_ROTATION"} and c.name in (
            "Copy Location",
            "Copy Rotation",
        ):
            c.target = obj
            c.subtarget = ""
            c.target_space = "WORLD"
            c.owner_space = "WORLD"


def _set_influence_driver(
    arm_obj,
    data_path: str,
    *,
    var_name: str,
    expression: str,
    scripted: bool,
) -> None:
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    _remove_driver(arm_obj, data_path)
    fcu = arm_obj.driver_add(data_path)
    drv = fcu.driver
    drv.type = "SCRIPTED" if scripted else "AVERAGE"
    drv.expression = expression
    while drv.variables:
        drv.variables.remove(drv.variables[0])
    var = drv.variables.new()
    var.name = var_name
    var.type = "SINGLE_PROP"
    tgt = var.targets[0]
    tgt.id_type = "OBJECT"
    tgt.id = arm_obj
    tgt.data_path = _prop_path(var_name)


def _set_head_follow_ik_driver(arm_obj) -> None:
    """Head follows IK_Head while look-at IK is on.

    Kiss must not zero this: DGB multiplied by ``1-_Kiss`` and swapped in a
    WORLD Replace from IK_Head, which snaps our Y-up skull ~90°. Kiss only
    moves IK_Head_Target to the Lips gizmo; look-at stays on.
    """
    data_path = 'pose.bones["Head_M"].constraints["Copy Rotation"].influence'
    if arm_obj.pose.bones.get("Head_M") is None:
        return
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    _remove_driver(arm_obj, data_path)
    fcu = arm_obj.driver_add(data_path)
    drv = fcu.driver
    drv.type = "SCRIPTED"
    drv.expression = "IK"
    while drv.variables:
        drv.variables.remove(drv.variables[0])
    var = drv.variables.new()
    var.name = "IK"
    var.type = "SINGLE_PROP"
    tgt = var.targets[0]
    tgt.id_type = "OBJECT"
    tgt.id = arm_obj
    tgt.data_path = _prop_path("IK")


def release_head_ik_for_clip(arm_obj) -> None:
    """Turn look-at off so an applied clip owns Head/Neck/Chest."""
    if arm_obj is None or getattr(arm_obj, "type", None) != "ARMATURE":
        return
    pb_ui = arm_obj.pose.bones.get(_UI_IK)
    if pb_ui is not None:
        pb_ui.location[1] = 0.0
    pb_extra = arm_obj.pose.bones.get(_UI_SPINE)
    if pb_extra is not None:
        pb_extra.location[1] = 0.0
    pb_tgt = arm_obj.pose.bones.get(_TARGET)
    if pb_tgt is not None:
        if "IK" in pb_tgt:
            pb_tgt["IK"] = 0.0
        if "Extra" in pb_tgt:
            pb_tgt["Extra"] = 0.0
    _set_head_follow_ik_driver(arm_obj)
    print("[GR2 LAB] head look-at off for clip — slide UI_Head_IK (or IK prop) to pose")


def _apply_head_ik_sliders(arm_obj, ik: float, kiss: float) -> None:
    travel = ui_travel_pose()
    for name, amount in ((_UI_IK, ik), (_UI_KISS, kiss)):
        pb = arm_obj.pose.bones.get(name)
        if pb is not None:
            pb.location[1] = float(travel * max(0.0, min(1.0, amount)))


def _evaluated_arm(context, arm_obj):
    context.view_layer.update()
    return arm_obj.evaluated_get(context.evaluated_depsgraph_get())


def _head_world_matrix(eval_arm) -> Matrix:
    pb = eval_arm.pose.bones.get("Head_M")
    if pb is None:
        return Matrix.Identity(4)
    return eval_arm.matrix_world @ pb.matrix


def _clip_head_reference(context, arm_obj, frame: int) -> Matrix:
    scene = context.scene
    scene.frame_set(frame)
    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
    return _head_world_matrix(_evaluated_arm(context, arm_obj))


def _heuristic_target_pose(eval_arm, e_ik, e_head, e_tgt, rot_delta, tip_offset):
    """Aim point at the IK chain tip (DGB), rotation tracks Head_M in parent space."""
    head_q = _head_quat_in_parent(eval_arm, e_head, e_tgt)
    tip_p = _ik_tip_in_parent(eval_arm, e_ik, e_tgt)
    loc = tip_p + tip_offset
    rot = (head_q @ rot_delta).normalized()
    return loc, rot


def _arm_at_clip_frame(context, arm_obj, frame: int):
    context.scene.frame_set(frame)
    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
    return _evaluated_arm(context, arm_obj)


def _mute_target_fcurves(arm_obj, mute: bool) -> None:
    ad = arm_obj.animation_data
    if ad is None or ad.action is None:
        return
    from ..bridge_blender import _iter_action_fcurves

    for fc in _iter_action_fcurves(ad.action, anim_data=ad):
        if _TARGET in (fc.data_path or ""):
            fc.mute = mute


def _pin_target_to_ik_tail(arm_obj) -> Optional[Vector]:
    """Pose IK_Head_Target so its head sits on the animated IK_Head tail."""
    ik = arm_obj.pose.bones.get(_IK_HEAD)
    tgt = arm_obj.pose.bones.get(_TARGET)
    if ik is None or tgt is None or tgt.bone is None:
        return None
    tgt.matrix = Matrix.LocRotScale(
        Vector(ik.tail),
        tgt.bone.matrix_local.to_quaternion(),
        Vector((1.0, 1.0, 1.0)),
    )
    return Vector(tgt.location)


def _pose_chain_root_first(*bones):
    """Ancestors of the given pose bones, parents before children."""
    pending = {}
    for pb in bones:
        cur = pb
        while cur is not None:
            pending[cur.name] = cur
            cur = cur.parent
    ordered = []
    while pending:
        progressed = False
        for name, pb in list(pending.items()):
            parent = pb.parent
            if parent is None or parent.name not in pending:
                ordered.append(pb)
                del pending[name]
                progressed = True
        if not progressed:
            ordered.extend(pending.values())
            break
    return ordered


def _channel_samples(fc, frames, default: float) -> List[float]:
    if fc is None:
        return [default] * len(frames)
    return [float(fc.evaluate(float(fr))) for fr in frames]


def _fk_target_locations(arm_obj, action, frames, *, anim_data=None) -> Optional[List[Vector]]:
    """IK_Head_Target locations from F-curves, without scene.frame_set.

    Same result as pinning the target onto the FK chain tip when head IK
    influence is 0. Returns None if the channels cannot be read.
    """
    ik = arm_obj.pose.bones.get(_IK_HEAD)
    tgt = arm_obj.pose.bones.get(_TARGET)
    if ik is None or tgt is None or ik.bone is None or tgt.bone is None:
        return None
    if not frames:
        return []
    from ..bridge_blender import _iter_action_fcurves

    chain = _pose_chain_root_first(ik, tgt)
    curves = {}
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        path = fc.data_path or ""
        if not path.startswith('pose.bones["'):
            continue
        try:
            name, prop = path[len('pose.bones["') :].split('"].', 1)
        except ValueError:
            continue
        slot = curves.get(name)
        if slot is None:
            slot = {
                "location": [None, None, None],
                "rotation_quaternion": [None, None, None, None],
                "rotation_euler": [None, None, None],
                "scale": [None, None, None],
            }
            curves[name] = slot
        arr = slot.get(prop)
        if arr is None:
            continue
        idx = int(fc.array_index)
        if 0 <= idx < len(arr):
            arr[idx] = fc

    sampled = {}
    for pb in chain:
        slot = curves.get(pb.name) or {}
        loc_c = slot.get("location") or [None, None, None]
        quat_c = slot.get("rotation_quaternion") or [None, None, None, None]
        euler_c = slot.get("rotation_euler") or [None, None, None]
        scale_c = slot.get("scale") or [None, None, None]
        sampled[pb.name] = {
            "mode": pb.rotation_mode,
            "loc": [
                _channel_samples(loc_c[i], frames, 0.0) for i in range(3)
            ],
            "quat": [
                _channel_samples(quat_c[0], frames, 1.0),
                _channel_samples(quat_c[1], frames, 0.0),
                _channel_samples(quat_c[2], frames, 0.0),
                _channel_samples(quat_c[3], frames, 0.0),
            ],
            "euler": [
                _channel_samples(euler_c[i], frames, 0.0) for i in range(3)
            ],
            "scale": [
                _channel_samples(scale_c[i], frames, 1.0) for i in range(3)
            ],
            "rest": pb.bone.matrix_local.copy(),
            "rest_inv": pb.bone.matrix_local.inverted(),
            "parent": pb.parent.name if pb.parent else "",
        }

    rest_rot = tgt.bone.matrix_local.to_quaternion()
    ik_len = float(ik.bone.length)
    tgt_rest = sampled[tgt.name]["rest"]
    tgt_parent = sampled[tgt.name]["parent"]
    parent_rest = sampled[tgt_parent]["rest"] if tgt_parent else None
    out: List[Vector] = []
    n = len(frames)
    for i in range(n):
        mats = {}
        for pb in chain:
            ch = sampled[pb.name]
            loc = Vector((ch["loc"][0][i], ch["loc"][1][i], ch["loc"][2][i]))
            scale = Vector((ch["scale"][0][i], ch["scale"][1][i], ch["scale"][2][i]))
            if ch["mode"] == "QUATERNION":
                rot = Quaternion(
                    (ch["quat"][0][i], ch["quat"][1][i], ch["quat"][2][i], ch["quat"][3][i])
                )
                if rot.magnitude > 1.0e-8:
                    rot.normalize()
            else:
                rot = Euler(
                    (ch["euler"][0][i], ch["euler"][1][i], ch["euler"][2][i]),
                    ch["mode"],
                )
            basis = Matrix.LocRotScale(loc, rot, scale)
            parent = ch["parent"]
            if not parent:
                mats[pb.name] = ch["rest"] @ basis
            else:
                mats[pb.name] = (
                    mats[parent]
                    @ sampled[parent]["rest_inv"]
                    @ ch["rest"]
                    @ basis
                )
        tail = mats[ik.name] @ Vector((0.0, ik_len, 0.0))
        desired = Matrix.LocRotScale(tail, rest_rot, Vector((1.0, 1.0, 1.0)))
        if not tgt_parent:
            basis_t = tgt_rest.inverted() @ desired
        else:
            basis_t = (
                tgt_rest.inverted()
                @ parent_rest
                @ mats[tgt_parent].inverted()
                @ desired
            )
        out.append(basis_t.to_translation())
    return out


def _bone_world(arm_obj, name: str):
    pb = arm_obj.pose.bones.get(name)
    if pb is None:
        return None
    return arm_obj.matrix_world @ pb.matrix


def _calibrate_ik_target_rotation(context, arm_obj) -> Quaternion:
    """One pose quaternion so IK=1 (neck copy included) leaves head/spine put.

    Location must already be pinned to the chain tip. The offset is rig-constant
    on this helper layout, so it is solved once and reused for every frame.
    """
    bones = ("Head_M", "Neck_M", "Chest_M", "Spine2_M", "Spine1_M")
    pb_ik = arm_obj.pose.bones.get(_IK_HEAD)
    pb_tgt = arm_obj.pose.bones.get(_TARGET)
    pb_head = arm_obj.pose.bones.get("Head_M")
    pb_neck = arm_obj.pose.bones.get("Neck_M")
    if pb_ik is None or pb_tgt is None or pb_head is None or pb_neck is None:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    ik_c = pb_ik.constraints.get("IK")
    head_c = pb_head.constraints.get("Copy Rotation")
    neck_c = pb_neck.constraints.get("Copy Rotation")
    if ik_c is None or neck_c is None:
        return Quaternion(pb_tgt.rotation_quaternion)

    ad = arm_obj.animation_data
    muted = []
    if ad is not None and ad.drivers:
        for drv in ad.drivers:
            if not drv.mute:
                drv.mute = True
                muted.append(drv)

    def _set_inf(ik_v, head_v, neck_v):
        ik_c.influence = ik_v
        if head_c is not None:
            head_c.influence = head_v
        neck_c.influence = neck_v

    _set_inf(0.0, 0.0, 0.0)
    context.view_layer.update()
    eval_arm = arm_obj.evaluated_get(context.evaluated_depsgraph_get())
    ref = {}
    for name in bones:
        pb = eval_arm.pose.bones.get(name)
        if pb is None:
            continue
        mat = eval_arm.matrix_world @ pb.matrix
        ref[name] = (mat.translation.copy(), mat.to_quaternion())
    base_loc = Vector(pb_tgt.location)

    def _neck_angle(quat: Quaternion) -> float:
        pb_tgt.location = base_loc
        pb_tgt.rotation_quaternion = quat
        _set_inf(0.0, 0.0, 1.0)
        context.view_layer.update()
        eval_arm = arm_obj.evaluated_get(context.evaluated_depsgraph_get())
        pb = eval_arm.pose.bones.get("Neck_M")
        if pb is None:
            return 99.0
        got = (eval_arm.matrix_world @ pb.matrix).to_quaternion()
        ref_q = ref["Neck_M"][1]
        got.make_compatible(ref_q)
        return got.rotation_difference(ref_q).angle

    best = Quaternion((1.0, 0.0, 0.0, 0.0))
    best_e = _neck_angle(best)
    step = 0.5
    for _ in range(16):
        improved = False
        for axis in range(3):
            for sign in (1.0, -1.0):
                euler = best.to_euler("XYZ")
                euler[axis] += sign * step
                trial = euler.to_quaternion()
                err = _neck_angle(trial)
                if err < best_e - 1e-7:
                    best = trial
                    best_e = err
                    improved = True
        step *= 0.55
        if not improved and step < 0.002:
            break

    _set_inf(0.0, 0.0, 0.0)
    context.view_layer.update()
    return best


def _refine_neutral_quat(context, arm_obj, quat: Quaternion, *, rounds: int = 2) -> Quaternion:
    """Nudge the rig-constant aim rotation so this frame's neck copy stays neutral."""
    pb_tgt = arm_obj.pose.bones.get(_TARGET)
    pb_ik = arm_obj.pose.bones.get(_IK_HEAD)
    pb_neck = arm_obj.pose.bones.get("Neck_M")
    pb_head = arm_obj.pose.bones.get("Head_M")
    if pb_tgt is None or pb_neck is None or pb_ik is None:
        return quat
    neck_c = pb_neck.constraints.get("Copy Rotation")
    ik_c = pb_ik.constraints.get("IK") if pb_ik else None
    head_c = pb_head.constraints.get("Copy Rotation") if pb_head else None
    if neck_c is None:
        return quat
    if ik_c is not None:
        ik_c.influence = 0.0
    if head_c is not None:
        head_c.influence = 0.0
    neck_c.influence = 0.0
    context.view_layer.update()
    pinned = _pin_target_to_ik_tail(arm_obj)
    if pinned is None:
        return quat
    context.view_layer.update()
    eval_arm = arm_obj.evaluated_get(context.evaluated_depsgraph_get())
    neck = eval_arm.pose.bones.get("Neck_M")
    if neck is None:
        return quat
    ref_q = (eval_arm.matrix_world @ neck.matrix).to_quaternion()
    loc = Vector(pb_tgt.location)

    def _ang(trial: Quaternion) -> float:
        pb_tgt.location = loc
        pb_tgt.rotation_quaternion = trial
        neck_c.influence = 1.0
        context.view_layer.update()
        ev = arm_obj.evaluated_get(context.evaluated_depsgraph_get())
        got_pb = ev.pose.bones.get("Neck_M")
        if got_pb is None:
            return 99.0
        got = (ev.matrix_world @ got_pb.matrix).to_quaternion()
        got.make_compatible(ref_q)
        return got.rotation_difference(ref_q).angle

    best = Quaternion(quat)
    best_e = _ang(best)
    step = 0.06
    for _ in range(rounds):
        for axis in range(3):
            for sign in (1.0, -1.0):
                euler = best.to_euler("XYZ")
                euler[axis] += sign * step
                trial = euler.to_quaternion()
                err = _ang(trial)
                if err < best_e - 1e-7:
                    best = trial
                    best_e = err
        step *= 0.45
    neck_c.influence = 0.0
    if ik_c is not None:
        ik_c.influence = 0.0
    pb_tgt.location = loc
    pb_tgt.rotation_quaternion = best
    return best


def _target_pose_from_anchor_tip(
    eval_arm, e_ik, e_head, e_tgt, tip_offset: Vector, rot_delta: Quaternion
) -> Tuple[Vector, Quaternion]:
    """Frame 1 style: tip + fixed offset, rotation follows Head_M (IK toggle-safe)."""
    tip_p = _ik_tip_in_parent(eval_arm, e_ik, e_tgt)
    head_q = _head_quat_in_parent(eval_arm, e_head, e_tgt)
    loc = tip_p + tip_offset
    rot = (head_q @ rot_delta).normalized()
    return loc, rot


def _refine_to_match_head(
    context,
    arm_obj,
    pb_tgt,
    ref_m: Matrix,
    loc: Vector,
    quat: Quaternion,
    *,
    ik: float,
    kiss: float,
    lips_obj=None,
    lips_world: Optional[Matrix] = None,
    max_iter: int = 16,
) -> Tuple[Vector, Quaternion, Optional[Matrix]]:
    """Nudge target / Lips so evaluated Head_M matches the clip (IK=0) pose."""
    lips_m = lips_world.copy() if lips_world is not None else None
    for _ in range(max_iter):
        pb_tgt.location = loc
        pb_tgt.rotation_quaternion = quat
        _apply_head_ik_sliders(arm_obj, ik, kiss)
        if lips_obj is not None and lips_m is not None and kiss > 0.05:
            lips_obj.matrix_world = lips_m
        cur = _head_world_matrix(_evaluated_arm(context, arm_obj))
        pos_err = (cur.translation - ref_m.translation).length
        rq = ref_m.to_quaternion()
        cq = cur.to_quaternion()
        cq.make_compatible(rq)
        ang_err = cq.rotation_difference(rq).angle
        if pos_err < 0.00035 and ang_err < math.radians(0.2):
            break
        world_delta = ref_m.translation - cur.translation
        fix_q = (rq @ cq.inverted()).normalized()
        if kiss < 0.05:
            parent_pb = pb_tgt.parent
            if parent_pb is not None:
                eval_arm = _evaluated_arm(context, arm_obj)
                p_w = _parent_world(eval_arm, eval_arm.pose.bones.get(_TARGET))
                loc += p_w.inverted().to_3x3() @ (world_delta * 0.5)
            quat = (fix_q @ quat).normalized()
        elif lips_obj is not None and lips_m is not None:
            lips_m.translation += world_delta * 0.55
            fix_loc = fix_q @ (lips_m.translation - ref_m.translation)
            lips_m.translation = ref_m.translation + fix_loc * 0.15 + world_delta * 0.45
    return loc, quat, lips_m


def _find_lips_for_arm(arm_obj):
    key = getattr(arm_obj, "name", "") or ""
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.get(_KISS_KEY) == key:
            return obj
    return None


def _detach_lips_action(lips_obj) -> None:
    """Lips must not share the armature Action (breaks slotted rigs on Blender 5)."""
    if lips_obj is None:
        return
    lad = lips_obj.animation_data
    if lad is None:
        return
    try:
        lad.action = None
    except Exception:
        pass
    if hasattr(lad, "action_slot"):
        try:
            lad.action_slot = None
        except Exception:
            pass


def _simplify_vec3_keys(
    frame_list: List[int],
    vx: List[float],
    vy: List[float],
    vz: List[float],
    *,
    pos_tol: float = 0.0012,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    n = len(frame_list)
    if n <= 2:
        return frame_list, vx, vy, vz

    def _seg_err(i0: int, i1: int, im: int) -> float:
        t0, t1, tm = float(frame_list[i0]), float(frame_list[i1]), float(frame_list[im])
        if t1 <= t0:
            return 0.0
        u = (tm - t0) / (t1 - t0)
        p0 = Vector((vx[i0], vy[i0], vz[i0]))
        p1 = Vector((vx[i1], vy[i1], vz[i1]))
        pm = Vector((vx[im], vy[im], vz[im]))
        return (pm - p0.lerp(p1, u)).length / max(pos_tol, 1e-9)

    def _rdp(start: int, end: int) -> List[int]:
        if end <= start + 1:
            return [start, end] if end != start else [start]
        worst = 0.0
        split = -1
        for i in range(start + 1, end):
            err = _seg_err(start, end, i)
            if err > worst:
                worst = err
                split = i
        if worst <= 1.0:
            return [start, end]
        left = _rdp(start, split)
        right = _rdp(split, end)
        return left[:-1] + right

    keep = _rdp(0, n - 1)
    return (
        [frame_list[i] for i in keep],
        [vx[i] for i in keep],
        [vy[i] for i in keep],
        [vz[i] for i in keep],
    )


def _smooth_lips_location_fcurves(action, lips_obj) -> None:
    if not action or lips_obj is None:
        return
    lad = lips_obj.animation_data
    if lad is None:
        return
    from ..bridge_blender import _iter_action_fcurves

    for fc in _iter_action_fcurves(action, anim_data=lad):
        if fc.data_path != "location":
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"
        try:
            fc.update()
        except Exception:
            pass


def _head_animation_frames(
    action, *, anim_data=None, max_samples: int = 96
) -> List[int]:
    """Keyframe times on Head_M / Neck_M only (same motion source as the clip)."""
    if action is None:
        return []
    from ..bridge_blender import _iter_action_fcurves

    names = {"Head_M", "Neck_M"}
    frames: set = set()
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        dp = fc.data_path
        if not dp.startswith('pose.bones["'):
            continue
        try:
            bone = dp.split('"')[1]
        except IndexError:
            continue
        if bone not in names:
            continue
        for kp in fc.keyframe_points:
            frames.add(int(round(kp.co[0])))
    if not frames:
        try:
            start, end = action.frame_range
            frames = set(range(int(start), int(end) + 1))
        except Exception:
            return []
    ordered = sorted(frames)
    if len(ordered) <= max_samples:
        return ordered
    step = max(1, len(ordered) // max_samples)
    return ordered[::step]


def _parent_world(eval_arm, pb_child) -> Matrix:
    parent = pb_child.parent
    if parent is None:
        return eval_arm.matrix_world.copy()
    return eval_arm.matrix_world @ parent.matrix


def _head_quat_in_parent(eval_arm, eval_head, eval_tgt) -> Quaternion:
    """Head orientation in IK_Head_Target parent space (Dummy_Root)."""
    parent_w = _parent_world(eval_arm, eval_tgt)
    head_w = eval_arm.matrix_world @ eval_head.matrix
    return (parent_w.inverted() @ head_w).to_quaternion().normalized()


def _ik_tip_in_parent(eval_arm, eval_ik, eval_tgt) -> Vector:
    parent_w = _parent_world(eval_arm, eval_tgt)
    tip_w = eval_arm.matrix_world @ Vector(eval_ik.tail)
    return parent_w.inverted() @ tip_w


def _simplify_ik_target_keys(
    frame_list: List[int],
    loc_x: List[float],
    loc_y: List[float],
    loc_z: List[float],
    qw: List[float],
    qx: List[float],
    qy: List[float],
    qz: List[float],
    *,
    pos_tol: float = 0.0012,
    rot_tol_deg: float = 0.4,
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Drop keys where linear pos + slerp rot between neighbors matches the clip."""
    n = len(frame_list)
    if n <= 2:
        return frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz
    rot_tol = math.radians(rot_tol_deg)

    def _q(i: int) -> Quaternion:
        return Quaternion((qw[i], qx[i], qy[i], qz[i])).normalized()

    def _segment_error(i0: int, i1: int, im: int) -> float:
        t0, t1, tm = float(frame_list[i0]), float(frame_list[i1]), float(frame_list[im])
        if t1 <= t0:
            return 0.0
        u = (tm - t0) / (t1 - t0)
        p0 = Vector((loc_x[i0], loc_y[i0], loc_z[i0]))
        p1 = Vector((loc_x[i1], loc_y[i1], loc_z[i1]))
        pm = Vector((loc_x[im], loc_y[im], loc_z[im]))
        pos_err = (pm - p0.lerp(p1, u)).length / max(pos_tol, 1e-9)
        q0, q1, qm = _q(i0), _q(i1), _q(im)
        q_interp = q0.slerp(q1, u)
        qm.make_compatible(q_interp)
        rot_err = qm.rotation_difference(q_interp).angle / max(rot_tol, 1e-9)
        return max(pos_err, rot_err)

    def _rdp(start: int, end: int) -> List[int]:
        if end <= start + 1:
            return [start, end] if end != start else [start]
        worst = 0.0
        split = -1
        for i in range(start + 1, end):
            err = _segment_error(start, end, i)
            if err > worst:
                worst = err
                split = i
        if worst <= 1.0:
            return [start, end]
        left = _rdp(start, split)
        right = _rdp(split, end)
        return left[:-1] + right

    keep = _rdp(0, n - 1)
    out_f: List[int] = []
    out_lx: List[float] = []
    out_ly: List[float] = []
    out_lz: List[float] = []
    out_qw: List[float] = []
    out_qx: List[float] = []
    out_qy: List[float] = []
    out_qz: List[float] = []
    for i in keep:
        out_f.append(frame_list[i])
        out_lx.append(loc_x[i])
        out_ly.append(loc_y[i])
        out_lz.append(loc_z[i])
        out_qw.append(qw[i])
        out_qx.append(qx[i])
        out_qy.append(qy[i])
        out_qz.append(qz[i])
    return out_f, out_lx, out_ly, out_lz, out_qw, out_qx, out_qy, out_qz


def _space_ik_target_keys(
    frame_list: List[int],
    loc_x: List[float],
    loc_y: List[float],
    loc_z: List[float],
    qw: List[float],
    qx: List[float],
    qy: List[float],
    qz: List[float],
    *,
    min_gap: int = 5,
):
    """Drop keys that sit on the next frame. First and last stay."""
    n = len(frame_list)
    if n <= 2:
        return frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz
    keep = [0]
    for i in range(1, n - 1):
        if int(frame_list[i]) - int(frame_list[keep[-1]]) >= min_gap:
            keep.append(i)
    if int(frame_list[-1]) - int(frame_list[keep[-1]]) >= min_gap:
        keep.append(n - 1)
    elif keep[-1] != n - 1:
        keep.append(n - 1)
    cols = (loc_x, loc_y, loc_z, qw, qx, qy, qz)
    picked = [[] for _ in cols]
    out_f = []
    for i in keep:
        out_f.append(frame_list[i])
        for dest, src in zip(picked, cols):
            dest.append(src[i])
    return (out_f, *picked)


_CHECK_BONES = ("Head_M", "Neck_M", "Chest_M", "Spine2_M", "Spine1_M")


def _ik_toggle_error(context, arm_obj, frame: int) -> Tuple[float, float]:
    """Max head/spine change (metres, degrees) when the IK tweaker goes 0 → 1."""
    context.scene.frame_set(int(frame))
    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
    context.view_layer.update()
    off = arm_obj.evaluated_get(context.evaluated_depsgraph_get())
    ref = {}
    for name in _CHECK_BONES:
        pb = off.pose.bones.get(name)
        if pb is None:
            continue
        mat = off.matrix_world @ pb.matrix
        ref[name] = (mat.translation.copy(), mat.to_quaternion())
    _apply_head_ik_sliders(arm_obj, 1.0, 0.0)
    context.view_layer.update()
    on = arm_obj.evaluated_get(context.evaluated_depsgraph_get())
    worst_m = 0.0
    worst_deg = 0.0
    for name, (pos, quat_ref) in ref.items():
        pb = on.pose.bones.get(name)
        if pb is None:
            continue
        mat = on.matrix_world @ pb.matrix
        got = mat.to_quaternion()
        got.make_compatible(quat_ref)
        worst_m = max(worst_m, (mat.translation - pos).length)
        worst_deg = max(worst_deg, math.degrees(got.rotation_difference(quat_ref).angle))
    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
    return worst_m, worst_deg


def _rows_at(
    frames: Sequence[int],
    src_f: List[int],
    loc_x: List[float],
    loc_y: List[float],
    loc_z: List[float],
    qw: List[float],
    qx: List[float],
    qy: List[float],
    qz: List[float],
):
    index = {int(src_f[i]): i for i in range(len(src_f))}
    out_f: List[int] = []
    out = [[] for _ in range(7)]
    cols = (loc_x, loc_y, loc_z, qw, qx, qy, qz)
    for frame in frames:
        i = index.get(int(frame))
        if i is None:
            continue
        out_f.append(int(frame))
        for col, src in zip(out, cols):
            col.append(src[i])
    return (out_f, *out)


def bake_ik_head_target_from_action(
    context,
    arm_obj,
    *,
    pos_tol: float = 0.0012,
    rot_tol_deg: float = 0.4,
    clip_match: bool = False,
    anchor_frame: Optional[int] = None,
    max_samples: int = 400,
    check_toggle_gaps: bool = True,
) -> int:
    """Key IK_Head_Target on the chain tip, with one rig rotation that keeps IK neutral.

    Head, neck, and spine stay on the clip when the IK tweaker goes from 0 to 1.
    ``clip_match=True`` is the slow solver — not used on import.
    """
    if bpy is None or context is None or arm_obj is None:
        return 0
    ad = arm_obj.animation_data
    if ad is None or ad.action is None:
        return 0
    pb_tgt = arm_obj.pose.bones.get(_TARGET)
    pb_head = arm_obj.pose.bones.get("Head_M")
    pb_ik = arm_obj.pose.bones.get(_IK_HEAD)
    if pb_tgt is None or pb_head is None or pb_ik is None:
        return 0

    from ..bridge_blender import (
        _assign_armature_action,
        _bulk_fcurve_channel,
        _bulk_object_channel,
        _ensure_action_fcurves,
        _smooth_pose_bone_fcurves,
    )

    action = ad.action
    _assign_armature_action(arm_obj, action)
    lips_obj = _find_lips_for_arm(arm_obj)
    if lips_obj is not None:
        _detach_lips_action(lips_obj)

    max_samples = max(32, int(max_samples or 400))
    frames = _head_animation_frames(action, anim_data=ad, max_samples=max_samples)
    if not frames:
        return 0

    scene = context.scene
    saved = scene.frame_current
    if anchor_frame is None:
        anchor_frame = max(int(getattr(scene, "frame_start", 1) or 1), 1)
    pb_tgt.rotation_mode = "QUATERNION"
    try:
        pb_tgt.lock_location = (False, False, False)
        pb_tgt.lock_rotation = (False, False, False)
    except Exception:
        pass

    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
    _mute_target_fcurves(arm_obj, True)
    held_drivers = []
    if ad.drivers:
        for drv in ad.drivers:
            if not drv.mute:
                drv.mute = True
                held_drivers.append(drv)
    neutral_rot = Quaternion((1.0, 0.0, 0.0, 0.0))
    if not clip_match:
        try:
            scene.frame_set(anchor_frame)
            context.view_layer.update()
            _pin_target_to_ik_tail(arm_obj)
            neutral_rot = _calibrate_ik_target_rotation(context, arm_obj)
        except Exception as exc:
            print(f"[GR2 LAB] Head IK calibration failed: {exc}")
            neutral_rot = Quaternion((1.0, 0.0, 0.0, 0.0))

    try:
        scene.frame_set(anchor_frame)
        context.view_layer.update()
        _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
        eval_arm = _evaluated_arm(context, arm_obj)
        e_ik = eval_arm.pose.bones.get(_IK_HEAD)
        e_head = eval_arm.pose.bones.get("Head_M")
        e_tgt = eval_arm.pose.bones.get(_TARGET)
        if e_ik is None or e_head is None or e_tgt is None:
            _mute_target_fcurves(arm_obj, False)
            for drv in held_drivers:
                drv.mute = False
            return 0
        bind_head_q = _head_quat_in_parent(eval_arm, e_head, e_tgt)
        bind_tip = _ik_tip_in_parent(eval_arm, e_ik, e_tgt)
        bind_loc = Vector(pb_tgt.location)
        bind_rot = Quaternion(pb_tgt.rotation_quaternion)
        tip_offset = bind_loc - bind_tip
        rot_delta = bind_rot @ bind_head_q.inverted()
    except Exception as exc:
        print(f"[GR2 LAB] Head IK target bake setup failed: {exc}")
        _mute_target_fcurves(arm_obj, False)
        for drv in held_drivers:
            drv.mute = False
        return 0

    frame_list: List[int] = []
    loc_x: List[float] = []
    loc_y: List[float] = []
    loc_z: List[float] = []
    qw: List[float] = []
    qx: List[float] = []
    qy: List[float] = []
    qz: List[float] = []
    lip_frames: List[int] = []
    lip_x: List[float] = []
    lip_y: List[float] = []
    lip_z: List[float] = []

    fk_locs = None
    if not clip_match:
        try:
            fk_locs = _fk_target_locations(
                arm_obj, action, frames, anim_data=ad
            )
        except Exception as exc:
            print(f"[GR2 LAB] Head IK F-curve pin failed, using frame_set: {exc}")
            fk_locs = None
        if fk_locs is not None:
            # A few depsgraph samples must match. A wrong chain falls back.
            for idx in {0, len(frames) // 2, len(frames) - 1}:
                scene.frame_set(int(frames[idx]))
                pinned = _pin_target_to_ik_tail(arm_obj)
                delta = None if pinned is None else (Vector(pinned) - fk_locs[idx]).length
                if delta is None or delta > 1.0e-4:
                    print(
                        f"[GR2 LAB] Head IK F-curve pin mismatch "
                        f"{0.0 if delta is None else delta:.6f}m at frame {frames[idx]}, "
                        "using frame_set",
                        flush=True,
                    )
                    fk_locs = None
                    break

    for frame_i, frame in enumerate(frames):
        try:
            if clip_match:
                eval_arm = _arm_at_clip_frame(context, arm_obj, frame)
                e_ik = eval_arm.pose.bones.get(_IK_HEAD)
                e_head = eval_arm.pose.bones.get("Head_M")
                e_tgt = eval_arm.pose.bones.get(_TARGET)
                if e_ik is None or e_head is None or e_tgt is None:
                    continue
                loc, rot = _heuristic_target_pose(
                    eval_arm, e_ik, e_head, e_tgt, rot_delta, tip_offset
                )
            elif fk_locs is not None:
                loc = fk_locs[frame_i]
                rot = Quaternion(neutral_rot)
            else:
                # Rotation is rig-constant (calibrated once above). Per frame we
                # only pin the target onto the animated chain tip — the old
                # per-frame neck search was a full depsgraph solve and dominated import.
                scene.frame_set(int(frame))
                if _pin_target_to_ik_tail(arm_obj) is None:
                    continue
                loc = Vector(pb_tgt.location)
                rot = Quaternion(neutral_rot)
            if clip_match:
                ref_m = _clip_head_reference(context, arm_obj, frame)
                loc, rot, _ = _refine_to_match_head(
                    context,
                    arm_obj,
                    pb_tgt,
                    ref_m,
                    loc,
                    rot,
                    ik=1.0,
                    kiss=0.0,
                )
            frame_list.append(int(frame))
            loc_x.append(float(loc.x))
            loc_y.append(float(loc.y))
            loc_z.append(float(loc.z))
            if qw:
                prev = Quaternion((qw[-1], qx[-1], qy[-1], qz[-1]))
                if rot.rotation_difference(prev).angle < math.radians(0.7):
                    rot = prev
            qw.append(float(rot.w))
            qx.append(float(rot.x))
            qy.append(float(rot.y))
            qz.append(float(rot.z))

            if clip_match and lips_obj is not None:
                mouth_w = _mouth_world(eval_arm)
                lips_m = _kiss_world_matrix(eval_arm, location=mouth_w)
                loc, rot, lips_m = _refine_to_match_head(
                    context,
                    arm_obj,
                    pb_tgt,
                    ref_m,
                    loc,
                    rot,
                    ik=1.0,
                    kiss=1.0,
                    lips_obj=lips_obj,
                    lips_world=lips_m,
                )
                lips_obj.matrix_world = lips_m
                context.view_layer.update()
                lip_frames.append(int(frame))
                lip_x.append(float(lips_obj.location.x))
                lip_y.append(float(lips_obj.location.y))
                lip_z.append(float(lips_obj.location.z))
        except Exception:
            continue

    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)

    sampled = len(frame_list)
    keyed = 0
    src_f = list(frame_list)
    src_lx, src_ly, src_lz = list(loc_x), list(loc_y), list(loc_z)
    src_qw, src_qx, src_qy, src_qz = list(qw), list(qx), list(qy), list(qz)
    if sampled and not clip_match:
        frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz = _simplify_ik_target_keys(
            frame_list,
            loc_x,
            loc_y,
            loc_z,
            qw,
            qx,
            qy,
            qz,
            pos_tol=0.0045,
            rot_tol_deg=1.1,
        )
        frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz = _space_ik_target_keys(
            frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz, min_gap=5
        )
    elif sampled:
        frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz = _simplify_ik_target_keys(
            frame_list,
            loc_x,
            loc_y,
            loc_z,
            qw,
            qx,
            qy,
            qz,
            pos_tol=0.00035,
            rot_tol_deg=0.15,
        )
    if sampled:
        fcurves = _ensure_action_fcurves(arm_obj, action)
        _bulk_fcurve_channel(
            fcurves,
            _TARGET,
            "location",
            frame_list,
            [loc_x, loc_y, loc_z],
        )
        _bulk_fcurve_channel(
            fcurves,
            _TARGET,
            "rotation_quaternion",
            frame_list,
            [qw, qx, qy, qz],
        )
        _smooth_pose_bone_fcurves(action, _TARGET, anim_data=ad)
        _mute_target_fcurves(arm_obj, False)
        for drv in held_drivers:
            drv.mute = False
        keyed = len(frame_list)
        if check_toggle_gaps and not clip_match and len(src_f) > len(frame_list):
            # Only split a wide gap, and only when the IK tweaker would visibly
            # move the head or spine. Never drop a key one frame away from another.
            pos_lim = 0.0035
            rot_lim = 1.4
            min_gap = 6
            for _ in range(3):
                kept = set(frame_list)
                added = False
                ordered = list(frame_list)
                for left, right in zip(ordered, ordered[1:]):
                    if right - left < min_gap * 2:
                        continue
                    mid = (left + right) // 2
                    nearest = min(src_f, key=lambda fr: abs(fr - mid))
                    if nearest - left < min_gap or right - nearest < min_gap:
                        continue
                    if nearest in kept:
                        continue
                    mm, deg = _ik_toggle_error(context, arm_obj, nearest)
                    if max(mm / pos_lim, deg / rot_lim) <= 1.0:
                        continue
                    kept.add(nearest)
                    added = True
                if not added:
                    break
                frame_list, loc_x, loc_y, loc_z, qw, qx, qy, qz = _rows_at(
                    sorted(kept),
                    src_f,
                    src_lx,
                    src_ly,
                    src_lz,
                    src_qw,
                    src_qx,
                    src_qy,
                    src_qz,
                )
                fcurves = _ensure_action_fcurves(arm_obj, action)
                _bulk_fcurve_channel(
                    fcurves, _TARGET, "location", frame_list, [loc_x, loc_y, loc_z]
                )
                _bulk_fcurve_channel(
                    fcurves,
                    _TARGET,
                    "rotation_quaternion",
                    frame_list,
                    [qw, qx, qy, qz],
                )
                _smooth_pose_bone_fcurves(action, _TARGET, anim_data=ad)
            keyed = len(frame_list)
    else:
        _mute_target_fcurves(arm_obj, False)
        for drv in held_drivers:
            drv.mute = False

    if clip_match and lip_frames and lips_obj is not None:
        lip_frames, lip_x, lip_y, lip_z = _simplify_vec3_keys(
            lip_frames, lip_x, lip_y, lip_z, pos_tol=pos_tol
        )
        if lips_obj.animation_data is None:
            lips_obj.animation_data_create()
        lip_action = bpy.data.actions.new(f"{action.name}_Lips")
        lips_obj.animation_data.action = lip_action
        _assign_armature_action(lips_obj, lip_action)
        lip_fcurves = _ensure_action_fcurves(lips_obj, lip_action)
        _bulk_object_channel(lip_fcurves, "location", lip_frames, [lip_x, lip_y, lip_z])
        _smooth_lips_location_fcurves(lip_action, lips_obj)

    scene.frame_set(saved)
    _apply_head_ik_sliders(arm_obj, 0.0, 0.0)
    context.view_layer.update()
    if keyed:
        tag = "clip-match" if clip_match else "tip+neutral IK"
        note = "IK toggle–safe" if check_toggle_gaps or clip_match else "import samples"
        print(
            f"[GR2 LAB] Head IK bake ({tag}): {_TARGET} {keyed} keys"
            f" (from {sampled} samples, {note})"
        )
    return keyed


_SPINE_FOLLOW = (
    ("Chest_M", "Extra * 0.650"),
    ("Spine2_M", "Extra * 0.350"),
    ("Spine1_M", "Extra * 0.150"),
)


def _setup_spine_follow(arm_obj) -> None:
    """DGB Extra/Spine: copy IK_Head_Target rotation down Chest/Spine2/Spine1.

    Karlach pointed these at a missing bone named ``0``; we use the chevron.
    """
    for bone_name, expr in _SPINE_FOLLOW:
        pb = arm_obj.pose.bones.get(bone_name)
        if pb is None:
            continue
        ccr = _ensure_constraint(pb, "COPY_ROTATION", "Copy Rotation")
        ccr.target = arm_obj
        ccr.subtarget = _TARGET
        ccr.owner_space = "LOCAL"
        ccr.target_space = "LOCAL_OWNER_ORIENT"
        ccr.use_x = ccr.use_y = ccr.use_z = True
        ccr.invert_x = ccr.invert_y = ccr.invert_z = False
        try:
            ccr.mix_mode = "AFTER"
        except Exception:
            pass
        if hasattr(ccr, "use_offset"):
            ccr.use_offset = True
        ccr.influence = 0.0
        _set_influence_driver(
            arm_obj,
            f'pose.bones["{bone_name}"].constraints["Copy Rotation"].influence',
            var_name="Extra",
            expression=expr,
            scripted=True,
        )


def _assign_collection(arm_obj, bone_name: str) -> None:
    data = arm_obj.data
    colls = getattr(data, "collections", None)
    if colls is None:
        return
    dest_name = _COLL.get(bone_name)
    if not dest_name:
        return
    dest = colls.get(dest_name) if hasattr(colls, "get") else None
    if dest is None:
        idx = colls.find(dest_name) if hasattr(colls, "find") else -1
        dest = colls[idx] if idx >= 0 else None
    if dest is None:
        dest = colls.new(dest_name)
    bone = data.bones.get(bone_name)
    if bone is None:
        return
    try:
        dest.assign(bone)
    except Exception:
        pass
    for col in list(bone.collections):
        if col != dest:
            try:
                col.unassign(bone)
            except Exception:
                pass


def _create_ui_bones(eb, head_m, dummy) -> None:
    """2D HUD parented to Dummy_Root — left of the head, facing the character."""
    up = Vector((0.0, 1.0, 0.0))
    z_roll = Vector((0.0, 0.0, 1.0))
    center = Vector(head_m.head) + UI_HEAD_FROM_HEAD
    origin = center - ui_panel_center()
    _ensure_edit_bone(
        eb,
        _UI_ROOT,
        parent=dummy,
        head=center,
        tail=center + up * 0.002,
        use_connect=False,
        roll=z_roll,
    )
    ui_root = eb[_UI_ROOT]
    try:
        ui_root.use_inherit_rotation = False
    except Exception:
        pass
    field_x = ui_knob_rest_x()
    for i, knob in enumerate((_UI_IK, _UI_KISS, _UI_SPINE)):
        h = origin + Vector((field_x, ui_row_y(i), 0.0))
        _ensure_edit_bone(
            eb,
            knob,
            parent=ui_root,
            head=h,
            tail=h + Vector((0.002, 0.0, 0.0)),
            use_connect=False,
            roll=z_roll,
        )
        track = eb.get(f"{knob}_Track")
        if track is not None:
            eb.remove(track)
        child = eb.get(knob)
        if child is not None:
            try:
                child.inherit_scale = "FULL"
            except Exception:
                pass
    _bake_ui_head_pose_into_rest(eb)


def _bake_ui_head_pose_into_rest(eb) -> None:
    """Yaw 90° + scale 3.23 on UI_Head rest. Knobs stay in the panel plane.

    Parent roll alone does not update child roll. Leaving the old +Z roll after
    yaw stands UI_Head_IK / Kiss / Spine on edge (90° off the card).
    """
    root = eb.get(_UI_ROOT)
    if root is None:
        return
    scl = float(UI_HEAD_POSE_SCALE)
    try:
        inv = root.matrix.inverted()
    except ValueError:
        inv = root.matrix.inverted_safe()
    stored: List[Tuple[str, Vector]] = []
    for name in (_UI_IK, _UI_SPINE, _UI_KISS):
        bone = eb.get(name)
        if bone is None:
            continue
        stored.append((name, inv @ Vector(bone.head)))
    root.tail = Vector(root.head) + (Vector(root.tail) - Vector(root.head)) * scl
    root.roll += UI_HEAD_POSE_YAW
    pmat = root.matrix.copy()
    axis_x = Vector(root.x_axis)
    axis_z = Vector(root.z_axis)
    knob_len = 0.002 * scl
    for name, loc_h in stored:
        bone = eb[name]
        bone.use_connect = False
        bone.head = pmat @ (Vector(loc_h) * scl)
        bone.tail = Vector(bone.head) + axis_x * knob_len
        try:
            bone.align_roll(axis_z)
        except Exception:
            pass


def setup_head_ik_on_armature(
    arm_obj,
    *,
    context=None,
    mode_set: Optional[ModeSet] = None,
    lookat_on: bool = False,
) -> Tuple[List[str], List[str]]:
    """Create helper bones, 3D UI sliders, constraints, and drivers.

    Look-at starts off (``UI_Head_IK`` at 0) so bind/clip pose stays put
    until the user slides the HUD.
    """
    if not arm_obj or arm_obj.type != "ARMATURE":
        return [], ["No armature"]
    data = arm_obj.data
    if data.bones.get("Head_M") is None or data.bones.get("Neck_M") is None:
        return [], ["Head_M/Neck_M missing — skip head IK"]
    if data.bones.get("Dummy_Root") is None:
        return [], ["Dummy_Root missing — skip head IK"]

    ctx = context or bpy.context
    errors: List[str] = []
    ok: List[str] = []
    try:
        _set_mode(ctx, "EDIT", arm_obj, mode_set)
        eb = data.edit_bones
        head_m = eb["Head_M"]
        neck = eb["Neck_M"]
        dummy = eb["Dummy_Root"]
        # DGB Z-up layout mapped into this Y-up armature: forward = local -Z, up = +Y.
        forward = Vector((0.0, 0.0, -1.0))
        up = Vector((0.0, 1.0, 0.0))
        origin = Vector(head_m.head)
        ik_head = origin + forward * 0.007 - up * 0.013
        ik_tail = ik_head + forward * 0.221
        tip_tail = ik_tail + up * 0.083
        tgt_tail = ik_tail + forward * 0.092
        _ensure_edit_bone(
            eb,
            _IK_HEAD,
            parent=neck,
            head=ik_head,
            tail=ik_tail,
            use_connect=False,
            roll=head_m.z_axis,
        )
        _ensure_edit_bone(
            eb,
            _IK_HEAD_TIP,
            parent=eb[_IK_HEAD],
            head=ik_tail,
            tail=tip_tail,
            use_connect=True,
            roll=head_m.z_axis,
        )
        _ensure_edit_bone(
            eb, _TARGET, parent=dummy, head=ik_tail, tail=tgt_tail, use_connect=False
        )
        _create_ui_bones(eb, head_m, dummy)
        ok.append("helper bones")
    except Exception as exc:
        errors.append(str(exc))
        return ok, errors
    finally:
        try:
            _set_mode(ctx, "POSE", arm_obj, mode_set)
        except Exception:
            pass

    for name in HELPER_BONES:
        db = data.bones.get(name)
        if db is None:
            continue
        db.use_deform = False
        db.hide = name not in _VISIBLE
        try:
            # Extra (IK_Head) stays selectable, same as DGB. Only HUD tracks lock.
            db.hide_select = name.endswith("_Track")
        except Exception:
            pass
        if name in (_UI_ROOT, _UI_IK, _UI_SPINE, _UI_KISS, _TARGET):
            try:
                db.show_wire = False
            except Exception:
                pass
        _assign_collection(arm_obj, name)

    pb_ik = arm_obj.pose.bones.get(_IK_HEAD)
    pb_tgt = arm_obj.pose.bones.get(_TARGET)
    pb_head = arm_obj.pose.bones.get("Head_M")
    pb_neck = arm_obj.pose.bones.get("Neck_M")
    if not (pb_ik and pb_tgt and pb_head and pb_neck):
        errors.append("pose bones missing after create")
        return ok, errors

    ik_on = 1.0 if lookat_on else 0.0
    _float_slider(pb_tgt, "IK", ik_on, "Head look-at IK (0 = animation, 1 = follow target)")
    _float_slider(pb_tgt, "Extra", 0.0, "Chest mix toward the head IK target")
    _float_slider(pb_tgt, "!Kiss", 0.0, "World copy from Lips mesh object")
    if "Spine" in pb_tgt:
        try:
            del pb_tgt["Spine"]
        except Exception:
            pass

    for knob in (_UI_IK, _UI_SPINE, _UI_KISS):
        pb_ui = arm_obj.pose.bones.get(knob)
        if pb_ui is not None:
            pb_ui.location[0] = 0.0
            pb_ui.location[1] = ui_travel_pose() if knob == _UI_IK and lookat_on else 0.0
            pb_ui.location[2] = 0.0
            _lock_slider(pb_ui)
    pb_panel = arm_obj.pose.bones.get(_UI_ROOT)
    if pb_panel is not None:
        try:
            pb_panel.rotation_mode = "XYZ"
        except Exception:
            pass
        pb_panel.location = Vector((0.0, 0.0, 0.0))
        pb_panel.rotation_euler = Euler((0.0, 0.0, 0.0))
        pb_panel.scale = Vector((1.0, 1.0, 1.0))
        pb_panel.lock_location = (False, False, False)
        pb_panel.lock_rotation = (False, False, False)
        pb_panel.lock_scale = (False, False, False)
        try:
            pb_panel.lock_rotation_w = False
        except Exception:
            pass
    _remove_driver(arm_obj, f'pose.bones["{_TARGET}"]["Spine"]')
    _drive_from_ui_bone(arm_obj, f'pose.bones["{_TARGET}"]["IK"]', _UI_IK)
    _drive_from_ui_bone(arm_obj, f'pose.bones["{_TARGET}"]["Extra"]', _UI_SPINE)
    _drive_from_ui_bone(arm_obj, f'pose.bones["{_TARGET}"]["!Kiss"]', _UI_KISS)

    for pb in (pb_ik, pb_head, pb_neck):
        pb.ik_rotation_weight = 1.0
        pb.ik_linear_weight = 1.0
    pb_chest = arm_obj.pose.bones.get("Chest_M")
    if pb_chest is not None:
        pb_chest.ik_rotation_weight = 1.0
        pb_chest.ik_linear_weight = 1.0
    pb_sp2 = arm_obj.pose.bones.get("Spine2_M")
    if pb_sp2 is not None:
        pb_sp2.ik_rotation_weight = 1.0
        pb_sp2.ik_linear_weight = 1.0

    ik_con = _ensure_constraint(pb_ik, "IK", "IK")
    ik_con.target = arm_obj
    ik_con.subtarget = _TARGET
    ik_con.chain_count = 5
    ik_con.use_stretch = True
    # Position only, like DGB. use_rotation copies IK_Head_Target's Dummy-root
    # axes onto IK_Head; Head_M then Copy-Rotates that and the skull snaps.
    ik_con.use_rotation = False
    ik_con.use_tail = True
    ik_con.influence = 0.0
    _set_influence_driver(
        arm_obj,
        f'pose.bones["{_IK_HEAD}"].constraints["IK"].influence',
        var_name="IK",
        expression="IK",
        scripted=False,
    )

    cr = _ensure_constraint(pb_head, "COPY_ROTATION", "Copy Rotation")
    cr.target = arm_obj
    cr.subtarget = _IK_HEAD
    cr.owner_space = "LOCAL"
    cr.target_space = "LOCAL_OWNER_ORIENT"
    cr.use_x = cr.use_y = cr.use_z = True
    cr.invert_x = cr.invert_y = cr.invert_z = False
    try:
        cr.mix_mode = "ADD"
    except Exception:
        pass
    if hasattr(cr, "use_offset"):
        cr.use_offset = True
    cr.influence = 0.0
    _set_head_follow_ik_driver(arm_obj)

    cr2 = _ensure_constraint(pb_head, "COPY_ROTATION", "Copy Rotation.001")
    cr2.target = arm_obj
    cr2.subtarget = _IK_HEAD
    cr2.owner_space = "WORLD"
    cr2.target_space = "WORLD"
    cr2.use_x = cr2.use_y = cr2.use_z = True
    cr2.invert_x = cr2.invert_y = cr2.invert_z = False
    try:
        cr2.mix_mode = "REPLACE"
    except Exception:
        pass
    if hasattr(cr2, "use_offset"):
        cr2.use_offset = False
    cr2.influence = 0.0
    _remove_driver(
        arm_obj,
        f'pose.bones["Head_M"].constraints["Copy Rotation.001"].influence',
    )

    ncr = _ensure_constraint(pb_neck, "COPY_ROTATION", "Copy Rotation")
    ncr.target = arm_obj
    ncr.subtarget = _TARGET
    ncr.owner_space = "LOCAL_WITH_PARENT"
    ncr.target_space = "LOCAL_OWNER_ORIENT"
    ncr.use_x = ncr.use_y = ncr.use_z = True
    ncr.invert_x = ncr.invert_y = ncr.invert_z = False
    try:
        ncr.mix_mode = "REPLACE"
    except Exception:
        pass
    if hasattr(ncr, "use_offset"):
        ncr.use_offset = False
    ncr.influence = 0.0
    _set_influence_driver(
        arm_obj,
        f'pose.bones["Neck_M"].constraints["Copy Rotation"].influence',
        var_name="IK",
        expression="IK/2",
        scripted=True,
    )

    _setup_spine_follow(arm_obj)

    try:
        lips = ensure_kiss_object(arm_obj, ctx)
    except Exception as exc:
        errors.append(f"Lips: {exc}")
        lips = None
    cloc = _ensure_constraint(pb_tgt, "COPY_LOCATION", "Copy Location")
    cloc.target = lips
    cloc.subtarget = ""
    cloc.owner_space = "WORLD"
    cloc.target_space = "WORLD"
    cloc.use_x = cloc.use_y = cloc.use_z = True
    if hasattr(cloc, "use_offset"):
        cloc.use_offset = False
    cloc.influence = 0.0
    _set_influence_driver(
        arm_obj,
        f'pose.bones["{_TARGET}"].constraints["Copy Location"].influence',
        var_name="_Influence",
        expression="_Influence",
        scripted=False,
    )

    crot = _ensure_constraint(pb_tgt, "COPY_ROTATION", "Copy Rotation")
    crot.target = lips
    crot.subtarget = ""
    crot.owner_space = "WORLD"
    crot.target_space = "WORLD"
    try:
        crot.mix_mode = "AFTER"
    except Exception:
        pass
    if hasattr(crot, "use_offset"):
        crot.use_offset = True
    crot.influence = 0.0
    _set_influence_driver(
        arm_obj,
        f'pose.bones["{_TARGET}"].constraints["Copy Rotation"].influence',
        var_name="_Influence",
        expression="_Influence",
        scripted=False,
    )

    ok.append("constraints+drivers")
    print("[GR2 LAB] Head IK: floating HUD UI_Head_IK / Extra / Kiss → IK_Head_Target")
    return ok, errors
