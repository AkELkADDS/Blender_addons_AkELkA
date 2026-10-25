"""Optional sibling tail armature (Tiefling / Dragonborn Tail_A).

Basic GR2-style rig — same bind convention as Body_Base (Dummy_Root identity,
object Rx+90). Copy Transforms from the body onto shared pelvis/spine bones.
No Dummy_Tail_IK. Does not bake, densify, or encode.
"""
from __future__ import annotations

from typing import Optional

import bpy

_FOLLOW_BONES = (
    "Dummy_Root",
    "Root_M",
    "Spine1_M",
    "Spine2_M",
    "Chest_M",
    "Hip_L_Twist_01",
    "Hip_R_Twist_01",
    "Hip_L",
    "Hip_R",
)


def _body_prefix(body_id: str) -> str:
    from bind_pose import bind_pose_stem  # type: ignore

    stem = bind_pose_stem(body_id)
    low = stem.lower()
    if low.endswith("_tail_base"):
        return stem[: -len("_Tail_Base")]
    if low.endswith("_base"):
        return stem[:-5]
    return stem


def _iter_tail_rigs(body_arm) -> list:
    name = getattr(body_arm, "name", "") or ""
    if not name:
        return []
    return [
        obj
        for obj in bpy.data.objects
        if obj.get("gr2lab_tail_of") == name and getattr(obj, "type", None) == "ARMATURE"
    ]


def remove_tail_rigs_of(body_arm) -> int:
    """Delete sibling tail armatures (and their mesh children) for ``body_arm``."""
    doomed = _iter_tail_rigs(body_arm)
    n = 0
    for arm in doomed:
        for child in list(arm.children):
            try:
                bpy.data.objects.remove(child, do_unlink=True)
            except Exception:
                pass
        try:
            bpy.data.objects.remove(arm, do_unlink=True)
        except Exception:
            pass
        n += 1
    return n


def sync_tail_world(body_arm) -> None:
    """Keep sibling tail armatures on the body's world matrix."""
    if body_arm is None:
        return
    mw = body_arm.matrix_world.copy()
    for arm in _iter_tail_rigs(body_arm):
        arm.matrix_world = mw.copy()


def _absorb_into_collections(obj, dests) -> None:
    dests = [c for c in dests if c is not None]
    if not dests:
        return
    for dest in dests:
        try:
            dest.objects.link(obj)
        except RuntimeError:
            pass
    for coll in list(obj.users_collection):
        if coll in dests:
            continue
        try:
            coll.objects.unlink(obj)
        except RuntimeError:
            pass


def _copy_transforms(tail_arm, body_arm) -> int:
    n = 0
    body_bones = set(body_arm.data.bones.keys())
    for name in _FOLLOW_BONES:
        if name not in body_bones:
            continue
        pb = tail_arm.pose.bones.get(name)
        if pb is None:
            continue
        con = None
        for existing in pb.constraints:
            if existing.type == "COPY_TRANSFORMS" and existing.name == "GR2LAB_TailFollow":
                con = existing
                break
        if con is None:
            con = pb.constraints.new("COPY_TRANSFORMS")
            con.name = "GR2LAB_TailFollow"
        con.target = body_arm
        con.subtarget = name
        con.target_space = "WORLD"
        con.owner_space = "WORLD"
        if hasattr(con, "mix_mode"):
            con.mix_mode = "REPLACE"
        con.influence = 1.0
        n += 1
    return n


def _coll_get(colls, name: str):
    getter = getattr(colls, "get", None)
    if callable(getter):
        return getter(name)
    idx = colls.find(name)
    return colls[idx] if idx >= 0 else None


def _force_collection(arm_obj, bone_name: str, dest_name: str) -> None:
    data = arm_obj.data
    dest = _coll_get(data.collections, dest_name)
    if dest is None:
        dest = data.collections.new(dest_name)
        try:
            dest.is_visible = dest_name not in {"FX", "Extra"}
        except Exception:
            pass
    bone = data.bones.get(bone_name)
    if bone is None:
        return
    try:
        dest.assign(bone)
    except Exception:
        return
    for col in list(bone.collections):
        if col != dest:
            try:
                col.unassign(bone)
            except Exception:
                pass


def _apply_tail_collections(tail_arm) -> None:
    from . import _apply_collections

    _apply_collections(tail_arm)
    extra = _coll_get(tail_arm.data.collections, "Extra")
    if extra is not None:
        try:
            extra.is_visible = False
        except Exception:
            pass
    for bone in tail_arm.data.bones:
        if bone.name.startswith("Tail"):
            _force_collection(tail_arm, bone.name, "Tail")
        else:
            _force_collection(tail_arm, bone.name, "Extra")
    keep = {"Tail", "Extra"}
    for col in list(tail_arm.data.collections):
        if col.name in keep:
            continue
        try:
            if len(col.bones) == 0:
                tail_arm.data.collections.remove(col)
        except Exception:
            pass


def _apply_tail_display(context, tail_arm, *, shapes: bool) -> None:
    from . import (
        _WIDGET_COLL,
        _apply_armature_display,
        _apply_bone_tone,
        _assign_pose_shape,
        _ensure_generated_widget,
        _ensure_scene_collection,
        _mesh_hoop,
        _mesh_plain_axes,
        load_snapshot,
    )

    snap = load_snapshot()
    _apply_armature_display(tail_arm, snap)
    _apply_tail_collections(tail_arm)

    if shapes:
        widget_coll = _ensure_scene_collection(context, _WIDGET_COLL)
        widgets = {
            "hoop": _ensure_generated_widget("hoop", widget_coll, *_mesh_hoop()),
            "axes": _ensure_generated_widget(
                "axes", widget_coll, *_mesh_plain_axes()
            ),
        }
        for pb in tail_arm.pose.bones:
            if not pb.name.startswith("Tail"):
                continue
            _assign_pose_shape(pb, {}, widgets, arm_data=tail_arm.data)
        try:
            tail_arm.data.display_type = "STICK"
        except Exception:
            pass

    for pb in tail_arm.pose.bones:
        _apply_bone_tone(pb)


def attach_tail_rig(
    context,
    body_arm,
    base_id: Optional[str] = None,
    *,
    shapes: bool = True,
    ik: bool = False,
    y_up_display: bool = True,
) -> Optional[bpy.types.Object]:
    """Build ``GR2LAB_{prefix}_Tail`` beside ``body_arm``. Skip races with no sidecar.

    ``ik`` is ignored — tail bases stay a plain FK chain like Body_Base.
    """
    if body_arm is None or getattr(body_arm, "type", None) != "ARMATURE":
        return None
    if body_arm.get("gr2lab_tail"):
        return None
    from bind_pose import (  # type: ignore
        bind_pose_sidecar_path,
        gr2lab_doc_from_bind,
        load_bind_pose,
        tail_base_id_for_body,
    )

    ident = tail_base_id_for_body(base_id or "")
    if not ident:
        return None
    try:
        path = bind_pose_sidecar_path(ident)
        bind = load_bind_pose(path)
    except FileNotFoundError:
        return None

    remove_tail_rigs_of(body_arm)

    from ..bridge_blender import import_gr2lab_from_doc

    prefix = _body_prefix(base_id or ident)
    doc = gr2lab_doc_from_bind(bind, run_id=ident)
    tail_arm = import_gr2lab_from_doc(
        context,
        doc,
        y_up_display=bool(y_up_display),
        animation=False,
        rig_display=False,
        import_meshes="NONE",
        import_akelka=False,
        import_tail=False,
        mesh_base=ident,
    )
    if tail_arm is None:
        return None

    wanted = f"GR2LAB_{prefix}_Tail"
    try:
        tail_arm.name = wanted
        tail_arm.data.name = wanted
    except Exception:
        pass
    try:
        tail_arm["gr2lab_tail"] = 1
        tail_arm["gr2lab_tail_of"] = body_arm.name
    except Exception:
        pass
    tail_arm.parent = None
    tail_arm.matrix_world = body_arm.matrix_world.copy()
    _absorb_into_collections(tail_arm, list(body_arm.users_collection))

    _copy_transforms(tail_arm, body_arm)
    if shapes:
        _apply_tail_display(context, tail_arm, shapes=True)
    else:
        _apply_tail_collections(tail_arm)

    try:
        from .mesh_kits import retarget_tail_meshes

        n_mesh = retarget_tail_meshes(
            context, body_arm, tail_arm, base_id=base_id or ident
        )
    except Exception:
        n_mesh = 0
        import traceback

        traceback.print_exc()
        print("[GR2 LAB] tail mesh retarget failed (armature is intact)")

    extra = f" · meshes={n_mesh}" if n_mesh else ""
    print(f"[GR2 LAB] tail rig {tail_arm.name!r} on {body_arm.name!r}{extra}")
    return tail_arm
