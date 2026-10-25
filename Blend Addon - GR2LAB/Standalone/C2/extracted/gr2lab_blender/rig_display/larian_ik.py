"""Four Larian IK constraints + ankle/wrist length scale (Animation Fixer 16).

Copied into GR2 LAB so import does not depend on the Fixer addon.
Does not include quat-fix, gaussian smooth, or auto-bezier.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import bpy

_LARIAN_IK_LIMBS = (
    ("Left foot", "leg", ("Ankle_L",), ("Dummy_L_Foot_IK",)),
    ("Right foot", "leg", ("Ankle_R",), ("Dummy_R_Foot_IK",)),
    ("Left hand", "hand", ("Wrist_L",), ("Dummy_L_Hand_IK",)),
    ("Right hand", "hand", ("Wrist_R",), ("Dummy_R_Hand_IK",)),
)

_IK_BONE_LENGTH_SCALE = 0.005
_IK_BONE_ALREADY_SMALL_MAX = 0.02

ModeSet = Callable[..., None]


def resolve_bone_on_armature(arm_obj, *candidates: str) -> str:
    """Return first bone name that exists on arm_obj, else first candidate."""
    if not candidates:
        return ""
    if arm_obj and getattr(arm_obj, "type", None) == "ARMATURE":
        for name in candidates:
            if name and arm_obj.pose.bones.get(name):
                return name
    return candidates[0]


def resolve_larian_ik_limbs(arm_obj) -> List[dict]:
    """Resolve the 4 Larian end-effector IK pairs (bone, IK target, limb kind)."""
    limbs = []
    for label, kind, bone_cands, target_cands in _LARIAN_IK_LIMBS:
        limbs.append(
            {
                "label": label,
                "kind": kind,
                "bone": resolve_bone_on_armature(arm_obj, *bone_cands),
                "target": resolve_bone_on_armature(arm_obj, *target_cands),
            }
        )
    return limbs


def ensure_ik_constraint(
    pb_parent,
    arm_obj,
    ik_target_name: str,
    chain_count: int,
    constraint_name: str,
    use_rotation: bool = True,
):
    """Add or update one IK constraint on a pose bone."""
    con = None
    for c in pb_parent.constraints:
        if c.type == "IK" and c.name == constraint_name:
            con = c
            break
    if con is None:
        con = pb_parent.constraints.new("IK")
        con.name = constraint_name
    con.target = arm_obj
    con.subtarget = ik_target_name
    con.chain_count = max(0, int(chain_count))
    con.influence = 1.0
    con.use_stretch = False
    con.use_rotation = bool(use_rotation)
    return con


def remove_aaf_ik_constraints(arm_obj) -> None:
    """Remove prior AAF_IK_* constraints so re-run moves them to the correct bones."""
    for pb in arm_obj.pose.bones:
        for con in list(pb.constraints):
            if con.type == "IK" and con.name.startswith("AAF_IK_"):
                pb.constraints.remove(con)


def setup_larian_ik_constraints_on_armature(
    arm_obj,
    leg_chain_count: int = 3,
    hand_chain_count: int = 3,
    use_rotation: bool = True,
    *,
    hands: bool = True,
    legs: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Create/update 4 IK constraints on Ankle_L/R and Wrist_L/R targeting Dummy_*_IK bones.
    Skip a limb if the effector or Dummy IK target is missing.
    Returns (created_or_updated_names, error_messages).
    """
    if not arm_obj or arm_obj.type != "ARMATURE":
        return [], ["No armature"]
    if not hands and not legs:
        return [], []
    remove_aaf_ik_constraints(arm_obj)
    ok: List[str] = []
    errors: List[str] = []
    for spec in resolve_larian_ik_limbs(arm_obj):
        if spec["kind"] == "hand" and not hands:
            continue
        if spec["kind"] == "leg" and not legs:
            continue
        label = spec["label"]
        bone = spec["bone"]
        target = spec["target"]
        chain_count = hand_chain_count if spec["kind"] == "hand" else leg_chain_count
        if not bone or not arm_obj.pose.bones.get(bone):
            errors.append(f"{label}: missing bone {bone or '?'}")
            continue
        if not target or not arm_obj.pose.bones.get(target):
            errors.append(f"{label}: missing IK target {target or '?'}")
            continue
        pb = arm_obj.pose.bones.get(bone)
        con_name = f"AAF_IK_{target}"
        ensure_ik_constraint(
            pb, arm_obj, target, chain_count, con_name, use_rotation=use_rotation
        )
        ok.append(f"{label} ({bone}→{target}, chain={chain_count})")
        rot_s = "rot=ON" if use_rotation else "rot=OFF"
        print(
            f"[GR2 LAB] IK {label}: {bone} -> {target} "
            f"chain_count={chain_count} {rot_s}"
        )
    return ok, errors


def ik_bones_already_small(arm_obj) -> bool:
    """True when ankle/wrist bones are already at scaled (×0.005) length."""
    if not arm_obj or arm_obj.type != "ARMATURE" or not arm_obj.data:
        return False
    checked = 0
    for spec in resolve_larian_ik_limbs(arm_obj):
        name = spec["bone"]
        bone = arm_obj.data.bones.get(name) if name else None
        if not bone:
            continue
        checked += 1
        if bone.length > _IK_BONE_ALREADY_SMALL_MAX:
            return False
    return checked > 0


def scale_larian_ik_bone_lengths(
    arm_obj,
    factor: float = _IK_BONE_LENGTH_SCALE,
    *,
    context=None,
    mode_set: Optional[ModeSet] = None,
    hands: bool = True,
    legs: bool = True,
) -> Tuple[List[str], List[str]]:
    """Scale Ankle_L/R and Wrist_L/R edit-bone lengths by factor (default 0.005)."""
    if not arm_obj or arm_obj.type != "ARMATURE":
        return [], ["No armature"]
    if not hands and not legs:
        return [], []
    if ik_bones_already_small(arm_obj):
        return [], []
    bone_names: List[str] = []
    for spec in resolve_larian_ik_limbs(arm_obj):
        if spec["kind"] == "hand" and not hands:
            continue
        if spec["kind"] == "leg" and not legs:
            continue
        name = spec["bone"]
        if name and arm_obj.data.bones.get(name):
            bone_names.append(name)
    if not bone_names:
        return [], ["No IK bones found (Ankle_L/R, Wrist_L/R)"]

    ctx = context or bpy.context
    errors: List[str] = []
    ok: List[str] = []
    try:
        if mode_set is not None:
            mode_set(ctx, "EDIT", active=arm_obj)
        else:
            ctx.view_layer.objects.active = arm_obj
            if arm_obj.mode != "EDIT":
                bpy.ops.object.mode_set(mode="EDIT")
        eb = arm_obj.data.edit_bones
        for name in bone_names:
            if name not in eb:
                errors.append(f"{name}: missing in edit bones")
                continue
            b = eb[name]
            old_len = b.length
            b.length = max(1e-8, old_len * factor)
            ok.append(name)
            print(
                f"[GR2 LAB] IK bone length {name}: "
                f"{old_len:.6f} -> {b.length:.6f} (×{factor})"
            )
    except Exception as e:
        errors.append(str(e))
    finally:
        try:
            if mode_set is not None:
                mode_set(ctx, "OBJECT", active=arm_obj)
            else:
                if arm_obj.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    return ok, errors
