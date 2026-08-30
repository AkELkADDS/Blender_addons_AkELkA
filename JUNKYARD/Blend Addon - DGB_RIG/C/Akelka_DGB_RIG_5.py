# SPDX-License-Identifier: GPL-3.0-or-later
"""
Akelka DGB RIG — spine animation helper.

Reproduces the original constraint-rig spine bend from Control_Spine alone.
All constraint math is hardcoded; no reference armature is required.
"""

bl_info = {
    "name": "Akelka DGB RIG",
    "author": "Akelka",
    "version": (1, 8, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Akelka tools",
    "description": "Bend spine bones from Control_Spine and keyframe the result",
    "category": "Animation",
}

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix, Quaternion, Vector

DEFAULT_SPINE_BONES = ("Root_M", "Spine1_M", "Spine2_M", "Chest_M")
DEFAULT_CONTROL_BONE = "Control_Spine"

_SESSION = None
_frame_change_handler = None

# ---------------------------------------------------------------------------
# Hardcoded rest-pose data extracted from the original constraint rig.
# ---------------------------------------------------------------------------

_REST_FLAT = {
    "Dummy_Root": (
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        -0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ),
    "0": (
        0.0, 0.0, -1.0, -0.0,
        -1.0, 0.0, -0.0, 0.028245,
        0.0, 1.0, 0.0, 1.426394,
        0.0, 0.0, 0.0, 1.0,
    ),
    "1": (
        0.0, 0.0, -1.0, -0.0,
        -1.0, 0.0, -0.0, 0.028245,
        0.0, 1.0, 0.0, 1.624042,
        0.0, 0.0, 0.0, 1.0,
    ),
    "2": (
        0.0, 0.0, -1.0, -0.0,
        -1.0, 0.0, -0.0, 0.028245,
        0.0, 1.0, 0.0, 1.507366,
        0.0, 0.0, 0.0, 1.0,
    ),
    "3": (
        0.0, 0.0, -1.0, -0.0,
        -1.0, 0.0, -0.0, 0.028245,
        0.0, 1.0, 0.0, 1.327619,
        0.0, 0.0, 0.0, 1.0,
    ),
    "4": (
        0.0, 0.0, -1.0, -0.0,
        -1.0, 0.0, -0.0, 0.028245,
        0.0, 1.0, 0.0, 1.212843,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Fake_4": (
        0.0, 0.0, -1.0, -0.0,
        -0.032942, 0.999457, 0.0, 0.028245,
        0.999457, 0.032942, 0.0, 1.116675,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Fake_3": (
        -0.0, 0.0, -1.0, -0.0,
        -0.02205, 0.999757, 0.0, 0.024936,
        0.999757, 0.02205, -0.0, 1.217066,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Fake_2": (
        0.0, 0.0, -1.0, -0.0,
        -0.142161, 0.989843, 0.0, 0.022932,
        0.989844, 0.142161, 0.0, 1.307927,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Fake_1": (
        0.0, 0.0, -1.0, -0.0,
        -0.094596, 0.995516, 0.0, 0.010012,
        0.995516, 0.094596, 0.0, 1.397886,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Root_M": (
        0.0, 0.0, -1.0, -0.0,
        -0.032942, 0.999457, 0.0, 0.028245,
        0.999457, 0.032942, 0.0, 1.116675,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Spine1_M": (
        -0.0, 0.0, -1.0, -0.0,
        -0.02205, 0.999757, 0.0, 0.024936,
        0.999757, 0.02205, -0.0, 1.217066,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Spine2_M": (
        0.0, 0.0, -1.0, -0.0,
        -0.142161, 0.989843, 0.0, 0.022932,
        0.989844, 0.142161, 0.0, 1.307927,
        0.0, 0.0, 0.0, 1.0,
    ),
    "Chest_M": (
        0.0, 0.0, -1.0, -0.0,
        -0.094596, 0.995516, 0.0, 0.010012,
        0.995516, 0.094596, 0.0, 1.397886,
        0.0, 0.0, 0.0, 1.0,
    ),
}

_COPY_INF = {1: 0.205, 2: 0.438, 3: 0.798, 4: 0.973}
_CHILD_INF = {index: 1.0 - _COPY_INF[index] for index in _COPY_INF}
_Z_OFF = {4: 0.213551, 3: 0.098775, 2: -0.080972, 1: -0.197648}
_ROT_WEIGHTS = {
    "Root_M": 0.05,
    "Spine1_M": 0.15,
    "Spine2_M": 0.35,
    "Chest_M": 0.65,
}
_SPINE_CHAIN = ("Root_M", "Spine1_M", "Spine2_M", "Chest_M")
_FAKE_CHAIN = ("Fake_4", "Fake_3", "Fake_2", "Fake_1")
_FAKE_TARGET = {"Fake_4": 4, "Fake_3": 3, "Fake_2": 2, "Fake_1": 1}
_SPINE_FAKE = {
    "Root_M": "Fake_4",
    "Spine1_M": "Fake_3",
    "Spine2_M": "Fake_2",
    "Chest_M": "Fake_1",
}
_PARENT = {
    "Root_M": "Dummy_Root",
    "Spine1_M": "Root_M",
    "Spine2_M": "Spine1_M",
    "Chest_M": "Spine2_M",
    "Fake_4": "Dummy_Root",
    "Fake_3": "Fake_4",
    "Fake_2": "Fake_3",
    "Fake_1": "Fake_2",
}


def _matrix_from_flat(flat):
    return Matrix([flat[index * 4:(index + 1) * 4] for index in range(4)])


_REST = {name: _matrix_from_flat(flat) for name, flat in _REST_FLAT.items()}
_REST_TARGETS = {index: _REST[str(index)].translation.copy() for index in _COPY_INF}
_CONTROL_REST = _REST["0"]


_TRANSFORM_OPS = frozenset({
    "TRANSFORM_OT_translate",
    "TRANSFORM_OT_rotate",
    "TRANSFORM_OT_resize",
    "TRANSFORM_OT_transform",
})


class SpineSession:
    __slots__ = (
        "arm_name",
        "prepared_frame",
        "last_apply_frame",
        "spine_bones",
        "control_bone",
        "control_location",
        "control_rotation",
        "user_offset_loc",
        "user_offset_rot",
        "muted_constraints",
        "bone_rotations",
        "last_deltas",
        "animation_base",
        "frame_bases",
    )

    def __init__(self, arm_obj, spine_bones, control_bone, frame):
        ctrl = arm_obj.pose.bones[control_bone]
        self.arm_name = arm_obj.name
        self.prepared_frame = frame
        self.last_apply_frame = frame
        self.spine_bones = tuple(spine_bones)
        self.control_bone = control_bone
        self.control_location = ctrl.location.copy()
        self.control_rotation = ctrl.rotation_quaternion.copy()
        self.user_offset_loc = Vector((0.0, 0.0, 0.0))
        self.user_offset_rot = Quaternion((1.0, 0.0, 0.0, 0.0))
        self.muted_constraints = [
            (constraint, constraint.mute)
            for constraint in ctrl.constraints
        ]
        for constraint, _was_muted in self.muted_constraints:
            constraint.mute = True
        self.bone_rotations = {
            name: arm_obj.pose.bones[name].rotation_quaternion.copy()
            for name in spine_bones
        }
        self.last_deltas = {
            name: (
                Vector((0.0, 0.0, 0.0)),
                Quaternion((1.0, 0.0, 0.0, 0.0)),
            )
            for name in spine_bones
        }
        self.animation_base = {}
        self.frame_bases = {}
        _snapshot_animation_base(arm_obj, self)
        self.frame_bases[frame] = _copy_pose_map(self.animation_base)

    def release_constraints(self, arm_obj):
        for constraint, was_muted in self.muted_constraints:
            constraint.mute = was_muted
        self.muted_constraints = []

    def restore(self, arm_obj):
        self._strip_addon_delta(arm_obj)
        self.release_constraints(arm_obj)
        bpy.context.view_layer.update()

    def _strip_addon_delta(self, arm_obj):
        bpy.context.view_layer.update()
        for name in reversed(self.spine_bones):
            offset_loc, offset_rot = self.last_deltas.get(name, (None, None))
            if offset_loc is None or _offset_is_identity(offset_loc, offset_rot):
                continue
            pose_bone = arm_obj.pose.bones[name]
            parent_matrix = _pose_parent_matrix(arm_obj, pose_bone)
            animated_local = parent_matrix.inverted() @ pose_bone.matrix
            anim_loc, anim_rot, anim_scale = animated_local.decompose()
            clean_local = Matrix.LocRotScale(
                anim_loc - offset_loc,
                offset_rot.inverted() @ anim_rot,
                anim_scale,
            )
            pose_bone.matrix = parent_matrix @ clean_local
            self.last_deltas[name] = (
                Vector((0.0, 0.0, 0.0)),
                Quaternion((1.0, 0.0, 0.0, 0.0)),
            )


def _pose_parent_matrix(arm_obj, pose_bone):
    parent = pose_bone.parent
    if parent is None:
        return _REST["Dummy_Root"].copy()
    return parent.matrix.copy()


def _sim_parent_matrix(sim_results, bone_name):
    parent_name = _PARENT[bone_name]
    if parent_name == "Dummy_Root":
        return _REST["Dummy_Root"]
    return sim_results[parent_name]


def _fk_parent_matrix(parent_matrix, child_name):
    if parent_matrix is None:
        return _REST[child_name].copy()
    parent_name = _PARENT[child_name]
    relative = _REST[parent_name].inverted() @ _REST[child_name]
    return parent_matrix @ relative


def _control_head_from_location(location, rotation):
    return _control_pose_matrix(location, rotation).translation


def _control_pose_matrix(location, rotation):
    loc = Vector(location)
    rot = Quaternion(rotation)
    return _CONTROL_REST @ Matrix.LocRotScale(loc, rot, Vector((1.0, 1.0, 1.0)))


def _mapped_control_rotation(owner_rest, control_pose_matrix, influence):
    control_delta = _CONTROL_REST.inverted() @ control_pose_matrix
    mapped = (
        owner_rest.inverted()
        @ _CONTROL_REST
        @ control_delta
        @ _CONTROL_REST.inverted()
        @ owner_rest
    )
    return Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(mapped.to_quaternion(), influence)


def _target_head(target_index, control_head):
    offset = Vector((0.0, 0.0, _Z_OFF[target_index]))
    return (
        _REST_TARGETS[target_index] * _CHILD_INF[target_index]
        + (control_head - offset) * _COPY_INF[target_index]
    )


def _damped_track_x(bone_matrix, target_head):
    head = bone_matrix.translation.copy()
    track_axis = (bone_matrix.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    to_target = target_head - head
    if to_target.length < 1e-8:
        return bone_matrix.copy()
    to_target.normalize()
    rotation = track_axis.rotation_difference(to_target)
    rotation_matrix = rotation.to_matrix().to_4x4()
    return Matrix.Translation(head) @ rotation_matrix @ Matrix.Translation(-head) @ bone_matrix


def _live_rot_weights():
    settings = getattr(getattr(bpy.context, "scene", None), "akelka_dgb_rig", None)
    if settings is None:
        return dict(_ROT_WEIGHTS)
    return {
        "Root_M": settings.rot_root,
        "Spine1_M": settings.rot_spine1,
        "Spine2_M": settings.rot_spine2,
        "Chest_M": settings.rot_chest,
    }


def _live_rot_strengths():
    settings = getattr(getattr(bpy.context, "scene", None), "akelka_dgb_rig", None)
    if settings is None:
        return {name: 1.0 for name in _SPINE_CHAIN}
    return {
        "Root_M": settings.str_root,
        "Spine1_M": settings.str_spine1,
        "Spine2_M": settings.str_spine2,
        "Chest_M": settings.str_chest,
    }


def _scale_rotation_offset(offset_rot, strength):
    if abs(strength - 1.0) <= 1e-6:
        return offset_rot.copy()
    if strength <= 1e-6:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis, angle = offset_rot.to_axis_angle()
    return Quaternion(axis, angle * strength)


def _on_tweak_update(self, context):
    global _SESSION
    if _SESSION is None or not _SESSION.animation_base:
        return
    if all(
        _offset_is_identity(offset_loc, offset_rot)
        for offset_loc, offset_rot in _SESSION.last_deltas.values()
    ):
        return
    arm = bpy.data.objects.get(_SESSION.arm_name)
    if arm is None:
        return
    apply_spine_from_control(arm, _SESSION)


def _simulate_spine(control_location, control_rotation, rot_weights=None):
    if rot_weights is None:
        rot_weights = _live_rot_weights()
    control_pose_matrix = _control_pose_matrix(control_location, control_rotation)
    control_head = control_pose_matrix.translation
    targets = {
        index: _target_head(index, control_head)
        for index in _COPY_INF
    }

    fake_parent = None
    fake_results = {}
    for fake_name in _FAKE_CHAIN:
        base_matrix = _fk_parent_matrix(fake_parent, fake_name)
        target_index = _FAKE_TARGET[fake_name]
        fake_results[fake_name] = _damped_track_x(base_matrix, targets[target_index])
        fake_parent = fake_results[fake_name]

    spine_results = {}
    for spine_name in _SPINE_CHAIN:
        fake_name = _SPINE_FAKE[spine_name]
        fake_parent_name = _PARENT[fake_name]
        fake_parent_matrix = (
            None if fake_parent_name == "Dummy_Root" else fake_results[fake_parent_name]
        )
        fake_rest = _fk_parent_matrix(fake_parent_matrix, fake_name)
        fake_delta = fake_rest.inverted() @ fake_results[fake_name]

        spine_parent_name = _PARENT[spine_name]
        if spine_parent_name == "Dummy_Root":
            parent_matrix = _REST["Dummy_Root"]
        else:
            parent_matrix = spine_results[spine_parent_name]

        copy_transforms_matrix = _fk_parent_matrix(parent_matrix, spine_name) @ fake_delta
        local_matrix = parent_matrix.inverted() @ copy_transforms_matrix
        partial_rotation = _mapped_control_rotation(
            _REST[spine_name],
            control_pose_matrix,
            rot_weights.get(spine_name, _ROT_WEIGHTS[spine_name]),
        )
        local_matrix @= Matrix.LocRotScale(
            Vector((0.0, 0.0, 0.0)),
            partial_rotation,
            Vector((1.0, 1.0, 1.0)),
        )
        spine_results[spine_name] = parent_matrix @ local_matrix

    return spine_results


def _get_armature(context):
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    settings = context.scene.akelka_dgb_rig
    if settings.armature and settings.armature.type == "ARMATURE":
        return settings.armature
    return None


def _parse_spine_bones(settings):
    names = [part.strip() for part in settings.spine_bones.split(",") if part.strip()]
    return names or list(DEFAULT_SPINE_BONES)


def _validate_bones(arm_obj, settings):
    missing = []
    if settings.control_bone not in arm_obj.pose.bones:
        missing.append(settings.control_bone)
    for name in _parse_spine_bones(settings):
        if name not in arm_obj.pose.bones:
            missing.append(name)
    return missing


def _simulate_spine_local_offset(session, control_location, control_rotation):
    """Local rotation (around bone head) and translation offset from control movement."""
    rot_weights = _live_rot_weights()
    strengths = _live_rot_strengths()
    sim_base = _simulate_spine(
        session.control_location,
        session.control_rotation,
        rot_weights,
    )
    sim_now = _simulate_spine(
        Vector(control_location),
        Quaternion(control_rotation),
        rot_weights,
    )
    offsets = {}
    for name in session.spine_bones:
        parent_base = _sim_parent_matrix(sim_base, name)
        parent_now = _sim_parent_matrix(sim_now, name)
        local_base = parent_base.inverted() @ sim_base[name]
        local_now = parent_now.inverted() @ sim_now[name]
        base_loc, base_rot, _ = local_base.decompose()
        now_loc, now_rot, _ = local_now.decompose()
        offsets[name] = (
            (now_loc - base_loc) * strengths.get(name, 1.0),
            _scale_rotation_offset(
                now_rot @ base_rot.inverted(),
                strengths.get(name, 1.0),
            ),
        )
    return offsets


def _compute_bend_offsets(session):
    """Spine bend from manual control offset.

    Depends ONLY on session.control_* baseline + session.user_offset_*.
    Frame number never enters this function.
    """
    manual_loc, manual_rot = _manual_pose(session)
    return _simulate_spine_local_offset(session, manual_loc, manual_rot)


def _has_manual_offset(session):
    if session.user_offset_loc.length > _OFFSET_LOC_EPS:
        return True
    return abs(session.user_offset_rot.angle) > _OFFSET_ROT_EPS


def _peel_bend_from_pose(arm_obj, session, bend_offsets):
    """Inverse of apply — remove a bend layer from the current spine pose."""
    bpy.context.view_layer.update()
    for name in reversed(session.spine_bones):
        offset_loc, offset_rot = bend_offsets.get(name, (None, None))
        if offset_loc is None or _offset_is_identity(offset_loc, offset_rot):
            continue
        pose_bone = arm_obj.pose.bones[name]
        parent_matrix = _pose_parent_matrix(arm_obj, pose_bone)
        current_local = parent_matrix.inverted() @ pose_bone.matrix
        loc, rot, scale = current_local.decompose()
        clean_local = Matrix.LocRotScale(
            loc - offset_loc,
            offset_rot.inverted() @ rot,
            scale,
        )
        pose_bone.matrix = parent_matrix @ clean_local


_OFFSET_LOC_EPS = 1e-6
_OFFSET_ROT_EPS = 1e-6


def _offset_is_identity(offset_loc, offset_rot):
    if offset_loc.length > _OFFSET_LOC_EPS:
        return False
    return abs(offset_rot.angle) <= _OFFSET_ROT_EPS


_TRANSFORM_KEYS = frozenset({
    "G",
    "R",
    "S",
    "E",
})


def _control_bone_selected(context, control_bone):
    if context.active_pose_bone is not None:
        if context.active_pose_bone.name == control_bone:
            return True
    return any(
        pose_bone.name == control_bone
        for pose_bone in (context.selected_pose_bones or [])
    )


def _control_moved_from_manual(arm_obj, session, control_bone):
    ctrl = arm_obj.pose.bones[control_bone]
    manual_loc, manual_rot = _manual_pose(session)
    if (ctrl.location - manual_loc).length > _OFFSET_LOC_EPS:
        return True
    return ctrl.rotation_quaternion.rotation_difference(manual_rot).angle > _OFFSET_ROT_EPS


def _manual_pose(session):
    return (
        session.control_location + session.user_offset_loc,
        session.user_offset_rot @ session.control_rotation,
    )


def _is_user_dragging_control(context, control_bone):
    if context.mode != "POSE":
        return False
    op = context.active_operator
    if op is None or op.bl_idname not in _TRANSFORM_OPS:
        return False
    if context.active_pose_bone is not None:
        if context.active_pose_bone.name == control_bone:
            return True
    return any(
        pose_bone.name == control_bone
        for pose_bone in (context.selected_pose_bones or [])
    )


def _hold_control_manual_pose(arm_obj, session):
    """Show baseline + manual offset; blocks control-bone animation on frame scrub."""
    ctrl = arm_obj.pose.bones[session.control_bone]
    manual_loc, manual_rot = _manual_pose(session)
    ctrl.location = manual_loc.copy()
    ctrl.rotation_quaternion = manual_rot.copy()


def _capture_user_offset(session, ctrl):
    new_offset_loc = ctrl.location - session.control_location
    new_offset_rot = ctrl.rotation_quaternion @ session.control_rotation.inverted()
    changed = (
        (new_offset_loc - session.user_offset_loc).length > _OFFSET_LOC_EPS
        or abs(new_offset_rot.rotation_difference(session.user_offset_rot).angle) > _OFFSET_ROT_EPS
    )
    if changed:
        session.user_offset_loc = new_offset_loc.copy()
        session.user_offset_rot = new_offset_rot.copy()
    return changed


def _reset_addon_deltas(session):
    for name in session.spine_bones:
        session.last_deltas[name] = (
            Vector((0.0, 0.0, 0.0)),
            Quaternion((1.0, 0.0, 0.0, 0.0)),
        )


def _bake_spine_and_zero_control_math(arm_obj, session, *, clear_location, clear_rotation):
    """Bake current spine pose, zero control channels, start math from rest.

    Spine bones keep their visual pose. Further control moves add a new bend.
    """
    bpy.context.view_layer.update()
    _snapshot_animation_base(arm_obj, session)
    _reset_addon_deltas(session)
    session.frame_bases[session.prepared_frame] = _copy_pose_map(session.animation_base)

    ctrl = arm_obj.pose.bones[session.control_bone]
    if clear_location:
        ctrl.location = Vector((0.0, 0.0, 0.0))
    if clear_rotation:
        ctrl.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        if ctrl.rotation_mode != "QUATERNION":
            ctrl.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    session.control_location = ctrl.location.copy()
    session.control_rotation = ctrl.rotation_quaternion.copy()
    session.user_offset_loc = Vector((0.0, 0.0, 0.0))
    session.user_offset_rot = Quaternion((1.0, 0.0, 0.0, 0.0))


def _clear_control_and_apply_math(arm_obj, session, *, clear_location, clear_rotation):
    """Normal Alt-G / Alt-R: zero the control and let spine math follow."""
    ctrl = arm_obj.pose.bones[session.control_bone]
    if clear_location:
        ctrl.location = Vector((0.0, 0.0, 0.0))
    if clear_rotation:
        ctrl.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        if ctrl.rotation_mode != "QUATERNION":
            ctrl.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    _capture_user_offset(session, ctrl)
    apply_spine_from_control(arm_obj, session)
    session.last_apply_frame = bpy.context.scene.frame_current


_MOUSEMOVE_EVENTS = frozenset({
    "MOUSEMOVE",
    "INBETWEEN_MOUSEMOVE",
    "PADPAN",
})


def _restore_spine_to_animation_base(arm_obj, session):
    """Reset the spine chain to the frozen animation base (parent to child)."""
    for name in session.spine_bones:
        pose_bone = arm_obj.pose.bones[name]
        parent_matrix = _pose_parent_matrix(arm_obj, pose_bone)
        pose_bone.matrix = parent_matrix @ session.animation_base[name]


def _copy_pose_map(pose_map):
    return {name: matrix.copy() for name, matrix in pose_map.items()}


def _snapshot_animation_base(arm_obj, session):
    """Freeze the current animated local pose per spine bone for this frame."""
    bpy.context.view_layer.update()
    session.animation_base = {}
    for name in session.spine_bones:
        pose_bone = arm_obj.pose.bones[name]
        parent_matrix = _pose_parent_matrix(arm_obj, pose_bone)
        session.animation_base[name] = (
            parent_matrix.inverted() @ pose_bone.matrix
        ).copy()


def _spine_bone_selected(context, session):
    names = set(session.spine_bones)
    if context.active_pose_bone is not None:
        if context.active_pose_bone.name in names:
            return True
    return any(
        pose_bone.name in names
        for pose_bone in (context.selected_pose_bones or [])
    )


def _is_user_dragging_spine(context, session):
    if context.mode != "POSE":
        return False
    op = context.active_operator
    if op is None or op.bl_idname not in _TRANSFORM_OPS:
        return False
    if _control_bone_selected(context, session.control_bone):
        return False
    return _spine_bone_selected(context, session)


def _matrices_close(matrix_a, matrix_b, eps=1e-4):
    for row in range(4):
        for col in range(4):
            if abs(matrix_a[row][col] - matrix_b[row][col]) > eps:
                return False
    return True


def _current_local_minus_bend(arm_obj, session):
    """Current spine local pose with the control overlay removed."""
    bpy.context.view_layer.update()
    cleaned = {}
    for name in session.spine_bones:
        pose_bone = arm_obj.pose.bones[name]
        parent_matrix = _pose_parent_matrix(arm_obj, pose_bone)
        current_local = parent_matrix.inverted() @ pose_bone.matrix
        loc, rot, scale = current_local.decompose()
        offset_loc, offset_rot = session.last_deltas.get(
            name,
            (Vector((0.0, 0.0, 0.0)), Quaternion((1.0, 0.0, 0.0, 0.0))),
        )
        cleaned[name] = Matrix.LocRotScale(
            loc - offset_loc,
            offset_rot.inverted() @ rot,
            scale,
        )
    return cleaned


def _spine_edits_pending(arm_obj, session):
    if not session.animation_base:
        return False
    cleaned = _current_local_minus_bend(arm_obj, session)
    return any(
        not _matrices_close(cleaned[name], session.animation_base[name])
        for name in session.spine_bones
    )


def _commit_spine_edits(arm_obj, session):
    """Bake manual spine bone edits into the animation base for this frame."""
    _peel_bend_from_pose(arm_obj, session, session.last_deltas)
    bpy.context.view_layer.update()
    _snapshot_animation_base(arm_obj, session)
    _reset_addon_deltas(session)
    session.frame_bases[session.prepared_frame] = _copy_pose_map(session.animation_base)


def _commit_spine_edits_if_changed(arm_obj, session):
    if not _spine_edits_pending(arm_obj, session):
        return False
    _commit_spine_edits(arm_obj, session)
    return True


def _begin_control_drag(arm_obj, session):
    """Start a new grab: spine pose now is the base, control now is zero delta."""
    bpy.context.view_layer.update()
    _snapshot_animation_base(arm_obj, session)
    _reset_addon_deltas(session)
    ctrl = arm_obj.pose.bones[session.control_bone]
    session.control_location = ctrl.location.copy()
    session.control_rotation = ctrl.rotation_quaternion.copy()
    session.user_offset_loc = Vector((0.0, 0.0, 0.0))
    session.user_offset_rot = Quaternion((1.0, 0.0, 0.0, 0.0))


def _forget_overlay_on_frame_change(session, context):
    """On scrub: stop tracking overlay. Do not write spine bones — keys win."""
    frame = context.scene.frame_current
    if frame == session.prepared_frame:
        return False
    session.prepared_frame = frame
    session.last_apply_frame = frame
    _reset_addon_deltas(session)
    session.animation_base = {}
    session.user_offset_loc = Vector((0.0, 0.0, 0.0))
    session.user_offset_rot = Quaternion((1.0, 0.0, 0.0, 0.0))
    return True


@persistent
def _akelka_on_frame_change(scene):
    """Forget overlay tracking on timeline change. Never overwrite keyed bones."""
    global _SESSION
    if _SESSION is None:
        return
    _forget_overlay_on_frame_change(_SESSION, bpy.context)


def _register_session_handlers():
    global _frame_change_handler
    if _akelka_on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_akelka_on_frame_change)
    _frame_change_handler = _akelka_on_frame_change


def _unregister_session_handlers():
    global _frame_change_handler
    if _akelka_on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_akelka_on_frame_change)
    _frame_change_handler = None


def _apply_control_move(arm_obj, session, context):
    if not session.animation_base:
        _begin_control_drag(arm_obj, session)
    ctrl = arm_obj.pose.bones[session.control_bone]
    if not _capture_user_offset(session, ctrl):
        return False
    apply_spine_from_control(arm_obj, session)
    session.last_apply_frame = context.scene.frame_current
    return True


def _try_apply_manual_move(arm_obj, session, context, control_bone, *, live_timer=False):
    """Apply spine while the control bone is being transformed."""
    if _is_user_dragging_spine(context, session):
        return False
    dragging = _is_user_dragging_control(context, control_bone)
    moved = _control_moved_from_manual(arm_obj, session, control_bone)
    if not dragging and not (live_timer and moved):
        return False
    return _apply_control_move(arm_obj, session, context)


def apply_spine_from_control(arm_obj, session):
    """animated_pose(frame) + bend(user_offset).

    bend() is hardcoded and frame-independent.
    animation_base is the only thing that changes per frame.
    """
    if not session.animation_base:
        _snapshot_animation_base(arm_obj, session)

    bend_offsets = _compute_bend_offsets(session)
    if all(
        _offset_is_identity(offset_loc, offset_rot)
        for offset_loc, offset_rot in bend_offsets.values()
    ):
        if all(
            _offset_is_identity(offset_loc, offset_rot)
            for offset_loc, offset_rot in session.last_deltas.values()
        ):
            return
        _restore_spine_to_animation_base(arm_obj, session)
        _reset_addon_deltas(session)
        bpy.context.view_layer.update()
        return

    _restore_spine_to_animation_base(arm_obj, session)
    bpy.context.view_layer.update()

    for name in session.spine_bones:
        pose_bone = arm_obj.pose.bones[name]
        parent_matrix = _pose_parent_matrix(arm_obj, pose_bone)
        animated_local = session.animation_base[name]
        anim_loc, anim_rot, anim_scale = animated_local.decompose()
        offset_loc, offset_rot = bend_offsets[name]
        final_local = Matrix.LocRotScale(
            anim_loc + offset_loc,
            offset_rot @ anim_rot,
            anim_scale,
        )
        pose_bone.matrix = parent_matrix @ final_local
        session.last_deltas[name] = (offset_loc.copy(), offset_rot.copy())

    bpy.context.view_layer.update()


def keyframe_spine_pose(arm_obj, session, frame):
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()

    if arm_obj.animation_data.action is None:
        arm_obj.animation_data.action = bpy.data.actions.new(name=f"{arm_obj.name}_Action")

    for name in session.spine_bones:
        pose_bone = arm_obj.pose.bones[name]
        pose_bone.keyframe_insert(data_path="location", frame=frame)
        pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def _delete_control_keys_on_frame(arm_obj, control_bone, frame):
    """Drop auto-keys on the helper control so it never holds animation."""
    ctrl = arm_obj.pose.bones[control_bone]
    for data_path in (
        "location",
        "rotation_quaternion",
        "rotation_euler",
        "rotation_axis_angle",
        "scale",
    ):
        try:
            ctrl.keyframe_delete(data_path=data_path, frame=frame)
        except RuntimeError:
            pass


def _autokey_enabled(context):
    return bool(context.scene.tool_settings.use_keyframe_insert_auto)


def _autokey_spine_not_control(arm_obj, session, context):
    """If auto-key is on, key spine bones and strip keys on the control."""
    frame = context.scene.frame_current
    keyframe_spine_pose(arm_obj, session, frame)
    _delete_control_keys_on_frame(arm_obj, session.control_bone, frame)


def _make_session(context, arm_obj, settings):
    return SpineSession(
        arm_obj,
        _parse_spine_bones(settings),
        settings.control_bone,
        context.scene.frame_current,
    )


class AkelkaDGBRigSettings(PropertyGroup):
    armature: PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    control_bone: StringProperty(
        name="Control Bone",
        default=DEFAULT_CONTROL_BONE,
    )
    spine_bones: StringProperty(
        name="Spine Bones",
        description="Comma-separated bone names, base to chest",
        default=", ".join(DEFAULT_SPINE_BONES),
    )
    session_active: BoolProperty(
        name="Session Active",
        default=False,
        options={"SKIP_SAVE"},
    )
    bake_on_alt_clear: BoolProperty(
        name="Alt-G / Alt-R Bake Reset",
        description="ON: bake spine pose, zero control, keep bones. OFF: zero control and spine follows the math (unbend)",
        default=True,
    )
    rot_root: FloatProperty(
        name="Root Rotation",
        description="How much Root_M copies Control_Spine rotation",
        default=0.05,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_tweak_update,
    )
    rot_spine1: FloatProperty(
        name="Spine1 Rotation",
        description="How much Spine1_M copies Control_Spine rotation",
        default=0.15,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_tweak_update,
    )
    rot_spine2: FloatProperty(
        name="Spine2 Rotation",
        description="How much Spine2_M copies Control_Spine rotation",
        default=0.35,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_tweak_update,
    )
    rot_chest: FloatProperty(
        name="Chest Rotation",
        description="How much Chest_M copies Control_Spine rotation",
        default=0.65,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        update=_on_tweak_update,
    )
    str_root: FloatProperty(
        name="Root Strength",
        description="Scale Root_M bend from the control. 1.0 is default",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_on_tweak_update,
    )
    str_spine1: FloatProperty(
        name="Spine1 Strength",
        description="Scale Spine1_M bend from the control. 1.0 is default",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_on_tweak_update,
    )
    str_spine2: FloatProperty(
        name="Spine2 Strength",
        description="Scale Spine2_M bend from the control. 1.0 is default",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_on_tweak_update,
    )
    str_chest: FloatProperty(
        name="Chest Strength",
        description="Scale Chest_M bend from the control. 1.0 is default",
        default=1.0,
        min=0.0,
        max=2.0,
        update=_on_tweak_update,
    )


class AKELKA_OT_spine_adjust(Operator):
    """Move Control_Spine; spine updates live while dragging. Enter confirms."""

    bl_idname = "akelka_dgb_rig.spine_adjust"
    bl_label = "Adjust Spine"
    bl_options = {"REGISTER", "UNDO"}

    _drag_timer = None

    def _start_drag_timer(self, context):
        if self._drag_timer is None:
            self._drag_timer = context.window_manager.event_timer_add(
                0.03,
                window=context.window,
            )

    def _stop_drag_timer(self, context):
        if self._drag_timer is not None:
            context.window_manager.event_timer_remove(self._drag_timer)
            self._drag_timer = None

    def _pause_autokey(self, context):
        tool_settings = context.scene.tool_settings
        if self._saved_autokey is None:
            self._saved_autokey = tool_settings.use_keyframe_insert_auto
            tool_settings.use_keyframe_insert_auto = False

    def _resume_autokey(self, context):
        if self._saved_autokey is not None:
            context.scene.tool_settings.use_keyframe_insert_auto = self._saved_autokey
            self._saved_autokey = None

    @classmethod
    def poll(cls, context):
        if _SESSION is not None:
            return False
        arm = _get_armature(context)
        if arm is None:
            return False
        return not _validate_bones(arm, context.scene.akelka_dgb_rig)

    def modal(self, context, event):
        global _SESSION

        arm = bpy.data.objects.get(_SESSION.arm_name) if _SESSION else None

        if arm is None:
            self._cleanup(context)
            return {"CANCELLED"}

        settings = context.scene.akelka_dgb_rig
        control_bone = settings.control_bone

        if self._strip_control_keys:
            _delete_control_keys_on_frame(
                arm,
                control_bone,
                context.scene.frame_current,
            )

        frame_before = _SESSION.prepared_frame
        _forget_overlay_on_frame_change(_SESSION, context)

        if (
            context.scene.frame_current != frame_before
            and self._drag_timer is not None
            and not _is_user_dragging_control(context, control_bone)
        ):
            self._stop_drag_timer(context)

        if event.value == "PRESS":
            if (
                event.alt
                and event.type in {"G", "R"}
                and _control_bone_selected(context, control_bone)
            ):
                if settings.bake_on_alt_clear:
                    _bake_spine_and_zero_control_math(
                        arm,
                        _SESSION,
                        clear_location=(event.type == "G"),
                        clear_rotation=(event.type == "R"),
                    )
                else:
                    if not _SESSION.animation_base:
                        _begin_control_drag(arm, _SESSION)
                    _clear_control_and_apply_math(
                        arm,
                        _SESSION,
                        clear_location=(event.type == "G"),
                        clear_rotation=(event.type == "R"),
                    )
                    if _autokey_enabled(context):
                        _autokey_spine_not_control(arm, _SESSION, context)
                        self._strip_control_keys = True
                self._drag_active = False
                self._spine_edit_active = False
                self._stop_drag_timer(context)
                return {"RUNNING_MODAL"}

            if event.type in _TRANSFORM_KEYS and not event.alt:
                if _control_bone_selected(context, control_bone):
                    _begin_control_drag(arm, _SESSION)
                    self._pause_autokey(context)
                    self._drag_offset_loc = _SESSION.user_offset_loc.copy()
                    self._drag_offset_rot = _SESSION.user_offset_rot.copy()
                    self._drag_active = False
                    self._spine_edit_active = False
                    self._start_drag_timer(context)
                elif _spine_bone_selected(context, _SESSION):
                    self._spine_edit_active = True
                    self._stop_drag_timer(context)

        live_timer = self._drag_timer is not None
        if event.type == "TIMER" and live_timer:
            if _try_apply_manual_move(
                arm,
                _SESSION,
                context,
                control_bone,
                live_timer=True,
            ):
                self._drag_active = True
                self._pause_autokey(context)

        if event.type in _MOUSEMOVE_EVENTS:
            if _try_apply_manual_move(arm, _SESSION, context, control_bone):
                self._drag_active = True
                self._pause_autokey(context)

        if event.type == "RIGHTMOUSE" and event.value == "PRESS":
            if self._spine_edit_active:
                self._spine_edit_active = False
            elif self._drag_active or live_timer:
                _SESSION.user_offset_loc = self._drag_offset_loc.copy()
                _SESSION.user_offset_rot = self._drag_offset_rot.copy()
                _hold_control_manual_pose(arm, _SESSION)
                apply_spine_from_control(arm, _SESSION)
                self._drag_active = False
                self._stop_drag_timer(context)
                self._resume_autokey(context)
                self._strip_control_keys = False

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self._spine_edit_active:
                self._spine_edit_active = False
            elif self._drag_active:
                _apply_control_move(arm, _SESSION, context)
                should_autokey = self._saved_autokey is True or (
                    self._saved_autokey is None and _autokey_enabled(context)
                )
                if should_autokey:
                    _autokey_spine_not_control(arm, _SESSION, context)
                    self._strip_control_keys = True
            self._drag_active = False
            self._stop_drag_timer(context)
            self._resume_autokey(context)

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._stop_drag_timer(context)
            if self._drag_active:
                _apply_control_move(arm, _SESSION, context)
            keyframe_spine_pose(arm, _SESSION, context.scene.frame_current)
            _delete_control_keys_on_frame(
                arm,
                settings.control_bone,
                context.scene.frame_current,
            )
            self.report({"INFO"}, "Spine keyframed on frame %d" % context.scene.frame_current)
            self._cleanup(context)
            return {"FINISHED"}

        if event.type == "ESC" and event.value == "PRESS":
            _SESSION.restore(arm)
            self.report({"INFO"}, "Spine adjust cancelled")
            self._cleanup(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def _cleanup(self, context):
        global _SESSION
        self._stop_drag_timer(context)
        self._resume_autokey(context)
        if _SESSION is not None:
            arm = bpy.data.objects.get(_SESSION.arm_name)
            if arm is not None:
                _SESSION.release_constraints(arm)
        _unregister_session_handlers()
        _SESSION = None
        context.scene.akelka_dgb_rig.session_active = False

    def invoke(self, context, event):
        global _SESSION

        settings = context.scene.akelka_dgb_rig
        arm = _get_armature(context)
        missing = _validate_bones(arm, settings)
        if missing:
            self.report({"ERROR"}, "Missing bones: %s" % ", ".join(missing))
            return {"CANCELLED"}

        settings.armature = arm
        _SESSION = _make_session(context, arm, settings)
        settings.session_active = True
        _register_session_handlers()
        self._drag_timer = None
        self._drag_active = False
        self._spine_edit_active = False
        self._saved_autokey = None
        self._strip_control_keys = False
        self._drag_offset_loc = Vector((0.0, 0.0, 0.0))
        self._drag_offset_rot = Quaternion((1.0, 0.0, 0.0, 0.0))

        context.window_manager.modal_handler_add(self)
        self.report(
            {"INFO"},
            "Grab Control_Spine — spine updates live while dragging. Enter keys spine, Esc cancels",
        )
        return {"RUNNING_MODAL"}


class AKELKA_OT_spine_apply(Operator):
    """Apply spine pose from the current control position."""

    bl_idname = "akelka_dgb_rig.spine_apply"
    bl_label = "Apply Spine"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        arm = _get_armature(context)
        return arm is not None and not _validate_bones(arm, context.scene.akelka_dgb_rig)

    def execute(self, context):
        global _SESSION
        settings = context.scene.akelka_dgb_rig
        arm = _get_armature(context)
        missing = _validate_bones(arm, settings)
        if missing:
            self.report({"ERROR"}, "Missing bones: %s" % ", ".join(missing))
            return {"CANCELLED"}

        if _SESSION is not None and _SESSION.arm_name == arm.name:
            apply_spine_from_control(arm, _SESSION)
        else:
            session = _make_session(context, arm, settings)
            apply_spine_from_control(arm, session)
        self.report({"INFO"}, "Spine pose applied")
        return {"FINISHED"}


class AKELKA_OT_spine_keyframe(Operator):
    """Keyframe spine bones on the current frame."""

    bl_idname = "akelka_dgb_rig.spine_keyframe"
    bl_label = "Keyframe Spine"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        arm = _get_armature(context)
        return arm is not None and not _validate_bones(arm, context.scene.akelka_dgb_rig)

    def execute(self, context):
        global _SESSION
        settings = context.scene.akelka_dgb_rig
        arm = _get_armature(context)
        missing = _validate_bones(arm, settings)
        if missing:
            self.report({"ERROR"}, "Missing bones: %s" % ", ".join(missing))
            return {"CANCELLED"}

        if _SESSION is not None and _SESSION.arm_name == arm.name:
            session = _SESSION
        else:
            session = _make_session(context, arm, settings)
        apply_spine_from_control(arm, session)
        keyframe_spine_pose(arm, session, context.scene.frame_current)
        self.report({"INFO"}, "Keyed on frame %d" % context.scene.frame_current)
        return {"FINISHED"}


class AKELKA_OT_spine_reset(Operator):
    """Reset control and spine to the pose captured when the session started."""

    bl_idname = "akelka_dgb_rig.spine_reset"
    bl_label = "Reset Spine Session"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _SESSION is not None

    def execute(self, context):
        global _SESSION
        arm = bpy.data.objects.get(_SESSION.arm_name)
        if arm is not None:
            _SESSION.restore(arm)
        _SESSION = None
        context.scene.akelka_dgb_rig.session_active = False
        self.report({"INFO"}, "Spine session reset")
        return {"FINISHED"}


class AKELKA_OT_tweaks_reset(Operator):
    """Restore default rotation influence and strength for each spine bone."""

    bl_idname = "akelka_dgb_rig.tweaks_reset"
    bl_label = "Reset Tweaks"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.akelka_dgb_rig
        settings.rot_root = _ROT_WEIGHTS["Root_M"]
        settings.rot_spine1 = _ROT_WEIGHTS["Spine1_M"]
        settings.rot_spine2 = _ROT_WEIGHTS["Spine2_M"]
        settings.rot_chest = _ROT_WEIGHTS["Chest_M"]
        settings.str_root = 1.0
        settings.str_spine1 = 1.0
        settings.str_spine2 = 1.0
        settings.str_chest = 1.0
        self.report({"INFO"}, "Spine tweaks restored to defaults")
        return {"FINISHED"}


class AKELKA_PT_tools(Panel):
    bl_label = "Spine Helper"
    bl_idname = "AKELKA_PT_dgb_rig"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Akelka tools"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.akelka_dgb_rig
        arm = _get_armature(context)

        if arm is None:
            layout.label(text="Select an armature", icon="ERROR")
        else:
            layout.label(text=arm.name, icon="ARMATURE_DATA")

        box = layout.box()
        box.label(text="Bones")
        box.prop(settings, "control_bone", text="Control")
        box.prop(settings, "spine_bones", text="Spine")

        box = layout.box()
        box.prop(settings, "bake_on_alt_clear")

        box = layout.box()
        row = box.row()
        row.label(text="Rotation Tweaks")
        row.operator("akelka_dgb_rig.tweaks_reset", text="", icon="LOOP_BACK")
        grid = box.grid_flow(row_major=True, columns=2, align=True)
        grid.label(text="Influence")
        grid.label(text="Strength")
        grid.prop(settings, "rot_root", text="Root", slider=True)
        grid.prop(settings, "str_root", text="Root", slider=True)
        grid.prop(settings, "rot_spine1", text="Spine1", slider=True)
        grid.prop(settings, "str_spine1", text="Spine1", slider=True)
        grid.prop(settings, "rot_spine2", text="Spine2", slider=True)
        grid.prop(settings, "str_spine2", text="Spine2", slider=True)
        grid.prop(settings, "rot_chest", text="Chest", slider=True)
        grid.prop(settings, "str_chest", text="Chest", slider=True)

        layout.label(text="Control bends spine only while you drag it", icon="MOD_SIMPLEDEFORM")

        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("akelka_dgb_rig.spine_adjust", icon="MOD_SIMPLEDEFORM")
        row = col.row(align=True)
        row.operator("akelka_dgb_rig.spine_apply", icon="FILE_REFRESH")
        row.operator("akelka_dgb_rig.spine_keyframe", icon="KEY_HLT")

        if settings.session_active:
            layout.label(text="Session on — auto-key writes spine, not control", icon="INFO")
            layout.operator("akelka_dgb_rig.spine_reset", icon="LOOP_BACK")


classes = (
    AkelkaDGBRigSettings,
    AKELKA_OT_spine_adjust,
    AKELKA_OT_spine_apply,
    AKELKA_OT_spine_keyframe,
    AKELKA_OT_spine_reset,
    AKELKA_OT_tweaks_reset,
    AKELKA_PT_tools,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.akelka_dgb_rig = PointerProperty(type=AkelkaDGBRigSettings)


def unregister():
    global _SESSION
    _SESSION = None
    _unregister_session_handlers()
    del bpy.types.Scene.akelka_dgb_rig
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
