
bl_info = {
    "name": "Akelka Animation Fixer",
    "author": "Akelka",
    "version": (1, 1, 1),
    "blender": (4, 5, 2),
    "location": "View3D > Sidebar (N) > Pose Align",
    "description": "Rotate parent bone so child (head/tail) moves close to target (head/tail). Supports multiple Parent/Child/Target combos (add/remove). Analytic + iterative solvers, modal bake, pick selected bone. Uses Graph Editor Gaussian Smooth for bone F-curves (calls bpy.ops.graph.gaussian_smooth safely). Includes preset sets and a one-click quickfix.",
    "category": "Rigging",
}

import bpy
import math
import os
import traceback
import concurrent.futures
from contextlib import contextmanager
from mathutils import Euler, Quaternion, Vector

# -----------------------------
# PropertyGroup for a triple
# -----------------------------
class AlignTriple(bpy.types.PropertyGroup):
    parent_bone: bpy.props.StringProperty(name="Parent Bone")
    child_bone: bpy.props.StringProperty(name="Child Bone")
    target_bone: bpy.props.StringProperty(name="Target Bone")
    locked_axis: bpy.props.EnumProperty(
        name="Locked Axis",
        items=[
            ('X', "X", "Lock X axis (prevent left/right rotation)"),
            ('Y', "Y", "Lock Y axis (prevent forward/back rotation)"),
            ('Z', "Z", "Lock Z axis (prevent up/down rotation)"),
        ],
        default='X',
        description="Which axis to lock for this bone combo (prevents rotation around this axis)"
    )

# -----------------------------
# Helper function to get armature from context
# -----------------------------
def get_armature_from_context(context):
    """Get armature from active/selected objects in context."""
    obj = context.active_object
    if obj and getattr(obj, 'type', None) == 'ARMATURE':
        return obj
    for o in context.selected_objects:
        if getattr(o, 'type', None) == 'ARMATURE':
            return o
    return None

def update_show_advanced(self, context):
    """Auto-load 4 sets when advanced settings are first shown."""
    if not self.show_advanced:
        return  # Don't do anything when hiding
    
    # Auto-load 4 sets if no triples exist
    if len(self.align_triples) == 0:
        try:
            bpy.ops.scene.load_default_sets()
        except Exception:
            pass  # Silently fail if it doesn't work

def register_props():
    sc = bpy.types.Scene

    # Show/hide manual work settings (collapsible)
    sc.show_advanced = bpy.props.BoolProperty(
        name="Manual Work",
        description="Show manual triples, smoothing, and bake controls inside Old Method",
        default=False,
        update=update_show_advanced,
    )

    # Single legacy fields (kept for backwards compatibility)
    sc.align_parent_bone = bpy.props.StringProperty(name="Parent Bone")
    sc.align_child_bone = bpy.props.StringProperty(name="Child Bone")
    sc.align_target_bone = bpy.props.StringProperty(name="Target Bone")

    sc.align_mode = bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('TAIL_TO_TAIL', "Tail → Tail", "Minimize distance: child.tail -> target.tail"),
            ('HEAD_TO_HEAD', "Head → Head", "Minimize distance: child.head -> target.head"),
            ('TAIL_TO_HEAD', "Tail → Head", "Minimize distance: child.tail -> target.head"),
            ('HEAD_TO_TAIL', "Head → Tail", "Minimize distance: child.head -> target.tail"),
        ],
        default='HEAD_TO_HEAD'
    )

    sc.align_initial_step = bpy.props.FloatProperty(
        name="Initial step (rad)", default=0.4, min=1e-4, max=3.14,
        description="Starting angle step in radians for iterative search"
    )
    sc.align_tol = bpy.props.FloatProperty(
        name="Tolerance (rad)", default=1e-4, min=1e-6,
        description="Stop when step goes below this"
    )
    sc.align_max_iter = bpy.props.IntProperty(
        name="Max iterations", default=200, min=1,
        description="Maximum outer iterations for iterative solver"
    )

    sc.align_bake_method = bpy.props.EnumProperty(
        name="Method",
        items=[
            ('ANALYTIC', "Analytic", "Apply analytic per frame"),
            ('ITERATIVE', "Iterative", "Apply iterative per frame"),
            ('COMBO', "Analytic + Iterative", "Analytic then iterative per frame"),
        ],
        default='COMBO'
    )

    sc.align_bake_mode = bpy.props.EnumProperty(
        name="Bake Mode",
        items=[
            ('RANGE', "Bake by Range", "Bake using scene frame range"),
            ('KEYFRAMES', "Bake by Keyframes", "Bake from first to last keyframe in action"),
        ],
        default='KEYFRAMES',
        description="Choose whether to bake by scene range or by keyframe range"
    )

    # Collection of triples
    sc.align_triples = bpy.props.CollectionProperty(type=AlignTriple)
    sc.align_triples_index = bpy.props.IntProperty(name="Active Triple Index", default=0)

    # Lightweight progress flags
    sc.align_is_baking = bpy.props.BoolProperty(name="Align Is Baking", default=False)
    sc.align_bake_progress = bpy.props.IntProperty(name="Align Bake Progress", default=0)
    sc.align_bake_cancel = bpy.props.BoolProperty(name="Align Bake Cancel", default=False)

    # Threading options (analytic-only)
    sc.align_use_threading = bpy.props.BoolProperty(
        name="Use threading for analytic bake", default=True,
        description="Offload analytic rotation math to worker threads; bpy calls remain on main thread"
    )
    sc.align_thread_workers = bpy.props.IntProperty(
        name="Worker threads", default=max(1, (os.cpu_count() or 2) - 1), min=1, max=32,
        description="Number of worker threads to use for analytic bake"
    )

    # Graph smooth properties (we'll still allow user to choose "only selected bones")
    sc.smooth_only_selected_bones = bpy.props.BoolProperty(
        name="Only pose-selected bones", default=False,
        description="If true, only bones selected in Pose Mode will be smoothed"
    )

    # Legacy single-field locked axis (for backwards compatibility)
    sc.align_locked_axis = bpy.props.EnumProperty(
        name="Locked Axis",
        items=[
            ('X', "X", "Lock X axis (prevent left/right rotation)"),
            ('Y', "Y", "Lock Y axis (prevent forward/back rotation)"),
            ('Z', "Z", "Lock Z axis (prevent up/down rotation)"),
        ],
        default='X',
        description="Which axis to lock for legacy single-field mode"
    )

    # Quickfix smooth factor
    sc.quickfix_smooth_count = bpy.props.FloatProperty(
        name="Smooth Factor", default=1.0, min=0.0, max=10.0, step=0.1,
        description="Gaussian smooth factor (0 = skip smoothing, 1.0 = full smoothing, 0.5 = half smoothing, etc.)"
    )

    # Manual work smooth factor
    sc.manual_smooth_count = bpy.props.FloatProperty(
        name="Smooth Factor", default=1.0, min=0.0, max=10.0, step=0.1,
        description="Gaussian smooth factor for manual work (0 = skip smoothing, 1.0 = full smoothing, 0.5 = half smoothing, etc.)"
    )

    # Bake debug (console) — one bone name, e.g. Hip_R
    sc.align_debug_bake = bpy.props.BoolProperty(
        name="Debug bake to console", default=True,
        description="Print per-frame rotation info for one parent bone during bake (see System Console)"
    )
    sc.align_debug_bone = bpy.props.StringProperty(
        name="Debug bone", default="Elbow_L",
        description="Parent bone name to log during bake (case-sensitive); ignored when Log all parents is on"
    )
    sc.align_debug_log_all_parents = bpy.props.BoolProperty(
        name="Log all parents", default=False,
        description="Log every parent bone in the triple list each frame (overrides single debug bone)"
    )
    sc.align_debug_spike_deg = bpy.props.FloatProperty(
        name="Spike threshold (deg)", default=45.0, min=0.01, max=180.0, step=0.01,
        description="Log *** SPIKE *** when rotation jumps more than this vs previous baked frame"
    )

    sc.ik_leg_chain_count = bpy.props.IntProperty(
        name="Leg IK chain", default=3, min=0, max=32,
        description="IK chain length on Ankle_L/R (bones counted from the constrained bone)",
    )
    sc.ik_hand_chain_count = bpy.props.IntProperty(
        name="Hand IK chain", default=3, min=0, max=32,
        description="IK chain length on Wrist_L/R (bones counted from the constrained bone)",
    )
    sc.aaf_ik_rotation = bpy.props.BoolProperty(
        name="IK target rotation", default=True,
        description="Match end bone rotation to the Dummy_*_IK target (IK constraint Rotation)",
    )
    sc.aaf_auto_bezier = bpy.props.BoolProperty(
        name="Auto Bezier", default=True,
        description="Automatically set keyframe interpolation to Bezier during addon operations",
    )
    sc.aaf_ik_bone_length_scale = bpy.props.BoolProperty(
        name="IK bone length ×0.005", default=True,
        description="Scale Ankle_L/R and Wrist_L/R bone length ×0.005 when setting up IK",
    )
    sc.aaf_smooth_mode = bpy.props.EnumProperty(
        name="Smooth",
        items=[
            ('ROOT_NLA', "Root NLA", "Smooth Root_M on bottom NLA track"),
            ('WHOLE', "Whole Rig", "Smooth whole animation on the active armature"),
        ],
        default='ROOT_NLA',
        description="Animation smooth to run when setting up IK",
    )
    sc.aaf_smooth_passes = bpy.props.IntProperty(
        name="Smooth passes", default=3, min=0, max=20,
        description="Gaussian smooth passes (×3, ×4, etc.)",
    )

def unregister_props():
    for p in ("show_advanced",
              "align_parent_bone", "align_child_bone", "align_target_bone",
              "align_mode", "align_initial_step", "align_tol", "align_max_iter", "align_bake_method", "align_bake_mode",
              "align_triples", "align_triples_index",
              "align_is_baking", "align_bake_progress", "align_bake_cancel",
              "align_use_threading", "align_thread_workers",
              "smooth_only_selected_bones", "align_locked_axis", "quickfix_smooth_count", "manual_smooth_count",
              "align_debug_bake", "align_debug_bone", "align_debug_log_all_parents", "align_debug_spike_deg",
              "ik_leg_chain_count", "ik_hand_chain_count",
              "aaf_ik_rotation", "aaf_auto_bezier", "aaf_ik_bone_length_scale",
              "aaf_smooth_mode", "aaf_smooth_passes"):
        if hasattr(bpy.types.Scene, p):
            delattr(bpy.types.Scene, p)

# -----------------------------
# Helpers (main-thread safe)
# -----------------------------
def pose_point_world(arm_obj, pb, use_head: bool):
    """Return world-space location of the pose bone head or tail."""
    if use_head:
        return arm_obj.matrix_world @ pb.head
    else:
        return arm_obj.matrix_world @ pb.tail

def bone_rotation_as_quaternion(pb_parent):
    """Current parent rotation as a quaternion regardless of rotation_mode."""
    if pb_parent.rotation_mode == 'QUATERNION':
        return pb_parent.rotation_quaternion.copy()
    return pb_parent.rotation_euler.to_quaternion()

def quat_make_continuous(new_q, ref_q):
    """Pick the quaternion hemisphere closest to ref_q (q and -q are the same rotation)."""
    if new_q.dot(ref_q) < 0.0:
        new_q.negate()

def _debug_bone_enabled(sc, bone_name):
    if sc is None or not getattr(sc, 'align_debug_bake', False):
        return False
    if getattr(sc, 'align_debug_log_all_parents', False):
        return bone_name in _collect_bake_parent_bone_names(sc)
    target = (getattr(sc, 'align_debug_bone', None) or '').strip()
    if not target:
        return False
    return bone_name == target

def quat_delta_deg(q_a, q_b):
    """Angle in degrees between two unit quaternions (same rotation if q and -q)."""
    d = abs(min(1.0, max(-1.0, q_a.dot(q_b))))
    return math.degrees(2.0 * math.acos(d))

def _fmt_euler(pb):
    if pb.rotation_mode == 'QUATERNION':
        e = pb.rotation_quaternion.to_euler('XYZ')
        mode = 'QUAT->XYZ'
    else:
        e = pb.rotation_euler
        mode = pb.rotation_mode
    return mode, (round(e.x, 4), round(e.y, 4), round(e.z, 4))

def align_debug_log_bone(sc, pb_parent, frame, stage, bake_state=None, dist=None, extra=""):
    """Console log for one debug bone during bake. Open Window > Toggle System Console."""
    if not _debug_bone_enabled(sc, pb_parent.name):
        return
    mode, eul = _fmt_euler(pb_parent)
    q = bone_rotation_as_quaternion(pb_parent)
    q_str = tuple(round(v, 4) for v in q)

    delta_deg = None
    quat_flipped = False
    prev = bake_state.get(pb_parent.name) if bake_state else None
    if prev is not None:
        delta_deg = quat_delta_deg(q, prev['quat'])
        quat_flipped = q.dot(prev['quat']) < 0.0

    spike_thr = float(getattr(sc, 'align_debug_spike_deg', 45.0))
    # Only flag spikes on keyed frames (not per-solver-step micro-rotations)
    is_spike = (
        delta_deg is not None
        and delta_deg >= spike_thr
        and stage == 'KEYFRAME'
    )
    tag = "*** SPIKE ***" if is_spike else "ok"

    dist_s = f" dist={dist:.6f}" if dist is not None else ""
    delta_s = f" delta_deg={delta_deg:.2f}" if delta_deg is not None else " delta_deg=n/a"
    flip_s = " quat_hemisphere_flip=True" if quat_flipped else ""
    extra_s = f" {extra}" if extra else ""

    print(
        f"[AAF][{pb_parent.name}] frame={int(frame)} {stage} [{tag}] "
        f"mode={mode} euler={eul} quat={q_str}{delta_s}{flip_s}{dist_s}{extra_s}"
    )

def _aaf_auto_bezier_on(sc):
    return bool(getattr(sc, 'aaf_auto_bezier', True))

def _aaf_ik_rotation_on(sc):
    return bool(getattr(sc, 'aaf_ik_rotation', True))

def _aaf_ik_bone_length_scale_on(sc, arm_obj=None):
    if arm_obj and ik_bones_already_small(arm_obj):
        return False
    return bool(getattr(sc, 'aaf_ik_bone_length_scale', True))

def _aaf_toggle_label(enabled, on_text, off_text):
    return on_text if enabled else off_text

def resolve_bone_on_armature(arm_obj, *candidates):
    """Return first bone name that exists on arm_obj, else first candidate."""
    if not candidates:
        return ""
    if arm_obj and getattr(arm_obj, 'type', None) == 'ARMATURE':
        for name in candidates:
            if name and arm_obj.pose.bones.get(name):
                return name
    return candidates[0]

_LARIAN_IK_LIMBS = (
    ('Left foot',  'leg',  ('Ankle_L',),   ('Dummy_L_Foot_IK',)),
    ('Right foot', 'leg',  ('Ankle_R',),   ('Dummy_R_Foot_IK',)),
    ('Left hand',  'hand', ('Wrist_L',),   ('Dummy_L_Hand_IK',)),
    ('Right hand', 'hand', ('Wrist_R',),   ('Dummy_R_Hand_IK',)),
)

def resolve_larian_ik_limbs(arm_obj):
    """Resolve the 4 Larian end-effector IK pairs (bone, IK target, limb kind)."""
    limbs = []
    for label, kind, bone_cands, target_cands in _LARIAN_IK_LIMBS:
        limbs.append({
            'label': label,
            'kind': kind,
            'bone': resolve_bone_on_armature(arm_obj, *bone_cands),
            'target': resolve_bone_on_armature(arm_obj, *target_cands),
        })
    return limbs

def ensure_ik_constraint(pb_parent, arm_obj, ik_target_name, chain_count, constraint_name, use_rotation=True):
    """Add or update one IK constraint on a pose bone."""
    con = None
    for c in pb_parent.constraints:
        if c.type == 'IK' and c.name == constraint_name:
            con = c
            break
    if con is None:
        con = pb_parent.constraints.new('IK')
        con.name = constraint_name
    con.target = arm_obj
    con.subtarget = ik_target_name
    con.chain_count = max(0, int(chain_count))
    con.influence = 1.0
    con.use_stretch = False
    con.use_rotation = bool(use_rotation)
    return con

def remove_aaf_ik_constraints(arm_obj):
    """Remove prior AAF_IK_* constraints so re-run moves them to the correct bones."""
    for pb in arm_obj.pose.bones:
        for con in list(pb.constraints):
            if con.type == 'IK' and con.name.startswith('AAF_IK_'):
                pb.constraints.remove(con)

def setup_larian_ik_constraints_on_armature(arm_obj, leg_chain_count=3, hand_chain_count=3, use_rotation=True):
    """
    Create/update 4 IK constraints on Ankle_L/R and Wrist_L/R targeting Dummy_*_IK bones.
    Returns (created_or_updated_names, error_messages).
    """
    if not arm_obj or arm_obj.type != 'ARMATURE':
        return [], ["No armature"]
    remove_aaf_ik_constraints(arm_obj)
    ok = []
    errors = []
    for spec in resolve_larian_ik_limbs(arm_obj):
        label = spec['label']
        bone = spec['bone']
        target = spec['target']
        chain_count = hand_chain_count if spec['kind'] == 'hand' else leg_chain_count
        if not bone or not arm_obj.pose.bones.get(bone):
            errors.append(f"{label}: missing bone {bone or '?'}")
            continue
        if not target or not arm_obj.pose.bones.get(target):
            errors.append(f"{label}: missing IK target {target or '?'}")
            continue
        pb = arm_obj.pose.bones.get(bone)
        con_name = f"AAF_IK_{target}"
        ensure_ik_constraint(pb, arm_obj, target, chain_count, con_name, use_rotation=use_rotation)
        ok.append(f"{label} ({bone}→{target}, chain={chain_count})")
        rot_s = "rot=ON" if use_rotation else "rot=OFF"
        print(f"[AAF] IK {label}: {bone} -> {target} chain_count={chain_count} {rot_s}")
    return ok, errors

_IK_BONE_LENGTH_SCALE = 0.005
_IK_BONE_ALREADY_SMALL_MAX = 0.02

def ik_bones_already_small(arm_obj):
    """True when ankle/wrist bones are already at scaled (×0.005) length."""
    if not arm_obj or arm_obj.type != 'ARMATURE' or not arm_obj.data:
        return False
    checked = 0
    for spec in resolve_larian_ik_limbs(arm_obj):
        name = spec['bone']
        bone = arm_obj.data.bones.get(name) if name else None
        if not bone:
            continue
        checked += 1
        if bone.length > _IK_BONE_ALREADY_SMALL_MAX:
            return False
    return checked > 0

def scale_larian_ik_bone_lengths(arm_obj, factor=_IK_BONE_LENGTH_SCALE):
    """Scale Ankle_L/R and Wrist_L/R edit-bone lengths by factor (default 0.005)."""
    if not arm_obj or arm_obj.type != 'ARMATURE':
        return [], ["No armature"]
    bone_names = []
    for spec in resolve_larian_ik_limbs(arm_obj):
        name = spec['bone']
        if name and arm_obj.data.bones.get(name):
            bone_names.append(name)
    if not bone_names:
        return [], ["No IK bones found (Ankle_L/R, Wrist_L/R)"]

    prev_mode = arm_obj.mode
    prev_active = bpy.context.view_layer.objects.active
    errors = []
    ok = []
    try:
        bpy.context.view_layer.objects.active = arm_obj
        if arm_obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        eb = arm_obj.data.edit_bones
        for name in bone_names:
            if name not in eb:
                errors.append(f"{name}: missing in edit bones")
                continue
            b = eb[name]
            old_len = b.length
            b.length = max(1e-8, old_len * factor)
            ok.append(name)
            print(f"[AAF] IK bone length {name}: {old_len:.6f} -> {b.length:.6f} (×{factor})")
    except Exception as e:
        errors.append(str(e))
    finally:
        try:
            bpy.context.view_layer.objects.active = arm_obj
            if prev_mode == 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            elif prev_mode == 'POSE':
                bpy.ops.object.mode_set(mode='POSE')
            else:
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        try:
            if prev_active:
                bpy.context.view_layer.objects.active = prev_active
        except Exception:
            pass
    return ok, errors

_AAF_ROOT_SMOOTH_BONE = "Root_M"
_AAF_SMOOTH_FACTOR = 1.0

def _aaf_find_window_region(area):
    for region in area.regions:
        if region.type == 'WINDOW':
            return region
    return None

def _aaf_find_area(screen, area_type):
    for area in screen.areas:
        if area.type == area_type:
            return area
    return None

def _aaf_object_override(ctx, arm_obj):
    return {
        'window': ctx.window,
        'screen': ctx.screen,
        'active_object': arm_obj,
        'object': arm_obj,
        'selected_objects': [arm_obj],
        'selected_editable_objects': [arm_obj],
    }

def _aaf_run_in_editor(ctx, arm_obj, area_type, operator_name, **operator_kwargs):
    screen = ctx.screen
    area = _aaf_find_area(screen, area_type)
    restored_type = None

    if area is None:
        if not screen.areas:
            return False
        area = screen.areas[0]
        restored_type = area.type
        area.type = area_type

    region = _aaf_find_window_region(area)
    if region is None:
        if restored_type is not None:
            area.type = restored_type
        return False

    override = {**_aaf_object_override(ctx, arm_obj), 'area': area, 'region': region}
    if area_type == 'GRAPH_EDITOR':
        override['space_data'] = area.spaces.active

    try:
        with ctx.temp_override(**override):
            op = getattr(bpy.ops, operator_name.split('.')[0])
            op_fn = getattr(op, operator_name.split('.')[1])
            if not op_fn.poll():
                return False
            op_fn(**operator_kwargs)
        return True
    finally:
        if restored_type is not None:
            try:
                area.type = restored_type
            except Exception:
                pass

def get_bottom_nla_strip(arm_obj):
    """Return (action, track_name, strip) for the bottom-most NLA track."""
    ad = arm_obj.animation_data
    if ad is None or len(ad.nla_tracks) == 0:
        return None, None, None

    bottom_track = ad.nla_tracks[0]
    for strip in bottom_track.strips:
        if strip.action is not None:
            return strip.action, bottom_track.name, strip
    return None, bottom_track.name, None

def _aaf_find_active_nla_strip(ad):
    for track in ad.nla_tracks:
        for strip in track.strips:
            if strip.active:
                return strip
    return None

def _aaf_find_track_for_strip(ad, strip):
    for track in ad.nla_tracks:
        for s in track.strips:
            if s == strip:
                return track
    return None

def _aaf_save_strip_selection(ad):
    saved = []
    for track in ad.nla_tracks:
        for strip in track.strips:
            saved.append((strip, strip.select))
    return saved

def _aaf_restore_strip_selection(saved):
    for strip, selected in saved:
        try:
            strip.select = selected
        except Exception:
            pass

def _aaf_select_nla_strip(ad, strip):
    track = _aaf_find_track_for_strip(ad, strip)
    if track is None:
        return
    ad.nla_tracks.active = track
    for t in ad.nla_tracks:
        for s in t.strips:
            s.select = (s == strip)

@contextmanager
def nla_tweak_strip(ctx, arm_obj, strip):
    """Enter NLA tweak mode on strip so its action is editable, then restore."""
    ad = arm_obj.animation_data
    if ad is None:
        raise RuntimeError("Armature has no animation data.")

    if arm_obj.name not in ctx.view_layer.objects:
        raise RuntimeError(f"Armature '{arm_obj.name}' is not in the current view layer.")

    saved = {
        'action': ad.action,
        'use_tweak_mode': ad.use_tweak_mode,
        'active_strip': _aaf_find_active_nla_strip(ad),
        'active_track': ad.nla_tracks.active,
        'strip_select': _aaf_save_strip_selection(ad),
        'active': ctx.view_layer.objects.active,
        'mode': ctx.mode,
    }
    entered_tweak = False

    try:
        arm_obj.select_set(True)
        ctx.view_layer.objects.active = arm_obj

        active_strip = _aaf_find_active_nla_strip(ad)
        if ad.use_tweak_mode and active_strip is not None and active_strip != strip:
            _aaf_run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_exit')

        _aaf_select_nla_strip(ad, strip)

        if ad.use_tweak_mode and _aaf_find_active_nla_strip(ad) == strip:
            entered_tweak = False
        elif _aaf_run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_enter'):
            entered_tweak = True
        else:
            ad.use_tweak_mode = True
            if strip.action:
                ad.action = strip.action
            entered_tweak = True

        action = ad.action or strip.action
        if action is None:
            raise RuntimeError("NLA strip has no action to edit.")

        yield action

    finally:
        if entered_tweak and ad.use_tweak_mode:
            _aaf_run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_exit')

        if saved['use_tweak_mode'] and saved['active_strip'] is not None:
            _aaf_select_nla_strip(ad, saved['active_strip'])
            if not ad.use_tweak_mode:
                _aaf_run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_enter')
            _aaf_restore_strip_selection(saved['strip_select'])
        else:
            try:
                ad.action = saved['action']
                ad.nla_tracks.active = saved['active_track']
            except Exception:
                pass
            _aaf_restore_strip_selection(saved['strip_select'])

        if saved['active'] and saved['active'].name in ctx.view_layer.objects:
            try:
                ctx.view_layer.objects.active = saved['active']
            except Exception:
                pass

def _aaf_action_has_usable_fcurves(action, bone_names=None):
    if not action or not action.fcurves:
        return False
    bone_filter = set(bone_names) if bone_names else None
    for fcu in action.fcurves:
        if bone_filter is not None:
            if not fcu.data_path.startswith('pose.bones'):
                continue
            dp = fcu.data_path
            start = dp.find('["')
            end = dp.find('"]', start + 1)
            if start == -1 or end == -1:
                start = dp.find("['")
                end = dp.find("']", start + 1)
                if start == -1 or end == -1:
                    continue
            bone = dp[start + 2:end]
            if bone not in bone_filter:
                continue
        return True
    return False

def resolve_whole_smooth_source(arm):
    """Return (action, nla_strip, track_name) for whole-rig smoothing."""
    ad = arm.animation_data
    if ad is None:
        return None, None, None

    # Larian rigs usually animate on NLA; prefer bottom strip when tracks exist.
    if len(ad.nla_tracks) > 0:
        action, track_name, strip = get_bottom_nla_strip(arm)
        if strip and action and _aaf_action_has_usable_fcurves(action):
            return action, strip, track_name
        for track in ad.nla_tracks:
            for s in track.strips:
                if s.action and _aaf_action_has_usable_fcurves(s.action):
                    return s.action, s, track.name

    if ad.action and _aaf_action_has_usable_fcurves(ad.action):
        return ad.action, None, None

    return None, None, None

def smooth_root_nla_bottom(context, arm, passes):
    if passes <= 0:
        raise RuntimeError("NLA smooth passes must be at least 1.")

    if arm.pose.bones.get(_AAF_ROOT_SMOOTH_BONE) is None:
        raise RuntimeError(f"Bone '{_AAF_ROOT_SMOOTH_BONE}' not found on '{arm.name}'.")

    action, track_name, strip = get_bottom_nla_strip(arm)
    if strip is None or action is None:
        msg = f"No action on bottom NLA track for '{arm.name}'."
        if track_name:
            msg = f"No action on bottom NLA track '{track_name}' ({arm.name})."
        raise RuntimeError(msg)

    with nla_tweak_strip(context, arm, strip) as tweak_action:
        total = 0
        for _ in range(passes):
            total = apply_graph_gaussian_smooth_for_armature_operator(
                arm,
                factor=_AAF_SMOOTH_FACTOR,
                action=tweak_action,
                bone_names={_AAF_ROOT_SMOOTH_BONE},
                verbose=True,
            )

    if total == 0:
        raise RuntimeError(
            f"No F-curves found for '{_AAF_ROOT_SMOOTH_BONE}' in '{action.name}' ({arm.name})."
        )

    return total, track_name, action.name

def smooth_whole_armature_action(context, arm, passes):
    if passes <= 0:
        raise RuntimeError("Whole animation smooth passes must be at least 1.")

    action, strip, track_name = resolve_whole_smooth_source(arm)
    if action is None:
        raise RuntimeError(f"No animation data on '{arm.name}'.")

    if strip:
        with nla_tweak_strip(context, arm, strip) as tweak_action:
            total = 0
            for _ in range(passes):
                total = apply_graph_gaussian_smooth_for_armature_operator(
                    arm,
                    factor=_AAF_SMOOTH_FACTOR,
                    action=tweak_action,
                    smooth_all_fcurves=True,
                    verbose=True,
                )
        source_label = f"NLA '{track_name}'" if track_name else "NLA"
    else:
        total = 0
        for _ in range(passes):
            total = apply_graph_gaussian_smooth_for_armature_operator(
                arm,
                factor=_AAF_SMOOTH_FACTOR,
                action=action,
                smooth_all_fcurves=True,
                verbose=True,
            )
        source_label = action.name

    if total == 0:
        raise RuntimeError(f"No bone F-curves found for '{arm.name}' ({source_label}).")

    return total, action.name

def apply_aaf_ik_smooth_extras(context, arm_obj, sc):
    warnings = []
    mode = getattr(sc, 'aaf_smooth_mode', 'ROOT_NLA')
    passes = int(getattr(sc, 'aaf_smooth_passes', 3))
    if passes <= 0:
        return warnings
    try:
        if mode == 'ROOT_NLA':
            curves, track_name, action_name = smooth_root_nla_bottom(context, arm_obj, passes)
            print(
                f"[AAF] Root NLA smooth {arm_obj.name}: {_AAF_ROOT_SMOOTH_BONE} x{passes} "
                f"(NLA '{track_name}', {curves} curves, {action_name})"
            )
        elif mode == 'WHOLE':
            curves, action_name = smooth_whole_armature_action(context, arm_obj, passes)
            print(
                f"[AAF] Whole rig smooth {arm_obj.name}: x{passes} "
                f"({action_name}, {curves} curves)"
            )
    except Exception as e:
        label = "Root NLA smooth" if mode == 'ROOT_NLA' else "Whole rig smooth"
        warnings.append(f"{label}: {e}")
        print(f"[AAF] {label} failed: {e}")
    return warnings

def _upgrade_hand_parent_to_elbow(arm_obj, tri, triple_index=None):
    """Wrist targets are reached by elbow flex, not whole-shoulder swings."""
    if tri.child_bone not in ('Wrist_L', 'Wrist_R'):
        return
    shoulder_names = ('Should_L', 'Should_R', 'Shoulder_L', 'Shoulder_R')
    if tri.parent_bone not in shoulder_names:
        return
    side = 'L' if tri.child_bone.endswith('_L') else 'R'
    elbow = resolve_bone_on_armature(arm_obj, f'Elbow_{side}')
    if not elbow or tri.parent_bone == elbow:
        return
    old = tri.parent_bone
    tri.parent_bone = elbow
    tri.locked_axis = 'X'
    label = f"triple {triple_index + 1}" if triple_index is not None else "triple"
    print(f"[AAF] {label}: hand parent '{old}' -> '{elbow}' (elbow-driven IK)")

# Common Larian / BG3 naming variants (short vs full shoulder names, etc.)
_LARIAN_BONE_ALIASES = {
    'Should_L': ('Shoulder_L', 'Should_L'),
    'Should_R': ('Shoulder_R', 'Should_R'),
    'Shoulder_L': ('Shoulder_L', 'Should_L'),
    'Shoulder_R': ('Shoulder_R', 'Should_R'),
}

def autoresolve_align_triples(arm_obj, sc):
    """Fix triple bone names using known aliases when preset names don't match the rig."""
    if not arm_obj or arm_obj.type != 'ARMATURE':
        return
    if len(sc.align_triples) > 0:
        for i, tri in enumerate(sc.align_triples):
            for field in ('parent_bone', 'child_bone', 'target_bone'):
                cur = getattr(tri, field, '') or ''
                if cur and arm_obj.pose.bones.get(cur):
                    continue
                cands = _LARIAN_BONE_ALIASES.get(cur)
                if not cands and field == 'parent_bone':
                    if tri.child_bone == 'Wrist_L':
                        cands = ('Shoulder_L', 'Should_L')
                    elif tri.child_bone == 'Wrist_R':
                        cands = ('Shoulder_R', 'Should_R')
                if cands:
                    resolved = resolve_bone_on_armature(arm_obj, *cands)
                    if resolved and resolved != cur:
                        setattr(tri, field, resolved)
                        print(f"[AAF] triple {i + 1}: resolved {field} '{cur}' -> '{resolved}'")
            _upgrade_hand_parent_to_elbow(arm_obj, tri, i)
    else:
        for prop in ('align_parent_bone', 'align_child_bone', 'align_target_bone'):
            cur = getattr(sc, prop, '') or ''
            if cur and arm_obj.pose.bones.get(cur):
                continue
            cands = _LARIAN_BONE_ALIASES.get(cur)
            if cands:
                resolved = resolve_bone_on_armature(arm_obj, *cands)
                if resolved and resolved != cur:
                    setattr(sc, prop, resolved)
                    print(f"[AAF] legacy: resolved {prop} '{cur}' -> '{resolved}'")

def _collect_bake_parent_bone_names(sc):
    names = set()
    if len(sc.align_triples) > 0:
        for tri in sc.align_triples:
            if tri.parent_bone:
                names.add(tri.parent_bone)
    elif sc.align_parent_bone:
        names.add(sc.align_parent_bone)
    return names

def validate_align_triples(arm_obj, sc):
    """Print per-triple bone resolution status. Returns (valid_count, total_count)."""
    valid = 0
    total = 0
    if len(sc.align_triples) > 0:
        for i, tri in enumerate(sc.align_triples):
            total += 1
            p_name, c_name, t_name = tri.parent_bone, tri.child_bone, tri.target_bone
            p_ok = bool(p_name and arm_obj.pose.bones.get(p_name))
            c_ok = bool(c_name and arm_obj.pose.bones.get(c_name))
            t_ok = bool(t_name and arm_obj.pose.bones.get(t_name))
            if p_ok and c_ok and t_ok:
                valid += 1
                print(
                    f"[AAF] triple {i + 1}: {p_name} -> {c_name} -> {t_name}  OK"
                )
            else:
                parts = []
                parts.append(f"{p_name or '?'} {'OK' if p_ok else 'MISSING'}")
                parts.append(f"{c_name or '?'} {'OK' if c_ok else 'MISSING'}")
                parts.append(f"{t_name or '?'} {'OK' if t_ok else 'MISSING'}")
                print(f"[AAF] triple {i + 1}: {' / '.join(parts)}  SKIPPED")
    else:
        total = 1
        p_name = sc.align_parent_bone
        c_name = sc.align_child_bone
        t_name = sc.align_target_bone
        p_ok = bool(p_name and arm_obj.pose.bones.get(p_name))
        c_ok = bool(c_name and arm_obj.pose.bones.get(c_name))
        t_ok = bool(t_name and arm_obj.pose.bones.get(t_name))
        if p_ok and c_ok and t_ok:
            valid = 1
            print(f"[AAF] legacy: {p_name} -> {c_name} -> {t_name}  OK")
        else:
            print(f"[AAF] legacy: parent={p_name} child={c_name} target={t_name}  SKIPPED")
    if total > 0 and valid == 0:
        print("[AAF] ERROR: no valid parent/child/target triples — bake would do nothing.")
    return valid, total

def ensure_armature_has_action(arm_obj):
    """Ensure armature has animation_data and an action (required for keyframe_insert)."""
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    if arm_obj.animation_data.action is None:
        action = bpy.data.actions.new(name=f"{arm_obj.name}_Action")
        arm_obj.animation_data.action = action
        print(f"[AAF] created action '{action.name}' on armature '{arm_obj.name}'")
    return arm_obj.animation_data.action

def apply_bezier_interpolation_to_armature(arm_obj, verbose=False):
    """Set all keyframes on the armature action to Bezier interpolation."""
    if not arm_obj or arm_obj.type != 'ARMATURE':
        return 0
    action = ensure_armature_has_action(arm_obj)
    if not action:
        return 0
    changed = 0
    for fcu in action.fcurves:
        updated = False
        for kp in fcu.keyframe_points:
            if kp.interpolation != 'BEZIER':
                kp.interpolation = 'BEZIER'
                changed += 1
                updated = True
        if updated:
            fcu.update()
    if verbose and changed:
        print(f"[AAF] Make Bezier: {changed} key(s) on '{arm_obj.name}' action '{action.name}'")
    return changed

def apply_bezier_if_enabled(arm_obj, sc, verbose=False):
    if sc is not None and _aaf_auto_bezier_on(sc):
        return apply_bezier_interpolation_to_armature(arm_obj, verbose=verbose)
    return 0

def set_bone_rotation_keyframes_bezier_at_frame(arm_obj, bone_name, frame):
    """Set Bezier interpolation on rotation keys for one bone at one frame."""
    if arm_obj is None or not getattr(arm_obj, 'animation_data', None):
        return
    action = arm_obj.animation_data.action
    if not action:
        return
    prefix = f'pose.bones["{bone_name}"].'
    frame_f = float(frame)
    for fcu in action.fcurves:
        if not fcu.data_path.startswith(prefix):
            continue
        if 'rotation_euler' not in fcu.data_path and 'rotation_quaternion' not in fcu.data_path:
            continue
        updated = False
        for kp in fcu.keyframe_points:
            if abs(kp.co[0] - frame_f) < 1e-4:
                kp.interpolation = 'BEZIER'
                updated = True
        if updated:
            fcu.update()

def suggest_debug_bone(arm_obj, sc):
    """Set align_debug_bone to a valid parent name if current value is missing or not a bake parent."""
    parent_names = sorted(_collect_bake_parent_bone_names(sc))
    if not parent_names:
        return
    dbg = (getattr(sc, 'align_debug_bone', None) or '').strip()
    if dbg and dbg in parent_names and arm_obj.pose.bones.get(dbg):
        return
    for preferred in ('Elbow_L', 'Elbow_R', 'Shoulder_L', 'Shoulder_R', 'Hip_L', 'Hip_R'):
        if preferred in parent_names and arm_obj.pose.bones.get(preferred):
            sc.align_debug_bone = preferred
            print(f"[AAF] debug bone auto-set to '{preferred}'")
            return
    sc.align_debug_bone = parent_names[0]
    print(f"[AAF] debug bone auto-set to '{parent_names[0]}'")

def strip_conflicting_parent_rotation_fcurves(arm_obj, parent_bone_names):
    """Remove euler f-curves on quaternion bones (and vice versa) so only one channel drives rotation."""
    if not arm_obj.animation_data or not arm_obj.animation_data.action:
        return 0
    action = arm_obj.animation_data.action
    to_remove = []
    for fcu in action.fcurves:
        bone = _extract_bone_name_from_path(fcu.data_path)
        if bone not in parent_bone_names:
            continue
        pb = arm_obj.pose.bones.get(bone)
        if not pb:
            continue
        if pb.rotation_mode == 'QUATERNION' and 'rotation_euler' in fcu.data_path:
            to_remove.append(fcu)
        elif pb.rotation_mode != 'QUATERNION' and 'rotation_quaternion' in fcu.data_path:
            to_remove.append(fcu)
    for fcu in to_remove:
        action.fcurves.remove(fcu)
    return len(to_remove)

def remove_subframe_rotation_keys(action, bone_name, frame_start, frame_end):
    """Delete rotation keys that sit between integer frames (common source of in-between pops)."""
    removed = 0
    for fcu in list(action.fcurves):
        if _extract_bone_name_from_path(fcu.data_path) != bone_name:
            continue
        if 'rotation_euler' not in fcu.data_path and 'rotation_quaternion' not in fcu.data_path:
            continue
        fcu_removed = False
        while True:
            victim = None
            for kp in fcu.keyframe_points:
                fr = float(kp.co[0])
                if frame_start <= fr <= frame_end and abs(fr - round(fr)) > 1e-4:
                    victim = kp
                    break
            if victim is None:
                break
            try:
                fcu.keyframe_points.remove(victim, fast=True)
                removed += 1
                fcu_removed = True
            except RuntimeError:
                break
        if fcu_removed:
            try:
                fcu.update()
            except Exception:
                pass
    return removed

def prepare_bake_parent_bones(arm_obj, sc, frame_start, frame_end, verbose=False):
    parent_names = _collect_bake_parent_bone_names(sc)
    if not parent_names:
        return
    removed_fcurves = strip_conflicting_parent_rotation_fcurves(arm_obj, parent_names)
    removed_keys = 0
    action = arm_obj.animation_data.action if arm_obj.animation_data else None
    if action:
        for name in parent_names:
            removed_keys += remove_subframe_rotation_keys(action, name, frame_start, frame_end)
    if verbose:
        print(
            f"[AAF] bake prep parents={sorted(parent_names)} "
            f"removed_fcurves={removed_fcurves} removed_subframe_keys={removed_keys}"
        )

def apply_baked_rotation_to_pose(pb_parent, entry):
    """Apply a bake_state entry to the pose bone without re-deriving euler/quat."""
    mode = entry['mode']
    pb_parent.rotation_mode = mode
    if mode == 'QUATERNION':
        pb_parent.rotation_quaternion = entry['quat'].copy()
    elif entry.get('euler') is not None:
        pb_parent.rotation_euler = entry['euler'].copy()
    else:
        pb_parent.rotation_quaternion = entry['quat'].copy()

def seed_parent_rotation_from_bake(pb_parent, bake_state):
    """Use last baked rotation instead of action interpolation (fixes BEFORE_SOLVE spikes)."""
    prev = bake_state.get(pb_parent.name)
    if prev is None:
        return False
    apply_baked_rotation_to_pose(pb_parent, prev)
    return True

def seed_all_bake_parents(arm_obj, sc, bake_state, frame, deps):
    for name in _collect_bake_parent_bone_names(sc):
        pb = arm_obj.pose.bones.get(name)
        if not pb:
            continue
        if seed_parent_rotation_from_bake(pb, bake_state):
            if _debug_bone_enabled(sc, name):
                align_debug_log_bone(sc, pb, frame, "SEEDED", bake_state, extra="from_prev_bake")
    deps.update()

def euler_unwrap_continuous(new_euler, prev_euler, order='XYZ'):
    """Shift euler angles by +/- 2pi so they stay close to the previous frame."""
    out = [float(new_euler[i]) for i in range(3)]
    prev = [float(prev_euler[i]) for i in range(3)]
    for i in range(3):
        while out[i] - prev[i] > math.pi:
            out[i] -= 2.0 * math.pi
        while out[i] - prev[i] < -math.pi:
            out[i] += 2.0 * math.pi
    return Euler(out, order)

def bake_keyframe_parent_rotation(pb_parent, frame, bake_state, rotation_mode_override=None, keyframe_counts=None):
    """
    Keyframe parent rotation with frame-to-frame continuity to avoid 180-degree flips.
    bake_state: dict bone_name -> {'quat': Quaternion, 'euler': Euler|None, 'mode': str}
    rotation_mode_override: use when pose was applied in QUATERNION mode but bone normally uses euler.
    keyframe_counts: optional dict to increment per-bone keyframe insert count.
    Returns True if keyframe_insert succeeded.
    """
    bone_name = pb_parent.name
    prev_mode = rotation_mode_override or pb_parent.rotation_mode
    if bone_name in bake_state and rotation_mode_override is None:
        prev_mode = bake_state[bone_name]['mode']
    new_q = bone_rotation_as_quaternion(pb_parent)

    prev = bake_state.get(bone_name)
    quat_flipped_before = False
    if prev is not None:
        quat_flipped_before = new_q.dot(prev['quat']) < 0.0
        quat_make_continuous(new_q, prev['quat'])

    entry = {'quat': new_q.copy(), 'euler': None, 'mode': prev_mode}

    if prev_mode == 'QUATERNION':
        pb_parent.rotation_mode = 'QUATERNION'
        pb_parent.rotation_quaternion = new_q
    else:
        try:
            if prev is not None and prev.get('euler') is not None:
                eul = euler_unwrap_continuous(
                    new_q.to_euler(prev_mode), prev['euler'], prev_mode
                )
            else:
                eul = new_q.to_euler(prev_mode)
            entry['euler'] = eul.copy()
            pb_parent.rotation_euler = eul
            pb_parent.rotation_mode = prev_mode
            entry['quat'] = eul.to_quaternion().copy()
        except Exception:
            pb_parent.rotation_mode = 'QUATERNION'
            pb_parent.rotation_quaternion = new_q

    sc = bpy.context.scene
    if _debug_bone_enabled(sc, bone_name):
        extra_parts = []
        if quat_flipped_before:
            extra_parts.append("hemisphere_corrected=True")
        if entry.get('euler') is not None:
            e = entry['euler']
            extra_parts.append(f"keyed_euler={tuple(round(v, 4) for v in e)}")
        align_debug_log_bone(
            sc, pb_parent, frame, "KEYFRAME", bake_state, extra=" ".join(extra_parts)
        )

    bake_state[bone_name] = entry

    try:
        if prev_mode == 'QUATERNION':
            pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        elif entry.get('euler') is not None:
            pb_parent.keyframe_insert(data_path="rotation_euler", frame=frame)
        else:
            pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if keyframe_counts is not None:
            keyframe_counts[bone_name] = keyframe_counts.get(bone_name, 0) + 1
        if _aaf_auto_bezier_on(sc):
            set_bone_rotation_keyframes_bezier_at_frame(pb_parent.id_data, bone_name, frame)
        return True
    except Exception as e:
        print(f"[AAF] keyframe failed {bone_name} frame={int(frame)}: {e}")
        return False

def bake_hold_keyframe_parent_rotation(pb_parent, frame, bake_state, keyframe_counts=None):
    """
    Keyframe the exact previous baked rotation (no re-derive drift on HOLD frames).
    Keeps F-curves continuous without baking solver micro-jitter.
    """
    bone_name = pb_parent.name
    prev = bake_state.get(bone_name)
    if prev is None:
        return bake_keyframe_parent_rotation(
            pb_parent, frame, bake_state, keyframe_counts=keyframe_counts,
        )

    apply_baked_rotation_to_pose(pb_parent, prev)

    sc = bpy.context.scene
    if _debug_bone_enabled(sc, bone_name):
        align_debug_log_bone(
            sc, pb_parent, frame, "KEYFRAME", bake_state, extra="HOLD(exact_prev)",
        )

    try:
        if prev['mode'] == 'QUATERNION':
            pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        elif prev.get('euler') is not None:
            pb_parent.keyframe_insert(data_path="rotation_euler", frame=frame)
        else:
            pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if keyframe_counts is not None:
            keyframe_counts[bone_name] = keyframe_counts.get(bone_name, 0) + 1
        if _aaf_auto_bezier_on(sc):
            set_bone_rotation_keyframes_bezier_at_frame(pb_parent.id_data, bone_name, frame)
        return True
    except Exception as e:
        print(f"[AAF] hold keyframe failed {bone_name} frame={int(frame)}: {e}")
        return False

# -----------------------------
# Pure-Python vector/quaternion utilities for worker threads
# -----------------------------
def v_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def v_len(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def v_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def v_cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def v_normalize(a):
    L = v_len(a)
    if L < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0]/L, a[1]/L, a[2]/L)

def quat_normalize(q):
    w,x,y,z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)

def quat_mul(a, b):
    aw,ax,ay,az = a
    bw,bx,by,bz = b
    return (aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw)

def quat_inv(q):
    w,x,y,z = q
    return (w, -x, -y, -z)

def rotation_between_unit_vectors(u, v):
    d = v_dot(u, v)
    if d > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    if d < -0.999999:
        axis = (1.0, 0.0, 0.0)
        if abs(u[0]) > 0.9:
            axis = (0.0, 1.0, 0.0)
        ax = v_cross(u, axis)
        axn = v_normalize(ax)
        return quat_normalize((0.0, axn[0], axn[1], axn[2]))
    cross = v_cross(u, v)
    q = (1.0 + d, cross[0], cross[1], cross[2])
    return quat_normalize(q)

# Worker analytic compute (runs in background thread) — computes new parent quaternion for one triple
def worker_analytic_compute(snapshot):
    pivot = snapshot['pivot']
    child = snapshot['child']
    target = snapshot['target']
    arm_wq = snapshot['arm_world_quat']  # (w,x,y,z)
    parent_q = snapshot['parent_quat']   # (w,x,y,z)

    cur = (child[0]-pivot[0], child[1]-pivot[1], child[2]-pivot[2])
    dst = (target[0]-pivot[0], target[1]-pivot[1], target[2]-pivot[2])

    if v_len(cur) < 1e-12 or v_len(dst) < 1e-12:
        return parent_q

    cur_n = v_normalize(cur)
    dst_n = v_normalize(dst)
    rot_world = rotation_between_unit_vectors(cur_n, dst_n)

    arm_inv = quat_inv(arm_wq)
    rot_local = quat_mul(quat_mul(arm_inv, rot_world), arm_wq)

    new_parent = quat_mul(rot_local, parent_q)
    return quat_normalize(new_parent)

# -----------------------------
# Core algorithmic helpers
# -----------------------------
def analytic_rotate_core(arm_obj, pb_parent, pb_child, pb_target, mode):
    """Rotate parent to align child to target."""
    pivot = arm_obj.matrix_world @ pb_parent.head
    child_use_head = mode.startswith('HEAD_')
    target_use_head = mode.endswith('_HEAD')

    cur = pose_point_world(arm_obj, pb_child, child_use_head) - pivot
    dst = pose_point_world(arm_obj, pb_target, target_use_head) - pivot

    if cur.length < 1e-9 or dst.length < 1e-9:
        return False

    rot_world = cur.normalized().rotation_difference(dst.normalized())
    arm_world_quat = arm_obj.matrix_world.to_quaternion()
    rot_local = arm_world_quat.inverted() @ rot_world @ arm_world_quat

    prev_mode = pb_parent.rotation_mode
    pb_parent.rotation_mode = 'QUATERNION'
    pb_parent.rotation_quaternion = rot_local @ pb_parent.rotation_quaternion

    if prev_mode != 'QUATERNION':
        pb_parent.rotation_euler = pb_parent.rotation_quaternion.to_euler(prev_mode)
        pb_parent.rotation_mode = prev_mode

    return True

def iterative_minimize_core(arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char='X'):
    """Iterative minimization solver.
    locked_axis_char: 'X', 'Y', or 'Z' - which axis to lock (default 'X')"""
    lock_rot = getattr(pb_parent, "lock_rotation", (False, False, False))
    
    # Convert axis char to index: X=0, Y=1, Z=2
    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    locked_axis_idx = axis_map.get(locked_axis_char, 0)  # Default to X if invalid

    prev_mode = pb_parent.rotation_mode
    if prev_mode == 'QUATERNION':
        cur_euler = pb_parent.rotation_quaternion.to_euler('XYZ')
    else:
        cur_euler = pb_parent.rotation_euler.to_euler('XYZ')

    pb_parent.rotation_mode = 'XYZ'
    best_angles = [float(a) for a in cur_euler]
    pb_parent.rotation_euler = best_angles
    deps.update()
    last_applied = best_angles.copy()

    mode = sc.align_mode
    child_use_head = mode.startswith('HEAD_')
    target_use_head = mode.endswith('_HEAD')

    def current_distance():
        p = pose_point_world(arm_obj, pb_child, child_use_head)
        t = pose_point_world(arm_obj, pb_target, target_use_head)
        return (p - t).length

    best_dist = current_distance()

    step = float(sc.align_initial_step)
    tol = float(sc.align_tol)
    max_iter = int(sc.align_max_iter)

    iter_count = 0

    while step > tol and iter_count < max_iter:
        iter_count += 1
        improved = False

        for axis in range(3):
            if lock_rot[axis] or axis == locked_axis_idx:  # Skip locked axis and bone-locked axes
                continue

            for direction in (1.0, -1.0):
                candidate = best_angles.copy()
                candidate[axis] += direction * step

                if candidate != last_applied:
                    pb_parent.rotation_euler = candidate
                    deps.update()
                    last_applied = candidate.copy()

                dist = current_distance()
                if dist + 1e-12 < best_dist:
                    best_dist = dist
                    best_angles = candidate.copy()
                    improved = True

        if last_applied != best_angles:
            pb_parent.rotation_euler = best_angles
            deps.update()
            last_applied = best_angles.copy()

        if best_dist <= 1e-8:
            break

        if not improved:
            step *= 0.5

    pb_parent.rotation_euler = best_angles
    deps.update()
    final_dist = current_distance()

    if prev_mode == 'QUATERNION':
        q = pb_parent.rotation_euler.to_quaternion()
        pb_parent.rotation_mode = 'QUATERNION'
        pb_parent.rotation_quaternion = q
    else:
        try:
            pb_parent.rotation_euler = Euler(best_angles, 'XYZ').to_euler(prev_mode)
            pb_parent.rotation_mode = prev_mode
        except Exception:
            pb_parent.rotation_mode = 'XYZ'

    return final_dist, iter_count

def iterative_single_step_core(arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char='X'):
    """Do just one iteration step of the iterative solver (for testing).
    If no improvement with current step, tries smaller steps until improvement or minimum step reached.
    locked_axis_char: 'X', 'Y', or 'Z' - which axis to lock (default 'X')"""
    lock_rot = getattr(pb_parent, "lock_rotation", (False, False, False))
    
    # Convert axis char to index: X=0, Y=1, Z=2
    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    locked_axis_idx = axis_map.get(locked_axis_char, 0)  # Default to X if invalid

    prev_mode = pb_parent.rotation_mode
    if prev_mode == 'QUATERNION':
        cur_euler = pb_parent.rotation_quaternion.to_euler('XYZ')
        orig_parent_q = pb_parent.rotation_quaternion.copy()
    else:
        cur_euler = pb_parent.rotation_euler.to_euler('XYZ')
        orig_parent_q = pb_parent.rotation_euler.to_quaternion()

    pb_parent.rotation_mode = 'XYZ'
    best_angles = [float(a) for a in cur_euler]
    pb_parent.rotation_euler = best_angles
    deps.update()

    mode = sc.align_mode
    child_use_head = mode.startswith('HEAD_')
    target_use_head = mode.endswith('_HEAD')

    def current_distance():
        p = pose_point_world(arm_obj, pb_child, child_use_head)
        t = pose_point_world(arm_obj, pb_target, target_use_head)
        return (p - t).length

    best_dist = current_distance()
    initial_step = float(sc.align_initial_step)
    tol = float(sc.align_tol)
    step = initial_step
    improved = False

    # Try with progressively smaller steps until we find improvement or hit tolerance
    while step >= tol and not improved:
        # Try all 6 directions (3 axes × 2 directions) with current step size
        # Skip the locked axis and any bone-locked axes
        for axis in range(3):
            if lock_rot[axis] or axis == locked_axis_idx:  # Skip locked axis and any bone-locked axes
                continue

            for direction in (1.0, -1.0):
                candidate = best_angles.copy()
                candidate[axis] += direction * step

                pb_parent.rotation_euler = candidate
                deps.update()

                dist = current_distance()
                
                # Check alignment improvement
                if dist + 1e-12 < best_dist:
                    best_dist = dist
                    best_angles = candidate.copy()
                    improved = True
                    # Keep this rotation applied (it's the best so far)
                    break  # Found improvement, exit inner loops
                else:
                    # Restore to best position if this candidate isn't better
                    pb_parent.rotation_euler = best_angles
                    deps.update()
            
            if improved:
                break  # Exit axis loop too
        
        # If no improvement with current step, try smaller step
        if not improved:
            step *= 0.5

    # Store initial distance for reporting
    initial_dist = best_dist

    # Apply the best result
    pb_parent.rotation_euler = best_angles
    deps.update()
    final_dist = current_distance()

    # Restore original rotation mode
    if prev_mode == 'QUATERNION':
        q = pb_parent.rotation_euler.to_quaternion()
        pb_parent.rotation_mode = 'QUATERNION'
        pb_parent.rotation_quaternion = q
    else:
        try:
            pb_parent.rotation_euler = Euler(best_angles, 'XYZ').to_euler(prev_mode)
            pb_parent.rotation_mode = prev_mode
        except Exception:
            pb_parent.rotation_mode = 'XYZ'

    return final_dist, improved, initial_dist

def save_parent_rotation(pb_parent):
    mode = pb_parent.rotation_mode
    if mode == 'QUATERNION':
        return {'mode': mode, 'quat': pb_parent.rotation_quaternion.copy()}
    return {'mode': mode, 'euler': pb_parent.rotation_euler.copy()}

def restore_parent_rotation(pb_parent, saved):
    mode = saved['mode']
    pb_parent.rotation_mode = mode
    if mode == 'QUATERNION':
        pb_parent.rotation_quaternion = saved['quat']
    else:
        pb_parent.rotation_euler = saved['euler']

def align_triple_distance(arm_obj, pb_child, pb_target, align_mode):
    child_use_head = align_mode.startswith('HEAD_')
    target_use_head = align_mode.endswith('_HEAD')
    p = pose_point_world(arm_obj, pb_child, child_use_head)
    t = pose_point_world(arm_obj, pb_target, target_use_head)
    return (p - t).length

def effective_bake_method_for_triple(sc, tri):
    """Analytic/COMBO overshoots on arm chains — iterative-only for wrist hand triples."""
    if tri.child_bone in ('Wrist_L', 'Wrist_R'):
        if sc.align_bake_method in ('COMBO', 'ANALYTIC'):
            return 'ITERATIVE'
    return sc.align_bake_method

def apply_align_solve(arm_obj, pb_parent, pb_child, pb_target, sc, deps, method, locked_axis, align_mode):
    """
    Run solver; revert parent rotation if distance did not improve.
    Returns (dist_before, dist_after, improved).
    """
    saved = save_parent_rotation(pb_parent)
    d0 = align_triple_distance(arm_obj, pb_child, pb_target, align_mode)
    try:
        if method == 'ANALYTIC':
            analytic_rotate_core(arm_obj, pb_parent, pb_child, pb_target, align_mode)
        elif method == 'ITERATIVE':
            iterative_minimize_core(
                arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char=locked_axis
            )
        else:
            analytic_rotate_core(arm_obj, pb_parent, pb_child, pb_target, align_mode)
            deps.update()
            d_mid = align_triple_distance(arm_obj, pb_child, pb_target, align_mode)
            if d_mid > d0 + 1e-6:
                restore_parent_rotation(pb_parent, saved)
                deps.update()
            iterative_minimize_core(
                arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char=locked_axis
            )
    except Exception as e:
        restore_parent_rotation(pb_parent, saved)
        deps.update()
        print(f"[AAF] solve failed {pb_parent.name}: {e}")
        return d0, d0, False
    deps.update()
    d1 = align_triple_distance(arm_obj, pb_child, pb_target, align_mode)
    if d1 >= d0 - 1e-6:
        restore_parent_rotation(pb_parent, saved)
        deps.update()
        return d0, d1, False
    return d0, d1, True

# -----------------------------
# Operators for add/remove triples
# -----------------------------
class SCENE_OT_align_triple_add(bpy.types.Operator):
    bl_idname = "scene.align_triple_add"
    bl_label = "Add Triple"

    def execute(self, context):
        sc = context.scene
        item = sc.align_triples.add()
        item.parent_bone = ""
        item.child_bone = ""
        item.target_bone = ""
        sc.align_triples_index = len(sc.align_triples) - 1
        return {'FINISHED'}

class SCENE_OT_align_triple_remove(bpy.types.Operator):
    bl_idname = "scene.align_triple_remove"
    bl_label = "Remove Triple"

    def execute(self, context):
        sc = context.scene
        idx = sc.align_triples_index
        if 0 <= idx < len(sc.align_triples):
            sc.align_triples.remove(idx)
            sc.align_triples_index = max(0, idx - 1)
        return {'FINISHED'}

# -----------------------------
# Operator to load 4 preset sets
# -----------------------------
class SCENE_OT_load_default_sets(bpy.types.Operator):
    bl_idname = "scene.load_default_sets"
    bl_label = "Load 4 Sets"
    bl_description = "Populate 4 preset Parent/Child/Target sets (L/R foot & hand IK) into the list"

    def execute(self, context):
        sc = context.scene
        arm = get_armature_from_context(context)
        # Clear existing items
        sc.align_triples.clear()

        # Preset 1 (Left foot)
        t1 = sc.align_triples.add()
        t1.parent_bone = resolve_bone_on_armature(arm, "Hip_L")
        t1.child_bone = resolve_bone_on_armature(arm, "Ankle_L")
        t1.target_bone = resolve_bone_on_armature(arm, "Dummy_L_Foot_IK")

        # Preset 2 (Right foot)
        t2 = sc.align_triples.add()
        t2.parent_bone = resolve_bone_on_armature(arm, "Hip_R")
        t2.child_bone = resolve_bone_on_armature(arm, "Ankle_R")
        t2.target_bone = resolve_bone_on_armature(arm, "Dummy_R_Foot_IK")

        # Preset 3 (Left hand) — elbow flex places wrist on hand IK (not shoulder swing)
        t3 = sc.align_triples.add()
        t3.parent_bone = resolve_bone_on_armature(arm, "Elbow_L")
        t3.child_bone = resolve_bone_on_armature(arm, "Wrist_L")
        t3.target_bone = resolve_bone_on_armature(arm, "Dummy_L_Hand_IK")
        t3.locked_axis = 'X'

        # Preset 4 (Right hand)
        t4 = sc.align_triples.add()
        t4.parent_bone = resolve_bone_on_armature(arm, "Elbow_R")
        t4.child_bone = resolve_bone_on_armature(arm, "Wrist_R")
        t4.target_bone = resolve_bone_on_armature(arm, "Dummy_R_Hand_IK")
        t4.locked_axis = 'X'

        sc.align_triples_index = 0
        self.report({'INFO'}, "Loaded 4 preset sets.")
        return {'FINISHED'}

class POSE_OT_setup_larian_ik_constraints(bpy.types.Operator):
    bl_idname = "pose.setup_larian_ik_constraints"
    bl_label = "Setup 4 IK Constraints"
    bl_description = (
        "Add IK constraints on Ankle_L/R and Wrist_L/R bones targeting "
        "Dummy_*_Foot_IK and Dummy_*_Hand_IK"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = get_armature_from_context(context)
        return arm is not None and arm.type == 'ARMATURE'

    def execute(self, context):
        arm_obj = get_armature_from_context(context)
        if not arm_obj:
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}
        sc = context.scene
        ok, errors = setup_larian_ik_constraints_on_armature(
            arm_obj,
            leg_chain_count=int(sc.ik_leg_chain_count),
            hand_chain_count=int(sc.ik_hand_chain_count),
            use_rotation=_aaf_ik_rotation_on(sc),
        )
        if not ok:
            msg = "; ".join(errors) if errors else "No limbs configured"
            self.report({'ERROR'}, f"IK setup failed: {msg}")
            return {'CANCELLED'}
        if _aaf_ik_bone_length_scale_on(sc, arm_obj):
            scaled, scale_errors = scale_larian_ik_bone_lengths(arm_obj)
            if scale_errors and not scaled:
                self.report({'WARNING'}, f"IK set but bone length scale failed: {'; '.join(scale_errors)}")
            elif scaled:
                print(f"[AAF] IK bone lengths scaled on: {', '.join(scaled)}")
        if errors:
            self.report({'WARNING'}, f"IK set ({len(ok)}/4): {'; '.join(errors)}")
        else:
            self.report({'INFO'}, f"IK constraints set on {len(ok)} limbs.")
        apply_bezier_if_enabled(arm_obj, sc, verbose=True)
        smooth_warnings = apply_aaf_ik_smooth_extras(context, arm_obj, sc)
        if smooth_warnings:
            self.report({'WARNING'}, "; ".join(smooth_warnings))
        return {'FINISHED'}

class POSE_OT_scale_ik_bone_lengths(bpy.types.Operator):
    bl_idname = "pose.scale_ik_bone_lengths"
    bl_label = "Scale IK Bones ×0.005"
    bl_description = "Set Ankle_L/R and Wrist_L/R bone length to current length × 0.005"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = get_armature_from_context(context)
        return arm is not None and arm.type == 'ARMATURE'

    def execute(self, context):
        arm_obj = get_armature_from_context(context)
        if not arm_obj:
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}
        ok, errors = scale_larian_ik_bone_lengths(arm_obj)
        if not ok:
            msg = "; ".join(errors) if errors else "No bones scaled"
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Scaled {len(ok)} IK bone(s): {', '.join(ok)}")
        return {'FINISHED'}

# -----------------------------
# Quickfix operator: smooth -> load presets -> bake (respects align mode from UI)
class SCENE_OT_make_larian_good(bpy.types.Operator):
    bl_idname = "scene.make_larian_good"
    bl_label = "Make Larian Animation Good"
    bl_description = "Runs Gaussian Smooth, loads 4 presets, then bakes (COMBO). Uses Align mode from the panel."
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene

        # Get armature from context
        arm = get_armature_from_context(context)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an Armature and press Make Larian Animation Good.")
            return {'CANCELLED'}

        apply_bezier_if_enabled(arm, sc, verbose=True)

        # Run gaussian smooth with the specified factor
        smooth_factor = float(sc.quickfix_smooth_count)
        if smooth_factor > 0:
            try:
                # Use the smoothing function directly with the factor
                affected = apply_graph_gaussian_smooth_for_armature_operator(
                    arm, 
                    only_selected_bones=False, 
                    factor=smooth_factor, 
                    verbose=False
                )
                try:
                    context.view_layer.update()
                except Exception:
                    pass
            except Exception as e:
                tb = traceback.format_exc()
                self.report({'WARNING'}, f"Smoothing step error: {e}")
                print("Smoothing error:", e)
                print(tb)

        # Load the 4 preset sets
        try:
            bpy.ops.scene.load_default_sets()
        except Exception as e:
            print("Load default sets error:", e)

        # COMBO bake; keep user's align_mode / bake settings from the panel
        try:
            sc.align_bake_method = 'COMBO'
            sc.align_use_threading = True
            sc.align_thread_workers = max(1, (os.cpu_count() or 2) - 1)
            suggest_debug_bone(arm, sc)
        except Exception:
            pass

        # Start the fast bake operator (invoke)
        try:
            bpy.ops.pose.align_child_bake_fast('INVOKE_DEFAULT')
        except Exception as e:
            tb = traceback.format_exc()
            self.report({'WARNING'}, f"Failed to start bake: {e}")
            print("Bake start error:", e)
            print(tb)
            return {'FINISHED'}

        self.report({'INFO'}, f"Old Method started: smoothing (factor={smooth_factor}), loaded presets, started COMBO bake.")
        return {'FINISHED'}

class SCENE_OT_make_bezier(bpy.types.Operator):
    bl_idname = "scene.make_bezier"
    bl_label = "Make Bezier"
    bl_description = "Set all keyframes on the selected armature action to Bezier interpolation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        arm = get_armature_from_context(context)
        return arm is not None and arm.type == 'ARMATURE'

    def execute(self, context):
        arm = get_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}
        changed = apply_bezier_interpolation_to_armature(arm, verbose=True)
        if changed:
            self.report({'INFO'}, f"Bezier interpolation set on {changed} keyframe(s).")
        else:
            self.report({'INFO'}, "All keyframes already Bezier (or no action).")
        return {'FINISHED'}

# -----------------------------
# Operator to toggle advanced settings visibility (unhide/hide UI)
# -----------------------------
class SCENE_OT_toggle_advanced_settings(bpy.types.Operator):
    bl_idname = "scene.toggle_advanced_settings"
    bl_label = "Toggle Advanced Settings"
    bl_description = "Toggle visibility of advanced settings"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene
        sc.show_advanced = not bool(sc.show_advanced)
        self.report({'INFO'}, f"Advanced settings {'shown' if sc.show_advanced else 'hidden'}.")
        return {'FINISHED'}

class SCENE_OT_toggle_threading(bpy.types.Operator):
    bl_idname = "scene.toggle_threading"
    bl_label = "Toggle Threading"
    bl_description = "Toggle threading for analytic bake"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene
        sc.align_use_threading = not bool(sc.align_use_threading)
        status = "enabled" if sc.align_use_threading else "disabled"
        self.report({'INFO'}, f"Threading {status}.")
        return {'FINISHED'}

# -----------------------------
# Operators (adapted to support active triple or legacy single fields)
# -----------------------------
class POSE_OT_analytic_rotate(bpy.types.Operator):
    bl_idname = "pose.align_child_analytic"
    bl_label = "Analytic Rotate (one-shot)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        arm_obj = get_armature_from_context(context)
        if not arm_obj or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}

        # decide which triple to use
        if len(sc.align_triples) > 0:
            tri = sc.align_triples[sc.align_triples_index]
            pb_parent = arm_obj.pose.bones.get(tri.parent_bone)
            pb_child = arm_obj.pose.bones.get(tri.child_bone)
            pb_target = arm_obj.pose.bones.get(tri.target_bone)
        else:
            pb_parent = arm_obj.pose.bones.get(sc.align_parent_bone)
            pb_child = arm_obj.pose.bones.get(sc.align_child_bone)
            pb_target = arm_obj.pose.bones.get(sc.align_target_bone)

        if not pb_parent or not pb_child or not pb_target:
            self.report({'ERROR'}, "Bone not found in armature.")
            return {'CANCELLED'}

        prev_active = context.view_layer.objects.active
        need_restore_active = False
        try:
            if prev_active != arm_obj:
                context.view_layer.objects.active = arm_obj
                need_restore_active = True
                bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass

        deps = bpy.context.evaluated_depsgraph_get()
        ok = analytic_rotate_core(arm_obj, pb_parent, pb_child, pb_target, sc.align_mode)
        if ok:
            deps.update()
            self.report({'INFO'}, "Applied analytic rotation (direction alignment).")
        else:
            self.report({'ERROR'}, "Zero-length vector encountered, analytic rotation aborted.")
        if need_restore_active:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass
        return {'FINISHED'}

class POSE_OT_iterative_minimize(bpy.types.Operator):
    bl_idname = "pose.align_child_iterative"
    bl_label = "Iterative Minimize (auto)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        arm_obj = get_armature_from_context(context)
        if not arm_obj or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}

        if len(sc.align_triples) > 0:
            tri = sc.align_triples[sc.align_triples_index]
            pb_parent = arm_obj.pose.bones.get(tri.parent_bone)
            pb_child = arm_obj.pose.bones.get(tri.child_bone)
            pb_target = arm_obj.pose.bones.get(tri.target_bone)
        else:
            pb_parent = arm_obj.pose.bones.get(sc.align_parent_bone)
            pb_child = arm_obj.pose.bones.get(sc.align_child_bone)
            pb_target = arm_obj.pose.bones.get(sc.align_target_bone)

        if not pb_parent or not pb_child or not pb_target:
            self.report({'ERROR'}, "Bone not found in armature.")
            return {'CANCELLED'}

        prev_active = context.view_layer.objects.active
        need_restore_active = False
        try:
            if prev_active != arm_obj:
                context.view_layer.objects.active = arm_obj
                need_restore_active = True
                bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass

        deps = bpy.context.evaluated_depsgraph_get()
        # Get locked axis from triple or use legacy property
        if len(sc.align_triples) > 0:
            locked_axis = sc.align_triples[sc.align_triples_index].locked_axis
        else:
            locked_axis = sc.align_locked_axis
        final_dist, iter_count = iterative_minimize_core(arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char=locked_axis)

        if need_restore_active:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass

        self.report({'INFO'}, f"Iterative done in {iter_count} iterations — final distance: {final_dist:.6f}")
        return {'FINISHED'}

class POSE_OT_analytic_then_iterative(bpy.types.Operator):
    bl_idname = "pose.align_child_combo"
    bl_label = "Analytic + Iterative"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        arm_obj = get_armature_from_context(context)
        if not arm_obj or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}

        if len(sc.align_triples) > 0:
            tri = sc.align_triples[sc.align_triples_index]
            pb_parent = arm_obj.pose.bones.get(tri.parent_bone)
            pb_child = arm_obj.pose.bones.get(tri.child_bone)
            pb_target = arm_obj.pose.bones.get(tri.target_bone)
        else:
            pb_parent = arm_obj.pose.bones.get(sc.align_parent_bone)
            pb_child = arm_obj.pose.bones.get(sc.align_child_bone)
            pb_target = arm_obj.pose.bones.get(sc.align_target_bone)

        if not pb_parent or not pb_child or not pb_target:
            self.report({'ERROR'}, "Bone not found in armature.")
            return {'CANCELLED'}

        prev_active = context.view_layer.objects.active
        need_restore_active = False
        try:
            if prev_active != arm_obj:
                context.view_layer.objects.active = arm_obj
                need_restore_active = True
                bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass

        deps = bpy.context.evaluated_depsgraph_get()
        analytic_ok = analytic_rotate_core(arm_obj, pb_parent, pb_child, pb_target, sc.align_mode)
        if analytic_ok:
            deps.update()
        # Get locked axis from triple or use legacy property
        if len(sc.align_triples) > 0:
            locked_axis = sc.align_triples[sc.align_triples_index].locked_axis
        else:
            locked_axis = sc.align_locked_axis
        final_dist, iter_count = iterative_minimize_core(arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char=locked_axis)
        if need_restore_active:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass

        self.report({'INFO'}, f"Analytic+Iterative finished — iter {iter_count}, dist {final_dist:.6f}")
        return {'FINISHED'}

class POSE_OT_iterative_single_step(bpy.types.Operator):
    bl_idname = "pose.align_child_iterative_single"
    bl_label = "Iterative Single Step (Test)"
    bl_description = "Apply one iteration step of the iterative solver (for testing)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        arm_obj = get_armature_from_context(context)
        if not arm_obj or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}

        if len(sc.align_triples) > 0:
            tri = sc.align_triples[sc.align_triples_index]
            pb_parent = arm_obj.pose.bones.get(tri.parent_bone)
            pb_child = arm_obj.pose.bones.get(tri.child_bone)
            pb_target = arm_obj.pose.bones.get(tri.target_bone)
        else:
            pb_parent = arm_obj.pose.bones.get(sc.align_parent_bone)
            pb_child = arm_obj.pose.bones.get(sc.align_child_bone)
            pb_target = arm_obj.pose.bones.get(sc.align_target_bone)

        if not pb_parent or not pb_child or not pb_target:
            self.report({'ERROR'}, "Bone not found in armature.")
            return {'CANCELLED'}

        prev_active = context.view_layer.objects.active
        need_restore_active = False
        try:
            if prev_active != arm_obj:
                context.view_layer.objects.active = arm_obj
                need_restore_active = True
                bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass

        deps = bpy.context.evaluated_depsgraph_get()
        # Get locked axis from triple or use legacy property
        if len(sc.align_triples) > 0:
            locked_axis = sc.align_triples[sc.align_triples_index].locked_axis
        else:
            locked_axis = sc.align_locked_axis
        final_dist, improved, initial_dist = iterative_single_step_core(arm_obj, pb_parent, pb_child, pb_target, sc, deps, locked_axis_char=locked_axis)

        if need_restore_active:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass

        if improved:
            reduction_pct = ((initial_dist - final_dist) / initial_dist * 100) if initial_dist > 1e-9 else 0
            self.report({'INFO'}, f"Single step improved — distance: {final_dist:.6f} (reduced by {reduction_pct:.1f}%)")
        else:
            self.report({'WARNING'}, f"Single step: no improvement — distance: {final_dist:.6f} (already at tolerance or step too small)")
        return {'FINISHED'}

# -----------------------------
# Cancel operator for Stop button
# -----------------------------
class POSE_OT_bake_cancel(bpy.types.Operator):
    bl_idname = "pose.align_child_bake_cancel"
    bl_label = "Stop Bake"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        context.scene.align_bake_cancel = True
        return {'FINISHED'}

# -----------------------------
# Bake: fast modal operator (processes multiple triples per frame)
# -----------------------------
class POSE_OT_bake_fast(bpy.types.Operator):
    bl_idname = "pose.align_child_bake_fast"
    bl_label = "Bake Alignment Over Range (Fast)"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    CHUNK = 16
    REDRAW_EVERY = 32

    def invoke(self, context, event=None):
        # adapted to allow invocation from other operators
        return self._invoke_common(context)

    def _invoke_common(self, context):
        sc = context.scene
        arm_obj = get_armature_from_context(context)
        if not arm_obj or getattr(arm_obj, 'type', None) != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature.")
            return {'CANCELLED'}

        autoresolve_align_triples(arm_obj, sc)
        valid_triples, _ = validate_align_triples(arm_obj, sc)
        if valid_triples == 0:
            self.report({'ERROR'}, "No valid parent/child/target triples — check bone names in Manual Work.")
            return {'CANCELLED'}

        ensure_armature_has_action(arm_obj)
        suggest_debug_bone(arm_obj, sc)

        self.scene = sc
        self.arm_obj = arm_obj
        self._keyframe_counts = {}
        # Determine frame range based on bake mode
        if sc.align_bake_mode == 'KEYFRAMES':
            # Get keyframe range from action
            action = None
            if arm_obj.animation_data and arm_obj.animation_data.action:
                action = arm_obj.animation_data.action
            if action and action.fcurves:
                # Find min and max keyframe times
                all_frames = []
                for fcu in action.fcurves:
                    if fcu.keyframe_points:
                        for kp in fcu.keyframe_points:
                            all_frames.append(int(kp.co[0]))
                if all_frames:
                    self.start = min(all_frames)
                    self.end = max(all_frames)
                else:
                    self.report({'ERROR'}, "No keyframes found in action.")
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "No action or keyframes found on armature.")
                return {'CANCELLED'}
        else:
            # Use scene frame range
            self.start = sc.frame_start
            self.end = sc.frame_end
        
        self.frame = self.start
        self.method = sc.align_bake_method
        self.prev_frame = sc.frame_current
        self.prev_active = context.view_layer.objects.active

        try:
            if context.view_layer.objects.active != arm_obj:
                context.view_layer.objects.active = arm_obj
            bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass

        self.deps = bpy.context.evaluated_depsgraph_get()
        self._bake_rotation_state = {}

        prepare_bake_parent_bones(
            arm_obj, sc, self.start, self.end,
            verbose=getattr(sc, 'align_debug_bake', False),
        )
        try:
            self.deps.update()
        except Exception:
            pass

        if getattr(sc, 'align_debug_bake', False):
            if getattr(sc, 'align_debug_log_all_parents', False):
                dbg_label = f"all parents {sorted(_collect_bake_parent_bone_names(sc))}"
            else:
                dbg_label = (getattr(sc, 'align_debug_bone', None) or '').strip() or '(none)'
            print(
                f"[AAF] bake debug ON bone='{dbg_label}' frames={self.start}-{self.end} "
                f"method={self.method} mode={sc.align_mode} "
                f"spike_deg={getattr(sc, 'align_debug_spike_deg', 45.0)}"
            )

        wm = context.window_manager
        self.total = (self.end - self.start + 1)
        try:
            sc.align_is_baking = True
            sc.align_bake_progress = 0
            sc.align_bake_cancel = False
            wm.progress_begin(0, self.total)
        except Exception:
            pass

        try:
            context.window.cursor_modal_set('DEFAULT')
        except Exception:
            pass

        try:
            self._timer = wm.event_timer_add(0.02, window=context.window)
        except Exception:
            self._timer = None
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        sc = context.scene
        wm = context.window_manager

        if event.type == 'ESC' and event.value == 'PRESS':
            return self.finish(context, success=False)
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            return self.finish(context, success=False)
        if sc.align_bake_cancel:
            return self.finish(context, success=False)

        if event.type == 'TIMER':
            if sc.align_use_threading and self.method == 'ANALYTIC':
                # process CHUNK frames, and for each frame compute all triples via worker threads
                frames = []
                snapshots_per_frame = []
                processed = 0
                while processed < self.CHUNK and self.frame <= self.end:
                    if sc.align_bake_cancel:
                        return self.finish(context, success=False)
                    sc.frame_set(self.frame)
                    self.deps.update()
                    seed_all_bake_parents(
                        self.arm_obj, sc, self._bake_rotation_state, self.frame, self.deps
                    )

                    # collect snapshots for each triple
                    frame_snaps = []
                    if len(sc.align_triples) > 0:
                        for tri in sc.align_triples:
                            pb_parent = self.arm_obj.pose.bones.get(tri.parent_bone)
                            pb_child = self.arm_obj.pose.bones.get(tri.child_bone)
                            pb_target = self.arm_obj.pose.bones.get(tri.target_bone)
                            if not pb_parent or not pb_child or not pb_target:
                                frame_snaps.append(None)
                                continue

                            pivot = tuple((self.arm_obj.matrix_world @ pb_parent.head))
                            child_use_head = sc.align_mode.startswith('HEAD_')
                            target_use_head = sc.align_mode.endswith('_HEAD')
                            child_pt = tuple(pose_point_world(self.arm_obj, pb_child, child_use_head))
                            target_pt = tuple(pose_point_world(self.arm_obj, pb_target, target_use_head))
                            arm_wq = tuple(self.arm_obj.matrix_world.to_quaternion())
                            parent_q = tuple(pb_parent.rotation_quaternion)

                            frame_snaps.append({
                                'pivot': pivot,
                                'child': child_pt,
                                'target': target_pt,
                                'arm_world_quat': arm_wq,
                                'parent_quat': parent_q,
                            })
                    else:
                        # legacy single fields
                        pb_parent = self.arm_obj.pose.bones.get(sc.align_parent_bone)
                        pb_child = self.arm_obj.pose.bones.get(sc.align_child_bone)
                        pb_target = self.arm_obj.pose.bones.get(sc.align_target_bone)
                        if not pb_parent or not pb_child or not pb_target:
                            frame_snaps.append(None)
                        else:
                            pivot = tuple((self.arm_obj.matrix_world @ pb_parent.head))
                            child_use_head = sc.align_mode.startswith('HEAD_')
                            target_use_head = sc.align_mode.endswith('_HEAD')
                            child_pt = tuple(pose_point_world(self.arm_obj, pb_child, child_use_head))
                            target_pt = tuple(pose_point_world(self.arm_obj, pb_target, target_use_head))
                            arm_wq = tuple(self.arm_obj.matrix_world.to_quaternion())
                            parent_q = tuple(pb_parent.rotation_quaternion)
                            frame_snaps.append({
                                'pivot': pivot,
                                'child': child_pt,
                                'target': target_pt,
                                'arm_world_quat': arm_wq,
                                'parent_quat': parent_q,
                            })

                    snapshots_per_frame.append(frame_snaps)
                    frames.append(self.frame)

                    self.frame += 1
                    processed += 1

                if not snapshots_per_frame:
                    return self.finish(context, success=True)

                # For each frame, run worker threads for each triple (or serial fallback)
                for fi, frame_snaps in enumerate(snapshots_per_frame):
                    fnum = frames[fi]
                    results = []
                    # submit per-triple tasks
                    try:
                        max_workers = max(1, min(int(sc.align_thread_workers), max(1, len([s for s in frame_snaps if s]))))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                            futs = [ex.submit(worker_analytic_compute, s) if s else None for s in frame_snaps]
                            for fut in futs:
                                if fut is None:
                                    results.append(None)
                                else:
                                    results.append(fut.result())
                    except Exception:
                        results = [worker_analytic_compute(s) if s else None for s in frame_snaps]

                    # apply results in order
                    sc.frame_set(fnum)
                    self.deps.update()
                    seed_all_bake_parents(
                        self.arm_obj, sc, self._bake_rotation_state, fnum, self.deps
                    )
                    if len(sc.align_triples) > 0:
                        for idx, tri in enumerate(sc.align_triples):
                            res = results[idx] if idx < len(results) else None
                            if not res:
                                continue
                            pb_parent = self.arm_obj.pose.bones.get(tri.parent_bone)
                            if not pb_parent:
                                continue
                            try:
                                prev_mode = pb_parent.rotation_mode
                                pb_parent.rotation_mode = 'QUATERNION'
                                pb_parent.rotation_quaternion = Quaternion(res)
                                bake_keyframe_parent_rotation(
                                    pb_parent, fnum, self._bake_rotation_state, prev_mode,
                                    keyframe_counts=self._keyframe_counts,
                                )
                            except Exception as e:
                                print(f"[AAF] analytic apply failed {tri.parent_bone} frame={fnum}: {e}")
                    else:
                        # legacy single
                        res = results[0] if results else None
                        if res:
                            pb_parent = self.arm_obj.pose.bones.get(sc.align_parent_bone)
                            if pb_parent:
                                try:
                                    prev_mode = pb_parent.rotation_mode
                                    pb_parent.rotation_mode = 'QUATERNION'
                                    pb_parent.rotation_quaternion = Quaternion(res)
                                    bake_keyframe_parent_rotation(
                                        pb_parent, fnum, self._bake_rotation_state, prev_mode,
                                        keyframe_counts=self._keyframe_counts,
                                    )
                                except Exception as e:
                                    print(f"[AAF] analytic apply failed {sc.align_parent_bone} frame={fnum}: {e}")

                    sc.align_bake_progress = (fnum - self.start + 1)
                    try:
                        wm.progress_update(sc.align_bake_progress)
                    except Exception:
                        pass

                    if (fnum - self.start) % self.REDRAW_EVERY == 0:
                        for area in context.screen.areas:
                            if area.type == 'VIEW_3D':
                                area.tag_redraw()

                if self.frame > self.end:
                    return self.finish(context, success=True)
                return {'RUNNING_MODAL'}

            # non-threaded path: process frames and for each frame loop triples sequentially
            processed = 0
            while processed < self.CHUNK and self.frame <= self.end:
                if sc.align_bake_cancel:
                    return self.finish(context, success=False)

                sc.frame_set(self.frame)
                self.deps.update()
                seed_all_bake_parents(
                    self.arm_obj, sc, self._bake_rotation_state, self.frame, self.deps
                )

                if len(sc.align_triples) > 0:
                    for tri in sc.align_triples:
                        pb_parent = self.arm_obj.pose.bones.get(tri.parent_bone)
                        pb_child = self.arm_obj.pose.bones.get(tri.child_bone)
                        pb_target = self.arm_obj.pose.bones.get(tri.target_bone)
                        if not pb_parent or not pb_child or not pb_target:
                            continue

                        # Get locked axis for this triple
                        locked_axis = tri.locked_axis
                        tri_method = effective_bake_method_for_triple(sc, tri)

                        dbg_dist = None
                        if _debug_bone_enabled(sc, tri.parent_bone):
                            dbg_dist = align_triple_distance(
                                self.arm_obj, pb_child, pb_target, sc.align_mode
                            )
                            align_debug_log_bone(
                                sc, pb_parent, self.frame, "BEFORE_SOLVE",
                                self._bake_rotation_state, dist=dbg_dist,
                                extra=f"method={tri_method} lock={locked_axis}",
                            )

                        d0, d1, improved = apply_align_solve(
                            self.arm_obj, pb_parent, pb_child, pb_target,
                            sc, self.deps, tri_method, locked_axis, sc.align_mode,
                        )

                        if _debug_bone_enabled(sc, tri.parent_bone):
                            if improved:
                                tag_extra = ""
                            elif tri.parent_bone in self._bake_rotation_state:
                                tag_extra = " HOLD(exact_prev_key)"
                            else:
                                tag_extra = " FIRST_KEY"
                            align_debug_log_bone(
                                sc, pb_parent, self.frame, "AFTER_SOLVE",
                                self._bake_rotation_state, dist=d1,
                                extra=f"d0={d0:.6f}{tag_extra}",
                            )

                        if improved or tri.parent_bone not in self._bake_rotation_state:
                            bake_keyframe_parent_rotation(
                                pb_parent, self.frame, self._bake_rotation_state,
                                keyframe_counts=self._keyframe_counts,
                            )
                        else:
                            bake_hold_keyframe_parent_rotation(
                                pb_parent, self.frame, self._bake_rotation_state,
                                keyframe_counts=self._keyframe_counts,
                            )
                else:
                    pb_parent = self.arm_obj.pose.bones.get(sc.align_parent_bone)
                    pb_child = self.arm_obj.pose.bones.get(sc.align_child_bone)
                    pb_target = self.arm_obj.pose.bones.get(sc.align_target_bone)
                    if pb_parent and pb_child and pb_target:
                        locked_axis = sc.align_locked_axis
                        d0, d1, improved = apply_align_solve(
                            self.arm_obj, pb_parent, pb_child, pb_target,
                            sc, self.deps, self.method, locked_axis, sc.align_mode,
                        )
                        if improved or sc.align_parent_bone not in self._bake_rotation_state:
                            bake_keyframe_parent_rotation(
                                pb_parent, self.frame, self._bake_rotation_state,
                                keyframe_counts=self._keyframe_counts,
                            )
                        else:
                            bake_hold_keyframe_parent_rotation(
                                pb_parent, self.frame, self._bake_rotation_state,
                                keyframe_counts=self._keyframe_counts,
                            )

                sc.align_bake_progress = (self.frame - self.start + 1)
                try:
                    wm.progress_update(sc.align_bake_progress)
                except Exception:
                    pass

                if (self.frame - self.start) % self.REDRAW_EVERY == 0:
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()

                self.frame += 1
                processed += 1

            if self.frame > self.end:
                return self.finish(context, success=True)

            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def finish(self, context, success: bool):
        wm = context.window_manager
        sc = context.scene

        try:
            if self._timer is not None:
                wm.event_timer_remove(self._timer)
                self._timer = None
        except Exception:
            pass

        try:
            wm.progress_end()
        except Exception:
            pass

        sc.align_is_baking = False
        sc.align_bake_progress = 0
        sc.align_bake_cancel = False

        try:
            context.window.cursor_modal_restore()
        except Exception:
            pass

        try:
            sc.frame_set(self.prev_frame)
        except Exception:
            pass

        # restore previous active object (if any), then ensure we are in Object mode
        try:
            context.view_layer.objects.active = self.prev_active
        except Exception:
            pass

        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            # if mode set fails (bad context/no active object), ignore silently
            pass

        if success:
            apply_bezier_if_enabled(self.arm_obj, sc, verbose=True)
            counts = getattr(self, '_keyframe_counts', None) or {}
            if counts:
                total_keys = sum(counts.values())
                per_bone = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"[AAF] bake keyframe summary: total={total_keys} per_bone={{ {per_bone} }}")
            else:
                print("[AAF] bake keyframe summary: no keyframes inserted (check console for errors)")
            self.report({'INFO'}, f"Fast bake finished: frames {self.start}.{self.end}.")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Fast bake canceled.")
            return {'CANCELLED'}

# -----------------------------
# Pick selected bone operator (updated to work with triples)
# -----------------------------
class POSE_OT_pick_selected_bone(bpy.types.Operator):
    bl_idname = "pose.pick_selected_bone"
    bl_label = "Pick Selected Bone"
    bl_options = {'REGISTER', 'UNDO'}

    slot: bpy.props.EnumProperty(
        name="Slot",
        items=[('PARENT', 'Parent', ''), ('CHILD', 'Child', ''), ('TARGET', 'Target', '')]
    )
    index: bpy.props.IntProperty(name="Triple Index", default=-1)

    def execute(self, context):
        sc = context.scene
        arm_obj = get_armature_from_context(context)
        if not arm_obj or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature first.")
            return {'CANCELLED'}

        pb = None
        # try active pose bone first
        if context.active_object == arm_obj and getattr(context, 'active_pose_bone', None):
            pb = context.active_pose_bone
        else:
            for b in arm_obj.pose.bones:
                if getattr(b.bone, 'select', False):
                    pb = b
                    break

        if not pb:
            self.report({'ERROR'}, "No selected pose bone found on that armature.")
            return {'CANCELLED'}

        if 0 <= self.index < len(sc.align_triples):
            tri = sc.align_triples[self.index]
            if self.slot == 'PARENT':
                tri.parent_bone = pb.name
            elif self.slot == 'CHILD':
                tri.child_bone = pb.name
            else:
                tri.target_bone = pb.name
        else:
            # legacy single fields
            if self.slot == 'PARENT':
                sc.align_parent_bone = pb.name
            elif self.slot == 'CHILD':
                sc.align_child_bone = pb.name
            else:
                sc.align_target_bone = pb.name

        self.report({'INFO'}, f"Picked bone '{pb.name}' into {self.slot.lower()} field.")
        return {'FINISHED'}

# -----------------------------
# Small helpers used by operator-based smoothing
# -----------------------------
def _is_bone_fcurve(fcu):
    return fcu.data_path.startswith('pose.bones')

def _extract_bone_name_from_path(dp):
    try:
        start = dp.find('["')
        end = dp.find('"]', start+1)
        if start == -1 or end == -1:
            # try single-quote style
            start = dp.find("['")
            end = dp.find("']", start+1)
            if start == -1 or end == -1:
                return None
            return dp[start+2:end]
        return dp[start+2:end]
    except Exception:
        return None

# -----------------------------
# New: Operator-based smoothing wrapper (calls bpy.ops.graph.gaussian_smooth(factor=1))
# - Selects matching bone fcurves/keyframes
# - Uses an override with a Graph Editor area/region so the operator poll succeeds
# - Restores selection and (if temporarily changed) area.type
# -----------------------------
def apply_graph_gaussian_smooth_for_armature_operator(
    arm_obj, only_selected_bones=False, factor=1.0, verbose=False,
    action=None, bone_names=None, smooth_all_fcurves=False,
):
    """
    Apply Blender's built-in Gaussian smooth operator to bone F-curves
    in the armature's action. Temporarily switches to pose mode to ensure proper context.
    Args:
        arm_obj: The armature object
        only_selected_bones: If True, only smooth selected bones
        factor: Smooth factor (0.0 to 10.0, default 1.0)
        verbose: If True, print debug messages
        action: Optional action to smooth (defaults to animation_data.action)
        bone_names: Optional set/list of bone names to limit smoothing
        smooth_all_fcurves: If True, smooth every F-curve in the action
    Returns the number of fcurves targeted (approx).
    """
    ctx = bpy.context
    ob = arm_obj
    if ob is None or getattr(ob, 'type', None) != 'ARMATURE':
        raise RuntimeError("Provided object must be an Armature.")

    ad = ob.animation_data
    saved_action = None
    need_restore_action = False
    if action is None:
        if ad:
            action = ad.action
    elif ad is not None and ad.action != action:
        saved_action = ad.action
        ad.action = action
        need_restore_action = True

    if action is None or not action.fcurves:
        if verbose:
            print("apply_graph_gaussian_smooth_for_armature_operator: no action or fcurves found; nothing to smooth.")
        if need_restore_action and ad is not None:
            ad.action = saved_action
        return 0

    # Save current mode and active object
    original_mode = ctx.mode
    original_active = ctx.active_object
    need_restore_mode = False
    need_restore_active = False
    
    # Save bone selection states (from object mode if we're in object mode)
    bone_names_to_select = None
    if bone_names is not None:
        bone_names_to_select = set(bone_names)
    elif only_selected_bones:
        # Get currently selected bones (works in both modes)
        bone_names_to_select = {pb.name for pb in ob.pose.bones if pb.bone.select}
        if not bone_names_to_select:
            if verbose:
                print("apply_graph_gaussian_smooth_for_armature_operator: only_selected_bones=True but no bones selected.")
            if need_restore_action and ad is not None:
                ad.action = saved_action
            return 0

    # Save selection states for all fcurves & keypoints in action
    saved_states = []
    for fcu in action.fcurves:
        kp_sel = [kp.select_control_point for kp in fcu.keyframe_points]
        saved_states.append((fcu, fcu.select, kp_sel))

    # Decide which fcurves to select for the operator
    target_fcurves = []
    for fcu in action.fcurves:
        if smooth_all_fcurves:
            target_fcurves.append(fcu)
            continue
        if not _is_bone_fcurve(fcu):
            continue
        if bone_names_to_select is not None:
            bone = _extract_bone_name_from_path(fcu.data_path)
            if bone is None or bone not in bone_names_to_select:
                continue
        target_fcurves.append(fcu)

    if not target_fcurves:
        # nothing to change; restore and exit
        for fcu, fcu_sel, kp_sel in saved_states:
            try:
                fcu.select = fcu_sel
                for kp, sel in zip(fcu.keyframe_points, kp_sel):
                    kp.select_control_point = sel
            except Exception:
                pass
        if verbose:
            print("apply_graph_gaussian_smooth_for_armature_operator: no bone fcurves matched filter.")
        if need_restore_action and ad is not None:
            ad.action = saved_action
        return 0

    # Select the target fcurves & their keys (operator works on selected keyframes/fcurves)
    for fcu in action.fcurves:
        try:
            if fcu in target_fcurves:
                fcu.select = True
                for kp in fcu.keyframe_points:
                    kp.select_control_point = True
            else:
                # keep other fcurves unselected to avoid accidental smoothing
                fcu.select = False
                for kp in fcu.keyframe_points:
                    kp.select_control_point = False
        except Exception:
            pass

    affected_count = len(target_fcurves)
    
    # Variables for area restoration
    changed_area = False
    old_area_type = None
    area_to_restore = None
    
    try:
        # Ensure armature is active
        if ctx.active_object != ob:
            ctx.view_layer.objects.active = ob
            need_restore_active = True
        
        # Switch to pose mode if not already in pose mode
        if ctx.mode != 'POSE':
            try:
                bpy.ops.object.mode_set(mode='POSE')
                need_restore_mode = True
            except Exception as e:
                if verbose:
                    print(f"Failed to switch to pose mode: {e}")
                # Restore fcurve selections and exit
                for fcu, fcu_sel, kp_sel in saved_states:
                    try:
                        fcu.select = fcu_sel
                        for kp, sel in zip(fcu.keyframe_points, kp_sel):
                            kp.select_control_point = sel
                    except Exception:
                        pass
                if need_restore_active and original_active:
                    try:
                        ctx.view_layer.objects.active = original_active
                    except Exception:
                        pass
                if need_restore_action and ad is not None:
                    ad.action = saved_action
                return 0
        
        # Select bones in pose mode based on our filter
        if bone_names_to_select is not None:
            # Deselect all bones first
            bpy.ops.pose.select_all(action='DESELECT')
            # Select only the bones we want
            for bone_name in bone_names_to_select:
                pb = ob.pose.bones.get(bone_name)
                if pb:
                    pb.bone.select = True
        elif not only_selected_bones and not smooth_all_fcurves:
            # Select all bones if we're smoothing all bone fcurves
            bpy.ops.pose.select_all(action='SELECT')
        
        # Find or create Graph Editor area for context override
        screen = ctx.screen
        area = None
        region = None
        
        # Look for existing Graph Editor
        for a in screen.areas:
            if a.type == 'GRAPH_EDITOR':
                area = a
                break
        
        # If no Graph Editor found, temporarily change first area
        if area is None:
            if len(screen.areas) > 0:
                area = screen.areas[0]
                old_area_type = area.type
                area_to_restore = area
                try:
                    area.type = 'GRAPH_EDITOR'
                    changed_area = True
                except Exception:
                    if verbose:
                        print("Could not change area to Graph Editor")
                    area = None
                    area_to_restore = None
        
        # Find WINDOW region in the area
        if area:
            for r in area.regions:
                if r.type == 'WINDOW':
                    region = r
                    break
        
        # Get space_data if available
        space_data = None
        if area and area.type == 'GRAPH_EDITOR':
            space_data = area.spaces.active
        
        # Now call the operator with Graph Editor context override
        if area and region:
            override = {
                'window': ctx.window,
                'screen': screen,
                'area': area,
                'region': region,
                'active_object': ob,
                'object': ob,
                'selected_objects': [ob],
                'selected_editable_objects': [ob],
            }
            if space_data:
                override['space_data'] = space_data
            
            with ctx.temp_override(**override):
                bpy.ops.graph.gaussian_smooth(factor=factor)
        else:
            # Fallback: try without override (might work if Graph Editor is already active)
            bpy.ops.graph.gaussian_smooth(factor=factor)
        
        # Restore area type if we changed it
        if changed_area and area_to_restore and old_area_type:
            try:
                area_to_restore.type = old_area_type
            except Exception:
                pass
        
    except Exception as e:
        # Operator failed; restore state and re-raise so callers can handle if needed
        tb = traceback.format_exc()
        if verbose:
            print("graph.gaussian_smooth failed:", e)
            print(tb)
        # restore original select states
        for fcu, fcu_sel, kp_sel in saved_states:
            try:
                fcu.select = fcu_sel
                for kp, sel in zip(fcu.keyframe_points, kp_sel):
                    kp.select_control_point = sel
            except Exception:
                pass
        # restore area type if we changed it
        if changed_area and area_to_restore and old_area_type:
            try:
                area_to_restore.type = old_area_type
            except Exception:
                pass
        # restore original mode
        if need_restore_mode:
            try:
                if original_mode == 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                elif original_mode == 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')
                # Add other modes as needed
            except Exception:
                pass
        # restore original active object
        if need_restore_active and original_active:
            try:
                ctx.view_layer.objects.active = original_active
            except Exception:
                pass
        if need_restore_action and ad is not None:
            try:
                ad.action = saved_action
            except Exception:
                pass
        # re-raise wrapped error for caller to log if needed
        raise

    # restore original selection states
    for fcu, fcu_sel, kp_sel in saved_states:
        try:
            fcu.select = fcu_sel
            for kp, sel in zip(fcu.keyframe_points, kp_sel):
                kp.select_control_point = sel
        except Exception:
            pass

    # restore original mode
    if need_restore_mode:
        try:
            if original_mode == 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            elif original_mode == 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            # Add other modes as needed
        except Exception:
            pass
    
    # restore original active object
    if need_restore_active and original_active:
        try:
            ctx.view_layer.objects.active = original_active
        except Exception:
            pass

    if need_restore_action and ad is not None:
        try:
            ad.action = saved_action
        except Exception:
            pass

    # ensure depsgraph / UI update
    try:
        ctx.view_layer.update()
    except Exception:
        pass

    if verbose:
        print(f"Called bpy.ops.graph.gaussian_smooth(factor={factor}) on approximately {affected_count} bone fcurves.")
    return affected_count

# -----------------------------
# Operator wrapper for UI (calls the function above)
# -----------------------------
class SCENE_OT_gaussian_smooth_curves(bpy.types.Operator):
    bl_idname = "scene.gaussian_smooth_curves"
    bl_label = "Gaussian Smooth Curves (Graph Op)"
    bl_description = "Apply Graph Editor Gaussian smoothing to bone F-curves using Blender's operator with configurable factor."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        arm = get_armature_from_context(context)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature first.")
            return {'CANCELLED'}

        only_selected = bool(sc.smooth_only_selected_bones)
        smooth_factor = float(sc.manual_smooth_count)

        try:
            affected = apply_graph_gaussian_smooth_for_armature_operator(arm, only_selected_bones=only_selected, factor=smooth_factor, verbose=False)
            self.report({'INFO'}, f"Requested built-in Gaussian smooth (factor={smooth_factor}) on ~{affected} bone F-curves.")
            return {'FINISHED'}
        except Exception as e:
            tb = traceback.format_exc()
            self.report({'ERROR'}, f"Smooth failed: {e}")
            print("Gaussian smooth error (operator):", e)
            print(tb)
            return {'CANCELLED'}

# -----------------------------
# UI Panel
# -----------------------------
class VIEW3D_PT_pose_align_panel(bpy.types.Panel):
    bl_label = "Animation Fixer"
    bl_idname = "VIEW3D_PT_pose_align_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"

    @classmethod
    def poll(cls, context):
        return True

    def _draw_aaf_toggle(self, layout, sc, prop_name, on_text, off_text):
        row = layout.row(align=True)
        enabled = bool(getattr(sc, prop_name))
        row.prop(
            sc, prop_name,
            text=_aaf_toggle_label(enabled, on_text, off_text),
            toggle=True,
        )

    def _draw_smooth_option(self, layout, sc):
        row = layout.row(align=True)
        split = row.split(factor=0.8, align=True)
        split.row(align=True).prop(sc, "aaf_smooth_mode", expand=True)
        split.prop(sc, "aaf_smooth_passes", text="×")

    def _draw_ik_toggles(self, layout, sc, context):
        self._draw_aaf_toggle(layout, sc, "aaf_ik_rotation", "IK Target Rotation", "IK Target Rotation (Off)")
        arm = get_armature_from_context(context)
        bone_length_locked = ik_bones_already_small(arm) if arm else False
        bone_row = layout.row(align=True)
        bone_row.enabled = not bone_length_locked
        if bone_length_locked:
            bone_row.alignment = 'CENTER'
            bone_row.label(text="Bone Length ×0.005 (Off)")
        else:
            enabled = bool(getattr(sc, "aaf_ik_bone_length_scale"))
            bone_row.prop(
                sc, "aaf_ik_bone_length_scale",
                text=_aaf_toggle_label(enabled, "Bone Length ×0.005", "Bone Length ×0.005 (Off)"),
                toggle=True,
            )
        self._draw_aaf_toggle(layout, sc, "aaf_auto_bezier", "Auto Bezier", "Auto Bezier (Off)")
        self._draw_smooth_option(layout, sc)

    def draw(self, context):
        layout = self.layout
        sc = context.scene

        # --- IK & tools (top) ---
        tools_box = layout.box()
        tools_col = tools_box.column(align=True)
        tools_col.label(text="IK Setup:", icon='CONSTRAINT_BONE')
        ik_chain_row = tools_col.row(align=True)
        ik_chain_row.prop(sc, "ik_leg_chain_count", text="Leg chain")
        ik_chain_row.prop(sc, "ik_hand_chain_count", text="Hand chain")
        self._draw_ik_toggles(tools_col, sc, context)
        tools_col.operator(
            "pose.setup_larian_ik_constraints",
            icon='CONSTRAINT_BONE',
            text="Setup 4 IK Constraints",
        )

        layout.separator()
        self._draw_old_method_box(layout, context, sc)

    def _draw_old_method_box(self, layout, context, sc):
        old_box = layout.box()
        old_col = old_box.column(align=True)
        old_col.label(text="Old Method:", icon='PRESET')
        old_col.operator("scene.make_larian_good", icon='LIGHT_SUN', text="Make Larian Animation Good")
        smooth_row = old_col.row(align=True)
        smooth_row.prop(sc, "quickfix_smooth_count", text="Smooth animation")
        old_col.prop(sc, "align_mode", text="Align")
        bake_mode_row = old_col.row(align=True)
        bake_mode_row.prop(sc, "align_bake_mode", expand=True)

        old_col.separator()
        manual_row = old_col.row(align=True)
        manual_row.prop(sc, "show_advanced", toggle=True, text="Manual Work")

        if not sc.show_advanced:
            return

        arm = get_armature_from_context(context)
        if arm and arm.type == 'ARMATURE' and arm.data:
            manual_col = old_col.column(align=True)
            manual_col.separator()
            row = manual_col.row(align=True)
            row.operator("scene.load_default_sets", icon='OUTLINER_OB_ARMATURE', text="Load 4 Sets")
            row.operator(
                "pose.setup_larian_ik_constraints",
                icon='CONSTRAINT_BONE',
                text="Setup IK",
            )
            ik_row = manual_col.row(align=True)
            ik_row.prop(sc, "ik_leg_chain_count", text="Leg chain")
            ik_row.prop(sc, "ik_hand_chain_count", text="Hand chain")
            self._draw_ik_toggles(manual_col, sc, context)

            box = manual_col.box()
            col = box.column()
            for i, tri in enumerate(sc.align_triples):
                tri_box = col.box()

                r = tri_box.row(align=True)
                r.prop_search(tri, "parent_bone", arm.data, "bones", text="Parent")
                op = r.operator("pose.pick_selected_bone", text="", icon='EYEDROPPER')
                op.slot = 'PARENT'
                op.index = i

                r = tri_box.row(align=True)
                r.prop_search(tri, "child_bone", arm.data, "bones", text="Child")
                op = r.operator("pose.pick_selected_bone", text="", icon='EYEDROPPER')
                op.slot = 'CHILD'
                op.index = i

                r = tri_box.row(align=True)
                r.prop_search(tri, "target_bone", arm.data, "bones", text="Target")
                op = r.operator("pose.pick_selected_bone", text="", icon='EYEDROPPER')
                op.slot = 'TARGET'
                op.index = i

                axis_row = tri_box.row(align=True)
                axis_row.label(text="Lock Axis:")
                axis_row.prop(tri, "locked_axis", expand=True)

                tri_box.separator()

            r = box.row(align=True)
            r.operator('scene.align_triple_add', icon='ADD', text='Add')
            r.operator('scene.align_triple_remove', icon='REMOVE', text='Remove')

            sbox = manual_col.box()
            sbox.label(text="Smooth Bone Curves:")
            srow = sbox.row(align=True)
            srow.prop(sc, "smooth_only_selected_bones", text="Only selected bones")
            smooth_factor_row = sbox.row(align=True)
            smooth_factor_row.prop(sc, "manual_smooth_count", text="Smooth Factor")
            srow = sbox.row(align=True)
            srow.operator("scene.gaussian_smooth_curves", icon='SMOOTHCURVE', text='Gaussian Smooth')

            box2 = manual_col.box()
            box2.label(text="Bake Animation:")
            box2.prop(sc, "align_bake_method")

            worker_row = box2.row(align=True)
            worker_row.operator("scene.toggle_threading", icon='CHECKBOX_HLT' if sc.align_use_threading else 'CHECKBOX_DEHLT', text="Threading")
            workers_sub = worker_row.row(align=True)
            workers_sub.enabled = bool(sc.align_use_threading)
            workers_sub.prop(sc, "align_thread_workers", text="Workers")

            bake_col = box2.column(align=True)
            bake_col.enabled = not sc.align_is_baking
            bake_col.operator("pose.align_child_bake_fast", icon='REC', text=("Bake" if not sc.align_is_baking else "Baking..."))
            bake_mode_row = bake_col.row(align=True)
            bake_mode_row.prop(sc, "align_bake_mode", expand=True)

            dbg_box = box2.box()
            dbg_box.label(text="Bake debug (System Console):")
            dbg_box.prop(sc, "align_debug_bake", text="Log bone each frame")
            dbg_box.prop(sc, "align_debug_log_all_parents", text="Log all parent bones")
            dbg_row = dbg_box.row(align=True)
            dbg_row.enabled = sc.align_debug_bake and not sc.align_debug_log_all_parents
            dbg_row.prop(sc, "align_debug_bone", text="Bone")
            dbg_row.prop(sc, "align_debug_spike_deg", text="Spike °")

            box2.label(text="Cancel: Esc or Right-click")

            if sc.align_is_baking:
                total = (sc.frame_end - sc.frame_start + 1)
                box2.label(text=f"Progress: {sc.align_bake_progress}/{total}")
        else:
            old_col.label(text="Select an armature", icon='INFO')

# -----------------------------
# Addon Preferences
# -----------------------------
class ADDON_PREFS_akelka_bone_alignment(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        layout.label(text="Support the development:")
        row = layout.row()
        op = row.operator("wm.url_open", text="Support on Patreon", icon='FUND')
        op.url = "https://www.patreon.com/c/AkELkA"

# register / unregister
classes = (
    AlignTriple,
    SCENE_OT_align_triple_add,
    SCENE_OT_align_triple_remove,
    SCENE_OT_load_default_sets,
    POSE_OT_setup_larian_ik_constraints,
    POSE_OT_scale_ik_bone_lengths,
    SCENE_OT_make_larian_good,
    SCENE_OT_make_bezier,
    SCENE_OT_toggle_advanced_settings,
    SCENE_OT_toggle_threading,
    POSE_OT_analytic_rotate,
    POSE_OT_iterative_minimize,
    POSE_OT_iterative_single_step,
    POSE_OT_analytic_then_iterative,
    POSE_OT_bake_fast,
    POSE_OT_bake_cancel,
    POSE_OT_pick_selected_bone,
    SCENE_OT_gaussian_smooth_curves,
    VIEW3D_PT_pose_align_panel,
    ADDON_PREFS_akelka_bone_alignment,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    register_props()

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_props()

if __name__ == "__main__":
    register()
