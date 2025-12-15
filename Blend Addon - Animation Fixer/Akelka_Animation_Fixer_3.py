
bl_info = {
    "name": "Akelka Animation Fixer",
    "author": "Akelka",
    "version": (1, 0, 1),
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
        name="Show Manual Work",
        description="Toggle visibility of manual work options (triples, smoothing, bake settings)",
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

def unregister_props():
    for p in ("show_advanced",
              "align_parent_bone", "align_child_bone", "align_target_bone",
              "align_mode", "align_initial_step", "align_tol", "align_max_iter", "align_bake_method", "align_bake_mode",
              "align_triples", "align_triples_index",
              "align_is_baking", "align_bake_progress", "align_bake_cancel",
              "align_use_threading", "align_thread_workers",
              "smooth_only_selected_bones", "align_locked_axis", "quickfix_smooth_count", "manual_smooth_count"):
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
        # Clear existing items
        sc.align_triples.clear()

        # Preset 1 (Left foot)
        t1 = sc.align_triples.add()
        t1.parent_bone = "Hip_L"
        t1.child_bone = "Ankle_L"
        t1.target_bone = "Dummy_L_Foot_IK"

        # Preset 2 (Right foot)
        t2 = sc.align_triples.add()
        t2.parent_bone = "Hip_R"
        t2.child_bone = "Ankle_R"
        t2.target_bone = "Dummy_R_Foot_IK"

        # Preset 3 (Left hand)
        t3 = sc.align_triples.add()
        t3.parent_bone = "Should_L"
        t3.child_bone = "Wrist_L"
        t3.target_bone = "Dummy_L_Hand_IK"

        # Preset 4 (Right hand)
        t4 = sc.align_triples.add()
        t4.parent_bone = "Should_R"
        t4.child_bone = "Wrist_R"
        t4.target_bone = "Dummy_R_Hand_IK"

        sc.align_triples_index = 0
        self.report({'INFO'}, "Loaded 4 preset sets.")
        return {'FINISHED'}

# -----------------------------
# Quickfix operator: runs Smooth x3 -> Load 4 Sets -> Bake iterative (multithread on)
# Quickfix now sets selected armature into the scene before running but does NOT force UI hiding.
# -----------------------------
class SCENE_OT_make_larian_good(bpy.types.Operator):
    bl_idname = "scene.make_larian_good"
    bl_label = "Make Larian Animation Good"
    bl_description = "Runs Gaussian Smooth multiple times, loads 4 presets, then bakes (Iterative, multithread ON)."
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene

        # Get armature from context
        arm = get_armature_from_context(context)
        if not arm or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an Armature and press the Quickfix button.")
            return {'CANCELLED'}

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

        # Set iterative bake and enable threading
        try:
            sc.align_bake_method = 'ITERATIVE'
            sc.align_use_threading = True
            sc.align_thread_workers = max(1, (os.cpu_count() or 2) - 1)
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

        self.report({'INFO'}, f"Quickfix started: smoothing (factor={smooth_factor}), loaded presets, started iterative bake (multithread ON).")
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

        # validate triples if present
        if len(sc.align_triples) > 0:
            for t in sc.align_triples:
                # no hard validation here; we'll skip missing bones per-frame
                pass

        self.scene = sc
        self.arm_obj = arm_obj
        
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
                    if len(sc.align_triples) > 0:
                        for idx, tri in enumerate(sc.align_triples):
                            res = results[idx] if idx < len(results) else None
                            if not res:
                                continue
                            pb_parent = self.arm_obj.pose.bones.get(tri.parent_bone)
                            if not pb_parent:
                                continue
                            try:
                                pb_parent.rotation_mode = 'QUATERNION'
                                pb_parent.rotation_quaternion = Quaternion(res)
                                pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=fnum)
                            except Exception:
                                pass
                    else:
                        # legacy single
                        res = results[0] if results else None
                        if res:
                            pb_parent = self.arm_obj.pose.bones.get(sc.align_parent_bone)
                            if pb_parent:
                                try:
                                    pb_parent.rotation_mode = 'QUATERNION'
                                    pb_parent.rotation_quaternion = Quaternion(res)
                                    pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=fnum)
                                except Exception:
                                    pass

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

                if len(sc.align_triples) > 0:
                    for tri in sc.align_triples:
                        pb_parent = self.arm_obj.pose.bones.get(tri.parent_bone)
                        pb_child = self.arm_obj.pose.bones.get(tri.child_bone)
                        pb_target = self.arm_obj.pose.bones.get(tri.target_bone)
                        if not pb_parent or not pb_child or not pb_target:
                            continue

                        # Get locked axis for this triple
                        locked_axis = tri.locked_axis

                        if self.method == 'ANALYTIC':
                            analytic_rotate_core(self.arm_obj, pb_parent, pb_child, pb_target, sc.align_mode)
                        elif self.method == 'ITERATIVE':
                            iterative_minimize_core(self.arm_obj, pb_parent, pb_child, pb_target, sc, self.deps, locked_axis_char=locked_axis)
                        else:
                            analytic_rotate_core(self.arm_obj, pb_parent, pb_child, pb_target, sc.align_mode)
                            iterative_minimize_core(self.arm_obj, pb_parent, pb_child, pb_target, sc, self.deps, locked_axis_char=locked_axis)

                        # keyframe parent
                        try:
                            if pb_parent.rotation_mode == 'QUATERNION':
                                pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=self.frame)
                            else:
                                pb_parent.keyframe_insert(data_path="rotation_euler", frame=self.frame)
                        except Exception:
                            pass
                else:
                    # legacy single-fields behaviour
                    pb_parent = self.arm_obj.pose.bones.get(sc.align_parent_bone)
                    pb_child = self.arm_obj.pose.bones.get(sc.align_child_bone)
                    pb_target = self.arm_obj.pose.bones.get(sc.align_target_bone)
                    if pb_parent and pb_child and pb_target:
                        # Get locked axis from legacy property
                        locked_axis = sc.align_locked_axis

                        if self.method == 'ANALYTIC':
                            analytic_rotate_core(self.arm_obj, pb_parent, pb_child, pb_target, sc.align_mode)
                        elif self.method == 'ITERATIVE':
                            iterative_minimize_core(self.arm_obj, pb_parent, pb_child, pb_target, sc, self.deps, locked_axis_char=locked_axis)
                        else:
                            analytic_rotate_core(self.arm_obj, pb_parent, pb_child, pb_target, sc.align_mode)
                            iterative_minimize_core(self.arm_obj, pb_parent, pb_child, pb_target, sc, self.deps, locked_axis_char=locked_axis)

                        try:
                            if pb_parent.rotation_mode == 'QUATERNION':
                                pb_parent.keyframe_insert(data_path="rotation_quaternion", frame=self.frame)
                            else:
                                pb_parent.keyframe_insert(data_path="rotation_euler", frame=self.frame)
                        except Exception:
                            pass

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
def apply_graph_gaussian_smooth_for_armature_operator(arm_obj, only_selected_bones=False, factor=1.0, verbose=False):
    """
    Apply Blender's built-in Gaussian smooth operator to bone F-curves
    in the armature's action. Temporarily switches to pose mode to ensure proper context.
    Args:
        arm_obj: The armature object
        only_selected_bones: If True, only smooth selected bones
        factor: Smooth factor (0.0 to 10.0, default 1.0)
        verbose: If True, print debug messages
    Returns the number of fcurves targeted (approx).
    """
    ctx = bpy.context
    ob = arm_obj
    if ob is None or getattr(ob, 'type', None) != 'ARMATURE':
        raise RuntimeError("Provided object must be an Armature.")

    # Ensure there's animation data and fcurves
    action = None
    if ob.animation_data:
        action = ob.animation_data.action
    if action is None or not action.fcurves:
        if verbose:
            print("apply_graph_gaussian_smooth_for_armature_operator: no action or fcurves found; nothing to smooth.")
        return 0

    # Save current mode and active object
    original_mode = ctx.mode
    original_active = ctx.active_object
    need_restore_mode = False
    need_restore_active = False
    
    # Save bone selection states (from object mode if we're in object mode)
    bone_names_to_select = None
    if only_selected_bones:
        # Get currently selected bones (works in both modes)
        bone_names_to_select = {pb.name for pb in ob.pose.bones if pb.bone.select}
        if not bone_names_to_select:
            if verbose:
                print("apply_graph_gaussian_smooth_for_armature_operator: only_selected_bones=True but no bones selected.")
            return 0

    # Save selection states for all fcurves & keypoints in action
    saved_states = []
    for fcu in action.fcurves:
        kp_sel = [kp.select_control_point for kp in fcu.keyframe_points]
        saved_states.append((fcu, fcu.select, kp_sel))

    # Decide which fcurves to select for the operator: all bone fcurves matching filter
    target_fcurves = []
    for fcu in action.fcurves:
        if not _is_bone_fcurve(fcu):
            continue
        bone = _extract_bone_name_from_path(fcu.data_path)
        if bone is None:
            continue
        if bone_names_to_select is not None and bone not in bone_names_to_select:
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
                return 0
        
        # Select bones in pose mode based on our filter
        if only_selected_bones and bone_names_to_select:
            # Deselect all bones first
            bpy.ops.pose.select_all(action='DESELECT')
            # Select only the bones we want
            for bone_name in bone_names_to_select:
                pb = ob.pose.bones.get(bone_name)
                if pb:
                    pb.bone.select = True
        elif not only_selected_bones:
            # Select all bones if we're smoothing all
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

    def draw(self, context):
        layout = self.layout
        sc = context.scene

        # Quickfix button - placed at the top of everything
        quickfix_box = layout.box()
        quickfix_col = quickfix_box.column(align=True)
        quickfix_col.operator("scene.make_larian_good", icon='LIGHT_SUN', text="Make Larian Animation Good")
        smooth_row = quickfix_col.row(align=True)
        smooth_row.prop(sc, "quickfix_smooth_count", text="Smooth animation")
        bake_mode_row = quickfix_col.row(align=True)
        bake_mode_row.prop(sc, "align_bake_mode", expand=True)

        # Manual Work toggle (collapsible): this now hides/unhides manual work options
        adv_row = layout.row(align=True)
        adv_row.prop(sc, "show_advanced", toggle=True, text="Manual Work")

        # If manual work is collapsed, avoid drawing the manual work blocks (hides everything inside)
        if not sc.show_advanced:
            return

        # --- Manual Work: show Mode and all other controls ---
        arm = get_armature_from_context(context)
        if arm and arm.type == 'ARMATURE' and arm.data:
            layout.prop(sc, "align_mode")
            # Button that loads 4 preset sets
            row = layout.row(align=True)
            row.operator("scene.load_default_sets", icon='OUTLINER_OB_ARMATURE', text="Load 4 Sets")

            # UI list of sets/triples (no "Set:" label)
            box = layout.box()
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

                # Axis lock buttons (X, Y, Z) - only one can be active
                axis_row = tri_box.row(align=True)
                axis_row.label(text="Lock Axis:")
                axis_row.prop(tri, "locked_axis", expand=True)

                tri_box.separator()

            # add/remove buttons
            r = box.row(align=True)
            r.operator('scene.align_triple_add', icon='ADD', text='Add')
            r.operator('scene.align_triple_remove', icon='REMOVE', text='Remove')

            # --- Smoothing section using Graph Editor operator ---
            sbox = layout.box()
            sbox.label(text="Smooth Bone Curves:")
            srow = sbox.row(align=True)
            srow.prop(sc, "smooth_only_selected_bones", text="Only selected bones")
            smooth_factor_row = sbox.row(align=True)
            smooth_factor_row.prop(sc, "manual_smooth_count", text="Smooth Factor")
            srow = sbox.row(align=True)
            srow.operator("scene.gaussian_smooth_curves", icon='SMOOTHCURVE', text='Gaussian Smooth')

            # Test button section (single iteration) - HIDDEN
            # test_box = layout.box()
            # test_box.label(text="Test:")
            # test_row = test_box.row(align=True)
            # test_row.operator("pose.align_child_iterative_single", icon='PLAY', text="Single Step")

            # compact bake UI
            box2 = layout.box()
            box2.label(text="Bake Animation:")
            box2.prop(sc, "align_bake_method")

            # Threading button and workers in same row
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

            # advice about cancel
            box2.label(text="Cancel: Esc or Right-click")

            if sc.align_is_baking:
                total = (sc.frame_end - sc.frame_start + 1)
                box2.label(text=f"Progress: {sc.align_bake_progress}/{total}")

        else:
            layout.label(text="Select an armature", icon='INFO')

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
    SCENE_OT_make_larian_good,
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
