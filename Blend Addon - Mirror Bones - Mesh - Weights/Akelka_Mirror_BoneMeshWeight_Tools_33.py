bl_info = {
    "name": "Akelka Mirror Bone/Mesh Tools (Selection-aware + Mode-safe + WeightPaint-friendly) - Mirror ops replaced with v3 (selection-aware)",
    "author": "AkELkA (modified)",
    "version": (1, 0, 21),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Akelka Tools",
    "description": "Mirror bones, symmetrize mesh topology, mirror vertex weights. Works while armature is in Pose and while mesh is in Weight Paint (selection-aware). Preserves and restores armature/active object modes. (Mirror bone operators taken from v3, extended to support selected bones in Pose/Edit modes.)",
    "category": "Rigging",
}

import bpy
import bmesh
from mathutils import Vector, kdtree
from collections import deque, defaultdict

# --- Selection helpers (from v24) ---
def find_armature_in_selection(context):
    return next((o for o in context.selected_objects if o.type == 'ARMATURE'), None)

def find_armature_with_pose_selection():
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE':
            continue
        try:
            for pb in obj.pose.bones:
                bone = getattr(pb, "bone", None)
                sel_flags = (
                    getattr(pb, "select", False),
                    getattr(bone, "select", False) if bone is not None else False,
                    getattr(bone, "select_head", False) if bone is not None else False,
                    getattr(bone, "select_tail", False) if bone is not None else False,
                )
                if any(sel_flags):
                    return obj
        except Exception:
            continue
    return None

def find_armature_for_weights(context):
    arm = next((o for o in context.selected_objects if o.type == 'ARMATURE'), None)
    if arm:
        return arm
    arm = find_armature_with_pose_selection()
    if arm:
        return arm
    for o in bpy.data.objects:
        try:
            if o.type == 'ARMATURE' and getattr(o, "mode", None) == 'POSE':
                return o
        except Exception:
            continue
    return None

def find_mesh_for_weights(context):
    mesh = next((o for o in context.selected_objects if o.type == 'MESH'), None)
    if mesh:
        return mesh, False
    active = context.view_layer.objects.active
    if active and active.type == 'MESH':
        return active, True
    return None, False

def mesh_and_armature_available_for_weights(context):
    mesh_obj, _ = find_mesh_for_weights(context)
    arm_ok = find_armature_for_weights(context) is not None
    return (mesh_obj is not None) and arm_ok

# --- Properties (from v24) ---
def register_props():
    WM = bpy.types.WindowManager
    WM.akelka_mirror_direction = bpy.props.EnumProperty(
        name="Direction",
        items=[('L2R', "Left → Right", "Mirror left side onto right side"),
               ('R2L', "Right → Left", "Mirror right side onto left side")],
        default='L2R')
    WM.akelka_mirror_only_location = bpy.props.BoolProperty(
        name="Only Location",
        description="Mirror only bone head/tail positions (do not change roll/rotation/length)",
        default=True)

    # Single user-friendly float tweaker: 0.01 .. 1.0
    WM.akelka_mirror_tolerance = bpy.props.FloatProperty(
        name="Tolerance",
        description="Tolerance used when classifying vertices/bones for mirroring (0.01 - 1.0)",
        default=0.0002,
        min=0.0001,
        max=1.0,
        precision=6,
        step=0.01,
    )

def unregister_props():
    WM = bpy.types.WindowManager
    for p in ['akelka_mirror_direction', 'akelka_mirror_only_location',
              'akelka_mirror_tolerance']:
        if hasattr(WM, p):
            delattr(WM, p)

# --- Helpers for bones (keep mirror_bone_roll & get_opposite_name from v24) ---
def mirror_bone_roll(src_bone, tgt_bone):
    tgt_bone.roll = -src_bone.roll

def get_opposite_name(bone_name):
    """
    Find counterpart bone name using case-insensitive pattern matching.
    Supports: L_/R_ prefix, .L/.R suffix, _L_/_R_ middle, _L/_R ending
    """
    import re
    
    # Pattern 1: L_ or R_ at the start (case-insensitive)
    if re.match(r'^[lL]_', bone_name):
        return re.sub(r'^[lL]_', lambda m: 'R_' if m.group()[0].isupper() else 'r_', bone_name, count=1)
    if re.match(r'^[rR]_', bone_name):
        return re.sub(r'^[rR]_', lambda m: 'L_' if m.group()[0].isupper() else 'l_', bone_name, count=1)
    
    # Pattern 2: .L or .R at the end (case-insensitive)
    if re.search(r'\.[lL]$', bone_name):
        return re.sub(r'\.[lL]$', lambda m: '.R' if m.group()[1].isupper() else '.r', bone_name)
    if re.search(r'\.[rR]$', bone_name):
        return re.sub(r'\.[rR]$', lambda m: '.L' if m.group()[1].isupper() else '.l', bone_name)
    
    # Pattern 3: _L or _R at the end (case-insensitive)
    if re.search(r'_[lL]$', bone_name):
        return re.sub(r'_[lL]$', lambda m: '_R' if m.group()[1].isupper() else '_r', bone_name)
    if re.search(r'_[rR]$', bone_name):
        return re.sub(r'_[rR]$', lambda m: '_L' if m.group()[1].isupper() else '_l', bone_name)
    
    # Pattern 4: _L_ or _R_ in the middle (case-insensitive)
    if re.search(r'_[lL]_', bone_name):
        return re.sub(r'_[lL]_', lambda m: '_R_' if m.group()[1].isupper() else '_r_', bone_name, count=1)
    if re.search(r'_[rR]_', bone_name):
        return re.sub(r'_[rR]_', lambda m: '_L_' if m.group()[1].isupper() else '_l_', bone_name, count=1)
    
    return None

# --- Mirror bone operators (REPLACED: using implementations from version 3, extended) ---
class AKELKA_OT_mirror_bones_multiple(bpy.types.Operator):
    """Mirror selected bones by spatial symmetry — works in Edit or Pose mode (selection-aware)."""
    bl_idname = "akelka.mirror_bones_multiple"
    bl_label = "Mirror Bones (Selection-aware v3+)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mode = context.mode
        if mode not in {'EDIT_ARMATURE', 'POSE'}:
            return False
        ob = context.object
        if not ob or ob.type != 'ARMATURE':
            return False
        if mode == 'EDIT_ARMATURE':
            return len(context.selected_editable_bones) >= 1
        if mode == 'POSE':
            return len(context.selected_pose_bones) >= 1
        return False

    def execute(self, context):
        wm = context.window_manager
        arm_obj = context.object
        original_mode = context.mode
        original_active = context.view_layer.objects.active

        # gather selected bone names depending on mode
        selected_names = []
        if original_mode == 'POSE':
            selected_names = [pb.name for pb in context.selected_pose_bones]
        elif original_mode == 'EDIT_ARMATURE':
            selected_names = [b.name for b in context.selected_editable_bones]

        # Ensure we are in EDIT mode to modify bones
        try:
            if context.object and context.object.type == 'ARMATURE':
                context.view_layer.objects.active = arm_obj
                if arm_obj.mode != 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass

        ebones = arm_obj.data.edit_bones

        eps = getattr(wm, 'akelka_mirror_tolerance', 0.0002)
        dir_setting = getattr(wm, 'akelka_mirror_direction', 'L2R')

        # if selected_names exists -> operate only on those bones (mirror them)
        if selected_names:
            # For each selected bone, find counterpart and mirror
            for name in selected_names:
                if name not in ebones:
                    continue
                src = ebones[name]

                # Check if bone is on centerline (has no L/R side)
                is_center_bone = abs(src.head.x) <= eps

                # TRY NAME-BASED MATCHING FIRST (more reliable for named bones)
                tgt = None
                opp = get_opposite_name(src.name)
                if opp and opp in ebones:
                    tgt = ebones[opp]

                # If name matching fails, fall back to spatial matching (but not for center bones)
                if (tgt is None or tgt.name == src.name) and not is_center_bone:
                    mirrored_head = Vector((-src.head.x, src.head.y, src.head.z))

                    # build kd tree for all edit bones heads
                    all_eb = list(ebones)
                    kd = kdtree.KDTree(len(all_eb))
                    for i, b in enumerate(all_eb):
                        kd.insert(b.head, i)
                    kd.balance()

                    try:
                        _, idx, _ = kd.find(mirrored_head)
                        tgt = all_eb[idx]
                    except Exception:
                        tgt = None

                    # As a last resort, search nearest excluding the source itself
                    if tgt is None or tgt.name == src.name:
                        best = None
                        best_d = None
                        for b in ebones:
                            if b.name == src.name:
                                continue
                            d = (b.head - mirrored_head).length_squared
                            if best_d is None or d < best_d:
                                best_d = d
                                best = b
                        if best is not None:
                            tgt = best

                # Skip if no valid counterpart found (especially for center bones)
                if not tgt or tgt.name == src.name:
                    if is_center_bone:
                        # Silently skip center bones with no counterpart
                        continue
                    else:
                        # Warn about non-center bones
                        self.report({'WARNING'}, f"No counterpart found for bone '{src.name}'")
                        continue

                # Decide source/target considering direction
                is_source_left = (src.head.x < -eps)
                is_source_right = (src.head.x > eps)
                if dir_setting == 'L2R':
                    # only mirror left->right or if selected bone is on right side mirror its counterpart
                    should_src_be = is_source_left
                else:
                    should_src_be = is_source_right

                # If user selected a bone on the "target" side, we will flip roles so selected bone becomes target
                if not should_src_be:
                    # swap roles: use tgt as source, src as target
                    real_src = tgt
                    real_tgt = src
                else:
                    real_src = src
                    real_tgt = tgt

                if getattr(wm, 'akelka_mirror_only_location', False):
                    old_head = real_tgt.head.copy()
                    old_tail = real_tgt.tail.copy()
                    offset = old_tail - old_head
                    new_head = Vector((-real_src.head.x, real_src.head.y, real_src.head.z))
                    real_tgt.head = new_head
                    real_tgt.tail = new_head + offset
                else:
                    real_tgt.head = Vector((-real_src.head.x, real_src.head.y, real_src.head.z))
                    real_tgt.tail = Vector((-real_src.tail.x, real_src.tail.y, real_src.tail.z))
                    try:
                        mirror_bone_roll(real_src, real_tgt)
                    except Exception:
                        pass

            # restore original mode
            try:
                if original_mode and getattr(arm_obj, "mode", None) != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass

            if original_active:
                try:
                    context.view_layer.objects.active = original_active
                except Exception:
                    pass

            return {'FINISHED'}

        # Otherwise (no selection), fall back to spatial batch mirror (like original)
        # select all edit bones set and run KD matching
        try:
            left = [b for b in ebones if b.head.x < 0]
            right = [b for b in ebones if b.head.x >= 0]
            sources, targets = (left, right) if dir_setting == 'L2R' else (right, left)

            if not sources or not targets:
                self.report({'ERROR'}, "Need bones on both sides for mirroring")
                # restore
                try:
                    if original_mode and getattr(arm_obj, "mode", None) != original_mode:
                        bpy.ops.object.mode_set(mode=original_mode)
                except Exception:
                    pass
                if original_active:
                    try:
                        context.view_layer.objects.active = original_active
                    except Exception:
                        pass
                return {'CANCELLED'}

            kd = kdtree.KDTree(len(targets))
            for i, b in enumerate(targets):
                kd.insert(b.head, i)
            kd.balance()

            for src in sources:
                mirrored_head = Vector((-src.head.x, src.head.y, src.head.z))
                _, idx, _ = kd.find(mirrored_head)
                tgt = targets[idx]

                if getattr(wm, 'akelka_mirror_only_location', False):
                    old_head = tgt.head.copy()
                    old_tail = tgt.tail.copy()
                    offset = old_tail - old_head
                    tgt.head = mirrored_head
                    tgt.tail = mirrored_head + offset
                else:
                    tgt.head = mirrored_head
                    tgt.tail = Vector((-src.tail.x, src.tail.y, src.tail.z))
                    try:
                        mirror_bone_roll(src, tgt)
                    except Exception:
                        pass
        finally:
            # restore original mode
            try:
                if original_mode and getattr(arm_obj, "mode", None) != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass
            if original_active:
                try:
                    context.view_layer.objects.active = original_active
                except Exception:
                    pass

        return {'FINISHED'}


class AKELKA_OT_mirror_bones_from_object_mode(bpy.types.Operator):
    """Mirror bones by spatial symmetry — works from Object mode but also accepts Pose/Edit contexts; selection-aware."""
    bl_idname = "akelka.mirror_bones_from_object_mode"
    bl_label = "Mirror Bones (Selection-aware v3+)"  # <-- label changed to match the second button
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # allow if there's an armature selected or pose selection exists; accept OBJECT/POSE/EDIT_ARMATURE modes
        if context.mode not in {'OBJECT', 'POSE', 'EDIT_ARMATURE'}:
            return False
        if any(o.type == 'ARMATURE' for o in context.selected_objects):
            return True
        # also accept case where current object is armature with pose selection
        ob = context.object
        if ob and ob.type == 'ARMATURE':
            # if in pose/edit, require at least one selected bone
            if context.mode == 'POSE' and len(context.selected_pose_bones) >= 1:
                return True
            if context.mode == 'EDIT_ARMATURE' and len(context.selected_editable_bones) >= 1:
                return True
            # in object mode, it's OK if armature present
            if context.mode == 'OBJECT':
                return True
        return False

    def execute(self, context):
        wm = context.window_manager

        # determine armature object: prefer selected armature, otherwise active if armature
        arm_obj = next((o for o in context.selected_objects if o.type == 'ARMATURE'), None)
        if arm_obj is None and context.object and context.object.type == 'ARMATURE':
            arm_obj = context.object

        if arm_obj is None:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        original_active = context.view_layer.objects.active
        original_mode = None
        try:
            original_mode = arm_obj.mode
        except Exception:
            original_mode = context.mode

        # If user is in POSE mode and selected bones are present, we want to honor selected bones.
        selected_pose_names = []
        try:
            if context.mode == 'POSE' and context.object == arm_obj:
                selected_pose_names = [pb.name for pb in context.selected_pose_bones]
        except Exception:
            selected_pose_names = []

        # Activate armature
        try:
            context.view_layer.objects.active = arm_obj
        except Exception:
            pass

        # Switch to edit mode for modifications
        try:
            if arm_obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass

        ebones = arm_obj.data.edit_bones

        # If there are selected bones (pose or edit), mirror only those (selection-aware)
        sel_edit_names = []
        # if we had pose selection earlier, map names to edit bone names
        if selected_pose_names:
            for n in selected_pose_names:
                if n in ebones:
                    sel_edit_names.append(n)
        else:
            # fallback: see if user had edit selection
            try:
                sel_edit_names = [b.name for b in ebones if b.select]
            except Exception:
                sel_edit_names = []

        wm_dir = getattr(wm, 'akelka_mirror_direction', 'L2R')
        eps = getattr(wm, 'akelka_mirror_tolerance', 0.0002)

        if sel_edit_names:
            # mirror for each selected edit bone
            all_eb = list(ebones)
            kd_all = kdtree.KDTree(len(all_eb))
            for i, b in enumerate(all_eb):
                kd_all.insert(b.head, i)
            kd_all.balance()

            for name in sel_edit_names:
                if name not in ebones:
                    continue
                src = ebones[name]
                
                # Check if bone is on centerline (has no L/R side)
                is_center_bone = abs(src.head.x) <= eps
                
                # TRY NAME-BASED MATCHING FIRST (more reliable for named bones)
                tgt = None
                opp = get_opposite_name(src.name)
                if opp and opp in ebones:
                    tgt = ebones[opp]

                # If name matching fails, fall back to spatial matching (but not for center bones)
                if (tgt is None or tgt.name == src.name) and not is_center_bone:
                    mirrored_head = Vector((-src.head.x, src.head.y, src.head.z))

                    # find nearest bone head
                    try:
                        _, idx, _ = kd_all.find(mirrored_head)
                        tgt = all_eb[idx]
                    except Exception:
                        tgt = None

                    if tgt is None or tgt.name == src.name:
                        best = None
                        best_d = None
                        for b in ebones:
                            if b.name == src.name:
                                continue
                            d = (b.head - mirrored_head).length_squared
                            if best_d is None or d < best_d:
                                best_d = d
                                best = b
                        if best is not None:
                            tgt = best

                # Skip if no valid counterpart found (especially for center bones)
                if not tgt or tgt.name == src.name:
                    if is_center_bone:
                        # Silently skip center bones with no counterpart
                        continue
                    else:
                        # Warn about non-center bones
                        self.report({'WARNING'}, f"No counterpart found for bone '{src.name}'")
                        continue

                # direction logic (left/right)
                is_src_left = (src.head.x < -eps)
                is_src_right = (src.head.x > eps)
                if wm_dir == 'L2R':
                    should_src_be = is_src_left
                else:
                    should_src_be = is_src_right

                if not should_src_be:
                    real_src = tgt
                    real_tgt = src
                else:
                    real_src = src
                    real_tgt = tgt

                if getattr(wm, 'akelka_mirror_only_location', False):
                    old_head = real_tgt.head.copy()
                    old_tail = real_tgt.tail.copy()
                    offset = old_tail - old_head
                    new_head = Vector((-real_src.head.x, real_src.head.y, real_src.head.z))
                    real_tgt.head = new_head
                    real_tgt.tail = new_head + offset
                else:
                    real_tgt.head = Vector((-real_src.head.x, real_src.head.y, real_src.head.z))
                    real_tgt.tail = Vector((-real_src.tail.x, real_src.tail.y, real_src.tail.z))
                    try:
                        mirror_bone_roll(real_src, real_tgt)
                    except Exception:
                        pass

            # restore mode and active
            try:
                if original_mode and getattr(arm_obj, "mode", None) != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass
            if original_active:
                try:
                    context.view_layer.objects.active = original_active
                except Exception:
                    pass
            return {'FINISHED'}

        # If no selection, do the spatial full-set approach
        try:
            left = [b for b in ebones if b.head.x < 0]
            right = [b for b in ebones if b.head.x >= 0]
            sources, targets = (left, right) if wm_dir == 'L2R' else (right, left)

            if not sources or not targets:
                self.report({'ERROR'}, "Need bones on both sides for mirroring")
                try:
                    if original_mode and getattr(arm_obj, "mode", None) != original_mode:
                        bpy.ops.object.mode_set(mode=original_mode)
                except Exception:
                    pass
                if original_active:
                    try:
                        context.view_layer.objects.active = original_active
                    except Exception:
                        pass
                return {'CANCELLED'}

            kd = kdtree.KDTree(len(targets))
            for i, b in enumerate(targets):
                kd.insert(b.head, i)
            kd.balance()

            for src in sources:
                mirrored_head = Vector((-src.head.x, src.head.y, src.head.z))
                _, idx, _ = kd.find(mirrored_head)
                tgt = targets[idx]

                if getattr(wm, 'akelka_mirror_only_location', False):
                    old_head = tgt.head.copy()
                    old_tail = tgt.tail.copy()
                    offset = old_tail - old_head
                    tgt.head = mirrored_head
                    tgt.tail = mirrored_head + offset
                else:
                    tgt.head = mirrored_head
                    tgt.tail = Vector((-src.tail.x, src.tail.y, src.tail.z))
                    try:
                        mirror_bone_roll(src, tgt)
                    except Exception:
                        pass
        finally:
            try:
                if original_mode and getattr(arm_obj, "mode", None) != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass
            if original_active:
                try:
                    context.view_layer.objects.active = original_active
                except Exception:
                    pass

        self.report({'INFO'}, f"Mirrored bones (object-mode style) on armature '{arm_obj.name}'")
        return {'FINISHED'}


# --- Topology-based helpers (from v24) ---
def get_axis_index(axis: str):
    return {'X': 0, 'Y': 1, 'Z': 2}[axis.upper()]

def build_adjacency(bm):
    bm.verts.ensure_lookup_table()
    adj = {v.index: set() for v in bm.verts}
    for e in bm.edges:
        a = e.verts[0].index
        b = e.verts[1].index
        adj[a].add(b)
        adj[b].add(a)
    return adj

def classify_vertices_by_side(bm, axis_idx, center, eps):
    left = set()
    right = set()
    middle = set()
    for v in bm.verts:
        coord = v.co[axis_idx]
        if coord > center + eps:
            right.add(v.index)
        elif coord < center - eps:
            left.add(v.index)
        else:
            middle.add(v.index)
    return left, right, middle

def wl_iterative_labels(adj, iters=4):
    labels = {v: len(neigh) for v, neigh in adj.items()}
    for it in range(iters):
        map_key_to_id = {}
        next_id = 1
        new_labels = {}
        for v in adj:
            neigh_labels = tuple(sorted(labels[n] for n in adj[v]))
            key = (labels[v], neigh_labels)
            if key not in map_key_to_id:
                map_key_to_id[key] = next_id
                next_id += 1
            new_labels[v] = map_key_to_id[key]
        labels = new_labels
    return labels

def bfs_level_signature(adj, start, max_depth=6):
    visited = {start}
    q = deque([(start, 0)])
    level_map = defaultdict(list)
    while q:
        v, d = q.popleft()
        level_map[d].append(v)
        if d >= max_depth:
            continue
        for n in adj[v]:
            if n not in visited:
                visited.add(n)
                q.append((n, d+1))
    sig = []
    for d in range(max_depth+1):
        if d in level_map:
            degrees = tuple(sorted(len(adj[w]) for w in level_map[d]))
            sig.append(degrees)
        else:
            sig.append(tuple())
    return tuple(sig)

def mirror_coordinate(co: Vector, axis_idx: int, center: float):
    new = Vector(co)
    new[axis_idx] = 2*center - co[axis_idx]
    return new

# --- Symmetrize mesh operator (from v24) ---
class AKELKA_OT_symmetrize_mesh(bpy.types.Operator):
    bl_idname = "akelka.symmetrize_mesh"
    bl_label = "Symmetrize Mesh (Topology X)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return ((context.mode == 'EDIT_MESH' and context.object and context.object.type == 'MESH')
                or any(o.type == 'MESH' for o in context.selected_objects))

    def execute(self, context):
        wm = context.window_manager
        axis = 'X'
        center = 0.0
        eps = getattr(wm, 'akelka_mirror_tolerance', 0.01)
        iterations = 4
        max_bfs_depth = 6
        dry_run = False
        use_nearest_tiebreak = True

        dir_setting = getattr(wm, 'akelka_mirror_direction', 'L2R')
        source_side = 'NEG' if dir_setting == 'L2R' else 'POS'

        obj = context.object
        if not (context.mode == 'EDIT_MESH' and obj and obj.type == 'MESH'):
            obj = next((o for o in context.selected_objects if o.type == 'MESH'), None)
            if not obj:
                self.report({'ERROR'}, "No mesh selected for symmetrize")
                return {'CANCELLED'}

        in_edit = (context.mode == 'EDIT_MESH' and context.object and context.object.type == 'MESH' and context.object == obj)

        if in_edit:
            bm = bmesh.from_edit_mesh(obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)

        bm.verts.ensure_lookup_table()
        axis_idx = get_axis_index(axis)
        left, right, middle = classify_vertices_by_side(bm, axis_idx, center, eps)

        if source_side == 'POS':
            source_set = right
            target_set = left
        else:
            source_set = left
            target_set = right

        if not source_set:
            self.report({'ERROR'}, "No vertices found on source side (check center/axis/eps)")
            if not in_edit:
                bm.free()
            return {'CANCELLED'}
        if not target_set:
            self.report({'ERROR'}, "No vertices found on target side (check center/axis/eps)")
            if not in_edit:
                bm.free()
            return {'CANCELLED'}

        adj = build_adjacency(bm)
        labels = wl_iterative_labels(adj, iters=iterations)
        label_to_targets = defaultdict(list)
        for t in target_set:
            label_to_targets[labels[t]].append(t)

        bfs_sig_cache = {}
        for idx in list(source_set) + list(target_set):
            bfs_sig_cache[idx] = bfs_level_signature(adj, idx, max_depth=max_bfs_depth)

        mapping = {}
        ambiguous = []
        unmatched_source = []

        for s in source_set:
            label = labels[s]
            candidates = label_to_targets.get(label, [])
            if not candidates:
                unmatched_source.append(s)
                continue
            if len(candidates) == 1:
                mapping[candidates[0]] = s
                continue
            s_sig = bfs_sig_cache[s]
            exact = [c for c in candidates if bfs_sig_cache[c] == s_sig]
            if len(exact) == 1:
                mapping[exact[0]] = s
                continue
            if len(exact) > 1:
                ambiguous.append((s, exact))
                continue
            scored = []
            for c in candidates:
                score = 0
                s_sig2 = s_sig
                c_sig2 = bfs_sig_cache[c]
                for level in range(min(len(s_sig2), len(c_sig2))):
                    if s_sig2[level] == c_sig2[level]:
                        score += 1
                    else:
                        break
                scored.append((score, c))
            scored.sort(reverse=True)
            best_score = scored[0][0]
            top = [c for sc, c in scored if sc == best_score]
            if len(top) == 1:
                mapping[top[0]] = s
            else:
                ambiguous.append((s, top))

        if use_nearest_tiebreak:
            for s, cand_list in list(ambiguous):
                s_co = bm.verts[s].co
                mirror_s = mirror_coordinate(s_co, axis_idx, center)
                best_c = None
                best_d = None
                for c in cand_list:
                    c_co = bm.verts[c].co
                    d = (c_co - mirror_s).length_squared
                    if best_d is None or d < best_d:
                        best_d = d
                        best_c = c
                if best_c is not None:
                    mapping[best_c] = s
                    try:
                        ambiguous.remove((s, cand_list))
                    except ValueError:
                        pass

        for s, cand_list in ambiguous:
            chosen = cand_list[0]
            mapping[chosen] = s

        matched_targets = set(mapping.keys())
        moved = 0
        unmapped_targets = target_set - matched_targets

        if not dry_run:
            for t_idx, s_idx in mapping.items():
                s_co = bm.verts[s_idx].co
                new_co = mirror_coordinate(s_co, axis_idx, center)
                bm.verts[t_idx].co = new_co
                moved += 1

            if in_edit:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            else:
                bm.to_mesh(obj.data)
                obj.data.update()
                bm.free()

        self.report({'INFO'}, f"Source verts: {len(source_set)}, target verts: {len(target_set)}, mapped: {len(mapping)}, moved: {moved}, unmapped targets: {len(unmapped_targets)}")
        if unmapped_targets:
            print("Unmapped target verts (indices):", sorted(list(unmapped_targets)))
        return {'FINISHED'}

# --- Mirror Weights operator with saving/restoring non-selected groups (from v24) ---
class AKELKA_OT_mirror_weights(bpy.types.Operator):
    bl_idname = "akelka.mirror_weights"
    bl_label = "Mirror Weights (Topology X)"
    bl_options = {'REGISTER', 'UNDO'}

    clear_mode: bpy.props.EnumProperty(
        name="Clear Target Weights",
        description="What to do with existing weights in target vertex groups before/after copying",
        items=[
            ('NONE', "Keep (default)", "Do not clear any existing weights"),
            ('CLEAR_ALL', "Clear all", "Remove all vertex assignments from the target groups before copying"),
            ('ZERO_UNMAPPED', "Clear unmapped", "Remove weights from unmapped target-side vertices after copying")
        ],
        default='CLEAR_ALL'
    )

    @classmethod
    def poll(cls, context):
        mesh_obj, _ = find_mesh_for_weights(context)
        arm_ok = find_armature_for_weights(context) is not None
        return (mesh_obj is not None) and arm_ok

    def execute(self, context):
        wm = context.window_manager
        dir_setting = getattr(wm, 'akelka_mirror_direction', 'L2R')
        axis = 'X'
        center = 0.0
        # use user-exposed tolerance directly
        eps = getattr(wm, 'akelka_mirror_tolerance', 0.01)

        iterations = 4
        max_bfs_depth = 6
        use_nearest_tiebreak = True

        arm_obj = find_armature_for_weights(context)
        mesh_obj, used_active = find_mesh_for_weights(context)
        if not arm_obj or not mesh_obj:
            self.report({'ERROR'}, "Select an armature (or select pose bones) and a mesh object together")
            return {'CANCELLED'}

        if used_active:
            self.report({'INFO'}, f"Using active mesh '{mesh_obj.name}' (it was not selected).")

        original_active = context.view_layer.objects.active
        original_active_mode = None
        if original_active:
            try:
                original_active_mode = original_active.mode
            except Exception:
                original_active_mode = None
        arm_obj_mode = None
        try:
            arm_obj_mode = arm_obj.mode
        except Exception:
            arm_obj_mode = None

        # --- build bone lists and KD trees once ---
        sel_pose_names = []
        try:
            try:
                if hasattr(arm_obj, "pose") and arm_obj.mode == 'POSE':
                    sel_pose_names = [pb.name for pb in arm_obj.pose.bones if getattr(pb.bone, "select", False)]
            except Exception:
                sel_pose_names = []

            try:
                context.view_layer.objects.active = arm_obj
            except Exception:
                pass

            try:
                if getattr(arm_obj, "mode", None) != 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass

            ebones = arm_obj.data.edit_bones

            left_bones = [b for b in ebones if b.head.x < -eps]
            right_bones = [b for b in ebones if b.head.x > eps]

            if dir_setting == 'L2R':
                source_bones = left_bones
                target_bones = right_bones
            else:
                source_bones = right_bones
                target_bones = left_bones

            if not source_bones or not target_bones:
                self.report({'ERROR'}, "Need bones on both sides for mirroring weights")
                return {'CANCELLED'}

            kd_target = kdtree.KDTree(len(target_bones))
            for i, b in enumerate(target_bones):
                kd_target.insert(b.head, i)
            kd_target.balance()

            all_ebones = list(ebones)
            kd_all = kdtree.KDTree(len(all_ebones))
            for i, b in enumerate(all_ebones):
                kd_all.insert(b.head, i)
            kd_all.balance()

            bone_pairs_for_selected = []
            if sel_pose_names:
                pairs_set = set()
                if dir_setting == 'L2R':
                    is_source_coord = (lambda x: x < -eps)
                else:
                    is_source_coord = (lambda x: x > eps)

                for sel_name in sel_pose_names:
                    sel_eb = ebones.get(sel_name) if hasattr(ebones, "get") else (ebones[sel_name] if sel_name in ebones else None)
                    if sel_eb is None:
                        continue

                    # TRY NAME-BASED MATCHING FIRST (more reliable for named bones)
                    tgt_eb = None
                    opp_name = get_opposite_name(sel_eb.name)
                    if opp_name and opp_name in ebones:
                        tgt_eb = ebones[opp_name]

                    # If name matching fails, fall back to spatial matching
                    if tgt_eb is None:
                        mirrored_head = Vector((-sel_eb.head.x, sel_eb.head.y, sel_eb.head.z))
                        try:
                            _, idx, _ = kd_all.find(mirrored_head)
                            tgt_eb = all_ebones[idx]
                        except Exception:
                            tgt_eb = None

                    # Last resort: find nearest bone
                    if tgt_eb is None:
                        mirrored_head = Vector((-sel_eb.head.x, sel_eb.head.y, sel_eb.head.z))
                        best = None
                        best_d = None
                        for b in ebones:
                            d = (b.head - mirrored_head).length_squared
                            if best_d is None or d < best_d:
                                best_d = d
                                best = b
                        if best is not None and best_d < 10000.0:
                            tgt_eb = best

                    if tgt_eb is None:
                        continue

                    if is_source_coord(sel_eb.head.x):
                        src_name = sel_eb.name
                        tgt_name = tgt_eb.name
                    else:
                        src_name = tgt_eb.name
                        tgt_name = sel_eb.name

                    pairs_set.add((src_name, tgt_name))

                bone_pairs_for_selected = list(pairs_set)

            bone_pairs_all = []
            for src_bone in source_bones:
                # TRY NAME-BASED MATCHING FIRST (more reliable for named bones)
                opp = get_opposite_name(src_bone.name)
                if opp and opp in ebones:
                    bone_pairs_all.append((src_bone.name, opp))
                else:
                    # Fall back to spatial matching
                    mirrored_head = Vector((-src_bone.head.x, src_bone.head.y, src_bone.head.z))
                    try:
                        _, idx, _ = kd_target.find(mirrored_head)
                        tgt_bone = target_bones[idx]
                        bone_pairs_all.append((src_bone.name, tgt_bone.name))
                    except Exception:
                        # Last resort: find nearest bone
                        best = None
                        best_d = None
                        for b in all_ebones:
                            d = (b.head - mirrored_head).length_squared
                            if best_d is None or d < best_d:
                                best_d = d
                                best = b
                        if best is not None:
                            bone_pairs_all.append((src_bone.name, best.name))
        finally:
            try:
                if arm_obj:
                    context.view_layer.objects.active = arm_obj
                    if arm_obj_mode and getattr(arm_obj, "mode", None) != arm_obj_mode:
                        try:
                            bpy.ops.object.mode_set(mode=arm_obj_mode)
                        except Exception:
                            try:
                                if arm_obj_mode == 'POSE':
                                    bpy.ops.object.mode_set(mode='POSE')
                                elif arm_obj_mode == 'EDIT':
                                    bpy.ops.object.mode_set(mode='EDIT')
                            except Exception:
                                pass
            except Exception:
                pass
            if original_active:
                try:
                    context.view_layer.objects.active = original_active
                    if original_active_mode and getattr(context.view_layer.objects.active, "mode", None) != original_active_mode:
                        try:
                            bpy.ops.object.mode_set(mode=original_active_mode)
                        except Exception:
                            pass
                except Exception:
                    pass

        # --- build vertex mapping for ALL vertices (including center vertices) ---
        in_edit_mesh = (mesh_obj == context.object and context.mode == 'EDIT_MESH')
        if in_edit_mesh:
            bm = bmesh.from_edit_mesh(mesh_obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh_obj.data)

        bm.verts.ensure_lookup_table()
        axis_idx = get_axis_index(axis)
        vert_coords_x = [v.co[axis_idx] for v in bm.verts]

        left_v, right_v, middle_v = classify_vertices_by_side(bm, axis_idx, center, eps)
        all_verts = set(range(len(bm.verts)))

        adj = build_adjacency(bm)
        labels = wl_iterative_labels(adj, iters=iterations)

        bfs_sig_cache = {}
        for idx in all_verts:
            bfs_sig_cache[idx] = bfs_level_signature(adj, idx, max_depth=max_bfs_depth)

        def create_full_vertex_mapping():
            mapping = {}
            label_to_verts = defaultdict(list)
            for v_idx in all_verts:
                label_to_verts[labels[v_idx]].append(v_idx)

            for v_idx in all_verts:
                v_co = bm.verts[v_idx].co
                mirror_co = mirror_coordinate(v_co, axis_idx, center)

                if v_idx in middle_v:
                    mapping[v_idx] = v_idx
                    continue

                label = labels[v_idx]
                candidates = [c for c in label_to_verts[label] if c != v_idx]

                if not candidates:
                    candidates = [c for c in all_verts if c != v_idx]

                if not candidates:
                    continue

                v_sig = bfs_sig_cache[v_idx]
                exact_matches = [c for c in candidates if bfs_sig_cache[c] == v_sig]
                if exact_matches:
                    candidates = exact_matches

                best_candidate = None
                best_distance = None
                for c in candidates:
                    c_co = bm.verts[c].co
                    dist = (c_co - mirror_co).length_squared
                    if best_distance is None or dist < best_distance:
                        best_distance = dist
                        best_candidate = c

                if best_candidate is not None:
                    mapping[v_idx] = best_candidate

            return mapping

        vertex_mapping = create_full_vertex_mapping()

        if not in_edit_mesh:
            bm.free()

        eps_auto = max(eps, 1e-4)
        middle_v_final = set(i for i, x in enumerate(vert_coords_x) if abs(x - center) <= eps_auto)
        if not middle_v_final:
            middle_v_final = set(middle_v)

        # --- perform weight copy (but optionally save and restore groups not related to selection) ---
        mesh = mesh_obj
        try:
            context.view_layer.objects.active = mesh
        except Exception:
            pass
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        vg_cache = {g.name: g for g in mesh.vertex_groups}
        all_vertex_indices = [v.index for v in mesh.data.vertices]

        # -------------------------
        # CHANGED BEHAVIOR STARTS HERE:
        # If pose bones are selected we DO NOT run the original "pose-mode special mirroring".
        # Instead: run the object-mode style mirroring (full bone_pairs_all) but restrict which
        # vertex groups are allowed to be changed: only the selected bone(s) and their counterpart(s)
        # are allowed — everything else is saved and restored at the end.
        # -------------------------
        if sel_pose_names:
            # Force object-mode style run: use full bone pair list
            active_pairs = bone_pairs_all.copy()

            # Build allowed_groups as only the selected bones and their matched counterparts
            allowed_groups = set()
            # For each selected bone, add matched pairs from bone_pairs_all
            for s in sel_pose_names:
                for a, b in bone_pairs_all:
                    if a == s or b == s:
                        allowed_groups.add(a)
                        allowed_groups.add(b)
                # Also try name-based opposite if no spatial pair found
                opp = get_opposite_name(s)
                if opp:
                    allowed_groups.add(opp)
                allowed_groups.add(s)

        else:
            # normal (no pose-selection) behavior: operate on full set of vertex groups that correspond to bones
            active_pairs = bone_pairs_all
            arm_bone_names = {b.name for b in arm_obj.data.bones}
            allowed_groups = set(n for n in vg_cache.keys() if n in arm_bone_names)
        # -------------------------
        # CHANGED BEHAVIOR ENDS HERE
        # -------------------------

        vg_cache = {g.name: g for g in mesh.vertex_groups}
        arm_bone_names = {b.name for b in arm_obj.data.bones}

        # groups_to_save = groups we will restore later (everything except allowed_groups)
        all_vg_names = set(vg_cache.keys())
        groups_to_save = sorted(list(all_vg_names - set(allowed_groups)))

        saved_group_weights = {}
        if groups_to_save:
            for gname in groups_to_save:
                g = vg_cache.get(gname)
                if g is None:
                    continue
                gidx = g.index
                weights = {}
                for v in mesh.data.vertices:
                    for gr in v.groups:
                        if gr.group == gidx:
                            if gr.weight != 0.0:
                                weights[v.index] = gr.weight
                            break
                saved_group_weights[gname] = weights

        # perform copy for active_pairs (which is bone_pairs_all normally, or full when sel_pose_names present)
        for src_bone, tgt_bone in active_pairs:
            src_vg = vg_cache.get(src_bone)
            if src_vg is None:
                continue

            tgt_vg = vg_cache.get(tgt_bone)
            if tgt_vg is None:
                tgt_vg = mesh.vertex_groups.new(name=tgt_bone)
                vg_cache[tgt_bone] = tgt_vg

            if self.clear_mode == 'CLEAR_ALL':
                try:
                    tgt_vg.remove(all_vertex_indices)
                except Exception:
                    pass

            src_idx = src_vg.index
            tgt_idx = tgt_vg.index

            for src_vert_idx, tgt_vert_idx in vertex_mapping.items():
                w = 0.0
                for g in mesh.data.vertices[src_vert_idx].groups:
                    if g.group == src_idx:
                        w = g.weight
                        break

                if w > 0.0:
                    tgt_vg.add([tgt_vert_idx], w, 'REPLACE')
                else:
                    # Remove vertex from target group if source has no weight (or is not in group)
                    try:
                        tgt_vg.remove([tgt_vert_idx])
                    except Exception:
                        pass

        if middle_v_final:
            for mid_idx in middle_v_final:
                v = mesh.data.vertices[mid_idx]
                for src_bone, tgt_bone in active_pairs:
                    src_vg = vg_cache.get(src_bone)
                    if src_vg is None:
                        continue
                    tgt_vg = vg_cache.get(tgt_bone)
                    if tgt_vg is None:
                        tgt_vg = mesh.vertex_groups.new(name=tgt_bone)
                        vg_cache[tgt_bone] = tgt_vg

                    w = 0.0
                    for g in v.groups:
                        if g.group == src_vg.index:
                            w = g.weight
                            break

                    cur_w = 0.0
                    for g in v.groups:
                        if g.group == tgt_vg.index:
                            cur_w = g.weight
                            break

                    if abs(cur_w - w) > 1e-6:
                        try:
                            if w > 0.0:
                                tgt_vg.add([mid_idx], w, 'REPLACE')
                            else:
                                try:
                                    tgt_vg.remove([mid_idx])
                                except Exception:
                                    pass
                        except Exception:
                            pass

        # ----------------- BEGIN: v12-style self-mirror for "solo" groups -----------------
        # (detect vertex-groups that have no matched counterpart and self-mirror them)
        try:
            # helper to compute total weight for a VG on this mesh
            def total_group_weight_on_mesh(group_name):
                g = vg_cache.get(group_name)
                if g is None:
                    return 0.0
                total = 0.0
                g_index = g.index
                for v in mesh.data.vertices:
                    for gr in v.groups:
                        if gr.group == g_index:
                            total += gr.weight
                            break
                return total

            # Build matched bone name set from spatial pairing (bone_pairs_all)
            matched_bone_names = set()
            for a, b in bone_pairs_all:
                matched_bone_names.add(a)
                matched_bone_names.add(b)

            arm_bone_names = {b.name for b in arm_obj.data.bones}

            # 1) Groups whose bone exists but that bone wasn't in matched pairs
            groups_with_no_matched_bone = [gname for gname in vg_cache.keys()
                                          if gname in arm_bone_names and gname not in matched_bone_names]

            # 2) Groups where one side of the pair is missing as a VG
            groups_missing_counterparty = []
            for src_bone, tgt_bone in bone_pairs_all:
                src_exists = src_bone in vg_cache
                tgt_exists = tgt_bone in vg_cache
                if src_exists and not tgt_exists:
                    groups_missing_counterparty.append((src_bone, tgt_bone, 'src_missing_tgt'))
                if tgt_exists and not src_exists:
                    groups_missing_counterparty.append((tgt_bone, src_bone, 'tgt_missing_src'))

            # Combine into solo candidates
            solo_candidate_names = set(groups_with_no_matched_bone)
            for existing_group, missing_counter, _ in groups_missing_counterparty:
                solo_candidate_names.add(existing_group)

            # Filter to existing VGs with significant weight AND that are in allowed_groups (selected bones)
            solo_to_mirror = []
            for name in sorted(solo_candidate_names):
                if name in vg_cache and name in allowed_groups:
                    tw = total_group_weight_on_mesh(name)
                    if tw > 1e-8:
                        solo_to_mirror.append(name)

            # If we saved weights for groups_to_save earlier, prevent overwriting these solo groups:
            # remove them from saved_group_weights (so the restore step won't clobber the self-mirror)
            if 'saved_group_weights' in locals() and saved_group_weights:
                for name in list(solo_to_mirror):
                    if name in saved_group_weights:
                        del saved_group_weights[name]

            # Ensure we have vert_coords_x and eps_auto / is_source_vert locally
            try:
                _ = vert_coords_x  # already exists from mapping stage
            except Exception:
                # fallback: compute X coords directly from mesh (axis X assumed)
                vert_coords_x = [v.co[0] for v in mesh.data.vertices]

            try:
                _ = eps_auto
            except Exception:
                eps_auto = eps  # fallback to user tolerance if eps_auto not present

            is_source_vert = (lambda x: x < -eps_auto) if dir_setting == 'L2R' else (lambda x: x > eps_auto)

            # Perform self-mirroring (copy inside same VG from source-side verts -> mirrored verts)
            self_mirror_assignments = 0
            if solo_to_mirror:
                for vg_name in solo_to_mirror:
                    vg = vg_cache.get(vg_name)
                    if vg is None:
                        continue
                    vg_index = vg.index

                    # copy for mapped source->target verts (only source-side verts)
                    for src_idx, tgt_idx in vertex_mapping.items():
                        src_x = vert_coords_x[src_idx]
                        if not is_source_vert(src_x):
                            continue

                        w = 0.0
                        for gr in mesh.data.vertices[src_idx].groups:
                            if gr.group == vg_index:
                                w = gr.weight
                                break

                        if w > 0.0:
                            try:
                                vg.add([tgt_idx], w, 'REPLACE')
                                self_mirror_assignments += 1
                            except Exception:
                                pass
                        else:
                            # Remove vertex from target side if source has no weight (or is not in group)
                            try:
                                vg.remove([tgt_idx])
                            except Exception:
                                pass

                # middle vertices: match existing logic used for active_pairs (keep parity)
                if middle_v_final:
                    for mid_idx in middle_v_final:
                        v = mesh.data.vertices[mid_idx]
                        for vg_name in solo_to_mirror:
                            src_vg = vg_cache.get(vg_name)
                            if src_vg is None:
                                continue
                            tgt_vg = src_vg  # same group (self-mirror)
                            w = 0.0
                            for g in v.groups:
                                if g.group == src_vg.index:
                                    w = g.weight
                                    break

                            cur_w = 0.0
                            for g in v.groups:
                                if g.group == tgt_vg.index:
                                    cur_w = g.weight
                                    break

                            if abs(cur_w - w) > 1e-6:
                                try:
                                    if w > 0.0:
                                        tgt_vg.add([mid_idx], w, 'REPLACE')
                                    else:
                                        try:
                                            tgt_vg.remove([mid_idx])
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

            # optionally print a short debug summary
            if solo_to_mirror:
                print(f"AKELKA: self-mirrored solo groups: {len(solo_to_mirror)} groups, assignments: {self_mirror_assignments}")

        except Exception as _e:
            # don't abort the operation on any unexpected error in this optional step
            print("AKELKA: self-mirror solo groups step failed:", _e)
        # ----------------- END: v12-style self-mirror for "solo" groups -----------------

        # restore saved groups (everything except allowed_groups)
        if saved_group_weights:
            vg_cache = {g.name: g for g in mesh.vertex_groups}
            for gname, weights in saved_group_weights.items():
                vg = vg_cache.get(gname)
                if vg is None:
                    vg = mesh.vertex_groups.new(name=gname)
                    vg_cache[gname] = vg
                try:
                    vg.remove(all_vertex_indices)
                except Exception:
                    pass
                for vidx, w in weights.items():
                    try:
                        vg.add([vidx], w, 'REPLACE')
                    except Exception:
                        pass

        try:
            if original_active:
                context.view_layer.objects.active = original_active
                if original_active_mode and getattr(context.view_layer.objects.active, "mode", None) != original_active_mode:
                    try:
                        bpy.ops.object.mode_set(mode=original_active_mode)
                    except Exception:
                        pass
        except Exception:
            pass

        report_msg = (f"Mirrored weights (object-mode style run applied). "
                      f"Selected pairs: {len(bone_pairs_for_selected)}, "
                      f"full pairs used: {len(bone_pairs_all)}, "
                      f"active pairs used: {len(active_pairs)}, "
                      f"vertex mappings: {len(vertex_mapping)}")
        self.report({'INFO'}, report_msg)
        print("AKELKA Mirror Weights summary:", report_msg)

        # We no longer warn if pose bones were selected but no matched counterpart found,
        # because pose selection is intentionally treated as a request to run object-mode style mirroring.
        return {'FINISHED'}


# --- Mirror weights between 2 selected bones operator ---
class AKELKA_OT_mirror_weights_two_bones(bpy.types.Operator):
    """Mirror weights between exactly 2 selected bones"""
    bl_idname = "akelka.mirror_weights_two_bones"
    bl_label = "Mirror Weights (2 Selected Bones)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Always return True to keep button always available
        return True

    def execute(self, context):
        # Find armature
        arm_obj = find_armature_for_weights(context)
        if not arm_obj:
            self.report({'ERROR'}, "No armature found. Select an armature with pose bones.")
            return {'CANCELLED'}

        # Find mesh
        mesh_obj, used_active = find_mesh_for_weights(context)
        if not mesh_obj:
            self.report({'ERROR'}, "No mesh found. Select a mesh object.")
            return {'CANCELLED'}

        # Get selected pose bones
        sel_pose_bones = []
        try:
            if hasattr(arm_obj, "pose"):
                sel_pose_bones = [pb for pb in arm_obj.pose.bones if getattr(pb.bone, "select", False)]
        except Exception:
            pass

        # Check if exactly 2 bones are selected
        if len(sel_pose_bones) != 2:
            self.report({'ERROR'}, f"Error: Select only 2 bones to copy. Currently selected: {len(sel_pose_bones)} bones.")
            return {'CANCELLED'}

        bone1_name = sel_pose_bones[0].name
        bone2_name = sel_pose_bones[1].name

        # Get settings from window manager
        wm = context.window_manager
        dir_setting = getattr(wm, 'akelka_mirror_direction', 'L2R')
        axis = 'X'
        center = 0.0
        eps = getattr(wm, 'akelka_mirror_tolerance', 0.01)
        
        # Determine which bone is on the left and which is on the right (same logic as main mirror weights operator)
        bone1_x = sel_pose_bones[0].bone.head_local.x
        bone2_x = sel_pose_bones[1].bone.head_local.x
        
        # Classify bones as left or right based on X coordinate
        bone1_is_left = bone1_x < -eps
        bone1_is_right = bone1_x > eps
        bone2_is_left = bone2_x < -eps
        bone2_is_right = bone2_x > eps
        
        # Determine source and target based on direction setting (same as main operator)
        if dir_setting == 'L2R':
            # Copy from left to right
            if bone1_is_left and bone2_is_right:
                src_bone_name = bone1_name
                tgt_bone_name = bone2_name
            elif bone2_is_left and bone1_is_right:
                src_bone_name = bone2_name
                tgt_bone_name = bone1_name
            else:
                # If both on same side or in middle, use first->second as fallback
                src_bone_name = bone1_name
                tgt_bone_name = bone2_name
        else:  # R2L
            # Copy from right to left
            if bone1_is_right and bone2_is_left:
                src_bone_name = bone1_name
                tgt_bone_name = bone2_name
            elif bone2_is_right and bone1_is_left:
                src_bone_name = bone2_name
                tgt_bone_name = bone1_name
            else:
                # If both on same side or in middle, use second->first as fallback
                src_bone_name = bone2_name
                tgt_bone_name = bone1_name
        iterations = 4
        max_bfs_depth = 6
        use_nearest_tiebreak = True

        # Save original state
        original_active = context.view_layer.objects.active
        original_active_mode = None
        if original_active:
            try:
                original_active_mode = original_active.mode
            except Exception:
                original_active_mode = None

        # Build vertex mapping for mesh
        in_edit_mesh = (mesh_obj == context.object and context.mode == 'EDIT_MESH')
        if in_edit_mesh:
            bm = bmesh.from_edit_mesh(mesh_obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh_obj.data)

        bm.verts.ensure_lookup_table()
        axis_idx = get_axis_index(axis)
        vert_coords_x = [v.co[axis_idx] for v in bm.verts]

        left_v, right_v, middle_v = classify_vertices_by_side(bm, axis_idx, center, eps)
        all_verts = set(range(len(bm.verts)))

        adj = build_adjacency(bm)
        labels = wl_iterative_labels(adj, iters=iterations)

        bfs_sig_cache = {}
        for idx in all_verts:
            bfs_sig_cache[idx] = bfs_level_signature(adj, idx, max_depth=max_bfs_depth)

        def create_full_vertex_mapping():
            mapping = {}
            label_to_verts = defaultdict(list)
            for v_idx in all_verts:
                label_to_verts[labels[v_idx]].append(v_idx)

            for v_idx in all_verts:
                v_co = bm.verts[v_idx].co
                mirror_co = mirror_coordinate(v_co, axis_idx, center)

                if v_idx in middle_v:
                    mapping[v_idx] = v_idx
                    continue

                label = labels[v_idx]
                candidates = [c for c in label_to_verts[label] if c != v_idx]

                if not candidates:
                    candidates = [c for c in all_verts if c != v_idx]

                if not candidates:
                    continue

                v_sig = bfs_sig_cache[v_idx]
                exact_matches = [c for c in candidates if bfs_sig_cache[c] == v_sig]
                if exact_matches:
                    candidates = exact_matches

                best_candidate = None
                best_distance = None
                for c in candidates:
                    c_co = bm.verts[c].co
                    dist = (c_co - mirror_co).length_squared
                    if best_distance is None or dist < best_distance:
                        best_distance = dist
                        best_candidate = c

                if best_candidate is not None:
                    mapping[v_idx] = best_candidate

            return mapping

        vertex_mapping = create_full_vertex_mapping()

        if not in_edit_mesh:
            bm.free()

        eps_auto = max(eps, 1e-4)
        middle_v_final = set(i for i, x in enumerate(vert_coords_x) if abs(x - center) <= eps_auto)
        if not middle_v_final:
            middle_v_final = set(middle_v)

        # Switch to object mode for weight operations
        mesh = mesh_obj
        try:
            context.view_layer.objects.active = mesh
        except Exception:
            pass
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        vg_cache = {g.name: g for g in mesh.vertex_groups}
        all_vertex_indices = [v.index for v in mesh.data.vertices]

        # Mirror weights from source bone to target bone (respects direction setting)
        src_vg = vg_cache.get(src_bone_name)
        tgt_vg = vg_cache.get(tgt_bone_name)

        if src_vg is None:
            self.report({'ERROR'}, f"Source bone '{src_bone_name}' has no vertex group.")
            # Restore original state
            try:
                if original_active:
                    context.view_layer.objects.active = original_active
                    if original_active_mode and getattr(context.view_layer.objects.active, "mode", None) != original_active_mode:
                        try:
                            bpy.ops.object.mode_set(mode=original_active_mode)
                        except Exception:
                            pass
            except Exception:
                pass
            return {'CANCELLED'}

        # Create target vertex group if needed
        if tgt_vg is None:
            tgt_vg = mesh.vertex_groups.new(name=tgt_bone_name)
            vg_cache[tgt_bone_name] = tgt_vg

        # Clear target group before copying
        try:
            tgt_vg.remove(all_vertex_indices)
        except Exception:
            pass

        # Mirror weights from source to target (one direction only)
        src_idx = src_vg.index
        tgt_idx = tgt_vg.index

        for src_vert_idx, tgt_vert_idx in vertex_mapping.items():
            # Copy from source bone to target bone
            w = 0.0
            for g in mesh.data.vertices[src_vert_idx].groups:
                if g.group == src_idx:
                    w = g.weight
                    break

            if w > 0.0:
                tgt_vg.add([tgt_vert_idx], w, 'REPLACE')
            else:
                try:
                    tgt_vg.remove([tgt_vert_idx])
                except Exception:
                    pass

        # Handle middle vertices
        if middle_v_final:
            for mid_idx in middle_v_final:
                v = mesh.data.vertices[mid_idx]

                # Get weight from source bone
                w = 0.0
                for g in v.groups:
                    if g.group == src_idx:
                        w = g.weight
                        break

                # Apply to target bone (for middle vertices, copy the weight)
                if w > 0.0:
                    try:
                        tgt_vg.add([mid_idx], w, 'REPLACE')
                    except Exception:
                        pass
                else:
                    try:
                        tgt_vg.remove([mid_idx])
                    except Exception:
                        pass

        # Restore original state
        try:
            if original_active:
                context.view_layer.objects.active = original_active
                if original_active_mode and getattr(context.view_layer.objects.active, "mode", None) != original_active_mode:
                    try:
                        bpy.ops.object.mode_set(mode=original_active_mode)
                    except Exception:
                        pass
        except Exception:
            pass

        direction_text = "Left → Right" if dir_setting == 'L2R' else "Right → Left"
        self.report({'INFO'}, f"Mirrored weights from '{src_bone_name}' → '{tgt_bone_name}' ({direction_text}, {len(vertex_mapping)} vertex mappings)")
        return {'FINISHED'}


# --- UI Panel (selection-aware) (from v24) ---
class VIEW3D_PT_mirror_panel(bpy.types.Panel):
    bl_label = "Mirror Bone/Mesh/Weight Tools"
    bl_idname = "VIEW3D_PT_akelka_mirror_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        arm = find_armature_in_selection(context)
        mesh_present = any(o.type == 'MESH' for o in context.selected_objects)

        box = layout.box()
        box.label(text="Direction:", icon='ARROW_LEFTRIGHT')
        box.prop(wm, 'akelka_mirror_direction', expand=True)
        layout.separator()

        bone_box = layout.box()
        bone_box.label(text="Bone Tools", icon='BONE_DATA')
        bone_box.prop(wm, 'akelka_mirror_only_location', text="Location Only")

        if arm:
            # only show the top operator (Object/Pose/Edit mode) and use the new label
            bone_box.operator(AKELKA_OT_mirror_bones_from_object_mode.bl_idname, icon='GHOST_ENABLED')
        else:
            row = bone_box.row()
            row.enabled = False
            row.label(text="Select an Armature", icon='ERROR')

        layout.separator()

        mesh_box = layout.box()
        mesh_box.label(text="Mesh Tools", icon='MESH_DATA')
        mesh_available = ((context.mode == 'EDIT_MESH' and context.object and context.object.type == 'MESH')
                          or any(o.type == 'MESH' for o in context.selected_objects))
        if mesh_available:
            mesh_box.operator(AKELKA_OT_symmetrize_mesh.bl_idname, icon='MOD_MIRROR')
        else:
            row = mesh_box.row()
            row.enabled = False
            row.operator(AKELKA_OT_symmetrize_mesh.bl_idname, icon='ERROR', text="Select Mesh")

        layout.separator()

        weight_box = layout.box()
        weight_box.label(text="Weight Tools", icon='GROUP_VERTEX')

        # compact single tweaker (0.01 .. 1.0) — no exponent text, no extra numeric line
        row = weight_box.row(align=True)
        row.prop(wm, 'akelka_mirror_tolerance', text="")

        # user text as requested
        weight_box.label(text="Tweak if wrong mirroring:")

        if mesh_and_armature_available_for_weights(context):
            weight_box.operator(AKELKA_OT_mirror_weights.bl_idname, icon='MOD_MIRROR')
        else:
            row = weight_box.row()
            row.enabled = False
            row.label(text="Select Armature (or select pose bones) + Mesh", icon='ERROR')

        # Add separator and the new 2 bones button
        weight_box.separator()
        weight_box.operator(AKELKA_OT_mirror_weights_two_bones.bl_idname, icon='ARROW_LEFTRIGHT')

# --- Registration ---
classes = (
    AKELKA_OT_mirror_bones_multiple,
    AKELKA_OT_mirror_bones_from_object_mode,
    AKELKA_OT_symmetrize_mesh,
    AKELKA_OT_mirror_weights,
    AKELKA_OT_mirror_weights_two_bones,
    VIEW3D_PT_mirror_panel,
)

def register():
    register_props()
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_props()

if __name__ == '__main__':
    register()
