bl_info = {
    "name": "Akelka Bone Tracker",
    "author": "Akelka",
    "version": (1, 0, 6),
    "blender": (4, 5, 2),
    "location": "View3D > N Panel > Akelka Tools",
    "description": "Track and display pose mode bone position changes",
    "category": "Rigging",
}

import bpy
import math
import time
from mathutils import Vector, Matrix
from bpy.props import BoolProperty
from bpy.types import Operator, Panel


# Global reference to the active operator instance
_active_operator = None

# Store bone change history for display
_bone_changes = []


def get_pose_bone_head_tail_world(obj, pose_bone, rest_bone):
    """Get posed head and tail in world space. Uses bone local space (head=origin, tail=+Y*length)."""
    # In bone local space: head at origin, tail at (0, length, 0) — Blender Y is along bone
    head_4d = pose_bone.matrix @ Vector((0, 0, 0, 1.0))
    tail_4d = pose_bone.matrix @ Vector((0, rest_bone.length, 0, 1.0))
    world_head = obj.matrix_world @ Vector((head_4d.x, head_4d.y, head_4d.z))
    world_tail = obj.matrix_world @ Vector((tail_4d.x, tail_4d.y, tail_4d.z))
    return world_head, world_tail


def _vec_roll_to_mat3(vec, roll):
    """Blender's vec_roll_to_mat3: build 3x3 from bone direction (Y) + roll.
    Port of updated C from armature.c (T39470). Matrix rows/cols: [row][col]."""
    nor = vec.normalized().copy()
    THETA_THRESHOLD_NEGY = 1.0e-9
    THETA_THRESHOLD_NEGY_CLOSE = 1.0e-5
    bMatrix = Matrix.Identity(3)
    theta = 1.0 + nor.y
    if (theta > THETA_THRESHOLD_NEGY_CLOSE) or ((nor.x or nor.z) and theta > THETA_THRESHOLD_NEGY):
        bMatrix[1][0] = -nor.x
        bMatrix[0][1] = nor.x
        bMatrix[1][1] = nor.y
        bMatrix[2][1] = nor.z
        bMatrix[1][2] = -nor.z
        if theta > THETA_THRESHOLD_NEGY_CLOSE:
            bMatrix[0][0] = 1.0 - nor.x * nor.x / theta
            bMatrix[2][2] = 1.0 - nor.z * nor.z / theta
            bMatrix[0][2] = bMatrix[2][0] = -nor.x * nor.z / theta
        else:
            theta = nor.x * nor.x + nor.z * nor.z
            bMatrix[0][0] = (nor.x + nor.z) * (nor.x - nor.z) / -theta
            bMatrix[2][2] = -bMatrix[0][0]
            bMatrix[0][2] = bMatrix[2][0] = 2.0 * nor.x * nor.z / theta
    else:
        bMatrix = Matrix.Identity(3)
        bMatrix[0][0] = bMatrix[1][1] = -1.0
    rMatrix = Matrix.Rotation(roll, 3, nor)
    return rMatrix @ bMatrix


def _mat3_to_vec_roll(mat):
    """Blender's mat3_to_vec_roll: get bone direction (vec) and roll from 3x3.
    Roll is the angle that with vec reconstructs the matrix. Returns (vec, roll in radians)."""
    vec = mat.col[1].xyz.copy()
    vecmat = _vec_roll_to_mat3(vec, 0.0)
    rollmat = vecmat.inverted() @ mat
    # Roll from the XZ part of the rotation around Y (Blender C: atan2(rollmat[0][2], rollmat[2][2]))
    roll = math.atan2(rollmat[0][2], rollmat[2][2])
    return vec, roll


def get_pose_bone_roll_radians(pose_bone, rest_bone):
    """Roll from pose bone using Blender's exact mat3_to_vec_roll.
    Direction (normal) + roll = full orientation. This is what edit_bone.roll uses."""
    pose_3 = pose_bone.matrix.to_3x3()
    _, roll = _mat3_to_vec_roll(pose_3)
    return roll


class POSE_OT_track_bones(Operator):
    """Track pose mode bone position changes when user manipulates bones"""
    bl_idname = "pose.track_bones"
    bl_label = "Track Bone Positions"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _previous_positions = {}
    _last_check = 0
    _check_cooldown = 0.1
    _is_manipulating = False
    _manipulation_start_positions = {}
    _initial_pose_transforms = {}
    _record_pending = False  # Defer record until next frame so transform is applied first
    _transform_was_active = False  # True after we've seen a transform op; avoids recording on first tick after G/S/R

    def modal(self, context, event):
        # Check if tracking was stopped via button
        if not context.scene.pose_tracking_active:
            print("[DEBUG] modal: Tracking stopped, canceling operator")
            self.cancel(context)
            return {'CANCELLED'}
        
        if not context.active_object or context.active_object.type != 'ARMATURE':
            self.cancel(context)
            return {'CANCELLED'}

        # Allow mode switching - only process pose-specific events when in pose mode
        is_pose_mode = context.mode == 'POSE'
        
        # If in edit mode, just keep running but don't process pose events
        if not is_pose_mode:
            return {'PASS_THROUGH'}

        # Check for G, S, R key presses (grab/move, scale, rotate)
        if event.type in {'G', 'S', 'R'} and event.value == 'PRESS':
            print(f"[DEBUG] Key {event.type} pressed")
            if not self._is_manipulating:
                # Start tracking - save initial positions
                print("[DEBUG] Starting manipulation tracking")
                self._is_manipulating = True
                self._transform_was_active = False
                self.save_initial_positions(context)
            else:
                print("[DEBUG] Already manipulating, ignoring key press")
        
        # Check for mouse events - defer record until next frame so transform is applied first
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            print(f"[DEBUG] LEFTMOUSE RELEASE, is_manipulating: {self._is_manipulating}")
            if self._is_manipulating:
                print("[DEBUG] Left mouse released, deferring record to next frame (so rotation/transform is applied)")
                self._is_manipulating = False
                self._record_pending = True  # Record on next TIMER when pose_bone.matrix is updated

        # Check for changes periodically via timer while manipulating
        if event.type == 'TIMER':
            # Record deferred change from LMB release (after transform operator has applied)
            if self._record_pending:
                print("[DEBUG] TIMER: Recording deferred change (transform should be applied now)")
                self.update_positions(context)
                self.record_change(context)
                self._record_pending = False
                self.redraw_panel(context)

            if self._is_manipulating:
                current_time = time.time()
                if current_time - self._last_check >= self._check_cooldown:
                    self._last_check = current_time
                    # Update positions
                    self.update_positions(context)
                    
                    # Check if transform operator has ended (user clicked to confirm)
                    is_transforming = False
                    active_ops = []
                    try:
                        wm = context.window_manager
                        if hasattr(wm, 'operators'):
                            for op in wm.operators:
                                if op and hasattr(op, 'bl_idname'):
                                    op_idname = op.bl_idname
                                    active_ops.append(op_idname)
                                    if any(x in op_idname for x in ['transform', 'rotate', 'scale', 'translate', 'grab', 'resize']):
                                        is_transforming = True
                    except Exception as e:
                        print(f"[DEBUG] Error checking operators: {e}")
                    
                    if active_ops:
                        print(f"[DEBUG] Active operators: {active_ops}, is_transforming: {is_transforming}")
                    if is_transforming:
                        self._transform_was_active = True
                    # If transform ended (and we had seen it active), record the change
                    if not is_transforming and self._transform_was_active:
                        print("[DEBUG] Transform ended, recording change")
                        self.record_change(context)
                        self._is_manipulating = False
                        print("[DEBUG] Manipulation tracking stopped")
                        # Force UI redraw to update the panel
                        self.redraw_panel(context)

        return {'PASS_THROUGH'}

    def save_initial_positions(self, context):
        """Save bone positions when manipulation starts"""
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            print("[DEBUG] save_initial_positions: No armature found")
            return

        pose_bones = obj.pose.bones
        armature = obj.data

        # Get selected bones
        selected_bone_names = [b.name for b in pose_bones if b.bone.select]
        if not selected_bone_names:
            if hasattr(context, 'active_pose_bone') and context.active_pose_bone:
                selected_bone_names = [context.active_pose_bone.name]
            else:
                selected_bone_names = [b.name for b in pose_bones]

        print(f"[DEBUG] save_initial_positions: Tracking {len(selected_bone_names)} bones: {selected_bone_names}")

        # Save initial positions and pose bone transformations
        self._manipulation_start_positions = {}
        self._initial_pose_transforms = {}  # Store pose bone transforms for restoration
        for bone_name in selected_bone_names:
            if bone_name not in pose_bones or bone_name not in armature.bones:
                continue

            pose_bone = pose_bones[bone_name]
            rest_bone = armature.bones[bone_name]
            initial_head, initial_tail = get_pose_bone_head_tail_world(obj, pose_bone, rest_bone)
            initial_roll = get_pose_bone_roll_radians(pose_bone, rest_bone)
            self._manipulation_start_positions[bone_name] = {
                'head': initial_head.copy(),
                'tail': initial_tail.copy(),
                'roll': initial_roll,
            }
            
            # Store initial pose bone transformations for restoration
            self._initial_pose_transforms[bone_name] = {
                'location': pose_bone.location.copy(),
                'rotation_quaternion': pose_bone.rotation_quaternion.copy(),
                'rotation_euler': pose_bone.rotation_euler.copy(),
                'scale': pose_bone.scale.copy(),
                'rotation_mode': pose_bone.rotation_mode
            }
        
        print(f"[DEBUG] save_initial_positions: Saved {len(self._manipulation_start_positions)} bone positions")

    def _record_initial_state(self, context):
        """Record current pose as initial state (zero deltas) so panel shows activity from start."""
        global _bone_changes
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            return
        pose_bones = obj.pose.bones
        armature = obj.data
        selected_bone_names = [b.name for b in pose_bones if b.bone.select]
        if not selected_bone_names:
            if hasattr(context, 'active_pose_bone') and context.active_pose_bone:
                selected_bone_names = [context.active_pose_bone.name]
            else:
                selected_bone_names = [b.name for b in pose_bones]
        try:
            frame = bpy.context.scene.frame_current
        except Exception:
            frame = 0
        zero_vec = Vector((0.0, 0.0, 0.0))
        for bone_name in selected_bone_names:
            if bone_name not in pose_bones or bone_name not in armature.bones:
                continue
            pose_bone = pose_bones[bone_name]
            rest_bone = armature.bones[bone_name]
            head, tail = get_pose_bone_head_tail_world(obj, pose_bone, rest_bone)
            roll = get_pose_bone_roll_radians(pose_bone, rest_bone)
            _bone_changes.append({
                'bone_name': bone_name,
                'frame': frame,
                'previous_head': head.copy(),
                'previous_tail': tail.copy(),
                'current_head': head.copy(),
                'current_tail': tail.copy(),
                'head_delta': zero_vec.copy(),
                'tail_delta': zero_vec.copy(),
                'previous_roll': roll,
                'current_roll': roll,
                'roll_delta': 0.0,
                'timestamp': time.time(),
            })
        print(f"[DEBUG] _record_initial_state: Recorded {len(_bone_changes)} bone(s) as initial state")

    def update_positions(self, context):
        """Update current positions during manipulation (for tracking)"""
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            return

        pose_bones = obj.pose.bones
        armature = obj.data

        # Update positions for tracked bones
        for bone_name in self._manipulation_start_positions.keys():
            if bone_name not in pose_bones or bone_name not in armature.bones:
                continue

            pose_bone = pose_bones[bone_name]
            rest_bone = armature.bones[bone_name]
            current_head, current_tail = get_pose_bone_head_tail_world(obj, pose_bone, rest_bone)
            self._previous_positions[bone_name] = {
                'head': current_head.copy(),
                'tail': current_tail.copy()
            }

    def record_change(self, context):
        """Record the final change when manipulation ends (all selected bones)"""
        global _bone_changes
        print(f"[DEBUG] record_change: Called, tracking {len(self._manipulation_start_positions)} bones")
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            print("[DEBUG] record_change: No armature found")
            return

        pose_bones = obj.pose.bones
        armature = obj.data

        # This recording replaces any previous one (one batch per G/S/R action)
        _bone_changes.clear()
        changes_recorded = 0
        # Calculate final positions and record changes for each tracked bone
        for bone_name, initial_pos in self._manipulation_start_positions.items():
            if bone_name not in pose_bones or bone_name not in armature.bones:
                continue

            pose_bone = pose_bones[bone_name]
            rest_bone = armature.bones[bone_name]
            final_head, final_tail = get_pose_bone_head_tail_world(obj, pose_bone, rest_bone)
            final_roll = get_pose_bone_roll_radians(pose_bone, rest_bone)
            # Calculate differences (world space; roll in radians)
            head_diff = final_head - initial_pos['head']
            tail_diff = final_tail - initial_pos['tail']
            roll_delta = final_roll - initial_pos['roll']
            
            # Only record if there's a significant change (position or roll)
            threshold_pos = 0.0001
            threshold_roll = 0.0001  # radians
            if (head_diff.length > threshold_pos or tail_diff.length > threshold_pos or abs(roll_delta) > threshold_roll):
                try:
                    frame = bpy.context.scene.frame_current
                except:
                    frame = 0
                
                print(f"[DEBUG] record_change: Recording change for {bone_name}")
                print(f"[DEBUG]   Head delta: {head_diff.length:.6f}, Tail delta: {tail_diff.length:.6f}, Roll delta: {roll_delta:.6f} rad")
                print(f"[DEBUG]   Head: {initial_pos['head']} -> {final_head}")
                print(f"[DEBUG]   Tail: {initial_pos['tail']} -> {final_tail}")
                
                change_entry = {
                    'bone_name': bone_name,
                    'frame': frame,
                    'previous_head': initial_pos['head'].copy(),
                    'previous_tail': initial_pos['tail'].copy(),
                    'current_head': final_head.copy(),
                    'current_tail': final_tail.copy(),
                    'head_delta': head_diff.copy(),
                    'tail_delta': tail_diff.copy(),
                    'previous_roll': initial_pos['roll'],
                    'current_roll': final_roll,
                    'roll_delta': roll_delta,
                    'timestamp': time.time()
                }
                _bone_changes.append(change_entry)
                changes_recorded += 1
            else:
                print(f"[DEBUG] record_change: Change too small for {bone_name} (head: {head_diff.length:.6f}, tail: {tail_diff.length:.6f}, roll: {roll_delta:.6f})")
        
        print(f"[DEBUG] record_change: Recorded {changes_recorded} changes for {len(_bone_changes)} bone(s)")
        
        # Apply to edit bones if enabled (all recorded bones)
        if context.scene.pose_tracking_auto_apply_edit and changes_recorded > 0 and _bone_changes:
            print(f"[DEBUG] record_change: Auto-applying to {len(_bone_changes)} edit bone(s)")
            apply_all_changes_to_edit_bones(context, return_to_pose=True)
        
        # Restore bone positions if enabled
        if context.scene.pose_tracking_restore_position and changes_recorded > 0:
            self.restore_bone_positions(context)

    def restore_bone_positions(self, context):
        """Restore bones to their original positions"""
        print("[DEBUG] restore_bone_positions: Restoring bone positions")
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            return
        
        pose_bones = obj.pose.bones
        
        for bone_name, initial_transforms in self._initial_pose_transforms.items():
            if bone_name not in pose_bones:
                continue
            
            pose_bone = pose_bones[bone_name]
            print(f"[DEBUG] restore_bone_positions: Restoring {bone_name}")
            
            # Restore pose bone transformations
            pose_bone.location = initial_transforms['location']
            pose_bone.scale = initial_transforms['scale']
            
            # Restore rotation based on rotation mode
            pose_bone.rotation_mode = initial_transforms['rotation_mode']
            if pose_bone.rotation_mode == 'QUATERNION':
                pose_bone.rotation_quaternion = initial_transforms['rotation_quaternion']
            else:
                pose_bone.rotation_euler = initial_transforms['rotation_euler']
        
        print("[DEBUG] restore_bone_positions: Bone positions restored")

    def redraw_panel(self, context):
        """Force redraw of the UI panel"""
        try:
            # Find all UI regions and redraw them
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'UI':
                                region.tag_redraw()
                                print("[DEBUG] redraw_panel: UI region redraw triggered")
        except Exception as e:
            print(f"[DEBUG] redraw_panel: Error redrawing: {e}")

    def invoke(self, context, event):
        if context.active_object is None or context.active_object.type != 'ARMATURE':
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        if context.mode != 'POSE':
            self.report({'ERROR'}, "Must be in Pose Mode")
            return {'CANCELLED'}

        # Check if already running
        if context.scene.pose_tracking_active:
            # Stop the tracking
            print("[DEBUG] invoke: Stopping tracking")
            self.cancel(context)
            return {'FINISHED'}

        # Start the tracking
        print("[DEBUG] invoke: Starting tracking")
        global _bone_changes, _active_operator
        _bone_changes = []  # Clear previous changes
        self._last_check = time.time()
        self._is_manipulating = False
        self._record_pending = False
        self._transform_was_active = False
        self._manipulation_start_positions = {}
        self._initial_pose_transforms = {}
        self._previous_positions = {}

        _active_operator = self
        context.scene.pose_tracking_active = True
        # Record current state as "original" so panel is active from start (zero deltas)
        self._record_initial_state(context)
        print("[DEBUG] invoke: Tracking started, waiting for G/S/R key press")

        # Add modal handler to keep operator alive
        context.window_manager.modal_handler_add(self)

        # Add timer to keep modal running
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        """Stop the tracking operation"""
        global _active_operator
        print("[DEBUG] cancel: Stopping tracking")
        
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        _active_operator = None
        context.scene.pose_tracking_active = False
        self._previous_positions = {}
        self._manipulation_start_positions = {}
        self._initial_pose_transforms = {}
        self._is_manipulating = False
        self._record_pending = False
        self._transform_was_active = False
        print("[DEBUG] cancel: Tracking stopped")


class POSE_OT_bone_info(Operator):
    """Track pose mode bone position changes only when user finishes manipulation"""
    bl_idname = "pose.track_bones_on_finish"
    bl_label = "Track On Finish"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _previous_positions = {}
    _last_check = 0
    _check_cooldown = 0.1
    _last_transform_time = 0
    _finish_delay = 0.5  # Wait 0.5 seconds after transform ends before tracking
    _last_track_time = 0  # Track when we last recorded a change

    def modal(self, context, event):
        if not context.active_object or context.active_object.type != 'ARMATURE':
            self.cancel(context)
            return {'CANCELLED'}

        if context.mode != 'POSE':
            self.cancel(context)
            return {'CANCELLED'}

        current_time = time.time()
        
        # Check for mouse button events and update transform time
        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE'}:
            if event.value in {'PRESS', 'REPEAT'}:
                self._last_transform_time = current_time
                return {'PASS_THROUGH'}
        
        # Use timer to check periodically if transforms have finished
        if event.type == 'TIMER':
            # Check for active transform operators
            is_transforming = False
            try:
                wm = context.window_manager
                if hasattr(wm, 'operators'):
                    for op in wm.operators:
                        if op and hasattr(op, 'bl_idname'):
                            op_idname = op.bl_idname
                            if any(x in op_idname for x in ['transform', 'rotate', 'scale', 'translate', 'grab', 'resize', 'pose']):
                                is_transforming = True
                                self._last_transform_time = current_time
                                break
            except:
                pass
            
            # If currently transforming, don't track
            if is_transforming:
                return {'PASS_THROUGH'}
            
            # Check if enough time has passed since last transform
            time_since_transform = current_time - self._last_transform_time
            if time_since_transform >= self._finish_delay:
                # Enough time has passed, check for changes (but only once per delay period)
                if current_time - self._last_check >= self._check_cooldown:
                    self._last_check = current_time
                    self.track_bone_changes(bpy.context)

        return {'PASS_THROUGH'}

    def track_bone_changes(self, context):
        """Track changes in pose mode bone positions - only when transforms finish"""
        global _bone_changes
        current_time = time.time()
        
        # Don't track too frequently (cooldown)
        if current_time - self._last_track_time < 0.2:
            return
        
        # Double-check we're not currently transforming (safety check)
        is_transforming = False
        try:
            wm = bpy.context.window_manager
            if hasattr(wm, 'operators'):
                for op in wm.operators:
                    if op and hasattr(op, 'bl_idname'):
                        op_idname = op.bl_idname
                        if any(x in op_idname for x in ['transform', 'rotate', 'scale', 'translate', 'grab', 'resize', 'pose']):
                            # Still transforming, update time and skip
                            is_transforming = True
                            self._last_transform_time = current_time
                            break
        except:
            pass
        
        if is_transforming:
            return
        
        # Verify enough time has passed since last transform
        time_since_transform = current_time - self._last_transform_time
        if time_since_transform < self._finish_delay:
            return
        
        # Safely get active object from context
        try:
            if isinstance(context, dict):
                obj = context.get('active_object')
            else:
                obj = getattr(context, 'active_object', None)
        except:
            obj = None
            
        if not obj or obj.type != 'ARMATURE':
            return

        pose_bones = obj.pose.bones
        armature = obj.data

        # Get selected bones (check bone selection state)
        selected_bone_names = [b.name for b in pose_bones if b.bone.select]
        
        # If no bones selected, track active bone or all bones
        if not selected_bone_names:
            try:
                if isinstance(context, dict):
                    active_pose_bone = context.get('active_pose_bone')
                else:
                    active_pose_bone = getattr(context, 'active_pose_bone', None)
                if active_pose_bone:
                    selected_bone_names = [active_pose_bone.name]
                else:
                    # Track all bones if nothing selected
                    selected_bone_names = [b.name for b in pose_bones]
            except:
                # Track all bones if nothing selected
                selected_bone_names = [b.name for b in pose_bones]

        # Track changes for selected bones (world space)
        for bone_name in selected_bone_names:
            if bone_name not in pose_bones or bone_name not in armature.bones:
                continue

            pose_bone = pose_bones[bone_name]
            rest_bone = armature.bones[bone_name]
            current_head, current_tail = get_pose_bone_head_tail_world(obj, pose_bone, rest_bone)

            # Check if this bone has changed
            if bone_name in self._previous_positions:
                prev_head = self._previous_positions[bone_name]['head']
                prev_tail = self._previous_positions[bone_name]['tail']
                
                # Calculate differences
                head_diff = current_head - prev_head
                tail_diff = current_tail - prev_tail
                
                # Only record if there's a significant change (threshold to avoid noise)
                threshold = 0.0001
                if (head_diff.length > threshold or tail_diff.length > threshold):
                    # Get frame from bpy.context to avoid IDProperties issues
                    try:
                        frame = bpy.context.scene.frame_current
                    except:
                        frame = 0
                    
                    # Replace the last change instead of appending
                    change_entry = {
                        'bone_name': bone_name,
                        'frame': frame,
                        'previous_head': prev_head.copy(),
                        'previous_tail': prev_tail.copy(),
                        'current_head': current_head.copy(),
                        'current_tail': current_tail.copy(),
                        'head_delta': head_diff.copy(),
                        'tail_delta': tail_diff.copy(),
                        'timestamp': time.time()
                    }
                    # Replace last change (only keep one)
                    _bone_changes = [change_entry]
                    self._last_track_time = current_time

            # Update stored position
            self._previous_positions[bone_name] = {
                'head': current_head.copy(),
                'tail': current_tail.copy()
            }

    def initialize_positions(self, context):
        """Initialize stored bone positions"""
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            return

        pose_bones = obj.pose.bones
        armature = obj.data
        self._previous_positions = {}

        # Get selected bones (check bone selection state)
        selected_bone_names = [b.name for b in pose_bones if b.bone.select]
        
        # If no bones selected, track active bone or all bones
        if not selected_bone_names:
            if hasattr(context, 'active_pose_bone') and context.active_pose_bone:
                selected_bone_names = [context.active_pose_bone.name]
            else:
                # Track all bones if nothing selected
                selected_bone_names = [b.name for b in pose_bones]

        # Get initial pose positions in world space
        for bone_name in selected_bone_names:
            if bone_name not in armature.bones or bone_name not in pose_bones:
                continue
            rest_bone = armature.bones[bone_name]
            pose_bone = pose_bones[bone_name]
            initial_head, initial_tail = get_pose_bone_head_tail_world(obj, pose_bone, rest_bone)
            self._previous_positions[bone_name] = {
                'head': initial_head.copy(),
                'tail': initial_tail.copy()
            }

    def invoke(self, context, event):
        if context.active_object is None or context.active_object.type != 'ARMATURE':
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        if context.mode != 'POSE':
            self.report({'ERROR'}, "Must be in Pose Mode")
            return {'CANCELLED'}

        # Check if already running
        if context.scene.pose_tracking_on_finish_active:
            # Stop the tracking
            self.cancel(context)
            return {'FINISHED'}

        # Start the tracking
        global _bone_changes, _active_operator
        _bone_changes = []  # Clear previous changes
        self._last_check = time.time()
        self._last_transform_time = time.time()  # Initialize transform time
        self.initialize_positions(context)

        # Stop continuous tracking if it's running
        if context.scene.pose_tracking_active:
            # Cancel any existing continuous tracking
            if _active_operator and hasattr(_active_operator, 'cancel'):
                _active_operator.cancel(context)

        _active_operator = self
        context.scene.pose_tracking_on_finish_active = True
        context.scene.pose_tracking_active = False

        # Add depsgraph update handler
        if depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)

        # Add modal handler to keep operator alive
        context.window_manager.modal_handler_add(self)

        # Add timer to keep modal running
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        """Stop the tracking operation"""
        global _active_operator
        
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        # Remove handler
        if depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)

        _active_operator = None
        context.scene.pose_tracking_on_finish_active = False
        self._previous_positions = {}


class POSE_OT_bone_info(Operator):
    """Report bone head and tail positions in edit and pose mode"""
    bl_idname = "pose.bone_info"
    bl_label = "Bone Info"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        armature = obj.data
        
        # Get selected bone
        selected_bone = None
        bone_name = None
        
        if context.mode == 'POSE':
            # In pose mode, use context.active_pose_bone or check selected bones
            if hasattr(context, 'active_pose_bone') and context.active_pose_bone:
                selected_bone = context.active_pose_bone
                bone_name = selected_bone.name
            else:
                # Fallback: check selected bones in pose mode
                selected_bones = [b for b in obj.pose.bones if b.bone.select]
                if selected_bones:
                    bone_name = selected_bones[0].name
                else:
                    # Another fallback: check armature's active bone
                    if armature.bones.active:
                        bone_name = armature.bones.active.name
                    else:
                        self.report({'ERROR'}, "No bone selected in pose mode")
                        return {'CANCELLED'}
        elif context.mode == 'EDIT':
            # In edit mode, use armature.bones.active (not edit_bones.active)
            if armature.bones.active:
                selected_bone = armature.bones.active
                bone_name = selected_bone.name
            else:
                # Fallback: check selected bones
                selected_bones = [b for b in armature.edit_bones if b.select]
                if selected_bones:
                    bone_name = selected_bones[0].name
                else:
                    self.report({'ERROR'}, "No bone selected in edit mode")
                    return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Must be in Pose or Edit Mode")
            return {'CANCELLED'}

        if not bone_name:
            self.report({'ERROR'}, "No bone selected")
            return {'CANCELLED'}

        # Get edit mode positions (rest pose)
        edit_head = None
        edit_tail = None
        was_in_edit = context.mode == 'EDIT'
        original_mode = context.mode
        
        # Switch to edit mode to get rest pose positions
        if not was_in_edit:
            try:
                bpy.ops.object.mode_set(mode='EDIT')
            except Exception as e:
                print(f"Could not switch to edit mode: {e}")
        
        # Now we should be in edit mode, get the bone positions
        try:
            if bone_name in armature.edit_bones:
                edit_bone = armature.edit_bones[bone_name]
                edit_head = edit_bone.head.copy()
                edit_tail = edit_bone.tail.copy()
        except Exception as e:
            print(f"Could not access edit_bones: {e}")
        
        # Get pose mode positions
        pose_head = None
        pose_tail = None
        
        # Switch to pose mode to get pose bone data
        if original_mode != 'POSE':
            try:
                bpy.ops.object.mode_set(mode='POSE')
            except Exception as e:
                print(f"Could not switch to pose mode: {e}")
        
        # Now calculate pose positions from rest pose
        if bone_name in obj.pose.bones and edit_head and edit_tail:
            pose_bone = obj.pose.bones[bone_name]
            
            # Get the pose bone's matrix (includes all transformations)
            # This matrix transforms from rest pose to current pose in object space
            bone_matrix = pose_bone.matrix
            
            # Transform rest head and tail by pose matrix (object space)
            head_4d = Vector((edit_head.x, edit_head.y, edit_head.z, 1.0))
            tail_4d = Vector((edit_tail.x, edit_tail.y, edit_tail.z, 1.0))
            
            # Apply pose transformation
            pose_head_4d = bone_matrix @ head_4d
            pose_tail_4d = bone_matrix @ tail_4d
            
            # Convert back to 3D (object space)
            pose_head = Vector((pose_head_4d.x, pose_head_4d.y, pose_head_4d.z))
            pose_tail = Vector((pose_tail_4d.x, pose_tail_4d.y, pose_tail_4d.z))
        
        # Restore original mode
        if context.mode != original_mode:
            try:
                bpy.ops.object.mode_set(mode=original_mode)
            except:
                pass
        
        # Build report message
        msg = f"\n=== Bone Info: {bone_name} ===\n"
        msg += f"Current Frame: {context.scene.frame_current}\n\n"
        
        if edit_head and edit_tail:
            msg += f"EDIT MODE (Object Space):\n"
            msg += f"  Head: ({edit_head.x:.4f}, {edit_head.y:.4f}, {edit_head.z:.4f})\n"
            msg += f"  Tail: ({edit_tail.x:.4f}, {edit_tail.y:.4f}, {edit_tail.z:.4f})\n\n"
        else:
            msg += f"EDIT MODE: Could not get positions\n\n"
        
        if pose_head and pose_tail:
            msg += f"POSE MODE - Frame {context.scene.frame_current} (Object Space):\n"
            msg += f"  Head: ({pose_head.x:.4f}, {pose_head.y:.4f}, {pose_head.z:.4f})\n"
            msg += f"  Tail: ({pose_tail.x:.4f}, {pose_tail.y:.4f}, {pose_tail.z:.4f})\n"
            
            # Also show the difference
            if edit_head and edit_tail:
                head_diff = pose_head - edit_head
                tail_diff = pose_tail - edit_tail
                msg += f"\nDifference from Edit Mode:\n"
                msg += f"  Head Delta: ({head_diff.x:.4f}, {head_diff.y:.4f}, {head_diff.z:.4f})\n"
                msg += f"  Tail Delta: ({tail_diff.x:.4f}, {tail_diff.y:.4f}, {tail_diff.z:.4f})\n"
        else:
            msg += f"POSE MODE: Could not get positions\n"
        
        # Print to console
        print(msg)
        
        # Also show in info area
        self.report({'INFO'}, f"Bone info printed to console (see Info area)")
        
        return {'FINISHED'}


class POSE_OT_clear_changes(Operator):
    """Clear bone change history"""
    bl_idname = "pose.clear_changes"
    bl_label = "Clear Changes"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _bone_changes
        _bone_changes = []
        self.report({'INFO'}, "Change history cleared")
        return {'FINISHED'}


def _apply_one_change_to_edit_bone(obj, armature, change):
    """Apply a single bone's change; must already be in EDIT mode. Returns True if applied."""
    bone_name = change['bone_name']
    if bone_name not in armature.edit_bones:
        print(f"[DEBUG] _apply_one_change_to_edit_bone: Bone {bone_name} not found")
        return False
    edit_bone = armature.edit_bones[bone_name]
    mw_inv = obj.matrix_world.to_3x3().inverted()
    head_delta_local = mw_inv @ change['head_delta']
    tail_delta_local = mw_inv @ change['tail_delta']
    edit_bone.head += head_delta_local
    edit_bone.tail += tail_delta_local
    current_roll = change.get('current_roll')
    if current_roll is not None:
        edit_bone.roll = current_roll
    print(f"[DEBUG] Applied to {bone_name}: head+tail deltas, roll={math.degrees(current_roll or 0):.2f}°")
    return True


def apply_change_to_edit_bone(context, change, return_to_pose=True):
    """Apply one tracked change to edit mode (single bone)."""
    obj = context.active_object
    if not obj or obj.type != 'ARMATURE':
        return False
    original_mode = context.mode
    was_in_pose = original_mode == 'POSE'
    if original_mode != 'EDIT':
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception as e:
            print(f"[DEBUG] apply_change_to_edit_bone: Could not switch to edit mode: {e}")
            return False
    try:
        armature = obj.data
        ok = _apply_one_change_to_edit_bone(obj, armature, change)
    except Exception as e:
        print(f"[DEBUG] apply_change_to_edit_bone: Error: {e}")
        ok = False
    if return_to_pose and was_in_pose:
        try:
            bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass
    return ok


def apply_all_changes_to_edit_bones(context, return_to_pose=True):
    """Apply all recorded changes to edit bones (multi-bone). Single mode switch."""
    global _bone_changes
    obj = context.active_object
    if not obj or obj.type != 'ARMATURE':
        return 0
    if not _bone_changes:
        return 0
    original_mode = context.mode
    was_in_pose = original_mode == 'POSE'
    try:
        bpy.ops.object.mode_set(mode='EDIT')
    except Exception as e:
        print(f"[DEBUG] apply_all_changes_to_edit_bones: Could not switch to edit mode: {e}")
        return 0
    armature = obj.data
    applied = 0
    for change in _bone_changes:
        if _apply_one_change_to_edit_bone(obj, armature, change):
            applied += 1
    if return_to_pose and was_in_pose:
        try:
            bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass
    print(f"[DEBUG] apply_all_changes_to_edit_bones: Applied {applied}/{len(_bone_changes)} bones")
    return applied


class POSE_OT_apply_to_edit_bone(Operator):
    """Apply all tracked changes to edit mode bones"""
    bl_idname = "pose.apply_to_edit_bone"
    bl_label = "Apply to Edit Bones"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global _bone_changes
        
        if not _bone_changes:
            self.report({'WARNING'}, "No tracked changes to apply")
            return {'CANCELLED'}
        
        applied = apply_all_changes_to_edit_bones(context, return_to_pose=True)
        if applied > 0:
            self.report({'INFO'}, f"Applied changes to {applied} bone(s) in edit mode")
        else:
            self.report({'ERROR'}, "Failed to apply changes to edit bones")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class POSE_PT_sync_panel(Panel):
    """Panel for Pose Bone Position Tracking"""
    bl_label = "Pose Bone Tracker"
    bl_idname = "POSE_PT_sync_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and
                context.active_object.type == 'ARMATURE' and
                (context.mode == 'POSE' or context.mode == 'EDIT'))

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        global _bone_changes

        # Bone info button
        layout.label(text="Diagnostics:")
        layout.operator("pose.bone_info", text="Show Bone Info", icon='INFO')
        
        layout.separator()

        # Tracking button (only in pose mode)
        if context.mode == 'POSE':
            row = layout.row()
            if scene.pose_tracking_active:
                row.operator("pose.track_bones", text="Stop Tracking", icon='PAUSE')
            else:
                row.operator("pose.track_bones", text="Start Tracking", icon='PLAY')
            
            layout.label(text="(Press G/S/R, click to confirm)", icon='INFO')
            
            # Restore position checkbox
            layout.prop(scene, "pose_tracking_restore_position", text="Restore Bone Position After Recording")
            
            # Auto-apply to edit bone checkbox
            layout.prop(scene, "pose_tracking_auto_apply_edit", text="Auto-Apply to Edit Bone After Recording")
            
            layout.separator()
        
        # Display recorded changes (all bones)
        if _bone_changes:
            n = len(_bone_changes)
            layout.label(text=f"Recorded changes ({n} bone{'s' if n != 1 else ''}):", icon='INFO')
            
            for change in _bone_changes:
                box = layout.box()
                row = box.row()
                row.label(text=f"Bone: {change['bone_name']}", icon='BONE_DATA')
                row.label(text=f"Frame: {change['frame']}")
                
                box.separator()
                box.label(text="Head Position:")
                row = box.row()
                col1, col2, col3 = row.column(align=True), row.column(align=True), row.column(align=True)
                col1.label(text="Previous:")
                col1.label(text=f"X: {change['previous_head'].x:.4f}")
                col1.label(text=f"Y: {change['previous_head'].y:.4f}")
                col1.label(text=f"Z: {change['previous_head'].z:.4f}")
                col2.label(text="Current:")
                col2.label(text=f"X: {change['current_head'].x:.4f}")
                col2.label(text=f"Y: {change['current_head'].y:.4f}")
                col2.label(text=f"Z: {change['current_head'].z:.4f}")
                col3.label(text="Change:")
                delta = change['head_delta']
                col3.label(text=f"X: {delta.x:+.4f}", icon='RIGHTARROW' if delta.x > 0 else 'TRIA_LEFT' if delta.x < 0 else 'DOT')
                col3.label(text=f"Y: {delta.y:+.4f}", icon='RIGHTARROW' if delta.y > 0 else 'TRIA_LEFT' if delta.y < 0 else 'DOT')
                col3.label(text=f"Z: {delta.z:+.4f}", icon='RIGHTARROW' if delta.z > 0 else 'TRIA_LEFT' if delta.z < 0 else 'DOT')
                
                box.separator()
                box.label(text="Tail Position:")
                row = box.row()
                col1, col2, col3 = row.column(align=True), row.column(align=True), row.column(align=True)
                col1.label(text="Previous:")
                col1.label(text=f"X: {change['previous_tail'].x:.4f}")
                col1.label(text=f"Y: {change['previous_tail'].y:.4f}")
                col1.label(text=f"Z: {change['previous_tail'].z:.4f}")
                col2.label(text="Current:")
                col2.label(text=f"X: {change['current_tail'].x:.4f}")
                col2.label(text=f"Y: {change['current_tail'].y:.4f}")
                col2.label(text=f"Z: {change['current_tail'].z:.4f}")
                col3.label(text="Change:")
                delta = change['tail_delta']
                col3.label(text=f"X: {delta.x:+.4f}", icon='RIGHTARROW' if delta.x > 0 else 'TRIA_LEFT' if delta.x < 0 else 'DOT')
                col3.label(text=f"Y: {delta.y:+.4f}", icon='RIGHTARROW' if delta.y > 0 else 'TRIA_LEFT' if delta.y < 0 else 'DOT')
                col3.label(text=f"Z: {delta.z:+.4f}", icon='RIGHTARROW' if delta.z > 0 else 'TRIA_LEFT' if delta.z < 0 else 'DOT')
                
                box.separator()
                box.label(text="Roll (twist around bone axis):")
                roll_delta = change.get('roll_delta', 0.0)
                row = box.row()
                row.label(text=f"Previous: {math.degrees(change.get('previous_roll', 0.0)):.4f}°")
                row.label(text=f"Current: {math.degrees(change.get('current_roll', 0.0)):.4f}°")
                row.label(text=f"Delta: {math.degrees(roll_delta):+.4f}° ({roll_delta:+.6f} rad)")
            
            layout.separator()
            if context.mode == 'POSE':
                layout.operator("pose.apply_to_edit_bone", text="Apply to Edit Bones", icon='ARMATURE_DATA')
                layout.label(text=f"(Applies head/tail/roll to all {n} bone(s))", icon='INFO')
                layout.separator()
            layout.operator("pose.clear_changes", text="Clear", icon='TRASH')
        else:
            layout.label(text="No changes tracked yet", icon='INFO')
            if context.mode == 'POSE':
                layout.label(text="Start tracking to see changes")


# Register properties
def register():
    bpy.utils.register_class(POSE_OT_track_bones)
    bpy.utils.register_class(POSE_OT_bone_info)
    bpy.utils.register_class(POSE_OT_clear_changes)
    bpy.utils.register_class(POSE_OT_apply_to_edit_bone)
    bpy.utils.register_class(POSE_PT_sync_panel)
    bpy.types.Scene.pose_tracking_active = BoolProperty(
        name="Pose Tracking Active",
        default=False,
        description="Whether pose bone position tracking is currently active"
    )
    bpy.types.Scene.pose_tracking_restore_position = BoolProperty(
        name="Restore Bone Position",
        default=True,
        description="Restore bone to original position after recording change"
    )
    bpy.types.Scene.pose_tracking_auto_apply_edit = BoolProperty(
        name="Auto-Apply to Edit Bone",
        default=True,
        description="Automatically apply tracked change to edit mode bone after recording"
    )


def unregister():
    bpy.utils.unregister_class(POSE_OT_track_bones)
    bpy.utils.unregister_class(POSE_OT_bone_info)
    bpy.utils.unregister_class(POSE_OT_clear_changes)
    bpy.utils.unregister_class(POSE_OT_apply_to_edit_bone)
    bpy.utils.unregister_class(POSE_PT_sync_panel)
    del bpy.types.Scene.pose_tracking_active
    del bpy.types.Scene.pose_tracking_restore_position
    del bpy.types.Scene.pose_tracking_auto_apply_edit


if __name__ == "__main__":
    register()  

