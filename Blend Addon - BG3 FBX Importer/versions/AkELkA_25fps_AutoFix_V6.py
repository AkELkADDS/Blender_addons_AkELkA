bl_info = {
    "name": "AkELkA - 25fps AutoFix",
    "author": "AkELkA",
    "version": (1, 3, 3),
    "blender": (3, 0, 0),
    "location": "View3D > N Panel > Akelka Tools > Animation",
    "description": "Automatically fixes BG3’s 25 FPS animation shrink for Noira imports. Auto-fix (when 'Is animation' checked) + manual fix.",
    "doc_url": "https://www.patreon.com/AkELkA",
    "category": "Animation",
}

import bpy
import os
import math
from bpy.props import StringProperty, BoolProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator

# ---------------------
# Utilities
# ---------------------
def deselect_all():
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    bpy.ops.object.select_all(action='DESELECT')

def select_object(obj, isActive=True):
    if obj is None:
        return
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    deselect_all()
    try:
        obj.select_set(True)
    except Exception:
        pass
    if isActive:
        try:
            bpy.context.view_layer.objects.active = obj
        except Exception:
            pass

def backup_context_mode():
    current_mode = bpy.context.mode
    active_object = bpy.context.view_layer.objects.active
    selected_objects = list(bpy.context.selected_objects)
    return current_mode, active_object, selected_objects

def restore_context_mode(current_mode, active_object, selected_objects):
    try:
        if len(bpy.context.selected_objects) > 0:
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        pass

    if active_object:
        try:
            bpy.context.view_layer.objects.active = active_object
            for obj in selected_objects:
                obj.select_set(True)
        except Exception:
            pass

    try:
        if 'EDIT' in current_mode:
            bpy.ops.object.mode_set(mode='EDIT')
        elif 'POSE' in current_mode:
            bpy.ops.object.mode_set(mode='POSE')
    except Exception:
        pass

def apply_all_transforms_for_selected():
    current_mode, active_object, selected_objects = backup_context_mode()
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    for obj in bpy.context.selected_objects:
        try:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            obj.matrix_basis.identity()
        except Exception:
            pass
    restore_context_mode(current_mode, active_object, selected_objects)

def apply_all_transforms(obj=None, loc=True, rot=True, scale=True):
    currentMode = 'OBJECT'
    try:
        if bpy.context.object:
            currentMode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    for object in bpy.context.selected_objects:
        try:
            bpy.ops.object.transform_apply(location=loc, rotation=rot, scale=scale)
            object.matrix_basis.identity()
        except Exception:
            pass

    try:
        bpy.ops.object.mode_set(mode=currentMode)
    except Exception:
        pass

def is_collection_child(collection, potential_parent):
    for child_collection in potential_parent.children:
        if child_collection == collection:
            return True
    return False

def recurLayerCollection(layerColl, collName):
    if layerColl.name == collName:
        return layerColl
    for layer in layerColl.children:
        found = recurLayerCollection(layer, collName)
        if found:
            return found
    return None

def set_layer_collection_active(colName):
    layer_collection = bpy.context.view_layer.layer_collection
    layerColl = recurLayerCollection(layer_collection, colName)
    if layerColl:
        bpy.context.view_layer.active_layer_collection = layerColl
        return layerColl
    return None

def add_armature_modifier(armature_obj, mesh_obj):
    if mesh_obj and armature_obj and armature_obj.type == 'ARMATURE':
        armature_modifier = None
        for modifier in mesh_obj.modifiers:
            if modifier.type == 'ARMATURE':
                armature_modifier = modifier
                break

        if armature_modifier:
            armature_modifier.object = armature_obj
        else:
            try:
                armature_modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
                armature_modifier.object = armature_obj
            except Exception:
                pass

def set_pose_as_rest(armature):
    if armature and armature.type == 'ARMATURE':
        current_mode, active_object, selected_objects = backup_context_mode()
        deselect_all()
        select_object(armature, True)
        apply_all_transforms_for_selected()
        try:
            bpy.ops.object.mode_set(mode='POSE')
            vanillaSelection = bpy.context.selected_pose_bones

            for posBone in armature.pose.bones:
                armature.pose.bones[posBone.name].bone.select = True

            bpy.ops.pose.armature_apply(selected=True)

            for bone in armature.pose.bones:
                bone.bone.select = False

            for bone in vanillaSelection:
                bone.bone.select = True
        except Exception:
            pass
        restore_context_mode(current_mode, active_object, selected_objects)

# ---------------------
# rotateObjectEachFrame (safe: only keys existing frames; called after scaling & clamping)
# ---------------------
def rotateObjectEachFrame(obj):
    """
    Only insert rotation keyframes on frames that actually exist in the imported (and clamped) animation.
    This preserves all rotation correction logic but avoids baking flat junk keys across the whole scene range.
    """
    if obj is None:
        return

    scene = bpy.context.scene
    frames = set()

    try:
        ad = obj.animation_data
    except Exception:
        ad = None

    if ad:
        action = ad.action if hasattr(ad, "action") else None
        if action:
            for fcurve in action.fcurves:
                for kp in fcurve.keyframe_points:
                    frames.add(int(round(kp.co.x)))

        if hasattr(ad, "nla_tracks"):
            for track in ad.nla_tracks:
                for strip in track.strips:
                    try:
                        start = int(round(strip.frame_start))
                        end = int(round(strip.frame_end))
                        frames.update(range(start, end + 1))
                    except Exception:
                        pass
                    try:
                        sact = strip.action
                        if sact:
                            for fcurve in sact.fcurves:
                                for kp in fcurve.keyframe_points:
                                    frames.add(int(round(kp.co.x)))
                    except Exception:
                        pass

    if not frames:
        return

    for frame in sorted(frames):
        try:
            scene.frame_set(frame)
            obj.rotation_euler = (0, 0, 0)
            obj.keyframe_insert(data_path="rotation_euler", index=-1)
        except Exception:
            continue

# ---------------------
# Animation FPS fix + clamp logic
# ---------------------
def scale_action_keyframes(action, scale):
    if not action:
        return
    for fcurve in action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.co.x = kp.co.x * scale
            kp.handle_left.x = kp.handle_left.x * scale
            kp.handle_right.x = kp.handle_right.x * scale
        try:
            fcurve.update()
        except Exception:
            pass

def gather_original_frames_for_object(obj):
    """Gather the set of keyframe frames from object's action and NLA strips BEFORE scaling."""
    frames = set()
    if obj is None:
        return frames
    try:
        ad = obj.animation_data
    except Exception:
        ad = None

    if not ad:
        return frames

    action = ad.action if hasattr(ad, "action") else None
    if action:
        for fcurve in action.fcurves:
            for kp in fcurve.keyframe_points:
                frames.add(float(kp.co.x))

    if hasattr(ad, "nla_tracks"):
        for track in ad.nla_tracks:
            for strip in track.strips:
                try:
                    # include strip range as possible frames (float)
                    frames.add(float(strip.frame_start))
                    frames.add(float(strip.frame_end))
                except Exception:
                    pass
                try:
                    sact = strip.action
                    if sact:
                        for fcurve in sact.fcurves:
                            for kp in fcurve.keyframe_points:
                                frames.add(float(kp.co.x))
                except Exception:
                    pass
    return frames

def clamp_action_to_max_frame(action, max_frame):
    """Remove any keyframes in action that are strictly greater than max_frame."""
    if not action:
        return
    for fcurve in action.fcurves:
        to_remove = [kp for kp in fcurve.keyframe_points if kp.co.x > max_frame]
        for kp in to_remove:
            try:
                fcurve.keyframe_points.remove(kp)
            except Exception:
                pass
        try:
            fcurve.update()
        except Exception:
            pass

def fix_imported_animation_timing_and_clamp(imported_objects, original_fps, post_import_scene_fps, do_fix):
    """
    Steps:
      1) Gather each object's original max keyframe (before scaling).
      2) Compute scale = original_fps / 25 if FBX used 25fps.
      3) Scale keys and NLA strips.
      4) Clamp keys to floor(max_original * scale).
      5) Clamp strip.frame_end accordingly.
    """
    if not do_fix:
        return

    FBX_FPS = 25.0
    try:
        post = float(post_import_scene_fps)
    except Exception:
        post = FBX_FPS

    if post != FBX_FPS:
        # If FBX didn't force 25 fps, skip auto-scaling
        return

    if original_fps not in (30, 60):
        # Only auto-fix for requested target fps (30 and 60)
        return

    scale = float(original_fps) / FBX_FPS  # 1.2 or 2.4 etc.

    if abs(scale - 1.0) < 1e-6:
        return

    # 1) Gather original max per-object
    original_max_map = {}
    for obj in imported_objects:
        frames = gather_original_frames_for_object(obj)
        if frames:
            original_max_map[obj.name] = max(frames)
        else:
            original_max_map[obj.name] = None

    # 2) Perform scaling
    for obj in imported_objects:
        if obj is None:
            continue
        # scale direct action
        if obj.animation_data and obj.animation_data.action:
            try:
                scale_action_keyframes(obj.animation_data.action, scale)
            except Exception:
                pass

        # scale NLA strips and their action keyframes
        if obj.animation_data and hasattr(obj.animation_data, "nla_tracks"):
            for track in obj.animation_data.nla_tracks:
                for strip in track.strips:
                    try:
                        strip.frame_start = strip.frame_start * scale
                        strip.frame_end = strip.frame_end * scale
                    except Exception:
                        pass
                    try:
                        if strip.action:
                            scale_action_keyframes(strip.action, scale)
                    except Exception:
                        pass

    # 3) Clamp to floored scaled max per-object
    for obj in imported_objects:
        if obj is None:
            continue
        orig_max = original_max_map.get(obj.name, None)
        if orig_max is None:
            # nothing to clamp if no original frames
            continue
        max_scaled = int(math.floor(orig_max * scale + 1e-8))  # floor with small epsilon

        # clamp direct action keyframes
        if obj.animation_data and obj.animation_data.action:
            try:
                clamp_action_to_max_frame(obj.animation_data.action, max_scaled)
            except Exception:
                pass

        # clamp NLA strips
        if obj.animation_data and hasattr(obj.animation_data, "nla_tracks"):
            for track in obj.animation_data.nla_tracks:
                for strip in track.strips:
                    try:
                        if strip.frame_end > max_scaled:
                            strip.frame_end = max_scaled
                            if strip.frame_start > strip.frame_end:
                                strip.frame_start = strip.frame_end
                    except Exception:
                        pass
                    try:
                        if strip.action:
                            clamp_action_to_max_frame(strip.action, max_scaled)
                    except Exception:
                        pass

# ---------------------
# Main Import Operator
# ---------------------
class O
