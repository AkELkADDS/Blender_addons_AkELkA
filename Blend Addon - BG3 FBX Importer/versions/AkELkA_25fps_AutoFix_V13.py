bl_info = {
    "name": "AkELkA - BG3 FBX Importer",
    "author": "AkELkA",
    "version": (1, 3, 5),
    "blender": (3, 0, 0),
    "location": "View3D > N Panel > Akelka Tools > Animation",
    "description": "Adjusted standalone version of No-I-ra's BG3 FBX importer. Fixes issues with framerate and animation length.",
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
# rotation bake (keys only integer frames inside final clamped range)
# ---------------------
def rotateObjectEachFrame(obj):
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
                    try:
                        frames.add(float(kp.co.x))
                    except Exception:
                        pass

        if hasattr(ad, "nla_tracks"):
            for track in ad.nla_tracks:
                for strip in track.strips:
                    try:
                        frames.add(float(strip.frame_start))
                        frames.add(float(strip.frame_end))
                    except Exception:
                        pass
                    try:
                        if strip.action:
                            for fcurve in strip.action.fcurves:
                                for kp in fcurve.keyframe_points:
                                    try:
                                        frames.add(float(kp.co.x))
                                    except Exception:
                                        pass
                    except Exception:
                        pass

    if not frames:
        return

    # integer frames after floor; avoid fractional
    int_frames = set()
    for f in frames:
        try:
            int_frames.add(int(math.floor(f + 1e-8)))
        except Exception:
            pass

    if not int_frames:
        return

    max_frame = max(int_frames)
    int_frames = {f for f in int_frames if f <= max_frame}

    for frame in sorted(int_frames):
        try:
            scene.frame_set(frame)
            obj.rotation_euler = (0, 0, 0)
            obj.keyframe_insert("rotation_euler", index=-1)
        except Exception:
            pass

# ---------------------
# Normalization helpers
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
                try:
                    frames.add(float(kp.co.x))
                except Exception:
                    pass

    if hasattr(ad, "nla_tracks"):
        for track in ad.nla_tracks:
            for strip in track.strips:
                try:
                    frames.add(float(strip.frame_start))
                    frames.add(float(strip.frame_end))
                except Exception:
                    pass
                try:
                    sact = strip.action
                    if sact:
                        for fcurve in sact.fcurves:
                            for kp in fcurve.keyframe_points:
                                try:
                                    frames.add(float(kp.co.x))
                                except Exception:
                                    pass
                except Exception:
                    pass
    return frames

def normalize_fcurve_to_integer_frames(fcurve, frames_to_keep):
    """
    Evaluate the fcurve at each integer frame in frames_to_keep,
    remove old keys, insert integer keys with evaluated values.
    """
    if not fcurve:
        return
    frames_sorted = sorted(frames_to_keep)
    if not frames_sorted:
        return

    # evaluate values at those frames
    new_points = []
    for fr in frames_sorted:
        try:
            val = fcurve.evaluate(fr)
            new_points.append((fr, val))
        except Exception:
            pass

    # clear existing points and insert new integer-keyed points
    try:
        # work on copy of keypoints to avoid mutating while iterating
        fcurve.keyframe_points.clear()
    except Exception:
        # fallback: try to remove individually
        try:
            for kp in list(fcurve.keyframe_points):
                try:
                    fcurve.keyframe_points.remove(kp)
                except Exception:
                    pass
        except Exception:
            pass

    for fr, val in new_points:
        try:
            kp = fcurve.keyframe_points.insert(fr, val, options={'FAST'})
            # set handles to integer to avoid fractional handle x positions
            try:
                kp.handle_left.x = fr
                kp.handle_right.x = fr
            except Exception:
                pass
        except Exception:
            pass

    try:
        fcurve.update()
    except Exception:
        pass

def normalize_action_to_integer_frames(action, max_frame):
    """Normalize every fcurve in action to integer frames (<= max_frame)."""
    if not action:
        return
    for fcurve in action.fcurves:
        # determine integer frames to keep from existing keyframes (floored)
        frames = set()
        for kp in fcurve.keyframe_points:
            try:
                if kp.co.x <= max_frame + 1e-8:
                    frames.add(int(math.floor(kp.co.x + 1e-8)))
            except Exception:
                pass
        if not frames:
            continue
        normalize_fcurve_to_integer_frames(fcurve, frames)

def clamp_and_normalize_strip(strip, max_frame):
    """Clamp a strip and normalize its action if any, both floored to integers."""
    try:
        # floor the start and end then clamp
        start = int(math.floor(strip.frame_start + 1e-8))
        end = int(math.floor(strip.frame_end + 1e-8))
        if end > max_frame:
            end = max_frame
        if start > end:
            start = end
        strip.frame_start = start
        strip.frame_end = end
    except Exception:
        pass

    try:
        if strip.action:
            normalize_action_to_integer_frames(strip.action, strip.frame_end)
    except Exception:
        pass

# ---------------------
# Main scaling + clamp + normalize routine
# ---------------------
def fix_imported_animation_timing_and_clamp(imported_objects, original_fps, post_import_scene_fps, do_fix):
    """
    Steps:
      1) Gather each object's original max keyframe (before scaling).
      2) Compute scale = original_fps / 25 if FBX used 25fps.
      3) Scale keys and NLA strips.
      4) Clamp keys to floor(max_original * scale).
      5) Normalize every fcurve to integer frames (evaluate and rebuild).
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

    # 2) Scale actions & strips
    for obj in imported_objects:
        if obj is None:
            continue
        if obj.animation_data and obj.animation_data.action:
            try:
                scale_action_keyframes(obj.animation_data.action, scale)
            except Exception:
                pass

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

    # 3) Clamp and normalize to integer frames
    for obj in imported_objects:
        if obj is None:
            continue
        orig_max = original_max_map.get(obj.name, None)
        if orig_max is None:
            continue
        max_scaled = int(math.floor(orig_max * scale + 1e-8))

        # Normalize main action if present
        if obj.animation_data and obj.animation_data.action:
            try:
                normalize_action_to_integer_frames(obj.animation_data.action, max_scaled)
            except Exception:
                pass

        # Clamp and normalize all NLA strips and their actions
        if obj.animation_data and hasattr(obj.animation_data, "nla_tracks"):
            for track in obj.animation_data.nla_tracks:
                for strip in track.strips:
                    try:
                        clamp_and_normalize_strip(strip, max_scaled)
                    except Exception:
                        pass

# ---------------------
# Main Import Operator
# ---------------------
class OBJECT_OT_Noira_FBXImporter(Operator, ImportHelper):
    bl_idname = "object.noira_fbx_importer"
    bl_label = "Import BG3 FBX"
    filter_glob: StringProperty(default='*.fbx', options={'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        scene = context.scene
        original_fps = scene.render.fps
        original_frame_start = scene.frame_start
        original_frame_end = scene.frame_end

        bpy.ops.outliner.orphans_purge()

        if bpy.context.active_object is not None:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
            deselect_all()

        importedObjectsAllFiles = []

        for current_file in self.files:
            filepath = os.path.join(self.directory, current_file.name)
            fileNameNoExt = os.path.splitext(os.path.basename(current_file.name))[0]

            # optional prefix/suffix removal
            if hasattr(context.scene, "str_suffixe_to_remove") and context.scene.str_suffixe_to_remove:
                split_string = fileNameNoExt.split(context.scene.str_suffixe_to_remove)
                if split_string[0]:
                    fileNameNoExt = split_string[0]

            if hasattr(context.scene, "str_prefix_to_remove") and context.scene.str_prefix_to_remove:
                split_string = fileNameNoExt.split(context.scene.str_prefix_to_remove)
                if len(split_string) > 1 and split_string[1]:
                    fileNameNoExt = split_string[1]

            deselect_all()

            # Import FBX. The importer may set scene.render.fps to whatever the FBX file says (often 25).
            try:
                bpy.ops.import_scene.fbx(filepath=str(filepath), axis_forward='-Z', axis_up='Y', global_scale=100)
            except Exception as e:
                self.report({'WARNING'}, f"FBX import failed for {filepath}: {e}")
                continue

            # Capture FPS after import (FBX importer may have changed it)
            post_import_fps = context.scene.render.fps

            # Restore the user's original FPS right away
            try:
                context.scene.render.fps = original_fps
            except Exception:
                pass

            importedObjects = list(bpy.context.selected_objects)
            importedObjectsAllFiles.extend(importedObjects)

            deselect_all()

            armatureObj = None
            meshes = []

            # Gather normal processing (names, transforms, parenting, etc.)
            for obj in importedObjects:
                select_object(obj, True)

                # rotate to match expected orientation
                try:
                    obj.rotation_euler = (0, 0, 0)
                    obj.rotation_euler.rotate_axis('X', math.radians(90))
                except Exception:
                    pass

                # apply transforms
                try:
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    obj.matrix_basis.identity()
                except Exception:
                    pass

                # set source path custom property
                try:
                    obj["source_path"] = filepath
                except Exception:
                    pass

                if obj.type == 'ARMATURE':
                    armatureObj = obj
                    try:
                        obj.name = fileNameNoExt
                    except Exception:
                        pass

                    try:
                        armature_data = obj.data
                        armature_data.name = obj.name
                    except Exception:
                        pass

                    if context.scene.bool_isAnimation == False:
                        if obj.animation_data:
                            obj.animation_data.action = None
                            try:
                                obj.animation_data_clear()
                            except Exception:
                                pass

                    try:
                        apply_all_transforms(obj)
                    except Exception:
                        pass

                    try:
                        bpy.ops.object.mode_set(mode='OBJECT')
                        select_object(obj, True)
                        bpy.ops.object.mode_set(mode='EDIT')

                        bone_Dummy_Root = obj.data.edit_bones.new("Dummy_Root")
                        bone_Dummy_Root.head = (0, 0, 0)
                        bone_Dummy_Root.tail = (0, 0, 1)

                        bpy.context.view_layer.update()

                        for bone in obj.data.edit_bones:
                            if bone.name != "Dummy_Root":
                                if bone.parent is None:
                                    bone.parent = bone_Dummy_Root

                        bpy.context.view_layer.update()
                        bpy.ops.object.mode_set(mode='OBJECT')
                    except Exception:
                        try:
                            bpy.ops.object.mode_set(mode='OBJECT')
                        except Exception:
                            pass

                else:
                    if obj.type == "MESH":
                        meshes.append(obj)
                        try:
                            if obj.data.materials:
                                obj.data.materials.clear()
                        except Exception:
                            pass
                        try:
                            if obj.data:
                                obj.data.name = obj.name
                        except Exception:
                            pass

            deselect_all()

            # Post-import mesh/armature handling (head parenting, vertex groups, collections)
            if armatureObj:
                for mesh in meshes:
                    select_object(mesh, True)
                    try:
                        if "Head" in mesh.name:
                            if mesh.parent is None:
                                mesh.parent = armatureObj
                                add_armature_modifier(armatureObj, mesh)

                            if "Ears" in mesh.name:
                                try:
                                    vertGroup = bpy.context.active_object.vertex_groups.new(name='Head_M')
                                    verts = [v.index for v in mesh.data.vertices]
                                    vertGroup.add(verts, 1.0, 'REPLACE')
                                except Exception:
                                    pass
                    except Exception:
                        pass

                colName = armatureObj.name
                split_string = colName.split("_Base")
                if split_string and split_string[0]:
                    colName = split_string[0]

                active_collection = bpy.context.collection

                for obj in importedObjects:
                    try:
                        collection_name = colName
                        new_collection = bpy.data.collections.get(collection_name)
                        if new_collection is None:
                            new_collection = bpy.data.collections.new(collection_name)
                        else:
                            if new_collection.children:
                                try:
                                    old_parent = new_collection.children[0]
                                    old_parent.children.unlink(new_collection)
                                except Exception:
                                    pass

                        if not is_collection_child(new_collection, active_collection):
                            try:
                                active_collection.children.link(new_collection)
                            except Exception:
                                pass

                        obj_ref = bpy.data.objects.get(obj.name)
                        if obj_ref:
                            try:
                                old_parent = obj_ref.users_collection[0]
                                old_parent.objects.unlink(obj_ref)
                                new_collection.objects.link(obj_ref)
                            except Exception:
                                pass
                    except Exception:
                        pass

                deselect_all()
                select_object(armatureObj, True)

            # APPLY AUTO-FPS FIX + CLAMP + NORMALIZE
            try:
                fix_imported_animation_timing_and_clamp(importedObjects, original_fps, post_import_fps, context.scene.bool_isAnimation)
            except Exception:
                pass

            # Now run rotation-bake (which will only key the clamped integer frames)
            for obj in importedObjects:
                if obj and obj.type == 'ARMATURE':
                    try:
                        rotateObjectEachFrame(obj)
                    except Exception:
                        pass

        # cleanup & restore original frame range just in case
        try:
            scene.frame_start = original_frame_start
            scene.frame_end = original_frame_end
        except Exception:
            pass

        deselect_all()
        return {'FINISHED'}

# ---------------------
# UI Panel
# ---------------------
class PANEL_PT_Akelka_BG3Animation(bpy.types.Panel):
    bl_idname = "PANEL_PT_Akelka_BG3Animation"
    bl_label = "AkELkA Animation Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(context.scene, "str_prefix_to_remove")
        row = layout.row()
        row.prop(context.scene, "str_suffixe_to_remove")
        row = layout.row()
        row.prop(context.scene, "bool_isAnimation", text="Is animation")
        row = layout.row()
        row.operator("object.noira_fbx_importer", text="Import FBX Files")

# ---------------------
# Registration
# ---------------------
classes = (
    PANEL_PT_Akelka_BG3Animation,
    OBJECT_OT_Noira_FBXImporter,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.str_prefix_to_remove = StringProperty(
        name="Prefix",
        description="to remove from the file name",
        default=""
    )
    bpy.types.Scene.str_suffixe_to_remove = StringProperty(
        name="Suffixe",
        description="to remove from the file name",
        default=""
    )
    bpy.types.Scene.bool_isAnimation = BoolProperty(
        name="Is Animation",
        description="Check to import an animation (auto-fix 25->30/60)",
        default=False
    )

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    try:
        del bpy.types.Scene.str_prefix_to_remove
        del bpy.types.Scene.str_suffixe_to_remove
        del bpy.types.Scene.bool_isAnimation
    except Exception:
        pass

if __name__ == "__main__":
    register()
