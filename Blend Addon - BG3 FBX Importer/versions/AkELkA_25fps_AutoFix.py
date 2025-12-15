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

BAD_FPS = 25
FIX_TAG = "akelka_25fps_fixed"


# -------------------------------------------------------
# Scene properties (create only if not present)
# -------------------------------------------------------
if not hasattr(bpy.types.Scene, "akelka_target_fps"):
    bpy.types.Scene.akelka_target_fps = bpy.props.EnumProperty(
        name="Target FPS",
        description="FPS to convert imported BG3 animations to",
        items=[
            ('30', "30 FPS (Default)", "Standard BG3 framerate"),
            ('60', "60 FPS (Cinematic)", "High framerate for cinematics"),
        ],
        default='30'
    )

if not hasattr(bpy.types.Scene, "akelka_autofix_enabled"):
    bpy.types.Scene.akelka_autofix_enabled = bpy.props.BoolProperty(
        name="Enable Auto-Fix",
        description="Automatically fix animations imported at 25 FPS (Noira importer only when 'Is animation' checked)",
        default=True
    )

if not hasattr(bpy.types.Scene, "akelka_autofix_running"):
    bpy.types.Scene.akelka_autofix_running = bpy.props.BoolProperty(
        name="AutoFix Running (internal)",
        description="Internal lock to prevent handler re-entry",
        default=False
    )


# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
def get_imported_armatures_unfixed():
    """Return armatures imported by Noira that were not yet fixed."""
    result = []
    for obj in bpy.data.objects:
        if obj and obj.type == 'ARMATURE' and "source_path" in obj.keys():
            if not obj.get(FIX_TAG, False):
                result.append(obj)
    return result


def collect_actions_for_armature(arm_obj):
    """Gather all actions related to an imported armature."""
    actions = set()

    ad = getattr(arm_obj, "animation_data", None)
    if ad:
        # Direct action
        try:
            if getattr(ad, "action", None):
                actions.add(ad.action)
        except Exception:
            pass

        # NLA actions
        try:
            for track in getattr(ad, "nla_tracks", []):
                for strip in getattr(track, "strips", []):
                    if getattr(strip, "action", None):
                        actions.add(strip.action)
        except Exception:
            pass

    # Heuristic: add actions referencing pose bones (conservative)
    try:
        for act in bpy.data.actions:
            if act in actions:
                continue
            for fc in getattr(act, "fcurves", []):
                if "pose.bones" in (fc.data_path or ""):
                    if act.users > 0 or (arm_obj.name and arm_obj.name in (act.name or "")):
                        actions.add(act)
                        break
    except Exception:
        pass

    return list(actions)


def fix_actions_timing(actions, target_fps):
    """Stretch keyframe timings from 25 → target FPS."""
    fixed_actions = 0
    ratio = float(target_fps) / BAD_FPS

    for action in actions:
        if not action or not getattr(action, "fcurves", None):
            continue
        try:
            for fcurve in action.fcurves:
                for kp in fcurve.keyframe_points:
                    kp.co[0] = kp.co[0] * ratio
                    try:
                        kp.handle_left[0] = kp.handle_left[0] * ratio
                        kp.handle_right[0] = kp.handle_right[0] * ratio
                    except Exception:
                        pass
            fixed_actions += 1
        except Exception as e:
            print(f"AkELkA AutoFix: failed on {getattr(action, 'name', '<no name>')}: {e}")
    return fixed_actions


def fix_imported_armature(arm_obj, target_fps, scene_for_fps=None):
    """
    Apply FPS fix to actions related to arm_obj.
    If scene_for_fps provided, update that scene's render.fps (used by handler).
    """
    actions = collect_actions_for_armature(arm_obj)
    fixed = fix_actions_timing(actions, target_fps)

    # Set scene FPS if provided (handler passes the scene it received)
    try:
        if scene_for_fps is not None:
            scene_for_fps.render.fps = target_fps
            scene_for_fps.render.fps_base = 1.0
        else:
            # fallback to current context scene
            bpy.context.scene.render.fps = target_fps
            bpy.context.scene.render.fps_base = 1.0
    except Exception:
        pass

    # Mark as fixed (prevents double-stretch)
    try:
        arm_obj[FIX_TAG] = True
    except Exception:
        pass

    return fixed, len(actions)


# -------------------------------------------------------
# Manual operator
# -------------------------------------------------------
class AKELKA_OT_Apply25FPSFix(bpy.types.Operator):
    bl_idname = "akelka.apply_25fps_fix"
    bl_label = "Apply FPS Fix Now"
    bl_description = "Convert BG3’s 25 FPS animations (Noira imports only) to the target FPS"

    def execute(self, context):
        target = int(context.scene.akelka_target_fps)
        imported = get_imported_armatures_unfixed()

        if not imported:
            self.report({'INFO'}, "No imported BG3 animations found (or already fixed).")
            return {'CANCELLED'}

        total_fixed = 0
        total_actions = 0

        for arm in imported:
            f, a = fix_imported_armature(arm, target, context.scene)
            total_fixed += f
            total_actions += a

        self.report({'INFO'}, f"Fixed {total_fixed} actions (from {total_actions} detected).")
        return {'FINISHED'}


# -------------------------------------------------------
# Auto-fix handler (Noira only) — with lock to prevent re-entry
# -------------------------------------------------------
def akelka_noira_autofix_handler(scene):
    # quick exit if user disabled auto-fix
    if not getattr(scene, "akelka_autofix_enabled", False):
        return

    # Only when Noira importer "Is animation" is enabled
    if not getattr(scene, "bool_isAnimation", False):
        return

    # Prevent re-entry / infinite loop
    if getattr(scene, "akelka_autofix_running", False):
        return

    armatures = get_imported_armatures_unfixed()
    if not armatures:
        return

    # Lock
    try:
        scene.akelka_autofix_running = True
    except Exception:
        # If we can't set the lock, bail out to be safe
        return

    target = int(getattr(scene, "akelka_target_fps", 30))
    total_fixed = 0
    total_actions = 0

    try:
        for arm in armatures:
            f, a = fix_imported_armature(arm, target, scene_for_fps=scene)
            total_fixed += f
            total_actions += a
        print(f"[AkELkA AutoFix] Fixed {total_fixed} actions (from {total_actions}).")
    except Exception as e:
        print(f"[AkELkA AutoFix] Unexpected error during autofix: {e}")
    finally:
        # Unlock (important to avoid permanent lock)
        try:
            scene.akelka_autofix_running = False
        except Exception:
            pass


# -------------------------------------------------------
# Panels
# -------------------------------------------------------
class AKELKA_PT_AnimationRoot(bpy.types.Panel):
    bl_label = "Animation"
    bl_idname = "AKELKA_PT_AnimationRoot"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"

    def draw(self, context):
        self.layout.label(text="Animation Tools")


class AKELKA_PT_25FPSAutoFix(bpy.types.Panel):
    bl_label = "AkELkA - 25fps AutoFix - BG3 FBX Anim"
    bl_idname = "AKELKA_PT_25FPSAutoFix"
    bl_parent_id = "AKELKA_PT_AnimationRoot"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        info = layout.box()
        info.label(text="What this fixes:", icon='INFO')
        info.label(text="• BG3 FBX imported animations at 25 FPS")
        info.label(text="• Only when 'Is animation' is checked")
        info.label(text="• Stretches keyframes 25 → target FPS (30/60)")

        layout.separator()

        box = layout.box()
        box.prop(scene, "akelka_target_fps")
        box.prop(scene, "akelka_autofix_enabled", toggle=True)

        layout.separator()

        layout.operator("akelka.apply_25fps_fix", icon='RECOVER_LAST')


# -------------------------------------------------------
# Register
# -------------------------------------------------------
classes = (
    AKELKA_PT_AnimationRoot,
    AKELKA_PT_25FPSAutoFix,
    AKELKA_OT_Apply25FPSFix,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)

    handler = akelka_noira_autofix_handler
    if handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(handler)


def unregister():
    handler = akelka_noira_autofix_handler
    if handler in bpy.app.handlers.depsgraph_update_post:
        try:
            bpy.app.handlers.depsgraph_update_post.remove(handler)
        except Exception:
            pass

    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass


if __name__ == "__main__":
    register()
