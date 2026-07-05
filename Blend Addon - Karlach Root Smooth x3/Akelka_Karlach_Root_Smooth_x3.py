bl_info = {
    "name": "Karlach Root Smooth x3",
    "author": "Akelka",
    "version": (1, 2, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Item",
    "description": "Smooth Root_M x3 on bottom NLA (active armature). Whole animation x4 on second armature.",
    "category": "Animation",
}

import bpy
import traceback
from contextlib import contextmanager

TARGET_BONE = "Root_M"
SMOOTH_FACTOR = 1.0
DEFAULT_NLA_PASSES = 3
DEFAULT_WHOLE_PASSES = 4


class KarlachRootSmoothSettings(bpy.types.PropertyGroup):
    show_info: bpy.props.BoolProperty(
        name="Show Info",
        description="Show armature and NLA/action details",
        default=True,
    )
    nla_passes: bpy.props.IntProperty(
        name="NLA",
        description="Gaussian smooth passes for Root_M on bottom NLA (active armature)",
        default=DEFAULT_NLA_PASSES,
        min=0,
        max=20,
        soft_max=10,
    )
    whole_passes: bpy.props.IntProperty(
        name="Whole",
        description="Gaussian smooth passes for whole animation (second armature)",
        default=DEFAULT_WHOLE_PASSES,
        min=0,
        max=20,
        soft_max=10,
    )


def get_settings(context):
    return context.scene.karlach_root_smooth


def get_selected_armatures(context):
    return [o for o in context.selected_objects if o.type == 'ARMATURE']


def get_smooth_targets(context):
    """Return (primary, secondary) armatures. Primary is active; secondary is the other selection."""
    arms = get_selected_armatures(context)
    if not arms:
        return None, None

    primary = context.active_object if context.active_object and context.active_object.type == 'ARMATURE' else None
    if primary is None:
        primary = arms[0]

    secondary = None
    if len(arms) >= 2:
        others = [a for a in arms if a != primary]
        if others:
            secondary = others[0]

    return primary, secondary


def get_button_label(primary, secondary, nla_passes, whole_passes):
    if secondary is not None and whole_passes > 0:
        return f"Smooth {TARGET_BONE} x{nla_passes} + Whole Anim x{whole_passes}"
    return f"Smooth {TARGET_BONE} x{nla_passes}"


def get_bottom_nla_strip(arm_obj):
    """Return (action, track_name, strip) for the bottom-most NLA track."""
    ad = arm_obj.animation_data
    if ad is None or len(ad.nla_tracks) == 0:
        return None, None, None

    # Blender lists NLA tracks bottom-to-top: index 0 = bottom, -1 = top.
    bottom_track = ad.nla_tracks[0]
    for strip in bottom_track.strips:
        if strip.action is not None:
            return strip.action, bottom_track.name, strip
    return None, bottom_track.name, None


def _window_region(area):
    for region in area.regions:
        if region.type == 'WINDOW':
            return region
    return None


def _find_area(screen, area_type):
    for area in screen.areas:
        if area.type == area_type:
            return area
    return None


def _object_override(ctx, arm_obj):
    return {
        'window': ctx.window,
        'screen': ctx.screen,
        'active_object': arm_obj,
        'object': arm_obj,
        'selected_objects': [arm_obj],
        'selected_editable_objects': [arm_obj],
    }


def _run_in_editor(ctx, arm_obj, area_type, operator_name, **operator_kwargs):
    screen = ctx.screen
    area = _find_area(screen, area_type)
    restored_type = None

    if area is None:
        if not screen.areas:
            return False
        area = screen.areas[0]
        restored_type = area.type
        area.type = area_type

    region = _window_region(area)
    if region is None:
        if restored_type is not None:
            area.type = restored_type
        return False

    override = {**_object_override(ctx, arm_obj), 'area': area, 'region': region}
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


def _find_active_strip(ad):
    for track in ad.nla_tracks:
        for strip in track.strips:
            if strip.active:
                return strip
    return None


def _find_track_for_strip(ad, strip):
    for track in ad.nla_tracks:
        for s in track.strips:
            if s == strip:
                return track
    return None


def _save_strip_selection(ad):
    saved = []
    for track in ad.nla_tracks:
        for strip in track.strips:
            saved.append((strip, strip.select))
    return saved


def _restore_strip_selection(saved):
    for strip, selected in saved:
        try:
            strip.select = selected
        except Exception:
            pass


def _select_strip(ad, strip):
    track = _find_track_for_strip(ad, strip)
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
        'active_strip': _find_active_strip(ad),
        'active_track': ad.nla_tracks.active,
        'strip_select': _save_strip_selection(ad),
        'active': ctx.view_layer.objects.active,
        'mode': ctx.mode,
    }
    entered_tweak = False

    try:
        arm_obj.select_set(True)
        ctx.view_layer.objects.active = arm_obj

        active_strip = _find_active_strip(ad)
        if ad.use_tweak_mode and active_strip is not None and active_strip != strip:
            _run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_exit')

        _select_strip(ad, strip)

        if ad.use_tweak_mode and _find_active_strip(ad) == strip:
            entered_tweak = False
        elif _run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_enter'):
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
            _run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_exit')

        if saved['use_tweak_mode'] and saved['active_strip'] is not None:
            _select_strip(ad, saved['active_strip'])
            if not ad.use_tweak_mode:
                _run_in_editor(ctx, arm_obj, 'NLA_EDITOR', 'nla.tweakmode_enter')
            _restore_strip_selection(saved['strip_select'])
        else:
            try:
                ad.action = saved['action']
                ad.nla_tracks.active = saved['active_track']
            except Exception:
                pass
            _restore_strip_selection(saved['strip_select'])

        if saved['active'] and saved['active'].name in ctx.view_layer.objects:
            try:
                ctx.view_layer.objects.active = saved['active']
            except Exception:
                pass


def _is_bone_fcurve(fcu):
    return fcu.data_path.startswith('pose.bones')


def _extract_bone_name_from_path(dp):
    try:
        start = dp.find('["')
        end = dp.find('"]', start + 1)
        if start == -1 or end == -1:
            start = dp.find("['")
            end = dp.find("']", start + 1)
            if start == -1 or end == -1:
                return None
            return dp[start + 2:end]
        return dp[start + 2:end]
    except Exception:
        return None


def apply_graph_gaussian_smooth(arm_obj, action, bone_names=None, factor=1.0):
    """Apply graph.gaussian_smooth to bone F-curves. bone_names=None smooths all bones."""
    ctx = bpy.context
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        raise RuntimeError("Provided object must be an Armature.")
    if action is None or not action.fcurves:
        return 0

    filter_bones = bone_names is not None
    bone_filter = set(bone_names) if bone_names else set()

    saved_states = []
    for fcu in action.fcurves:
        kp_sel = [kp.select_control_point for kp in fcu.keyframe_points]
        saved_states.append((fcu, fcu.select, kp_sel))

    target_fcurves = []
    for fcu in action.fcurves:
        if not _is_bone_fcurve(fcu):
            continue
        if filter_bones:
            bone = _extract_bone_name_from_path(fcu.data_path)
            if bone is None or bone not in bone_filter:
                continue
        target_fcurves.append(fcu)

    if not target_fcurves:
        return 0

    for fcu in action.fcurves:
        try:
            if fcu in target_fcurves:
                fcu.select = True
                for kp in fcu.keyframe_points:
                    kp.select_control_point = True
            else:
                fcu.select = False
                for kp in fcu.keyframe_points:
                    kp.select_control_point = False
        except Exception:
            pass

    arm_obj.select_set(True)
    ctx.view_layer.objects.active = arm_obj
    if not _run_in_editor(ctx, arm_obj, 'GRAPH_EDITOR', 'graph.gaussian_smooth', factor=factor):
        raise RuntimeError("graph.gaussian_smooth could not run (no editable animation data).")

    for fcu, fcu_sel, kp_sel in saved_states:
        try:
            fcu.select = fcu_sel
            for kp, sel in zip(fcu.keyframe_points, kp_sel):
                kp.select_control_point = sel
        except Exception:
            pass

    try:
        ctx.view_layer.update()
        action.update_tag()
    except Exception:
        pass

    return len(target_fcurves)


def smooth_root_nla(context, arm, passes):
    if passes <= 0:
        raise RuntimeError("NLA smooth passes must be at least 1.")

    if arm.pose.bones.get(TARGET_BONE) is None:
        raise RuntimeError(f"Bone '{TARGET_BONE}' not found on '{arm.name}'.")

    action, track_name, strip = get_bottom_nla_strip(arm)
    if strip is None or action is None:
        msg = f"No action on bottom NLA track for '{arm.name}'."
        if track_name:
            msg = f"No action on bottom NLA track '{track_name}' ({arm.name})."
        raise RuntimeError(msg)

    with nla_tweak_strip(context, arm, strip) as tweak_action:
        total = 0
        for _ in range(passes):
            total = apply_graph_gaussian_smooth(
                arm,
                tweak_action,
                bone_names={TARGET_BONE},
                factor=SMOOTH_FACTOR,
            )

    if total == 0:
        raise RuntimeError(f"No F-curves found for '{TARGET_BONE}' in '{action.name}' ({arm.name}).")

    return total, track_name, action.name


def smooth_whole_action(context, arm, passes):
    if passes <= 0:
        raise RuntimeError("Whole animation smooth passes must be at least 1.")

    ad = arm.animation_data
    if ad is None or ad.action is None:
        raise RuntimeError(f"No active action on '{arm.name}'.")

    action = ad.action
    total = 0
    for _ in range(passes):
        total = apply_graph_gaussian_smooth(
            arm,
            action,
            bone_names=None,
            factor=SMOOTH_FACTOR,
        )

    if total == 0:
        raise RuntimeError(f"No bone F-curves found in '{action.name}' ({arm.name}).")

    return total, action.name


class ANIM_OT_karlach_root_smooth_x3(bpy.types.Operator):
    bl_idname = "anim.karlach_root_smooth_x3"
    bl_label = "Smooth Root_M x3"
    bl_description = (
        "Active armature: smooth Root_M x3 on bottom NLA track. "
        "Second selected armature: smooth whole animation x4 on its action."
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = get_settings(context)
        primary, secondary = get_smooth_targets(context)
        if primary is None:
            self.report({'ERROR'}, "Select an armature first.")
            return {'CANCELLED'}

        nla_passes = settings.nla_passes
        whole_passes = settings.whole_passes

        if nla_passes <= 0 and (secondary is None or whole_passes <= 0):
            self.report({'ERROR'}, "Set at least one smooth pass count above 0.")
            return {'CANCELLED'}

        saved_active = context.view_layer.objects.active
        messages = []
        warnings = []

        try:
            if nla_passes > 0:
                try:
                    curves, track_name, action_name = smooth_root_nla(context, primary, nla_passes)
                    messages.append(
                        f"{primary.name}: {TARGET_BONE} x{nla_passes} "
                        f"(NLA '{track_name}', {curves} curves)"
                    )
                except Exception as e:
                    self.report({'ERROR'}, f"Active armature failed: {e}")
                    print("Karlach Root Smooth x3 error (primary):", e)
                    print(traceback.format_exc())
                    return {'CANCELLED'}
            else:
                warnings.append(f"{primary.name}: NLA smooth skipped (passes=0)")

            if secondary is not None and whole_passes > 0:
                try:
                    curves, action_name = smooth_whole_action(context, secondary, whole_passes)
                    messages.append(
                        f"{secondary.name}: whole animation x{whole_passes} "
                        f"({action_name}, {curves} curves)"
                    )
                except Exception as e:
                    warnings.append(f"{secondary.name}: {e}")
                    print("Karlach Root Smooth x3 error (secondary):", e)
                    print(traceback.format_exc())
            elif secondary is not None and whole_passes <= 0:
                warnings.append(f"{secondary.name}: whole smooth skipped (passes=0)")

            if not messages and warnings:
                self.report({'WARNING'}, " | ".join(warnings))
            elif warnings:
                self.report({'WARNING'}, " | ".join(messages + warnings))
            else:
                self.report({'INFO'}, " | ".join(messages))
            return {'FINISHED'}

        finally:
            if saved_active and saved_active.name in context.view_layer.objects:
                try:
                    context.view_layer.objects.active = saved_active
                except Exception:
                    pass


class VIEW3D_PT_karlach_root_smooth(bpy.types.Panel):
    bl_label = "Root Smooth"
    bl_idname = "VIEW3D_PT_karlach_root_smooth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Item"
    bl_order = 100

    @classmethod
    def poll(cls, context):
        return get_smooth_targets(context)[0] is not None

    def draw(self, context):
        layout = self.layout
        settings = get_settings(context)
        primary, secondary = get_smooth_targets(context)

        col = layout.column(align=True)
        header = col.row(align=True)
        header.prop(settings, "show_info", text="Info", icon='INFO', toggle=True)
        header.prop(settings, "nla_passes", text="NLA")
        header.prop(settings, "whole_passes", text="Whole")

        if settings.show_info:
            info = col.column(align=True)
            if primary:
                action, track_name, _strip = get_bottom_nla_strip(primary)
                info.label(text=f"Active: {primary.name}", icon='ARMATURE_DATA')
                if track_name and action:
                    info.label(text=f"  NLA: {track_name}")
                    info.label(text=f"  {TARGET_BONE} x{settings.nla_passes}")
                elif track_name:
                    info.label(text=f"  NLA: {track_name} (no action)", icon='ERROR')
                else:
                    info.label(text="  No NLA tracks", icon='ERROR')

            if secondary:
                sec_action = secondary.animation_data.action if secondary.animation_data else None
                info.separator()
                info.label(text=f"Other: {secondary.name}", icon='ARMATURE_DATA')
                if sec_action:
                    info.label(text=f"  Action: {sec_action.name}")
                    info.label(text=f"  Whole animation x{settings.whole_passes}")
                else:
                    info.label(text="  No active action", icon='ERROR')

        col.operator(
            "anim.karlach_root_smooth_x3",
            icon='SMOOTHCURVE',
            text=get_button_label(primary, secondary, settings.nla_passes, settings.whole_passes),
        )


classes = (
    KarlachRootSmoothSettings,
    ANIM_OT_karlach_root_smooth_x3,
    VIEW3D_PT_karlach_root_smooth,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.karlach_root_smooth = bpy.props.PointerProperty(type=KarlachRootSmoothSettings)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.karlach_root_smooth


if __name__ == "__main__":
    register()
