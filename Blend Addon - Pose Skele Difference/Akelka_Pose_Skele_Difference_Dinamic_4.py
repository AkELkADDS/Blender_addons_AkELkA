bl_info = {
    "name": "Armature Difference Pose Live Link",
    "author": "AkELkA",
    "version": (1, 0, 3),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N) > Akelka Tools",
    "description": "Live link one armature to another using matching bone names",
    "category": "Rigging",
}

import bpy

PREFIX_ROT = "DP_LIVE_ROT_"
PREFIX_LOC = "DP_LIVE_LOC_"
SCENE_PROP = "dp2_head_only"


def get_selected_armatures(context):
    return [obj for obj in context.selected_objects if obj and obj.type == 'ARMATURE']


def get_driver_and_driven(context):
    selected = get_selected_armatures(context)
    active = context.active_object

    if len(selected) != 2:
        return None, None, "Select exactly 2 armatures"

    if not active or active.type != 'ARMATURE':
        return None, None, "Make the armature you want to FOLLOW the active object"

    driven = active
    driver = [obj for obj in selected if obj != active][0]
    return driver, driven, None


def matches_filter(bone_name, head_only):
    if not head_only:
        return True
    n = bone_name.lower()
    return ("head" in n) or ("neck" in n)


def build_driver_bone_lookup(driver):
    exact = set(driver.pose.bones.keys())
    by_lower = {}
    for name in driver.pose.bones.keys():
        key = name.lower()
        if key not in by_lower:
            by_lower[key] = name
    return exact, by_lower


def resolve_driver_bone_name(driven_bone_name, exact, by_lower):
    if driven_bone_name in exact:
        return driven_bone_name
    return by_lower.get(driven_bone_name.lower())


def remove_live_constraints(armature_obj):
    if not armature_obj or armature_obj.type != 'ARMATURE':
        return 0

    removed = 0
    for pb in armature_obj.pose.bones:
        for c in list(pb.constraints):
            if c.name.startswith(PREFIX_ROT) or c.name.startswith(PREFIX_LOC):
                pb.constraints.remove(c)
                removed += 1
    return removed


def add_live_constraints(driver, driven, head_only=False):
    matched = 0
    skipped_filter = 0
    skipped_no_match = 0

    driver.data.pose_position = 'POSE'
    driven.data.pose_position = 'POSE'

    exact, by_lower = build_driver_bone_lookup(driver)

    for pb in driven.pose.bones:
        if not matches_filter(pb.name, head_only):
            skipped_filter += 1
            continue

        driver_bone = resolve_driver_bone_name(pb.name, exact, by_lower)
        if not driver_bone:
            skipped_no_match += 1
            continue

        matched += 1

        con_rot = pb.constraints.new('COPY_ROTATION')
        con_rot.name = PREFIX_ROT + pb.name
        con_rot.target = driver
        con_rot.subtarget = driver_bone
        con_rot.owner_space = 'WORLD'
        con_rot.target_space = 'WORLD'
        con_rot.mix_mode = 'REPLACE'
        con_rot.influence = 1.0

        con_loc = pb.constraints.new('COPY_LOCATION')
        con_loc.name = PREFIX_LOC + pb.name
        con_loc.target = driver
        con_loc.subtarget = driver_bone
        con_loc.owner_space = 'WORLD'
        con_loc.target_space = 'WORLD'
        con_loc.influence = 1.0

    return matched, skipped_filter, skipped_no_match


class ARMATURE_OT_dp2_live_enable(bpy.types.Operator):
    bl_idname = "armature.dp2_live_enable"
    bl_label = "Enable Live Link"
    bl_description = "Make the active armature follow the other selected armature in real time"

    def execute(self, context):
        driver, driven, err = get_driver_and_driven(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        head_only = getattr(context.scene, SCENE_PROP, False)

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        removed = remove_live_constraints(driven)

        bpy.ops.object.select_all(action='DESELECT')
        driven.select_set(True)
        context.view_layer.objects.active = driven
        bpy.ops.object.mode_set(mode='POSE')

        matched, skipped_filter, skipped_no_match = add_live_constraints(
            driver, driven, head_only=head_only
        )

        context.view_layer.update()

        if matched == 0:
            self.report({'WARNING'}, "No matching bones found")
        else:
            msg = f"Linked {matched} bones (removed {removed} old constraints)"
            if skipped_filter:
                msg += f"; {skipped_filter} skipped (Head/Neck filter is ON)"
            if skipped_no_match:
                msg += f"; {skipped_no_match} skipped (no matching name on driver)"
            self.report({'INFO'}, msg)

        return {'FINISHED'}


class ARMATURE_OT_dp2_live_disable(bpy.types.Operator):
    bl_idname = "armature.dp2_live_disable"
    bl_label = "Disable Live Link"
    bl_description = "Remove live link constraints from the active armature"

    def execute(self, context):
        active = context.active_object

        if not active or active.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an armature")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        removed = remove_live_constraints(active)
        context.view_layer.update()

        self.report({'INFO'}, f"Removed {removed} live constraints")
        return {'FINISHED'}


class VIEW3D_PT_dp2_live_panel(bpy.types.Panel):
    bl_label = "Difference Pose Live"
    bl_idname = "VIEW3D_PT_dp2_live_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if hasattr(scene, SCENE_PROP):
            layout.prop(scene, SCENE_PROP)
        else:
            layout.label(text="Reload addon (Preferences)", icon='ERROR')
        layout.operator("armature.dp2_live_enable", icon='CONSTRAINT_BONE')
        layout.operator("armature.dp2_live_disable", icon='X')


classes = (
    ARMATURE_OT_dp2_live_enable,
    ARMATURE_OT_dp2_live_disable,
    VIEW3D_PT_dp2_live_panel,
)


def _safe_register_class(cls):
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
        bpy.utils.register_class(cls)


def _register_scene_prop():
    if hasattr(bpy.types.Scene, SCENE_PROP):
        return
    bpy.types.Scene.dp2_head_only = bpy.props.BoolProperty(
        name="Head / Neck Only",
        description="Only link bones whose names contain head or neck",
        default=False,
    )


def register():
    _register_scene_prop()
    for cls in classes:
        _safe_register_class(cls)
    print("[Pose Skele Live] registered - panel: N > Akelka Tools > Difference Pose Live")


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, SCENE_PROP):
        try:
            del bpy.types.Scene.dp2_head_only
        except Exception:
            pass


if __name__ == "__main__":
    register()