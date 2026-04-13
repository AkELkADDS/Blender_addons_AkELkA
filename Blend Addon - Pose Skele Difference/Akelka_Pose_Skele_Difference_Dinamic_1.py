bl_info = {
    "name": "Armature Difference Pose Live Link",
    "author": "AkELkA",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Tool",
    "description": "Live link one armature to another using matching bone names",
    "category": "Rigging",
}

import bpy

PREFIX_ROT = "DP_LIVE_ROT_"
PREFIX_LOC = "DP_LIVE_LOC_"


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

    # Make sure both are in pose position
    driver.data.pose_position = 'POSE'
    driven.data.pose_position = 'POSE'

    for pb in driven.pose.bones:
        if not matches_filter(pb.name, head_only):
            continue

        if pb.name not in driver.pose.bones:
            continue

        matched += 1

        # Copy Rotation
        con_rot = pb.constraints.new('COPY_ROTATION')
        con_rot.name = PREFIX_ROT + pb.name
        con_rot.target = driver
        con_rot.subtarget = pb.name
        con_rot.owner_space = 'WORLD'
        con_rot.target_space = 'WORLD'
        con_rot.mix_mode = 'REPLACE'
        con_rot.influence = 1.0

        # Copy Location
        con_loc = pb.constraints.new('COPY_LOCATION')
        con_loc.name = PREFIX_LOC + pb.name
        con_loc.target = driver
        con_loc.subtarget = pb.name
        con_loc.owner_space = 'WORLD'
        con_loc.target_space = 'WORLD'
        con_loc.influence = 1.0

    return matched


class ARMATURE_OT_dp_live_enable(bpy.types.Operator):
    bl_idname = "armature.dp_live_enable"
    bl_label = "Enable Live Link"
    bl_description = "Make the active armature follow the other selected armature in real time"

    def execute(self, context):
        driver, driven, err = get_driver_and_driven(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        head_only = context.scene.dp_head_only

        # Work in Object mode first
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Clear old link constraints from the driven armature
        removed = remove_live_constraints(driven)

        # Switch to pose mode for the driven armature
        bpy.ops.object.select_all(action='DESELECT')
        driven.select_set(True)
        context.view_layer.objects.active = driven
        bpy.ops.object.mode_set(mode='POSE')

        matched = add_live_constraints(driver, driven, head_only=head_only)

        context.view_layer.update()

        if matched == 0:
            self.report({'WARNING'}, "No matching bones found")
        else:
            self.report({'INFO'}, f"Live link enabled. Matched bones: {matched}, removed old constraints: {removed}")

        return {'FINISHED'}


class ARMATURE_OT_dp_live_disable(bpy.types.Operator):
    bl_idname = "armature.dp_live_disable"
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


class VIEW3D_PT_dp_live_panel(bpy.types.Panel):
    bl_label = "Difference Pose Live"
    bl_idname = "VIEW3D_PT_dp_live_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "dp_head_only")
        layout.operator("armature.dp_live_enable", icon='CONSTRAINT_BONE')
        layout.operator("armature.dp_live_disable", icon='X')


classes = (
    ARMATURE_OT_dp_live_enable,
    ARMATURE_OT_dp_live_disable,
    VIEW3D_PT_dp_live_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.dp_head_only = bpy.props.BoolProperty(
        name="Head / Neck Only",
        description="Only link bones whose names contain head or neck",
        default=True
    )


def unregister():
    del bpy.types.Scene.dp_head_only

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()