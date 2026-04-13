bl_info = {
    "name": "Armature Difference Pose (Robust)",
    "blender": (3, 0, 0),
    "category": "Animation",
}

import bpy


class ARMATURE_OT_difference_pose(bpy.types.Operator):
    bl_idname = "armature.difference_pose"
    bl_label = "Create Difference Pose"
    bl_description = "Match active armature to selected target armature using same bone names"

    bake: bpy.props.BoolProperty(
        name="Bake Pose",
        default=True
    )

    head_only: bpy.props.BoolProperty(
        name="Head/Neck Only",
        description="Only affect head and neck bones",
        default=False
    )

    def execute(self, context):
        selected = [obj for obj in context.selected_objects if obj.type == 'ARMATURE']

        if len(selected) != 2:
            self.report({'ERROR'}, "Select exactly 2 armatures")
            return {'CANCELLED'}

        source = context.active_object
        target = [obj for obj in selected if obj != source][0]

        print("\n=== Difference Pose Debug ===")
        print(f"Source: {source.name}")
        print(f"Target: {target.name}")

        bpy.ops.object.mode_set(mode='POSE')

        matched = 0

        for bone in source.pose.bones:

            # Optional filter
            if self.head_only:
                if not any(k in bone.name.lower() for k in ["head", "neck"]):
                    continue

            if bone.name in target.pose.bones:
                matched += 1

                # COPY ROTATION
                con_rot = bone.constraints.new('COPY_ROTATION')
                con_rot.target = target
                con_rot.subtarget = bone.name
                con_rot.owner_space = 'WORLD'
                con_rot.target_space = 'WORLD'

                # COPY LOCATION
                con_loc = bone.constraints.new('COPY_LOCATION')
                con_loc.target = target
                con_loc.subtarget = bone.name
                con_loc.owner_space = 'WORLD'
                con_loc.target_space = 'WORLD'

        print(f"Matched bones: {matched}")

        # Force update
        context.view_layer.update()

        if self.bake:
            bpy.ops.nla.bake(
                frame_start=1,
                frame_end=1,
                only_selected=False,
                visual_keying=True,
                clear_constraints=True,
                use_current_action=False,
                bake_types={'POSE'}
            )

        self.report({'INFO'}, f"Done. Matched bones: {matched}")
        return {'FINISHED'}


class VIEW3D_PT_difference_pose_panel(bpy.types.Panel):
    bl_label = "Armature Difference Pose"
    bl_idname = "VIEW3D_PT_difference_pose_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        op = layout.operator("armature.difference_pose")
        layout.prop(context.scene, "dp_head_only")
        layout.prop(context.scene, "dp_bake")


def register():
    bpy.utils.register_class(ARMATURE_OT_difference_pose)
    bpy.utils.register_class(VIEW3D_PT_difference_pose_panel)

    bpy.types.Scene.dp_head_only = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.dp_bake = bpy.props.BoolProperty(default=True)


def unregister():
    bpy.utils.unregister_class(ARMATURE_OT_difference_pose)
    bpy.utils.unregister_class(VIEW3D_PT_difference_pose_panel)

    del bpy.types.Scene.dp_head_only
    del bpy.types.Scene.dp_bake


if __name__ == "__main__":
    register()