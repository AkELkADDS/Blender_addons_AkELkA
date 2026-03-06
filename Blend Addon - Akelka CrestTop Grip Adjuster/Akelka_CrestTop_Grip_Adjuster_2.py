bl_info = {
    "name": "Akelka CrestTop Grip Adjuster",
    "author": "akelka",
    "version": (1, 0, 1),
    "blender": (4, 5, 3),
    "location": "View3D > N Panel > Akelka Tools",
    "description": "Rigging utilities used for CrestTop grip adjustments",
    "category": "Rigging",
}

import bpy


# ------------------------------
# Operator 1
# Apply Pose -> Rest + remove constraints
# ------------------------------

class AKELKA_OT_apply_pose_remove_constraints(bpy.types.Operator):
    bl_idname = "akelka.apply_pose_remove_constraints"
    bl_label = "Apply Pose -> Rest & Remove Constraints"
    bl_description = "Apply selected pose to rest on selected armatures, then remove constraints from selected bones."

    def execute(self, context):
        ctx = context
        sel_armatures = [o for o in ctx.selected_objects if o.type == 'ARMATURE']

        if not sel_armatures:
            self.report({'WARNING'}, "No armatures selected")
            return {'CANCELLED'}

        prev_active = ctx.view_layer.objects.active
        prev_mode = getattr(prev_active, "mode", None)

        for obj in sel_armatures:
            ctx.view_layer.objects.active = obj

            try:
                if obj.mode != 'POSE':
                    bpy.ops.object.mode_set(mode='POSE')
            except Exception:
                pass

            try:
                bpy.ops.pose.armature_apply(selected=True)
            except Exception as e:
                # ignore failures (e.g., no bones selected) but inform user
                self.report({'INFO'}, f"pose.armature_apply on '{obj.name}': {e}")

            for pbone in obj.pose.bones:
                if getattr(pbone.bone, "select", False):
                    for c in list(pbone.constraints):
                        pbone.constraints.remove(c)

        # restore previous active object/mode
        if prev_active:
            ctx.view_layer.objects.active = prev_active
            try:
                if prev_mode is not None and getattr(prev_active, "mode", None) != prev_mode:
                    bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass

        self.report({'INFO'}, "Done: applied pose->rest and removed constraints.")
        return {'FINISHED'}


# ------------------------------
# Operator 2
# Add Child Of Constraints
# ------------------------------

class AKELKA_OT_add_childof(bpy.types.Operator):
    bl_idname = "akelka.add_childof_constraints"
    bl_label = "Add Child Of (Parent Filter)"
    bl_description = "Add Child Of constraints to bones whose parent bone matches the given name. Active armature = target."

    def execute(self, context):
        TARGET_PARENT_NAME = context.scene.akelka_parent_name.strip()
        active_obj = context.active_object

        if not TARGET_PARENT_NAME:
            self.report({'ERROR'}, "Parent bone name is empty.")
            return {'CANCELLED'}

        if not active_obj or active_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an armature (the target).")
            return {'CANCELLED'}

        sel_armatures = [o for o in context.selected_objects if o.type == 'ARMATURE']
        if active_obj not in sel_armatures:
            sel_armatures.append(active_obj)

        for arm in sel_armatures:
            if arm == active_obj:
                continue

            context.view_layer.objects.active = arm
            try:
                bpy.ops.object.mode_set(mode='POSE')
            except Exception:
                pass

            pose_bones = arm.pose.bones

            for pbone in pose_bones:
                if not pbone.parent:
                    continue
                if pbone.parent.name != TARGET_PARENT_NAME:
                    continue

                # skip if identical constraint already exists
                already = False
                for c in pbone.constraints:
                    if c.type == 'CHILD_OF' and c.target == active_obj and c.subtarget == pbone.name:
                        already = True
                        break
                if already:
                    continue

                con = pbone.constraints.new('CHILD_OF')
                con.name = f"ChildOf_{pbone.name}"
                con.target = active_obj
                con.subtarget = pbone.name

                # try to set inverse using operator (safer) — fall back if fails
                try:
                    for pb in pose_bones:
                        pb.bone.select = False
                    pbone.bone.select = True
                    arm.data.bones.active = pbone.bone
                    context.view_layer.update()
                    bpy.ops.pose.constraint_childof_set_inverse(constraint=con.name)
                except Exception:
                    self.report({'WARNING'}, f"Couldn't set inverse for '{pbone.name}' on '{arm.name}'. Set inverse manually if needed.")

            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

        context.view_layer.objects.active = active_obj
        self.report({'INFO'}, f"Done: Child Of constraints added where parent == '{TARGET_PARENT_NAME}'.")
        return {'FINISHED'}


# ------------------------------
# Panel (renamed to addon proper, kept inside Akelka Tools tab)
# Default-closed so it's toggled off by default
# ------------------------------

class AKELKA_PT_cresttop_panel(bpy.types.Panel):
    bl_label = "Akelka CrestTop Grip Adjuster"   # panel proper name
    bl_idname = "AKELKA_PT_cresttop_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"                # keeps it in the existing Akelka Tools tab
    bl_options = {'DEFAULT_CLOSED'}             # <-- collapsed by default

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Apply Pose → Rest", icon='ARMATURE_DATA')

        warn = box.row()
        warn.alert = True
        warn.label(text="GO TO POSE MODE!!!", icon='ERROR')

        box.operator("akelka.apply_pose_remove_constraints", icon="ARMATURE_DATA")

        box = layout.box()
        box.label(text="Child Of Generator", icon='CONSTRAINT')

        box.prop(scene, "akelka_parent_name", text="Parent Bone")
        box.operator("akelka.add_childof_constraints", icon="CONSTRAINT")


# ------------------------------
# Registration
# ------------------------------

classes = (
    AKELKA_OT_apply_pose_remove_constraints,
    AKELKA_OT_add_childof,
    AKELKA_PT_cresttop_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.akelka_parent_name = bpy.props.StringProperty(
        name="Parent Bone",
        default="HornOffset_Grp"
    )


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.akelka_parent_name


if __name__ == "__main__":
    register()