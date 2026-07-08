bl_info = {
    "name": "Akelka Bake Animation",
    "author": "Akelka",
    "version": (1, 7, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Item / Akelka Tools > Bake Animation",
    "description": "Bake pose action, clear NLA strips, delete bones and their animation",
    "category": "Animation",
}

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup, UIList


def _bone_path(name):
    return f'pose.bones["{name}"]'


def _path_targets_bone(data_path, bone_name):
    return _bone_path(bone_name) in data_path


def get_armature(context):
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    for obj in context.selected_objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def get_selected_bone_names(armature):
    return [pb.name for pb in armature.pose.bones if pb.bone and pb.bone.select]


EXCLUDE_BONE_DEFAULT = "Dummy_Root"


def get_bones_to_bake(armature, props):
    if props.auto_bone_mode == "ALL":
        return [pb.name for pb in armature.pose.bones]
    if props.auto_bone_mode == "NO_ROOT":
        return [
            pb.name for pb in armature.pose.bones
            if pb.name != EXCLUDE_BONE_DEFAULT
        ]
    return get_selected_bone_names(armature)


def count_action_keys(action):
    if not action:
        return 0, 0
    return len(action.fcurves), sum(len(fc.keyframe_points) for fc in action.fcurves)


def get_frame_range(props, scene):
    if props.use_scene_range:
        return scene.frame_start, scene.frame_end
    return props.frame_start, props.frame_end


def build_channel_types(props):
    channels = set()
    if props.channel_location:
        channels.add("LOCATION")
    if props.channel_rotation:
        channels.add("ROTATION")
    if props.channel_scale:
        channels.add("SCALE")
    if props.channel_bbone:
        channels.add("BBONE")
    if props.channel_custom_props:
        channels.add("PROPS")
    return channels


def build_bake_types(props):
    types = set()
    if props.bake_pose:
        types.add("POSE")
    if props.bake_object:
        types.add("OBJECT")
    return types or {"POSE"}


def _other_objects_selected(context, armature):
    return any(
        o.select_get() and o != armature
        for o in context.view_layer.objects
    )


def prepare_armature_context(context, armature, bones_to_bake, only_selected):
    """Match what Blender needs: only armature selected, pose mode, bones selected."""
    needs_object_setup = (
        context.active_object != armature
        or _other_objects_selected(context, armature)
        or context.mode not in {"POSE", "OBJECT"}
    )

    if needs_object_setup:
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
    elif context.mode != "POSE":
        bpy.ops.object.mode_set(mode="POSE")

    bpy.ops.pose.select_all(action="DESELECT")
    if only_selected:
        for name in bones_to_bake:
            pb = armature.pose.bones.get(name)
            if pb:
                pb.bone.select = True
        if bones_to_bake:
            pb = armature.pose.bones.get(bones_to_bake[0])
            if pb:
                armature.data.bones.active = pb.bone
    else:
        bpy.ops.pose.select_all(action="SELECT")

    context.view_layer.update()


def remove_all_drivers(armature):
    ad = armature.animation_data
    if not ad:
        return
    for drv in list(ad.drivers):
        ad.drivers.remove(drv)


def invoke_nla_bake(context, armature, props, frame_start, frame_end):
    """Call bpy.ops.nla.bake — same operator as Pose > Animation > Bake Action."""
    channels = build_channel_types(props)
    if not channels:
        raise RuntimeError("Enable at least one channel (Loc/Rot/Scale/B-Bone/Props)")

    kwargs = dict(
        frame_start=frame_start,
        frame_end=frame_end,
        step=max(1, props.frame_step),
        only_selected=props.only_selected,
        visual_keying=props.visual_keying,
        clear_constraints=props.clear_constraints,
        clear_parents=props.clear_parents,
        use_current_action=props.overwrite_action,
        clean_curves=props.clean_curves,
        bake_types=build_bake_types(props),
        channel_types=channels,
    )

    legacy_kwargs = dict(
        frame_start=frame_start,
        frame_end=frame_end,
        step=max(1, props.frame_step),
        only_selected=props.only_selected,
        visual_keying=props.visual_keying,
        clear_constraints=props.clear_constraints,
        clear_parents=props.clear_parents,
        use_current_action=props.overwrite_action,
        clean_curves=props.clean_curves,
        bake_types=build_bake_types(props),
        channel_location=props.channel_location,
        channel_rotation=props.channel_rotation,
        channel_scale=props.channel_scale,
        channel_bbone=props.channel_bbone,
        channel_custom_props=props.channel_custom_props,
    )

    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type != "WINDOW":
                    continue
                override = {
                    "window": window,
                    "screen": window.screen,
                    "area": area,
                    "region": region,
                    "scene": context.scene,
                    "view_layer": context.view_layer,
                    "active_object": armature,
                    "object": armature,
                    "selected_objects": [armature],
                    "selected_editable_objects": [armature],
                }
                with context.temp_override(**override):
                    try:
                        return bpy.ops.nla.bake(**kwargs)
                    except TypeError:
                        return bpy.ops.nla.bake(**legacy_kwargs)

    try:
        return bpy.ops.nla.bake(**kwargs)
    except TypeError:
        return bpy.ops.nla.bake(**legacy_kwargs)


def bake_pose_action(armature, context, props, bones_to_bake):
    frame_start, frame_end = get_frame_range(props, context.scene)
    if frame_start > frame_end:
        raise RuntimeError("Start frame must be <= end frame")
    if props.only_selected and not bones_to_bake:
        raise RuntimeError("Select pose bones to bake")

    prepare_armature_context(context, armature, bones_to_bake, props.only_selected)

    # Standard Blender bake does NOT strip drivers or pre-create actions.
    if props.clear_drivers:
        remove_all_drivers(armature)

    result = invoke_nla_bake(context, armature, props, frame_start, frame_end)
    if result != {"FINISHED"}:
        raise RuntimeError("Bake failed — check console")

    action = armature.animation_data.action if armature.animation_data else None
    curves, keys = count_action_keys(action)
    baked_bones = len(bones_to_bake) if props.only_selected else len(armature.pose.bones)
    return frame_start, frame_end, baked_bones, curves, keys


def clear_all_nla(armature):
    ad = armature.animation_data
    if not ad:
        return
    for track in list(ad.nla_tracks):
        ad.nla_tracks.remove(track)


def collect_bones_to_delete(armature, props):
    seen = set()
    result = []
    for item in props.bones_to_delete:
        if item.name and item.name not in seen and item.name in armature.pose.bones:
            seen.add(item.name)
            result.append(item.name)
    return result


def remove_bone_animation(armature, bone_names):
    ad = armature.animation_data
    if not ad or not bone_names:
        return
    action = ad.action
    if action:
        for fc in [fc for fc in action.fcurves if any(_path_targets_bone(fc.data_path, n) for n in bone_names)]:
            action.fcurves.remove(fc)
    for drv in list(ad.drivers):
        if any(_path_targets_bone(drv.data_path, n) for n in bone_names):
            ad.drivers.remove(drv)


def delete_armature_bones(armature, bone_names):
    if not bone_names:
        return []
    remove_bone_animation(armature, bone_names)
    bpy.ops.object.mode_set(mode="EDIT")
    removed = []
    for name in bone_names:
        eb = armature.data.edit_bones.get(name)
        if eb:
            armature.data.edit_bones.remove(eb)
            removed.append(name)
    bpy.ops.object.mode_set(mode="POSE")
    return removed


def purge_deleted_from_list(props, deleted_names):
    deleted = set(deleted_names)
    for i in range(len(props.bones_to_delete) - 1, -1, -1):
        if props.bones_to_delete[i].name in deleted:
            props.bones_to_delete.remove(i)


class BAC_BoneDeleteItem(PropertyGroup):
    name: StringProperty(name="Bone")


class BAC_Properties(PropertyGroup):
    use_scene_range: BoolProperty(name="Scene", default=True)
    frame_start: IntProperty(name="Start", default=0, min=0)
    frame_end: IntProperty(name="End", default=430, min=0)
    frame_step: IntProperty(name="Step", default=1, min=1)

    only_selected: BoolProperty(name="Selected", default=True)
    auto_bone_mode: EnumProperty(
        name="Auto Bones",
        description="Auto-select all bones before bake (click active button again for manual selection)",
        items=(
            ("OFF", "Manual", "Use current pose bone selection"),
            ("ALL", "Bake All Bones", "Select all bones including root"),
            ("NO_ROOT", "Without Root", "Select all bones except Dummy_Root"),
        ),
        default="NO_ROOT",
    )
    visual_keying: BoolProperty(name="Visual", default=True)
    clear_constraints: BoolProperty(name="Constraints", default=True)
    clear_parents: BoolProperty(name="Parents", default=True)
    overwrite_action: BoolProperty(name="Overwrite", default=True)
    clean_curves: BoolProperty(name="Clean", default=True)
    clear_drivers: BoolProperty(
        name="Clr Drivers",
        description="Remove ALL drivers before bake (off = same as Blender Bake Action dialog)",
        default=False,
    )
    clear_nla: BoolProperty(
        name="NLA",
        description="Clear NLA strips after bake (not during)",
        default=True,
    )

    bake_pose: BoolProperty(name="Pose", default=True)
    bake_object: BoolProperty(name="Object", default=False)

    channel_location: BoolProperty(name="Loc", default=True)
    channel_rotation: BoolProperty(name="Rot", default=True)
    channel_scale: BoolProperty(name="Scale", default=False)
    channel_bbone: BoolProperty(name="B-Bone", default=True)
    channel_custom_props: BoolProperty(name="Props", default=True)

    bones_to_delete: CollectionProperty(type=BAC_BoneDeleteItem)
    bone_list_index: IntProperty(default=0)


class BAC_UL_bones_to_delete(UIList):
    bl_idname = "BAC_UL_bones_to_delete"

    def draw_item(self, _context, layout, _data, item, _icon, _active, _prop, _index):
        layout.label(text=item.name, icon="DOT")


class BAC_OT_set_auto_bone_mode(Operator):
    bl_idname = "bac.set_auto_bone_mode"
    bl_label = "Set Auto Bone Mode"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        items=(
            ("OFF", "Manual", ""),
            ("ALL", "Bake All Bones", ""),
            ("NO_ROOT", "Without Root", ""),
        ),
    )

    def execute(self, context):
        props = context.scene.bac_props
        props.auto_bone_mode = "OFF" if props.auto_bone_mode == self.mode else self.mode
        return {"FINISHED"}


class BAC_OT_add_bones_from_selection(Operator):
    bl_idname = "bac.add_bones_from_selection"
    bl_label = "Add Selected"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return get_armature(context) and context.mode == "POSE"

    def execute(self, context):
        arm = get_armature(context)
        props = context.scene.bac_props
        existing = {item.name for item in props.bones_to_delete}
        added = 0
        for pb in arm.pose.bones:
            if pb.bone and pb.bone.select and pb.name not in existing:
                props.bones_to_delete.add().name = pb.name
                existing.add(pb.name)
                added += 1
        self.report({"INFO"}, f"+{added}" if added else "—")
        return {"FINISHED"}


class BAC_OT_clear_delete_list(Operator):
    bl_idname = "bac.clear_delete_list"
    bl_label = "Clear"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.bac_props.bones_to_delete.clear()
        return {"FINISHED"}


class BAC_OT_run(Operator):
    bl_idname = "bac.run"
    bl_label = "Bake & Cleanup"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return get_armature(context) is not None

    def execute(self, context):
        arm = get_armature(context)
        props = context.scene.bac_props
        bones_to_bake = get_bones_to_bake(arm, props)
        bones_to_delete = collect_bones_to_delete(arm, props)

        try:
            f0, f1, n_bake, curves, keys = bake_pose_action(arm, context, props, bones_to_bake)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if props.clear_nla:
            clear_all_nla(arm)

        deleted = delete_armature_bones(arm, bones_to_delete)
        if deleted:
            purge_deleted_from_list(props, deleted)

        msg = f"Baked {n_bake} | {f0}-{f1} | {curves} curves {keys} keys"
        if deleted:
            msg += f" | -{len(deleted)} bones"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _toggle_row(layout, props, a, b):
    row = layout.row(align=True)
    row.prop(props, a, toggle=True)
    row.prop(props, b, toggle=True)


class BAC_PT_item(Panel):
    bl_label = "Bake Animation"
    bl_idname = "BAC_PT_item"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Item"
    bl_order = 100

    @classmethod
    def poll(cls, context):
        return get_armature(context) is not None

    def draw(self, context):
        layout = self.layout
        props = context.scene.bac_props
        arm = get_armature(context)

        layout.operator("bac.run", icon="RENDER_ANIMATION")

        bones = collect_bones_to_delete(arm, props)
        box = layout.box()
        col = box.column(align=True)
        if bones:
            for name in bones[:5]:
                col.label(text=name, icon="DOT")
            if len(bones) > 5:
                col.label(text=f"+{len(bones) - 5}")
        else:
            col.label(text="—", icon="BONE_DATA")


class BAC_PT_main(Panel):
    bl_label = "Bake Animation"
    bl_idname = "BAC_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Akelka Tools"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        props = context.scene.bac_props

        col = layout.column(align=True)

        top_box = col.box()
        top_col = top_box.column(align=True)
        top_col.operator("bac.run", icon="RENDER_ANIMATION")

        row = top_col.row(align=True)
        op = row.operator(
            "bac.set_auto_bone_mode",
            text="Bake All Bones",
            depress=props.auto_bone_mode == "ALL",
            icon="BONE_DATA",
        )
        op.mode = "ALL"
        op = row.operator(
            "bac.set_auto_bone_mode",
            text="Without Root",
            depress=props.auto_bone_mode == "NO_ROOT",
            icon="BONE_DATA",
        )
        op.mode = "NO_ROOT"

        row = top_col.row(align=True)
        row.prop(props, "use_scene_range", toggle=True, text="Scene")
        sub = row.row(align=True)
        sub.enabled = not props.use_scene_range
        sub.prop(props, "frame_start", text="")
        sub.prop(props, "frame_end", text="")
        sub.prop(props, "frame_step", text="Step")

        box = col.box()
        toggle_col = box.column(align=True)
        _toggle_row(toggle_col, props, "only_selected", "visual_keying")
        _toggle_row(toggle_col, props, "clear_constraints", "clear_parents")
        _toggle_row(toggle_col, props, "overwrite_action", "clean_curves")
        _toggle_row(toggle_col, props, "clear_drivers", "clear_nla")
        _toggle_row(toggle_col, props, "bake_pose", "bake_object")
        _toggle_row(toggle_col, props, "channel_location", "channel_rotation")
        _toggle_row(toggle_col, props, "channel_scale", "channel_bbone")
        row = toggle_col.row(align=True)
        row.prop(props, "channel_custom_props", toggle=True)

        list_box = col.box()
        list_col = list_box.column(align=True)
        row = list_col.row(align=True)
        row.operator("bac.add_bones_from_selection", icon="BONE_DATA")
        row.operator("bac.clear_delete_list", text="", icon="X")
        list_col.template_list(
            "BAC_UL_bones_to_delete", "",
            props, "bones_to_delete",
            props, "bone_list_index",
            rows=2,
        )


classes = (
    BAC_BoneDeleteItem,
    BAC_Properties,
    BAC_UL_bones_to_delete,
    BAC_OT_set_auto_bone_mode,
    BAC_OT_add_bones_from_selection,
    BAC_OT_clear_delete_list,
    BAC_OT_run,
    BAC_PT_item,
    BAC_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bac_props = bpy.props.PointerProperty(type=BAC_Properties)


def unregister():
    del bpy.types.Scene.bac_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
