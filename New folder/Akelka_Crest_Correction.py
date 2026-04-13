bl_info = {
    "name": "Shape Key Proximity Deformer",
    "author": "OpenAI",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Deform",
    "description": "Make one mesh follow another mesh's animated shape keys using Geometry Nodes proximity.",
    "category": "Object",
}

import bpy
from bpy.props import PointerProperty, FloatProperty
from bpy.types import Operator, Panel, PropertyGroup


MOD_NAME_PREFIX = "SK Proximity Deformer"
GROUP_NAME_PREFIX = "SKPD"


# ------------------------------------------------------------
# Geometry Nodes builder
# ------------------------------------------------------------

def _set_socket_defaults(socket, *, default=None, min_value=None, max_value=None):
    if default is not None and hasattr(socket, "default_value"):
        try:
            socket.default_value = default
        except Exception:
            pass
    if min_value is not None and hasattr(socket, "min_value"):
        try:
            socket.min_value = min_value
        except Exception:
            pass
    if max_value is not None and hasattr(socket, "max_value"):
        try:
            socket.max_value = max_value
        except Exception:
            pass


def _set_modifier_input(mod, node_group, socket_name: str, value) -> bool:
    """
    Try to set a Geometry Nodes modifier group input.

    Blender sometimes exposes modifier properties by the socket's `identifier`
    instead of its display `name`, so we try both.
    """
    items = getattr(getattr(node_group, "interface", None), "items_tree", []) or []
    for item in items:
        if getattr(item, "item_type", None) != "SOCKET":
            continue
        if getattr(item, "name", None) != socket_name:
            continue

        identifier = getattr(item, "identifier", None)
        if identifier:
            try:
                mod[identifier] = value
                return True
            except Exception:
                pass

    # Fallbacks: exact name and common variants.
    candidates = {
        socket_name,
        socket_name.replace(" ", "_"),
        socket_name.replace(" ", ""),
    }
    for key in candidates:
        try:
            mod[key] = value
            return True
        except Exception:
            pass

    return False


def build_node_group(group_name: str, source_obj: bpy.types.Object):
    """Create or rebuild the node group for one target/source pair."""
    ng = bpy.data.node_groups.get(group_name)
    if ng is None:
        ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')
    else:
        ng.nodes.clear()
        # Removing interface sockets is awkward, so we reuse the same socket layout.
        # If the group already existed from a previous bind, we simply overwrite the nodes.

    # Recreate interface safely if this is a fresh node tree.
    if len(ng.interface.items_tree) == 0:
        s_geo_in = ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        s_strength = ng.interface.new_socket(name="Strength", in_out='INPUT', socket_type='NodeSocketFloat')
        s_distance = ng.interface.new_socket(name="Max Distance", in_out='INPUT', socket_type='NodeSocketFloat')
        s_geo_out = ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        _set_socket_defaults(s_strength, default=1.0, min_value=0.0, max_value=1.0)
        _set_socket_defaults(s_distance, default=10.0, min_value=0.0, max_value=100000.0)
    else:
        # Existing group: update any float defaults we can find.
        for item in ng.interface.items_tree:
            if getattr(item, "item_type", None) == 'SOCKET':
                if item.name == "Strength":
                    _set_socket_defaults(item, default=1.0, min_value=0.0, max_value=1.0)
                elif item.name == "Max Distance":
                    _set_socket_defaults(item, default=10.0, min_value=0.0, max_value=100000.0)

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-900, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (520, 0)

    obj_info = nodes.new("GeometryNodeObjectInfo")
    obj_info.location = (-680, 220)
    # Blender node sockets/properties differ slightly between versions.
    # Some versions expose `as_instance`, others don't.
    try:
        if hasattr(obj_info, "as_instance"):
            obj_info.as_instance = False
    except Exception:
        pass
    try:
        obj_info.inputs[0].default_value = source_obj
    except Exception:
        pass

    position = nodes.new("GeometryNodeInputPosition")
    position.location = (-680, -120)

    proximity = nodes.new("GeometryNodeProximity")
    proximity.location = (-380, 100)

    # Distance -> blend factor:
    # We want factor=0 when distance=0 (no snap on bind if meshes already overlap),
    # and factor grows as distance increases (and then caps at Strength due to clamping).
    safe_max = nodes.new("ShaderNodeMath")
    safe_max.location = (-120, -160)
    safe_max.operation = 'MAXIMUM'
    safe_max.use_clamp = False
    safe_max.inputs[1].default_value = 0.000001  # Prevent divide-by-zero

    ratio = nodes.new("ShaderNodeMath")
    ratio.location = (-120, -110)
    ratio.operation = 'DIVIDE'
    ratio.use_clamp = False

    mult = nodes.new("ShaderNodeMath")
    mult.location = (-120, 40)
    mult.operation = 'MULTIPLY'
    mult.use_clamp = True

    mix = nodes.new("ShaderNodeMix")
    mix.location = (160, 0)
    mix.data_type = 'VECTOR'

    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (380, 0)

    # Geometry through to output
    links.new(group_in.outputs["Geometry"], set_pos.inputs["Geometry"])
    links.new(set_pos.outputs["Geometry"], group_out.inputs["Geometry"])

    # Source object evaluated geometry into proximity
    links.new(obj_info.outputs["Geometry"], proximity.inputs["Geometry"])

    # Use target vertex position for the proximity query
    # Socket names differ slightly across Blender versions, so prefer name match
    # then fall back to the common "2nd input" layout.
    target_pos_out = None
    for s in position.outputs:
        if s.name == "Position":
            target_pos_out = s
            break
    if target_pos_out is None:
        target_pos_out = position.outputs[0]

    target_pos_in = None
    for s in proximity.inputs:
        if s.name in {"Target Position", "TargetPosition", "Target", "Position"}:
            target_pos_in = s
            break
    if target_pos_in is None:
        # Typically: inputs[0] = Geometry, inputs[1] = Target/Position
        target_pos_in = proximity.inputs[1] if len(proximity.inputs) > 1 else proximity.inputs[0]

    links.new(target_pos_out, target_pos_in)

    # Distance remap -> multiply by Strength -> Mix factor
    links.new(group_in.outputs["Max Distance"], safe_max.inputs[0])
    links.new(proximity.outputs["Distance"], ratio.inputs[0])
    links.new(safe_max.outputs[0], ratio.inputs[1])
    # ratio = distance / max_distance
    # mult = Strength * ratio (clamped by mult.use_clamp)
    links.new(ratio.outputs[0], mult.inputs[0])
    links.new(group_in.outputs["Strength"], mult.inputs[1])
    links.new(mult.outputs[0], mix.inputs["Factor"])

    # Blend original position toward closest point on source surface
    links.new(position.outputs["Position"], mix.inputs["A"])
    links.new(proximity.outputs["Position"], mix.inputs["B"])

    links.new(mix.outputs["Result"], set_pos.inputs["Position"])

    return ng


# ------------------------------------------------------------
# Addon properties
# ------------------------------------------------------------

class SKPD_Props(PropertyGroup):
    source_object: PointerProperty(
        name="Source Object",
        type=bpy.types.Object,
        description="Animated source mesh. Its evaluated mesh includes shapekeys.",
    )
    target_object: PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Mesh that will follow the source.",
    )
    strength: FloatProperty(
        name="Strength",
        description="How much the target moves toward the source surface.",
        default=1.0,
        min=0.0,
        max=1.0,
    )
    max_distance: FloatProperty(
        name="Max Distance",
        description="Distance where the effect fades to zero.",
        default=10.0,
        min=0.0,
    )


# ------------------------------------------------------------
# Operators
# ------------------------------------------------------------

class OBJECT_OT_skpd_bind(Operator):
    bl_idname = "object.skpd_bind"
    bl_label = "Bind"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.skpd_props
        source = props.source_object
        target = props.target_object

        if source is None or target is None:
            self.report({'ERROR'}, "Pick both a Source Object and a Target Object.")
            return {'CANCELLED'}

        if source.type != 'MESH' or target.type != 'MESH':
            self.report({'ERROR'}, "Both objects must be meshes.")
            return {'CANCELLED'}

        group_name = f"{GROUP_NAME_PREFIX}_{target.name}"
        mod_name = f"{MOD_NAME_PREFIX}"

        node_group = build_node_group(group_name, source)

        mod = target.modifiers.get(mod_name)
        if mod is None or mod.type != 'NODES':
            mod = target.modifiers.new(mod_name, 'NODES')

        mod.node_group = node_group
        # Push UI properties into the Geometry Nodes modifier group inputs.
        ok_strength = _set_modifier_input(mod, node_group, "Strength", props.strength)
        ok_dist = _set_modifier_input(mod, node_group, "Max Distance", props.max_distance)
        if not (ok_strength and ok_dist):
            self.report({'WARNING'}, "Some node inputs may not be wired to UI values.")

        self.report({'INFO'}, f"Bound {target.name} to {source.name}")
        return {'FINISHED'}


class OBJECT_OT_skpd_rebuild(Operator):
    bl_idname = "object.skpd_rebuild"
    bl_label = "Rebuild"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.skpd_props
        source = props.source_object
        target = props.target_object

        if source is None or target is None:
            self.report({'ERROR'}, "Pick both a Source Object and a Target Object.")
            return {'CANCELLED'}

        if target.type != 'MESH':
            self.report({'ERROR'}, "Target must be a mesh.")
            return {'CANCELLED'}

        group_name = f"{GROUP_NAME_PREFIX}_{target.name}"
        mod_name = f"{MOD_NAME_PREFIX}"

        node_group = build_node_group(group_name, source)

        mod = target.modifiers.get(mod_name)
        if mod is None or mod.type != 'NODES':
            mod = target.modifiers.new(mod_name, 'NODES')
        mod.node_group = node_group
        ok_strength = _set_modifier_input(mod, node_group, "Strength", props.strength)
        ok_dist = _set_modifier_input(mod, node_group, "Max Distance", props.max_distance)
        if not (ok_strength and ok_dist):
            self.report({'WARNING'}, "Some node inputs may not be wired to UI values.")

        self.report({'INFO'}, f"Rebuilt {mod_name} on {target.name}")
        return {'FINISHED'}


class OBJECT_OT_skpd_clear(Operator):
    bl_idname = "object.skpd_clear"
    bl_label = "Clear"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.skpd_props
        target = props.target_object

        if target is None:
            self.report({'ERROR'}, "Pick a Target Object first.")
            return {'CANCELLED'}

        mod_name = f"{MOD_NAME_PREFIX}"
        mod = target.modifiers.get(mod_name)
        if mod is not None:
            target.modifiers.remove(mod)

        self.report({'INFO'}, f"Removed {mod_name} from {target.name}")
        return {'FINISHED'}


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

class VIEW3D_PT_skpd_panel(Panel):
    bl_label = "Shape Key Proximity Deformer"
    bl_idname = "VIEW3D_PT_skpd_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Akelka Tools'

    def draw(self, context):
        layout = self.layout
        props = context.scene.skpd_props

        layout.prop(props, "source_object")
        layout.prop(props, "target_object")
        layout.prop(props, "strength")
        layout.prop(props, "max_distance")

        row = layout.row(align=True)
        row.operator("object.skpd_bind", icon='GEOMETRY_NODES')
        row.operator("object.skpd_rebuild", icon='FILE_REFRESH')
        row.operator("object.skpd_clear", icon='TRASH')

        box = layout.box()
        box.label(text="How it works:")
        box.label(text="- The source object's evaluated mesh is sampled")
        box.label(text="- The target moves toward the nearest point")
        box.label(text="- Shape keys on the source are included live")
        box.label(text="- Bind creates the modifier")
        box.label(text="- Panel appears under Akelka Tools in the N-panel")


# ------------------------------------------------------------
# Register
# ------------------------------------------------------------

classes = (
    SKPD_Props,
    OBJECT_OT_skpd_bind,
    OBJECT_OT_skpd_rebuild,
    OBJECT_OT_skpd_clear,
    VIEW3D_PT_skpd_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.skpd_props = PointerProperty(type=SKPD_Props)


def unregister():
    del bpy.types.Scene.skpd_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
