bl_info = {
    "name": "UV Guided Vertex Snap → Shape Key",
    "author": "AkELkA",
    "version": (1, 1),
    "blender": (2, 93, 0),
    "location": "View3D > N-panel > UV Snap",
    "description": "Create a shape key on the active object with vertex positions snapped to another object's vertices using matching UVs",
    "category": "Mesh",
}

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree


class UVSNAP_OT_create_shapekey(bpy.types.Operator):
    bl_idname = "uvsnap.create_shapekey"
    bl_label = "Create Shape Key from UV Snap"
    bl_description = "Create a shape key where source vertices are snapped to corresponding target vertex positions using UV match"
    bl_options = {'REGISTER', 'UNDO'}

    uv_threshold: bpy.props.FloatProperty(
        name="UV distance threshold",
        description="Max distance in UV space to consider a match (set small if UVs are identical)",
        default=1e-4,
        min=0.0,
        precision=6
    )

    shape_key_name: bpy.props.StringProperty(
        name="Shape Key Name",
        description="Name for the created shape key",
        default="UV_Snap_Target"
    )

    overwrite_existing: bpy.props.BoolProperty(
        name="Overwrite existing",
        description="If a shape key with the same name exists, overwrite its vertex data instead of creating a new key",
        default=False
    )

    def execute(self, context):
        sel = [o for o in context.selected_objects if o.type == 'MESH']
        if len(sel) != 2:
            self.report({'ERROR'}, "Select exactly two mesh objects (active = source, other = target).")
            return {'CANCELLED'}

        source = context.active_object
        target = next(o for o in sel if o is not source)

        src_mesh = source.data
        tgt_mesh = target.data

        if not src_mesh.uv_layers.active:
            self.report({'ERROR'}, f"Source object '{source.name}' has no active UV map.")
            return {'CANCELLED'}
        if not tgt_mesh.uv_layers.active:
            self.report({'ERROR'}, f"Target object '{target.name}' has no active UV map.")
            return {'CANCELLED'}

        # Build target loop UV -> target vertex index table
        tgt_uv_layer = tgt_mesh.uv_layers.active.data
        tgt_loops = tgt_mesh.loops
        if len(tgt_loops) == 0:
            self.report({'ERROR'}, "Target mesh has no loops.")
            return {'CANCELLED'}

        tgt_uvs = []
        for li, loop in enumerate(tgt_loops):
            uv = tgt_uv_layer[li].uv
            tgt_uvs.append(((uv.x, uv.y), loop.vertex_index))

        # KDTree on target UVs
        kd = KDTree(len(tgt_uvs))
        for i, (uv, v_idx) in enumerate(tgt_uvs):
            kd.insert((uv[0], uv[1], 0.0), i)
        kd.balance()

        # Source loops and UVs
        src_uv_layer = src_mesh.uv_layers.active.data
        src_loops = src_mesh.loops

        # Prepare target vertex world positions
        tgt_mat = target.matrix_world.copy()
        tgt_vert_world = [tgt_mat @ v.co for v in tgt_mesh.vertices]

        # Map source vertex -> list of matched target world positions
        vertex_matches = {v.index: [] for v in src_mesh.vertices}

        for li, loop in enumerate(src_loops):
            uv = src_uv_layer[li].uv
            co = (uv.x, uv.y, 0.0)
            found = kd.find(co)
            if found is None:
                continue
            _, found_index, dist = found
            if dist > self.uv_threshold:
                continue
            tgt_vert_idx = tgt_uvs[found_index][1]
            tgt_world_pos = tgt_vert_world[tgt_vert_idx]
            src_vert_idx = loop.vertex_index
            vertex_matches[src_vert_idx].append(tgt_world_pos)

        # Ensure Object mode for shape key ops
        prev_mode = None
        try:
            prev_mode = source.mode
        except Exception:
            prev_mode = 'OBJECT'
        if prev_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Ensure a Basis exists (Blender creates Basis automatically when first key added,
        # but we'll create it explicitly if no shape_keys block exists)
        if src_mesh.shape_keys is None:
            # shape_key_add when no keys exist will create a Basis; create one.
            source.shape_key_add(name="Basis", from_mix=False)

        key = None
        ks = src_mesh.shape_keys.key_blocks

        if self.shape_key_name in ks:
            if self.overwrite_existing:
                key = ks[self.shape_key_name]
            else:
                # create a unique name by appending a number
                base = self.shape_key_name
                i = 1
                new_name = f"{base}.{i:03d}"
                while new_name in ks:
                    i += 1
                    new_name = f"{base}.{i:03d}"
                key = source.shape_key_add(name=new_name, from_mix=False)
                # set the operator-reported name accordingly
                self.shape_key_name = new_name
        else:
            key = source.shape_key_add(name=self.shape_key_name, from_mix=False)

        # Now write matched positions into the shape key data
        inv_src_mat = source.matrix_world.inverted()
        assigned = 0
        for vidx, world_positions in vertex_matches.items():
            if not world_positions:
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for p in world_positions:
                avg += p
            avg /= len(world_positions)
            local_pos = inv_src_mat @ avg
            # Write into shape key data (object local space)
            key.data[vidx].co = local_pos
            assigned += 1

        # Update mesh and restore mode
        src_mesh.update()
        if prev_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except Exception:
                pass

        self.report({'INFO'}, f"Assigned {assigned} vertices to shape key '{self.shape_key_name}'.")
        return {'FINISHED'}


class UVSNAP_PT_panel(bpy.types.Panel):
    bl_label = "UV Snap → Shape Key"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka Tools"                # keeps it in the existing Akelka Tools tab
    bl_options = {'DEFAULT_CLOSED'}             # <-- collapsed by default

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Select two meshes: active = source")
        col.label(text="Other selected = target")
        op = col.operator(UVSNAP_OT_create_shapekey.bl_idname, text="Create Shape Key from UV Snap")
        # expose defaults; properties are editable in operator redo panel or F6
        col.prop(op, "uv_threshold")
        col.prop(op, "shape_key_name")
        col.prop(op, "overwrite_existing")
        col.separator()
        col.label(text="Tip: Both objects need an active UV map")
        col.label(text="and very similar/identical UV layouts.")


classes = (UVSNAP_OT_create_shapekey, UVSNAP_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
