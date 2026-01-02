bl_info = {
    "name": "AkELkA UV Tools",
    "author": "AkELkA",
    "maintainer": "AkELkA",
    "version": (1, 7),
    "blender": (3, 6, 0),
    "location": "UV Editor > N Panel > AkELkA UV Tools",
    "description": "UV utilities for snapping UVs between similar meshes",
    "support": "COMMUNITY",
    "category": "UV",
}

import bpy
import bmesh


# -------------------------------------------------
# Operator
# -------------------------------------------------

class UV_OT_snap_to_closest_mesh(bpy.types.Operator):
    bl_idname = "akelka_uv.snap_to_closest_mesh"
    bl_label = "Snap UVs to Closest"
    bl_description = (
        "How it works:\n"
        "- Select TWO mesh objects\n"
        "- Active object = target (will be changed)\n"
        "- Other selected object = source\n"
        "- Snaps each UV to closest UV by distance\n"
        "- Meshes should have similar topology"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if len(objs) != 2:
            self.report({'ERROR'}, "Select exactly two mesh objects")
            return {'CANCELLED'}

        target = context.active_object
        source = [o for o in objs if o != target][0]

        if context.mode != 'EDIT_MESH':
            self.report({'ERROR'}, "Must be in Edit Mode")
            return {'CANCELLED'}

        # Source UVs
        src_bm = bmesh.new()
        src_bm.from_mesh(source.data)
        src_uv = src_bm.loops.layers.uv.active

        if not src_uv:
            src_bm.free()
            self.report({'ERROR'}, "Source mesh has no UVs")
            return {'CANCELLED'}

        source_uvs = [
            l[src_uv].uv.copy()
            for f in src_bm.faces
            for l in f.loops
        ]
        src_bm.free()

        # Target UVs
        tgt_bm = bmesh.from_edit_mesh(target.data)
        tgt_uv = tgt_bm.loops.layers.uv.active

        if not tgt_uv:
            self.report({'ERROR'}, "Target mesh has no UVs")
            return {'CANCELLED'}

        for f in tgt_bm.faces:
            for l in f.loops:
                uv = l[tgt_uv].uv
                uv[:] = min(
                    source_uvs,
                    key=lambda suv: (uv - suv).length_squared
                )

        bmesh.update_edit_mesh(target.data)
        return {'FINISHED'}


# -------------------------------------------------
# Addon Preferences
# -------------------------------------------------

class AkELkASnapUVPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        layout.label(text="AkELkA - Snap UV")
        layout.separator()

        box = layout.box()
        box.label(text="Support the development:")
        box.operator(
            "wm.url_open",
            text="Support on Patreon",
            icon='HEART'
        ).url = "https://www.patreon.com/c/AkELkA"


# -------------------------------------------------
# UI Panel
# -------------------------------------------------

class UV_PT_akelka_snap(bpy.types.Panel):
    bl_label = "Snap UV"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "AkELkA UV Tools"

    def draw(self, context):
        self.layout.operator(UV_OT_snap_to_closest_mesh.bl_idname)


# -------------------------------------------------
# Register
# -------------------------------------------------

classes = (
    UV_OT_snap_to_closest_mesh,
    AkELkASnapUVPreferences,
    UV_PT_akelka_snap,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
