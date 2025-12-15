bl_info = {
    "name": "Akelka Trim Small Weights",
    "author": "Converted by ChatGPT for Akelka",
    "version": (2, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Item",
    "description": "Trim small vertex weights.",
    "category": "Rigging",
}

import bpy

PATREON_URL = "https://www.patreon.com/c/AkELkA"

def register_props():
    WM = bpy.types.WindowManager
    WM.akelka_trim_threshold = bpy.props.FloatProperty(
        name="Threashold", default=0.01, min=0.0, max=1.0, precision=4
    )

def unregister_props():
    WM = bpy.types.WindowManager
    if hasattr(WM, 'akelka_trim_threshold'):
        delattr(WM, 'akelka_trim_threshold')

# --- Helpers ---
def _normalize_mode(mode_str):
    if not mode_str: return 'OBJECT'
    m = mode_str.upper()
    if m.startswith('EDIT'): return 'EDIT'
    if 'WEIGHT' in m or 'PAINT_WEIGHT' in m: return 'WEIGHT_PAINT'
    if 'VERTEX' in m and 'PAINT' in m: return 'VERTEX_PAINT'
    if 'SCULPT' in m: return 'SCULPT'
    if 'POSE' in m: return 'POSE'
    if 'TEXTURE' in m or 'PAINT_TEXTURE' in m: return 'TEXTURE_PAINT'
    if 'PARTICLE' in m or 'PARTICLE_EDIT' in m: return 'PARTICLE_EDIT'
    return m

def _ensure_active_and_selected(obj, context):
    if obj is None: return
    try: obj.select_set(True)
    except Exception: pass
    try: context.view_layer.objects.active = obj
    except Exception: pass

def _find_view3d_area_region(context):
    area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
    if not area: return None, None
    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
    return area, region

def _set_mode(mode, prev_active, context):
    if prev_active is not None: _ensure_active_and_selected(prev_active, context)
    try:
        bpy.ops.object.mode_set(mode=mode)
        if context.mode == mode: return True
    except Exception: pass
    try:
        area, region = _find_view3d_area_region(context)
        if area and region:
            override = {'window': context.window, 'screen': context.window.screen, 'area': area, 'region': region, 'scene': context.scene, 'object': prev_active, 'active_object': prev_active, 'selected_objects': list(context.selected_objects)}
            try:
                bpy.ops.object.mode_set(override, mode=mode)
                if context.mode == mode: return True
            except Exception: pass
    except Exception: pass
    try: bpy.ops.object.mode_set(mode='OBJECT')
    except Exception: pass
    try:
        bpy.ops.object.mode_set(mode=mode)
        if context.mode == mode: return True
    except Exception: pass
    try:
        area, region = _find_view3d_area_region(context)
        if area and region:
            override = {'window': context.window, 'screen': context.window.screen, 'area': area, 'region': region, 'scene': context.scene, 'object': prev_active, 'active_object': prev_active}
            try:
                bpy.ops.object.mode_set(override, mode='OBJECT')
                bpy.ops.object.mode_set(override, mode=mode)
                if context.mode == mode: return True
            except Exception: pass
    except Exception: pass
    return False

def _restore_selection_and_active(prev_selected, prev_active):
    try:
        for o in list(bpy.context.selected_objects):
            try: o.select_set(False)
            except Exception: pass
    except Exception: pass
    for o in prev_selected:
        if o is None: continue
        try: o.select_set(True)
        except Exception: pass
    if prev_active is not None:
        try: bpy.context.view_layer.objects.active = prev_active
        except Exception: pass

# --- Operator ---
class AKELKA_OT_trim_small_weights(bpy.types.Operator):
    bl_idname = "akelka.trim_small_weights"
    bl_label = "Trim Small Weights"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        wm = context.window_manager
        thresh = float(getattr(wm, 'akelka_trim_threshold', 0.01))

        prev_mode_raw = context.mode
        prev_mode = _normalize_mode(prev_mode_raw)
        prev_active = context.active_object
        prev_selected = list(context.selected_objects)

        switched_to_object = False
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
                switched_to_object = True
            except Exception:
                self.report({'WARNING'}, "Could not switch to OBJECT mode; attempting to continue.")

        # Use selected meshes, fallback to active mesh if nothing selected
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not objs:
            if prev_active and prev_active.type == 'MESH':
                objs = [prev_active]

        if not objs:
            if switched_to_object:
                _restore_selection_and_active(prev_selected, prev_active)
                try:
                    if prev_mode != 'OBJECT': _set_mode(prev_mode, prev_active, context)
                except Exception: pass
            self.report({'ERROR'}, "No mesh objects selected or active.")
            return {'CANCELLED'}

        total_removed = 0
        total_checked = 0

        for mesh_obj in objs:
            vg_by_index = {g.index: g for g in mesh_obj.vertex_groups}
            mesh = mesh_obj.data
            for v in mesh.vertices:
                total_checked += 1
                to_remove_indices = [g.group for g in v.groups if g.weight < thresh]
                for gi in to_remove_indices:
                    vg = vg_by_index.get(gi)
                    if vg is None: continue
                    try:
                        vg.remove([v.index])
                        total_removed += 1
                    except Exception:
                        pass

        if switched_to_object:
            _restore_selection_and_active(prev_selected, prev_active)

        restored = False
        if switched_to_object:
            target_mode = prev_mode
            if target_mode == 'WEIGHT_PAINT' and (prev_active is None or prev_active.type != 'MESH'):
                self.report({'INFO'}, "Started in Weight Paint but no mesh active to return to.")
            else:
                if prev_active is not None: _ensure_active_and_selected(prev_active, context)
                restored = _set_mode(target_mode, prev_active, context)

        if restored:
            self.report({'INFO'}, f"Removed {total_removed} weights (checked {total_checked} verts). Restored mode '{prev_mode}'.")
        else:
            if prev_mode != 'OBJECT' and not restored:
                if prev_active is not None: _ensure_active_and_selected(prev_active, context)
                ok = _set_mode(prev_mode, prev_active, context)
                if ok:
                    self.report({'INFO'}, f"Removed {total_removed} weights (checked {total_checked} verts). Restored mode '{prev_mode}'.")
                    return {'FINISHED'}
            self.report({'INFO'}, f"Removed {total_removed} weights (checked {total_checked} verts).")

        return {'FINISHED'}

# --- UI: compact single-row threshold (75%) + smaller Trim button (25%), no box background ---
class VIEW3D_PT_trim_small_weights_panel(bpy.types.Panel):
    bl_label = "Akelka Trim Small Weights"
    bl_idname = "VIEW3D_PT_trim_small_weights_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Item"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        row = layout.row(align=True)
        split = row.split(factor=0.6)
        left = split.row(align=True)
        right = split.row(align=True)
        left.prop(wm, "akelka_trim_threshold", text="Wight:")
        right.operator(AKELKA_OT_trim_small_weights.bl_idname, text="Trim", icon='TRASH')

class AkelkaTrimPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        op = row.operator("wm.url_open", text="Become a Patron", icon='URL')
        op.url = PATREON_URL

# --- Dynamic parent search: try to parent under any panel whose label contains "vertex weights" ---
def _find_panel_with_label_contains(substring):
    substring = substring.lower()
    for name, cls in vars(bpy.types).items():
        try:
            if isinstance(cls, type) and issubclass(cls, bpy.types.Panel):
                lbl = getattr(cls, 'bl_label', '')
                if lbl and substring in lbl.lower():
                    return getattr(cls, 'bl_idname', None)
        except Exception:
            continue
    return None

classes = (
    AKELKA_OT_trim_small_weights,
    VIEW3D_PT_trim_small_weights_panel,
    AkelkaTrimPreferences,
)

def register():
    register_props()
    # try to find a Vertex Weights panel by label and parent under it
    parent_id = _find_panel_with_label_contains("vertex weights")
    if parent_id:
        try:
            VIEW3D_PT_trim_small_weights_panel.bl_parent_id = parent_id
            print(f"Akelka Trim: parented under panel id '{parent_id}' (matched label contains 'vertex weights').")
        except Exception:
            pass
    else:
        print("Akelka Trim: no panel with label containing 'vertex weights' found — registering as top-level in Item tab.")

    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_props()

if __name__ == "__main__":
    register()
