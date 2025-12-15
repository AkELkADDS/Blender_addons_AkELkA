# -*- coding: utf-8 -*-
"""
Vertex Group Filter Add-on for Blender
Filters and manages vertex groups based on selected vertices and weight thresholds.
"""

bl_info = {
    "name": "Akelka - Vertex Group Filter",
    "author": "Akelka",
    "version": (1, 1, 7),
    "blender": (4, 5, 2),
    "location": "View3D > Sidebar > Item Tab",
    "description": "Filter and manage vertex groups based on selected vertices and weight thresholds",
    "category": "Mesh",
    "doc_url": "https://www.patreon.com/c/AkELkA",
    "tracker_url": "",
}

import bpy
import bmesh
import time

# ——————————————————————————
# Module‑level Live Update Flag and Timer Control
# ——————————————————————————
live_update_enabled = True

# Global variables (multi-object mode only)
mixed_filtered_groups = []  # List of tuples: (object name, vertex group index, effective_weight)
prev_selected_indices = {}  # Stores previous vertex selection per object {obj_name: tuple(sorted_indices)}

# Global for throttling updates
_last_update_time = 0.0
_UPDATE_DELAY = 0.15  # Slightly increased delay for stability

# Global flag to control timer update (for addon unregistration)
timer_running = False

# Global variable to track previous mode for handler
_previous_mode = None

# Global variable for merge mode. Holds a tuple (object_name, group_index) or None.
active_merge_group = None

# Global variable to track which vertex group names are currently selected for weight painting
# Set of tuples: (object_name, group_index) - allows multiple selections (one per object)
selected_vgroups = set()
# Global variable to track which vertex group is currently being renamed
# Tuple of (object_name, group_index) or None
selected_vgroup_for_rename = None

# ——————————————————————————
# Helper Functions
# ——————————————————————————
def get_effective_weight(obj, group_index, selected_indices, use_average):
    """ Calculates the effective weight for a group based on selected vertices. """
    if not selected_indices:
        return 0.0
    total_weight = 0.0
    count_with_weight = 0
    max_weight = 0.0
    vertex_groups = obj.vertex_groups
    if not (0 <= group_index < len(vertex_groups)):
        return 0.0
    vg = vertex_groups[group_index]
    relevant_weights = []
    for v_idx in selected_indices:
         try:
             vert = obj.data.vertices[v_idx]
             for g in vert.groups:
                 if g.group == group_index:
                     weight = g.weight
                     if weight > 1e-6:
                         relevant_weights.append(weight)
                     break
         except IndexError:
             continue
    if not relevant_weights:
        return 0.0
    if use_average:
        return sum(relevant_weights) / len(relevant_weights)
    else:
        return max(relevant_weights)

def update_filtered_groups_list(context):
    """ Core logic to rebuild the mixed_filtered_groups list based on current state. """
    global mixed_filtered_groups, prev_selected_indices
    # Safely access scene properties
    try:
        if not hasattr(context, 'scene') or not context.scene:
            return 0
        threshold = context.scene.vg_weight_threshold
        use_average = context.scene.vg_use_average_weight
    except (AttributeError, RuntimeError):
        return 0
    new_filtered_groups = []
    current_selection_snapshot = {}
    for obj in context.selected_editable_objects:
        if obj.type != 'MESH' or obj.mode != 'EDIT':
            continue
        try:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            selected_indices = tuple(sorted(v.index for v in bm.verts if v.select))
            current_selection_snapshot[obj.name] = selected_indices
            if not selected_indices:
                continue
            for i, vg in enumerate(obj.vertex_groups):
                effective_weight = get_effective_weight(obj, i, selected_indices, use_average)
                if effective_weight >= threshold:
                    new_filtered_groups.append((obj.name, i, effective_weight))
        except Exception as e:
            print(f"Error processing object {obj.name}: {e}")
            if obj.name in current_selection_snapshot:
                del current_selection_snapshot[obj.name]
            continue
    def get_sort_key(entry):
        obj_name, gi, _ = entry
        obj = bpy.data.objects.get(obj_name)
        group_name = ""
        if obj and 0 <= gi < len(obj.vertex_groups):
            group_name = obj.vertex_groups[gi].name
        return (obj_name, group_name)
    new_filtered_groups.sort(key=get_sort_key)
    mixed_filtered_groups = new_filtered_groups
    current_valid_obj_names = set(current_selection_snapshot.keys())
    for name in list(prev_selected_indices.keys()):
         if name not in current_valid_obj_names:
             del prev_selected_indices[name]
    prev_selected_indices.update(current_selection_snapshot)
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'UI':
                        region.tag_redraw()
    return len(mixed_filtered_groups)

def update_vg_name(self, context):
    """ Callback for vg_editing_name property change - no longer auto-applies. """
    # This function is kept for compatibility but doesn't auto-apply changes
    # Changes are now confirmed via Enter key or left-click outside, cancelled via right-click
    pass

def update_filter_settings(self, context):
    """ Callback for threshold or average weight property changes. """
    update_filtered_groups_list(context)

def update_search_filter(self, context):
    """ Callback for search filter changes - triggers UI redraw. """
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'UI':
                        region.tag_redraw()

# ——————————————————————————
# Operators
# ——————————————————————————
class OBJECT_OT_update_filtered_vgroups(bpy.types.Operator):
    """ Manually trigger the update of the filtered vertex group list. """
    bl_idname = "object.update_filtered_vgroups"
    bl_label = "Update Filtered Groups List"
    bl_description = "Manually update the vertex groups list based on current selection and filter settings"
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context): return True
    def execute(self, context):
        total_count = update_filtered_groups_list(context)
        return {'FINISHED'}

class OBJECT_OT_reset_filtered_vgroups(bpy.types.Operator):
    """ Show all groups for selected objects, ignoring filters. """
    bl_idname = "object.reset_filtered_vgroups"
    bl_label = "Show All Groups (Ignore Filter)"
    bl_description = "Temporarily show all vertex groups for selected editable meshes with selected vertices (ignores threshold/method)"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global mixed_filtered_groups, prev_selected_indices
        new_filtered_groups = []
        total_count = 0
        current_selection_snapshot = {}
        for obj in context.selected_editable_objects:
            if obj.type != 'MESH' or obj.mode != 'EDIT': continue
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                selected_indices = tuple(sorted(v.index for v in bm.verts if v.select))
                current_selection_snapshot[obj.name] = selected_indices
                if selected_indices:
                    for i, vg in enumerate(obj.vertex_groups):
                        new_filtered_groups.append((obj.name, i, 0.0))
                        total_count += 1
            except Exception as e:
                 print(f"Error accessing edit mesh for reset on {obj.name}: {e}")
                 if obj.name in current_selection_snapshot: del current_selection_snapshot[obj.name]
                 continue
        def get_sort_key(entry):
            obj_name, gi, _ = entry
            obj = bpy.data.objects.get(obj_name)
            group_name = ""
            if obj and 0 <= gi < len(obj.vertex_groups):
                group_name = obj.vertex_groups[gi].name
            return (obj_name, group_name)
        new_filtered_groups.sort(key=get_sort_key)
        mixed_filtered_groups = new_filtered_groups
        current_valid_obj_names = set(current_selection_snapshot.keys())
        for name in list(prev_selected_indices.keys()):
             if name not in current_valid_obj_names: del prev_selected_indices[name]
        prev_selected_indices.update(current_selection_snapshot)
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'UI': region.tag_redraw()
        self.report({'INFO'}, f"Showing all {total_count} groups for objects with selected vertices.")
        return {'FINISHED'}

class OBJECT_OT_show_zero_weight_info(bpy.types.Operator):
    """ Display info about groups with no vertex weights assigned. """
    bl_idname = "object.show_zero_weight_info"
    bl_label = "Zero Weight Info"
    bl_description = "List vertex groups with zero total weight across all vertices for selected mesh objects"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        info_lines = ["Zero-Weight Groups per Selected Mesh:"]
        found_mesh = False
        for obj in context.selected_objects:
            if obj.type != 'MESH': continue
            found_mesh = True
            if not obj.data or not hasattr(obj.data, 'vertices'):
                 info_lines.append(f"- {obj.name}: (No mesh data)")
                 continue
            if not obj.vertex_groups:
                 info_lines.append(f"- {obj.name}: (No vertex groups)")
                 continue
            group_has_weight = {vg.index: False for vg in obj.vertex_groups}
            valid_group_indices = set(group_has_weight.keys())
            for v in obj.data.vertices:
                for g in v.groups:
                    if g.group in valid_group_indices and not group_has_weight[g.group]:
                         if g.weight > 1e-6: group_has_weight[g.group] = True
            zero_weight_group_names = [vg.name for vg in obj.vertex_groups
                                       if vg.index in valid_group_indices and not group_has_weight[vg.index]]
            if zero_weight_group_names:
                info_lines.append(f"- {obj.name} [{len(zero_weight_group_names)}]: {', '.join(zero_weight_group_names)}")
            else: info_lines.append(f"- {obj.name}: (None)")
        info_text = "\n".join(info_lines) if found_mesh else "No mesh objects selected."
        def draw_popup(self_popup, context_popup):
            layout = self_popup.layout
            for line in info_text.splitlines(): layout.label(text=line)
        context.window_manager.popup_menu(draw_popup, title="Zero Weight Info", icon='INFO')
        return {'FINISHED'}

class MESH_OT_select_vgroup_verts(bpy.types.Operator):
    """ Select vertices belonging to the clicked group. """
    bl_idname = "mesh.select_vgroup_verts"
    bl_label = "Select Vertices in Group"
    bl_description = "Select all vertices influenced by this vertex group (Turns Live Update OFF)"
    bl_options = {'REGISTER', 'UNDO'}
    group_idx: bpy.props.IntProperty(options={'HIDDEN'})
    group_obj: bpy.props.StringProperty(options={'HIDDEN'})
    @classmethod
    def poll(cls, context):
        obj = bpy.data.objects.get(cls.group_obj) if hasattr(cls, 'group_obj') else context.active_object
        return obj and obj.mode == 'EDIT' and obj.type == 'MESH'
    def execute(self, context):
        # If rename mode is active, confirm the rename first
        if context.scene.vg_rename_mode:
            bpy.ops.mesh.confirm_vgroup_rename('EXEC_DEFAULT')
        global live_update_enabled
        target_obj = bpy.data.objects.get(self.group_obj)
        if not target_obj or target_obj.type != 'MESH':
            self.report({'ERROR'}, f"Target object '{self.group_obj}' not found or not a mesh.")
            return {'CANCELLED'}
        if context.active_object != target_obj or target_obj.mode != 'EDIT':
             try:
                 bpy.ops.object.mode_set(mode='OBJECT')
                 bpy.ops.object.select_all(action='DESELECT')
                 target_obj.select_set(True)
                 context.view_layer.objects.active = target_obj
                 bpy.ops.object.mode_set(mode='EDIT')
             except RuntimeError as e:
                 self.report({'ERROR'}, f"Failed to activate/set Edit Mode for {target_obj.name}: {e}")
                 return {'CANCELLED'}
        if not (0 <= self.group_idx < len(target_obj.vertex_groups)):
            self.report({'ERROR'}, f"Invalid group index {self.group_idx} for {target_obj.name}.")
            return {'CANCELLED'}
        vg = target_obj.vertex_groups[self.group_idx]
        live_update_enabled = False
        context.scene.vg_live_update = False
        try:
            target_obj.vertex_groups.active_index = self.group_idx
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.vertex_group_select()
        except RuntimeError as e:
             self.report({'ERROR'}, f"Error during selection for group '{vg.name}': {e}")
             return {'CANCELLED'}
        self.report({'INFO'}, f"Selected verts of '{vg.name}'. Live updates turned OFF.")
        return {'FINISHED'}

class OBJECT_OT_toggle_live_update(bpy.types.Operator):
    """ Toggle the automatic update of the group list on/off. """
    bl_idname = "object.toggle_live_update"
    bl_label = "Toggle Live Update"
    bl_description = "Toggle automatic filtering based on vertex selection changes"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global live_update_enabled
        live_update_enabled = not live_update_enabled
        context.scene.vg_live_update = live_update_enabled
        status = "ON" if live_update_enabled else "OFF"
        if live_update_enabled: bpy.ops.object.update_filtered_vgroups('INVOKE_DEFAULT')
        return {'FINISHED'}

class OBJECT_OT_lock_matching_vgroups(bpy.types.Operator):
    """ Lock groups with the same name across selected objects. """
    bl_idname = "object.lock_matching_vgroups"
    bl_label = "Lock Shared Groups"
    bl_description = ("Lock vertex groups sharing the same name across selected mesh objects"
                      " to prevent accidental weight edits")
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
         selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
         return len(selected_meshes) >= 2
    def execute(self, context):
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if len(selected_objects) < 2:
             self.report({'WARNING'}, "Select at least two mesh objects.")
             return {'CANCELLED'}
        group_references = {}
        for obj in selected_objects:
            for i, vg in enumerate(obj.vertex_groups):
                if vg.name not in group_references: group_references[vg.name] = []
                group_references[vg.name].append((obj, i))
        names_to_lock = {name for name, refs in group_references.items() if len(refs) > 1}
        if not names_to_lock:
             self.report({'INFO'}, "No vertex group names are shared by more than one selected object.")
             return {'FINISHED'}
        locked_count = 0
        locked_names = set()
        for name in names_to_lock:
            for obj, vg_index in group_references[name]:
                 try:
                    vg = obj.vertex_groups[vg_index]
                    if not vg.lock_weight:
                         vg.lock_weight = True
                         locked_count += 1
                         locked_names.add(name)
                 except (IndexError, AttributeError) as e:
                     print(f"Warning: Error accessing group '{name}' on {obj.name}: {e}")
                     continue
        if locked_count > 0: self.report({'INFO'}, f"Locked {locked_count} instances of {len(locked_names)} shared group(s).")
        else: self.report({'INFO'}, "All shared groups were already locked.")
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}

class OBJECT_OT_clear_vgroup_search(bpy.types.Operator):
    """ Clear the search filter. """
    bl_idname = "object.clear_vgroup_search"
    bl_label = "Clear Search"
    bl_description = "Clear the search filter"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}
    
    def execute(self, context):
        context.scene.vg_search_filter = ""
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}

class OBJECT_OT_toggle_locked_vgroups_filter(bpy.types.Operator):
    """ Toggle visibility of locked groups in the panel list. """
    bl_idname = "object.toggle_locked_vgroups_filter"
    bl_label = "Toggle Show Locked Groups"
    bl_description = "Toggle displaying locked vertex groups in the filtered list"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        scene = context.scene
        scene.show_locked_vgroups = not scene.show_locked_vgroups
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}

def find_object_list_under_mouse(context, event):
    """ Find which object's list the mouse is over based on mouse position. """
    global mixed_filtered_groups
    
    # Find the UI region that contains the mouse
    ui_region = None
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'UI':
                    # Check if mouse is within this region's bounds
                    if (region.x <= event.mouse_x <= region.x + region.width and
                        region.y <= event.mouse_y <= region.y + region.height):
                        ui_region = region
                        break
            if ui_region:
                break
    
    if not ui_region:
        return None
    
    # Calculate Y position from top of region
    # In Blender: region.y is bottom Y in screen coords, mouse_y is screen Y
    # Region coordinates: (0,0) is bottom-left, Y increases upward
    # UI coordinates: (0,0) is top-left, Y increases downward
    # Convert: mouse_y_in_ui = region.height - (event.mouse_y - region.y)
    mouse_y_screen = event.mouse_y
    region_bottom_y = ui_region.y
    mouse_y_in_ui = ui_region.height - (mouse_y_screen - region_bottom_y)
    
    # Build list of all visible groups with their approximate Y positions
    grouped_by_object = {}
    for entry in mixed_filtered_groups:
        obj_name, gi, weight_val = entry
        grouped_by_object.setdefault(obj_name, []).append((gi, weight_val))
    
    # Approximate positions - adjust these values if needed
    # Account for: settings box (~80px), control rows (~50px), separator (~5px), label (~20px)
    # Total header: ~155px
    start_y = 155  # Start of group list from top of region
    obj_header_height = 30  # Object name header
    button_height = 28  # Button height
    spacing = 0  # Spacing between buttons
    box_padding_top = 8  # Padding at top of box
    
    current_y = start_y + box_padding_top
    
    # Calculate column widths (assuming equal distribution)
    num_objects = len(grouped_by_object)
    if num_objects == 0:
        return None
    
    # Approximate column width (this is a rough estimate)
    # We'll check if mouse X is within reasonable bounds for each column
    region_width = ui_region.width
    column_width = region_width / num_objects if num_objects > 0 else region_width
    
    for obj_idx, (obj_name, entries) in enumerate(sorted(grouped_by_object.items())):
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            continue
        
        # Check if mouse X is within this column's bounds
        column_start_x = obj_idx * column_width
        column_end_x = (obj_idx + 1) * column_width
        mouse_x_in_ui = event.mouse_x - ui_region.x
        
        # Check if mouse is within this column (with some tolerance)
        if not (column_start_x - 10 <= mouse_x_in_ui <= column_end_x + 10):
            # Skip to next object, but still need to calculate Y positions
            current_y += obj_header_height
            entries_sorted = sorted(entries, key=lambda x: obj.vertex_groups[x[0]].name if 0 <= x[0] < len(obj.vertex_groups) else "")
            for entry in entries_sorted:
                gi, weight_val = entry
                if not (0 <= gi < len(obj.vertex_groups)):
                    continue
                vg = obj.vertex_groups[gi]
                if not context.scene.show_locked_vgroups and vg.lock_weight:
                    continue
                current_y += button_height + spacing
            continue
        
        # Mouse is in this column, check Y position
        current_y += obj_header_height
        
        entries_sorted = sorted(entries, key=lambda x: obj.vertex_groups[x[0]].name if 0 <= x[0] < len(obj.vertex_groups) else "")
        
        # Check if mouse is in the object header area
        if current_y - obj_header_height <= mouse_y_in_ui <= current_y:
            return obj_name
        
        # Check if mouse is over any button in this object's list
        for entry in entries_sorted:
            gi, weight_val = entry
            if not (0 <= gi < len(obj.vertex_groups)):
                continue
            vg = obj.vertex_groups[gi]
            if not context.scene.show_locked_vgroups and vg.lock_weight:
                continue
            
            button_top = current_y
            button_bottom = current_y + button_height
            
            if button_top <= mouse_y_in_ui <= button_bottom:
                return obj_name
            
            current_y += button_height + spacing
        
        # If we get here and mouse is in this column, return this object
        if column_start_x <= mouse_x_in_ui <= column_end_x:
            return obj_name
    
    return None

def scroll_vgroup_selection(context, obj_name, direction):
    """ Scroll through vertex groups for a specific object. """
    global selected_vgroups, selected_vgroup_for_rename, mixed_filtered_groups
    
    obj = bpy.data.objects.get(obj_name)
    if not obj or obj.type != 'MESH':
        return False
    
    # Apply search filter to mixed_filtered_groups
    search_term = context.scene.vg_search_filter.strip().lower() if context.scene.vg_search_filter else ""
    filtered_by_search = mixed_filtered_groups
    if search_term:
        filtered_by_search = []
        for entry in mixed_filtered_groups:
            entry_obj_name, gi, weight_val = entry
            entry_obj = bpy.data.objects.get(entry_obj_name)
            if entry_obj and 0 <= gi < len(entry_obj.vertex_groups):
                group_name = entry_obj.vertex_groups[gi].name.lower()
                if search_term in group_name:
                    filtered_by_search.append(entry)
    
    # Get filtered groups for this object (after search filter)
    obj_groups = []
    for entry in filtered_by_search:
        entry_obj_name, gi, weight_val = entry
        if entry_obj_name == obj_name:
            # Also check locked groups filter
            if 0 <= gi < len(obj.vertex_groups):
                vg = obj.vertex_groups[gi]
                if context.scene.show_locked_vgroups or not vg.lock_weight:
                    obj_groups.append((gi, weight_val))
    
    if not obj_groups:
        return False
    
    # Sort by group name
    obj_groups.sort(key=lambda x: obj.vertex_groups[x[0]].name if 0 <= x[0] < len(obj.vertex_groups) else "")
    
    # Find currently selected group for this object
    current_selected_idx = -1
    for i, (gi, _) in enumerate(obj_groups):
        if (obj_name, gi) in selected_vgroups:
            current_selected_idx = i
            break
    
    # Determine new index
    if direction > 0:  # Scroll down (next)
        new_idx = (current_selected_idx + 1) % len(obj_groups)
    else:  # Scroll up (previous)
        new_idx = (current_selected_idx - 1) % len(obj_groups) if current_selected_idx >= 0 else len(obj_groups) - 1
    
    # Get the new group
    new_gi, _ = obj_groups[new_idx]
    new_selection_tuple = (obj_name, new_gi)
    
    # Update selection - remove other groups from same object, add new one
    selected_vgroups = {t for t in selected_vgroups if t[0] != obj_name}
    selected_vgroups.add(new_selection_tuple)
    selected_vgroup_for_rename = new_selection_tuple
    context.scene.vg_rename_mode = False
    
    # Set active index for this object
    try:
        obj.vertex_groups.active_index = new_gi
    except Exception as e:
        print(f"Note: Could not set active index for {obj.name}: {e}")
    
    # Redraw UI
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    
    return True

class MESH_OT_vgroup_wheel_scroll(bpy.types.Operator):
    """ Handle mouse wheel scrolling to change weight group selection. """
    bl_idname = "mesh.vgroup_wheel_scroll"
    bl_label = "Scroll Vertex Groups"
    bl_description = "Scroll through vertex groups with mouse wheel"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}
    
    _instance = None
    _running = False
    
    def modal(self, context, event):
        # Always stay running if in edit mode
        if not self.poll(context):
            # If we leave edit mode, cancel but allow restart
            MESH_OT_vgroup_wheel_scroll._running = False
            return {'PASS_THROUGH'}
        
        # Handle mouse wheel events
        if event.type == 'WHEELUPMOUSE':
            obj_name = find_object_list_under_mouse(context, event)
            if obj_name:
                scroll_vgroup_selection(context, obj_name, -1)  # Scroll up (previous)
                return {'RUNNING_MODAL'}
        
        elif event.type == 'WHEELDOWNMOUSE':
            obj_name = find_object_list_under_mouse(context, event)
            if obj_name:
                scroll_vgroup_selection(context, obj_name, 1)  # Scroll down (next)
                return {'RUNNING_MODAL'}
        
        # Pass through all other events
        return {'PASS_THROUGH'}
    
    def invoke(self, context, event):
        if not self.poll(context):
            return {'CANCELLED'}
        
        # Store instance and start modal
        MESH_OT_vgroup_wheel_scroll._instance = self
        MESH_OT_vgroup_wheel_scroll._running = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        """ Execute method to allow starting via EXEC_DEFAULT. """
        if not self.poll(context):
            return {'CANCELLED'}
        
        # Store instance and start modal
        MESH_OT_vgroup_wheel_scroll._instance = self
        MESH_OT_vgroup_wheel_scroll._running = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        """Called when the operator is cancelled."""
        MESH_OT_vgroup_wheel_scroll._running = False
        MESH_OT_vgroup_wheel_scroll._instance = None
    
    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.active_object and context.active_object.type == 'MESH')
    
    @classmethod
    def ensure_running(cls, context):
        """ Ensure the modal is running if conditions are met. """
        if cls.poll(context) and not cls._running:
            try:
                # Check if we're in a safe state to invoke operators
                # Can't invoke during draw/rendering
                if bpy.app.is_rendering or bpy.app.is_exporting:
                    return
                
                # Use timer to defer the operator invocation if we're in draw context
                def deferred_start():
                    try:
                        if cls.poll(bpy.context) and not cls._running:
                            bpy.ops.mesh.vgroup_wheel_scroll('INVOKE_DEFAULT')
                    except Exception:
                        pass
                    return None  # Run once
                
                # Try immediate start first
                try:
                    bpy.ops.mesh.vgroup_wheel_scroll('INVOKE_DEFAULT')
                except RuntimeError as e:
                    if "can't modify blend data in this state" in str(e):
                        # We're in draw context, defer it
                        if not bpy.app.timers.is_registered(deferred_start):
                            bpy.app.timers.register(deferred_start, first_interval=0.01)
                    else:
                        raise
            except Exception:
                pass

class MESH_OT_vgroup_name_click(bpy.types.Operator):
    """ Handle clicks on group names for selection and rename trigger. """
    bl_idname = "mesh.vgroup_name_click"
    bl_label = "Vertex Group Name Click"
    bl_description = "First click selects group, second click enables renaming"
    bl_options = {'REGISTER', 'UNDO'}
    group_idx: bpy.props.IntProperty(options={'HIDDEN'})
    group_obj: bpy.props.StringProperty(options={'HIDDEN'})
    def execute(self, context):
        global selected_vgroup_for_rename, selected_vgroups
        obj = bpy.data.objects.get(self.group_obj)
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Invalid object for name click.")
            return {'CANCELLED'}
        if context.view_layer.objects.active != obj:
             if obj in context.selectable_objects: context.view_layer.objects.active = obj
             else: self.report({'WARNING'}, f"{obj.name} not active/selectable.")
        if not (0 <= self.group_idx < len(obj.vertex_groups)):
            self.report({'ERROR'}, "Invalid group index for name click.")
            selected_vgroup_for_rename = None
            context.scene.vg_rename_mode = False
            return {'CANCELLED'}
        vg = obj.vertex_groups[self.group_idx]
        current_selection_tuple = (self.group_obj, self.group_idx)
        
        # Check if this group is already selected
        is_already_selected = current_selection_tuple in selected_vgroups
        
        if is_already_selected and selected_vgroup_for_rename == current_selection_tuple:
            # Second click on already selected group - enter rename mode
            context.scene.vg_rename_mode = True
            context.scene.vg_editing_name = vg.name
            # Force UI redraw to show the text field
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
            # Start modal operator for Enter/Esc key handling
            # The modal passes through all mouse events, so it won't interfere with text field
            bpy.ops.mesh.rename_vgroup_modal('INVOKE_DEFAULT')
        else:
            # First click or click on different group - toggle selection
            if is_already_selected:
                # Remove from selection (toggle off)
                selected_vgroups.discard(current_selection_tuple)
                if selected_vgroup_for_rename == current_selection_tuple:
                    selected_vgroup_for_rename = None
                    context.scene.vg_rename_mode = False
            else:
                # Add to selection - but only one group per object
                # Remove any other groups from the same object first
                selected_vgroups = {t for t in selected_vgroups if t[0] != self.group_obj}
                selected_vgroups.add(current_selection_tuple)
                selected_vgroup_for_rename = current_selection_tuple
                context.scene.vg_rename_mode = False
            # Set active index for this object
            try: obj.vertex_groups.active_index = self.group_idx
            except Exception as e: print(f"Note: Could not set active index for {obj.name}: {e}")
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}
    
    def invoke(self, context, event):
        # Handle right-click to cancel if in rename mode
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS' and context.scene.vg_rename_mode:
            bpy.ops.mesh.cancel_vgroup_rename('EXEC_DEFAULT')
            return {'FINISHED'}
        # For all other events (including left-clicks), execute normally
        return self.execute(context)

class MESH_OT_confirm_vgroup_rename(bpy.types.Operator):
    """ Confirm and apply the vertex group name change. """
    bl_idname = "mesh.confirm_vgroup_rename"
    bl_label = "Confirm Rename"
    bl_description = "Apply the new vertex group name (or press Enter)"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global selected_vgroup_for_rename
        # If not in rename mode, do nothing (button is shown for layout consistency)
        if not context.scene.vg_rename_mode or not selected_vgroup_for_rename:
            return {'CANCELLED'}
        obj_name, group_idx = selected_vgroup_for_rename
        obj = bpy.data.objects.get(obj_name)
        new_name = context.scene.vg_editing_name.strip()
        if not new_name:
            self.report({'WARNING'}, "Vertex group name cannot be empty.")
            if obj and 0 <= group_idx < len(obj.vertex_groups):
                context.scene.vg_editing_name = obj.vertex_groups[group_idx].name
            return {'CANCELLED'}
        if not obj or not (0 <= group_idx < len(obj.vertex_groups)):
            self.report({'ERROR'}, "Invalid object or group index.")
            selected_vgroup_for_rename = None
            context.scene.vg_rename_mode = False
            return {'CANCELLED'}
        current_name = obj.vertex_groups[group_idx].name
        if new_name == current_name:
            # Same name, just exit rename mode
            selected_vgroup_for_rename = None
            context.scene.vg_rename_mode = False
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
            return {'FINISHED'}
        existing_names = {vg.name for i, vg in enumerate(obj.vertex_groups) if i != group_idx}
        if new_name not in existing_names:
            try:
                obj.vertex_groups[group_idx].name = new_name
                self.report({'INFO'}, f"Renamed '{current_name}' to '{new_name}'.")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename group: {e}")
                context.scene.vg_editing_name = current_name
                return {'CANCELLED'}
        else:
            self.report({'WARNING'}, f"Name '{new_name}' already exists on {obj_name}. Reverting.")
            context.scene.vg_editing_name = current_name
            return {'CANCELLED'}
        # Exit rename mode
        MESH_OT_rename_vgroup_modal._running = False
        selected_vgroup_for_rename = None
        context.scene.vg_rename_mode = False
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}

class MESH_OT_cancel_vgroup_rename(bpy.types.Operator):
    """ Cancel the vertex group rename operation. """
    bl_idname = "mesh.cancel_vgroup_rename"
    bl_label = "Cancel Rename"
    bl_description = "Cancel renaming (or right-click)"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        global selected_vgroup_for_rename
        MESH_OT_rename_vgroup_modal._running = False
        if selected_vgroup_for_rename:
            obj_name, group_idx = selected_vgroup_for_rename
            obj = bpy.data.objects.get(obj_name)
            if obj and 0 <= group_idx < len(obj.vertex_groups):
                # Restore original name
                context.scene.vg_editing_name = obj.vertex_groups[group_idx].name
        selected_vgroup_for_rename = None
        context.scene.vg_rename_mode = False
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}
    def invoke(self, context, event):
        # Handle right-click
        if event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            return self.execute(context)
        return {'CANCELLED'}

class MESH_OT_rename_vgroup_modal(bpy.types.Operator):
    """ Modal operator to handle Enter/Esc keys during rename. """
    bl_idname = "mesh.rename_vgroup_modal"
    bl_label = "Rename Vertex Group Modal"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}
    
    _running = False
    
    def modal(self, context, event):
        # Exit if rename mode is no longer active
        if not context.scene.vg_rename_mode:
            MESH_OT_rename_vgroup_modal._running = False
            return {'CANCELLED'}
        
        # Handle Enter key - confirm rename
        if event.type == 'RET' and event.value == 'PRESS':
            MESH_OT_rename_vgroup_modal._running = False
            bpy.ops.mesh.confirm_vgroup_rename('EXEC_DEFAULT')
            return {'FINISHED'}
        
        # Handle Esc key - cancel rename
        if event.type == 'ESC' and event.value == 'PRESS':
            MESH_OT_rename_vgroup_modal._running = False
            bpy.ops.mesh.cancel_vgroup_rename('EXEC_DEFAULT')
            return {'CANCELLED'}
        
        # Pass through ALL events (including mouse) to allow normal text editing and UI interaction
        # The text field will handle its own clicks, and we only intercept keyboard events above
        return {'PASS_THROUGH'}
    
    def invoke(self, context, event):
        # Only start if not already running and rename mode is active
        if context.scene.vg_rename_mode and not MESH_OT_rename_vgroup_modal._running:
            MESH_OT_rename_vgroup_modal._running = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        return {'CANCELLED'}

class MESH_OT_toggle_vgroup_lock(bpy.types.Operator):
    """ Toggle lock state of a vertex group. """
    bl_idname = "mesh.toggle_vgroup_lock"
    bl_label = "Toggle Vertex Group Lock"
    bl_description = "Toggle lock state of this vertex group"
    bl_options = {'REGISTER', 'UNDO'}
    group_idx: bpy.props.IntProperty(options={'HIDDEN'})
    group_obj: bpy.props.StringProperty(options={'HIDDEN'})
    @classmethod
    def poll(cls, context):
        obj = bpy.data.objects.get(cls.group_obj) if hasattr(cls, 'group_obj') else context.active_object
        return obj and obj.type == 'MESH'
    def execute(self, context):
        # If rename mode is active, confirm the rename first
        if context.scene.vg_rename_mode:
            bpy.ops.mesh.confirm_vgroup_rename('EXEC_DEFAULT')
        target_obj = bpy.data.objects.get(self.group_obj)
        if not target_obj or target_obj.type != 'MESH':
            self.report({'ERROR'}, f"Target object '{self.group_obj}' not found or not a mesh.")
            return {'CANCELLED'}
        if not (0 <= self.group_idx < len(target_obj.vertex_groups)):
            self.report({'ERROR'}, f"Invalid group index {self.group_idx} for {target_obj.name}.")
            return {'CANCELLED'}
        vg = target_obj.vertex_groups[self.group_idx]
        vg.lock_weight = not vg.lock_weight
        status = "locked" if vg.lock_weight else "unlocked"
        self.report({'INFO'}, f"Group '{vg.name}' {status}.")
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}

class MESH_OT_rename_vgroup_left_to_right(bpy.types.Operator):
    """ Rename the left object's selected group to match the right object's selected group name. """
    bl_idname = "mesh.rename_vgroup_left_to_right"
    bl_label = "Rename Left to Right"
    bl_description = "Rename the left object's selected group to match the right object's selected group name"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global selected_vgroups
        if len(selected_vgroups) < 2:
            self.report({'WARNING'}, "Please select groups from two different objects.")
            return {'CANCELLED'}
        
        # Get selected groups sorted by object name (to determine left/right)
        selected_list = sorted(selected_vgroups, key=lambda x: x[0])
        if len(selected_list) < 2:
            self.report({'WARNING'}, "Please select groups from two different objects.")
            return {'CANCELLED'}
        
        left_obj_name, left_group_idx = selected_list[0]
        right_obj_name, right_group_idx = selected_list[1]
        
        left_obj = bpy.data.objects.get(left_obj_name)
        right_obj = bpy.data.objects.get(right_obj_name)
        
        if not left_obj or not right_obj:
            self.report({'ERROR'}, "One or both objects not found.")
            return {'CANCELLED'}
        
        if not (0 <= left_group_idx < len(left_obj.vertex_groups)) or \
           not (0 <= right_group_idx < len(right_obj.vertex_groups)):
            self.report({'ERROR'}, "Invalid group indices.")
            return {'CANCELLED'}
        
        right_group = right_obj.vertex_groups[right_group_idx]
        target_name = right_group.name
        
        # Check if name already exists on left object
        existing_names = {vg.name for i, vg in enumerate(left_obj.vertex_groups) if i != left_group_idx}
        if target_name in existing_names:
            self.report({'WARNING'}, f"Name '{target_name}' already exists on {left_obj.name}.")
            return {'CANCELLED'}
        
        left_group = left_obj.vertex_groups[left_group_idx]
        old_name = left_group.name
        try:
            left_group.name = target_name
            self.report({'INFO'}, f"Renamed '{old_name}' to '{target_name}' on {left_obj.name}.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to rename: {e}")
            return {'CANCELLED'}
        
        # Update selection to reflect new name (group index stays the same)
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        
        return {'FINISHED'}

class MESH_OT_rename_vgroup_right_to_left(bpy.types.Operator):
    """ Rename the right object's selected group to match the left object's selected group name. """
    bl_idname = "mesh.rename_vgroup_right_to_left"
    bl_label = "Rename Right to Left"
    bl_description = "Rename the right object's selected group to match the left object's selected group name"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global selected_vgroups
        if len(selected_vgroups) < 2:
            self.report({'WARNING'}, "Please select groups from two different objects.")
            return {'CANCELLED'}
        
        # Get selected groups sorted by object name (to determine left/right)
        selected_list = sorted(selected_vgroups, key=lambda x: x[0])
        if len(selected_list) < 2:
            self.report({'WARNING'}, "Please select groups from two different objects.")
            return {'CANCELLED'}
        
        left_obj_name, left_group_idx = selected_list[0]
        right_obj_name, right_group_idx = selected_list[1]
        
        left_obj = bpy.data.objects.get(left_obj_name)
        right_obj = bpy.data.objects.get(right_obj_name)
        
        if not left_obj or not right_obj:
            self.report({'ERROR'}, "One or both objects not found.")
            return {'CANCELLED'}
        
        if not (0 <= left_group_idx < len(left_obj.vertex_groups)) or \
           not (0 <= right_group_idx < len(right_obj.vertex_groups)):
            self.report({'ERROR'}, "Invalid group indices.")
            return {'CANCELLED'}
        
        left_group = left_obj.vertex_groups[left_group_idx]
        target_name = left_group.name
        
        # Check if name already exists on right object
        existing_names = {vg.name for i, vg in enumerate(right_obj.vertex_groups) if i != right_group_idx}
        if target_name in existing_names:
            self.report({'WARNING'}, f"Name '{target_name}' already exists on {right_obj.name}.")
            return {'CANCELLED'}
        
        right_group = right_obj.vertex_groups[right_group_idx]
        old_name = right_group.name
        try:
            right_group.name = target_name
            self.report({'INFO'}, f"Renamed '{old_name}' to '{target_name}' on {right_obj.name}.")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to rename: {e}")
            return {'CANCELLED'}
        
        # Update selection to reflect new name (group index stays the same)
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        
        return {'FINISHED'}

class MESH_OT_merge_vgroup(bpy.types.Operator):
    """ Merge source vertex group into target group. """
    bl_idname = "mesh.merge_vgroup"
    bl_label = "Merge Vertex Group"
    bl_description = ("1st click: Set source. 2nd click (diff group, same object): Merge source into target. "
                      "Click source again to cancel. Source deleted if empty.")
    bl_options = {'REGISTER', 'UNDO'}
    group_idx: bpy.props.IntProperty(options={'HIDDEN'})
    group_obj: bpy.props.StringProperty(options={'HIDDEN'})
    @classmethod
    def poll(cls, context):
        obj = bpy.data.objects.get(cls.group_obj) if hasattr(cls, 'group_obj') else context.active_object
        return obj and obj.type == 'MESH'
    def execute(self, context):
        # If rename mode is active, confirm the rename first
        if context.scene.vg_rename_mode:
            bpy.ops.mesh.confirm_vgroup_rename('EXEC_DEFAULT')
        global active_merge_group
        target_obj = bpy.data.objects.get(self.group_obj)
        if not target_obj or target_obj.type != 'MESH':
            self.report({'ERROR'}, "Merge object error.")
            return {'CANCELLED'}
        original_mode = target_obj.mode
        is_edit_mode = (original_mode == 'EDIT')
        if is_edit_mode:
            try: bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError as e:
                self.report({'ERROR'}, f"Could not switch to Object Mode for merge: {e}")
                return {'CANCELLED'}
        try:
            if not (0 <= self.group_idx < len(target_obj.vertex_groups)):
                self.report({'ERROR'}, f"Group index {self.group_idx} invalid for {target_obj.name}.")
                return {'CANCELLED'}
            clicked_group_tuple = (self.group_obj, self.group_idx)
            if active_merge_group is None:
                active_merge_group = clicked_group_tuple
                source_name = target_obj.vertex_groups[self.group_idx].name
                self.report({'INFO'}, f"Merge Source: '{source_name}'. Click target group.")
            elif active_merge_group == clicked_group_tuple:
                active_merge_group = None
                self.report({'INFO'}, "Merge cancelled.")
            else:
                source_obj_name, source_idx = active_merge_group
                target_obj_name, target_idx = clicked_group_tuple
                if source_obj_name != target_obj_name:
                    self.report({'ERROR'}, "Cannot merge groups from different objects.")
                    active_merge_group = None
                    return {'CANCELLED'}
                if not (0 <= source_idx < len(target_obj.vertex_groups)) or \
                   not (0 <= target_idx < len(target_obj.vertex_groups)):
                    self.report({'ERROR'}, "Source/target group index became invalid.")
                    active_merge_group = None
                    return {'CANCELLED'}
                source_vg = target_obj.vertex_groups[source_idx]
                target_vg = target_obj.vertex_groups[target_idx]
                source_name = source_vg.name
                target_name = target_vg.name
                verts_to_update = {}
                for v in target_obj.data.vertices:
                    source_weight = 0.0
                    for g in v.groups:
                        if g.group == source_idx: source_weight = g.weight; break
                    if source_weight > 1e-6: verts_to_update[v.index] = source_weight
                if not verts_to_update:
                    self.report({'WARNING'}, f"Source '{source_name}' has no weights to merge.")
                else:
                    for v_idx, s_weight in verts_to_update.items():
                        target_vg.add([v_idx], s_weight, 'ADD')
                    source_vg.remove(list(verts_to_update.keys()))
                    self.report({'INFO'}, f"Merged '{source_name}' into '{target_name}'.")
                source_is_empty = True
                for v in target_obj.data.vertices:
                     for g in v.groups:
                          if g.group == source_idx and g.weight > 1e-6: source_is_empty = False; break
                     if not source_is_empty: break
                if source_is_empty:
                     try:
                          target_obj.vertex_groups.remove(source_vg)
                          self.report({'INFO'}, f"Removed empty source group '{source_name}'.")
                     except Exception as e:
                          self.report({'WARNING'}, f"Could not remove source group '{source_name}': {e}")
                active_merge_group = None  # Reset state after merge attempt
        except Exception as e:
            self.report({'ERROR'}, f"Unexpected error during merge: {e}")
            active_merge_group = None
            return {'CANCELLED'}
        finally:
            if target_obj.mode != original_mode:
                try: bpy.ops.object.mode_set(mode=original_mode)
                except RuntimeError as e: print(f"Warning: Could not restore original mode '{original_mode}': {e}")
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        return {'FINISHED'}

# ——————————————————————————
# Panel
# ——————————————————————————
class VIEW3D_PT_filtered_vertex_groups(bpy.types.Panel):
    """ UI Panel for displaying and interacting with filtered vertex groups. """
    bl_label = "Filtered Vertex Groups"
    bl_idname = "VIEW3D_PT_filtered_vertex_groups"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Item'
    bl_context = "mesh_edit"
    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH' and
                context.active_object and context.active_object.type == 'MESH')
    def draw(self, context):
        global mixed_filtered_groups, active_merge_group, selected_vgroup_for_rename, selected_vgroups
        
        # DO NOT call ensure_running() from draw() - it causes "can't modify blend data in this state" error
        # The timer and handlers will start the modal operator instead
        
        layout = self.layout
        scene = context.scene

        # --- Settings Box ---
        settings_box = layout.box()
        col_settings = settings_box.column(align=True)
        row_thresh = col_settings.row(align=True)
        row_thresh.prop(scene, "vg_weight_threshold", text="Threshold")
        row_thresh.prop(scene, "vg_use_average_weight", text="Avg", toggle=True)

        # --- Control Row ---
        row_controls = layout.row(align=True)
        live_icon = 'CHECKBOX_HLT' if scene.vg_live_update else 'CHECKBOX_DEHLT'
        row_controls.operator("object.toggle_live_update", text="Live", icon=live_icon, depress=scene.vg_live_update)
        row_controls.operator("object.update_filtered_vgroups", text="Update", icon='FILE_REFRESH')
        row_controls.operator("object.reset_filtered_vgroups", text="All", icon='SELECT_SET')

        # --- Lock & Info Row ---
        row_locks = layout.row(align=True)
        row_locks.operator("object.lock_matching_vgroups", text="Lock Shared", icon='LOCKED')
        lock_filter_icon = 'HIDE_OFF' if scene.show_locked_vgroups else 'HIDE_ON'
        row_locks.operator("object.toggle_locked_vgroups_filter", text="", icon=lock_filter_icon)
        row_locks.operator("object.show_zero_weight_info", text="", icon='QUESTION')

        layout.separator()

        # --- Search Box ---
        row_search = layout.row(align=True)
        row_search.prop(scene, "vg_search_filter", text="", icon='VIEWZOOM', placeholder="Search groups...")
        if scene.vg_search_filter:
            # Clear button
            op_clear_search = row_search.operator("object.clear_vgroup_search", text="", icon='X')
        
        layout.separator()

        # --- Filtered Groups List ---
        # Apply search filter to mixed_filtered_groups
        search_term = scene.vg_search_filter.strip().lower() if scene.vg_search_filter else ""
        filtered_by_search = mixed_filtered_groups
        if search_term:
            filtered_by_search = []
            for entry in mixed_filtered_groups:
                obj_name, gi, weight_val = entry
                obj = bpy.data.objects.get(obj_name)
                if obj and 0 <= gi < len(obj.vertex_groups):
                    group_name = obj.vertex_groups[gi].name.lower()
                    if search_term in group_name:
                        filtered_by_search.append(entry)
        
        total = len(filtered_by_search)
        thresh_val = scene.vg_weight_threshold
        method_str = "Avg" if scene.vg_use_average_weight else "Max"
        if search_term:
            layout.label(text=f"{total} Groups ({method_str} >= {thresh_val:.3f}, filtered: '{scene.vg_search_filter}')")
        else:
            layout.label(text=f"{total} Groups ({method_str} >= {thresh_val:.3f})")

        if filtered_by_search:
            # Group entries by object
            grouped_by_object = {}
            for entry in filtered_by_search:
                obj_name, gi, weight_val = entry
                grouped_by_object.setdefault(obj_name, []).append((gi, weight_val))
            # Create a row to hold columns for each object
            obj_row = layout.row(align=True)
            for obj_name, entries in grouped_by_object.items():
                obj = bpy.data.objects.get(obj_name)
                if not obj:
                    continue
                # Create a column for this object
                obj_col = obj_row.column(align=True)
                # Use a box inside the column for better separation
                obj_box = obj_col.box()
                col_obj = obj_box.column(align=True)
                row_obj_header = col_obj.row(align=True)
                row_obj_header.alignment = 'LEFT'
                row_obj_header.label(text=f"{obj.name}", icon='OBJECT_DATA')
                row_obj_header.label(text=f"({len(entries)}/{len(obj.vertex_groups)})")
                # Sort entries by group name
                entries.sort(key=lambda x: obj.vertex_groups[x[0]].name if 0 <= x[0] < len(obj.vertex_groups) else "")
                for entry in entries:
                    gi, weight_val = entry
                    if not (0 <= gi < len(obj.vertex_groups)):
                        continue
                    vg = obj.vertex_groups[gi]
                    if not scene.show_locked_vgroups and vg.lock_weight:
                        continue
                    # Row for this vertex group's controls
                    row_vg = col_obj.row()
                    split = row_vg.split(factor=0.65)
                    col_name = split.column()
                    current_group_tuple = (obj_name, gi)
                    is_selected = current_group_tuple in selected_vgroups
                    is_selected_for_rename = (selected_vgroup_for_rename == current_group_tuple)
                    is_renaming_this_group = is_selected_for_rename and scene.vg_rename_mode
                    if is_renaming_this_group:
                        col_name.prop(scene, "vg_editing_name", text="")
                    else:
                        op = col_name.operator("mesh.vgroup_name_click", text=vg.name, depress=is_selected)
                        op.group_obj = obj_name
                        op.group_idx = gi
                    col_buttons = split.column(align=True)
                    row_buttons_inner = col_buttons.row(align=True)
                    if is_renaming_this_group:
                        # Replace lock button with confirm button during rename
                        op_confirm = row_buttons_inner.operator("mesh.confirm_vgroup_rename", text="", icon='CHECKMARK')
                    else:
                        # Normal lock button
                        lock_icon = 'LOCKED' if vg.lock_weight else 'UNLOCKED'
                        op_lock = row_buttons_inner.operator("mesh.toggle_vgroup_lock", text="", icon=lock_icon, depress=vg.lock_weight)
                        op_lock.group_obj = obj_name
                        op_lock.group_idx = gi
                    op_sel = row_buttons_inner.operator("mesh.select_vgroup_verts", text="", icon='RESTRICT_SELECT_OFF')
                    op_sel.group_obj = obj_name
                    op_sel.group_idx = gi
                    merge_icon = 'AUTOMERGE_OFF'
                    depress_merge = False
                    if active_merge_group == (obj_name, gi):
                        merge_icon = 'AUTOMERGE_ON'
                        depress_merge = True
                    elif active_merge_group is not None and active_merge_group[0] == obj_name:
                        merge_icon = 'AUTOMERGE_ON'
                    op_merge = row_buttons_inner.operator("mesh.merge_vgroup", text="", icon=merge_icon, depress=depress_merge)
                    op_merge.group_obj = obj_name
                    op_merge.group_idx = gi
                    row_buttons_inner.label(text=f"{weight_val:.2f}")
            layout.separator()
        
        # --- Info Block: Rename and Conflict Detection ---
        # Only show when exactly 2 groups from 2 different objects are selected
        if len(selected_vgroups) == 2:
            # Get selected groups sorted by object name (to determine left/right)
            selected_list = sorted(selected_vgroups, key=lambda x: x[0])
            left_obj_name, left_group_idx = selected_list[0]
            right_obj_name, right_group_idx = selected_list[1]
            
            # Only show if groups are from different objects
            if left_obj_name != right_obj_name:
                left_obj = bpy.data.objects.get(left_obj_name)
                right_obj = bpy.data.objects.get(right_obj_name)
                
                if left_obj and right_obj and \
                   (0 <= left_group_idx < len(left_obj.vertex_groups)) and \
                   (0 <= right_group_idx < len(right_obj.vertex_groups)):
                    
                    info_box = layout.box()
                    info_box.label(text="Rename Groups", icon='INFO')
                    
                    left_group = left_obj.vertex_groups[left_group_idx]
                    right_group = right_obj.vertex_groups[right_group_idx]
                    left_name = left_group.name
                    right_name = right_group.name
                    
                    # Rename buttons row
                    row_rename = info_box.row(align=True)
                    op_rename_ltr = row_rename.operator("mesh.rename_vgroup_left_to_right", text="← Rename Left to Right")
                    op_rename_rtl = row_rename.operator("mesh.rename_vgroup_right_to_left", text="Rename Right to Left →")
                    
                    # Conflict detection - Two column layout (one line each)
                    info_box.separator()
                    left_name_in_right = any(vg.name == left_name for vg in right_obj.vertex_groups if vg.index != right_group_idx)
                    right_name_in_left = any(vg.name == right_name for vg in left_obj.vertex_groups if vg.index != left_group_idx)
                    
                    # Create two columns with split - one line each
                    row_status = info_box.row(align=True)
                    split_status = row_status.split(factor=0.5)
                    col_left = split_status.column(align=True)
                    col_right = split_status.column(align=True)
                    
                    # Left column - one line: name and status icon
                    if left_name_in_right:
                        col_left.label(text=f"'{left_name}' exists on right", icon='ERROR')
                    else:
                        col_left.label(text=f"'{left_name}'", icon='CHECKMARK')
                    
                    # Right column - one line: name and status icon
                    if right_name_in_left:
                        col_right.label(text=f"'{right_name}' exists on left", icon='ERROR')
                    else:
                        col_right.label(text=f"'{right_name}'", icon='CHECKMARK')

# ——————————————————————————
# Handlers and Timer
# ——————————————————————————
def timer_wheel_modal_check():
    """ Timer function to ensure wheel scroll modal is running when needed. """
    try:
        context = bpy.context
        if MESH_OT_vgroup_wheel_scroll.poll(context) and not MESH_OT_vgroup_wheel_scroll._running:
            # Directly invoke from timer (timers run in safe context)
            try:
                bpy.ops.mesh.vgroup_wheel_scroll('INVOKE_DEFAULT')
            except Exception:
                pass  # Modal might already be starting
    except (AttributeError, RuntimeError):
        pass
    # Check every 1 second
    return 1.0

def handler_mode_change(scene):
    """ Handler to ensure wheel scroll modal is running when in edit mode. """
    try:
        context = bpy.context
        if MESH_OT_vgroup_wheel_scroll.poll(context) and not MESH_OT_vgroup_wheel_scroll._running:
            # Use timer to defer - handlers might also be called during draw
            def deferred_start():
                try:
                    if MESH_OT_vgroup_wheel_scroll.poll(bpy.context) and not MESH_OT_vgroup_wheel_scroll._running:
                        bpy.ops.mesh.vgroup_wheel_scroll('INVOKE_DEFAULT')
                except Exception:
                    pass
                return None  # Run once
            if not bpy.app.timers.is_registered(deferred_start):
                bpy.app.timers.register(deferred_start, first_interval=0.05)
    except (AttributeError, RuntimeError):
        pass

def handler_load_post(dummy):
    """ Handler to ensure timers start after Blender loads. """
    global timer_running, live_update_enabled
    try:
        # Ensure live update timer is running
        if not bpy.app.timers.is_registered(timer_update):
            timer_running = True
            bpy.app.timers.register(timer_update, first_interval=0.5)
        
        # Ensure wheel scroll modal timer is running
        if not bpy.app.timers.is_registered(timer_wheel_modal_check):
            bpy.app.timers.register(timer_wheel_modal_check, first_interval=0.5)
        
        # Try to start modal if in edit mode - use deferred start
        def start_modal_after_load():
            try:
                context = bpy.context
                if MESH_OT_vgroup_wheel_scroll.poll(context):
                    bpy.ops.mesh.vgroup_wheel_scroll('INVOKE_DEFAULT')
            except Exception:
                pass
            return None  # Run once
        
        # Register deferred start
        if not bpy.app.timers.is_registered(start_modal_after_load):
            bpy.app.timers.register(start_modal_after_load, first_interval=0.1)
        
        # Sync live_update_enabled with scene property
        try:
            if hasattr(bpy.context, 'scene') and bpy.context.scene:
                live_update_enabled = bpy.context.scene.vg_live_update
        except (AttributeError, RuntimeError):
            pass
    except Exception:
        pass

def timer_update():
    """ Periodic check for selection changes if Live Update is ON. """
    global timer_running, live_update_enabled, _last_update_time, prev_selected_indices
    # Always return a delay to keep timer running, even if not active
    if not timer_running: 
        return _UPDATE_DELAY
    context = bpy.context
    if not (context.mode == 'EDIT_MESH' and context.active_object and context.active_object.type == 'MESH'):
         return _UPDATE_DELAY
    # Safely check scene property and sync global flag
    try:
        if not hasattr(context, 'scene') or not context.scene:
            return _UPDATE_DELAY
        # Sync global flag with scene property
        live_update_enabled = context.scene.vg_live_update
        if not live_update_enabled:
            return _UPDATE_DELAY
    except (AttributeError, RuntimeError):
        return _UPDATE_DELAY
    now = time.time()
    if now - _last_update_time < _UPDATE_DELAY:
        return _UPDATE_DELAY
    update_needed = False
    current_selection_snapshot = {}
    edit_objects_found = False
    try:
        for obj in context.selected_editable_objects:
            if obj.type == 'MESH' and obj.mode == 'EDIT':
                edit_objects_found = True
                try:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bm.verts.ensure_lookup_table()
                    sel = tuple(sorted(v.index for v in bm.verts if v.select))
                    current_selection_snapshot[obj.name] = sel
                    if sel != prev_selected_indices.get(obj.name): update_needed = True
                except Exception:
                    if obj.name in prev_selected_indices:
                        update_needed = True
                        if obj.name in current_selection_snapshot: del current_selection_snapshot[obj.name]
                    continue
    except ReferenceError: return _UPDATE_DELAY
    if set(current_selection_snapshot.keys()) != set(prev_selected_indices.keys()): update_needed = True
    if not edit_objects_found and prev_selected_indices:
         prev_selected_indices = {}; update_needed = True
    if update_needed:
        _last_update_time = now
        prev_selected_indices = current_selection_snapshot.copy()
        bpy.ops.object.update_filtered_vgroups('INVOKE_DEFAULT')
    return _UPDATE_DELAY

# ——————————————————————————
# Registration
# ——————————————————————————
classes = [
    OBJECT_OT_update_filtered_vgroups,
    OBJECT_OT_reset_filtered_vgroups,
    OBJECT_OT_show_zero_weight_info,
    MESH_OT_select_vgroup_verts,
    OBJECT_OT_toggle_live_update,
    OBJECT_OT_lock_matching_vgroups,
    OBJECT_OT_toggle_locked_vgroups_filter,
    OBJECT_OT_clear_vgroup_search,
    MESH_OT_vgroup_wheel_scroll,
    MESH_OT_vgroup_name_click,
    MESH_OT_confirm_vgroup_rename,
    MESH_OT_cancel_vgroup_rename,
    MESH_OT_rename_vgroup_modal,
    MESH_OT_toggle_vgroup_lock,
    MESH_OT_rename_vgroup_left_to_right,
    MESH_OT_rename_vgroup_right_to_left,
    MESH_OT_merge_vgroup,
    VIEW3D_PT_filtered_vertex_groups,
]

def register():
    global timer_running, live_update_enabled, _last_update_time, _previous_mode
    global prev_selected_indices, mixed_filtered_groups, active_merge_group, selected_vgroup_for_rename, selected_vgroups
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.vg_live_update = bpy.props.BoolProperty(name="Live Update Filter", default=True, description="Automatically update list on selection changes")
    bpy.types.Scene.show_locked_vgroups = bpy.props.BoolProperty(name="Show Locked Groups", default=True, description="Display locked vertex groups in the list")
    bpy.types.Scene.vg_use_average_weight = bpy.props.BoolProperty(name="Use Average Weight", default=False, description="Filter by average weight (else max weight)", update=update_filter_settings)
    bpy.types.Scene.vg_weight_threshold = bpy.props.FloatProperty(name="Weight Threshold", default=0.001, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="Minimum effective weight for a group to be shown", update=update_filter_settings)
    bpy.types.Scene.vg_rename_mode = bpy.props.BoolProperty(name="Rename Mode Active", default=False, description="Internal flag for UI rename state")
    bpy.types.Scene.vg_editing_name = bpy.props.StringProperty(name="Editing Group Name", default="", description="Temporary storage for group name edits", update=update_vg_name)
    bpy.types.Scene.vg_search_filter = bpy.props.StringProperty(name="Search Filter", default="", description="Filter vertex groups by name (case-insensitive)", update=update_search_filter)
    # Safely get live_update_enabled from context if available, otherwise use default
    try:
        if hasattr(bpy.context, 'scene') and bpy.context.scene:
            live_update_enabled = bpy.context.scene.vg_live_update
        else:
            live_update_enabled = True  # Default value
    except (AttributeError, RuntimeError):
        live_update_enabled = True  # Default value
    _last_update_time = 0.0
    _previous_mode = None
    prev_selected_indices = {}
    mixed_filtered_groups = []
    active_merge_group = None
    selected_vgroup_for_rename = None
    selected_vgroups = set()
    # Register handlers first
    if handler_mode_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(handler_mode_change)
    if handler_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(handler_load_post)
    
    # Start timers immediately - they will keep running
    timer_running = True
    if not bpy.app.timers.is_registered(timer_update):
        bpy.app.timers.register(timer_update, first_interval=0.5)
    if not bpy.app.timers.is_registered(timer_wheel_modal_check):
        bpy.app.timers.register(timer_wheel_modal_check, first_interval=0.5)
    
    # Try to start modal if already in edit mode
    try:
        context = bpy.context
        if MESH_OT_vgroup_wheel_scroll.poll(context):
            MESH_OT_vgroup_wheel_scroll.ensure_running(context)
    except Exception:
        pass  # Context might not be ready yet
    
    # Call load_post handler immediately to ensure initialization
    try:
        handler_load_post(None)
    except Exception:
        pass
    print("Filtered Vertex Groups Add-on: Registered")

def unregister():
    global timer_running
    if bpy.app.timers.is_registered(timer_update): bpy.app.timers.unregister(timer_update)
    if bpy.app.timers.is_registered(timer_wheel_modal_check): bpy.app.timers.unregister(timer_wheel_modal_check)
    # Remove handlers
    if handler_mode_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(handler_mode_change)
    if handler_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(handler_load_post)
    timer_running = False
    # Stop the wheel scroll modal if running
    MESH_OT_vgroup_wheel_scroll._running = False
    for cls in reversed(classes):
        try: bpy.utils.unregister_class(cls)
        except RuntimeError: pass
    props_to_delete = ["vg_live_update", "show_locked_vgroups", "vg_use_average_weight", "vg_weight_threshold", "vg_rename_mode", "vg_editing_name", "vg_search_filter"]
    for prop in props_to_delete:
         try: delattr(bpy.types.Scene, prop)
         except AttributeError: pass
    print("Filtered Vertex Groups Add-on: Unregistered")

if __name__ == "__main__":
    # try: unregister() except Exception: pass # Optional force unregister
    register()
