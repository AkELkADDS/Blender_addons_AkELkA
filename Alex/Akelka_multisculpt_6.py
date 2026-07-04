bl_info = {
    "name": "Akelka Multisculpt",
    "author": "Akelka",
    "version": (2, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > N-Panel > Akelka Multisculpt",
    "description": "Seamlessly sculpt, paint and weight multiple objects at once",
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "Sculpting",
}

import bpy
from bpy import context
from bpy.app.handlers import persistent
from mathutils import Matrix
import time

addon_keymaps = []
creating_multisculpt = False
last_mode = None
_services_initialized = False


def deferred_mode_check():
    """Check for mode changes and trigger operations safely"""
    global last_mode, creating_multisculpt
    
    try:
        context = bpy.context
        if not context or not context.object:
            return 0.1
        
        current_mode = context.object.mode
        
        if last_mode != current_mode:
            selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
            
            # Auto-create when entering SCULPT with 2+ objects
            if current_mode == 'SCULPT' and len(selected_objects) >= 2 and not creating_multisculpt:
                creating_multisculpt = True
                try:
                    bpy.ops.multisculpt.create()
                    bpy.ops.object.mode_set(mode='SCULPT')
                except:
                    pass
                finally:
                    creating_multisculpt = False
            
            # Auto-create when entering WEIGHT_PAINT or VERTEX_PAINT with 2+ objects
            elif current_mode in ('WEIGHT_PAINT', 'VERTEX_PAINT') and len(selected_objects) >= 2 and not creating_multisculpt:
                creating_multisculpt = True
                try:
                    bpy.ops.multisculpt.create()
                    bpy.ops.object.mode_set(mode=current_mode)
                except:
                    pass
                finally:
                    creating_multisculpt = False
            
            # Auto-transfer when going to OBJECT from any paint/sculpt mode
            elif current_mode == 'OBJECT' and last_mode in ('SCULPT', 'WEIGHT_PAINT', 'VERTEX_PAINT', 'TEXTURE_PAINT') and not creating_multisculpt:
                proxy_obj = context.object
                if not proxy_obj or "multisculpt_source_objects" not in proxy_obj:
                    proxy_objs = [obj for obj in context.selected_objects if "multisculpt_source_objects" in obj]
                    proxy_obj = proxy_objs[0] if proxy_objs else None
                if proxy_obj:
                    creating_multisculpt = True
                    try:
                        context.view_layer.objects.active = proxy_obj
                        bpy.ops.multisculpt.transfer()
                    except:
                        pass
                    finally:
                        creating_multisculpt = False
            
            last_mode = current_mode
    except:
        pass
    
    return 0.2  # Check every 200ms


class MultiSculptProperties(bpy.types.PropertyGroup):
    """Properties for Akelka Multisculpt addon"""
    keep_common_modifiers: bpy.props.BoolProperty(
        name="Keep Common Modifiers",
        description="Propagate modifiers that are present on all objects to the proxy",
        default=False
    )
    use_multires_level: bpy.props.BoolProperty(
        name="Use Current Multires Level",
        description="Use current Sculpt Mode Multires level instead of base geometry",
        default=False
    )
    transfer_color: bpy.props.BoolProperty(
        name="Transfer Color",
        description="Transfer vertex colors during multisculpt transfer",
        default=True
    )
    transfer_uv: bpy.props.BoolProperty(
        name="Transfer UV Maps (Forward Only)",
        description="Unify UV maps during proxy creation (for painting)",
        default=True
    )
    transfer_vertex_groups: bpy.props.BoolProperty(
        name="Transfer Vertex Groups",
        description="Transfer vertex groups during multisculpt transfer",
        default=True
    )
    auto_clear_instancing: bpy.props.BoolProperty(
        name="Auto Clear Instancing",
        description="Automatically make original object datablocks single-user",
        default=True
    )
    shape_key_transfer_mode: bpy.props.EnumProperty(
        name="Shape Key Transfer Mode",
        description="How to transfer deformations from shape keys",
        items=[
            ("BASIS", "To Basis", "Transfer to basis shape key"),
            ("ACTIVE", "To Active", "Transfer to active shape key"),
            ("NEW", "To New", "Create new shape key with transfer"),
        ],
        default="NEW"
    )
    compensate_shape_keys: bpy.props.BoolProperty(
        name="Compensate Shape Keys",
        description="Account for deformations from existing active shape keys",
        default=False
    )
    pie_menu_variant: bpy.props.EnumProperty(
        name="Pie Menu Variant",
        description="Choose pie menu behavior",
        items=[
            ("AUTO_FAST", "Auto Fast", "Fast mode switching (may lose undo)"),
            ("AUTO", "Auto", "Preserves undo steps"),
            ("MANUAL", "Manual", "Manual mode control"),
        ],
        default="AUTO_FAST"
    )


class MULTISCULPT_OT_CreateMultisculpt(bpy.types.Operator):
    """Create a multisculpt proxy object from selected meshes"""
    bl_idname = "multisculpt.create"
    bl_label = "Create Multisculpt"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        if len(selected_objects) < 2:
            self.report({'ERROR'}, "Select at least 2 mesh objects")
            return {'CANCELLED'}

        props = scene.multisculpt_props

        try:
            start_time = time.time()
            print(f"\n=== MULTISCULPT CREATE START ===")
            
            # Ensure we're in Object mode first
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            
            for obj in selected_objects:
                obj.select_set(True)
            
            context.view_layer.objects.active = selected_objects[0]
            
            stored_color_data = []
            stored_uv_data = []
            stored_vg_data = []
            
            print(f"[PROFILE] Data collection start...")
            t1 = time.time()
            for obj in selected_objects:
                # Only collect colors if transfer enabled
                if props.transfer_color:
                    t_obj_start = time.time()
                    active_color = None
                    attrs = obj.data.color_attributes
                    idx = attrs.active_color_index
                    
                    t_get_start = time.time()
                    if 0 <= idx < len(attrs):
                        active_color = attrs[idx]
                    t_get_end = time.time()
                    
                    if active_color:
                        # Use foreach_get for speed instead of loop
                        t_col_start = time.time()
                        color_count = len(active_color.data)
                        colors_flat = [0] * (color_count * 4)  # RGBA = 4 values per color
                        active_color.data.foreach_get('color', colors_flat)
                        # SKIP tuple conversion - keep as flat list!
                        color_data = {
                            'name': active_color.name,
                            'domain': active_color.domain,
                            'type': 'FLOAT_COLOR',
                            'data': colors_flat  # Store flat!
                        }
                        stored_color_data.append(color_data)
                        t_col_end = time.time()
                        print(f"Object {obj.name}: Color {color_count} verts - total:{t_col_end-t_obj_start:.4f}s")
                    else:
                        stored_color_data.append(None)
                else:
                    stored_color_data.append(None)
                
                # Only collect UV if transfer enabled
                if props.transfer_uv:
                    active_uv = obj.data.uv_layers.active
                    if active_uv:
                        stored_uv_data.append(active_uv.name)
                        print(f"Object {obj.name}: Using UV map '{active_uv.name}'")
                    else:
                        stored_uv_data.append(None)
                        print(f"Object {obj.name}: No UV maps found")
                
                # Only collect vertex groups if transfer enabled
                if props.transfer_vertex_groups:
                    active_vg = None
                    if obj.vertex_groups and len(obj.vertex_groups) > 0:
                        active_vg = obj.vertex_groups.active
                    
                    if active_vg:
                        stored_vg_data.append(active_vg.name)
                        print(f"Object {obj.name}: Using vertex group '{active_vg.name}'")
                    else:
                        stored_vg_data.append(None)
                        print(f"Object {obj.name}: No vertex groups found")
            
            for obj in selected_objects:
                if props.transfer_uv and stored_uv_data[selected_objects.index(obj)]:
                    uv_layer = obj.data.uv_layers.active
                    uv_layer.name = "Multisculpt_UV_Temp"
                
                if props.transfer_vertex_groups and stored_vg_data[selected_objects.index(obj)]:
                    vg = obj.vertex_groups.active
                    vg.name = "Multisculpt_VG_Temp"
            
            t2 = time.time()
            print(f"[PROFILE] Data collection: {t2-t1:.4f}s")
            
            bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
            
            t3 = time.time()
            print(f"[PROFILE] Duplicate operation: {t3-t2:.4f}s")
            
            for obj in selected_objects:
                if props.transfer_uv and stored_uv_data[selected_objects.index(obj)]:
                    uv_layer = obj.data.uv_layers["Multisculpt_UV_Temp"]
                    uv_layer.name = stored_uv_data[selected_objects.index(obj)]
                
                if props.transfer_vertex_groups and stored_vg_data[selected_objects.index(obj)]:
                    vg = obj.vertex_groups["Multisculpt_VG_Temp"]
                    vg.name = stored_vg_data[selected_objects.index(obj)]
            
            duplicated_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
            
            if len(duplicated_objects) > 1:
                context.view_layer.objects.active = duplicated_objects[0]
                for obj in duplicated_objects:
                    obj.select_set(True)
                bpy.ops.object.join()
            
            t4 = time.time()
            print(f"[PROFILE] Join objects: {t4-t3:.4f}s")
            
            proxy_obj = context.active_object
            proxy_obj.name = "MultiSculpt_Proxy"
            
            for mod in proxy_obj.modifiers:
                proxy_obj.modifiers.remove(mod)
            
            for attr in list(proxy_obj.data.color_attributes):
                try:
                    proxy_obj.data.color_attributes.remove(attr)
                except:
                    pass
            
            t_hide_start = time.time()
            for obj in selected_objects:
                obj.hide_set(True)
                obj.hide_render = True
            t_hide_end = time.time()
            print(f"[PROFILE]   Hide objects: {t_hide_end-t_hide_start:.4f}s")

            if props.auto_clear_instancing:
                t_inst_start = time.time()
                for obj in selected_objects:
                    if obj.data.users > 1:
                        obj.data = obj.data.copy()
                t_inst_end = time.time()
                print(f"[PROFILE]   Clear instancing: {t_inst_end-t_inst_start:.4f}s")

            t_ranges_start = time.time()
            vert_ranges = []
            vert_offset = 0
            stored_matrices = []
            
            for obj in selected_objects:
                vert_count = len(obj.data.vertices)
                vert_ranges.append((vert_offset, vert_offset + vert_count))
                stored_matrices.append(list(obj.matrix_world[:]))
                vert_offset += vert_count
            t_ranges_end = time.time()
            print(f"[PROFILE]   Build ranges: {t_ranges_end-t_ranges_start:.4f}s")

            t_props_start = time.time()
            proxy_obj["multisculpt_source_objects"] = [obj.name for obj in selected_objects]
            proxy_obj["multisculpt_vert_ranges"] = vert_ranges
            proxy_obj["multisculpt_matrices"] = stored_matrices
            proxy_obj["multisculpt_proxy_matrix"] = list(proxy_obj.matrix_world[:])
            # REMOVED: color_data storage - we use proxy mesh colors directly during transfer
            proxy_obj["multisculpt_uv_data"] = stored_uv_data
            proxy_obj["multisculpt_vg_data"] = stored_vg_data
            t_props_end = time.time()
            print(f"[PROFILE]   Store properties: {t_props_end-t_props_start:.4f}s")
            
            t5 = time.time()
            print(f"[PROFILE] Setup metadata total: {t5-t4:.4f}s")
            
            if props.transfer_color and any(stored_color_data):
                unified_attr = proxy_obj.data.color_attributes.new(
                    name="Multisculpt_Color",
                    type='FLOAT_COLOR',
                    domain='POINT'
                )
                unified_attr_data = unified_attr.data
                
                # Use foreach_set for massive speedup - colors already flat!
                all_colors_flat = []
                for obj_idx, color_info in enumerate(stored_color_data):
                    if color_info:
                        # Already flat, just extend
                        all_colors_flat.extend(color_info['data'])
                
                # Bulk assign with foreach_set - only assign the exact number of colors we have
                if all_colors_flat and len(all_colors_flat) == len(unified_attr_data) * 4:
                    # Only proceed if sizes match exactly
                    unified_attr_data.foreach_set('color', all_colors_flat)
                elif all_colors_flat:
                    # Fallback: assign colors one by one if size doesn't match
                    color_idx = 0
                    for i in range(len(unified_attr_data)):
                        if color_idx + 3 < len(all_colors_flat):
                            unified_attr_data[i].color = tuple(all_colors_flat[color_idx:color_idx+4])
                            color_idx += 4
                
                proxy_obj.data.color_attributes.active = unified_attr
                
                to_delete = [attr.name for attr in proxy_obj.data.color_attributes if attr.name != "Multisculpt_Color"]
                for attr_name in to_delete:
                    attr = proxy_obj.data.color_attributes.get(attr_name)
                    if attr:
                        try:
                            proxy_obj.data.color_attributes.remove(attr)
                        except:
                            pass
            
            if proxy_obj.data.uv_layers:
                for uv in proxy_obj.data.uv_layers:
                    if uv.name == "Multisculpt_UV_Temp":
                        uv.name = "Multisculpt_UV"
            
            if proxy_obj.vertex_groups:
                for vg in proxy_obj.vertex_groups:
                    if vg.name == "Multisculpt_VG_Temp":
                        vg.name = "Multisculpt_VG"
                
                if "Multisculpt_VG" in proxy_obj.vertex_groups:
                    proxy_obj.vertex_groups.active = proxy_obj.vertex_groups["Multisculpt_VG"]

            bpy.context.view_layer.objects.active = proxy_obj
            proxy_obj.select_set(True)

            self.report({'INFO'}, f"Created Multisculpt proxy from {len(selected_objects)} objects")
            
            # Don't set mode here - let the caller decide what mode to use
            # bpy.ops.object.mode_set(mode='SCULPT')
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"[PROFILE] Total multisculpt creation time: {total_time:.4f}s")
            print(f"=== MULTISCULPT CREATE END ===\n")
            
            return {'FINISHED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create multisculpt: {str(e)}")
            return {'CANCELLED'}


class MULTISCULPT_OT_TransferMultisculpt(bpy.types.Operator):
    """Transfer changes from proxy back to original objects"""
    bl_idname = "multisculpt.transfer"
    bl_label = "Transfer Multisculpt"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            start_time = time.time()
            print(f"\n=== MULTISCULPT TRANSFER START ===")
            
            proxy_obj = context.active_object
            
            if not proxy_obj or proxy_obj.type != 'MESH':
                self.report({'ERROR'}, "Select the Multisculpt proxy object")
                return {'CANCELLED'}

            if "multisculpt_source_objects" not in proxy_obj:
                self.report({'ERROR'}, "This is not a valid Multisculpt proxy object")
                return {'CANCELLED'}

            source_names = proxy_obj["multisculpt_source_objects"]
            vert_ranges = proxy_obj.get("multisculpt_vert_ranges", [])
            
            source_objects = [bpy.data.objects[name] for name in source_names if name in bpy.data.objects]
            props = context.scene.multisculpt_props

            if not source_objects:
                self.report({'ERROR'}, "Source objects not found")
                return {'CANCELLED'}

            t1 = time.time()
            self._transfer_geometry(proxy_obj, source_objects, vert_ranges)
            t2 = time.time()
            print(f"[PROFILE] Transfer geometry: {t2-t1:.4f}s")
            
            if props.transfer_color:
                self._transfer_colors(proxy_obj, source_objects, props, vert_ranges)
            t3 = time.time()
            if props.transfer_color:
                print(f"[PROFILE] Transfer colors: {t3-t2:.4f}s")
            
            # NOTE: UV maps don't need to be transferred back - originals keep their UVs!
            # Only vertex groups if needed
            if props.transfer_vertex_groups:
                self._transfer_vertex_groups_back(proxy_obj, source_objects, vert_ranges)
            t4 = time.time()
            if props.transfer_vertex_groups:
                print(f"[PROFILE] Transfer vertex groups: {t4-t3:.4f}s")
            
            self._apply_shape_key_changes(source_objects)
            t5 = time.time()
            print(f"[PROFILE] Apply shape keys: {t5-t4:.4f}s")

            for obj in source_objects:
                obj.hide_set(False)
                obj.hide_render = False
            
            stored_uv_data = proxy_obj.get("multisculpt_uv_data", [])
            stored_vg_data = proxy_obj.get("multisculpt_vg_data", [])
            
            for idx, obj in enumerate(source_objects):
                if idx < len(stored_uv_data) and stored_uv_data[idx]:
                    original_uv_name = stored_uv_data[idx]
                    if "Multisculpt_UV_Temp" in obj.data.uv_layers:
                        obj.data.uv_layers["Multisculpt_UV_Temp"].name = original_uv_name
                
                if idx < len(stored_vg_data) and stored_vg_data[idx]:
                    original_vg_name = stored_vg_data[idx]
                    
                    if "Multisculpt_VG_Temp" in obj.vertex_groups:
                        obj.vertex_groups.remove(obj.vertex_groups["Multisculpt_VG_Temp"])
                    
                    if "Multisculpt_VG" in obj.vertex_groups:
                        obj.vertex_groups.remove(obj.vertex_groups["Multisculpt_VG"])
            
            proxy_mesh = proxy_obj.data
            bpy.data.objects.remove(proxy_obj)
            bpy.data.meshes.remove(proxy_mesh)
            
            for obj in source_objects:
                obj.select_set(True)
            
            if source_objects:
                context.view_layer.objects.active = source_objects[0]

            end_time = time.time()
            total_time = end_time - start_time
            print(f"[PROFILE] Total transfer time: {total_time:.4f}s")
            print(f"=== MULTISCULPT TRANSFER END ===\n")
            
            self.report({'INFO'}, f"Transferred changes to {len(source_objects)} objects")
            return {'FINISHED'}
        
        except Exception as e:
            self.report({'ERROR'}, f"Transfer failed: {str(e)}")
            return {'CANCELLED'}

    def _transfer_geometry(self, proxy_obj, source_objects, vert_ranges):
        """Transfer mesh geometry - SUPER OPTIMIZED using foreach_set"""
        proxy_mesh = proxy_obj.data
        proxy_vertices = proxy_mesh.vertices
        stored_matrices = proxy_obj.get("multisculpt_matrices", [])
        proxy_matrix = Matrix(proxy_obj.get("multisculpt_proxy_matrix", proxy_obj.matrix_world))
        
        for idx, source_obj in enumerate(source_objects):
            if idx >= len(vert_ranges):
                continue
            
            start_vert, end_vert = vert_ranges[idx]
            source_mesh = source_obj.data
            source_verts_count = len(source_mesh.vertices)
            
            if idx < len(stored_matrices):
                transform = source_obj.matrix_world.inverted() @ proxy_matrix
            else:
                transform = Matrix.Identity(4)
            
            # Use foreach_set for massive speedup - collect flat list of coordinates
            coords_flat = []
            for proxy_idx in range(start_vert, min(end_vert, len(proxy_vertices))):
                transformed_co = transform @ proxy_vertices[proxy_idx].co
                coords_flat.extend(transformed_co)
            
            # Use foreach_set for bulk assignment (10x faster than loop)
            if coords_flat:
                source_mesh.vertices.foreach_set('co', coords_flat)
            
            source_mesh.update()

    def _transfer_colors(self, proxy_obj, source_objects, props, vert_ranges):
        """Transfer unified color attribute - SUPER OPTIMIZED with foreach_set"""
        if not props.transfer_color:
            return

        stored_color_data = proxy_obj.get("multisculpt_color_data", [])
        proxy_color_layers = proxy_obj.data.color_attributes

        if "Multisculpt_Color" not in proxy_color_layers:
            return

        unified_attr = proxy_color_layers["Multisculpt_Color"]
        unified_attr_data = unified_attr.data

        for idx, source_obj in enumerate(source_objects):
            if idx >= len(vert_ranges):
                continue
            
            if idx >= len(stored_color_data) or not stored_color_data[idx]:
                continue
            
            original_color_info = stored_color_data[idx]
            original_name = original_color_info['name']
            domain = original_color_info['domain']
            color_type = original_color_info.get('type', 'FLOAT_COLOR')
            
            try:
                if original_name not in source_obj.data.color_attributes:
                    source_obj.data.color_attributes.new(
                        name=original_name,
                        type=color_type,
                        domain=domain
                    )
                
                target_layer = source_obj.data.color_attributes.get(original_name)
                if not target_layer:
                    continue
                
                start_vert, end_vert = vert_ranges[idx]
                target_layer_data = target_layer.data
                
                # Batch collect colors as flat RGBA list
                colors_flat = []
                for proxy_idx in range(start_vert, end_vert):
                    if proxy_idx < len(unified_attr_data):
                        color = unified_attr_data[proxy_idx].color
                        colors_flat.extend(color)
                
                # Use foreach_set for bulk assignment
                if colors_flat:
                    target_layer_data.foreach_set('color', colors_flat)
                
                source_obj.data.color_attributes.active = target_layer
            
            except Exception as e:
                pass

    def _transfer_vertex_groups(self, proxy_obj, source_objects, vert_ranges):
        """Transfer vertex group data from proxy to source objects using vertex ranges"""
        for group_idx, group in enumerate(proxy_obj.vertex_groups):
            for obj_idx, source_obj in enumerate(source_objects):
                if obj_idx >= len(vert_ranges):
                    continue

                if group.name not in source_obj.vertex_groups:
                    source_obj.vertex_groups.new(name=group.name)

                target_group = source_obj.vertex_groups[group.name]
                start_vert, end_vert = vert_ranges[obj_idx]
                proxy_vert_idx = start_vert
                
                for source_vert_idx in range(len(source_obj.data.vertices)):
                    if proxy_vert_idx < end_vert and proxy_vert_idx < len(proxy_obj.data.vertices):
                        try:
                            weight = group.weight(proxy_vert_idx)
                            target_group.add([source_vert_idx], weight, 'REPLACE')
                        except RuntimeError:
                            pass
                        proxy_vert_idx += 1

    def _transfer_uvs(self, proxy_obj, source_objects, vert_ranges):
        """Transfer unified UV map data back to source objects with original UV names - OPTIMIZED"""
        stored_uv_data = proxy_obj.get("multisculpt_uv_data", [])
        
        if "Multisculpt_UV" not in proxy_obj.data.uv_layers:
            return
        
        proxy_uv = proxy_obj.data.uv_layers["Multisculpt_UV"]
        proxy_uv_data = proxy_uv.data
        
        proxy_loop_offset = 0
        
        for idx, source_obj in enumerate(source_objects):
            if idx >= len(stored_uv_data) or not stored_uv_data[idx]:
                continue
            
            original_name = stored_uv_data[idx]
            source_loop_count = len(source_obj.data.loops)
            
            try:
                if original_name in source_obj.data.uv_layers:
                    source_obj.data.uv_layers.remove(source_obj.data.uv_layers[original_name])
                
                target_uv = source_obj.data.uv_layers.new(name=original_name)
                source_obj.data.uv_layers.active = target_uv
                target_uv_data = target_uv.data
                
                # Batch collect all UV values first
                uv_values = []
                for i in range(source_loop_count):
                    proxy_idx = proxy_loop_offset + i
                    if proxy_idx < len(proxy_uv_data):
                        uv_values.append(proxy_uv_data[proxy_idx].uv)
                
                # Then batch assign them
                for source_loop_idx, uv in enumerate(uv_values):
                    target_uv_data[source_loop_idx].uv = uv
                
                proxy_loop_offset += source_loop_count
            
            except Exception as e:
                pass

    def _transfer_vertex_groups_back(self, proxy_obj, source_objects, vert_ranges):
        """Transfer vertex groups - Simple working method"""
        if "Multisculpt_VG" not in proxy_obj.vertex_groups:
            return
        
        proxy_vg = proxy_obj.vertex_groups["Multisculpt_VG"]
        stored_vg_data = proxy_obj.get("multisculpt_vg_data", [])
        
        for idx, source_obj in enumerate(source_objects):
            if idx >= len(vert_ranges):
                continue
            
            original_name = stored_vg_data[idx] if idx < len(stored_vg_data) else None
            if not original_name:
                continue
                
            start_vert, end_vert = vert_ranges[idx]
            start_vert = int(start_vert)
            end_vert = int(end_vert)
            
            try:
                if original_name in source_obj.vertex_groups:
                    source_obj.vertex_groups.remove(source_obj.vertex_groups[original_name])
                
                target_vg = source_obj.vertex_groups.new(name=original_name)
                
                # Simple: add vertices one at a time with their weight
                for source_vert_idx in range(len(source_obj.data.vertices)):
                    proxy_idx = start_vert + source_vert_idx
                    if proxy_idx < end_vert:
                        try:
                            weight = proxy_vg.weight(proxy_idx)
                            if weight > 0:
                                target_vg.add([source_vert_idx], weight, 'ADD')
                        except:
                            pass
            
            except Exception as e:
                pass

    def _apply_shape_key_changes(self, source_objects):
        """Apply shape key changes to basis geometry"""
        for obj in source_objects:
            if obj.data.shape_keys:
                for key_block in obj.data.shape_keys.key_blocks:
                    if key_block.name != "Basis":
                        key_block.value = 0.0


class MULTISCULPT_OT_IsolateMultisculpt(bpy.types.Operator):
    """Hide original objects and show Multisculpt proxy"""
    bl_idname = "multisculpt.isolate"
    bl_label = "Isolate Multisculpt"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        active_obj = context.active_object
        
        if active_obj and "multisculpt_source_objects" in active_obj:
            proxy_obj = active_obj
        else:
            selected = [obj for obj in context.selected_objects if "multisculpt_source_objects" in obj]
            if not selected:
                self.report({'ERROR'}, "Select a Multisculpt proxy object or objects with proxy")
                return {'CANCELLED'}
            proxy_obj = selected[0]

        source_names = proxy_obj["multisculpt_source_objects"]
        source_objects = [bpy.data.objects[name] for name in source_names if name in bpy.data.objects]

        for obj in source_objects:
            obj.hide_set(True)
            obj.hide_render = True

        proxy_obj.hide_set(False)
        proxy_obj.hide_render = False
        context.view_layer.objects.active = proxy_obj

        self.report({'INFO'}, "Isolated Multisculpt proxy")
        return {'FINISHED'}


class MULTISCULPT_OT_HideMultisculpt(bpy.types.Operator):
    """Hide Multisculpt proxy and show original objects"""
    bl_idname = "multisculpt.hide_proxy"
    bl_label = "Hide Multisculpt Object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        proxy_obj = context.active_object
        
        if not proxy_obj or "multisculpt_source_objects" not in proxy_obj:
            self.report({'ERROR'}, "Select a Multisculpt proxy object")
            return {'CANCELLED'}

        source_names = proxy_obj["multisculpt_source_objects"]
        source_objects = [bpy.data.objects[name] for name in source_names if name in bpy.data.objects]

        proxy_obj.hide_set(True)
        proxy_obj.hide_render = True

        for obj in source_objects:
            obj.hide_set(False)
            obj.hide_render = False

        if source_objects:
            context.view_layer.objects.active = source_objects[0]

        self.report({'INFO'}, "Hidden Multisculpt proxy")
        return {'FINISHED'}


class MULTISCULPT_OT_ApplyMultiresBase(bpy.types.Operator):
    """Apply base subdivision level for Multires modifiers on selected objects"""
    bl_idname = "multisculpt.apply_multires_base"
    bl_label = "Apply Multires Base"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        applied_count = 0

        for obj in selected:
            multires_mod = None
            for mod in obj.modifiers:
                if mod.type == 'MULTIRES':
                    multires_mod = mod
                    break

            if multires_mod and not obj.data.shape_keys:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=multires_mod.name)
                applied_count += 1

        self.report({'INFO'}, f"Applied Multires base on {applied_count} objects")
        return {'FINISHED'}


class MULTISCULPT_PT_Panel(bpy.types.Panel):
    """Main Multisculpt panel in N-Panel"""
    bl_label = "Akelka Multisculpt"
    bl_idname = "MULTISCULPT_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Akelka tools"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.multisculpt_props

        layout.operator("multisculpt.create", text="Create Multisculpt", icon='ADD')
        layout.operator("multisculpt.transfer", text="Transfer Multisculpt", icon='BACK')

        layout.separator()
        layout.prop(props, "transfer_color")
        layout.prop(props, "transfer_uv", text="Transfer UV Maps (Slow)")
        layout.prop(props, "transfer_vertex_groups", text="Transfer Vertex Groups (Slow)")
        layout.prop(props, "auto_clear_instancing")

        layout.separator()
        layout.operator("multisculpt.isolate", text="Isolate", icon='HIDE_OFF')
        layout.operator("multisculpt.hide_proxy", text="Hide Proxy", icon='HIDE_ON')


class MULTISCULPT_OT_CreateAndWeightPaint(bpy.types.Operator):
    """Create multisculpt and enter weight paint mode"""
    bl_idname = "multisculpt.create_weight_paint"
    bl_label = "Create & Weight Paint"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        # Create multisculpt first (it handles mode switching to object mode internally)
        if len(selected_objects) >= 2:
            try:
                bpy.ops.multisculpt.create()
            except Exception as e:
                self.report({'WARNING'}, f"Multisculpt creation: {str(e)}")
        
        # Then switch to weight paint mode
        try:
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        except Exception as e:
            self.report({'WARNING'}, f"Weight paint mode: {str(e)}")
        
        return {'FINISHED'}


class MULTISCULPT_OT_CreateAndVertexPaint(bpy.types.Operator):
    """Create multisculpt and enter vertex paint mode"""
    bl_idname = "multisculpt.create_vertex_paint"
    bl_label = "Create & Vertex Paint"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        selected_objects = [obj for obj in context.selected_objects if obj.type == 'MESH']
        
        # Create multisculpt first (it handles mode switching to object mode internally)
        if len(selected_objects) >= 2:
            try:
                bpy.ops.multisculpt.create()
            except Exception as e:
                self.report({'WARNING'}, f"Multisculpt creation: {str(e)}")
        
        # Then switch to vertex paint mode
        try:
            bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        except Exception as e:
            self.report({'WARNING'}, f"Vertex paint mode: {str(e)}")
        
        return {'FINISHED'}


class MULTISCULPT_MT_PieMenu(bpy.types.Menu):
    """Pie menu for quick mode switching and multisculpt creation"""
    bl_label = "Multisculpt Modes"
    bl_idname = "MULTISCULPT_MT_pie_menu"

    def draw(self, context):
        layout = self.layout
        pie = layout.menu_pie()

        pie.operator("multisculpt.create", text="Create & Sculpt", icon='SCULPTMODE_HLT')
        pie.operator("multisculpt.create_weight_paint", text="Weight Paint", icon='WPAINT_HLT')
        pie.operator("multisculpt.create_vertex_paint", text="Vertex Paint", icon='VPAINT_HLT')
        pie.operator("object.mode_set", text="Texture Paint", icon='TPAINT_HLT').mode = 'TEXTURE_PAINT'
        pie.operator("multisculpt.transfer", text="Transfer", icon='BACK')
        pie.operator("multisculpt.isolate", text="Isolate", icon='HIDE_OFF')
        pie.operator("multisculpt.hide_proxy", text="Hide Proxy", icon='HIDE_ON')
        pie.operator("object.mode_set", text="Object Mode", icon='OBJECT_DATAMODE').mode = 'OBJECT'


def draw_pie_menu(self, context):
    """Draw pie menu in header"""
    layout = self.layout
    layout.menu(MULTISCULPT_MT_PieMenu.bl_idname, icon='SCULPTMODE_HLT')


def _unregister_keymaps():
    """Remove addon keymap items"""
    global addon_keymaps
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()


def _register_keymaps():
    """Register Ctrl+Tab pie menu keymap"""
    global addon_keymaps

    _unregister_keymaps()

    wm = bpy.context.window_manager
    if not wm:
        return False

    kc = wm.keyconfigs.addon
    if not kc:
        return False

    km = kc.keymaps.get("3D View")
    if not km:
        km = kc.keymaps.new(name="3D View", space_type='VIEW_3D')

    kmi = km.keymap_items.new(
        "wm.call_menu_pie",
        'TAB',
        'PRESS',
        ctrl=True,
        head=True,
    )
    kmi.properties.name = "MULTISCULPT_MT_pie_menu"
    addon_keymaps.append((km, kmi))
    return True


def _start_mode_monitor():
    """Start persistent mode-change monitor timer"""
    try:
        bpy.app.timers.unregister(deferred_mode_check)
    except Exception:
        pass
    bpy.app.timers.register(deferred_mode_check, first_interval=0.2, persistent=True)


def _init_addon_services():
    """Deferred init - WM/keyconfig may not exist during early addon register()"""
    global _services_initialized, last_mode

    keymaps_ok = _register_keymaps()
    _start_mode_monitor()

    if keymaps_ok:
        _services_initialized = True
        return None

    return 0.5


@persistent
def _load_post_handler(_dummy):
    """Re-init services after file load if needed"""
    global last_mode
    last_mode = None
    _start_mode_monitor()
    if not _services_initialized:
        bpy.app.timers.register(_init_addon_services, first_interval=0.1)


def register():
    """Register addon classes and properties"""
    global _services_initialized, last_mode

    bpy.utils.register_class(MultiSculptProperties)
    bpy.utils.register_class(MULTISCULPT_OT_CreateMultisculpt)
    bpy.utils.register_class(MULTISCULPT_OT_TransferMultisculpt)
    bpy.utils.register_class(MULTISCULPT_OT_IsolateMultisculpt)
    bpy.utils.register_class(MULTISCULPT_OT_HideMultisculpt)
    bpy.utils.register_class(MULTISCULPT_OT_ApplyMultiresBase)
    bpy.utils.register_class(MULTISCULPT_OT_CreateAndWeightPaint)
    bpy.utils.register_class(MULTISCULPT_OT_CreateAndVertexPaint)
    bpy.utils.register_class(MULTISCULPT_PT_Panel)
    bpy.utils.register_class(MULTISCULPT_MT_PieMenu)

    bpy.types.Scene.multisculpt_props = bpy.props.PointerProperty(type=MultiSculptProperties)

    last_mode = None
    _services_initialized = False

    if _load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post_handler)

    bpy.app.timers.register(_init_addon_services, first_interval=0.1)


def unregister():
    """Unregister addon classes and properties"""
    global _services_initialized, last_mode

    if _load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post_handler)

    for timer_func in (_init_addon_services, deferred_mode_check):
        try:
            bpy.app.timers.unregister(timer_func)
        except Exception:
            pass

    _unregister_keymaps()
    _services_initialized = False
    last_mode = None

    bpy.utils.unregister_class(MultiSculptProperties)
    bpy.utils.unregister_class(MULTISCULPT_OT_CreateMultisculpt)
    bpy.utils.unregister_class(MULTISCULPT_OT_TransferMultisculpt)
    bpy.utils.unregister_class(MULTISCULPT_OT_IsolateMultisculpt)
    bpy.utils.unregister_class(MULTISCULPT_OT_HideMultisculpt)
    bpy.utils.unregister_class(MULTISCULPT_OT_ApplyMultiresBase)
    bpy.utils.unregister_class(MULTISCULPT_OT_CreateAndWeightPaint)
    bpy.utils.unregister_class(MULTISCULPT_OT_CreateAndVertexPaint)
    bpy.utils.unregister_class(MULTISCULPT_PT_Panel)
    bpy.utils.unregister_class(MULTISCULPT_MT_PieMenu)

    del bpy.types.Scene.multisculpt_props


if __name__ == "__main__":
    register()
