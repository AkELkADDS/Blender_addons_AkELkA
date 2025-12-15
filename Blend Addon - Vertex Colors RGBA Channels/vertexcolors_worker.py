# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "VertexColor split",
    "author": "vitos1k",
    "version": (0, 1, 7),
    "blender": (2, 80 ,0),
    "location": "View3D > UI panel",
    "description": "Split Vertex Colors to RGB components and back",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "3D View"}

import bpy 
from bpy.props import *
from colorsys import rgb_to_hsv

class FillSelectedVertexColors(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "paint.fill_selected_vertex_colors"
    bl_label = "Fill selected vertices with color of the layer"
    bl_space_type = "VIEW_3D"
    bl_options = {'REGISTER', 'UNDO'}  
    fill_power: bpy.props.FloatProperty(name="Fill Value", description="Fill value of the vertices", default=1.0,min=0,soft_max=1.0)
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        fillvcol(context,self.fill_power)
        return {'FINISHED'}


class BlurVertexColors(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "paint.blur_vertex_colors"
    bl_label = "Blur active vertex color layer"
    bl_space_type = "VIEW_3D"
    bl_options = {'REGISTER', 'UNDO'}    
    blur_power: bpy.props.FloatProperty(name="BlurPower", description="Blur amount Power ", default=0.1,min=0.0,soft_max=1.0)
    blur_repeat: bpy.props.IntProperty(name="BlurRepeat", description="Blur repeat cycles ", default=1,min=1,soft_max=5)
    blur_expand: bpy.props.FloatProperty(name="Expand_Contract", description="Expand / Contract borders", default=0.0,soft_min=-5.0,soft_max=5.0)

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        blurvcol(context,self.blur_power,self.blur_repeat,self.blur_expand)    
        return {'FINISHED'}

class combinevcol(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "paint.combine_vcol"
    bl_label = "Combine vertex color channels"
    bl_space_type = "VIEW_3D"
    bl_options = {'REGISTER', 'UNDO'}
       
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        compose(context)        
        return {'FINISHED'}

class separatevcol(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "paint.separate_vcol"
    bl_label = "Separate vertex color channels"
    bl_space_type = "VIEW_3D"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        decompose(context)        
        return {'FINISHED'}

class VertexColorToolsPanel(bpy.types.Panel):
    bl_label = "VertexPainTools"
    #bl_category = "Tools"
    bl_category = "VertexPainTools"
    bl_space_type = 'VIEW_3D'
    #bl_region_type = 'TOOLS'
    bl_region_type = 'UI'
    bl_context = 'vertexpaint'


    def draw(self, context):
        layout = self.layout
        col = layout.column()        
        colrow = col.row(align=True)
        col.separator()
        col.operator("paint.separate_vcol", text="Separate")        
        col.operator("paint.combine_vcol", text="Combine")        
        col.operator("paint.blur_vertex_colors", text="Blur")
        col.operator("paint.fill_selected_vertex_colors", text="Fill")
        col.separator()

def decompose(context):
    obj=context.active_object
    if len(obj.data.vertex_colors)>0:
        vcolnames = [vc.name.strip().upper() for vc in obj.data.vertex_colors]
    else:
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = "There are no vertex_color layers")
        return    
    if ('RED_CHAN' in vcolnames) or ('GREEN_CHAN' in vcolnames) or ('BLUE_CHAN' in vcolnames) or ('ALPHA_CHAN' in vcolnames) :
    #if ('RED_CHAN' in vcolnames) or ('GREEN_CHAN' in vcolnames) or ('BLUE_CHAN' in vcolnames):
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = "Can't decompose because there are vcol layers named after channels names")
        return
    
    layername = obj.data.vertex_colors.active.name
    obj.data.vertex_colors.new(name = 'RED_CHAN')
    obj.data.vertex_colors.new(name = 'GREEN_CHAN')
    obj.data.vertex_colors.new(name = 'BLUE_CHAN')
    obj.data.vertex_colors.new(name = 'ALPHA_CHAN')
    for poly in obj.data.polygons:
        for loop in poly.loop_indices:
            #vertindex=obj.data.loops[loop].vertex_index
            col = obj.data.vertex_colors[layername].data[loop].color
            if len(col)==4:
                alpha = col[3]
            else:
                alpha = 1
            obj.data.vertex_colors['RED_CHAN'].data[loop].color = (col[0],0,0,1)
            obj.data.vertex_colors['GREEN_CHAN'].data[loop].color = (0,col[1],0,1)
            obj.data.vertex_colors['BLUE_CHAN'].data[loop].color = (0,0,col[2],1)
            obj.data.vertex_colors['ALPHA_CHAN'].data[loop].color = (alpha,alpha,alpha,1)
    obj.data.vertex_colors.remove(obj.data.vertex_colors[layername])
    obj.data.vertex_colors['RED_CHAN'].active  = True
    obj.data.vertex_colors['RED_CHAN'].active_render = True
    obj.data.update()
    return {'FINISHED'}

def compose(context):
    obj=context.active_object
    if len(obj.data.vertex_colors)>0:
        vcolnames = [vc.name.strip().upper() for vc in obj.data.vertex_colors]
    else:
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = "There are no vertex_color layers")
        return

    if ('RED_CHAN' not in vcolnames) or ('GREEN_CHAN' not in vcolnames) or ('BLUE_CHAN' not in vcolnames) or ('ALPHA_CHAN' not in vcolnames):
    #if ('RED_CHAN' not in vcolnames) or ('GREEN_CHAN' not in vcolnames) or ('BLUE_CHAN' not in vcolnames):
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = "Can't compose because need 4 layers named after channels")
        return
    if ('COMBINED' in vcolnames):
        bpy.ops.error.message('INVOKE_DEFAULT',
                type = "Error",
                message = "Combined layer already exists.Delete it and repeat")
        return
    obj.data.vertex_colors.new(name = 'COMBINED')
    obj.data.vertex_colors['COMBINED'].active = True   
        
    for poly in obj.data.polygons:
        for loop in poly.loop_indices:
            #vertindex=obj.data.loops[loop].vertex_index
            redcolor = obj.data.vertex_colors['RED_CHAN'].data[loop].color[0]
            greencolor = obj.data.vertex_colors['GREEN_CHAN'].data[loop].color[1]
            bluecolor = obj.data.vertex_colors['BLUE_CHAN'].data[loop].color[2]
            alphacolor = obj.data.vertex_colors['ALPHA_CHAN'].data[loop].color[1]            
            obj.data.vertex_colors['COMBINED'].data[loop].color = (redcolor,greencolor,bluecolor,alphacolor)
    obj.data.vertex_colors.remove(obj.data.vertex_colors['RED_CHAN'])
    obj.data.vertex_colors.remove(obj.data.vertex_colors['GREEN_CHAN'])
    obj.data.vertex_colors.remove(obj.data.vertex_colors['BLUE_CHAN'])
    obj.data.vertex_colors.remove(obj.data.vertex_colors['ALPHA_CHAN'])    
    obj.data.vertex_colors['COMBINED'].active = True
    obj.data.vertex_colors['COMBINED'].active_render = True
    obj.data.update()
    return {'FINISHED'}

def blurvcol(context,blur_power,blur_repeat,blur_expand):
    obj = context.active_object
    if len(obj.data.vertex_colors)>0:
        vcolname = obj.data.vertex_colors.active.name
        vcolname = vcolname.strip().upper()
    else:
        bpy.ops.error.message('INVOKE_DEFAULT',
                type = "Error",
                message = "There are no vertex_color layers in this object")
        return

    if ('RED_CHAN' != vcolname) and ('GREEN_CHAN' != vcolname) and ('BLUE_CHAN' != vcolname) and ('ALPHA_CHAN' != vcolname):
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = "I can blur only individual channels")
        return
    if len(obj.vertex_groups)>0:
        vgroups = [vg.name for vg in obj.vertex_groups]
        curractivename = obj.vertex_groups.active.name
        if ('TEMP_FOR_TRANSLATE' in vgroups):
            obj.vertex_groups.remove(obj.vertex_groups['TEMP_FOR_TRANSLATE'])
    obj.vertex_groups.new(name='TEMP_FOR_TRANSLATE')
        
    if not transferVertexCol2Weight(context,obj.vertex_groups['TEMP_FOR_TRANSLATE'],obj.data.vertex_colors[vcolname]):
        return

    obj.vertex_groups.active_index = obj.vertex_groups['TEMP_FOR_TRANSLATE'].index
    current_mode = context.object.mode
    bpy.ops.object.mode_set(mode = 'WEIGHT_PAINT')
    current_paintmask = obj.data.use_paint_mask_vertex
    obj.data.use_paint_mask_vertex = True
    bpy.ops.object.vertex_group_smooth(group_select_mode='ACTIVE', factor=blur_power, repeat=blur_repeat, expand=blur_expand)
    #bpy.ops.object.vertex_group_normalize()
    obj.data.use_paint_mask_vertex = current_paintmask
    bpy.ops.object.mode_set(mode = current_mode)
    if not transferWeight2VertexCol(context,obj.vertex_groups['TEMP_FOR_TRANSLATE'],obj.data.vertex_colors[vcolname]):
        return
    obj.vertex_groups.remove(obj.vertex_groups['TEMP_FOR_TRANSLATE'])
    if len(obj.vertex_groups)>0:
        obj.vertex_groups.active_index = obj.vertex_groups[curractivename].index
    return {'FINISHED'}

def transferVertexCol2Weight(context,vgroup,vcol):
    obj=context.active_object
    
    try:
        assert obj.vertex_groups
        assert obj.data.vertex_colors
        
    except AssertionError:
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = 'you need at least one vertex group and one color group')
        return False

    if (vcol is not None) and (vgroup is not None): 
        print ("enough parameters")

        for poly in obj.data.polygons:
            for loop in poly.loop_indices:
                if vcol.name.strip().upper() == 'RED_CHAN':
                    weight = vcol.data[loop].color[0]
                elif vcol.name.strip().upper() == 'GREEN_CHAN':
                    weight = vcol.data[loop].color[1]
                elif vcol.name.strip().upper() == 'BLUE_CHAN':
                    weight = vcol.data[loop].color[2]
                elif vcol.name.strip().upper() == 'ALPHA_CHAN':
                    col = vcol.data[loop].color
                    weight = rgb_to_hsv(col[0],col[1],col[2])[2]
                else:
                    col = vcol.data[loop].color
                    weight = rgb_to_hsv(col[0],col[1],col[2])[2]
                vertindex=obj.data.loops[loop].vertex_index
                if (obj.data.vertices[vertindex].select == True):
                    vgroup.add([vertindex],weight,'ADD')
    else:
        return False
    return True


def transferWeight2VertexCol(context,vgroup,vcol):
    obj=context.active_object    
    
    try:
        assert obj.vertex_groups
        assert obj.data.vertex_colors
        
    except AssertionError:
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = 'you need at least one vertex group and one color group')
        return False
    
    if (vcol is not None) and (vgroup is not None): 
        print ("enough parameters")        

        for poly in obj.data.polygons:
            for loop in poly.loop_indices:
                vertindex=obj.data.loops[loop].vertex_index
                if (obj.data.vertices[vertindex].select == True):        
                    try:
                        weight=vgroup.weight(vertindex)
                    except:
                        continue
                    rgb = (0,0,0,1)
                    if vcol.name.strip().upper() == 'RED_CHAN':
                        rgb=(weight,0,0,1)
                    elif vcol.name.strip().upper() == 'GREEN_CHAN':
                        rgb=(0,weight,0,1)                        
                    elif vcol.name.strip().upper() == 'BLUE_CHAN':
                        rgb=(0,0,weight,1)
                    elif vcol.name.strip().upper() == 'ALPHA_CHAN':
                        rgb=(weight,weight,weight,1)
                    else:
                        rgb=(weight,weight,weight,1)
                    vcol.data[loop].color = rgb
    else:
        return False
    return True

def fillvcol(context,fill_power):
    obj = context.active_object
    try:        
        assert obj.data.vertex_colors        
    except AssertionError:
        bpy.ops.error.message('INVOKE_DEFAULT', 
                type = "Error",
                message = 'you need at least one vertex color layer')
        return False
    vcol = obj.data.vertex_colors.active
    vcolname = vcol.name.strip().upper()
    fill_power = float(fill_power)
    for poly in obj.data.polygons:
        for loop in poly.loop_indices:
            vertindex=obj.data.loops[loop].vertex_index
            if (obj.data.vertices[vertindex].select == True):
                if  vcolname == 'RED_CHAN':
                    r=fill_power
                    g=0.0
                    b=0.0
                elif vcolname == 'GREEN_CHAN':
                    r=0.0
                    g=fill_power
                    b=0.0
                elif vcolname == 'BLUE_CHAN':
                    r=0.0
                    g=0.0
                    b=fill_power
                elif vcolname == 'ALPHA_CHAN':
                    r=fill_power
                    g=fill_power
                    b=fill_power
                else:
                    r=fill_power
                    g=fill_power
                    b=fill_power
                vcol.data[loop].color = (r, g, b, 1.0)


class MessageOperator(bpy.types.Operator):
    bl_idname = "error.message"
    bl_label = "Message"
    ftype: StringProperty()
    message: StringProperty()
 
    def execute(self, context):
        self.report({'INFO'}, self.message)
        print(self.message)
        return {'FINISHED'}
 
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_popup(self, width=800, height=200)
 
    def draw(self, context):
        self.layout.label("A message has arrived")
        row = self.layout.split(0.25)
        row.prop(self, "ftype")
        row.prop(self, "message")
        row = self.layout.split(0.80)
        row.label("") 
        row.operator("error.ok")
 
#
#   The OK button in the error dialog
#
class OkOperator(bpy.types.Operator):
    bl_idname = "error.ok"
    bl_label = "OK"
    def execute(self, context):
        return {'FINISHED'}
    
"""
def menu_draw(self, context): 
    self.layout.operator_context = 'INVOKE_REGION_WIN' 
    self.layout.operator(Bevel.bl_idname, "Weight2VertexCol") 
"""

def register():
    bpy.utils.register_class(combinevcol)
    bpy.utils.register_class(separatevcol)
    bpy.utils.register_class(VertexColorToolsPanel)
    bpy.utils.register_class(BlurVertexColors)
    bpy.utils.register_class(FillSelectedVertexColors)
    print('registred')
    #bpy.types.VIEW3D_MT_edit_mesh_specials.prepend(menu_draw) 
    
    #error window
    bpy.utils.register_class(OkOperator)
    bpy.utils.register_class(MessageOperator)


def unregister():
    #bpy.types.VIEW3D_MT_edit_mesh_specials.remove(menu_draw) 
    bpy.utils.unregister_class(combinevcol)
    bpy.utils.unregister_class(separatevcol)
    bpy.utils.unregister_class(VertexColorToolsPanel)
    bpy.utils.unregister_class(BlurVertexColors)
    bpy.utils.unregister_class(FillSelectedVertexColors)

if __name__ == "__main__":
    register()