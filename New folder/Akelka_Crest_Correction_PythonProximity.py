bl_info = {
    "name": "Shape Key Proximity Deformer (Python)",
    "author": "OpenAI",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Deform",
    "description": "Pure Python proximity deformer: updates target shape key each frame based on evaluated source (including shape keys).",
    "category": "Object",
}

import bpy
from bpy.props import PointerProperty, FloatProperty
from bpy.types import Operator, Panel, PropertyGroup

from mathutils import Vector
from mathutils.bvhtree import BVHTree


MOD_SHAPEKEY_NAME = "SKPD_PY_Deform"

# Runtime-only storage (in-memory). This is enough for interactive playback; it is not persisted across Blender restarts.
_RUNTIME = {}


def _get_or_create_shape_key(target_obj: bpy.types.Object) -> bpy.types.Key:
    if target_obj.type != "MESH":
        raise TypeError("Target must be a mesh")

    if not target_obj.data.shape_keys:
        target_obj.shape_key_add(name="Basis", from_mix=False)

    kb = target_obj.data.shape_keys.key_blocks
    if MOD_SHAPEKEY_NAME in kb:
        sk = kb[MOD_SHAPEKEY_NAME]
    else:
        sk = target_obj.shape_key_add(name=MOD_SHAPEKEY_NAME, from_mix=False)

    # Ensure the deform actually influences the result.
    try:
        sk.value = 1.0
    except Exception:
        pass
    return sk


def _triangulate_faces(mesh) -> list[tuple[int, int, int]]:
    # Use loop triangles for a fast, consistent triangle list for BVH.
    mesh.calc_loop_triangles()
    faces = []
    for tri in mesh.loop_triangles:
        faces.append(tuple(tri.vertices))
    return faces


def _eval_mesh_world(obj: bpy.types.Object, depsgraph) -> tuple[bpy.types.Object, list[Vector], list[tuple[int, int, int]], bpy.types.Mesh]:
    """Return (obj_eval, world_verts, tri_faces, mesh_in_object_space). Caller must clear the mesh."""
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    mat = obj_eval.matrix_world
    world_verts = [mat @ v.co for v in mesh.vertices]
    faces = _triangulate_faces(mesh)
    return obj_eval, world_verts, faces, mesh


def _compute_falloff(distance: float, strength: float, max_distance: float) -> float:
    """
    Dynamic "bone-like" falloff.
    - distance=0 -> 0 (prevents a jump when everything overlaps on frame 0)
    - distance increases -> factor increases up to 1
    """
    if max_distance <= 0.0:
        return 0.0
    t = distance / max_distance
    w = strength * t
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


def _frame_update(scene: bpy.types.Scene, depsgraph):
    # Called from frame-change handler. Keep it exception-safe.
    props = getattr(scene, "skpd_py_props", None)
    scene_strength = getattr(props, "strength", 1.0) if props else 1.0
    scene_max_distance = getattr(props, "max_distance", 120.0) if props else 120.0

    for key, data in list(_RUNTIME.items()):
        if not isinstance(data, dict):
            continue
        target_name = data["target_name"]
        source_name = data["source_name"]
        target_obj = bpy.data.objects.get(target_name)
        source_obj = bpy.data.objects.get(source_name)
        if target_obj is None or source_obj is None:
            _RUNTIME.pop(key, None)
            continue

        if target_obj.type != "MESH":
            continue

        # If user cleared shape keys or deleted target, stop.
        if not target_obj.data.shape_keys:
            continue

        sk = target_obj.data.shape_keys.key_blocks.get(MOD_SHAPEKEY_NAME)
        if sk is None:
            continue

        base_positions_local = data["base_positions_local"]
        prev_positions_local = data["prev_positions_local"]
        offsets_local = data.get("offsets_local", None)
        prev_closest_world = data.get("prev_closest_world", None)
        initialized = data.get("initialized", False)

        if offsets_local is None or prev_closest_world is None:
            # If runtime data was created by an older version (or got corrupted),
            # we fall back to rebuilding state on the fly.
            offsets_local = [Vector((0.0, 0.0, 0.0)) for _ in range(len(target_obj.data.vertices))]
            prev_closest_world = [Vector((0.0, 0.0, 0.0)) for _ in range(len(target_obj.data.vertices))]
            data["offsets_local"] = offsets_local
            data["prev_closest_world"] = prev_closest_world
            initialized = False
            data["initialized"] = False
        strength = scene_strength
        max_distance = scene_max_distance
        last_frame = data.get("last_frame", None)
        if last_frame == scene.frame_current:
            continue
        data["last_frame"] = scene.frame_current

        # Evaluate current source mesh (includes source shape keys).
        try:
            obj_eval, world_verts, tri_faces, src_mesh = _eval_mesh_world(source_obj, depsgraph)
        except Exception:
            continue

        tree = None
        try:
            tree = BVHTree.FromPolygons(world_verts, tri_faces, all_triangles=True)
        except Exception:
            try:
                # Fallback (sometimes the all_triangles flag differs).
                tree = BVHTree.FromPolygons(world_verts, tri_faces)
            except Exception:
                tree = None

        # Clear evaluated mesh to avoid memory buildup.
        try:
            obj_eval.to_mesh_clear()
        except Exception:
            pass

        if tree is None:
            continue

        inv_tgt = target_obj.matrix_world.inverted()
        # Delta mode:
        # For each vertex i:
        #   closest_world_current = closest point on source at THIS frame
        #   delta_world = closest_world_current - closest_world_previous
        #   apply only delta_world to the target via falloff factor

        if not initialized:
            # First successful BVH build after bind: populate prev closest points,
            # but do NOT apply any delta yet (prevents a big jump).
            for i in range(len(target_obj.data.vertices)):
                query_local = prev_positions_local[i]
                query_world = target_obj.matrix_world @ query_local
                res = tree.find_nearest(query_world)
                prev_closest_world[i] = Vector(res[0])
            data["initialized"] = True
            continue

        for i in range(len(target_obj.data.vertices)):
            base_local = base_positions_local[i]
            query_local = prev_positions_local[i]
            query_world = target_obj.matrix_world @ query_local

            res = tree.find_nearest(query_world)
            closest_world_current = res[0]
            if len(res) >= 4:
                distance = float(res[3])
            else:
                distance = (closest_world_current - query_world).length

            factor = _compute_falloff(distance, strength, max_distance)
            delta_world = closest_world_current - prev_closest_world[i]
            delta_local = inv_tgt @ delta_world

            offsets_local[i] = offsets_local[i] + (factor * delta_local)
            new_local = base_local + offsets_local[i]

            sk.data[i].co = new_local
            prev_positions_local[i] = new_local
            prev_closest_world[i] = Vector(closest_world_current)


def _handler(scene, depsgraph):
    # Wrapper for the handler signature. The handler will always call the update.
    _frame_update(scene, depsgraph)


def _register_handler():
    # Register once.
    if "SKPD_PY_handler" in _RUNTIME:
        return
    def _cb(scene):
        # Re-grab depsgraph each call in case of context changes.
        dg = bpy.context.evaluated_depsgraph_get()
        _frame_update(scene, dg)

    _RUNTIME["SKPD_PY_handler"] = _cb
    bpy.app.handlers.frame_change_post.append(_cb)


def _unregister_handler():
    cb = _RUNTIME.get("SKPD_PY_handler", None)
    if cb is not None:
        try:
            bpy.app.handlers.frame_change_post.remove(cb)
        except Exception:
            pass
    _RUNTIME.pop("SKPD_PY_handler", None)


class SKPD_PY_Props(PropertyGroup):
    source_object: PointerProperty(
        name="Source Object",
        type=bpy.types.Object,
        description="Animated source mesh. Its evaluated shape keys are used.",
    )
    target_object: PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Mesh whose shape key will be driven each frame.",
    )
    strength: FloatProperty(
        name="Strength",
        description="Multiplicative falloff strength. Larger values make vertices follow more aggressively.",
        default=1.0,
        min=0.0,
        max=10000.0,
    )
    max_distance: FloatProperty(
        name="Max Distance",
        description="Distance scale used for falloff (acts like influence radius).",
        default=120.0,
        min=0.0,
    )


class OBJECT_OT_skpd_py_bind(Operator):
    bl_idname = "object.skpd_py_bind"
    bl_label = "Bind"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.skpd_py_props
        source = props.source_object
        target = props.target_object
        if source is None or target is None:
            self.report({"ERROR"}, "Pick both a Source Object and a Target Object.")
            return {"CANCELLED"}
        if source.type != "MESH" or target.type != "MESH":
            self.report({"ERROR"}, "Both objects must be meshes.")
            return {"CANCELLED"}

        # Prepare the target shape key and base positions.
        sk = _get_or_create_shape_key(target)

        base_positions_local = [Vector(v.co) for v in target.data.vertices]
        prev_positions_local = [Vector(v.co) for v in target.data.vertices]
        offsets_local = [Vector((0.0, 0.0, 0.0)) for _ in range(len(target.data.vertices))]

        # Initialize the shape key coordinates to base so frame 0 doesn't jump.
        try:
            for i in range(len(target.data.vertices)):
                sk.data[i].co = base_positions_local[i]
        except Exception:
            pass

        # Initialize prev_closest_world at the current frame so the next frame applies a delta.
        prev_closest_world = [Vector((0.0, 0.0, 0.0)) for _ in range(len(target.data.vertices))]
        dg = context.evaluated_depsgraph_get()
        try:
            obj_eval, world_verts, tri_faces, src_mesh = _eval_mesh_world(source, dg)
            tree = BVHTree.FromPolygons(world_verts, tri_faces, all_triangles=True)
        except Exception:
            tree = None
        finally:
            # Clear evaluated mesh to avoid memory buildup.
            try:
                if 'obj_eval' in locals():
                    obj_eval.to_mesh_clear()
            except Exception:
                pass
            try:
                if 'src_mesh' in locals():
                    src_mesh.clear()
            except Exception:
                pass

        initialized = tree is not None
        if initialized:
            inv_tgt = target.matrix_world.inverted()
            for i in range(len(target.data.vertices)):
                query_world = target.matrix_world @ base_positions_local[i]
                res = tree.find_nearest(query_world)
                prev_closest_world[i] = Vector(res[0])

        # Register/update runtime data for this (single) pair.
        # Use a stable key in case user has multiple bind attempts.
        bind_key = f"{target.name}::{source.name}"
        _RUNTIME[bind_key] = {
            "target_name": target.name,
            "source_name": source.name,
            "base_positions_local": base_positions_local,
            "prev_positions_local": prev_positions_local,
            "offsets_local": offsets_local,
            "prev_closest_world": prev_closest_world,
            # Start delta mode on the *next* frame.
            "last_frame": context.scene.frame_current,
            "initialized": initialized,
        }

        # Ensure handler is active.
        _register_handler()

        # No need to force update: we initialized prev closest point already.

        self.report({"INFO"}, f"Bound Python proximity deformer on {target.name}")
        return {"FINISHED"}


class OBJECT_OT_skpd_py_clear(Operator):
    bl_idname = "object.skpd_py_clear"
    bl_label = "Clear"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.skpd_py_props
        target = props.target_object
        source = props.source_object

        if target is None:
            self.report({"ERROR"}, "Pick a Target Object first.")
            return {"CANCELLED"}

        # Remove runtime entries matching this target.
        for k in list(_RUNTIME.keys()):
            if k == "SKPD_PY_handler":
                continue
            if _RUNTIME[k].get("target_name") == target.name:
                _RUNTIME.pop(k, None)

        # Remove the generated shape key (if present).
        if target.data.shape_keys and target.data.shape_keys.key_blocks.get(MOD_SHAPEKEY_NAME):
            try:
                target.shape_key_remove(target.data.shape_keys.key_blocks[MOD_SHAPEKEY_NAME])
            except Exception:
                pass

        # If no pairs remain, remove handler.
        remaining_pairs = [k for k in _RUNTIME.keys() if k != "SKPD_PY_handler"]
        if len(remaining_pairs) == 0:
            _unregister_handler()

        self.report({"INFO"}, f"Cleared Python proximity deformer on {target.name}")
        return {"FINISHED"}


class VIEW3D_PT_skpd_py_panel(Panel):
    bl_label = "Python Proximity Deformer"
    bl_idname = "VIEW3D_PT_skpd_py_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Akelka Tools"

    def draw(self, context):
        layout = self.layout
        props = context.scene.skpd_py_props

        layout.prop(props, "source_object")
        layout.prop(props, "target_object")
        layout.prop(props, "strength")
        layout.prop(props, "max_distance")

        row = layout.row(align=True)
        row.operator("object.skpd_py_bind", icon="GEOMETRY_NODES")
        row.operator("object.skpd_py_clear", icon="TRASH")

        box = layout.box()
        box.label(text="How it works:")
        box.label(text="- Evaluates source shape keys each frame")
        box.label(text="- Finds closest point using BVH")
        box.label(text="- Drives target via shape key")


classes = (
    SKPD_PY_Props,
    OBJECT_OT_skpd_py_bind,
    OBJECT_OT_skpd_py_clear,
    VIEW3D_PT_skpd_py_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.skpd_py_props = PointerProperty(type=SKPD_PY_Props)


def unregister():
    _unregister_handler()
    # Remove properties
    try:
        del bpy.types.Scene.skpd_py_props
    except Exception:
        pass
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

