bl_info = {
    "name": "Akelka Undercoat Fur UV",
    "author": "AkELkA",
    "version": (1, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Akelka Tools > Undercoat Fur",
    "description": "Add UndercoatFurUV on fur/feather cards via body UV island projection (BG3 shared coloring)",
    "category": "UV",
}

import bpy
from bpy.props import BoolProperty, FloatProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from collections import defaultdict, deque
from mathutils import Vector
from mathutils.bvhtree import BVHTree


UNDERCOAT_UV_LAYER_NAME = "UndercoatFurUV"


def _mesh_uv_hashes(me, skip_name=None):
    out = {}
    for uv in me.uv_layers:
        if skip_name and uv.name == skip_name:
            continue
        h = 0
        for d in uv.data:
            h ^= hash((round(d.uv.x, 6), round(d.uv.y, 6)))
        out[uv.name] = h
    return out


def _get_evaluated_mesh(obj, depsgraph, use_evaluated=True):
    if not use_evaluated:
        return obj, obj.data, False
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    return eval_obj, me, True


def _uv_equal(a, b, eps=1e-6):
    return (a - b).length <= eps


def _closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), (1.0, 0.0, 0.0)
    bp = p - b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), (0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v, (1.0 - v, v, 0.0)
    cp = p - c
    d5 = ab.dot(cp)
    d6 = ac.dot(cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), (0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w, (1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w, (0.0, 1.0 - w, w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, (1.0 - v - w, v, w)


class BodyUVProjector:
    """Body mesh triangles in world space + UV islands (union-find on UV continuity)."""

    def __init__(self, body_mesh, body_uv_layer, body_matrix_world):
        self.triangles = []
        body_mesh.calc_loop_triangles()

        for tri in body_mesh.loop_triangles:
            world_verts = [
                body_matrix_world @ body_mesh.vertices[i].co.copy()
                for i in tri.vertices
            ]
            loops = tri.loops
            uv = [
                body_uv_layer.data[loops[i]].uv.copy()
                for i in range(3)
            ]
            self.triangles.append({
                "verts": world_verts,
                "uv": uv,
                "vertices": tuple(tri.vertices),
            })

        verts = []
        polys = []
        for t in self.triangles:
            i0 = len(verts)
            verts.extend(t["verts"])
            polys.append((i0, i0 + 1, i0 + 2))
        self.bvh = BVHTree.FromPolygons(verts, polys) if polys else None

        self.triangle_island = self._build_uv_islands()
        self.island_triangles = defaultdict(list)
        for i, island_id in enumerate(self.triangle_island):
            self.island_triangles[island_id].append(i)

    def _shared_edge_uv_continuous(self, t1, t2):
        v1 = t1["vertices"]
        v2 = t2["vertices"]
        shared = set(v1) & set(v2)
        if len(shared) != 2:
            return False
        for vertex_index in shared:
            uv1 = uv2 = None
            for i, vi in enumerate(v1):
                if vi == vertex_index:
                    uv1 = t1["uv"][i]
            for i, vi in enumerate(v2):
                if vi == vertex_index:
                    uv2 = t2["uv"][i]
            if uv1 is None or uv2 is None or not _uv_equal(uv1, uv2):
                return False
        return True

    def _build_uv_islands(self):
        n = len(self.triangles)
        edge_to_tris = defaultdict(list)
        for ti, tri in enumerate(self.triangles):
            v = tri["vertices"]
            edges = (
                tuple(sorted((v[0], v[1]))),
                tuple(sorted((v[1], v[2]))),
                tuple(sorted((v[2], v[0]))),
            )
            for edge in edges:
                edge_to_tris[edge].append(ti)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for tris in edge_to_tris.values():
            if len(tris) < 2:
                continue
            for i in range(len(tris)):
                for j in range(i + 1, len(tris)):
                    a, b = tris[i], tris[j]
                    if self._shared_edge_uv_continuous(self.triangles[a], self.triangles[b]):
                        union(a, b)

        island_ids = {}
        triangle_island = []
        for i in range(n):
            root = find(i)
            if root not in island_ids:
                island_ids[root] = len(island_ids)
            triangle_island.append(island_ids[root])
        return triangle_island

    def closest_body_triangle(self, point_world):
        if self.bvh is None:
            return None
        hit = self.bvh.find_nearest(point_world)
        if hit is None:
            return None
        _location, _normal, index, distance = hit
        return index, distance

    def closest_in_island(self, point_world, island_id):
        best_distance = float("inf")
        best_triangle = None
        best_bary = None
        for ti in self.island_triangles.get(island_id, []):
            tri = self.triangles[ti]
            location, bary = _closest_point_on_triangle(
                point_world,
                tri["verts"][0],
                tri["verts"][1],
                tri["verts"][2],
            )
            distance_sq = (point_world - location).length_squared
            if distance_sq < best_distance:
                best_distance = distance_sq
                best_triangle = ti
                best_bary = bary
        if best_triangle is None:
            return None
        return best_triangle, best_bary, best_distance ** 0.5

    def uv_from_triangle(self, tri_index, bary):
        tri = self.triangles[tri_index]
        return (
            tri["uv"][0] * bary[0]
            + tri["uv"][1] * bary[1]
            + tri["uv"][2] * bary[2]
        )


def _fur_vertex_components(fur_mesh):
    adjacency = [[] for _ in fur_mesh.vertices]
    for edge in fur_mesh.edges:
        a, b = edge.vertices
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = set()
    components = []
    for start in range(len(fur_mesh.vertices)):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component = []
        while queue:
            v = queue.popleft()
            component.append(v)
            for n in adjacency[v]:
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
        components.append(component)
    return components


def _ensure_output_uv_layer(fur_mesh, name):
    layer = fur_mesh.uv_layers.get(name)
    if layer is None:
        layer = fur_mesh.uv_layers.new(name=name)
    return layer


def transfer_fur_object(fur_obj, projector, props, stats, depsgraph):
    eval_obj, eval_me, is_temp = _get_evaluated_mesh(
        fur_obj, depsgraph, props.use_evaluated_mesh
    )
    fur_mesh = fur_obj.data
    fur_matrix = eval_obj.matrix_world

    world_pos = {
        v.index: fur_matrix @ v.co
        for v in fur_mesh.vertices
    }

    out_uv = _ensure_output_uv_layer(fur_mesh, props.uv_layer_name)
    components = _fur_vertex_components(fur_mesh)
    stats["cards"] += len(components)

    for component in components:
        root_vertex = None
        root_distance = float("inf")
        root_tri = None

        for vi in component:
            hit = projector.closest_body_triangle(world_pos[vi])
            if hit is None:
                continue
            tri_index, distance = hit
            if distance < root_distance:
                root_distance = distance
                root_vertex = vi
                root_tri = tri_index

        if root_vertex is None:
            stats["skipped"] += 1
            continue

        chosen_island = projector.triangle_island[root_tri]

        vertex_uv = {}
        for vi in component:
            p = world_pos[vi]
            result = projector.closest_in_island(p, chosen_island)
            if result is None:
                continue
            tri_index, bary, _dist = result
            vertex_uv[vi] = projector.uv_from_triangle(tri_index, bary)

        for poly in fur_mesh.polygons:
            if not any(vi in vertex_uv for vi in poly.vertices):
                continue
            for loop_index in poly.loop_indices:
                vi = fur_mesh.loops[loop_index].vertex_index
                if vi in vertex_uv:
                    out_uv.data[loop_index].uv = vertex_uv[vi]

    if is_temp:
        eval_obj.to_mesh_clear()


def build_projector(source_obj, props, depsgraph):
    eval_obj, body_mesh, is_temp = _get_evaluated_mesh(
        source_obj, depsgraph, props.use_evaluated_mesh
    )
    if not body_mesh.uv_layers:
        if is_temp:
            eval_obj.to_mesh_clear()
        return None, None

    if props.body_uv_name:
        body_uv = body_mesh.uv_layers.get(props.body_uv_name)
    else:
        body_uv = body_mesh.uv_layers[0]

    if body_uv is None:
        if is_temp:
            eval_obj.to_mesh_clear()
        return None, None

    projector = BodyUVProjector(body_mesh, body_uv, eval_obj.matrix_world)
    if is_temp:
        eval_obj.to_mesh_clear()
    return projector, None


class AUF_Properties(PropertyGroup):
    uv_layer_name: StringProperty(
        name="New UV Layer",
        default=UNDERCOAT_UV_LAYER_NAME,
    )
    body_uv_name: StringProperty(
        name="Body UV",
        default="",
        description="Undercoat UV layer to sample (empty = first layer)",
    )
    use_evaluated_mesh: BoolProperty(
        name="Apply Modifiers",
        default=True,
    )


class AUF_OT_transfer_undercoat_uv(Operator):
    bl_idname = "akelka.transfer_undercoat_fur_uv"
    bl_label = "Transfer Undercoat UV"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        active = context.active_object
        if not active or active.type != "MESH":
            return False
        targets = [o for o in context.selected_objects if o.type == "MESH" and o != active]
        return len(targets) > 0

    def execute(self, context):
        props = context.scene.auf_props
        source_obj = context.active_object
        targets = [o for o in context.selected_objects if o.type == "MESH" and o != source_obj]

        before_hashes = {
            obj.name: _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
            for obj in targets
        }

        depsgraph = context.evaluated_depsgraph_get()
        projector, err = build_projector(source_obj, props, depsgraph)
        if projector is None:
            self.report({"ERROR"}, err or "Could not build body UV projector")
            return {"CANCELLED"}

        stats = {"cards": 0, "skipped": 0}
        for obj in targets:
            transfer_fur_object(obj, projector, props, stats, depsgraph)

        for obj in targets:
            after = _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
            if before_hashes.get(obj.name) != after:
                self.report({"ERROR"}, f"Existing UV layers changed on {obj.name}")
                return {"CANCELLED"}

        n_islands = len(projector.island_triangles)
        msg = (
            f"{len(targets)} mesh(es), {stats['cards']} cards, "
            f"body UV islands: {n_islands}, skipped: {stats['skipped']}"
        )
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class AUF_PT_undercoat_fur(Panel):
    bl_label = "Undercoat Fur"
    bl_idname = "AUF_PT_undercoat_fur"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Akelka Tools"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        props = context.scene.auf_props

        box = layout.box()
        col = box.column(align=True)
        col.label(text="Active: undercoat (body UV source)", icon="MESH_DATA")
        col.label(text="Selected: fur / feather cards", icon="OUTLINER_OB_MESH")
        col.operator("akelka.transfer_undercoat_fur_uv", icon="UV")

        settings = layout.box()
        scol = settings.column(align=True)
        scol.prop(props, "uv_layer_name")
        scol.prop(props, "body_uv_name")
        scol.prop(props, "use_evaluated_mesh")


classes = (
    AUF_Properties,
    AUF_OT_transfer_undercoat_uv,
    AUF_PT_undercoat_fur,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.auf_props = PointerProperty(type=AUF_Properties)


def unregister():
    del bpy.types.Scene.auf_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
