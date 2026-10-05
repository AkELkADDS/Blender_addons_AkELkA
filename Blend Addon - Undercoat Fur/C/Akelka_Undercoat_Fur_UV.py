bl_info = {
    "name": "Akelka Undercoat Fur UV",
    "author": "AkELkA",
    "version": (1, 9, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Akelka Tools > Undercoat Fur",
    "description": "Add UndercoatFurUV on fur/feather cards via body UV island projection (BG3 shared coloring)",
    "category": "UV",
}

import bpy
from bpy.props import BoolProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from collections import defaultdict, deque
from mathutils import Vector
from mathutils.bvhtree import BVHTree


UNDERCOAT_UV_LAYER_NAME = "UndercoatFurUV"


def _mesh_object_poll(_self, obj):
    return obj is not None and obj.type == "MESH"


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
            uv = [body_uv_layer.data[loops[i]].uv.copy() for i in range(3)]
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
        self.island_uv_sum = defaultdict(lambda: Vector((0.0, 0.0)))
        self.island_uv_count = defaultdict(int)
        for i, island_id in enumerate(self.triangle_island):
            self.island_triangles[island_id].append(i)
            center = (
                self.triangles[i]["uv"][0]
                + self.triangles[i]["uv"][1]
                + self.triangles[i]["uv"][2]
            ) / 3.0
            self.island_uv_sum[island_id] += center
            self.island_uv_count[island_id] += 1

    def island_uv_centroid(self, island_id):
        count = self.island_uv_count.get(island_id, 0)
        if count <= 0:
            return Vector((0.5, 0.5))
        return self.island_uv_sum[island_id] / count

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


def _bake_reference_from_body(ref_obj, body_projector, props, depsgraph):
    """Fill reference UndercoatFurUV from undercoat (per-vertex body sample)."""
    eval_obj, _eval_me, is_temp = _get_evaluated_mesh(
        ref_obj, depsgraph, props.use_evaluated_mesh
    )
    ref_mesh = ref_obj.data
    fur_matrix = eval_obj.matrix_world
    out_uv = _ensure_output_uv_layer(ref_mesh, props.uv_layer_name)

    for v in ref_mesh.vertices:
        world = fur_matrix @ v.co
        hit = body_projector.closest_body_triangle(world)
        if hit is None:
            continue
        tri_index, _dist = hit
        island = body_projector.triangle_island[tri_index]
        result = body_projector.closest_in_island(world, island)
        if result is None:
            continue
        tri_index, bary, _d = result
        uv = body_projector.uv_from_triangle(tri_index, bary)
        for loop in ref_mesh.loops:
            if loop.vertex_index == v.index:
                out_uv.data[loop.index].uv = uv

    if is_temp:
        eval_obj.to_mesh_clear()


def build_body_projector(source_obj, props, depsgraph):
    eval_obj, body_mesh, is_temp = _get_evaluated_mesh(
        source_obj, depsgraph, props.use_evaluated_mesh
    )
    if not body_mesh.uv_layers:
        if is_temp:
            eval_obj.to_mesh_clear()
        return None

    if props.body_uv_name:
        body_uv = body_mesh.uv_layers.get(props.body_uv_name)
    else:
        body_uv = body_mesh.uv_layers[0]

    if body_uv is None:
        if is_temp:
            eval_obj.to_mesh_clear()
        return None

    projector = BodyUVProjector(body_mesh, body_uv, eval_obj.matrix_world)
    if is_temp:
        eval_obj.to_mesh_clear()
    return projector


def _triangle_texel(tri):
    """World length per UV unit on a body triangle."""
    scales = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        uv_len = (tri["uv"][i] - tri["uv"][j]).length
        world_len = (tri["verts"][i] - tri["verts"][j]).length
        if uv_len > 1e-5 and world_len > 1e-8:
            scales.append(world_len / uv_len)
    if not scales:
        return 0.45
    scales.sort()
    texel = scales[len(scales) // 2]
    if texel < 0.05 or texel > 3.0:
        return 0.45
    return texel


def _uv_face_signs(fur_mesh, group_set, vertex_uv):
    """Count faces with positive, negative, and ~zero UV area."""
    pos = neg = zero = 0
    for poly in fur_mesh.polygons:
        verts = poly.vertices
        if not all(vi in group_set and vi in vertex_uv for vi in verts):
            continue
        uvs = [vertex_uv[vi] for vi in verts]
        area = 0.0
        for i in range(1, len(uvs) - 1):
            e1 = uvs[i] - uvs[0]
            e2 = uvs[i + 1] - uvs[0]
            area += e1.x * e2.y - e1.y * e2.x
        if abs(area) < 1e-8:
            zero += 1
        elif area > 0.0:
            pos += 1
        else:
            neg += 1
    return pos, neg, zero


def _repair_bad_card(fur_mesh, vertex_uv, world_pos, component, projector, root_vertex, root_tri, island):
    """Rebuild twisted, line, dot, or border-smeared cards as one planar island.

    Anchored at the root's projected UV so the card stays on the right part of the atlas.
    """
    if root_vertex not in vertex_uv or root_vertex not in world_pos or root_tri is None:
        return False
    group = [vi for vi in component if vi in vertex_uv and vi in world_pos]
    if len(group) < 3:
        return False

    xs = [vertex_uv[vi].x for vi in group]
    ys = [vertex_uv[vi].y for vi in group]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    border = 0
    for vi in group:
        u = vertex_uv[vi]
        if u.x <= 0.02 or u.x >= 0.98 or u.y <= 0.02 or u.y >= 0.98:
            border += 1

    pos, neg, zero = _uv_face_signs(fur_mesh, set(group), vertex_uv)
    signed = pos + neg
    # One face wound the other way is enough to fold the island.
    twisted = pos > 0 and neg > 0
    collapsed = (min(dx, dy) < 0.012 and max(dx, dy) > 0.008) or max(dx, dy) < 0.01
    border_smear = border >= 3 and border >= 0.25 * len(group)
    flat_faces = zero >= 1 and (zero >= max(1, signed) or border_smear)
    if not (twisted or collapsed or border_smear or flat_faces):
        return False

    root_w = world_pos[root_vertex]
    anchor = vertex_uv[root_vertex].copy()
    normal = Vector((0.0, 0.0, 0.0))
    for poly in fur_mesh.polygons:
        verts = [vi for vi in poly.vertices if vi in world_pos]
        if len(verts) < 3 or not all(vi in group for vi in poly.vertices):
            continue
        for i in range(1, len(poly.vertices) - 1):
            a = world_pos[poly.vertices[0]]
            b = world_pos[poly.vertices[i]]
            c = world_pos[poly.vertices[i + 1]]
            normal += (b - a).cross(c - a)
    if normal.length_squared < 1e-12:
        normal = Vector((0.0, 0.0, 1.0))
    else:
        normal.normalize()

    longest = 0.0
    axis_u = None
    for vi in group:
        if vi == root_vertex:
            continue
        delta = world_pos[vi] - root_w
        length = delta.length
        if length > longest:
            longest = length
            axis_u = delta / length
    if axis_u is None:
        return False
    axis_u = axis_u - normal * axis_u.dot(normal)
    if axis_u.length_squared < 1e-10:
        return False
    axis_u.normalize()
    axis_v = normal.cross(axis_u)
    if axis_v.length_squared < 1e-10:
        return False
    axis_v.normalize()

    coords = []
    max_r = 0.0
    for vi in group:
        delta = world_pos[vi] - root_w
        s = delta.dot(axis_u)
        t = delta.dot(axis_v)
        coords.append((vi, s, t))
        radius = (s * s + t * t) ** 0.5
        if radius > max_r:
            max_r = radius
    if max_r < 1e-8:
        return False

    natural = max_r / _triangle_texel(projector.triangles[root_tri])
    radius_uv = min(0.05, max(0.02, natural * 0.65))
    scale = radius_uv / max_r

    inward = projector.island_uv_centroid(island) - anchor
    if inward.length_squared < 1e-10:
        inward = Vector((0.0, 1.0))
    else:
        inward.normalize()
    tangent = Vector((-inward.y, inward.x))

    for vi, s, t in coords:
        vertex_uv[vi] = anchor + tangent * (s * scale) + inward * (t * scale)

    pos2, neg2, _zero2 = _uv_face_signs(fur_mesh, set(group), vertex_uv)
    if neg2 > pos2:
        for vi, s, t in coords:
            vertex_uv[vi] = anchor + tangent * (s * scale) - inward * (t * scale)
    return True


def transfer_fur_object(fur_obj, projector, props, stats, depsgraph):
    eval_obj, _eval_me, is_temp = _get_evaluated_mesh(
        fur_obj, depsgraph, props.use_evaluated_mesh
    )
    fur_mesh = fur_obj.data
    fur_matrix = eval_obj.matrix_world

    world_pos = {v.index: fur_matrix @ v.co for v in fur_mesh.vertices}

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

        if _repair_bad_card(
            fur_mesh,
            vertex_uv,
            world_pos,
            component,
            projector,
            root_vertex,
            root_tri,
            chosen_island,
        ):
            stats["repaired"] = stats.get("repaired", 0) + 1

        for poly in fur_mesh.polygons:
            if not any(vi in vertex_uv for vi in poly.vertices):
                continue
            for loop_index in poly.loop_indices:
                vi = fur_mesh.loops[loop_index].vertex_index
                if vi in vertex_uv:
                    out_uv.data[loop_index].uv = vertex_uv[vi]

    if is_temp:
        eval_obj.to_mesh_clear()


def _diagnose_mesh_uv(fur_obj, uv_layer_name):
    """Quick sanity stats after transfer (for MCP / operator report)."""
    me = fur_obj.data
    if uv_layer_name not in me.uv_layers:
        return {"dot_cards": 0, "total_cards": 0, "avg_span": 0.0}

    uvlay = me.uv_layers[uv_layer_name]

    def vert_uv(vi):
        for loop in me.loops:
            if loop.vertex_index == vi:
                return uvlay.data[loop.index].uv.copy()
        return None

    dot_cards = 0
    spans = []
    comps = _fur_vertex_components(me)
    for comp in comps:
        if len(comp) < 2:
            continue
        uvs = []
        for vi in comp:
            u = vert_uv(vi)
            if u is not None:
                uvs.append(u)
        if len(uvs) < 2:
            continue
        unique = len({(round(u.x, 5), round(u.y, 5)) for u in uvs})
        if unique == 1:
            dot_cards += 1
        root = uvs[0]
        spans.append(max((u - root).length for u in uvs))

    n = len([c for c in comps if len(c) >= 2])
    return {
        "dot_cards": dot_cards,
        "total_cards": n,
        "avg_span": sum(spans) / len(spans) if spans else 0.0,
    }


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
    reference_object: PointerProperty(
        name="UV Reference",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
        description=(
            "Optional mesh (e.g. scalp): only its UndercoatFurUV is filled from the "
            "undercoat. Feather cards always project from the undercoat (v1.1 behavior)"
        ),
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

        reference = props.reference_object
        if reference is not None:
            if reference.type != "MESH":
                self.report({"ERROR"}, "UV Reference must be a mesh")
                return {"CANCELLED"}
            if reference == source_obj:
                reference = None

        before_hashes = {
            obj.name: _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
            for obj in targets
        }

        depsgraph = context.evaluated_depsgraph_get()
        body_projector = build_body_projector(source_obj, props, depsgraph)
        if body_projector is None:
            self.report({"ERROR"}, "Could not build body UV projector")
            return {"CANCELLED"}

        ref_label = ""
        if reference is not None:
            _bake_reference_from_body(reference, body_projector, props, depsgraph)
            ref_label = reference.name

        card_projector = body_projector

        stats = {"cards": 0, "skipped": 0}
        card_meshes = []
        for obj in targets:
            if reference is not None and obj == reference:
                continue
            transfer_fur_object(obj, card_projector, props, stats, depsgraph)
            card_meshes.append(obj)

        for obj in targets:
            after = _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
            if before_hashes.get(obj.name) != after:
                self.report({"ERROR"}, f"Existing UV layers changed on {obj.name}")
                return {"CANCELLED"}

        dots = 0
        total = 0
        for obj in card_meshes:
            d = _diagnose_mesh_uv(obj, props.uv_layer_name)
            dots += d["dot_cards"]
            total += d["total_cards"]

        n_islands = len(body_projector.island_triangles)
        if ref_label:
            msg = (
                f"{len(card_meshes)} card mesh(es), baked ref '{ref_label}', "
                f"{stats['cards']} components, skipped {stats['skipped']}"
            )
        else:
            msg = (
                f"{len(card_meshes)} mesh(es), {stats['cards']} components, "
                f"body islands {n_islands}, skipped {stats['skipped']}"
            )
        if total:
            msg += f", dot-cards {dots}/{total}"
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
        scol.prop(props, "reference_object")
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
