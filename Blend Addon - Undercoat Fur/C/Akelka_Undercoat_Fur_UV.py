bl_info = {
    "name": "Akelka Undercoat Fur UV",
    "author": "AkELkA",
    "version": (1, 13, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Akelka Tools > Undercoat Fur",
    "description": "Add UndercoatFurUV on fur/feather cards via body UV island projection (BG3 shared coloring)",
    "category": "UV",
}

import bpy
from bpy.props import FloatProperty, PointerProperty, StringProperty
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

    def closest_in_island(self, point_world, island_id, anchor_uv=None):
        best_score = float("inf")
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
            distance = (point_world - location).length
            score = distance
            if anchor_uv is not None:
                uv = (
                    tri["uv"][0] * bary[0]
                    + tri["uv"][1] * bary[1]
                    + tri["uv"][2] * bary[2]
                )
                score += (uv - anchor_uv).length * 0.12
            if score < best_score:
                best_score = score
                best_triangle = ti
                best_bary = bary
        if best_triangle is None:
            return None
        return best_triangle, best_bary, best_score

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


def _point_in_uv_triangle(point, a, b, c, eps=1e-5):
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-14:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _uv_inside_island(projector, island_id, uv):
    px = uv.x
    py = uv.y
    for tri_index in projector.island_triangles.get(island_id, ()):
        uvs = projector.triangles[tri_index]["uv"]
        min_x = min(uvs[0].x, uvs[1].x, uvs[2].x) - 1e-5
        if px < min_x:
            continue
        max_x = max(uvs[0].x, uvs[1].x, uvs[2].x) + 1e-5
        if px > max_x:
            continue
        min_y = min(uvs[0].y, uvs[1].y, uvs[2].y) - 1e-5
        if py < min_y:
            continue
        max_y = max(uvs[0].y, uvs[1].y, uvs[2].y) + 1e-5
        if py > max_y:
            continue
        if _point_in_uv_triangle(uv, uvs[0], uvs[1], uvs[2]):
            return True
    return False


def _contain_card_in_island(vertex_uv, group, anchor, projector, island_id, cross_scale):
    """Shrink a rebuilt card toward its anchor until it sits on that chart.

    The anchor stays where the card was sampled. The card is not slid across
    the chart, so a neck feather cannot be moved onto the front of the head.
    """
    del cross_scale
    base = {vi: vertex_uv[vi].copy() for vi in group}

    def _inside(factor):
        for vi in group:
            point = anchor + (base[vi] - anchor) * factor
            if not _uv_inside_island(projector, island_id, point):
                return False
        return True

    if _inside(1.0):
        return
    low = 0.0
    high = 1.0
    best = 0.0
    for _ in range(14):
        mid = (low + high) * 0.5
        if _inside(mid):
            best = mid
            low = mid
        else:
            high = mid
    if best < 0.15:
        best = 0.15
    for vi in group:
        vertex_uv[vi] = anchor + (base[vi] - anchor) * best


def build_body_projector(source_obj, props, depsgraph):
    eval_obj, body_mesh, is_temp = _get_evaluated_mesh(source_obj, depsgraph)
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


def _edge_stretch_ratio(fur_mesh, component, vertex_uv, world_pos):
    """Largest edge scale divided by the smallest. 1 means an even card."""
    group = set(vi for vi in component if vi in vertex_uv and vi in world_pos)
    ratios = []
    seen = set()
    for poly in fur_mesh.polygons:
        verts = poly.vertices
        if not all(vi in group for vi in verts):
            continue
        count = len(verts)
        for i in range(count):
            a = verts[i]
            b = verts[(i + 1) % count]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            world_len = (world_pos[a] - world_pos[b]).length
            uv_len = (vertex_uv[a] - vertex_uv[b]).length
            if world_len > 1e-6 and uv_len > 1e-8:
                ratios.append(uv_len / world_len)
    if len(ratios) < 4:
        return None
    ratios.sort()
    return ratios[-1] / ratios[0]


def _card_needs_repair(fur_mesh, vertex_uv, component, world_pos=None):
    group = [vi for vi in component if vi in vertex_uv]
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
    twisted = pos > 0 and neg > 0
    collapsed = (min(dx, dy) < 0.012 and max(dx, dy) > 0.008) or max(dx, dy) < 0.01
    border_smear = border >= 3 and border >= 0.25 * len(group)
    flat_faces = zero >= 1 and (zero >= max(1, signed) or border_smear)
    stretched = False
    if world_pos is not None:
        ratio = _edge_stretch_ratio(fur_mesh, component, vertex_uv, world_pos)
        stretched = ratio is not None and ratio > 8.0
    return twisted or collapsed or border_smear or flat_faces or stretched


def _sample_surface_uv(projector, world):
    hit = projector.closest_body_triangle(world)
    if hit is None:
        return None
    tri_index, distance = hit
    tri = projector.triangles[tri_index]
    _location, bary = _closest_point_on_triangle(
        world, tri["verts"][0], tri["verts"][1], tri["verts"][2]
    )
    uv = projector.uv_from_triangle(tri_index, bary)
    return distance, tri_index, uv


def _choose_card_anchor(projector, world_pos, component, band=0.003, bin_size=0.08):
    """Lock the card to the chart most of its vertices actually touch.

    A feather tip can be closer to a different body part than the base is to
    the skin it grows from. The closest vertex alone would paint the whole
    card with that other chart.
    """
    samples = []
    for vi in component:
        sampled = _sample_surface_uv(projector, world_pos[vi])
        if sampled is None:
            continue
        distance, tri_index, uv = sampled
        island = projector.triangle_island[tri_index]
        samples.append((distance, vi, tri_index, uv, island))
    if not samples:
        return None

    by_island = defaultdict(list)
    for item in samples:
        by_island[item[4]].append(item)

    def _island_rank(items):
        distances = sorted(item[0] for item in items)
        median = distances[len(distances) // 2]
        # Equal counts: keep the chart the card stands off, not the one a tip touches.
        return (len(items), median)

    home = max(by_island.values(), key=_island_rank)
    home.sort(key=lambda item: item[0])
    min_distance = home[0][0]
    near = [item for item in home if item[0] <= min_distance + band]
    if not near:
        near = home[:1]

    bins = defaultdict(list)
    for item in near:
        uv = item[3]
        key = (int(uv.x / bin_size), int(uv.y / bin_size))
        bins[key].append(item)

    def _bin_rank(items):
        return (len(items), -min(item[0] for item in items))

    best = max(bins.values(), key=_bin_rank)
    best.sort(key=lambda item: item[0])
    _distance, vertex, tri_index, _uv, _island = best[0]
    return vertex, tri_index


def _median_uv_per_world(fur_mesh, component, vertex_uv, world_pos):
    """Median UV length per world length on one card."""
    group = set(vi for vi in component if vi in vertex_uv and vi in world_pos)
    ratios = []
    for poly in fur_mesh.polygons:
        verts = poly.vertices
        if not all(vi in group for vi in verts):
            continue
        count = len(verts)
        for i in range(count):
            a = verts[i]
            b = verts[(i + 1) % count]
            world_len = (world_pos[a] - world_pos[b]).length
            uv_len = (vertex_uv[a] - vertex_uv[b]).length
            if world_len > 1e-6 and uv_len > 1e-8:
                ratios.append(uv_len / world_len)
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def _repair_bad_card(
    fur_mesh, vertex_uv, world_pos, component, projector, root_vertex, root_tri, island,
    target_uv_per_world=None, island_scale=1.0, cross_scale=1.0, force=False,
):
    """Rebuild twisted, line, dot, or border-smeared cards as one planar island.

    Anchored at the root's projected UV. UV units per world unit follow the median
    scale of cards that did not need a rebuild, times island_scale.
    """
    if root_vertex not in vertex_uv or root_vertex not in world_pos or root_tri is None:
        return False
    if not force and not _card_needs_repair(fur_mesh, vertex_uv, component, world_pos):
        return False
    group = [vi for vi in component if vi in vertex_uv and vi in world_pos]
    if len(group) < 3:
        return False

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

    center_w = Vector((0.0, 0.0, 0.0))
    for vi in group:
        center_w += world_pos[vi]
    center_w /= len(group)

    longest = 0.0
    axis_u = None
    for vi in group:
        delta = world_pos[vi] - center_w
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
    for vi in group:
        delta = world_pos[vi] - center_w
        coords.append((vi, delta.dot(axis_u), delta.dot(axis_v)))

    local_uv_per_world = 1.0 / _triangle_texel(projector.triangles[root_tri])
    uv_per_world = target_uv_per_world if target_uv_per_world else local_uv_per_world
    if uv_per_world < local_uv_per_world * 0.35:
        uv_per_world = local_uv_per_world * 0.35
    elif uv_per_world > local_uv_per_world * 2.5:
        uv_per_world = local_uv_per_world * 2.5
    scale = uv_per_world * island_scale

    uv_center = Vector((0.0, 0.0))
    for vi in group:
        uv_center += vertex_uv[vi]
    uv_center /= len(group)
    if not _uv_inside_island(projector, island, uv_center):
        uv_center = anchor

    for vi, s, t in coords:
        vertex_uv[vi] = Vector((uv_center.x + s * scale, uv_center.y + t * scale))

    pos2, neg2, _zero2 = _uv_face_signs(fur_mesh, set(group), vertex_uv)
    if neg2 > pos2:
        for vi, s, t in coords:
            vertex_uv[vi] = Vector((uv_center.x + s * scale, uv_center.y - t * scale))
    _contain_card_in_island(
        vertex_uv, group, uv_center, projector, island, cross_scale
    )
    return True


def _projection_already_placed(fur_mesh, vertex_uv, component, projector, island_id, world_pos=None):
    """True when the body projection is already a solid, even card on its chart.

    Twisted faces, line faces, and cards whose edges stretch far past each
    other still need a rebuild.
    """
    group = [vi for vi in component if vi in vertex_uv]
    if len(group) < 3:
        return False
    xs = [vertex_uv[vi].x for vi in group]
    ys = [vertex_uv[vi].y for vi in group]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    if max(dx, dy) < 0.02 or min(dx, dy) < 0.012:
        return False
    group_set = set(group)
    pos, neg, zero = _uv_face_signs(fur_mesh, group_set, vertex_uv)
    if (pos > 0 and neg > 0) or zero > 0:
        return False
    faces = 0
    thin = 0
    for poly in fur_mesh.polygons:
        verts = poly.vertices
        if not all(vi in group_set for vi in verts):
            continue
        faces += 1
        uvs = [vertex_uv[vi] for vi in verts]
        face_dx = max(u.x for u in uvs) - min(u.x for u in uvs)
        face_dy = max(u.y for u in uvs) - min(u.y for u in uvs)
        if min(face_dx, face_dy) < 0.004:
            thin += 1
    if faces and thin >= 2 and thin >= 0.2 * faces:
        return False
    if world_pos is not None:
        ratio = _edge_stretch_ratio(fur_mesh, component, vertex_uv, world_pos)
        if ratio is not None and ratio > 8.0:
            return False
    for vi in group:
        if not _uv_inside_island(projector, island_id, vertex_uv[vi]):
            return False
    return True


def _component_hits(component, selected_verts):
    if selected_verts is None:
        return True
    return any(vi in selected_verts for vi in component)


def transfer_fur_object(fur_obj, projector, props, stats, depsgraph, selected_verts=None, force_repair=False):
    eval_obj, _eval_me, is_temp = _get_evaluated_mesh(fur_obj, depsgraph)
    fur_mesh = fur_obj.data
    fur_matrix = eval_obj.matrix_world

    world_pos = {v.index: fur_matrix @ v.co for v in fur_mesh.vertices}

    out_uv = _ensure_output_uv_layer(fur_mesh, props.uv_layer_name)
    components = _fur_vertex_components(fur_mesh)
    island_scale = props.island_scale
    cross_scale = props.cross_scale

    def _write_card(vertex_uv):
        for poly in fur_mesh.polygons:
            if not any(vi in vertex_uv for vi in poly.vertices):
                continue
            for loop_index in poly.loop_indices:
                vi = fur_mesh.loops[loop_index].vertex_index
                if vi in vertex_uv:
                    out_uv.data[loop_index].uv = vertex_uv[vi]

    projected = []
    good_scales = []
    for component in components:
        anchor = _choose_card_anchor(projector, world_pos, component)
        if anchor is None:
            stats["skipped"] += 1
            continue
        root_vertex, root_tri = anchor

        chosen_island = projector.triangle_island[root_tri]
        root_sample = _sample_surface_uv(projector, world_pos[root_vertex])
        anchor_uv = root_sample[2] if root_sample is not None else None
        vertex_uv = {}
        for vi in component:
            result = projector.closest_in_island(world_pos[vi], chosen_island, anchor_uv)
            if result is None:
                continue
            tri_index, bary, _dist = result
            vertex_uv[vi] = projector.uv_from_triangle(tri_index, bary)
        if anchor_uv is not None:
            vertex_uv[root_vertex] = anchor_uv

        projected.append((component, vertex_uv, root_vertex, root_tri, chosen_island))
        chosen = _component_hits(component, selected_verts)
        placed = _projection_already_placed(
            fur_mesh, vertex_uv, component, projector, chosen_island, world_pos
        )
        if chosen and force_repair:
            continue
        if placed or not _card_needs_repair(fur_mesh, vertex_uv, component, world_pos):
            scale = _median_uv_per_world(fur_mesh, component, vertex_uv, world_pos)
            if scale:
                good_scales.append(scale)

    good_scales.sort()
    target_scale = good_scales[len(good_scales) // 2] if good_scales else None

    for component, vertex_uv, root_vertex, root_tri, chosen_island in projected:
        if not _component_hits(component, selected_verts):
            continue
        stats["cards"] += 1
        placed = _projection_already_placed(
            fur_mesh, vertex_uv, component, projector, chosen_island, world_pos
        )
        if placed and not force_repair:
            _write_card(vertex_uv)
            continue
        if _repair_bad_card(
            fur_mesh,
            vertex_uv,
            world_pos,
            component,
            projector,
            root_vertex,
            root_tri,
            chosen_island,
            target_scale,
            island_scale,
            cross_scale,
            force=force_repair,
        ):
            stats["repaired"] = stats.get("repaired", 0) + 1
        _write_card(vertex_uv)

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
        description="UV layer written on the feather cards. Other UV layers are left alone",
    )
    body_uv_name: StringProperty(
        name="Body UV",
        default="",
        description="Which UV layer on the undercoat to copy from. Empty = the first UV layer",
    )
    undercoat_object: PointerProperty(
        name="Undercoat",
        type=bpy.types.Object,
        poll=_mesh_object_poll,
        description="Skin mesh that owns the body UV. Transfer sets this from the active object",
    )
    island_scale: FloatProperty(
        name="Island Scale",
        default=1.0,
        min=0.2,
        max=2.0,
        description="Size of rebuilt cards around their anchor. 1 = average of the clean islands. Lower pulls them back in",
    )
    cross_scale: FloatProperty(
        name="Cross Scale",
        default=1.0,
        min=0.2,
        max=1.0,
        description="A rebuilt card is fitted just inside its own chart. Lowering this does not shrink that fit",
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

        props.undercoat_object = source_obj

        depsgraph = context.evaluated_depsgraph_get()
        body_projector = build_body_projector(source_obj, props, depsgraph)
        if body_projector is None:
            self.report({"ERROR"}, "Could not build body UV projector")
            return {"CANCELLED"}

        stats = {"cards": 0, "skipped": 0}
        card_meshes = []
        for obj in targets:
            transfer_fur_object(obj, body_projector, props, stats, depsgraph)
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
        msg = (
            f"{len(card_meshes)} mesh(es), {stats['cards']} components, "
            f"body islands {n_islands}, skipped {stats['skipped']}"
        )
        if total:
            msg += f", dot-cards {dots}/{total}"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class AUF_OT_reassemble_selected(Operator):
    bl_idname = "akelka.reassemble_selected_fur_uv"
    bl_label = "Reassemble Selected"
    bl_description = "Rebuild UndercoatFurUV only on the faces selected in Edit Mode"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.edit_object
        if not obj or obj.type != "MESH" or context.mode != "EDIT_MESH":
            return False
        undercoat = context.scene.auf_props.undercoat_object
        if not undercoat or undercoat.type != "MESH":
            return False
        import bmesh
        bm = bmesh.from_edit_mesh(obj.data)
        return any(face.select for face in bm.faces)

    def execute(self, context):
        import bmesh
        props = context.scene.auf_props
        obj = context.edit_object
        undercoat = props.undercoat_object
        if not undercoat or undercoat.type != "MESH":
            self.report({"ERROR"}, "Set the Undercoat mesh first")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        selected_faces = {face.index for face in bm.faces if face.select}
        selected_verts = {
            vert.index
            for face in bm.faces if face.select
            for vert in face.verts
        }
        if not selected_verts:
            self.report({"ERROR"}, "Select faces to reassemble")
            return {"CANCELLED"}

        before = _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            depsgraph = context.evaluated_depsgraph_get()
            projector = build_body_projector(undercoat, props, depsgraph)
            if projector is None:
                self.report({"ERROR"}, "Could not build undercoat UV projector")
                return {"CANCELLED"}
            stats = {"cards": 0, "skipped": 0, "repaired": 0}
            transfer_fur_object(
                obj, projector, props, stats, depsgraph,
                selected_verts=selected_verts,
                force_repair=True,
            )
            after = _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
            if before != after:
                self.report({"ERROR"}, f"Existing UV layers changed on {obj.name}")
                return {"CANCELLED"}
        finally:
            if context.mode != "EDIT_MESH":
                bpy.ops.object.mode_set(mode="EDIT")
            bm = bmesh.from_edit_mesh(obj.data)
            for face in bm.faces:
                face.select = face.index in selected_faces
            bmesh.update_edit_mesh(obj.data)

        self.report(
            {"INFO"},
            f"Reassembled {stats['cards']} card(s) at scale {props.island_scale:.2f}",
        )
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
        col.label(text="Active: undercoat (skin UV source)", icon="MESH_DATA")
        col.label(text="Selected objects: whole feather meshes", icon="OUTLINER_OB_MESH")
        col.operator("akelka.transfer_undercoat_fur_uv", icon="UV")
        col.operator("akelka.reassemble_selected_fur_uv", icon="UV_SYNC_SELECT")

        settings = layout.box()
        scol = settings.column(align=True)
        scol.prop(props, "uv_layer_name")
        scol.prop(props, "body_uv_name")
        scol.prop(props, "undercoat_object")
        scol.prop(props, "island_scale")
        scol.prop(props, "cross_scale")


classes = (
    AUF_Properties,
    AUF_OT_transfer_undercoat_uv,
    AUF_OT_reassemble_selected,
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
