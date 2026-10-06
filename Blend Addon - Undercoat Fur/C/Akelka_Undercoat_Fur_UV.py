bl_info = {
    "name": "Akelka Undercoat Fur UV",
    "author": "AkELkA",
    "version": (1, 18, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar (N) > Akelka Tools > Undercoat Fur",
    "description": "Add UndercoatFurUV on fur/feather cards via body UV island projection (BG3 shared coloring)",
    "category": "UV",
}

import math

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


def _component_neighbors(fur_mesh, group_set):
    neighbors = defaultdict(set)
    for edge in fur_mesh.edges:
        a, b = edge.vertices
        if a in group_set and b in group_set:
            neighbors[a].add(b)
            neighbors[b].add(a)
    return neighbors


def _island_world_center(projector, island_id):
    cache = getattr(projector, "_island_center", None)
    if cache is None:
        cache = {}
        projector._island_center = cache
    if island_id in cache:
        return cache[island_id]
    acc = Vector((0.0, 0.0, 0.0))
    count = 0
    for tri_index in projector.island_triangles.get(island_id, ()):
        for vert in projector.triangles[tri_index]["verts"]:
            acc += vert
            count += 1
    center = acc / count if count else Vector((0.0, 0.0, 0.0))
    cache[island_id] = center
    return center


def _choose_card_island(projector, world_pos, hits):
    """Pick the chart from the nearer half of the card, not from one touching tip.

    A card split across the left and right charts uses the chart under its center.
    """
    hits.sort()
    half = hits[:max(3, (len(hits) + 1) // 2)]
    counts = {}
    for _dist, island_id, _vi in half:
        counts[island_id] = counts.get(island_id, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    lead = ranked[0][1]
    leaders = [island_id for island_id, count in ranked if count >= lead - 1 and count >= lead * 0.75]
    center = Vector((0.0, 0.0, 0.0))
    for _dist, _island_id, vi in hits:
        center += world_pos[vi]
    center /= len(hits)
    center_hit = projector.closest_body_triangle(center)
    center_island = projector.triangle_island[center_hit[0]] if center_hit else leaders[0]
    def _median_dist(island_id):
        dists = sorted(dist for dist, isl, _vi in half if isl == island_id)
        return dists[len(dists) // 2]

    if len(leaders) >= 2:
        xs = [_island_world_center(projector, island_id).x for island_id in leaders]
        straddles = min(xs) < center.x < max(xs)
        if straddles and center_hit is not None:
            return center_island
        half_winner = min(leaders, key=_median_dist)
    else:
        half_winner = leaders[0]
    # Tips that graze a neighboring chart are closer than the rest of the card,
    # so the nearer half can steal a cheek card onto the eye. If more vertices
    # belong to the chart under the center, and that chart is as close, use it.
    full = {}
    for _dist, island_id, _vi in hits:
        full[island_id] = full.get(island_id, 0) + 1
    if center_hit is not None and center_island != half_winner:
        on_center = projector.closest_in_island(center, center_island)
        on_winner = projector.closest_in_island(center, half_winner)
        center_closer = (
            on_center is not None
            and on_winner is not None
            and on_center[2] <= on_winner[2] * 1.05
        )
        if center_closer and full.get(center_island, 0) > full.get(half_winner, 0):
            return center_island
    return half_winner


def _project_card_v1(projector, world_pos, component):
    """Place the card on the chart under its nearer half, then sample every vertex there."""
    hits = []
    for vi in component:
        hit = projector.closest_body_triangle(world_pos[vi])
        if hit is None:
            continue
        tri_index, distance = hit
        hits.append((distance, projector.triangle_island[tri_index], vi))
    if not hits:
        return None
    island = _choose_card_island(projector, world_pos, hits)
    on_island = [hit for hit in hits if hit[1] == island]
    if on_island:
        root_vertex = min(on_island)[2]
    else:
        center = Vector((0.0, 0.0, 0.0))
        for _dist, _island_id, vi in hits:
            center += world_pos[vi]
        center /= len(hits)
        root_vertex = min(hits, key=lambda hit: (world_pos[hit[2]] - center).length_squared)[2]
    vertex_uv = {}
    for vi in component:
        result = projector.closest_in_island(world_pos[vi], island)
        if result is None:
            continue
        tri_index, bary, _dist = result
        vertex_uv[vi] = projector.uv_from_triangle(tri_index, bary)
    if len(vertex_uv) < 3 or root_vertex not in vertex_uv:
        return None
    return island, vertex_uv, root_vertex


def _median_value(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _uv_pca_off(uvs):
    """How far UV points sit off their best-fit line, compared with the long side."""
    xs = [uv.x for uv in uvs]
    ys = [uv.y for uv in uvs]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    long = max(dx, dy)
    short = min(dx, dy)
    area = 0.0
    for i in range(1, len(uvs) - 1):
        e1 = uvs[i] - uvs[0]
        e2 = uvs[i + 1] - uvs[0]
        area += e1.x * e2.y - e1.y * e2.x
    if long <= 1e-8:
        return 0.0, long, short, area
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    sxx = syy = sxy = 0.0
    for uv in uvs:
        x = uv.x - cx
        y = uv.y - cy
        sxx += x * x
        syy += y * y
        sxy += x * y
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    ca = math.cos(ang)
    sa = math.sin(ang)
    off = 0.0
    for uv in uvs:
        x = uv.x - cx
        y = uv.y - cy
        off = max(off, abs(-sa * x + ca * y))
    return off / long, long, short, area


def _face_world_aspect(world_pos, verts):
    pts = [world_pos[vi] for vi in verts]
    center = Vector((0.0, 0.0, 0.0))
    for point in pts:
        center += point
    center /= len(pts)
    nx = ny = nz = 0.0
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        nx += (a.y - b.y) * (a.z + b.z)
        ny += (a.z - b.z) * (a.x + b.x)
        nz += (a.x - b.x) * (a.y + b.y)
    normal = Vector((nx, ny, nz))
    if normal.length_squared < 1e-20:
        return 1.0
    normal.normalize()
    tangent = Vector((1.0, 0.0, 0.0))
    if abs(normal.x) > 0.9:
        tangent = Vector((0.0, 1.0, 0.0))
    tangent = tangent - normal * tangent.dot(normal)
    if tangent.length_squared < 1e-20:
        return 1.0
    tangent.normalize()
    bitangent = normal.cross(tangent)
    us = [(point - center).dot(tangent) for point in pts]
    vs = [(point - center).dot(bitangent) for point in pts]
    du = max(us) - min(us)
    dv = max(vs) - min(vs)
    long = max(du, dv)
    if long <= 1e-8:
        return 1.0
    return min(du, dv) / long


def _face_defect(uvs, world_pos, verts):
    """A face overlapped onto a line or a dot, plus its UV winding sign."""
    off_ratio, long, short, area = _uv_pca_off(uvs)
    aspect3 = _face_world_aspect(world_pos, verts)
    flat3 = aspect3 < 0.15
    uv_aspect = (short / long) if long > 1e-8 else 0.0
    # A line overlap is flatter than the 3D face, not a card that is already thin in 3D.
    is_line = (not flat3) and (
        long <= 1e-6 or (off_ratio < 0.15 and uv_aspect < aspect3 * 0.5)
    )
    is_dot = abs(area) < 1e-8 or ((not flat3) and long > 1e-6 and uv_aspect < aspect3 * 0.25)
    if area > 1e-8:
        sign = 1
    elif area < -1e-8:
        sign = -1
    else:
        sign = 0
    return is_line, is_dot, sign


def _iter_card_faces(fur_mesh, group_set):
    for poly in fur_mesh.polygons:
        verts = list(poly.vertices)
        if len(verts) >= 3 and all(vi in group_set for vi in verts):
            yield poly.index, verts


def _measure_card(fur_mesh, group_set, vertex_uv, world_pos):
    """Line faces, dot faces, flipped faces, and vertices that sit on a good face."""
    records = []
    pos = neg = 0
    for poly_index, verts in _iter_card_faces(fur_mesh, group_set):
        uvs = [vertex_uv[vi] for vi in verts if vi in vertex_uv]
        if len(uvs) != len(verts):
            continue
        is_line, is_dot, sign = _face_defect(uvs, world_pos, verts)
        records.append((poly_index, verts, is_line, is_dot, sign))
        if not is_line and not is_dot and sign != 0:
            if sign > 0:
                pos += 1
            else:
                neg += 1
    majority = 1 if pos > neg else (-1 if neg > pos else 0)
    lines = dots = minority = 0
    good_verts = set()
    healthy = {}
    for poly_index, verts, is_line, is_dot, sign in records:
        flipped = (
            majority != 0
            and sign != 0
            and sign != majority
            and not is_line
            and not is_dot
        )
        if is_line:
            lines += 1
        if is_dot:
            dots += 1
        if flipped:
            minority += 1
        if not (is_line or is_dot or flipped):
            healthy[poly_index] = sign
            good_verts.update(verts)
    return lines, dots, minority, good_verts, healthy


def _scale_medians(fur_mesh, group_set, vertex_uv, world_pos, good_verts):
    healthy_ratios = []
    loose_ratios = []
    seen = set()
    for _poly_index, verts in _iter_card_faces(fur_mesh, group_set):
        count = len(verts)
        for i in range(count):
            a = verts[i]
            b = verts[(i + 1) % count]
            key = (a, b) if a < b else (b, a)
            if key in seen or a not in vertex_uv or b not in vertex_uv:
                continue
            seen.add(key)
            world_len = (world_pos[a] - world_pos[b]).length
            uv_len = (vertex_uv[a] - vertex_uv[b]).length
            if world_len <= 1e-6 or uv_len <= 1e-5:
                continue
            ratio = uv_len / world_len
            loose_ratios.append(ratio)
            if a in good_verts and b in good_verts:
                healthy_ratios.append(ratio)
    return _median_value(healthy_ratios), _median_value(loose_ratios)


def _island_wraps(projector, island_id, axis):
    cache = getattr(projector, "_wrap_cache", None)
    if cache is None:
        cache = {}
        projector._wrap_cache = cache
    key = (island_id, axis)
    if key in cache:
        return cache[key]
    lo = 1.0
    hi = 0.0
    found = False
    for tri_index in projector.island_triangles.get(island_id, ()):
        for uv in projector.triangles[tri_index]["uv"]:
            value = uv.x if axis == 0 else uv.y
            lo = min(lo, value)
            hi = max(hi, value)
            found = True
    cache[key] = found and lo < 0.15 and hi > 0.85
    return cache[key]


def _on_same_island(uv, projector, island_id):
    """True when uv, or the same point continued across a wrap, lies on this chart."""
    if _uv_inside_island(projector, island_id, uv):
        return True
    for shift_x, shift_y in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        wrapped = Vector((uv.x + shift_x, uv.y + shift_y))
        if _uv_inside_island(projector, island_id, wrapped):
            return True
    return False


def _keep_solved(solved, pin_uv, projector, island_id, source_uv=None):
    """Keep a point on the chart, including one prolonged past a wrapping border."""
    if source_uv is not None:
        dx = abs(solved.x - source_uv.x)
        dy = abs(solved.y - source_uv.y)
        seam_x = abs(dx - 1.0) <= 0.35 and dy <= 0.35
        seam_y = abs(dy - 1.0) <= 0.35 and dx <= 0.35
        if (dx > 0.35 or dy > 0.35) and not (seam_x or seam_y):
            return False
    if _on_same_island(solved, projector, island_id):
        return True
    for axis in (0, 1):
        if not _island_wraps(projector, island_id, axis):
            continue
        pin_c = pin_uv[axis]
        solved_c = solved[axis]
        if pin_c < 0.08 and solved_c < pin_c and (pin_c - solved_c) < 0.5:
            return True
        if pin_c > 0.92 and solved_c > pin_c and (solved_c - pin_c) < 0.5:
            return True
        if pin_c < 0.0 and solved_c < 0.2:
            return True
        if pin_c > 1.0 and solved_c > 0.8:
            return True
    # A short step into empty space past the chart border stays with this card.
    # A step that lands on a different chart does not.
    if (solved - pin_uv).length > 0.2:
        return False
    for other_id in projector.island_triangles:
        if other_id == island_id:
            continue
        if _uv_inside_island(projector, other_id, solved):
            return False
    return True


def _seam_unfold_card(vertex_uv, projector, island_id):
    """Join a card split across a wrapping chart. The piece may sit outside 0-1."""
    shifted = 0
    for axis in (0, 1):
        if not _island_wraps(projector, island_id, axis):
            continue
        coords = sorted((vertex_uv[vi][axis], vi) for vi in vertex_uv)
        if len(coords) < 2:
            continue
        best_gap = 0.0
        best_i = None
        for i in range(len(coords) - 1):
            gap = coords[i + 1][0] - coords[i][0]
            if gap > best_gap:
                best_gap = gap
                best_i = i
        if best_i is None or best_gap <= 0.5:
            continue
        low = [vi for _value, vi in coords[: best_i + 1]]
        high = [vi for _value, vi in coords[best_i + 1 :]]

        def _cluster_on_island(group):
            for vi in group:
                if _uv_inside_island(projector, island_id, vertex_uv[vi]):
                    return True
            return False

        if not _cluster_on_island(low) or not _cluster_on_island(high):
            continue
        if len(low) <= len(high):
            delta = 1.0
            group = low
        else:
            delta = -1.0
            group = high
        for vi in group:
            uv = vertex_uv[vi]
            if axis == 0:
                vertex_uv[vi] = Vector((uv.x + delta, uv.y))
            else:
                vertex_uv[vi] = Vector((uv.x, uv.y + delta))
            shifted += 1
    return shifted


def _pin_vertices(fur_mesh, component, vertex_uv, world_pos):
    """Vertices whose projection is even stay put. Twisted, stretched, and dot verts are free."""
    group = [vi for vi in component if vi in vertex_uv and vi in world_pos]
    group_set = set(group)
    neighbors = _component_neighbors(fur_mesh, group_set)
    edge_ratio = {}
    collapsed = []
    for a in group:
        for b in neighbors.get(a, ()):
            if a > b:
                continue
            world_len = (world_pos[a] - world_pos[b]).length
            uv_len = (vertex_uv[a] - vertex_uv[b]).length
            if world_len <= 1e-6:
                continue
            if uv_len <= 1e-8:
                collapsed.append((a, b))
            else:
                edge_ratio[(a, b)] = uv_len / world_len
    if len(edge_ratio) < 3:
        return set(), None, neighbors

    ordered = sorted(edge_ratio.values())
    median = ordered[len(ordered) // 2]
    if median < 1e-8:
        return set(), None, neighbors

    bad = set()
    for a, b in collapsed:
        bad.add(a)
        bad.add(b)
    for (a, b), ratio in edge_ratio.items():
        if ratio > median * 3.0 or ratio < median / 3.0:
            bad.add(a)
            bad.add(b)

    pos, neg, _zero = _uv_face_signs(fur_mesh, group_set, vertex_uv)
    minority_positive = pos > 0 and neg > pos
    minority_negative = neg > 0 and pos > neg
    for poly in fur_mesh.polygons:
        verts = poly.vertices
        if not all(vi in group_set for vi in verts):
            continue
        uvs = [vertex_uv[vi] for vi in verts]
        area = 0.0
        for i in range(1, len(uvs) - 1):
            e1 = uvs[i] - uvs[0]
            e2 = uvs[i + 1] - uvs[0]
            area += e1.x * e2.y - e1.y * e2.x
        face_dx = max(u.x for u in uvs) - min(u.x for u in uvs)
        face_dy = max(u.y for u in uvs) - min(u.y for u in uvs)
        hairline = abs(area) < 1e-8 or min(face_dx, face_dy) < 0.004
        flipped = (area > 0.0 and minority_positive) or (area < 0.0 and minority_negative)
        if hairline or flipped:
            bad.update(vi for vi in verts if vi in group_set)

    pinned = group_set - bad
    pinned_ratios = [
        ratio for (a, b), ratio in edge_ratio.items()
        if a in pinned and b in pinned
    ]
    if len(pinned_ratios) >= 2:
        pinned_ratios.sort()
        median = pinned_ratios[len(pinned_ratios) // 2]
    return pinned, median, neighbors


def _pull_toward_pin(uv, pin_uv, projector, island_id):
    if _uv_inside_island(projector, island_id, uv):
        return uv
    low = 0.0
    high = 1.0
    best = pin_uv.copy()
    for _ in range(12):
        mid = (low + high) * 0.5
        sample = pin_uv + (uv - pin_uv) * mid
        if _uv_inside_island(projector, island_id, sample):
            best = sample
            low = mid
        else:
            high = mid
    return best


def _uv_centroid(vertex_uv):
    center = Vector((0.0, 0.0))
    for uv in vertex_uv.values():
        center += Vector((uv.x, uv.y))
    return center / max(len(vertex_uv), 1)


def _uv_span(vertex_uv):
    xs = [uv.x for uv in vertex_uv.values()]
    ys = [uv.y for uv in vertex_uv.values()]
    if not xs:
        return 0.0
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _stretch_score(fur_mesh, component, vertex_uv, world_pos):
    group = set(component)
    seen = set()
    ratios = []
    for poly in fur_mesh.polygons:
        verts = poly.vertices
        if not all(vi in group and vi in vertex_uv for vi in verts):
            continue
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            world_len = (world_pos[a] - world_pos[b]).length
            if world_len <= 1e-6:
                continue
            ratios.append((vertex_uv[a] - vertex_uv[b]).length / world_len)
    if len(ratios) < 4:
        return None
    ratios.sort()
    return ratios[-1] / max(ratios[0], 1e-12)


def _place_from_edge(uv_a, uv_b, world_a, world_b, world_c, island_scale, orient):
    """Put C in UV from a known edge AB, copying the 3D triangle at the pin scale."""
    ab_w = world_b - world_a
    ac_w = world_c - world_a
    ab_len = ab_w.length
    ab_uv = uv_b - uv_a
    if ab_len < 1e-8 or ab_uv.length < 1e-8:
        return None
    scale = (ab_uv.length / ab_len) * island_scale
    along = ac_w.dot(ab_w) / ab_len
    normal = ab_w.cross(ac_w)
    perp = Vector((-ab_uv.y, ab_uv.x)).normalized()
    if normal.length_squared < 1e-16:
        return uv_a + ab_uv.normalized() * (along * scale)
    bitangent = normal.cross(ab_w).normalized()
    side = ac_w.dot(bitangent)
    return uv_a + ab_uv.normalized() * (along * scale) + perp * (side * scale * orient)


def _pin_orient(fur_mesh, pinned, vertex_uv, world_pos):
    """Which UV side matches the pinned triangles. +1 keeps the 3D left side."""
    same = opposite = 0
    for poly in fur_mesh.polygons:
        verts = [vi for vi in poly.vertices if vi in pinned and vi in vertex_uv]
        if len(verts) < 3:
            continue
        a, b, c = verts[0], verts[1], verts[2]
        predicted = _place_from_edge(
            vertex_uv[a], vertex_uv[b], world_pos[a], world_pos[b], world_pos[c], 1.0, 1.0
        )
        if predicted is None:
            continue
        ab = vertex_uv[b] - vertex_uv[a]
        actual = vertex_uv[c] - vertex_uv[a]
        guess = predicted - vertex_uv[a]
        cross_actual = ab.x * actual.y - ab.y * actual.x
        cross_guess = ab.x * guess.y - ab.y * guess.x
        if cross_actual * cross_guess > 0.0:
            same += 1
        elif cross_actual * cross_guess < 0.0:
            opposite += 1
    return -1.0 if opposite > same else 1.0


def _principal_axis(points):
    center = Vector((0.0, 0.0, 0.0))
    for point in points:
        center += point
    center /= len(points)
    cov = [[0.0, 0.0, 0.0] for _ in range(3)]
    for point in points:
        delta = point - center
        values = (delta.x, delta.y, delta.z)
        for i in range(3):
            for j in range(3):
                cov[i][j] += values[i] * values[j]
    axis = Vector((1.0, 0.0, 0.0))
    for _ in range(12):
        x = cov[0][0] * axis.x + cov[0][1] * axis.y + cov[0][2] * axis.z
        y = cov[1][0] * axis.x + cov[1][1] * axis.y + cov[1][2] * axis.z
        z = cov[2][0] * axis.x + cov[2][1] * axis.y + cov[2][2] * axis.z
        axis = Vector((x, y, z))
        if axis.length_squared < 1e-20:
            return Vector((0.0, 0.0, 1.0))
        axis.normalize()
    return axis


def _card_normal(fur_mesh, group_set, world_pos):
    normal = Vector((0.0, 0.0, 0.0))
    for _poly_index, verts in _iter_card_faces(fur_mesh, group_set):
        pts = [world_pos[vi] for vi in verts]
        normal += (pts[1] - pts[0]).cross(pts[2] - pts[0])
    if normal.length_squared < 1e-20:
        return Vector((0.0, 0.0, 1.0))
    return normal.normalized()


def _layout_from_root(fur_mesh, component, vertex_uv, world_pos, projector, island_id, root, scale):
    """Open a card that has no healthy face, keeping the v1 root where it is."""
    if root not in vertex_uv or root not in world_pos or not scale:
        return 0
    group = [vi for vi in component if vi in vertex_uv and vi in world_pos]
    if len(group) < 3:
        return 0
    axis = _principal_axis([world_pos[vi] for vi in group])
    bitangent = _card_normal(fur_mesh, set(group), world_pos).cross(axis)
    if bitangent.length_squared < 1e-16:
        bitangent = Vector((0.0, 1.0, 0.0)).cross(axis)
    if bitangent.length_squared < 1e-16:
        return 0
    bitangent.normalize()
    far = max(group, key=lambda vi: (world_pos[vi] - world_pos[root]).length_squared)
    if (world_pos[far] - world_pos[root]).dot(axis) < 0.0:
        axis = -axis
    delta_uv = vertex_uv[far] - vertex_uv[root]
    uv_t = delta_uv.normalized() if delta_uv.length > 1e-6 else Vector((1.0, 0.0))
    uv_b = Vector((-uv_t.y, uv_t.x))
    root_uv = vertex_uv[root].copy()
    moved = 0
    for vi in group:
        if vi == root:
            continue
        delta = world_pos[vi] - world_pos[root]
        solved = root_uv + uv_t * (delta.dot(axis) * scale) + uv_b * (delta.dot(bitangent) * scale)
        if not _keep_solved(solved, root_uv, projector, island_id, vertex_uv[vi]):
            continue
        vertex_uv[vi] = solved
        moved += 1
    return moved


def _healthy_faces_intact(fur_mesh, group_set, vertex_uv, world_pos, healthy):
    for poly_index, sign in healthy.items():
        verts = None
        for index, face_verts in _iter_card_faces(fur_mesh, group_set):
            if index == poly_index:
                verts = face_verts
                break
        if verts is None:
            return False
        uvs = [vertex_uv[vi] for vi in verts if vi in vertex_uv]
        if len(uvs) != len(verts):
            return False
        is_line, is_dot, new_sign = _face_defect(uvs, world_pos, verts)
        if is_line or is_dot or (sign != 0 and new_sign != 0 and new_sign != sign):
            return False
    return True


def _fill_unplaced(
    fur_mesh, group_set, vertex_uv, world_pos, projector, island_id, pinned, scale, orient
):
    """Place free vertices that do not share an edge with two pinned vertices."""
    if not scale or len(pinned) < 2:
        return 0
    best = None
    best_len = -1.0
    for _poly_index, order in _iter_card_faces(fur_mesh, group_set):
        count = len(order)
        for i in range(count):
            a = order[i]
            b = order[(i + 1) % count]
            if a not in pinned or b not in pinned or a not in vertex_uv or b not in vertex_uv:
                continue
            edge_len = (vertex_uv[a] - vertex_uv[b]).length
            if edge_len > best_len:
                best_len = edge_len
                best = (a, b)
    if best is None or best_len < 1e-8:
        return 0
    a, b = best
    ab_w = world_pos[b] - world_pos[a]
    if ab_w.length < 1e-8:
        return 0
    axis = ab_w.normalized()
    uv_t = (vertex_uv[b] - vertex_uv[a]).normalized()
    uv_b = Vector((-uv_t.y, uv_t.x))
    bitangent = _card_normal(fur_mesh, group_set, world_pos).cross(axis)
    if bitangent.length_squared < 1e-16:
        return 0
    bitangent.normalize()
    moved = 0
    anchors = list(pinned)
    for vi in list(vertex_uv):
        if vi in pinned or vi not in world_pos:
            continue
        pin = min(anchors, key=lambda other: (world_pos[vi] - world_pos[other]).length_squared)
        delta = world_pos[vi] - world_pos[pin]
        solved = (
            vertex_uv[pin]
            + uv_t * (delta.dot(axis) * scale)
            + uv_b * (delta.dot(bitangent) * scale * orient)
        )
        if not _keep_solved(solved, vertex_uv[pin], projector, island_id, vertex_uv[vi]):
            continue
        vertex_uv[vi] = solved
        pinned.add(vi)
        moved += 1
    return moved


def _reflect_across_edge(point, a, b):
    ab = b - a
    denom = ab.length_squared
    if denom < 1e-12:
        return point
    t = (point - a).dot(ab) / denom
    projected = a + ab * t
    return projected * 2.0 - point


def _unflip_free(fur_mesh, group_set, vertex_uv, world_pos, locked, projector, island_id):
    """Mirror a free vertex only when that reflection lowers the card's flip count."""
    changed = 0
    for _pass in range(6):
        _lines, _dots, minority, _good, _healthy = _measure_card(
            fur_mesh, group_set, vertex_uv, world_pos
        )
        if minority == 0:
            break
        progress = False
        for _poly_index, verts in _iter_card_faces(fur_mesh, group_set):
            uvs = [vertex_uv[vi] for vi in verts if vi in vertex_uv]
            if len(uvs) != len(verts):
                continue
            _is_line, _is_dot, sign = _face_defect(uvs, world_pos, verts)
            if sign == 0:
                continue
            pos = neg = 0
            for _index, face_verts in _iter_card_faces(fur_mesh, group_set):
                face_uvs = [vertex_uv[vi] for vi in face_verts if vi in vertex_uv]
                if len(face_uvs) != len(face_verts):
                    continue
                line, dot, face_sign = _face_defect(face_uvs, world_pos, face_verts)
                if line or dot or face_sign == 0:
                    continue
                if face_sign > 0:
                    pos += 1
                else:
                    neg += 1
            majority = 1 if pos >= neg else -1
            if sign == majority:
                continue
            for vi in verts:
                if vi in locked or vi not in vertex_uv:
                    continue
                others = [other for other in verts if other != vi and other in vertex_uv]
                if len(others) < 2:
                    continue
                old = vertex_uv[vi].copy()
                solved = _reflect_across_edge(old, vertex_uv[others[0]], vertex_uv[others[1]])
                if not _keep_solved(solved, vertex_uv[others[0]], projector, island_id, old):
                    continue
                vertex_uv[vi] = solved
                _l, _d, new_minor, _g, _h = _measure_card(
                    fur_mesh, group_set, vertex_uv, world_pos
                )
                if new_minor < minority:
                    minority = new_minor
                    changed += 1
                    progress = True
                    break
                vertex_uv[vi] = old
            if progress:
                break
        if not progress:
            break
    return changed


def _island_uv_scale(projector, island_id):
    """Median UV length per world length on this body chart."""
    cache = getattr(projector, "_island_uv_scale", None)
    if cache is None:
        cache = {}
        projector._island_uv_scale = cache
    if island_id in cache:
        return cache[island_id]
    ratios = []
    for tri_index in projector.island_triangles.get(island_id, ()):
        tri = projector.triangles[tri_index]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            uv_len = (tri["uv"][i] - tri["uv"][j]).length
            world_len = (tri["verts"][i] - tri["verts"][j]).length
            if uv_len > 1e-5 and world_len > 1e-8:
                ratios.append(uv_len / world_len)
    ratios.sort()
    scale = ratios[len(ratios) // 2] if ratios else 2.0
    cache[island_id] = scale
    return scale


def _clear_other_islands(solved, root_uv, projector, island_id):
    """Keep a point on this chart or in empty space. Slide it back if it hits another chart."""
    def blocked(uv):
        if _on_same_island(uv, projector, island_id):
            return False
        for other_id in projector.island_triangles:
            if other_id != island_id and _uv_inside_island(projector, other_id, uv):
                return True
        return False

    if not blocked(solved):
        return solved
    best = root_uv.copy()
    low = 0.0
    high = 1.0
    for _step in range(10):
        mid = (low + high) * 0.5
        sample = root_uv + (solved - root_uv) * mid
        if blocked(sample):
            high = mid
        else:
            best = sample
            low = mid
    if (best - root_uv).length < 1e-4:
        return None
    return best


def _uv_spread_axis(vertex_uv, group, root_uv):
    """Direction the projected card already runs in UV."""
    xx = xy = yy = 0.0
    far = 0.0
    for vi in group:
        delta = vertex_uv[vi] - root_uv
        far = max(far, delta.length)
        xx += delta.x * delta.x
        xy += delta.x * delta.y
        yy += delta.y * delta.y
    if far < 1e-4:
        return Vector((1.0, 0.0))
    axis = Vector((1.0, 0.0))
    for _step in range(8):
        axis = Vector((xx * axis.x + xy * axis.y, xy * axis.x + yy * axis.y))
        if axis.length_squared < 1e-16:
            return Vector((1.0, 0.0))
        axis.normalize()
    return axis


def _chart_axis_uv(projector, point, island_id, axis):
    """How this chart's UV moves when the surface moves along axis."""
    base = projector.closest_in_island(point, island_id)
    if base is None:
        return None
    uv0 = projector.uv_from_triangle(base[0], base[1])
    best = None
    step = 0.008
    for direction in (axis, -axis):
        hit = projector.closest_in_island(point + direction * step, island_id)
        if hit is None:
            continue
        delta = projector.uv_from_triangle(hit[0], hit[1]) - uv0
        if best is None or delta.length_squared > best[0]:
            best = (delta.length_squared, delta, direction)
    if best is None or best[0] < 1e-8:
        return None
    sign = 1.0 if best[2].dot(axis) >= 0.0 else -1.0
    return (best[1] * sign).normalized()


def _open_collapsed_card(
    fur_mesh, component, vertex_uv, world_pos, projector, island_id, island_scale, root
):
    """Lay a line or dot card back out at the body chart's scale, from its root.

    Returns a UV dict when the line count drops, otherwise None. vertex_uv is restored.
    """
    group = [vi for vi in component if vi in vertex_uv and vi in world_pos]
    group_set = set(group)
    if root not in group_set or len(group) < 3:
        return None
    lines, dots, minority, _good, _healthy = _measure_card(
        fur_mesh, group_set, vertex_uv, world_pos
    )
    axis = _principal_axis([world_pos[vi] for vi in group])
    uv_t = _chart_axis_uv(projector, world_pos[root], island_id, axis)
    if uv_t is None:
        uv_t = _uv_spread_axis(vertex_uv, group, vertex_uv[root])
    uv_b = Vector((-uv_t.y, uv_t.x))
    bitangent = _card_normal(fur_mesh, group_set, world_pos).cross(axis)
    if bitangent.length_squared < 1e-16:
        return None
    bitangent.normalize()
    ratio = _island_uv_scale(projector, island_id) * island_scale
    if ratio < 1e-6:
        return None
    original = {vi: vertex_uv[vi].copy() for vi in group}
    root_uv = original[root].copy()
    max_len = 0.05
    for vi in group:
        max_len = max(max_len, (world_pos[vi] - world_pos[root]).length * ratio * 1.25)

    def _attempt(orient):
        for vi, uv in original.items():
            vertex_uv[vi] = uv.copy()
        moved = 0
        for vi in group:
            if vi == root:
                continue
            delta = world_pos[vi] - world_pos[root]
            solved = (
                root_uv
                + uv_t * (delta.dot(axis) * ratio)
                + uv_b * (delta.dot(bitangent) * ratio * orient)
            )
            if (solved - root_uv).length > max_len:
                continue
            solved = _clear_other_islands(solved, root_uv, projector, island_id)
            if solved is None:
                continue
            vertex_uv[vi] = solved
            moved += 1
        return moved

    best_uv = None
    best_score = lines + dots + minority
    for orient in (1.0, -1.0):
        if not _attempt(orient):
            continue
        for _flip_pass in range(3):
            flipped = _unflip_free(
                fur_mesh, group_set, vertex_uv, world_pos, {root}, projector, island_id
            )
            if not flipped:
                break
        after_lines, after_dots, after_minor, _good, _healthy = _measure_card(
            fur_mesh, group_set, vertex_uv, world_pos
        )
        score = after_lines + after_dots + after_minor
        if (
            after_lines < lines
            and after_dots <= dots
            and after_minor <= minority
            and score < best_score
        ):
            best_score = score
            best_uv = {vi: vertex_uv[vi].copy() for vi in group}
    for vi, uv in original.items():
        vertex_uv[vi] = uv.copy()
    return best_uv


def _uv_outliers(vertex_uv):
    """Vertices that sit far from every other vertex, a spike off the card."""
    verts = list(vertex_uv)
    if len(verts) < 4:
        return set()
    nearest = {}
    for vi in verts:
        ui = vertex_uv[vi]
        best = None
        for other in verts:
            if other == vi:
                continue
            dist = (ui - vertex_uv[other]).length
            if best is None or dist < best:
                best = dist
        nearest[vi] = 0.0 if best is None else best
    ordered = sorted(nearest.values())
    median = ordered[len(ordered) // 2]
    limit = max(0.08, median * 3.0)
    return {vi for vi, dist in nearest.items() if dist > limit}


def _shape_uv_layer(fur_mesh, output_name):
    """The card's own UV, used only to keep a jumped vertex beside its neighbors."""
    for layer in fur_mesh.uv_layers:
        if layer.name != output_name and len(layer.data):
            return layer
    return None


def _shape_uvs(fur_mesh, shape_layer):
    found = {}
    if shape_layer is None:
        return found
    for loop_index, loop in enumerate(fur_mesh.loops):
        vi = loop.vertex_index
        if vi not in found:
            found[vi] = shape_layer.data[loop_index].uv.copy()
    return found


def _segments_cross(a, b, c, d):
    def _orient(p, q, r):
        return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

    ab_c = _orient(a, b, c)
    ab_d = _orient(a, b, d)
    cd_a = _orient(c, d, a)
    cd_b = _orient(c, d, b)
    return ((ab_c > 1e-9 and ab_d < -1e-9) or (ab_c < -1e-9 and ab_d > 1e-9)) and (
        (cd_a > 1e-9 and cd_b < -1e-9) or (cd_a < -1e-9 and cd_b > 1e-9)
    )


def _crossing_count(fur_mesh, group_set, uv_of):
    """How many non-touching edges of this card cross in UV."""
    edges = []
    seen = set()
    for _poly_index, verts in _iter_card_faces(fur_mesh, group_set):
        count = len(verts)
        for i in range(count):
            a = verts[i]
            b = verts[(i + 1) % count]
            if a not in uv_of or b not in uv_of:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)
    crossings = 0
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1 :]:
            if a == c or a == d or b == c or b == d:
                continue
            if _segments_cross(uv_of[a], uv_of[b], uv_of[c], uv_of[d]):
                crossings += 1
    return crossings


def _shape_is_one_piece(shape, group):
    present = [vi for vi in group if vi in shape]
    if len(present) < 4:
        return False
    for axis in (0, 1):
        coords = sorted(shape[vi][axis] for vi in present)
        for i in range(len(coords) - 1):
            if coords[i + 1] - coords[i] > 0.35:
                return False
    return True


def _similarity_map(src_points, dst_points, mirror):
    """Map the card's own UV onto the projected one without folding it."""
    count = len(src_points)
    src_center = Vector((0.0, 0.0))
    dst_center = Vector((0.0, 0.0))
    for src, dst in zip(src_points, dst_points):
        src_center += src
        dst_center += dst
    src_center /= count
    dst_center /= count
    align = 0.0
    turn = 0.0
    src_len = 0.0
    for src, dst in zip(src_points, dst_points):
        sx = src.x - src_center.x
        sy = src.y - src_center.y
        if mirror:
            sy = -sy
        tx = dst.x - dst_center.x
        ty = dst.y - dst_center.y
        align += sx * tx + sy * ty
        turn += sx * ty - sy * tx
        src_len += sx * sx + sy * sy
    if src_len < 1e-12:
        return None
    length = math.hypot(align, turn)
    if length < 1e-12:
        cos_t = 1.0
        sin_t = 0.0
        scale = 1.0
    else:
        cos_t = align / length
        sin_t = turn / length
        scale = length / src_len
    if scale < 1e-4 or scale > 40.0:
        return None

    def _apply(src):
        sx = src.x - src_center.x
        sy = src.y - src_center.y
        if mirror:
            sy = -sy
        rx = sx * cos_t - sy * sin_t
        ry = sx * sin_t + sy * cos_t
        return dst_center + Vector((rx * scale, ry * scale))

    return _apply


def _best_similarity(src_of, dst_of, group):
    src = [src_of[vi] for vi in group]
    dst = [dst_of[vi] for vi in group]
    best = None
    best_error = None
    for mirror in (False, True):
        mapped = _similarity_map(src, dst, mirror)
        if mapped is None:
            continue
        error = 0.0
        for src_pt, dst_pt in zip(src, dst):
            error += (mapped(src_pt) - dst_pt).length_squared
        if best_error is None or error < best_error:
            best_error = error
            best = mapped
    if best is None:
        return None
    return {vi: best(src_of[vi]) for vi in group}


def _plane_coords(fur_mesh, group, world_pos):
    """Flatten the card onto its own plane. A flat card cannot cross itself."""
    present = [vi for vi in group if vi in world_pos]
    if len(present) < 4:
        return None
    points = [world_pos[vi] for vi in present]
    axis = _principal_axis(points)
    normal = _card_normal(fur_mesh, set(present), world_pos)
    bitangent = normal.cross(axis)
    if bitangent.length_squared < 1e-16:
        return None
    bitangent.normalize()
    center = Vector((0.0, 0.0, 0.0))
    for point in points:
        center += point
    center /= len(points)
    return {
        vi: Vector((
            (world_pos[vi] - center).dot(axis),
            (world_pos[vi] - center).dot(bitangent),
        ))
        for vi in present
    }


def _untwist_card(fur_mesh, component, vertex_uv, world_pos, output_name):
    """Replace a folded UV with an unfolded shape, held on the projected spot."""
    group = [vi for vi in component if vi in vertex_uv]
    group_set = set(group)
    if len(group) < 4:
        return 0
    before = _crossing_count(fur_mesh, group_set, vertex_uv)
    if before == 0:
        return 0
    dst = {vi: Vector((vertex_uv[vi].x, vertex_uv[vi].y)) for vi in group}
    sources = []
    shape = _shape_uvs(fur_mesh, _shape_uv_layer(fur_mesh, output_name))
    if _shape_is_one_piece(shape, group) and _crossing_count(fur_mesh, group_set, shape) == 0:
        sources.append({vi: Vector((shape[vi].x, shape[vi].y)) for vi in group})
    plane = _plane_coords(fur_mesh, group, world_pos)
    if plane is not None and len(plane) == len(group):
        sources.append(plane)
    best_uv = None
    best_cross = before
    for src in sources:
        fitted = _best_similarity(src, dst, group)
        if fitted is None:
            continue
        crossings = _crossing_count(fur_mesh, group_set, fitted)
        if crossings < best_cross:
            best_cross = crossings
            best_uv = fitted
    if best_uv is None:
        return 0
    for vi, uv in best_uv.items():
        vertex_uv[vi] = uv
    return before - best_cross


def _pull_spikes(fur_mesh, component, vertex_uv, world_pos, output_name):
    """Move a vertex that jumped away back beside the rest of its card."""
    shape = _shape_uvs(fur_mesh, _shape_uv_layer(fur_mesh, output_name))
    moved = 0
    group = [vi for vi in component if vi in vertex_uv and vi in world_pos]
    for _pass in range(4):
        outliers = _uv_outliers(vertex_uv)
        main = [vi for vi in group if vi not in outliers]
        if not outliers or len(main) < 2:
            break
        ratios = []
        main_set = set(main)
        for poly in fur_mesh.polygons:
            verts = list(poly.vertices)
            if len(verts) < 3 or not all(vi in main_set for vi in verts):
                continue
            for i, a in enumerate(verts):
                b = verts[(i + 1) % len(verts)]
                if a > b or a not in shape or b not in shape:
                    continue
                shape_len = (shape[a] - shape[b]).length
                if shape_len < 1e-6:
                    continue
                ratios.append((vertex_uv[a] - vertex_uv[b]).length / shape_len)
        ratios.sort()
        shape_scale = ratios[len(ratios) // 2] if ratios else None
        center = Vector((0.0, 0.0))
        for vi in main:
            center += Vector((vertex_uv[vi].x, vertex_uv[vi].y))
        center /= len(main)
        progress = False
        for vi in list(outliers):
            if vi not in vertex_uv:
                continue
            anchors = sorted(main, key=lambda other: (world_pos[vi] - world_pos[other]).length_squared)
            a, b = anchors[0], anchors[1]
            current = (vertex_uv[vi] - center).length
            best = None
            best_dist = current
            for orient in (1.0, -1.0):
                solved = _place_from_edge(
                    vertex_uv[a], vertex_uv[b],
                    world_pos[a], world_pos[b], world_pos[vi],
                    1.0, orient,
                )
                if solved is None:
                    continue
                if (
                    shape_scale
                    and vi in shape
                    and a in shape
                    and (shape[vi] - shape[a]).length > 1e-6
                ):
                    wanted = (shape[vi] - shape[a]).length * shape_scale
                    offset = solved - vertex_uv[a]
                    if offset.length > 1e-8:
                        solved = vertex_uv[a] + offset.normalized() * wanted
                dist = (solved - center).length
                if dist + 1e-6 < best_dist:
                    best_dist = dist
                    best = solved
            if best is None:
                continue
            vertex_uv[vi] = best
            moved += 1
            progress = True
        if not progress:
            break
    return moved


def _reunwrap_free_verts(
    fur_mesh, component, vertex_uv, world_pos, projector, island_id, island_scale, root, scale
):
    """Move twisted, stretched, line, and dot vertices. Healthy UVs stay on the v1 chart."""
    group_set = set(vi for vi in component if vi in vertex_uv and vi in world_pos)
    lines, dots, minority, good_verts, healthy = _measure_card(
        fur_mesh, group_set, vertex_uv, world_pos
    )
    if lines == 0 and dots == 0 and minority == 0:
        return 0
    outliers = _uv_outliers({vi: vertex_uv[vi] for vi in group_set})
    if outliers:
        good_verts = set(good_verts) - outliers
        if root in outliers and root in world_pos:
            main = [vi for vi in group_set if vi not in outliers and vi in world_pos]
            if main:
                origin = world_pos[root]
                root = min(main, key=lambda vi: (world_pos[vi] - origin).length_squared)
    original = {vi: vertex_uv[vi].copy() for vi in group_set}
    face_total = 0
    bad_faces = 0
    for _poly_index, verts in _iter_card_faces(fur_mesh, group_set):
        face_total += 1
        uvs = [vertex_uv[vi] for vi in verts if vi in vertex_uv]
        if len(uvs) != len(verts):
            continue
        is_line, is_dot, _sign = _face_defect(uvs, world_pos, verts)
        if is_line or is_dot:
            bad_faces += 1
    opened_uv = None
    if face_total and bad_faces * 2 >= face_total:
        opened_uv = _open_collapsed_card(
            fur_mesh, component, vertex_uv, world_pos, projector, island_id, island_scale, root
        )
    pinned = set(good_verts)
    if root in group_set:
        pinned.add(root)
    if not good_verts and root not in group_set:
        pinned = set()
    free = [vi for vi in group_set if vi not in pinned]
    if not free:
        if opened_uv is not None:
            for vi, uv in opened_uv.items():
                vertex_uv[vi] = uv
            return 1
        return 0

    def _restore():
        for vi, uv in original.items():
            vertex_uv[vi] = uv.copy()

    def _attempt(orient):
        _restore()
        if not good_verts:
            moved = _layout_from_root(
                fur_mesh, component, vertex_uv, world_pos, projector, island_id, root, scale
            )
            for _flip_pass in range(3):
                flipped = _unflip_free(
                    fur_mesh, group_set, vertex_uv, world_pos, {root}, projector, island_id
                )
                moved += flipped
                if not flipped:
                    break
            return moved
        placed = set(pinned)
        moved = 0
        for _pass in range(len(free) + 1):
            pending = {}
            for _poly_index, order in _iter_card_faces(fur_mesh, group_set):
                free_here = [vi for vi in order if vi not in placed]
                if not free_here:
                    continue
                best = None
                best_len = -1.0
                count = len(order)
                for i in range(count):
                    a = order[i]
                    b = order[(i + 1) % count]
                    if a not in placed or b not in placed:
                        continue
                    edge_len = (vertex_uv[a] - vertex_uv[b]).length
                    if edge_len > best_len:
                        best_len = edge_len
                        best = (a, b)
                if best is None or best_len < 1e-8:
                    continue
                a, b = best
                for vi in free_here:
                    if vi in pending and pending[vi][0] >= best_len:
                        continue
                    solved = _place_from_edge(
                        vertex_uv[a], vertex_uv[b],
                        world_pos[a], world_pos[b], world_pos[vi],
                        island_scale, orient,
                    )
                    if solved is None or not _keep_solved(solved, vertex_uv[a], projector, island_id, original[vi]):
                        continue
                    pending[vi] = (best_len, solved)
            if not pending:
                break
            for vi, (_edge_len, solved) in pending.items():
                vertex_uv[vi] = solved
                placed.add(vi)
                moved += 1
        moved += _fill_unplaced(
            fur_mesh, group_set, vertex_uv, world_pos, projector, island_id, placed, scale, orient
        )
        for _flip_pass in range(3):
            flipped = _unflip_free(
                fur_mesh, group_set, vertex_uv, world_pos, pinned, projector, island_id
            )
            moved += flipped
            if not flipped:
                break
        return moved

    orients = (1.0,)
    if good_verts:
        base_orient = _pin_orient(fur_mesh, pinned, vertex_uv, world_pos)
        orients = (base_orient, -base_orient)
    best_moved = 0
    best_uv = None
    best_score = lines + dots + minority
    for orient in orients:
        moved = _attempt(orient)
        after_lines, after_dots, after_minor, _good, _healthy = _measure_card(
            fur_mesh, group_set, vertex_uv, world_pos
        )
        score = after_lines + after_dots + after_minor
        intact = _healthy_faces_intact(fur_mesh, group_set, vertex_uv, world_pos, healthy)
        if (
            moved
            and score < best_score
            and after_lines < lines
            and after_dots <= dots
            and after_minor <= minority
            and intact
        ):
            best_score = score
            best_moved = moved
            best_uv = {vi: vertex_uv[vi].copy() for vi in group_set}
    if opened_uv is not None:
        for vi, uv in opened_uv.items():
            vertex_uv[vi] = uv
        open_lines, open_dots, open_minor, _open_good, _open_healthy = _measure_card(
            fur_mesh, group_set, vertex_uv, world_pos
        )
        open_score = open_lines + open_dots + open_minor
        if best_uv is None or open_score < best_score:
            best_uv = opened_uv
            best_moved = max(best_moved, 1)
    if best_uv is None:
        _restore()
        return 0
    for vi, uv in best_uv.items():
        vertex_uv[vi] = uv
    return best_moved


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

    def _write_card(vertex_uv):
        for poly in fur_mesh.polygons:
            if not any(vi in vertex_uv for vi in poly.vertices):
                continue
            for loop_index in poly.loop_indices:
                vi = fur_mesh.loops[loop_index].vertex_index
                if vi in vertex_uv:
                    out_uv.data[loop_index].uv = vertex_uv[vi]

    prepared = []
    island_scales = defaultdict(list)
    for component in components:
        if not _component_hits(component, selected_verts):
            continue
        projected = _project_card_v1(projector, world_pos, component)
        if projected is None:
            stats["skipped"] += 1
            continue
        island, vertex_uv, root = projected
        _seam_unfold_card(vertex_uv, projector, island)
        group_set = set(vi for vi in component if vi in vertex_uv)
        _lines, _dots, _minor, good_verts, _healthy = _measure_card(
            fur_mesh, group_set, vertex_uv, world_pos
        )
        healthy_median, loose_median = _scale_medians(
            fur_mesh, group_set, vertex_uv, world_pos, good_verts
        )
        scale = healthy_median if healthy_median is not None else loose_median
        if scale is not None:
            island_scales[island].append(scale)
        prepared.append((component, island, vertex_uv, root, scale))

    for component, island, vertex_uv, root, scale in prepared:
        if scale is None:
            scale = _median_value(island_scales.get(island, ()))
        fit_scale = island_scale * props.cross_scale
        if scale is not None:
            scale *= fit_scale
        stats["cards"] += 1
        moved = _reunwrap_free_verts(
            fur_mesh, component, vertex_uv, world_pos, projector, island,
            fit_scale, root, scale,
        )
        if moved:
            stats["repaired"] = stats.get("repaired", 0) + 1
        _pull_spikes(fur_mesh, component, vertex_uv, world_pos, props.uv_layer_name)
        _untwist_card(fur_mesh, component, vertex_uv, world_pos, props.uv_layer_name)
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
        description="Size of cards that were opened back out from a line or a dot. 1 keeps the body chart size",
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

        bpy.ops.object.mode_set(mode="OBJECT")
        # Edit Mode does not expose UV loop data, so a hash taken there is
        # always empty and the check below would cancel every run.
        before = _mesh_uv_hashes(obj.data, skip_name=props.uv_layer_name)
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
            f"Reassembled {stats['cards']} card(s), island {props.island_scale:.2f}, cross {props.cross_scale:.2f}",
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
