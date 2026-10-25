"""Append Body / Body+cloth mesh kits onto a GR2 LAB armature.

Library lives in addon ``mesh_kits/GR2LAB_mesh_kits.blend`` (or prefs path).
The AkELkA toggle swaps Dragonborn onto extra DGB kits; other races stay usual.
Head kits are authored against a face armature. Attach merges leftover face
weights onto the body bones that exist (jaw/lips/lids -> Head_M). Does not
bake, densify, or encode.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import bpy
from mathutils import Euler, Matrix, Vector

_CATALOG_NAME = "catalog.json"
_LIBRARY_NAME = "GR2LAB_mesh_kits.blend"
_MESH_KEY = "gr2lab_mesh_kit"


def _addon_mesh_kits_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "mesh_kits"


def _prefs_mesh_kits_dir(context=None) -> Path:
    ctx = context or bpy.context
    try:
        prefs = ctx.preferences.addons["gr2lab_blender"].preferences
        raw = (getattr(prefs, "mesh_kits_dir", "") or "").strip()
        if raw:
            return Path(raw)
    except Exception:
        pass
    return Path()


def mesh_kits_dir(context=None) -> Path:
    pref = _prefs_mesh_kits_dir(context)
    if pref and (pref / _LIBRARY_NAME).is_file():
        return pref
    if pref.is_file() and pref.name.lower().endswith(".blend"):
        return pref.parent
    return _addon_mesh_kits_dir()


def library_path(context=None) -> Optional[Path]:
    folder = mesh_kits_dir(context)
    p = folder / _LIBRARY_NAME
    if p.is_file():
        return p
    if folder.is_file() and folder.suffix.lower() == ".blend":
        return folder
    return None


def load_catalog(context=None) -> Dict[str, Any]:
    folder = mesh_kits_dir(context)
    cat = folder / _CATALOG_NAME
    if not cat.is_file():
        bundled = _addon_mesh_kits_dir() / _CATALOG_NAME
        cat = bundled if bundled.is_file() else cat
    if not cat.is_file():
        return {"library": _LIBRARY_NAME, "bases": {}}
    try:
        data = json.loads(cat.read_text(encoding="utf-8"))
    except Exception:
        return {"library": _LIBRARY_NAME, "bases": {}}
    if not isinstance(data, dict):
        return {"library": _LIBRARY_NAME, "bases": {}}
    data.setdefault("bases", {})
    return data


def normalize_base_id(raw: Optional[str]) -> str:
    stem = Path(str(raw or "").strip()).name
    if not stem:
        return ""
    if stem.lower().endswith(".gr2"):
        stem = Path(stem).stem
    if stem.lower().endswith(".bind_pose.json"):
        stem = stem[: -len(".bind_pose.json")]
    return stem


_AKELKA_ATTACH_MODES = frozenset({"BODY", "BODY_CLOTH"})


def normalize_mesh_mode(raw: Optional[str]) -> str:
    mode = str(raw or "NONE").strip().upper().replace("+", "_").replace(" ", "_")
    if mode in {"BODY_CLOTH", "CLOTH", "BODY_PLUS_CLOTH"}:
        return "BODY_CLOTH"
    if mode == "BODY":
        return "BODY"
    return "NONE"


def is_dragonborn_base(ident: Optional[str]) -> bool:
    stem = normalize_base_id(ident).upper()
    return stem.startswith("DGB_")


def uses_akelka_kit(
    ident: Optional[str],
    mode: Optional[str],
    akelka: bool = False,
) -> bool:
    """AkELkA kits replace usual DGB body meshes when the toggle is on."""
    if not akelka:
        return False
    if normalize_mesh_mode(mode) not in _AKELKA_ATTACH_MODES:
        return False
    return is_dragonborn_base(ident)


# No elf/half-elf kits yet — same bone names as human; human head (no pointed ears).
_MESH_KIT_HUMAN_FALLBACK: Dict[str, str] = {
    "ELF_M": "HUM_M_Base",
    "ELF_F": "HUM_F_Base",
    "ELF_MS": "HUM_MS_Base",
    "ELF_FS": "HUM_FS_Base",
    "HEL_M": "HUM_M_Base",
    "HEL_F": "HUM_F_Base",
}


def _rig_family_for_kit(ident: str, template_path: Optional[str] = None) -> str:
    stem = normalize_base_id(ident)
    # An explicit body (DGB_F_Base) wins over the clip name. A borrowed
    # DGB_M file must not pull the male kit back onto a DGB_F swap.
    if stem.lower().endswith("_base"):
        return stem[: -len("_Base")] if stem.endswith("_Base") else stem[: -len("_base")]
    if template_path:
        try:
            from rig_catalog import classify_rig  # type: ignore

            fam = classify_rig(Path(template_path).name).get("rig_family")
            if fam:
                return str(fam)
        except Exception:
            pass
    if "_Rig_" in stem:
        try:
            from rig_catalog import classify_rig  # type: ignore

            fam = classify_rig(f"{stem}.GR2").get("rig_family")
            if fam:
                return str(fam)
        except Exception:
            pass
    return ""


def _catalog_base_for_kit(
    ident: str,
    cat: Dict[str, Any],
    *,
    template_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve catalog key; optional human-kit fallback for missing race kits."""
    bases = cat.get("bases") or {}

    def _ok(key: str) -> bool:
        ent = bases.get(key) or {}
        return bool(ent) and not ent.get("missing")

    raw = normalize_base_id(ident)
    fam = _rig_family_for_kit(ident, template_path)
    candidates: List[str] = []
    if raw.lower().endswith("_base"):
        candidates.append(raw)
    if fam:
        candidates.append(f"{fam}_Base")
    for key in candidates:
        if _ok(key):
            return key, None
    fb = _MESH_KIT_HUMAN_FALLBACK.get(fam) if fam else None
    if fb and _ok(fb):
        return fb, fam or raw
    return (candidates[0] if candidates else raw), None


def resolve_kit_base(
    raw: Optional[str] = None,
    *,
    doc: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
) -> str:
    ident = normalize_base_id(raw)
    if ident:
        return ident
    data = doc if isinstance(doc, dict) else {}
    for key in ("body_base", "source_gr2", "source"):
        ident = normalize_base_id(data.get(key) if isinstance(data.get(key), str) else "")
        if ident:
            if ident.lower().endswith("_base"):
                return ident
            break
    src = (
        (data.get("source_gr2") if isinstance(data.get("source_gr2"), str) else "")
        or (data.get("source") if isinstance(data.get("source"), str) else "")
        or (template_path or "")
    )
    try:
        from bind_pose import match_body_base_for_source  # type: ignore

        matched = match_body_base_for_source(Path(src).name)
        ident = normalize_base_id(matched or "")
        if ident:
            return ident
    except Exception:
        pass
    return normalize_base_id(src)


def _mesh_aabb_size(obj) -> Tuple[float, float, float]:
    mesh = getattr(obj, "data", None)
    verts = getattr(mesh, "vertices", None) if mesh is not None else None
    if not verts:
        return (0.0, 0.0, 0.0)
    xs = [v.co.x for v in verts]
    ys = [v.co.y for v in verts]
    zs = [v.co.z for v in verts]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def _aabb_y_up(obj) -> bool:
    _dx, dy, dz = _mesh_aabb_size(obj)
    return dy > dz * 1.15


def _aabb_z_up(obj) -> bool:
    _dx, dy, dz = _mesh_aabb_size(obj)
    return dz > dy * 1.15


def _mesh_data_z_up(obj) -> bool:
    """True when vertex data already stands (height along +Z).

    Pants / low cloth fail the centroid-height test (waist sits under 0.5m)
    but their AABB is still taller in Z than Y.
    """
    if _centroid_y_up(obj):
        return False
    if _centroid_z_up(obj):
        return True
    return _aabb_z_up(obj)


def _name_l(obj) -> str:
    return (obj.name or "").lower()


_HEAD_ACC_TAGS = ("horn", "ear", "hair", "eye", "crest")
_SKIP_MESH_TAGS = ("tearline", "eyeshadow", "mouth_interior")
_UPPER_CLOTH_TAGS = ("shirt", "robe", "tunic", "armor", "jacket", "vest")


def _is_lod(obj) -> bool:
    return "_lod" in _name_l(obj)


def _is_skipped_kit_mesh(obj) -> bool:
    n = _name_l(obj)
    return any(tag in n for tag in _SKIP_MESH_TAGS)


def _is_upper_cloth(obj) -> bool:
    """Shirt-like cloth that covers the chest hide-mask."""
    n = _name_l(obj)
    if "clt_" not in n:
        return False
    return any(tag in n for tag in _UPPER_CLOTH_TAGS)


_FACE_WEIGHT_TAGS = (
    "jaw",
    "lip",
    "chin",
    "gullet",
    "tongue",
    "teeth",
    "brow",
    "lid",
    "orbit",
    "cheek",
    "jugal",
    "nose",
    "nostril",
    "beard",
    "piercing",
    "eyelid",
    "philtrum",
    "temple",
    "forehead",
    "nasol",
    "eye_l_",
    "eye_r_",
    "eyeorbit",
)
_WEIGHT_ALIASES = {
    "HeadEnd_M_endBone": "Head_M",
    "Dummy_HeadFX": "Head_M",
    "Dummy_MouthFX": "Head_M",
    "Dummy_EyeFX_01": "Head_M",
    "Dummy_EyeFX_02": "Head_M",
    "Dummy_StatusFX": "Head_M",
    "Dummy_OverheadFX": "Head_M",
    "Dummy_NeckFX": "Neck_M",
    "Tail8_endBone": "Tail8_M_endBone",
}
# Lie-down clips omit the twist bones the standing body kit is painted to.
# Walk these onto the deform bone that is actually in the rig.
_MISSING_BONE_PARENTS = {
    "Hip_L_Twist_01": "Hip_L",
    "Hip_L_Twist_02": "Hip_L_Twist_01",
    "Hip_R_Twist_01": "Hip_R",
    "Hip_R_Twist_02": "Hip_R_Twist_01",
    "Shoulder_L_Twist_01": "Shoulder_L",
    "Shoulder_L_Twist_02": "Shoulder_L_Twist_01",
    "Shoulder_R_Twist_01": "Shoulder_R",
    "Shoulder_R_Twist_02": "Shoulder_R_Twist_01",
    "Elbow_Twist_L": "Elbow_L",
    "Elbow_Twist_R": "Elbow_R",
}


def _body_base_bone_names() -> Set[str]:
    names: Set[str] = set()
    folder = Path(__file__).resolve().parent.parent / "Body_Base"
    if not folder.is_dir():
        return names
    for path in folder.glob("*.bind_pose.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for bone in data.get("bones") or []:
            name = bone.get("name")
            if name:
                names.add(str(name))
    return names


def _parent_map_from_armature(arm) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if arm is None or getattr(arm, "type", None) != "ARMATURE":
        return out
    for bone in arm.data.bones:
        out[bone.name] = bone.parent.name if bone.parent else ""
    return out


def _looks_like_face_bone(name: str) -> bool:
    n = (name or "").lower()
    if n in ("eye_l", "eye_r", "eye"):
        return True
    return any(tag in n for tag in _FACE_WEIGHT_TAGS)


def _is_mask_weight_group(name: str) -> bool:
    n = (name or "").lower()
    return n in {"mask", "group"} or "mask" in n


def _mask_modifier_groups(obj) -> Set[str]:
    names: Set[str] = set()
    for mod in getattr(obj, "modifiers", []) or []:
        if getattr(mod, "type", "") != "MASK":
            continue
        vg = getattr(mod, "vertex_group", "") or ""
        if vg:
            names.add(vg)
    return names


def _weight_dest(
    name: str,
    keep: Set[str],
    parent_of: Dict[str, str],
    default: Optional[str],
) -> Optional[str]:
    if name in keep:
        return name
    if name in ("Tail8_endBone", "Tail8_M_endBone"):
        for alt in ("Tail8_M_endBone", "Tail8_endBone"):
            if alt in keep:
                return alt
    seen = set()
    cur = name
    while cur and cur not in seen:
        seen.add(cur)
        if cur in keep:
            return cur
        cur = parent_of.get(cur) or _WEIGHT_ALIASES.get(cur, "")
    alias = _WEIGHT_ALIASES.get(name)
    if alias and alias in keep:
        return alias
    if _looks_like_face_bone(name) and "Head_M" in keep:
        return "Head_M"
    if _is_mask_weight_group(name):
        return name
    if name.endswith("_endBone"):
        stem = name[: -len("_endBone")]
        if stem in keep:
            return stem
        if "head" in name.lower() and "Head_M" in keep:
            return "Head_M"
    if default and default in keep:
        return default
    return name


def collapse_mesh_weights_to_bones(
    obj,
    keep: Iterable[str],
    parent_of: Optional[Dict[str, str]] = None,
    *,
    default: Optional[str] = None,
) -> int:
    """Merge vertex groups that are not in ``keep`` onto ancestor/fallback bones.

    Face-rig weights (jaw, lips, lids, …) land on ``Head_M`` when that bone
    exists. Returns how many groups were removed.
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return 0
    keep_set = {n for n in keep if n}
    if not keep_set:
        return 0
    vgs = obj.vertex_groups
    if not vgs:
        return 0
    parents = parent_of or {}
    protected = set(_mask_modifier_groups(obj))
    dests = []
    for vg in vgs:
        if vg.name in protected or _is_mask_weight_group(vg.name):
            dests.append(vg.name)
        else:
            dests.append(_weight_dest(vg.name, keep_set, parents, default))
    if all(d == vg.name for d, vg in zip(dests, vgs)):
        return 0
    acc: List[Dict[str, float]] = [{} for _ in obj.data.vertices]
    for vert in obj.data.vertices:
        bucket = acc[vert.index]
        for grp in vert.groups:
            gi = grp.group
            if gi < 0 or gi >= len(dests):
                continue
            dest = dests[gi]
            if not dest:
                continue
            bucket[dest] = bucket.get(dest, 0.0) + grp.weight
    before = len(vgs)
    keep_present = sorted({d for d in dests if d})
    for vg in list(vgs):
        vgs.remove(vg)
    created = {name: vgs.new(name=name) for name in keep_present}
    for index, bucket in enumerate(acc):
        for name, weight in bucket.items():
            if weight > 1e-8:
                created[name].add([index], weight if weight <= 1.0 else 1.0, "REPLACE")
    return before - len(vgs)


def collapse_mesh_weights_to_armature(obj, arm_obj) -> int:
    """Skin leftover face groups onto bones that exist on ``arm_obj``."""
    if arm_obj is None or getattr(arm_obj, "type", None) != "ARMATURE":
        return 0
    keep = {b.name for b in arm_obj.data.bones}
    parents = _parent_map_from_armature(arm_obj)
    for child, parent in _MISSING_BONE_PARENTS.items():
        parents.setdefault(child, parent)
    default = "Head_M" if _is_head_piece(obj) else None
    n = collapse_mesh_weights_to_bones(obj, keep, parents, default=default)
    _ensure_head_skin(obj, arm_obj)
    if _bind_eyes_to_eye_bones(obj, arm_obj):
        n += 1
    return n


def _bind_eyes_to_eye_bones(obj, arm_obj) -> bool:
    """Dragonborn clips pose Dummy_EyeFX. Eyes skinned only to Head_M stay in the skull."""
    if not _is_eyes_mesh(obj) or obj.parent is not arm_obj:
        return False
    bones = []
    for name in ("Dummy_EyeFX_01", "Dummy_EyeFX_02"):
        bone = arm_obj.data.bones.get(name)
        if bone is None:
            return False
        bones.append((name, bone.head_local.copy()))
    mesh_local = obj.matrix_local
    clusters = {name: [] for name, _head in bones}
    for vert in obj.data.vertices:
        point = mesh_local @ vert.co
        name = min(bones, key=lambda item: (item[1] - point).length)[0]
        clusters[name].append(vert.index)
    if any(len(idxs) < 3 for idxs in clusters.values()):
        return False
    groups = obj.vertex_groups
    for group in list(groups):
        groups.remove(group)
    for name, idxs in clusters.items():
        groups.new(name=name).add(idxs, 1.0, "REPLACE")
    return True


def _ensure_head_skin(obj, arm_obj) -> None:
    """ANIM_SETUP ears often have zero groups — bind the whole mesh to Head_M."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return
    if obj.vertex_groups:
        return
    if not (_is_head_piece(obj) or _is_head_accessory(obj)):
        return
    if arm_obj.data.bones.get("Head_M") is None:
        return
    vg = obj.vertex_groups.new(name="Head_M")
    idxs = [v.index for v in obj.data.vertices]
    if idxs:
        vg.add(idxs, 1.0, "REPLACE")


def _vg_weight(obj, name: str, index: int) -> float:
    group = obj.vertex_groups.get(name)
    if group is None:
        return 0.0
    try:
        return float(group.weight(index))
    except RuntimeError:
        return 0.0


def _ensure_mask_modifier(obj, group_name: str, *, invert: bool = True) -> None:
    found = None
    for mod in obj.modifiers:
        if mod.type == "MASK":
            found = mod
            break
    if found is None:
        found = obj.modifiers.new(name="Mask", type="MASK")
    found.vertex_group = group_name
    try:
        found.invert_vertex_group = bool(invert)
    except Exception:
        pass


def paint_chest_cloth_mask(obj) -> int:
    """Paint invert-Mask on shirt/pants-covered torso. Returns verts hidden."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return 0
    vg = obj.vertex_groups.get("Mask")
    if vg is None:
        vg = obj.vertex_groups.new(name="Mask")
    painted = 0
    for vert in obj.data.vertices:
        idx = vert.index
        chest = _vg_weight(obj, "Chest_M", idx) + _vg_weight(obj, "Spine2_M", idx)
        neck = _vg_weight(obj, "Neck_M", idx) + _vg_weight(obj, "Head_M", idx)
        arm = (
            _vg_weight(obj, "Shoulder_L", idx)
            + _vg_weight(obj, "Shoulder_R", idx)
            + _vg_weight(obj, "Shoulder_Twist_L", idx)
            + _vg_weight(obj, "Shoulder_Twist_R", idx)
            + _vg_weight(obj, "Shoulder_L_Twist_01", idx)
            + _vg_weight(obj, "Shoulder_R_Twist_01", idx)
        )
        hip = _vg_weight(obj, "Hip_L", idx) + _vg_weight(obj, "Hip_R", idx)
        hide = (chest >= 0.32 and neck < 0.22 and arm < 0.55) or (
            hip >= 0.55 and chest < 0.18
        )
        if hide:
            vg.add([idx], 1.0, "REPLACE")
            painted += 1
        else:
            try:
                vg.remove([idx])
            except Exception:
                pass
    _ensure_mask_modifier(obj, "Mask", invert=True)
    return painted


def ensure_cloth_hide_mask(obj, *, allow_paint: bool = True) -> int:
    """Keep existing Mask/Group paints; optionally paint a chest/hip hide.

    No-cloth kits (Dragonborn) and pants-only kits (HFL_M) must not enable a
    chest hide-mask — that deletes the torso when nothing covers it.
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return 0
    n = _name_l(obj)
    if "clt_" in n or "body" not in n:
        return 0
    for group in obj.vertex_groups:
        if not _is_mask_weight_group(group.name):
            continue
        found = next((m for m in obj.modifiers if m.type == "MASK"), None)
        if found is None:
            _ensure_mask_modifier(obj, group.name, invert=True)
        elif not (found.vertex_group or ""):
            found.vertex_group = group.name
        return 0
    if not allow_paint:
        return 0
    return paint_chest_cloth_mask(obj)


def _is_tail(obj) -> bool:
    return "tail" in _name_l(obj)


def _is_feet(obj) -> bool:
    n = _name_l(obj)
    return "feet" in n or n.endswith("_foot") or "_foot_" in n


def _is_head_piece(obj) -> bool:
    n = _name_l(obj)
    if _is_tail(obj) or _is_lod(obj):
        return False
    return any(tag in n for tag in ("head",) + _HEAD_ACC_TAGS)


def _is_head_accessory(obj) -> bool:
    n = _name_l(obj)
    if _is_lod(obj):
        return False
    return any(tag in n for tag in _HEAD_ACC_TAGS)


def _is_eyes_mesh(obj) -> bool:
    n = _name_l(obj)
    if _is_lod(obj) or _is_skipped_kit_mesh(obj):
        return False
    return "eyes_mesh" in n


_WIDGET_PREFIX = "GR2LAB_WGT_"


def _is_rig_overlay_mesh(obj) -> bool:
    """IK widgets / Lips — not character skin."""
    n = obj.name or ""
    if n.startswith(_WIDGET_PREFIX):
        return True
    if obj.get("gr2lab_kiss_target"):
        return True
    if obj.get("gr2lab_hud"):
        return True
    return False


def iter_connected_character_meshes(arm_obj):
    """Meshes parented or skinned to ``arm_obj``, skipping widgets and bone props."""
    if arm_obj is None or getattr(arm_obj, "type", None) != "ARMATURE":
        return
    seen = set()
    for obj in list(getattr(arm_obj, "children", None) or []):
        if getattr(obj, "type", None) != "MESH":
            continue
        if obj.parent_type == "BONE":
            continue
        if _is_rig_overlay_mesh(obj):
            continue
        seen.add(obj)
        yield obj
    for obj in bpy.data.objects:
        if obj in seen or getattr(obj, "type", None) != "MESH":
            continue
        if obj.parent_type == "BONE":
            continue
        if _is_rig_overlay_mesh(obj):
            continue
        if obj.parent not in (None, arm_obj):
            continue
        for mod in getattr(obj, "modifiers", None) or []:
            if mod.type == "ARMATURE" and getattr(mod, "object", None) == arm_obj:
                yield obj
                break


def strip_connected_meshes(
    arm_obj, *, keep=None, keep_kit: bool = False
) -> Dict[str, Any]:
    """Delete character meshes on ``arm_obj``. Keeps Lips, IK widgets, bone-parented props.

    ``keep_kit`` leaves meshes tagged ``gr2lab_mesh_kit`` (the current kit).
    """
    out: Dict[str, Any] = {"count": 0, "removed": []}
    keep_objs = set()
    for item in keep or []:
        if isinstance(item, str):
            obj = bpy.data.objects.get(item)
            if obj is not None:
                keep_objs.add(obj)
        else:
            keep_objs.add(item)
    removed: List[str] = []
    for obj in list(iter_connected_character_meshes(arm_obj)):
        if obj in keep_objs:
            continue
        if keep_kit and obj.get(_MESH_KEY):
            continue
        name = obj.name
        mesh = obj.data if obj.type == "MESH" else None
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            continue
        if mesh is not None and mesh.users == 0:
            try:
                bpy.data.meshes.remove(mesh)
            except Exception:
                pass
        removed.append(name)
    out["removed"] = removed
    out["count"] = len(removed)
    if removed:
        print(
            f"[GR2 LAB] stripped {len(removed)} meshes from {arm_obj.name!r}: "
            + ", ".join(removed)
        )
    return out


def _mesh_centroid(obj) -> Vector:
    mesh = getattr(obj, "data", None)
    verts = getattr(mesh, "vertices", None) if mesh is not None else None
    if not verts:
        return Vector((0.0, 0.0, 0.0))
    acc = Vector((0.0, 0.0, 0.0))
    for v in verts:
        acc += v.co
    return acc / float(len(verts))


def _centroid_z_up(obj) -> bool:
    """True when mesh data already stands in body/Dummy Z-up."""
    c = _mesh_centroid(obj)
    return abs(c.z) > abs(c.y) * 1.5 and abs(c.z) > 0.5


def _centroid_y_up(obj) -> bool:
    """True when mesh data still lies in old SK Y-up (will look like floor)."""
    c = _mesh_centroid(obj)
    return abs(c.y) > abs(c.z) * 1.5 and abs(c.y) > 0.5


def _kit_head_mesh(meshes: Sequence[Any]):
    heads = []
    for obj in meshes:
        n = _name_l(obj)
        if _is_lod(obj) or _is_tail(obj) or "clt_" in n:
            continue
        if any(tag in n for tag in _HEAD_ACC_TAGS):
            continue
        if "head" in n:
            heads.append(obj)
    return heads[0] if heads else None


def _needs_stand_rx90(obj, *, head_needs: bool) -> bool:
    """Stand a mesh if its own data is still Y-up.

    Kit ears/horns must not follow the skull AABB: ANIM_SETUP heads are Z-up
    cubes, so that test stays false and leftover Y-up ears stay on the floor.
    """
    if _is_tail(obj) or _is_feet(obj):
        return False
    if _centroid_z_up(obj):
        return False
    if _centroid_y_up(obj):
        return True
    if _is_head_piece(obj):
        return bool(head_needs)
    return _aabb_y_up(obj)


def _ident_rotation(obj) -> None:
    if obj.rotation_mode == "QUATERNION":
        obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    elif obj.rotation_mode == "AXIS_ANGLE":
        obj.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    else:
        obj.rotation_euler = (0.0, 0.0, 0.0)


def _set_rotation_quat(obj, quat) -> None:
    if obj.rotation_mode == "QUATERNION":
        obj.rotation_quaternion = quat
    elif obj.rotation_mode == "AXIS_ANGLE":
        axis, angle = quat.to_axis_angle()
        obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
    else:
        obj.rotation_euler = quat.to_euler(obj.rotation_mode)


def _reparent_keep_world(obj, arm_obj, *, stand_rx90: bool, force_display_cancel: bool = False) -> None:
    """Parent a kit mesh into the armature's bind frame.

    Kit meshes append at the scene origin. Restoring that world after parent
    (keep-world) only works when the armature is also at the origin. A swapped
    rig parented to an Empty (offset / yaw) leaves the mesh at 0,0,0 while the
    bones sit on the Empty — the armature modifier blows the mesh up.

    Y-up mesh data inherits the armature object (Rx90 + Empty).
    Z-up mesh data inherits the Empty/offset and cancels only the display Rx90.
    ``stand_rx90`` is kept for callers; Z-up is detected from mesh data.

    Tail meshes are long along Y, so the height test calls them Y-up even
    though they share the body kit's Z-up space. ``force_display_cancel``
    applies the same Rx90 cancel the body meshes use.
    """
    obj.parent_type = "OBJECT"
    try:
        obj.parent_bone = ""
    except Exception:
        pass
    obj.parent = arm_obj
    obj.matrix_parent_inverse.identity()
    obj.location = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    # Z-up kit vertices into this skeleton's Y-up bone space (Rx -90).
    # Inverting matrix_basis is wrong here: slot scale (-1, 1, 1) reads as a
    # half-turn, and a swap attaches after seating has already moved the
    # display Rx90 onto the empty, so the armature rotation is 0.
    bones_along_y = True
    spine = arm_obj.data.bones.get("Spine1_M") if arm_obj.data else None
    if spine is not None:
        head = spine.head_local
        bones_along_y = abs(float(head.y)) >= abs(float(head.z))
    if bones_along_y and (
        force_display_cancel or ((not stand_rx90) and _mesh_data_z_up(obj))
    ):
        _set_rotation_quat(
            obj, Euler((math.radians(-90.0), 0.0, 0.0), "XYZ").to_quaternion()
        )
    else:
        _ident_rotation(obj)


def _apply_cloth_masks(obj, *, enabled: bool) -> None:
    """Body-only: hide the clothing Mask. Body+cloth: keep it on."""
    for mod in obj.modifiers:
        if mod.type != "MASK":
            continue
        mod.show_viewport = bool(enabled)
        mod.show_render = bool(enabled)


def _ensure_armature_modifier(obj, arm_obj) -> None:
    found = None
    extras = []
    for mod in list(obj.modifiers):
        if mod.type != "ARMATURE":
            continue
        if found is None:
            found = mod
        else:
            extras.append(mod)
    for mod in extras:
        try:
            obj.modifiers.remove(mod)
        except Exception:
            pass
    if found is None:
        found = obj.modifiers.new(name="Armature", type="ARMATURE")
    found.object = arm_obj
    try:
        found.use_vertex_groups = True
    except Exception:
        pass
    try:
        found.use_bone_envelopes = False
    except Exception:
        pass


def _absorb_into_armature_collections(obj, arm_obj, context) -> None:
    dests = list(arm_obj.users_collection)
    if not dests:
        dests = [context.scene.collection]
    for dest in dests:
        try:
            dest.objects.link(obj)
        except RuntimeError:
            pass
    for coll in list(obj.users_collection):
        if coll in dests:
            continue
        try:
            coll.objects.unlink(obj)
        except RuntimeError:
            pass


def _discard_empty_kit_collections(colls, context) -> None:
    """Drop leftover *_body / *_cloth collections so they are not extra outliner rows."""
    scene_root = getattr(getattr(context, "scene", None), "collection", None)
    for coll in colls:
        if coll is None:
            continue
        if len(coll.objects) or len(coll.children):
            coll.hide_viewport = True
            coll.hide_render = True
            continue
        for parent in list(bpy.data.collections):
            if parent.children.get(coll.name) is not None:
                try:
                    parent.children.unlink(coll)
                except RuntimeError:
                    pass
        if scene_root is not None and scene_root.children.get(coll.name) is not None:
            try:
                scene_root.children.unlink(coll)
            except RuntimeError:
                pass
        try:
            bpy.data.collections.remove(coll)
        except Exception:
            pass


def _append_collections(lib: Path, names: Sequence[str]) -> List[Any]:
    want = [n for n in names if n]
    if not want:
        return []
    with bpy.data.libraries.load(str(lib), link=False) as (data_from, data_to):
        available = set(data_from.collections)
        data_to.collections = [n for n in want if n in available]
    return [c for c in data_to.collections if c is not None]


def attach_mesh_kit(
    context,
    arm_obj,
    *,
    base_id: Optional[str] = None,
    mode: str = "NONE",
    akelka: bool = False,
    doc: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Append kit meshes and skin them to ``arm_obj``. Returns a small report."""
    out: Dict[str, Any] = {"ok": False, "meshes": 0, "mode": "NONE", "base": ""}
    if arm_obj is None or getattr(arm_obj, "type", None) != "ARMATURE":
        out["error"] = "no armature"
        return out
    mode_n = normalize_mesh_mode(mode)
    out["mode"] = mode_n
    if mode_n == "NONE":
        out["ok"] = True
        out["skipped"] = "NONE"
        return out
    ident = resolve_kit_base(base_id, doc=doc, template_path=template_path)
    if not ident:
        out["error"] = "no body base"
        return out
    cat = load_catalog(context)
    kit_ident, fb_fam = _catalog_base_for_kit(
        ident, cat, template_path=template_path
    )
    if fb_fam and kit_ident != ident:
        out["kit_fallback"] = f"{fb_fam} → {kit_ident}"
        print(
            f"[GR2 LAB] mesh kit fallback: {fb_fam!r} uses human kit {kit_ident!r}"
        )
    ident = kit_ident
    out["base"] = ident
    entry = (cat.get("bases") or {}).get(ident) or {}
    if not entry or entry.get("missing"):
        out["error"] = f"no mesh kit for {ident}"
        return out
    lib = library_path(context)
    if lib is None or not lib.is_file():
        out["error"] = "Mesh kit library missing (addon mesh_kits/GR2LAB_mesh_kits.blend)"
        return out

    coll_names: List[str] = []
    body_c = entry.get("body_collection")
    cloth_c = entry.get("cloth_collection")
    akelka = uses_akelka_kit(ident, mode_n, akelka)
    akelka_c = str(entry.get("akelka_collection") or "")
    if akelka and akelka_c:
        coll_names.append(akelka_c)
        out["akelka"] = True
    else:
        if akelka and not akelka_c:
            out["akelka_fallback"] = "body"
            print(f"[GR2 LAB] {ident} has no AkELkA kit — appending usual body meshes")
        if body_c:
            coll_names.append(str(body_c))
    want_cloth = mode_n == "BODY_CLOTH" and not akelka
    has_cloth = bool(cloth_c) and bool(entry.get("cloth") or [])
    if want_cloth and has_cloth:
        coll_names.append(str(cloth_c))
    elif want_cloth and not has_cloth:
        out["cloth_fallback"] = "body"
        print(f"[GR2 LAB] {ident} has no cloth kit — appending body meshes only")

    loaded = _append_collections(lib, coll_names)
    if not loaded:
        out["error"] = f"library has no collections for {ident}"
        return out

    attached: List[str] = []
    kit_meshes = [
        obj
        for coll in loaded
        for obj in list(coll.objects)
        if obj.type == "MESH" and not _is_lod(obj) and not _is_skipped_kit_mesh(obj)
    ]
    # AkELkA dragonborn heads already have eyes in the sculpt. The stock
    # Head_*_Eyes mesh is a second pair and sits in the wrong place.
    # Chest hide-mask only when a shirt actually landed. Pants-only (HFL_M)
    # would punch a hole in the torso with nothing to cover it.
    has_upper_cloth = any(_is_upper_cloth(obj) for obj in kit_meshes)
    mask_on = want_cloth and has_upper_cloth
    if want_cloth and has_cloth and not has_upper_cloth:
        print(f"[GR2 LAB] {ident} cloth has no shirt — body hide-mask off")
    head = _kit_head_mesh(kit_meshes)
    head_needs = _aabb_y_up(head) if head is not None else False
    for obj in kit_meshes:
        _reparent_keep_world(
            obj, arm_obj, stand_rx90=_needs_stand_rx90(obj, head_needs=head_needs)
        )
        _ensure_armature_modifier(obj, arm_obj)
        if _is_tail(obj):
            restore_tail_skin(obj)
        else:
            collapse_mesh_weights_to_armature(obj, arm_obj)
        ensure_cloth_hide_mask(obj, allow_paint=mask_on)
        _apply_cloth_masks(obj, enabled=mask_on)
        _absorb_into_armature_collections(obj, arm_obj, context)
        try:
            obj[_MESH_KEY] = ident
            obj.hide_viewport = False
            obj.hide_render = False
        except Exception:
            pass
        attached.append(obj.name)
    _discard_empty_kit_collections(loaded, context)
    out["ok"] = True
    out["meshes"] = len(attached)
    out["names"] = attached
    try:
        arm_obj["gr2lab_mesh_kit"] = ident
        arm_obj["gr2lab_mesh_count"] = len(attached)
    except Exception:
        pass
    print(
        f"[GR2 LAB] mesh kit {ident} {mode_n}: "
        f"{len(attached)} meshes from {lib.name}"
    )
    return out


_TAIL_SKIN_NAME = "tail_skin.json"
_TAIL_SKIN_CACHE: Optional[Dict[str, Any]] = None


def _load_tail_skin() -> Dict[str, Any]:
    global _TAIL_SKIN_CACHE
    if _TAIL_SKIN_CACHE is not None:
        return _TAIL_SKIN_CACHE
    path = _addon_mesh_kits_dir() / _TAIL_SKIN_NAME
    if not path.is_file():
        _TAIL_SKIN_CACHE = {}
        return _TAIL_SKIN_CACHE
    try:
        _TAIL_SKIN_CACHE = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _TAIL_SKIN_CACHE = {}
    if not isinstance(_TAIL_SKIN_CACHE, dict):
        _TAIL_SKIN_CACHE = {}
    return _TAIL_SKIN_CACHE


def restore_tail_skin(obj) -> int:
    """Fill missing Tail* vertex groups from shipped TAILS weights."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return 0
    if not _is_tail(obj):
        return 0
    if any(g.name.startswith("Tail") for g in obj.vertex_groups):
        return 0
    key = (obj.name or "").split(".")[0]
    doc = _load_tail_skin().get(key)
    if not isinstance(doc, dict):
        return 0
    if int(doc.get("verts") or 0) != len(obj.data.vertices):
        print(
            f"[GR2 LAB] tail skin skip {obj.name}: "
            f"verts {len(obj.data.vertices)} != {doc.get('verts')}"
        )
        return 0
    n = 0
    for gname, pairs in (doc.get("groups") or {}).items():
        if not str(gname).startswith("Tail"):
            continue
        vg = obj.vertex_groups.get(gname) or obj.vertex_groups.new(name=str(gname))
        for item in pairs or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            vg.add([int(item[0])], float(item[1]), "REPLACE")
            n += 1
    if n:
        print(f"[GR2 LAB] tail skin {obj.name}: {n} Tail* weights")
    return n


def _append_body_tail_meshes(context, arm_obj, base_id: Optional[str]) -> List:
    """When AkELkA (or similar) skipped Tail_A, pull it from the usual body kit."""
    ident = resolve_kit_base(base_id)
    if not ident:
        return []
    cat = load_catalog(context)
    entry = (cat.get("bases") or {}).get(ident) or {}
    body_c = entry.get("body_collection")
    if not body_c:
        return []
    lib = library_path(context)
    if lib is None or not lib.is_file():
        return []
    loaded = _append_collections(lib, [str(body_c)])
    tails = []
    for coll in loaded:
        for obj in list(coll.objects):
            if obj.type != "MESH" or _is_lod(obj) or _is_skipped_kit_mesh(obj):
                continue
            if _is_tail(obj):
                tails.append(obj)
            else:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass
    _discard_empty_kit_collections(loaded, context)
    for obj in tails:
        _reparent_keep_world(obj, arm_obj, stand_rx90=False, force_display_cancel=True)
        _ensure_armature_modifier(obj, arm_obj)
        restore_tail_skin(obj)
        try:
            obj[_MESH_KEY] = ident
            obj.hide_viewport = False
            obj.hide_render = False
        except Exception:
            pass
        _absorb_into_armature_collections(obj, arm_obj, context)
    return tails


def retarget_tail_meshes(
    context, body_arm, tail_arm, *, base_id: Optional[str] = None
) -> int:
    """Move Tail_A kit meshes off the body onto the sibling tail armature."""
    if body_arm is None or tail_arm is None:
        return 0
    if getattr(body_arm, "type", None) != "ARMATURE":
        return 0
    if getattr(tail_arm, "type", None) != "ARMATURE":
        return 0
    tails = [
        obj
        for obj in list(body_arm.children)
        if getattr(obj, "type", None) == "MESH" and _is_tail(obj)
    ]
    if not tails:
        tails = _append_body_tail_meshes(context, body_arm, base_id)
    moved = 0
    for obj in tails:
        restore_tail_skin(obj)
        _reparent_keep_world(obj, tail_arm, stand_rx90=False, force_display_cancel=True)
        _ensure_armature_modifier(obj, tail_arm)
        collapse_mesh_weights_to_armature(obj, tail_arm)
        _absorb_into_armature_collections(obj, tail_arm, context)
        moved += 1
    return moved
