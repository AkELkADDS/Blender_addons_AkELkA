"""GPU race chip in the 3D view.

Sits on a GR2 LAB armature the way Character Chops pins the rig name.
Click opens the other body races that have this same scene clip.
Picking one swaps the rig: that race's animation, body, and cloth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import traceback

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

_HANDLE = None
_KEYMAPS: List[Tuple[object, object]] = []
# region pointer -> list of (kind, x, y, w, h, arm_name, path)
_HITS: Dict[int, list] = {}
_OPEN = {"arm": "", "region": 0}
_MOUSE = {"region": 0, "x": -1.0, "y": -1.0, "hit": None}
_RACE_CLICK = {"t": 0.0, "xy": (0, 0)}
_ACTIVE_AREA = 0
_RACE_CACHE: Dict[str, Tuple[float, List[Tuple[str, str, str]]]] = {}
_ROLE_CACHE: Dict[Tuple[str, str], bool] = {}
_HIT_PAD = 8.0
_MODAL_WANT_STOP = False
_BATCH_BANNER_H = 18.0
_BATCH_SWAP: Dict[str, object] = {"queue": [], "restore": []}
_BATCH_ON = False
_HIT_CACHE_KEY: Dict[int, object] = {}


def _shader():
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _rect(x, y, w, h, color) -> None:
    shader = _shader()
    batch = batch_for_shader(
        shader,
        "TRI_FAN",
        {"pos": ((x, y), (x + w, y), (x + w, y + h), (x, y + h))},
    )
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _source_path(arm) -> Optional[Path]:
    raw = str(arm.get("gr2lab_full_source_gr2") or arm.get("gr2lab_source_gr2") or "")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        return None
    return path


def _race_label(folder_name: str) -> str:
    name = folder_name
    if name.endswith("_Rig"):
        name = name[: -len("_Rig")]
    return name or folder_name


def _split_race(filename: str) -> Tuple[str, str]:
    """``HUM_F_Rig_SCENE_...PLYR.GR2`` or ``HUM_M_Rig_LIE_SCENE_...GR2``.

    The race is the text before ``_Rig_``. The suffix is everything after it,
    including a lie-down tag such as ``LIE_``, as long as ``SCENE_`` is present.
    """
    marker = "_Rig_"
    i = filename.find(marker)
    if i <= 0:
        return "", ""
    rest = filename[i + len(marker) :]
    if "SCENE_" not in rest:
        return "", ""
    return filename[:i], rest


def _scene_dir(source: Path) -> Path:
    if source.parent.name.endswith("_Rig"):
        return source.parent.parent
    return source.parent


def _role_exists(scene_dir: Path, suffix: str) -> bool:
    """True when some race has this exact role clip."""
    key = (str(scene_dir).lower(), suffix.lower())
    cached = _ROLE_CACHE.get(key)
    if cached is not None:
        return cached
    found = False
    try:
        if any(scene_dir.glob(f"*_Rig_{suffix}")):
            found = True
    except OSError:
        found = False
    if not found:
        try:
            children = list(scene_dir.iterdir())
        except OSError:
            children = []
        for folder in children:
            if not folder.is_dir() or not folder.name.endswith("_Rig"):
                continue
            if (folder / f"{folder.name}_{suffix}").is_file():
                found = True
                break
    _ROLE_CACHE[key] = found
    return found


def _canonical_suffix(source: Path, race: str, suffix: str) -> str:
    """Drop a body tag such as ``_LongSnout`` when the plain role clip exists.

    Dragonborn stores this beat as ``..._PLYR_LongSnout.GR2``. After that
    swap the chooser must still list the other races' ``..._PLYR.GR2``.
    """
    del race
    ext = Path(suffix).suffix
    stem = Path(suffix).stem
    if not ext or "_" not in stem:
        return suffix
    base_stem, _tag = stem.rsplit("_", 1)
    if not base_stem or not _tag:
        return suffix
    base_suffix = base_stem + ext
    if _role_exists(_scene_dir(source), base_suffix):
        return base_suffix
    return suffix


def _variant_tag(filename: str, suffix: str) -> str:
    """Extra body tag when the clip is this role plus ``_LongSnout``.

    Empty string means the filename is the exact role. Missing means it is
    a different clip.
    """
    _race, file_suffix = _split_race(filename)
    if not file_suffix:
        return ""
    if file_suffix.lower() == suffix.lower():
        return ""
    ext = Path(suffix).suffix
    if not ext or not file_suffix.lower().endswith(ext.lower()):
        return ""
    stem = Path(suffix).stem
    body = file_suffix[: -len(ext)]
    prefix = stem + "_"
    if not body.lower().startswith(prefix.lower()):
        return ""
    tag = body[len(prefix) :]
    if not tag or "_" in tag:
        return ""
    return tag


def sibling_races(source: Path) -> List[Tuple[str, str, str]]:
    """Race clips that share this scene role.

    Files live either as ``Scene/HUM_F_Rig/HUM_F_Rig_SCENE_....GR2`` or flat
    as ``Scene/HUM_M_Rig_SCENE_....GR2``. The role (player, Karlach, …) is
    the suffix after ``_Rig_``. A race that only has a body variant of that
    role, such as dragonborn ``_LongSnout``, is included too.
    """
    race, suffix = _split_race(source.name)
    if not race or not suffix:
        return []
    suffix = _canonical_suffix(source, race, suffix)
    scene_dir = _scene_dir(source)
    key = f"{scene_dir}|{suffix}"
    try:
        stamp = scene_dir.stat().st_mtime
    except OSError:
        return []
    cached = _RACE_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    found: Dict[str, str] = {}
    variants: Dict[str, Dict[str, str]] = {}

    def _take_exact(label: str, path: Path) -> None:
        if label:
            found[label] = str(path)

    def _take_variant(label: str, tag: str, path: Path) -> None:
        if not label or not tag or label in found:
            return
        variants.setdefault(label, {})[tag] = str(path)

    def _consider(path: Path) -> None:
        if not path.is_file():
            return
        label, _suf = _split_race(path.name)
        tag = _variant_tag(path.name, suffix)
        if tag == "" and _split_race(path.name)[1].lower() == suffix.lower():
            _take_exact(label, path)
        elif tag:
            _take_variant(label, tag, path)

    stem = Path(suffix).stem
    ext = Path(suffix).suffix or ".GR2"
    try:
        flat = list(scene_dir.glob(f"*_Rig_{suffix}"))
        flat += list(scene_dir.glob(f"*_Rig_{stem}_*{ext}"))
    except OSError:
        flat = []
    for cand in flat:
        _consider(cand)
    try:
        children = list(scene_dir.iterdir())
    except OSError:
        children = []
    for folder in children:
        if not folder.is_dir() or not folder.name.endswith("_Rig"):
            continue
        label = _race_label(folder.name)
        exact = folder / f"{folder.name}_{suffix}"
        if exact.is_file():
            _take_exact(label, exact)
            continue
        try:
            extra = list(folder.glob(f"{folder.name}_{stem}_*{ext}"))
        except OSError:
            extra = []
        for cand in extra:
            _consider(cand)
    for label, tags in variants.items():
        if label in found:
            continue
        if len(tags) == 1:
            found[label] = next(iter(tags.values()))
        else:
            for tag, path in tags.items():
                found[f"{label} {tag}"] = path
    rows = _with_every_base(found)
    _RACE_CACHE[key] = (stamp, rows)
    return rows


def _listed_bases() -> List[Tuple[str, str]]:
    """Body bases as (race id, file name). Tails are not a chooser row."""
    try:
        from bind_pose import list_body_bases  # type: ignore

        raw = list_body_bases() or []
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for item in raw:
        name = item.get("name") if isinstance(item, dict) else str(item or "")
        if not name:
            continue
        stem = Path(name).stem
        if not stem.lower().endswith("_base"):
            continue
        race = stem[: -len("_base")]
        if "tail" in race.lower():
            continue
        out.append((race, Path(name).name))
    return out


def _donor_for(race: str, available: set) -> str:
    """A race in this scene that can play when ``race`` has no own clip."""
    if race in available:
        return race
    family, _, kind = race.rpartition("_")
    if not family:
        family, kind = race, ""
    if kind == "FS":
        same = ("FS", "F")
    elif kind == "MS":
        same = ("MS", "M")
    elif kind == "F":
        same = ("F", "FS")
    elif kind == "M":
        same = ("M", "MS")
    else:
        same = ()
    prefs: List[str] = []
    for token in same:
        prefs.append(f"{family}_{token}")
    for token in same:
        prefs.append(f"HUM_{token}")
    opposite = {"F": "M", "M": "F", "FS": "MS", "MS": "FS"}.get(kind, "")
    if opposite:
        prefs.append(f"{family}_{opposite}")
    prefs.extend(sorted(available))
    for cand in prefs:
        if cand in available:
            return cand
    return ""


def _with_every_base(found: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """Every body base, using that race's clip or the closest one this scene has."""
    bases = _listed_bases()
    base_of = {race: filename for race, filename in bases}
    labels = sorted(set(base_of) | set(found))
    available = set(found)
    rows: List[Tuple[str, str, str]] = []
    for race in labels:
        if " " in race or race.lower().endswith("_tail"):
            continue
        if race in found:
            anim = found[race]
        else:
            donor = _donor_for(race, available)
            if not donor:
                continue
            anim = found[donor]
        rows.append((race, anim, base_of.get(race, "")))
    return rows


def _choice_rows(races: List[Tuple[str, str, str]]) -> List[Tuple[str, str, bool, str]]:
    """One row per body base. Dragonborn also gets the stock body and the AkELkA body.

    A race with no clip of its own still appears. It plays the closest clip
    in this scene, on that race's body base.
    """
    rows: List[Tuple[str, str, bool, str]] = []
    for label, path, body in races:
        if label.startswith("DGB_"):
            rows.append((label, path, False, body))
            rows.append((f"{label} AkELkA", path, True, body))
        else:
            rows.append((label, path, False, body))
    return rows


def _wearing_akelka(arm) -> bool:
    try:
        kids = list(arm.children_recursive)
    except Exception:
        kids = []
    for obj in kids:
        if "AKELKA" in str(getattr(obj, "name", "") or "").upper():
            return True
    return False


def _body_label(arm, src: Path) -> str:
    saved = str(arm.get("gr2lab_body_choice") or "")
    if saved:
        return saved
    race = _split_race(src.name)[0] or _race_label(src.parent.name)
    if race.startswith("DGB_") and _wearing_akelka(arm):
        return f"{race} AkELkA"
    return race


def _head_anchor(arm) -> Vector:
    """A point just past the crown, not in the face.

    ``HeadEnd_M_endBone`` sits on the head bone's local X, which is the top
    of the skull. The short head-bone Y axis points across the face.
    """
    pose = getattr(arm, "pose", None)
    world = arm.matrix_world
    head = pose.bones.get("Head_M") or pose.bones.get("head") if pose else None
    end = pose.bones.get("HeadEnd_M_endBone") if pose else None
    if head is not None and end is not None:
        origin = world @ head.head
        crown = world @ end.head
        direction = crown - origin
        if direction.length > 1e-4:
            return crown + direction.normalized() * 0.05
        return crown
    if head is not None:
        axis = (world @ head.matrix).to_3x3().col[0]
        if axis.length > 1e-5:
            return (world @ head.head) + axis.normalized() * 0.22
        return world @ head.head
    return world.translation.copy()


def _selected_body_rigs(context):
    """Active + selected body armatures. Tails are not their own race chip."""
    rigs = []
    seen = set()

    def _take(obj) -> None:
        if obj is None or obj.name in seen:
            return
        if getattr(obj, "type", None) != "ARMATURE":
            return
        if obj.get("gr2lab_tail"):
            return
        try:
            if not obj.visible_get():
                return
        except Exception:
            pass
        rigs.append(obj)
        seen.add(obj.name)

    _take(getattr(context.view_layer.objects, "active", None))
    for obj in list(getattr(context, "selected_objects", None) or []):
        _take(obj)
    return rigs


def _arms(context):
    out = []
    for arm in _selected_body_rigs(context):
        src = _source_path(arm)
        if src is None:
            continue
        races = sibling_races(src)
        if not races:
            continue
        out.append((arm, src, races))
    return out


def _font(size: int) -> int:
    font_id = 0
    try:
        blf.size(font_id, size)
    except TypeError:
        blf.size(font_id, size, 72)
    return font_id


def _text_width(font_id: int, text: str) -> float:
    try:
        return float(blf.dimensions(font_id, text)[0])
    except Exception:
        return float(len(text) * 7)


_ICON_PX = 13.0
_ICON_TEX = None
_ICON_SHADER = None
_ICON_READY = False


def _label_parts(label: str) -> Tuple[str, bool, bool]:
    """Race text, whether to draw the AkELkA mark, and the dropdown caret."""
    caret = "▾" in label
    body = label.replace("▾", " ").replace("AkELkA", " ")
    body = " ".join(body.split())
    return body, "AkELkA" in label, caret


def _label_width(font_id: int, label: str) -> float:
    body, icon, caret = _label_parts(label)
    width = _text_width(font_id, body) if body else 0.0
    if icon:
        width += 4.0 + _ICON_PX
    if caret:
        width += 4.0 + _text_width(font_id, "▾")
    return width


def _icon_texture():
    """One 48px texture for the whole session. Never rebuilt per redraw."""
    global _ICON_TEX, _ICON_SHADER, _ICON_READY
    if _ICON_READY:
        return _ICON_TEX
    _ICON_READY = True
    path = Path(__file__).resolve().parent / "icons" / "akelka.png"
    if not path.is_file():
        return None
    try:
        img = bpy.data.images.load(str(path), check_existing=True)
        img.name = ".gr2lab_akelka"
        try:
            img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        _ICON_TEX = gpu.texture.from_image(img)
        _ICON_SHADER = gpu.shader.from_builtin("IMAGE")
    except Exception:
        _ICON_TEX = None
        _ICON_SHADER = None
    return _ICON_TEX


def _draw_icon(x: float, y: float) -> None:
    tex = _icon_texture()
    shader = _ICON_SHADER
    if tex is None or shader is None:
        return
    size = _ICON_PX
    batch = batch_for_shader(
        shader,
        "TRI_FAN",
        {
            "pos": ((x, y), (x + size, y), (x + size, y + size), (x, y + size)),
            "texCoord": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        },
    )
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_sampler("image", tex)
    batch.draw(shader)


def _draw_choice(font_id: int, x: float, baseline: float, icon_y: float, label: str) -> None:
    body, icon, caret = _label_parts(label)
    cursor = x
    if body:
        blf.position(font_id, cursor, baseline, 0.0)
        blf.draw(font_id, body)
        cursor += _text_width(font_id, body)
    if icon:
        cursor += 4.0
        _draw_icon(cursor, icon_y)
        cursor += _ICON_PX
    if caret:
        cursor += 4.0
        blf.position(font_id, cursor, baseline, 0.0)
        blf.draw(font_id, "▾")


def _menu_multi_batch_slot(context) -> bool:
    """Menu layout reserves a banner row when several rigs are selected."""
    if not _OPEN.get("arm"):
        return False
    return len(_selected_body_rigs(context)) > 1


def _batch_apply_armed(context) -> bool:
    return bool(_BATCH_ON) and _menu_multi_batch_slot(context)


def _batch_banner_text(context) -> str:
    n = len(_selected_body_rigs(context))
    if n <= 1:
        return ""
    if _BATCH_ON:
        return f"All {n} rigs  ·  ON"
    return f"All {n} rigs  ·  click to arm"


def _menu_table_metrics(
    font_id: int,
    choices,
    context,
    pad_x: float,
    row_h: float,
) -> Tuple[float, int, int, float, float, float, str]:
    cols = 2 if len(choices) > 1 else 1
    rows_n = (len(choices) + cols - 1) // cols
    item_w = 0.0
    for label_i, _path, _akelka, _body in choices:
        item_w = max(item_w, _label_width(font_id, label_i) + pad_x * 2.0)
    banner_h = _BATCH_BANNER_H if _menu_multi_batch_slot(context) else 0.0
    banner_text = _batch_banner_text(context) if banner_h else ""
    banner_w = (_text_width(font_id, banner_text) + pad_x * 2.0) if banner_text else 0.0
    grid_w = item_w * float(cols)
    total_w = max(grid_w, banner_w)
    total_h = banner_h + row_h * float(rows_n)
    return item_w, cols, rows_n, total_w, total_h, banner_h, banner_text


def _invalidate_hit_cache() -> None:
    _HIT_CACHE_KEY.clear()


def _hits_cache_key(context, region) -> tuple:
    sel = tuple(sorted(arm.name for arm in _selected_body_rigs(context)))
    return (
        int(region.as_pointer()),
        str(_OPEN.get("arm") or ""),
        int(_OPEN.get("region") or 0),
        sel,
        bool(_menu_multi_batch_slot(context)),
        int(getattr(region, "width", 0) or 0),
        int(getattr(region, "height", 0) or 0),
    )


def _refresh_hits_cached(context, region, *, force: bool = False) -> list:
    key = _hits_cache_key(context, region)
    ptr = int(region.as_pointer())
    if not force and _HIT_CACHE_KEY.get(ptr) == key:
        return list(_HITS.get(ptr) or [])
    hits = _refresh_hits(context, region)
    _HITS[ptr] = hits
    _HIT_CACHE_KEY[ptr] = key
    return hits


def _accent_colors(batch: bool, over: bool) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if batch:
        bar = (0.98, 0.62, 0.22, 1.0) if over else (0.92, 0.48, 0.14, 1.0)
        hover_bg = (0.42, 0.26, 0.10, 0.98)
        idle_bg = (0.14, 0.10, 0.08, 0.94)
        return bar, (hover_bg if over else idle_bg)
    bar = (0.55, 0.82, 1.0, 1.0) if over else (0.45, 0.72, 0.95, 1.0)
    hover_bg = (0.32, 0.55, 0.82, 0.98)
    idle_bg = (0.10, 0.11, 0.13, 0.94)
    return bar, (hover_bg if over else idle_bg)


def _resolve_choice_for_arm(arm, choice_label: str, want_akelka: bool):
    src = _source_path(arm)
    if src is None:
        return None
    races = sibling_races(src)
    if not races:
        return None
    for label, path, akelka, body in _choice_rows(races):
        if label == choice_label and bool(akelka) == bool(want_akelka):
            return str(path), bool(akelka), body, label
    return None


def _import_busy() -> bool:
    try:
        from .import_progress_ui import _STATE as import_state

        return bool(import_state.get("active"))
    except Exception:
        return False


def _restore_arm_selection(context, names: List[str]) -> None:
    if not names:
        return
    try:
        for obj in context.view_layer.objects:
            obj.select_set(False)
        for name in names:
            arm = bpy.data.objects.get(name)
            if arm is not None:
                arm.select_set(True)
        first = bpy.data.objects.get(names[0])
        if first is not None:
            context.view_layer.objects.active = first
    except Exception:
        pass


def _pump_batch_swap() -> Optional[float]:
    state = _BATCH_SWAP
    if _import_busy():
        return 0.15
    queue = list(state.get("queue") or [])
    if not queue:
        restore = list(state.get("restore") or [])
        window = state.get("window")
        area = state.get("area")
        region = state.get("region")
        if window is not None and restore:
            try:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    _restore_arm_selection(bpy.context, restore)
            except Exception:
                pass
        state["queue"] = []
        state["restore"] = []
        _tag()
        return None
    arm_name, path, akelka, body, choice = queue.pop(0)
    state["queue"] = queue
    window = state.get("window")
    area = state.get("area")
    region = state.get("region")
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            _start_swap(bpy.context, arm_name, path, akelka, body, choice)
    except Exception as exc:
        print(f"[GR2 LAB] race batch swap failed: {exc}", flush=True)
    return 0.15


def _enqueue_swap_batch(context, targets, window, area, region) -> None:
    if not targets:
        return
    names = [row[0] for row in targets]
    _BATCH_SWAP.update(
        {
            "queue": list(targets),
            "restore": names,
            "window": window,
            "area": area,
            "region": region,
        }
    )
    bpy.app.timers.register(_pump_batch_swap, first_interval=0.02)


def _corner_origin_y(context) -> float:
    """Bottom-left race chip sits above the animation strip when that strip is on."""
    gap = 10.0
    try:
        from . import anim_browser

        if anim_browser._browser_on(context):
            return anim_browser.STRIP_MARGIN_Y + anim_browser.STRIP_HEIGHT + gap
    except Exception:
        pass
    return 16.0


def _overlay_prefs(context):
    props = getattr(getattr(context, "scene", None), "gr2lab", None)
    if props is None:
        return True, "CHARACTER"
    enabled = bool(getattr(props, "race_overlay", True))
    place = str(getattr(props, "race_overlay_place", "CHARACTER") or "CHARACTER")
    return enabled, place


def active_area_ptr() -> int:
    return int(_ACTIVE_AREA or 0)


def note_area(area) -> None:
    """Remember the 3D view the user dropped into, for the load bar only."""
    global _ACTIVE_AREA
    ptr = 0
    try:
        if area is not None and getattr(area, "type", "") == "VIEW_3D":
            ptr = int(area.as_pointer())
    except Exception:
        ptr = 0
    if not ptr or ptr == _ACTIVE_AREA:
        return
    _ACTIVE_AREA = ptr
    _tag()


def _refresh_hits(context, region) -> list:
    """Hit boxes in region pixels. Recomputed on click so picks match the cursor."""
    if region is None or getattr(region, "type", "") != "WINDOW":
        return []
    area = getattr(context, "area", None)
    if area is None or getattr(area, "type", "") != "VIEW_3D":
        return []
    rv3d = getattr(context, "region_data", None)
    if rv3d is None:
        return []
    try:
        from .import_progress_ui import _STATE as import_state

        if import_state.get("active"):
            return []
    except Exception:
        pass
    enabled, place = _overlay_prefs(context)
    if not enabled:
        return []
    font_id = _font(12)
    row_h = 21.0
    pad_x = 6.0
    batch_tint = _batch_apply_armed(context)
    hits: list = []
    shown = _arms(context)
    open_arm = _OPEN.get("arm") or ""
    shown_names = {row[0].name for row in shown}
    if open_arm and open_arm not in shown_names:
        _OPEN["arm"] = ""
        open_arm = ""
    layouts = []
    for index, (arm, src, races) in enumerate(shown):
        current = _body_label(arm, src)
        label = f"{current}  ▾" if len(races) > 1 else current
        tw = _label_width(font_id, label)
        bw = tw + pad_x * 2.0
        bh = row_h
        if place == "CORNER":
            x = 16.0
            y = _corner_origin_y(context) + index * (row_h + 6.0)
        else:
            try:
                xy = location_3d_to_region_2d(region, rv3d, _head_anchor(arm))
            except Exception:
                xy = None
            if xy is None:
                continue
            x = float(xy[0]) - bw * 0.5
            y = float(xy[1]) + 4.0
        layouts.append((arm, src, races, current, label, x, y, bw, bh))
    for arm, _src, _races, _current, _label, x, y, bw, bh in layouts:
        hits.append(("button", x, y, bw, bh, arm.name, "", False, "", ""))
    for arm, _src, races, current, _label, x, y, bw, bh in layouts:
        if open_arm != arm.name or int(_OPEN.get("region") or 0) != region.as_pointer():
            continue
        choices = _choice_rows(races)
        item_w, cols, rows_n, total_w, total_h, banner_h, _banner = _menu_table_metrics(
            font_id, choices, context, pad_x, row_h
        )
        menu_y = y + bh + 2.0 if place == "CORNER" else y - total_h - 2.0
        if place != "CORNER" and menu_y < 8.0:
            menu_y = y + bh + 2.0
        menu_x = x
        view_w = float(getattr(region, "width", 0) or 0)
        if view_w and menu_x + total_w > view_w - 8.0:
            menu_x = max(8.0, view_w - 8.0 - total_w)
        if banner_h > 0.0:
            by = menu_y + total_h - banner_h - row_h * float(rows_n)
            hits.append(("batch", menu_x, by, total_w, banner_h, arm.name, "", False, "", ""))
        for i, (label_i, path_i, akelka, body) in enumerate(choices):
            col = i % cols
            row = i // cols
            ix = menu_x + item_w * float(col)
            iy = menu_y + total_h - row_h * float(row + 1)
            ih = row_h
            hits.append(("item", ix, iy, item_w, ih, arm.name, path_i, bool(akelka), body, label_i))
    return hits


def _draw() -> None:
    context = bpy.context
    region = getattr(context, "region", None)
    if region is None or getattr(region, "type", "") != "WINDOW":
        return
    area = getattr(context, "area", None)
    if area is None or getattr(area, "type", "") != "VIEW_3D":
        return
    rv3d = getattr(context, "region_data", None)
    if rv3d is None:
        return
    try:
        from .import_progress_ui import _STATE as import_state

        if import_state.get("active"):
            _HITS[region.as_pointer()] = []
            return
    except Exception:
        pass
    enabled, place = _overlay_prefs(context)
    if not enabled:
        _HITS[region.as_pointer()] = []
        return
    font_id = _font(12)
    row_h = 21.0
    pad_x = 6.0
    batch_tint = _batch_apply_armed(context)
    shown = _arms(context)
    open_arm = _OPEN.get("arm") or ""
    layouts = []
    for index, (arm, src, races) in enumerate(shown):
        current = _body_label(arm, src)
        label = f"{current}  ▾" if len(races) > 1 else current
        tw = _label_width(font_id, label)
        bw = tw + pad_x * 2.0
        bh = row_h
        if place == "CORNER":
            x = 16.0
            y = _corner_origin_y(context) + index * (row_h + 6.0)
        else:
            try:
                xy = location_3d_to_region_2d(region, rv3d, _head_anchor(arm))
            except Exception:
                xy = None
            if xy is None:
                continue
            x = float(xy[0]) - bw * 0.5
            y = float(xy[1]) + 4.0
        layouts.append((arm, src, races, current, label, x, y, bw, bh))
    for arm, _src, _races, _current, label, x, y, bw, bh in layouts:
        over_button = _pointer_over(region, x, y, bw, bh)
        if batch_tint:
            chip = (0.28, 0.18, 0.12, 0.96) if over_button else (0.12, 0.08, 0.06, 0.90)
        else:
            chip = (0.18, 0.22, 0.28, 0.94) if over_button else (0.07, 0.08, 0.10, 0.88)
        _rect(x, y, bw, bh, chip)
        bar, _row_bg = _accent_colors(batch_tint, over_button)
        _rect(x, y, 3.0, bh, bar)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0 if over_button else 0.94)
        _draw_choice(font_id, x + pad_x, y + 4.0, y + 4.0, label)
    for arm, _src, races, current, _label, x, y, bw, bh in layouts:
        if open_arm != arm.name or int(_OPEN.get("region") or 0) != region.as_pointer():
            continue
        choices = _choice_rows(races)
        item_w, cols, rows_n, total_w, total_h, banner_h, _banner = _menu_table_metrics(
            font_id, choices, context, pad_x, row_h
        )
        menu_y = y + bh + 2.0 if place == "CORNER" else y - total_h - 2.0
        if place != "CORNER" and menu_y < 8.0:
            menu_y = y + bh + 2.0
        menu_x = x
        view_w = float(getattr(region, "width", 0) or 0)
        if view_w and menu_x + total_w > view_w - 8.0:
            menu_x = max(8.0, view_w - 8.0 - total_w)
        frame = (0.35, 0.18, 0.06, 0.72) if batch_tint else (0.02, 0.02, 0.03, 0.55)
        _rect(menu_x - 1.0, menu_y - 1.0, total_w + 2.0, total_h + 2.0, frame)
        if banner_h > 0.0:
            banner_text = _batch_banner_text(context)
            by = menu_y + total_h - banner_h - row_h * float(rows_n)
            over_banner = _pointer_over(region, menu_x, by, total_w, banner_h)
            if batch_tint:
                banner_bg = (0.42, 0.22, 0.08, 0.98) if over_banner else (0.22, 0.12, 0.06, 0.98)
                blf.color(font_id, 1.0, 0.86, 0.62, 1.0)
            else:
                banner_bg = (0.18, 0.20, 0.24, 0.96) if over_banner else (0.10, 0.11, 0.13, 0.94)
                blf.color(font_id, 0.78, 0.80, 0.84, 1.0)
            _rect(menu_x, by, total_w, banner_h, banner_bg)
            blf.position(font_id, menu_x + pad_x, by + 3.0, 0.0)
            blf.draw(font_id, banner_text)
        for i, (label_i, path_i, akelka, body) in enumerate(choices):
            col = i % cols
            row = i // cols
            ix = menu_x + item_w * float(col)
            iy = menu_y + total_h - row_h * float(row + 1)
            ih = row_h
            current_row = label_i == current
            over = _pointer_over(region, ix, iy, item_w, ih)
            _bar, bg = _accent_colors(batch_tint, over)
            if current_row and not over:
                bg = (0.28, 0.16, 0.10, 0.94) if batch_tint else (0.16, 0.28, 0.40, 0.94)
            _rect(ix, iy, item_w, ih, bg)
            if over:
                _rect(ix, iy, item_w, 2.0, (0.82, 0.94, 1.0, 1.0))
                _rect(ix, iy + ih - 2.0, item_w, 2.0, (0.82, 0.94, 1.0, 1.0))
                _rect(ix, iy, 3.0, ih, _bar[:3] + (1.0,))
            own = _has_own_clip(label_i, path_i)
            if own or over:
                blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            else:
                blf.color(font_id, 0.52, 0.55, 0.60, 1.0)
            nudge = 2.0 if over else 0.0
            _draw_choice(font_id, ix + pad_x + nudge, iy + 4.0, iy + 4.0, label_i)


def _has_own_clip(label: str, path: str) -> bool:
    """True when this row plays that race's own scene clip, not a borrowed one."""
    race = label.replace(" AkELkA", "").strip()
    return bool(race) and Path(path).name.startswith(race + "_")


def note_pointer(context, event) -> None:
    """Remember the cursor in 3D-view pixels so race rows can highlight."""
    from . import anim_browser

    found = anim_browser.view3d_under_pointer(context, event)
    if found is None:
        if _MOUSE.get("hit") is not None:
            _MOUSE["region"] = 0
            _MOUSE["x"] = -1.0
            _MOUSE["y"] = -1.0
            _MOUSE["hit"] = None
            _tag()
        return
    area, region, mx, my = found
    note_area(area)
    window = getattr(context, "window", None)
    try:
        with context.temp_override(window=window, area=area, region=region):
            hits = _refresh_hits_cached(bpy.context, region)
    except Exception:
        hits = list(_HITS.get(region.as_pointer()) or [])
    hit = _hit(region, mx, my, hits)
    same = (
        int(_MOUSE.get("region") or 0) == region.as_pointer()
        and hit == _MOUSE.get("hit")
    )
    _MOUSE["region"] = region.as_pointer()
    _MOUSE["x"] = mx
    _MOUSE["y"] = my
    _MOUSE["hit"] = hit
    if not same:
        _tag()


def _pointer_over(region, x, y, w, h) -> bool:
    if region is None or int(_MOUSE.get("region") or 0) != region.as_pointer():
        return False
    mx = float(_MOUSE.get("x") or -1.0)
    my = float(_MOUSE.get("y") or -1.0)
    return x <= mx <= x + w and y <= my <= y + h


def _hit(region, mx: float, my: float, rows=None, pad: float = 0.0):
    """Row under the cursor. The drawn row wins over a neighbour's padding.

    Rows sit flush. An 8px pad made the bottom of DGB_M AkELkA count as
    DWR_M, because the later row was tested first.
    """
    if region is None:
        return None
    if rows is None:
        rows = _HITS.get(region.as_pointer()) or []
    inset = float(pad or 0.0)

    def _inside(x, y, w, h, extra: float) -> bool:
        return (
            x - extra <= mx <= x + w + extra
            and y - extra <= my <= y + h + extra
        )

    def _closest(cands):
        best = None
        best_d = None
        for kind, x, y, w, h, arm_name, path, akelka, body, choice in cands:
            dx = mx - (x + w * 0.5)
            dy = my - (y + h * 0.5)
            dist = dx * dx + dy * dy
            if best is None or dist < best_d:
                best = (kind, arm_name, path, bool(akelka), body, choice)
                best_d = dist
        return best

    exact = [
        row for row in rows
        if _inside(row[1], row[2], row[3], row[4], 0.0)
    ]
    if exact:
        return _closest(exact)
    padded = [
        row for row in rows
        if inset and _inside(row[1], row[2], row[3], row[4], inset)
    ]
    return _closest(padded)


def try_race_click(context, event, mx=None, my=None) -> bool:
    """Handle a 3D-view click on the race chip or menu. True if the click was used."""
    region = getattr(context, "region", None)
    area = getattr(context, "area", None)
    if region is None or area is None or getattr(area, "type", "") != "VIEW_3D":
        return False
    if getattr(region, "type", "") != "WINDOW":
        return False
    enabled, _place = _overlay_prefs(context)
    if not enabled:
        return False
    note_area(area)
    if mx is None:
        mx = float(event.mouse_region_x)
    if my is None:
        my = float(event.mouse_region_y)
    mx = float(mx)
    my = float(my)
    stamp = (int(getattr(event, "mouse_x", 0)), int(getattr(event, "mouse_y", 0)))
    now = time.perf_counter()
    if now - float(_RACE_CLICK["t"]) < 0.08 and stamp == _RACE_CLICK["xy"]:
        return True
    hits = _refresh_hits_cached(context, region, force=True)
    hit = _hit(region, mx, my, hits, pad=_HIT_PAD)
    if hit is None:
        if _OPEN.get("arm"):
            _OPEN["arm"] = ""
            _invalidate_hit_cache()
            _tag()
        return False
    kind, arm_name, path, akelka, body, choice = hit
    _RACE_CLICK["t"] = now
    _RACE_CLICK["xy"] = stamp
    if kind == "batch":
        global _BATCH_ON
        _BATCH_ON = not _BATCH_ON
        _invalidate_hit_cache()
        _tag()
        return True
    if kind == "button":
        arm = bpy.data.objects.get(arm_name)
        src = _source_path(arm) if arm is not None else None
        if src is None or len(sibling_races(src)) < 2:
            return False
        if _OPEN.get("arm") == arm_name:
            _OPEN["arm"] = ""
        else:
            _OPEN["arm"] = arm_name
            _OPEN["region"] = region.as_pointer()
        _invalidate_hit_cache()
        _tag()
        return True
    if kind == "item" and path:
        window = context.window
        area = context.area
        region = context.region
        batch = _batch_apply_armed(context)
        if batch:
            targets = []
            for arm in _selected_body_rigs(context):
                resolved = _resolve_choice_for_arm(arm, choice, akelka)
                if resolved is None:
                    continue
                p, ak, bb, lab = resolved
                targets.append((arm.name, p, ak, bb, lab))
            _OPEN["arm"] = ""

            def _later_batch():
                try:
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        _enqueue_swap_batch(bpy.context, targets, window, area, region)
                except Exception as exc:
                    print(f"[GR2 LAB] race batch failed: {exc}", flush=True)
                return None

            bpy.app.timers.register(_later_batch, first_interval=0.01)
        else:

            def _later():
                try:
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        _start_swap(bpy.context, arm_name, path, akelka, body, choice)
                except Exception as exc:
                    print(f"[GR2 LAB] race swap failed: {exc}", flush=True)
                return None

            bpy.app.timers.register(_later, first_interval=0.01)
        _tag()
        return True
    return False


def _tag() -> None:
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _matched_body(path: Path) -> str:
    try:
        from bind_pose import match_body_base_for_source  # type: ignore

        return str(match_body_base_for_source(path.name) or "")
    except Exception:
        return ""


def _start_swap(
    context,
    arm_name: str,
    path: str,
    akelka: bool = False,
    body_base: str = "",
    choice: str = "",
) -> None:
    arm = bpy.data.objects.get(arm_name)
    src = Path(path)
    if arm is None or not src.is_file():
        return
    _OPEN["arm"] = ""
    try:
        for obj in context.view_layer.objects:
            obj.select_set(False)
        arm.select_set(True)
        context.view_layer.objects.active = arm
    except Exception:
        pass
    if body_base:
        hint = f"Auto: {Path(body_base).name}"
    else:
        matched = _matched_body(src)
        hint = f"Auto: {Path(matched).name}" if matched else ""
    # Operator redraws re-match the body from the GR2 name and can drop a
    # hidden hint. The rig keeps the clicked body until import reads it.
    base_name = Path(body_base).name if body_base else ""
    if not base_name and hint.lower().startswith("auto:"):
        base_name = hint.split(":", 1)[1].strip()
    try:
        if base_name:
            arm["gr2lab_swap_body"] = base_name
            arm["gr2lab_swap_choice"] = choice or ""
            arm["gr2lab_swap_akelka"] = bool(akelka)
    except Exception:
        pass
    # The import panel re-matches body base from the GR2 name. A borrowed clip
    # (DGB_F AkELkA playing a DGB_M file) would be forced back to DGB_M.
    # bb_synced_for marks this filepath as already resolved.
    kwargs = dict(
        filepath=str(src),
        second_character=False,
        scene_empties=False,
        add_rig=False,
        import_apply="SWAP_RIG",
        import_meshes="BODY_CLOTH",
        import_tail=True,
        target_armature=arm.name,
        bb_hint=hint,
        import_akelka=bool(akelka),
        body_choice=choice,
        bb_synced_for=str(src),
    )
    base_name = Path(body_base).name if body_base else ""
    if base_name:
        kwargs["body_base"] = base_name
    bpy.ops.import_scene.gr2_full("EXEC_DEFAULT", **kwargs)


class GR2LAB_OT_race_picker_hover(bpy.types.Operator):
    """Highlight the race row under the cursor."""

    bl_idname = "gr2lab.race_picker_hover"
    bl_label = "GR2 race hover"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        note_pointer(context, event)
        return {"PASS_THROUGH"}


class GR2LAB_OT_race_picker(bpy.types.Operator):
    """Pick another race clip for this scene character."""

    bl_idname = "gr2lab.race_picker"
    bl_label = "GR2 race"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        from . import anim_browser

        if anim_browser.handle_strip_click(context, event):
            return {"FINISHED"}
        if try_race_click(context, event):
            return {"FINISHED"}
        return {"PASS_THROUGH"}


class GR2LAB_OT_race_picker_modal(bpy.types.Operator):
    """Catch viewport clicks on the race chip before tools / other keymaps."""

    bl_idname = "gr2lab.race_picker_modal"
    bl_label = "GR2 race modal"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        wm = context.window_manager
        booted = set(wm.get("gr2lab_race_modal_wins") or ())
        ptr = context.window.as_pointer()
        if ptr in booted:
            return {"CANCELLED"}
        booted.add(ptr)
        wm["gr2lab_race_modal_wins"] = list(booted)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _MODAL_WANT_STOP
        if _MODAL_WANT_STOP:
            _MODAL_WANT_STOP = False
            return {"CANCELLED"}
        if event.type == "ESC" and event.value == "PRESS" and _OPEN.get("arm"):
            _OPEN["arm"] = ""
            _invalidate_hit_cache()
            _tag()
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE":
            note_pointer(context, event)
        from . import anim_browser

        if anim_browser.modal_strip_event(context, event):
            return {"RUNNING_MODAL"}
        if (
            event.type == "LEFTMOUSE"
            and event.value == "PRESS"
            and not (event.shift or event.alt)
        ):
            found = anim_browser.view3d_under_pointer(context, event)
            if found is not None:
                area, region, mx, my = found
                if not anim_browser.pointer_over_strip(region, mx, my):
                    window = getattr(context, "window", None)
                    try:
                        with context.temp_override(window=window, area=area, region=region):
                            if try_race_click(bpy.context, event, mx, my):
                                return {"RUNNING_MODAL"}
                    except Exception:
                        traceback.print_exc()
        return {"PASS_THROUGH"}

    def cancel(self, context):
        wm = context.window_manager
        booted = set(wm.get("gr2lab_race_modal_wins") or ())
        booted.discard(context.window.as_pointer())
        wm["gr2lab_race_modal_wins"] = list(booted)
        return None


def _boot_race_modal() -> None:
    wm = bpy.context.window_manager
    booted = set(wm.get("gr2lab_race_modal_wins") or ())
    for window in wm.windows:
        ptr = window.as_pointer()
        if ptr in booted:
            continue
        launched = False
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type != "WINDOW":
                    continue
                try:
                    with bpy.context.temp_override(
                        window=window, area=area, region=region
                    ):
                        bpy.ops.gr2lab.race_picker_modal("INVOKE_DEFAULT")
                    launched = True
                    break
                except Exception:
                    continue
            if launched:
                break
    return None


def register() -> None:
    global _HANDLE
    bpy.utils.register_class(GR2LAB_OT_race_picker_hover)
    bpy.utils.register_class(GR2LAB_OT_race_picker)
    bpy.utils.register_class(GR2LAB_OT_race_picker_modal)
    if _HANDLE is None:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_PIXEL"
        )
    from .anim_browser import view3d_keyconfig

    wm = bpy.context.window_manager
    kc = view3d_keyconfig(wm)
    if kc is None:
        return
    km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    kmi = km.keymap_items.new("gr2lab.race_picker", "LEFTMOUSE", "PRESS", head=True)
    _KEYMAPS.append((km, kmi))
    km_hover = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    kmi_hover = km_hover.keymap_items.new(
        "gr2lab.race_picker_hover", "MOUSEMOVE", "ANY", head=False
    )
    _KEYMAPS.append((km_hover, kmi_hover))
    bpy.app.timers.register(_boot_race_modal, first_interval=0.35)
    bpy.app.timers.register(_boot_race_modal, first_interval=1.5)


def unregister() -> None:
    global _HANDLE, _MODAL_WANT_STOP
    _MODAL_WANT_STOP = True
    try:
        wm = bpy.context.window_manager
        wm["gr2lab_race_modal_wins"] = []
    except Exception:
        pass
    while _KEYMAPS:
        km, kmi = _KEYMAPS.pop()
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    if _HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, "WINDOW")
        except Exception:
            pass
        _HANDLE = None
    _HITS.clear()
    _OPEN["arm"] = ""
    _MOUSE["region"] = 0
    _MOUSE["hit"] = None
    for cls in (
        GR2LAB_OT_race_picker_modal,
        GR2LAB_OT_race_picker,
        GR2LAB_OT_race_picker_hover,
    ):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
