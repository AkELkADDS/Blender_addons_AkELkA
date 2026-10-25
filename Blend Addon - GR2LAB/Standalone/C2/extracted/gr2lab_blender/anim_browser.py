"""Bottom 3D-view browser for unpacked BG3 cinematic clips.

Hover renders five camera stills of the whole scene in a separate Blender.
The open file is not modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import struct
import subprocess
import threading
import time
import traceback
import contextlib
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import (
    region_2d_to_origin_3d,
    region_2d_to_vector_3d,
)
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.geometry import intersect_line_plane

from .import_modal import _DECODE_LOCK

PREVIEW_REV = 4
PREVIEW_MAX_STILLS = 24
PREVIEW_PLAY_FPS = 12.0
PREVIEW_ASPECT = 16.0 / 9.0
PREVIEW_RENDER_W = 512
PREVIEW_RENDER_H = 288

STICK_SAMPLE_FPS = 50.0
STICK_MAX_FRAMES = 300
STICK_CACHE_VER = 5
STICK_HOVER_DEBOUNCE = 0.11
STICK_PARTNER_SHIFT = 1.15
STRIP_MARGIN_X = 8.0
STRIP_MARGIN_Y = 6.0
STRIP_HEIGHT = 224.0
PANE_SCENE = 0.30
PANE_CLIP = 0.46
CLIP_DRAG_THRESHOLD = 10.0
_TOOLBAR_CHIP_KINDS = frozenset(
    {
        "act",
        "camp",
        "rom",
        "whole",
        "center",
        "edit",
        "explorer",
        "prev",
        "next",
        "prepare",
    }
)
# Drag draws a hologram only. The rig is imported after the mouse button goes up.
CLIP_PRELOAD_TIMER_HZ = 20.0
CLIP_PRELOAD_PUMP_EVERY = 1
CLIP_PRELOAD_COMMIT_SLICES = 16
CLIP_PRELOAD_BUDGET_MOVE_S = 0.006
CLIP_PRELOAD_BUDGET_PAUSE_S = 0.014
CLIP_PRELOAD_BUDGET_IDLE_S = 0.02
CLIP_PRELOAD_SLICES_MOVE = 1
CLIP_PRELOAD_SLICES_PAUSE = 3
CLIP_PRELOAD_SLICES_IDLE = 5
CLIP_DRAG_HIT_INTERVAL_S = 0.04
CLIP_DRAG_HIT_MIN_PX = 28.0
CLIP_DRAG_RIPPLE_LITE_COUNT = 2
CLIP_DRAG_RIPPLE_LITE_SEGMENTS = 36
CLIP_DROP_LIFT = 0.002
CLIP_DROP_LINE_PX = 1.35
CLIP_DROP_RIPPLE_COUNT = 6
CLIP_DROP_RIPPLE_MAX_R = 0.78
CLIP_DROP_RIPPLE_SPEED = 0.36
CLIP_DROP_RIPPLE_SEGMENTS = 72
CLIP_DROP_RIPPLE_ALPHA_PEAK = 0.42
CLIP_DROP_BRACKET_HALF_U = 0.4
CLIP_DROP_BRACKET_HALF_V = 0.3
CLIP_DROP_BRACKET_ARM = 0.22
CLIP_DROP_STAKE_HEIGHT = 1.65
_DROP_WHITE = (1.0, 1.0, 1.0, 0.94)

_WM_CLIP_DROP_KEY = "gr2lab_clip_drop_pending"

_HANDLE = None
_VIEW_DRAG_HANDLE = None
_KEYMAPS: List[Tuple[object, object]] = []
_ALIVE = False
_TIMER_ON = False

_TREES = (
    Path("Models/Public/SharedDev/Assets/Characters/_Anims/_Cinematic"),
    Path("Models/Public/Shared/Assets/Characters/_Anims/_Cinematic"),
)
_COMPANIONS = (
    "Karlach",
    "Shadowheart",
    "Gale",
    "Laezel",
    "Wyll",
    "Astarion",
    "Halsin",
    "Minthara",
    "Jaheira",
    "Minsc",
)
_TAG_WORDS = {"CAMP", "ROM", "CRD", "SD", "SEX", "DATE"}
_ACTS = ("All", "Act01", "Act02", "Act03")

_PREP_LOCK = threading.Lock()
_PREP = {
    "running": False,
    "scenes": 0,
    "phase": "",
    "error": "",
    "done": False,
    "cancel": False,
}

_STATE = {
    "act": "All",
    "camp": False,
    "rom": False,
    "query": "",
    "search_focus": False,
    "search_modal": False,
    "search_sel": False,
    "search_caret": 0,
    "search_scroll": 0.0,
    "search_blink_phase": -1,
    "search_refilter_at": 0.0,
    "tag_pending": False,
    "scene_scroll": 0,
    "clip_scroll": 0,
    "scene_id": "",
    "whole_scene": True,
    "center_parent": False,
    "edit_copy": False,
    "filtered_key": "",
    "filtered": [],
    "pane_scene": PANE_SCENE,
    "pane_clip": PANE_CLIP,
}
_POINTER = {"mx": 0.0, "my": 0.0, "over_strip": False}
_INDEX = {"mtime": 0.0, "data": None}
_HITS: Dict[int, list] = {}
_LAYOUTS: Dict[int, dict] = {}
_DRAG = {
    "active": False,
    "filepath": "",
    "label": "",
    "region": 0,
    "sx": 0.0,
    "sy": 0.0,
    "mx": 0.0,
    "my": 0.0,
    "point": None,
    "over_strip": False,
    "shift": False,
    "normal": None,
    "region_w": 0.0,
    "region_h": 0.0,
    "t0": 0.0,
    "wm_timer": None,
    "preload": None,
    "preload_failed": "",
}
_HOVER = {
    "path": "",
    "since": 0.0,
    "thread": None,
    "gen": 0,
    "result": None,
    "shown": "",
    "poses": None,
    "pose_i": 0,
    "pose_tick": 0.0,
    "status": "",
    "mem": {},
}


def cache_dir() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home())
    path = Path(local) / "gr2lab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def index_path() -> Path:
    return cache_dir() / "anim_index.json"


def pose_dir() -> Path:
    path = cache_dir() / "preview_poses"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shader():
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _rect(x, y, w, h, color) -> None:
    if w <= 0 or h <= 0:
        return
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


def _lines(pairs, color) -> None:
    if len(pairs) < 2:
        return
    shader = _shader()
    batch = batch_for_shader(shader, "LINES", {"pos": pairs})
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _polyline_2d(
    points: Tuple[Tuple[float, float], ...],
    color: Tuple[float, float, float, float],
    width_px: float,
) -> None:
    if len(points) < 2:
        return
    viewport = _gpu_viewport_size()
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": points})
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("viewportSize", viewport)
    shader.uniform_float("lineWidth", float(width_px))
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


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


def _clip_text(font_id: int, text: str, width: float) -> str:
    if _text_width(font_id, text) <= width:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = text[:mid] + ell
        if _text_width(font_id, trial) <= width:
            best = trial
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _tag(min_interval: float = 0.0) -> None:
    if min_interval > 0.0:
        now = time.perf_counter()
        wait = float(_STATE.get("tag_wait_until") or 0.0)
        if now < wait:
            _STATE["tag_pending"] = True
            return
        _STATE["tag_wait_until"] = now + min_interval
    _STATE["tag_pending"] = False
    try:
        wm = bpy.context.window_manager
    except Exception:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _flush_tag_pending() -> None:
    if _STATE.get("tag_pending"):
        _STATE["tag_wait_until"] = 0.0
        _tag()


def _prefs_unpacked_raw(context) -> str:
    try:
        addon = context.preferences.addons.get("gr2lab_blender")
        prefs = addon.preferences if addon is not None else None
        return (getattr(prefs, "bg3_unpacked_dir", "") or "").strip() if prefs else ""
    except Exception:
        return ""


def _set_prefs_unpacked_dir(context, directory: str) -> None:
    path = str(directory or "").strip()
    if not path:
        return
    try:
        addon = context.preferences.addons.get("gr2lab_blender")
        prefs = addon.preferences if addon is not None else None
        if prefs is not None and hasattr(prefs, "bg3_unpacked_dir"):
            prefs.bg3_unpacked_dir = path
    except Exception:
        pass


def unpacked_root(context) -> Optional[Path]:
    """Addon prefs first, then env / Glasses MCP / default UnpackedData roots."""
    extra = []
    raw = _prefs_unpacked_raw(context)
    if raw:
        extra.append(raw)
    try:
        from . import bg3_unpacked_resolve as unpacked

        roots = unpacked.unpacked_roots(extra or None)
    except Exception:
        roots = []
    if roots:
        return roots[0]
    if raw:
        path = Path(raw)
        return path if path.is_dir() else None
    return None


def _search_roots(root: Path) -> List[Path]:
    found = [root]
    if root.name.lower() == "gustav" and root.parent.is_dir():
        found.append(root.parent)
    return found


def _act_of(path: Path) -> str:
    for part in path.parts:
        low = part.lower()
        if low.startswith("act") and len(part) >= 4 and part[3:].isdigit():
            return "Act" + part[3:]
    return ""


def _tags_of(path: Path) -> List[str]:
    tags = set()
    blob = path.name
    for token in blob.replace("-", "_").split("_"):
        word = token.strip()
        if word.upper() in _TAG_WORDS:
            tags.add(word.upper())
        for companion in _COMPANIONS:
            if word.lower() == companion.lower():
                tags.add(companion)
    for part in path.parts:
        if part.upper() == "CAMP":
            tags.add("CAMP")
    return sorted(tags)


def _role(name: str) -> str:
    upper = name.upper()
    if "PLYR" in upper:
        return "player"
    for companion in _COMPANIONS:
        if companion.upper() in upper:
            return "partner"
    return "other"


def _scene_dual_char(clips: List[str]) -> bool:
    has_player = False
    has_partner = False
    for name in clips or []:
        role = _role(name)
        if role == "player":
            has_player = True
        elif role == "partner":
            has_partner = True
        if has_player and has_partner:
            return True
    return False


def _role_clip_colors(name: str, dual: bool) -> Tuple[str, Tuple[float, float, float]]:
    role = _role(name)
    if not dual:
        return role, (0.94, 0.95, 0.96)
    if role == "player":
        return "you", (0.45, 0.82, 1.0)
    if role == "partner":
        return "npc", (0.98, 0.72, 0.42)
    return role, (0.78, 0.80, 0.84)


_CHAR_DUP = {"t": 0.0, "c": ""}
_SEARCH_HISTORY = {"undo": [], "redo": [], "max": 80}


def _search_snapshot() -> dict:
    return {
        "query": _STATE.get("query") or "",
        "caret": _search_caret_index(),
        "sel": bool(_STATE.get("search_sel")),
    }


def _search_history_begin() -> None:
    _SEARCH_HISTORY["undo"] = []
    _SEARCH_HISTORY["redo"] = []


def _search_active() -> bool:
    return bool(_STATE.get("search_focus") or _STATE.get("search_modal"))


def _search_push_undo() -> None:
    snap = _search_snapshot()
    stack = _SEARCH_HISTORY["undo"]
    stack.append(snap)
    if len(stack) > int(_SEARCH_HISTORY["max"]):
        stack.pop(0)
    _SEARCH_HISTORY["redo"] = []


def _restore_search_snapshot(snap: dict) -> None:
    _STATE["search_sel"] = bool(snap.get("sel"))
    text = snap.get("query") or ""
    caret = int(snap.get("caret") or 0)
    _STATE["query"] = text
    _STATE["filtered_key"] = ""
    _STATE["search_refilter_at"] = time.perf_counter() + 0.02
    _set_search_caret(caret)
    _STATE["tag_wait_until"] = 0.0
    _tag()


def _undo_search() -> None:
    stack = _SEARCH_HISTORY["undo"]
    if not stack:
        return
    _SEARCH_HISTORY["redo"].append(_search_snapshot())
    snap = stack.pop()
    _restore_search_snapshot(snap)


def _redo_search() -> None:
    stack = _SEARCH_HISTORY["redo"]
    if not stack:
        return
    _SEARCH_HISTORY["undo"].append(_search_snapshot())
    snap = stack.pop()
    _restore_search_snapshot(snap)


def _apply_search_edit(text: str, *, refilter: bool = True, record_undo: bool = True) -> None:
    prev = _STATE.get("query") or ""
    if text == prev:
        return
    if record_undo:
        _search_push_undo()
    _STATE["query"] = text
    if refilter:
        _STATE["search_refilter_at"] = time.perf_counter() + 0.05
    _STATE["tag_wait_until"] = 0.0
    _tag()


def _search_caret_index() -> int:
    query = _STATE.get("query") or ""
    return max(0, min(len(query), int(_STATE.get("search_caret") or 0)))


def _set_search_caret(index: int) -> None:
    query = _STATE.get("query") or ""
    _STATE["search_caret"] = max(0, min(len(query), int(index)))


def _insert_search(text: str) -> None:
    if not text:
        return
    query = _STATE.get("query") or ""
    if _STATE.get("search_sel"):
        _STATE["search_sel"] = False
        _apply_search_edit(text)
        _set_search_caret(len(text))
        return
    index = _search_caret_index()
    _apply_search_edit(query[:index] + text + query[index:])
    _set_search_caret(index + len(text))


def _place_search_caret(region, mx: float) -> None:
    query = _STATE.get("query") or ""
    layout = _LAYOUTS.get(region.as_pointer() if region else 0) or {}
    box = layout.get("search_box")
    font_id = _font(12)
    scroll = float(_STATE.get("search_scroll") or 0.0)
    origin = (float(box[0]) + 8.0) if box else 0.0
    local = float(mx) - origin + scroll
    _STATE["search_sel"] = False
    if not query or local <= 0.0:
        _set_search_caret(0)
        _tag()
        return
    best = len(query)
    prev_w = 0.0
    for index in range(1, len(query) + 1):
        width = _text_width(font_id, query[:index])
        if local <= width:
            best = index - 1 if (local - prev_w) < (width - local) else index
            break
        prev_w = width
    _set_search_caret(best)
    _tag()


_SEARCH_KEY_CHARS = {
    "SPACE": " ",
    "MINUS": "-",
    "NUMPAD_MINUS": "-",
    "PERIOD": ".",
    "NUMPAD_PERIOD": ".",
    "COMMA": ",",
    "EQUAL": "=",
    "NUMPAD_EQUAL": "=",
    "SLASH": "/",
    "BACK_SLASH": "\\",
    "SEMI_COLON": ";",
    "QUOTE": "'",
    "ACCENT_GRAVE": "`",
    "LEFT_BRACKET": "[",
    "RIGHT_BRACKET": "]",
}


def _search_event_char(event) -> str:
    if event.type == "TEXTINPUT":
        return (getattr(event, "unicode", "") or "").replace("\r", "").replace("\n", "")
    if event.value not in {"PRESS", "REPEAT"}:
        return ""
    if event.ctrl or event.alt or event.oskey:
        return ""
    if len(event.type) == 1:
        return event.type
    mapped = _SEARCH_KEY_CHARS.get(event.type)
    if mapped is not None:
        return mapped
    ascii = getattr(event, "ascii", "") or ""
    if len(ascii) == 1 and ascii.isprintable():
        return ascii
    return ""


def _push_search_char(char: str) -> None:
    now = time.perf_counter()
    if now < float(_CHAR_DUP.get("skip_until") or 0.0):
        return
    if char == _CHAR_DUP["c"] and (now - float(_CHAR_DUP["t"])) < 0.04:
        return
    _CHAR_DUP["t"] = now
    _CHAR_DUP["c"] = char
    _insert_search(char)


def _clipboard_text() -> str:
    try:
        raw = bpy.context.window_manager.clipboard or ""
    except Exception:
        return ""
    return " ".join(str(raw).replace("\r", " ").replace("\n", " ").split())


def _paste_search() -> None:
    text = _clipboard_text()
    _CHAR_DUP["skip_until"] = time.perf_counter() + 0.08
    if not text:
        _tag()
        return
    _insert_search(text)


def _copy_search() -> None:
    query = _STATE.get("query") or ""
    if not query:
        return
    if not _STATE.get("search_sel"):
        _select_all_search()
    try:
        bpy.context.window_manager.clipboard = _STATE.get("query") or ""
    except Exception:
        pass


def _cut_search() -> None:
    query = _STATE.get("query") or ""
    if not query:
        return
    if not _STATE.get("search_sel"):
        _select_all_search()
    try:
        bpy.context.window_manager.clipboard = query
    except Exception:
        pass
    _STATE["search_sel"] = False
    _apply_search_edit("")
    _set_search_caret(0)


def _search_shortcut(event) -> bool:
    if event.alt:
        return False
    if not (event.ctrl or event.oskey):
        return False
    if event.value not in {"PRESS", "CLICK"}:
        return False
    key = str(event.type or "")
    if len(key) == 1:
        key = key.upper()
    if key == "A":
        _select_all_search()
        return True
    if key == "C":
        _copy_search()
        return True
    if key == "X":
        _cut_search()
        return True
    if key == "V":
        _paste_search()
        return True
    if key == "INSERT":
        _paste_search()
        return True
    if key in {"DELETE", "DEL"}:
        _cut_search()
        return True
    if key == "Z":
        if event.shift:
            _redo_search()
        else:
            _undo_search()
        return True
    if key == "Y":
        _redo_search()
        return True
    return True


def _select_all_search() -> None:
    query = _STATE.get("query") or ""
    if query:
        _STATE["search_sel"] = True
        _set_search_caret(len(query))
    _CHAR_DUP["skip_until"] = time.perf_counter() + 0.08
    _tag()


def _backspace_search() -> None:
    if _STATE.get("search_sel"):
        _STATE["search_sel"] = False
        _apply_search_edit("")
        _set_search_caret(0)
        return
    query = _STATE.get("query") or ""
    index = _search_caret_index()
    if index <= 0:
        return
    _apply_search_edit(query[: index - 1] + query[index:])
    _set_search_caret(index - 1)


def _delete_search() -> None:
    if _STATE.get("search_sel"):
        _STATE["search_sel"] = False
        _apply_search_edit("")
        _set_search_caret(0)
        return
    query = _STATE.get("query") or ""
    index = _search_caret_index()
    if index >= len(query):
        return
    _apply_search_edit(query[:index] + query[index + 1 :])


def _move_search_caret(delta: int) -> None:
    _STATE["search_sel"] = False
    _set_search_caret(_search_caret_index() + delta)
    _tag(0.028)


def _begin_gpu_search(region=None, mx: Optional[float] = None) -> set:
    if region is not None and mx is not None:
        _place_search_caret(region, mx)
    else:
        _set_search_caret(len(_STATE.get("query") or ""))
    _STATE["search_focus"] = True
    _tag()
    if _STATE.get("search_modal"):
        return {"FINISHED"}
    try:
        bpy.ops.gr2lab.anim_browser_type("INVOKE_DEFAULT")
    except Exception:
        return {"CANCELLED"}
    return {"FINISHED"}


def build_index(root: Path, cancel) -> Optional[dict]:
    """Walk cinematic folders. Does not open GR2 files."""
    scenes = []
    seen = set()
    for base in _search_roots(root):
        for rel in _TREES:
            tree = base / rel
            if not tree.is_dir():
                continue
            tree_key = str(tree.resolve()).lower()
            if tree_key in seen:
                continue
            seen.add(tree_key)
            with _PREP_LOCK:
                _PREP["phase"] = tree.name
            for dirpath, _dirnames, filenames in os.walk(tree):
                if cancel():
                    return None
                clips = sorted(name for name in filenames if name.lower().endswith(".gr2"))
                if not clips:
                    continue
                folder = Path(dirpath)
                act = _act_of(folder)
                scenes.append(
                    {
                        "id": str(folder).lower(),
                        "name": folder.name,
                        "act": act,
                        "tags": _tags_of(folder),
                        "dir": str(folder),
                        "clips": clips,
                    }
                )
                if len(scenes) % 200 == 0:
                    with _PREP_LOCK:
                        _PREP["scenes"] = len(scenes)
    scenes.sort(key=lambda row: (row.get("act") or "ZZZ", row.get("name") or ""))
    return {
        "version": 1,
        "unpacked": str(root),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenes": scenes,
    }


def _load_index() -> Optional[dict]:
    path = index_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _INDEX["data"] = None
        _INDEX["mtime"] = 0.0
        return None
    if _INDEX["data"] is not None and _INDEX["mtime"] == mtime:
        return _INDEX["data"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    _INDEX["data"] = data
    _INDEX["mtime"] = mtime
    return data


def _index_matches(context, data: Optional[dict]) -> bool:
    if not data:
        return False
    root = unpacked_root(context)
    if root is None:
        return False
    stored = str(data.get("unpacked") or "")
    try:
        return Path(stored).resolve() == root.resolve()
    except OSError:
        return stored.lower() == str(root).lower()


def _search_tokens(query: str) -> List[str]:
    return [part for part in (query or "").casefold().split() if part]


def _normalize_search_blob(text: str) -> str:
    """Fold separators so CAMP_Karlach_CRD_Date matches 'karlach date'."""
    chars: List[str] = []
    for ch in (text or "").casefold():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def _text_has_tokens(text: str, tokens: List[str]) -> bool:
    if not tokens:
        return True
    blob = _normalize_search_blob(text)
    return all(token in blob for token in tokens)


def _scene_has_tokens(scene: dict, tokens: List[str]) -> bool:
    if not tokens:
        return True
    parts = [scene.get("name") or ""]
    parts.extend(scene.get("clips") or [])
    return _text_has_tokens(" ".join(parts), tokens)


def _filtered(data: dict) -> List[dict]:
    key = "|".join(
        (
            str(_INDEX["mtime"]),
            _STATE["act"],
            "1" if _STATE["camp"] else "0",
            "1" if _STATE["rom"] else "0",
            _STATE["query"].casefold(),
        )
    )
    if key == _STATE["filtered_key"]:
        return _STATE["filtered"]
    query = _STATE["query"].strip()
    tokens = _search_tokens(query)
    act = _STATE["act"]
    rows = []
    for scene in data.get("scenes") or []:
        if act != "All" and (scene.get("act") or "") != act:
            continue
        tags = set(scene.get("tags") or [])
        if _STATE["camp"] and "CAMP" not in tags:
            continue
        if _STATE["rom"] and "ROM" not in tags:
            continue
        if tokens and not _scene_has_tokens(scene, tokens):
            continue
        rows.append(scene)
    _STATE["filtered_key"] = key
    _STATE["filtered"] = rows
    if _STATE["scene_id"] not in {row["id"] for row in rows}:
        _STATE["scene_id"] = rows[0]["id"] if rows else ""
        _STATE["clip_scroll"] = 0
    return rows


def _selected(rows: List[dict]) -> Optional[dict]:
    want = _STATE["scene_id"]
    for row in rows:
        if row["id"] == want:
            return row
    return rows[0] if rows else None


def _pose_file(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:24]
    return pose_dir() / f"{key}.json"


def _remember_pose(path: str, data: dict) -> None:
    mem = _HOVER["mem"]
    mem[path] = data
    if len(mem) > 32:
        mem.pop(next(iter(mem)))
    if data.get("kind") != "scene":
        return
    try:
        _pose_file(Path(path)).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def _cached_pose(path: str) -> Optional[dict]:
    mem = _HOVER["mem"].get(path)
    if mem:
        return mem
    try:
        file_path = _pose_file(Path(path))
        if not file_path.is_file():
            return None
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shots = data.get("shots") or []
    if data.get("kind") != "scene" or not data.get("duration_sec") or len(shots) < 8:
        return None
    if not all(Path(shot).is_file() for shot in shots):
        return None
    _HOVER["mem"][path] = data
    return data


def _stick_attach_timing(data: dict) -> dict:
    poses = data.get("poses") or []
    data["_pose_times"] = _stick_pose_times(data, poses)
    return data


def _show_pose(path: str, data: dict, status: str) -> None:
    if data.get("kind") == "sticks":
        data = _stick_attach_timing(data)
    _HOVER["shown"] = path
    _HOVER["poses"] = data
    _HOVER["pose_i"] = 0
    _HOVER["pose_tick"] = time.perf_counter()
    _HOVER["textures"] = {}
    _HOVER["status"] = status
    _tag()


def _shot_dir(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(
        f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|pv{PREVIEW_REV}".encode()
    ).hexdigest()[:24]
    folder = cache_dir() / "preview_shots" / key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _stick_cache_file(gr2_path: str) -> Path:
    src = Path(_preview_source(gr2_path))
    partner = _partner_gr2_fast(gr2_path)
    blob = str(src)
    if partner is not None:
        blob = f"{blob}|{partner}"
    st = src.stat()
    key = hashlib.sha1(
        f"{blob}|{st.st_mtime_ns}|{st.st_size}|sc{STICK_CACHE_VER}|{STICK_SAMPLE_FPS}".encode()
    ).hexdigest()[:24]
    folder = cache_dir() / "preview_sticks"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.stk"


def _stick_mem_get(path: str) -> Optional[dict]:
    if not path:
        return None
    mem = _HOVER["mem"].get(path)
    if mem and mem.get("kind") == "sticks":
        return mem
    return None


def _stick_load_disk(path: str) -> Optional[dict]:
    if not path:
        return None
    try:
        file_path = _stick_cache_file(path)
        if not file_path.is_file():
            return None
        with file_path.open("rb") as handle:
            data = pickle.load(handle)
    except (OSError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(data, dict) or data.get("kind") != "sticks" or not data.get("poses"):
        return None
    data = _stick_attach_timing(dict(data))
    _HOVER["mem"][path] = data
    return data


def _remember_stick(path: str, data: dict) -> None:
    _HOVER["mem"][path] = data
    if len(_HOVER["mem"]) > 24:
        _HOVER["mem"].pop(next(iter(_HOVER["mem"])))
    try:
        packed = {
            key: value
            for key, value in data.items()
            if key not in ("_pose_times", "seconds", "cached", "path")
        }
        poses = []
        for pose in packed.get("poses") or []:
            slim = {"t": pose.get("t"), "frac": pose.get("frac"), "rig": pose.get("rig")}
            if pose.get("points"):
                slim["points"] = pose.get("points")
            poses.append(slim)
        packed["poses"] = poses
        with _stick_cache_file(path).open("wb") as handle:
            pickle.dump(packed, handle, protocol=4)
    except OSError:
        pass


def _partner_gr2_fast(gr2_path: str) -> Optional[Path]:
    folder = Path(gr2_path).parent
    try:
        mine = Path(gr2_path).resolve()
    except OSError:
        mine = Path(gr2_path)
    for cand in sorted(folder.glob("*.GR2")) + sorted(folder.glob("*.gr2")):
        try:
            if cand.resolve() == mine:
                continue
        except OSError:
            if cand == Path(gr2_path):
                continue
        upper = cand.name.upper()
        if "PLYR" in upper:
            continue
        if any(token in upper for token in ("HUM_", "TIF_", "DGB_", "GTY_", "HFL_", "GOB_", "ELF_")):
            return cand
    return None


def _merge_stick_pair(player: dict, partner: dict) -> dict:
    merged = []
    p_edges = [[f"P:{a}", f"P:{b}"] for a, b in (partner.get("edges") or [])]
    p_order = [f"P:{n}" for n in (partner.get("bone_order") or [])]
    p_parents = {
        f"P:{k}": (f"P:{v}" if v else "")
        for k, v in (partner.get("parents") or {}).items()
    }
    bone_order = list(player.get("bone_order") or []) + p_order
    parents = {**(player.get("parents") or {}), **p_parents}
    for pose, other in zip(player.get("poses") or [], partner.get("poses") or []):
        rig = dict(pose.get("rig") or {})
        for name, tr in (other.get("rig") or {}).items():
            rig[f"P:{name}"] = tr
        merged.append({"t": pose.get("t"), "frac": pose.get("frac"), "rig": rig})
    return {
        **player,
        "poses": merged,
        "edges": list(player.get("edges") or []) + p_edges,
        "bones": int(player.get("bones") or 0) + int(partner.get("bones") or 0),
        "bone_order": bone_order,
        "parents": parents,
    }


def _stick_preview_job(path: str) -> Tuple[str, Optional[dict], str]:
    from .local_pipeline import preview_stick_poses, stick_sample_fractions

    cached = _stick_mem_get(path) or _stick_load_disk(path)
    if cached is not None:
        cached = dict(cached)
        cached["seconds"] = 0.0
        cached["cached"] = True
        return path, cached, ""

    try:
        source = _preview_source(path)
        with _DECODE_LOCK:
            player = preview_stick_poses(
                source,
                sample_fps=STICK_SAMPLE_FPS,
                max_frames=STICK_MAX_FRAMES,
            )
            partner = _partner_gr2_fast(path)
            if partner is not None:
                duration = float(player.get("duration") or 0.0)
                fractions = stick_sample_fractions(
                    duration, fps=STICK_SAMPLE_FPS, max_frames=STICK_MAX_FRAMES
                )
                other = preview_stick_poses(str(partner), fractions=fractions)
                player = _merge_stick_pair(player, other)
        duration = float(player.get("duration") or 0.0)
        player["kind"] = "sticks"
        player["duration_sec"] = duration
        player["play_fps"] = len(player.get("poses") or []) / duration if duration > 0 else STICK_SAMPLE_FPS
        player = _stick_attach_timing(player)
        _remember_stick(path, player)
        return path, player, ""
    except Exception as exc:
        traceback.print_exc()
        return path, None, str(exc)


def _preview_source(path: str) -> str:
    """Character clip for a whole-scene preview. Props borrow the player rig."""
    file_path = Path(path)
    upper = file_path.name.upper()
    if any(token in upper for token in ("HUM_", "TIF_", "DGB_", "GTY_", "HFL_", "GOB_", "ELF_")):
        return str(file_path)
    try:
        clips = sorted(file_path.parent.glob("*.GR2")) + sorted(file_path.parent.glob("*.gr2"))
    except OSError:
        return str(file_path)
    for cand in clips:
        name = cand.name.upper()
        if "PLYR" in name and any(token in name for token in ("HUM_", "TIF_", "DGB_")):
            return str(cand)
    return str(file_path)


def _snapshot_ids() -> dict:
    return {
        "objects": {obj.name for obj in bpy.data.objects},
        "meshes": {mesh.name for mesh in bpy.data.meshes},
        "armatures": {arm.name for arm in bpy.data.armatures},
        "cameras": {cam.name for cam in bpy.data.cameras},
        "actions": {action.name for action in bpy.data.actions},
        "collections": {col.name for col in bpy.data.collections},
        "worlds": {world.name for world in bpy.data.worlds},
        "scenes": {scene.name for scene in bpy.data.scenes},
        "materials": {mat.name for mat in bpy.data.materials},
        "images": {image.name for image in bpy.data.images},
        "node_groups": {group.name for group in bpy.data.node_groups},
    }


def _drop_new_ids(before: dict) -> None:
    for obj in list(bpy.data.objects):
        if obj.name not in before["objects"]:
            bpy.data.objects.remove(obj, do_unlink=True)
    window = getattr(bpy.context, "window", None)
    window_scene = window.scene if window is not None else bpy.context.scene
    for scene in list(bpy.data.scenes):
        if scene.name not in before["scenes"] and scene != window_scene:
            bpy.data.scenes.remove(scene)
    banks = (
        ("collections", bpy.data.collections),
        ("meshes", bpy.data.meshes),
        ("armatures", bpy.data.armatures),
        ("cameras", bpy.data.cameras),
        ("materials", bpy.data.materials),
        ("images", bpy.data.images),
        ("actions", bpy.data.actions),
        ("worlds", bpy.data.worlds),
        ("node_groups", bpy.data.node_groups),
    )
    for _pass in range(4):
        for key, datablocks in banks:
            known = before.get(key) or set()
            for block in list(datablocks):
                if block.name in known:
                    continue
                if getattr(block, "use_fake_user", False):
                    try:
                        block.use_fake_user = False
                    except Exception:
                        pass
                if block.users == 0:
                    datablocks.remove(block)


def _glasses_sidecar() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(local) / "GlassesMcp" / "gr2lab_preview.json"


def _lookat_camera(scene, camera):
    from mathutils import Vector

    points = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        camera.location = (4.0, -4.0, 1.6)
        return
    center = sum(points, Vector()) / len(points)
    size = max(
        max(point.x for point in points) - min(point.x for point in points),
        max(point.y for point in points) - min(point.y for point in points),
        max(point.z for point in points) - min(point.z for point in points),
        1.0,
    )
    distance = size * 2.1
    camera.location = center + Vector((distance * 0.45, -distance, size * 0.2))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 36.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = max(distance * 40.0, 1000.0)


def _play_frames(start: int, end: int, scene_fps: float) -> Tuple[List[int], float, float]:
    """Sparse stills that play back in real time (cap count for build speed)."""
    count = max(1, end - start + 1)
    fps = float(scene_fps or 30.0)
    duration = count / fps
    step = max(1, int(math.ceil(count / PREVIEW_MAX_STILLS)))
    step = max(step, max(1, int(round(fps / PREVIEW_PLAY_FPS))))
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    play_fps = len(frames) / duration if duration > 0 else PREVIEW_PLAY_FPS
    return frames, duration, play_fps


def _fit_aspect_rect(rect) -> Tuple[float, float, float, float]:
    """16:9 draw rect inside the preview tile (no stretch)."""
    x, y, w, h = rect
    if w <= 1.0 or h <= 1.0:
        return rect
    if w / h > PREVIEW_ASPECT:
        fit_w = h * PREVIEW_ASPECT
        fit_h = h
    else:
        fit_w = w
        fit_h = w / PREVIEW_ASPECT
    return (x + (w - fit_w) * 0.5, y + (h - fit_h) * 0.5, fit_w, fit_h)


def _read_play_manifest(folder: Path, source: str) -> Optional[dict]:
    path = folder / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    count = int(data.get("count") or 0)
    if int(data.get("preview_rev") or 0) != PREVIEW_REV:
        return None
    if count < 8 or not data.get("duration_sec"):
        return None
    shots = [folder / f"{index}.png" for index in range(count)]
    if not all(shot.is_file() and shot.stat().st_size > 0 for shot in shots):
        return None
    data = dict(data)
    data["shots"] = [str(shot) for shot in shots]
    data["seconds"] = 0.0
    data["cached"] = True
    data["source"] = source
    data["kind"] = "scene"
    return data


def build_scene_preview(context, gr2_path: str, *, inplace: bool = False, fast: bool = True) -> dict:
    """Render the clip as a playback sequence, then delete that preview."""
    from . import scene_stage_empties
    from .bridge_blender import (
        _apply_layout_from_prefs,
        _resolve_body_base_for_import,
        import_gr2lab_from_doc,
    )
    from .local_pipeline import import_gr2_pipeline_local

    source = _preview_source(gr2_path)
    folder = _shot_dir(Path(source))
    cached = _read_play_manifest(folder, source)
    if cached is not None:
        return cached

    windows = list(getattr(context.window_manager, "windows", []) or [])
    window = getattr(context, "window", None) or (windows[0] if windows else None)
    user_scene = window.scene if window is not None else context.scene
    frame = user_scene.frame_current
    user_render = (
        user_scene.render.filepath,
        user_scene.render.resolution_x,
        user_scene.render.resolution_y,
        user_scene.render.resolution_percentage,
        user_scene.render.engine,
    )
    view_layer = window.view_layer if window is not None else user_scene.view_layers[0]
    user_range = (user_scene.frame_start, user_scene.frame_end)
    user_props = {
        key: user_scene[key]
        for key in ("gr2lab_glasses_preview", "gr2lab_glasses_time_sec")
        if key in user_scene.keys()
    }
    previous_active = view_layer.objects.active
    previous_mode = getattr(previous_active, "mode", "OBJECT") if previous_active else "OBJECT"
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None) if window is not None else None
    region = None
    if area is not None:
        region = next((item for item in area.regions if item.type == "WINDOW"), None)
    before = _snapshot_ids()
    sidecar = _glasses_sidecar()
    sidecar_bytes = sidecar.read_bytes() if sidecar.is_file() else None
    t0 = time.perf_counter()
    if inplace:
        preview = user_scene
        for name in ("Cube", "Light", "Camera"):
            obj = preview.objects.get(name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
        previous_active = None
        previous_mode = "OBJECT"
    else:
        preview = bpy.data.scenes.new("GR2LAB_Preview")
    preview.render.resolution_x = PREVIEW_RENDER_W
    preview.render.resolution_y = PREVIEW_RENDER_H
    preview.render.resolution_percentage = 100
    preview.render.image_settings.file_format = "PNG"
    preview.render.film_transparent = False
    try:
        preview.render.engine = "BLENDER_WORKBENCH"
    except TypeError:
        pass
    blocked = [
        name
        for name in ("Initiator_1", "Additional_1", "Additional_2", "Additional_3", "GR2LAB_Scene_Camera")
        if bpy.data.objects.get(name) is not None
    ]
    built = {}
    try:
        override = {"scene": preview, "view_layer": preview.view_layers[0]}
        if window is not None:
            override["window"] = window
        if area is not None and region is not None:
            override["area"] = area
            override["region"] = region
        runner = context.temp_override(**override) if not inplace else contextlib.nullcontext()
        with runner:
            print("STAGE layout", flush=True)
            _apply_layout_from_prefs(bpy.context)
            base = _resolve_body_base_for_import(
                choice="AUTO", bb_hint="", prefs_base="", source_gr2=source
            )
            print("STAGE decode", base, flush=True)
            _dec, doc = import_gr2_pipeline_local(source, body_base=base)
            print("STAGE primary", flush=True)
            import_gr2lab_from_doc(
                bpy.context,
                doc,
                fps=30,
                y_up_display=True,
                template_path=source,
                rig_display=False,
                rig_parts={"shapes": False, "ik_hands": False, "ik_legs": False, "ik_head": False},
                anim_apply="NEW_RIG",
                import_meshes="BODY_CLOTH",
                import_tail=not fast,
                mesh_base=base,
            )
            partner = None
            if fast:
                partner = _partner_gr2_fast(source)
                print("STAGE fast partner", partner, flush=True)
                if partner is not None:
                    pbase = _resolve_body_base_for_import(
                        choice="AUTO", bb_hint="", prefs_base="", source_gr2=str(partner)
                    )
                    _pdec, pdoc = import_gr2_pipeline_local(str(partner), body_base=pbase)
                    import_gr2lab_from_doc(
                        bpy.context,
                        pdoc,
                        fps=30,
                        y_up_display=True,
                        template_path=str(partner),
                        rig_display=False,
                        rig_parts={"shapes": False, "ik_hands": False, "ik_legs": False, "ik_head": False},
                        anim_apply="NEW_RIG",
                        import_meshes="BODY_CLOTH",
                        import_tail=False,
                        mesh_base=pbase,
                    )
            elif not blocked:
                print("STAGE plan", flush=True)
                plan = scene_stage_empties.plan_scene_empties(source)
                if plan is not None:
                    print("STAGE place", flush=True)
                    scene_stage_empties.place_for_import(
                        bpy.context,
                        source,
                        y_up_display=True,
                        parent_rig=True,
                        plan=plan,
                        defer_cinematic=True,
                        skip_cleanup=True,
                    )
                    partner = scene_stage_empties.find_partner_gr2(source, primary_plan=plan)
                    print("STAGE partner", partner, flush=True)
                    if partner is not None:
                        pbase = _resolve_body_base_for_import(
                            choice="AUTO", bb_hint="", prefs_base="", source_gr2=str(partner)
                        )
                        _pdec, pdoc = import_gr2_pipeline_local(str(partner), body_base=pbase)
                        import_gr2lab_from_doc(
                            bpy.context,
                            pdoc,
                            fps=30,
                            y_up_display=True,
                            template_path=str(partner),
                            rig_display=False,
                            rig_parts={"shapes": False, "ik_hands": False, "ik_legs": False, "ik_head": False},
                            anim_apply="NEW_RIG",
                            import_meshes="BODY_CLOTH",
                            import_tail=True,
                            mesh_base=pbase,
                        )
                    print("STAGE sync", flush=True)
                    col = bpy.data.collections.get(plan.stage.name)
                    if col is not None:
                        note = scene_stage_empties.apply_cinematic_timeline_sync(
                            bpy.context,
                            col,
                            source,
                            plan,
                            y_up_display=True,
                        )
                        print(
                            "SYNC",
                            note,
                            "phase",
                            plan.phase_index,
                            "cam",
                            bpy.data.objects.get("GR2LAB_Scene_Camera") is not None,
                            flush=True,
                        )
            camera = bpy.data.objects.get("GR2LAB_Scene_Camera")
            if camera is None or camera.name in before["objects"] or camera.type != "CAMERA":
                camera = bpy.data.objects.new("GR2LAB_Preview_Camera", bpy.data.cameras.new("GR2LAB_Preview_Camera"))
                preview.collection.objects.link(camera)
                _lookat_camera(preview, camera)
            preview.camera = camera
            for obj in preview.objects:
                if obj.type in {"EMPTY", "ARMATURE"}:
                    obj.hide_render = True
            start = int(preview.frame_start)
            end = max(start, int(preview.frame_end))
            scene_fps = float(preview.render.fps or 30)
            play_frames, duration_sec, play_fps = _play_frames(start, end, scene_fps)
            shots = [folder / f"{index}.png" for index in range(len(play_frames))]
            play_fps = round(play_fps, 3)
            built["mesh_render"] = sum(
                1 for obj in preview.objects if obj.type == "MESH" and not obj.hide_render
            )
            built["camera"] = camera.name
            built["span"] = (start, end)
            built["partner"] = Path(str(partner)).name if partner else ""
            built["play_fps"] = play_fps
            built["duration_sec"] = duration_sec
            built["play_frames"] = play_frames
            built["shots"] = shots
            fallback = camera.name == "GR2LAB_Preview_Camera"
            if fallback:
                preview.frame_set(play_frames[len(play_frames) // 2])
                preview.view_layers[0].update()
                _lookat_camera(preview, camera)
            print(
                "STAGE camera",
                camera.name,
                "meshes",
                built.get("mesh_render"),
                "frames",
                len(play_frames),
                "fps",
                play_fps,
                flush=True,
            )
            for index, frame_num in enumerate(play_frames):
                preview.frame_set(frame_num)
                preview.render.filepath = str(shots[index])
                preview.render.use_file_extension = True
                if index == 0 or index == len(play_frames) - 1 or index % 30 == 0:
                    print("STAGE render", index, frame_num, flush=True)
                bpy.ops.render.render(write_still=True)
    finally:
        if window is not None:
            window.scene = user_scene
        try:
            user_scene.frame_set(frame)
            user_scene.render.filepath = user_render[0]
            user_scene.render.resolution_x = user_render[1]
            user_scene.render.resolution_y = user_render[2]
            user_scene.render.resolution_percentage = user_render[3]
            user_scene.render.engine = user_render[4]
            user_scene.frame_start = user_range[0]
            user_scene.frame_end = user_range[1]
            for key in ("gr2lab_glasses_preview", "gr2lab_glasses_time_sec"):
                if key in user_props:
                    user_scene[key] = user_props[key]
                elif key in user_scene.keys():
                    del user_scene[key]
        except Exception:
            pass
        if previous_active is not None and previous_active.name in bpy.data.objects:
            try:
                view_layer.objects.active = previous_active
                if previous_mode and previous_mode != "OBJECT" and previous_active.mode != previous_mode:
                    with context.temp_override(window=window, area=area, region=region):
                        bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:
                pass
        _drop_new_ids(before)
        if sidecar_bytes is None:
            if sidecar.is_file():
                try:
                    sidecar.unlink()
                except OSError:
                    pass
        else:
            try:
                sidecar.write_bytes(sidecar_bytes)
            except OSError:
                pass
    shots = list(built.get("shots") or [])
    missing = [shot.name for shot in shots if not shot.is_file()]
    if len(shots) < 8 or missing:
        raise RuntimeError("scene preview did not write " + ", ".join(missing[:8] or ["frames"]))
    play_fps = float(built.get("play_fps") or 0.0)
    duration_sec = float(built.get("duration_sec") or 0.0)
    result = {
        "kind": "scene",
        "preview_rev": PREVIEW_REV,
        "shots": [str(shot) for shot in shots],
        "count": len(shots),
        "play_fps": play_fps,
        "duration_sec": duration_sec,
        "seconds": round(time.perf_counter() - t0, 3),
        "cached": False,
        "source": source,
        "blocked_scene": bool(blocked),
        "mesh_render": built.get("mesh_render", 0),
        "camera": built.get("camera", ""),
        "span": list(built.get("span") or []),
        "partner": built.get("partner", ""),
    }
    try:
        (folder / "manifest.json").write_text(
            json.dumps({key: value for key, value in result.items() if key != "shots"}),
            encoding="utf-8",
        )
    except OSError:
        pass
    return result


def _spawn_preview(path: str) -> None:
    """Render outside this Blender. The open file is not imported into."""
    old = _HOVER.get("proc")
    if old is not None and old.poll() is None:
        try:
            old.kill()
        except OSError:
            pass
    log_handle = _HOVER.get("proc_log")
    if log_handle is not None:
        try:
            log_handle.close()
        except OSError:
            pass
    source = _preview_source(path)
    out = _shot_dir(Path(source)) / "result.json"
    try:
        if out.is_file():
            out.unlink()
    except OSError:
        pass
    script = Path(__file__).with_name("scene_preview_job.py")
    prefs = bpy.context.preferences.addons[__package__].preferences
    unpacked = str(getattr(prefs, "bg3_unpacked_dir", "") or "")
    log_path = out.with_suffix(".log")
    log_handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            bpy.app.binary_path,
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            path,
            str(out),
            unpacked,
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _HOVER["proc"] = proc
    _HOVER["proc_log"] = log_handle
    _HOVER["proc_out"] = str(out)
    _HOVER["proc_path"] = path


def _start_preview(path: str) -> None:
    if not path:
        return
    cached = _stick_mem_get(path)
    if cached is not None:
        _show_pose(path, cached, "sticks cached")
        _tag()
        return
    _HOVER["status"] = "sticks…"
    _HOVER["result"] = None
    thread = _HOVER.get("thread")
    if thread is not None and thread.is_alive():
        return
    gen = int(_HOVER["gen"])

    def run() -> None:
        if int(_HOVER.get("gen") or 0) != gen:
            return
        if _HOVER.get("path") != path:
            return
        _HOVER["result"] = _stick_preview_job(path)

    _HOVER["thread"] = threading.Thread(target=run, name="gr2lab-stick-preview", daemon=True)
    _HOVER["thread"].start()
    _tag()


def _path_key(path: str) -> str:
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).replace("/", "\\").casefold()


def _armature_of(obj):
    if obj is None:
        return None
    if getattr(obj, "type", None) == "ARMATURE":
        return obj
    parent = getattr(obj, "parent", None)
    if parent is not None and getattr(parent, "type", None) == "ARMATURE":
        return parent
    for mod in getattr(obj, "modifiers", ()) or ():
        if getattr(mod, "type", "") == "ARMATURE" and getattr(mod, "object", None) is not None:
            return mod.object
    return None


def _rig_source_path(arm) -> str:
    src = str(arm.get("gr2lab_full_source_gr2") or arm.get("gr2lab_source_gr2") or "").strip()
    if src:
        return src
    anim = getattr(arm, "animation_data", None)
    action = getattr(anim, "action", None) if anim is not None else None
    if action is None:
        return ""
    return str(action.get("gr2lab_source_gr2") or "").strip()


_CLIP_LOOKUP = {"mtime": -1.0, "by_path": {}, "by_name": {}}


def _norm_clip_path(path: str) -> str:
    return str(path).replace("/", "\\").casefold()


def _ensure_clip_lookup(data: dict) -> None:
    """Map clip path → scene once. Selecting a rig must not walk the library."""
    mtime = float(_INDEX.get("mtime") or 0.0)
    if _CLIP_LOOKUP["mtime"] == mtime and _CLIP_LOOKUP["by_path"]:
        return
    by_path: Dict[str, tuple] = {}
    by_name: Dict[str, list] = {}
    for row in data.get("scenes") or []:
        folder = str(row.get("dir") or "")
        prefix = _norm_clip_path(folder)
        if prefix and not prefix.endswith("\\"):
            prefix += "\\"
        for clip in row.get("clips") or []:
            full = prefix + _norm_clip_path(clip)
            by_path[full] = (row, clip)
            by_name.setdefault(str(clip).casefold(), []).append((row, clip))
    _CLIP_LOOKUP["mtime"] = mtime
    _CLIP_LOOKUP["by_path"] = by_path
    _CLIP_LOOKUP["by_name"] = by_name


def _indexed_clip(data: dict, source: str):
    _ensure_clip_lookup(data)
    want = _norm_clip_path(source)
    hit = _CLIP_LOOKUP["by_path"].get(want)
    if hit is not None:
        return hit
    try:
        resolved = _norm_clip_path(str(Path(source).resolve()))
    except OSError:
        resolved = ""
    if resolved and resolved != want:
        hit = _CLIP_LOOKUP["by_path"].get(resolved)
        if hit is not None:
            return hit
    hits = _CLIP_LOOKUP["by_name"].get(Path(source).name.casefold()) or []
    if len(hits) == 1:
        return hits[0]
    return None, None


def _show_indexed_clip(scene: dict, clip: str) -> None:
    tags = set(scene.get("tags") or [])
    act = str(scene.get("act") or "")
    if _STATE.get("act") not in ("All", act):
        _STATE["act"] = act or "All"
        _STATE["filtered_key"] = ""
    if _STATE.get("camp") and "CAMP" not in tags:
        _STATE["camp"] = False
        _STATE["filtered_key"] = ""
    if _STATE.get("rom") and "ROM" not in tags:
        _STATE["rom"] = False
        _STATE["filtered_key"] = ""
    tokens = _search_tokens(_STATE.get("query") or "")
    scene_name = scene.get("name") or ""
    if tokens and (
        not _scene_has_tokens(scene, tokens)
        or not _text_has_tokens(f"{scene_name} {clip}", tokens)
    ):
        _STATE["query"] = ""
        _STATE["search_caret"] = 0
        _STATE["search_sel"] = False
        _STATE["filtered_key"] = ""
    data = _INDEX.get("data") or {}
    rows = _filtered(data)
    ids = [row.get("id") for row in rows]
    try:
        scene_index = ids.index(scene.get("id"))
    except ValueError:
        scene_index = 0
    _STATE["scene_id"] = scene.get("id") or ""
    _STATE["scene_scroll"] = max(0, scene_index - 3)
    clips = list(scene.get("clips") or [])
    tokens = _search_tokens(_STATE.get("query") or "")
    ordered = sorted(clips, key=lambda name: (_role(name), name.lower()))
    if tokens:
        ordered = [
            name
            for name in ordered
            if _text_has_tokens(f"{scene_name} {name}", tokens)
        ]
    clip_index = 0
    want = clip.casefold()
    for i, name in enumerate(ordered):
        if name.casefold() == want:
            clip_index = i
            clip = name
            break
    _STATE["clip_scroll"] = max(0, clip_index - 2)
    full = str(Path(scene.get("dir") or "") / clip)
    _STATE["clip_path"] = full
    if _HOVER.get("path") != full:
        _HOVER["path"] = full
        _HOVER["since"] = time.time()
        _HOVER["gen"] = int(_HOVER.get("gen") or 0) + 1
        _HOVER["status"] = ""
    _tag()


def _clip_is_selected(filepath: str) -> bool:
    path = str(filepath or "").strip()
    if not path:
        return False
    return _path_key(path) == _path_key(str(_STATE.get("clip_path") or ""))


def _select_strip_clip(context, filepath: str) -> None:
    """Highlight a clip in the strip (no import)."""
    path = str(filepath or "").strip()
    if not path:
        return
    data = _load_index()
    if data:
        scene, clip_name = _indexed_clip(data, path)
        if scene is not None and clip_name:
            _show_indexed_clip(scene, clip_name)
            return
    _STATE["clip_path"] = path
    if _HOVER.get("path") != path:
        _HOVER["path"] = path
        _HOVER["since"] = time.time()
        _HOVER["gen"] = int(_HOVER.get("gen") or 0) + 1
        _HOVER["status"] = ""
    _tag()


def _sync_browser_to_rig() -> None:
    if _DRAG.get("active") or _STATE.get("search_focus"):
        return
    try:
        ctx = bpy.context
        if not _browser_on(ctx):
            return
        arm = _armature_of(getattr(ctx.view_layer.objects, "active", None))
    except Exception:
        return
    if arm is None:
        _STATE["rig_sync"] = ""
        return
    src = _rig_source_path(arm)
    token = f"{arm.as_pointer()}|{src}"
    if token == _STATE.get("rig_sync"):
        return
    _STATE["rig_sync"] = token
    if not src:
        return
    data = _load_index()
    if not data:
        return
    scene, clip = _indexed_clip(data, src)
    if scene is None or not clip:
        _STATE["clip_path"] = ""
        _tag()
        return
    _show_indexed_clip(scene, clip)


def _tick() -> Optional[float]:
    if not _ALIVE:
        return None
    _sync_browser_to_rig()
    changed = False
    result = _HOVER.get("result")
    if result:
        _HOVER["result"] = None
        path, data, err = result
        if err or not data:
            if _HOVER["path"] == path:
                _HOVER["shown"] = path
                _HOVER["poses"] = None
                _HOVER["status"] = err or "preview failed"
                changed = True
        else:
            _remember_pose(path, data)
            if _HOVER["path"] == path:
                kind = data.get("kind")
                shown_kind = (_HOVER.get("poses") or {}).get("kind")
                if kind == "sticks" and shown_kind == "scene":
                    pass
                else:
                    if kind == "scene":
                        label = "mesh cached" if data.get("cached") else f"mesh {data.get('seconds', 0)}s"
                    elif kind == "sticks":
                        label = f"sticks {data.get('seconds', 0)}s"
                    else:
                        label = f"{data.get('seconds', 0)}s"
                    _show_pose(path, data, label)
                    if kind == "scene":
                        _HOVER["textures"] = {}
            changed = True
    proc = _HOVER.get("proc")
    if proc is not None:
        current = _HOVER.get("path") or ""
        wanted = _HOVER.get("proc_path") or ""
        if current and _preview_source(current) != _preview_source(wanted):
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            _HOVER["proc"] = None
            thread = _HOVER.get("thread")
            if thread is not None and thread.is_alive():
                _HOVER["gen"] = int(_HOVER.get("gen") or 0) + 1
        elif proc.poll() is None:
            return 0.3
        else:
            _HOVER["proc"] = None
            log_handle = _HOVER.get("proc_log")
            if log_handle is not None:
                try:
                    log_handle.close()
                except OSError:
                    pass
                _HOVER["proc_log"] = None
            out = Path(str(_HOVER.get("proc_out") or ""))
            err = ""
            data = None
            if proc.returncode == 0 and out.is_file():
                try:
                    data = json.loads(out.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    err = str(exc)
            else:
                err = "preview failed"
            _HOVER["result"] = (wanted, data, err)
            changed = True
    path = _HOVER["path"]
    if path and _HOVER["shown"] != path and not _HOVER.get("result"):
        if (time.time() - float(_HOVER["since"])) >= STICK_HOVER_DEBOUNCE:
            cached = _stick_mem_get(path)
            if cached is not None:
                _show_pose(path, cached, "sticks cached")
                changed = True
            else:
                thread = _HOVER.get("thread")
                if thread is None or not thread.is_alive():
                    disk = _stick_load_disk(path)
                    if disk is not None:
                        _show_pose(path, disk, "sticks cached")
                        changed = True
                    else:
                        _start_preview(path)
                        changed = True
    poses = _HOVER.get("poses") or {}
    frames = poses.get("shots") or poses.get("poses") or []
    duration_sec = float(poses.get("duration_sec") or 0.0)
    play_fps = float(poses.get("play_fps") or 0.0)
    if path and path == _HOVER["shown"] and poses.get("kind") == "sticks" and duration_sec > 0:
        changed = True
    elif path and path == _HOVER["shown"] and len(frames) > 1 and duration_sec > 0:
        elapsed = time.time() - float(_HOVER["pose_tick"])
        index = int((elapsed % duration_sec) / duration_sec * len(frames)) % len(frames)
        if index != int(_HOVER["pose_i"]):
            _HOVER["pose_i"] = index
            changed = True
        _preload_play(poses, index)
        changed = True
    elif path and path == _HOVER["shown"] and len(frames) > 1:
        if time.time() - float(_HOVER["pose_tick"]) >= 1.0:
            _HOVER["pose_i"] = (int(_HOVER["pose_i"]) + 1) % len(frames)
            _HOVER["pose_tick"] = time.time()
            changed = True
    refilter_at = float(_STATE.get("search_refilter_at") or 0.0)
    if refilter_at and time.perf_counter() >= refilter_at:
        _STATE["search_refilter_at"] = 0.0
        _STATE["filtered_key"] = ""
        _STATE["scene_scroll"] = 0
        _STATE["clip_scroll"] = 0
        changed = True
    if _STATE.get("search_focus"):
        phase = int(time.time() * 2) % 2
        if phase != int(_STATE.get("search_blink_phase", -1)):
            _STATE["search_blink_phase"] = phase
            changed = True
    if changed or _PREP["running"]:
        _tag()
    _flush_tag_pending()
    if _STATE.get("search_focus"):
        return 0.08
    if duration_sec > 0 and path and path == _HOVER["shown"]:
        if poses.get("kind") == "sticks":
            return 0.016
        return 0.033
    return 0.1


def _ensure_timer() -> None:
    global _TIMER_ON
    if _TIMER_ON:
        return
    _TIMER_ON = True
    bpy.app.timers.register(_tick, first_interval=0.1, persistent=True)


def view3d_keyconfig(wm) -> Optional[object]:
    """Addon keymaps; fall back to user/active when addon config is missing."""
    if wm is None:
        return None
    kc = wm.keyconfigs.addon
    if kc is not None:
        return kc
    kc = wm.keyconfigs.user
    if kc is not None:
        return kc
    return wm.keyconfigs.active


def _browser_on(context) -> bool:
    try:
        addon = context.preferences.addons.get("gr2lab_blender")
        prefs = addon.preferences if addon is not None else None
        return bool(getattr(prefs, "anim_browser", True))
    except Exception:
        return False


def _browser_whole_scene(context=None) -> bool:
    """Whole scene in the strip: partner rig, stage empties, timeline camera (import dialog)."""
    return bool(_STATE.get("whole_scene", True))


def _browser_center_parent(context=None) -> bool:
    """Centralize scene (top parent at origin) — same as import dialog toggle."""
    return bool(_STATE.get("center_parent"))


def edit_copy_enabled() -> bool:
    return bool(_STATE.get("edit_copy"))


def _sync_browser_center_parent(context, value: bool) -> None:
    _STATE["center_parent"] = bool(value)
    if context is None:
        return
    try:
        prefs = context.preferences.addons[__package__].preferences
        prefs.center_parent = bool(value)
    except Exception:
        pass


def _active_browser_clip_path() -> str:
    path = str(_STATE.get("clip_path") or "").strip()
    if path:
        return path
    hover = str(_HOVER.get("path") or "").strip()
    if hover:
        return hover
    return ""


def _cursor_drop_point(context):
    return Vector(context.scene.cursor.location)


def _gr2lab_armatures():
    if bpy is None:
        return []
    return [
        obj
        for obj in bpy.data.objects
        if getattr(obj, "type", "") == "ARMATURE" and obj.get("gr2lab_run_id")
    ]


def _resolve_browser_edit_arm(context, clip_path: str = ""):
    """Rig to duplicate: active GR2 LAB arm, else one imported from this clip path."""
    if bpy is None:
        return None
    active = _armature_of(getattr(context, "active_object", None))
    if active is not None and active.get("gr2lab_run_id"):
        return active
    path = str(clip_path or "").strip()
    if path:
        key = _path_key(path)
        for arm in _gr2lab_armatures():
            src = _rig_source_path(arm)
            if src and _path_key(src) == key:
                return arm
    arms = _gr2lab_armatures()
    if len(arms) == 1:
        return arms[0]
    return None


def schedule_duplicate_edit_rig(context, arm) -> None:
    """Duplicate aside on the next timer tick (keeps UI out of import modal)."""
    if bpy is None or context is None or arm is None:
        return
    arm_name = str(arm.name)

    def _run():
        try:
            obj = bpy.data.objects.get(arm_name)
            if obj is None or getattr(obj, "type", "") != "ARMATURE":
                return None
            print(
                "[GR2 LAB] duplicate for edit: working (long clips can take a few minutes)…",
                flush=True,
            )
            duplicate_edit_rig_now(bpy.context, obj)
        except Exception:
            traceback.print_exc()
        return None

    bpy.app.timers.register(_run, first_interval=0.05)


def finish_deferred_edit_copy(context, import_state) -> None:
    """Run edit duplicate after import modal ends (ops do not run inside import modal)."""
    if bpy is None or context is None or not import_state:
        return
    if not import_state.pop("defer_edit_copy", False):
        return
    arm = import_state.get("arm")
    if arm is None:
        name = str(import_state.get("arm_name") or "").strip()
        if name:
            arm = bpy.data.objects.get(name)
    if arm is None or getattr(arm, "type", "") != "ARMATURE":
        print("[GR2 LAB] edit copy: no armature after import", flush=True)
        return
    try:
        from .import_progress_ui import update as overlay_update

        overlay_update(0.99, "Edit copy…", detail="duplicate rig + NLA")
    except Exception:
        pass
    duplicate_edit_rig_now(context, arm)


def duplicate_edit_rig_now(context, arm):
    """Copy rig + meshes aside with one full NLA strip + copied Action."""
    if bpy is None or context is None or arm is None:
        return None
    from . import scene_stage_empties

    dup = scene_stage_empties.duplicate_gr2lab_edit_rig(
        context,
        arm,
        offset=(3.0, 0.0, 0.0),
        scene_start=1,
        duplicate_action=True,
    )
    if dup is not None:
        try:
            bpy.ops.object.select_all(action="DESELECT")
            dup.select_set(True)
            context.view_layer.objects.active = dup
        except Exception:
            pass
        print(f"[GR2 LAB] duplicate for edit: {dup.name}", flush=True)
    return dup


def browser_duplicate_edit_rig(context, clip_path: str = "") -> Tuple[bool, str]:
    """Duplicate an already-imported rig (does not re-import the clip)."""
    arm = _resolve_browser_edit_arm(context, clip_path)
    if arm is None:
        return (
            False,
            "Drop or import the clip first, select its armature, then Duplicate for edit.",
        )
    schedule_duplicate_edit_rig(context, arm)
    return True, f"Duplicating {arm.name} for edit…"


def _place_strip_clip_at_cursor(context, filepath: str) -> None:
    """Drop the selected strip clip at the 3D cursor (second click / click without drag)."""
    path = str(filepath or "").strip()
    if not path:
        return
    point = _cursor_drop_point(context)
    _import_clip_at(
        context,
        path,
        point,
        place=True,
        whole_scene=_browser_whole_scene(context),
    )


def _strip_import_kwargs(context, filepath: str) -> dict:
    """Browser drag / preload uses the same NEW_RIG + Whole scene flags as a quick drop."""
    from .bridge_blender import gr2_import_exec_kwargs

    path = str(filepath or "").strip()
    whole = _browser_whole_scene(context)
    kwargs = gr2_import_exec_kwargs(context, path)
    kwargs["import_apply"] = "NEW_RIG"
    kwargs["target_armature"] = ""
    kwargs["add_rig"] = True
    kwargs["second_character"] = whole
    kwargs["scene_empties"] = whole
    kwargs["center_parent"] = _browser_center_parent(context) if whole else False
    return kwargs


def _importing() -> bool:
    try:
        from .import_progress_ui import _STATE as import_state

        return bool(import_state.get("active"))
    except Exception:
        return False


def _step_scene(delta: int) -> None:
    rows = _STATE["filtered"]
    if not rows:
        return
    ids = [row["id"] for row in rows]
    current = _STATE["scene_id"]
    try:
        index = ids.index(current)
    except ValueError:
        index = 0
    index = (index + delta) % len(ids)
    _STATE["scene_id"] = ids[index]
    _STATE["clip_scroll"] = 0
    visible = 7
    if index < _STATE["scene_scroll"]:
        _STATE["scene_scroll"] = index
    elif index >= _STATE["scene_scroll"] + visible:
        _STATE["scene_scroll"] = index - visible + 1


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    dist_left = abs(estimate - left)
    dist_up = abs(estimate - up)
    dist_up_left = abs(estimate - up_left)
    if dist_left <= dist_up and dist_left <= dist_up_left:
        return left
    if dist_up <= dist_up_left:
        return up
    return up_left


def _decode_png(path: str) -> Tuple[int, int, bytes]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("png")
    pos = 8
    width = height = color = 0
    idat = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit, color, _comp, _filt, inter = struct.unpack(">IIBBBBB", chunk)
            if bit != 8 or inter != 0 or color not in (2, 6):
                raise ValueError("png format")
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    raw = zlib.decompress(b"".join(idat))
    channels = 4 if color == 6 else 3
    stride = width * channels
    rows = []
    cursor = 0
    previous = bytearray(stride)
    for _y in range(height):
        filter_type = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        if filter_type == 1:
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                row[x] = (row[x] + left) & 255
        elif filter_type == 2:
            for x in range(stride):
                row[x] = (row[x] + previous[x]) & 255
        elif filter_type == 3:
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                row[x] = (row[x] + ((left + previous[x]) // 2)) & 255
        elif filter_type == 4:
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                up = previous[x]
                up_left = previous[x - channels] if x >= channels else 0
                row[x] = (row[x] + _paeth(left, up, up_left)) & 255
        elif filter_type != 0:
            raise ValueError("png filter")
        previous = row
        if channels == 3:
            rgba = bytearray(width * 4)
            for x in range(width):
                rgba[x * 4 : x * 4 + 3] = row[x * 3 : x * 3 + 3]
                rgba[x * 4 + 3] = 255
            row = rgba
        rows.append(bytes(row))
    rows.reverse()
    return width, height, b"".join(rows)


def _texture_from_png(path: str):
    width, height, pixels = _decode_png(path)
    import numpy as np

    values = np.frombuffer(pixels, dtype=np.uint8).astype(np.float32) * (1.0 / 255.0)
    buffer = gpu.types.Buffer("FLOAT", values.size, values)
    return gpu.types.GPUTexture((width, height), format="RGBA8", data=buffer)


def _shot_texture(path: str):
    cache = _HOVER.setdefault("textures", {})
    tex = cache.get(path)
    if tex is not None:
        return tex
    tex = _texture_from_png(path)
    cache[path] = tex
    return tex


def _preload_play(data: dict, index: int) -> None:
    shots = data.get("shots") or []
    if not shots:
        return
    cache = _HOVER.setdefault("textures", {})
    loaded = 0
    count = len(shots)
    for offset in range(10):
        path = shots[(int(index) + offset) % count]
        if path in cache:
            continue
        try:
            cache[path] = _texture_from_png(path)
        except Exception:
            cache[path] = None
        loaded += 1
        if loaded >= 2:
            break


def _stick_clock(data: dict) -> float:
    duration = float(data.get("duration_sec") or data.get("duration") or 0.0)
    if duration <= 0:
        return 0.0
    elapsed = time.perf_counter() - float(_HOVER["pose_tick"])
    return elapsed % duration


def _stick_pose_times(data: dict, poses: list) -> List[float]:
    duration = float(data.get("duration_sec") or data.get("duration") or 0.0)
    times: List[float] = []
    for pose in poses:
        if pose.get("t") is not None:
            times.append(float(pose["t"]))
        else:
            times.append(float(pose.get("frac", 0.0)) * duration)
    return times


def _stick_points_for_pose(data: dict, pose: dict) -> Dict[str, List[float]]:
    bone_order = data.get("bone_order") or []
    parents = data.get("parents") or {}
    if pose.get("rig") and bone_order:
        from .local_pipeline import stick_blend_poses

        return _stick_apply_partner_shift(
            stick_blend_poses(bone_order, parents, pose, pose, 0.0)
        )
    return _stick_apply_partner_shift(dict(pose.get("points") or {}))


def _stick_apply_partner_shift(points: Dict[str, List[float]]) -> Dict[str, List[float]]:
    shift = float(STICK_PARTNER_SHIFT)
    out: Dict[str, List[float]] = {}
    for name, pt in points.items():
        if name.startswith("P:"):
            out[name] = [float(pt[0]) + shift, float(pt[1]), float(pt[2])]
        else:
            out[name] = [float(pt[0]), float(pt[1]), float(pt[2])]
    return out


def _stick_interp_points(data: dict, clock: float) -> Dict[str, List[float]]:
    poses = data.get("poses") or []
    if not poses:
        return {}
    if len(poses) == 1:
        return _stick_points_for_pose(data, poses[0])
    times = data.get("_pose_times")
    if not times or len(times) != len(poses):
        times = _stick_pose_times(data, poses)
    duration = float(data.get("duration_sec") or times[-1] or 0.0)
    if clock <= times[0]:
        return _stick_points_for_pose(data, poses[0])
    if clock >= times[-1] or duration <= 0:
        return _stick_points_for_pose(data, poses[-1])
    index = 0
    lo, hi = 0, len(times) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if clock < times[mid]:
            hi = mid - 1
        elif clock > times[mid + 1]:
            lo = mid + 1
        else:
            index = mid
            break
    else:
        index = max(0, min(len(times) - 2, lo))
    start = times[index]
    end = times[index + 1]
    span = end - start
    alpha = 0.0 if span <= 1e-8 else (clock - start) / span
    left_pose = poses[index]
    right_pose = poses[index + 1]
    bone_order = data.get("bone_order") or []
    parents = data.get("parents") or {}
    if bone_order and left_pose.get("rig") and right_pose.get("rig"):
        from .local_pipeline import stick_blend_poses

        fk = stick_blend_poses(bone_order, parents, left_pose, right_pose, alpha)
        if fk:
            return _stick_apply_partner_shift(fk)
    left = left_pose.get("points") or {}
    right = right_pose.get("points") or {}
    blended: Dict[str, List[float]] = {}
    for name in set(left) | set(right):
        a = left.get(name)
        b = right.get(name)
        if a is None and b is None:
            continue
        if a is None:
            blended[name] = [float(b[0]), float(b[1]), float(b[2])]
        elif b is None:
            blended[name] = [float(a[0]), float(a[1]), float(a[2])]
        else:
            blended[name] = [
                float(a[0]) + (float(b[0]) - float(a[0])) * alpha,
                float(a[1]) + (float(b[1]) - float(a[1])) * alpha,
                float(a[2]) + (float(b[2]) - float(a[2])) * alpha,
            ]
    return _stick_apply_partner_shift(blended)


def _draw_shot(rect, path: str) -> None:
    if not rect or not path:
        return
    cache = _HOVER.setdefault("textures", {})
    tex = cache.get(path)
    if path not in cache:
        try:
            tex = _texture_from_png(path)
        except Exception:
            tex = None
        cache[path] = tex
    if tex is None:
        return
    x, y, w, h = _fit_aspect_rect(rect)
    shader = gpu.shader.from_builtin("IMAGE")
    batch = batch_for_shader(
        shader,
        "TRI_FAN",
        {
            "pos": ((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
            "texCoord": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        },
    )
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_sampler("image", tex)
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _draw_stick(rect, data, pose_i: int = 0, *, clock: Optional[float] = None) -> None:
    if not rect or not data:
        return
    x, y, w, h = rect
    poses = data.get("poses") or []
    if not poses:
        return
    if clock is not None:
        points = _stick_interp_points(data, clock)
    else:
        pose = poses[int(pose_i) % len(poses)]
        points = _stick_points_for_pose(data, pose)
    if not points:
        return
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    zs = [p[2] for p in points.values()]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = max(max_x - min_x, 0.05)
    span_z = max(max_z - min_z, 0.05)
    pad = 10.0
    scale = min((w - pad * 2) / span_x, (h - pad * 2) / span_z)
    cx = (min_x + max_x) * 0.5
    cz = (min_z + max_z) * 0.5

    def project(point):
        px = x + w * 0.5 + (point[0] - cx) * scale
        py = y + h * 0.45 + (point[2] - cz) * scale
        return (px, py)

    player_pairs = []
    partner_pairs = []
    for a, b in data.get("edges") or []:
        if a not in points or b not in points:
            continue
        seg = (project(points[a]), project(points[b]))
        if str(a).startswith("P:") or str(b).startswith("P:"):
            partner_pairs.extend(seg)
        else:
            player_pairs.extend(seg)
    if partner_pairs:
        _lines(player_pairs, (0.45, 0.82, 1.0, 1.0))
        _lines(partner_pairs, (0.98, 0.72, 0.42, 1.0))
    else:
        _lines(player_pairs, (0.93, 0.78, 0.42, 1.0))
    del ys


def _draw() -> None:
    context = bpy.context
    region = getattr(context, "region", None)
    if region is None or getattr(region, "type", "") != "WINDOW":
        return
    area = getattr(context, "area", None)
    if area is None or getattr(area, "type", "") != "VIEW_3D":
        return
    if not _browser_on(context):
        _HITS[region.as_pointer()] = []
        _LAYOUTS.pop(region.as_pointer(), None)
        return
    if _importing():
        _HITS[region.as_pointer()] = []
        return
    font_id = _font(12)
    width = float(region.width)
    x0 = STRIP_MARGIN_X
    y0 = STRIP_MARGIN_Y
    strip_w = max(240.0, width - STRIP_MARGIN_X * 2.0)
    strip_h = STRIP_HEIGHT
    hits = []
    root = unpacked_root(context)
    data = _load_index()
    ready = root is not None and _index_matches(context, data)
    _LAYOUTS[region.as_pointer()] = {
        "strip": (x0, y0, strip_w, strip_h),
        "scene_box": None,
        "clip_box": None,
        "preview": None,
    }
    _rect(x0, y0, strip_w, strip_h, (0.05, 0.06, 0.07, 0.92))
    _rect(x0, y0 + strip_h - 2.0, strip_w, 2.0, (0.45, 0.72, 0.95, 1.0))
    if not ready:
        message = "Set unpacked data and Prepare library."
        if root is None:
            message = "Set BG3 unpacked data, then Prepare library."
        elif _PREP["running"]:
            message = f"Preparing library… {_PREP['scenes']} scenes"
        elif _PREP["error"]:
            message = _PREP["error"]
        blf.color(font_id, 0.92, 0.94, 0.96, 1.0)
        blf.position(font_id, x0 + 10.0, y0 + 78.0, 0.0)
        blf.draw(font_id, message)
        bx, by, bw, bh = x0 + 10.0, y0 + 36.0, 160.0, 24.0
        _rect(bx, by, bw, bh, (0.20, 0.36, 0.55, 1.0))
        blf.position(font_id, bx + 8.0, by + 6.0, 0.0)
        blf.draw(font_id, "Prepare library")
        hits.append(("prepare", bx, by, bw, bh, ""))
        _HITS[region.as_pointer()] = hits
        return

    rows = _filtered(data)
    scene = _selected(rows)
    row_h = 20.0
    top = y0 + strip_h - 28.0
    chip_y = top + 4.0
    cursor_x = x0 + 8.0

    def chip(text, kind, payload, on=False, action=False):
        nonlocal cursor_x
        tw = _text_width(font_id, text) + 16.0
        if action:
            bg = (0.16, 0.28, 0.20, 1.0)
        elif on:
            bg = (0.24, 0.42, 0.62, 1.0)
        else:
            bg = (0.14, 0.16, 0.18, 1.0)
        _rect(cursor_x, chip_y, tw, 20.0, bg)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, cursor_x + 8.0, chip_y + 4.0, 0.0)
        blf.draw(font_id, text)
        hits.append((kind, cursor_x, chip_y, tw, 20.0, payload))
        cursor_x += tw + 4.0

    chip(_STATE["act"], "act", _STATE["act"], True)
    chip("Scene", "whole", "", bool(_STATE.get("whole_scene")))
    chip("Center", "center", "", bool(_STATE.get("center_parent")))
    chip("Edit", "edit", "", bool(_STATE.get("edit_copy")))
    chip("CAMP", "camp", "", _STATE["camp"])
    chip("ROM", "rom", "", _STATE["rom"])
    search_x = cursor_x
    search_h = 20.0
    count_label = f"{len(rows)} scenes"
    count_w = _text_width(font_id, count_label) + 10.0
    nav_btn_w = 28.0
    expl_label = "Explorer"
    expl_w = _text_width(font_id, expl_label) + 16.0
    right_edge = x0 + strip_w - 8.0
    count_x = right_edge - count_w
    nav_next_x = count_x - 6.0 - nav_btn_w
    nav_prev_x = nav_next_x - 4.0 - nav_btn_w
    expl_x = nav_prev_x - 6.0 - expl_w
    # Never grow the search field leftward over Scene / other chips.
    search_w = max(0.0, expl_x - search_x - 6.0)
    search_focus = bool(_STATE.get("search_focus"))
    query = _STATE.get("query") or ""
    caret = _search_caret_index()
    inner_w = max(40.0, search_w - 16.0)
    caret_px = _text_width(font_id, query[:caret]) if query else 0.0
    scroll = float(_STATE.get("search_scroll") or 0.0)
    if caret_px - scroll > inner_w - 6.0:
        scroll = max(0.0, caret_px - (inner_w - 6.0))
    if caret_px < scroll:
        scroll = caret_px
    if query and _text_width(font_id, query) <= inner_w:
        scroll = 0.0
    _STATE["search_scroll"] = scroll
    placeholder = "Search"
    show_text = query if query else placeholder
    text_color = (0.94, 0.96, 0.98, 1.0) if query else (0.55, 0.58, 0.62, 1.0)
    search_bg = (0.12, 0.14, 0.18, 1.0)
    if search_focus:
        search_bg = (0.16, 0.20, 0.28, 1.0)
    elif _POINTER.get("over_strip"):
        search_bg = (0.14, 0.16, 0.20, 1.0)
    if search_w >= 28.0:
        _rect(search_x, chip_y, search_w, search_h, search_bg)
        if search_focus or query:
            _rect(search_x, chip_y, search_w, 1.5, (0.45, 0.72, 0.95, 1.0))
        text_x = search_x + 8.0 - scroll
        if search_focus and _STATE.get("search_sel") and query:
            sel_w = min(_text_width(font_id, query), inner_w)
            _rect(search_x + 6.0, chip_y + 3.0, sel_w + 4.0, 14.0, (0.20, 0.42, 0.72, 1.0))
        visible = show_text
        draw_x = text_x
        if query:
            start = 0
            if scroll > 0.0:
                for index in range(len(query) + 1):
                    if _text_width(font_id, query[:index]) >= scroll:
                        start = max(0, index - 1)
                        break
            rest = query[start:]
            fit = 0
            for count in range(len(rest) + 1):
                if _text_width(font_id, rest[:count]) > inner_w:
                    break
                fit = count
            visible = rest[:fit]
            draw_x = search_x + 8.0 + _text_width(font_id, query[:start]) - scroll
        blf.color(font_id, *text_color[:3], text_color[3])
        blf.position(font_id, draw_x, chip_y + 4.0, 0.0)
        blf.draw(font_id, visible)
        if search_focus and int(time.time() * 2) % 2 == 0:
            caret_x = search_x + 8.0 + caret_px - scroll
            caret_x = max(search_x + 6.0, min(caret_x, search_x + search_w - 4.0))
            _rect(caret_x, chip_y + 4.0, 1.0, 12.0, (0.92, 0.94, 0.96, 1.0))
        hits.append(("search", search_x, chip_y, search_w, search_h, ""))
        _LAYOUTS[region.as_pointer()]["search_box"] = (
            search_x,
            chip_y,
            search_w,
            search_h,
        )
    def chip_at(x, text, kind, payload, on=False):
        tw = max(nav_btn_w, _text_width(font_id, text) + 16.0)
        bg = (0.24, 0.42, 0.62, 1.0) if on else (0.14, 0.16, 0.18, 1.0)
        _rect(x, chip_y, tw, 20.0, bg)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, x + (tw - _text_width(font_id, text)) * 0.5, chip_y + 4.0, 0.0)
        blf.draw(font_id, text)
        hits.append((kind, x, chip_y, tw, 20.0, payload))

    chip_at(expl_x, expl_label, "explorer", "", False)
    chip_at(nav_prev_x, "◀", "prev", "", False)
    chip_at(nav_next_x, "▶", "next", "", False)
    blf.color(font_id, 0.70, 0.74, 0.78, 1.0)
    blf.position(font_id, count_x, chip_y + 4.0, 0.0)
    blf.draw(font_id, count_label)

    body_y = y0 + 6.0
    body_h = top - body_y - 4.0
    scene_w, clip_w, prev_w = _pane_pixel_sizes(strip_w - 12.0)
    scene_x = x0 + 6.0
    clip_x = scene_x + scene_w + 4.0
    prev_x = clip_x + clip_w + 4.0
    _rect(scene_x, body_y, scene_w, body_h, (0.08, 0.09, 0.10, 1.0))
    _rect(clip_x, body_y, clip_w, body_h, (0.08, 0.09, 0.10, 1.0))
    _rect(prev_x, body_y, prev_w, body_h, (0.03, 0.04, 0.05, 1.0))
    split_scene = (clip_x - 6.0, body_y, 8.0, body_h)
    split_clip = (prev_x - 6.0, body_y, 8.0, body_h)
    _LAYOUTS[region.as_pointer()]["scene_box"] = (scene_x, body_y, scene_w, body_h)
    _LAYOUTS[region.as_pointer()]["clip_box"] = (clip_x, body_y, clip_w, body_h)
    _LAYOUTS[region.as_pointer()]["split_scene"] = split_scene
    _LAYOUTS[region.as_pointer()]["split_clip"] = split_clip
    mx = float(_POINTER.get("mx") or -1.0)
    my = float(_POINTER.get("my") or -1.0)
    _split_grip(split_scene, hot=_over_box(split_scene, mx, my))
    _split_grip(split_clip, hot=_over_box(split_clip, mx, my))
    preview_top = 4.0
    status_h = 16.0
    gap = 4.0
    preview_footer = gap + status_h + 4.0
    preview_h = max(36.0, body_h - preview_top - preview_footer)
    preview_w = min(prev_w - 8.0, preview_h * PREVIEW_ASPECT)
    preview_x = prev_x + 4.0 + (prev_w - 8.0 - preview_w) * 0.5
    preview_y = body_y + preview_top
    _LAYOUTS[region.as_pointer()]["preview"] = (preview_x, preview_y, preview_w, preview_h)
    _LAYOUTS[region.as_pointer()]["preview_status_y"] = body_y + body_h - status_h - 2.0

    visible = max(1, int(body_h // row_h))
    scroll = max(0, min(_STATE["scene_scroll"], max(0, len(rows) - visible)))
    _STATE["scene_scroll"] = scroll
    for i in range(visible):
        index = scroll + i
        if index >= len(rows):
            break
        row = rows[index]
        iy = body_y + body_h - row_h * (i + 1)
        selected = scene is not None and row["id"] == scene["id"]
        if selected:
            _rect(scene_x, iy, scene_w, row_h, (0.16, 0.28, 0.40, 1.0))
        label = f"{row.get('act') or '—'}  {row.get('name') or ''}"
        blf.color(font_id, 0.95, 0.96, 0.97, 1.0)
        blf.position(font_id, scene_x + 6.0, iy + 4.0, 0.0)
        blf.draw(font_id, _clip_text(font_id, label, scene_w - 12.0))
        hits.append(("scene", scene_x, iy, scene_w, row_h, row["id"]))

    clips = []
    dual_char = False
    if scene is not None:
        all_clips = list(scene.get("clips") or [])
        dual_char = _scene_dual_char(all_clips)
        clips = sorted(all_clips, key=lambda name: (_role(name), name.lower()))
        clip_tokens = _search_tokens(_STATE["query"])
        if clip_tokens:
            scene_name = scene.get("name") or ""
            clips = [
                name
                for name in clips
                if _text_has_tokens(f"{scene_name} {name}", clip_tokens)
            ]
    clip_scroll = max(0, min(_STATE["clip_scroll"], max(0, len(clips) - visible)))
    _STATE["clip_scroll"] = clip_scroll
    folder = scene.get("dir") if scene else ""
    for i in range(visible):
        index = clip_scroll + i
        if index >= len(clips):
            break
        name = clips[index]
        iy = body_y + body_h - row_h * (i + 1)
        full = str(Path(folder) / name) if folder else ""
        picked = bool(full) and _path_key(full) == _path_key(str(_STATE.get("clip_path") or ""))
        if picked:
            _rect(clip_x, iy, clip_w, row_h, (0.12, 0.34, 0.55, 1.0))
            _rect(clip_x, iy, 3.0, row_h, (0.45, 0.86, 1.0, 1.0))
        elif full and full == _HOVER["path"]:
            _rect(clip_x, iy, clip_w, row_h, (0.28, 0.24, 0.12, 1.0))
        role_label, role_rgb = _role_clip_colors(name, dual_char)
        tx = clip_x + 6.0
        blf.color(font_id, role_rgb[0], role_rgb[1], role_rgb[2], 1.0)
        blf.position(font_id, tx, iy + 4.0, 0.0)
        blf.draw(font_id, f"{role_label}  ")
        tx += _text_width(font_id, f"{role_label}  ")
        blf.color(font_id, 0.94, 0.95, 0.96, 1.0)
        blf.position(font_id, tx, iy + 4.0, 0.0)
        blf.draw(font_id, _clip_text(font_id, name, clip_w - 12.0 - (tx - clip_x - 6.0)))
        hits.append(("clip", clip_x, iy, clip_w, row_h, full))

    status = _HOVER["status"] or "Hover a clip"
    shown = _HOVER.get("poses") or {}
    if _HOVER["shown"] and shown.get("kind") == "scene" and shown.get("duration_sec"):
        shots = shown.get("shots") or []
        duration = float(shown["duration_sec"])
        if shots and duration > 0:
            seconds = (int(_HOVER["pose_i"]) / len(shots)) * duration
            status = f"{seconds:.1f}s / {duration:.1f}s  {status}"
    elif _HOVER["shown"] and shown.get("kind") == "sticks" and shown.get("duration_sec"):
        duration = float(shown["duration_sec"])
        if duration > 0:
            seconds = _stick_clock(shown)
            status = f"sticks {seconds:.2f}s / {duration:.1f}s  {status}"
    elif _HOVER["shown"] and shown.get("poses"):
        frames = shown.get("poses") or []
        frac = frames[int(_HOVER["pose_i"]) % len(frames)].get("frac", 0)
        status = f"{int(float(frac) * 100)}%  {status}"
    blf.color(font_id, 0.75, 0.78, 0.82, 1.0)
    status_y = float(
        (_LAYOUTS.get(region.as_pointer()) or {}).get("preview_status_y")
        or (body_y + body_h - status_h - 2.0)
    )
    blf.position(font_id, prev_x + 8.0, status_y, 0.0)
    blf.draw(font_id, _clip_text(font_id, status, prev_w - 16.0))
    if _HOVER["shown"] == _HOVER["path"] and shown:
        rect = _LAYOUTS[region.as_pointer()]["preview"]
        shots = shown.get("shots") or []
        if shown.get("kind") == "scene" and shots:
            _draw_shot(rect, shots[int(_HOVER["pose_i"]) % len(shots)])
        elif shown.get("kind") == "sticks":
            _draw_stick(rect, shown, clock=_stick_clock(shown))
        elif shown.get("poses"):
            _draw_stick(rect, shown, int(_HOVER["pose_i"]))
    chip_hits = [row for row in hits if row[0] in _TOOLBAR_CHIP_KINDS]
    other_hits = [row for row in hits if row[0] not in _TOOLBAR_CHIP_KINDS]
    hits = other_hits + chip_hits
    _HITS[region.as_pointer()] = hits


def _pane_floors(usable: float) -> Tuple[float, float, float]:
    """Scene, clip, and preview cannot collapse. Narrow strips scale the floors."""
    scene, clip, prev = 180.0, 220.0, 200.0
    need = scene + clip + prev
    if usable >= need:
        return scene, clip, prev
    scale = usable / need
    return scene * scale, clip * scale, prev * scale


def _pane_pixel_sizes(inner: float) -> Tuple[float, float, float]:
    """Scene list, clip list, preview widths inside the strip body."""
    gaps = 8.0
    usable = max(360.0, float(inner) - gaps)
    min_scene, min_clip, min_prev = _pane_floors(usable)
    scene = usable * float(_STATE.get("pane_scene") or PANE_SCENE)
    clip = usable * float(_STATE.get("pane_clip") or PANE_CLIP)
    scene = max(min_scene, min(scene, usable - min_clip - min_prev))
    clip = max(min_clip, min(clip, usable - scene - min_prev))
    prev = max(min_prev, usable - scene - clip)
    return scene, clip, prev


def _split_grip(box, *, hot: bool) -> None:
    if not box:
        return
    x, y, w, h = box
    gx = x + (w - 2.0) * 0.5
    color = (0.55, 0.74, 0.95, 0.95) if hot else (0.32, 0.36, 0.40, 0.9)
    _rect(gx, y + 4.0, 2.0, max(8.0, h - 8.0), color)


def _split_at(layout, mx, my) -> str:
    if _over_box(layout.get("split_scene"), mx, my):
        return "scene"
    if _over_box(layout.get("split_clip"), mx, my):
        return "clip"
    return ""


def _set_pane_from_mouse(which: str, mx: float, region_width: float) -> None:
    strip_w = max(240.0, float(region_width) - STRIP_MARGIN_X * 2.0)
    inner = strip_w - 12.0
    gaps = 8.0
    usable = max(360.0, inner - gaps)
    scene_x = STRIP_MARGIN_X + 6.0
    scene_w, clip_w, _prev = _pane_pixel_sizes(inner)
    if which == "scene":
        scene_w = float(mx) - scene_x
    else:
        clip_right = float(mx)
        clip_w = clip_right - (scene_x + scene_w + 4.0)
    min_scene, min_clip, min_prev = _pane_floors(usable)
    scene_w = max(min_scene, min(scene_w, usable - min_clip - min_prev))
    clip_w = max(min_clip, min(clip_w, usable - scene_w - min_prev))
    _STATE["pane_scene"] = scene_w / usable
    _STATE["pane_clip"] = clip_w / usable


def _hit(region, mx, my):
    if region is None:
        return None
    for kind, x, y, w, h, payload in reversed(_HITS.get(region.as_pointer()) or []):
        if x <= mx <= x + w and y <= my <= y + h:
            return kind, payload
    return None


def _over_box(box, mx, my) -> bool:
    if not box:
        return False
    x, y, w, h = box
    return x <= mx <= x + w and y <= my <= y + h


def strip_box_for_region(region) -> Tuple[float, float, float, float]:
    width = float(getattr(region, "width", 0) or 0)
    strip_w = max(240.0, width - STRIP_MARGIN_X * 2.0)
    return (STRIP_MARGIN_X, STRIP_MARGIN_Y, strip_w, STRIP_HEIGHT)


def refresh_strip_pick_data(context) -> None:
    """Rebuild hit boxes for the active 3D view (same pass as GPU draw)."""
    if not _browser_on(context) or _importing():
        return
    area = getattr(context, "area", None)
    region = getattr(context, "region", None)
    if area is None or getattr(area, "type", "") != "VIEW_3D":
        return
    if region is None or getattr(region, "type", "") != "WINDOW":
        return
    try:
        _draw()
    except Exception:
        traceback.print_exc()


def pointer_over_strip(region, mx, my) -> bool:
    if region is None or not _browser_on(bpy.context):
        return False
    box = (_LAYOUTS.get(region.as_pointer()) or {}).get("strip")
    if not box:
        return False
    return _over_box(box, float(mx), float(my))


_FORCE_XY: Optional[Tuple[float, float]] = None
_CLICK_STAMP = {"t": 0.0, "xy": (0, 0, "")}


def _event_xy(event) -> Tuple[float, float]:
    if _FORCE_XY is not None:
        return _FORCE_XY
    return float(event.mouse_region_x), float(event.mouse_region_y)


def _region_contains(region, wx: float, wy: float) -> bool:
    w = float(getattr(region, "width", 0) or 0)
    h = float(getattr(region, "height", 0) or 0)
    if w <= 1.0 or h <= 1.0:
        return False
    x = float(region.x)
    y = float(region.y)
    return x <= wx < x + w and y <= wy < y + h


def view3d_under_pointer(context, event):
    """3D View window under the cursor, using window coordinates (not context.region).

    Blender's WINDOW region rectangle also covers the N-panel, the toolbar,
    and the headers. Those regions are drawn on top. A click there belongs
    to that panel, not to the animation strip.
    """
    window = getattr(context, "window", None)
    screen = getattr(window, "screen", None) if window is not None else None
    if screen is None:
        screen = getattr(context, "screen", None)
    if screen is None:
        return None
    wx = float(getattr(event, "mouse_x", 0.0))
    wy = float(getattr(event, "mouse_y", 0.0))
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        window_region = None
        covered = False
        for region in area.regions:
            if not _region_contains(region, wx, wy):
                continue
            if region.type == "WINDOW":
                window_region = region
            else:
                covered = True
        if covered or window_region is None:
            continue
        x = float(window_region.x)
        y = float(window_region.y)
        return area, window_region, wx - x, wy - y
    return None


def _fresh_click(event) -> bool:
    now = time.perf_counter()
    xy = (
        int(getattr(event, "mouse_x", 0)),
        int(getattr(event, "mouse_y", 0)),
        str(getattr(event, "value", "")),
    )
    if now - float(_CLICK_STAMP["t"]) < 0.08 and xy == _CLICK_STAMP["xy"]:
        return False
    _CLICK_STAMP["t"] = now
    _CLICK_STAMP["xy"] = xy
    return True


def modal_strip_event(context, event) -> bool:
    """Handle hover / wheel / click on the strip. True when the event was used."""
    global _FORCE_XY
    if not _browser_on(context):
        return False
    found = view3d_under_pointer(context, event)
    if found is None:
        return False
    area, region, mx, my = found
    if clip_drag_active() and event.type in {"MOUSEMOVE", "LEFTMOUSE"}:
        return False
    over = pointer_over_strip(region, mx, my)
    if event.type == "MOUSEMOVE":
        if not over:
            return False
        window = getattr(context, "window", None)
        try:
            with context.temp_override(window=window, area=area, region=region):
                _FORCE_XY = (mx, my)
                _on_move(bpy.context, event)
        except Exception:
            traceback.print_exc()
        finally:
            _FORCE_XY = None
        return True
    if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"} and over:
        window = getattr(context, "window", None)
        try:
            with context.temp_override(window=window, area=area, region=region):
                _FORCE_XY = (mx, my)
                handled = _on_wheel(bpy.context, event) == {"FINISHED"}
        except Exception:
            traceback.print_exc()
            handled = False
        finally:
            _FORCE_XY = None
        return handled
    if event.type == "LEFTMOUSE" and event.value == "PRESS" and over:
        if not _fresh_click(event):
            return True
        window = getattr(context, "window", None)
        try:
            with context.temp_override(window=window, area=area, region=region):
                _FORCE_XY = (mx, my)
                refresh_strip_pick_data(bpy.context)
                handle_strip_click(bpy.context, event, mx, my)
        except Exception:
            traceback.print_exc()
        finally:
            _FORCE_XY = None
        return True
    return False


def clip_drag_active() -> bool:
    return bool(_DRAG.get("active"))


def _drop_plane_basis(normal: Vector) -> Tuple[Vector, Vector, Vector]:
    n = normal.normalized() if normal.length_squared > 1e-12 else Vector((0.0, 0.0, 1.0))
    ref = Vector((0.0, 1.0, 0.0))
    if abs(n.dot(ref)) > 0.95:
        ref = Vector((0.0, 0.0, 1.0))
    u = ref.cross(n)
    if u.length_squared < 1e-12:
        u = Vector((1.0, 0.0, 0.0))
    else:
        u.normalize()
    v = n.cross(u).normalized()
    return u, v, n


def _drop_view_ray(context, mx: float, my: float):
    region = getattr(context, "region", None)
    rv3d = getattr(context, "region_data", None)
    if region is None or rv3d is None:
        return None, None
    coord = (float(mx), float(my))
    try:
        origin = region_2d_to_origin_3d(region, rv3d, coord)
        direction = region_2d_to_vector_3d(region, rv3d, coord)
    except Exception:
        return None, None
    if direction.length_squared < 1e-12:
        return None, None
    direction.normalize()
    return origin, direction


def _drop_on_plane(origin, direction, plane_co: Vector, plane_no: Vector) -> Optional[Vector]:
    if origin is None or direction is None:
        return None
    hit = intersect_line_plane(origin, origin + direction, plane_co, plane_no)
    if hit is None:
        return None
    return Vector(hit)


def _drop_plane_hit(context, mx: float, my: float) -> Tuple[Optional[Vector], Vector]:
    """World point under the cursor (mesh raycast, else ground plane Z=0)."""
    region = getattr(context, "region", None)
    rv3d = getattr(context, "region_data", None)
    if region is None or rv3d is None:
        return None, Vector((0.0, 0.0, 1.0))
    coord = (float(mx), float(my))
    try:
        ray_origin = region_2d_to_origin_3d(region, rv3d, coord)
        ray_dir = region_2d_to_vector_3d(region, rv3d, coord)
    except Exception:
        return None, Vector((0.0, 0.0, 1.0))
    if ray_dir.length_squared < 1e-12:
        return None, Vector((0.0, 0.0, 1.0))
    ray_dir.normalize()
    try:
        depsgraph = context.evaluated_depsgraph_get()
        hit, location, normal, _index, _obj, _matrix = context.scene.ray_cast(
            depsgraph, ray_origin, ray_dir, distance=1.0e6
        )
    except Exception:
        hit = False
        location = None
        normal = Vector((0.0, 0.0, 1.0))
    if hit and location is not None:
        if normal.length_squared < 1e-12:
            normal = Vector((0.0, 0.0, 1.0))
        return location.copy(), normal
    plane_loc = intersect_line_plane(
        ray_origin,
        ray_origin + ray_dir,
        Vector((0.0, 0.0, 0.0)),
        Vector((0.0, 0.0, 1.0)),
    )
    if plane_loc is not None:
        return plane_loc, Vector((0.0, 0.0, 1.0))
    return None, Vector((0.0, 0.0, 1.0))


def _gpu_viewport_size() -> Tuple[float, float]:
    region = getattr(bpy.context, "region", None)
    if region is not None:
        return (float(region.width), float(region.height))
    return (
        float(_DRAG.get("region_w") or 1920.0),
        float(_DRAG.get("region_h") or 1080.0),
    )


def _drop_path_key(filepath: str) -> str:
    try:
        return Path(filepath).resolve().as_posix().casefold()
    except OSError:
        return str(filepath or "").casefold()


def _pending_clip_drop_raw(context=None) -> Optional[dict]:
    """Read pending drop from WM (Blender stores this as IDPropertyGroup, not dict)."""
    ctx = context or bpy.context
    try:
        wm = ctx.window_manager
        raw = wm.get(_WM_CLIP_DROP_KEY)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        path = raw.get("path")
        point = raw.get("point")
    except AttributeError:
        return None
    if not path or point is None:
        return None
    return {
        "path": str(path),
        "point": point,
        "whole": bool(raw.get("whole")),
    }


def _pending_clip_drop_point(raw: dict) -> Optional[Vector]:
    pt = raw.get("point")
    if pt is None:
        return None
    if isinstance(pt, Vector):
        return pt.copy()
    try:
        return Vector((float(pt[0]), float(pt[1]), float(pt[2])))
    except (TypeError, ValueError, IndexError):
        return None


def register_pending_clip_drop(
    filepath: str, point: Vector, context=None, *, whole_scene: bool = False
) -> None:
    payload = {
        "path": _drop_path_key(filepath),
        "point": (float(point.x), float(point.y), float(point.z)),
        "whole": bool(whole_scene),
    }
    try:
        (context or bpy.context).window_manager[_WM_CLIP_DROP_KEY] = payload
    except Exception:
        pass


def clear_pending_clip_drop(context=None) -> None:
    try:
        wm = (context or bpy.context).window_manager
        if _WM_CLIP_DROP_KEY in wm:
            del wm[_WM_CLIP_DROP_KEY]
    except Exception:
        pass


def pending_clip_drop_blocks_scene_place(filepath: str, context=None) -> bool:
    pending = _pending_clip_drop_raw(context)
    if not pending or pending.get("whole"):
        return False
    return pending.get("path") == _drop_path_key(filepath)


def apply_pending_clip_drop(
    context, filepath: str, arm, *, consume: bool = True
) -> bool:
    pending = _pending_clip_drop_raw(context)
    if pending is None or arm is None:
        return False
    if pending.get("path") != _drop_path_key(filepath):
        return False
    point = _pending_clip_drop_point(pending)
    if point is None:
        if consume:
            clear_pending_clip_drop(context)
        return False
    if pending.get("whole"):
        if not _move_stage_to_drop(context, arm, point):
            _snap_armature_to_drop(arm, point)
    else:
        _snap_armature_to_drop(arm, point)
    _select_armature(context, arm)
    if consume:
        clear_pending_clip_drop(context)
        anchor = _armature_anchor_world(arm)
        print(
            f"[GR2 LAB] strip drop placed {arm.name} anchor "
            f"({anchor.x:.3f}, {anchor.y:.3f}, {anchor.z:.3f})",
            flush=True,
        )
    return True


def _circle_on_plane(
    center: Vector, u: Vector, v: Vector, radius: float, segments: int
) -> List[Vector]:
    pts: List[Vector] = []
    for i in range(segments + 1):
        t = (i / float(segments)) * (2.0 * math.pi)
        pts.append(center + u * (math.cos(t) * radius) + v * (math.sin(t) * radius))
    return pts


def _drop_plane_point(
    center: Vector, u: Vector, v: Vector, n: Vector, u_off: float, v_off: float
) -> Vector:
    return center + n * CLIP_DROP_LIFT + u * u_off + v * v_off


def _drop_ripple_strips(
    center: Vector,
    u: Vector,
    v: Vector,
    n: Vector,
    now: float,
    *,
    ring_count: Optional[int] = None,
    segments: Optional[int] = None,
) -> List[Tuple[List[Vector], float, float]]:
    """Expanding circular line rings."""
    base = center + n * CLIP_DROP_LIFT
    out: List[Tuple[List[Vector], float, float]] = []
    rings = int(ring_count if ring_count is not None else CLIP_DROP_RIPPLE_COUNT)
    segs = int(segments if segments is not None else CLIP_DROP_RIPPLE_SEGMENTS)
    for i in range(rings):
        phase = (now * CLIP_DROP_RIPPLE_SPEED + i / float(max(1, rings))) % 1.0
        radius = 0.12 + phase * CLIP_DROP_RIPPLE_MAX_R
        alpha = (1.0 - phase) ** 2.1 * CLIP_DROP_RIPPLE_ALPHA_PEAK
        if alpha < 0.02:
            continue
        ring = _circle_on_plane(base, u, v, radius, segs)
        width = CLIP_DROP_LINE_PX * (1.05 - phase * 0.25)
        out.append((ring, alpha, width))
    return out


def _drop_bracket_strips(
    center: Vector, u: Vector, v: Vector, n: Vector
) -> List[List[Vector]]:
    hu = CLIP_DROP_BRACKET_HALF_U
    hv = CLIP_DROP_BRACKET_HALF_V
    arm = CLIP_DROP_BRACKET_ARM
    corners = (
        (-hu, -hv, 1, 1),
        (hu, -hv, -1, 1),
        (hu, hv, -1, -1),
        (-hu, hv, 1, -1),
    )
    strips: List[List[Vector]] = []
    for u0, v0, du, dv in corners:
        p = _drop_plane_point(center, u, v, n, u0, v0)
        strips.append([p, p + u * (du * arm)])
        strips.append([p, p + v * (dv * arm)])
    return strips


def _drop_stake_strip(
    center: Vector, u: Vector, v: Vector, n: Vector
) -> List[Vector]:
    foot = _drop_plane_point(center, u, v, n, 0.0, 0.0)
    return [foot, foot + n * CLIP_DROP_STAKE_HEIGHT]


def _draw_polyline_strips_3d(
    strips: List[List[Vector]],
    color: Tuple[float, float, float, float],
    width_px: float,
) -> None:
    if not strips:
        return
    viewport = _gpu_viewport_size()
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    try:
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
    except Exception:
        pass
    shader.bind()
    shader.uniform_float("viewportSize", viewport)
    shader.uniform_float("lineWidth", float(width_px))
    shader.uniform_float("color", color)
    for strip in strips:
        if len(strip) < 2:
            continue
        batch = batch_for_shader(
            shader,
            "LINE_STRIP",
            {"pos": [tuple(p) for p in strip]},
        )
        batch.draw(shader)
    try:
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set("NONE")
    except Exception:
        pass
    gpu.state.blend_set("NONE")


def _holo_point(center, u, v, n, x: float, y: float, z: float) -> Vector:
    return center + n * (CLIP_DROP_LIFT + z) + u * x + v * y


def _holo_ring(center, u, v, n, z: float, radius: float, segments: int) -> List[Vector]:
    pts: List[Vector] = []
    for i in range(segments + 1):
        t = (i / float(segments)) * (2.0 * math.pi)
        pts.append(
            _holo_point(
                center, u, v, n, math.cos(t) * radius, math.sin(t) * radius, z
            )
        )
    return pts


def _holo_circle_vertical(center, u, v, n, z: float, radius: float, segments: int, axis: int) -> List[Vector]:
    pts: List[Vector] = []
    for i in range(segments + 1):
        t = (i / float(segments)) * (2.0 * math.pi)
        c = math.cos(t) * radius
        s = math.sin(t) * radius
        if axis == 0:
            pts.append(_holo_point(center, u, v, n, c, 0.0, z + s))
        else:
            pts.append(_holo_point(center, u, v, n, 0.0, c, z + s))
    return pts


def _hologram_layers(center, u, v, n, now: float):
    """Wire figure, floor rings, and a scan band. No scene objects."""
    body: List[List[Vector]] = []
    scan: List[List[Vector]] = []
    flicker = 0.72 + 0.28 * math.sin(now * 17.0)
    joints = (
        ((0.0, 0.0, 1.46), (0.0, 0.0, 0.92)),
        ((-0.24, 0.0, 1.38), (0.24, 0.0, 1.38)),
        ((-0.24, 0.0, 1.38), (-0.34, 0.0, 1.05)),
        ((-0.34, 0.0, 1.05), (-0.28, 0.05, 0.76)),
        ((0.24, 0.0, 1.38), (0.34, 0.0, 1.05)),
        ((0.34, 0.0, 1.05), (0.28, 0.05, 0.76)),
        ((-0.13, 0.0, 0.92), (0.13, 0.0, 0.92)),
        ((-0.13, 0.0, 0.92), (-0.15, 0.0, 0.46)),
        ((-0.15, 0.0, 0.46), (-0.13, 0.03, 0.02)),
        ((0.13, 0.0, 0.92), (0.15, 0.0, 0.46)),
        ((0.15, 0.0, 0.46), (0.13, 0.03, 0.02)),
        ((0.0, 0.0, 0.02), (0.0, 0.0, 1.72)),
    )
    for a, b in joints:
        body.append([
            _holo_point(center, u, v, n, *a),
            _holo_point(center, u, v, n, *b),
        ])
    body.append(_holo_circle_vertical(center, u, v, n, 1.58, 0.12, 18, 0))
    body.append(_holo_circle_vertical(center, u, v, n, 1.58, 0.12, 18, 1))
    body.append(_holo_ring(center, u, v, n, 1.22, 0.16, 16))
    body.append(_holo_ring(center, u, v, n, 0.92, 0.13, 14))
    phase = (now * 0.42) % 1.0
    scan_z = 0.04 + phase * 1.68
    scan.append(_holo_ring(center, u, v, n, scan_z, 0.22 + 0.06 * math.sin(phase * math.pi), 28))
    echo = scan_z - 0.08
    if echo > 0.02:
        scan.append(_holo_ring(center, u, v, n, echo, 0.16, 20))
    near: List[List[Vector]] = []
    far: List[List[Vector]] = []
    for i, radius in enumerate((0.28, 0.55)):
        spin = now * (0.35 + i * 0.12)
        segs = 28
        bucket = near if i == 0 else far
        shift = int(spin * segs)
        for s in range(segs):
            if (s + shift) % 9 < 2:
                continue
            t0a = (s / float(segs)) * (2.0 * math.pi)
            t1a = ((s + 1) / float(segs)) * (2.0 * math.pi)
            bucket.append([
                _holo_point(center, u, v, n, math.cos(t0a) * radius, math.sin(t0a) * radius, 0.0),
                _holo_point(center, u, v, n, math.cos(t1a) * radius, math.sin(t1a) * radius, 0.0),
            ])
    floor = ((near, 0.35), (far, 0.18))
    return body, scan, floor, flicker


def _draw_view_drop() -> None:
    if not _DRAG.get("active"):
        return
    point = _DRAG.get("point")
    if point is None or _DRAG.get("over_strip"):
        return
    normal = _DRAG.get("normal") or Vector((0.0, 0.0, 1.0))
    u, v, n = _drop_plane_basis(normal)
    t0 = float(_DRAG.get("t0") or 0.0)
    now = time.perf_counter() - t0 if t0 > 0.0 else 0.0
    body, scan, floor, flicker = _hologram_layers(point, u, v, n, now)
    if _STATE.get("whole_scene"):
        other, other_scan, _floor, _flicker = _hologram_layers(
            point + u * 0.9, u, v, n, now + 0.35
        )
        body.extend(other)
        scan.extend(other_scan)
    cyan = (0.45, 0.92, 1.0)
    _draw_polyline_strips_3d(body, (cyan[0], cyan[1], cyan[2], 0.55 * flicker), 1.25)
    _draw_polyline_strips_3d(scan, (0.75, 1.0, 1.0, 0.85), 1.6)
    for strips, alpha in floor:
        _draw_polyline_strips_3d(
            strips, (cyan[0], cyan[1], cyan[2], alpha * flicker), 1.15
        )


def _armature_anchor_world(arm) -> Vector:
    for name in ("Dummy_Root", "Root_M"):
        bone = arm.data.bones.get(name) if arm is not None else None
        if bone is not None:
            return (arm.matrix_world @ bone.head_local).copy()
    return arm.matrix_world.translation.copy()


def _stage_group(arm) -> set:
    """Stage empties, both rigs, and anything parented to them."""
    found = set()
    if arm is None:
        return found
    parent = getattr(arm, "parent", None)
    for col in bpy.data.collections:
        if not col.get("gr2lab_scene_empties"):
            continue
        names = {obj.name for obj in col.objects}
        if (parent is not None and parent.name in names) or arm.name in names:
            found.update(col.objects)
    if parent is not None:
        found.add(parent)
    found.add(arm)
    changed = True
    while changed:
        changed = False
        for obj in bpy.data.objects:
            if obj in found or obj.parent not in found:
                continue
            found.add(obj)
            changed = True
    return found


def _move_stage_to_drop(context, arm, point: Vector) -> bool:
    """Slide the whole stage so this rig's anchor lands on the drop point."""
    if arm is None or point is None:
        return False
    try:
        context.view_layer.update()
    except Exception:
        pass
    group = _stage_group(arm)
    if len(group) < 2 and getattr(arm, "parent", None) is None:
        return False
    delta = point - _armature_anchor_world(arm)
    roots = [obj for obj in group if obj.parent not in group]
    for obj in roots:
        world = obj.matrix_world.copy()
        world.translation += delta
        obj.matrix_world = world
    try:
        context.view_layer.update()
    except Exception:
        pass
    return True


def _snap_armature_to_drop(arm, point: Vector) -> None:
    if arm is None or point is None:
        return
    mat = arm.matrix_world.copy()
    mat.translation += point - _armature_anchor_world(arm)
    arm.matrix_world = mat


def _armature_for_imported_clip(filepath: str, before: set) -> Optional[bpy.types.Object]:
    from .bridge_blender import PROP_SOURCE_GR2

    new_ones = [
        obj
        for obj in bpy.data.objects
        if getattr(obj, "type", None) == "ARMATURE" and obj.name not in before
    ]
    if not new_ones:
        return None
    want_key = Path(filepath).resolve().as_posix().casefold()
    matched = []
    for obj in new_ones:
        src = str(obj.get(PROP_SOURCE_GR2) or "").strip()
        if not src:
            continue
        try:
            if Path(src).resolve().as_posix().casefold() == want_key:
                matched.append(obj)
        except OSError:
            continue
    if matched:
        return matched[-1]
    return new_ones[-1]


def _select_armature(context, arm) -> None:
    if arm is None or getattr(arm, "type", None) != "ARMATURE":
        return
    try:
        for obj in context.view_layer.objects:
            obj.select_set(obj == arm)
        context.view_layer.objects.active = arm
    except Exception:
        pass


class _StripImportHost:
    """Stand-in for the GR2 import operator while a strip drag preloads."""

    def __init__(self, fields: dict) -> None:
        self.fps = 30
        self.body_base = "AUTO"
        self.bb_hint = ""
        self.body_choice = ""
        self.import_apply = "NEW_RIG"
        self.add_rig = True
        self.second_character = False
        self.scene_empties = False
        self.center_parent = False
        self.import_meshes = "BODY_CLOTH"
        self.import_akelka = True
        self.import_clear_meshes = True
        self.import_tail = True
        self.rig_shapes = True
        self.rig_ik_limbs = True
        self.rig_ik_head = True
        self.partner_body_base = "AUTO"
        for key, value in fields.items():
            setattr(self, key, value)

    def report(self, _level, message) -> None:
        print(f"[GR2 LAB] {message}", flush=True)


def _end_preload_ui(pre: Optional[dict]) -> None:
    if not pre:
        return
    progress = pre.get("progress")
    if progress is not None:
        try:
            progress.end()
        except Exception:
            pass
        pre["progress"] = None
    try:
        from .import_progress_ui import end as overlay_end

        overlay_end()
    except Exception:
        pass


_DROP_HOLD_COLLECTION = "GR2LAB_preload_hold"


def _find_layer_collection(layer, name: str):
    if layer.name == name:
        return layer
    for child in layer.children:
        found = _find_layer_collection(child, name)
        if found is not None:
            return found
    return None


def _hold_collection(context, pre: dict):
    name = str(pre.get("hold_col") or _DROP_HOLD_COLLECTION)
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    root = context.scene.collection
    if col.name not in root.children:
        root.children.link(col)
    pre["hold_col"] = col.name
    return col


def _set_hold_excluded(context, name: str, excluded: bool) -> None:
    layer = _find_layer_collection(context.view_layer.layer_collection, name)
    if layer is None or bool(layer.exclude) == bool(excluded):
        return
    try:
        layer.exclude = bool(excluded)
    except Exception:
        pass


_DROP_BUILD_LAYER = "GR2LAB_preload_build"


def _ensure_drop_build_layer(context, pre: dict) -> None:
    """User view hides the hold collection. A second view layer can still edit it."""
    col = _hold_collection(context, pre)
    _set_hold_excluded(context, col.name, True)
    scene = context.scene
    vl = scene.view_layers.get(_DROP_BUILD_LAYER)
    if vl is None:
        vl = scene.view_layers.new(_DROP_BUILD_LAYER)
    layer = _find_layer_collection(vl.layer_collection, col.name)
    if layer is not None and layer.exclude:
        try:
            layer.exclude = False
        except Exception:
            pass
    pre["build_vl"] = vl.name
    pre["state"]["strip_hold_collection"] = col.name


def _drop_build_view_layer(context, pre: dict) -> None:
    name = str(pre.get("build_vl") or "")
    pre["build_vl"] = ""
    if not name or context is None:
        return
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    vl = scene.view_layers.get(name)
    if vl is None or vl == getattr(context, "view_layer", None):
        return
    active = getattr(vl.objects, "active", None)
    if active is not None and getattr(active, "mode", "OBJECT") != "OBJECT":
        try:
            with context.temp_override(
                scene=scene, view_layer=vl, active_object=active, object=active
            ):
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    try:
        scene.view_layers.remove(vl)
    except Exception:
        pass


def _stash_preload_collection(context, pre: dict) -> None:
    """Move the in-progress rig into a view-layer-excluded collection."""
    before = pre.get("before") or set()
    col = _hold_collection(context, pre)
    for obj in bpy.data.objects:
        if obj.name in before:
            continue
        try:
            if obj.name not in col.objects:
                col.objects.link(obj)
        except Exception:
            pass
        for user in list(obj.users_collection):
            if user == col:
                continue
            try:
                user.objects.unlink(obj)
            except Exception:
                pass
    _set_hold_excluded(context, col.name, True)


def _show_preload_collection(context, pre: dict) -> None:
    name = str(pre.get("hold_col") or "")
    if name:
        _set_hold_excluded(context, name, False)


def _release_preload_collection(context, pre: dict) -> None:
    """Put the finished rig back in the scene collection and drop the hold."""
    _drop_build_view_layer(context, pre)
    name = str(pre.get("hold_col") or "")
    if not name:
        return
    _set_hold_excluded(context, name, False)
    col = bpy.data.collections.get(name)
    root = context.scene.collection
    if col is not None:
        for obj in list(col.objects):
            try:
                if obj.name not in root.objects:
                    root.objects.link(obj)
            except Exception:
                pass
            try:
                col.objects.unlink(obj)
            except Exception:
                pass
        try:
            if col.name in root.children:
                root.children.unlink(col)
        except Exception:
            pass
        try:
            if col.users == 0:
                bpy.data.collections.remove(col)
        except Exception:
            pass
    pre["hold_col"] = ""


def _hide_preload_objects(pre: dict) -> None:
    before = pre.get("before") or set()
    hidden = pre.setdefault("hidden", set())
    for obj in bpy.data.objects:
        if obj.name in before:
            continue
        try:
            if not obj.hide_viewport:
                obj.hide_viewport = True
            hidden.add(obj.name)
        except Exception:
            pass


def _reveal_preload_objects(pre: dict) -> None:
    for name in list(pre.get("hidden") or ()):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        try:
            obj.hide_viewport = False
        except Exception:
            pass


def _place_ready_drop(context, pre: dict, point: Optional[Vector]) -> None:
    arm = (pre.get("state") or {}).get("arm")
    if point is None or arm is None:
        _reveal_preload_objects(pre)
        return
    register_pending_clip_drop(
        pre["path"], point, context, whole_scene=_browser_whole_scene(context)
    )
    _release_preload_collection(context, pre)
    apply_pending_clip_drop(context, pre["path"], arm, consume=False)
    try:
        arm.hide_viewport = False
    except Exception:
        pass
    _reveal_preload_objects(pre)
    _select_armature(context, arm)


def _discard_datablock(data) -> None:
    if data is None or getattr(data, "users", 1) != 0:
        return
    ident = str(getattr(getattr(data, "bl_rna", None), "identifier", "") or "")
    coll_name = ident[:1].lower() + ident[1:] + "s" if ident else ""
    coll = getattr(bpy.data, coll_name, None) if coll_name else None
    if coll is None:
        return
    try:
        if coll.get(data.name) != data:
            return
        coll.remove(data)
    except Exception:
        pass


def _delete_preload_objects(pre: dict) -> None:
    before = pre.get("before") or set()
    actions = []
    for obj in list(bpy.data.objects):
        if obj.name in before:
            continue
        anim = getattr(obj, "animation_data", None)
        action = getattr(anim, "action", None) if anim is not None else None
        if action is not None:
            actions.append(action)
        data = obj.data
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            continue
        _discard_datablock(data)
    for action in actions:
        _discard_datablock(action)
    cols = pre.get("cols") or set()
    for col in list(bpy.data.collections):
        if col.name in cols or len(col.objects) != 0:
            continue
        try:
            bpy.data.collections.remove(col)
        except Exception:
            pass


def _abort_clip_preload() -> None:
    pre = _DRAG.get("preload")
    if not pre or pre.get("aborted"):
        _DRAG["preload"] = None
        return
    pre["aborted"] = True
    _DRAG["preload"] = None
    if not pre.get("done"):
        _drop_build_view_layer(bpy.context, pre)
        _delete_preload_objects(pre)
        _release_preload_collection(bpy.context, pre)
        print("[GR2 LAB] strip preload cancelled", flush=True)
    _end_preload_ui(pre)
    clear_pending_clip_drop()


def _show_preload_bar(context, pre: dict) -> None:
    if pre.get("bar") or pre.get("ready"):
        return
    from .import_modal import stage_count
    from .import_progress_ui import begin as overlay_begin

    overlay_begin(
        stage_total=stage_count(pre["host"]),
        context=context,
        area_ptr=int(pre.get("area_ptr") or 0),
    )
    pre["bar"] = True


def _begin_clip_preload(context, filepath: str, point: Vector) -> None:
    from .bridge_blender import Gr2LabProgress

    path = str(filepath or "").strip()
    if not path or _DRAG.get("preload") is not None:
        return
    host = _StripImportHost(_strip_import_kwargs(context, path))
    prefs = context.preferences.addons[__package__].preferences
    area = getattr(context, "area", None)
    area_ptr = (
        int(area.as_pointer())
        if area is not None and getattr(area, "type", "") == "VIEW_3D"
        else 0
    )
    state = {
        "step": 0,
        "path": path,
        "apply": "NEW_RIG",
        "target": None,
        "prefs": prefs,
        "preload_quiet": True,
        "defer_clip_snap": True,
        "strip_hide_build": True,
    }
    pre = {
        "path": path,
        "host": host,
        "state": state,
        "progress": None,
        "before": {obj.name for obj in bpy.data.objects},
        "cols": {col.name for col in bpy.data.collections},
        "hidden": set(),
        "done": False,
        "commit": False,
        "commit_point": None,
        "ready": False,
        "err": "",
        "aborted": False,
        "bar": False,
        "area_ptr": area_ptr,
    }
    _DRAG["preload"] = pre
    _DRAG["preload_failed"] = ""
    try:
        progress = Gr2LabProgress(context, header=False)
        progress.begin(100)
        pre["progress"] = progress
        register_pending_clip_drop(
            path, point, context, whole_scene=_browser_whole_scene(context)
        )
    except Exception:
        _abort_clip_preload()
        raise
    try:
        _ensure_drop_build_layer(context, pre)
    except Exception:
        traceback.print_exc()
    print(f"[GR2 LAB] strip preload started {Path(path).name}", flush=True)


def _preload_pump_limits(committed: bool, pre: dict) -> Tuple[int, Optional[float]]:
    if pre.get("ready") and not committed:
        return 0, None
    if committed:
        return CLIP_PRELOAD_COMMIT_SLICES, None
    return 1, 0.008


def _pump_clip_preload(context) -> None:
    pre = _DRAG.get("preload")
    if not pre or pre.get("done") or pre.get("aborted"):
        return
    committed = bool(pre.get("commit"))
    point = pre.get("commit_point") if committed else _DRAG.get("point")
    if point is not None:
        register_pending_clip_drop(
            pre["path"], point, context, whole_scene=_browser_whole_scene(context)
        )
    from . import bridge_blender
    from .import_modal import advance

    state = pre["state"]
    state["preload_quiet"] = not committed
    progress = pre.get("progress") if committed else None
    max_slices, budget = _preload_pump_limits(committed, pre)
    if max_slices <= 0:
        return
    t_budget = time.perf_counter() + budget if budget else 0.0
    for _ in range(max_slices):
        if budget and time.perf_counter() >= t_budget:
            break
        step_before = int(state.get("step") or 0)
        build_vl = None
        build_name = str(pre.get("build_vl") or "")
        if build_name:
            build_vl = context.scene.view_layers.get(build_name)
        try:
            if build_vl is not None:
                with context.temp_override(view_layer=build_vl):
                    finished, err = advance(
                        pre["host"],
                        context,
                        state,
                        progress=progress,
                        bridge=bridge_blender,
                    )
            else:
                finished, err = advance(
                    pre["host"],
                    context,
                    state,
                    progress=progress,
                    bridge=bridge_blender,
                )
        except Exception as exc:
            traceback.print_exc()
            finished, err = True, str(exc)
        step_after = int(state.get("step") or 0)
        if err:
            pre["err"] = str(err)
            _DRAG["preload_failed"] = str(err)
            _abort_clip_preload()
            return
        if finished:
            _release_preload_collection(context, pre)
            if not committed:
                pre["ready"] = True
                point = _DRAG.get("point")
                _place_ready_drop(context, pre, point)
                print("[GR2 LAB] strip preload ready at cursor", flush=True)
                return
            pre["done"] = True
            arm = state.get("arm")
            if arm is not None:
                try:
                    arm.hide_viewport = False
                except Exception:
                    pass
            _reveal_preload_objects(pre)
            try:
                finish_deferred_edit_copy(context, state)
            except Exception:
                traceback.print_exc()
            _end_preload_ui(pre)
            return
        _stash_preload_collection(context, pre)
        if step_after == step_before:
            return


def _maybe_begin_clip_preload(context) -> None:
    if _DRAG.get("shift") or _DRAG.get("preload") is not None:
        return
    if _DRAG.get("preload_failed") or _DRAG.get("over_strip"):
        return
    point = _DRAG.get("point")
    path = str(_DRAG.get("filepath") or "")
    if point is None or not path:
        return
    dx = float(_DRAG.get("mx") or 0.0) - float(_DRAG.get("sx") or 0.0)
    dy = float(_DRAG.get("my") or 0.0) - float(_DRAG.get("sy") or 0.0)
    if (dx * dx + dy * dy) < CLIP_DRAG_THRESHOLD * CLIP_DRAG_THRESHOLD:
        return
    try:
        _begin_clip_preload(context, path, point)
        _pump_clip_preload(context)
    except Exception:
        traceback.print_exc()
        _abort_clip_preload()


def _import_clip_at(
    context, filepath: str, point: Optional[Vector], *, place: bool, whole_scene: bool = False
) -> None:
    from .bridge_blender import exec_gr2_import_quick

    if place and point is not None:
        register_pending_clip_drop(
            filepath, point, context, whole_scene=whole_scene
        )
    else:
        clear_pending_clip_drop(context)
    center = _browser_center_parent(context) if whole_scene else False
    exec_gr2_import_quick(
        context, filepath, whole_scene=whole_scene, center_parent=center
    )


def _begin_clip_drag(context, event, filepath: str) -> bool:
    path = str(filepath or "")
    if not path:
        return False
    try:
        bpy.ops.gr2lab.anim_clip_drag(
            "INVOKE_DEFAULT",
            filepath=path,
            start_x=float(event.mouse_region_x),
            start_y=float(event.mouse_region_y),
            use_dialog=bool(getattr(event, "shift", False)),
        )
        return True
    except Exception:
        traceback.print_exc()
        return False


def _on_move(context, event) -> set:
    region = getattr(context, "region", None)
    if region is None:
        return {"PASS_THROUGH"}
    mx, my = _event_xy(event)
    found = _hit(region, mx, my)
    path = ""
    if found and found[0] == "clip":
        path = found[1]
    layout = _LAYOUTS.get(region.as_pointer() if region else 0) or {}
    _POINTER["mx"] = mx
    _POINTER["my"] = my
    _POINTER["over_strip"] = _over_box(layout.get("strip"), _POINTER["mx"], _POINTER["my"])
    split = _split_at(layout, _POINTER["mx"], _POINTER["my"])
    was = bool(_POINTER.get("split_cursor"))
    if split and not was:
        try:
            context.window.cursor_set("MOVE_X")
        except Exception:
            pass
    elif was and not split:
        try:
            context.window.cursor_set("DEFAULT")
        except Exception:
            pass
    _POINTER["split_cursor"] = bool(split)
    if path != _HOVER["path"]:
        _HOVER["path"] = path
        _HOVER["since"] = time.time()
        _HOVER["gen"] = int(_HOVER.get("gen") or 0) + 1
        if not path:
            _HOVER["status"] = ""
            _HOVER["shown"] = ""
            _HOVER["poses"] = None
        else:
            cached = _stick_mem_get(path)
            if cached is not None:
                _show_pose(path, cached, "sticks cached")
            else:
                _HOVER["status"] = "sticks…"
        _tag()
    return {"PASS_THROUGH"}


def _on_wheel(context, event) -> set:
    region = getattr(context, "region", None)
    layout = _LAYOUTS.get(region.as_pointer() if region else 0) or {}
    mx, my = _event_xy(event)
    if not _over_box(layout.get("strip"), mx, my):
        return {"PASS_THROUGH"}
    delta = -1 if event.type == "WHEELUPMOUSE" else 1
    if _over_box(layout.get("clip_box"), mx, my):
        _STATE["clip_scroll"] = max(0, _STATE["clip_scroll"] + delta)
    else:
        _STATE["scene_scroll"] = max(0, _STATE["scene_scroll"] + delta)
    _tag()
    return {"FINISHED"}


def _explorer_target() -> str:
    """Highlighted clip, then the clip under the pointer, then the open scene folder."""
    for candidate in (
        str(_STATE.get("clip_path") or ""),
        str(_HOVER.get("path") or ""),
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file() or path.is_dir():
            return str(path)
    data = _INDEX.get("data") or _load_index()
    if not data:
        return ""
    scene = _selected(_filtered(data))
    folder = str((scene or {}).get("dir") or "")
    if folder and Path(folder).is_dir():
        return folder
    return ""


_REVEAL_PS = r"""
$target = $env:GR2LAB_REVEAL
$unpacked = $env:GR2LAB_UNPACKED
$dry = $env:GR2LAB_REVEAL_DRY
$sep = [string][IO.Path]::DirectorySeparatorChar
function Norm([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return "" }
  try { $p = [IO.Path]::GetFullPath($p) } catch { return "" }
  $p = $p.ToLowerInvariant()
  if ($p.Length -gt 3 -and $p.EndsWith($sep)) { $p = $p.TrimEnd($sep) }
  return $p
}
function IsUnder([string]$child, [string]$parent) {
  if (-not $parent -or -not $child) { return $false }
  if ($child -eq $parent) { return $true }
  $prefix = $parent
  if (-not $prefix.EndsWith($sep)) { $prefix += $sep }
  return $child.StartsWith($prefix)
}
if ([string]::IsNullOrWhiteSpace($target)) { Write-Output "NONE"; exit 1 }
$item = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
if (-not $item) { Write-Output "NONE"; exit 1 }
if ($item.PSIsContainer) {
  $folder = $item.FullName
  $leaf = ""
} else {
  $folder = $item.DirectoryName
  $leaf = $item.Name
}
$folderN = Norm $folder
$unpackedN = Norm $unpacked
$shell = New-Object -ComObject Shell.Application
$best = $null
$bestPath = ""
$bestScore = -1
foreach ($w in @($shell.Windows())) {
  try {
    $exe = [string]$w.FullName
    if ($exe -notlike "*explorer.exe") { continue }
    $p = [string]$w.Document.Folder.Self.Path
  } catch { continue }
  $pn = Norm $p
  if (-not $pn) { continue }
  $exact = ($pn -eq $folderN)
  $windowInUnpacked = [bool]($unpackedN -and (IsUnder $pn $unpackedN))
  $targetInWindow = IsUnder $folderN $pn
  $ok = $exact -or $windowInUnpacked
  if (-not $ok) { continue }
  if ($exact) { $score = 3000000 }
  elseif ($targetInWindow) { $score = 2000000 + $pn.Length }
  else { $score = 1000000 }
  if ($score -gt $bestScore) {
    $bestScore = $score
    $best = $w
    $bestPath = $p
  }
}
if (-not $best) { Write-Output "NONE"; exit 1 }
if ($dry -eq "1") { Write-Output ("REUSED " + $bestPath); exit 0 }
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Gr2LabWin552 {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
}
"@
try { $hwnd = New-Object IntPtr ([int64]$best.HWND) } catch { $hwnd = [IntPtr]::Zero }
if ($hwnd -ne [IntPtr]::Zero) {
  if ([Gr2LabWin552]::IsIconic($hwnd)) { [void][Gr2LabWin552]::ShowWindow($hwnd, 9) }
  [Gr2LabWin552]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)
  [void][Gr2LabWin552]::SetForegroundWindow($hwnd)
  [Gr2LabWin552]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
}
$cur = Norm ([string]$best.Document.Folder.Self.Path)
if ($cur -ne $folderN) {
  $best.Navigate($folder)
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 50
    try {
      $cur = Norm ([string]$best.Document.Folder.Self.Path)
      if ($cur -eq $folderN) { break }
    } catch {}
  }
}
if ($leaf -and ((Norm ([string]$best.Document.Folder.Self.Path)) -eq $folderN)) {
  try {
    $parsed = $best.Document.Folder.ParseName($leaf)
    if ($parsed) { [void]$best.Document.SelectItem($parsed, 0x1D) }
    else { [void]$best.Document.SelectItem($leaf, 0x1D) }
  } catch {}
}
Write-Output "REUSED"
exit 0
"""


def _reveal_in_open_explorer(path: str, unpacked: str, *, dry: bool = False) -> bool:
    """Select path in an Explorer window already on the unpacked data. False if none is open."""
    if not path:
        return False
    import base64

    encoded = base64.b64encode(_REVEAL_PS.encode("utf-16le")).decode("ascii")
    env = os.environ.copy()
    env["GR2LAB_REVEAL"] = str(path)
    env["GR2LAB_UNPACKED"] = str(unpacked or "")
    env["GR2LAB_REVEAL_DRY"] = "1" if dry else "0"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            capture_output=True,
            text=True,
            timeout=8,
            env=env,
            creationflags=flags,
        )
    except Exception:
        traceback.print_exc()
        return False
    out = proc.stdout or ""
    if "REUSED" not in out and proc.returncode not in (0, 1) and proc.stderr:
        print("gr2lab explorer:", proc.stderr.strip())
    return "REUSED" in out


def _open_explorer() -> None:
    target = _explorer_target()
    if not target:
        return
    path = str(Path(target))
    root = unpacked_root(bpy.context)
    unpacked = str(root) if root else ""
    if _reveal_in_open_explorer(path, unpacked):
        return
    if Path(path).is_file():
        subprocess.Popen(["explorer", f"/select,{path}"])
    else:
        subprocess.Popen(["explorer", path])


def handle_strip_click(context, event, mx=None, my=None) -> bool:
    """True when the animation strip used this click."""
    if not _browser_on(context):
        return False
    region = getattr(context, "region", None)
    if region is None:
        return False
    refresh_strip_pick_data(context)
    layout = _LAYOUTS.get(region.as_pointer()) or {}
    if mx is None:
        mx = float(event.mouse_region_x)
    if my is None:
        my = float(event.mouse_region_y)
    mx = float(mx)
    my = float(my)
    strip = layout.get("strip")
    if not strip or not _over_box(strip, mx, my):
        return False
    which = _split_at(layout, mx, my)
    if which and getattr(event, "value", "PRESS") == "PRESS":
        try:
            bpy.ops.gr2lab.anim_pane_resize("INVOKE_DEFAULT", which=which)
        except Exception:
            traceback.print_exc()
        return True
    found = _hit(region, mx, my)
    if found is None:
        return False
    kind, payload = found
    if kind == "prepare":
        try:
            bpy.ops.gr2lab.anim_prepare("INVOKE_DEFAULT")
        except RuntimeError:
            pass
        _tag()
        return True
    if kind == "act":
        try:
            index = _ACTS.index(_STATE["act"])
        except ValueError:
            index = 0
        _STATE["act"] = _ACTS[(index + 1) % len(_ACTS)]
        _STATE["scene_scroll"] = 0
        _STATE["filtered_key"] = ""
    elif kind == "camp":
        _STATE["camp"] = not _STATE["camp"]
        _STATE["filtered_key"] = ""
        _STATE["scene_scroll"] = 0
    elif kind == "rom":
        _STATE["rom"] = not _STATE["rom"]
        _STATE["filtered_key"] = ""
        _STATE["scene_scroll"] = 0
    elif kind == "whole":
        _STATE["whole_scene"] = not bool(_STATE.get("whole_scene"))
    elif kind == "center":
        nxt = not bool(_STATE.get("center_parent"))
        _sync_browser_center_parent(context, nxt)
    elif kind == "edit":
        _STATE["edit_copy"] = not bool(_STATE.get("edit_copy"))
    elif kind == "explorer":
        _open_explorer()
    elif kind == "search":
        _begin_gpu_search(region, mx)
        return True
    elif kind == "prev":
        _step_scene(-1)
    elif kind == "next":
        _step_scene(1)
    elif kind == "scene":
        _STATE["scene_id"] = payload
        _STATE["clip_scroll"] = 0
    elif kind == "clip" and payload:
        ev_val = getattr(event, "value", "PRESS")
        if ev_val == "RELEASE":
            return False
        path = str(payload)
        if ev_val == "DOUBLE_CLICK":
            _select_strip_clip(context, path)
            _place_strip_clip_at_cursor(context, path)
            return True
        if not _clip_is_selected(path):
            _select_strip_clip(context, path)
            return True
        if _begin_clip_drag(context, event, path):
            return True
        return False
    _tag()
    return True


def _on_click(context, event) -> set:
    if handle_strip_click(context, event):
        return {"FINISHED"}
    return {"PASS_THROUGH"}


class GR2LAB_OT_anim_pane_resize(bpy.types.Operator):
    """Drag the divider between animation-strip columns."""

    bl_idname = "gr2lab.anim_pane_resize"
    bl_label = "Resize animation strip columns"
    bl_options = {"INTERNAL"}

    which: bpy.props.StringProperty(options={"SKIP_SAVE"})

    def invoke(self, context, event):
        which = str(self.which or "")
        if which not in {"scene", "clip"}:
            return {"CANCELLED"}
        self._which = which
        self._scene = float(_STATE.get("pane_scene") or 0.38)
        self._clip = float(_STATE.get("pane_clip") or 0.26)
        try:
            context.window.cursor_modal_set("MOVE_X")
        except Exception:
            pass
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _restore_cursor(self) -> None:
        try:
            bpy.context.window.cursor_modal_restore()
        except Exception:
            pass

    def modal(self, context, event):
        region = getattr(context, "region", None)
        width = float(region.width) if region is not None else 0.0
        if event.type == "MOUSEMOVE" and width > 0.0:
            _set_pane_from_mouse(self._which, float(event.mouse_region_x), width)
            _tag()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._restore_cursor()
            _tag()
            return {"FINISHED"}
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            _STATE["pane_scene"] = self._scene
            _STATE["pane_clip"] = self._clip
            self._restore_cursor()
            _tag()
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


class GR2LAB_OT_anim_clip_drag(bpy.types.Operator):
    """Drag a cinematic clip from the strip into the 3D view."""

    bl_idname = "gr2lab.anim_clip_drag"
    bl_label = "Drag animation clip"
    bl_options = {"INTERNAL"}

    filepath: bpy.props.StringProperty(options={"SKIP_SAVE"})
    start_x: bpy.props.FloatProperty(default=0.0, options={"SKIP_SAVE"})
    start_y: bpy.props.FloatProperty(default=0.0, options={"SKIP_SAVE"})
    use_dialog: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})

    def _clear(self) -> None:
        pre = _DRAG.get("preload")
        if pre is not None and not pre.get("done"):
            _abort_clip_preload()
        else:
            _DRAG["preload"] = None
        _DRAG["preload_failed"] = ""
        timer = _DRAG.get("wm_timer")
        if timer is not None:
            try:
                bpy.context.window_manager.event_timer_remove(timer)
            except Exception:
                pass
        _DRAG["active"] = False
        _DRAG["filepath"] = ""
        _DRAG["label"] = ""
        _DRAG["point"] = None
        _DRAG["normal"] = None
        _DRAG["over_strip"] = False
        _DRAG["t0"] = 0.0
        _DRAG["wm_timer"] = None
        try:
            bpy.context.window.cursor_modal_restore()
        except Exception:
            pass
        _tag()

    def _sync_pointer(self, context, event) -> None:
        region = getattr(context, "region", None)
        if region is None:
            return
        mx = float(event.mouse_region_x)
        my = float(event.mouse_region_y)
        _DRAG["mx"] = mx
        _DRAG["my"] = my
        _DRAG["region"] = region.as_pointer()
        _DRAG["region_w"] = float(region.width)
        _DRAG["region_h"] = float(region.height)
        layout = _LAYOUTS.get(region.as_pointer()) or {}
        _DRAG["over_strip"] = _over_box(layout.get("strip"), mx, my)
        _DRAG["last_move_t"] = time.perf_counter()
        now = float(_DRAG["last_move_t"])
        last_hit = float(_DRAG.get("hit_t") or 0.0)
        hmx = float(_DRAG.get("hit_mx") or mx)
        hmy = float(_DRAG.get("hit_my") or my)
        ddx = mx - hmx
        ddy = my - hmy
        moved = (ddx * ddx + ddy * ddy) >= (
            CLIP_DRAG_HIT_MIN_PX * CLIP_DRAG_HIT_MIN_PX
        )
        origin, direction = _drop_view_ray(context, mx, my)
        plane_co = _DRAG.get("plane_co")
        plane_no = _DRAG.get("normal")
        if origin is not None and plane_co is not None and plane_no is not None:
            slid = _drop_on_plane(origin, direction, plane_co, plane_no)
            if slid is not None:
                _DRAG["point"] = slid
        if plane_co is None or moved or (now - last_hit) >= CLIP_DRAG_HIT_INTERVAL_S:
            point, normal = _drop_plane_hit(context, mx, my)
            _DRAG["point"] = point
            _DRAG["normal"] = normal
            _DRAG["plane_co"] = point.copy() if point is not None else None
            _DRAG["hit_mx"] = mx
            _DRAG["hit_my"] = my
            _DRAG["hit_t"] = now
        _tag()

    def invoke(self, context, event):
        path = str(self.filepath or "")
        if not path:
            return {"CANCELLED"}
        _DRAG["active"] = True
        _DRAG["filepath"] = path
        _DRAG["label"] = Path(path).stem
        _DRAG["sx"] = float(self.start_x or event.mouse_region_x)
        _DRAG["sy"] = float(self.start_y or event.mouse_region_y)
        _DRAG["shift"] = bool(self.use_dialog)
        _DRAG["t0"] = time.perf_counter()
        _DRAG["last_move_t"] = _DRAG["t0"]
        _DRAG["hit_t"] = 0.0
        _DRAG["hit_mx"] = float(self.start_x or event.mouse_region_x)
        _DRAG["hit_my"] = float(self.start_y or event.mouse_region_y)
        _DRAG["timer_tick"] = 0
        _DRAG["wm_timer"] = context.window_manager.event_timer_add(
            1.0 / 30.0, window=context.window
        )
        self._sync_pointer(context, event)
        try:
            context.window.cursor_modal_set("HAND")
        except Exception:
            pass
        context.window_manager.modal_handler_add(self)
        _tag()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not _ALIVE:
            self._clear()
            return {"CANCELLED"}
        if event.type == "TIMER":
            _DRAG["timer_tick"] = int(_DRAG.get("timer_tick") or 0) + 1
            pre = _DRAG.get("preload")
            if pre is not None and pre.get("commit") and not pre.get("done"):
                tick = int(_DRAG.get("timer_tick") or 0)
                if (tick % CLIP_PRELOAD_PUMP_EVERY) == 0:
                    _pump_clip_preload(context)
                pre = _DRAG.get("preload")
                failed = str(_DRAG.get("preload_failed") or "")
                if pre is not None and pre.get("commit") and (
                    failed or pre.get("done")
                ):
                    self._clear()
                    if failed:
                        self.report({"ERROR"}, failed)
                        return {"CANCELLED"}
                    return {"FINISHED"}
            redraw_iv = 0.0
            _tag(redraw_iv)
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE":
            self._sync_pointer(context, event)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            mx = float(event.mouse_region_x)
            my = float(event.mouse_region_y)
            dx = mx - float(_DRAG.get("sx") or 0.0)
            dy = my - float(_DRAG.get("sy") or 0.0)
            moved = (dx * dx + dy * dy) >= CLIP_DRAG_THRESHOLD * CLIP_DRAG_THRESHOLD
            path = str(_DRAG.get("filepath") or "")
            region = getattr(context, "region", None)
            layout = _LAYOUTS.get(region.as_pointer() if region else 0) or {}
            over_strip = _over_box(layout.get("strip"), mx, my)
            point, _normal = _drop_plane_hit(context, mx, my)
            use_dialog = bool(_DRAG.get("shift"))
            pre = _DRAG.get("preload")
            failed = str(_DRAG.get("preload_failed") or "")
            if failed:
                self._clear()
                self.report({"ERROR"}, failed)
                return {"CANCELLED"}
            if not path:
                self._clear()
                return {"CANCELLED"}
            if use_dialog and not moved:
                self._clear()
                bpy.ops.import_scene.gr2_full("INVOKE_DEFAULT", filepath=path)
                return {"FINISHED"}
            whole_scene = _browser_whole_scene(context)
            if not moved:
                self._clear()
                _place_strip_clip_at_cursor(context, path)
                return {"FINISHED"}
            if over_strip or point is None:
                self._clear()
                return {"CANCELLED"}
            if pre is not None:
                pre["commit"] = True
                pre["commit_point"] = point
                register_pending_clip_drop(
                    pre["path"], point, context, whole_scene=whole_scene
                )
                _DRAG["active"] = False
                if pre.get("ready"):
                    _place_ready_drop(context, pre, point)
                    pre["done"] = True
                    _end_preload_ui(pre)
                    self._clear()
                    return {"FINISHED"}
                pre["state"]["preload_quiet"] = False
                pre["state"]["defer_clip_snap"] = False
                _show_preload_bar(context, pre)
                _pump_clip_preload(context)
                pre = _DRAG.get("preload")
                failed = str(_DRAG.get("preload_failed") or "")
                if failed:
                    self._clear()
                    self.report({"ERROR"}, failed)
                    return {"CANCELLED"}
                if pre is not None and not pre.get("done"):
                    _tag()
                    return {"RUNNING_MODAL"}
                self._clear()
                return {"FINISHED"}
            self._clear()
            _import_clip_at(
                context, path, point, place=True, whole_scene=whole_scene
            )
            return {"FINISHED"}
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._clear()
            return {"CANCELLED"}
        if event.type in {
            "MIDDLEMOUSE",
            "WHEELUPMOUSE",
            "WHEELDOWNMOUSE",
            "TRACKPADPAN",
            "TRACKPADZOOM",
        }:
            return {"PASS_THROUGH"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._clear()


class GR2LAB_OT_anim_browser_modal(bpy.types.Operator):
    """Viewport modal so the animation strip receives clicks on every window."""

    bl_idname = "gr2lab.anim_browser_modal"
    bl_label = "GR2 animation browser modal"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        wm = context.window_manager
        booted = set(wm.get("gr2lab_browser_modal_wins") or ())
        ptr = context.window.as_pointer()
        if ptr in booted:
            return {"CANCELLED"}
        booted.add(ptr)
        wm["gr2lab_browser_modal_wins"] = list(booted)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not _ALIVE:
            return {"CANCELLED"}
        if modal_strip_event(context, event):
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
        wm = context.window_manager
        booted = set(wm.get("gr2lab_browser_modal_wins") or ())
        booted.discard(context.window.as_pointer())
        wm["gr2lab_browser_modal_wins"] = list(booted)
        return None


def _boot_browser_modal() -> None:
    wm = bpy.context.window_manager
    booted = set(wm.get("gr2lab_browser_modal_wins") or ())
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
                        bpy.ops.gr2lab.anim_browser_modal("INVOKE_DEFAULT")
                    launched = True
                    break
                except Exception:
                    continue
            if launched:
                break
    return None


class GR2LAB_OT_anim_browser_pointer(bpy.types.Operator):
    """Hover, scroll, and click the animation browser."""

    bl_idname = "gr2lab.anim_browser_pointer"
    bl_label = "GR2 animation browser"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        from .race_picker import note_pointer, try_race_click

        if event.type == "MOUSEMOVE":
            note_pointer(context, event)
        if _browser_on(context) and modal_strip_event(context, event):
            return {"FINISHED"}
        if (
            event.type == "LEFTMOUSE"
            and event.value == "PRESS"
            and not event.shift
            and not event.alt
        ):
            found = view3d_under_pointer(context, event)
            if found is not None:
                area, region, mx, my = found
                if not pointer_over_strip(region, mx, my):
                    window = getattr(context, "window", None)
                    try:
                        with context.temp_override(window=window, area=area, region=region):
                            if try_race_click(bpy.context, event, mx, my):
                                return {"FINISHED"}
                    except Exception:
                        traceback.print_exc()
        return {"PASS_THROUGH"}


class GR2LAB_OT_anim_browser_search_undo(bpy.types.Operator):
    """Undo in the animation strip search field."""

    bl_idname = "gr2lab.anim_browser_search_undo"
    bl_label = "Animation search undo"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _browser_on(context) and _search_active()

    def execute(self, context):
        _undo_search()
        return {"FINISHED"}


class GR2LAB_OT_anim_browser_search_redo(bpy.types.Operator):
    """Redo in the animation strip search field."""

    bl_idname = "gr2lab.anim_browser_search_redo"
    bl_label = "Animation search redo"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _browser_on(context) and _search_active()

    def execute(self, context):
        _redo_search()
        return {"FINISHED"}


class GR2LAB_OT_anim_browser_type(bpy.types.Operator):
    """Type into the GPU search field on the animation strip."""

    bl_idname = "gr2lab.anim_browser_type"
    bl_label = "Animation search type"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        if not _browser_on(context):
            return False
        if getattr(getattr(context, "area", None), "type", "") != "VIEW_3D":
            return False
        return _search_active()

    def invoke(self, context, event):
        _STATE["search_focus"] = True
        if _STATE.get("search_modal"):
            _tag()
            return {"FINISHED"}
        _search_history_begin()
        _STATE["search_modal"] = True
        context.window_manager.modal_handler_add(self)
        _tag()
        return {"RUNNING_MODAL"}

    def _stop(self) -> None:
        _STATE["search_modal"] = False
        _STATE["search_focus"] = False
        _STATE["search_sel"] = False
        _tag()

    def modal(self, context, event):
        if not _ALIVE or not _browser_on(context):
            self._stop()
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            _on_move(context, event)
            return {"PASS_THROUGH"}
        if event.type in {
            "MIDDLEMOUSE",
            "RIGHTMOUSE",
            "WHEELUPMOUSE",
            "WHEELDOWNMOUSE",
            "TRACKPADPAN",
            "TRACKPADZOOM",
        }:
            if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                handled = _on_wheel(context, event)
                if handled == {"FINISHED"}:
                    return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            found = view3d_under_pointer(context, event)
            if found is None:
                self._stop()
                return {"PASS_THROUGH"}
            area, region, mx, my = found
            window = getattr(context, "window", None)
            try:
                with context.temp_override(window=window, area=area, region=region):
                    refresh_strip_pick_data(bpy.context)
                    hit = _hit(region, mx, my)
                    if hit is not None and hit[0] == "search":
                        _place_search_caret(region, mx)
                        return {"RUNNING_MODAL"}
                    self._stop()
                    if hit is not None:
                        handle_strip_click(bpy.context, event, mx, my)
                        return {"FINISHED"}
            except Exception:
                traceback.print_exc()
            self._stop()
            return {"PASS_THROUGH"}
        if event.type in {"TIMER", "INBETWEEN_MOUSEMOVE"}:
            return {"PASS_THROUGH"}
        if _search_shortcut(event):
            return {"RUNNING_MODAL"}
        if event.type == "INSERT" and event.value == "PRESS" and event.shift:
            _paste_search()
            return {"RUNNING_MODAL"}
        if event.alt:
            return {"PASS_THROUGH"}
        if event.type in {"LEFT_ARROW", "RIGHT_ARROW"} and event.value in {"PRESS", "REPEAT"}:
            _move_search_caret(-1 if event.type == "LEFT_ARROW" else 1)
            return {"RUNNING_MODAL"}
        if event.type == "HOME" and event.value == "PRESS":
            _STATE["search_sel"] = False
            _set_search_caret(0)
            _tag()
            return {"RUNNING_MODAL"}
        if event.type == "END" and event.value == "PRESS":
            _STATE["search_sel"] = False
            _set_search_caret(len(_STATE.get("query") or ""))
            _tag()
            return {"RUNNING_MODAL"}
        if event.type in {"BACK_SPACE", "BACKSPACE"} and event.value in {"PRESS", "REPEAT", "CLICK"}:
            _backspace_search()
            return {"RUNNING_MODAL"}
        if event.type in {"DELETE", "DEL"} and event.value in {"PRESS", "REPEAT", "CLICK"}:
            _delete_search()
            return {"RUNNING_MODAL"}
        if event.type == "ESC" and event.value == "PRESS":
            _apply_search_edit("")
            _set_search_caret(0)
            self._stop()
            return {"CANCELLED"}
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._stop()
            return {"FINISHED"}
        char = _search_event_char(event)
        if char and char.isprintable() and char not in "\t":
            _push_search_char(char)
            return {"RUNNING_MODAL"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._stop()


class GR2LAB_OT_anim_edit_drop(bpy.types.Operator):
    """Duplicate the selected (or clip-matched) GR2 LAB rig for editing — no re-import."""

    bl_idname = "gr2lab.anim_edit_drop"
    bl_label = "GR2 Lab: Duplicate for edit"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(name="Clip", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _browser_on(context)

    def invoke(self, context, event):
        path = (self.filepath or "").strip() or _active_browser_clip_path() or ""
        ok, msg = browser_duplicate_edit_rig(context, path)
        if not ok:
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class GR2LAB_OT_anim_prepare(bpy.types.Operator):
    """Index cinematic folders in the unpacked data. Does not open GR2 files."""

    bl_idname = "gr2lab.anim_prepare"
    bl_label = "Prepare animation library"
    bl_options = {"REGISTER"}

    directory: bpy.props.StringProperty(
        name="Unpacked data",
        description="BG3 Multitool UnpackedData folder (contains Gustav/…/Timeline)",
        subtype="DIR_PATH",
        default="",
    )

    _timer = None

    def _start_index(self, context, root: Path):
        if _PREP["running"]:
            self.report({"WARNING"}, "Library prepare is already running")
            return {"CANCELLED"}
        with _PREP_LOCK:
            _PREP["running"] = True
            _PREP["done"] = False
            _PREP["cancel"] = False
            _PREP["error"] = ""
            _PREP["scenes"] = 0
            _PREP["phase"] = "starting"

        def cancel() -> bool:
            return bool(_PREP["cancel"])

        def run() -> None:
            try:
                data = build_index(root, cancel)
                if data is None:
                    with _PREP_LOCK:
                        _PREP["error"] = "Prepare cancelled"
                else:
                    index_path().write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
                    with _PREP_LOCK:
                        _PREP["scenes"] = len(data.get("scenes") or [])
                        _PREP["phase"] = "done"
                with _PREP_LOCK:
                    _PREP["done"] = True
            except Exception as exc:
                traceback.print_exc()
                with _PREP_LOCK:
                    _PREP["error"] = str(exc)
                    _PREP["done"] = True
            finally:
                with _PREP_LOCK:
                    _PREP["running"] = False

        threading.Thread(target=run, name="gr2lab-anim-index", daemon=True).start()
        self._timer = context.window_manager.event_timer_add(0.3, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        root = unpacked_root(context)
        if root is None:
            from . import bg3_unpacked_resolve

            hint = _prefs_unpacked_raw(context)
            if not hint and bg3_unpacked_resolve.DEFAULT_UNPACKED.is_dir():
                hint = str(bg3_unpacked_resolve.DEFAULT_UNPACKED)
            if hint:
                self.directory = hint
            self.report({"INFO"}, "Choose BG3 unpacked data folder")
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}
        return self._start_index(context, root)

    def execute(self, context):
        path = (self.directory or "").strip()
        if not path:
            self.report({"ERROR"}, "Choose your BG3 UnpackedData folder")
            return {"CANCELLED"}
        folder = Path(path)
        if not folder.is_dir():
            self.report({"ERROR"}, f"Not a folder: {path}")
            return {"CANCELLED"}
        _set_prefs_unpacked_dir(context, str(folder))
        return self._start_index(context, folder)

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            _PREP["cancel"] = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        _tag()
        if not _PREP["done"]:
            return {"PASS_THROUGH"}
        self._remove_timer(context)
        _INDEX["data"] = None
        if _PREP["error"]:
            self.report({"WARNING"}, _PREP["error"])
        else:
            self.report({"INFO"}, f"Animation library: {_PREP['scenes']} scenes")
        _tag()
        return {"FINISHED"}

    def cancel(self, context):
        _PREP["cancel"] = True
        self._remove_timer(context)

    def _remove_timer(self, context):
        timer = getattr(self, "_timer", None)
        if timer is None:
            return
        try:
            context.window_manager.event_timer_remove(timer)
        except Exception:
            pass
        self._timer = None


def register() -> None:
    global _HANDLE, _VIEW_DRAG_HANDLE, _ALIVE
    _ALIVE = True
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        _STATE["center_parent"] = bool(getattr(prefs, "center_parent", False))
    except Exception:
        pass
    bpy.utils.register_class(GR2LAB_OT_anim_edit_drop)
    bpy.utils.register_class(GR2LAB_OT_anim_prepare)
    bpy.utils.register_class(GR2LAB_OT_anim_browser_search_undo)
    bpy.utils.register_class(GR2LAB_OT_anim_browser_search_redo)
    bpy.utils.register_class(GR2LAB_OT_anim_browser_type)
    bpy.utils.register_class(GR2LAB_OT_anim_clip_drag)
    bpy.utils.register_class(GR2LAB_OT_anim_pane_resize)
    bpy.utils.register_class(GR2LAB_OT_anim_browser_modal)
    bpy.utils.register_class(GR2LAB_OT_anim_browser_pointer)
    if _HANDLE is None:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_PIXEL")
    if _VIEW_DRAG_HANDLE is None:
        _VIEW_DRAG_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_view_drop, (), "WINDOW", "POST_VIEW"
        )
    wm = bpy.context.window_manager
    # Only register on the addon keyconfig. Writing a "Window" keymap into
    # keyconfigs.user replaces Blender's user Window map and drops F2/F3, etc.
    kc_win = wm.keyconfigs.addon
    if kc_win is None:
        kc_win = wm.keyconfigs.active
    if kc_win is not None:
        km_win = kc_win.keymaps.new(name="Window", space_type="EMPTY")
        for kmi in list(km_win.keymap_items):
            if kmi.idname == "gr2lab.anim_browser_pointer":
                km_win.keymap_items.remove(kmi)
        for event_name, value, shift, alt in (
            ("LEFTMOUSE", "PRESS", False, False),
            ("LEFTMOUSE", "PRESS", True, False),
            ("LEFTMOUSE", "PRESS", False, True),
            ("WHEELUPMOUSE", "PRESS", False, False),
            ("WHEELDOWNMOUSE", "PRESS", False, False),
            ("MOUSEMOVE", "ANY", False, False),
        ):
            item = km_win.keymap_items.new(
                "gr2lab.anim_browser_pointer",
                event_name,
                value,
                shift=shift,
                alt=alt,
                head=True,
            )
            _KEYMAPS.append((km_win, item))
    kc = view3d_keyconfig(wm)
    if kc is not None:
        km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        text = km.keymap_items.new(
            "gr2lab.anim_browser_type", "TEXTINPUT", "ANY", head=True
        )
        _KEYMAPS.append((km, text))
        for ctrl, oskey in ((True, False), (False, True)):
            undo = km.keymap_items.new(
                "gr2lab.anim_browser_search_undo",
                "Z",
                "PRESS",
                ctrl=ctrl,
                oskey=oskey,
                head=True,
            )
            _KEYMAPS.append((km, undo))
            redo_y = km.keymap_items.new(
                "gr2lab.anim_browser_search_redo",
                "Y",
                "PRESS",
                ctrl=ctrl,
                oskey=oskey,
                head=True,
            )
            _KEYMAPS.append((km, redo_y))
            redo_z = km.keymap_items.new(
                "gr2lab.anim_browser_search_redo",
                "Z",
                "PRESS",
                ctrl=ctrl,
                oskey=oskey,
                shift=True,
                head=True,
            )
            _KEYMAPS.append((km, redo_z))
    _ensure_timer()
    bpy.app.timers.register(_boot_browser_modal, first_interval=0.2)
    bpy.app.timers.register(_boot_browser_modal, first_interval=1.5)


def unregister() -> None:
    global _HANDLE, _VIEW_DRAG_HANDLE, _ALIVE, _TIMER_ON
    _ALIVE = False
    _DRAG["active"] = False
    _STATE["search_modal"] = False
    _STATE["search_focus"] = False
    _PREP["cancel"] = True
    _HOVER["gen"] = int(_HOVER["gen"]) + 1
    while _KEYMAPS:
        km, item = _KEYMAPS.pop()
        try:
            km.keymap_items.remove(item)
        except Exception:
            pass
    if _HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, "WINDOW")
        except Exception:
            pass
        _HANDLE = None
    if _VIEW_DRAG_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_VIEW_DRAG_HANDLE, "WINDOW")
        except Exception:
            pass
        _VIEW_DRAG_HANDLE = None
    _HITS.clear()
    _LAYOUTS.clear()
    _TIMER_ON = False
    try:
        wm = bpy.context.window_manager
        wm["gr2lab_browser_modal_wins"] = []
    except Exception:
        pass
    for cls in (
        GR2LAB_OT_anim_pane_resize,
        GR2LAB_OT_anim_clip_drag,
        GR2LAB_OT_anim_browser_modal,
        GR2LAB_OT_anim_browser_pointer,
        GR2LAB_OT_anim_browser_type,
        GR2LAB_OT_anim_browser_search_redo,
        GR2LAB_OT_anim_browser_search_undo,
        GR2LAB_OT_anim_edit_drop,
        GR2LAB_OT_anim_prepare,
    ):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
