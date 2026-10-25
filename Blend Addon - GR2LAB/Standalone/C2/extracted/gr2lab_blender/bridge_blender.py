"""Blender armature bridge + Import/Export operators (no HTTP server)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
import uuid
from array import array
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bpy
from bl_operators.presets import AddPresetBase
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Quaternion, Vector

# ---------------------------------------------------------------------------
# Local in-process codec (no always-on server)
# ---------------------------------------------------------------------------


class Gr2LabApiError(RuntimeError):
    def __init__(self, message: str, *, payload: Optional[dict] = None):
        super().__init__(message)
        self.payload = payload or {}


class Gr2LabProgress:
    """Import uses the viewport bar only. header=True is the window progress bar."""

    def __init__(self, context, *, header: bool = True):
        self._context = context
        self._wm = context.window_manager
        self._header = bool(header)
        self._wm_bar = False
        self._active = False
        self._last_ui = 0.0

    def begin(self, maximum: int = 100) -> None:
        self._active = True
        self._last_ui = 0.0
        self._wm_bar = False
        if self._header:
            self._wm.progress_begin(0, max(1, int(maximum)))
            self._wm_bar = True

    def step(self, value: int, label: str = "", *, force_ui: bool = False) -> None:
        if not self._active:
            return
        if self._wm_bar:
            self._wm.progress_update(int(value))
        if not self._header:
            now = time.perf_counter()
            if not force_ui and (now - self._last_ui) < 0.05:
                return
            self._last_ui = now
            from .import_progress_ui import update as overlay_update

            overlay_update(
                max(0.0, min(1.0, float(value) / 100.0)),
                label or "Importing…",
            )
            _redraw_windows(self._context)
            return
        now = time.perf_counter()
        if not force_ui and (now - self._last_ui) < 0.25:
            return
        self._last_ui = now
        if label:
            try:
                self._wm.status_text_set(f"GR2 LAB: {label}")
            except (AttributeError, TypeError):
                pass
        _redraw_windows(self._context)

    def end(self) -> None:
        if self._wm_bar:
            self._wm.progress_end()
            self._wm_bar = False
        self._active = False
        if self._header:
            try:
                self._wm.status_text_set(None)
            except (AttributeError, TypeError):
                try:
                    self._wm.status_text_set("")
                except (AttributeError, TypeError):
                    pass
        _redraw_windows(self._context)


def _redraw_windows(context, *, swap: bool = False) -> None:
    # tag_redraw only — never redraw_timer inside a blocking operator (causes freezes).
    wm = context.window_manager
    for window in wm.windows:
        screen = window.screen
        if not screen:
            continue
        for area in screen.areas:
            area.tag_redraw()

def _apply_layout_from_prefs(context=None) -> None:
    """Sync codec cache/Body_Base roots from addon preferences."""
    from .local_pipeline import apply_prefs

    prefs = None
    try:
        if context is not None:
            prefs = context.preferences.addons[__package__].preferences
        else:
            prefs = _addon_prefs()
    except Exception:
        prefs = None
    cache = getattr(prefs, "cache_root", "") if prefs else ""
    body_dir = getattr(prefs, "body_base_dir", "") if prefs else ""
    apply_prefs(cache_root=cache or "", body_base=body_dir or "")


def fetch_body_bases(
    base_url: str = "",
    *,
    source: Optional[str] = None,
) -> dict:
    _apply_layout_from_prefs()
    from .local_pipeline import list_body_bases_local

    return list_body_bases_local(source=source)


# Blender EnumProperty callbacks run every redraw. Keep one tuple list alive:
# RNA holds pointers into these strings, and rebuilding/GC makes the menu lag.
_IMPORT_BB_ITEMS: List[Tuple[str, str, str]] = [
    ("NONE", "(no Body_Base folder)", "Set Body_Base folder in addon preferences")
]
_IMPORT_BB_KEY: Tuple[Any, ...] = ()
_NPANEL_BB_ITEMS: List[Tuple[str, str, str]] = [
    ("NONE", "(no Body_Base)", "Add Body_Base/*.bind_pose.json")
]
_NPANEL_BB_KEY: Tuple[str, ...] = ()


def _import_body_base_enum_items(self, context) -> List[Tuple[str, str, str]]:
    global _IMPORT_BB_ITEMS, _IMPORT_BB_KEY
    src = ""
    fp = getattr(self, "filepath", "") or ""
    if fp:
        src = Path(fp).name
    try:
        _apply_layout_from_prefs(context)
        from bind_pose import list_body_bases, match_body_base_for_source  # type: ignore

        names = tuple(
            str(b.get("name") or "")
            for b in list_body_bases()
            if b.get("name")
        )
        matched = (match_body_base_for_source(src) if src else None) or ""
        matched = str(matched).strip()
    except Exception:
        names = ()
        matched = ""
    key = (src, matched, names)
    if key == _IMPORT_BB_KEY:
        return _IMPORT_BB_ITEMS
    items: List[Tuple[str, str, str]] = []
    for name in names:
        stem = Path(name).stem
        if matched and name == matched:
            items.append((name, f"{stem} (Auto)", name))
        else:
            items.append((name, stem, name))
    if not items:
        items.append(
            ("NONE", "(no Body_Base folder)", "Set Body_Base folder in addon preferences")
        )
    _IMPORT_BB_ITEMS = items
    _IMPORT_BB_KEY = key
    return _IMPORT_BB_ITEMS


def _partner_body_base_enum_items(self, context) -> List[Tuple[str, str, str]]:
    """Body list for the second character. Auto matches that GR2, not the first."""
    items = [("AUTO", "Auto", "Match the second character's GR2")]
    for ident, label, desc in _import_body_base_enum_items(self, context):
        if ident in {"AUTO", "NONE"}:
            if ident == "NONE":
                items.append((ident, label, desc))
            continue
        items.append((ident, label.replace(" (Auto)", ""), desc))
    return items


_BODY_BASE_SYNCING = False


def _update_import_body_base(self, context) -> None:
    """A dropdown edit is an explicit body choice. File sync is not.

    A race-picker label (``DGB_F AkELkA``) must survive assigning ``body_base``.
    """
    if _BODY_BASE_SYNCING:
        return
    current = str(getattr(self, "body_choice", "") or "").strip()
    if current and current != "user":
        return
    try:
        self.body_choice = "user"
    except Exception:
        pass


def _same_import_path(a: str, b: str) -> bool:
    """FILE_PATH RNA rewrites slashes. The pin and the operator path are one file."""
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return Path(a).resolve().as_posix().casefold() == Path(b).resolve().as_posix().casefold()
    except OSError:
        return a.replace("/", "\\").casefold() == b.replace("/", "\\").casefold()


def _sync_import_body_base(op, context) -> None:
    global _BODY_BASE_SYNCING
    fp = getattr(op, "filepath", "") or ""
    synced = getattr(op, "bb_synced_for", "") or ""
    if not fp or _same_import_path(synced, fp):
        return
    _BODY_BASE_SYNCING = True
    try:
        try:
            op.bb_synced_for = fp
        except Exception:
            pass
        try:
            op.bb_hint = ""
        except Exception:
            pass
        # A body picked for the previous clip is not a choice for this file.
        try:
            op.body_choice = ""
        except Exception:
            pass
        try:
            data = fetch_body_bases(source=Path(fp).name)
            matched = (data.get("matched") or "").strip()
            if matched:
                op.body_base = matched
        except Exception:
            pass
    finally:
        _BODY_BASE_SYNCING = False


def _body_file_for_race_label(label: str) -> str:
    """``DGB_F AkELkA`` → ``DGB_F_Base.GR2``. Empty when the label is not a race."""
    token = (label or "").strip().split()[0] if label else ""
    if not token or token.lower() == "user":
        return ""
    try:
        from bind_pose import list_body_bases  # type: ignore

        for item in list_body_bases() or []:
            name = item.get("name") if isinstance(item, dict) else str(item or "")
            if not name:
                continue
            stem = Path(name).stem
            if not stem.lower().endswith("_base"):
                continue
            race = stem[: -len("_base")]
            if race.lower() == token.lower():
                return Path(name).name
    except Exception:
        return ""
    return ""


def _take_swap_body_pin(arm) -> Optional[Dict[str, Any]]:
    """Race-row click stored on the rig. Read once so a later import does not reuse it."""
    if arm is None or not arm.get("gr2lab_swap_body"):
        return None
    pin = {
        "base": str(arm.get("gr2lab_swap_body") or ""),
        "choice": str(arm.get("gr2lab_swap_choice") or ""),
        "akelka": bool(arm.get("gr2lab_swap_akelka")),
    }
    for key in ("gr2lab_swap_body", "gr2lab_swap_choice", "gr2lab_swap_akelka"):
        try:
            del arm[key]
        except Exception:
            pass
    if not pin["base"]:
        return None
    return pin


def _body_base_family(name: str) -> str:
    stem = Path((name or "").strip()).stem
    if stem.lower().endswith("_base"):
        stem = stem[: -len("_base")]
    return stem


def _resolve_body_base_for_import(
    *,
    choice: str,
    bb_hint: str,
    prefs_base: str,
    source_gr2: str,
    explicit_choice: bool = True,
) -> Optional[str]:
    """Pick Body_Base for import.

    Priority:
      1) explicit choice (race row or dropdown) — never a leftover hint
      2) bb_hint \"Auto: Name.GR2\"
      3) filename match (family base when that file exists)
      4) prefs, only when it is the same body family or the clip has no family
    """
    hint = (bb_hint or "").strip()
    c = (choice or "").strip()
    # A clicked race (DGB_M AkELkA) is the body to build. A leftover
    # Auto hint from the previous swap must not replace it with DWR_M.
    if explicit_choice and c and c not in {"AUTO", "NONE"}:
        return Path(c).name
    if hint.lower().startswith("auto:"):
        auto_name = hint.split(":", 1)[1].strip()
        if auto_name.lower().endswith(".gr2"):
            return Path(auto_name).name

    detail: Dict[str, Any] = {}
    matched: Optional[str] = None
    try:
        from rig_catalog import match_body_base_detail  # type: ignore

        raw_detail = match_body_base_detail(Path(source_gr2).name) or {}
        if isinstance(raw_detail, dict):
            detail = raw_detail
        raw = detail.get("matched")
        matched = str(raw).strip() if raw else None
    except Exception:
        try:
            from bind_pose import match_body_base_for_source  # type: ignore

            raw = match_body_base_for_source(Path(source_gr2).name)
            matched = str(raw).strip() if raw else None
        except Exception:
            matched = None

    src_fam = str(detail.get("rig_family") or "")
    src_kind = str(detail.get("rig_kind") or "")

    def _other_family(name: str) -> bool:
        if src_kind != "body" or not src_fam:
            return False
        fam = _body_base_family(name)
        return bool(fam) and fam.lower() != src_fam.lower()

    prefs_name = Path((prefs_base or "").strip()).name
    c = (choice or "").strip()
    if c and c not in {"AUTO", "NONE"}:
        chosen = Path(c).name
        # A click on DGB_F (race row or the dropdown) is the body to build.
        # The saved default often is that same DGB_F, and a borrowed clip is
        # still named DGB_M. Treating "same family as the default" as stale
        # threw the click away and built DGB_M AkELkA.
        if explicit_choice:
            return chosen
        stale_prefs = bool(prefs_name) and (
            _body_base_family(chosen).lower() == _body_base_family(prefs_name).lower()
        )
        if matched and stale_prefs and chosen.lower() != Path(matched).name.lower():
            return matched
        if matched and _other_family(chosen):
            return matched
        if not matched and _other_family(chosen) and stale_prefs:
            return None
        return chosen

    if matched:
        return matched
    prefs_full = (prefs_base or "").strip()
    if prefs_full and _other_family(prefs_full):
        return None
    return prefs_full or None


def _resolve_body_base_choice(
    choice: str,
    prefs_base: str = "",
    *,
    source_gr2: str = "",
) -> Optional[str]:
    """Compat wrapper."""
    return _resolve_body_base_for_import(
        choice=choice,
        bb_hint="",
        prefs_base=prefs_base,
        source_gr2=source_gr2,
    )


def import_gr2_pipeline(
    base_url: str,
    gr2_path: str,
    *,
    body_base: Optional[str] = None,
    progress: Optional["Gr2LabProgress"] = None,
) -> tuple[dict, dict]:
    """Decode + densify in-process. base_url ignored (compat)."""
    _apply_layout_from_prefs()
    from .local_pipeline import import_gr2_pipeline_local

    return import_gr2_pipeline_local(gr2_path, body_base=body_base, progress=progress)


def pack_encode_gr2(
    base_url: str,
    run_id: str,
    doc: dict,
    dest_gr2_path: str,
    *,
    source_gr2: Optional[str] = None,
    progress: Optional["Gr2LabProgress"] = None,
    timer=None,
) -> dict:
    _apply_layout_from_prefs()
    from .local_pipeline import dump_still_exists, pack_encode_local

    if not dump_still_exists(run_id):
        raise Gr2LabApiError(
            f"Decode dump “{run_id}” not found in local cache. "
            "Re-import the GR2 via File → Import → GR2, then export again."
        )
    return pack_encode_local(
        run_id,
        doc,
        dest_gr2_path,
        source_gr2=source_gr2,
        progress=progress,
        timer=timer,
    )


def export_gr2_pipeline(
    base_url: str,
    run_id: str,
    doc: dict,
    dest_gr2_path: str,
    *,
    source_gr2: Optional[str] = None,
    gr2lab_path: Optional[str] = None,
    progress: Optional["Gr2LabProgress"] = None,
) -> dict:
    if gr2lab_path and Path(gr2lab_path).is_file():
        doc = json.loads(Path(gr2lab_path).read_text(encoding="utf-8"))
    doc = dict(doc)
    doc["run_id"] = run_id
    return pack_encode_gr2(
        base_url,
        run_id,
        doc,
        dest_gr2_path,
        source_gr2=source_gr2,
        progress=progress,
    )


def import_scene_partner_pipeline(*_a, **_k):
    raise Gr2LabApiError(
        "Scene partner import needs the optional site tools (not in standalone addon yet)."
    )



# ---------------------------------------------------------------------------
# GR2LAB bridge (armature + F-curves)
# ---------------------------------------------------------------------------

FORMAT_ID = "gr2lab"
FPS_DEFAULT = 30.0
TIME_EPS = 1e-4  # snap F-curve seconds back onto template knot times
NOOP_POS_EPS = 2e-4
NOOP_QUAT_DOT_EPS = 0.999
# Whole-character placement — keep template unless user-affected + real delta.
PROTECTED_ROOT_BONES = frozenset({"Dummy_Root", "Root_M"})

# Blender Z-up Dummy_Root rest is Rx(+90°). Body_Base templates sometimes store that
# as an orientation track; applying it as Granny anim tips the character 90° on site.
_RX90_QUAT = (0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
_RX_NEG90_QUAT = (-0.7071067811865475, 0.0, 0.0, 0.7071067811865475)


def _quat_dot_abs(a, b) -> float:
    return abs(
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )


def _is_blender_zup_root_ori(q) -> bool:
    """True when quat is ~Rx(±90°) — Blender Z-up Dummy_Root rest, not Granny anim."""
    if not q or len(q) < 4:
        return False
    return (
        _quat_dot_abs(q, _RX90_QUAT) >= 0.995
        or _quat_dot_abs(q, _RX_NEG90_QUAT) >= 0.995
    )


def _identity_ori_channel() -> Dict[str, Any]:
    return {
        "edit_mode": "none",
        "editable": False,
        "times": [0.0],
        "values": [[0.0, 0.0, 0.0, 1.0]],
        "note": "identity orientation",
    }


def _sanitize_dummy_root_ori(ori: Dict[str, Any]) -> Dict[str, Any]:
    """Drop false Blender Z-up Rx(90°) Dummy_Root anim — keep Granny identity."""
    vals = ori.get("values") or []
    if not vals:
        return _identity_ori_channel()
    if all(_is_blender_zup_root_ori(v) for v in vals):
        return _identity_ori_channel()
    return ori


def _uniform_time_grid(duration: float, fps: float) -> List[float]:
    dur = max(0.0, float(duration))
    rate = max(1.0, float(fps))
    n = max(2, int(dur * rate) + 1)
    if dur <= 1e-12:
        return [0.0]
    return [i * dur / (n - 1) for i in range(n)]


def _lerp_sample_n(
    knots: Sequence[float], controls: Sequence[Sequence[float]], t: float
) -> List[float]:
    if not knots or not controls:
        return []
    if len(knots) == 1 or t <= knots[0]:
        return list(controls[0])
    if t >= knots[-1]:
        return list(controls[-1])
    for i in range(len(knots) - 1):
        if knots[i] <= t <= knots[i + 1]:
            span = knots[i + 1] - knots[i]
            u = 0.0 if span <= 1e-12 else (t - knots[i]) / span
            a, b = controls[i], controls[i + 1]
            return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(len(a))]
    return list(controls[0])


def _ensure_hemi_quats(controls: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = [list(controls[0])]
    for q in controls[1:]:
        prev = out[-1]
        dot = sum(float(prev[i]) * float(q[i]) for i in range(4))
        if dot < 0:
            out.append([-float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])])
        else:
            out.append(list(q))
    return out


def _pad_knots(knots: Sequence[float], degree: int) -> List[float]:
    t0 = float(knots[0])
    t1 = float(knots[-1])
    return [t0] + [float(k) for k in knots] + [t1] * int(degree)


def _bspline_span(u_vec: Sequence[float], degree: int, n_ctrl: int, t: float) -> int:
    if t >= u_vec[n_ctrl]:
        return n_ctrl - 1
    if t <= u_vec[degree]:
        return degree
    low, high = degree, n_ctrl
    mid = (low + high) // 2
    while t < u_vec[mid] or t >= u_vec[mid + 1]:
        if t < u_vec[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _sample_bspline_py(
    degree: int,
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    t: float,
    *,
    normalize: bool = False,
) -> List[float]:
    """Pure-Python Granny-style Deg2+ B-spline (no DLL). Deg≤1 → lerp."""
    n = len(knots)
    if n == 0 or not controls:
        return []
    dim = len(controls[0])
    degree = int(degree)
    if degree <= 1 or n < degree + 1:
        return _lerp_sample_n(knots, controls, t)
    t0, t1 = float(knots[0]), float(knots[-1])
    if float(t) <= t0:
        result = list(controls[0])
    elif float(t) >= t1:
        result = list(controls[-1])
    else:
        u_vec = _pad_knots(knots, degree)
        i = _bspline_span(u_vec, degree, n, float(t))
        d = [list(controls[i - degree + j]) for j in range(degree + 1)]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                left = u_vec[i - degree + j]
                right = u_vec[i + 1 + j - r]
                denom = right - left
                alpha = 0.0 if abs(denom) < 1e-20 else (float(t) - left) / denom
                d[j] = [
                    (1.0 - alpha) * d[j - 1][c] + alpha * d[j][c] for c in range(dim)
                ]
        result = d[degree]
    if normalize and dim == 4:
        nrm = math.sqrt(sum(x * x for x in result)) or 1.0
        result = [x / nrm for x in result]
    return result


def _sample_bspline_quat_py(
    degree: int,
    knots: Sequence[float],
    quats: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    return _sample_bspline_py(
        degree, knots, _ensure_hemi_quats(quats), t, normalize=True
    )


def _ensure_doc_bspline_dense(doc: Dict[str, Any], fps: float) -> Dict[str, Any]:
    """Densify Deg≥2 channels onto an fps grid so Blender LINEAR looks game-smooth.

    Runs client-side even if the .gr2lab was not Download-for-Blender densified.
    Already-dense Deg1 channels are left alone. Root_M ori is marked display_densified.
    """
    duration = float(doc.get("duration") or 0.0)
    grid = _uniform_time_grid(duration, fps)
    fps_i = int(round(float(fps)))
    densified = 0
    for slot in doc.get("tracks") or []:
        name = slot.get("name") or ""
        for ch_key, is_quat in (("position", False), ("orientation", True)):
            ch = slot.get(ch_key) or {}
            mode = ch.get("edit_mode") or "none"
            if mode in ("none", "constant"):
                continue
            note = str(ch.get("note") or "")
            if "bspline_densified" in note or "k16_display_densified" in note:
                continue
            ts = [float(x) for x in (ch.get("times") or [])]
            vs = [list(map(float, v)) for v in (ch.get("values") or [])]
            if len(ts) < 2 or len(vs) != len(ts):
                continue
            # Already ~fps dense Deg1 — skip
            deg = int(ch.get("degree") or 0)
            if deg <= 1 and len(ts) >= max(8, int(duration * float(fps) * 0.85)):
                continue
            if name == "Dummy_Root" and ch_key == "orientation":
                continue
            # Save/24: Deg≥2 k16/float/quat → B-spline Deg1 float (not lerp).
            if deg < 2:
                deg = 2 if mode in ("quat_k16", "k16_offsets") else 1
            if deg < 2:
                continue
            samples = (
                [_sample_bspline_quat_py(deg, ts, vs, t) for t in grid]
                if is_quat
                else [_sample_bspline_py(deg, ts, vs, t) for t in grid]
            )
            display = name in ("Dummy_Root", "Root_M") or mode == "k16_offsets"
            slot[ch_key] = {
                "edit_mode": "float_controls",
                "editable": True,
                "format": 1,
                "capacity": len(grid),
                "times": list(grid),
                "values": samples,
                "degree": 1,
                "note": f"bspline_densified@{fps_i}",
                "source_interp": "bspline_dense",
                "display_densified": True if display else ch.get("display_densified"),
            }
            densified += 1
    doc["source_interp"] = "bspline_dense"
    doc["densify_fps"] = float(fps)
    doc["densified_channels"] = int(doc.get("densified_channels") or 0) + densified
    doc["blender_import_densified"] = densified
    return {"densified": densified, "grid": len(grid), "fps": float(fps)}


PROP_DOC = "gr2lab_json"
PROP_RUN = "gr2lab_run_id"
PROP_TEMPLATE_PATH = "gr2lab_template_path"
WM_LAST_TEMPLATE = "gr2lab_last_template"
WM_LAST_PACK_RUN = "gr2lab_last_pack_run"
WM_LAST_SOURCE_GR2 = "gr2lab_last_source_gr2"
WM_LAST_IMPORT_OPTS = "gr2lab_last_import_opts"


def _matched_body_base_name(source_gr2: str) -> str:
    name = Path(source_gr2).name
    if not name:
        return ""
    try:
        data = fetch_body_bases(source=name)
        return str((data.get("matched") or "")).strip()
    except Exception:
        return ""


def remember_import_op_options(op) -> None:
    """Remember rig/mesh choices from the drop dialog for strip imports."""
    try:
        wm = bpy.context.window_manager
    except Exception:
        return
    opts = {}
    for key in _IMPORT_PRESET_KEYS:
        if not hasattr(op, key):
            continue
        val = getattr(op, key)
        if isinstance(val, bool):
            opts[key] = bool(val)
        elif isinstance(val, int):
            opts[key] = int(val)
        else:
            opts[key] = str(val or "")
    try:
        wm[WM_LAST_IMPORT_OPTS] = opts
    except Exception:
        pass


def gr2_import_prefs_from_invoke(prefs) -> dict:
    """Same add-rig / whole-scene rules as IMPORT_OT_gr2_full.invoke."""
    two = bool(getattr(prefs, "second_character", False))
    one = bool(getattr(prefs, "add_rig", True))
    if two and one:
        two = False
    add_rig = one and not two
    second_character = two
    return {
        "add_rig": add_rig,
        "second_character": second_character,
        "scene_empties": bool(second_character),
        "center_parent": bool(getattr(prefs, "center_parent", False)),
        "partner_body_base": str(getattr(prefs, "partner_body_base", "AUTO") or "AUTO"),
        "partner_meshes": str(getattr(prefs, "partner_meshes", "BODY_CLOTH") or "BODY_CLOTH"),
        "partner_akelka": bool(getattr(prefs, "partner_akelka", True)),
        "partner_tail": bool(getattr(prefs, "partner_tail", True)),
        "partner_rig_setup": str(getattr(prefs, "partner_rig_setup", "ADVANCED") or "ADVANCED"),
        "partner_rig_shapes": bool(getattr(prefs, "partner_rig_shapes", True)),
        "partner_rig_ik_limbs": bool(getattr(prefs, "partner_rig_ik_limbs", True)),
        "partner_rig_ik_head": bool(getattr(prefs, "partner_rig_ik_head", True)),
    }


def gr2_import_exec_kwargs(context, filepath: str) -> dict:
    """Keyword args for bpy.ops.import_scene.gr2_full from prefs + last dialog + file match."""
    path = str(filepath or "").strip()
    prefs = context.preferences.addons[__package__].preferences
    kwargs = dict(gr2_import_prefs_from_invoke(prefs))
    kwargs.update(
        {
            "filepath": path,
            "import_apply": "NEW_RIG",
            "import_meshes": "BODY_CLOTH",
            "import_akelka": True,
            "import_clear_meshes": True,
            "import_tail": True,
            "rig_setup": "ADVANCED",
            "rig_shapes": True,
            "rig_ik_limbs": True,
            "rig_ik_head": True,
        }
    )
    try:
        wm = context.window_manager
        cached = wm.get(WM_LAST_IMPORT_OPTS)
        if isinstance(cached, dict):
            for key in _IMPORT_PRESET_KEYS:
                if key in cached and key not in {
                    "body_base",
                    "add_rig",
                    "second_character",
                    "scene_empties",
                    "center_parent",
                }:
                    kwargs[key] = cached[key]
    except Exception:
        pass
    kwargs.update(gr2_import_prefs_from_invoke(prefs))
    matched = _matched_body_base_name(path)
    if matched:
        kwargs["body_base"] = matched
        kwargs["bb_hint"] = f"Auto: {matched}"
    else:
        fallback = str(getattr(prefs, "body_base", "") or "").strip()
        if fallback:
            try:
                from rig_catalog import classify_rig  # type: ignore

                info = classify_rig(Path(path).name)
                fam = str(info.get("rig_family") or "")
                kind = str(info.get("rig_kind") or "")
                fb_fam = _body_base_family(fallback)
                if kind == "body" and fam and fb_fam and fb_fam.lower() != fam.lower():
                    fallback = ""
            except Exception:
                pass
        if fallback:
            kwargs["body_base"] = fallback
    return kwargs


def exec_gr2_import_quick(
    context,
    filepath: str,
    *,
    whole_scene: bool = False,
    center_parent: Optional[bool] = None,
) -> None:
    """Run a strip / programmatic import with correct body base and remembered options."""
    kwargs = gr2_import_exec_kwargs(context, filepath)
    # Strip / browser drops always add a fresh rig — never swap onto the selection.
    kwargs["import_apply"] = "NEW_RIG"
    kwargs["target_armature"] = ""
    kwargs["add_rig"] = True
    if whole_scene:
        kwargs["second_character"] = True
        kwargs["scene_empties"] = True
    else:
        kwargs["second_character"] = False
        kwargs["scene_empties"] = False
    if center_parent is not None:
        kwargs["center_parent"] = bool(center_parent)
    bpy.ops.import_scene.gr2_full("EXEC_DEFAULT", **kwargs)


def _quat_gr_to_bl(q: Sequence[float]) -> Quaternion:
    """Granny/Three [x,y,z,w] в†’ Blender Quaternion(w,x,y,z)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return Quaternion((w, x, y, z))


def _quat_bl_to_gr(q: Quaternion) -> List[float]:
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _trs_matrix(t: Sequence[float], r: Sequence[float], s: Sequence[float]) -> Matrix:
    loc = Vector((float(t[0]), float(t[1]), float(t[2])))
    rot = _quat_gr_to_bl(r)
    scl = Vector((float(s[0]), float(s[1]), float(s[2])))
    return Matrix.LocRotScale(loc, rot, scl)


def _sample_vec_at(
    ts: Sequence[float],
    vs: Sequence[Sequence[float]],
    t: float,
    n: int,
    default: Sequence[float],
) -> List[float]:
    """Linear sample of an n-D channel. Binary search — not an O(n) scan."""
    if not ts or not vs:
        return list(default)
    if len(ts) == 1 or t <= ts[0]:
        return list(vs[0])
    if t >= ts[-1]:
        return list(vs[-1])
    i = bisect_right(ts, t) - 1
    if i < 0:
        return list(vs[0])
    if i >= len(ts) - 1:
        return list(vs[-1])
    t0 = ts[i]
    t1 = ts[i + 1]
    span = t1 - t0
    u = 0.0 if span <= 1e-12 else (t - t0) / span
    a, b = vs[i], vs[i + 1]
    return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(n)]


def _sample_quat_gr_at(
    ts: Sequence[float],
    vs: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    """Granny xyzw slerp. Binary search — not an O(n) scan."""
    if not ts or not vs:
        return [0.0, 0.0, 0.0, 1.0]
    if len(ts) == 1 or t <= ts[0]:
        return list(vs[0])
    if t >= ts[-1]:
        return list(vs[-1])
    i = bisect_right(ts, t) - 1
    if i < 0:
        return list(vs[0])
    if i >= len(ts) - 1:
        return list(vs[-1])
    t0 = ts[i]
    t1 = ts[i + 1]
    span = t1 - t0
    u = 0.0 if span <= 1e-12 else (t - t0) / span
    if u <= 0.0:
        return list(vs[i])
    if u >= 1.0:
        return list(vs[i + 1])
    qa = _quat_gr_to_bl(vs[i])
    qb = _quat_gr_to_bl(vs[i + 1])
    qb.make_compatible(qa)
    return _quat_bl_to_gr(qa.slerp(qb, u))


def _quat_wxyz_from_mats(rot):
    """Vectorized Shepperd. ``rot`` is (n, 3, 3). Returns (n, 4) wxyz."""
    import numpy as np

    n = int(rot.shape[0])
    t = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    qw = np.empty(n)
    qx = np.empty(n)
    qy = np.empty(n)
    qz = np.empty(n)
    s = np.empty(n)
    m0 = t > 0.0
    if np.any(m0):
        s[m0] = np.sqrt(t[m0] + 1.0) * 2.0
        qw[m0] = 0.25 * s[m0]
        qx[m0] = (rot[m0, 2, 1] - rot[m0, 1, 2]) / s[m0]
        qy[m0] = (rot[m0, 0, 2] - rot[m0, 2, 0]) / s[m0]
        qz[m0] = (rot[m0, 1, 0] - rot[m0, 0, 1]) / s[m0]
    m1 = (~m0) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    if np.any(m1):
        s[m1] = np.sqrt(1.0 + rot[m1, 0, 0] - rot[m1, 1, 1] - rot[m1, 2, 2]) * 2.0
        qw[m1] = (rot[m1, 2, 1] - rot[m1, 1, 2]) / s[m1]
        qx[m1] = 0.25 * s[m1]
        qy[m1] = (rot[m1, 0, 1] + rot[m1, 1, 0]) / s[m1]
        qz[m1] = (rot[m1, 0, 2] + rot[m1, 2, 0]) / s[m1]
    m2 = (~m0) & (~m1) & (rot[:, 1, 1] > rot[:, 2, 2])
    if np.any(m2):
        s[m2] = np.sqrt(1.0 + rot[m2, 1, 1] - rot[m2, 0, 0] - rot[m2, 2, 2]) * 2.0
        qw[m2] = (rot[m2, 0, 2] - rot[m2, 2, 0]) / s[m2]
        qx[m2] = (rot[m2, 0, 1] + rot[m2, 1, 0]) / s[m2]
        qy[m2] = 0.25 * s[m2]
        qz[m2] = (rot[m2, 1, 2] + rot[m2, 2, 1]) / s[m2]
    m3 = (~m0) & (~m1) & (~m2)
    if np.any(m3):
        s[m3] = np.sqrt(1.0 + rot[m3, 2, 2] - rot[m3, 0, 0] - rot[m3, 1, 1]) * 2.0
        qw[m3] = (rot[m3, 1, 0] - rot[m3, 0, 1]) / s[m3]
        qx[m3] = (rot[m3, 0, 2] + rot[m3, 2, 0]) / s[m3]
        qy[m3] = (rot[m3, 1, 2] + rot[m3, 2, 1]) / s[m3]
        qz[m3] = 0.25 * s[m3]
    q = np.stack((qw, qx, qy, qz), axis=1)
    nrm = np.linalg.norm(q, axis=1)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    return q / nrm[:, None]


def _basis_batch(rinv, pts, quats_xyzw, scales):
    """Rest-inverse × key TRS for many keys. Returns loc (n,3), quat wxyz, scale."""
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    quats_xyzw = np.asarray(quats_xyzw, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    x, y, z, w = quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2], quats_xyzw[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    n = int(pts.shape[0])
    mats = np.zeros((n, 4, 4), dtype=np.float64)
    sx, sy, sz = scales[:, 0], scales[:, 1], scales[:, 2]
    mats[:, 0, 0] = r00 * sx
    mats[:, 1, 0] = r10 * sx
    mats[:, 2, 0] = r20 * sx
    mats[:, 0, 1] = r01 * sy
    mats[:, 1, 1] = r11 * sy
    mats[:, 2, 1] = r21 * sy
    mats[:, 0, 2] = r02 * sz
    mats[:, 1, 2] = r12 * sz
    mats[:, 2, 2] = r22 * sz
    mats[:, 0, 3] = pts[:, 0]
    mats[:, 1, 3] = pts[:, 1]
    mats[:, 2, 3] = pts[:, 2]
    mats[:, 3, 3] = 1.0
    posed = np.einsum("ij,njk->nik", np.asarray(rinv, dtype=np.float64), mats)
    loc = posed[:, :3, 3]
    cols = posed[:, :3, :3]
    sc = np.linalg.norm(cols, axis=1)
    sc = np.where(sc < 1e-12, 1.0, sc)
    rot = cols / sc[:, None, :]
    det = np.linalg.det(rot)
    neg = det < 0.0
    if np.any(neg):
        sc = sc.copy()
        sc[neg, 2] *= -1.0
        rot = rot.copy()
        rot[neg, :, 2] *= -1.0
    return loc, _quat_wxyz_from_mats(rot), sc


def _compat_bl_quat_lists(ws: List[float], xs: List[float], ys: List[float], zs: List[float]) -> None:
    """Flip consecutive Blender wxyz keys onto the short arc (in place)."""
    n = len(ws)
    for i in range(1, n):
        if (
            ws[i] * ws[i - 1]
            + xs[i] * xs[i - 1]
            + ys[i] * ys[i - 1]
            + zs[i] * zs[i - 1]
            < 0.0
        ):
            ws[i] = -ws[i]
            xs[i] = -xs[i]
            ys[i] = -ys[i]
            zs[i] = -zs[i]


def _mat_to_trs(m: Matrix) -> Tuple[List[float], List[float], List[float]]:
    loc, rot, scl = m.decompose()
    return (
        [float(loc.x), float(loc.y), float(loc.z)],
        _quat_bl_to_gr(rot),
        [float(scl.x), float(scl.y), float(scl.z)],
    )


def _stabilize_local_cache_quats(
    local_cache: Dict[float, Dict[str, Tuple[List[float], List[float], List[float]]]],
) -> None:
    """
    Matrix.decompose() picks q or -q arbitrarily each frame. Consecutive sign flips
    look like shake after import (Granny/site lerp without a short-path fix on every path).
    Walk each bone's exported series and keep the short hemisphere.
    """
    if not local_cache:
        return
    frames = sorted(local_cache.keys())
    names = set()
    for fr in frames:
        names.update(local_cache[fr].keys())
    for name in names:
        prev: Optional[Quaternion] = None
        for fr in frames:
            bucket = local_cache.get(fr) or {}
            if name not in bucket:
                continue
            loc, quat_xyzw, scl = bucket[name]
            # Granny xyzw в†’ Blender wxyz
            cur = Quaternion(
                (
                    float(quat_xyzw[3]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                )
            )
            cur.normalize()
            if prev is not None:
                cur.make_compatible(prev)
            prev = cur
            bucket[name] = (loc, _quat_bl_to_gr(cur), scl)


def _make_quat_series_compatible(values: List[List[float]]) -> None:
    """In-place Granny xyzw series в†’ short-arc consecutive keys."""
    prev: Optional[Quaternion] = None
    for i, q in enumerate(values):
        if not q or len(q) < 4:
            continue
        cur = Quaternion((float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        cur.normalize()
        if prev is not None:
            cur.make_compatible(prev)
        prev = cur
        values[i] = _quat_bl_to_gr(cur)


def _load_doc(path: str) -> Dict[str, Any]:
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".gr2"):
        raise ValueError(
            f"{p.name} is a game .GR2 file, not a .gr2lab bridge.\n"
            "Use File > Import > GR2 (via GR2 Lab) for .GR2 files."
        )
    raw = p.read_bytes()
    doc = _parse_gr2lab_bytes(raw)
    if not isinstance(doc.get("skeleton"), list) or not isinstance(doc.get("tracks"), list):
        raise ValueError(f"Invalid GR2LAB file (missing skeleton/tracks): {Path(path).name}")
    return doc


def _bone_order(skeleton: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parents before children."""
    by_name = {b["name"]: b for b in skeleton if b.get("name")}
    remaining = set(by_name.keys())
    ordered: List[Dict[str, Any]] = []
    while remaining:
        progress = False
        for name in list(remaining):
            b = by_name[name]
            p = b.get("parent_name")
            if not p or p not in by_name or p not in remaining:
                ordered.append(b)
                remaining.remove(name)
                progress = True
        if not progress:
            # cycle / missing parent вЂ” append rest
            for name in sorted(remaining):
                ordered.append(by_name[name])
            break
    return ordered


def _rest_worlds(skeleton: List[Dict[str, Any]]) -> Dict[str, Matrix]:
    worlds: Dict[str, Matrix] = {}
    for b in _bone_order(skeleton):
        name = b["name"]
        local = _trs_matrix(
            b.get("rest_translation") or [0, 0, 0],
            b.get("rest_rotation") or [0, 0, 0, 1],
            b.get("rest_scale") or [1, 1, 1],
        )
        p = b.get("parent_name")
        worlds[name] = (worlds[p] @ local) if p in worlds else local
    return worlds


def _bind_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    bind = doc.get("bind") or {}
    bones = bind.get("bones") if isinstance(bind, dict) else None
    if not bones:
        return {}
    return {b["name"]: b for b in bones if b.get("name")}


def _skeleton_with_bind_parents(
    skeleton: List[Dict[str, Any]], bind_by_name: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Override parent_name from bind (Body_Base / LSLib hierarchy) when present."""
    if not bind_by_name:
        return list(skeleton)
    out = []
    for b in skeleton:
        nb = dict(b)
        bb = bind_by_name.get(b.get("name") or "")
        if bb is not None and "parent_name" in bb:
            nb["parent_name"] = bb.get("parent_name")
        out.append(nb)
    return out


def _bind_local_for_bone(
    b: Dict[str, Any], bind_by_name: Dict[str, Dict[str, Any]]
) -> Matrix:
    """Edit-mode / pose basis local matrix: bind when present, else clip rest_*."""
    bb = bind_by_name.get(b["name"])
    if bb:
        return _trs_matrix(
            bb.get("bind_translation") or [0, 0, 0],
            bb.get("bind_rotation") or [0, 0, 0, 1],
            bb.get("bind_scale") or [1, 1, 1],
        )
    return _trs_matrix(
        b.get("rest_translation") or [0, 0, 0],
        b.get("rest_rotation") or [0, 0, 0, 1],
        b.get("rest_scale") or [1, 1, 1],
    )


def _bind_worlds(
    skeleton: List[Dict[str, Any]], bind_by_name: Dict[str, Dict[str, Any]]
) -> Dict[str, Matrix]:
    """FK worlds using skeleton parent_name (already bind-overridden when available)."""
    worlds: Dict[str, Matrix] = {}
    for b in _bone_order(skeleton):
        name = b["name"]
        local = _bind_local_for_bone(b, bind_by_name)
        p = b.get("parent_name")
        worlds[name] = (worlds[p] @ local) if p in worlds else local
    return worlds


def _track_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["name"]: t for t in doc.get("tracks") or [] if t.get("name")}


def _channel_times(ch: Dict[str, Any], mode: str) -> List[float]:
    if mode in ("constant", "none"):
        return [0.0]
    ts = [float(t) for t in (ch.get("times") or [0.0])]
    return ts if ts else [0.0]


def _times_match(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b) or not a:
        return False
    if abs(float(a[0]) - float(b[0])) > 1e-8 or abs(float(a[-1]) - float(b[-1])) > 1e-8:
        return False
    if len(a) <= 2:
        return True
    mid = len(a) // 2
    return abs(float(a[mid]) - float(b[mid])) <= 1e-8


def _column_or_none(times_list, own: bool, src_ts, src_vs):
    if own and len(src_vs) == len(times_list):
        return src_vs
    if src_ts and src_vs and _times_match(times_list, src_ts) and len(src_vs) == len(times_list):
        return src_vs
    return None


def _bone_channel_payload(b: Dict[str, Any], tr: Dict[str, Any], rmat: Matrix, fps: float) -> Dict[str, Any]:
    """Pose curves for one bone. Numpy does the long grids off the UI thread."""
    import numpy as np

    rinv = np.array(rmat.inverted(), dtype=np.float64)
    pos = tr.get("position") or {}
    ori = tr.get("orientation") or {}
    scl = tr.get("scale") or {}
    pos_mode = pos.get("edit_mode") or "none"
    ori_mode = ori.get("edit_mode") or "none"
    scl_mode = scl.get("edit_mode") or "none"
    pos_ts = [float(x) for x in (pos.get("times") or [])]
    pos_vs = pos.get("values") or []
    ori_ts = [float(x) for x in (ori.get("times") or [])]
    ori_vs = ori.get("values") or []
    scl_ts = [float(x) for x in (scl.get("times") or [])]
    scl_vs = scl.get("values") or []
    rest_t = b.get("rest_translation") or [0, 0, 0]
    rest_s = b.get("rest_scale") or [1, 1, 1]
    loc_times = _channel_times(pos, pos_mode if pos_mode not in ("none",) else "constant")
    rot_times = _channel_times(ori, ori_mode if ori_mode not in ("none",) else "constant")
    loc_own = len(loc_times) == len(pos_ts) == len(pos_vs) and len(pos_ts) > 0
    rot_own = len(rot_times) == len(ori_ts) == len(ori_vs) and len(ori_ts) > 0
    scl_editable = bool(scl.get("editable") and scl_mode != "none")
    scl_own_times = _channel_times(scl, scl_mode) if scl_editable else None
    scl_own = bool(
        scl_own_times is not None
        and len(scl_own_times) == len(scl_ts) == len(scl_vs)
        and len(scl_ts) > 0
    )

    def one_math(t, *, own_pos=False, own_ori=False, own_scl=False, i=0):
        pt = pos_vs[i] if own_pos else _sample_vec_at(pos_ts, pos_vs, t, 3, rest_t)
        qt = ori_vs[i] if own_ori else _sample_quat_gr_at(ori_ts, ori_vs, t)
        st = scl_vs[i] if own_scl else _sample_vec_at(scl_ts, scl_vs, t, 3, rest_s)
        loc, rot, sc = (rmat.inverted() @ _trs_matrix(pt, qt, st)).decompose()
        return loc, rot, sc

    def batch_or_none(times_list, *, own_pos=False, own_ori=False, own_scl=False):
        pts = _column_or_none(times_list, own_pos, pos_ts, pos_vs)
        qts = _column_or_none(times_list, own_ori, ori_ts, ori_vs)
        if scl_own or (scl_ts and scl_vs):
            sts = _column_or_none(times_list, own_scl, scl_ts, scl_vs)
        else:
            sts = [list(rest_s) for _ in times_list] if times_list else None
        if pts is None or qts is None or sts is None or not times_list:
            return None
        loc, quat, sc = _basis_batch(rinv, pts, qts, sts)
        frames = (1.0 + np.asarray(times_list, dtype=np.float64) * float(fps)).tolist()
        return frames, loc, quat, sc

    def comps_of(kind, frames, loc, quat, sc):
        if kind == "location":
            cols = (loc[:, 0], loc[:, 1], loc[:, 2])
        elif kind == "rotation_quaternion":
            cols = (quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])
        else:
            cols = (sc[:, 0], sc[:, 1], sc[:, 2])
        out = [c.tolist() for c in cols]
        if kind == "rotation_quaternion":
            _compat_bl_quat_lists(out[0], out[1], out[2], out[3])
        return frames, out

    channels = []
    loc_batch = batch_or_none(loc_times, own_pos=loc_own)
    rot_batch = batch_or_none(rot_times, own_ori=rot_own)
    if loc_batch is None:
        frames, comps = [], None
        for i, t in enumerate(loc_times):
            loc, rot, sc = one_math(t, own_pos=loc_own, i=i)
            vec = (loc.x, loc.y, loc.z)
            if comps is None:
                comps = [[] for _ in vec]
            frames.append(_sec_to_frame(t, fps))
            for ci, v in enumerate(vec):
                comps[ci].append(float(v))
        if frames and comps:
            channels.append(("location", frames, comps))
    else:
        frames, loc, quat, sc = loc_batch
        fr, cols = comps_of("location", frames, loc, quat, sc)
        channels.append(("location", fr, cols))

    if rot_batch is None:
        frames, comps = [], None
        for i, t in enumerate(rot_times):
            loc, rot, sc = one_math(t, own_ori=rot_own, i=i)
            vec = (rot.w, rot.x, rot.y, rot.z)
            if comps is None:
                comps = [[] for _ in vec]
            frames.append(_sec_to_frame(t, fps))
            for ci, v in enumerate(vec):
                comps[ci].append(float(v))
        if comps and len(comps) == 4:
            _compat_bl_quat_lists(comps[0], comps[1], comps[2], comps[3])
        if frames and comps:
            channels.append(("rotation_quaternion", frames, comps))
    else:
        frames, loc, quat, sc = rot_batch
        fr, cols = comps_of("rotation_quaternion", frames, loc, quat, sc)
        channels.append(("rotation_quaternion", fr, cols))

    loc0, rot0, sc0 = one_math(0.0)
    if scl_editable:
        scl_times = scl_own_times or _channel_times(scl, scl_mode)
        scl_batch = batch_or_none(scl_times, own_scl=scl_own)
        if scl_batch is None:
            frames, comps = [], None
            for i, t in enumerate(scl_times):
                loc, rot, sc = one_math(t, own_scl=scl_own, i=i)
                vec = (sc.x, sc.y, sc.z)
                if comps is None:
                    comps = [[] for _ in vec]
                frames.append(_sec_to_frame(t, fps))
                for ci, v in enumerate(vec):
                    comps[ci].append(float(v))
            if frames and comps:
                channels.append(("scale", frames, comps))
        else:
            frames, loc, quat, sc = scl_batch
            fr, cols = comps_of("scale", frames, loc, quat, sc)
            channels.append(("scale", fr, cols))
    else:
        channels.append(
            (
                "scale",
                [_sec_to_frame(0.0, fps)],
                [[float(sc0.x)], [float(sc0.y)], [float(sc0.z)]],
            )
        )
    return {
        "pos_mode": pos_mode,
        "ori_mode": ori_mode,
        "scl_mode": scl_mode,
        "k16": pos_mode == "k16_offsets",
        "pose": (
            (float(loc0.x), float(loc0.y), float(loc0.z)),
            (float(rot0.w), float(rot0.x), float(rot0.y), float(rot0.z)),
            (float(sc0.x), float(sc0.y), float(sc0.z)),
        ),
        "channels": channels,
    }


def prepare_pose_pack(doc: Dict[str, Any], fps: float) -> Dict[str, Any]:
    """Curve arrays for every bone. Safe off Blender's main thread."""
    fps = float(fps or doc.get("densify_fps") or FPS_DEFAULT)
    _ensure_doc_bspline_dense(doc, fps)
    skeleton = list(doc.get("skeleton") or [])
    bind_by_name = _bind_map(doc)
    skeleton = _skeleton_with_bind_parents(skeleton, bind_by_name)
    try:
        from bind_pose import expand_bind_map  # type: ignore

        bind_meta = doc.get("bind") if isinstance(doc.get("bind"), dict) else {}
        bind_by_name = expand_bind_map(
            bind_by_name,
            source_gr2=str(doc.get("source_gr2") or ""),
            bind_source_gr2=str((bind_meta or {}).get("source_gr2") or ""),
        )
    except Exception:
        traceback.print_exc()
    tracks = _track_map(doc)
    pack: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for b in skeleton:
        name = b.get("name")
        if not name:
            continue
        pack[name] = _bone_channel_payload(
            b, tracks.get(name) or {}, _bind_local_for_bone(b, bind_by_name), fps
        )
        time.sleep(0)
    print(
        f"[GR2 LAB] pose pack {len(pack)} bones in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return pack


def _apply_packed_bone(pb, fcurves, name: str, packed: Dict[str, Any]) -> None:
    loc, rot, sc = packed["pose"]
    pb.location = loc
    pb.rotation_quaternion = rot
    pb.scale = sc
    for prop, frames, comps in packed.get("channels") or []:
        _bulk_fcurve_channel(fcurves, name, prop, frames, comps)


def _sec_to_frame(t: float, fps: float) -> float:
    return 1.0 + float(t) * float(fps)


def _frame_to_sec(frame: float, fps: float) -> float:
    return max(0.0, (float(frame) - 1.0) / float(fps))


def _nearest_sorted(
    sorted_vals: Sequence[float], t: float
) -> Tuple[Optional[float], Optional[float]]:
    """Nearest value in a sorted list and |delta|. None if empty."""
    n = len(sorted_vals)
    if n == 0:
        return None, None
    i = bisect_left(sorted_vals, t)
    best: Optional[float] = None
    best_d: Optional[float] = None
    for j in (i - 1, i):
        if 0 <= j < n:
            d = abs(float(sorted_vals[j]) - float(t))
            if best_d is None or d < best_d:
                best_d = d
                best = float(sorted_vals[j])
    return best, best_d


def _snap_times_to_template(
    fc_times: Sequence[float],
    orig_times: Sequence[float],
    *,
    eps: float = TIME_EPS,
) -> List[float]:
    """Map F-curve seconds onto template knots; keep only truly new keys.

    Avoids densifying float/quat grids when sec-frame float noise would otherwise
    make ``set(orig) | set(fc)`` grow the curve on a no-op round-trip.
    """
    orig = [float(t) for t in orig_times] if orig_times else [0.0]
    orig_sorted = sorted(orig)
    if not fc_times:
        return list(orig) if orig else [0.0]
    out: List[float] = []
    seen = set()
    for raw in fc_times:
        t = float(raw)
        nearest, best = _nearest_sorted(orig_sorted, t)
        use = float(nearest) if nearest is not None and best is not None and best <= eps else t
        key = round(use, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(use)
    return sorted(out) or [0.0]


def _collapse_times_eps(
    times: Sequence[float],
    *,
    prefer: Optional[Sequence[float]] = None,
    eps: float = TIME_EPS,
) -> List[float]:
    """Merge times within eps; prefer frame-grid values when present in a cluster.

    Exact ``set()`` unions of template JSON floats and ``(frame-1)/fps`` create
    near-zero-Δt pairs with inconsistent samples → site/game finger flicker.
    """
    ordered = sorted({float(t) for t in times})
    if not ordered:
        return [0.0]
    pref = sorted({float(t) for t in (prefer or [])})
    out: List[float] = []
    cluster: List[float] = [ordered[0]]

    def flush(cl: List[float]) -> None:
        chosen = cl[-1]
        hit = None
        for c in cl:
            nearest, dist = _nearest_sorted(pref, c)
            if nearest is not None and dist is not None and dist <= eps:
                hit = nearest
                break
        if hit is not None:
            chosen = hit
        if out and abs(chosen - out[-1]) <= eps:
            nearest, dist = _nearest_sorted(pref, chosen)
            if nearest is not None and dist is not None and dist <= eps:
                out[-1] = chosen
            return
        out.append(chosen)

    for t in ordered[1:]:
        if t - cluster[-1] <= eps:
            cluster.append(t)
        else:
            flush(cluster)
            cluster = [t]
    flush(cluster)
    return out or [0.0]


def _lerp_vec3(
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    """Linear sample a vec3 series (matches site linear display)."""
    ts = [float(x) for x in times] if times else [0.0]
    vs = [list(v) for v in values] if values else []
    if not vs:
        return [0.0, 0.0, 0.0]
    if len(ts) == 1 or t <= ts[0]:
        return [float(vs[0][j]) for j in range(3)]
    if t >= ts[-1]:
        return [float(vs[-1][j]) for j in range(3)]
    i = bisect_right(ts, t) - 1
    i = max(0, min(i, len(ts) - 2))
    span = ts[i + 1] - ts[i]
    u = 0.0 if span <= 1e-12 else (t - ts[i]) / span
    a, b = vs[i], vs[i + 1]
    n = min(3, len(a), len(b))
    return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(n)]


def _k16_pos_matches_template(
    samples: Dict[float, List[float]],
    template_times: Sequence[float],
    template_vals: Sequence[Sequence[float]],
) -> bool:
    """True when exported Granny locals still match dequantized k16 template motion.

    Untouched display keys from import в†’ bias-only export (keep k16_offsets).
    Real multi-key edits that leave the template curve в†’ upgrade to float on site.
    """
    if not template_vals:
        return True
    if not samples:
        return True
    tmpl_t = [float(x) for x in template_times] if template_times else [0.0]
    for t, v in samples.items():
        expected = _lerp_vec3(tmpl_t, template_vals, float(t))
        if not _v3_close(v, expected, NOOP_POS_EPS):
            return False
    return True


def _root_samples_differ_from_template(
    samples: Dict[float, List[float]],
    template_ch: Dict[str, Any],
    *,
    is_quat: bool,
) -> bool:
    """True when sampled motion leaves the template beyond noop eps."""
    tmpl_t = [float(x) for x in (template_ch.get("times") or [0.0])]
    tmpl_v = template_ch.get("values") or []
    if not samples or not tmpl_v:
        return False
    if is_quat:
        for t, v in samples.items():
            expected = _sample_quat_channel(
                {"times": tmpl_t, "values": tmpl_v}, float(t)
            )
            if expected is None or not _quat_close(v, expected, NOOP_QUAT_DOT_EPS):
                return True
        return False
    for t, v in samples.items():
        expected = _lerp_vec3(tmpl_t, tmpl_v, float(t))
        if not _v3_close(v, expected, NOOP_POS_EPS):
            return True
    return False


def _channel_is_display_densified(ch: Dict[str, Any]) -> bool:
    note = str((ch or {}).get("note") or "")
    return (
        "bspline_densified" in note
        or "k16_display_densified" in note
        or bool((ch or {}).get("display_densified"))
    )


def _strip_root_display_densify(
    ch: Dict[str, Any], *, kind: str = "locomotion"
) -> Dict[str, Any]:
    """Mark a rewritten root channel as a real edit, not import display densify.

    Dense rewrites are always Degree=1 so BG3/LSLib linear-eval matches Blender.
    """
    out = dict(ch)
    out.pop("display_densified", None)
    note = str(out.get("note") or "")
    if (
        "bspline_densified" in note
        or "k16_display_densified" in note
        or not note
        or note.startswith("root_")
    ):
        out["note"] = f"root_{kind}_edit"
    out["degree"] = 1
    if len(out.get("times") or []) > 1:
        out["edit_mode"] = "float_controls"
        out["editable"] = True
    return out


def _allow_protected_root_rewrite(
    bone_affected: bool,
    samples: Dict[float, List[float]],
    template_ch: Dict[str, Any],
    *,
    is_quat: bool,
) -> bool:
    """Rewrite protected roots only when sampled motion actually leaves the template.

    Same motion (including bake densify of vanilla Root_M ori) → keep the original
    sparse curve. Different motion (NLA COMBINE / keyed loc+rot) → rewrite even
    on a dense 30fps grid. Constraint-affected is not required: NLA COMBINE is
    not a constraint, and Root_M Copy* constraints are often influence 0.
    """
    del bone_affected
    return _root_samples_differ_from_template(samples, template_ch, is_quat=is_quat)


def _is_mass_grid_densify_times(
    template_times: Sequence[float],
    sample_times: Sequence[float],
) -> bool:
    """True when F-curve times are a much denser grid than the template (frame bake)."""
    tmpl = [float(t) for t in template_times] if template_times else []
    samp = [float(t) for t in sample_times] if sample_times else []
    if not tmpl or not samp or len(samp) <= len(tmpl):
        return False
    cap = max(8, int(0.05 * len(tmpl) + 0.5))
    tmpl_sorted = sorted(tmpl)
    far = 0
    for t in samp:
        _nearest, best = _nearest_sorted(tmpl_sorted, t)
        if best is not None and best > TIME_EPS:
            far += 1
    return far > cap


def _drop_on_curve_extra_times(
    times: Sequence[float],
    samples: Dict[float, List[float]],
    template_times: Sequence[float],
    template_vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
) -> List[float]:
    """Keep template knots; drop extras whose sample matches the template curve.

    Stops Spine*_M-style on-curve densify keys from a Blender no-op export.
    """
    if not times:
        return [0.0]
    tmpl_t = [float(t) for t in template_times] if template_times else [0.0]
    tmpl_v = [list(v) for v in template_vals] if template_vals else []
    close = (
        (lambda a, b: _quat_close(a, b, NOOP_QUAT_DOT_EPS))
        if is_quat
        else (lambda a, b: _v3_close(a, b, NOOP_POS_EPS))
    )

    def on_template(t: float) -> bool:
        return any(abs(float(t) - o) <= TIME_EPS for o in tmpl_t)

    def sample_tmpl(t: float) -> List[float]:
        if not tmpl_v:
            return [0.0, 0.0, 0.0, 1.0] if is_quat else [0.0, 0.0, 0.0]
        ts = tmpl_t
        vs = tmpl_v
        if len(ts) == 1 or t <= ts[0]:
            return list(vs[0])
        if t >= ts[-1]:
            return list(vs[-1])
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i + 1]:
                span = ts[i + 1] - ts[i]
                u = 0.0 if span <= 1e-12 else (t - ts[i]) / span
                if is_quat:
                    qa = _quat_gr_to_bl(vs[i])
                    qb = _quat_gr_to_bl(vs[i + 1])
                    qb.make_compatible(qa)
                    return _quat_bl_to_gr(qa.slerp(qb, u))
                a, b = vs[i], vs[i + 1]
                n = min(len(a), len(b), 3)
                return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(n)]
        return list(vs[-1])

    out: List[float] = []
    for t in times:
        tf = float(t)
        if on_template(tf):
            out.append(tf)
            continue
        got = samples.get(tf)
        if got is None:
            continue
        if close(got, sample_tmpl(tf)):
            continue
        out.append(tf)
    return sorted(out) or list(tmpl_t[:1]) or [0.0]


def _nla_pushdown_current_action(arm_obj) -> bool:
    """Move the assigned action onto a new NLA track. Returns True if pushed."""
    ad = getattr(arm_obj, "animation_data", None)
    if not ad or not ad.action:
        return False
    try:
        ad.use_tweak_mode = False
    except Exception:
        pass
    action = ad.action
    slot = getattr(ad, "action_slot", None)
    start = 1
    try:
        start = int(action.frame_range[0])
    except Exception:
        start = 1
    if start < 1:
        start = 1
    tracks = ad.nla_tracks
    try:
        track = tracks.new(prev=None)
    except TypeError:
        track = tracks.new()
    track.name = (action.name or "Action")[:63]
    try:
        strip = track.strips.new((action.name or "Action")[:63], start, action)
    except Exception:
        traceback.print_exc()
        return False
    if slot is not None and hasattr(strip, "action_slot"):
        try:
            strip.action_slot = slot
        except Exception:
            pass
    ad.action = None
    if hasattr(ad, "action_slot"):
        try:
            ad.action_slot = None
        except Exception:
            pass
    return True


_NLA_STRIP_COPY_ATTRS = (
    "use_sync_length",
    "scale",
    "repeat",
    "action_frame_start",
    "action_frame_end",
    "frame_start",
    "frame_end",
    "blend_type",
    "mute",
    "influence",
    "extrapolation",
    "blend_in",
    "blend_out",
    "use_auto_blend",
    "use_animated_influence",
    "use_reverse",
    "use_animated_time",
    "use_animated_time_cyclic",
)


def _copy_nla_strip(dst_track, src_strip) -> bool:
    """Duplicate one strip onto ``dst_track``. Shares the same Action (no bake)."""
    action = getattr(src_strip, "action", None)
    if action is None:
        return False
    name = (src_strip.name or action.name or "Strip")[:63]
    raw_start = getattr(src_strip, "frame_start", 1.0)
    start = 1.0 if raw_start is None else float(raw_start)
    start_i = int(round(start))
    try:
        dst = dst_track.strips.new(name, start_i, action)
    except TypeError:
        dst = dst_track.strips.new(name, start, action)
    except Exception:
        traceback.print_exc()
        return False
    src_slot = getattr(src_strip, "action_slot", None)
    if src_slot is not None and hasattr(dst, "action_slot"):
        try:
            dst.action_slot = src_slot
        except Exception:
            pass
    for attr in _NLA_STRIP_COPY_ATTRS:
        if not hasattr(src_strip, attr) or not hasattr(dst, attr):
            continue
        try:
            setattr(dst, attr, getattr(src_strip, attr))
        except Exception:
            pass
    return True


def _copy_nla_stack(src_arm, dst_arm, keep_action=None) -> int:
    """Copy NLA tracks/strips from ``src_arm`` onto ``dst_arm``.

    Strip timing stays. The Action on those strips becomes the new rig's
    clip, so a race swap does not keep playing the previous body.
    """
    if src_arm is None or dst_arm is None or src_arm == dst_arm:
        return 0
    src_ad = getattr(src_arm, "animation_data", None)
    if not src_ad or not src_ad.nla_tracks:
        return 0
    if not dst_arm.animation_data:
        dst_arm.animation_data_create()
    dst_ad = dst_arm.animation_data
    if keep_action is None:
        keep_action = dst_ad.action
    keep_slot = getattr(dst_ad, "action_slot", None)
    if keep_action is not None and keep_slot is None:
        slots = list(getattr(keep_action, "slots", None) or [])
        if slots:
            keep_slot = slots[0]
    try:
        dst_ad.use_tweak_mode = False
    except Exception:
        pass
    dst_ad.use_nla = True
    copied = 0
    prev = None
    for src_tr in list(src_ad.nla_tracks):
        try:
            dst_tr = dst_ad.nla_tracks.new(prev=prev)
        except TypeError:
            dst_tr = dst_ad.nla_tracks.new()
        dst_tr.name = src_tr.name or "NlaTrack"
        dst_tr.mute = bool(src_tr.mute)
        for src_s in list(src_tr.strips):
            if _copy_nla_strip(dst_tr, src_s):
                copied += 1
        if hasattr(src_tr, "lock") and hasattr(dst_tr, "lock"):
            try:
                dst_tr.lock = bool(src_tr.lock)
            except Exception:
                pass
        prev = dst_tr
    # The copied strips still reference the previous rig's Action. Leaving
    # them there means a race swap (GNO_F over a DGB_M) plays the old clip
    # as soon as the new Action is unlinked, and the body drops into that pose.
    retargeted = 0
    if keep_action is not None:
        for dst_tr in dst_ad.nla_tracks:
            for strip in dst_tr.strips:
                if getattr(strip, "action", None) == keep_action:
                    continue
                try:
                    strip.action = keep_action
                except Exception:
                    continue
                if keep_slot is not None and hasattr(strip, "action_slot"):
                    try:
                        strip.action_slot = keep_slot
                    except Exception:
                        pass
                retargeted += 1
        if retargeted:
            print(
                f"[GR2 LAB] swap NLA retargeted {retargeted} strips "
                f"onto {keep_action.name!r}"
            )
    from .scene_stage_empties import (
        _has_scene_segment_nla,
        _release_active_action_over_scene_nla,
    )

    if _has_scene_segment_nla(dst_arm):
        # Timeline strips already point at the new clip. Leaving the Action
        # assigned makes Blender ignore those strips.
        _release_active_action_over_scene_nla(dst_arm)
    else:
        dst_ad.action = keep_action
        if keep_slot is not None and hasattr(dst_ad, "action_slot"):
            try:
                dst_ad.action_slot = keep_slot
            except Exception:
                pass
    return copied


def _nla_add_action_strip(arm_obj, action, *, fps: float, duration: float) -> None:
    """Park ``action`` on a new NLA track and clear the assigned action slot."""
    if arm_obj is None or action is None:
        return
    if not arm_obj.animation_data:
        arm_obj.animation_data_create()
    ad = arm_obj.animation_data
    try:
        ad.use_tweak_mode = False
    except Exception:
        pass
    ad.use_nla = True
    slot = getattr(ad, "action_slot", None)
    start = 1
    try:
        start = int(action.frame_range[0])
    except Exception:
        start = 1
    if start < 1:
        start = 1
    _ = (fps, duration)
    tracks = ad.nla_tracks
    try:
        track = tracks.new(prev=None)
    except TypeError:
        track = tracks.new()
    track.name = (action.name or "GR2LAB")[:63]
    strip = track.strips.new((action.name or "GR2LAB")[:63], start, action)
    if slot is not None and hasattr(strip, "action_slot"):
        try:
            strip.action_slot = slot
        except Exception:
            pass
    ad.action = None
    if hasattr(ad, "action_slot"):
        try:
            ad.action_slot = None
        except Exception:
            pass


def _stamp_imported_action(
    action, *, run_id: str, fps: float, duration: float
) -> None:
    if action is None:
        return
    try:
        action["gr2lab_run_id"] = run_id or ""
        action["gr2lab_fps"] = float(fps)
        action["gr2lab_duration"] = float(duration)
    except Exception:
        pass


def _import_gr2lab_steps(
    context,
    doc: Dict[str, Any],
    *,
    fps: Optional[float] = None,
    y_up_display: bool = True,
    template_path: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
    template_doc: Optional[Dict[str, Any]] = None,
    timer=None,
    animation: bool = True,
    rig_display: bool = True,
    rig_parts: Optional[Dict[str, bool]] = None,
    target_arm=None,
    anim_apply: str = "NEW_RIG",
    import_meshes: str = "NONE",
    import_akelka: bool = False,
    import_tail: bool = False,
    import_clear_meshes: bool = False,
    mesh_base: Optional[str] = None,
    import_fast_head_ik: bool = False,
    hide_viewport_build: bool = False,
    pose_pack: Optional[Dict[str, Any]] = None,
    link_collection=None,
) -> bpy.types.Object:
    """Build armature + action from a GR2LAB document dict.

    ``animation=False`` builds bind-pose bones only (no F-curves, scene fps
    unchanged) — used by the Body Base N-panel.

    ``rig_display=False`` skips custom shapes, IK, and HUD (plain imported bones).
    ``rig_parts`` selects shapes / hands IK / legs IK / head IK when display is on.

    ``anim_apply`` ``REPLACE`` / ``ADD`` / ``NLA`` writes the clip onto
    ``target_arm`` (no new armature, no IK/shapes). ``NEW_RIG`` creates one.
    ``SWAP_RIG`` creates a new armature like ``NEW_RIG``, then retargets the
    selected similar rig's meshes / parents / constraints onto it.

    ``import_meshes`` ``NONE`` / ``BODY`` / ``BODY_CLOTH`` appends kit meshes
    onto a newly created or swapped armature. ``import_akelka`` swaps Dragonborn
    onto the AkELkA DGB kits.     ``import_tail`` adds a sibling tail armature
    (Tiefling / Dragonborn) after the kit attach. Swap always drops the
    previous character meshes and attaches the kit for the chosen body base
    (Lips / IK widgets / bone-parented props stay). Replace / Add
    action / NLA skip both.

    Does not leave the scene playhead on frame 1: the current frame is restored
    to whatever it was before this import.
    """
    prev_frame = int(getattr(context.scene, "frame_current", 1) or 1)
    fps = float(fps if fps is not None else doc.get("densify_fps") or FPS_DEFAULT)
    densify_info: Dict[str, Any] = {"densified": 0}
    if animation:
        from timing import maybe_timer

        with maybe_timer(timer, "densify_ensure"):
            densify_info = _ensure_doc_bspline_dense(doc, fps)
        if densify_info.get("densified"):
            print(
                f"GR2LAB: import densified {densify_info['densified']} channels "
                f"@ {densify_info['fps']:.0f}fps ({densify_info['grid']} keys)"
            )
    skeleton = doc.get("skeleton") or []
    tracks = _track_map(doc)
    duration = float(doc.get("duration") or 0.0)
    run_id = doc.get("run_id") or ""
    root_name = doc.get("root") or (skeleton[0]["name"] if skeleton else "Root")
    bind_by_name = _bind_map(doc)
    if not bind_by_name:
        print(
            "GR2LAB warning: no bind section — Edit mode uses clip t=0 rest "
            "(character imports need Body_Base/*.bind_pose.json from GR2 Lab 2.1+)"
        )

    # Hierarchy: prefer Body_Base / LSLib parents from bind (fixes flying IK)
    skeleton = _skeleton_with_bind_parents(skeleton, bind_by_name)
    display_skeleton = skeleton
    grafted_names: List[str] = []
    try:
        from bind_pose import expand_bind_map, graft_leaf_end_bones  # type: ignore

        bind_meta = doc.get("bind") if isinstance(doc.get("bind"), dict) else {}
        bind_by_name = expand_bind_map(
            bind_by_name,
            source_gr2=str(doc.get("source_gr2") or ""),
            bind_source_gr2=str((bind_meta or {}).get("source_gr2") or ""),
        )
        display_skeleton, grafted_names = graft_leaf_end_bones(skeleton, bind_by_name)
        if grafted_names:
            print(
                f"[GR2 LAB] grafted {len(grafted_names)} missing end bones: "
                + ", ".join(grafted_names)
            )
    except Exception:
        traceback.print_exc()
        display_skeleton = skeleton
        grafted_names = []

    anim_apply = str(anim_apply or "NEW_RIG").strip().upper() or "NEW_RIG"
    onto_existing = anim_apply in {"REPLACE", "ADD", "NLA"}
    swap_from = None
    restore_mode = None
    if anim_apply == "SWAP_RIG":
        if target_arm is None or getattr(target_arm, "type", None) != "ARMATURE":
            raise Gr2LabApiError(
                "Select an armature to swap, or choose Add rig"
            )
        swap_from = target_arm
    if onto_existing:
        if target_arm is None or getattr(target_arm, "type", None) != "ARMATURE":
            raise Gr2LabApiError(
                "Select an armature to apply animation, or choose Add rig"
            )
        arm_obj = target_arm
        restore_mode = getattr(arm_obj, "mode", "OBJECT")
        if restore_mode == "EDIT":
            _mode_set_safe(context, "OBJECT", active=arm_obj)
            restore_mode = "OBJECT"
        context.view_layer.objects.active = arm_obj
        arm_obj.select_set(True)
        print(
            f"[GR2 LAB] apply {anim_apply} animation onto {arm_obj.name!r} "
            f"(no new rig, no IK/shapes)"
        )
        try:
            from .rig_display import ensure_missing_end_bones

            added = ensure_missing_end_bones(
                arm_obj, context, mode_set=_mode_set_safe
            )
            if added:
                print(
                    f"[GR2 LAB] grafted {len(added)} missing end bones onto "
                    f"{arm_obj.name!r}: " + ", ".join(added)
                )
        except Exception:
            traceback.print_exc()
    else:
        # Leave any prior Edit/Pose mode without ops that need a VIEW_3D poll.
        prev = context.view_layer.objects.active
        if prev is not None and getattr(prev, "mode", "OBJECT") != "OBJECT":
            _mode_set_safe(context, "OBJECT", active=prev)
        _deselect_all_objects(context)

        arm_data = bpy.data.armatures.new(f"GR2LAB_{run_id or 'rig'}")
        arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
        dest = link_collection if link_collection is not None else context.collection
        dest.objects.link(arm_obj)
        edit_vl = context.view_layer
        if getattr(dest, "name", "") == "GR2LAB_preload_hold":
            build = context.scene.view_layers.get("GR2LAB_preload_build")
            if build is not None:
                edit_vl = build
        edit_vl.objects.active = arm_obj
        try:
            arm_obj.select_set(True)
        except RuntimeError:
            pass

        # Optional display-only Y-up → Z-up (rotate armature object, not keys)
        if y_up_display:
            arm_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

        worlds = _bind_worlds(display_skeleton, bind_by_name)

        _mode_set_safe(context, "EDIT", active=arm_obj)
        edit_bones = arm_data.edit_bones
        created = {}
        bone_slice = time.perf_counter()
        for b in _bone_order(display_skeleton):
            name = b["name"]
            eb = edit_bones.new(name)
            # Unit-length bone along local +Y in rest (Blender bone axis)
            m = worlds.get(name, Matrix.Identity(4))
            head = m.to_translation()
            # Tail: small offset along bone's local Y (Blender convention)
            axis = (m.to_3x3() @ Vector((0.0, 0.05, 0.0)))
            if axis.length < 1e-8:
                axis = Vector((0.0, 0.05, 0.0))
            eb.head = head
            eb.tail = head + axis
            # Align roll via matrix
            eb.matrix = m
            created[name] = eb
            if time.perf_counter() - bone_slice >= 0.012:
                yield from _yield_viewport_hidden(
                    arm_obj, hide_viewport_build, 36, "Building bones…"
                )
                bone_slice = time.perf_counter()

        for b in display_skeleton:
            name = b.get("name")
            p = b.get("parent_name")
            if name in created and p in created:
                created[name].parent = created[p]
                created[name].use_connect = False

        from . import scene_stage_empties as _stage

        if _stage._graft_diag_socket_edit(arm_obj):
            grafted_names.append(_stage._DIAG_CAMERA_SOCKET)

        _mode_set_safe(context, "POSE", active=arm_obj)
        for name in grafted_names:
            pb = arm_obj.pose.bones.get(name)
            if pb is not None:
                pb["gr2lab_grafted"] = True
                pb.rotation_mode = "QUATERNION"

        # Stamp the pre-densify sparse pack (vanilla formats). Densified tracks are
        # F-curves only — export Root_M protect needs original quat_k16.
        stamp_src = template_doc if isinstance(template_doc, dict) else doc
        doc_store = dict(stamp_src)
        doc_store["skeleton"] = skeleton
        if doc_store.get("bind") and isinstance(doc_store["bind"], dict):
            pass
        from timing import maybe_timer

        with maybe_timer(timer, "stamp_id_props"):
            arm_obj[PROP_DOC] = json.dumps(
                doc_store, separators=(",", ":"), ensure_ascii=False
            )
            arm_obj[PROP_RUN] = run_id
            try:
                arm_obj[PROP_IMPORT_SNAPSHOT] = json.dumps(
                    {
                        "tracks": doc_store.get("tracks") or [],
                        "duration": doc_store.get("duration"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            except Exception:
                pass
            arm_obj["gr2lab_fps"] = float(fps)
            arm_obj["gr2lab_y_up_display"] = bool(y_up_display)
            arm_obj["gr2lab_has_bind"] = bool(bind_by_name)
            arm_obj["gr2lab_source_interp"] = str(doc.get("source_interp") or "")
            if doc.get("densify_fps") is not None:
                arm_obj["gr2lab_densify_fps"] = float(doc.get("densify_fps"))
            arm_obj["gr2lab_import_densified"] = int(
                densify_info.get("densified") or doc.get("densified_channels") or 0
            )
            if template_path:
                arm_obj[PROP_TEMPLATE_PATH] = str(Path(template_path).resolve())
                remember_template_path(template_path)

    # Granny keys are parent-relative to bind rest (Y-up Dummy_Root = identity).
    # DGB / LSLib Dummy_Root rest is Rx(+90°) in the armature. Converting against
    # that rest writes pose Rx(-90°) and the clip lies on its side. Always use
    # bind (same as Add rig): pose identity × DGB Dummy rest = standing.
    rest_local: Dict[str, Matrix] = {}
    for b in skeleton:
        name = b["name"]
        rest_local[name] = _bind_local_for_bone(b, bind_by_name)
        pb = arm_obj.pose.bones.get(name)
        if pb is not None:
            pb.rotation_mode = "QUATERNION"

    _t_write = 0.0
    _t_fin = 0.0
    action = None
    if animation:
        if onto_existing and anim_apply in {"ADD", "NLA"}:
            _nla_pushdown_current_action(arm_obj)
        action = bpy.data.actions.new(name=f"GR2LAB_{run_id or 'anim'}")
        _stamp_imported_action(action, run_id=run_id, fps=fps, duration=duration)
        _assign_armature_action(arm_obj, action)
        fcurves = _ensure_action_fcurves(arm_obj, action)

        scene = context.scene
        if not onto_existing:
            scene.render.fps = int(round(fps))
            scene.frame_start = 1
            scene.frame_end = max(1, int(math.ceil(_sec_to_frame(duration, fps))))

        if timer is not None and hasattr(timer, "live_print"):
            timer.live_print(f"bone_keys {len(skeleton)} bones…")
        bones_to_key = skeleton
    else:
        arm_obj["gr2lab_bind_only"] = True
        bones_to_key = []

    slice_t0 = time.perf_counter()
    if bones_to_key:
        yield from _yield_viewport_hidden(
            arm_obj, hide_viewport_build, 58, f"Bone keys 0/{len(bones_to_key)}"
        )
        slice_t0 = time.perf_counter()

    for bone_i, b in enumerate(bones_to_key):
        if getattr(arm_obj, "mode", "OBJECT") != "POSE":
            _mode_set_safe(context, "POSE", active=arm_obj)
        name = b["name"]
        if progress and bone_i % max(1, len(skeleton) // 24) == 0:
            progress.step(
                60 + int(38 * bone_i / max(1, len(skeleton))),
                f"Bone keys {bone_i + 1}/{len(skeleton)}: {name}",
            )
        pb = arm_obj.pose.bones.get(name)
        if not pb:
            continue
        pb.rotation_mode = "QUATERNION"
        packed = (pose_pack or {}).get(name)
        _tw = time.perf_counter()
        if packed is None:
            packed = _bone_channel_payload(
                b, tracks.get(name) or {}, rest_local[name], fps
            )
            time.sleep(0)
        _apply_packed_bone(pb, fcurves, name, packed)
        _t_write += time.perf_counter() - _tw
        if not onto_existing:
            pb.lock_scale = (True, True, True)
            pb["gr2lab_pos_mode"] = packed["pos_mode"]
            pb["gr2lab_ori_mode"] = packed["ori_mode"]
            pb["gr2lab_scl_mode"] = packed["scl_mode"]
            if packed.get("k16"):
                pb["gr2lab_k16_display_keys"] = True
        if (bone_i + 1) < len(bones_to_key) and (time.perf_counter() - slice_t0) >= 0.012:
            pct = 60 + int(28 * (bone_i + 1) / max(1, len(bones_to_key)))
            yield from _yield_viewport_hidden(
                arm_obj,
                hide_viewport_build,
                pct,
                f"Bone keys {bone_i + 1}/{len(bones_to_key)}",
            )
            slice_t0 = time.perf_counter()

    if timer is not None:
        timer.add("bone_keys_write", _t_write)
        timer.add("bone_keys_finalize", _t_fin)
        if hasattr(timer, "live_print"):
            timer.live_print(
                f"bone_keys_write {_t_write:.2f}s  bone_keys_finalize {_t_fin:.2f}s"
            )

    if animation and action is not None:
        landed = _ensure_action_fcurves(arm_obj, action)
        nfc = 0
        try:
            nfc = len(landed) if landed is not None else 0
        except Exception:
            nfc = 0
        print(f"[GR2 LAB] import action {action.name!r} fcurves={nfc}")
        if nfc == 0:
            print(
                "[GR2 LAB] import wrote 0 F-curves — Action slot is empty. "
                "Reload the addon and re-import."
            )
        if onto_existing and anim_apply == "NLA":
            _nla_add_action_strip(arm_obj, action, fps=fps, duration=duration)
        else:
            _assign_armature_action(arm_obj, action)
        if onto_existing:
            try:
                from .rig_display import head_ik as _head_ik

                _head_ik.release_head_ik_for_clip(arm_obj)
            except Exception:
                traceback.print_exc()
    import_want_head_ik = False
    if onto_existing:
        import_want_head_ik = arm_obj.pose.bones.get("IK_Head_Target") is not None
    else:
        parts = rig_parts if isinstance(rig_parts, dict) else {}
        want_shapes = bool(parts.get("shapes", rig_display))
        want_hands = bool(parts.get("ik_hands", rig_display))
        want_legs = bool(parts.get("ik_legs", rig_display))
        want_head = bool(parts.get("ik_head", rig_display))
        if not rig_display:
            want_shapes = want_hands = want_legs = want_head = False
        import_want_head_ik = want_head
        if want_shapes or want_hands or want_legs or want_head:
            yield from _yield_viewport_hidden(
                arm_obj, hide_viewport_build, 76, "Rig controls…"
            )
            try:
                from .rig_display import apply_imported_rig_display
                from timing import maybe_timer as _maybe_timer

                with _maybe_timer(timer, "rig_display"):
                    apply_imported_rig_display(
                        context,
                        arm_obj,
                        mode_set=_mode_set_safe,
                        shapes=want_shapes,
                        ik_hands=want_hands,
                        ik_legs=want_legs,
                        ik_head=want_head,
                        lookat_on=False,
                    )
            except Exception:
                traceback.print_exc()
                print("[GR2 LAB] DGB display setup failed (import animation is intact)")
        else:
            print("[GR2 LAB] rig setup BASE — skipped IK, shapes, and HUD")
        tag = []
        if want_shapes:
            tag.append("shapes")
        if want_hands:
            tag.append("hands")
        if want_legs:
            tag.append("legs")
        if want_head:
            tag.append("head")
        try:
            arm_obj["gr2lab_rig_setup"] = "+".join(tag) if tag else "BASE"
        except Exception:
            pass
    if (
        animation
        and action is not None
        and import_want_head_ik
        and arm_obj.pose.bones.get("IK_Head_Target") is not None
    ):
        yield from _yield_viewport_hidden(
            arm_obj, hide_viewport_build, 84, "Head look…"
        )
        try:
            from .rig_display import head_ik as _head_ik
            from timing import maybe_timer as _maybe_timer

            with _maybe_timer(timer, "head_ik_bake"):
                _head_ik.bake_ik_head_target_from_action(
                    context,
                    arm_obj,
                    clip_match=False,
                    check_toggle_gaps=False,
                    max_samples=160 if import_fast_head_ik else 400,
                )
        except Exception:
            traceback.print_exc()
    if swap_from is not None:
        # The previous body is skinned to the old rest pose. Moving it onto
        # the new base blows it up. Drop those meshes, then the kit for the
        # chosen base is attached below. Lips, widgets, and bone props stay.
        _maybe_clear_swap_meshes(swap_from)
        try:
            _swap_scene_rig(context, swap_from, arm_obj, keep_action=action)
        except Exception:
            traceback.print_exc()
            print("[GR2 LAB] swap rig failed (generated armature is intact)")
        _maybe_clear_swap_meshes(arm_obj)
    yield from _yield_viewport_hidden(arm_obj, hide_viewport_build, 90, "Meshes…")
    if timer is not None:
        from timing import maybe_timer as _maybe_timer

        with _maybe_timer(timer, "mesh_kit"):
            _maybe_attach_mesh_kit(
                context,
                arm_obj,
                anim_apply=anim_apply,
                import_meshes=import_meshes,
                import_akelka=import_akelka,
                mesh_base=mesh_base,
                doc=doc,
                template_path=template_path,
            )
    else:
        _maybe_attach_mesh_kit(
            context,
            arm_obj,
            anim_apply=anim_apply,
            import_meshes=import_meshes,
            import_akelka=import_akelka,
            mesh_base=mesh_base,
            doc=doc,
            template_path=template_path,
        )
    yield from _yield_viewport_hidden(arm_obj, hide_viewport_build, 94, "Tail…")
    if timer is not None:
        from timing import maybe_timer as _maybe_timer

        with _maybe_timer(timer, "tail_rig"):
            _maybe_attach_tail_rig(
                context,
                arm_obj,
                anim_apply=anim_apply,
                import_tail=import_tail,
                mesh_base=mesh_base,
                swap_from=swap_from,
                rig_display=rig_display,
                rig_parts=rig_parts,
                y_up_display=y_up_display,
            )
    else:
        _maybe_attach_tail_rig(
            context,
            arm_obj,
            anim_apply=anim_apply,
            import_tail=import_tail,
            mesh_base=mesh_base,
            swap_from=swap_from,
            rig_display=rig_display,
            rig_parts=rig_parts,
            y_up_display=y_up_display,
        )
    if swap_from is not None:
        _discard_replaced_rig(swap_from)
    # Edit/Pose were only needed to build bones, keys, and widgets. Leave Object
    # after Add rig; Replace/Add action/NLA restore whatever mode the rig was in.
    if onto_existing and restore_mode == "POSE":
        _mode_set_safe(context, "POSE", active=arm_obj)
    else:
        _mode_set_safe(context, "OBJECT", active=arm_obj)
    if animation and action is not None:
        try:
            from . import scene_stage_empties as _stage

            gr2_src = str(
                arm_obj.get("gr2lab_full_source_gr2")
                or arm_obj.get("gr2lab_source_gr2")
                or template_path
                or ""
            )
            if gr2_src:
                note = _stage.apply_glasses_body_hold_for_gr2(
                    context, arm_obj, gr2_src
                )
                if note:
                    print(f"[GR2 LAB] {note}")
        except Exception:
            traceback.print_exc()
    if progress:
        progress.step(99, "Import done")
    try:
        if int(context.scene.frame_current) != prev_frame:
            context.scene.frame_set(prev_frame)
    except Exception:
        pass
    return arm_obj


def import_gr2lab_from_doc(*args, **kwargs):
    """Build the rig on this thread. Modal import steps the generator instead."""
    gen = _import_gr2lab_steps(*args, **kwargs)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def import_gr2lab(
    context,
    filepath: str,
    *,
    fps: float = FPS_DEFAULT,
    y_up_display: bool = True,
    progress: Optional[Gr2LabProgress] = None,
    rig_display: bool = True,
    rig_parts: Optional[Dict[str, bool]] = None,
    target_arm=None,
    anim_apply: str = "NEW_RIG",
    import_meshes: str = "NONE",
    import_akelka: bool = False,
    import_tail: bool = False,
    import_clear_meshes: bool = False,
    mesh_base: Optional[str] = None,
) -> bpy.types.Object:
    if progress:
        progress.step(10, f"Reading {Path(filepath).name}…")
    doc = _load_doc(filepath)
    return import_gr2lab_from_doc(
        context,
        doc,
        fps=fps,
        y_up_display=y_up_display,
        template_path=filepath,
        progress=progress,
        rig_display=rig_display,
        rig_parts=rig_parts,
        target_arm=target_arm,
        anim_apply=anim_apply,
        import_meshes=import_meshes,
        import_akelka=import_akelka,
        import_tail=import_tail,
        import_clear_meshes=import_clear_meshes,
        mesh_base=mesh_base,
    )


def _assign_armature_action(arm_obj, action) -> None:
    """Assign action (+ slot on Blender 4.4+) so keyframe_insert writes F-curves."""
    if arm_obj is None or action is None:
        return
    if not arm_obj.animation_data:
        arm_obj.animation_data_create()
    ad = arm_obj.animation_data
    ad.action = action
    if not hasattr(ad, "action_slot"):
        return
    suitable = list(getattr(ad, "action_suitable_slots", None) or [])
    cur = getattr(ad, "action_slot", None)
    if suitable:
        if cur is None or cur not in suitable:
            ad.action_slot = suitable[0]
        return
    # Pose-bone keys live on the armature Object (OB), not Armature data (AR).
    if cur is not None:
        ad.action_slot = None
    slots = getattr(action, "slots", None)
    if slots is None:
        return
    owner_type = str(getattr(arm_obj, "id_type", "OBJECT") or "OBJECT")
    slot_name = (arm_obj.name or "Armature")[:63]
    try:
        slot = slots.new(id_type=owner_type, name=slot_name)
    except TypeError:
        try:
            slot = slots.new(owner_type, slot_name)
        except Exception:
            return
    ad.action_slot = slot


def _ensure_action_fcurves(arm_obj, action):
    """FCurve collection for new keys (slotted channelbag on Blender 4.4+/5, else action.fcurves)."""
    ad = getattr(arm_obj, "animation_data", None)
    if ad is None or action is None:
        return getattr(action, "fcurves", None)
    try:
        from bpy_extras import anim_utils

        ensure_slot = getattr(anim_utils, "action_ensure_channelbag_for_slot", None)
        slot = getattr(ad, "action_slot", None)
        if ensure_slot is not None and slot is not None:
            bag = ensure_slot(action, slot)
            if bag is not None and hasattr(bag, "fcurves"):
                return bag.fcurves
        ensure_ad = getattr(anim_utils, "animdata_ensure_channelbag_for_assigned_slot", None)
        if ensure_ad is not None:
            bag = ensure_ad(ad)
            if bag is not None and hasattr(bag, "fcurves"):
                return bag.fcurves
        get_bag = getattr(anim_utils, "animdata_get_channelbag_for_assigned_slot", None)
        if get_bag is not None:
            bag = get_bag(ad)
            if bag is not None and hasattr(bag, "fcurves"):
                return bag.fcurves
    except (ImportError, AttributeError, TypeError):
        pass
    layers = getattr(action, "layers", None)
    slot = getattr(ad, "action_slot", None) if ad else None
    if layers is not None and slot is not None:
        try:
            if len(layers) == 0:
                layers.new("Layer")
            layer = layers[0]
            strips = getattr(layer, "strips", None)
            if strips is not None:
                if len(strips) == 0:
                    try:
                        strips.new(type="KEYFRAME")
                    except TypeError:
                        strips.new()
                strip = strips[0]
                cb = getattr(strip, "channelbag", None)
                if callable(cb):
                    bag = cb(slot, ensure=True)
                    if bag is not None:
                        return bag.fcurves
        except Exception:
            pass
    return getattr(action, "fcurves", None)


def _bulk_fcurve_channel(fcurves, bone_name: str, prop: str, frames, component_values) -> None:
    """Write LINEAR F-curves via foreach_set (same keys as keyframe_insert)."""
    # Empty bpy_prop_collection is falsy — do not treat that as missing. Blender 5
    # slotted imports start with an empty channelbag; skipping leaves a dead Action.
    if fcurves is None or not frames:
        return
    data_path = f'pose.bones["{bone_name}"].{prop}'
    n = len(frames)
    for index, values in enumerate(component_values):
        if len(values) != n:
            continue
        fc = None
        try:
            fc = fcurves.find(data_path, index=index)
        except TypeError:
            try:
                fc = fcurves.find(data_path, index)
            except Exception:
                fc = None
        if fc is None:
            try:
                fc = fcurves.new(data_path, index=index, action_group=bone_name)
            except TypeError:
                try:
                    fc = fcurves.new(data_path, index=index)
                except TypeError:
                    fc = fcurves.new(data_path, index)
        try:
            fc.keyframe_points.clear()
        except Exception:
            pass
        fc.keyframe_points.add(n)
        co = [0.0] * (n * 2)
        for i in range(n):
            co[i * 2] = float(frames[i])
            co[i * 2 + 1] = float(values[i])
        fc.keyframe_points.foreach_set("co", co)
        _fcurve_set_linear_bulk(fc, n)


def _bulk_object_channel(fcurves, prop: str, frames, component_values) -> None:
    """Object-level F-curves (location/rotation/scale) via foreach_set."""
    if fcurves is None or not frames:
        return
    n = len(frames)
    for index, values in enumerate(component_values):
        if len(values) != n:
            continue
        fc = None
        try:
            fc = fcurves.find(prop, index=index)
        except TypeError:
            try:
                fc = fcurves.find(prop, index)
            except Exception:
                fc = None
        if fc is None:
            try:
                fc = fcurves.new(prop, index=index)
            except TypeError:
                fc = fcurves.new(prop, index)
        try:
            fc.keyframe_points.clear()
        except Exception:
            pass
        fc.keyframe_points.add(n)
        co = [0.0] * (n * 2)
        for i in range(n):
            co[i * 2] = float(frames[i])
            co[i * 2 + 1] = float(values[i])
        fc.keyframe_points.foreach_set("co", co)
        _fcurve_set_linear_bulk(fc, n)


def _iter_action_fcurves(action, *, anim_data=None):
    """Yield F-curves from legacy action.fcurves (≤4.5) and slotted channelbags (4.4+/5.x).

    Blender 4.4+/5.x RNA reuses the same Python proxy while iterating a
    channelbag's fcurves collection. Deduping by id(fc) then drops almost all
    keys (baked Action.* exports look like rest-pose in GR2). Materialize with
    list() and dedupe by (data_path, array_index) instead.
    """
    if not action:
        return
    seen: set[Tuple[str, int]] = set()

    def _emit(fcs):
        if fcs is None:
            return
        try:
            curves = list(fcs)
        except TypeError:
            return
        for fc in curves:
            try:
                key = (str(fc.data_path), int(fc.array_index))
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            yield fc

    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from _emit(legacy)

    if anim_data is not None:
        try:
            from bpy_extras import anim_utils

            bag = anim_utils.animdata_get_channelbag_for_assigned_slot(anim_data)
            if bag is not None:
                yield from _emit(getattr(bag, "fcurves", None))
        except (ImportError, AttributeError, TypeError):
            pass

    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []) or []:
                for bag in getattr(strip, "channelbags", []) or []:
                    yield from _emit(getattr(bag, "fcurves", None))


def _quat_pose_bone_short_arc(action, bone_name: str, *, anim_data=None) -> None:
    """Flip quaternion keys to the short arc (w,x,y,z component curves)."""
    if not action or not bone_name:
        return
    needle = f'pose.bones["{bone_name}"]'
    quat_fcs: Dict[int, Any] = {}
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        if needle not in fc.data_path:
            continue
        if fc.data_path.endswith("rotation_quaternion"):
            quat_fcs[fc.array_index] = fc
    if not all(i in quat_fcs for i in range(4)):
        return
    w_fc, x_fc, y_fc, z_fc = quat_fcs[0], quat_fcs[1], quat_fcs[2], quat_fcs[3]
    n = len(w_fc.keyframe_points)
    if n < 2 or not (
        len(x_fc.keyframe_points) == n
        and len(y_fc.keyframe_points) == n
        and len(z_fc.keyframe_points) == n
    ):
        return
    prev = Quaternion(
        (
            w_fc.keyframe_points[0].co[1],
            x_fc.keyframe_points[0].co[1],
            y_fc.keyframe_points[0].co[1],
            z_fc.keyframe_points[0].co[1],
        )
    )
    for i in range(1, n):
        cur = Quaternion(
            (
                w_fc.keyframe_points[i].co[1],
                x_fc.keyframe_points[i].co[1],
                y_fc.keyframe_points[i].co[1],
                z_fc.keyframe_points[i].co[1],
            )
        )
        cur.make_compatible(prev)
        w_fc.keyframe_points[i].co[1] = cur.w
        x_fc.keyframe_points[i].co[1] = cur.x
        y_fc.keyframe_points[i].co[1] = cur.y
        z_fc.keyframe_points[i].co[1] = cur.z
        prev = cur


def _finalize_pose_bone_fcurves(action, bone_name: str, *, anim_data=None) -> None:
    """LINEAR interpolation + quaternion continuity (matches Granny/site lerpВ·slerp)."""
    if not action or not bone_name:
        return
    needle = f'pose.bones["{bone_name}"]'
    _quat_pose_bone_short_arc(action, bone_name, anim_data=anim_data)
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        if needle not in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _smooth_pose_bone_fcurves(action, bone_name: str, *, anim_data=None) -> None:
    """Bezier auto-clamped handles for Blender-authored helper keys (not GR2 import)."""
    if not action or not bone_name:
        return
    needle = f'pose.bones["{bone_name}"]'
    _quat_pose_bone_short_arc(action, bone_name, anim_data=anim_data)
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        if needle not in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"
        try:
            fc.update()
        except Exception:
            pass


def _build_fcurve_index(
    action, *, anim_data=None
) -> Dict[str, Dict[Tuple[str, int], Any]]:
    """bone_name -> {(data_path, array_index): fcurve}."""
    out: Dict[str, Dict[Tuple[str, int], Any]] = {}
    prefix = 'pose.bones["'
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        dp = fc.data_path
        if not dp.startswith(prefix):
            continue
        end = dp.find('"]', len(prefix))
        if end < 0:
            continue
        bone = dp[len(prefix) : end]
        prop = dp[end + 2 :]
        if prop.startswith("."):
            prop = prop[1:]
        bucket = out.setdefault(bone, {})
        bucket[(prop, int(fc.array_index))] = fc
    return out


def _all_action_frames(action, *, anim_data=None) -> List[float]:
    frames = set()
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        for kp in fc.keyframe_points:
            frames.add(round(float(kp.co[0]), 6))
    return sorted(frames)


def _deselect_all_objects(context) -> None:
    """Deselect without bpy.ops (File Browser / wrong area breaks select_all.poll)."""
    for obj in context.view_layer.objects:
        try:
            obj.select_set(False)
        except RuntimeError:
            pass


def _temp_view3d_override(context, *, active=None, selected=None):
    """Build a VIEW_3D temp_override dict for mode_set / bake when context is thin."""
    win = context.window
    scr = win.screen if win else None
    area = next((a for a in (scr.areas if scr else []) if a.type == "VIEW_3D"), None)
    region = None
    if area is not None:
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is None and area.regions:
            region = area.regions[-1]
    act = active or context.view_layer.objects.active
    sel = list(selected) if selected is not None else ([act] if act else [])
    override = {
        "window": win,
        "screen": scr,
        "area": area,
        "region": region,
        "active_object": act,
        "object": act,
        "selected_objects": sel,
        "selected_editable_objects": sel,
    }
    return {k: v for k, v in override.items() if v is not None}


def _yield_viewport_hidden(arm, hide: bool, pct: int, label: str):
    """Hide the rig across a modal yield so the viewport does not draw the build."""
    if hide and arm is not None:
        try:
            arm.hide_viewport = True
        except Exception:
            pass
    yield (pct, label)
    if hide and arm is not None:
        try:
            arm.hide_viewport = False
        except Exception:
            pass


def _mode_set_safe(context, mode: str, *, active=None) -> None:
    """object.mode_set with VIEW_3D override fallback (import from File Browser).

    A strip drag keeps the rig in an excluded collection. Edit mode then uses the
    hidden build view layer, so the viewport the user is looking at never draws it.
    """
    if active is not None and bool(getattr(active, "hide_viewport", False)):
        active.hide_viewport = False
    scene = getattr(context, "scene", None)
    vl = getattr(context, "view_layer", None)
    if active is not None and scene is not None:
        in_hold = any(
            getattr(col, "name", "") == "GR2LAB_preload_hold"
            for col in getattr(active, "users_collection", ())
        )
        if in_hold:
            build = scene.view_layers.get("GR2LAB_preload_build")
            if build is not None:
                vl = build
    if active is not None and vl is not None:
        try:
            vl.objects.active = active
        except Exception:
            pass
    override = _temp_view3d_override(context, active=active)
    if scene is not None:
        override["scene"] = scene
    if vl is not None:
        override["view_layer"] = vl
    if active is not None:
        override["active_object"] = active
        override["object"] = active
        override["selected_objects"] = [active]
        override["selected_editable_objects"] = [active]
    try:
        with context.temp_override(**override):
            if active is not None:
                try:
                    active.select_set(True)
                except RuntimeError:
                    pass
            bpy.ops.object.mode_set(mode=mode)
    except (RuntimeError, TypeError):
        if vl is not None and active is not None:
            try:
                vl.objects.active = active
            except Exception:
                pass
        bpy.ops.object.mode_set(mode=mode)


def _resolve_action(arm_obj) -> Any:
    """Active action on armature (NLA strip fallback when action slot is empty)."""
    ad = arm_obj.animation_data if arm_obj else None
    if not ad:
        return None
    if ad.action:
        return ad.action
    if ad.nla_tracks:
        for track in ad.nla_tracks:
            if track.mute:
                continue
            for strip in track.strips:
                if strip.mute or not strip.action:
                    continue
                return strip.action
    return None


def _export_frame_list(context, arm_obj, action) -> List[float]:
    """Every scene frame in range + F-curve/NLA key times (what you see when scrubbing)."""
    scene = context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    if end < start:
        start, end = end, start
    frames: set = set(range(start, end + 1))
    ad = arm_obj.animation_data if arm_obj else None
    for fr in _all_action_frames(action, anim_data=ad):
        frames.add(int(round(fr)))
    if ad and ad.nla_tracks:
        for track in ad.nla_tracks:
            if track.mute:
                continue
            for strip in track.strips:
                if strip.mute or not strip.action:
                    continue
                base = float(strip.action_frame_start or 0.0)
                for fr in _all_action_frames(strip.action):
                    frames.add(int(round(strip.frame_start + fr - base)))
    return sorted(float(f) for f in frames) if frames else [1.0]


def _imported_clip_frame_end(arm_obj, scene) -> Optional[int]:
    """Last scene frame of the imported GR2 clip, or None if not an import."""
    if not arm_obj or not scene:
        return None
    raw_fp = arm_obj.get(PROP_IMPORT_FP)
    if not raw_fp:
        return None
    try:
        fp = json.loads(raw_fp) if isinstance(raw_fp, str) else raw_fp
        dur = float((fp or {}).get("duration") or 0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if dur <= 0:
        return None
    fps = float(arm_obj.get("gr2lab_fps") or scene.render.fps or FPS_DEFAULT)
    start = int(scene.frame_start)
    n = max(2, int(round(dur * fps)) + 1)
    return start + n - 1


def _import_locked_frame_list(
    context, arm_obj, action
) -> List[float]:
    """Scene/NLA/F-curve frames, capped to the imported GR2 duration when present."""
    frames: set = set(_export_frame_list(context, arm_obj, action))
    fps = float(
        arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT
    )
    start = int(context.scene.frame_start)
    lock_end = _imported_clip_frame_end(arm_obj, context.scene)
    raw_fp = arm_obj.get(PROP_IMPORT_FP) if arm_obj else None
    if raw_fp:
        try:
            fp = json.loads(raw_fp) if isinstance(raw_fp, str) else raw_fp
            dur = float((fp or {}).get("duration") or 0)
            if dur > 0:
                n = max(2, int(round(dur * fps)) + 1)
                for i in range(n):
                    frames.add(float(start + i))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if lock_end is not None:
        frames = {f for f in frames if start - 1e-6 <= float(f) <= float(lock_end) + 1e-6}
    return sorted(frames) if frames else [1.0]


def _frames_for_props(fcs: Dict[Tuple[str, int], Any], props: Sequence[str]) -> List[float]:
    frames = set()
    for (prop, _idx), fc in fcs.items():
        if prop not in props:
            continue
        for kp in fc.keyframe_points:
            frames.add(round(float(kp.co[0]), 6))
    return sorted(frames)


def _v3_close(a: Sequence[float], b: Sequence[float], eps: float = 1e-5) -> bool:
    return all(abs(float(a[i]) - float(b[i])) <= eps for i in range(3))


def _quat_close(a: Sequence[float], b: Sequence[float], eps: float = 0.99999) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    dot = abs(
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )
    return dot >= eps


def _eval_prop(
    fcs: Dict[Tuple[str, int], Any],
    prop: str,
    n: int,
    frame: float,
    defaults: Sequence[float],
) -> List[float]:
    vals = []
    for i in range(n):
        fc = fcs.get((prop, i))
        if fc is not None:
            vals.append(float(fc.evaluate(frame)))
        else:
            vals.append(float(defaults[i]))
    return vals


def _pose_local_trs(pb) -> Tuple[List[float], List[float], List[float]]:
    """Parent-relative local TRS from evaluated pose (includes constraints)."""
    if pb.parent:
        local = pb.parent.matrix.inverted() @ pb.matrix
    else:
        local = pb.matrix.copy()
    return _mat_to_trs(local)


def _constraint_affected_bones(arm_obj) -> set:
    """Bones with constraints, their targets, and IK chain members."""
    affected = set()
    for pb in arm_obj.pose.bones:
        for c in pb.constraints:
            affected.add(pb.name)
            st = getattr(c, "subtarget", None) or ""
            if st:
                affected.add(st)
            if c.type == "IK":
                chain = max(1, int(getattr(c, "chain_count", 1) or 1))
                cur = pb
                for _ in range(chain):
                    if not cur:
                        break
                    affected.add(cur.name)
                    cur = cur.parent
    return affected


def _parse_arm_doc(arm_obj) -> Optional[Dict[str, Any]]:
    """Read stamped gr2lab_json from an armature, or None if missing."""
    if not arm_obj:
        return None
    raw = arm_obj.get(PROP_DOC)
    if not raw:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError(f"{arm_obj.name}: gr2lab_json is not an object")
    return doc


def _iter_stamped_armatures(context, *, exclude=None):
    """Scene armatures that carry GR2LAB bridge data."""
    exclude_name = exclude.name if exclude else None
    for obj in context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        if exclude_name and obj.name == exclude_name:
            continue
        if obj.get(PROP_DOC):
            yield obj


def _addon_prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is None:
        addon = bpy.context.preferences.addons.get("gr2lab_blender")
    return addon.preferences if addon else None


def remember_template_path(path: str) -> None:
    """Persist last .gr2lab template so Export can run with one click."""
    if not path:
        return
    try:
        resolved = str(Path(path).expanduser().resolve())
    except OSError:
        resolved = str(path)
    if not Path(resolved).is_file():
        return
    prefs = _addon_prefs()
    if prefs is not None and hasattr(prefs, "last_template"):
        prefs.last_template = resolved
    try:
        bpy.context.window_manager[WM_LAST_TEMPLATE] = resolved
    except Exception:
        pass


def _is_gr2lab_path(path: Path) -> bool:
    low = path.name.lower()
    return low.endswith(".gr2lab") or low.endswith(".gr2lab.json")


def _blend_dir_gr2lab_files() -> List[Path]:
    """*.gr2lab next to the current .blend (newest first)."""
    if not bpy.data.filepath:
        return []
    parent = Path(bpy.data.filepath).parent
    if not parent.is_dir():
        return []
    found: List[Path] = []
    for p in parent.iterdir():
        if p.is_file() and _is_gr2lab_path(p):
            found.append(p)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def auto_template_path(context, arm_obj=None) -> Optional[str]:
    """
    Find a .gr2lab without the user picking one.
    Order: armature path в†’ prefs в†’ window manager в†’ single/newest file next to .blend
    """
    candidates: List[str] = []
    if arm_obj:
        stored = arm_obj.get(PROP_TEMPLATE_PATH)
        if stored:
            candidates.append(str(stored))
    prefs = _addon_prefs()
    last_tpl = getattr(prefs, "last_template", None) if prefs else None
    if last_tpl:
        candidates.append(str(last_tpl))
    try:
        wm_path = bpy.context.window_manager.get(WM_LAST_TEMPLATE)
        if wm_path:
            candidates.append(str(wm_path))
    except Exception:
        pass
    for p in _blend_dir_gr2lab_files():
        candidates.append(str(p))

    seen = set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = Path(raw).expanduser()
        if path.is_file() and _is_gr2lab_path(path):
            return str(path.resolve())
    return None


def _is_synthesized_gr2lab_doc(doc: Dict[str, Any]) -> bool:
    """True for synthesize_gr2lab_from_armature stamps (run_id blender / empty)."""
    rid = (doc.get("run_id") or "").strip()
    return not rid or rid == "blender"


def _pack_run_id(arm_obj) -> str:
    if not arm_obj:
        return ""
    for key in (PROP_PACK_RUN, PROP_RUN):
        raw = str(arm_obj.get(key) or "").strip()
        if raw and raw != "blender":
            return raw
    return ""


def _find_scene_template_armature(
    context, arm_obj, pack_run: str
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Scene armature with matching pack run and a real (non-synthesized) template."""
    if not pack_run or not context:
        return None
    for obj in _iter_stamped_armatures(context, exclude=arm_obj):
        if _pack_run_id(obj) != pack_run:
            continue
        doc = _parse_arm_doc(obj)
        if doc and not _is_synthesized_gr2lab_doc(doc):
            return obj, doc
    return None


def peek_export_template_hint(context, arm_obj=None) -> Tuple[str, str]:
    """
    UI hint: (status, detail).
    status: 'native' | 'auto' | 'synthesize'
    """
    arm_obj = arm_obj or (context.view_layer.objects.active if context else None)
    if arm_obj and arm_obj.type == "ARMATURE" and _parse_arm_doc(arm_obj) is not None:
        active_doc = _parse_arm_doc(arm_obj)
        if active_doc and _is_synthesized_gr2lab_doc(active_doc):
            match = _find_scene_template_armature(
                context, arm_obj, _pack_run_id(arm_obj)
            )
            if match:
                return "auto", match[0].name
        rid = arm_obj.get(PROP_RUN) or ""
        return "native", str(rid)
    path = auto_template_path(context, arm_obj)
    if path:
        return "auto", Path(path).name
    stamped = list(_iter_stamped_armatures(context, exclude=arm_obj)) if context else []
    if len(stamped) == 1:
        return "auto", stamped[0].name
    if arm_obj and arm_obj.type == "ARMATURE":
        n = len(arm_obj.data.bones)
        return "synthesize", f"{n} bones"
    return "synthesize", ""


def _bone_rest_local_matrix(bone) -> Matrix:
    """Parent-relative rest matrix in armature space."""
    if bone.parent:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local
    return bone.matrix_local.copy()


# Blender Z-up armature space в†’ Granny / BG3 Y-up
_RX_NEG_90 = Matrix.Rotation(math.radians(-90.0), 4, "X")


def _is_granny_armature_space(arm_obj) -> bool:
    """
    True when armature bone data is already Granny Y-up (GR2LAB Import).

    GR2LAB Import puts Granny locals into the armature and rotates the *object*
    +90В° X for Blender display. LSLib / Body_Base rigs keep Blender Z-up in the
    bones (Dummy_Root rest often ~90В° X) with no display offset вЂ” those need
    Blenderв†’Granny conversion on export.
    """
    if not arm_obj:
        return False
    if not arm_obj.get("gr2lab_y_up_display"):
        return False
    ex = abs(float(arm_obj.matrix_world.to_euler("XYZ").x))
    return abs(ex - math.radians(90.0)) < math.radians(20.0)


def _topo_bone_names(
    names: Sequence[str], parents: Dict[str, Optional[str]]
) -> List[str]:
    """Parents before children (needed for FK world build)."""
    name_set = set(names)
    remaining = set(names)
    ordered: List[str] = []
    while remaining:
        progress = False
        for name in list(remaining):
            p = parents.get(name) or None
            if not p or p not in name_set or p not in remaining:
                ordered.append(name)
                remaining.remove(name)
                progress = True
        if not progress:
            ordered.extend(sorted(remaining))
            break
    return ordered


def _blender_zup_locals_to_granny(
    parent_rel: Dict[str, Matrix],
    parents: Dict[str, Optional[str]],
) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    """
    Parent-relative Blender Z-up armature locals в†’ Granny Y-up parent-relative TRS.

    Builds Blender FK worlds, applies Rx(-90В°), re-extracts locals. Fixes Dummy_Root
    location that otherwise maps Blender horizontal motion onto Granny +Y (flies up).
    """
    order = _topo_bone_names(list(parent_rel.keys()), parents)
    worlds_bl: Dict[str, Matrix] = {}
    for name in order:
        local = parent_rel.get(name)
        if local is None:
            continue
        p = parents.get(name) or None
        worlds_bl[name] = (
            (worlds_bl[p] @ local) if p and p in worlds_bl else local.copy()
        )
    worlds_g = {n: _RX_NEG_90 @ w for n, w in worlds_bl.items()}
    out: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for name in order:
        if name not in worlds_g:
            continue
        p = parents.get(name) or None
        if p and p in worlds_g:
            local = worlds_g[p].inverted() @ worlds_g[name]
        else:
            local = worlds_g[name]
        out[name] = _mat_to_trs(local)
    return out


def synthesize_gr2lab_from_armature(arm_obj, context) -> Tuple[Dict[str, Any], float]:
    """
    Build a .gr2lab document from a plain LSLib / Body_Base-like armature.
    No site template required вЂ” site Import uses Pack-from dump + bone-name match.

    Rests are converted Blender Z-up в†’ Granny Y-up (same as export sampling).
    """
    if not arm_obj or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature to export")
    fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
    scene = context.scene
    duration = max(0.0, (float(scene.frame_end) - 1.0) / fps)

    from .rig_display.head_ik import is_helper_bone

    bl_rest: Dict[str, Matrix] = {}
    parents: Dict[str, Optional[str]] = {}
    order: List[str] = []
    for bone in arm_obj.data.bones:
        if is_helper_bone(bone.name):
            continue
        order.append(bone.name)
        bl_rest[bone.name] = _bone_rest_local_matrix(bone)
        parent_name = bone.parent.name if bone.parent else None
        if parent_name and is_helper_bone(parent_name):
            parent_name = None
        parents[bone.name] = parent_name

    if _is_granny_armature_space(arm_obj):
        granny_rest = {n: _mat_to_trs(m) for n, m in bl_rest.items()}
    else:
        granny_rest = _blender_zup_locals_to_granny(bl_rest, parents)

    skeleton: List[Dict[str, Any]] = []
    tracks: List[Dict[str, Any]] = []
    root_name = ""
    for i, name in enumerate(order):
        t, r, s = granny_rest.get(name) or ([0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
        parent = parents.get(name) or ""
        if not parent and not root_name:
            root_name = name
        skeleton.append(
            {
                "name": name,
                "index": i,
                "parent_name": parent or None,
                "rest_translation": t,
                "rest_rotation": r,
                "rest_scale": s,
            }
        )
        tracks.append(
            {
                "name": name,
                "index": i,
                "position": {
                    "edit_mode": "float_controls",
                    "editable": True,
                    "times": [0.0],
                    "values": [t],
                },
                "orientation": {
                    "edit_mode": "float_controls",
                    "editable": True,
                    "times": [0.0],
                    "values": [r],
                },
                "scale": {
                    "edit_mode": "constant",
                    "editable": False,
                    "times": [0.0],
                    "values": [s],
                },
            }
        )

    if not skeleton:
        raise ValueError("Armature has no bones")

    doc: Dict[str, Any] = {
        "format": FORMAT_ID,
        "version": 1,
        "run_id": "blender",
        "source_gr2": "",
        "duration": float(duration),
        "root": root_name or skeleton[0]["name"],
        "skeleton": skeleton,
        "tracks": tracks,
    }
    return doc, fps


def resolve_export_template(
    context,
    arm_obj,
    *,
    template_armature=None,
    template_filepath: Optional[str] = None,
) -> Tuple[Dict[str, Any], float, str]:
    """
    Resolve (doc, fps, source_label) for export.

    Order: explicit file → explicit scene armature → native stamp (non-synthesized)
    → matching pack-run scene template → other stamped scene armatures
    → synthesized active stamp → auto file → synthesize from armature.
    """
    path = (template_filepath or "").strip()
    if path:
        doc = _load_doc(path)
        fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
        remember_template_path(path)
        if arm_obj:
            arm_obj[PROP_TEMPLATE_PATH] = str(Path(path).resolve())
        return doc, fps, Path(path).name

    if template_armature and getattr(template_armature, "type", None) == "ARMATURE":
        doc = _parse_arm_doc(template_armature)
        if doc is not None:
            fps = float(
                template_armature.get("gr2lab_fps")
                or arm_obj.get("gr2lab_fps")
                or context.scene.render.fps
                or FPS_DEFAULT
            )
            return doc, fps, template_armature.name

    active_doc = _parse_arm_doc(arm_obj)
    if active_doc is not None and not _is_synthesized_gr2lab_doc(active_doc):
        fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
        return active_doc, fps, "active armature"

    pack_run = _pack_run_id(arm_obj)
    match = _find_scene_template_armature(context, arm_obj, pack_run)
    if match:
        obj, doc = match
        fps = float(
            obj.get("gr2lab_fps")
            or arm_obj.get("gr2lab_fps")
            or context.scene.render.fps
            or FPS_DEFAULT
        )
        return doc, fps, obj.name

    stamped = list(_iter_stamped_armatures(context, exclude=arm_obj))
    for obj in stamped:
        doc = _parse_arm_doc(obj)
        if doc and not _is_synthesized_gr2lab_doc(doc):
            fps = float(
                obj.get("gr2lab_fps")
                or arm_obj.get("gr2lab_fps")
                or context.scene.render.fps
                or FPS_DEFAULT
            )
            return doc, fps, obj.name
    if len(stamped) == 1:
        doc = _parse_arm_doc(stamped[0])
        if doc is not None:
            fps = float(
                stamped[0].get("gr2lab_fps")
                or context.scene.render.fps
                or FPS_DEFAULT
            )
            return doc, fps, stamped[0].name

    if active_doc is not None:
        fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
        return active_doc, fps, "active armature"

    path = auto_template_path(context, arm_obj) or ""
    if path:
        doc = _load_doc(path)
        fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
        remember_template_path(path)
        if arm_obj:
            arm_obj[PROP_TEMPLATE_PATH] = str(Path(path).resolve())
        return doc, fps, Path(path).name

    # Naked LSLib / Body_Base-like rig: build bridge from the armature itself
    doc, fps = synthesize_gr2lab_from_armature(arm_obj, context)
    return doc, fps, "synthesize"


def stamp_gr2lab_on_armature(
    arm_obj,
    doc: Dict[str, Any],
    *,
    fps: float = FPS_DEFAULT,
    y_up_display: Optional[bool] = None,
    template_path: Optional[str] = None,
) -> None:
    """Stamp bridge metadata onto an armature without rebuilding bones."""
    if not arm_obj or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature to attach GR2LAB data")
    doc_store = dict(doc)
    bind_by_name = _bind_map(doc_store)
    skeleton = _skeleton_with_bind_parents(doc_store.get("skeleton") or [], bind_by_name)
    doc_store["skeleton"] = skeleton
    doc_store["format"] = FORMAT_ID
    arm_obj[PROP_DOC] = json.dumps(doc_store)
    arm_obj[PROP_RUN] = doc_store.get("run_id") or ""
    arm_obj["gr2lab_fps"] = float(fps)
    if y_up_display is not None:
        arm_obj["gr2lab_y_up_display"] = bool(y_up_display)
    elif "gr2lab_y_up_display" not in arm_obj:
        # False for naked LSLib attach/synthesize; Import GR2LAB sets True explicitly
        arm_obj["gr2lab_y_up_display"] = False
    arm_obj["gr2lab_has_bind"] = bool(bind_by_name)
    if template_path:
        try:
            arm_obj[PROP_TEMPLATE_PATH] = str(Path(template_path).resolve())
            remember_template_path(template_path)
        except OSError:
            arm_obj[PROP_TEMPLATE_PATH] = str(template_path)


def _sample_quat_channel(ch: dict, t: float) -> Optional[List[float]]:
    times = [float(x) for x in (ch.get("times") or [])]
    vals = ch.get("values") or []
    if not vals:
        return None
    if len(times) == 1 or t <= times[0]:
        return list(vals[0])
    if t >= times[-1]:
        return list(vals[-1])
    i = bisect_right(times, t) - 1
    i = max(0, min(i, len(times) - 2))
    span = times[i + 1] - times[i]
    u = 0.0 if span <= 1e-12 else (t - times[i]) / span
    qa = _quat_gr_to_bl(vals[i])
    qb = _quat_gr_to_bl(vals[i + 1])
    qb.make_compatible(qa)
    return _quat_bl_to_gr(qa.slerp(qb, u))


def _sample_vec_channel(ch: dict, t: float) -> Optional[List[float]]:
    times = [float(x) for x in (ch.get("times") or [])]
    vals = ch.get("values") or []
    if not vals:
        return None
    if len(times) == 1 or t <= times[0]:
        return list(vals[0])
    if t >= times[-1]:
        return list(vals[-1])
    i = bisect_right(times, t) - 1
    i = max(0, min(i, len(times) - 2))
    span = times[i + 1] - times[i]
    u = 0.0 if span <= 1e-12 else (t - times[i]) / span
    a, b = vals[i], vals[i + 1]
    return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(3)]


def _count_motion_delta_bones(tmpl_doc: dict, export_doc: dict) -> int:
    """How export differs from the import snapshot (duration, keys, pose samples)."""
    t_dur = float(tmpl_doc.get("duration") or 0.0)
    e_dur = float(export_doc.get("duration") or 0.0)
    if abs(t_dur - e_dur) > 1e-4:
        return max(1, len(export_doc.get("tracks") or []))

    tm = {t["name"]: t for t in (tmpl_doc.get("tracks") or []) if t.get("name")}
    probes = [0.0]
    if e_dur > 0:
        probes.extend([e_dur * 0.25, e_dur * 0.5, e_dur * 0.75, max(0.0, e_dur - 1e-4)])
    probes = sorted(set(probes))
    changed = 0
    for tr in export_doc.get("tracks") or []:
        name = tr.get("name")
        if not name or name not in tm:
            continue
        tb = tm[name]
        bone_changed = False
        for ch_key, is_quat in (("orientation", True), ("position", False)):
            ea = tr.get(ch_key) or {}
            eb = tb.get(ch_key) or {}
            ta = [float(x) for x in (ea.get("times") or [])]
            tb_t = [float(x) for x in (eb.get("times") or [])]
            if len(ta) != len(tb_t):
                bone_changed = True
                break
            if ta and tb_t and max(abs(a - b) for a, b in zip(ta, tb_t)) > 1e-3:
                bone_changed = True
                break
            sampler = _sample_quat_channel if is_quat else _sample_vec_channel
            close = (
                (lambda a, b: _quat_close(a, b, 0.9995))
                if is_quat
                else (lambda a, b: _v3_close(a, b, 2e-4))
            )
            for t in probes:
                sa = sampler(ea, t)
                sb = sampler(eb, t)
                if sa and sb and not close(sa, sb):
                    bone_changed = True
                    break
            if bone_changed:
                break
        if bone_changed:
            changed += 1
    return changed


def _trim_channel_to_duration(ch: dict, max_t: float) -> dict:
    out = dict(ch)
    times = [float(t) for t in (ch.get("times") or [])]
    vals = ch.get("values") or []
    if not times:
        return out
    nt, nv = [], []
    for t, v in zip(times, vals):
        if float(t) <= max_t + TIME_EPS:
            nt.append(float(t))
            nv.append(v)
    if not nt:
        nt, nv = [0.0], [vals[0]]
    out["times"] = nt
    out["values"] = nv
    return out


def _trim_tracks_to_duration(tracks: List[dict], duration: float) -> None:
    for tr in tracks:
        for key in ("position", "orientation", "scale"):
            ch = tr.get(key)
            if ch:
                tr[key] = _trim_channel_to_duration(ch, duration)


def _granny_local_from_pose(
    name: str,
    fr: float,
    *,
    pb,
    fcs: Dict,
    rest_local: Dict[str, Matrix],
    identity_loc,
    identity_quat,
    identity_scl,
    use_matrix_eval: bool,
) -> Matrix:
    """Inverse of import: basis F-curves -> rest @ basis = Granny local (preferred)."""
    if use_matrix_eval and pb is not None:
        if pb.parent:
            return pb.parent.matrix.inverted() @ pb.matrix
        return pb.matrix.copy()
    loc = _eval_prop(fcs, "location", 3, fr, identity_loc)
    qwxyz = _eval_prop(fcs, "rotation_quaternion", 4, fr, identity_quat)
    scl = _eval_prop(fcs, "scale", 3, fr, identity_scl)
    w, x, y, z = qwxyz
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm > 1e-12:
        w, x, y, z = w / nrm, x / nrm, y / nrm, z / nrm
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    basis = Matrix.LocRotScale(
        Vector(loc), Quaternion((w, x, y, z)), Vector(scl)
    )
    return rest_local[name] @ basis


def _nla_active_strips(ad) -> List[Any]:
    """Unmuted NLA strips that carry an action."""
    if not ad or not ad.nla_tracks:
        return []
    out: List[Any] = []
    for track in ad.nla_tracks:
        if track.mute:
            continue
        for strip in track.strips:
            if strip.mute or not strip.action:
                continue
            out.append(strip)
    return out


def _nla_strips_overlap(a, b) -> bool:
    return float(a.frame_start) < float(b.frame_end) and float(b.frame_start) < float(
        a.frame_end
    )


def _nla_strips_active(arm_obj) -> bool:
    """True when unmuted NLA strips will influence the visible pose."""
    ad = arm_obj.animation_data if arm_obj else None
    if not ad:
        return False
    strips = _nla_active_strips(ad)
    if not strips:
        return False
    # use_nla blends strips; without an action the strips are the only source.
    return bool(ad.use_nla) or not ad.action


def _nla_needs_flatten(arm_obj) -> bool:
    """True when multiple NLA strips must be blended (multi-track / overlap / retime)."""
    ad = arm_obj.animation_data if arm_obj else None
    if not ad:
        return False
    strips = _nla_active_strips(ad)
    if len(strips) <= 1:
        if not strips:
            return False
        scale = float(getattr(strips[0], "scale", 1.0) or 1.0)
        return abs(scale - 1.0) > 1e-5
    for i, a in enumerate(strips):
        if abs(float(getattr(a, "scale", 1.0) or 1.0) - 1.0) > 1e-5:
            return True
        for b in strips[i + 1:]:
            if _nla_strips_overlap(a, b):
                return True
    active_tracks = 0
    for track in ad.nla_tracks:
        if track.mute:
            continue
        if any(not s.mute and s.action for s in track.strips):
            active_tracks += 1
    return active_tracks > 1


def _armature_has_constraints(arm_obj) -> bool:
    """True when the armature object or any pose bone has constraints."""
    if not arm_obj or arm_obj.type != "ARMATURE":
        return False
    if len(arm_obj.constraints) > 0:
        return True
    if arm_obj.pose:
        for pb in arm_obj.pose.bones:
            if len(pb.constraints) > 0:
                return True
    return False


def _export_should_bake(arm_obj, mode: str = "AUTO") -> bool:
    """
    ALWAYS: destructive visual flatten (NLA + constraints → one Action).
    AUTO: flatten when unmuted NLA strips and/or any constraints are present.
    NEVER: no mutation — F-curves + constraint-affected depsgraph eval.
    """
    m = (mode or "AUTO").upper()
    if m == "ALWAYS":
        return True
    if m == "NEVER":
        return False
    return _nla_strips_active(arm_obj) or _armature_has_constraints(arm_obj)


def _export_sampling_hint(arm_obj, bake_mode: str = "AUTO") -> str:
    """Short label for export UI / stats."""
    if _export_should_bake(arm_obj, bake_mode):
        return "visual_flatten"
    if _nla_strips_active(arm_obj):
        if _nla_needs_flatten(arm_obj):
            return "nla_eval_blend"
        return "nla_eval_strip"
    if _armature_has_constraints(arm_obj):
        return "fcurve_constraint_eval"
    return "fcurves"


def _bake_action_frame_range(action) -> Optional[Tuple[int, int]]:
    if action is None:
        return None
    if hasattr(action, "curve_frame_range"):
        lo, hi = action.curve_frame_range
        if hi >= lo and any(True for _ in _iter_action_fcurves(action)):
            return int(round(lo)), int(round(hi))
        if hi >= lo and not hasattr(action, "layers"):
            return int(round(lo)), int(round(hi))
    lo = hi = None
    for fc in _iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            f = kp.co[0]
            lo = f if lo is None else min(lo, f)
            hi = f if hi is None else max(hi, f)
    if lo is None:
        return None
    return int(round(lo)), int(round(hi))


def _bake_nla_frame_range(anim) -> Optional[Tuple[int, int]]:
    if anim is None:
        return None
    lo = hi = None
    for track in anim.nla_tracks:
        if track.mute:
            continue
        for strip in track.strips:
            if strip.mute:
                continue
            lo = strip.frame_start if lo is None else min(lo, strip.frame_start)
            hi = strip.frame_end if hi is None else max(hi, strip.frame_end)
    if lo is None:
        return None
    return int(round(lo)), int(round(hi))


def _bake_union_ranges(*ranges: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    lo = hi = None
    for r in ranges:
        if not r:
            continue
        a, b = r
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    if lo is None:
        return None
    return int(lo), int(hi)


def determine_bake_range(obj, scene) -> Tuple[int, int]:
    """Frame range from Action ∪ NLA strips, else scene range.

    Imported GR2 clips stay locked to the source duration so bake/export
    cannot stretch 9.000 s → 9.033 s from a longer scene range.
    """
    anim = obj.animation_data
    action_range = _bake_action_frame_range(anim.action if anim else None)
    nla_range = _bake_nla_frame_range(anim)
    combined = _bake_union_ranges(action_range, nla_range)
    if combined:
        lo, hi = combined
    else:
        lo, hi = int(scene.frame_start), int(scene.frame_end)
    lock_end = _imported_clip_frame_end(obj, scene)
    if lock_end is not None:
        lo = int(scene.frame_start)
        hi = min(int(hi), int(lock_end))
        if hi < lo:
            hi = lo
    return lo, hi


def _prepare_anim_for_eval(obj) -> None:
    anim = obj.animation_data
    if anim is None:
        return
    if getattr(anim, "use_tweak_mode", False):
        anim.use_tweak_mode = False
    anim.use_nla = True


def _clear_nla_tracks(obj) -> int:
    anim = obj.animation_data
    if anim is None:
        return 0
    if getattr(anim, "use_tweak_mode", False):
        anim.use_tweak_mode = False
    count = len(anim.nla_tracks)
    while anim.nla_tracks:
        anim.nla_tracks.remove(anim.nla_tracks[0])
    return count


def _clear_all_constraints(obj) -> Tuple[int, int]:
    obj_count = len(obj.constraints)
    while obj.constraints:
        obj.constraints.remove(obj.constraints[0])
    bone_count = 0
    if obj.pose:
        for pb in obj.pose.bones:
            bone_count += len(pb.constraints)
            while pb.constraints:
                pb.constraints.remove(pb.constraints[0])
    return obj_count, bone_count


def _decompose_local(mat: Matrix, rotation_mode: str):
    loc, rot_quat, scale = mat.decompose()
    if rotation_mode == "QUATERNION":
        rot = rot_quat
    elif rotation_mode == "AXIS_ANGLE":
        axis, angle = rot_quat.to_axis_angle()
        rot = (angle, axis.x, axis.y, axis.z)
    else:
        rot = rot_quat.to_euler(rotation_mode)
    return loc, rot, scale


def _apply_bone_transform(pb, loc, rot, scale, rotation_mode: str) -> None:
    pb.location = loc
    pb.scale = scale
    if rotation_mode == "QUATERNION":
        pb.rotation_quaternion = rot
    elif rotation_mode == "AXIS_ANGLE":
        pb.rotation_axis_angle = rot
    else:
        pb.rotation_euler = rot


def _keyframe_bone(pb, frame: int) -> None:
    group = pb.name
    pb.keyframe_insert(data_path="location", frame=frame, group=group)
    pb.keyframe_insert(data_path="scale", frame=frame, group=group)
    mode = pb.rotation_mode
    if mode == "QUATERNION":
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=group)
    elif mode == "AXIS_ANGLE":
        pb.keyframe_insert(data_path="rotation_axis_angle", frame=frame, group=group)
    else:
        pb.keyframe_insert(data_path="rotation_euler", frame=frame, group=group)


def _keyframe_object(obj, frame: int) -> None:
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)
    mode = obj.rotation_mode
    if mode == "QUATERNION":
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    elif mode == "AXIS_ANGLE":
        obj.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def _find_coll_fcurve(fcurves, data_path: str, index: int):
    if fcurves is None:
        return None
    try:
        return fcurves.find(data_path, index=index)
    except TypeError:
        try:
            return fcurves.find(data_path, index)
        except Exception:
            return None


_BEZT_IPO_LIN = 1  # DNA IpoInterpolationMode; foreach_set rejects "LINEAR" strings
_LIN_PAYLOAD_CACHE: Dict[int, Any] = {}


def _linear_interp_payload(n: int):
    payload = _LIN_PAYLOAD_CACHE.get(n)
    if payload is None:
        payload = array("b", [_BEZT_IPO_LIN] * n)
        _LIN_PAYLOAD_CACHE[n] = payload
    return payload


def _fcurve_set_linear_bulk(fc, n: int) -> None:
    """LINEAR via int enum only. String foreach_set TypeError can smash BezTriples."""
    kps = fc.keyframe_points
    cached = _linear_interp_payload(n)
    try:
        kps.foreach_set("interpolation", cached)
        return
    except Exception:
        pass
    for payload in (
        [_BEZT_IPO_LIN] * n,
        array("i", [_BEZT_IPO_LIN]) * n,
    ):
        try:
            kps.foreach_set("interpolation", payload)
            return
        except Exception:
            continue
    for kp in kps:
        kp.interpolation = "LINEAR"


def _fcurve_keys_match_samples(fc, frames, values) -> bool:
    """Read back first/mid/last keys so a fast write cannot silently keep rest pose."""
    try:
        kps = fc.keyframe_points
        n = len(frames)
        if len(kps) != n:
            return False
        probes = (0, n // 2, n - 1) if n >= 3 else range(n)
        for i in probes:
            co = kps[i].co
            if abs(float(co[0]) - float(frames[i])) > 0.51:
                return False
            if abs(float(co[1]) - float(values[i])) > 1e-3:
                return False
        return True
    except Exception:
        return False


def _fcurve_replace_keys(fc, frames, values) -> bool:
    """Replace all keys in one RNA call. True only when read-back matches samples."""
    if fc is None or not frames:
        return False
    n = len(frames)
    if len(values) != n:
        return False
    kps = fc.keyframe_points
    try:
        kps.clear()
    except Exception:
        pass
    try:
        if len(kps) != 0:
            kps.clear()
    except Exception:
        pass
    try:
        kps.add(n)
    except Exception:
        return False
    if len(kps) != n:
        return False
    co = [0.0] * (n * 2)
    for i in range(n):
        co[i * 2] = float(frames[i])
        co[i * 2 + 1] = float(values[i])
    try:
        kps.foreach_set("co", co)
    except Exception:
        return False
    try:
        fc.update()
    except Exception:
        pass
    try:
        _fcurve_set_linear_bulk(fc, n)
    except Exception:
        pass
    return _fcurve_keys_match_samples(fc, frames, values)


def _pack_baked_bone_channels(pb, frames, bone_samples):
    """Parallel loc/rot/scale arrays for one bone (quat signs made compatible)."""
    name = pb.name
    mode = pb.rotation_mode
    n = len(frames)
    loc_cs = [[0.0] * n for _ in range(3)]
    scl_cs = [[0.0] * n for _ in range(3)]
    if mode == "QUATERNION":
        rot_cs = [[0.0] * n for _ in range(4)]
        rot_prop = "rotation_quaternion"
    elif mode == "AXIS_ANGLE":
        rot_cs = [[0.0] * n for _ in range(4)]
        rot_prop = "rotation_axis_angle"
    else:
        rot_cs = [[0.0] * n for _ in range(3)]
        rot_prop = "rotation_euler"
    prev_q = None
    for i, f in enumerate(frames):
        loc, rot, scale, _mode = bone_samples[f][name]
        loc_cs[0][i] = float(loc.x)
        loc_cs[1][i] = float(loc.y)
        loc_cs[2][i] = float(loc.z)
        scl_cs[0][i] = float(scale.x)
        scl_cs[1][i] = float(scale.y)
        scl_cs[2][i] = float(scale.z)
        if mode == "QUATERNION":
            q = rot
            if prev_q is not None:
                q = q.copy()
                q.make_compatible(prev_q)
            prev_q = q
            rot_cs[0][i] = float(q.w)
            rot_cs[1][i] = float(q.x)
            rot_cs[2][i] = float(q.y)
            rot_cs[3][i] = float(q.z)
        elif mode == "AXIS_ANGLE":
            rot_cs[0][i] = float(rot[0])
            rot_cs[1][i] = float(rot[1])
            rot_cs[2][i] = float(rot[2])
            rot_cs[3][i] = float(rot[3])
        else:
            rot_cs[0][i] = float(rot[0])
            rot_cs[1][i] = float(rot[1])
            rot_cs[2][i] = float(rot[2])
    return loc_cs, rot_cs, scl_cs, rot_prop


def _bulk_write_existing_channels(fcurves, data_path: str, frames_f, component_values) -> bool:
    """foreach_set onto curves that keyframe_insert already created. No fcurves.new."""
    if fcurves is None:
        return False
    for index, values in enumerate(component_values):
        fc = _find_coll_fcurve(fcurves, data_path, index)
        if not _fcurve_replace_keys(fc, frames_f, values):
            return False
    return True


def _object_visual_basis_matrix(obj, eval_obj) -> Matrix:
    """Matrix_basis that reproduces evaluated world transform without constraints."""
    mw = eval_obj.matrix_world.copy()
    if obj.parent:
        parent_world = obj.parent.matrix_world @ obj.matrix_parent_inverse
        return parent_world.inverted_safe() @ mw
    return mw


def _set_action_fcurves_linear(action, *, anim_data=None) -> None:
    for fc in _iter_action_fcurves(action, anim_data=anim_data):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _fcurve_is_object_transform(data_path: str) -> bool:
    return data_path in {
        "location",
        "rotation_euler",
        "rotation_quaternion",
        "rotation_axis_angle",
        "scale",
    } or data_path.startswith("rotation")


def _has_object_motion_or_constraints(obj) -> bool:
    if len(obj.constraints) > 0:
        return True
    anim = obj.animation_data
    if anim and anim.action:
        for fc in _iter_action_fcurves(anim.action, anim_data=anim):
            if _fcurve_is_object_transform(fc.data_path):
                return True
    if anim:
        for track in anim.nla_tracks:
            for strip in track.strips:
                act = strip.action
                if not act:
                    continue
                for fc in _iter_action_fcurves(act):
                    if _fcurve_is_object_transform(fc.data_path):
                        return True
    return False


_BAKE_ISO_NAME = "_GR2LAB_BAKE"
_CONSTRAINT_OBJ_ATTRS = ("target", "pole_target", "space_object")


def _bake_keep_objects(obj):
    """Armature plus constraint/parent/driver targets that must stay in the depsgraph."""
    keep = {obj}
    parent = getattr(obj, "parent", None)
    while parent is not None:
        keep.add(parent)
        parent = getattr(parent, "parent", None)

    def add_id(ident):
        if ident is None:
            return
        if isinstance(ident, bpy.types.Object):
            keep.add(ident)
            p = ident.parent
            while p is not None:
                keep.add(p)
                p = p.parent

    def add_constraint(con):
        for attr in _CONSTRAINT_OBJ_ATTRS:
            add_id(getattr(con, attr, None))

    for con in obj.constraints:
        add_constraint(con)
    if obj.pose is not None:
        for pb in obj.pose.bones:
            for con in pb.constraints:
                add_constraint(con)
    ad = obj.animation_data
    if ad is not None:
        for fc in getattr(ad, "drivers", []) or []:
            drv = getattr(fc, "driver", None)
            if drv is None:
                continue
            for var in drv.variables:
                for tgt in var.targets:
                    add_id(getattr(tgt, "id", None))
    return keep


def _iter_layer_collections(lc):
    yield lc
    for child in lc.children:
        yield from _iter_layer_collections(child)


def _find_layer_collection(root_lc, col):
    for lc in _iter_layer_collections(root_lc):
        if lc.collection == col:
            return lc
    return None


def _layer_collection_path(root_lc, target):
    def walk(lc, trail):
        here = trail + [lc]
        if lc == target:
            return here
        for child in lc.children:
            hit = walk(child, here)
            if hit:
                return hit
        return None

    return walk(root_lc, [])


def _bake_view_isolate(context, obj):
    """Hide/exclude everything except this rig so frame_set is cheap.

    Viewport hide + collection exclude. Constraint/parent/driver targets stay.
    Always pair with _bake_view_restore.
    """
    scene = getattr(context, "scene", None)
    view_layer = getattr(context, "view_layer", None)
    if scene is None or view_layer is None or obj is None:
        return None
    keep = _bake_keep_objects(obj)
    if not keep:
        return None
    iso = bpy.data.collections.get(_BAKE_ISO_NAME)
    if iso is None:
        iso = bpy.data.collections.new(_BAKE_ISO_NAME)
    iso.hide_viewport = False
    iso.hide_render = False
    try:
        if iso.name not in scene.collection.children:
            scene.collection.children.link(iso)
    except RuntimeError:
        pass
    for ob in keep:
        try:
            if ob.name not in iso.objects:
                iso.objects.link(ob)
        except RuntimeError:
            pass
    try:
        view_layer.update()
    except Exception:
        pass

    root_lc = view_layer.layer_collection
    iso_lc = _find_layer_collection(root_lc, iso)
    keep_lc = set()
    if iso_lc is not None:
        keep_lc.update(_layer_collection_path(root_lc, iso_lc) or [])
        keep_lc.add(iso_lc)

    exclude_saved = []
    for lc in _iter_layer_collections(root_lc):
        if lc == root_lc:
            continue
        try:
            exclude_saved.append((lc, bool(lc.exclude)))
            lc.exclude = lc not in keep_lc
        except Exception:
            pass
    if iso_lc is not None:
        try:
            iso_lc.exclude = False
            if hasattr(iso_lc, "hide_viewport"):
                iso_lc.hide_viewport = False
        except Exception:
            pass

    unlinked_root = []
    for ob in list(scene.collection.objects):
        if ob in keep:
            continue
        try:
            scene.collection.objects.unlink(ob)
            unlinked_root.append(ob)
        except RuntimeError:
            pass

    hide_saved = []
    for ob in list(getattr(view_layer, "objects", []) or []):
        if ob in keep:
            continue
        try:
            hide_saved.append(
                (ob, bool(ob.hide_get()), bool(getattr(ob, "hide_viewport", False)))
            )
            ob.hide_set(True)
            ob.hide_viewport = True
        except Exception:
            pass

    try:
        view_layer.update()
    except Exception:
        pass
    n_excl = 0
    for lc, _was in exclude_saved:
        try:
            if lc.exclude:
                n_excl += 1
        except Exception:
            pass
    print(
        f"[GR2 LAB] bake isolate keep={len(keep)} hidden={len(hide_saved)} "
        f"excluded_cols={n_excl} iso_lc={'yes' if iso_lc else 'NO'}"
    )
    return {
        "iso": iso,
        "exclude": exclude_saved,
        "unlinked_root": unlinked_root,
        "hide": hide_saved,
        "scene": scene,
        "view_layer": view_layer,
    }


def _bake_view_restore(context, state) -> None:
    if not state:
        return
    scene = state.get("scene") or getattr(context, "scene", None)
    view_layer = state.get("view_layer") or getattr(context, "view_layer", None)
    for ob, was_hide, was_vp in state.get("hide") or []:
        try:
            ob.hide_set(was_hide)
            ob.hide_viewport = was_vp
        except Exception:
            pass
    for lc, was in state.get("exclude") or []:
        try:
            if lc is not None:
                lc.exclude = was
        except Exception:
            pass
    iso = state.get("iso")
    if iso is not None:
        for ob in list(getattr(iso, "objects", []) or []):
            try:
                iso.objects.unlink(ob)
            except RuntimeError:
                pass
        if scene is not None:
            try:
                if iso.name in scene.collection.children:
                    scene.collection.children.unlink(iso)
            except RuntimeError:
                pass
        try:
            bpy.data.collections.remove(iso)
        except Exception:
            pass
    if scene is not None:
        for ob in state.get("unlinked_root") or []:
            try:
                if ob.name not in scene.collection.objects:
                    scene.collection.objects.link(ob)
            except (RuntimeError, ReferenceError):
                pass
    if view_layer is not None:
        try:
            view_layer.update()
        except Exception:
            pass


def bake_armature_nla_and_constraints(
    context,
    obj,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    step: int = 1,
    bake_object: Optional[bool] = None,
    *,
    push_undo: bool = True,
):
    """
    Bake evaluated NLA + constraints into one Action, then remove NLA + constraints.

    Generator: yields (pct, label) progress tuples; returns result dict via StopIteration.value.
    """
    if not obj or obj.type != "ARMATURE" or obj.pose is None:
        raise TypeError("Active object must be an Armature")
    if obj.library or obj.override_library:
        raise RuntimeError(
            f"'{obj.name}' is linked/overridden. Make a local copy before baking."
        )

    if push_undo:
        try:
            bpy.ops.ed.undo_push(message="GR2 LAB: Bake NLA + Constraints")
        except Exception:
            pass

    scene = context.scene
    if frame_start is None or frame_end is None:
        auto_s, auto_e = determine_bake_range(obj, scene)
        frame_start = auto_s if frame_start is None else frame_start
        frame_end = auto_e if frame_end is None else frame_end

    frame_start = int(frame_start)
    frame_end = int(frame_end)
    step = max(1, int(step))
    if frame_end < frame_start:
        raise ValueError(f"Invalid frame range: {frame_start} .. {frame_end}")

    if bake_object is None:
        bake_object = _has_object_motion_or_constraints(obj)

    _prepare_anim_for_eval(obj)
    from .rig_display.head_ik import is_helper_bone

    pose_bones = [pb for pb in obj.pose.bones if not is_helper_bone(pb.name)]
    if not pose_bones:
        raise RuntimeError(f"Armature '{obj.name}' has no pose bones")

    frames = list(range(frame_start, frame_end + 1, step))
    if frames and frames[-1] != frame_end:
        frames.append(frame_end)
    if not frames:
        raise ValueError(f"Empty bake frame list: {frame_start} .. {frame_end}")

    iso_state = None
    try:
        iso_state = _bake_view_isolate(context, obj)
    except Exception as exc:
        print(f"[GR2 LAB] bake isolate skipped: {exc}")
        iso_state = None

    try:
        yield (3, "Isolating scene for bake…")
        yield (4, f"Sampling visual pose ({len(frames)} frames)…")

        bone_samples: Dict[int, Dict[str, tuple]] = {}
        object_samples: Dict[int, tuple] = {}
        view_layer = context.view_layer
        n_frames = len(frames)
        sample_s = 0.0
        write_s = 0.0
        write_mode = "insert"
        t_sample = time.perf_counter()
        for i, f in enumerate(frames):
            scene.frame_set(f)
            view_layer.update()
            depsgraph = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)

            frame_bones = {}
            for pb in pose_bones:
                eval_pb = eval_obj.pose.bones[pb.name]
                # Original Object/PoseBone + evaluated pose matrix (convert_space on
                # evaluated IDs is unreliable across Blender versions).
                local_mat = obj.convert_space(
                    pose_bone=pb,
                    matrix=eval_pb.matrix.copy(),
                    from_space="POSE",
                    to_space="LOCAL",
                )
                loc, rot, scale = _decompose_local(local_mat, pb.rotation_mode)
                frame_bones[pb.name] = (
                    Vector(loc),
                    rot.copy() if hasattr(rot, "copy") else tuple(rot),
                    Vector(scale),
                    pb.rotation_mode,
                )
            bone_samples[f] = frame_bones

            if bake_object:
                basis = _object_visual_basis_matrix(obj, eval_obj)
                loc, rot, scale = _decompose_local(basis, obj.rotation_mode)
                object_samples[f] = (
                    Vector(loc),
                    rot.copy() if hasattr(rot, "copy") else tuple(rot),
                    Vector(scale),
                    obj.rotation_mode,
                )

            if i == 0 or i == n_frames - 1 or (i % 8) == 0:
                pct = 4 + int(4 * (i + 1) / max(1, n_frames))
                yield (pct, f"Sampling visual pose {i + 1}/{n_frames}…")
        sample_s = time.perf_counter() - t_sample

        yield (8, "Clearing NLA + constraints…")
        nla_removed = _clear_nla_tracks(obj)
        obj_c_removed, bone_c_removed = _clear_all_constraints(obj)

        action_name = f"{obj.name}_Baked"
        existing = bpy.data.actions.get(action_name)
        if existing is not None:
            if obj.animation_data and obj.animation_data.action == existing:
                obj.animation_data.action = None
            bpy.data.actions.remove(existing)

        action = bpy.data.actions.new(action_name)
        _assign_armature_action(obj, action)
        if obj.animation_data:
            obj.animation_data.use_nla = True
            obj.animation_data.use_tweak_mode = False

        yield (8, f"Writing Action '{action_name}'…")
        t_write = time.perf_counter()
        f0 = frames[0]
        for pb in pose_bones:
            loc, rot, scale, mode = bone_samples[f0][pb.name]
            _apply_bone_transform(pb, loc, rot, scale, mode)
            _keyframe_bone(pb, f0)
        if bake_object and f0 in object_samples:
            loc, rot, scale, mode = object_samples[f0]
            obj.location = loc
            obj.scale = scale
            if mode == "QUATERNION":
                obj.rotation_quaternion = rot
            elif mode == "AXIS_ANGLE":
                obj.rotation_axis_angle = rot
            else:
                obj.rotation_euler = rot
            _keyframe_object(obj, f0)

        fcurves = _ensure_action_fcurves(obj, action)
        frames_f = [float(f) for f in frames]
        n_bones = max(1, len(pose_bones))
        bulk_bones = 0
        fallback_bones = 0
        bone_chunk = 8
        for bi, pb in enumerate(pose_bones):
            loc_cs, rot_cs, scl_cs, rot_prop = _pack_baked_bone_channels(
                pb, frames, bone_samples
            )
            loc_path = f'pose.bones["{pb.name}"].location'
            rot_path = f'pose.bones["{pb.name}"].{rot_prop}'
            scl_path = f'pose.bones["{pb.name}"].scale'
            ok = (
                _bulk_write_existing_channels(fcurves, loc_path, frames_f, loc_cs)
                and _bulk_write_existing_channels(fcurves, rot_path, frames_f, rot_cs)
                and _bulk_write_existing_channels(fcurves, scl_path, frames_f, scl_cs)
            )
            if ok:
                bulk_bones += 1
            else:
                fallback_bones += 1
                if fallback_bones == 1:
                    print(
                        f"[GR2 LAB] bake write fallback bone={pb.name!r} "
                        "(bulk keys missing or values did not match samples)"
                    )
                for f in frames:
                    loc, rot, scale, mode = bone_samples[f][pb.name]
                    _apply_bone_transform(pb, loc, rot, scale, mode)
                    _keyframe_bone(pb, f)
            if bi == 0 or bi + 1 == n_bones or ((bi + 1) % bone_chunk) == 0:
                yield (
                    8,
                    f"Writing Action '{action_name}' ({bi + 1}/{n_bones} bones)…",
                )

        if bake_object and object_samples:
            obj_mode = obj.rotation_mode
            n = len(frames)
            o_loc = [[0.0] * n for _ in range(3)]
            o_scl = [[0.0] * n for _ in range(3)]
            if obj_mode == "QUATERNION":
                o_rot = [[0.0] * n for _ in range(4)]
                o_prop = "rotation_quaternion"
            elif obj_mode == "AXIS_ANGLE":
                o_rot = [[0.0] * n for _ in range(4)]
                o_prop = "rotation_axis_angle"
            else:
                o_rot = [[0.0] * n for _ in range(3)]
                o_prop = "rotation_euler"
            obj_ok = True
            for i, f in enumerate(frames):
                if f not in object_samples:
                    obj_ok = False
                    break
                loc, rot, scale, mode = object_samples[f]
                o_loc[0][i] = float(loc.x)
                o_loc[1][i] = float(loc.y)
                o_loc[2][i] = float(loc.z)
                o_scl[0][i] = float(scale.x)
                o_scl[1][i] = float(scale.y)
                o_scl[2][i] = float(scale.z)
                if mode == "QUATERNION":
                    o_rot[0][i] = float(rot.w)
                    o_rot[1][i] = float(rot.x)
                    o_rot[2][i] = float(rot.y)
                    o_rot[3][i] = float(rot.z)
                elif mode == "AXIS_ANGLE":
                    o_rot[0][i] = float(rot[0])
                    o_rot[1][i] = float(rot[1])
                    o_rot[2][i] = float(rot[2])
                    o_rot[3][i] = float(rot[3])
                else:
                    o_rot[0][i] = float(rot[0])
                    o_rot[1][i] = float(rot[1])
                    o_rot[2][i] = float(rot[2])
            if obj_ok:
                obj_ok = (
                    _bulk_write_existing_channels(fcurves, "location", frames_f, o_loc)
                    and _bulk_write_existing_channels(fcurves, o_prop, frames_f, o_rot)
                    and _bulk_write_existing_channels(fcurves, "scale", frames_f, o_scl)
                )
            if not obj_ok:
                for f in frames:
                    if f not in object_samples:
                        continue
                    loc, rot, scale, mode = object_samples[f]
                    obj.location = loc
                    obj.scale = scale
                    if mode == "QUATERNION":
                        obj.rotation_quaternion = rot
                    elif mode == "AXIS_ANGLE":
                        obj.rotation_axis_angle = rot
                    else:
                        obj.rotation_euler = rot
                    _keyframe_object(obj, f)

        write_s = time.perf_counter() - t_write
        write_mode = "bulk" if fallback_bones == 0 else (
            "insert" if bulk_bones == 0 else "mixed"
        )
        print(
            f"[GR2 LAB] bake write {write_mode} bulk_bones={bulk_bones} "
            f"fallback_bones={fallback_bones} {write_s:.2f}s"
        )

        final_action = obj.animation_data.action if obj.animation_data else action
        if final_action and fallback_bones:
            _set_action_fcurves_linear(
                final_action,
                anim_data=obj.animation_data,
            )

        nla_removed += _clear_nla_tracks(obj)
        oc2, bc2 = _clear_all_constraints(obj)
        obj_c_removed += oc2
        bone_c_removed += bc2

        if obj.animation_data:
            obj.animation_data.use_tweak_mode = False
            obj.animation_data.action = final_action
            _assign_armature_action(obj, final_action)

        yield (9, "Bake done")
        return {
            "object": obj.name,
            "action": final_action.name if final_action else action_name,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "step": step,
            "frames_keyed": len(frames),
            "bones": len(pose_bones),
            "baked_object": bool(bake_object),
            "nla_tracks_removed": nla_removed,
            "object_constraints_removed": obj_c_removed,
            "bone_constraints_removed": bone_c_removed,
            "sample_s": float(sample_s),
            "write_s": float(write_s),
            "write_mode": write_mode,
        }
    finally:
        _bake_view_restore(context, iso_state)


def _bake_timeline_for_export(context, arm_obj):
    """
    Visual flatten for export. Always a generator: yields (pct, label);
    returns result dict (or None) via StopIteration.value.
    """
    if not arm_obj or arm_obj.type != "ARMATURE":
        return None
        yield  # pragma: no cover — keep generator type
    gen = bake_armature_nla_and_constraints(context, arm_obj, push_undo=True)
    try:
        while True:
            yield next(gen)
    except StopIteration as stop:
        return stop.value
    finally:
        try:
            gen.close()
        except Exception:
            pass


BAKE_MODE_ITEMS = (
    (
        "AUTO",
        "Auto",
        "Flatten when unmuted NLA strips and/or constraints are present (destructive)",
    ),
    (
        "ALWAYS",
        "Always",
        "Always flatten NLA + constraints into one Action (destructive, slow)",
    ),
    (
        "NEVER",
        "Never",
        "Never flatten — F-curves + constraint-affected depsgraph eval only",
    ),
)


def export_gr2lab(
    context,
    filepath: str,
    *,
    arm_obj=None,
    template_armature=None,
    template_filepath: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
    bake_mode: str = "AUTO",
    cooperative: bool = False,
    timer=None,
):
    """
    Export tracks from arm_obj by exact bone-name match against a GR2LAB template.
    F-curves by default; evaluated pose for constraint-affected bones.
    When cooperative=True, yields (pct, label) so a modal TIMER can refresh the UI,
    then returns stats via StopIteration.value.
    """
    from timing import StageTimer

    timer = timer or StageTimer("export")
    arm_obj = arm_obj or context.view_layer.objects.active
    if not arm_obj or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature to export animation from")

    baked = False
    bake_info = None
    should_bake = _export_should_bake(arm_obj, bake_mode)
    sampling_hint = _export_sampling_hint(arm_obj, bake_mode)
    _t_bake = time.perf_counter()
    if cooperative:
        yield (2, "Preparing export…")
    if should_bake:
        if progress:
            progress.step(4, "Flattening NLA + constraints…")
        if cooperative:
            yield (4, "Flattening NLA + constraints…")
        bake_gen = _bake_timeline_for_export(context, arm_obj)
        try:
            while True:
                item = next(bake_gen)
                if progress:
                    progress.step(int(item[0]), str(item[1]))
                if cooperative:
                    yield item
        except StopIteration as stop:
            bake_info = stop.value
        baked = bool(bake_info)
        if baked:
            sampling_hint = "visual_flatten"
        done_label = (
            f"Bake done ({bake_info.get('action')})"
            if baked and isinstance(bake_info, dict)
            else "Bake skipped — sampling F-curves…"
        )
        if progress:
            progress.step(9, done_label)
        if cooperative:
            yield (9, done_label)
    elif progress:
        progress.step(4, "Skipping bake (action F-curves)…")
    if cooperative and not should_bake:
        yield (4, "Skipping bake (action F-curves)…")
    timer.add("bake", time.perf_counter() - _t_bake)
    if isinstance(bake_info, dict):
        if bake_info.get("sample_s") is not None:
            timer.add("bake_sample", float(bake_info["sample_s"]))
        if bake_info.get("write_s") is not None:
            timer.add("bake_write", float(bake_info["write_s"]))

    _t_tpl = time.perf_counter()
    if progress:
        progress.step(10, "Loading export template…")
    active_doc_before = _parse_arm_doc(arm_obj)
    had_stamp = active_doc_before is not None
    active_was_synthesized = (
        had_stamp
        and active_doc_before is not None
        and _is_synthesized_gr2lab_doc(active_doc_before)
    )
    doc, fps, template_source = resolve_export_template(
        context,
        arm_obj,
        template_armature=template_armature,
        template_filepath=template_filepath,
    )
    snapshot_raw = arm_obj.get(PROP_IMPORT_SNAPSHOT) if arm_obj else None
    snapshot_doc = None
    if snapshot_raw:
        try:
            snapshot_doc = json.loads(snapshot_raw)
        except Exception:
            snapshot_doc = None
    delta_ref_doc = snapshot_doc if snapshot_doc else doc
    # First export from a plain rig: stamp template onto the armature so later
    # exports are native (no scene GR2LAB_* / no re-picking the file).
    stamped_native = False
    if not had_stamp:
        tpl_path = (template_filepath or "").strip() or auto_template_path(context, arm_obj)
        stamp_gr2lab_on_armature(arm_obj, doc, fps=fps, template_path=tpl_path)
        stamped_native = True
    bind_by_name = _bind_map(doc)
    skeleton = _skeleton_with_bind_parents(doc.get("skeleton") or [], bind_by_name)
    tracks = _track_map(doc)
    ad0 = arm_obj.animation_data
    prev_action = ad0.action if ad0 else None
    # NLA-only (empty action slot): do not assign the first strip. That REPLACE
    # action hides COMBINE extra-fix / location tracks during depsgraph sample.
    nla_only = bool(ad0 and prev_action is None and _nla_strips_active(arm_obj) and not baked)
    action = _resolve_action(arm_obj)
    if ad0 and action and not nla_only:
        _assign_armature_action(arm_obj, action)
    fc_index = _build_fcurve_index(action, anim_data=arm_obj.animation_data)

    template_names = {b["name"] for b in skeleton if b.get("name")}
    arm_bone_names = {b.name for b in arm_obj.data.bones}
    matched_names = template_names & arm_bone_names
    extras = len(arm_bone_names - template_names)

    # F-curve quat keys may already flip hemispheres (paste / bake). Fix before sample.
    # Skip on NLA-only — would mutate the vanilla strip action, not the stack.
    if action and not nla_only:
        for _bn in matched_names:
            if _bn in fc_index:
                _finalize_pose_bone_fcurves(
                    action, _bn, anim_data=arm_obj.animation_data
                )

    # Two spaces:
    # - GR2LAB Import (+90В° object): armature bones are already Granny вЂ” rest @ basis
    # - LSLib / Body_Base (Z-up bones): sample Blender locals, convert Rx(-90В°) в†’ Granny
    granny_space = _is_granny_armature_space(arm_obj)

    rest_local: Dict[str, Matrix] = {}
    parents: Dict[str, Optional[str]] = {}
    for b in skeleton:
        name = b["name"]
        bone = arm_obj.data.bones.get(name)
        if granny_space:
            rest_local[name] = _bind_local_for_bone(b, bind_by_name)
            parents[name] = b.get("parent_name") or (
                bone.parent.name if bone and bone.parent else None
            )
        else:
            # Always Blender rest for Z-up в†’ Granny convert (ignore template rests)
            if bone:
                rest_local[name] = _bone_rest_local_matrix(bone)
                parents[name] = bone.parent.name if bone.parent else None
            else:
                rest_local[name] = _bind_local_for_bone(b, bind_by_name)
                parents[name] = b.get("parent_name") or None

    affected = _constraint_affected_bones(arm_obj)
    nla_eval = _nla_strips_active(arm_obj) and not baked
    # After bake, visual pose is in F-curves — skip depsgraph.
    # NLA (single strip or blended): eval all matched bones so push-down matches action export.
    # Empty F-curve index on a slotted bake (Blender 5 RNA) → eval pose so we never
    # export rest-only tracks while the viewport shows motion.
    fcurve_bones = {n for n in matched_names if fc_index.get(n)}
    sparse_fcurves = bool(action) and len(fcurve_bones) < max(1, int(0.5 * len(matched_names)))
    if baked:
        need_eval_names: set = set()
        sample_names = set(matched_names)
    elif nla_eval or sparse_fcurves:
        need_eval_names = set(matched_names)
        sample_names = set(matched_names)
    else:
        need_eval_names = affected & matched_names
        sample_names = {n for n in matched_names if fc_index.get(n) or n in affected}

    identity_loc = (0.0, 0.0, 0.0)
    identity_quat = (1.0, 0.0, 0.0, 0.0)
    identity_scl = (1.0, 1.0, 1.0)

    scene = context.scene
    prev_frame = scene.frame_current
    frame_list = _import_locked_frame_list(context, arm_obj, action)

    local_cache: Dict[float, Dict[str, Tuple[List[float], List[float], List[float]]]] = {}
    fr_total = max(1, len(frame_list))
    # Smaller batches when depsgraph eval is needed; larger for pure F-curve path.
    sample_batch = 4 if need_eval_names else 48
    timer.add("load_template", time.perf_counter() - _t_tpl)
    _t_samp = time.perf_counter()
    try:
        fi = 0
        while fi < len(frame_list):
            batch_end = min(fi + sample_batch, len(frame_list))
            for fr in frame_list[fi:batch_end]:
                key = round(fr, 6)
                bl_bucket: Dict[str, Matrix] = {}
                arm_eval = None
                if need_eval_names:
                    scene.frame_set(int(round(fr)))
                    depsgraph = context.evaluated_depsgraph_get()
                    arm_eval = arm_obj.evaluated_get(depsgraph)
                for name in sample_names:
                    if name not in rest_local:
                        continue
                    fcs = fc_index.get(name) or {}
                    pb = arm_eval.pose.bones.get(name) if arm_eval else None
                    use_matrix = name in need_eval_names and pb is not None
                    bl_bucket[name] = _granny_local_from_pose(
                        name,
                        fr,
                        pb=pb,
                        fcs=fcs,
                        rest_local=rest_local,
                        identity_loc=identity_loc,
                        identity_quat=identity_quat,
                        identity_scl=identity_scl,
                        use_matrix_eval=use_matrix,
                    )
                for name in sample_names:
                    if name not in bl_bucket and name in rest_local:
                        bl_bucket[name] = rest_local[name].copy()

                if granny_space:
                    local_cache[key] = {
                        n: _mat_to_trs(m) for n, m in bl_bucket.items()
                    }
                else:
                    local_cache[key] = _blender_zup_locals_to_granny(bl_bucket, parents)
            fi = batch_end
            pct = 12 + int(56 * fi / max(1, fr_total))
            label = f"Sampling frames {fi}/{fr_total}…"
            if progress:
                progress.step(pct, label)
            if cooperative:
                yield (pct, label)
    finally:
        if need_eval_names:
            scene.frame_set(prev_frame)
        ad_now = arm_obj.animation_data
        if ad_now is not None and ad_now.action is not prev_action:
            ad_now.action = prev_action

    timer.add("sample", time.perf_counter() - _t_samp)

    # Decompose + Z-up convert both pick q/-q per frame — kill export shake here.
    _stabilize_local_cache_quats(local_cache)
    cache_keys_sorted = sorted(local_cache.keys())

    def sample_at(name: str, frame: float) -> Tuple[List[float], List[float], List[float]]:
        key = round(frame, 6)
        bucket = local_cache.get(key)
        if bucket is None:
            if not cache_keys_sorted:
                if granny_space:
                    return _mat_to_trs(rest_local.get(name, Matrix.Identity(4)))
                rest_g = _blender_zup_locals_to_granny(rest_local, parents)
                return rest_g.get(name) or ([0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
            nearest, _dist = _nearest_sorted(cache_keys_sorted, key)
            bucket = local_cache.get(nearest) if nearest is not None else None
        if bucket and name in bucket:
            return bucket[name]
        if granny_space:
            return _mat_to_trs(rest_local.get(name, Matrix.Identity(4)))
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0])

    def times_collapse(samples: Dict[float, List[float]], is_quat: bool) -> List[float]:
        items = sorted(samples.items())
        if not items:
            return [0.0]
        _t0, v0 = items[0]
        close = _quat_close if is_quat else _v3_close
        if all(close(v0, v) for _, v in items):
            return [0.0]
        return [float(t) for t, _ in items]

    _t_tr = time.perf_counter()
    new_tracks = []
    global_times = [_frame_to_sec(fr, fps) for fr in frame_list]
    matched = 0
    missing = 0
    # Prefer import-time channels for protected roots (not a later densified stamp).
    snap_tracks = _track_map(snapshot_doc) if snapshot_doc else {}

    track_batch = 6
    for bi, b in enumerate(skeleton):
        name = b["name"]
        tr = tracks.get(name) or {
            "name": name,
            "position": {
                "edit_mode": "none",
                "editable": False,
                "times": [0.0],
                "values": [[0, 0, 0]],
            },
            "orientation": {
                "edit_mode": "none",
                "editable": False,
                "times": [0.0],
                "values": [[0, 0, 0, 1]],
            },
            "scale": {
                "edit_mode": "none",
                "editable": False,
                "times": [0.0],
                "values": [[1, 1, 1]],
            },
        }

        # Template bone not on this armature — keep original track values
        if name not in matched_names:
            missing += 1
            new_tracks.append(
                {
                    "name": name,
                    "index": tr.get("index", b.get("index")),
                    "position": dict(tr.get("position") or {}),
                    "orientation": dict(tr.get("orientation") or {}),
                    "scale": dict(tr.get("scale") or {}),
                }
            )
            continue

        # Rest-only (no F-curves, not constraint/NLA eval): leave template track.
        if name not in sample_names:
            matched += 1
            new_tracks.append(
                {
                    "name": name,
                    "index": tr.get("index", b.get("index")),
                    "position": dict(tr.get("position") or {}),
                    "orientation": dict(tr.get("orientation") or {}),
                    "scale": dict(tr.get("scale") or {}),
                }
            )
            continue

        matched += 1
        pos = dict(tr.get("position") or {})
        ori = dict(tr.get("orientation") or {})
        scl = dict(tr.get("scale") or {})
        pos_mode = pos.get("edit_mode") or "none"
        ori_mode = ori.get("edit_mode") or "none"
        scl_mode = scl.get("edit_mode") or "none"
        fcs = fc_index.get(name) or {}
        bone_affected = name in affected

        def base_times(ch, mode, props, *, upgradable: bool):
            if mode == "none" and not upgradable:
                return []
            orig = [float(t) for t in (ch.get("times") or [0.0])] or [0.0]
            fc_times = [_frame_to_sec(fr, fps) for fr in _frames_for_props(fcs, props)]
            export_t = global_times
            # NLA stack eval: extra-fix loc/rot may live on COMBINE strips, not
            # the first-strip F-curves. Sample the evaluated pose grid.
            if nla_eval and upgradable and mode in ("constant", "none", "k16_offsets"):
                return _collapse_times_eps(
                    list(orig) + fc_times + export_t, prefer=export_t
                ) or [0.0]
            if mode == "constant" and not upgradable:
                return [0.0]
            if mode == "k16_offsets" and not upgradable:
                return [0.0]
            # k16: import places full dequantized display keys; snap to template.
            # Collapse to bias-only later when samples still match the template.
            if mode == "k16_offsets":
                if len(fc_times) <= 1:
                    return [0.0]
                return _snap_times_to_template(fc_times, orig) or [0.0]
            # constant: F-curve keys only. Extra location keys = user wants multi-key
            # в†’ site upgrades to float.
            if mode == "constant":
                if len(fc_times) <= 1:
                    return [0.0]
                return sorted(set(fc_times)) or [0.0]
            # float / quat_k16: template knots + every exported scene frame (dense bake).
            # Collapse float-noise near-dups; prefer exact frame-grid times.
            snapped = _snap_times_to_template(fc_times, orig)
            merged = _collapse_times_eps(
                list(snapped) + export_t, prefer=export_t
            )
            if bone_affected and upgradable:
                merged = _collapse_times_eps(
                    list(merged) + export_t, prefer=export_t
                ) or [0.0]
            return merged or [0.0]

        cand_pos = base_times(pos, pos_mode, ("location",), upgradable=True)
        cand_ori = base_times(ori, ori_mode, ("rotation_quaternion",), upgradable=True)
        cand_scl = base_times(scl, scl_mode, ("scale",), upgradable=False)

        all_t = _collapse_times_eps(
            list(cand_pos or []) + list(cand_ori or []) + list(cand_scl or []),
            prefer=global_times,
        )
        cache: Dict[float, Tuple[List[float], List[float], List[float]]] = {}
        for t in all_t:
            cache[t] = sample_at(name, _sec_to_frame(t, fps))

        pos_samples = {t: cache[t][0] for t in (cand_pos or [0.0]) if t in cache}
        ori_samples = {t: cache[t][1] for t in (cand_ori or [0.0]) if t in cache}

        if pos_mode == "none" and not pos.get("editable"):
            # Identity Granny position (DaIdentity). Still ship if the user
            # added location motion — encode upgrades none → float Deg1.
            collapsed = times_collapse(pos_samples, False)
            need_pos = collapsed if len(collapsed) > 1 else []
        elif pos_mode == "k16_offsets":
            # Untouched display keys в†’ bias-only; frame-bake densify / real edits в†’ multi-key
            tmpl_t = [float(t) for t in (pos.get("times") or [0.0])]
            tmpl_v = pos.get("values") or []
            sample_times = list(pos_samples.keys())
            if _is_mass_grid_densify_times(tmpl_t, sample_times):
                need_pos = times_collapse(pos_samples, False)
            elif _channel_is_display_densified(pos) and len(sample_times) > 1:
                need_pos = times_collapse(pos_samples, False)
            elif _k16_pos_matches_template(pos_samples, tmpl_t, tmpl_v):
                need_pos = [0.0]
            else:
                need_pos = times_collapse(pos_samples, False)
        elif pos_mode in ("constant", "none"):
            # Collapse flat bias; keep multi-key only when F-curves actually vary
            need_pos = times_collapse(pos_samples, False)
        else:
            # Keep snapped F-curve times; collapse float-noise near-dups onto frame grid.
            need_pos = _collapse_times_eps(
                list(pos_samples.keys()), prefer=global_times
            ) or [0.0]

        if ori_mode == "none" and not ori.get("editable"):
            collapsed_ori = times_collapse(ori_samples, True)
            need_ori = collapsed_ori if len(collapsed_ori) > 1 else []
        elif ori_mode in ("constant", "none"):
            need_ori = times_collapse(ori_samples, True)
        else:
            need_ori = _collapse_times_eps(
                list(ori_samples.keys()), prefer=global_times
            ) or [0.0]

        # Never collapse real location/rotation motion back to a single bias key.
        # Compare against the template curve (not "does this channel vary over
        # time") so untouched k16 locomotion is not densified on a no-op export.
        if len(need_pos) <= 1 and _root_samples_differ_from_template(
            pos_samples, pos, is_quat=False
        ):
            need_pos = times_collapse(pos_samples, False)
        if len(need_ori) <= 1 and _root_samples_differ_from_template(
            ori_samples, ori, is_quat=True
        ):
            need_ori = times_collapse(ori_samples, True)

        need_scl = [0.0]

        def rewrite_channel(ch, mode, times_list, pick, *, upgradable: bool):
            out = dict(ch)
            if mode == "none" and not upgradable:
                return out
            if mode in ("constant", "k16_offsets") and not upgradable:
                t0 = 0.0
                if t0 not in cache:
                    cache[t0] = sample_at(name, _sec_to_frame(t0, fps))
                out["times"] = [0.0]
                out["values"] = [pick(cache[t0])]
                return out
            if not times_list:
                return out
            times = []
            values = []
            seen = set()
            for t in times_list:
                key = round(t, 6)
                if key in seen:
                    continue
                seen.add(key)
                if t not in cache:
                    cache[t] = sample_at(name, _sec_to_frame(t, fps))
                times.append(float(t))
                values.append(pick(cache[t]))
            out["times"] = times
            out["values"] = values
            if mode in ("constant", "k16_offsets", "none") and len(times) > 1:
                out["editable"] = True
                out["edit_mode"] = "float_controls"
                out["degree"] = 1
            return out

        # Protect Dummy_Root / Root_M: keep original channels unless sampled
        # motion actually leaves the template (NLA COMBINE / keyed loc+rot).
        # Same-motion bake densify must not remake Root_M quat_k16.
        protect_root = name in PROTECTED_ROOT_BONES
        root_keep = snap_tracks.get(name) or tr
        allow_root_pos = (not protect_root) or _allow_protected_root_rewrite(
            bone_affected, pos_samples, pos, is_quat=False
        )
        allow_root_ori = (not protect_root) or _allow_protected_root_rewrite(
            bone_affected, ori_samples, ori, is_quat=True
        )
        if allow_root_pos:
            pos = rewrite_channel(
                pos, pos_mode, need_pos, lambda s: s[0], upgradable=True
            )
            if protect_root:
                pos = _strip_root_display_densify(pos, kind="locomotion")
        else:
            pos = dict(root_keep.get("position") or tr.get("position") or {})
        if allow_root_ori:
            ori = rewrite_channel(
                ori, ori_mode, need_ori, lambda s: s[1], upgradable=True
            )
            if ori.get("values") and len(ori["values"]) > 1:
                _make_quat_series_compatible(ori["values"])
            if protect_root:
                ori = _strip_root_display_densify(ori, kind="ori")
        else:
            ori = dict(root_keep.get("orientation") or tr.get("orientation") or {})
            # Same-motion keep: never ship import display densify (that used to
            # remake Root_M quat_k16 as Deg1×~600 of the *vanilla* curve).
            if protect_root and (
                _channel_is_display_densified(ori)
                or (
                    (ori.get("edit_mode") or "") == "float_controls"
                    and len(ori.get("times") or []) >= 100
                )
            ):
                sparse = dict(tr.get("orientation") or {})
                if not _channel_is_display_densified(sparse) and not (
                    (sparse.get("edit_mode") or "") == "float_controls"
                    and len(sparse.get("times") or []) >= 100
                ):
                    ori = sparse
        # Body_Base / DGB templates often bake Dummy_Root rest Rx(90°) into the
        # orientation track. Protect-keep would ship that as Granny anim → 90° tip.
        if name == "Dummy_Root":
            ori = _sanitize_dummy_root_ori(ori)
        scl = rewrite_channel(scl, scl_mode, need_scl, lambda s: s[2], upgradable=False)

        new_tracks.append(
            {
                "name": name,
                "index": tr.get("index", b.get("index")),
                "position": pos,
                "orientation": ori,
                "scale": scl,
            }
        )
        if (bi + 1) % track_batch == 0 or bi + 1 == len(skeleton):
            pct = 70 + int(18 * (bi + 1) / max(1, len(skeleton)))
            label = f"Building tracks {bi + 1}/{len(skeleton)}…"
            if progress:
                progress.step(pct, label)
            if cooperative:
                yield (pct, label)

    timer.add("build_tracks", time.perf_counter() - _t_tr)
    out_doc = dict(doc)
    out_doc["skeleton"] = skeleton
    out_doc["tracks"] = new_tracks
    export_duration = max(0.0, (float(frame_list[-1]) - 1.0) / float(fps))
    out_doc["duration"] = export_duration
    _trim_tracks_to_duration(new_tracks, export_duration)
    out_doc["blender_full_export"] = True
    out_doc["format"] = FORMAT_ID
    out_doc["version"] = int(out_doc.get("version") or 1)
    if not out_doc.get("run_id"):
        out_doc["run_id"] = "blender"
    # Forensic stamp for root-shift investigation (server writes EXPORT_META.json)
    try:
        euler = arm_obj.rotation_euler
        euler_x_deg = float(euler[0]) * 180.0 / 3.141592653589793
    except Exception:
        euler_x_deg = None
    y_up = False
    try:
        y_up = bool(arm_obj.get("gr2lab_y_up_display"))
    except Exception:
        y_up = False
    out_doc["export_meta"] = {
        "space": "granny" if granny_space else "blender_zup->granny",
        "granny_space": bool(granny_space),
        "y_up_display": y_up,
        "object_euler_x_deg": euler_x_deg,
        "frames_sampled": len(frame_list),
        "duration_sec": export_duration,
        "baked": bool(baked),
        "nla_eval": bool(nla_eval),
        "sampling": sampling_hint,
        "template_source": template_source,
        "fps": float(fps),
    }
    out_doc["export_space"] = out_doc["export_meta"]["space"]
    if progress:
        progress.step(92, "Writing GR2LAB document…")
    if cooperative:
        yield (92, "Writing GR2LAB document…")
    if filepath:
        Path(filepath).write_text(
            json.dumps(out_doc, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
    if stamped_native or template_source == "synthesize":
        stamp_gr2lab_on_armature(
            arm_obj,
            out_doc,
            fps=fps,
            y_up_display=False if template_source == "synthesize" else None,
        )
    elif active_was_synthesized and template_source != "active armature":
        stamp_gr2lab_on_armature(arm_obj, out_doc, fps=fps, y_up_display=False)
    if progress:
        progress.step(94, "GR2LAB document ready")
    stats = {
        "matched": matched,
        "missing": missing,
        "extras": extras,
        "template_source": template_source,
        "run_id": out_doc.get("run_id") or "",
        "stamped_native": stamped_native or template_source == "synthesize",
        "space": "granny" if granny_space else "blender_zup->granny",
        "frames_sampled": len(frame_list),
        "duration_sec": export_duration,
        "motion_delta_bones": _count_motion_delta_bones(delta_ref_doc, out_doc),
        "baked": baked,
        "nla_eval": bool(nla_eval),
        "sampling": sampling_hint,
        "eval_bones": len(need_eval_names),
        "doc": out_doc,
        "_stage_timer": timer,
    }
    if cooperative:
        return stats
    return stats


def _consume_export_gr2lab(gen_or_stats):
    """Run cooperative export generator to completion (or pass through dict)."""
    if isinstance(gen_or_stats, dict):
        return gen_or_stats
    last = None
    try:
        while True:
            last = next(gen_or_stats)
    except StopIteration as stop:
        stats = stop.value
        if isinstance(stats, dict):
            return stats
        raise RuntimeError("Export generator finished without stats") from stop
    raise RuntimeError(f"Export generator stalled at {last!r}")


def export_gr2lab_doc(
    context,
    *,
    arm_obj=None,
    template_armature=None,
    template_filepath: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
    bake_mode: str = "AUTO",
    cooperative: bool = False,
):
    """Build export document in memory (for GR2 Lab server encode).

    cooperative=True → returns a generator yielding (pct, label), final stats in StopIteration.value.
    """
    if cooperative:
        return export_gr2lab(
            context,
            "",
            arm_obj=arm_obj,
            template_armature=template_armature,
            template_filepath=template_filepath,
            progress=progress,
            bake_mode=bake_mode,
            cooperative=True,
        )
    stats = _consume_export_gr2lab(
        export_gr2lab(
            context,
            "",
            arm_obj=arm_obj,
            template_armature=template_armature,
            template_filepath=template_filepath,
            progress=progress,
            bake_mode=bake_mode,
            cooperative=True,
        )
    )
    return stats


# ---------------------------------------------------------------------------
# Blender operators
# ---------------------------------------------------------------------------

PROP_PACK_RUN = "gr2lab_full_pack_run"
PROP_SOURCE_GR2 = "gr2lab_full_source_gr2"
PROP_IMPORT_FP = "gr2lab_import_fp"
PROP_IMPORT_SNAPSHOT = "gr2lab_import_snapshot"
PROP_PARTNER_RUN = "gr2lab_partner_run"
PROP_PARTNER_OF = "gr2lab_partner_of"
PROP_PARTNER_PRIMARY = "gr2lab_partner_primary_run"


def _active_armature(context):
    """Prefer active armature; else first selected armature; else None.

    A sibling tail rig is viewport-only — export/bake use the body instead.
    """
    obj = getattr(context.view_layer.objects, "active", None)
    if obj and obj.type == "ARMATURE":
        return _unwrap_tail_armature(obj)
    for obj in getattr(context, "selected_objects", None) or []:
        if obj.type == "ARMATURE":
            return _unwrap_tail_armature(obj)
    return None


def _unwrap_tail_armature(obj):
    if obj is None or not obj.get("gr2lab_tail"):
        return obj
    body = bpy.data.objects.get(str(obj.get("gr2lab_tail_of") or ""))
    if body is not None and getattr(body, "type", None) == "ARMATURE":
        return body
    return obj


def _normalize_run_id(run_id: str) -> str:
    rid = (run_id or "").strip()
    if not rid or rid == "blender":
        return ""
    # Blender may rename duplicates: GR2LAB_nightfall.001
    if "." in rid and rid.rsplit(".", 1)[-1].isdigit():
        rid = rid.rsplit(".", 1)[0]
    return rid


def _pack_run_id(arm_obj) -> str:
    if not arm_obj:
        return ""
    rid = _normalize_run_id(str(arm_obj.get(PROP_PACK_RUN) or ""))
    if rid:
        return rid
    return _normalize_run_id(str(arm_obj.get(PROP_RUN) or ""))


def remember_pack_run(run_id: str, source_gr2: str = "") -> None:
    """Remember last Import GR2 dump so Export can pack even after retarget."""
    rid = _normalize_run_id(run_id)
    if not rid:
        return
    prefs = _addon_prefs()
    if prefs is not None and hasattr(prefs, "last_pack_run"):
        prefs.last_pack_run = rid
        if source_gr2 and hasattr(prefs, "last_source_gr2"):
            prefs.last_source_gr2 = str(source_gr2)
    try:
        wm = bpy.context.window_manager
        wm[WM_LAST_PACK_RUN] = rid
        if source_gr2:
            wm[WM_LAST_SOURCE_GR2] = str(source_gr2)
    except Exception:
        pass


def _remembered_pack_run() -> Tuple[str, str]:
    prefs = _addon_prefs()
    rid = ""
    src = ""
    if prefs is not None:
        rid = _normalize_run_id(str(getattr(prefs, "last_pack_run", "") or ""))
        src = str(getattr(prefs, "last_source_gr2", "") or "").strip()
    try:
        wm = bpy.context.window_manager
        rid = rid or _normalize_run_id(str(wm.get(WM_LAST_PACK_RUN) or ""))
        src = src or str(wm.get(WM_LAST_SOURCE_GR2) or "").strip()
    except Exception:
        pass
    return rid, src


def _iter_pack_armatures(context, *, exclude=None):
    """Scene armatures that carry a decode pack run id."""
    exclude_name = exclude.name if exclude else None
    for obj in context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        if exclude_name and obj.name == exclude_name:
            continue
        if _pack_run_id(obj):
            yield obj


def _iter_arm_actions(arm_obj):
    """Yield Action datablocks used by the armature (active + NLA strips)."""
    if not arm_obj or not getattr(arm_obj, "animation_data", None):
        return
    ad = arm_obj.animation_data
    if ad.action:
        yield ad.action
    for tr in ad.nla_tracks:
        for st in tr.strips:
            if getattr(st, "action", None):
                yield st.action


def _run_id_from_action_name(action) -> str:
    if not action:
        return ""
    name = str(getattr(action, "name", "") or "")
    if not name.startswith("GR2LAB_"):
        return ""
    return _normalize_run_id(name[len("GR2LAB_") :])


def _find_pack_donor(context, arm_obj) -> Tuple[Any, str, str]:
    """Find imported GR2LAB armature / run matching this arm's action or NLA.

    Returns (donor_arm_or_None, run_id, source_gr2).
    """
    if not context or not arm_obj:
        return None, "", ""
    actions = list(_iter_arm_actions(arm_obj))
    if not actions:
        return None, "", ""

    # Same Action datablock as an imported GR2LAB_* armature (usual paste workflow).
    for act in actions:
        for obj in context.scene.objects:
            if obj.type != "ARMATURE" or obj == arm_obj:
                continue
            oad = obj.animation_data
            if not oad or oad.action != act:
                continue
            rid = _pack_run_id(obj)
            if rid:
                return obj, rid, str(obj.get(PROP_SOURCE_GR2) or "").strip()

    # Action / object named GR2LAB_<run>
    for act in actions:
        rid = _run_id_from_action_name(act)
        if not rid:
            continue
        for obj in context.scene.objects:
            if obj.type != "ARMATURE" or obj == arm_obj:
                continue
            if _pack_run_id(obj) == rid:
                return obj, rid, str(obj.get(PROP_SOURCE_GR2) or "").strip()
            oname = obj.name or ""
            if oname == f"GR2LAB_{rid}" or oname.startswith(f"GR2LAB_{rid}."):
                orun = _pack_run_id(obj) or rid
                return obj, orun, str(obj.get(PROP_SOURCE_GR2) or "").strip()
        return None, rid, ""

    return None, "", ""


def _stamp_pack_from_donor(arm_obj, donor) -> None:
    """Copy dump/pack stamps from imported GR2LAB arm onto a working retarget arm."""
    if not arm_obj or not donor:
        return
    rid = _pack_run_id(donor)
    src = str(donor.get(PROP_SOURCE_GR2) or "").strip()
    stamp_pack_run(arm_obj, rid, src)
    for key in (
        PROP_IMPORT_SNAPSHOT,
        PROP_IMPORT_FP,
        "gr2lab_y_up_display",
        "gr2lab_source_interp",
        "gr2lab_densify_fps",
        "gr2lab_import_densified",
    ):
        if key in donor:
            try:
                arm_obj[key] = donor[key]
            except Exception:
                pass


def link_pack_from_action(context, arm_obj, *, force: bool = False) -> Tuple[str, str, str]:
    """Stamp pack dump onto arm from its GR2LAB_* action / donor. Returns (rid, src, note)."""
    if not arm_obj:
        return "", "", ""
    donor, rid, src = _find_pack_donor(context, arm_obj)
    current = _pack_run_id(arm_obj)
    if not rid:
        return current, str(arm_obj.get(PROP_SOURCE_GR2) or "").strip(), ""
    if not force and current == rid:
        return current, str(arm_obj.get(PROP_SOURCE_GR2) or src).strip(), "already linked"
    if donor:
        _stamp_pack_from_donor(arm_obj, donor)
        remember_pack_run(rid, str(arm_obj.get(PROP_SOURCE_GR2) or src))
        return rid, str(arm_obj.get(PROP_SOURCE_GR2) or src).strip(), f"linked from {donor.name}"
    stamp_pack_run(arm_obj, rid, src)
    remember_pack_run(rid, src)
    return rid, src, f"from action GR2LAB_{rid}"


def _decode_run_ids(base_url: str = "") -> set:
    """Legacy helper — prefer dump_still_exists(run_id) (no full cache scan)."""
    _apply_layout_from_prefs()
    from .local_pipeline import decode_run_ids
    try:
        return set(decode_run_ids())
    except Exception:
        return set()


def _pack_dump_exists(run_id: str) -> bool:
    """Cheap single-folder check (does not scan all dumps)."""
    if not (run_id or "").strip():
        return False
    _apply_layout_from_prefs()
    try:
        from .local_pipeline import dump_still_exists

        return bool(dump_still_exists(run_id))
    except Exception:
        return False

def resolve_pack_run(context, arm_obj=None) -> Tuple[str, str]:
    """
    Resolve (run_id, source_gr2) for Export GR2.

    Order:
      1) GR2LAB_* action / donor armature (NLA paste workflow) — overrides stale props
      2) stamps on the active arm
      3) single other stamped scene arm
      4) last Import GR2 remember
    """
    arm_obj = arm_obj or _active_armature(context)
    if context and arm_obj:
        rid, src, _note = link_pack_from_action(context, arm_obj, force=False)
        if rid:
            return rid, src

    rid = _pack_run_id(arm_obj)
    src = str(arm_obj.get(PROP_SOURCE_GR2) or "").strip() if arm_obj else ""
    if rid:
        return rid, src

    stamped = list(_iter_pack_armatures(context, exclude=arm_obj)) if context else []
    if len(stamped) == 1:
        other = stamped[0]
        return _pack_run_id(other), str(other.get(PROP_SOURCE_GR2) or "").strip()

    return _remembered_pack_run()


def stamp_pack_run(arm_obj, run_id: str, source_gr2: str = "") -> None:
    if not arm_obj:
        return
    rid = _normalize_run_id(run_id)
    if not rid:
        return
    arm_obj[PROP_PACK_RUN] = rid
    arm_obj[PROP_RUN] = rid
    if source_gr2:
        arm_obj[PROP_SOURCE_GR2] = source_gr2


def _server_url(context) -> str:
    return "local"



def _update_anim_browser(self, context):
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class GR2LabFullPreferences(bpy.types.AddonPreferences):
    bl_idname = "gr2lab_blender"

    cache_root: StringProperty(
        name="Cache folder",
        description="Local dump/encode cache (OUT_FILES / OUT_GR2). Empty = %TEMP%/gr2lab_blender_cache",
        default="",
        subtype="DIR_PATH",
    )
    bg3_unpacked_dir: StringProperty(
        name="BG3 unpacked data",
        description=(
            "Multitool UnpackedData (or the Gustav folder inside it). "
            "Timeline/scene pairs and Models GR2 bank. "
            "Also reads %LOCALAPPDATA%/GlassesMcp/tmlgen.json when present"
        ),
        default=r"D:\Igor\Adquantum\Scripts\El\Scripted_Mods\bg3-modders-multitool\UnpackedData",
        subtype="DIR_PATH",
    )
    body_base_dir: StringProperty(
        name="Body_Base folder",
        description="Folder of *_Base.GR2 bind poses. Empty = addon bundled Body_Base",
        default="",
        subtype="DIR_PATH",
    )
    body_base: StringProperty(
        name="Default Body base name",
        description="Optional default Body_Base GR2 name when Auto-match fails",
        default="",
    )
    y_up_display: BoolProperty(
        name="Y-up display on GR2 import",
        description="Rotate imported armature +90 deg X for Blender Z-up view",
        default=True,
    )
    anim_browser: BoolProperty(
        name="Animation browser",
        description="Show the cinematic clip strip at the bottom of the 3D view",
        default=True,
        update=_update_anim_browser,
    )
    add_rig: BoolProperty(
        name="Add rig",
        description="Import the GR2 as a rig. Turn this off to drop scene empties only",
        default=True,
    )
    scene_empties: BoolProperty(
        name="Scene empties on GR2 import",
        description="When this is on, importing a GR2 also drops start-position empties for the characters in the matching timeline scene",
        default=False,
    )
    center_parent: BoolProperty(
        name="Center top parent",
        description="Move the top parent empty to the middle of the scene. The other characters keep their spacing",
        default=False,
    )
    second_character: BoolProperty(
        name="Second character",
        description="Also import the other character's GR2 from the same folder. With Empties on, it is placed on its empty",
        default=True,
    )
    partner_body_base: StringProperty(
        name="Second body",
        description="Body base for the second character. Auto matches that GR2",
        default="AUTO",
    )
    partner_meshes: StringProperty(
        name="Second meshes",
        description="Mesh kit for the second character: NONE, BODY, or BODY_CLOTH",
        default="BODY_CLOTH",
    )
    partner_akelka: BoolProperty(
        name="Second AkELkA DGB",
        description="Use AkELkA Dragonborn meshes for the second character",
        default=True,
    )
    partner_tail: BoolProperty(
        name="Second tail",
        description="Add a tail armature for the second character",
        default=True,
    )
    partner_rig_setup: StringProperty(
        name="Second rig",
        description="ADVANCED or BASIC for the second character",
        default="ADVANCED",
    )
    partner_rig_shapes: BoolProperty(name="Second shapes", default=True)
    partner_rig_ik_limbs: BoolProperty(name="Second limbs IK", default=True)
    partner_rig_ik_head: BoolProperty(name="Second head IK", default=True)
    last_template: StringProperty(
        name="Last GR2LAB template",
        description="Last .gr2lab used as export template (auto-filled)",
        default="",
        subtype="FILE_PATH",
    )
    last_pack_run: StringProperty(
        name="Last pack dump",
        description="Decode run_id from last Import GR2",
        default="",
    )
    last_source_gr2: StringProperty(
        name="Last source GR2",
        description="Source .GR2 path from last Import GR2",
        default="",
    )
    mesh_kits_dir: StringProperty(
        name="Mesh kits folder",
        description="Folder with GR2LAB_mesh_kits.blend and catalog.json. Empty = addon mesh_kits/",
        default="",
        subtype="DIR_PATH",
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Support the development:")
        op = box.operator("wm.url_open", text="Support on Patreon", icon="FUND")
        op.url = "https://www.patreon.com/c/AkELkA"
        layout.separator()
        layout.label(text="Import → edit → Export. No server. Leave paths empty.", icon="INFO")
        layout.prop(self, "y_up_display")
        layout.label(text="Import runs in the viewport with a progress bar (Esc cancels)", icon="TIME")
        _draw_general_settings(layout, self)
        layout.operator(
            "gr2lab_full.root_gate",
            icon="VIEWZOOM",
            text="Check root motion (active armature)",
        )
        box = layout.box()
        box.label(text="Advanced (optional overrides)")
        box.prop(self, "cache_root")
        box.prop(self, "bg3_unpacked_dir")
        box.prop(self, "body_base_dir")
        box.prop(self, "body_base")
        box.prop(self, "mesh_kits_dir")
        # last_* are auto-filled by Import/Export — not for daily editing
        row = box.row()
        row.enabled = False
        row.prop(self, "last_pack_run")
        row = box.row()
        row.enabled = False
        row.prop(self, "last_source_gr2")
        row = box.row()
        row.enabled = False
        row.prop(self, "last_template")


_RIG_SETUP_ITEMS = (
    (
        "ADVANCED",
        "Advanced",
        "Bone widgets, limb IK, head HUD, optional tail after import",
    ),
    (
        "BASIC",
        "Basic",
        "Bones and animation as imported — no IK, shapes, or HUD",
    ),
)
_RIG_SETUP_UPDATING = False

_IMPORT_APPLY_ITEMS = (
    (
        "NEW_RIG",
        "Add rig",
        "Create a new armature from this GR2.",
    ),
    (
        "SWAP_RIG",
        "Swap rig",
        "Generate a new armature from this GR2 and use it instead of the selected similar rig. Copies that rig's NLA onto the new one.",
    ),
    (
        "REPLACE",
        "Replace Action",
        "Replace the current action on the selected rig.",
    ),
    (
        "ADD",
        "Add action",
        "Add a new action on the selected rig. Current action is pushed to NLA.",
    ),
    (
        "NLA",
        "Add NLA",
        "Add a new NLA strip on the selected rig.",
    ),
)

_IMPORT_MESHES_ITEMS = (
    (
        "NONE",
        "None",
        "Do not append body meshes",
    ),
    (
        "BODY",
        "Body",
        "Append weighted body meshes from the mesh kit",
    ),
    (
        "BODY_CLOTH",
        "Body+cloth",
        "Append body and cloth meshes from the mesh kit",
    ),
)

# Keys saved into File > Import / N-panel presets (skip if the holder has no RNA).
_IMPORT_PRESET_KEYS = (
    "import_apply",
    "import_meshes",
    "import_akelka",
    "import_clear_meshes",
    "rig_setup",
    "rig_shapes",
    "rig_ik_limbs",
    "rig_ik_head",
    "import_tail",
    "fps",
    "body_base",
    "add_rig",
    "scene_empties",
    "center_parent",
    "second_character",
    "partner_body_base",
    "partner_meshes",
    "partner_akelka",
    "partner_tail",
    "partner_rig_setup",
    "partner_rig_shapes",
    "partner_rig_ik_limbs",
    "partner_rig_ik_head",
)
_IMPORT_PRESET_SUBDIR = "gr2lab/import"
_IMPORT_PRESET_OPS = {
    "IMPORT_SCENE_OT_gr2_full",
    "IMPORT_SCENE_OT_gr2lab_full",
}
# Last holder that drew the preset row (file browser, drop dialog, or N-panel).
_PRESET_UI_HOLDER = None


def _capture_import_target(op, context) -> None:
    arm = _active_armature(context)
    op.target_armature = arm.name if arm else ""


def _resolve_import_target(op, context):
    arm = _active_armature(context)
    if arm is not None:
        return arm
    name = (getattr(op, "target_armature", "") or "").strip()
    if not name:
        return None
    obj = bpy.data.objects.get(name)
    if obj is not None and getattr(obj, "type", None) == "ARMATURE":
        return obj
    return None


def _update_rig_setup(self, context) -> None:
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING:
        return
    on = (getattr(self, "rig_setup", "ADVANCED") or "ADVANCED") != "BASIC"
    _RIG_SETUP_UPDATING = True
    try:
        self.rig_shapes = on
        self.rig_ik_limbs = on
        self.rig_ik_head = on
        self.import_tail = on
    finally:
        _RIG_SETUP_UPDATING = False


def _update_rig_part(self, context) -> None:
    """Turning any part on leaves Basic and shows Advanced; clearing all returns to Basic."""
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING:
        return
    any_on = (
        bool(getattr(self, "rig_shapes", False))
        or bool(getattr(self, "rig_ik_limbs", False))
        or bool(getattr(self, "rig_ik_head", False))
        or bool(getattr(self, "import_tail", False))
    )
    want = "ADVANCED" if any_on else "BASIC"
    if (getattr(self, "rig_setup", "") or "") == want:
        return
    _RIG_SETUP_UPDATING = True
    try:
        self.rig_setup = want
    finally:
        _RIG_SETUP_UPDATING = False


def _rig_parts_from_op(op) -> Dict[str, bool]:
    return {
        "shapes": bool(getattr(op, "rig_shapes", True)),
        "ik_hands": bool(getattr(op, "rig_ik_limbs", True)),
        "ik_legs": bool(getattr(op, "rig_ik_limbs", True)),
        "ik_head": bool(getattr(op, "rig_ik_head", True)),
    }


def _apply_creates_rig(apply: str) -> bool:
    return str(apply or "").strip().upper() in {"NEW_RIG", "SWAP_RIG"}


def _maybe_attach_mesh_kit(
    context,
    arm_obj,
    *,
    anim_apply: str,
    import_meshes: Optional[str] = None,
    import_akelka: bool = False,
    mesh_base: Optional[str] = None,
    doc: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
) -> None:
    if not _apply_creates_rig(anim_apply):
        return
    try:
        from .rig_display.mesh_kits import attach_mesh_kit

        report = attach_mesh_kit(
            context,
            arm_obj,
            base_id=mesh_base,
            mode=import_meshes or "NONE",
            akelka=bool(import_akelka),
            doc=doc,
            template_path=template_path,
        )
    except Exception:
        traceback.print_exc()
        print("[GR2 LAB] mesh kit attach failed (armature is intact)")
        return
    err = (report or {}).get("error")
    if err:
        print(f"[GR2 LAB] mesh kit skipped: {err}")


def _maybe_clear_swap_meshes(arm_obj) -> None:
    """Delete leftover character meshes after Swap (widgets / props stay)."""
    if arm_obj is None:
        return
    try:
        from .rig_display.mesh_kits import strip_connected_meshes

        report = strip_connected_meshes(arm_obj, keep_kit=False)
    except Exception:
        traceback.print_exc()
        print("[GR2 LAB] swap mesh clear failed (armature is intact)")
        return
    n = int((report or {}).get("count") or 0)
    if n:
        print(f"[GR2 LAB] swap cleared {n} meshes from {arm_obj.name!r}")


def _maybe_attach_tail_rig(
    context,
    arm_obj,
    *,
    anim_apply: str,
    import_tail: bool = False,
    mesh_base: Optional[str] = None,
    swap_from=None,
    rig_display: bool = True,
    rig_parts: Optional[Dict[str, bool]] = None,
    y_up_display: bool = True,
) -> None:
    if not _apply_creates_rig(anim_apply):
        return
    try:
        from .rig_display import tail_rig
    except Exception:
        traceback.print_exc()
        print("[GR2 LAB] tail rig module failed to load")
        return
    if swap_from is not None:
        try:
            tail_rig.remove_tail_rigs_of(swap_from)
        except Exception:
            traceback.print_exc()
    if not import_tail:
        return
    ident = mesh_base
    if not ident:
        props = getattr(context.scene, "gr2lab", None)
        ident = getattr(props, "body_base", None) if props is not None else None
    parts = rig_parts if isinstance(rig_parts, dict) else {}
    shapes = bool(parts.get("shapes", True)) if rig_display else False
    try:
        tail_rig.attach_tail_rig(
            context,
            arm_obj,
            ident,
            shapes=shapes,
            y_up_display=bool(y_up_display),
        )
    except Exception:
        traceback.print_exc()
        print("[GR2 LAB] tail rig attach failed (body armature is intact)")


def _apply_needs_target(apply: str) -> bool:
    return str(apply or "").strip().upper() != "NEW_RIG"


def _swap_match_bone_name(old_arm, new_arm) -> Optional[str]:
    old_names = set(old_arm.data.bones.keys())
    new_names = set(new_arm.data.bones.keys())
    for name in ("Dummy_Root", "Root_M"):
        if name in old_names and name in new_names:
            return name
    for name in new_arm.data.bones.keys():
        if name in old_names:
            return name
    return None


def _copy_arm_local_channels(old_arm, new_arm) -> None:
    """Same parent-relative seating. Do not assign matrix_world.

    A seated rig is scale (-1, 1, 1). Pushing that through matrix_world and
    back decomposes as (-1, -1, -1) plus a half-turn, and the new base mesh
    blows up.
    """
    new_arm.matrix_parent_inverse = old_arm.matrix_parent_inverse.copy()
    new_arm.rotation_mode = old_arm.rotation_mode
    new_arm.location = old_arm.location.copy()
    new_arm.scale = old_arm.scale.copy()
    new_arm.delta_location = old_arm.delta_location.copy()
    new_arm.delta_rotation_euler = old_arm.delta_rotation_euler.copy()
    new_arm.delta_scale = old_arm.delta_scale.copy()
    if old_arm.rotation_mode == "QUATERNION":
        new_arm.rotation_quaternion = old_arm.rotation_quaternion.copy()
    elif old_arm.rotation_mode == "AXIS_ANGLE":
        new_arm.rotation_axis_angle = old_arm.rotation_axis_angle[:]
    else:
        new_arm.rotation_euler = old_arm.rotation_euler.copy()


def _align_generated_rig_to(old_arm, new_arm) -> None:
    """Put the generated rig in the same slot as the one it replaces."""
    parent = old_arm.parent
    if parent is not None and (old_arm.parent_type or "OBJECT") != "BONE":
        new_arm.parent = parent
        new_arm.parent_type = "OBJECT"
        new_arm.parent_bone = ""
        _copy_arm_local_channels(old_arm, new_arm)
        return
    bone_name = _swap_match_bone_name(old_arm, new_arm)
    old_world = old_arm.matrix_world.copy()
    if bone_name:
        old_rest = old_arm.data.bones[bone_name].matrix_local.copy()
        new_rest = new_arm.data.bones[bone_name].matrix_local.copy()
        desired = old_world @ old_rest @ new_rest.inverted()
    else:
        desired = old_world.copy()
    new_arm.parent = None
    new_arm.matrix_world = desired
    if parent is None:
        return
    new_arm.parent = parent
    new_arm.parent_type = old_arm.parent_type or "OBJECT"
    if old_arm.parent_type == "BONE":
        new_arm.parent_bone = old_arm.parent_bone
    else:
        new_arm.parent_bone = ""
    new_arm.matrix_world = desired


def _reparent_keep_world(obj, new_parent, parent_bone: str, new_bones) -> None:
    mw = obj.matrix_world.copy()
    if parent_bone and parent_bone in new_bones:
        obj.parent_type = "BONE"
        obj.parent_bone = parent_bone
    else:
        obj.parent_type = "OBJECT"
        obj.parent_bone = ""
    obj.parent = new_parent
    obj.matrix_world = mw


def _mesh_skinned_to(obj, arm) -> bool:
    if getattr(obj, "type", None) != "MESH":
        return False
    for mod in getattr(obj, "modifiers", None) or []:
        if mod.type == "ARMATURE" and getattr(mod, "object", None) == arm:
            return True
    return False


def _retarget_constraint_target(con, old_arm, new_arm) -> int:
    n = 0
    if getattr(con, "target", None) == old_arm:
        con.target = new_arm
        n += 1
    for tgt in getattr(con, "targets", None) or []:
        if getattr(tgt, "target", None) == old_arm:
            tgt.target = new_arm
            n += 1
    return n


def _retarget_id_drivers(id_data, old_arm, new_arm) -> int:
    ad = getattr(id_data, "animation_data", None)
    if ad is None:
        return 0
    n = 0
    for fcu in list(getattr(ad, "drivers", None) or []):
        drv = getattr(fcu, "driver", None)
        if drv is None:
            continue
        for var in drv.variables:
            for tgt in var.targets:
                if tgt.id == old_arm:
                    tgt.id = new_arm
                    n += 1
    return n


def _owner_viewport_hidden(obj) -> bool:
    try:
        if obj.hide_get():
            return True
    except Exception:
        pass
    return bool(getattr(obj, "hide_viewport", False))


def _hide_like_owner(obj, owner) -> None:
    hidden = _owner_viewport_hidden(owner)
    try:
        obj.hide_set(hidden)
    except Exception:
        pass
    obj.hide_viewport = hidden


def _swap_kiss_widgets(old_arm, new_arm) -> None:
    """Move this character's Lips with the swap; do not dump other rigs' gizmos.

    Leftover Lips parented under the old arm but tagged to someone else go back
    to that owner. The swapped character's own Lips retarget onto the new arm
    (HUD duplicate is removed). Other scene Lips are left alone.
    """
    from .rig_display.head_ik import (
        _KISS_KEY,
        _TARGET,
        _kiss_name_for_arm,
        _parent_object_to_armature,
        ensure_kiss_object,
    )

    for obj in list(old_arm.children):
        owner_name = str(obj.get(_KISS_KEY) or "")
        if not owner_name or owner_name in {old_arm.name, new_arm.name}:
            continue
        owner = bpy.data.objects.get(owner_name)
        if owner is None or getattr(owner, "type", None) != "ARMATURE":
            continue
        _parent_object_to_armature(obj, owner)
        _hide_like_owner(obj, owner)

    old_lips = [
        obj
        for obj in bpy.data.objects
        if obj.get(_KISS_KEY) == old_arm.name
    ]
    hud_lips = [
        obj
        for obj in bpy.data.objects
        if obj.get(_KISS_KEY) == new_arm.name
    ]
    keep = None
    if old_lips:
        keep = old_lips[0]
        keep[_KISS_KEY] = new_arm.name
        try:
            keep.name = _kiss_name_for_arm(new_arm)
        except Exception:
            pass
        keep.hide_viewport = False
        try:
            keep.hide_set(False)
        except Exception:
            pass
        for extra in list(old_lips[1:]) + list(hud_lips):
            if extra == keep:
                continue
            try:
                bpy.data.objects.remove(extra, do_unlink=True)
            except Exception:
                pass

    lips = ensure_kiss_object(new_arm)
    pb = new_arm.pose.bones.get(_TARGET) if new_arm.pose is not None else None
    if lips is None or pb is None:
        return
    for cname in ("Copy Location", "Copy Rotation"):
        con = pb.constraints.get(cname)
        if con is not None and hasattr(con, "target"):
            con.target = lips


def _swap_scene_rig(context, old_arm, new_arm, keep_action=None) -> Dict[str, int]:
    """Retarget scene users of ``old_arm`` onto the generated ``new_arm``."""
    counts = {
        "parents": 0,
        "modifiers": 0,
        "constraints": 0,
        "drivers": 0,
        "nla_strips": 0,
    }
    if old_arm is None or new_arm is None or old_arm == new_arm:
        return counts
    if getattr(old_arm, "type", None) != "ARMATURE":
        return counts
    if getattr(new_arm, "type", None) != "ARMATURE":
        return counts

    new_bones = set(new_arm.data.bones.keys())
    _align_generated_rig_to(old_arm, new_arm)

    kit_reparent = None
    needs_stand = None
    try:
        from .rig_display.mesh_kits import (
            _needs_stand_rx90,
            _reparent_keep_world as _reparent_kit_mesh,
        )

        kit_reparent = _reparent_kit_mesh
        needs_stand = _needs_stand_rx90
    except Exception:
        traceback.print_exc()

    for obj in list(old_arm.children):
        if obj == new_arm:
            continue
        if obj.get("gr2lab_tail"):
            continue
        if obj.get("gr2lab_kiss_target"):
            continue
        if (
            kit_reparent is not None
            and needs_stand is not None
            and _mesh_skinned_to(obj, old_arm)
            and obj.parent_type != "BONE"
        ):
            kit_reparent(
                obj, new_arm, stand_rx90=needs_stand(obj, head_needs=False)
            )
        else:
            bone = obj.parent_bone if obj.parent_type == "BONE" else ""
            _reparent_keep_world(obj, new_arm, bone, new_bones)
        counts["parents"] += 1

    try:
        _swap_kiss_widgets(old_arm, new_arm)
    except Exception:
        traceback.print_exc()
        print("[GR2 LAB] swap Lips gizmo failed (armature is intact)")

    for obj in bpy.data.objects:
        if obj in (old_arm, new_arm):
            counts["drivers"] += _retarget_id_drivers(obj, old_arm, new_arm)
            continue
        for mod in getattr(obj, "modifiers", None) or []:
            if getattr(mod, "object", None) == old_arm:
                mod.object = new_arm
                counts["modifiers"] += 1
        for con in getattr(obj, "constraints", None) or []:
            counts["constraints"] += _retarget_constraint_target(
                con, old_arm, new_arm
            )
        if obj.type == "ARMATURE" and obj.pose is not None:
            for pb in obj.pose.bones:
                for con in pb.constraints:
                    counts["constraints"] += _retarget_constraint_target(
                        con, old_arm, new_arm
                    )
        counts["drivers"] += _retarget_id_drivers(obj, old_arm, new_arm)

    for coll in list(old_arm.users_collection):
        try:
            if new_arm.name not in coll.objects:
                coll.objects.link(new_arm)
        except RuntimeError:
            pass

    try:
        old_arm["gr2lab_swapped_to"] = new_arm.name
        new_arm["gr2lab_swapped_from"] = old_arm.name
    except Exception:
        pass

    try:
        counts["nla_strips"] = _copy_nla_stack(old_arm, new_arm, keep_action=keep_action)
    except Exception:
        traceback.print_exc()
        print("[GR2 LAB] swap NLA copy failed (generated action is intact)")

    print(
        f"[GR2 LAB] swap {old_arm.name!r} → {new_arm.name!r}: "
        f"{counts['parents']} parents, {counts['modifiers']} modifiers, "
        f"{counts['constraints']} constraints, {counts['drivers']} drivers, "
        f"{counts['nla_strips']} nla strips"
    )
    return counts


def _drop_unused_character_blocks() -> None:
    """Drop mesh, armature, and action data left behind by a replaced rig."""
    for block in list(bpy.data.armatures):
        if block.users == 0:
            bpy.data.armatures.remove(block)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.actions):
        if block.users == 0 and not getattr(block, "use_fake_user", False):
            bpy.data.actions.remove(block)


def _discard_replaced_rig(old_arm) -> None:
    """Delete a rig that swap already replaced. Hide is not enough; it stays in the file."""
    if old_arm is None:
        return
    try:
        name = old_arm.name
        data = old_arm.data
    except ReferenceError:
        return
    action = None
    anim = getattr(old_arm, "animation_data", None)
    if anim is not None:
        action = anim.action
    for obj in bpy.data.objects:
        if obj == old_arm:
            continue
        for mod in getattr(obj, "modifiers", None) or []:
            if getattr(mod, "object", None) == old_arm:
                try:
                    old_arm.hide_set(True)
                    old_arm.hide_viewport = True
                except Exception:
                    pass
                print(f"[GR2 LAB] kept replaced rig {name!r} (still used by {obj.name!r})")
                return
    for child in list(old_arm.children):
        try:
            bpy.data.objects.remove(child, do_unlink=True)
        except Exception:
            pass
    try:
        bpy.data.objects.remove(old_arm, do_unlink=True)
    except Exception:
        traceback.print_exc()
        try:
            old_arm.hide_set(True)
            old_arm.hide_viewport = True
        except Exception:
            pass
        return
    if data is not None and getattr(data, "users", 1) == 0:
        try:
            bpy.data.armatures.remove(data)
        except Exception:
            pass
    if (
        action is not None
        and getattr(action, "users", 1) == 0
        and not getattr(action, "use_fake_user", False)
    ):
        try:
            bpy.data.actions.remove(action)
        except Exception:
            pass
    _drop_unused_character_blocks()
    print(f"[GR2 LAB] removed replaced rig {name!r}")


def _import_preset_is_op(holder) -> bool:
    return getattr(holder, "bl_idname", "") in _IMPORT_PRESET_OPS


def _import_preset_holder(context):
    global _PRESET_UI_HOLDER
    if _PRESET_UI_HOLDER is not None:
        return _PRESET_UI_HOLDER
    op = getattr(context, "active_operator", None)
    if _import_preset_is_op(op):
        return op
    return getattr(getattr(context, "scene", None), "gr2lab", None)


def _snapshot_import_preset(holder) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if holder is None:
        return data
    for key in _IMPORT_PRESET_KEYS:
        if not hasattr(holder, key):
            continue
        try:
            data[key] = getattr(holder, key)
        except Exception:
            continue
    return data


def _assign_import_preset(holder, data: Dict[str, Any]) -> None:
    global _RIG_SETUP_UPDATING
    if holder is None or not data:
        return
    _RIG_SETUP_UPDATING = True
    try:
        for key in _IMPORT_PRESET_KEYS:
            if key not in data or not hasattr(holder, key):
                continue
            try:
                setattr(holder, key, data[key])
            except Exception:
                continue
    finally:
        _RIG_SETUP_UPDATING = False


def apply_import_preset(context, data: Dict[str, Any]) -> None:
    """Load a named import preset onto the file-browser operator and the N-panel."""
    holders = []
    seen = set()

    def _add(holder) -> None:
        if holder is None:
            return
        ident = id(holder)
        if ident in seen:
            return
        seen.add(ident)
        holders.append(holder)

    _add(_PRESET_UI_HOLDER)
    op = getattr(context, "active_operator", None)
    if _import_preset_is_op(op):
        _add(op)
    _add(getattr(getattr(context, "scene", None), "gr2lab", None))
    for holder in holders:
        _assign_import_preset(holder, data)


def _import_preset_current_name() -> str:
    try:
        label = bpy.types.GR2LAB_MT_import_presets.bl_label
    except Exception:
        return ""
    if not label or label in {"Import Presets", "Presets"}:
        return ""
    return label


def _import_preset_filepath(name: str) -> str:
    if not name:
        return ""
    path = bpy.utils.preset_find(name, _IMPORT_PRESET_SUBDIR, ext=".py")
    if not path:
        path = bpy.utils.preset_find(
            name, _IMPORT_PRESET_SUBDIR, display_name=True, ext=".py"
        )
    return path or ""


def _write_import_preset_file(filepath: str, data: Dict[str, Any]) -> None:
    text = (
        "import bpy\n"
        "from gr2lab_blender.bridge_blender import apply_import_preset\n"
        f"apply_import_preset(bpy.context, {repr(data)})\n"
    )
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(text)


def _draw_import_presets_row(layout, holder) -> None:
    global _PRESET_UI_HOLDER
    _PRESET_UI_HOLDER = holder
    current = _import_preset_current_name() or "Presets"
    col = layout.column(align=True)
    row = col.row(align=True)
    cluster = row.row(align=True)
    cluster.ui_units_x = 9
    cluster.menu("GR2LAB_MT_import_presets", text=current, icon="PRESET")
    cluster.operator("gr2lab_full.import_preset_add", text="", icon="ADD")
    cluster.operator("gr2lab_full.import_preset_add", text="", icon="REMOVE").remove_active = True
    actions = col.row(align=True)
    actions.enabled = current != "Presets"
    actions.operator("gr2lab_full.import_preset_update", text="Save")
    actions.operator("gr2lab_full.import_preset_rename", text="Rename")


def _draw_meshes_row(
    layout,
    holder,
    *,
    title: str = "",
    meshes_prop: str = "import_meshes",
    akelka_prop: str = "import_akelka",
) -> None:
    """Meshes dropdown and AkELkA toggle on one row."""
    if title:
        layout.label(text=title)
    split = layout.split(factor=0.5, align=True)
    split.prop(holder, meshes_prop, text="")
    split.prop(holder, akelka_prop, text="AkELkA DGB", toggle=True)


def _draw_body_base_row(
    layout,
    holder,
    body_prop: str,
    *,
    import_apply: str = "NEW_RIG",
) -> None:
    """Body base dropdown in the left half of the row (same width as mesh dropdown)."""
    row = layout.row(align=True)
    split = row.split(factor=0.5, align=True)
    split.prop(holder, body_prop, text="")


def _tail_toggle_enabled(holder) -> bool:
    base = str(getattr(holder, "body_base", "") or "").strip()
    if not base or base in {"NONE", "AUTO"}:
        hint = str(getattr(holder, "bb_hint", "") or "").strip()
        base = hint or base
    if not base or base in {"NONE", "AUTO"}:
        return True
    try:
        from bind_pose import tail_base_id_for_body  # type: ignore

        return tail_base_id_for_body(base) is not None
    except Exception:
        return False


def _draw_toggle_pair(col, holder, left_prop, left_text, right_prop, right_text, *, right_enabled: bool = True) -> None:
    """Two equal toggle buttons, flush with the rows above and below."""
    row = col.row(align=True)
    row.prop(holder, left_prop, text=left_text, toggle=True)
    if right_enabled:
        row.prop(holder, right_prop, text=right_text, toggle=True)
        return
    cell = row.row(align=True)
    cell.enabled = False
    cell.prop(holder, right_prop, text=right_text, toggle=True)


def _draw_rig_setup_toggles(layout, holder, *, title: str = ""):
    if title:
        layout.label(text=title)
    col = layout.column(align=True)
    _draw_toggle_pair(col, holder, "rig_shapes", "Shapes", "rig_ik_limbs", "Limbs IK")
    _draw_toggle_pair(
        col,
        holder,
        "rig_ik_head",
        "Head IK",
        "import_tail",
        "Tail",
    )
    return col


def _draw_fps_base_row(layout, op, context=None) -> None:
    """FPS plus optional Body Base dropdown on one row (no Base: label)."""
    if context is not None and hasattr(op, "body_base"):
        try:
            _sync_import_body_base(op, context)
        except Exception:
            pass
    row = layout.row(align=True)
    row.prop(op, "fps")
    if hasattr(op, "body_base"):
        row.prop(op, "body_base", text="")


def _partner_tail_enabled(holder) -> bool:
    base = str(getattr(holder, "partner_body_base", "") or "").strip()
    if not base or base in {"NONE", "AUTO"}:
        return True
    try:
        from bind_pose import tail_base_id_for_body  # type: ignore

        return tail_base_id_for_body(base) is not None
    except Exception:
        return True


def _update_partner_rig_setup(self, context) -> None:
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING:
        return
    on = (getattr(self, "partner_rig_setup", "ADVANCED") or "ADVANCED") != "BASIC"
    _RIG_SETUP_UPDATING = True
    try:
        self.partner_rig_shapes = on
        self.partner_rig_ik_limbs = on
        self.partner_rig_ik_head = on
        self.partner_tail = on
    finally:
        _RIG_SETUP_UPDATING = False


def _update_partner_rig_part(self, context) -> None:
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING:
        return
    any_on = (
        bool(getattr(self, "partner_rig_shapes", False))
        or bool(getattr(self, "partner_rig_ik_limbs", False))
        or bool(getattr(self, "partner_rig_ik_head", False))
        or bool(getattr(self, "partner_tail", False))
    )
    want = "ADVANCED" if any_on else "BASIC"
    if (getattr(self, "partner_rig_setup", "") or "") == want:
        return
    _RIG_SETUP_UPDATING = True
    try:
        self.partner_rig_setup = want
    finally:
        _RIG_SETUP_UPDATING = False


def _draw_partner_rig_toggles(layout, holder, *, title: str = "Settings of rig:") -> None:
    if title:
        layout.label(text=title)
    col = layout.column(align=True)
    _draw_toggle_pair(
        col, holder, "partner_rig_shapes", "Shapes", "partner_rig_ik_limbs", "Limbs IK"
    )
    _draw_toggle_pair(
        col,
        holder,
        "partner_rig_ik_head",
        "Head IK",
        "partner_tail",
        "Tail",
    )
    return col


def _partner_rig_parts(op) -> Dict[str, bool]:
    # Follow the four toggles. The Advanced/Basic enum is only a master switch
    # for those toggles; a stale Basic value must not hide shapes that are on.
    return {
        "shapes": bool(getattr(op, "partner_rig_shapes", True)),
        "ik_hands": bool(getattr(op, "partner_rig_ik_limbs", True)),
        "ik_legs": bool(getattr(op, "partner_rig_ik_limbs", True)),
        "ik_head": bool(getattr(op, "partner_rig_ik_head", True)),
    }


def _partner_wants_display(op) -> bool:
    return any(_partner_rig_parts(op).values())


def _draw_second_character_column(layout, holder, context=None) -> None:
    """Same block layout as the primary rig (toggles, meshes, body base in one column)."""
    layout.label(text="Settings of a 2nd rig:")
    col = _draw_partner_rig_toggles(layout, holder, title="")
    _draw_meshes_row(
        col,
        holder,
        meshes_prop="partner_meshes",
        akelka_prop="partner_akelka",
    )
    if hasattr(holder, "partner_body_base"):
        _draw_body_base_row(col, holder, "partner_body_base")


def _whole_scene(holder) -> bool:
    """Primary rig, the other character, stage empties, and the timeline camera."""
    apply = str(getattr(holder, "import_apply", "NEW_RIG") or "NEW_RIG").strip().upper()
    return apply == "NEW_RIG" and bool(getattr(holder, "second_character", False))


def _empties_only(holder) -> bool:
    """No rig mode selected."""
    if not hasattr(holder, "add_rig"):
        return False
    if bool(getattr(holder, "add_rig", True)) or bool(getattr(holder, "second_character", False)):
        return False
    apply = str(getattr(holder, "import_apply", "NEW_RIG") or "NEW_RIG").strip().upper()
    return apply == "NEW_RIG"


def _scene_place_allowed(holder) -> bool:
    """Stage empties belong to Whole scene, or a drop with no rig mode."""
    if _empties_only(holder):
        return True
    apply = str(getattr(holder, "import_apply", "NEW_RIG") or "NEW_RIG").strip().upper()
    adding = bool(getattr(holder, "add_rig", False)) or bool(
        getattr(holder, "second_character", False)
    )
    return apply == "NEW_RIG" and adding


def _draw_rig_mesh_settings(layout, op, context) -> None:
    apply = str(getattr(op, "import_apply", "NEW_RIG") or "NEW_RIG").strip().upper()
    if apply not in {"NEW_RIG", "SWAP_RIG"}:
        return
    col = _draw_rig_setup_toggles(layout, op, title="Settings of rig:")
    _draw_meshes_row(col, op)
    if hasattr(op, "body_base"):
        if context is not None:
            try:
                _sync_import_body_base(op, context)
            except Exception:
                pass
        _draw_body_base_row(col, op, "body_base", import_apply=apply)


def _draw_drop_dialog_title(col) -> None:
    """Title flush to top of popup content (no extra header row)."""
    col.label(text="GR2 LAB")


def _draw_fps_button(layout, holder) -> None:
    """Compact FPS (integer, same height as toggles)."""
    cell = layout.row(align=True)
    cell.ui_units_x = 6.5
    cell.prop(holder, "fps", text="FPS")


def _draw_scene_place_row(layout, holder, *, details: bool = False) -> None:
    """Whole scene: Centralize scene on the left, FPS on the right. Otherwise FPS only."""
    del details
    row = layout.row(align=True)
    if _whole_scene(holder) and hasattr(holder, "center_parent"):
        split = row.split(factor=0.5, align=True)
        left = split.row(align=True)
        left.prop(holder, "center_parent", text="Centralize scene", toggle=True)
        if hasattr(holder, "fps"):
            right = split.row(align=True)
            right.alignment = "RIGHT"
            _draw_fps_button(right, holder)
        return
    if hasattr(holder, "fps"):
        _draw_fps_button(row, holder)


def _draw_general_settings(
    layout, holder, context=None, *, details: bool = False
) -> None:
    layout.label(text="General settings:")
    col = layout.column(align=True)
    _draw_scene_place_row(col, holder, details=details)


def _update_import_apply(self, context) -> None:
    """Swap and the action buttons clear Add rig and Whole scene."""
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING or not hasattr(self, "add_rig"):
        return
    apply = str(getattr(self, "import_apply", "NEW_RIG") or "NEW_RIG")
    _RIG_SETUP_UPDATING = True
    try:
        if apply != "NEW_RIG":
            self.add_rig = False
            if hasattr(self, "second_character"):
                self.second_character = False
            if hasattr(self, "scene_empties"):
                self.scene_empties = False
        elif not bool(getattr(self, "second_character", False)):
            self.add_rig = True
            if hasattr(self, "scene_empties"):
                self.scene_empties = False
    finally:
        _RIG_SETUP_UPDATING = False


def _update_add_rig(self, context) -> None:
    """Add rig is one character. It turns Whole scene off."""
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING or not bool(getattr(self, "add_rig", True)):
        return
    _RIG_SETUP_UPDATING = True
    try:
        if hasattr(self, "second_character"):
            self.second_character = False
        if hasattr(self, "scene_empties"):
            self.scene_empties = False
        self.import_apply = "NEW_RIG"
    finally:
        _RIG_SETUP_UPDATING = False


def _update_second_character(self, context) -> None:
    """Whole scene imports both characters, empties, and the timeline camera."""
    global _RIG_SETUP_UPDATING
    if _RIG_SETUP_UPDATING:
        return
    if not bool(getattr(self, "second_character", False)):
        if hasattr(self, "scene_empties"):
            self.scene_empties = False
        return
    _RIG_SETUP_UPDATING = True
    try:
        self.add_rig = False
        self.import_apply = "NEW_RIG"
        if hasattr(self, "scene_empties"):
            self.scene_empties = True
    finally:
        _RIG_SETUP_UPDATING = False


def _draw_import_apply(layout, op, context) -> None:
    arm = _resolve_import_target(op, context)
    apply = getattr(op, "import_apply", "NEW_RIG") or "NEW_RIG"

    _draw_import_presets_row(layout, op)

    modes = layout.column(align=True)
    row = modes.row(align=True)
    if hasattr(op, "add_rig"):
        row.prop(op, "add_rig", text="Add rig", toggle=True)
    row.prop_enum(op, "import_apply", "SWAP_RIG")
    if hasattr(op, "second_character"):
        row.prop(op, "second_character", text="Whole scene", toggle=True)
    row = modes.row(align=True)
    row.prop_enum(op, "import_apply", "REPLACE")
    row.prop_enum(op, "import_apply", "ADD")
    row.prop_enum(op, "import_apply", "NLA")

    if _empties_only(op):
        return

    if apply in {"NEW_RIG", "SWAP_RIG"}:
        _draw_rig_mesh_settings(layout, op, context)

    if (
        apply == "NEW_RIG"
        and hasattr(op, "partner_meshes")
        and bool(getattr(op, "second_character", False))
    ):
        _draw_second_character_column(layout, op, context)
    if apply == "SWAP_RIG" and arm:
        layout.label(text=arm.name, icon="ARMATURE_DATA")
    if apply in {"REPLACE", "ADD", "NLA"}:
        if arm:
            layout.label(text=arm.name, icon="ARMATURE_DATA")
        else:
            layout.label(text="Select an armature first", icon="INFO")


class GR2LAB_MT_import_presets(bpy.types.Menu):
    bl_label = "Import Presets"
    preset_subdir = _IMPORT_PRESET_SUBDIR
    preset_operator = "script.execute_preset"
    draw = bpy.types.Menu.draw_preset


class GR2LAB_OT_import_preset_add(AddPresetBase, bpy.types.Operator):
    bl_idname = "gr2lab_full.import_preset_add"
    bl_label = "Add Import Preset"
    bl_description = "Save or remove a GR2 LAB import preset"
    preset_menu = "GR2LAB_MT_import_presets"
    preset_defines = []
    preset_values = []
    preset_subdir = _IMPORT_PRESET_SUBDIR

    def invoke(self, context, event):
        if not (self.remove_active or self.remove_name):
            holder = _import_preset_holder(context)
            self._preset_data = _snapshot_import_preset(holder)
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def add(self, context, filepath):
        data = getattr(self, "_preset_data", None)
        if not data:
            holder = _import_preset_holder(context)
            data = _snapshot_import_preset(holder)
        if not data:
            self.report({"ERROR"}, "No import settings to save")
            return
        _write_import_preset_file(filepath, data)

    def draw(self, context):
        self.layout.prop(self, "name")


class GR2LAB_OT_import_preset_update(bpy.types.Operator):
    bl_idname = "gr2lab_full.import_preset_update"
    bl_label = "Save"
    bl_description = "Overwrite the selected preset with the current import settings"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return bool(_import_preset_current_name())

    def execute(self, context):
        name = _import_preset_current_name()
        filepath = _import_preset_filepath(name)
        if not filepath:
            self.report({"ERROR"}, "Select a preset to save")
            return {"CANCELLED"}
        holder = _import_preset_holder(context)
        data = _snapshot_import_preset(holder)
        if not data:
            self.report({"ERROR"}, "No import settings to save")
            return {"CANCELLED"}
        _write_import_preset_file(filepath, data)
        self.report({"INFO"}, f'Saved preset "{name}"')
        return {"FINISHED"}


class GR2LAB_OT_import_preset_rename(bpy.types.Operator):
    bl_idname = "gr2lab_full.import_preset_rename"
    bl_label = "Rename"
    bl_description = "Rename the selected GR2 LAB import preset"
    bl_options = {"INTERNAL"}

    name: StringProperty(name="Name", maxlen=64)

    @classmethod
    def poll(cls, context):
        return bool(_import_preset_current_name())

    def invoke(self, context, event):
        self.name = _import_preset_current_name()
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        old = _import_preset_current_name()
        src = _import_preset_filepath(old)
        filename = AddPresetBase.as_filename((self.name or "").strip())
        if not src or not filename:
            self.report({"ERROR"}, "Select a preset to rename")
            return {"CANCELLED"}
        dest = os.path.join(os.path.dirname(src), filename + ".py")
        same = os.path.normcase(os.path.abspath(src)) == os.path.normcase(
            os.path.abspath(dest)
        )
        if not same and os.path.exists(dest):
            self.report({"ERROR"}, f'Preset "{filename}" already exists')
            return {"CANCELLED"}
        if not same:
            os.rename(src, dest)
        bpy.types.GR2LAB_MT_import_presets.bl_label = bpy.path.display_name(
            filename, title_case=False
        )
        return {"FINISHED"}

    def draw(self, context):
        self.layout.prop(self, "name")


def _stamp_action_source(arm, path: str, apply: str) -> None:
    if not arm or not path:
        return
    ad = getattr(arm, "animation_data", None)
    action = None
    if ad and ad.action:
        action = ad.action
    elif ad:
        for track in reversed(list(ad.nla_tracks or [])):
            strips = getattr(track, "strips", None) or []
            if strips:
                cand = strips[-1].action
                if cand is not None:
                    action = cand
                    break
    if action is None:
        return
    try:
        action["gr2lab_source_gr2"] = str(Path(path).resolve())
        action["gr2lab_import_apply"] = apply
    except Exception:
        pass


class GR2LABFULL_FH_gr2(bpy.types.FileHandler):
    """Drag-and-drop .GR2 into the 3D View → import via GR2 Lab."""

    bl_idname = "GR2LABFULL_FH_gr2"
    bl_label = "GR2 (GR2 Lab)"
    bl_import_operator = "import_scene.gr2_full"
    bl_file_extensions = ".gr2;.GR2"

    @classmethod
    def poll_drop(cls, context):
        area = getattr(context, "area", None)
        return area is not None and area.type in {"VIEW_3D", "OUTLINER"}


def _import_finish_primary_reports(op, context, state: dict) -> None:
    """User reports + prefs sync after primary rig is built (modal step 3)."""
    path = state["path"]
    dec = state["dec"]
    doc = state["doc"]
    arm = state["arm"]
    apply = state["apply"]
    target = state.get("target")
    base = state.get("base")
    prefs = state["prefs"]
    stale = state.get("stale_stage")
    if stale:
        op.report(
            {"INFO"},
            f"Scene placement: «{stale}» → {state.get('plan').stage.name} (re-drop updates empties)",
        )
    if state.get("partner_name"):
        op.report({"INFO"}, f"Partner GR2: {state['partner_name']}")
    run_id = dec.get("run_id") or doc.get("run_id") or ""
    if _apply_creates_rig(apply):
        arm[PROP_PACK_RUN] = run_id
        arm[PROP_SOURCE_GR2] = str(Path(path).resolve())
        try:
            for obj in context.view_layer.objects:
                obj.select_set(False)
            arm.select_set(True)
            context.view_layer.objects.active = arm
        except Exception:
            pass
        choice = str(state.get("body_choice") or "").strip()
        if not choice or choice == "user":
            choice = str(getattr(op, "body_choice", "") or "").strip()
        if choice and choice != "user":
            arm["gr2lab_body_choice"] = choice
        try:
            arm[PROP_IMPORT_FP] = json.dumps(
                {"bones": len(doc.get("skeleton") or []), "duration": doc.get("duration")}
            )
        except Exception:
            pass
    else:
        _stamp_action_source(arm, path, apply)
    if apply == "NEW_RIG" and arm is not None and not state.get("defer_clip_snap"):
        try:
            from . import anim_browser

            anim_browser.apply_pending_clip_drop(
                context, path, arm, consume=False
            )
        except Exception:
            traceback.print_exc()
    remember_pack_run(run_id, str(Path(path).resolve()))
    dens = doc.get("densified_channels")
    interp = doc.get("source_interp") or ""
    imp_d = arm.get("gr2lab_import_densified") if _apply_creates_rig(apply) else None
    if apply == "SWAP_RIG":
        src = arm.get("gr2lab_swapped_from") or (target.name if target else "?")
        msg = (
            f"Imported {Path(path).name} · swapped {src} → {arm.name} · "
            f"run={run_id}"
        )
    elif apply == "NEW_RIG":
        parts = _rig_parts_from_op(op)
        setup = "+".join(k for k, v in parts.items() if v) or "base"
        msg = (
            f"Imported {Path(path).name} · {setup} · run={run_id} · "
            f"{len(doc.get('skeleton') or [])} bones"
        )
    else:
        how = {
            "REPLACE": "replaced action",
            "ADD": "added action",
            "NLA": "NLA strip",
        }.get(apply, apply)
        msg = f"Imported {Path(path).name} · {how} on {arm.name} · run={run_id}"
    if interp:
        msg += f" · {interp}"
    if dens:
        msg += f" · densified={dens}"
    if imp_d:
        msg += f" · import-bake={imp_d}"
    n_mesh = arm.get("gr2lab_mesh_count") if arm else None
    if n_mesh:
        msg += f" · meshes={n_mesh}"
    op.report({"INFO"}, msg)
    timer = state.get("timer")
    if timer is not None:
        try:
            timer.print_block()
            dump_dir = Path(str(dec.get("files_dir") or dec.get("dump_dir") or ""))
            if dump_dir.is_dir():
                timer.write_json(dump_dir / "TIMING.json")
        except Exception:
            pass
        op.report({"INFO"}, timer.format_compact())
    if base:
        op.report({"INFO"}, f"Body base: {Path(base).name}")
    if dec.get("bind_warning"):
        bw = str(dec["bind_warning"])
        if dec.get("bind_optional"):
            op.report({"INFO"}, bw)
        else:
            op.report({"WARNING"}, bw)
    elif dec.get("bind"):
        op.report(
            {"INFO"},
            f"Bind T-pose OK ({(dec.get('bind') or {}).get('bones', '?')} bones)",
        )
    elif not dec.get("bind_optional"):
        op.report(
            {"WARNING"},
            "No T-pose bind — rest pose / NLA on other rigs will be wrong.",
        )
    prefs.add_rig = bool(op.add_rig)
    prefs.scene_empties = bool(op.scene_empties)
    prefs.center_parent = bool(op.center_parent)
    prefs.second_character = bool(op.second_character)
    prefs.partner_body_base = str(op.partner_body_base or "AUTO")
    prefs.partner_meshes = str(op.partner_meshes or "NONE")
    prefs.partner_akelka = bool(op.partner_akelka)
    prefs.partner_tail = bool(op.partner_tail)
    prefs.partner_rig_setup = str(op.partner_rig_setup or "ADVANCED")
    prefs.partner_rig_shapes = bool(op.partner_rig_shapes)
    prefs.partner_rig_ik_limbs = bool(op.partner_rig_ik_limbs)
    prefs.partner_rig_ik_head = bool(op.partner_rig_ik_head)
    remember_import_op_options(op)


class IMPORT_OT_gr2_full(bpy.types.Operator, ImportHelper):
    """Decode GR2 in-process, densify B-splines, import rig + animation (modal UI)."""

    bl_idname = "import_scene.gr2_full"
    bl_label = "GR2 (GR2 Lab)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".GR2"
    filter_glob: StringProperty(default="*.GR2;*.gr2", options={"HIDDEN"})
    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    fps: IntProperty(name="FPS", default=int(FPS_DEFAULT), min=1, max=120)
    body_base: EnumProperty(
        name="Body Base",
        description="Body_Base GR2 for bind pose and IK hierarchy (same as site Decode)",
        items=_import_body_base_enum_items,
        default=0,
        update=_update_import_body_base,
    )
    # Blender 5.x: do not use leading-underscore RNA property names
    bb_hint: StringProperty(name="Body Base Hint", default="", options={"HIDDEN"})
    body_choice: StringProperty(name="Body Choice", default="", options={"HIDDEN"})
    bb_synced_for: StringProperty(name="Body Base Synced", default="", options={"HIDDEN"})
    rig_setup: EnumProperty(
        name="Rig Setup",
        description="Advanced widgets+IK, or basic bones only",
        items=_RIG_SETUP_ITEMS,
        default="ADVANCED",
        update=_update_rig_setup,
    )
    rig_shapes: BoolProperty(
        name="Shapes",
        description="Bone widgets (boxes, hoops, IK targets)",
        default=True,
        update=_update_rig_part,
    )
    rig_ik_limbs: BoolProperty(
        name="Limbs IK",
        description="Wrist and ankle IK to Dummy hand/foot targets",
        default=True,
        update=_update_rig_part,
    )
    rig_ik_head: BoolProperty(
        name="Head IK",
        description="Head look-at IK, HUD sliders, and target widget",
        default=True,
        update=_update_rig_part,
    )
    import_apply: EnumProperty(
        name="Import",
        description="How to apply this GR2.",
        items=_IMPORT_APPLY_ITEMS,
        default="NEW_RIG",
        update=_update_import_apply,
    )
    import_meshes: EnumProperty(
        name="Meshes",
        description="Append a body mesh kit onto a new or swapped armature",
        items=_IMPORT_MESHES_ITEMS,
        default="BODY_CLOTH",
    )
    import_akelka: BoolProperty(
        name="AkELkA DGB",
        description="Use AkELkA Dragonborn meshes instead of the usual DGB body kit",
        default=True,
    )
    import_clear_meshes: BoolProperty(
        name="Clear meshes",
        description="Swap always replaces character meshes with the chosen body base kit",
        default=True,
    )
    import_tail: BoolProperty(
        name="Tail",
        description="Add a sibling tail armature and skin Tail_A to it (Tiefling / Dragonborn)",
        default=True,
        update=_update_rig_part,
    )
    target_armature: StringProperty(name="Target Armature", default="", options={"HIDDEN"})
    add_rig: BoolProperty(
        name="Add rig",
        description="Import this GR2 as a rig. Turn off to drop scene empties only",
        default=True,
        update=_update_add_rig,
    )
    scene_empties: BoolProperty(
        name="Scene empties",
        description="Whole scene drops start-position empties and the timeline camera",
        default=False,
    )
    center_parent: BoolProperty(
        name="Centralize scene",
        description="Move the top parent empty to the middle of the scene. The other characters keep their spacing",
        default=False,
    )
    second_character: BoolProperty(
        name="Whole scene",
        description="Import this GR2, the other character, stage empties, and the timeline camera",
        default=False,
        update=_update_second_character,
    )
    partner_body_base: EnumProperty(
        name="Second body",
        description="Body base for the second character. Auto matches that GR2",
        items=_partner_body_base_enum_items,
        default=0,
    )
    partner_meshes: EnumProperty(
        name="Second meshes",
        description="Mesh kit for the second character",
        items=_IMPORT_MESHES_ITEMS,
        default="BODY_CLOTH",
    )
    partner_akelka: BoolProperty(
        name="Second AkELkA DGB",
        description="Use AkELkA Dragonborn meshes for the second character",
        default=True,
    )
    partner_tail: BoolProperty(
        name="Second tail",
        description="Add a tail armature for the second character",
        default=True,
        update=_update_partner_rig_part,
    )
    partner_rig_setup: EnumProperty(
        name="Second rig",
        description="Advanced widgets and IK, or basic bones, for the second character",
        items=_RIG_SETUP_ITEMS,
        default="ADVANCED",
        update=_update_partner_rig_setup,
    )
    partner_rig_shapes: BoolProperty(
        name="Second shapes",
        description="Bone widgets for the second character",
        default=True,
        update=_update_partner_rig_part,
    )
    partner_rig_ik_limbs: BoolProperty(
        name="Second limbs IK",
        description="Wrist and ankle IK for the second character",
        default=True,
        update=_update_partner_rig_part,
    )
    partner_rig_ik_head: BoolProperty(
        name="Second head IK",
        description="Head look-at IK for the second character",
        default=True,
        update=_update_partner_rig_part,
    )

    def invoke(self, context, event):
        global _RIG_SETUP_UPDATING
        area = getattr(context, "area", None)
        self._view_area = (
            int(area.as_pointer())
            if area is not None and getattr(area, "type", "") == "VIEW_3D"
            else 0
        )
        _capture_import_target(self, context)
        prefs = context.preferences.addons[__package__].preferences
        _RIG_SETUP_UPDATING = True
        try:
            two = bool(getattr(prefs, "second_character", False))
            one = bool(getattr(prefs, "add_rig", True))
            if two and one:
                two = False
            self.add_rig = one and not two
            self.second_character = two
        finally:
            _RIG_SETUP_UPDATING = False
        self.scene_empties = bool(self.second_character)
        self.center_parent = bool(getattr(prefs, "center_parent", False))
        saved_body = str(getattr(prefs, "partner_body_base", "AUTO") or "AUTO")
        saved_meshes = str(getattr(prefs, "partner_meshes", "BODY_CLOTH") or "BODY_CLOTH")
        try:
            self.partner_body_base = saved_body
        except Exception:
            pass
        if saved_meshes in {"NONE", "BODY", "BODY_CLOTH"}:
            try:
                self.partner_meshes = saved_meshes
            except Exception:
                pass
        self.partner_akelka = bool(getattr(prefs, "partner_akelka", True))
        saved_setup = str(getattr(prefs, "partner_rig_setup", "ADVANCED") or "ADVANCED")
        _RIG_SETUP_UPDATING = True
        try:
            if saved_setup in {"ADVANCED", "BASIC"}:
                try:
                    self.partner_rig_setup = saved_setup
                except Exception:
                    pass
            self.partner_rig_shapes = bool(getattr(prefs, "partner_rig_shapes", True))
            self.partner_rig_ik_limbs = bool(getattr(prefs, "partner_rig_ik_limbs", True))
            self.partner_rig_ik_head = bool(getattr(prefs, "partner_rig_ik_head", True))
            self.partner_tail = bool(getattr(prefs, "partner_tail", True))
            any_on = (
                bool(self.partner_rig_shapes)
                or bool(self.partner_rig_ik_limbs)
                or bool(self.partner_rig_ik_head)
                or bool(self.partner_tail)
            )
            self.partner_rig_setup = "ADVANCED" if any_on else "BASIC"
        finally:
            _RIG_SETUP_UPDATING = False
        path = (self.filepath or "").strip()
        if path and Path(path).is_file():
            return context.window_manager.invoke_props_dialog(
                self,
                width=260,
                title="",
                translate=False,
            )
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        # Blender 5.x file selector / dialog cancel must return None, not a set.
        return None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False
        path = (self.filepath or "").strip()
        drop = bool(path and Path(path).is_file())
        col = layout.column(align=True) if drop else layout
        if drop:
            _draw_drop_dialog_title(col)
        _draw_import_apply(col, self, context)
        _draw_general_settings(col, self, context, details=True)

    def _cleanup_import_modal(self, context) -> None:
        from .import_progress_ui import end as overlay_end

        wm = context.window_manager
        if getattr(self, "_import_timer", None) is not None:
            try:
                wm.event_timer_remove(self._import_timer)
            except Exception:
                pass
            self._import_timer = None
        if getattr(self, "_progress", None) is not None:
            self._progress.end()
            self._progress = None
        overlay_end()

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup_import_modal(context)
            self.report({"WARNING"}, "Import cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        from .import_modal import advance

        finished, err = advance(
            self,
            context,
            self._import_state,
            progress=self._progress,
            bridge=sys.modules[__name__],
        )
        if err:
            self._cleanup_import_modal(context)
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        if finished:
            try:
                from . import anim_browser

                anim_browser.finish_deferred_edit_copy(context, self._import_state)
            except Exception:
                traceback.print_exc()
            self._cleanup_import_modal(context)
            return {"FINISHED"}
        _redraw_windows(context)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        path = self.filepath
        if not path or not Path(path).is_file():
            self.report({"ERROR"}, "Select a .GR2 file")
            return {"CANCELLED"}
        prefs = context.preferences.addons[__package__].preferences
        if _empties_only(self):
            prefs.add_rig = False
            prefs.scene_empties = bool(self.scene_empties)
            prefs.center_parent = bool(self.center_parent)
            prefs.second_character = bool(self.second_character)
            if not self.scene_empties:
                self.report({"WARNING"}, "Turn Add rig or Whole scene on")
                return {"CANCELLED"}
            try:
                from . import scene_stage_empties

                note = scene_stage_empties.place_for_import(
                    context,
                    path,
                    None,
                    y_up_display=bool(prefs.y_up_display),
                    parent_rig=False,
                    center_parent=bool(self.center_parent),
                )
                if note:
                    self.report({"INFO"}, note)
                else:
                    self.report({"WARNING"}, "No scene empties for this file")
            except Exception:
                traceback.print_exc()
                self.report({"WARNING"}, "Scene empties were not created")
            return {"FINISHED"}
        apply = (self.import_apply or "NEW_RIG").strip() or "NEW_RIG"
        target = _resolve_import_target(self, context)
        if _apply_needs_target(apply) and target is None:
            if apply == "SWAP_RIG":
                self.report({"ERROR"}, "Select an armature to swap")
            else:
                self.report(
                    {"ERROR"},
                    "Select an armature to apply animation, or choose Add rig",
                )
            return {"CANCELLED"}
        from .import_modal import stage_count
        from .import_progress_ui import begin as overlay_begin

        self._import_state = {
            "step": 0,
            "path": path,
            "apply": apply,
            "target": target,
            "prefs": prefs,
        }
        overlay_begin(
            stage_total=stage_count(self),
            context=context,
            area_ptr=int(getattr(self, "_view_area", 0) or 0),
        )
        self._progress = Gr2LabProgress(context, header=False)
        self._progress.begin(100)
        self._import_timer = context.window_manager.event_timer_add(
            0.02, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        print("[GR2 LAB] import: viewport bar only (Esc cancels, Blender stays usable)", flush=True)
        return {"RUNNING_MODAL"}

    def _import_second_character(
        self,
        context,
        path,
        prefs,
        parts,
        *,
        seat_on_empty: bool = False,
        progress: Optional[Gr2LabProgress] = None,
        partner_path: str = "",
        primary_plan=None,
        import_state: Optional[dict] = None,
        phase: str = "all",
        job_timer=None,
    ) -> None:
        from timing import maybe_timer

        from . import scene_stage_empties

        partner: Optional[Path] = None
        if partner_path:
            candidate = Path(partner_path)
            if candidate.is_file():
                partner = candidate
        if partner is None and phase != "build":
            partner = scene_stage_empties.find_partner_gr2(
                path, primary_plan=primary_plan
            )
        if partner is None and phase == "build" and import_state:
            raw = import_state.get("partner_path") or ""
            if raw and Path(raw).is_file():
                partner = Path(raw)
        if partner is None:
            if import_state is not None:
                import_state["partner_warn"] = True
            if phase != "build":
                self.report(
                    {"WARNING"},
                    "No second character GR2 next to this file",
                )
            return
        if phase == "decode" and import_state is not None:
            import_state["partner_path"] = str(partner)
            import_state["partner_name"] = partner.name
        pbase = _resolve_body_base_for_import(
            choice=str(getattr(self, "partner_body_base", "AUTO") or "AUTO"),
            bb_hint="",
            prefs_base="",
            source_gr2=str(partner),
        )
        if phase == "build" and import_state is not None:
            dec = import_state.get("partner_dec")
            doc = import_state.get("partner_doc")
            pbase = import_state.get("partner_pbase") or pbase
            if not isinstance(dec, dict) or doc is None:
                import_state["partner_warn"] = True
                return
        else:
            url = _server_url(context)
            dec, doc = import_gr2_pipeline(
                url, str(partner), body_base=pbase, progress=progress
            )
            if phase == "decode" and import_state is not None:
                import_state["partner_dec"] = dec
                import_state["partner_doc"] = doc
                import_state["partner_pbase"] = pbase
                return
        codec_timer = dec.get("timer") if isinstance(dec, dict) else None
        build_kwargs = dict(
            fps=self.fps,
            y_up_display=bool(prefs.y_up_display),
            template_path=None,
            progress=progress,
            template_doc=dec.get("sparse_doc") if isinstance(dec, dict) else None,
            timer=codec_timer,
            rig_display=_partner_wants_display(self),
            rig_parts=parts,
            target_arm=None,
            anim_apply="NEW_RIG",
            import_fast_head_ik=True,
            import_meshes=str(getattr(self, "partner_meshes", "NONE") or "NONE"),
            import_akelka=bool(getattr(self, "partner_akelka", False)),
            import_tail=bool(getattr(self, "partner_tail", True)),
            import_clear_meshes=False,
            mesh_base=pbase,
        )
        if import_state is None:
            with maybe_timer(job_timer, "partner_codec_wall"):
                arm = import_gr2lab_from_doc(context, doc, **build_kwargs)
        else:
            gen = import_state.get("partner_build_gen")
            if gen is None:
                seat_src = primary_plan or import_state.get("plan")
                if (
                    seat_src is not None
                    and import_state.get("partner_plan_thread") is None
                ):
                    import threading

                    box = import_state
                    ppath = str(partner)

                    def _run_partner_plan():
                        try:
                            box["partner_plan_result"] = (
                                scene_stage_empties.plan_scene_empties_at_stage(
                                    ppath, seat_src
                                )
                            )
                        except Exception:
                            box["partner_plan_error"] = traceback.format_exc()

                    th = threading.Thread(
                        target=_run_partner_plan,
                        name="gr2lab-partner-plan",
                        daemon=True,
                    )
                    import_state["partner_plan_thread"] = th
                    import_state["partner_plan_t0"] = time.perf_counter()
                    th.start()
                gen = _import_gr2lab_steps(context, doc, **build_kwargs)
                import_state["partner_build_gen"] = gen
            try:
                item = next(gen)
            except StopIteration as stop:
                arm = stop.value
                import_state.pop("partner_build_gen", None)
                import_state["partner_build_pending"] = False
            else:
                import_state["partner_slice"] = item
                import_state["partner_build_pending"] = True
                return
        if job_timer is not None and codec_timer is not None:
            job_timer.merge(codec_timer, prefix="partner_codec_")
        run_id = dec.get("run_id") or doc.get("run_id") or ""
        arm[PROP_PACK_RUN] = run_id
        arm[PROP_SOURCE_GR2] = str(partner.resolve())
        note = ""
        seat_primary = primary_plan
        primary_path = str(path)
        seat_snapshot = None
        if import_state is not None:
            if seat_primary is None:
                seat_primary = import_state.get("plan")
            primary_path = str(import_state.get("path") or path)
            seat_snapshot = import_state.get("seat_primary")
        partner_plan = None
        if import_state is not None:
            th = import_state.get("partner_plan_thread")
            if th is not None:
                th.join()
                err = import_state.get("partner_plan_error")
                if err:
                    print(err, flush=True)
                elif job_timer is not None:
                    job_timer.add(
                        "partner_plan_bg",
                        time.perf_counter()
                        - float(import_state.get("partner_plan_t0") or time.perf_counter()),
                    )
                partner_plan = import_state.get("partner_plan_result")
        if seat_on_empty:
            note = scene_stage_empties.parent_arm_to_scene_empty(
                context,
                str(partner),
                arm,
                y_up_display=bool(prefs.y_up_display),
                primary_plan=seat_primary,
                primary_gr2_path=primary_path,
                seat_snapshot=seat_snapshot,
                timer=job_timer,
                partner_plan=partner_plan,
            )
        stage_name = ""
        pplan = scene_stage_empties.resolve_primary_plan_for_partner_seat(
            seat_primary,
            primary_path,
            seat_snapshot=seat_snapshot,
        )
        if pplan is None:
            try:
                pplan = scene_stage_empties.plan_scene_empties(path)
            except Exception:
                pplan = None
        if pplan is not None:
            stage_name = pplan.stage.name
        suffix = f" · stage {stage_name}" if stage_name else ""
        self.report(
            {"INFO"},
            note or f"Second character: {partner.name}{suffix}",
        )


class EXPORT_OT_gr2_full(bpy.types.Operator, ExportHelper):
    """Encode GR2 in-process — modal TIMER so Blender stays responsive."""

    bl_idname = "export_scene.gr2_full"
    bl_label = "GR2 (GR2 Lab)"
    bl_options = {"REGISTER"}

    filename_ext = ".GR2"
    filter_glob: StringProperty(default="*.GR2;*.gr2", options={"HIDDEN"})
    arm_name: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    pack_run: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    source_gr2: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    bake_mode: EnumProperty(
        name="Bake before export",
        description=(
            "Visual flatten: sample NLA+constraints into one Action, then clear "
            "NLA tracks and constraints (destructive; Ctrl+Z undoes)"
        ),
        items=BAKE_MODE_ITEMS,
        default="AUTO",
    )

    def _resolve_live_pack(self, context, arm) -> Tuple[str, str]:
        """Resolve pack dump; if missing on disk, relink from GR2LAB_* action."""
        rid, src = resolve_pack_run(context, arm)
        if rid and not _pack_dump_exists(rid):
            # Stale custom prop after re-import under a new lyric suffix.
            if arm and arm.get(PROP_PACK_RUN) == rid:
                try:
                    del arm[PROP_PACK_RUN]
                except Exception:
                    arm[PROP_PACK_RUN] = ""
            link_pack_from_action(context, arm, force=True)
            rid, src = resolve_pack_run(context, arm)
            if rid and not _pack_dump_exists(rid):
                rem_rid, rem_src = _remembered_pack_run()
                if rem_rid and _pack_dump_exists(rem_rid):
                    stamp_pack_run(arm, rem_rid, rem_src)
                    rid, src = rem_rid, rem_src
        if arm and rid:
            stamp_pack_run(arm, rid, src)
            remember_pack_run(rid, src)
        return rid, src

    def invoke(self, context, event):
        arm = _active_armature(context)
        rid, src = self._resolve_live_pack(context, arm) if arm else ("", "")
        self.arm_name = arm.name if arm else ""
        self.pack_run = rid
        self.source_gr2 = src
        # Prefer a short default name in the current/home dir — do NOT open the
        # original IN_GR2 path (huge folders make the file browser crawl).
        stem = Path(src).stem if src else (Path(str(arm.get(PROP_SOURCE_GR2) or "export")).stem if arm else "export")
        if not stem:
            stem = "export"
        self.filepath = str(Path(stem + "_encoded.GR2"))
        # Cache UI bits once — draw() must stay cheap (file browser redraws a lot)
        self._ui_rid = rid
        self._ui_bake = False
        self._ui_hint = ""
        if arm:
            try:
                self._ui_bake = bool(_export_should_bake(arm, self.bake_mode))
            except Exception:
                self._ui_bake = False
            try:
                status, detail = peek_export_template_hint(context, arm)
                self._ui_hint = f"{status}" + (f" ({detail})" if detail else "")
            except Exception:
                self._ui_hint = ""
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        # Keep this trivial — Blender calls draw on every file-browser refresh.
        layout = self.layout
        rid = (getattr(self, "_ui_rid", None) or self.pack_run or "").strip()
        if rid:
            layout.label(text=f"Pack from dump: {rid}", icon="CHECKMARK")
        else:
            layout.label(text="No pack run — Import GR2 first", icon="ERROR")
        hint = getattr(self, "_ui_hint", "") or ""
        if hint:
            layout.label(text=f"Export source: {hint}")
        if getattr(self, "_ui_bake", False):
            layout.label(text="Will flatten NLA + constraints", icon="TIME")
        layout.prop(self, "bake_mode")
        layout.label(text="Encode runs in background (Esc cancels)")
    def execute(self, context):
        arm = None
        if self.arm_name and self.arm_name in bpy.data.objects:
            obj = bpy.data.objects[self.arm_name]
            if obj.type == "ARMATURE":
                arm = obj
        arm = arm or _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select the animated armature")
            return {"CANCELLED"}
        run_id, source_gr2 = self._resolve_live_pack(context, arm)
        source_gr2 = source_gr2 or str(arm.get(PROP_SOURCE_GR2) or "").strip() or None
        if not run_id:
            self.report(
                {"ERROR"},
                "No decode run_id — Import GR2 (via GR2 Lab), keep that action "
                "(GR2LAB_<dump>) on this armature / NLA, then Export again",
            )
            return {"CANCELLED"}
        stamp_pack_run(arm, run_id, source_gr2 or "")
        remember_pack_run(run_id, source_gr2 or "")
        self.pack_run = run_id
        self.source_gr2 = source_gr2 or ""

        self._arm_name = arm.name
        self._run_id = run_id
        self._source_gr2 = source_gr2
        self._server_url = _server_url(context)
        self._phase = "sample"
        self._net_thread = None
        self._net_result = None
        self._net_error = None
        self._stats = None
        self._progress = Gr2LabProgress(context)
        self._progress.begin(100)
        self._progress.step(1, "Starting export…")
        self._gen = export_gr2lab_doc(
            context,
            arm_obj=arm,
            progress=self._progress,
            bake_mode=self.bake_mode,
            cooperative=True,
        )
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _cleanup_timer(self, context):
        wm = context.window_manager
        if getattr(self, "_timer", None) is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        if getattr(self, "_progress", None) is not None:
            self._progress.end()
            self._progress = None

    def _start_network(self):
        stats = self._stats or {}
        doc = dict(stats.get("doc") or {})
        doc["run_id"] = self._run_id
        url = self._server_url
        run_id = self._run_id
        timer = stats.get("_stage_timer")
        dest = self.filepath
        source_gr2 = self._source_gr2

        def worker():
            try:
                self._net_result = pack_encode_gr2(
                    url,
                    run_id,
                    doc,
                    dest,
                    source_gr2=source_gr2,
                    progress=None,
                    timer=timer,
                )
            except Exception as ex:
                self._net_error = ex

        self._phase = "network"
        self._progress.step(86, "Encoding GR2…")
        self._net_thread = __import__("threading").Thread(target=worker, daemon=True)
        self._net_thread.start()

    def modal(self, context, event):
        if event.type == "ESC":
            gen = getattr(self, "_gen", None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
                self._gen = None
            self._cleanup_timer(context)
            self.report({"WARNING"}, "Export cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._phase == "sample":
            try:
                pct, label = next(self._gen)
                if self._progress:
                    self._progress.step(int(pct), label)
            except StopIteration as stop:
                self._stats = stop.value if isinstance(stop.value, dict) else None
                if not self._stats or "doc" not in self._stats:
                    self._cleanup_timer(context)
                    self.report({"ERROR"}, "Export produced no document")
                    return {"CANCELLED"}
                self._start_network()
            except Exception as e:
                traceback.print_exc()
                self._cleanup_timer(context)
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if self._phase == "network":
            if self._net_thread and self._net_thread.is_alive():
                if self._progress:
                    self._progress.step(90, "Encoding GR2…")
                return {"PASS_THROUGH"}
            self._cleanup_timer(context)
            if self._net_error is not None:
                err = self._net_error
                if isinstance(err, Gr2LabApiError):
                    self.report({"ERROR"}, str(err))
                else:
                    traceback.print_exc()
                    self.report({"ERROR"}, str(err))
                return {"CANCELLED"}
            enc = self._net_result or {}
            if enc.get("ok") is False:
                self.report(
                    {"ERROR"},
                    str(enc.get("error") or enc.get("message") or "Export refused"),
                )
                return {"CANCELLED"}
            imp = enc.get("import_result") or {}
            stats = self._stats or {}
            mode = enc.get("mode") or enc.get("badge") or "encoded"
            bones_applied = int(imp.get("bones_edited") or 0)
            ops_applied = int(imp.get("ops") or 0)
            apply_noop = bool(enc.get("apply_noop")) or not bool(imp.get("changed"))
            bake_note = (
                "baked" if stats.get("baked") else f"fcurve+eval{stats.get('eval_bones', 0)}"
            )
            self.report(
                {"INFO"},
                f"Wrote {Path(self.filepath).name} · dump={self._run_id} · {mode} · "
                f"{bake_note} · applied {bones_applied} bones / {ops_applied} ops · "
                f"{stats.get('matched', 0)} matched · {stats.get('frames_sampled', 0)} frames",
            )
            if apply_noop:
                self.report({"INFO"}, "Motion matched dump (noop apply) — GR2 still written")
            elif enc.get("warning"):
                self.report({"WARNING"}, str(enc["warning"]))
            if enc.get("hybrid_warning"):
                self.report({"INFO"}, str(enc["hybrid_warning"]))
            if enc.get("root_gate_warning"):
                self.report({"WARNING"}, str(enc["root_gate_warning"]))
            if stats.get("missing"):
                self.report(
                    {"WARNING"},
                    f"{stats['missing']} template bones missing — original tracks kept",
                )
            timer = enc.get("timer") or stats.get("_stage_timer")
            if timer is not None:
                try:
                    timer.print_block()
                except Exception:
                    pass
                self.report({"INFO"}, timer.format_compact())
            return {"FINISHED"}

        return {"PASS_THROUGH"}

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None



class IMPORT_OT_gr2lab_full(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.gr2lab_full"
    bl_label = "GR2LAB (.gr2lab)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".gr2lab"
    filter_glob: StringProperty(default="*.gr2lab;*.gr2lab.json", options={"HIDDEN"})
    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    fps: IntProperty(name="FPS", default=int(FPS_DEFAULT), min=1, max=120)
    rig_setup: EnumProperty(
        name="Rig Setup",
        description="Advanced widgets+IK, or basic bones only",
        items=_RIG_SETUP_ITEMS,
        default="ADVANCED",
        update=_update_rig_setup,
    )
    rig_shapes: BoolProperty(
        name="Shapes",
        description="Bone widgets (boxes, hoops, IK targets)",
        default=True,
        update=_update_rig_part,
    )
    rig_ik_limbs: BoolProperty(
        name="Limbs IK",
        description="Wrist and ankle IK to Dummy hand/foot targets",
        default=True,
        update=_update_rig_part,
    )
    rig_ik_head: BoolProperty(
        name="Head IK",
        description="Head look-at IK, HUD sliders, and target widget",
        default=True,
        update=_update_rig_part,
    )
    import_apply: EnumProperty(
        name="Import",
        description="How to apply this GR2.",
        items=_IMPORT_APPLY_ITEMS,
        default="NEW_RIG",
        update=_update_import_apply,
    )
    import_meshes: EnumProperty(
        name="Meshes",
        description="Append a body mesh kit onto a new or swapped armature",
        items=_IMPORT_MESHES_ITEMS,
        default="BODY_CLOTH",
    )
    import_akelka: BoolProperty(
        name="AkELkA DGB",
        description="Use AkELkA Dragonborn meshes instead of the usual DGB body kit",
        default=True,
    )
    import_clear_meshes: BoolProperty(
        name="Clear meshes",
        description="Swap always replaces character meshes with the chosen body base kit",
        default=True,
    )
    import_tail: BoolProperty(
        name="Tail",
        description="Add a sibling tail armature and skin Tail_A to it (Tiefling / Dragonborn)",
        default=True,
        update=_update_rig_part,
    )
    target_armature: StringProperty(name="Target Armature", default="", options={"HIDDEN"})

    def invoke(self, context, event):
        _capture_import_target(self, context)
        path = (self.filepath or "").strip()
        if path and Path(path).is_file():
            return context.window_manager.invoke_props_dialog(self, width=260)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False
        _draw_import_apply(layout, self, context)
        _draw_fps_base_row(layout, self)

    def execute(self, context):
        apply = (self.import_apply or "NEW_RIG").strip() or "NEW_RIG"
        target = _resolve_import_target(self, context)
        if _apply_needs_target(apply) and target is None:
            if apply == "SWAP_RIG":
                self.report({"ERROR"}, "Select an armature to swap")
            else:
                self.report(
                    {"ERROR"},
                    "Select an armature to apply animation, or choose Add rig",
                )
            return {"CANCELLED"}
        progress = Gr2LabProgress(context)
        progress.begin(100)
        try:
            progress.step(5, "Reading GR2LAB…")
            parts = _rig_parts_from_op(self)
            want_display = _apply_creates_rig(apply) and any(parts.values())
            arm = import_gr2lab(
                context,
                self.filepath,
                fps=self.fps,
                progress=progress,
                rig_display=want_display,
                rig_parts=parts,
                target_arm=target if _apply_needs_target(apply) else None,
                anim_apply=apply,
                import_meshes=getattr(self, "import_meshes", "NONE"),
                import_akelka=bool(getattr(self, "import_akelka", False)),
                import_tail=bool(getattr(self, "import_tail", False)),
                import_clear_meshes=bool(
                    getattr(self, "import_clear_meshes", False)
                ),
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        finally:
            progress.end()
        if apply == "SWAP_RIG":
            src = arm.get("gr2lab_swapped_from") or (target.name if target else "?")
            self.report(
                {"INFO"},
                f"Imported {Path(self.filepath).name} · swapped {src} → {arm.name}",
            )
        elif apply != "NEW_RIG":
            _stamp_action_source(arm, self.filepath, apply)
            how = {
                "REPLACE": "replaced action",
                "ADD": "added action",
                "NLA": "NLA strip",
            }.get(apply, apply)
            self.report({"INFO"}, f"Imported {Path(self.filepath).name} · {how} on {arm.name}")
        else:
            self.report({"INFO"}, f"Imported {Path(self.filepath).name}")
        return {"FINISHED"}


class EXPORT_OT_gr2lab_full(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.gr2lab_full"
    bl_label = "GR2LAB (.gr2lab)"
    bl_options = {"REGISTER"}

    filename_ext = ".gr2lab"
    filter_glob: StringProperty(default="*.gr2lab;*.gr2lab.json", options={"HIDDEN"})
    bake_mode: EnumProperty(
        name="Bake before export",
        description=(
            "Visual flatten: sample NLA+constraints into one Action, then clear "
            "NLA tracks and constraints (destructive; Ctrl+Z undoes)"
        ),
        items=BAKE_MODE_ITEMS,
        default="AUTO",
    )

    def draw(self, context):
        self.layout.prop(self, "bake_mode")

    def execute(self, context):
        arm = _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        progress = Gr2LabProgress(context)
        progress.begin(100)
        try:
            progress.step(2, "Exporting GR2LAB…", force_ui=True)
            stats = _consume_export_gr2lab(
                export_gr2lab(
                    context,
                    self.filepath,
                    arm_obj=arm,
                    progress=progress,
                    bake_mode=self.bake_mode,
                    cooperative=True,
                )
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        finally:
            progress.end()
        bake_note = "baked" if stats.get("baked") else f"fcurve+eval{stats.get('eval_bones', 0)}"
        self.report(
            {"INFO"},
            f"Exported · {stats.get('matched', 0)} matched · {bake_note} · "
            f"source={stats.get('template_source')}",
        )
        return {"FINISHED"}


class OBJECT_OT_gr2lab_bake_head_ik_target(bpy.types.Operator):
    """Key IK_Head_Target from frame-1 offset (match your manual frame-1 tweak)."""

    bl_idname = "object.gr2lab_bake_head_ik_target"
    bl_label = "GR2 Lab: Bake Head IK Target"
    bl_description = (
        "Use IK_Head_Target on the timeline start (usually frame 1) as the "
        "aim offset; keys the whole clip so IK on/off keeps Head_M pose"
    )
    bl_options = {"REGISTER", "UNDO"}

    anchor_frame: bpy.props.IntProperty(name="Anchor Frame", default=1, min=0)

    @classmethod
    def poll(cls, context):
        obj = _active_armature(context)
        return obj is not None and obj.pose.bones.get("IK_Head_Target") is not None

    def execute(self, context):
        obj = _active_armature(context)
        from .rig_display import head_ik as _head_ik

        n = _head_ik.bake_ik_head_target_from_action(
            context,
            obj,
            anchor_frame=int(self.anchor_frame),
        )
        if n <= 0:
            self.report({"WARNING"}, "No head IK keys written (need Action + Head IK)")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Head IK target: {n} keys (anchor frame {self.anchor_frame})")
        return {"FINISHED"}


class OBJECT_OT_gr2lab_duplicate_edit_rig(bpy.types.Operator):
    """Copy rig + meshes aside with one full-length NLA strip (not scene segment cuts)."""

    bl_idname = "object.gr2lab_duplicate_edit_rig"
    bl_label = "GR2 Lab: Duplicate edit rig (full NLA)"
    bl_description = (
        "Duplicate the GR2 LAB character (meshes and tail rig), move aside, "
        "and replace scene Hold/Seg NLA with one strip for the full GR2 action"
    )
    bl_options = {"REGISTER", "UNDO"}

    offset: bpy.props.FloatVectorProperty(
        name="Offset",
        size=3,
        default=(3.0, 0.0, 0.0),
        subtype="TRANSLATION",
        description="World offset for the duplicate",
    )
    scene_start: bpy.props.IntProperty(
        name="Timeline start",
        default=1,
        min=0,
        description="Scene frame where the full clip begins",
    )
    duplicate_action: bpy.props.BoolProperty(
        name="Copy Action",
        default=True,
        description="Duplicate the Action so edits do not affect the cinematic rig",
    )
    rewire_only: bpy.props.BoolProperty(
        name="Rewire only",
        default=False,
        description="Do not duplicate — only replace NLA on the selected rig",
    )

    @classmethod
    def poll(cls, context):
        arm = _active_armature(context)
        return arm is not None and bool(arm.get("gr2lab_run_id"))

    def execute(self, context):
        from . import scene_stage_empties

        arm = _active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select a GR2 LAB armature")
            return {"CANCELLED"}
        offset = tuple(float(v) for v in self.offset)
        if self.rewire_only:
            n = scene_stage_empties.rewire_gr2lab_rig_full_nla(
                arm,
                scene_start=int(self.scene_start),
                duplicate_action=bool(self.duplicate_action),
            )
            if n <= 0:
                self.report({"ERROR"}, "No GR2LAB action found on this rig")
                return {"CANCELLED"}
            self.report({"INFO"}, f"Full NLA on {n} armature(s) — {arm.name}")
            return {"FINISHED"}
        dup = scene_stage_empties.duplicate_gr2lab_edit_rig(
            context,
            arm,
            offset=offset,
            scene_start=int(self.scene_start),
            duplicate_action=bool(self.duplicate_action),
        )
        if dup is None:
            self.report({"ERROR"}, "Duplicate failed")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Edit copy: {dup.name} (full NLA, offset applied)")
        try:
            bpy.ops.object.select_all(action="DESELECT")
            dup.select_set(True)
            context.view_layer.objects.active = dup
        except Exception:
            pass
        return {"FINISHED"}


class OBJECT_OT_gr2lab_bake_nla_constraints(bpy.types.Operator):
    """Bake armature NLA + constraints into one Action, then clear NLA and constraints."""

    bl_idname = "object.gr2lab_bake_nla_constraints"
    bl_label = "GR2 Lab: Bake NLA + Constraints"
    bl_description = (
        "Sample evaluated pose (NLA + constraints), write one Action, "
        "then remove all NLA tracks and constraints"
    )
    bl_options = {"REGISTER", "UNDO"}

    frame_start: bpy.props.IntProperty(name="Start", default=1)
    frame_end: bpy.props.IntProperty(name="End", default=250)
    step: bpy.props.IntProperty(name="Step", default=1, min=1)
    auto_range: BoolProperty(
        name="Auto Frame Range",
        default=True,
        description="Detect frame range from NLA strips / Action",
    )
    bake_object: BoolProperty(
        name="Bake Object Transform",
        default=True,
        description="Also bake the armature object's visual Loc/Rot/Scale",
    )

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None

    def invoke(self, context, event):
        obj = _active_armature(context)
        fs, fe = determine_bake_range(obj, context.scene)
        self.frame_start = fs
        self.frame_end = fe
        self.bake_object = _has_object_motion_or_constraints(obj)
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "auto_range")
        col = layout.column(align=True)
        col.enabled = not self.auto_range
        col.prop(self, "frame_start")
        col.prop(self, "frame_end")
        layout.prop(self, "step")
        layout.prop(self, "bake_object")
        layout.separator()
        layout.label(text="Result: 1 Action, no NLA, no constraints.", icon="INFO")

    def execute(self, context):
        obj = _active_armature(context)
        if not obj:
            self.report({"ERROR"}, "Select an Armature")
            return {"CANCELLED"}

        prev_mode = obj.mode
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        try:
            bpy.ops.object.mode_set(mode="POSE")
        except Exception:
            pass

        fs = None if self.auto_range else self.frame_start
        fe = None if self.auto_range else self.frame_end
        try:
            gen = bake_armature_nla_and_constraints(
                context,
                obj,
                frame_start=fs,
                frame_end=fe,
                step=self.step,
                bake_object=self.bake_object,
                push_undo=False,  # operator already has UNDO
            )
            result = None
            try:
                while True:
                    next(gen)
            except StopIteration as stop:
                result = stop.value
            finally:
                try:
                    gen.close()
                except Exception:
                    pass
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        try:
            if prev_mode in {"POSE", "OBJECT", "EDIT"}:
                bpy.ops.object.mode_set(
                    mode=prev_mode if prev_mode != "EDIT" else "POSE"
                )
            else:
                bpy.ops.object.mode_set(mode="POSE")
        except Exception:
            pass

        if not result:
            self.report({"ERROR"}, "Bake produced no result")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            (
                f"Baked '{result['action']}' "
                f"[{result['frame_start']}-{result['frame_end']}] | "
                f"NLA tracks removed: {result['nla_tracks_removed']} | "
                f"Constraints removed: "
                f"{result['object_constraints_removed']}+{result['bone_constraints_removed']}"
            ),
        )
        return {"FINISHED"}


class OBJECT_OT_gr2lab_import_partner(bpy.types.Operator):
    bl_idname = "object.gr2lab_import_partner"
    bl_label = "GR2 Lab: Import scene partner"
    bl_description = "Not available in standalone addon yet (optional site tools)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        self.report(
            {"ERROR"},
            "Scene partner is not in the standalone addon yet. Use the optional site for dual-character preview.",
        )
        return {"CANCELLED"}


class OBJECT_OT_gr2lab_cinematic_sync(bpy.types.Operator):
    """Re-key Initiator / Additional empties from the dialog timeline (romance slot snaps)."""

    bl_idname = "object.gr2lab_cinematic_sync"
    bl_label = "GR2 Lab: Sync scene slots to timeline"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .scene_stage_empties import run_cinematic_sync_from_ui

        ok, msg = run_cinematic_sync_from_ui(context)
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        from .scene_stage_empties import collection_for_cinematic_sync

        return collection_for_cinematic_sync(context) is not None


class OBJECT_OT_gr2lab_glasses_preview(bpy.types.Operator):
    """Write dialog / stage / time for Glasses MCP (%LOCALAPPDATA%/GlassesMcp/gr2lab_preview.json)."""

    bl_idname = "object.gr2lab_glasses_preview"
    bl_label = "GR2 Lab: Write Glasses preview JSON"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from .scene_stage_empties import (
            _primary_plan_from_scene_collections,
            glasses_preview_from_plan,
            read_glasses_mcp_sidecar,
            write_glasses_mcp_sidecar,
        )

        preview = read_glasses_mcp_sidecar()
        if preview is None:
            plan = _primary_plan_from_scene_collections()
            if plan is None:
                self.report(
                    {"ERROR"},
                    "Import a scene GR2 with empties first (no gr2lab scene collection found)",
                )
                return {"CANCELLED"}
            preview = glasses_preview_from_plan(plan)
        path = write_glasses_mcp_sidecar(preview)
        if path is None:
            self.report({"ERROR"}, "Could not write GlassesMcp/gr2lab_preview.json")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            (
                f"Glasses: {preview.get('dialogName')} · {preview.get('sceneStage')} "
                f"@ {preview.get('timeSeconds')}s → {path}"
            ),
        )
        return {"FINISHED"}


class OBJECT_OT_gr2lab_link_dump(bpy.types.Operator):
    """Stamp GR2 Lab dump id onto the active armature from its GR2LAB_* action / NLA."""

    bl_idname = "object.gr2lab_link_dump"
    bl_label = "GR2 Lab: Link dump from action"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        rid, src, note = link_pack_from_action(context, arm, force=True)
        if not rid:
            self.report(
                {"ERROR"},
                "No GR2LAB_* action/NLA found — Import GR2 first, then paste that action here",
            )
            return {"CANCELLED"}
        ids = _decode_run_ids(_server_url(context))
        if ids and rid not in ids:
            self.report(
                {"WARNING"},
                f"Linked dump={rid} but it is not on GR2 Lab right now — re-import that GR2",
            )
        else:
            self.report({"INFO"}, f"Dump={rid}" + (f" · {note}" if note else ""))
        if src:
            self.report({"INFO"}, f"Source GR2: {src}")
        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None



class OBJECT_OT_gr2lab_root_gate(bpy.types.Operator):
    bl_idname = "gr2lab_full.root_gate"
    bl_label = "GR2 Lab: Check root motion"
    bl_description = "Compare source vs last encoded GR2 for Dummy_Root / Root_M shift"
    bl_options = {"REGISTER"}

    encoded_path: StringProperty(
        name="Encoded GR2",
        description="Encoded .GR2 to validate",
        default="",
        subtype="FILE_PATH",
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        self.layout.prop(self, "encoded_path")

    def execute(self, context):
        _apply_layout_from_prefs(context)
        arm = _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select a GR2 Lab armature")
            return {"CANCELLED"}
        rid = (arm.get(PROP_PACK_RUN) or "").strip()
        src = (arm.get(PROP_SOURCE_GR2) or "").strip()
        enc = (self.encoded_path or "").strip()
        if not src or not Path(src).is_file():
            self.report({"ERROR"}, "Armature missing source GR2 path — re-import first")
            return {"CANCELLED"}
        if not enc or not Path(enc).is_file():
            self.report({"ERROR"}, "Set Encoded GR2 path")
            return {"CANCELLED"}
        try:
            from .codec.root_gate import validate_root_export
            from .local_pipeline import resolve_dump_dir

            dump = None
            if rid:
                try:
                    dump = resolve_dump_dir(rid)
                except Exception:
                    dump = None
            report = validate_root_export(
                Path(src),
                Path(enc),
                dump=dump if dump and dump.is_dir() else None,
                allow_root_delta=False,
                expect_noop=False,
            )
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        ok = report.get("ok")
        badge = report.get("badge") or ("PASS" if ok else "FAIL")
        lat = report.get("max_root_lateral")
        msg = f"Root gate {badge}"
        if isinstance(lat, (int, float)):
            msg += f" · lateral={lat:.4f}m"
        if ok:
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        fails = report.get("fails") or []
        self.report(
            {"ERROR"},
            msg + ((" · " + "; ".join(map(str, fails[:3]))) if fails else ""),
        )
        return {"CANCELLED"}



def menu_import(self, context):
    self.layout.operator(IMPORT_OT_gr2_full.bl_idname, text="GR2 (GR2 Lab)")
    self.layout.operator(IMPORT_OT_gr2lab_full.bl_idname, text="GR2LAB (.gr2lab)")


def menu_export(self, context):
    self.layout.operator(EXPORT_OT_gr2_full.bl_idname, text="GR2 (GR2 Lab)")
    self.layout.operator(EXPORT_OT_gr2lab_full.bl_idname, text="GR2LAB (.gr2lab)")


def menu_object(self, context):
    self.layout.separator()
    self.layout.operator(
        OBJECT_OT_gr2lab_duplicate_edit_rig.bl_idname, icon="DUPLICATE"
    )
    self.layout.operator(OBJECT_OT_gr2lab_bake_nla_constraints.bl_idname, icon="ACTION")
    self.layout.operator(OBJECT_OT_gr2lab_bake_head_ik_target.bl_idname, icon="CON_KINEMATIC")
    self.layout.operator(OBJECT_OT_gr2lab_link_dump.bl_idname)
    self.layout.operator(OBJECT_OT_gr2lab_root_gate.bl_idname, icon="VIEWZOOM")
    self.layout.operator(OBJECT_OT_gr2lab_import_partner.bl_idname)
    self.layout.operator(OBJECT_OT_gr2lab_glasses_preview.bl_idname, icon="TIME")
    self.layout.operator(
        OBJECT_OT_gr2lab_cinematic_sync.bl_idname, icon="KEY_HLT"
    )
    self.layout.operator(
        OBJECT_OT_gr2lab_strip_meshes.bl_idname, icon="TRASH"
    )


_COLLECTION_PURPLE = "COLOR_06"
_BASES_FOLDER = "GR2LAB_Bases"


def _unlink_collection_parent(child, parent) -> None:
    try:
        parent.children.unlink(child)
    except Exception:
        pass


def _ensure_bases_folder(context):
    """Scene-root folder that holds every Body Base collection."""
    scene = context.scene.collection
    folder = bpy.data.collections.get(_BASES_FOLDER)
    if folder is None:
        folder = bpy.data.collections.new(_BASES_FOLDER)
    if scene.children.get(folder.name) is None:
        for other in list(bpy.data.collections):
            if other == folder:
                continue
            if other.children.get(folder.name) is not None:
                _unlink_collection_parent(folder, other)
        scene.children.link(folder)
    try:
        folder.color_tag = _COLLECTION_PURPLE
    except Exception:
        pass
    return folder


def _ensure_purple_collection(context, name: str):
    folder = _ensure_bases_folder(context)
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    scene = context.scene.collection
    if scene.children.get(coll.name) is not None:
        _unlink_collection_parent(coll, scene)
    for other in list(bpy.data.collections):
        if other in (folder, coll):
            continue
        if other.children.get(coll.name) is not None:
            _unlink_collection_parent(coll, other)
    if folder.children.get(coll.name) is None:
        folder.children.link(coll)
    try:
        coll.color_tag = _COLLECTION_PURPLE
    except Exception:
        pass
    return coll


def _npanel_body_base_items(self, context):
    global _NPANEL_BB_ITEMS, _NPANEL_BB_KEY
    try:
        _apply_layout_from_prefs(context)
        from bind_pose import list_bind_pose_catalog  # type: ignore

        ids = tuple(
            str(b.get("id") or "")
            for b in list_bind_pose_catalog()
            if b.get("id")
        )
    except Exception:
        ids = ()
    if ids == _NPANEL_BB_KEY:
        return _NPANEL_BB_ITEMS
    if ids:
        items = [(ident, ident, ident) for ident in ids]
    else:
        items = [("NONE", "(no Body_Base)", "Add Body_Base/*.bind_pose.json")]
    _NPANEL_BB_ITEMS = items
    _NPANEL_BB_KEY = ids
    return _NPANEL_BB_ITEMS


def _move_object_to_collection(obj, coll) -> None:
    if coll.objects.get(obj.name) is None:
        coll.objects.link(obj)
    for other in list(obj.users_collection):
        if other != coll:
            try:
                other.objects.unlink(obj)
            except Exception:
                pass


def _tag_gr2lab_view3d(self, context) -> None:
    wm = getattr(context, "window_manager", None) if context is not None else None
    if wm is None:
        return
    for window in wm.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class GR2LAB_PG_npanel(bpy.types.PropertyGroup):
    body_base: bpy.props.EnumProperty(
        name="Body Base",
        description="BG3 Body_Base bind pose (no animation)",
        items=_npanel_body_base_items,
    )
    import_meshes: bpy.props.EnumProperty(
        name="Meshes",
        description="Append a body mesh kit onto the created Body Base armature",
        items=_IMPORT_MESHES_ITEMS,
        default="BODY_CLOTH",
    )
    import_akelka: bpy.props.BoolProperty(
        name="AkELkA DGB",
        description="Use AkELkA Dragonborn meshes instead of the usual DGB body kit",
        default=True,
    )
    import_tail: bpy.props.BoolProperty(
        name="Tail",
        description="Add a sibling tail armature and skin Tail_A to it (Tiefling / Dragonborn)",
        default=True,
        update=_update_rig_part,
    )
    body_base_spacing: bpy.props.FloatProperty(
        name="Spacing",
        description="World XY cell size for Create all (grid on the floor)",
        default=2.0,
        min=0.25,
        max=20.0,
        unit="LENGTH",
        subtype="DISTANCE",
    )
    rig_setup: bpy.props.EnumProperty(
        name="Rig Setup",
        description="Advanced widgets+IK, or basic bones only",
        items=_RIG_SETUP_ITEMS,
        default="ADVANCED",
        update=_update_rig_setup,
    )
    rig_shapes: bpy.props.BoolProperty(
        name="Shapes",
        description="Bone widgets (boxes, hoops, IK targets)",
        default=True,
        update=_update_rig_part,
    )
    rig_ik_limbs: bpy.props.BoolProperty(
        name="Limbs IK",
        description="Wrist and ankle IK to Dummy hand/foot targets",
        default=True,
        update=_update_rig_part,
    )
    rig_ik_head: bpy.props.BoolProperty(
        name="Head IK",
        description="Head look-at IK, HUD sliders, Lips gizmo, and target widget",
        default=True,
        update=_update_rig_part,
    )
    race_overlay: bpy.props.BoolProperty(
        name="Swap Race",
        description="Draw the race chooser in every 3D view",
        default=True,
        update=_tag_gr2lab_view3d,
    )
    race_overlay_place: bpy.props.EnumProperty(
        name="Swap Race place",
        description="Race chooser above the selected rig, or fixed at the lower left of the view",
        items=(
            ("CHARACTER", "On character", "Above the selected rig"),
            ("CORNER", "Bottom left", "Fixed at the lower left of the view"),
        ),
        default="CHARACTER",
        update=_tag_gr2lab_view3d,
    )


def _npanel_base_ids(context) -> List[str]:
    _apply_layout_from_prefs(context)
    from bind_pose import list_bind_pose_catalog  # type: ignore

    return [
        str(b.get("id") or "")
        for b in list_bind_pose_catalog()
        if b.get("id") and str(b.get("id")) != "NONE"
    ]


def _create_body_base_armature(
    context,
    choice: str,
    *,
    import_meshes: str,
    import_akelka: bool = False,
    import_tail: bool = False,
):
    _apply_layout_from_prefs(context)
    from bind_pose import (  # type: ignore
        bind_pose_sidecar_path,
        gr2lab_doc_from_bind,
        load_bind_pose,
    )
    from .rig_display.head_ik import _KISS_KEY
    from .rig_display import tail_rig

    path = bind_pose_sidecar_path(choice)
    bind = load_bind_pose(path)
    doc = gr2lab_doc_from_bind(bind, run_id=choice)
    props = getattr(context.scene, "gr2lab", None)
    parts = _rig_parts_from_op(props) if props is not None else {
        "shapes": True,
        "ik_hands": True,
        "ik_legs": True,
        "ik_head": True,
    }
    arm = import_gr2lab_from_doc(
        context,
        doc,
        y_up_display=True,
        animation=False,
        import_meshes=import_meshes,
        import_akelka=bool(import_akelka),
        import_tail=bool(import_tail),
        mesh_base=choice,
        rig_display=any(parts.values()),
        rig_parts=parts,
    )
    coll = _ensure_purple_collection(context, choice)
    _move_object_to_collection(arm, coll)
    for child in list(arm.children):
        if child.get("gr2lab_mesh_kit") or child.get(_KISS_KEY) == arm.name:
            _move_object_to_collection(child, coll)
    for tail in tail_rig._iter_tail_rigs(arm):
        _move_object_to_collection(tail, coll)
        for child in list(tail.children):
            _move_object_to_collection(child, coll)
    return arm


class OBJECT_OT_gr2lab_strip_meshes(bpy.types.Operator):
    bl_idname = "gr2lab.strip_armature_meshes"
    bl_label = "Clear other meshes"
    bl_description = (
        "Delete leftover character meshes on the selected armature. "
        "Keeps the current kit, Lips, IK widgets, and bone-parented props"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _active_armature(context)
        if arm is None:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        try:
            from .rig_display.mesh_kits import strip_connected_meshes
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        report = strip_connected_meshes(arm, keep_kit=True)
        n = int((report or {}).get("count") or 0)
        self.report({"INFO"}, f"Removed {n} meshes from {arm.name}")
        return {"FINISHED"}


class OBJECT_OT_gr2lab_create_body_base(bpy.types.Operator):
    bl_idname = "gr2lab.create_body_base"
    bl_label = "Create"
    bl_description = "Add a Body_Base armature in bind pose, with no animation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = getattr(context.scene, "gr2lab", None)
        choice = (getattr(props, "body_base", "") or "").strip()
        if not choice or choice == "NONE":
            self.report({"ERROR"}, "No Body_Base selected")
            return {"CANCELLED"}
        try:
            arm = _create_body_base_armature(
                context,
                choice,
                import_meshes=getattr(props, "import_meshes", "NONE"),
                import_akelka=bool(getattr(props, "import_akelka", False)),
                import_tail=bool(getattr(props, "import_tail", False)),
            )
            context.view_layer.objects.active = arm
            arm.select_set(True)
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        n_mesh = arm.get("gr2lab_mesh_count") if arm else None
        extra = f" · meshes={n_mesh}" if n_mesh else ""
        self.report({"INFO"}, f"Created {choice} in collection {choice}{extra}")
        return {"FINISHED"}


class OBJECT_OT_gr2lab_create_all_body_bases(bpy.types.Operator):
    bl_idname = "gr2lab.create_all_body_bases"
    bl_label = "Create all"
    bl_description = "Add every Body_Base in a floor grid (world X/Y)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = getattr(context.scene, "gr2lab", None)
        try:
            ids = _npanel_base_ids(context)
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if not ids:
            self.report({"ERROR"}, "No Body_Base sidecars found")
            return {"CANCELLED"}
        meshes = getattr(props, "import_meshes", "NONE") if props else "NONE"
        akelka = bool(getattr(props, "import_akelka", False)) if props else False
        want_tail = bool(getattr(props, "import_tail", False)) if props else False
        spacing = float(getattr(props, "body_base_spacing", 2.0) or 2.0)
        cols = max(1, int(math.ceil(math.sqrt(len(ids)))))
        created: List[str] = []
        last = None
        choice = ""
        try:
            from .rig_display import tail_rig

            for i, choice in enumerate(ids):
                arm = _create_body_base_armature(
                    context,
                    choice,
                    import_meshes=meshes,
                    import_akelka=akelka,
                    import_tail=want_tail,
                )
                col = i % cols
                row = i // cols
                mw = arm.matrix_world.copy()
                loc = mw.translation.copy()
                loc.x = col * spacing
                loc.y = row * spacing
                mw.translation = loc
                arm.matrix_world = mw
                tail_rig.sync_tail_world(arm)
                created.append(arm.name)
                last = arm
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, f"{choice}: {e}")
            return {"CANCELLED"}
        _deselect_all_objects(context)
        for name in created:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True)
        if last is not None:
            context.view_layer.objects.active = last
        rows = int(math.ceil(len(created) / cols)) if created else 0
        self.report(
            {"INFO"},
            f"Created {len(created)} Body Bases · {cols}×{rows} grid · "
            f"spacing={spacing:g}m",
        )
        return {"FINISHED"}


class OBJECT_OT_gr2lab_glasses_play(bpy.types.Operator):
    """Play or stop Blender and the open Glasses timeline from the same frame."""

    bl_idname = "object.gr2lab_glasses_play"
    bl_label = "Play with Glasses"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        from .scene_stage_empties import glasses_link_is_active, glasses_linked_play_toggle

        if glasses_link_is_active(context):
            _ok, msg = glasses_linked_play_toggle(context, stop=True)
            self.report({"INFO"}, msg)
            return {"FINISHED"}
        ok, msg = glasses_linked_play_toggle(context, stop=False)
        if not ok:
            self.report({"WARNING"}, msg)
            return {"CANCELLED"}
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, msg)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        screen = context.screen
        if screen is not None and screen.is_animation_playing:
            return {"PASS_THROUGH"}
        from .scene_stage_empties import glasses_linked_play_toggle

        glasses_linked_play_toggle(
            context, stop=True, blender_already_stopped=True
        )
        self._remove_timer(context)
        return {"FINISHED"}

    def cancel(self, context):
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


class OBJECT_OT_gr2lab_glasses_sync(bpy.types.Operator):
    """Keep the Glasses timeline on the Blender frame while you scrub."""

    bl_idname = "object.gr2lab_glasses_sync"
    bl_label = "Sync with Glasses"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from .scene_stage_empties import glasses_frame_sync_toggle

        ok, msg = glasses_frame_sync_toggle(context)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class VIEW3D_PT_gr2lab_bases(bpy.types.Panel):
    bl_label = "GR2 LAB"
    bl_idname = "VIEW3D_PT_gr2lab_bases"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Akelka Tools"

    def draw(self, context):
        layout = self.layout
        props = getattr(context.scene, "gr2lab", None)
        col = layout.column(align=True)
        col.label(text="Body Base")
        if props is not None:
            col.prop(props, "body_base", text="")
            _draw_meshes_row(col, props, title="Mesh settings")
            _draw_rig_setup_toggles(col, props, title="Settings of rig:")
        row = col.row(align=True)
        row.operator(
            OBJECT_OT_gr2lab_create_body_base.bl_idname,
            text="Create",
            icon="ARMATURE_DATA",
        )
        row.operator(
            OBJECT_OT_gr2lab_create_all_body_bases.bl_idname,
            text="Create all",
        )
        if props is not None:
            col.prop(props, "body_base_spacing", text="Spacing")

        layout.separator()
        gpu = layout.column(align=True)
        if props is not None:
            gpu.prop(
                props,
                "race_overlay",
                text="Swap Race" if props.race_overlay else "Swap Race off",
                toggle=True,
            )
            place = gpu.row(align=True)
            place.enabled = bool(props.race_overlay)
            place.prop_enum(props, "race_overlay_place", "CHARACTER")
            place.prop_enum(props, "race_overlay_place", "CORNER")

        layout.separator()
        browser = layout.column(align=True)
        addon = context.preferences.addons.get(__package__) if context else None
        prefs = addon.preferences if addon is not None else None
        if prefs is not None and hasattr(prefs, "anim_browser"):
            browser.prop(
                prefs,
                "anim_browser",
                text="Animation browser" if prefs.anim_browser else "Animation browser off",
                toggle=True,
            )
        browser.operator("gr2lab.anim_prepare", text="Prepare animation library", icon="FILEBROWSER")
        browser.operator(
            "gr2lab.anim_edit_drop",
            text="Duplicate for edit (browser)",
            icon="IMPORT",
        )
        browser.operator(
            OBJECT_OT_gr2lab_duplicate_edit_rig.bl_idname,
            text="Duplicate edit rig (full NLA)",
            icon="DUPLICATE",
        )
        layout.separator()
        from .scene_stage_empties import glasses_frame_sync_is_active

        syncing = glasses_frame_sync_is_active(context)
        if syncing:
            from .scene_stage_empties import _glasses_sync_timer_ensure

            _glasses_sync_timer_ensure()
        layout.operator(
            OBJECT_OT_gr2lab_glasses_sync.bl_idname,
            text="Stop sync with Glasses" if syncing else "Sync with Glasses",
            icon="LINKED" if syncing else "UNLINKED",
        )


classes = (
    GR2LabFullPreferences,
    GR2LABFULL_FH_gr2,
    GR2LAB_MT_import_presets,
    GR2LAB_OT_import_preset_add,
    GR2LAB_OT_import_preset_update,
    GR2LAB_OT_import_preset_rename,
    IMPORT_OT_gr2_full,
    EXPORT_OT_gr2_full,
    IMPORT_OT_gr2lab_full,
    EXPORT_OT_gr2lab_full,
    OBJECT_OT_gr2lab_bake_head_ik_target,
    OBJECT_OT_gr2lab_duplicate_edit_rig,
    OBJECT_OT_gr2lab_bake_nla_constraints,
    OBJECT_OT_gr2lab_link_dump,
    OBJECT_OT_gr2lab_root_gate,
    OBJECT_OT_gr2lab_import_partner,
    OBJECT_OT_gr2lab_glasses_preview,
    OBJECT_OT_gr2lab_glasses_play,
    OBJECT_OT_gr2lab_glasses_sync,
    OBJECT_OT_gr2lab_cinematic_sync,
    GR2LAB_PG_npanel,
    OBJECT_OT_gr2lab_strip_meshes,
    OBJECT_OT_gr2lab_create_body_base,
    OBJECT_OT_gr2lab_create_all_body_bases,
    VIEW3D_PT_gr2lab_bases,
)


def register():
    GR2LAB_MT_import_presets.bl_label = "Import Presets"
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gr2lab = bpy.props.PointerProperty(type=GR2LAB_PG_npanel)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)
    bpy.types.VIEW3D_MT_object.append(menu_object)
    from .scene_stage_empties import glasses_frame_sync_handlers_install

    glasses_frame_sync_handlers_install()
    from . import anim_browser
    from . import race_picker

    anim_browser.register()
    race_picker.register()


def unregister():
    from .scene_stage_empties import (
        _glasses_sync_timer_stop,
        glasses_frame_sync_handlers_remove,
    )

    try:
        glasses_frame_sync_handlers_remove()
        _glasses_sync_timer_stop()
    except Exception:
        pass
    try:
        from . import race_picker

        from . import anim_browser

        anim_browser.unregister()
        race_picker.unregister()
    except Exception:
        pass
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.VIEW3D_MT_object.remove(menu_object)
    if hasattr(bpy.types.Scene, "gr2lab"):
        del bpy.types.Scene.gr2lab
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
