"""Viewport overlay while GR2 LAB import runs in a modal operator."""

from __future__ import annotations

from typing import Optional, Tuple

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

_DRAW_HANDLE = None
_STATE = {
    "active": False,
    "pct": 0.0,
    "title": "GR2 LAB",
    "label": "",
    "detail": "",
    "stage": 0,
    "stage_total": 1,
    "area": 0,
}


def _area_ptr(area) -> int:
    try:
        return int(area.as_pointer()) if area is not None else 0
    except Exception:
        return 0


def _view3d_areas(context=None):
    ctx = context or bpy.context
    found = []
    win = getattr(ctx, "window", None)
    screen = getattr(win, "screen", None) if win is not None else None
    screens = [screen] if screen is not None else []
    wm = getattr(ctx, "window_manager", None) or bpy.context.window_manager
    if not screens:
        for window in getattr(wm, "windows", []) or []:
            scr = getattr(window, "screen", None)
            if scr is not None:
                screens.append(scr)
    for scr in screens:
        for area in scr.areas:
            if area.type == "VIEW_3D" and area not in found:
                found.append(area)
    return found


def resolve_view3d_area(context=None, area_ptr: int = 0):
    """The 3D view the user dropped into or is using. Not every split."""
    ctx = context or bpy.context
    areas = _view3d_areas(ctx)
    want = int(area_ptr or 0)
    if not want:
        area = getattr(ctx, "area", None)
        if area is not None and getattr(area, "type", "") == "VIEW_3D":
            want = _area_ptr(area)
    if not want:
        try:
            from . import race_picker

            want = int(race_picker.active_area_ptr() or 0)
        except Exception:
            want = 0
    if want:
        for area in areas:
            if _area_ptr(area) == want:
                return area
    if len(areas) == 1:
        return areas[0]
    best = None
    best_px = -1
    for area in areas:
        px = 0
        for region in area.regions:
            if region.type == "WINDOW":
                px = max(px, int(region.width) * int(region.height))
        if px > best_px:
            best = area
            best_px = px
    return best


def _tag_area(area_ptr: int = 0) -> None:
    ptr = int(area_ptr or _STATE.get("area") or 0)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            if ptr and _area_ptr(area) != ptr:
                continue
            area.tag_redraw()


def begin(title: str = "GR2 LAB", stage_total: int = 6, context=None, area_ptr: int = 0) -> None:
    global _DRAW_HANDLE
    area = resolve_view3d_area(context, area_ptr)
    _STATE.update(
        {
            "active": True,
            "pct": 0.0,
            "title": title,
            "label": "Starting…",
            "detail": "",
            "stage": 0,
            "stage_total": max(1, int(stage_total)),
            "area": _area_ptr(area),
        }
    )
    if area is not None:
        try:
            from . import race_picker

            race_picker.note_area(area)
        except Exception:
            pass
    if _DRAW_HANDLE is None:
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay,
            (),
            "WINDOW",
            "POST_PIXEL",
        )
    _tag_area()


def end() -> None:
    global _DRAW_HANDLE
    _STATE["active"] = False
    ptr = int(_STATE.get("area") or 0)
    _STATE["area"] = 0
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        except Exception:
            pass
        _DRAW_HANDLE = None
    _tag_area(ptr)


def update(
    pct: float,
    label: str,
    *,
    detail: str = "",
    stage: Optional[int] = None,
) -> None:
    _STATE["pct"] = max(0.0, min(1.0, float(pct)))
    _STATE["label"] = label or ""
    if detail:
        _STATE["detail"] = detail
    if stage is not None:
        _STATE["stage"] = int(stage)
    _tag_area()


def _truncate_middle(text: str, max_px: float, font_id: int, size: int) -> str:
    if not text:
        return ""
    blf.size(font_id, size)
    if blf.dimensions(font_id, text)[0] <= max_px:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        chunk = text[:mid] + ell
        if blf.dimensions(font_id, chunk)[0] <= max_px:
            lo = mid + 1
        else:
            hi = mid
    cut = max(0, lo - 1)
    return (text[:cut] + ell) if cut else ell


def _panel_rect(region) -> Tuple[float, float, float, float]:
    w = float(region.width)
    h = float(region.height)
    pw = min(520.0, w * 0.55)
    ph = 96.0
    x = (w - pw) * 0.5
    y = h * 0.22
    return x, y, pw, ph


def _draw_rect(x, y, w, h, color):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
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


def _draw_overlay():
    if not _STATE.get("active"):
        return
    region = bpy.context.region
    area = getattr(bpy.context, "area", None)
    if region is None or area is None or getattr(area, "type", "") != "VIEW_3D":
        return
    want = int(_STATE.get("area") or 0)
    if want and _area_ptr(area) != want:
        return
    x, y, pw, ph = _panel_rect(region)
    _draw_rect(x, y, pw, ph, (0.04, 0.05, 0.08, 0.88))
    _draw_rect(x, y + ph - 3.0, pw, 3.0, (0.12, 0.14, 0.2, 0.95))
    bar_h = 8.0
    bar_y = y + 12.0
    inner_w = pw - 28.0
    font_id = 0
    _draw_rect(x + 14.0, bar_y, inner_w, bar_h, (0.1, 0.11, 0.14, 0.95))
    fill = inner_w * float(_STATE.get("pct") or 0.0)
    if fill > 1.0:
        _draw_rect(x + 14.0, bar_y, fill, bar_h, (0.25, 0.62, 0.95, 0.95))
        _draw_rect(x + 14.0 + fill * 0.55, bar_y, fill * 0.45, bar_h, (0.45, 0.82, 1.0, 0.85))
    blf.size(font_id, 14)
    blf.color(font_id, 0.92, 0.94, 0.98, 1.0)
    title = str(_STATE.get("title") or "GR2 LAB")
    blf.position(font_id, x + 14.0, y + ph - 28.0, 0)
    blf.draw(font_id, title)
    stage = int(_STATE.get("stage") or 0)
    total = int(_STATE.get("stage_total") or 1)
    pct_i = int(round(100.0 * float(_STATE.get("pct") or 0.0)))
    blf.size(font_id, 11)
    blf.color(font_id, 0.55, 0.62, 0.72, 1.0)
    blf.position(font_id, x + pw - 88.0, y + ph - 26.0, 0)
    blf.draw(font_id, f"{pct_i}% · {stage}/{total}")
    blf.size(font_id, 12)
    blf.color(font_id, 0.78, 0.84, 0.92, 1.0)
    blf.position(font_id, x + 14.0, y + 52.0, 0)
    blf.draw(font_id, str(_STATE.get("label") or ""))
    detail = str(_STATE.get("detail") or "")
    if detail:
        blf.size(font_id, 10)
        blf.color(font_id, 0.62, 0.7, 0.8, 1.0)
        blf.position(font_id, x + 14.0, y + 30.0, 0)
        blf.draw(font_id, _truncate_middle(detail, inner_w, font_id, 10))
