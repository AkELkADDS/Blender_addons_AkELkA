"""Stage timing for GR2 Lab decode/encode (console + TIMING.json)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _ms(sec: float) -> int:
    return int(round(sec * 1000.0))


_LIVE_PREFIXES = (
    "densify",
    "bone_keys",
    "rig_display",
    "pack_sample",
    "stamp",
    "head_ik",
    "mesh_kit",
    "tail_rig",
    "scene_",
    "plan_",
    "partner_",
    "partner_find_",
    "decode_",
    "build_",
    "import_",
)


def _live_stage(name: str) -> bool:
    return any(name.startswith(p) for p in _LIVE_PREFIXES)


class StageTimer:
    def __init__(self, label: str = "", live: bool = True) -> None:
        self.label = label
        self.live = live
        self._t0 = time.perf_counter()
        self._mark = self._t0
        self.stages: Dict[str, int] = {}

    def live_print(self, msg: str) -> None:
        if self.live:
            print(f"[GR2 LAB timing] {msg}", flush=True)

    def span(self, name: str) -> "_StageSpan":
        return _StageSpan(self, name)

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0) + _ms(seconds)

    def merge(self, other: Optional["StageTimer"], *, prefix: str = "") -> None:
        if other is None:
            return
        for name, ms in other.stages.items():
            key = f"{prefix}{name}" if prefix else name
            self.stages[key] = self.stages.get(key, 0) + int(ms)

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0) + _ms(now - self._mark)
        self._mark = now

    def total_ms(self) -> int:
        return _ms(time.perf_counter() - self._t0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "total_ms": self.total_ms(),
            "stages_ms": dict(self.stages),
        }

    def format_block(self) -> str:
        total_ms = self.total_ms()
        total_s = total_ms / 1000.0
        lines = [f"[GR2 LAB timing] {self.label} total={total_s:.2f}s"]
        if not self.stages:
            lines.append("  (no stages)")
            return "\n".join(lines)
        accounted = 0
        ranked = sorted(self.stages.items(), key=lambda kv: kv[1], reverse=True)
        name_w = max(len(k) for k, _ in ranked)
        for name, ms in ranked:
            accounted += ms
            sec = ms / 1000.0
            pct = (100.0 * ms / total_ms) if total_ms else 0.0
            lines.append(f"  {name:<{name_w}}  {sec:7.2f}s  {pct:5.1f}%")
        missing = total_ms - accounted
        if abs(missing) >= 50:
            pct = (100.0 * missing / total_ms) if total_ms else 0.0
            label = "(unattributed)" if missing > 0 else "(overlap)"
            lines.append(
                f"  {label:<{name_w}}  {missing / 1000.0:7.2f}s  {pct:5.1f}%"
            )
        return "\n".join(lines)

    def format_compact(self) -> str:
        total_s = self.total_ms() / 1000.0
        bits = [f"{k} {v / 1000.0:.1f}" for k, v in self.stages.items()]
        return f"timing {total_s:.1f}s · " + " · ".join(bits)

    def print_block(self) -> None:
        print(self.format_block(), flush=True)

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )


class _StageSpan:
    def __init__(self, timer: StageTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_StageSpan":
        self._t0 = time.perf_counter()
        if _live_stage(self._name):
            self._timer.live_print(f"{self._name}…")
        return self

    def __exit__(self, *exc: Any) -> None:
        dt = time.perf_counter() - self._t0
        self._timer.add(self._name, dt)
        if _live_stage(self._name) or dt >= 0.15:
            self._timer.live_print(f"{self._name} {dt:.2f}s")


def dump_json_compact(obj: Any, path: Path) -> None:
    Path(path).write_text(
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def maybe_timer(timer: Optional[StageTimer], name: str):
    if timer is None:
        return _NullSpan()
    return timer.span(name)


class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
