"""Resolve BG3 dialog timeline/scene assets from Multitool UnpackedData."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_UNPACKED = Path(
    r"D:\Igor\Adquantum\Scripts\El\Scripted_Mods\bg3-modders-multitool\UnpackedData"
)

_GENERATED_REL = Path("Gustav") / "Public" / "GustavDev" / "Timeline" / "Generated"
_MODELS_REL = Path("Models")


def _read_tmlgen_paths() -> List[Path]:
    out: List[Path] = []
    local = os.environ.get("LOCALAPPDATA") or ""
    if not local:
        return out
    path = Path(local) / "GlassesMcp" / "tmlgen.json"
    if not path.is_file():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(raw, dict):
        return out
    for key in ("unpackedData", "unpacked_data", "unpackedDir", "bg3_unpacked"):
        val = raw.get(key)
        if val:
            out.append(Path(str(val)))
    return out


def unpacked_roots(extra: Optional[Sequence[str]] = None) -> List[Path]:
    """Roots that contain Gustav/.../Timeline/Generated or Models."""
    candidates: List[Path] = []
    env = (os.environ.get("GR2LAB_BG3_UNPACKED") or "").strip()
    if env:
        candidates.append(Path(env))
    if extra:
        for item in extra:
            text = (item or "").strip()
            if text:
                candidates.append(Path(text))
    candidates.extend(_read_tmlgen_paths())
    if DEFAULT_UNPACKED.is_dir():
        candidates.append(DEFAULT_UNPACKED)

    roots: List[Path] = []
    seen = set()
    for base in candidates:
        if base is None:
            continue
        for candidate in (base, base / "Gustav"):
            try:
                key = str(candidate.resolve()).lower()
            except OSError:
                key = str(candidate).lower()
            if key in seen or not candidate.is_dir():
                continue
            seen.add(key)
            roots.append(candidate)
    return roots


def timeline_data_out() -> Optional[Path]:
    """Decoder timeline caches are not a data source."""
    return None


_GENERATED_DIALOGS: Optional[Dict[str, Dict[str, str]]] = None


def generated_dialog_index(
    extra: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Dialog name → scene lsx path, from unpacked Generated only."""
    global _GENERATED_DIALOGS
    if _GENERATED_DIALOGS is not None and not extra:
        return _GENERATED_DIALOGS
    out: Dict[str, Dict[str, str]] = {}
    suffix = "_Scene.lsx"
    for root in unpacked_roots(extra):
        gen = generated_dir(root)
        if gen is None:
            continue
        try:
            paths = gen.glob("*" + suffix)
        except OSError:
            continue
        for path in paths:
            name = path.name[: -len(suffix)]
            if name and name not in out:
                out[name] = {"scene": str(path)}
    if not extra:
        _GENERATED_DIALOGS = out
    return out


def generated_dir(root: Path) -> Optional[Path]:
    path = root / _GENERATED_REL
    return path if path.is_dir() else None


def models_dir(root: Path) -> Optional[Path]:
    path = root / _MODELS_REL
    if path.is_dir():
        return path
    alt = root.parent / _MODELS_REL if root.name.lower() == "gustav" else None
    return alt if alt is not None and alt.is_dir() else None


def find_dialog_assets(dialog_name: str, roots: Optional[Sequence[Path]] = None) -> Dict[str, Optional[Path]]:
    """Scene .lsx and timeline .lsf/.lsx under Generated for one dialog."""
    name = (dialog_name or "").strip()
    out: Dict[str, Optional[Path]] = {
        "scene_lsx": None,
        "timeline_lsf": None,
        "timeline_lsx": None,
        "generated_dir": None,
    }
    if not name:
        return out
    scene_name = f"{name}_Scene.lsx"
    timeline_lsf = f"{name}.lsf"
    timeline_lsx = f"{name}.lsx"
    for root in roots or unpacked_roots():
        gen = generated_dir(root)
        if gen is None:
            continue
        out["generated_dir"] = gen
        scene = gen / scene_name
        if scene.is_file() and out["scene_lsx"] is None:
            out["scene_lsx"] = scene
        lsf = gen / timeline_lsf
        if lsf.is_file() and out["timeline_lsf"] is None:
            out["timeline_lsf"] = lsf
        lsx = gen / timeline_lsx
        if lsx.is_file() and out["timeline_lsx"] is None:
            out["timeline_lsx"] = lsx
        if out["scene_lsx"] is not None and (
            out["timeline_lsf"] is not None or out["timeline_lsx"] is not None
        ):
            break
    return out


def _copy_if_needed(src: Path, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    except OSError:
        return dest if dest.is_file() else src
    return dest


ProgressFn = Optional[Callable[[str], None]]


def _convert_lsf(
    lsf: Path,
    dest_lsx: Path,
    exe: Path,
    divine_run: Callable[..., bool],
    progress: ProgressFn = None,
) -> Optional[Path]:
    if dest_lsx.is_file() and dest_lsx.stat().st_size > 1000:
        return dest_lsx
    if not lsf.is_file() or not exe.is_file():
        return None
    if progress is not None:
        progress(f"Converting timeline LSF… {lsf.name}")
    if divine_run(
        exe,
        [
            "-a",
            "convert-resource",
            "-s",
            str(lsf.resolve()),
            "-d",
            str(dest_lsx.resolve()),
        ],
    ):
        if dest_lsx.is_file() and dest_lsx.stat().st_size > 1000:
            return dest_lsx
    return None


def ensure_lsx_from_loose(
    dialog_name: str,
    cache: Path,
    exe: Path,
    divine_run: Callable[..., bool],
    *,
    extra_roots: Optional[Sequence[str]] = None,
    progress: ProgressFn = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (timeline_lsx, scene_lsx) in cache, converting timeline LSF on demand."""
    name = (dialog_name or "").strip()
    if not name:
        return None, None
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    timeline_dest = cache / f"{name}.lsx"
    scene_dest = cache / f"{name}_Scene.lsx"
    assets = find_dialog_assets(name, unpacked_roots(extra_roots))

    scene_src = assets.get("scene_lsx")
    if scene_src is not None:
        scene_dest = _copy_if_needed(scene_src, scene_dest)

    timeline_lsx: Optional[Path] = None
    if assets.get("timeline_lsx") is not None:
        timeline_lsx = _copy_if_needed(assets["timeline_lsx"], timeline_dest)
    elif assets.get("timeline_lsf") is not None:
        timeline_lsx = _convert_lsf(
            assets["timeline_lsf"], timeline_dest, exe, divine_run, progress
        )

    if timeline_lsx is None and timeline_dest.is_file() and timeline_dest.stat().st_size > 1000:
        timeline_lsx = timeline_dest
    if scene_dest.is_file() and scene_dest.stat().st_size > 1000:
        scene_out: Optional[Path] = scene_dest
    else:
        scene_out = scene_src

    return timeline_lsx, scene_out


def find_gr2_in_models(stem: str, roots: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """Locate a scene GR2 by stem under UnpackedData/Models (case-insensitive)."""
    want = (stem or "").lower()
    if not want:
        return None
    filename = stem if stem.lower().endswith(".gr2") else f"{stem}.GR2"
    for root in roots or unpacked_roots():
        models = models_dir(root)
        if models is None:
            continue
        direct = list(models.rglob(filename))
        if direct:
            return direct[0]
        for path in models.rglob("*.GR2"):
            if path.stem.lower() == want:
                return path
    return None


def load_dialog_assets_index(cache: Path) -> Dict[str, Dict[str, str]]:
    path = cache / "dialog_assets.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}
