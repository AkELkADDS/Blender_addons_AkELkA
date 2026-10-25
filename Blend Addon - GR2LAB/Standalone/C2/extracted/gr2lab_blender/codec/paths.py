"""Addon cache layout for GR2 Lab (no project IN_GR2 server tree required).

Cache roots are configured via configure_layout() from Blender preferences.
Default: %TEMP%/gr2lab_blender_cache
"""

from __future__ import annotations

import os
import random
import re
import tempfile
from pathlib import Path

_ADDON_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BODY = _ADDON_ROOT / "Body_Base"

# Mutable roots — set by configure_layout()
ROOT: Path = Path(tempfile.gettempdir()) / "gr2lab_blender_cache"
IN_GR2: Path = ROOT / "IN_GR2"
OUT_FILES: Path = ROOT / "OUT_FILES"
OUT_GR2: Path = ROOT / "OUT_GR2"
OUT_WATCH: Path = ROOT / "OUT_WATCH"
BODY_BASE: Path = _DEFAULT_BODY if _DEFAULT_BODY.is_dir() else ROOT / "Body_Base"
SHARED_FILES: Path = ROOT / "Shared Files"
# Skip mkdir/rebind when prefs haven't changed (enum callbacks poll this constantly).
_LAYOUT_INPUTS: tuple[str, str] | None = None
_LAYOUT_ENSURED: tuple[str, ...] | None = None

_WIN_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RUN_ID_MAX_LEN = 180
_BODY_PREFIX_RE = re.compile(r"^([A-Z]{2,5}_[A-Z0-9]{1,4})_Rig_SCENE_", re.I)
_SCENE_BUCKET = frozenset({"camp", "glo"})
_SOURCE_NAME_MUTE = frozenset(
    {"rig", "scene", "sd", "rom", "nested", "plyr", "sk", "el"}
)

_LYRIC_SEED = (
    "war-paint",
    "ember-waltz",
    "shadow-heart",
    "nightfall",
    "starfall",
    "paradox",
    "riptide",
    "zenith",
    "crossfade",
    "daybreak",
    "moonshine",
    "torchlight",
    "stardust",
    "wildfire",
    "icebound",
    "limelight",
    "afterglow",
    "quicksilver",
    "evergreen",
    "flashpoint",
)
_LYRIC_A = (
    "ash",
    "bay",
    "red",
    "oak",
    "ice",
    "sun",
    "sea",
    "sky",
    "dry",
    "hot",
    "pale",
    "dark",
    "soft",
    "wild",
    "calm",
    "blue",
    "grey",
    "gold",
    "rose",
    "iron",
    "jade",
    "coal",
    "salt",
    "mist",
    "fog",
    "dew",
    "ink",
    "elm",
    "fir",
    "yew",
    "rye",
    "rust",
)
_LYRIC_B = (
    "wolf",
    "moon",
    "tide",
    "rain",
    "fire",
    "dawn",
    "dusk",
    "wind",
    "wave",
    "star",
    "root",
    "leaf",
    "bone",
    "soul",
    "glow",
    "haze",
    "flame",
    "spark",
    "shard",
    "veil",
    "thorn",
    "reed",
    "lake",
    "peak",
    "vale",
    "glen",
    "moor",
    "snow",
    "frost",
    "storm",
    "cloud",
    "shade",
)
LYRIC_NAMES: list[str] = list(
    dict.fromkeys(list(_LYRIC_SEED) + [f"{a}-{b}" for a in _LYRIC_A for b in _LYRIC_B])
)
_LYRIC_NAMES_SET = frozenset(n.lower() for n in LYRIC_NAMES)
_LYRIC_NAMES_BY_LEN = tuple(sorted(LYRIC_NAMES, key=len, reverse=True))


def configure_layout(
    *,
    cache_root: str | Path | None = None,
    body_base: str | Path | None = None,
) -> None:
    """Point OUT_* / optional Body_Base at user-chosen folders."""
    global ROOT, IN_GR2, OUT_FILES, OUT_GR2, OUT_WATCH, BODY_BASE, SHARED_FILES
    global _LAYOUT_INPUTS
    cache_s = str(cache_root).strip() if cache_root else ""
    body_s = str(body_base).strip() if body_base else ""
    inputs = (cache_s, body_s)
    if _LAYOUT_INPUTS == inputs:
        return
    if cache_s:
        ROOT = Path(cache_s)
    else:
        ROOT = Path(tempfile.gettempdir()) / "gr2lab_blender_cache"
    IN_GR2 = ROOT / "IN_GR2"
    OUT_FILES = ROOT / "OUT_FILES"
    OUT_GR2 = ROOT / "OUT_GR2"
    OUT_WATCH = ROOT / "OUT_WATCH"
    SHARED_FILES = ROOT / "Shared Files"
    if body_s:
        BODY_BASE = Path(body_s)
    elif _DEFAULT_BODY.is_dir():
        BODY_BASE = _DEFAULT_BODY
    else:
        BODY_BASE = ROOT / "Body_Base"
    _LAYOUT_INPUTS = inputs
    ensure_layout()


def ensure_layout() -> None:
    global _LAYOUT_ENSURED
    sig = (
        str(IN_GR2),
        str(OUT_FILES),
        str(OUT_GR2),
        str(OUT_WATCH),
        str(BODY_BASE),
        str(SHARED_FILES),
    )
    if _LAYOUT_ENSURED == sig:
        return
    IN_GR2.mkdir(parents=True, exist_ok=True)
    OUT_FILES.mkdir(parents=True, exist_ok=True)
    OUT_GR2.mkdir(parents=True, exist_ok=True)
    OUT_WATCH.mkdir(parents=True, exist_ok=True)
    BODY_BASE.mkdir(parents=True, exist_ok=True)
    SHARED_FILES.mkdir(parents=True, exist_ok=True)
    _LAYOUT_ENSURED = sig


def is_watch_run(run_id: str | None) -> bool:
    if not run_id:
        return False
    return Path(run_id).name.startswith("watch-")


def resolve_dump_dir(run_id: str | None) -> Path:
    ensure_layout()
    if not run_id:
        raise ValueError("run id required")
    safe = Path(run_id).name
    if not safe or safe in {".", ".."}:
        raise ValueError("Invalid run id")
    # Absolute dump path stamped by addon
    as_path = Path(run_id)
    if as_path.is_absolute() and as_path.is_dir():
        return as_path
    watch = OUT_WATCH / safe
    files = OUT_FILES / safe
    if is_watch_run(safe):
        if watch.is_dir():
            return watch
        if files.is_dir():
            return files
        return watch
    if files.is_dir():
        return files
    if watch.is_dir():
        return watch
    return files


def _used_names(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.iterdir() if p.is_dir()}


def _used_names_lower(folder: Path) -> set[str]:
    return {n.lower() for n in _used_names(folder)}


def source_stem_for_run_id(source: str) -> str:
    raw = Path(str(source).replace("\\", "/")).name
    if raw.lower().endswith(".gr2"):
        raw = raw[:-4]
    raw = _WIN_INVALID.sub("_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._ ")
    if not raw:
        raw = "clip"
    if len(raw) > _RUN_ID_MAX_LEN:
        raw = raw[:_RUN_ID_MAX_LEN].rstrip("._ ")
    return raw


def _source_name_tokens(name: str) -> list[str]:
    raw = Path(str(name).replace("\\", "/")).name
    if raw.lower().endswith(".gr2"):
        raw = raw[:-4]
    return re.findall(r"[A-Za-z0-9]+|_|(?:\.[A-Za-z0-9]+)", raw) or [raw]


def _is_source_name_mute_token(tok: str) -> bool:
    if not tok or tok in ("_", "."):
        return True
    if re.fullmatch(r"\.[a-z0-9]+", tok, flags=re.I):
        return True
    if tok.isdigit():
        return True
    return tok.lower() in _SOURCE_NAME_MUTE


def source_hot_stem_for_run_id(source: str) -> str:
    hot = [t for t in _source_name_tokens(source) if not _is_source_name_mute_token(t)]
    if not hot:
        return source_stem_for_run_id(source)
    raw = "_".join(hot)
    raw = _WIN_INVALID.sub("_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._ ")
    if not raw:
        return source_stem_for_run_id(source)
    if len(raw) > _RUN_ID_MAX_LEN:
        raw = raw[:_RUN_ID_MAX_LEN].rstrip("._ ")
    return raw


def _extract_body_prefix(raw: str) -> str | None:
    m = _BODY_PREFIX_RE.match(raw)
    return m.group(1) if m else None


def _anim_tail_tokens(raw: str) -> list[str]:
    upper = raw.upper()
    if "_ROM_" in upper:
        idx = upper.rfind("_ROM_")
        tail = raw[idx + 5 :]
    elif "_RIG_SCENE_" in upper:
        idx = upper.find("_RIG_SCENE_")
        rest = raw[idx + len("_Rig_SCENE_") :]
        parts = rest.split("_")
        while parts and parts[0].lower() in _SCENE_BUCKET:
            parts.pop(0)
        tail = "_".join(parts)
    else:
        return []
    tail = re.sub(r"_\d+$", "", tail, flags=re.I)
    return [
        t for t in re.findall(r"[A-Za-z0-9]+", tail) if not _is_source_name_mute_token(t)
    ]


def _sanitize_run_stem(raw: str) -> str:
    raw = _WIN_INVALID.sub("_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._ ")
    if not raw:
        return "clip"
    if len(raw) > _RUN_ID_MAX_LEN:
        raw = raw[:_RUN_ID_MAX_LEN].rstrip("._ ")
    return raw


def source_clip_stem_for_run_id(source: str) -> str:
    raw = Path(str(source).replace("\\", "/")).name
    if raw.lower().endswith(".gr2"):
        raw = raw[:-4]
    body = _extract_body_prefix(raw)
    hot = _anim_tail_tokens(raw)
    if hot:
        bits = ([body] if body else []) + hot
        return _sanitize_run_stem("_".join(bits))
    return _sanitize_run_stem(source_hot_stem_for_run_id(source))


def _split_lyric_suffix(name: str) -> tuple[str, str | None]:
    if not name:
        return name, None
    low = name.lower()
    if "--" in name:
        base, suffix = name.rsplit("--", 1)
        if suffix.lower() in _LYRIC_NAMES_SET or suffix.isdigit():
            return base, suffix
    for lyric in _LYRIC_NAMES_BY_LEN:
        tag = f"_{lyric}"
        if low.endswith(tag):
            return name[: -len(tag)], lyric
    return name, None


def read_dump_source_name(dump_dir: Path) -> str | None:
    src_txt = dump_dir / "SOURCE.txt"
    if not src_txt.is_file():
        return None
    try:
        for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("name="):
                val = line.split("=", 1)[1].strip()
                return val or None
    except OSError:
        pass
    return None


def read_dump_source_path(dump_dir: Path) -> Path | None:
    """Absolute source GR2 path from SOURCE.txt path= line."""
    src_txt = dump_dir / "SOURCE.txt"
    if not src_txt.is_file():
        return None
    try:
        for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("path="):
                val = line.split("=", 1)[1].strip()
                if val:
                    p = Path(val)
                    if p.is_file():
                        return p
    except OSError:
        pass
    return None


def read_dump_body_base(dump_dir: Path) -> str | None:
    bind_txt = dump_dir / "BIND_SOURCE.txt"
    if not bind_txt.is_file():
        return None
    try:
        for line in bind_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.lower().startswith("name="):
                val = line.split("=", 1)[1].strip()
                return val or None
    except OSError:
        pass
    return None


def _unique_stem_run_id(base: str, folder: Path) -> str:
    ensure_layout()
    folder.mkdir(parents=True, exist_ok=True)
    if not base:
        return next_lyric_name(folder)
    used = _used_names_lower(folder)
    for lyric in LYRIC_NAMES:
        candidate = f"{base}_{lyric}"
        if candidate.lower() not in used:
            return candidate
    n = 2
    while f"{base}_{n}".lower() in used:
        n += 1
    return f"{base}_{n}"


def unique_run_id_from_source(source: str, folder: Path | None = None) -> str:
    target = folder or OUT_FILES
    base = source_clip_stem_for_run_id(source)
    return _unique_stem_run_id(base, target)


def unique_encode_run_id(preferred_name: str | None) -> str:
    raw = str(preferred_name or "").strip()
    name = Path(raw.replace("\\", "/")).name if raw else ""
    name, _ = _split_lyric_suffix(name)
    peer = OUT_FILES / name
    if peer.is_dir():
        src = read_dump_source_name(peer)
        if src:
            base = source_clip_stem_for_run_id(src)
            return _unique_stem_run_id(base, OUT_GR2)
    if name.lower().endswith(".gr2"):
        return _unique_stem_run_id(source_clip_stem_for_run_id(name), OUT_GR2)
    return _unique_stem_run_id(_sanitize_run_stem(name) if name else "", OUT_GR2)


def next_lyric_name(folder: Path) -> str:
    ensure_layout()
    folder.mkdir(parents=True, exist_ok=True)
    used = _used_names(folder)
    pool = [n for n in LYRIC_NAMES if n not in used]
    if pool:
        return random.choice(pool)
    base = random.choice(LYRIC_NAMES)
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    return f"{base}-{n}"


def make_run_dirs(
    action: str = "run", *, preferred_name: str | None = None
) -> tuple[str, Path, Path]:
    ensure_layout()
    action = (action or "run").lower()

    if action.startswith("encode"):
        safe = Path(preferred_name).name if preferred_name else ""
        if safe and safe not in {".", ".."} and not is_watch_run(safe):
            run_id = unique_encode_run_id(safe)
        else:
            run_id = next_lyric_name(OUT_GR2)
        files_dir = OUT_FILES / run_id
        gr2_dir = OUT_GR2 / run_id
        gr2_dir.mkdir(parents=True, exist_ok=True)
        return run_id, files_dir, gr2_dir

    if preferred_name and str(preferred_name).strip() not in {".", ".."}:
        run_id = unique_run_id_from_source(str(preferred_name).strip())
    else:
        run_id = next_lyric_name(OUT_FILES)
    files_dir = OUT_FILES / run_id
    gr2_dir = OUT_GR2 / run_id
    files_dir.mkdir(parents=True, exist_ok=True)
    return run_id, files_dir, gr2_dir


def resolve_input_gr2(name: str) -> Path:
    """Resolve a source GR2 from absolute path, cache IN_GR2, or bare name."""
    ensure_layout()
    if not name:
        raise FileNotFoundError("No GR2 path given")
    ref = (name or "").strip().replace("\\", "/")
    while ref.startswith("./"):
        ref = ref[2:]
    as_path = Path(ref)
    if as_path.is_file() and as_path.suffix.lower() == ".gr2":
        return as_path.resolve()
    if as_path.is_absolute():
        if as_path.is_file() and as_path.suffix.lower() == ".gr2":
            return as_path.resolve()
        raise FileNotFoundError(f"GR2 not found: {ref}")
    # Relative under cache IN_GR2
    target = (IN_GR2 / ref).resolve()
    try:
        target.relative_to(IN_GR2.resolve())
    except ValueError as e:
        raise ValueError("Forbidden path") from e
    if target.is_file() and target.suffix.lower() == ".gr2":
        return target
    bare = Path(ref).name
    root_hit = IN_GR2 / bare
    if root_hit.is_file() and root_hit.suffix.lower() == ".gr2":
        return root_hit
    raise FileNotFoundError(f"GR2 not found: {ref}")


def list_input_gr2s() -> list[Path]:
    ensure_layout()
    out: list[Path] = []
    for pattern in ("*.GR2", "*.gr2"):
        out.extend(p for p in IN_GR2.rglob(pattern) if p.is_file())
    return sorted(set(out), key=lambda p: p.name.lower())


def list_dir_files(folder: Path) -> list[dict]:
    if not folder.is_dir():
        return []
    return [
        {"name": p.name, "size": p.stat().st_size, "path": str(p)}
        for p in sorted(folder.iterdir())
        if p.is_file()
    ]


def list_decode_runs() -> list[dict]:
    ensure_layout()
    runs = []
    for d in sorted(OUT_FILES.iterdir(), key=lambda p: -p.stat().st_mtime):
        if not d.is_dir() or is_watch_run(d.name):
            continue
        has_sections = any(
            p.name.startswith("section_") and p.name.endswith(".bin")
            for p in d.iterdir()
            if p.is_file()
        )
        if not has_sections:
            continue
        meta = {
            "id": d.name,
            "label": d.name,
            "path": str(d),
            "has_sections": True,
        }
        src = read_dump_source_name(d)
        if src:
            meta["source"] = src
        body = read_dump_body_base(d)
        if body:
            meta["body_base"] = body
        runs.append(meta)
    return runs


def invalidate_folder_tree_cache() -> None:
    pass


def invalidate_in_gr2_cache() -> None:
    pass
