"""Per-dump animation stage history (Original → Saves → Current).

Snapshots live under OUT_FILES/<run>/history/<id>/ so the UI can scrub
pre-edit states without re-decoding.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ORIGINAL_ID = "000_original"
MAX_NON_ORIGINAL = 40

# Anim-affecting files copied into each stage folder.
_SNAPSHOT_NAMES = (
    "section_0.bin",
    "anim_map.json",
    "skeleton.json",
    "section_0.relocations.bin",
    "SOURCE.txt",
)

_STAGE_DIR_RE = re.compile(r"^(\d{3})_(.+)$")


def history_root(dump: Path) -> Path:
    return Path(dump) / "history"


def resolve_stage_dir(dump: Path, stage_id: Optional[str] = None) -> Path:
    """Return directory to load anim bytes from.

    ``current`` / empty → dump root. Otherwise ``history/<stage_id>``.
    """
    dump = Path(dump)
    sid = (stage_id or "current").strip() or "current"
    if sid == "current":
        return dump
    hist = history_root(dump) / sid
    if not hist.is_dir():
        raise FileNotFoundError(f"History stage not found: {sid}")
    if not (hist / "section_0.bin").is_file():
        raise FileNotFoundError(f"History stage missing section_0.bin: {sid}")
    return hist


def has_original(dump: Path) -> bool:
    return (history_root(dump) / ORIGINAL_ID / "section_0.bin").is_file()


_ENCODE_COMPARE_RE = re.compile(r"^section_\d+\.(bin|relocations\.bin|mm\.bin)$")


def _file_digest(path: Path) -> str:
    from gr2_codec import sha256_bytes

    return sha256_bytes(path.read_bytes())


def _encode_file_names(dump: Path, orig: Path) -> List[str]:
    names: set[str] = set()
    for folder in (orig, dump):
        if folder.is_dir():
            for p in folder.iterdir():
                if p.is_file() and _ENCODE_COMPARE_RE.match(p.name):
                    names.add(p.name)
    return sorted(names)


def mark_blender_import(dump: Path) -> None:
    """Record that a Blender .gr2lab was applied (Encode should be available)."""
    dump = Path(dump)
    (dump / "FROM_BLENDER.txt").write_text(
        f"imported={_utc_now()}\n",
        encoding="utf-8",
    )


def dump_has_edits_fast(dump: Path) -> bool:
    """Cheap sidebar hint — skips byte-for-byte section compares on /api/status."""
    dump = Path(dump)
    if (dump / "FROM_BLENDER.txt").is_file():
        return True

    edit_path = dump / "EDIT.json"
    if edit_path.is_file():
        try:
            doc = json.loads(edit_path.read_text(encoding="utf-8"))
            edits = doc.get("edits") or []
            if isinstance(edits, list) and edits:
                return True
        except Exception:
            pass

    orig = history_root(dump) / ORIGINAL_ID
    if not (orig / "section_0.bin").is_file():
        return False

    root = history_root(dump)
    if root.is_dir():
        for p in root.iterdir():
            if (
                p.is_dir()
                and p.name != ORIGINAL_ID
                and (p / "section_0.bin").is_file()
            ):
                return True
    return False


def live_matches_original(dump: Path) -> bool:
    """True when live section bins already match history/000_original."""
    dump = Path(dump)
    orig = history_root(dump) / ORIGINAL_ID
    if not (orig / "section_0.bin").is_file():
        return False
    names = _encode_file_names(dump, orig)
    if not names:
        return False
    for name in names:
        live = dump / name
        ref = orig / name
        if live.is_file() != ref.is_file():
            return False
        if live.is_file() and _file_digest(live) != _file_digest(ref):
            return False
    return True


def dump_has_edits(dump: Path) -> bool:
    """True when live dump bytes differ from the post-decode original snapshot."""
    dump = Path(dump)
    if dump_has_edits_fast(dump):
        return True

    orig = history_root(dump) / ORIGINAL_ID
    if not (orig / "section_0.bin").is_file():
        return False

    for name in _encode_file_names(dump, orig):
        live = dump / name
        ref = orig / name
        if not live.is_file():
            continue
        if not ref.is_file():
            continue
        if _file_digest(live) != _file_digest(ref):
            return True

    edit_path = dump / "EDIT.json"
    if edit_path.is_file():
        try:
            doc = json.loads(edit_path.read_text(encoding="utf-8"))
            edits = doc.get("edits") or []
            if isinstance(edits, list) and edits:
                return True
        except Exception:
            pass

    return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _next_stage_id(dump: Path, kind: str) -> str:
    root = history_root(dump)
    root.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _STAGE_DIR_RE.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    suffix = "original" if kind == "original" else ("gr2lab" if kind == "gr2lab" else "save")
    return f"{max_n + 1:03d}_{suffix}"


def _copy_snapshot_files(src: Path, dest: Path) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    names: set[str] = set(_SNAPSHOT_NAMES)
    if src.is_dir():
        for p in src.iterdir():
            if p.is_file() and _ENCODE_COMPARE_RE.match(p.name):
                names.add(p.name)
    for name in sorted(names):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dest / name)
            copied.append(name)
    if "section_0.bin" not in copied:
        raise FileNotFoundError(f"Cannot snapshot — missing section_0.bin in {src}")
    return copied


def snapshot_dump(
    dump: Path,
    *,
    kind: str = "save",
    label: str = "",
    bones: Optional[Sequence[str]] = None,
    ops_summary: Optional[str] = None,
    force_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Copy live anim files into history/<id>/. Returns meta dict."""
    dump = Path(dump)
    if not (dump / "section_0.bin").is_file():
        raise FileNotFoundError(f"Cannot snapshot — missing section_0.bin in {dump}")

    kind = (kind or "save").strip().lower()
    if kind not in ("original", "save", "gr2lab"):
        kind = "save"

    if force_id:
        stage_id = force_id
    elif kind == "original":
        stage_id = ORIGINAL_ID
    else:
        stage_id = _next_stage_id(dump, kind)

    dest = history_root(dump) / stage_id
    if dest.exists():
        shutil.rmtree(dest)
    copied = _copy_snapshot_files(dump, dest)

    bone_list = [str(b) for b in (bones or []) if b]
    if not label:
        if kind == "original":
            label = "Original"
        elif kind == "gr2lab":
            label = "Before Blender import"
            if bone_list:
                label = f"Before Blender · {', '.join(bone_list[:4])}"
                if len(bone_list) > 4:
                    label += f" +{len(bone_list) - 4}"
        else:
            label = "Before save"
            if bone_list:
                shown = ", ".join(bone_list[:4])
                if len(bone_list) > 4:
                    shown += f" +{len(bone_list) - 4}"
                label = f"Before save · {shown}"

    meta: Dict[str, Any] = {
        "id": stage_id,
        "label": label,
        "kind": kind,
        "created": _utc_now(),
        "bones": bone_list,
        "ops_summary": ops_summary or "",
        "files": copied,
    }
    if extra:
        meta["extra"] = extra
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if kind != "original":
        prune_history(dump, keep=MAX_NON_ORIGINAL)

    return meta


def snapshot_original(dump: Path) -> Dict[str, Any]:
    """Write/replace 000_original from current dump (call right after decode)."""
    return snapshot_dump(
        dump,
        kind="original",
        label="Original",
        force_id=ORIGINAL_ID,
    )


def _read_meta(stage_dir: Path) -> Dict[str, Any]:
    meta_path = stage_dir / "meta.json"
    if meta_path.is_file():
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("id"):
                return doc
        except Exception:
            pass
    return {
        "id": stage_dir.name,
        "label": stage_dir.name,
        "kind": "save",
        "created": "",
        "bones": [],
        "ops_summary": "",
    }


def list_history(dump: Path) -> Dict[str, Any]:
    """Sorted history stages plus synthetic ``current``."""
    dump = Path(dump)
    root = history_root(dump)
    stages: List[Dict[str, Any]] = []
    if root.is_dir():
        dirs = [p for p in root.iterdir() if p.is_dir() and (p / "section_0.bin").is_file()]
        dirs.sort(key=lambda p: p.name)
        for p in dirs:
            stages.append(_read_meta(p))

    stages.append(
        {
            "id": "current",
            "label": "Current",
            "kind": "current",
            "created": "",
            "bones": [],
            "ops_summary": "",
        }
    )
    return {
        "ok": True,
        "run_id": dump.name,
        "stages": stages,
        "has_original": has_original(dump),
    }


def prune_history(dump: Path, keep: int = MAX_NON_ORIGINAL) -> int:
    """Delete oldest non-original stages beyond ``keep``. Returns count removed."""
    dump = Path(dump)
    root = history_root(dump)
    if not root.is_dir():
        return 0
    non_orig = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name == ORIGINAL_ID:
            continue
        if (p / "section_0.bin").is_file():
            non_orig.append(p)
    non_orig.sort(key=lambda p: p.name)
    removed = 0
    while len(non_orig) > keep:
        victim = non_orig.pop(0)
        shutil.rmtree(victim, ignore_errors=True)
        removed += 1
    return removed


def restore_stage(dump: Path, stage_id: str) -> Dict[str, Any]:
    """Copy a history stage over the live dump anim files."""
    dump = Path(dump)
    sid = (stage_id or "").strip()
    if not sid or sid == "current":
        raise ValueError("Cannot restore current onto itself — pick a history stage")
    src = resolve_stage_dir(dump, sid)
    copied = _copy_snapshot_files(src, dump)
    # Drop live encode sidecars that the stage did not restore. After a grown
    # apply, section_*.relocations.bin / *.mm.bin can point past the restored
    # section_0 size — load_dump_gr2 would then crash in build_anim_map.
    purged: List[str] = []
    copied_set = set(copied)
    if dump.is_dir():
        for p in list(dump.iterdir()):
            if not p.is_file() or not _ENCODE_COMPARE_RE.match(p.name):
                continue
            if p.name in copied_set:
                continue
            try:
                p.unlink()
                purged.append(p.name)
            except OSError:
                pass
    # Refresh EDIT.json note
    note = {
        "restored_from": sid,
        "restored_at": _utc_now(),
        "files": copied,
        "purged": purged,
    }
    (dump / "EDIT.json").write_text(
        json.dumps(
            {
                "from_dump": dump.name,
                "overwrite": True,
                "restore": note,
                "edits": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "run_id": dump.name,
        "restored_from": sid,
        "files": copied,
        "purged": purged,
    }


def label_for_bones(bones: Sequence[str], *, prefix: str = "Before save") -> str:
    bone_list = [str(b) for b in bones if b]
    if not bone_list:
        return prefix
    shown = ", ".join(bone_list[:4])
    if len(bone_list) > 4:
        shown += f" +{len(bone_list) - 4}"
    return f"{prefix} · {shown}"
