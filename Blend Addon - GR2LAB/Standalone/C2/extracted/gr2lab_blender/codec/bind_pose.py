"""Bind / T-pose extraction from Body_Base GR2s (display + Blender Edit basis).

Encode never reads this — TransformTrack curves remain the sole GR2 write path.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FORMAT_ID = "gr2lab_bind"
FORMAT_VERSION = 1

# Directory listing is polled by Blender EnumProperty callbacks every redraw.
_BODY_BASES_CACHE: List[Dict[str, Any]] = []
_BODY_BASES_CACHE_KEY: Tuple[str, int] | None = None
_BIND_CATALOG_CACHE: List[Dict[str, str]] = []
_BIND_CATALOG_CACHE_KEY: Tuple[str, int] | None = None
_TAIL_BASE_RE = re.compile(r"_Tail_Base$", re.I)
_ADDON_BODY_BASE = Path(__file__).resolve().parent.parent / "Body_Base"


def _body_base_dir_key() -> Tuple[str, int]:
    from paths import BODY_BASE

    folder = BODY_BASE
    try:
        return (str(folder), folder.stat().st_mtime_ns)
    except OSError:
        return (str(folder), 0)


def list_body_bases() -> List[Dict[str, Any]]:
    """Catalog Body_Base/*.GR2 for UI."""
    global _BODY_BASES_CACHE, _BODY_BASES_CACHE_KEY
    from paths import BODY_BASE

    key = _body_base_dir_key()
    if _BODY_BASES_CACHE_KEY == key:
        return _BODY_BASES_CACHE
    folder = BODY_BASE
    uniq: List[Dict[str, Any]] = []
    seen = set()
    try:
        if folder.is_dir():
            names = sorted(
                p.name
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() == ".gr2"
            )
            for name in names:
                if name in seen or is_tail_base_id(name):
                    continue
                seen.add(name)
                uniq.append({"name": name, "path": str(folder / name)})
    except OSError:
        uniq = []
    _BODY_BASES_CACHE = uniq
    _BODY_BASES_CACHE_KEY = key
    return uniq


def bind_pose_stem(name: str) -> str:
    raw = Path(name or "").name
    if raw.lower().endswith(".bind_pose.json"):
        return raw[: -len(".bind_pose.json")]
    stem = Path(raw).stem
    return stem or raw


def is_tail_base_id(name: str) -> bool:
    """True for TIF_FS_Tail_Base / TIF_FS_Tail_Base.GR2 sidecars."""
    return bool(_TAIL_BASE_RE.search(bind_pose_stem(name)))


def _bind_sidecar_folders() -> List[Path]:
    from paths import BODY_BASE, ensure_layout

    ensure_layout()
    folders: List[Path] = []
    for folder in (BODY_BASE, _ADDON_BODY_BASE):
        if folder not in folders:
            folders.append(folder)
    return folders


def bind_pose_sidecar_path(name: str) -> Path:
    """TIF_FS_Base / TIF_FS_Base.GR2 → Body_Base/TIF_FS_Base.bind_pose.json."""
    stem = bind_pose_stem(name)
    for folder in _bind_sidecar_folders():
        p = folder / f"{stem}.bind_pose.json"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Bind pose not found: {stem}.bind_pose.json")


def tail_base_id_for_body(body_id: str) -> Optional[str]:
    """TIF_FS_Base → TIF_FS_Tail_Base when that sidecar exists, else None."""
    stem = bind_pose_stem(body_id)
    if not stem or stem.upper() in {"NONE", "AUTO"}:
        return None
    if is_tail_base_id(stem):
        cand = stem
    else:
        prefix = stem[:-5] if stem.lower().endswith("_base") else stem
        if not prefix:
            return None
        cand = f"{prefix}_Tail_Base"
    try:
        bind_pose_sidecar_path(cand)
    except FileNotFoundError:
        return None
    return cand


def list_bind_pose_catalog() -> List[Dict[str, str]]:
    """Dropdown rows: id TIF_FS_Base, path to sidecar json."""
    global _BIND_CATALOG_CACHE, _BIND_CATALOG_CACHE_KEY
    from paths import BODY_BASE

    key = _body_base_dir_key()
    if _BIND_CATALOG_CACHE_KEY == key:
        return _BIND_CATALOG_CACHE
    folder = BODY_BASE
    items: List[Dict[str, str]] = []
    seen = set()
    try:
        if folder.is_dir():
            names = sorted(
                p.name
                for p in folder.iterdir()
                if p.is_file() and p.name.lower().endswith(".bind_pose.json")
            )
            for name in names:
                stem = name[: -len(".bind_pose.json")]
                if stem in seen or is_tail_base_id(stem):
                    continue
                seen.add(stem)
                items.append({"id": stem, "name": stem, "path": str(folder / name)})
    except OSError:
        items = []
    if not items:
        for b in list_body_bases():
            stem = bind_pose_stem(b.get("name") or "")
            if not stem or stem in seen:
                continue
            seen.add(stem)
            items.append({"id": stem, "name": stem, "path": str(b.get("path") or "")})
    _BIND_CATALOG_CACHE = items
    _BIND_CATALOG_CACHE_KEY = key
    return items


def gr2lab_doc_from_bind(bind: Dict[str, Any], *, run_id: str) -> Dict[str, Any]:
    """GR2LAB dump with bind skeleton only — no animation tracks."""
    skeleton = []
    bind_bones = []
    for b in bind.get("bones") or []:
        name = b.get("name")
        if not name:
            continue
        t = list(b.get("bind_translation") or [0, 0, 0])
        r = list(b.get("bind_rotation") or [0, 0, 0, 1])
        s = list(b.get("bind_scale") or [1, 1, 1])
        parent = b.get("parent_name")
        skeleton.append(
            {
                "name": name,
                "parent_name": parent,
                "rest_translation": t,
                "rest_rotation": r,
                "rest_scale": s,
            }
        )
        bind_bones.append(
            {
                "name": name,
                "parent_name": parent,
                "bind_translation": t,
                "bind_rotation": r,
                "bind_scale": s,
                "from": "bind",
            }
        )
    return {
        "format": "gr2lab",
        "version": 1,
        "run_id": run_id,
        "source_gr2": bind.get("source_gr2") or f"{run_id}.GR2",
        "duration": 0.0,
        "root": bind.get("root") or (skeleton[0]["name"] if skeleton else ""),
        "skeleton": skeleton,
        "tracks": [],
        "bind": {
            "format": bind.get("format"),
            "version": bind.get("version"),
            "source_gr2": bind.get("source_gr2"),
            "root": bind.get("root"),
            "bones": bind_bones,
        },
    }


def resolve_body_base(name: str) -> Path:
    from paths import BODY_BASE, ensure_layout

    ensure_layout()
    safe = Path(name).name
    p = BODY_BASE / safe
    if not p.is_file():
        raise FileNotFoundError(f"Body base not found: {safe}")
    if p.suffix.lower() != ".gr2":
        raise ValueError("Body base must be a .GR2")
    return p


def match_body_base_for_source(source_gr2_name: str) -> Optional[str]:
    """TIF_FS_Rig_... → TIF_FS_Base.GR2; tail/head-aware via rig_catalog."""
    try:
        from rig_catalog import match_body_base_detail

        detail = match_body_base_detail(source_gr2_name)
        return detail.get("matched")
    except Exception:
        pass
    stem = Path(Path(source_gr2_name or "").name).stem
    if not stem:
        return None
    # Fallback: longest matching Base stem
    bases = list_body_bases()
    if not bases:
        return None
    candidates: List[Tuple[int, str]] = []
    for b in bases:
        bstem = Path(b["name"]).stem  # e.g. TIF_FS_Base
        prefix = bstem
        if prefix.lower().endswith("_base"):
            prefix = prefix[: -len("_base")]
        sl = stem.lower()
        pl = prefix.lower()
        if sl == pl or sl.startswith(pl + "_"):
            candidates.append((len(pl), b["name"]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _mat_to_trs(m: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
    """Collada row-major 4x4 → translation, quat xyzw, scale."""
    r00, r01, r02, tx = float(m[0]), float(m[1]), float(m[2]), float(m[3])
    r10, r11, r12, ty = float(m[4]), float(m[5]), float(m[6]), float(m[7])
    r20, r21, r22, tz = float(m[8]), float(m[9]), float(m[10]), float(m[11])
    sx = math.sqrt(r00 * r00 + r10 * r10 + r20 * r20) or 1.0
    sy = math.sqrt(r01 * r01 + r11 * r11 + r21 * r21) or 1.0
    sz = math.sqrt(r02 * r02 + r12 * r12 + r22 * r22) or 1.0
    r00, r10, r20 = r00 / sx, r10 / sx, r20 / sx
    r01, r11, r21 = r01 / sy, r11 / sy, r21 / sy
    r02, r12, r22 = r02 / sz, r12 / sz, r22 / sz
    tr = r00 + r11 + r22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (r21 - r12) / S
        y = (r02 - r20) / S
        z = (r10 - r01) / S
    elif r00 > r11 and r00 > r22:
        S = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / S
        x = 0.25 * S
        y = (r01 + r10) / S
        z = (r02 + r20) / S
    elif r11 > r22:
        S = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / S
        x = (r01 + r10) / S
        y = 0.25 * S
        z = (r12 + r21) / S
    else:
        S = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / S
        x = (r02 + r20) / S
        y = (r12 + r21) / S
        z = 0.25 * S
    return [tx, ty, tz], [x, y, z, w], [sx, sy, sz]


def parse_bind_from_dae(dae_text: str, *, source_gr2: str = "") -> Dict[str, Any]:
    """Parse Divine Collada JOINT hierarchy into bind_pose doc."""
    stack: List[str] = []
    bones: Dict[str, Dict[str, Any]] = {}
    token_re = re.compile(
        r'(<node id="Bone_([^"]+)" name="([^"]+)"[^>]*type="JOINT">)'
        r'|(<matrix sid="Transform">([^<]+)</matrix>)'
        r'|(</node>)'
    )
    for m in token_re.finditer(dae_text):
        if m.group(1):
            name = m.group(3)
            parent = stack[-1] if stack else None
            stack.append(name)
            bones[name] = {
                "name": name,
                "parent_name": parent,
                "bind_translation": [0.0, 0.0, 0.0],
                "bind_rotation": [0.0, 0.0, 0.0, 1.0],
                "bind_scale": [1.0, 1.0, 1.0],
            }
        elif m.group(4):
            if not stack:
                continue
            name = stack[-1]
            vals = [float(x) for x in m.group(5).split()]
            if len(vals) >= 16:
                t, r, s = _mat_to_trs(vals)
                bones[name]["bind_translation"] = t
                bones[name]["bind_rotation"] = r
                bones[name]["bind_scale"] = s
        elif m.group(6):
            if stack:
                stack.pop()

    ordered = list(bones.values())
    root = ordered[0]["name"] if ordered else ""
    for b in ordered:
        if not b.get("parent_name"):
            root = b["name"]
            break
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "source_gr2": source_gr2,
        "root": root,
        "bones": ordered,
    }


def bind_sidecar_path(base_gr2: Path) -> Path:
    """Shipped T-pose JSON next to Body_Base GR2 (no Divine at runtime)."""
    p = Path(base_gr2)
    return p.with_name(f"{p.stem}.bind_pose.json")


def extract_bind_pose(
    base_gr2: Path,
    *,
    work_dir: Optional[Path] = None,
    force_divine: bool = False,
) -> Dict[str, Any]:
    """Body_Base GR2 → bind_pose document.

    Prefers the shipped ``*.bind_pose.json`` sidecar so zip installs work
    without Divine.exe. Divine is only a fallback for regenerating sidecars.
    """
    from gr2_dump import try_divine_dae

    base_gr2 = Path(base_gr2)
    sidecar = bind_sidecar_path(base_gr2)
    if not force_divine and sidecar.is_file():
        doc = load_bind_pose(sidecar)
        if not doc.get("source_gr2"):
            doc["source_gr2"] = base_gr2.name
        return doc

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="gr2lab_bind_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    ok = try_divine_dae(base_gr2, work_dir)
    dae = work_dir / "decoded.dae"
    if not ok or not dae.is_file():
        raise RuntimeError(
            f"No T-pose bind for {base_gr2.name}: missing {sidecar.name} and Divine.exe. "
            "Character rest pose will be wrong. Reinstall GR2 Lab 2.1+ (includes bind JSON)."
        )
    text = dae.read_text(encoding="utf-8", errors="replace")
    doc = parse_bind_from_dae(text, source_gr2=base_gr2.name)
    if not doc.get("bones"):
        raise RuntimeError(f"No JOINT bones in DAE for {base_gr2.name}")
    return doc


def bake_bind_sidecar(base_gr2: Path, *, work_dir: Optional[Path] = None) -> Path:
    """Run Divine and write ``<stem>.bind_pose.json`` next to the Body_Base GR2."""
    base_gr2 = Path(base_gr2)
    doc = extract_bind_pose(base_gr2, work_dir=work_dir, force_divine=True)
    return write_bind_pose(doc, bind_sidecar_path(base_gr2))


def bake_all_bind_sidecars(*, force_divine: bool = True) -> List[Dict[str, Any]]:
    """Bake bind JSON for every Body_Base/*.GR2. Used when shipping the addon."""
    results: List[Dict[str, Any]] = []
    for info in list_body_bases():
        path = Path(info["path"])
        sidecar = bind_sidecar_path(path)
        try:
            if force_divine:
                out = bake_bind_sidecar(path)
            elif sidecar.is_file():
                out = sidecar
            else:
                out = bake_bind_sidecar(path)
            doc = load_bind_pose(out)
            results.append(
                {
                    "ok": True,
                    "base": path.name,
                    "sidecar": str(out),
                    "bones": len(doc.get("bones") or []),
                }
            )
        except Exception as e:
            results.append({"ok": False, "base": path.name, "error": str(e)})
    return results


def write_bind_pose(doc: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_bind_pose(path: Path) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_bind_pose(doc)
    return doc


def validate_bind_pose(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("bind_pose: root must be object")
    if doc.get("format") != FORMAT_ID:
        raise ValueError(f"bind_pose: expected format={FORMAT_ID!r}")
    if not isinstance(doc.get("bones"), list) or not doc["bones"]:
        raise ValueError("bind_pose: missing bones[]")


def extract_bind_to_dump(
    dump_dir: Path,
    base_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract bind into OUT_FILES/<run>/bind_pose.json + BIND_SOURCE.txt."""
    from paths import OUT_FILES

    dump_dir = Path(dump_dir)
    if not dump_dir.is_dir():
        # allow run id
        dump_dir = OUT_FILES / Path(str(dump_dir)).name
    if not dump_dir.is_dir():
        raise FileNotFoundError(f"Dump not found: {dump_dir}")

    if not base_name:
        # try SOURCE.txt then BIND_SOURCE
        src = ""
        src_txt = dump_dir / "SOURCE.txt"
        if src_txt.is_file():
            for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("name="):
                    src = line.split("=", 1)[1].strip()
                    break
        base_name = match_body_base_for_source(src)
        if not base_name:
            raise ValueError(
                "No Body_Base match for this dump — pick a base in the UI "
                f"(source={src or '?'})"
            )

    base_path = resolve_body_base(base_name)
    work = dump_dir / "_bind_divine"
    doc = extract_bind_pose(base_path, work_dir=work)
    write_bind_pose(doc, dump_dir / "bind_pose.json")
    (dump_dir / "BIND_SOURCE.txt").write_text(
        f"name={base_path.name}\npath={base_path.resolve()}\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "run_id": dump_dir.name,
        "source_gr2": base_path.name,
        "bones": len(doc["bones"]),
        "root": doc.get("root"),
        "path": str(dump_dir / "bind_pose.json"),
    }


def load_bind_for_dump(dump_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(dump_dir) / "bind_pose.json"
    if not p.is_file():
        return None
    return load_bind_pose(p)


def apply_bind_parents_to_skeleton(
    skel: Dict[str, Any],
    bind: Dict[str, Any],
) -> Dict[str, Any]:
    """Override skeleton parent_name from Body_Base bind (LSLib-accurate hierarchy)."""
    by_bind = {b["name"]: b for b in bind.get("bones") or [] if b.get("name")}
    bones = []
    for b in skel.get("bones") or []:
        name = b.get("name")
        bb = by_bind.get(name) if name else None
        if bb is not None:
            nb = dict(b)
            # Prefer bind parent even when None (root); only fall back if key absent
            if "parent_name" in bb:
                nb["parent_name"] = bb.get("parent_name")
            bones.append(nb)
        else:
            bones.append(b)
    out = dict(skel)
    out["bones"] = bones
    return out


# Larian humanoid length tips. Animation GR2s omit them (no TransformTracks).
LEAF_END_PARENTS: Dict[str, str] = {
    "HeadEnd_M_endBone": "Head_M",
    "ToesEnd_L_endBone": "Toes_L",
    "ToesEnd_R_endBone": "Toes_R",
}
for _kind in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
    for _side in ("L", "R"):
        LEAF_END_PARENTS[f"{_kind}Finger4_{_side}_endBone"] = f"{_kind}Finger3_{_side}"


def expand_bind_map(
    bind_by_name: Dict[str, Dict[str, Any]],
    *,
    source_gr2: str = "",
    bind_source_gr2: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Union Body_Base sidecar bones so clip-missing *_endBone tips have bind TRS."""
    out = dict(bind_by_name or {})
    for src in (bind_source_gr2, source_gr2):
        if not src:
            continue
        try:
            matched = match_body_base_for_source(src)
            key = matched or Path(src).name
            doc = load_bind_pose(bind_pose_sidecar_path(key))
        except Exception:
            continue
        loaded = False
        for b in doc.get("bones") or []:
            name = b.get("name")
            if name and name not in out:
                out[name] = b
            if name:
                loaded = True
        if loaded:
            return out
    return out


def graft_leaf_end_bones(
    skeleton: List[Dict[str, Any]],
    bind_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Append missing Larian ``*_endBone`` tips for a complete display skeleton.

    Display / Edit-mode only. Callers must keep the clip skeleton for encode.
    """
    bind_by_name = bind_by_name or {}
    out = [dict(b) for b in skeleton if b.get("name")]
    by = {b["name"]: b for b in out}
    candidates: Dict[str, str] = dict(LEAF_END_PARENTS)
    for name, bb in bind_by_name.items():
        if not name or not name.endswith("_endBone"):
            continue
        parent = bb.get("parent_name") or candidates.get(name)
        if parent:
            candidates[name] = parent
    grafted: List[str] = []
    changed = True
    while changed:
        changed = False
        for name, parent in candidates.items():
            if name in by or parent not in by:
                continue
            bb = bind_by_name.get(name) or {}
            parent_b = by[parent]
            if bb.get("bind_translation") or bb.get("rest_translation"):
                t = list(bb.get("bind_translation") or bb.get("rest_translation") or [0, 0, 0])
                r = list(bb.get("bind_rotation") or bb.get("rest_rotation") or [0, 0, 0, 1])
                s = list(bb.get("bind_scale") or bb.get("rest_scale") or [1, 1, 1])
                parent_name = bb.get("parent_name") or parent
            else:
                pt = list(parent_b.get("rest_translation") or [0.04, 0.0, 0.0])
                px = abs(float(pt[0] or 0.0))
                if name.startswith("HeadEnd"):
                    t = [0.16, 0.0, 0.0]
                elif name.startswith("ToesEnd"):
                    t = [-(px or 0.08), 0.0, 0.0]
                else:
                    t = [px or 0.03, 0.0, 0.0]
                r = [0.0, 0.0, 0.0, 1.0]
                s = [1.0, 1.0, 1.0]
                parent_name = parent
            nb = {
                "name": name,
                "parent_name": parent_name,
                "rest_translation": t,
                "rest_rotation": r,
                "rest_scale": s,
                "gr2lab_grafted": True,
            }
            out.append(nb)
            by[name] = nb
            grafted.append(name)
            changed = True
    return out, grafted


def merge_bind_with_clip_skeleton(
    bind: Dict[str, Any],
    clip_skel: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Per anim bone: use bind TRS when name matches, else clip rest_*."""
    by_bind = {b["name"]: b for b in bind.get("bones") or [] if b.get("name")}
    out = []
    for b in clip_skel.get("bones") or []:
        name = b.get("name")
        if not name:
            continue
        bb = by_bind.get(name)
        if bb:
            # Bind parent wins (Body_Base / LSLib hierarchy) — required for IK targets
            out.append(
                {
                    "name": name,
                    "parent_name": bb.get("parent_name"),
                    "bind_translation": list(bb.get("bind_translation") or [0, 0, 0]),
                    "bind_rotation": list(bb.get("bind_rotation") or [0, 0, 0, 1]),
                    "bind_scale": list(bb.get("bind_scale") or [1, 1, 1]),
                    "from": "bind",
                }
            )
        else:
            out.append(
                {
                    "name": name,
                    "parent_name": b.get("parent_name"),
                    "bind_translation": list(b.get("rest_translation") or [0, 0, 0]),
                    "bind_rotation": list(b.get("rest_rotation") or [0, 0, 0, 1]),
                    "bind_scale": list(b.get("rest_scale") or [1, 1, 1]),
                    "from": "clip_t0",
                }
            )
    seen = {b["name"] for b in out}
    for bb in bind.get("bones") or []:
        name = bb.get("name")
        if not name or name in seen or not name.endswith("_endBone"):
            continue
        parent = bb.get("parent_name")
        if parent and parent not in seen:
            continue
        out.append(
            {
                "name": name,
                "parent_name": parent,
                "bind_translation": list(bb.get("bind_translation") or [0, 0, 0]),
                "bind_rotation": list(bb.get("bind_rotation") or [0, 0, 0, 1]),
                "bind_scale": list(bb.get("bind_scale") or [1, 1, 1]),
                "from": "bind_graft",
            }
        )
        seen.add(name)
    return out
