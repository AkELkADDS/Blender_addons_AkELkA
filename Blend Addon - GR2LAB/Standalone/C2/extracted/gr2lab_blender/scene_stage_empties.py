"""On GR2 import, find the BG3 timeline scene with a close filename and drop empties.

The .gr2 has the animation only. Positions live in ``*_Scene.lsx`` next to the
dialog timeline. Those filenames share the dialog tokens (Karlach, CRD, Date, …).
A leftover token such as Sex picks the stage, and the stage whose name contains
that token and "Start" is the placement at the beginning of that beat.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import bpy
    from mathutils import Euler, Matrix, Quaternion, Vector
except ImportError:  # matcher can be tested outside Blender
    bpy = None
    Matrix = None
    Quaternion = None

# Tokens that show up in clip filenames but are not the dialog or the stage.
_STOP = {
    "gr2lab",
    "gr2",
    "lab",
    "rig",
    "scene",
    "base",
    "el",
    "plyr",
    "sk",
    "comp",
    "fs",
    "ms",
    "mf",
    "vo",
    "tc",
    "in",
    "out",
    "int",
    "ext",
    "new",
    "red",
    "moor",
    "tallraces",
    "shortraces",
    "tall",
    "short",
    "longsnout",
    "shortsnout",
}
_BODY = {
    "dgb",
    "dwr",
    "hum",
    "hfl",
    "hrc",
    "gno",
    "gty",
    "tif",
    "elf",
    "ogr",
    "dra",
}
# Companion display name → the body prefix their clips use.
_COMPANION_BODY = {
    "karlach": "TIF",
    "laezel": "GTY",
    "gale": "HUM",
    "astarion": "ELF",
    "shadowheart": "HFL",
    "wyll": "HUM",
    "minthara": "DWR",
    "halsin": "ELF",
    "jaheira": "HFL",
    "minsc": "HUM",
}
_SHORT_OK = {"crd", "rom", "camp", "sd"}
# Clip / role suffixes in GR2 names (LOOP vs Start stage, PLYR vs EL partner).
_CLIP_KIND = frozenset({"loop", "intro", "outro"})
_ROLE_KIND = frozenset({"plyr", "el", "sk", "comp"})
_BEATS = frozenset(
    {
        "holdhands",
        "hug",
        "kiss",
        "grasps",
        "placehand",
        "fiddles",
        "romance",
        "sex",
        "toastkiss",
    }
)
_ZERO_GUID = "00000000-0000-0000-0000-000000000000"
_SPEAKER_RE = re.compile(
    r"<Track\b[^>]*\bName=\"([^\"]*)\"[^>]*\bType=\"ActorTrackSpeaker\""
    r"[^>]*\bActorId=\"([0-9a-fA-F-]{36})\"[^>]*\bIsPeanut=\"([^\"]+)\""
    r"[^>]*\bSpeakerId=\"(\d+)\"[^>]*\bIsPlayer=\"([^\"]+)\""
    r"[^>]*\bSceneActorType=\"([^\"]+)\"",
)
_STEM_RE = re.compile(
    r"^(?P<stem>.+?)(?:_\d{3}_[0-9a-fA-F-]{36})?_Scene$",
    re.I,
)
_SCENE_CACHE: Optional[List[Path]] = None
_FIND_SCENE_CACHE: Dict[str, Optional[Tuple[Path, int]]] = {}
_PLAN_CACHE: Dict[str, ScenePlan] = {}
_BINDING_FOR_SCENE_CACHE: Dict[Tuple[str, str], Optional[ClipBinding]] = {}
_SCENE_ROOT_CACHE: Dict[str, Optional[ET.Element]] = {}
_MAX_DIALOG_CANDIDATES = 10
_MAX_SCENE_BINDING_PROBES = 8
_TEMPLATE_STAGE_INDEX: Optional[Dict[str, str]] = None
_SOURCE_GUID_CACHE: Dict[Tuple[str, str], str] = {}
_INFER_SOURCE_CACHE: Dict[Tuple[str, str], str] = {}


def _scene_root_cached(scene_path: Path) -> Optional[ET.Element]:
    try:
        key = str(scene_path.resolve()).lower()
    except OSError:
        key = str(scene_path).lower()
    if key not in _SCENE_ROOT_CACHE:
        try:
            _SCENE_ROOT_CACHE[key] = ET.parse(scene_path).getroot()
        except (ET.ParseError, OSError):
            _SCENE_ROOT_CACHE[key] = None
    return _SCENE_ROOT_CACHE.get(key)


@dataclass
class Speaker:
    name: str
    actor_id: str
    speaker_id: int
    is_player: bool
    scene_actor_type: str


@dataclass
class ActorXform:
    actor_type: int
    template_id: str
    position: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]  # x, y, z, w


@dataclass
class StageChoice:
    name: str
    guid: str
    how: str


@dataclass
class PlannedEmpty:
    name: str
    speaker_id: int
    is_player: bool
    position: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    template_id: str
    actor_type: int = 0


@dataclass
class ScenePlan:
    scene_path: Path
    stage: StageChoice
    empties: List[PlannedEmpty] = field(default_factory=list)
    rig_empty_name: str = ""
    score: int = 0
    phase_index: int = -1
    clip_start_sec: float = 0.0
    clip_binding: Optional["ClipBinding"] = None


def _tokens(text: str) -> List[str]:
    return [p for p in re.split(r"[^A-Za-z0-9]+", text) if p]


def significant_tokens(text: str) -> List[str]:
    out: List[str] = []
    for raw in _tokens(text):
        low = raw.lower()
        if low in _STOP or low in _BODY:
            continue
        if low.isdigit():
            continue
        # Stage words (Sex, Bed) are 3 letters. Shorter codes stay on the whitelist.
        if len(raw) < 3 and low not in _SHORT_OK:
            continue
        out.append(raw)
    return out


def _token_weight(token: str) -> int:
    low = token.lower()
    if low in _SHORT_OK:
        return 3
    return len(token)


def body_prefix(name: str) -> str:
    for raw in _tokens(name):
        if raw.lower() in _BODY:
            return raw.upper()
    return ""


def rig_body_key(name: str) -> str:
    """Body family + size token (TIF_M vs TIF_FS), not just TIF."""
    toks = _tokens(name)
    for i, raw in enumerate(toks):
        if raw.lower() not in _BODY:
            continue
        fam = raw.upper()
        if i + 1 < len(toks):
            nxt = toks[i + 1].lower()
            if nxt in {"m", "f", "fs", "ms", "mf", "ss"}:
                return f"{fam}_{toks[i + 1].upper()}"
        return fam
    return ""


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _stem_is_nonsex(stem: str) -> bool:
    return "nonsex" in _norm_label(stem)


def _nonsex_stage_choice(
    stages: Sequence[StageChoice], gr2_path: str
) -> Optional[StageChoice]:
    """Scene file stage for NonSex clips (not restaurant mocap or sex beds)."""
    want_supine = "supine" in _norm_label(Path(gr2_path).stem)
    best: Optional[StageChoice] = None
    for stage in stages:
        norm = _norm_label(stage.name)
        if "nonsex" not in norm:
            continue
        if want_supine:
            if "supine" not in norm:
                continue
        elif "supine" in norm:
            continue
        if best is None or len(norm) < len(_norm_label(best.name)):
            best = stage
    return best


def _stage_is_nonsex_int(label: str) -> bool:
    norm = _norm_label(label)
    return "nonsex" in norm and "supine" not in norm


def _partner_index_pref(gr2_path: str) -> int:
    if _stem_is_nonsex(Path(gr2_path).stem):
        return 0
    return -1


def _clip_kind_tokens(name: str) -> set:
    return {t.lower() for t in _tokens(name) if t.lower() in _CLIP_KIND}


def _variant_suffix_tokens(name: str) -> set:
    """LOOP / IN / OUT clip suffixes on companion GR2 names (not stage INT/EXT)."""
    toks = {t.lower() for t in _tokens(Path(name).stem)}
    out = set(_clip_kind_tokens(name))
    if "in" in toks and "intro" not in toks:
        out.add("in")
    if "out" in toks and "outro" not in toks:
        out.add("out")
    return out


def _role_kind_tokens(name: str) -> set:
    return {t.lower() for t in _tokens(name) if t.lower() in _ROLE_KIND}


def _beat_in_stage(hint: str, stage_name: str) -> bool:
    """HoldHands matches 'Hold Hands Loop - INT'."""
    h = _norm_label(hint)
    if not h:
        return False
    return h in _norm_label(stage_name)


def _bg3_data_roots() -> List[Path]:
    """Installed game Data / paks are not a data source."""
    return []


def _lsx_cache_dir() -> Path:
    """Converted timeline LSX written by this addon. Not a decoder cache."""
    local = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(local) / "gr2lab" / "timeline_lsx"


def _unpacked_generated_dirs() -> List[Path]:
    try:
        from . import bg3_unpacked_resolve as unpacked
    except ImportError:
        return []
    extra = _addon_unpacked_dir()
    roots = unpacked.unpacked_roots([extra] if extra else None)
    out: List[Path] = []
    seen = set()
    for root in roots:
        gen = unpacked.generated_dir(root)
        if gen is None:
            continue
        try:
            key = str(gen.resolve()).lower()
        except OSError:
            key = str(gen).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(gen)
    return out


def _fixed_project_caches() -> List[Path]:
    """Decoder timeline caches are not a data source."""
    return []


def _decoder_timeline_caches(extra_roots: Optional[Sequence[Path]] = None) -> List[Path]:
    """Addon conversion cache, then unpacked Gustav Generated. Nothing else."""
    del extra_roots
    caches: List[Path] = []
    seen: set = set()

    def add(cache: Path) -> None:
        try:
            key = str(cache.resolve()).lower()
        except OSError:
            key = str(cache).lower()
        if key in seen or not cache.is_dir():
            return
        seen.add(key)
        caches.append(cache)

    add(_lsx_cache_dir())
    for gen in _unpacked_generated_dirs():
        add(gen)
    return caches


def _scene_paths_from_decoder_caches(
    caches: Sequence[Path],
    seen: set,
    found: List[Path],
) -> None:
    for cache in caches:
        for pattern in ("*_Scene.lsx", "*Scene.lsx"):
            for path in cache.glob(pattern):
                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(path)


def scene_files() -> List[Path]:
    global _SCENE_CACHE
    if _SCENE_CACHE is not None:
        return _SCENE_CACHE
    found: List[Path] = []
    seen = set()
    _scene_paths_from_decoder_caches(_decoder_timeline_caches(), seen, found)
    _SCENE_CACHE = found
    return found


def _decoder_scenes_for_gr2(gr2_path: str) -> List[Path]:
    """Decoder-cache scene files for this dialog (not the whole cache)."""
    found: List[Path] = []
    seen: set = set()
    for name in _timeline_names_for_gr2(gr2_path):
        path = _named_lsx(f"{name}_Scene.lsx")
        if path is not None:
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                found.append(path)
    caches = _decoder_timeline_caches([Path(gr2_path)])
    lookup_keys = _dialog_lookup_keys(gr2_path)
    for cache in caches:
        for key in lookup_keys:
            if not key:
                continue
            for path in cache.glob(f"*{key}*_Scene.lsx"):
                if not path.is_file() or not _scene_matches_dialog(
                    gr2_path, path, lookup_keys=lookup_keys
                ):
                    continue
                slot = str(path).lower()
                if slot in seen:
                    continue
                seen.add(slot)
                found.append(path)
    return found


def _path_rank(path: Path) -> Tuple[int, int]:
    text = str(path).replace("\\", "/").lower()
    if "/public/" in text and "/generated/" in text:
        band = 0
    elif "/public/" in text:
        band = 1
    else:
        band = 2
    return (band, len(text))


def score_scene_name(gr2_name: str, scene_name: str) -> int:
    gr2 = significant_tokens(Path(gr2_name).stem)
    scene = {t.lower() for t in significant_tokens(scene_name)}
    if not gr2 or not scene:
        return 0
    shared = [t for t in gr2 if t.lower() in scene]
    # One long word is not enough (every Karlach clip would hit every Karlach file
    # only when a second token agrees).
    if len({t.lower() for t in shared}) < 2:
        return 0
    score = sum(_token_weight(t) for t in shared)
    gr2_stem = _norm_label(Path(gr2_name).stem)
    scene_stem = _norm_label(scene_name)
    for token in gr2:
        beat = _norm_label(token)
        if len(beat) >= 6 and beat in scene_stem:
            score += 8
    if score < 10:
        return 0
    return score


def dialog_stem(scene_path: Path) -> str:
    m = _STEM_RE.match(scene_path.stem)
    if m:
        return m.group("stem")
    return scene_path.stem


def _dialog_lookup_keys(gr2_name: str) -> set:
    """Folder / timeline / SCENE_* tokens used to pick the dialog scene file."""
    keys: set = set()
    path = Path(gr2_name)
    tn = (_timeline_name_for_gr2(gr2_name) or "").strip()
    if tn:
        keys.add(tn.lower())
    for folder in (path.parent, path.parent.parent):
        name = folder.name
        if not name or name.lower() in {"gno_m_rig", "body_base", "__pycache__"}:
            continue
        keys.add(name.lower())
    stem = path.stem
    for beat in _BEATS:
        m = re.search(
            rf"_scene_(.+?)_{beat}(?:_|$)",
            stem,
            re.I,
        )
        if m:
            keys.add(m.group(1).lower())
            break
    norm = _norm_label(stem)
    if "karlach" in norm and "crd" in norm and "date" in norm:
        keys.add("camp_karlach_crd_rom_date")
        keys.add("camp_karlach_crd_date")
    return keys


def _forced_dialog_from_stem(gr2_name: str) -> List[str]:
    """Hard match for DGB/GNO export names without SCENE_CAMP_* in the stem."""
    norm = _norm_label(Path(gr2_name).stem)
    out: List[str] = []
    if "karlach" in norm and "crd" in norm and "date" in norm:
        out.extend(
            [
                "CAMP_Karlach_CRD_ROM_Date",
                "CAMP_Karlach_CRD_Date",
            ]
        )
    assets = _dialog_assets_index()
    for dialog in assets:
        dn = _norm_label(dialog)
        if not dn:
            continue
        if dn in norm or norm in dn:
            out.append(dialog)
        elif "karlach" in norm and "karlach" in dn and "crd" in dn and "date" in dn:
            out.append(dialog)
    seen: set = set()
    ordered: List[str] = []
    for name in out:
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        ordered.append(name)
    return ordered


def _scene_matches_dialog(
    gr2_name: str,
    scene_path: Path,
    *,
    lookup_keys: Optional[set] = None,
) -> bool:
    keys = lookup_keys if lookup_keys is not None else _dialog_lookup_keys(gr2_name)
    scene = dialog_stem(scene_path)
    scene_low = scene.lower()
    scene_norm = _norm_label(scene)
    for key in keys:
        if not key:
            continue
        kn = _norm_label(key)
        if scene_low == key.lower() or scene_norm == kn:
            return True
        if kn and (kn in scene_norm or scene_norm in kn):
            return True
    targets = set()
    for key in keys:
        targets |= {t.lower() for t in significant_tokens(key)}
    scene_tokens = {t.lower() for t in significant_tokens(scene)}
    if targets:
        need = min(3, len(targets))
        shared = sum(1 for t in targets if t in scene_tokens)
        if shared >= need:
            return True
    return score_scene_name(gr2_name, scene) >= 20


def _same_dialog_scene(a: Path, b: Path) -> bool:
    try:
        if a.resolve() == b.resolve():
            return True
    except OSError:
        pass
    return dialog_stem(a).lower() == dialog_stem(b).lower()


def _timeline_names_for_gr2(gr2_path: str) -> List[str]:
    """All dialog timelines that reference this clip (game bank)."""
    stem = Path(gr2_path).stem.lower()
    hint = Path(gr2_path)
    guid = _timeline_source_guid(stem, hint)
    names: List[str] = []
    if guid:
        names = list(_animation_timeline_index(hint).get(guid) or [])
    single = (_timeline_name_for_gr2(gr2_path) or "").strip()
    if single and single not in names:
        names.append(single)
    out: List[str] = []
    seen = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _gr2_resolve_key(gr2_name: str) -> str:
    p = Path(gr2_name)
    try:
        if p.is_file():
            return str(p.resolve()).lower()
    except OSError:
        pass
    return str(gr2_name).lower()


def _scene_path_key(scene_path: Path) -> str:
    try:
        return str(scene_path.resolve()).lower()
    except OSError:
        return str(scene_path).lower()


_GUID_TEXT_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_TIMELINE_SOURCES_CACHE: Dict[str, frozenset] = {}


def _timeline_animation_sources(scene_path: Path) -> frozenset:
    key = _scene_path_key(scene_path)
    cached = _TIMELINE_SOURCES_CACHE.get(key)
    if cached is not None:
        return cached
    sources: set = set()
    timeline = _timeline_beside_scene(scene_path)
    if timeline is not None:
        doc = _parse_timeline_bindings(timeline)
        for anim in doc.get("anims") or []:
            src = anim.get("source")
            if src:
                sources.add(str(src).lower())
    out = frozenset(sources)
    _TIMELINE_SOURCES_CACHE[key] = out
    return out


def _guids_in_gr2_file(path: Path) -> List[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    text = data.decode("latin-1", errors="ignore")
    found: List[str] = []
    seen: set = set()
    for match in _GUID_TEXT_RE.finditer(text):
        guid = match.group(0).lower()
        if guid in seen:
            continue
        seen.add(guid)
        found.append(guid)
    return found


def _is_companion_export_stem(stem_l: str) -> bool:
    body = body_prefix(stem_l)
    low = stem_l
    return any(
        re.search(rf"(?:^|_){comp}(?:_|$)", low) and body == prefix
        for comp, prefix in _COMPANION_BODY.items()
    )


def _locator_stem_for_gr2(gr2_path: str) -> str:
    """Map DGB/GNO export names (Sex_ash-rain) → game locator stem (…_Sex_PLYR)."""
    stem_l = Path(gr2_path).stem.lower()
    loc = _gr2_pak_locator()
    if stem_l in loc:
        return stem_l
    body = rig_body_key(stem_l)
    if not body:
        return ""
    hints = _filename_beat_hints(stem_l)
    is_comp = _is_companion_export_stem(stem_l)
    file_tokens = {
        t.lower() for t in significant_tokens(Path(gr2_path).stem)
    } | set(hints)
    best = ""
    best_score = 0
    for key, entry in loc.items():
        if rig_body_key(key) != body:
            continue
        key_l = key.lower()
        key_tokens = {t.lower() for t in _tokens(key)} - {"rig", "scene"}
        shared = file_tokens & key_tokens
        if not (
            {"karlach", "crd"}.issubset(shared)
            or {"camp", "karlach"}.issubset(shared)
            or len(shared) >= 4
        ):
            continue
        if is_comp:
            if "plyr" in key_l:
                continue
        elif "plyr" not in key_l:
            continue
        if _stem_is_nonsex(stem_l):
            if "nonsex" not in key_l:
                continue
        elif "sex" in hints:
            if "nonsex" in key_l or "sex" not in key_l:
                continue
        elif "nonsex" in key_l:
            continue
        score = sum(10 for hint in hints if hint in key_l)
        score += len(shared) * 4
        if "sex" in hints and "sex" in key_l:
            score += 14
        if score > best_score:
            best_score = score
            best = key
    return best if best_score >= 28 else ""


def _source_guid_from_gr2_bytes(gr2_path: Path, scene_path: Path) -> str:
    sources = _timeline_animation_sources(scene_path)
    if not sources or not gr2_path.is_file():
        return ""
    for guid in _guids_in_gr2_file(gr2_path):
        if guid in sources:
            return guid
    return ""


def _infer_sex_timeline_source(gr2_path: str, scene_path: Path) -> str:
    """Last resort: pick the sex-phase TLAnimation row for this body role."""
    stem_l = Path(gr2_path).stem.lower()
    hints = _filename_beat_hints(stem_l)
    if "sex" not in hints or _stem_is_nonsex(stem_l):
        return ""
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        return ""
    doc = _parse_timeline_bindings(timeline)
    anims = doc.get("anims") or []
    switches = doc.get("switches") or []
    actors = doc.get("actors") or {}
    root = _scene_root_cached(scene_path)
    is_comp = _is_companion_export_stem(stem_l)
    kinds = _clip_kind_tokens(gr2_path)
    want_start = "start" in kinds or "in" in kinds
    votes: Dict[str, int] = {}
    for anim in anims:
        src = str(anim.get("source") or "").lower()
        if not src:
            continue
        meta = actors.get(str(anim.get("actor") or "").lower(), {})
        at = int(meta.get("type") or 0)
        if is_comp:
            if at != 8:
                continue
        elif at != 2:
            continue
        stage_g = _stage_for_clip(
            switches, int(anim.get("phase", -1)), float(anim.get("start") or 0.0)
        )
        label = ""
        if root is not None and stage_g:
            label = _stage_display_name(root, stage_g)
        if not label and stage_g:
            label = _template_stage_name(scene_path, stage_g)
        norm = _norm_label(label)
        if "sex" not in norm or "non" in norm:
            continue
        is_start_stage = "start" in norm
        is_bed_stage = "bed" in norm
        if want_start and not is_start_stage:
            continue
        if not want_start and is_start_stage and not is_bed_stage:
            continue
        score = 10
        if is_bed_stage and not want_start:
            score += 24
        if is_start_stage and want_start:
            score += 24
        dialog_key = _norm_label(dialog_stem(scene_path))
        phase_n = int(anim.get("phase", -1))
        if "karlachcrdromdate" in dialog_key and phase_n == 109:
            score += 80
        elif "karlachcrdromdate" in dialog_key and phase_n > 110:
            score -= 50
        if anim.get("continuous"):
            score += 10
        if anim.get("pose"):
            score += 8
        try:
            score += int(float(anim.get("end") or 0) - float(anim.get("start") or 0))
        except (TypeError, ValueError):
            pass
        votes[src] = max(votes.get(src, 0), score)
    if not votes:
        return ""
    return max(votes.items(), key=lambda item: item[1])[0]


def _infer_nonsex_timeline_source(gr2_path: str, scene_path: Path) -> str:
    """Last resort: Non Sex - INT stage rows (not sex phase or restaurant mocap)."""
    stem_l = Path(gr2_path).stem.lower()
    if not _stem_is_nonsex(stem_l):
        return ""
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        return ""
    doc = _parse_timeline_bindings(timeline)
    anims = doc.get("anims") or []
    switches = doc.get("switches") or []
    actors = doc.get("actors") or {}
    root = _scene_root_cached(scene_path)
    is_comp = _is_companion_export_stem(stem_l)
    votes: Dict[str, int] = {}
    for anim in anims:
        src = str(anim.get("source") or "").lower()
        if not src:
            continue
        meta = actors.get(str(anim.get("actor") or "").lower(), {})
        at = int(meta.get("type") or 0)
        idx = int(meta.get("index") or 0)
        if is_comp:
            if at != 8 or idx != 0:
                continue
        elif at != 2 or idx != 0:
            continue
        stage_g = _stage_for_clip(
            switches,
            int(anim.get("phase", -1)),
            float(anim.get("start") or 0.0),
        )
        label = ""
        if root is not None and stage_g:
            label = _stage_display_name(root, stage_g)
        if not label and stage_g:
            label = _template_stage_name(scene_path, stage_g)
        if not _stage_is_nonsex_int(label):
            continue
        score = 80
        phase_n = int(anim.get("phase", -1))
        if phase_n == 103:
            score += 40
        try:
            when = float(anim.get("start") or 0.0)
        except (TypeError, ValueError):
            when = 0.0
        score -= when * 0.05
        votes[src] = max(votes.get(src, 0), score)
    if not votes:
        return ""
    return max(votes.items(), key=lambda item: item[1])[0]


def _timeline_source_guid_cached(gr2_name: str, scene_path: Path) -> str:
    stem_l = Path(gr2_name).stem.lower()
    scene_key = _scene_path_key(scene_path)
    cache_key = (stem_l, scene_key)
    if cache_key in _SOURCE_GUID_CACHE:
        return _SOURCE_GUID_CACHE[cache_key]
    guid = _timeline_source_guid(
        stem_l, Path(gr2_name) if gr2_name else None, scene_path
    )
    _SOURCE_GUID_CACHE[cache_key] = guid
    return guid


def _scene_supports_gr2(gr2_name: str, scene_path: Path) -> bool:
    """Fast check for find_scene_file probes (no full ClipBinding build)."""
    stem = Path(gr2_name).stem.lower()
    source = _timeline_source_guid_cached(gr2_name, scene_path)
    if not source:
        return False
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        return False
    doc = _parse_timeline_bindings(timeline)
    return any(a.get("source") == source for a in doc.get("anims") or [])


def _binding_for_scene(
    gr2_name: str,
    scene_path: Path,
    *,
    prefer_phase: int = -1,
):
    """Return clip binding when timeline + scene pair resolve; else None."""
    scene_key = _scene_path_key(scene_path)
    cache_key = (_gr2_resolve_key(gr2_name), scene_key, int(prefer_phase))
    if cache_key in _BINDING_FOR_SCENE_CACHE:
        return _BINDING_FOR_SCENE_CACHE[cache_key]
    root = _scene_root_cached(scene_path)
    if root is None:
        _BINDING_FOR_SCENE_CACHE[cache_key] = None
        return None
    binding = resolve_clip_binding(
        gr2_name, scene_path, root, prefer_phase=prefer_phase
    )
    if binding is not None:
        _BINDING_FOR_SCENE_CACHE[cache_key] = binding
    return binding


def _dialog_assets_index() -> Dict[str, Dict[str, str]]:
    try:
        from . import bg3_unpacked_resolve as _unpacked
    except ImportError:
        return {}
    extra = _addon_unpacked_dir()
    return _unpacked.generated_dialog_index([extra] if extra else None)


def _fuzzy_dialog_names(key: str, assets: Dict[str, Dict[str, str]]) -> List[str]:
    """Folder / token keys → Generated dialog names (e.g. CRD_Date → CRD_ROM_Date)."""
    kn = _norm_label(key)
    if not kn or not assets:
        return []
    out: List[str] = []
    seen = set()
    for dialog in assets:
        dn = _norm_label(dialog)
        if not dn:
            continue
        if dn == kn or kn in dn or dn in kn:
            pass
        elif kn.replace("rom", "") == dn.replace("rom", ""):
            pass
        else:
            continue
        low = dialog.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(dialog)
    return out


def _canonical_dialog_name(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    assets = _dialog_assets_index()
    if text in assets:
        return text
    low = text.lower()
    for dialog in assets:
        if dialog.lower() == low:
            return dialog
    fuzzy = _fuzzy_dialog_names(text, assets)
    if len(fuzzy) == 1:
        return fuzzy[0]
    if fuzzy:
        fuzzy.sort(key=lambda d: score_scene_name(text, d), reverse=True)
        return fuzzy[0]
    return text


def _candidate_dialog_names(gr2_name: str) -> List[str]:
    names: List[str] = []
    seen = set()
    for name in _timeline_names_for_gr2(gr2_name):
        key = name.lower()
        if key not in seen:
            seen.add(key)
            names.append(name)
    single = (_timeline_name_for_gr2(gr2_name) or "").strip()
    if single and single.lower() not in seen:
        seen.add(single.lower())
        names.append(single)
    assets = _dialog_assets_index()
    for key in _dialog_lookup_keys(gr2_name):
        if not key:
            continue
        for dialog in _fuzzy_dialog_names(key, assets) or [key]:
            kn = dialog.lower()
            if kn in seen:
                continue
            seen.add(kn)
            names.append(dialog)
    if not names:
        gr2_norm = _norm_label(Path(gr2_name).stem)
        fuzzy_extra: List[str] = []
        for dialog in assets:
            dn = _norm_label(dialog)
            if not dn:
                continue
            if dn in gr2_norm or gr2_norm in dn:
                fuzzy_extra.append(dialog)
        fuzzy_extra.sort(
            key=lambda d: score_scene_name(gr2_name, d),
            reverse=True,
        )
        for dialog in fuzzy_extra[:_MAX_DIALOG_CANDIDATES]:
            kn = dialog.lower()
            if kn in seen:
                continue
            seen.add(kn)
            names.append(dialog)
    names.sort(
        key=lambda d: score_scene_name(gr2_name, d),
        reverse=True,
    )
    return names[:_MAX_DIALOG_CANDIDATES]


def _scene_path_for_dialog(dialog_name: str) -> Optional[Path]:
    dialog_name = _canonical_dialog_name(dialog_name)
    assets = _dialog_assets_index()
    row = assets.get(dialog_name) or assets.get(dialog_name.strip())
    if row and row.get("scene"):
        path = Path(row["scene"])
        if path.is_file():
            return path
    try:
        from . import bg3_unpacked_resolve as _unpacked

        found = _unpacked.find_dialog_assets(dialog_name)
        scene = found.get("scene_lsx")
        if scene is not None and scene.is_file():
            return scene
    except ImportError:
        pass
    named = _named_lsx(f"{dialog_name}_Scene.lsx")
    return named


def find_scene_file(gr2_name: str, timer=None) -> Optional[Tuple[Path, int]]:
    import time

    from timing import maybe_timer

    cache_key = _gr2_resolve_key(gr2_name)
    if cache_key in _FIND_SCENE_CACHE:
        if timer is not None:
            timer.add("scene_find_cache_hit", 0.0)
        return _FIND_SCENE_CACHE[cache_key]

    probes = 0
    name_hit: Optional[Tuple[Path, int]] = None
    with maybe_timer(timer, "scene_find_candidates"):
        names = _candidate_dialog_names(gr2_name)
    for dialog in names:
        if probes >= _MAX_SCENE_BINDING_PROBES:
            break
        scene_path = _scene_path_for_dialog(dialog)
        if scene_path is None:
            continue
        t0 = time.perf_counter()
        if _timeline_beside_scene(scene_path) is None:
            _ensure_timeline_pair(dialog)
        if timer is not None:
            timer.add("scene_ensure_timeline", time.perf_counter() - t0)
        if _timeline_beside_scene(scene_path) is None:
            continue
        probes += 1
        score = score_scene_name(gr2_name, dialog_stem(scene_path))
        if score <= 0:
            score = score_scene_name(gr2_name, dialog) or 20
        t1 = time.perf_counter()
        bound = _scene_supports_gr2(gr2_name, scene_path)
        if timer is not None:
            timer.add("scene_binding_probe", time.perf_counter() - t1)
        if name_hit is None or score > name_hit[1]:
            name_hit = (scene_path, score)
        if bound:
            _binding_for_scene(gr2_name, scene_path)
            result = (scene_path, score)
            _FIND_SCENE_CACHE[cache_key] = result
            return result

    if os.environ.get("GR2LAB_FULL_SCENE_SCAN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        best: Optional[Path] = None
        best_score = 0
        for path in scene_files():
            sc = score_scene_name(gr2_name, path.stem)
            if sc <= 0:
                continue
            if _timeline_beside_scene(path) is None:
                continue
            if _binding_for_scene(gr2_name, path) is not None and sc > best_score:
                best_score = sc
                best = path
        if best is not None:
            result = (best, best_score)
            _FIND_SCENE_CACHE[cache_key] = result
            return result

    lookup = _dialog_lookup_keys(gr2_name)
    fallback_key = (_timeline_name_for_gr2(gr2_name) or "").strip() or next(
        iter(lookup), ""
    )
    fallback = _scene_path_for_dialog(_canonical_dialog_name(fallback_key))
    if fallback is not None and _scene_supports_gr2(gr2_name, fallback):
        _binding_for_scene(gr2_name, fallback)
        score = score_scene_name(gr2_name, dialog_stem(fallback)) or 10
        result = (fallback, score)
        _FIND_SCENE_CACHE[cache_key] = result
        return result

    # Clip stem is often absent from the animation index (variant suffixes).
    # Still place empties on the filename-matched dialog scene.
    if name_hit is not None and name_hit[1] < 28:
        for dialog in _forced_dialog_from_stem(gr2_name):
            scene_path = _scene_path_for_dialog(dialog)
            if scene_path is None or _timeline_beside_scene(scene_path) is None:
                continue
            score = score_scene_name(gr2_name, dialog_stem(scene_path))
            if score > name_hit[1]:
                name_hit = (scene_path, score)
                break
    if name_hit is not None:
        _FIND_SCENE_CACHE[cache_key] = name_hit
        return name_hit
    if fallback is not None:
        score = score_scene_name(gr2_name, dialog_stem(fallback)) or 10
        result = (fallback, score)
        _FIND_SCENE_CACHE[cache_key] = result
        return result

    _FIND_SCENE_CACHE[cache_key] = None
    return None


def _attr(node: ET.Element, key: str) -> Optional[ET.Element]:
    for child in list(node):
        if child.tag == "attribute" and child.get("id") == key:
            return child
    return None


def _attr_value(node: ET.Element, key: str, default: str = "") -> str:
    el = _attr(node, key)
    if el is None:
        return default
    return el.get("value") or default


def _vec(node: ET.Element, key: str, n: int) -> Optional[Tuple[float, ...]]:
    el = _attr(node, key)
    if el is None:
        return None
    raw = (el.get("value") or "").strip()
    if raw:
        parts = [float(p) for p in raw.split() if p]
        if len(parts) >= n:
            return tuple(parts[:n])
    for sub in list(el):
        if sub.tag not in {"float3", "float4"}:
            continue
        keys = ("x", "y", "z", "w")[:n]
        return tuple(float(sub.get(k) or 0.0) for k in keys)
    return None


def _child_nodes(node: ET.Element) -> List[ET.Element]:
    for child in list(node):
        if child.tag == "children":
            return [c for c in list(child) if c.tag == "node"]
    return []


def _find_child(node: ET.Element, node_id: str) -> Optional[ET.Element]:
    for child in _child_nodes(node):
        if child.get("id") == node_id:
            return child
    return None


def parse_stages(root: ET.Element) -> List[StageChoice]:
    stages: List[StageChoice] = []
    for node in root.iter("node"):
        if node.get("id") != "TLStage":
            continue
        name = _attr_value(node, "Name")
        guid = _attr_value(node, "Identifier")
        if name and guid:
            stages.append(StageChoice(name=name, guid=guid.lower(), how=""))
    return stages


def choose_stage(gr2_name: str, scene_name: str, stages: Sequence[StageChoice]) -> Optional[StageChoice]:
    if not stages:
        return None
    file_tokens = {t.lower() for t in significant_tokens(scene_name)}
    hints = []
    seen = set()
    for token in significant_tokens(gr2_name):
        low = token.lower()
        if low in file_tokens or low in seen:
            continue
        seen.add(low)
        if any(_beat_in_stage(low, stage.name) or low in stage.name.lower() for stage in stages):
            hints.append(low)
    for kind in _clip_kind_tokens(gr2_name):
        if kind not in seen:
            seen.add(kind)
            hints.append(kind)
    if not hints:
        opening = stages[0]
        return StageChoice(name=opening.name, guid=opening.guid, how="first stage in the scene file")

    gr2_kinds = _clip_kind_tokens(gr2_name)
    beat_hints = [h for h in hints if h not in _CLIP_KIND]

    def score(stage: StageChoice) -> int:
        name = stage.name.lower()
        words = {t.lower() for t in _tokens(stage.name)}
        total = 0
        for hint in hints:
            if _beat_in_stage(hint, stage.name) or hint in name:
                total += 10
        if "sex" in hints and "non" not in {t.lower() for t in _tokens(stage.name)}:
            if "sex" in name:
                total += 40
            if "hold" in name or "hands" in name:
                total -= 40
        if beat_hints and any(_beat_in_stage(h, stage.name) for h in beat_hints):
            if "sex" not in hints and ("hold" in name or "hands" in name):
                total += 14
            if "bronze" in name or ("camp" in name and "hold" not in name):
                total -= 28
        if "loop" in gr2_kinds:
            if "loop" in words:
                total += 18
            elif "hold" in name or "hands" in name:
                total += 12
            if "start" in words and "loop" not in words:
                total -= 14
        elif "start" in words:
            total += 6
        if "non" in words and "non" not in hints:
            total -= 8
        return total

    ranked = sorted(stages, key=lambda s: (-score(s), stages.index(s)))
    winner = ranked[0]
    if score(winner) <= 0:
        opening = stages[0]
        return StageChoice(name=opening.name, guid=opening.guid, how="first stage in the scene file")
    hint_txt = ", ".join(hints)
    return StageChoice(
        name=winner.name,
        guid=winner.guid,
        how=f"stage name matches {hint_txt} and is the start of that beat",
    )


def _pose_from_map_object(obj: ET.Element) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    value = _find_child(obj, "MapValue")
    if value is not None:
        pos = _vec(value, "Position", 3)
        quat = _vec(value, "RotationQuat", 4)
        if pos is not None and quat is not None:
            return (
                (float(pos[0]), float(pos[1]), float(pos[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
    for child in obj.findall("attribute"):
        if child.get("id") != "MapValue" or child.get("type") != "mat4x4":
            continue
        rows: List[List[float]] = []
        for f4 in child.findall(".//float4"):
            rows.append(
                [
                    float(f4.get("x") or 0),
                    float(f4.get("y") or 0),
                    float(f4.get("z") or 0),
                    float(f4.get("w") or 0),
                ]
            )
        if len(rows) < 3:
            continue
        tx, ty, tz = rows[0][3], rows[1][3], rows[2][3]
        yaw = math.atan2(rows[2][0], rows[0][0])
        hy = yaw * 0.5
        quat = (0.0, math.sin(hy), 0.0, math.cos(hy))
        return (float(tx), float(ty), float(tz)), quat
    return None


def _actor_transform(actor: ET.Element, stage_guid: str) -> Optional[ActorXform]:
    want = (stage_guid or "").lower()
    found = None
    fallback = None
    for obj in actor.findall('.//node[@id="Object"]'):
        key = _attr_value(obj, "MapKey").lower()
        if not key:
            continue
        packed = _pose_from_map_object(obj)
        if packed is None:
            continue
        if key == want:
            found = packed
            break
        if key == _ZERO_GUID and fallback is None:
            fallback = packed
    chosen = found or fallback
    if chosen is None:
        return None
    pos, quat = chosen
    try:
        actor_type = int(_attr_value(actor, "ActorType", "0") or 0)
    except ValueError:
        actor_type = 0
    return ActorXform(
        actor_type=actor_type,
        template_id=_attr_value(actor, "TemplateId").lower(),
        position=pos,
        quat=quat,
    )


def parse_actors(root: ET.Element, stage_guid: str) -> List[ActorXform]:
    actors: List[ActorXform] = []
    want = (stage_guid or "").lower()
    for node in root.iter("node"):
        if node.get("id") != "TLActor":
            continue
        xform = _actor_transform(node, want)
        if xform is None:
            continue
        if xform.actor_type not in {2, 8}:
            continue
        actors.append(xform)
    return actors


_NESTED_LSX_INDEX: Optional[Dict[str, str]] = None


def _nested_template_dirs() -> List[Path]:
    rel = Path("Public/Shared/Timeline/Scenes/Default/Nested Scene Templates")
    out: List[Path] = []
    seen = set()
    try:
        from . import bg3_unpacked_resolve as unpacked
    except ImportError:
        return out
    extra = _addon_unpacked_dir()
    for root in unpacked.unpacked_roots([extra] if extra else None):
        for folder in (
            root / "Shared" / rel,
            root / rel,
            root / "Shared" / "Public" / "Shared" / "Timeline" / "Scenes" / "Default" / "Nested Scene Templates",
        ):
            try:
                key = str(folder.resolve()).lower()
            except OSError:
                key = str(folder).lower()
            if key in seen or not folder.is_dir():
                continue
            seen.add(key)
            out.append(folder)
    return out


def _nested_lsx_basename_index() -> Dict[str, str]:
    global _NESTED_LSX_INDEX
    if _NESTED_LSX_INDEX is not None:
        return _NESTED_LSX_INDEX
    idx: Dict[str, str] = {}
    for folder in _nested_template_dirs():
        try:
            for path in folder.glob("*.lsx"):
                idx.setdefault(path.name.lower(), str(path))
        except OSError:
            continue
    _NESTED_LSX_INDEX = idx
    return idx


def _nested_scene_template_refs(root: ET.Element) -> List[str]:
    refs: List[str] = []
    seen: set = set()
    for attr in root.iter("attribute"):
        val = (attr.get("value") or "").replace("\\", "/")
        if "Nested Scene Templates/" not in val:
            continue
        low = val.lower()
        if not low.endswith(".lsx") and not low.endswith(".lsf"):
            continue
        if val not in seen:
            seen.add(val)
            refs.append(val)
    return refs


def _resolve_nested_scene_template(ref: str) -> Optional[Path]:
    rel = ref.replace("\\", "/").lstrip("/")
    name = Path(rel).name
    if name.lower().endswith(".lsf"):
        name = name[:-4] + ".lsx"
        if rel.lower().endswith(".lsf"):
            rel = rel[:-4] + ".lsx"
    try:
        from . import bg3_unpacked_resolve as unpacked
    except ImportError:
        unpacked = None
    if unpacked is not None:
        extra = _addon_unpacked_dir()
        for root in unpacked.unpacked_roots([extra] if extra else None):
            for candidate in (
                root / rel,
                root / "Shared" / rel,
                root / "Gustav" / rel,
            ):
                if candidate.is_file():
                    return candidate
    for cache in _decoder_timeline_caches():
        candidate = cache / name
        if candidate.is_file():
            return candidate
    hit = _nested_lsx_basename_index().get(name.lower())
    return Path(hit) if hit else None


def _actor_scene_roots(scene_path: Path) -> List[ET.Element]:
    roots: List[ET.Element] = []
    main = _scene_root_cached(scene_path)
    if main is None:
        return roots
    roots.append(main)
    for ref in _nested_scene_template_refs(main):
        tmpl = _resolve_nested_scene_template(ref)
        if tmpl is None:
            continue
        sub = _scene_root_cached(tmpl)
        if sub is not None:
            roots.append(sub)
    return roots


def parse_actors_for_scene(scene_path: Path, stage_guid: str) -> List[ActorXform]:
    want = (stage_guid or "").lower()
    out: List[ActorXform] = []
    seen: set = set()
    for root in _actor_scene_roots(scene_path):
        for actor in parse_actors(root, want):
            key = (
                actor.actor_type,
                actor.template_id,
                actor.position,
                actor.quat,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(actor)
    return out


def _stage_has_actors(scene_path: Path, stage: StageChoice) -> bool:
    return bool(parse_actors_for_scene(scene_path, (stage.guid or "").lower()))


def _resolve_placement_stage(
    gr2_path: str,
    scene_path: Path,
    root: ET.Element,
    stages: Sequence[StageChoice],
    binding: Optional["ClipBinding"],
) -> Optional[StageChoice]:
    """Timeline stage GUIDs often lack TLActor MapKeys — pick a stage that has transforms."""
    if not stages:
        return None

    ordered: List[StageChoice] = []
    if binding is not None and binding.stage_guid:
        g = binding.stage_guid.lower()
        ordered.append(
            StageChoice(
                name=binding.stage_name or g[:8],
                guid=g,
                how="dialog timeline phase that plays this animation",
            )
        )
    chosen = choose_stage(gr2_path, scene_path.stem, stages)
    if chosen is not None:
        ordered.append(chosen)
    if binding is not None and binding.stage_name:
        bn = _norm_label(binding.stage_name)
        for st in stages:
            sn = _norm_label(st.name)
            if bn and (bn in sn or sn in bn or _beat_in_stage(bn, st.name)):
                ordered.append(st)
    ordered.extend(stages)

    seen: set = set()
    for st in ordered:
        g = (st.guid or "").lower()
        if g in seen:
            continue
        seen.add(g)
        if _stage_has_actors(scene_path, st):
            return StageChoice(name=st.name, guid=g, how=st.how or "stage with actor transforms")
    if binding is not None and binding.stage_guid:
        g = binding.stage_guid.lower()
        if _stage_has_actors(scene_path, StageChoice(name=g[:8], guid=g, how="")):
            return StageChoice(
                name=binding.stage_name or g[:8],
                guid=g,
                how="nested template stage (timeline GUID)",
            )
    return None


def _tml_candidates(scene_path: Path) -> Iterable[Path]:
    stem = dialog_stem(scene_path)
    yield scene_path.with_name(stem + ".tml")
    yield scene_path.parent / (stem + ".tml")
    for data in _bg3_data_roots():
        gen = data / "Editor" / "Mods" / "GustavDev" / "Timeline" / "Generated"
        yield gen / (stem + ".tml")
        pub = data / "Public" / "GustavDev" / "Timeline" / "Generated"
        yield pub / (stem + ".tml")


def load_speakers(scene_path: Path) -> List[Speaker]:
    for path in _tml_candidates(scene_path):
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()[:80]
        except OSError:
            continue
        if b"<" not in raw[:40] and not raw.startswith(b"\xef\xbb\xbf"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        speakers: List[Speaker] = []
        for m in _SPEAKER_RE.finditer(text):
            if m.group(3).lower() == "true":
                continue
            speakers.append(
                Speaker(
                    name=m.group(1) or "",
                    actor_id=m.group(2).lower(),
                    speaker_id=int(m.group(4)),
                    is_player=m.group(5).lower() == "true",
                    scene_actor_type=m.group(6),
                )
            )
        if speakers:
            return speakers
    return []


def _safe_name(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return cleaned or fallback


def _label_actors(actors: Sequence[ActorXform], speakers: Sequence[Speaker]) -> List[PlannedEmpty]:
    used = set()
    pairs: List[Tuple[ActorXform, Optional[Speaker]]] = []
    unmatched: List[ActorXform] = []
    by_id = {s.actor_id: s for s in speakers}
    for actor in actors:
        speaker = by_id.get(actor.template_id)
        if speaker is not None and id(speaker) not in used:
            used.add(id(speaker))
            pairs.append((actor, speaker))
        else:
            unmatched.append(actor)

    def take(pred) -> Optional[Speaker]:
        for speaker in speakers:
            if id(speaker) in used or not pred(speaker):
                continue
            used.add(id(speaker))
            return speaker
        return None

    for actor in unmatched:
        if actor.actor_type == 2:
            speaker = take(lambda s: s.is_player or s.scene_actor_type == "Initiator")
        elif actor.actor_type == 8:
            speaker = take(lambda s: s.scene_actor_type == "Additional")
        else:
            speaker = None
        pairs.append((actor, speaker))

    counts = {"Initiator": 0, "Additional": 0, "Actor": 0}
    planned: List[PlannedEmpty] = []
    seen_names = set()
    for actor, speaker in pairs:
        if speaker is not None and speaker.name:
            name = _safe_name(speaker.name, "Actor")
            speaker_id = speaker.speaker_id
            is_player = speaker.is_player
        elif actor.actor_type == 2:
            counts["Initiator"] += 1
            name = f"Initiator_{counts['Initiator']}"
            speaker_id = 50 + counts["Initiator"]
            is_player = True
        elif actor.actor_type == 8:
            counts["Additional"] += 1
            name = f"Additional_{counts['Additional']}"
            speaker_id = counts["Additional"] - 1
            is_player = False
        else:
            counts["Actor"] += 1
            name = f"Actor_{counts['Actor']}"
            speaker_id = 80 + counts["Actor"]
            is_player = False
        base = name
        n = 2
        while name.lower() in seen_names:
            name = f"{base}_{n}"
            n += 1
        seen_names.add(name.lower())
        planned.append(
            PlannedEmpty(
                name=name,
                speaker_id=speaker_id,
                is_player=is_player,
                position=actor.position,
                quat=actor.quat,
                template_id=actor.template_id,
                actor_type=int(actor.actor_type),
            )
        )
    return planned


def _closest_empty_pair(
    empties: Sequence[PlannedEmpty],
) -> Tuple[Optional[Tuple[PlannedEmpty, PlannedEmpty]], float]:
    best: Optional[Tuple[PlannedEmpty, PlannedEmpty]] = None
    best_d = 1e9
    for i, left in enumerate(empties):
        for right in empties[i + 1 :]:
            dx = left.position[0] - right.position[0]
            dy = left.position[1] - right.position[1]
            dz = left.position[2] - right.position[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < best_d:
                best_d = dist
                best = (left, right)
    return best, best_d


def _sex_bed_pair_names(
    gr2_name: str, empties: Sequence[PlannedEmpty]
) -> Tuple[str, str]:
    """Two stage slots that form the intimacy bed (closest pair in Larian space)."""
    if "sex" not in _filename_beat_hints(Path(gr2_name).stem):
        return ("", "")
    if len(empties) < 3:
        return ("", "")
    pair, dist = _closest_empty_pair(empties)
    if pair is None or dist > 1.6:
        return ("", "")
    return pair[0].name, pair[1].name


def _sex_bed_seat(gr2_name: str, empties: Sequence[PlannedEmpty]) -> str:
    """Sex Start keeps the couple on the two closest slots, not the standing Initiator."""
    left, right = _sex_bed_pair_names(gr2_name, empties)
    if not left:
        return ""
    pair = tuple(e for e in empties if e.name in (left, right))
    if len(pair) != 2:
        return ""
    ordered = sorted(pair, key=lambda e: (e.speaker_id, e.name))
    low = Path(gr2_name).stem.lower()
    body = body_prefix(gr2_name)
    companion = any(
        re.search(rf"(?:^|_){comp}(?:_|$)", low) and body == prefix
        for comp, prefix in _COMPANION_BODY.items()
    )
    return ordered[-1].name if companion else ordered[0].name


def rig_empty_name(gr2_name: str, empties: Sequence[PlannedEmpty]) -> str:
    """Which empty this dropped clip belongs on. Empty string if it is unclear."""
    if not empties:
        return ""
    bed = _sex_bed_seat(gr2_name, empties)
    if bed:
        return bed
    low = Path(gr2_name).stem.lower()
    body = body_prefix(gr2_name)
    playerish = "plyr" in _tokens(low) or "_sk_" in f"_{low}_"
    companionish = any(flag in _tokens(low) for flag in ("el", "comp"))
    additionals = sorted(
        [e for e in empties if not e.is_player],
        key=lambda e: (e.speaker_id, e.name),
    )
    roles = _partner_role_names(gr2_name)
    if roles and additionals:
        role = next(iter(sorted(roles)))
        if role in _COMPANION_BODY and len(additionals) > 1:
            return max(additionals, key=lambda e: (e.speaker_id, e.name)).name
        if role not in _COMPANION_BODY:
            return min(additionals, key=lambda e: (e.speaker_id, e.name)).name
    matched_companion = None
    for comp, prefix in _COMPANION_BODY.items():
        # KarlachUpgrades is the dialog, not the Karlach body.
        if re.search(rf"(?:^|_){comp}(?:_|$)", low) and body == prefix:
            matched_companion = comp
            break
    if playerish and not companionish:
        matched_companion = None
    if matched_companion and additionals:
        if len(additionals) > 1:
            return max(additionals, key=lambda e: (e.speaker_id, e.name)).name
        return min(additionals, key=lambda e: (e.speaker_id, e.name)).name
    players = [e for e in empties if e.is_player]
    if playerish and players:
        return min(players, key=lambda e: (e.speaker_id, e.name)).name
    return ""


def _build_scene_plan(
    gr2_path: str,
    scene_path: Path,
    score: int,
    stage: StageChoice,
    root: ET.Element,
    timer=None,
) -> Optional[ScenePlan]:
    from timing import maybe_timer

    actors = parse_actors_for_scene(scene_path, stage.guid)
    if not actors:
        return None
    with maybe_timer(timer, "plan_load_speakers"):
        speakers = load_speakers(scene_path)
    empties = _label_actors(actors, speakers)
    return ScenePlan(
        scene_path=scene_path,
        stage=stage,
        empties=empties,
        rig_empty_name=rig_empty_name(gr2_path, empties),
        score=score,
    )


def _beat_tokens(name: str) -> set:
    return {t.lower() for t in _tokens(name) if t.lower() in _BEATS}


def _filename_beat_hints(stem: str) -> set:
    """Beat words from a GR2 filename (HoldHands → holdhands)."""
    norm = _norm_label(stem)
    if "nonsex" in norm:
        return {"nonsex"}
    hints = set(_beat_tokens(stem))
    for tok in significant_tokens(stem):
        tok_norm = _norm_label(tok)
        if len(tok_norm) < 5:
            continue
        for beat in _BEATS:
            if beat in tok_norm:
                hints.add(beat)
    if "sex" in norm:
        hints.add("sex")
    return hints


@dataclass
class ClipBinding:
    """What the dialog timeline says this GR2 is doing."""

    source_id: str
    stage_guid: str
    stage_name: str
    actor_type: int
    actor_index: int
    partner_stem: str = ""
    partner_actor_type: int = 0
    partner_actor_index: int = -1
    pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    partner_pose: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    play_start: float = 0.0
    partner_play_start: float = 0.0
    phase_index: int = -1
    clip_start: float = 0.0
    clip_end: float = 0.0


_TIMELINE_CACHE: Dict[str, Dict] = {}
_ANIM_INDEX: Optional[Dict[str, str]] = None
_TIMELINE_INDEX: Optional[Dict[str, List[str]]] = None
_ENSURE_TRIED: set = set()


def _animation_source_index(hint: Optional[Path] = None) -> Dict[str, str]:
    """GR2 stem → animation resource GUID, from a Glasses/game extract."""
    global _ANIM_INDEX
    if _ANIM_INDEX is None:
        _ANIM_INDEX = {}
    index = _ANIM_INDEX
    caches = list(_decoder_timeline_caches([hint] if hint is not None else None))
    if hint is not None:
        caches.append(hint if hint.is_dir() else hint.parent)
    seen = set()
    for cache in caches:
        path = cache / "animation_source_index.json"
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            for name, guid in raw.items():
                if name and guid:
                    index[str(name).lower()] = str(guid).lower()
    return index


def _animation_timeline_index(hint: Optional[Path] = None) -> Dict[str, List[str]]:
    """Animation resource GUID → dialog timeline names, from the game timeline bank."""
    global _TIMELINE_INDEX
    if _TIMELINE_INDEX is not None:
        return _TIMELINE_INDEX
    index: Dict[str, List[str]] = {}
    caches = list(_decoder_timeline_caches([hint] if hint is not None else None))
    if hint is not None:
        caches.append(hint if hint.is_dir() else hint.parent)
    for cache in caches:
        path = cache / "animation_timeline_index.json"
        if not path.is_file():
            continue
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        for guid, names in raw.items():
            if not guid or not isinstance(names, list):
                continue
            index[str(guid).lower()] = [str(name) for name in names if name]
        break
    _TIMELINE_INDEX = index
    return index


def _performance_tokens(name: str) -> List[str]:
    """Beat words only: HoldHands, not CAMP / LOOP / PLYR / the body prefix."""
    return [t.lower() for t in significant_tokens(name) if t.lower() not in _SIBLING_SKIP]


def _scene_folder_key(internal: str) -> str:
    """Content folder for a scene GR2. Race takes live in a nested ``*_Rig`` directory."""
    parent = Path(internal).parent
    if parent.name.lower().endswith("_rig"):
        parent = parent.parent
    return str(parent).replace("\\", "/").lower()


def _same_role(a: str, b: str) -> bool:
    """Player clips pair with player clips. Companion clips stay with companions."""
    def plyr(name: str) -> bool:
        return "plyr" in {t.lower() for t in _tokens(name)}

    return plyr(a) == plyr(b)


def _indexed_sibling_stem(stem: str, hint: Optional[Path] = None) -> str:
    """Take in this scene folder that a timeline actually plays.

    In/out/loop files and the other race versions often never appear as an
    animation id. The timeline names one body. The rest stand on that stage.
    Same body wins. Another race is used only when the beat and the player
    role match.
    """
    loc = _gr2_pak_locator().get(stem.lower())
    folders = _LOCATOR_FOLDERS or {}
    if not loc or not folders:
        return ""
    siblings = folders.get(_scene_folder_key(loc[1]), [])
    anim = _animation_source_index(hint)
    timelines = _animation_timeline_index(hint)
    mine = _performance_tokens(stem)
    mine_set = set(mine)
    body = rig_body_key(stem)
    if not mine_set or not body:
        return ""
    best_same = ""
    best_same_score = 0
    best_cross = ""
    best_cross_score = 0
    for other in siblings:
        if other == stem.lower() or not _same_role(stem, other):
            continue
        guid = anim.get(other, "")
        if not guid or not timelines.get(guid):
            continue
        other_beats = _performance_tokens(other)
        shared = mine_set & set(other_beats)
        if not shared:
            continue
        shorter = mine if len(mine) <= len(other_beats) else other_beats
        if not all(token in shared for token in shorter):
            continue
        score = sum(len(token) for token in shared)
        if rig_body_key(other) == body:
            if score > best_same_score:
                best_same_score = score
                best_same = other
        elif score > best_cross_score:
            best_cross_score = score
            best_cross = other
    return best_same or best_cross


def _anim_profile(stem: str) -> Tuple[frozenset, str, frozenset, bool]:
    stem_l = (stem or "").lower()
    return (
        frozenset(_performance_tokens(stem_l)),
        rig_body_key(stem_l),
        frozenset(_clip_kind_tokens(stem_l)),
        "plyr" in stem_l,
    )


def _infer_source_guid_from_timeline(
    stem: str,
    scene_path: Path,
    *,
    gr2_path: Optional[Path] = None,
    extra_stems: Optional[Sequence[str]] = None,
) -> str:
    """When animation_source_index misses a stem, vote on timeline player clips."""
    stem_l = (stem or "").lower()
    scene_key = _scene_path_key(scene_path)
    cache_key = (stem_l, scene_key)
    if cache_key in _INFER_SOURCE_CACHE:
        return _INFER_SOURCE_CACHE[cache_key]
    loc = _gr2_pak_locator().get(stem_l)
    if not loc and gr2_path is not None:
        canon = _locator_stem_for_gr2(str(gr2_path))
        if canon:
            loc = _gr2_pak_locator().get(canon)
            stem_l = canon
    if not loc:
        _INFER_SOURCE_CACHE[cache_key] = ""
        return ""
    folder_path = _scene_folder_key(loc[1])
    my_prof = _anim_profile(stem_l)
    locator = _gr2_pak_locator()
    profile_stems = [
        s
        for s, entry in locator.items()
        if _scene_folder_key(entry[1]) == folder_path and _anim_profile(s) == my_prof
    ]
    if stem_l not in profile_stems:
        profile_stems = list(profile_stems)
        if extra_stems:
            for s in extra_stems:
                sl = (s or "").lower()
                if sl and sl not in profile_stems:
                    profile_stems.append(sl)
        if stem_l not in profile_stems:
            profile_stems.append(stem_l)
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        _INFER_SOURCE_CACHE[cache_key] = ""
        return ""
    doc = _parse_timeline_bindings(timeline)
    actors = doc.get("actors") or {}
    anims = doc.get("anims") or []
    anim_index = _animation_source_index(scene_path)
    gr2_sources: set = set()
    if gr2_path is not None and gr2_path.is_file():
        gr2_sources = set(_guids_in_gr2_file(gr2_path)) & set(
            _timeline_animation_sources(scene_path)
        )
    is_plyr = my_prof[3]
    kinds = my_prof[2]
    want_loop = "loop" in kinds
    want_in = "in" in kinds
    want_out = "out" in kinds
    src_from_profile: Dict[str, int] = {}
    for s in profile_stems:
        g = anim_index.get(s)
        if g:
            src_from_profile[g] = src_from_profile.get(g, 0) + 1

    def _actor_ok(actor_id: str) -> bool:
        meta = actors.get(actor_id, {})
        at = int(meta.get("type") or 0)
        if is_plyr:
            return at == 2
        return at == 8

    guid_votes: Dict[str, int] = {}
    for anim in anims:
        if not _actor_ok(anim.get("actor") or ""):
            continue
        if want_loop and not anim.get("continuous"):
            continue
        if want_in and anim.get("continuous"):
            continue
        if want_out and anim.get("continuous"):
            continue
        src = anim.get("source") or ""
        if not src:
            continue
        boost = src_from_profile.get(src, 0)
        if boost <= 0 and src not in gr2_sources:
            continue
        score = 1 + boost * 12
        if src in gr2_sources:
            score += 40
        guid_votes[src] = max(guid_votes.get(src, 0), score)
    if not guid_votes:
        _INFER_SOURCE_CACHE[cache_key] = ""
        return ""
    picked = max(guid_votes.items(), key=lambda item: item[1])[0]
    _INFER_SOURCE_CACHE[cache_key] = picked
    return picked


def _timeline_source_guid(
    stem: str,
    hint: Optional[Path] = None,
    scene_path: Optional[Path] = None,
) -> str:
    """Animation id to read the timeline with: this file, or its indexed sibling."""
    anim = _animation_source_index(hint)
    timelines = _animation_timeline_index(hint)
    stem_l = (stem or "").lower()
    gr2_file = hint if hint is not None and hint.suffix.lower() == ".gr2" else None
    canon = ""
    if gr2_file is not None:
        canon = _locator_stem_for_gr2(str(gr2_file))
    guid = anim.get(stem_l, "")
    if guid and timelines.get(guid):
        return guid
    if canon:
        guid = anim.get(canon, "")
        if guid and timelines.get(guid):
            return guid
    sibling = _indexed_sibling_stem(stem_l, hint)
    if sibling:
        guid = anim.get(sibling, "")
        if guid and timelines.get(guid):
            return guid
    if scene_path is not None:
        if gr2_file is not None:
            from_bytes = _source_guid_from_gr2_bytes(gr2_file, scene_path)
            if from_bytes:
                return from_bytes
        infer_stem = canon or stem_l
        inferred = _infer_source_guid_from_timeline(
            infer_stem,
            scene_path,
            gr2_path=gr2_file,
            extra_stems=[stem_l] if stem_l != infer_stem else None,
        )
        if inferred:
            return inferred
        if gr2_file is not None:
            nonsex_src = _infer_nonsex_timeline_source(str(gr2_file), scene_path)
            if nonsex_src:
                return nonsex_src
            sex_src = _infer_sex_timeline_source(str(gr2_file), scene_path)
            if sex_src:
                return sex_src
    return ""


def _timeline_name_from_pak_folder(stem: str) -> str:
    loc = _gr2_pak_locator().get(stem.lower())
    if not loc:
        return ""
    return Path(_scene_folder_key(loc[1])).name


def _timeline_name_for_gr2(gr2_path: str) -> str:
    """Dialog whose timeline plays this GR2, from the game bank rather than a filename guess."""
    stem = Path(gr2_path).stem.lower()
    hint = Path(gr2_path)
    guid = _timeline_source_guid(stem, hint)
    if not guid:
        folder = _canonical_dialog_name(_timeline_name_from_pak_folder(stem))
        if folder:
            _ensure_timeline_pair(folder)
        return folder
    names = _animation_timeline_index(hint).get(guid) or []
    if not names:
        return ""
    best = ""
    best_score = 0
    for name in names:
        score = score_scene_name(gr2_path, name)
        if score > best_score:
            best_score = score
            best = name
    if best:
        return best
    loc = _gr2_pak_locator().get(stem)
    folder_name = Path(_scene_folder_key(loc[1])).name if loc else ""
    if folder_name:
        for name in names:
            if name.lower() == folder_name:
                return name
    if len(names) == 1:
        return names[0]
    return ""


def _named_lsx(filename: str) -> Optional[Path]:
    for cache in _decoder_timeline_caches():
        path = cache / filename
        if path.is_file() and path.stat().st_size > 1000:
            return path
    return None


def _primary_timeline_cache() -> Optional[Path]:
    path = _lsx_cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def _divine_exe() -> Optional[Path]:
    root = Path(__file__).resolve().parent
    for exe in (
        root / "tools" / "ExportTool-v1.20.4" / "Packed" / "Tools" / "Divine.exe",
        root / "tools" / "Divine.exe",
    ):
        if exe.is_file():
            return exe
    return None


def _timeline_paks() -> List[Path]:
    paks: List[Path] = []
    for data in _bg3_data_roots():
        found = []
        try:
            found = [p for p in data.glob("*.pak") if p.is_file()]
        except OSError:
            continue
        for pak in found:
            low = pak.name.lower()
            if any(tok in low for tok in ("gustav", "shared", "patch", "game")):
                paks.append(pak)
    def rank(path: Path) -> Tuple[int, str]:
        low = path.name.lower()
        if low.startswith("gustav"):
            band = 0
        elif low.startswith("patch"):
            band = 1
        else:
            band = 2
        return (band, low)
    paks.sort(key=rank)
    return paks


def _divine_run(exe: Path, args: Sequence[str]) -> bool:
    try:
        result = subprocess.run(
            [str(exe), "-g", "bg3", *args],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(exe.parent),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _convert_generated_lsf(exe: Path, extract_dir: Path, filename: str, dest: Path) -> Optional[Path]:
    hits = [
        path
        for path in extract_dir.rglob(filename)
        if "timeline" in str(path).lower() and "generated" in str(path).lower()
    ]
    if not hits:
        hits = list(extract_dir.rglob(filename))
    if not hits:
        return None
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    if not _divine_run(
        exe,
        ["-a", "convert-resource", "-s", str(hits[0].resolve()), "-d", str(dest.resolve())],
    ):
        return None
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    return None


def _addon_unpacked_dir() -> str:
    if bpy is None:
        return ""
    try:
        prefs = bpy.context.preferences.addons.get("gr2lab_blender")
        if prefs is not None:
            return (getattr(prefs.preferences, "bg3_unpacked_dir", None) or "").strip()
    except (AttributeError, TypeError):
        pass
    return ""


def _timeline_convert_progress(label: str) -> None:
    if bpy is None or not label:
        return
    try:
        bpy.context.window_manager.status_text_set(f"GR2 LAB: {label}")
    except (AttributeError, RuntimeError, TypeError):
        pass


def _ensure_timeline_pair(timeline_name: str) -> Optional[Path]:
    """Cache the dialog timeline and its scene. Returns the scene lsx."""
    if not timeline_name:
        return None
    scene_name = f"{timeline_name}_Scene.lsx"
    timeline_file = f"{timeline_name}.lsx"
    scene = _named_lsx(scene_name)
    timeline = _named_lsx(timeline_file)
    if scene is not None and timeline is not None:
        return scene
    key = timeline_name.lower()
    if key in _ENSURE_TRIED:
        return scene
    _ENSURE_TRIED.add(key)
    exe = _divine_exe()
    cache = _primary_timeline_cache()
    if exe is None or cache is None:
        return scene
    try:
        from . import bg3_unpacked_resolve as _unpacked
    except ImportError:
        _unpacked = None
    if _unpacked is not None:
        extra_dir = _addon_unpacked_dir()
        extra = [extra_dir] if extra_dir else None
        roots = _unpacked.unpacked_roots(extra)
        if roots or _unpacked.timeline_data_out() is not None:
            tl_path, sc_path = _unpacked.ensure_lsx_from_loose(
                timeline_name,
                cache,
                exe,
                _divine_run,
                extra_roots=extra,
                progress=_timeline_convert_progress,
            )
            if timeline is None and tl_path is not None:
                timeline = tl_path
            if scene is None and sc_path is not None:
                scene = sc_path
            if scene is not None and timeline is not None:
                return scene
    return scene


def _timeline_beside_scene(scene_path: Path) -> Optional[Path]:
    dialog = dialog_stem(scene_path)
    beside = scene_path.with_name(f"{dialog}.lsx")
    if beside.is_file() and beside.resolve() != scene_path.resolve():
        return beside
    cache = _primary_timeline_cache()
    if cache is not None:
        cached = cache / f"{dialog}.lsx"
        if cached.is_file() and cached.stat().st_size > 1000:
            return cached
    try:
        from . import bg3_unpacked_resolve as _unpacked

        assets = _unpacked.find_dialog_assets(dialog)
        lsx = assets.get("timeline_lsx")
        if lsx is not None and lsx.is_file():
            return lsx
    except ImportError:
        pass
    return None


def _parse_timeline_bindings(path: Path) -> Dict:
    key = str(path.resolve()).lower()
    cached = _TIMELINE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        _TIMELINE_CACHE[key] = {}
        return {}
    actors: Dict[str, Dict[str, int]] = {}
    anims = []
    switches = []
    for node in root.iter("node"):
        if node.get("id") != "Object" or node.get("key") != "MapKey":
            continue
        mk = _attr_value(node, "MapKey")
        if len(mk) != 36:
            continue
        value = _find_child(node, "Value")
        if value is None:
            continue
        if not _attr_value(value, "SceneActorType"):
            continue
        try:
            actors[mk.lower()] = {
                "type": int(_attr_value(value, "SceneActorType", "0") or 0),
                "index": int(_attr_value(value, "SceneActorIndex", "0") or 0),
            }
        except ValueError:
            continue
    for node in root.iter("node"):
        if node.get("id") != "EffectComponent":
            continue
        etype = _attr_value(node, "Type")
        if etype == "TLAnimation":
            src = _attr_value(node, "AnimationSourceId").lower()
            if not src:
                continue
            actor = ""
            actor_node = _find_child(node, "Actor")
            if actor_node is not None:
                actor = _attr_value(actor_node, "UUID").lower()
            try:
                phase = int(_attr_value(node, "PhaseIndex", "-1") or -1)
                start = float(_attr_value(node, "StartTime", "0") or 0)
                end = float(_attr_value(node, "EndTime", "0") or 0)
                play_start = float(_attr_value(node, "AnimationPlayStartOffset", "0") or 0)
                play_rate = float(_attr_value(node, "AnimationPlayRate", "1") or 1)
            except ValueError:
                continue
            continuous = str(_attr_value(node, "Continuous", "") or "").lower() == "true"
            anims.append(
                {
                    "source": src,
                    "actor": actor,
                    "phase": phase,
                    "start": start,
                    "end": end,
                    "play_start": play_start,
                    "play_rate": play_rate,
                    "continuous": continuous,
                    "pose": _staging_pose(node),
                }
            )
        elif etype == "TLSwitchStageEvent":
            try:
                phase = int(_attr_value(node, "PhaseIndex", "-1") or -1)
            except ValueError:
                phase = -1
            keyed = False
            for key_node in node.iter("node"):
                if key_node.get("id") != "Key":
                    continue
                stage = _attr_value(key_node, "SwitchStageEventID").lower()
                if not stage:
                    continue
                try:
                    when = float(_attr_value(key_node, "Time", "0") or 0)
                except ValueError:
                    when = 0.0
                switches.append({"phase": phase, "stage": stage, "time": when})
                keyed = True
            if not keyed:
                stage = _attr_value(node, "SwitchStageEventID").lower()
                if stage:
                    try:
                        when = float(_attr_value(node, "StartTime", "0") or 0)
                    except ValueError:
                        when = 0.0
                    switches.append({"phase": phase, "stage": stage, "time": when})
    parsed = {"actors": actors, "anims": anims, "switches": switches}
    _TIMELINE_CACHE[key] = parsed
    return parsed


def _staging_pose(node: ET.Element):
    """Offset-type 2/5 TargetTransform is where that actor stands for this clip."""
    if _attr_value(node, "OffsetType") not in {"2", "5"}:
        return None
    target = _find_child(node, "TargetTransform")
    if target is None:
        return None
    pos = _vec(target, "Position", 3)
    quat = _vec(target, "RotationQuat", 4)
    if pos is None:
        return None
    if quat is None:
        quat = (0.0, 0.0, 0.0, 1.0)
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )


def _folder_path_for_gr2(gr2_path: str) -> str:
    loc = _gr2_pak_locator().get(Path(gr2_path).stem.lower())
    if not loc:
        return ""
    return _scene_folder_key(loc[1])


def _stem_for_animation_source(
    source_guid: str,
    folder_path: str,
    *,
    companion_for_player: bool,
    primary_gr2_path: str = "",
) -> str:
    """Map timeline AnimationSourceId → GR2 stem in this scene folder (beat-aware)."""
    guid = (source_guid or "").lower()
    if not guid:
        return ""
    primary = Path(primary_gr2_path).stem.lower() if primary_gr2_path else ""
    want_beats = set(_performance_tokens(primary)) if primary else set()
    want_kinds = frozenset(_clip_kind_tokens(primary)) if primary else frozenset()
    is_plyr_primary = "plyr" in primary if primary else False

    def _role_ok(stem: str) -> bool:
        is_plyr = "plyr" in stem
        if companion_for_player:
            return not is_plyr
        return is_plyr if is_plyr_primary else not is_plyr

    def _beat_score(stem: str) -> int:
        if not want_beats:
            return 0
        beats = set(_performance_tokens(stem))
        shared = want_beats & beats
        if not shared:
            return 0
        score = sum(len(t) for t in shared)
        stem_l = stem.lower()
        for beat in want_beats:
            if beat in stem_l:
                score += 6
        if want_kinds:
            kinds = frozenset(_clip_kind_tokens(stem))
            if kinds and kinds != want_kinds:
                score -= 4
        if companion_for_player and primary:
            if stem_l == primary or stem_l in primary or primary in stem_l:
                return 0
            for sep in ("holdhands", "sex", "toastkiss", "menu"):
                if sep in primary and sep in stem_l:
                    tail_a = primary.split(sep, 1)[-1]
                    tail_b = stem_l.split(sep, 1)[-1]
                    if tail_a and tail_a == tail_b:
                        return 0
        return score

    locator = _gr2_pak_locator()
    anim = _animation_source_index(None)
    best = ""
    best_score = 0
    for stem, entry in locator.items():
        if folder_path and _scene_folder_key(entry[1]) != folder_path:
            continue
        if not _role_ok(stem):
            continue
        beat_sc = _beat_score(stem)
        if primary and beat_sc <= 0:
            continue
        score = beat_sc
        if anim.get(stem, "").lower() == guid:
            score += 8
        if score > best_score:
            best_score = score
            best = stem
    if best:
        return best
    for stem, mapped in anim.items():
        if mapped.lower() != guid:
            continue
        loc = locator.get(stem)
        if folder_path and (not loc or _scene_folder_key(loc[1]) != folder_path):
            continue
        if primary and _beat_score(stem) <= 0:
            continue
        if not _role_ok(stem):
            continue
        return stem
    return ""


def _partner_anim_for_hit(
    hit: Dict,
    doc: Dict,
    source_id: str,
    *,
    prefer_companion_index: int = -1,
) -> Optional[Dict]:
    near: List[Tuple[float, Dict]] = []
    best_dt = None
    actors = doc.get("actors") or {}
    for other in doc.get("anims") or []:
        if other["phase"] != hit["phase"] or other["source"] == source_id:
            continue
        dt = abs(other["start"] - hit["start"])
        if best_dt is None or dt < best_dt:
            best_dt = dt
        near.append((dt, other))
    if not near or best_dt is None:
        return None
    tied = [other for dt, other in near if dt <= best_dt + 0.05]
    others = [
        other
        for other in tied
        if other.get("actor") and other.get("actor") != hit.get("actor")
    ]
    pool = others or tied
    posed = [other for other in pool if other.get("pose")]
    if prefer_companion_index >= 0:
        indexed = []
        for other in pool:
            meta = actors.get(other.get("actor") or "", {})
            if (
                int(meta.get("type") or 0) == 8
                and int(meta.get("index") or -1) == prefer_companion_index
            ):
                indexed.append(other)
        posed_indexed = [other for other in indexed if other.get("pose")]
        if posed_indexed:
            return posed_indexed[0]
        if indexed and not posed:
            return indexed[0]
    if posed:
        return posed[0]
    return pool[0]


def _hit_stage_norm(
    hit: Dict,
    doc: Dict,
    scene_root: Optional[ET.Element],
) -> str:
    if scene_root is None:
        return ""
    stage_guid = _stage_for_clip(
        doc.get("switches") or [],
        int(hit.get("phase", -1)),
        float(hit["start"]),
    )
    if not stage_guid:
        return ""
    return _norm_label(_stage_display_name(scene_root, stage_guid))


def _score_timeline_anim_hit(
    gr2_path: str,
    hit: Dict,
    doc: Dict,
    by_guid: Dict[str, str],
    prefer_phase: int = -1,
    scene_root: Optional[ET.Element] = None,
) -> float:
    """Pick the dialog phase row that matches this exported GR2 beat, not a reuse tail."""
    stem = Path(gr2_path).stem
    score = (float(hit["end"]) - float(hit["start"])) * 4.0
    if hit.get("continuous"):
        score -= 800.0
    if float(hit.get("play_start") or 0.0) > 0.5:
        score += 40.0
    hints = _filename_beat_hints(stem)
    file_hints = {t.lower() for t in significant_tokens(stem)}
    snorm = _hit_stage_norm(hit, doc, scene_root)
    for hint in hints | file_hints:
        if not hint or len(hint) < 3:
            continue
        if _beat_in_stage(hint, snorm) or hint in snorm:
            score += 2800.0
        elif snorm and hint == "sex" and ("hold" in snorm or "hand" in snorm):
            score -= 3500.0
        elif snorm and hint == "holdhands" and "sex" in snorm:
            score -= 3500.0
    partner = _partner_anim_for_hit(hit, doc, hit["source"])
    if partner is not None:
        pstem = by_guid.get(partner["source"], "")
        folder = _folder_path_for_gr2(gr2_path)
        inferred_p = _stem_for_animation_source(
            partner["source"],
            folder,
            companion_for_player=True,
            primary_gr2_path=gr2_path,
        )
        if inferred_p:
            pstem = inferred_p
        pnorm = _norm_label(pstem)
        for beat in hints:
            if beat and beat in pnorm:
                score += 500.0
        if hints and pstem:
            overlap = _overlap_score(stem, pstem)
            if overlap >= 8:
                score += float(overlap) * 3.0
        if hints:
            pbeats = _beat_tokens(pstem) | {
                b for b in _BEATS if b in pnorm
            }
            for beat in hints:
                if beat in pbeats:
                    score += 1200.0
                elif beat in _BEATS and pbeats:
                    score -= 1800.0
    if prefer_phase >= 0 and int(hit.get("phase", -1)) == prefer_phase:
        score += 10_000.0
    if "nonsex" in hints:
        phase_n = int(hit.get("phase", -1))
        if phase_n >= 109:
            score -= 25_000.0
        sn = snorm or _hit_stage_norm(hit, doc, scene_root)
        if sn and _stage_is_nonsex_int(sn):
            score += 30_000.0
        elif sn and "sex" in sn:
            score -= 25_000.0
        if sn and any(w in sn for w in ("restaurant", "mocap", "establish")):
            score -= 8000.0
    elif "sex" in hints:
        phase_n = int(hit.get("phase", -1))
        if phase_n == 109:
            score += 25_000.0
        elif phase_n > 110:
            score -= 20_000.0
        try:
            when = float(hit.get("start") or 0.0)
        except (TypeError, ValueError):
            when = 0.0
        if phase_n == 109:
            score -= when * 20.0
    return score


def _prefer_nonsex_placement_hit(
    gr2_path: str, hits: Sequence[Dict], doc: Dict, scene_root: Optional[ET.Element]
) -> Optional[Dict]:
    """First player row on Non Sex - INT (phase 103 @ ~902s in Karlach date)."""
    hints = _filename_beat_hints(Path(gr2_path).stem)
    if "nonsex" not in hints:
        return None
    actors = doc.get("actors") or {}
    on_stage: List[Dict] = []
    for row in hits:
        sn = _hit_stage_norm(row, doc, scene_root)
        if not _stage_is_nonsex_int(sn):
            continue
        meta = actors.get(row.get("actor") or "", {})
        if int(meta.get("type") or 0) != 2:
            continue
        on_stage.append(row)
    if not on_stage:
        return None
    return min(on_stage, key=lambda row: float(row.get("start") or 0.0))


def _prefer_sex_placement_hit(
    gr2_path: str, hits: Sequence[Dict]
) -> Optional[Dict]:
    """First staged sex row in phase 109 (Glasses: start of beat, not a later reuse)."""
    hints = _filename_beat_hints(Path(gr2_path).stem)
    if "sex" not in hints or "nonsex" in hints:
        return None
    staged = [
        h
        for h in hits
        if int(h.get("phase", -1)) == 109 and h.get("pose") is not None
    ]
    if not staged:
        return None
    return min(staged, key=lambda row: float(row.get("start") or 0.0))


def _pick_timeline_anim_hit(
    gr2_path: str,
    hits: Sequence[Dict],
    doc: Dict,
    by_guid: Dict[str, str],
    prefer_phase: int = -1,
    scene_root: Optional[ET.Element] = None,
) -> Dict:
    if len(hits) == 1:
        return hits[0]
    return max(
        hits,
        key=lambda row: (
            _score_timeline_anim_hit(
                gr2_path, row, doc, by_guid, prefer_phase, scene_root
            ),
            float(row["end"]) - float(row["start"]),
            -int(row.get("phase") or 0),
        ),
    )


def _stage_for_clip(switches: Sequence[Dict], phase: int, t: float) -> str:
    """Stage active when the clip starts: this phase's keys, else inherited global switches."""
    local = [sw for sw in switches if int(sw.get("phase", -1)) == phase]
    stage = _stage_at_time(local, t)
    if stage:
        return stage
    return _stage_at_time(switches, t)


def _stage_at_time(switches: Sequence[Dict], t: float) -> str:
    """Last stage switch at or before the clip start. Empty if the timeline has none."""
    best_time = None
    best = ""
    for sw in switches:
        stage = str(sw.get("stage") or "")
        if not stage:
            continue
        try:
            when = float(sw.get("time") if sw.get("time") is not None else -1.0)
        except (TypeError, ValueError):
            continue
        if when > t + 0.02:
            continue
        if best_time is None or when >= best_time:
            best_time = when
            best = stage
    return best


def _stage_display_name(scene_root: ET.Element, guid: str) -> str:
    want = guid.lower()
    for stage in parse_stages(scene_root):
        if stage.guid.lower() == want:
            return stage.name
    return ""


def resolve_clip_binding(
    gr2_path: str,
    scene_path: Path,
    scene_root: Optional[ET.Element] = None,
    prefer_phase: int = -1,
) -> Optional[ClipBinding]:
    """Match this GR2 to the dialog timeline phase (game extract, not filename guesses)."""
    stem = Path(gr2_path).stem.lower()
    source = _timeline_source_guid_cached(gr2_path, scene_path)
    if not source:
        return None
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        return None
    doc = _parse_timeline_bindings(timeline)
    hits = [a for a in doc.get("anims") or [] if a["source"] == source]
    if not hits:
        return None
    if prefer_phase >= 0:
        phased = [h for h in hits if int(h.get("phase", -1)) == prefer_phase]
        if len(phased) == 1:
            hit = phased[0]
            stage = _stage_for_clip(
                doc.get("switches") or [],
                int(hit.get("phase", -1)),
                float(hit["start"]),
            )
            actor_meta = (doc.get("actors") or {}).get(hit["actor"], {})
            by_guid = {
                guid: name for name, guid in _animation_source_index(scene_path).items()
            }
            partner_stem = ""
            partner_meta: Dict[str, int] = {}
            partner_pose = None
            closest = _partner_anim_for_hit(
                hit,
                doc,
                source,
                prefer_companion_index=_partner_index_pref(gr2_path),
            )
            if closest is not None:
                folder = _folder_path_for_gr2(gr2_path)
                companion = int(actor_meta.get("type") or 0) in {2, 8}
                partner_stem = _stem_for_animation_source(
                    closest["source"],
                    folder,
                    companion_for_player=companion,
                    primary_gr2_path=gr2_path,
                ) or by_guid.get(closest["source"], "")
                partner_meta = (doc.get("actors") or {}).get(closest["actor"], {})
                partner_pose = closest.get("pose")
            stage_name = ""
            if scene_root is not None and stage:
                stage_name = _stage_display_name(scene_root, stage)
            if stage and not stage_name:
                stage_name = _template_stage_name(scene_path, stage) or stage[:8]
            return ClipBinding(
                source_id=source,
                stage_guid=stage or _ZERO_GUID,
                stage_name=stage_name,
                actor_type=int(actor_meta.get("type") or 0),
                actor_index=int(actor_meta.get("index") or 0),
                partner_stem=partner_stem,
                partner_actor_type=int(partner_meta.get("type") or 0),
                partner_actor_index=int(partner_meta.get("index") or -1),
                pose=hit.get("pose"),
                partner_pose=partner_pose,
                play_start=float(hit.get("play_start") or 0.0),
                partner_play_start=float(
                    (closest or {}).get("play_start") or 0.0
                ),
                phase_index=int(hit.get("phase", -1)),
                clip_start=float(hit["start"]),
                clip_end=float(hit["end"]),
            )
        if phased:
            hits = phased
    stem_path = Path(gr2_path).stem
    hints = _filename_beat_hints(stem_path)
    file_tokens = {t.lower() for t in significant_tokens(stem_path)}
    stem_norm = _norm_label(stem_path)
    if "nonsex" in hints and len(hits) > 1:
        filtered: List[Dict] = []
        for row in hits:
            sn = _hit_stage_norm(row, doc, scene_root)
            if int(row.get("phase", -1)) >= 109:
                continue
            if sn and "sex" in sn and "nonsex" not in sn:
                continue
            filtered.append(row)
        if filtered:
            hits = filtered
    if (
        ("sex" in hints and "nonsex" not in hints)
        or ("sex" in file_tokens and "nonsex" not in stem_norm)
    ) and len(hits) > 1:
        sex_hits = [
            h
            for h in hits
            if "sex" in _hit_stage_norm(h, doc, scene_root)
        ]
        if sex_hits:
            hits = sex_hits
        else:
            filtered = [
                h
                for h in hits
                if "hold" not in _hit_stage_norm(h, doc, scene_root)
                and "hand" not in _hit_stage_norm(h, doc, scene_root)
            ]
            if filtered:
                hits = filtered
            else:
                by_partner: List[Dict] = []
                folder = _folder_path_for_gr2(gr2_path)
                by_guid_pre = {
                    guid: name
                    for name, guid in _animation_source_index(scene_path).items()
                }
                for h in hits[:24]:
                    partner = _partner_anim_for_hit(h, doc, source)
                    if partner is None:
                        continue
                    pstem = by_guid_pre.get(partner["source"], "")
                    if not pstem:
                        inferred = _stem_for_animation_source(
                            partner["source"],
                            folder,
                            companion_for_player=True,
                            primary_gr2_path=gr2_path,
                        )
                        if inferred:
                            pstem = inferred
                    if "sex" in _norm_label(pstem):
                        by_partner.append(h)
                if by_partner:
                    hits = by_partner
    by_guid = {guid: name for name, guid in _animation_source_index(scene_path).items()}
    hit = _prefer_nonsex_placement_hit(gr2_path, hits, doc, scene_root)
    if hit is None:
        hit = _prefer_sex_placement_hit(gr2_path, hits)
    if hit is None:
        hit = _pick_timeline_anim_hit(
            gr2_path, hits, doc, by_guid, prefer_phase, scene_root
        )
    # The phase often changes stage again while the clip is already playing.
    # The bodies are on the stage that is active when the clip starts.
    stage = _stage_for_clip(
        doc.get("switches") or [],
        int(hit.get("phase", -1)),
        float(hit["start"]),
    )
    actor_meta = (doc.get("actors") or {}).get(hit["actor"], {})
    partner_stem = ""
    partner_meta: Dict[str, int] = {}
    partner_pose = None
    closest = _partner_anim_for_hit(
        hit,
        doc,
        source,
        prefer_companion_index=_partner_index_pref(gr2_path),
    )
    if closest is not None:
        folder = _folder_path_for_gr2(gr2_path)
        companion = int(actor_meta.get("type") or 0) in {2, 8}
        partner_stem = _stem_for_animation_source(
            closest["source"],
            folder,
            companion_for_player=companion,
            primary_gr2_path=gr2_path,
        ) or by_guid.get(closest["source"], "")
        partner_meta = (doc.get("actors") or {}).get(closest["actor"], {})
        partner_pose = closest.get("pose")
    stage_name = ""
    if scene_root is not None and stage:
        stage_name = _stage_display_name(scene_root, stage)
    if stage and not stage_name:
        # Nested template stage (AOM/OOM) is not always listed on the dialog scene.
        stage_name = _template_stage_name(scene_path, stage) or stage[:8]
    return ClipBinding(
        source_id=source,
        stage_guid=stage or _ZERO_GUID,
        stage_name=stage_name or "Default",
        actor_type=int(actor_meta.get("type") or 0),
        actor_index=int(actor_meta.get("index") or 0),
        partner_stem=partner_stem,
        partner_actor_type=int(partner_meta.get("type") or 0),
        partner_actor_index=int(partner_meta["index"]) if "index" in partner_meta else -1,
        pose=hit.get("pose"),
        partner_pose=partner_pose,
        play_start=float(hit.get("play_start") or 0.0),
        partner_play_start=float(closest.get("play_start") or 0.0) if closest is not None else 0.0,
        phase_index=int(hit.get("phase", -1)),
        clip_start=float(hit.get("start") or 0.0),
        clip_end=float(hit.get("end") or 0.0),
    )


def _template_stage_index() -> Dict[str, str]:
    """GUID → stage label from nested scene templates (built once per session)."""
    global _TEMPLATE_STAGE_INDEX
    if _TEMPLATE_STAGE_INDEX is not None:
        return _TEMPLATE_STAGE_INDEX
    index: Dict[str, str] = {}
    seen_files: set = set()
    for folder in _nested_template_dirs():
        if not folder.is_dir():
            continue
        for path in folder.glob("*.lsx"):
            fkey = str(path).lower()
            if fkey in seen_files:
                continue
            seen_files.add(fkey)
            try:
                tree = ET.parse(path)
            except (ET.ParseError, OSError):
                continue
            for stage in parse_stages(tree.getroot()):
                g = stage.guid.lower()
                if g and g not in index:
                    index[g] = stage.name
    _TEMPLATE_STAGE_INDEX = index
    return index


def _template_stage_name(scene_path: Path, guid: str) -> str:
    return _template_stage_index().get((guid or "").lower(), "")


def _filename_stage_override(
    gr2_path: str,
    scene_path: Path,
    stages: Sequence[StageChoice],
    stage: StageChoice,
) -> StageChoice:
    """When timeline binding lands on the wrong beat (Sex clip → Hold Hands stage)."""
    hints = _filename_beat_hints(Path(gr2_path).stem)
    if "nonsex" in hints:
        return stage
    if "sex" not in hints:
        return stage
    if "sex" in _norm_label(stage.name):
        return stage
    alt = choose_stage(gr2_path, scene_path.stem, stages)
    if alt is None or "sex" not in _norm_label(alt.name):
        return stage
    return StageChoice(
        name=alt.name,
        guid=alt.guid,
        how=f"{alt.how} (filename beat overrides {stage.name!r})",
    )


def _timeline_hit_for_stage(
    gr2_path: str,
    scene_path: Path,
    scene_root: ET.Element,
    stage_guid: str,
) -> Optional[Dict]:
    source = _timeline_source_guid_cached(gr2_path, scene_path)
    if not source:
        return None
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        return None
    doc = _parse_timeline_bindings(timeline)
    want = (stage_guid or "").lower()
    for anim in doc.get("anims") or []:
        if anim.get("source") != source:
            continue
        sg = _stage_for_clip(
            doc.get("switches") or [],
            int(anim.get("phase", -1)),
            float(anim["start"]),
        )
        if sg.lower() == want:
            return anim
    return None


def _empty_for_binding(empties: Sequence[PlannedEmpty], binding: ClipBinding) -> str:
    hit = _empty_by_actor(empties, binding.actor_type, binding.actor_index)
    return hit.name if hit is not None else ""


def _partner_empty_name(empties: Sequence[PlannedEmpty], binding: ClipBinding) -> str:
    hit = _empty_by_actor(
        empties, binding.partner_actor_type, binding.partner_actor_index
    )
    return hit.name if hit is not None else ""


def _empty_by_actor(empties: Sequence[PlannedEmpty], actor_type: int, actor_index: int) -> Optional[PlannedEmpty]:
    if actor_type not in {2, 8} or actor_index < 0:
        return None
    typed = [e for e in empties if e.actor_type == actor_type]
    if actor_index >= len(typed):
        return None
    return typed[actor_index]


def _apply_clip_poses(plan: ScenePlan, binding: ClipBinding) -> None:
    """Use the clip's first-frame position. Keep the stage yaw.

    The clip quaternion is a few degrees off the stage (Sex Start player is
    -94° instead of -90°). That shows up as a crooked empty. Alt+R then
    R Z -90 is the stage yaw, so the position is the only thing the clip replaces.
    """
    for pose, actor_type, actor_index in (
        (binding.pose, binding.actor_type, binding.actor_index),
        (binding.partner_pose, binding.partner_actor_type, binding.partner_actor_index),
    ):
        if not pose:
            continue
        empty = _empty_by_actor(plan.empties, actor_type, actor_index)
        if empty is None:
            continue
        empty.position = pose[0]


def warm_scene_lookup(gr2_path: str) -> Optional[Tuple[Path, int]]:
    """Resolve dialog scene + timeline once (LSF convert). Does not build empties."""
    return find_scene_file(gr2_path)


def plan_scene_empties(
    gr2_path: str,
    *,
    use_cache: bool = True,
    timer=None,
) -> Optional[ScenePlan]:
    from timing import maybe_timer

    key = _gr2_resolve_key(gr2_path)
    if use_cache and key in _PLAN_CACHE:
        if timer is not None:
            timer.live_print("plan_cache_hit 0.00s")
        return _PLAN_CACHE[key]
    with maybe_timer(timer, "plan_find_scene"):
        found = find_scene_file(gr2_path, timer=timer)
    if found is None:
        return None
    scene_path, score = found
    with maybe_timer(timer, "plan_parse_scene"):
        root = _scene_root_cached(scene_path)
        if root is None:
            return None
        stages = parse_stages(root)
    with maybe_timer(timer, "plan_resolve_binding"):
        binding = _binding_for_scene(gr2_path, scene_path)
        if binding is None:
            binding = resolve_clip_binding(gr2_path, scene_path, root)
    stage = _resolve_placement_stage(
        gr2_path, scene_path, root, stages, binding
    )
    if stage is None:
        return None
    if _stem_is_nonsex(Path(gr2_path).stem):
        ns = _nonsex_stage_choice(stages, gr2_path)
        if ns is not None and _stage_has_actors(scene_path, ns):
            stage = StageChoice(
                name=ns.name,
                guid=ns.guid,
                how="Non Sex scene stage (Glasses)",
            )
    elif binding is not None and binding.stage_guid:
        bound = StageChoice(
            name=binding.stage_name or binding.stage_guid[:8],
            guid=binding.stage_guid.lower(),
            how="dialog timeline at clip start",
        )
        if _stage_has_actors(scene_path, bound):
            stage = bound
    elif binding is None:
        override = _filename_stage_override(gr2_path, scene_path, stages, stage)
        if _stage_has_actors(scene_path, override):
            stage = override
    with maybe_timer(timer, "plan_build_empties"):
        plan = _build_scene_plan(
            gr2_path, scene_path, score, stage, root, timer=timer
        )
    if plan is None:
        print(
            f"[GR2 LAB] scene plan: no TLActor transforms for stage "
            f"{stage.name!r} ({scene_path.name})",
            flush=True,
        )
        return None
    plan.clip_binding = binding
    if binding is not None:
        hit = _timeline_hit_for_stage(gr2_path, scene_path, root, stage.guid)
        if hit is not None:
            plan.phase_index = int(hit.get("phase", -1))
            plan.clip_start_sec = float(hit.get("start") or 0.0)
        else:
            plan.phase_index = int(binding.phase_index)
            plan.clip_start_sec = float(binding.clip_start)
        seated = _empty_for_binding(plan.empties, binding)
        if seated:
            plan.rig_empty_name = seated
        if not _stem_is_nonsex(Path(gr2_path).stem):
            _apply_clip_poses(plan, binding)
    elif plan.empties and not plan.rig_empty_name:
        plan.rig_empty_name = rig_empty_name(gr2_path, plan.empties)
    if binding is None:
        bed = _sex_bed_seat(gr2_path, plan.empties)
        if bed:
            plan.rig_empty_name = bed
    if use_cache:
        _PLAN_CACHE[key] = plan
    return plan


def plan_scene_empties_at_stage(
    gr2_path: str,
    primary: ScenePlan,
    *,
    timer=None,
) -> Optional[ScenePlan]:
    """Same timeline stage as the dropped clip (for partner GR2 matching)."""
    from timing import maybe_timer

    scene_path = primary.scene_path
    score = primary.score
    with maybe_timer(timer, "partner_plan_parse"):
        root = _scene_root_cached(scene_path)
        if root is None:
            return None
    stages = parse_stages(root)
    stage = primary.stage
    if not _stage_has_actors(scene_path, stage):
        binding_hint = _binding_for_scene(gr2_path, scene_path)
        alt = _resolve_placement_stage(
            gr2_path, scene_path, root, stages, binding_hint
        )
        if alt is not None:
            stage = alt
    with maybe_timer(timer, "partner_plan_build"):
        plan = _build_scene_plan(
            gr2_path, scene_path, score, stage, root, timer=timer
        )
    if plan is None:
        return None
    prefer = int(primary.phase_index) if primary.phase_index >= 0 else -1
    with maybe_timer(timer, "partner_plan_binding"):
        binding = _binding_for_scene(
            gr2_path, scene_path, prefer_phase=prefer
        )
    if binding is not None:
        plan.phase_index = int(binding.phase_index)
        plan.clip_start_sec = float(binding.clip_start)
        seated = _empty_for_binding(plan.empties, binding)
        if seated:
            plan.rig_empty_name = seated
        if not _stem_is_nonsex(Path(gr2_path).stem):
            _apply_clip_poses(plan, binding)
    elif plan.empties and not plan.rig_empty_name:
        plan.rig_empty_name = rig_empty_name(gr2_path, plan.empties)
    if binding is None:
        bed = _sex_bed_seat(gr2_path, plan.empties)
        if bed:
            plan.rig_empty_name = bed
    return plan


def larian_quat_to_timeline_euler_xyz(
    quat: Tuple[float, float, float, float],
) -> "Euler":
    """Euler XYZ radians — same as Glasses timeline inspector (Larian quat x,y,z,w)."""
    qx, qy, qz, qw = quat
    return Quaternion((float(qw), float(qx), float(qy), float(qz))).to_euler(
        "XYZ"
    )


def larian_quat_to_slot_euler_xyz(
    quat: Tuple[float, float, float, float],
) -> "Euler":
    """Slot euler. Delta Rx+90 still stands the rig up. A lean pivots on the hip.

    Inspector euler plus Rx+90 faces the right way, but the lean around the
    body's side axis is mirrored. The hip swings and the pivot lands under
    the floor. Glasses keeps the hip still and slides the feet, so flip that
    lean (world-Y part of up). A pure yaw is unchanged.
    """
    qx, qy, qz, qw = quat
    stand = Matrix.Rotation(math.pi / 2.0, 3, "X")
    world = stand @ Quaternion(
        (float(qw), float(qx), float(qy), float(qz))
    ).to_euler("XYZ").to_matrix()
    up = world.col[1].copy()
    up_fixed = Vector((up.x, -up.y, up.z))
    if up_fixed.length < 1e-8:
        return (stand.inverted() @ world).to_euler("XYZ")
    up_fixed.normalize()
    face = world.col[0].copy()
    face = face - up_fixed * face.dot(up_fixed)
    if face.length < 1e-8:
        return (stand.inverted() @ world).to_euler("XYZ")
    face.normalize()
    flat = Vector((world.col[0].x, world.col[0].y, 0.0))
    if flat.length > 1e-8 and face.dot(flat) < 0.0:
        face = -face
    right = face.cross(up_fixed)
    fixed = Matrix((face, up_fixed, right)).transposed()
    return (stand.inverted() @ fixed).to_euler("XYZ")


def larian_timeline_slot_location(
    pos: Tuple[float, float, float],
    y_up_display: bool,
    swap_location: bool,
) -> "Vector":
    tx, ty, tz = pos
    if y_up_display:
        # Glasses is Y-up. Blender Z is up, so height is Glasses Y and the
        # Glasses Z channel is depth (Blender Y). Copying Z into Blender Z
        # drops the body when that depth value changes (Karlach 1.043 → 0.95).
        loc = (tx, tz, ty)
        del swap_location
    else:
        loc = (tx, ty, tz)
    return Vector(loc)


GR2LAB_RX90_SUFFIX = "__gr2lab_rx"
_GLASSES_DELTA_RX90 = (math.radians(90.0), 0.0, 0.0)


def _uses_glasses_slot_euler(empty) -> bool:
    """Slot empty shows Glasses timeline X/Y/Z; +90° X is Delta Transform."""
    if empty is None:
        return False
    if empty.get("gr2lab_glasses_euler_direct"):
        return True
    for col in empty.users_collection:
        if col.get("gr2lab_cinematic_sync"):
            return True
    return False


def _apply_glasses_delta_rx90(empty) -> None:
    """Constant Delta Transform rotation X = 90°. Not keyed."""
    if empty is None:
        return
    empty.rotation_mode = "XYZ"
    empty.delta_rotation_euler = _GLASSES_DELTA_RX90


def _remove_glasses_rx_child(empty) -> None:
    """Drop the old animated ``*__gr2lab_rx`` empty. Rigs move onto the slot."""
    if bpy is None or empty is None:
        return
    child = bpy.data.objects.get(empty.name + GR2LAB_RX90_SUFFIX)
    if child is None:
        return
    for obj in list(child.children):
        obj.parent = empty
        obj.matrix_parent_inverse = Matrix.Identity(4)
    _clear_empty_slot_action(child)
    bpy.data.objects.remove(child, do_unlink=True)


def _write_glasses_slot_channels(empty, pos, quat, y_up_display: bool, swap_location: bool) -> None:
    """Location + rotation_euler = Glasses inspector. Delta X stays +90°."""
    if empty is None:
        return
    empty.rotation_mode = "XYZ"
    empty.delta_rotation_euler = (0.0, 0.0, 0.0)
    empty.location = larian_timeline_slot_location(pos, y_up_display, swap_location)
    empty.rotation_euler = larian_quat_to_slot_euler_xyz(quat)
    empty.scale = (1.0, 1.0, 1.0)
    _apply_glasses_delta_rx90(empty)
    empty["gr2lab_glasses_euler_direct"] = True
    _remove_glasses_rx_child(empty)


def _matrix_from_timeline_slot_pose(
    pos,
    quat,
    y_up_display: bool,
    swap_location: bool,
) -> "Matrix":
    """Glasses inspector Euler XYZ (no delta). Delta +90° X is separate."""
    if not y_up_display or swap_location:
        return _matrix_from_larian_pose(pos, quat, y_up_display, swap_location)
    return Matrix.LocRotScale(
        larian_timeline_slot_location(pos, y_up_display, swap_location),
        larian_quat_to_timeline_euler_xyz(quat),
        (1.0, 1.0, 1.0),
    )


def _stamp_glasses_inspector_euler(empty, quat) -> None:
    """Read-only Glasses timeline X/Y/Z (Y-up) for comparison — not Blender rotation_euler."""
    if empty is None:
        return
    eul = larian_quat_to_timeline_euler_xyz(quat)
    empty["gr2lab_glasses_rot_deg"] = (
        f"{math.degrees(eul.x):.4f},"
        f"{math.degrees(eul.y):.4f},"
        f"{math.degrees(eul.z):.4f}"
    )


def larian_to_blender(pos, quat, y_up_display: bool, swap_location: bool = True):
    """Larian → Blender for legacy / NonSex mirror paths.

    Rotation uses the Y-up display quat remap (qw, qx, qz, qy) for seated armatures.
    Timeline slot empties use ``_matrix_from_timeline_slot_pose`` instead so X/Y/Z
    match the Glasses inspector.
    """
    tx, ty, tz = pos
    qx, qy, qz, qw = quat
    if y_up_display:
        loc = (tx, tz, ty) if swap_location else (tx, ty, tz)
        rot = (qw, qx, qz, qy)  # Blender w, x, y, z
    else:
        loc = (tx, ty, tz)
        rot = (qw, qx, qy, qz)
    return loc, rot


# Cinematic slots: rotation_euler = Glasses X/Y/Z; Delta Transform rotation X = +90°.
_EMPTY_ROTATION_MODE = "XYZ"


def _matrix_from_larian_pose(
    pos,
    quat,
    y_up_display: bool,
    swap_location: bool,
) -> "Matrix":
    loc, rot = larian_to_blender(pos, quat, y_up_display, swap_location)
    return Matrix.LocRotScale(
        Vector(loc),
        Quaternion(rot),
        (1.0, 1.0, 1.0),
    )


def _apply_empty_world_matrix(obj, world_matrix: "Matrix") -> None:
    if bpy is None or obj is None:
        return
    obj.rotation_mode = _EMPTY_ROTATION_MODE
    obj.matrix_world = world_matrix


def _object_action_fcurves(obj):
    if bpy is None or obj is None:
        return None
    ad = getattr(obj, "animation_data", None)
    if ad is None or ad.action is None:
        return None
    try:
        from .bridge_blender import _ensure_action_fcurves

        return _ensure_action_fcurves(obj, ad.action)
    except Exception:
        return getattr(ad.action, "fcurves", None)


def _strip_empty_rotation_channels(obj, *, keep_euler: bool) -> None:
    """XYZ empties must not keep stale rotation_quaternion keys (ignored by the UI)."""
    fcurves = _object_action_fcurves(obj)
    if fcurves is None:
        return
    drop = []
    for fc in fcurves:
        path = fc.data_path
        if path == "rotation_quaternion":
            drop.append(fc)
        elif keep_euler and path == "rotation_euler":
            continue
        elif not keep_euler and path == "rotation_euler":
            drop.append(fc)
    for fc in drop:
        try:
            fcurves.remove(fc)
        except Exception:
            pass


def _keyframe_empty_transform(obj, frame: int) -> None:
    if obj is None:
        return
    if not obj.animation_data:
        obj.animation_data_create()
    obj.rotation_mode = _EMPTY_ROTATION_MODE
    _strip_empty_rotation_channels(obj, keep_euler=True)
    obj.keyframe_insert(data_path="location", frame=int(frame))
    obj.keyframe_insert(data_path="rotation_euler", frame=int(frame))


def _parent_keep_world(child, parent) -> None:
    # Child still stores its world pose in location/rotation. Parent inverse
    # cancels the parent so that pose stays put in the world.
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


def _subtree_objects(root) -> List:
    stack = [root]
    out: List = []
    while stack:
        obj = stack.pop()
        out.append(obj)
        stack.extend(list(obj.children))
    return out


def _center_root_at_origin(root) -> bool:
    """After placement is done: slide the whole hierarchy so the root empty is at (0,0,0)."""
    if bpy is None or root is None:
        return False
    shift = -root.matrix_world.translation
    if shift.length_squared < 1e-10:
        return True
    for obj in _subtree_objects(root):
        obj.matrix_world = Matrix.Translation(shift) @ obj.matrix_world
    return True


def _overlap_score(left: str, right: str) -> int:
    shared = {t.lower() for t in significant_tokens(left)} & {
        t.lower() for t in significant_tokens(right)
    }
    if len(shared) < 2:
        return 0
    return sum(max(len(token), 3) for token in shared)


def nearby_gr2s(gr2_path: Path) -> List[Path]:
    """GR2 files beside the drop, and one folder up or down."""
    folders = [gr2_path.parent]
    parent = gr2_path.parent.parent
    if parent != gr2_path.parent:
        folders.append(parent)
    found: List[Path] = []
    seen = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        paths = list(folder.glob("*.gr2")) + list(folder.glob("*.GR2"))
        try:
            subs = [
                p
                for p in folder.iterdir()
                if p.is_dir() and p.name.lower() not in {"body_base", "__pycache__"}
            ]
        except OSError:
            subs = []
        for sub in subs:
            paths.extend(sub.glob("*.gr2"))
            paths.extend(sub.glob("*.GR2"))
        for path in paths:
            try:
                key = str(path.resolve()).lower()
            except OSError:
                continue
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            found.append(path)
    return found


def _partner_role_names(name: str) -> set:
    """Role suffix on paired clips (HoldHands_Dammon vs HoldHands_Karlach)."""
    stem = Path(name).stem
    m = re.search(
        r"(?:HoldHands|Hug|Kiss|Grasps|PlaceHand|Fiddles|Romance)_([A-Za-z]+)(?:_|$)",
        stem,
    )
    if m:
        role = m.group(1).lower()
        if role not in _STOP and role not in _BODY and role not in _ROLE_KIND:
            return {role}
    low = stem.lower()
    if "_plyr" in low or low.endswith("_plyr"):
        return set()
    out = set()
    for comp in _COMPANION_BODY:
        if re.search(rf"(?:^|_){comp}(?:_|$)", low):
            out.add(comp)
    return out


def _score_partner_candidate(
    primary: Path,
    cand: Path,
    *,
    plan: Optional[ScenePlan],
) -> int:
    overlap = _overlap_score(primary.name, cand.name)
    if overlap < 10:
        return 0
    primary_kinds = _variant_suffix_tokens(primary.name)
    cand_kinds = _variant_suffix_tokens(cand.name)
    if primary_kinds and not (primary_kinds & cand_kinds):
        return 0
    primary_roles = _role_kind_tokens(primary.name)
    cand_roles = _role_kind_tokens(cand.name)
    if primary_roles & cand_roles:
        return 0
    primary_key = rig_body_key(primary.name)
    cand_key = rig_body_key(cand.name)
    if primary_key and cand_key == primary_key:
        return 0
    proles = _partner_role_names(primary.name)
    croles = _partner_role_names(cand.name)
    if proles and croles and proles & croles:
        return 0
    primary_beats = _beat_tokens(primary.name)
    cand_beats = _beat_tokens(cand.name)
    if primary_beats and cand_beats and not (primary_beats & cand_beats):
        return 0
    score = overlap
    if primary_kinds & cand_kinds:
        score += 40
    if "plyr" in primary_roles and cand_roles & {"el", "sk", "comp"}:
        score += 25
    if proles and croles and not (proles & croles):
        score += 55
    same_beat = bool(primary_beats and primary_beats & cand_beats)
    different_role = bool(proles and croles and not (proles & croles))
    if plan is not None and plan.rig_empty_name:
        cand_lower = cand.stem.lower()
        primary_player = (
            plan.rig_empty_name.lower().startswith("initiator")
            or "plyr" in primary.stem.lower()
        )
        cand_player = "plyr" in cand_lower
        cand_companion = any(
            tok in cand_lower for tok in ("_el", "_sk", "_comp", "karlach", "dammon")
        )
        if primary_player and cand_companion and not cand_player:
            score += 50
        elif not primary_player and cand_player:
            score += 50
        elif cand_player == primary_player and not (same_beat and different_role):
            return 0
    elif same_beat and different_role:
        score += 45
    return score


_GR2_BY_STEM: Dict[str, Optional[Path]] = {}
_GR2_STEM_MISS: set = set()


def _save_rank(path: Path) -> Tuple[int, float]:
    number = -1
    for part in path.parts:
        if part.isdigit():
            number = int(part)
            break
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (number, mtime)


def _extracted_save_roots() -> List[Path]:
    """Decoder Save trees are not a data source."""
    return []


_PAK_LOCATOR: Optional[Dict[str, List[str]]] = None
_LOCATOR_FOLDERS: Optional[Dict[str, List[str]]] = None
# Role and clip-kind words that sit on every take of a beat. Sibling matching
# ignores them so "spit blood" cannot attach to "bite" just because both say CAMP.
_SIBLING_SKIP = _SHORT_OK | _CLIP_KIND | {"ast", "sm", "start", "end"}


def _gr2_pak_locator() -> Dict[str, List[str]]:
    """Stem → [pak name, path inside the pak] for scene GR2s that are still packed."""
    global _PAK_LOCATOR, _LOCATOR_FOLDERS
    if _PAK_LOCATOR is not None:
        return _PAK_LOCATOR
    index: Dict[str, List[str]] = {}
    folders: Dict[str, List[str]] = {}
    for cache in _decoder_timeline_caches():
        path = cache / "gr2_pak_locator.json"
        if not path.is_file():
            continue
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            for stem, loc in raw.items():
                if stem and isinstance(loc, list) and len(loc) >= 2:
                    key = str(stem).lower()
                    index[key] = [str(loc[0]), str(loc[1])]
                    folders.setdefault(_scene_folder_key(loc[1]), []).append(key)
            break
    _PAK_LOCATOR = index
    _LOCATOR_FOLDERS = folders
    return index


def _pak_file(pak_name: str) -> Optional[Path]:
    for data in _bg3_data_roots():
        candidate = data / pak_name
        if candidate.is_file():
            return candidate
    return None


def _extract_gr2_from_pak(stem: str) -> Optional[Path]:
    """Pull one scene GR2 out of the game pak when the timeline names it and it is not on disk."""
    loc = _gr2_pak_locator().get(stem.lower())
    if not loc:
        return None
    pak_name, internal = loc[0], loc[1]
    filename = Path(internal).name
    if not filename:
        return None
    cache = _primary_timeline_cache()
    exe = _divine_exe()
    if cache is None or exe is None:
        return None
    extract_dir = cache / "_extract_gr2"
    if extract_dir.is_dir():
        found = list(extract_dir.rglob(filename))
        if found:
            return found[0]
    pak = _pak_file(pak_name)
    if pak is None:
        return None
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if not _divine_run(
        exe,
        ["-a", "extract-package", "-s", str(pak), "-d", str(extract_dir), "-x", f"*{filename}"],
    ):
        return None
    found = list(extract_dir.rglob(filename))
    return found[0] if found else None


def _gr2_by_stem(stem: str, *, near: Optional[Path] = None) -> Optional[Path]:
    """Partner GR2 by stem — search beside the drop first, never scan whole saves."""
    key = (stem or "").lower()
    if not key:
        return None
    if key in _GR2_BY_STEM:
        return _GR2_BY_STEM[key]
    if key in _GR2_STEM_MISS:
        return None
    hits: List[Path] = []
    if near is not None:
        folder = near.parent
        for ext in (".GR2", ".gr2"):
            direct = folder / f"{stem}{ext}"
            if direct.is_file():
                _GR2_BY_STEM[key] = direct
                return direct
        for cand in nearby_gr2s(near):
            if cand.stem.lower() == key:
                hits.append(cand)
        for extra in (folder.parent,):
            try:
                for ext in (".GR2", ".gr2"):
                    direct = extra / f"{stem}{ext}"
                    if direct.is_file():
                        hits.append(direct)
            except OSError:
                pass
    for root in _extracted_save_roots():
        try:
            for path in root.glob(f"**/{stem}.GR2"):
                if path.is_file():
                    hits.append(path)
            for path in root.glob(f"**/{stem}.gr2"):
                if path.is_file():
                    hits.append(path)
        except OSError:
            continue
        if hits:
            break
    if not hits:
        _GR2_STEM_MISS.add(key)
        _GR2_BY_STEM[key] = None
        return None
    best = max(hits, key=_save_rank)
    _GR2_BY_STEM[key] = best
    return best


def find_partner_gr2(
    gr2_path: str,
    *,
    primary_plan: Optional[ScenePlan] = None,
    timer=None,
) -> Optional[Path]:
    """The other character's clip next to this file: same scene, other body."""
    from timing import maybe_timer

    primary = Path(gr2_path)
    with maybe_timer(timer, "partner_find_plan"):
        plan = primary_plan or plan_scene_empties(str(primary))
    best: Optional[Path] = None
    best_score = 0
    try:
        primary_key = str(primary.resolve()).lower()
    except OSError:
        primary_key = str(primary).lower()
    binding = None
    if plan is not None:
        binding = plan.clip_binding or _binding_for_scene(
            str(primary), plan.scene_path
        )
    if binding is None:
        found_scene = find_scene_file(str(primary))
        if found_scene is not None:
            binding = _binding_for_scene(str(primary), found_scene[0])
    if binding is not None and binding.partner_stem:
        want = binding.partner_stem.lower()
        with maybe_timer(timer, "partner_find_nearby"):
            for cand in nearby_gr2s(primary):
                if cand.stem.lower() == want:
                    return cand
        with maybe_timer(timer, "partner_find_stem"):
            banked = _gr2_by_stem(want, near=primary)
            if banked is not None:
                return banked
        try:
            from . import bg3_unpacked_resolve as _unpacked

            with maybe_timer(timer, "partner_find_models"):
                loose = _unpacked.find_gr2_in_models(want)
                if loose is not None:
                    return loose
        except ImportError:
            pass
        with maybe_timer(timer, "partner_find_pak"):
            extracted = _extract_gr2_from_pak(want)
            if extracted is not None:
                return extracted
    wants_player = (
        binding is not None
        and not binding.partner_stem
        and binding.partner_actor_type == 2
    )
    with maybe_timer(timer, "partner_score_nearby"):
        nearby = nearby_gr2s(primary)
    for cand in nearby:
        try:
            cand_key = str(cand.resolve()).lower()
        except OSError:
            continue
        if cand_key == primary_key:
            continue
        if cand.stem.lower().endswith("_base"):
            continue
        if wants_player and "plyr" not in cand.stem.lower():
            continue
        score = _score_partner_candidate(primary, cand, plan=plan)
        if score > best_score:
            best_score = score
            best = cand
    return best


def _tail_rigs_of(arm) -> list:
    """Sibling tail armatures built beside this body. Their meshes are their children."""
    if bpy is None or arm is None:
        return []
    name = getattr(arm, "name", "") or ""
    if not name:
        return []
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "ARMATURE" and obj.get("gr2lab_tail_of") == name
    ]


def _seat_arm(arm, empty, y_up_display: bool) -> None:
    # The tail is not a child of the body. It was matched to the body at the
    # origin, so seating only the body leaves the tail and its meshes behind.
    if y_up_display and _uses_glasses_slot_euler(empty):
        _apply_glasses_delta_rx90(empty)
        _remove_glasses_rx_child(empty)
        for obj in [arm, *_tail_rigs_of(arm)]:
            obj.parent = empty
            obj.location = (0.0, 0.0, 0.0)
            # Local +X is the body's left. Glasses shows the other leg leading
            # and the lean the other way. Root sits on the origin, so this
            # mirror leaves the slot where it is.
            obj.scale = (-1.0, 1.0, 1.0)
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.matrix_parent_inverse = Matrix.Identity(4)
        empty["gr2lab_imported_rig"] = arm.name
        return
    for obj in [arm, *_tail_rigs_of(arm)]:
        obj.parent = empty
        obj.location = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)
        obj.rotation_mode = "XYZ"
        if y_up_display:
            obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        else:
            obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.matrix_parent_inverse = Matrix.Identity(4)
    empty["gr2lab_imported_rig"] = arm.name


def _mirror_y_through_point(obj, center) -> None:
    """Ctrl+M, Y, with the cursor at the scene center.

    The initiator's position flips across that center, and so does its facing.
    """
    world = obj.matrix_world.copy()
    flip = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    pre = Matrix.Translation(Vector((0.0, -float(center.y), 0.0)))
    post = Matrix.Translation(Vector((0.0, float(center.y), 0.0)))
    obj.matrix_world = post @ flip @ pre @ world


def _yaw_world_z(obj) -> None:
    """Turn a rig 180° around world up, about its own origin.

    NonSex partner clips lie down along the stage. Without this turn Karlach's
    head points the opposite way from the timeline camera, so her legs land
    where her head should be.
    """
    if obj is None:
        return
    world = obj.matrix_world.copy()
    origin = world.translation.copy()
    turn = Matrix.Rotation(math.pi, 4, "Z")
    obj.matrix_world = Matrix.Translation(origin) @ turn @ Matrix.Translation(-origin) @ world


def _mirror_y_about_origin(obj) -> None:
    """Ctrl+M, Y about this object. Location stays. Facing is mirrored."""
    world = obj.matrix_world.copy()
    flip = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    linear = flip @ world.to_3x3().to_4x4()
    obj.matrix_world = Matrix.Translation(world.translation) @ linear


def _mirror_player_empties(empties, anchor) -> None:
    center = anchor.matrix_world.translation
    for empty in empties:
        if empty.get("gr2lab_is_player"):
            _mirror_y_through_point(empty, center)


def _stage_for_gr2_scene(
    gr2_path: str,
    scene_path: Path,
    score: int,
    *,
    timer=None,
) -> Optional[Tuple[StageChoice, Optional[ClipBinding], int, float]]:
    """Stage + binding for a known scene file (no actor empties required)."""
    from timing import maybe_timer

    with maybe_timer(timer, "plan_parse_scene"):
        root = _scene_root_cached(scene_path)
        if root is None:
            return None
        stages = parse_stages(root)
    with maybe_timer(timer, "plan_resolve_binding"):
        binding = _binding_for_scene(gr2_path, scene_path)
        if binding is None:
            binding = resolve_clip_binding(gr2_path, scene_path, root)
    stage = _resolve_placement_stage(
        gr2_path, scene_path, root, stages, binding
    )
    if stage is None:
        return None
    if _stem_is_nonsex(Path(gr2_path).stem):
        ns = _nonsex_stage_choice(stages, gr2_path)
        if ns is not None and _stage_has_actors(scene_path, ns):
            stage = StageChoice(
                name=ns.name,
                guid=ns.guid,
                how="Non Sex scene stage (Glasses)",
            )
    elif binding is not None and binding.stage_guid:
        bound = StageChoice(
            name=binding.stage_name or binding.stage_guid[:8],
            guid=binding.stage_guid.lower(),
            how="dialog timeline at clip start",
        )
        if _stage_has_actors(scene_path, bound):
            stage = bound
    elif binding is None:
        override = _filename_stage_override(gr2_path, scene_path, stages, stage)
        if _stage_has_actors(scene_path, override):
            stage = override
    phase = -1
    clip_start = 0.0
    if binding is not None:
        hit = _timeline_hit_for_stage(gr2_path, scene_path, root, stage.guid)
        if hit is not None:
            phase = int(hit.get("phase", -1))
            clip_start = float(hit.get("start") or 0.0)
        else:
            phase = int(binding.phase_index)
            clip_start = float(binding.clip_start)
    return stage, binding, phase, clip_start


def minimal_primary_plan_for_seat(
    gr2_path: str,
    *,
    timer=None,
) -> Optional[ScenePlan]:
    """Scene + stage for partner seating when actor empties were not built."""
    key = _gr2_resolve_key(gr2_path)
    found = _FIND_SCENE_CACHE.get(key)
    if found is None:
        found = find_scene_file(gr2_path, timer=timer)
    if found is None:
        return None
    scene_path, score = found
    resolved = _stage_for_gr2_scene(gr2_path, scene_path, score, timer=timer)
    if resolved is None:
        return None
    stage, binding, phase, clip_start = resolved
    return ScenePlan(
        scene_path=scene_path,
        stage=stage,
        empties=[],
        score=int(score),
        phase_index=int(phase),
        clip_start_sec=float(clip_start),
        clip_binding=binding,
    )


def seat_snapshot_for_gr2(gr2_path: str, *, timer=None) -> Optional[Dict[str, object]]:
    """Scene/stage handle when full ScenePlan build fails (e.g. no stage actors)."""
    plan = minimal_primary_plan_for_seat(gr2_path, timer=timer)
    if plan is None:
        return None
    return seat_primary_snapshot(plan)


def seat_primary_snapshot(plan: ScenePlan) -> Dict[str, object]:
    """JSON-safe scene/stage handle for modal import state (survives module reload)."""
    return {
        "scene_path": str(plan.scene_path),
        "stage_name": plan.stage.name,
        "stage_guid": plan.stage.guid,
        "stage_how": plan.stage.how,
        "score": int(plan.score),
        "phase_index": int(plan.phase_index),
        "clip_start_sec": float(plan.clip_start_sec),
    }


def _scene_plan_from_seat_snapshot(data: Optional[Dict]) -> Optional[ScenePlan]:
    if not data:
        return None
    try:
        scene_path = Path(str(data["scene_path"]))
        stage = StageChoice(
            name=str(data["stage_name"]),
            guid=str(data["stage_guid"]),
            how=str(data.get("stage_how") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None
    try:
        scene_path = scene_path.resolve()
    except OSError:
        pass
    return ScenePlan(
        scene_path=scene_path,
        stage=stage,
        empties=[],
        score=int(data.get("score") or 0),
        phase_index=int(data.get("phase_index", -1)),
        clip_start_sec=float(data.get("clip_start_sec") or 0.0),
    )


def glasses_preview_from_plan(
    plan: ScenePlan, gr2_path: str = ""
) -> Dict[str, object]:
    """MCP/Glasses: dialog, stage, scrub time, and timeline .lsx from the import plan."""
    dialog = _canonical_dialog_name(dialog_stem(plan.scene_path))
    if gr2_path:
        named = (_timeline_name_for_gr2(gr2_path) or "").strip()
        if named:
            dialog = _canonical_dialog_name(named)
    timeline = _timeline_beside_scene(plan.scene_path)
    when = float(plan.clip_start_sec or 0.0)
    return {
        "dialogName": dialog,
        "sceneStage": plan.stage.name,
        "timeSeconds": round(when, 3),
        "phaseIndex": int(plan.phase_index),
        "timelineLsx": str(timeline) if timeline is not None else "",
        "sceneLsx": str(plan.scene_path),
        "sourceGr2": str(gr2_path or ""),
    }


def _glasses_mcp_sidecar_path() -> Path:
    import os

    return Path(os.environ.get("LOCALAPPDATA", "")) / "GlassesMcp" / "gr2lab_preview.json"


def write_glasses_mcp_sidecar(preview: Dict[str, object]) -> Optional[Path]:
    """Last GR2 import preview — Glasses MCP tools can read this path."""
    import json

    path = _glasses_mcp_sidecar_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    except OSError:
        return None
    return path


def read_glasses_mcp_sidecar() -> Optional[Dict[str, object]]:
    import json

    path = _glasses_mcp_sidecar_path()
    if not path.is_file():
        plan = _primary_plan_from_scene_collections()
        if plan is None:
            return None
        return glasses_preview_from_plan(plan)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def glasses_link_is_active(context) -> bool:
    if bpy is None or context is None:
        return False
    return bool(context.scene.get("gr2lab_glasses_linked_play"))


def glasses_frame_sync_is_active(context) -> bool:
    if bpy is None or context is None:
        return False
    return bool(context.scene.get("gr2lab_glasses_frame_sync"))


def _glasses_phase_band(context) -> Optional[float]:
    col = collection_for_cinematic_sync(context)
    if col is None or col.get("gr2lab_panel_band_sec") is None:
        return None
    try:
        return float(col["gr2lab_panel_band_sec"])
    except (TypeError, ValueError):
        return None


def _glasses_mcp_tool(name: str, arguments: Dict[str, object], timeout: float = 4.0) -> str:
    """One JSON-RPC tools/call to the Glasses MCP host on 127.0.0.1:9877."""
    import json
    import socket

    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    ).encode("utf-8")
    header = (
        "POST /mcp HTTP/1.1\r\n"
        "Host: 127.0.0.1:9877\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    sock = socket.create_connection(("127.0.0.1", 9877), timeout)
    try:
        sock.settimeout(timeout)
        sock.sendall(header + payload)
        chunks = []
        while True:
            piece = sock.recv(65536)
            if not piece:
                break
            chunks.append(piece)
    finally:
        sock.close()
    raw = b"".join(chunks).decode("utf-8", "replace")
    body = raw.split("\r\n\r\n", 1)[-1]
    try:
        data = json.loads(body)
    except ValueError:
        return body[-500:]
    result = data.get("result") if isinstance(data, dict) else None
    if isinstance(result, dict):
        content = result.get("content") or []
        if content and isinstance(content, list) and isinstance(content[0], dict):
            return str(content[0].get("text") or "")
    if isinstance(data, dict) and data.get("error"):
        return "FAIL=" + str(data["error"])
    return body[-500:]


def glasses_linked_play_toggle(
    context,
    *,
    stop: bool,
    blender_already_stopped: bool = False,
) -> Tuple[bool, str]:
    """Start or stop Blender playback and the Glasses timeline from this frame."""
    if bpy is None or context is None:
        return False, "Blender is not available"
    scene = context.scene
    screen = getattr(context, "screen", None)
    if stop:
        scene["gr2lab_glasses_linked_play"] = False
        if (
            not blender_already_stopped
            and screen is not None
            and screen.is_animation_playing
        ):
            bpy.ops.screen.animation_play()
        try:
            note = _glasses_mcp_tool(
                "glasses_timeline_scrub",
                {"action": "pause"},
            )
        except OSError as exc:
            note = f"FAIL=Glasses is not running ({exc})"
        return True, "Stopped. " + _glasses_transport_brief(note)

    band = _glasses_phase_band(context)
    if band is None:
        return False, "Sync the scene once so the phase start is known"
    fps = float(scene.render.fps or 30) / max(float(scene.render.fps_base or 1.0), 1e-6)
    frame = int(scene.frame_current)
    try:
        note = _glasses_mcp_tool(
            "glasses_timeline_scrub",
            {
                "action": "play",
                "glassesFrame": str(frame),
                "fps": str(fps),
                "phaseBandSec": str(band),
            },
        )
    except OSError as exc:
        return False, f"Glasses is not running ({exc})"
    if not _glasses_play_started(note):
        return False, _glasses_transport_brief(note)
    scene["gr2lab_glasses_linked_play"] = True
    globals()["_glasses_sync_transport_paused"] = False
    globals()["_glasses_sync_drag_pause_armed"] = False
    if screen is None or not screen.is_animation_playing:
        bpy.ops.screen.animation_play()
    return True, f"Playing from frame {frame}. " + _glasses_transport_brief(note)


_glasses_sync_last_frame = None
_glasses_sync_pending_frame = None
_glasses_sync_next_ok = 0.0
_glasses_sync_prev_playing = False
_glasses_sync_pause_sent = False
_glasses_sync_log = []
_glasses_sync_min_interval = 0.045
_glasses_sync_drag_interval = 0.014


def glasses_frame_sync_toggle(context) -> Tuple[bool, str]:
    """Follow the Blender playhead in the open Glasses timeline, or stop doing that."""
    if bpy is None or context is None:
        return False, "Blender is not available"
    scene = context.scene
    if scene.get("gr2lab_glasses_frame_sync"):
        scene["gr2lab_glasses_frame_sync"] = False
        scene["gr2lab_glasses_linked_play"] = False
        _glasses_sync_timer_stop()
        return True, "Glasses playhead sync off"
    if _glasses_phase_band(context) is None:
        return False, "Import the scene so the phase start is known"
    scene["gr2lab_glasses_frame_sync"] = True
    scene["gr2lab_glasses_linked_play"] = False
    globals()["_glasses_sync_pause_sent"] = False
    _glasses_sync_timer_ensure()
    glasses_frame_sync_handlers_install()
    ok, msg = glasses_seek_playhead(context)
    if not ok:
        scene["gr2lab_glasses_frame_sync"] = False
        _glasses_sync_timer_stop()
        return False, msg
    return True, "Sync on. " + msg


def glasses_seek_playhead(context) -> Tuple[bool, str]:
    """Move the Glasses timeline to this Blender frame. Does not start playback."""
    global _glasses_sync_last_frame
    if bpy is None or context is None:
        return False, "Blender is not available"
    scene = context.scene
    band = _glasses_phase_band(context)
    if band is None:
        return False, "Import the scene so the phase start is known"
    fps = float(scene.render.fps or 30) / max(float(scene.render.fps_base or 1.0), 1e-6)
    frame = int(scene.frame_current)
    # The open phase plays 0 .. duration. band+frame sits past that and snaps
    # back to the start, same as the play button.
    seconds = frame / fps if fps else 0.0
    try:
        note = _glasses_mcp_tool(
            "glasses_timeline_scrub",
            {
                "timeSeconds": str(seconds),
                "refreshPreview": "true",
            },
            timeout=1.5,
        )
    except OSError as exc:
        return False, f"Glasses is not running ({exc})"
    brief = note or ""
    if "FAIL=" in brief or "NOT_READY" in brief or "BUSY=" in brief:
        return False, _glasses_transport_brief(brief)
    _glasses_sync_last_frame = frame
    return True, f"Glasses at frame {frame}. " + _glasses_transport_brief(note)


def _glasses_sync_timer_ensure() -> None:
    if bpy is None:
        return
    glasses_frame_sync_handlers_install()
    if bpy.app.timers.is_registered(_glasses_sync_timer):
        return
    bpy.app.timers.register(_glasses_sync_timer, first_interval=0.05, persistent=True)


def _glasses_sync_timer_stop() -> None:
    global _glasses_sync_last_frame, _glasses_sync_pending_frame, _glasses_sync_prev_playing
    _glasses_sync_last_frame = None
    _glasses_sync_pending_frame = None
    _glasses_sync_prev_playing = False
    globals()["_glasses_sync_pause_sent"] = False
    if bpy is None:
        return
    if bpy.app.timers.is_registered(_glasses_sync_timer):
        bpy.app.timers.unregister(_glasses_sync_timer)


def _glasses_blender_playing() -> bool:
    if bpy is None or getattr(bpy.context, "window_manager", None) is None:
        return False
    try:
        windows = bpy.context.window_manager.windows
    except Exception:
        return False
    for window in windows:
        screen = getattr(window, "screen", None)
        if screen is not None and screen.is_animation_playing:
            return True
    return False


def _glasses_sync_on_frame(scene, *args) -> None:
    """Scrub only — never start Glasses transport from here."""
    del args
    if bpy is None or scene is None or not scene.get("gr2lab_glasses_frame_sync"):
        return
    frame = int(scene.frame_current)
    _glasses_sync_note_frame_activity(frame)
    if _glasses_blender_playing() and not _glasses_sync_drag_active():
        return
    _glasses_sync_queue_scrub(scene)


def _glasses_sync_note_frame_activity(frame: int) -> None:
    """Timeline LMB drag jumps frames quickly. Blender may flag is_animation_playing then."""
    import time

    now = time.monotonic()
    prev_t = float(globals().get("_glasses_sync_activity_t", 0.0) or 0.0)
    prev_f = int(globals().get("_glasses_sync_activity_f", frame))
    rapid = (now - prev_t) < 0.14 and frame != prev_f
    globals()["_glasses_sync_activity_t"] = now
    globals()["_glasses_sync_activity_f"] = int(frame)
    if rapid:
        globals()["_glasses_sync_drag_until"] = now + 0.45
    drag = now < float(globals().get("_glasses_sync_drag_until", 0.0) or 0.0)
    globals()["_glasses_sync_drag_active_flag"] = drag
    if drag:
        globals()["_glasses_sync_drag_pause_armed"] = True


def _glasses_sync_drag_active() -> bool:
    return bool(globals().get("_glasses_sync_drag_active_flag"))


def _glasses_sync_queue_scrub(scene) -> None:
    global _glasses_sync_pending_frame
    frame = int(scene.frame_current)
    if frame == _glasses_sync_last_frame and _glasses_sync_pending_frame is None:
        return
    _glasses_sync_pending_frame = frame
    _glasses_sync_flush_scrub(scene, force=False)


def _glasses_pulse_sock():
    """One non-blocking UDP socket. Drags must not open a TCP connection per frame."""
    sock = globals().get("_glasses_pulse_socket")
    if sock is not None:
        return sock
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setblocking(False)
    except OSError:
        pass
    globals()["_glasses_pulse_socket"] = sock
    return sock


def _glasses_sync_pulse(seconds: float) -> None:
    """Latest-wins playhead. Glasses applies it on the UI thread; Blender does not wait."""
    import struct

    packet = b"G" + struct.pack("<d", float(seconds))
    try:
        _glasses_pulse_sock().sendto(packet, ("127.0.0.1", 9878))
    except OSError:
        globals()["_glasses_pulse_socket"] = None


def _glasses_pulse_boot_async() -> None:
    """Touch the addon once so the UDP listener exists. Does not block the timeline."""
    if globals().get("_glasses_pulse_booted") or globals().get("_glasses_pulse_boot_started"):
        return
    globals()["_glasses_pulse_boot_started"] = True
    import threading

    def run() -> None:
        try:
            note = _glasses_mcp_tool("glasses_timeline_pulse", {}, timeout=1.2)
        except OSError:
            globals()["_glasses_pulse_boot_started"] = False
            return
        if "pulsePort=9878" in (note or ""):
            globals()["_glasses_pulse_booted"] = True
            return
        globals()["_glasses_pulse_boot_started"] = False

    threading.Thread(target=run, daemon=True).start()


def _glasses_sync_flush_scrub(scene, *, force: bool) -> None:
    """Move the Glasses playhead to this Blender frame without stalling the drag."""
    global _glasses_sync_last_frame, _glasses_sync_pending_frame
    frame = _glasses_sync_pending_frame
    if frame is None:
        frame = int(scene.frame_current)
    if frame == _glasses_sync_last_frame and not force:
        _glasses_sync_pending_frame = None
        return
    _glasses_pulse_boot_async()
    if not _glasses_sync_drag_active():
        _glasses_sync_pause_for_scrub()
    fps = float(scene.render.fps or 30) / max(float(scene.render.fps_base or 1.0), 1e-6)
    seconds = frame / fps if fps else 0.0
    _glasses_sync_pulse(seconds)
    _glasses_sync_last_frame = int(frame)
    _glasses_sync_pending_frame = None
    _glasses_sync_log.append(int(frame))
    if len(_glasses_sync_log) > 8:
        del _glasses_sync_log[:-8]
    if not _glasses_sync_drag_active():
        globals()["_glasses_sync_drag_pause_armed"] = False
    if not force:
        return
    try:
        _glasses_mcp_tool(
            "glasses_timeline_scrub",
            {
                "timeSeconds": str(seconds),
                "refreshPreview": "true",
            },
            timeout=0.55,
        )
    except OSError:
        return


def _glasses_sync_pause_transport(*, force: bool) -> None:
    """Keep Glasses in pause while the Blender timeline is being dragged."""
    if not force and globals().get("_glasses_sync_transport_paused"):
        return
    try:
        _glasses_mcp_tool(
            "glasses_timeline_scrub",
            {"action": "pause"},
            timeout=0.45,
        )
    except OSError:
        return
    globals()["_glasses_sync_transport_paused"] = True
    if bpy is not None and bpy.context is not None and bpy.context.scene is not None:
        bpy.context.scene["gr2lab_glasses_linked_play"] = False


def _glasses_sync_pause_for_scrub() -> None:
    """If linked play was on, stop Glasses once before seeking."""
    if bpy is None or bpy.context is None:
        return
    scene = bpy.context.scene
    if not scene.get("gr2lab_glasses_linked_play"):
        return
    scene["gr2lab_glasses_linked_play"] = False
    if globals().get("_glasses_sync_pause_sent"):
        return
    try:
        _glasses_mcp_tool(
            "glasses_timeline_scrub",
            {"action": "pause"},
            timeout=0.6,
        )
    except OSError:
        return
    globals()["_glasses_sync_pause_sent"] = True


def _glasses_sync_on_play_edge() -> None:
    """Only stop Glasses when linked play ends. Never auto-start play while sync is on."""
    import time

    global _glasses_sync_prev_playing, _glasses_sync_next_ok
    if bpy is None or bpy.context is None:
        return
    if _glasses_sync_drag_active():
        return
    scene = bpy.context.scene
    playing = _glasses_blender_playing()
    was = _glasses_sync_prev_playing
    _glasses_sync_prev_playing = playing
    if playing:
        return
    if not was:
        return
    if not scene.get("gr2lab_glasses_linked_play"):
        return
    scene["gr2lab_glasses_linked_play"] = False
    globals()["_glasses_sync_transport_paused"] = False
    if time.monotonic() < _glasses_sync_next_ok:
        return
    try:
        _glasses_mcp_tool(
            "glasses_timeline_scrub",
            {"action": "pause"},
            timeout=0.55,
        )
    except OSError:
        pass
    _glasses_sync_next_ok = time.monotonic() + 0.12
    if bpy.context.scene is not None:
        _glasses_sync_flush_scrub(bpy.context.scene, force=True)


def _glasses_busy_wait(note: str) -> float:
    import re

    match = re.search(r"retryAfterMs=(\d+)", note or "")
    if not match:
        return 0.35
    return max(0.05, int(match.group(1)) / 1000.0)


def _glasses_sync_on_depsgraph(scene, depsgraph) -> None:
    """Timeline drags refresh the depsgraph before frame_change_post runs."""
    del depsgraph
    if bpy is None or scene is None or not scene.get("gr2lab_glasses_frame_sync"):
        return
    frame = int(scene.frame_current)
    _glasses_sync_note_frame_activity(frame)
    if _glasses_blender_playing() and not _glasses_sync_drag_active():
        return
    if frame == _glasses_sync_last_frame and frame != _glasses_sync_pending_frame:
        return
    _glasses_sync_queue_scrub(scene)


def glasses_frame_sync_handlers_install() -> None:
    if bpy is None:
        return
    frame_handlers = bpy.app.handlers.frame_change_post
    if _glasses_sync_on_frame not in frame_handlers:
        frame_handlers.append(_glasses_sync_on_frame)
    graph_handlers = bpy.app.handlers.depsgraph_update_post
    if _glasses_sync_on_depsgraph not in graph_handlers:
        graph_handlers.append(_glasses_sync_on_depsgraph)


def glasses_frame_sync_handlers_remove() -> None:
    if bpy is None:
        return
    for handlers, fn in (
        (bpy.app.handlers.frame_change_post, _glasses_sync_on_frame),
        (bpy.app.handlers.depsgraph_update_post, _glasses_sync_on_depsgraph),
    ):
        try:
            handlers.remove(fn)
        except ValueError:
            pass


def _glasses_sync_timer() -> Optional[float]:
    if bpy is None:
        return None
    scene = bpy.context.scene if bpy.context is not None else None
    if scene is None or not scene.get("gr2lab_glasses_frame_sync"):
        return None
    dragging = _glasses_sync_drag_active()
    if dragging:
        if _glasses_sync_pending_frame is not None:
            _glasses_sync_flush_scrub(scene, force=False)
        else:
            frame = int(scene.frame_current)
            if frame != (_glasses_sync_last_frame or -1):
                _glasses_sync_queue_scrub(scene)
    elif not _glasses_blender_playing():
        if _glasses_sync_pending_frame is not None:
            _glasses_sync_flush_scrub(scene, force=False)
        elif int(scene.frame_current) != (_glasses_sync_last_frame or -1):
            _glasses_sync_queue_scrub(scene)
    if not dragging:
        _glasses_sync_on_play_edge()
    return 0.03 if dragging else 0.08


def glasses_frame_sync_resume() -> None:
    """Addon reload: keep an already-enabled playhead sync running."""
    if bpy is None:
        return
    for scene in bpy.data.scenes:
        if scene.get("gr2lab_glasses_frame_sync"):
            _glasses_sync_timer_ensure()
            return


def _glasses_play_started(note: str) -> bool:
    for line in (note or "").replace("\r", "").split("\n"):
        text = line.strip()
        if text in {"transport=PlayMode=Play", "transport=Play", "playMode=Play"}:
            return True
    return False


def _glasses_transport_brief(note: str) -> str:
    text = " ".join((note or "").split())
    if len(text) > 180:
        text = text[:180] + "…"
    return text or "no reply"


def _stamp_glasses_preview_on_collection(col, preview: Dict[str, object]) -> None:
    col["gr2lab_dialog_name"] = str(preview.get("dialogName") or "")
    col["gr2lab_glasses_stage"] = str(preview.get("sceneStage") or "")
    col["gr2lab_glasses_time_sec"] = float(preview.get("timeSeconds") or 0.0)
    if preview.get("timelineLsx"):
        col["gr2lab_timeline_lsx"] = str(preview["timelineLsx"])


def _primary_plan_from_scene_collections() -> Optional[ScenePlan]:
    if bpy is None:
        return None
    for col in bpy.data.collections:
        if not col.get("gr2lab_scene_empties"):
            continue
        sp = col.get("gr2lab_scene_file")
        sg = col.get("gr2lab_stage_guid")
        if not sp or not sg:
            continue
        name = col.name
        if name.endswith(" positions"):
            name = name[: -len(" positions")]
        return ScenePlan(
            scene_path=Path(str(sp)),
            stage=StageChoice(
                name=name,
                guid=str(sg),
                how=str(col.get("gr2lab_stage_how") or ""),
            ),
            empties=[],
            score=20,
            phase_index=int(col.get("gr2lab_phase_index", -1)),
            clip_start_sec=float(col.get("gr2lab_timeline_clip_start_sec") or 0.0),
        )
    return None


def resolve_primary_plan_for_partner_seat(
    primary_plan: Optional[ScenePlan],
    primary_gr2_path: str,
    *,
    seat_snapshot: Optional[Dict] = None,
    timer=None,
) -> Optional[ScenePlan]:
    """Reuse primary import scene/stage so partner seating skips find_scene_file."""
    if primary_plan is not None:
        return primary_plan
    from timing import maybe_timer

    snap_plan = _scene_plan_from_seat_snapshot(seat_snapshot)
    if snap_plan is not None:
        return snap_plan
    if primary_gr2_path:
        minimal = minimal_primary_plan_for_seat(primary_gr2_path, timer=timer)
        if minimal is not None:
            return minimal
    key = _gr2_resolve_key(primary_gr2_path) if primary_gr2_path else ""
    if key:
        cached = _PLAN_CACHE.get(key)
        if cached is not None:
            return cached
    blend_plan = _primary_plan_from_scene_collections()
    if blend_plan is not None:
        return blend_plan
    if not primary_gr2_path:
        return None
    with maybe_timer(timer, "partner_seat_primary_lookup"):
        full = plan_scene_empties(primary_gr2_path, timer=timer)
        if full is not None:
            return full
        return minimal_primary_plan_for_seat(primary_gr2_path, timer=timer)


def _primary_plan_for_partner_seat(
    primary_plan: Optional[ScenePlan],
    primary_gr2_path: str,
) -> Optional[ScenePlan]:
    return resolve_primary_plan_for_partner_seat(
        primary_plan, primary_gr2_path
    )


def parent_arm_to_scene_empty(
    context,
    gr2_path: str,
    arm,
    y_up_display: bool = True,
    *,
    primary_plan: Optional[ScenePlan] = None,
    primary_gr2_path: str = "",
    seat_snapshot: Optional[Dict] = None,
    timer=None,
    partner_plan: Optional[ScenePlan] = None,
) -> str:
    """Put an already imported armature on the empty this clip belongs to."""
    from timing import maybe_timer

    if bpy is None or arm is None:
        return ""
    sync_note = ""
    with maybe_timer(timer, "partner_seat_plan"):
        seat_primary = resolve_primary_plan_for_partner_seat(
            primary_plan,
            primary_gr2_path,
            seat_snapshot=seat_snapshot,
            timer=timer,
        )
        if partner_plan is not None:
            plan = partner_plan
        elif seat_primary is not None:
            plan = plan_scene_empties_at_stage(
                gr2_path, seat_primary, timer=timer
            )
        else:
            print(
                "[GR2 LAB] partner seat: no primary scene plan — "
                "running full scene lookup on partner GR2 (slow)",
                flush=True,
            )
            plan = plan_scene_empties(gr2_path, timer=timer)
            if plan is None and primary_gr2_path:
                seat_primary = minimal_primary_plan_for_seat(
                    primary_gr2_path, timer=timer
                )
                if seat_primary is not None:
                    plan = plan_scene_empties_at_stage(
                        gr2_path, seat_primary, timer=timer
                    )
    if plan is None or not plan.rig_empty_name:
        return ""
    empty = bpy.data.objects.get(plan.rig_empty_name)
    if empty is None or empty.type != "EMPTY":
        return ""
    if arm.parent is not None and arm.parent != empty:
        world = arm.matrix_world.copy()
        for tail in _tail_rigs_of(arm):
            tail_world = tail.matrix_world.copy()
            tail.parent = None
            tail.matrix_world = tail_world
        arm.parent = None
        arm.matrix_world = world
    if arm.parent == empty:
        return f"Second character already on {empty.name}"
    with maybe_timer(timer, "partner_seat_apply"):
        _seat_arm(arm, empty, y_up_display)
        if y_up_display and empty.get("gr2lab_is_player"):
            if context is not None:
                context.view_layer.update()
            _mirror_y_about_origin(arm)
        if context is not None:
            context.view_layer.update()
        seat_plan = seat_primary or plan
        if seat_plan is not None and context is not None:
            col = None
            for name in (seat_plan.stage.name, f"{seat_plan.stage.name} positions"):
                candidate = bpy.data.collections.get(name)
                if candidate is not None and candidate.get("gr2lab_scene_empties"):
                    col = candidate
                    break
            if col is not None:
                sync_note = apply_cinematic_timeline_sync(
                    context,
                    col,
                    gr2_path,
                    seat_plan,
                    y_up_display=y_up_display,
                    timeline_gr2_path=primary_gr2_path or gr2_path,
                )
                if sync_note:
                    print(f"[GR2 LAB] {sync_note}", flush=True)
    if _stem_is_nonsex(Path(gr2_path).stem) and not empty.get("gr2lab_is_player"):
        if context is not None:
            context.view_layer.update()
        _yaw_world_z(arm)
        for tail in _tail_rigs_of(arm):
            _yaw_world_z(tail)
        if context is not None:
            context.view_layer.update()
    note = f"Second character on {empty.name}: {Path(gr2_path).name}"
    if sync_note:
        note = f"{note} · {sync_note}"
    return note


_QUAT_ANCHOR_Y90 = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
_QUAT_PLAYER_Y90 = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)


GLASSES_DEFAULT_FIRST_BODY_FRAME = 12


def panel_seconds_to_timeline_frame(panel_sec: float, fps: float) -> int:
    """Glasses phase ruler at 30fps: panel 0.39s → frame 12, 3.06s → frame 92."""
    return max(0, int(round(float(panel_sec) * float(fps))))


def _glasses_last_body_frame(arm, first_body_frame: int) -> int:
    """Scene frame for end of body strip (import keys are action frames 1..N)."""
    first = max(0, int(first_body_frame))
    action = _armature_action(arm)
    if action is None:
        return max(first, 250)
    try:
        a0, a1 = action.frame_range
        span = max(1, int(round(float(a1) - float(a0))) + 1)
    except Exception:
        span = 250
    return first + span - 1


def apply_glasses_timeline_range(
    scene,
    *,
    band_sec: float,
    fps: float,
    panel_end: float,
    playhead_panel: float,
) -> int:
    """Blender timeline matches Glasses phase scrub (frame 0 = phase band)."""
    if scene is None:
        return 0
    end_frame = panel_seconds_to_timeline_frame(panel_end, fps)
    scene.frame_start = 0
    scene.frame_end = max(12, end_frame)
    scene.frame_set(panel_seconds_to_timeline_frame(playhead_panel, fps))
    scene["gr2lab_timeline_band_sec"] = float(band_sec)
    scene["gr2lab_timeline_glasses_frames"] = True
    scene["gr2lab_timeline_fps"] = float(fps)
    return int(scene.frame_end)


def _keyframe_empty_hold(anchor, player_empty, frame: int) -> None:
    for obj in (anchor, player_empty):
        _keyframe_empty_transform(obj, frame)


def _action_slot_for_arm(ad) -> Optional[object]:
    if ad is None:
        return None
    slot = getattr(ad, "action_slot", None)
    if slot is not None:
        return slot
    for track in ad.nla_tracks or []:
        for strip in track.strips:
            slot = getattr(strip, "action_slot", None)
            if slot is not None:
                return slot
    return None


def _park_body_action_from_scene_frame(
    arm,
    *,
    first_scene_frame: int,
    last_scene_frame: int,
) -> bool:
    """One GR2 action: hold pose on frames 0..N-1, play from first body frame (Glasses)."""
    if bpy is None or arm is None:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    ad = arm.animation_data
    action_slot = _action_slot_for_arm(ad)
    action = _armature_action(arm)
    _remove_cinematic_nla(arm)
    if action is None:
        action = _armature_action(arm)
    if action is None:
        rid = arm.get("gr2lab_run_id")
        if rid:
            for act in bpy.data.actions:
                if str(rid) in act.name and act.name.startswith("GR2LAB"):
                    action = act
                    break
    if action is None:
        return False
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])
    try:
        ad.use_tweak_mode = False
    except Exception:
        pass
    ad.use_nla = True
    first = max(0, int(first_scene_frame))
    last = max(first, int(last_scene_frame))
    try:
        action_end = max(1.0, float(action.frame_range[1]))
    except Exception:
        action_end = float(max(1, last - first + 1))
    # Hold and play live on different tracks so they can share frame 12.
    # Blender 5 slotted NLA evaluates strip-local frame 0 as bind (T-pose);
    # play starts at `first` with action frame 1 (not bind).
    if first > 0:
        hold_track = ad.nla_tracks.new()
        hold_track.name = "GR2LAB_Hold"
        hold = hold_track.strips.new("hold", 0, action)
        hold.frame_start = 0.0
        hold.action_frame_start = 1.0
        hold.action_frame_end = 1.0
        hold.frame_end = float(max(last, first + 1))
        hold.blend_type = "REPLACE"
        hold.extrapolation = "HOLD_FORWARD"
        if action_slot is not None and hasattr(hold, "action_slot"):
            try:
                hold.action_slot = action_slot
            except Exception:
                pass
    play_track = ad.nla_tracks.new()
    play_track.name = "GR2LAB_Play"
    play = play_track.strips.new("body", first, action)
    play.frame_start = float(first)
    play.action_frame_start = 1.0
    play.action_frame_end = action_end
    play.frame_end = float(max(last, first + 2))
    play.blend_type = "REPLACE"
    if action_slot is not None and hasattr(play, "action_slot"):
        try:
            play.action_slot = action_slot
        except Exception:
            pass
    ad.action = None
    if hasattr(ad, "action_slot"):
        try:
            ad.action_slot = None
        except Exception:
            pass
    _ensure_gr2lab_nla_audible(ad)
    return True


def _gr2_action_frame_for_play_sec(play_sec: float, fps: float) -> float:
    """Imported GR2 keys use bridge ``_sec_to_frame``: frame 1 = t=0."""
    return 1.0 + max(0.0, float(play_sec)) * float(fps)


def _imported_action_max_play_sec(action, fps: float) -> float:
    """Last keyed second in the imported GR2 action (Blender action frame 1 = t=0)."""
    if action is None:
        return 0.0
    try:
        a1 = float(action.frame_range[1])
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return max(0.0, (a1 - 1.0) / max(float(fps), 1.0))


def _gr2_play_sec_for_action(play_sec: float, fps: float, action) -> Tuple[float, float]:
    """Return (effective_play_sec, timeline_play_sec) capped to imported keys."""
    timeline = max(0.0, float(play_sec))
    cap = _imported_action_max_play_sec(action, fps)
    if cap <= 0.0:
        return timeline, timeline
    effective = min(timeline, cap)
    return effective, timeline


def _segment_play_bounds(
    segments: Sequence[Dict],
    index: int,
    *,
    is_player: bool,
) -> Tuple[float, float]:
    """Play this TLAnimation from its own offset at AnimationPlayRate.

    The next row is another component. Its offset is where that clip starts,
    not a playhead the current clip has to reach. Missing rate means 1.
    """
    seg = segments[index]
    key = "player_play" if is_player else "partner_play"
    rate_key = "player_rate" if is_player else "partner_rate"
    play_start = float(seg.get(key) or 0.0)
    panel_start = float(seg["panel_start"])
    panel_end = float(seg["panel_end"])
    rate = float(seg.get(rate_key) or 1.0)
    if rate <= 1e-6:
        rate = 1.0
    play_end = play_start + max(0.0, panel_end - panel_start) * rate
    return play_start, play_end


def _apply_glasses_segment_body_nla(
    arm,
    segments: Sequence[Dict],
    *,
    fps: float,
    first_body_frame: int,
    is_player: bool,
) -> bool:
    """Hold 0..N-1, then per-timeline-row body strips (play offset + rate at each snap)."""
    if bpy is None or arm is None or not segments:
        return False
    action = _armature_action(arm)
    if action is None:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    ad = arm.animation_data
    action_slot = _action_slot_for_arm(ad)
    _remove_cinematic_nla(arm)
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])
    try:
        ad.use_tweak_mode = False
    except Exception:
        pass
    ad.use_nla = True
    first = max(0, int(first_body_frame))
    last_panel = max(float(s["panel_end"]) for s in segments)
    last_frame = max(first + 2, panel_seconds_to_timeline_frame(last_panel, fps))
    if first > 0:
        hold_track = ad.nla_tracks.new()
        hold_track.name = "GR2LAB_Hold"
        hold = hold_track.strips.new("hold", 0, action)
        hold.frame_start = 0.0
        hold.action_frame_start = 1.0
        hold.action_frame_end = 1.0
        hold.frame_end = float(last_frame)
        hold.blend_type = "REPLACE"
        hold.extrapolation = "HOLD_FORWARD"
        if action_slot is not None and hasattr(hold, "action_slot"):
            try:
                hold.action_slot = action_slot
            except Exception:
                pass
    for i, seg in enumerate(segments):
        panel_start = float(seg["panel_start"])
        panel_end = float(seg["panel_end"])
        if panel_end <= panel_start + 1e-6:
            panel_end = panel_start + 1.0 / max(fps, 1.0)
        play_raw, play_end_raw = _segment_play_bounds(segments, i, is_player=is_player)
        scene_start = panel_seconds_to_timeline_frame(panel_start, fps)
        scene_end_ex = panel_seconds_to_timeline_frame(panel_end, fps)
        if scene_end_ex <= scene_start:
            scene_end_ex = scene_start + 1
        play_start, _ = _gr2_play_sec_for_action(play_raw, fps, action)
        play_end, _ = _gr2_play_sec_for_action(play_end_raw, fps, action)
        action_start = _gr2_action_frame_for_play_sec(play_start, fps)
        action_end = _gr2_action_frame_for_play_sec(play_end, fps)
        # The first frame of an NLA strip evaluates as bind. Start one frame
        # early on its own track so body motion begins on the logical start.
        nla_start = scene_start - 1 if scene_start > 0 else scene_start
        scene_len = float(scene_end_ex - nla_start)
        action_len = float(action_end - action_start)
        if scene_len > 1e-6 and abs(action_len) > 1e-6:
            action_start_nla = action_start - action_len * (
                float(scene_start - nla_start) / scene_len
            )
            action_end_nla = action_start_nla + action_len
        else:
            action_start_nla = action_start
            action_end_nla = action_end
        track = ad.nla_tracks.new()
        track.name = f"GR2LAB_Seg{i}_{int(scene_start)}_{int(scene_end_ex)}"
        strip = track.strips.new(f"seg{i}", nla_start, action)
        _commit_nla_span(strip, nla_start, scene_end_ex, action_start_nla, action_end_nla)
        try:
            strip["gr2lab_logical_start"] = int(scene_start)
            strip["gr2lab_logical_end"] = int(scene_end_ex)
        except Exception:
            pass
        if action_slot is not None and hasattr(strip, "action_slot"):
            try:
                strip.action_slot = action_slot
            except Exception:
                pass
    ad.action = None
    if hasattr(ad, "action_slot"):
        try:
            ad.action_slot = None
        except Exception:
            pass
    _ensure_gr2lab_nla_audible(ad)
    return True


def _is_gr2lab_timeline_track(track) -> bool:
    name = track.name or ""
    return (
        name in ("GR2LAB_Hold", "GR2LAB_Play", "GR2LAB_Cinematic", "GR2LAB_Full")
        or name.startswith("GR2LAB_Seg")
    )


def _armature_has_gr2lab_timeline_nla(ad) -> bool:
    if ad is None or not ad.nla_tracks:
        return False
    return any(_is_gr2lab_timeline_track(t) for t in ad.nla_tracks)


def _ensure_gr2lab_nla_audible(ad) -> None:
    """Leave every GR2LAB timeline track/strip unmuted (no frame-based channel switching)."""
    if ad is None or not ad.nla_tracks:
        return
    for track in ad.nla_tracks:
        if not _is_gr2lab_timeline_track(track):
            continue
        track.mute = False
        if hasattr(track, "is_solo"):
            track.is_solo = False
        for strip in track.strips:
            strip.mute = False


def _nla_strip_logical_frame(strip, first_body: int) -> int:
    try:
        stored = strip.get("gr2lab_logical_start")
        if stored is not None:
            return int(stored)
    except Exception:
        pass
    try:
        return int(strip.frame_start)
    except Exception:
        return int(first_body)


def _park_rig_family_from_scene_frame(
    arm,
    *,
    first_scene_frame: int,
    last_scene_frame: int,
) -> int:
    count = 0
    if arm is None:
        return 0
    for obj in (arm, *_tail_rigs_of(arm)):
        if _park_body_action_from_scene_frame(
            obj,
            first_scene_frame=first_scene_frame,
            last_scene_frame=last_scene_frame,
        ):
            count += 1
    return count


try:
    from bpy.app.handlers import persistent as _persistent
except ImportError:

    def _persistent(fn):
        return fn


@_persistent
def gr2lab_prebody_nla_mute_handler(scene):
    """Glasses rigs: clear direct Action before first body frame; keep NLA tracks unmuted."""
    if bpy is None or scene is None:
        return
    first = int(scene.get("gr2lab_first_body_frame") or 0)
    if first <= 0 and scene.get("gr2lab_timeline_glasses_frames"):
        first = GLASSES_DEFAULT_FIRST_BODY_FRAME
    if first <= 0:
        return
    frame = int(scene.frame_current)
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE" or not obj.get("gr2lab_run_id"):
            continue
        ad = getattr(obj, "animation_data", None)
        if ad is None or not ad.nla_tracks:
            continue
        # A clip imported on its own keeps its Action. Only Glasses body
        # tracks (hold / play / segments) may drop the direct Action before
        # the first body frame. Clearing every GR2 LAB rig unlinks that
        # Action permanently — scrubbing back does not put it back.
        if not _armature_has_gr2lab_timeline_nla(ad):
            continue
        if ad.action is not None and frame < first:
            ad.action = None
        if hasattr(ad, "action_slot") and frame < first:
            try:
                ad.action_slot = None
            except Exception:
                pass
        _ensure_gr2lab_nla_audible(ad)


def register_prebody_nla_handler() -> None:
    if bpy is None:
        return
    handler = gr2lab_prebody_nla_mute_handler
    if handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(handler)


def unregister_prebody_nla_handler() -> None:
    if bpy is None:
        return
    handler = gr2lab_prebody_nla_mute_handler
    while handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(handler)


def _has_scene_segment_nla(arm) -> bool:
    """True when this rig already plays the Glasses timeline (Seg strips)."""
    ad = getattr(arm, "animation_data", None)
    if ad is None:
        return False
    for track in ad.nla_tracks or []:
        if (track.name or "").startswith("GR2LAB_Seg"):
            return True
    return False


def _release_active_action_over_scene_nla(arm) -> bool:
    """Seg strips are the game timing. An assigned Action plays the raw clip instead."""
    ad = getattr(arm, "animation_data", None)
    if ad is None or not _has_scene_segment_nla(arm):
        return False
    ad.use_nla = True
    changed = ad.action is not None
    if changed:
        ad.action = None
    if hasattr(ad, "action_slot"):
        try:
            if getattr(ad, "action_slot", None) is not None:
                changed = True
            ad.action_slot = None
        except Exception:
            pass
    return changed


def apply_glasses_body_hold_for_gr2(
    context,
    arm,
    gr2_path: str,
    *,
    plan: Optional[ScenePlan] = None,
) -> str:
    """After GR2 import: park action so frames 0..11 are static (Glasses pre-roll)."""
    if bpy is None or context is None or arm is None:
        return ""
    # A race swap already copied this character's scene strips and pointed
    # them at the new clip. Replacing that stack with one Play strip, or
    # leaving the Action assigned, plays a different moment — and the
    # pre-roll handler then clears the Action on every other scene character.
    if _has_scene_segment_nla(arm):
        _release_active_action_over_scene_nla(arm)
        return "kept scene timeline"
    scene = context.scene
    fps = float(scene.render.fps or 30)
    first = GLASSES_DEFAULT_FIRST_BODY_FRAME
    last = _glasses_last_body_frame(arm, first)
    segments: List[Dict] = []
    if plan is None:
        plan = plan_scene_empties(gr2_path)
    if plan is not None:
        binding = plan.clip_binding or _binding_for_scene(gr2_path, plan.scene_path)
        if binding is not None:
            segments, band = collect_paired_slot_segments(
                gr2_path, plan.scene_path, binding
            )
            if segments:
                first = panel_seconds_to_timeline_frame(
                    float(segments[0]["panel_start"]), fps
                )
                last = panel_seconds_to_timeline_frame(
                    float(segments[-1]["panel_end"]), fps
                )
                last = max(last, _glasses_last_body_frame(arm, first))
    if _armature_action(arm) is None:
        return ""
    scene["gr2lab_timeline_glasses_frames"] = True
    scene["gr2lab_first_body_frame"] = int(first)
    if int(getattr(scene, "frame_start", 1) or 1) > 0:
        scene.frame_start = 0
    register_prebody_nla_handler()
    n = _park_rig_family_from_scene_frame(
        arm, first_scene_frame=first, last_scene_frame=last
    )
    gr2lab_prebody_nla_mute_handler(scene)
    if n == 0:
        return ""
    return f"body hold 0–{first - 1}, play from {first} ({n} rigs)"


def _parse_scene_tl_cameras(scene_path: Path) -> Dict[str, Dict[str, object]]:
    """TLCameras blocks in *_Scene.lsx (Identifier → name + Larian transform)."""
    try:
        root = ET.parse(scene_path).getroot()
    except (ET.ParseError, OSError):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for node in root.iter("node"):
        if node.get("id") != "TLCameras":
            continue
        ident = (_attr_value(node, "Identifier") or "").lower()
        name = _attr_value(node, "Name") or "GR2LAB_Camera"
        pos = None
        quat = None
        for mv in node.iter("node"):
            if mv.get("id") != "MapValue":
                continue
            p = _vec(mv, "Position", 3)
            q = _vec(mv, "RotationQuat", 4)
            if p is not None:
                pos = (float(p[0]), float(p[1]), float(p[2]))
            if q is not None:
                quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        if ident and pos is not None and quat is not None:
            out[ident] = {"name": name, "pos": pos, "quat": quat}
    return out


def _parse_timeline_tl_shots(
    timeline_path: Path,
    *,
    phase: int,
    band_sec: float,
    fps: float,
) -> List[Dict[str, object]]:
    """TLShot rows → Glasses timeline frames + camera actor guid."""
    try:
        root = ET.parse(timeline_path).getroot()
    except (ET.ParseError, OSError):
        return []
    shots: List[Dict[str, object]] = []
    for node in root.iter("node"):
        if node.get("id") != "EffectComponent":
            continue
        if _attr_value(node, "Type") != "TLShot":
            continue
        try:
            ph = int(_attr_value(node, "PhaseIndex", "-1") or -1)
        except ValueError:
            continue
        if ph != int(phase):
            continue
        try:
            start = float(_attr_value(node, "StartTime", "0") or 0)
        except ValueError:
            continue
        panel = start - float(band_sec)
        cam_guid = ""
        for child in node.iter("node"):
            if child.get("id") != "CameraContainer":
                continue
            cam_guid = (_attr_value(child, "Object") or "").lower()
            break
        shots.append(
            {
                "frame": panel_seconds_to_timeline_frame(panel, fps),
                "panel": panel,
                "camera_guid": cam_guid,
            }
        )
    shots.sort(key=lambda row: int(row["frame"]))
    return shots


def _timeline_actor_camera_map(timeline_path: Path) -> Dict[str, str]:
    """Timeline actor UUID → TLCameras Identifier (when present on actor value)."""
    try:
        root = ET.parse(timeline_path).getroot()
    except (ET.ParseError, OSError):
        return {}
    mapping: Dict[str, str] = {}
    for node in root.iter("node"):
        if node.get("id") != "Object" or node.get("key") != "MapKey":
            continue
        mk = _attr_value(node, "MapKey").lower()
        value = _find_child(node, "Value")
        if not mk or value is None:
            continue
        cam_id = (
            _attr_value(value, "Camera")
            or _attr_value(value, "CameraIdentifier")
            or ""
        ).lower()
        if cam_id:
            mapping[mk] = cam_id
    return mapping


def _key_frame_of_reference(key_node: ET.Element) -> Optional[Tuple[str, str]]:
    for child in key_node.iter("node"):
        if child.get("id") != "frameOfReference":
            continue
        target = (_attr_value(child, "targetId") or "").lower()
        bone = (_attr_value(child, "targetBone") or "").strip()
        if target:
            return target, bone
    return None


def _camera_actor_slot_map(
    timeline_path: Path,
    phase: int,
    player_source: str,
    *,
    player_empty_name: str = "Initiator_1",
    partner_empty_name: str = "Additional_1",
) -> Dict[str, str]:
    """Dialog actor UUID → slot empty for camera frameOfReference."""
    try:
        doc = _parse_timeline_bindings(timeline_path)
    except Exception:
        return {}
    source = (player_source or "").lower()
    if not source:
        return {}
    partner_sources: set = set()
    for row in doc.get("anims") or []:
        if int(row.get("phase", -1)) != int(phase):
            continue
        if row.get("source") != source:
            continue
        partner = _partner_anim_for_hit(row, doc, source, prefer_companion_index=-1)
        if partner:
            ps = str(partner.get("source") or "").lower()
            if ps:
                partner_sources.add(ps)
    out: Dict[str, str] = {}
    for row in doc.get("anims") or []:
        if int(row.get("phase", -1)) != int(phase):
            continue
        actor = str(row.get("actor") or "").lower()
        src = str(row.get("source") or "").lower()
        if not actor:
            continue
        if src == source:
            out[actor] = player_empty_name
        elif src in partner_sources:
            out[actor] = partner_empty_name
    return out


def _parse_phase_camera_curves(
    timeline_path: Path, phase: int
) -> Tuple[Dict[str, Dict[str, list]], Dict[str, list]]:
    """Phase TLTransform + TLCameraFoV curves keyed by camera actor UUID.

    Transform channels are X, Y, Z, then a quaternion (xyzw). Times stay in
    timeline seconds so a shot can sample the curve between keys.
    """
    try:
        root = ET.parse(timeline_path).getroot()
    except (ET.ParseError, OSError):
        return {}, {}
    transforms: Dict[str, Dict[str, list]] = {}
    fovs: Dict[str, list] = {}
    for node in root.iter("node"):
        if node.get("id") != "EffectComponent":
            continue
        try:
            ph = int(_attr_value(node, "PhaseIndex", "-1") or -1)
        except ValueError:
            continue
        if ph != int(phase):
            continue
        kind = _attr_value(node, "Type")
        actor = ""
        actor_node = _find_child(node, "Actor")
        if actor_node is not None:
            actor = (_attr_value(actor_node, "UUID") or "").lower()
        if not actor:
            continue
        if kind == "TLTransform":
            channels = _find_child(node, "TransformChannels")
            if channels is None:
                continue
            parsed: List[list] = []
            ref_rows: List[Tuple[float, str, str]] = []
            channel_index = 0
            for channel in _child_nodes(channels):
                if channel.get("id") != "TransformChannel":
                    continue
                keys_node = _find_child(channel, "Keys")
                rows = []
                if keys_node is not None:
                    for key in _child_nodes(keys_node):
                        if key.get("id") != "Key":
                            continue
                        try:
                            when = float(_attr_value(key, "Time", "0") or 0)
                        except ValueError:
                            continue
                        if channel_index == 5:
                            ref = _key_frame_of_reference(key)
                            if ref is not None:
                                ref_rows.append((when, ref[0], ref[1]))
                            continue
                        raw = (_attr_value(key, "Value") or "").strip()
                        if not raw:
                            continue
                        parts = [float(p) for p in raw.split() if p]
                        if len(parts) == 1:
                            rows.append((when, parts[0]))
                        elif len(parts) >= 4:
                            rows.append((when, tuple(parts[:4])))
                if channel_index < 4:
                    parsed.append(rows)
                channel_index += 1
            if len(parsed) < 4:
                continue
            bag = transforms.setdefault(
                actor, {"x": [], "y": [], "z": [], "quat": [], "ref": []}
            )
            for name, rows in zip(("x", "y", "z", "quat"), parsed[:4]):
                bag[name].extend(rows)
            bag["ref"].extend(ref_rows)
        elif kind == "TLCameraFoV":
            keys_node = _find_child(node, "Keys")
            if keys_node is None:
                continue
            rows = fovs.setdefault(actor, [])
            for key in _child_nodes(keys_node):
                if key.get("id") != "Key":
                    continue
                try:
                    when = float(_attr_value(key, "Time", "0") or 0)
                    fov = float(_attr_value(key, "FoV", "0") or 0)
                except ValueError:
                    continue
                if fov > 0.0:
                    rows.append((when, fov))
    for bag in transforms.values():
        for name, rows in bag.items():
            if rows:
                rows.sort(key=lambda row: row[0])
    for rows in fovs.values():
        rows.sort(key=lambda row: row[0])
    return transforms, fovs


def _sample_curve(keys: list, time_sec: float):
    """Linear sample. Clamps outside the keys. Quaternion rows are nlerped."""
    if not keys:
        return None
    if time_sec <= keys[0][0]:
        return keys[0][1]
    if time_sec >= keys[-1][0]:
        return keys[-1][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if time_sec > t1:
            continue
        span = t1 - t0
        alpha = 0.0 if span <= 1e-8 else (time_sec - t0) / span
        if isinstance(v0, tuple):
            if sum(a * b for a, b in zip(v0, v1)) < 0.0:
                v1 = tuple(-part for part in v1)
            mixed = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(v0, v1))
            length = math.sqrt(sum(part * part for part in mixed)) or 1.0
            return tuple(part / length for part in mixed)
        return (1.0 - alpha) * float(v0) + alpha * float(v1)
    return keys[-1][1]


def _sample_ref_step(keys: list, time_sec: float) -> Optional[Tuple[str, str]]:
    """Latest frameOfReference at or before time_sec.

    Shot cuts are sampled on panel frames; the ref key time is often a few
    milliseconds later in the LSX, so clamp to the first ref when still early.
    """
    if not keys:
        return None
    chosen = None
    for when, target, bone in keys:
        if when <= time_sec + 0.05:
            chosen = (target, bone)
        else:
            break
    if chosen is None:
        when, target, bone = keys[0]
        chosen = (target, bone)
    return chosen


# Body-visual rest for Socket_DIAG_Camera, parented to Dummy_Root.
# The cinematic GR2 and Body_Base only carry deform bones. These translations
# are the identity-rotation socket on each race body (_merged.lsf).
_DIAG_CAMERA_SOCKET = "Socket_DIAG_Camera"
_DIAG_SOCKET_REST = {
    "DWR_F": (-0.276169, 1.337018, 0.082000),
    "DWR_M": (-0.270490, 1.393900, 0.082000),
    "ELF_FS": (-0.289000, 1.896000, 0.082000),
    "ELF_F": (-0.287000, 1.692000, 0.094000),
    "ELF_MS": (-0.262229, 2.006283, 0.082000),
    "ELF_M": (-0.294000, 1.794000, 0.082000),
    "GTY_F": (-0.288000, 1.695000, 0.094000),
    "GTY_M": (-0.311782, 1.770910, 0.082000),
    "GNO_F": (-0.266000, 1.191227, 0.082000),
    "GNO_M": (-0.266000, 1.225000, 0.082000),
    "HEL_FS": (-0.289000, 1.896000, 0.082000),
    "HEL_F": (-0.287000, 1.692000, 0.094000),
    "HEL_MS": (-0.262229, 2.006283, 0.082000),
    "HEL_M": (-0.294000, 1.794000, 0.082000),
    "HFL_F": (-0.266000, 1.191000, 0.082000),
    "HFL_M": (-0.266000, 1.225000, 0.082000),
    "HUM_CH": (-0.294000, 1.184641, 0.082000),
    "HUM_FS": (-0.289000, 1.896000, 0.082000),
    "HUM_F": (-0.287000, 1.692000, 0.094000),
    "HUM_MS": (-0.262229, 2.006283, 0.082000),
    "HUM_M": (-0.293590, 1.794367, 0.082059),
    "TIF_CH": (-0.294000, 1.185000, 0.082000),
    "TIF_FS": (-0.289436, 1.896434, 0.082000),
    "TIF_F": (-0.256000, 1.733000, 0.082000),
    "TIF_MS": (-0.262229, 2.006283, 0.082000),
    "TIF_M": (-0.294000, 1.794000, 0.082000),
}


def _diag_socket_rest_for_arm(arm) -> Optional[Tuple[float, float, float]]:
    """Body prefix in the armature name → socket rest, longest prefix first."""
    name = str(getattr(arm, "name", "") or "").upper()
    for prefix in sorted(_DIAG_SOCKET_REST, key=len, reverse=True):
        if f"_{prefix}_" in name:
            return _DIAG_SOCKET_REST[prefix]
    return None


def _graft_diag_socket_edit(arm) -> bool:
    """Add Socket_DIAG_Camera in the current edit mode. Rest rotation is identity."""
    if bpy is None or arm is None or arm.type != "ARMATURE":
        return False
    rest = _diag_socket_rest_for_arm(arm)
    if rest is None:
        return False
    bones = arm.data.edit_bones
    parent = bones.get("Dummy_Root")
    if parent is None:
        return False
    bone = bones.get(_DIAG_CAMERA_SOCKET)
    if bone is None:
        bone = bones.new(_DIAG_CAMERA_SOCKET)
    bone.parent = parent
    bone.use_connect = False
    bone.use_deform = False
    # Same orientation as Dummy_Root, shifted by the body socket. A zero-length
    # bone loses its axes, so keep a short tail along the parent's Y.
    head = (parent.matrix @ Matrix.Translation(Vector(rest))).to_translation()
    direction = parent.y_axis.normalized()
    if direction.length < 1e-8:
        direction = Vector((0.0, 1.0, 0.0))
    bone.head = head
    bone.tail = head + direction * 0.05
    roll_axis = parent.z_axis
    if roll_axis.length < 1e-8:
        roll_axis = Vector((0.0, 0.0, 1.0))
    bone.align_roll(roll_axis)
    return True


def _ensure_diag_camera_socket(context, arm) -> bool:
    """Put the dialogue camera socket on a rig that was imported without it."""
    if bpy is None or context is None or arm is None:
        return False
    if arm.mode == "EDIT":
        return _graft_diag_socket_edit(arm)
    if _diag_socket_rest_for_arm(arm) is None:
        return False
    if arm.data.bones.get("Dummy_Root") is None:
        return False
    prev = context.view_layer.objects.active
    # The sync operator's context stays on the slot empty. mode_set then only
    # offers OBJECT, so the armature has to be the override's active object.
    window = getattr(context, "window", None)
    area = None
    region = None
    screen = getattr(context, "screen", None)
    if screen is not None:
        for candidate in screen.areas:
            if candidate.type != "VIEW_3D":
                continue
            area = candidate
            region = next(
                (item for item in candidate.regions if item.type == "WINDOW"),
                None,
            )
            break
    override = {"active_object": arm, "object": arm, "selected_objects": [arm]}
    if window is not None:
        override["window"] = window
    if area is not None and region is not None:
        override["area"] = area
        override["region"] = region
    with context.temp_override(**override):
        context.view_layer.objects.active = arm
        try:
            arm.select_set(True)
        except RuntimeError:
            pass
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            added = _graft_diag_socket_edit(arm)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
    if prev is not None:
        context.view_layer.objects.active = prev
    if added:
        pose = arm.pose.bones.get(_DIAG_CAMERA_SOCKET)
        if pose is not None:
            pose["gr2lab_socket"] = True
    return added


def _diag_socket_world_matrix(
    context,
    target_uuid: str,
    bone_name: str,
    frame: int,
    actor_slots: Dict[str, str],
):
    """Posed Socket_DIAG_Camera in world space. No head/chest fallback."""
    if bpy is None or context is None:
        return None
    if bone_name != _DIAG_CAMERA_SOCKET:
        return None
    target = (target_uuid or "").lower()
    empty_name = actor_slots.get(target)
    if not empty_name:
        return None
    empty = bpy.data.objects.get(empty_name)
    if empty is None:
        return None
    arms = _armatures_on_empty(empty)
    if not arms:
        return None
    arm = arms[0]
    scene = context.scene
    scene.frame_set(int(frame))
    for handler in bpy.app.handlers.frame_change_pre:
        if getattr(handler, "__name__", "") == "gr2lab_prebody_nla_mute_handler":
            handler(scene)
    context.view_layer.update()
    pose = arm.pose.bones.get(_DIAG_CAMERA_SOCKET)
    if pose is None:
        return None
    return arm.matrix_world @ pose.matrix


def _socket_camera_pose(parent, pos, quat):
    """Bone-local timeline offset on the socket. Local +Z is the look axis."""
    qx, qy, qz, qw = (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )
    offset = Matrix.LocRotScale(
        Vector((float(pos[0]), float(pos[1]), float(pos[2]))),
        Quaternion((qw, qx, qy, qz)).normalized(),
        Vector((1.0, 1.0, 1.0)),
    )
    world = parent @ offset
    look = world.to_3x3().col[2]
    up = world.to_3x3().col[1]
    if look.length < 1e-8:
        look = Vector((0.0, 1.0, 0.0))
    look.normalize()
    if up.length < 1e-8:
        up = Vector((0.0, 0.0, 1.0))
    up.normalize()
    back = -look
    x_axis = up.cross(back)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0)).cross(back)
        if x_axis.length < 1e-6:
            x_axis = Vector((0.0, 1.0, 0.0))
    x_axis.normalize()
    y_axis = back.cross(x_axis).normalized()
    matrix = Matrix((x_axis, y_axis, back)).transposed()
    return world.to_translation(), matrix.to_quaternion()


def _camera_parent_matrix(
    context,
    target_uuid: str,
    bone_name: str,
    frame: int,
    actor_slots: Dict[str, str],
) -> Optional[Matrix]:
    if bpy is None or context is None:
        return None
    target = (target_uuid or "").lower()
    if not target or target == "00000000-0000-0000-0000-000000000000":
        return None
    empty_name = actor_slots.get(target)
    if not empty_name:
        return None
    empty = bpy.data.objects.get(empty_name)
    if empty is None:
        return None
    arms = _armatures_on_empty(empty)
    if not arms:
        return None
    arm = arms[0]
    scene = context.scene
    scene.frame_set(int(frame))
    context.view_layer.update()
    fallbacks = []
    if bone_name:
        fallbacks.append(bone_name)
    fallbacks.extend(
        (
            "Socket_DIAG_Camera",
            "Chest_M",
            "Head_M",
            "Neck_M",
            "Dummy_Root",
            "Root_M",
        )
    )
    for name in fallbacks:
        pb = arm.pose.bones.get(name)
        if pb is not None:
            return arm.matrix_world @ pb.matrix
    return arm.matrix_world.copy()


def _camera_pose_is_placeholder(pos) -> bool:
    """Unplaced BG3 camera rest: origin orbit, not a staged shot."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return abs(x) < 1e-3 and abs(y - 3.535534) < 0.05 and abs(z + 3.535534) < 0.05


def _glasses_camera_pose(
    pos,
    quat,
    y_up_display: bool,
    *,
    forward_axis: int = 2,
):
    """Y-up cinematic camera → Blender camera (looks down local -Z, +Y up).

    World shots use local +Z as look. Bone-attached CU shots use local +Y.
    """
    tx, ty, tz = (float(pos[0]), float(pos[1]), float(pos[2]))
    qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    loc = Vector((tx, tz, ty) if y_up_display else (tx, ty, tz))
    rotation = Quaternion((qw, qx, qy, qz)).normalized().to_matrix()
    if y_up_display:
        basis = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
        rot_b = basis @ rotation
    else:
        rot_b = rotation
    axis = int(forward_axis)
    if axis == 1:
        forward = rot_b.col[1].copy()
        up = rot_b.col[2].copy()
    else:
        forward = rot_b.col[2].copy()
        up = rot_b.col[1].copy()
    if forward.length < 1e-8:
        forward = Vector((0.0, 1.0, 0.0))
    forward.normalize()
    if up.length < 1e-8:
        up = Vector((0.0, 0.0, 1.0))
    up.normalize()
    z_axis = -forward
    x_axis = up.cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0)).cross(z_axis)
        if x_axis.length < 1e-6:
            x_axis = Vector((0.0, 1.0, 0.0))
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    matrix = Matrix((x_axis, y_axis, z_axis)).transposed()
    return loc, matrix.to_quaternion()


def _resolve_camera_pose_at(
    context,
    when: float,
    frame: int,
    bag: Dict[str, list],
    actor_slots: Dict[str, str],
    y_up_display: bool,
) -> Optional[Tuple[Vector, Quaternion]]:
    if not bag:
        return None
    px = _sample_curve(bag.get("x") or [], when)
    py = _sample_curve(bag.get("y") or [], when)
    pz = _sample_curve(bag.get("z") or [], when)
    quat = _sample_curve(bag.get("quat") or [], when)
    if None in (px, py, pz, quat):
        return None
    pos = (float(px), float(py), float(pz))
    ref = _sample_ref_step(bag.get("ref") or [], when)
    if ref is not None:
        parent = _camera_parent_matrix(
            context, ref[0], ref[1], frame, actor_slots
        )
        if parent is not None:
            best_world = None
            best_dot = -1.0
            targets: List[Vector] = []
            for empty_name in dict.fromkeys(actor_slots.values()):
                empty = bpy.data.objects.get(empty_name)
                if empty is None:
                    continue
                for arm in _armatures_on_empty(empty):
                    for bone_name in ("Head_M", "Chest_M"):
                        pb = arm.pose.bones.get(bone_name)
                        if pb is not None:
                            targets.append(arm.matrix_world @ pb.head)
            for axis in (2, 1):
                loc, rot = _glasses_camera_pose(
                    pos, quat, y_up_display, forward_axis=axis
                )
                world = parent @ Matrix.LocRotScale(
                    loc, rot, Vector((1.0, 1.0, 1.0))
                )
                if not targets:
                    return world.to_translation(), world.to_quaternion()
                look = (-world.to_3x3().col[2]).normalized()
                origin = world.to_translation()
                score = max(
                    look.dot((point - origin).normalized())
                    for point in targets
                    if (point - origin).length > 1e-4
                )
                if score > best_dot:
                    best_dot = float(score)
                    best_world = world
            if best_world is not None:
                return best_world.to_translation(), best_world.to_quaternion()
    loc, rot = _glasses_camera_pose(pos, quat, y_up_display, forward_axis=2)
    return loc, rot


def _replace_id_action(id_block, action_name: str) -> None:
    ad = id_block.animation_data
    if ad is not None and ad.action is not None:
        old = ad.action
        ad.action = None
        if old.users == 0:
            bpy.data.actions.remove(old)
    if id_block.animation_data is None:
        id_block.animation_data_create()
    id_block.animation_data.action = bpy.data.actions.new(action_name)


def _set_shot_curve_interpolation(id_block, cut_frames: set) -> None:
    """Hold across a shot cut, move between keys inside one shot."""
    fcurves = _object_action_fcurves(id_block)
    if fcurves is None:
        return
    for fc in fcurves:
        points = list(fc.keyframe_points)
        for index, kp in enumerate(points):
            nxt = points[index + 1] if index + 1 < len(points) else None
            if nxt is not None and int(round(nxt.co[0])) in cut_frames:
                kp.interpolation = "CONSTANT"
            elif nxt is None:
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = "LINEAR"


def ensure_scene_timeline_camera(
    context,
    *,
    scene_path: Path,
    timeline_path: Path,
    phase: int,
    band_sec: float,
    y_up_display: bool = True,
    player_source: str = "",
    player_empty_name: str = "Initiator_1",
    partner_empty_name: str = "Additional_1",
) -> str:
    """Keyframe the scene camera from the active shot's TLTransform."""
    if bpy is None or context is None:
        return ""
    fps = float(context.scene.render.fps or 30)
    shots = _parse_timeline_tl_shots(
        timeline_path, phase=phase, band_sec=band_sec, fps=fps
    )
    if not shots:
        return ""
    cams = _parse_scene_tl_cameras(scene_path)
    actor_cams = _timeline_actor_camera_map(timeline_path)
    curves, fov_curves = _parse_phase_camera_curves(timeline_path, phase)
    actor_slots = _camera_actor_slot_map(
        timeline_path,
        phase,
        player_source,
        player_empty_name=player_empty_name,
        partner_empty_name=partner_empty_name,
    )
    obj = bpy.data.objects.get("GR2LAB_Scene_Camera")
    if obj is None or obj.type != "CAMERA":
        obj = bpy.data.objects.new(
            "GR2LAB_Scene_Camera", bpy.data.cameras.new("GR2LAB_Scene_Camera")
        )
        context.scene.collection.objects.link(obj)
    obj.rotation_mode = "QUATERNION"
    obj.scale = (1.0, 1.0, 1.0)
    obj.data.sensor_fit = "VERTICAL"
    _replace_id_action(obj, "GR2LAB_Scene_CameraAction")
    _replace_id_action(obj.data, "GR2LAB_Scene_CameraDataAction")
    keyed = 0
    cut_frames = {int(shot["frame"]) for shot in shots}

    def frame_to_time(frame: int) -> float:
        return float(band_sec) + (float(frame) / fps)

    for index, shot in enumerate(shots):
        start = int(shot["frame"])
        guid = str(shot.get("camera_guid") or "")
        bag = curves.get(guid)
        fov_keys = fov_curves.get(guid) or []
        end = int(shots[index + 1]["frame"]) if index + 1 < len(shots) else None
        # Bone-local offsets ride the imported Socket_DIAG_Camera. One key per
        # timeline knot (and the shot ends), not a key on every frame.
        if bag and bag.get("ref"):
            ref = _sample_ref_step(bag.get("ref") or [], frame_to_time(start))
            if ref is None or ref[1] != _DIAG_CAMERA_SOCKET:
                continue
            sample_frames = {start}
            for rows in (bag.get("x"), bag.get("y"), bag.get("z"), bag.get("quat")):
                if not rows:
                    continue
                for when, _value in rows:
                    frame = panel_seconds_to_timeline_frame(
                        when - float(band_sec), fps
                    )
                    if frame >= start and (end is None or frame < end):
                        sample_frames.add(frame)
            if end is not None and end - 1 > start:
                sample_frames.add(end - 1)
            for frame in sorted(sample_frames):
                when = frame_to_time(frame)
                px = _sample_curve(bag.get("x") or [], when)
                py = _sample_curve(bag.get("y") or [], when)
                pz = _sample_curve(bag.get("z") or [], when)
                quat = _sample_curve(bag.get("quat") or [], when)
                if None in (px, py, pz, quat):
                    continue
                step = _sample_ref_step(bag.get("ref") or [], when) or ref
                parent = _diag_socket_world_matrix(
                    context, step[0], step[1], frame, actor_slots
                )
                if parent is None:
                    continue
                loc, rot = _socket_camera_pose(parent, (px, py, pz), quat)
                obj.location = loc
                obj.rotation_quaternion = rot
                obj.scale = (1.0, 1.0, 1.0)
                obj.keyframe_insert(data_path="location", frame=frame)
                obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                fov = _sample_curve(fov_keys, when) if fov_keys else None
                if fov:
                    half = math.radians(float(fov)) * 0.5
                    obj.data.lens = float(obj.data.sensor_height) / (
                        2.0 * math.tan(half)
                    )
                    obj.data.keyframe_insert(data_path="lens", frame=frame)
                keyed += 1
            continue
        sample_frames = {start}
        if bag:
            for rows in (bag.get("x"), bag.get("y"), bag.get("z"), bag.get("quat")):
                if not rows:
                    continue
                for when, _value in rows:
                    frame = panel_seconds_to_timeline_frame(
                        when - float(band_sec), fps
                    )
                    if frame >= start and (end is None or frame < end):
                        sample_frames.add(frame)
        if end is not None and end - 1 > start:
            sample_frames.add(end - 1)
        elif end is None and bag:
            last = start
            for rows in (bag.get("x"), bag.get("y"), bag.get("z"), bag.get("quat")):
                if not rows:
                    continue
                for when, _value in rows:
                    last = max(
                        last,
                        panel_seconds_to_timeline_frame(
                            when - float(band_sec), fps
                        ),
                    )
            if last > start:
                sample_frames.add(last)
        for frame in sorted(sample_frames):
            when = frame_to_time(frame)
            resolved = None
            if bag and bag.get("x") and bag.get("y") and bag.get("z") and bag.get("quat"):
                px = _sample_curve(bag["x"], when)
                py = _sample_curve(bag["y"], when)
                pz = _sample_curve(bag["z"], when)
                quat = _sample_curve(bag["quat"], when)
                if None not in (px, py, pz, quat):
                    resolved = _glasses_camera_pose(
                        (px, py, pz), quat, y_up_display
                    )
            if resolved is None and frame == start:
                cam_id = actor_cams.get(guid, "")
                block = cams.get(cam_id) if cam_id else None
                if block is not None:
                    resolved = _glasses_camera_pose(
                        block["pos"], block["quat"], y_up_display, forward_axis=2
                    )
            if resolved is None:
                continue
            loc, rot = resolved
            obj.location = loc
            obj.rotation_quaternion = rot
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            fov = _sample_curve(fov_keys, when) if fov_keys else None
            if fov:
                # Timeline FoV is vertical degrees. Blender keys lens, and with
                # sensor_fit VERTICAL that lens is the vertical field.
                half = math.radians(float(fov)) * 0.5
                obj.data.lens = float(obj.data.sensor_height) / (2.0 * math.tan(half))
                obj.data.keyframe_insert(data_path="lens", frame=frame)
            keyed += 1
    _set_shot_curve_interpolation(obj, cut_frames)
    _set_shot_curve_interpolation(obj.data, cut_frames)
    context.scene.camera = obj
    return f"scene camera: {keyed} timeline keys"


def _phase_panel_band_start(timeline_path: Path, phase: int) -> float:
    """Glasses phase ruler 0.00 — first keyed event in this phase."""
    try:
        root = ET.parse(timeline_path).getroot()
    except (ET.ParseError, OSError):
        return 0.0
    best = None
    for node in root.iter("node"):
        if _attr_value(node, "PhaseIndex") != str(phase):
            continue
        try:
            when = float(_attr_value(node, "StartTime", "0") or 0)
        except ValueError:
            continue
        if best is None or when < best:
            best = when
    return float(best or 0.0)


def _larian_pos_blender(
    pos: Tuple[float, float, float], y_up_display: bool, swap_location: bool
) -> Tuple[float, float, float]:
    loc, _ = larian_to_blender(pos, (0.0, 0.0, 0.0, 1.0), y_up_display, swap_location)
    return (float(loc[0]), float(loc[1]), float(loc[2]))


def _segment_slot_spacing_m(
    seg: Dict,
    *,
    y_up_display: bool,
    swap_location: bool,
) -> Optional[float]:
    player_pose = seg.get("player_pose")
    partner_pose = seg.get("partner_pose")
    if not player_pose or not partner_pose:
        return None
    a = Vector(_larian_pos_blender(player_pose[0], y_up_display, swap_location))
    b = Vector(_larian_pos_blender(partner_pose[0], y_up_display, swap_location))
    return float((a - b).length)


def _commit_nla_span(strip, scene_start: float, scene_end: float, action_start: float, action_end: float) -> None:
    """Force a strip to cover the Glasses frame window at the requested action range.

    Blender rewrites ``action_frame_end`` when the scene span and the action span
    disagree, which was cutting segment 0 off near frame 61 and restarting the clip.
    """
    scene_start = float(scene_start)
    scene_end = float(scene_end)
    action_start = float(action_start)
    action_end = float(action_end)
    if scene_end <= scene_start:
        scene_end = scene_start + 1.0
    if action_end <= action_start:
        action_end = action_start + 1.0
    strip.blend_type = "REPLACE"
    try:
        strip.extrapolation = "HOLD_FORWARD"
    except Exception:
        pass
    try:
        strip.repeat = 1.0
    except Exception:
        pass
    try:
        strip.use_sync_length = False
    except Exception:
        pass
    # Blender's strip scale is scene-length / action-length. Playback speed is
    # 1/scale, so a 0.62 play rate must be stored as scale 1/0.62. The other
    # way loops the clip every few dozen frames (the frame 60–80 restart).
    strip.scale = 1.0
    strip.frame_start = scene_start
    strip.action_frame_start = action_start
    strip.action_frame_end = action_end
    scene_len = scene_end - scene_start
    action_len = action_end - action_start
    if scene_len > 0 and action_len > 0:
        strip.scale = scene_len / action_len
    strip.frame_start = scene_start
    strip.frame_end = scene_end


def _slot_matrix_from_pose(
    pose,
    base_pos: Tuple[float, float, float],
    *,
    is_player: bool,
    y_up_display: bool,
    swap_location: bool,
    anchor_relative: bool = False,
) -> "Matrix":
    """Stage empty from timeline TargetTransform (Glasses position + inspector Euler)."""
    del is_player, base_pos, anchor_relative
    pos = pose[0] if pose else (0.0, 0.0, 0.0)
    quat = pose[1] if pose and len(pose) > 1 else (0.0, 0.0, 0.0, 1.0)
    return _matrix_from_timeline_slot_pose(pos, quat, y_up_display, swap_location)


def collect_paired_slot_segments(
    gr2_path: str,
    scene_path: Path,
    binding: ClipBinding,
) -> Tuple[List[Dict], float]:
    """Player body rows in one phase, with the partner body row at the same time.

    Rows without a TargetTransform still change AnimationPlayStartOffset, so they
    stay in the list. Slot empties are keyed only when a pose is present.
    """
    timeline = _timeline_beside_scene(scene_path)
    if timeline is None:
        return [], 0.0
    doc = _parse_timeline_bindings(timeline)
    source = (binding.source_id or "").lower()
    phase = int(binding.phase_index)
    if not source or phase < 0:
        return [], 0.0
    hits = [
        row
        for row in doc.get("anims") or []
        if row.get("source") == source
        and int(row.get("phase", -1)) == phase
    ]
    hits.sort(key=lambda row: float(row.get("start") or 0.0))
    if len(hits) < 2:
        return [], 0.0
    band = _phase_panel_band_start(timeline, phase)
    segments: List[Dict] = []
    for i, hit in enumerate(hits):
        partner = _partner_anim_for_hit(
            hit,
            doc,
            source,
            prefer_companion_index=_partner_index_pref(gr2_path),
        )
        panel_start = float(hit["start"]) - band
        if i + 1 < len(hits):
            panel_end = float(hits[i + 1]["start"]) - band
        else:
            panel_end = float(hit.get("end") or hit["start"]) - band
            if panel_end <= panel_start + 1e-4:
                panel_end = panel_start + max(
                    0.05, float(hit.get("end") or 0) - float(hit.get("start") or 0)
                )
        segments.append(
            {
                "panel_start": panel_start,
                "panel_end": panel_end,
                "player_pose": hit.get("pose"),
                "partner_pose": (partner or {}).get("pose"),
                "player_play": float(hit.get("play_start") or 0.0),
                "partner_play": float((partner or {}).get("play_start") or 0.0),
                "player_rate": float(hit.get("play_rate") or 1.0),
                "partner_rate": float((partner or {}).get("play_rate") or 1.0),
            }
        )
    return segments, band


def _armature_action(arm) -> Optional[object]:
    ad = getattr(arm, "animation_data", None)
    if ad is None:
        return None
    if ad.action is not None:
        return ad.action
    for track in ad.nla_tracks or []:
        for strip in track.strips:
            if strip.action is not None:
                return strip.action
    run_id = arm.get("gr2lab_run_id") if arm is not None else None
    if run_id:
        rid = str(run_id)
        for act in bpy.data.actions:
            if rid in act.name and act.name.startswith("GR2LAB"):
                return act
    return None


def _apply_segment_nla(
    arm,
    segments: Sequence[Dict],
    *,
    fps: float,
    first_panel: float,
    is_player: bool,
) -> bool:
    """Map scene frames to GR2 time per timeline row (play offset + rate)."""
    action = _armature_action(arm)
    if action is None or not segments:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    ad = arm.animation_data
    action_slot = getattr(ad, "action_slot", None)
    for track in ad.nla_tracks or []:
        for strip in track.strips:
            if action_slot is None and getattr(strip, "action_slot", None) is not None:
                action_slot = strip.action_slot
    try:
        ad.use_tweak_mode = False
    except Exception:
        pass
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])
    ad.use_nla = True
    track = ad.nla_tracks.new()
    track.name = "GR2LAB_Cinematic"
    for i, seg in enumerate(segments):
        panel_start = float(seg["panel_start"])
        panel_end = float(seg["panel_end"])
        if panel_end <= panel_start + 1e-6:
            panel_end = panel_start + 1.0 / max(fps, 1.0)
        play = float(seg["player_play"] if is_player else seg["partner_play"])
        rate = float(seg["player_rate"] if is_player else seg["partner_rate"])
        if rate <= 1e-6:
            rate = 1.0
        scene_start = max(1, int(round(1.0 + (panel_start - first_panel) * fps)))
        scene_end = max(
            scene_start,
            int(round(1.0 + (panel_end - first_panel) * fps)) - 1,
        )
        action_start = max(1, int(round(1.0 + play * fps)))
        action_span = max(
            1, int(round((panel_end - panel_start) * rate * fps))
        )
        action_end = action_start + action_span - 1
        scene_len = max(1.0, float(scene_end - scene_start + 1))
        action_len = max(1.0, float(action_end - action_start + 1))
        strip = track.strips.new(f"seg{i}", scene_start, action)
        strip.action_frame_start = float(action_start)
        strip.action_frame_end = float(action_end)
        strip.scale = action_len / scene_len
        strip.frame_start = float(scene_start)
        strip.frame_end = float(scene_end)
        strip.blend_type = "REPLACE"
        if action_slot is not None and hasattr(strip, "action_slot"):
            try:
                strip.action_slot = action_slot
            except Exception:
                pass
    ad.action = None
    if hasattr(ad, "action_slot"):
        try:
            ad.action_slot = None
        except Exception:
            pass
    _ensure_gr2lab_nla_audible(ad)
    return True


def _rig_draw_objects(arm) -> List:
    """Armature, skinned meshes, and tail rig — what TLShowVisual toggles."""
    if arm is None:
        return []
    out = []
    seen = set()

    def add(obj):
        if obj is None or obj.name in seen:
            return
        seen.add(obj.name)
        out.append(obj)

    add(arm)
    for obj in _subtree_objects(arm):
        add(obj)
    for tail in _tail_rigs_of(arm):
        add(tail)
        for obj in _subtree_objects(tail):
            add(obj)
    return out


def _source_actor_uuid(doc: Dict, source_id: str, phase: int) -> str:
    source = (source_id or "").lower()
    for row in doc.get("anims") or []:
        if row.get("source") == source and int(row.get("phase", -1)) == phase:
            actor = str(row.get("actor") or "")
            if actor:
                return actor
    return ""


def _show_visual_hide_keys(
    timeline: Path,
    phase: int,
    actor_uuid: str,
    fps: float,
    band: float,
) -> List[Tuple[int, bool]]:
    """(frame, hidden) from TLShowVisual. Missing ShowVisual means visible."""
    want = (actor_uuid or "").lower()
    if not want:
        return []
    try:
        root = ET.parse(timeline).getroot()
    except (ET.ParseError, OSError):
        return []
    keys: List[Tuple[float, bool]] = []
    for node in root.iter("node"):
        if node.get("id") != "EffectComponent":
            continue
        if _attr_value(node, "Type") != "TLShowVisual":
            continue
        if _attr_value(node, "PhaseIndex") != str(phase):
            continue
        actor_node = _find_child(node, "Actor")
        if actor_node is None:
            continue
        if _attr_value(actor_node, "UUID").lower() != want:
            continue
        keys_node = _find_child(node, "Keys")
        if keys_node is None:
            continue
        for key_node in _child_nodes(keys_node):
            if key_node.get("id") != "Key":
                continue
            try:
                when = float(_attr_value(key_node, "Time", "0") or 0)
            except ValueError:
                continue
            shown = _attr_value(key_node, "ShowVisual", "True")
            hidden = str(shown).lower() == "false"
            keys.append((when, hidden))
    keys.sort(key=lambda item: item[0])
    out: List[Tuple[int, bool]] = []
    for when, hidden in keys:
        frame = panel_seconds_to_timeline_frame(float(when) - float(band), fps)
        out.append((frame, hidden))
    return out


def _apply_show_visual_hide(scene, objects, keys: Sequence[Tuple[int, bool]]) -> str:
    if bpy is None or not objects or not keys:
        return ""
    for obj in objects:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=0)
        obj.keyframe_insert(data_path="hide_render", frame=0)
    for frame, hidden in keys:
        for obj in objects:
            obj.hide_viewport = bool(hidden)
            obj.hide_render = bool(hidden)
            obj.keyframe_insert(data_path="hide_viewport", frame=int(frame))
            obj.keyframe_insert(data_path="hide_render", frame=int(frame))
    for obj in objects:
        _constant_keyframes(obj)
    hidden_at = [str(frame) for frame, hidden in keys if hidden]
    if not hidden_at:
        return ""
    return "hide visual @ " + ",".join(hidden_at)


def _actor_root_world(arm) -> Optional["Vector"]:
    """Dummy_Root world origin. Glasses 'selected actor' XYZ includes this motion."""
    if arm is None or getattr(arm, "pose", None) is None:
        return None
    pb = arm.pose.bones.get("Dummy_Root") or arm.pose.bones.get("Root_M")
    if pb is None:
        return None
    return (arm.matrix_world @ pb.matrix).to_translation()


def _root_armature_basis(arm):
    """Dummy_Root rotation in armature space. Identity until the clip turns it."""
    if arm is None or getattr(arm, "pose", None) is None:
        return None
    pb = arm.pose.bones.get("Dummy_Root") or arm.pose.bones.get("Root_M")
    if pb is None:
        return None
    return pb.matrix.to_3x3()


def _absorb_root_rotation(empty, arm) -> None:
    """Glasses yaw is the actor's world facing, and the root bone may already hold it.

    Pasting that yaw onto the empty as well spins the body a second time
    (frame 1218, still facing the wrong way at frame 2018). Fold the bone
    rotation back into the empty so the world facing stays on the Glasses yaw.
    """
    basis_b = _root_armature_basis(arm)
    if empty is None or basis_b is None:
        return
    if basis_b.to_quaternion().angle < math.radians(2.0):
        return
    target = empty.matrix_world.to_3x3()
    solved = target @ basis_b.inverted()
    delta = Euler(empty.delta_rotation_euler, "XYZ").to_matrix()
    empty.rotation_euler = (delta.inverted() @ solved).to_euler("XYZ")


def _object_world_from_channels(obj):
    """World matrix from loc/rot/scale channels, without a depsgraph update.

    Matches Blender for these slot empties: delta loc, loc, delta rot, rot,
    delta scale, scale. Parent world is whatever the last update left it.
    """
    if obj.rotation_mode == "QUATERNION":
        rot = obj.rotation_quaternion.to_matrix().to_4x4()
    elif obj.rotation_mode == "AXIS_ANGLE":
        ang = obj.rotation_axis_angle
        rot = Matrix.Rotation(float(ang[0]), 4, Vector((ang[1], ang[2], ang[3])))
    else:
        rot = Euler(obj.rotation_euler, obj.rotation_mode).to_matrix().to_4x4()
    drot = Euler(obj.delta_rotation_euler, "XYZ").to_matrix().to_4x4()
    local = (
        Matrix.Translation(obj.delta_location)
        @ Matrix.Translation(obj.location)
        @ drot
        @ rot
        @ Matrix.Diagonal((*obj.delta_scale, 1.0)).to_4x4()
        @ Matrix.Diagonal((*obj.scale, 1.0)).to_4x4()
    )
    if obj.parent is None:
        return local
    return obj.parent.matrix_world @ obj.matrix_parent_inverse @ local


def _refit_slot_locations_excluding_root(scene, frames, pairs) -> None:
    """TargetTransform is actor world position. Root motion stays on the bone.

    Shift the empty so Dummy_Root lands on that position. The Glasses Z
    channel is already mapped to Blender Y, so a depth change does not move
    the body vertically.

    A later key on the same facing sometimes only exists to cancel sideways
    root drift (player Y 1.113 from frame 160, then 0.947 at frame 511).
    He is hidden at 160 and reappears at 210 already on the drifted line.
    While the empty's X is unchanged, keep the later Y for the whole span
    so he appears on the corrected line.
    """
    if bpy is None or scene is None:
        return
    solved = {}
    for frame in sorted({int(f) for f in frames}):
        scene.frame_set(int(frame))
        gr2lab_prebody_nla_mute_handler(scene)
        # One depsgraph pass per slot frame. Root world after the yaw absorb
        # is composed from the channels, so each character does not update again.
        bpy.context.view_layer.update()
        captured = []
        for empty, arm in pairs:
            if empty is None or arm is None:
                continue
            pb = None
            if getattr(arm, "pose", None) is not None:
                pb = arm.pose.bones.get("Dummy_Root") or arm.pose.bones.get("Root_M")
            captured.append(
                (
                    empty,
                    arm,
                    float(empty.rotation_euler.x),
                    float(empty.rotation_euler.y),
                    arm.matrix_local.copy(),
                    arm.matrix_parent_inverse.copy(),
                    None if pb is None else pb.matrix.copy(),
                )
            )
            _absorb_root_rotation(empty, arm)
            empty.keyframe_insert(data_path="rotation_euler", frame=int(frame))
        for empty, arm, glasses_pitch, glasses_yaw, arm_local, arm_pinv, root_mat in captured:
            if root_mat is None:
                continue
            empty_world = _object_world_from_channels(empty)
            root_w = (empty_world @ arm_pinv @ arm_local @ root_mat).to_translation()
            desired = empty_world.to_translation()
            delta = desired - root_w
            if delta.length >= 1e-5:
                if empty.parent is None:
                    empty.location = Vector(empty.location) + delta
                else:
                    empty.location = Vector(empty.location) + (
                        empty_world.to_3x3().inverted() @ delta
                    )
                empty.keyframe_insert(data_path="location", frame=int(frame))
            solved.setdefault(empty, []).append(
                (
                    int(frame),
                    Vector(empty.location),
                    glasses_pitch,
                    glasses_yaw,
                )
            )
    for empty, samples in solved.items():
        if empty is None or len(samples) < 2:
            continue
        for i in range(len(samples) - 2, -1, -1):
            frame0, loc0, pitch0, yaw0 = samples[i]
            _frame1, loc1, pitch1, yaw1 = samples[i + 1]
            if abs(float(loc1.x) - float(loc0.x)) > 0.05:
                continue
            if abs(pitch1 - pitch0) > math.radians(20.0):
                continue
            dyaw = abs(yaw1 - yaw0) % (2.0 * math.pi)
            dyaw = min(dyaw, (2.0 * math.pi) - dyaw)
            if dyaw > math.radians(15.0):
                continue
            if abs(float(loc1.y) - float(loc0.y)) < 0.05:
                continue
            empty.location = Vector((loc0.x, loc1.y, loc0.z))
            empty.keyframe_insert(data_path="location", frame=int(frame0))
            samples[i] = (frame0, Vector(empty.location), pitch0, yaw0)
    for empty, _arm in pairs:
        if empty is not None:
            _constant_keyframes(empty)


def _constant_keyframes(obj) -> None:
    fcurves = _object_action_fcurves(obj)
    if fcurves is None:
        return
    for fc in fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def _clear_empty_slot_action(empty) -> None:
    ad = getattr(empty, "animation_data", None)
    if ad is None:
        return
    act = ad.action
    empty.animation_data_clear()
    if act is not None and act.users == 0:
        bpy.data.actions.remove(act)


def _apply_slot_pair_pose(
    anchor,
    player_empty,
    seg: Dict,
    *,
    base_partner_pos: Tuple[float, float, float],
    y_up_display: bool,
    swap_location: bool,
) -> bool:
    partner_pose = seg.get("partner_pose")
    player_pose = seg.get("player_pose")
    if not partner_pose or not player_pose:
        return False
    m_anchor = _slot_matrix_from_pose(
        partner_pose,
        base_partner_pos,
        is_player=False,
        y_up_display=y_up_display,
        swap_location=swap_location,
    )
    m_player = _slot_matrix_from_pose(
        player_pose,
        base_partner_pos,
        is_player=True,
        y_up_display=y_up_display,
        swap_location=swap_location,
    )
    if anchor.parent is not None:
        anchor.parent = None
    if y_up_display and not swap_location:
        _write_glasses_slot_channels(
            anchor, partner_pose[0], partner_pose[1], y_up_display, swap_location
        )
    else:
        _apply_empty_world_matrix(anchor, m_anchor)
    _stamp_glasses_inspector_euler(anchor, partner_pose[1])
    if player_empty.parent is not None:
        player_empty.parent = None
    if y_up_display and not swap_location:
        _write_glasses_slot_channels(
            player_empty, player_pose[0], player_pose[1], y_up_display, swap_location
        )
    else:
        _apply_empty_world_matrix(player_empty, m_player)
    _stamp_glasses_inspector_euler(player_empty, player_pose[1])
    return True


def _keyframe_slot_pair(
    anchor,
    player_empty,
    seg: Dict,
    frame: int,
    *,
    base_partner_pos: Tuple[float, float, float],
    y_up_display: bool,
    swap_location: bool,
) -> None:
    if not _apply_slot_pair_pose(
        anchor,
        player_empty,
        seg,
        base_partner_pos=base_partner_pos,
        y_up_display=y_up_display,
        swap_location=swap_location,
    ):
        return
    _keyframe_empty_transform(anchor, frame)
    _keyframe_empty_transform(player_empty, frame)


_LIVE_CINEMATIC_SLOTS: Dict[str, Dict] = {}
_LIVE_SLOT_BUSY = False


def _segment_for_frame(
    segments: Sequence[Dict],
    first_panel: float,
    fps: float,
    frame: int,
) -> Optional[Dict]:
    if not segments:
        return None
    chosen = segments[0]
    for seg in segments:
        seg_frame = panel_seconds_to_timeline_frame(float(seg["panel_start"]), fps)
        if seg_frame <= int(frame):
            chosen = seg
    return chosen


def _apply_live_cinematic_slots(scene) -> None:
    global _LIVE_SLOT_BUSY
    if bpy is None or scene is None or _LIVE_SLOT_BUSY:
        return
    if not _LIVE_CINEMATIC_SLOTS:
        return
    frame = int(scene.frame_current or 1)
    _LIVE_SLOT_BUSY = True
    try:
        for _name, spec in list(_LIVE_CINEMATIC_SLOTS.items()):
            anchor = bpy.data.objects.get(spec.get("anchor_name", ""))
            player_empty = bpy.data.objects.get(spec.get("player_name", ""))
            if anchor is None or player_empty is None:
                continue
            fps = float(spec.get("fps") or 30.0)
            first_panel = float(spec.get("first_panel") or 0.0)
            seg = _segment_for_frame(
                spec.get("segments") or (),
                first_panel,
                fps,
                frame,
            )
            if seg is None:
                continue
            _apply_slot_pair_pose(
                anchor,
                player_empty,
                seg,
                base_partner_pos=spec["base_partner_pos"],
                y_up_display=bool(spec.get("y_up_display", True)),
                swap_location=bool(spec.get("swap_location", False)),
            )
    finally:
        _LIVE_SLOT_BUSY = False


try:
    from bpy.app.handlers import persistent
except ImportError:
    def persistent(fn):
        return fn


@persistent
def gr2lab_live_cinematic_slot_handler(scene):
    _apply_live_cinematic_slots(scene)


def register_live_cinematic_handlers() -> None:
    if bpy is None:
        return
    handler = gr2lab_live_cinematic_slot_handler
    if handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(handler)


def unregister_live_cinematic_handlers() -> None:
    if bpy is None:
        return
    handler = gr2lab_live_cinematic_slot_handler
    while handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(handler)
    _LIVE_CINEMATIC_SLOTS.clear()


def _enable_live_cinematic_slots(
    collection,
    *,
    anchor,
    player_empty,
    sync_segments: Sequence[Dict],
    first_panel: float,
    fps: float,
    base_partner_pos: Tuple[float, float, float],
    y_up_display: bool,
    swap_location: bool,
) -> None:
    _LIVE_CINEMATIC_SLOTS[collection.name] = {
        "anchor_name": anchor.name,
        "player_name": player_empty.name,
        "segments": list(sync_segments),
        "first_panel": float(first_panel),
        "fps": float(fps),
        "base_partner_pos": base_partner_pos,
        "y_up_display": bool(y_up_display),
        "swap_location": bool(swap_location),
    }
    collection["gr2lab_cinematic_live"] = True
    register_live_cinematic_handlers()


def _armatures_on_empty(empty) -> List:
    """Body armatures on the slot empty or under ``{empty}__gr2lab_rx``."""
    if empty is None:
        return []
    out: List = []
    for child in empty.children:
        if child.type == "ARMATURE" and not child.get("gr2lab_tail_of"):
            out.append(child)
        elif child.type == "EMPTY" and child.get("gr2lab_glasses_rx90"):
            for grand in child.children:
                if grand.type == "ARMATURE" and not grand.get("gr2lab_tail_of"):
                    out.append(grand)
    if not out:
        rig_name = empty.get("gr2lab_imported_rig")
        if rig_name:
            arm = bpy.data.objects.get(rig_name) if bpy else None
            if (
                arm is not None
                and arm.type == "ARMATURE"
                and not arm.get("gr2lab_tail_of")
            ):
                parent = arm.parent
                if parent == empty or (
                    parent is not None
                    and parent.get("gr2lab_glasses_rx90")
                    and parent.parent == empty
                ):
                    out.append(arm)
    return out


def _cinematic_rig_pair(player_empty, anchor) -> Tuple[Optional[object], Optional[object]]:
    """Player / partner armatures parented under stage empties (not gr2lab_imported_rig)."""
    player_arm = None
    partner_arm = None
    arms_player = _armatures_on_empty(player_empty)
    if arms_player:
        player_arm = arms_player[0]
    arms_partner = _armatures_on_empty(anchor)
    if arms_partner:
        partner_arm = arms_partner[0]
    return player_arm, partner_arm


def _remove_cinematic_nla(arm) -> None:
    ad = getattr(arm, "animation_data", None)
    if ad is None:
        return
    for track in list(ad.nla_tracks):
        if track.name != "GR2LAB_Cinematic":
            continue
        while track.strips:
            track.strips.remove(track.strips[0])
        ad.nla_tracks.remove(track)


def _use_single_imported_action(arm) -> bool:
    """One GR2 action on the armature — no GR2LAB_Cinematic NLA stack."""
    if bpy is None or arm is None:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    ad = arm.animation_data
    action = _armature_action(arm)
    _remove_cinematic_nla(arm)
    if action is None:
        action = _armature_action(arm)
    if action is None:
        return False
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])
    ad.use_nla = False
    ad.action = action
    return True


def _action_frame_span(action) -> Tuple[float, float]:
    if action is None:
        return 1.0, 250.0
    try:
        a0, a1 = action.frame_range
        start = float(max(1.0, a0))
        end = float(max(start, a1))
        return start, end
    except Exception:
        return 1.0, 250.0


def _copy_edit_action(action):
    """One independent Action for an edit duplicate (slow on long clips)."""
    if bpy is None or action is None:
        return None
    base = action.name or "GR2LAB"
    print(
        f"[GR2 LAB] edit copy: copying Action '{base}' (can be slow on long clips)…",
        flush=True,
    )
    act = action.copy()
    act.name = bpy.path.clean_name(f"{base}_Edit")
    n = 1
    while bpy.data.actions.get(act.name) is not None:
        act.name = bpy.path.clean_name(f"{base}_Edit.{n:03d}")
        n += 1
    return act


def apply_full_clip_nla(
    arm,
    action,
    *,
    scene_start: int = 1,
    duplicate_action: bool = True,
    track_name: str = "GR2LAB_Full",
) -> bool:
    """One NLA strip for the whole GR2 action (no Hold / Seg scene cuts)."""
    if bpy is None or arm is None or action is None:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    ad = arm.animation_data
    slot = _action_slot_for_arm(ad)
    if duplicate_action:
        act = _copy_edit_action(action)
        if act is None:
            return False
    else:
        act = action
    while ad.nla_tracks:
        ad.nla_tracks.remove(ad.nla_tracks[0])
    try:
        ad.use_tweak_mode = False
    except Exception:
        pass
    ad.use_nla = True
    a_start, a_end = _action_frame_span(act)
    scene_start = int(scene_start)
    scene_len = max(1, int(round(a_end - a_start)) + 1)
    scene_end = scene_start + scene_len - 1
    track = ad.nla_tracks.new()
    track.name = (track_name or "GR2LAB_Full")[:63]
    strip = track.strips.new((act.name or "full")[:63], float(scene_start), act)
    strip.action_frame_start = a_start
    strip.action_frame_end = a_end
    strip.frame_start = float(scene_start)
    strip.frame_end = float(scene_end)
    strip.scale = 1.0
    strip.blend_type = "REPLACE"
    try:
        strip.extrapolation = "HOLD_FORWARD"
    except Exception:
        pass
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
    _ensure_gr2lab_nla_audible(ad)
    try:
        arm["gr2lab_edit_copy"] = True
        arm["gr2lab_full_nla_scene_end"] = int(scene_end)
    except Exception:
        pass
    return True


def rewire_gr2lab_rig_full_nla(
    arm,
    *,
    scene_start: int = 1,
    duplicate_action: bool = True,
) -> int:
    """Replace scene-segment NLA on this rig family with one full clip strip per armature."""
    if bpy is None or arm is None:
        return 0
    rigs = [arm, *_tail_rigs_of(arm)]
    shared = None
    if duplicate_action:
        shared = _copy_edit_action(_armature_action(arm))
        if shared is None:
            return 0
    count = 0
    for rig in rigs:
        source = _armature_action(rig) or _armature_action(arm)
        if source is None:
            continue
        act = shared if shared is not None else source
        if apply_full_clip_nla(
            rig,
            act,
            scene_start=scene_start,
            duplicate_action=False,
        ):
            count += 1
    return count


def duplicate_gr2lab_edit_rig(
    context,
    arm,
    *,
    offset: Tuple[float, float, float] = (3.0, 0.0, 0.0),
    scene_start: int = 1,
    duplicate_action: bool = True,
) -> Optional[object]:
    """Duplicate rig + meshes (+ tail), offset aside, one full NLA strip per armature."""
    if bpy is None or context is None or arm is None:
        return None
    family = _rig_draw_objects(arm)
    if not family:
        return None
    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    prev_selected = list(context.selected_objects or [])
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass
    for obj in family:
        try:
            obj.select_set(True)
        except Exception:
            pass
    view_layer.objects.active = arm
    try:
        print(
            f"[GR2 LAB] edit copy: duplicate {len(family)} object(s)…",
            flush=True,
        )
        bpy.ops.object.duplicate()
    except Exception:
        for obj in prev_selected:
            try:
                obj.select_set(True)
            except Exception:
                pass
        if prev_active is not None:
            view_layer.objects.active = prev_active
        return None
    dupes = list(context.selected_objects or [])
    dup_arm = view_layer.objects.active
    if dup_arm is None or dup_arm.type != "ARMATURE":
        for obj in dupes:
            if obj.type == "ARMATURE" and not obj.get("gr2lab_tail_of"):
                dup_arm = obj
                break
    if dup_arm is None:
        return None
    for obj in dupes:
        tail_of = obj.get("gr2lab_tail_of")
        if tail_of == arm.name:
            try:
                obj["gr2lab_tail_of"] = dup_arm.name
            except Exception:
                pass
    delta = Vector(offset)
    for obj in dupes:
        try:
            obj.location += delta
        except Exception:
            pass
    rewire_gr2lab_rig_full_nla(
        dup_arm,
        scene_start=scene_start,
        duplicate_action=duplicate_action,
    )
    try:
        dup_arm["gr2lab_edit_copy"] = True
    except Exception:
        pass
    scene = context.scene
    if scene is not None:
        try:
            end = int(dup_arm.get("gr2lab_full_nla_scene_end") or scene.frame_end)
            if end > int(scene.frame_end):
                scene.frame_end = end
        except Exception:
            pass
    return dup_arm


def _binding_from_collection_props(
    collection,
    scene_path: Path,
    gr2_path: str,
) -> Optional[ClipBinding]:
    """When GR2→GUID index misses, use stamped phase + clip start on the collection."""
    if collection is None:
        return None
    try:
        phase = int(collection.get("gr2lab_phase_index", -1) or -1)
    except (TypeError, ValueError):
        phase = -1
    if phase < 0:
        return None
    tl_raw = collection.get("gr2lab_timeline_lsx")
    timeline = Path(str(tl_raw)) if tl_raw else None
    if timeline is None or not timeline.is_file():
        timeline = _timeline_beside_scene(scene_path)
    if timeline is None or not timeline.is_file():
        return None
    try:
        clip_start = float(collection.get("gr2lab_timeline_clip_start_sec", 0) or 0)
    except (TypeError, ValueError):
        clip_start = 0.0
    doc = _parse_timeline_bindings(timeline)
    actors = doc.get("actors") or {}
    rows = [
        row
        for row in doc.get("anims") or []
        if int(row.get("phase", -1)) == phase
        and abs(float(row.get("start") or 0) - clip_start) < 0.05
    ]
    if not rows:
        return None
    stem_l = Path(gr2_path).stem.lower()
    companion_clip = "plyr" not in stem_l

    def _row_is_primary(row: Dict) -> bool:
        meta = actors.get(row.get("actor") or "", {})
        atype = int(meta.get("type") or 0)
        if companion_clip:
            return atype == 8
        return atype == 2

    primary_rows = [row for row in rows if _row_is_primary(row)]
    hit = primary_rows[0] if primary_rows else rows[0]
    source = hit["source"]
    actor_meta = actors.get(hit.get("actor") or "", {})
    partner = _partner_anim_for_hit(
        hit,
        doc,
        source,
        prefer_companion_index=_partner_index_pref(gr2_path),
    )
    partner_meta = (
        actors.get(partner.get("actor") or "", {}) if partner is not None else {}
    )
    stage_guid = str(collection.get("gr2lab_stage_guid") or _ZERO_GUID)
    stage_name = str(collection.get("gr2lab_glasses_stage") or "")
    return ClipBinding(
        source_id=source,
        stage_guid=stage_guid,
        stage_name=stage_name,
        actor_type=int(actor_meta.get("type") or 0),
        actor_index=int(actor_meta.get("index") or 0),
        partner_stem="",
        partner_actor_type=int(partner_meta.get("type") or 0),
        partner_actor_index=int(partner_meta.get("index") or -1),
        pose=hit.get("pose"),
        partner_pose=(partner or {}).get("pose"),
        play_start=float(hit.get("play_start") or 0.0),
        partner_play_start=float((partner or {}).get("play_start") or 0.0),
        phase_index=phase,
        clip_start=float(hit.get("start") or clip_start),
        clip_end=float(hit.get("end") or clip_start),
    )


def scene_plan_from_collection(col) -> Optional[ScenePlan]:
    if bpy is None or col is None or not col.get("gr2lab_scene_empties"):
        return None
    sp = col.get("gr2lab_scene_file")
    sg = col.get("gr2lab_stage_guid")
    if not sp or not sg:
        return None
    name = col.name
    if name.endswith(" positions"):
        name = name[: -len(" positions")]
    return ScenePlan(
        scene_path=Path(str(sp)),
        stage=StageChoice(
            name=name,
            guid=str(sg),
            how=str(col.get("gr2lab_stage_how") or ""),
        ),
        empties=[],
        score=20,
        phase_index=int(col.get("gr2lab_phase_index", -1) or -1),
        clip_start_sec=float(col.get("gr2lab_timeline_clip_start_sec") or 0.0),
    )


def collection_for_cinematic_sync(context) -> Optional[object]:
    if bpy is None or context is None:
        return None
    active = context.active_object
    if active is not None:
        for col in active.users_collection:
            if col.get("gr2lab_scene_empties"):
                return col
    for col in bpy.data.collections:
        if col.get("gr2lab_scene_empties"):
            return col
    return None


def _gr2_stem_from_armature(arm) -> str:
    if arm is None:
        return ""
    for key in ("gr2lab_full_source_gr2", "gr2lab_source_gr2"):
        raw = arm.get(key)
        if raw:
            return Path(str(raw)).stem
    return ""


def player_gr2_stem_for_timeline_sync(context, collection) -> str:
    if bpy is None:
        return ""
    for name in ("Initiator_1", "Initiator"):
        empty = bpy.data.objects.get(name)
        if empty is None:
            continue
        for child in empty.children:
            if child.type == "ARMATURE":
                stem = _gr2_stem_from_armature(child)
                if stem:
                    return stem
    if collection is not None:
        for obj in collection.objects:
            if obj.type != "ARMATURE":
                continue
            if obj.get("gr2lab_is_player") or "plyr" in obj.name.lower():
                stem = _gr2_stem_from_armature(obj)
                if stem:
                    return stem
    wm = getattr(context, "window_manager", None)
    if wm is not None:
        raw = str(wm.get("gr2lab_last_source_gr2") or "").strip()
        if raw:
            return Path(raw).stem
    return ""


def run_cinematic_sync_from_ui(context) -> Tuple[bool, str]:
    """User-facing sync (Object menu). Returns ok, message."""
    col = collection_for_cinematic_sync(context)
    if col is None:
        return (
            False,
            "Select an object in a gr2lab scene collection "
            "(e.g. Sex Start - INT positions)",
        )
    plan = scene_plan_from_collection(col)
    if plan is None:
        return False, f"Collection {col.name!r} is missing gr2lab scene metadata"
    gr2 = player_gr2_stem_for_timeline_sync(context, col)
    if not gr2:
        return (
            False,
            "No player GR2 found — import PLYR Sex Start (or seat partner) first",
        )
    y_up = True
    note = apply_cinematic_timeline_sync(
        context,
        col,
        gr2,
        plan,
        y_up_display=y_up,
        timeline_gr2_path=gr2,
    )
    if not note:
        return (
            False,
            "Sync did nothing — check gr2lab_timeline_lsx and gr2lab_phase_index on the collection",
        )
    return True, note


def apply_cinematic_timeline_sync(
    context,
    collection,
    gr2_path: str,
    plan: ScenePlan,
    *,
    y_up_display: bool = True,
    timeline_gr2_path: str = "",
) -> str:
    """Keyframe scene slots on timeline restages (romance / multi-row clips)."""
    if bpy is None or collection is None or plan is None:
        return ""
    prefer_phase = int(plan.phase_index) if plan.phase_index >= 0 else -1
    if prefer_phase < 0:
        try:
            prefer_phase = int(collection.get("gr2lab_phase_index", -1) or -1)
        except (TypeError, ValueError):
            prefer_phase = -1
    binding = plan.clip_binding or _binding_for_scene(
        timeline_gr2_path or gr2_path,
        plan.scene_path,
        prefer_phase=prefer_phase,
    )
    if binding is None:
        binding = _binding_from_collection_props(
            collection, plan.scene_path, timeline_gr2_path or gr2_path
        )
    if binding is None:
        return ""
    # Same Glasses coordinates as the other clips. The old NonSex Y mirror
    # put the camera on the opposite side of the stage, so the shot was flipped.
    mirror_player = False
    swap_location = mirror_player
    segments, band = collect_paired_slot_segments(
        timeline_gr2_path or gr2_path, plan.scene_path, binding
    )
    if len(segments) < 2:
        return ""
    anchor = bpy.data.objects.get("Additional_1")
    player_empty = bpy.data.objects.get("Initiator_1")
    if anchor is None or player_empty is None:
        for obj in collection.objects:
            if obj.type != "EMPTY":
                continue
            if obj.get("gr2lab_is_player"):
                player_empty = obj
            elif int(obj.get("gr2lab_speaker_id", 999)) == 0:
                anchor = obj
    if anchor is None or player_empty is None:
        return ""
    first_partner = segments[0].get("partner_pose")
    if not first_partner:
        return ""
    base_partner_pos = first_partner[0]
    scene = context.scene
    fps = float(scene.render.fps or 30)
    first_panel = float(segments[0]["panel_start"])
    phase_panel_max = 72.0
    sync_segments: List[Dict] = []
    for seg in segments:
        if float(seg["panel_start"]) - first_panel > phase_panel_max:
            break
        sync_segments.append(seg)
    if len(sync_segments) < 2:
        return ""
    _LIVE_CINEMATIC_SLOTS.pop(collection.name, None)
    collection["gr2lab_cinematic_live"] = False
    _clear_empty_slot_action(anchor)
    _clear_empty_slot_action(player_empty)
    first_body_frame = panel_seconds_to_timeline_frame(
        float(segments[0]["panel_start"]), fps
    )
    last_scene_frame = panel_seconds_to_timeline_frame(
        float(sync_segments[-1]["panel_end"]), fps
    )
    apply_glasses_timeline_range(
        scene,
        band_sec=float(band),
        fps=fps,
        panel_end=float(sync_segments[-1]["panel_end"]),
        playhead_panel=0.0,
    )
    scene["gr2lab_first_body_frame"] = int(first_body_frame)
    _apply_slot_pair_pose(
        anchor,
        player_empty,
        sync_segments[0],
        base_partner_pos=base_partner_pos,
        y_up_display=y_up_display,
        swap_location=swap_location,
    )
    _keyframe_empty_hold(anchor, player_empty, 0)
    keyed = 1
    last_key_frame = 0
    slot_frames = [0]
    first_slot_frame = panel_seconds_to_timeline_frame(
        float(sync_segments[0]["panel_start"]), fps
    )
    slot_frames.append(first_slot_frame)
    _keyframe_slot_pair(
        anchor,
        player_empty,
        sync_segments[0],
        first_slot_frame,
        base_partner_pos=base_partner_pos,
        y_up_display=y_up_display,
        swap_location=swap_location,
    )
    keyed += 1
    last_key_frame = max(last_key_frame, first_slot_frame)
    for seg_index, seg in enumerate(sync_segments):
        if seg_index == 0:
            continue
        frame = panel_seconds_to_timeline_frame(float(seg["panel_start"]), fps)
        # Offset types other than SelfStart/SceneStart have no staging pose.
        # Refitting those frames treats the held empty as the actor world
        # position and teleports the body (player X key at frame 716).
        if seg.get("player_pose") or seg.get("partner_pose"):
            slot_frames.append(frame)
        _keyframe_slot_pair(
            anchor,
            player_empty,
            seg,
            frame,
            base_partner_pos=base_partner_pos,
            y_up_display=y_up_display,
            swap_location=swap_location,
        )
        keyed += 1
        last_key_frame = max(last_key_frame, frame)
    if keyed > 0:
        _constant_keyframes(anchor)
        _constant_keyframes(player_empty)
    register_prebody_nla_handler()
    single_action_rigs = 0
    player_arm, partner_arm = _cinematic_rig_pair(player_empty, anchor)
    for arm, is_player in ((player_arm, True), (partner_arm, False)):
        if arm is None:
            continue
        for rig in (arm, *_tail_rigs_of(arm)):
            if _apply_glasses_segment_body_nla(
                rig,
                sync_segments,
                fps=fps,
                first_body_frame=first_body_frame,
                is_player=is_player,
            ):
                single_action_rigs += 1
    gr2lab_prebody_nla_mute_handler(scene)
    cam_note = ""
    timeline = _timeline_beside_scene(plan.scene_path)
    collection["gr2lab_cinematic_sync"] = True
    anchor["gr2lab_glasses_euler_direct"] = True
    player_empty["gr2lab_glasses_euler_direct"] = True
    if partner_arm is not None:
        _seat_arm(partner_arm, anchor, y_up_display)
    else:
        _apply_glasses_delta_rx90(anchor)
        _remove_glasses_rx_child(anchor)
    if player_arm is not None:
        _seat_arm(player_arm, player_empty, y_up_display)
    else:
        _apply_glasses_delta_rx90(player_empty)
        _remove_glasses_rx_child(player_empty)
    _refit_slot_locations_excluding_root(
        scene,
        slot_frames,
        ((anchor, partner_arm), (player_empty, player_arm)),
    )
    context.view_layer.update()
    for arm in (partner_arm, player_arm):
        if arm is not None:
            _ensure_diag_camera_socket(context, arm)
    context.view_layer.update()
    if timeline is not None and plan.phase_index >= 0:
        cam_note = ensure_scene_timeline_camera(
            context,
            scene_path=plan.scene_path,
            timeline_path=timeline,
            phase=int(plan.phase_index),
            band_sec=float(band),
            y_up_display=y_up_display,
            player_source=str(binding.source_id or ""),
            player_empty_name=player_empty.name,
            partner_empty_name=anchor.name,
        )
    hide_note = ""
    timeline = _timeline_beside_scene(plan.scene_path)
    if timeline is not None and player_arm is not None and binding is not None:
        doc = _parse_timeline_bindings(timeline)
        actor_uuid = _source_actor_uuid(
            doc, binding.source_id, int(binding.phase_index)
        )
        hide_keys = _show_visual_hide_keys(
            timeline,
            int(binding.phase_index),
            actor_uuid,
            fps,
            float(band),
        )
        hide_note = _apply_show_visual_hide(
            scene,
            [
                obj
                for obj in _rig_draw_objects(player_arm)
                if getattr(obj, "type", None) == "MESH"
            ],
            hide_keys,
        )
    collection["gr2lab_panel_band_sec"] = float(band)
    collection["gr2lab_slot_segment_count"] = int(keyed)
    collection["gr2lab_panel_frame1_sec"] = float(first_panel)
    collection["gr2lab_glasses_first_body_frame"] = panel_seconds_to_timeline_frame(
        float(segments[0]["panel_start"]), fps
    )
    context.view_layer.update()
    if keyed == 0 and single_action_rigs == 0 and not cam_note:
        return ""
    suffix = f" · {cam_note}" if cam_note else ""
    if hide_note:
        suffix += f" · {hide_note}"
    return (
        f"cinematic sync: {keyed} slot keys @ Glasses frames, "
        f"{single_action_rigs} rigs (hold 0–{first_body_frame - 1}, "
        f"play from {first_body_frame}){suffix}"
    )


def timeline_play_start_seconds(gr2_path: str) -> float:
    """Seconds into the GR2 when the dialog clip starts (AnimationPlayStartOffset)."""
    found = find_scene_file(gr2_path)
    if found is None:
        return 0.0
    binding = resolve_clip_binding(gr2_path, found[0])
    if binding is None:
        return 0.0
    return max(float(binding.play_start or 0.0), float(binding.partner_play_start or 0.0))


def apply_timeline_play_start(context, gr2_path: str) -> str:
    """Scrub to the frame the game timeline uses at clip start (both bodies stay in sync)."""
    if bpy is None:
        return ""
    offset = timeline_play_start_seconds(gr2_path)
    if offset <= 1e-6:
        return ""
    scene = context.scene
    fps = float(scene.render.fps or 30)
    # Match import_gr2lab_from_doc key framing: frame 1 = t=0, frame N = t=(N-1)/fps.
    frame = max(1, int(round(1.0 + float(offset) * fps)))
    end = int(scene.frame_end or frame)
    if frame > end:
        scene.frame_end = frame
    scene.frame_set(frame)
    scene["gr2lab_timeline_play_start_sec"] = float(offset)
    scene["gr2lab_timeline_play_start_frame"] = int(frame)
    found = find_scene_file(gr2_path)
    phase_note = ""
    if found is not None:
        try:
            root = ET.parse(found[0]).getroot()
        except (ET.ParseError, OSError):
            root = None
        binding = resolve_clip_binding(gr2_path, found[0], root)
        if binding is not None and binding.phase_index >= 0:
            scene["gr2lab_phase_index"] = int(binding.phase_index)
            scene["gr2lab_timeline_clip_start_sec"] = float(binding.clip_start)
            phase_note = f" · phase {binding.phase_index} @ {binding.clip_start:.2f}s"
    for col in bpy.data.collections:
        if not col.get("gr2lab_scene_empties"):
            continue
        col["gr2lab_play_start_sec"] = float(offset)
        col["gr2lab_play_start_frame"] = int(frame)
    if context is not None:
        context.view_layer.update()
    return f"Timeline play start {offset:.3f}s → frame {frame}{phase_note}"


def cleanup_stale_tail_rigs() -> int:
    """Drop tail armatures whose body was re-imported under a new name."""
    if bpy is None:
        return 0
    bodies = {
        o.name
        for o in bpy.data.objects
        if getattr(o, "type", None) == "ARMATURE" and not o.get("gr2lab_tail")
    }
    removed = 0
    for obj in list(bpy.data.objects):
        if getattr(obj, "type", None) != "ARMATURE" or not obj.get("gr2lab_tail"):
            continue
        owner = str(obj.get("gr2lab_tail_of") or "")
        if owner and owner not in bodies:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def cleanup_orphan_body_meshes() -> int:
    """Remove leftover kit meshes at the origin after a re-import under a new rig name."""
    if bpy is None:
        return 0
    armed_stems = set()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.parent is None or obj.parent.type != "ARMATURE":
            continue
        armed_stems.add(obj.name.split(".")[0])
    kit_markers = ("Body", "CLT_", "NKD_", "AKELKA_", "Ears_", "Head_", "Horn", "Tail_A_Mesh")
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or obj.parent is not None:
            continue
        stem = obj.name.split(".")[0]
        if not any(m in obj.name for m in kit_markers):
            continue
        # Drop unparented kit pieces when a sibling copy is parented to a GR2LAB armature.
        if stem in armed_stems or any(
            s.startswith(stem) or stem.startswith(s.split(".")[0]) for s in armed_stems
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
            continue
        if obj.name.startswith("GR2LAB_"):
            continue
        # Orphan kit mesh on the floor (no armature parent) after a partial re-import.
        if any(m in obj.name for m in ("NKD_", "CLT_")) and obj.matrix_world.translation.length < 0.25:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def hide_dialog_lips_in_viewport() -> int:
    """Lips gizmo meshes stretch badly on cinematic SCENE clips; hide in the viewport."""
    if bpy is None:
        return 0
    hidden = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not str(obj.name).startswith("Lips_"):
            continue
        if not obj.hide_viewport:
            obj.hide_viewport = True
            hidden += 1
    return hidden


def _mirror_keyed_camera_y(obj, center_y: float, cut_frames: set) -> None:
    """Same Y mirror the NonSex player slot got. The timeline camera stays in
    the pre-mirror space, so a close-up lands on the legs after the body moves.
    """
    if bpy is None or obj is None:
        return
    fcurves = _object_action_fcurves(obj)
    if not fcurves:
        return
    frames = set()
    for fc in fcurves:
        if fc.data_path != "location":
            continue
        for kp in fc.keyframe_points:
            frames.add(int(round(kp.co[0])))
    if not frames:
        return
    scene = bpy.context.scene
    original = int(scene.frame_current)
    flip = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))
    pre = Matrix.Translation(Vector((0.0, -float(center_y), 0.0)))
    post = Matrix.Translation(Vector((0.0, float(center_y), 0.0)))
    samples = []
    for frame in sorted(frames):
        scene.frame_set(int(frame))
        samples.append((int(frame), obj.matrix_world.copy()))
    for frame, world in samples:
        mirrored = post @ flip @ pre @ world
        # A Y mirror has a negative determinant. Writing that straight into
        # matrix_world stores the reflection as negative scale and turns the
        # camera around. Keep the mirrored look/up as a normal rotation.
        look = (-mirrored.to_3x3().col[2])
        up = mirrored.to_3x3().col[1]
        if look.length < 1e-8:
            continue
        look.normalize()
        if up.length < 1e-8:
            up = Vector((0.0, 0.0, 1.0))
        up.normalize()
        back = -look
        x_axis = up.cross(back)
        if x_axis.length < 1e-6:
            x_axis = Vector((1.0, 0.0, 0.0)).cross(back)
        x_axis.normalize()
        y_axis = back.cross(x_axis).normalized()
        rotation = Matrix((x_axis, y_axis, back)).transposed().to_quaternion()
        obj.matrix_world = Matrix.LocRotScale(
            mirrored.translation, rotation, Vector((1.0, 1.0, 1.0))
        )
        obj.keyframe_insert(data_path="location", frame=int(frame))
        obj.keyframe_insert(data_path="rotation_quaternion", frame=int(frame))
    _set_shot_curve_interpolation(obj, cut_frames)
    scene.frame_set(original)


def _ensure_imported_scene_camera(
    context,
    collection,
    plan: ScenePlan,
    binding: Optional[ClipBinding],
    *,
    y_up_display: bool,
    mirror_center_y: Optional[float] = None,
) -> str:
    """Keyframe GR2LAB_Scene_Camera from the phase the GR2 just resolved.

    Full cinematic sync does this for sex clips. NonSex skips that sync
    because its slot mirror is separate, and the timeline camera was skipped
    with it. The shots still live in the scene and timeline files.
    """
    if bpy is None or context is None or collection is None or plan is None:
        return ""
    if binding is None or int(plan.phase_index) < 0:
        return ""
    timeline = _timeline_beside_scene(plan.scene_path)
    if timeline is None:
        cached = collection.get("gr2lab_timeline_lsx")
        if cached:
            timeline = Path(str(cached))
    if timeline is None or not Path(timeline).is_file():
        return ""
    timeline = Path(timeline)
    phase = int(plan.phase_index)
    player_name = "Initiator_1"
    partner_name = "Additional_1"
    for obj in collection.objects:
        if getattr(obj, "type", None) != "EMPTY":
            continue
        if obj.get("gr2lab_is_player"):
            player_name = obj.name
        elif int(obj.get("gr2lab_speaker_id", 999) or 999) == 0:
            partner_name = obj.name
        for child in obj.children:
            if getattr(child, "type", None) == "ARMATURE":
                _ensure_diag_camera_socket(context, child)
    source = str(binding.source_id or "")
    player_empty = bpy.data.objects.get(player_name)
    if player_empty is not None:
        for child in player_empty.children:
            if getattr(child, "type", None) != "ARMATURE":
                continue
            player_path = ""
            for key in ("gr2lab_full_source_gr2", "gr2lab_source_gr2"):
                raw = child.get(key)
                if raw:
                    player_path = str(raw)
                    break
            if not player_path:
                continue
            player_binding = _binding_for_scene(
                player_path, plan.scene_path, prefer_phase=phase
            )
            if player_binding is not None and player_binding.source_id:
                source = str(player_binding.source_id)
            break
    band = _phase_panel_band_start(timeline, phase)
    note = ensure_scene_timeline_camera(
        context,
        scene_path=plan.scene_path,
        timeline_path=timeline,
        phase=phase,
        band_sec=float(band),
        y_up_display=y_up_display,
        player_source=source,
        player_empty_name=player_name,
        partner_empty_name=partner_name,
    )
    if note:
        collection["gr2lab_panel_band_sec"] = float(band)
        collection["gr2lab_timeline_lsx"] = str(timeline)
        if mirror_center_y is not None:
            shots = _parse_timeline_tl_shots(
                timeline, phase=phase, band_sec=float(band), fps=float(
                    context.scene.render.fps or 30
                )
            )
            cam = bpy.data.objects.get("GR2LAB_Scene_Camera")
            _mirror_keyed_camera_y(
                cam,
                float(mirror_center_y),
                {int(shot["frame"]) for shot in shots},
            )
    return note


def place_for_import(
    context,
    gr2_path: str,
    arm=None,
    y_up_display: bool = True,
    parent_rig: bool = True,
    center_parent: bool = False,
    parent_stage_empties: bool = False,
    plan: Optional[ScenePlan] = None,
    defer_cinematic: bool = False,
    skip_cleanup: bool = False,
) -> str:
    """Create start-stage empties for the scene this GR2 name points at."""
    if bpy is None:
        return ""
    if not skip_cleanup:
        cleanup_stale_tail_rigs()
        cleanup_orphan_body_meshes()
    if plan is None:
        plan = plan_scene_empties(gr2_path)
    if plan is None or not plan.empties:
        return ""
    binding_warn = ""
    binding = plan.clip_binding or _binding_for_scene(gr2_path, plan.scene_path)
    if binding is None:
        binding_warn = (
            " · WARNING: no timeline binding — empties use filename guess only"
        )
    col_name = plan.stage.name
    col = bpy.data.collections.get(col_name)
    if col is not None and not col.get("gr2lab_scene_empties"):
        col = None
        col_name = f"{plan.stage.name} positions"
        col = bpy.data.collections.get(col_name)
        if col is not None and not col.get("gr2lab_scene_empties"):
            col = None
    if col is None:
        col = bpy.data.collections.new(col_name)
        col["gr2lab_scene_empties"] = True
        context.scene.collection.children.link(col)
    col["gr2lab_scene_file"] = str(plan.scene_path)
    col["gr2lab_stage_guid"] = plan.stage.guid
    col["gr2lab_stage_how"] = plan.stage.how
    if plan.phase_index >= 0:
        col["gr2lab_phase_index"] = int(plan.phase_index)
    if plan.clip_start_sec > 0:
        col["gr2lab_timeline_clip_start_sec"] = float(plan.clip_start_sec)
    preview = glasses_preview_from_plan(plan, gr2_path)
    _stamp_glasses_preview_on_collection(col, preview)
    sidecar = write_glasses_mcp_sidecar(preview)
    if bpy.context.scene is not None:
        bpy.context.scene["gr2lab_glasses_preview"] = preview.get("dialogName", "")
        bpy.context.scene["gr2lab_glasses_time_sec"] = float(
            preview.get("timeSeconds") or 0.0
        )

    # Update in place so a second character drop does not delete the empties
    # the first rig is already parented to.
    # Glasses coordinates, including NonSex. Mirroring the player across the
    # anchor faces him away from the timeline camera, and mirroring the camera
    # to follow makes the whole shot a left-right flip.
    nonsex = _stem_is_nonsex(Path(gr2_path).stem)
    mirror_player = y_up_display and binding is None and not nonsex
    made: Dict[str, object] = {}
    for item in plan.empties:
        empty = bpy.data.objects.get(item.name)
        if empty is None or empty.type != "EMPTY":
            empty = bpy.data.objects.new(item.name, None)
            col.objects.link(empty)
        elif empty.name not in col.objects:
            col.objects.link(empty)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.22
        if empty.parent is not None:
            empty.parent = None
        if y_up_display and not mirror_player:
            _write_glasses_slot_channels(
                empty,
                item.position,
                item.quat,
                y_up_display,
                swap_location=mirror_player,
            )
            _stamp_glasses_inspector_euler(empty, item.quat)
        else:
            loc, rot = larian_to_blender(
                item.position,
                item.quat,
                y_up_display,
                swap_location=mirror_player,
            )
            _apply_empty_world_matrix(
                empty,
                Matrix.LocRotScale(
                    Vector(loc), Quaternion(rot), Vector((1.0, 1.0, 1.0))
                ),
            )
        empty.hide_viewport = False
        empty.hide_render = False
        empty["gr2lab_scene_stage"] = plan.stage.name
        empty["gr2lab_speaker_id"] = int(item.speaker_id)
        empty["gr2lab_is_player"] = bool(item.is_player)
        empty["gr2lab_template_id"] = item.template_id
        made[item.name] = empty

    context.view_layer.update()
    anchor = min(made.values(), key=lambda obj: int(obj.get("gr2lab_speaker_id", 999)))
    if anchor.parent is not None:
        anchor.parent = None
    if binding is not None:
        used_slots = {plan.rig_empty_name}
        partner_slot = _partner_empty_name(plan.empties, binding)
        if partner_slot:
            used_slots.add(partner_slot)
        for name, empty in made.items():
            if name not in used_slots:
                empty.hide_viewport = True
                empty.hide_render = True
    if parent_stage_empties:
        for empty in made.values():
            if empty == anchor or empty.parent == anchor:
                continue
            _parent_keep_world(empty, anchor)
    context.view_layer.update()

    rig_name = plan.rig_empty_name if parent_rig else ""
    rig_empty = made.get(rig_name)
    just_seated = False
    if arm is not None and rig_empty is not None:
        if arm.parent is not None and arm.parent != rig_empty:
            world = arm.matrix_world.copy()
            for tail in _tail_rigs_of(arm):
                tail_world = tail.matrix_world.copy()
                tail.parent = None
                tail.matrix_world = tail_world
            arm.parent = None
            arm.matrix_world = world
        if arm.parent is None or arm.parent != rig_empty:
            _seat_arm(arm, rig_empty, y_up_display)
            just_seated = True
    context.view_layer.update()
    if (
        nonsex
        and just_seated
        and arm is not None
        and rig_empty is not None
        and not rig_empty.get("gr2lab_is_player")
        and arm.parent == rig_empty
    ):
        _yaw_world_z(arm)
        for tail in _tail_rigs_of(arm):
            _yaw_world_z(tail)
        context.view_layer.update()
    # A missing timeline binding still mirrors the player slot. Sex and NonSex
    # with a binding keep the Glasses placement. The NonSex partner yaw above
    # is what puts her head on the camera's up side.
    if mirror_player:
        _mirror_player_empties(made.values(), anchor)
        context.view_layer.update()
        if (
            arm is not None
            and rig_empty is not None
            and rig_empty.get("gr2lab_is_player")
            and arm.parent == rig_empty
        ):
            _mirror_y_about_origin(arm)
            for tail in _tail_rigs_of(arm):
                _mirror_y_about_origin(tail)

    parent_centered = False
    if center_parent:
        context.view_layer.update()
        parent_centered = _center_root_at_origin(anchor)
        context.view_layer.update()

    sync_note = ""
    # NonSex keeps its imported clip. Full sync rewrites the body NLA, which
    # this path is not doing. Whole scene defers that sex sync until the
    # second rig is seated. NonSex never runs it, so the timeline camera is
    # created here either way.
    if (
        binding is not None
        and not mirror_player
        and not nonsex
        and not defer_cinematic
    ):
        sync_note = apply_cinematic_timeline_sync(
            context,
            col,
            gr2_path,
            plan,
            y_up_display=y_up_display,
            timeline_gr2_path=gr2_path,
        )
    cam_note = ""
    if binding is not None and "scene camera:" not in sync_note and (not defer_cinematic or nonsex):
        cam_note = _ensure_imported_scene_camera(
            context,
            col,
            plan,
            binding,
            y_up_display=y_up_display,
            mirror_center_y=(
                float(anchor.matrix_world.translation.y) if mirror_player else None
            ),
        )

    names = ", ".join(item.name for item in plan.empties)
    parented = ""
    if arm is not None and rig_empty is not None and arm.parent == rig_empty:
        parented = f" · rig on {rig_empty.name}"
    centered = " · top parent at center" if center_parent and parent_centered else ""
    phase_note = ""
    if plan.phase_index >= 0:
        phase_note = f" · phase {plan.phase_index}"
        if plan.clip_start_sec > 0:
            phase_note += f" @ {plan.clip_start_sec:.2f}s"
    glasses_note = ""
    if sidecar is not None:
        glasses_note = (
            f" · Glasses: {preview.get('dialogName')} @ "
            f"{preview.get('timeSeconds')}s ({preview.get('sceneStage')})"
        )
    sync_suffix = f" · {sync_note}" if sync_note else ""
    if cam_note:
        sync_suffix += f" · {cam_note}"
    return (
        f"Scene empties: {plan.stage.name} · {len(plan.empties)} characters "
        f"({names}) · {dialog_stem(plan.scene_path)}{phase_note}{parented}{centered}"
        f"{glasses_note}{sync_suffix}{binding_warn}"
    )


def _release_armatures_from_empty(empty) -> None:
    if bpy is None or empty is None:
        return
    for child in list(empty.children):
        if child.type != "ARMATURE" and not child.get("gr2lab_tail_of"):
            continue
        world = child.matrix_world.copy()
        child.parent = None
        child.matrix_world = world


def remove_transient_scene_empties(context, gr2_path: str) -> int:
    """Keep world placement after seating rigs, then delete temporary stage empties."""
    if bpy is None:
        return 0
    plan = plan_scene_empties(gr2_path)
    if plan is None or not plan.empties:
        return 0
    empties = []
    for item in plan.empties:
        obj = bpy.data.objects.get(item.name)
        if obj is not None and obj.type == "EMPTY":
            empties.append(obj)
    for empty in empties:
        _release_armatures_from_empty(empty)
    if context is not None:
        context.view_layer.update()
    removed = 0
    for empty in empties:
        try:
            bpy.data.objects.remove(empty, do_unlink=True)
            removed += 1
        except RuntimeError:
            pass
    stage = plan.stage.name
    for col_name in (stage, f"{stage} positions"):
        col = bpy.data.collections.get(col_name)
        if col is None or not col.get("gr2lab_scene_empties"):
            continue
        if len(col.objects) == 0 and len(col.children) == 0:
            try:
                bpy.data.collections.remove(col)
            except RuntimeError:
                pass
    if context is not None:
        context.view_layer.update()
    return removed
