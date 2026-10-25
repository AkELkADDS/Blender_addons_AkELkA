"""Rig family taxonomy for BG3 animation GR2 files (body-base matching + validation)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_RIG_TAIL = re.compile(
    r"^(?P<prefix>[A-Za-z0-9_]+)_Tail_Rig_", re.I
)
_RIG_HEAD = re.compile(
    r"^(?P<prefix>[A-Za-z0-9_]+)_Head(?:_[A-Za-z0-9_]+)?_Rig_", re.I
)
_RIG_STD = re.compile(r"^(?P<prefix>[A-Za-z0-9_]+)_Rig_", re.I)
_PROP_HINTS = (
    "CINE_Generic_Object",
    "Dagger",
    "SCENE_",
    "Elder_Brain",
    "Generic_Object",
)
_CREATURE_HINTS = (
    "DILOPH",
    "DRAGON",
    "DOG_",
    "CAT_",
    "BEAR_",
    "HPHANT",
    "CAMBION",
    "CAMBION_F_Wing",
)

_RIG_MAP_PATH = Path(__file__).resolve().parent.parent / "Body_Base" / "rig_map.json"
_rig_map_cache: Optional[Dict[str, Any]] = None


def load_rig_map() -> Dict[str, Any]:
    global _rig_map_cache
    if _rig_map_cache is not None:
        return _rig_map_cache
    if _RIG_MAP_PATH.is_file():
        try:
            _rig_map_cache = json.loads(_RIG_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            _rig_map_cache = {}
    else:
        _rig_map_cache = {}
    return _rig_map_cache


def _basename(source: str) -> str:
    return Path((source or "").replace("\\", "/")).name


def classify_rig(source_gr2_name: str) -> Dict[str, Any]:
    """Classify a source animation filename for bind + validation tiers."""
    name = _basename(source_gr2_name)
    stem = Path(name).stem
    out: Dict[str, Any] = {
        "source": name,
        "stem": stem,
        "rig_family": stem.split("_")[0] if stem else "",
        "rig_kind": "unknown",
        "is_head": False,
        "is_tail": False,
        "is_creature": False,
        "is_prop": False,
        "bind_optional": False,
        "min_bones": 40,
        "expected_base": None,
    }
    if not stem:
        out["bind_optional"] = True
        out["min_bones"] = 1
        return out

    rig_map = load_rig_map()
    bind_optional_families = set(rig_map.get("bind_optional_families") or [])
    bind_optional_prefixes = rig_map.get("bind_optional_prefixes") or []

    m_head = _RIG_HEAD.match(stem)
    m_tail = _RIG_TAIL.match(stem)
    m_std = _RIG_STD.match(stem)

    if m_head:
        prefix = m_head.group("prefix")
        out.update(
            {
                "rig_family": prefix,
                "rig_kind": "head",
                "is_head": True,
                "min_bones": 8,
                "expected_base": f"{prefix}_Base.GR2",
            }
        )
    elif m_tail:
        prefix = m_tail.group("prefix")
        out.update(
            {
                "rig_family": prefix + "_Tail",
                "rig_kind": "tail",
                "is_tail": True,
                "min_bones": 15,
                "expected_base": f"{prefix}_Tail_Base.GR2",
            }
        )
    elif m_std:
        prefix = m_std.group("prefix")
        out.update(
            {
                "rig_family": prefix,
                "rig_kind": "body",
                "expected_base": f"{prefix}_Base.GR2",
            }
        )
    else:
        parts = stem.split("_")
        if len(parts) >= 2:
            out["rig_family"] = "_".join(parts[:2])
        out["rig_kind"] = "other"

    fam = out["rig_family"]
    upper = stem.upper()
    if any(h in upper for h in _CREATURE_HINTS) or fam in bind_optional_families:
        out["is_creature"] = True
        out["min_bones"] = min(out["min_bones"], 12)
    if any(upper.startswith(p) or p in upper for p in _PROP_HINTS):
        out["is_prop"] = True
        out["bind_optional"] = True
        out["min_bones"] = 1
    if any(upper.startswith(p) for p in bind_optional_prefixes):
        out["bind_optional"] = True

    overrides = rig_map.get("overrides") or {}
    if fam in overrides:
        ob = overrides[fam]
        if ob:
            out["expected_base"] = ob
        else:
            # Null override means "no required bind", not "never use a base".
            # Keep the conventional "{family}_Base.GR2" name from the rig parse.
            # Matching accepts it only when that file is actually in Body_Base.
            out["bind_optional"] = True
    if fam in bind_optional_families:
        out["bind_optional"] = True

    head_families = rig_map.get("head_families") or []
    if fam in head_families:
        out["is_head"] = True
        out["min_bones"] = min(out["min_bones"], 8)

    return out


def _listed_base_name(bases: List[Dict[str, Any]], filename: str) -> Optional[str]:
    """Exact Body_Base filename, or None when that file is not in the list."""
    want = Path(filename or "").name.lower()
    if not want:
        return None
    for b in bases:
        name = str(b.get("name") or "")
        if name.lower() == want:
            return name
    return None


def _base_prefix(filename: str) -> str:
    stem = Path(filename or "").stem
    if stem.lower().endswith("_base"):
        stem = stem[: -len("_base")]
    return stem


def _family_base_on_disk(
    fam: str,
    expected: Optional[str],
    bases: List[Dict[str, Any]],
) -> Optional[str]:
    """This family's body file only. A different race (DGB, HUM, …) is not a match."""
    if expected:
        found = _listed_base_name(bases, expected)
        if found:
            return found
    if not fam:
        return None
    found = _listed_base_name(bases, f"{fam}_Base.GR2")
    if found:
        return found
    fam_l = fam.lower()
    hits: List[str] = []
    for b in bases:
        name = str(b.get("name") or "")
        if name and _base_prefix(name).lower() == fam_l:
            hits.append(name)
    if len(hits) == 1:
        return hits[0]
    return None


def match_body_base_detail(source_gr2_name: str) -> Dict[str, Any]:
    """Match Body_Base with rig-aware scoring and confidence."""
    from bind_pose import list_body_bases

    info = classify_rig(source_gr2_name)
    stem = info["stem"]
    rig_map = load_rig_map()
    overrides = rig_map.get("overrides") or {}
    fam = info["rig_family"]

    if fam in overrides:
        ob = overrides[fam]
        bases = list_body_bases()
        if ob:
            found = _listed_base_name(bases, str(ob))
            if found:
                return {
                    **info,
                    "matched": found,
                    "confidence": "exact",
                    "reason": "rig_map override",
                    "bind_optional": False,
                    "expected_base": found,
                }
            # Named override is not on disk — do not invent that base.
            return {
                **info,
                "matched": None,
                "confidence": "none",
                "reason": "rig_map override (bind optional)",
            }
        found = _family_base_on_disk(str(fam or ""), info.get("expected_base"), bases)
        if found:
            return {
                **info,
                "matched": found,
                "confidence": "exact",
                "reason": "family body base",
                "bind_optional": False,
                "expected_base": found,
            }
        return {
            **info,
            "matched": None,
            "confidence": "none",
            "reason": "rig_map override (bind optional)",
        }

    bases = list_body_bases()
    if not bases or not stem:
        return {**info, "matched": None, "confidence": "none", "reason": "no bases"}

    candidates: List[tuple[int, int, str]] = []
    sl = stem.lower()
    for b in bases:
        bname = b["name"]
        bstem = Path(bname).stem
        prefix = bstem[:-5] if bstem.lower().endswith("_base") else bstem
        pl = prefix.lower()
        if sl == pl or sl.startswith(pl + "_"):
            score = len(pl)
            bonus = 0
            if info["is_tail"] and "_tail" in pl:
                bonus += 100
            elif info["is_tail"] and "_tail" not in pl:
                bonus -= 50
            if info["is_head"] and "_head" in pl:
                bonus += 80
            elif info["is_head"] and "_head" not in pl:
                bonus -= 20
            if info["rig_kind"] == "body" and "_tail" in pl:
                bonus -= 40
            if info["rig_kind"] == "body" and "_head" in pl:
                bonus -= 40
            candidates.append((score + bonus, score, bname))

    if not candidates:
        return {**info, "matched": None, "confidence": "none", "reason": "no prefix match"}

    candidates.sort(key=lambda x: (-x[0], -x[1]))
    best_score, raw_len, best = candidates[0]
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        conf = "fuzzy"
    elif raw_len >= 5:
        conf = "exact"
    else:
        conf = "fuzzy"
    return {**info, "matched": best, "confidence": conf, "reason": "prefix match"}
