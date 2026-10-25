"""In-process decode / densify / pack-encode - no HTTP server."""

from __future__ import annotations

import json
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Flat codec modules on sys.path
from . import codec as _codec_pkg  # noqa: F401

from paths import (  # type: ignore
    BODY_BASE,
    configure_layout,
    ensure_layout,
    invalidate_folder_tree_cache,
    is_watch_run,
    list_decode_runs,
    list_dir_files,
    make_run_dirs,
    read_dump_source_path,
    resolve_dump_dir,
    resolve_input_gr2,
)

# Re-export for operators
__all__ = [
    "apply_prefs",
    "decode_gr2_local",
    "decode_run_ids",
    "dump_still_exists",
    "encode_from_dump",
    "import_gr2_pipeline_local",
    "list_body_bases_local",
    "pack_encode_local",
    "pack_gr2lab_doc",
    "preview_stick_poses",
    "resolve_dump_dir",
]
from gr2_codec import (  # type: ignore
    decode_sections,
    encode_after_edit,
    encode_passthrough,
    encode_size_breakdown_from_headers,
    first_diff,
    load_gr2,
    mark_section_edited,
    sha256_bytes,
)
from gr2_format import parse_gr2  # type: ignore
from gr2_dump import dump_gr2  # type: ignore
from gr2lab import (  # type: ignore
    DEFAULT_DENSIFY_FPS,
    apply_gr2lab_to_dump,
    densify_doc_bspline,
    pack_dump,
    suggested_filename,
    validate_gr2lab,
    write_gr2lab,
)
from timing import StageTimer, maybe_timer  # type: ignore


def apply_prefs(*, cache_root: str = "", body_base: str = "") -> None:
    configure_layout(
        cache_root=cache_root.strip() or None,
        body_base=body_base.strip() or None,
    )


def list_body_bases_local(source: Optional[str] = None) -> Dict[str, Any]:
    from bind_pose import list_body_bases, match_body_base_for_source  # type: ignore

    bases = list_body_bases()
    matched = match_body_base_for_source(source) if source else None
    return {"ok": True, "bases": bases, "matched": matched, "body_base_dir": str(BODY_BASE)}


def decode_gr2_local(
    gr2_path: str | Path,
    *,
    body_base: Optional[str] = None,
    timer: Optional[StageTimer] = None,
) -> Dict[str, Any]:
    """Decode a GR2 file into OUT_FILES dump. Returns run_id + metadata."""
    ensure_layout()
    path = Path(gr2_path)
    if not path.is_file():
        raise FileNotFoundError(f"GR2 not found: {gr2_path}")
    run_id, files_dir, _gr2_dir = make_run_dirs("decode", preferred_name=path.name)
    invalidate_folder_tree_cache()
    dump_gr2(path, files_dir, timer=timer, scan_utf16=False)

    anim_bones = 0
    anim_path = files_dir / "anim_map.json"
    if anim_path.is_file():
        try:
            anim_bones = len(
                json.loads(anim_path.read_text(encoding="utf-8")).get("bones") or []
            )
        except Exception:
            pass

    bind_info = None
    bind_warning = None
    bind_optional = False
    try:
        from rig_catalog import classify_rig  # type: ignore

        bind_optional = bool(classify_rig(path.name).get("bind_optional"))
    except Exception:
        bind_optional = False
    try:
        from bind_pose import extract_bind_to_dump, match_body_base_for_source  # type: ignore

        with maybe_timer(timer, "bind_pose"):
            base = (body_base or "").strip() or None
            if not base:
                base = match_body_base_for_source(path.name)
            if base:
                bind_info = extract_bind_to_dump(files_dir, base)
            else:
                bind_warning = f"No Body_Base match for {path.name}"
    except Exception as e:
        bind_warning = str(e)

    if not bind_info and not bind_optional and not bind_warning:
        bind_warning = (
            f"No T-pose bind for {path.name} - rest pose / NLA on other rigs will be wrong."
        )

    history_meta = None
    try:
        from dump_history import snapshot_original  # type: ignore

        with maybe_timer(timer, "history_snapshot"):
            history_meta = snapshot_original(files_dir)
    except Exception:
        pass

    return {
        "ok": True,
        "message": f"Decoded {path.name}",
        "run_id": run_id,
        "source": path.name,
        "source_path": str(path.resolve()),
        "history": history_meta,
        "anim_bones": anim_bones,
        "bind": bind_info,
        "bind_warning": bind_warning,
        "bind_optional": bind_optional,
        "artifacts": list_dir_files(files_dir),
        "files_dir": str(files_dir),
        "dump_dir": str(files_dir),
    }


_PREVIEW_BONES = {
    "Root_M",
    "Pelvis_M",
    "Spine1_M",
    "Spine2_M",
    "Spine3_M",
    "Chest_M",
    "Neck_M",
    "Head_M",
    "Shoulder_L",
    "Elbow_L",
    "Wrist_L",
    "Shoulder_R",
    "Elbow_R",
    "Wrist_R",
    "Hip_L",
    "Knee_L",
    "Ankle_L",
    "Hip_R",
    "Knee_R",
    "Ankle_R",
    "Tail0_M",
    "Tail1_M",
    "Tail2_M",
    "Tail3_M",
}
_PREVIEW_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4)


def _preview_degree(channel: Dict[str, Any]) -> int:
    degree = int(channel.get("degree") or 0)
    if degree > 0:
        return degree
    mode = str(channel.get("edit_mode") or "")
    if mode in {"quat_k16", "k16_offsets"}:
        return 2
    return 1


def _preview_eval(channel: Optional[Dict[str, Any]], t: float, fallback: List[float], *, quat: bool) -> List[float]:
    ch = channel or {}
    times = [float(x) for x in (ch.get("times") or [])]
    values = ch.get("values") or []
    if not times or not values or len(times) != len(values):
        return list(fallback)
    if len(times) == 1 or t <= times[0]:
        return [float(v) for v in values[0]]
    if t >= times[-1]:
        return [float(v) for v in values[-1]]
    degree = _preview_degree(ch)
    try:
        if degree >= 2 and len(times) >= degree + 1:
            if quat:
                from granny_bspline import sample_bspline_quat  # type: ignore

                out = sample_bspline_quat(degree, times, values, t)
            else:
                from granny_bspline import sample_bspline  # type: ignore

                out = sample_bspline(degree, times, values, t)
            if out and len(out) == len(fallback):
                return [float(v) for v in out]
    except Exception:
        pass
    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        if t0 <= t <= t1:
            span = t1 - t0
            u = 0.0 if span <= 1e-8 else (t - t0) / span
            a = values[i]
            b = values[i + 1]
            return [float(a[k]) * (1.0 - u) + float(b[k]) * u for k in range(len(fallback))]
    return [float(v) for v in values[-1]]


def _preview_wanted(skeleton: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {b.get("name"): b for b in skeleton if b.get("name")}
    wanted = set()
    for name, bone in by_name.items():
        if name in _PREVIEW_BONES or str(name).startswith("Tail"):
            if str(name).startswith("Tail") and name not in _PREVIEW_BONES:
                if name not in {"Tail0_M", "Tail1_M", "Tail2_M", "Tail3_M"}:
                    continue
            wanted.add(name)
            parent = bone.get("parent_name")
            while parent and parent in by_name and parent not in wanted:
                wanted.add(parent)
                parent = by_name[parent].get("parent_name")
    ordered: List[Dict[str, Any]] = []
    pending = set(wanted)
    while pending:
        grew = False
        for name in list(pending):
            parent = by_name[name].get("parent_name")
            if not parent or parent not in pending:
                ordered.append(by_name[name])
                pending.remove(name)
                grew = True
        if not grew:
            for name in sorted(pending):
                ordered.append(by_name[name])
            break
    return ordered


def _stick_from_doc(
    doc: Dict[str, Any],
    *,
    fractions: Optional[Tuple[float, ...]] = None,
    include_points: bool = False,
) -> Dict[str, Any]:
    from mathutils import Matrix, Quaternion, Vector

    skeleton = [b for b in (doc.get("skeleton") or []) if b.get("name")]
    tracks = {t.get("name"): t for t in (doc.get("tracks") or []) if t.get("name")}
    bones = _preview_wanted(skeleton)
    names = [b["name"] for b in bones]
    edges = []
    for bone in bones:
        parent = bone.get("parent_name")
        if parent in names:
            edges.append([parent, bone["name"]])
    duration = max(0.0, float(doc.get("duration") or 0.0))
    samples = fractions if fractions is not None else _PREVIEW_FRACTIONS
    parents = {b["name"]: (b.get("parent_name") or "") for b in bones}
    poses = []
    for frac in samples:
        t = duration * float(frac)
        worlds: Dict[str, Matrix] = {}
        points: Dict[str, List[float]] = {} if include_points else {}
        rig: Dict[str, Dict[str, List[float]]] = {}
        for bone in bones:
            name = bone["name"]
            track = tracks.get(name) or {}
            pos = _preview_eval(
                track.get("position"),
                t,
                [float(v) for v in (bone.get("rest_translation") or [0.0, 0.0, 0.0])],
                quat=False,
            )
            quat = _preview_eval(
                track.get("orientation"),
                t,
                [float(v) for v in (bone.get("rest_rotation") or [0.0, 0.0, 0.0, 1.0])],
                quat=True,
            )
            scale = [float(v) for v in (bone.get("rest_scale") or [1.0, 1.0, 1.0])]
            if len(pos) < 3:
                pos = (pos + [0.0, 0.0, 0.0])[:3]
            if len(quat) < 4:
                quat = [0.0, 0.0, 0.0, 1.0]
            rig[name] = {
                "p": [float(pos[0]), float(pos[1]), float(pos[2])],
                "q": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            }
            local = Matrix.LocRotScale(
                Vector((pos[0], pos[1], pos[2])),
                Quaternion((quat[3], quat[0], quat[1], quat[2])),
                Vector((scale[0], scale[1], scale[2])),
            )
            parent = bone.get("parent_name")
            world = (worlds[parent] @ local) if parent in worlds else local
            worlds[name] = world
            if include_points:
                loc = world.to_translation()
                # Granny Y-up to Blender Z-up (same Rx +90 as import).
                points[name] = [float(loc.x), float(-loc.z), float(loc.y)]
        pose: Dict[str, Any] = {"t": t, "frac": frac, "rig": rig}
        if include_points:
            pose["points"] = points
        poses.append(pose)
    return {
        "duration": round(duration, 4),
        "edges": edges,
        "poses": poses,
        "bones": len(names),
        "bone_order": names,
        "parents": parents,
    }


def stick_blend_poses(
    bone_order: List[str],
    parents: Dict[str, str],
    pose_a: Dict[str, Any],
    pose_b: Dict[str, Any],
    alpha: float,
) -> Dict[str, List[float]]:
    """Interpolate local bone TRS then FK — smoother than lerping world stick points."""
    from mathutils import Matrix, Quaternion, Vector

    rig_a = pose_a.get("rig") or {}
    rig_b = pose_b.get("rig") or {}
    if not rig_a or not rig_b:
        return {}
    u = max(0.0, min(1.0, float(alpha)))
    worlds: Dict[str, Matrix] = {}
    points: Dict[str, List[float]] = {}
    for name in bone_order:
        ra = rig_a.get(name)
        rb = rig_b.get(name)
        if not ra or not rb:
            continue
        p0, p1 = ra["p"], rb["p"]
        pos = (
            float(p0[0]) + (float(p1[0]) - float(p0[0])) * u,
            float(p0[1]) + (float(p1[1]) - float(p0[1])) * u,
            float(p0[2]) + (float(p1[2]) - float(p0[2])) * u,
        )
        q0, q1 = ra["q"], rb["q"]
        qa = Quaternion((float(q0[3]), float(q0[0]), float(q0[1]), float(q0[2])))
        qb = Quaternion((float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])))
        quat = qa.slerp(qb, u)
        local = Matrix.LocRotScale(Vector(pos), quat, Vector((1.0, 1.0, 1.0)))
        parent = parents.get(name) or ""
        world = (worlds[parent] @ local) if parent and parent in worlds else local
        worlds[name] = world
        loc = world.to_translation()
        points[name] = [float(loc.x), float(-loc.z), float(loc.y)]
    return points


def stick_sample_fractions(
    duration: float,
    *,
    fps: float = 30.0,
    max_frames: int = 220,
) -> Tuple[float, ...]:
    """Evenly spaced 0..1 samples for real-time stick playback."""
    if duration <= 1e-6:
        return (0.0,)
    count = min(max_frames, max(2, int(round(duration * fps))))
    if count <= 1:
        return (0.0,)
    return tuple(i / (count - 1) for i in range(count))


def preview_stick_poses(
    gr2_path: str | Path,
    *,
    fractions: Optional[Tuple[float, ...]] = None,
    sample_fps: float = 30.0,
    max_frames: int = 220,
) -> Dict[str, Any]:
    """Stick poses from sparse GR2 decode. No Blender objects.

    The temp dump is removed before return.
    """
    ensure_layout()
    path = Path(gr2_path)
    if not path.is_file():
        raise FileNotFoundError(f"GR2 not found: {gr2_path}")
    t0 = time.perf_counter()
    run_id, files_dir, _gr2_dir = make_run_dirs("preview", preferred_name=path.name)
    try:
        dump_gr2(path, files_dir, scan_utf16=False)
        doc = pack_dump(files_dir, run_id=run_id, densify_bspline=False)
        if fractions is None:
            duration = max(0.0, float(doc.get("duration") or 0.0))
            fractions = stick_sample_fractions(duration, fps=sample_fps, max_frames=max_frames)
        stick = _stick_from_doc(doc, fractions=fractions)
    finally:
        shutil.rmtree(files_dir, ignore_errors=True)
    stick["seconds"] = round(time.perf_counter() - t0, 3)
    stick["path"] = str(path)
    return stick


def pack_gr2lab_doc(
    run_id: str,
    *,
    densify: bool = True,
    timer: Optional[StageTimer] = None,
    write_bridge: bool = False,
) -> Dict[str, Any]:
    """Build densified .gr2lab doc from a dump (same as former /api/gr2lab/export).

    Import uses the in-memory doc; the 25MB pretty bridge.gr2lab is optional.
    ``sparse`` is the pre-densify pack (vanilla formats for export Root_M protect).
    """
    sec_dir = resolve_dump_dir(run_id)
    if not sec_dir.is_dir():
        raise FileNotFoundError(f"Dump not found: {run_id}")
    with maybe_timer(timer, "pack_sample"):
        doc = pack_dump(
            sec_dir,
            run_id=sec_dir.name,
            densify_bspline=False,
            densify_fps=DEFAULT_DENSIFY_FPS,
        )
    doc["run_id"] = sec_dir.name
    sparse = deepcopy(doc)
    if densify:
        from granny_anim import load_dump_gr2  # type: ignore

        with maybe_timer(timer, "densify_load"):
            _gr2, anim, _s0 = load_dump_gr2(sec_dir)
        densify_doc_bspline(doc, anim, fps=DEFAULT_DENSIFY_FPS, timer=timer)
    elif timer is not None:
        timer.add("densify", 0.0)
    filename = "bridge.gr2lab"
    out_path = sec_dir / filename
    if write_bridge:
        with maybe_timer(timer, "write_bridge_json"):
            try:
                write_gr2lab(doc, out_path)
            except OSError:
                out_path = sec_dir / filename
    elif timer is not None:
        timer.add("write_bridge_json", 0.0)
    return {
        "ok": True,
        "run_id": sec_dir.name,
        "filename": filename,
        "path": str(out_path),
        "doc": doc,
        "sparse": sparse,
        "bones": len(doc.get("skeleton") or []),
        "tracks": len(doc.get("tracks") or []),
        "duration": doc.get("duration"),
        "has_bind": bool(doc.get("bind")),
        "source_interp": doc.get("source_interp"),
        "densified_channels": doc.get("densified_channels"),
        "densify_fps": doc.get("densify_fps"),
    }


def import_gr2_pipeline_local(
    gr2_path: str | Path,
    *,
    body_base: Optional[str] = None,
    progress=None,
    timer: Optional[StageTimer] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Decode + densify. Returns (decode_info, gr2lab_doc)."""
    timer = timer or StageTimer("import")
    if progress:
        progress.step(8, f"Decoding {Path(gr2_path).name}...")
    dec = decode_gr2_local(gr2_path, body_base=body_base, timer=timer)
    run_id = dec["run_id"]
    if progress:
        progress.step(48, "Densifying animation for Blender...")
    packed = pack_gr2lab_doc(run_id, densify=True, timer=timer, write_bridge=False)
    doc = packed["doc"]
    doc["run_id"] = run_id
    dec["sparse_doc"] = packed.get("sparse")
    dec["timer"] = timer
    if progress:
        progress.step(58, "Preparing Blender rig...")
    return dec, doc


def apply_gr2lab_local(
    doc: dict,
    from_run: Optional[str] = None,
    *,
    timer: Optional[StageTimer] = None,
) -> Dict[str, Any]:
    """Apply Blender gr2lab onto dump (reset to original first when history exists)."""
    from dump_history import (  # type: ignore
        ORIGINAL_ID,
        dump_has_edits,
        live_matches_original,
        mark_blender_import,
        restore_stage,
    )
    from granny_anim import invalidate_dump_cache  # type: ignore

    validate_gr2lab(doc)
    rid = Path(from_run).name if from_run else Path(str(doc.get("run_id"))).name
    sec_dir = resolve_dump_dir(rid)
    if not sec_dir.is_dir():
        raise FileNotFoundError(f"Dump '{rid}' not found - re-import the GR2 first")
    if from_run:
        doc = dict(doc)
        doc["run_id"] = rid

    restored_from = None
    hist = sec_dir / "history" / ORIGINAL_ID
    with maybe_timer(timer, "restore_history"):
        if hist.is_dir() and (hist / "section_0.bin").is_file():
            try:
                if live_matches_original(sec_dir):
                    restored_from = ORIGINAL_ID
                else:
                    restore_stage(sec_dir, ORIGINAL_ID)
                    restored_from = ORIGINAL_ID
                    invalidate_dump_cache(sec_dir)
            except Exception as exc:
                restored_from = f"failed:{exc}"
                invalidate_dump_cache(sec_dir)

    t0 = time.perf_counter()
    result = apply_gr2lab_to_dump(sec_dir, doc, overwrite=True, timer=timer)
    result["ok"] = True
    result["run_id"] = rid
    result["reset_to_original"] = restored_from == ORIGINAL_ID
    result["restored_from"] = restored_from
    result["apply_ms"] = int((time.perf_counter() - t0) * 1000)
    if result.get("changed"):
        mark_blender_import(sec_dir)
        result["has_edits"] = True
    else:
        result["has_edits"] = dump_has_edits(sec_dir)
    return result


def _named_track_count_from_anim(anim) -> int:
    return sum(1 for t in (getattr(anim, "tracks", None) or []) if getattr(t, "name", None))


def _named_track_count_from_map_file(path: Path) -> int:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return sum(1 for t in (doc.get("tracks") or []) if t.get("name"))
    except Exception:
        return 0


def _named_track_count_from_bytes(data: bytes) -> int:
    """Count animation tracks with non-empty names (S1 reloc -> S0 string)."""
    if not data:
        return 0
    try:
        _g, anim, _s0 = _anim_from_bytes(data)
        return _named_track_count_from_anim(anim)
    except Exception:
        return 0


def _anim_from_bytes(data: bytes, source: str = ""):
    from granny_anim import build_anim_map, extract_section_fixups  # type: ignore

    g = parse_gr2(data)
    decode_sections(g, decompress=True)
    try:
        extract_section_fixups(g)
    except Exception:
        pass
    anim = build_anim_map(g, source=source)
    s0 = g.sections[0].uncompressed or b""
    return g, anim, s0


def encode_from_dump(
    from_run: str,
    *,
    source_gr2: Optional[str | Path] = None,
    dest_gr2_path: Optional[str | Path] = None,
    timer: Optional[StageTimer] = None,
) -> Dict[str, Any]:
    """Encode dump -> GR2 bytes (hybrid COMP_NONE for dirty sections)."""
    if is_watch_run(from_run):
        raise ValueError("Watch dumps are not packable")
    sec_dir = resolve_dump_dir(from_run)
    if not sec_dir.is_dir():
        raise FileNotFoundError(f"Decode dump not found: {from_run}")

    path: Optional[Path] = None
    if source_gr2:
        cand = Path(source_gr2)
        if cand.is_file():
            path = cand
    if path is None:
        path = read_dump_source_path(sec_dir)
    if path is None:
        name = None
        src_txt = sec_dir / "SOURCE.txt"
        if src_txt.is_file():
            for line in src_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("name="):
                    name = line.split("=", 1)[1].strip()
                    break
        if name:
            try:
                path = resolve_input_gr2(name)
            except Exception:
                path = None
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Source GR2 for dump '{from_run}' not found. "
            "Re-import the original .GR2 so SOURCE.txt has a valid path=."
        )

    pack_from = Path(from_run).name
    run_id, files_dir, gr2_dir = make_run_dirs("encode", preferred_name=pack_from)
    invalidate_folder_tree_cache()
    out_name = path.stem + "_encoded.GR2"

    from granny_anim import build_anim_map, extract_section_fixups  # type: ignore
    from dump_history import dump_has_edits  # type: ignore

    with maybe_timer(timer, "load_source_gr2"):
        gr2 = load_gr2(path)
        original = gr2.original_bytes if gr2.original_bytes is not None else path.read_bytes()
        decode_sections(gr2, decompress=True)
        try:
            extract_section_fixups(gr2)
        except Exception:
            pass
        anim_base = build_anim_map(gr2, source=str(path))
        s0_base = bytes(gr2.sections[0].uncompressed or b"")
        orig_headers = list(gr2.original_section_headers or [s.header for s in gr2.sections])

    named_src = _named_track_count_from_anim(anim_base)
    if named_src == 0:
        named_src = _named_track_count_from_map_file(sec_dir / "anim_map.json")

    hist = sec_dir / "history" / "000_original"
    trust_live_dump = dump_has_edits(sec_dir)

    def _matches_history(i: int, data: bytes | None, reloc_data: bytes | None) -> bool:
        if not hist.is_dir():
            return False
        hp = hist / f"section_{i}.bin"
        if data is None or not hp.is_file():
            return False
        if sha256_bytes(data) != sha256_bytes(hp.read_bytes()):
            return False
        hr = hist / f"section_{i}.relocations.bin"
        if hr.is_file():
            if reloc_data is None:
                return False
            return sha256_bytes(reloc_data) == sha256_bytes(hr.read_bytes())
        return True

    for i, sec in enumerate(gr2.sections):
        p = sec_dir / f"section_{i}.bin"
        data = None
        if p.is_file() and sec.header.uncompressed_size > 0:
            data = p.read_bytes()
        reloc_p = sec_dir / f"section_{i}.relocations.bin"
        mm_p = sec_dir / f"section_{i}.mm.bin"
        reloc_data = reloc_p.read_bytes() if reloc_p.is_file() else None
        mm_data = mm_p.read_bytes() if mm_p.is_file() else None

        if _matches_history(i, data, reloc_data) and not trust_live_dump:
            if reloc_data is not None:
                sec.relocations = reloc_data
            if mm_data is not None:
                sec.mixed_marshalling = mm_data
            continue

        payload_changed = data is not None and data != (sec.uncompressed or b"")
        reloc_changed = reloc_data is not None and reloc_data != (sec.relocations or b"")
        mm_changed = mm_data is not None and mm_data != (sec.mixed_marshalling or b"")

        if payload_changed or reloc_changed or mm_changed:
            use = data if data is not None else (sec.uncompressed or b"")
            mark_section_edited(gr2, i, use, preserve_fixups=True)
        if reloc_data is not None:
            sec.relocations = reloc_data
        if mm_data is not None:
            sec.mixed_marshalling = mm_data

    dirty = any(s.dirty for s in gr2.sections)
    with maybe_timer(timer, "encode_after_edit"):
        encoded = encode_after_edit(gr2) if dirty else encode_passthrough(gr2)
    mode = "edited" if dirty else "identity"
    (gr2_dir / "FROM.txt").write_text(
        f"from_dump={from_run}\nbase={path.name}\nmode={mode}\n",
        encoding="utf-8",
    )

    enc_gr2 = enc_anim = s0_enc = None
    with maybe_timer(timer, "named_track_verify"):
        enc_gr2, enc_anim, s0_enc = _anim_from_bytes(encoded, source=str(path))
        named_enc = _named_track_count_from_anim(enc_anim)
        if named_src > 0 and named_enc == 0:
            raise ValueError(
                "Export refused: encoded GR2 has no bone names "
                "(S1 relocation table empty/zeroed). Source had "
                f"{named_src} named tracks - not writing {out_name}."
            )

    out_path = Path(dest_gr2_path) if dest_gr2_path else (gr2_dir / out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encoded)
    cache_copy = gr2_dir / out_name
    if out_path.resolve() != cache_copy.resolve():
        shutil.copy2(out_path, cache_copy)

    identical = encoded == original
    if identical:
        mode = "identity"

    size_info = None
    hybrid_warning = None
    if not identical:
        with maybe_timer(timer, "size_breakdown"):
            size_info = encode_size_breakdown_from_headers(
                orig_headers,
                enc_gr2.sections,
                original_size=len(original),
                encoded_size=len(encoded),
            )
            (gr2_dir / "SIZE.json").write_text(
                json.dumps(size_info, separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
        became = [
            s
            for s in (size_info.get("sections") or [])
            if s.get("became_uncompressed")
        ]
        if became:
            hybrid_warning = (
                f"Export OK - {len(became)} section(s) stored uncompressed "
                "(normal: BitKnit recompress unavailable). File is larger but usually game-OK."
            )
    elif timer is not None:
        timer.add("size_breakdown", 0.0)

    root_gate = None
    root_gate_warning = None
    try:
        from root_gate import validate_root_export, write_export_meta  # type: ignore

        with maybe_timer(timer, "root_gate"):
            report = validate_root_export(
                path,
                out_path if out_path.suffix.lower() == ".gr2" else cache_copy,
                dump=sec_dir,
                allow_root_delta=False,
                expect_noop=bool(identical),
                base_anim=(None, anim_base, s0_base),
                encoded_anim=(enc_gr2, enc_anim, s0_enc),
            )
        root_gate = {
            "ok": bool(report.get("ok")),
            "badge": report.get("badge"),
            "max_root_lateral": report.get("max_root_lateral"),
            "max_root_dy": report.get("max_root_dy"),
            "fails": report.get("fails") or [],
            "warns": (report.get("warns") or [])[:8],
        }
        try:
            write_export_meta(sec_dir, {"root_gate": root_gate})
        except Exception:
            pass
        if not root_gate["ok"]:
            lat = root_gate.get("max_root_lateral")
            if isinstance(lat, (int, float)):
                root_gate_warning = (
                    f"Root shift gate FAIL (lateral={lat:.4f} m) - "
                    "in-game camera cuts may jump the character."
                )
            else:
                root_gate_warning = "Root shift gate FAIL - see root_gate.fails"
    except Exception as exc:
        root_gate = {"ok": None, "error": str(exc)}

    payload: Dict[str, Any] = {
        "ok": True,
        "message": f"Encoded from '{from_run}' -> {out_path.name}",
        "run_id": run_id,
        "from_run": from_run,
        "output": str(out_path),
        "output_name": out_path.name,
        "saved_path": str(out_path),
        "size": len(encoded),
        "original_size": len(original),
        "sha256": sha256_bytes(encoded),
        "original_sha256": sha256_bytes(original),
        "identical": identical,
        "mode": mode,
        "badge": "identity" if identical else "edited",
        "first_diff": first_diff(encoded, original),
        "size_breakdown": size_info,
        "hybrid_warning": hybrid_warning,
        "root_gate": root_gate,
        "root_gate_warning": root_gate_warning,
        "files_dir": str(files_dir) if files_dir.is_dir() else None,
        "gr2_dir": str(gr2_dir),
        "artifacts": list_dir_files(gr2_dir),
    }
    if identical:
        payload["ok"] = True
        payload["apply_noop"] = True
        payload["note"] = (
            "Encode matches source byte-for-byte (noop / identity). "
            "File written anyway."
        )
    if timer is not None:
        payload["timing"] = timer.as_dict()
        try:
            timer.write_json(gr2_dir / "TIMING.json")
        except OSError:
            pass
    return payload


def pack_encode_local(
    run_id: str,
    doc: dict,
    dest_gr2_path: str | Path,
    *,
    source_gr2: Optional[str | Path] = None,
    progress=None,
    timer: Optional[StageTimer] = None,
) -> Dict[str, Any]:
    """Apply gr2lab + encode atomically (former /api/gr2lab/pack-encode)."""
    timer = timer or StageTimer("export")
    if progress:
        progress.step(80, f"Verifying dump {run_id}...")
    sec_dir = resolve_dump_dir(run_id)
    if not sec_dir.is_dir():
        raise FileNotFoundError(
            f"Decode dump '{run_id}' not found. Re-import the GR2, then export again."
        )
    if progress:
        progress.step(86, "Applying edits + encoding...")
    imp = apply_gr2lab_local(doc, from_run=run_id, timer=timer)
    enc = encode_from_dump(
        run_id,
        source_gr2=source_gr2,
        dest_gr2_path=dest_gr2_path,
        timer=timer,
    )
    enc["import_result"] = imp
    enc["from_run"] = run_id
    if not imp.get("changed"):
        enc["apply_noop"] = True
        enc["note"] = (
            "Animation matched the dump (noop apply); encoded GR2 written anyway"
        )
    elif imp.get("changed") and enc.get("identical"):
        enc["warning"] = (
            "Apply reported changes but encode is byte-identical to the source GR2"
        )
        enc["warning_code"] = "IDENTITY_AFTER_APPLY"
    if progress:
        progress.step(99, "Export done")
    enc["timer"] = timer
    return enc

def dump_still_exists(run_id: str) -> bool:
    try:
        return resolve_dump_dir(run_id).is_dir() and any(
            (resolve_dump_dir(run_id) / f"section_{i}.bin").is_file() for i in range(2)
        )
    except Exception:
        return False


def decode_run_ids() -> List[str]:
    return [str(r.get("id") or "") for r in list_decode_runs() if r.get("id")]
