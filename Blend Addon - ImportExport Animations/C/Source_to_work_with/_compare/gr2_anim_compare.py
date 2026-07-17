"""
BG3 GR2 / GLB animation comparison tool.
Generates comparison_report.html + comparison_report.txt
"""

from __future__ import annotations

import html
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
REPORT_HTML = SCRIPT_DIR / "comparison_report.html"
REPORT_TXT = SCRIPT_DIR / "comparison_report.txt"
VIEWER_JS = SCRIPT_DIR / "viewer.js"

REFERENCE_GR2 = SOURCE_DIR / "TIF_FS_Rig_SCENE_CAMP_Karlach_SD_ROM_ForgingOfTheHeart_Romance_Karlach_000_TIF_SK.GR2"
TEST_GR2 = SOURCE_DIR / "TIF_FS_Rig_SCENE_CAMP_Karlach_SD_ROM_ForgingOfTheHeart_Romance_Karlach_DGB_EL_1_sk_tif.GR2"
RAW_GR2 = SOURCE_DIR / "TIF_FS_Rig_SCENE_CAMP_Karlach_SD_ROM_ForgingOfTheHeart_Romance_Karlach_000.GR2"

REFERENCE_GLB = SCRIPT_DIR / "lslib.glb"
TEST_GLB = SCRIPT_DIR / "blender.glb"

DIVINE = Path(r"C:\BG3_Modding\000_Mod_tools_progs\Mod_tools_progs\ExportTool\Packed\Tools\Divine.exe")

FOCUS_BONES = [
    "Dummy_Root", "Root_M", "Chest_M",
    "Dummy_L_Foot_IK", "Dummy_R_Foot_IK",
    "Dummy_L_Hand_IK", "Dummy_R_Hand_IK",
    "Toes_L", "Head_M",
]

CAMERA_CUT_T = 12.3
SKELETON_FPS = 30  # match Blender/game fps for smooth preview

# Body chain for readable 3D skeleton (skip FX / Dummy clutter)
BODY_BONES = {
    "Dummy_Root", "Root_M",
    "Hip_L", "Knee_L", "Ankle_L", "Toes_L",
    "Hip_R", "Knee_R", "Ankle_R", "Toes_R",
    "Spine1_M", "Spine2_M", "Chest_M", "Neck_M", "Head_M",
    "Scapula_L", "Shoulder_L", "Elbow_L", "Wrist_L",
    "Scapula_R", "Shoulder_R", "Elbow_R", "Wrist_R",
    "ThumbFinger1_L", "ThumbFinger2_L", "ThumbFinger3_L",
    "IndexFinger1_L", "IndexFinger2_L", "IndexFinger3_L",
    "MiddleFinger1_L", "MiddleFinger2_L", "MiddleFinger3_L",
    "RingFinger1_L", "RingFinger2_L", "RingFinger3_L",
    "PinkyFinger1_L", "PinkyFinger2_L", "PinkyFinger3_L",
    "ThumbFinger1_R", "ThumbFinger2_R", "ThumbFinger3_R",
    "IndexFinger1_R", "IndexFinger2_R", "IndexFinger3_R",
    "MiddleFinger1_R", "MiddleFinger2_R", "MiddleFinger3_R",
    "RingFinger1_R", "RingFinger2_R", "RingFinger3_R",
    "PinkyFinger1_R", "PinkyFinger2_R", "PinkyFinger3_R",
}
IK_BONES = {
    "Dummy_L_Foot_IK", "Dummy_R_Foot_IK",
    "Dummy_L_Hand_IK", "Dummy_R_Hand_IK",
}

BONE_HELP = {
    "Dummy_Root": "Moves the whole character in the scene (cinematic world position).",
    "Root_M": "Main body anchor on the rig.",
    "Chest_M": "Upper torso.",
    "Dummy_L_Foot_IK": "Left foot IK target — game plants foot here.",
    "Dummy_R_Foot_IK": "Right foot IK target.",
    "Dummy_L_Hand_IK": "Left hand IK target.",
    "Dummy_R_Hand_IK": "Right hand IK target.",
    "Toes_L": "Left toe bone.",
    "Head_M": "Head orientation.",
}


@dataclass
class Gr2Scan:
    label: str
    path: Path
    size: int
    bones: list[str]


@dataclass
class TrackDiff:
    bone: str
    prop: str
    ref_keys: int
    test_keys: int
    max_err: float
    unit: str
    worst_t: float
    status: str
    series_t: list[float]
    series_err: list[float]


# --- GLB parsing ---

def parse_glb(path: Path):
    data = path.read_bytes()
    offset = 12
    chunks = []
    while offset < len(data):
        chunk_len, _ = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks.append(data[offset : offset + chunk_len])
        offset += chunk_len
    return json.loads(chunks[0].decode()), chunks[1]


def read_accessor(js, blob, idx):
    acc = js["accessors"][idx]
    bv = js["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count, comp, ctype = acc["count"], acc["componentType"], acc["type"]
    dtype = {5126: "f4"}[comp]
    comps = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[ctype]
    itemsize = np.dtype(dtype).itemsize * comps
    stride = bv.get("byteStride", 0) or itemsize
    out = []
    for i in range(count):
        chunk = blob[start + i * stride : start + i * stride + itemsize]
        out.append(np.frombuffer(chunk, dtype=dtype, count=comps))
    return out


def get_anim_duration(path: Path) -> float:
    js, blob = parse_glb(path)
    if not js.get("animations"):
        return 20.0
    anim = js["animations"][0]
    max_t = 0.0
    for ch in anim.get("channels", []):
        s = anim["samplers"][ch["sampler"]]
        times = read_accessor(js, blob, s["input"])
        if times:
            max_t = max(max_t, float(times[-1][0]))
    return max_t if max_t > 0 else 20.0


def get_tracks(path: Path):
    js, blob = parse_glb(path)
    nodes = [n.get("name", "") for n in js["nodes"]]
    anim = js["animations"][0]
    tracks = {}
    for ch in anim["channels"]:
        bone = nodes[ch["target"]["node"]]
        prop = ch["target"]["path"]
        sampler = anim["samplers"][ch["sampler"]]
        times = np.array(read_accessor(js, blob, sampler["input"]), dtype=float).reshape(-1)
        vals = np.array(read_accessor(js, blob, sampler["output"]), dtype=float)
        tracks.setdefault(bone, {})[prop] = (len(times), times, vals)
    return tracks


# --- Skeleton pose math ---

def quat_to_mat3(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def mat4_from_trs(t, r, s):
    m = np.eye(4)
    rm = quat_to_mat3(r)
    m[:3, :3] = rm * s.reshape(3, 1)
    m[:3, 3] = t
    return m


def node_rest_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=float)
    r = np.array(node.get("rotation", [0, 0, 0, 1]), dtype=float)
    s = np.array(node.get("scale", [1, 1, 1]), dtype=float)
    return mat4_from_trs(t, r, s)


def sample_vec(times, vals, t):
    if len(times) == 0:
        return vals[0] if len(vals) else np.zeros(3)
    idx = max(0, int(np.searchsorted(times, t, side="right")) - 1)
    return vals[idx]


def build_hierarchy(js):
    nodes = js["nodes"]
    parents = [None] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parents[c] = i
    return nodes, parents


def is_display_bone(name: str) -> bool:
    return name in BODY_BONES or name in IK_BONES


def get_bone_edges(js, body_only: bool = True):
    nodes = js["nodes"]
    edges = []
    for i, n in enumerate(nodes):
        pname = n.get("name", f"node_{i}")
        for ci in n.get("children", []):
            cname = nodes[ci].get("name", f"node_{ci}")
            if body_only and not (pname in BODY_BONES and cname in BODY_BONES):
                continue
            edges.append([pname, cname])
    return edges


def compute_world_positions(path: Path, t: float) -> dict[str, np.ndarray]:
    js, blob = parse_glb(path)
    nodes, parents = build_hierarchy(js)
    tracks = get_tracks(path)
    names = [n.get("name", f"node_{i}") for i, n in enumerate(nodes)]
    local_mats = [node_rest_matrix(n) for n in nodes]

    for i, name in enumerate(names):
        ch = tracks.get(name, {})
        if "translation" in ch:
            _, ts, vs = ch["translation"]
            local_mats[i][:3, 3] = sample_vec(ts, vs, t)
        if "rotation" in ch:
            _, ts, vs = ch["rotation"]
            q = sample_vec(ts, vs, t)
            s = np.linalg.norm(local_mats[i][:3, :3], axis=0)
            s = np.where(s > 1e-8, s, 1.0)
            local_mats[i][:3, :3] = quat_to_mat3(q) * s.reshape(3, 1)
        if "scale" in ch:
            _, ts, vs = ch["scale"]
            sc = sample_vec(ts, vs, t)
            rot = local_mats[i][:3, :3]
            rs = np.linalg.norm(rot, axis=0)
            rs = np.where(rs > 1e-8, rs, 1.0)
            basis = rot / rs
            local_mats[i][:3, :3] = basis * sc.reshape(3, 1)

    world = [np.eye(4)] * len(nodes)
    for i in range(len(nodes)):
        p = parents[i]
        world[i] = world[p] @ local_mats[i] if p is not None else local_mats[i].copy()

    return {names[i]: world[i][:3, 3].copy() for i in range(len(nodes))}


def export_skeleton_data(path: Path, duration: float) -> dict:
    js, _ = parse_glb(path)
    all_names = [js["nodes"][i].get("name", f"node_{i}") for i in range(len(js["nodes"]))]
    # Keep body + IK only so the 3D view looks like a real character, not FX clutter
    bones = [n for n in all_names if is_display_bone(n)]
    name_set = set(bones)
    edges = [[a, b] for a, b in get_bone_edges(js, body_only=True) if a in name_set and b in name_set]
    n_frames = max(2, int(duration * SKELETON_FPS) + 1)
    times = [round(i / SKELETON_FPS, 3) for i in range(n_frames)]
    if times[-1] < duration:
        times.append(round(duration, 3))

    positions = []
    for t in times:
        wp = compute_world_positions(path, t)
        positions.append([wp.get(b, [0.0, 0.0, 0.0]).tolist() for b in bones])

    return {
        "bones": bones,
        "edges": edges,
        "times": times,
        "positions": positions,
        "bodyBones": sorted(BODY_BONES & name_set),
        "ikBones": sorted(IK_BONES & name_set),
        "fps": SKELETON_FPS,
    }


def decode_glb_text(path: Path, label: str) -> str:
    js, blob = parse_glb(path)
    nodes, parents = build_hierarchy(js)
    lines = [
        f"=== DECODED GLB: {label} ===",
        f"File: {path.name}",
        f"Nodes (bones): {len(nodes)}",
        f"Meshes: {len(js.get('meshes', []))}  (0 = animation/skeleton only, no body mesh)",
        f"Skins: {len(js.get('skins', []))}",
        f"Animations: {len(js.get('animations', []))}",
        "",
        "BONE HIERARCHY (parent -> child):",
        "-" * 50,
    ]
    for i, n in enumerate(nodes):
        name = n.get("name", f"node_{i}")
        p = parents[i]
        pname = nodes[p].get("name", "?") if p is not None else "(root)"
        lines.append(f"  {pname} -> {name}")

    tracks = get_tracks(path)
    lines += ["", "ANIMATION TRACKS (every channel in file):", "-" * 50]
    anim = js["animations"][0]
    node_names = [n.get("name", "") for n in nodes]
    for ch in anim["channels"]:
        bone = node_names[ch["target"]["node"]]
        prop = ch["target"]["path"]
        s = anim["samplers"][ch["sampler"]]
        times = read_accessor(js, blob, s["input"])
        nkeys = len(times)
        t0 = float(times[0][0]) if times else 0
        t1 = float(times[-1][0]) if times else 0
        lines.append(f"  {bone:28} {prop:12}  {nkeys:4d} keys  ({t0:.2f}s - {t1:.2f}s)")

    return "\n".join(lines) + "\n"


# --- Comparison helpers ---

def quat_err(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return float(np.degrees(2 * np.arccos(min(abs(float(np.dot(q1, q2))), 1.0))))


def scan_gr2(label: str, path: Path) -> Gr2Scan | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    names = [b"Dummy_Root", b"Dummy_L_Foot_IK", b"Dummy_R_Foot_IK",
             b"Dummy_L_Hand_IK", b"Dummy_R_Hand_IK", b"Root_M", b"Chest_M"]
    found = [n.decode() for n in names if data.find(n) >= 0]
    return Gr2Scan(label, path, len(data), found)


def classify_translation(cm: float) -> str:
    if cm < 1.0: return "ok"
    if cm < 5.0: return "warn"
    return "bad"


def classify_rotation(deg: float) -> str:
    if deg < 3.0: return "ok"
    if deg < 15.0: return "warn"
    return "bad"


def status_label(status: str) -> str:
    return {"ok": "OK", "warn": "WARNING", "bad": "BAD", "missing": "MISSING"}[status]


def human_issue(d: TrackDiff) -> str:
    bone_desc = BONE_HELP.get(d.bone, "Animated bone.")
    at_cut = abs(d.worst_t - CAMERA_CUT_T) < 1.5
    cut_note = " Near camera cut — likely in-game drift." if at_cut else ""
    key_note = f" Keys {d.ref_keys}->{d.test_keys} (resampled)." if d.test_keys > d.ref_keys * 2 else ""
    if d.status == "ok":
        return f"{d.bone} / {d.prop}: OK. Max {d.max_err:.2f} {d.unit}."
    return (
        f"{d.bone} / {d.prop}: {status_label(d.status)}. "
        f"Max error {d.max_err:.2f} {d.unit} at {d.worst_t:.2f}s. "
        f"{bone_desc}{key_note}{cut_note}"
    )


def compare_track(bone, prop, ref_tracks, test_tracks) -> TrackDiff | None:
    if bone not in ref_tracks or prop not in ref_tracks[bone]:
        return None
    if bone not in test_tracks or prop not in test_tracks[bone]:
        return TrackDiff(bone, prop, 0, 0, 0, "missing", 0, "bad", [], [])

    k1, t1, v1 = ref_tracks[bone][prop]
    k2, t2, v2 = test_tracks[bone][prop]
    step = max(1, len(t1) // 120)
    series_t, series_err, worst, worst_t = [], [], 0.0, 0.0
    unit = "cm" if prop == "translation" else "deg"
    for t in t1[::step]:
        i = max(0, int(np.searchsorted(t2, t, side="right")) - 1)
        j = max(0, int(np.searchsorted(t1, t, side="right")) - 1)
        err = float(np.linalg.norm(v1[j] - v2[i])) * 100 if prop == "translation" else quat_err(v1[j], v2[i])
        series_t.append(float(t))
        series_err.append(err)
        if err > worst:
            worst, worst_t = err, float(t)
    status = classify_translation(worst) if prop == "translation" else classify_rotation(worst)
    return TrackDiff(bone, prop, k1, k2, worst, unit, worst_t, status, series_t, series_err)


def convert_gr2_to_glb(gr2: Path, glb: Path) -> tuple[bool, str]:
    if not DIVINE.exists():
        return False, f"Divine not found: {DIVINE}"
    subprocess.run([str(DIVINE), "-a", "convert-resource", "-g", "bg3", "-s", str(gr2),
                      "-d", str(glb), "-i", "gr2", "-o", "gltf", "-e", "yup"],
                     capture_output=True, text=True)
    return (True, "converted") if glb.exists() and glb.stat().st_size > 0 else (False, "failed")


def svg_sparkline(times, errs, unit, cut_t):
    if not times:
        return "<span class='muted'>no data</span>"
    w, h, pad = 220, 44, 4
    t0, t1 = min(times), max(times)
    e_max = max(errs) or 1.0

    def px(t): return pad + (t - t0) / max(t1 - t0, 0.001) * (w - 2 * pad)
    def py(e): return h - pad - (e / e_max) * (h - 2 * pad)

    pts = " ".join(f"{px(t):.1f},{py(e):.1f}" for t, e in zip(times, errs))
    cut_x = px(cut_t) if t0 <= cut_t <= t1 else None
    cut_line = f"<line x1='{cut_x:.1f}' y1='{pad}' x2='{cut_x:.1f}' y2='{h-pad}' stroke='#f59e0b' stroke-width='1' stroke-dasharray='3,2'/>" if cut_x else ""
    return f"<svg class='spark' viewBox='0 0 {w} {h}' width='{w}' height='{h}'><rect width='{w}' height='{h}' fill='#1e1e2e' rx='4'/>{cut_line}<polyline fill='none' stroke='#60a5fa' stroke-width='1.5' points='{pts}'/></svg>"


def build_text_report(gr2_scans, diffs, duration, ref_decode, test_decode) -> str:
    lines = [
        "BG3 GR2 ANIMATION COMPARISON — FULL READABLE REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Animation length: ~{duration:.1f} seconds",
        f"Camera-cut zone: ~{CAMERA_CUT_T}s",
        "",
        "WHAT THE FILES ARE",
        "-" * 40,
    ]
    for s in gr2_scans:
        bones = ", ".join(s.bones) if s.bones else "no bone names (animation-only)"
        lines += [f"  [{s.label}] {s.path.name}", f"    Size: {s.size:,} bytes", f"    Bones: {bones}", ""]

    bad = [d for d in diffs if d.status == "bad"]
    warn = [d for d in diffs if d.status == "warn"]
    lines += [
        "SUMMARY", "-" * 40,
        f"  {len(bad)} BAD  |  {len(warn)} WARN  |  {len(diffs) - len(bad) - len(warn)} OK", "",
    ]
    if bad:
        lines += ["CRITICAL ISSUES", "-" * 40]
        for d in sorted(bad, key=lambda x: -x.max_err):
            lines.append(f"  * {human_issue(d)}")
        lines.append("")
    if warn:
        lines += ["WARNINGS", "-" * 40]
        for d in warn:
            lines.append(f"  * {human_issue(d)}")
        lines.append("")

    lines += [
        "WHAT THIS MEANS IN-GAME", "-" * 40,
        "  Scene cinematics use Dummy_Root position + IK foot/hand bones.",
        "  Wrong values at camera cuts (~12s) = character jumps or slides.", "",
        "ALL TRACKS", "-" * 40,
    ]
    for d in diffs:
        lines.append(f"  [{d.status.upper():7}] {d.bone:22} {d.prop:11}  {d.max_err:.2f} {d.unit} @ {d.worst_t:.2f}s  keys {d.ref_keys}->{d.test_keys}")

    lines += ["", ref_decode, "", test_decode]
    return "\n".join(lines) + "\n"


def build_html(gr2_scans, diffs, duration, full_text, skeleton_json, viewer_js) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bad = [d for d in diffs if d.status == "bad"]
    warn = [d for d in diffs if d.status == "warn"]

    issue_items = "".join(f"<li class='{d.status}'>{html.escape(human_issue(d))}</li>" for d in bad + warn) or "<li class='ok'>All tracks OK.</li>"

    rows = []
    for d in diffs:
        badge = d.status.upper()
        cut_flag = " &#9888; cut" if abs(d.worst_t - CAMERA_CUT_T) < 1.5 else ""
        rows.append(
            f"<tr class='{d.status}'><td><strong>{html.escape(d.bone)}</strong></td>"
            f"<td>{d.prop}</td><td>{d.ref_keys}&rarr;{d.test_keys}</td>"
            f"<td><strong>{d.max_err:.2f}</strong> {d.unit}</td>"
            f"<td>{d.worst_t:.2f}s{cut_flag}</td><td><span class='badge {d.status}'>{badge}</span></td>"
            f"<td>{svg_sparkline(d.series_t, d.series_err, d.unit, CAMERA_CUT_T)}</td></tr>"
        )

    gr2_cards = "".join(
        f"<div class='card'><h3>{html.escape(s.label)}</h3><p class='mono'>{html.escape(s.path.name)}</p>"
        f"<p><strong>{s.size:,}</strong> bytes</p></div>" for s in gr2_scans
    )

    skel_json = json.dumps(skeleton_json, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>GR2 Animation Comparison</title>
<style>
:root{{--bg:#0f1117;--panel:#1a1d27;--text:#e5e7eb;--muted:#9ca3af;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444;--accent:#60a5fa}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Segoe UI,sans-serif;background:var(--bg);color:var(--text);line-height:1.55}}
.wrap{{max-width:1100px;margin:0 auto;padding:14px 16px 24px}}
h1{{margin:0 0 4px;font-size:1.25rem}} h2{{margin:16px 0 8px;color:var(--accent);font-size:1rem}}
.sub{{color:var(--muted);margin-bottom:10px;font-size:.85rem}}
.card{{background:var(--panel);border:1px solid #2a2f3d;border-radius:8px;padding:10px 12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}}
.summary-box{{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:10px 12px;margin:10px 0;font-size:.9rem}}
.issues{{list-style:none;padding:0;margin:0}} .issues li{{padding:7px 10px;margin-bottom:5px;border-radius:6px;font-size:.85rem}}
.issues li.bad{{background:#3f1515;border-left:3px solid var(--bad)}}
.issues li.warn{{background:#3f2e0f;border-left:3px solid var(--warn)}}
.issues li.ok{{background:#14321e;border-left:3px solid var(--ok)}}
.viewer-wrap{{background:var(--panel);border:1px solid #2a2f3d;border-radius:8px;padding:10px;overscroll-behavior:contain;touch-action:none}}
.viewer-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
@media(max-width:900px){{.viewer-grid{{grid-template-columns:1fr}}}}
.viewer-panel{{position:relative}} .viewer-panel canvas{{width:100%;height:360px;display:block;border-radius:6px;background:#07090e;cursor:grab;touch-action:none;user-select:none;-webkit-user-select:none}}
.viewer-panel canvas:active{{cursor:grabbing}}
.viewer-wrap.viewport-active{{outline:1px solid #60a5fa}}
.controls{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px;padding-top:8px;border-top:1px solid #2a2f3d}}
.controls button{{background:#2563eb;color:#fff;border:none;padding:6px 10px;border-radius:5px;cursor:pointer;font-size:.82rem}}
.controls button.sec{{background:#374151}}
.controls input[type=range]{{flex:1;min-width:120px}}
.time-display{{font-family:Consolas,monospace;color:var(--accent);min-width:70px;font-size:.85rem}}
.fps-badge{{font-family:Consolas,monospace;background:#14532d;color:#86efac;padding:3px 8px;border-radius:999px;font-size:.72rem;font-weight:700}}
.fulltext{{background:#0a0c10;border:1px solid #2a2f3d;border-radius:8px;padding:10px;max-height:320px;overflow:auto;
  font-family:Consolas,Monaco,monospace;font-size:11px;line-height:1.4;white-space:pre-wrap;word-break:break-word;color:#d1d5db}}
.tabs{{display:flex;gap:4px;margin-bottom:4px}}
.tabs button{{background:#374151;color:#fff;border:none;padding:6px 10px;border-radius:5px 5px 0 0;cursor:pointer;font-size:.8rem}}
.tabs button.active{{background:var(--accent);color:#000;font-weight:600}}
.tab-panel{{display:none}} .tab-panel.active{{display:block}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:8px;overflow:hidden}}
th,td{{padding:6px 8px;border-bottom:1px solid #2a2f3d;text-align:left;font-size:.8rem}}
th{{background:#12151d;color:var(--muted);font-size:.68rem;text-transform:uppercase}}
tr.bad td:nth-child(4){{color:var(--bad)}} tr.warn td:nth-child(4){{color:var(--warn)}} tr.ok td:nth-child(4){{color:var(--ok)}}
.badge{{display:inline-block;padding:1px 6px;border-radius:999px;font-size:.65rem;font-weight:700}}
.badge.ok{{background:#14532d;color:#86efac}} .badge.warn{{background:#78350f;color:#fcd34d}} .badge.bad{{background:#7f1d1d;color:#fca5a5}}
.note{{background:#1e293b;border-left:3px solid var(--warn);padding:8px 10px;border-radius:5px;font-size:.82rem;margin-bottom:8px}}
.mono{{font-family:Consolas,monospace;font-size:.72rem;color:var(--muted);word-break:break-all}}
#skel-status{{color:var(--muted);font-size:.78rem;margin-top:4px}}
.card h3{{margin:0 0 4px;color:var(--accent);font-size:.72rem;text-transform:uppercase}}
.spark{{max-width:150px}}
</style></head><body>
<div class="wrap">
<h1>GR2 Animation Comparison</h1>
<p class="sub">Generated {now} &mdash; Reference vs Test</p>

<div class="summary-box">
<p><strong>Quick summary:</strong> {len(bad)} critical issues, {len(warn)} warnings. Animation {duration:.1f}s.
Camera-cut problem zone ~{CAMERA_CUT_T}s. GLB files contain <strong>118 bones, no body mesh</strong> — skeleton-only animation.</p>
</div>

<h2>Critical issues</h2>
<ul class="issues">{issue_items}</ul>

<h2>3D skeleton viewport</h2>
<div class="note">
  Preview is a lightweight stick-figure viewer (not Blender). Now baked/played at <strong>{SKELETON_FPS} FPS</strong> to match your anim.
  Free camera: LMB orbit · RMB/Shift pan · wheel zoom. Green = reference, Red = Blender, Yellow = IK.
</div>
<div class="viewer-wrap">
  <div class="viewer-grid">
    <div class="viewer-panel"><canvas id="skel-ref"></canvas></div>
    <div class="viewer-panel"><canvas id="skel-test"></canvas></div>
  </div>
  <div class="controls">
    <button id="skel-play">&#9654; Play</button>
    <button class="sec" id="skel-pause">Pause</button>
    <button class="sec" id="skel-cut">Jump to camera cut ({CAMERA_CUT_T}s)</button>
    <button class="sec" id="skel-fit">Fit to frame</button>
    <button class="sec" id="skel-reset">Reset view</button>
    <input type="range" id="skel-slider" min="0" max="100" value="0"/>
    <span class="time-display" id="skel-time">0.00s</span>
    <span class="fps-badge" id="skel-fps">{SKELETON_FPS} FPS</span>
  </div>
  <div id="skel-status">Loading 3D skeleton...</div>
</div>

<h2>Full readable report &amp; decoded file contents</h2>
<div class="tabs">
  <button class="active" data-tab="tab-report">Full report</button>
  <button data-tab="tab-ref">Decoded reference GLB</button>
  <button data-tab="tab-test">Decoded test GLB</button>
</div>
<div id="tab-report" class="tab-panel active"><pre class="fulltext">{html.escape(full_text)}</pre></div>
<div id="tab-ref" class="tab-panel"><pre class="fulltext">{html.escape(skeleton_json.get('_ref_decode',''))}</pre></div>
<div id="tab-test" class="tab-panel"><pre class="fulltext">{html.escape(skeleton_json.get('_test_decode',''))}</pre></div>

<h2>GR2 files</h2>
<div class="grid">{gr2_cards}</div>

<h2>Track comparison table</h2>
<table><thead><tr><th>Bone</th><th>Channel</th><th>Keys</th><th>Max error</th><th>Time</th><th>Status</th><th>Chart</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</div>

<script>window.SKELETON_DATA={skel_json};</script>
<script>{viewer_js}</script>
<script>
document.querySelectorAll('.tabs button').forEach(btn=>{{
  btn.onclick=()=>{{
    document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }};
}});
</script>
</body></html>"""


def ensure_glb(src: Path, dest: Path, progress=None) -> Path:
    """Return a GLB path for comparison. Converts GR2 via Divine if needed."""
    src = Path(src)
    dest = Path(dest)
    if src.suffix.lower() == ".glb":
        return src
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    if progress:
        progress(f"Converting {src.name} → GLB...")
    ok, msg = convert_gr2_to_glb(src, dest)
    if not ok or not dest.exists():
        raise RuntimeError(f"Could not convert {src.name}: {msg}")
    return dest


def run_compare(
    ref_src: Path,
    test_src: Path,
    *,
    progress=None,
    out_html: Path | None = None,
    out_txt: Path | None = None,
    camera_cut: float = CAMERA_CUT_T,
) -> dict:
    """
    Compare two GR2/GLB animation files.
    progress(msg, pct=None) optional callback.
    Returns dict with report paths, duration, diffs summary, skeleton_json meta.
    """
    def prog(msg, pct=None):
        if progress:
            progress(msg, pct)

    ref_src = Path(ref_src)
    test_src = Path(test_src)
    if not ref_src.exists():
        raise FileNotFoundError(f"Reference not found: {ref_src}")
    if not test_src.exists():
        raise FileNotFoundError(f"Test not found: {test_src}")

    out_html = Path(out_html or REPORT_HTML)
    out_txt = Path(out_txt or REPORT_TXT)
    work = SCRIPT_DIR / "_work"
    work.mkdir(exist_ok=True)

    prog("Preparing files...", 2)
    ref_glb = ensure_glb(ref_src, work / "ref.glb", progress=lambda m, p=None: prog(m, 5))
    test_glb = ensure_glb(test_src, work / "test.glb", progress=lambda m, p=None: prog(m, 10))

    # Also keep copies named for the old viewer paths if useful
    # (report embeds data; no need for lslib.glb/blender.glb)

    gr2_scans = []
    for label, path in [("reference", ref_src), ("test", test_src)]:
        if path.suffix.lower() == ".gr2":
            if scan := scan_gr2(label, path):
                gr2_scans.append(scan)
        else:
            gr2_scans.append(Gr2Scan(label, path, path.stat().st_size, ["(glb)"]))

    prog("Reading animation duration...", 15)
    duration = max(get_anim_duration(ref_glb), get_anim_duration(test_glb))

    prog(f"Baking reference skeleton @ {SKELETON_FPS} FPS ({duration:.1f}s)...", 20)
    ref_skel = export_skeleton_data(ref_glb, duration)
    prog("Baking test skeleton...", 45)
    test_skel = export_skeleton_data(test_glb, duration)

    prog("Decoding tracks...", 60)
    ref_decode = decode_glb_text(ref_glb, f"REFERENCE ({ref_src.name})")
    test_decode = decode_glb_text(test_glb, f"TEST ({test_src.name})")

    prog("Comparing bone tracks...", 75)
    ref_tracks = get_tracks(ref_glb)
    test_tracks = get_tracks(test_glb)
    diffs = []
    for bone in FOCUS_BONES:
        for prop in ("translation", "rotation"):
            if d := compare_track(bone, prop, ref_tracks, test_tracks):
                diffs.append(d)

    prog("Building report...", 90)
    # Temporarily override camera cut for this run via skeleton json only
    global CAMERA_CUT_T
    old_cut = CAMERA_CUT_T
    CAMERA_CUT_T = camera_cut
    try:
        full_text = build_text_report(gr2_scans, diffs, duration, ref_decode, test_decode)
        skeleton_json = {
            "duration": duration,
            "cutT": camera_cut,
            "fps": SKELETON_FPS,
            "ref": ref_skel,
            "test": test_skel,
            "bodyBones": ref_skel.get("bodyBones", []),
            "ikBones": ref_skel.get("ikBones", []),
            "_ref_decode": ref_decode,
            "_test_decode": test_decode,
        }
        viewer_js = VIEWER_JS.read_text(encoding="utf-8") if VIEWER_JS.exists() else ""
        out_html.write_text(
            build_html(gr2_scans, diffs, duration, full_text, skeleton_json, viewer_js),
            encoding="utf-8",
        )
        out_txt.write_text(full_text, encoding="utf-8")
    finally:
        CAMERA_CUT_T = old_cut

    bad = sum(1 for d in diffs if d.status == "bad")
    warn = sum(1 for d in diffs if d.status == "warn")
    prog("Done.", 100)
    return {
        "html": str(out_html),
        "txt": str(out_txt),
        "duration": duration,
        "bad": bad,
        "warn": warn,
        "ok": len(diffs) - bad - warn,
        "ref": str(ref_src),
        "test": str(test_src),
    }


def main() -> int:
    print("BG3 GR2 Animation Compare")
    print("=" * 40)
    print("Tip: run START_COMPARE_UI.bat for file picker in browser.")
    print()

    def cli_progress(msg, pct=None):
        if pct is not None:
            print(f"[{pct:3.0f}%] {msg}")
        else:
            print(msg)

    try:
        result = run_compare(REFERENCE_GR2, TEST_GR2, progress=cli_progress)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"\nHTML: {result['html']}")
    print(f"Text: {result['txt']}")
    print(f"Summary: {result['bad']} bad, {result['warn']} warn, {result['ok']} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
