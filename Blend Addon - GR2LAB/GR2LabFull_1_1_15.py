bl_info = {
    "name": "GR2 LAB",
    "author": "GR2 Lab",
    "version": (1, 1, 15),
    "blender": (3, 6, 0),
    "location": "File > Import/Export",
    "description": "Import game GR2 and export encoded GR2 via local GR2 Lab (decode, B-spline densify, encode)",
    "category": "Import-Export",
}

import json
import math
import os
import socket
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Quaternion, Vector

# ---------------------------------------------------------------------------
# GR2 Lab HTTP API
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:8765"
GR2LAB_PORT = 8765
TIMEOUT_SEC = 120
DISCOVER_HTTP_TIMEOUT = 0.4
DISCOVER_PORT_TIMEOUT = 0.12


class Gr2LabApiError(RuntimeError):
    def __init__(self, message: str, *, payload: Optional[dict] = None):
        super().__init__(message)
        self.payload = payload or {}


class Gr2LabProgress:
    """Header progress + status text. UI paints when a modal op yields (TIMER)."""

    def __init__(self, context):
        self._context = context
        self._wm = context.window_manager
        self._active = False

    def begin(self, maximum: int = 100) -> None:
        self._wm.progress_begin(0, max(1, int(maximum)))
        self._active = True

    def step(self, value: int, label: str = "", *, force_ui: bool = False) -> None:
        if not self._active:
            return
        self._wm.progress_update(int(value))
        if label:
            try:
                self._wm.status_text_set(f"GR2 LAB: {label}")
            except (AttributeError, TypeError):
                pass
        _redraw_windows(self._context)

    def end(self) -> None:
        if self._active:
            self._wm.progress_end()
            self._active = False
        try:
            self._wm.status_text_set(None)
        except (AttributeError, TypeError):
            try:
                self._wm.status_text_set("")
            except (AttributeError, TypeError):
                pass
        _redraw_windows(self._context)


def _redraw_windows(context, *, swap: bool = False) -> None:
    # tag_redraw only — never redraw_timer inside a blocking operator (causes freezes).
    wm = context.window_manager
    for window in wm.windows:
        screen = window.screen
        if not screen:
            continue
        for area in screen.areas:
            area.tag_redraw()

def _base_url(url: str) -> str:
    return (url or DEFAULT_BASE_URL).rstrip("/")


def _build_url(base_url: str, path: str) -> str:
    """Join base + API path; encode spaces/parens in filenames and query values."""
    if path.startswith("http://") or path.startswith("https://"):
        parsed_abs = urlparse(path)
        path = parsed_abs.path + (f"?{parsed_abs.query}" if parsed_abs.query else "")
    if not path.startswith("/"):
        path = "/" + path
    parsed = urlparse(path)
    enc_path = "/".join(quote(part, safe="") for part in parsed.path.split("/"))
    query = urlencode(parse_qsl(parsed.query, keep_blank_values=True), quote_via=quote) if parsed.query else ""
    rel = enc_path + (f"?{query}" if query else "")
    return urljoin(_base_url(base_url) + "/", rel.lstrip("/"))


def _decode_http_text(raw: bytes, *, url: str = "") -> str:
    """Decode JSON HTTP bodies; never treat GR2 binary as UTF-8 silently."""
    if not raw:
        return ""
    if raw[:1] in (b"{", b"["):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        hint = ""
        if _looks_like_gr2_binary(raw):
            hint = " (response looks like a .GR2 file, not JSON — is GR2 Lab running on the right port?)"
        where = f" from {url}" if url else ""
        raise Gr2LabApiError(
            f"GR2 Lab returned non-text data{where}: {e}{hint}"
        ) from e


def _looks_like_gr2_binary(raw: bytes) -> bool:
    """Heuristic: game GR2 is binary and fails UTF-8 at bytes 0-1."""
    if not raw or raw[:1] in (b"{", b"["):
        return False
    try:
        raw.decode("utf-8")
        return False
    except UnicodeDecodeError as e:
        return "position 0" in str(e) or "position 0-1" in str(e)


def _parse_gr2lab_bytes(raw: bytes) -> dict:
    if not raw:
        raise Gr2LabApiError("Empty .gr2lab response from GR2 Lab")
    if _looks_like_gr2_binary(raw):
        raise Gr2LabApiError(
            "GR2 Lab returned a binary .GR2 instead of .gr2lab JSON. "
            "Restart Start_GR2_Lab.bat and use File > Import > GR2 (via GR2 Lab)."
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise Gr2LabApiError(f"Cannot read .gr2lab as UTF-8: {e}") from e
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise Gr2LabApiError(f"Invalid .gr2lab JSON from server: {e}") from e
    if doc.get("format") != "gr2lab":
        raise Gr2LabApiError(f"Expected gr2lab format, got {doc.get('format')!r}")
    return doc


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: Optional[dict] = None,
    timeout: float = TIMEOUT_SEC,
) -> dict:
    url = _build_url(base_url, path)
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = _decode_http_text(raw, url=url)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(detail)
            msg = payload.get("error") or payload.get("message") or detail
            raise Gr2LabApiError(str(msg), payload=payload) from e
        except json.JSONDecodeError:
            raise Gr2LabApiError(f"HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise Gr2LabApiError(
            f"Cannot reach GR2 Lab at {_base_url(base_url)} вЂ” start Start_GR2_Lab.bat first.\n{e}"
        ) from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise Gr2LabApiError(f"Invalid JSON from {url}: {text[:200]}") from e


def _request_bytes(
    base_url: str,
    path: str,
    *,
    timeout: float = TIMEOUT_SEC,
) -> bytes:
    url = _build_url(base_url, path)
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise Gr2LabApiError(f"HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise Gr2LabApiError(
            f"Cannot reach GR2 Lab at {_base_url(base_url)} вЂ” start Start_GR2_Lab.bat first.\n{e}"
        ) from e


def ping(base_url: str = DEFAULT_BASE_URL) -> dict:
    return _request_json(base_url, "/api/status")


def _looks_like_gr2lab_status(data: dict) -> bool:
    return bool(data.get("ok")) and "in_gr2" in data and "out_files" in data


def _probe_gr2lab(base_url: str, *, timeout: float = DISCOVER_HTTP_TIMEOUT) -> Optional[dict]:
    try:
        data = _request_json(base_url, "/api/status", timeout=timeout)
        if _looks_like_gr2lab_status(data):
            return data
    except Exception:
        pass
    return None


def _local_ipv4_addresses() -> List[str]:
    addrs: set = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        addrs.add(sock.getsockname()[0])
        sock.close()
    except OSError:
        pass
    return sorted(addrs)


def _subnet_host_ips() -> List[str]:
    hosts: set = set()
    for local in _local_ipv4_addresses():
        parts = local.split(".")
        if len(parts) != 4:
            continue
        prefix = ".".join(parts[:3])
        for last in range(1, 255):
            hosts.add(f"{prefix}.{last}")
    return sorted(hosts)


def _port_open(host: str, port: int, timeout: float) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def discover_gr2lab_servers() -> List[Tuple[str, dict]]:
    """Find GR2 Lab on this PC (127.0.0.1) or another machine on the LAN."""
    found: Dict[str, dict] = {}
    order: List[str] = []

    def add(base: str, status: dict) -> None:
        base = base.rstrip("/")
        if base not in found:
            found[base] = status
            order.append(base)

    local = _probe_gr2lab(DEFAULT_BASE_URL)
    if local:
        add(DEFAULT_BASE_URL, local)
        return [(u, found[u]) for u in order]

    open_hosts: List[str] = []
    candidates = _subnet_host_ips()
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {
            pool.submit(_port_open, host, GR2LAB_PORT, DISCOVER_PORT_TIMEOUT): host
            for host in candidates
        }
        for fut in as_completed(futures):
            if fut.result():
                open_hosts.append(futures[fut])

    for host in sorted(open_hosts):
        base = f"http://{host}:{GR2LAB_PORT}"
        status = _probe_gr2lab(base)
        if status:
            add(base, status)

    return [(u, found[u]) for u in order]


def upload_gr2(base_url: str, gr2_path: str) -> dict:
    path = Path(gr2_path)
    if not path.is_file():
        raise Gr2LabApiError(f"GR2 not found: {gr2_path}")
    boundary = uuid.uuid4().hex
    filename = path.name
    payload = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii") + payload + f"\r\n--{boundary}--\r\n".encode("ascii")
    url = urljoin(_base_url(base_url) + "/", "api/upload")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=TIMEOUT_SEC) as resp:
            raw = resp.read()
            text = _decode_http_text(raw, url=url)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise Gr2LabApiError(f"Upload failed HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise Gr2LabApiError(f"Upload failed - is GR2 Lab running? {e}") from e
    data = json.loads(text)
    if not data.get("ok"):
        raise Gr2LabApiError(data.get("error") or "Upload failed", payload=data)
    return data


def decode_gr2(
    base_url: str,
    filename: str,
    *,
    body_base: Optional[str] = None,
) -> dict:
    body: Dict[str, Any] = {"file": filename, "dae": False}
    if body_base:
        body["body_base"] = body_base
    data = _request_json(base_url, "/api/decode", method="POST", body=body)
    if not data.get("ok"):
        raise Gr2LabApiError(data.get("error") or "Decode failed", payload=data)
    return data


def fetch_body_bases(
    base_url: str,
    *,
    source: Optional[str] = None,
) -> dict:
    """List Body_Base/*.GR2 and optional auto-match for a source animation name."""
    path = "/api/body-bases"
    if source:
        path = f"/api/body-bases?source={quote(Path(source).name)}"
    data = _request_json(base_url, path)
    if not data.get("ok"):
        raise Gr2LabApiError(data.get("error") or "body-bases failed", payload=data)
    return data


_BODY_BASE_ENUM_CACHE: Dict[str, List[Tuple[str, str, str]]] = {}


def _import_body_base_enum_items(self, context) -> List[Tuple[str, str, str]]:
    key = Path(self.filepath).name.lower() if getattr(self, "filepath", None) else ""
    if key not in _BODY_BASE_ENUM_CACHE:
        items: List[Tuple[str, str, str]] = [
            ("AUTO", "Auto (match filename)", "Let GR2 Lab pick from the GR2 name"),
        ]
        try:
            src = Path(self.filepath).name if self.filepath else None
            data = fetch_body_bases(_server_url(context), source=src)
            matched = (data.get("matched") or "").strip()
            for b in data.get("bases") or []:
                n = (b.get("name") if isinstance(b, dict) else str(b)).strip()
                if not n or n == "AUTO":
                    continue
                tip = "Suggested for this GR2" if n == matched else "Body_Base bind pose"
                items.append((n, n, tip))
        except Exception:
            pass
        _BODY_BASE_ENUM_CACHE[key] = items
    return _BODY_BASE_ENUM_CACHE.get(key) or [
        ("AUTO", "Auto (match filename)", "Let GR2 Lab pick from the GR2 name"),
    ]


def _sync_import_body_base(op, context) -> None:
    """Pick enum default once per selected filepath (import dialog)."""
    src = Path(op.filepath).name if op.filepath else ""
    if getattr(op, "_bb_synced_for", None) == src:
        return
    op._bb_synced_for = src
    op._bb_hint = ""
    items = _import_body_base_enum_items(op, context)
    ids = {i[0] for i in items}
    prefs = context.preferences.addons[__name__].preferences
    pref_bb = (prefs.body_base or "").strip()
    try:
        data = fetch_body_bases(_server_url(context), source=src or None)
        matched = (data.get("matched") or "").strip()
    except Exception as ex:
        op._bb_hint = str(ex)[:120]
        matched = ""
    if pref_bb and pref_bb in ids:
        op.body_base = pref_bb
        op._bb_hint = f"Using preference base {pref_bb}"
    elif matched and matched in ids:
        op.body_base = matched
        op._bb_hint = f"Matched {matched} for {src or 'this GR2'}"
    elif src and not matched:
        op.body_base = "AUTO"
        op._bb_hint = f"No auto-match for {src} — pick the race Body_Base"
    else:
        op.body_base = "AUTO"


def _resolve_body_base_choice(choice: str, prefs_base: str = "") -> Optional[str]:
    bb = (choice or "").strip()
    if bb and bb != "AUTO":
        return bb
    pref = (prefs_base or "").strip()
    return pref or None


def fetch_gr2lab_doc(base_url: str, run_id: str) -> dict:
    """Build densified .gr2lab on server, then download it."""
    rid = quote(str(run_id), safe="")
    info = _request_json(
        base_url,
        "/api/gr2lab/export",
        method="POST",
        body={"from_run": run_id},
    )
    if not info.get("ok"):
        raise Gr2LabApiError(info.get("error") or "gr2lab export failed", payload=info)
    download = info.get("download_url")
    if not download:
        fn = quote(str(info.get("filename") or "clip.gr2lab"), safe="")
        download = f"/api/file/{fn}?run={rid}&kind=files"
    raw = _request_bytes(base_url, download)
    return _parse_gr2lab_bytes(raw)


def import_gr2lab_to_dump(base_url: str, run_id: str, doc: dict) -> dict:
    data = _request_json(
        base_url,
        "/api/gr2lab/import",
        method="POST",
        body={"from_run": run_id, "doc": doc},
    )
    if not data.get("ok"):
        raise Gr2LabApiError(data.get("error") or "Import to dump failed", payload=data)
    return data


def encode_gr2(base_url: str, run_id: str) -> dict:
    data = _request_json(
        base_url,
        "/api/encode",
        method="POST",
        body={"from_run": run_id},
    )
    if not data.get("ok"):
        raise Gr2LabApiError(data.get("error") or "Encode failed", payload=data)
    return data


def download_encoded_gr2(base_url: str, download_url: str, dest_path: str) -> str:
    data = _request_bytes(base_url, download_url)
    out = Path(dest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return str(out.resolve())


def import_gr2_pipeline(
    base_url: str,
    gr2_path: str,
    *,
    body_base: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
) -> tuple[dict, dict]:
    """Upload → decode → densified gr2lab. Returns (decode_info, gr2lab_doc)."""
    if progress:
        progress.step(5, f"Uploading {Path(gr2_path).name}…")
    up = upload_gr2(base_url, gr2_path)
    if progress:
        progress.step(28, "Decoding on GR2 Lab…")
    dec = decode_gr2(base_url, up["name"], body_base=body_base)
    run_id = dec["run_id"]
    if progress:
        progress.step(48, "Densifying animation for Blender…")
    doc = fetch_gr2lab_doc(base_url, run_id)
    doc["run_id"] = run_id
    if progress:
        progress.step(58, "Preparing Blender rig…")
    return dec, doc


def _verify_pack_run(base_url: str, run_id: str) -> None:
    """Ensure the decode dump folder still exists on GR2 Lab."""
    status = ping(base_url)
    runs = status.get("decode_runs") or []
    ids = {str(r.get("id") or "") for r in runs}
    if run_id in ids:
        return
    raise Gr2LabApiError(
        f"Decode dump “{run_id}” not found on GR2 Lab. "
        "Re-import the GR2 via File → Import → GR2 (via GR2 Lab), then export again."
    )


def pack_encode_gr2(
    base_url: str,
    run_id: str,
    doc: dict,
    dest_gr2_path: str,
    *,
    source_gr2: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
) -> dict:
    """Apply gr2lab + encode atomically on the server (matches site workflow)."""
    if progress:
        progress.step(80, f"Verifying dump {run_id}…")
    _verify_pack_run(base_url, run_id)
    body: Dict[str, Any] = {"from_run": run_id, "doc": doc}
    src = (source_gr2 or "").strip()
    if src:
        body["file"] = Path(src).name
    if progress:
        progress.step(86, "Applying + encoding on GR2 Lab…")
    data = _request_json(
        base_url,
        "/api/gr2lab/pack-encode",
        method="POST",
        body=body,
    )
    if not data.get("ok"):
        raise Gr2LabApiError(
            data.get("error") or "Pack-encode failed",
            payload=data,
        )
    url = data.get("download_url") or ""
    if not url:
        raise Gr2LabApiError("Pack-encode response missing download_url", payload=data)
    if progress:
        progress.step(94, f"Saving {Path(dest_gr2_path).name}…")
    out = download_encoded_gr2(base_url, url, dest_gr2_path)
    data["saved_path"] = out
    if progress:
        progress.step(99, "Export done")
    return data


def export_gr2_pipeline(
    base_url: str,
    run_id: str,
    doc: dict,
    dest_gr2_path: str,
    *,
    source_gr2: Optional[str] = None,
    gr2lab_path: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
) -> dict:
    """Apply gr2lab → encode → download GR2 (uses server pack-encode)."""
    if gr2lab_path and Path(gr2lab_path).is_file():
        doc = json.loads(Path(gr2lab_path).read_text(encoding="utf-8"))
    doc = dict(doc)
    doc["run_id"] = run_id
    return pack_encode_gr2(
        base_url,
        run_id,
        doc,
        dest_gr2_path,
        source_gr2=source_gr2,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# GR2LAB bridge (armature + F-curves)
# ---------------------------------------------------------------------------

FORMAT_ID = "gr2lab"
FPS_DEFAULT = 30.0
TIME_EPS = 1e-4  # snap F-curve seconds back onto template knot times
NOOP_POS_EPS = 2e-4
NOOP_QUAT_DOT_EPS = 0.999
# Whole-character placement — keep template unless user-affected + real delta.
PROTECTED_ROOT_BONES = frozenset({"Dummy_Root", "Root_M"})

# Blender Z-up Dummy_Root rest is Rx(+90°). Body_Base templates sometimes store that
# as an orientation track; applying it as Granny anim tips the character 90° on site.
_RX90_QUAT = (0.7071067811865475, 0.0, 0.0, 0.7071067811865475)
_RX_NEG90_QUAT = (-0.7071067811865475, 0.0, 0.0, 0.7071067811865475)


def _quat_dot_abs(a, b) -> float:
    return abs(
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )


def _is_blender_zup_root_ori(q) -> bool:
    """True when quat is ~Rx(±90°) — Blender Z-up Dummy_Root rest, not Granny anim."""
    if not q or len(q) < 4:
        return False
    return (
        _quat_dot_abs(q, _RX90_QUAT) >= 0.995
        or _quat_dot_abs(q, _RX_NEG90_QUAT) >= 0.995
    )


def _identity_ori_channel() -> Dict[str, Any]:
    return {
        "edit_mode": "none",
        "editable": False,
        "times": [0.0],
        "values": [[0.0, 0.0, 0.0, 1.0]],
        "note": "identity orientation",
    }


def _sanitize_dummy_root_ori(ori: Dict[str, Any]) -> Dict[str, Any]:
    """Drop false Blender Z-up Rx(90°) Dummy_Root anim — keep Granny identity."""
    vals = ori.get("values") or []
    if not vals:
        return _identity_ori_channel()
    if all(_is_blender_zup_root_ori(v) for v in vals):
        return _identity_ori_channel()
    return ori


def _uniform_time_grid(duration: float, fps: float) -> List[float]:
    dur = max(0.0, float(duration))
    rate = max(1.0, float(fps))
    n = max(2, int(dur * rate) + 1)
    if dur <= 1e-12:
        return [0.0]
    return [i * dur / (n - 1) for i in range(n)]


def _lerp_sample_n(
    knots: Sequence[float], controls: Sequence[Sequence[float]], t: float
) -> List[float]:
    if not knots or not controls:
        return []
    if len(knots) == 1 or t <= knots[0]:
        return list(controls[0])
    if t >= knots[-1]:
        return list(controls[-1])
    for i in range(len(knots) - 1):
        if knots[i] <= t <= knots[i + 1]:
            span = knots[i + 1] - knots[i]
            u = 0.0 if span <= 1e-12 else (t - knots[i]) / span
            a, b = controls[i], controls[i + 1]
            return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(len(a))]
    return list(controls[0])


def _ensure_hemi_quats(controls: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = [list(controls[0])]
    for q in controls[1:]:
        prev = out[-1]
        dot = sum(float(prev[i]) * float(q[i]) for i in range(4))
        if dot < 0:
            out.append([-float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])])
        else:
            out.append(list(q))
    return out


def _pad_knots(knots: Sequence[float], degree: int) -> List[float]:
    t0 = float(knots[0])
    t1 = float(knots[-1])
    return [t0] + [float(k) for k in knots] + [t1] * int(degree)


def _bspline_span(u_vec: Sequence[float], degree: int, n_ctrl: int, t: float) -> int:
    if t >= u_vec[n_ctrl]:
        return n_ctrl - 1
    if t <= u_vec[degree]:
        return degree
    low, high = degree, n_ctrl
    mid = (low + high) // 2
    while t < u_vec[mid] or t >= u_vec[mid + 1]:
        if t < u_vec[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _sample_bspline_py(
    degree: int,
    knots: Sequence[float],
    controls: Sequence[Sequence[float]],
    t: float,
    *,
    normalize: bool = False,
) -> List[float]:
    """Pure-Python Granny-style Deg2+ B-spline (no DLL). Deg≤1 → lerp."""
    n = len(knots)
    if n == 0 or not controls:
        return []
    dim = len(controls[0])
    degree = int(degree)
    if degree <= 1 or n < degree + 1:
        return _lerp_sample_n(knots, controls, t)
    t0, t1 = float(knots[0]), float(knots[-1])
    if float(t) <= t0:
        result = list(controls[0])
    elif float(t) >= t1:
        result = list(controls[-1])
    else:
        u_vec = _pad_knots(knots, degree)
        i = _bspline_span(u_vec, degree, n, float(t))
        d = [list(controls[i - degree + j]) for j in range(degree + 1)]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                left = u_vec[i - degree + j]
                right = u_vec[i + 1 + j - r]
                denom = right - left
                alpha = 0.0 if abs(denom) < 1e-20 else (float(t) - left) / denom
                d[j] = [
                    (1.0 - alpha) * d[j - 1][c] + alpha * d[j][c] for c in range(dim)
                ]
        result = d[degree]
    if normalize and dim == 4:
        nrm = math.sqrt(sum(x * x for x in result)) or 1.0
        result = [x / nrm for x in result]
    return result


def _sample_bspline_quat_py(
    degree: int,
    knots: Sequence[float],
    quats: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    return _sample_bspline_py(
        degree, knots, _ensure_hemi_quats(quats), t, normalize=True
    )


def _ensure_doc_bspline_dense(doc: Dict[str, Any], fps: float) -> Dict[str, Any]:
    """Densify Deg≥2 channels onto an fps grid so Blender LINEAR looks game-smooth.

    Runs client-side even if the .gr2lab was not Download-for-Blender densified.
    Already-dense Deg1 channels are left alone. Root_M ori is marked display_densified.
    """
    duration = float(doc.get("duration") or 0.0)
    grid = _uniform_time_grid(duration, fps)
    fps_i = int(round(float(fps)))
    densified = 0
    for slot in doc.get("tracks") or []:
        name = slot.get("name") or ""
        for ch_key, is_quat in (("position", False), ("orientation", True)):
            ch = slot.get(ch_key) or {}
            mode = ch.get("edit_mode") or "none"
            if mode in ("none", "constant"):
                continue
            note = str(ch.get("note") or "")
            if "bspline_densified" in note or "k16_display_densified" in note:
                continue
            ts = [float(x) for x in (ch.get("times") or [])]
            vs = [list(map(float, v)) for v in (ch.get("values") or [])]
            if len(ts) < 2 or len(vs) != len(ts):
                continue
            # Already ~fps dense Deg1 — skip
            deg = int(ch.get("degree") or 0)
            if deg <= 1 and len(ts) >= max(8, int(duration * float(fps) * 0.85)):
                continue
            if name == "Dummy_Root" and ch_key == "orientation":
                continue
            # Save/24: Deg≥2 k16/float/quat → B-spline Deg1 float (not lerp).
            if deg < 2:
                deg = 2 if mode in ("quat_k16", "k16_offsets") else 1
            if deg < 2:
                continue
            samples = (
                [_sample_bspline_quat_py(deg, ts, vs, t) for t in grid]
                if is_quat
                else [_sample_bspline_py(deg, ts, vs, t) for t in grid]
            )
            display = name in ("Dummy_Root", "Root_M") or mode == "k16_offsets"
            slot[ch_key] = {
                "edit_mode": "float_controls",
                "editable": True,
                "format": 1,
                "capacity": len(grid),
                "times": list(grid),
                "values": samples,
                "degree": 1,
                "note": f"bspline_densified@{fps_i}",
                "source_interp": "bspline_dense",
                "display_densified": True if display else ch.get("display_densified"),
            }
            densified += 1
    doc["source_interp"] = "bspline_dense"
    doc["densify_fps"] = float(fps)
    doc["densified_channels"] = int(doc.get("densified_channels") or 0) + densified
    doc["blender_import_densified"] = densified
    return {"densified": densified, "grid": len(grid), "fps": float(fps)}


PROP_DOC = "gr2lab_json"
PROP_RUN = "gr2lab_run_id"
PROP_TEMPLATE_PATH = "gr2lab_template_path"
WM_LAST_TEMPLATE = "gr2lab_last_template"
WM_LAST_PACK_RUN = "gr2lab_last_pack_run"
WM_LAST_SOURCE_GR2 = "gr2lab_last_source_gr2"


def _quat_gr_to_bl(q: Sequence[float]) -> Quaternion:
    """Granny/Three [x,y,z,w] в†’ Blender Quaternion(w,x,y,z)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return Quaternion((w, x, y, z))


def _quat_bl_to_gr(q: Quaternion) -> List[float]:
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def _trs_matrix(t: Sequence[float], r: Sequence[float], s: Sequence[float]) -> Matrix:
    loc = Vector((float(t[0]), float(t[1]), float(t[2])))
    rot = _quat_gr_to_bl(r)
    scl = Vector((float(s[0]), float(s[1]), float(s[2])))
    return Matrix.LocRotScale(loc, rot, scl)


def _mat_to_trs(m: Matrix) -> Tuple[List[float], List[float], List[float]]:
    loc, rot, scl = m.decompose()
    return (
        [float(loc.x), float(loc.y), float(loc.z)],
        _quat_bl_to_gr(rot),
        [float(scl.x), float(scl.y), float(scl.z)],
    )


def _stabilize_local_cache_quats(
    local_cache: Dict[float, Dict[str, Tuple[List[float], List[float], List[float]]]],
) -> None:
    """
    Matrix.decompose() picks q or -q arbitrarily each frame. Consecutive sign flips
    look like shake after import (Granny/site lerp without a short-path fix on every path).
    Walk each bone's exported series and keep the short hemisphere.
    """
    if not local_cache:
        return
    frames = sorted(local_cache.keys())
    names = set()
    for fr in frames:
        names.update(local_cache[fr].keys())
    for name in names:
        prev: Optional[Quaternion] = None
        for fr in frames:
            bucket = local_cache.get(fr) or {}
            if name not in bucket:
                continue
            loc, quat_xyzw, scl = bucket[name]
            # Granny xyzw в†’ Blender wxyz
            cur = Quaternion(
                (
                    float(quat_xyzw[3]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                )
            )
            cur.normalize()
            if prev is not None:
                cur.make_compatible(prev)
            prev = cur
            bucket[name] = (loc, _quat_bl_to_gr(cur), scl)


def _make_quat_series_compatible(values: List[List[float]]) -> None:
    """In-place Granny xyzw series в†’ short-arc consecutive keys."""
    prev: Optional[Quaternion] = None
    for i, q in enumerate(values):
        if not q or len(q) < 4:
            continue
        cur = Quaternion((float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        cur.normalize()
        if prev is not None:
            cur.make_compatible(prev)
        prev = cur
        values[i] = _quat_bl_to_gr(cur)


def _load_doc(path: str) -> Dict[str, Any]:
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".gr2"):
        raise ValueError(
            f"{p.name} is a game .GR2 file, not a .gr2lab bridge.\n"
            "Use File > Import > GR2 (via GR2 Lab) for .GR2 files."
        )
    raw = p.read_bytes()
    doc = _parse_gr2lab_bytes(raw)
    if not isinstance(doc.get("skeleton"), list) or not isinstance(doc.get("tracks"), list):
        raise ValueError(f"Invalid GR2LAB file (missing skeleton/tracks): {Path(path).name}")
    return doc


def _bone_order(skeleton: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parents before children."""
    by_name = {b["name"]: b for b in skeleton if b.get("name")}
    remaining = set(by_name.keys())
    ordered: List[Dict[str, Any]] = []
    while remaining:
        progress = False
        for name in list(remaining):
            b = by_name[name]
            p = b.get("parent_name")
            if not p or p not in by_name or p not in remaining:
                ordered.append(b)
                remaining.remove(name)
                progress = True
        if not progress:
            # cycle / missing parent вЂ” append rest
            for name in sorted(remaining):
                ordered.append(by_name[name])
            break
    return ordered


def _rest_worlds(skeleton: List[Dict[str, Any]]) -> Dict[str, Matrix]:
    worlds: Dict[str, Matrix] = {}
    for b in _bone_order(skeleton):
        name = b["name"]
        local = _trs_matrix(
            b.get("rest_translation") or [0, 0, 0],
            b.get("rest_rotation") or [0, 0, 0, 1],
            b.get("rest_scale") or [1, 1, 1],
        )
        p = b.get("parent_name")
        worlds[name] = (worlds[p] @ local) if p in worlds else local
    return worlds


def _bind_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    bind = doc.get("bind") or {}
    bones = bind.get("bones") if isinstance(bind, dict) else None
    if not bones:
        return {}
    return {b["name"]: b for b in bones if b.get("name")}


def _skeleton_with_bind_parents(
    skeleton: List[Dict[str, Any]], bind_by_name: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Override parent_name from bind (Body_Base / LSLib hierarchy) when present."""
    if not bind_by_name:
        return list(skeleton)
    out = []
    for b in skeleton:
        nb = dict(b)
        bb = bind_by_name.get(b.get("name") or "")
        if bb is not None and "parent_name" in bb:
            nb["parent_name"] = bb.get("parent_name")
        out.append(nb)
    return out


def _bind_local_for_bone(
    b: Dict[str, Any], bind_by_name: Dict[str, Dict[str, Any]]
) -> Matrix:
    """Edit-mode / pose basis local matrix: bind when present, else clip rest_*."""
    bb = bind_by_name.get(b["name"])
    if bb:
        return _trs_matrix(
            bb.get("bind_translation") or [0, 0, 0],
            bb.get("bind_rotation") or [0, 0, 0, 1],
            bb.get("bind_scale") or [1, 1, 1],
        )
    return _trs_matrix(
        b.get("rest_translation") or [0, 0, 0],
        b.get("rest_rotation") or [0, 0, 0, 1],
        b.get("rest_scale") or [1, 1, 1],
    )


def _bind_worlds(
    skeleton: List[Dict[str, Any]], bind_by_name: Dict[str, Dict[str, Any]]
) -> Dict[str, Matrix]:
    """FK worlds using skeleton parent_name (already bind-overridden when available)."""
    worlds: Dict[str, Matrix] = {}
    for b in _bone_order(skeleton):
        name = b["name"]
        local = _bind_local_for_bone(b, bind_by_name)
        p = b.get("parent_name")
        worlds[name] = (worlds[p] @ local) if p in worlds else local
    return worlds


def _track_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["name"]: t for t in doc.get("tracks") or [] if t.get("name")}


def _sec_to_frame(t: float, fps: float) -> float:
    return 1.0 + float(t) * float(fps)


def _frame_to_sec(frame: float, fps: float) -> float:
    return max(0.0, (float(frame) - 1.0) / float(fps))


def _snap_times_to_template(
    fc_times: Sequence[float],
    orig_times: Sequence[float],
    *,
    eps: float = TIME_EPS,
) -> List[float]:
    """Map F-curve seconds onto template knots; keep only truly new keys.

    Avoids densifying float/quat grids when secв†”frame float noise would otherwise
    make ``set(orig) | set(fc)`` grow the curve on a no-op round-trip.
    """
    orig = [float(t) for t in orig_times] if orig_times else [0.0]
    if not fc_times:
        return list(orig) if orig else [0.0]
    out: List[float] = []
    seen = set()
    for raw in fc_times:
        t = float(raw)
        nearest = None
        best = None
        for o in orig:
            d = abs(t - o)
            if best is None or d < best:
                best = d
                nearest = o
        use = float(nearest) if nearest is not None and best is not None and best <= eps else t
        key = round(use, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(use)
    return sorted(out) or [0.0]


def _collapse_times_eps(
    times: Sequence[float],
    *,
    prefer: Optional[Sequence[float]] = None,
    eps: float = TIME_EPS,
) -> List[float]:
    """Merge times within eps; prefer frame-grid values when present in a cluster.

    Exact ``set()`` unions of template JSON floats and ``(frame-1)/fps`` create
    near-zero-Δt pairs with inconsistent samples → site/game finger flicker.
    """
    ordered = sorted({float(t) for t in times})
    if not ordered:
        return [0.0]
    pref = [float(t) for t in (prefer or [])]
    out: List[float] = []
    cluster: List[float] = [ordered[0]]

    def flush(cl: List[float]) -> None:
        chosen = cl[-1]
        for p in pref:
            if any(abs(p - c) <= eps for c in cl):
                chosen = p
                break
        if out and abs(chosen - out[-1]) <= eps:
            for p in pref:
                if abs(chosen - p) <= eps:
                    out[-1] = chosen
                    return
            return
        out.append(chosen)

    for t in ordered[1:]:
        if t - cluster[-1] <= eps:
            cluster.append(t)
        else:
            flush(cluster)
            cluster = [t]
    flush(cluster)
    return out or [0.0]


def _lerp_vec3(
    times: Sequence[float],
    values: Sequence[Sequence[float]],
    t: float,
) -> List[float]:
    """Linear sample a vec3 series (matches site linear display)."""
    ts = [float(x) for x in times] if times else [0.0]
    vs = [list(v) for v in values] if values else []
    if not vs:
        return [0.0, 0.0, 0.0]
    if len(ts) == 1 or t <= ts[0]:
        return [float(vs[0][j]) for j in range(3)]
    if t >= ts[-1]:
        return [float(vs[-1][j]) for j in range(3)]
    for i in range(len(ts) - 1):
        if ts[i] <= t <= ts[i + 1]:
            span = ts[i + 1] - ts[i]
            u = 0.0 if span <= 1e-12 else (t - ts[i]) / span
            a, b = vs[i], vs[i + 1]
            return [
                float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(3)
            ]
    return [float(vs[-1][j]) for j in range(3)]


def _k16_pos_matches_template(
    samples: Dict[float, List[float]],
    template_times: Sequence[float],
    template_vals: Sequence[Sequence[float]],
) -> bool:
    """True when exported Granny locals still match dequantized k16 template motion.

    Untouched display keys from import в†’ bias-only export (keep k16_offsets).
    Real multi-key edits that leave the template curve в†’ upgrade to float on site.
    """
    if not template_vals:
        return True
    if not samples:
        return True
    tmpl_t = [float(x) for x in template_times] if template_times else [0.0]
    for t, v in samples.items():
        expected = _lerp_vec3(tmpl_t, template_vals, float(t))
        if not _v3_close(v, expected, NOOP_POS_EPS):
            return False
    return True


def _root_samples_differ_from_template(
    samples: Dict[float, List[float]],
    template_ch: Dict[str, Any],
    *,
    is_quat: bool,
) -> bool:
    """True when sampled motion leaves the template beyond noop eps."""
    tmpl_t = [float(x) for x in (template_ch.get("times") or [0.0])]
    tmpl_v = template_ch.get("values") or []
    if not samples or not tmpl_v:
        return False
    if is_quat:
        for t, v in samples.items():
            expected = _sample_quat_channel(
                {"times": tmpl_t, "values": tmpl_v}, float(t)
            )
            if expected is None or not _quat_close(v, expected, NOOP_QUAT_DOT_EPS):
                return True
        return False
    for t, v in samples.items():
        expected = _lerp_vec3(tmpl_t, tmpl_v, float(t))
        if not _v3_close(v, expected, NOOP_POS_EPS):
            return True
    return False


def _channel_is_display_densified(ch: Dict[str, Any]) -> bool:
    note = str((ch or {}).get("note") or "")
    return (
        "bspline_densified" in note
        or "k16_display_densified" in note
        or bool((ch or {}).get("display_densified"))
    )


def _allow_protected_root_rewrite(
    bone_affected: bool,
    samples: Dict[float, List[float]],
    template_ch: Dict[str, Any],
    *,
    is_quat: bool,
) -> bool:
    """Only intentional sparse root edits — never full bake densify."""
    if not bone_affected:
        return False
    # Densified bridge/template channels are display-only; never rewrite roots.
    if _channel_is_display_densified(template_ch):
        return False
    if not _root_samples_differ_from_template(samples, template_ch, is_quat=is_quat):
        return False
    tmpl_t = [float(t) for t in (template_ch.get("times") or [])]
    samp_t = list(samples.keys())
    # Bake / frame-union densify must not rewrite whole-character placement.
    if _is_mass_grid_densify_times(tmpl_t, samp_t):
        return False
    if len(samp_t) >= max(100, int(len(tmpl_t) * 1.5) + 1):
        return False
    return True


def _is_mass_grid_densify_times(
    template_times: Sequence[float],
    sample_times: Sequence[float],
) -> bool:
    """True when F-curve times are a much denser grid than the template (frame bake)."""
    tmpl = [float(t) for t in template_times] if template_times else []
    samp = [float(t) for t in sample_times] if sample_times else []
    if not tmpl or not samp or len(samp) <= len(tmpl):
        return False
    cap = max(8, int(0.05 * len(tmpl) + 0.5))
    far = 0
    for t in samp:
        best = None
        for o in tmpl:
            d = abs(t - o)
            if best is None or d < best:
                best = d
        if best is not None and best > TIME_EPS:
            far += 1
    return far > cap


def _drop_on_curve_extra_times(
    times: Sequence[float],
    samples: Dict[float, List[float]],
    template_times: Sequence[float],
    template_vals: Sequence[Sequence[float]],
    *,
    is_quat: bool,
) -> List[float]:
    """Keep template knots; drop extras whose sample matches the template curve.

    Stops Spine*_M-style on-curve densify keys from a Blender no-op export.
    """
    if not times:
        return [0.0]
    tmpl_t = [float(t) for t in template_times] if template_times else [0.0]
    tmpl_v = [list(v) for v in template_vals] if template_vals else []
    close = (
        (lambda a, b: _quat_close(a, b, NOOP_QUAT_DOT_EPS))
        if is_quat
        else (lambda a, b: _v3_close(a, b, NOOP_POS_EPS))
    )

    def on_template(t: float) -> bool:
        return any(abs(float(t) - o) <= TIME_EPS for o in tmpl_t)

    def sample_tmpl(t: float) -> List[float]:
        if not tmpl_v:
            return [0.0, 0.0, 0.0, 1.0] if is_quat else [0.0, 0.0, 0.0]
        ts = tmpl_t
        vs = tmpl_v
        if len(ts) == 1 or t <= ts[0]:
            return list(vs[0])
        if t >= ts[-1]:
            return list(vs[-1])
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i + 1]:
                span = ts[i + 1] - ts[i]
                u = 0.0 if span <= 1e-12 else (t - ts[i]) / span
                if is_quat:
                    qa = _quat_gr_to_bl(vs[i])
                    qb = _quat_gr_to_bl(vs[i + 1])
                    qb.make_compatible(qa)
                    return _quat_bl_to_gr(qa.slerp(qb, u))
                a, b = vs[i], vs[i + 1]
                n = min(len(a), len(b), 3)
                return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(n)]
        return list(vs[-1])

    out: List[float] = []
    for t in times:
        tf = float(t)
        if on_template(tf):
            out.append(tf)
            continue
        got = samples.get(tf)
        if got is None:
            continue
        if close(got, sample_tmpl(tf)):
            continue
        out.append(tf)
    return sorted(out) or list(tmpl_t[:1]) or [0.0]


def import_gr2lab_from_doc(
    context,
    doc: Dict[str, Any],
    *,
    fps: Optional[float] = None,
    y_up_display: bool = True,
    template_path: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
) -> bpy.types.Object:
    """Build armature + action from a GR2LAB document dict."""
    fps = float(fps if fps is not None else doc.get("densify_fps") or FPS_DEFAULT)
    # Always densify Deg≥2 → fps grid so Blender LINEAR matches game B-spline,
    # even if the .gr2lab was saved without Download-for-Blender densify.
    densify_info = _ensure_doc_bspline_dense(doc, fps)
    if densify_info.get("densified"):
        print(
            f"GR2LAB: import densified {densify_info['densified']} channels "
            f"@ {densify_info['fps']:.0f}fps ({densify_info['grid']} keys)"
        )
    skeleton = doc.get("skeleton") or []
    tracks = _track_map(doc)
    duration = float(doc.get("duration") or 0.0)
    run_id = doc.get("run_id") or ""
    root_name = doc.get("root") or (skeleton[0]["name"] if skeleton else "Root")
    bind_by_name = _bind_map(doc)
    if not bind_by_name:
        print(
            "GR2LAB warning: no bind section вЂ” Edit mode uses clip t=0 rest "
            "(extract Body_Base bind on the site for a real T-pose)"
        )

    # Hierarchy: prefer Body_Base / LSLib parents from bind (fixes flying IK)
    skeleton = _skeleton_with_bind_parents(skeleton, bind_by_name)

    # Leave any prior Edit/Pose mode without ops that need a VIEW_3D poll.
    prev = context.view_layer.objects.active
    if prev is not None and getattr(prev, "mode", "OBJECT") != "OBJECT":
        _mode_set_safe(context, "OBJECT", active=prev)
    _deselect_all_objects(context)

    arm_data = bpy.data.armatures.new(f"GR2LAB_{run_id or 'rig'}")
    arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
    context.collection.objects.link(arm_obj)
    context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    # Optional display-only Y-up в†’ Z-up (rotate armature object, not keys)
    if y_up_display:
        arm_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    worlds = _bind_worlds(skeleton, bind_by_name)

    _mode_set_safe(context, "EDIT", active=arm_obj)
    edit_bones = arm_data.edit_bones
    created = {}
    for b in _bone_order(skeleton):
        name = b["name"]
        eb = edit_bones.new(name)
        # Unit-length bone along local +Y in rest (Blender bone axis)
        m = worlds.get(name, Matrix.Identity(4))
        head = m.to_translation()
        # Tail: small offset along bone's local Y (Blender convention)
        axis = (m.to_3x3() @ Vector((0.0, 0.05, 0.0)))
        if axis.length < 1e-8:
            axis = Vector((0.0, 0.05, 0.0))
        eb.head = head
        eb.tail = head + axis
        # Align roll via matrix
        eb.matrix = m
        created[name] = eb

    for b in skeleton:
        name = b.get("name")
        p = b.get("parent_name")
        if name in created and p in created:
            created[name].parent = created[p]
            created[name].use_connect = False

    _mode_set_safe(context, "POSE", active=arm_obj)

    # Store bridge doc for export (skeleton parents already bind-corrected in memory;
    # keep original doc JSON but also stamp hierarchy onto a working copy in PROP)
    doc_store = dict(doc)
    doc_store["skeleton"] = skeleton
    if doc_store.get("bind") and isinstance(doc_store["bind"], dict):
        # Keep bind bone parents authoritative
        pass
    arm_obj[PROP_DOC] = json.dumps(doc_store)
    arm_obj[PROP_RUN] = run_id
    try:
        arm_obj[PROP_IMPORT_SNAPSHOT] = json.dumps(
            {"tracks": doc_store.get("tracks") or [], "duration": doc_store.get("duration")}
        )
    except Exception:
        pass
    arm_obj["gr2lab_fps"] = float(fps)
    arm_obj["gr2lab_y_up_display"] = bool(y_up_display)
    arm_obj["gr2lab_has_bind"] = bool(bind_by_name)
    # Smooth download / import densify stamps source_interp=bspline_dense.
    arm_obj["gr2lab_source_interp"] = str(doc.get("source_interp") or "")
    if doc.get("densify_fps") is not None:
        arm_obj["gr2lab_densify_fps"] = float(doc.get("densify_fps"))
    arm_obj["gr2lab_import_densified"] = int(densify_info.get("densified") or 0)
    if template_path:
        arm_obj[PROP_TEMPLATE_PATH] = str(Path(template_path).resolve())
        remember_template_path(template_path)

    # Bind (or clip-rest fallback) local matrices for basis conversion
    rest_local: Dict[str, Matrix] = {}
    for b in skeleton:
        name = b["name"]
        rest_local[name] = _bind_local_for_bone(b, bind_by_name)

    action = bpy.data.actions.new(name=f"GR2LAB_{run_id or 'anim'}")
    if not arm_obj.animation_data:
        arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    scene = context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start = 1
    scene.frame_end = max(1, int(math.ceil(_sec_to_frame(duration, fps))))

    for bone_i, b in enumerate(skeleton):
        name = b["name"]
        if progress and bone_i % max(1, len(skeleton) // 24) == 0:
            progress.step(
                60 + int(38 * bone_i / max(1, len(skeleton))),
                f"Bone keys {bone_i + 1}/{len(skeleton)}: {name}",
            )
        pb = arm_obj.pose.bones.get(name)
        if not pb:
            continue
        pb.rotation_mode = "QUATERNION"
        tr = tracks.get(name) or {}
        rmat = rest_local[name]
        rinv = rmat.inverted()

        pos = tr.get("position") or {}
        ori = tr.get("orientation") or {}
        scl = tr.get("scale") or {}

        # Sample helpers вЂ” linear / slerp (matches site; game needs Degree=1 for these samples)
        def sample_vec(ch, t, n, default):
            ts = [float(x) for x in (ch.get("times") or [])]
            vs = ch.get("values") or []
            if not ts or not vs:
                return list(default)
            if len(ts) == 1:
                return list(vs[0])
            if t <= ts[0]:
                return list(vs[0])
            if t >= ts[-1]:
                return list(vs[-1])
            for i in range(len(ts) - 1):
                if ts[i] <= t <= ts[i + 1]:
                    span = ts[i + 1] - ts[i]
                    u = 0.0 if span <= 1e-12 else (t - ts[i]) / span
                    a, b = vs[i], vs[i + 1]
                    return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(n)]
            return list(vs[-1])

        def sample_quat(ch, t):
            ts = [float(x) for x in (ch.get("times") or [])]
            vs = ch.get("values") or []
            if not ts or not vs:
                return [0.0, 0.0, 0.0, 1.0]
            if len(ts) == 1:
                return list(vs[0])
            if t <= ts[0]:
                return list(vs[0])
            if t >= ts[-1]:
                return list(vs[-1])
            for i in range(len(ts) - 1):
                if ts[i] <= t <= ts[i + 1]:
                    span = ts[i + 1] - ts[i]
                    u = 0.0 if span <= 1e-12 else (t - ts[i]) / span
                    qa = _quat_gr_to_bl(vs[i])
                    qb = _quat_gr_to_bl(vs[i + 1])
                    qb.make_compatible(qa)
                    q = qa.slerp(qb, u)
                    return _quat_bl_to_gr(q)
            return list(vs[-1])

        # For constant channels, only need one key (t=0)
        pos_mode = pos.get("edit_mode") or "none"
        ori_mode = ori.get("edit_mode") or "none"
        scl_mode = scl.get("edit_mode") or "none"

        def channel_times(ch, mode):
            # constant / none: single bias key (Save cannot rewrite those formats).
            # k16_offsets: full dequantized keys for playback (foot IK / Root locomotion).
            # Export collapses untouched k16 display keys back to bias-only.
            if mode in ("constant", "none"):
                return [0.0]
            ts = [float(t) for t in (ch.get("times") or [0.0])]
            return ts if ts else [0.0]

        # Key EACH channel only on its own times вЂ” never union.
        # (Union bloated float position past GR2 capacity on export.)
        # Playback: site uses linear lerp/slerp between knots вЂ” use LINEAR
        # F-curves (not Bezier, not CONSTANT step-hold).
        def insert_keys(data_path, times_list, apply_basis_at_t):
            for t in times_list:
                apply_basis_at_t(t)
                pb.keyframe_insert(data_path=data_path, frame=_sec_to_frame(t, fps))

        def set_basis_at(t):
            pt = sample_vec(pos, t, 3, b.get("rest_translation") or [0, 0, 0])
            qt = sample_quat(ori, t)
            st = sample_vec(scl, t, 3, b.get("rest_scale") or [1, 1, 1])
            key_local = _trs_matrix(pt, qt, st)
            basis = rinv @ key_local
            loc, rot, sc = basis.decompose()
            pb.location = loc
            pb.rotation_quaternion = rot
            pb.scale = sc

        # Always set a rest pose at t=0 first
        set_basis_at(0.0)

        # Position/rotation unlocked вЂ” site upgrades constant/identity when needed.
        insert_keys(
            "location",
            channel_times(pos, pos_mode if pos_mode not in ("none",) else "constant"),
            set_basis_at,
        )
        insert_keys(
            "rotation_quaternion",
            channel_times(ori, ori_mode if ori_mode not in ("none",) else "constant"),
            set_basis_at,
        )

        if scl.get("editable") and scl_mode != "none":
            insert_keys("scale", channel_times(scl, scl_mode), set_basis_at)
        else:
            set_basis_at(0.0)
            pb.keyframe_insert(data_path="scale", frame=_sec_to_frame(0.0, fps))

        # LINEAR в‰€ Granny/site lerpВ·slerp. Avoid default Bezier (overshoot) and
        # CONSTANT (steppy/clingy). Stabilize quat signs against long-path flips.
        _finalize_pose_bone_fcurves(action, name)

        # Scale stays locked (v1). Loc/rot unlocked вЂ” site upgrades formats.
        pb.lock_scale = (True, True, True)
        pb["gr2lab_pos_mode"] = pos_mode
        pb["gr2lab_ori_mode"] = ori_mode
        pb["gr2lab_scl_mode"] = scl_mode
        # Full k16 location keys are display/playback only until the user edits them.
        if pos_mode == "k16_offsets":
            pb["gr2lab_k16_display_keys"] = True

    _mode_set_safe(context, "OBJECT", active=arm_obj)
    scene.frame_set(1)
    if progress:
        progress.step(99, "Import done")
    return arm_obj


def import_gr2lab(
    context,
    filepath: str,
    *,
    fps: float = FPS_DEFAULT,
    y_up_display: bool = True,
    progress: Optional[Gr2LabProgress] = None,
) -> bpy.types.Object:
    if progress:
        progress.step(10, f"Reading {Path(filepath).name}…")
    doc = _load_doc(filepath)
    return import_gr2lab_from_doc(
        context,
        doc,
        fps=fps,
        y_up_display=y_up_display,
        template_path=filepath,
        progress=progress,
    )


def _iter_action_fcurves(action):
    """Yield all F-curves from legacy action.fcurves and Blender 4.4+ channelbags."""
    if not action:
        return
    for fc in action.fcurves:
        yield fc
    if not hasattr(action, "layers"):
        return
    for layer in action.layers:
        for strip in getattr(layer, "strips", []) or []:
            bags = list(getattr(strip, "channelbags", []) or [])
            for bag in bags:
                for fc in getattr(bag, "fcurves", []) or []:
                    yield fc


def _finalize_pose_bone_fcurves(action, bone_name: str) -> None:
    """LINEAR interpolation + quaternion continuity (matches Granny/site lerpВ·slerp)."""
    if not action or not bone_name:
        return
    needle = f'pose.bones["{bone_name}"]'
    quat_fcs: Dict[int, Any] = {}
    for fc in _iter_action_fcurves(action):
        if needle not in fc.data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        if fc.data_path.endswith("rotation_quaternion"):
            quat_fcs[fc.array_index] = fc
    # Make consecutive quat keys take the short arc (w,x,y,z = indices 0..3)
    if not all(i in quat_fcs for i in range(4)):
        return
    w_fc, x_fc, y_fc, z_fc = quat_fcs[0], quat_fcs[1], quat_fcs[2], quat_fcs[3]
    n = len(w_fc.keyframe_points)
    if n < 2 or not (
        len(x_fc.keyframe_points) == n
        and len(y_fc.keyframe_points) == n
        and len(z_fc.keyframe_points) == n
    ):
        return
    prev = Quaternion(
        (
            w_fc.keyframe_points[0].co[1],
            x_fc.keyframe_points[0].co[1],
            y_fc.keyframe_points[0].co[1],
            z_fc.keyframe_points[0].co[1],
        )
    )
    for i in range(1, n):
        cur = Quaternion(
            (
                w_fc.keyframe_points[i].co[1],
                x_fc.keyframe_points[i].co[1],
                y_fc.keyframe_points[i].co[1],
                z_fc.keyframe_points[i].co[1],
            )
        )
        cur.make_compatible(prev)
        w_fc.keyframe_points[i].co[1] = cur.w
        x_fc.keyframe_points[i].co[1] = cur.x
        y_fc.keyframe_points[i].co[1] = cur.y
        z_fc.keyframe_points[i].co[1] = cur.z
        prev = cur


def _build_fcurve_index(action) -> Dict[str, Dict[Tuple[str, int], Any]]:
    """bone_name -> {(data_path, array_index): fcurve}."""
    out: Dict[str, Dict[Tuple[str, int], Any]] = {}
    prefix = 'pose.bones["'
    for fc in _iter_action_fcurves(action):
        dp = fc.data_path
        if not dp.startswith(prefix):
            continue
        end = dp.find('"]', len(prefix))
        if end < 0:
            continue
        bone = dp[len(prefix) : end]
        prop = dp[end + 2 :]
        if prop.startswith("."):
            prop = prop[1:]
        bucket = out.setdefault(bone, {})
        bucket[(prop, int(fc.array_index))] = fc
    return out


def _all_action_frames(action) -> List[float]:
    frames = set()
    for fc in _iter_action_fcurves(action):
        for kp in fc.keyframe_points:
            frames.add(round(float(kp.co[0]), 6))
    return sorted(frames)


def _deselect_all_objects(context) -> None:
    """Deselect without bpy.ops (File Browser / wrong area breaks select_all.poll)."""
    for obj in context.view_layer.objects:
        try:
            obj.select_set(False)
        except RuntimeError:
            pass


def _temp_view3d_override(context, *, active=None, selected=None):
    """Build a VIEW_3D temp_override dict for mode_set / bake when context is thin."""
    win = context.window
    scr = win.screen if win else None
    area = next((a for a in (scr.areas if scr else []) if a.type == "VIEW_3D"), None)
    region = None
    if area is not None:
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        if region is None and area.regions:
            region = area.regions[-1]
    act = active or context.view_layer.objects.active
    sel = list(selected) if selected is not None else ([act] if act else [])
    override = {
        "window": win,
        "screen": scr,
        "area": area,
        "region": region,
        "active_object": act,
        "object": act,
        "selected_objects": sel,
        "selected_editable_objects": sel,
    }
    return {k: v for k, v in override.items() if v is not None}


def _mode_set_safe(context, mode: str, *, active=None) -> None:
    """object.mode_set with VIEW_3D override fallback (import from File Browser)."""
    if active is not None:
        context.view_layer.objects.active = active
        try:
            active.select_set(True)
        except RuntimeError:
            pass
    try:
        bpy.ops.object.mode_set(mode=mode)
        return
    except RuntimeError:
        pass
    override = _temp_view3d_override(context, active=active)
    with context.temp_override(**override):
        bpy.ops.object.mode_set(mode=mode)


def _resolve_action(arm_obj) -> Any:
    """Active action on armature (NLA strip fallback when action slot is empty)."""
    ad = arm_obj.animation_data if arm_obj else None
    if not ad:
        return None
    if ad.action:
        return ad.action
    if ad.nla_tracks:
        for track in ad.nla_tracks:
            if track.mute:
                continue
            for strip in track.strips:
                if strip.mute or not strip.action:
                    continue
                return strip.action
    return None


def _export_frame_list(context, arm_obj, action) -> List[float]:
    """Every scene frame in range + F-curve/NLA key times (what you see when scrubbing)."""
    scene = context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    if end < start:
        start, end = end, start
    frames: set = set(range(start, end + 1))
    for fr in _all_action_frames(action):
        frames.add(int(round(fr)))
    ad = arm_obj.animation_data if arm_obj else None
    if ad and ad.nla_tracks:
        for track in ad.nla_tracks:
            if track.mute:
                continue
            for strip in track.strips:
                if strip.mute or not strip.action:
                    continue
                base = float(strip.action_frame_start or 0.0)
                for fr in _all_action_frames(strip.action):
                    frames.add(int(round(strip.frame_start + fr - base)))
    return sorted(float(f) for f in frames) if frames else [1.0]


def _frames_for_props(fcs: Dict[Tuple[str, int], Any], props: Sequence[str]) -> List[float]:
    frames = set()
    for (prop, _idx), fc in fcs.items():
        if prop not in props:
            continue
        for kp in fc.keyframe_points:
            frames.add(round(float(kp.co[0]), 6))
    return sorted(frames)


def _v3_close(a: Sequence[float], b: Sequence[float], eps: float = 1e-5) -> bool:
    return all(abs(float(a[i]) - float(b[i])) <= eps for i in range(3))


def _quat_close(a: Sequence[float], b: Sequence[float], eps: float = 0.99999) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    dot = abs(
        float(a[0]) * float(b[0])
        + float(a[1]) * float(b[1])
        + float(a[2]) * float(b[2])
        + float(a[3]) * float(b[3])
    )
    return dot >= eps


def _eval_prop(
    fcs: Dict[Tuple[str, int], Any],
    prop: str,
    n: int,
    frame: float,
    defaults: Sequence[float],
) -> List[float]:
    vals = []
    for i in range(n):
        fc = fcs.get((prop, i))
        if fc is not None:
            vals.append(float(fc.evaluate(frame)))
        else:
            vals.append(float(defaults[i]))
    return vals


def _pose_local_trs(pb) -> Tuple[List[float], List[float], List[float]]:
    """Parent-relative local TRS from evaluated pose (includes constraints)."""
    if pb.parent:
        local = pb.parent.matrix.inverted() @ pb.matrix
    else:
        local = pb.matrix.copy()
    return _mat_to_trs(local)


def _constraint_affected_bones(arm_obj) -> set:
    """Bones with constraints, their targets, and IK chain members."""
    affected = set()
    for pb in arm_obj.pose.bones:
        for c in pb.constraints:
            affected.add(pb.name)
            st = getattr(c, "subtarget", None) or ""
            if st:
                affected.add(st)
            if c.type == "IK":
                chain = max(1, int(getattr(c, "chain_count", 1) or 1))
                cur = pb
                for _ in range(chain):
                    if not cur:
                        break
                    affected.add(cur.name)
                    cur = cur.parent
    return affected


def _parse_arm_doc(arm_obj) -> Optional[Dict[str, Any]]:
    """Read stamped gr2lab_json from an armature, or None if missing."""
    if not arm_obj:
        return None
    raw = arm_obj.get(PROP_DOC)
    if not raw:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError(f"{arm_obj.name}: gr2lab_json is not an object")
    return doc


def _iter_stamped_armatures(context, *, exclude=None):
    """Scene armatures that carry GR2LAB bridge data."""
    exclude_name = exclude.name if exclude else None
    for obj in context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        if exclude_name and obj.name == exclude_name:
            continue
        if obj.get(PROP_DOC):
            yield obj


def _addon_prefs():
    addon = bpy.context.preferences.addons.get(__name__)
    return addon.preferences if addon else None


def remember_template_path(path: str) -> None:
    """Persist last .gr2lab template so Export can run with one click."""
    if not path:
        return
    try:
        resolved = str(Path(path).expanduser().resolve())
    except OSError:
        resolved = str(path)
    if not Path(resolved).is_file():
        return
    prefs = _addon_prefs()
    if prefs is not None and hasattr(prefs, "last_template"):
        prefs.last_template = resolved
    try:
        bpy.context.window_manager[WM_LAST_TEMPLATE] = resolved
    except Exception:
        pass


def _is_gr2lab_path(path: Path) -> bool:
    low = path.name.lower()
    return low.endswith(".gr2lab") or low.endswith(".gr2lab.json")


def _blend_dir_gr2lab_files() -> List[Path]:
    """*.gr2lab next to the current .blend (newest first)."""
    if not bpy.data.filepath:
        return []
    parent = Path(bpy.data.filepath).parent
    if not parent.is_dir():
        return []
    found: List[Path] = []
    for p in parent.iterdir():
        if p.is_file() and _is_gr2lab_path(p):
            found.append(p)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


def auto_template_path(context, arm_obj=None) -> Optional[str]:
    """
    Find a .gr2lab without the user picking one.
    Order: armature path в†’ prefs в†’ window manager в†’ single/newest file next to .blend
    """
    candidates: List[str] = []
    if arm_obj:
        stored = arm_obj.get(PROP_TEMPLATE_PATH)
        if stored:
            candidates.append(str(stored))
    prefs = _addon_prefs()
    last_tpl = getattr(prefs, "last_template", None) if prefs else None
    if last_tpl:
        candidates.append(str(last_tpl))
    try:
        wm_path = bpy.context.window_manager.get(WM_LAST_TEMPLATE)
        if wm_path:
            candidates.append(str(wm_path))
    except Exception:
        pass
    for p in _blend_dir_gr2lab_files():
        candidates.append(str(p))

    seen = set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = Path(raw).expanduser()
        if path.is_file() and _is_gr2lab_path(path):
            return str(path.resolve())
    return None


def peek_export_template_hint(context, arm_obj=None) -> Tuple[str, str]:
    """
    UI hint: (status, detail).
    status: 'native' | 'auto' | 'synthesize'
    """
    arm_obj = arm_obj or (context.view_layer.objects.active if context else None)
    if arm_obj and arm_obj.type == "ARMATURE" and _parse_arm_doc(arm_obj) is not None:
        rid = arm_obj.get(PROP_RUN) or ""
        return "native", str(rid)
    path = auto_template_path(context, arm_obj)
    if path:
        return "auto", Path(path).name
    stamped = list(_iter_stamped_armatures(context, exclude=arm_obj)) if context else []
    if len(stamped) == 1:
        return "auto", stamped[0].name
    if arm_obj and arm_obj.type == "ARMATURE":
        n = len(arm_obj.data.bones)
        return "synthesize", f"{n} bones"
    return "synthesize", ""


def _bone_rest_local_matrix(bone) -> Matrix:
    """Parent-relative rest matrix in armature space."""
    if bone.parent:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local
    return bone.matrix_local.copy()


# Blender Z-up armature space в†’ Granny / BG3 Y-up
_RX_NEG_90 = Matrix.Rotation(math.radians(-90.0), 4, "X")


def _is_granny_armature_space(arm_obj) -> bool:
    """
    True when armature bone data is already Granny Y-up (GR2LAB Import).

    GR2LAB Import puts Granny locals into the armature and rotates the *object*
    +90В° X for Blender display. LSLib / Body_Base rigs keep Blender Z-up in the
    bones (Dummy_Root rest often ~90В° X) with no display offset вЂ” those need
    Blenderв†’Granny conversion on export.
    """
    if not arm_obj:
        return False
    if not arm_obj.get("gr2lab_y_up_display"):
        return False
    ex = abs(float(arm_obj.matrix_world.to_euler("XYZ").x))
    return abs(ex - math.radians(90.0)) < math.radians(20.0)


def _topo_bone_names(
    names: Sequence[str], parents: Dict[str, Optional[str]]
) -> List[str]:
    """Parents before children (needed for FK world build)."""
    name_set = set(names)
    remaining = set(names)
    ordered: List[str] = []
    while remaining:
        progress = False
        for name in list(remaining):
            p = parents.get(name) or None
            if not p or p not in name_set or p not in remaining:
                ordered.append(name)
                remaining.remove(name)
                progress = True
        if not progress:
            ordered.extend(sorted(remaining))
            break
    return ordered


def _blender_zup_locals_to_granny(
    parent_rel: Dict[str, Matrix],
    parents: Dict[str, Optional[str]],
) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    """
    Parent-relative Blender Z-up armature locals в†’ Granny Y-up parent-relative TRS.

    Builds Blender FK worlds, applies Rx(-90В°), re-extracts locals. Fixes Dummy_Root
    location that otherwise maps Blender horizontal motion onto Granny +Y (flies up).
    """
    order = _topo_bone_names(list(parent_rel.keys()), parents)
    worlds_bl: Dict[str, Matrix] = {}
    for name in order:
        local = parent_rel.get(name)
        if local is None:
            continue
        p = parents.get(name) or None
        worlds_bl[name] = (
            (worlds_bl[p] @ local) if p and p in worlds_bl else local.copy()
        )
    worlds_g = {n: _RX_NEG_90 @ w for n, w in worlds_bl.items()}
    out: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for name in order:
        if name not in worlds_g:
            continue
        p = parents.get(name) or None
        if p and p in worlds_g:
            local = worlds_g[p].inverted() @ worlds_g[name]
        else:
            local = worlds_g[name]
        out[name] = _mat_to_trs(local)
    return out


def synthesize_gr2lab_from_armature(arm_obj, context) -> Tuple[Dict[str, Any], float]:
    """
    Build a .gr2lab document from a plain LSLib / Body_Base-like armature.
    No site template required вЂ” site Import uses Pack-from dump + bone-name match.

    Rests are converted Blender Z-up в†’ Granny Y-up (same as export sampling).
    """
    if not arm_obj or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature to export")
    fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
    scene = context.scene
    duration = max(0.0, (float(scene.frame_end) - 1.0) / fps)

    bl_rest: Dict[str, Matrix] = {}
    parents: Dict[str, Optional[str]] = {}
    order: List[str] = []
    for bone in arm_obj.data.bones:
        order.append(bone.name)
        bl_rest[bone.name] = _bone_rest_local_matrix(bone)
        parents[bone.name] = bone.parent.name if bone.parent else None

    if _is_granny_armature_space(arm_obj):
        granny_rest = {n: _mat_to_trs(m) for n, m in bl_rest.items()}
    else:
        granny_rest = _blender_zup_locals_to_granny(bl_rest, parents)

    skeleton: List[Dict[str, Any]] = []
    tracks: List[Dict[str, Any]] = []
    root_name = ""
    for i, name in enumerate(order):
        t, r, s = granny_rest.get(name) or ([0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
        parent = parents.get(name) or ""
        if not parent and not root_name:
            root_name = name
        skeleton.append(
            {
                "name": name,
                "index": i,
                "parent_name": parent or None,
                "rest_translation": t,
                "rest_rotation": r,
                "rest_scale": s,
            }
        )
        tracks.append(
            {
                "name": name,
                "index": i,
                "position": {
                    "edit_mode": "float_controls",
                    "editable": True,
                    "times": [0.0],
                    "values": [t],
                },
                "orientation": {
                    "edit_mode": "float_controls",
                    "editable": True,
                    "times": [0.0],
                    "values": [r],
                },
                "scale": {
                    "edit_mode": "constant",
                    "editable": False,
                    "times": [0.0],
                    "values": [s],
                },
            }
        )

    if not skeleton:
        raise ValueError("Armature has no bones")

    doc: Dict[str, Any] = {
        "format": FORMAT_ID,
        "version": 1,
        "run_id": "blender",
        "source_gr2": "",
        "duration": float(duration),
        "root": root_name or skeleton[0]["name"],
        "skeleton": skeleton,
        "tracks": tracks,
    }
    return doc, fps


def resolve_export_template(
    context,
    arm_obj,
    *,
    template_armature=None,
    template_filepath: Optional[str] = None,
) -> Tuple[Dict[str, Any], float, str]:
    """
    Resolve (doc, fps, source_label) for export.
    Order: active stamp в†’ explicit/auto file в†’ scene armature в†’ synthesize from armature.
    """
    doc = _parse_arm_doc(arm_obj)
    if doc is not None:
        fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
        return doc, fps, "active armature"

    path = (template_filepath or "").strip() or (auto_template_path(context, arm_obj) or "")
    if path:
        doc = _load_doc(path)
        fps = float(arm_obj.get("gr2lab_fps") or context.scene.render.fps or FPS_DEFAULT)
        remember_template_path(path)
        if arm_obj:
            arm_obj[PROP_TEMPLATE_PATH] = str(Path(path).resolve())
        return doc, fps, Path(path).name

    if template_armature and getattr(template_armature, "type", None) == "ARMATURE":
        doc = _parse_arm_doc(template_armature)
        if doc is not None:
            fps = float(
                template_armature.get("gr2lab_fps")
                or arm_obj.get("gr2lab_fps")
                or context.scene.render.fps
                or FPS_DEFAULT
            )
            return doc, fps, template_armature.name

    stamped = list(_iter_stamped_armatures(context, exclude=arm_obj))
    if len(stamped) == 1:
        doc = _parse_arm_doc(stamped[0])
        if doc is not None:
            fps = float(
                stamped[0].get("gr2lab_fps")
                or context.scene.render.fps
                or FPS_DEFAULT
            )
            return doc, fps, stamped[0].name

    # Naked LSLib / Body_Base-like rig: build bridge from the armature itself
    doc, fps = synthesize_gr2lab_from_armature(arm_obj, context)
    return doc, fps, "synthesize"


def stamp_gr2lab_on_armature(
    arm_obj,
    doc: Dict[str, Any],
    *,
    fps: float = FPS_DEFAULT,
    y_up_display: Optional[bool] = None,
    template_path: Optional[str] = None,
) -> None:
    """Stamp bridge metadata onto an armature without rebuilding bones."""
    if not arm_obj or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature to attach GR2LAB data")
    doc_store = dict(doc)
    bind_by_name = _bind_map(doc_store)
    skeleton = _skeleton_with_bind_parents(doc_store.get("skeleton") or [], bind_by_name)
    doc_store["skeleton"] = skeleton
    doc_store["format"] = FORMAT_ID
    arm_obj[PROP_DOC] = json.dumps(doc_store)
    arm_obj[PROP_RUN] = doc_store.get("run_id") or ""
    arm_obj["gr2lab_fps"] = float(fps)
    if y_up_display is not None:
        arm_obj["gr2lab_y_up_display"] = bool(y_up_display)
    elif "gr2lab_y_up_display" not in arm_obj:
        # False for naked LSLib attach/synthesize; Import GR2LAB sets True explicitly
        arm_obj["gr2lab_y_up_display"] = False
    arm_obj["gr2lab_has_bind"] = bool(bind_by_name)
    if template_path:
        try:
            arm_obj[PROP_TEMPLATE_PATH] = str(Path(template_path).resolve())
            remember_template_path(template_path)
        except OSError:
            arm_obj[PROP_TEMPLATE_PATH] = str(template_path)


def _sample_quat_channel(ch: dict, t: float) -> Optional[List[float]]:
    times = [float(x) for x in (ch.get("times") or [])]
    vals = ch.get("values") or []
    if not vals:
        return None
    if len(times) == 1 or t <= times[0]:
        return list(vals[0])
    if t >= times[-1]:
        return list(vals[-1])
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            u = 0.0 if span <= 1e-12 else (t - times[i]) / span
            qa = _quat_gr_to_bl(vals[i])
            qb = _quat_gr_to_bl(vals[i + 1])
            qb.make_compatible(qa)
            return _quat_bl_to_gr(qa.slerp(qb, u))
    return list(vals[-1])


def _sample_vec_channel(ch: dict, t: float) -> Optional[List[float]]:
    times = [float(x) for x in (ch.get("times") or [])]
    vals = ch.get("values") or []
    if not vals:
        return None
    if len(times) == 1 or t <= times[0]:
        return list(vals[0])
    if t >= times[-1]:
        return list(vals[-1])
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            u = 0.0 if span <= 1e-12 else (t - times[i]) / span
            a, b = vals[i], vals[i + 1]
            return [float(a[j]) + (float(b[j]) - float(a[j])) * u for j in range(3)]
    return list(vals[-1])


def _count_motion_delta_bones(tmpl_doc: dict, export_doc: dict) -> int:
    """How export differs from the import snapshot (duration, keys, pose samples)."""
    t_dur = float(tmpl_doc.get("duration") or 0.0)
    e_dur = float(export_doc.get("duration") or 0.0)
    if abs(t_dur - e_dur) > 1e-4:
        return max(1, len(export_doc.get("tracks") or []))

    tm = {t["name"]: t for t in (tmpl_doc.get("tracks") or []) if t.get("name")}
    probes = [0.0]
    if e_dur > 0:
        probes.extend([e_dur * 0.25, e_dur * 0.5, e_dur * 0.75, max(0.0, e_dur - 1e-4)])
    probes = sorted(set(probes))
    changed = 0
    for tr in export_doc.get("tracks") or []:
        name = tr.get("name")
        if not name or name not in tm:
            continue
        tb = tm[name]
        bone_changed = False
        for ch_key, is_quat in (("orientation", True), ("position", False)):
            ea = tr.get(ch_key) or {}
            eb = tb.get(ch_key) or {}
            ta = [float(x) for x in (ea.get("times") or [])]
            tb_t = [float(x) for x in (eb.get("times") or [])]
            if len(ta) != len(tb_t):
                bone_changed = True
                break
            if ta and tb_t and max(abs(a - b) for a, b in zip(ta, tb_t)) > 1e-3:
                bone_changed = True
                break
            sampler = _sample_quat_channel if is_quat else _sample_vec_channel
            close = (
                (lambda a, b: _quat_close(a, b, 0.9995))
                if is_quat
                else (lambda a, b: _v3_close(a, b, 2e-4))
            )
            for t in probes:
                sa = sampler(ea, t)
                sb = sampler(eb, t)
                if sa and sb and not close(sa, sb):
                    bone_changed = True
                    break
            if bone_changed:
                break
        if bone_changed:
            changed += 1
    return changed


def _trim_channel_to_duration(ch: dict, max_t: float) -> dict:
    out = dict(ch)
    times = [float(t) for t in (ch.get("times") or [])]
    vals = ch.get("values") or []
    if not times:
        return out
    nt, nv = [], []
    for t, v in zip(times, vals):
        if float(t) <= max_t + TIME_EPS:
            nt.append(float(t))
            nv.append(v)
    if not nt:
        nt, nv = [0.0], [vals[0]]
    out["times"] = nt
    out["values"] = nv
    return out


def _trim_tracks_to_duration(tracks: List[dict], duration: float) -> None:
    for tr in tracks:
        for key in ("position", "orientation", "scale"):
            ch = tr.get(key)
            if ch:
                tr[key] = _trim_channel_to_duration(ch, duration)


def _granny_local_from_pose(
    name: str,
    fr: float,
    *,
    pb,
    fcs: Dict,
    rest_local: Dict[str, Matrix],
    identity_loc,
    identity_quat,
    identity_scl,
    use_matrix_eval: bool,
) -> Matrix:
    """Inverse of import: basis F-curves -> rest @ basis = Granny local (preferred)."""
    if use_matrix_eval and pb is not None:
        if pb.parent:
            return pb.parent.matrix.inverted() @ pb.matrix
        return pb.matrix.copy()
    loc = _eval_prop(fcs, "location", 3, fr, identity_loc)
    qwxyz = _eval_prop(fcs, "rotation_quaternion", 4, fr, identity_quat)
    scl = _eval_prop(fcs, "scale", 3, fr, identity_scl)
    w, x, y, z = qwxyz
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm > 1e-12:
        w, x, y, z = w / nrm, x / nrm, y / nrm, z / nrm
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    basis = Matrix.LocRotScale(
        Vector(loc), Quaternion((w, x, y, z)), Vector(scl)
    )
    return rest_local[name] @ basis


def _nla_strips_active(arm_obj) -> bool:
    """True when unmuted NLA strips will influence the visible pose."""
    ad = arm_obj.animation_data if arm_obj else None
    if not ad or not ad.nla_tracks:
        return False
    has_strip = False
    for track in ad.nla_tracks:
        if track.mute:
            continue
        for strip in track.strips:
            if strip.mute or not strip.action:
                continue
            has_strip = True
            break
        if has_strip:
            break
    if not has_strip:
        return False
    # use_nla blends strips; without an action the strips are the only source.
    return bool(ad.use_nla) or not ad.action


def _export_should_bake(arm_obj, mode: str = "AUTO") -> bool:
    """
    AUTO: bake only when NLA stack must be flattened (constraints use eval sampling).
    ALWAYS / NEVER: force.
    """
    m = (mode or "AUTO").upper()
    if m == "ALWAYS":
        return True
    if m == "NEVER":
        return False
    return _nla_strips_active(arm_obj)


def _bake_timeline_for_export(context, arm_obj) -> bool:
    """Bake what you see on the timeline into the armature action."""
    if not arm_obj or arm_obj.type != "ARMATURE":
        return False
    ad = arm_obj.animation_data
    if not ad:
        ad = arm_obj.animation_data_create()
    scene = context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    if end < start:
        start, end = end, start
    if end <= start:
        return False
    nla_states = []
    for track in ad.nla_tracks:
        nla_states.append((track, track.mute))
        track.mute = False
    try:
        win = context.window
        area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), None)
        override = {
            "window": win,
            "screen": win.screen,
            "area": area,
            "region": area.regions[-1] if area else None,
            "active_object": arm_obj,
            "object": arm_obj,
            "selected_objects": [arm_obj],
            "selected_editable_objects": [arm_obj],
        }
        with context.temp_override(**{k: v for k, v in override.items() if v is not None}):
            bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.nla.bake(
                frame_start=start,
                frame_end=end,
                step=1,
                only_selected=False,
                visual_keying=True,
                clear_constraints=False,
                clean_curves=False,
                bake_types={"POSE"},
            )
    except Exception as ex:
        print(f"GR2 LAB: bake skipped ({ex})")
        return False
    finally:
        for track, was in nla_states:
            track.mute = was
    return True


def export_gr2lab(
    context,
    filepath: str,
    *,
    arm_obj=None,
    template_armature=None,
    template_filepath: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
    bake_mode: str = "AUTO",
    cooperative: bool = False,
):
    """
    Export tracks from arm_obj by exact bone-name match against a GR2LAB template.
    F-curves by default; evaluated pose for constraint-affected bones.
    When cooperative=True, yields (pct, label) so a modal TIMER can refresh the UI,
    then returns stats via StopIteration.value.
    """
    arm_obj = arm_obj or context.view_layer.objects.active
    if not arm_obj or arm_obj.type != "ARMATURE":
        raise ValueError("Select an armature to export animation from")

    baked = False
    if cooperative:
        yield (2, "Preparing export…")
    if _export_should_bake(arm_obj, bake_mode):
        if progress:
            progress.step(4, "Baking timeline (NLA)…")
        if cooperative:
            yield (4, "Baking timeline (NLA)…")
        baked = _bake_timeline_for_export(context, arm_obj)
        if progress:
            progress.step(
                9,
                "Bake done" if baked else "Bake skipped — sampling F-curves…",
            )
        if cooperative:
            yield (9, "Bake done" if baked else "Bake skipped — sampling F-curves…")
    elif progress:
        progress.step(4, "Skipping bake (action F-curves)…")
    if cooperative and not _export_should_bake(arm_obj, bake_mode):
        yield (4, "Skipping bake (action F-curves)…")

    if progress:
        progress.step(10, "Loading export template…")
    had_stamp = _parse_arm_doc(arm_obj) is not None
    doc, fps, template_source = resolve_export_template(
        context,
        arm_obj,
        template_armature=template_armature,
        template_filepath=template_filepath,
    )
    snapshot_raw = arm_obj.get(PROP_IMPORT_SNAPSHOT) if arm_obj else None
    snapshot_doc = None
    if snapshot_raw:
        try:
            snapshot_doc = json.loads(snapshot_raw)
        except Exception:
            snapshot_doc = None
    delta_ref_doc = snapshot_doc if snapshot_doc else doc
    # First export from a plain rig: stamp template onto the armature so later
    # exports are native (no scene GR2LAB_* / no re-picking the file).
    stamped_native = False
    if not had_stamp:
        tpl_path = (template_filepath or "").strip() or auto_template_path(context, arm_obj)
        stamp_gr2lab_on_armature(arm_obj, doc, fps=fps, template_path=tpl_path)
        stamped_native = True
    bind_by_name = _bind_map(doc)
    skeleton = _skeleton_with_bind_parents(doc.get("skeleton") or [], bind_by_name)
    tracks = _track_map(doc)
    action = _resolve_action(arm_obj)
    if arm_obj.animation_data and action:
        arm_obj.animation_data.action = action
    fc_index = _build_fcurve_index(action)

    template_names = {b["name"] for b in skeleton if b.get("name")}
    arm_bone_names = {b.name for b in arm_obj.data.bones}
    matched_names = template_names & arm_bone_names
    extras = len(arm_bone_names - template_names)

    # F-curve quat keys may already flip hemispheres (paste / bake). Fix before sample.
    if action:
        for _bn in matched_names:
            if _bn in fc_index:
                _finalize_pose_bone_fcurves(action, _bn)

    # Two spaces:
    # - GR2LAB Import (+90В° object): armature bones are already Granny вЂ” rest @ basis
    # - LSLib / Body_Base (Z-up bones): sample Blender locals, convert Rx(-90В°) в†’ Granny
    granny_space = _is_granny_armature_space(arm_obj)

    rest_local: Dict[str, Matrix] = {}
    parents: Dict[str, Optional[str]] = {}
    for b in skeleton:
        name = b["name"]
        bone = arm_obj.data.bones.get(name)
        if granny_space:
            rest_local[name] = _bind_local_for_bone(b, bind_by_name)
            parents[name] = b.get("parent_name") or (
                bone.parent.name if bone and bone.parent else None
            )
        else:
            # Always Blender rest for Z-up в†’ Granny convert (ignore template rests)
            if bone:
                rest_local[name] = _bone_rest_local_matrix(bone)
                parents[name] = bone.parent.name if bone.parent else None
            else:
                rest_local[name] = _bind_local_for_bone(b, bind_by_name)
                parents[name] = b.get("parent_name") or None

    affected = _constraint_affected_bones(arm_obj)
    # After a successful bake, visual pose is in F-curves — skip depsgraph.
    # Otherwise only evaluate bones that constraints actually move.
    if baked:
        need_eval_names: set = set()
    else:
        need_eval_names = affected & matched_names

    identity_loc = (0.0, 0.0, 0.0)
    identity_quat = (1.0, 0.0, 0.0, 0.0)
    identity_scl = (1.0, 1.0, 1.0)

    scene = context.scene
    prev_frame = scene.frame_current
    frame_list = _export_frame_list(context, arm_obj, action)

    local_cache: Dict[float, Dict[str, Tuple[List[float], List[float], List[float]]]] = {}
    fr_total = max(1, len(frame_list))
    # Smaller batches when depsgraph eval is needed; larger for pure F-curve path.
    sample_batch = 4 if need_eval_names else 48
    try:
        fi = 0
        while fi < len(frame_list):
            batch_end = min(fi + sample_batch, len(frame_list))
            for fr in frame_list[fi:batch_end]:
                key = round(fr, 6)
                bl_bucket: Dict[str, Matrix] = {}
                arm_eval = None
                if need_eval_names:
                    scene.frame_set(int(round(fr)))
                    depsgraph = context.evaluated_depsgraph_get()
                    arm_eval = arm_obj.evaluated_get(depsgraph)
                for name in matched_names:
                    if name not in rest_local:
                        continue
                    fcs = fc_index.get(name) or {}
                    pb = arm_eval.pose.bones.get(name) if arm_eval else None
                    use_matrix = name in need_eval_names and pb is not None
                    bl_bucket[name] = _granny_local_from_pose(
                        name,
                        fr,
                        pb=pb,
                        fcs=fcs,
                        rest_local=rest_local,
                        identity_loc=identity_loc,
                        identity_quat=identity_quat,
                        identity_scl=identity_scl,
                        use_matrix_eval=use_matrix,
                    )
                for name in matched_names:
                    if name not in bl_bucket and name in rest_local:
                        bl_bucket[name] = rest_local[name].copy()

                if granny_space:
                    local_cache[key] = {
                        n: _mat_to_trs(m) for n, m in bl_bucket.items()
                    }
                else:
                    local_cache[key] = _blender_zup_locals_to_granny(bl_bucket, parents)
            fi = batch_end
            pct = 12 + int(56 * fi / max(1, fr_total))
            label = f"Sampling frames {fi}/{fr_total}…"
            if progress:
                progress.step(pct, label)
            if cooperative:
                yield (pct, label)
    finally:
        if need_eval_names:
            scene.frame_set(prev_frame)

    # Decompose + Z-up convert both pick q/-q per frame вЂ” kill export shake here.
    _stabilize_local_cache_quats(local_cache)

    def sample_at(name: str, frame: float) -> Tuple[List[float], List[float], List[float]]:
        key = round(frame, 6)
        if key not in local_cache:
            if not local_cache:
                if granny_space:
                    return _mat_to_trs(rest_local.get(name, Matrix.Identity(4)))
                rest_g = _blender_zup_locals_to_granny(rest_local, parents)
                return rest_g.get(name) or ([0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
            key = min(local_cache.keys(), key=lambda k: abs(k - frame))
        bucket = local_cache[key]
        if name in bucket:
            return bucket[name]
        if granny_space:
            return _mat_to_trs(rest_local.get(name, Matrix.Identity(4)))
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0])

    def times_collapse(samples: Dict[float, List[float]], is_quat: bool) -> List[float]:
        items = sorted(samples.items())
        if not items:
            return [0.0]
        _t0, v0 = items[0]
        close = _quat_close if is_quat else _v3_close
        if all(close(v0, v) for _, v in items):
            return [0.0]
        return [float(t) for t, _ in items]

    new_tracks = []
    global_times = [_frame_to_sec(fr, fps) for fr in frame_list]
    matched = 0
    missing = 0
    # Prefer import-time channels for protected roots (not a later densified stamp).
    snap_tracks = _track_map(snapshot_doc) if snapshot_doc else {}

    track_batch = 6
    for bi, b in enumerate(skeleton):
        name = b["name"]
        tr = tracks.get(name) or {
            "name": name,
            "position": {
                "edit_mode": "none",
                "editable": False,
                "times": [0.0],
                "values": [[0, 0, 0]],
            },
            "orientation": {
                "edit_mode": "none",
                "editable": False,
                "times": [0.0],
                "values": [[0, 0, 0, 1]],
            },
            "scale": {
                "edit_mode": "none",
                "editable": False,
                "times": [0.0],
                "values": [[1, 1, 1]],
            },
        }

        # Template bone not on this armature вЂ” keep original track values
        if name not in matched_names:
            missing += 1
            new_tracks.append(
                {
                    "name": name,
                    "index": tr.get("index", b.get("index")),
                    "position": dict(tr.get("position") or {}),
                    "orientation": dict(tr.get("orientation") or {}),
                    "scale": dict(tr.get("scale") or {}),
                }
            )
            continue

        matched += 1
        pos = dict(tr.get("position") or {})
        ori = dict(tr.get("orientation") or {})
        scl = dict(tr.get("scale") or {})
        pos_mode = pos.get("edit_mode") or "none"
        ori_mode = ori.get("edit_mode") or "none"
        scl_mode = scl.get("edit_mode") or "none"
        fcs = fc_index.get(name) or {}
        bone_affected = name in affected

        def base_times(ch, mode, props, *, upgradable: bool):
            if mode == "none" and not upgradable:
                return []
            orig = [float(t) for t in (ch.get("times") or [0.0])] or [0.0]
            fc_times = [_frame_to_sec(fr, fps) for fr in _frames_for_props(fcs, props)]
            if mode == "constant" and not upgradable:
                return [0.0]
            if mode == "k16_offsets" and not upgradable:
                return [0.0]
            # k16: import places full dequantized display keys; snap to template.
            # Collapse to bias-only later when samples still match the template.
            if mode == "k16_offsets":
                if len(fc_times) <= 1:
                    return [0.0]
                return _snap_times_to_template(fc_times, orig) or [0.0]
            # constant: F-curve keys only. Extra location keys = user wants multi-key
            # в†’ site upgrades to float.
            if mode == "constant":
                if len(fc_times) <= 1:
                    return [0.0]
                return sorted(set(fc_times)) or [0.0]
            # float / quat_k16: template knots + every exported scene frame (dense bake).
            # Collapse float-noise near-dups; prefer exact frame-grid times.
            snapped = _snap_times_to_template(fc_times, orig)
            export_t = [_frame_to_sec(fr, fps) for fr in frame_list]
            merged = _collapse_times_eps(
                list(snapped) + export_t, prefer=export_t
            )
            if bone_affected and upgradable:
                merged = _collapse_times_eps(
                    list(merged) + export_t, prefer=export_t
                ) or [0.0]
            return merged or [0.0]

        cand_pos = base_times(pos, pos_mode, ("location",), upgradable=True)
        cand_ori = base_times(ori, ori_mode, ("rotation_quaternion",), upgradable=True)
        cand_scl = base_times(scl, scl_mode, ("scale",), upgradable=False)

        all_t = _collapse_times_eps(
            list(cand_pos or []) + list(cand_ori or []) + list(cand_scl or []),
            prefer=global_times,
        )
        cache: Dict[float, Tuple[List[float], List[float], List[float]]] = {}
        for t in all_t:
            cache[t] = sample_at(name, _sec_to_frame(t, fps))

        pos_samples = {t: cache[t][0] for t in (cand_pos or [0.0]) if t in cache}
        ori_samples = {t: cache[t][1] for t in (cand_ori or [0.0]) if t in cache}

        if pos_mode == "none" and not pos.get("editable"):
            need_pos: List[float] = []
        elif pos_mode == "k16_offsets":
            # Untouched display keys в†’ bias-only; frame-bake densify / real edits в†’ multi-key
            tmpl_t = [float(t) for t in (pos.get("times") or [0.0])]
            tmpl_v = pos.get("values") or []
            sample_times = list(pos_samples.keys())
            if _is_mass_grid_densify_times(tmpl_t, sample_times):
                need_pos = times_collapse(pos_samples, False)
            elif _k16_pos_matches_template(pos_samples, tmpl_t, tmpl_v):
                need_pos = [0.0]
            else:
                need_pos = times_collapse(pos_samples, False)
        elif pos_mode in ("constant", "none"):
            # Collapse flat bias; keep multi-key only when F-curves actually vary
            need_pos = times_collapse(pos_samples, False)
        else:
            # Keep snapped F-curve times; collapse float-noise near-dups onto frame grid.
            need_pos = _collapse_times_eps(
                list(pos_samples.keys()), prefer=global_times
            ) or [0.0]

        if ori_mode == "none" and not ori.get("editable"):
            need_ori: List[float] = []
        elif ori_mode in ("constant", "none"):
            need_ori = times_collapse(ori_samples, True)
        else:
            need_ori = _collapse_times_eps(
                list(ori_samples.keys()), prefer=global_times
            ) or [0.0]

        need_scl = [0.0]

        def rewrite_channel(ch, mode, times_list, pick, *, upgradable: bool):
            out = dict(ch)
            if mode == "none" and not upgradable:
                return out
            if mode in ("constant", "k16_offsets") and not upgradable:
                t0 = 0.0
                if t0 not in cache:
                    cache[t0] = sample_at(name, _sec_to_frame(t0, fps))
                out["times"] = [0.0]
                out["values"] = [pick(cache[t0])]
                return out
            if not times_list:
                return out
            times = []
            values = []
            seen = set()
            for t in times_list:
                key = round(t, 6)
                if key in seen:
                    continue
                seen.add(key)
                if t not in cache:
                    cache[t] = sample_at(name, _sec_to_frame(t, fps))
                times.append(float(t))
                values.append(pick(cache[t]))
            out["times"] = times
            out["values"] = values
            if mode in ("constant", "k16_offsets", "none") and len(times) > 1:
                out["editable"] = True
            return out

        # Protect Dummy_Root / Root_M: finger/body polish must not rewrite whole-
        # character placement. Keep import-snapshot channels unless the user
        # sparsely edited a constraint-affected root (never full bake densify).
        protect_root = name in PROTECTED_ROOT_BONES
        root_keep = snap_tracks.get(name) or tr
        allow_root_pos = (not protect_root) or _allow_protected_root_rewrite(
            bone_affected, pos_samples, pos, is_quat=False
        )
        allow_root_ori = (not protect_root) or _allow_protected_root_rewrite(
            bone_affected, ori_samples, ori, is_quat=True
        )
        if allow_root_pos:
            pos = rewrite_channel(
                pos, pos_mode, need_pos, lambda s: s[0], upgradable=True
            )
        else:
            pos = dict(root_keep.get("position") or tr.get("position") or {})
        # Root_M ori: never bake-rewrite — keep non-densified snapshot/template.
        # Densify→float was shipping Deg1×~600 and a different animation.
        if name == "Root_M":
            keep_ori = dict(root_keep.get("orientation") or tr.get("orientation") or {})
            if _channel_is_display_densified(keep_ori):
                keep_ori = dict(tr.get("orientation") or {})
            if _channel_is_display_densified(keep_ori) or (
                (keep_ori.get("edit_mode") or "") == "float_controls"
                and len(keep_ori.get("times") or []) >= 100
            ):
                # Fall back to sparse template track if snapshot was densified.
                keep_ori = dict(tr.get("orientation") or {})
            ori = keep_ori
        elif allow_root_ori:
            ori = rewrite_channel(
                ori, ori_mode, need_ori, lambda s: s[1], upgradable=True
            )
            if ori.get("values") and len(ori["values"]) > 1:
                _make_quat_series_compatible(ori["values"])
        else:
            ori = dict(root_keep.get("orientation") or tr.get("orientation") or {})
        # Body_Base / DGB templates often bake Dummy_Root rest Rx(90°) into the
        # orientation track. Protect-keep would ship that as Granny anim → 90° tip.
        if name == "Dummy_Root":
            ori = _sanitize_dummy_root_ori(ori)
        scl = rewrite_channel(scl, scl_mode, need_scl, lambda s: s[2], upgradable=False)

        new_tracks.append(
            {
                "name": name,
                "index": tr.get("index", b.get("index")),
                "position": pos,
                "orientation": ori,
                "scale": scl,
            }
        )
        if (bi + 1) % track_batch == 0 or bi + 1 == len(skeleton):
            pct = 70 + int(18 * (bi + 1) / max(1, len(skeleton)))
            label = f"Building tracks {bi + 1}/{len(skeleton)}…"
            if progress:
                progress.step(pct, label)
            if cooperative:
                yield (pct, label)

    out_doc = dict(doc)
    out_doc["skeleton"] = skeleton
    out_doc["tracks"] = new_tracks
    export_duration = max(0.0, (float(frame_list[-1]) - 1.0) / float(fps))
    out_doc["duration"] = export_duration
    _trim_tracks_to_duration(new_tracks, export_duration)
    out_doc["blender_full_export"] = True
    out_doc["format"] = FORMAT_ID
    out_doc["version"] = int(out_doc.get("version") or 1)
    if not out_doc.get("run_id"):
        out_doc["run_id"] = "blender"
    # Forensic stamp for root-shift investigation (server writes EXPORT_META.json)
    try:
        euler = arm_obj.rotation_euler
        euler_x_deg = float(euler[0]) * 180.0 / 3.141592653589793
    except Exception:
        euler_x_deg = None
    y_up = False
    try:
        y_up = bool(arm_obj.get("gr2lab_y_up_display"))
    except Exception:
        y_up = False
    out_doc["export_meta"] = {
        "space": "granny" if granny_space else "blender_zup->granny",
        "granny_space": bool(granny_space),
        "y_up_display": y_up,
        "object_euler_x_deg": euler_x_deg,
        "frames_sampled": len(frame_list),
        "duration_sec": export_duration,
        "baked": bool(baked),
        "template_source": template_source,
        "fps": float(fps),
    }
    out_doc["export_space"] = out_doc["export_meta"]["space"]
    if progress:
        progress.step(92, "Writing GR2LAB document…")
    if cooperative:
        yield (92, "Writing GR2LAB document…")
    if filepath:
        Path(filepath).write_text(
            json.dumps(out_doc, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
    if stamped_native or template_source == "synthesize":
        stamp_gr2lab_on_armature(
            arm_obj,
            out_doc,
            fps=fps,
            y_up_display=False if template_source == "synthesize" else None,
        )
    if progress:
        progress.step(94, "GR2LAB document ready")
    stats = {
        "matched": matched,
        "missing": missing,
        "extras": extras,
        "template_source": template_source,
        "run_id": out_doc.get("run_id") or "",
        "stamped_native": stamped_native or template_source == "synthesize",
        "space": "granny" if granny_space else "blender_zup->granny",
        "frames_sampled": len(frame_list),
        "duration_sec": export_duration,
        "motion_delta_bones": _count_motion_delta_bones(delta_ref_doc, out_doc),
        "baked": baked,
        "eval_bones": len(need_eval_names),
        "doc": out_doc,
    }
    if cooperative:
        return stats
    return stats


def _consume_export_gr2lab(gen_or_stats):
    """Run cooperative export generator to completion (or pass through dict)."""
    if isinstance(gen_or_stats, dict):
        return gen_or_stats
    last = None
    try:
        while True:
            last = next(gen_or_stats)
    except StopIteration as stop:
        stats = stop.value
        if isinstance(stats, dict):
            return stats
        raise RuntimeError("Export generator finished without stats") from stop
    raise RuntimeError(f"Export generator stalled at {last!r}")


def export_gr2lab_doc(
    context,
    *,
    arm_obj=None,
    template_armature=None,
    template_filepath: Optional[str] = None,
    progress: Optional[Gr2LabProgress] = None,
    bake_mode: str = "AUTO",
    cooperative: bool = False,
):
    """Build export document in memory (for GR2 Lab server encode).

    cooperative=True → returns a generator yielding (pct, label), final stats in StopIteration.value.
    """
    if cooperative:
        return export_gr2lab(
            context,
            "",
            arm_obj=arm_obj,
            template_armature=template_armature,
            template_filepath=template_filepath,
            progress=progress,
            bake_mode=bake_mode,
            cooperative=True,
        )
    stats = _consume_export_gr2lab(
        export_gr2lab(
            context,
            "",
            arm_obj=arm_obj,
            template_armature=template_armature,
            template_filepath=template_filepath,
            progress=progress,
            bake_mode=bake_mode,
            cooperative=True,
        )
    )
    return stats


# ---------------------------------------------------------------------------
# Blender operators
# ---------------------------------------------------------------------------

PROP_PACK_RUN = "gr2lab_full_pack_run"
PROP_SOURCE_GR2 = "gr2lab_full_source_gr2"
PROP_IMPORT_FP = "gr2lab_import_fp"
PROP_IMPORT_SNAPSHOT = "gr2lab_import_snapshot"


def _active_armature(context):
    """Prefer active armature; else first selected armature; else None."""
    obj = getattr(context.view_layer.objects, "active", None)
    if obj and obj.type == "ARMATURE":
        return obj
    for obj in getattr(context, "selected_objects", None) or []:
        if obj.type == "ARMATURE":
            return obj
    return None


def _normalize_run_id(run_id: str) -> str:
    rid = (run_id or "").strip()
    if not rid or rid == "blender":
        return ""
    # Blender may rename duplicates: GR2LAB_nightfall.001
    if "." in rid and rid.rsplit(".", 1)[-1].isdigit():
        rid = rid.rsplit(".", 1)[0]
    return rid


def _pack_run_id(arm_obj) -> str:
    if not arm_obj:
        return ""
    rid = _normalize_run_id(str(arm_obj.get(PROP_PACK_RUN) or ""))
    if rid:
        return rid
    return _normalize_run_id(str(arm_obj.get(PROP_RUN) or ""))


def remember_pack_run(run_id: str, source_gr2: str = "") -> None:
    """Remember last Import GR2 dump so Export can pack even after retarget."""
    rid = _normalize_run_id(run_id)
    if not rid:
        return
    prefs = _addon_prefs()
    if prefs is not None and hasattr(prefs, "last_pack_run"):
        prefs.last_pack_run = rid
        if source_gr2 and hasattr(prefs, "last_source_gr2"):
            prefs.last_source_gr2 = str(source_gr2)
    try:
        wm = bpy.context.window_manager
        wm[WM_LAST_PACK_RUN] = rid
        if source_gr2:
            wm[WM_LAST_SOURCE_GR2] = str(source_gr2)
    except Exception:
        pass


def _remembered_pack_run() -> Tuple[str, str]:
    prefs = _addon_prefs()
    rid = ""
    src = ""
    if prefs is not None:
        rid = _normalize_run_id(str(getattr(prefs, "last_pack_run", "") or ""))
        src = str(getattr(prefs, "last_source_gr2", "") or "").strip()
    try:
        wm = bpy.context.window_manager
        rid = rid or _normalize_run_id(str(wm.get(WM_LAST_PACK_RUN) or ""))
        src = src or str(wm.get(WM_LAST_SOURCE_GR2) or "").strip()
    except Exception:
        pass
    return rid, src


def _iter_pack_armatures(context, *, exclude=None):
    """Scene armatures that carry a decode pack run id."""
    exclude_name = exclude.name if exclude else None
    for obj in context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        if exclude_name and obj.name == exclude_name:
            continue
        if _pack_run_id(obj):
            yield obj


def _iter_arm_actions(arm_obj):
    """Yield Action datablocks used by the armature (active + NLA strips)."""
    if not arm_obj or not getattr(arm_obj, "animation_data", None):
        return
    ad = arm_obj.animation_data
    if ad.action:
        yield ad.action
    for tr in ad.nla_tracks:
        for st in tr.strips:
            if getattr(st, "action", None):
                yield st.action


def _run_id_from_action_name(action) -> str:
    if not action:
        return ""
    name = str(getattr(action, "name", "") or "")
    if not name.startswith("GR2LAB_"):
        return ""
    return _normalize_run_id(name[len("GR2LAB_") :])


def _find_pack_donor(context, arm_obj) -> Tuple[Any, str, str]:
    """Find imported GR2LAB armature / run matching this arm's action or NLA.

    Returns (donor_arm_or_None, run_id, source_gr2).
    """
    if not context or not arm_obj:
        return None, "", ""
    actions = list(_iter_arm_actions(arm_obj))
    if not actions:
        return None, "", ""

    # Same Action datablock as an imported GR2LAB_* armature (usual paste workflow).
    for act in actions:
        for obj in context.scene.objects:
            if obj.type != "ARMATURE" or obj == arm_obj:
                continue
            oad = obj.animation_data
            if not oad or oad.action != act:
                continue
            rid = _pack_run_id(obj)
            if rid:
                return obj, rid, str(obj.get(PROP_SOURCE_GR2) or "").strip()

    # Action / object named GR2LAB_<run>
    for act in actions:
        rid = _run_id_from_action_name(act)
        if not rid:
            continue
        for obj in context.scene.objects:
            if obj.type != "ARMATURE" or obj == arm_obj:
                continue
            if _pack_run_id(obj) == rid:
                return obj, rid, str(obj.get(PROP_SOURCE_GR2) or "").strip()
            oname = obj.name or ""
            if oname == f"GR2LAB_{rid}" or oname.startswith(f"GR2LAB_{rid}."):
                orun = _pack_run_id(obj) or rid
                return obj, orun, str(obj.get(PROP_SOURCE_GR2) or "").strip()
        return None, rid, ""

    return None, "", ""


def _stamp_pack_from_donor(arm_obj, donor) -> None:
    """Copy dump/pack stamps from imported GR2LAB arm onto a working retarget arm."""
    if not arm_obj or not donor:
        return
    rid = _pack_run_id(donor)
    src = str(donor.get(PROP_SOURCE_GR2) or "").strip()
    stamp_pack_run(arm_obj, rid, src)
    for key in (
        PROP_IMPORT_SNAPSHOT,
        PROP_IMPORT_FP,
        "gr2lab_y_up_display",
        "gr2lab_source_interp",
        "gr2lab_densify_fps",
        "gr2lab_import_densified",
    ):
        if key in donor:
            try:
                arm_obj[key] = donor[key]
            except Exception:
                pass


def link_pack_from_action(context, arm_obj, *, force: bool = False) -> Tuple[str, str, str]:
    """Stamp pack dump onto arm from its GR2LAB_* action / donor. Returns (rid, src, note)."""
    if not arm_obj:
        return "", "", ""
    donor, rid, src = _find_pack_donor(context, arm_obj)
    current = _pack_run_id(arm_obj)
    if not rid:
        return current, str(arm_obj.get(PROP_SOURCE_GR2) or "").strip(), ""
    if not force and current == rid:
        return current, str(arm_obj.get(PROP_SOURCE_GR2) or src).strip(), "already linked"
    if donor:
        _stamp_pack_from_donor(arm_obj, donor)
        remember_pack_run(rid, str(arm_obj.get(PROP_SOURCE_GR2) or src))
        return rid, str(arm_obj.get(PROP_SOURCE_GR2) or src).strip(), f"linked from {donor.name}"
    stamp_pack_run(arm_obj, rid, src)
    remember_pack_run(rid, src)
    return rid, src, f"from action GR2LAB_{rid}"


def _decode_run_ids(base_url: str) -> set:
    try:
        status = ping(base_url)
        return {str(r.get("id") or "") for r in (status.get("decode_runs") or []) if r.get("id")}
    except Exception:
        return set()


def resolve_pack_run(context, arm_obj=None) -> Tuple[str, str]:
    """
    Resolve (run_id, source_gr2) for Export GR2.

    Order:
      1) GR2LAB_* action / donor armature (NLA paste workflow) — overrides stale props
      2) stamps on the active arm
      3) single other stamped scene arm
      4) last Import GR2 remember
    """
    arm_obj = arm_obj or _active_armature(context)
    if context and arm_obj:
        rid, src, _note = link_pack_from_action(context, arm_obj, force=False)
        if rid:
            return rid, src

    rid = _pack_run_id(arm_obj)
    src = str(arm_obj.get(PROP_SOURCE_GR2) or "").strip() if arm_obj else ""
    if rid:
        return rid, src

    stamped = list(_iter_pack_armatures(context, exclude=arm_obj)) if context else []
    if len(stamped) == 1:
        other = stamped[0]
        return _pack_run_id(other), str(other.get(PROP_SOURCE_GR2) or "").strip()

    return _remembered_pack_run()


def stamp_pack_run(arm_obj, run_id: str, source_gr2: str = "") -> None:
    if not arm_obj:
        return
    rid = _normalize_run_id(run_id)
    if not rid:
        return
    arm_obj[PROP_PACK_RUN] = rid
    arm_obj[PROP_RUN] = rid
    if source_gr2:
        arm_obj[PROP_SOURCE_GR2] = source_gr2


def _server_url(context) -> str:
    prefs = context.preferences.addons[__name__].preferences
    return (prefs.server_url or DEFAULT_BASE_URL).strip()


class GR2LabFullPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    server_url: StringProperty(
        name="GR2 Lab server",
        description="GR2 Lab URL — use Find on network if the server runs on another PC",
        default=DEFAULT_BASE_URL,
    )
    body_base: StringProperty(
        name="Body base (optional)",
        description="Body_Base name for bind on decode вЂ” leave empty for auto-match",
        default="",
    )
    y_up_display: BoolProperty(
        name="Y-up display on GR2 import",
        description="Rotate imported armature +90В° X for Blender Z-up view",
        default=True,
    )
    last_template: StringProperty(
        name="Last GR2LAB template",
        description="Last .gr2lab used as export template (auto-filled)",
        default="",
        subtype="FILE_PATH",
    )
    last_pack_run: StringProperty(
        name="Last pack dump",
        description="Decode run_id from last Import GR2 — used when Export armature has no stamp",
        default="",
    )
    last_source_gr2: StringProperty(
        name="Last source GR2",
        description="Source .GR2 name from last Import GR2",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "server_url")
        row = layout.row(align=True)
        row.operator("gr2lab_full.find_server", icon="VIEWZOOM", text="Find on network")
        row.operator("gr2lab_full.test_server", icon="URL", text="Test")
        layout.prop(self, "body_base")
        layout.prop(self, "y_up_display")
        layout.prop(self, "last_template")
        layout.prop(self, "last_pack_run")
        layout.prop(self, "last_source_gr2")


class GR2LABFULL_OT_find_server(bpy.types.Operator):
    bl_idname = "gr2lab_full.find_server"
    bl_label = "Find GR2 Lab on network"
    bl_description = (
        "Try this PC first (127.0.0.1), then scan the local Wi-Fi/LAN for GR2 Lab on port 8765"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        try:
            hits = discover_gr2lab_servers()
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if not hits:
            self.report(
                {"ERROR"},
                "No GR2 Lab found. On the main PC run Start_GR2_Lab.bat and allow port 8765 "
                "through Windows Firewall (private network).",
            )
            return {"CANCELLED"}
        base, status = hits[0]
        prefs.server_url = base
        n = int((status.get("in_gr2_stats") or {}).get("file_count") or len(status.get("files") or []))
        if len(hits) == 1:
            self.report({"INFO"}, f"GR2 Lab at {base} · {n} GR2 in IN_GR2")
        else:
            also = ", ".join(u for u, _ in hits[1:4])
            self.report(
                {"INFO"},
                f"Using {base} · {n} GR2 (also on LAN: {also})",
            )
        return {"FINISHED"}


class GR2LABFULL_OT_test_server(bpy.types.Operator):
    bl_idname = "gr2lab_full.test_server"
    bl_label = "Test GR2 Lab connection"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            data = ping(_server_url(context))
            n = int((data.get("in_gr2_stats") or {}).get("file_count") or len(data.get("files") or []))
            self.report({"INFO"}, f"GR2 Lab OK В· {n} GR2 in IN_GR2 В· root={data.get('root', '?')}")
        except Gr2LabApiError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class GR2LABFULL_FH_gr2(bpy.types.FileHandler):
    """Drag-and-drop .GR2 into the 3D View → import via GR2 Lab."""

    bl_idname = "GR2LABFULL_FH_gr2"
    bl_label = "GR2 via GR2 Lab"
    bl_import_operator = "import_scene.gr2_full"
    bl_file_extensions = ".gr2;.GR2"

    @classmethod
    def poll_drop(cls, context):
        area = getattr(context, "area", None)
        return area is not None and area.type in {"VIEW_3D", "OUTLINER"}


class IMPORT_OT_gr2_full(bpy.types.Operator, ImportHelper):
    """Upload GR2 to GR2 Lab, decode, densify B-splines, import rig + animation"""

    bl_idname = "import_scene.gr2_full"
    bl_label = "GR2 (via GR2 Lab)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".GR2"
    filter_glob: StringProperty(default="*.GR2;*.gr2", options={"HIDDEN"})
    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    fps: FloatProperty(name="FPS", default=FPS_DEFAULT, min=1.0, max=120.0)
    body_base: EnumProperty(
        name="Body Base",
        description="Body_Base GR2 for bind pose and IK hierarchy (same as site Decode)",
        items=_import_body_base_enum_items,
        default=0,
    )
    _bb_hint: StringProperty(default="", options={"HIDDEN"})
    _bb_synced_for: StringProperty(default="", options={"HIDDEN"})

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "fps")
        try:
            _sync_import_body_base(self, context)
        except Exception:
            pass
        layout.prop(self, "body_base", text="Body Base")
        if self._bb_hint:
            layout.label(text=self._bb_hint, icon="INFO")

    def execute(self, context):
        path = self.filepath
        if not path or not Path(path).is_file():
            self.report({"ERROR"}, "Select a .GR2 file")
            return {"CANCELLED"}
        prefs = context.preferences.addons[__name__].preferences
        base = _resolve_body_base_choice(self.body_base, prefs.body_base)
        url = _server_url(context)
        progress = Gr2LabProgress(context)
        progress.begin(100)
        dec = doc = arm = None
        try:
            progress.step(1, "Starting import…")
            dec, doc = import_gr2_pipeline(
                url, path, body_base=base, progress=progress
            )
            arm = import_gr2lab_from_doc(
                context,
                doc,
                fps=self.fps,
                y_up_display=bool(prefs.y_up_display),
                template_path=None,
                progress=progress,
            )
        except Gr2LabApiError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except Exception as e:
            traceback.print_exc()
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        finally:
            progress.end()

        run_id = dec.get("run_id") or doc.get("run_id") or ""

        arm[PROP_PACK_RUN] = run_id
        arm[PROP_SOURCE_GR2] = Path(path).name
        remember_pack_run(run_id, Path(path).name)
        try:
            arm[PROP_IMPORT_FP] = json.dumps(
                {"bones": len(doc.get("skeleton") or []), "duration": doc.get("duration")}
            )
        except Exception:
            pass
        dens = doc.get("densified_channels")
        interp = doc.get("source_interp") or ""
        imp_d = arm.get("gr2lab_import_densified")
        msg = (
            f"Imported {Path(path).name} · run={run_id} · "
            f"{len(doc.get('skeleton') or [])} bones"
        )
        if interp:
            msg += f" · {interp}"
        if dens:
            msg += f" · densified={dens}"
        if imp_d:
            msg += f" · import-bake={imp_d}"
        self.report({"INFO"}, msg)
        if dec.get("bind_warning"):
            self.report({"WARNING"}, str(dec["bind_warning"]))
        elif base:
            self.report({"INFO"}, f"Body base: {Path(base).name}")
        return {"FINISHED"}


class EXPORT_OT_gr2_full(bpy.types.Operator, ExportHelper):
    """Export animation via GR2 Lab — modal TIMER so Blender stays responsive."""

    bl_idname = "export_scene.gr2_full"
    bl_label = "GR2 (via GR2 Lab)"
    bl_options = {"REGISTER"}

    filename_ext = ".GR2"
    filter_glob: StringProperty(default="*.GR2;*.gr2", options={"HIDDEN"})
    arm_name: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    pack_run: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    source_gr2: StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    bake_mode: EnumProperty(
        name="Bake before export",
        description="NLA bake is slow. AUTO skips it for a normal action",
        items=(
            ("AUTO", "Auto", "Bake only when unmuted NLA strips affect the pose"),
            ("ALWAYS", "Always", "Always run NLA bake (slow)"),
            ("NEVER", "Never", "Never bake — F-curves + constraint eval only"),
        ),
        default="AUTO",
    )

    def _resolve_live_pack(self, context, arm) -> Tuple[str, str]:
        """Resolve pack dump; if stale/missing on server, relink from GR2LAB_* action."""
        rid, src = resolve_pack_run(context, arm)
        ids = _decode_run_ids(_server_url(context))
        if rid and ids and rid not in ids:
            # Stale custom prop (e.g. dead-letter after re-import as nightfall).
            if arm and arm.get(PROP_PACK_RUN) == rid:
                try:
                    del arm[PROP_PACK_RUN]
                except Exception:
                    arm[PROP_PACK_RUN] = ""
            link_pack_from_action(context, arm, force=True)
            rid, src = resolve_pack_run(context, arm)
            if rid and rid not in ids:
                rem_rid, rem_src = _remembered_pack_run()
                if rem_rid and rem_rid in ids:
                    stamp_pack_run(arm, rem_rid, rem_src)
                    rid, src = rem_rid, rem_src
        if arm and rid:
            stamp_pack_run(arm, rid, src)
            remember_pack_run(rid, src)
        return rid, src

    def invoke(self, context, event):
        arm = _active_armature(context)
        rid, src = self._resolve_live_pack(context, arm) if arm else ("", "")
        self.arm_name = arm.name if arm else ""
        self.pack_run = rid
        self.source_gr2 = src
        if arm and arm.get(PROP_SOURCE_GR2):
            self.filepath = str(arm.get(PROP_SOURCE_GR2))
        elif src:
            self.filepath = src
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        layout = self.layout
        arm = None
        if self.arm_name and self.arm_name in bpy.data.objects:
            obj = bpy.data.objects[self.arm_name]
            if obj.type == "ARMATURE":
                arm = obj
        arm = arm or _active_armature(context)
        rid = (self.pack_run or "").strip() or resolve_pack_run(context, arm)[0]
        if rid:
            layout.label(text=f"Pack from dump: {rid}", icon="CHECKMARK")
            if arm:
                act = arm.animation_data.action if arm.animation_data else None
                if act and str(act.name).startswith("GR2LAB_"):
                    layout.label(text=f"From action: {act.name}", icon="ACTION")
        else:
            layout.label(text="No pack run — Import GR2 (via GR2 Lab) first", icon="ERROR")
        if arm:
            status, detail = peek_export_template_hint(context, arm)
            layout.label(text=f"Export source: {status}" + (f" ({detail})" if detail else ""))
            if _export_should_bake(arm, self.bake_mode):
                layout.label(text="Will bake NLA (slow)", icon="TIME")
            else:
                layout.label(text="Fast path: F-curve sample", icon="CHECKMARK")
        layout.prop(self, "bake_mode")
        layout.label(text="Export runs in background slices (Esc to cancel)")

    def execute(self, context):
        arm = None
        if self.arm_name and self.arm_name in bpy.data.objects:
            obj = bpy.data.objects[self.arm_name]
            if obj.type == "ARMATURE":
                arm = obj
        arm = arm or _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select the animated armature")
            return {"CANCELLED"}
        run_id, source_gr2 = self._resolve_live_pack(context, arm)
        source_gr2 = source_gr2 or str(arm.get(PROP_SOURCE_GR2) or "").strip() or None
        if not run_id:
            self.report(
                {"ERROR"},
                "No decode run_id — Import GR2 (via GR2 Lab), keep that action "
                "(GR2LAB_<dump>) on this armature / NLA, then Export again",
            )
            return {"CANCELLED"}
        stamp_pack_run(arm, run_id, source_gr2 or "")
        remember_pack_run(run_id, source_gr2 or "")
        self.pack_run = run_id
        self.source_gr2 = source_gr2 or ""

        self._arm_name = arm.name
        self._run_id = run_id
        self._source_gr2 = source_gr2
        self._server_url = _server_url(context)
        self._phase = "sample"
        self._net_thread = None
        self._net_result = None
        self._net_error = None
        self._stats = None
        self._progress = Gr2LabProgress(context)
        self._progress.begin(100)
        self._progress.step(1, "Starting export…")
        self._gen = export_gr2lab_doc(
            context,
            arm_obj=arm,
            progress=self._progress,
            bake_mode=self.bake_mode,
            cooperative=True,
        )
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _cleanup_timer(self, context):
        wm = context.window_manager
        if getattr(self, "_timer", None) is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        if getattr(self, "_progress", None) is not None:
            self._progress.end()
            self._progress = None

    def _start_network(self):
        stats = self._stats or {}
        doc = dict(stats.get("doc") or {})
        doc["run_id"] = self._run_id
        url = self._server_url
        run_id = self._run_id
        dest = self.filepath
        source_gr2 = self._source_gr2

        def worker():
            try:
                self._net_result = pack_encode_gr2(
                    url,
                    run_id,
                    doc,
                    dest,
                    source_gr2=source_gr2,
                    progress=None,
                )
            except Exception as ex:
                self._net_error = ex

        self._phase = "network"
        self._progress.step(86, "Pack-encode on GR2 Lab…")
        self._net_thread = __import__("threading").Thread(target=worker, daemon=True)
        self._net_thread.start()

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup_timer(context)
            self.report({"WARNING"}, "Export cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._phase == "sample":
            try:
                pct, label = next(self._gen)
                if self._progress:
                    self._progress.step(int(pct), label)
            except StopIteration as stop:
                self._stats = stop.value if isinstance(stop.value, dict) else None
                if not self._stats or "doc" not in self._stats:
                    self._cleanup_timer(context)
                    self.report({"ERROR"}, "Export produced no document")
                    return {"CANCELLED"}
                self._start_network()
            except Exception as e:
                traceback.print_exc()
                self._cleanup_timer(context)
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if self._phase == "network":
            if self._net_thread and self._net_thread.is_alive():
                if self._progress:
                    self._progress.step(90, "Pack-encode on GR2 Lab…")
                return {"PASS_THROUGH"}
            self._cleanup_timer(context)
            if self._net_error is not None:
                err = self._net_error
                if isinstance(err, Gr2LabApiError):
                    self.report({"ERROR"}, str(err))
                else:
                    traceback.print_exc()
                    self.report({"ERROR"}, str(err))
                return {"CANCELLED"}
            enc = self._net_result or {}
            imp = enc.get("import_result") or {}
            stats = self._stats or {}
            mode = enc.get("mode") or enc.get("badge") or "encoded"
            bones_applied = int(imp.get("bones_edited") or 0)
            ops_applied = int(imp.get("ops") or 0)
            apply_noop = bool(enc.get("apply_noop")) or not bool(imp.get("changed"))
            bake_note = (
                "baked" if stats.get("baked") else f"fcurve+eval{stats.get('eval_bones', 0)}"
            )
            self.report(
                {"INFO"},
                f"Wrote {Path(self.filepath).name} · dump={self._run_id} · {mode} · "
                f"{bake_note} · applied {bones_applied} bones / {ops_applied} ops · "
                f"{stats.get('matched', 0)} matched · {stats.get('frames_sampled', 0)} frames",
            )
            if apply_noop:
                self.report({"INFO"}, "Motion matched dump (noop apply) — GR2 still written")
            elif enc.get("warning"):
                self.report({"WARNING"}, str(enc["warning"]))
            if stats.get("missing"):
                self.report(
                    {"WARNING"},
                    f"{stats['missing']} template bones missing — original tracks kept",
                )
            return {"FINISHED"}

        return {"PASS_THROUGH"}

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None



class IMPORT_OT_gr2lab_full(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.gr2lab_full"
    bl_label = "GR2LAB (.gr2lab)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".gr2lab"
    filter_glob: StringProperty(default="*.gr2lab;*.gr2lab.json", options={"HIDDEN"})
    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    fps: FloatProperty(name="FPS", default=FPS_DEFAULT, min=1.0, max=120.0)

    def execute(self, context):
        progress = Gr2LabProgress(context)
        progress.begin(100)
        try:
            progress.step(5, "Reading GR2LAB…")
            import_gr2lab(
                context,
                self.filepath,
                fps=self.fps,
                progress=progress,
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        finally:
            progress.end()
        self.report({"INFO"}, f"Imported {Path(self.filepath).name}")
        return {"FINISHED"}


class EXPORT_OT_gr2lab_full(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.gr2lab_full"
    bl_label = "GR2LAB (.gr2lab)"
    bl_options = {"REGISTER"}

    filename_ext = ".gr2lab"
    filter_glob: StringProperty(default="*.gr2lab;*.gr2lab.json", options={"HIDDEN"})
    bake_mode: EnumProperty(
        name="Bake before export",
        description="NLA bake is slow. AUTO skips it for a normal action",
        items=(
            ("AUTO", "Auto", "Bake only when unmuted NLA strips affect the pose"),
            ("ALWAYS", "Always", "Always run NLA bake (slow)"),
            ("NEVER", "Never", "Never bake — F-curves + constraint eval only"),
        ),
        default="AUTO",
    )

    def draw(self, context):
        self.layout.prop(self, "bake_mode")

    def execute(self, context):
        arm = _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        progress = Gr2LabProgress(context)
        progress.begin(100)
        try:
            progress.step(2, "Exporting GR2LAB…", force_ui=True)
            stats = _consume_export_gr2lab(
                export_gr2lab(
                    context,
                    self.filepath,
                    arm_obj=arm,
                    progress=progress,
                    bake_mode=self.bake_mode,
                    cooperative=True,
                )
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        finally:
            progress.end()
        bake_note = "baked" if stats.get("baked") else f"fcurve+eval{stats.get('eval_bones', 0)}"
        self.report(
            {"INFO"},
            f"Exported · {stats.get('matched', 0)} matched · {bake_note} · "
            f"source={stats.get('template_source')}",
        )
        return {"FINISHED"}


class OBJECT_OT_gr2lab_link_dump(bpy.types.Operator):
    """Stamp GR2 Lab dump id onto the active armature from its GR2LAB_* action / NLA."""

    bl_idname = "object.gr2lab_link_dump"
    bl_label = "GR2 Lab: Link dump from action"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _active_armature(context)
        if not arm:
            self.report({"ERROR"}, "Select an armature")
            return {"CANCELLED"}
        rid, src, note = link_pack_from_action(context, arm, force=True)
        if not rid:
            self.report(
                {"ERROR"},
                "No GR2LAB_* action/NLA found — Import GR2 first, then paste that action here",
            )
            return {"CANCELLED"}
        ids = _decode_run_ids(_server_url(context))
        if ids and rid not in ids:
            self.report(
                {"WARNING"},
                f"Linked dump={rid} but it is not on GR2 Lab right now — re-import that GR2",
            )
        else:
            self.report({"INFO"}, f"Dump={rid}" + (f" · {note}" if note else ""))
        if src:
            self.report({"INFO"}, f"Source GR2: {src}")
        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        return _active_armature(context) is not None


def menu_import(self, context):
    self.layout.operator(IMPORT_OT_gr2_full.bl_idname, text="GR2 (via GR2 Lab)")
    self.layout.operator(IMPORT_OT_gr2lab_full.bl_idname, text="GR2LAB (.gr2lab)")


def menu_export(self, context):
    self.layout.operator(EXPORT_OT_gr2_full.bl_idname, text="GR2 (via GR2 Lab)")
    self.layout.operator(EXPORT_OT_gr2lab_full.bl_idname, text="GR2LAB (.gr2lab)")


def menu_object(self, context):
    self.layout.separator()
    self.layout.operator(OBJECT_OT_gr2lab_link_dump.bl_idname)


classes = (
    GR2LabFullPreferences,
    GR2LABFULL_OT_find_server,
    GR2LABFULL_OT_test_server,
    GR2LABFULL_FH_gr2,
    IMPORT_OT_gr2_full,
    EXPORT_OT_gr2_full,
    IMPORT_OT_gr2lab_full,
    EXPORT_OT_gr2lab_full,
    OBJECT_OT_gr2lab_link_dump,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_export)
    bpy.types.VIEW3D_MT_object.append(menu_object)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_export)
    bpy.types.VIEW3D_MT_object.remove(menu_object)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
