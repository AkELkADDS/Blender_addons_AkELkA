"""
Local web UI for BG3 GR2 animation compare.
Open http://127.0.0.1:8765 — pick 2 files, Run, watch progress.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
HOST = "127.0.0.1"
PORT = 8765

# Shared job state
_job_lock = threading.Lock()
_job = {
    "running": False,
    "pct": 0,
    "msg": "Idle",
    "started": 0.0,
    "elapsed": 0.0,
    "error": None,
    "result": None,
}


def _set_progress(msg, pct=None):
    with _job_lock:
        _job["msg"] = msg
        if pct is not None:
            _job["pct"] = max(0, min(100, float(pct)))
        if _job["started"]:
            _job["elapsed"] = time.time() - _job["started"]


def _run_job(ref_path: str, test_path: str, camera_cut: float):
    import gr2_anim_compare as cmp

    with _job_lock:
        _job["running"] = True
        _job["pct"] = 0
        _job["msg"] = "Starting..."
        _job["started"] = time.time()
        _job["elapsed"] = 0.0
        _job["error"] = None
        _job["result"] = None

    try:
        result = cmp.run_compare(
            Path(ref_path),
            Path(test_path),
            progress=_set_progress,
            camera_cut=camera_cut,
        )
        with _job_lock:
            _job["result"] = result
            _job["pct"] = 100
            _job["msg"] = "Done"
            _job["elapsed"] = time.time() - _job["started"]
    except Exception as e:
        with _job_lock:
            _job["error"] = f"{e}\n{traceback.format_exc()}"
            _job["msg"] = f"Failed: {e}"
            _job["elapsed"] = time.time() - _job["started"]
    finally:
        with _job_lock:
            _job["running"] = False


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>BG3 Anim Compare</title>
<style>
:root {
  --bg:#0f1117; --panel:#1a1d27; --text:#e5e7eb; --muted:#9ca3af;
  --accent:#60a5fa; --ok:#22c55e; --bad:#ef4444; --line:#2a2f3d;
}
* { box-sizing:border-box; }
body {
  margin:0; font-family:Segoe UI,system-ui,sans-serif;
  background:var(--bg); color:var(--text); line-height:1.4;
}
.wrap { max-width:720px; margin:0 auto; padding:16px; }
h1 { margin:0 0 4px; font-size:1.2rem; }
.sub { color:var(--muted); font-size:.82rem; margin-bottom:14px; }
.panel {
  background:var(--panel); border:1px solid var(--line);
  border-radius:8px; padding:12px; margin-bottom:12px;
}
label { display:block; font-size:.75rem; color:var(--accent);
  text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
.row { margin-bottom:10px; }
.file-row { display:flex; gap:6px; align-items:center; }
input[type=text], input[type=number] {
  flex:1; background:#0a0c10; border:1px solid var(--line);
  color:var(--text); border-radius:5px; padding:7px 8px; font-size:.85rem;
  font-family:Consolas,monospace;
}
button {
  background:#2563eb; color:#fff; border:none; border-radius:5px;
  padding:7px 12px; cursor:pointer; font-size:.85rem;
}
button.sec { background:#374151; }
button:disabled { opacity:.5; cursor:not-allowed; }
.hint { font-size:.75rem; color:var(--muted); margin-top:4px; }
.progress-wrap {
  background:#0a0c10; border:1px solid var(--line);
  border-radius:6px; height:10px; overflow:hidden; margin:8px 0;
}
.progress-bar {
  height:100%; width:0%; background:linear-gradient(90deg,#2563eb,#60a5fa);
  transition:width .2s ease;
}
.meta { display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); }
.msg { font-size:.85rem; margin-top:6px; min-height:1.2em; }
.msg.err { color:var(--bad); white-space:pre-wrap; font-size:.75rem; }
.msg.ok { color:var(--ok); }
.actions { display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }
.quick { font-size:.78rem; color:var(--muted); }
.quick a { color:var(--accent); cursor:pointer; text-decoration:underline; }
</style>
</head>
<body>
<div class="wrap">
  <h1>BG3 Animation Compare</h1>
  <p class="sub">Pick reference (game/LSLib) + test (Blender export). Runs locally — files stay on your PC.</p>

  <div class="panel">
    <div class="row">
      <label>1 · Reference file (GR2 or GLB)</label>
      <div class="file-row">
        <input type="text" id="refPath" placeholder="C:\...\something_TIF_SK.GR2" spellcheck="false"/>
        <button class="sec" type="button" id="btnBrowseRef">Browse</button>
      </div>
      <input type="file" id="refFile" accept=".gr2,.glb,.GR2,.GLB" hidden/>
    </div>
    <div class="row">
      <label>2 · Test file (GR2 or GLB)</label>
      <div class="file-row">
        <input type="text" id="testPath" placeholder="C:\...\blender_export.GR2" spellcheck="false"/>
        <button class="sec" type="button" id="btnBrowseTest">Browse</button>
      </div>
      <input type="file" id="testFile" accept=".gr2,.glb,.GR2,.GLB" hidden/>
      <p class="hint">Browse uploads a copy into the tool folder. Or paste a full path if the file is already on disk.</p>
    </div>
    <div class="row">
      <label>Camera-cut time (seconds)</label>
      <input type="number" id="cutT" value="12.3" step="0.1" min="0" style="max-width:120px"/>
    </div>
    <p class="quick">Quick fill from Source_to_work_with:
      <a id="fillDefault">Karlach romance pair</a>
    </p>
    <div class="actions">
      <button type="button" id="btnRun">Run compare</button>
      <button class="sec" type="button" id="btnOpen" disabled>Open report</button>
    </div>
  </div>

  <div class="panel">
    <label>Progress</label>
    <div class="progress-wrap"><div class="progress-bar" id="bar"></div></div>
    <div class="meta">
      <span id="pct">0%</span>
      <span id="elapsed">0.0s</span>
    </div>
    <div class="msg" id="msg">Idle — choose two files and press Run.</div>
  </div>
</div>
<script>
const refPath = document.getElementById('refPath');
const testPath = document.getElementById('testPath');
const cutT = document.getElementById('cutT');
const bar = document.getElementById('bar');
const pctEl = document.getElementById('pct');
const elapsedEl = document.getElementById('elapsed');
const msgEl = document.getElementById('msg');
const btnRun = document.getElementById('btnRun');
const btnOpen = document.getElementById('btnOpen');
let pollTimer = null;
let lastReport = null;

document.getElementById('btnBrowseRef').onclick = () => document.getElementById('refFile').click();
document.getElementById('btnBrowseTest').onclick = () => document.getElementById('testFile').click();

async function upload(input, target) {
  const f = input.files && input.files[0];
  if (!f) return;
  msgEl.textContent = 'Uploading ' + f.name + '...';
  msgEl.className = 'msg';
  const fd = new FormData();
  fd.append('file', f);
  const res = await fetch('/api/upload', { method:'POST', body: fd });
  const data = await res.json();
  if (!res.ok) {
    msgEl.textContent = data.error || 'Upload failed';
    msgEl.className = 'msg err';
    return;
  }
  target.value = data.path;
  msgEl.textContent = 'Ready: ' + f.name;
}

document.getElementById('refFile').onchange = (e) => upload(e.target, refPath);
document.getElementById('testFile').onchange = (e) => upload(e.target, testPath);

document.getElementById('fillDefault').onclick = async () => {
  const res = await fetch('/api/defaults');
  const d = await res.json();
  if (d.ref) refPath.value = d.ref;
  if (d.test) testPath.value = d.test;
  if (d.cut != null) cutT.value = d.cut;
};

function setProgress(p) {
  bar.style.width = (p.pct || 0) + '%';
  pctEl.textContent = Math.round(p.pct || 0) + '%';
  elapsedEl.textContent = (p.elapsed || 0).toFixed(1) + 's';
  msgEl.textContent = p.msg || '';
  msgEl.className = p.error ? 'msg err' : (p.pct >= 100 && p.result ? 'msg ok' : 'msg');
  if (p.result && p.result.html) {
    lastReport = p.result.html;
    btnOpen.disabled = false;
  }
  btnRun.disabled = !!p.running;
}

async function poll() {
  try {
    const res = await fetch('/api/status');
    const p = await res.json();
    setProgress(p);
    if (!p.running && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (e) {
    msgEl.textContent = 'Lost connection to local server.';
    msgEl.className = 'msg err';
  }
}

btnRun.onclick = async () => {
  if (!refPath.value || !testPath.value) {
    msgEl.textContent = 'Choose both reference and test files.';
    msgEl.className = 'msg err';
    return;
  }
  btnOpen.disabled = true;
  lastReport = null;
  btnRun.disabled = true;
  msgEl.className = 'msg';
  msgEl.textContent = 'Starting...';
  const res = await fetch('/api/run', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      ref: refPath.value,
      test: testPath.value,
      cut: parseFloat(cutT.value) || 12.3,
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    msgEl.textContent = data.error || 'Could not start';
    msgEl.className = 'msg err';
    btnRun.disabled = false;
    return;
  }
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(poll, 400);
  poll();
};

btnOpen.onclick = () => {
  if (lastReport) window.open('/report', '_blank');
};
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[ui] {args[0]}" if args else fmt)

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code, data: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._bytes(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            with _job_lock:
                snap = dict(_job)
            self._json(200, snap)
            return
        if path == "/api/defaults":
            ref = SOURCE_DIR / "TIF_FS_Rig_SCENE_CAMP_Karlach_SD_ROM_ForgingOfTheHeart_Romance_Karlach_000_TIF_SK.GR2"
            test = SOURCE_DIR / "TIF_FS_Rig_SCENE_CAMP_Karlach_SD_ROM_ForgingOfTheHeart_Romance_Karlach_DGB_EL_1_sk_tif.GR2"
            # fallback: any *TIF_SK*.GR2 + any *DGB*.GR2 / blender-ish
            if not test.exists():
                cands = sorted(SOURCE_DIR.glob("*sk_tif*.GR2")) + sorted(SOURCE_DIR.glob("*DGB*.GR2"))
                if cands:
                    test = cands[0]
            self._json(200, {
                "ref": str(ref) if ref.exists() else "",
                "test": str(test) if test.exists() else "",
                "cut": 12.3,
                "source_dir": str(SOURCE_DIR),
            })
            return
        if path == "/report":
            report = SCRIPT_DIR / "comparison_report.html"
            if not report.exists():
                self._json(404, {"error": "No report yet. Run a compare first."})
                return
            data = report.read_bytes()
            self._bytes(200, data, "text/html; charset=utf-8")
            return
        # static helpers (viewer.js if needed)
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            fp = SCRIPT_DIR / name
            if fp.exists() and fp.is_file():
                ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
                self._bytes(200, fp.read_bytes(), ctype)
                return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""

        if path == "/api/run":
            with _job_lock:
                if _job["running"]:
                    self._json(409, {"error": "Compare already running"})
                    return
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "Invalid JSON"})
                return
            ref = payload.get("ref", "").strip()
            test = payload.get("test", "").strip()
            cut = float(payload.get("cut", 12.3) or 12.3)
            if not ref or not test:
                self._json(400, {"error": "Need both ref and test paths"})
                return
            if not Path(ref).exists() or not Path(test).exists():
                self._json(400, {"error": "File not found. Use Browse to upload, or paste a valid full path."})
                return
            t = threading.Thread(target=_run_job, args=(ref, test, cut), daemon=True)
            t.start()
            self._json(200, {"ok": True})
            return

        if path == "/api/upload":
            # multipart/form-data
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self._json(400, {"error": "Expected multipart upload"})
                return
            boundary = None
            for part in ctype.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part.split("=", 1)[1].strip().strip('"')
            if not boundary:
                self._json(400, {"error": "Missing boundary"})
                return
            try:
                filename, filedata = _parse_multipart(raw, boundary.encode("ascii"))
            except Exception as e:
                self._json(400, {"error": f"Upload parse failed: {e}"})
                return
            if not filename or not filedata:
                self._json(400, {"error": "No file in upload"})
                return
            safe = Path(filename).name
            if not safe.lower().endswith((".gr2", ".glb")):
                self._json(400, {"error": "Only .GR2 / .GLB allowed"})
                return
            uploads = SCRIPT_DIR / "_uploads"
            uploads.mkdir(exist_ok=True)
            dest = uploads / safe
            dest.write_bytes(filedata)
            self._json(200, {"path": str(dest), "name": safe, "size": len(filedata)})
            return

        self._json(404, {"error": "not found"})


def _parse_multipart(body: bytes, boundary: bytes):
    """Minimal multipart parser for a single file field."""
    sep = b"--" + boundary
    parts = body.split(sep)
    for part in parts:
        if part in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        if part.startswith(b"--"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        header_blob, _, data = part.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = header_blob.decode("utf-8", errors="replace")
        if "filename=" not in headers:
            continue
        # filename="..."
        fname = None
        for line in headers.split("\r\n"):
            if "filename=" in line:
                fname = line.split("filename=", 1)[1].strip().strip('"')
        if data.endswith(b"\r\n"):
            data = data[:-2]
        return fname, data
    return None, None


def main():
    # Ensure imports work when launched from bat
    import sys
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print("=" * 50)
    print(" BG3 Animation Compare UI")
    print("=" * 50)
    print(f" Open: {url}")
    print(" Leave this window open while using the site.")
    print(" Press Ctrl+C to stop.")
    print("=" * 50)
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
