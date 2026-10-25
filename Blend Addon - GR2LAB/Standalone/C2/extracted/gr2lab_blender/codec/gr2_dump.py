"""Readable dumps from a decoded GR2 (strings + JSON metadata)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from gr2_codec import decode_sections, gr2_to_dict, load_gr2, sha256_bytes
from timing import maybe_timer

from paths import IN_GR2, list_input_gr2s

ROOT = Path(__file__).resolve().parent.parent  # addon root
TOOLS = ROOT / "tools"


def extract_ascii_strings(data: bytes, min_len: int = 4) -> List[str]:
    pat = re.compile(rb"[\x20-\x7e]{%d,}" % min_len)
    return [m.group().decode("ascii", "replace") for m in pat.finditer(data)]


def extract_utf16le_strings(data: bytes, min_len: int = 4) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(data)
    while i + 2 <= n:
        chars = []
        j = i
        while j + 2 <= n:
            c = data[j] | (data[j + 1] << 8)
            if 0x20 <= c <= 0x7E:
                chars.append(chr(c))
                j += 2
            else:
                break
        if len(chars) >= min_len:
            out.append("".join(chars))
            i = j + 2
        else:
            i += 2
    return out


def dump_gr2(
    path: Path,
    out_dir: Path,
    *,
    timer=None,
    scan_utf16: bool = False,
) -> dict:
    """Write section bins + compact debug sidecars. UTF-16 string scan is optional
    (unused by encode; the naive walk of S0 is expensive on long clips)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with maybe_timer(timer, "read_gr2"):
        gr2 = load_gr2(path)
    with maybe_timer(timer, "decompress"):
        decode_sections(gr2, decompress=True)

    all_strings: List[Tuple[int, str]] = []
    with maybe_timer(timer, "write_bins"):
        for i, sec in enumerate(gr2.sections):
            raw = sec.uncompressed or b""
            if raw:
                (out_dir / f"section_{i}.bin").write_bytes(raw)

    with maybe_timer(timer, "strings_ascii"):
        for i, sec in enumerate(gr2.sections):
            raw = sec.uncompressed or b""
            for s in extract_ascii_strings(raw):
                all_strings.append((i, s))
    if scan_utf16:
        with maybe_timer(timer, "strings_utf16"):
            for i, sec in enumerate(gr2.sections):
                raw = sec.uncompressed or b""
                for s in extract_utf16le_strings(raw):
                    all_strings.append((i, "[u16] " + s))
    elif timer is not None:
        timer.add("strings_utf16", 0.0)

    seen = set()
    lines = []
    for sec_i, s in all_strings:
        key = (sec_i, s)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[S{sec_i}] {s}")
    (out_dir / "strings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with maybe_timer(timer, "decoded_json"):
        meta = gr2_to_dict(gr2, include_blobs=False)
        (out_dir / "decoded.json").write_text(
            json.dumps(meta, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
    try:
        rel_name = path.resolve().relative_to(IN_GR2.resolve()).as_posix()
    except (ValueError, OSError):
        rel_name = path.name
    (out_dir / "SOURCE.txt").write_text(
        f"name={rel_name}\n"
        f"path={path.resolve()}\n",
        encoding="utf-8",
    )
    try:
        from granny_anim import build_anim_map, extract_section_fixups, write_anim_map

        with maybe_timer(timer, "anim_map"):
            extract_section_fixups(gr2)
            for i, sec in enumerate(gr2.sections):
                if sec.relocations:
                    (out_dir / f"section_{i}.relocations.bin").write_bytes(sec.relocations)
                if sec.mixed_marshalling:
                    (out_dir / f"section_{i}.mm.bin").write_bytes(sec.mixed_marshalling)
            anim = build_anim_map(gr2, source=str(path))
            write_anim_map(anim, out_dir / "anim_map.json")
            from granny_anim import build_skeleton, write_skeleton

            write_skeleton(
                build_skeleton(anim, gr2.sections[0].uncompressed or b""),
                out_dir / "skeleton.json",
            )
    except Exception:
        pass
    return meta


def try_divine_dae(path: Path, out_dir: Path) -> bool:
    """Convert GR2 → DAE via LSLib Divine.exe (optional; used for Body_Base bind)."""
    addon_root = Path(__file__).resolve().parent.parent  # gr2lab_blender/
    candidates = [
        addon_root / "tools" / "ExportTool-v1.20.4" / "Packed" / "Tools" / "Divine.exe",
        TOOLS / "ExportTool-v1.20.4" / "Packed" / "Tools" / "Divine.exe",
        addon_root / "tools" / "Divine.exe",
    ]

    divine = next((c for c in candidates if c.is_file()), None)
    if divine is None:
        return False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dae = out_dir / "decoded.dae"
    native_dll = addon_root / "native" / "granny2.dll"
    granny_dst = divine.parent / "granny2.dll"
    try:
        src = native_dll if native_dll.is_file() else (divine.parent.parent / "granny2.dll")
        if src.is_file() and (
            not granny_dst.is_file() or granny_dst.stat().st_size != src.stat().st_size
        ):
            granny_dst.write_bytes(src.read_bytes())
    except OSError:
        pass

    cmd = [
        str(divine),
        "-g",
        "bg3",
        "-a",
        "convert-model",
        "-s",
        str(Path(path).resolve()),
        "-d",
        str(out_dae.resolve()),
    ]
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(divine.parent),
        )
        return out_dae.is_file() and out_dae.stat().st_size > 0
    except Exception:
        return False


if __name__ == "__main__":
    from paths import make_run_dirs

    files = list_input_gr2s()
    if not files:
        raise SystemExit(f"No GR2 in {IN_GR2}")
    gr2_path = files[0]
    run_id, files_dir, _ = make_run_dirs("decode")
    print("Dumping", gr2_path.name, "→", files_dir)
    dump_gr2(gr2_path, files_dir)
    ok = try_divine_dae(gr2_path, files_dir)
    print("Divine DAE:", ok)
    print("Run:", run_id)
