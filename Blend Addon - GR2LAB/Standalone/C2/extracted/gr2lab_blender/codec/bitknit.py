"""BitKnit / Granny section compression via granny2.dll."""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

_CODEC = Path(__file__).resolve().parent
_ADDON = _CODEC.parent
_NATIVE = _ADDON / "native"
_TOOLS = _ADDON / "tools"  # optional; not required for decompress

_dll: Optional[ctypes.CDLL] = None
_helper_checked: bool = False
_helper_path: Optional[Path] = None

# Section compression type codes (GR2 section header)
COMP_NONE = 0
COMP_OODLE0 = 1
COMP_OODLE1 = 2
COMP_BITKNIT = 4


def _find_granny2() -> Path:
    candidates = [
        _NATIVE / "granny2.dll",
        _ADDON / "granny2.dll",
        _TOOLS / "granny2.dll",
        _TOOLS / "ExportTool-v1.20.4" / "Packed" / "granny2.dll",
        _TOOLS / "ExportTool-v1.15.13" / "ExportTool-v1.15.13" / "granny2.dll",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "granny2.dll not found — expected beside addon at native/granny2.dll"
    )


def _find_bk_compress() -> Optional[Path]:
    """Locate optional bk_compress.exe (BitKnit recompress — currently dead)."""
    global _helper_checked, _helper_path
    if _helper_checked:
        return _helper_path
    _helper_checked = True
    for cand in (
        _NATIVE / "bk_compress.exe",
        _TOOLS / "native" / "bk_compress.exe",
    ):
        if cand.is_file():
            _helper_path = cand
            return _helper_path
    _helper_path = None
    return None


def load_granny2(path: Optional[Path] = None) -> ctypes.CDLL:
    global _dll
    if _dll is not None:
        return _dll
    dll_path = path or _find_granny2()
    os.add_dll_directory(str(dll_path.parent))
    _dll = ctypes.CDLL(str(dll_path))
    return _dll


def _setup_file_decomp(dll: ctypes.CDLL) -> None:
    Begin = dll.GrannyBeginFileDecompression
    Begin.restype = ctypes.c_void_p
    Begin.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    Inc = dll.GrannyDecompressIncremental
    Inc.restype = ctypes.c_int32
    Inc.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    End = dll.GrannyEndFileDecompression
    End.restype = ctypes.c_int32
    End.argtypes = [ctypes.c_void_p]


def decompress_bitknit(compressed: bytes, uncompressed_size: int) -> bytes:
    if uncompressed_size == 0:
        return b""
    if not compressed:
        raise ValueError("Empty compressed buffer for non-zero uncompressed size")
    return _decompress_via_file_api(
        load_granny2(), compressed, uncompressed_size, compression_type=COMP_BITKNIT
    )


def _decompress_via_file_api(
    dll: ctypes.CDLL, compressed: bytes, uncompressed_size: int, compression_type: int
) -> bytes:
    _setup_file_decomp(dll)
    Begin = dll.GrannyBeginFileDecompression
    Inc = dll.GrannyDecompressIncremental
    End = dll.GrannyEndFileDecompression

    out = (ctypes.c_ubyte * uncompressed_size)()
    work_size = max(0x40000, len(compressed) * 2, 0x10000)
    work = (ctypes.c_ubyte * work_size)()

    state = Begin(
        compression_type,
        0,
        uncompressed_size,
        ctypes.cast(out, ctypes.c_void_p),
        work_size,
        ctypes.cast(work, ctypes.c_void_p),
    )
    if not state:
        raise RuntimeError("GrannyBeginFileDecompression returned NULL")

    src = (ctypes.c_ubyte * len(compressed)).from_buffer_copy(compressed)
    ok = Inc(state, len(compressed), ctypes.cast(src, ctypes.c_void_p))
    end_ok = End(state)
    if not ok or not end_ok:
        raise RuntimeError(f"Incremental decompress failed ok={ok} end={end_ok}")
    return bytes(out)


def _compress_via_helper(uncompressed: bytes) -> Optional[bytes]:
    """Run tools/native/bk_compress.exe in a subprocess (crash-isolated).

    Returns BitKnit blob on success, or None if helper missing/fails.
    See tools/native/SPIKE_RESULT.md — current granny2.dll compress path is dead.
    """
    exe = _find_bk_compress()
    if exe is None:
        return None
    try:
        dll = _find_granny2()
    except FileNotFoundError:
        return None

    with tempfile.TemporaryDirectory(prefix="bk_comp_") as td:
        tdir = Path(td)
        raw_path = tdir / "raw.bin"
        out_path = tdir / "out.bk"
        raw_path.write_bytes(uncompressed)
        try:
            proc = subprocess.run(
                [
                    str(exe),
                    str(raw_path),
                    str(out_path),
                    "--dll",
                    str(dll),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0 or not out_path.is_file():
            return None
        blob = out_path.read_bytes()
        if not blob:
            return None
        try:
            if decompress_bitknit(blob, len(uncompressed)) != uncompressed:
                return None
        except Exception:
            return None
        return blob


def compress_section(uncompressed: bytes, prefer_bitknit: bool = True) -> Tuple[int, bytes]:
    """Compress section data. Returns (compression_type, compressed_bytes).

    Tries the native ``bk_compress.exe`` helper (subprocess) when prefer_bitknit
    is True. The shipped ExportTool granny2.dll currently cannot compress
    (NULL / AV — see tools/native/SPIKE_RESULT.md), so this falls back to
    COMP_NONE. Hybrid encode keeps untouched sections as original BitKnit.
    """
    if not uncompressed:
        return COMP_NONE, b""
    if prefer_bitknit:
        blob = _compress_via_helper(uncompressed)
        if blob is not None:
            return COMP_BITKNIT, blob
    return COMP_NONE, uncompressed


def compress_bitknit(uncompressed: bytes) -> bytes:
    """Compress with BitKnit, or raise if unavailable (use compress_section instead)."""
    ctype, blob = compress_section(uncompressed, prefer_bitknit=True)
    if ctype != COMP_BITKNIT:
        raise RuntimeError(
            "BitKnit compress unavailable (bk_compress helper failed; "
            "see tools/native/SPIKE_RESULT.md). "
            "use compress_section() which falls back to uncompressed"
        )
    return blob


def pack_bitknit_fixup_stream(raw_table: bytes) -> bytes:
    """Pack reloc/mm as Granny BitKnit on-disk stream: uint32 csize + blob + pad4."""
    if not raw_table:
        return b""
    blob = compress_bitknit(raw_table)
    stream = struct.pack("<I", len(blob)) + blob
    pad = (4 - (len(stream) % 4)) % 4
    if pad:
        stream += b"\x00" * pad
    return stream


def bitknit_compress_available() -> bool:
    """True if the native helper exists and can compress a tiny buffer."""
    if _find_bk_compress() is None:
        return False
    ctype, _ = compress_section(b"\x00" * 64, prefer_bitknit=True)
    return ctype == COMP_BITKNIT


def _try_compress_contents_of_memory(uncompressed: bytes, compression_type: int) -> Optional[bytes]:
    dll = load_granny2()
    Comp = dll.GrannyCompressContentsOfMemory
    Comp.restype = ctypes.c_int32
    Comp.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    src = (ctypes.c_ubyte * len(uncompressed)).from_buffer_copy(uncompressed)
    out_size = ctypes.c_int32(0)
    out_ptr = ctypes.c_void_p()
    ok = Comp(
        compression_type,
        0,
        len(uncompressed),
        ctypes.cast(src, ctypes.c_void_p),
        ctypes.byref(out_size),
        ctypes.byref(out_ptr),
    )
    if not ok or not out_ptr or out_size.value <= 0:
        return None
    result = ctypes.string_at(out_ptr, out_size.value)
    for free_name in ("GrannyFreeBuilderResult", "GrannyFreeMemoryWriterBuffer"):
        try:
            Free = getattr(dll, free_name)
            Free.argtypes = [ctypes.c_void_p]
            Free(out_ptr)
            break
        except Exception:
            continue
    return result


def _try_begin_file_compression(uncompressed: bytes, compression_type: int) -> Optional[bytes]:
    dll = load_granny2()
    Begin = dll.GrannyBeginFileCompression
    Begin.restype = ctypes.c_void_p
    Begin.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    Write = dll.GrannyWriteFileChunk
    Write.restype = ctypes.c_int32
    Write.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    End = dll.GrannyEndFileCompression
    End.restype = ctypes.c_int32
    End.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_void_p),
    ]

    buf_size = ctypes.c_int32(0)
    buf_ptr = ctypes.c_void_p()
    state = Begin(
        len(uncompressed),
        compression_type,
        0,
        ctypes.byref(buf_size),
        ctypes.byref(buf_ptr),
    )
    if not state:
        return None

    if buf_size.value > 0 and not buf_ptr:
        return None

    src = (ctypes.c_ubyte * len(uncompressed)).from_buffer_copy(uncompressed)
    if not Write(state, len(uncompressed), ctypes.cast(src, ctypes.c_void_p)):
        return None
    csize = ctypes.c_int32(0)
    cptr = ctypes.c_void_p()
    if not End(state, ctypes.byref(csize), ctypes.byref(cptr)):
        return None
    if not cptr or csize.value <= 0:
        return None
    return ctypes.string_at(cptr, csize.value)


def decompress_section(
    compression: int,
    compressed: bytes,
    uncompressed_size: int,
    first16bit: int = 0,
    first8bit: int = 0,
) -> bytes:
    if uncompressed_size == 0:
        return b""
    if compression == COMP_NONE:
        if len(compressed) >= uncompressed_size:
            return compressed[:uncompressed_size]
        return compressed + b"\x00" * (uncompressed_size - len(compressed))
    return _decompress_via_file_api(
        load_granny2(), compressed, uncompressed_size, compression_type=compression
    )
