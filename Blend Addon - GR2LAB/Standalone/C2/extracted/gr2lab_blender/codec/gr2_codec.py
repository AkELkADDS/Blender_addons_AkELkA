"""GR2 decode/encode: parse container, decompress sections, passthrough or recompress."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from bitknit import compress_section, decompress_section
from gr2_format import (
    COMP_BITKNIT,
    COMP_NONE,
    GR2File,
    Section,
    SectionHeader,
    encode_gr2,
    parse_gr2,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_gr2(path: Path | str) -> GR2File:
    data = Path(path).read_bytes()
    return parse_gr2(data)


def decode_sections(gr2: GR2File, decompress: bool = True) -> GR2File:
    """Optionally decompress all sections into section.uncompressed."""
    if not decompress:
        return gr2
    for i, sec in enumerate(gr2.sections):
        if sec.header.uncompressed_size == 0:
            sec.uncompressed = b""
            continue
        if sec.header.compression == COMP_NONE:
            sec.uncompressed = sec.compressed
            continue
        sec.uncompressed = decompress_section(
            sec.header.compression,
            sec.compressed,
            sec.header.uncompressed_size,
            sec.header.first16bit,
            sec.header.first8bit,
        )
        if len(sec.uncompressed) != sec.header.uncompressed_size:
            raise ValueError(
                f"Section {i}: got {len(sec.uncompressed)} bytes, "
                f"expected {sec.header.uncompressed_size}"
            )
    return gr2


def encode_passthrough(gr2: GR2File) -> bytes:
    """Encode without changing compressed payloads — must be byte-identical.

    When nothing is dirty, return the original file bytes. Rebuilding a hybrid
    COMP_NONE GR2 from payloads + trailing zeros the interleaved reloc/mm
    tables (bone names live in S1 relocs → S0 strings).
    """
    if gr2.original_bytes is not None and not any(s.dirty for s in gr2.sections):
        return gr2.original_bytes
    return encode_gr2(gr2, recompute_crc=True)


def mark_section_edited(
    gr2: GR2File,
    index: int,
    new_uncompressed: bytes,
    *,
    preserve_fixups: bool = False,
) -> None:
    sec = gr2.sections[index]
    sec.uncompressed = new_uncompressed
    sec.dirty = True
    sec.header.uncompressed_size = len(new_uncompressed)
    if preserve_fixups:
        sec.preserve_fixups = True


def recompress_dirty(gr2: GR2File) -> None:
    for sec in gr2.sections:
        if not sec.dirty:
            continue
        assert sec.uncompressed is not None
        if len(sec.uncompressed) == 0:
            sec.compressed = b""
            sec.header.compression = COMP_NONE
            sec.header.compressed_size = 0
            sec.header.uncompressed_size = 0
            continue

        # Prefer BitKnit via native helper; fall back to COMP_NONE.
        if sec.preserve_fixups:
            if sec.relocations is None or sec.mixed_marshalling is None:
                from granny_anim import extract_section_fixups

                extract_section_fixups(gr2)
            orig_usize = sec.header.uncompressed_size
            orig_first16 = sec.header.first16bit
            orig_first8 = sec.header.first8bit
            ctype, blob = compress_section(sec.uncompressed, prefer_bitknit=True)
            sec.compressed = blob
            sec.header.compression = ctype
            sec.header.compressed_size = len(blob)
            sec.header.uncompressed_size = len(sec.uncompressed)
            if ctype == COMP_NONE or len(sec.uncompressed) != orig_usize:
                sec.header.first16bit = len(sec.uncompressed)
                sec.header.first8bit = len(sec.uncompressed)
            else:
                sec.header.first16bit = orig_first16
                sec.header.first8bit = orig_first8
            # reloc/mm offsets filled in encode_gr2 when packing body
            continue

        ctype, blob = compress_section(sec.uncompressed, prefer_bitknit=True)
        sec.compressed = blob
        sec.header.compression = ctype
        sec.header.compressed_size = len(blob)
        sec.header.uncompressed_size = len(sec.uncompressed)
        sec.header.first16bit = len(sec.uncompressed)
        sec.header.first8bit = len(sec.uncompressed)
        if ctype != COMP_BITKNIT:
            # Legacy uncompressed path drops fixups
            sec.header.num_relocations = 0
            sec.header.num_mixed_marshalling_data = 0
            sec.header.relocations_offset = 0
            sec.header.mixed_marshalling_data_offset = 0


def encode_size_breakdown_from_headers(
    orig_headers: List[SectionHeader],
    enc_sections: List[Section],
    *,
    original_size: int,
    encoded_size: int,
) -> Dict[str, Any]:
    """SIZE.json from already-parsed headers (no extra parse_gr2)."""
    sections = []
    for i, (oh, esec) in enumerate(zip(orig_headers, enc_sections)):
        eh = esec.header
        sections.append(
            {
                "index": i,
                "orig_compression": oh.compression,
                "enc_compression": eh.compression,
                "orig_compressed_size": oh.compressed_size,
                "enc_compressed_size": eh.compressed_size,
                "orig_uncompressed_size": oh.uncompressed_size,
                "enc_uncompressed_size": eh.uncompressed_size,
                "delta_on_disk": eh.compressed_size - oh.compressed_size,
                "kept_bitknit": (
                    oh.compression == COMP_BITKNIT
                    and eh.compression == COMP_BITKNIT
                ),
                "became_uncompressed": (
                    oh.compression != COMP_NONE and eh.compression == COMP_NONE
                ),
            }
        )
    dirty_uncomp = sum(
        s["enc_compressed_size"] - s["orig_compressed_size"]
        for s in sections
        if s["became_uncompressed"]
    )
    return {
        "original_size": original_size,
        "encoded_size": encoded_size,
        "delta": encoded_size - original_size,
        "sections": sections,
        "dirty_section_decompress_delta": dirty_uncomp,
        "note": (
            "Edited sections use BitKnit when tools/native/bk_compress.exe works; "
            "otherwise COMP_NONE (see tools/native/SPIKE_RESULT.md). "
            "Untouched sections keep original BitKnit."
        ),
    }


def encode_size_breakdown(original: bytes, encoded: bytes) -> Dict[str, Any]:
    """Explain why an edited encode differs in size from the original GR2."""
    from gr2_format import parse_gr2

    o = parse_gr2(original)
    e = parse_gr2(encoded)
    orig_headers = o.original_section_headers or [s.header for s in o.sections]
    return encode_size_breakdown_from_headers(
        orig_headers,
        e.sections,
        original_size=len(original),
        encoded_size=len(encoded),
    )


def encode_after_edit(gr2: GR2File) -> bytes:
    # Auto-enable preserve_fixups when dirty sections keep original uncompressed size
    # OR when caller already set preserve_fixups (grown section with retargeted relocs).
    orig_headers = gr2.original_section_headers
    if orig_headers is None and gr2.original_bytes is not None:
        orig_headers = parse_gr2(gr2.original_bytes).original_section_headers
    if orig_headers is not None:
        for i, sec in enumerate(gr2.sections):
            if not sec.dirty or sec.uncompressed is None:
                continue
            if sec.preserve_fixups:
                # Grown or in-place: ensure fixup tables are present
                if sec.relocations is None:
                    try:
                        from granny_anim import extract_section_fixups

                        extract_section_fixups(gr2)
                    except Exception:
                        pass
                continue
            from granny_anim import extract_section_fixups

            if sec.relocations is None:
                try:
                    extract_section_fixups(gr2)
                except Exception:
                    pass
            oh = orig_headers[i] if i < len(orig_headers) else sec.header
            if len(sec.uncompressed) == oh.uncompressed_size:
                if sec.relocations is not None and sec.mixed_marshalling is not None:
                    sec.preserve_fixups = True
                    sec.header.num_relocations = oh.num_relocations
                    sec.header.num_mixed_marshalling_data = oh.num_mixed_marshalling_data
            elif sec.relocations is not None and sec.mixed_marshalling is not None:
                # Section grew (e.g. quat_k16 KnotsControls append) — still preserve fixups
                sec.preserve_fixups = True
                sec.header.num_relocations = len(sec.relocations) // 12
                sec.header.num_mixed_marshalling_data = len(sec.mixed_marshalling) // 16

    recompress_dirty(gr2)
    return encode_gr2(gr2, recompute_crc=True)


def build_uncompressed_gr2(gr2: GR2File) -> bytes:
    """Write a GR2 with compression=0 and uncompressed section payloads (no fixups)."""
    if any(s.uncompressed is None for s in gr2.sections):
        decode_sections(gr2, decompress=True)

    # Clone structure with uncompressed payloads
    new = GR2File(
        magic=gr2.magic,
        header=gr2.header,
        sections=[],
        header_padding=b"",
        trailing=b"",
        original_bytes=None,
    )
    for sec in gr2.sections:
        data = sec.uncompressed or b""
        sh = SectionHeader(
            compression=COMP_NONE,
            offset_in_file=0,
            compressed_size=len(data),
            uncompressed_size=len(data),
            alignment=sec.header.alignment or 4,
            first16bit=len(data),
            first8bit=len(data),
            relocations_offset=0,
            num_relocations=0,
            mixed_marshalling_data_offset=0,
            num_mixed_marshalling_data=0,
        )
        new.sections.append(
            Section(header=sh, compressed=data, uncompressed=data, dirty=True)
        )
    # Force dirty rebuild path
    for s in new.sections:
        s.dirty = True
    return encode_gr2(new, recompute_crc=True)


def gr2_to_dict(gr2: GR2File, include_blobs: bool = False) -> Dict[str, Any]:
    sections: List[Dict[str, Any]] = []
    for i, sec in enumerate(gr2.sections):
        d: Dict[str, Any] = {
            "index": i,
            "compression": sec.header.compression,
            "offset_in_file": sec.header.offset_in_file,
            "compressed_size": sec.header.compressed_size,
            "uncompressed_size": sec.header.uncompressed_size,
            "alignment": sec.header.alignment,
            "first16bit": sec.header.first16bit,
            "first8bit": sec.header.first8bit,
            "num_relocations": sec.header.num_relocations,
            "num_mixed_marshalling_data": sec.header.num_mixed_marshalling_data,
            "dirty": sec.dirty,
            "compressed_sha256": sha256_bytes(sec.compressed) if sec.compressed else None,
            "uncompressed_sha256": (
                sha256_bytes(sec.uncompressed) if sec.uncompressed is not None else None
            ),
        }
        if include_blobs:
            d["compressed_hex_head"] = sec.compressed[:32].hex()
            if sec.uncompressed is not None:
                d["uncompressed_hex_head"] = sec.uncompressed[:32].hex()
        sections.append(d)
    return {
        "magic": {
            "signature_hex": gr2.magic.signature.hex(),
            "headers_size": gr2.magic.headers_size,
            "header_format": gr2.magic.header_format,
        },
        "header": {
            "version": gr2.header.version,
            "file_size": gr2.header.file_size,
            "crc": f"0x{gr2.header.crc:08X}",
            "sections_offset": gr2.header.sections_offset,
            "num_sections": gr2.header.num_sections,
            "root_type": [gr2.header.root_type_section, gr2.header.root_type_offset],
            "root_node": [gr2.header.root_node_section, gr2.header.root_node_offset],
            "tag": f"0x{gr2.header.tag:08X}",
            "extra_tags": [f"0x{t:08X}" for t in gr2.header.extra_tags],
        },
        "sections": sections,
        "original_sha256": sha256_bytes(gr2.original_bytes) if gr2.original_bytes else None,
    }


def first_diff(a: bytes, b: bytes) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None
