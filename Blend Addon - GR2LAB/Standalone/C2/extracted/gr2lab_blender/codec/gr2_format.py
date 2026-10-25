"""Granny GR2 v7 container structures (little-endian)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field, replace
from typing import List, Optional

MAGIC_SIZE = 16
SECTION_HEADER_SIZE = 44
HEADER_V7_SIZE = 0x48  # version through reserved3 (after magic block)

# Compression types used in section headers
COMP_NONE = 0
COMP_OODLE0 = 1
COMP_OODLE1 = 2
COMP_BITKNIT = 4


@dataclass
class Magic:
    signature: bytes
    headers_size: int
    header_format: int
    reserved1: int
    reserved2: int

    def pack(self) -> bytes:
        return self.signature + struct.pack(
            "<IIII",
            self.headers_size,
            self.header_format,
            self.reserved1,
            self.reserved2,
        )

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "Magic":
        sig = data[offset : offset + 16]
        headers_size, header_format, r1, r2 = struct.unpack_from("<IIII", data, offset + 16)
        return cls(sig, headers_size, header_format, r1, r2)


@dataclass
class Header:
    version: int
    file_size: int
    crc: int
    sections_offset: int
    num_sections: int
    root_type_section: int
    root_type_offset: int
    root_node_section: int
    root_node_offset: int
    tag: int
    extra_tags: List[int]
    string_table_crc: int
    reserved1: int
    reserved2: int
    reserved3: int

    def pack(self) -> bytes:
        parts = [
            struct.pack(
                "<IIIII",
                self.version,
                self.file_size,
                self.crc,
                self.sections_offset,
                self.num_sections,
            ),
            struct.pack("<II", self.root_type_section, self.root_type_offset),
            struct.pack("<II", self.root_node_section, self.root_node_offset),
            struct.pack("<I", self.tag),
            struct.pack("<4I", *self.extra_tags),
        ]
        if self.version >= 7:
            parts.append(
                struct.pack(
                    "<IIII",
                    self.string_table_crc,
                    self.reserved1,
                    self.reserved2,
                    self.reserved3,
                )
            )
        return b"".join(parts)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "Header":
        version, file_size, crc, sections_offset, num_sections = struct.unpack_from(
            "<IIIII", data, offset
        )
        o = offset + 20
        root_type_section, root_type_offset = struct.unpack_from("<II", data, o)
        o += 8
        root_node_section, root_node_offset = struct.unpack_from("<II", data, o)
        o += 8
        (tag,) = struct.unpack_from("<I", data, o)
        o += 4
        extra_tags = list(struct.unpack_from("<4I", data, o))
        o += 16
        string_table_crc = reserved1 = reserved2 = reserved3 = 0
        if version >= 7:
            string_table_crc, reserved1, reserved2, reserved3 = struct.unpack_from(
                "<IIII", data, o
            )
        return cls(
            version,
            file_size,
            crc,
            sections_offset,
            num_sections,
            root_type_section,
            root_type_offset,
            root_node_section,
            root_node_offset,
            tag,
            extra_tags,
            string_table_crc,
            reserved1,
            reserved2,
            reserved3,
        )

    def size(self) -> int:
        # Size of header struct alone (not including magic). V7 = 0x48.
        return 0x48 if self.version >= 7 else 0x38


@dataclass
class SectionHeader:
    compression: int
    offset_in_file: int
    compressed_size: int
    uncompressed_size: int
    alignment: int
    first16bit: int
    first8bit: int
    relocations_offset: int
    num_relocations: int
    mixed_marshalling_data_offset: int
    num_mixed_marshalling_data: int

    def pack(self) -> bytes:
        return struct.pack(
            "<11I",
            self.compression,
            self.offset_in_file,
            self.compressed_size,
            self.uncompressed_size,
            self.alignment,
            self.first16bit,
            self.first8bit,
            self.relocations_offset,
            self.num_relocations,
            self.mixed_marshalling_data_offset,
            self.num_mixed_marshalling_data,
        )

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> "SectionHeader":
        vals = struct.unpack_from("<11I", data, offset)
        return cls(*vals)


@dataclass
class Section:
    header: SectionHeader
    compressed: bytes = b""
    uncompressed: Optional[bytes] = None
    # Relocations / marshalling live after section data in the file for compressed GR2s.
    # For passthrough we keep the raw compressed blob only; fixups are inside compressed
    # stream offsets relative to section data in original layout.
    dirty: bool = False  # True if uncompressed was edited and needs recompress
    # Decompressed fixup tables (12-byte reloc entries / 16-byte mixed-marshall entries).
    # Populated by granny_anim.extract_section_fixups or encode_after_edit preserve path.
    relocations: Optional[bytes] = None
    mixed_marshalling: Optional[bytes] = None
    # When True, dirty section keeps original size + emits uncompressed payload with
    # preserved reloc/mm tables (in-place float patches).
    preserve_fixups: bool = False


@dataclass
class GR2File:
    magic: Magic
    header: Header
    sections: List[Section] = field(default_factory=list)
    # Bytes between end of section headers and first section data (usually empty).
    header_padding: bytes = b""
    # Trailing bytes after last section payload (usually empty).
    trailing: bytes = b""
    # Original full file bytes for CRC verification / passthrough.
    original_bytes: Optional[bytes] = None
    # Section headers as parsed (before mark_section_edited mutates sizes).
    original_section_headers: Optional[List[SectionHeader]] = None

    @property
    def headers_end(self) -> int:
        return self.magic.headers_size


def granny_crc32(data: bytes) -> int:
    """Granny uses CRC-32 with inverted init/final (same as zlib with ~)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def compute_file_crc(file_bytes: bytes, crc_field_offset: int) -> int:
    """CRC over entire file with the CRC field itself zeroed."""
    buf = bytearray(file_bytes)
    struct.pack_into("<I", buf, crc_field_offset, 0)
    return granny_crc32(bytes(buf))


def _original_fixup_stream(
    file_data: bytes,
    offset: int,
    count: int,
    entry_size: int,
    compression: int,
) -> bytes:
    """Slice the on-disk reloc/mm stream for a section (raw or BitKnit).

    BitKnit fixups are stored as ``uint32 csize`` + compressed bytes, often
    padded to 4-byte alignment before the next table.
    """
    if count <= 0 or offset <= 0 or offset >= len(file_data):
        return b""
    if compression == COMP_NONE:
        end = offset + count * entry_size
        return file_data[offset:end]
    # BitKnit / other compressed fixup tables
    if offset + 4 > len(file_data):
        return b""
    csize = struct.unpack_from("<I", file_data, offset)[0]
    raw_end = offset + 4 + csize
    if raw_end > len(file_data):
        return b""
    pad = (4 - (raw_end % 4)) % 4
    end = min(raw_end + pad, len(file_data))
    return file_data[offset:end]


def crc_field_offset() -> int:
    # magic(32) + version(4) + file_size(4) => crc at offset 40
    return MAGIC_SIZE + 8 + 4  # 16 + 16 reserved block? Wait:
    # Magic: 16 sig + 4 headersSize + 4 headerFormat + 4 r1 + 4 r2 = 32
    # Header: version@32, fileSize@36, crc@40
    return 40


def parse_gr2(data: bytes) -> GR2File:
    if len(data) < 32:
        raise ValueError("File too small for GR2 magic")
    magic = Magic.unpack(data, 0)
    if magic.header_format != 0:
        raise ValueError(f"Compressed GR2 headers (format={magic.header_format}) not supported")
    header = Header.unpack(data, 32)
    if header.version not in (6, 7):
        raise ValueError(f"Unsupported GR2 version {header.version}")
    if header.file_size != len(data):
        # Allow mismatch but warn via exception only if wildly off
        if header.file_size > len(data):
            raise ValueError(
                f"Header file_size {header.file_size} > actual {len(data)}"
            )

    sections_off = 32 + header.sections_offset
    # In LSLib, sectionsOffset is relative to start of Header (after magic),
    # and equals header.Size(). Magic is 32 bytes, so absolute = 32 + sectionsOffset.
    # Our dump showed sections at 0x68 = 104 = 32 + 72 = 32 + 0x48. Yes.
    if sections_off != magic.headers_size - header.num_sections * SECTION_HEADER_SIZE:
        # Prefer absolute layout from magic.headers_size
        pass

    # Absolute section table start: after magic+header = 32 + header.size()
    section_table_abs = 32 + header.size()
    if header.sections_offset != header.size():
        # Still trust sections_offset relative to header start
        section_table_abs = 32 + header.sections_offset

    sections: List[Section] = []
    for i in range(header.num_sections):
        sh = SectionHeader.unpack(data, section_table_abs + i * SECTION_HEADER_SIZE)
        blob = b""
        if sh.compressed_size > 0:
            end = sh.offset_in_file + sh.compressed_size
            if end > len(data):
                raise ValueError(f"Section {i} extends past EOF")
            blob = data[sh.offset_in_file : end]
        sections.append(Section(header=sh, compressed=blob))

    # Padding between end of section headers and first non-empty section data
    table_end = section_table_abs + header.num_sections * SECTION_HEADER_SIZE
    first_data = magic.headers_size
    header_padding = data[table_end:first_data] if table_end <= first_data else b""

    # Trailing after last section blob
    last_end = first_data
    for s in sections:
        if s.header.compressed_size > 0:
            last_end = max(last_end, s.header.offset_in_file + s.header.compressed_size)
    trailing = data[last_end:] if last_end < len(data) else b""

    return GR2File(
        magic=magic,
        header=header,
        sections=sections,
        header_padding=header_padding,
        trailing=trailing,
        original_bytes=data,
        original_section_headers=[replace(s.header) for s in sections],
    )


def encode_gr2(gr2: GR2File, recompute_crc: bool = True) -> bytes:
    """Encode GR2. Uses original compressed blobs unless a section is dirty."""
    magic = gr2.magic
    header = gr2.header
    sections = gr2.sections

    # Rebuild section payloads sequentially starting at headers_size
    cursor = magic.headers_size
    new_sections: List[SectionHeader] = []
    payloads: List[bytes] = []

    for sec in sections:
        sh = SectionHeader(
            compression=sec.header.compression,
            offset_in_file=cursor,
            compressed_size=sec.header.compressed_size,
            uncompressed_size=sec.header.uncompressed_size,
            alignment=sec.header.alignment,
            first16bit=sec.header.first16bit,
            first8bit=sec.header.first8bit,
            relocations_offset=sec.header.relocations_offset,
            num_relocations=sec.header.num_relocations,
            mixed_marshalling_data_offset=sec.header.mixed_marshalling_data_offset,
            num_mixed_marshalling_data=sec.header.num_mixed_marshalling_data,
        )

        if sec.dirty:
            if sec.uncompressed is None:
                raise ValueError("Dirty section missing uncompressed data")
            payload = sec.compressed  # caller must have set new compressed
            sh.compression = sec.header.compression
            sh.compressed_size = len(payload)
            sh.uncompressed_size = len(sec.uncompressed)
        else:
            payload = sec.compressed
            sh.compressed_size = len(payload)
            # Keep original offset layout for identity: if original offsets
            # already match sequential layout, fine; else preserve original
            # offsets when not dirty and original_bytes present.
            if gr2.original_bytes is not None and not any(s.dirty for s in sections):
                sh = SectionHeader(
                    compression=sec.header.compression,
                    offset_in_file=sec.header.offset_in_file,
                    compressed_size=sec.header.compressed_size,
                    uncompressed_size=sec.header.uncompressed_size,
                    alignment=sec.header.alignment,
                    first16bit=sec.header.first16bit,
                    first8bit=sec.header.first8bit,
                    relocations_offset=sec.header.relocations_offset,
                    num_relocations=sec.header.num_relocations,
                    mixed_marshalling_data_offset=sec.header.mixed_marshalling_data_offset,
                    num_mixed_marshalling_data=sec.header.num_mixed_marshalling_data,
                )
                payload = sec.compressed

        new_sections.append(sh)
        payloads.append(payload)
        if sh.compressed_size > 0 and (
            gr2.original_bytes is None or any(s.dirty for s in sections)
        ):
            cursor += sh.compressed_size

    # Passthrough identity path: reconstruct exact original layout
    if gr2.original_bytes is not None and not any(s.dirty for s in sections):
        # Rebuild from parsed fields — must match original byte-for-byte
        out = bytearray()
        out += magic.pack()
        # Temporarily zero CRC
        hdr = Header(
            version=header.version,
            file_size=header.file_size,
            crc=0 if recompute_crc else header.crc,
            sections_offset=header.sections_offset,
            num_sections=header.num_sections,
            root_type_section=header.root_type_section,
            root_type_offset=header.root_type_offset,
            root_node_section=header.root_node_section,
            root_node_offset=header.root_node_offset,
            tag=header.tag,
            extra_tags=list(header.extra_tags),
            string_table_crc=header.string_table_crc,
            reserved1=header.reserved1,
            reserved2=header.reserved2,
            reserved3=header.reserved3,
        )
        out += hdr.pack()
        # Section table should start immediately after header
        for sh in new_sections:
            out += sh.pack()
        out += gr2.header_padding
        # Place section payloads at their offsets
        if len(out) < magic.headers_size:
            out += b"\x00" * (magic.headers_size - len(out))
        # Expand to file size and write blobs
        file_size = header.file_size
        if len(out) < file_size:
            out += b"\x00" * (file_size - len(out))
        elif len(out) > file_size:
            out = out[:file_size]
        for sh, payload in zip(new_sections, payloads):
            if sh.compressed_size:
                out[sh.offset_in_file : sh.offset_in_file + sh.compressed_size] = payload
        # Reloc/mm tables are not inside sec.compressed. Hybrid COMP_NONE
        # stores them between section payloads; trailing-at-EOF misses them.
        orig = gr2.original_bytes
        if orig is not None:
            def _place_stream(offset: int, stream: bytes) -> None:
                if not stream or offset < 0:
                    return
                end = offset + len(stream)
                if end > len(out):
                    out.extend(b"\x00" * (end - len(out)))
                out[offset:end] = stream

            for sh in new_sections:
                if sh.num_relocations:
                    _place_stream(
                        sh.relocations_offset,
                        _original_fixup_stream(
                            orig,
                            sh.relocations_offset,
                            sh.num_relocations,
                            12,
                            sh.compression,
                        ),
                    )
                if sh.num_mixed_marshalling_data:
                    _place_stream(
                        sh.mixed_marshalling_data_offset,
                        _original_fixup_stream(
                            orig,
                            sh.mixed_marshalling_data_offset,
                            sh.num_mixed_marshalling_data,
                            16,
                            sh.compression,
                        ),
                    )
        # Relocations / mixed-marshalling leftover after last payload (BitKnit).
        if gr2.trailing:
            trail_off = file_size - len(gr2.trailing)
            if trail_off >= 0 and trail_off + len(gr2.trailing) <= file_size:
                out[trail_off : trail_off + len(gr2.trailing)] = gr2.trailing
            else:
                # Fallback: append after last written payload end
                last = magic.headers_size
                for sh in new_sections:
                    if sh.compressed_size:
                        last = max(last, sh.offset_in_file + sh.compressed_size)
                end = last + len(gr2.trailing)
                if len(out) < end:
                    out += b"\x00" * (end - len(out))
                out[last : last + len(gr2.trailing)] = gr2.trailing
        result = bytes(out)
        if recompute_crc:
            crc = compute_file_crc(result, crc_field_offset())
            result = bytearray(result)
            struct.pack_into("<I", result, crc_field_offset(), crc)
            result = bytes(result)
        return result

    # Dirty / rebuild path
    preserve_all_dirty = any(s.dirty for s in sections) and all(
        (not s.dirty) or s.preserve_fixups for s in sections
    )

    body_start = magic.headers_size
    if preserve_all_dirty:
        # Hybrid preserve encode:
        # - dirty sections: COMP_NONE payload + raw (decompressed) reloc/mm
        # - clean sections: keep original compressed payload + original
        #   BitKnit fixup streams (minimum interference / size)
        if any(s.relocations is None for s in sections if s.header.num_relocations):
            from granny_anim import extract_section_fixups

            extract_section_fixups(gr2)

        pos = body_start
        rebuilt_sections: List[SectionHeader] = []
        body = bytearray()
        orig = gr2.original_bytes
        for sec in sections:
            h = sec.header
            if sec.dirty and sec.preserve_fixups:
                if sec.uncompressed is None and h.uncompressed_size:
                    raise ValueError(
                        "Section missing uncompressed data for preserve encode"
                    )
                raw_payload = sec.uncompressed if sec.uncompressed is not None else b""
                reloc_blob = sec.relocations or b""
                mm_blob = sec.mixed_marshalling or b""
                n_reloc = len(reloc_blob) // 12
                n_mm = len(mm_blob) // 16
                # Prefer BitKnit payload + BitKnit fixup streams when
                # recompress_dirty set compression=COMP_BITKNIT.
                # Otherwise COMP_NONE + raw fixup tables (hybrid fallback).
                use_bitknit = h.compression == COMP_BITKNIT and sec.compressed is not None

                if use_bitknit:
                    from bitknit import pack_bitknit_fixup_stream

                    payload = sec.compressed
                    try:
                        reloc_stream = pack_bitknit_fixup_stream(
                            reloc_blob[: n_reloc * 12]
                        )
                        mm_stream = pack_bitknit_fixup_stream(mm_blob[: n_mm * 16])
                    except RuntimeError:
                        use_bitknit = False
                        payload = raw_payload
                        reloc_stream = reloc_blob[: n_reloc * 12]
                        mm_stream = mm_blob[: n_mm * 16]
                else:
                    payload = raw_payload
                    reloc_stream = reloc_blob[: n_reloc * 12]
                    mm_stream = mm_blob[: n_mm * 16]

                payload_off = pos
                body += payload
                pos += len(payload)
                reloc_off = pos
                body += reloc_stream
                pos += len(reloc_stream)
                mm_off = pos
                body += mm_stream
                pos += len(mm_stream)
                rebuilt_sections.append(
                    SectionHeader(
                        compression=COMP_BITKNIT if use_bitknit else COMP_NONE,
                        offset_in_file=payload_off,
                        compressed_size=len(payload),
                        uncompressed_size=len(raw_payload),
                        alignment=h.alignment or 4,
                        first16bit=h.first16bit if use_bitknit else len(raw_payload),
                        first8bit=h.first8bit if use_bitknit else len(raw_payload),
                        relocations_offset=reloc_off,
                        num_relocations=n_reloc,
                        mixed_marshalling_data_offset=mm_off,
                        num_mixed_marshalling_data=n_mm,
                    )
                )
                continue

            # Clean (or non-preserve dirty handled elsewhere): keep original
            # compressed blob + original on-disk fixup streams.
            payload = sec.compressed or b""
            reloc_stream = b""
            mm_stream = b""
            if orig is not None and h.num_relocations:
                reloc_stream = _original_fixup_stream(
                    orig, h.relocations_offset, h.num_relocations, 12, h.compression
                )
            if orig is not None and h.num_mixed_marshalling_data:
                mm_stream = _original_fixup_stream(
                    orig,
                    h.mixed_marshalling_data_offset,
                    h.num_mixed_marshalling_data,
                    16,
                    h.compression,
                )
            # Fallback: emit decompressed tables as COMP_NONE-style raw if we
            # cannot recover the original compressed fixup stream.
            if h.num_relocations and not reloc_stream and sec.relocations:
                reloc_stream = sec.relocations[: h.num_relocations * 12]
            if h.num_mixed_marshalling_data and not mm_stream and sec.mixed_marshalling:
                mm_stream = sec.mixed_marshalling[
                    : h.num_mixed_marshalling_data * 16
                ]

            payload_off = pos
            body += payload
            pos += len(payload)
            reloc_off = pos
            body += reloc_stream
            pos += len(reloc_stream)
            mm_off = pos
            body += mm_stream
            pos += len(mm_stream)
            rebuilt_sections.append(
                SectionHeader(
                    compression=h.compression,
                    offset_in_file=payload_off,
                    compressed_size=len(payload),
                    uncompressed_size=h.uncompressed_size,
                    alignment=h.alignment or 4,
                    first16bit=h.first16bit,
                    first8bit=h.first8bit,
                    relocations_offset=reloc_off,
                    num_relocations=h.num_relocations,
                    mixed_marshalling_data_offset=mm_off,
                    num_mixed_marshalling_data=h.num_mixed_marshalling_data,
                )
            )
    else:
        # Legacy: payload only, drop fixups
        rebuilt_sections = []
        body = bytearray()
        pos = body_start
        for i, (sec, payload) in enumerate(zip(sections, payloads)):
            sh = new_sections[i]
            nsh = SectionHeader(
                compression=sh.compression if payload else COMP_NONE,
                offset_in_file=pos if payload else pos,
                compressed_size=len(payload),
                uncompressed_size=sec.header.uncompressed_size
                if sec.uncompressed is None
                else len(sec.uncompressed),
                alignment=sec.header.alignment,
                first16bit=sec.header.first16bit,
                first8bit=sec.header.first8bit,
                relocations_offset=pos + len(payload) if sec.header.num_relocations else pos,
                num_relocations=0,
                mixed_marshalling_data_offset=pos + len(payload),
                num_mixed_marshalling_data=0,
            )
            if not payload and sec.header.uncompressed_size == 0:
                nsh.offset_in_file = pos
                nsh.relocations_offset = pos
                nsh.mixed_marshalling_data_offset = pos
            rebuilt_sections.append(nsh)
            body += payload
            pos += len(payload)

    file_size = body_start + len(body)
    magic2 = Magic(
        signature=magic.signature,
        headers_size=body_start,
        header_format=0,
        reserved1=magic.reserved1,
        reserved2=magic.reserved2,
    )
    hdr = Header(
        version=header.version,
        file_size=file_size,
        crc=0,
        sections_offset=header.size(),
        num_sections=len(rebuilt_sections),
        root_type_section=header.root_type_section,
        root_type_offset=header.root_type_offset,
        root_node_section=header.root_node_section,
        root_node_offset=header.root_node_offset,
        tag=header.tag,
        extra_tags=list(header.extra_tags),
        string_table_crc=header.string_table_crc,
        reserved1=header.reserved1,
        reserved2=header.reserved2,
        reserved3=header.reserved3,
    )
    out = bytearray()
    out += magic2.pack()
    out += hdr.pack()
    for sh in rebuilt_sections:
        out += sh.pack()
    if len(out) < body_start:
        out += b"\x00" * (body_start - len(out))
    out += body
    result = bytes(out)
    if recompute_crc:
        crc = compute_file_crc(result, crc_field_offset())
        result = bytearray(result)
        struct.pack_into("<I", result, crc_field_offset(), crc)
        result = bytes(result)
    return result
