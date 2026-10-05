"""Parse Granny skeleton bones from a Tail_Base GR2 (no animation tracks)."""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

codec = Path(
    r"E:\Users\eloct\AppData\Roaming\Blender Foundation\Blender\5.2"
    r"\scripts\addons\gr2lab_blender\codec"
)
sys.path.insert(0, str(codec))

from granny_anim import extract_section_fixups
from gr2_codec import decode_sections, load_gr2

SRC_DIR = Path(r"D:\AkELkA_Mods\BG3_Mods\1_DGB_to_DGB_Anims\Source\0_Body_Base")
OUT_DIR = Path(
    r"E:\Users\eloct\AppData\Roaming\Blender Foundation\Blender\5.2"
    r"\scripts\addons\gr2lab_blender\Body_Base"
)

SKIP = {"Bones", "BoneBindings", "BoneName", "BonesForTriangle", "TriangleToBoneIndices"}
STRIDE = 176


def _s0_strings(s0: bytes) -> dict[int, str]:
    out = {}
    for m in re.finditer(rb"[\x20-\x7e]{3,}", s0):
        out[m.start()] = m.group().decode()
    return out


def parse_skeleton(gr2) -> dict:
    extract_section_fixups(gr2)
    secs = [bytes(s.uncompressed or b"") for s in gr2.sections]
    s0_map = _s0_strings(secs[0])
    raw = secs[2]
    rel = getattr(gr2.sections[2], "relocations", None) or b""
    reloc0 = {}
    for i in range(0, len(rel), 12):
        off, sec_i, toff = struct.unpack_from("<III", rel, i)
        if sec_i == 0:
            reloc0[off] = toff

    n = len(raw) // STRIDE
    bones = []
    for i in range(n):
        p = i * STRIDE
        toff = reloc0.get(p)
        name = s0_map.get(toff, "") if toff is not None else ""
        if not name or name in SKIP:
            continue
        parent = struct.unpack_from("<i", raw, p + 8)[0]
        flags = struct.unpack_from("<I", raw, p + 16)[0]
        tx, ty, tz = struct.unpack_from("<3f", raw, p + 20)
        qx, qy, qz, qw = struct.unpack_from("<4f", raw, p + 32)
        ss = struct.unpack_from("<9f", raw, p + 48)
        sx, sy, sz = abs(ss[0]), abs(ss[4]), abs(ss[8])
        bones.append(
            {
                "name": name,
                "parent": parent,
                "flags": flags,
                "bind_translation": [round(float(tx), 6), round(float(ty), 6), round(float(tz), 6)],
                "bind_rotation": [
                    round(float(qx), 6),
                    round(float(qy), 6),
                    round(float(qz), 6),
                    round(float(qw), 6),
                ],
                "bind_scale": [round(float(sx), 6), round(float(sy), 6), round(float(sz), 6)],
            }
        )

    # parent indices are into the full 176-stride array, including skipped rows
    index_names = []
    for i in range(n):
        p = i * STRIDE
        toff = reloc0.get(p)
        index_names.append(s0_map.get(toff, "") if toff is not None else "")

    out_bones = []
    for b in bones:
        pi = b.pop("parent")
        b.pop("flags")
        parent_name = None
        if 0 <= pi < len(index_names) and index_names[pi] and index_names[pi] not in SKIP:
            parent_name = index_names[pi]
        b["parent_name"] = parent_name
        out_bones.append(b)

    return {
        "format": "gr2lab_bind",
        "version": 1,
        "root": out_bones[0]["name"] if out_bones else "",
        "bones": out_bones,
        "_n_slots": n,
    }


def main() -> None:
    for p in sorted(SRC_DIR.glob("*_Tail_Base.GR2")):
        gr2 = load_gr2(p)
        decode_sections(gr2, True)
        doc = parse_skeleton(gr2)
        doc["source_gr2"] = p.name
        n_slots = doc.pop("_n_slots")
        print(p.name, "bones", len(doc["bones"]), "slots", n_slots)
        for b in doc["bones"]:
            print(
                f"  {b['name']:20} <- {str(b['parent_name']):16} "
                f"t={b['bind_translation']} r={b['bind_rotation']} s={b['bind_scale']}"
            )
        out = OUT_DIR / f"{p.stem}.bind_pose.json"
        out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print("WROTE", out.name)


if __name__ == "__main__":
    main()
