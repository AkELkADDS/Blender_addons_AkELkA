import json
import re
import sys
from pathlib import Path

codec = Path(
    r"E:\Users\eloct\AppData\Roaming\Blender Foundation\Blender\5.2"
    r"\scripts\addons\gr2lab_blender\codec"
)
sys.path.insert(0, str(codec))

from granny_anim import build_anim_map, build_skeleton, extract_section_fixups
from gr2_codec import decode_sections, load_gr2

src = Path(
    r"D:\AkELkA_Mods\BG3_Mods\1_DGB_to_DGB_Anims\Source\0_Body_Base"
    r"\TIF_FS_Tail_Base.GR2"
)
gr2 = load_gr2(src)
decode_sections(gr2, True)
extract_section_fixups(gr2)
print("sections", [(i, s.header.uncompressed_size, s.header.compression) for i, s in enumerate(gr2.sections)])
anim = build_anim_map(gr2, source=str(src))
print("tracks", len(anim.tracks), "dur", anim.duration)
s0 = gr2.sections[0].uncompressed or b""
skel = build_skeleton(anim, s0)
for b in skel["bones"]:
    t = [round(x, 5) for x in b["rest_translation"]]
    r = [round(x, 4) for x in b["rest_rotation"]]
    print(f"{b['name']:24} parent={str(b.get('parent_name')):16} t={t} r={r}")
print("--- strings ---")
for i, sec in enumerate(gr2.sections):
    raw = sec.uncompressed or b""
    for m in re.finditer(rb"[\x20-\x7e]{4,}", raw):
        s = m.group().decode()
        if any(
            k in s
            for k in ("Tail", "Root", "Hip", "Spine", "Dummy", "Chest", "Bone")
        ):
            print(i, m.start(), s)
out = Path(r"C:\CG\Blender_addons_AkELkA\Blend Addon - GR2LAB\Standalone\C\_tail_skel.json")
out.write_text(json.dumps(skel, indent=2), encoding="utf-8")
print("WROTE", out)
