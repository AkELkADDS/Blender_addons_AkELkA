"""Background Blender entry: render a hover preview without touching the open file."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :]
    gr2_path, out_path = argv[0], argv[1]
    unpacked = argv[2] if len(argv) > 2 else ""
    if unpacked:
        os.environ["GR2LAB_BG3_UNPACKED"] = unpacked
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import gr2lab_blender

    gr2lab_blender.register()
    import bpy
    from gr2lab_blender.anim_browser import build_scene_preview

    data = build_scene_preview(bpy.context, gr2_path, inplace=True, fast=True)
    Path(out_path).write_text(json.dumps(data), encoding="utf-8")
    print("PREVIEW_OK", flush=True)
    if bpy.app.background:
        return
    def _quit():
        try:
            bpy.ops.wm.quit_blender()
        finally:
            os._exit(0)

    bpy.app.timers.register(_quit, first_interval=0.2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
