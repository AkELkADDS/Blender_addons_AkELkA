from __future__ import annotations

import importlib
import sys

bl_info = {
    "name": "GR2 LAB",
    "author": "AkELkA",
    "version": (2, 1, 585),
    "blender": (3, 6, 0),
    "location": "File > Import/Export · Object > GR2 Lab",
    "description": "Standalone GR2 import/export (decode, densify, encode) — no web server",
    "doc_url": "https://www.patreon.com/c/AkELkA",
    "category": "Import-Export",
}

# Flat names injected when codec/ is on sys.path — safe to drop/reload.
# Never pop gr2lab_blender.* package modules during register (Blender 5.x enable breaks).
_FLAT_CODEC_MODULES = (
    "paths",
    "bitknit",
    "gr2_format",
    "gr2_codec",
    "gr2_dump",
    "gr2lab",
    "granny_anim",
    "granny_quat",
    "granny_bspline",
    "bind_pose",
    "rig_catalog",
    "dump_history",
    "root_gate",
    "timing",
)


def _purge_flat_codec_modules() -> None:
    for k in _FLAT_CODEC_MODULES:
        sys.modules.pop(k, None)


def register():
    # Clear stale flat imports from a previous enable (same Blender session).
    _purge_flat_codec_modules()
    from . import codec  # noqa: F401 — puts codec/ on sys.path
    from . import local_pipeline
    from . import rig_display
    from .rig_display import head_ik
    from .rig_display import larian_ik
    from .rig_display import mesh_kits
    from .rig_display import tail_rig
    from . import scene_stage_empties
    from . import race_picker
    from . import anim_browser
    from . import bridge_blender

    importlib.reload(anim_browser)
    importlib.reload(race_picker)
    importlib.reload(scene_stage_empties)
    importlib.reload(head_ik)
    importlib.reload(larian_ik)
    importlib.reload(mesh_kits)
    importlib.reload(rig_display)
    importlib.reload(tail_rig)
    importlib.reload(local_pipeline)
    importlib.reload(bridge_blender)
    bridge_blender.register()
    scene_stage_empties.register_prebody_nla_handler()


def unregister():
    from . import bridge_blender
    from . import scene_stage_empties

    scene_stage_empties.unregister_live_cinematic_handlers()
    scene_stage_empties.unregister_prebody_nla_handler()
    bridge_blender.unregister()
    _purge_flat_codec_modules()
