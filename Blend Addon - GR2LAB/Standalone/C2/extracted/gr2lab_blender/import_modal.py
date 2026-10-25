"""Staged GR2 import for modal operator (yields UI between heavy steps)."""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import bpy

from .import_progress_ui import update as overlay_update

_DECODE_LOCK = threading.Lock()


class _DecodePulse:
    """Thread-safe percent. No bpy — the modal timer paints the viewport bar."""

    def __init__(self) -> None:
        self.value = 8
        self.label = "Decoding…"

    def step(self, value, label="", **_kwargs) -> None:
        try:
            self.value = int(value)
        except (TypeError, ValueError):
            self.value = 8
        if label:
            self.label = str(label)


def _start_decode_job(state: Dict[str, Any], key: str, gr2_path: str, body_base, fps=None) -> None:
    from .local_pipeline import import_gr2_pipeline_local

    pulse = _DecodePulse()
    box: Dict[str, Any] = {}
    path_s = str(gr2_path)
    base = body_base
    pack_fps = fps

    def run() -> None:
        try:
            with _DECODE_LOCK:
                dec, doc = import_gr2_pipeline_local(
                    path_s, body_base=base, progress=pulse
                )
            if key == "primary" and isinstance(doc, dict):
                try:
                    from .bridge_blender import FPS_DEFAULT, prepare_pose_pack

                    use = float(pack_fps or doc.get("densify_fps") or FPS_DEFAULT)
                    box["pose_pack"] = prepare_pose_pack(doc, use)
                except Exception:
                    traceback.print_exc()
                    box["pose_pack"] = None
            box["ok"] = (dec, doc)
        except Exception as exc:
            box["err"] = exc
            traceback.print_exc()

    thread = threading.Thread(target=run, name="gr2lab-decode", daemon=True)
    thread.start()
    state[key + "_thread"] = thread
    state[key + "_pulse"] = pulse
    state[key + "_box"] = box


def _decode_job_status(state: Dict[str, Any], key: str):
    thread = state.get(key + "_thread")
    if thread is None:
        return None
    pulse = state[key + "_pulse"]
    if thread.is_alive():
        return ("WAIT", int(getattr(pulse, "value", 8) or 8), str(getattr(pulse, "label", "") or "Decoding…"))
    box = state.get(key + "_box") or {}
    err = box.get("err")
    if err is not None:
        return ("ERR", err)
    return ("DONE", box.get("ok"))


def _want_scene_plan(op) -> bool:
    return bool(getattr(op, "second_character", False)) or bool(
        getattr(op, "scene_empties", False)
    )


def _start_scene_plan(state: Dict[str, Any], path: str) -> None:
    """Dialog lookup is file IO. It runs beside decode and the rig build."""
    if state.get("scene_plan_thread") is not None:
        return
    state["scene_plan_t0"] = time.perf_counter()

    def _run(box, gr2_path):
        try:
            from . import scene_stage_empties

            box["scene_plan_result"] = scene_stage_empties.plan_scene_empties(gr2_path)
        except Exception:
            box["scene_plan_error"] = traceback.format_exc()

    thread = threading.Thread(
        target=_run,
        args=(state, path),
        name="gr2lab-scene-plan",
        daemon=True,
    )
    state["scene_plan_thread"] = thread
    thread.start()


def _note_scene_plan_bg(state: Dict[str, Any], job) -> None:
    if state.get("scene_plan_bg_noted"):
        return
    thread = state.get("scene_plan_thread")
    if thread is None or thread.is_alive():
        return
    thread.join()
    state["scene_plan_bg_noted"] = True
    err = state.get("scene_plan_error")
    if err:
        print(err, flush=True)
        return
    job.add(
        "scene_plan_bg",
        time.perf_counter() - float(state.get("scene_plan_t0") or time.perf_counter()),
    )


def _kick_partner_decode(op, context, state, path, bridge, job) -> None:
    """Start the second GR2 decode as soon as the scene plan exists.

    The decode thread blocks on the same lock as the primary decode, then
    runs while the first rig is still being built.
    """
    if state.get("partner_decode_kicked") or state.get("partner_warn"):
        return
    apply = (state.get("apply") or "").strip()
    if not (bool(getattr(op, "second_character", False)) and apply == "NEW_RIG"):
        return
    thread = state.get("scene_plan_thread")
    if thread is None or thread.is_alive() or state.get("scene_plan_error"):
        return
    plan = state.get("scene_plan_result")
    if plan is None:
        return
    state["partner_decode_kicked"] = True
    _note_scene_plan_bg(state, job)
    try:
        from . import scene_stage_empties

        partner = scene_stage_empties.find_partner_gr2(path, primary_plan=plan)
        state["partner_path"] = str(partner) if partner else ""
        if partner is not None:
            state["partner_name"] = partner.name
    except Exception:
        traceback.print_exc()
        state["partner_path"] = ""
    if not state.get("partner_path"):
        state["partner_warn"] = True
        return
    try:
        bridge._apply_layout_from_prefs(context)
        state["partner_pbase"] = bridge._resolve_body_base_for_import(
            choice=str(getattr(op, "partner_body_base", "AUTO") or "AUTO"),
            bb_hint="",
            prefs_base="",
            source_gr2=str(state["partner_path"]),
        )
    except Exception:
        traceback.print_exc()
        state["partner_warn"] = True
        return
    state["decode_partner_t0"] = time.perf_counter()
    _start_decode_job(state, "partner", state["partner_path"], state.get("partner_pbase"))


_FINISH_STEP = 7


def stage_count(op) -> int:
    apply = (getattr(op, "import_apply", "NEW_RIG") or "NEW_RIG").strip()
    if bool(getattr(op, "second_character", False)) and apply == "NEW_RIG":
        return 8
    return 6


def advance(
    op,
    context,
    state: Dict[str, Any],
    *,
    progress,
    bridge,
) -> Tuple[bool, Optional[str]]:
    """Run one import stage. Returns (finished, error_message)."""
    from timing import StageTimer

    step = int(state.get("step") or 0)
    path = state["path"]
    prefs = state["prefs"]
    apply = state["apply"]
    target = state.get("target")
    job: StageTimer = state.setdefault("job_timer", StageTimer("import_job"))

    def _ui(pct: int, label: str, detail: str = "") -> None:
        if state.get("preload_quiet"):
            return
        if progress is not None:
            progress.step(int(pct), label, force_ui=True)
        overlay_update(
            float(pct) / 100.0,
            label,
            detail=detail or Path(path).name,
            stage=min(step + 1, stage_count(op)),
        )

    def _pump() -> None:
        if state.get("preload_quiet"):
            return
        bridge._redraw_windows(context)

    if step == 0:
        _ui(2, "Preparing import…", Path(path).name)
        pin = bridge._take_swap_body_pin(target)
        label = str(getattr(op, "body_choice", "") or "").strip()
        hint = str(getattr(op, "bb_hint", "") or "")
        if pin:
            label = str(pin.get("choice") or label)
            hint = f"Auto: {Path(pin['base']).name}"
            try:
                op.body_choice = label
                op.bb_hint = hint
                op.import_akelka = bool(pin.get("akelka"))
            except Exception:
                pass
        from_label = bridge._body_file_for_race_label(label)
        picked_user = label == "user"
        pinned_name = str((pin or {}).get("base") or "") or from_label
        if pinned_name:
            choice = Path(pinned_name).name
            explicit = True
        elif picked_user:
            choice = (op.body_base or "AUTO").strip()
            explicit = True
        else:
            choice = "AUTO"
            explicit = False
        state["body_choice"] = label if label and label != "user" else ""
        state["base"] = bridge._resolve_body_base_for_import(
            choice=choice,
            bb_hint=hint,
            prefs_base=prefs.body_base,
            source_gr2=path,
            explicit_choice=explicit,
        )
        print(
            f"[GR2 LAB] import body_base resolve: choice={op.body_base!r} "
            f"hint={getattr(op, 'bb_hint', '')!r} -> {state['base']!r}"
        )
        state["step"] = 1
        _pump()
        return False, None

    if step == 1:
        status = _decode_job_status(state, "primary")
        if status is None:
            _ui(12, "Decoding primary GR2…", Path(path).name)
            try:
                bridge._apply_layout_from_prefs(context)
            except Exception:
                traceback.print_exc()
            state["decode_t0"] = time.perf_counter()
            _start_decode_job(state, "primary", path, state.get("base"), fps=op.fps)
            if _want_scene_plan(op):
                _start_scene_plan(state, path)
            _pump()
            return False, None
        if status[0] == "WAIT":
            _ui(int(status[1]), str(status[2]), Path(path).name)
            _kick_partner_decode(op, context, state, path, bridge, job)
            return False, None
        if status[0] == "ERR":
            err = status[1]
            if isinstance(err, bridge.Gr2LabApiError):
                return True, str(err)
            traceback.print_exc()
            return True, str(err)
        pair = status[1]
        if not pair:
            return True, "Decode returned nothing"
        dec, doc = pair
        state["dec"] = dec
        state["doc"] = doc
        state["pose_pack"] = (state.get("primary_box") or {}).get("pose_pack")
        state["timer"] = dec.get("timer") if isinstance(dec, dict) else None
        codec_t = state["timer"]
        if codec_t is not None:
            job.merge(codec_t, prefix="primary_decode_")
        job.add("decode_primary", time.perf_counter() - float(state.get("decode_t0") or time.perf_counter()))
        state["step"] = 2
        _pump()
        return False, None

    if step == 2:
        dec = state["dec"]
        doc = state["doc"]
        timer = state.get("timer")
        parts = bridge._rig_parts_from_op(op)
        want_display = bridge._apply_creates_rig(apply) and any(parts.values())
        gen = state.get("build_gen")
        if gen is None:
            _ui(38, "Building rig…", Path(path).name)
            state["build_t0"] = time.perf_counter()
            if _want_scene_plan(op):
                _start_scene_plan(state, path)
            _kick_partner_decode(op, context, state, path, bridge, job)
            hold = None
            hold_name = state.get("strip_hold_collection")
            if hold_name:
                hold = bpy.data.collections.get(str(hold_name))
            gen = bridge._import_gr2lab_steps(
                context,
                doc,
                fps=op.fps,
                y_up_display=bool(prefs.y_up_display),
                template_path=None,
                progress=progress,
                template_doc=dec.get("sparse_doc") if isinstance(dec, dict) else None,
                timer=timer,
                rig_display=want_display,
                rig_parts=parts,
                target_arm=target if bridge._apply_needs_target(apply) else None,
                anim_apply=apply,
                import_meshes=getattr(op, "import_meshes", "NONE"),
                import_akelka=bool(getattr(op, "import_akelka", False)),
                import_tail=bool(getattr(op, "import_tail", False)),
                import_clear_meshes=bool(getattr(op, "import_clear_meshes", False)),
                mesh_base=state["base"],
                hide_viewport_build=bool(state.get("strip_hide_build")),
                pose_pack=state.get("pose_pack"),
                link_collection=hold,
            )
            state["build_gen"] = gen
        try:
            item = next(gen)
        except StopIteration as stop:
            state["arm"] = stop.value
            try:
                state["arm_name"] = str(stop.value.name)
            except Exception:
                pass
            state.pop("build_gen", None)
            if timer is not None:
                job.merge(timer, prefix="primary_build_")
            job.add(
                "build_primary",
                time.perf_counter() - float(state.get("build_t0") or time.perf_counter()),
            )
            _kick_partner_decode(op, context, state, path, bridge, job)
        except Exception as e:
            traceback.print_exc()
            return True, str(e)
        else:
            if item:
                _ui(int(item[0]), str(item[1]), Path(path).name)
            _kick_partner_decode(op, context, state, path, bridge, job)
            return False, None
        state["step"] = 3
        _pump()
        return False, None

    if step == 3:
        _ui(55, "Stamping import metadata…", "")
        bridge._import_finish_primary_reports(op, context, state)
        state["step"] = 4
        _pump()
        return False, None

    if step == 4:
        whole_scene = bridge._whole_scene(op)
        if whole_scene:
            try:
                op.scene_empties = True
            except Exception:
                pass
        transient_place = False
        use_scene_place = whole_scene or (
            bridge._scene_place_allowed(op) and bool(getattr(op, "scene_empties", False))
        )
        try:
            from . import anim_browser

            if anim_browser.pending_clip_drop_blocks_scene_place(path, context):
                use_scene_place = False
        except Exception:
            pass
        state["use_scene_place"] = use_scene_place
        state["transient_place"] = transient_place
        if not use_scene_place and not (
            bool(getattr(op, "second_character", False)) and apply == "NEW_RIG"
        ):
            state["step"] = _FINISH_STEP
            _pump()
            return False, None
        _ui(60, "Scene plan…", Path(path).name)
        plan = None
        try:
            from . import scene_stage_empties

            th = state.get("scene_plan_thread")
            if th is not None:
                th.join()
            _note_scene_plan_bg(state, job)
            with job.span("scene_plan"):
                plan = scene_stage_empties.plan_scene_empties(path, timer=job)
            state["plan"] = plan
            if plan is not None:
                state["seat_primary"] = scene_stage_empties.seat_primary_snapshot(
                    plan
                )
            else:
                snap = scene_stage_empties.seat_snapshot_for_gr2(path, timer=job)
                if snap is not None:
                    state["seat_primary"] = snap
            if plan is not None:
                for item in plan.empties:
                    empty = context.blend_data.objects.get(item.name)
                    if empty is None:
                        continue
                    old = empty.get("gr2lab_scene_stage")
                    if old and old != plan.stage.name:
                        state["stale_stage"] = str(old)
                        break
        except Exception:
            traceback.print_exc()
        if use_scene_place:
            _ui(65, "Scene empties / placement…", Path(path).name)
            try:
                from . import scene_stage_empties

                defer_cinematic = bool(op.second_character) and apply == "NEW_RIG"
                state["cinematic_deferred"] = defer_cinematic
                with job.span("scene_place"):
                    note = scene_stage_empties.place_for_import(
                        context,
                        path,
                        state.get("arm"),
                        y_up_display=bool(prefs.y_up_display),
                        parent_rig=bridge._apply_creates_rig(apply),
                        center_parent=bool(op.center_parent),
                        plan=plan,
                        defer_cinematic=defer_cinematic,
                    )
                state["place_note"] = note
            except Exception:
                traceback.print_exc()
                state["place_note"] = ""
        if bool(op.second_character) and apply == "NEW_RIG":
            state["step"] = 5
        else:
            state["step"] = _FINISH_STEP
        _pump()
        return False, None

    if step == 5:
        partner_name = state.get("partner_name") or ""
        status = _decode_job_status(state, "partner")
        if status is None and state.get("partner_warn"):
            op.report({"WARNING"}, "No second character GR2 next to this file")
            state["step"] = _FINISH_STEP
            _pump()
            return False, None
        if status is None:
            _ui(72, "Second character — decode…", partner_name)
            try:
                with job.span("partner_resolve"):
                    from . import scene_stage_empties

                    partner = scene_stage_empties.find_partner_gr2(
                        path, primary_plan=state.get("plan"), timer=job
                    )
                    state["partner_path"] = str(partner) if partner else ""
                    if partner is not None:
                        state["partner_name"] = partner.name
                        partner_name = partner.name
            except Exception:
                traceback.print_exc()
                state["partner_path"] = ""
            if not state.get("partner_path"):
                state["partner_warn"] = True
                op.report({"WARNING"}, "No second character GR2 next to this file")
                state["step"] = _FINISH_STEP
                _pump()
                return False, None
            try:
                bridge._apply_layout_from_prefs(context)
                state["partner_pbase"] = bridge._resolve_body_base_for_import(
                    choice=str(getattr(op, "partner_body_base", "AUTO") or "AUTO"),
                    bb_hint="",
                    prefs_base="",
                    source_gr2=str(state["partner_path"]),
                )
            except Exception:
                traceback.print_exc()
                state["partner_warn"] = True
                state["step"] = _FINISH_STEP
                _pump()
                return False, None
            state["decode_partner_t0"] = time.perf_counter()
            _start_decode_job(
                state, "partner", state["partner_path"], state.get("partner_pbase")
            )
            return False, None
        if status[0] == "WAIT":
            _ui(max(72, int(status[1])), str(status[2]), partner_name)
            return False, None
        if status[0] == "ERR":
            traceback.print_exc()
            state["partner_warn"] = True
            state["step"] = _FINISH_STEP
            _pump()
            return False, None
        dec, doc = status[1]
        state["partner_dec"] = dec
        state["partner_doc"] = doc
        job.add(
            "partner_decode",
            time.perf_counter() - float(state.get("decode_partner_t0") or time.perf_counter()),
        )
        state["step"] = 6
        _pump()
        return False, None

    if step == 6:
        partner_name = state.get("partner_name") or ""
        try:
            op._import_second_character(
                context,
                path,
                prefs,
                parts=bridge._partner_rig_parts(op),
                seat_on_empty=bool(state.get("use_scene_place")),
                progress=progress,
                partner_path=state.get("partner_path") or "",
                primary_plan=state.get("plan"),
                import_state=state,
                phase="build",
                job_timer=job,
            )
        except Exception:
            traceback.print_exc()
            state["partner_warn"] = True
            state["step"] = _FINISH_STEP
            _pump()
            return False, None
        if state.get("partner_build_pending"):
            item = state.get("partner_slice") or (84, "Second character — rig build…")
            _ui(int(item[0]), str(item[1]), partner_name)
            return False, None
        state["step"] = _FINISH_STEP
        _pump()
        return False, None

    if step == _FINISH_STEP:
        phase = int(state.get("finish_phase") or 0)
        if phase == 0:
            _ui(95, "Finishing…", "cleanup")
            if state.get("transient_place"):
                try:
                    from . import scene_stage_empties

                    n = scene_stage_empties.remove_transient_scene_empties(
                        context, path
                    )
                    state["removed_empties"] = n
                except Exception:
                    traceback.print_exc()
            state["finish_phase"] = 1
            _pump()
            return False, None
        if phase == 1:
            _ui(97, "Finishing…", "timeline")
            if state.get("cinematic_deferred") and state.get("partner_warn"):
                try:
                    from . import scene_stage_empties

                    plan = state.get("plan")
                    col = None
                    if plan is not None:
                        col = context.blend_data.collections.get(plan.stage.name)
                    if plan is not None and col is not None:
                        sync_note = scene_stage_empties.apply_cinematic_timeline_sync(
                            context,
                            col,
                            path,
                            plan,
                            y_up_display=bool(prefs.y_up_display),
                            timeline_gr2_path=path,
                        )
                        if sync_note:
                            prev = state.get("place_note") or ""
                            state["place_note"] = f"{prev} · {sync_note}".strip(" ·")
                            print(f"[GR2 LAB] {sync_note}", flush=True)
                except Exception:
                    traceback.print_exc()
            state["finish_phase"] = 2
            _pump()
            return False, None
        if phase == 2:
            _ui(98, "Finishing…", "place")
            try:
                from . import anim_browser

                arm = state.get("arm")
                if (
                    arm is not None
                    and state.get("apply") == "NEW_RIG"
                    and not state.get("defer_clip_snap")
                ):
                    anim_browser.apply_pending_clip_drop(
                        context, path, arm, consume=True
                    )
            except Exception:
                traceback.print_exc()
            state["finish_phase"] = 3
            _pump()
            return False, None
        if phase == 3:
            want_edit = False
            try:
                from . import anim_browser

                want_edit = bool(
                    anim_browser.edit_copy_enabled() and state.get("arm") is not None
                )
            except Exception:
                want_edit = False
            if want_edit:
                state["defer_edit_copy"] = True
            state["finish_phase"] = 5
            _pump()
            return False, None
        job.print_block()
        note = state.get("place_note")
        if note:
            if state.get("transient_place"):
                note = note.replace("Scene empties:", "Scene placement:")
            op.report({"INFO"}, note)
        n = state.get("removed_empties")
        if n:
            op.report(
                {"INFO"},
                f"Removed {n} temporary empties (rigs kept at stage positions)",
            )
        if state.get("partner_warn"):
            op.report({"WARNING"}, "Second character was not imported")
        _ui(100, "Done", "")
        return True, None

    return True, None
