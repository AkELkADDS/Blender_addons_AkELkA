/* Free 3D skeleton viewport — no page scroll steal, unlimited orbit */
(function () {
  const data = window.SKELETON_DATA;
  if (!data) return;

  const CUT_T = data.cutT;
  const ref = data.ref;
  const test = data.test;
  const PLAY_FPS = data.fps || data.ref.fps || 30;
  const bodySet = new Set(data.bodyBones || []);
  const ikSet = new Set(data.ikBones || []);

  const cvRef = document.getElementById("skel-ref");
  const cvTest = document.getElementById("skel-test");
  const slider = document.getElementById("skel-slider");
  const timeLabel = document.getElementById("skel-time");
  const fpsBadge = document.getElementById("skel-fps");
  const status = document.getElementById("skel-status");
  const wrap = document.querySelector(".viewer-wrap");

  let playing = false;
  let frameIdx = 0;
  let playTime = 0; // continuous seconds for smooth playback
  let lastNow = 0;
  let displayFps = 0;
  let fpsFrames = 0;
  let fpsLast = 0;

  // Spherical camera around target — no hard lock
  let yaw = Math.PI * 0.25;
  let pitch = 0.35;
  let distance = 3.5;
  let target = [0, 0.95, 0];

  let dragging = false;
  let dragMode = "orbit";
  let lastX = 0;
  let lastY = 0;
  let activeCanvas = null;
  let pointerId = null;

  slider.max = ref.times.length - 1;

  function boneIdx(bones, name) {
    return bones.indexOf(name);
  }

  function isBodyBone(name) {
    if (bodySet.size) return bodySet.has(name);
    return !/Dummy_|FX|Twist|endBone|Sheath|Tentacle/.test(name);
  }

  function isIkBone(name) {
    if (ikSet.size) return ikSet.has(name);
    return /_IK$/.test(name);
  }

  function frameBounds(skel, frame) {
    const positions = skel.positions[frame];
    let min = [Infinity, Infinity, Infinity];
    let max = [-Infinity, -Infinity, -Infinity];
    let count = 0;
    for (let i = 0; i < skel.bones.length; i++) {
      const name = skel.bones[i];
      if (!isBodyBone(name) && !isIkBone(name)) continue;
      const p = positions[i];
      if (!p) continue;
      for (let a = 0; a < 3; a++) {
        min[a] = Math.min(min[a], p[a]);
        max[a] = Math.max(max[a], p[a]);
      }
      count++;
    }
    if (!count) return { center: [0, 0.95, 0], size: 2 };
    return {
      center: [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
      ],
      size: Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 0.5),
    };
  }

  function fitCamera(frame) {
    const b1 = frameBounds(ref, frame);
    const b2 = frameBounds(test, frame);
    target = [
      (b1.center[0] + b2.center[0]) / 2,
      (b1.center[1] + b2.center[1]) / 2,
      (b1.center[2] + b2.center[2]) / 2,
    ];
    const size = Math.max(b1.size, b2.size);
    distance = Math.max(2.0, size * 2.6);
    yaw = Math.PI * 0.25;
    pitch = 0.35;
  }

  function clampPitch(p) {
    // Almost full sphere — leave tiny epsilon so basis never flips
    const lim = Math.PI / 2 - 0.02;
    return Math.max(-lim, Math.min(lim, p));
  }

  function cameraPos() {
    const p = clampPitch(pitch);
    const cp = Math.cos(p);
    const sp = Math.sin(p);
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    return [
      target[0] + distance * cp * sy,
      target[1] + distance * sp,
      target[2] + distance * cp * cy,
    ];
  }

  function cameraBasis(cam) {
    // Look direction: from camera toward target
    let fx = target[0] - cam[0];
    let fy = target[1] - cam[1];
    let fz = target[2] - cam[2];
    let fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl; fy /= fl; fz /= fl;

    // Right = normalize(cross(look, worldUp))? 
    // Standard orbit: right = normalize(cross(worldUp, -look)) when look is from cam to target
    // = normalize(cross(look, worldUp))? Let's use:
    // right = normalize(cross(worldUp, -forward)) where forward = look
    const worldUp = [0, 1, 0];
    let rx = worldUp[1] * (-fz) - worldUp[2] * (-fy);
    let ry = worldUp[2] * (-fx) - worldUp[0] * (-fz);
    let rz = worldUp[0] * (-fy) - worldUp[1] * (-fx);
    let rl = Math.hypot(rx, ry, rz);
    if (rl < 1e-6) {
      // Looking nearly straight up/down — use X axis fallback
      rx = 1; ry = 0; rz = 0;
      rl = 1;
    }
    rx /= rl; ry /= rl; rz /= rl;

    // Up = cross(right, look)? For RH with look=forward: up = cross(right, forward) gives wrong sign
    // up = cross(look, right) wait:
    // If look=(0,0,-1), right=(1,0,0): cross(look,right)=(0,1,0)? 
    // cross((0,0,-1),(1,0,0)) = (0*(-0)-(-1)*0, (-1)*1-0*0, 0*0-0*1) = (0,-1,0)
    // up = cross(right, look): cross((1,0,0),(0,0,-1)) = (0* -1 - 0*0, 0*0 - 1*(-1), 1*0 - 0*0) = (0,1,0) YES
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    return {
      f: [fx, fy, fz],
      r: [rx, ry, rz],
      u: [ux, uy, uz],
    };
  }

  function project(p, cam, basis, w, h) {
    const dx = p[0] - cam[0];
    const dy = p[1] - cam[1];
    const dz = p[2] - cam[2];
    const { f, r, u } = basis;

    const localX = dx * r[0] + dy * r[1] + dz * r[2];
    const localY = dx * u[0] + dy * u[1] + dz * u[2];
    const localZ = dx * f[0] + dy * f[1] + dz * f[2];

    if (localZ <= 0.01) return null;
    const fov = 50 * Math.PI / 180;
    const scale = (h * 0.5) / Math.tan(fov / 2) / localZ;
    return {
      x: w * 0.5 + localX * scale,
      y: h * 0.5 - localY * scale,
      z: localZ,
      scale,
    };
  }

  function drawBoneShape(ctx, a, b, color) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const mx = a.x + dx * 0.22;
    const my = a.y + dy * 0.22;
    const thickness = Math.max(4, Math.min(14, (a.scale + b.scale) * 0.014));

    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(mx + nx * thickness, my + ny * thickness);
    ctx.lineTo(b.x, b.y);
    ctx.lineTo(mx - nx * thickness, my - ny * thickness);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.95;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawGrid(ctx, cam, basis, w, h) {
    const step = 0.5;
    const extent = 4.0;
    ctx.lineWidth = 1;
    for (let i = -extent; i <= extent + 0.001; i += step) {
      ctx.strokeStyle = Math.abs(i) < 0.001 ? "#334155" : "#15202b";
      const a1 = project([i, 0, -extent], cam, basis, w, h);
      const b1 = project([i, 0, extent], cam, basis, w, h);
      const a2 = project([-extent, 0, i], cam, basis, w, h);
      const b2 = project([extent, 0, i], cam, basis, w, h);
      if (a1 && b1) {
        ctx.beginPath();
        ctx.moveTo(a1.x, a1.y);
        ctx.lineTo(b1.x, b1.y);
        ctx.stroke();
      }
      if (a2 && b2) {
        ctx.beginPath();
        ctx.moveTo(a2.x, a2.y);
        ctx.lineTo(b2.x, b2.y);
        ctx.stroke();
      }
    }
  }

  function lerpPose(skel, t) {
    const times = skel.times;
    if (!times.length) return skel.positions[0];
    if (t <= times[0]) return skel.positions[0];
    if (t >= times[times.length - 1]) return skel.positions[times.length - 1];

    let lo = 0;
    let hi = times.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t) lo = mid;
      else hi = mid;
    }
    const t0 = times[lo];
    const t1 = times[hi];
    const a = skel.positions[lo];
    const b = skel.positions[hi];
    const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    if (u <= 0) return a;
    if (u >= 1) return b;
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const pa = a[i];
      const pb = b[i];
      if (!pa || !pb) {
        out[i] = pa || pb || [0, 0, 0];
        continue;
      }
      out[i] = [
        pa[0] + (pb[0] - pa[0]) * u,
        pa[1] + (pb[1] - pa[1]) * u,
        pa[2] + (pb[2] - pa[2]) * u,
      ];
    }
    return out;
  }

  function drawSkeleton(ctx, w, h, skel, positions, label, color) {
    ctx.fillStyle = "#07090e";
    ctx.fillRect(0, 0, w, h);

    const cam = cameraPos();
    const basis = cameraBasis(cam);

    drawGrid(ctx, cam, basis, w, h);

    const projPts = new Array(skel.bones.length);
    for (let i = 0; i < skel.bones.length; i++) {
      const p = positions[i];
      if (!p) continue;
      projPts[i] = project(p, cam, basis, w, h);
    }

    const boneDraws = [];
    for (const [pa, ch] of skel.edges) {
      if (!isBodyBone(pa) || !isBodyBone(ch)) continue;
      const ip = boneIdx(skel.bones, pa);
      const ic = boneIdx(skel.bones, ch);
      if (ip < 0 || ic < 0) continue;
      const a = projPts[ip];
      const b = projPts[ic];
      if (!a || !b) continue;
      boneDraws.push({ a, b, z: (a.z + b.z) * 0.5 });
    }
    boneDraws.sort((u, v) => v.z - u.z);
    for (const bd of boneDraws) drawBoneShape(ctx, bd.a, bd.b, color);

    for (let i = 0; i < skel.bones.length; i++) {
      const name = skel.bones[i];
      const p = projPts[i];
      if (!p) continue;
      if (isIkBone(name)) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = "#fbbf24";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "#fde68a";
        ctx.font = "10px Segoe UI, sans-serif";
        ctx.fillText(name.replace("Dummy_", ""), p.x + 8, p.y - 6);
        continue;
      }
      if (!isBodyBone(name)) continue;
      const r = /Head|Chest|Root_M|Dummy_Root/.test(name) ? 5.5 : 3.8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    ctx.fillStyle = color;
    ctx.font = "bold 14px Segoe UI, sans-serif";
    ctx.fillText(label, 12, 24);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px Consolas, monospace";
    ctx.fillText("LMB orbit · RMB/Shift pan · wheel zoom · " + PLAY_FPS + " FPS bake", 12, h - 12);
  }

  function renderAtTime(t) {
    const duration = ref.times[ref.times.length - 1] || data.duration || 1;
    playTime = ((t % duration) + duration) % duration;

    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < ref.times.length; i++) {
      const d = Math.abs(ref.times[i] - playTime);
      if (d < bestD) { bestD = d; best = i; }
    }
    frameIdx = best;
    slider.value = frameIdx;
    timeLabel.textContent = playTime.toFixed(2) + "s";
    if (fpsBadge) fpsBadge.textContent = PLAY_FPS + " FPS · live " + displayFps;

    const refPose = lerpPose(ref, playTime);
    const testPose = lerpPose(test, playTime);

    for (const [cv, skel, pose, label, color] of [
      [cvRef, ref, refPose, "REFERENCE", "#4ade80"],
      [cvTest, test, testPose, "BLENDER EXPORT", "#f87171"],
    ]) {
      const panel = cv.parentElement;
      const w = Math.max(320, panel.clientWidth || 480);
      const h = 360;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.floor(w * dpr);
      cv.height = Math.floor(h * dpr);
      cv.style.width = w + "px";
      cv.style.height = h + "px";
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawSkeleton(ctx, w, h, skel, pose, label, color);
    }
  }

  function renderFrame(idx) {
    frameIdx = Math.max(0, Math.min(ref.times.length - 1, idx | 0));
    playTime = ref.times[frameIdx];
    renderAtTime(playTime);
  }

  function panByPixels(dx, dy, canvasH) {
    const cam = cameraPos();
    const basis = cameraBasis(cam);
    const fov = 50 * Math.PI / 180;
    const worldPerPixel = (2 * Math.tan(fov / 2) * distance) / Math.max(canvasH, 1);
    // Move target opposite to mouse so scene follows cursor
    target[0] -= (basis.r[0] * dx - basis.u[0] * dy) * worldPerPixel;
    target[1] -= (basis.r[1] * dx - basis.u[1] * dy) * worldPerPixel;
    target[2] -= (basis.r[2] * dx - basis.u[2] * dy) * worldPerPixel;
  }

  function setPageScrollLock(on) {
    if (on) {
      document.documentElement.style.overflow = "hidden";
      document.body.style.overflow = "hidden";
      document.body.style.overscrollBehavior = "none";
      if (wrap) wrap.classList.add("viewport-active");
    } else {
      document.documentElement.style.overflow = "";
      document.body.style.overflow = "";
      document.body.style.overscrollBehavior = "";
      if (wrap) wrap.classList.remove("viewport-active");
    }
  }

  function bindViewport(cv) {
    // Stop browser from scrolling / selecting while using the viewport
    cv.style.touchAction = "none";
    cv.style.userSelect = "none";
    cv.style.webkitUserSelect = "none";

    cv.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
    });

    cv.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragging = true;
      activeCanvas = cv;
      pointerId = e.pointerId;
      lastX = e.clientX;
      lastY = e.clientY;
      dragMode = (e.button === 2 || e.button === 1 || e.shiftKey) ? "pan" : "orbit";
      setPageScrollLock(true);
      try { cv.setPointerCapture(e.pointerId); } catch (_) {}
    });

    cv.addEventListener("pointermove", (e) => {
      if (!dragging || activeCanvas !== cv) return;
      e.preventDefault();
      e.stopPropagation();
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;

      if (dragMode === "pan") {
        panByPixels(dx, dy, cv.clientHeight || 500);
      } else {
        yaw += dx * 0.01;
        pitch = clampPitch(pitch + dy * 0.01);
      }
      renderFrame(frameIdx);
    });

    function endDrag(e) {
      if (pointerId !== null && e && e.pointerId !== pointerId && e.type !== "pointercancel") return;
      dragging = false;
      activeCanvas = null;
      pointerId = null;
      setPageScrollLock(false);
    }

    cv.addEventListener("pointerup", endDrag);
    cv.addEventListener("pointercancel", endDrag);
    cv.addEventListener("lostpointercapture", endDrag);

    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      e.stopPropagation();
      // Smooth zoom — wide range, no fighting
      const factor = Math.exp(e.deltaY * 0.0015);
      distance = Math.max(0.15, Math.min(80, distance * factor));
      renderFrame(frameIdx);
    }, { passive: false });

    // Also block touch scroll on the panel
    cv.addEventListener("touchstart", (e) => {
      e.preventDefault();
    }, { passive: false });
    cv.addEventListener("touchmove", (e) => {
      e.preventDefault();
    }, { passive: false });
  }

  bindViewport(cvRef);
  bindViewport(cvTest);

  // Lock scroll while pointer is over viewer (wheel only)
  if (wrap) {
    wrap.addEventListener("wheel", (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, { passive: false });
  }

  document.getElementById("skel-play").onclick = () => {
    playing = true;
    lastNow = 0;
  };
  document.getElementById("skel-pause").onclick = () => { playing = false; };
  document.getElementById("skel-cut").onclick = () => {
    playing = false;
    playTime = CUT_T;
    renderAtTime(playTime);
    fitCamera(frameIdx);
    renderAtTime(playTime);
  };

  const resetBtn = document.getElementById("skel-reset");
  if (resetBtn) {
    resetBtn.onclick = () => {
      fitCamera(frameIdx);
      renderAtTime(playTime);
    };
  }
  const fitBtn = document.getElementById("skel-fit");
  if (fitBtn) {
    fitBtn.onclick = () => {
      const b1 = frameBounds(ref, frameIdx);
      const b2 = frameBounds(test, frameIdx);
      target = [
        (b1.center[0] + b2.center[0]) / 2,
        (b1.center[1] + b2.center[1]) / 2,
        (b1.center[2] + b2.center[2]) / 2,
      ];
      distance = Math.max(2.0, Math.max(b1.size, b2.size) * 2.6);
      renderAtTime(playTime);
    };
  }

  slider.oninput = () => {
    playing = false;
    renderFrame(parseInt(slider.value, 10));
  };

  function tick(now) {
    requestAnimationFrame(tick);
    fpsFrames++;
    if (!fpsLast) fpsLast = now;
    if (now - fpsLast >= 500) {
      displayFps = Math.round((fpsFrames * 1000) / (now - fpsLast));
      fpsFrames = 0;
      fpsLast = now;
      if (fpsBadge) fpsBadge.textContent = PLAY_FPS + " FPS · live " + displayFps;
    }

    if (playing) {
      if (!lastNow) lastNow = now;
      const dt = Math.min(0.05, (now - lastNow) / 1000);
      lastNow = now;
      playTime += dt;
      renderAtTime(playTime);
    }
  }

  window.addEventListener("resize", () => renderAtTime(playTime));

  fitCamera(0);
  status.textContent =
    "Preview baked at " + PLAY_FPS + " FPS (was 10). This is a stick-figure debug view — Blender will always look smoother. " +
    (data.bodyBones || []).length + " body bones.";
  if (fpsBadge) fpsBadge.textContent = PLAY_FPS + " FPS";
  renderAtTime(0);
  requestAnimationFrame(tick);
})();
