/* RVR showroom, reconstructed from the client video (assets/RVR1.mp4).

   What is drawn: for every keyframe of the video there is a 3D surface (a depth mesh) that was
   reconstructed from the footage, placed where the camera really was when that frame was filmed
   (camera path recovered by structure-from-motion). Each surface is textured with its original
   video frame. Scrolling moves a real 3D camera along the recovered camera path; the two nearest
   keyframe surfaces are rendered from that camera and blended, so you see true perspective and
   parallax. The mouse moves the camera a little off the path to show the depth.

   Version 2:
   - The REVERE shutter is rebuilt from the footage frame by frame: the camera stands still while it
     rolls up, the shutter moves as one rigid sheet, and its edge position was measured in every
     video frame (24 per second). The shutter is drawn moving with that measured edge, so the
     roll-up is continuous at any scroll speed (no cross-fading between shutter positions).
   - The video's own hard cuts are crossed with a short timed cross-fade (it never stays half-way
     because the scroll stopped there).
   - Keyframes are blended with a check: where the two disagree (misaligned detail), the nearer one
     wins instead of showing both (no double images).
   - Scrolling drives the camera through a critically damped spring (smooth start and stop).
   - Two neighbouring keyframes are only cross-faded for long when their viewpoints are nearly the
     same; when the scroll stops, the picture settles on one keyframe (never two images at rest).
   - At the video's cuts the camera holds still during the short fade.
   - The camera never runs ahead of the downloads: it waits (with the "Loading" notice) where the
     data ends. A long jump goes straight to its destination with the cut fade once that has loaded.
     Prefetch: from the camera to the scroll target, plus 12 s of the tour beyond it.

   If 3D can't run, the page switches to the original scroll-video (window.RVR.useVideo). */
import * as THREE from 'three';

const RVR = window.RVR || { mode: '3d', useVideo() {} };
const params = new URLSearchParams(location.search);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = matchMedia('(pointer: fine)').matches;
const $ = (id) => document.getElementById(id);
const BASE = 'assets/scene/';
const CUT_FADE = reduceMotion ? 0 : 0.45; // seconds of real time for crossing one of the video's cuts
const FAR_JUMP = 8; // a scroll target further away than this (seconds of the tour) is a long jump

/* ------------------------------------------------------------------ quality tiers */

const TIERS = {
  high: { maxDpr: 1.5, images: 'hd', gpuCache: 12 },
  medium: { maxDpr: 1.25, images: 'hd', gpuCache: 10 },
  low: { maxDpr: 1.0, images: 'sd', gpuCache: 8 },
};
const ORDER = ['low', 'medium', 'high'];

function detectTier() {
  if (TIERS[params.get('quality')]) return params.get('quality');
  let gpu = '';
  try {
    const gl = document.createElement('canvas').getContext('webgl2');
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    gpu = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)).toLowerCase();
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  } catch (e) { /* ignore */ }
  const mobile = matchMedia('(pointer: coarse)').matches || /android|iphone|ipad|mobile/i.test(navigator.userAgent);
  const memory = navigator.deviceMemory || 4;
  const software = /swiftshader|llvmpipe|software|basic render/.test(gpu);
  if (software) return 'low';
  if (mobile) return memory >= 4 ? 'medium' : 'low';
  return memory <= 2 ? 'medium' : 'high';
}

/* ------------------------------------------------------------------ helpers */

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

function setStatus(text, fraction) {
  const status = $('loader-status');
  if (status && text) status.textContent = text;
  if (fraction != null) $('loader-bar').style.width = `${Math.round(fraction * 100)}%`;
}

async function fetchOk(url, type) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return type === 'json' ? res.json() : type === 'blob' ? res.blob() : res.arrayBuffer();
}

/** Catmull-Rom weights for parameter f in [0, 1]. */
function crWeights(f) {
  const f2 = f * f;
  const f3 = f2 * f;
  return [-0.5 * f3 + f2 - 0.5 * f, 1.5 * f3 - 2.5 * f2 + 1, -1.5 * f3 + 2 * f2 + 0.5 * f, 0.5 * f3 - 0.5 * f2];
}

/* ------------------------------------------------------------------ the reconstruction */

class Reconstruction {
  constructor(manifest) {
    this.m = manifest;
    const [gw, gh] = manifest.grid;
    this.gw = gw;
    this.gh = gh;
    this.kfs = manifest.keyframes.map((k, i) => ({ ...k, index: i, blob: null, depthData: null, geo: null, used: 0 }));
    this.shots = manifest.shots.map((s, i) => {
      const path = manifest.path.filter((p) => p.shot === i).map((p) => ({
        t: p.t, p: new THREE.Vector3(...p.p), q: new THREE.Quaternion(...p.q),
      }));
      // keep neighbouring quaternions in the same hemisphere so interpolating them never flips
      for (let k = 1; k < path.length; k++) if (path[k].q.dot(path[k - 1].q) < 0) path[k].q.set(-path[k].q.x, -path[k].q.y, -path[k].q.z, -path[k].q.w);
      return { ...s, index: i, kfs: this.kfs.filter((k) => k.shot === i), path };
    });
    // How long two neighbouring keyframes may be cross-faded depends on how different their
    // viewpoints are: the camera baseline relative to the keyframe's parallax budget (1 = near and far
    // things shift against each other by ~3% of the picture). Nearly identical views blend over the
    // middle half of the interval; clearly different views only switch over a short moment, so two
    // differently placed images are never shown together for long (no "CREATE CREATE").
    for (const sh of this.shots) {
      const list = sh.kfs.filter((k) => k.depth);
      for (let k = 0; k + 1 < list.length; k++) {
        const a = list[k];
        const b = list[k + 1];
        const base = Math.hypot(a.C[0] - b.C[0], a.C[1] - b.C[1], a.C[2] - b.C[2]);
        const r = base / Math.max(1e-6, Math.min(a.par || Infinity, b.par || Infinity));
        a.blendHalf = 0.28 - 0.24 * smoothstep(0.15, 0.6, r);
      }
    }
    // shutter roll-up (camera standing still): measured edge row for every video frame
    this.shutter = manifest.shutter || null;
    if (this.shutter) {
      this.shutter.closedKf = this.kfs[this.shutter.closed];
      this.shutter.openKf = this.kfs[this.shutter.open];
      this.shutter.t1 = this.shutter.closedKf ? this.shutter.openKf.t : 0;
    }
    // shared triangle list for every keyframe grid
    const idx = new Uint16Array((gw - 1) * (gh - 1) * 6);
    let o = 0;
    for (let y = 0; y < gh - 1; y++) {
      for (let x = 0; x < gw - 1; x++) {
        const a = y * gw + x;
        const b = a + 1;
        const c = a + gw;
        const d = c + 1;
        idx.set([a, c, b, b, c, d], o);
        o += 6;
      }
    }
    this.index = new THREE.BufferAttribute(idx, 1);
    const uv = new Float32Array(gw * gh * 2);
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
      uv[(y * gw + x) * 2] = x / (gw - 1);
      uv[(y * gw + x) * 2 + 1] = y / (gh - 1); // textures are uploaded top row first (flipY = false)
    }
    this.uv = new THREE.BufferAttribute(uv, 2);
    this.buildTimeline();
  }

  /* Scroll timeline: every shot plays at video speed, back to back. The video's cuts sit between
     shots; crossing one starts a short timed cross-fade (see draw()). */
  buildTimeline() {
    const segs = [];
    let x = 0;
    this.shots.forEach((s, i) => {
      const len = s.end - s.start;
      segs.push({ type: 'shot', shot: i, x0: x, x1: x + len });
      x += len;
    });
    this.segments = segs;
    this.length = x;
  }

  /** Position on the timeline (0..length) of a video time. */
  xOfTime(t) {
    const last = this.segments.length - 1;
    for (const [i, s] of this.segments.entries()) {
      const sh = this.shots[s.shot];
      // a shot's last frame sits just before the cut (the cut position itself belongs to the next shot)
      if (t <= sh.end + 1e-3) return s.x0 + clamp(t - sh.start, 0, sh.end - sh.start - (i < last ? 1e-4 : 0));
    }
    return this.length;
  }

  /** Camera pose on the recovered path at video time t inside shot s. The path has the camera of
      every video frame; between frames, Catmull-Rom through the camera centres and through the
      (hemisphere-aligned) orientation quaternions, so position and rotation change smoothly. */
  poseAt(s, t, outP, outQ) {
    const path = this.shots[s].path;
    const n = path.length;
    if (t <= path[0].t) { outP.copy(path[0].p); outQ.copy(path[0].q); return; }
    if (t >= path[n - 1].t) { outP.copy(path[n - 1].p); outQ.copy(path[n - 1].q); return; }
    // binary search: the path has a sample for every video frame (24 per second)
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (path[mid].t <= t) lo = mid; else hi = mid;
    }
    const i = lo;
    const a = path[i];
    const b = path[i + 1];
    const f = clamp((t - a.t) / (b.t - a.t), 0, 1);
    const p0 = path[Math.max(i - 1, 0)];
    const p3 = path[Math.min(i + 2, n - 1)];
    const w = crWeights(f);
    outP.set(0, 0, 0).addScaledVector(p0.p, w[0]).addScaledVector(a.p, w[1]).addScaledVector(b.p, w[2]).addScaledVector(p3.p, w[3]);
    const q = [p0.q, a.q, b.q, p3.q];
    let x = 0; let y = 0; let z = 0; let ww = 0;
    for (let k = 0; k < 4; k++) { x += q[k].x * w[k]; y += q[k].y * w[k]; z += q[k].z * w[k]; ww += q[k].w * w[k]; }
    outQ.set(x, y, z, ww).normalize();
  }

  /** The two keyframes around time t in shot s and the blend weight of the second one. */
  keyframesAt(s, t) {
    const list = this.shots[s].kfs.filter((k) => k.depth); // keyframes that have their own 3D surface
    let i = 0;
    while (i < list.length - 2 && list[i + 1].t <= t) i++;
    const a = list[i];
    const b = list[Math.min(i + 1, list.length - 1)];
    const f = b.t > a.t ? clamp((t - a.t) / (b.t - a.t), 0, 1) : 0;
    // most of the interval shows one keyframe; the hand-over happens in the middle, and is shorter
    // the more the two viewpoints differ
    const hw = a.blendHalf ?? 0.28;
    return { a, b, f, w: smoothstep(0.5 - hw, 0.5 + hw, f) };
  }

  /** Edge row (0..1 from the top) of the shutter at video time t. */
  shutterEdge(t) {
    const sh = this.shutter;
    const x = (t - sh.t0) * sh.fps;
    const i = clamp(Math.floor(x), 0, sh.edge.length - 1);
    const j = Math.min(i + 1, sh.edge.length - 1);
    const f = clamp(x - i, 0, 1);
    return sh.edge[i] * (1 - f) + sh.edge[j] * f;
  }

  inShutter(t) {
    const sh = this.shutter;
    return !!sh && sh.closedKf && t >= sh.closedKf.t - 1e-4 && t <= sh.openKf.t + 1e-4;
  }

  /** Everything needed to draw timeline position x. */
  frameAt(x, out) {
    // half-open ranges: a cut position belongs to the shot that starts there
    const last = this.segments.length - 1;
    const seg = this.segments.find((s, i) => x < s.x1 || i === last);
    const sh = this.shots[seg.shot];
    const t = sh.start + clamp(x - seg.x0, 0, sh.end - sh.start);
    out.segment = seg;
    out.shot = seg.shot;
    out.time = t;
    this.poseAt(seg.shot, t, out.pos, out.quat);
    out.shutter = this.inShutter(t);
    if (out.shutter) {
      // the camera stands still during the roll-up: hold it exactly at the closed-door view
      // (the recovered poses there only differ by sub-pixel noise)
      this.poseAt(seg.shot, this.shutter.closedKf.t, out.pos, out.quat);
      // shutter roll-up: the static keyframes just before and after t (images only)
      const list = this.shots[seg.shot].kfs.filter((k) => k.t >= this.shutter.closedKf.t - 1e-4 && k.t <= this.shutter.openKf.t + 1e-4);
      let i = 0;
      while (i < list.length - 2 && list[i + 1].t <= t) i++;
      out.kfA = list[i];
      out.kfB = list[Math.min(i + 1, list.length - 1)];
      out.w = out.kfB.t > out.kfA.t ? clamp((t - out.kfA.t) / (out.kfB.t - out.kfA.t), 0, 1) : 0;
      out.edge = this.shutterEdge(t);
      return out;
    }
    const k = this.keyframesAt(seg.shot, t);
    out.kfA = k.a;
    out.kfB = k.b;
    out.w = k.w;
    out.f = k.f;
    return out;
  }

  /* -------- loading -------- */

  /** A keyframe = its video frame (image) + its depth grid; both small files, loaded together.
      Keyframes inside the shutter roll-up only need the image (they share the two anchor surfaces). */
  async load(kf, set) {
    if (kf.blob && (kf.depthData || !kf.depth)) return kf;
    if (!kf.loadPromise) {
      kf.loadPromise = Promise.all([
        fetchOk(`${BASE}${set}/${kf.image}`, 'blob'),
        kf.depth ? fetchOk(`${BASE}depth/${kf.depth}`, 'buffer') : Promise.resolve(null),
      ]).then(([blob, buf]) => {
        kf.blob = blob;
        if (buf) kf.depthData = new Uint16Array(buf);
        return kf;
      }).catch((e) => { kf.loadPromise = null; throw e; });
    }
    return kf.loadPromise;
  }

  loaded(kf) { return !!kf.blob && (!kf.depth || !!kf.depthData); }

  /** World positions of a keyframe's grid vertices from its depth grid (+ depth-jump flags). */
  positions(kf) {
    const { gw, gh } = this;
    const n = gw * gh;
    const q = kf.depthData;
    if (q.length !== n) throw new Error(`depth grid of ${kf.depth} has ${q.length} values, expected ${n}`);
    const pos = new Float32Array(n * 3);
    const edge = new Float32Array(n);
    const [i0, i1] = kf.inv;
    const M = kf.M; // camera (OpenCV axes) -> world rotation, column-major 3x3
    const [cx, cy, cz] = kf.C;
    const tx = kf.tan[0];
    const ty = kf.tan[1];
    for (let y = 0; y < gh; y++) {
      const vy = (2 * y / (gh - 1) - 1) * ty;
      for (let x = 0; x < gw; x++) {
        const i = y * gw + x;
        const v = q[i];
        edge[i] = v & 1;
        const inv = i0 + ((v >> 1) / 32767) * (i1 - i0);
        const z = 1 / inv;
        const xc = (2 * x / (gw - 1) - 1) * tx * z;
        const yc = vy * z;
        pos[i * 3] = M[0] * xc + M[3] * yc + M[6] * z + cx;
        pos[i * 3 + 1] = M[1] * xc + M[4] * yc + M[7] * z + cy;
        pos[i * 3 + 2] = M[2] * xc + M[5] * yc + M[8] * z + cz;
      }
    }
    return { pos, edge };
  }

  /** Builds the 3D surface of a keyframe from its depth grid (once, then cached). */
  geometry(kf) {
    if (kf.geo) return kf.geo;
    const { pos, edge } = this.positions(kf);
    const g = new THREE.BufferGeometry();
    g.setIndex(this.index);
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('uv', this.uv);
    g.setAttribute('edge', new THREE.BufferAttribute(edge, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(...kf.C), 1e4); // never culled
    kf.geo = g;
    return g;
  }

  /** The shutter surface: closed-door and open-door positions for every vertex; the vertex shader
      picks the open one below the moving shutter edge (inside the door). */
  shutterGeometry() {
    if (this.shutterGeo) return this.shutterGeo;
    const closed = this.positions(this.shutter.closedKf);
    const open = this.positions(this.shutter.openKf);
    const g = new THREE.BufferGeometry();
    g.setIndex(this.index);
    g.setAttribute('position', new THREE.BufferAttribute(closed.pos, 3));
    g.setAttribute('positionOpen', new THREE.BufferAttribute(open.pos, 3));
    g.setAttribute('uv', this.uv);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(...this.shutter.closedKf.C), 1e4);
    this.shutterGeo = g;
    return g;
  }
}

/* ------------------------------------------------------------------ boot */

if (RVR.mode === '3d') {
  RVR.started = true;
  start().catch((err) => {
    console.error(err);
    RVR.useVideo(`3D failed to start: ${err && err.message}`);
  });
}

async function start() {
  let tierName = detectTier();
  let tier = TIERS[tierName];
  const canvas = $('showroom-canvas');
  const section = $('showroom');
  const stage = section.querySelector('.showroom__stage');

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, stencil: false, powerPreference: 'high-performance' });
  // The video frames are shown exactly as they are: no colour conversion, no tone mapping.
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.setClearColor(0x000000, 1);
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); RVR.useVideo('graphics context lost'); });

  /* ---- 1. manifest + the first keyframes ---- */
  setStatus('Loading the showroom', 0.03);
  const manifest = await fetchOk(BASE + 'manifest.json', 'json');
  const recon = new Reconstruction(manifest);

  const state = {
    progress: 0, target: 0, vel: 0, look: new THREE.Vector2(), pointer: new THREE.Vector2(), pointerActive: false,
    frame: { pos: new THREE.Vector3(), quat: new THREE.Quaternion() },
  };
  const scrollable = () => Math.max(1, section.offsetHeight - stage.offsetHeight);
  function readScroll() {
    const rect = section.getBoundingClientRect();
    state.target = clamp(-rect.top / scrollable(), 0, 1);
  }
  readScroll();
  state.progress = state.target;
  recon.frameAt(state.progress * recon.length, state.frame);

  // what must be ready before the loader goes away: the keyframes on screen (image + depth each),
  // and the two shutter anchors if the page opens inside the roll-up
  const fr0 = state.frame;
  const first = [...new Set([fr0.kfA, fr0.kfB, ...(fr0.shutter ? [recon.shutter.closedKf, recon.shutter.openKf] : [])])];
  let done = 0;
  const tick1 = () => { done += 1; setStatus(null, 0.05 + 0.8 * (done / first.length)); };
  await Promise.all(first.map((k) => recon.load(k, tier.images).then(tick1)));

  /* ---- 2. GPU side: render targets, materials ---- */
  setStatus('Building the 3D space', 0.88);
  await nextFrame();

  const camA = new THREE.PerspectiveCamera(50, 1, 0.02, 2000);
  const camB = new THREE.PerspectiveCamera(50, 1, 0.02, 2000);
  const rtOpts = { depthBuffer: true, stencilBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false };
  const rtA = new THREE.WebGLRenderTarget(1, 1, rtOpts);
  const rtB = new THREE.WebGLRenderTarget(1, 1, rtOpts);
  let frameRT = new THREE.WebGLRenderTarget(1, 1, { ...rtOpts, depthBuffer: true });
  let snapRT = new THREE.WebGLRenderTarget(1, 1, { ...rtOpts, depthBuffer: false });

  const surfaceVS = /* glsl */`
    attribute float edge;
    varying vec2 vUv;
    varying float vEdge;
    void main() {
      vUv = uv;
      vEdge = edge;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;
  // alpha = how much this pixel can be trusted: low on the 'rubber' triangles that bridge a depth
  // jump (those only fill gaps that no keyframe saw). 0 = nothing drawn here.
  const surfaceFS = /* glsl */`
    uniform sampler2D map;
    varying vec2 vUv;
    varying float vEdge;
    void main() {
      vec3 c = texture2D(map, vUv).rgb;
      float trust = 1.0 - 0.97 * smoothstep(0.34, 0.99, vEdge);
      gl_FragColor = vec4(c, trust);
    }`;
  const makeSurfaceMaterial = () => new THREE.ShaderMaterial({
    uniforms: { map: { value: null } }, vertexShader: surfaceVS, fragmentShader: surfaceFS,
    side: THREE.DoubleSide, blending: THREE.NoBlending,
  });
  const sceneA = new THREE.Scene();
  const sceneB = new THREE.Scene();
  const meshA = new THREE.Mesh(new THREE.BufferGeometry(), makeSurfaceMaterial());
  const meshB = new THREE.Mesh(new THREE.BufferGeometry(), makeSurfaceMaterial());
  meshA.frustumCulled = false;
  meshB.frustumCulled = false;
  sceneA.add(meshA);
  sceneB.add(meshB);

  // Shutter roll-up. The camera stands still and the shutter is a rigid sheet moving up, so every
  // instant can be put together from two nearby frames of the video: the shutter part is taken from
  // each frame *shifted* by how far the edge has moved since/until then (no double edges), the
  // revealed gallery from where that frame already shows it, the rest is cross-faded.
  const shutterMat = new THREE.ShaderMaterial({
    uniforms: {
      mapA: { value: null }, mapB: { value: null }, w: { value: 0 },
      yA: { value: 0 }, yB: { value: 0 }, yT: { value: 0 },
      door: { value: new THREE.Vector4(0, 1, 0, 1) },
    },
    vertexShader: /* glsl */`
      attribute vec3 positionOpen;
      uniform float yT;
      uniform vec4 door;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        bool open = uv.x >= door.x && uv.x <= door.y && uv.y >= yT + 0.006 && uv.y <= door.w + 0.02;
        vec3 p = open ? positionOpen : position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D mapA, mapB;
      uniform float w, yA, yB, yT;
      uniform vec4 door; // x0, x1, top, bottom (image coordinates, 0..1 from the top-left)
      varying vec2 vUv;
      // the measured edge row is the top of the shutter's bright bottom lip; the lip itself reaches
      // about 6 px (of 1080) further down and moves with the sheet
      const float LIP = 0.006;
      vec3 tex(sampler2D m, vec2 uv, vec2 dx, vec2 dy) { return textureGrad(m, uv, dx, dy).rgb; }
      void main() {
        vec2 uv = vUv;
        vec2 dx = dFdx(uv);
        vec2 dy = dFdy(uv);
        vec3 c;
        bool inDoor = uv.x >= door.x && uv.x <= door.y && uv.y >= door.z && uv.y <= door.w;
        if (!inDoor) {
          c = mix(tex(mapA, uv, dx, dy), tex(mapB, uv, dx, dy), w);
        } else if (uv.y < yT + LIP) {
          // shutter: where this bit of the sheet was in frame A (lower) and in frame B (higher)
          vec3 a = tex(mapA, vec2(uv.x, uv.y + (yA - yT)), dx, dy);
          float vb = uv.y - (yT - yB);
          c = vb < door.z ? a : mix(a, tex(mapB, vec2(uv.x, vb), dx, dy), w);
        } else {
          // gallery below the edge: frame B always shows it here, frame A only well below its own lip
          // (a margin of ~13 px of 1080: the measured edge can be a few pixels off while the shutter
          // moves fast, and any bit of A's lip there would show as a faint second line)
          vec3 b = tex(mapB, uv, dx, dy);
          c = uv.y >= yA + LIP + 0.012 ? mix(tex(mapA, uv, dx, dy), b, w) : b;
        }
        gl_FragColor = vec4(c, 1.0);
      }`,
    side: THREE.DoubleSide, blending: THREE.NoBlending,
  });
  const shutterScene = new THREE.Scene();
  const shutterMesh = new THREE.Mesh(new THREE.BufferGeometry(), shutterMat);
  shutterMesh.frustumCulled = false;
  shutterScene.add(shutterMesh);
  if (recon.shutter) {
    const d = recon.shutter.door;
    shutterMat.uniforms.door.value.set(d[0], d[1], d[2], d[3]);
  }

  // Blending of the two keyframe layers. Where they agree (or one has no data) it is a normal
  // cross-fade; where they clearly disagree - a detail that the depth doesn't line up perfectly -
  // the hand-over is made sharp, so you see one or the other but never both at once.
  const composite = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    uniforms: { tA: { value: rtA.texture }, tB: { value: rtB.texture }, wB: { value: 0 }, texel: { value: new THREE.Vector2() } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: /* glsl */`
      uniform sampler2D tA, tB;
      uniform float wB;
      uniform vec2 texel;
      varying vec2 vUv;
      void main() {
        vec4 a = texture2D(tA, vUv);
        vec4 b = texture2D(tB, vUv);
        // disagreement measured on a small neighbourhood (so edges of letters switch together)
        float d = 0.0;
        for (int i = 0; i < 5; i++) {
          vec2 o = i == 0 ? vec2(0.0) : (i == 1 ? vec2(2.5, 0.0) : (i == 2 ? vec2(-2.5, 0.0) : (i == 3 ? vec2(0.0, 2.5) : vec2(0.0, -2.5))));
          vec3 da = texture2D(tA, vUv + o * texel).rgb - texture2D(tB, vUv + o * texel).rgb;
          d = max(d, max(max(abs(da.r), abs(da.g)), abs(da.b)));
        }
        float both = step(0.02, a.a) * step(0.02, b.a);
        float s = smoothstep(0.08, 0.22, d) * both;
        float w = mix(wB, smoothstep(0.42, 0.58, wB), s);
        float wa = (1.0 - w + 0.002) * a.a * a.a;
        float wb = (w + 0.002) * b.a * b.a;
        vec3 c = (a.rgb * wa + b.rgb * wb) / max(wa + wb, 1e-6);
        gl_FragColor = vec4(c, 1.0);
      }`,
    depthTest: false, depthWrite: false,
  }));
  composite.frustumCulled = false;
  const compScene = new THREE.Scene();
  compScene.add(composite);
  const compCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  // final pass to the screen: the current frame, cross-faded with a still of the last frame
  // shown before one of the video's cuts was crossed
  const present = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    uniforms: { tNow: { value: null }, tSnap: { value: null }, fade: { value: 1 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: /* glsl */`
      uniform sampler2D tNow, tSnap;
      uniform float fade;
      varying vec2 vUv;
      void main() {
        vec3 n = texture2D(tNow, vUv).rgb;
        vec3 s = texture2D(tSnap, vUv).rgb;
        gl_FragColor = vec4(mix(s, n, fade), 1.0);
      }`,
    depthTest: false, depthWrite: false,
  }));
  present.frustumCulled = false;
  const presentScene = new THREE.Scene();
  presentScene.add(present);

  /* ---- textures: compressed images stay in memory, a few are decoded on the GPU at a time ---- */
  const gpu = new Map(); // kf index -> texture
  let useCounter = 0;
  const decoding = new Set();
  const pinned = () => new Set([state.frame.kfA, state.frame.kfB, recon.shutter?.closedKf, recon.shutter?.openKf].filter(Boolean));

  async function decode(kf) {
    if (gpu.has(kf.index) || decoding.has(kf.index)) return;
    decoding.add(kf.index);
    try {
      await recon.load(kf, tier.images);
      const { blob } = kf;
      let img;
      try {
        img = await createImageBitmap(blob, { imageOrientation: 'none', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
      } catch (e) {
        img = new Image();
        img.src = URL.createObjectURL(blob);
        await img.decode();
        URL.revokeObjectURL(img.src);
      }
      const tex = new THREE.Texture(img);
      tex.flipY = false;
      tex.colorSpace = THREE.NoColorSpace;
      tex.generateMipmaps = true;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
      tex.needsUpdate = true;
      renderer.initTexture(tex);
      gpu.set(kf.index, tex);
      kf.used = ++useCounter;
      // free the least recently used textures
      if (gpu.size > tier.gpuCache) {
        const keep = pinned();
        const old = [...gpu.keys()].map((i) => recon.kfs[i]).filter((o) => !keep.has(o)).sort((x, y) => x.used - y.used);
        for (const o of old.slice(0, gpu.size - tier.gpuCache)) {
          const t = gpu.get(o.index);
          t.dispose();
          if (t.image && t.image.close) t.image.close();
          gpu.delete(o.index);
          if (o.geo) { o.geo.dispose(); o.geo = null; } // rebuilt from the depth grid when needed again
        }
      }
      renderNow = true;
    } finally {
      decoding.delete(kf.index);
    }
  }

  const ready = (kf) => !!kf && gpu.has(kf.index) && recon.loaded(kf);
  // downloaded (image + depth) for what a frame needs; the roll-up only needs its two anchors
  const downloaded = (fr) => (fr.shutter
    ? recon.loaded(recon.shutter.closedKf) && recon.loaded(recon.shutter.openKf)
    : recon.loaded(fr.kfA) && recon.loaded(fr.kfB));
  const wantFrame = (fr) => {
    if (fr.shutter) { want(recon.shutter.closedKf); want(recon.shutter.openKf); } else { want(fr.kfA); want(fr.kfB); }
  };
  const want = (kf) => {
    if (!kf) return;
    if (recon.loaded(kf)) decode(kf).catch((e) => console.warn(e));
    else recon.load(kf, tier.images).then(() => { renderNow = true; }).catch((e) => console.warn(e));
  };

  /* ---- background loading: the shutter anchors first, then outwards from where the visitor is
          heading (ahead of the scroll direction counts more) ---- */
  let loadingActive = 0;
  function pumpLoading() {
    // order: the shutter anchors, then everything between the camera and where the scroll is heading
    // (nearest first), then up to 12 s of the tour beyond it; the rest waits until it comes closer
    // (a long jump loads its destination first)
    const x = state.progress * recon.length;
    const xt = state.target * recon.length;
    const far = Math.abs(xt - x) > FAR_JUMP;
    const dir = state.target >= state.progress - 1e-4 ? 1 : -1;
    const reach = Math.abs(xt - x) + 12;
    const todo = recon.kfs.filter((k) => !recon.loaded(k) && !k.loadPromise)
      .map((k) => {
        const kx = recon.xOfTime(k.t);
        const d = (kx - x) * dir;
        const anchor = recon.shutter && (k === recon.shutter.closedKf || k === recon.shutter.openKf);
        const dest = far && kx > xt - 1 && kx < xt + 3;
        return { k, d, score: anchor ? -1 : dest ? -0.9 + Math.abs(kx - xt) / 10 : (d >= -0.3 ? Math.abs(d) : -d * 2.5) };
      })
      .filter((o) => o.score < 0 || (o.d > -4 && o.d < reach))
      .sort((p, q) => p.score - q.score);
    while (loadingActive < 4 && todo.length) {
      const { k } = todo.shift();
      loadingActive += 1;
      recon.load(k, tier.images).catch((e) => console.warn(e)).finally(() => { loadingActive -= 1; pumpLoading(); });
    }
  }

  /** Picks what to draw for the current frame (with stand-ins while keyframes are still loading). */
  function choose(fr) {
    // keep the next keyframes in the scroll direction decoded
    const dir = state.target >= state.progress - 1e-4 ? 1 : -1;
    const list = recon.shots[fr.shot].kfs;
    const j = list.indexOf(fr.kfB);
    const settling = Math.abs(state.target - state.progress) * recon.length < 1.5;
    for (const k of [fr.kfA, fr.kfB, list[j + dir], list[j + 2 * dir]]) {
      if (!k) continue;
      if (recon.loaded(k)) decode(k).catch((e) => console.warn(e));
      else if (settling) want(k);
    }
    if (fr.shutter) {
      const sh = recon.shutter;
      want(sh.closedKf);
      want(sh.openKf);
      if (!ready(sh.closedKf) || !ready(sh.openKf)) return null;
      // nearest ready frames around t (the two anchors always work)
      const stat = list.filter((k) => k.t >= sh.closedKf.t - 1e-4 && k.t <= sh.openKf.t + 1e-4);
      let a = fr.kfA;
      let b = fr.kfB;
      if (!ready(a)) a = stat.filter((k) => k.t <= fr.time && ready(k)).pop() || sh.closedKf;
      if (!ready(b)) b = stat.find((k) => k.t >= fr.time && ready(k)) || sh.openKf;
      const w = b.t > a.t ? clamp((fr.time - a.t) / (b.t - a.t), 0, 1) : 0;
      $('scene-status').hidden = !waiting && a === fr.kfA && b === fr.kfB;
      a.used = ++useCounter;
      b.used = ++useCounter;
      return { a, b, w, exact: a === fr.kfA && b === fr.kfB };
    }
    let a = fr.kfA;
    let b = fr.kfB;
    let w = fr.w;
    if (!ready(a) || !ready(b)) {
      // downloaded and only being decoded (a few milliseconds): keep the last picture
      if (recon.loaded(a) && recon.loaded(b) && lastDrawn) return null;
      // stand-in while the right keyframes download: the nearest ready keyframes of the same shot;
      // otherwise keep the last picture
      $('scene-status').hidden = false;
      const pool = list.filter((k) => k.depth && ready(k));
      if (!pool.length) return null;
      const byDist = pool.slice().sort((p, q) => Math.abs(p.t - fr.time) - Math.abs(q.t - fr.time));
      if (!ready(a)) a = ready(b) ? b : byDist[0];
      if (!ready(b)) b = ready(a) ? a : byDist[0];
      if (a === b) w = 0;
    } else {
      $('scene-status').hidden = !waiting;
    }
    a.used = ++useCounter;
    b.used = ++useCounter;
    return { a, b, w, exact: a === fr.kfA && b === fr.kfB };
  }

  /* ---- camera: lens of the original footage, cropped to fill the screen ---- */
  const OVERSCAN = 0.92; // a little narrower than the footage, so moving off the path never shows the frame edge
  let viewW = 1;
  let viewH = 1;
  function lens(cam, kf) {
    const aspect = viewW / viewH;
    const imgAspect = kf.tan[0] / kf.tan[1];
    let tanY = kf.tan[1] * OVERSCAN;
    if (aspect > imgAspect) tanY = (kf.tan[0] * OVERSCAN) / aspect; // wider screen: keep the width, crop top/bottom
    cam.aspect = aspect;
    cam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(tanY));
    cam.updateProjectionMatrix();
  }

  const right = new THREE.Vector3();
  const upv = new THREE.Vector3();
  const fwd = new THREE.Vector3();
  const focus = new THREE.Vector3();
  function placeCamera(cam, pos, quat, kf, look) {
    cam.position.copy(pos);
    cam.quaternion.copy(quat);
    cam.up.set(0, 1, 0);
    if (look.x || look.y) {
      // step sideways/up a little and keep looking at the same point (about the median depth):
      // near things then move against far things = real parallax. How far: a share of the typical
      // depth, capped by the keyframe's parallax budget so close-ups don't tear apart.
      const D = kf.dmed;
      const reach = Math.min(0.03 * D, 0.5 * (kf.par || Infinity));
      right.set(1, 0, 0).applyQuaternion(quat);
      upv.set(0, 1, 0).applyQuaternion(quat);
      fwd.set(0, 0, -1).applyQuaternion(quat);
      focus.copy(pos).addScaledVector(fwd, D);
      cam.position.addScaledVector(right, look.x * reach).addScaledVector(upv, look.y * reach * 0.5);
      cam.up.copy(upv);
      cam.lookAt(focus);
    }
    lens(cam, kf);
  }

  function resize() {
    viewW = stage.clientWidth || window.innerWidth;
    viewH = stage.clientHeight || window.innerHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, tier.maxDpr));
    renderer.setSize(viewW, viewH, false);
    const pr = renderer.getPixelRatio();
    const w = Math.round(viewW * pr);
    const h = Math.round(viewH * pr);
    for (const rt of [rtA, rtB, frameRT, snapRT]) rt.setSize(w, h);
    composite.material.uniforms.texel.value.set(1 / w, 1 / h);
    renderNow = true;
  }

  let renderNow = true;
  let lastDrawn = null;
  let settle = 0; // 0 = moving, 1 = scroll at rest (see tick)
  let waiting = false; // the camera is waiting for keyframes to download
  let forceFade = false; // next draw starts the cut fade even inside one shot (long jumps)
  const probe = { pos: new THREE.Vector3(), quat: new THREE.Quaternion() };
  let restFor = 0;
  let shownShot = null;
  let fadeStart = -1;
  const clockNow = () => performance.now() / 1000;

  function draw() {
    const fr = state.frame;
    const pick = choose(fr);
    if (!pick) return false;
    const { a, b } = pick;
    let { w } = pick;
    // scroll at rest: settle on the nearer keyframe, so a still picture never shows two images
    if (!fr.shutter && pick.exact && settle > 0 && w > 0 && w < 1) w += ((fr.f < 0.5 ? 0 : 1) - w) * settle;
    // crossing one of the video's cuts: keep a still of the last frame and fade from it
    if (shownShot !== null && (fr.shot !== shownShot || forceFade) && CUT_FADE > 0) {
      const t = snapRT;
      snapRT = frameRT;
      frameRT = t;
      fadeStart = clockNow();
    }
    shownShot = fr.shot;
    forceFade = false;

    renderer.setRenderTarget(frameRT);
    renderer.clear();
    if (fr.shutter) {
      shutterMesh.geometry = recon.shutterGeometry();
      const u = shutterMat.uniforms;
      u.mapA.value = gpu.get(a.index);
      u.mapB.value = gpu.get(b.index);
      u.w.value = w;
      u.yA.value = recon.shutterEdge(a.t);
      u.yB.value = recon.shutterEdge(b.t);
      u.yT.value = fr.edge;
      placeCamera(camA, fr.pos, fr.quat, recon.shutter.closedKf, state.look);
      renderer.render(shutterScene, camA);
    } else {
      meshA.geometry = recon.geometry(a);
      meshB.geometry = recon.geometry(b);
      meshA.material.uniforms.map.value = gpu.get(a.index);
      meshB.material.uniforms.map.value = gpu.get(b.index);
      placeCamera(camA, fr.pos, fr.quat, a, state.look);
      camB.copy(camA); // both layers seen from the same camera
      renderer.setRenderTarget(rtA);
      renderer.clear();
      renderer.render(sceneA, camA);
      renderer.setRenderTarget(rtB);
      renderer.clear();
      renderer.render(sceneB, camB);
      renderer.setRenderTarget(frameRT);
      composite.material.uniforms.wB.value = w;
      renderer.render(compScene, compCam);
    }
    renderer.setRenderTarget(null);
    const f = fadeStart < 0 ? 1 : smoothstep(0, CUT_FADE, clockNow() - fadeStart);
    if (f >= 1) fadeStart = -1;
    present.material.uniforms.tNow.value = frameRT.texture;
    present.material.uniforms.tSnap.value = snapRT.texture;
    present.material.uniforms.fade.value = f;
    renderer.render(presentScene, compCam);
    lastDrawn = { a: a.index, b: b.index, w, time: fr.time, shot: fr.shot, shutter: fr.shutter, fade: f, exact: pick.exact && f >= 1 };
    return true;
  }

  resize();

  /* ---- 3. first frame, then reveal ---- */
  setStatus('Preparing the first view', 0.94);
  await Promise.all(first.map((k) => decode(k)));
  if (!draw()) throw new Error('first keyframes did not load');
  await nextFrame();
  draw();
  setStatus('Ready', 1);
  await nextFrame();

  const loader = $('loader');
  loader.classList.add('is-done');
  document.body.classList.remove('is-loading');
  setTimeout(() => { loader.hidden = true; }, reduceMotion ? 0 : 950);
  readScroll();
  pumpLoading();

  /* ---- UI: hint + area navigation ---- */
  const hint = $('scroll-hint');
  const navButtons = [...document.querySelectorAll('#area-nav button')];
  const areas = manifest.areas.map((a) => recon.xOfTime(a.t) / recon.length);
  navButtons.forEach((btn) => btn.addEventListener('click', () => {
    const u = areas[Number(btn.dataset.area)];
    const top = window.scrollY + section.getBoundingClientRect().top + u * scrollable();
    window.scrollTo({ top, behavior: reduceMotion ? 'auto' : 'smooth' });
  }));
  let lastArea = -1;
  function updateUI() {
    hint.classList.toggle('is-hidden', state.target > 0.006);
    let area = 0;
    areas.forEach((u, i) => { if (state.progress >= u - 0.004) area = i; });
    if (area !== lastArea) {
      navButtons.forEach((bt, i) => bt.setAttribute('aria-current', String(i === area)));
      lastArea = area;
    }
  }

  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse') return;
    const rect = canvas.getBoundingClientRect();
    state.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -(((e.clientY - rect.top) / rect.height) * 2 - 1));
    state.pointerActive = true;
  });
  canvas.addEventListener('pointerleave', () => { state.pointer.set(0, 0); });
  window.addEventListener('scroll', readScroll, { passive: true });
  window.addEventListener('resize', () => { resize(); readScroll(); });

  /* ---- adaptive quality: lower the resolution if the device can't keep up ---- */
  const perf = { frames: 0, sum: 0 };
  function watchPerformance(dt) {
    if (params.get('quality')) return;
    perf.frames += 1;
    if (perf.frames < 30) return;
    perf.sum += dt;
    if (perf.frames - 30 >= 90) {
      const avg = perf.sum / 90;
      perf.frames = 0;
      perf.sum = 0;
      const i = ORDER.indexOf(tierName);
      if (avg > 1 / 28 && i > 0) {
        tierName = ORDER[i - 1];
        tier = { ...TIERS[tierName], images: tier.images }; // keep the image set that is already loaded
        console.info(`RVR showroom: ${Math.round(1 / avg)} fps, quality -> ${tierName}`);
        resize();
      }
    }
  }

  /* ---- render loop ---- */
  const clock = new THREE.Clock();
  let idle = 0;
  let frameNo = 0;
  let pumpTimer = 0;
  const OMEGA = 5.0; // scroll spring: critically damped, settles in about 0.9 s, no overshoot
  function tick() {
    const dt = Math.min(clock.getDelta(), 0.05);
    const time = clock.elapsedTime;
    frameNo += 1;
    const before = state.progress;
    // crossing one of the video's cuts near where the scroll stops: the camera holds still until the
    // short fade has finished (it does not keep drifting while two pictures are on screen)
    const holdForCut = fadeStart >= 0 && clockNow() - fadeStart < CUT_FADE
      && Math.abs(state.target - state.progress) * recon.length < 2.5;
    if (holdForCut) {
      state.vel = 0;
    } else if (reduceMotion) {
      state.progress = state.target;
      state.vel = 0;
    } else {
      // critically damped spring: the camera eases in and out of every scroll instead of jumping
      // to full speed on each wheel step
      const steps = Math.max(1, Math.ceil(dt / (1 / 120)));
      const h = dt / steps;
      for (let s = 0; s < steps; s++) {
        const acc = OMEGA * OMEGA * (state.target - state.progress) - 2 * OMEGA * state.vel;
        state.vel += acc * h;
        state.progress += state.vel * h;
      }
      if (Math.abs(state.target - state.progress) < 2e-6 && Math.abs(state.vel) < 2e-5) {
        state.progress = state.target;
        state.vel = 0;
      }
      state.progress = clamp(state.progress, 0, 1);
    }

    // pointer parallax (mouse); on touch screens a slow, small sway shows the depth instead
    const lookBefore = state.look.clone();
    if (!reduceMotion) {
      let gx = state.pointer.x;
      let gy = state.pointer.y;
      if (!finePointer && !state.pointerActive) {
        gx = Math.sin(time * 0.42) * 0.55;
        gy = Math.sin(time * 0.29) * 0.35;
      }
      const kl = 1 - Math.exp(-dt * 2.2);
      state.look.x += (gx - state.look.x) * kl;
      state.look.y += (gy - state.look.y) * kl;
    }
    recon.frameAt(state.progress * recon.length, state.frame);
    // the keyframes for the new position haven't downloaded yet: wait here (with the "Loading"
    // notice) instead of running ahead and showing stand-in pictures that later jump
    waiting = !downloaded(state.frame);
    if (waiting) {
      const far = Math.abs(state.target - before) * recon.length > FAR_JUMP;
      if (far) recon.frameAt(state.target * recon.length, probe);
      if (far && downloaded(probe)) {
        // a long jump (area buttons, fast scrolling) whose destination has arrived but the way there
        // hasn't: go straight there with the short cut fade instead of crawling through the tour
        state.progress = state.target;
        state.vel = 0;
        forceFade = true;
        waiting = false;
      } else {
        wantFrame(far ? probe : state.frame);
        state.progress = before;
        state.vel = 0;
      }
      recon.frameAt(state.progress * recon.length, state.frame);
      if (waiting) $('scene-status').hidden = false;
    }
    // settle factor: rises once the camera has been at rest for a moment, drops at once on scrolling
    const atRest = state.progress === state.target && state.vel === 0;
    restFor = atRest ? restFor + dt : 0;
    const settleBefore = settle;
    if (reduceMotion) settle = atRest ? 1 : 0;
    else settle = restFor > 0.12 ? Math.min(1, settle + dt / 0.3) : Math.max(0, settle - dt / 0.1);
    const fading = fadeStart >= 0;
    const moving = Math.abs(state.progress - before) > 1e-7 || state.look.distanceTo(lookBefore) > 1e-4 || fading
      || settle !== settleBefore;
    updateUI();
    if ((pumpTimer += dt) > 0.25) { pumpTimer = 0; pumpLoading(); }

    idle = moving || renderNow ? 0 : idle + 1;
    if (idle < 3 || frameNo % 8 === 0) { // nothing changes when idle: draw rarely
      renderNow = false;
      draw();
    }
    if (moving) watchPerformance(dt);
  }
  renderer.setAnimationLoop(tick);

  window.__rvrRecon = {
    renderer, recon, state, gpu,
    get tier() { return tierName; },
    get last() { return lastDrawn; },
    get waiting() { return waiting; },
    get settle() { return settle; },
    timelineX: (t) => recon.xOfTime(t) / recon.length,
    loaded: () => recon.kfs.filter((k) => recon.loaded(k)).length,
    loadAll: () => Promise.all(recon.kfs.map((k) => recon.load(k, tier.images))),
    forceDraw: () => { lastDrawn = null; renderNow = true; },
    drawNow: () => (draw(), lastDrawn), // testing: draw the current state right away
  };
}
