/* RVR showroom, reconstructed from the client video (assets/RVR1.mp4).

   What is drawn: for every keyframe of the video there is a 3D surface (a depth mesh) that was
   reconstructed from the footage, placed where the camera really was when that frame was filmed
   (camera path recovered by structure-from-motion). Each surface is textured with its original
   video frame. Scrolling moves a real 3D camera along the recovered camera path; the two nearest
   keyframe surfaces are rendered from that camera and blended, so you see true perspective and
   parallax. The mouse moves the camera a little off the path to show the depth.

   If 3D can't run, the page switches to the original scroll-video (window.RVR.useVideo). */
import * as THREE from 'three';

const RVR = window.RVR || { mode: '3d', useVideo() {} };
const params = new URLSearchParams(location.search);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = matchMedia('(pointer: fine)').matches;
const $ = (id) => document.getElementById(id);
const BASE = 'assets/scene/';

/* ------------------------------------------------------------------ quality tiers */

const TIERS = {
  high: { maxDpr: 1.5, images: 'hd', gpuCache: 10 },
  medium: { maxDpr: 1.25, images: 'hd', gpuCache: 8 },
  low: { maxDpr: 1.0, images: 'sd', gpuCache: 6 },
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

/* ------------------------------------------------------------------ the reconstruction */

class Reconstruction {
  constructor(manifest) {
    this.m = manifest;
    const [gw, gh] = manifest.grid;
    this.gw = gw;
    this.gh = gh;
    this.kfs = manifest.keyframes.map((k, i) => ({ ...k, index: i, blob: null, depthData: null, geo: null, used: 0 }));
    this.shots = manifest.shots.map((s, i) => ({
      ...s,
      index: i,
      kfs: this.kfs.filter((k) => k.shot === i),
      path: manifest.path.filter((p) => p.shot === i).map((p) => ({
        t: p.t,
        p: new THREE.Vector3(...p.p),
        q: new THREE.Quaternion(...p.q),
      })),
    }));
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
    this.tmp = { pA: new THREE.Vector3(), qA: new THREE.Quaternion(), pB: new THREE.Vector3(), qB: new THREE.Quaternion() };
    this.buildTimeline();
  }

  /* Scroll timeline: every shot plays at video speed; between shots there is a short transition. */
  buildTimeline() {
    const segs = [];
    let x = 0;
    this.shots.forEach((s, i) => {
      const len = s.end - s.start;
      segs.push({ type: 'shot', shot: i, x0: x, x1: x + len });
      x += len;
      const tr = this.m.transitions[i];
      if (tr) {
        segs.push({ type: tr.type, from: i, to: i + 1, x0: x, x1: x + tr.length });
        x += tr.length;
      }
    });
    this.segments = segs;
    this.length = x;
  }

  /** Position on the timeline (0..length) of a video time. */
  xOfTime(t) {
    for (const s of this.segments) {
      if (s.type !== 'shot') continue;
      const sh = this.shots[s.shot];
      if (t <= sh.end + 1e-3) return s.x0 + clamp(t - sh.start, 0, sh.end - sh.start);
    }
    return this.length;
  }

  /** Camera pose on the recovered path at video time t inside shot s (continues straight on a
      little before the first and after the last sample, for the dissolves between shots). */
  poseAt(s, t, outP, outQ) {
    const path = this.shots[s].path;
    const n = path.length;
    if (t <= path[0].t || t >= path[n - 1].t) {
      const end = t >= path[n - 1].t;
      const a = end ? path[n - 2] : path[0];
      const b = end ? path[n - 1] : path[1];
      const ref = end ? b : a;
      const k = (t - ref.t) / Math.max(b.t - a.t, 1e-6);
      outP.copy(ref.p).addScaledVector(b.p, k).addScaledVector(a.p, -k);
      outQ.copy(ref.q);
      return;
    }
    let i = 0;
    while (i < n - 2 && path[i + 1].t <= t) i++;
    const a = path[i];
    const b = path[i + 1];
    const f = clamp((t - a.t) / (b.t - a.t), 0, 1);
    // Catmull-Rom through the camera centres, slerp for the orientation
    const p0 = path[Math.max(i - 1, 0)].p;
    const p3 = path[Math.min(i + 2, n - 1)].p;
    const f2 = f * f;
    const f3 = f2 * f;
    outP.set(0, 0, 0)
      .addScaledVector(p0, -0.5 * f3 + f2 - 0.5 * f)
      .addScaledVector(a.p, 1.5 * f3 - 2.5 * f2 + 1)
      .addScaledVector(b.p, -1.5 * f3 + 2 * f2 + 0.5 * f)
      .addScaledVector(p3, 0.5 * f3 - 0.5 * f2);
    outQ.slerpQuaternions(a.q, b.q, f);
  }

  /** The two keyframes around time t in shot s and the blend weight of the second one. */
  keyframesAt(s, t) {
    const list = this.shots[s].kfs;
    let i = 0;
    while (i < list.length - 2 && list[i + 1].t <= t) i++;
    const a = list[i];
    const b = list[Math.min(i + 1, list.length - 1)];
    const f = b.t > a.t ? clamp((t - a.t) / (b.t - a.t), 0, 1) : 0;
    return { a, b, w: smoothstep(0.12, 0.88, f) };
  }

  /** Everything needed to draw timeline position x. */
  frameAt(x, out) {
    const seg = this.segments.find((s) => x <= s.x1) || this.segments[this.segments.length - 1];
    out.segment = seg;
    if (seg.type === 'shot') {
      const sh = this.shots[seg.shot];
      const t = sh.start + clamp(x - seg.x0, 0, sh.end - sh.start);
      out.time = t;
      this.poseAt(seg.shot, t, out.posA, out.quatA);
      out.posB.copy(out.posA);
      out.quatB.copy(out.quatA);
      const k = this.keyframesAt(seg.shot, t);
      out.kfA = k.a;
      out.kfB = k.b;
      out.w = k.w;
      out.sharedCamera = true;
      return out;
    }
    // transition between two shots
    const f = clamp((x - seg.x0) / (seg.x1 - seg.x0), 0, 1);
    const from = this.shots[seg.from];
    const to = this.shots[seg.to];
    out.kfA = from.kfs[from.kfs.length - 1];
    out.kfB = to.kfs[0];
    out.time = f < 0.5 ? from.end : to.start;
    const { pA, qA, pB, qB } = this.tmp;
    this.poseAt(seg.from, from.end, pA, qA);
    this.poseAt(seg.to, to.start, pB, qB);
    if (seg.type === 'fly') {
      // both shots live in one coordinate system: the camera travels from one to the other
      const e = smoothstep(0, 1, f);
      out.posA.lerpVectors(pA, pB, e);
      out.quatA.slerpQuaternions(qA, qB, e);
      out.posB.copy(out.posA);
      out.quatB.copy(out.quatA);
      out.w = smoothstep(0.3, 0.7, f);
      out.sharedCamera = true;
    } else {
      // shots that could not be placed in one space: each keeps its own camera, which keeps moving
      // the way it was moving, and the picture dissolves from one shot to the other
      const half = (seg.x1 - seg.x0) * 0.1; // only a short way past the footage: further out, the 3D breaks up
      this.poseAt(seg.from, from.end + f * half, out.posA, out.quatA);
      this.poseAt(seg.to, to.start - (1 - f) * half, out.posB, out.quatB);
      out.w = smoothstep(0.2, 0.8, f);
      out.sharedCamera = false;
    }
    return out;
  }

  /* -------- loading -------- */

  /** A keyframe = its video frame (image) + its depth grid; both small files, loaded together. */
  async load(kf, set) {
    if (kf.blob && kf.depthData) return kf;
    if (!kf.loadPromise) {
      kf.loadPromise = Promise.all([
        fetchOk(`${BASE}${set}/${kf.image}`, 'blob'),
        fetchOk(`${BASE}depth/${kf.depth}`, 'buffer'),
      ]).then(([blob, buf]) => {
        kf.blob = blob;
        kf.depthData = new Uint16Array(buf);
        return kf;
      }).catch((e) => { kf.loadPromise = null; throw e; });
    }
    return kf.loadPromise;
  }

  /** Builds the 3D surface of a keyframe from its depth grid (once, then cached). */
  geometry(kf) {
    if (kf.geo) return kf.geo;
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
    const g = new THREE.BufferGeometry();
    g.setIndex(this.index);
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('uv', this.uv);
    g.setAttribute('edge', new THREE.BufferAttribute(edge, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(cx, cy, cz), 1e4); // never culled
    kf.geo = g;
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
    progress: 0, target: 0, look: new THREE.Vector2(), pointer: new THREE.Vector2(), pointerActive: false,
    frame: { posA: new THREE.Vector3(), quatA: new THREE.Quaternion(), posB: new THREE.Vector3(), quatB: new THREE.Quaternion() },
  };
  const scrollable = () => Math.max(1, section.offsetHeight - stage.offsetHeight);
  function readScroll() {
    const rect = section.getBoundingClientRect();
    state.target = clamp(-rect.top / scrollable(), 0, 1);
  }
  readScroll();
  state.progress = state.target;
  recon.frameAt(state.progress * recon.length, state.frame);

  // what must be ready before the loader goes away: the 2 keyframes on screen (image + depth each)
  const first = [...new Set([state.frame.kfA, state.frame.kfB])];
  let done = 0;
  const tick1 = () => { done += 1; setStatus(null, 0.05 + 0.8 * (done / first.length)); };
  await Promise.all(first.map((k) => recon.load(k, tier.images).then(tick1)));

  /* ---- 2. GPU side: render targets, materials ---- */
  setStatus('Building the 3D space', 0.88);
  await nextFrame();

  const camA = new THREE.PerspectiveCamera(50, 1, 0.02, 2000);
  const camB = new THREE.PerspectiveCamera(50, 1, 0.02, 2000);
  const rtOpts = { depthBuffer: true, stencilBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false };
  const rtA = new THREE.WebGLRenderTarget(1, 1, rtOpts);
  const rtB = new THREE.WebGLRenderTarget(1, 1, rtOpts);

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

  const composite = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    uniforms: { tA: { value: rtA.texture }, tB: { value: rtB.texture }, wB: { value: 0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: /* glsl */`
      uniform sampler2D tA, tB;
      uniform float wB;
      varying vec2 vUv;
      void main() {
        vec4 a = texture2D(tA, vUv);
        vec4 b = texture2D(tB, vUv);
        float wa = (1.0 - wB + 0.002) * a.a * a.a;
        float wb = (wB + 0.002) * b.a * b.a;
        vec3 c = (a.rgb * wa + b.rgb * wb) / max(wa + wb, 1e-6);
        gl_FragColor = vec4(c, 1.0);
      }`,
    depthTest: false, depthWrite: false,
  }));
  composite.frustumCulled = false;
  const compScene = new THREE.Scene();
  compScene.add(composite);
  const compCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  /* ---- textures: compressed images stay in memory, a few are decoded on the GPU at a time ---- */
  const gpu = new Map(); // kf index -> texture
  let useCounter = 0;
  const decoding = new Set();

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
        const old = [...gpu.keys()].map((i) => recon.kfs[i]).sort((x, y) => x.used - y.used);
        for (const o of old.slice(0, gpu.size - tier.gpuCache)) {
          if (o === state.frame.kfA || o === state.frame.kfB) continue;
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

  const ready = (kf) => gpu.has(kf.index) && !!kf.depthData;

  /* ---- background loading, nearest to the current position first (ahead counts more) ---- */
  let loadingActive = 0;
  function pumpLoading() {
    const x = state.target * recon.length; // where the visitor is heading
    const dir = state.target >= state.progress ? 1 : -1;
    const todo = recon.kfs.filter((k) => !k.depthData && !k.loadPromise)
      .map((k) => {
        const kx = recon.xOfTime(k.t);
        const d = (kx - x) * dir;
        return { k, score: d >= 0 ? d : -d * 2.5 };
      })
      .sort((p, q) => p.score - q.score);
    while (loadingActive < 3 && todo.length) {
      const { k } = todo.shift();
      loadingActive += 1;
      recon.load(k, tier.images).catch((e) => console.warn(e)).finally(() => { loadingActive -= 1; pumpLoading(); });
    }
  }

  /** Makes sure the keyframes needed now (and next) are decoded; returns the two to draw. */
  function choose(fr) {
    const want = [fr.kfA, fr.kfB];
    // prefetch the following keyframes in the scroll direction
    const dir = state.target >= state.progress ? 1 : -1;
    const sh = recon.shots[fr.kfB.shot].kfs;
    const j = sh.indexOf(fr.kfB);
    for (let d = 1; d <= 2; d++) {
      const n = sh[clamp(j + d * dir, 0, sh.length - 1)];
      if (n) want.push(n);
    }
    // while the camera is still gliding towards a far-away scroll position, don't spend bandwidth on
    // the keyframes it passes on the way (the background loader fetches around the target first)
    const settling = Math.abs(state.target - state.progress) * recon.length < 1.5;
    for (const k of want) {
      if (k.blob && k.depthData) decode(k).catch((e) => console.warn(e));
      else if (settling) recon.load(k, tier.images).then(() => { renderNow = true; }).catch((e) => console.warn(e));
    }
    let a = fr.kfA;
    let b = fr.kfB;
    let w = fr.w;
    if (!ready(a) || !ready(b)) {
      // stand-in while the right keyframes download: the nearest ready keyframes that live in the
      // same 3D space (same shot, or shots joined into one space); otherwise keep the last picture
      $('scene-status').hidden = false;
      const worldOf = (k) => recon.shots[k.shot].world;
      const pool = recon.kfs.filter((k) => ready(k) && (fr.sharedCamera ? worldOf(k) === worldOf(a) : k.shot === a.shot || k.shot === b.shot));
      if (!pool.length) return null;
      const t = fr.time;
      const byDist = pool.slice().sort((p, q) => Math.abs(p.t - t) - Math.abs(q.t - t));
      if (!ready(a)) a = ready(b) ? b : byDist[0];
      if (!ready(b)) b = ready(a) ? a : byDist[0];
      if (a === b) w = 0;
    } else {
      $('scene-status').hidden = true;
    }
    a.used = ++useCounter;
    b.used = ++useCounter;
    return { a, b, w };
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
    if (look.x || look.y) {
      // step sideways/up a little and keep looking at the same point (about the median depth):
      // near things then move against far things = real parallax
      // how far: a share of the typical depth, capped by the keyframe's parallax budget (computed
      // from its depth range) so close-up shots don't tear apart
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
    rtA.setSize(Math.round(viewW * pr), Math.round(viewH * pr));
    rtB.setSize(Math.round(viewW * pr), Math.round(viewH * pr));
    renderNow = true;
  }

  let renderNow = true;
  let lastDrawn = null;
  function draw() {
    const fr = state.frame;
    const pick = choose(fr);
    if (!pick) return false;
    const { a, b, w } = pick;
    meshA.geometry = recon.geometry(a);
    meshB.geometry = recon.geometry(b);
    meshA.material.uniforms.map.value = gpu.get(a.index);
    meshB.material.uniforms.map.value = gpu.get(b.index);
    placeCamera(camA, fr.posA, fr.quatA, a, state.look);
    if (fr.sharedCamera) camB.copy(camA); // both layers seen from the same camera
    else placeCamera(camB, fr.posB, fr.quatB, b, state.look);
    renderer.setRenderTarget(rtA);
    renderer.clear();
    renderer.render(sceneA, camA);
    renderer.setRenderTarget(rtB);
    renderer.clear();
    renderer.render(sceneB, camB);
    renderer.setRenderTarget(null);
    composite.material.uniforms.wB.value = w;
    renderer.render(compScene, compCam);
    lastDrawn = { a: a.index, b: b.index, w, time: fr.time, segment: fr.segment.type };
    return true;
  }

  resize();

  /* ---- 3. first frame, then reveal ---- */
  setStatus('Preparing the first view', 0.94);
  await Promise.all(first.map((k) => decode(k)));  // eslint-disable-line
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
  function tick() {
    const dt = Math.min(clock.getDelta(), 0.1);
    const time = clock.elapsedTime;
    frameNo += 1;
    const before = state.progress;
    const k = reduceMotion ? 1 : 1 - Math.exp(-dt * 3.5);
    state.progress += (state.target - state.progress) * k;
    if (Math.abs(state.target - state.progress) < 1e-6) state.progress = state.target;

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
    const moving = Math.abs(state.progress - before) > 1e-7 || state.look.distanceTo(lookBefore) > 1e-4;

    recon.frameAt(state.progress * recon.length, state.frame);
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
    timelineX: (t) => recon.xOfTime(t) / recon.length,
    loaded: () => recon.kfs.filter((k) => k.blob && k.depthData).length,
  };
}
