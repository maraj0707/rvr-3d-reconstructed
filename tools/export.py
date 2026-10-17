"""Turns the reconstruction into the website's scene files.

Reads  : <r2>/feats.npz, <r2>/frames/*.png, <r2>/shot{k}.npz, <r2>/sims.npy, the video
Writes : <site>/assets/scene/manifest.json
         <site>/assets/scene/depth/kNNN.bin     (uint16 per grid vertex: 15-bit inverse depth << 1 | edge flag)
         <site>/assets/scene/hd/kNNN.webp       (1600x900 original frames)
         <site>/assets/scene/sd/kNNN.webp       (960x540)
and diagnostic images in <r2>/export/.

Usage: python3 export.py <video> <r2_dir> <site_dir> [--only-depth K,K,..] [--no-images]
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

from depth import DW, DH, floor_plane, keyframe_depth

SHOT_NAMES = ['Entrance', 'Gallery', 'Gallery tables', 'Towards the workshop', 'Workshop bench', 'Lounge']
# Where the camera jumps inside a reconstructed shot (hidden cut with a 1 m jump at 18.625 s): the website
# treats it like a cut (separate parts, dissolve) instead of flying the camera through the jump.
JUMPS = {0: [18.625]}
AREAS = [('Entrance', 0.0), ('Gallery', 15.75), ('Workshop', 28.9), ('Lounge', 40.5)]
GW, GH = 192, 108
HD, SD = (1600, 900), (960, 540)
# Measured (eval_depth.py): clamping depth to a floor plane made reprojection worse in every shot (the
# mirror images in the glossy floor behave like real geometry below it), so it is switched off.
FLOOR_SHOTS = ()
# Cuts whose bridge was good enough to fly the camera through (checked visually): shot 2 -> shot 3.
# The 3 -> 4 bridge has a few degrees of error and looked broken mid-flight, so that cut dissolves.
FLY_CUTS = {(1, 2)}
HIDDEN_CUTS = {0: [15.625, 18.625]}  # jumps inside shot 1 that the cut detector missed (frame differences)
STATIC = (6.0, 15.59)  # shot 1: the camera stands still in front of the door while the shutter rolls up
OPENING = (7.25, 14.5)  # shutter moving: keyframes every 0.25 s instead of 0.5 s
T0 = time.time()
REUSE = False


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def load(r2):
    f = np.load(os.path.join(r2, 'feats.npz'))
    times, shot = f['times'], f['shot']
    shots = [dict(np.load(os.path.join(r2, f'shot{s}.npz'))) for s in range(5)]
    bridges = {}
    p = os.path.join(r2, 'bridges.npy')
    if os.path.exists(p):
        for k, v in np.load(p, allow_pickle=True):
            bridges[tuple(k)] = v
    return times, shot, shots, bridges


def world_frames(shots, bridges):
    """Per shot: (scale, Rw, origin) so that X_world = scale * Rw @ (X - origin), +Y up, floor y = 0.
    Consecutive shots whose cut could be bridged reliably share one world (the first shot of the chain
    defines it); the others keep their own floor-aligned frame. 'group' tells which shots share a world."""
    out = []
    for s, sd in enumerate(shots):
        Rs = sd['R']
        C = np.array([-R.T @ t for R, t in zip(Rs, sd['t'])])
        n, p0, cnt = floor_plane(sd['pts'], list(Rs), list(C))
        y = n
        fwd = np.mean([R[2] for R in Rs], axis=0)  # camera +z (viewing direction) in shot coords
        z = -(fwd - (fwd @ y) * y)
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        Rw = np.stack([x, y, z])
        h = np.median((C - p0) @ n)
        out.append(dict(scale=1.6 / h, Rw=Rw, origin=C[0] - ((C[0] - p0) @ n) * n, floor_n=n, floor_p=p0,
                        cam_h=h, floor_support=cnt, group=s))
    chain = {}  # shot -> (s, R, t) with X_root = s R X_shot + t
    for s in range(1, len(shots)):
        br = bridges.get((s - 1, s))
        if br is None or br[3] < 3 or br[5] > 3.0 or (s - 1, s) not in FLY_CUTS:
            continue
        sc, R, t = br[0], br[1], br[2]  # X_{s-1} = sc R X_s + t
        root = out[s - 1]['group']
        if s - 1 in chain:
            s1, R1, t1 = chain[s - 1]
        else:
            s1, R1, t1 = 1.0, np.eye(3), np.zeros(3)
        sk, Rk, tk = s1 * sc, R1 @ R, s1 * R1 @ t + t1
        chain[s] = (sk, Rk, tk)
        w0 = out[root]
        o_s = np.linalg.solve(sk * Rk, (w0['origin'] - tk))
        own_n = out[s]['floor_n']
        out[s].update(scale=w0['scale'] * sk, Rw=w0['Rw'] @ Rk, origin=o_s, group=root)
        # check: this shot's own floor, carried into the shared world, should be level and at y = 0
        n_w = out[s]['Rw'] @ own_n
        fp_w = out[s]['scale'] * out[s]['Rw'] @ (out[s]['floor_p'] - o_s)
        out[s]['check'] = (float(np.degrees(np.arccos(np.clip(n_w[1], -1, 1)))), float(fp_w[1]))
    return out


def to_world(wf, X):
    return wf['scale'] * (np.asarray(X) - wf['origin']) @ wf['Rw'].T


def cam_world(wf, R, t):
    """OpenCV world->camera (R, t) in shot coords -> (R_new, C_world) in world coords."""
    C = -R.T @ t
    return R @ wf['Rw'].T, to_world(wf, C)


def three_quat(R_new):
    """Quaternion (x, y, z, w) of a Three.js camera (looks down -Z, +Y up) with OpenCV rotation R_new."""
    M = R_new.T @ np.diag([1.0, -1.0, -1.0])
    tr = np.trace(M)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (M[2, 1] - M[1, 2]) / S, (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w, x, y, z = (M[2, 1] - M[1, 2]) / S, 0.25 * S, (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w, x, y, z = (M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S, 0.25 * S, (M[1, 2] + M[2, 1]) / S
    else:
        S = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w, x, y, z = (M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S, (M[1, 2] + M[2, 1]) / S, 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def pick_keyframes(times, frames, cuts=()):
    """Indices (into frames) of keyframes: every ~0.5 s, every SfM frame (0.25 s) while the shutter
    moves, and always the frames on both sides of a (hidden) cut and of the static period."""
    ts = times[frames]
    must = {0, len(frames) - 1}
    for c in list(cuts) + [STATIC[0], STATIC[1]]:
        before = np.where(ts < c)[0]
        after = np.where(ts >= c)[0]
        if len(before) and len(after):
            must |= {int(before[-1]), int(after[0])}
    keep = [0]
    for k in range(1, len(frames)):
        gap = 0.24 if OPENING[0] <= ts[k] <= OPENING[1] else 0.49
        if k in must or ts[k] - ts[keep[-1]] >= gap - 1e-6:
            if keep[-1] in must or k in must or ts[k] - ts[keep[-1]] >= 0.2:
                keep.append(k)
    return sorted(set(keep) | must)


def repair_poses(ts, Rs, tvecs, cuts=(), log=print):
    """Frames whose camera jumps off the smooth path (e.g. when a moving video wall fills the view)
    get their pose interpolated from the neighbouring good frames. Real jumps at cuts are kept."""
    from scipy.spatial.transform import Rotation, Slerp
    C = np.array([-R.T @ t for R, t in zip(Rs, tvecs)])
    seg = np.searchsorted(np.array(sorted(cuts)), ts, side='right')
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    med_step = np.median(steps[steps > 1e-6]) if np.any(steps > 1e-6) else 1.0
    bad = np.zeros(len(C), bool)
    for sg in np.unique(seg):
        idx = np.where(seg == sg)[0]
        if len(idx) < 5:
            continue
        med = np.array([np.median(C[idx[max(0, k - 3):k + 4]], axis=0) for k in range(len(idx))])
        dev = np.linalg.norm(C[idx] - med, axis=1)
        thr = max(4 * np.median(dev), 0.5 * med_step)
        bad[idx[dev > thr]] = True
    Rs, tvecs = Rs.copy(), tvecs.copy()
    for k in np.where(bad)[0]:
        same = np.where((seg == seg[k]) & ~bad)[0]
        lo = same[same < k]
        hi = same[same > k]
        if not len(lo) or not len(hi):
            continue
        a_, b_ = lo[-1], hi[0]
        f = (ts[k] - ts[a_]) / (ts[b_] - ts[a_])
        rot = Slerp([0, 1], Rotation.from_matrix([Rs[a_], Rs[b_]]))([f]).as_matrix()[0]
        Ck = (1 - f) * C[a_] + f * C[b_]
        Rs[k] = rot
        tvecs[k] = -rot @ Ck
        log(f'  pose of t={ts[k]:.2f}s repaired (was {np.linalg.norm(C[k] - Ck) / med_step:.1f} steps off the path)')
    return Rs, tvecs


def shutter_edges(imgs, closed, opened):
    """Door rectangle (from the closed vs open frame) and, for each image, the row of the shutter's
    bottom edge (the strongest dark-to-bright step, moving only upwards over time)."""
    d = cv2.GaussianBlur(np.abs(closed.astype(np.float32) - opened.astype(np.float32)).max(-1), (5, 5), 0) > 18
    rows = d.mean(1)
    door_rows = np.where(rows > 0.55)[0]
    y0, y1 = int(door_rows.min()), int(door_rows.max())
    cols = d[y0:y1 + 1].mean(0)
    door_cols = np.where(cols > 0.5)[0]
    x0, x1 = int(door_cols.min()), int(door_cols.max())
    edges = []
    prev = y1
    vel = 0.0
    for im in imgs:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prof = np.median(g[:, x0 + 10:x1 - 10], axis=1)
        pred = prev + vel  # the shutter rises steadily: look near where it should be now
        lo_, hi_ = int(max(y0 + 3, pred - 14)), int(min(y1, prev + 2))
        best = (-1e9, None)
        for y in range(lo_, hi_ + 1):
            step = prof[y:y + 4].mean() - prof[y - 4:y].mean()
            if step > best[0]:
                best = (step, y)
        edge = best[1] if best[1] is not None and best[0] > 15 else int(round(pred))
        edge = int(np.clip(edge, y0, prev))
        if len(edges) >= 1 and edge < prev:
            vel = 0.5 * vel + 0.5 * (edge - prev)
        edges.append(edge)
        prev = edge
    return (x0, x1, y0, y1), edges


def grid_from_depth(d):
    """Depth map (DH, DW) -> inverse depth at the GW x GH grid vertices (vertex centres on pixel edges)."""
    xs = np.linspace(0, DW - 1, GW)
    ys = np.linspace(0, DH - 1, GH)
    inv = 1.0 / d
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(inv.astype(np.float32), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)


def parallax_budget(d_world, f, frac=0.034):
    """How far (world units) the camera may move sideways from this keyframe so that near things shift
    against far things by about `frac` of the picture width. Measured on the middle of the picture
    (the floor right in front of the camera would otherwise dominate)."""
    mid = d_world[int(DH * 0.1):int(DH * 0.72)]
    d10, d90 = np.percentile(mid, 10), np.percentile(mid, 90)
    span = max(1 / d10 - 1 / d90, 1e-6)
    return float(frac * 960 / (f * span))


def edge_flags(g, thr=0.045):
    """Vertices next to a depth jump. On a plane inverse depth is linear in the image, so the second
    difference is ~0; a jump makes it large on both sides."""
    e = np.zeros(g.shape, bool)
    lx = np.abs(g[:, :-2] - 2 * g[:, 1:-1] + g[:, 2:]) / g[:, 1:-1]
    ly = np.abs(g[:-2, :] - 2 * g[1:-1, :] + g[2:, :]) / g[1:-1, :]
    e[:, 1:-1] |= lx > thr
    e[1:-1, :] |= ly > thr
    # also a large ratio between direct neighbours
    rx = np.abs(np.diff(g, axis=1)) / np.minimum(g[:, 1:], g[:, :-1])
    ry = np.abs(np.diff(g, axis=0)) / np.minimum(g[1:, :], g[:-1, :])
    e[:, 1:] |= rx > 0.25
    e[:, :-1] |= rx > 0.25
    e[1:, :] |= ry > 0.25
    e[:-1, :] |= ry > 0.25
    return e


def main(video, r2, site, only=None, images=True):
    times, shot_of, shots, bridges = load(r2)
    wfs = world_frames(shots, bridges)
    for s, wf in enumerate(wfs):
        log(f'shot {s + 1}: floor support {wf["floor_support"]}, camera height {wf["cam_h"]:.3f} (shot units), '
            f'shares the world of shot {wf["group"] + 1}, scale {wf["scale"]:.3f}, check {wf.get("check")}')
    scene = os.path.join(site, 'assets', 'scene')
    os.makedirs(os.path.join(scene, 'hd'), exist_ok=True)
    os.makedirs(os.path.join(scene, 'sd'), exist_ok=True)
    os.makedirs(os.path.join(scene, 'depth'), exist_ok=True)
    diag = os.path.join(r2, 'export')
    os.makedirs(diag, exist_ok=True)

    small = {}

    def frame_small(fi):
        if fi not in small:
            im = cv2.imread(os.path.join(r2, 'frames', f'{int(round(times[fi] * 24)):04d}.png'))
            small[fi] = cv2.resize(im, (DW, DH), interpolation=cv2.INTER_AREA)
        return small[fi]

    manifest = dict(version=1, source='RVR1.mp4', video=dict(duration=46.54, fps=24, width=1920, height=1080),
                    grid=[GW, GH], images=dict(hd=list(HD), sd=list(SD)), shots=[], keyframes=[], path=[],
                    transitions=[], areas=[dict(name=n, t=t) for n, t in AREAS])
    kf_frames = {}  # video frame number -> keyframe name
    for s, sd in enumerate(shots):
        wf = wfs[s]
        frames = sd['frames']
        f = float(sd['f'])
        for _ in range(2):  # twice: a run of bad frames can hide the last one from the first pass
            sd['R'], sd['t'] = repair_poses(times[frames], sd['R'], sd['t'], HIDDEN_CUTS.get(s, ()), log)
        K = np.array([[f * DW / 960, 0, DW / 2], [0, f * DW / 960, DH / 2], [0, 0, 1]])
        P = sd['pts']
        link = sd['link']
        # path: every registered frame
        shot_path = []
        for k, fi in enumerate(frames):
            Rn, Cw = cam_world(wf, sd['R'][k], sd['t'][k])
            shot_path.append(dict(shot=s, t=round(float(times[fi]), 4), p=[round(float(v), 5) for v in Cw],
                                  q=[round(float(v), 6) for v in three_quat(Rn)]))
        keep = pick_keyframes(times, frames, HIDDEN_CUTS.get(s, ()) if s == 0 else ())
        depth_grids = []
        shot_kfs = []
        static = None
        if s == 0:
            ts_ = times[frames]
            in_static = [k for k in keep if STATIC[0] - 1e-3 <= ts_[k] <= STATIC[1]]
            static = dict(keys=set(in_static), closed=in_static[0], open=in_static[-1])

        def sparse_for(kk):
            fi = frames[kk]
            R, t = sd['R'][kk], sd['t'][kk]
            obs_pts = np.unique(link[link[:, 0] == fi][:, 2])
            Xc = P[obs_pts] @ R.T + t
            uv = Xc[:, :2] / Xc[:, 2:] * K[0, 0] + K[:2, 2]
            ok = (Xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < DW) & (uv[:, 1] >= 0) & (uv[:, 1] < DH)
            return np.column_stack([uv[ok], Xc[ok, 2]])

        # optional floor limit (off, see FLOOR_SHOTS)
        floor = (wf['floor_n'], wf['floor_p']) if s in FLOOR_SHOTS else None

        def mvs(kk, nb):
            d, c = keyframe_depth(frame_small(frames[kk]), [frame_small(frames[j]) for j in nb], K,
                                  (sd['R'][kk], sd['t'][kk]), [(sd['R'][j], sd['t'][j]) for j in nb], sparse_for(kk),
                                  floor=floor)
            # remove small spikes (mostly reflections) without moving real depth edges
            d = 1 / cv2.medianBlur((1 / d).astype(np.float32), 5)
            return d, c

        if static and (only is None or any(len(manifest['keyframes']) + i in only for i, k in enumerate(keep) if k in static['keys'])):
            ts_ = times[frames]
            # closed shutter: seen while the camera approaches (frames from 4 s until it stops)
            nb_c = [j for j in range(len(frames)) if 4.0 - 1e-3 <= ts_[j] < STATIC[0] - 1e-3]
            # open door: the gallery seen from the door and from just inside it (after the hidden cut)
            nb_o = [j for j in range(len(frames)) if STATIC[1] < ts_[j] <= 17.25]
            static['D_closed'], _ = mvs(static['closed'], nb_c)
            static['D_open'], _ = mvs(static['open'], nb_o)
            ordered = sorted(static['keys'])
            static['rect'], edges = shutter_edges([frame_small(frames[k]) for k in ordered],
                                                  frame_small(frames[static['closed']]), frame_small(frames[static['open']]))
            static['edge'] = dict(zip(ordered, edges))
            log(f'  static period: door {static["rect"]}, shutter edge rows {edges}')

        for kk in keep:
            fi = frames[kk]
            R, t = sd['R'][kk], sd['t'][kk]
            name = f'k{len(manifest["keyframes"]) + len(shot_kfs):03d}.webp'
            sparse = sparse_for(kk)
            npy = os.path.join(diag, f'{name[:-5]}_depth.npy')
            if REUSE and os.path.exists(npy):
                d = np.load(npy)
            elif only is None or len(manifest['keyframes']) + len(shot_kfs) in only:
                if static and kk in static['keys']:
                    # camera stands still: shutter part from the closed-door depth, the opening (below the
                    # shutter's bottom edge, inside the door) from the open-door depth
                    x0, x1, y0, y1 = static['rect']
                    d = static['D_closed'].copy()
                    ye = static['edge'][kk]
                    d[ye:y1 + 1, x0:x1 + 1] = static['D_open'][ye:y1 + 1, x0:x1 + 1]
                    conf = np.zeros_like(d)
                    conf[ye:y1 + 1, x0:x1 + 1] = 1
                else:
                    # neighbours: up to 4 registered frames on each side
                    nb = [j for j in range(kk - 4, kk + 5) if j != kk and 0 <= j < len(frames)]
                    d, conf = mvs(kk, nb)
                np.save(os.path.join(diag, f'{name[:-5]}_depth.npy'), d)
                # diagnostic picture: frame | depth | confident pixels
                inv = 1 / d
                lo, hi = np.percentile(inv, 1), np.percentile(inv, 99)
                vis = cv2.applyColorMap((np.clip((inv - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                cv2.imwrite(os.path.join(diag, f'{name[:-5]}.jpg'),
                            np.hstack([frame_small(fi), vis, cv2.cvtColor((conf * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
            elif os.path.exists(os.path.join(diag, f'{name[:-5]}_depth.npy')):
                d = np.load(os.path.join(diag, f'{name[:-5]}_depth.npy'))
            else:
                shot_kfs.append(None)  # test run: keyframe skipped
                continue
            d_world = d * wf['scale']
            g = grid_from_depth(d_world)
            e = edge_flags(g)
            i0, i1 = float(g.min()), float(g.max())
            q = np.round((g - i0) / max(i1 - i0, 1e-9) * 32767).astype(np.uint16)
            depth_grids.append((q << 1) | e.astype(np.uint16))
            Rn, Cw = cam_world(wf, R, t)
            shot_kfs.append(dict(shot=s, t=round(float(times[fi]), 4), image=name, inv=[i0, i1],
                                 M=[round(float(v), 7) for v in Rn.ravel()],  # row-major R = column-major R^T
                                 C=[round(float(v), 5) for v in Cw],
                                 tan=[round(960 / 2 / f, 6), round(540 / 2 / f, 6)],
                                 dmed=round(float(np.median(d_world)), 4),
                                 par=round(parallax_budget(d_world, f), 5)))
            kf_frames[int(round(times[fi] * 24))] = name
            log(f'shot {s + 1} keyframe {name} t={times[fi]:.2f}s: {len(sparse)} SfM points, edges {e.mean() * 100:.1f}%')
        if only is not None:
            manifest['keyframes'] += shot_kfs
            continue
        # one website "shot" per part between camera jumps
        bounds = [-1e9] + JUMPS.get(s, []) + [1e9]
        for part in range(len(bounds) - 1):
            lo, hi = bounds[part], bounds[part + 1]
            m_idx = len(manifest['shots'])
            sel = [i for i, k in enumerate(shot_kfs) if lo <= k['t'] < hi]
            for i in sel:
                shot_kfs[i]['shot'] = m_idx
            pth = [dict(p, shot=m_idx) for p in shot_path if lo <= p['t'] < hi]
            # one small depth file per keyframe, so each keyframe loads on its own (image + depth)
            for i in sel:
                dname = shot_kfs[i]['image'].replace('.webp', '.bin')
                depth_grids[i].ravel().astype('<u2').tofile(os.path.join(scene, 'depth', dname))
                shot_kfs[i]['depth'] = dname
            manifest['keyframes'] += [shot_kfs[i] for i in sel]
            manifest['path'] += pth
            if m_idx > 0:
                fly = part == 0 and wfs[s]['group'] == wfs[s - 1]['group']
                manifest['transitions'].append(dict(type='fly' if fly else 'dissolve', length=0.9 if fly else 0.6))
            manifest['shots'].append(dict(name=SHOT_NAMES[m_idx], start=pth[0]['t'], end=pth[-1]['t'],
                                          world=int(wf['group']), source_shot=s + 1))
    if only is not None:
        return
    with open(os.path.join(scene, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, separators=(',', ':'))
    log(f'manifest: {len(manifest["keyframes"])} keyframes, {len(manifest["path"])} path samples')

    if images and all(os.path.exists(os.path.join(scene, 'hd', n)) for n in kf_frames.values()) and '--redo-images' not in sys.argv:
        log('images already there (same keyframes): kept')
        images = False
    if images:
        cap = cv2.VideoCapture(video)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in kf_frames:
                rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                rgb.resize(HD, Image.LANCZOS).save(os.path.join(scene, 'hd', kf_frames[i]), 'WEBP', quality=80, method=6)
                rgb.resize(SD, Image.LANCZOS).save(os.path.join(scene, 'sd', kf_frames[i]), 'WEBP', quality=78, method=6)
            i += 1
        log('images written')


if __name__ == '__main__':
    a = sys.argv[1:]
    only = None
    REUSE = '--reuse-depth' in a
    if '--only-depth' in a:
        only = {int(x) for x in a[a.index('--only-depth') + 1].split(',')}
    main(a[0], a[1], a[2], only, '--no-images' not in a)
