"""Measures how well a keyframe's depth predicts the neighbouring real frames (view synthesis test).

For a keyframe, each depth variant is used to reproject the keyframe image into the camera of other
frames of the same shot (forward splatting with a z-buffer). The error against the real frame shows
how good the depth is for what the website does (moving the camera away from the keyframe).

Usage: python3 eval_depth.py <r2> <shot> <local frame index> [variants]
"""
import os
import sys
import time

import cv2
import numpy as np

import depth as D
from mvs import plane_sweep

r2 = sys.argv[1]
s = int(sys.argv[2])
kk = int(sys.argv[3])
variants = sys.argv[4].split(',') if len(sys.argv) > 4 else ['mvs4', 'mvs8', 'sparse', 'plane']
f_ = np.load(os.path.join(r2, 'feats.npz'))
times = f_['times']
sd = dict(np.load(os.path.join(r2, f'shot{s}.npz')))
frames = sd['frames']
f = float(sd['f'])
DW, DH = D.DW, D.DH
K = np.array([[f * DW / 960, 0, DW / 2], [0, f * DW / 960, DH / 2], [0, 0, 1]])


def img(fi):
    im = cv2.imread(os.path.join(r2, 'frames', f'{int(round(times[fi] * 24)):04d}.png'))
    return cv2.resize(im, (DW, DH), interpolation=cv2.INTER_AREA)


def pose(k):
    return sd['R'][k], sd['t'][k]


fi = frames[kk]
R, t = pose(kk)
link = sd['link']
obs_pts = np.unique(link[link[:, 0] == fi][:, 2])
Xc = sd['pts'][obs_pts] @ R.T + t
uv = Xc[:, :2] / Xc[:, 2:] * K[0, 0] + K[:2, 2]
ok = (Xc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < DW) & (uv[:, 1] >= 0) & (uv[:, 1] < DH)
sparse = np.column_stack([uv[ok], Xc[ok, 2]])
Cs = np.array([-Rr.T @ tt for Rr, tt in zip(sd['R'], sd['t'])])
n_fl, p_fl, _ = D.floor_plane(sd['pts'], list(sd['R']), list(Cs))
ref = img(fi)


def warp(depth, k_to):
    """Forward-splat the keyframe into frame k_to's camera. Returns image and validity mask."""
    Rt, tt = pose(k_to)
    ys, xs = np.mgrid[0:DH, 0:DW].astype(np.float64)
    Kinv = np.linalg.inv(K)
    P = (np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1) @ Kinv.T) * depth[..., None]
    Pw = (P - t) @ R  # R^T (P - t)
    Pt = Pw @ Rt.T + tt
    z = Pt[..., 2]
    u = Pt[..., 0] / z * K[0, 0] + K[0, 2]
    v = Pt[..., 1] / z * K[1, 1] + K[1, 2]
    out = np.zeros_like(ref)
    zb = np.full((DH, DW), np.inf)
    order = np.argsort(-z.ravel())  # far first, near overwrite
    uu = u.ravel()[order]
    vv = v.ravel()[order]
    cols = ref.reshape(-1, 3)[order]
    zz = z.ravel()[order]
    for dx in (0, 1):
        for dy in (0, 1):
            xi = np.floor(uu - 0.5).astype(int) + dx
            yi = np.floor(vv - 0.5).astype(int) + dy
            m = (zz > 0) & (xi >= 0) & (xi < DW) & (yi >= 0) & (yi < DH)
            out[yi[m], xi[m]] = cols[m]
            zb[yi[m], xi[m]] = np.minimum(zb[yi[m], xi[m]], zz[m])
    return out, np.isfinite(zb)


def score(depth, label):
    errs = []
    for d in (-4, -2, 2, 4):
        k2 = kk + d
        if not 0 <= k2 < len(frames):
            continue
        tgt = img(frames[k2]).astype(np.float32)
        w, m = warp(depth, k2)
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3))) > 0
        m[:12] = m[-12:] = False
        m[:, :12] = m[:, -12:] = False
        e = np.abs(w.astype(np.float32) - tgt).mean(-1)[m].mean()
        e0 = np.abs(ref.astype(np.float32) - tgt).mean(-1)[m].mean()  # no reprojection at all
        errs.append((d, round(float(e), 2), round(float(e0), 2)))
    print(f'{label:8s}', ' '.join(f'{d:+d}:{e:5.2f}/{e0:5.2f}' for d, e, e0 in errs), ' mean', round(np.mean([e for _, e, _ in errs]), 2))
    return w


def run_mvs(span, floor=True, median=False):
    nb = [j for j in range(kk - span, kk + span + 1) if j != kk and 0 <= j < len(frames)]
    d, conf = D.keyframe_depth(ref, [img(frames[j]) for j in nb], K, (R, t), [pose(j) for j in nb], sparse,
                               floor=(n_fl, p_fl) if floor else None)
    if median:
        d = 1 / cv2.medianBlur((1 / d).astype(np.float32), 5)
    return d


T = time.time()
res = {}
for v in variants:
    if v.startswith('mvs'):
        res[v] = run_mvs(4, 'F' in v, 'M' in v)
    elif v.startswith('sparse'):
        inv_obs = np.zeros((DH, DW), np.float32)
        w = np.zeros((DH, DW), np.float32)
        for u_, v_, z_ in sparse:
            x, y = int(u_), int(v_)
            inv_obs[y, x] = 1 / z_
            w[y, x] = 4.0
        lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
        inv_f = D.edge_aware_fill(inv_obs, w, lab)
        dd = 1 / np.maximum(inv_f, 1e-6)
        fl = D.floor_depth_map(K, R, t, n_fl, p_fl, DW, DH)
        res[v] = dd.astype(np.float32) if v.endswith('nf') else np.where(np.isfinite(fl) & (dd > fl), fl, dd).astype(np.float32)
    elif v == 'plane':
        res[v] = np.full((DH, DW), np.median(sparse[:, 2]), np.float32)
    score(res[v], v)
np.save(f'/tmp/claude-0/-home-claude/86ab42b2-768f-56d4-a69e-f0df51fbfa5c/scratchpad/recon/eval_{s}_{kk}.npy', np.stack([res[v] for v in variants]))
print('time', round(time.time() - T, 1))
