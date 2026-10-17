"""Plane-sweep multi-view stereo: dense depth for one reference frame from its neighbours (known poses).

All functions work in a camera convention x_cam = R @ X_world + t  (OpenCV), pixels with focal f and
principal point c at the working resolution.
"""
import cv2
import numpy as np


def _box(img, r):
    return cv2.boxFilter(img, cv2.CV_32F, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def plane_sweep(ref, nbrs, K, Rr, tr, nbr_poses, dmin, dmax, planes=96, win=4):
    """ref: gray float32 HxW; nbrs: list of gray float32 images; nbr_poses: list of (R, t).
    Returns depth (H,W), cost (H,W) (lower = better match, 0..2), and 'parallax' (px of disparity range)."""
    H, W = ref.shape
    Kinv = np.linalg.inv(K)
    inv = np.linspace(1.0 / dmin, 1.0 / dmax, planes)  # sample uniformly in inverse depth
    mu_r = _box(ref, win)
    sd_r = np.sqrt(np.maximum(_box(ref * ref, win) - mu_r ** 2, 1e-6))
    costs = np.full((planes, H, W), 2.0, np.float32)
    per_nbr = []
    n_vec = np.array([0, 0, 1.0])
    rel = []
    for (Rn, tn) in nbr_poses:
        R_rel = Rn @ Rr.T
        t_rel = tn - R_rel @ tr
        rel.append((R_rel, t_rel))
    for k, (img, (R_rel, t_rel)) in enumerate(zip(nbrs, rel)):
        c_k = np.empty((planes, H, W), np.float32)
        for p, idep in enumerate(inv):
            Hm = K @ (R_rel + np.outer(t_rel, n_vec) * idep) @ Kinv
            warped = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            valid = warped >= 0
            w = np.where(valid, warped, mu_r)
            mu_w = _box(w, win)
            sd_w = np.sqrt(np.maximum(_box(w * w, win) - mu_w ** 2, 1e-6))
            cov = _box(ref * w, win) - mu_r * mu_w
            zncc = cov / (sd_r * sd_w)
            c_k[p] = np.where(_box(valid.astype(np.float32), win) > 0.9, 1 - zncc, 2.0)
        per_nbr.append(c_k)
    stack = np.stack(per_nbr)  # (N, P, H, W)
    stack.sort(axis=0)
    best_k = max(1, int(np.ceil(len(nbrs) / 2)))
    costs = stack[:best_k].mean(axis=0)  # average of the best half: robust to occlusion
    idx = costs.argmin(axis=0)
    cmin = np.take_along_axis(costs, idx[None], 0)[0]
    # sub-plane refinement (parabola)
    i0 = np.clip(idx, 1, planes - 2)
    cm = np.take_along_axis(costs, (i0 - 1)[None], 0)[0]
    c0 = np.take_along_axis(costs, i0[None], 0)[0]
    cp = np.take_along_axis(costs, (i0 + 1)[None], 0)[0]
    denom = cm - 2 * c0 + cp
    off = np.where(np.abs(denom) > 1e-6, 0.5 * (cm - cp) / denom, 0)
    fidx = np.clip(i0 + np.clip(off, -0.5, 0.5), 0, planes - 1)
    idep = np.interp(fidx, np.arange(planes), inv)
    depth = 1.0 / idep
    # how many pixels the near and far plane differ by in the neighbours: low = depth is unobservable here
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ Kinv.T
    span = np.zeros((H, W), np.float32)
    for R_rel, t_rel in rel:
        pts = []
        for idep_ in (inv[0], inv[-1]):
            X = rays / idep_
            Xn = X @ R_rel.T + t_rel
            pts.append(Xn[..., :2] / np.maximum(Xn[..., 2:3], 1e-6) * K[0, 0])
        span = np.maximum(span, np.linalg.norm(pts[0] - pts[1], axis=-1))
    # distinctiveness: how much better the best plane is than planes far from it
    far = np.abs(np.arange(planes)[:, None, None] - idx[None]) > planes // 12
    second = np.where(far, costs, 9).min(axis=0)
    return depth.astype(np.float32), cmin.astype(np.float32), span, (second - cmin).astype(np.float32)


def floor_depth(K, R, t, H, W, floor_y=0.0):
    """Depth (camera z) at which each pixel's ray hits the floor plane y = floor_y (inf if it doesn't)."""
    Kinv = np.linalg.inv(K)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d_cam = np.stack([xs, ys, np.ones_like(xs)], -1) @ Kinv.T  # z = 1 rays
    d_world = d_cam @ R  # R^T applied to each row vector
    C = -R.T @ t
    s = (floor_y - C[1]) / np.where(np.abs(d_world[..., 1]) > 1e-6, d_world[..., 1], np.nan)
    return np.where(s > 0, s, np.inf).astype(np.float32)


def fill_depth(depth, valid, guide, iters=6):
    """Edge-aware hole filling in inverse depth: push-pull with joint-bilateral weights from the image."""
    inv = np.where(valid, 1.0 / np.maximum(depth, 1e-6), 0).astype(np.float32)
    w = valid.astype(np.float32)
    g = guide.astype(np.float32)
    # coarse-to-fine pyramid fill
    levels = []
    cur_v, cur_w, cur_g = inv * w, w, g
    while min(cur_w.shape) > 8:
        levels.append((cur_v, cur_w, cur_g))
        cur_v = cv2.pyrDown(cur_v)
        cur_w = cv2.pyrDown(cur_w)
        cur_g = cv2.pyrDown(cur_g)
    est = np.where(cur_w > 1e-4, cur_v / np.maximum(cur_w, 1e-6), np.median(inv[valid]) if valid.any() else 0)
    for v, wt, gg in reversed(levels):
        up = cv2.resize(est, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        known = wt > 0.98
        est = np.where(known, v / np.maximum(wt, 1e-6), up)
        # joint-bilateral smoothing of the filled parts only
        for _ in range(iters):
            sm = cv2.ximgproc.jointBilateralFilter(gg, est, 7, 12, 4) if hasattr(cv2, 'ximgproc') else cv2.GaussianBlur(est, (5, 5), 0)
            est = np.where(known, est, sm)
    return 1.0 / np.maximum(est, 1e-6)
