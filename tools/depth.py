"""Dense depth for keyframes of a reconstructed shot.

For each keyframe:
  1. plane-sweep stereo (ZNCC) against its neighbours in the same shot, at 480x270;
  2. keep only confident pixels (clear cost minimum, enough parallax);
  3. add the SfM points seen in that frame (most reliable);
  4. edge-aware fill: solve a confidence-weighted, image-guided smoothness problem in inverse depth,
     so depth edges follow image edges;
  5. pixels on or 'below' the floor (mirror reflections in the glossy floor) are put on the floor plane.

Camera convention: x_cam = R X + t (OpenCV), focal f at 960 px width, principal point at the centre.
"""
import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from mvs import plane_sweep

DW, DH = 480, 270


def floor_plane(pts, R_list, C_list):
    """RANSAC plane among SfM points: normal close to 'up', below the cameras, most support."""
    up = -np.mean([R[1] for R in R_list], axis=0)  # camera -y axis in world coords (y is down in OpenCV)
    up /= np.linalg.norm(up)
    cmean = np.mean(C_list, axis=0)
    h = (pts - cmean) @ up
    cand = pts[h < 0]
    scale = np.median(np.abs(h))
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(4000):
        s3 = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ up < 0:
            n = -n
        if n @ up < 0.95:
            continue
        cnt = np.sum(np.abs((cand - s3[0]) @ n) < 0.01 * scale)
        if cnt > best[0]:
            best = (cnt, (n, s3[0]))
    n, p0 = best[1]
    inl = cand[np.abs((cand - p0) @ n) < 0.01 * scale]
    # refine with least squares (SVD) on inliers
    c = inl.mean(0)
    n2 = np.linalg.svd(inl - c)[2][-1]
    if n2 @ up < 0:
        n2 = -n2
    return n2, c, len(inl)


def floor_depth_map(K, R, t, n, p0, W, H):
    """Camera-z depth where each pixel's ray meets the plane n.(X - p0) = 0; inf if it doesn't (above horizon)."""
    Kinv = np.linalg.inv(K)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    rays_c = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs)], -1) @ Kinv.T  # z = 1
    rays_w = rays_c @ R  # R^T d
    C = -R.T @ t
    denom = rays_w @ n
    s = ((p0 - C) @ n) / np.where(np.abs(denom) > 1e-9, denom, np.nan)
    return np.where(np.isfinite(s) & (s > 0), s, np.inf).astype(np.float32)


def edge_aware_fill(inv_obs, w_obs, img_lab, lam=40.0, sigma=6.0):
    """argmin_x sum w_obs (x - inv_obs)^2 + lam * sum_edges w_ij (x_i - x_j)^2, w_ij = exp(-|dI|^2 / sigma^2)."""
    H, W = inv_obs.shape
    N = H * W
    idx = np.arange(N).reshape(H, W)
    rows, cols, vals = [], [], []
    for (a, b, da) in ((idx[:, :-1], idx[:, 1:], img_lab[:, :-1] - img_lab[:, 1:]),
                       (idx[:-1, :], idx[1:, :], img_lab[:-1, :] - img_lab[1:, :])):
        w = lam * np.exp(-np.sum(da.astype(np.float32) ** 2, -1) / sigma ** 2) + 1e-4
        a, b, w = a.ravel(), b.ravel(), w.ravel()
        rows += [a, b, a, b]
        cols += [a, b, b, a]
        vals += [w, w, -w, -w]
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    A = L + sp.diags(w_obs.ravel() + 1e-6)
    b = (w_obs * inv_obs).ravel()
    # initial guess: pyramid push-pull of the observations
    x0 = push_pull(inv_obs, w_obs).ravel()
    M = sp.diags(1.0 / A.diagonal())
    x, info = spl.cg(A, b, x0=x0, M=M, rtol=1e-6, maxiter=800)
    return x.reshape(H, W).astype(np.float32)


def push_pull(v, w):
    levels = []
    cv_, cw = (v * w).astype(np.float32), w.astype(np.float32)
    while min(cw.shape) > 4:
        levels.append((cv_, cw))
        cv_, cw = cv2.pyrDown(cv_), cv2.pyrDown(cw)
    est = np.full(cw.shape, (cv_.sum() / max(cw.sum(), 1e-6)), np.float32)
    for lv, lw in reversed(levels):
        up = cv2.resize(est, (lv.shape[1], lv.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(lw * 4, 0, 1)
        est = a * (lv / np.maximum(lw, 1e-6)) + (1 - a) * up
    return est


def keyframe_depth(ref_bgr, nbr_bgr, K, pose_ref, nbr_poses, sparse_uvz, floor=None, planes=128):
    """ref_bgr/nbr_bgr at DW x DH. sparse_uvz: (N,3) pixel coords (at DW x DH) + camera depth of SfM
    points observed in this frame. floor: (n, p0) or None. Returns depth (DH, DW) and a confidence map."""
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    z = sparse_uvz[:, 2]
    dmin, dmax = np.percentile(z, 1) * 0.75, np.percentile(z, 99) * 1.4
    depth, cost, par, dist = plane_sweep(gray(ref_bgr), [gray(b) for b in nbr_bgr], K, pose_ref[0], pose_ref[1],
                                         nbr_poses, dmin, dmax, planes=planes, win=3)
    conf = ((cost < 0.35) & (dist > 0.06) & (par > 4)).astype(np.float32)
    # remove isolated confident speckles: keep where the local median agrees
    inv = 1.0 / depth
    med = cv2.medianBlur(inv.astype(np.float32), 5)
    conf *= (np.abs(inv - med) < 0.15 * med + 1e-4)
    w = conf * np.clip((0.35 - cost) / 0.2, 0.2, 1.0)
    inv_obs = np.where(conf > 0, inv, 0)
    # SfM points: strong weight, splat 3x3
    for u, v, zz in sparse_uvz:
        x, y = int(u), int(v)
        if 1 <= x < DW - 1 and 1 <= y < DH - 1:
            inv_obs[y - 1:y + 2, x - 1:x + 2] = 1 / zz
            w[y - 1:y + 2, x - 1:x + 2] = 4.0
    fl = None
    if floor is not None:
        fl = floor_depth_map(K, pose_ref[0], pose_ref[1], floor[0], floor[1], DW, DH)
        # an observation that lies behind the floor surface is a reflection: replace it by the floor
        below = np.isfinite(fl) & (inv_obs > 0) & (1 / np.maximum(inv_obs, 1e-9) > fl * 0.97)
        inv_obs[below] = 1 / fl[below]
    lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    inv_f = edge_aware_fill(inv_obs, w, lab)
    inv_f = np.clip(inv_f, 1 / (dmax * 2), 1 / (dmin * 0.5))
    d = 1 / inv_f
    if fl is not None:
        d = np.where(np.isfinite(fl) & (d > fl), fl, d)  # nothing ends up below the floor
    return d.astype(np.float32), conf
