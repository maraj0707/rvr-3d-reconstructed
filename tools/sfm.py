"""Incremental structure-from-motion for one continuous shot of RVR1.mp4 (OpenCV + SciPy, CPU only).

Recovers camera poses (the real camera path), a shared focal length and a sparse 3D point cloud,
then reports how geometrically consistent the footage is (reprojection error after bundle adjustment).

Usage: python3 sfm.py <video> <t_start> <t_end> <fps> <out.npz> [width]
"""
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import os
FIX_F = 'RVR_FIX_F' in os.environ


def extract(video, t0, t1, fps, width):
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frames, times = [], []
    step = src_fps / fps
    k = 0
    while True:
        t = t0 + k / fps
        if t > t1:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * src_fps))
        ok, img = cap.read()
        if not ok:
            break
        h = int(img.shape[0] * width / img.shape[1])
        frames.append(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))
        times.append(t)
        k += 1
    return frames, times


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def project(pts, rvec, t, f, c):
    P, _ = cv2.projectPoints(pts, rvec, t, np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]]), None)
    return P.reshape(-1, 2)


def bundle_adjust(cams, pts3, obs, f, c, fix_first=True, iters=60):
    """cams: (N,6) rvec|t ; pts3: (M,3) ; obs: list of (cam_idx, pt_idx, x, y)."""
    obs = np.asarray(obs)
    ci, pi = obs[:, 0].astype(int), obs[:, 1].astype(int)
    uv = obs[:, 2:4]
    N, M = len(cams), len(pts3)

    def unpack(x):
        return x[:N * 6].reshape(N, 6), x[N * 6:N * 6 + M * 3].reshape(M, 3), x[-1]

    def resid(x):
        cm, P, ff = unpack(x)
        R = np.stack([cv2.Rodrigues(r)[0] for r in cm[:, :3]])
        Xc = np.einsum('nij,nj->ni', R[ci], P[pi]) + cm[ci, 3:]
        z = np.maximum(Xc[:, 2], 1e-6)
        proj = ff * Xc[:, :2] / z[:, None] + c
        r = (proj - uv).ravel()
        if fix_first:
            r = np.concatenate([r, (cm[0] - cams[0]) * 1e3])
        if FIX_F:
            r = np.concatenate([r, [(ff - f) * 1e3]])
        return r

    m = len(obs) * 2 + (6 if fix_first else 0) + (1 if FIX_F else 0)
    S = lil_matrix((m, N * 6 + M * 3 + 1), dtype=int)
    rows = np.arange(len(obs))
    for k in range(6):
        S[2 * rows, ci * 6 + k] = 1
        S[2 * rows + 1, ci * 6 + k] = 1
    for k in range(3):
        S[2 * rows, N * 6 + pi * 3 + k] = 1
        S[2 * rows + 1, N * 6 + pi * 3 + k] = 1
    S[:2 * len(obs), -1] = 1
    if FIX_F:
        S[m - 1, -1] = 1
    if fix_first:
        for k in range(6):
            S[2 * len(obs) + k, k] = 1
    x0 = np.concatenate([cams.ravel(), pts3.ravel(), [f]])
    res = least_squares(resid, x0, jac_sparsity=S, loss='huber', f_scale=2.0, x_scale='jac',
                        max_nfev=iters, method='trf', tr_solver='lsmr')
    cm, P, ff = unpack(res.x)
    return cm, P, ff


def run(video, t0, t1, fps, out, width=960):
    T = time.time()
    frames, times = extract(video, t0, t1, fps, width)
    n = len(frames)
    H, W = frames[0].shape[:2]
    c = np.array([W / 2, H / 2])
    import os
    f = float(os.environ.get('RVR_FIX_F', 1.0 * W))  # initial guess, refined by bundle adjustment
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    kps, des = [], []
    for img in frames:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)  # the footage is dark: boost local contrast for features
        k, d = sift.detectAndCompute(g, None)
        kps.append(np.array([p.pt for p in k], np.float32))
        des.append(d)
    print(f'{n} frames {W}x{H}, avg {np.mean([len(k) for k in kps]):.0f} features ({time.time() - T:.0f}s)')

    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 64})
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
    pair_inl = {}
    uf = UF()
    WIN = 6
    for i in range(n):
        for j in range(i + 1, min(n, i + WIN + 1)):
            m = flann.knnMatch(des[i], des[j], k=2)
            good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
            if len(good) < 40:
                continue
            p1 = kps[i][[g.queryIdx for g in good]]
            p2 = kps[j][[g.trainIdx for g in good]]
            E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0)
            if E is None:
                continue
            inl = mask.ravel().astype(bool)
            pair_inl[(i, j)] = (len(good), int(inl.sum()))
            for g, ok in zip(good, inl):
                if ok:
                    uf.union((i, g.queryIdx), (j, g.trainIdx))
    ratios = [b / a for a, b in pair_inl.values()]
    print(f'pairs {len(pair_inl)}, median E-inlier ratio {np.median(ratios):.2f} ({time.time() - T:.0f}s)')

    # build tracks (one observation per frame)
    groups = {}
    for node in list(uf.p.keys()):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = []
    for g in groups.values():
        fr = [a for a, _ in g]
        if len(g) >= 3 and len(set(fr)) == len(fr):
            tracks.append(sorted(g))
    print(f'tracks >=3 views: {len(tracks)}, mean length {np.mean([len(t) for t in tracks]):.1f}')
    obs_by_frame = [dict() for _ in range(n)]  # frame -> {track: kp}
    for ti, tr in enumerate(tracks):
        for fi, kp in tr:
            obs_by_frame[fi][ti] = kp

    # initial pair: most shared tracks with enough baseline
    best = None
    for (i, j), (_, ninl) in pair_inl.items():
        if j - i < 3:
            continue
        shared = set(obs_by_frame[i]) & set(obs_by_frame[j])
        if len(shared) < 80:
            continue
        p1 = np.array([kps[i][obs_by_frame[i][t]] for t in shared])
        p2 = np.array([kps[j][obs_by_frame[j][t]] for t in shared])
        par = np.median(np.linalg.norm(p1 - p2, axis=1))
        score = len(shared) * min(par, 40)
        if best is None or score > best[0]:
            best = (score, i, j, list(shared), p1, p2)
    _, i0, j0, shared, p1, p2 = best
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC, threshold=1.0)
    _, R, t, mask2 = cv2.recoverPose(E, p1, p2, K, mask=mask)
    poses = {i0: (np.zeros(3), np.zeros(3)), j0: (cv2.Rodrigues(R)[0].ravel(), t.ravel())}
    pts3 = {}

    def Pmat(fi):
        r, tt = poses[fi]
        return K @ np.hstack([cv2.Rodrigues(r)[0], tt.reshape(3, 1)])

    def triangulate(fa, fb, cand):
        if not cand:
            return
        a = np.array([kps[fa][obs_by_frame[fa][t]] for t in cand]).T
        b = np.array([kps[fb][obs_by_frame[fb][t]] for t in cand]).T
        X = cv2.triangulatePoints(Pmat(fa), Pmat(fb), a, b)
        X = (X[:3] / X[3]).T
        for t, x in zip(cand, X):
            ok = True
            for fi in (fa, fb):
                r, tt = poses[fi]
                xc = cv2.Rodrigues(r)[0] @ x + tt
                if xc[2] <= 0:
                    ok = False
            if ok and np.all(np.isfinite(x)):
                pts3[t] = x

    triangulate(i0, j0, shared)
    print(f'init pair {i0}-{j0}: {len(pts3)} points')

    # incremental registration
    while True:
        cand = []
        for fi in range(n):
            if fi in poses:
                continue
            common = [t for t in obs_by_frame[fi] if t in pts3]
            cand.append((len(common), fi, common))
        cand = [x for x in cand if x[0] >= 25]
        if not cand:
            break
        _, fi, common = max(cand)
        X = np.array([pts3[t] for t in common])
        x = np.array([kps[fi][obs_by_frame[fi][t]] for t in common])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, reprojectionError=3.0, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 20:
            obs_by_frame[fi] = {}
            continue
        rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], x[inl.ravel()], K, None, rvec, tvec)
        poses[fi] = (rvec.ravel(), tvec.ravel())
        # triangulate new tracks with the nearest registered frame
        for fj in sorted(poses, key=lambda k: abs(k - fi)):
            if fj == fi:
                continue
            new = [t for t in obs_by_frame[fi] if t not in pts3 and t in obs_by_frame[fj]]
            triangulate(fi, fj, new)
            if abs(fj - fi) > 4:
                break

    # global bundle adjustment (focal length refined too)
    order = sorted(poses)
    cam_index = {fi: k for k, fi in enumerate(order)}
    pids = sorted(pts3)
    pt_index = {t: k for k, t in enumerate(pids)}
    obs = []
    for fi in order:
        for t, kp in obs_by_frame[fi].items():
            if t in pt_index:
                u, v = kps[fi][kp]
                obs.append((cam_index[fi], pt_index[t], u, v))
    cams = np.array([np.concatenate(poses[fi]) for fi in order])
    P3 = np.array([pts3[t] for t in pids])

    def rmse(cm, P, ff):
        o = np.asarray(obs)
        errs = []
        for k in range(len(cm)):
            sel = o[:, 0] == k
            if sel.any():
                pr = project(P[o[sel, 1].astype(int)], cm[k, :3], cm[k, 3:], ff, c)
                errs.append(np.linalg.norm(pr - o[sel, 2:4], axis=1))
        e = np.concatenate(errs)
        return np.sqrt(np.mean(e ** 2)), np.median(e), e

    r0 = rmse(cams, P3, f)
    print(f'before BA: {len(order)}/{n} cams, {len(P3)} pts, {len(obs)} obs, reproj RMSE {r0[0]:.2f}px median {r0[1]:.2f}px')
    for rnd in range(2):
        cams, P3, f = bundle_adjust(cams, P3, obs, f, c)
        _, _, e = rmse(cams, P3, f)
        # drop outlier observations, then refine again
        thr = max(2.5, np.percentile(e, 90))
        keep = e < thr * 2
        obs = [o for o, k in zip(obs, keep) if k]
    r1 = rmse(cams, P3, f)
    hfov = np.degrees(2 * np.arctan(W / 2 / f))
    print(f'after BA: reproj RMSE {r1[0]:.2f}px, median {r1[1]:.2f}px, 95th pct {np.percentile(r1[2], 95):.2f}px, focal {f:.0f}px (hFOV {hfov:.1f} deg)')

    centers = np.array([-cv2.Rodrigues(cm[:3])[0].T @ cm[3:] for cm in cams])
    path_len = np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))
    print(f'camera path length {path_len:.2f} (scene units), registered {len(order)}/{n} frames, total {time.time() - T:.0f}s')
    # point colours
    cols = []
    o = np.asarray(obs)
    for k in range(len(P3)):
        sel = np.where(o[:, 1] == k)[0]
        if len(sel):
            cam, _, u, v = o[sel[0]]
            img = frames[order[int(cam)]]
            cols.append(img[int(min(v, H - 1)), int(min(u, W - 1))][::-1])
        else:
            cols.append([128, 128, 128])
    np.savez(out, cams=cams, order=np.array(order), times=np.array(times), pts=P3, cols=np.array(cols, np.uint8),
             f=f, c=c, W=W, H=H, rmse=r1[0], median=r1[1])


if __name__ == '__main__':
    a = sys.argv
    run(a[1], float(a[2]), float(a[3]), float(a[4]), a[5], int(a[6]) if len(a) > 6 else 960)
