"""Per-shot structure-from-motion for RVR1.mp4, then links the shots into one coordinate system.

Why per shot: one global incremental solve drifted in scale across the weak cuts (shot 4 -> 5 has only
~20 shared inliers). Each shot on its own is well conditioned, so every shot is solved separately and
the shots are then joined with similarity transforms (rotation + translation + scale) estimated from 3D
points that both shots reconstructed (found through matches across the cut).

Stages (each cached in out_dir):
  features : frames (960 px wide) + SIFT                       -> feats.npz, frames/*.png
  match    : intra-shot pairs + cross-shot pairs                -> matches.npz
  sfm      : per-shot incremental SfM + bundle adjustment       -> shot{k}.npz
  link     : Sim3 between shots, floor plane, metres, export    -> sfm_world.npz

Usage: python3 recon2.py <video> <out_dir> [stage ...]
"""
import os
import sys
import time

import cv2
import numpy as np

from sfm import UF, bundle_adjust

SHOTS = [(0.0, 24.66), (24.75, 28.79), (28.88, 33.54), (33.63, 40.42), (40.50, 46.50)]
FPS = [4, 6, 6, 6, 6]
W_SFM = 960
F0 = 960.0
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ features
def stage_features(video, out):
    cap = cv2.VideoCapture(video)
    os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    times, shot, kp_all, des_all, offs = [], [], [], [], [0]
    for s, (t0, t1) in enumerate(SHOTS):
        fr0, fr1 = int(np.ceil(t0 * 24 - 1e-6)), int(np.floor(t1 * 24 + 1e-6))
        step = 24 / FPS[s]
        frs = sorted({int(round(fr0 + k * step)) for k in range(int((fr1 - fr0) / step) + 1)} | {fr1})
        for fr in frs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            small = cv2.resize(img, (W_SFM, W_SFM * img.shape[0] // img.shape[1]), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out, 'frames', f'{fr:04d}.png'), small)
            k, d = sift.detectAndCompute(clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)), None)
            kp_all.append(np.array([p.pt for p in k], np.float32))
            des_all.append(d)
            offs.append(offs[-1] + len(k))
            times.append(fr / 24)
            shot.append(s)
    np.savez(os.path.join(out, 'feats.npz'), times=np.array(times), shot=np.array(shot), offs=np.array(offs),
             kps=np.concatenate(kp_all), des=np.concatenate(des_all), H=small.shape[0], W=small.shape[1])
    log(f'features: {len(times)} frames')


def load_feats(out):
    d = np.load(os.path.join(out, 'feats.npz'))
    o = d['offs']
    allk, alld = d['kps'], d['des']  # read once (each d[...] access re-reads the whole array)
    kps = [allk[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    des = [alld[o[i]:o[i + 1]] for i in range(len(o) - 1)]
    return d['times'], d['shot'], kps, des, int(d['W']), int(d['H'])


# ------------------------------------------------------------------ matching
def match_pair(flann, K, kps, des, i, j, ratio=0.75, thr=1.0):
    m = flann.knnMatch(des[i], des[j], k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < ratio * y.distance]
    if len(good) < 25:
        return None
    qi = np.array([g.queryIdx for g in good])
    ti = np.array([g.trainIdx for g in good])
    E, mask = cv2.findEssentialMat(kps[i][qi], kps[j][ti], K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=thr)
    if E is None or E.shape != (3, 3):
        return None
    inl = mask.ravel().astype(bool)
    return qi[inl], ti[inl]


def stage_match(out):
    times, shot, kps, des, W, H = load_feats(out)
    n = len(times)
    K = np.array([[F0, 0, W / 2], [0, F0, H / 2], [0, 0, 1]])
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 64})
    pairs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 8)) if shot[i] == shot[j]]
    cross = []
    for s in range(len(SHOTS) - 1):  # around each cut
        a = np.where(shot == s)[0][-10:]
        b = np.where(shot == s + 1)[0][:10]
        cross += [(i, j) for i in a for j in b]
    for s in range(len(SHOTS)):  # every 3rd frame of a shot against every 3rd frame of all earlier shots
        for s2 in range(s - 1):
            a = np.where(shot == s2)[0][::3]
            b = np.where(shot == s)[0][::3]
            cross += [(i, j) for i in a for j in b]
    rec_i, rec_j, rec_a, rec_b = [], [], [], []
    stat = {}
    for k, (i, j) in enumerate(pairs + cross):
        r = match_pair(flann, K, kps, des, i, j)
        if r is None:
            continue
        a, b = r
        is_cross = shot[i] != shot[j]
        if is_cross and len(a) < 25:
            continue
        if is_cross:
            key = (int(shot[i]), int(shot[j]))
            stat[key] = max(stat.get(key, 0), len(a))
        rec_i.append(np.full(len(a), i))
        rec_j.append(np.full(len(a), j))
        rec_a.append(a)
        rec_b.append(b)
    np.savez(os.path.join(out, 'matches.npz'), i=np.concatenate(rec_i), j=np.concatenate(rec_j),
             a=np.concatenate(rec_a), b=np.concatenate(rec_b))
    log('matching done; best cross-shot inliers:', {f'{a + 1}-{b + 1}': v for (a, b), v in sorted(stat.items())})


def load_matches(out):
    d = np.load(os.path.join(out, 'matches.npz'))
    return d['i'], d['j'], d['a'], d['b']


# ------------------------------------------------------------------ per-shot SfM
def sfm_shot(s, times, shot, kps, W, H, mi, mj, ma, mb, init_hint=None):
    frames = np.where(shot == s)[0]
    fset = set(frames.tolist())
    sel = np.array([(i in fset and j in fset) for i, j in zip(mi, mj)])
    uf = UF()
    for i, j, a, b in zip(mi[sel], mj[sel], ma[sel], mb[sel]):
        uf.union((int(i), int(a)), (int(j), int(b)))
    groups = {}
    for node in list(uf.p.keys()):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = [sorted(g) for g in groups.values() if len(g) >= 3 and len({x for x, _ in g}) == len(g)]
    obs = {fi: {} for fi in frames}
    kp2track = {}
    for ti, tr in enumerate(tracks):
        for fi, kp in tr:
            obs[fi][ti] = kp
            kp2track[(fi, kp)] = ti
    f = F0
    c = np.array([W / 2, H / 2])
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])

    # initial pair: many shared tracks and enough parallax
    best = None
    for ii, i in enumerate(frames):
        if init_hint and not (init_hint[0] <= times[i] <= init_hint[1]):
            continue
        for j in frames[ii + 2: ii + 7]:
            shared = list(set(obs[i]) & set(obs[j]))
            if len(shared) < 80:
                continue
            p1 = np.array([kps[i][obs[i][t]] for t in shared])
            p2 = np.array([kps[j][obs[j][t]] for t in shared])
            E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC, threshold=1.0)
            if E is None or E.shape != (3, 3):
                continue
            # parallax measured after removing the rotation (homography fit): pure rotation gives none
            Hh, hm = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
            h_ratio = hm.sum() / max(1, mask.sum())
            par = np.median(np.linalg.norm(cv2.perspectiveTransform(p1[None].astype(np.float64), Hh)[0] - p2, axis=1))
            score = mask.sum() * min(par + 0.5, 6) * (1.2 - min(h_ratio, 1))
            if best is None or score > best[0]:
                best = (score, i, j, shared, p1, p2, E, mask)
    _, i0, j0, shared, p1, p2, E, mask = best
    _, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
    poses = {i0: (np.zeros(3), np.zeros(3)), j0: (cv2.Rodrigues(R)[0].ravel(), t.ravel())}
    pts3 = {}

    def Rt(fi):
        return cv2.Rodrigues(poses[fi][0])[0], poses[fi][1]

    def triangulate(fa, fb, cand, min_ang=0.8):
        if not cand:
            return
        Ra, ta = Rt(fa)
        Rb, tb = Rt(fb)
        a = np.array([kps[fa][obs[fa][tk]] for tk in cand]).T
        b = np.array([kps[fb][obs[fb][tk]] for tk in cand]).T
        X = cv2.triangulatePoints(K @ np.hstack([Ra, ta[:, None]]), K @ np.hstack([Rb, tb[:, None]]), a, b)
        X = (X[:3] / X[3]).T
        Ca, Cb = -Ra.T @ ta, -Rb.T @ tb
        za = X @ Ra.T + ta
        zb = X @ Rb.T + tb
        va, vb = X - Ca, X - Cb
        cosang = np.sum(va * vb, 1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        pa = za[:, :2] / np.maximum(za[:, 2:], 1e-9) * f + c
        pb = zb[:, :2] / np.maximum(zb[:, 2:], 1e-9) * f + c
        err = np.maximum(np.linalg.norm(pa - a.T, axis=1), np.linalg.norm(pb - b.T, axis=1))
        for tk, x, ok in zip(cand, X, (za[:, 2] > 0) & (zb[:, 2] > 0) & (ang > min_ang) & (err < 3) & np.all(np.isfinite(X), 1)):
            if ok:
                pts3[tk] = x

    triangulate(i0, j0, shared)
    init_n = len(pts3)

    def ba(iters):
        nonlocal f, K
        order = sorted(poses)
        ci = {fi: k for k, fi in enumerate(order)}
        pids = sorted(pts3)
        pi = {tk: k for k, tk in enumerate(pids)}
        ob = [(ci[fi], pi[tk], *kps[fi][kp]) for fi in order for tk, kp in obs[fi].items() if tk in pi]
        cams = np.array([np.concatenate(poses[fi]) for fi in order])
        P3 = np.array([pts3[tk] for tk in pids])
        cams, P3, f = bundle_adjust(cams, P3, ob, f, c, iters=iters)
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
        for k, fi in enumerate(order):
            poses[fi] = (cams[k, :3], cams[k, 3:])
        o = np.asarray(ob)
        Rs = np.stack([cv2.Rodrigues(cm[:3])[0] for cm in cams])
        Xc = np.einsum('nij,nj->ni', Rs[o[:, 0].astype(int)], P3[o[:, 1].astype(int)]) + cams[o[:, 0].astype(int), 3:]
        proj = f * Xc[:, :2] / np.maximum(Xc[:, 2:3], 1e-6) + c
        err = np.linalg.norm(proj - o[:, 2:4], axis=1)
        bad = np.zeros(len(P3), bool)
        np.maximum.at(bad, o[:, 1].astype(int), (err > 4) | (Xc[:, 2] <= 0))
        for k, tk in enumerate(pids):
            if bad[k]:
                del pts3[tk]
            else:
                pts3[tk] = P3[k]
        return float(np.median(err)), float(np.sqrt(np.mean(err ** 2)))

    failed = set()
    since = 0
    while True:
        cand = []
        for fi in frames:
            if fi in poses or fi in failed:
                continue
            common = [tk for tk in obs[fi] if tk in pts3]
            if len(common) >= 15:
                # prefer frames next to registered ones (keeps the growth local and well conditioned)
                adj = min(abs(fi - k) for k in poses)
                cand.append((adj == 1, len(common), fi, common))
        if not cand:
            break
        _, _, fi, common = max(cand)
        X = np.array([pts3[tk] for tk in common])
        x = np.array([kps[fi][obs[fi][tk]] for tk in common])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, reprojectionError=3.0, iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 12:
            failed.add(fi)
            continue
        rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], x[inl.ravel()], K, None, rvec, tvec)
        poses[fi] = (rvec.ravel(), tvec.ravel())
        for fj in sorted((k for k in poses if k != fi), key=lambda k: abs(k - fi))[:6]:
            triangulate(fi, fj, [tk for tk in obs[fi] if tk not in pts3 and tk in obs[fj]])
        since += 1
        if since >= 8:
            ba(20)
            since = 0
            failed.clear()
    med, rm = ba(100)

    # Final passes: frames whose points were lost (e.g. during a camera turn) are placed again with
    # PnP, and every untriangulated track is triangulated from its two most distant registered views.
    for _ in range(3):
        counts = {fi: sum(1 for tk in obs[fi] if tk in pts3) for fi in poses}
        for fi, cnt in counts.items():
            if cnt >= 300:
                continue
            common = [tk for tk in obs[fi] if tk in pts3]
            if len(common) < 12:
                continue
            X = np.array([pts3[tk] for tk in common])
            x = np.array([kps[fi][obs[fi][tk]] for tk in common])
            r0, t0 = poses[fi]
            ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, rvec=r0.reshape(3, 1).copy(), tvec=t0.reshape(3, 1).copy(),
                                                     useExtrinsicGuess=True, reprojectionError=3.0, iterationsCount=2000,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= 10:
                rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], x[inl.ravel()], K, None, rvec, tvec)
                poses[fi] = (rvec.ravel(), tvec.ravel())
        added = 0
        for tk, tr in enumerate(tracks):
            if tk in pts3:
                continue
            reg = [fi for fi, _ in tr if fi in poses]
            if len(reg) < 2:
                continue
            before = len(pts3)
            triangulate(reg[0], reg[-1], [tk], min_ang=0.5)
            added += len(pts3) - before
        med, rm = ba(60)
        weak = sum(1 for fi in poses if sum(1 for tk in obs[fi] if tk in pts3) < 150)
        log(f'  shot {s + 1} refinement: +{added} points, frames with <150 points: {weak}, median {med:.2f}px')
    med, rm = ba(60)
    order = sorted(poses)
    log(f'shot {s + 1}: init {i0}-{j0} (t {times[i0]:.2f}-{times[j0]:.2f}, {init_n} pts); {len(order)}/{len(frames)} frames, '
        f'{len(pts3)} pts, reproj median {med:.2f}px RMSE {rm:.2f}px, f {f:.0f}')
    pids = sorted(pts3)
    track_kps = [tracks[tk] for tk in pids]
    # flat arrays of (frame, kp, point index) for linking
    link = np.array([(fi, kp, k) for k, tk in enumerate(pids) for fi, kp in tracks[tk]], np.int64)
    Rs = np.array([cv2.Rodrigues(poses[fi][0])[0] for fi in order])
    ts = np.array([poses[fi][1] for fi in order])
    return dict(frames=np.array(order), R=Rs, t=ts, pts=np.array([pts3[tk] for tk in pids]), f=f,
                link=link, rmse=rm, median=med, all_frames=frames), track_kps


def stage_sfm(out, which=None):
    times, shot, kps, des, W, H = load_feats(out)
    mi, mj, ma, mb = load_matches(out)
    hints = {0: (14.0, 21.0)}  # shot 1: start inside the gallery, where the forward glide gives strong parallax
    for s in range(len(SHOTS)):
        if which and s not in which:
            continue
        res, _ = sfm_shot(s, times, shot, kps, W, H, mi, mj, ma, mb, hints.get(s))
        np.savez(os.path.join(out, f'shot{s}.npz'), **res)


# ------------------------------------------------------------------ linking
def umeyama(A, B):
    """Similarity (s, R, t) with B ~ s R A + t."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / A0.var(0).sum()
    return s, R, mb - s * R @ ma


def stage_link(out):
    times, shot, kps, des, W, H = load_feats(out)
    mi, mj, ma, mb = load_matches(out)
    shots = [dict(np.load(os.path.join(out, f'shot{s}.npz'))) for s in range(len(SHOTS))]
    kp2pt = []
    for sd in shots:
        kp2pt.append({(int(fi), int(kp)): int(k) for fi, kp, k in sd['link']})
    # 3D-3D correspondences between each pair of shots
    corr = {}
    for i, j, a, b in zip(mi, mj, ma, mb):
        si, sj = int(shot[i]), int(shot[j])
        if si == sj:
            continue
        pa = kp2pt[si].get((int(i), int(a)))
        pb = kp2pt[sj].get((int(j), int(b)))
        if pa is None or pb is None:
            continue
        corr.setdefault((si, sj), set()).add((pa, pb))
    rng = np.random.default_rng(0)
    sims = {}
    for (si, sj), cs in sorted(corr.items()):
        cs = np.array(sorted(cs))
        A = shots[sj]['pts'][cs[:, 1]]  # map shot sj into shot si
        B = shots[si]['pts'][cs[:, 0]]
        if len(cs) < 6:
            log(f'link {si + 1}<-{sj + 1}: only {len(cs)} correspondences')
            continue
        scaleB = np.median(np.linalg.norm(B - np.median(B, 0), axis=1))
        best = (0, None)
        for _ in range(4000):
            idx = rng.choice(len(cs), 3, replace=False)
            try:
                s_, R_, t_ = umeyama(A[idx], B[idx])
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(s_) or s_ <= 0:
                continue
            d = np.linalg.norm((s_ * A @ R_.T + t_) - B, axis=1)
            inl = d < 0.05 * scaleB
            if inl.sum() > best[0]:
                best = (inl.sum(), inl)
        if best[1] is None or best[0] < 5:
            log(f'link {si + 1}<-{sj + 1}: {len(cs)} correspondences, no consistent similarity')
            continue
        s_, R_, t_ = umeyama(A[best[1]], B[best[1]])
        d = np.linalg.norm((s_ * A @ R_.T + t_) - B, axis=1)
        inl = d < 0.05 * scaleB
        s_, R_, t_ = umeyama(A[inl], B[inl])
        sims[(si, sj)] = (s_, R_, t_, int(inl.sum()), len(cs))
        log(f'link {si + 1}<-{sj + 1}: {len(cs)} correspondences, {inl.sum()} inliers, scale {s_:.3f}')
    np.save(os.path.join(out, 'sims.npy'), np.array([(k, v) for k, v in sims.items()], dtype=object), allow_pickle=True)


def rot_avg(Rs):
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def stage_link2(out):
    """Places frames of shot j into shot i's coordinates with PnP (2D features of shot j matched to 3D
    points of shot i), then derives the similarity X_i = s R X_j + t from those camera poses."""
    times, shot, kps, des, W, H = load_feats(out)
    mi, mj, ma, mb = load_matches(out)
    shots = [dict(np.load(os.path.join(out, f'shot{s}.npz'))) for s in range(len(SHOTS))]
    kp2pt = [{(int(fi), int(kp)): int(k) for fi, kp, k in sd['link']} for sd in shots]
    sims = {}
    for si in range(len(SHOTS)):
        for sj in range(len(SHOTS)):
            if si == sj:
                continue
            fi_ = shots[si]['f']
            K = np.array([[fi_, 0, W / 2], [0, fi_, H / 2], [0, 0, 1]])
            per_frame = {}
            for i, j, a, b in zip(mi, mj, ma, mb):
                # orient the match as (frame of si, frame of sj)
                if shot[i] == si and shot[j] == sj:
                    fa, ka, fb, kb = i, a, j, b
                elif shot[j] == si and shot[i] == sj:
                    fa, ka, fb, kb = j, b, i, a
                else:
                    continue
                p = kp2pt[si].get((int(fa), int(ka)))
                if p is not None:
                    per_frame.setdefault(int(fb), {})[int(kb)] = p  # one 3D point per keypoint
            regs = []
            own = {int(f): k for k, f in enumerate(shots[sj]['frames'])}
            for fb, corr in per_frame.items():
                if len(corr) < 25 or fb not in own:
                    continue
                X = shots[si]['pts'][list(corr.values())]
                x = kps[fb][list(corr.keys())]
                ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, reprojectionError=2.5, iterationsCount=3000, flags=cv2.SOLVEPNP_EPNP)
                if not ok or inl is None or len(inl) < 20 or len(inl) < 0.35 * len(corr):
                    continue
                rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], x[inl.ravel()], K, None, rvec, tvec)
                Ri = cv2.Rodrigues(rvec)[0]
                k = own[fb]
                Rown, town = shots[sj]['R'][k], shots[sj]['t'][k]
                regs.append(dict(f=fb, n=len(inl), Ri=Ri, Ci=-Ri.T @ tvec.ravel(), Rown=Rown, Cown=-Rown.T @ town))
            if len(regs) < 2:
                continue
            R = rot_avg([r['Ri'].T @ r['Rown'] for r in regs])  # X_i = s R X_j + t  =>  R_own = R_i R
            # scale from distances between registered camera centres (pairs with a real baseline)
            ratios = []
            for a in range(len(regs)):
                for b in range(a + 1, len(regs)):
                    do = np.linalg.norm(regs[a]['Cown'] - regs[b]['Cown'])
                    di = np.linalg.norm(regs[a]['Ci'] - regs[b]['Ci'])
                    if do > 1e-9:
                        ratios.append((do, di / do))
            ratios.sort(key=lambda r: -r[0])
            top = [r for _, r in ratios[:max(3, len(ratios) // 3)]]
            s_ = float(np.median(top))
            t_ = np.median([r['Ci'] - s_ * R @ r['Cown'] for r in regs], axis=0)
            resid = [np.linalg.norm(r['Ci'] - (s_ * R @ r['Cown'] + t_)) for r in regs]
            rot_err = [np.degrees(np.arccos(np.clip((np.trace((r['Ri'] @ R).T @ r['Rown']) - 1) / 2, -1, 1))) for r in regs]
            spread = np.std(top) / s_ if len(top) > 1 else 1.0
            sims[(si, sj)] = (s_, R, t_, len(regs), sum(r['n'] for r in regs), float(np.median(resid)), float(np.median(rot_err)), float(spread))
            log(f'pnp link {si + 1}<-{sj + 1}: {len(regs)} frames placed ({sum(r["n"] for r in regs)} inliers), '
                f'scale {s_:.3f} (spread {spread:.2f}), centre residual {np.median(resid):.3f}, rot err {np.median(rot_err):.2f} deg')
    np.save(os.path.join(out, 'sims_pnp.npy'), np.array([(k, v) for k, v in sims.items()], dtype=object), allow_pickle=True)


def stage_bridge(out):
    """Joins consecutive shots across each cut with two-view geometry: for the frame pairs that match
    best across the cut, the essential matrix gives the relative rotation and the direction of travel;
    the length of the move and the relative scale of the two shots come from comparing triangulated
    depths with the depths of the same points in each shot's own reconstruction."""
    times, shot, kps, des, W, H = load_feats(out)
    mi, mj, ma, mb = load_matches(out)
    shots = [dict(np.load(os.path.join(out, f'shot{s}.npz'))) for s in range(len(SHOTS))]
    kp2pt = [{(int(fi), int(kp)): int(k) for fi, kp, k in sd['link']} for sd in shots]
    pose_of = [{int(f): k for k, f in enumerate(sd['frames'])} for sd in shots]
    bridges = {}
    for s in range(len(SHOTS) - 1):
        cand = []
        pairs = {}
        for i, j, a, b in zip(mi, mj, ma, mb):
            if shot[i] == s and shot[j] == s + 1:
                pairs.setdefault((int(i), int(j)), []).append((int(a), int(b)))
        for (i, j), m in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:6]:
            f = float(shots[s]['f'])
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
            a_idx = np.array([x for x, _ in m])
            b_idx = np.array([y for _, y in m])
            p1, p2 = kps[i][a_idx].astype(np.float64), kps[j][b_idx].astype(np.float64)
            E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.USAC_MAGSAC, threshold=1.0)
            if E is None or E.shape != (3, 3):
                continue
            n_in, Rab, tab, mask2 = cv2.recoverPose(E, p1, p2, K, mask=mask)
            inl = mask2.ravel() > 0
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([Rab, tab])
            X = cv2.triangulatePoints(P1, P2, p1[inl].T, p2[inl].T)
            X = (X[:3] / X[3]).T  # camera-a coordinates, unit baseline
            Xb = X @ Rab.T + tab.ravel()
            ang = np.degrees(np.arccos(np.clip(np.sum(X * (X - (-Rab.T @ tab).ravel()), 1) /
                                               (np.linalg.norm(X, axis=1) * np.linalg.norm(X - (-Rab.T @ tab).ravel(), axis=1) + 1e-12), -1, 1)))
            good = (X[:, 2] > 0) & (Xb[:, 2] > 0) & (ang > 1.0)
            ka, kb = a_idx[inl], b_idx[inl]
            # depths of the same points in each shot's own reconstruction
            Ra, ta = shots[s]['R'][pose_of[s][i]], shots[s]['t'][pose_of[s][i]]
            Rb, tb = shots[s + 1]['R'][pose_of[s + 1][j]], shots[s + 1]['t'][pose_of[s + 1][j]]
            r1, r2 = [], []
            for n_, (x, y) in enumerate(zip(ka, kb)):
                if not good[n_]:
                    continue
                pa = kp2pt[s].get((i, int(x)))
                if pa is not None:
                    za = (shots[s]['pts'][pa] @ Ra.T + ta)[2]
                    if za > 0:
                        r1.append(za / X[n_, 2])
                pb = kp2pt[s + 1].get((j, int(y)))
                if pb is not None:
                    zb = (shots[s + 1]['pts'][pb] @ Rb.T + tb)[2]
                    if zb > 0:
                        r2.append(zb / Xb[n_, 2])
            if len(r1) < 8 or len(r2) < 8:
                continue
            sc1, sc2 = float(np.median(r1)), float(np.median(r2))
            spread1 = float(np.median(np.abs(np.array(r1) / sc1 - 1)))
            spread2 = float(np.median(np.abs(np.array(r2) / sc2 - 1)))
            Rb_s = Rab @ Ra
            tb_s = Rab @ ta + sc1 * tab.ravel()
            Cb_s = -Rb_s.T @ tb_s
            R = Rb_s.T @ Rb
            sc = sc1 / sc2
            Cb_own = -Rb.T @ tb
            t = Cb_s - sc * R @ Cb_own
            cand.append(dict(i=i, j=j, n=int(inl.sum()), n1=len(r1), n2=len(r2), sc=sc, R=R, t=t,
                             spread=max(spread1, spread2), move=sc1))
        if not cand:
            log(f'bridge {s + 1}->{s + 2}: no usable pair')
            continue
        for c in cand:
            log(f'  pair {c["i"]}-{c["j"]} (t {times[c["i"]]:.2f}/{times[c["j"]]:.2f}): {c["n"]} inliers, '
                f'{c["n1"]}/{c["n2"]} depth links, rel scale {c["sc"]:.3f}, spread {c["spread"]:.2f}, move {c["move"]:.3f}')
        # agreement between the candidate pairs
        best = min(cand, key=lambda c: c['spread'])
        rots = [np.degrees(np.arccos(np.clip((np.trace(best['R'].T @ c['R']) - 1) / 2, -1, 1))) for c in cand]
        scs = [c['sc'] / best['sc'] for c in cand]
        good = [c for c, r, q in zip(cand, rots, scs) if r < 3 and abs(np.log(q)) < 0.15]
        R = rot_avg([c['R'] for c in good])
        sc = float(np.exp(np.median([np.log(c['sc']) for c in good])))
        t = np.median([c['t'] for c in good], axis=0) if len(good) > 1 else best['t']
        bridges[(s, s + 1)] = (sc, R, t, len(good), len(cand), float(np.median(rots)), float(np.std(np.log(scs))))
        log(f'bridge {s + 1}->{s + 2}: {len(good)}/{len(cand)} pairs agree (rotation spread {np.median(rots):.2f} deg, '
            f'scale spread {np.std(np.log(scs)):.2f}), relative scale {sc:.3f}')
    np.save(os.path.join(out, 'bridges.npy'), np.array([(k, v) for k, v in bridges.items()], dtype=object), allow_pickle=True)


if __name__ == '__main__':
    video, out = sys.argv[1], sys.argv[2]
    stages = sys.argv[3:] or ['features', 'match', 'sfm', 'link']
    os.makedirs(out, exist_ok=True)
    for st in stages:
        if st == 'features':
            stage_features(video, out)
        elif st == 'match':
            stage_match(out)
        elif st.startswith('sfm'):
            which = [int(x) - 1 for x in st[3:].split(',')] if len(st) > 3 else None
            stage_sfm(out, which)
        elif st == 'link':
            stage_link(out)
        elif st == 'link2':
            stage_link2(out)
        elif st == 'bridge':
            stage_bridge(out)
