"""Registers EVERY video frame (24 per second) against the reconstruction, so the website's camera
follows the original camera path frame by frame (the SfM itself used 4-6 frames per second, which
misses sudden changes such as the speed-up at 36.46 s).

For each frame: SIFT features -> matched to the two nearest SfM frames' features that have 3D points
-> PnP (RANSAC, then refined). Output: <r2>/dense_poses.npz with, per shot, frame numbers, R, t (shot
coordinates, OpenCV convention) and the number of inliers.

Usage: python3 register_all.py <video> <r2>
"""
import os
import sys
import time

import cv2
import numpy as np

from recon2 import SHOTS, load_feats

video, r2 = sys.argv[1], sys.argv[2]
times, shot, kps, des, W, H = load_feats(r2)
shots = [dict(np.load(os.path.join(r2, f'shot{s}.npz'))) for s in range(len(SHOTS))]
kp2pt = [{(int(fi), int(kp)): int(k) for fi, kp, k in sd['link']} for sd in shots]
sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
clahe = cv2.createCLAHE(2.0, (8, 8))
flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 64})
cap = cv2.VideoCapture(video)
out = {s: dict(frame=[], R=[], t=[], n=[]) for s in range(len(SHOTS))}
T = time.time()
i = 0
while True:
    ok, img = cap.read()
    if not ok:
        break
    t = i / 24
    s = next((k for k, (a, b) in enumerate(SHOTS) if a - 1e-3 <= t <= b + 1e-3), None)
    if s is None:
        i += 1
        continue
    sd = shots[s]
    frames = sd['frames']
    ft = times[frames]
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    k, d = sift.detectAndCompute(clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)), None)
    pk = np.array([p.pt for p in k], np.float32)
    # the two nearest registered SfM frames
    order = np.argsort(np.abs(ft - t))[:2]
    X, x = [], []
    seen = set()
    for j in order:
        fi = int(frames[j])
        m = flann.knnMatch(d, des[fi], k=2)
        for pr in m:
            if len(pr) < 2 or pr[0].distance > 0.75 * pr[1].distance:
                continue
            p = kp2pt[s].get((fi, pr[0].trainIdx))
            if p is None or (pr[0].queryIdx, p) in seen:
                continue
            seen.add((pr[0].queryIdx, p))
            X.append(sd['pts'][p])
            x.append(pk[pr[0].queryIdx])
    # initial guess: pose of the nearest SfM frame
    j0 = int(order[0])
    R0, t0 = sd['R'][j0], sd['t'][j0]
    res = None
    if len(X) >= 20:
        X = np.array(X, np.float64)
        x = np.array(x, np.float64)
        K = np.array([[float(sd['f']), 0, W / 2], [0, float(sd['f']), H / 2], [0, 0, 1]])
        okp, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, rvec=cv2.Rodrigues(R0)[0].copy(), tvec=t0.reshape(3, 1).copy(),
                                                  useExtrinsicGuess=True, reprojectionError=2.0, iterationsCount=2000,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if okp and inl is not None and len(inl) >= 15:
            rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], x[inl.ravel()], K, None, rvec, tvec)
            res = (cv2.Rodrigues(rvec)[0], tvec.ravel(), len(inl))
    if res is not None:
        o = out[s]
        o['frame'].append(i)
        o['R'].append(res[0])
        o['t'].append(res[1])
        o['n'].append(res[2])
    if i % 100 == 0:
        print(f'[{time.time() - T:5.0f}s] frame {i} shot {s + 1}: {res[2] if res else "failed"} inliers', flush=True)
    i += 1
np.savez(os.path.join(r2, 'dense_poses.npz'), **{f'{key}{s}': np.array(v) for s, o in out.items() for key, v in o.items()})
for s, o in out.items():
    print(f'shot {s + 1}: {len(o["frame"])} frames registered, median inliers {np.median(o["n"]) if o["n"] else 0}')
