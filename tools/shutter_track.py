"""Measures the REVERE shutter roll-up in RVR1.mp4, frame by frame.

While the camera stands still (6.0 - 15.58 s) the shutter rises as a rigid sheet. For every video
frame this finds the image row of the shutter's bottom edge (sub-pixel, full resolution), and checks
that the slats above the edge really move rigidly with it (vertical cross-correlation).

Writes a JSON with the door rectangle and the edge row per frame (normalised 0..1 image coords).
Usage: python3 shutter_track.py <video> <out.json>
"""
import json
import sys

import cv2
import numpy as np

F0, F1 = 144, 374  # 6.000 s .. 15.583 s (static camera)
video, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video)
frames = {}
i = 0
while True:
    ok, img = cap.read()
    if not ok or i > F1:
        break
    if i >= F0:
        frames[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i += 1
H, W = frames[F0].shape

# door rectangle: where the fully open frame differs from the closed one
d = cv2.GaussianBlur(np.abs(frames[F1] - frames[F0]), (9, 9), 0) > 18
rows = d.mean(1)
door_rows = np.where(rows > 0.55)[0]
y0, y1 = int(door_rows.min()), int(door_rows.max())
cols = d[y0:y1 + 1].mean(0)
door_cols = np.where(cols > 0.5)[0]
x0, x1 = int(door_cols.min()), int(door_cols.max())
print('door rect (px)', x0, x1, y0, y1, 'of', W, H)

# 1) coarse edge per frame: strongest dark->bright step in the median row profile, tracked upwards
edges = []
prev, vel = float(y1), 0.0
band = slice(x0 + 40, x1 - 40)
for fr in range(F0, F1 + 1):
    prof = np.median(frames[fr][:, band], axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    step = np.zeros_like(prof)
    k = 10
    step[k:-k] = np.array([prof[y:y + k].mean() - prof[y - k:y].mean() for y in range(k, len(prof) - k)])
    pred = prev + vel
    lo, hi = int(max(y0 + 12, pred - 50)), int(min(y1, prev + 6))
    seg = step[lo:hi + 1]
    if len(seg) and seg.max() > 15:
        y = lo + int(np.argmax(seg))
        # parabolic sub-pixel refinement
        if lo < y < hi:
            a, b, c = step[y - 1], step[y], step[y + 1]
            den = a - 2 * b + c
            y = y + (0.5 * (a - c) / den if abs(den) > 1e-6 else 0)
    else:
        y = pred
    y = float(np.clip(y, y0, prev))
    if y < prev - 0.5:
        vel = 0.6 * vel + 0.4 * (y - prev)
    edges.append(y)
    prev = y
edges = np.array(edges)

# 2) rigidity check + refinement: the slats just above the edge must move by the same amount
#    as the edge between consecutive frames
shifts = []
for n in range(1, len(edges)):
    e = edges[n]
    if e - y0 < 80:
        shifts.append(np.nan)
        continue
    a = frames[F0 + n - 1][int(y0 + 10):int(e - 8), band]
    b = frames[F0 + n][int(y0 + 10):int(e - 8), band]
    pa = a.mean(1) - a.mean()
    pb = b.mean(1) - b.mean()
    best, bs = None, -1e9
    for s10 in range(0, 200):  # shift 0..20 px upwards, 0.1 px steps
        s = s10 / 10
        xs = np.arange(len(pb))
        shifted = np.interp(xs + s, np.arange(len(pa)), pa)
        sc = -np.mean((shifted[: len(pb) - 25] - pb[: len(pb) - 25]) ** 2)
        if sc > bs:
            bs, best = sc, s
    shifts.append(best)
shifts = np.array(shifts)
edge_steps = -np.diff(edges)
ok = ~np.isnan(shifts)
print('frames', len(edges), '| edge moves', edges[0], '->', edges[-1])
print('slat shift vs edge step (median abs diff, px):', np.nanmedian(np.abs(shifts[ok] - edge_steps[ok])))

# 3) final curve: integrate the (more precise) slat shifts where the shutter is visible, anchor to
#    the measured edges, and force it to be monotonic and smooth
curve = edges.copy()
integ = [edges[0]]
for n in range(1, len(edges)):
    s = shifts[n - 1]
    step_ = s if not np.isnan(s) and abs(s - edge_steps[n - 1]) < 4 else edge_steps[n - 1]
    integ.append(integ[-1] - step_)
integ = np.array(integ)
# anchor: least-squares offset so the integrated curve matches the measured edges in the middle part
mid = (edges < y1 - 10) & (edges > y0 + 60)
if mid.any():
    integ += np.median(edges[mid] - integ[mid])
curve = np.minimum.accumulate(np.clip(integ, y0, y1))
# gentle smoothing (keeps monotonic)
ker = np.array([1, 2, 3, 2, 1], float)
ker /= ker.sum()
sm = np.convolve(np.pad(curve, 2, mode='edge'), ker, mode='valid')
curve = np.minimum.accumulate(sm)
print('max |curve - edges| px', np.max(np.abs(curve - edges)), 'median', np.median(np.abs(curve - edges)))
json.dump(dict(fps=24, first_frame=F0, t0=F0 / 24, door=[x0 / W, x1 / W, y0 / H, y1 / H],
               edge=[round(float(v) / H, 5) for v in curve], edge_raw=[round(float(v) / H, 5) for v in edges]),
          open(out, 'w'))
