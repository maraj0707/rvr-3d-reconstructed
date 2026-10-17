# RVR showroom: 3D reconstruction of the client video

This version of the website is built **from the client video itself** (`RVR1.mp4`). There are no
invented models. Everything you see in 3D was reconstructed from the footage:

- the REVERE shutter
- the gallery
- the workshop
- the lounge

## What was reconstructed, and how

1. **Camera path (structure-from-motion).** The video was split into its shots. The camera motion
   and about 50,000 3D points of the room were recovered for every frame, using:
   - SIFT features
   - matching across frames
   - incremental pose estimation
   - bundle adjustment

   Every frame of all five shots is placed with a median reprojection error of about 0.5 pixels.
   The video has two more cuts that are easy to miss: at 15.6 s (after the shutter) and at 18.6 s
   (the camera jumps about 1 m forward).
2. **Camera pose for every video frame** (`tools/register_all.py`). The structure-from-motion step
   used 4–6 frames per second. To follow the real camera exactly, all video frames (24 per second)
   were then placed against the reconstructed 3D points. Results:
   - 1,108 poses
   - the path passes exactly through the bundle-adjusted SfM poses (where the keyframes are); the
     per-frame poses add the motion between them
   - the few per-frame poses that are off (where the moving video wall fills the view, around
     22.75–23.0 s) are replaced by interpolation through the SfM poses
   - a very light smoothing (over 3 frames) removes measurement jitter
   - during the roll-up (6.0–15.6 s) the camera is held at one pose, as in the video

   This also found a jump of the camera at 36.46 s in the workshop (a hidden edit in the video),
   which is now handled as a cut.
3. **Dense depth (multi-view stereo).** There is a *keyframe* (221 in total):
   - every 0.25 s from the entrance through the gallery (0–24.6 s)
   - every 1/6 s where the camera moves fast past nearby objects (towards the workshop, the workshop
     bench, the lounge)
   - every 1/3 s at the gallery tables

   For each keyframe, a depth map was computed:
   - **Plane-sweep stereo** against the neighbouring frames.
   - **The SfM points**, mixed in as the most reliable depths.
   - **Edge-aware filling**, so depth edges follow the edges in the picture.

   While the camera stands still in front of the door, the shutter and the opening behind it get
   separate depth: the shutter's depth from the approach, and the gallery's depth from the frames
   after the door.
4. **The shutter roll-up, frame by frame** (`tools/shutter_track.py`). From 6.0 s to 15.6 s the
   camera stands still and the shutter rises as one rigid sheet: its slats move exactly with its bottom
   edge, to within 0.2 px. The edge position was measured in every one of the 231 video frames. The
   website draws each moment from the two nearest video frames:
   - The shutter part is taken from each frame shifted by how far the edge has moved since then, so
     there is never a double edge.
   - The gallery comes from where those frames already show it.

   The edge follows the measured curve continuously, at any scroll speed.
5. **Website (Three.js).** Each keyframe becomes a 3D surface: a mesh of 192 × 108 vertices pushed
   out to the reconstructed depth. It is placed where the camera really was, and textured with that
   original video frame.
   - Scrolling moves a real 3D camera along the recovered camera path, one pose per video frame.
     Between them, position and rotation are interpolated with smooth curves (Catmull-Rom), so speed
     and turning change gradually.
   - During the roll-up the camera is held exactly where the real camera stood.
   - The scroll drives the camera through a critically damped spring, so it eases in and out
     instead of jumping on every wheel step.
   - The two nearest keyframe surfaces are rendered from that camera. Most of the time one of them
     is shown; the hand-over happens in the middle. Where the two disagree (a detail the depth
     doesn't line up perfectly), the hand-over is sharp, so there is never a double image.
   - Moving the mouse moves the camera off the path. On touch screens, a slow sway does the same.
     Near and far things then shift against each other: real parallax.

### Cuts
The video has seven hard cuts or jumps: 15.6 s, 18.6 s, 24.7 s, 28.8 s, 33.6 s, 36.5 s and 40.5 s.
The website keeps them in the same places. When the scroll crosses one, the picture cross-fades over 0.45 s of real
time; it is never left half-faded when the scroll stops, and the camera holds still during the fade.
With reduced motion turned on, it cuts directly, like the video.

### No double images at rest
Two neighbouring keyframes are cross-faded only around the middle of the interval between them, and
the more their viewpoints differ, the shorter that moment is. When the scroll stops, the picture
settles on the nearer keyframe within about 0.3 s, so a still picture never shows two images.

## Files

| What | Where |
|---|---|
| Page, loading screen, video fallback | `index.html` |
| Styles for the 3D page | `css/showroom.css` (the original `style.css` is unchanged) |
| Runtime: loading, camera path, rendering, quality | `js/recon.js` |
| Three.js r170 (MIT), stored locally, no CDN needed | `js/vendor/three/` |
| Reconstruction data: keyframes, camera path (one pose per video frame), shutter edge curve | `assets/scene/manifest.json` |
| Reconstruction data: one depth grid per keyframe (192 × 108 values), named after the video frame | `assets/scene/depth/f0000.bin` … |
| Original video frames used as textures | `assets/scene/hd/` (1600×900), `assets/scene/sd/` (960×540) |
| Fallback video | `assets/RVR1.mp4`, `assets/RVR1-mobile.mp4` |
| The reconstruction pipeline (Python + OpenCV, not needed on the website) | `tools/` |

## Loading and quality

- **Loading screen.** It stays up until the camera path and the first two keyframes (image + depth)
  have loaded and the first view has been drawn. That is about 1 MB, including Three.js.
- **Background loading.** The two shutter frames (closed and open) load first. Then everything
  between the camera and where you are scrolling to (nearest first), plus 12 s of the tour beyond it;
  the rest loads as you get closer. Keyframes inside the shutter roll-up only need their image.
- **No running ahead.** If you scroll faster than the data arrives, the camera waits where the data
  ends and a small "Loading" notice shows. A long jump (the area buttons) goes straight to its
  destination with the short cut fade as soon as that has loaded.
- **Memory.** Only a few frames are kept on the graphics card at a time.
- **Quality levels.** These are chosen automatically:

  | Level | Pixel density | Images |
  |---|---|---|
  | high | up to 1.5× | 1600×900 |
  | medium | up to 1.25× | 1600×900 |
  | low (phones, software graphics) | 1× | 960×540 |

  If the frame rate drops below about 28 fps, the resolution steps down by itself.
- **Testing switches** (add to the address):
  - `?quality=high`, `?quality=medium` or `?quality=low` forces a level.
  - `?video` shows the original scroll-video version.
- **Fallback.** Browsers without WebGL 2 or import maps get the original scroll-controlled video
  (`script.js`, unchanged). So do browsers where the 3D code fails or the graphics context is lost.
- **Reduced motion.** With "reduce motion" on, there is no camera easing, no mouse parallax and no sway.

## Limitations (honest list)

- **The 3D is view-based, not a complete free-roam model.** Each keyframe has its own 3D surface,
  so the camera can only move a limited distance away from where the real camera was:
  - the scroll path
  - plus the small mouse or sway movement

  Areas the video never shows (behind objects, behind the camera, other rooms) don't exist in the
  data and are not invented.
- **Not a NeRF or Gaussian Splatting model.** Those need a GPU and machine-learning libraries,
  which weren't available here. The pipeline is classic photogrammetry: structure-from-motion and
  multi-view stereo.
- **The shutter** is shown from two surfaces: the closed door and the open door. Its motion is
  exact as seen from the original camera position. If you move the mouse during the roll-up, the
  shutter's bottom edge has a slight stretch, where the shutter meets the gallery behind it.
- **Lighting during the roll-up** (the light spilling onto the floor outside) is cross-faded between
  video frames 0.25 s apart, not taken from every single frame.
- **Video screens, reflections and glossy surfaces.** The video screens (EXPERIENCES wall, monitors)
  change content, and the glossy floor shows reflections. These are view-dependent in the footage,
  so they can shimmer a little when blending or when the mouse moves the camera.
- **Depth is not perfect.** It is noisy on large plain or dark areas, and around thin objects
  (stools, plant leaves). Moving far off the path would show stretching, which is why mouse movement
  is limited per keyframe.
- **Not one shared space.** The camera doesn't fly from one shot to the next; the cuts are
  cross-faded instead.

## Update of 18 Oct 2026 (shutter, camera, double images)

- The keyframes are now named after their video frame (`f0144.webp` …). The older files
  (`k000.webp` … and `k000.bin` …) are no longer used; `.vercelignore` keeps them out of the
  deployment, and you can delete them.
- `_backup_before_fixes/` holds the previous `js/recon.js`, `assets/scene/manifest.json` and README.
  To go back: copy those three files over the new ones and remove the three `k*` lines from
  `.vercelignore`. `_backup_before_fixes/js/recon.before-issue3-4.js` is the version from just before
  the blending and loading changes (same manifest as now).

## Run locally

Open the folder in VS Code → right-click `index.html` → **Open with Live Server**.

Double-clicking `index.html` won't work, because browsers block module scripts and data files from
`file://`.

## Deploy (Vercel)

Push the folder to GitHub and import it in Vercel as a static site (no build step). All paths are
relative. `.vercelignore` keeps `tools/` and this README out of the deployment.

## Rebuilding the reconstruction

You need Python 3 with `opencv-python` (5.x or 4.x with SIFT), `numpy`, `scipy` and `Pillow`.

```
cd tools
python3 recon2.py RVR1_original.mp4 work features match sfm bridge
python3 shutter_track.py RVR1_original.mp4 work/shutter.json
python3 register_all.py RVR1_original.mp4 work
python3 export.py RVR1_original.mp4 work ..
```

Use the original 1920×1080 video. The whole run takes about half an hour on a normal computer.
`eval_depth.py` measures depth quality: it reprojects a keyframe into neighbouring frames and
compares the result with the real frames.
