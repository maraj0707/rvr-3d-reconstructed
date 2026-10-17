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
   The video has two more jumps that are not visible as cuts: at 15.6 s (after the shutter) and at
   18.6 s (the camera jumps about 1 m forward). A few camera poses where the moving video wall filled
   the view (22.75–23.0 s) were off the path; they are interpolated from their neighbours.
2. **Dense depth (multi-view stereo).** About every half second of video there is a *keyframe*,
   and every quarter second while the shutter rolls up. For each keyframe, a depth map was computed:
   - **Plane-sweep stereo** against the neighbouring frames.
   - **The SfM points**, mixed in as the most reliable depths.
   - **Edge-aware filling**, so depth edges follow the edges in the picture.

   While the camera stands still in front of the door, the shutter and the opening behind it get
   separate depth: the shutter's depth from the approach, and the gallery's depth from the frames
   after the door.
3. **Website (Three.js).** Each keyframe becomes a 3D surface: a mesh of 192 × 108 vertices pushed
   out to the reconstructed depth. It is placed where the camera really was, and textured with that
   original video frame.
   - Scrolling moves a real 3D camera along the recovered camera path.
   - The two nearest keyframe surfaces are rendered from that camera and blended.
   - Moving the mouse moves the camera off the path. On touch screens, a slow sway does the same.
     Near and far things then shift against each other: real parallax.

### Transitions between shots
| Cut | Result |
|---|---|
| Shot 2 → shot 3 (28.8 s) | The cut could be bridged reliably, so both shots share one 3D space and the camera flies from one to the other. |
| The 18.6 s jump, 24.7 s, 33.6 s and 40.5 s | These shots couldn't be joined accurately enough: the footage doesn't overlap enough, or the bridge was a few degrees off. Each side keeps its own camera and the picture dissolves. |

## Files

| What | Where |
|---|---|
| Page, loading screen, video fallback | `index.html` |
| Styles for the 3D page | `css/showroom.css` (the original `style.css` is unchanged) |
| Runtime: loading, camera path, rendering, quality | `js/recon.js` |
| Three.js r170 (MIT), stored locally, no CDN needed | `js/vendor/three/` |
| Reconstruction data: keyframes, camera path, transitions | `assets/scene/manifest.json` |
| Reconstruction data: one depth grid per keyframe (192 × 108 values) | `assets/scene/depth/k000.bin` … `k115.bin` |
| Original video frames used as textures | `assets/scene/hd/` (1600×900), `assets/scene/sd/` (960×540) |
| Fallback video | `assets/RVR1.mp4`, `assets/RVR1-mobile.mp4` |
| The reconstruction pipeline (Python + OpenCV, not needed on the website) | `tools/` |

## Loading and quality

- **Loading screen.** It stays up until the camera path and the first two keyframes (image + depth)
  have loaded and the first view has been drawn. That is about 1 MB, including Three.js.
- **Background loading.** The rest loads in the background, starting with what is nearest to where
  you are scrolling. If you jump far ahead, a small "Loading" notice shows until the keyframes there
  have arrived; this takes about 2 seconds on an 8 Mbit/s connection.
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
- **The shutter** is a moving object. It is shown as 29 keyframes (every 0.25 s) while it rolls up,
  each with its own depth, so it moves in steps that are blended, not as a smooth animation.
- **Video screens, reflections and glossy surfaces.** The video screens (EXPERIENCES wall, monitors)
  change content, and the glossy floor shows reflections. These are view-dependent in the footage,
  so they can shimmer a little when blending or when the mouse moves the camera.
- **Depth is not perfect.** It is noisy on large plain or dark areas, and around thin objects
  (stools, plant leaves). Moving far off the path would show stretching, which is why mouse movement
  is limited per keyframe.
- **Not one shared space.** Shots 1, 2 + 3, 4 and 5 are separate 3D spaces (see the transitions table).

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
python3 export.py RVR1_original.mp4 work ..
```

Use the original 1920×1080 video. The whole run takes about half an hour on a normal computer.
`eval_depth.py` measures depth quality: it reprojects a keyframe into neighbouring frames and
compares the result with the real frames.
