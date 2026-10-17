/*
  Scroll-controlled video
  -----------------------
  The page's scroll position decides which frame of the video is shown.
  Scroll down -> the video moves forward. Scroll up -> it moves back.
  The video never plays on its own.

  How it works:
  1. Work out how far the user has scrolled through the .scroll-video section (0 to 1).
  2. Turn that into a time in the video (0 to video.duration).
  3. On every animation frame, move the video toward that time with video.currentTime.
     A new seek is only started once the previous one has finished, so the
     browser never gets a pile-up of seeks (that is what causes blank or frozen frames).
  4. In the background, download the whole video into memory once, then switch the
     player to that in-memory copy so every later seek is instant. The file itself is
     not changed: same bytes, same resolution, same quality.
*/

(() => {
  'use strict';

  /* ---------- Settings ---------- */

  // Smoothing: how much of the remaining distance to cover per screen frame.
  // 1 = follow the scroll exactly, lower = softer, more "cinematic" motion.
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const EASE = reduceMotion ? 1 : 0.18;

  // Skip seeks smaller than this while moving (seconds). The final resting
  // position is always seeked exactly.
  const MIN_STEP = 1 / 60;

  // Stop just before the very end so browsers show the last frame instead of a blank one.
  const END_PADDING = 0.01;

  /* ---------- Elements ---------- */

  const section = document.getElementById('scroll-video');
  const stage = section.querySelector('.scroll-video__stage');
  let video = stage.querySelector('video');

  // Phones held upright only show the middle of the video, so they load a smaller,
  // centre-cropped copy (data-mobile-src in index.html) instead of the full frame.
  const mobileSrc = video.getAttribute('data-mobile-src');
  if (mobileSrc && window.matchMedia('(max-aspect-ratio: 2/3)').matches) {
    video.setAttribute('src', mobileSrc);
  }

  const SOURCE = video.getAttribute('src');

  /* ---------- State ---------- */

  let duration = 0;     // video length in seconds, known once metadata has loaded
  let target = 0;       // the time the scroll position asks for
  let current = 0;      // the smoothed time we are moving through
  let requested = -1;   // the last time we asked the video to show
  let rafId = 0;

  /* ---------- Scroll -> time ---------- */

  function scrollProgress() {
    const rect = section.getBoundingClientRect();
    // The video is pinned while the section scrolls by this many pixels.
    const distance = section.offsetHeight - stage.offsetHeight;
    if (distance <= 0) return 0;
    return Math.min(1, Math.max(0, -rect.top / distance));
  }

  function maxTime() {
    return Math.max(0, duration - END_PADDING);
  }

  /* ---------- Animation loop ---------- */

  function schedule() {
    if (!rafId) rafId = requestAnimationFrame(tick);
  }

  function tick() {
    rafId = 0;
    if (!duration) return; // metadata not loaded yet

    target = scrollProgress() * maxTime();

    // Ease toward the target, then snap when close enough.
    current += (target - current) * EASE;
    if (Math.abs(target - current) < 0.001) current = target;

    // Only start a new seek when the previous one is done.
    if (!video.seeking && current !== requested) {
      const settled = current === target;
      if (settled || Math.abs(current - requested) >= MIN_STEP) {
        requested = current;
        video.currentTime = current;
      }
    }

    // Keep looping until the video shows exactly what the scroll position asks for.
    if (current !== target || requested !== current || video.seeking) {
      schedule();
    }
  }

  /* ---------- Video setup ---------- */

  // Some phones (iPhone especially) won't draw any frames until the video has
  // been played once. Play and pause straight away, so nothing actually plays.
  function prime(v) {
    v.__priming = true;
    let result;
    try {
      result = v.play();
    } catch (e) {
      result = null;
    }
    const stop = () => {
      v.pause();
      v.__priming = false;
      if (v === video) {
        requested = -1; // play/pause can nudge the time; force an exact re-seek
        schedule();
      }
    };
    if (result && typeof result.then === 'function') {
      // If blocked (e.g. iPhone Low Power Mode) we simply try again on the first touch.
      return result.then(stop, () => { v.__priming = false; });
    }
    stop();
    return Promise.resolve();
  }

  function onMetadata(e) {
    const v = e.target;
    if (v !== video) return;
    duration = v.duration;
    // Start at the frame that matches the current scroll position (e.g. after a page refresh).
    target = scrollProgress() * maxTime();
    current = target;
    requested = -1;
    prime(v);
    schedule();
  }

  function attach(v) {
    v.addEventListener('loadedmetadata', onMetadata);
    v.addEventListener('seeked', schedule);
    // Safety net: if anything ever starts playback, stop it. Scroll is the only control.
    v.addEventListener('playing', () => {
      if (!v.__priming) v.pause();
    });
  }

  attach(video);
  if (video.readyState >= 1) onMetadata({ target: video });

  /* ---------- Load the whole video into memory (smoother seeking) ---------- */

  async function loadFullVideo() {
    // fetch() doesn't work when the page is opened directly as a file (file://).
    // In that case the browser reads the local file directly, which is fast anyway.
    if (location.protocol === 'file:' || !window.fetch || !window.URL) {
      video.preload = 'auto';
      return;
    }
    try {
      const response = await fetch(SOURCE);
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const data = await response.blob();
      // Some servers send a generic file type; tell the browser it's a video.
      const type = data.type && data.type.indexOf('video/') === 0 ? data.type : 'video/mp4';
      const blob = new Blob([data], { type });
      swapToSource(URL.createObjectURL(blob));
    } catch (err) {
      // Couldn't download it in one go; let the browser buffer the file itself.
      video.preload = 'auto';
    }
  }

  // Switch to the in-memory copy without a black flash:
  // load it in a second <video> underneath, line it up on the same frame, then remove the old one.
  function swapToSource(url) {
    const next = video.cloneNode(false);
    next.removeAttribute('src');
    next.preload = 'auto';
    next.muted = true;
    next.playsInline = true;
    stage.insertBefore(next, video); // earlier in the DOM = drawn underneath

    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      const old = video;
      video = next;
      attach(next);
      duration = next.duration;
      requested = -1;
      old.remove();
      old.removeAttribute('src');
      old.load(); // release the old player's memory
      schedule();
    };

    // If the in-memory copy can't be played for any reason, keep the original player.
    next.addEventListener('error', () => {
      if (done) return;
      done = true;
      next.remove();
      URL.revokeObjectURL(url);
    }, { once: true });

    next.addEventListener('loadeddata', () => {
      prime(next).then(() => {
        next.addEventListener('seeked', finish, { once: true });
        next.currentTime = requested >= 0 ? requested : current;
      });
    }, { once: true });

    next.src = url;
    next.load();
  }

  /* ---------- Events ---------- */

  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule, { passive: true });
  window.addEventListener('orientationchange', schedule, { passive: true });

  // If the phone blocked the first priming (Low Power Mode etc.), try again on the first touch.
  const primeOnGesture = () => {
    prime(video);
    ['touchend', 'pointerdown', 'keydown'].forEach((type) =>
      window.removeEventListener(type, primeOnGesture)
    );
  };
  ['touchend', 'pointerdown', 'keydown'].forEach((type) =>
    window.addEventListener(type, primeOnGesture, { passive: true })
  );

  loadFullVideo();
})();
