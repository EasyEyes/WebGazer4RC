/* eslint-disable */
// Search LOOP for the loop function.

import "@tensorflow/tfjs";
//import(/* webpackPreload: true */ '@tensorflow/tfjs');
//import(/* webpackChunkName: 'pageA' */ './vendors~main.js')

import "regression";
import params from "./params.mjs";
import "./dom_util.mjs";
import localforage from "localforage";
import TFFaceMesh from "./facemesh.mjs";
import Reg from "./ridgeReg.mjs";
import ridgeRegWeighted from "./ridgeWeightedReg.mjs";
import ridgeRegThreaded from "./ridgeRegThreaded.mjs";
import util from "./util.mjs";
import { VideoLiveMonitor } from './videoLiveMonitor.mjs';
import Swal from 'sweetalert2';

function sleep(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

function toFixedNumber(n, digits) {
  let pow = Math.pow(10, digits);
  return Math.round(n * pow) / pow;
}

const webgazer = {};
webgazer.tracker = {};
webgazer.tracker.TFFaceMesh = TFFaceMesh;
webgazer.reg = Reg;
webgazer.reg.RidgeWeightedReg = ridgeRegWeighted.RidgeWeightedReg;
webgazer.reg.RidgeRegThreaded = ridgeRegThreaded.RidgeRegThreaded;
webgazer.util = util;
webgazer.params = params;
webgazer.videoParamsToReport = {height: 0, width: 0, maxHeight: 0, maxWidth: 0, frameRate: 0, maxFrameRate: 0};

//PRIVATE VARIABLES

//video elements
var videoStream = null;
var videoContainerElement = null;
var videoElement = null;
var videoElementCanvas = null;
var faceOverlay = null;
var faceFeedbackBox = null;
var gazeDot = null;
var gazeDotPopped = false;
// Why is this not in webgazer.params ?
var debugVideoLoc = "";

/*
 * Initialises variables used to store accuracy eigenValues
 * This is used by the calibration example file
 */
var xPast50 = new Array(50);
var yPast50 = new Array(50);

// loop parameters
var clockStart = performance.now();
var latestEyeFeatures = null;
var latestGazeData = null;

// FPS throttling for main loop
const targetLoopFPS = 30;
const loopFrameInterval = 1000 / targetLoopFPS;
let lastLoopFrameTime = 0;

/* -------------------------------------------------------------------------- */
webgazer.params.paused = false;

webgazer.params.greedyLearner = false;
webgazer.params.framerate = 30;
webgazer.params.showGazeDot = false;

webgazer.params.getLatestVideoFrameTimestamp = () => {};
webgazer.params.activeCamera = {
  label: "",
  id: "",
};

// Desired camera resolution and frame rate (set via options)
// When set, findBestCameraMode() probes the camera to find the closest available mode.
webgazer.params.desiredCameraResolution = null; // [width, height] or null
webgazer.params.desiredCameraHz = null;         // number or null

/* -------------------------------------------------------------------------- */

// Common webcam resolutions to probe (width x height)
const COMMON_CAMERA_RESOLUTIONS = [
  [320, 240],   // QVGA
  [352, 288],   // CIF
  [640, 360],   // nHD
  [640, 480],   // VGA
  [800, 600],   // SVGA
  [960, 540],   // qHD
  [1024, 576],  // WSVGA
  [1024, 768],  // XGA
  [1280, 720],  // HD 720p
  [1280, 960],
  [1280, 1024], // SXGA
  [1600, 1200], // UXGA
  [1920, 1080], // Full HD 1080p
  [2560, 1440], // QHD
  [3840, 2160], // 4K UHD
];

// Common frame rates to probe
const COMMON_CAMERA_FRAMERATES = [5, 10, 15, 20, 24, 25, 30, 50, 60, 120];

// When getCapabilities() is unavailable (Firefox) we fall back to live
// applyConstraints probing. Each probe forces the OS to reconfigure the
// capture device (150-400ms on macOS), so the sweep is bounded on three
// axes: how many resolutions we try, how long a single probe may take,
// and the total wall-clock budget. Whatever we found when a limit is hit
// is good enough — the cost function already knows how to rank a partial
// candidate set.
const PROBE_MAX_RESOLUTIONS = 4;
const PROBE_TIMEOUT_MS = 500;
const PROBE_BUDGET_MS = 3000;

// getUserMedia errors worth retrying. On macOS the capture device is not
// always released synchronously, so a reopen shortly after a stop can
// fail transiently with these even though the camera is perfectly fine.
const TRANSIENT_GUM_ERRORS = new Set([
  'NotReadableError',
  'AbortError',
  'TrackStartError',
]);

const _startupAborted = () => !!webgazer._startupAbort?.aborted;

const _startupAbortError = () => {
  const err = new Error('Camera startup aborted');
  err.name = 'AbortError';
  err.startupAbort = true;
  return err;
};

const _throwIfStartupAborted = () => {
  if (_startupAborted()) throw _startupAbortError();
};

const _stopStream = stream => {
  try {
    stream?.getTracks?.().forEach(track => track.stop());
  } catch (_) {
    /* already stopped */
  }
};

const _discardIncompleteVideoDom = () => {
  if (webgazer.params.videoIsOn) return;
  try {
    videoContainerElement?.remove();
  } catch (_) {
    /* not in DOM */
  }
  videoContainerElement = null;
  videoElement = null;
  videoElementCanvas = null;
  faceOverlay = null;
  faceFeedbackBox = null;
  videoStream = null;
  if (liveMonitor) {
    try {
      liveMonitor.stop();
    } catch (_) {
      /* ignore */
    }
    liveMonitor = null;
  }
};

// Best mode per (device, desired mode). Reconnects and camera switches
// re-request the same combination, and the answer cannot change while the
// page is open, so probing/deriving it once is enough.
const _cameraModeCache = new Map();

const _cameraModeCacheKey = (deviceId, desiredX, desiredY, desiredHz) =>
  `${deviceId || 'default'}|${desiredX}x${desiredY}@${desiredHz}`;

/**
 * Timings for the camera-startup path, in ms. Read by RemoteCalibrator so
 * the duration to find the first camera can be saved with the results.
 */
webgazer.cameraTiming = {
  firstStreamMs: null,
  probeMs: null,
  probeCount: null,
  probeMethod: null,
};

const _rejectAfter = (ms, message) =>
  new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms));

const _withTimeout = (promise, ms, message) =>
  Promise.race([promise, _rejectAfter(ms, message)]);

/**
 * getUserMedia with a hard timeout and a retry for transient device errors.
 * Without the timeout a wedged capture device leaves the caller awaiting
 * forever, which strands the whole calibration flow on its loading screen.
 */
async function getUserMediaResilient(constraints, {
  timeoutMs = 15000,
  attempts = 3,
  retryDelayMs = 400,
} = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    _throwIfStartupAborted();
    try {
      const stream = await _withTimeout(
        navigator.mediaDevices.getUserMedia(constraints),
        timeoutMs,
        `getUserMedia timed out after ${timeoutMs}ms`,
      );
      if (_startupAborted()) {
        _stopStream(stream);
        throw _startupAbortError();
      }
      return stream;
    } catch (err) {
      lastError = err;
      if (err?.startupAbort) throw err;
      const isTimeout = err instanceof Error && /timed out/.test(err.message);
      const retryable = isTimeout || TRANSIENT_GUM_ERRORS.has(err?.name);
      if (!retryable || attempt === attempts) throw err;
      console.warn(
        `[getUserMedia] ${err?.name || 'Error'} on attempt ${attempt}/${attempts}, retrying in ${retryDelayMs}ms:`,
        err?.message || err,
      );
      await new Promise(r => setTimeout(r, retryDelayMs));
      _throwIfStartupAborted();
    }
  }
  throw lastError;
}

webgazer.getUserMediaResilient = getUserMediaResilient;

/**
 * True when getCapabilities() returned usable width/height ranges. Chrome,
 * Edge and Safari 17.4+ report them; Firefox returns an empty object.
 */
function _capabilitiesAreUsable(capabilities) {
  return !!(
    capabilities &&
    capabilities.width &&
    capabilities.height &&
    Number.isFinite(capabilities.width.max) &&
    Number.isFinite(capabilities.height.max)
  );
}

/**
 * Rank every candidate mode against the desired one using `cameraCost` and
 * return the winner, without touching the camera.
 *
 * A browser that reports capabilities tells us the supported width, height
 * and frame-rate ranges up front, and it will scale or crop to satisfy any
 * size inside those ranges. That makes live probing pure overhead: the same
 * cost function applied to the capability ranges picks the same mode in
 * microseconds instead of ~30 seconds.
 *
 * Out-of-range resolutions are still scored (with the unavailability tax)
 * so a camera that cannot reach the desired size at all still resolves to
 * the closest thing it can do.
 */
function _rankModes(capabilities, desiredX, desiredY, desiredHz, resolutions) {
  const wMin = capabilities?.width?.min ?? 0;
  const wMax = capabilities?.width?.max ?? Infinity;
  const hMin = capabilities?.height?.min ?? 0;
  const hMax = capabilities?.height?.max ?? Infinity;
  const frMin = capabilities?.frameRate?.min ?? 0;
  const frMax = capabilities?.frameRate?.max ?? Infinity;

  const frCandidates = [...new Set([...COMMON_CAMERA_FRAMERATES, desiredHz])]
    .filter(fr => fr >= frMin && fr <= frMax)
    .sort((a, b) => a - b);
  // Every rate we know about is outside the camera's range, so the best it
  // can do is the end of its own range nearest the desired rate.
  if (frCandidates.length === 0) {
    frCandidates.push(Math.min(Math.max(desiredHz, frMin), frMax));
  }

  const modes = [];
  for (const [w, h] of resolutions) {
    const available = w >= wMin && w <= wMax && h >= hMin && h <= hMax;
    if (available) {
      for (const fr of frCandidates) {
        modes.push({ width: w, height: h, frameRate: fr, available: true });
      }
    } else {
      modes.push({ width: w, height: h, frameRate: desiredHz, available: false });
    }
  }

  let bestMode = null;
  let bestCost = Infinity;
  for (const mode of modes) {
    const cost = cameraCost(
      mode.width, mode.height, mode.frameRate,
      desiredX, desiredY, desiredHz,
      mode.available,
    );
    if (cost < bestCost) {
      bestCost = cost;
      bestMode = mode;
    }
  }
  return { bestMode, bestCost, modeCount: modes.length };
}

/**
 * Candidate resolutions: the common list plus the desired size, deduped.
 */
function _resolutionCandidates(desiredX, desiredY) {
  const seen = new Set();
  const out = [];
  for (const [w, h] of [...COMMON_CAMERA_RESOLUTIONS, [desiredX, desiredY]]) {
    const key = `${w}x${h}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push([w, h]);
  }
  return out;
}

/**
 * Apply a chosen mode to a live track, preferring exact sizing but falling
 * back to `ideal` so an unexpected rejection cannot leave the caller
 * without a usable stream.
 */
async function _applyMode(track, mode) {
  try {
    await track.applyConstraints({
      width: { exact: mode.width },
      height: { exact: mode.height },
      frameRate: { ideal: mode.frameRate },
    });
    return true;
  } catch (e) {
    console.warn(
      `[findBestCameraMode] Exact ${mode.width}x${mode.height} rejected, retrying as ideal:`,
      e?.message || e,
    );
    try {
      await track.applyConstraints({
        width: { ideal: mode.width },
        height: { ideal: mode.height },
        frameRate: { ideal: mode.frameRate },
      });
      return true;
    } catch (e2) {
      console.warn(
        '[findBestCameraMode] Could not apply best mode, keeping current settings:',
        e2?.message || e2,
      );
      return false;
    }
  }
}

/**
 * Cost function for evaluating how close a camera mode is to the desired settings.
 * Lower cost is better. Unavailable modes receive a heavy penalty of 100.
 *
 * cost = log10(tryX/desiredX)^2 + log10(tryY/desiredY)^2 + log10(tryHz/desiredHz)^2 + unavailabilityTax
 *
 * @param {number} tryX - Candidate width
 * @param {number} tryY - Candidate height
 * @param {number} tryHz - Candidate frame rate
 * @param {number} desiredX - Desired width
 * @param {number} desiredY - Desired height
 * @param {number} desiredHz - Desired frame rate
 * @param {boolean} available - Whether this mode is available on the camera
 * @returns {number} - Cost value (lower is better)
 */
function cameraCost(tryX, tryY, tryHz, desiredX, desiredY, desiredHz, available) {
  if (tryX <= 0 || tryY <= 0 || tryHz <= 0 || desiredX <= 0 || desiredY <= 0 || desiredHz <= 0) {
    return Infinity;
  }
  const logRatioX = Math.log10(tryX / desiredX);
  const logRatioY = Math.log10(tryY / desiredY);
  const logRatioHz = Math.log10(tryHz / desiredHz);
  const unavailabilityTax = available ? 0 : 100;
  return logRatioX ** 2 + logRatioY ** 2 + logRatioHz ** 2 + unavailabilityTax;
}

/**
 * Bounded applyConstraints probing, for browsers that do not report
 * capabilities (Firefox). Only the resolutions closest to the desired one
 * are tried, frame rate is requested once as `ideal` rather than swept,
 * and the loop abandons ship when it runs out of time.
 */
async function _probeModes(track, desiredX, desiredY, desiredHz, resolutions) {
  const ranked = resolutions
    .map(([w, h]) => ({
      width: w,
      height: h,
      cost: cameraCost(w, h, desiredHz, desiredX, desiredY, desiredHz, true),
    }))
    .sort((a, b) => a.cost - b.cost)
    .slice(0, PROBE_MAX_RESOLUTIONS);

  const modes = [];
  const deadline = performance.now() + PROBE_BUDGET_MS;
  let probeCount = 0;

  for (const candidate of ranked) {
    if (performance.now() > deadline) {
      console.warn('[findBestCameraMode] Probe budget exhausted, using candidates found so far');
      break;
    }
    probeCount++;
    try {
      await _withTimeout(
        track.applyConstraints({
          width: { exact: candidate.width },
          height: { exact: candidate.height },
          frameRate: { ideal: desiredHz },
        }),
        PROBE_TIMEOUT_MS,
        'applyConstraints timed out',
      );
      const settings = track.getSettings();
      modes.push({
        width: settings.width || candidate.width,
        height: settings.height || candidate.height,
        frameRate: settings.frameRate || desiredHz,
        available: true,
      });
    } catch (e) {
      modes.push({
        width: candidate.width,
        height: candidate.height,
        frameRate: desiredHz,
        available: false,
      });
    }
  }

  let bestMode = null;
  let bestCost = Infinity;
  for (const mode of modes) {
    const cost = cameraCost(
      mode.width, mode.height, mode.frameRate,
      desiredX, desiredY, desiredHz,
      mode.available,
    );
    if (cost < bestCost) {
      bestCost = cost;
      bestMode = mode;
    }
  }
  return { bestMode, bestCost, probeCount };
}

async function findBestCameraMode(deviceId, desiredX, desiredY, desiredHz) {
  console.log(`[findBestCameraMode] Searching for best match: ${desiredX}x${desiredY} @ ${desiredHz}Hz`);
  const startTime = performance.now();

  // 1. Open the stream already asking for the desired mode, so a camera
  //    that supports it natively is configured correctly from the start and
  //    the applyConstraints below becomes a no-op.
  const rawDeviceId = typeof deviceId === 'object' ? deviceId?.exact : deviceId;
  const videoConstraints = rawDeviceId
    ? { deviceId: { exact: rawDeviceId } }
    : { facingMode: 'user' };
  const tempStream = await getUserMediaResilient({
    video: {
      ...videoConstraints,
      width: { ideal: desiredX },
      height: { ideal: desiredY },
      frameRate: { ideal: desiredHz },
    },
  });
  try {
    _throwIfStartupAborted();
  const track = tempStream.getVideoTracks()[0];
  const firstStreamMs = performance.now() - startTime;

  // 2. Get capabilities if the browser supports it (Chrome, Edge, Safari 17.4+)
  const capabilities = typeof track.getCapabilities === 'function' ? track.getCapabilities() : null;
  if (capabilities) {
    console.log(`[findBestCameraMode] Capabilities: ` +
      `width ${capabilities.width?.min}-${capabilities.width?.max}, ` +
      `height ${capabilities.height?.min}-${capabilities.height?.max}, ` +
      `frameRate ${capabilities.frameRate?.min}-${capabilities.frameRate?.max}`);
  }

  const cacheKey = _cameraModeCacheKey(rawDeviceId, desiredX, desiredY, desiredHz);
  const resolutions = _resolutionCandidates(desiredX, desiredY);

  let bestMode;
  let bestCost;
  let probeCount = 0;
  let probeMethod;

  const cached = _cameraModeCache.get(cacheKey);
  if (cached) {
    bestMode = { ...cached, available: true };
    bestCost = 0;
    probeMethod = 'cache';
    console.log(`[findBestCameraMode] Reusing cached mode ${cached.width}x${cached.height} @ ${cached.frameRate}Hz`);
  } else if (_capabilitiesAreUsable(capabilities)) {
    const ranked = _rankModes(capabilities, desiredX, desiredY, desiredHz, resolutions);
    bestMode = ranked.bestMode;
    bestCost = ranked.bestCost;
    probeMethod = 'capabilities';
    console.log(`[findBestCameraMode] Ranked ${ranked.modeCount} candidate mode(s) from capabilities (no probing needed)`);
  } else {
    console.log('[findBestCameraMode] Capabilities unavailable, falling back to bounded probing');
    const probed = await _probeModes(track, desiredX, desiredY, desiredHz, resolutions);
    bestMode = probed.bestMode;
    bestCost = probed.bestCost;
    probeCount = probed.probeCount;
    probeMethod = 'probe';
  }

  if (bestMode) {
    console.log(`[findBestCameraMode] Best: ${bestMode.width}x${bestMode.height} @ ${bestMode.frameRate}Hz (cost: ${bestCost.toFixed(4)})`);
  }

  // 3. Apply the winning mode, unless the stream already landed on it.
  const current = track.getSettings();
  const alreadyThere =
    bestMode &&
    current.width === bestMode.width &&
    current.height === bestMode.height &&
    Math.round(current.frameRate || 0) === Math.round(bestMode.frameRate);
  let modeApplied = false;
  if (alreadyThere) {
    modeApplied = true;
    console.log('[findBestCameraMode] Stream already at the best mode, skipping applyConstraints');
  } else if (bestMode && bestMode.available) {
    modeApplied = await _applyMode(track, bestMode);
  }

  const finalSettings = track.getSettings();
  const elapsed = performance.now() - startTime;
  // Only cache a mode that was actually negotiated successfully. Caching
  // whatever the track happened to land on after a failed apply (e.g.
  // another app briefly holding the camera) would pin that degraded mode
  // for every later reconnect/switch until page reload.
  if (modeApplied) {
    _cameraModeCache.set(cacheKey, {
      width: finalSettings.width,
      height: finalSettings.height,
      frameRate: finalSettings.frameRate,
    });
  } else if (probeMethod === 'cache') {
    // The cached mode no longer applies; drop it so the next attempt
    // re-ranks from capabilities instead of retrying a stale answer.
    _cameraModeCache.delete(cacheKey);
  }
  webgazer.cameraTiming = {
    ...webgazer.cameraTiming,
    firstStreamMs: Math.round(firstStreamMs),
    probeMs: Math.round(elapsed),
    probeCount,
    probeMethod,
  };
  console.log(`[findBestCameraMode] Final: ${finalSettings.width}x${finalSettings.height} @ ${finalSettings.frameRate}Hz (via ${probeMethod}, ${probeCount} probe(s), took ${elapsed.toFixed(0)}ms)`);

  const capMaxWidth = capabilities?.width?.max || finalSettings.width;
  const capMaxHeight = capabilities?.height?.max || finalSettings.height;
  const capMaxFrameRate = capabilities?.frameRate?.max || finalSettings.frameRate;

  return {
    stream: tempStream,
    width: finalSettings.width,
    height: finalSettings.height,
    frameRate: finalSettings.frameRate,
    capMaxWidth,
    capMaxHeight,
    capMaxFrameRate
  };
  } catch (err) {
    if (err?.startupAbort || _startupAborted()) _stopStream(tempStream);
    throw err;
  }
}

/* -------------------------------------------------------------------------- */

let videoInputs = [];

// registered callback for loop
var nopCallback = function (data) {};
var callback = nopCallback;

//Types that regression systems should handle
//Describes the source of data so that regression systems may ignore or handle differently the various generating events
var eventTypes = ["click", "move"];

//movelistener timeout clock parameters
var moveClock = performance.now();
//currently used tracker and regression models, defaults to clmtrackr and linear regression
var curTracker = new webgazer.tracker.TFFaceMesh();
var regs = [new webgazer.reg.RidgeReg()];
// var blinkDetector = new webgazer.BlinkDetector();

//lookup tables
var curTrackerMap = {
  TFFacemesh: function () {
    return new webgazer.tracker.TFFaceMesh();
  },
  TFFacemesh_unrefined_landmarks: function () {
    return new webgazer.tracker.TFFaceMesh(false);
  },
};
var regressionMap = {
  ridge: function () {
    return new webgazer.reg.RidgeReg();
  },
  weightedRidge: function () {
    return new webgazer.reg.RidgeWeightedReg();
  },
  threadedRidge: function () {
    return new webgazer.reg.RidgeRegThreaded();
  },
};

//localstorage name
var localstorageDataLabel = "webgazerGlobalData";
var localstorageSettingsLabel = "webgazerGlobalSettings";
//settings object for future storage of settings
var settings = {};
var data = [];
var defaults = {
  data: [],
  settings: {},
};

const  streamHasVideo = (stream) => {
  return !!stream && stream.getVideoTracks().length > 0;
}

const hasLiveVideo = (stream) => {
  if (!stream) return false;
  const t = stream.getVideoTracks()[0];
  return !!t && t.readyState === "live" && !t.muted;
}

let liveMonitor = null;
let _isReconnecting = false;
let _isSwappingCamera = false;

function startOrUpdateLiveMonitor(stream) {
  console.log('[CameraReconnect] startOrUpdateLiveMonitor called, liveMonitor exists:', !!liveMonitor, '_isReconnecting:', _isReconnecting, '_isSwappingCamera:', _isSwappingCamera);
  if (liveMonitor) {
    liveMonitor.updateStream(stream, videoElement);
    return;
  }
  liveMonitor = new VideoLiveMonitor(stream, videoElement);
  liveMonitor.onChange((snap) => {
    console.log('[CameraReconnect] onChange callback fired, status:', snap.status, '_isReconnecting:', _isReconnecting, '_isSwappingCamera:', _isSwappingCamera);
    if ((snap.status === 'ended' || snap.status === 'inactive') && !_isReconnecting && !_isSwappingCamera) {
      console.error('[CameraReconnect] >>> TRIGGERING DISCONNECT POPUP <<<', snap);
      showCameraReconnectionPopup(
        `Camera status: ${snap.status} (track: ${snap.trackReadyState}, stream active: ${snap.streamActive})`
      );
    }
  });
  liveMonitor.start();
}

//PRIVATE FUNCTIONS

/**
 * Computes the size of the face overlay validation box depending on the size of the video preview window.
 * @returns {Object} The dimensions of the validation box as top, left, width, height.
 */
webgazer.computeValidationBoxSize = function () {
  var vw = videoElement.videoWidth;
  var vh = videoElement.videoHeight;
  var pw = parseInt(Math.min(videoElement.offsetWidth,videoContainerElement.offsetWidth));
  var ph = parseInt(videoContainerElement.offsetHeight);

  if (!pw || !ph) {
    const styleW = parseInt(videoContainerElement.style.width);
    const styleH = parseInt(videoContainerElement.style.height);
    pw = styleW || webgazer.params.videoViewerWidth;
    ph = styleH || webgazer.params.videoViewerHeight;
  }

  // Find the size of the box.
  // Pick the smaller of the two video preview sizes
  var smaller = Math.min(vw, vh);
  var larger = Math.max(vw, vh);

  // Overall scalar
  var scalar = vw == larger ? pw / vw : ph / vh;

  // Multiply this by 2/3, then adjust it to the size of the preview
  var boxSize = smaller * webgazer.params.faceFeedbackBoxRatio * scalar;

  // Set the boundaries of the face overlay validation box based on the preview
  var topVal = (ph - boxSize) / 2;
  var leftVal = (pw - boxSize) / 2;

  // top, left, width, height
  return [topVal, leftVal, boxSize, boxSize];
};

let _w,
  _h,
  _smaller,
  _boxSize,
  _topBound,
  _leftBound,
  _rightBound,
  _bottomBound;
let _eyeLX, _eyeLY, _eyeRX, _eyeRY;
let hasBounds = false;
let gettingBounds = false;

function _helper_getBounds() {
  gettingBounds = true;
  setTimeout(() => {
    _w = videoElement.videoWidth;
    _h = videoElement.videoHeight;

    // Find the size of the box.
    // Pick the smaller of the two video preview sizes
    _smaller = Math.min(_w, _h);
    _boxSize = _smaller * webgazer.params.faceFeedbackBoxRatio;

    // Set the boundaries of the face overlay validation box based on the preview
    _topBound = (_h - _boxSize) / 2;
    _leftBound = (_w - _boxSize) / 2;
    _rightBound = _leftBound + _boxSize;
    _bottomBound = _topBound + _boxSize;

    hasBounds = true;
    gettingBounds = false;
  }, 500);
}

// TODO WebGazer doesn't provide correct validation feedback
/**
 * Checks if the pupils are in the position box on the video
 */
function checkEyesInValidationBox() {
  if (faceFeedbackBox !== null && latestEyeFeatures) {
    if (!hasBounds && !gettingBounds) _helper_getBounds();

    // Get the x and y positions of the left and right eyes
    _eyeLX = _w - latestEyeFeatures.left.imagex;
    _eyeRX = _w - latestEyeFeatures.right.imagex;
    _eyeLY = latestEyeFeatures.left.imagey;
    _eyeRY = latestEyeFeatures.right.imagey;

    if (
      hasBounds &&
      _eyeLX > _leftBound &&
      _eyeLX < _rightBound &&
      _eyeRX > _leftBound &&
      _eyeRX < _rightBound &&
      _eyeLY > _topBound &&
      _eyeLY < _bottomBound &&
      _eyeRY > _topBound &&
      _eyeRY < _bottomBound
    ) {
      faceFeedbackBox.style.border = "solid gray 2px";
    } else {
      faceFeedbackBox.style.border = "solid red 4px";
    }
  } else {
    if(!latestEyeFeatures) {
      /*
      sometimes latestEyeFeatures is false because the length 
      of "predictions" inside "getEyePatches" is zero (no face detected by the model) even 
      though the face is in the camera view. So this stops the red border from appearing in that case
      */
      faceFeedbackBox.style.border = "solid gray 2px";
    }
    else faceFeedbackBox.style.border = "solid red 4px";}
}

/**
 * This draws the point (x,y) onto the canvas in the HTML
 * @param {colour} colour - The colour of the circle to plot
 * @param {x} x - The x co-ordinate of the desired point to plot
 * @param {y} y - The y co-ordinate of the desired point to plot
 */
function drawCoordinates(colour, x, y) {
  var ctx = document.getElementById("gaze-accuracy-canvas").getContext("2d");
  ctx.fillStyle = colour; // Red color
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2, true);
  ctx.fill();
}

/**
 * Gets the pupil features by following the pipeline which threads an eyes object through each call:
 * curTracker gets eye patches -> blink detector -> pupil detection
 * @param {Canvas} canvas - a canvas which will have the video drawn onto it
 * @param {Number} width - the width of canvas
 * @param {Number} height - the height of canvas
 */
function getPupilFeatures(canvas, width, height) {
  if (!canvas) {
    return;
  }
  try {
    return curTracker.getEyePatches(videoElement, canvas, width, height);
  } catch (err) {
    console.log("can't get pupil features ", err);
    return null;
  }
}

/**
 * Gets the most current frame of video and paints it to a resized version of the canvas with width and height
 * @param {Canvas} canvas - the canvas to paint the video on to
 * @param {Number} width - the new width of the canvas
 * @param {Number} height - the new height of the canvas
 */
function paintCurrentFrame(canvas, width, height) {
  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== height) {
    canvas.height = height;
  }

  var ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
}

/**
 * Paints the video to a canvas and runs the prediction pipeline to get a prediction
 * @param {Number|undefined} regModelIndex - The prediction index we're looking for
 * @returns {*}
 */
async function getPrediction(regModelIndex, eyeFeatures = latestEyeFeatures) {
  var predictions = [];

  if (regs.length === 0) {
    console.log("regression not set, call setRegression()");
    return null;
  }

  for (var reg in regs) {
    predictions.push(regs[reg].predict(eyeFeatures));
  }

  if (regModelIndex !== undefined) {
    return predictions[regModelIndex] === null
      ? null
      : {
          x: predictions[regModelIndex].x,
          y: predictions[regModelIndex].y,
          eyeFeatures: eyeFeatures,
        };
  } else {
    return predictions.length === 0 || predictions[0] === null
      ? null
      : {
          x: predictions[0].x,
          y: predictions[0].y,
          eyeFeatures: eyeFeatures,
          all: predictions,
        };
  }
}

/* -------------------------------------------------------------------------- */
/*                                    LOOP                                    */
/* -------------------------------------------------------------------------- */

/**
 * Runs every available animation frame if webgazer is not paused
 */
var smoothingVals = new util.DataWindow(4);
var k = 0;

let _now = null;
let _last = -1;

// From getting the video frame to get data
let _oneLoopFinished = true;

async function gazePrep(forcedPrep = false) {
  paintCurrentFrame(
    videoElementCanvas,
    videoElementCanvas.width,
    videoElementCanvas.height
  );

  // [20200617 xk] TODO: this call should be made async somehow. will take some work.
  if (!webgazer.params.paused || forcedPrep) {
    latestEyeFeatures = await getPupilFeatures(
      videoElementCanvas,
      videoElementCanvas.width,
      videoElementCanvas.height
    );
    // console.log(videoElementCanvas, videoElementCanvas.width, videoElementCanvas.height);
  }

  // We don't need these for getGazeNow() right?
  // Draw face overlay
  if (webgazer.params.showFaceOverlay) {
    // Get tracker object
    var tracker = webgazer.getTracker();
    faceOverlay
      .getContext("2d")
      .clearRect(0, 0, videoElement.videoWidth, videoElement.videoHeight);
    tracker.drawFaceOverlay(
      faceOverlay.getContext("2d"),
      tracker.getPositions()
    );
  }

  // Feedback box
  // Check that the eyes are inside of the validation box
  if (webgazer.params.showFaceFeedbackBox) checkEyesInValidationBox();
}

function gazePrepForGetGazeNow1() {
  paintCurrentFrame(
    videoElementCanvas,
    videoElementCanvas.width,
    videoElementCanvas.height
  );
}

async function gazePrepForGetGazeNow2() {
  latestEyeFeatures = await getPupilFeatures(
    videoElementCanvas,
    videoElementCanvas.width,
    videoElementCanvas.height
  );
  return latestEyeFeatures;
}

async function loop(currentTime) {
  _now = window.performance.now();

  // Throttle the entire loop to 30fps
  if (currentTime - lastLoopFrameTime >= loopFrameInterval) {
    if (webgazer.params.videoIsOn) {
      // [20200617 XK] TODO: there is currently lag between the camera input and the face overlay. This behavior
      // is not seen in the facemesh demo. probably need to optimize async implementation. I think the issue lies
      // in the implementation of getPrediction().

      // Paint the latest video frame into the canvas which will be analyzed by WebGazer
      // [20180729 JT] Why do we need to do this? clmTracker does this itself _already_, which is just duplicating the work.
      // Is it because other trackers need a canvas instead of an img/video element?
      if (_oneLoopFinished) {
        _oneLoopFinished = false;
        webgazer.params.getLatestVideoFrameTimestamp(performance.now());
      }
      await gazePrep();
    }

    if (!webgazer.params.paused) {
      if (_now - _last >= 1000 / webgazer.params.framerate) {
        _last = _now;

        // Get gaze prediction (ask clm to track; pass the data to the regressor; get back a prediction)
        latestGazeData = getPrediction();
        // Count time
        // var elapsedTime = performance.now() - clockStart;

        latestGazeData = await latestGazeData;

        // [20200623 xk] callback to function passed into setGazeListener(fn)
        callback(latestGazeData);
        _oneLoopFinished = true;

        if (latestGazeData) {
          // [20200608 XK] Smoothing across the most recent 4 predictions, do we need this with Kalman filter?
          smoothingVals.push(latestGazeData);
          var x = 0;
          var y = 0;
          var len = smoothingVals.length;
          for (var d in smoothingVals.data) {
            x += smoothingVals.get(d).x;
            y += smoothingVals.get(d).y;
          }

          var pred = util.bound({ x: x / len, y: y / len });

          if (webgazer.params.storingPoints) {
            // drawCoordinates('blue', pred.x, pred.y); //draws the previous predictions
            // store the position of the past fifty occuring tracker preditions
            webgazer.storePoints(pred.x, pred.y, k);
            ++k;
            if (k == 50) k = 0;
          }

          gazeDot.style.opacity = "";
          gazeDot.style.left = `${pred.x}px`;
          gazeDot.style.top = `${pred.y}px`;
        }
      }
    } else {
      if (gazeDot && !gazeDotPopped) gazeDot.style.opacity = "0";
    }

    lastLoopFrameTime = currentTime;
  }

  requestAnimationFrame(loop);
}

//is problematic to test
//because latestEyeFeatures is not set in many cases

/**
 * Records screen position data based on current pupil feature and passes it
 * to the regression model.
 * @param {Number} x - The x screen position
 * @param {Number} y - The y screen position
 * @param {String} eventType - The event type to store
 * @returns {null}
 */
var recordScreenPosition = function (x, y, eventType) {
  if (webgazer.params.paused) {
    return;
  }
  if (regs.length === 0) {
    console.log("regression not set, call setRegression()");
    return null;
  }
  for (var reg in regs) {
    if (latestEyeFeatures)
      regs[reg].addData(latestEyeFeatures, [x, y], eventType);
  }
};

/**
 * Records click data and passes it to the regression model
 * @param {Event} event - The listened event
 */
var clickListener = async function (event) {
  recordScreenPosition(event.clientX, event.clientY, eventTypes[0]); // eventType[0] === 'click'

  if (webgazer.params.saveDataAcrossSessions) {
    // Each click stores the next data point into localforage.
    await setGlobalData();

    // // Debug line
    // console.log('Model size: ' + JSON.stringify(await localforage.getItem(localstorageDataLabel)).length / 1000000 + 'MB');
  }
};

/**
 * Records mouse movement data and passes it to the regression model
 * @param {Event} event - The listened event
 */
var moveListener = function (event) {
  if (webgazer.params.paused) {
    return;
  }

  var now = performance.now();
  if (now < moveClock + webgazer.params.moveTickSize) {
    return;
  } else {
    moveClock = now;
  }
  recordScreenPosition(event.clientX, event.clientY, eventTypes[1]); //eventType[1] === 'move'
};

/**
 * Add event listeners for mouse click and move.
 */
var addMouseEventListeners = function (options = {}) {
  // third argument set to true so that we get event on 'capture' instead of 'bubbling'
  // this prevents a client using event.stopPropagation() preventing our access to the click
  options = Object.assign(
    {
      click: true,
      move: true,
    },
    options
  );
  if (options.click) document.addEventListener("click", clickListener, true);
  if (options.move) document.addEventListener("mousemove", moveListener, true);
};

/**
 * Remove event listeners for mouse click and move.
 */
var removeMouseEventListeners = function (options = {}) {
  // must set third argument to same value used in addMouseEventListeners
  // for this to work.
  options = Object.assign(
    {
      click: true,
      move: true,
    },
    options
  );
  if (options.click) document.removeEventListener("click", clickListener, true);
  if (options.move)
    document.removeEventListener("mousemove", moveListener, true);
};

/**
 * Loads the global data and passes it to the regression model
 */
async function loadGlobalData() {
  // Get settings object from localforage
  // [20200611 xk] still unsure what this does, maybe would be good for Kalman filter settings etc?
  settings = await localforage.getItem(localstorageSettingsLabel);
  settings = settings || defaults;

  // Get click data from localforage
  var loadData = await localforage.getItem(localstorageDataLabel);
  loadData = loadData || defaults;

  // Set global var data to newly loaded data
  data = loadData;

  // Load data into regression model(s)
  for (var reg in regs) {
    regs[reg].setData(loadData);
  }

  console.log("loaded stored data into regression model");
}

/**
 * Adds data to localforage
 */
async function setGlobalData() {
  // Grab data from regression model
  var storeData = regs[0].getData() || data; // Array

  // Store data into localforage
  localforage.setItem(localstorageSettingsLabel, settings); // [20200605 XK] is 'settings' ever being used?
  localforage.setItem(localstorageDataLabel, storeData);
  //TODO data should probably be stored in webgazer object instead of each regression model
  //     -> requires duplication of data, but is likely easier on regression model implementors
}

/**
 * Clears data from model and global storage
 */
function clearData() {
  // Removes data from localforage
  localforage.clear();

  // Removes data from regression model
  for (var reg in regs) {
    regs[reg].init();
  }
}

/**
 * Initializes all needed dom elements and begins the loop
 * @param {URL} stream - The video stream to use
 */
async function init(initMode = "all", stream) {
  //////////////////////////
  // Video and video preview
  //////////////////////////

  if (!webgazer.params.videoIsOn) {
    // used for webgazer.stopVideo() and webgazer.setCameraConstraints()
    videoStream = stream;

    // create a video element container to enable customizable placement on the page
    videoContainerElement = document.createElement("div");
    videoContainerElement.id = webgazer.params.videoContainerId;
    // Start hidden - will be shown when needed (prevents flash before popup)
    videoContainerElement.style.display = "none";
    // videoContainerElement.style.visibility = webgazer.params.showVideo ? 'visible' : 'hidden';
    videoContainerElement.style.opacity = webgazer.params.showVideo ? 0.8 : 0;
    // videoContainerElement.style.position = 'fixed';
    videoContainerElement.style.left = "10px";
    videoContainerElement.style.bottom = "10px";
    videoContainerElement.style.width = webgazer.params.videoViewerWidth + "px";
    videoContainerElement.style.height =
      webgazer.params.videoViewerHeight + "px";

    videoElement = document.createElement("video");
    videoElement.setAttribute("playsinline", "");
    videoElement.id = webgazer.params.videoElementId;
    videoElement.srcObject = stream;
    videoElement.autoplay = true;
    videoElement.style.display = "block";
    // videoElement.style.visibility = webgazer.params.showVideo ? 'visible' : 'hidden';
    videoElement.style.position = "absolute";
    // We set these to stop the video appearing too large when it is added for the very first time
    videoElement.style.width = webgazer.params.videoViewerWidth + "px";
    videoElement.style.height = webgazer.params.videoViewerHeight + "px";
    
    // Enforce landscape orientation for video element
    videoElement.addEventListener('loadedmetadata', function() {
      if (videoElement.videoWidth < videoElement.videoHeight) {
        // If video is portrait, rotate it to landscape
        videoElement.style.transform = videoElement.style.transform + ' rotate(90deg)';
        // Adjust container size for rotated video
        videoContainerElement.style.width = webgazer.params.videoViewerHeight + "px";
        videoContainerElement.style.height = webgazer.params.videoViewerWidth + "px";
      }
    });

    // Canvas for drawing video to pass to clm tracker
    videoElementCanvas = document.createElement("canvas");
    videoElementCanvas.id = webgazer.params.videoElementCanvasId;
    videoElementCanvas.style.display = "block";
    videoElementCanvas.style.opacity = 0;
    // videoElementCanvas.style.visibility = 'hidden';

    // Face overlay
    // Shows the CLM tracking result
    faceOverlay = document.createElement("canvas");
    faceOverlay.id = webgazer.params.faceOverlayId;
    faceOverlay.style.display = webgazer.params.showFaceOverlay
      ? "block"
      : "none";
    faceOverlay.style.position = "absolute";

    // Mirror video feed
    if (webgazer.params.mirrorVideo) {
      videoElement.style.setProperty("-moz-transform", "scale(-1, 1)");
      videoElement.style.setProperty("-webkit-transform", "scale(-1, 1)");
      videoElement.style.setProperty("-o-transform", "scale(-1, 1)");
      videoElement.style.setProperty("transform", "scale(-1, 1)");
      videoElement.style.setProperty("filter", "FlipH");
      faceOverlay.style.setProperty("-moz-transform", "scale(-1, 1)");
      faceOverlay.style.setProperty("-webkit-transform", "scale(-1, 1)");
      faceOverlay.style.setProperty("-o-transform", "scale(-1, 1)");
      faceOverlay.style.setProperty("transform", "scale(-1, 1)");
      faceOverlay.style.setProperty("filter", "FlipH");
    }

    // Feedback box
    // Lets the user know when their face is in the middle
    faceFeedbackBox = document.createElement("canvas");
    faceFeedbackBox.id = webgazer.params.faceFeedbackBoxId;
    faceFeedbackBox.style.display = webgazer.params.showFaceFeedbackBox
      ? "block"
      : "none";
    faceFeedbackBox.style.border = "solid red 4px";
    faceFeedbackBox.style.position = "absolute";

    // Add other preview/feedback elements to the screen once the video has shown and its parameters are initialized
    videoContainerElement.appendChild(videoElement);
    document.body.appendChild(videoContainerElement);
    function setupPreviewVideo(e) {
      // All video preview parts have now been added, so set the size both internally and in the preview window.
      setInternalVideoBufferSizes(
        videoElement.videoWidth,
        videoElement.videoHeight
      );
      webgazer.setVideoViewerSize(
        webgazer.params.videoViewerWidth,
        webgazer.params.videoViewerHeight
      );

      videoContainerElement.appendChild(videoElementCanvas);
      webgazer.videoCanvas = videoElementCanvas; // !
      videoContainerElement.appendChild(faceOverlay);
      videoContainerElement.appendChild(faceFeedbackBox);

      // Run this only once, so remove the event listener
      e.target.removeEventListener(e.type, setupPreviewVideo);
    }
    videoElement.addEventListener("timeupdate", setupPreviewVideo);

    startOrUpdateLiveMonitor(stream);
  }

  if (initMode != "video") {
    // Gaze dot
    // Starts offscreen
    gazeDot = document.createElement("div");
    gazeDot.id = webgazer.params.gazeDotId;
    gazeDot.style.display = webgazer.params.showGazeDot ? "block" : "none";
    // gazeDot.style.position = 'fixed';
    // gazeDot.style.zIndex = 99999;
    // TODO Customizable width and height
    gazeDot.style.width = "10px";
    gazeDot.style.height = "10px";
    gazeDot.style.left = "-10px";
    gazeDot.style.top = "-10px"; // Width and height are 10px by default
    gazeDot.style.transform = `translate(-5px, -5px)`;

    document.body.appendChild(gazeDot);

    addMouseEventListeners();

    //BEGIN CALLBACK LOOP
    webgazer.params.paused = false;
    clockStart = performance.now();
  }

  // load the distance model
  await curTracker.loadModel();
  _throwIfStartupAborted();
  await loop();
}

/**
 * Initializes navigator.mediaDevices.getUserMedia
 * depending on the browser capabilities
 *
 * @return Promise
 */
function setUserMediaVariable() {
  if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
  }

  if (navigator.mediaDevices.getUserMedia === undefined) {
    navigator.mediaDevices.getUserMedia = function (constraints) {
      // gets the alternative old getUserMedia is possible
      var getUserMedia =
        navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

      // set an error message if browser doesn't support getUserMedia
      if (!getUserMedia) {
        return Promise.reject(
          new Error(
            "Unfortunately, your browser does not support access to the webcam through the getUserMedia API. Try to use the latest version of Google Chrome, Mozilla Firefox, Opera, or Microsoft Edge instead."
          )
        );
      }

      // uses navigator.getUserMedia for older browsers
      return new Promise(function (resolve, reject) {
        getUserMedia.call(navigator, constraints, resolve, reject);
      });
    };
  }
}

//PUBLIC FUNCTIONS - CONTROL

/**
 * Starts all state related to webgazer -> dataLoop, video collection, click listener
 * If starting fails, call `onFail` param function.
 * @param {Function} onFail - Callback to call in case it is impossible to find user camera
 * @returns {*}
 */
webgazer.begin = function (onFail, signal) {
  // if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.chrome){
  //   alert("WebGazer works only over https. If you are doing local development, you need to run a local server.");
  // }

  // Load model data stored in localforage.
  // if (webgazer.params.saveDataAcrossSessions) {
  //   loadGlobalData();
  // }

  // onFail = onFail || function() {console.log('No stream')};

  // if (debugVideoLoc) {
  //   init(debugVideoLoc);
  //   return webgazer;
  // }

  return webgazer._begin(false, onFail, signal);
};

/**
 * Start the video element.
 */
webgazer.beginVideo = function (onFail, signal) {
  return webgazer._begin(true, onFail, signal);
};

/* ------------------------------ Video switch ------------------------------ */

const _foldString = (str) => {
  if (str.length < 8) return str;
  else return str.slice(0, 8) + "...";
};

const _setUpActiveCameraSwitch = (inputs) => {
  const parent = videoContainerElement;

  const selectElement = document.createElement("select");
  selectElement.className = selectElement.id = "webgazer-videoinput-select";
  selectElement.name = "videoinput";
  inputs.forEach((input, ind) => {
    selectElement.innerHTML += `<option value="${
      input.deviceId + "%" + input.label
    }"${ind === 0 ? " selected" : ""} name="1">${_foldString(
      input.label
    )}</option>`;
  });

  selectElement.onchange = (e) => {
    const [id, label] = selectElement.value.split("%");
    webgazer.params.activeCamera.label = label;
    webgazer.params.activeCamera.id = id;

    webgazer.setCameraConstraints(
      _setUpConstraints(webgazer.params.camConstraints)
    );
  };

  parent.appendChild(selectElement);
};

const _gotSources = (sources) => {
  videoInputs = [];
  let preferredLabel, preferredDeviceId;
  let nonIPhoneCameras = [];

  sources.forEach((device) => {
    if (device.kind === "videoinput") {
      videoInputs.push(device);
      
      // Separate FaceTime cameras and non-iPhone cameras
      if (device.label.includes("FaceTime")) {
        preferredLabel = device.label;
        preferredDeviceId = device.deviceId;
      } else if (!device.label.includes("iPhone")) {
        // Only include non-iPhone cameras as fallback
        nonIPhoneCameras.push(device);
      }
    }
  });

  if (videoInputs.length) {
    // If we found a FaceTime camera, use it
    if (preferredLabel && preferredDeviceId) {
      webgazer.params.activeCamera.label = preferredLabel;
      webgazer.params.activeCamera.id = preferredDeviceId;
    } else if (nonIPhoneCameras.length > 0) {
      // Use the first non-iPhone camera as fallback
      webgazer.params.activeCamera.label = nonIPhoneCameras[0].label;
      webgazer.params.activeCamera.id = nonIPhoneCameras[0].deviceId;
    } else {
      // Last resort: use any camera (including iPhone)
      webgazer.params.activeCamera.label = videoInputs[0].label;
      webgazer.params.activeCamera.id = videoInputs[0].deviceId;
    }
  }
};

const _setUpConstraints = (originalConstraints) => {
  if (!webgazer.params.activeCamera.id) return originalConstraints;
  
  // Enforce landscape orientation
  const landscapeConstraints = {
    ...originalConstraints.video,
    deviceId: webgazer.params.activeCamera.id,
    // Ensure landscape aspect ratio (width > height)
    aspectRatio: { min: 1.33, ideal: 1.78, max: 2.33 },
    // Set minimum width to be greater than minimum height
    width: { min: 640, ideal: 1920, max: 7680 },
    height: { min: 360, ideal: 1080, max: 4320 }
  };
  
  return {
    video: landscapeConstraints,
  };
};

webgazer._begin = function (videoOnly, onVideoFail, signal) {
  if (signal) webgazer._startupAbort = signal;

  // SETUP VIDEO ELEMENTS
  // Sets .mediaDevices.getUserMedia depending on browser
  if (!webgazer.params.videoIsOn) {
    setUserMediaVariable();

    return new Promise(async (resolve, reject) => {
      let stream;
      // `onVideoFail` shows a blocking error dialog, so it must fire at most
      // once per attempt no matter which layer catches the failure.
      let videoFailReported = false;
      const reportVideoFail = inputs => {
        if (videoFailReported) return;
        videoFailReported = true;
        if (typeof onVideoFail === 'function') onVideoFail(inputs);
      };

      try {
        _throwIfStartupAborted();
        if (
          typeof navigator.mediaDevices !== "undefined" &&
          typeof navigator.mediaDevices.enumerateDevices === "function"
        ) {
          const enumerateStart = performance.now();
          const availableDevices =
            await navigator.mediaDevices.enumerateDevices();
          _throwIfStartupAborted();
          webgazer.cameraTiming = {
            ...webgazer.cameraTiming,
            enumerateMs: Math.round(performance.now() - enumerateStart),
          };
          // pick the default source
          _gotSources(availableDevices);
          // no valid video input devices
          if (videoInputs.length === 0) {
            reportVideoFail(videoInputs);
            throw new Error("We can't find any video input devices.");
          }

          // await navigator.mediaDevices.getUserMedia({
          //   video: {
          //     facingMode: 'user',
          //     deviceId: preferredDeviceId,
          //   }
          // })

          try {
            const desiredRes = webgazer.params.desiredCameraResolution;
            const desiredHz = webgazer.params.desiredCameraHz;

            if (desiredRes && desiredHz) {
              // Use cost-function-based probing to find the best available mode
              console.log(`Using findBestCameraMode: desired ${desiredRes[0]}x${desiredRes[1]} @ ${desiredHz}Hz`);
              const result = await findBestCameraMode(null, desiredRes[0], desiredRes[1], desiredHz);
              stream = result.stream;
              _throwIfStartupAborted();

              webgazer.videoParamsToReport = {
                height: result.height,
                width: result.width,
                maxHeight: result.capMaxHeight,
                maxWidth: result.capMaxWidth,
                frameRate: result.frameRate,
                maxFrameRate: result.capMaxFrameRate
              };
              console.log(`Camera resolution (probed): ${result.width}x${result.height} @ ${result.frameRate}Hz, capability max: ${result.capMaxWidth}x${result.capMaxHeight}`);
            } else {
              // No desired resolution specified
              const gumStart = performance.now();
              stream = await getUserMediaResilient({
                video: {
                  width: { ideal: 1920 },
                  height: { ideal: 1080 },
                  aspectRatio: { min: 1.33, ideal: 1.78, max: 2.33 },
                  facingMode: "user"
                }
              });
              _throwIfStartupAborted();
              webgazer.cameraTiming = {
                ...webgazer.cameraTiming,
                firstStreamMs: Math.round(performance.now() - gumStart),
                probeMs: 0,
                probeCount: 0,
                probeMethod: 'none',
              };

              const videoTrack = stream.getVideoTracks()[0];
              const settings = videoTrack.getSettings();
              const width = settings.width;
              const height = settings.height;

              const cap = typeof videoTrack.getCapabilities === 'function' ? videoTrack.getCapabilities() : null;
              const capMaxWidth = cap?.width?.max || width;
              const capMaxHeight = cap?.height?.max || height;
              const actualFrameRate = settings.frameRate || 0;
              const capMaxFrameRate = cap?.frameRate?.max || actualFrameRate;
              console.log(`Camera resolution: ${width}x${height} @ ${actualFrameRate}Hz, capability max: ${capMaxWidth}x${capMaxHeight} @ ${capMaxFrameRate}Hz`);

              webgazer.videoParamsToReport = { 
                height, 
                width,
                maxHeight: capMaxHeight,
                maxWidth: capMaxWidth,
                frameRate: actualFrameRate,
                maxFrameRate: capMaxFrameRate
              };
            }
          } catch (error) {
            if (!error?.startupAbort) reportVideoFail(videoInputs);
            throw error;
          }

          // When this promise resolves, the video container is in the DOM —
          // callers no longer poll for it.
          await init(videoOnly ? "video" : "all", stream);
          _throwIfStartupAborted();
          ////
          webgazer.params.videoIsOn = true;
          ////
          resolve(webgazer);
        } else {
          reportVideoFail([]);
          throw new Error("navigator.mediaDevices is unavailable.");
        }
      } catch (err) {
        _stopStream(stream);
        _discardIncompleteVideoDom();

        // Pass the real device list: with cameras present the dialog says
        // "camera use denied" instead of the misleading "no camera" that an
        // empty array produces (e.g. on a model-download or network error).
        if (!err?.startupAbort) reportVideoFail(videoInputs);

        // Log the error itself: JSON.stringify() turns a DOMException into
        // "{}", which is what made past camera failures unreadable.
        console.error('[webgazer._begin] Failed to start video:', err);

        reject(err);
      }
    });
  } else {
    // Video is ON
    // e.g. tracking viewing distance already.
    // Returned so the caller can await it -- init() loads the face model,
    // and an unhandled rejection from that download aborts the experiment.
    return init("gaze");
  }
};

/**
 * Checks if webgazer has finished initializing after calling begin()
 * [20180729 JT] This seems like a bad idea for how this function should be implemented.
 * @returns {boolean} if webgazer is ready
 */
webgazer.isReady = function () {
  if (videoElementCanvas === null) {
    return false;
  }
  return videoElementCanvas.width > 0;
};

/**
 * Stops collection of data and predictions
 * @returns {webgazer} this
 */
webgazer.pause = function () {
  webgazer.params.paused = true;
  return webgazer;
};

/* -------------------------------------------------------------------------- */

webgazer.stopLearning = function (options) {
  removeMouseEventListeners(options);
  return webgazer;
};

webgazer.startLearning = function (options) {
  addMouseEventListeners(options);
  return webgazer;
};

/* -------------------------------------------------------------------------- */

/**
 * Resumes collection of data and predictions if paused
 * @returns {webgazer} this
 */
webgazer.resume = async function () {
  if (!webgazer.params.paused) {
    return webgazer;
  }
  webgazer.params.paused = false;
  _oneLoopFinished = true;

  // in case called getGazeNow() during the pause
  if (gazeDotPopInterval.current) {
    clearInterval(gazeDotPopInterval.current);
    gazeDotPopInterval.current = undefined;
    gazeDotPopped = false;

    gazeDot.style.backgroundColor = "";
    gazeDot.style.opacity = "";
  }

  await loop();
  return webgazer;
};

/**
 * stops collection of data and removes dom modifications, must call begin() to reset up
 * @return {webgazer} this
 */
webgazer.end = function (endAll = false) {
  if (endAll) {
    if (liveMonitor) {
      liveMonitor.stop();
      liveMonitor = null;
    }

    smoothingVals = new util.DataWindow(4);
    k = 0;
    _now = null;
    _last = -1;

    hasBounds = false;

    webgazer.params.videoIsOn = false;
    setTimeout(() => {
      webgazer.stopVideo();

      // remove video element and canvas
      videoContainerElement.remove();
    }, 500);
  }
  return webgazer;
};

/**
 * Stops the video camera from streaming and removes the video outlines
 * @return {webgazer} this
 */
webgazer.stopVideo = function () {
  if (liveMonitor) {
    liveMonitor.stop();
    liveMonitor = null;
  }

  // Stops the video from streaming
  videoStream.getTracks()[0].stop();

  // Removes the outline of the face
  // document.body.removeChild( faceOverlay );

  // Removes the box around the face
  // document.body.removeChild( faceFeedbackBox );

  return webgazer;
};

//PUBLIC FUNCTIONS - DEBUG

/**
 * Returns if the browser is compatible with webgazer
 * @return {boolean} if browser is compatible
 */
webgazer.detectCompatibility = function () {
  var getUserMedia =
    navigator.mediaDevices.getUserMedia ||
    navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia;

  return getUserMedia !== undefined;
};

/**
 * Set whether to show any of the video previews (camera, face overlay, feedback box).
 * If true: visibility depends on corresponding params (default all true).
 * If false: camera, face overlay, feedback box are all hidden
 * @param {bool} val
 * @return {webgazer} this
 */
webgazer.showVideoPreview = function (val) {
  webgazer.params.showVideoPreview = val;
  webgazer.showVideo(val && webgazer.params.showVideo);
  webgazer.showFaceOverlay(val && webgazer.params.showFaceOverlay);
  webgazer.showFaceFeedbackBox(val && webgazer.params.showFaceFeedbackBox);
  return webgazer;
};

/**
 * hides a video element (videoElement or videoContainerElement)
 * uses display = 'none' for all browsers except Safari, which uses opacity = '1'
 * because Safari optimizes out video element if display = 'none'
 * @param {Object} element
 * @return {null}
 */
function hideVideoElement(val) {
  if (navigator.vendor && navigator.vendor.indexOf("Apple") > -1) {
    val.style.opacity = webgazer.params.showVideo ? "1" : "0";
    val.style.display = "block";
  } else {
    val.style.display = webgazer.params.showVideo ? "block" : "none";
  }
}

/**
 * Set whether the camera video preview is visible or not (default true).
 * @param {*} bool
 * @return {webgazer} this
 */
webgazer.showVideo = function (val, opacity = 0.8) {
  webgazer.params.showVideo = val;
  if (videoElement) {
    hideVideoElement(videoElement);
  }
  if (videoContainerElement) {
    hideVideoElement(videoContainerElement);
  }
  return webgazer;
};

/**
 * Set whether the face overlay is visible or not (default true).
 * @param {*} bool
 * @return {webgazer} this
 */
webgazer.showFaceOverlay = function (val) {
  webgazer.params.showFaceOverlay = val;
  if (faceOverlay) {
    faceOverlay.style.display = val ? "block" : "none";
  }
  return webgazer;
};

/**
 * Set whether the face feedback box is visible or not (default true).
 * @param {*} bool
 * @return {webgazer} this
 */
webgazer.showFaceFeedbackBox = function (val) {
  webgazer.params.showFaceFeedbackBox = val;
  if (faceFeedbackBox) {
    faceFeedbackBox.style.display = val ? "block" : "none";
  }
  return webgazer;
};

/**
 * Set whether the gaze prediction point(s) are visible or not.
 * Multiple because of a trail of past dots. Default true
 * @return {webgazer} this
 */
webgazer.showPredictionPoints = function (val) {
  webgazer.params.showGazeDot = val;
  if (gazeDot) {
    gazeDot.style.display = val ? "block" : "none";
  }
  return webgazer;
};

const gazeDotPopInterval = { current: undefined };

webgazer.popPredictionPoints = function () {
  if (gazeDotPopInterval.current) {
    clearInterval(gazeDotPopInterval.current);
    gazeDotPopInterval.current = undefined;
    gazeDotPopped = false;
  }

  if (gazeDot && webgazer.params.showGazeDot) {
    gazeDotPopped = true;

    // gazeDot.style.display = 'block'
    gazeDot.style.backgroundColor = "red";
    gazeDot.style.opacity = 1;
    gazeDotPopInterval.current = setInterval(() => {
      gazeDot.style.opacity -= 0.02;
      if (gazeDot.style.opacity <= 0.02) {
        clearInterval(gazeDotPopInterval.current);
        gazeDotPopInterval.current = undefined;
        gazeDotPopped = false;

        // gazeDot.style.display = 'none'
        gazeDot.style.backgroundColor = "";
        gazeDot.style.opacity = "";
      }
    }, 50); // 20 * 50 = 1 second
  }
  return webgazer;
};

/**
 * Set whether previous calibration data (from localforage) should be loaded.
 * Default true.
 *
 * NOTE: Should be called before webgazer.begin() -- see www/js/main.js for example
 *
 * @param val
 * @returns {webgazer} this
 */
webgazer.saveDataAcrossSessions = function (val) {
  webgazer.params.saveDataAcrossSessions = val;
  return webgazer;
};

/**
 * Set whether a Kalman filter will be applied to gaze predictions (default true);
 * @return {webgazer} this
 */
webgazer.applyKalmanFilter = function (val) {
  webgazer.params.applyKalmanFilter = val;
  return webgazer;
};

/**
 * Define constraints on the video camera that is used. Useful for non-standard setups.
 * This can be set before calling webgazer.begin(), but also mid stream.
 *
 * @param {Object} constraints Example constraints object:
 * { width: { min: 320, ideal: 1280, max: 1920 }, height: { min: 240, ideal: 720, max: 1080 }, facingMode: "user" };
 *
 * Follows definition here:
 * https://developer.mozilla.org/en-US/docs/Web/API/Media_Streams_API/Constraints
 *
 * Note: The constraints set here are applied to the video track only. They also _replace_ any constraints, so be sure to set everything you need.
 * Warning: Setting a large video resolution will decrease performance, and may require
 */
// webgazer.setCameraConstraints = async function (constraints) {
//   // var videoTrack, videoSettings;
  
//   // Enforce landscape orientation in constraints
//   const landscapeConstraints = {
//     ...constraints,
//     video: {
//       ...constraints.video,
//       // Ensure landscape aspect ratio
//       aspectRatio: { min: 1.33, ideal: 1.78, max: 2.33 },
//       // Set minimum width to be greater than minimum height
//       width: { min: 640, ideal: 1920, max: 7680 },
//       height: { min: 360, ideal: 1080, max: 4320 }
//     }
//   };
  
//   webgazer.params.camConstraints = landscapeConstraints;

//   // If the camera stream is already up...
//   if (videoStream) {
//     webgazer.pause();
//     // videoTrack = videoStream.getVideoTracks()[0];
//     try {
//       // await videoTrack.applyConstraints( webgazer.params.camConstraints );
//       videoStream.getVideoTracks().forEach((track) => {
//         track.stop();
//       });
//       const stream = await navigator.mediaDevices.getUserMedia(
//         webgazer.params.camConstraints
//       );


//       const hasLiveVideo = async (stream) => {
//         if (!stream) return false;
//         const track = stream.getVideoTracks()[0];
//         return !!track && track.readyState === "live" && !track.muted;
//       }

//       if (hasLiveVideo(stream)) {
//         console.log("Live video feed is active");
//       } else {
//         console.warn("Video track is missing or inactive");
//       }

//       setTimeout(() => {
//         const videoTrack = stream.getVideoTracks()[0];
//         const videoSettings = videoTrack.getSettings();
        
//         // Enforce landscape orientation if video is in portrait
//         if (videoSettings.width < videoSettings.height) {
//           try {
//             videoTrack.applyConstraints({
//               width: { min: 640, ideal: 1920, max: 7680 },
//               height: { min: 360, ideal: 1080, max: 4320 },
//               aspectRatio: { min: 1.33, ideal: 1.78, max: 2.33 }
//             });
//           } catch (constraintError) {
//             console.warn('Could not enforce landscape orientation:', constraintError);
//           }
//         }
        
//         videoStream = stream;
//         videoElement.srcObject = stream;
//         setInternalVideoBufferSizes(videoSettings.width, videoSettings.height);
//         webgazer.videoParamsToReport = { height: videoSettings.height, width: videoSettings.width };
//       }, 1500);
//     } catch (err) {
//       console.log(err);
//       return;
//     }
//     // Reset and recompute sizes of the video viewer.
//     // This is only to adjust the feedback box, say, if the aspect ratio of the video has changed.
//     // webgazer.setVideoViewerSize( webgazer.params.videoViewerWidth, webgazer.params.videoViewerHeight )
//     // webgazer.getTracker().reset();
//     await webgazer.resume();
//   }
// };

webgazer.getCameraResolutionXY = function () {
  try {
    const videoTrack = videoStream.getVideoTracks()[0];
    const videoSettings = videoTrack.getSettings();
    return {
      width: videoSettings.width,
      height: videoSettings.height,
    }
  } catch (error) {
    console.error("Error getting camera resolution:", error);
    return { width: 0, height: 0 };
  }
}
webgazer.setCameraConstraints = async function (constraints, knownResolution = null) {
  const deviceId = constraints.video?.deviceId;
  
  if (videoStream) {
    _isSwappingCamera = true;
    if (liveMonitor) liveMonitor.pause();
    console.log('[CameraReconnect] setCameraConstraints START — _isSwappingCamera = true, monitor paused');
    webgazer.pause();
    const previousStream = videoStream;
    try {
      // The old stream is released only once the new one is live (see the
      // end of this block). Stopping first leaves the participant with a
      // dead <video> if acquiring the replacement fails, and on macOS it
      // can also make the immediate reopen fail while the OS is still
      // tearing the capture device down.

      let stream;
      const desiredRes = webgazer.params.desiredCameraResolution;
      const desiredHz = webgazer.params.desiredCameraHz;

      if (desiredRes && desiredHz) {
        // Pick the closest available mode for this device (cached per device,
        // so switching back to a camera seen earlier costs nothing).
        const rawDeviceId = deviceId?.exact || deviceId;
        console.log(`setCameraConstraints: using findBestCameraMode for ${desiredRes[0]}x${desiredRes[1]} @ ${desiredHz}Hz`);
        const result = await findBestCameraMode(rawDeviceId, desiredRes[0], desiredRes[1], desiredHz);
        stream = result.stream;

        const w = result.width;
        const h = result.height;
        console.log(`setCameraConstraints (probed): ${w}x${h} @ ${result.frameRate}Hz, capability max: ${result.capMaxWidth}x${result.capMaxHeight}`);

        videoStream = stream;
        videoElement.srcObject = stream;
        setInternalVideoBufferSizes(w, h);
        startOrUpdateLiveMonitor(stream);

        webgazer.videoParamsToReport = {
          height: h, width: w,
          maxHeight: result.capMaxHeight, maxWidth: result.capMaxWidth,
          frameRate: result.frameRate,
          maxFrameRate: result.capMaxFrameRate
        };
      } else {
        // No desired mode configured: request the known resolution if we
        // have one, else the highest the camera offers. Expressed as `ideal`
        // so an unsatisfiable value degrades gracefully instead of throwing
        // and forcing a retry ladder.
        const idealWidth = knownResolution?.width || 1920;
        const idealHeight = knownResolution?.height || 1080;
        console.log(`setCameraConstraints: requesting ideal ${idealWidth}x${idealHeight}`);
        stream = await getUserMediaResilient({
          video: {
            deviceId: deviceId,
            width: { ideal: idealWidth },
            height: { ideal: idealHeight },
            aspectRatio: { min: 1.33, ideal: 1.78, max: 2.33 },
            facingMode: "user"
          }
        });

        const videoTrack = stream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        const w = settings.width || 640;
        const h = settings.height || 480;

        const cap = typeof videoTrack.getCapabilities === 'function' ? videoTrack.getCapabilities() : null;
        const capMaxW = cap?.width?.max || w;
        const capMaxH = cap?.height?.max || h;
        const actualFR = settings.frameRate || 0;
        const capMaxFR = cap?.frameRate?.max || actualFR;
        console.log(`setCameraConstraints: ${w}x${h} @ ${actualFR}Hz, capability max: ${capMaxW}x${capMaxH} @ ${capMaxFR}Hz`);

        videoStream = stream;
        videoElement.srcObject = stream;
        setInternalVideoBufferSizes(w, h);
        startOrUpdateLiveMonitor(stream);

        webgazer.videoParamsToReport = { 
          height: h, width: w,
          maxHeight: capMaxH, maxWidth: capMaxW,
          frameRate: actualFR,
          maxFrameRate: capMaxFR
        };
      }

      // New stream is attached and monitored, so the old one can go.
      if (previousStream && previousStream !== videoStream) {
        previousStream.getVideoTracks().forEach(t => t.stop());
      }
    } catch (err) {
      console.error('[CameraReconnect] setCameraConstraints ERROR:', err);
      // Leave the participant looking at the previous camera rather than a
      // dead <video> element.
      if (previousStream && previousStream.active) {
        videoStream = previousStream;
        if (videoElement) videoElement.srcObject = previousStream;
        startOrUpdateLiveMonitor(previousStream);
      }
      _isSwappingCamera = false;
      await webgazer.resume();
      return;
    }
    console.log('[CameraReconnect] setCameraConstraints END — _isSwappingCamera = false');
    _isSwappingCamera = false;
    await webgazer.resume();
  }
};

/**
 * Fast-path camera reconnect: open a stream and apply exact constraints for a
 * previously-known resolution and frame rate, skipping the full probing sweep.
 * Throws if the exact mode cannot be applied so the caller can fall back.
 */
webgazer.setCameraConstraintsDirect = async function (constraints, width, height, frameRate) {
  const deviceId = constraints.video?.deviceId;

  if (videoStream) {
    _isSwappingCamera = true;
    if (liveMonitor) liveMonitor.pause();
    console.log('[CameraReconnect] setCameraConstraintsDirect START — _isSwappingCamera = true, monitor paused');
    webgazer.pause();
    const previousStream = videoStream;
    try {
      const rawDeviceId = deviceId?.exact || deviceId;
      const videoConstraints = rawDeviceId
        ? { deviceId: { exact: rawDeviceId } }
        : { facingMode: 'user' };
      const stream = await getUserMediaResilient({
        video: {
          ...videoConstraints,
          width: { ideal: width },
          height: { ideal: height },
          frameRate: { ideal: frameRate },
        },
      });
      const track = stream.getVideoTracks()[0];

      await track.applyConstraints({
        width: { exact: width },
        height: { exact: height },
        frameRate: { ideal: frameRate }
      });

      const settings = track.getSettings();
      const w = settings.width;
      const h = settings.height;
      const actualFR = settings.frameRate || 0;
      console.log(`setCameraConstraintsDirect: ${w}x${h} @ ${actualFR}Hz`);

      const cap = typeof track.getCapabilities === 'function' ? track.getCapabilities() : null;

      videoStream = stream;
      videoElement.srcObject = stream;
      setInternalVideoBufferSizes(w, h);
      startOrUpdateLiveMonitor(stream);

      webgazer.videoParamsToReport = {
        height: h, width: w,
        maxHeight: cap?.height?.max || h,
        maxWidth: cap?.width?.max || w,
        frameRate: actualFR,
        maxFrameRate: cap?.frameRate?.max || actualFR
      };

      if (previousStream && previousStream !== stream) {
        previousStream.getVideoTracks().forEach(t => t.stop());
      }
    } catch (err) {
      console.error('[CameraReconnect] setCameraConstraintsDirect ERROR:', err);
      // The caller falls back to setCameraConstraints, so keep the previous
      // stream alive and attached until that succeeds.
      if (previousStream && previousStream.active) {
        videoStream = previousStream;
        if (videoElement) videoElement.srcObject = previousStream;
      }
      _isSwappingCamera = false;
      throw err;
    }
    console.log('[CameraReconnect] setCameraConstraintsDirect END — _isSwappingCamera = false');
    _isSwappingCamera = false;
    await webgazer.resume();
  }
};

/**
 * Does what it says on the tin.
 * @param {*} width
 * @param {*} height
 */
function setInternalVideoBufferSizes(width, height) {
  // Re-set the canvas size used by the internal processes
  if (videoElementCanvas) {
    videoElementCanvas.width = width;
    videoElementCanvas.height = height;
  }

  // Re-set the face overlay canvas size
  if (faceOverlay) {
    faceOverlay.width = width;
    faceOverlay.height = height;
  }
}

/**
 * Callback function for camera reconnection events.
 * Can be set by parent application to handle camera disconnection.
 * @type {Function|null}
 */
webgazer.onCameraDisconnected = null;

/**
 * Callback for when the participant clicks Quit on the camera reconnect popup.
 * @type {Function|null}
 */
webgazer.onQuit = null;

/**
 * Shows a camera reconnection popup or notification when camera is disconnected.
 * This function can be overridden by the parent application to provide custom UI.
 * @param {string} message - The message to display
 */
async function _enumerateCameras() {
  try {
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter(d => d.kind === 'videoinput');
    }
  } catch (e) {
    console.warn('Could not enumerate cameras:', e);
  }
  return [];
}

function _getPhrase(key) {
  const p = webgazer.params.phrases;
  const lang = webgazer.params.language || 'en-US';
  const value = (p && p[key] && p[key][lang]) || (p && p[key] && p[key]['en-US']) || '';
  console.log(`[CameraReconnect] phrase ${key}[${lang}] = "${value}"`);
  return value;
}

// True when the participant's currently-selected language reads
// right-to-left (e.g. Arabic, Hebrew). Driven by RC, which writes
// `languageDirection` into webgazer.params on init and on every
// language change (see GazeTracker.setupCameraMonitoring).
function _isRTL() {
  return (webgazer.params.languageDirection || 'LTR').toUpperCase() === 'RTL';
}

function _dimPageContent() {
  for (const child of document.body.children) {
    if (child.classList && child.classList.contains('swal2-container')) continue;
    child.dataset.rcOriginalFilter = child.style.filter || '';
    child.style.filter = 'contrast(0.5)';
  }
}

function _makeSwalBackdropTransparent() {
  const swalContainer = document.querySelector('.swal2-container');
  if (swalContainer) {
    swalContainer.style.setProperty('background', 'transparent', 'important');
    swalContainer.style.setProperty('background-color', 'transparent', 'important');
  }
}

function _restorePageContrast() {
  _cleanupReconnectOverlay();
  for (const child of document.body.children) {
    if (child.classList && child.classList.contains('swal2-container')) continue;
    if ('rcOriginalFilter' in (child.dataset || {})) {
      child.style.filter = child.dataset.rcOriginalFilter;
      delete child.dataset.rcOriginalFilter;
    } else {
      child.style.filter = '';
    }
  }
}

const RC_SNAPSHOT_ID = 'rc-reconnect-page-snapshot';

/**
 * Prepare the page for the reconnection popup:
 *  1. Hide the EasyEyes calibration panel (`rc-panel-holder`) so it
 *     doesn't show through when the current Swal is replaced.
 *  2. Clone ONLY the Swal popup (which contains camera previews,
 *     arrow, privacy text, etc.) and body-level fixed elements
 *     (`rc-camera-title-top-right`, `rc-resolution-video-wrapper`,
 *     `rc-resolution-setting-message`) that `willClose` will remove.
 *  3. Place the clones in a full-viewport white overlay so the
 *     participant sees the correct page dimmed behind the popup.
 */
function _prepareReconnectOverlay() {
  _cleanupReconnectOverlay();

  // Hide the EasyEyes calibration panel — save its display so we
  // can restore it later.
  const panel = document.getElementById('rc-panel-holder');
  if (panel) {
    panel.dataset.rcSavedDisplay = panel.style.display || '';
    panel.style.display = 'none';
  }

  // Remove the Choose Camera bottom-row preview wrapper (which was
  // previously promoted to <body> by _promoteCameraPreviewsBottomToBody)
  // BEFORE we capture the snapshot and dim the page. The owning Swal's
  // willClose handler is supposed to clean this up, but its timing is
  // not synchronously guaranteed across all SweetAlert versions and
  // browsers. Clearing it here ensures the participant never sees a
  // stale bottom-row tile fixed to the viewport behind the
  // reconnection popup. The element is re-created from scratch when
  // the Choose Camera page is rebuilt after reconnect.
  let staleBottomOuter = document.getElementById('rc-camera-previews-bottom-outer');
  while (staleBottomOuter) {
    console.log('[CameraReconnect] _prepareReconnectOverlay: removing stale rc-camera-previews-bottom-outer');
    // Stop any video tracks still attached to bottom-row previews so
    // they don't keep the camera busy after the participant pulls it.
    staleBottomOuter.querySelectorAll('video').forEach(v => {
      try {
        if (v.srcObject) {
          v.srcObject.getTracks().forEach(t => t.stop());
        }
        v.srcObject = null;
      } catch (_) { /* noop */ }
    });
    staleBottomOuter.remove();
    staleBottomOuter = document.getElementById('rc-camera-previews-bottom-outer');
  }

  // Full-viewport opaque wrapper that sits above everything except
  // the reconnection Swal (SweetAlert2 uses z-index ~1060).
  const wrapper = document.createElement('div');
  wrapper.id = RC_SNAPSHOT_ID;
  wrapper.style.cssText =
    'position:fixed;top:0;left:0;width:100vw;height:100vh;' +
    'z-index:999;pointer-events:none;background:#fff;overflow:hidden;';

  let hasContent = false;

  // Clone the active Swal popup (camera selection, resolution, etc.).
  // Elements like camera previews, arrow, and privacy text live INSIDE
  // the popup — cloning the popup captures them all in one go.
  const swalPopup = document.querySelector('.swal2-popup');
  if (swalPopup) {
    const clone = swalPopup.cloneNode(true);
    clone.removeAttribute('id');
    clone.style.position = 'fixed';
    clone.style.top = '50%';
    clone.style.left = '50%';
    clone.style.transform = 'translate(-50%, -50%)';
    clone.style.margin = '0';
    clone.style.pointerEvents = 'none';
    clone.querySelectorAll('video').forEach(v => { try { v.pause(); v.srcObject = null; } catch(_){} });
    wrapper.appendChild(clone);
    hasContent = true;
  }

  // Clone body-level fixed elements that sit OUTSIDE the Swal and
  // would be removed by the Swal's willClose handler.
  const outsideSwalIds = [
    'rc-camera-title-top-right',
    'rc-resolution-video-wrapper',
    'rc-resolution-setting-message',
  ];
  for (const id of outsideSwalIds) {
    const el = document.getElementById(id);
    if (!el) continue;
    const clone = el.cloneNode(true);
    clone.removeAttribute('id');
    clone.style.pointerEvents = 'none';
    wrapper.appendChild(clone);
    hasContent = true;
  }

  if (!hasContent) { wrapper.remove(); return; }
  document.body.appendChild(wrapper);
}

function _cleanupReconnectOverlay() {
  // Remove the snapshot overlay
  const snapshot = document.getElementById(RC_SNAPSHOT_ID);
  if (snapshot) snapshot.remove();

  // Restore the EasyEyes calibration panel
  const panel = document.getElementById('rc-panel-holder');
  if (panel && 'rcSavedDisplay' in (panel.dataset || {})) {
    panel.style.display = panel.dataset.rcSavedDisplay;
    delete panel.dataset.rcSavedDisplay;
  }
}

function _styleReconnectSwal() {
  _makeSwalBackdropTransparent();

  const isRTL = _isRTL();

  // Apply RTL/LTR direction at the popup level so the title, body
  // text, and any inline content inherit the correct reading
  // direction. Buttons are reordered separately via Swal's
  // `reverseButtons` option (see showCameraReconnectionPopup).
  const popup = Swal.getPopup();
  if (popup) {
    popup.dir = isRTL ? 'rtl' : 'ltr';
  }
  const title = Swal.getTitle();
  if (title) {
    title.dir = isRTL ? 'rtl' : 'ltr';
    title.style.textAlign = 'center';
  }
  const htmlContainer = Swal.getHtmlContainer();
  if (htmlContainer) {
    htmlContainer.dir = isRTL ? 'rtl' : 'ltr';
    htmlContainer.style.textAlign = 'center';
  }

  const confirmBtn = Swal.getConfirmButton();
  if (confirmBtn) {
    confirmBtn.style.backgroundColor = '#28a745';
    confirmBtn.style.borderColor = '#28a745';
    confirmBtn.style.color = '#fff';
    confirmBtn.style.fontSize = '1.2rem';
    confirmBtn.style.padding = '0.6rem 2.5rem';
  }
  const cancelBtn = Swal.getCancelButton();
  if (cancelBtn) {
    cancelBtn.style.backgroundColor = '#dc3545';
    cancelBtn.style.borderColor = '#dc3545';
    cancelBtn.style.color = '#fff';
    cancelBtn.style.fontSize = '0.85rem';
    cancelBtn.style.padding = '0.4rem 1.2rem';
  }
  const actions = Swal.getActions();
  if (actions) {
    actions.style.justifyContent = 'space-between';
    actions.style.width = '100%';
    actions.style.padding = '0 1rem';
    // Mirror button order with the page direction:
    //   LTR: [ Proceed ........ Quit ]
    //   RTL: [ Quit ........ Proceed ]
    // Setting `dir` on the flex container is enough — flexbox
    // auto-mirrors children when the writing direction is rtl, and
    // `justify-content: space-between` keeps them pinned to the
    // outer edges in both cases.
    actions.dir = isRTL ? 'rtl' : 'ltr';
  }
}

async function _tryReconnectOriginalCamera() {
  const currentId = webgazer.params.activeCamera?.id || '';
  const currentLabel = webgazer.params.activeCamera?.label || '';
  console.log('[CameraReconnect] Attempting to reconnect camera:', JSON.stringify({ id: currentId, label: currentLabel }));
  if (!currentId) return false;

  const cameras = await _enumerateCameras();
  console.log('[CameraReconnect] Scan found', cameras.length, 'camera(s):', cameras.map(c => c.label || c.deviceId));
  const found = cameras.find(c => c.deviceId === currentId);
  if (!found) {
    console.log('[CameraReconnect] Original camera NOT found in scan:', JSON.stringify({ id: currentId, label: currentLabel }));
    return false;
  }
  console.log('[CameraReconnect] Original camera found:', JSON.stringify({ id: found.deviceId, label: found.label }));

  if (!webgazer.params.camConstraints) return false;

  const constraints = {
    video: {
      deviceId: { exact: currentId },
      facingMode: 'user',
    },
  };

  const prevReport = webgazer.videoParamsToReport || {};
  const prevWidth = prevReport.width;
  const prevHeight = prevReport.height;
  const prevFrameRate = prevReport.frameRate;

  if (prevWidth && prevHeight && prevFrameRate) {
    console.log(`[CameraReconnect] Trying fast reconnect: ${prevWidth}x${prevHeight} @ ${prevFrameRate}Hz`);
    try {
      await webgazer.setCameraConstraintsDirect(constraints, prevWidth, prevHeight, prevFrameRate);
      console.log('[CameraReconnect] Fast reconnect succeeded. _isReconnecting:', _isReconnecting, '_isSwappingCamera:', _isSwappingCamera);
    } catch (fastErr) {
      console.warn('[CameraReconnect] Fast reconnect failed, falling back to full probing:', fastErr);
      await webgazer.setCameraConstraints(constraints, null);
      console.log('[CameraReconnect] Full-probing fallback completed. _isReconnecting:', _isReconnecting, '_isSwappingCamera:', _isSwappingCamera);
    }
  } else {
    console.log('[CameraReconnect] No previous resolution/Hz known, using full probing.');
    await webgazer.setCameraConstraints(constraints, null);
    console.log('[CameraReconnect] setCameraConstraints completed. _isReconnecting:', _isReconnecting, '_isSwappingCamera:', _isSwappingCamera);
  }

  if (typeof webgazer.onCameraReconnected === 'function') {
    console.log('[CameraReconnect] Firing onCameraReconnected callback');
    webgazer.onCameraReconnected();
  }

  return true;
}

async function showCameraReconnectionPopup(message) {
  console.log('[CameraReconnect] showCameraReconnectionPopup called, _isReconnecting:', _isReconnecting);
  if (_isReconnecting) {
    console.log('[CameraReconnect] showCameraReconnectionPopup BLOCKED — already reconnecting');
    return;
  }
  _isReconnecting = true;

  if (liveMonitor) liveMonitor.pause();

  const cameraToReconnect = webgazer.params.activeCamera?.label || webgazer.params.activeCamera?.id || 'unknown';
  console.warn("Camera paused:", message);
  console.log('[CameraReconnect] Camera to reconnect:', cameraToReconnect);

  if (typeof webgazer.onCameraDisconnected === 'function') {
    webgazer.onCameraDisconnected(message);
  }

  // Hide the EasyEyes panel and snapshot the current page so the
  // participant sees the correct background (dimmed) behind the
  // reconnection popup instead of the calibration panel.
  _prepareReconnectOverlay();

  // Close any existing Swal so its willClose handler runs and cleans up.
  // The overlay already captured the visual state, so losing the Swal is fine.
  Swal.close();

  _dimPageContent();

  const titleText = _getPhrase('RC_CameraReconnectTitle');
  const resumeText = _getPhrase('RC_Proceed');
  const quitText = _getPhrase('RC_Quit');

  const result = await Swal.fire({
    icon: undefined,
    title: titleText,
    showConfirmButton: true,
    showCancelButton: true,
    confirmButtonText: resumeText,
    cancelButtonText: quitText,
    allowEscapeKey: false,
    allowOutsideClick: false,
    backdrop: 'rgba(0,0,0,0)',
    reverseButtons: false,
    customClass: {
      container: 'camera-reconnect-container',
      popup: 'camera-reconnection-popup',
      confirmButton: 'swal2-confirm-resume',
      cancelButton: 'swal2-cancel-quit',
    },
    didOpen: _styleReconnectSwal,
  });

  if (!result.isConfirmed) {
    console.log('[CameraReconnect] Quit button pressed');
    _restorePageContrast();
    if (typeof webgazer.onQuit === 'function') webgazer.onQuit();
    _isReconnecting = false;
    return;
  }

  // Retry loop: try to reconnect, and if the camera is still missing,
  // always show the "Sorry. Can't find ..." page (never go back to
  // the initial "To save power ..." page).
  //
  // Build the spinner text from the i18n phrase
  // RC_CameraReconnecting = "Reconnecting camera at [[RRR]] ..."
  // where [[RRR]] is replaced with the previous camera mode in the form
  // "WIDTH × HEIGHT, FRAMERATE Hz" (integers, e.g. "640 × 480, 15 Hz").
  // If we have no previous resolution/Hz info we drop the " at [[RRR]]"
  // segment entirely.
  const prevReport = webgazer.videoParamsToReport || {};
  const haveRes = prevReport.width && prevReport.height;
  const haveHz = !!prevReport.frameRate;
  const rrrText = (haveRes && haveHz)
    ? `${Math.round(prevReport.width)} \u00d7 ${Math.round(prevReport.height)}, ${Math.round(prevReport.frameRate)} Hz`
    : '';

  const reconnectingTemplate = _getPhrase('RC_CameraReconnecting')
    || 'Reconnecting camera at [[RRR]] ...';

  let spinnerDetail;
  if (rrrText) {
    spinnerDetail = reconnectingTemplate.replace(/\[\[RRR\]\]/gi, rrrText);
  } else {
    // Drop " at [[RRR]]" (with the leading space and any surrounding
    // whitespace) when we have no resolution info, so the message reads
    // naturally as "Reconnecting camera ...".
    spinnerDetail = reconnectingTemplate
      .replace(/\s*at\s*\[\[RRR\]\]/gi, '')
      .replace(/\[\[RRR\]\]/gi, '');
  }

  while (true) {
    const isRTLNow = _isRTL();
    const spinnerDir = isRTLNow ? 'rtl' : 'ltr';
    Swal.fire({
      title: undefined,
      html: `<p dir="${spinnerDir}" style="margin: 0.5rem 0; line-height: 1.6; font-size: 0.95rem; color: #555; text-align: center; direction: ${spinnerDir};">${spinnerDetail}</p>`,
      allowOutsideClick: false,
      allowEscapeKey: false,
      showConfirmButton: false,
      backdrop: 'rgba(0,0,0,0)',
      customClass: {
        container: 'camera-reconnect-container',
      },
      didOpen: () => {
        _makeSwalBackdropTransparent();
        const popup = Swal.getPopup();
        if (popup) popup.dir = spinnerDir;
        Swal.showLoading();
      },
    });

    const spinnerStart = performance.now();
    let reconnected = false;
    try {
      reconnected = await _tryReconnectOriginalCamera();
    } catch (error) {
      console.error('[CameraReconnect] Failed to reconnect camera:', error);
    }

    if (reconnected) {
      const elapsed = performance.now() - spinnerStart;
      const MIN_SPINNER_MS = 2000;
      if (elapsed < MIN_SPINNER_MS) {
        await new Promise(r => setTimeout(r, MIN_SPINNER_MS - elapsed));
      }
      console.log('[CameraReconnect] Successfully reconnected original camera');
      _restorePageContrast();
      Swal.close();
      break;
    }

    const cameraLabel = webgazer.params.activeCamera?.label || '';
    const cantFindTemplate = _getPhrase('RC_CameraReconnectCantFindIt');
    const cantFindText = cantFindTemplate.replace(/\[\[xxx\]\]/gi, `"${cameraLabel}"`);
    const isRTLRetry = _isRTL();
    const retryDir = isRTLRetry ? 'rtl' : 'ltr';

    const retryResult = await Swal.fire({
      icon: undefined,
      title: titleText,
      html: `<p dir="${retryDir}" style="margin: 0.5rem 0; line-height: 1.6; text-align: center; direction: ${retryDir};">${cantFindText}</p>`,
      showConfirmButton: true,
      showCancelButton: true,
      confirmButtonText: resumeText,
      cancelButtonText: quitText,
      allowEscapeKey: false,
      allowOutsideClick: false,
      backdrop: 'rgba(0,0,0,0)',
      reverseButtons: false,
      customClass: {
        container: 'camera-reconnect-container',
        popup: 'camera-reconnection-popup',
        confirmButton: 'swal2-confirm-resume',
        cancelButton: 'swal2-cancel-quit',
      },
      didOpen: _styleReconnectSwal,
    });

    if (!retryResult.isConfirmed) {
      console.log('[CameraReconnect] Quit button pressed');
      _restorePageContrast();
      if (typeof webgazer.onQuit === 'function') webgazer.onQuit();
      break;
    }
    console.log('[CameraReconnect] Retrying camera reconnection');
  }

  console.log('[CameraReconnect] showCameraReconnectionPopup DONE — setting _isReconnecting = false');
  _isReconnecting = false;
}

/**
 * Set a custom callback for camera disconnection events.
 * This allows parent applications to show custom UI instead of the default alert.
 * @param {Function} callback - Function to call when camera is disconnected
 * @return {webgazer} this
 */
webgazer.setOnCameraDisconnected = function(callback) {
  webgazer.onCameraDisconnected = callback;
  return webgazer;
};

/**
 * Callback function for camera reconnection events.
 * @type {Function|null}
 */
webgazer.onCameraReconnected = null;

/**
 * Set a custom callback for successful camera reconnection events.
 * @param {Function} callback - Function to call when camera is reconnected
 * @return {webgazer} this
 */
webgazer.setOnCameraReconnected = function(callback) {
  webgazer.onCameraReconnected = callback;
  return webgazer;
};

/**
 * Set a callback for when the participant clicks Quit on the camera reconnect popup.
 * @param {Function} callback - Function to call when quit is requested
 * @return {webgazer} this
 */
webgazer.setOnQuit = function(callback) {
  webgazer.onQuit = callback;
  return webgazer;
};

/**
 *  Set a static video file to be used instead of webcam video
 *  @param {String} videoLoc - video file location
 *  @return {webgazer} this
 */
webgazer.setStaticVideo = function (videoLoc) {
  debugVideoLoc = videoLoc;
  return webgazer;
};

/**
 * Set the size of the video viewer
 */
webgazer.setVideoViewerSize = function (w, h) {
  webgazer.params.videoViewerWidth = w;
  webgazer.params.videoViewerHeight = h;

  // Change the video viewer
  videoElement.style.width = "100%" //w + "px";
  videoElement.style.height = "100%"; //h + "px";
  videoElement.style.objectFit = "cover";

  // Change video container
  videoContainerElement.style.width = w + "px";
  videoContainerElement.style.height = h + "px";

  // Change the face overlay
  faceOverlay.style.width = w + "px";
  faceOverlay.style.height = h + "px";

  // Change the feedback box size
  // Compute the boundaries of the face overlay validation box based on the video size
  var tlwh = webgazer.computeValidationBoxSize();
  // Assign them to the object
  faceFeedbackBox.style.top = tlwh[0] + "px";
  faceFeedbackBox.style.left = tlwh[1] + "px";
  faceFeedbackBox.style.width = tlwh[2] + "px";
  faceFeedbackBox.style.height = tlwh[3] + "px";
};

/**
 *  Add the mouse click and move listeners that add training data.
 *  @return {webgazer} this
 */
webgazer.addMouseEventListeners = function () {
  addMouseEventListeners();
  return webgazer;
};

/**
 *  Remove the mouse click and move listeners that add training data.
 *  @return {webgazer} this
 */
webgazer.removeMouseEventListeners = function () {
  removeMouseEventListeners();
  return webgazer;
};

/**
 *  Records current screen position for current pupil features.
 *  @param {String} x - position on screen in the x axis
 *  @param {String} y - position on screen in the y axis
 *  @param {String} eventType - "click" or "move", as per eventTypes
 *  @return {webgazer} this
 */
webgazer.recordScreenPosition = function (x, y, eventType) {
  // give this the same weight that a click gets.
  recordScreenPosition(x, y, eventType || eventTypes[0]);
  return webgazer;
};

/*
 * Stores the position of the fifty most recent tracker preditions
 */
webgazer.storePoints = function (x, y, k) {
  xPast50[k] = x;
  yPast50[k] = y;
};

//SETTERS
/**
 * Sets the tracking module
 * @param {String} name - The name of the tracking module to use
 * @return {webgazer} this
 */
webgazer.setTracker = async function (name) {
  if (curTrackerMap[name] === undefined) {
    console.log("Invalid tracker selection");
    console.log("Options are: ");
    for (var t in curTrackerMap) {
      console.log(t);
    }
    return webgazer;
  }
  curTracker = curTrackerMap[name]();
  await curTracker.loadModel();
  return webgazer;
};

/**
 * Sets the regression module and clears any other regression modules
 * @param {String} name - The name of the regression module to use
 * @return {webgazer} this
 */
webgazer.setRegression = function (name) {
  if (regressionMap[name] === undefined) {
    console.log("Invalid regression selection");
    console.log("Options are: ");
    for (var reg in regressionMap) {
      console.log(reg);
    }
    return webgazer;
  }
  data = regs[0].getData();
  regs = [regressionMap[name]()];
  regs[0].setData(data);
  return webgazer;
};

/**
 * Adds a new tracker module so that it can be used by setTracker()
 * @param {String} name - the new name of the tracker
 * @param {Function} constructor - the constructor of the curTracker object
 * @return {webgazer} this
 */
webgazer.addTrackerModule = function (name, constructor) {
  curTrackerMap[name] = function () {
    return new constructor();
  };
};

/**
 * Adds a new regression module so that it can be used by setRegression() and addRegression()
 * @param {String} name - the new name of the regression
 * @param {Function} constructor - the constructor of the regression object
 */
webgazer.addRegressionModule = function (name, constructor) {
  regressionMap[name] = function () {
    return new constructor();
  };
};

/**
 * Adds a new regression module to the list of regression modules, seeding its data from the first regression module
 * @param {String} name - the string name of the regression module to add
 * @return {webgazer} this
 */
webgazer.addRegression = function (name) {
  var newReg = regressionMap[name]();
  data = regs[0].getData();
  newReg.setData(data);
  regs.push(newReg);
  return webgazer;
};

/**
 * Sets a callback to be executed on every gaze event (currently all time steps)
 * @param {function} listener - The callback function to call (it must be like function(data, elapsedTime))
 * - No elapsedTime needed for Toolbox
 * @return {webgazer} this
 */
webgazer.setGazeListener = function (listener) {
  callback = listener;
  return webgazer;
};

/**
 * Removes the callback set by setGazeListener
 * @return {webgazer} this
 */
webgazer.clearGazeListener = function () {
  callback = nopCallback;
  return webgazer;
};

/**
 * Set the video element canvas; useful if you want to run WebGazer on your own canvas (e.g., on any random image).
 * @return The current video element canvas
 */
webgazer.setVideoElementCanvas = function (canvas) {
  videoElementCanvas = canvas;
  return videoElementCanvas;
};

/**
 * Clear data from localforage and from regs
 */
webgazer.clearData = async function () {
  clearData();
};

//GETTERS
/**
 * Returns the tracker currently in use
 * @return {tracker} an object following the tracker interface
 */
webgazer.getTracker = function () {
  return curTracker;
};

/**
 * Returns the regression currently in use
 * @return {Array.<Object>} an array of regression objects following the regression interface
 */
webgazer.getRegression = function () {
  return regs;
};

/**
 * getGazeNow
 * Requests an immediate prediction
 * @return {object} prediction data object
 */
webgazer.getCurrentPrediction = async function (
  regIndex = 0,
  wait = 150,
  frames = 5
) {
  let totalTimeStamps = 0;
  const eyeFeatures = [];
  const predictions = [];

  for (let frame = 0; frame < frames; frame++) {
    totalTimeStamps += performance.now();
    gazePrepForGetGazeNow1();
    eyeFeatures.push(await gazePrepForGetGazeNow2());

    await sleep(1000 / webgazer.params.framerate);
  }
  webgazer.params.getLatestVideoFrameTimestamp(
    Math.round(totalTimeStamps / frames)
  );

  await sleep(wait);

  let prediction;
  for (const eyeFeature of eyeFeatures) {
    for (let i = 0; i < 2; i++)
      prediction = await getPrediction(undefined, eyeFeature);

    predictions.push(prediction);
  }

  const finalPredictionX = Math.round(
    predictions.reduce((a, b) => {
      if (b.x >= 0 && b.x <= window.innerWidth) return a + b.x;
      else return a;
    }, 0) / predictions.length
  );
  const finalPredictionY = Math.round(
    predictions.reduce((a, b) => {
      if (b.y >= 0 && b.y <= window.innerHeight) return a + b.y;
      else return a;
    }, 0) / predictions.length
  );

  if (gazeDot) {
    // const boundedPrediction = util.bound({
    //   x: finalPredictionX,
    //   y: finalPredictionY,
    // })
    gazeDot.style.left = `${finalPredictionX}px`;
    gazeDot.style.top = `${finalPredictionY}px`;
  }

  return {
    x: finalPredictionX,
    y: finalPredictionY,
    raw: predictions.map((prediction) => ({
      x: toFixedNumber(prediction.x, 0),
      y: toFixedNumber(prediction.y, 0),
    })),
  };
};

/**
 * returns the different event types that may be passed to regressions when calling regression.addData()
 * @return {Array} array of strings where each string is an event type
 */
webgazer.params.getEventTypes = function () {
  return eventTypes.slice();
};

/**
 * Get the video element canvas that WebGazer uses internally on which to run its face tracker.
 * @return The current video element canvas
 */
webgazer.getVideoElementCanvas = function () {
  return videoElementCanvas;
};

/**
 * @return array [a,b] where a is width ratio and b is height ratio
 */
webgazer.getVideoPreviewToCameraResolutionRatio = function () {
  return [
    webgazer.params.videoViewerWidth / videoElement.videoWidth,
    webgazer.params.videoViewerHeight / videoElement.videoHeight,
  ];
};

/*
 * Gets the fifty most recent tracker predictions
 */
webgazer.getStoredPoints = function () {
  return [xPast50, yPast50];
};

export default webgazer;
