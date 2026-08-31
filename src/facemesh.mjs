import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { useFullRangeModel } from './useFullRangeModel.mjs';

const MODEL_LOAD_ATTEMPTS = 3;
const MODEL_LOAD_RETRY_DELAY_MS = 700;

const _detectorConfig = refineLandmarks => ({
  runtime: 'tfjs',
  detectorModelUrl: './models/detector/model.json',
  landmarkModelUrl: refineLandmarks ? './models/landmark_attention/model.json' : './models/landmark/model.json',
  refineLandmarks: refineLandmarks,
});

/**
 * Constructor of TFFaceMesh object
 * @constructor
 * */
const TFFaceMesh = function(refineLandmarks = true) {
  this.refineLandmarks = refineLandmarks;
  this.model = null;
  this.predictionReady = false;
  this.modelLoaded = false;
};

const _modelLoadingInProgress = { current: false, resolves: [], rejects: [] };

const _sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Download and initialize the face landmark model.
 *
 * The model weights are several megabytes fetched over the network, so this
 * can fail transiently -- and a rejection here used to be fatal: the
 * detector promise was created in the constructor and left unhandled, so a
 * failed download surfaced as an unhandled rejection. PsychoJS installs a
 * `window.onunhandledrejection` handler that aborts the whole experiment,
 * meaning one flaky fetch ended the participant's session.
 *
 * The download therefore starts here rather than in the constructor (so the
 * promise always has a handler attached), and is retried before giving up.
 */
TFFaceMesh.prototype.loadModel = async function() {
  if (this.modelLoaded) return;

  if (_modelLoadingInProgress.current) {
    // Join the in-flight load rather than starting a second download.
    return await new Promise((resolve, reject) => {
      _modelLoadingInProgress.resolves.push(resolve);
      _modelLoadingInProgress.rejects.push(reject);
    });
  }

  _modelLoadingInProgress.current = true;

  try {
    let lastError;
    for (let attempt = 1; attempt <= MODEL_LOAD_ATTEMPTS; attempt++) {
      try {
        this.model = await faceLandmarksDetection.createDetector(
          faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
          _detectorConfig(this.refineLandmarks),
        );
        useFullRangeModel(this.model);
        this.modelLoaded = true;
        lastError = null;
        break;
      } catch (err) {
        lastError = err;
        this.model = null;
        if (attempt < MODEL_LOAD_ATTEMPTS) {
          console.warn(
            `[TFFaceMesh] Model load attempt ${attempt}/${MODEL_LOAD_ATTEMPTS} failed, retrying in ${MODEL_LOAD_RETRY_DELAY_MS}ms:`,
            err?.message || err,
          );
          await _sleep(MODEL_LOAD_RETRY_DELAY_MS);
        }
      }
    }

    if (lastError) {
      console.error(
        `[TFFaceMesh] Model load failed after ${MODEL_LOAD_ATTEMPTS} attempts.`,
        'Check that the models/ folder is reachable from the page URL.',
        lastError,
      );
      _modelLoadingInProgress.rejects.forEach(reject => reject(lastError));
      throw lastError;
    }

    _modelLoadingInProgress.resolves.forEach(resolve => resolve());
  } finally {
    _modelLoadingInProgress.current = false;
    _modelLoadingInProgress.resolves = [];
    _modelLoadingInProgress.rejects = [];
  }
};

// Global variable for face landmark positions array
TFFaceMesh.prototype.positionsArray = null;

/**
 * Isolates the two patches that correspond to the user's eyes
 * @param  {Object} video - the video element itself
 * @param  {Canvas} imageCanvas - canvas corresponding to the webcam stream
 * @param  {Number} width - of imageCanvas
 * @param  {Number} height - of imageCanvas
 * @return {Object} the two eye-patches, first left, then right eye
 */
TFFaceMesh.prototype.getEyePatches = async function(video, imageCanvas, width, height) {

  if (imageCanvas.width === 0) {
    return null;
  }

  // Load the MediaPipe facemesh model.
  if(!this.modelLoaded) await this.loadModel();
  
  const model = this.model;
  // useFullRangeModel(model);

  // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain an
  // array of detected faces from the MediaPipe graph.
  const predictions = await model.estimateFaces(imageCanvas, {
    flipHorizontal: false,
  });

  if (predictions.length === 0){
    this.positionsArray = null;
    return false;
  }

  // Save positions to global variable
  this.positionsArray = predictions[0].keypoints;
  const prediction = predictions[0]
  // const positions = this.positionsArray;

  // const { scaledMesh } = predictions[0];

  // Keypoints indexes are documented at
  // https://github.com/tensorflow/tfjs-models/blob/118d4727197d4a21e2d4691e134a7bc30d90deee/face-landmarks-detection/mesh_map.jpg
  // https://stackoverflow.com/questions/66649492/how-to-get-specific-landmark-of-face-like-lips-or-eyes-using-tensorflow-js-face
  const [leftBBox, rightBBox] = [
    // left
    {
      eyeTopArc: [466, 388, 387, 386, 385, 384, 398].map(ind => prediction.keypoints[ind]),
      eyeBottomArc: [263, 249, 390, 373, 374, 380, 381, 382, 362].map(ind => prediction.keypoints[ind])
    },
    // right
    {
      eyeTopArc: [246, 161, 160, 159, 158, 157, 173].map(ind => prediction.keypoints[ind]),
      eyeBottomArc: [33, 7, 163, 144, 145, 153, 154, 155, 133].map(ind => prediction.keypoints[ind])
    },
  ].map(({ eyeTopArc, eyeBottomArc }) => {
    const topLeftOrigin = {
      x: Math.round(Math.min(...eyeTopArc.map(v => v.x))),
      y: Math.round(Math.min(...eyeTopArc.map(v => v.y))),
    };
    const bottomRightOrigin = {
      x: Math.round(Math.max(...eyeBottomArc.map(v => v.x))),
      y: Math.round(Math.max(...eyeBottomArc.map(v => v.y))),
    };

    return {
      origin: topLeftOrigin,
      width: bottomRightOrigin.x - topLeftOrigin.x,
      height: bottomRightOrigin.y - topLeftOrigin.y,
    }
  });
  var leftOriginX = leftBBox.origin.x;
  var leftOriginY = leftBBox.origin.y;
  var leftWidth = leftBBox.width;
  var leftHeight = leftBBox.height;
  var rightOriginX = rightBBox.origin.x;
  var rightOriginY = rightBBox.origin.y;
  var rightWidth = rightBBox.width;
  var rightHeight = rightBBox.height;

  if (leftWidth === 0 || rightWidth === 0){
    console.log('an eye patch had zero width');
    return null;
  }

  if (leftHeight === 0 || rightHeight === 0){
    console.log('an eye patch had zero height');
    return null;
  }

  // Start building object to be returned
  var eyeObjs = {};

  // Use willReadFrequently for better performance with repeated getImageData calls
  var ctx = imageCanvas.getContext('2d', { willReadFrequently: true });
  
  var leftImageData = ctx.getImageData(leftOriginX, leftOriginY, leftWidth, leftHeight);
  eyeObjs.left = {
    patch: leftImageData,
    imagex: leftOriginX,
    imagey: leftOriginY,
    width: leftWidth,
    height: leftHeight
  };

  var rightImageData = ctx.getImageData(rightOriginX, rightOriginY, rightWidth, rightHeight);
  eyeObjs.right = {
    patch: rightImageData,
    imagex: rightOriginX,
    imagey: rightOriginY,
    width: rightWidth,
    height: rightHeight
  };

  this.predictionReady = true;

  return eyeObjs;
};

/**
 * Returns the positions array corresponding to the last call to getEyePatches.
 * Requires that getEyePatches() was called previously, else returns null.
 */
TFFaceMesh.prototype.getPositions = function () {
  return this.positionsArray;
}

/**
 * Reset the tracker to default values
 */
TFFaceMesh.prototype.reset = function(){
  console.log( "Unimplemented; Tracking.js has no obvious reset function" );
}

/**
 * Draw TF_FaceMesh_Overlay
 */
TFFaceMesh.prototype.drawFaceOverlay = function(ctx, keypoints){
  // If keypoints is falsy, don't do anything
  if (keypoints) {
    ctx.fillStyle = '#32EEDB';
    ctx.strokeStyle = '#32EEDB';
    ctx.lineWidth = 0.5;

    for (let i = 0; i < keypoints.length; i++) {
      const x = keypoints[i][0];
      const y = keypoints[i][1];

      ctx.beginPath();
      ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
      ctx.closePath();
      ctx.fill();
    }
  }
}

/**
 * The TFFaceMesh object name
 * @type {string}
 */
TFFaceMesh.prototype.name = 'TFFaceMesh';

export default TFFaceMesh;