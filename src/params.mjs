const params = {
  moveTickSize: 50,
  videoContainerId: 'webgazerVideoContainer',
  videoElementId: 'webgazerVideoFeed',
  videoElementCanvasId: 'webgazerVideoCanvas',
  faceOverlayId: 'webgazerFaceOverlay',
  faceFeedbackBoxId: 'webgazerFaceFeedbackBox',
  gazeDotId: 'webgazerGazeDot',
  videoViewerWidth: 320,
  videoViewerHeight: 240,
  faceFeedbackBoxRatio: 0.66,
  // View options
  showVideo: true,
  mirrorVideo: true,
  showFaceOverlay: true,
  showFaceFeedbackBox: true,
  showGazeDot: true,
  // Force best resolution with min constraints (browsers must respect min or fail)
  // Uses min: 1920x1080, falls back to 1280x720, then ideal-only for older cameras
  camConstraints: { 
    video: { 
      width: { min: 1920, ideal: 7680 }, 
      height: { min: 1080, ideal: 4320 }, 
      aspectRatio: { min: 1.33, ideal: 1.78, max: 2.33 },
      facingMode: "user" 
    } 
  },
  dataTimestep: 50,
  showVideoPreview: true,
  applyKalmanFilter: true,
  saveDataAcrossSessions: true,
  // Whether or not to store accuracy eigenValues, used by the calibration example file
  storingPoints: false,
  ////
  videoIsOn: false,
  trackEye: 'both',
};

export default params;
