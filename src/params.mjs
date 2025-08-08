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
  //setting high width and height to get maximum resolution from the webcam with landscape enforcement
  camConstraints: { 
    video: { 
      width: { min: 640, ideal: 1920, max: 7680 }, 
      height: { min: 360, ideal: 1080, max: 4320 }, 
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
