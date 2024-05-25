import * as tf from '@tensorflow/tfjs-core'

const createSsdAnchors = config => {
  // set defaults
  if (config.reduceBoxesInLowestLayer === null) {
    config.reduceBoxesInLowestLayer = false
  }
  if (config.interpolatedScaleAspectRatio === null) {
    config.interpolatedScaleAspectRatio = 1.0
  }
  if (config.fixedAnchorSize === null) {
    config.fixedAnchorSize = false
  }

  const anchors = []
  let layerId = 0
  while (layerId < config.numLayers) {
    const anchorHeight = []
    const anchorWidth = []
    const aspectRatios = []
    const scales = []

    // for same strides, we merge the anchors in the same order
    let lastSameStrideLayer = layerId
    while (
      lastSameStrideLayer < config.strides.length &&
      config.strides[lastSameStrideLayer] === config.strides[layerId]
    ) {
      const scale = calculateScale(
        config.minScale,
        config.maxScale,
        lastSameStrideLayer,
        config.strides.length,
      )
      if (lastSameStrideLayer === 0 && config.reduceBoxesInLowestLayer) {
        // for first layer, it can be specified to use predefined anchors
        aspectRatios.push(1)
        aspectRatios.push(2)
        aspectRatios.push(0.5)
        scales.push(0.1)
        scales.push(scale)
        scales.push(scale)
      } else {
        for (
          let aspectRatioId = 0;
          aspectRatioId < config.aspectRatios.length;
          ++aspectRatioId
        ) {
          aspectRatios.push(config.aspectRatios[aspectRatioId])
          scales.push(scale)
        }
        if (config.interpolatedScaleAspectRatio > 0.0) {
          const scaleNext =
            lastSameStrideLayer === config.strides.length - 1
              ? 1.0
              : calculateScale(
                  config.minScale,
                  config.maxScale,
                  lastSameStrideLayer + 1,
                  config.strides.length,
                )
          scales.push(Math.sqrt(scale * scaleNext))
          aspectRatios.push(config.interpolatedScaleAspectRatio)
        }
      }
      lastSameStrideLayer++
    }

    for (let i = 0; i < aspectRatios.length; ++i) {
      const ratioSqrts = Math.sqrt(aspectRatios[i])
      anchorHeight.push(scales[i] / ratioSqrts)
      anchorWidth.push(scales[i] * ratioSqrts)
    }

    let featureMapHeight = 0
    let featureMapWidth = 0
    if (config.featureMapHeight.length > 0) {
      featureMapHeight = config.featureMapHeight[layerId]
      featureMapWidth = config.featureMapWidth[layerId]
    } else {
      const stride = config.strides[layerId]
      featureMapHeight = Math.ceil(config.inputSizeHeight / stride)
      featureMapWidth = Math.ceil(config.inputSizeWidth / stride)
    }

    for (let y = 0; y < featureMapHeight; ++y) {
      for (let x = 0; x < featureMapWidth; ++x) {
        for (let anchorId = 0; anchorId < anchorHeight.length; ++anchorId) {
          const xCenter = (x + config.anchorOffsetX) / featureMapWidth
          const yCenter = (y + config.anchorOffsetY) / featureMapHeight

          const newAnchor = { xCenter, yCenter, width: 0, height: 0 }

          if (config.fixedAnchorSize) {
            newAnchor.width = 1.0
            newAnchor.height = 1.0
          } else {
            newAnchor.width = anchorWidth[anchorId]
            newAnchor.height = anchorHeight[anchorId]
          }
          anchors.push(newAnchor)
        }
      }
    }
    layerId = lastSameStrideLayer
  }

  return anchors
}

const calculateScale = (minScale, maxScale, strideIndex, numStrides) => {
  if (numStrides === 1) {
    return (minScale + maxScale) * 0.5
  } else {
    return minScale + ((maxScale - minScale) * strideIndex) / (numStrides - 1)
  }
}

const FULL_RANGE_TENSORS_TO_DETECTION_CONFIG = {
  applyExponentialOnBoxSize: false,
  flipVertically: false,
  ignoreClasses: [],
  numClasses: 1,
  numBoxes: 2304,
  numCoords: 16,
  boxCoordOffset: 0,
  keypointCoordOffset: 4,
  numKeypoints: 6,
  numValuesPerKeypoint: 2,
  sigmoidScore: true,
  scoreClippingThresh: 100.0,
  reverseOutputOrder: true,
  xScale: 192.0,
  yScale: 192.0,
  hScale: 192.0,
  wScale: 192.0,
  minScoreThresh: 0.6,
}

const FULL_RANGE_DETECTOR_ANCHOR_CONFIG = {
  reduceBoxesInLowestLayer: false,
  interpolatedScaleAspectRatio: 0.0,
  featureMapHeight: [],
  featureMapWidth: [],
  numLayers: 1,
  minScale: 0.1484375,
  maxScale: 0.75,
  inputSizeHeight: 192,
  inputSizeWidth: 192,
  anchorOffsetX: 0.5,
  anchorOffsetY: 0.5,
  strides: [4],
  aspectRatios: [1.0],
  fixedAnchorSize: true,
}

const FULL_RANGE_IMAGE_TO_TENSOR_CONFIG = {
  outputTensorSize: { width: 192, height: 192 },
  keepAspectRatio: true,
  outputTensorFloatRange: [-1, 1],
  borderMode: 'zero',
}

export const useFullRangeModel = model => {
  model.detector.imageToTensorConfig = FULL_RANGE_IMAGE_TO_TENSOR_CONFIG
  model.detector.tensorsToDetectionConfig =
    FULL_RANGE_TENSORS_TO_DETECTION_CONFIG
  model.detector.anchors = createSsdAnchors(FULL_RANGE_DETECTOR_ANCHOR_CONFIG)

  const anchorW = tf.tensor1d(model.detector.anchors.map(a => a.width))
  const anchorH = tf.tensor1d(model.detector.anchors.map(a => a.height))
  const anchorX = tf.tensor1d(model.detector.anchors.map(a => a.xCenter))
  const anchorY = tf.tensor1d(model.detector.anchors.map(a => a.yCenter))

  model.detector.anchorTensor = {
    x: anchorX,
    y: anchorY,
    w: anchorW,
    h: anchorH,
  }
}
