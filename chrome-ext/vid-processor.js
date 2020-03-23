console.log('Video processor');

// Face Api Paths
const tinyLmPath = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_tiny_model-weights_manifest.json';
const tinyDetPath = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json';
const useTinyModel = true;
const faceDetectOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 512,
    scoreThreshold: 0.5
});

// Face processing
function process(inputVid, canvas) {
    const canvasContext = canvas.getContext('2d');
    canvasContext.drawImage(inputVid, 0, 0);
    drawLandmarks(inputVid, canvas);
}

function loadNets() {
    return Promise.all([
        faceapi.nets.faceLandmark68TinyNet.load(tinyLmPath),
        faceapi.nets.tinyFaceDetector.load(tinyDetPath)
    ])
}

function drawLandmarks(inputVid, canvas) {
    if (!faceapi.nets.tinyFaceDetector.params) return loadNets();

    faceapi.nets.tinyFaceDetector
        .locateFaces(inputVid, faceDetectOptions)
        .then((detections) => {
            detections = faceapi.resizeResults(detections, {
                width: canvas.width,
                height: canvas.height
            });
            faceapi.draw.drawDetections(canvas, detections)
        });
}

// Loop processing of video
function videoLoop(canvas, fps=30) {
    (function loop() {
        process(inputVid, canvas);
        setTimeout(loop, 1000 / fps);
    })();
}

document
    .querySelector('#inputVideo')
    .addEventListener('play',
        videoLoop.bind(videoLoop, document.querySelector('#drawCanvas'), FPS)
    , 0);
