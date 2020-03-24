console.log('FaceApi Landmark filter');

const drawRate = 15;
const useTiny = true;
const faceDetectOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 512,
    scoreThreshold: 0.5
});
const drawCanvas = document.createElement('canvas');
drawCanvas.height = outCanvas.height;
drawCanvas.width = outCanvas.width;

(function mainLoop() {
    drawLandmarks();
    setTimeout(mainLoop, 1000 / drawRate);
})();

function loadNets() {
    return Promise.all([
        faceapi.nets.faceLandmark68TinyNet.load(tinyLmPath),
        faceapi.nets.tinyFaceDetector.load(tinyDetPath)
    ]);
}


function drawLandmarks() {
    if (!faceapi.nets.tinyFaceDetector.params) return loadNets();

    console.log('detecting landmarks');
    faceapi
        .detectSingleFace(inputVid, faceDetectOptions)
        .withFaceLandmarks(useTiny)
        .then((detected) => {
            if (!detected) {
                console.log('no landmarks');
                return;
            }
            console.log('drawing landmarks');
            const resized = faceapi.resizeResults(detected, {
                width: outCanvas.width,
                height: outCanvas.height
            });
            outContext.fillStyle = 'white';
            outContext.fillRect(0, 0, outCanvas.width, outCanvas.height);
            new faceapi.draw.DrawFaceLandmarks(resized.landmarks, {
                drawLines: true,
                lineWidth: 3,
                lineColor: 'green',
                drawPoints: true
            }).draw(outCanvas);
        });
}
