console.log('Video processor');

// Pix2pix
const drawCanvas = document.createElement('canvas');
drawCanvas.height = PIX2PIX_SIZE;
drawCanvas.width = 2*PIX2PIX_SIZE;

const pix2pix = ml5.pix2pix(pix2pixPath, modelLoaded);

function modelLoaded() {
    console.log('loaded test pix2pix');
}

// Face Detection Opts
const useTiny = true;
const faceDetectOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 512,
    scoreThreshold: 0.5
});

// Face processing
function process(inputVid, canvas) {
    const canvasContext = canvas.getContext('2d');
    canvasContext.fillStyle = 'black';
    canvasContext.fillRect(0, 0, canvas.width, canvas.height);
    canvasContext.drawImage(inputVid, PIX2PIX_SIZE, 0, PIX2PIX_SIZE, PIX2PIX_SIZE);
    drawLandmarks(inputVid, canvas);
}

function loadNets() {
    return Promise.all([
        faceapi.nets.faceLandmark68TinyNet.load(tinyLmPath),
        faceapi.nets.tinyFaceDetector.load(tinyDetPath)
    ]);
}

function drawLandmarks(inputVid, canvas, usePix2pix=false) {
    if (!faceapi.nets.tinyFaceDetector.params) return loadNets();

    console.log('detecting landmarks');
    faceapi
        .detectSingleFace(inputVid, faceDetectOptions)
        .withFaceLandmarks(useTiny)
        .then((detected) => {
            var overlayCtx = outCanvas.getContext("2d");
            overlayCtx.fillStyle = 'white';
            overlayCtx.fillRect(0, 0, outCanvas.width, outCanvas.height);

            if (!detected) {
                console.log('no landmarks');
                return;
            }
            console.log('drawing landmarks');

            const resized = faceapi.resizeResults(detected, {
                width: PIX2PIX_SIZE,
                height: PIX2PIX_SIZE
            });

            const drawLandmarks = new faceapi.draw.DrawFaceLandmarks(resized.landmarks, {
                drawLines: true,
                lineWidth: 3,
                lineColor: usePix2pix ? 'white' : 'black',
                drawPoints: false
            });

            if (usePix2pix){
                drawLandmarks.draw(canvas);
            } else {
                drawLandmarks.draw(outCanvas);
                return;
            }

            console.log('pix2pix transfer');
            pix2pix.transfer(canvas, function (err, result) {
                if (err) {
                    console.log(err);
                    return;
                }
                console.log('drawing pix2pix');
                var image = new Image();
                image.src = result.src;
                overlayCtx.drawImage(image, 0, 0);
            });
        });
}

// Loop processing of video
function videoLoop(canvas, fps=30) {
    (function loop() {
        process(inputVid, canvas);
        setTimeout(loop, 1000 / fps);
    })();
}

inputVid
    .addEventListener('play',
        videoLoop.bind(videoLoop, drawCanvas, FPS)
    , 0);
