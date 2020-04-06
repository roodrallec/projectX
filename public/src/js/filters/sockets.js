/*
    Opens a websocket connection and streams cam to backend
    for filter application.
*/
var socket = new WebSocket('ws://192.168.1.131:8888/ws');
socket.binaryType = 'arraybuffer';
var offscreen = new OffscreenCanvas(256, 256);
var offscreenCtx = offscreen.getContext('2d');

function buildCanvas(reload, cb) {
    var cWidth = inputVid.videoWidth;
    var cHeight = inputVid.videoHeight;
    offscreenCtx.drawImage(inputVid, 0, 0);

    if (reload == false) return cb();

    offscreenCtx.drawImage(inputVid, cWidth, 0);
    var srcImg = new Image;
    srcImg.crossOrigin = 'Anonymous';
    srcImg.onload = function () {
        offscreenCtx.drawImage(srcImg,
            0, 0, srcImg.width, srcImg.height,
            cWidth*2, 0, cWidth, cHeight
        );
        cb();
    }
    srcImg.src = SRC_IMG_URL;
}
function sendToServer() {
    console.log('Sending to server');
    offscreen.convertToBlob().then(function(blob) {
        socket.send(blob);
    });
}
function displayToUser(imgBytes, cb) {
    console.log('displayToUser');
    var img = new Image();
    img.onload = function(){
        outContext.drawImage(img, 0, 0);
        cb();
    }
    img.src = URL.createObjectURL(new Blob([imgBytes.buffer]));
}
function onSocketOpen() {
    if (inputVid.videoHeight == 0) return setTimeout(onSocketOpen, 1000);
    console.log('video height' + inputVid.videoHeight);
    offscreen.width = inputVid.videoWidth * 3;
    offscreen.height = inputVid.videoHeight;
    buildCanvas(true, sendToServer);
}
socket.onopen = onSocketOpen;
socket.onclose = function(event) {
    console.log('Disconnected from WebSocket.');
};
socket.onerror = function(error) {
    console.error(error);
};
socket.onmessage = function(msg) {
    console.log('Msg received ')
    var bytes = new Uint8Array(msg.data);
    if (bytes.byteLength == 0) {
        buildCanvas(true, sendToServer);
    } else {
        displayToUser(bytes, buildCanvas.bind(this, false, sendToServer));
    }
};
