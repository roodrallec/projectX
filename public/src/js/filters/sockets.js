/*
    Opens a websocket connection and streams cam to backend
    for filter application.
*/
const socket = new WebSocket("wss://localhost:8888/ws");
socket.binaryType = 'arraybuffer';
const offscreen = new OffscreenCanvas(0, 0);
const offscreenCtx = offscreen.getContext("2d");
const payload = {
  client_id: 'test-client',
  driving_img: null,
};
var fpsSt = new Date().getTime();
function sendToServer() {
  if (inputVid.videoWidth > 0) {
    offscreen.width = inputVid.videoWidth;
    offscreen.height = inputVid.videoHeight;
    offscreenCtx.drawImage(inputVid, 0, 0);
    return offscreen.convertToBlob().then((blob) => {
      socket.send(blob);
    });
  } else {
    return setTimeout(sendToServer, 1000);
  }
}
function onSocketMessage(msg) {
  console.log(
    "FPS: " + Math.round(1000 / (new Date().getTime() - fpsSt)).toString()
  );
  fpsSt = new Date().getTime();
  var imgBytes = new Uint8Array(msg.data);
  if (imgBytes.length == 0) return setTimeout(sendToServer, 3000);
  var img = new Image();
  img.onload = function(){
    outContext.drawImage(img, 0, 0);
  }
  img.src = URL.createObjectURL(new Blob([imgBytes.buffer]));
  sendToServer();
}
socket.onopen = sendToServer;
socket.onmessage = onSocketMessage;
socket.onclose = console.log;
socket.onerror = console.error;
