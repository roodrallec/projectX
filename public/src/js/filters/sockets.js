/*
    Opens a websocket connection and streams cam to backend
    for filter application.
*/
const socket = new WebSocket("wss://192.168.1.131:8888/ws");
const offscreen = new OffscreenCanvas(256, 256);
const offscreenCtx = offscreen.getContext("2d");
const payload = {
  client_id: null,
  driving_img: null,
};
var fpsSt = new Date().getTime();
function sendToServer() {
  if (inputVid.videoWidth > 0) {
    console.log(
      "FPS: " + Math.round(1000 / (new Date().getTime() - fpsSt)).toString()
    );
    fpsSt = new Date().getTime();
    offscreenCtx.drawImage(
      inputVid,
      0,
      0,
      inputVid.videoWidth,
      inputVid.videoHeight,
      0,
      0,
      offscreen.width,
      offscreen.height
    );
    return offscreen.convertToBlob().then((blob) => {
      var reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = function () {
        payload.driving_img = reader.result.split(",")[1];
        var encoded = btoa(JSON.stringify(payload));
        socket.send(encoded);
      };
    });
  } else {
    return setTimeout(sendToServer, 1000);
  }
}
function onSocketMessage(msg) {
  msg = JSON.parse(msg.data);
  if (msg.client_id) payload.client_id = msg.client_id;
  if (msg.output_img) {
    const img = new Image();
    img.onload = function () {
      outContext.drawImage(
        img,
        0,
        0,
        img.width,
        img.height,
        0,
        0,
        inputVid.videoWidth,
        inputVid.videoHeight
      );
    };
    img.src = "data:image/png;base64," + msg.output_img;
  }
  if (msg.error) console.warn(msg.error);
  sendToServer();
}
socket.onopen = sendToServer;
socket.onmessage = onSocketMessage;
socket.onclose = console.log;
socket.onerror = console.error;
