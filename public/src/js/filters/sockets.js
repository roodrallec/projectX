/*
    Opens a websocket connection and streams cam to backend
    for filter application.
*/
const socket = new WebSocket("ws://192.168.1.131:8888/ws");
const offscreen = new OffscreenCanvas(256, 256);
const offscreenCtx = offscreen.getContext("2d");
const payload = {
  client_id: null,
  driving_img: null,
};
function sendToServer() {
  if (inputVid.videoWidth > 0) {
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
        reader.onloadend = function() {
            payload.driving_img = reader.result.split(',')[1];
            console.log("Sending to server", payload);
            var encoded = btoa(JSON.stringify(payload));
            socket.send(encoded);
        }
    });
  } else {
    return setTimeout(sendToServer, 1000);
  }
}
function onSocketMessage(msg) {
  console.log("Msg received ", msg.data);
  msg = JSON.parse(msg.data);
  if (msg.client_id) payload.client_id = msg.client_id;
  if (msg.output_img) {
    const img = new Image();
    img.onload = function () {
      outContext.drawImage(img, 0, 0);
    };
    img.src = msg.output_img;
  }
  if (msg.error) console.error(msg.error);
  sendToServer();
}
socket.onopen = sendToServer;
socket.onmessage = onSocketMessage;
socket.onclose = console.log;
socket.onerror = console.error;
