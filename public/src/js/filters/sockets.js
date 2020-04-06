/*
    Opens a websocket connection and streams cam to backend
    for filter application.
*/
var socket = new WebSocket('ws://192.168.1.131:8888/ws');
socket.binaryType = 'arraybuffer';
var open = false;
var offscreen = new OffscreenCanvas(256, 256);
var offscreenCtx = offscreen.getContext('2d');

socket.onopen = function(event) {
    console.log('Socket open: ' + event);
    open = true
};
socket.onclose = function(event) {
    open = false;
    console.log('Disconnected from WebSocket.');
};
socket.onerror = function(error) {
    console.error(error);
};
socket.onmessage = function(msg) {
    console.log('Msg received ')
    var bytes = new Uint8Array(msg.data);
    var img = new Image();
    img.onload = function(){
        offscreenCtx.drawImage(img, 0, 0)
        outContext.drawImage(img, 0, 0)
    }
    img.src = URL.createObjectURL(new Blob([bytes.buffer]))
};
(function mainLoop() {
    if (open) {
        offscreen.width = inputVid.videoWidth * 3
        offscreen.height = inputVid.videoHeight
        offscreenCtx.drawImage(inputVid, 0, 0);
        offscreen.convertToBlob().then(function(blob) {
            const img = new Image();
            img.src = window.URL.createObjectUrl(blob);
            socket.send(blob);
        });
    }
    setTimeout(mainLoop, 1000);
})();

if len(faces) == 1:
driving_bbox = faces[0]
f_left, f_top, f_right, f_bot = driving_bbox
if f_left > int(w/3):
    print("INFO: Missing driving face")
    return
else:
    driving_img = img_np[0:, 0:int(w/3)]
    driving_initial = driving_img
    source_img = download_face_img()
    faces = fast_faces(source_img)
    source_face = face_portrait(source_img, faces[0])

    if source_face is None:
        print("INFO: Generated source has no face")
        return

    source_w_bg = np.zeros(driving_img.shape)
    length = min(driving_img.shape)
    source_face = resize(source_face, (length, length))
    source_w_bg[0:length, 0:length] = source_face
    return concatenate(driving_img, driving_initial, source_w_bg)
