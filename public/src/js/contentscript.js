/*
  Content script which injects the required js for webcam feed filters
*/

var docEl = (document.head||document.documentElement);

var div = document.createElement('div');
div.innerHTML = '<video id="inputVideo" autoplay hidden></video>';
div.innerHTML += '<canvas id="outCanvas" hidden></canvas>';
docEl.appendChild(div);

var getUserMediaOverload = document.createElement('script');
getUserMediaOverload.textContent = "\
var FPS = 30;\
var TINY_LM = '" + chrome.extension.getURL('src/ml-models/tiny_lm.json') + "';\
var TINY_DET = '" + chrome.extension.getURL('src/ml-models/tiny_det.json') + "';\
var SRC_IMG_URL = '"+ chrome.extension.getURL('src/images/face.png') + "';\
const inputVid = document.querySelector('#inputVideo');\
const outCanvas = document.querySelector('#outCanvas');\
const outContext = outCanvas.getContext('2d'); \
const canvasStream = outCanvas.captureStream();\
const getUserMedia = navigator.mediaDevices.getUserMedia;\
inputVid.style.display = 'none'; \
outCanvas.style.display = 'none'; \
navigator.mediaDevices.getUserMedia({  video: true }).then((stream) => {\
  const videoTracks = stream.getVideoTracks();\
  if (!videoTracks || !videoTracks.length) return; \
  const track = stream.getVideoTracks()[0].getSettings(); \
  FPS = track.frameRate; \
  outCanvas.height = track.height; \
  outCanvas.width = track.width; \
  inputVid.srcObject = stream; \
});\
navigator.mediaDevices.getUserMedia = (args) => {\
  return getUserMedia.bind(navigator.mediaDevices, args)().then((stream) => {\
    const audioTracks = stream.getAudioTracks();\
    if (audioTracks && audioTracks[0]) canvasStream.addTrack(audioTracks[0]);\
    return canvasStream;\
  });\
};\
";
docEl.appendChild(getUserMediaOverload);

function scriptFromFile(name) {
  var script = document.createElement('script');
  script.src = chrome.extension.getURL(name);
  script.async = false;
  return script;
}

function inject(scripts) {
  if (scripts.length === 0)
      return;
  var script = scripts[0];
  var otherScripts = scripts.slice(1);
  var el = (document.head || document.documentElement)
  el.appendChild(script);
  inject(otherScripts);
}

inject([
  scriptFromFile("src/js/filters/sockets.js"),
  // scriptFromFile("src/js/filters/landmark.js")
]);