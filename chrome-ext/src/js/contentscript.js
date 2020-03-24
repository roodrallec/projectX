var docEl = (document.head||document.documentElement);

var div = document.createElement('div');
div.innerHTML = '<video id="inputVideo" autoplay hidden></video>';
div.innerHTML += '<canvas id="outCanvas" hidden></canvas>';
docEl.appendChild(div);

var getUserMediaOverload = document.createElement('script');
getUserMediaOverload.textContent = "\
var FPS = 30;\
var tinyLmPath = '" + chrome.extension.getURL('src/ml-models/tiny_lm.json') + "';\
var tinyDetPath = '" + chrome.extension.getURL('src/ml-models/tiny_det.json') + "';\
const inputVid = document.querySelector('#inputVideo');\
inputVid.style.display = 'none'; \
const outCanvas = document.querySelector('#outCanvas');\
outCanvas.style.display = 'none'; \
const outContext = outCanvas.getContext('2d'); \
const canvasStream = outCanvas.captureStream();\
const getUserMedia = navigator.mediaDevices.getUserMedia;\
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
  scriptFromFile("lib/face-api.min.js"),
  scriptFromFile("src/js/filters/landmark.js")
]);