var docEl = (document.head||document.documentElement);

var div = document.createElement('div');
div.innerHTML = '<video id="inputVideo" autoplay hidden></video>';
div.innerHTML += '<canvas id="outCanvas" hidden></canvas>';
docEl.appendChild(div);

var getUserMediaOverload = document.createElement('script');
getUserMediaOverload.textContent = "\
var FPS;\
var VIDEO_WIDTH;\
var VIDEO_HEIGHT;\
var PIX2PIX_SIZE = 256;\
var tinyLmPath = '" + chrome.extension.getURL('src/ml-models/tiny_lm.json') + "';\
var tinyDetPath = '" + chrome.extension.getURL('src/ml-models/tiny_det.json') + "';\
var pix2pixPath = '" + chrome.extension.getURL('src/ml-models/pix2pix.pict') + "';\
const inputVid = document.querySelector('#inputVideo');\
const outCanvas = document.querySelector('#outCanvas');\
outCanvas.height = PIX2PIX_SIZE; \
outCanvas.width = PIX2PIX_SIZE; \
const canvasStream = outCanvas.captureStream();\
const getUserMedia = navigator.mediaDevices.getUserMedia;\
navigator.mediaDevices.getUserMedia({  video: true }).then((stream) => {\
    const videoTracks = stream.getVideoTracks();\
    if (!videoTracks || !videoTracks.length) {\
      return;\
    } \
    FPS = stream.getVideoTracks()[0].getSettings().frameRate; \
    VIDEO_WIDTH = stream.getVideoTracks()[0].getSettings().height; \
    VIDEO_HEIGHT = stream.getVideoTracks()[0].getSettings().width; \
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
  // script.parentNode.removeChild(script); needed?
  inject(otherScripts);
}

inject([
  scriptFromFile("lib/face-api.min.js"),
  scriptFromFile("lib/ml5.min.js"),
  scriptFromFile("src/js/vid-processor.js")
]);