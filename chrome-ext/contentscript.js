var docEl = (document.head||document.documentElement);

var div = document.createElement('div');
div.innerHTML = '<video id="inputVideo" autoplay hidden></video>';
div.innerHTML += '<canvas id="drawCanvas" width="512" height="512" hidden></canvas>';
docEl.appendChild(div);

var getUserMediaOverload = document.createElement('script');
getUserMediaOverload.textContent = "\
var FPS; \
const inputVid = document.querySelector('#inputVideo');\
const canvas = document.querySelector('#drawCanvas');\
const canvasStream = canvas.captureStream();\
const getUserMedia = navigator.mediaDevices.getUserMedia;\
navigator.mediaDevices.getUserMedia({  video: true }).then((stream) => {\
    const videoTracks = stream.getVideoTracks();\
    if (!videoTracks || !videoTracks.length) {\
      return;\
    } \
    canvas.height = videoTracks[0].getSettings().height; \
    canvas.width = stream.getVideoTracks()[0].getSettings().width; \
    FPS = stream.getVideoTracks()[0].getSettings().frameRate; \
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
    scriptFromFile("face-api.min.js"),
    scriptFromFile("vid-processor.js")
]);