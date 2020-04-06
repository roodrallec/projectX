(function mainLoop() {
  outContext.drawImage(inputVid, 0, 0);
  setTimeout(mainLoop, 1000 / FPS);
})();