(function loop() {
  outContext.drawImage(inputVid, 0, 0);
  setTimeout(loop, 1000 / FPS);
}, 0);