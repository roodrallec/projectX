import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
import numpy as np
import cv2
from process_img import process
from time import time


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')

    def on_message(self, message):
        print('message received')
        nparr = np.fromstring(message, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        st = time()
        result = process(img_np)
        print(f"FPS: {int(1/(time() - st))}")
        if result is not None:
            success, encoded_image = cv2.imencode('.png', result)
            self.write_message(encoded_image.tobytes(), binary=True)
        else:
            self.write_message(b'', binary=True)

    def on_close(self):
        print('connection closed')

    def check_origin(self, origin):
        return True


application = tornado.web.Application([
    (r'/ws', WSHandler),
])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    myIP = socket.gethostbyname(socket.gethostname())
    print('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()
