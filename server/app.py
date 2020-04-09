import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
import json
import uuid
import base64
import numpy as np
import cv2
from process_frame import process, Frame
from errors import PortraitOOB
from time import time

frames = {}


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')

    def on_message(self, message):
        print('message received')
        st = time()
        message = base64.b64decode(message)

        json_dict = json.loads(message)
        driving_img_b64 = json_dict.get('driving_img')
        if not driving_img_b64 or len(driving_img_b64) == 0:
            return
        client_id = json_dict.get('client_id')
        client_id = client_id if client_id else str(uuid.uuid4())

        blob = base64.b64decode(driving_img_b64)
        nparr = np.fromstring(blob, np.uint8)
        driving_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            out_img, frame = process(driving_img,
                                     frames.get(client_id, Frame(client_id)))
            frames[client_id] = frame
            print(f"FPS {1/(time()-st)}")
            self.write_message(json.dumps({
                "client_id": client_id,
                "output_img": out_img
            }))
        except PortraitOOB:
            self.write_message(json.dumps({
                "client_id": client_id,
                "error": "Driving img out of bounds, try looking at webcam!"
            }))

    def on_close(self):
        print('connection closed')

    def check_origin(self, origin):
        return True


application = tornado.web.Application([
    (r'/ws', WSHandler),
], debug=True)

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    myIP = socket.gethostbyname(socket.gethostname())
    print('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()
