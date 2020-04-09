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
import os
from time import time
from process_frame import process, Frame
from errors import PortraitOOB, MissingDrivingFace

frames = {}


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')

    def on_message(self, message):
        print('message received')
        message = base64.b64decode(message)
        json_dict = json.loads(message)
        driving_img_b64 = json_dict.get('driving_img')
        if not driving_img_b64 or len(driving_img_b64) == 0:
            self.write_message(json.dumps({"error": "No driving image"}))
            return
        client_id = json_dict.get('client_id')
        if not client_id:
            self.write_message(json.dumps({"client_id": str(uuid.uuid4())}))
            return
        blob = base64.b64decode(driving_img_b64)
        nparr = np.fromstring(blob, np.uint8)
        driving_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            st = time()
            out_img, frame = process(driving_img,
                                     frames.get(client_id, Frame(client_id)))
            print(f"Process fps: {int(1/(time()-st))}")
            frames[client_id] = frame
            success, encoded_image = cv2.imencode('.png', out_img)
            self.write_message(json.dumps({
                "client_id": client_id,
                "output_img": base64.b64encode(encoded_image).decode()
            }))
        except MissingDrivingFace:
            frames.pop(client_id, None)
            self.write_message(json.dumps({
                "error": "Missing driving face, try looking at webcam!"
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


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        loader = tornado.template.Loader(".")
        self.write(loader.load("test.html").generate())


application = tornado.web.Application([
    (r'/ws', WSHandler),
    (r"/test.html", MainHandler)
], debug=True)

if __name__ == "__main__":
    ssl_options_dict = {
        "certfile": os.path.join("server", "ssl.crt"),
        "keyfile": os.path.join("server", "ssl.key"),
    }
    http_server = tornado.httpserver.HTTPServer(application,
                                                ssl_options=ssl_options_dict)
    http_server.listen(8888)
    myIP = socket.gethostbyname(socket.gethostname())
    print('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()
