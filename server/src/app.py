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
from process_frame import FrameProcessor, Frame
from errors import PortraitOOB, MissingDrivingFace


frames = {}
frame_processor = FrameProcessor()

CLIENT_ID = 'test'


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')
        self.fps = time()

    def on_message(self, message):
        print('message received')
        nparr = np.fromstring(message, np.uint8)
        driving_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # import ipdb; ipdb.set_trace();
        frame = frames.get(CLIENT_ID, Frame(CLIENT_ID))

        try:
            result, frame = frame_processor.process_frame(driving_img, frame)
            frames[CLIENT_ID] = frame
            if result is not None:
                print(f"FPS: {int(1/(time() - self.fps))}")
                success, encoded_image = cv2.imencode('.png', result)
                self.fps = time()
                self.write_message(encoded_image.tobytes(), binary=True)
            else:
                self.write_message(b'', binary=True)
        except MissingDrivingFace:
            self.write_message(b'', binary=True)
        except PortraitOOB:
            self.write_message(b'', binary=True)


    # def on_message(self, message):
    #     print('message received')
    #     json_dict = json.loads(base64.b64decode(message))

    #     driving_img_b64 = json_dict.get('driving_img')
    #     if not driving_img_b64 or len(driving_img_b64) == 0:
    #         self.write_message(json.dumps({"error": "No driving image"}))
    #         return

    #     client_id = json_dict.get('client_id')
    #     if not client_id:
    #         self.write_message(json.dumps({"client_id": str(uuid.uuid4())}))
    #         return

    #     nparr = np.fromstring(base64.b64decode(driving_img_b64), np.uint8)
    #     driving_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #     try:
    #         st = time()
    #         frame = frames.get(client_id, Frame(client_id))
    #         out_img, frame = frame_processor.process_frame(driving_img, frame)
    #         frames[client_id] = frame
    #         success, encoded_image = cv2.imencode('.png', out_img)
    #         print(f"Process fps: {int(1/(time()-st))}")
    #         self.write_message(json.dumps({
    #             "client_id": client_id,
    #             "output_img": base64.b64encode(encoded_image).decode()
    #         }))
    #     except MissingDrivingFace:
    #         frames.pop(client_id, None)
    #         self.write_message(json.dumps({
    #             "error": "Missing driving face, try looking at webcam!"
    #         }))
    #     except PortraitOOB:
    #         self.write_message(json.dumps({
    #             "client_id": client_id,
    #             "error": "Driving img out of bounds, try looking at webcam!"
    #         }))

    def on_close(self):
        print('connection closed')

    def check_origin(self, origin):
        return True


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        loader = tornado.template.Loader(".")
        self.write(loader.load("src/test.html").generate())


if __name__ == "__main__":
    application = tornado.web.Application([
        (r'/ws', WSHandler),
        (r"/test.html", MainHandler)
    ], debug=True)
    ssl_options_dict = {
        "certfile": os.environ.get('SSL_CERT', os.path.join("ssl", "ssl.crt")),
        "keyfile": os.environ.get('SSL_KEY', os.path.join("ssl", "ssl.key"))
    }
    http_server = tornado.httpserver.HTTPServer(application,
                                                ssl_options=ssl_options_dict)
    http_server.listen(os.environ.get('PORT', 8888))
    myIP = socket.gethostbyname(socket.gethostname())
    print('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()
