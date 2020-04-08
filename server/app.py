import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
import json
import uuid
from .process_img import process
from time import time

frames = {}


class Frame():

    def __init__(self, client_id):
        self.client_id = client_id
        self.driving_img = None
        self.initial_img = None
        self.inital_bbox = None
        self.source_img = None
        self.output_img = None


class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')

    def on_message(self, message):
        print('message received')
        json_dict = json.loads(message)
        client_id = json_dict.get('client_id', uuid.uuid4())
        frame = frames.get(client_id, Frame(client_id))
        frame.driving_img = json_dict.get('driving_img')
        st = time()
        process(frame)
        frames[frame.client_id] = frame
        print(f"FPS {1/(time()-st)}")
        self.write_message(json.dumps({
            "client_id": self.client_id,
            "output_img": self.output_img
        }))

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
