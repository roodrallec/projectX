from flask import Flask, send_from_directory
from flask_cors import CORS

# Flask App
APP = Flask(__name__, static_folder='public')
CORS(APP)

# Endpoints
@APP.route('/<path:path>')
def send_js(path):
    return send_from_directory('public', path)


@APP.route('/facejs', methods=['GET'])
def facejs():
    return APP.send_static_file('face-api.js')


@APP.route('/', methods=['GET'])
def index():
    return APP.send_static_file('index.html')


if __name__ == "__main__":
    APP.run(port=5000)
