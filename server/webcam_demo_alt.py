import imageio
import cv2
import torch
import numpy as np
import os
import imutils
import yaml
import face_alignment

from urllib.request import urlopen, Request
from skimage.transform import resize
from imutils.video import WebcamVideoStream
from scipy.spatial import ConvexHull
from time import time
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback

# CONFIG
DLIB_DOWNSAMPLE = 4
DLIB_FULL_HEAD_BUFF = 1.0
TENSOR_SIZE = 256
INITIAL_RESET_S = 10
WEBCAM_H = 480
WEBCAM_W = 640
DEBUG = True
ZMQ_HOST = 'tcp://localhost:5555'
MODEL_CONFIG = './server/vox-256.yaml'
MODEL_PATH = './server/vox-cpk.pth.tar'
IMAGE_GEN_URL = 'https://thispersondoesnotexist.com/image'
IMAGE_PATH = '02.png'
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))


def normalize_kp(kp_source, kp_driving, kp_driving_initial):
    use_relative_movement = True
    use_relative_jacobian = True
    adapt_movement_scale = False

    if adapt_movement_scale:
        source_area = ConvexHull(
            kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving['jacobian'],
                torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(
                jacobian_diff, kp_source['jacobian'])

    return kp_new


def load_checkpoints(config_path, checkpoint_path):
    """ LOAD ML MODELS """
    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(
        **config['model_params']['generator_params'],
        **config['model_params']['common_params'])
    generator.cuda()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.cuda()
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    return generator, kp_detector


def load_source_img(path):
    if not os.path.exists(path):
        download_face_img(path)

    source_image = imageio.imread(path)[..., :3]
    h, w, colors = source_image.shape

    if colors != 3:
        raise Exception("Source img not in RGB color")

    if h > WEBCAM_H:
        source_image = imutils.resize(source_image, height=WEBCAM_H)
        h, w, colors = source_image.shape

    if w > WEBCAM_W:
        source_image = imutils.resize(source_image, width=WEBCAM_W)
        h, w, colors = source_image.shape

    return source_image


def download_face_img(save_path):
    req = Request(IMAGE_GEN_URL)
    req.add_header('Referer', "https://www.256kilobytes.com")
    req.add_header('User-Agent', "Some bot")
    resp = urlopen(req)
    content = resp.read()
    f = open(save_path, "bw")
    f.write(content)
    return


def acceptable_kp(fa, source_lm, driving, thresh=1):
    def normalize(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    kp_source = normalize(source_lm[0])
    kp_driving = fa.get_landmarks(255 * driving)

    if kp_driving is None or len(kp_driving) == 0:
        return False

    kp_driving = normalize(kp_driving[0])
    norm = (np.abs(kp_source - kp_driving) ** 2).sum()
    print(norm)
    return norm < thresh


def detect_face(fa, img):
    image = 255 * img
    faces = fa.face_detector.detect_from_image(image[..., ::-1])
    lm = fa.get_landmarks(image, detected_faces=faces)
    return faces, lm


def eye_center(lm):
    points = [lm[LEFT_EYE_POINTS], lm[RIGHT_EYE_POINTS]]
    dist = np.array(np.mean(points, axis=0))
    return (int(dist[0][0]), int(dist[0][1]))


def portrait(fa, frame):
    """
        TENSOR_SIZE Box encapsulating whole face and head
    """
    faces, lm = detect_face(fa, frame)

    if faces is None or lm is None:
        return None

    y_c, x_c = eye_center(lm[0])
    face_left, face_top, face_right, face_bot = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
    buff = face_bot - face_top
    cutout = frame[int(y_c-buff):int(y_c+buff),
                   int(x_c-buff):int(x_c+buff)]

    portrait = resize(cutout, (TENSOR_SIZE, TENSOR_SIZE), anti_aliasing=True)
    return np.round(portrait, 3)


def main():
    # context = zmq.Context()
    # footage_socket = context.socket(zmq.PUB)
    # footage_socket.connect(ZMQ_HOST)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    source_image = load_source_img(IMAGE_PATH)
    source_face = portrait(fa, source_image)

    if source_face is None:
        raise Exception("No face detected in source")

    generator, kp_detector = load_checkpoints(config_path=MODEL_CONFIG,
                                              checkpoint_path=MODEL_PATH)
    with torch.no_grad():
        source = torch.tensor(source_face[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2).cuda()
        kp_source = kp_detector(source)
        kp_driving_initial = None
        vs = WebcamVideoStream(src=0).start()

        driving_face = None
        while True:
            frame_start = time()
            frame = vs.read()

            if DEBUG:
                cv2.imshow('frame', frame)
                wk = cv2.waitKey(1)

            driving_face = portrait(fa, frame)

            if driving_face is None:
                continue

            driving_f = driving_face[np.newaxis].astype(np.float32)
            driving_f = torch.tensor(driving_f).permute(0, 3, 1, 2).cuda()
            kp_driving = kp_detector(driving_f)

            if kp_driving_initial is None:
                kp_driving_initial = kp_driving

            kp_norm = normalize_kp(kp_source, kp_driving, kp_driving_initial)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            out = out['prediction'].data.cpu().numpy()
            out_img = np.transpose(out, [0, 2, 3, 1])[0]*255
            out_img = out_img.astype(np.uint8)

            if DEBUG:
                cv2.imshow('source', source_face)
                cv2.imshow('driving', driving_face)
                cv2.imshow('final', out_img)
                wk = cv2.waitKey(1)
                if wk == ord('r'):
                    kp_driving_initial = None
                if wk == ord('q'):
                    cv2.destroyAllWindows()
                    break

            # footage_socket.send_string(base64.b64encode(frame))
            print(f"FPS: {1/(time() - frame_start)}")


if __name__ == '__main__':
    main()
