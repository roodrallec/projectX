import imageio
import cv2
import torch
import numpy as np
import dlib
import pyfakewebcam
import imutils
from skimage.transform import resize
from imutils.video import WebcamVideoStream
from tqdm import tqdm
from demo import load_checkpoints
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from time import time


SOURCE_IMG_PATH = 'data/image.png'
FACE_DETECTOR = dlib.get_frontal_face_detector()
DLIB_DOWNSAMPLE = 4
DLIB_FULL_HEAD_BUFF = 1.0
TENSOR_SIZE = 256
INITIAL_RESET_S = 10
WEBCAM_H = 480
WEBCAM_W = 640
CV_SHOW = True


def portrait(face_bbox, frame):
    """
        Box encapsulating whole face and head
    """
    left, top, right, bot = face_bbox
    dy = (bot-top)
    buff = dy*DLIB_FULL_HEAD_BUFF
    y_c = top+(dy/2)
    y = y_c-buff
    y_end = y_c+buff
    dx = (right-left)
    x_c = left+(dx/2)
    x = x_c-buff
    x_end = x_c+buff

    # Out of bounds check
    if y < 0 or x < 0 or y_end > frame.shape[0] or x_end > frame.shape[1]:
        print("Portrait out of frame")
        return None

    return int(y), int(x), int(2*buff)


def face_portrait(frame, curr_portrait=None, downsample=DLIB_DOWNSAMPLE):
    """
        Face detect and box return
    """
    frame_resize = cv2.resize(
        frame, None, fx=1 / downsample, fy=1 / downsample)
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR(gray, 1)

    if not faces:
        return None

    face = faces[0]
    face_left, face_top, face_right, face_bot = np.array([
        face.left(), face.top(), face.right(), face.bottom()])*downsample

    if curr_portrait is None:
        return portrait((face_left, face_top, face_right, face_bot), frame)

    y, x, l = curr_portrait
    if face_top < y or face_bot > face_top+l:
        print("Y out of portrait")
        return None

    if face_left < x or face_right > x+l:
        print("X out of portrait")
        return None

    return curr_portrait


def load_source_img(path):
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


def main():
    canvas = np.zeros((WEBCAM_H, WEBCAM_W, 3)).astype(np.uint8)
    source_image = load_source_img(SOURCE_IMG_PATH)
    source_portrait = face_portrait(source_image)

    if source_portrait is None:
        raise Exception("No face detected in source img")

    sp_y, sp_x, sp_len = source_portrait
    source_face = source_image[sp_y:sp_y+sp_len, sp_x:sp_x+sp_len]
    source_face = resize(
        source_face, (TENSOR_SIZE, TENSOR_SIZE), anti_aliasing=True)
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                              checkpoint_path='models/vox-cpk.pth.tar')
    with torch.no_grad():
        source = torch.tensor(source_face[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2).cuda()
        kp_source = kp_detector(source)
        camera = pyfakewebcam.FakeWebcam('/dev/video2', 640, 480)
        vs = WebcamVideoStream(src=0).start()
        kp_driving_initial = None
        kp_driving_bbox = None

        while True:
            frame_start = time()
            frame = vs.read()
            kp_driving_bbox = face_portrait(
                frame, curr_portrait=kp_driving_bbox)

            if kp_driving_bbox is None:
                print('Face missing or out of bounds')
                kp_driving_initial = None
                continue

            drv_y, drv_x, drv_len = kp_driving_bbox
            driving_face = frame[drv_y:drv_y+drv_len, drv_x:drv_x+drv_len]
            driving_face = resize(
                driving_face, (TENSOR_SIZE, TENSOR_SIZE), anti_aliasing=True)

            driving_f = driving_face[np.newaxis].astype(np.float32)
            driving_f = np.round(driving_f, 2)
            driving_f = torch.tensor(driving_f).permute(0, 3, 1, 2).cuda()
            kp_driving = kp_detector(driving_f)

            if kp_driving_initial is None:
                kp_driving_initial = kp_driving
                continue

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=True,
                                   use_relative_jacobian=True, adapt_movement_scale=True)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            out = out['prediction'].data.cpu().numpy()
            out_img = np.transpose(out, [0, 2, 3, 1])[0]
            out_img = resize(out_img, (sp_len, sp_len), anti_aliasing=True)
            canvas[sp_y:sp_y+sp_len, sp_x:sp_x+sp_len] = out_img*255

            if CV_SHOW:
                cv2.imshow('source', source_face)
                cv2.imshow('driving', driving_face)
                cv2.imshow('final', canvas)
                wk = cv2.waitKey(1)
                if wk == ord('c'):
                    continue
                if wk == ord('q'):
                    break
            else:
                camera.schedule_frame(canvas)
                print(f"FPS: {1/(time() - frame_start)}")
                continue


if __name__ == '__main__':
    main()
