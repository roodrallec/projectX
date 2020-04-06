import dlib
import cv2
import numpy as np
import torch
import yaml
from scipy.spatial import ConvexHull
from skimage.transform import resize
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback


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


MODEL_CONFIG = './server/models/vox-256.yaml'
MODEL_CKPT = './server/models/vox-cpk.pth.tar'
FACE_DETECTOR = dlib.get_frontal_face_detector()
GENERATOR, KP_DETECTOR = load_checkpoints(
    config_path=MODEL_CONFIG,
    checkpoint_path=MODEL_CKPT)
DLIB_DOWNSAMPLE = 4
TENSOR_SIZE = 256
DEBUG = False


def normalize_kp(kp_source, kp_driving, kp_driving_initial,
                 use_relative_movement=True,
                 use_relative_jacobian=True,
                 adapt_movement_scale=False):

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


def fast_faces(img):
    """
        Fast face detector
    """
    frame_resize = cv2.resize(
        img, None, fx=1 / DLIB_DOWNSAMPLE, fy=1 / DLIB_DOWNSAMPLE)
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR(gray, 1)
    if not faces:
        return []
    return [
        np.array([
            face.left(), face.top(), face.right(), face.bottom()
        ])*DLIB_DOWNSAMPLE for face in faces
    ]


def face_portrait(frame, face_bbox, initial_portrait=None):
    """
        Return portrait capturing bbox face from frame and cutout from frame
    """
    if initial_portrait is None:
        b_left, b_top, b_right, b_bot = (0, 0, frame.shape[1], frame.shape[0])
        initial_portrait = portrait(face_bbox)
        f_left, f_top, f_right, f_bot = initial_portrait
    else:
        b_left, b_top, b_right, b_bot = initial_portrait
        f_left, f_top, f_right, f_bot = face_bbox

    if f_left < b_left or f_top < b_top or f_bot > b_bot or f_right > b_right:
        print("Warn: Face out of bounds")
        return None

    return initial_portrait


def portrait(face_bbox):
    """
        Returns BBox encapsulating whole face and head
    """
    left, top, right, bot = face_bbox
    dy = (bot-top)
    buff = dy
    y_c = top+(dy/2)
    y = y_c-buff
    y_end = y_c+buff
    dx = (right-left)
    x_c = left+(dx/2)
    x = x_c-buff
    x_end = x_c+buff
    return int(x), int(y), int(x_end), int(y_end)


def cutout_portrait(frame, portrait):
    f_left, f_top, f_right, f_bot = portrait
    return frame[f_top:f_bot, f_left:f_right]


def tensor_from_img(img, portrait):
    tensor = resize(
        cutout_portrait(img, portrait),
        (TENSOR_SIZE, TENSOR_SIZE), anti_aliasing=True)
    tensor = tensor[np.newaxis].astype(np.float32)
    return torch.tensor(tensor).permute(0, 3, 1, 2).cuda()


def debug(img, fname='debug'):
    if DEBUG is True:
        cv2.imshow(fname, img)
        cv2.imwrite(fname + ".jpg", img)
        wk = cv2.waitKey(0)
        if wk == ord('c'):
            return
        if wk == ord('q'):
            quit()


def process(input_img_arr):
    """
        Converts img from bytes, splits 50/50 with vertical line
        left half used as driving img, right half used as source
        images used to generate new face and returned
    """
    input_height, input_width, input_channels = input_img_arr.shape
    faces = fast_faces(input_img_arr)
    debug(input_img_arr)

    if len(faces) != 3:
        print("INFO: Missing or too many faces!")
        return None

    images = []
    bboxes = []
    for idx, face in enumerate(sorted(faces, key=lambda f: f[0])):
        (left, top, right, bot) = face
        left_bound = int(idx*input_width/3)
        right_bound = int((idx+1)*input_width/3)

        if left < left_bound or right > right_bound:
            print("WARN: Face OOB")
            return None

        images.append(input_img_arr[:, left_bound:right_bound])
        bboxes.append((left-left_bound, top, right-left_bound, bot))

    driving_img, driving_initial_img, source_img = images
    driving_bbox, driving_initial_bbox, source_bbox = bboxes
    driving_initial_portrait = face_portrait(driving_initial_img,
                                             driving_initial_bbox)
    driving_portrait = face_portrait(driving_img, driving_bbox,
                                     initial_portrait=driving_initial_portrait)
    source_portrait = face_portrait(source_img, source_bbox)

    if driving_initial_portrait is None or driving_portrait is None \
       or source_portrait is None:
        print("WARN: Portrait OOB!")
        print([driving_initial_portrait, driving_portrait, source_portrait])
        return None

    with torch.no_grad():
        source_tensor = tensor_from_img(source_img, source_portrait)
        kp_source = KP_DETECTOR(source_tensor)
        driving_tensor = tensor_from_img(driving_img, driving_portrait)
        driving_init_t = tensor_from_img(driving_initial_img,
                                         driving_initial_portrait)
        kp_norm = normalize_kp(kp_source=kp_source,
                               kp_driving=KP_DETECTOR(driving_tensor),
                               kp_driving_initial=KP_DETECTOR(driving_init_t),
                               use_relative_movement=True,
                               use_relative_jacobian=True,
                               adapt_movement_scale=True)
        out = GENERATOR(source_tensor, kp_source=kp_source, kp_driving=kp_norm)
        out = out['prediction'].data.cpu().numpy()
        out_img = np.transpose(out, [0, 2, 3, 1])[0]*255

    debug(out_img/255, 'out')
    sz = min(input_height, input_height)
    return resize(out_img, (sz, sz))
