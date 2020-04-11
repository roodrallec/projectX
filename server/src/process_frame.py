import cv2
import numpy as np
import torch
import yaml
import face_alignment
import requests
import os
from scipy.spatial import ConvexHull
from skimage.transform import resize
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback
from time import time
from errors import MissingDrivingImage, PortraitOOB, MissingDrivingFace


DLIB_DOWNSAMPLE = 4
TENSOR_SIZE = 256
DEBUG = False
SRC_IMG_URL = os.environ.get(
    'SRC_IMG_URL', 'https://thispersondoesnotexist.com/image')
MODEL_CFG = os.environ.get('MODEL_CFG', 'models/vox-256.yaml')
MODEL_CKPT = os.environ.get('MODEL_PATH', 'models/vox-cpk.pth.tar')


class Frame():

    def __init__(self, client_id):
        self.initial_portrait = None
        self.initial_tensor = None
        self.source_tensor = None
        self.src_img_path = client_id


class FrameProcessor():
    def __init__(self):
        g, k = self.load_checkpoints(
            config_path=MODEL_CFG, checkpoint_path=MODEL_CKPT)
        self.generator = g
        self.kp_detector = k
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, device='cuda')

    def load_checkpoints(self, config_path, checkpoint_path):
        """ LOAD ML MODELS """
        with open(config_path) as f:
            config = yaml.load(f)

        generator = OcclusionAwareGenerator(
            **config['model_params']['generator_params'],
            **config['model_params']['common_params'])
        generator.cuda()
        kp_detector = KPDetector(
            **config['model_params']['kp_detector_params'],
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

    def normalize_kp(self, kp_source, kp_driving, kp_driving_initial,
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

    def fast_faces(self, img):
        """
            Fast face detector
        """
        st = time()
        img = cv2.resize(img, None, fx=1 / DLIB_DOWNSAMPLE,
                         fy=1 / DLIB_DOWNSAMPLE)
        faces = self.fa.face_detector.detect_from_image(img)
        print(f"face detect time: {np.round(time()-st, 2)}")
        if not faces:
            return []
        return [
            np.array([
                face[0], face[1], face[2], face[3]
            ])*DLIB_DOWNSAMPLE for face in faces
        ]

    def face_portrait(self, img, tube_bbox, increase_area=0.1):
        frame_shape = img.shape
        left, top, right, bot = tube_bbox
        width = right - left
        height = bot - top
        width_increase = max(
            increase_area, ((1+2*increase_area) * height-width) / (2 * width))
        height_increase = max(
            increase_area, ((1+2*increase_area) * width-height) / (2 * height))
        left = int(left - width_increase * width)
        top = int(top - height_increase * height)
        right = int(right + width_increase * width)
        bot = int(bot + height_increase * height)
        top, bot, left, right = max(0, top), min(
            bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
        return left, top, right, bot

    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def valid_bbox(self, bboxA, bboxB, iou_thresh=0.25):
        if self.bb_intersection_over_union(bboxA, bboxB) > iou_thresh:
            return True
        else:
            return False

    def cutout_portrait(self, img, portrait):
        f_left, f_top, f_right, f_bot = portrait
        return img[f_top:f_bot, f_left:f_right]

    def debug(self, img, fname='debug'):
        if DEBUG is True:
            cv2.imshow(fname, img)
            cv2.imwrite(fname + ".jpg", img)
            wk = cv2.waitKey(0)
            if wk == ord('c'):
                return
            if wk == ord('q'):
                quit()

    def download_source(self, save_name):
        r = requests.get(SRC_IMG_URL, headers={
                         'User-Agent': 'My User Agent 1.0'})
        with open(save_name, 'wb') as f:
            f.write(r.content)

    def load_source_tensor(self, src_img_name):
        if not os.path.isfile(src_img_name):
            self.download_source(src_img_name)

        src_img = cv2.imread(src_img_name)
        src_border = 200
        source_portrait = None
        src_img = cv2.copyMakeBorder(
            src_img, src_border, src_border, src_border, src_border,
            cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = self.fast_faces(src_img)

        if not faces or len(faces) == 0:
            raise Exception('Missing face in source')

        source_portrait = self.face_portrait(src_img, faces[0])
        src_tensor = resize(self.cutout_portrait(src_img, source_portrait),
                            (TENSOR_SIZE, TENSOR_SIZE))

        return src_tensor[np.newaxis].astype(np.float32)

    def process_frame(self, driving_image, frame):
        """
            images used to generate new face and returned
        """
        with torch.no_grad():
            if driving_image is None:
                raise MissingDrivingImage('Missing driving image')
            driving_faces = self.fast_faces(driving_image)
            if len(driving_faces) == 0:
                raise MissingDrivingFace(
                    'Missing or wrong number of driving faces')
            driving_face = driving_faces[0]

            if frame.source_tensor is None:
                frame.source_tensor = self.load_source_tensor(
                    frame.src_img_path)
                frame.source_tensor = torch.tensor(
                    frame.source_tensor).permute(0, 3, 1, 2).cuda()
                frame.kp_source = self.kp_detector(frame.source_tensor)

            if frame.initial_tensor is None:
                frame.initial_portrait = self.face_portrait(
                    driving_image, driving_face)
                initial_tensor = resize(self.cutout_portrait(
                    driving_image, frame.initial_portrait),
                    (TENSOR_SIZE, TENSOR_SIZE))
                frame.initial_tensor = torch.tensor(
                    initial_tensor[np.newaxis].astype(np.float32)
                ).permute(0, 3, 1, 2).cuda()
                frame.initial_kp = self.kp_detector(frame.initial_tensor)

            driving_portrait = self.face_portrait(driving_image, driving_face)

            if not self.valid_bbox(driving_portrait, frame.initial_portrait):
                raise PortraitOOB('Portrait OOB!')

            driving_cutout = self.cutout_portrait(
                driving_image, driving_portrait)
            driving_tensor = resize(driving_cutout, (TENSOR_SIZE, TENSOR_SIZE))
            driving_tensor = driving_tensor[np.newaxis].astype(np.float32)
            driving_tensor = torch.tensor(
                driving_tensor).permute(0, 3, 1, 2).cuda()
            kp_norm = self.normalize_kp(
                kp_source=frame.kp_source,
                kp_driving=self.kp_detector(driving_tensor),
                kp_driving_initial=frame.initial_kp,
                use_relative_movement=True,
                use_relative_jacobian=True,
                adapt_movement_scale=True)
            out = self.generator(frame.source_tensor,
                                 kp_driving=kp_norm, kp_source=frame.kp_source)
            out = out['prediction'].data.cpu().numpy()
            out_img = np.transpose(out, [0, 2, 3, 1])[0]*255

        return out_img.astype(np.uint8), frame
