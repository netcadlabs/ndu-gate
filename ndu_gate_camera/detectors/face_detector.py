import errno
import os
import time
from os import path
import cv2

from ndu_gate_camera.detectors.vision.ssd.config.fd_config import define_img_size
from ndu_gate_camera.detectors.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from ndu_gate_camera.utility.constants import NDU_GATE_MODEL_FOLDER


class FaceDetector:

    def __init__(self, threshold=0.8, candidate_size=1000):
        label_path = path.dirname(path.abspath(__file__)) + "/data/vision/voc-model-labels.txt"
        face_model_path = path.dirname(path.abspath(__file__)) + "/data/vision/version-RFB-640.pth"

        if not path.isfile(label_path):
            label_path = NDU_GATE_MODEL_FOLDER + "/vision/voc-model-labels.txt"

        if not path.isfile(face_model_path):
            face_model_path = NDU_GATE_MODEL_FOLDER + "/vision/voc-model-labels.txt"

        if not path.isfile(label_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), label_path)

        if not path.isfile(face_model_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), face_model_path)

        define_img_size(480)
        test_device = "cpu"

        self.__class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.__threshold = threshold
        self.__candidate_size = candidate_size
        self.faceNetModel = create_Mb_Tiny_RFB_fd(len(self.__class_names), is_test=True, device=test_device)
        self.facePredictor = create_Mb_Tiny_RFB_fd_predictor(self.faceNetModel, candidate_size=self.__candidate_size, device=test_device)
        self.faceNetModel.load(self.face_model_path)

    def face_detector3(self, frame, person_counter):
        time_time = time.time()
        (h, w) = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.facePredictor.predict(image, self.__candidate_size / 2, self.__threshold)

        face_list = []
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            (startX, startY) = (max(0, int(box[0]) - 10), max(0, int(box[1]) - 10))
            (endX, endY) = (min(w - 1, int(box[2]) + 10), min(h - 1, int(box[3]) + 10))
            face_list.append(frame[startY:endY, startX:endX])

        return face_list