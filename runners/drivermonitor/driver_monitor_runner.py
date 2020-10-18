import errno
from threading import Thread
from random import choice
from string import ascii_lowercase
import numpy as np
import cv2
import onnxruntime as rt
import os

import errno
from os import path

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class driver_monitor_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.statistics = {'MessagesReceived': 0,
                           'MessagesSent': 0}
        self.setName(config.get("name", 'DriverMonitorRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        self.__connector_type = connector_type

        self.__detect_talking_to_cellphone = config.get("detect_talking_to_cellphone", False)
        self.__detect_wearing_mask = config.get("detect_wearing_mask", False)
        self.__detect_smoking = config.get("detect_smoking", False)
        self.__detect_distracted = config.get("detect_distracted", False)

        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/googlenet-9.onnx"
        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/synset.txt"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)
        self.__onnx_sess, self.__onnx_input_name, self.__onnx_class_names, self.__onnx_output_name = self._create_session(onnx_fn, class_names_fn)

    def get_name(self):
        return "DriverMonitorRunner"

    def get_settings(self):
        settings = {'interval': 10,
                    'always': True}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        phone_face_padding = int((frame.shape[0] + frame.shape[1]) * 0.03)
        count_talking_to_cellphone = 0
        count_seatbelt = 0

        if extra_data is not None:
            results = extra_data.get("results", None)
            if results is not None:
                rect_persons = []
                rect_faces = []
                rect_cellphones = []
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "face":
                            rect_face = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_face is not None:
                                rect_faces.append(rect_face)
                        elif class_name == "cell phone":
                            rect_cellphone = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_cellphone is not None:
                                rect_cellphones.append(rect_cellphone)
                        elif class_name == "person":
                            rect_person = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_person is not None:
                                rect_persons.append(rect_person)
                rect_faces_padded = []
                for face in rect_faces:
                    cell_phone_handled = False
                    for cellphone in rect_cellphones:
                        if driver_monitor_runner._intersects(face, cellphone):
                            count_talking_to_cellphone += 1
                            cell_phone_handled = True
                            break
                    if not cell_phone_handled:
                        face1 = face.copy()
                        face1[0] -= phone_face_padding
                        face1[1] -= phone_face_padding
                        face1[2] += phone_face_padding
                        face1[3] += phone_face_padding
                        rect_faces_padded.append(face1)

                # index_phone = 487
                # index_seatbelt = 785
                count_talking_to_cellphone += self._get_count(frame, rect_faces_padded, 0.1, 487)
                count_seatbelt += self._get_count(frame, rect_persons, 0.1, 785)

        data = {}
        if count_talking_to_cellphone > 0:
            data['talking to phone count'] = count_talking_to_cellphone
        if count_seatbelt > 0:
            data['wearing seatbelt count'] = count_seatbelt

        if len(data) > 0:
            return [{constants.RESULT_KEY_DATA: data}]
        else:
            return []

    def _get_count(self, frame, boxes, threshold, index):
        count = 0
        for bbox in boxes:
            y1 = max(int(bbox[0]), 0)
            x1 = max(int(bbox[1]), 0)
            y2 = max(int(bbox[2]), 0)
            x2 = max(int(bbox[3]), 0)
            image = frame[y1:y2, x1:x2]

            blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (123.68, 116.779, 103.939))
            preds = self.__onnx_sess.run([self.__onnx_output_name], {self.__onnx_input_name: blob})[0]

            # cv2.imshow(str(index), image)
            # cv2.waitKey(100)

            score = preds[0][index]
            if score > threshold:
                count += 1
        return count

    # @staticmethod
    # def _preprocess(img):
    #
    #     # from mxnet.gluon.data.vision import transforms
    #     # transform_fn = transforms.Compose([
    #     #     transforms.Resize(256),
    #     #     transforms.CenterCrop(224),
    #     #     transforms.ToTensor(),
    #     #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     # ])
    #     # img = transform_fn(img)
    #     # img = img.expand_dims(axis=0)
    #     # return img

    @staticmethod
    def _create_session(onnx_fn, classes_filename):
        class_names = [line.rstrip('\n') for line in open(classes_filename)]

        sess = rt.InferenceSession(onnx_fn)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        return sess, input_name, class_names, output_name

    @staticmethod
    def _intersects(b1, b2):
        return rectangle(b1).intersects(rectangle(b2))


class rectangle:
    def intersects(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        return x1 < x2 and y1 < y2

    def _set(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __init__(self, bbox):
        self._set(bbox[0], bbox[1], bbox[2], bbox[3])
