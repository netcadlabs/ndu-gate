import errno
import os
from threading import Thread
from random import choice
from string import ascii_lowercase

import cv2

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants, onnx_helper

import onnxruntime as rt
from os import path
import numpy as np


class SsdMobilenetRunner(Thread, NDUCameraRunner):

    def __init__(self, config, connector_type):
        super().__init__()
        onnx_fn = config.get("onnx_fn", "ssd_mobilenet_v1_10.onnx")
        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = config.get("classes_filename", "mscoco_label_map.pbtxt.json")
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)

        self.class_names = self._parse_pbtxt(classes_filename)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn, config.get("max_engine_count", 0))

    def _parse_pbtxt(self, file_name):
        import json
        with open(file_name, encoding='utf-8') as json_file:
            data = json.load(json_file)
            class_names = {}
            for item in data:
                class_names[item["id"]] = item["display_name"]
            return class_names

    def get_name(self):
        return "PersonCounterRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_data = np.expand_dims(image.astype(np.uint8), axis=0)

        result = onnx_helper.run(self.sess_tuple, [img_data])

        detection_boxes, detection_classes, detection_scores, num_detections = result
        h, w, *_ = image.shape
        out_boxes = []
        out_classes = []
        out_scores = []
        batch_size = num_detections.shape[0]
        for batch in range(batch_size):
            for detection in range(int(num_detections[batch])):
                class_index = int(detection_classes[batch][detection])
                out_classes.append(self.class_names[class_index])
                out_scores.append(detection_scores[batch][detection])
                box = detection_boxes[batch][detection]
                box[0] *= h
                box[1] *= w
                box[2] *= h
                box[3] *= w
                out_boxes.append(box)

        res = []
        for i in range(len(out_boxes)):
            res.append({constants.RESULT_KEY_RECT: out_boxes[i], constants.RESULT_KEY_SCORE: out_scores[i], constants.RESULT_KEY_CLASS_NAME: out_classes[i]})
        return res
