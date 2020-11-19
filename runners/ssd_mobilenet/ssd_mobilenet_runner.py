import errno
import os
from threading import Thread
from random import choice
from string import ascii_lowercase

import cv2

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants

import onnxruntime as rt
from os import path
import numpy as np


class SsdMobilenetRunner(Thread, NDUCameraRunner):

    def __init__(self, config, connector_type):
        super().__init__()
        # onnx_fn = path.dirname(path.abspath(__file__)) + "/data/ssd_mobilenet_v1_10.onnx"
        # class_names_fn = path.dirname(path.abspath(__file__)) + "/data/mscoco_label_map.pbtxt.json"
        #
        # if not path.isfile(onnx_fn):
        #     raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        # if not path.isfile(class_names_fn):
        #     raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)

        onnx_fn = config.get("onnx_fn", "ssd_mobilenet_v1_10.onnx")
        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = config.get("classes_filename", "mscoco_label_map.pbtxt.json")
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)

        def _create_session(onnx_fn, classes_fn):
            sess = rt.InferenceSession(onnx_fn)
            input_name = sess.get_inputs()[0].name
            outputs = sess.get_outputs()
            output_names = []
            for output in outputs:
                output_names.append(output.name)
            # class_names = [line.rstrip('\n') for line in open(classes_fn)]

            def _parse_pbtxt(file_name):
                import json
                with open(file_name) as json_file:
                    data = json.load(json_file)
                    class_names = {}
                    for item in data:
                        class_names[item["id"]] = item["display_name"]
                    return class_names

            class_names = _parse_pbtxt(classes_fn)
            return sess, input_name, output_names, class_names
        self.__onnx_sess, self.__onnx_input_name, self.__onnx_output_names, self.__onnx_class_names = _create_session(onnx_fn, classes_filename)

    def get_name(self):
        return "PersonCounterRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_data = np.expand_dims(image.astype(np.uint8), axis=0)

        result = self.__onnx_sess.run(self.__onnx_output_names, {self.__onnx_input_name: img_data})

        detection_boxes, detection_classes, detection_scores, num_detections = result
        h, w, *_ = image.shape
        out_boxes = []
        out_classes = []
        out_scores = []
        batch_size = num_detections.shape[0]
        for batch in range(batch_size):
            for detection in range(int(num_detections[batch])):
                class_index = int(detection_classes[batch][detection])
                out_classes.append(self.__onnx_class_names[class_index])
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
