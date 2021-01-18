import numpy as np
import cv2
import onnxruntime as rt
import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper, yolo_helper


class Yolov4TinyRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.input_size = config.get("input_size", 512)

        self.input_size = 416
        onnx_fn = "/data/yolov4-tiny_1_3_416_416_static.onnx"

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = config.get("classes_filename", "/data/coco.names")
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)

    def get_name(self):
        return "yolov4tiny"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
