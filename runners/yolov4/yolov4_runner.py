import numpy as np
import cv2
import onnxruntime as rt
import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper, yolo_helper


class Yolov4Runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.input_size = config.get("input_size", 512)

        #### OLD
        # # 416 msec üst üste var
        # self.input_size = 512
        # onnx_fn = "/data/yolov4_-1_3_512_512_dynamic.onnx"

        # # 550 msec başarılı (default)
        # self.input_size = 608
        # onnx_fn = "/data/yolov4_-1_3_608_608_dynamic.onnx"

        # ### başarısız
        # # 335 msec
        # self.input_size = 512
        # onnx_fn = "/data/old/yolov4-csp_1_3_512_512_static.onnx"

        # # 793 msec  üst üste çok var
        # self.input_size = 640
        # onnx_fn = "/data/yolov4x-mish_1_3_640_640_static.onnx"

        # ### OK
        # 550 msec  başarılı  - vino onnx 350
        self.input_size = 608
        onnx_fn = "/data/yolov4-1_3_608_608_static.onnx"


        # # 22 msec  güzel  - vino onnx 16
        # self.input_size = 416
        # onnx_fn = "/data/yolov4-tiny_1_3_416_416_static.onnx"



        # # vehicles
        # self.input_size = 416
        # onnx_fn = "/data/yolov4-tiny_vehicles_416_static.onnx"


        # # vehicles
        # self.input_size = 416
        # onnx_fn = "/data/yolov4-tiny_vehicle_lp_7_416_static.onnx"






        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = config.get("classes_filename", "/data/coco.names")
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)

    def get_name(self):
        return "yolov4"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
