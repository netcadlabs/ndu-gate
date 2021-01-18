import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper, yolo_helper, constants
from ndu_gate_camera.utility.geometry_helper import *
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class AerialVehicleRunner(NDUCameraRunner):
    def __init__(self, config, _connector_type):
        super().__init__()

        onnx_fn = "/data/yolov4_aerial_vehicle.onnx"
        self.input_size = 416

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = "/data/class.names"
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)

    def get_name(self):
        return "aerial_vehicle"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
