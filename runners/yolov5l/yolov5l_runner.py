import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper, yolo_helper


class Yolov5lRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.onnx_fn = "/data/yolov5l.onnx"
        classes_filename = "/data/coco.names"
        self.input_size = 640

        if not os.path.isfile(self.onnx_fn):
            self.onnx_fn = os.path.dirname(os.path.abspath(__file__)) + self.onnx_fn.replace("/", os.path.sep)

        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

    def get_name(self):
        return "yolov5l"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v5(self.onnx_fn, self.input_size, self.class_names, frame)
