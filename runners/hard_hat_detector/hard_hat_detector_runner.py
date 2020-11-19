import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper
from ndu_gate_camera.utility.yolo_helper import yolo_helper


class hard_hat_detector_runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        # onnx_fn = "/data/hard_hat_ep1.onnx"
        onnx_fn = "/data/hard_hat_2.onnx"
        # onnx_fn = "/data/best.onnx"
        # onnx_fn = "/data/last.onnx"
        classes_filename = "/data/class.names"
        self.input_size = 640

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.sess_tuple = onnx_helper.create_sess_tuple(onnx_fn)

    def get_name(self):
        return "hard_hat_detector"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v5(self.sess_tuple, self.input_size, self.class_names, frame)

