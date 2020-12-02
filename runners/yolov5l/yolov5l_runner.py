import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper, yolo_helper

class Yolov5lRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        onnx_fn = "/data/yolov5l.onnx"

        # # python -m onnxruntime_tools.optimizer_cli --input bert_large.onnx --output bert_large_fp16.onnx --num_heads 16 --hidden_size 1024 --float16
        #
        # # onnx_fn = "/data/optimized0.onnx" #############koray  fps:18/19
        # # python -m onnxruntime_tools.optimizer_cli --input yolov5l.onnx --output optimized0.onnx
        #
        # onnx_fn = "/data/optimized1.onnx" #############koray  fps:3
        # # python -m onnxruntime_tools.optimizer_cli --input yolov5l.onnx --output optimized1.onnx --use_gpu --opt_level 99

        classes_filename = "/data/coco.names"
        self.input_size = 640

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.sess_tuple = onnx_helper.create_sess_tuple(onnx_fn)

    def get_name(self):
        return "yolov5l"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v5(self.sess_tuple, self.input_size, self.class_names, frame)

