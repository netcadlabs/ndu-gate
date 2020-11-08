import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper
from ndu_gate_camera.utility.yolo_helper import yolo_helper


class yolov5_runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.input_size = config.get("input_size", 640)

        # self.onnx_fn = config.get("onnx_fn", "/data/yolov5s.onnx")
        # onnx_fn = "/data/yolov5s.onnx"
        # onnx_fn = "/data/yolov5m.onnx"
        onnx_fn = "/data/yolov5l.onnx"
        # onnx_fn = "/data/yolov5x.onnx"
        self.input_size = 640
        # python models/export.py --weights yolov5s.pt --img 640 --batch 1
        # Namespace(batch_size=1, img_size=[640, 640], weights='yolov5s.pt')

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = config.get("classes_filename", "coco.names")
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.sess_tuple = onnx_helper.create_sess_tuple(onnx_fn)

    def get_name(self):
        return "yolov5"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        return yolo_helper.predict_v5(self.sess_tuple, self.input_size, self.class_names, frame)

