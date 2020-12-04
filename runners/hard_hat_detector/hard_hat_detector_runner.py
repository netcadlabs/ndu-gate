import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import onnx_helper, yolo_helper, constants
from ndu_gate_camera.utility.geometry_helper import *
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class HardHatDetectorRunner(NDUCameraRunner):
    def __init__(self, config, _connector_type):
        super().__init__()

        # # onnx_fn = "/data/small_30.onnx"
        # onnx_fn = "/data/best.onnx"
        # # onnx_fn = "/data/last.onnx"
        # self.input_size = 640

        onnx_fn = "/data/yolov4_helmet.onnx"
        self.input_size = 416

        self.average_of_frames = config.get("average_of_frames", 1)
        self.confirm_count = config.get("confirm_count", 1)
        self.confirm_val = 0

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = "/data/class.names"
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = onnx_helper.parse_class_names(classes_filename)

        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.sess_tuple = onnx_helper.create_sess_tuple(onnx_fn)
        self.debug_mode = NDUUtility.is_debug_mode()
        self.last_data = None
        self.queue = []

    def get_name(self):
        return "hard_hat_detector"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        # res = yolo_helper.predict_v5(self.sess_tuple, self.input_size, self.class_names, frame)
        res = yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
        no_helmet_count = 0
        helmet_count = 0
        for i in range(len(res)):
            item = res[i]
            if constants.RESULT_KEY_CLASS_NAME in item:
                rect = item[constants.RESULT_KEY_RECT]
                p_rect = item["p_rect"] = add_padding_rect(rect, 0.1)
                score = item[constants.RESULT_KEY_SCORE]
                item["is_ok"] = True
                for j in range(i):
                    item1 = res[j]
                    if item1["is_ok"] and rects_intersect(p_rect, item1["p_rect"]):
                        score1 = item1[constants.RESULT_KEY_SCORE]
                        if score < score1:
                            item["is_ok"] = False
                        else:
                            item1["is_ok"] = False

        for i in range(len(res) - 1, -1, -1):
            item = res[i]
            if constants.RESULT_KEY_CLASS_NAME in item:
                if item["is_ok"]:
                    del item["is_ok"]
                    del item["p_rect"]
                else:
                    del res[i]

        res_for_qu = []
        for item in res:
            if constants.RESULT_KEY_CLASS_NAME in item:
                class_name = item[constants.RESULT_KEY_CLASS_NAME]
                rect = item[constants.RESULT_KEY_RECT]
                name_counts = {class_name: 1}
                for res1 in self.queue:
                    for item1 in res1:
                        rect1 = item1[constants.RESULT_KEY_RECT]
                        if rects_intersect(rect, rect1):
                            class_name1 = item1[constants.RESULT_KEY_CLASS_NAME]
                            if class_name1 in name_counts:
                                name_counts[class_name1] += 1
                            else:
                                name_counts[class_name1] = 1
                res_for_qu.append(item.copy())
                class_name_max = max(name_counts, key=name_counts.get)
                if class_name != class_name_max:
                    item[constants.RESULT_KEY_CLASS_NAME] = class_name_max

        self.queue.append(res_for_qu)
        if len(self.queue) > self.average_of_frames:
            self.queue.pop(0)

        for i in range(len(res) - 1, -1, -1):
            item = res[i]
            if constants.RESULT_KEY_CLASS_NAME in item:
                class_name = item[constants.RESULT_KEY_CLASS_NAME]
                if class_name == "helmet":
                    helmet_count += 1
                    item[constants.RESULT_KEY_RECT_COLOR] = [255, 55, 55]
                elif class_name == "no helmet":
                    no_helmet_count += 1
                    item[constants.RESULT_KEY_RECT_COLOR] = [0, 0, 255]
                else:
                    raise Exception("Bad class name: '{}'".format(class_name))

        data = {"no_helmet_count": no_helmet_count, "helmet_count": helmet_count, "no_helmet_exists": no_helmet_count > 0}
        if self.debug_mode and no_helmet_count > 0:
            res.append({constants.RESULT_KEY_DEBUG: "Baret takmayan var!"})
        if data != self.last_data:
            self.confirm_val -= 1
            if self.confirm_val <= 0:
                self.confirm_val = self.confirm_count
                self.last_data = data
                res.append({constants.RESULT_KEY_DATA: data})

        return res
