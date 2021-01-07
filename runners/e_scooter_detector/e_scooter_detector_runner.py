import numpy as np
import cv2
import onnxruntime as rt
import os

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper, yolo_helper, geometry_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class EScooterDetectorRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config

        self.input_size = 416
        onnx_fn = "/data/yolov4_e_scooter.onnx"

        ## cpu'da daha hızlı ama false positive fazla
        # self.input_size = 640
        # onnx_fn = "/data/yolov5_e_scooter.onnx"

        onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_fn = "/data/class.names"
        classes_fn = os.path.dirname(os.path.abspath(__file__)) + classes_fn.replace("/", os.path.sep)

        self.class_names = onnx_helper.parse_class_names(classes_fn)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn, config.get("max_engine_count", 0))

        # region Demo
        self._last_sent = None
        self._last_candidate = None
        self._last_candidate_counter = 0
        # endregion

    def get_name(self):
        return "EScooterRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        res = []
        scooters = yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
        # scooters = yolo_helper.predict_v5(self.sess_tuple, self.input_size, self.class_names, frame)
        overload_count = 0
        if len(scooters) > 0:
            pnts = []
            for class_name, score, rect in NDUUtility.enumerate_results(extra_data, ["person"], False):
                if rect is not None and class_name == "person":
                    [y1, x1, y2, x2] = rect
                    p1 = int(x1 + (x2 - x1) * 0.5), int(y2)  # bottom
                    # cv2.circle(frame, p1, 5, (0, 255, 255), 5) # debug
                    pnts.append(p1)

            i = 0
            for item in scooters:
                rect = item["rect"]
                score = item["score"]
                [y1, x1, y2, x2] = rect
                # rect_half = [y1 + (y2 - y1) * 0.5, x1, y2, x2]
                rect_half = [y1 + (y2 - y1) * 0.6, x1, y2, x2]
                person_count = 0
                for pnt in pnts:
                    if geometry_helper.is_inside_rect(rect_half, pnt):
                        person_count += 1
                i += 1
                # data = {constants.RESULT_KEY_DATA: {"scooter{} binen kişi sayısı".format(str(i)): person_count}}

                res_item = {constants.RESULT_KEY_RECT: rect,
                            constants.RESULT_KEY_SCORE: score,
                            constants.RESULT_KEY_CLASS_NAME: "scooter",
                            constants.RESULT_KEY_RECT_DEBUG_TEXT: "{} kişi".format(person_count),
                            # constants.RESULT_KEY_DEBUG: "scooter{} binen kişi sayısı {}".format(str(i), person_count)
                            }
                if person_count > 1:
                    overload_count += 1
                    res_item[constants.RESULT_KEY_RECT_COLOR] = [0, 0, 255]
                    res_item[constants.RESULT_KEY_DEBUG] = "Scooter kural ihlali!"

                res.append(res_item)

        # region Demo
        # self._last_sent = None
        # self._last_candidate = None
        # self._last_candidate_counter = 0

        if overload_count != self._last_candidate:
            self._last_candidate = overload_count
            self._last_candidate_counter = 0
        else:
            self._last_candidate_counter += 1
            if self._last_candidate > 0:
                max_count = 3
            else:
                max_count = 30
            if self._last_candidate_counter > max_count:
                if self._last_sent != self._last_candidate:
                    data = {constants.RESULT_KEY_DATA: {"scooter overload count": self._last_candidate}}
                    res.append(data)
                    self._last_sent = self._last_candidate
                self._last_candidate = None
                self._last_candidate_counter = 0

        # endregion

        return res
