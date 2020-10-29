from threading import Thread
import numpy as np
import cv2
import onnxruntime as rt
import os

import errno
from os import path

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.ndu_utility import NDUUtility
from ndu_gate_camera.utility.image_helper import image_helper


class gender_age_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        def _create_session(onnx_fn):
            sess = rt.InferenceSession(onnx_fn)
            input_name = sess.get_inputs()[0].name
            outputs = sess.get_outputs()
            output_names = []
            for output in outputs:
                output_names.append(output.name)
            return sess, input_name, output_names

        super().__init__()

        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/weights.29-3.76_utk.hdf5.onnx"

        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        self.__onnx_sess, self.__onnx_input_name, self.__onnx_output_names = _create_session(onnx_fn)

    def get_name(self):
        return "gender_age_runner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        res = []
        results = extra_data.get("results", None)
        if results is not None:
            for runner_name, result in results.items():
                for item in result:
                    class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                    if class_name == "face":
                        rect_face = item.get(constants.RESULT_KEY_RECT, None)
                        if rect_face is not None:
                            # bbox = rect_face
                            # y1 = max(int(bbox[0]), 0)
                            # x1 = max(int(bbox[1]), 0)
                            # y2 = max(int(bbox[2]), 0)
                            # x2 = max(int(bbox[3]), 0)
                            # image = frame[y1:y2, x1:x2]

                            # padding_ratio = 0.15
                            padding_ratio = 0.3
                            bbox = rect_face
                            y1 = max(int(bbox[0]), 0)
                            x1 = max(int(bbox[1]), 0)
                            y2 = max(int(bbox[2]), 0)
                            x2 = max(int(bbox[3]), 0)
                            w = x2 - x1
                            h = y2 - y1
                            dw = int(w * padding_ratio)
                            dh = int(h * padding_ratio)
                            x1 -= dw
                            x2 += dw
                            y1 -= dh
                            y2 += dh
                            y1 = max(y1, 0)
                            x1 = max(x1, 0)
                            y2 = max(y2, 0)
                            x2 = max(x2, 0)
                            image = frame[y1:y2, x1:x2]

                            # # square
                            # padding_ratio = 0.75
                            # bbox = rect_face
                            # y1 = max(int(bbox[0]), 0)
                            # x1 = max(int(bbox[1]), 0)
                            # y2 = max(int(bbox[2]), 0)
                            # x2 = max(int(bbox[3]), 0)
                            # w = x2 - x1
                            # h = y2 - y1
                            # side = max(w, h)
                            # side *= padding_ratio
                            # center = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                            # x1 = center[0] - side
                            # x2 = center[0] + side
                            # y1 = center[1] - side
                            # y2 = center[1] + side
                            # y1 = int(max(y1, 0))
                            # x1 = int(max(x1, 0))
                            # y2 = int(max(y2, 0))
                            # x2 = int(max(x2, 0))
                            # image = frame[y1:y2, x1:x2]

                            # cv2.imshow("aaa", image)
                            # cv2.waitKey(100)

                            # test koray
                            item.pop(constants.RESULT_KEY_CLASS_NAME)
                            item.pop(constants.RESULT_KEY_SCORE)
                            item.pop(constants.RESULT_KEY_RECT)

                            input_shape = (1, 64, 64, 3)
                            image = image_helper.resize_best_quality(image, (64, 64))
                            img_data = np.array(image).astype(np.float32)
                            img_data = np.resize(img_data, input_shape)

                            pred = self.__onnx_sess.run(self.__onnx_output_names, {self.__onnx_input_name: img_data})

                            predicted_gender = pred[0][0]
                            ages = np.arange(0, 101).reshape(101, 1)
                            predicted_age = pred[1].dot(ages).flatten()

                            preview_key = "KADIN" if predicted_gender[0] > 0.5 else "ERKEK"
                            name = "{}-{}".format(preview_key, int(predicted_age))

                            res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: name, constants.RESULT_KEY_PREVIEW_KEY: preview_key})

        return res
