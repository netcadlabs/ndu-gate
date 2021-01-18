from threading import Thread
import numpy as np
import os

import errno
from os import path

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper


class GenderAgeRunner(Thread, NDUCameraRunner):
    def __init__(self, config, _connector_type):
        super().__init__()

        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/weights.29-3.76_utk.hdf5.onnx"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)

    def get_name(self):
        return "GenderAgeRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        res = []
        count_female = 0
        count_male = 0
        results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
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

                            # # test
                            # item.pop(constants.RESULT_KEY_CLASS_NAME)
                            # item.pop(constants.RESULT_KEY_SCORE)
                            # item.pop(constants.RESULT_KEY_RECT)

                            input_shape = (1, 64, 64, 3)
                            image = image_helper.resize_best_quality(image, (64, 64))
                            img_data = np.array(image).astype(np.float32)
                            img_data = np.resize(img_data, input_shape)

                            pred = onnx_helper.run(self.sess_tuple, [img_data])

                            predicted_gender = pred[0][0]
                            ages = np.arange(0, 101).reshape(101, 1)
                            predicted_age = pred[1].dot(ages).flatten()

                            preview_key = "KADIN" if predicted_gender[0] > 0.5 else "ERKEK"
                            name = "{}-{}".format(preview_key, int(predicted_age))

                            res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: name, constants.RESULT_KEY_PREVIEW_KEY: preview_key})
                            if predicted_gender[0] > 0.5:
                                count_female += 1
                            else:
                                count_male += 1

        data = {'female_count': count_female, 'male_count': count_male}
        res.append({constants.RESULT_KEY_DATA: data})
        return res
