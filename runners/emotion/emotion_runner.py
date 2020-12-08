from threading import Thread
import numpy as np
import cv2
import os

import errno
from os import path

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, image_helper, onnx_helper


class EmotionRunner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.__connector_type = connector_type

        self.onnx_fn = path.dirname(path.abspath(__file__)) + "/data/emotion-ferplus-8.onnx"

        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/emotion.names"
        if not path.isfile(self.onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)

        self.class_names = onnx_helper.parse_class_names(class_names_fn)

    def get_name(self):
        return "EmotionRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        res = []
        if extra_data is not None:
            results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
            if results is not None:
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "face":
                            rect_face = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_face is not None:
                                bbox = rect_face
                                y1 = max(int(bbox[0]), 0)
                                x1 = max(int(bbox[1]), 0)
                                y2 = max(int(bbox[2]), 0)
                                x2 = max(int(bbox[3]), 0)
                                image = frame[y1:y2, x1:x2]

                                # # padding_ratio = 0.05
                                # padding_ratio = -0.2
                                # bbox = rect_face
                                # y1 = max(int(bbox[0]), 0)
                                # x1 = max(int(bbox[1]), 0)
                                # y2 = max(int(bbox[2]), 0)
                                # x2 = max(int(bbox[3]), 0)
                                # w = x2 - x1
                                # h = y2 - y1
                                # dw = int(w * padding_ratio)
                                # dh = int(h * padding_ratio)
                                # x1 -= dw
                                # x2 += dw
                                # y1 -= dh
                                # y2 += dh
                                # y1 = max(y1, 0)
                                # x1 = max(x1, 0)
                                # y2 = max(y2, 0)
                                # x2 = max(x2, 0)
                                # image = frame[y1:y2, x1:x2]

                                # input_shape = (1, 1, 64, 64)
                                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                # image =  cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
                                # img_data = np.array(image).astype(np.float32)
                                # img_data = np.resize(img_data, input_shape)

                                input_shape = (1, 1, 64, 64)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                image = image_helper.resize_best_quality(image, (64, 64))
                                img_data = np.array(image).astype(np.float32)
                                img_data = np.resize(img_data, input_shape)

                                # from PIL import Image
                                # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                # img = Image.fromarray(rgb)
                                # input_shape = (1, 1, 64, 64)
                                # img = img.resize((64, 64), Image.ANTIALIAS)
                                # img_data = np.array(img).astype(np.float32)
                                # img_data = np.resize(img_data, input_shape)

                                preds0 = onnx_helper.run(self.onnx_fn, [img_data])
                                preds = preds0[0][0]

                                index = int(np.argmax(preds))
                                score = preds[index]
                                emotion_name = self.class_names[index]

                                # #test
                                # item.pop(constants.RESULT_KEY_CLASS_NAME)
                                # item.pop(constants.RESULT_KEY_SCORE)
                                # item.pop(constants.RESULT_KEY_RECT)

                                res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: emotion_name})

        return res
