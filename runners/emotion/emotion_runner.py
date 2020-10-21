import errno
from threading import Thread
from random import choice
from string import ascii_lowercase
import numpy as np
import cv2
import onnxruntime as rt
import os

import errno
from os import path

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class emotion_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config
        self.__connector_type = connector_type

        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/emotion-ferplus-8.onnx"

        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/emotion.names"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)

        def _create_session(onnx_fn, classes_fn):
            sess = rt.InferenceSession(onnx_fn)
            input_name = sess.get_inputs()[0].name
            outputs = sess.get_outputs()
            output_names = []
            for output in outputs:
                output_names.append(output.name)
            class_names = [line.rstrip('\n') for line in open(classes_fn)]
            return sess, input_name, output_names, class_names

        self.__onnx_sess, self.__onnx_input_name, self.__onnx_output_names, self.__onnx_class_names = _create_session(onnx_fn, class_names_fn)

    def get_name(self):
        return "emotion_runner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        res = []
        if extra_data is not None:
            results = extra_data.get("results", None)
            if results is not None:
                # rect_faces = []
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "face":
                            rect_face = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_face is not None:
                                # rect_faces.append(rect_face)
                                bbox = rect_face
                                y1 = max(int(bbox[0]), 0)
                                x1 = max(int(bbox[1]), 0)
                                y2 = max(int(bbox[2]), 0)
                                x2 = max(int(bbox[3]), 0)
                                image = frame[y1:y2, x1:x2]

                                input_shape = (1, 1, 64, 64)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                # image = cv2.resize(image, (64, 64))
                                image =  cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
                                img_data = np.array(image).astype(np.float32)
                                img_data = np.resize(img_data, input_shape)


                                # from PIL import Image
                                # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                # img = Image.fromarray(rgb)
                                # input_shape = (1, 1, 64, 64)
                                # img = img.resize((64, 64), Image.ANTIALIAS)
                                # img_data = np.array(img).astype(np.float32)
                                # img_data = np.resize(img_data, input_shape)


                                preds0 = self.__onnx_sess.run(self.__onnx_output_names, {self.__onnx_input_name: img_data})
                                preds = preds0[0][0]

                                index = int(np.argmax(preds))
                                score = preds[index]
                                emotion_name = self.__onnx_class_names[index]

                                ####koray sil test
                                item.pop(constants.RESULT_KEY_CLASS_NAME)
                                item.pop(constants.RESULT_KEY_SCORE)
                                item.pop(constants.RESULT_KEY_RECT)

                                res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: emotion_name})

        return res
