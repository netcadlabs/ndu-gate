import errno
import os
from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants

import onnxruntime as rt
from os import path
import numpy as np

from ndu_gate_camera.utility.ndu_utility import NDUUtility


class ClassCounterRunner(Thread, NDUCameraRunner):

    def __init__(self, config, connector_type):
        super().__init__()
        self._classes = config.get("classes", None)
        self._last_data = None
        self._debug = True

    def get_name(self):
        return "ClassCounterRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        counts = {}
        if extra_data is not None:
            results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
            if results is not None:
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if self._classes is None or class_name in self._classes:
                            if class_name not in counts:
                                counts[class_name] = 1
                            else:
                                counts[class_name] += 1

        res = []
        data = {}
        for class_name, count in counts.items():
            data["{}_count".format(class_name)] = count
        if self._classes is not None:
            for class_name in self._classes:
                data["{}_exists".format(class_name)] = class_name in counts

        if self._last_data != data:
            self._last_data = data
            res.append({constants.RESULT_KEY_DATA: data})
        if self._debug:
            for class_name, count in counts.items():
                class_name = NDUUtility.debug_conv_turkish(class_name)
                res.append({constants.RESULT_KEY_DEBUG: "{}: {}".format(class_name, count)})

        return res
