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


class person_counter_runner(Thread, NDUCameraRunner):

    def __init__(self, config, connector_type):
        super().__init__()
        # self.setName(config.get("name", 'PersonCounterRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))

    def get_name(self):
        return "PersonCounterRunner"

    def get_settings(self):
        # settings = {'interval': 6, 'person': True, 'face': True}
        settings = {}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)

        person_count = 0
        if extra_data is not None:
            results = extra_data.get("results", None)
            if results is not None:
                rect_persons = []
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "person":
                            person_count += 1

        if person_count > 0:
            return [{
                constants.RESULT_KEY_DATA: {"person_count": person_count}
            }]
        else:
            return None
