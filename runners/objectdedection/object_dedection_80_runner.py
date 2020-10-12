from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log


class ObjectDedection80Runner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.__config = config

    def get_name(self):
        return "ObjectDedection80Runner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)
        log.debug("ObjectDedection80Runner içindeyim")
        result = {
            "time": 1224124124,
            "count": 3
        }
        return result
