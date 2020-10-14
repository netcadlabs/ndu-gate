from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants


class PersonCounterRunner(Thread, NDUCameraRunner):

    def __init__(self, config, connector_type):
        super().__init__()
        self.setName(config.get("name", 'PersonCounterRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config

    def get_name(self):
        return "PersonCounterRunner"

    def get_settings(self):
        settings = {'interval': 6, 'person': True, 'face': True}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)
        num_pedestrians = extra_data.get("num_pedestrians", 0)

        if num_pedestrians > 0:
            return [{
                constants.RESULT_KEY_DATA: {"person_count": num_pedestrians}
            }]

        return None
