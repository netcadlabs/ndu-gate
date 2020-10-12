from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner


class SocialDistanceRunner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.setName(config.get("name", 'DriverMonitorRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        self.__connector_type = connector_type

    # def open(self):
    #     self.__stopped = False
    #     self.start()

    def get_name(self):
        return "SocialDistanceRunner"

    def get_settings(self):
        settings = {'interval': 10,
                    'always': False,
                    'person': True}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)
        result = {
            "time": 1224124124,
            "count": 3
        }
        return result
