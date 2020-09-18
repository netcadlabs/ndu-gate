from threading import Thread

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner


class SocialDistanceRunner(NDUCameraRunner, Thread):
    def __init__(self, camera_service, config, connector_type):
        super().__init__()
        self.__stopped = True
        pass

    def open(self):  # Function called by gateway on start
        self.__stopped = False
        self.start()

    def get_name(self):
        return "SocialDistanceRunner"

    def get_settings(self):
        settings = {'interval': 10,
                    'always': False}
        return settings

    def process_frame(self, frame):
        result = {
            "time": 1224124124,
            "count": 3
        }
        return result
