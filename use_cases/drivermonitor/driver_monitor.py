from threading import Thread

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner


class DriverMonitorRunner(NDUCameraRunner, Thread):
    def get_name(self):
        return "DriverMonitorRunner"

    def get_settings(self):
        settings = {'interval': 10,
                    'always': True}
        return settings

    def process_frame(self, frame):
        result = {
            "time": 1224124124,
            "behav": "sleeping"
        }
        return result
