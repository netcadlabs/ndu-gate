from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class driver_monitor_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.statistics = {'MessagesReceived': 0,
                           'MessagesSent': 0}
        self.setName(config.get("name", 'DriverMonitorRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        self.__connector_type = connector_type

        self.__detect_talking_to_cellphone = config.get("detect_talking_to_cellphone", False)
        self.__detect_wearing_mask = config.get("detect_wearing_mask", False)
        self.__detect_smoking = config.get("detect_smoking", False)
        self.__detect_distracted = config.get("detect_distracted", False)

    def get_name(self):
        return "DriverMonitorRunner"

    def get_settings(self):
        settings = {'interval': 10,
                    'always': True}
        return settings

    def process_frame(self, frame, extra_data):
        super().process_frame(frame)

        count_talking_to_cellphone = 0

        if extra_data is not None:
            results = extra_data.get("results", None)
            if results is not None:
                rect_faces = []
                rect_cellphones = []
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "face":
                            rect_face = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_face is not None:
                                rect_faces.append(rect_face)
                        elif class_name == "cell phone":
                            rect_cellphone = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_cellphone is not None:
                                rect_cellphones.append(rect_cellphone)
                for face in rect_faces:
                    for cellphone in rect_cellphones:
                        if driver_monitor_runner._intersects(face, cellphone):
                            count_talking_to_cellphone += 1

        data = {}
        if count_talking_to_cellphone > 0:
            data['talking to phone count'] = count_talking_to_cellphone

        if len(data) > 0:
            return [{constants.RESULT_KEY_DATA: data}]
        else:
            return []

    @staticmethod
    def _intersects(b1, b2):
        return rectangle(b1).intersects(rectangle(b2))


class rectangle:
    def intersects(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        return x1 < x2 and y1 < y2

    def _set(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __init__(self, bbox):
        self._set(bbox[0], bbox[1], bbox[2], bbox[3])
















