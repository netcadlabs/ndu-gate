from threading import Thread
import re
import sched
import time
from copy import copy
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner, log


class DriverMonitorRunner(Thread, NDUCameraRunner):
    def __init__(self, camera_service, config, connector_type):
        super().__init__()
        self.statistics = {'MessagesReceived': 0,
                           'MessagesSent': 0}
        self.setName(config.get("name", 'DriverMonitorRunner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__gateway = camera_service
        self.__config = config
        self.__gateway = camera_service
        self.__connector_type = connector_type
        self.__connected = False
        self.__stopped = True
        self.daemon = True

    def open(self):  # Function called by gateway on start
        self.__stopped = False
        self.start()

    def run(self):  # Main loop of thread
        try:
            while True:
                log.debug("test")
                ts = time.time()
                log.info("test %s", ts)
                time.sleep(float(self.__config.get("interval", 3)))
        except Exception as e:
            log.exception(e)

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
