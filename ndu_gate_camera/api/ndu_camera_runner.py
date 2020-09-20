from abc import ABC, abstractmethod

import logging

log = logging.getLogger("connector")


class NDUCameraRunner(ABC):

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_settings(self):
        pass

    @abstractmethod
    def process_frame(self, frame):
        pass
