from abc import ABC, abstractmethod

import logging

log = logging.getLogger("connector")


class NDUCameraRunner(ABC):
    frame_count = 0

    def __init__(self):
        self.frame_count = 0

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_settings(self):
        pass

    @abstractmethod
    def process_frame(self, frame, extra_data):
        """
        :param frame: tek bir frame datası
        :param extra_data pre process sırasında frameden çıkarılan veriler, person, face gibi key ler içeren dictionary
        """
        self.frame_count = self.frame_count + 1
        log.debug("%s processing frame %s", self.get_name(), self.frame_count)
