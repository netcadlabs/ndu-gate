from abc import ABC, abstractmethod

import logging
from enum import Enum

log = logging.getLogger("video_source")


class VideoSource(ABC):

    @abstractmethod
    def get_frames(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class VideoSourceType(Enum):
    VIDEO_FILE = 1
    PI_CAMERA = 2
    CAMERA = 3
    IP_CAMERA = 4
    VIDEO_URL = 5
    YOUTUBE = 6
