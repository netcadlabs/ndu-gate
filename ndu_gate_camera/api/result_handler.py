from abc import ABC, abstractmethod

import logging

log = logging.getLogger("video_source")


class ResultHandler(ABC):

    @abstractmethod
    def save_result(self, result, runner_name=None):
        pass
