import threading
import time

import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class IPCameraVideoSource(VideoSource):
    def __init__(self, source_config):
        super().__init__()
        self.__video_url = source_config.get("url", None)
        if self.__video_url is None:
            raise ValueError("Video url is empty")
        self._set_capture()
        self.mode = source_config.get("mode", 1)  # 1 - QUEUE , 0 = SEND ALL
        self.count = 0
        self.frame = None
        if self.mode != 0:
            p1 = threading.Thread(target=self.read_frame)
            p1.start()

    def read_frame(self):
        while self.__capture.isOpened():
            ret, frame = self.__capture.read()
            if ret is False:
                break

            self.frame = frame
            self.count += 1

        self.count = -1
        self.__capture.release()

    def get_frames(self):
        log.debug("start ip camera video streaming..")

        if self.mode == 0:
            while self.__capture.isOpened():
                ret, frame = self.__capture.read()
                if ret is False:
                    break

                yield self.count, frame
                self.count += 1
        else:
            while self.count == 0:
                time.sleep(1)
            if self.count == -1:
                raise Exception('Cannot capture ip camera')
            while True:
                yield self.count, self.frame

        self.__capture.release()

    def reset(self):
        self._set_capture()

    def stop(self):
        # TODO
        pass

    def _set_capture(self):
        # TODO - https://www.pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/
        self.__capture = cap = cv2.VideoCapture(self.__video_url)
        pass
