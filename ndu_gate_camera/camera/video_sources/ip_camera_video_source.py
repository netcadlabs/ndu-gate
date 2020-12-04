import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class IPCameraVideoSource(VideoSource):
    def __init__(self, source_config):
        super().__init__()
        self.__video_url = source_config.get("url", None)
        if self.__video_url is None:
            raise ValueError("Video url is empty")
        self._set_capture()

    def get_frames(self):
        log.debug("start ip camera video streaming..")
        count = 0
        while self.__capture.isOpened():
            ret, frame = self.__capture.read()
            if ret is False:
                break

            yield count, frame
            count += 1

        self.__capture.release()
        cv2.destroyAllWindows()
        pass

    def reset(self):
        self._set_capture()

    def stop(self):
        # TODO
        pass

    def _set_capture(self):
        # TODO - https://www.pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/
        self.__capture = cap = cv2.VideoCapture(self.__video_url)
        pass