from os.path import isfile

from cv2 import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class FileVideoSource(VideoSource):
    def __init__(self, source_config):
        super().__init__()
        self.__video_path = source_config.get("file_path", None)
        if self.__video_path is None:
            raise ValueError("Video file path is empty")
        self._set_capture()

    def get_frames(self):
        log.debug("start video streaming..")
        count = 0
        # TODO - bitince başa sar?
        while self.__capture.isOpened():
            ret, frame = self.__capture.read()
            if ret is False:
                break

            # TODO - burayı kaldır ?
            cv2.imwrite('files/FileVideoSource_' + str(count) + '.jpg', frame)
            yield count, frame
            count += 1

        log.debug("video finished..")
        self.__capture.release()
        cv2.destroyAllWindows()
        pass

    def reset(self):
        self._set_capture()

    def stop(self):
        # TODO
        pass

    def _set_capture(self):
        if isfile(self.__video_path) is False:
            raise ValueError("Video file is not exist")
        self.__capture = cap = cv2.VideoCapture(self.__video_path)
