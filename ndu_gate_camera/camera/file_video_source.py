from os.path import isfile

from cv2 import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class FileVideoSource(VideoSource):
    def __init__(self, video_path):
        super().__init__()
        self.__video_path = video_path
        if isfile(video_path) is False:
            raise ValueError("Video file is not exist")
        self.__capture = cap = cv2.VideoCapture(video_path)
        pass

    def get_frames(self):
        log.debug("start video streaming..")
        i = 0
        while self.__capture.isOpened():
            ret, frame = self.__capture.read()
            if ret is False:
                break
            cv2.imwrite('files/FileVideoSource_' + str(i) + '.jpg', frame)
            yield i, frame
            i += 1

        log.debug("video finished..")
        self.__capture.release()
        cv2.destroyAllWindows()
        pass

    def reset(self):
        pass

    def stop(self):
        pass
