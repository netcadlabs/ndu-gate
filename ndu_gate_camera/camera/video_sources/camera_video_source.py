from time import sleep
import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class CameraVideoSource(VideoSource):
    def __init__(self, source_config):
        super().__init__()
        self.__capture = self._find_vide_capture()
        # self.__show_preview = source_config.get("show_preview", False)
        self.__mirror = source_config.get("mirror", False)
        self.__sleep = source_config.get("sleep", 0)

    def get_frames(self):
        log.info("start camera streaming..")
        count = 0
        try:
            while self.__capture.isOpened():
                ok, frame = self.__capture.read()
                if not ok:
                    break
                if self.__mirror:
                    frame = cv2.flip(frame, 1)
                count += 1
                yield count, frame
                if self.__sleep > 0:
                    sleep(self.__sleep)
        except Exception as e:
            log.error(e)
        finally:
            log.info("camera stream finished..")
            self.__capture.release()
            cv2.destroyAllWindows()

    def reset(self):
        pass

    def stop(self):
        pass

    @staticmethod
    def _find_vide_capture():
        index = -1
        cap = cv2.VideoCapture(index)
        r, fr = cap.read()
        while fr is None:
            index += 1
            if index > 100:
                raise Exception('Cannot capture camera video')
            try:
                cap = cv2.VideoCapture(index)
                r, fr = cap.read()
            except:
                fr = None
        return cap
