from time import sleep
import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class CameraVideoSource(VideoSource):
    def __init__(self, device_index_name=-2, show_preview=False):
        super().__init__()
        if device_index_name == -2:
            self.__capture = self._find_vide_capture()
        else:
            self.__capture = cv2.VideoCapture(device_index_name)
        self.__show_preview = show_preview

    def get_frames(self):
        log.info("start camera streaming..")
        count = 0

        while True:
            try:
                ret, frame = self.__capture.read()

                if frame is None:
                    continue

                if self.__show_preview:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    cv2.imshow('os default camera', rgb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        out = cv2.imwrite('capture.jpg', frame)
                        break

                count += 1
                yield count, frame
                sleep(0.1)
            except Exception as e:
                log.error(e)
                cv2.destroyAllWindows()
                break

        log.info("camera stream finished..")
        self.__capture.release()
        cv2.destroyAllWindows()
        pass

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
            cap = cv2.VideoCapture(index)
            r, fr = cap.read()
        return cap
