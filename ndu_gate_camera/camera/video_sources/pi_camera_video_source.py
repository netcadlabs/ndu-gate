import numpy as np
from cv2 import cv2

from ndu_gate_camera.api.video_source import VideoSource, log
from ndu_gate_camera.utility.ndu_utility import NDUUtility

# try:
#     from picamera import PiCamera
# except ImportError:
#     print("picamera library not found - installing...")
#     if NDUUtility.install_package("picamera"):
#         import picamera
#
# try:
#     from picamera.array import PiRGBArray
# except ImportError:
#     print("picamera library not found - installing...")
#     if NDUUtility.install_package("picamera[array]"):
#         import picamera.array


class PiCameraVideoSource(VideoSource):
    def __init__(self, frame_width=640, frame_height=480, framerate=32, rotation=180, show_preview=False):
        super().__init__()
        self.__camera = None
        self.__frame_width = frame_width
        self.__frame_height = frame_height
        self.__framerate = framerate
        self.__rotation = rotation
        self.__show_preview = show_preview

        self._set_capture()

    def get_frames(self):
        log.info("start camera streaming..")
        frame_num = 0

        for frameFromCam in self.__camera.capture_continuous(self.__rawCapture, format="bgr", use_video_port=True):
            try:
                frame = np.copy(frameFromCam.array)
                frame_num += 1
                frame_h = frame.shape[0]
                frame_w = frame.shape[1]
                yield frame_num, frame
            except KeyboardInterrupt:
                self.__rawCapture.truncate(0)
                self.__camera.close()
                cv2.destroyAllWindows()
                log.info("exit from pi camera streaming")
                break

            log.info("finish pi camera streaming")

    def reset(self):
        pass

    def stop(self):
        pass

    def _set_capture(self):
        self.__camera = PiCamera()
        self.__camera.resolution = (self.__frame_width, self.__frame_height)
        self.__camera.framerate = self.__framerate
        self.__camera.rotation = self.__rotation
        self.__rawCapture = PiRGBArray(self.__camera, size=(self.__frame_width, self.__frame_height))
