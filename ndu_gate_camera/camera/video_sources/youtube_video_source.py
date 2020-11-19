from os import path
import cv2
import vidgear
import time

from ndu_gate_camera.utility.ndu_utility import NDUUtility

try:
    from vidgear.gears import CamGear
except ImportError:
    if NDUUtility.install_package("vidgear") == 0:
        from vidgear.gears import CamGear

from ndu_gate_camera.api.video_source import VideoSource, log


class YoutubeVideoSource(VideoSource):
    def __init__(self, source_config):
        """

        :param source_config:
        """
        super().__init__()
        log.info("source_config %s", source_config)
        self.__show_preview = source_config.get("show_preview", False)
        self.__url = source_config.get("url", None)
        if self.__url is None:
            raise ValueError("Video url is empty")

        if not NDUUtility.is_url_valid(url=self.__url):
            raise ValueError("Video url is not valid:  %s0", self.__url)

        self.__sleep = source_config.get("sleep", 0)
        self.__cam_gear_options = source_config.get("cam_gear_options", {"CAP_PROP_FRAME_WIDTH ": 320, "CAP_PROP_FRAME_HEIGHT": 240, "CAP_PROP_FPS ": 1})

        self.__stream = None
        self._set_capture()

    def get_frames(self):
        log.debug("start video streaming..")
        count = 0
        # TODO - bitince başa sar?
        self.__stream.start()
        try:
            while True:
                frame = self.__stream.read()
                if frame is None:
                    break
                yield count, frame
                count += 1
                if self.__sleep > 0:
                    time.sleep(self.__sleep)
        except StopIteration:
            pass
        finally:
            self.stop()

    def reset(self):
        self.__stream.stop()
        self._set_capture()

    def stop(self):
        log.debug("Youtube video finished..")
        if self.__show_preview:
            cv2.destroyAllWindows()
        self.__stream.stop()

    def _set_capture(self):
        self.__stream = CamGear(source=self.__url, y_tube=True, time_delay=1, logging=True, **self.__cam_gear_options)
