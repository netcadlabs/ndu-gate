from os import path
import cv2

from ndu_gate_camera.api.video_source import VideoSource, log


class ImageVideoSource(VideoSource):
    def __init__(self, source_config):
        super().__init__()
        log.info("source_config %s", source_config)
        self.__image_path = source_config.get("image_path", None)
        if self.__image_path is None:
            raise ValueError("Image file path is empty")

        if path.isfile(self.__image_path) is False:
            # TODO - get default installation path using OS
            self.__image_path = source_config.get("data_folder", "var/lib/ndu_gate_camera/data/".replace('/',
                                                                                                         path.sep)) + self.__image_path

        if path.isfile(self.__image_path) is False:
            raise ValueError("Image file is not exist : %s", self.__image_path)


    def get_frames(self):
        log.debug("start image streaming..")
        count = 0
        while True:
            frame = cv2.imread(self.__image_path)
            yield count, frame
            count += 1



    def reset(self):
        pass

    def stop(self):
        # TODO
        pass
