import time
from os import path
import cv2

from ndu_gate_camera.api.video_source import VideoSource, log
from ndu_gate_camera.utility import file_helper


class ImageVideoSource(VideoSource):
    def __init__(self, source_config):
        super().__init__()
        log.info("source_config %s", source_config)
        image_path = source_config.get("image_path", None)
        if image_path is None:
            raise ValueError("Image file path is empty")

        if path.isfile(image_path) is False:
            # TODO - get default installation path using OS
            image_path = file_helper.path_join(source_config.get("data_folder", "var/lib/ndu_gate_camera/data/".replace('/', path.sep)), image_path)
            if path.isfile(image_path) is False:
                dir_name = image_path[:image_path.rfind(path.sep)]
                file_name = image_path[image_path.rfind(path.sep) + len(path.sep):]
                self._images = []
                if path.isdir(dir_name):
                    for name in file_helper.enumerate_files(dir_name, False, file_name):
                        self._images.append(file_helper.path_join(dir_name, name))
            else:
                self._images = [image_path]
        else:
            self._images = [image_path]

        if len(self._images) == 0:
            raise ValueError("Image file is not exist : %s", image_path)

    def get_frames(self):
        log.debug("start image streaming..")
        count = 0
        i = 0
        start = time.time()
        while True:
            try:
                image_path = self._images[i]
                frame = cv2.imread(image_path)
                yield count, frame
                count += 1
            except:
                start = 0
            if time.time() - start > 5:
                start = time.time()
                i += 1
                if i == len(self._images):
                    i = 0

    def reset(self):
        pass

    def stop(self):
        # TODO
        pass
