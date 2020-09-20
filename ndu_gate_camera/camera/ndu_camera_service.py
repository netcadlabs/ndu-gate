from os import path, uname
from time import sleep
import logging
import logging.config
import logging.handlers

from yaml import safe_load
from simplejson import load, dumps, loads

import cv2

from ndu_gate_camera.api.video_source import VideoSourceType
from ndu_gate_camera.camera.file_video_source import FileVideoSource
from ndu_gate_camera.utility.ndu_utility import NDUUtility

name = uname()

import numpy as np

DEFAULT_RUNNERS = {
    "drivermonitor": "DriverMonitorRunner",
    "socialdistance": "SocialDistanceRunner",
    "emotionanalysis": "EmotionAnalysisRunner",
}

SOURCE_TYPE = VideoSourceType.VIDEO_FILE


class NDUCameraService:
    def __init__(self, gateway_config_file=None):
        if gateway_config_file is None:
            gateway_config_file = path.dirname(path.dirname(path.abspath(__file__))) + '/../config/tb_gateway.yaml'.replace('/', path.sep)

        with open(gateway_config_file) as general_config:
            self.__config = safe_load(general_config)
        self._config_dir = path.dirname(path.abspath(gateway_config_file)) + path.sep

        logging_error = None
        try:
            logging.config.fileConfig(self._config_dir + "logs.conf", disable_existing_loggers=False)
        except Exception as e:
            logging_error = e

        global log
        log = logging.getLogger('service')
        log.info("NDUCameraService starting...")

        self._default_runners = DEFAULT_RUNNERS
        self._implemented_runners = {}
        self.available_connectors = {}
        self.frame_num = 0

        self._load_runners()
        self._connect_with_connectors()

        self.video_source = None
        self._set_camera()
        self._start()

    def _load_runners(self):
        """
        config dosyasında belirtilen NDUCameraRunner imaplementasyonlarını bulur ve _implemented_runners içerisine ekler
        Aynı şekilde herbir runner için config dosyalarını bulur ve connectors_configs içerisine ekler
        """
        self.connectors_configs = {}
        if self.__config.get("connectors"):
            for connector in self.__config['connectors']:
                try:
                    connector_class = NDUUtility.check_and_import(connector["type"], self._default_runners.get(connector["type"], connector.get("class")))
                    self._implemented_runners[connector["type"]] = connector_class
                    with open(self._config_dir + connector['configuration'], 'r', encoding="UTF-8") as conf_file:
                        connector_conf = load(conf_file)
                        if not self.connectors_configs.get(connector['type']):
                            self.connectors_configs[connector['type']] = []
                        connector_conf["name"] = connector["name"]
                        self.connectors_configs[connector['type']].append({"name": connector["name"], "config": {connector['configuration']: connector_conf}})
                except Exception as e:
                    log.error("Error on loading connector:")
                    log.exception(e)
        else:
            log.error("Connectors - not found! Check your configuration!")

    def _connect_with_connectors(self):
        """
        connectors_configs içindeki configleri kullanarak sırayla yüklenen runner sınıflarının instance'larını oluşturur
        oluşturulan bu nesneleri available_connectors içerisine ekler.
        """
        for connector_type in self.connectors_configs:
            for connector_config in self.connectors_configs[connector_type]:
                for config in connector_config["config"]:
                    connector = None
                    try:
                        if connector_config["config"][config] is not None:
                            connector = self._implemented_runners[connector_type](self, connector_config["config"][config], connector_type)
                            connector.setName(connector_config["name"])
                            self.available_connectors[connector.get_name()] = connector
                            # connector.open()
                        else:
                            log.info("Config not found for %s", connector_type)
                    except Exception as e:
                        log.exception(e)
                        if connector is not None and NDUUtility.has_method(connector, 'close'):
                            connector.close()

    def _set_camera(self):
        name = uname()
        if SOURCE_TYPE is VideoSourceType.VIDEO_FILE:
            video_path = path.dirname(path.dirname(path.abspath(__file__))) + '/data/duygu2.mp4'.replace('/', path.sep)
            self.video_source = FileVideoSource(video_path)
            pass
        elif SOURCE_TYPE is VideoSourceType.PI_CAMERA:
            # TODO
            pass
        elif SOURCE_TYPE is VideoSourceType.VIDEO_URL:
            pass
        else:
            log.error("Video source type is not supported : %s ", SOURCE_TYPE.value)

    def _start(self):
        if self.video_source is None:
            log.error("video source is not set!")
            pass

        for i, frame in self.video_source.get_frames():
            if i % 100 is 0:
                log.debug("frame count %s ", i)
            # print(frame)

        frameWidth = 640
        frameHeight = 480
        frame_num = 0
        is_rasp = False  # TODO dedect OS

        # TODO - set camera_perspective
        # if is_rasp:
        #     from picamera.array import PiRGBArray
        #     from picamera import PiCamera
        #     camera = PiCamera()
        #     camera.resolution = (frameWidth, frameHeight)
        #     camera.framerate = 32
        #     camera.rotation = 180
        #     rawCapture = PiRGBArray(camera, size=(frameWidth, frameHeight))
        #     for frameFromCam in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        #         try:
        #             frame = np.copy(frameFromCam.array)
        #             frame_num += 1
        #             frame_h = frame.shape[0]
        #             frame_w = frame.shape[1]
        #             self._process_frame(frame)
        #         except KeyboardInterrupt:
        #             rawCapture.truncate(0)
        #             camera.close()
        #             cv2.destroyAllWindows()
        #             print("exit")
        #             break
        # else:
        #     cap = cv2.VideoCapture(0)
        #     while True:
        #         try:
        #             ret, frame = cap.read()
        #             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        #
        #             cv2.imshow('macbook pro cam', rgb)
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 out = cv2.imwrite('capture.jpg', frame)
        #                 break
        #
        #             self._process_frame(frame)
        #             sleep(1)
        #         except:
        #             cv2.destroyAllWindows()
        #             print("exit")
        #             break

    def _process_frame(self, frame):
        self.frame_num += 1
        print("Processing frame: ", self.frame_num)

        pass


if __name__ == '__main__':
    NDUCameraService(path.dirname(path.dirname(path.abspath(__file__))) + '/config/tb_gateway.yaml'.replace('/', path.sep))
