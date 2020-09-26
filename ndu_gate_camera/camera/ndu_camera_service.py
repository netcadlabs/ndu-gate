from os import path, uname
import logging
import logging.config
import logging.handlers

from yaml import safe_load
from simplejson import load

from ndu_gate_camera.api.video_source import VideoSourceType
from ndu_gate_camera.camera.result_handlers.result_handler_file import ResultHandlerFile
from ndu_gate_camera.camera.result_handlers.result_handler_socket import ResultHandlerSocket
from ndu_gate_camera.camera.video_sources.camera_video_source import CameraVideoSource
from ndu_gate_camera.camera.video_sources.file_video_source import FileVideoSource
from ndu_gate_camera.camera.video_sources.ip_camera_video_source import IPCameraVideoSource
from ndu_gate_camera.camera.video_sources.pi_camera_video_source import PiCameraVideoSource
from ndu_gate_camera.detectors.face_detector import FaceDetector
from ndu_gate_camera.detectors.person_detector import PersonDetector
from ndu_gate_camera.utility.ndu_utility import NDUUtility

name = uname()

DEFAULT_RUNNERS = {
    "drivermonitor": "DriverMonitorRunner",
    "socialdistance": "SocialDistanceRunner",
    "emotionanalysis": "EmotionAnalysisRunner",
}


class NDUCameraService:
    def __init__(self, gateway_config_file=None):
        if gateway_config_file is None:
            gateway_config_file = path.dirname(path.dirname(path.abspath(__file__))) + '/../config/tb_gateway.yaml'.replace('/', path.sep)

        with open(gateway_config_file) as general_config:
            self.__config = safe_load(general_config)
        self._config_dir = path.dirname(path.abspath(gateway_config_file)) + path.sep

        self.SOURCE_TYPE = VideoSourceType.CAMERA
        self.SOURCE_CONFIG = None
        if self.__config.get("video_source"):
            self.SOURCE_CONFIG = self.__config.get("video_source")
            type_str = self.SOURCE_CONFIG.get("type", "CAMERA")
            if VideoSourceType[type_str]:
                self.SOURCE_TYPE = VideoSourceType[type_str]
            else:
                self.SOURCE_TYPE = VideoSourceType.CAMERA

        logging_error = None
        try:
            import platform
            if platform.system() == "Darwin":
                logging.config.fileConfig(self._config_dir + "logs_macosx.conf", disable_existing_loggers=False)
            else:
                logging.config.fileConfig(self._config_dir + "logs.conf", disable_existing_loggers=False)
        except Exception as e:
            logging_error = e

        global log
        log = logging.getLogger('service')
        log.info("NDUCameraService starting...")

        # self.__result_handler = ResultHandlerFile()
        self.__result_handler = ResultHandlerSocket()

        self.PRE_FIND_PERSON = False
        self.PRE_FIND_FACES = False
        self.__personDetector = None
        self.__faceDetector = None
        try:
            self.__personDetector = PersonDetector()
        except Exception as e:
            log.error("Can not create person detector")
            log.error(e)
        try:
            self.__faceDetector = FaceDetector()
        except Exception as e:
            log.error("Can not create face detector")
            log.error(e)

        self._default_runners = DEFAULT_RUNNERS
        self._implemented_runners = {}
        self.available_runners = {}
        self.frame_num = 0

        self._load_runners()
        self._connect_with_connectors()

        self.video_source = None
        self._set_video_source()
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
                    if connector.get("runner", None) is None:
                        continue
                    connector_class = NDUUtility.check_and_import(connector["type"], self._default_runners.get(connector["type"], connector.get("class")))
                    self._implemented_runners[connector["type"]] = connector_class
                    config_file = self._config_dir + connector['configuration']

                    if not self.connectors_configs.get(connector['type']):
                        self.connectors_configs[connector['type']] = []

                    if path.isfile(config_file):
                        with open(self._config_dir + connector['configuration'], 'r', encoding="UTF-8") as conf_file:
                            connector_conf = load(conf_file)
                            connector_conf["name"] = connector["name"]
                            self.connectors_configs[connector['type']].append({"name": connector["name"], "config": {connector['configuration']: connector_conf}})
                    else:
                        log.error("config not found %s", config_file)
                        connector_conf = {"name": connector["name"]}
                        self.connectors_configs[connector['type']].append({"name": connector["name"], "config": {connector['configuration']: connector_conf}})
                except Exception as e:
                    log.error("Error on loading connector config")
                    log.exception(e)
        else:
            log.error("Connectors - not found! Check your configuration!")

    def _connect_with_connectors(self):
        """
        connectors_configs içindeki configleri kullanarak sırayla yüklenen runner sınıflarının instance'larını oluşturur
        oluşturulan bu nesneleri available_runners içerisine ekler.
        """
        for connector_type in self.connectors_configs:
            for connector_config in self.connectors_configs[connector_type]:
                for config in connector_config["config"]:
                    runner = None
                    try:
                        if connector_config["config"][config] is not None:
                            if self._implemented_runners[connector_type] is None:
                                log.error("implemented runner not found for %s", connector_type)
                            else:
                                runner = self._implemented_runners[connector_type](self, connector_config["config"][config], connector_type)
                                runner.setName(connector_config["name"])
                                settings = runner.get_settings()
                                if settings is not None:
                                    if settings.get("person", False):
                                        self.PRE_FIND_PERSON = True
                                    if settings.get("face", False):
                                        self.PRE_FIND_FACES = True
                                self.available_runners[runner.get_name()] = runner
                        else:
                            log.info("Config not found for %s", connector_type)
                    except Exception as e:
                        log.exception(e)
                        if runner is not None and NDUUtility.has_method(runner, 'close'):
                            runner.close()

    def _set_video_source(self):
        """
        SOURCE_TYPE değerine göre video_source değişkenini oluşturur.
        """
        if self.SOURCE_TYPE is VideoSourceType.VIDEO_FILE:
            self.SOURCE_CONFIG["test_data_path"] = path.dirname(path.dirname(path.abspath(__file__))) + '/data/'.replace('/', path.sep)
            self.video_source = FileVideoSource(self.SOURCE_CONFIG)
            pass
        elif self.SOURCE_TYPE is VideoSourceType.PI_CAMERA:
            self.video_source = PiCameraVideoSource(show_preview=True)
        elif self.SOURCE_TYPE is VideoSourceType.VIDEO_URL:
            # TODO
            pass
        elif self.SOURCE_TYPE is VideoSourceType.IP_CAMERA:
            self.video_source = IPCameraVideoSource(self.SOURCE_CONFIG)
            pass
        elif self.SOURCE_TYPE is VideoSourceType.CAMERA:
            self.video_source = CameraVideoSource(show_preview=True, device_index_name=0)
        elif self.SOURCE_TYPE is VideoSourceType.YOUTUBE:
            # TODO
            pass
        else:
            log.error("Video source type is not supported : %s ", self.SOURCE_TYPE.value)
            exit(101)

    def _start(self):
        if self.video_source is None:
            log.error("video source is not set!")
            exit(102)

        # TODO - çalıştırma sırasına göre sonuçlar bir sonraki runnera aktarılabilir
        # TODO - runner dependency ile kimin çıktısı kimn giridisi olacak şeklinde de olabilir
        for i, frame in self.video_source.get_frames():
            if i % 100 == 0:
                log.debug("frame count %s ", i)

            pedestrian_boxes = None
            num_pedestrians = None
            person_image_list = None
            face_list = None
            if self.PRE_FIND_PERSON:
                pedestrian_boxes, num_pedestrians, person_image_list = self.__personDetector.find_person(frame)
            if self.PRE_FIND_FACES:
                if len(person_image_list) > 0:
                    face_list = self.__faceDetector.face_detector3(frame, num_pedestrians)
                    # for face in face_list:
                    #     res = self._get_emotions_analysis(face)
                    #     log.debug(res)

            extra_data = {
                "pedestrian_boxes": pedestrian_boxes,
                "num_pedestrians": num_pedestrians,
                "person_image_list": person_image_list,
                "face_list": face_list
            }

            # TODO - check runner settings before send the frame to runner

            for current_connector in self.available_runners:
                try:
                    result = self.available_runners[current_connector].process_frame(frame, extra_data=extra_data)
                    log.debug("result : %s", result)

                    if result is not None:
                        self.__result_handler.add_result(result, runner_name=current_connector)

                except Exception as e:
                    log.exception(e)

            # print(frame)

        # TODO - set camera_perspective

    def _process_frame(self, frame):
        self.frame_num += 1
        print("Processing frame: ", self.frame_num)

        pass


if __name__ == '__main__':
    NDUCameraService(path.dirname(path.dirname(path.abspath(__file__))) + '/config/tb_gateway.yaml'.replace('/', path.sep))
