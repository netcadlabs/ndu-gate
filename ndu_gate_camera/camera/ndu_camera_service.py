from os import path, uname
import logging
import logging.config
import logging.handlers

from yaml import safe_load
from simplejson import load

from ndu_gate_camera.api.video_source import VideoSourceType
from ndu_gate_camera.camera.frame_pre_processors import FramePreProcessors
from ndu_gate_camera.camera.runner_result_handler import RunnerResultHandler
from ndu_gate_camera.camera.video_sources.camera_video_source import CameraVideoSource
from ndu_gate_camera.camera.video_sources.file_video_source import FileVideoSource
from ndu_gate_camera.camera.video_sources.pi_camera_video_source import PiCameraVideoSource
from ndu_gate_camera.utility.ndu_utility import NDUUtility

name = uname()

DEFAULT_RUNNERS = {
    "drivermonitor": "DriverMonitorRunner",
    "socialdistance": "SocialDistanceRunner",
    "emotionanalysis": "EmotionAnalysisRunner",
}

SOURCE_TYPE = VideoSourceType.CAMERA


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

        self.PRE_FIND_PERSON = False
        self.PRE_FIND_FACES = False

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
        oluşturulan bu nesneleri available_runners içerisine ekler.
        """
        for connector_type in self.connectors_configs:
            for connector_config in self.connectors_configs[connector_type]:
                for config in connector_config["config"]:
                    runner = None
                    try:
                        if connector_config["config"][config] is not None:
                            runner = self._implemented_runners[connector_type](self, connector_config["config"][config], connector_type)
                            runner.setName(connector_config["name"])
                            settings = runner.get_settings()
                            if settings is not None:
                                if settings.get("find_person", False):
                                    self.PRE_FIND_PERSON = True
                            self.available_runners[runner.get_name()] = runner
                            # connector.open()
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
        if SOURCE_TYPE is VideoSourceType.VIDEO_FILE:
            video_path = path.dirname(path.dirname(path.abspath(__file__))) + '/data/duygu2.mp4'.replace('/', path.sep)
            self.video_source = FileVideoSource(video_path)
            pass
        elif SOURCE_TYPE is VideoSourceType.PI_CAMERA:
            self.video_source = PiCameraVideoSource(show_preview=True)
        elif SOURCE_TYPE is VideoSourceType.VIDEO_URL:
            # TODO
            pass
        elif SOURCE_TYPE is VideoSourceType.IP_CAMERA:
            # TODO
            pass
        elif SOURCE_TYPE is VideoSourceType.CAMERA:
            self.video_source = CameraVideoSource(show_preview=True, device_index_name=0)
        elif SOURCE_TYPE is VideoSourceType.YOUTUBE:
            # TODO
            pass
        else:
            log.error("Video source type is not supported : %s ", SOURCE_TYPE.value)
            exit(101)

    def _start(self):
        if self.video_source is None:
            log.error("video source is not set!")
            exit(102)

        # TODO - çalıştırma sırasına göre sonuçlar bir sonraki runnera aktarılabilir
        # TODO - runner dependency ile kimin çıktısı kimn giridisi olacak şekilnde de olabilir
        for i, frame in self.video_source.get_frames():
            if i % 100 == 0:
                log.debug("frame count %s ", i)

            # res_person = None
            # if self.PRE_FIND_PERSON:
            #     res_person = FramePreProcessors.find_persons(frame)
            # if self.PRE_FIND_FACES:
            #     res_faces = FramePreProcessors.find_faces(frame)

            for current_connector in self.available_runners:
                try:
                    result = self.available_runners[current_connector].process_frame(frame=frame)
                    log.debug("result : %s", result)

                    # RunnerResultHandler.add_result(result, runner_name=current_connector)

                except Exception as e:
                    log.exception(e)

            # TODO - check runner settings before send the frame to runner

            # print(frame)

        # TODO - set camera_perspective

    def _process_frame(self, frame):
        self.frame_num += 1
        print("Processing frame: ", self.frame_num)

        pass


if __name__ == '__main__':
    NDUCameraService(path.dirname(path.dirname(path.abspath(__file__))) + '/config/tb_gateway.yaml'.replace('/', path.sep))
