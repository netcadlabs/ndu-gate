from os import path, uname
import logging
import logging.config
import logging.handlers

from yaml import safe_load
from simplejson import load

from ndu_gate_camera.api.video_source import VideoSourceType
from ndu_gate_camera.camera.ndu_logger import NDULoggerHandler
from ndu_gate_camera.camera.result_handlers.result_handler_file import ResultHandlerFile
from ndu_gate_camera.camera.result_handlers.result_handler_socket import ResultHandlerSocket
from ndu_gate_camera.camera.video_sources.camera_video_source import CameraVideoSource
from ndu_gate_camera.camera.video_sources.file_video_source import FileVideoSource
from ndu_gate_camera.camera.video_sources.ip_camera_video_source import IPCameraVideoSource
from ndu_gate_camera.camera.video_sources.pi_camera_video_source import PiCameraVideoSource
from ndu_gate_camera.camera.video_sources.youtube_video_source import YoutubeVideoSource
from ndu_gate_camera.detectors.face_detector import FaceDetector
from ndu_gate_camera.detectors.person_detector import PersonDetector
from ndu_gate_camera.utility.ndu_utility import NDUUtility

from ndu_gate_camera.utility.constants import DEFAULT_NDU_GATE_CONF

name = uname()

DEFAULT_RUNNERS = {
    "drivermonitor": "DriverMonitorRunner",
    "socialdistance": "SocialDistanceRunner",
    "emotionanalysis": "EmotionAnalysisRunner",
}


class NDUCameraService:
    def __init__(self, ndu_gate_config_file=None):
        if ndu_gate_config_file is None:
            ndu_gate_config_file = path.dirname(path.dirname(path.abspath(__file__))) + '/config/ndu_gate.yaml'.replace('/', path.sep)

        with open(ndu_gate_config_file) as general_config:
            self.__ndu_gate_config = safe_load(general_config)
        self._ndu_gate_config_dir = path.dirname(path.abspath(ndu_gate_config_file)) + path.sep

        self.SOURCE_TYPE = VideoSourceType.CAMERA
        self.SOURCE_CONFIG = None
        if self.__ndu_gate_config.get("video_source"):
            self.SOURCE_CONFIG = self.__ndu_gate_config.get("video_source")
            type_str = self.SOURCE_CONFIG.get("type", "CAMERA")
            if VideoSourceType[type_str]:
                self.SOURCE_TYPE = VideoSourceType[type_str]
            else:
                self.SOURCE_TYPE = VideoSourceType.CAMERA

        self.SOURCE_CONFIG["show_preview"] = NDUUtility.is_debug_mode()

        logging_error = None
        logging_config_file = self._ndu_gate_config_dir + "logs.conf"
        try:
            import platform
            if platform.system() == "Darwin":
                self._ndu_gate_config_dir + "logs_macosx.conf"
            logging.config.fileConfig(logging_config_file, disable_existing_loggers=False)
        except Exception as e:
            print(e)
            logging_error = e
            NDULoggerHandler.set_default_handler()

        global log
        log = logging.getLogger('service')
        log.info("NDUCameraService starting...")
        log.info("NDU-Gate config file: %s", ndu_gate_config_file)
        log.info("NDU-Gate logging config file: %s", logging_config_file)
        log.info("NDU-Gate logging service level: %s", log.level)

        result_hand_conf = self.__ndu_gate_config.get("result_handler", None)
        default_result_file_path = "/var/lib/thingsboard_gateway/extensions/camera/"
        if result_hand_conf is None:
            result_hand_conf = {
                "type": "FILE",
                "file_path": default_result_file_path
            }

        if str(result_hand_conf.get("type", "FILE")) == str("SOCKET"):
            self.__result_handler = ResultHandlerSocket(socket_port=result_hand_conf.get("port", 60060),
                                                        socket_host=result_hand_conf.get("host", '127.0.0.1'))
        else:
            self.__result_handler = ResultHandlerFile(result_hand_conf.get("file_path", default_result_file_path))

        self.PRE_FIND_PERSON = False
        self.PRE_FIND_FACES = False
        self.__personDetector = None
        self.__faceDetector = None
        try:
            self.__personDetector = PersonDetector()
        except Exception as e:
            log.error("Can not create person detector")
            log.error(e, stack_info=True)
            print(e)
        try:
            self.__faceDetector = FaceDetector()
        except Exception as e:
            log.error("Can not create face detector")
            log.error(e, stack_info=True)

        self._default_runners = DEFAULT_RUNNERS
        self._implemented_runners = {}
        self.available_runners = {}
        self.frame_num = 0

        self._load_runners()
        self._connect_with_runners()

        self.video_source = None
        self._set_video_source()
        self._start()
        log.info("NDUCameraService exiting...")

    def _load_runners(self):
        """
        config dosyasında belirtilen NDUCameraRunner imaplementasyonlarını bulur ve _implemented_runners içerisine ekler
        Aynı şekilde herbir runner için config dosyalarını bulur ve connectors_configs içerisine ekler
        """
        self.runners_configs = {}
        if self.__ndu_gate_config.get("runners"):
            for runner in self.__ndu_gate_config['runners']:
                try:
                    # if connector.get("runner", None) is None:
                    #     continue
                    runner_class = NDUUtility.check_and_import(runner["type"], self._default_runners.get(runner["type"], runner.get("class")))
                    self._implemented_runners[runner["type"]] = runner_class
                    config_file = self._ndu_gate_config_dir + runner['configuration']

                    if not self.runners_configs.get(runner['type']):
                        self.runners_configs[runner['type']] = []

                    if path.isfile(config_file):
                        with open(self._ndu_gate_config_dir + runner['configuration'], 'r', encoding="UTF-8") as conf_file:
                            runner_conf = load(conf_file)
                            runner_conf["name"] = runner["name"]
                            self.runners_configs[runner['type']].append({"name": runner["name"], "config": {runner['configuration']: runner_conf}})
                    else:
                        log.error("config file is not found %s", config_file)
                        runner_conf = {"name": runner["name"]}
                        self.runners_configs[runner['type']].append({"name": runner["name"], "config": {runner['configuration']: runner_conf}})
                except Exception as e:
                    log.error("Error on loading runner config")
                    log.exception(e)
        else:
            log.error("Runners - not found! Check your configuration!")

    def _connect_with_runners(self):
        """
        runners_configs içindeki configleri kullanarak sırayla yüklenen runner sınıflarının instance'larını oluşturur
        oluşturulan bu nesneleri available_runners içerisine ekler.
        """
        for runner_type in self.runners_configs:
            for runner_config in self.runners_configs[runner_type]:
                for config in runner_config["config"]:
                    runner = None
                    try:
                        if runner_config["config"][config] is not None:
                            if self._implemented_runners[runner_type] is None:
                                log.error("implemented runner not found for %s", runner_type)
                            else:
                                runner = self._implemented_runners[runner_type](self, runner_config["config"][config], runner_type)
                                runner.setName(runner_config["name"])
                                settings = runner.get_settings()
                                if settings is not None:
                                    if settings.get("person", False):
                                        self.PRE_FIND_PERSON = True
                                    if settings.get("face", False):
                                        self.PRE_FIND_FACES = True
                                self.available_runners[runner.get_name()] = runner
                        else:
                            log.warning("Config not found for %s", runner_type)
                    except Exception as e:
                        log.exception(e)
                        if runner is not None and NDUUtility.has_method(runner, 'close'):
                            runner.close()

    def _set_video_source(self):
        """
        SOURCE_TYPE değerine göre video_source değişkenini oluşturur.
        """
        try:
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
                self.video_source = CameraVideoSource()
            elif self.SOURCE_TYPE is VideoSourceType.YOUTUBE:
                self.video_source = YoutubeVideoSource(self.SOURCE_CONFIG)
            else:
                log.error("Video source type is not supported : %s ", self.SOURCE_TYPE.value)
                exit(101)
        except Exception as e:
            log.error("Error during setting up video source")
            log.error(e)

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

            for current_runner in self.available_runners:
                try:
                    result = self.available_runners[current_runner].process_frame(frame, extra_data=extra_data)
                    log.debug("result : %s", result)

                    if result is not None:
                        self.__result_handler.save_result(result, runner_name=current_runner)

                except Exception as e:
                    log.exception(e)

            # print(frame)

        # TODO - set camera_perspective
        log.info("Video source is finished")

    def _process_frame(self, frame):
        self.frame_num += 1
        print("Processing frame: ", self.frame_num)

        pass


if __name__ == '__main__':
    NDUCameraService(DEFAULT_NDU_GATE_CONF.replace('/', path.sep))
