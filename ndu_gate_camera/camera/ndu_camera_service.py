from os import path, uname
import logging
import logging.config
import logging.handlers
from yaml import safe_load
from simplejson import load

import cv2
import numpy as np

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
    # "drivermonitor": "DriverMonitorRunner",
    # "socialdistance": "SocialDistanceRunner",
    # "emotionanalysis": "EmotionAnalysisRunner",
}


class NDUCameraService:
    def __init__(self, ndu_gate_config_file=None):
        if ndu_gate_config_file is None:
            ndu_gate_config_file = path.dirname(path.dirname(path.abspath(__file__))) + '/config/ndu_gate.yaml'.replace(
                '/', path.sep)

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

        if self.SOURCE_CONFIG.get("show_preview", None) is None:
            self.SOURCE_CONFIG["show_preview"] = NDUUtility.is_debug_mode()
        self.__show_preview = self.SOURCE_CONFIG.get("show_preview", False)
        self.__write_preview = self.SOURCE_CONFIG.get("write_preview", False)
        if self.__write_preview:
            self.__write_preview_file_name = self.SOURCE_CONFIG.get("write_preview_file_name", "")

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
            self.__result_handler = ResultHandlerSocket(result_hand_conf.get("socket", 60060))
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

        self.default_runners = DEFAULT_RUNNERS
        self.runners_configs = []
        self.runners_configs_by_key = {}
        self.implemented_runners = {}
        self.available_runners = {}

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
        runners_configs_temp = {}
        last_priority = 1000000

        if self.__ndu_gate_config.get("runners"):
            for runner in self.__ndu_gate_config['runners']:
                log.debug("runner config : %s", runner)
                try:
                    runner_type = runner.get("type", None)
                    if runner_type is None:
                        log.warning("type not found for %s", runner)
                        continue

                    class_name = self.default_runners.get(runner_type, runner.get("class", None))
                    if class_name is None:
                        log.warning("class name not found for %s", runner)
                        continue

                    runner_class = NDUUtility.check_and_import(runner_type, class_name,
                                                               package_uuids=runner.get("uuids", None))
                    if runner_class is None:
                        log.warning("class name implementation not found for %s - %s", runner_type, class_name)
                        continue

                    runner_key = self.__get_runner_key(runner_type, class_name)
                    self.implemented_runners[runner_key] = runner_class

                    configuration_name = runner['configuration']
                    config_file = self._ndu_gate_config_dir + configuration_name

                    runner_conf = {}
                    if path.isfile(config_file):
                        with open(config_file, 'r', encoding="UTF-8") as conf_file:
                            runner_conf = load(conf_file)
                            runner_conf["name"] = runner["name"]
                    else:
                        log.error("config file is not found %s", config_file)
                        runner_conf = {"name": runner["name"]}

                    runner_unique_key = self.__get_runner_configuration_key(runner_type, class_name, configuration_name)

                    runner_priority = runner.get("priority", None)
                    if runner_priority is None:
                        runner_priority = last_priority
                        last_priority = last_priority + 100

                    runners_configs_temp[runner_unique_key] = {
                        "name": runner["name"],
                        "type": runner_type,
                        "class": runner_class,
                        "configuration_name": configuration_name,
                        "config": runner_conf,
                        "priority": runner_priority,
                        "runner_key": runner_key,
                        "runner_unique_key": runner_unique_key
                    }

                except Exception as e:
                    log.error("Error on loading runner config")
                    log.exception(e)

            runner_arr = []
            # add all configs to array
            for key in runners_configs_temp:
                runner_arr.append(runners_configs_temp[key])

            self.runners_configs_by_key = runners_configs_temp
            self.runners_configs = sorted(runner_arr, key=lambda x: x["priority"], reverse=False)
        else:
            log.error("Runners - not found! Check your configuration!")
        dummy = True

    def _connect_with_runners(self):
        """
        runners_configs içindeki configleri kullanarak sırayla yüklenen runner sınıflarının instance'larını oluşturur
        oluşturulan bu nesneleri available_runners içerisine ekler.
        """
        for runner_config in self.runners_configs:
            runner = None
            try:
                runner_key = runner_config["runner_key"]
                runner_type = runner_config["type"]
                runner_unique_key = runner_config["runner_unique_key"]

                if runner_config["config"] is None:
                    log.warning("Config not found for %s", runner_key)
                    continue

                if self.implemented_runners[runner_key] is None:
                    log.error("Implemented runner not found for %s", runner_key)
                else:
                    runner = self.implemented_runners[runner_key](runner_config["config"], runner_type)
                    # runner.setName(runner_config["name"])
                    settings = runner.get_settings()
                    if settings is not None:
                        if settings.get("person", False):
                            self.PRE_FIND_PERSON = True
                        if settings.get("face", False):
                            self.PRE_FIND_FACES = True
                    self.available_runners[runner_unique_key] = runner

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
                self.SOURCE_CONFIG["test_data_path"] = path.dirname(
                    path.dirname(path.abspath(__file__))) + '/data/'.replace('/', path.sep)
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
                self.video_source = CameraVideoSource(self.SOURCE_CONFIG)
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
                "face_list": face_list,
                "results": {}
            }

            # TODO - check runner settings before send the frame to runner

            results = []
            for runner_unique_key in self.available_runners:
                try:
                    runner_conf = self.runners_configs_by_key[runner_unique_key]
                    result = self.available_runners[runner_unique_key].process_frame(frame, extra_data=extra_data)
                    extra_data["results"][runner_unique_key] = result
                    log.debug("result : %s", result)

                    if result is not None:
                        self.__result_handler.save_result(result, runner_name=runner_conf["name"])
                        results.append(result)
                except Exception as e:
                    log.exception(e)

            if self.__show_preview:
                preview = self._get_preview(frame, results)
                cv2.imshow("ndu_gate_camera preview", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if self.__write_preview:
                    self._write_frame(preview)
            elif self.__write_preview:
                self._write_frame(frame)

        if self.__write_preview and self.__out is not None:
            self.__out.release()

        # TODO - set camera_perspective
        log.info("Video source is finished")

    def _write_frame(self, frame):
        if not hasattr(self, "__out"):
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.__out = cv2.VideoWriter(self.__write_preview_file_name, fourcc, 20.0, (1280, 720))
        self.__out.write(frame)

    def _get_preview(self, image, results):
        def put_text(img, text, center, color=[255, 255, 255], font_scale=0.5):
            cv2.putText(img=img, text=text, org=(center[0] + 50, center[1]),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=[0, 0, 0], lineType=cv2.LINE_AA,
                        thickness=2)
            cv2.putText(img=img, text=text, org=(center[0] + 50, center[1]),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale, color=color,
                        lineType=cv2.LINE_AA, thickness=1)

        def draw_rect(image, c1, c2):
            cv2.rectangle(image, c1, c2, 0, 2)
            cv2.rectangle(image, c1, c2, 0, 2.2)

        if results is not None:
            line_height = 20
            rect = "rect"
            score = "score"
            class_name = "class_name"
            _texts = "_texts"
            for result in results:
                item = {rect: None, score: None, class_name: None, _texts: []}
                for res in result:
                    for key in res:
                        if not key in item:
                            item[_texts].append(key + ": " + str(res[key]))
                        else:
                            item[key] = res[key]
                if item[class_name] is not None:
                    text = str(item[class_name])
                    if item[score] is not None:
                        text = f"{text} - %{item[score] * 100:.2f}"
                    item[_texts].insert(0, text)

                if item[rect] is not None:
                    coor = np.array(item[rect][:4], dtype=np.int32)
                    c1, c2 = [coor[1], coor[0]], [coor[3], coor[2]]
                    draw_rect(image, c1, c2)
                    for text in item[_texts]:
                        c1[1] = c1[1] + line_height
                        put_text(image, text, c1)
                else:
                    c1 = [line_height, line_height]
                    for text in item[_texts]:
                        c1[1] = c1[1] + line_height
                        put_text(image, text, c1)
            ####koray sil
            # bbox = out_boxes[i]
            #
            # if not bbox is None:
            #     coor = np.array(bbox[:4], dtype=np.int32)
            #     c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            #
            #     cv2.rectangle(image, c1, c2, 0, 2)
            #     score = out_scores[i]
            #     # cv2.putText(image, f'%{score*100:.2f}' + ' ' + out_classes[i], (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
            #     #             0.5, (255, 255, 255), 2 // 2, lineType=cv2.LINE_AA)
            #     put_text(image, f'%{score * 100:.2f}' + ' ' + out_classes[i], (c1[0], c1[1] - 2))
            #     drawn = True
            # else:
            #     c1 = (0, h)
            #     h = h + 30
            #     score = out_scores[i]
            #     # cv2.putText(image, f'%{score*100:.2f}' + ' ' + out_classes[i], (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
            #     #             0.8, (255, 255, 255), 2 // 2, lineType=cv2.LINE_AA)
            #     put_text(image, f'%{score * 100:.2f}' + ' ' + out_classes[i], (c1[0], c1[1] - 2))
            #     drawn = True

            # global last_drawn
            # if not drawn and last_drawn is not None:
            #     return draw(*last_drawn, image)
            # elif drawn:
            #     last_drawn = out_boxes, out_scores, out_classes, len
        return image

    def __get_runner_configuration_key(self, runner_type, class_name, configuration):
        if configuration is None:
            configuration = runner_type + ".json"
        return runner_type + "_" + class_name + "_" + configuration

    def __get_runner_key(self, type, class_name):
        return type + "_" + class_name


if __name__ == '__main__':
    NDUCameraService(DEFAULT_NDU_GATE_CONF.replace('/', path.sep))
