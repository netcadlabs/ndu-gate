import os
import random
import time
from threading import Thread
from os import path
from logging import getLogger

from simplejson import load
import cv2
import numpy as np

from ndu_gate_camera.api.video_source import VideoSourceType
from ndu_gate_camera.camera.video_sources.camera_video_source import CameraVideoSource
from ndu_gate_camera.camera.video_sources.file_video_source import FileVideoSource
from ndu_gate_camera.camera.video_sources.ip_camera_video_source import IPCameraVideoSource
from ndu_gate_camera.camera.video_sources.pi_camera_video_source import PiCameraVideoSource
from ndu_gate_camera.camera.video_sources.youtube_video_source import YoutubeVideoSource
from ndu_gate_camera.camera.video_sources.image_video_source import ImageVideoSource
from ndu_gate_camera.utility import constants, image_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility

DEFAULT_RUNNERS = {
    # "drivermonitor": "DriverMonitorRunner",
    # "socialdistance": "SocialDistanceRunner",
    # "emotionanalysis": "EmotionAnalysisRunner",
}

log = getLogger("service")


class NDUCameraService(Thread):
    def __init__(self, instance={}, config_dir="", handler=None):
        super().__init__()
        self.__result_handler = handler
        self._ndu_gate_config_dir = config_dir
        self.RUNNERS = instance.get("runners", [])
        self.SOURCE_TYPE = VideoSourceType.CAMERA
        self.SOURCE_CONFIG = None

        if instance.get("source"):
            self.SOURCE_CONFIG = instance.get("source")
            type_str = self.SOURCE_CONFIG.get("type", "PI_CAMERA")
            if VideoSourceType[type_str]:
                self.SOURCE_TYPE = VideoSourceType[type_str]
            else:
                self.SOURCE_TYPE = VideoSourceType.PI_CAMERA

        self.__frame_send_interval = self.SOURCE_CONFIG.get("frame_send_interval", 1000)
        self.__preview_show = self.SOURCE_CONFIG.get("preview_show", False)
        self.__preview_show_debug_texts = self.SOURCE_CONFIG.get("preview_show_debug_texts", True)
        self.__preview_show_runner_info = self.SOURCE_CONFIG.get("preview_show_runner_info", True)
        self.__preview_show_score = self.SOURCE_CONFIG.get("preview_show_score", True)
        self.__preview_show_rect_name = self.SOURCE_CONFIG.get("preview_show_rect_name", True)
        self.__preview_show_rect_filter = self.SOURCE_CONFIG.get("preview_show_rect_filter", None)

        self.__preview_write = self.SOURCE_CONFIG.get("preview_write", False)
        self.__preview_last_data = []
        self.__preview_last_data_show_counts = []
        self.__preview_write_file_name = self.SOURCE_CONFIG.get("preview_write_file_name", "")

        self.__max_frame_dim = self.SOURCE_CONFIG.get("max_frame_dim", None)
        self.__min_frame_dim = self.SOURCE_CONFIG.get("min_frame_dim", None)
        self.__skip_frame = self.SOURCE_CONFIG.get("skip_frame", 0)
        self.__color_toggle = None

        self.frame_sent = self.SOURCE_CONFIG.get("frame_sent", False)


        self.default_runners = DEFAULT_RUNNERS
        self.runners_configs = []
        self.runners_configs_by_key = {}
        self.implemented_runners = {}
        self.available_runners = {}

        self._load_runners()
        self._connect_with_runners()

        self.video_source = None
        self._set_video_source()
        # self._start()

    # def run(self):
    #     self.start()

    def _load_runners(self):
        """
        config dosyasında belirtilen NDUCameraRunner imaplementasyonlarını bulur ve _implemented_runners içerisine ekler
        Aynı şekilde herbir runner için config dosyalarını bulur ve connectors_configs içerisine ekler
        """
        runners_configs_temp = {}
        last_priority = 1000000

        if self.RUNNERS:
            for runner in self.RUNNERS:
                log.debug("runner config : %s", runner)
                try:
                    if runner.get("status", 1) == 0:  # runner status default value is 1>>>
                        log.debug("runner is not active %s", runner)
                        continue

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

                    if path.isfile(config_file):
                        with open(config_file, 'r', encoding="UTF-8") as conf_file:
                            runner_conf = load(conf_file)
                            runner_conf["name"] = runner["name"]
                    else:
                        log.error("config file is not found %s", config_file)
                        runner_conf = {"name": runner["name"]}

                    runner_custom_conf = {}
                    custom_config_file = self._ndu_gate_config_dir + runner_type + "_custom.json"
                    if path.isfile(custom_config_file):
                        with open(custom_config_file, 'r', encoding="UTF-8") as conf_file:
                            runner_custom_conf = load(conf_file)
                    else:
                        log.error("custom config file is not found %s", custom_config_file)

                    runner_conf["custom_config"] = runner_custom_conf

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
            elif self.SOURCE_TYPE is VideoSourceType.IMAGE_FILE:
                self.video_source = ImageVideoSource(self.SOURCE_CONFIG)
            else:
                log.error("Video source type is not supported : %s ", self.SOURCE_TYPE.value)
                # exit(101)
        except Exception as e:
            log.error("Error during setting up video source")
            log.error(e)

    def run(self):
        if self.video_source is None:
            log.error("video source is not set!")
            return
            # exit(102)
        start_total = None
        skip = 0
        pause = False
        # TODO - çalıştırma sırasına göre sonuçlar bir sonraki runnera aktarılabilir
        # TODO - runner dependency ile kimin çıktısı kimn giridisi olacak şeklinde de olabilir

        winname = None
        if self.__preview_show:
            winname = "ndu_gate_camera preview"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 40, 30)

        try:
            device = self.SOURCE_CONFIG.get("device", None)
            for i, frame in self.video_source.get_frames():
                if i % 500 == 0:
                    log.debug("frame count %s ", i)
                    print("Source Device : {} - frame {}".format(device, i))
                if self.__skip_frame > 1 and i % self.__skip_frame != 0:
                    continue

                if i % self.__frame_send_interval == 0:
                    try:
                        camera_capture_base64 = image_helper.frame2base64(frame)
                        log.info("CAMERA_CAPTURE size : %s", len(camera_capture_base64))
                        print("CAMERA_CAPTURE size : {}".format(len(camera_capture_base64)))
                        if self.frame_sent:
                            self.__result_handler.save_result([{"data": {"CAMERA_CAPTURE": camera_capture_base64}}], device=device, data_type='attribute')
                        self.__result_handler.save_result([{"data": {"FRAME_COUNT": i}}], device=device, data_type='attribute')
                        self.frame_sent = True
                    except Exception as e:
                        log.exception(e)
                        log.error("can not create CAMERA_CAPTURE")

                if self.__max_frame_dim is not None:
                    frame = image_helper.resize_if_larger(frame, self.__max_frame_dim)
                if self.__min_frame_dim is not None:
                    frame = image_helper.resize_if_smaller(frame, self.__min_frame_dim)

                results = []
                if skip <= 0:
                    if self.__preview_show:
                        start_total = time.time()
                    extra_data = {
                        constants.EXTRA_DATA_KEY_RESULTS: {}
                    }

                    # TODO - check runner settings before send the frame to runner
                    for runner_unique_key in self.available_runners:
                        try:
                            runner_conf = self.runners_configs_by_key[runner_unique_key]
                            start = time.time()
                            result = self.available_runners[runner_unique_key].process_frame(frame, extra_data=extra_data)
                            elapsed = time.time() - start
                            if self.__preview_show and result is not None:
                                result.append({"elapsed_time": '{}: {:.4f}sn fps:{:.0f}'.format(runner_conf["type"], elapsed, 1.0 / max(elapsed, 0.001))})
                            extra_data[constants.EXTRA_DATA_KEY_RESULTS][runner_unique_key] = result
                            log.debug("result : %s", result)

                            if result is not None:
                                results.append(result)
                                self.__result_handler.save_result(result, device=device, runner_name=runner_conf["name"])

                        except Exception as e:
                            log.exception(e)

                if self.__preview_show:
                    if skip > 0:
                        skip = skip - 1
                        preview = frame
                    else:
                        total_elapsed_time = time.time() - start_total
                        results.append([{"total_elapsed_time": '{:.0f}msec fps:{:.0f}'.format(total_elapsed_time * 1000, (1.0 / max(total_elapsed_time, 0.001)))}])
                        preview = self._get_preview(frame, results)

                    cv2.imshow(winname, preview)
                    # while True:
                    #     k = cv2.waitKey(100) & 0xFF
                    #     print(k)
                    exit_requested = False
                    while True:
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("q"):
                            exit_requested = True
                            break
                        elif k == ord("s"):
                            skip = 10
                        elif k == 32:  # space key
                            pause = not pause
                        if not pause:
                            break
                    if exit_requested:
                        break
                    if self.__preview_write:
                        self._write_frame(preview)
                elif self.__preview_write:
                    self._write_frame(frame)

            if self.__preview_write and self.__out is not None:
                self.__out.release()
                import ffmpeg  # pip3 install ffmpeg-python & brew install ffmpeg
                try:
                    # daha az sıkışmış, quicktime çalabiliyor
                    fn = self.__preview_write_file_name
                    ffmpeg.input(fn).output(fn + '.mp4').run(capture_stdout=True, capture_stderr=True)

                    # # süper sıkışmış ama quicktime çalamıyor. Benim denemelerimde yarım yamalak kaydedebildi!
                    ffmpeg.input(fn) \
                        .output(fn + '2.mp4', vcodec='libx265', crf=24, t=5) \
                        .run(capture_stdout=True, capture_stderr=True)
                    # # os.remove(fn)
                except ffmpeg.Error as e:
                    print('stdout:', e.stdout.decode('utf8'))
                    print('stderr:', e.stderr.decode('utf8'))
                    raise e

            # TODO - set camera_perspective
        except Exception as e:
            log.exception(e)

        log.info("Video source is finished")

    def _write_frame(self, frame):
        def get_free_file_name(fn):
            if os.path.exists(fn):
                filename, file_extension = os.path.splitext(fn)
                i = 1
                while os.path.exists(fn):
                    # sil fn = f"{filename}{i}{file_extension}"
                    fn = "{}{}{}".format(filename, i, file_extension)
                    i += 1
            return fn

        try:
            self.__out.write(frame)
        except:  # daha iyi bir yolunu bulursanız, bana da gösterin -> korhun :)
            shape = frame.shape[1], frame.shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # fourcc = cv2.VideoWriter_fourcc(*'AVC1')
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.__preview_write_file_name = get_free_file_name(self.__preview_write_file_name)
            self.__out = cv2.VideoWriter(self.__preview_write_file_name, fourcc, 24.0, shape)
            self.__out.write(frame)

    def _new_color(self):
        if self.__color_toggle is None:
            self.__color_toggle = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
            return self.__color_toggle
        else:
            c = self.__color_toggle
            self.__color_toggle = None
            return [max(50, 255 - c[0]), max(50, 255 - c[1]), max(50, 255 - c[2])]

    def _get_preview(self, image, results):
        def draw_rect(obj, img, c1_, c2_, class_preview_key_, color=None):
            if color is None:
                color = [255, 255, 255]
                if class_preview_key_ is not None:
                    if not hasattr(obj, "__colors"):
                        setattr(obj, "__colors", {})
                    dic = getattr(obj, "__colors")
                    if class_preview_key_ in dic:
                        color = dic[class_preview_key_]
                    else:
                        color = dic[class_preview_key_] = self._new_color()

            cv2.rectangle(img, (c1_[0], c1_[1]), (c2_[0], c2_[1]), color=[0, 0, 0], thickness=3)
            cv2.rectangle(img, (c1_[0], c1_[1]), (c2_[0], c2_[1]), color=color, thickness=2)

        show_debug_texts = self.__preview_show_debug_texts
        show_runner_info = self.__preview_show_runner_info
        show_score = self.__preview_show_score
        rect_filter = self.__preview_show_rect_filter
        show_rect_name = self.__preview_show_rect_name

        h, w, *_ = image.shape
        line_height = 20
        font_scale = 0.4
        current_line = [20, 0]
        current_line_bottom = [20, h]
        data_added = []
        if results is not None:
            for result in results:
                debug_texts = []
                text_type = ""
                has_data = False
                for item in result:
                    if show_runner_info:
                        total_elapsed_time = item.get("total_elapsed_time", None)
                        if total_elapsed_time is not None:
                            image_helper.put_text(image, total_elapsed_time, [w - 150, h - 30], color=[200, 200, 128], font_scale=0.4)
                            continue

                    values = {}
                    elapsed_time = values["elapsed_time"] = item.get("elapsed_time", None)
                    class_name = values[constants.RESULT_KEY_CLASS_NAME] = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                    class_preview_key = values[constants.RESULT_KEY_PREVIEW_KEY] = item.get(constants.RESULT_KEY_PREVIEW_KEY, class_name)
                    score = values[constants.RESULT_KEY_SCORE] = item.get(constants.RESULT_KEY_SCORE, None)
                    rect = values[constants.RESULT_KEY_RECT] = item.get(constants.RESULT_KEY_RECT, None)
                    data = values[constants.RESULT_KEY_DATA] = item.get(constants.RESULT_KEY_DATA, None)
                    debug_text = values[constants.RESULT_KEY_DEBUG] = item.get(constants.RESULT_KEY_DEBUG, None)

                    if debug_text is not None:
                        debug_texts.append(debug_text)

                    text = ""
                    if class_name is not None:
                        text = NDUUtility.debug_conv_turkish(class_name) + " "
                        if show_score and score is not None:
                            text = "{} - %{:.2f} ".format(text, score * 100)
                    elif show_score and score is not None:
                        text = "%{:.2f} ".format(score * 100)

                    if data is not None:
                        add_txt = " data: " + str(data)
                        data_added.append(add_txt)
                        text = text + add_txt
                        has_data = True
                    if rect is not None and (rect_filter is None or NDUUtility.wildcard(class_name, rect_filter)):
                        c = np.array(rect[:4], dtype=np.int32)
                        c1, c2 = [c[1], c[0]], (c[3], c[2])
                        draw_rect(self, image, c1, c2, class_preview_key, item.get(constants.RESULT_KEY_RECT_COLOR, None))
                        if show_rect_name and len(text) > 0:
                            c1[1] = c1[1] + line_height
                            image_helper.put_text(image, text, c1)
                            text = ""
                    if elapsed_time is not None:
                        text_type = elapsed_time + " " + text_type
                    if len(text) > 0:
                        text_type = text_type + text + " "
                if show_runner_info and len(text_type) > 0:
                    current_line[1] += line_height
                    if show_runner_info:
                        if not has_data:
                            image_helper.put_text(image, text_type, current_line, font_scale=font_scale)
                        else:
                            image_helper.put_text(image, text_type, current_line, color=[0, 0, 255], font_scale=font_scale)
                if show_debug_texts and len(debug_texts) > 0:
                    for debug_text in debug_texts:
                        current_line_bottom[1] -= line_height * 2
                        image_helper.put_text(image, debug_text, current_line_bottom, color=[255, 250, 99], font_scale=font_scale * 2.75, thickness=2, back_color=[0, 0, 0])

        if show_runner_info:
            if len(data_added) > 0:
                show_last_data_frame_count = 15
                for data in data_added:
                    if data not in self.__preview_last_data:
                        self.__preview_last_data.append(data)
                        self.__preview_last_data_show_counts.append(show_last_data_frame_count)
                    else:
                        self.__preview_last_data_show_counts[self.__preview_last_data.index(data)] = show_last_data_frame_count
            for i in reversed(range(len(self.__preview_last_data))):
                self.__preview_last_data_show_counts[i] -= 1
                if self.__preview_last_data_show_counts[i] <= 0:
                    del self.__preview_last_data_show_counts[i]
                    del self.__preview_last_data[i]
            for last_data in self.__preview_last_data:
                current_line[1] += line_height
                image_helper.put_text(image, last_data, current_line, color=[0, 255, 255])

        return image

    def __get_runner_configuration_key(self, runner_type, class_name, configuration):
        if configuration is None:
            configuration = runner_type + ".json"
        return runner_type + "_" + class_name + "_" + configuration

    def __get_runner_key(self, type, class_name):
        return type + "_" + class_name


if __name__ == '__main__':
    NDUCameraService(constants.DEFAULT_NDU_GATE_CONF.replace('/', path.sep))
