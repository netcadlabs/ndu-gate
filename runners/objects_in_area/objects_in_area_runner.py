from threading import Thread

import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.geometry_helper import geometry_helper
from ndu_gate_camera.utility.image_helper import image_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class objects_in_area_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.connector_type = connector_type
        self.__classes = config.get("classes", None)
        self.__areas = []
        self.__frame_num = 0
        self.__last_data = {}

        self.__debug = False
        self.__debug_color_normal = (255, 255, 255)
        self.__debug_color_alert = (0, 0, 255)
        self.__debug_thickness_normal = 1
        self.__debug_thickness_alert = 4

    def get_name(self):
        return "objects_in_area_runner"

    def get_settings(self):
        settings = {}
        return settings

    @staticmethod
    def get_center_int(pnts):
        x = 0
        y = 0
        for pnt in pnts:
            x += pnt[0]
            y += pnt[1]
        length = float(len(pnts))
        return [int(x / length), int(y / length)]

    # @staticmethod
    # def get_box_center(box):
    #     (x, y) = (int(box[0]), int(box[1]))
    #     (w, h) = (int(box[2]), int(box[3]))
    #     return (x + w / 2, y + h / 2)

    @staticmethod
    def _get_rect_ref_point(rect):
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        # return y2, int(x1 + (x2 - x1) * 0.5)  # bottom center
        return int(y1 + (y2 - y1) * 0.5), x2  # bottom center

    area_prefix = "Area"

    def debug_area_name(self, area_name):
        return area_name.replace(self.area_prefix, "ALAN")

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        res = []
        if self.__classes is None:
            return res

        self.__frame_num += 1
        if len(self.__areas) == 0 and self.__frame_num == 1:
            # areas = image_helper.select_areas(frame, self.get_name(), color=self.__debug_color_false, opacity=0.3, thickness=self.__debug_thickness_false, max_count=None, next_area_key="n", finish_key="s")
            areas = [[(503, 167), (403, 225), (565, 270), (639, 203)],
                     [(334, 484), (307, 483), (277, 481), (229, 491), (210, 518), (196, 546), (213, 580), (230, 605), (280, 626), (309, 627), (350, 615), (374, 598), (405, 567), (411, 532), (384, 501)]]  # vid_short.mp4
            # areas = [[(144, 361), (645, 368), (983, 824), (897, 892), (5, 894), (19, 427)]] # yaya4.mp4
            # areas = [[(164, 191), (310, 185), (355, 213), (451, 244), (512, 263), (354, 299), (244, 247), (205, 222), (190, 216)]] # araba2.mp4
            # areas = [[(654, 565), (665, 576), (666, 583), (660, 589), (653, 592), (642, 599), (630, 603), (621, 607), (939, 558), (864, 539), (767, 550)]] # meydan2.mp4
            area_counter = 0
            for area in areas:
                area_counter += 1
                self.__areas.append({"name": f"{self.area_prefix}{area_counter}", "area": area})

        if self.__debug:
            for item in self.__areas:
                area_name = self.debug_area_name(item.get("name"))
                area = item.get("area")
                pnts = np.array(area, np.int32)
                image_helper.fill_polyline_transparent(frame, [pnts], color=self.__debug_color_normal, opacity=0.3, thickness=self.__debug_thickness_normal)
                center = self.get_center_int(area)
                image_helper.put_text(frame, area_name, center, color=self.__debug_color_normal, font_scale=0.75)

        results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
        active_areas = {}
        if results is not None:
            counts = {}
            for runner_name, result in results.items():
                for item in result:
                    class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                    if class_name in self.__classes:
                        rect = item.get(constants.RESULT_KEY_RECT, None)
                        if rect is not None:
                            for area_item in self.__areas:
                                area = area_item.get("area")
                                ref_pnt = self._get_rect_ref_point(rect)
                                # if self.__debug:
                                #     cv2.circle(frame, ref_pnt, 4, color=(0, 0, 255), thickness=4)
                                if geometry_helper.is_inside_polygon(area, ref_pnt):
                                    area_name = area_item.get("name")
                                    if area_name not in counts:
                                        counts[area_name] = {class_name: 1}
                                    elif class_name not in counts[area_name]:
                                        counts[area_name][class_name] = 1
                                    else:
                                        counts[area_name][class_name] += 1

                                    if self.__debug:  # and self.__frame_num % 2 == 0:
                                        pnts = np.array(area, np.int32)
                                        image_helper.fill_polyline_transparent(frame, [pnts], color=self.__debug_color_alert, opacity=0.5, thickness=self.__debug_thickness_alert)
                                        center = self.get_center_int(area)
                                        image_helper.put_text(frame, self.debug_area_name(area_name), center, color=self.__debug_color_alert, font_scale=0.75)

            debug_texts = debug_text = None
            if self.__debug:
                debug_texts = []
            for area_name, item in counts.items():
                active_areas[area_name] = True
                if self.__debug:
                    debug_text = f"{self.debug_area_name(area_name)}: "
                for class_name, value in counts.get(area_name).items():
                    if self.__debug:
                        debug_text += f"{NDUUtility.debug_conv_turkish(class_name)}: {value} "

                    # data_val = class_name + "_" + str(value)
                    # changed = False
                    # if area_name not in self.__last_data or not self.__last_data[area_name] == data_val:
                    #     self.__last_data[area_name] = data_val
                    #     changed = True
                    # if changed:
                    #     telemetry = area_name # + "_" + class_name
                    #     # data = {telemetry: value}
                    #     data = {telemetry: True}
                    #     res.append({constants.RESULT_KEY_DATA: data})

                res.append({constants.RESULT_KEY_DEBUG: debug_text})
                # res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: name, constants.RESULT_KEY_PREVIEW_KEY: preview_key})

        data_txt = ""
        data_items = []
        for item in self.__areas:
            name = item.get("name")
            if name in active_areas:
                data_items.append({name: True})
                data_txt += name + "_1"
            else:
                data_items.append({name: False})
                data_txt += name + "_0"
        ####if True:
        if self.__last_data != data_txt:
            self.__last_data = data_txt
            for data in data_items:
                res.append({constants.RESULT_KEY_DATA: data})

        return res
