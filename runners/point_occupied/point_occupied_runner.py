from threading import Thread

import cv2
import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.geometry_helper import geometry_helper
from ndu_gate_camera.utility.image_helper import image_helper


class point_occupied_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.connector_type = connector_type
        self.__classes = config.get("classes", None)
        self.__points = []
        self.__last_data = {}
        self.__frame_num = 0
        self.__average_len = 10
        self.__average_accept = 1
        self.__average_cur = 10

        self.__debug = True  ####koray
        self.__debug_color_false = (200, 200, 200)
        self.__debug_color_true = (0, 255, 0)
        self.__debug_thickness_false = 4
        self.__debug_thickness_true = 4
        self.__debug_radius_false = 4
        self.__debug_radius_true = 4

    def get_name(self):
        return "point_occupied_runner"

    def get_settings(self):
        settings = {}
        return settings

    # @staticmethod
    # def get_center_int(pnts):
    #     x = 0
    #     y = 0
    #     for pnt in pnts:
    #         x += pnt[0]
    #         y += pnt[1]
    #     length = float(len(pnts))
    #     return [int(x / length), int(y / length)]

    # @staticmethod
    # def get_box_center(box):
    #     (x, y) = (int(box[0]), int(box[1]))
    #     (w, h) = (int(box[2]), int(box[3]))
    #     return (x + w / 2, y + h / 2)

    # @staticmethod
    # def _get_rect_ref_point(rect):
    #     x1 = rect[0]
    #     y1 = rect[1]
    #     x2 = rect[2]
    #     y2 = rect[3]
    #     # return y2, int(x1 + (x2 - x1) * 0.5)  # bottom center
    #     return int(y1 + (y2 - y1) * 0.5), x2  # bottom center
    pnt_prefix = "P"

    def debug_pnt_name(self, pnt_name):
        return pnt_name
        # return pnt_name.replace(self.pnt_prefix, "P")

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        res = []
        if self.__classes is None:
            return res

        self.__frame_num += 1
        if len(self.__points) == 0 and self.__frame_num == 1:
            pnts = image_helper.select_points(frame, self.get_name())
            # pnts = [(617, 462), (591, 276), (281, 324), (349, 518), (470, 378), (467, 210)] #vid_short
            # otopark_trim_8x
            # pnts = [(266, 638), (446, 643), (580, 648), (694, 649), (821, 654), (954, 657), (1066, 661), (1197, 654), (1274, 636), (1025, 482), (981, 480), (892, 468), (814, 467), (723, 451), (626, 467), (551, 449),
            #         (436, 431), (334, 425), (240, 411), (164, 406), (69, 391)]
            # pnts = [(34, 604), (269, 611), (572, 633), (770, 635), (1011, 643), (1216, 623), (48, 375), (160, 372), (286, 373), (400, 392), (519, 415), (629, 433), (714, 433), (807, 440), (911, 446), (971, 452)]

            pnt_counter = 0
            for pnt in pnts:
                pnt_counter += 1
                self.__points.append({"name": f"{self.pnt_prefix}{pnt_counter}", "pnt": pnt, "true_count": 0, "is_true": False})

        # if self.__debug:
        #     for item in self.__points:
        #         pnt_name = self.debug_pnt_name(item.get("name"))
        #         pnt = item.get("pnt")
        #         cv2.circle(frame, pnt, self.__debug_radius_false, self.__debug_color_false, self.__debug_thickness_false)
        #         image_helper.put_text(frame, pnt_name, pnt, color=self.__debug_color_false, font_scale=0.75)

        results = extra_data.get("results", None)
        # active_pnts = {}
        if results is not None:
            # counts = {}
            for runner_name, result in results.items():
                for item in result:
                    class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                    if class_name in self.__classes:
                        rect = item.get(constants.RESULT_KEY_RECT, None)
                        if rect is not None:
                            for pnt_item in self.__points:
                                pnt = pnt_item.get("pnt")
                                pnt_name = pnt_item.get("name")
                                if geometry_helper.is_inside_rect(rect, pnt):
                                    pnt_item["true_count"] += 1
                                    #
                                    # if pnt_name not in counts:
                                    #     counts[pnt_name] = {class_name: 1}
                                    # elif class_name not in counts[pnt_name]:
                                    #     counts[pnt_name][class_name] = 1
                                    # else:
                                    #     counts[pnt_name][class_name] += 1

                                    # if self.__debug:  # and self.__frame_num % 2 == 0:
                                    #     pnt_name1 = self.debug_pnt_name(pnt_name)
                                    #     pnt = pnt_item.get("pnt")
                                    #     cv2.circle(frame, pnt, self.__debug_radius_true, self.__debug_color_true, self.__debug_thickness_true)
                                    #     image_helper.put_text(frame, pnt_name1, pnt, color=self.__debug_color_true, font_scale=0.75)

            # for pnt_name, item in counts.items():
            #     active_pnts[pnt_name] = True
        if self.__average_cur < self.__average_len:
            self.__average_cur += 1
        else:
            self.__average_cur = 0
            data_txt = ""
            data_items = []
            true_count = 0
            for pnt_item in self.__points:
                pnt_name = pnt_item.get("name")
                count = pnt_item["true_count"]
                pnt_item["true_count"] = 0
                if count >= self.__average_accept:
                    true_count += 1
                    pnt_item["is_true"] = True
                    data_items.append({pnt_name: True})
                    data_txt += pnt_name + ":1 "
                else:
                    pnt_item["is_true"] = False
                    data_items.append({pnt_name: False})
                    data_txt += pnt_name + ":0 "

            # for item in self.__points:
            #     pnt_name = item.get("name")
            #     if pnt_name in active_pnts:
            #         data_items.append({pnt_name: True})
            #         data_txt += pnt_name + ":1 "
            #     else:
            #         data_items.append({pnt_name: False})
            #         data_txt += pnt_name + ":0 "

            if self.__last_data != data_txt:
                self.__last_data = data_txt
                for data in data_items:
                    res.append({constants.RESULT_KEY_DATA: data})
                res.append({constants.RESULT_KEY_DATA: {"true_count": true_count}})

        if self.__debug:
            for pnt_item in self.__points:
                pnt_name = self.debug_pnt_name(pnt_item.get("name"))
                pnt = pnt_item.get("pnt")
                if pnt_item["is_true"]:
                    cv2.circle(frame, pnt, self.__debug_radius_true, self.__debug_color_true, self.__debug_thickness_true)
                    image_helper.put_text(frame, pnt_name, pnt, color=self.__debug_color_true, font_scale=0.75)
                else:
                    cv2.circle(frame, pnt, self.__debug_radius_false, self.__debug_color_false, self.__debug_thickness_false)
                    image_helper.put_text(frame, pnt_name, pnt, color=self.__debug_color_false, font_scale=0.75)

        return res
