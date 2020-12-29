import math
import sys

import numpy as np
import cv2

from threading import Thread

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, geometry_helper, image_helper, string_helper
from filterpy.kalman import KalmanFilter

np.random.seed(0)


class ObjectCounterRunner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.connector_type = connector_type

        self.__classes = config.get("classes", None)
        self.__concat_data_classes = config.get("concat_data_classes", None)

        # self.__class_name_to_group_name = {}
        self.__frame_num = 0
        self.config = config
        self.track = config.get("track", "center")
        self.__gates = []
        self.__debug = True  ####koray
        # self.__debug = False
        self.__debug_last = {}
        self._result_style = config.get("result_style", 0)  # 0:cumulative - 1:aggregate
        self._track_pnts = {}
        self._last_data = {}

        self._send_sub_classes = True

    def get_name(self):
        return "ObjectCounterRunner"

    def get_settings(self):
        settings = {}
        return settings

    @staticmethod
    def intersect(a, b, c, d):
        p0_x = float(a[0])
        p0_y = float(a[1])
        p1_x = float(b[0])
        p1_y = float(b[1])
        p2_x = float(c[0])
        p2_y = float(c[1])
        p3_x = float(d[0])
        p3_y = float(d[1])

        s1_x = p1_x - p0_x
        s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x
        s2_y = p3_y - p2_y

        div1 = (-s2_x * s1_y + s1_x * s2_y)
        if math.fabs(div1) < 0.0000001:
            return False
        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / div1
        div2 = (-s2_x * s1_y + s1_x * s2_y)
        if math.fabs(div2) < 0.0000001:
            return False
        t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / div2

        return -0.01 <= s <= 1.01 and -0.01 <= t <= 1.01

    @staticmethod
    def get_center(line):
        return [(line[0][0] + line[1][0]) * 0.5, (line[0][1] + line[1][1]) * 0.5]

    @staticmethod
    def on_left(line, p):
        ax = line[0][0]
        ay = line[0][1]
        bx = line[1][0]
        by = line[1][1]
        x = p[0]
        y = p[1]
        return ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) > 0

    @staticmethod
    def rotate_line(line, center, deg=-90):
        def rotate_point(point, angle, center_point=(0, 0)):
            angle_rad = math.radians(angle % 360)
            new_point = (point[0] - center_point[0], point[1] - center_point[1])
            new_point = (new_point[0] * math.cos(angle_rad) - new_point[1] * math.sin(angle_rad),
                         new_point[0] * math.sin(angle_rad) + new_point[1] * math.cos(angle_rad))
            new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
            return int(new_point[0]), int(new_point[1])

        line[0] = rotate_point(line[0], deg, center)
        line[1] = rotate_point(line[1], deg, center)

    gate_prefix = "Gate"

    def debug_gate_name(self, gate_name):
        return gate_name.replace(self.gate_prefix, "KAPI")

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        res = []
        if self.__classes is None:
            return res

        self.__frame_num += 1
        if len(self.__gates) == 0 and self.__frame_num == 1:
            n_lines = image_helper.convert_lines_list2tuple(self.config.get("gates", []))

            # # test
            # lines0 = image_helper.select_lines(frame, self.get_name())
            # n_lines = image_helper.normalize_lines(frame, lines0)
            # n_lines_for_config = image_helper.convert_lines_tuple2list(n_lines)
            # # test

            lines = image_helper.denormalize_lines(frame, n_lines)
            ln_counter = 0
            for line in lines:
                ln_counter += 1
                gate = {"g": {"sorts": {}, "memory": {}, "handled_indexes": []},
                        "line": line, "name": self.gate_prefix + str(ln_counter),
                        "groups": {}}
                for group_name, sub_classes in self.__classes.items():
                    gate["groups"][group_name] = {"enter": 0, "exit": 0}
                    if self._send_sub_classes:
                        for name in sub_classes:
                            gate["groups"][name] = {"enter": 0, "exit": 0}
                self.__gates.append(gate)
            for group_name, sub_classes in self.__classes.items():
                self._last_data[group_name] = None
                if self._send_sub_classes:
                    for class_name in sub_classes:
                        self._last_data[class_name] = None

        for gate in self.__gates:
            line = gate["line"]
            gate_name = gate["name"]
            pts = np.array(line, np.int32)

            if self.__debug:
                center = self.get_center(line)
                cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)
                image_helper.put_text(frame, self.debug_gate_name(gate_name), center, color=(0, 255, 255), font_scale=0.75)

                p0 = center
                p1 = [line[1][0], line[1][1]]
                p1[0] = p0[0] + (p1[0] - p0[0]) * 0.2
                p1[1] = p0[1] + (p1[1] - p0[1]) * 0.2
                arrow = [p0, p1]
                self.rotate_line(arrow, arrow[0], -90)
                cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 0, 0), thickness=3)
                cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 200, 200), thickness=2)
                image_helper.put_text(frame, "giris", arrow[1], color=(0, 200, 200), font_scale=0.5)

        changed = False
        results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
        if results is not None:
            for gate in self.__gates:
                g = gate.get("g")
                g_handled_indexes = g.get("handled_indexes")
                for runner_name, result in results.items():
                    if result is not None:
                        for item in result:
                            class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                            if class_name is not None:
                                for group_name, sub_classes in self.__classes.items():
                                    if class_name in sub_classes:
                                        track_id = item.get(constants.RESULT_KEY_TRACK_ID, None)
                                        #######
                                        if self.__concat_data_classes is not None:
                                            for runner_name1, result1 in results.items():
                                                if result1 is not None:
                                                    for item1 in result1:
                                                        class_name1 = item1.get(constants.RESULT_KEY_CLASS_NAME, None)
                                                        if class_name1 is not None:
                                                            for group_name_, data_classes1 in self.__concat_data_classes.items():
                                                                for data_class_name1 in data_classes1:
                                                                    if string_helper.wildcard(class_name1, data_class_name1):
                                                                        # item1[constants.RESULT_KEY_TRACK_ID] = track_id
                                                                        # item1[constants.RESULT_KEY_DATA] = class_name1
                                                                        key = "concat_data" + str(track_id)
                                                                        gate["groups"][group_name_][key] = class_name1
                                        #######

                                        if track_id is not None and track_id not in g_handled_indexes:
                                            rect = item.get(constants.RESULT_KEY_RECT, None)
                                            [x1, y1, x2, y2] = rect

                                            if self.track == "bottom":
                                                p1 = int(y1 + (y2 - y1) * 0.5), int(x2)
                                            else:  # center - default value
                                                p1 = int(y1 + (y2 - y1) * 0.5), int(x1 + (x2 - x1) * 0.5)

                                            if track_id not in self._track_pnts:
                                                self._track_pnts[track_id] = [p1]
                                            else:
                                                self._track_pnts[track_id].append(p1)
                                                line = gate["line"]
                                                p0 = self._track_pnts[track_id][0]
                                                # p0 = self._track_pnts[track_id][-2]

                                                # if self.__debug:
                                                #     cv2.line(frame, p0, p1, [0, 0, 255], 3)
                                                #     cv2.line(frame, p1, p1, [255, 255, 0], 5)

                                                if self.intersect(p0, p1, line[0], line[1]):
                                                    g_handled_indexes.append(track_id)
                                                    del self._track_pnts[track_id]

                                                    names = [group_name]
                                                    if self._send_sub_classes:
                                                        names.append(class_name)
                                                    for name in names:
                                                        group = gate["groups"][name]
                                                        if self.on_left(line, p0):
                                                            group["enter"] += 1
                                                        else:
                                                            group["exit"] += 1
                                                        group[constants.RESULT_KEY_TRACK_ID] = track_id
                                                        changed = True

                                # if self.__concat_data_classes is not None:
                                #     for group_name, data_classes in self.__concat_data_classes.items():
                                #         for data_class_name in data_classes:
                                #             if string_helper.wildcard(class_name, data_class_name):
                                #                 gate["groups"][group_name]["concat_data"] = class_name

        if changed or self.frame_count == 1:
            for gate in self.__gates:
                gate_name = gate["name"]
                for group_name, group in gate["groups"].items():
                    tel_name = gate_name + "_" + group_name
                    all_enter = group["enter"]
                    all_exit = group["exit"]
                    all_inside = all_enter - all_exit
                    telemetry_inside = tel_name + "_inside"
                    telemetry_enter = tel_name + "_enter"
                    telemetry_exit = tel_name + "_exit"
                    if self._result_style == 1:
                        data = {telemetry_inside: all_inside, telemetry_enter: all_enter, telemetry_exit: all_exit}
                    else:
                        data = {telemetry_enter: all_enter, telemetry_exit: all_exit}
                        group["enter"] = 0
                        group["exit"] = 0
                    track_id = group.get(constants.RESULT_KEY_TRACK_ID, None)
                    if track_id is not None:
                        key = "concat_data" + str(track_id)
                        concat_data = group.get(key, None)
                        if concat_data is not None:
                            del group[key]
                            data["concat_data"] = concat_data
                    if self._result_style != 1 or self._last_data[group_name] != data:
                        if track_id is not None:
                            res.append({constants.RESULT_KEY_DATA: data, constants.RESULT_KEY_TRACK_ID: track_id})
                        else:
                            res.append({constants.RESULT_KEY_DATA: data})
                        self._last_data[group_name] = data

                    # if self.__debug:
                    if self.__debug and (all_enter > 0 or all_exit > 0) and self._result_style == 1:
                        debug_text = "{} - Giren:{} Cikan:{}".format(tel_name.replace("Gate", "Kapi"), all_enter, all_exit)
                        # debug_text = "{} - Giren:{} Cikan:{}".format(gate_name.replace("Gate", "Kapi"), all_enter, all_exit)
                        self.__debug_last[tel_name] = debug_text
                        res.append({constants.RESULT_KEY_DEBUG: debug_text})
        elif self.__debug:
            for tel_name, debug_text in self.__debug_last.items():
                res.append({constants.RESULT_KEY_DEBUG: debug_text})

        return res
