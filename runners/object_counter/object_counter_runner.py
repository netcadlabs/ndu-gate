import math

import numpy as np
import cv2

from threading import Thread

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants
from sort import Sort


class object_counter_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.connector_type = connector_type
        self.__classes = config.get("classes", None)
        self.__frame_num = 0
        self.__gates = []
        self.__last_data = {}

    def get_name(self):
        return "object_counter_runner"

    def get_settings(self):
        settings = {}
        return settings

    @staticmethod
    def _select_lines(frame, window_name):
        lines = []
        line = []

        def get_mouse_points(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(line) < 2:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), 10)
                    line.append((x, y))

        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 40, 30)
        cv2.setMouseCallback(window_name, get_mouse_points)

        while True:
            for ln in lines:
                pts = np.array(ln, np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("s"):
                cv2.destroyWindow(window_name)
                break
            if len(line) == 2:
                lines.append(line)
                line = []

        return lines

    @staticmethod
    def put_text(img, text, center, color=None, font_scale=0.5):
        if color is None:
            color = [255, 255, 255]
        cv2.putText(img=img, text=text, org=(int(center[0]) + 5, int(center[1])),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=[0, 0, 0], lineType=cv2.LINE_AA,
                    thickness=2)
        cv2.putText(img=img, text=text, org=(int(center[0]) + 5, int(center[1])),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, color=color,
                    lineType=cv2.LINE_AA, thickness=1)

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

        # return 0 <= s <= 1 and 0 <= t <= 1
        return -0.01 <= s <= 1.01 and -0.01 <= t <= 1.01

        # # Return true if line segments AB and CD intersect
        # def ccw(a_, b_, c_):
        #     return (c_[1] - a_[1]) * (b_[0] - a_[0]) > (b_[1] - a_[1]) * (c_[0] - a_[0])
        #
        # return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        # @staticmethod
        # def intersect(a1, a2, b1, b2):
        #     def perp(a):
        #         b = np.empty_like(a)
        #         b[0] = -a[1]
        #         b[1] = a[0]
        #         return b
        #     a1 = np.array(a1)
        #     a2 = np.array(a2)
        #     b1 = np.array(b1)
        #     b2 = np.array(b2)
        #     da = a2 - a1
        #     db = b2 - b1
        #     dp = a1 - b1
        #     dap = perp(da)
        #     denom = np.dot(dap, db)
        #     num = np.dot(dap, dp)
        #     return (num / denom.astype(float)) * db + b1

    @staticmethod
    def get_center(line):
        return [(line[0][0] + line[1][0]) * 0.5, (line[0][1] + line[1][1]) * 0.5]

    @staticmethod
    def get_box_center(box):
        (x, y) = (int(box[0]), int(box[1]))
        (w, h) = (int(box[2]), int(box[3]))
        return (x + w / 2, y + h / 2)

    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

    def process_frame(self, frame, extra_data=None):

        w, h = frame.shape[:2]
        new_size = 1024*768
        from ndu_gate_camera.utility.image_helper import image_helper
        frame = image_helper.resize_total_pixel_count(frame, new_size)
        w1, h1 = frame.shape[:2]
        size1 = w1 * h1

        super().process_frame(frame)
        res = []
        if self.__classes is None:
            return res

        self.__frame_num += 1
        if len(self.__gates) == 0 and self.__frame_num == 1:
            lines = self._select_lines(frame, self.get_name())
            # lines = [[(622, 520), (167, 500)], [(907, 535), (670, 535)], [(1468, 717), (1201, 492)]]
            ln_counter = 0
            for line in lines:
                ln_counter += 1
                # self.__gates.append({"line": line, "enter": 0, "exit": 0, "name": "Kapı" + str(ln_counter)})
                self.__gates.append({"g": {"sorts": {}, "memory": {}, "handled_indexes": []},
                                     "line": line, "classes": {}, "name": "KAPI" + str(ln_counter)})

        for gate in self.__gates:
            line = gate["line"]
            name = gate["name"]
            pts = np.array(
                line, np.int32
            )
            cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)
            center = self.get_center(line)
            self.put_text(frame, name, center, color=(0, 255, 255), font_scale=0.75)

            p0 = center
            p1 = [line[1][0], line[1][1]]
            p1[0] = p0[0] + (p1[0] - p0[0]) * 0.2
            p1[1] = p0[1] + (p1[1] - p0[1]) * 0.2
            arrow = [p0, p1]
            self.rotate_line(arrow, arrow[0], -90)
            cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 0, 0), thickness=3)
            cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 200, 200), thickness=2)
            self.put_text(frame, "giris", arrow[1], color=(0, 200, 200), font_scale=0.5)

        box_class_names = []
        active_counts = {}
        class_dets = {}
        results = extra_data.get("results", None)
        if results is not None:
            for runner_name, result in results.items():
                for item in result:
                    class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                    for group_name, sub_classes in self.__classes.items():
                        if class_name in sub_classes:
                            rect = item.get(constants.RESULT_KEY_RECT, None)
                            if rect is not None:
                                if group_name not in class_dets:
                                    class_dets[group_name] = []
                                score = item.get(constants.RESULT_KEY_SCORE, 0.9)
                                # det = [rect[0], rect[1], rect[2], rect[3], score]
                                det = [rect[1], rect[0], rect[3], rect[2], score]
                                class_dets[group_name].append(det)

                                ######koray  sil
                                if class_name == "person":
                                    class_name = "insan"

                                if class_name == "car":
                                    class_name = "otomobil"

                                if class_name == "bicycle":
                                    class_name = "bisiklet"

                                if class_name == "motorbike":
                                    class_name = "motosiklet"

                                if class_name == "truck":
                                    class_name = "kamyon"

                                if class_name == "bus":
                                    class_name = "otobus"

                                box_class_names.append({"center": self.get_box_center(det), "class_name": class_name})

        for gate in self.__gates:
            g = gate.get("g")
            g_sorts = g.get("sorts")
            g_memory = g.get("memory")
            g_handled_indexes = g.get("handled_indexes")
            for name, dets in class_dets.items():
                if name not in g_sorts:
                    # g_sorts[name] = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
                    # g_sorts[name] = Sort()
                    # g_sorts[name] = Sort(max_age=1, min_hits=1, iou_threshold=0.03)
                    # g_sorts[name] = Sort(max_age=10, min_hits=0, iou_threshold=0.000001)
                    g_sorts[name] = Sort(max_age=10, min_hits=1, iou_threshold=0.001)
                    # g_sorts[name] = Sort(max_age=30, min_hits=1, iou_threshold=0.001)

                sort = g_sorts[name]  # ref: https://github.com/abewley/sort   https://github.com/HodenX/python-traffic-counter-with-yolo-and-sort
                dets1 = np.array(dets)
                tracks = sort.update(dets1)

                active_counts[name] = len(tracks)

                boxes = []
                index_ids = []
                previous = g_memory.copy()
                g_memory = {}

                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3], track[4]])
                    index_id = int(track[4])
                    index_ids.append(index_id)
                    if index_id in previous:
                        track0 = previous[index_id]
                        dist = (math.fabs(track[0] - track0[0]) + math.fabs(track[1] - track0[1]) + math.fabs(track[2] - track0[2]) + math.fabs(track[3] - track0[3])) / 4.0
                        if dist > 30:
                            # if False: #########
                            g_memory[index_id] = boxes[-1]
                        else:
                            g_memory[index_id] = track0
                    else:
                        g_memory[index_id] = boxes[-1]

                if len(boxes) > 0:
                    i = 0
                    for box in boxes:
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

                        index_id = index_ids[i]
                        # if index_id not in g_handled_indexes and index_id in previous:
                        if index_id in previous:  #########
                            previous_box = previous.get(index_id, None)
                            if previous_box is not None:
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))

                                # center
                                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                                # # bottom
                                # p0 = (int(x + (w - x) / 2), int(y + (h - y)))
                                # p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2)))

                                # cv2.line(frame, p0, p1, [0, 0, 255], 3)
                                # # cv2.line(frame, p0, p0, [0, 0, 255], 3)

                                handled = False
                                line = gate["line"]
                                # if self.intersect(p0, p1, line[0], line[1]):
                                if index_id not in g_handled_indexes and self.intersect(p0, p1, line[0], line[1]):  #########
                                    handled = True
                                    g_handled_indexes.append(index_id)
                                    classes = gate["classes"]

                                    c = self.get_box_center(box)
                                    min_dist = 9999999999
                                    # box_class_names.append({"center": self.get_box_center(det), "class_name": class_name})
                                    for item in box_class_names:
                                        c0 = item["center"]
                                        dist = self.distance(c, c0)
                                        if dist < min_dist:
                                            min_dist = dist
                                            name = item["class_name"]

                                    if name not in classes:
                                        classes[name] = {"enter": 0, "exit": 0}
                                    # if not self.on_left(line, p0):
                                    if self.on_left(line, p1):  #####
                                        classes[name]["enter"] += 1
                                    else:
                                        classes[name]["exit"] += 1

                                # if not handled:
                                #     g_memory[index_id] = previous_box
                        i += 1

            g["sorts"] = g_sorts
            g["memory"] = g_memory
            g["handled_indexes"] = g_handled_indexes
            gate["g"] = g

        debug_texts = []
        for gate in self.__gates:
            for name, val in gate["classes"].items():
                gate_name = gate["name"]
                debug_text = gate_name + ": "

                enter_val = val["enter"]
                exit_val = val["exit"]
                debug_text += f"{name} - Giren:{enter_val} Cikan:{exit_val}"

                debug_texts.append(debug_text)


                data_val = str(enter_val) + "_" + str(exit_val)
                changed = False
                if name not in self.__last_data or not self.__last_data[name] == data_val:
                    self.__last_data[name] = data_val
                    changed = True
                if changed:
                    telemetry_enter = gate_name + "_" + name + "_giris"
                    telemetry_exit = gate_name + "_" + name + "_cikis"
                    data = {telemetry_enter: enter_val, telemetry_exit: exit_val}
                    res.append({constants.RESULT_KEY_DATA: data})

        for name, value in active_counts.items():
            debug_texts.append(f"gorunen '{name}': {value}")

        for debug_text in debug_texts:
            # res.append({constants.RESULT_KEY_DEBUG: debug_text})
            res.append({constants.RESULT_KEY_DEBUG: debug_text})


        # res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: name, constants.RESULT_KEY_PREVIEW_KEY: preview_key})
        return res
