import math
import sys

import numpy as np
import cv2

from threading import Thread

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, geometry_helper, image_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility
# from sort import Sort
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)


class ObjectCounterRunner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.connector_type = connector_type
        self.__classes = config.get("classes", None)
        self.__frame_num = 0
        self.config = config
        self.track = config.get("track", "center")
        self.__gates = []
        self.__last_data = {}
        # self.__debug = True  #####koray

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

        # return 0 <= s <= 1 and 0 <= t <= 1
        return -0.01 <= s <= 1.01 and -0.01 <= t <= 1.01

    @staticmethod
    def get_center(line):
        return [(line[0][0] + line[1][0]) * 0.5, (line[0][1] + line[1][1]) * 0.5]

    @staticmethod
    def get_box_center(box):
        (x, y) = (int(box[0]), int(box[1]))
        (w, h) = (int(box[2]), int(box[3]))
        return x + w / 2, y + h / 2

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

            # test
            lines0 = image_helper.select_lines(frame, self.get_name())
            n_lines = image_helper.normalize_lines(frame, lines0)
            n_lines_for_config = image_helper.convert_lines_tuple2list(n_lines)

            # test

            lines = image_helper.denormalize_lines(frame, n_lines)
            ln_counter = 0
            for line in lines:
                ln_counter += 1
                self.__gates.append({"g": {"sorts": {}, "memory": {}, "handled_indexes": []},
                                     "line": line, "classes": {}, "name": self.gate_prefix + str(ln_counter)})

        for gate in self.__gates:
            line = gate["line"]
            gate_name = gate["name"]
            pts = np.array(line, np.int32)

            if self.__frame_num == 1:
                for group_name, sub_classes in self.__classes.items():
                    for class_name in sub_classes:
                        telemetry_enter = gate_name + "_" + class_name + "_enter"
                        telemetry_exit = gate_name + "_" + class_name + "_exit"
                        telemetry_inside = gate_name + "_" + class_name + "_inside_count"
                        data = {telemetry_enter: 0, telemetry_exit: 0, telemetry_inside: 0}
                        res.append({constants.RESULT_KEY_DATA: data})

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

        box_class_names = []
        active_counts = {}
        class_dets = {}
        results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
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
                                det = [rect[1], rect[0], rect[3], rect[2], score]
                                class_dets[group_name].append(det)
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
                        if index_id in previous:
                            previous_box = previous.get(index_id, None)
                            if previous_box is not None:
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))

                                if self.track == "bottom":
                                    p0 = (int(x + (w - x) / 2), int(y + (h - y)))
                                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2)))
                                else:  # center - default value
                                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))

                                # if self.__debug:
                                #     # cv2.line(frame, p0, p1, [0, 0, 255], 3)
                                #     cv2.line(frame, p0, p0, [0, 0, 255], 4)

                                # handled = False
                                line = gate["line"]
                                # if self.intersect(p0, p1, line[0], line[1]):
                                if index_id not in g_handled_indexes and self.intersect(p0, p1, line[0], line[1]):
                                    # handled = True
                                    g_handled_indexes.append(index_id)
                                    classes = gate["classes"]

                                    c = self.get_box_center(box)
                                    min_dist = sys.maxsize
                                    # box_class_names.append({"center": self.get_box_center(det), "class_name": class_name})
                                    for item in box_class_names:
                                        c0 = item["center"]
                                        dist = geometry_helper.distance(c, c0)
                                        if dist < min_dist:
                                            min_dist = dist
                                            name = item["class_name"]

                                    if name not in classes:
                                        classes[name] = {"enter": 0, "exit": 0}
                                    if self.on_left(line, p1):
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

        debug_texts = debug_text = None
        if self.__debug:
            debug_texts = []
        for gate in self.__gates:
            gate_name = gate["name"]
            all_inside = 0
            all_enter = 0
            all_exit = 0
            changed = False
            for name, val in gate["classes"].items():
                if self.__debug:
                    debug_text = self.debug_gate_name(gate_name) + ": "

                enter_val = val["enter"]
                exit_val = val["exit"]
                count_val = enter_val - exit_val
                all_inside += count_val
                all_enter += enter_val
                all_exit += exit_val
                if self.__debug:
                    debug_text += "{} - Giren:{} Cikan:{}".format(NDUUtility.debug_conv_turkish(name), enter_val, exit_val)
                    debug_texts.append(debug_text)

                data_val = str(enter_val) + "_" + str(exit_val)
                if name not in self.__last_data or not self.__last_data[name] == data_val:
                    self.__last_data[name] = data_val
                    changed = True
                    telemetry_enter = gate_name + "_" + name + "_enter"
                    telemetry_exit = gate_name + "_" + name + "_exit"
                    telemetry_inside = gate_name + "_" + name + "_inside"
                    data = {telemetry_enter: enter_val, telemetry_exit: exit_val, telemetry_inside: count_val}
                    res.append({constants.RESULT_KEY_DATA: data})
            if changed:
                telemetry_inside = gate_name + "_inside"
                telemetry_enter = gate_name + "_enter"
                telemetry_exit = gate_name + "_exit"
                data = {telemetry_inside: all_inside, telemetry_enter: all_enter, telemetry_exit: all_exit}
                res.append({constants.RESULT_KEY_DATA: data})

        if self.__debug:
            for name, value in active_counts.items():
                debug_texts.append("Gorunen '{}': {}".format(NDUUtility.debug_conv_turkish(name), value))

            for debug_text in debug_texts:
                res.append({constants.RESULT_KEY_DEBUG: debug_text})

        # res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: name, constants.RESULT_KEY_PREVIEW_KEY: preview_key})
        return res


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    @staticmethod
    def linear_assignment(cost_matrix):
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

        # try:
        #     _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        #     row_sol, cost, v, u, cost_mat = lapjv_python(cost_matrix, calc_reduced_cost_matrix=True)
        #     return np.array([[y[i], i] for i in x if i >= 0])  #
        # except ImportError:
        #     from scipy.optimize import linear_sum_assignment
        #     x, y = linear_sum_assignment(cost_matrix)
        #     return np.array(list(zip(x, y)))

    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """
        From SORT: Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o

    class KalmanBoxTracker(object):
        @staticmethod
        def convert_x_to_bbox(x, score=None):
            """
            Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
              [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
            """
            w = np.sqrt(x[2] * x[3])
            h = x[2] / w
            if score is None:
                return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
            else:
                return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

        @staticmethod
        def convert_bbox_to_z(bbox):
            """
            Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
              [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
              the aspect ratio
            """
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = bbox[0] + w / 2.
            y = bbox[1] + h / 2.
            s = w * h  # scale is just area
            r = w / float(h)
            return np.array([x, y, s, r]).reshape((4, 1))

        """
        This class represents the internal state of individual tracked objects observed as bbox.
        """
        count = 0

        def __init__(self, bbox):
            """
            Initialises a tracker using initial bounding box.
            """
            # define constant velocity model
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
            self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

            self.kf.R[2:, 2:] *= 10.
            self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
            self.kf.P *= 10.
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01

            self.kf.x[:4] = self.convert_bbox_to_z(bbox)
            self.time_since_update = 0
            self.id = self.count
            self.count += 1
            self.history = []
            self.hits = 0
            self.hit_streak = 0
            self.age = 0

        def update(self, bbox):
            """
            Updates the state vector with observed bbox.
            """
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(self.convert_bbox_to_z(bbox))

        def predict(self):
            """
            Advances the state vector and returns the predicted bounding box estimate.
            """
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1
            self.history.append(self.convert_x_to_bbox(self.kf.x))
            return self.history[-1]

        def get_state(self):
            """
            Returns the current bounding box estimate.
            """
            return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = Sort.iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = Sort.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = self.KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            ret = np.concatenate(ret)
            if len(dets) > len(ret):
                ret = []
                i = len(self.trackers)
                for trk in reversed(self.trackers):
                    d = trk.get_state()[0]
                    if (trk.time_since_update < self.max_age) and (trk.hit_streak >= 0 or self.frame_count <= 0):
                        ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                    i -= 1
            else:
                return ret
        else:
            return np.empty((0, 5))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
