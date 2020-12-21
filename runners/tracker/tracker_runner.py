import math
import sys

import cv2
import numpy as np

from threading import Thread

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, geometry_helper, image_helper

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)


class TrackerRunner(Thread, NDUCameraRunner):
    def __init__(self, config, _connector_type):
        super().__init__()

        self._classes = config.get("classes", None)
        if self._classes is not None:
            self._class_name_to_group_name = {}
            for group_name, sub_classes in self._classes.items():
                for class_name in sub_classes:
                    self._class_name_to_group_name[class_name] = group_name

        self._tracker_algorithm = config.get("tracker_algorithm", "KCF_600")
        self._g = {}
        self._cv_tracker = CvTracker(self._tracker_algorithm)
        self._history = {}
        self._last_items = []

    def get_name(self):
        return "TrackerRunner"

    def get_settings(self):
        settings = {}
        return settings

    @staticmethod
    def _get_box_center(box):
        (x, y) = (int(box[0]), int(box[1]))
        (w, h) = (int(box[2]), int(box[3]))
        return x + w / 2, y + h / 2

    def _get_group_name(self, class_name):
        if self._classes is not None and len(self._classes) > 0:
            for group_name, sub_classes in self._classes.items():
                if class_name in sub_classes:
                    return group_name
        else:
            return class_name

    @staticmethod
    def _det_exists(dets, det, accept_ratio=0.8):
        if dets is not None and len(dets) > 0:
            x1, y1, x2, y2, _ = tuple(det)
            w = x2 - x1
            h = y2 - y1
            for det_ in dets:
                x1_, y1_, x2_, y2_, _ = tuple(det_)
                w_intr = min(x2, x2_) - max(x1, x1_)
                if w_intr / w > accept_ratio:
                    h_intr = min(y2, y2_) - max(y1, y1_)
                    if h_intr / h > accept_ratio:
                        return True
        return False

    @staticmethod
    def _rect_to_bbox(rect, max_x, max_y):
        y1, x1, y2, x2 = tuple(rect)
        x1 = max(0, min(x1, max_x))
        x2 = max(0, min(x2, max_x))
        y1 = max(0, min(y1, max_y))
        y2 = max(0, min(y2, max_y))
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            return x1, y1, w, h
        else:
            return None

    @staticmethod
    def _bbox_to_rect(bbox):
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        return [y1, x1, y2, x2]

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        items = []
        results = extra_data.get(constants.EXTRA_DATA_KEY_RESULTS, None)
        if results is not None:
            h0, w0 = image_helper.image_h_w(frame)
            max_x = w0 - 1
            max_y = h0 - 1
            for runner_name, result in results.items():
                remove_indexes = []
                for result_index, item in enumerate(result):
                    rect = item.get(constants.RESULT_KEY_RECT, None)
                    if rect is not None:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        group_name = self._get_group_name(class_name)
                        if group_name is not None:
                            bbox = self._rect_to_bbox(rect, max_x, max_y)
                            if bbox is not None:
                                item["tr_bbox"] = bbox
                                item["tr_group_name"] = group_name
                                remove_indexes.append(result_index)
                                handled = False
                                for i, item0 in enumerate(items):
                                    rect0 = item0[constants.RESULT_KEY_RECT]
                                    if geometry_helper.rects_overlap(rect, rect0, 0.75):
                                        handled = True
                                        if item.get(constants.RESULT_KEY_SCORE, 0.9) > item0.get(constants.RESULT_KEY_SCORE, 0.9):
                                            items[i] = item
                                        break
                                if not handled:
                                    items.append(item)
                remove_indexes.sort(reverse=True)
                for i in remove_indexes:
                    del result[i]

        if len(items) > 0:
            self._cv_tracker = CvTracker(self._tracker_algorithm)
            self._last_items = items
            for item in items:
                bbox = item["tr_bbox"]
                self._cv_tracker.add(frame, bbox)
        else:
            items = self._last_items
            items_ok = []
            for i, (ok, bbox) in enumerate(self._cv_tracker.update(frame)):
                if ok:
                    item = items[i]
                    item["tr_bbox"] = bbox
                    item[constants.RESULT_KEY_RECT] = self._bbox_to_rect(bbox)
                    items_ok.append(item)
                    # p1 = (int(bbox[0]), int(bbox[1]))
                    # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    # cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                # else:
                #     item = items[i]
                #     bbox = item["tr_bbox"]
                #     item[constants.RESULT_KEY_RECT] = self._bbox_to_rect(bbox)
                #     items_ok.append(item)

            items = items_ok

        center_item_index = []
        active_counts = {}
        class_dets = {}
        for i, item in enumerate(items):
            if "tr_bbox" in item:
                rect = item.get(constants.RESULT_KEY_RECT, None)
                group_name = item["tr_group_name"]
                if group_name not in class_dets:
                    class_dets[group_name] = []
                score = item.get(constants.RESULT_KEY_SCORE, 0.9)
                det = [rect[1], rect[0], rect[3], rect[2], score]
                if not self._det_exists(class_dets[group_name], det):
                    class_dets[group_name].append(det)
                    center_item_index.append((self._get_box_center(det), item, i))

        g = self._g
        g_sorts = g.get("sorts", {})
        g_memory = g.get("memory", {})

        previous = g_memory.copy()
        g_memory = {}

        for name, dets in class_dets.items():
            if name not in g_sorts:
                g_sorts[name] = Sort(max_age=10, min_hits=1, iou_threshold=0.001)
                # g_sorts[name] = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
                # g_sorts[name] = Sort()
                # g_sorts[name] = Sort(max_age=1, min_hits=1, iou_threshold=0.03)
                # g_sorts[name] = Sort(max_age=10, min_hits=0, iou_threshold=0.000001)
                # g_sorts[name] = Sort(max_age=30, min_hits=1, iou_threshold=0.001)

            sort = g_sorts[name]  # ref: https://github.com/abewley/sort   https://github.com/HodenX/python-traffic-counter-with-yolo-and-sort
            dets1 = np.array(dets)
            tracks = sort.update(dets1)

            active_counts[name] = len(tracks)

            boxes = []
            track_ids = []

            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3], track[4]])
                track_id = int(track[4])
                track_ids.append(track_id)
                if track_id in previous:
                    track0 = previous[track_id]
                    dist = (math.fabs(track[0] - track0[0]) + math.fabs(track[1] - track0[1]) + math.fabs(track[2] - track0[2]) + math.fabs(track[3] - track0[3])) / 4.0
                    if dist > 30:
                        g_memory[track_id] = boxes[-1]
                    else:
                        g_memory[track_id] = track0
                else:
                    g_memory[track_id] = boxes[-1]

            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    track_id = track_ids[i]
                    if track_id in previous:
                        previous_box = previous.get(track_id, None)
                        if previous_box is not None:
                            c = self._get_box_center(box)
                            min_dist = sys.maxsize
                            result_item = None
                            # result_item_index = None
                            for center, item, i0 in center_item_index:
                                dist = geometry_helper.distance(c, center)
                                if dist < min_dist:
                                    min_dist = dist
                                    result_item = item
                                    # result_item_index = i0
                            if result_item is not None:
                                result_item[constants.RESULT_KEY_RECT_TRACK_ID] = track_id
                                result_item[constants.RESULT_KEY_PREVIEW_KEY] = track_id
                                if track_id not in self._history:
                                    self._history[track_id] = {"item": result_item, "age": 0}
                                else:
                                    item0 = self._history[track_id]["item"]
                                    self._history[track_id]["age"] = 0
                                    if result_item.get(constants.RESULT_KEY_SCORE, 0.9) < item0.get(constants.RESULT_KEY_SCORE, 0.9):
                                        result_item[constants.RESULT_KEY_CLASS_NAME] = item0[constants.RESULT_KEY_CLASS_NAME]
                                        result_item[constants.RESULT_KEY_SCORE] = item0[constants.RESULT_KEY_SCORE]
                                        # items[result_item_index] = item0
                                    else:
                                        self._history[track_id]["item"] = result_item

        del_lst = []
        for track_id, hist in self._history.items():
            if hist["age"] > 50:
                del_lst.append(track_id)
            else:
                hist["age"] += 1
        for track_id in del_lst:
            del self._history[track_id]

        g["sorts"] = g_sorts
        g["memory"] = g_memory
        return items


class CvTracker(object):
    @staticmethod
    def _create_tracker(tracker_algorithm):
        # python3 -m pip install opencv-contrib-python
        if tracker_algorithm == "KCF": return cv2.TrackerKCF_create()  # fps:17  - Default  / dim düşünce performansı arttı, başarım iyi
        if tracker_algorithm == "CSRT": return cv2.TrackerCSRT_create()  # fps:4   - Başarısı daha iyi ama yavaş
        if tracker_algorithm == "MOSSE": return cv2.TrackerMOSSE_create()  # fps:150 - Başarı düşük ama çok hızlı  / dim yükselince performansı biraz düştü ama başarı artmadı

        if tracker_algorithm == "MedianFlow": return cv2.TrackerMedianFlow_create()  # fps:15
        if tracker_algorithm == "Boosting": return cv2.TrackerBoosting_create()  # fps:4
        if tracker_algorithm == "MIL": return cv2.TrackerMIL_create()  # fps:2
        if tracker_algorithm == "TLD": return cv2.TrackerTLD_create()  # fps:1
        if tracker_algorithm == "GOTURN":  cv2.TrackerGOTURN_create()  # fps:3

        raise Exception("Bad tracker_algorithm: {}".format(tracker_algorithm))

    def __init__(self, tracker_algorithm):
        self._cv_trackers = []
        self._tracker_algorithm = tracker_algorithm
        if tracker_algorithm == "KCF_600":
            self._ratio_h = self._ratio_w = 0
            self._dim = 600
            self.add = self._add_resized
            self.update = self._update_resized
            self._tracker_algorithm = "KCF"
        else:
            self.add = self._add
            self.update = self._update

    def _add(self, image, bbox):
        tracker = self._create_tracker(self._tracker_algorithm)
        tracker.init(image, bbox)
        self._cv_trackers.append(tracker)

    def _update(self, frame):
        res = []
        for tracker in self._cv_trackers:
            ok, bbox = tracker.update(frame)
            res.append((ok, bbox))
        return res

    def _add_resized(self, image, bbox):
        h0, w0 = image_helper.image_h_w(image)
        image = image_helper.resize_if_larger(image, self._dim)
        h1, w1 = image_helper.image_h_w(image)
        self._ratio_h = h0 / h1
        self._ratio_w = w0 / w1
        tracker = self._create_tracker(self._tracker_algorithm)
        x1, y1, w, h = bbox
        bbox = (x1 / self._ratio_w, y1 / self._ratio_h, w / self._ratio_w, h / self._ratio_h)
        tracker.init(image, bbox)
        self._cv_trackers.append(tracker)

    def _update_resized(self, image):
        image = image_helper.resize_if_larger(image, self._dim)
        res = []
        for tracker in self._cv_trackers:
            ok, bbox = tracker.update(image)
            if ok:
                x1, y1, w, h = bbox
                bbox = (x1 * self._ratio_w, y1 * self._ratio_h, w * self._ratio_w, h * self._ratio_h)
            res.append((ok, bbox))
        return res


def linear_assignment(cost_matrix):
    # try:
    #     import lap
    #     _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    #     return np.array([[y[i], i] for i in x if i >= 0])  #
    # except ImportError:
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
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


class KalmanBoxTracker(object):
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

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
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
        self.kf.update(convert_bbox_to_z(bbox))

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
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
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
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
