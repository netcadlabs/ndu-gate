import errno
import os
from os import path
from threading import Thread
from typing import Optional

import cv2
import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, onnx_helper, geometry_helper, string_helper, image_helper
from ndu_gate_camera.utility.geometry_helper import add_padding_rect
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class IntersectorRunner(Thread, NDUCameraRunner):

    def _init_classification(self):
        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/googlenet-9.onnx"
        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/synset.txt"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)
        self.__onnx_classify_names = onnx_helper.parse_class_names(class_names_fn)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)

    _ST_OR = "or"
    _ST_AND = "and"
    _ST_TOUCH = "touch"
    _ST_DIST = "dist"
    _ST_STATIONARY = "stationary"

    def _check_config(self, config):

        class ObjDet(object):
            def __init__(self):
                self.ground = None
                self.dist = None
                self.stationary_frame_count = None
                self.rects = []
                self.all_rect_names = []

        class CssDet(object):
            def __init__(self):
                self.threshold = 0.5
                self.padding = 0
                self.rect_names = []
                self.classify_indexes = []

        class ConfigDef(object):
            def __init__(self):
                self.groups = []
                self.all_rect_names = []

        class GroupDef(object):
            def __init__(self):
                self.name = ""
                self.obj_detection = ObjDet()
                self.classification = CssDet()
                self._M = None
                self._M_size = None
                self._dist_warped_sq = None
                self.active_frame = None

            def _warp_point(self, p):
                matrix = self._M
                px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
                py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
                return [int(px), int(py)]

            def _warp_rect(self, rect):
                y1, x1, y2, x2 = geometry_helper.get_rect_values(rect)
                p1 = self._warp_point([x1, y1])
                p2 = self._warp_point([x2, y2])
                return geometry_helper.get_rect(p1, p2)

            def rects_within_dist(self, r1, r2):
                if self._M is None:
                    ground = self.obj_detection.ground
                    dist = self.obj_detection.dist

                    # ground: [[112, 1043], [565, 1027], [533, 934], [148, 938]]
                    ground1 = []
                    for i in [3, 2, 0, 1]:
                        ground1.append(ground[i])
                    pts1 = np.float32(ground1)
                    x1 = 450
                    y1 = 450
                    x2 = 550
                    y2 = 550
                    pts2 = np.float32([[y1, x1], [y2, x1], [y1, x2], [y2, x2]])

                    self._M = cv2.getPerspectiveTransform(pts1, pts2)
                    self._M_size = (1000, 1000)

                    dist_w = [self._warp_point(dist[0]), self._warp_point(dist[1])]
                    self._dist_warped_sq = geometry_helper.get_dist_sq(dist_w[0], dist_w[1])

                rect1 = self._warp_rect(r1)
                rect2 = self._warp_rect(r2)

                p1 = geometry_helper.get_rect_bottom_center(rect1)
                p2 = geometry_helper.get_rect_bottom_center(rect2)
                d_sq = geometry_helper.get_dist_sq(p1, p2)
                res = d_sq <= self._dist_warped_sq
                if self.active_frame is not None:  # self._debug_mode
                    frame = self.active_frame
                    dst = cv2.warpPerspective(frame, self._M, self._M_size)

                    # cv2.polylines(dst, [np.array(rect1, np.int32)], True, color=(255, 255, 255), thickness=1)
                    # cv2.polylines(dst, [np.array(rect2, np.int32)], True, color=(255, 255, 255), thickness=1)
                    color = [0, 0, 255] if res else [128, 128, 128]
                    image_helper.draw_rect(dst, rect1, color=color)
                    image_helper.draw_rect(dst, rect2, color=color)

                    cv2.imshow("intersector_runner", dst)
                    cv2.waitKey(1)

                    ground = self.obj_detection.ground
                    g = np.array(ground, np.int32)
                    cv2.polylines(frame, [g], True, color=(0, 255, 255), thickness=2)
                    dist = self.obj_detection.dist
                    d = np.array(dist, np.int32)
                    cv2.polylines(frame, [d], False, color=(255, 255, 0), thickness=2)
                return res

        class RectDet(object):
            def __init__(self):
                self.padding = 0
                self.style = IntersectorRunner._ST_OR
                self.class_names = []

        def get_obj_detection(_gr0) -> Optional[ObjDet]:
            if "obj_detection" not in _gr0:
                return None
            else:
                def get_rect(r0_):
                    r_ = RectDet()
                    r_.padding = r0_["padding"] if "padding" in r0_ else 0
                    r_.style = r0_["style"] if "style" in r0_ else "or"
                    if "class_names" in r0_:
                        for class_name in r0_["class_names"]:
                            r_.class_names.append(class_name)
                    return r_

                od0 = _gr0["obj_detection"]
                od = ObjDet()
                od.ground = od0["ground"] if "ground" in od0 else None
                od.dist = od0["dist"] if "dist" in od0 else None
                od.stationary_frame_count = od0["stationary_frame_count"] if "stationary_frame_count" in od0 else None
                if "rects" not in od0:
                    raise Exception("Bad config! no rects in obj_detection node.")
                for r0 in od0["rects"]:
                    r = get_rect(r0)
                    if len(r.class_names) == 0:
                        raise Exception("Bad config! no class_name in rects.")
                    od.rects.append(r)
                    for name1 in r.class_names:
                        if name1 not in od.all_rect_names:
                            od.all_rect_names.append(name1)
                return od

        def get_classification(_gr0) -> Optional[CssDet]:
            if "classification" not in _gr0:
                return None
            else:
                cs0 = _gr0["classification"]
                cs = CssDet()
                cs.threshold = cs0["threshold"] if "threshold" in cs0 else 0.5
                cs.padding = cs0["padding"] if "padding" in cs0 else 0

                for r0 in cs0["rect_names"]:
                    cs.rect_names.append(r0)

                if "classify_names" not in cs0:
                    raise Exception("Bad config! no classify_names in classificaion node.")
                for name1 in cs0["classify_names"]:
                    for i in range(len(self.__onnx_classify_names)):
                        css_name = self.__onnx_classify_names[i]
                        if string_helper.wildcard(css_name, name1):
                            cs.classify_indexes.append(i)
                            break
                return cs

        def enumerate_rect_class_names(_gr) -> str:
            if _gr.obj_detection is not None:
                for r in _gr.obj_detection.rects:
                    for name_ in r.class_names:
                        yield name_
            if _gr.classification is not None:
                for rect_name in _gr.classification.rect_names:
                    yield rect_name

        self.__onnx_classify_names = None
        self._conf = ConfigDef()
        for group_name, gr0 in config.items():
            gr = GroupDef()
            gr.name = group_name
            gr.obj_detection = get_obj_detection(gr0)
            if self.__onnx_classify_names is None and gr.classification is not None:
                self._init_classification()
            gr.classification = get_classification(gr0)
            self._conf.groups.append(gr)
            for name in enumerate_rect_class_names(gr):
                if name not in self._conf.all_rect_names:
                    self._conf.all_rect_names.append(name)

    def __init__(self, config, _):
        super().__init__()
        self._selection_mode = config.get("ground_dist_selection_mode", False)
        self._debug_mode = config.get("debug_mode", False)
        if not self._selection_mode:
            self._check_config(config.get("groups", None))
        self._last_data = {}

    def get_name(self):
        return "IntersectorRunner"

    def get_settings(self):
        settings = {}
        return settings

    def _classify(self, frame, bbox, threshold, classify_indexes):
        y1 = max(int(bbox[0]), 0)
        x1 = max(int(bbox[1]), 0)
        y2 = max(int(bbox[2]), 0)
        x2 = max(int(bbox[3]), 0)
        image = frame[y1:y2, x1:x2]

        blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (123.68, 116.779, 103.939))
        preds = onnx_helper.run(self.sess_tuple, [blob])[0]

        # cv2.imshow(str(classify_indexes), image)
        # cv2.waitKey(100)

        for classify_index in classify_indexes:
            score = preds[0][classify_index]
            if score > threshold:
                return True
        return False

    def process_frame(self, frame, extra_data=None):
        if self._selection_mode:
            areas = image_helper.select_areas(frame, "select intersector ground", max_count=1, max_point_count=4, next_area_key="n", finish_key="s")
            gr = []
            for p in areas[0]:
                gr.append(list(p))
            print('"ground": {},'.format(str(gr)))
            lines = image_helper.select_lines(frame, "select intersector dist", max_count=1)
            dist = []
            for p in lines[0]:
                dist.append(list(p))
            print('"dist": {},'.format(str(dist)))
            exit(1)

        super().process_frame(frame)
        res = []
        if self._conf is None:
            return res

        def has_or(r_, rect_names0_):
            _count = 0
            for r_name in r_.class_names:
                if string_helper.wildcard_has_match(rect_names0_, r_name):
                    return True
            return False

        def has_and(r_, rect_names0_):
            _all_ok = False
            for r_name in r_.class_names:
                if not string_helper.wildcard_has_match(rect_names0_, r_name):
                    return False
                else:
                    _all_ok = True
            return _all_ok

        def _has_interaction(r_, rect_names0_, rects0_, touch_check_func):
            if has_and(r_, rect_names0_):
                class RDef:
                    def __init__(self):
                        self.name = ""
                        self.rect = None

                rects_ = []
                for class_name1, score1, rect1 in rects0_:
                    ok_ = False
                    for r_name in r_.class_names:
                        if string_helper.wildcard(class_name1, r_name):
                            ok_ = True
                            break
                    if ok_:
                        r1_ = RDef()
                        r1_.name = class_name1
                        if r_.padding != 0:
                            rect1 = add_padding_rect(rect1, r_.padding)
                        r1_.rect = rect1
                        rects_.append(r1_)

                for r1_ in rects_:
                    for r2_ in rects_:
                        if r1_ != r2_:
                            if touch_check_func(r1_.rect, r2_.rect):
                                if self._debug_mode:
                                    cv2.rectangle(frame, (int(r1_.rect[1]), int(r1_.rect[0])), (int(r1_.rect[3]), int(r1_.rect[2])), color=[0, 0, 0], thickness=3)
                                    cv2.rectangle(frame, (int(r2_.rect[1]), int(r2_.rect[0])), (int(r2_.rect[3]), int(r2_.rect[2])), color=[255, 255, 255], thickness=3)
                                return True, r1_, r2_
            return False, None, None

        def has_touch(r_, rect_names0_, rects0_):
            return _has_interaction(r_, rect_names0_, rects0_, geometry_helper.rects_intersect)

        def has_dist(gr_, r_, rect_names0_, rects0_):
            return _has_interaction(r_, rect_names0_, rects0_, gr_.rects_within_dist)

        def has_classification(gr_, rects0_):
            rects_css = []
            for class_name_, score_, rect_ in rects0_:
                for css_rect_name in gr_.classification.rect_names:
                    if string_helper.wildcard(class_name_, css_rect_name):
                        rects_css.append(rect_)
                        break
            for rect_css in rects_css:
                if self._classify(frame, rect_css, gr.classification.threshold, gr.classification.classify_indexes):
                    return True
            return False

        counts = {}
        rects = {}
        for gr in self._conf.groups:
            rects[gr.name] = []
            count = 0
            if self._debug_mode:
                gr.active_frame = frame
            if gr.obj_detection is not None:
                rects0 = []
                rect_names0 = []
                for class_name, score, rect in NDUUtility.enumerate_results(extra_data, gr.obj_detection.all_rect_names, True):
                    if rect is not None:
                        rects0.append((class_name, score, rect))
                        rect_names0.append(class_name)
                if len(rects0) > 0:
                    for r in gr.obj_detection.rects:
                        if r.style == self._ST_OR:
                            if has_or(r, rect_names0):
                                count += 1
                        elif r.style == self._ST_AND:
                            if has_and(r, rect_names0):
                                count += 1
                        elif r.style == self._ST_TOUCH:
                            ok, r1, r2 = has_touch(r, rect_names0, rects0)
                            if ok:
                                count += 1
                                rects[gr.name].extend([r1.rect, r2.rect])
                        elif r.style == self._ST_DIST:
                            ok, r1, r2 = has_dist(gr, r, rect_names0, rects0)
                            if ok:
                                count += 1
                                rects[gr.name].extend([r1.rect, r2.rect])

            if count == 0 and gr.classification is not None:
                rects0 = []
                for class_name, score, rect in NDUUtility.enumerate_results(extra_data, gr.classification.rect_names, True):
                    if rect is not None:
                        rects0.append((class_name, score, rect))
                        rects[gr.name].append(rect)
                if len(rects0) > 0:
                    if has_classification(gr, rects0):
                        count += 1

            counts[gr.name] = count
            if count > 0:
                for rect in rects[gr.name]:
                    res.append({constants.RESULT_KEY_RECT: rect, constants.RESULT_KEY_CLASS_NAME: gr.name})

        for gr_name, count in counts.items():
            if gr_name not in self._last_data:
                self._last_data[gr_name] = None
            val = {constants.RESULT_KEY_DEBUG: "{} - {}".format(gr_name, count)}
            if self._last_data[gr_name] != count:
                self._last_data[gr_name] = count
                val[constants.RESULT_KEY_DATA] = {gr_name: count, "{} exists".format(gr_name).replace(" ", "_"): count > 0}
            res.append(val)
        return res
