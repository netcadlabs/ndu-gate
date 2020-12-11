import errno
import os
from os import path
from threading import Thread
from typing import Optional

import cv2

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, onnx_helper, geometry_helper
from ndu_gate_camera.utility.geometry_helper import add_padding_rect
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class IntersectorRunner(Thread, NDUCameraRunner):

    def _init_classification(self, config):
        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/googlenet-9.onnx"
        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/synset.txt"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)
        self.__onnx_classify_names = onnx_helper.parse_class_names(class_names_fn)
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn, self._onnx_max_engine_count)

    _ST_OR = "or"
    _ST_AND = "and"
    _ST_TOUCH = "touch"
    _ST_DIST = "dist"

    def _check_config(self, config):

        class ObjDet(object):
            def __init__(self):
                self.ground = None
                self.dist = None
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
                        if NDUUtility.wildcard(css_name, name1):
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
        self._check_config(config.get("groups", None))
        self._onnx_max_engine_count = config.get("max_engine_count", 0)

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
        super().process_frame(frame)
        res = []
        if self._conf is None:
            return res

        def has_or(r_, rect_names0_):
            _count = 0
            for r_name in r_.class_names:
                if NDUUtility.wildcard_has_match(rect_names0_, r_name):
                    return True
            return False

        def has_and(r_, rect_names0_):
            _all_ok = False
            for r_name in r_.class_names:
                if not NDUUtility.wildcard_has_match(rect_names0_, r_name):
                    return False
                else:
                    _all_ok = True
            return _all_ok

        def has_touch(r_, rect_names0_, rects0_):
            if has_and(r_, rect_names0_):
                class RDef:
                    def __init__(self):
                        self.name = ""
                        self.rect = None

                rects = []
                for class_name1, score1, rect1 in rects0_:
                    ok = False
                    for r_name in r_.class_names:
                        if NDUUtility.wildcard(class_name1, r_name):
                            ok = True
                            break
                    if ok:
                        r1 = RDef()
                        r1.name = class_name1
                        if r_.padding != 0:
                            rect1 = add_padding_rect(rect1, r_.padding)
                        r1.rect = rect1
                        rects.append(r1)

                for r1 in rects:
                    for r2 in rects:
                        if r1 != r2:
                            if geometry_helper.rects_intersect(r1.rect, r2.rect):
                                # cv2.rectangle(frame, (int(r1.rect[1]), int(r1.rect[0])), (int(r1.rect[3]), int(r1.rect[2])), color=[0, 0, 0], thickness=3)
                                # cv2.rectangle(frame, (int(r2.rect[1]), int(r2.rect[0])), (int(r2.rect[3]), int(r2.rect[2])), color=[255, 255, 255], thickness=3)
                                return True
            return False

        def has_classification(gr_, rects0_):
            rects_css = []
            for class_name_, score_, rect_ in rects0_:
                for css_rect_name in gr_.classification.rect_names:
                    if NDUUtility.wildcard(class_name_, css_rect_name):
                        rects_css.append(rect_)
                        break
            for rect_css in rects_css:
                if self._classify(frame, rect_css, gr.classification.threshold, gr.classification.classify_indexes):
                    return True
            return False

        counts = {}
        for gr in self._conf.groups:
            count = 0
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
                            if has_touch(r, rect_names0, rects0):
                                count += 1
                        elif r.style == self._ST_DIST:
                            # if gr.obj_detection.ground is None or len(gr.)

                            pass  ####

            if count == 0 and gr.classification is not None:
                rects0 = []
                for class_name, score, rect in NDUUtility.enumerate_results(extra_data, gr.classification.rect_names, True):
                    if rect is not None:
                        rects0.append((class_name, score, rect))
                if len(rects0) > 0:
                    if has_classification(gr, rects0):
                        count += 1

            if count > 0:
                counts[gr.name] = count

        for gr_name, count in counts.items():
            res.append({constants.RESULT_KEY_DATA: {gr_name: count}})
        return res
