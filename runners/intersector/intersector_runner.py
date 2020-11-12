import os
from threading import Thread

import cv2
import numpy as np
from os import path
import errno

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, onnx_helper
from ndu_gate_camera.utility.geometry_helper import geometry_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class intersector_runner(Thread, NDUCameraRunner):

    def _init_classification(self):
        onnx_fn = path.dirname(path.abspath(__file__)) + "/data/googlenet-9.onnx"
        class_names_fn = path.dirname(path.abspath(__file__)) + "/data/synset.txt"
        if not path.isfile(onnx_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), onnx_fn)
        if not path.isfile(class_names_fn):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), class_names_fn)
        self.__onnx_sess_tu = onnx_helper.create_sess_tuple(onnx_fn)
        self.__onnx_classify_names = onnx_helper.parse_class_names(class_names_fn)

    def _check_config(self, config):
        def get_obj_detection(_gr0):
            if "obj_detection" not in _gr0:
                return None
            else:
                def get_rect(r0):
                    class rect_det(object):
                        pass

                    r = rect_det()
                    r.padding = r0["padding"] if "padding" in r0 else 0
                    r.style = r0["style"] if "style" in r0 else "or"
                    r.class_names = []
                    if "class_names" in r0:
                        for class_name in r0["class_names"]:
                            r.class_names.append(class_name)
                    return r

                class obj_det(object):
                    pass

                od0 = _gr0["obj_detection"]
                od = obj_det()
                od.ground = od0["ground"] if "ground" in od0 else None
                od.dist = od0["dist"] if "dist" in od0 else None
                if "rects" not in od0:
                    raise Exception("Bad config! no rects in obj_detection node.")
                od.rects = []
                for r0 in od0["rects"]:
                    od.rects.append(get_rect(r0))
                return od

        def get_classification(_gr0):
            if "classification" not in _gr0:
                return None
            else:
                class cs_det(object):
                    pass

                cs0 = _gr0["classification"]
                cs = cs_det()
                cs.threshold = cs0["threshold"] if "threshold" in cs0 else 0.5
                cs.padding = cs0["padding"] if "padding" in cs0 else 0

                cs.rect_names = []
                for r0 in cs0["rect_names"]:
                    cs.rect_names.append(r0)

                if "classify_names" not in cs0:
                    raise Exception("Bad config! no classify_names in classificaion node.")
                cs.classify_names = []
                for name in cs0["classify_names"]:
                    cs.classify_names.append(name)
                return cs

        def enumerate_rect_class_names(_gr):
            if _gr.obj_detection is not None:
                for r in _gr.obj_detection.rects:
                    for name in r.class_names:
                        yield  name
            if _gr.classification is not None:
                for rect_name in _gr.classification.rect_names:
                    yield rect_name

        class config_def(object):
            pass
        class group_def(object):
            pass

        self.__onnx_classify_names = None
        self._conf = config_def()
        self._conf.groups = []
        self._conf.all_rect_names = []
        for group_name, gr0 in config.items():
            gr = group_def()
            gr.name = group_name
            gr.obj_detection = get_obj_detection(gr0)
            gr.classification = get_classification(gr0)
            if self.__onnx_classify_names is None and gr.classification is not None:
                self._init_classification()
            self._conf.groups.append(gr)
            for name in enumerate_rect_class_names(gr):
                if name not in self._conf.all_rect_names:
                    self._conf.all_rect_names.append(name)

    def __init__(self, config, connector_type):
        super().__init__()
        self._check_config(config.get("groups", None))

    def get_name(self):
        return "intersector_runner"

    def get_settings(self):
        settings = {}
        return settings

    def _classify(self, frame, bbox, threshold, classify_index):
        y1 = max(int(bbox[0]), 0)
        x1 = max(int(bbox[1]), 0)
        y2 = max(int(bbox[2]), 0)
        x2 = max(int(bbox[3]), 0)
        image = frame[y1:y2, x1:x2]

        blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (123.68, 116.779, 103.939))
        #####preds = self.__onnx_sess.run([self.__onnx_output_name], {self.__onnx_input_name: blob})[0]
        preds = onnx_helper.run(self.__onnx_sess_tu, [blob])[0]

        # cv2.imshow(str(index), image)
        # cv2.waitKey(100)

        score = preds[0][classify_index]
        return score > threshold

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        res = []
        config = self.__config
        if config is None:
            return res

        rects = []
        for class_name, score, rect in NDUUtility.enumerate_results(extra_data, self.__rect_names):
            if rect is not None:
                rects.append((class_name, score, rect))
        if len(rects) == 0:
            return res

        counts = {}
        for group_name, group in config.items():
            i_rects = {}
            st = group[self._STYLE]
            all_required = st == self._ST_TOUCH or st == self._ST_AND

            rects_ok = all_required
            rect_exists = False
            classify_required = []
            if self._RECTS in group:
                classify_exists = self._CLASSIFY in group
                for class_name in group[self._RECTS]:
                    rect_exists = True
                    if class_name not in i_rects:
                        i_rects[class_name] = []
                    appended = False
                    for rect_name, score, rect in rects:
                        if class_name == rect_name:
                            i_rects[class_name].append(rect)
                            appended = True
                            if classify_exists:
                                for css_name in group[self._CLASSIFY]:
                                    classify_required.append((css_name, rect_name, rect))
                    if all_required and not appended:
                        rects_ok = False
                        break
                    elif not all_required and appended:
                        rects_ok = True
                        break

            if rect_exists and not rects_ok:
                continue

            if len(classify_required) > 0:
                for css_name, rect_name, rect in classify_required:
                    self._classify(frame, rect, )

            if st == self._ST_TOUCH:
                if len(i_rects) > 1:
                    for name0, rects0 in i_rects.items():
                        if rects_ok:
                            for name1, rects1 in i_rects.items():
                                for rect0 in rects0:
                                    for rect1 in rects1:
                                        if name0 != name1 and not geometry_helper.rects_intersect(rect0, rect1):
                                            rects_ok = False
                                            break
                if not rects_ok:
                    continue

                if len(i_cssfs) > 1:
                    for c in i_cssfs:
                        if not c:
                            cssfs_ok = False
                            break
                if not cssfs_ok:
                    continue

            if rects_ok and cssfs_ok:
                if not group_name in counts:
                    counts[group_name] = 1
                else:
                    counts[group_name] += 1

        for group_name, count in counts.items():
            res.append({constants.RESULT_KEY_DATA: {group_name: count}})
        return res
