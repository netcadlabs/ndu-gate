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
    _GROUPS = "groups"
    # _RECTS = "rects"
    # _PADDING = "padding"
    # _OBJ_DETECTION = "obj_detection"
    # _GROUND = "ground"
    # _DIST = "dist"
    # _CLASS_NAMES = "class_names"
    #
    # _STYLE = "style"
    # _ST_OR = "or"
    # _ST_AND = "and"
    # _ST_TOUCH = "touch"
    # _ST_DIST = "dist"
    #
    # _CLASSIFICATION = "classification"
    # _CLASSIFY_NAMES = "classify_names"

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
        class struct:
            obj_detection = {}
            classification = {}
            def __init__(self, **entries):
                self.__dict__.update(entries)
        groups = {}
        for group_name, group in config.items():
            groups[group_name] = struct(**group)

            aaa = groups[group_name].obj_detection
            bbb = aaa.rects



        rect_names = []
        self.__onnx_classify_names = None
        for group_name, group in config.items():
            style = group.get(self._STYLE, None)
            if style not in [self._ST_OR, self._ST_AND, self._ST_TOUCH]:
                raise Exception(f"Bad config: {group_name} has bad style")
            if self._RECTS in group:
                for name in group[self._RECTS]:
                    if name not in rect_names:
                        rect_names.append(name)
            else:
                group[self._RECTS] = []
            if self._CLASSIFY in group:
                if self.__onnx_classify_names is None:
                    self._init_classification()
                for css_name in group[self._CLASSIFY]:
                    i = 0
                    for css_name0 in self.__onnx_classify_names:
                        if css_name in css_name0:
                            group[self._CLASSIFY_INDEX] = i
                            break
                        i += 1

        if len(rect_names) > 0:
            return config, rect_names
        else:
            return None

    def __init__(self, config, connector_type):
        super().__init__()
        self.__config, self.__rect_names = self._check_config(config.get("groups", None))

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
