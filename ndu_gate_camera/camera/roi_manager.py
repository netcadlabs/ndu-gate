import sys
from threading import Thread

import cv2
import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, geometry_helper, image_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class ROIManager():
    def __init__(self, config, name):
        self.Name = name
        self._polygons = config.get("polygons", [])
        self._apply_mask = config.get("apply_mask", True)
        self._apply_crop = config.get("apply_crop", True)
        self._preview = config.get("preview", False)
        self._pyrUp = config.get("pyrUp", 0)
        self._select_polygons_mode = config.get("select_polygons_mode", False)

        self._inited = False
        self._mask = None
        self._roi = None
        self._last_frame = None
        self._rev_div = 1

    def forward(self, frame):
        self._last_frame = frame
        if not self._inited:
            self._inited = True
            if self._select_polygons_mode:
                polygons = image_helper.select_areas(frame, "Select points for roi config {}".format(self.Name), max_count=None, next_area_key="n", finish_key="s", return_tuples=False)
                arr1 = []
                for polygon in polygons:
                    arr = []
                    for p in polygon:
                        arr.append(list(p))
                    arr1.append(arr)
                print('"polygons": {},'.format(str(arr1)))
                exit(1)

            min_max = []
            for i in range(len(self._polygons)):
                area = np.array(self._polygons[i], np.int32)
                self._polygons[i] = area
                min_max.append(area.min(axis=0, initial=None))
                min_max.append(area.max(axis=0, initial=None))
            min_max = np.array(min_max, np.int32)
            min_ = min_max.min(axis=0, initial=None)
            max_ = min_max.max(axis=0, initial=None)

            if self._apply_crop:
                self._roi = [min_[1], min_[0], max_[1], max_[0]]
            if self._apply_mask:
                self._mask = image_helper.get_mask(frame.shape, self._polygons)
                if self._roi is not None:
                    self._mask = image_helper.crop(self._mask, self._roi)

        # masked = image_helper.apply_mask(frame, self._mask)
        img = frame
        if self._apply_crop:
            img = image_helper.crop(img, self._roi)
        if self._apply_mask:
            img = image_helper.apply_mask(img, self._mask)
        if self._pyrUp > 0:
            self._rev_div = 1
            for i in range(self._pyrUp):
                self._rev_div *= 2
                img = cv2.pyrUp(img, img)

        # img = image_helper.equalize_hist_bgr(img)
        # img = image_helper.adaptive_equalize_hist_bgr(img)

        if self._preview:
            cv2.imshow("ROIManager", img)
        return img

    def reverse(self, frame, results):
        if self._apply_crop and results is not None and len(results) > 0:
            for result in results:
                if "rect" in result:
                    rect = result["rect"]
                    if self._rev_div > 1:
                        for i in range(len(rect)):
                            rect[i] = int(rect[i] / self._rev_div)
                    rect[0] += self._roi[0]
                    rect[1] += self._roi[1]
                    rect[2] += self._roi[0]
                    rect[3] += self._roi[1]
        return self._last_frame, results
