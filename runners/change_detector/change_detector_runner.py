import errno
import os
import time
from os import path
from threading import Thread
from typing import Optional

import cv2
import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants, onnx_helper, geometry_helper, string_helper, image_helper
from ndu_gate_camera.utility.geometry_helper import add_padding_rect
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class ChangeDetectorRunner(Thread, NDUCameraRunner):

    def __init__(self, config, _):
        super().__init__()
        self._selection_mode = config.get("selection_mode", False)
        self._debug_mode = config.get("debug_mode", False)
        self._change_kernel = config.get("change_kernel", 21)
        self._areas = config.get("areas", [])

        areas_info = config.get("areas_info", [])
        self._area_names = []
        self._area_keys = []
        for i in range(len(self._areas)):
            if len(areas_info) > i:
                self._area_names.append(areas_info[i]["name"])
                self._area_keys.append(areas_info[i]["key"])
            else:
                self._area_names.append("area" + str(i))
                self._area_keys.append("area" + str(i))

        self._last_data_txt = None
        self._frame_index = 0
        self._masks = None
        self._ground_truths = None

    def get_name(self):
        return "ChangeDetectorRunner"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):
        self._frame_index += 1
        if self._selection_mode:
            areas = image_helper.select_areas(frame, "select change detection areas", max_count=None, max_point_count=None, next_area_key="n", finish_key="s")
            gr = []
            for area0 in areas:
                area = []
                for pnt in area0:
                    area.append(list(pnt))
                gr.append(list(area))
            print('"areas": {},'.format(str(gr)))
            exit(1)

        if self._frame_index == 1:
            self._update_ground_truths(frame)
            return []

        super().process_frame(frame)
        res = []

        data_items = []
        data_txt = ""
        for i in range(len(self._areas)):
            mask = self._masks[i]
            ground_truth = self._ground_truths[i]
            has_changed = self._has_changed(frame, mask, ground_truth)
            if self._debug_mode:
                area_name = self._area_names[i]
                area = self._areas[i]
                g = np.array(area, np.int32)
                color = (0, 0, 255) if has_changed else (0, 255, 255)
                thickness = 4 if has_changed else 2

                cv2.polylines(frame, [g], True, color=color, thickness=thickness)
                center = geometry_helper.get_center_int(area)
                image_helper.put_text(frame, area_name, center, color=color, font_scale=0.75)

            area_key = self._area_keys[i]
            data_items.append({area_key: has_changed})
            data_txt = data_txt + area_key + str(has_changed)

        if self._last_data_txt != data_txt:
            self._last_data_txt = data_txt
            for data in data_items:
                res.append({constants.RESULT_KEY_DATA: data})

        return res

    def _update_ground_truths(self, frame):
        self._masks = []
        self._ground_truths = []
        for area in self._areas:
            mask = image_helper.get_mask(frame.shape, [area])
            self._masks.append(mask)
            self._ground_truths.append(self._process_frame(frame, mask))

    def _process_frame(self, frame, mask):
        img = image_helper.apply_mask(frame, mask)
        gray = image_helper.resize(img, width=500, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self._change_kernel, self._change_kernel), 0)
        return gray

    def _has_changed(self, frame, mask, ground_truth):
        gray = self._process_frame(frame, mask)
        frame_delta = cv2.absdiff(ground_truth, gray)
        last_thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        no_zero = cv2.countNonZero(last_thresh)
        # if self._debug_mode:
        #     image_helper.put_text(last_thresh, str(no_zero), [50, 50])
        #     cv2.imshow("last_thresh", last_thresh)
        #     cv2.waitKey(1)
        return no_zero > 1
