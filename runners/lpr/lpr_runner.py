import json
import os

import cv2

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from openalpr import Alpr

from ndu_gate_camera.utility import constants, image_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class LprRunner(NDUCameraRunner):
    def __init__(self, config, _connector_type):
        super().__init__()

        self._rects = config.get("rects", None)
        self._min_confidence = config.get("min_confidence_percentage", 0) / 100.0
        self._resize_width = config.get("resize_width", 0)
        self._send_data = config.get("send_data", False)

        city_codes_fn = "/data/city_codes.json"
        if not os.path.isfile(city_codes_fn):
            city_codes_fn = os.path.dirname(os.path.abspath(__file__)) + city_codes_fn.replace("/", os.path.sep)
        with open(city_codes_fn, encoding="UTF-8") as f_in:
            self._cities = json.load(f_in)

        conf_fn = "/data/openalpr_64/openalpr.conf"
        # conf_fn = "/data/openalpr_64/runtime_data/config/eu.conf"
        if not os.path.isfile(conf_fn):
            conf_fn = os.path.dirname(os.path.abspath(__file__)) + conf_fn.replace("/", os.path.sep)

        runtime_data = "/data/openalpr_64/runtime_data/"
        if not os.path.isdir(runtime_data):
            runtime_data = os.path.dirname(os.path.abspath(__file__)) + runtime_data.replace("/", os.path.sep)

        # self._alpr = Alpr("us", "/path/to/openalpr.conf", "/path/to/runtime_data")
        # self._alpr = Alpr("eu", conf_fn, runtime_data)
        self._alpr = Alpr("tr", conf_fn, runtime_data)

        if not self._alpr.is_loaded():
            print("Error loading OpenALPR")

        # self._alpr.set_top_n(20)
        # self._alpr.set_default_region("md")
        # self._alpr.set_top_n(1)
        self._alpr.set_default_region("tr")
        self._alpr.set_country("tr")

        # results = self._alpr.recognize_file("C:\_koray\.sabit\OpenALPR\openalpr_64\samples\eu-1.jpg")
        # print(results)

    def get_name(self):
        return "lpr"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):

        def to_bbox(coordinates, rect_, rh_, rw_):
            x1 = coordinates[0]["x"] * rw_
            y1 = coordinates[0]["y"] * rh_
            x2 = coordinates[2]["x"] * rw_
            y2 = coordinates[2]["y"] * rh_
            if rect_ is not None:
                x1 += rect_[1]
                y1 += rect_[0]
                x2 += rect_[1]
                y2 += rect_[0]
            return [y1, x1, y2, x2]

        def enumerate_images(frame_):
            if self._rects is None:
                yield frame_, None, None
            else:
                for _class_name, _score, rect_, item_ in NDUUtility.enumerate_results(extra_data, class_name_filters=self._rects, use_wildcard=False, return_item=True):
                    y1, x1, y2, x2 = rect_
                    w = x2 - x1
                    #############if w > 200:
                    if w > 30:
                        # yield image_helper.crop(frame, rect_), rect_, item_
                        h = y2 - y1
                        # h_margin = min(w, h) * 0.5
                        h_margin = w * 0.5
                        r = [y2 - h_margin, x1, y2 + h_margin * 0.5, x2]
                        yield image_helper.crop(frame, r), r, item_

        res = []
        for image, rect, item in enumerate_images(frame):
            h0, w0 = image_helper.image_h_w(image)
            if self._resize_width > 0:
                image = image_helper.resize(image, width=self._resize_width)
            # while w0 < 1200:
            #     image = cv2.pyrUp(image)
            #     h0, w0 = image_helper.image_h_w(image)

            # # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # # image = cv2.pyrUp(image)
            # # image = cv2.pyrUp(image)
            cv2.imshow("lpr", image)
            cv2.waitKey(200)

            h1, w1 = image_helper.image_h_w(image)
            rh = h0 / h1
            rw = h0 / h1

            success, encoded_image = cv2.imencode('.jpg', image)
            content2 = encoded_image.tobytes()
            results = self._alpr.recognize_array(content2)

            for plate in results['results']:
                txt = plate["plate"]
                if len(txt) > 2:
                    score = plate["confidence"] / 100.0
                    if score > self._min_confidence:
                        city_code = txt[0:2]
                        city_name = None
                        if city_code in self._cities:
                            city_name = self._cities[city_code]
                        if city_name is None:
                            class_name = "PL: {}".format(txt)
                        else:
                            class_name = "PL: {} {}".format(city_name, txt)

                        val = {constants.RESULT_KEY_RECT: to_bbox(plate["coordinates"], rect, rh, rw),
                               constants.RESULT_KEY_SCORE: score,
                               constants.RESULT_KEY_CLASS_NAME: class_name}
                        # if city_name is not None:
                        #     val[constants.RESULT_KEY_DATA] = {"license_plate": city_name}
                        if item is not None:
                            track_id = item.get(constants.RESULT_KEY_TRACK_ID, None)
                            if track_id is not None:
                                val[constants.RESULT_KEY_TRACK_ID] = track_id
                        if self._send_data:
                            if city_name is None:
                                val[constants.RESULT_KEY_DATA] = {"pl": txt}
                            else:
                                val[constants.RESULT_KEY_DATA] = {"pl": txt, "city": city_name}

                        res.append(val)
        return res
