import json
import os

import cv2

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from openalpr import Alpr

# import openalpr
from ndu_gate_camera.utility import constants, image_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class LprRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()

        self._rects = config.get("rects", None)
        self._min_confidence = config.get("min_confidence_percentage", 0) / 100.0

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
        # else:
        #     print("aaaaaaaaaaaaaaaaaaa")
        #

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
                yield frame_, None
            else:
                for _class_name, _score, rect_, item_ in NDUUtility.enumerate_results(extra_data, class_name_filters=self._rects, use_wildcard=False, return_item=True):
                    yield image_helper.crop(frame, rect_), rect_, item_

        res = []
        city_counts = {}
        for image, rect, item in enumerate_images(frame):
            h0, w0 = image_helper.image_h_w(image)
            # if max(h0, w0) > 200:
            if w0 > 200:
                # image = image_helper.resize_if_smaller(image, 1280)
                image = image_helper.resize(image, width=1200)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = cv2.pyrUp(image)
                # image = cv2.pyrUp(image)
                # cv2.imshow("aaaa", image)
                # cv2.waitKey(1)

                h1, w1 = image_helper.image_h_w(image)
                rh = h0 / h1
                rw = h0 / h1

                # success, encoded_image = cv2.imencode('.png', frame)
                success, encoded_image = cv2.imencode('.jpg', image)
                content2 = encoded_image.tobytes()
                results = self._alpr.recognize_array(content2)

                for plate in results['results']:
                    txt = plate["plate"]
                    if len(txt) > 2:
                        # if txt[0] == "Q":
                        #     txt = txt.replace('Q', '0', 1)
                        score = plate["confidence"] / 100.0
                        if score > self._min_confidence:
                            city_code = txt[0:2]
                            city = None
                            if city_code in self._cities:
                                city = self._cities[city_code]
                                if city not in city_counts:
                                    city_counts[city] = 1
                                else:
                                    city_counts[city] += 1
                            if city is None:
                                class_name = "PL: {}".format(txt)
                            else:
                                class_name = "PL: {} {}".format(city, txt)
                            res.append({constants.RESULT_KEY_RECT: to_bbox(plate["coordinates"], rect, rh, rw),
                                        constants.RESULT_KEY_SCORE: score,
                                        constants.RESULT_KEY_CLASS_NAME: class_name})

                        # print("   %12s %12s" % ("Plate", "Confidence"))
                        # for candidate in plate['candidates']:
                        #     prefix = "-"
                        #     if candidate['matches_template']:
                        #         prefix = "*"
                        #
                        #     print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
                        #     break

                # alpr.unload()

        for name, count in city_counts.items():
            res.append({constants.RESULT_KEY_DATA: {name: count}})
        return res
