import json
import os

import cv2

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from openalpr import Alpr

# import openalpr
from ndu_gate_camera.utility import constants


class LprRunner(NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()

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
        self._alpr = Alpr("eu", conf_fn, runtime_data)
        # self._alpr = Alpr("tr", conf_fn, runtime_data)
        # self._alpr = Alpr("au", conf_fn, runtime_data)
        # self._alpr = Alpr("kr2", conf_fn, runtime_data)
        # self._alpr = Alpr("us", conf_fn, runtime_data)

        if not self._alpr.is_loaded():
            print("Error loading OpenALPR")
        # else:
        #     print("aaaaaaaaaaaaaaaaaaa")
        #


        # self._alpr.set_top_n(20)
        # self._alpr.set_default_region("md")
        self._alpr.set_top_n(20)
        self._alpr.set_default_region("tr")



        # results = self._alpr.recognize_file("C:\_koray\.sabit\OpenALPR\openalpr_64\samples\eu-1.jpg")
        # print(results)

    def get_name(self):
        return "lpr"

    def get_settings(self):
        settings = {}
        return settings

    def process_frame(self, frame, extra_data=None):

        def to_bbox(coordinates):
            x1 = coordinates[0]["x"]
            y1 = coordinates[0]["y"]
            x2 = coordinates[2]["x"]
            y2 = coordinates[2]["y"]
            return [y1, x1, y2, x2]

        # success, encoded_image = cv2.imencode('.png', frame)
        success, encoded_image = cv2.imencode('.jpg', frame)
        content2 = encoded_image.tobytes()
        results = self._alpr.recognize_array(content2)

        res = []
        city_counts = {}
        for plate in results['results']:
            txt = plate["plate"]
            if len(txt) > 2:
                if txt[0] == "Q":
                    txt[0] = "0"
                res.append({constants.RESULT_KEY_RECT: to_bbox(plate["coordinates"]),
                            constants.RESULT_KEY_SCORE: plate["confidence"] / 100.0,
                            constants.RESULT_KEY_CLASS_NAME: txt})
                city_code = txt[0:2]
                if city_code in self._cities:
                    city = self._cities[city_code]
                    if city not in city_counts:
                        city_counts[city] = 1
                    else:
                        city_counts[city] += 1

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
