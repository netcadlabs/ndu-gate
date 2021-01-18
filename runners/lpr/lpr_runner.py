import json
import os

import cv2
import numpy as np

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from openalpr import Alpr

from ndu_gate_camera.utility import constants, image_helper, onnx_helper, yolo_helper, geometry_helper
from ndu_gate_camera.utility.ndu_utility import NDUUtility


class LprRunner(NDUCameraRunner):
    def __init__(self, config, _connector_type):
        super().__init__()

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

        # region lp detection,
        onnx_fn = "/data/yolov4-tiny_lp_416_static.onnx"
        self.input_size = 416

        if not os.path.isfile(onnx_fn):
            onnx_fn = os.path.dirname(os.path.abspath(__file__)) + onnx_fn.replace("/", os.path.sep)

        classes_filename = "/data/class.names"
        if not os.path.isfile(classes_filename):
            classes_filename = os.path.dirname(os.path.abspath(__file__)) + classes_filename.replace("/", os.path.sep)
        self.class_names = ["lp"]
        self.sess_tuple = onnx_helper.get_sess_tuple(onnx_fn)
        # endregion

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
            result = yolo_helper.predict_v4(self.sess_tuple, self.input_size, self.class_names, frame)
            for _class_name, _score, rect0, item_ in NDUUtility.enumerate_result_items(result, return_item=True):
                rect1 = geometry_helper.add_padding_rect(rect0, 0.5)
                yield image_helper.crop(frame, rect1), rect0, item_

        res = []
        for image, rect, item in enumerate_images(frame):
            h0, w0 = image_helper.image_h_w(image)

            # def order_points(pts):
            #     # initialzie a list of coordinates that will be ordered
            #     # such that the first entry in the list is the top-left,
            #     # the second entry is the top-right, the third is the
            #     # bottom-right, and the fourth is the bottom-left
            #     rect = np.zeros((4, 2), dtype="float32")
            #
            #     # the top-left point will have the smallest sum, whereas
            #     # the bottom-right point will have the largest sum
            #     s = pts.sum(axis=1)
            #     rect[0] = pts[np.argmin(s)]
            #     rect[2] = pts[np.argmax(s)]
            #
            #     # now, compute the difference between the points, the
            #     # top-right point will have the smallest difference,
            #     # whereas the bottom-left will have the largest difference
            #     diff = np.diff(pts, axis=1)
            #     rect[1] = pts[np.argmin(diff)]
            #     rect[3] = pts[np.argmax(diff)]
            #
            #     # return the ordered coordinates
            #     return rect
            #
            # def four_point_transform(image, pts):
            #     # obtain a consistent order of the points and unpack them
            #     # individually
            #     rect = order_points(pts)
            #     (tl, tr, br, bl) = rect
            #
            #     # compute the width of the new image, which will be the
            #     # maximum distance between bottom-right and bottom-left
            #     # x-coordiates or the top-right and top-left x-coordinates
            #     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            #     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            #     maxWidth = max(int(widthA), int(widthB))
            #
            #     # compute the height of the new image, which will be the
            #     # maximum distance between the top-right and bottom-right
            #     # y-coordinates or the top-left and bottom-left y-coordinates
            #     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            #     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            #     maxHeight = max(int(heightA), int(heightB))
            #
            #     # now that we have the dimensions of the new image, construct
            #     # the set of destination points to obtain a "birds eye view",
            #     # (i.e. top-down view) of the image, again specifying points
            #     # in the top-left, top-right, bottom-right, and bottom-left
            #     # order
            #     dst = np.array([
            #         [0, 0],
            #         [maxWidth - 1, 0],
            #         [maxWidth - 1, maxHeight - 1],
            #         [0, maxHeight - 1]], dtype="float32")
            #
            #     # compute the perspective transform matrix and then apply it
            #     M = cv2.getPerspectiveTransform(rect, dst)
            #     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            #     return warped
            #
            # def deskew(image):
            #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #     gray = cv2.bitwise_not(gray)
            #     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            #     coords = np.column_stack(np.where(thresh > 0))
            #     angle = cv2.minAreaRect(coords)[-1]
            #     if angle < -45:
            #         angle = -(90 + angle)
            #     else:
            #         angle = -angle
            #     (h, w) = image.shape[:2]
            #     center = (w // 2, h // 2)
            #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
            #     return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            #
            # # def remove_noise_and_smooth(file_name):
            # #     img = cv2.imread(file_name, 0)
            # #     filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
            # #     kernel = np.ones((1, 1), np.uint8)
            # #     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            # #     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # #     img = image_smoothening(img)
            # #     or_image = cv2.bitwise_or(img, closing)
            # #     return or_image
            #
            # # image = deskew(image)
            # # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            h1, w1 = image_helper.image_h_w(image)
            while w1 < 400:
                image = cv2.pyrUp(image)
                h1, w1 = image_helper.image_h_w(image)
            # cv2.imshow("lpr", image)
            # cv2.waitKey(500)

            success, encoded_image = cv2.imencode('.jpg', image)
            content2 = encoded_image.tobytes()
            results = self._alpr.recognize_array(content2)

            added = False
            for plate in results['results']:
                txt = plate["plate"]
                if len(txt) > 2:
                    score = plate["confidence"] / 100.0
                    if score > 0.01:
                        city_code = txt[0:2]
                        city_name = None
                        if city_code in self._cities:
                            city_name = self._cities[city_code]
                        if city_name is None:
                            class_name = "PL: {}".format(txt)
                        else:
                            class_name = "PL: {} {}".format(city_name, txt)

                        val = {constants.RESULT_KEY_RECT: rect,
                               constants.RESULT_KEY_SCORE: score,
                               constants.RESULT_KEY_CLASS_NAME: class_name}

                        if self._send_data:
                            if city_name is None:
                                val[constants.RESULT_KEY_DATA] = {"pl": txt}
                            else:
                                val[constants.RESULT_KEY_DATA] = {"pl": txt, "city": city_name}

                        res.append(val)
                        added = True
            if not added:
                val = {constants.RESULT_KEY_RECT: rect,
                       constants.RESULT_KEY_SCORE: 0,
                       constants.RESULT_KEY_CLASS_NAME: "PL: "}
                res.append(val)
        return res
