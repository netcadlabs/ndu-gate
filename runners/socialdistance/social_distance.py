import numpy as np
import cv2

from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner
from ndu_gate_camera.utility import constants


class social_distance_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.setName(config.get("name", 'social_distance_runner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        self.__connector_type = connector_type

        self.__frame_num = 0

        # self.__mouse_pts = None
        self.__mouse_pts = config.get("mouse_points", None)
        self.__M = None

    def get_name(self):
        return "social_distance_runner"

    def get_settings(self):
        return {}

    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)
        res = []

        scale_w = 1.2 / 2
        scale_h = 4 / 2

        self.__frame_num += 1

        if self.__mouse_pts is None and self.__frame_num == 1:
            global mouse_pts
            def get_mouse_points(event, x, y, flags, param):
                # Used to mark 4 points on the frame zero of the video that will be warped
                # Used to mark 2 points on the frame zero of the video that are 6 feet away
                global mouseX, mouseY, mouse_pts
                if event == cv2.EVENT_LBUTTONDOWN:
                    mouseX, mouseY = x, y
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), 10)
                    if "mouse_pts" not in globals():
                        mouse_pts = []
                    mouse_pts.append((x, y))
                    print("Point detected")
                    print(mouse_pts)

            cv2.namedWindow("image")
            cv2.setMouseCallback("image", get_mouse_points)

            # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
            mouse_pts = []
            while True:
                cv2.imshow("image", frame)
                cv2.waitKey(1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("image")
                    break
                self.__mouse_pts = mouse_pts
            # self.__mouse_pts = select_points(frame, 6, self.get_name())

        # draw polygon of ROI
        pts = np.array(
            [self.__mouse_pts[0], self.__mouse_pts[1], self.__mouse_pts[3], self.__mouse_pts[2]], np.int32
        )
        cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

        # Detect person and bounding boxes using DNN
        # pedestrian_boxes, num_pedestrians = DNN.detect_pedestrians(frame)
        pedestrian_boxes, num_pedestrians = self._get_pedestrians(extra_data)

        if len(pedestrian_boxes) > 0:
            # pedestrian_detect = self._plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)

            if self.__M is None:
                self.__M, self.__d_thresh = self._get_params(frame, self.__mouse_pts)

            warped_pts, bird_image = self._plot_points_on_bird_eye_view(
                frame, pedestrian_boxes, self.__M, scale_w, scale_h
            )
            six_feet_violations, ten_feet_violations, pairs = self._plot_lines_between_nodes(
                warped_pts, bird_image, self.__d_thresh
            )

            # res.append({constants.RESULT_KEY_RECT: rect_face, constants.RESULT_KEY_CLASS_NAME: self.__onnx_class_names[class_id], constants.RESULT_KEY_SCORE: score})
            res.append({constants.RESULT_KEY_CLASS_NAME: six_feet_violations})

            # # plot_violation_rectangles(pedestrian_boxes, )
            # total_pedestrians_detected += num_pedestrians
            # total_pairs += pairs
            #
            # total_six_feet_violations += six_feet_violations / fps
            # abs_six_feet_violations += six_feet_violations
            # pedestrian_per_sec, sh_index = calculate_stay_at_home_index(
            #     total_pedestrians_detected, self.__frame_num, fps
            # )

        # last_h = 75
        # text = "# 6ft violations: " + str(int(total_six_feet_violations))
        # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
        #
        # text = "Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
        # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
        #
        # if total_pairs != 0:
        #     sc_index = 1 - abs_six_feet_violations / total_pairs
        #
        # text = "Social-distancing Index: " + str(np.round(100 * sc_index, 1)) + "%"
        # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
        #
        # cv2.imshow("Street Cam", pedestrian_detect)
        # cv2.waitKey(1)
        # output_movie.write(pedestrian_detect)
        # bird_movie.write(bird_image)
        return res

    @staticmethod
    def _get_params(frame, mouse_pts):
        def _get_camera_perspective(img, src_points):
            IMAGE_H = img.shape[0]
            IMAGE_W = img.shape[1]
            src = np.float32(np.array(src_points))
            dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

            M = cv2.getPerspectiveTransform(src, dst)
            M_inv = cv2.getPerspectiveTransform(dst, src)

            return M, M_inv

        # SOLID_BACK_COLOR = (41, 41, 41)
        # frame_h = frame.shape[0]
        # frame_w = frame.shape[1]

        # Get perspective
        M, Minv = _get_camera_perspective(frame, mouse_pts[0:4])

        pts = np.float32(np.array([mouse_pts[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        # bird_image = np.zeros(
        #     (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        # )
        #
        # bird_image[:] = SOLID_BACK_COLOR
        return M, d_thresh

    # @staticmethod
    # def _plot_pedestrian_boxes_on_image(frame, pedestrian_boxes):
    #     frame_h = frame.shape[0]
    #     frame_w = frame.shape[1]
    #     thickness = 2
    #     # color_node = (192, 133, 156)
    #     color_node = (160, 48, 112)
    #     # color_10 = (80, 172, 110)
    #     frame_with_boxes = None
    #     for i in range(len(pedestrian_boxes)):
    #         pt1 = (
    #             int(pedestrian_boxes[i][1] * frame_w),
    #             int(pedestrian_boxes[i][0] * frame_h),
    #         )
    #         pt2 = (
    #             int(pedestrian_boxes[i][3] * frame_w),
    #             int(pedestrian_boxes[i][2] * frame_h),
    #         )
    #
    #         frame_with_boxes = cv2.rectangle(frame, pt1, pt2, color_node, thickness)
    #     return frame_with_boxes

    @staticmethod
    def _plot_lines_between_nodes(warped_points, bird_image, d_thresh):
        from scipy.spatial.distance import pdist, squareform

        p = np.array(warped_points)
        dist_condensed = pdist(p)
        dist = squareform(dist_condensed)

        # Close enough: 10 feet mark
        dd = np.where(dist < d_thresh * 6 / 10)
        close_p = []
        color_10 = (80, 172, 110)
        lineThickness = 4
        ten_feet_violations = len(np.where(dist_condensed < 10 / 6 * d_thresh)[0])
        for i in range(int(np.ceil(len(dd[0]) / 2))):
            if dd[0][i] != dd[1][i]:
                point1 = dd[0][i]
                point2 = dd[1][i]

                close_p.append([point1, point2])

                cv2.line(
                    bird_image,
                    (p[point1][0], p[point1][1]),
                    (p[point2][0], p[point2][1]),
                    color_10,
                    lineThickness,
                )

        # Really close: 6 feet mark
        dd = np.where(dist < d_thresh)
        six_feet_violations = len(np.where(dist_condensed < d_thresh)[0])
        total_pairs = len(dist_condensed)
        danger_p = []
        color_6 = (52, 92, 227)
        for i in range(int(np.ceil(len(dd[0]) / 2))):
            if dd[0][i] != dd[1][i]:
                point1 = dd[0][i]
                point2 = dd[1][i]

                danger_p.append([point1, point2])
                cv2.line(
                    bird_image,
                    (p[point1][0], p[point1][1]),
                    (p[point2][0], p[point2][1]),
                    color_6,
                    lineThickness,
                )
        # # Display Birdeye view
        # cv2.imshow("Bird Eye View", bird_image)
        # cv2.waitKey(1)

        return six_feet_violations, ten_feet_violations, total_pairs

    @staticmethod
    def _plot_points_on_bird_eye_view(frame, pedestrian_boxes, M, scale_w, scale_h):
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        node_radius = 10
        color_node = (192, 133, 156)
        thickness_node = 20
        solid_back_color = (41, 41, 41)

        blank_image = np.zeros(
            (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        )
        blank_image[:] = solid_back_color
        warped_pts = []
        bird_image = None
        for i in range(len(pedestrian_boxes)):
            mid_point_x = int(
                (pedestrian_boxes[i][1] * frame_w + pedestrian_boxes[i][3] * frame_w) / 2
            )
            mid_point_y = int(
                (pedestrian_boxes[i][0] * frame_h + pedestrian_boxes[i][2] * frame_h) / 2
            )

            pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
            warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
            warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]

            warped_pts.append(warped_pt_scaled)
            bird_image = cv2.circle(
                blank_image,
                (warped_pt_scaled[0], warped_pt_scaled[1]),
                node_radius,
                color_node,
                thickness_node,
            )

        return warped_pts, bird_image

    @staticmethod
    def _get_pedestrians(extra_data):
        pedestrian_boxes = []
        num_pedestrians = 0
        if extra_data is not None:
            results = extra_data.get("results", None)
            if results is not None:
                for runner_name, result in results.items():
                    for item in result:
                        class_name = item.get(constants.RESULT_KEY_CLASS_NAME, None)
                        if class_name == "person":
                            rect_face = item.get(constants.RESULT_KEY_RECT, None)
                            if rect_face is not None:
                                bbox = rect_face
                                pedestrian_boxes.append(bbox)
                                num_pedestrians += 1
                                # y1 = max(int(bbox[0]), 0)
                                # x1 = max(int(bbox[1]), 0)
                                # y2 = max(int(bbox[2]), 0)
                                # x2 = max(int(bbox[3]), 0)
                                # w = x2 - x1
                                # h = y2 - y1
                                # dw = int(w * 0.25)
                                # dh = int(h * 0.25)
                                # x1 -= dw
                                # x2 += dw
                                # y1 -= dh
                                # y2 += dh
                                # y1 = max(y1, 0)
                                # x1 = max(x1, 0)
                                # y2 = max(y2, 0)
                                # x2 = max(x2, 0)

        return pedestrian_boxes, num_pedestrians
