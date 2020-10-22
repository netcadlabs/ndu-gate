import numpy as np
import cv2

from threading import Thread
from random import choice
from string import ascii_lowercase

from ndu_gate_camera.api.ndu_camera_runner import NDUCameraRunner


class social_distance_runner(Thread, NDUCameraRunner):
    def __init__(self, config, connector_type):
        super().__init__()
        self.setName(config.get("name", 'social_distance_runner' + ''.join(choice(ascii_lowercase) for _ in range(5))))
        self.__config = config
        self.__connector_type = connector_type

        self.frame_num = 0  ####koray sil

    def get_name(self):
        return "SocialDistanceRunner"

    def get_settings(self):
        return {}


    def process_frame(self, frame, extra_data=None):
        super().process_frame(frame)

        frame_num = self.frame_num ####koray sil
        frame_num += 1

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        # if frame_num == 1:
        #     # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        #     while True:
        #         image = frame
        #         cv2.imshow("image", image)
        #         cv2.waitKey(1)
        #         if len(mouse_pts) == 7:
        #             cv2.destroyWindow("image")
        #             break
        #         first_frame_display = False
        #     four_points = mouse_pts
        #
        #     # Get perspective
        #     M, Minv = get_camera_perspective(frame, four_points[0:4])
        #     pts = src = np.float32(np.array([four_points[4:]]))
        #     warped_pt = cv2.perspectiveTransform(pts, M)[0]
        #     d_thresh = np.sqrt(
        #         (warped_pt[0][0] - warped_pt[1][0]) ** 2
        #         + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        #     )
        #     bird_image = np.zeros(
        #         (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        #     )
        #
        #     bird_image[:] = SOLID_BACK_COLOR
        #     pedestrian_detect = frame


        # draw polygon of ROI
        pts = np.array(
            [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
        )
        cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

        # Detect person and bounding boxes using DNN
        pedestrian_boxes, num_pedestrians = DNN.detect_pedestrians(frame)

        if len(pedestrian_boxes) > 0:
            pedestrian_detect = plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)
            warped_pts, bird_image = plot_points_on_bird_eye_view(
                frame, pedestrian_boxes, M, scale_w, scale_h
            )
            six_feet_violations, ten_feet_violations, pairs = plot_lines_between_nodes(
                warped_pts, bird_image, d_thresh
            )
            # plot_violation_rectangles(pedestrian_boxes, )
            total_pedestrians_detected += num_pedestrians
            total_pairs += pairs

            total_six_feet_violations += six_feet_violations / fps
            abs_six_feet_violations += six_feet_violations
            pedestrian_per_sec, sh_index = calculate_stay_at_home_index(
                total_pedestrians_detected, frame_num, fps
            )

        last_h = 75
        text = "# 6ft violations: " + str(int(total_six_feet_violations))
        pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

        text = "Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
        pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

        if total_pairs != 0:
            sc_index = 1 - abs_six_feet_violations / total_pairs

        text = "Social-distancing Index: " + str(np.round(100 * sc_index, 1)) + "%"
        pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

        cv2.imshow("Street Cam", pedestrian_detect)
        cv2.waitKey(1)
        output_movie.write(pedestrian_detect)
        bird_movie.write(bird_image)




