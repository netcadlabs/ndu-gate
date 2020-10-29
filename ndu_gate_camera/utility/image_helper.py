import cv2
import base64

class image_helper:
    @staticmethod
    def resize_best_quality(image, size):
        size0 = max(image.shape[0], image.shape[1])
        size1 = max(size[0], size[1])
        if size0 > size1:
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        else:
            return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def resize_if_larger(image, max_dim, interpolation=None):
        h, w = image.shape[:2]
        if w > h:
            if w > max_dim:
                return image_helper.resize(image, width=max_dim)
            else:
                return image
        else:
            if h > max_dim:
                return image_helper.resize(image, height=max_dim)
            else:
                return image

    @staticmethod
    def resize(image, width=None, height=None, interpolation=None):
        dim = None
        h, w = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)

        if interpolation is None:
            return image_helper.resize_best_quality(image, dim)
        else:
            return cv2.resize(image, dim, interpolation=interpolation)

    @staticmethod
    def rescale_frame(frame, percent):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return frame

    @staticmethod
    def frame2base64(frame, scale=40):
        scaled_frame = image_helper.rescale_frame(frame, scale)
        res, frame = cv2.imencode('.png', scaled_frame)
        base64_data = base64.b64encode(frame)
        return base64_data.decode('utf-8')


# def select_points(frame, count, window_name):
#     mouse_pts = []
#
#     def get_mouse_points(event, x, y, flags, param):
#         # global mouseX, mouseY, mouse_pts
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if len(mouse_pts) < count:
#                 cv2.circle(frame, (x, y), 10, (0, 255, 255), 10)
#                 mouse_pts.append((x, y))
#
#     cv2.namedWindow(window_name)
#     # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
#     # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     cv2.setMouseCallback(window_name, get_mouse_points)
#
#     while True:
#         cv2.imshow(window_name, frame)
#         cv2.waitKey(1)
#         if len(mouse_pts) == count:
#             cv2.imshow(window_name, frame)
#             cv2.waitKey(1000)
#             cv2.destroyWindow(window_name)
#             break
#     return mouse_pts
