import cv2

def resize_best_quality(image, size):
    size0 = max(image.shape[0], image.shape[1])
    size1 = max(size[0], size[1])
    if size0 > size1:
        return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)