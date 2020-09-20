from ndu_gate_camera.api.video_source import VideoSource
from ndu_gate_camera.utility.ndu_utility import NDUUtility

try:
    from picamera import PiCamera
except ImportError:
    print("picamera library not found - installing...")
    NDUUtility.install_package("picamera")
    import picamera

try:
    from picamera.array import PiRGBArray
except ImportError:
    print("picamera library not found - installing...")
    NDUUtility.install_package("picamera[array]")
    import picamera.array

class PiCameraVideoSource(VideoSource):
    def __init__(self):
        super().__init__()
        pass

    def get_frames(self):
        pass

    def reset(self):
        pass

    def stop(self):
        pass
