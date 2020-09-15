
from os import path, listdir, mkdir, curdir
from ndu_gate_camera.camera.ndu_camera_service import NDUCameraService


def main():
    if "logs" not in listdir(curdir):
        mkdir("logs")
    NDUCameraService(path.dirname(path.abspath(__file__)) + '/config/tb_gateway.yaml'.replace('/', path.sep))


def daemon():
    NDUCameraService("/etc/thingsboard-gateway/config/tb_gateway.yaml".replace('/', path.sep))


if __name__ == '__main__':
    main()
