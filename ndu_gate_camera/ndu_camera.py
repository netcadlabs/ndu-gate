from os import path, listdir, mkdir, curdir
from ndu_gate_camera.camera.ndu_camera_service import NDUCameraService
from ndu_gate_camera.utility.constants import DEFAULT_NDU_GATE_CONF


def main():
    if "logs" not in listdir(curdir):
        mkdir("logs")
    # tb_gateway_config_file = path.dirname(path.abspath(__file__)) + '/config/tb_gateway.yaml'.replace('/', path.sep)
    ndu_gate_config_file = path.dirname(path.abspath(__file__)) + '/config/ndu_gate.yaml'.replace('/', path.sep)
    NDUCameraService(ndu_gate_config_file=ndu_gate_config_file)


def daemon():
    NDUCameraService(ndu_gate_config_file=DEFAULT_NDU_GATE_CONF.replace('/', path.sep))


if __name__ == '__main__':
    main()
