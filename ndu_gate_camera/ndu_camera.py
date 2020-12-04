import getopt
import sys
from os import path, listdir, mkdir, curdir
from ndu_gate_camera.camera.ndu_camera_service import NDUCameraService
from ndu_gate_camera.utility.constants import DEFAULT_NDU_GATE_CONF
from ndu_gate_camera.utility.ndu_utility import NDUUtility


def main(argv):

    ndu_gate_config_file = ""
    try:
        opts, args = getopt.getopt(argv, "c:", ["config="])
        for opt, arg in opts:
            if opt in ['-c', '--config']:
                ndu_gate_config_file = arg
    except getopt.GetoptError:
        print('ndu_camera.py -c <config_file_path>')
        sys.exit(2)

    if "logs" not in listdir(curdir):
        mkdir("logs")

    if not ndu_gate_config_file:
        config_file_name = "ndu_gate.yaml"
        if NDUUtility.is_debug_mode():
            config_file_name = "ndu_gate_debug.yaml"
        ndu_gate_config_file = path.dirname(path.abspath(__file__)) + '/config/'.replace('/', path.sep) + config_file_name

    NDUCameraService(ndu_gate_config_file=ndu_gate_config_file)


def daemon():
    NDUCameraService(ndu_gate_config_file=DEFAULT_NDU_GATE_CONF.replace('/', path.sep))


if __name__ == '__main__':
    main(sys.argv[1:])
