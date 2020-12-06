import getopt
import sys
import traceback
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

    try:
        NDUCameraService(ndu_gate_config_file=ndu_gate_config_file)
    except Exception as e:
        print("NDUCameraService PATLADI")
        print(e)
        print("----------------------")
        print(traceback.format_exc())


def daemon():
    NDUCameraService(ndu_gate_config_file=DEFAULT_NDU_GATE_CONF.replace('/', path.sep))


def daemon_with_conf(config_file):
    print("Start daemon_with_conf {} ".format(config_file))
    NDUCameraService(ndu_gate_config_file=config_file.replace('/', path.sep))


if __name__ == '__main__':
    main(sys.argv[1:])
