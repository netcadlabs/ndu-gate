import getopt
import logging
import sys
import time
import traceback
from os import path, listdir, mkdir, curdir
from sys import platform

from yaml import safe_load

from ndu_gate_camera.camera.ndu_camera_service import NDUCameraService
from ndu_gate_camera.camera.ndu_logger import NDULoggerHandler
from ndu_gate_camera.camera.result_handlers.result_handler_file import ResultHandlerFile
from ndu_gate_camera.camera.result_handlers.result_handler_socket import ResultHandlerSocket
from ndu_gate_camera.utility import onnx_helper
from ndu_gate_camera.utility.constants import DEFAULT_NDU_GATE_CONF, DEFAULT_NDU_GATE_CONF_WIN, DEFAULT_HANDLER_SETTINGS
from ndu_gate_camera.utility.ndu_utility import NDUUtility


def _get_config():
    print(sys.argv)
    ndu_gate_config_file = None

    if len(sys.argv) > 1:
        try:
            opts, _ = getopt.getopt(sys.argv[1:], "c:", ["config="])
            for opt, arg in opts:
                if opt in ['-c', '--config']:
                    ndu_gate_config_file = arg
        except getopt.GetoptError:
            print('ndu_camera.py -c <config_file_path>')

    config_file_name = "ndu_gate.yaml"
    if ndu_gate_config_file is None:
        config_file_name = "ndu_gate_debug.yaml"
        import os
        if os.environ.get('COMPUTERNAME', None) == "KORAY":
            config_file_name = "ndu_gate_debug_koray.yaml"

        config_file = path.dirname(path.abspath(__file__)) + '/config/'.replace('/', path.sep) + config_file_name
        if path.isfile(config_file):
            ndu_gate_config_file = config_file

    if ndu_gate_config_file is None:
        print("Config file not specified, going to use default")
        if platform == "win32":
            ndu_gate_config_file = DEFAULT_NDU_GATE_CONF_WIN.replace('/', path.sep)
        else:
            ndu_gate_config_file = DEFAULT_NDU_GATE_CONF.replace('/', path.sep)

    print("Config file is {}".format(ndu_gate_config_file))

    return ndu_gate_config_file


def main(ndu_gate_config_file):
    if "logs" not in listdir(curdir):
        mkdir("logs")

    try:
        if ndu_gate_config_file is None:
            ndu_gate_config_file = path.dirname(path.dirname(path.abspath(__file__))) + '/config/ndu_gate.yaml'.replace('/', path.sep)

        if not path.isfile(ndu_gate_config_file):
            print('config parameter is not a file : ', ndu_gate_config_file)
            sys.exit(2)

        print("Using config file : {}".format(ndu_gate_config_file))
        with open(ndu_gate_config_file, encoding="utf-8") as general_config:
            ndu_gate_config = safe_load(general_config)

        ndu_gate_config_dir = path.dirname(path.abspath(ndu_gate_config_file)) + path.sep

        logging_config_file = ndu_gate_config_dir + "logs.conf"
        if NDUUtility.is_debug_mode():
            logging_config_file = ndu_gate_config_dir + "logs_debug.conf"
        try:
            import platform
            if platform.system() == "Darwin":
                ndu_gate_config_dir + "logs_macosx.conf"
            # logging.config.fileConfig(logging_config_file, disable_existing_loggers=False)
        except Exception as e:
            print(e)
            NDULoggerHandler.set_default_handler()

        global log
        log = logging.getLogger('service')
        log.info("NDUCameraService starting...")
        log.info("NDU-Gate logging config file: %s", logging_config_file)
        log.info("NDU-Gate logging service level: %s", log.level)

        onnx_helper.init(ndu_gate_config.get("onnx_runner", None), log)
        extension_folder = ndu_gate_config.get("extension_folder", None)

        result_hand_conf = ndu_gate_config.get("result_handler", None)
        if result_hand_conf is None:
            result_hand_conf = DEFAULT_HANDLER_SETTINGS

        if str(result_hand_conf.get("type", "SOCKET")) == str("SOCKET"):
            result_handler = ResultHandlerSocket(result_hand_conf.get("socket", {}), result_hand_conf.get("device", None))
        else:
            result_handler = ResultHandlerFile(result_hand_conf.get("file_path", None))

        instances = ndu_gate_config.get("instances")
        if len(instances) > 1:
            services = []
            preview_exists = False
            for instance in instances:
                if instance["source"].get("preview_show", False):
                    preview_exists = True
                camera_service = NDUCameraService(instance=instance, config_dir=ndu_gate_config_dir, handler=result_handler, is_main_thread=False, extension_folder=extension_folder)
                camera_service.start()
                services.append(camera_service)
                log.info("NDU-Gate an instance started")

            log.info("NDU-Gate all instances are started")
            if preview_exists:
                alive_exists = True
                while alive_exists:
                    time.sleep(0.033333)
                    alive_exists = False
                    for s in services:
                        if s.is_alive():
                            alive_exists = True
                            s.check_for_preview()
                        else:
                            s.finish_preview()
            else:
                for service in services:
                    service.join()
        elif len(instances) == 1:
            camera_service = NDUCameraService(instance=instances[0], config_dir=ndu_gate_config_dir, handler=result_handler, is_main_thread=True, extension_folder=extension_folder)
            camera_service.run()
        else:
            log.error("NDUCameraService no source found!")

        log.info("NDUCameraService exiting...")

    except Exception as e:
        print("NDUCameraService Exited")
        print(e)
        print("----------------------")
        print(traceback.format_exc())


def daemon():
    print("Starting as daemon")
    main(_get_config())


def daemon_with_conf(config_file):
    print("Starting as daemon with conf {} ".format(config_file))
    main(config_file)


if __name__ == '__main__':
    main(_get_config())
