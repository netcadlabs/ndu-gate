import getopt
import logging
import sys
import traceback
from os import path, listdir, mkdir, curdir

from yaml import safe_load

from ndu_gate_camera.camera.ndu_camera_service import NDUCameraService
from ndu_gate_camera.camera.ndu_logger import NDULoggerHandler
from ndu_gate_camera.camera.result_handlers.result_handler_file import ResultHandlerFile
from ndu_gate_camera.camera.result_handlers.result_handler_socket import ResultHandlerSocket
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
        config_file_name = "ndu_gate_multiple_source.yaml"
        if NDUUtility.is_debug_mode():
            config_file_name = "ndu_gate_multiple_source_debug.yaml"
            import os
            if os.environ.get('COMPUTERNAME', None) == "KORAY":
                config_file_name = "ndu_gate_multiple_source_debug_koray.yaml"

        ndu_gate_config_file = path.dirname(path.abspath(__file__)) + '/config/'.replace('/', path.sep) + config_file_name

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

        result_hand_conf = ndu_gate_config.get("result_handler", None)
        default_result_file_path = "/var/lib/thingsboard_gateway/extensions/camera/"
        if result_hand_conf is None:
            result_hand_conf = {
                "type": "SOCKET",
                "socket": {
                    "port": 60060,
                    "host": "127.0.0.1"
                }
            }

        if str(result_hand_conf.get("type", "FILE")) == str("SOCKET"):
            result_handler = ResultHandlerSocket(result_hand_conf.get("socket", {}), result_hand_conf.get("device", None))
        else:
            result_handler = ResultHandlerFile(result_hand_conf.get("file_path", default_result_file_path))

        instances = ndu_gate_config.get("instances")
        if len(instances) > 1:
            services = []
            for instance in instances:
                instance["source"]["preview_show"] = False
                camera_service = NDUCameraService(instance=instance, config_dir=ndu_gate_config_dir, handler=result_handler)
                camera_service.start()
                services.append(camera_service)
                log.info("NDU-Gate an instance started")

            log.info("NDU-Gate all instances are started")
            for service in services:
                service.join()
        elif len(instances) == 1:
            camera_service = NDUCameraService(instance=instances[0], config_dir=ndu_gate_config_dir, handler=result_handler)
            camera_service.run()
        else:
            log.error("NDUCameraService no source found!")

        log.info("NDUCameraService exiting...")

    except Exception as e:
        print("NDUCameraService PATLADI")
        print(e)
        print("----------------------")
        print(traceback.format_exc())


def daemon():
    print("Start daemon")
    main(['-c', DEFAULT_NDU_GATE_CONF.replace('/', path.sep)])


def daemon_with_conf(config_file):
    print("Start daemon_with_conf {} ".format(config_file))
    main(['-c', config_file.replace('/', path.sep)])


if __name__ == '__main__':
    main(sys.argv[1:])
