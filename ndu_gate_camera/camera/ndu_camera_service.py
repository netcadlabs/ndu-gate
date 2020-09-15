from sys import getsizeof, executable, argv
from os import listdir, path, execv, pathsep, system
from time import time, sleep
import logging
import logging.config
import logging.handlers

from threading import Thread, RLock
from yaml import safe_load
from simplejson import load, dumps, loads

from ndu_gate_camera.utility.ndu_utility import NDUUtility

DEFAULT_RUNNERS = {
    "drivermonitor": "DriverMonitorRunner",
    "socialdistance": "SocialDistanceRunner",
    "emotionanalysis": "EmotionAnalysisRunner",
}


class NDUCameraService:
    def __init__(self, gateway_config_file=None):
        print("in NDUCameraService")

        if gateway_config_file is None:
            gateway_config_file = path.dirname(path.dirname(path.abspath(__file__))) + '/config/tb_gateway.yaml'.replace('/', path.sep)

        with open(gateway_config_file) as general_config:
            self.__config = safe_load(general_config)
        self._config_dir = path.dirname(path.abspath(gateway_config_file)) + path.sep

        logging_error = None
        try:
            logging.config.fileConfig(self._config_dir + "logs.conf", disable_existing_loggers=False)
        except Exception as e:
            logging_error = e

        global log
        log = logging.getLogger('service')
        log.info("NDUCameraService starting...")

        self._default_runners = DEFAULT_RUNNERS
        self._implemented_connectors = {}
        self.available_connectors = {}

        # NDUUtility.check_and_import_all_runners()
        self._load_connectors()
        pass

    def _load_connectors(self):
        self.connectors_configs = {}
        if self.__config.get("connectors"):
            for connector in self.__config['connectors']:
                try:
                    connector_class = NDUUtility.check_and_import(connector["type"], self._default_runners.get(connector["type"], connector.get("class")))
                    self._implemented_connectors[connector["type"]] = connector_class
                    with open(self._config_dir + connector['configuration'], 'r', encoding="UTF-8") as conf_file:
                        connector_conf = load(conf_file)
                        if not self.connectors_configs.get(connector['type']):
                            self.connectors_configs[connector['type']] = []
                        connector_conf["name"] = connector["name"]
                        self.connectors_configs[connector['type']].append({"name": connector["name"], "config": {connector['configuration']: connector_conf}})
                except Exception as e:
                    log.error("Error on loading connector:")
                    log.exception(e)
        else:
            log.error("Connectors - not found! Check your configuration!")
            self.__init_remote_configuration(force=True)
            log.info("Remote configuration is enabled forcibly!")


if __name__ == '__main__':
    NDUCameraService(path.dirname(path.dirname(path.abspath(__file__))) + '/config/camera.config'.replace('/', path.sep))
