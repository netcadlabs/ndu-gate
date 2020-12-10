import json
import time
import zmq

from ndu_gate_camera.api.result_handler import log
from ndu_gate_camera.utility import constants


class ResultHandlerSocket:
    def __init__(self, config, device_name):
        self.__socket_port = config.get("port", 60600)
        self.__socket_host = config.get("host", "127.0.0.1")
        self.__device_name = device_name

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect("tcp://{}:{}".format(self.__socket_host, self.__socket_port))
        time.sleep(.5)
        log.info("ResultHandlerSocket %s:%s", self.__socket_host, self.__socket_port)

    def save_result(self, result, device=None, runner_name=None, data_type='telemetry'):
        """

        :param self:
        :param result: {"key" : "telem-key", "value": 1, "ts" : timestamp }
        :param device:
        :param runner_name:
        :param data_type: telemetry or attribute
        :return:
        """
        if device is None:
            device = self.__device_name
        try:
            if result is not None:
                for item in result:
                    data = item.get(constants.RESULT_KEY_DATA, None)
                    if data is not None:
                        self.__send_item(data, device, runner_name, data_type)

        except Exception as e:
            log.error(e)

    def __send_item(self, item, device, runner_name, data_type):
        data = {
            data_type: item,
            "runner": runner_name,
            "deviceName": device
        }
        data_json = json.dumps(data)
        try:
            self.socket.send_string("ndugate " + data_json)
        except Exception as a:
            log.debug("exception sending item : %s", a)

    def clear_results(self, runner_name=None):
        pass
