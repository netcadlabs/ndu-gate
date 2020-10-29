import json
import time
import zmq

from ndu_gate_camera.api.result_handler import log
from ndu_gate_camera.utility import constants


class ResultHandlerSocket:
    def __init__(self, config):
        self.__socket_port = config.get("port", 60600)
        self.__socket_host = config.get("host", "localhost")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://{}:{}".format(self.__socket_host, self.__socket_port))
        time.sleep(.5)
        log.info("ResultHandlerSocket %s:%s", self.__socket_host, self.__socket_port)

    def save_result(self, result, runner_name=None, data_type='telem'):
        """

        :param self:
        :param result: {"key" : "telem-key", "value": 1, "ts" : timestamp }
        :param runner_name:
        :param data_type: telem or attr
        :return:
        """
        try:
            if result is not None:
                for item in result:
                    data = item.get(constants.RESULT_KEY_DATA, None)
                    if data is not None:
                        self.__send_item(data, runner_name, data_type)

        except Exception as e:
            log.error(e)

    def __send_item(self, item, runner_name, data_type):
        print("Sending telemetry data [ %s ]" % item)
        if not item.get("ts"):
            item["ts"] = int(time.time())
        # self.socket.send_json(item)
        data = {
            data_type: item,
            "runner": runner_name
        }

        data_json = json.dumps(data)
        log.debug("Sending data %s ", data_json)
        try:
            self.socket.send_string("ndugate " + data_json)
        except Exception as a:
            log.debug("exception sending item : %s", a)

    def clear_results(self, runner_name=None):
        pass
