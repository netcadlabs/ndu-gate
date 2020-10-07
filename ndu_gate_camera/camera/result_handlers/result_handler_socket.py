import json
import time
import zmq

from ndu_gate_camera.api.result_handler import log


class ResultHandlerSocket:
    def __init__(self, config):
        self.__socket_port = config.get("port", 60600)
        self.__socket_host = config.get("host", "localhost")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://" + str(self.__socket_host) + ":" + str(self.__socket_port))

        log.info("ResultHandlerSocket %s:%s", self.__socket_host, self.__socket_port)

    def save_result(self, result, runner_name=None):
        """

        :param self:
        :param result: {"key" : "telem-key", "value": 1, "ts" : timestamp }
        :param runner_name:
        :return:
        """
        try:
            if isinstance(result, list):
                for item in result:
                    self.__send_item(item)
            else:
                self.__send_item(result)
        except Exception as e:
            log.error(e)

    def __send_item(self, item):
        print("Sending request %s …" % item)
        if not item.get("ts"):
            item["ts"] = int(time.time())
        self.socket.send_json(item)
        message = self.socket.recv()
        print("Received reply [ %s ]" % message)

    def clear_results(self, runner_name=None):
        pass
