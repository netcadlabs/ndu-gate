import json
import socket

from ndu_gate_camera.api.result_handler import log


class ResultHandlerSocket:
    def __init__(self, folder):
        self.workingPath = folder
        self.__socket_port = 60600
        self.__socket_host = '127.0.0.1'
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        pass

    def open_socket(self):
        try:
            self.__socket.connect((self.__socket_host, self.__socket_port))
        except Exception as e:
            log.error(e)

    @staticmethod
    def add_result(self, result, runner_name=None):
        """

        :param self:
        :param result: {"key" : "telem-key", "value": 1, "ts" : timestamp }
        :param runner_name:
        :return:
        """
        try:
            data = json.dumps(result)
            data_byte = bytes(data, encoding="utf-8")
            self.__socket.sendall(data_byte)
            data = self.__socket.recv(1024)
            log.debug("data sent : %s", result)
        except Exception as e:
            log.error(e)

    @staticmethod
    def clear_results(self, runner_name=None):
        pass
