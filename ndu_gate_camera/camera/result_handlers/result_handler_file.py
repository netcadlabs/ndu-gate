from ndu_gate_camera.api.result_handler import ResultHandler, log
from os import path


class ResultHandlerFile(ResultHandler):
    def __init__(self, folder):
        self.__working_path = folder
        if not path.isdir(self.__working_path):
            log.warning("This path is not a folder %s", self.__working_path)
            self.__working_path = "/etc/ndu_gate/"

        self.__telemetry_file = self.__working_path + 'serviceTelemetry.txt'

        self.__working_path = path.expanduser("~") + path.sep + 'serviceTelemetry.txt'

        if not path.isfile(self.__telemetry_file):
            try:
                file = open(self.__telemetry_file, 'w+')
                file.close()
            except Exception as e:
                log.error(e)
                log.warning("Can not create telemetry file %s", self.__telemetry_file)

        log.info("ResultHandlerFile %s", self.__telemetry_file)

    def save_result(self, result, runner_name=None, data_type='telem'):
        """
        Verilen ölçümleri serviceTelemetry.txt dosyasına yazar.
        :param self:
        :param result: {"key" : "telem-key", "value": 1, "ts" : timestamp }
        :param runner_name:
        :return:
        """
        with open(self.__telemetry_file, 'r+') as f:
            f.seek(0)
            for i in range(len(result)):
                f.write(str(result[i]) + '\n')
            f.truncate()
            f.close()

    def clear_results(self, runner_name=None):
        pass
