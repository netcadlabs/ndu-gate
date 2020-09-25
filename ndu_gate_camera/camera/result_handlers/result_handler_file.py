from ndu_gate_camera.api.result_handler import ResultHandler


class ResultHandlerFile(ResultHandler):
    def save_result(self):
        pass

    def __init__(self, folder):
        self.workingPath = folder
        pass

    @staticmethod
    def save_result(self, result, runner_name=None):
        """

        :param self:
        :param result: {"key" : "telem-key", "value": 1, "ts" : timestamp }
        :param runner_name:
        :return:
        """
        with open(self.workingPath + 'serviceTelemetry.txt', 'r+') as f:
            f.seek(0)
            for i in range(len(result)):
                f.write(str(result[i]) + '\n')
            f.truncate()
            f.close()

        with open(self.workingPath + 'serviceTelemetry.txt', 'r') as myfile:
            personcount = myfile.read()
            print("person count: %s ", str(result))

    @staticmethod
    def clear_results(self, runner_name=None):
        pass
