import os
import time

from ndu_gate_camera.utility import file_helper


def parse_class_names(classes_fn):
    return [line.rstrip('\n') for line in open(classes_fn, encoding='utf-8')]


_handler = None


def init(onnx_runner, log):
    global _handler
    if onnx_runner is None or onnx_runner == "onnxruntime":
        _handler = _OnnxruntimeHandler()
    elif onnx_runner == "openvino":
        _handler = _OpenvinoHandler()
    else:
        log.error("Bad onnx_runner: {}".format(onnx_runner))
        _handler = _OnnxruntimeHandler()


def get_sess_tuple(onnx_fn, output_names=None):
    return _handler.get_sess_tuple(onnx_fn, output_names)


def run(sess_tuple, inputs):
    return _handler.run(sess_tuple, inputs)


class _OnnxruntimeHandler:
    @staticmethod
    def get_sess_tuple(onnx_fn, output_names):
        import onnxruntime as rt
        start = time.time()
        sess = rt.InferenceSession(onnx_fn)
        elapsed = time.time() - start
        print("onnx model {} load time: {:.0f}sn".format(onnx_fn, elapsed))

        input_names = []
        for sess_input in sess.get_inputs():
            input_names.append(sess_input.name)

        if output_names is None:
            outputs = sess.get_outputs()
            output_names = []
            for output in outputs:
                output_names.append(output.name)

        return sess, input_names, output_names

    @staticmethod
    def run(sess_tuple, inputs):
        sess, input_names, output_names = sess_tuple

        if len(input_names) > 1:
            input_item = {}
            for i in range(len(inputs)):
                name = input_names[i]
                input_item[name] = inputs[i]
        else:
            input_item = {input_names[0]: inputs[0]}
        return sess.run(output_names, input_item)
        # res = sess.run(output_names, input_item)
        # return res


class _OpenvinoHandler:
    @staticmethod
    def get_sess_tuple(onnx_fn, output_names):
        from openvino.inference_engine import IECore, IENetwork

        ie = IECore()

        dir_name, name, extension = file_helper.get_file_name_extension(onnx_fn)
        xml_fn = file_helper.path_join(dir_name, name + ".xml")
        bin_fn = file_helper.path_join(dir_name, name + ".bin")
        if os.path.isfile(xml_fn) and os.path.isfile(bin_fn):
            # net = IENetwork(model=xml_fn, weights=bin_fn)
            net = IENetwork(xml_fn, bin_fn)
        else:
            net = ie.read_network(onnx_fn)

        # net = ie.read_network(onnx_fn)

        # sess = ie.load_network(network=net, device_name="CPU")
        sess = ie.load_network(net, "CPU")

        # exec_net = ie.load_network(network=net, device_name="CPU")
        # input_blob = next(iter(net.inputs))
        # outputs = net.outputs.keys()

        input_names = []
        for sess_input in sess.inputs:
            input_names.append(sess_input)

        if output_names is None:
            output_names = []
            for output in sess.outputs:
                output_names.append(output)

        # if not has_single_input:
        #     _input_item = {}
        #     for i in range(len(input_names)):
        #         name = input_names[i]
        #         _input_item[name] = None
        # else:
        #     _input_item = {input_names[0]: None}

        return sess, input_names, output_names

    @staticmethod
    def run(sess_tuple, inputs):
        sess, input_names, output_names = sess_tuple

        if len(input_names) > 1:
            input_item = {}
            for i in range(len(inputs)):
                name = input_names[i]
                input_item[name] = inputs[i]
        else:
            input_item = {input_names[0]: inputs[0]}

        pred = sess.infer(input_item)
        res = []
        for key, val in pred.items():
            if key in output_names:
                res.append(val)
        return res

        # pred = sess.infer(input_item)
        # return [pred["output"]]
