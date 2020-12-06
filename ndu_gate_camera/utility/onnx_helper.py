import time

import onnxruntime as rt

from ndu_gate_camera.utility import constants
from ndu_gate_camera.utility.geometry_helper import add_padding_rect, rects_intersect


def create_sess_tuple(onnx_fn):
    start = time.time()

    sess = rt.InferenceSession(onnx_fn)

    # # sess =  rt.InferenceSession(onnx_fn, sess_options= SessionOptions.MakeSessionOptionWithCudaProvider(gpuIndex));
    #
    # # sess_options = rt.SessionOptions()
    # # sess_options.enable_profiling = True
    # # sess = rt.InferenceSession(onnx_fn, sess_options=sess_options)

    # so = rt.SessionOptions()
    #
    # # so.intra_op_num_threads = 16
    # #
    # # so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    # # # so.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    #
    # # so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    # # so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    #
    # # so.enable_profiling = True
    #
    # so.log_verbosity_level = 2
    # so.log_severity_level = 0
    #
    #
    # sess = rt.InferenceSession(onnx_fn, sess_options=so)
    # # sess.set_providers(['CUDAExecutionProvider'])

    elapsed = time.time() - start
    # sil print(f"onnx model {onnx_fn} load time: {elapsed:.0f}sn")
    print("onnx model {} load time: {:.0f}sn".format(onnx_fn, elapsed))

    input_names = []
    for sess_input in sess.get_inputs():
        input_names.append(sess_input.name)

    outputs = sess.get_outputs()
    output_names = []
    for output in outputs:
        output_names.append(output.name)

    return sess, input_names, output_names


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


def parse_class_names(classes_fn):
    return [line.rstrip('\n') for line in open(classes_fn, encoding='utf-8')]
