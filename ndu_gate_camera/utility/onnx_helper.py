import time

import onnxruntime as rt


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
    print(f"onnx model {onnx_fn} load time: {elapsed:.0f}sn")

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

    # sess.run(None, {input_name: image_data, "image_shape": img_size})
    return sess.run(output_names, input_item)

    # start = time.time()
    # res = sess.run(output_names, input_item)
    # elapsed = time.time() - start
    # print(f"sess.run time: {elapsed:.0f}sn")
    # return res


def parse_class_names(classes_fn):
    return [line.rstrip('\n') for line in open(classes_fn, encoding='utf-8')]
