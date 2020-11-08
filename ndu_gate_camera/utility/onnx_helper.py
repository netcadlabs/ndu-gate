import onnxruntime as rt


def create_sess_tuple(onnx_fn):
    sess = rt.InferenceSession(onnx_fn)

    input_names = []
    for input in sess.get_inputs():
        input_names.append(input.name)

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
            input = inputs[i]
            input_item[name] = input
    else:
        input_item = {input_names[0]: inputs[0]}
    # sess.run(None, {input_name: image_data, "image_shape": img_size})
    return sess.run(output_names, input_item)


def parse_class_names(classes_fn):
    return [line.rstrip('\n') for line in open(classes_fn)]
