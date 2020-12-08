import time
import onnxruntime as rt
from threading import Lock

locks_lock = Lock()
locks = {}
sess_tuples = {}


def run(onnx_fn, inputs):
    def get_lock(fn):
        global locks
        with locks_lock:
            if fn not in locks:
                locks[fn] = Lock()
            return locks[fn]

    def create_sess_tuple(onnx_fn_):
        start = time.time()
        sess_ = rt.InferenceSession(onnx_fn_)
        elapsed = time.time() - start
        print("onnx model {} load time: {:.0f}sn".format(onnx_fn_, elapsed))

        input_names_ = []
        for sess_input in sess_.get_inputs():
            input_names_.append(sess_input.name)

        outputs = sess_.get_outputs()
        output_names_ = []
        for output in outputs:
            output_names_.append(output.name)

        return sess_, input_names_, output_names_

    with get_lock(onnx_fn):
        global sess_tuples
        if onnx_fn in sess_tuples:
            tu = sess_tuples[onnx_fn]
        else:
            tu = create_sess_tuple(onnx_fn)
            sess_tuples[onnx_fn] = tu

        sess, input_names, output_names = tu
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
