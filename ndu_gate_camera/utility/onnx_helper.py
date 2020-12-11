# import time
# import onnxruntime as rt
# from threading import Lock
#
# locks_lock = Lock()
# locks = {}
# sess_tuples = {}
#
# max_count = 4  ####################koray
#
#
# def run(onnx_fn, inputs):
#     def get_lock(text):
#         global locks
#         if text not in locks:
#             with locks_lock:
#                 locks[text] = Lock()
#         return locks[text]
#
#     def create_sess_tuple(onnx_fn_):
#         start = time.time()
#         sess_ = rt.InferenceSession(onnx_fn_)
#         elapsed = time.time() - start
#         print("onnx model {} load time: {:.0f}sn".format(onnx_fn_, elapsed))
#
#         input_names_ = []
#         for sess_input in sess_.get_inputs():
#             input_names_.append(sess_input.name)
#
#         outputs = sess_.get_outputs()
#         output_names_ = []
#         for output in outputs:
#             output_names_.append(output.name)
#
#         return sess_, input_names_, output_names_
#
#     locker = get_lock(onnx_fn)
#     with locker:
#         global sess_tuples
#         if onnx_fn in sess_tuples:
#             tuples = sess_tuples[onnx_fn]
#             if len(tuples) < max_count:
#                 tuples.append(create_sess_tuple(onnx_fn))
#         else:
#             t = create_sess_tuple(onnx_fn)
#             tuples = [0, t]
#             sess_tuples[onnx_fn] = tuples
#
#     while tuples[0] >= max_count:
#         time.sleep(0.001)
#     with locker:
#         while tuples[0] >= max_count:
#             time.sleep(0.001)
#         tuples[0] = index = tuples[0] + 1
#         tu = tuples[index]
#     # tu = tuples[1]
#     try:
#         sess, input_names, output_names = tu
#         if len(input_names) > 1:
#             input_item = {}
#             for i in range(len(inputs)):
#                 name = input_names[i]
#                 input_item[name] = inputs[i]
#         else:
#             input_item = {input_names[0]: inputs[0]}
#
#         return sess.run(output_names, input_item)
#     finally:
#         # with locker:
#         tuples[0] -= 1
#
#
# def parse_class_names(classes_fn):
#     return [line.rstrip('\n') for line in open(classes_fn, encoding='utf-8')]


#
#
# # OK
# # ####################ok
# import time
# import onnxruntime as rt
# from threading import Lock
#
# locks_lock = Lock()
# locks = {}
# sess_tuples = {}
#
#
# def run(onnx_fn, inputs):
#     def get_lock(text):
#         global locks
#         with locks_lock:
#             if text not in locks:
#                 locks[text] = Lock()
#             return locks[text]
#
#     def create_sess_tuple(onnx_fn_):
#         start = time.time()
#         sess_ = rt.InferenceSession(onnx_fn_)
#         elapsed = time.time() - start
#         print("onnx model {} load time: {:.0f}sn".format(onnx_fn_, elapsed))
#
#         input_names_ = []
#         for sess_input in sess_.get_inputs():
#             input_names_.append(sess_input.name)
#
#         outputs = sess_.get_outputs()
#         output_names_ = []
#         for output in outputs:
#             output_names_.append(output.name)
#
#         return sess_, input_names_, output_names_
#
#     with get_lock(onnx_fn):
#         global sess_tuples
#         if onnx_fn in sess_tuples:
#             tu = sess_tuples[onnx_fn]
#         else:
#             tu = create_sess_tuple(onnx_fn)
#             sess_tuples[onnx_fn] = tu
#
#         sess, input_names, output_names = tu
#         if len(input_names) > 1:
#             input_item = {}
#             for i in range(len(inputs)):
#                 name = input_names[i]
#                 input_item[name] = inputs[i]
#         else:
#             input_item = {input_names[0]: inputs[0]}
#
#         return sess.run(output_names, input_item)
#
#
# def parse_class_names(classes_fn):
#     return [line.rstrip('\n') for line in open(classes_fn, encoding='utf-8')]


import time
import onnxruntime as rt

from ndu_gate_camera.utility import thread_helper

sess_tuples = {}


def get_sess_tuple(onnx_fn, max_engine_count=0):
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

        return sess_, input_names_, output_names_, onnx_fn_

    # max_engine_count = 0

    if max_engine_count < 1:
        return create_sess_tuple(onnx_fn)
    else:
        with thread_helper.get_lock(onnx_fn):
            if onnx_fn not in sess_tuples:
                tuples = [1]
                for i in range(max_engine_count):
                    tuples.append(create_sess_tuple(onnx_fn))
                sess_tuples[onnx_fn] = tuples
            else:
                tuples = sess_tuples[onnx_fn]
            index = tuples[0] + 1
            if index >= len(tuples):
                index = 1
            tuples[0] = index
            return tuples[index]


def run(sess_tuple, inputs):
    sess, input_names, output_names, onnx_fn = sess_tuple
    # print(sess_tuples[onnx_fn].index(sess_tuple))

    if len(input_names) > 1:
        input_item = {}
        for i in range(len(inputs)):
            name = input_names[i]
            input_item[name] = inputs[i]
    else:
        input_item = {input_names[0]: inputs[0]}
    return sess.run(output_names, input_item)


    # with thread_helper.get_lock(onnx_fn):
    #     if len(input_names) > 1:
    #         input_item = {}
    #         for i in range(len(inputs)):
    #             name = input_names[i]
    #             input_item[name] = inputs[i]
    #     else:
    #         input_item = {input_names[0]: inputs[0]}
    #     return sess.run(output_names, input_item)


def parse_class_names(classes_fn):
    return [line.rstrip('\n') for line in open(classes_fn, encoding='utf-8')]
