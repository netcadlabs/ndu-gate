# Already trained model available @
# https://github.com/tensorflow/models/tree/master/research/object_detection
# was used as a part of this code.
import errno
import glob, os, tarfile, urllib
import tensorflow as tf
from os import path

from ndu_gate_camera.detectors.model import label_map_util
from ndu_gate_camera.utility.constants import NDU_GATE_MODEL_FOLDER


def set_model(model_name, label_name):
    model_found = 0
    labels_found = 0

    for file in glob.glob("*"):
        if file == model_name:
            model_found = 1

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_model = path.dirname(path.abspath(__file__)) + ("/../../data/vision/" + model_name).replace('/', os.path.sep)
    path_to_model = path.abspath(path_to_model)
    if path.isfile(path_to_model):
        model_found = 1
    else:
        path_to_model = (NDU_GATE_MODEL_FOLDER + ("/vision/" + model_name)).replace('/', os.path.sep)
        if path.isfile(path_to_model):
            model_found = 1

    path_to_labels = os.path.dirname(os.path.abspath(__file__)) + ("/../../data/vision/" + label_name).replace('/', os.path.sep)
    path_to_labels = path.abspath(path_to_labels)
    if path.isfile(path_to_labels):
        labels_found = 1
    else:
        path_to_labels = (NDU_GATE_MODEL_FOLDER + ("/vision/" + label_name)).replace('/', os.path.sep)
        if path.isfile(path_to_labels):
            labels_found = 1

    if model_found == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_model)
    if labels_found == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_to_labels)

    num_classes = 90

    # Download Model if it has not been downloaded yet
    if model_found == 0:
        # What model to download.
        model_name = model_name
        model_file = model_name + ".tar.gz"
        download_base = "http://download.tensorflow.org/models/object_detection/"
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if "frozen_inference_graph.pb" in file_name:
                tar_file.extract(file, os.getcwd())

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_model, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I 		use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index
