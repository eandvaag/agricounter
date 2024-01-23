import os
import shutil
from pathlib import Path
import glob
import logging

from io_utils import json_io



def save_model_weights(model, config, seq_num, epoch):

    logger = logging.getLogger(__name__)

    weights_dir = os.path.join(config["weights_dir"], str(seq_num))
    weights_path = os.path.join(weights_dir, "weights.h5")
    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)
    os.makedirs(weights_dir)
    Path(weights_path).touch(exist_ok=True)

    # Save all layers as trainable, then restore any layer's frozen state afterwards
    trainable_lookup = {}
    for layer in model.layers:
        trainable_lookup[layer.name] = layer.trainable
        layer.trainable = True
    model.save_weights(filepath=weights_path, save_format="h5") #, "epoch-{}.h5".format(epoch)), save_format="h5")
    for layer in model.layers:
        if not trainable_lookup[layer.name]:
            layer.trainable = False

    logger.info("Saved model weights.")


def get_most_recent_weights_path(weights_dir):
    #logger = logging.getLogger(__name__)
    most_recent_dir_path = sorted([f for f in glob.glob(os.path.join(weights_dir, "*")) if os.path.isdir(f)])[-1]
    #logger.info("Location of most recent weights: {}".format(most_recent_dir_path))
    weights_path = os.path.join(most_recent_dir_path, "weights.h5")
    return weights_path

def load_all_weights(model, config):

    logger = logging.getLogger(__name__)

    weights_path = get_most_recent_weights_path(config["weights_dir"])
    model.load_weights(weights_path, by_name=False)

    logger.info("Loaded all model weights.")



def load_select_weights(model, config):

    logger = logging.getLogger(__name__)

    layer_lookup_path = os.path.join(config["weights_dir"], "layer_lookup.json")
    layer_lookup = json_io.load_json(layer_lookup_path)

    load_layer_names = []
    frozen_layer_names = []
    load_weights_config = config["training"]["active"]["load_weights_config"]

    parts = ["backbone", "neck", "head"]

    for part in parts:
        if load_weights_config[part] in ["load_frozen", "load_trainable"]:
            load_layer_names.extend(layer_lookup[part])
        if load_weights_config[part] == "load_frozen":
            frozen_layer_names.extend(layer_lookup[part])

    max_layer_name_len = 0

    for layer in model.layers:
        if layer.name not in load_layer_names:
            layer._name = "x" + layer.name
        if len(layer.name) > max_layer_name_len:
            max_layer_name_len = len(layer.name)


    #logger.info("Now loading weights for the following layers: {}".format(load_layer_names))

    weights_path = get_most_recent_weights_path(config["weights_dir"])
    model.load_weights(weights_path, by_name=True)

    for layer in model.layers:
        if layer.name not in load_layer_names:
            layer._name = layer.name[1:]
        if layer.name in frozen_layer_names:
            layer.trainable = False

    for layer in model.layers:
        logger.info("Layer name: {0:{1}} | Loaded: {2:5} | Trainable: {3:5}".format(
            layer.name, max_layer_name_len, str(layer.name in load_layer_names), str(layer.trainable)))

    logger.info("Loaded select model weights.")