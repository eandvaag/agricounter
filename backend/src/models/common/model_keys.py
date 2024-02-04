import numpy as np


def add_yolov4_keys(model_config):

    arch_config = model_config["arch"]

    arch_config["anchors_per_scale"] = 3

    arch_config["max_detections_per_scale"] = arch_config["max_detections"]


    arch_config["iou_loss_thresh"] = 0.5


    if arch_config["model_type"] == "yolov4":
        arch_config["num_scales"] = 3
        arch_config["strides"] = np.array([8, 16, 32])
        arch_config["xy_scales"] = [1.2, 1.1, 1.05]
        arch_config["anchors"] = np.array([
                                [[1.25,1.625], [2.0,3.75], [4.125,2.875]], 
                                [[1.875,3.8125], [3.875,2.8125], [3.6875,7.4375]], 
                                [[3.625,2.8125], [4.875,6.1875], [11.65625,10.1875]]
                            ], dtype=np.float32)

    elif arch_config["model_type"] == "yolov4_tiny":
        arch_config["num_scales"] = 2
        arch_config["strides"] = np.array([16, 32])
        arch_config["xy_scales"] = [1.1, 1.05]
        arch_config["anchors"] = np.array([
                                [[1.875,3.8125], [3.875,2.8125], [3.6875,7.4375]], 
                                [[3.625,2.8125], [4.875,6.1875], [11.65625,10.1875]]
                            ], dtype=np.float32)



def add_general_keys(model_config):
    
    model_config["arch"]["reverse_class_map"] = {v: k for k, v in model_config["arch"]["class_map"].items()}
    model_config["arch"]["num_classes"] = len(model_config["arch"]["class_map"].keys())

def add_specialized_keys(model_config):

    model_type = model_config["arch"]["model_type"]
    
    if model_type == "yolov4" or model_type == "yolov4_tiny":
        add_func = add_yolov4_keys
    else:
        raise RuntimeError("Unknown model type: '{}'.".format(model_type))


    add_func(model_config)