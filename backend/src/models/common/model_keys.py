from abc import ABC
import os
import copy
import numpy as np

# from models.common import class_map_utils

# class ModelConfig(ABC):

#     def __init__(self, model_dir):

#         self.model_dir = model_dir
#         self.weights_dir = os.path.join(model_dir, "weights")
#         self.loss_records_dir = os.path.join(model_dir, "loss_records")
#         self.predictions_dir = os.path.join(model_dir, "predictions")

#         self.arch = None
#         self.training = None
#         self.inference = None



#     def add_arch_config(self, arch_config):

#         self.arch = copy.deepcopy(arch_config)

#     def add_training_config(self, training_config):

#         self.training = copy.deepcopy(training_config)

#         class_map = self.arch["class_map"] if "class_map" in self.arch else None
#         class_map_utils.create_and_save_class_map_data(training_config, class_map=class_map)
#         class_map_data = class_map_utils.load_class_map_data(training_config["model_uuid"])

#         if self.arch is None:
#             raise RuntimeError("Architecture config must be added before training config.")
#         self.arch["class_map"] = class_map_data["class_map"]
#         self.arch["reverse_class_map"] = class_map_data["reverse_class_map"]
#         self.arch["num_classes"] = class_map_data["num_classes"]


#     def add_inference_config(self, inference_config):

#         self.inference = copy.deepcopy(inference_config)
        
#         class_map_data = class_map_utils.load_class_map_data(inference_config["model_uuid"])

#         if self.arch is None:
#             raise RuntimeError("Architecture config must be added before inference config.")
#         self.arch["class_map"] = class_map_data["class_map"]
#         self.arch["reverse_class_map"] = class_map_data["reverse_class_map"]
#         self.arch["num_classes"] = class_map_data["num_classes"]


# class RetinaNetConfig(ModelConfig):

#     def add_arch_config(self, arch_config):
#         super().add_arch_config(arch_config)

#         self.arch["gamma"] = 2.0
#         self.arch["alpha"] = 0.25
#         self.arch["delta"] = 1.0

#     #def add_training_config(self, training_config):



# class CenterNetConfig(ModelConfig):


#     def add_arch_config(self, arch_config):
#         super().add_arch_config(arch_config)

#         input_img_shape = {
#             "resnet18": (384, 384, 3),
#             "resnet34": (384, 384, 3),
#             "resnet50": (384, 384, 3),
#             "resnet101": (384, 384, 3), 
#             "resnet152": (384, 384, 3),

#             "D0": (512, 512, 3), "D1": (640, 640, 3), "D2": (768, 768, 3),
#             "D3": (896, 896, 3), "D4": (1024, 1024, 3), "D5": (1280, 1280, 3),
#             "D6": (1408, 1408, 3), "D7": (1536, 1536, 3)
#         }

#         downsampling_ratio = {
#             "resnet18": 4, "resnet34": 4, "resnet50": 4, "resnet101": 4, "resnet152": 4,
#             "D0": 8, "D1": 8, "D2": 8, "D3": 8, "D4": 8, "D5": 8, "D6": 8, "D7": 8
#         }

#         # efficientdet
#         #width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
#         #depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
#         #dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}
        
#         # bifpn channels
#         bifpn_width = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
        
#         # bifpn layers
#         bifpn_depth = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}

#         head_conv = {
#                     "no_conv_layer": 0, 
#                     "resnet18": 64, "resnet34": 64, "resnet50": 64, "resnet101": 64, "resnet152": 64,
#                     "D0": bifpn_width["D0"], "D1": bifpn_width["D1"], "D2": bifpn_width["D2"], "D3": bifpn_width["D3"],
#                     "D4": bifpn_width["D4"], "D5": bifpn_width["D5"], "D6": bifpn_width["D6"], "D7": bifpn_width["D7"]
#         }

#         backbone_type = arch_config["backbone_config"]["backbone_type"]

#         self.arch["backbone_config"] = arch_config["backbone_config"]
#         self.arch["max_detections"] = arch_config["max_detections"]


#         #self.arch["input_img_shape"] = input_img_shape[backbone_type]
#         self.arch["downsampling_ratio"] = downsampling_ratio[backbone_type]

#         self.arch["head_conv"] = head_conv[backbone_type]

#         self.arch["bifpn_width"] = bifpn_width[backbone_type] if backbone_type in bifpn_width else None
#         self.arch["bifpn_depth"] = bifpn_depth[backbone_type] if backbone_type in bifpn_depth else None
#         #self.dropout_rate = dropout_rate[self.backbone_type] if self.backbone_type in dropout_rate else None
#         #self.width_coefficient = width_coefficient[self.backbone_type] if self.backbone_type in width_coefficient else None
#         #self.depth_coefficient = depth_coefficient[self.backbone_type] if self.backbone_type in depth_coefficient else None

#         # weighting of loss components
#         self.arch["heatmap_loss_weight"] = 1.0
#         self.arch["size_loss_weight"] = 0.1
#         self.arch["offset_loss_weight"] = 1.0



# class YOLOv4Config(ModelConfig):


#     def add_arch_config(self, arch_config):
#         super().add_arch_config(arch_config)

#         self.arch["strides"] = np.array([8, 16, 32])
#         self.arch["xy_scales"] = [1.2, 1.1, 1.05]


#         # "We just sort of chose 9 clusters and 3 scales arbitrarily and then divided
#         # up the clusters evenly across scales. On the COCO dataset the 9 clusters were:
#         # (10x13), (16x30), (33x23), (30x61), (62x45), (59x119), (116x90) (156x198), (373x326).""


#         # the bounding box priors (a.k.a. anchors) defined here are expressed as their relative 
#         # size in the feature map they have been assigned to. We divide each anchor's width and
#         # height (in pixel coordinates) by the stride of the appropriate feature map.
#         # examples:
#         # stride 8, aspect ratio (10x13) --> (10 / 8, 13 / 8) --> (1.25, 1.625)
#         # stride 16, aspect ratio (30x61) --> (30 / 16, 61 / 16) --> (1.875, 3.8125)

#         # higher up the pyramid --> larger stride --> predict larger objects


#         self.arch["anchors_per_scale"] = 3

#         self.arch["max_detections_per_scale"] = self.arch["max_detections"]


#         self.arch["iou_loss_thresh"] = 0.5


#         if self.arch["model_type"] == "yolov4":
#             self.arch["num_scales"] = 3
#             self.arch["strides"] = np.array([8, 16, 32])
#             self.arch["xy_scales"] = [1.2, 1.1, 1.05]
#             self.arch["anchors"] = np.array([
#                                     [[1.25,1.625], [2.0,3.75], [4.125,2.875]], 
#                                     [[1.875,3.8125], [3.875,2.8125], [3.6875,7.4375]], 
#                                     [[3.625,2.8125], [4.875,6.1875], [11.65625,10.1875]]
#                                 ], dtype=np.float32)

#         elif self.arch["model_type"] == "yolov4_tiny":
#             self.arch["num_scales"] = 2
#             self.arch["strides"] = np.array([16, 32])
#             self.arch["xy_scales"] = [1.1, 1.05]
#             self.arch["anchors"] = np.array([
#                                     [[1.875,3.8125], [3.875,2.8125], [3.6875,7.4375]], 
#                                     [[3.625,2.8125], [4.875,6.1875], [11.65625,10.1875]]
#                                 ], dtype=np.float32)




def add_retinanet_keys(model_config):
    pass

def add_centernet_keys(model_config):
    pass

def add_efficientdet_keys(model_config):
    pass

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



def add_dir_shortcuts(model_config):
    model_config["model_dir"] = os.path.join("usr", "data", "models", model_config["model_uuid"])
    model_config["weights_dir"] = os.path.join(model_config["model_dir"], "weights")
    model_config["loss_records_dir"] = os.path.join(model_config["model_dir"], "loss_records")
    model_config["predictions_dir"] = os.path.join(model_config["model_dir"], "predictions")

def add_general_keys(model_config):
    
    model_config["arch"]["reverse_class_map"] = {v: k for k, v in model_config["arch"]["class_map"].items()}
    model_config["arch"]["num_classes"] = len(model_config["arch"]["class_map"].keys())

def add_specialized_keys(model_config):

    model_type = model_config["arch"]["model_type"]

    if model_type == "retinanet":
        add_func = add_retinanet_keys
    elif model_type == "centernet":
        add_func = add_centernet_keys
    elif model_type == "efficientdet":
        add_func = add_efficientdet_keys
    elif model_type == "yolov4" or model_type == "yolov4_tiny":
        add_func = add_yolov4_keys
    else:
        raise RuntimeError("Unknown model type: '{}'.".format(model_type))


    add_func(model_config)