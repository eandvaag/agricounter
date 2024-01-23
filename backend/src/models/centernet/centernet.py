# CenterNet implementation adapted from https://github.com/calmisential/CenterNet_TensorFlow2/tree/a30c9b4243c7c5f45a0cf47df655c171dcfef256

import tensorflow as tf

from models.centernet.backbones import resnet as resnet_backbone, \
                                       efficientnet as efficientnet_backbone
from models.centernet.necks import resnet_deconv as resnet_deconv_neck, \
                                   efficientdet as efficientdet_neck

# def get_base_network(config):

#     resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
#     efficient_dets = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

#     if config.arch["backbone_config"]["backbone_type"] in resnets:
#         return resnet.ModifiedResNet(config)
#     elif config.arch["backbone_config"]["backbone_type"] in efficient_dets:
#         return efficientdet.ModifiedEfficientDet(config)
#     else:
#         raise RuntimeError("Unsupported backbone: '{}'".format(config.arch["backbone_config"]["backbone_type"]))





def get_backbone_class(config):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    efficientnets = ["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

    if config.arch["backbone_config"]["backbone_type"] in resnets:
        return "resnet"
    elif config.arch["backbone_config"]["backbone_type"] in efficientnets:
        return "efficientnet"
    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config.arch["backbone_config"]["backbone_type"]))


def build_backbone(config):

    backbone_class = get_backbone_class(config)

    if backbone_class == "resnet":
        #return resnet.ModifiedResNet(config)
        return resnet_backbone.build_backbone(config)
    elif backbone_class == "efficientnet":
        #return efficientdet.ModifiedEfficientDet(config)
        return efficientnet_backbone.build_backbone(config)
    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config.arch["backbone_config"]["backbone_type"]))



def build_neck(config):

    neck_type = config.arch["neck_config"]["neck_type"]

    if neck_type == "resnet_deconv":
        #return resnet.ModifiedResNet(config)
        return resnet_deconv_neck.build_neck(config)
    elif neck_type == "efficientnet":
        #return efficientdet.ModifiedEfficientDet(config)
        return efficientdet_neck.build_neck(config)
    else:
        raise RuntimeError("Unsupported neck: '{}'".format(config.arch["neck_config"]["neck_type"]))


def build_head(head_conv, output_filters, name):

    head = tf.keras.Sequential(name=name)

    # additional layers for the network head -- this is not referred to in the original 'Objects as Points' paper.
    if head_conv > 0:
        head.add(tf.keras.layers.Conv2D(filters=head_conv, kernel_size=(3, 3), strides=1, padding="same"))
        head.add(tf.keras.layers.ReLU())
    
    head.add(tf.keras.layers.Conv2D(filters=output_filters, kernel_size=(1, 1), strides=1, padding="same"))

    return head



class CenterNet(tf.keras.Model):

    def __init__(self, config):
        super(CenterNet, self).__init__(name="CenterNet")
        self.backbone = build_backbone(config)
        self.neck = build_neck(config)
        #self.base_network = get_base_network(config)
        self.heatmap_head = build_head(config.arch["head_conv"], config.arch["num_classes"], name="heatmap_head")
        self.reg_head = build_head(config.arch["head_conv"], 2, name="regression_head")
        self.wh_head = build_head(config.arch["head_conv"], 2, name="size_head")

        self.imagenet_pretrained = "imagenet_pretrained" in config.arch["backbone_config"] and \
                                    config.arch["backbone_config"]["imagenet_pretrained"]
        self.backbone_class = get_backbone_class(config)

        self.backbone._name = "centernet_backbone"
        self.neck._name = "centernet_neck"
        self.heatmap_head._name = "centernet_heatmap_head"
        self.reg_head._name = "centernet_reg_head"
        self.wh_head._name = "centernet_wh_head"


    def call(self, images, training=None):
        images = self.possible_preprocess(images)
        x = self.backbone(images, training=training)
        x = self.neck(x, training=training)
        #features = self.base_network(image, training=training)
        heatmap_outputs = self.heatmap_head(x, training=training)
        reg_outputs = self.reg_head(x, training=training)
        wh_outputs = self.wh_head(x, training=training)
        
        return tf.concat(values=[heatmap_outputs, reg_outputs, wh_outputs], axis=-1)


    def get_layer_lookup(self):
        layer_lookup = {
                "backbone": [self.backbone.name],
                "neck": [self.neck.name],
                "head": [self.heatmap_head.name, self.reg_head.name, self.wh_head.name]
        }
        return layer_lookup



    def possible_preprocess(self, images):
        if self.imagenet_pretrained and self.backbone_class == "resnet":
            images = tf.keras.applications.resnet.preprocess_input(images)
        elif self.imagenet_pretrained and self.backbone_class == "efficientnet":
            images = tf.keras.applications.efficientnet.preprocess_input(images)

        return images