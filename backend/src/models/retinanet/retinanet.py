# RetinaNet implementation adapted from https://keras.io/examples/vision/retinanet/

import numpy as np
import tensorflow as tf

from models.retinanet.backbones import resnet as resnet_backbone
from models.retinanet.necks import resnet_fpn as resnet_fpn_neck


def get_backbone_class(config):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    if config.arch["backbone_config"]["backbone_type"] in resnets:
        return "resnet"
    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config.arch["backbone_config"]["backbone_type"]))


def build_backbone(config):

    backbone_class = get_backbone_class(config)

    if backbone_class == "resnet":
        return resnet_backbone.build_backbone(config)
    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config.arch["backbone_config"]["backbone_type"]))


def build_neck(config):

    neck_type = config.arch["neck_config"]["neck_type"]

    if neck_type == "resnet_fpn":
        return resnet_fpn_neck.build_neck(config)
    else:
        raise RuntimeError("Unsupported neck: '{}'".format(config.arch["neck_config"]["neck_type"]))



def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = tf.keras.Sequential()#[tf.keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(tf.keras.layers.ReLU())
    head.add(
        tf.keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head



class RetinaNet(tf.keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
    """

    def __init__(self, config):
        super(RetinaNet, self).__init__(name="RetinaNet")
        self.backbone = build_backbone(config)
        self.neck = build_neck(config)
        self.num_classes = config.arch["num_classes"]

        # prior_probability is designed to combat the effects of class
        # instability in early training (Since the foreground class is
        # rare, the value of p estimated by the model at the start of training 
        # should be small.) Every anchor will be labeled as foreground with 
        #confidence of approximately 0.01 at the start of training.
        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * self.num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")
        self.backbone_class = get_backbone_class(config)


        self.imagenet_pretrained = "imagenet_pretrained" in config.arch["backbone_config"] and \
                                    config.arch["backbone_config"]["imagenet_pretrained"]
        self.backbone_class = get_backbone_class(config)

        self.backbone._name = "retinanet_backbone"
        self.neck._name = "retinanet_neck"
        self.cls_head._name = "retinanet_cls_head"
        self.box_head._name = "retinanet_box_head"

    def call(self, images, training=None):
        images = self.possible_preprocess(images)
        x = self.backbone(images, training=training)
        features = self.neck(x, training=training)
        #features = self.fpn(image, training=training)
        N = tf.shape(images)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature, training=training), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature, training=training), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)

        return tf.concat([box_outputs, cls_outputs], axis=-1)


    def get_layer_lookup(self):
        layer_lookup = {
                "backbone": [self.backbone.name],
                "neck": [self.neck.name],
                "head": [self.cls_head.name, self.box_head.name]
        }
        return layer_lookup


    def possible_preprocess(self, images):
        if self.imagenet_pretrained and self.backbone_class == "resnet":
            #print("applying preprocessing")
            #print("images before", images)
            images = tf.keras.applications.resnet.preprocess_input(images)
            #print("images after", images)

        return images