# YOLOv4 implementation adapted from:
# https://github.com/sicara/tf2-yolov4/tree/85bc3bfc85e719297d221a2f4835213ecfdec65a
# https://github.com/youchangxin/YOLOv4_tensorflow2/tree/0dd287975f85badd51c8dd01d746dab88dc982e8

import tensorflow as tf

from models.yolov4.backbones import csp_darknet as csp_darknet_backbone, \
                                    csp_darknet_tiny as csp_darknet_tiny_backbone
from models.yolov4.necks import spp_pan as spp_pan_neck, \
                                yolov4_tiny_deconv as yolov4_tiny_deconv_neck

from models.yolov4.common import Conv2d


def build_backbone(config):


    if config["arch"]["backbone_config"]["backbone_type"] == "csp_darknet53":
        return csp_darknet_backbone.build_backbone(config)
    elif config["arch"]["backbone_config"]["backbone_type"] == "csp_darknet53_tiny":
        return csp_darknet_tiny_backbone.build_backbone(config)
    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config["arch"]["backbone_config"]["backbone_type"]))


def build_neck(config):

    neck_type = config["arch"]["neck_config"]["neck_type"]
    if neck_type == "spp_pan":
        return spp_pan_neck.build_neck(config)
    elif neck_type == "yolov4_tiny_deconv":
        return yolov4_tiny_deconv_neck.build_neck(config)
    else:
        raise RuntimeError("Unsupported neck: '{}'".format(config["arch"]["neck_config"]["neck_type"]))

def build_head(num_filters_1, num_filters_2):
    return YOLOv3Head(num_filters_1, num_filters_2)


class YOLOv3Head(tf.keras.layers.Layer):
    def __init__(self, num_filters_1, num_filters_2):
        super(YOLOv3Head, self).__init__()
        self.conv1 = Conv2d(filters=num_filters_1, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv2 = Conv2d(filters=num_filters_2, kernel_size=1, strides=1, activation="linear", use_batch_norm=False)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x



class YOLOv4(tf.keras.Model):

    def __init__(self, config):
        super(YOLOv4, self).__init__(name="YOLOv4")

        out_shape = config["arch"]["anchors_per_scale"] * (5 + config["arch"]["num_classes"])

        self.backbone = build_backbone(config)
        self.neck = build_neck(config)
        self.head_s = build_head(256, out_shape)
        self.head_m = build_head(512, out_shape)
        self.head_l = build_head(1024, out_shape)

        self.backbone._name = "yolov4_backbone"
        self.neck._name = "yolov4_neck"
        self.head_s._name = "yolov4_head_s"
        self.head_m._name = "yolov4_head_m"
        self.head_l._name = "yolov4_head_l"

    @tf.function
    def call(self, images, training=None):

        x = self.backbone(images, training=training)
        route_s, route_m, route_l = self.neck(x, training=training)
        out_s = self.head_s(route_s, training=training)
        out_m = self.head_m(route_m, training=training)
        out_l = self.head_l(route_l, training=training)

        return out_s, out_m, out_l

    def get_layer_lookup(self):
        layer_lookup = {
                "backbone": [self.backbone.name],
                "neck": [self.neck.name],
                "head": [self.head_s.name, self.head_m.name, self.head_l.name]
        }
        return layer_lookup


class YOLOv4Tiny(tf.keras.Model):

    def __init__(self, config):
        super(YOLOv4Tiny, self).__init__(name="YOLOv4Tiny")

        out_shape = config["arch"]["anchors_per_scale"] * (5 + config["arch"]["num_classes"])

        self.backbone = build_backbone(config)
        self.neck = build_neck(config)

        self.head_m = build_head(256, out_shape)
        self.head_l = build_head(512, out_shape)

        self.backbone._name = "yolov4_tiny_backbone"
        self.neck._name = "yolov4_tiny_neck"
        self.head_m._name = "yolov4_tiny_head_m"
        self.head_l._name = "yolov4_tiny_head_l"

    def call(self, images, training=None):

        x = self.backbone(images, training=training)
        route_medium, route_large = self.neck(x, training=training)

        out_m = self.head_m(route_medium, training=training)
        out_l = self.head_l(route_large, training=training)

        return [out_m, out_l]

    def get_layer_lookup(self):
        layer_lookup = {
            "backbone": [self.backbone.name],
            "neck": [self.neck.name],
            "head": [self.head_m.name, self.head_l.name]
        }
        return layer_lookup



class YOLOv4TinyBackbone(tf.keras.Model):

    def __init__(self, config, max_pool):
        super(YOLOv4TinyBackbone, self).__init__(name="YOLOv4TinyBackbone")

        out_shape = config["arch"]["anchors_per_scale"] * (5 + config["arch"]["num_classes"])

        self.backbone = build_backbone(config)
        self.neck = build_neck(config)

        self.head_m = build_head(256, out_shape)
        self.head_l = build_head(512, out_shape)

        self.backbone._name = "yolov4_tiny_backbone"
        self.neck._name = "yolov4_tiny_neck"
        self.head_m._name = "yolov4_tiny_head_m"
        self.head_l._name = "yolov4_tiny_head_l"

        self.apply_max_pool = max_pool
            
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        #self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=4)


    def call(self, images, training=None):
        x = self.backbone(images, training=training)
        #save = x[2]
        #x = x[:2]
        if self.apply_max_pool:
            res = self.max_pool(x[1])
            #res = self.pool1(x[1])
            #res = self.pool2(res)
        else:
            #res = save #x[1]
            res = x[1]
            #tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=)

        route_medium, route_large = self.neck(x, training=training)

        out_m = self.head_m(route_medium, training=training)
        out_l = self.head_l(route_large, training=training)


        
        return res #x[1]

    def get_layer_lookup(self):
        layer_lookup = {
            "backbone": [self.backbone.name],
            "neck": [self.neck.name],
            "head": [self.head_m.name, self.head_l.name]
        }
        return layer_lookup
