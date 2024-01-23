import tensorflow as tf

from models.yolov4.common import Conv2d

def build_neck(config):

    neck_config = config["arch"]["neck_config"]
    neck_type = neck_config["neck_type"]

    neck = None

    if neck_type == "yolov4_tiny_deconv":
        neck = YOLOv4TinyDeconv()

    if neck is None:
        raise RuntimeError("Invalid neck configuration: '{}'.".format(neck_config))

    return neck


class YOLOv4TinyDeconv(tf.keras.layers.Layer):

    def __init__(self):
        super(YOLOv4TinyDeconv, self).__init__()
        self.conv = Conv2d(filters=256, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv2 = Conv2d(filters=128, kernel_size=1, strides=1, activation="leaky_relu")
        self.upsample = tf.keras.layers.UpSampling2D()


    def call(self, inputs, training=False, **kwargs):

        route_1, x = inputs

        route_large = self.conv(x, training=training)

        x = self.conv2(route_large, training=training)
        x = self.upsample(x)
        route_medium = tf.concat([x, route_1], axis=-1)

        return [route_medium, route_large]







