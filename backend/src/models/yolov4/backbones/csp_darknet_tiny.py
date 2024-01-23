import tensorflow as tf

from models.yolov4.common import Conv2d, route_group


def build_backbone(config):

    backbone_config = config["arch"]["backbone_config"]
    backbone_type = backbone_config["backbone_type"]

    backbone = None

    if backbone_type == "csp_darknet53_tiny":
        backbone = CSPDarkNet53Tiny()


    if backbone is None:
        raise RuntimeError("Invalid backbone configuration: '{}'.".format(backbone_config))

    return backbone


class CSP_BlockTiny(tf.keras.layers.Layer):

    def __init__(self, num_filters_1, num_filters_2):
        super(CSP_BlockTiny, self).__init__()

        self.conv1 = Conv2d(filters=num_filters_1, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv2 = Conv2d(filters=num_filters_2, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv3 = Conv2d(filters=num_filters_2, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv4 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.max_pool = tf.keras.layers.MaxPool2D(2, 2, 'same')


    def call(self, inputs, training=False, **kwargs):

        x = self.conv1(inputs, training=training)
        route = x
        x = route_group(inputs, 2, 1)
        x = self.conv2(x, training=training)
        route_1 = x
        x = self.conv3(x, training=training)
        x = tf.concat([x, route_1], axis=-1)
        x = self.conv4(x, training=training)
        route_1 = x
        x = tf.concat([route, x], axis=-1)
        x = self.max_pool(x, training=training)
        return x, route_1




class CSPDarkNet53Tiny(tf.keras.layers.Layer):


    def __init__(self):
        super(CSPDarkNet53Tiny, self).__init__()

        self.conv = Conv2d(filters=32, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="leaky_relu")
        self.conv2 = Conv2d(filters=64, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="leaky_relu")

        self.csp_blocktiny1 = CSP_BlockTiny(64, 32)
        self.csp_blocktiny2 = CSP_BlockTiny(128, 64)
        self.csp_blocktiny3 = CSP_BlockTiny(256, 128)

        self.conv3 = Conv2d(filters=512, kernel_size=3, strides=1, activation="leaky_relu")


    def call(self, inputs, training=False, **kwargs):

        x = self.conv(inputs, training=training)
        x = self.conv2(x, training=training)
        #save = x

        x, _ = self.csp_blocktiny1(x, training=training)
        x, _ = self.csp_blocktiny2(x, training=training)
        x, route_1 = self.csp_blocktiny3(x, training=training)

        x = self.conv3(x, training=training)

        return route_1, x #, save
