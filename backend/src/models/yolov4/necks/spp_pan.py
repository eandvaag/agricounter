import tensorflow as tf

from models.yolov4.common import Conv2d


def build_neck(config):

    neck_config = config["arch"]["neck_config"]
    neck_type = neck_config["neck_type"]

    neck = None

    if neck_type == "spp_pan":
        neck = SPP_PAN()


    if neck is None:
        raise RuntimeError("Invalid neck configuration: '{}'.".format(neck_config))

    return neck










class SPP_PAN(tf.keras.layers.Layer):

    def __init__(self):
        super(SPP_PAN, self).__init__()
        self.spp = SPP()
        self.pan = PANet()

    def call(self, inputs, training=False, **kwargs):
        route_small, route_medium, route_large = inputs

        route_large = self.spp(route_large, training=training)
        x = self.pan([route_small, route_medium, route_large], training=training)
        return x



class PANet(tf.keras.layers.Layer):
    """
        PANet is a path-aggregation network. YOLOv3 used the Feature Pyramid Network
        as its path-aggregation network. YOLOv4 uses PANet. Like FPN, PANet has a bottom-up 
        and top-down path. To this it adds short-cut connections and an additional 
        "augmented bottom-up structure".
    """

    def __init__(self):

        super(PANet, self).__init__()

        self.upper_m = UpperConcatenate(256, 512)
        self.upper_s = UpperConcatenate(128, 256)

        self.merge_m = Merge(256, 512)
        self.merge_l = Merge(512, 1024)



    def call(self, inputs, training=False, **kwargs):
        route_small, route_medium, route_large = inputs

        route_medium = self.upper_m(route_medium, route_large, training=training)
        route_small = self.upper_s(route_small, route_medium, training=training)

        route_medium = self.merge_m(route_small, route_medium, training=training)
        route_large = self.merge_l(route_medium, route_large, training=training)

        return route_small, route_medium, route_large


class Merge(tf.keras.layers.Layer):

    def __init__(self, num_filters_1, num_filters_2):
        super(Merge, self).__init__()
        self.conv = Conv2d(filters=num_filters_1, kernel_size=3, strides=2, activation="leaky_relu")
        self.conv1 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv2 = Conv2d(filters=num_filters_2, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv3 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv4 = Conv2d(filters=num_filters_2, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv5 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")

    def call(self, input_1, input_2, training=False, **kwargs):
        x1 = self.conv(input_1, training=training)
        x = tf.keras.layers.Concatenate()([x1, input_2])
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)

        return x






class UpperConcatenate(tf.keras.layers.Layer):

    def __init__(self, num_filters_1, num_filters_2):

        super(UpperConcatenate, self).__init__()
        self.conv1 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv2 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.upsample = tf.keras.layers.UpSampling2D()
        self.conv3 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv4 = Conv2d(filters=num_filters_2, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv5 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv6 = Conv2d(filters=num_filters_2, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv7 = Conv2d(filters=num_filters_1, kernel_size=1, strides=1, activation="leaky_relu")

    def call(self, input_1, input_2, training=False, **kwargs):
        x1 = self.conv1(input_1, training=training)
        x2 = self.conv2(input_2, training=training)
        x2 = self.upsample(x2)
        x = tf.keras.layers.Concatenate()([x1, x2])
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)

        return x




class SPP(tf.keras.layers.Layer):
    """
        The original SPP by He et al. was designed to allow CNNs to handle arbitrary
        input sizes by transforming the output of the convolutional layers into fixed
        length feature vectors that could then be passed to the fully-connected layers.
        It did this by dividing the convolutional feature maps into a fixed number of 
        areas (the size of the area would change proportionally with the size of the
        feature map, but the number of areas was fixed).

        However, YOLOv4 is a fully-convolutional model, and so does not need SPP to
        produce a fixed-length output. It uses a modified version of SPP that produces a 
        variable-size output. The YOLOv4 paper justifies using SPP by saying it 
        "enhances/increases the receptive field of the backbone features" (because of
        the max-pooling applied with relatively large kernel sizes (1x1, 5x5, 9x9, 13x13)).
        YOLOv3 was the first YOLO model to use the SPP module. 

    """

    def __init__(self):

        super(SPP, self).__init__()
        self.conv1 = Conv2d(filters=512, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv2 = Conv2d(filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv3 = Conv2d(filters=512, kernel_size=1, strides=1, activation="leaky_relu")

        self.maxpool_5 = tf.keras.layers.MaxPool2D(pool_size=(5, 5), strides=1, padding="same")
        self.maxpool_9 = tf.keras.layers.MaxPool2D(pool_size=(9, 9), strides=1, padding="same")
        self.maxpool_13 = tf.keras.layers.MaxPool2D(pool_size=(13, 13), strides=1, padding="same")

        self.conv4 = Conv2d(filters=512, kernel_size=1, strides=1, activation="leaky_relu")
        self.conv5 = Conv2d(filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
        self.conv6 = Conv2d(filters=512, kernel_size=1, strides=1, activation="leaky_relu")


    def call(self, inputs, training=False, **kwargs):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)

        x1 = self.maxpool_5(x)
        x2 = self.maxpool_9(x)
        x3 = self.maxpool_13(x)
        x = tf.keras.layers.concatenate([x1, x2, x3, x], axis=-1)

        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.conv6(x, training=training)

        return x


