import tensorflow as tf

from models.yolov4.common import Conv2d

def build_backbone(config):

    backbone_config = config["arch"]["backbone_config"]
    backbone_type = backbone_config["backbone_type"]

    backbone = None

    if backbone_type == "csp_darknet53":
        backbone = CSPDarkNet53()


    if backbone is None:
        raise RuntimeError("Invalid backbone configuration: '{}'.".format(backbone_config))

    return backbone




# class BatchNormalization(tf.keras.layers.BatchNormalization):
#     """
#     "Frozen state" and "inference mode" are two separate concepts.
#     `layer.trainable = False` is to freeze the layer, so the layer will use
#     stored moving `var` and `mean` in the "inference mode", and both `gama`
#     and `beta` will not be updated !
#     """
#     def call(self, x, training=False):
#         if not training:
#             training = tf.constant(False)
#         training = tf.logical_and(training, self.trainable)
#         return super().call(x, training)




def build_Res_block(filters, repeat_num):
    block = tf.keras.Sequential()
    for _ in range(repeat_num):
        block.add(CSP_Res(filters=filters))
    return block


class CSP_Res(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(CSP_Res, self).__init__()
        self.conv1 = Conv2d(filters=filters, kernel_size=(1, 1), strides=1, activation="mish")
        self.conv2 = Conv2d(filters=filters, kernel_size=(3, 3), strides=1, activation="mish")
        #self.dropblock = DropBlock2D(keep_prob=0.9, block_size=3)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        #x = self.dropblock(x, training=training)
        output = tf.keras.layers.Add()([inputs, x])

        return output


class CSP_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, num_blocks):
        """
        Darknet53 is composed of a series of CSP Blocks. Each CSP Block contains a number
        of residual blocks. CSP Blocks are intended to improve the following:
            (1) Strengthen the learning ability of the CNN
            (2) Remove computational bottlenecks
            (3) Reduce memory costs

        A CSP Block (also called a Partial Dense Block or a CSPDenseNet) applies a convolution
        to its input volume then splits the resulting volume into two volumes. ** EDIT : the volume
        is not split, it simply goes down two different paths **
        One of those volumes takes a shortcut path; a single convolutional layer is applied to it. 
        The other volume goes through a convolutional layer, a series of residual blocks, 
        and then another convolutional layer. 

        Finally, the two resulting volumes are concatenated together. The concatenated volume is 
        sent through one last convolutional layer.
        

        :param num_filters: number of filters
        :param num_blocks: number of residual blocks
        """
        super(CSP_Block, self).__init__()
        split_filters = num_filters // 2

        self.preconv = Conv2d(filters=num_filters, kernel_size=(3, 3), strides=2, zero_pad=True, padding="valid", activation="mish")
        
        self.shortconv = Conv2d(filters=split_filters, kernel_size=(1, 1), strides=1, activation="mish")
        self.mainconv = Conv2d(filters=split_filters, kernel_size=(1, 1), strides=1, activation="mish")

        self.res_block = build_Res_block(filters=split_filters, repeat_num=num_blocks)

        self.postconv = Conv2d(filters=split_filters, kernel_size=(3, 3), strides=1, activation="mish")

        self.transition = Conv2d(filters=num_filters, kernel_size=(1, 1), strides=1, activation="mish")

    def call(self, inputs, training=False, **kwargs):

        x = self.preconv(inputs, training=training)
        shortcut = self.shortconv(x, training=training)
        mainstream = self.mainconv(x, training=training)
        res = self.res_block(mainstream)
        mainstream = self.postconv(res, training=training)
        outputs = tf.keras.layers.Concatenate()([mainstream, shortcut])
        outputs = self.transition(outputs, training=training)

        return outputs







class CSPDarkNet53(tf.keras.layers.Layer):


    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        #self.conv = Conv2d(filters=32, kernel=(3, 3), stride=1)
        #self.csp_block1 = CSP_Block(64, 1, allow_narrow=False)


        self.conv = Conv2d(filters=32, kernel_size=3, strides=1, activation="mish")
        self.preconv = Conv2d(filters=64, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="mish")
        self.csp_shortconv = Conv2d(filters=64, kernel_size=1, strides=1, activation="mish")

        self.mainconv = Conv2d(filters=64, kernel_size=1, strides=1, activation="mish")
        self.res_conv1 = Conv2d(filters=32, kernel_size=1, strides=1, activation="mish")
        self.res_conv2 = Conv2d(filters=64, kernel_size=3, strides=1, activation="mish")

        self.postconv = Conv2d(filters=64, kernel_size=1, strides=1, activation="mish")
        self.transition = Conv2d(filters=64, kernel_size=1, strides=1, activation="mish")

        self.csp_block2 = CSP_Block(128, 2)
        self.csp_block3 = CSP_Block(256, 8)
        self.csp_block4 = CSP_Block(512, 8)
        self.csp_block5 = CSP_Block(1024, 4)


    def call(self, inputs, training=False, **kwargs):


        x = self.conv(inputs, training=training)
        x = self.preconv(x, training=training)
        csp_shortcut = self.csp_shortconv(x, training=training)

        mainstream = self.mainconv(x, training=training)
        x = self.res_conv1(mainstream, training=training)
        x = self.res_conv2(x, training=training)
        x = tf.keras.layers.Add()([mainstream, x])

        x = self.postconv(x, training=training)
        x = tf.keras.layers.Concatenate()([x, csp_shortcut])
        x = self.transition(x, training=training)


        #x = self.csp_block1(x, training=training)
        x = self.csp_block2(x, training=training)
        route_small = self.csp_block3(x, training=training)
        route_medium = self.csp_block4(route_small, training=training)
        route_large = self.csp_block5(route_medium, training=training)

        return route_small, route_medium, route_large