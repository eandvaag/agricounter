import tensorflow as tf




def build_backbone(config):

    backbone_config = config.arch["backbone_config"]
    backbone_type = backbone_config["backbone_type"]

    backbone = None

    if backbone_config["keras_prebuilt"]:

        load_imagenet_weights = backbone_config["imagenet_pretrained"]
        
        if backbone_type == "resnet50":
            backbone = keras_resnet50_backbone(load_imagenet_weights)

        elif backbone_type == "resnet101":
            backbone = keras_resnet101_backbone(load_imagenet_weights)

        elif backbone_type == "resnet152":
            backbone = keras_resnet152_backbone(load_imagenet_weights)


    else:

        if backbone_type == "resnet18":
            backbone = resnet18_backbone()

        elif backbone_type == "resnet34":
            backbone = resnet34_backbone()

        elif backbone_type == "resnet50":
            backbone = resnet50_backbone()            

        elif backbone_type == "resnet101":
            backbone = resnet101_backbone()

        elif backbone_type == "resnet152":
            backbone = resnet152_backbone()

    if backbone is None:
        raise RuntimeError("Invalid backbone configuration: '{}'.".format(backbone_config))

    return backbone

def keras_resnet50_backbone(load_imagenet_weights):

    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.ResNet50(
        weights=weights,
        include_top=False, 
        input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        model.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def keras_resnet101_backbone(load_imagenet_weights):
    
    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.ResNet101(
        weights=weights,
        include_top=False,
        input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        model.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def keras_resnet152_backbone(load_imagenet_weights):

    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.ResNet152(
        weights=weights,
        include_top=False,
        input_shape=[None, None, 3]
    )

    c3_output, c4_output, c5_output = [
        model.get_layer(layer_name).output
        for layer_name in ["conv3_block8_out", "conv4_block36_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output]
    )






class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        if stride != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=(1, 1),
                                       strides=stride,
                                       use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.9)
            ])
        else:
            self.downsample = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output



class ResNetTypeI(tf.keras.layers.Layer):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = ResNetTypeI.__make_basic_block_layer(filters=64,
                                                           blocks=layer_params[0])
        self.layer2 = ResNetTypeI.__make_basic_block_layer(filters=128,
                                                           blocks=layer_params[1],
                                                           stride=2)
        self.layer3 = ResNetTypeI.__make_basic_block_layer(filters=256,
                                                           blocks=layer_params[2],
                                                           stride=2)
        self.layer4 = ResNetTypeI.__make_basic_block_layer(filters=512,
                                                           blocks=layer_params[3],
                                                           stride=2)

    @staticmethod
    def __make_basic_block_layer(filters, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BasicBlock(filters, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filters, stride=1))

        return res_block

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        l2_out = self.layer2(x, training=training)
        l3_out = self.layer3(l2_out, training=training)
        l4_out = self.layer4(l3_out, training=training)

        #x = self.transposed_conv_layers(x, training=training)
        #heatmap = self.heatmap_layer(x, training=training)
        #reg = self.reg_layer(x, training=training)
        #wh = self.wh_layer(x, training=training)

        return [l2_out, l3_out, l4_out] #[heatmap, reg, wh]



class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv3 = tf.keras.layers.Conv2D(filters=filters * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters * 4,
                                   kernel_size=(1, 1),
                                   strides=stride,
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.9)
        ])

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output



class ResNetTypeII(tf.keras.layers.Layer):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = ResNetTypeII.__make_bottleneck_layer(filters=64,
                                                           blocks=layer_params[0])
        self.layer2 = ResNetTypeII.__make_bottleneck_layer(filters=128,
                                                           blocks=layer_params[1],
                                                           stride=2)
        self.layer3 = ResNetTypeII.__make_bottleneck_layer(filters=256,
                                                           blocks=layer_params[2],
                                                           stride=2)
        self.layer4 = ResNetTypeII.__make_bottleneck_layer(filters=512,
                                                           blocks=layer_params[3],
                                                           stride=2)

    @staticmethod
    def __make_bottleneck_layer(filters, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filters, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BottleNeck(filters, stride=1))

        return res_block


    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        l2_out = self.layer2(x, training=training)
        l3_out = self.layer3(l2_out, training=training)
        l4_out = self.layer4(l3_out, training=training)

        return [l2_out, l3_out, l4_out]





def resnet18_backbone():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])


def resnet34_backbone():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])


def resnet50_backbone():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])


def resnet101_backbone():
    return ResNetTypeII(layer_params=[3, 4, 23, 3])


def resnet152_backbone():
    return ResNetTypeII(layer_params=[3, 8, 36, 3])


#weights_path = keras_utils.get_file(
#                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                WEIGHTS_PATH_NO_TOP,
#                cache_subdir='models',
#                md5_hash='a268eb855778b3df3c7506639542a6af')
#        model.load_weights(weights_path)
