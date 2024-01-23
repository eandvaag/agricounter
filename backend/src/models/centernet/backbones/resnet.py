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



class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=(1, 1),
                                       strides=stride,
                                       use_bias=False),
                tf.keras.layers.BatchNormalization()
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
        self.bn1 = tf.keras.layers.BatchNormalization()
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
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        #x = self.transposed_conv_layers(x, training=training)
        #heatmap = self.heatmap_layer(x, training=training)
        #reg = self.reg_layer(x, training=training)
        #wh = self.wh_layer(x, training=training)

        return x #[heatmap, reg, wh]



class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filters * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.downsample = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters * 4,
                                   kernel_size=(1, 1),
                                   strides=stride,
                                   use_bias=False),
            tf.keras.layers.BatchNormalization()
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
        self.bn1 = tf.keras.layers.BatchNormalization()
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
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        return x







def keras_resnet50_backbone(load_imagenet_weights):
    
    weights = "imagenet" if load_imagenet_weights else None

    return tf.keras.applications.ResNet50(
                weights=weights,
                include_top=False, 
                input_shape=[None, None, 3]
    )

def keras_resnet101_backbone(load_imagenet_weights):
    
    weights = "imagenet" if load_imagenet_weights else None

    return tf.keras.applications.ResNet101(
                weights=weights,
                include_top=False, 
                input_shape=[None, None, 3]
    )


def keras_resnet152_backbone(load_imagenet_weights):
    
    weights = "imagenet" if load_imagenet_weights else None

    return tf.keras.applications.ResNet152(
                weights=weights,
                include_top=False, 
                input_shape=[None, None, 3]
    )


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
