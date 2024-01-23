import tensorflow as tf
import numpy as np

from models.efficientdet.backbones.efficientnet import keras_D0_backbone

def get_backbone(backbone_type):

    if backbone_type == "D0":

        model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            #input_shape=[None, None, 3]
            input_shape=[512, 512, 3]
        )

        c3_output, c4_output, c5_output, c6_output, c7_output = [
            model.get_layer(layer_name).output
            for layer_name in ["block3b_add", "block4c_add", "block5c_add", "block6d_add", "block7a_project_bn"]
        ]

        model = tf.keras.Model(
            inputs=[model.inputs], outputs=[c3_output, c4_output, c5_output, c6_output, c7_output]
        )

    else:
        raise RuntimeError("Unsupported backbone: '{}'".format(config.backbone_type))

    return model




class BiFPN(tf.keras.layers.Layer):

    def __init__(self, output_channels, layers):
        super(BiFPN, self).__init__()

        self.transform_convs = []
        self.bifpn_modules = []

        # transform_convs perform 1x1 convolutions with `output_channel` filters.
        # This ensures that the input to each level of the first module of the BiFPN has
        # the right number of channels.
        self.levels = 5
        strides = [1, 1, 2, 2, 4]

        for i in range(self.levels):
            self.transform_convs.append(ConvNormAct(output_channels,
                                                    kernel_size=(1, 1),
                                                    strides=strides[i],#1,
                                                    padding="same"))
        for _ in range(layers):
            self.bifpn_modules.append(BiFPNModule(output_channels))

    def call(self, inputs, training=None, **kwargs):
        #for i in range(len(inputs)):
        #    print("\ninputs[{}].shape: {}".format(i, inputs[i].shape))


        assert len(inputs) == self.levels
        x = []
        for i in range(len(inputs)):
            x.append(self.transform_convs[i](inputs[i], training=training))
        
        #for i in range(len(x)):
        #    print("\nx[{}].shape: {}".format(i, x[i].shape))

        for i in range(len(self.bifpn_modules)):
            x = self.bifpn_modules[i](x, training=training)

        return x




class BiFPNModule(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super(BiFPNModule, self).__init__()
        self.w_fusion_lst = []
        num_fusion_components = 8
        for i in range(num_fusion_components):
            self.w_fusion_lst.append(WeightedFeatureFusion(output_channels))

        #self.upsample_2x = tf.keras.layers.UpSampling2D(size=(2, 2))
        #self.maxpool_2x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.upsampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs, training=None, **kwargs):
        assert len(inputs) == 5
        # f3, f4, f5, f6, f7 = inputs

        # f6_d = self.fusion_lst[0]([f6, self.upsample_2x(f7)], training=training)
        # f5_d = self.fusion_lst[1]([f5, self.upsample_2x(f6_d)], training=training)
        # f4_d = self.fusion_lst[2]([f4, self.upsample_2x(f5_d)], training=training)

        # f3_u = self.fusion_lst[3]([f3, self.upsample_2x(f4_d)], training=training)
        # f4_u = self.fusion_lst[4]([f4, f4_d, self.maxpool_2x(f3_u)], training=training)
        # f5_u = self.fusion_lst[5]([f5, f5_d, self.maxpool_2x(f4_u)], training=training)
        # f6_u = self.fusion_lst[6]([f6, f6_d, self.maxpool_2x(f5_u)], training=training)
        # f7_u = self.fusion_lst[7]([f7, self.maxpool_2x(f6_u)], training=training)

        f3, f4, f5, f6, f7 = inputs
        f6_d = self.w_fusion_lst[0]([f6, self.upsampling_1(f7)], training=training)
        f5_d = self.w_fusion_lst[1]([f5, self.upsampling_2(f6_d)], training=training)
        f4_d = self.w_fusion_lst[2]([f4, self.upsampling_3(f5_d)], training=training)

        f3_u = self.w_fusion_lst[3]([f3, self.upsampling_4(f4_d)], training=training)
        f4_u = self.w_fusion_lst[4]([f4, f4_d, self.maxpool_1(f3_u)], training=training)
        f5_u = self.w_fusion_lst[5]([f5, f5_d, self.maxpool_2(f4_u)], training=training)
        f6_u = self.w_fusion_lst[6]([f6, f6_d, self.maxpool_3(f5_u)], training=training)
        f7_u = self.w_fusion_lst[7]([f7, self.maxpool_4(f6_u)], training=training)

        return [f3_u, f4_u, f5_u, f6_u, f7_u]



class WeightedFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super(WeightedFeatureFusion, self).__init__()
        self.epsilon = 1e-4
        self.conv = SeparableConvNormAct(filters=output_channels, 
                                         kernel_size=(3, 3),
                                         strides=1,
                                         padding="same")

    def build(self, input_features):
        self.num_input_features = len(input_features)
        self.fusion_weights = self.add_weight(name="fusion_w",
                                              shape=(self.num_input_features, ),
                                              dtype=tf.dtypes.float32,
                                              initializer=tf.constant_initializer(value=(1.0 / self.num_input_features)),
                                              trainable=True)

    def call(self, inputs, training=None, **kwargs):
        fusion_w = tf.nn.relu(self.fusion_weights)
        sum_features = []
        for i in range(self.num_input_features):
            sum_features.append(fusion_w[i] * inputs[i])
        feature_sum = tf.reduce_sum(input_tensor=sum_features, axis=0)
        normalizer = tf.reduce_sum(input_tensor=fusion_w) + self.epsilon
        output_feature = feature_sum / normalizer
        return output_feature





class SeparableConvNormAct(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding):
        super(SeparableConvNormAct, self).__init__()
        self.conv = tf.keras.layers.SeparableConv2D(filters=filters,
                                                    kernel_size=kernel_size,
                                                    strides=strides,
                                                    padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x

class ConvNormAct(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding):
        super(ConvNormAct, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x




class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, output_channels, num_layers=5):
        super(TransposeLayer, self).__init__()
        self.layers = num_layers
        self.transpose_layers = []
        for i in range(self.layers - 1):
            self.transpose_layers.append(tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=(4, 4), strides=2, padding="same"),
                tf.keras.layers.BatchNormalization()
            ]))

    def call(self, inputs, training=None, **kwargs):
        assert len(inputs) == self.layers
        f3, f4, f5, f6, f7 = inputs
        f6 += tf.nn.swish(self.transpose_layers[0](f7, training=training))
        f5 += tf.nn.swish(self.transpose_layers[1](f6, training=training))
        f4 += tf.nn.swish(self.transpose_layers[2](f5, training=training))
        f3 += tf.nn.swish(self.transpose_layers[3](f4, training=training))
        return f3







def build_head(input_channels, output_channels, width, layers, bias_init):

    head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, input_channels])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(layers):
        head.add(
            tf.keras.layers.Conv2D(width, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(tf.keras.layers.Activation(tf.keras.activations.swish))
    head.add(
        tf.keras.layers.Conv2D(
            output_channels,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head



class EfficientDet(tf.keras.Model):

    def __init__(self, config):
        super(EfficientDet, self).__init__(name="EfficientDet")

        self.num_classes = config.img_set.num_classes
        self.backbone = get_backbone(config.backbone_type)
        #self.backbone = get_efficient_net(1.0, 1.0, 0.2)
        self.bifpn = BiFPN(output_channels=config.bifpn_width, layers=config.bifpn_depth)


        # A simple adaptation so that the CenterNet heads can be attached to an EfficientDet
        # self.transpose = TransposeLayer(output_channels=config.bifpn_width)
        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.box_head = build_head(config.bifpn_width, 9 * 4,
                                   config.bifpn_width, config.head_layers, "zeros")
        self.cls_head = build_head(config.bifpn_width, 9 * self.num_classes, 
                                   config.bifpn_width, config.head_layers, prior_probability)

    def call(self, inputs, training=None, **kwargs):
        #print("\n\n(1) inputs.shape\n\n", inputs.shape)
        N = tf.shape(inputs)[0]
        box_outputs = []
        cls_outputs = []
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)

        #print("\n\n(2) inputs.shape\n\n", inputs.shape)
        x = self.backbone(x, training=training)
        x = self.bifpn(x, training=training)
        #x = self.transpose(x, training=training)
        #print("x.shape", x.shape)

        for feature in x:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        box_outputs = tf.concat(box_outputs, axis=1)
        cls_outputs = tf.concat(cls_outputs, axis=1)

        return tf.concat([box_outputs, cls_outputs], axis=-1)

