import tensorflow as tf
import tensorflow_addons as tfa


class Conv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding="same", zero_pad=False, activation="mish", use_batch_norm=True):
        super(Conv2d, self).__init__()
        self.activation = activation
        # self.conv = tf.keras.layers.Conv2D(filters=filters,
        #                                    kernel_size=kernel_size,
        #                                    strides=strides,
        #                                    padding=padding,
        #                                    use_bias=False
        #                                    )
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           kernel_regularizer=None, #tf.keras.regularizers.l2(0.00005),
                                           #kernel_regularizer=tf.keras.regularizers.l2(0.00005),
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           padding=padding,
                                           use_bias=(not use_batch_norm)
                                           )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()
        self.zero_pad = zero_pad

    def call(self, inputs, training=False, **kwargs):
        if self.zero_pad:
            inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)

        x = self.conv(inputs)
        if self.use_batch_norm:
            x = self.bn(x, training=training)
        if self.activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "linear":
            x = tf.keras.activations.linear(x)
        elif self.activation == "linear_float32":
            x = tf.keras.layers.Activation('linear', dtype='float32', name='predictions')(x) #tf.keras.activations.linear(x, dtype="float32")    

        return x


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]