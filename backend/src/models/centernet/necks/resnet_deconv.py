import tensorflow as tf

from models.common import driver_utils


def build_neck(config):


    # Uses the approach found in "Simple Baselines for Human Pose Estimation and Tracking"
  
    num_layers = 3
    filters = [256, 256, 256]
    kernel_sizes = [4, 4, 4]

    layers = tf.keras.Sequential()
    for i in range(num_layers):
        layers.add(tf.keras.layers.Conv2DTranspose(filters=filters[i],
                                                   kernel_size=kernel_sizes[i],
                                                   strides=2,
                                                   padding="same",
                                                   use_bias=False))
        layers.add(tf.keras.layers.BatchNormalization())
        layers.add(tf.keras.layers.ReLU())
    return layers        




# class ModifiedResNet(tf.keras.layers.Layer):

#     def __init__(self, config, **kwargs):
#         super(ModifiedResNet, self).__init__(name="ModifiedResNet", **kwargs)
#         self.imagenet_pretrained = "imagenet_pretrained" in config.arch["backbone_config"] and \
#                                     config.arch["backbone_config"]["imagenet_pretrained"]
#         self.backbone = get_backbone(config.arch["backbone_config"])

#         # Uses the approach found in "Simple Baselines for Human Pose Estimation and Tracking"
#         self.transposed_conv_layers = self.__make_transposed_conv_layers(num_layers=3, 
#                                                                          filters=[256, 256, 256], 
#                                                                          kernel_sizes=[4, 4, 4])

#     def call(self, images, training=False):
#         if self.imagenet_pretrained:
#             images = tf.keras.applications.resnet.preprocess_input(images)
#         c5_output = self.backbone(images, training=training)
#         output = self.transposed_conv_layers(c5_output, training=training)
#         return output

#     def __make_transposed_conv_layers(self, num_layers, filters, kernel_sizes):
#         layers = tf.keras.Sequential()
#         for i in range(num_layers):
#             layers.add(tf.keras.layers.Conv2DTranspose(filters=filters[i],
#                                                        kernel_size=kernel_sizes[i],
#                                                        strides=2,
#                                                        padding="same",
#                                                        use_bias=False))
#             layers.add(tf.keras.layers.BatchNormalization())
#             layers.add(tf.keras.layers.ReLU())
#         return layers