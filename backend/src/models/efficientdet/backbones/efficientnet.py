import tensorflow as tf


def keras_D0_backbone(load_imagenet_weights):

    weights = "imagenet" if load_imagenet_weights else None

    model = tf.keras.applications.EfficientNetB0(
        weights=weights,
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

    return model