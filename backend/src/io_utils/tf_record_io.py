import os
import numpy as np
import tensorflow as tf
import tqdm


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# def create_patch_tf_records_for_image(image, patch_data, out_dir, is_annotated, require_box=False, require_no_box=False):

#     patch_tf_records = []

#     for i in range(len(patch_data["patches"])):

#         patch_tf_record = {
#             "image_path": bytes_feature(image.image_path),
#             "patch_path": bytes_feature(os.path.join(out_dir, patch_data["patch_names"][i])),
#             #"scenario_uuid": bytes_feature(scenario_uuid),
#             "patch_coords": int_feature_list(list(np.array(patch_data["patch_coords"][i]).astype(np.int64)))
#         }
#         if is_annotated:
#             if require_box and np.array(patch_data["patch_normalized_boxes"][i]).size == 0:
#                 continue
#             if require_no_box and np.array(patch_data["patch_normalized_boxes"][i]).size > 0:
#                 continue

#             patch_tf_record.update({
#                 "patch_normalized_boxes": float_feature_list(list(np.array(patch_data["patch_normalized_boxes"][i]).astype(np.float32).flatten())),
#                 "patch_abs_boxes": int_feature_list(list(np.array(patch_data["patch_abs_boxes"][i]).astype(np.int64).flatten())),
#                 "image_abs_boxes": int_feature_list(list(np.array(patch_data["image_abs_boxes"][i]).astype(np.int64).flatten())),
#                 "patch_classes": int_feature_list(np.array(patch_data["patch_classes"][i]).astype(np.int64))

#             })

#         patch_tf_records.append(tf.train.Example(features=tf.train.Features(feature=patch_tf_record)))

#     return patch_tf_records



def create_patch_tf_records(patch_data_lst, out_dir, is_annotated):

    patch_tf_records = []

    for patch_data in patch_data_lst:

        patch_tf_record = {
            "image_path": bytes_feature(patch_data["image_path"]),
            "image_name": bytes_feature(patch_data["image_name"]),
            "patch_path": bytes_feature(os.path.join(out_dir, patch_data["patch_name"])),
            #"scenario_uuid": bytes_feature(scenario_uuid),
            "patch_coords": int_feature_list(list(np.array(patch_data["patch_coords"]).astype(np.int64)))
        }
        if is_annotated:

            patch_tf_record.update({
                "patch_normalized_boxes": float_feature_list(list(np.array(patch_data["patch_normalized_boxes"]).astype(np.float32).flatten())),
                "patch_abs_boxes": int_feature_list(list(np.array(patch_data["patch_abs_boxes"]).astype(np.int64).flatten())),
                "image_abs_boxes": int_feature_list(list(np.array(patch_data["image_abs_boxes"]).astype(np.int64).flatten())),
                "patch_classes": int_feature_list(np.array(patch_data["patch_classes"]).astype(np.int64))

            })

        patch_tf_records.append(tf.train.Example(features=tf.train.Features(feature=patch_tf_record)))

    return patch_tf_records




# def create_patch_tf_records_for_image(image, patch_data_lst, out_dir, is_annotated, require_box=False, require_no_box=False):

#     patch_tf_records = []

#     for patch_data in patch_data_lst:

#         patch_tf_record = {
#             "image_path": bytes_feature(image.image_path),
#             "patch_path": bytes_feature(os.path.join(out_dir, patch_data["patch_name"])),
#             #"scenario_uuid": bytes_feature(scenario_uuid),
#             "patch_coords": int_feature_list(list(np.array(patch_data["patch_coords"]).astype(np.int64)))
#         }
#         if is_annotated:
#             if require_box and np.array(patch_data["patch_normalized_boxes"]).size == 0:
#                 continue
#             if require_no_box and np.array(patch_data["patch_normalized_boxes"]).size > 0:
#                 continue

#             patch_tf_record.update({
#                 "patch_normalized_boxes": float_feature_list(list(np.array(patch_data["patch_normalized_boxes"]).astype(np.float32).flatten())),
#                 "patch_abs_boxes": int_feature_list(list(np.array(patch_data["patch_abs_boxes"]).astype(np.int64).flatten())),
#                 "image_abs_boxes": int_feature_list(list(np.array(patch_data["image_abs_boxes"]).astype(np.int64).flatten())),
#                 "patch_classes": int_feature_list(np.array(patch_data["patch_classes"]).astype(np.int64))

#             })

#         patch_tf_records.append(tf.train.Example(features=tf.train.Features(feature=patch_tf_record)))

#     return patch_tf_records


# def create_patch_tf_prediction_records(patch_data):

#     patch_tf_records = []

#     for i in range(len(patch_data)):

#         patch_tf_record = {
#             "image_path": bytes_feature(patch_data[i]["image_path"]),
#             "patch_path": bytes_feature(patch_data[i]["patch_path"]),
#             #"scenario_uuid": bytes_feature(patch_data[i]["scenario_uuid"]),
#             "patch_coords": int_feature_list(list(np.array(patch_data[i]["patch_coords"]).astype(np.int64))),
#             "predicted_abs_boxes": float_feature_list(list(np.array(patch_data[i]["predicted_abs_boxes"]).astype(np.int64).flatten())),
#             "predicted_classes": int_feature_list(list(np.array(patch_data[i]["predicted_classes"]).astype(np.int64))),
#             "scores": float_feature_list(list(np.array(patch_data[i]["predicted_classes"]).astype(np.float32)))
#         }
#         patch_tf_records.append(tf.train.Example(features=tf.train.Features(feature=patch_tf_record)))

#     return patch_tf_records







def output_patch_tf_records(out_path, patch_tf_records):

    if os.path.exists(out_path):
        os.remove(out_path)

    with tf.io.TFRecordWriter(out_path) as writer:
        for patch_tf_record in patch_tf_records:
            writer.write(patch_tf_record.SerializeToString())


def parse_sample_from_tf_record(tf_sample, is_annotated):

    schema = {
        "image_path": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "patch_path": tf.io.FixedLenFeature([], tf.string),
        #"scenario_uuid": tf.io.FixedLenFeature([], tf.string),
        "patch_coords": tf.io.VarLenFeature(tf.int64)
    }
    if is_annotated:
        schema.update({
            "patch_normalized_boxes": tf.io.VarLenFeature(tf.float32),
            "patch_abs_boxes": tf.io.VarLenFeature(tf.int64),
            "image_abs_boxes": tf.io.VarLenFeature(tf.int64),
            "patch_classes": tf.io.VarLenFeature(tf.int64)
        })

    sample = tf.io.parse_single_example(tf_sample, schema)
    return sample