import logging
import os
import shutil
import random
import numpy as np


import extract_patches as ep
from io_utils import json_io, tf_record_io

from models.common import annotation_utils




def reset_loss_record(image_set_dir):

    loss_record_path = os.path.join(image_set_dir, "model", "training", "loss_record.json")
    loss_record = json_io.load_json(loss_record_path)
    loss_record["training_loss"]["values"].append([100000000])

    # loss_record = {
    #     "training_loss": { "values": [],
    #                     "best": 100000000,
    #                     "epochs_since_improvement": 0}, 
    #     "validation_loss": {"values": [],
    #                         "best": 100000000,
    #                         "epochs_since_improvement": 0},
    #     # "num_training_images": len(training_image_names)
    # }

    json_io.save_json(loss_record_path, loss_record)





#def update_training_tf_record(username, farm_name, field_name, mission_date, training_image_names):

# def update_training_tf_records(image_set_dir, changed_training_image_names, annotations):
def update_training_tf_records(image_set_dir, annotations):
    logger = logging.getLogger(__name__)

    patches_dir = os.path.join(image_set_dir, "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")
    patch_data = json_io.load_json(patch_data_path)

    training_dir = os.path.join(image_set_dir, "model", "training")
    training_records_dir = os.path.join(training_dir, "training_tf_records")
    # validation_records_dir = os.path.join(training_dir, "validation_tf_records")

    if os.path.exists(training_records_dir):
        shutil.rmtree(training_records_dir)

    # if os.path.exists(validation_records_dir):
    #     shutil.rmtree(validation_records_dir)

    os.makedirs(training_records_dir)
    # os.makedirs(validation_records_dir)

    # if not os.path.exists(training_records_dir):
    #     os.makedirs(training_records_dir)
    # if not os.path.exists(validation_records_dir):
    #     os.makedirs(validation_records_dir)

    for image_name in patch_data["patches"].keys():
        training_tf_record_path = os.path.join(training_records_dir, image_name + ".tfrec")
        # validation_tf_record_path = os.path.join(validation_records_dir, image_name + ".tfrec")


        # if annotations[image_name]["status"] == "completed_for_training" and image_name in changed_training_image_names:

        logger.info("Writing training records for image {} (from: {})".format(image_name, image_set_dir))

            # if os.path.exists(training_tf_record_path):
            #     os.remove(training_tf_record_path)
            # if os.path.exists(validation_tf_record_path):
            #     os.remove(validation_tf_record_path)

        ep.add_annotations_to_patch_records(patch_data["patches"][image_name], annotations[image_name])
        #patch_records.extend(patch_data[image_name]["patches"])

        patch_records = np.array(patch_data["patches"][image_name])

        training_patch_records = patch_records
        # training_size = round(patch_records.size * 0.8)
        # training_subset = random.sample(np.arange(patch_records.size).tolist(), training_size)

        # training_patch_records = patch_records[training_subset]
        # validation_patch_records = np.delete(patch_records, training_subset)

        training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
        tf_record_io.output_patch_tf_records(training_tf_record_path, training_tf_records)

        # validation_tf_records = tf_record_io.create_patch_tf_records(validation_patch_records, patches_dir, is_annotated=True)
        # tf_record_io.output_patch_tf_records(validation_tf_record_path, validation_tf_records)




    patch_data["num_training_regions"] = annotation_utils.get_num_training_regions(annotations)
    json_io.save_json(patch_data_path, patch_data)    


# def update_training_tf_record_dep(image_set_dir, annotations):

#     logger = logging.getLogger(__name__)

#     # image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)

#     patches_dir = os.path.join(image_set_dir, "patches")

#     patch_data_path = os.path.join(patches_dir, "patch_data.json")
#     patch_data = json_io.load_json(patch_data_path)

#     # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
#     # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

#     # available_for_training = []
#     # for image_name in annotations.keys():
#     #     if annotations[image_name]["available_for_training"]:
#     #         available_for_training.append(image_name)

#     training_dir = os.path.join(image_set_dir, "model", "training")
#     #training_images_path = os.path.join(training_dir, "training_images.json")
#     #cur_training_images = json_io.load_json(training_images_path)

#     #if len(available_for_training) != len(cur_training_images):



#     patch_records = []
#     # num_available = 0
#     # for image_name in annotations.keys():
#     #     if annotations[image_name]["status"] == "completed_for_training":
#     #         num_available += 1

#     #for image_name in training_image_names:
#     for image_name in annotations.keys():
#         if annotations[image_name]["status"] == "completed_for_training":
#             ep.add_annotations_to_patch_records(patch_data[image_name]["patches"], annotations[image_name])
#             patch_records.extend(patch_data[image_name]["patches"])

#     patch_records = np.array(patch_records)

#     training_size = round(patch_records.size * 0.8)
#     training_subset = random.sample(np.arange(patch_records.size).tolist(), training_size)

#     training_patch_records = patch_records[training_subset]
#     validation_patch_records = np.delete(patch_records, training_subset)

#     logger.info("Creating training and validation TF records")

#     training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
#     training_patches_record_path = os.path.join(training_dir, "training-patches-record.tfrec")
#     tf_record_io.output_patch_tf_records(training_patches_record_path, training_tf_records)

#     validation_tf_records = tf_record_io.create_patch_tf_records(validation_patch_records, patches_dir, is_annotated=True)
#     validation_patches_record_path = os.path.join(training_dir, "validation-patches-record.tfrec")
#     tf_record_io.output_patch_tf_records(validation_patches_record_path, validation_tf_records)

#     # return num_available






def update_prediction_tf_records(image_set_dir, image_names): #username, farm_name, field_name, mission_date, image_names):

    logger = logging.getLogger(__name__)
    #image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    patches_dir = os.path.join(image_set_dir, "patches")

    patch_data_path = os.path.join(patches_dir, "patch_data.json")
    patch_data = json_io.load_json(patch_data_path)

    # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    for image_name in image_names:
        #is_annotated = annotations[image_name]["status"] == "completed"

        logger.info("Writing inference records for image {} (from: {})".format(image_name, image_set_dir))

        patch_records = patch_data[image_name]["patches"]
        #if is_annotated:
        #    ep.add_annotations_to_patch_records(patch_data[image_name], annotations[image_name])


        # print("sample patch record:", patch_records[0])


        image_prediction_dir = os.path.join(image_set_dir, "model", "prediction", "images", image_name)
        os.makedirs(image_prediction_dir, exist_ok=True)

        tf_records = tf_record_io.create_patch_tf_records(patch_records, patches_dir, is_annotated=False) #is_annotated)
        patches_record_path = os.path.join(image_prediction_dir, "patches-record.tfrec")
        tf_record_io.output_patch_tf_records(patches_record_path, tf_records)
