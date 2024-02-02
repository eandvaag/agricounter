import logging
import os
import shutil
import numpy as np


from io_utils import json_io, tf_record_io
import extract_patches as ep
from models.common import annotation_utils




def reset_loss_record(image_set_dir):

    loss_record_path = os.path.join(image_set_dir, "model", "training", "loss_record.json")
    loss_record = json_io.load_json(loss_record_path)
    loss_record["training_loss"]["values"].append([100000000])

    json_io.save_json(loss_record_path, loss_record)




def update_training_tf_records(image_set_dir, annotations):
    logger = logging.getLogger(__name__)

    patches_dir = os.path.join(image_set_dir, "model", "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")
    patch_data = json_io.load_json(patch_data_path)

    training_dir = os.path.join(image_set_dir, "model", "training")
    training_records_dir = os.path.join(training_dir, "training_tf_records")

    if os.path.exists(training_records_dir):
        shutil.rmtree(training_records_dir)

    os.makedirs(training_records_dir)

    for image_name in patch_data.keys():
        training_tf_record_path = os.path.join(training_records_dir, image_name + ".tfrec")
        logger.info("Writing training records for image {} (from: {})".format(image_name, image_set_dir))

        ep.add_annotations_to_patch_records(patch_data[image_name], annotations[image_name])

        patch_records = np.array(patch_data[image_name])

        training_patch_records = patch_records

        training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
        tf_record_io.output_patch_tf_records(training_tf_record_path, training_tf_records)


    # patch_data["num_training_regions"] = annotation_utils.get_num_training_regions(annotations)
    json_io.save_json(patch_data_path, patch_data)    