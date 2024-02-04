import logging
import os
import glob
import argparse

import check_channels
import extract_metadata
import create_dzi

import emit
from io_utils import json_io

ROOT_DIR_NAMES = [
    "images",
    "dzi_images",
    "annotations",
    "metadata",
    "excess_green"
]

MODEL_DIR_NAMES = [
    "patches",
    "training",
    "prediction",
    "weights",
    "results"
]

RESULTS_DIR_NAMES = [
    "available",
    "aborted"
]



def process_upload(image_set_dir):


    try:
        upload_status_path = os.path.join(image_set_dir, "upload_status.json")

        config_path = os.path.join(image_set_dir, "config.json")
        config = json_io.load_json(config_path)

        username = config["username"]
        farm_name = config["farm_name"]
        field_name = config["field_name"]
        mission_date = config["mission_date"]
        is_ortho = config["is_ortho"]

        json_io.save_json(upload_status_path, {"status": "processing"})


        check_channels.check_channels(image_set_dir, is_ortho)


        for root_dir_name in ROOT_DIR_NAMES:
            os.makedirs(os.path.join(image_set_dir, root_dir_name), exist_ok=True)

        for model_dir_name in MODEL_DIR_NAMES:
            os.makedirs(os.path.join(image_set_dir, "model", model_dir_name), exist_ok=True)

        for results_dir_name in RESULTS_DIR_NAMES:
            os.makedirs(os.path.join(image_set_dir, "model", "results", results_dir_name), exist_ok=True)


        init_model_status = {
            "model_name": "",
            "model_creator": "",
            "state_name": "Idle",
            "progress": "",
            "error_message": "",
            "prediction_image_names": ""
        }

        model_status_path = os.path.join(image_set_dir, "model", "status.json")
        json_io.save_json(model_status_path, init_model_status)



        image_names = []
        for image_path in glob.glob(os.path.join(image_set_dir, "images", "*")):
            image_name = os.path.basename(image_path).split(".")[0]
            image_names.append(image_name)


        init_annotations = {}
        for image_name in image_names:
            init_annotations[image_name] = {
                "boxes": [],
                "classes": [],
                "regions_of_interest": [],
                "fine_tuning_regions": [],
                "test_regions": [],
                "source": "NA"
            }

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        json_io.save_json(annotations_path, init_annotations)

        tags_path = os.path.join(image_set_dir, "annotations", "tags.json")
        json_io.save_json(tags_path, {})


        extract_metadata.extract_metadata(config)


        excess_green_record = {}
        for image_name in image_names:
            excess_green_record[image_name] = 0
    
        excess_green_record_path = os.path.join(image_set_dir, "excess_green", "record.json")
        json_io.save_json(excess_green_record_path, excess_green_record)


        create_dzi.create_dzi(image_set_dir)

        os.remove(config_path)

        json_io.save_json(upload_status_path, {"status": "uploaded"})


    except Exception as e:
        
        json_io.save_json(upload_status_path, {"status": "failed", "error": str(e)})



    emit.emit_upload_change({
        "username": username, 
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date
    })

        
        
        

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    
    args = parser.parse_args()
    image_set_dir = args.image_set_dir


    process_upload(image_set_dir)
