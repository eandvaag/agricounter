import logging
import os
import shutil
import glob

from io_utils import json_io, w3c_io

import excess_green
import metadata

from image_set import Image
from models.common import annotation_utils


init_cameras = {
    "GoPro": {
        "HERO9 Black": {
            "sensor_width": 6.17,
            "sensor_height": 4.55,
            "focal_length": 3,
            "image_width_px": 4000,
            "image_height_px": 3000
        },
        "HERO6 Black": {
            "sensor_width": 6.17,
            "sensor_height": 4.55,
            "focal_length": 3,
            "image_width_px": 4000,
            "image_height_px": 3000
        }
    },
    "Hasselblad": {
        "L1D-20c": {
            "sensor_width": 13.2,
            "sensor_height": 8.8,
            "focal_length": 10.3,
            "image_width_px": 5472,
            "image_height_px": 3648
        }
    },
    "Phase One": {
        "iXU1000": {
            "sensor_width": 53.4,
            "sensor_height": 40.0,
            "focal_length": 55.0,
            "image_width_px": 11608,
            "image_height_px": 8708
        }
    }
}

def clear_usr_requests_and_results(username):

    usr_dir = os.path.join("usr", "data", username)
    for farm_path in glob.glob(os.path.join(usr_dir, "image_sets", "*")):
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                prediction_dir = os.path.join(mission_path, "model", "prediction")
                image_requests_dir = os.path.join(prediction_dir, "image_requests")
                shutil.rmtree(image_requests_dir)
                os.makedirs(image_requests_dir)

                images_dir = os.path.join(prediction_dir, "images")
                if os.path.exists(images_dir):
                    shutil.rmtree(images_dir)
                    os.makedirs(images_dir)

                image_set_requests_dir = os.path.join(prediction_dir, "image_set_requests")
                aborted_dir = os.path.join(image_set_requests_dir, "aborted")
                shutil.rmtree(aborted_dir)
                os.makedirs(aborted_dir)

                pending_dir = os.path.join(image_set_requests_dir, "pending")
                shutil.rmtree(pending_dir)
                os.makedirs(pending_dir)                

                results_dir = os.path.join(mission_path, "model", "results")
                shutil.rmtree(results_dir)
                os.makedirs(results_dir)


def make_all_image_sets_public(): #add_objects_names(): #fix_statuses():

    for usr_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(usr_path)
        for farm_path in glob.glob(os.path.join(usr_path, "image_sets", "*")):
            #farm_dir = os.path.basename(farm_path)
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                #field_dir = os.path.basename(field_path)
                for mission_path in glob.glob(os.path.join(field_path, "*")):

                    object_info_path = os.path.join(mission_path, "annotations", "object_info.json")
                    #object_info = {
                    #    "object_name": "canola_seedling"
                    #}
                    #json_io.save_json(object_info_path, object_info)

                    public_path = os.path.join(mission_path, "public.json")
                    os.remove(public_path)
                    os.remove(object_info_path)

                    metadata_path = os.path.join(mission_path, "metadata", "metadata.json")
                    metadata = json_io.load_json(metadata_path)

                    metadata["is_ortho"] = "no"
                    metadata["is_public"] = "yes"
                    metadata["object_name"] = "canola_seedling"

                    json_io.save_json(metadata_path, metadata)


                    # json_io.save_json(public_path, {})

                    #status_path = os.path.join(mission_path, "model", "status.json")
                    #status = json_io.load_json(status_path)

                    #status["model_name"] = "---"
                    #status["model_creator"] = "---"
                    #print(status)
                    #print()
                    #json_io.save_json(status_path, status)


def update_w3c_annotation_file(image_set_dir):
    
    annotations_w3c_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations_w3c = w3c_io.load_annotations(annotations_w3c_path, {"plant": 0})
    annotations = {}
    for image_name in annotations_w3c.keys():
        image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
        image = Image(image_path)
        w, h = image.get_wh()
        annotations[image_name] = {
            "boxes": [],
            "training_regions": [],
            "test_regions": [],
            "source": "NA"
        }
        annotations[image_name]["boxes"] = annotations_w3c[image_name]["boxes"]
        # if annotations_w3c[image_name]["status"] == "completed_for_training":
        #     annotations[image_name]["training_regions"].append([0, 0, h, w])
        if annotations_w3c[image_name]["status"] == "completed_for_testing" or annotations_w3c[image_name]["status"] == "completed_for_training":
            annotations[image_name]["test_regions"].append([0, 0, h, w])
            annotations[image_name]["source"] = "manually_annotated_from_scratch"

    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotation_utils.save_annotations(annotations_path, annotations)

def default_overlay_appearance():
    # return {
    #     "draw_order": ["region_of_interest", "training_region", "test_region", "annotation", "prediction"],
    #     "style": {
    #         "annotation": "strokeRect",
    #         "prediction": "strokeRect",
    #         "region_of_interest": "strokeRect",
    #         "training_region": "strokeRect",
    #         "test_region": "strokeRect"
    #     },
    #     "colors": {
    #         "annotation": "#0080ff",
    #         "prediction": "#ff4040",
    #         "region_of_interest": "#17c44b",
    #         "training_region": "#ff51eb",
    #         "test_region": "#ffae00"
    #     }
    # }

    return {
        "draw_order": ["region_of_interest", "training_region", "test_region", "annotation", "prediction"],
        "style": {
            "annotation": "strokeRect",
            "prediction": "strokeRect",
            "region_of_interest": "strokeRect",
            "training_region": "strokeRect",
            "test_region": "strokeRect"
        },
        "colors": {
            "annotation": ["#0080ff", "#ff0033", "#59ff00", "#8000ff", "#ff6200", "#00ff77", "#fb00ff", "#ffff00", "#00ffe5"],
            "prediction": ["#7dbeff", "#ff8099", "#acff80", "#bf80ff", "#ffb080", "#80ffbb", "#fd80ff", "#ffff80", "#80fff2"],
            "region_of_interest": "#a291ba",
            "training_region": "#a4ba91",
            "test_region": "#91bab9"
        }
    }


def init_usr(username):

    usr_dir = os.path.join("usr", "data", username)
    cameras_dir = os.path.join(usr_dir, "cameras")
    image_sets_dir = os.path.join(usr_dir, "image_sets")
    models_dir = os.path.join(usr_dir, "models")
    pending_dir = os.path.join(models_dir, "pending")
    aborted_dir = os.path.join(models_dir, "aborted")
    available_dir = os.path.join(models_dir, "available")
    public_dir = os.path.join(available_dir, "public")
    private_dir = os.path.join(available_dir, "private")


    os.makedirs(usr_dir)
    os.makedirs(cameras_dir)
    os.makedirs(image_sets_dir)
    os.makedirs(models_dir)
    os.makedirs(pending_dir)
    os.makedirs(aborted_dir)
    os.makedirs(available_dir)
    os.makedirs(public_dir)
    os.makedirs(private_dir)

    cameras_path = os.path.join(cameras_dir, "cameras.json")
    json_io.save_json(cameras_path, init_cameras)

    private_image_sets_path = os.path.join(usr_dir, "private_image_sets.json")
    json_io.save_json(private_image_sets_path, {})

    overlay_appearance_path = os.path.join(usr_dir, "overlay_appearance.json")
    overlay_appearance = default_overlay_appearance()
    json_io.save_json(overlay_appearance_path, overlay_appearance)


def update_init_cameras(replace=False):

    for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
        cameras_path = os.path.join(usr_dir, "cameras", "cameras.json")

        if os.path.exists(cameras_path):

            if replace:
                json_io.save_json(cameras_path, init_cameras)

            else:
                usr_cameras = json_io.load_json(cameras_path)
                for make in init_cameras.keys():
                    if make not in usr_cameras:
                        usr_cameras[make] = {}
                    for model in init_cameras[make].keys():
                        usr_cameras[make][model] = init_cameras[make][model]

                json_io.save_json(cameras_path, usr_cameras)



def fix_e():

    for farm_path in glob.glob(os.path.join("usr", "data", "erik", "image_sets", "*")):
        farm_dir = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_dir = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_dir = os.path.basename(mission_path)

                annotations_path = os.path.join(mission_path, "annotations", "annotations.json")
                try:
                    annotations = annotation_utils.load_annotations(annotations_path)
                except Exception as e:
                    continue
                for image_name in annotations.keys():
                    if annotations[image_name]["boxes"].size > 0:
                        annotations[image_name]["source"] = "manually_annotated_from_scratch"
                    else:
                        annotations[image_name]["source"] = "NA"
                    if "predictions_used_as_annotations" in annotations[image_name]:
                        del annotations[image_name]["predictions_used_as_annotations"]

                annotation_utils.save_annotations(annotations_path, annotations)

                # annotations_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                # annotations = json_io.load_json(annotations_path)
                # fully_trained = "True"
                # for image_name in annotations.keys():
                #     # annotations[image_name]["update_time"] = 0 #int(time.time())
                #     if annotations[image_name]["status"] == "completed_for_training":
                #         fully_trained = "False"

                # json_io.save_json(annotations_path, annotations)

                # # status = {
                # #     "status": "idle",
                # #     "fully_trained": "True",
                # #     "update_num": 0
                # # }
                # status_path = os.path.join(mission_path, "model", "status.json")
                # status = json_io.load_json(status_path)
                # status["fully_trained"] = fully_trained

                # json_io.save_json(status_path, status)


def delete_patches():

    for usr_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(usr_path)
        if username != "image_sets":
            for farm_path in glob.glob(os.path.join(usr_path, "image_sets", "*")):
                #farm_dir = os.path.basename(farm_path)
                for field_path in glob.glob(os.path.join(farm_path, "*")):
                    #field_dir = os.path.basename(field_path)
                    for mission_path in glob.glob(os.path.join(field_path, "*")):
                        #mission_dir = os.path.basename(mission_path)

                        patches_dir = os.path.join(mission_path, "patches")
                        shutil.rmtree(patches_dir)
                        os.makedirs(patches_dir)

                        training_dir = os.path.join(mission_path, "model", "training")
                        training_patches_record_path = os.path.join(training_dir, "training-patches-record.tfrec")
                        if os.path.exists(training_patches_record_path):
                            os.remove(training_patches_record_path)

                        validation_patches_record_path = os.path.join(training_dir, "validation-patches-record.tfrec")
                        if os.path.exists(validation_patches_record_path):
                            os.remove(validation_patches_record_path)

def fix_kaylie_sets():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    kaylie_image_sets_dir = os.path.join("usr", "data", "kaylie", "image_sets")

    #usr_dir = os.path.join("usr", "data", username)
    for farm_path in glob.glob(os.path.join("usr", "data", "image_sets", "*")):
        farm_dir = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_dir = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_dir = os.path.basename(mission_path)

                if (farm_dir == "row_spacing" and field_dir == "brown") and mission_dir == "2021-06-01-low_res":
                    pass
                else:

                    print("processing: {} {} {}".format(farm_dir, field_dir, mission_dir))

                    new_farm_path = os.path.join(kaylie_image_sets_dir, farm_dir)
                    new_field_path = os.path.join(new_farm_path, field_dir)
                    new_mission_path = os.path.join(new_field_path, mission_dir)

                    upload_status_path = os.path.join(new_mission_path, "upload_status.json")
                    if not os.path.exists(upload_status_path):

                        if not os.path.exists(new_farm_path):
                            os.makedirs(new_farm_path)
                        
                        if not os.path.exists(new_field_path):
                            os.makedirs(new_field_path)
                        
                        if not os.path.exists(new_mission_path):
                            os.makedirs(new_mission_path)

                        logger.info("Copying images")
                        images_path = os.path.join(mission_path, "images")
                        new_images_path = os.path.join(new_mission_path, "images")
                        # print(images_path)
                        # print(new_images_path)
                        shutil.copytree(images_path, new_images_path)

                        logger.info("Copying DZI images")
                        dzi_images_path = os.path.join(mission_path, "dzi_images")
                        new_dzi_images_path = os.path.join(new_mission_path, "dzi_images")
                        shutil.copytree(dzi_images_path, new_dzi_images_path)

                        logger.info("Copying annotations")
                        new_annotations_dir = os.path.join(new_mission_path, "annotations")
                        os.makedirs(new_annotations_dir)

                        old_annotations_w3c_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                        annotations_w3c_path = os.path.join(new_annotations_dir, "annotations_w3c.json")
                        shutil.copy(old_annotations_w3c_path, annotations_w3c_path)

                        # annotations_path = os.path.join(mission_path, "annotations")
                        # new_annotations_path = os.path.join(new_mission_path, "annotations")
                        # shutil.copytree(annotations_path, new_annotations_path)
                        
                        annotations = json_io.load_json(annotations_w3c_path)
                        fully_trained = "True"
                        for image_name in annotations.keys():
                            # annotations[image_name]["update_time"] = 0 #int(time.time())
                            if annotations[image_name]["status"] == "completed":
                                if mission_dir[:4] == "2022":
                                    annotations[image_name]["status"] = "completed_for_testing"
                                else:
                                    annotations[image_name]["status"] = "completed_for_training"
                                    fully_trained = "False"

                                

                        json_io.save_json(annotations_w3c_path, annotations)


                        logger.info("Creating directories")
                        patches_dir = os.path.join(new_mission_path, "patches")
                        os.makedirs(patches_dir)

                        model_dir = os.path.join(new_mission_path, "model")
                        os.makedirs(model_dir)
                        prediction_dir = os.path.join(model_dir, "prediction")
                        os.makedirs(prediction_dir)
                        image_requests_dir = os.path.join(prediction_dir, "image_requests")
                        os.makedirs(image_requests_dir)
                        images_dir = os.path.join(prediction_dir, "images")
                        os.makedirs(images_dir)
                        image_set_requests_dir = os.path.join(prediction_dir, "image_set_requests")
                        os.makedirs(image_set_requests_dir)
                        aborted_dir = os.path.join(image_set_requests_dir, "aborted")
                        os.makedirs(aborted_dir)
                        pending_dir = os.path.join(image_set_requests_dir, "pending")
                        os.makedirs(pending_dir)

                        results_dir = os.path.join(model_dir, "results")
                        os.makedirs(results_dir)
                        # retrieval_dir = os.path.join(results_dir, "retrieval")
                        # os.makedirs(retrieval_dir)
                        training_dir = os.path.join(model_dir, "training")
                        os.makedirs(training_dir)
                        usr_block = {}
                        usr_block_path = os.path.join(training_dir, "usr_block.json")
                        json_io.save_json(usr_block_path, usr_block)

                        loss_record = {
                            "training_loss": { "values": [],
                                            "best": 100000000,
                                            "epochs_since_improvement": 100000000}, 
                            "validation_loss": {"values": [],
                                                "best": 100000000,
                                                "epochs_since_improvement": 100000000},
                        }
                        loss_record_path = os.path.join(training_dir, "loss_record.json")
                        json_io.save_json(loss_record_path, loss_record)

                        weights_dir = os.path.join(model_dir, "weights")
                        os.makedirs(weights_dir)

                        default_weights_path = os.path.join("usr", "shared", "weights", "default_weights.h5")
                        shutil.copy(default_weights_path, os.path.join(weights_dir, "best_weights.h5"))
                        shutil.copy(default_weights_path, os.path.join(weights_dir, "cur_weights.h5"))

                        status = {
                            "status": "idle",
                            "fully_trained": fully_trained,
                            "update_num": 0
                        }
                        status_path = os.path.join(model_dir, "status.json")
                        json_io.save_json(status_path, status)


                        logger.info("Creating excess green images")
                        excess_green_dir = os.path.join(new_mission_path, "excess_green")
                        os.makedirs(excess_green_dir)
                        excess_green.create_excess_green_for_image_set(new_mission_path)


                        logger.info("Extracting metadata")
                        metadata.extract_metadata(new_mission_path, camera_height=2.0)

                        
                        logger.info("Writing upload status")
                        upload_status = {"status": "uploaded"}
                        json_io.save_json(upload_status_path, upload_status)

