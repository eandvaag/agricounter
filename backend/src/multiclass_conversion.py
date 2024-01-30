import os
import glob
import shutil

from io_utils import json_io
from models.common import inference_metrics


WRITE_CHANGES = True


default_overlay_appearance = {
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



def convert_overlay_colors():

    print("start convert_overlay_colors")

    for user_path in glob.glob(os.path.join("usr", "data", "*")):
        overlay_appearance_path = os.path.join(user_path, "overlay_appearance.json")


        if WRITE_CHANGES:
            json_io.save_json(overlay_appearance_path, default_overlay_appearance)


    print("end convert_overlay_colors")

def convert_annotations():

    print("start convert_annotations")

    for user_path in glob.glob(os.path.join("usr", "data", "*")):
        for farm_path in glob.glob(os.path.join(user_path, "image_sets", "*")):
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                for mission_path in glob.glob(os.path.join(field_path, "*")):

                    annotations_path = os.path.join(mission_path, "annotations", "annotations.json")

                    annotations = json_io.load_json(annotations_path)

                    for image_name in annotations.keys():
                        annotations[image_name]["classes"] = [0] * len(annotations[image_name]["boxes"])

                    if WRITE_CHANGES:
                        json_io.save_json(annotations_path, annotations)

    print("end convert_annotations")


def convert_metadata():

    print("start convert_metadata")

    for user_path in glob.glob(os.path.join("usr", "data", "*")):
        for farm_path in glob.glob(os.path.join(user_path, "image_sets", "*")):
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                for mission_path in glob.glob(os.path.join(field_path, "*")):

                    metadata_path = os.path.join(mission_path, "metadata", "metadata.json")

                    metadata = json_io.load_json(metadata_path)

                    if metadata["object_name"] == "canola_seedling":
                        metadata["object_classes"] = ["Canola Seedling"]
                    elif metadata["object_name"] == "wheat_head":
                        metadata["object_classes"] = ["Wheat Head"]

                    del metadata["object_name"]

                    if WRITE_CHANGES:
                        json_io.save_json(metadata_path, metadata)


    print("end convert_metadata")

def convert_model_predictions():


    print("start convert_model_predictions")

    for user_path in glob.glob(os.path.join("usr", "data", "*")):
        for farm_path in glob.glob(os.path.join(user_path, "image_sets", "*")):
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                for mission_path in glob.glob(os.path.join(field_path, "*")):

                    metadata_path = os.path.join(mission_path, "metadata", "metadata.json")
                    metadata = json_io.load_json(metadata_path)

                    object_name = metadata["object_classes"][0]
                    


                    prediction_images_dir = os.path.join(mission_path, "model", "prediction", "images")
                    for image_dir in glob.glob(os.path.join(prediction_images_dir, "*")):
                        predictions_path = os.path.join(image_dir, "predictions.json")

                        predictions = json_io.load_json(predictions_path)


                        for image_name in predictions.keys():
                            predictions[image_name]["classes"] = [0] * len(predictions[image_name]["scores"])

                        if WRITE_CHANGES:
                            json_io.save_json(predictions_path, predictions)




                    model_status_path = os.path.join(mission_path, "model", "status.json")
                    model_status = json_io.load_json(model_status_path)

                    if model_status["model_name"] == "random_weights":
                        model_status["model_name"] = "random_weights_1"

                    if WRITE_CHANGES:
                        json_io.save_json(model_status_path, model_status)


                    for result_dir in glob.glob(os.path.join(mission_path, "model", "results", "*")):
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = json_io.load_json(annotations_path)

                        for image_name in annotations.keys():
                            annotations[image_name]["classes"] = [0] * len(annotations[image_name]["boxes"])

                        if WRITE_CHANGES:
                            json_io.save_json(annotations_path, annotations)


                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = json_io.load_json(predictions_path)

                        for image_name in predictions.keys():
                            predictions[image_name]["classes"] = [0] * len(predictions[image_name]["boxes"])

                        if WRITE_CHANGES:
                            json_io.save_json(predictions_path, predictions)



                        full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                        full_predictions = json_io.load_json(full_predictions_path)

                        for image_name in full_predictions.keys():
                            full_predictions[image_name]["classes"] = [0] * len(full_predictions[image_name]["boxes"])

                        if WRITE_CHANGES:
                            json_io.save_json(full_predictions_path, full_predictions)



                        tmp_raw_outputs_path = os.path.join(result_dir, "tmp_raw_outputs")
                        raw_outputs_zip_path = os.path.join(result_dir, "raw_outputs.zip")

                        shutil.unpack_archive(raw_outputs_zip_path, tmp_raw_outputs_path)


                        annotations_path = os.path.join(tmp_raw_outputs_path, "annotations.json")
                        annotations = json_io.load_json(annotations_path)

                        for image_name in annotations.keys():
                            annotations[image_name]["classes"] = [0] * len(annotations[image_name]["annotations"])
                    
                        if WRITE_CHANGES:
                            json_io.save_json(annotations_path, annotations)



                        predictions_path = os.path.join(tmp_raw_outputs_path, "predictions.json")
                        predictions = json_io.load_json(predictions_path)

                        for image_name in predictions.keys():
                            predictions[image_name]["classes"] = [0] * len(predictions[image_name]["predictions"])

                        if WRITE_CHANGES:
                            json_io.save_json(predictions_path, predictions)


                        if WRITE_CHANGES:
                            shutil.make_archive(os.path.join(result_dir, "raw_outputs"), 'zip', tmp_raw_outputs_path)
                        
                        
                        shutil.rmtree(tmp_raw_outputs_path)



                        vegetation_record_path = os.path.join(result_dir, "vegetation_record.json")
                        if os.path.exists(vegetation_record_path):
                            vegetation_record = json_io.load_json(vegetation_record_path)

                            for image_name in vegetation_record.keys():
                                region_keys = list(vegetation_record[image_name]["obj_vegetation_percentage"].keys())
                                vegetation_record[image_name]["obj_vegetation_percentage"][object_name] = {}
                                for region_key in region_keys:
                                    v = vegetation_record[image_name]["obj_vegetation_percentage"][region_key]
                                    vegetation_record[image_name]["obj_vegetation_percentage"][object_name][region_key] = v

                                for region_key in region_keys:
                                    del vegetation_record[image_name]["obj_vegetation_percentage"][region_key]


                            if WRITE_CHANGES:
                                json_io.save_json(vegetation_record_path, vegetation_record)


                        metrics_path = os.path.join(result_dir, "metrics.json")
                        metrics = json_io.load_json(metrics_path)

                        d = {}
                        d[object_name] = metrics

                        if WRITE_CHANGES:
                            json_io.save_json(metrics_path, d)


                        request_path = os.path.join(result_dir, "request.json")
                        request = json_io.load_json(request_path)

                        regions_only = "regions_only" in request and request["regions_only"]
                        
                        if WRITE_CHANGES:
                            inference_metrics.create_spreadsheet(result_dir, regions_only=regions_only)


    print("end convert_model_predictions")


def convert_models():

    print("start convert_models")


    for user_path in glob.glob(os.path.join("usr", "data", "*")):

        model_paths = []
        model_paths.extend(
            list(glob.glob(os.path.join(user_path, "models", "available", "public", "*")))
        )
        model_paths.extend(
            list(glob.glob(os.path.join(user_path, "models", "available", "private", "*")))
        )

        for model_path in model_paths:

            log_path = os.path.join(model_path, "log.json")
            log = json_io.load_json(log_path)

            for image_set in log["image_sets"]:
                image_set["class_mapping"] = {"0": 0}
                del image_set["object_name"]

            old_object_name = log["model_object"]
            if old_object_name == "canola_seedling":
                log["object_classes"] = ["Canola Seedling"]

            elif old_object_name == "wheat_head":
                log["object_classes"] = ["Wheat Head"]

            if WRITE_CHANGES:
                json_io.save_json(log_path, log)


            for user_path in glob.glob(os.path.join(model_path, "annotations", "*")):
                for farm_path in glob.glob(os.path.join(user_path, "*")):
                    for field_path in glob.glob(os.path.join(farm_path, "*")):
                        for mission_path in glob.glob(os.path.join(field_path, "*")):

                            annotations_path = os.path.join(mission_path, "annotations.json")
                            annotations = json_io.load_json(annotations_path)

                            for image_name in annotations.keys():
                                annotations[image_name]["classes"] = [0] * len(annotations[image_name]["boxes"])

                            if WRITE_CHANGES:
                                json_io.save_json(annotations_path, annotations)


    print("end convert_models")


            




def update_shared_data():
    
    print("start update_shared_data")


    public_image_sets_path = os.path.join("usr", "shared", "public_image_sets.json")

    d = json_io.load_json(public_image_sets_path)
    for username in d.keys():
        for farm_name in d[username].keys():
            for field_name in d[username][farm_name].keys():
                for mission_date in d[username][farm_name][field_name].keys():
                    v = d[username][farm_name][field_name][mission_date]
                    z = {}
                    if v["object_name"] == "canola_seedling":
                        z["object_classes"] = ["Canola Seedling"]

                    elif v["object_name"] == "wheat_head":
                        z["object_classes"] = ["Wheat Head"]

                    d[username][farm_name][field_name][mission_date] = z

    if WRITE_CHANGES:
        json_io.save_json(public_image_sets_path, d)



    objects_path = os.path.join("usr", "shared", "objects.json")
    objects = {"object_names": ["Canola Seedling", "Wheat Head"]}

    if WRITE_CHANGES:
        json_io.save_json(objects_path, objects)


    print("Done updating shared data. Remember to copy over the random weights.")




def check_for_blocked_training():

    print("Started checking for blocked training")


    for user_path in glob.glob(os.path.join("usr", "data", "*")):
        for farm_path in glob.glob(os.path.join(user_path, "image_sets", "*")):
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                for mission_path in glob.glob(os.path.join(field_path, "*")):


                    training_dir = os.path.join(mission_path, "model", "training")

                    usr_block_path = os.path.join(training_dir, "usr_block.json")
                    sys_block_path = os.path.join(training_dir, "sys_block.json")
                    if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
                        print(mission_path)



    print("Done checking for blocked training")






def convert():

    check_for_blocked_training()
    exit()


    convert_overlay_colors()
    exit()


    convert_annotations()
    exit()


    convert_metadata()
    exit()


    convert_model_predictions()
    exit()


    convert_models()
    exit()


    update_shared_data()
    exit()



