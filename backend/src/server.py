import logging
import os
import glob
import shutil
import time
import traceback
import threading
import numpy as np

from flask import Flask, request


from io_utils import json_io, tf_record_io

import auxiliary as aux
import extract_patches as ep
from models.common import annotation_utils, inference_metrics
import excess_green
import models.yolov4.driver as yolov4_driver
from job_manager import JobManager
import emit
from image_set import Image


# JOBS_DIR = os.path.join("usr", "shared", "jobs")

# OCCUPIED_SETS = {}
# JOB_QUEUE = []

# job_lock = threading.Lock()
cv = threading.Condition()
# jm = JobManager()
queue = []
occupied_sets = {}

waiting_workers = 0
total_workers = 1

app = Flask(__name__)


REQUIRED_JOB_KEYS = [
    # "uuid",
    "key", 
    "task", 
    "request_time"
]

VALID_TASKS = [
    # "switch",
    "predict",
    "fine_tune",
    "train"
]




def check_job(req):

    for req_key in REQUIRED_JOB_KEYS:
        if not req_key in req:
            raise RuntimeError("Bad request")
        
    if req["task"] not in VALID_TASKS:
        raise RuntimeError("Bad request")
    


# def process_switch(job):

#     logger = logging.getLogger(__name__)


#     try:
#         username = job["username"]
#         farm_name = job["farm_name"]
#         field_name = job["field_name"]
#         mission_date = job["mission_date"]

#         image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
#         model_dir = os.path.join(image_set_dir, "model")

#         status_path = os.path.join(model_dir, "status.json")
#         status = json_io.load_json(status_path)

#         if "error_message" in status:
#             return


#         model_name = job["model_name"]
#         model_creator = job["model_creator"]

#         logger.info("Switching to model {}".format(model_name))
#         emit.set_image_set_status(username, farm_name, field_name, mission_date, 
#                                   {"state_name": emit.SWITCHING_MODELS, "progress": "In Progress"})


#         if model_name.startswith("random_weights") and model_creator == "":
#             model_path = os.path.join("usr", "shared", "weights", model_name)
#             average_patch_size = 416
#         else:

#             model_path = os.path.join("usr", "data", model_creator, "models")
#             public_model_path = os.path.join(model_path, "available", "public", model_name)
#             private_model_path = os.path.join(model_path, "available", "private", model_name)

#             if os.path.exists(public_model_path):
#                 model_path = public_model_path
#             elif os.path.exists(private_model_path):
#                 model_path = private_model_path
#             else:
#                 raise RuntimeError("Model weights could not be located.")
            
#             log_path = os.path.join(model_path, "log.json")
#             try:
#                 log = json_io.load_json(log_path)
#                 average_patch_size = log["average_patch_size"]
#             except Exception as e:
#                 raise RuntimeError("Model log could not be loaded.")

#         weights_path = os.path.join(model_path, "weights.h5")

#         weights_dir = os.path.join(model_dir, "weights")
#         tmp_weights_path = os.path.join(weights_dir, "tmp_weights.h5")
#         best_weights_path = os.path.join(weights_dir, "best_weights.h5")
#         cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")


#         try:
#             shutil.copy(weights_path, tmp_weights_path)
#         except Exception as e:
#             raise RuntimeError("Model weights could not be located.")

#         shutil.move(tmp_weights_path, best_weights_path)
#         shutil.copy(best_weights_path, cur_weights_path)


        
#         loss_record_path = os.path.join(model_dir, "training", "loss_record.json")

#         loss_record = {
#             "training_loss": { "values": [],
#                             "best": 100000000,
#                             "epochs_since_improvement": 100000000}, 
#             "validation_loss": {"values": [],
#                                 "best": 100000000,
#                                 "epochs_since_improvement": 100000000},
#         }
#         json_io.save_json(loss_record_path, loss_record)


#         training_dir = os.path.join(model_dir, "training")
#         training_records_dir = os.path.join(training_dir, "training_tf_records")
#         if os.path.exists(training_records_dir):
#             shutil.rmtree(training_records_dir)
#             os.makedirs(training_records_dir)


#         patches_dir = os.path.join(image_set_dir, "model", "patches")

#         shutil.rmtree(patches_dir)
#         os.makedirs(patches_dir)


#         status_path = os.path.join(model_dir, "status.json")
#         status = json_io.load_json(status_path)
#         # status["num_regions_fully_trained_on"] = 0
#         status["model_creator"] = model_creator
#         status["model_name"] = model_name
#         status["patch_size"] = round(average_patch_size)
#         json_io.save_json(status_path, status)

#         # emit.set_image_set_status(username, farm_name, field_name, mission_date, emit.FINISHED_SWITCHING_MODELS)

#         emit.set_image_set_status(username, farm_name, field_name, mission_date,  {"state_name": emit.IDLE})

#     except Exception as e:

#         trace = traceback.format_exc()
#         logger.error("Exception occurred in process_switch")
#         logger.error(e)
#         logger.error(trace)

#         emit.set_image_set_status(username, farm_name, field_name, mission_date, 
#                                   {"state_name": emit.IDLE, 
#                                    "error_setting": "switching models", 
#                                    "error_message": str(e)})


def set_enqueued_state(job):
    
    username = job["username"]
    farm_name = job["farm_name"]
    field_name = job["field_name"]
    mission_date = job["mission_date"]

    # if job["task"] == "switch":
    #     state = emit.SWITCHING_MODELS
    if job["task"] == "predict":
        state = emit.PREDICTING
    else:
        state = emit.FINE_TUNING

    emit.set_image_set_status(username, farm_name, field_name, mission_date, {"state_name": state, "progress": "Enqueued"})



@app.route(os.environ.get("AC_PATH") + '/health_request', methods=['POST'])
def health_request():
    return {"message": "alive"}



@app.route(os.environ.get("AC_PATH") + '/get_num_workers', methods=['POST'])
def get_num_workers():
    return {"num_workers": str(waiting_workers)}

@app.route(os.environ.get("AC_PATH") + '/add_request', methods=['POST'])
def add_request():
    
    logger = logging.getLogger(__name__)
    logger.info("POST to add_request")
    content_type = request.headers.get('Content-Type')

    if (content_type == 'application/json'):
        job = request.json
        logger.info("Got request: {}".format(job))

        # if "uuid" not in req:
        #     return {"message": "bad_request"}
        
        # request_uuid = req["uuid"]
        # job_path = os.path.join(JOBS_DIR, request_uuid + ".json")
        # job = json_io.load_json(job_path)
        

        try:
            check_job(job)
        except RuntimeError:
            return {"message": "Job is malformed."}
        
        # if jm.is_occupied(job["key"]):
        occupied = False
        with cv:
            if job["key"] in occupied_sets:
                occupied = True
        if occupied:
            return {"message": "The job cannot be enqueued due to an existing job that has not yet been processed."}
        
        if job["task"] != "train":
            try:
                set_enqueued_state(job)
            except Exception:
                return {"message": "Failed to set enqueued job state."}


        # if job["task"] == "switch":
        #     process_switch(job)
        # else:
        with cv:
            # jm.add_job(job)
            occupied_sets[job["key"]] = job
            queue.append(job["key"])
            cv.notify()



        return {"message": "ok"}


    else:
        return {"message": 'Content-Type not supported!'}
        

# @app.route(os.environ.get("AC_PATH") + '/check_if_occupied', methods=['POST'])
# def check_if_occupied():

        
#     logger = logging.getLogger(__name__)
#     logger.info("POST to add_request")
#     content_type = request.headers.get('Content-Type')

#     if (content_type == 'application/json'):
#         job_req = request.json
#         logger.info("Got request: {}".format(job_req))


#         if "key" not in job_req:
#             return {"message": "bad_request"}
        
#         if jm.is_occupied(job_req["key"]):
#             occupied = "yes"

#         else:
#             occupied = "no"

#         return {"is_occupied": occupied}


#     else:
#         return {"message": 'Content-Type not supported!'}




# def collect_jobs():


#     cur_jobs = []


#     job_paths = glob.glob(os.path.join(JOBS_DIR, "*"))

#     for job_path in job_paths:

#         job = json_io.load_json(job_path)
#         try:
#             check_job(job)
#             cur_jobs.append(job)
#         except RuntimeError:
#             os.remove(job_path)


#     cur_jobs.sort(key=lambda x: x["request_time"])

#     for job in cur_jobs:

#         jm.add_job(job)







def job_available():
    return len(queue) > 0

    # return jm.size() > 0




def create_vegetation_record(image_set_dir, excess_green_record, annotations, predictions):


    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info("Starting to calculate vegetation percentages...")

    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    if metadata["is_ortho"] == "yes":
        vegetation_record = excess_green.create_vegetation_record_for_orthomosaic(image_set_dir, excess_green_record, metadata, annotations, predictions)
    else:
        vegetation_record = excess_green.create_vegetation_record_for_image_set(image_set_dir, excess_green_record, metadata, annotations, predictions)

    end_time = time.time()
    elapsed = round(end_time - start_time, 2)
    logger.info("Finished calculating vegetation percentages. Took {} seconds.".format(elapsed))

    return vegetation_record





def collect_results(job):

    username = job["username"]
    farm_name = job["farm_name"]
    field_name = job["field_name"]
    mission_date = job["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)

    model_dir = os.path.join(image_set_dir, "model")
    results_dir = os.path.join(model_dir, "results", "available", job["uuid"])


    full_predictions_path = os.path.join(results_dir, "full_predictions.json")
    full_predictions = json_io.load_json(full_predictions_path)

    predictions_path = os.path.join(results_dir, "predictions.json")
    predictions = json_io.load_json(predictions_path)


    annotations_src_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_src_path)


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                               {"state_name": emit.PREDICTING, "progress": "Collecting Metrics"})
    metrics = inference_metrics.collect_image_set_metrics(predictions, annotations, metadata)

    
    metrics_path = os.path.join(results_dir, "metrics.json")
    json_io.save_json(metrics_path, metrics)

    excess_green_record_src_path = os.path.join(image_set_dir, "excess_green", "record.json")
    excess_green_record = json_io.load_json(excess_green_record_src_path)


    calculate_vegetation_coverage = "calculate_vegetation_coverage" in job and job["calculate_vegetation_coverage"]

    if calculate_vegetation_coverage:
        emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                                   {"state_name": emit.PREDICTING, "progress": "Calculating Vegetation Coverage"})
        vegetation_record = create_vegetation_record(image_set_dir, excess_green_record, annotations, predictions)

        results_vegetation_record_path = os.path.join(results_dir, "vegetation_record.json")
        json_io.save_json(results_vegetation_record_path, vegetation_record)


    excess_green_record_dst_path = os.path.join(results_dir, "excess_green_record.json")
    json_io.save_json(excess_green_record_dst_path, excess_green_record)

    tags_src_path = os.path.join(image_set_dir, "annotations", "tags.json")
    tags_dst_path = os.path.join(results_dir, "tags.json")
    shutil.copy(tags_src_path, tags_dst_path)
    

    annotations_dst_path = os.path.join(results_dir, "annotations.json")
    annotation_utils.save_annotations(annotations_dst_path, annotations)


    regions_only = "regions_only" in job and job["regions_only"]

    
    inference_metrics.create_spreadsheet(results_dir, regions_only=regions_only)




    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)

    if inference_metrics.can_calculate_density(metadata, camera_specs):
        emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                                   {"state_name": emit.PREDICTING, "progress": "Calculating Voronoi Areas"})
        inference_metrics.create_areas_spreadsheet(results_dir, regions_only=regions_only)

    raw_outputs_dir = os.path.join(results_dir, "raw_outputs")
    os.makedirs(raw_outputs_dir)

    downloadable_annotations = {}
    annotation_keys = [
        "boxes",
        "classes",
        "regions_of_interest",
        "fine_tuning_regions",
        "test_regions"
    ]


    for image_name in annotations.keys():
        downloadable_annotations[image_name] = {}
        for key in annotation_keys:

            downloadable_annotations[image_name][key] = []

            if key == "classes":
                downloadable_annotations[image_name][key] = annotations[image_name][key]

            elif key == "regions_of_interest":

                for poly in annotations[image_name]["regions_of_interest"]:
                    download_poly = []
                    for coord in poly:
                        download_poly.append([int(coord[1]), int(coord[0])])
                    downloadable_annotations[image_name][key].append(download_poly)
            
            else:
                for box in annotations[image_name][key]:
                    download_box = [
                        int(box[1]),
                        int(box[0]),
                        int(box[3]),
                        int(box[2])
                    ]
                    downloadable_annotations[image_name][key].append(download_box)
    
    json_io.save_json(os.path.join(raw_outputs_dir, "annotations.json"), downloadable_annotations)


    downloadable_predictions = {}
    for image_name in full_predictions.keys():
        downloadable_predictions[image_name] = {}

        downloadable_predictions[image_name]["predictions"] = []
        for box in full_predictions[image_name]["boxes"]:
            download_box = [
                int(box[1]),
                int(box[0]),
                int(box[3]),
                int(box[2])
            ]
            downloadable_predictions[image_name]["predictions"].append(download_box)

        downloadable_predictions[image_name]["confidence_scores"] = []
        for score in full_predictions[image_name]["scores"]:
            downloadable_predictions[image_name]["confidence_scores"].append(float(score))


        downloadable_predictions[image_name]["classes"] = full_predictions[image_name]["classes"]


    json_io.save_json(os.path.join(raw_outputs_dir, "predictions.json"), downloadable_predictions)

    shutil.make_archive(os.path.join(results_dir, "raw_outputs"), 'zip', raw_outputs_dir)
    shutil.rmtree(raw_outputs_dir)

    return





def process_predict(job):

    logger = logging.getLogger(__name__)

    try:

        username = job["username"]
        farm_name = job["farm_name"]
        field_name = job["field_name"]
        mission_date = job["mission_date"]

        image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)

        model_dir = os.path.join(image_set_dir, "model")


        status_path = os.path.join(model_dir, "status.json")
        status = json_io.load_json(status_path)

        if status["error_message"] != "":
            raise RuntimeError("Cannot run prediction during errored state.")


        logger.info("Starting to predict for {}".format(job["key"]))
        emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                                   {"state_name": emit.PREDICTING, "progress": "0% complete"})

        yolov4_driver.predict(job)

        if job["save_result"]:
            
            results_dir = os.path.join(model_dir, "results", "available", job["uuid"])
            saved_request_path = os.path.join(results_dir, "request.json")
            json_io.save_json(saved_request_path, job)

            collect_results(job)

            job = json_io.load_json(saved_request_path)
            end_time = int(time.time())
            job["end_time"] = end_time
            json_io.save_json(saved_request_path, job)

            emit.emit_results_change(username, farm_name, field_name, mission_date)



        logger.info("Finished predicting for {}".format(job["key"]))

        # emit.set_image_set_status(username, farm_name, field_name, mission_date, 
        #                            {"state_name": emit.PREDICTING, 
        #                             "progress": "Complete",
        #                             })

        with cv:
            del occupied_sets[job["key"]]


        emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                                  {"state_name": emit.IDLE,
                                   "prediction_image_names": ",".join(job["image_names"])})



    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_predict")
        logger.error(e)
        logger.error(trace)

        with cv:
            if job["key"] in occupied_sets:
                del occupied_sets[job["key"]]

        emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                                    {"state_name": emit.PREDICTING, "error_message": str(e)})

        if job["save_result"]:
            if results_dir is not None and os.path.exists(results_dir):
                shutil.rmtree(results_dir)


            aborted_dir = os.path.join(model_dir, "results", "aborted", job["uuid"])
            os.makedirs(aborted_dir)
            job["aborted_time"] = int(time.time())
            job["error_message"] = str(e)
            job["error_info"] = str(trace)
            aborted_job_path = os.path.join(aborted_dir, "request.json")
            json_io.save_json(aborted_job_path, job)


            emit.emit_results_change(username, farm_name, field_name, mission_date)





def process_fine_tune(job):

    logger = logging.getLogger(__name__)

    try:

        username = job["username"]
        farm_name = job["farm_name"]
        field_name = job["field_name"]
        mission_date = job["mission_date"]

        image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
        model_dir = os.path.join(image_set_dir, "model")

        status_path = os.path.join(model_dir, "status.json")
        status = json_io.load_json(status_path)

        
        if status["error_message"] != "":
            raise RuntimeError("Cannot run fine-tuning during errored state.")
        

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        num_fine_tuning_regions = annotation_utils.get_num_fine_tuning_regions(annotations)
        if num_fine_tuning_regions > 0:

            updated_patch_size = ep.update_model_patch_size(image_set_dir, annotations, ["fine_tuning_regions"])
            ep.update_training_patches(image_set_dir, annotations, updated_patch_size)

            aux.update_training_tf_records(image_set_dir, annotations)
            aux.reset_loss_record(image_set_dir)

            yolov4_driver.fine_tune(job)


        with cv:
            del occupied_sets[job["key"]]

        emit.set_image_set_status(username, farm_name, field_name, mission_date, {"state_name": emit.IDLE})

    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_fine_tune")
        logger.error(e)
        logger.error(trace)


        with cv:
            if job["key"] in occupied_sets:
                del occupied_sets[job["key"]]


        emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                                  {"state_name": emit.FINE_TUNING, "error_message": str(e)})

    

        
def process_train(job):


    logger = logging.getLogger(__name__)
    try:

        username = job["username"]
        model_name = job["model_name"]

        usr_dir = os.path.join("usr", "data", username)
        models_dir = os.path.join(usr_dir, "models")
        pending_dir = os.path.join(models_dir, "pending")
        baseline_pending_dir = os.path.join(pending_dir, model_name)
        log_path = os.path.join(baseline_pending_dir, "log.json")

        available_dir = os.path.join(models_dir, "available")
        if job["public"] == "yes":
            baseline_available_dir = os.path.join(available_dir, "public", model_name)
        else:
            baseline_available_dir = os.path.join(available_dir, "private", model_name)

        aborted_dir = os.path.join(models_dir, "aborted")
        baseline_aborted_dir = os.path.join(aborted_dir, model_name)

        if os.path.exists(baseline_available_dir) or os.path.exists(baseline_aborted_dir):
            if os.path.exists(baseline_available_dir):
                logger.info("Not training baseline: baseline_available_dir exists")
            else:
                logger.info("Not training baseline: baseline_aborted_dir exists")
            return

        logging.info("Starting to train baseline {}".format(model_name))

        patches_dir = os.path.join(baseline_pending_dir, "patches")
        annotations_dir = os.path.join(baseline_pending_dir, "annotations")
        model_dir = os.path.join(baseline_pending_dir, "model")
        training_dir = os.path.join(model_dir, "training")
        weights_dir = os.path.join(model_dir, "weights")

        log = json_io.load_json(log_path)

        if len(log["image_sets"]) == 0:
            raise RuntimeError("Image set list in model log file is empty.")

        
        if len(log["object_classes"]) == 0:
            raise RuntimeError("Class list in model log file is empty.")

        if "training_start_time" not in log:
            os.makedirs(patches_dir)
            os.makedirs(annotations_dir)
            os.makedirs(model_dir)
            os.makedirs(training_dir)
            os.makedirs(weights_dir)


            all_records = []
            for image_set_index, image_set in enumerate(log["image_sets"]):
                print("now_processing", image_set)

                username = image_set["username"]
                farm_name = image_set["farm_name"]
                field_name = image_set["field_name"]
                mission_date = image_set["mission_date"]
                logger.info("Baseline: Preparing patches from {} {} {} {}".format(
                    username, farm_name, field_name, mission_date))
                
                image_set_dir = os.path.join("usr", "data", username, "image_sets", 
                                            farm_name, field_name, mission_date)
                images_dir = os.path.join(image_set_dir, "images")

                metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
                metadata = json_io.load_json(metadata_path)
                is_ortho = metadata["is_ortho"] == "yes"

                annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)


                num_annotations = annotation_utils.get_num_annotations(annotations, ["fine_tuning_regions", "test_regions"])

                if num_annotations > 0:
                    average_box_area = annotation_utils.get_average_box_area(annotations, region_keys=["fine_tuning_regions", "test_regions"], measure="mean")
                    average_box_height = annotation_utils.get_average_box_height(annotations, region_keys=["fine_tuning_regions", "test_regions"], measure="mean")
                    average_box_width = annotation_utils.get_average_box_width(annotations, region_keys=["fine_tuning_regions", "test_regions"], measure="mean")
                    patch_size = annotation_utils.average_box_area_to_patch_size(average_box_area)
                else:
                    average_box_area = "NA"
                    average_box_height = "NA"
                    average_box_width = "NA"
                    patch_size = 416

                if "patch_size" in image_set:
                    patch_size = image_set["patch_size"]

                log["image_sets"][image_set_index]["num_annotations"] = num_annotations
                log["image_sets"][image_set_index]["average_box_area"] = average_box_area
                log["image_sets"][image_set_index]["average_box_height"] = average_box_height
                log["image_sets"][image_set_index]["average_box_width"] = average_box_width
                log["image_sets"][image_set_index]["patch_size"] = patch_size

                
                logger.info("Patch size: {} px".format(patch_size))

                for image_name in annotations.keys():
                    
                    if "taken_regions" in image_set:
                        if image_name in image_set["taken_regions"]:
                            regions = image_set["taken_regions"][image_name]
                        else:
                            regions = []
                    else:
                        regions = annotations[image_name]["fine_tuning_regions"] + annotations[image_name]["test_regions"]
                    
                    if "patch_overlap_percent" in image_set:
                        patch_overlap_percent = image_set["patch_overlap_percent"]
                    else:
                        patch_overlap_percent = 50

                    if "class_mapping" in image_set:

                        class_mapping = {int(k): v for k, v in image_set["class_mapping"].items()}
                        mask = np.isin(annotations[image_name]["classes"], list(class_mapping.keys()))
                        annotations[image_name]["boxes"] = annotations[image_name]["boxes"][mask]
                        annotations[image_name]["classes"] = annotations[image_name]["classes"][mask]
                        
                        for i in range(len(annotations[image_name]["classes"])):
                            annotations[image_name]["classes"][i] = class_mapping[annotations[image_name]["classes"][i]]
                        

                    if len(regions) > 0:

                        image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
                        image = Image(image_path)
                        patch_records = ep.extract_patch_records_from_image_tiled(
                            image,
                            patch_size,
                            image_annotations=annotations[image_name],
                            patch_overlap_percent=patch_overlap_percent,
                            regions=regions,
                            is_ortho=is_ortho,
                            include_patch_arrays=False,
                            out_dir=patches_dir)

                        all_records.extend(patch_records)

                image_set_annotations_dir = os.path.join(annotations_dir, 
                                                username, 
                                                farm_name,
                                                field_name,
                                                mission_date)
                os.makedirs(image_set_annotations_dir, exist_ok=True)
                image_set_annotations_path = os.path.join(image_set_annotations_dir, "annotations.json")
                annotation_utils.save_annotations(image_set_annotations_path, annotations)

            average_box_areas = []
            average_box_heights = []
            average_box_widths = []
            patch_sizes = []
            for i in range(len(log["image_sets"])):
                average_box_area = log["image_sets"][i]["average_box_area"]
                average_box_height = log["image_sets"][i]["average_box_height"]
                average_box_width = log["image_sets"][i]["average_box_width"]
                patch_size = log["image_sets"][i]["patch_size"]
                if not isinstance(average_box_area, str):
                    average_box_areas.append(average_box_area)
                    average_box_heights.append(average_box_height)
                    average_box_widths.append(average_box_width)
                patch_sizes.append(patch_size)

            if len(average_box_areas) > 0:
                log["average_box_area"] = np.mean(average_box_areas)
                log["average_box_height"] = np.mean(average_box_heights)
                log["average_box_width"] = np.mean(average_box_widths)
            else:
                log["average_box_area"] = "NA"
                log["average_box_height"] = "NA"
                log["average_box_width"] = "NA"
            log["average_patch_size"] = np.mean(patch_sizes)

            patch_records = np.array(all_records)

            training_patch_records = patch_records

            print("have {} patch records".format(patch_records.size))

            training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
            training_patches_record_path = os.path.join(training_dir, "training-patches-record.tfrec")
            tf_record_io.output_patch_tf_records(training_patches_record_path, training_tf_records)

            loss_record = {
                "training_loss": { "values": [] }
            }
            loss_record_path = os.path.join(baseline_pending_dir, "model", "training", "loss_record.json")
            json_io.save_json(loss_record_path, loss_record)

            aux.reset_loss_record(baseline_pending_dir)

            log["training_start_time"] = int(time.time())
            json_io.save_json(log_path, log)


        yolov4_driver.train(job)


        log["training_end_time"] = int(time.time())
        json_io.save_json(log_path, log)
        
        shutil.move(os.path.join(weights_dir, "best_weights.h5"),
                    os.path.join(baseline_pending_dir, "weights.h5"))
        
        shutil.move(os.path.join(training_dir, "loss_record.json"),
                    os.path.join(baseline_pending_dir, "loss_record.json"))

        shutil.rmtree(patches_dir)
        shutil.rmtree(model_dir)

        shutil.move(baseline_pending_dir, baseline_available_dir)

        with cv:
            del occupied_sets[job["key"]]


        emit.emit_model_change(job["username"])


    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_baseline")
        logger.error(e)
        logger.error(trace)

        with cv:
            if job["key"] in occupied_sets:
                del occupied_sets[job["key"]]

        if baseline_pending_dir is not None:

            log["aborted_time"] = int(time.time())
            log["error_message"] = str(e)

            os.makedirs(baseline_aborted_dir)
            json_io.save_json(os.path.join(baseline_aborted_dir, "log.json"), log)

            if os.path.exists(baseline_pending_dir):
                saved_pending_dir = os.path.join(baseline_aborted_dir, "saved_pending")
                shutil.move(baseline_pending_dir, saved_pending_dir)

            if os.path.exists(baseline_available_dir):
                saved_available_dir = os.path.join(baseline_aborted_dir, "saved_available")
                shutil.move(baseline_available_dir, saved_available_dir)

        emit.emit_model_change(job["username"])

def process_job(job_key):

    logger = logging.getLogger(__name__)
    try:
        job = occupied_sets[job_key]
        # job = jm.get_job(job_key)

        task = job["task"]
        # job_uuid = job["uuid"]
        # job_path = os.path.join(JOBS_DIR, job_uuid + ".json")

        if task == "predict":
            process_predict(job)

        elif task == "fine_tune":
            process_fine_tune(job)

        elif task == "train":
            process_train(job)

        else:
            logger.error("Unrecognized task", task)



        # os.remove(job_path)
        # jm.deoccupy(job_key)

    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_job")
        logger.error(e)
        logger.error(trace)

        with cv:
            if job_key in occupied_sets:
                del occupied_sets[job_key]
        # if job_path is not None and os.path.exists(job_path):
        #     os.remove(job_path)
        
        # jm.deoccupy(job_key)





def work():
    global waiting_workers

    while True:
        with cv:
            waiting_workers += 1
            emit.emit_worker_change(waiting_workers)
            cv.wait_for(job_available)
            # job_key = jm.dequeue()
            job_key = queue.pop(0)
            waiting_workers -= 1
            emit.emit_worker_change(waiting_workers)
        print("worker {} processing job {}".format(threading.current_thread().name, job_key))

        process_job(job_key)







if __name__ == "__main__":

    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    # exit()

    # # gpus = None


    # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


    logging.basicConfig(level=logging.INFO)

    # collect_jobs()

    # producer = threading.Thread(name="producer", target=produce)
    # producer.start()



    worker = threading.Thread(name="worker1", target=work)
    worker.start()


    # worker = threading.Thread(name="worker2", target=work)
    # worker.start()

    app.run(host=os.environ.get("AC_IP"), port=os.environ.get("AC_PY_PORT"))