import os
import glob
import shutil
import requests
import logging
import time
import traceback


from io_utils import json_io
from models.common import annotation_utils
import auto_select
#import image_set_model as ism

base_url = "https://" + os.environ.get("AC_IP") + ":" + os.environ.get("AC_PORT") + os.environ.get("AC_PATH")
status_notification_url = base_url + "/status_notification"
results_notification_url = base_url + "/results_notification"
model_notification_url = base_url + "/model_notification"

# INITIALIZING = "initializing"
IDLE = "Idle"
FINE_TUNING = "Fine-Tuning"
FINISHED_FINE_TUNING = "Finished Fine-Tuning"
# FINISHED_TRAINING = "Finished Training"
PREDICTING = "Predicting"
FINISHED_PREDICTING = "Finished Predicting"

COLLECTING_METRICS = "Collecting Metrics"
CALCULATING_VEGETATION_COVERAGE = "Calculating Vegetation Coverage"
CALCULATING_VORONOI_AREAS = "Calculating Voronoi Areas"

SWITCHING_MODELS = "Switching Models"
FINISHED_SWITCHING_MODELS = "Finished Switching Models"

SELECTING_MODEL = "Selecting Model"
FINISHED_SELECTING_MODEL = "Finished Selecting Model"
# RESTARTING = "Restarting"
# FINISHED_RESTARTING = "Finished Restarting"
# DETERMINING_PATCH_SIZE = "Determining Patch Size"
# TRAINING_BASELINE = "Training Baseline"
TRAINING = "Training"





def set_scheduler_status(username, farm_name, field_name, mission_date, status, extra_items={}):

    scheduler_status_path = os.path.join("usr", "shared", "scheduler_status.json")
    scheduler_status = json_io.load_json(scheduler_status_path)
        
    current_update_timestamp = int(time.time())
    update_num = scheduler_status["update_num"] + 1
    # seconds_since_last_update = current_update_timestamp - scheduler_status["timestamp"]
    scheduler_status = {
        "update_num": update_num,
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
        "status": status,
        "timestamp": current_update_timestamp
        # "seconds_since_last_update": seconds_since_last_update
    }

    for k, v in extra_items.items():
        scheduler_status[k] = v

    json_io.save_json(scheduler_status_path, scheduler_status)

    emit_scheduler_status_change(scheduler_status)





def emit_model_change(username):

    data = {
        "username": username
    }

    emit(model_notification_url, data)


def emit_scheduler_status_change(data):

    #data = {}

    emit(status_notification_url, data)

def emit_results_change(username, farm_name, field_name, mission_date):

    data = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
    }

    emit(results_notification_url, data)


# def notify(username, farm_name, field_name, mission_date, error=False, extra_items={}, results_notification=False):
#     #print("sending data", data)

#     logger = logging.getLogger(__name__)

#     status_path = os.path.join("usr", "data", username, "image_sets",
#                                 farm_name, field_name, mission_date, 
#                                 "model", "status.json")
#     status = json_io.load_json(status_path)

#     data = {
#         "username": username,
#         "farm_name": farm_name,
#         "field_name": field_name,
#         "mission_date": mission_date,
#         "error": error
#     }
#     for k, v in status.items():
#         data[k] = v
#     for k, v in extra_items.items():
#         data[k] = v

#     # print(data)

#     if results_notification:
#         url = results_notification_url
#     else:
#         url = status_notification_url
#     response = requests.post(url, data=data)
#     response.raise_for_status()  # raises exception when not a 2xx response
#     if response.status_code != 200:
#         logger.error("Response status code is not 200. Status code: {}".format(response.status_code))
#         response = response.json()
#         logger.error(response)

#     response = response.json()
#     if response["message"] != "received":
#         logger.error("Response message is not 'received'.")
#         logger.error(response)
#         #exit()


def emit(url, data):
    logger = logging.getLogger(__name__)

    logger.info("Emitting {} to {}".format(data, url))


    response = requests.post(url, data=data, verify=False)
    response.raise_for_status()  # raises exception when not a 2xx response
    if response.status_code != 200:
        logger.error("Response status code is not 200. Status code: {}".format(response.status_code))
        response = response.json()
        logger.error(response)

    response = response.json()
    if response["message"] != "received":
        logger.error("Response message is not 'received'.")
        logger.error(response)


# def check_for_predictions():

#     for username_path in glob.glob(os.path.join("usr", "data", "*")):
#         username = os.path.basename(username_path)
#         for user_dir in glob.glob(os.path.join(username_path, "*")):
#             if os.path.basename(user_dir) == "image_sets":
#                 for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
#                     farm_name = os.path.basename(farm_path)
#                     for field_path in glob.glob(os.path.join(farm_path, "*")):
#                         field_name = os.path.basename(field_path)
#                         for mission_path in glob.glob(os.path.join(field_path, "*")):
#                             mission_date = os.path.basename(mission_path)

#                             upload_status_path = os.path.join(mission_path, "upload_status.json")

#                             if os.path.exists(upload_status_path):
#                                 upload_status = json_io.load_json(upload_status_path)
#                                 if upload_status["status"] == "uploaded":

#                                     model_dir = os.path.join("usr", "data", username, "image_sets",
#                                                             farm_name, field_name, mission_date, 
#                                                             "model")
#                                     prediction_dir = os.path.join(model_dir, "prediction")
#                                     prediction_requests_dirs = [
#                                         os.path.join(prediction_dir, "image_requests"),
#                                         os.path.join(prediction_dir, "image_set_requests", "pending")
#                                     ]

#                                     for prediction_requests_dir in prediction_requests_dirs:
#                                         prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
#                                         if len(prediction_request_paths) > 0:
#                                             return True

#     return False


# def check_restart(farm_name=None, field_name=None, mission_date=None):
#     restart_requests_dir = os.path.join("usr", "requests", "restart")

#     restart_request_paths = glob.glob(os.path.join(restart_requests_dir, "*"))

#     ret = False
#     while len(restart_request_paths) > 0:
#         restart_request_path = restart_request_paths[0]

#         restart_request = json_io.load_json(restart_request_path)

#         req_farm_name = restart_request["farm_name"]
#         req_field_name = restart_request["field_name"]
#         req_mission_date = restart_request["mission_date"]

#         if req_farm_name == farm_name and \
#            req_field_name == field_name and \
#            req_mission_date == mission_date:
#             ret = True

#         print("restarting {}::{}::{}".format(req_farm_name, req_field_name, req_mission_date))

#         image_set_dir = os.path.join("usr", "data", "image_sets",
#                                      req_farm_name, req_field_name, req_mission_date)

#         status_path = os.path.join(image_set_dir, "model", "status.json")
#         status = json_io.load_json(status_path)
#         status["status"] = IDLE
#         status["num_images_fully_trained_on"] = 0
#         status["update_num"] = status["update_num"] + 1
#         json_io.save_json(status_path, status)

#         baseline_weights_path = os.path.join("usr", "data", "weights", "default_weights.h5")
#         cur_weights_path = os.path.join(image_set_dir, "model", "weights", "cur_weights.h5")
#         best_weights_path = os.path.join(image_set_dir, "model", "weights", "best_weights.h5")
#         shutil.copyfile(baseline_weights_path, cur_weights_path)
#         shutil.copyfile(baseline_weights_path, best_weights_path)

#         prediction_images_dir = os.path.join(image_set_dir, "model", "prediction", "images")
#         if os.path.exists(prediction_images_dir):
#             shutil.rmtree(prediction_images_dir)
#             os.makedirs(prediction_images_dir)

#         annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
#         annotations = json_io.load_json(annotations_path)
#         for image_name in annotations.keys():
#             if annotations[image_name]["status"] == "completed_for_training":
#                 annotations[image_name]["status"] = "completed_for_testing"
#         json_io.save_json(annotations_path, annotations)


#         loss_record_path = os.path.join(image_set_dir, "model", "training", "loss_record.json")
#         loss_record = {
#                 "training_loss": { "values": [],
#                                    "best": 100000000,
#                                    "epochs_since_improvement": 100000000}, 
#                 "validation_loss": {"values": [],
#                                     "best": 100000000,
#                                     "epochs_since_improvement": 100000000},
#                 "num_training_images": 0
#         }
#         json_io.save_json(loss_record_path, loss_record)

#         notify(req_farm_name, req_field_name, req_mission_date, extra_items={"restarted": True})

#         os.remove(restart_request_path)
#         restart_request_paths = glob.glob(os.path.join(restart_requests_dir, "*"))

#     return ret




def process_switch(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    switch_req_path = os.path.join(model_dir, "switch_request.json")
    if os.path.exists(switch_req_path):
        try:

            switch_req = json_io.load_json(switch_req_path)

            # if switch_req["auto"]:
            # set_scheduler_status(username, farm_name, field_name, mission_date, SELECTING_MODEL)
            # model_creator, model_name = auto_select.auto_select_model(item)
            # else:
            model_creator = switch_req["model_creator"]
            model_name = switch_req["model_name"]

            logger.info("Switching {}".format(item))
            set_scheduler_status(username, farm_name, field_name, mission_date, SWITCHING_MODELS)

            if model_name.startswith("random_weights") and model_creator == "":
                model_path = os.path.join("usr", "shared", "weights", model_name)
                average_patch_size = 416
            else:

                model_path = os.path.join("usr", "data", model_creator, "models")
                public_model_path = os.path.join(model_path, "available", "public", model_name) #, "weights.h5")
                private_model_path = os.path.join(model_path, "available", "private", model_name) #, "weights.h5")

                if os.path.exists(public_model_path):
                    model_path = public_model_path
                elif os.path.exists(private_model_path):
                    model_path = private_model_path
                else:
                    raise RuntimeError("Model weights could not be located.")
                
                log_path = os.path.join(model_path, "log.json")
                try:
                    log = json_io.load_json(log_path)
                    average_patch_size = log["average_patch_size"]
                except Exception as e:
                    raise RuntimeError("Model log could not be loaded.")

            weights_path = os.path.join(model_path, "weights.h5")


            weights_dir = os.path.join(model_dir, "weights")
            tmp_weights_path = os.path.join(weights_dir, "tmp_weights.h5")
            best_weights_path = os.path.join(weights_dir, "best_weights.h5")
            cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")

            try:
                shutil.copy(weights_path, tmp_weights_path)
            except Exception as e:
                raise RuntimeError("Model weights could not be located.")

            shutil.move(tmp_weights_path, best_weights_path)
            shutil.copy(best_weights_path, cur_weights_path)


            
            loss_record_path = os.path.join(model_dir, "training", "loss_record.json")

            loss_record = {
                "training_loss": { "values": [],
                                "best": 100000000,
                                "epochs_since_improvement": 100000000}, 
                "validation_loss": {"values": [],
                                    "best": 100000000,
                                    "epochs_since_improvement": 100000000},
            }
            json_io.save_json(loss_record_path, loss_record)

            
            

            # default_weights_path = os.path.join("usr", "shared", "weights", "default_weights.h5")

            # shutil.copy(default_weights_path, os.path.join(weights_dir, "cur_weights.h5"))
            # shutil.copy(default_weights_path, os.path.join(weights_dir, "best_weights.h5"))


            # results_dir = os.path.join(model_dir, "results")
            # results = glob.glob(os.path.join(results_dir, "*"))
            # for result in results:
            #     shutil.rmtree(result)

            # prediction_dir = os.path.join(model_dir, "prediction")
            # shutil.rmtree(prediction_dir)
            # os.makedirs(prediction_dir)

            # os.makedirs(os.path.join(prediction_dir, "image_requests"))
            # os.makedirs(os.path.join(prediction_dir, "images"))
            # image_set_requests = os.path.join(prediction_dir, "image_set_requests")
            # os.makedirs(image_set_requests)
            # os.makedirs(os.path.join(image_set_requests, "aborted"))
            # os.makedirs(os.path.join(image_set_requests, "pending"))

            prediction_dir = os.path.join(model_dir, "prediction")

            image_requests_dir = os.path.join(prediction_dir, "image_requests")
            image_set_requests_dir =  os.path.join(prediction_dir, "image_set_requests", "pending")

            shutil.rmtree(image_requests_dir)
            os.makedirs(image_requests_dir)
            shutil.rmtree(image_set_requests_dir)
            os.makedirs(image_set_requests_dir)


            training_dir = os.path.join(model_dir, "training")
            training_records_dir = os.path.join(training_dir, "training_tf_records")
            if os.path.exists(training_records_dir):
                shutil.rmtree(training_records_dir)
                os.makedirs(training_records_dir)
            validation_records_dir = os.path.join(training_dir, "validation_tf_records")
            if os.path.exists(validation_records_dir):
                shutil.rmtree(validation_records_dir)
                os.makedirs(validation_records_dir)

            patches_dir = os.path.join(image_set_dir, "patches")
            # patch_size_estimate_record_path = os.path.join(patches_dir, "patch_size_estimate_record.json")
            # patch_size_estimate_record = None
            # if os.path.exists(patch_size_estimate_record_path):
                # patch_size_estimate_record = json_io.load_json(patch_size_estimate_record_path)

            shutil.rmtree(patches_dir)
            os.makedirs(patches_dir)

            # if patch_size_estimate_record is not None:
                # json_io.save_json(patch_size_estimate_record_path, patch_size_estimate_record)


            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)
            for image_name in annotations.keys():
                # for region in annotations[image_name]["training_regions"]:
                annotations[image_name]["test_regions"].extend(annotations[image_name]["training_regions"])
                annotations[image_name]["training_regions"] = []
                # if annotations[image_name]["status"] == "completed_for_training":
                #     annotations[image_name]["status"] = "completed_for_testing"
            annotation_utils.save_annotations(annotations_path, annotations)


            status_path = os.path.join(model_dir, "status.json")
            status = json_io.load_json(status_path)
            status["num_regions_fully_trained_on"] = 0
            status["model_creator"] = model_creator
            status["model_name"] = model_name
            status["patch_size"] = round(average_patch_size)
            json_io.save_json(status_path, status)

            usr_block_path = os.path.join(training_dir, "usr_block.json")
            sys_block_path = os.path.join(training_dir, "sys_block.json")

            if os.path.exists(usr_block_path):
                os.remove(usr_block_path)
            if os.path.exists(sys_block_path):
                os.remove(sys_block_path)


            auto_select_path = os.path.join(model_dir, "auto_select_request.json")
            if os.path.exists(auto_select_path):
                os.remove(auto_select_path)



            os.remove(switch_req_path)

            # isa.emit_results_change(username, farm_name, field_name, mission_date)
            set_scheduler_status(username, farm_name, field_name, mission_date, FINISHED_SWITCHING_MODELS)

        except Exception as e:

            trace = traceback.format_exc()
            logger.error("Exception occurred in process_switch")
            logger.error(e)
            logger.error(trace)
            try:
                set_scheduler_status(username, farm_name, field_name, mission_date, FINISHED_SWITCHING_MODELS, 
                                extra_items={"error_setting": "model switching", "error_message": str(e)})
            except:
                trace = traceback.format_exc()
                logger.error("Exception occurred while handling original exception")
                logger.error(e)
                logger.error(trace)