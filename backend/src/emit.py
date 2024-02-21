import os
import requests
import logging

from io_utils import json_io

RUNNING_ON_CLUSTER = "RUNNING_ON_CLUSTER" in os.environ and os.environ["RUNNING_ON_CLUSTER"] == "yes"
if RUNNING_ON_CLUSTER:
    base_url = ""
else:
    base_url = "https://" + os.environ.get("AC_IP") + ":" + os.environ.get("AC_PORT") + os.environ.get("AC_PATH")
image_set_notification_url = base_url + "/image_set_notification"
results_notification_url = base_url + "/results_notification"
model_notification_url = base_url + "/model_notification"
workers_notification_url = base_url + "/workers_notification"
upload_notification_url = base_url + "/upload_notification"

IDLE = "Idle"
SWITCHING_MODELS = "Switching Models"
FINE_TUNING = "Fine-Tuning"
PREDICTING = "Predicting"



IMAGE_SET_STATE_KEYS = [
    "state_name",
    "progress",
    "error_message",
    "prediction_image_names"
]



def set_image_set_status(username, farm_name, field_name, mission_date, new_state):

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")

    model_status_path = os.path.join(model_dir, "status.json")
    model_status = json_io.load_json(model_status_path)

    for k in IMAGE_SET_STATE_KEYS:
        if k in new_state:
            model_status[k] = new_state[k]
        else:
            model_status[k] = ""

    json_io.save_json(model_status_path, model_status)


    job_update = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
    }

    for k, v in model_status.items():
        job_update[k] = v

    success = emit(image_set_notification_url, job_update)

    if not success:
        model_status["error_message"] = "Failed to emit message."
        json_io.save_json(model_status_path, model_status)




def emit_results_change(username, farm_name, field_name, mission_date):

    data = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
    }

    emit(results_notification_url, data)




def emit_model_change(username):

    data = {
        "username": username
    }

    emit(model_notification_url, data)


def emit_worker_change(num_workers):
    data = {
        "num_workers": num_workers
    }
    emit(workers_notification_url, data)



def emit_upload_change(data):
    emit(upload_notification_url, data)


def emit(url, data):
    logger = logging.getLogger(__name__)


    if RUNNING_ON_CLUSTER:
        return True

    logger.info("Emitting {} to {}".format(data, url))
    headers = {'API-Key': os.environ["AC_API_KEY"]}

    response = requests.post(url, data=data, headers=headers, verify=False)
    status_code = response.status_code
    json_response = response.json()
    # response.raise_for_status()  # raises exception when not a 2xx response
    if status_code != 200:
        logger.error("Response status code is not 200. Status code: {}".format(status_code))
        logger.error(json_response)
        return False

    if "message" not in json_response or json_response["message"] != "received":
        logger.error("Response message is not 'received'.")
        logger.error(json_response)
        return False


    return True
