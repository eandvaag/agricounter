import logging
import argparse
import os
import shutil

from io_utils import json_io
import emit


def process_switch(username, farm_name, field_name, mission_date, model_name, model_creator):

    logger = logging.getLogger(__name__)

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")

    status_path = os.path.join(model_dir, "status.json")
    status = json_io.load_json(status_path)

    if status["state_name"] != emit.IDLE or status["error_message"] != "":
        raise RuntimeError("Cannot process switch request due to illegal image set state: {}".format(status))
    
    emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                              {"state_name": emit.SWITCHING_MODELS, "progress": "In Progress"})


    logger.info("Switching to model {}".format(model_name))


    if model_name.startswith("random_weights") and model_creator == "":
        model_path = os.path.join("usr", "shared", "weights", model_name)
        average_patch_size = 416
    else:

        model_path = os.path.join("usr", "data", model_creator, "models")
        public_model_path = os.path.join(model_path, "available", "public", model_name)
        private_model_path = os.path.join(model_path, "available", "private", model_name)

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
        except Exception:
            raise RuntimeError("Model log could not be loaded.")

    weights_path = os.path.join(model_path, "weights.h5")

    weights_dir = os.path.join(model_dir, "weights")
    tmp_weights_path = os.path.join(weights_dir, "tmp_weights.h5")
    best_weights_path = os.path.join(weights_dir, "best_weights.h5")
    cur_weights_path = os.path.join(weights_dir, "cur_weights.h5")


    try:
        shutil.copy(weights_path, tmp_weights_path)
    except Exception:
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


    training_dir = os.path.join(model_dir, "training")
    training_records_dir = os.path.join(training_dir, "training_tf_records")
    if os.path.exists(training_records_dir):
        shutil.rmtree(training_records_dir)
        os.makedirs(training_records_dir)


    patches_dir = os.path.join(image_set_dir, "model", "patches")

    shutil.rmtree(patches_dir)
    os.makedirs(patches_dir)


    status_path = os.path.join(model_dir, "status.json")
    status = json_io.load_json(status_path)
    status["model_creator"] = model_creator
    status["model_name"] = model_name
    status["patch_size"] = round(average_patch_size)
    json_io.save_json(status_path, status)


    emit.set_image_set_status(username, farm_name, field_name, mission_date, 
                              {"state_name": emit.IDLE})


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str)
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("model_creator", nargs="?", default="", type=str)

    args = parser.parse_args()

    process_switch(args.username,
                   args.farm_name,
                   args.field_name,
                   args.mission_date,
                   args.model_name,
                   args.model_creator)
