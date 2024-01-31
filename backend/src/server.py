from flask import Flask, request
import time
import os
import shutil
import glob
import logging
import traceback
import multiprocessing as mp
import threading
import tensorflow as tf
import numpy as np

import image_set_aux
import image_set_actions as isa
import extract_patches as ep
from image_set import Image
import excess_green

from io_utils import json_io, tf_record_io
from models.yolov4 import yolov4_image_set_driver
from models.common import annotation_utils, inference_metrics

from lock_queue import LockQueue


MAX_STORED_SCHEDULER_UPDATES = 10

cv = threading.Condition()
sch_ctx = {}


app = Flask(__name__)


@app.route(os.environ.get("AC_PATH") + '/health_request', methods=['POST'])
def health_request():
    return {"message": "alive"}


@app.route(os.environ.get("AC_PATH") + '/add_request', methods=['POST'])
def add_request():
    logger = logging.getLogger(__name__)
    logger.info("POST to add_request")
    content_type = request.headers.get('Content-Type')

    if (content_type == 'application/json'):
        json_request = request.json
        logger.info("Got request: {}".format(json_request))
        if "request_type" not in json_request:
            return {"message": "bad_request"}
        req_type = json_request["request_type"]

        ok = True
        if req_type == "switch":
            sch_ctx["switch_queue"].enqueue(json_request)
        elif req_type == "prediction":
            sch_ctx["prediction_queue"].enqueue(json_request)
        elif req_type == "training":
            sch_ctx["training_queue"].enqueue(json_request)
        elif req_type == "baseline_training":
            sch_ctx["baseline_queue"].enqueue(json_request)
        else:
            ok = False

        with cv:
            cv.notify_all()

        if ok:
            return {"message": "ok"}
        else:
            return {"message": "bad_request"}

    else:
        return {"message": 'Content-Type not supported!'}
        

    

def needs_training(image_set_dir):

    upload_status_path = os.path.join(image_set_dir, "upload_status.json")
    if not os.path.exists(upload_status_path):
        return False

    upload_status = json_io.load_json(upload_status_path)
    if upload_status["status"] != "uploaded":
        return False
    
    model_dir = os.path.join(image_set_dir, "model")
    status_path = os.path.join(model_dir, "status.json")
    status = json_io.load_json(status_path)

    if status["model_name"] == "---" or status["model_creator"] == "---":
        return False


    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)
    num_training_regions = annotation_utils.get_num_training_regions(annotations)

    return status["num_regions_fully_trained_on"] < num_training_regions



def check_train(username, farm_name, field_name, mission_date):
    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    training_dir = os.path.join(image_set_dir, "model", "training")

    if not needs_training(image_set_dir):
        return

    usr_block_path = os.path.join(training_dir, "usr_block.json")
    sys_block_path = os.path.join(training_dir, "sys_block.json")
    if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
        return

    restart_req_path = os.path.join(training_dir, "restart_request.json")
    if os.path.exists(restart_req_path):
        return


    sch_ctx["training_queue"].enqueue({
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date
    })


def check_baseline(username):

    pending_dir = os.path.join("usr", "data", username, "models", "pending")
    for model_dir in glob.glob(os.path.join(pending_dir, "*")):
        log_path = os.path.join(model_dir, "log.json")
        if os.path.exists(log_path):
            log = json_io.load_json(log_path)
            sch_ctx["baseline_queue"].enqueue(log)




def process_baseline(item):
    logger = logging.getLogger(__name__)
    try:

        username = item["model_creator"]
        baseline_name = item["model_name"]

        usr_dir = os.path.join("usr", "data", username)
        models_dir = os.path.join(usr_dir, "models")
        pending_dir = os.path.join(models_dir, "pending")
        baseline_pending_dir = os.path.join(pending_dir, baseline_name)
        log_path = os.path.join(baseline_pending_dir, "log.json")

        available_dir = os.path.join(models_dir, "available")
        if item["public"] == "yes":
            baseline_available_dir = os.path.join(available_dir, "public", baseline_name)
        else:
            baseline_available_dir = os.path.join(available_dir, "private", baseline_name)

        aborted_dir = os.path.join(models_dir, "aborted")
        baseline_aborted_dir = os.path.join(aborted_dir, baseline_name)

        if os.path.exists(baseline_available_dir) or os.path.exists(baseline_aborted_dir):
            if os.path.exists(baseline_available_dir):
                logger.info("Not training baseline: baseline_available_dir exists")
            else:
                logger.info("Not training baseline: baseline_aborted_dir exists")
            return False

        logging.info("Starting to train baseline {}".format(item["model_name"]))
        # isa.set_scheduler_status(username, "---", "---", "---", isa.TRAINING)

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
        # log = {}
        # if resuming:
        #     log = json_io.load_json(log_path)
        # else:
        if "training_start_time" not in log:
            #log = {}

            # log["model_creator"] = item["model_creator"]
            # log["model_object"] = item["model_object"]
            # log["public"] = item["public"]
            # log["model_name"] = item["model_name"]
            # log["image_sets"] = item["image_sets"]
            # log["start_time"] = int(time.time())

            # os.makedirs(baseline_pending_dir)
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

                # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
                # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

                metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
                metadata = json_io.load_json(metadata_path)
                is_ortho = metadata["is_ortho"] == "yes"

                annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)



                # image_names = []
                # num_annotations = 0
                # for image_name in annotations.keys():
                #     for region_key in ["training_regions", "test_regions"]:
                #         for region in annotations[image_name][region_key]:
                #             inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], region)
                #             num_annotations += inds.size

                num_annotations = annotation_utils.get_num_annotations(annotations, ["training_regions", "test_regions"])
                    # if annotations[image_name]["status"] == "completed_for_training" or annotations[image_name]["status"] == "completed_for_testing":
                    #     image_names.append(image_name)
                    #     num_annotations += annotations[image_name]["boxes"].shape[0]
                if num_annotations > 0:
                    average_box_area = annotation_utils.get_average_box_area(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
                    average_box_height = annotation_utils.get_average_box_height(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
                    average_box_width = annotation_utils.get_average_box_width(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
                    

                    patch_size = annotation_utils.average_box_area_to_patch_size(average_box_area)
                else:
                    average_box_area = "NA"
                    average_box_height = "NA"
                    average_box_width = "NA"
                    patch_size = 416

                if "patch_size" in image_set:
                    patch_size = image_set["patch_size"]
                # log["image_sets"][image_set_index]["image_names"] = image_names
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
                        regions = annotations[image_name]["training_regions"] + annotations[image_name]["test_regions"]
                    
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
                            # class_mapping=class_mapping,
                            patch_overlap_percent=patch_overlap_percent, #50, 
                            regions=regions,
                            is_ortho=is_ortho,
                            include_patch_arrays=False,
                            out_dir=patches_dir)

                        # ep.write_patches(patches_dir, patch_records)
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

            # training_size = round(patch_records.size * 0.8)
            # training_subset = random.sample(np.arange(patch_records.size).tolist(), training_size)

            # training_patch_records = patch_records[training_subset]
            # validation_patch_records = np.delete(patch_records, training_subset)

            training_tf_records = tf_record_io.create_patch_tf_records(training_patch_records, patches_dir, is_annotated=True)
            training_patches_record_path = os.path.join(training_dir, "training-patches-record.tfrec")
            tf_record_io.output_patch_tf_records(training_patches_record_path, training_tf_records)

            # validation_tf_records = tf_record_io.create_patch_tf_records(validation_patch_records, patches_dir, is_annotated=True)
            # validation_patches_record_path = os.path.join(training_dir, "validation-patches-record.tfrec")
            # tf_record_io.output_patch_tf_records(validation_patches_record_path, validation_tf_records)

            loss_record = {
                "training_loss": { "values": [] }
            }
            loss_record_path = os.path.join(baseline_pending_dir, "model", "training", "loss_record.json")
            json_io.save_json(loss_record_path, loss_record)

            image_set_aux.reset_loss_record(baseline_pending_dir)

            log["training_start_time"] = int(time.time())
            json_io.save_json(log_path, log)


        # q = mp.Queue()
        # p = MyProcess(target=yolov4_image_set_driver.train_baseline, 
        #                 args=(sch_ctx, baseline_pending_dir, q))
        # p.start()
        # p.join()

        # if p.exception:
        #     exception, trace = p.exception
        #     raise exception

        # training_finished = q.get()

        training_finished = yolov4_image_set_driver.train_baseline(sch_ctx, baseline_pending_dir)

        if training_finished:
            # log = json_io.load_json(log_path)

            log["training_end_time"] = int(time.time())
            json_io.save_json(log_path, log)
            
            shutil.move(os.path.join(weights_dir, "best_weights.h5"),
                        os.path.join(baseline_pending_dir, "weights.h5"))
            
            shutil.move(os.path.join(training_dir, "loss_record.json"),
                        os.path.join(baseline_pending_dir, "loss_record.json"))

            shutil.rmtree(patches_dir)
            shutil.rmtree(model_dir)

            shutil.move(baseline_pending_dir, baseline_available_dir)

            isa.emit_model_change(item["model_creator"])

            return False
        else:
            return True


    except Exception as e:
        # if trace is None:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_baseline")
        logger.error(e)
        logger.error(trace)

        try:
            if baseline_pending_dir is not None:

                


                log["aborted_time"] = int(time.time())
                log["error_message"] = str(e)

                os.makedirs(baseline_aborted_dir)
                json_io.save_json(os.path.join(baseline_aborted_dir, "log.json"), log)

                if os.path.exists(baseline_pending_dir):
                    saved_pending_dir = os.path.join(baseline_aborted_dir, "saved_pending")
                    shutil.move(baseline_pending_dir, saved_pending_dir)
                    # shutil.rmtree(baseline_pending_dir)
                if os.path.exists(baseline_available_dir):
                    saved_available_dir = os.path.join(baseline_aborted_dir, "saved_available")
                    shutil.move(baseline_available_dir, saved_available_dir)
                    # shutil.rmtree(baseline_available_dir)

            isa.emit_model_change(item["model_creator"])
            #json_io.save_json(sys_block_path, {"error_message": str(e)})

            # isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_TRAINING,
            #                      extra_items={"error_setting": "training", "error_message": str(e)})
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Exception occurred while handling original exception")
            logger.error(e)
            logger.error(trace)

    # shutil.move(os.path.join(weights_dir, "best_weights.h5"),
    #             os.path.join("usr", "additional", "baselines", baseline_name + ".h5"))
    # shutil.rmtree(baseline_dir)
    # if completed:
    #     shutil.move("usr/data/baselines/training/" + baseline_name,
    #                 "usr/data/baselines/completed/" + baseline_name)

    # baseline_log["end_time"] = time.time()

    # log_dir = os.path.join("usr", "data", "baseline_logs")
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # baseline_log_path = os.path.join(log_dir, baseline_name + ".json")
    # json_io.save_json(baseline_log_path, baseline_log)


 

def process_train(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    training_dir = os.path.join(model_dir, "training")

    try:
        # trace = None

        if not needs_training(image_set_dir):
            return False

        usr_block_path = os.path.join(training_dir, "usr_block.json")
        sys_block_path = os.path.join(training_dir, "sys_block.json")
        if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
            return False

        switch_req_path = os.path.join(model_dir, "switch_request.json")
        if os.path.exists(switch_req_path):
            return False

        # logging.info("Starting to train {}".format(item))
        # isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINE_TUNING)

        # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
        # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)


        # training_image_names = []
        # for image_name in annotations.keys():
        #     if annotations[image_name]["status"] == "completed_for_training":
        #         training_image_names.append(image_name) 
        num_training_regions = annotation_utils.get_num_training_regions(annotations)

        logger.info("num_training_regions {}".format(num_training_regions))
        # 0
        # for image_name in annotations.keys():
        #     num_training_regions += len(annotations[image_name]["training_regions"])



        updated_patch_size = ep.update_model_patch_size(image_set_dir, annotations, ["training_regions"])
        update_applied = ep.update_training_patches(image_set_dir, annotations, updated_patch_size)
        #ep.update_patches(image_set_dir, annotations, training_image_names, updated_patch_size)
        

        # if len(changed_training_image_names) > 0:
        if update_applied:
            image_set_aux.update_training_tf_records(image_set_dir, annotations) #changed_training_image_names, annotations)
            image_set_aux.reset_loss_record(image_set_dir)

        if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
            return False
        if os.path.exists(switch_req_path):
            return False


        # q = mp.Queue()
        # p = MyProcess(target=collect_results, 
        #                 args=(sch_ctx, image_set_dir, q))
        # p.start()
        # p.join()

        # if p.exception:
        #     exception, trace = p.exception
        #     raise exception

        training_finished, re_enqueue = yolov4_image_set_driver.train(sch_ctx, image_set_dir) #q.get()
        
        # training_finished, re_enqueue = yolov4_image_set_driver.train(sch_ctx, image_set_dir)
        if training_finished:

            status_path = os.path.join(model_dir, "status.json")
            status = json_io.load_json(status_path)
            status["num_regions_fully_trained_on"] = num_training_regions #len(training_image_names)
            json_io.save_json(status_path, status)

            isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_FINE_TUNING)

            logger.info("Finished training {}".format(item))

        #     return False

        # else:
        #     return True
        return re_enqueue

    except Exception as e:
        # if trace is None:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_train")
        logger.error(e)
        logger.error(trace)

        try:
            json_io.save_json(sys_block_path, {"error_message": str(e)})

            isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_FINE_TUNING,
                                 extra_items={"error_setting": "fine-tuning", "error_message": str(e)})
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Exception occurred while handling original exception")
            logger.error(e)
            logger.error(trace)

        

        





def check_predict(username, farm_name, field_name, mission_date):

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    prediction_dir = os.path.join(model_dir, "prediction")
    prediction_requests_dirs = [
        os.path.join(prediction_dir, "image_requests"),
        os.path.join(prediction_dir, "image_set_requests", "pending")
    ]

    for prediction_requests_dir in prediction_requests_dirs:
        prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
        #for prediction_request_path in prediction_request_paths:
        if len(prediction_request_paths) > 0:
            sch_ctx["prediction_queue"].enqueue({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            })
            return




# def predict_on_images(image_set_dir, image_names, save_result):

#     # logger = logging.getLogger(__name__)

#     #image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)

#     # training_image_names = []
#     # for image_name in annotations.keys():
#     #     if annotations[image_name]["status"] == "completed_for_training":
#     #         training_image_names.append(image_name) 


#     # updated_patch_size = ep.get_updated_patch_size(annotations)

#     # # first make sure that training records are up to date, so that if the inference
#     # # request is changing the patch data for a training image we will reset the loss record and the
#     # # model will train later
#     # changed_training_image_names = ep.update_patches(image_set_dir, annotations, training_image_names, updated_patch_size)
#     # if len(changed_training_image_names) > 0:
#     #     image_set_aux.update_training_tf_records(image_set_dir, changed_training_image_names, annotations)
#     #     image_set_aux.reset_loss_record(image_set_dir)

#     # ep.update_patches(image_set_dir, annotations, image_names, updated_patch_size)

#     # image_set_aux.update_prediction_tf_records(image_set_dir, image_names=image_names)
    
#     end_time, _ = yolov4_image_set_driver.predict(image_set_dir, image_names=image_names, save_result=save_result)

#     return end_time

def create_vegetation_record(image_set_dir, excess_green_record, annotations, predictions):

    # vegetation_record_path = os.path.join(image_set_dir, "excess_green", "vegetation_record.json")
    # vegetation_record = json_io.load_json(vegetation_record_path)

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


# def create_maps(results_dir):
#     logger = logging.getLogger(__name__)

#     path_pieces = results_dir.split("/")
#     username = path_pieces[2]
#     farm_name = path_pieces[4]
#     field_name = path_pieces[5]
#     mission_date = path_pieces[6]
#     # image_set_dir = os.path.join(*path_pieces[:len(path_pieces)-3])
#     predictions_path = os.path.join(results_dir, "predictions.json")
#     out_dir = os.path.join(results_dir, "maps")
#     try:
#         for interpolation in ["linear", "nearest"]:
#             interpolate.create_interpolation_map(username,
#                                 farm_name,
#                                 field_name,
#                                 mission_date,
#                                 predictions_path, #annotations_path,
#                                 out_dir, 
#                                 # args.map_download_uuid,
#                                 interpolation=interpolation)
#     except Exception as e:
#         logger.info("Unable to produce density maps.")


def collect_results(image_set_dir, results_dir): #, annotations, fast_metrics):
    # print("running post_result")

    # create_maps(results_dir)
    
    path_pieces = results_dir.split("/")
    username = path_pieces[2]
    farm_name = path_pieces[4]
    field_name = path_pieces[5]
    mission_date = path_pieces[6]

    request_path = os.path.join(results_dir, "request.json")
    request = json_io.load_json(request_path)

    full_predictions_path = os.path.join(results_dir, "full_predictions.json")
    full_predictions = json_io.load_json(full_predictions_path)

    predictions_path = os.path.join(results_dir, "predictions.json")
    predictions = json_io.load_json(predictions_path)

    # excess_green_record_path = os.path.join(image_set_dir, "excess_green", "record.json")
    # if os.path.exists(excess_green_record_path):
    #     excess_green_record = json_io.load_json(excess_green_record_path)
    # else:
    #     excess_green_record = None

    annotations_src_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_src_path)


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.COLLECTING_METRICS)
    metrics = inference_metrics.collect_image_set_metrics(predictions, annotations, metadata) #, config)
    # metrics = fast_metrics #{}
    # metric_keys = list(fast_metrics.keys()) + list(slow_metrics.keys())
    
    # for metric_key in slow_metrics.keys():
    #     for image_name in annotations.keys():
    #         for region_key in ["training_regions", "test_regions"]:
    #             metrics[metric_key][image_name][region_key] = slow_metrics[metric_key][image_name][region_key]
                

    
    metrics_path = os.path.join(results_dir, "metrics.json")
    json_io.save_json(metrics_path, metrics)

    excess_green_record_src_path = os.path.join(image_set_dir, "excess_green", "record.json")
    excess_green_record = json_io.load_json(excess_green_record_src_path)
    # image_set_results_dir = os.path.join(results_dir, str(end_time))
    # os.makedirs(image_set_results_dir)
    # annotations_path = os.path.join(results_dir, "annotations.json")
    # json_io.save_json(annotations_path, annotations_json)
    # annotation_utils.save_annotations(annotations_path, annotations)

    # if excess_green_record is not None:
    #     excess_green_record_path = os.path.join(results_dir, "excess_green_record.json")
    #     json_io.save_json(excess_green_record_path, excess_green_record)

    # print("running collect_image_set_metrics")


    calculate_vegetation_coverage = "calculate_vegetation_coverage" in request and request["calculate_vegetation_coverage"]

    if calculate_vegetation_coverage:
        isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.CALCULATING_VEGETATION_COVERAGE)
        vegetation_record = create_vegetation_record(image_set_dir, excess_green_record, annotations, predictions)

        # vegetation_record_path = os.path.join(image_set_dir, "excess_green", "vegetation_record.json")
        # json_io.save_json(vegetation_record_path, updated_vegetation_record)

        results_vegetation_record_path = os.path.join(results_dir, "vegetation_record.json")
        json_io.save_json(results_vegetation_record_path, vegetation_record)

    # excess_green_record_src_path = os.path.join(image_set_dir, "excess_green", "record.json")
    # if os.path.exists(excess_green_record_src_path):
    excess_green_record_dst_path = os.path.join(results_dir, "excess_green_record.json")
    # shutil.copyfile(excess_green_record_src_path, excess_green_record_dst_path)
    json_io.save_json(excess_green_record_dst_path, excess_green_record)

    tags_src_path = os.path.join(image_set_dir, "annotations", "tags.json")
    tags_dst_path = os.path.join(results_dir, "tags.json")
    shutil.copy(tags_src_path, tags_dst_path)
    
    # annotations_src_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations_dst_path = os.path.join(results_dir, "annotations.json")
    annotation_utils.save_annotations(annotations_dst_path, annotations)
    # annotations = annotation_utils.load_annotations(annotations_dst_path)
    # shutil.copyfile(annotations_src_path, annotations_dst_path)



    regions_only = "regions_only" in request and request["regions_only"]

    
    inference_metrics.create_spreadsheet(results_dir, regions_only=regions_only)




    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)

    if inference_metrics.can_calculate_density(metadata, camera_specs):
        isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.CALCULATING_VORONOI_AREAS)
        inference_metrics.create_areas_spreadsheet(results_dir, regions_only=regions_only)

    raw_outputs_dir = os.path.join(results_dir, "raw_outputs")
    os.makedirs(raw_outputs_dir)
    # shutil.copyfile(annotations_dst_path, os.path.join(raw_outputs_dir, "annotations.json"))
    # shutil.copyfile(full_predictions_path, os.path.join(raw_outputs_dir, "predictions.json"))
    downloadable_annotations = {}
    int_to_ext_annotation_keys = {
        "boxes": "annotations",
        "classes": "classes",
        "regions_of_interest": "regions_of_interest",
        "training_regions": "fine_tuning_regions",
        "test_regions": "test_regions"
    }


    for image_name in annotations.keys():
        downloadable_annotations[image_name] = {}
        for key in int_to_ext_annotation_keys.keys():

            downloadable_annotations[image_name][int_to_ext_annotation_keys[key]] = []

            if key == "classes":
                downloadable_annotations[image_name][int_to_ext_annotation_keys[key]] = annotations[image_name][key]
                #(annotations[image_name][key]).tolist()

            elif key == "regions_of_interest":

                for poly in annotations[image_name]["regions_of_interest"]:
                    download_poly = []
                    for coord in poly:
                        download_poly.append([int(coord[1]), int(coord[0])])
                    downloadable_annotations[image_name][int_to_ext_annotation_keys[key]].append(download_poly)
            
            else:
                for box in annotations[image_name][key]:
                    download_box = [
                        int(box[1]),
                        int(box[0]),
                        int(box[3]),
                        int(box[2])
                    ]
                    downloadable_annotations[image_name][int_to_ext_annotation_keys[key]].append(download_box)
    
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
        #(full_predictions[image_name]["classes"]).tolist()


    json_io.save_json(os.path.join(raw_outputs_dir, "predictions.json"), downloadable_predictions)




    shutil.make_archive(os.path.join(results_dir, "raw_outputs"), 'zip', raw_outputs_dir)
    shutil.rmtree(raw_outputs_dir)


    # full_predictions_path = os.path.join(results_dir, "full_predictions.json")
    # json_io.save_json(full_predictions_path, full_predictions)


    # predictions_path = os.path.join(results_dir, "predictions.json")
    # json_io.save_json(predictions_path, predictions)



    # for image_name in predictions.keys():
    #     inds = np.array(predictions[image_name]["scores"]) >= 0.25
    #     predictions[image_name]["boxes"] = np.array(predictions[image_name]["boxes"])[inds].tolist()
    #     predictions[image_name]["scores"] = np.array(predictions[image_name]["scores"])[inds].tolist()



    
    # w3c_io.save_predictions(image_predictions_path, image_predictions, config)
    return


# def save_predictions(image_set_dir, image_names, predictions):
#     # print("Saving predictions")

#     model_dir = os.path.join(image_set_dir, "model")
#     predictions_dir = os.path.join(model_dir, "prediction")
#     for image_name in image_names:
#         image_predictions_dir = os.path.join(predictions_dir, "images", image_name)
#         os.makedirs(image_predictions_dir, exist_ok=True)
#         predictions_path = os.path.join(image_predictions_dir, "predictions.json")

#         # save_boxes = predictions[image_name]["boxes"]
#         # save_scores = predictions[image_name]["scores"]
#         # inds = np.array(predictions[image_name]["scores"]) >= 0.25

#         # print("{}: num_boxes: {} num_above_thresh_boxes: {}".format(image_name, len(save_boxes), np.sum(inds)))
#         # predictions[image_name]["boxes"] = np.array(predictions[image_name]["boxes"])[inds].tolist()
#         # predictions[image_name]["scores"] = np.array(predictions[image_name]["scores"])[inds].tolist()

#         json_io.save_json(predictions_path, {image_name: predictions[image_name]})

#         # predictions[image_name]["boxes"] = save_boxes
#         # predictions[image_name]["scores"] = save_scores


# def update_progress(image_set_dir, predictions, annotations):

#     fast_metrics = inference_metrics.collect_fast_image_set_metrics(image_set_dir, predictions, annotations)



#     return fast_metrics


def process_predict(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)

    model_dir = os.path.join(image_set_dir, "model")

    prediction_dir = os.path.join(model_dir, "prediction")
    prediction_requests_dirs = [
        os.path.join(prediction_dir, "image_requests"),
        os.path.join(prediction_dir, "image_set_requests", "pending")
    ]

    for prediction_requests_dir in prediction_requests_dirs:
        prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
        try:
            # trace = None
            while len(prediction_request_paths) > 0:
            
                results_dir = None
                
                prediction_request_path = prediction_request_paths[0]
                request = json_io.load_json(prediction_request_path)

                logger.info("Starting to predict for {}".format(item))
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.PREDICTING,
                                            extra_items={"percent_complete": 0})


                if request["save_result"]:
                    isa.emit_results_change(username, farm_name, field_name, mission_date)
                            

                interrupted = yolov4_image_set_driver.predict(sch_ctx, image_set_dir, request)
                # q = mp.Queue()
                # p = MyProcess(target=yolov4_image_set_driver.predict, 
                #             args=(sch_ctx, image_set_dir, request, q))
                # p.start()
                # p.join()

                # if p.exception:
                #     exception, trace = p.exception
                #     raise exception

                # interrupted = q.get()


                # interrupted, predictions, full_predictions = yolov4_image_set_driver.predict(
                #                     sch_ctx,
                #                     image_set_dir, 
                #                     image_names=request["image_names"], 
                #                     save_result=request["save_result"])
                

                #end_time = predict_on_images(
                #        image_set_dir,
                #        request["image_names"],
                #        request["save_result"]
                #)
                if interrupted:
                    return

                # save_predictions(image_set_dir, request["image_names"], predictions)


                # fast_metrics = update_progress(image_set_dir, predictions, annotations)

                if request["save_result"]:
                    
                    results_dir = os.path.join(model_dir, "results", request["request_uuid"])
                    saved_request_path = os.path.join(results_dir, "request.json")
                    json_io.save_json(saved_request_path, request)
                    # os.makedirs(results_dir)

                    # q = mp.Queue()
                    collect_results(image_set_dir, results_dir) #, fast_metrics, annotations)
                    # p = MyProcess(target=collect_results, 
                    #               args=(image_set_dir, results_dir))
                    # p.start()
                    # p.join()

                    # if p.exception:
                    #     exception, trace = p.exception
                    #     raise exception


                    # collect_results(image_set_dir, results_dir, predictions, full_predictions)
                    request = json_io.load_json(saved_request_path)
                    end_time = int(time.time())
                    request["end_time"] = end_time
                    json_io.save_json(saved_request_path, request)

                os.remove(prediction_request_path)

                logger.info("Finished predicting for {}".format(item))
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_PREDICTING, 
                                extra_items={"prediction_image_names": ",".join(request["image_names"])})

                if request["save_result"]:
                    isa.emit_results_change(username, farm_name, field_name, mission_date)

                prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
                


        except Exception as e:
            # if trace is None:
            trace = traceback.format_exc()
            logger.error("Exception occurred in process_predict")
            logger.error(e)
            logger.error(trace)

            try:
                if os.path.exists(prediction_request_path):
                    os.remove(prediction_request_path)

                if results_dir is not None and os.path.exists(results_dir):
                    # shutil.move(results_dir, os.path.join(prediction_dir, "image_set_requests", "aborted", os.path.basename(prediction_request_path))[:-5])
                    shutil.rmtree(results_dir)

                if request["save_result"]:
                    request["aborted_time"] = int(time.time())
                    request["error_message"] = str(e)
                    request["error_info"] = str(trace)
                    json_io.save_json(
                        os.path.join(prediction_dir, "image_set_requests", "aborted", os.path.basename(prediction_request_path)),
                        request)


                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_PREDICTING, 
                            extra_items={"error_setting": "prediction", "error_message": str(e)})

                if request["save_result"]:
                    isa.emit_results_change(username, farm_name, field_name, mission_date)


            except Exception as e:
                trace = traceback.format_exc()
                logger.error("Exception occurred while handling original exception")
                logger.error(e)
                logger.error(trace)
                




            # if save_result:
            #     json_io.save_json(os.path.join(model_dir, "results", str(end_time), "request.json"), request)


def check_switch(username, farm_name, field_name, mission_date):

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    switch_req_path = os.path.join(image_set_dir, "model", "switch_request.json")
    if os.path.exists(switch_req_path):
        sch_ctx["switch_queue"].enqueue({
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date
        })



def one_pass():
    logger = logging.getLogger(__name__)
    for username_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(username_path)
        # logger.info("checking {}".format(username_path))
        try:
            check_baseline(username)
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Exception occurred while performing pass (check_baseline)")
            logger.error(e)
            logger.error(trace)
        # for user_dir in glob.glob(os.path.join(username_path, "*")):
        #     if os.path.basename(user_dir) == "image_sets":
        for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
            farm_name = os.path.basename(farm_path)
            # logger.info("checking {}".format(farm_name))
            for field_path in glob.glob(os.path.join(farm_path, "*")):
                field_name = os.path.basename(field_path)
                for mission_path in glob.glob(os.path.join(field_path, "*")):
                    mission_date = os.path.basename(mission_path)
                    try:
                        check_switch(username, farm_name, field_name, mission_date)
                    except Exception as e:
                        trace = traceback.format_exc()
                        logger.error("Exception occurred while performing pass (check_switch)")
                        logger.error(e)
                        logger.error(trace)

                    try:
                        check_predict(username, farm_name, field_name, mission_date)
                    except Exception as e:
                        trace = traceback.format_exc()
                        logger.error("Exception occurred while performing pass (check_predict)")
                        logger.error(e)
                        logger.error(trace)
                    try:                            
                        check_train(username, farm_name, field_name, mission_date)
                    except Exception as e:
                        trace = traceback.format_exc()
                        logger.error("Exception occurred while performing pass (check_train)")
                        logger.error(e)
                        logger.error(trace)



def sweep():

    logger = logging.getLogger(__name__)
    logger.info("Sweeper started")

    while True:

        if not an_item_is_available():

            logger.info("Performing sweep")
            try:
                one_pass()
            except Exception as e:
                trace = traceback.format_exc()
                logger.error("Exception occurred during sweep")
                logger.error(e)
                logger.error(trace)



            if an_item_is_available():
                logger.info("Sweeper is notifying")
                with cv:
                    cv.notify_all()

            logger.info("Sweep complete")

        
        time.sleep(60 * 60)



def drain():
    logger = logging.getLogger(__name__)
    logger.info("Drain has started")

    while True:
        try:
            switch_queue_size = sch_ctx["switch_queue"].size()
            while switch_queue_size > 0:
                item = sch_ctx["switch_queue"].dequeue()
                isa.process_switch(item)
                switch_queue_size = sch_ctx["switch_queue"].size()


            pred_queue_size = sch_ctx["prediction_queue"].size()
            while pred_queue_size  > 0:
                item = sch_ctx["prediction_queue"].dequeue()
                process_predict(item)
                pred_queue_size = sch_ctx["prediction_queue"].size()


            train_queue_size = sch_ctx["training_queue"].size()
            if train_queue_size > 0:
                item = sch_ctx["training_queue"].dequeue()
                re_enqueue = process_train(item)
                if re_enqueue:
                    sch_ctx["training_queue"].enqueue(item)

            baseline_queue_size = sch_ctx["baseline_queue"].size()
            if baseline_queue_size > 0:
                item = sch_ctx["baseline_queue"].dequeue()
                re_enqueue = process_baseline(item)
                if re_enqueue:
                    sch_ctx["baseline_queue"].enqueue(item)


        
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Exception occurred while draining queue")
            logger.error(e)
            logger.error(trace)


        #time.sleep(5)    

        if not an_item_is_available():

            logger.info("Drain has finished")

            scheduler_status_path = os.path.join("usr", "shared", "scheduler_status.json")
            scheduler_status = json_io.load_json(scheduler_status_path)
            if scheduler_status["status"] != isa.IDLE:
                isa.set_scheduler_status("---", "---", "---", "---", isa.IDLE)
            return

             
def an_item_is_available():
    queue_names = ["switch_queue", "prediction_queue", "training_queue", "baseline_queue"]

    for queue_name in queue_names:
        if sch_ctx[queue_name].size() > 0:
            return True
    return False
    # return ((sch_ctx["switch_queue"].size() > 0 or sch_ctx["prediction_queue"].size() > 0)
    #          or sch_ctx["training_queue"].size() > 0) or sch_ctx["baseline_queue"].size() > 0


def work():
    logger = logging.getLogger(__name__)
    logger.info("Worker thread started")


    while True:
        drain()
        logger.info("Worker waiting")
        with cv:
            #while not an_item_is_available():
            #    cv.wait()
            cv.wait_for(an_item_is_available)
        logger.info("Worker woken up")



if __name__ == "__main__":

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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

    sch_ctx["switch_queue"] = LockQueue()
    sch_ctx["prediction_queue"] = LockQueue()
    sch_ctx["training_queue"] = LockQueue()
    sch_ctx["baseline_queue"] = LockQueue()


    one_pass()

    worker = threading.Thread(name="worker", target=work)
    worker.start()


    sweeper = threading.Thread(name="sweeper", target=sweep)
    sweeper.start()


    app.run(host=os.environ.get("AC_IP"), port=os.environ.get("AC_PY_PORT"))