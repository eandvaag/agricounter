import tensorflow as tf
import numpy as np
import time
import sys
import os
import glob
import shutil
import tqdm
import logging
import time
import gc
import random
import cv2

from image_set import DataSet, Image
import build_datasets

from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          model_vis

from models.yolov4.loss import YOLOv4Loss
from models.yolov4.yolov4 import YOLOv4, YOLOv4Tiny
import models.yolov4.data_load as data_load
from models.yolov4.encode import Decoder


from io_utils import json_io, tf_record_io, w3c_io
    





def generate_predictions(config):

    logger = logging.getLogger(__name__)


    inference_patches_dir = os.path.join(config["model_dir"], "inference_patches")

    for image_set_config in config["inference"]["image_sets"]:

        dataset = DataSet(image_set_config)
        image_set_patches_dir = os.path.join(inference_patches_dir, 
                                          image_set_config["farm_name"] + "_" + 
                                          image_set_config["field_name"] + "_" + 
                                          image_set_config["mission_date"])

        if config["arch"]["model_type"] == "yolov4":
            yolov4 = YOLOv4(config)
        elif config["arch"]["model_type"] == "yolov4_tiny":
            yolov4 = YOLOv4Tiny(config)
        decoder = Decoder(config)

        tf_record_names = ["annotated-patches-record.tfrec", "unannotated-patches-record.tfrec"]

        predictions = driver_utils.create_predictions_skeleton(dataset)
        inference_times = []

        for k, tf_record_name in enumerate(tf_record_names):
            is_annotated = k == 0
            tf_record_path = os.path.join(image_set_patches_dir, tf_record_name)
            data_loader = data_load.InferenceDataLoader(tf_record_path, config)
            tf_dataset, tf_dataset_size = data_loader.create_dataset()

            input_shape = (config["inference"]["batch_size"], *(data_loader.get_model_input_shape()))
            yolov4.build(input_shape=input_shape)
            model_io.load_all_weights(yolov4, config)


            steps = np.sum([1 for _ in tf_dataset])

            logger.info("{} ('{}'): Running inference on {} images.".format(config["arch"]["model_type"], 
                                                                            config["model_name"], 
                                                                            tf_dataset_size))

            
            for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

                batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, is_annotated)
                batch_size = batch_images.shape[0]


                start_inference_time = time.time()
                pred = yolov4(batch_images, training=False)
                detections = decoder(pred)

                batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

                batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)

                end_inference_time = time.time()

                inference_times.append(end_inference_time - start_inference_time)

                for i in range(batch_size):

                    pred_bbox = batch_pred_bbox[i]
                    ratio = batch_ratios[i]

                    patch_info = batch_info[i]

                    image_path = bytes.decode((patch_info["image_path"]).numpy())
                    patch_path = bytes.decode((patch_info["patch_path"]).numpy())
                    image_name = os.path.basename(image_path).split(".")[0]
                    patch_name = os.path.basename(patch_path).split(".")[0]
                    patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

                    if is_annotated:
                        patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                        patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()


                    pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = \
                            post_process_sample(pred_bbox, ratio, patch_coords, config)


                    predictions["patch_predictions"][patch_name] = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "patch_coords": patch_coords.tolist(),
                        "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
                        "pred_scores": pred_patch_scores.tolist(),
                        "pred_classes": pred_patch_classes.tolist()
                    }
                    if is_annotated:
                        predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
                        predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

                    if image_name not in predictions["image_predictions"]:
                        predictions["image_predictions"][image_name] = {
                            "image_path": image_path,
                            "pred_image_abs_boxes": [],
                            "pred_classes": [],
                            "pred_scores": [],
                            "patch_coords": []               
                        }

                    pred_image_abs_boxes, pred_image_scores, pred_image_classes = \
                        driver_utils.get_image_detections(pred_patch_abs_boxes, 
                                                        pred_patch_scores, 
                                                        pred_patch_classes, 
                                                        patch_coords, 
                                                        image_path, 
                                                        trim=True)

                    predictions["image_predictions"][image_name]["pred_image_abs_boxes"].extend(pred_image_abs_boxes.tolist())
                    predictions["image_predictions"][image_name]["pred_scores"].extend(pred_image_scores.tolist())
                    predictions["image_predictions"][image_name]["pred_classes"].extend(pred_image_classes.tolist())
                    predictions["image_predictions"][image_name]["patch_coords"].append(patch_coords.tolist())
                    

        driver_utils.clip_image_boxes(predictions["image_predictions"])

        driver_utils.apply_nms_to_image_boxes(predictions["image_predictions"], 
                                            iou_thresh=config["inference"]["image_nms_iou_thresh"])

        driver_utils.add_class_detections(predictions["image_predictions"], config)

        metrics = driver_utils.create_metrics_skeleton(dataset)

        all_image_names = predictions["image_predictions"].keys()
        inference_metrics.collect_statistics(all_image_names, metrics, predictions, config, inference_times=inference_times)
        inference_metrics.collect_metrics(all_image_names, metrics, predictions, dataset, config)

        results_dir = os.path.join("usr", "data", "results",
                                   dataset.farm_name, dataset.field_name, dataset.mission_date,
                                   config["job_uuid"],
                                   config["model_uuid"])
        os.makedirs(results_dir, exist_ok=True)

        pred_path = os.path.join(results_dir, "predictions.json")
        json_io.save_json(pred_path, predictions)

        metrics_path = os.path.join(results_dir, "metrics.json")
        json_io.save_json(metrics_path, metrics)

        annotations_path = os.path.join(results_dir, "annotations.json")
        w3c_io.save_predictions(annotations_path, predictions["image_predictions"], config)

        excel_path = os.path.join(results_dir, "results.xlsx")
        driver_utils.output_excel(excel_path, predictions, dataset, config)


    shutil.rmtree(inference_patches_dir)





def train(config):

    tf.keras.backend.clear_session()

    logger = logging.getLogger(__name__)

    training_patches_dir = os.path.join(config["model_dir"], "training_patches")

    for seq_num in range(len(config["training"]["training_sequence"])):
     
        training_patch_dir = os.path.join(training_patches_dir, str(seq_num), "training")
        validation_patch_dir = os.path.join(training_patches_dir, str(seq_num), "validation")

        training_tf_record_paths = [os.path.join(training_patch_dir, "annotated-patches-record.tfrec")]
        validation_tf_record_paths = [os.path.join(validation_patch_dir, "annotated-patches-record.tfrec")]

        driver_utils.set_active_training_params(config, seq_num)

        train_loader_is_preloaded, train_data_loader = data_load.get_data_loader(training_tf_record_paths, config, shuffle=True, augment=True)
        val_loader_is_preloaded, val_data_loader = data_load.get_data_loader(validation_tf_record_paths, config, shuffle=False, augment=False)
        

        logger.info("Data loaders created. Train loader is preloaded?: {}. Validation loader is preloaded?: {}.".format(
            train_loader_is_preloaded, val_loader_is_preloaded
        ))

        train_dataset, num_train_images = train_data_loader.create_batched_dataset(
                                                take_percent=config["training"]["active"]["percent_of_training_set_used"])

        val_dataset, num_val_images = val_data_loader.create_batched_dataset(
                                                 take_percent=config["training"]["active"]["percent_of_validation_set_used"])


        # try:
        #     train_data_loader = data_load.PreLoadedTrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
        #     val_data_loader = data_load.PreLoadedTrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

        #     train_dataset, num_train_images = train_data_loader.create_batched_dataset(
        #                                             take_percent=config["training"]["active"]["percent_of_training_set_used"])

        #     val_dataset, num_val_images = val_data_loader.create_batched_dataset(
        #                                             take_percent=config["training"]["active"]["percent_of_validation_set_used"])
        
        # except RuntimeError:
        #     logger.info("Switching to non-preloaded data loader due to high memory usage.")

        #     # if 'train_data_loader' in locals():
        #     #     del train_data_loader
        #     # if 'val_data_loader' in locals():
        #     #     del val_data_loader
        #     # if 'train_dataset' in locals():
        #     #     del train_dataset
        #     # if 'val_dataset' in locals():
        #     #     del val_dataset

        #     # gc.collect()

        #     train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
        #     val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)

        #     train_dataset, num_train_images = train_data_loader.create_batched_dataset(
        #                                             take_percent=config["training"]["active"]["percent_of_training_set_used"])

        #     val_dataset, num_val_images = val_data_loader.create_batched_dataset(
        #                                             take_percent=config["training"]["active"]["percent_of_validation_set_used"])


        logger.info("Building model...")


        if config["arch"]["model_type"] == "yolov4":
            yolov4 = YOLOv4(config)
        elif config["arch"]["model_type"] == "yolov4_tiny":
            yolov4 = YOLOv4Tiny(config)

        loss_fn = YOLOv4Loss(config)


        input_shape = (config["training"]["active"]["batch_size"], *(train_data_loader.get_model_input_shape()))
        yolov4.build(input_shape=input_shape)

        logger.info("Model built.")


        if seq_num == 0:
            layer_lookup = yolov4.get_layer_lookup()
            layer_lookup_path = os.path.join(config["weights_dir"], "layer_lookup.json")
            json_io.save_json(layer_lookup_path, layer_lookup)

        else:
            model_io.load_select_weights(yolov4, config)

        optimizer = tf.optimizers.Adam()


        train_loss_metric = tf.metrics.Mean()
        val_loss_metric = tf.metrics.Mean()

        @tf.function
        def train_step(batch_images, batch_labels):
            with tf.GradientTape() as tape:
                conv = yolov4(batch_images, training=True)
                loss_value = loss_fn(batch_labels, conv)
                loss_value += sum(yolov4.losses)

            #if np.isnan(loss_value):
            #    raise RuntimeError("NaN loss has occurred.")
            gradients = tape.gradient(target=loss_value, sources=yolov4.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, yolov4.trainable_variables))
            train_loss_metric.update_state(values=loss_value)



        train_steps_per_epoch = np.sum([1 for _ in train_dataset])
        val_steps_per_epoch = np.sum([1 for _ in val_dataset])
        loss_record = {
            "training_loss": { "values": [],
                               "best": {"epoch": -1, "value": sys.float_info.max},
                               "epochs_since_improvement": 0}, 
            "validation_loss": {"values": [],
                                "best": {"epoch": -1, "value": sys.float_info.max},
                                "epochs_since_improvement": 0}
        }

        logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
                     config["arch"]["model_type"], config["model_name"], num_train_images, num_val_images))


        loss_record_path = os.path.join(config["loss_records_dir"], str(seq_num) + ".json")

        max_num_epochs = config["training"]["active"]["max_num_epochs"]
        steps_taken = 0
        #with tf.profiler.experimental.Profile('logdir'):
        for epoch in range(max_num_epochs):
            if epoch == 0:
                disp_training_best = float("inf")
                disp_validation_best = float("inf")
            else:
                disp_training_best = loss_record["training_loss"]["best"]["value"]
                disp_validation_best = loss_record["validation_loss"]["best"]["value"]


            train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
            for batch_data in train_bar:
            #for step in tqdm.trange(train_steps_per_epoch):

                #with tf.profiler.experimental.Trace('train', step_num=steps_taken, _r=1):

                #batch_data = next(iter(train_dataset))

                optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))

                batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)

                train_step(batch_images, batch_labels)
                if np.isnan(train_loss_metric.result()):
                    raise RuntimeError("NaN loss has occurred (training dataset).")
                train_bar.set_description("Epoch: {}/{} | t. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                          epoch, max_num_epochs-1, train_loss_metric.result(), 
                                          disp_training_best,
                                          loss_record["training_loss"]["best"]["epoch"]))
                steps_taken += 1


            cur_training_loss = float(train_loss_metric.result())

            cur_training_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "training_loss", cur_training_loss, epoch)
            if cur_training_loss_is_best and config["training"]["active"]["save_method"] == "best_training_loss":
                model_io.save_model_weights(yolov4, config, seq_num, epoch)

            json_io.save_json(loss_record_path, loss_record)

            train_loss_metric.reset_states()


            val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
            
            for batch_data in val_bar:
                batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
                conv = yolov4(batch_images, training=False)
                loss_value = loss_fn(batch_labels, conv)
                loss_value += sum(yolov4.losses)

                val_loss_metric.update_state(values=loss_value)
                if np.isnan(val_loss_metric.result()):
                    raise RuntimeError("NaN loss has occurred (validation dataset).")

                val_bar.set_description("Epoch: {}/{} | v. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                        epoch, max_num_epochs-1, val_loss_metric.result(), 
                                        disp_validation_best,
                                        loss_record["validation_loss"]["best"]["epoch"]))

            
            cur_validation_loss = float(val_loss_metric.result())

            cur_validation_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "validation_loss", cur_validation_loss, epoch)
            if cur_validation_loss_is_best and config["training"]["active"]["save_method"] == "best_validation_loss":
                model_io.save_model_weights(yolov4, config, seq_num, epoch)    


            json_io.save_json(loss_record_path, loss_record)
            
            val_loss_metric.reset_states()

            if driver_utils.stop_early(config, loss_record):
                break

    shutil.rmtree(training_patches_dir)





def output_image_predictions(image_predictions, out_dir):
    for image_name in image_predictions.keys():
        image_path = image_predictions[image_name]["image_path"]
        image_array = Image(image_path).load_image_array()

        boxes_key = "pred_image_abs_boxes"
        classes_key = "pred_classes"
        scores_key = "pred_scores"

        out_array = model_vis.draw_boxes_on_image(image_array,
                                                  image_predictions[image_name][boxes_key], #"pred_image_abs_boxes"],
                                                  image_predictions[image_name][classes_key], #"pred_classes"],
                                                  image_predictions[image_name][scores_key], #"pred_scores"],
                                                  class_map={"plant": 0},
                                                  gt_boxes=None,
                                                  patch_coords=image_predictions[image_name]["patch_coords"], #None,
                                                  display_class=False,
                                                  display_score=True
                                                  )
        out_path = os.path.join(out_dir, image_name + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))


def output_patch_prediction(patch, pred_boxes, pred_classes, pred_scores, out_path):

    out_array = model_vis.draw_boxes_on_image(patch,
                      pred_boxes,
                      pred_classes,
                      pred_scores,
                      class_map={"plant": 0},
                      gt_boxes=None,
                      patch_coords=None,
                      display_class=False,
                      display_score=False)
    cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))