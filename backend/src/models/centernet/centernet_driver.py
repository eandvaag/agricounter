import tensorflow as tf
import numpy as np
import time
import os
import shutil
import tqdm
import logging
import time

from image_set import ImgSet

from models.common import box_utils, \
                          model_io, \
                          inference_metrics, \
                          driver_utils, \
                          inference_record_io

from models.centernet.loss import CenterNetLoss
from models.centernet.centernet import CenterNet
import models.centernet.data_load as data_load
from models.centernet.encode import Decoder


from io_utils import json_io
from io_utils import tf_record_io



def post_process_sample(detections, resize_ratios, index):

    num_detections = detections[0][index]
    boxes = detections[1][index][:num_detections]
    scores = detections[2][index][:num_detections]
    classes = detections[3][index][:num_detections]

    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()

    boxes = box_utils.swap_xy_np(boxes)
    boxes = np.stack([
        boxes[:, 0] * resize_ratios[index][0],
        boxes[:, 1] * resize_ratios[index][1],
        boxes[:, 2] * resize_ratios[index][0],
        boxes[:, 3] * resize_ratios[index][1]
    ], axis=-1)

    boxes = np.rint(boxes).astype(np.int32)
    scores = scores.astype(np.float32)
    classes = classes.astype(np.int32)

    return boxes, scores, classes


def generate_predictions(config):

    logger = logging.getLogger(__name__)

    dataset_index = 0
    for img_set_num, img_set_conf in enumerate(config.inference["image_sets"]):

        farm_name = img_set_conf["farm_name"]
        field_name = img_set_conf["field_name"]
        mission_date = img_set_conf["mission_date"] 
        img_set = ImgSet(farm_name, field_name, mission_date)

        driver_utils.set_active_inference_params(config, img_set_num)

        for dataset_name in img_set_conf["datasets"]:

            dataset = img_set.datasets[dataset_name]

            patch_dir, _ = driver_utils.create_patches(config.inference["active"]["inference_patch_extraction_params"], 
                                                        img_set, dataset_name)


            tf_record_path = os.path.join(patch_dir, "patches-record.tfrec")

            data_loader = data_load.InferenceDataLoader(tf_record_path, config)
            tf_dataset, tf_dataset_size = data_loader.create_dataset()

            centernet = CenterNet(config)
            decoder = Decoder(config)

            #weights_path = model_load.get_most_recent_checkpoint(config.weights_dir)
            #centernet.load_weights(weights_path)

            input_shape = (config.inference["active"]["batch_size"], *(config.arch["input_img_shape"]))
            centernet.build(input_shape=input_shape)

            model_io.load_all_weights(centernet, config)

            predictions = driver_utils.create_predictions_skeleton(img_set, dataset)
            
            steps = np.sum([1 for i in tf_dataset])


            logger.info("{} ('{}'): Running inference on {} images.".format(config.arch["model_type"], 
                                                                            config.arch["model_name"], 
                                                                            tf_dataset_size))

            inference_times = []
            for step, batch_data in enumerate(tqdm.tqdm(tf_dataset, total=steps, desc="Generating predictions")):

                batch_images, batch_ratios, batch_info = data_loader.read_batch_data(batch_data, dataset.is_annotated)
                batch_size = batch_images.shape[0]


                start_inference_time = time.time()
                pred = centernet(batch_images, training=False)
                detections = decoder(pred)
                end_inference_time = time.time()

                inference_times.append(end_inference_time - start_inference_time)

                for i in range(batch_size):

                    pred_patch_abs_boxes, pred_patch_scores, pred_patch_classes = post_process_sample(detections, batch_ratios, i)

                    patch_info = batch_info[i]

                    img_path = bytes.decode((patch_info["img_path"]).numpy())
                    patch_path = bytes.decode((patch_info["patch_path"]).numpy())
                    img_name = os.path.basename(img_path).split(".")[0]
                    patch_name = os.path.basename(patch_path).split(".")[0]
                    patch_coords = tf.sparse.to_dense(patch_info["patch_coords"]).numpy().astype(np.int32)

                    if dataset.is_annotated:
                        patch_boxes = tf.reshape(tf.sparse.to_dense(patch_info["patch_abs_boxes"]), shape=(-1, 4)).numpy().tolist()
                        patch_classes = tf.sparse.to_dense(patch_info["patch_classes"]).numpy().tolist()

                    if pred_patch_abs_boxes.size == 0:
                        pred_img_abs_boxes = np.array([], dtype=np.int32)
                    else:
                        pred_img_abs_boxes = (np.array(pred_patch_abs_boxes) + \
                                              np.tile(patch_coords[:2], 2)).astype(np.int32)

                    predictions["patch_predictions"][patch_name] = {
                        "img_name": img_name,
                        "img_path": img_path,
                        "patch_coords": patch_coords.tolist(),
                        "pred_patch_abs_boxes": pred_patch_abs_boxes.tolist(),
                        "pred_scores": pred_patch_scores.tolist(),
                        "pred_classes": pred_patch_classes.tolist()
                    }
                    if dataset.is_annotated:
                        predictions["patch_predictions"][patch_name]["patch_abs_boxes"] = patch_boxes
                        predictions["patch_predictions"][patch_name]["patch_classes"] = patch_classes

                    if img_name not in predictions["image_predictions"]:
                        predictions["image_predictions"][img_name] = {
                            "img_path": img_path,
                            "pred_img_abs_boxes": [],
                            "pred_classes": [],
                            "pred_scores": [],
                            "patch_coords": []
                        }

                    predictions["image_predictions"][img_name]["pred_img_abs_boxes"].extend(pred_img_abs_boxes.tolist())
                    predictions["image_predictions"][img_name]["pred_scores"].extend(pred_patch_scores.tolist())
                    predictions["image_predictions"][img_name]["pred_classes"].extend(pred_patch_classes.tolist())
                    predictions["image_predictions"][img_name]["patch_coords"].append(patch_coords.tolist())



            driver_utils.clip_img_boxes(predictions["image_predictions"])
            driver_utils.apply_nms_to_img_boxes(predictions["image_predictions"], 
                                                iou_thresh=config.inference["active"]["image_nms_iou_thresh"])
            driver_utils.add_class_detections(predictions["image_predictions"], img_set)

            inference_metrics.collect_statistics(predictions, img_set, dataset,
                                                 inference_times=inference_times)
            if dataset.is_annotated:
                inference_metrics.collect_metrics(predictions, img_set, dataset)

            pred_dirname = os.path.basename(patch_dir)
            pred_dir = os.path.join(config.model_dir, "predictions", pred_dirname)
            os.makedirs(pred_dir)
            pred_path = os.path.join(pred_dir, "predictions.json")
            json_io.save_json(pred_path, predictions)

            excel_path = os.path.join(pred_dir, "results.xlsx")
            driver_utils.output_excel(excel_path, predictions, img_set, dataset_name)

            inference_entry = {
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "dataset_name": dataset_name,
                "model_uuid": config.arch["model_uuid"],
                "value": {
                    "group_uuid": config.arch["group_uuid"],
                    "group_name": config.arch["group_name"],
                    "model_uuid": config.arch["model_uuid"],
                    "model_name": config.arch["model_name"],
                    "prediction_dirname": pred_dirname,
                    "metrics": predictions["metrics"]
                }
            }
            inference_record_io.add_entry_to_inference_record(inference_entry)

            dataset_index += 1


def train(config):

    logger = logging.getLogger(__name__)


    for seq_num in range(len(config.training["training_sequence"])):

        training_tf_record_paths = []
        validation_tf_record_paths = []

        driver_utils.set_active_training_params(config, seq_num)

        for img_set_conf in config.training["training_sequence"][seq_num]["image_sets"]:
            img_set = ImgSet(img_set_conf["farm_name"], img_set_conf["field_name"], img_set_conf["mission_date"])
            training_patch_dir, _ = driver_utils.create_patches(
                config.training["active"]["training_patch_extraction_params"], img_set, "training")
            validation_patch_dir, _ = driver_utils.create_patches(
                config.training["active"]["validation_patch_extraction_params"], img_set, "validation")

            #training_tf_record_paths.append(os.path.join(training_patch_dir, "patches-with-boxes-record.tfrec"))
            #validation_tf_record_paths.append(os.path.join(validation_patch_dir, "patches-with-boxes-record.tfrec"))
            training_tf_record_paths.append(os.path.join(training_patch_dir, "patches-record.tfrec"))
            validation_tf_record_paths.append(os.path.join(validation_patch_dir, "patches-record.tfrec"))


        train_data_loader = data_load.TrainDataLoader(training_tf_record_paths, config, shuffle=True, augment=True)
        train_dataset, train_dataset_size = train_data_loader.create_batched_dataset(
                                                take_percent=config.training["active"]["percent_of_training_set_used"])

        val_data_loader = data_load.TrainDataLoader(validation_tf_record_paths, config, shuffle=False, augment=False)
        val_dataset, val_dataset_size = val_data_loader.create_batched_dataset(
                                                take_percent=config.training["active"]["percent_of_validation_set_used"])


        
        centernet = CenterNet(config)
        loss_fn = CenterNetLoss(config)

        input_shape = (config.training["active"]["batch_size"], *(config.arch["input_img_shape"]))
        centernet.build(input_shape=input_shape)


        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
        #                                                             decay_steps=steps_per_epoch * Config.learning_rate_decay_epochs,
        #                                                             decay_rate=0.96)
        if seq_num == 0:
            layer_lookup = centernet.get_layer_lookup()
            layer_lookup_path = os.path.join(config.weights_dir, "layer_lookup.json")
            json_io.save_json(layer_lookup_path, layer_lookup)

        else:
            model_io.load_select_weights(centernet, config)

        optimizer = tf.optimizers.Adam()


        train_loss_metric = tf.metrics.Mean()
        val_loss_metric = tf.metrics.Mean()

        def train_step(batch_images, batch_labels):
            with tf.GradientTape() as tape:
                pred = centernet(batch_images, training=True)
                loss_value = loss_fn(y_true=batch_labels, y_pred=pred)

            if np.isnan(loss_value):
                raise RuntimeError("NaN loss has occurred.")
            gradients = tape.gradient(target=loss_value, sources=centernet.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, centernet.trainable_variables))
            train_loss_metric.update_state(values=loss_value)



        train_steps_per_epoch = np.sum([1 for i in train_dataset])
        val_steps_per_epoch = np.sum([1 for i in val_dataset])
        best_val_loss = float("inf")
        epochs_since_improvement = 0
        loss_record = {
            "training_loss": { "values": [],
                               "best": {"epoch": -1, "value": float("inf")},
                               "epochs_since_improvement": 0},
            "validation_loss": {"values": [],
                                "best": {"epoch": -1, "value": float("inf")},
                                "epochs_since_improvement": 0}
        }

        logger.info("{} ('{}'): Starting to train with {} training images and {} validation images.".format(
                     config.arch["model_type"], config.arch["model_name"], train_dataset_size, val_dataset_size))


        max_num_epochs = config.training["active"]["max_num_epochs"]
        steps_taken = 0
        for epoch in range(max_num_epochs):

            train_bar = tqdm.tqdm(train_dataset, total=train_steps_per_epoch)
            for batch_data in train_bar:

                optimizer.lr.assign(driver_utils.get_learning_rate(steps_taken, train_steps_per_epoch, config))

                batch_images, batch_labels = train_data_loader.read_batch_data(batch_data)
                train_step(batch_images, batch_labels)
                train_bar.set_description("Epoch: {}/{} | t. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                          epoch, max_num_epochs-1, train_loss_metric.result(), 
                                          loss_record["training_loss"]["best"]["value"],
                                          loss_record["training_loss"]["best"]["epoch"]))
                steps_taken += 1


            val_bar = tqdm.tqdm(val_dataset, total=val_steps_per_epoch)
            for batch_data in val_bar:
                batch_images, batch_labels = val_data_loader.read_batch_data(batch_data)
                pred = centernet(batch_images, training=False)
                loss_value = loss_fn(y_true=batch_labels, y_pred=pred)
                if np.isnan(loss_value):
                    raise RuntimeError("NaN loss has occurred.")
                val_loss_metric.update_state(values=loss_value)
                val_bar.set_description("Epoch: {}/{} | v. loss: {:.4f} | best: {:.4f} (ep. {})".format(
                                        epoch, max_num_epochs-1, val_loss_metric.result(), 
                                        loss_record["validation_loss"]["best"]["value"],
                                        loss_record["validation_loss"]["best"]["epoch"]))

            cur_training_loss = float(train_loss_metric.result())
            cur_validation_loss = float(val_loss_metric.result())

            cur_training_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "training_loss", cur_training_loss, epoch)
            if cur_training_loss_is_best and config.training["active"]["save_method"] == "best_training_loss":
                model_io.save_model_weights(centernet, config, seq_num, epoch)

            cur_validation_loss_is_best = driver_utils.update_loss_tracker_entry(loss_record, "validation_loss", cur_validation_loss, epoch)
            if cur_validation_loss_is_best and config.training["active"]["save_method"] == "best_validation_loss":
                model_io.save_model_weights(centernet, config, seq_num, epoch)    

            if driver_utils.stop_early(config, loss_record):
                break


            train_loss_metric.reset_states()
            val_loss_metric.reset_states()


        loss_record_path = os.path.join(config.loss_records_dir, str(seq_num) + ".json")
        json_io.save_json(loss_record_path, loss_record)