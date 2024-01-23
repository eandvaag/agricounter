import logging
import traceback
import os
import glob
import time
import random
import math as m
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.metrics import pairwise_distances, mutual_info_score
from sklearn.manifold import TSNE, MDS
from sklearn import decomposition
from scipy.stats import entropy, skew, kurtosis

import matplotlib.pyplot as plt

from models.common import model_keys
from models.yolov4.yolov4 import YOLOv4Tiny
from models.yolov4.encode import Decoder
from models.yolov4 import yolov4_image_set_driver



# import model_descriptor
from io_utils import json_io
from models.common import annotation_utils, box_utils
import extract_patches as ep
import image_set_actions as isa





def drain_switch_queue(sch_ctx, cur_image_set_dir=None):
    affected = False
    switch_queue_size = sch_ctx["switch_queue"].size()
    while switch_queue_size > 0:
        item = sch_ctx["switch_queue"].dequeue()
        isa.process_switch(item)
        switch_queue_size = sch_ctx["switch_queue"].size()

        if cur_image_set_dir is not None:
            affected_image_set_dir = os.path.join("usr", "data", item["username"],
                                                  "image_sets", 
                                                  item["farm_name"], item["field_name"], item["mission_date"])
            if affected_image_set_dir == cur_image_set_dir:
                affected = True

    return affected












def create_tsne(out_path, target_vectors, source_vectors, target_name, source_name):
    all_vectors = np.concatenate([target_vectors, source_vectors])
    X_embedded = TSNE(n_components=2).fit_transform(all_vectors)
    fig, ax = plt.subplots() #plt.figure()
    ax.scatter(X_embedded[:target_vectors.shape[0], 0], X_embedded[:target_vectors.shape[0], 1], c="blue", label=target_name)
    ax.scatter(X_embedded[target_vectors.shape[0]:, 0], X_embedded[target_vectors.shape[0]:, 1], c="red", label=source_name)
    ax.legend()
    fig.savefig(out_path)
    distances = pairwise_distances(X_embedded[:target_vectors.shape[0], :], X_embedded[target_vectors.shape[0]:, :], metric="l1")
    sel_distances = np.min(distances, axis=1)
    
    # score = np.divide(np.sum(distances ** 2), source_vectors.shape[0])
    score = np.sum(sel_distances)



    return score


def get_image_set_match_score(target_vectors, source_vectors):

    # print("target_vectors.shape", target_vectors.shape)
    # print("source_vectors.shape", source_vectors.shape)
    

    # dist_metric = DistanceMetric.get_metric('minkowski', p=0.5)
    # all_vectors = np.concatenate([target_vectors, source_vectors])
    # # pca = decomposition.PCA(n_components=10)
    # mds = MDS(n_components=10)
    # # pca.fit(all_vectors)
    # pca_all_vectors = mds.fit_transform(all_vectors)

    # target_vectors = pca_all_vectors[:target_vectors.shape[0], :]
    # source_vectors = pca_all_vectors[target_vectors.shape[0]:, :]


    distances = pairwise_distances(target_vectors, source_vectors, metric="l1") #cosine") #"l1") #euclidean")
    # distances = dist_metric.pairwise(target_vectors, source_vectors)
    # score = np.divide(np.sum(distances ** 2), source_vectors.shape[0])

    # print("distances.shape", distances.shape)
    sel_distances = np.min(distances, axis=1)
    score = np.sum(sel_distances)
    return score


    # score = 0
    # tree = KDTree(source_vectors, leaf_size=2, metric='l1')
    # # query_vector = target_vectors
    # dists, inds = tree.query(target_vectors, k=1)

    # score = np.sum(dists[:, 0] ** 2)

    # for i in range(target_vectors.shape[0]):
    #     query_vector = target_vectors[i]
    #     dists, inds = tree.query([query_vector], k=1) #candidate_vectors.shape[0]) #3)
    #     # print("dists", dists)
    #     dist = dists[0][0]
    #     score += (dist ** 2)
    return score

def get_confidence_quality_2(pred_scores):

    # scores_hist = np.histogram(pred_scores, bins=100, range=(0, 1))[0]

    # mask = pred_scores > 0.9
    # pred_scores[mask] = pred_scores[mask] ** 2

    scores_skew = skew(pred_scores) #scores_hist)
    # scores_kurtosis = kurtosis(pred_scores)
    # hist_mean = np.mean(scores_hist)
    # scores_mean = np.mean(pred_scores ** 2)
    # print("hist_mean", hist_mean)
    # print("scores_mean", scores_mean)

    return (-1) * scores_skew #hist_mean #np.mean(pred_scores) #scores_hist) #scores_kurtosis
    # ideal_hist = np.histogram(np.array([1.0]), bins=100, range=(0, 1))[0]
    # print(scores_hist.shape)
    # print(ideal_hist.shape)
    # quality = mutual_info_score(scores_hist, ideal_hist) #entropy(scores_hist, ideal_hist)
    # return quality


def get_confidence_quality(pred_scores):
    num_predictions = int(pred_scores.size)
    confidence_score = 0
    STEP_SIZE = 0.01
    if num_predictions > 0:
        for conf_val in np.arange(0.25, 1.0, STEP_SIZE): #25, 1.0, STEP_SIZE):
            num_in_range = float(
                np.sum(np.logical_and(pred_scores > conf_val, pred_scores <= conf_val+STEP_SIZE))
            )
            prob = num_in_range / num_predictions
            # confidence_score += prob * (conf_val ** 2) #(2 ** (conf_val* 100)) # * conf_val * conf_val)
            # confidence_score += prob * (conf_val * conf_val)
            confidence_score += prob * (1 / (1 + (m.e ** (-30 * (conf_val - 0.80)))))
            # confidence_score += prob * (2 ** (30 * (conf_val-1))) #( (2**7) * ((conf_val - 0.5) ** 7) )
    return confidence_score





def process_auto_select(item, sch_ctx):
    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]
    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")

    
    auto_select_path = os.path.join(model_dir, "auto_select_request.json")
    if os.path.exists(auto_select_path):
        try:
            isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.SELECTING_MODEL)
            model_creator, model_name, message = auto_select_model(item, sch_ctx)

            interrupted = model_creator is None and model_name is None

            if os.path.exists(auto_select_path):
                os.remove(auto_select_path)
            if not interrupted:
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_SELECTING_MODEL, 
                            extra_items={"model_creator": model_creator, "model_name": model_name, "message": message})


        except Exception as e:

            if os.path.exists(auto_select_path):
                os.remove(auto_select_path)

            trace = traceback.format_exc()
            logger.error("Exception occurred in process_auto_select")
            logger.error(e)
            logger.error(trace)
            try:
                isa.set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_SELECTING_MODEL, 
                                extra_items={"error_setting": "model auto-selection", "error_message": str(e)})
            except:
                trace = traceback.format_exc()
                logger.error("Exception occurred while handling original exception")
                logger.error(e)
                logger.error(trace)






def auto_select_model(item, sch_ctx):
    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    config = yolov4_image_set_driver.create_default_config()
    model_keys.add_general_keys(config)
    model_keys.add_specialized_keys(config)

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")


    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)
    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    is_ortho = metadata["is_ortho"] == "yes"
    object_name = metadata["object_name"]
    # patch_size = annotation_utils.get_patch_size(annotations, None)




    input_shape = (config["inference"]["batch_size"], *(config["arch"]["input_image_shape"]))
    yolov4 = YOLOv4Tiny(config)
    yolov4.build(input_shape=input_shape)
    decoder = Decoder(config)


    MAX_PATCHES = 500
    image_set_candidates = []
    for image_name in annotations.keys():
        image_set_candidates.extend([image_name] * annotations[image_name]["boxes"].shape[0])

    if len(image_set_candidates) <= MAX_PATCHES:
        sample_image_names = image_set_candidates
    else:
        sample_image_names = random.sample(image_set_candidates, MAX_PATCHES)

    models = {}
    for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
        model_dirs = []
        public_models_dir = os.path.join(usr_dir, "models", "available", "public")
        model_dirs.extend(glob.glob(os.path.join(public_models_dir, "*")))
        private_models_dir = os.path.join(usr_dir, "models", "available", "private")
        model_dirs.extend(glob.glob(os.path.join(private_models_dir, "*")))
        for model_dir in model_dirs:
            log_path = os.path.join(model_dir, "log.json")
            log = json_io.load_json(log_path)
            valid = True
            for image_set in log["image_sets"]:
                if image_set["username"] == username and \
                    image_set["farm_name"] == farm_name and \
                    image_set["field_name"] == field_name and \
                    image_set["mission_date"] == mission_date:
                    valid = False
                    break

            if valid and log["model_object"] == object_name:
                models[model_dir] = {
                    "num_detected": 0,
                    "score": 0,
                    "patch_size": round(log["average_patch_size"]),
                    "terminated": False
                }

    if len(models.keys()) == 0:
        raise RuntimeError("No models to choose from!")

    if len(models.keys()) == 1:
        selected_model_dir = list(models.keys())[0]
        selected_model_creator = selected_model_dir.split("/")[2]
        selected_model_name = os.path.basename(selected_model_dir)
        message = "Only one model was available to choose from."
        return selected_model_creator, selected_model_name, message

    # patches = []
    IOU_WEIGHT = 0.5
    CONFIDENCE_WEIGHT = 0.5
    max_score = 0
    num_total_samples = len(sample_image_names)
    num_remaining_samples = num_total_samples
    for image_name in np.unique(sample_image_names):

        image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
        num_image_samples = (np.where(np.array(sample_image_names) == image_name)[0]).size
        # print(image_name)

        sel_inds = random.sample(range(annotations[image_name]["boxes"].shape[0]), num_image_samples)

        for model_dir in models.keys():
            # print("\t{}\t{}".format(model_dir, models[model_dir]["patch_size"]))

            if (models[model_dir]["score"] + num_remaining_samples) < max_score:
                models[model_dir]["terminated"] = True


            if models[model_dir]["terminated"]:
                continue

            patches = ep.extract_box_patches(
                image_path, 
                annotations[image_name]["boxes"][sel_inds], #sample_image_candidates, 
                models[model_dir]["patch_size"],
                is_ortho
            )

            weights_path = os.path.join(model_dir, "weights.h5")
            yolov4.load_weights(weights_path, by_name=False)

            for i in range(0, len(patches), config["inference"]["batch_size"]):

                if sch_ctx["switch_queue"].size() > 0:
                    affected = drain_switch_queue(sch_ctx, cur_image_set_dir=image_set_dir)
                    if affected:
                        return None, None

                batch_patch_arrays = []
                batch_patch_coords = []
                batch_ratios = []
                batch_size = min(config["inference"]["batch_size"], len(patches) - i)
                for j in range(0, batch_size):
                    patch_coords = patches[i+j]["patch_coords"]
                    patch_array = tf.cast(patches[i+j]["patch"], dtype=tf.float32)
                    patch_ratio = np.array(patch_array.shape[:2]) / np.array(config["arch"]["input_image_shape"][:2])
                    patch_array = tf.image.resize(images=patch_array, size=config["arch"]["input_image_shape"][:2])


                    batch_patch_coords.append(patch_coords)
                    batch_patch_arrays.append(patch_array)
                    batch_ratios.append(patch_ratio)

                batch_patch_arrays = tf.stack(batch_patch_arrays, axis=0)

                pred = yolov4(batch_patch_arrays, training=False)
                detections = decoder(pred)

                batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

                batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)


                for j in range(batch_size):

                    pred_bbox = batch_pred_bbox[j]
                    ratio = batch_ratios[j]
                    patch_coords = batch_patch_coords[j]

                    

                    pred_patch_abs_boxes, pred_patch_scores, _ = \
                            yolov4_image_set_driver.post_process_sample(
                                pred_bbox, ratio, patch_coords, config, None, score_threshold=0.25 )#25) #01) #25) #config["inference"]["score_thresh"])


                    box = patches[i+j]["box"]
                    box_patch_coords = (np.array(box) - \
                            np.tile(patch_coords[:2], 2)).astype(np.int32)

                    score_mask = pred_patch_scores > 0.50
                    sel_pred_patch_abs_boxes = pred_patch_abs_boxes[score_mask]
                    sel_pred_patch_scores = pred_patch_scores[score_mask]
                    if sel_pred_patch_abs_boxes.size > 0:

                        iou_mat = box_utils.compute_iou(
                                            # tf.convert_to_tensor(predicted_boxes[i:i+1, :], dtype=tf.float64), 
                                            # tf.convert_to_tensor(annotated_boxes, dtype=tf.float64),

                                            tf.convert_to_tensor(np.expand_dims(box_patch_coords, axis=0), dtype=tf.float64),
                                            tf.convert_to_tensor(sel_pred_patch_abs_boxes, dtype=tf.float64), 
                                            box_format="corners_xy").numpy()
                        # max_inds = np.argmax(iou_mat, axis=0)
                        max_iou_ind = np.argmax(iou_mat[0, :])
                        iou_val = iou_mat[0, max_iou_ind]
                        conf_val = sel_pred_patch_scores[max_iou_ind]
                        detected = iou_val >= 0.50
                        
                        if detected:
                            models[model_dir]["num_detected"] += 1
                            models[model_dir]["score"] += IOU_WEIGHT * iou_val + CONFIDENCE_WEIGHT * conf_val
                            if models[model_dir]["score"] > max_score:
                                max_score = models[model_dir]["score"]


        num_remaining_samples -= num_image_samples

    # print(models)


    model_tuples = []
    for model_dir in models.keys():
        model_tuples.append((model_dir, models[model_dir]["score"]))
    model_tuples.sort(key=lambda x: x[1], reverse=True)

    best_model_dir = model_tuples[0][0]
    best_score = model_tuples[0][1]
    second_best_score = model_tuples[1][1]

    ratio = round(best_score / second_best_score, 2)
    percent_detected = round((models[best_model_dir]["num_detected"] / num_total_samples) * 100)

    message = "This model detected " + str(percent_detected) + "% of the objects it was tested on." + \
               " Its performance was determined to be " + str(ratio) + " times better than the second best model."


    # best_score = 0
    # best_model_dir = None
    # for model_dir in models.keys():
    #     print("{}: {}".format(os.path.basename(model_dir), models[model_dir]["score"]))
    #     if models[model_dir]["score"] > best_score:
    #         best_model_dir = model_dir
    #         best_score = models[model_dir]["score"]
    logger.info("All models:")
    for model_tuple in model_tuples:
        logger.info("{}: {}".format(model_tuple[0], str(model_tuple[1])))
    
    logger.info("Choosing: {}".format(best_model_dir))

    selected_model_creator = best_model_dir.split("/")[2]
    selected_model_name = os.path.basename(best_model_dir)
    return selected_model_creator, selected_model_name, message




# def auto_select_model_alt(item, sch_ctx):
#     logger = logging.getLogger(__name__)

#     username = item["username"]
#     farm_name = item["farm_name"]
#     field_name = item["field_name"]
#     mission_date = item["mission_date"]
#     # object_name = item["object_name"]


#     config = yolov4_image_set_driver.create_default_config()
#     model_keys.add_general_keys(config)
#     model_keys.add_specialized_keys(config)

#     #config["inference"]["batch_size"] = 128

#     image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
#     model_dir = os.path.join(image_set_dir, "model")


#     annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#     annotations = annotation_utils.load_annotations(annotations_path)
#     metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
#     metadata = json_io.load_json(metadata_path)
#     is_ortho = metadata["is_ortho"] == "yes"
#     object_name = metadata["object_name"]
#     patch_size = annotation_utils.get_patch_size(annotations, None)

#     # image_names = list(annotations.keys())
#     image_set_candidates = []
#     for image_name in annotations.keys():
#         image_set_candidates.extend([image_name] * annotations[image_name]["boxes"].shape[0])

#     if len(image_set_candidates) < 10:
#         raise RuntimeError("Auto-selection requires at least 10 annotated boxes.")

#     sample_image_names = random.sample(image_set_candidates, 10) #num_samples)
#     # sample_candidates = image_set_candidates[sample_inds]
#     # image_indices = np.unique(sample_candidates[:, 0])
#     patches = []
#     for image_name in np.unique(sample_image_names):
#         image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
#         num_samples = (np.where(np.array(sample_image_names) == image_name)[0]).size

#         sel_inds = random.sample(range(annotations[image_name]["boxes"].shape[0]), num_samples)

#         # for sel_ind in sel_inds:
#         image_patches = ep.extract_box_patches(
#             image_path, 
#             annotations[image_name]["boxes"][sel_inds], #sample_image_candidates, 
#             patch_size,
#             is_ortho
#         )

#         patches.extend(image_patches)


#     # patches = model_descriptor.get_sample_box_patches(image_set_dir, annotations, sample_rate=1, max_num_samples=10)



#     input_shape = (config["inference"]["batch_size"], *(config["arch"]["input_image_shape"]))
#     yolov4 = YOLOv4Tiny(config)
#     yolov4.build(input_shape=input_shape)
#     decoder = Decoder(config)

#     performance = {}
#     max_num_detected = 0
#     max_num_detected_model_dir = None
#     start_time = time.time()
#     for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
#         model_dirs = []
#         public_models_dir = os.path.join(usr_dir, "models", "available", "public")
#         model_dirs.extend(glob.glob(os.path.join(public_models_dir, "*")))
#         private_models_dir = os.path.join(usr_dir, "models", "available", "private")
#         model_dirs.extend(glob.glob(os.path.join(private_models_dir, "*")))
#         for model_dir in model_dirs:
#             log_path = os.path.join(model_dir, "log.json")
#             log = json_io.load_json(log_path)
#             if log["model_object"] == object_name: #in ["canola_seedling", "wheat_head"]: #== object_name:
#             # if log["model_name"] == "Arvalis_2":
            
#                 # print("NOW PROCESSING", log["model_name"])

#                 if sch_ctx["switch_queue"].size() > 0:
#                     affected = drain_switch_queue(sch_ctx, cur_image_set_dir=image_set_dir)
#                     if affected:
#                         return None, None


#                 num_detected = 0
#                 confidence_scores = []
#                 weights_path = os.path.join(model_dir, "weights.h5")
#                 yolov4.load_weights(weights_path, by_name=False)

#                 for i in range(0, len(patches), config["inference"]["batch_size"]):

#                     batch_patch_arrays = []
#                     batch_patch_coords = []
#                     batch_ratios = []
#                     batch_size = min(config["inference"]["batch_size"], len(patches) - i)
#                     for j in range(0, batch_size):
#                         patch_coords = patches[i+j]["patch_coords"]
#                         patch_array = tf.cast(patches[i+j]["patch"], dtype=tf.float32)
#                         patch_ratio = np.array(patch_array.shape[:2]) / np.array(config["arch"]["input_image_shape"][:2])
#                         patch_array = tf.image.resize(images=patch_array, size=config["arch"]["input_image_shape"][:2])


#                         batch_patch_coords.append(patch_coords)
#                         batch_patch_arrays.append(patch_array)
#                         batch_ratios.append(patch_ratio)

#                     batch_patch_arrays = tf.stack(batch_patch_arrays, axis=0)

                    
#                     # batch_images = data_augment.apply_inference_transform(batch_images, transform_type)


#                     pred = yolov4(batch_patch_arrays, training=False)
#                     detections = decoder(pred)

#                     batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

#                     batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)


#                     for j in range(batch_size):

#                         pred_bbox = batch_pred_bbox[j]
#                         ratio = batch_ratios[j]
#                         patch_coords = batch_patch_coords[j]

                        

#                         pred_patch_abs_boxes, pred_patch_scores, _ = \
#                                 yolov4_image_set_driver.post_process_sample(
#                                     pred_bbox, ratio, patch_coords, config, None, score_threshold=0.25 )#25) #01) #25) #config["inference"]["score_thresh"])


#                         box = patches[i+j]["box"]
#                         box_patch_coords = (np.array(box) - \
#                                 np.tile(patch_coords[:2], 2)).astype(np.int32)

#                         score_mask = pred_patch_scores > 0.50
#                         sel_pred_patch_abs_boxes = pred_patch_abs_boxes[score_mask]
#                         if sel_pred_patch_abs_boxes.size > 0:

#                             iou_mat = box_utils.compute_iou(
#                                                 # tf.convert_to_tensor(predicted_boxes[i:i+1, :], dtype=tf.float64), 
#                                                 # tf.convert_to_tensor(annotated_boxes, dtype=tf.float64),

#                                                 tf.convert_to_tensor(np.expand_dims(box_patch_coords, axis=0), dtype=tf.float64),
#                                                 tf.convert_to_tensor(sel_pred_patch_abs_boxes, dtype=tf.float64), 
#                                                 box_format="corners_xy").numpy()
#                             # max_inds = np.argmax(iou_mat, axis=0)
#                             max_iou_val = np.max(iou_mat[0, :])
#                             detected = max_iou_val >= 0.50
#                             if detected:
#                                 num_detected += 1
#                         confidence_scores.extend(pred_patch_scores)
#                         # confidence_score = get_confidence_quality(pred_patch_scores, min_score_thresh=0.25)
#                         # confidence_scores.append(confidence_score)
                            
#                         # max_vals = np.take_along_axis(iou_mat, np.expand_dims(max_inds, axis=0), axis=0)[0]
#                         # mask = max_vals >= 0.5 #iou_thresh
#                 # model_confidence_score = np.mean(confidence_scores)
#                 performance[model_dir] = {
#                     "num_detected": num_detected,
#                     "confidence_scores": confidence_scores
#                 }
#                 if num_detected > max_num_detected:
#                     max_num_detected = num_detected
#                     max_num_detected_model_dir = model_dir
#                 # print("\t\t{}: detected ({} / {})".format(log["model_name"], 
#                 # num_detected, len(patches)))
#     if len(performance.keys()) == 0:
#         raise RuntimeError("Unable to find any available models to choose from.")

#     if max_num_detected == 0:
#         logger.info("All models are terrible. Picking one at random.")
#         selected_model_dir = list(performance.keys())[0]
#         selected_model_creator = selected_model_dir.split("/")[2]
#         selected_model_name = os.path.basename(selected_model_dir)
#         return selected_model_creator, selected_model_name
#     # if max_num_detected == 1:
#     #     logger.info("All models are very bad. Picking one that detected one object.")
#     #     return max_num_detected_model_dir

#     candidates = {}
#     candidates_num_detected_tuples = []
#     # if max_num_detected_model_dir is not None:
#     for model_dir in performance.keys():
#         # if model_dir != max_num_detected_model_dir:
#         if performance[model_dir]["num_detected"] >= max_num_detected - 3 and performance[model_dir]["num_detected"] > 0:
#             candidates[model_dir] = performance[model_dir]["confidence_scores"]
#             candidates_num_detected_tuples.append((model_dir, performance[model_dir]["num_detected"]))

#     if len(candidates.keys()) == 1:
#         selected_model_dir = list(performance.keys())[0]
#         selected_model_creator = selected_model_dir.split("/")[2]
#         selected_model_name = os.path.basename(selected_model_dir)
#         logger.info("One model clearly out-performed the others. Picking: {}".format(model_dir))
#         return selected_model_creator, selected_model_name

#     candidates_num_detected_tuples.sort(key=lambda x: x[1], reverse=True)
#     sorted_candidate_names = [x[0] for x in candidates_num_detected_tuples]

#     # print("sorted_candidate_names", sorted_candidate_names)


#     NUM_PATCHES_FOR_CANDIDATES = 100000 #1024 #* 4 #100000 #1024 #* 2
#     num_patches_per_image = m.ceil(NUM_PATCHES_FOR_CANDIDATES / len(list(annotations.keys())))


#     NUM_REPS = 1 #8
#     # print("num_patches", len(patches))
#     candidate_performance = {}
#     final_candidate_quality_scores = {}
#     best_model_dir = None
#     best_score = 0
#     # print("Processing candidates:")
#     # for rep_num in range(NUM_REPS):
#     extraction_start_time = time.time()
#     patches = []
#     for image_path in glob.glob(os.path.join(image_set_dir, "images", "*")):

#         if sch_ctx["switch_queue"].size() > 0:
#             affected = drain_switch_queue(sch_ctx, cur_image_set_dir=image_set_dir)
#             if affected:
#                 return None, None


#         patches.extend(ep.extract_random_patches(image_path, num_patches_per_image, patch_size, is_ortho))
#     # patches = model_descriptor.get_sample_box_patches(image_set_dir, annotations, sample_rate=1, max_num_samples=1024)
#     random.shuffle(patches)
#     extraction_end_time = time.time()
#     elapsed = extraction_end_time - extraction_start_time
#     logger.info("Took {} seconds to extract {} patches.".format(elapsed, len(patches)))

#     for model_dir in sorted_candidate_names:
#         # print("Processing candidate", model_dir)
#         stabilized = False
#         quality_scores = []
#         confidence_scores = candidates[model_dir].copy()
#         # print("num_confidence_scores", len(confidence_scores))
#         weights_path = os.path.join(model_dir, "weights.h5")

#         yolov4.load_weights(weights_path, by_name=False)

#         for i in range(0, len(patches), config["inference"]["batch_size"]):


#             if sch_ctx["switch_queue"].size() > 0:
#                 affected = drain_switch_queue(sch_ctx, cur_image_set_dir=image_set_dir)
#                 if affected:
#                     return None, None

#             if stabilized:
#                 break

#             batch_patch_arrays = []
#             batch_patch_coords = []
#             batch_ratios = []
#             batch_size = min(config["inference"]["batch_size"], len(patches) - i)
            
#             #min(i+config["inference"]["batch_size"], len(patches))
#             # print("batch_size", batch_size)
#             for j in range(0, batch_size):
#                 patch_coords = patches[i+j]["patch_coords"]
#                 patch_array = tf.cast(patches[i+j]["patch"], dtype=tf.float32)
#                 patch_ratio = np.array(patch_array.shape[:2]) / np.array(config["arch"]["input_image_shape"][:2])
#                 patch_array = tf.image.resize(images=patch_array, size=config["arch"]["input_image_shape"][:2])


#                 batch_patch_coords.append(patch_coords)
#                 batch_patch_arrays.append(patch_array)
#                 batch_ratios.append(patch_ratio)

#             batch_patch_arrays = tf.stack(batch_patch_arrays, axis=0)

#             # print(tf.shape(batch_patch_arrays))

            
#             # batch_images = data_augment.apply_inference_transform(batch_images, transform_type)


#             pred = yolov4(batch_patch_arrays, training=False)
#             detections = decoder(pred)

#             batch_pred_bbox = [tf.reshape(x, (batch_size, -1, tf.shape(x)[-1])) for x in detections]

#             batch_pred_bbox = tf.concat(batch_pred_bbox, axis=1)


#             for j in range(batch_size):

#                 pred_bbox = batch_pred_bbox[j]
#                 ratio = batch_ratios[j]
#                 patch_coords = batch_patch_coords[j]

                

#                 pred_patch_abs_boxes, pred_patch_scores, _ = \
#                         yolov4_image_set_driver.post_process_sample(
#                             pred_bbox, ratio, patch_coords, config, None, score_threshold=0.25) #25) #01) #25) #config["inference"]["score_thresh"])


#                 # patch_height = patch_coords[2] - patch_coords[0]
#                 # patch_width = patch_coords[3] - patch_coords[1]
#                 # accept_bottom = patch_coords[0] + round(patch_height / 4) # - wiggle_room
#                 # accept_left = patch_coords[1] + round(patch_width / 4) #- wiggle_room
#                 # accept_top = patch_coords[2] - round(patch_height / 4) #+ wiggle_room
#                 # accept_right = patch_coords[3] - round(patch_width / 4) #+ wiggle_room


#                 # box_centres = (pred_patch_abs_boxes[..., :2] + pred_patch_abs_boxes[..., 2:]) / 2.0

#                 # # print("box_centres", box_centres)
#                 # # print("accept_bottom: {}, accept_left: {}, accept_top: {}, accept_right: {}".format(
#                 # #     accept_bottom, accept_left, accept_top, accept_right
#                 # # ))
#                 # mask = np.logical_and(
#                 #     np.logical_and(box_centres[:,0] >= accept_bottom, box_centres[:,0] < accept_top),
#                 #     np.logical_and(box_centres[:,1] >= accept_left, box_centres[:,1] < accept_right)
#                 # )

#                 # if pred_patch_scores.size > 0:
#                 # confidence_score = get_confidence_quality(pred_patch_scores, min_score_thresh=0.25)
#                 confidence_scores.extend(pred_patch_scores)

#             quality = get_confidence_quality(np.array(confidence_scores))
#             quality_scores.append(quality)
#             if model_dir not in candidate_performance:
#                 candidate_performance[model_dir] = []
#             candidate_performance[model_dir].append(quality)
#             if len(quality_scores) > 10:
#                 if np.all(np.isclose(quality_scores[len(quality_scores)-10:], quality_scores[len(quality_scores)-10], atol=0.005)):
#                     # print("{}: distribution stabilized after {} batches".format(os.path.basename(model_dir), len(quality_scores)))
#                     stabilized = True
#                     final_candidate_quality_scores[model_dir] = quality_scores[-1]

#                 if len(final_candidate_quality_scores.values()) > 0 and quality_scores[-1] < max(final_candidate_quality_scores.values()) - 0.1:
#                     # print("Terminating {}".format(os.path.basename(model_dir)))
#                     stabilized = True
#                     final_candidate_quality_scores[model_dir] = quality_scores[-1]

#             # print("num quality scores:", len(quality_scores))
#             # print("quality_scores", quality_scores)


    
            

#         # confidence_scores = np.array(confidence_scores)
#         # score = get_confidence_quality(confidence_scores) #, min_score_thresh=0.25)
#         # # score = np.mean(confidence_scores)
#         # print("Candidate: {}. Num conf scores: {}. Quality Score: {}".format(os.path.basename(model_dir), confidence_scores.size, score))
#         # if model_dir not in candidate_performance:
#         #     candidate_performance[model_dir] = []
#         # candidate_performance[model_dir].append(score)

#         # # if score > best_score:
#         # #     best_score = score
#         # #     best_model_dir = model_dir

#         # # if rep_num == 0:
#         #     #for i, model_dir in enumerate(candidate_performance.keys()):
#         # fig = plt.figure(figsize=(16, 8))
#         # ax = fig.add_subplot(111)
#         # print("num_scores", confidence_scores.size)
#         # counts, bins = np.histogram(confidence_scores, bins=np.arange(0, 1, 0.01), range=(0, 1))
#         # print("counts", counts)
#         # plt.stairs(counts, bins)
#         # plt.savefig("score_histograms/rep_" + str(rep_num) + "-" + os.path.basename(model_dir) + ".svg")



#     # best_model = None
#     # max_confidence_score = 0
#     # for model in candidates.keys():
#     #     if candidates[model] > max_confidence_score:
#     #         max_confidence_score = candidates[model]
#     #         best_model = model

#     # exit()
#     candidate_scores = []
#     for model in final_candidate_quality_scores.keys():
#         candidate_score = final_candidate_quality_scores[model]
#         candidate_scores.append((model, candidate_score))
#     candidate_scores.sort(key=lambda x: x[1])
#     # for candidate_score in candidate_scores:
#     #     print("Model: {}. Score: {}".format(os.path.basename(candidate_score[0]), candidate_score[1]))
#     # print("---")
#     # print("Best Model: {}".format(os.path.basename(candidate_scores[-1][0])))
    
#     selected_model_creator = candidate_scores[-1][0].split("/")[2]
#     selected_model_name = os.path.basename(candidate_scores[-1][0])

#     return selected_model_creator, selected_model_name



#     colors = ["blue", "red", "green", "purple", "orange", "pink", "black"]
#     fig = plt.figure(figsize=(16, 8))
#     ax = fig.add_subplot(111)
#     for i, model_dir in enumerate(candidate_performance.keys()):
#         min_val = min(candidate_performance[model_dir])
#         max_val = max(candidate_performance[model_dir])
#         ax.plot(np.arange(0, len(candidate_performance[model_dir]), 1), candidate_performance[model_dir], c=colors[i], label=os.path.basename(model_dir))
#         ax.plot([0, len(candidate_performance[model_dir])], [min_val, min_val], c=colors[i], linestyle="--", linewidth=1)
#         ax.plot([0, len(candidate_performance[model_dir])], [max_val, max_val], c=colors[i], linestyle="--", linewidth=1)
#         # ax.plot(np.arange(0, NUM_REPS, 1), candidate_performance[model_dir], c=colors[i], label=os.path.basename(model_dir))
#         # ax.plot([0, NUM_REPS-1], [min_val, min_val], c=colors[i], linestyle="--", linewidth=1)
#         # ax.plot([0, NUM_REPS-1], [max_val, max_val], c=colors[i], linestyle="--", linewidth=1)


#     ax.legend()
#     plt.tight_layout()
#     fig.savefig("candidate_scores_sigmoid_function_convergence.svg")

#     end_time = time.time()
#     elapsed = end_time - start_time
#     print(candidate_performance)
#     # logger.info("Finished fetching available models. Took {} seconds.".format(elapsed))
#     # logger.info("Best model: {}".format(best_model_dir))




# def the_new_batch(image_set_dir, annotations, patch_size, is_ortho):

#     NUM_PATCHES_FOR_CANDIDATES = 10000 #1024 #* 4 #100000 #1024 #* 2
#     num_patches_per_image = m.ceil(NUM_PATCHES_FOR_CANDIDATES / len(list(annotations.keys())))

#     patches = []
#     for image_path in glob.glob(os.path.join(image_set_dir, "images", "*")):
#             patches.extend(ep.extract_random_patches(image_path, num_patches_per_image, patch_size, is_ortho))
#     patches = np.array(patches)
#     num_patches = patches.size

#     model, model_input_shape = model_descriptor.get_feature_vector_model()
#     batch_size = 256 #1024
#     features_lst = []
#     for i in range(0, num_patches, batch_size):
#         batch_patches = []
#         # batch_patches = []
#         for j in range(i, min(i+batch_size, num_patches)):
#             patch = tf.convert_to_tensor(patches[j], dtype=tf.float32)
#             patch = tf.image.resize(images=patch, size=model_input_shape[:2])
#             batch_patches.append(patch)
#         batch_patches = tf.stack(values=batch_patches, axis=0)


#         batch_patches = tf.keras.applications.vgg16.preprocess_input(batch_patches)
#         features = model.predict(batch_patches)
#         # max_pool_2d = 
#         for j, f in enumerate(features):
#             # print(f.shape)
#             # # f = np.apply_over_axes(np.sum, f, [0, 1]).flatten()
            
#             # print(f.shape)
#             # exit()
#             # f = f.flatten()

#             patch = patches[i+j]
#             # height = patch.shape[0]
#             # width = patch.shape[1]
#             # ratio = max(height, width) / min(height, width)

#             # f = normalize(f)

#             feature = [] #[height, width, ratio]
#             feature.extend(f.tolist())

#             features_lst.append(feature)

#     features_array = np.array(features_lst)



# def auto_select_model_2(item):

#     logger = logging.getLogger(__name__)

#     username = item["username"]
#     farm_name = item["farm_name"]
#     field_name = item["field_name"]
#     mission_date = item["mission_date"]
#     object_name = item["object_name"]

#     logger.info("Creating target image set vector...")
#     start_time = time.time()

#     model, model_input_shape = model_descriptor.get_feature_vector_model()
#     image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
#     annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
#     annotations = annotation_utils.load_annotations(annotations_path)
#     target_vectors = model_descriptor.create_image_set_vectors(model, model_input_shape, image_set_dir, annotations, 
#                                         sample_rate=1.0, max_num_samples=20)
#     target_vectors = np.array(target_vectors)
#     # print("target_vectors.shape", target_vectors.shape)
#     # print("target_vectors", target_vectors)

#     end_time = time.time()
#     elapsed = end_time - start_time
#     logger.info("Finished creating target image set vector. Took {} seconds.".format(elapsed))
    
#     vector_ind_to_model = {}
#     candidate_vectors = []
#     best_score = 1e10
#     best_match = None
#     vector_ind = 0
#     scores = []
#     tsne_scores = []
#     standardize = True
    
#     logger.info("Fetching available models...")
#     start_time = time.time()
#     for usr_dir in glob.glob(os.path.join("usr", "data", "*")):
#         public_models_dir = os.path.join(usr_dir, "models", "available", "public")
#         for model_dir in glob.glob(os.path.join(public_models_dir, "*")):
#             log_path = os.path.join(model_dir, "log.json")
#             log = json_io.load_json(log_path)
#             if log["model_object"] in ["canola_seedling", "wheat_head"]: #== object_name:
#             # if log["model_name"] == "Arvalis_2":
#                 # print("NOW PROCESSING", log["model_name"])
#                 valid = True
#                 # for image_set in log["image_sets"]:

#                 #     if image_set["username"] == username and \
#                 #         image_set["farm_name"] == farm_name and \
#                 #         image_set["field_name"] == field_name and \
#                 #         image_set["mission_date"] == mission_date:

#                 #         valid = False
#                 #         break

#                 if valid:
#                     feature_vectors_path = os.path.join(model_dir, "feature_vectors.npy")
#                     source_vectors = np.load(feature_vectors_path)
#                     # feature_vectors = json_io.load_json(feature_vectors_path)
#                     # source_vectors = np.array(feature_vectors["feature_vectors"])

#                     standardized_target_vectors = np.copy(target_vectors)
#                     standardized_source_vectors = np.copy(source_vectors)


#                     # if standardize:
#                     #     # all_vectors = source_vectors 
#                     #     all_vectors = np.concatenate([target_vectors, source_vectors])
#                     #     if np.unique(all_vectors, axis=0).shape == all_vectors.shape:
#                     #         print("BEFORE STANDARDIZE: All row vectors are unique.")
#                     #     else:
#                     #         print("BEFORE STANDARDIZE: Not all row vectors are unique")
#                     #     vectors_mean = np.mean(all_vectors, axis=0)
#                     #     vectors_std = np.std(all_vectors, axis=0)

#                     #     valid_cols = vectors_std != 0
#                     #     print("{} invalid columns".format(np.sum(vectors_std == 0)))
#                     #     # print("valid_cols", valid_cols)
#                     #     if np.all(np.logical_not(valid_cols)):
#                     #         raise RuntimeError("Unable to standardize feature vectors.")
#                     #     all_vectors = all_vectors[:, valid_cols]
#                     #     # source_vectors = source_vectors[:, valid_cols]
#                     #     # target_vectors = target_vectors[:, valid_cols]
#                     #     vectors_mean = vectors_mean[valid_cols]
#                     #     vectors_std = vectors_std[valid_cols]

#                     #     all_vectors = np.divide((all_vectors - vectors_mean), vectors_std)
#                     #     if np.unique(all_vectors, axis=0).shape == all_vectors.shape:
#                     #         print("AFTER STANDARDIZE: All row vectors are unique.")
#                     #     else:
#                     #         print("AFTER STANDARDIZE: Not all row vectors are unique")

#                     #     standardized_target_vectors = all_vectors[:target_vectors.shape[0], :]
#                     #     standardized_source_vectors = all_vectors[target_vectors.shape[0]:, :]

#                     #     # source_vectors = np.divide((source_vectors - vectors_mean), vectors_std)
#                     #     # target_vectors = np.divide((target_vectors - vectors_mean), vectors_std)









#                     tsne_score = create_tsne(os.path.join(model_dir, "tsne.png"), standardized_target_vectors, standardized_source_vectors, "target", os.path.basename(model_dir))
#                     tsne_scores.append((tsne_score, os.path.basename(model_dir)))
#                     score = get_image_set_match_score(standardized_target_vectors, standardized_source_vectors)
#                     scores.append((score, os.path.basename(model_dir)))

#                     # for image_set in log["image_sets"]:
#                     #     print("valid: {}-{}-{}-{}".format(image_set["username"], 
#                     #             image_set["farm_name"], image_set["field_name"], image_set["mission_date"]))


#                     #     feature_vectors_path = os.path.join(model_dir, "feature_vectors.json")
#                     #     feature_vectors = json_io.load_json(feature_vectors_path)
#                     #     # for feature_vector in feature_vectors:
#                     #     for i in range(len(feature_vectors["feature_vectors"])):
#                     #         vector_ind_to_model[vector_ind] = {
#                     #             "username": image_set["username"],
#                     #             "farm_name": image_set["farm_name"],
#                     #             "field_name": image_set["field_name"],
#                     #             "mission_date": image_set["mission_date"]
#                     #         }
#                     #         candidate_vectors.append(feature_vectors["feature_vectors"][i])
#                     #         vector_ind += 1


#                     # print("Model: {}. Score: {}".format(os.path.basename(model_dir), score))

#                     # if score < best_score:
#                     #     best_score = score
#                     #     best_match = os.path.basename(model_dir)


#     end_time = time.time()
#     elapsed = end_time - start_time
#     logger.info("Finished fetching available models. Took {} seconds.".format(elapsed))
#     sorted_scores = sorted(scores, key=lambda x: x[0])
#     # for score in sorted_scores:
#     #     print("{:35}: {}".format(score[1], score[0]))
#     # print("Best match: {}".format(sorted_scores[0][1]))
#     # print()
#     # print("---")
#     # print()
#     sorted_tsne_scores = sorted(tsne_scores, key=lambda x: x[0])
#     # for score in sorted_tsne_scores:
#     #     print("{:35}: {}".format(score[1], score[0]))
#     # print("Best TSNE match: {}".format(sorted_tsne_scores[0][1]))
    
#     # candidate_vectors = np.array(candidate_vectors)

#     # # all_vectors = candidate_vectors
#     # # all_vectors.append(image_set_vector.tolist())
#     # # all_vectors = np.array(all_vectors)
#     # print("candidate_vectors.shape: {}".format(candidate_vectors.shape))

#     # # old_settings = np.seterr(all='ignore')
#     # # np.seterr(over='raise')


#     # candidates_mean = np.mean(candidate_vectors, axis=0)
#     # candidates_std = np.std(candidate_vectors, axis=0)

#     # candidate_vectors = np.divide((candidate_vectors - candidates_mean), 
#     #                                     candidates_std)

#     # invalid_cols = np.logical_or(np.any(np.isinf(candidate_vectors), axis=0), np.any(np.isnan(candidate_vectors), axis=0))
#     # print("invalid_cols", invalid_cols)
#     # candidate_vectors = candidate_vectors[:, np.logical_not(invalid_cols)]
#     # if candidate_vectors.size == 0:
#     #     raise RuntimeError("Unable to auto-select model.")

#     # print("candidate_vectors.shape: {}".format(candidate_vectors.shape))
#     # # candidate_vectors = candidate_vectors[:all_vectors.shape[0]-1, :]
#     # query_vector = np.divide((image_set_vector - candidates_mean), 
#     #                                     candidates_std)
#     # tree = KDTree(candidate_vectors, leaf_size=2)
#     # dists, inds = tree.query(np.expand_dims(query_vector, axis=0), k=candidate_vectors.shape[0]) #3)

#     # print("dists", dists)
#     # print("inds", inds)

#     # for ind, dist in zip(inds[0], dists[0]):
#     #     print("{}: {}".format(vector_ind_to_model[ind], dist))

#     # # logger.info("Chose {}".format(vector_ind_to_model[inds[0][0]]))




def test():
    logging.basicConfig(level=logging.INFO)

    item = {}
    item["username"] = "erik"
    # item["farm_name"] = "Arvalis_9"
    # item["field_name"] = "France"
    # item["mission_date"] = "2022-11-22"
    # item["object_name"] = "wheat_head"
    # item["farm_name"] = "ULiege-GxABT_1"
    # item["field_name"] = "Belgium"
    # item["mission_date"] = "2022-11-22"
    # item["object_name"] = "wheat_head"

    # item["farm_name"] = "BlaineLake"
    # item["field_name"] = "HornerWest"
    # item["mission_date"] = "2021-06-09"
    # item["object_name"] = "canola_seedling"

    item["farm_name"] = "UNI"
    item["field_name"] = "LowN2"
    item["mission_date"] = "2021-06-07"
    item["object_name"] = "canola_seedling"

    # item["farm_name"] = "Saskatoon"
    # item["field_name"] = "Norheim4"
    # item["mission_date"] = "2022-05-24"
    # item["object_name"] = "canola_seedling"

    # item["farm_name"] = "MORSE"
    # item["field_name"] = "Nasser"
    # item["mission_date"] = "2022-05-27"
    # item["object_name"] = "canola_seedling"


    # item["farm_name"] = "row_spacing"
    # item["field_name"] = "brown"
    # item["mission_date"] = "2021-06-01"
    # item["object_name"] = "canola_seedling"

    # item["farm_name"] = "Arvalis_11"
    # item["field_name"] = "France"
    # item["farm_name"] = "ETHZ_1"
    # item["field_name"] = "Switzerland"
    # item["mission_date"] = "2022-11-22"
    # item["object_name"] = "wheat_head"

    auto_select_model(item)


if __name__ == "__main__":
    test()