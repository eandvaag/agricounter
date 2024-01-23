import logging
import argparse
import tqdm
import os
import time
import random
import math as m
import numpy as np
import tensorflow as tf
from mean_average_precision import MetricBuilder
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry
import shapely.ops


import pandas as pd
import pandas.io.formats.excel
from natsort import index_natsorted

#from styleframe import StyleFrame



from models.common import box_utils, annotation_utils, poly_utils


from io_utils import json_io, w3c_io






def get_mAP_val(annotations, full_predictions, iou_thresh, assessment_images):
    
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    mAP_vals = []
    for image_name in tqdm.tqdm(assessment_images):
        annotated_boxes = np.array(annotations[image_name]["boxes"])
        predicted_scores = np.array(full_predictions[image_name]["scores"])
        predicted_boxes = np.array(full_predictions[image_name]["boxes"])

        # predicted_boxes = predicted_boxes[predicted_scores > 0.01]
        # predicted_scores = predicted_scores[predicted_scores > 0.01]


        annotated_classes = np.zeros(shape=(annotated_boxes.shape[0]))
        predicted_classes = np.zeros(shape=(predicted_boxes.shape[0]))

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
            predicted_boxes, 
            predicted_classes, 
            predicted_scores,
            annotated_boxes,
            annotated_classes)

        metric_fn.add(pred_for_mAP, true_for_mAP)

    if iou_thresh == ".50:.05:.95":
        mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".50":
        mAP = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".75":
        mAP = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".20":
        mAP = metric_fn.value(iou_thresholds=0.2, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    
    else:
        raise RuntimeError("Invalid IoU threshold: {}".format(iou_thresh))

        # mAP_vals.append(mAP)
    return mAP #np.mean(mAP_vals) #mAP





def get_global_accuracy(annotations, full_predictions, assessment_images, iou_thresh=0.5):

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for image_name in assessment_images:
        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(full_predictions[image_name]["boxes"])
        pred_scores = np.array(full_predictions[image_name]["scores"])

        sel_pred_boxes = pred_boxes[pred_scores > 0.50]

        num_predicted = sel_pred_boxes.shape[0]
        num_annotated = annotated_boxes.shape[0]

        if num_predicted > 0:
            if num_annotated > 0:
                true_positive, false_positive, false_negative = get_positives_and_negatives(annotated_boxes, sel_pred_boxes, iou_thresh=iou_thresh)
            else:
                true_positive = 0
                false_positive = num_predicted
                false_negative = 0
        else:
            if num_annotated > 0:
                true_positive = 0
                false_positive = 0
                false_negative = num_annotated
            else:
                true_positive = 0
                false_positive = 0
                false_negative = 0

        total_true_positives += true_positive
        total_false_positives += false_positive
        total_false_negatives += false_negative

    global_accuracy = total_true_positives / (total_true_positives + total_false_positives + total_false_negatives)
    return global_accuracy




# def DiC(actual, pred):
#     return actual - pred

# def abs_DiC(actual, pred):
#     return np.abs(actual - pred)

# def pct_DiC(actual, pred):
#     return np.divide(abs_DiC(actual, pred), actual, out=np.zeros_like(actual, dtype=np.float64), where=actual!=0) * 100

# def nonzero_abs_DiC(actual, pred):
#     mask = actual != 0
#     return abs_DiC(actual[mask], pred[mask])

# def squared_DiC(actual, pred):
#     return (actual - pred) ** 2

# def r_squared(actual, pred):
#     SS_res = np.sum((actual - pred) ** 2)
#     SS_tot = np.sum((actual - np.mean(actual)) ** 2)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         res = 1 - (SS_res / SS_tot)
#     if np.isnan(res):
#         res = 0.0 
#     return float(res)



# def boxplot_data(data):

#     data = np.array(data)
#     if data.size == 0:
#         res = {
#             "range_min": 0,
#             "whisker_min": 0,
#             "q1": 0,
#             "q2": 0,
#             "q3": 0,
#             "whisker_max": 0,
#             "range_max": 0,
#             "outliers": []
#         }
#     else:
#         q1, q2, q3 = np.percentile(data, [25, 50, 75], interpolation="midpoint")
#         iqr = q3 - q1

#         res = {
#             "range_min": float(np.min(data)),
#             "whisker_min": q1 - 1.5 * iqr,
#             "q1": q1,
#             "q2": q2,
#             "q3": q3,
#             "whisker_max": q3 + 1.5 * iqr,
#             "range_max": float(np.max(data))
#         }

#         res["outliers"] = data[np.logical_or(data < res["whisker_min"],
#                                              data > res["whisker_max"])].tolist()

#     return res




# # def five_num_summary(data):
# #     return np.percentile(np.array(data), [0, 25, 50, 75, 100], interpolation='midpoint').tolist()

# # def five_num_summary_DiC(actual, pred):
# #     return five_num_summary(actual - pred)

# # def DiC_occurrences(actual, pred):
# #     a = (actual - pred)
# #     vals = np.arange(a.min(), a.max() + 1)
# #     occurrences, _ = np.histogram(a, bins=(a.max() - a.min() + 1))
# #     return {
# #             "vals": vals.tolist(),
# #             "occurrences": occurrences.tolist()
# #            }


# def collect_statistics(image_names, metrics, predictions, config, inference_times=None):


#     # if "metrics" not in predictions:
#     #     predictions["metrics"] = {}

#     # if "point" not in predictions["metrics"]:
#     #     predictions["metrics"]["point"] = {}

#     # if "boxplot" not in predictions["metrics"]:
#     #     predictions["metrics"]["boxplot"] = {}

#     # datasets = {
#     #     "training": image_set.training_dataset,
#     #     "validation": image_set.validation_dataset,
#     #     "test", image_set.test_dataset
#     # }
#     #dataset = image_set.all_dataset

#     point_metrics = metrics["point"]
#     boxplot_metrics = metrics["boxplot"]

#     if inference_times is not None:
#         total_inference_time = float(np.sum(inference_times))
#         point_metrics["Total Inference Time (s)"] = {}
#         point_metrics["Total Inference Time (s)"]["---"] = total_inference_time
#         point_metrics["Per Patch Inference Time (s)"] = {}
#         point_metrics["Per Patch Inference Time (s)"]["---"] = total_inference_time / len(inference_times)
#         point_metrics["Per Image Inference Time (s)"] = {}
#         point_metrics["Per Image Inference Time (s)"]["---"] = total_inference_time / len(predictions["image_predictions"])

#     #boxplot_metrics["Inference Times (Per Patch)"] = {}
#     #boxplot_metrics["Inference Times (Per Patch)"]["---"] = boxplot_data(inference_times)

#     # point_metrics["Mean Confidence Score"] = {}
#     # point_metrics["Max Confidence Score"] = {}
#     # point_metrics["Max Box Area"] = {}
#     # point_metrics["Min Box Area"] = {}
#     # point_metrics["Mean Box Area"] = {}

#     boxplot_metrics["Confidence"] = {}
#     boxplot_metrics["Box Area"] = {}

#     confidences = {k: [] for k in config["arch"]["class_map"].keys()}
#     boxes = {k: [] for k in config["arch"]["class_map"].keys()}
#     #for image in dataset.images:
#     for image_name in image_names: #predictions["image_predictions"].keys():
#         for cls_name in config["arch"]["class_map"].keys():
#             cls_confs = predictions["image_predictions"][image_name]["pred_class_scores"][cls_name] 
#             confidences[cls_name].extend(cls_confs)

#             cls_boxes = predictions["image_predictions"][image_name]["pred_class_boxes"][cls_name]
#             boxes[cls_name].extend(cls_boxes)

#     num_predictions = np.sum(len(confidences[k]) for k in confidences.keys())

#     # point_metrics["Mean Confidence Score"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Max Confidence Score"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Mean Box Area"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Min Box Area"]["Cross-Class Weighted Average"] = 0
#     # point_metrics["Max Box Area"]["Cross-Class Weighted Average"] = 0

#     for cls_name in config["arch"]["class_map"].keys():

#         if len(confidences[cls_name]) > 0:
#             mean_cls_conf = float(np.mean(confidences[cls_name]))
#             max_cls_conf = float(np.max(confidences[cls_name]))
#             box_areas = box_utils.box_areas_np(np.array(boxes[cls_name]))
#             mean_box_area = float(np.mean(box_areas))
#             min_box_area = float(np.min(box_areas))
#             max_box_area = float(np.max(box_areas))

#         else:
#             mean_cls_conf = 0
#             max_cls_conf = 0
#             box_areas = np.array([])
#             mean_box_area = 0
#             min_box_area = 0
#             max_box_area = 0


#         boxplot_metrics["Confidence"][cls_name] = boxplot_data(confidences[cls_name])
#         boxplot_metrics["Box Area"][cls_name] = boxplot_data(box_areas)

#         # point_metrics["Mean Confidence Score"][cls_name] = mean_cls_conf
#         # point_metrics["Max Confidence Score"][cls_name] = max_cls_conf
#         # point_metrics["Mean Box Area"][cls_name] = mean_box_area
#         # point_metrics["Max Box Area"][cls_name] = max_box_area
#         # point_metrics["Min Box Area"][cls_name] = min_box_area
#         # if num_predictions > 0:
#         #     point_metrics["Mean Confidence Score"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * mean_cls_conf
#         #     point_metrics["Max Confidence Score"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * max_cls_conf

#         #     point_metrics["Mean Box Area"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * mean_box_area
#         #     point_metrics["Min Box Area"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * min_box_area
#         #     point_metrics["Max Box Area"]["Cross-Class Weighted Average"] += \
#         #         (len(confidences[cls_name]) / num_predictions) * max_box_area


# def calculate_optimal_score_threshold(annotations, predictions, image_names):

#     # currently assumes only class is plant class


#     logger = logging.getLogger(__name__)    

#     optimal_thresh_val = None
#     optimal_mean_abs_diff = np.inf
#     prev_mean_abs_diff = np.inf
#     thresh_vals = np.arange(0.0, 1.0, 0.01)
#     for thresh_val in tqdm.tqdm(thresh_vals, desc="Calculating optimal threshold value"):
#         abs_diffs = []
#         for image_name in image_names:
#             num_annotations = annotations[image_name]["boxes"].shape[0]
#             num_predictions = (np.where(np.array(predictions["image_predictions"][image_name]["pred_scores"]) >= thresh_val)[0]).size
#             abs_diffs.append(abs(num_annotations - num_predictions))
#         mean_abs_diff = float(np.mean(abs_diffs))

#         if prev_mean_abs_diff < mean_abs_diff:
#             break

#         if mean_abs_diff <= optimal_mean_abs_diff:
#             optimal_mean_abs_diff = mean_abs_diff
#             optimal_thresh_val = thresh_val

#         prev_mean_abs_diff = mean_abs_diff

#     logger.info("Optimal score threshold for all images is: {}".format(optimal_thresh_val))

#     return optimal_thresh_val, optimal_mean_abs_diff



# def similarity_analysis(config, predictions):

    

#     distance_record_path = os.path.join(config["model_dir"], "patch_distances.json")

#     if not os.path.exists(distance_record_path):
#         return

#     distance_record = json_io.load_json(distance_record_path)

#     class_map = config["arch"]["class_map"]
#     reverse_class_map = config["arch"]["reverse_class_map"]


#     # annotated_patch_counts = {k: [] for k in class_map.keys()}
#     # pred_patch_counts = {k: [] for k in class_map.keys()} 

#     res = []
#     for patch_name, patch_pred in predictions["patch_predictions"].items():
#         if "patch_classes" in patch_pred:

#             patch_classes = patch_pred["patch_classes"]
#             unique, counts = np.unique(patch_classes, return_counts=True)
#             class_num_to_count = dict(zip(unique, counts))
#             cur_patch_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
#             for class_num in class_num_to_count.keys():
#                 cur_patch_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
#                 #cur_patch_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

#             pred_patch_classes = patch_pred["pred_classes"]
#             pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
#             class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
#             cur_patch_pred_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}


#             annotated_patch_plant_count = cur_patch_class_counts["plant"]
#             pred_patch_plant_count = cur_patch_pred_class_counts["plant"]
            
#             abs_diff = abs(annotated_patch_plant_count - pred_patch_plant_count)
#             diff = (annotated_patch_plant_count - pred_patch_plant_count)
#             distance = distance_record[patch_name + ".png"]
#             res.append({
#                 "abs_diff": abs_diff,
#                 "diff": diff,
#                 "distance": distance
#             })
#             # for class_name in class_map.keys(): #config.arch["class_map"].keys():
#             #     annotated_patch_counts[class_name].append(cur_patch_class_counts[class_name])
#             #     pred_patch_counts[class_name].append(cur_patch_pred_class_counts[class_name])

#     xs = [d["distance"] for d in res]
#     ys = [d["abs_diff"] for d in res]

#     out_path = os.path.join(config["model_dir"], "distance_versus_abs_count_diff.png")
#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.xlabel("Euclidean Feature Distance")
#     plt.ylabel("Absolute difference in predicted count")
#     plt.savefig(out_path)

#     xs = [d["distance"] for d in res]
#     ys = [d["diff"] for d in res]

#     out_path = os.path.join(config["model_dir"], "distance_versus_count_diff.png")
#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.xlabel("Euclidean Feature Distance")
#     plt.ylabel("Difference in predicted count")
#     plt.savefig(out_path)

def get_positives_and_negatives(annotated_boxes, predicted_boxes, iou_thresh):

    # start_time = time.time()

    num_annotated = annotated_boxes.shape[0]
    num_predicted = predicted_boxes.shape[0]
    # print("num_annotated", num_annotated)
    # print("num_predicted", num_predicted)

    # if num_annotated == 0 or num_predicted == 0:
    #     return 0


    # num_true_positives: rows with at least one True
    # num_false_positives: cols with all False
    # num_false_negatives: rows with all False

    # true_positives = np.full(num_annotated, False)
    # false_positives = np.full(num_predicted, True)

    annotated_boxes = box_utils.swap_xy_np(annotated_boxes)
    predicted_boxes = box_utils.swap_xy_np(predicted_boxes)



    CHUNK_SIZE = 4000 #8000
    # CHUNK_SIZE = 1000000
    matches = np.full(num_predicted, -1)
    # max_iou_vals = np.full(num_predicted, 0)
    # for row_ind in tqdm.trange(0, num_predicted, CHUNK_SIZE):
    #     for col_ind in range(0, num_predicted, CHUNK_SIZE):

    #         iou_mat = box_utils.compute_iou(
    #                 tf.convert_to_tensor(annotated_boxes[row_ind:min(row_ind+CHUNK_SIZE, num_annotated), :], dtype=tf.float64), 
    #                 tf.convert_to_tensor(predicted_boxes[col_ind:min(col_ind+CHUNK_SIZE, num_predicted), :], dtype=tf.float64),
    #         box_format="corners_xy").numpy()
    #         # print("iou_mat.shape", iou_mat.shape)
    #         # print(iou_mat)

    #         if np.any(iou_mat):

    #             max_chunk_inds = np.argmax(iou_mat, axis=0)
    #             # print("max_chunk_inds", max_chunk_inds)
    #             # print("max_chunk_inds.shape", max_chunk_inds.shape)
    #             max_chunk_vals = np.take_along_axis(iou_mat, np.expand_dims(max_chunk_inds, axis=0), axis=0)[0]
    #             # num_predicted_in_chunk = min(col_ind+CHUNK_SIZE, num_predicted) - col_ind
    #             # thresh_vals = np.full(num_predicted_in_chunk, iou_thresh)
    #             mask = np.logical_and(max_chunk_vals >= iou_thresh, 
    #                                 max_chunk_vals > max_iou_vals[col_ind:min(col_ind+CHUNK_SIZE, num_predicted)]
    #             )
    #             # print("max_chunk_inds.size: {}, np.unique(max_chunk_inds).size: {}".format(max_chunk_inds.size, np.unique(max_chunk_inds.size)))
    #             # assert max_chunk_inds.size == np.unique(max_chunk_inds).size
    #             matches[col_ind:min(col_ind+CHUNK_SIZE, num_predicted)][mask] = max_chunk_inds[mask] + row_ind
    #             max_iou_vals[col_ind:min(col_ind+CHUNK_SIZE, num_predicted)][mask] = max_chunk_vals[mask]

    MAX_MAT_SIZE = 16000000
    STEP_SIZE = min(num_predicted, m.floor(MAX_MAT_SIZE / num_predicted))
    # print("STEP_SIZE", STEP_SIZE)
    # STEP_SIZE = 10
    for i in range(0, num_predicted, STEP_SIZE): #, CHUNK_SIZE):
        # for col_ind in range(0, num_predicted, CHUNK_SIZE):
        iou_mat = box_utils.compute_iou(
                    # tf.convert_to_tensor(predicted_boxes[i:i+1, :], dtype=tf.float64), 
                    # tf.convert_to_tensor(annotated_boxes, dtype=tf.float64),

                    tf.convert_to_tensor(annotated_boxes, dtype=tf.float64),
                    tf.convert_to_tensor(predicted_boxes[i:i+STEP_SIZE, :], dtype=tf.float64), 
                    box_format="corners_xy").numpy()


        # iou_mat = box_utils.compute_iou_np(
        #             annotated_boxes,
        #             predicted_boxes[i:i+STEP_SIZE, :]) #box_format="corners_xy")
        # max_ind = np.argmax(iou_mat)
        # if iou_mat[0][max_ind] >= iou_thresh:
        # if iou_mat[max_ind][0] >= iou_thresh:    
        #     matches[i] = max_ind

        max_inds = np.argmax(iou_mat, axis=0)
        max_vals = np.take_along_axis(iou_mat, np.expand_dims(max_inds, axis=0), axis=0)[0]
        mask = max_vals >= iou_thresh
        matches[i:i+STEP_SIZE][mask] = max_inds[mask]





    matched_elements = matches[matches > -1]
    true_positive = np.sum(np.unique(matches) != -1)
    false_positive = np.sum(matches == -1) + (len(matched_elements) - len(np.unique(matched_elements)))
    false_negative = num_annotated - true_positive

    # unq, unq_counts = np.unique(matches, return_counts=True)
    # print("unique counts", unq_counts)

    # print("number of matched_elements", matched_elements.size)
    # print("number of -1s", np.sum(matches == -1))
    # print("number of true positives", true_positive)
    # print("number of false positives", false_positive)
    # print("number of false negatives", false_negative)

    # end_time = time.time()
    # elapsed = end_time - start_time
    # print("positives_and_negatives took {} seconds.".format(elapsed))

    # acc = true_positive / (true_positive + false_positive + false_negative)
    return int(true_positive), int(false_positive), int(false_negative)

    # for row_ind in tqdm.trange(0, num_annotated, CHUNK_SIZE):
    #     for col_ind in range(0, num_predicted, CHUNK_SIZE):

    #         # start_time = time.time()

    #         iou_mat = box_utils.compute_iou(
    #                     tf.convert_to_tensor(annotated_boxes[row_ind:min(num_annotated, row_ind+CHUNK_SIZE), :], dtype=tf.float64), 
    #                     tf.convert_to_tensor(predicted_boxes[col_ind:min(num_predicted, col_ind+CHUNK_SIZE), :], dtype=tf.float64),
    #                     box_format="corners_yx")

    #         # iou_mat2 = box_utils.compute_iou(
    #         #             tf.convert_to_tensor(annotated_boxes[row_ind:min(num_annotated, row_ind+CHUNK_SIZE), :], dtype=tf.float64), 
    #         #             tf.convert_to_tensor(predicted_boxes[col_ind:min(num_predicted, col_ind+CHUNK_SIZE), :], dtype=tf.float64),
    #         #             box_format="corners_xy")

    #         # assert np.array_equal(iou_mat, iou_mat2)
    #         # iou_mat = box_utils.compute_iou_np(annotated_boxes[row_ind:min(num_annotated, row_ind+CHUNK_SIZE), :], 
    #         #                                    predicted_boxes[col_ind:min(num_predicted, col_ind+CHUNK_SIZE), :])
    #         # print(iou_mat)
            
    #         # end_time = time.time()
    #         # elapsed = end_time - start_time
    #         # print("took {} seconds to compute iou matrix.".format(elapsed))
    #         # print(iou_mat.shape, row_ind, col_ind)
    #         overlap_mat = iou_mat >= iou_thresh
    #         # print(overlap_mat)

    #         # start_time = time.time()
    #         # true_positives_chunk = true_positives[row_ind:min(num_annotated, row_ind+CHUNK_SIZE)]
    #         true_positives[row_ind:min(num_annotated, row_ind+CHUNK_SIZE)] = \
    #             np.logical_or(true_positives[row_ind:min(num_annotated, row_ind+CHUNK_SIZE)], 
    #                           np.any(overlap_mat, axis=1))

    #         # print("true_positives_chunk", true_positives_chunk)

    #         # end_time = time.time()
    #         # elapsed = end_time - start_time
    #         # print("took {} seconds to evaluate true positives chunk.".format(elapsed))                               
    #         # print("row_ind: {}, col_ind: {}".format(row_ind, col_ind))
    #         # print("false_positives.shape", false_positives.shape)
    #         # print("false_positives_sub.shape", false_positives[col_ind:min(num_predicted, col_ind+CHUNK_SIZE)].shape)
    #         # false_positives_chunk = false_positives[col_ind:min(num_predicted, col_ind+CHUNK_SIZE)]
    #         false_positives[col_ind:min(num_predicted, col_ind+CHUNK_SIZE)] = \
    #             np.logical_and(false_positives[col_ind:min(num_predicted, col_ind+CHUNK_SIZE)],
    #                            np.all(np.logical_not(overlap_mat), axis=0))

    # # print("true_positives", true_positives)
    # num_true_positives = np.sum(true_positives)
    # num_false_positives = np.sum(false_positives)
    # num_false_negatives = true_positives.size - num_true_positives



    # # iou_mat = box_utils.compute_iou_np(annotated_boxes, predicted_boxes)
    # # overlap_mat = iou_mat >= iou_thresh
    # # num_true_positives = np.any(overlap_mat, axis=1).sum()
    # # num_false_positives = np.all(np.logical_not(overlap_mat), axis=0).sum()
    # # num_false_negatives = np.all(np.logical_not(overlap_mat), axis=1).sum()

    # # print("num_true_positives", num_true_positives)
    # # print("num_false_positives", num_false_positives)
    # # print("num_false_negatives", num_false_negatives)

    # if num_true_positives == 0 and num_false_positives == 0:
    #     return 0

    # if num_true_positives == 0 and num_false_negatives == 0:
    #     return 0
    # precision = num_true_positives / (num_true_positives + num_false_positives)
    # recall = num_true_positives / (num_true_positives + num_false_negatives)


    # if precision == 0 and recall == 0:
    #     return 0

    # f1_score = (2 * precision * recall) / (precision + recall)

    # return float(f1_score)





def collect_image_set_metrics(predictions, annotations, metadata):


    logger = logging.getLogger(__name__)
    logger.info("Started collecting image set metrics.")


    metrics = {}
    metric_keys = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)"
    ]


    for obj_ind, obj_class in enumerate(metadata["object_classes"] + ["All Classes"]):
        metrics[obj_class] = {}

        for metric_key in metric_keys:
            metrics[obj_class][metric_key] = {}

            for image_name in predictions.keys():
                metrics[obj_class][metric_key][image_name] = {}

                for region_key in ["regions_of_interest", "training_regions", "test_regions"]:
                    metrics[obj_class][metric_key][image_name][region_key] = []


    for obj_ind, obj_class in enumerate(metadata["object_classes"] + ["All Classes"]):
        for image_name in predictions.keys():

            anno_boxes = annotations[image_name]["boxes"]
            anno_classes = annotations[image_name]["classes"]
            pred_boxes = np.array(predictions[image_name]["boxes"])
            pred_scores = np.array(predictions[image_name]["scores"])
            pred_classes = np.array(predictions[image_name]["classes"])

            if obj_class != "All Classes":
                anno_class_mask = anno_classes == obj_ind
                anno_boxes = anno_boxes[anno_class_mask]

                pred_class_mask = pred_classes == obj_ind
                pred_boxes = pred_boxes[pred_class_mask]
                pred_scores = pred_scores[pred_class_mask]

            for region_key in ["regions_of_interest", "training_regions", "test_regions"]:
            

                for region in annotations[image_name][region_key]:
                    annotated_centres = (anno_boxes[..., :2] + anno_boxes[..., 2:]) / 2.0
                    predicted_centres = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0
                        
                    if region_key == "regions_of_interest":
                        annotated_inds = poly_utils.get_contained_inds_for_points(annotated_centres, [region])
                        predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])
                    else:
                        annotated_inds = box_utils.get_contained_inds_for_points(annotated_centres, [region])
                        predicted_inds = box_utils.get_contained_inds_for_points(predicted_centres, [region])

                    region_anno_boxes = anno_boxes[annotated_inds]
                    region_pred_boxes = pred_boxes[predicted_inds]
                    region_pred_scores = pred_scores[predicted_inds]


                    sel_region_pred_boxes = region_pred_boxes[region_pred_scores > 0.50]
                    
                    num_predicted = sel_region_pred_boxes.shape[0]
                    num_annotated = region_anno_boxes.shape[0]
                    # print("num_predicted: {}, num_annotated: {}".format(num_predicted, num_annotated))

                    if num_predicted > 0:
                        if num_annotated > 0:
                            true_positive, false_positive, false_negative = get_positives_and_negatives(region_anno_boxes, sel_region_pred_boxes, 0.50)
                            print("anno: {}, pred: {}, tp: {}, fp: {}, fn: {}".format(
                                num_predicted, num_annotated, true_positive, false_positive, false_negative))
                            precision_050 = true_positive / (true_positive + false_positive)
                            recall_050 = true_positive / (true_positive + false_negative)
                            if precision_050 == 0 and recall_050 == 0:
                                f1_iou_050 = 0
                            else:
                                f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
                            acc_050 = true_positive / (true_positive + false_positive + false_negative)     
                        
                        else:
                            true_positive = 0
                            false_positive = num_predicted
                            false_negative = 0

                            precision_050 = 0.0
                            recall_050 = 0.0
                            f1_iou_050 = 0.0
                            acc_050 = 0.0
                    else:
                        if num_annotated > 0:
                            true_positive = 0
                            false_positive = 0
                            false_negative = num_annotated

                            precision_050 = 0.0
                            recall_050 = 0.0
                            f1_iou_050 = 0.0
                            acc_050 = 0.0
                        else:
                            true_positive = 0
                            false_positive = 0
                            false_negative = 0

                            precision_050 = 1.0
                            recall_050 = 1.0
                            f1_iou_050 = 1.0
                            acc_050 = 1.0

                    metrics[obj_class]["True Positives (IoU=.50, conf>.50)"][image_name][region_key].append(true_positive)
                    metrics[obj_class]["False Positives (IoU=.50, conf>.50)"][image_name][region_key].append(false_positive)
                    metrics[obj_class]["False Negatives (IoU=.50, conf>.50)"][image_name][region_key].append(false_negative)
                    metrics[obj_class]["Precision (IoU=.50, conf>.50)"][image_name][region_key].append(precision_050)
                    metrics[obj_class]["Recall (IoU=.50, conf>.50)"][image_name][region_key].append(recall_050)
                    metrics[obj_class]["Accuracy (IoU=.50, conf>.50)"][image_name][region_key].append(acc_050)
                    metrics[obj_class]["F1 Score (IoU=.50, conf>.50)"][image_name][region_key].append(f1_iou_050)


    return metrics




def collect_image_set_metrics_old(predictions, annotations, metadata):


    logger = logging.getLogger(__name__)
    logger.info("Started collecting image set metrics.")

    metrics_start_time = time.time()

    # num_classes = len(config["arch"]["class_map"].keys())

    # image_metrics = {}

    # metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    # metadata = json_io.load_json(metadata_path)
    # is_ortho = metadata["is_ortho"] == "yes"

    # if is_ortho:
    metrics = {}
    metric_keys = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)",
        # "AP (IoU=.50:.05:.95)",
        # "AP (IoU=.50)",
        # "AP (IoU=.75)"
    ]
    for metric_key in metric_keys:
        metrics[metric_key] = {}
    # else:
    #     metrics = {
    #         "AP (IoU=.50:.05:.95)": {},
    #         "AP (IoU=.50)": {},
    #         "AP (IoU=.75)": {},
    #         "F1 Score (IoU=.50, conf>=.50)" : {},
    #         "F1 Score (IoU=.75, conf>=.50)" : {}
    #     }


    # num_regions = 0
    # for image_name in annotations.keys():
    #     for metric_key in metric_keys:

    for image_name in tqdm.tqdm(predictions.keys(), desc="Calculating Metrics"):


        # print("collect_image_set_metrics", image_name)
        for metric_key in metric_keys:
            metrics[metric_key][image_name] = {}

        # metrics["F1 Score (IoU=.75, conf>.50)"][image_name] = {}


        annotated_boxes = annotations[image_name]["boxes"]
        pred_boxes = np.array(predictions[image_name]["boxes"])
        pred_scores = np.array(predictions[image_name]["scores"])


        for region_key in ["regions_of_interest", "training_regions", "test_regions"]:
            # for metric_key in metric_keys:
            #     metrics[metric_key][image_name][region_key] = {}[]

            for region in annotations[image_name][region_key]:
                annotated_centres = (annotated_boxes[..., :2] + annotated_boxes[..., 2:]) / 2.0
                predicted_centres = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0
                    
                if region_key == "regions_of_interest":
                    annotated_inds = poly_utils.get_contained_inds_for_points(annotated_centres, [region])
                    predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])
                else:
                    annotated_inds = box_utils.get_contained_inds_for_points(annotated_centres, [region])
                    predicted_inds = box_utils.get_contained_inds_for_points(predicted_centres, [region])

                region_annotated_boxes = annotated_boxes[annotated_inds]
                region_annotated_classes = annotated_classes[annotated_inds]
                region_pred_boxes = pred_boxes[predicted_inds]
                region_pred_scores = pred_scores[predicted_inds]
                region_pred_classes = pred_classes[predicted_inds]

                for obj_class_ind, obj_class in enumerate(metadata["object_classes"] + "All Classes"):

                    if obj_class == "All Classes":
                        cur_annotated_boxes = region_annotated_boxes
                        cur_pred_boxes = region_pred_boxes
                        cur_pred_scores = region_pred_scores
                    else:
                        annotated_class_mask = region_annotated_classes == obj_class_ind
                        cur_annotated_boxes = region_annotated_boxes[annotated_class_mask]

                        pred_class_mask = region_pred_classes == obj_class_ind
                        cur_pred_boxes = region_pred_boxes[pred_class_mask]
                        cur_pred_scores = region_pred_scores[pred_class_mask]




                # if fully_annotated:
                #     region_annotated_boxes = annotated_boxes
                #     region_pred_boxes = pred_boxes
                #     region_pred_scores = pred_scores
                # else:
                #     # region_annotated_inds = box_utils.get_contained_inds(annotated_boxes, [region])
                #     # region_annotated_boxes = annotated_boxes[region_annotated_inds]
                #     # # region_annotated_classes = np.zeros(shape=(region_annotated_boxes.shape[0]))
                    

                #     # region_pred_inds = box_utils.get_contained_inds(pred_boxes, [region])
                #     # region_pred_boxes = pred_boxes[region_pred_inds]
                #     # region_pred_scores = pred_scores[region_pred_inds]


                #     # annotated_inds = box_utils.get_contained_inds(annotated_boxes, [region])
                #     annotated_centres = (annotated_boxes[..., :2] + annotated_boxes[..., 2:]) / 2.0
                #     annotated_inds = box_utils.get_contained_inds_for_points(annotated_centres, [region])
                #     region_annotated_boxes = annotated_boxes[annotated_inds]

                #     # predicted_inds = box_utils.get_contained_inds(predicted_boxes, [region])
                #     predicted_centres = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0
                #     predicted_inds = box_utils.get_contained_inds_for_points(predicted_centres, [region])
                #     region_pred_boxes = pred_boxes[predicted_inds]

                #     region_pred_scores = pred_scores[predicted_inds]

                    
                
                # sel_region_pred_scores = region_pred_scores[region_pred_scores >= 0.50]
                
                # region_pred_classes = np.zeros(shape=(region_pred_boxes.shape[0]))

                # print("getting AP vals")
                # start_time = time.time()
                # AP_vals = get_AP_vals(region_annotated_boxes, region_pred_boxes, region_pred_scores)
                # end_time = time.time()
                # elapsed = end_time - start_time
                # print("Calculated AP vals in {} seconds.".format(elapsed))

                # metrics["AP (IoU=.50:.05:.95)"][image_name][region_key].append(AP_vals["AP (IoU=.50:.05:.95)"])
                # metrics["AP (IoU=.50)"][image_name][region_key].append(AP_vals["AP (IoU=.50)"])
                # metrics["AP (IoU=.75)"][image_name][region_key].append(AP_vals["AP (IoU=.75)"])

                # pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
                #     region_pred_boxes, region_pred_classes, region_pred_scores,
                #     region_annotated_boxes, region_annotated_classes)

                # image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
                # image_metric_fn.add(pred_for_mAP, true_for_mAP)
                # # pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
                # coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


                # image_metrics[image_name]["Image PASCAL VOC mAP"] = float(pascal_voc_mAP) * 100
                # metrics["MS COCO mAP"][image_name][region_key].append(MS_COCO_mAP)

                # print("getting F1 scores")
                sel_region_pred_boxes = region_pred_boxes[region_pred_scores > 0.50]
                
                num_predicted = sel_region_pred_boxes.shape[0]
                num_annotated = region_annotated_boxes.shape[0]
                # print("num_predicted: {}, num_annotated: {}".format(num_predicted, num_annotated))

                if num_predicted > 0:
                    if num_annotated > 0:
                        true_positive, false_positive, false_negative = get_positives_and_negatives(region_annotated_boxes, sel_region_pred_boxes, 0.50)
                        print("anno: {}, pred: {}, tp: {}, fp: {}, fn: {}".format(
                            num_predicted, num_annotated, true_positive, false_positive, false_negative))
                        precision_050 = true_positive / (true_positive + false_positive)
                        recall_050 = true_positive / (true_positive + false_negative)
                        if precision_050 == 0 and recall_050 == 0:
                            f1_iou_050 = 0
                        else:
                            f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
                        acc_050 = true_positive / (true_positive + false_positive + false_negative)
                        # true_positive, false_positive, false_negative = get_positives_and_negatives(annotated_boxes, sel_region_pred_boxes, 0.75)
                        # precision = true_positive / (true_positive + false_positive)
                        # recall = true_positive / (true_positive + false_negative)
                        # f1_iou_075 = (2 * precision * recall) / (precision + recall)                        

                        
                    
                    else:
                        true_positive = 0
                        false_positive = num_predicted
                        false_negative = 0

                        precision_050 = 0.0
                        recall_050 = 0.0
                        f1_iou_050 = 0.0
                        acc_050 = 0.0
                else:
                    if num_annotated > 0:
                        true_positive = 0
                        false_positive = 0
                        false_negative = num_annotated

                        precision_050 = 0.0
                        recall_050 = 0.0
                        f1_iou_050 = 0.0
                        acc_050 = 0.0
                    else:
                        true_positive = 0
                        false_positive = 0
                        false_negative = 0

                        precision_050 = 1.0
                        recall_050 = 1.0
                        f1_iou_050 = 1.0
                        acc_050 = 1.0
                        
                
                
                # f1_iou_050 = get_f1_score(region_annotated_boxes, sel_region_pred_boxes, iou_thresh=0.50)
                # f1_iou_075 = get_f1_score(region_annotated_boxes, sel_region_pred_boxes, iou_thresh=0.75)
                # f1_iou_09 = get_f1_score(region_annotated_boxes, sel_region_pred_boxes, iou_thresh=0.9)
                
                # metrics["F1 Score (IoU=.75, conf>.50)"][image_name][region_key].append(f1_iou_075)
                metrics["True Positives (IoU=.50, conf>.50)"][image_name][region_key].append(true_positive)
                metrics["False Positives (IoU=.50, conf>.50)"][image_name][region_key].append(false_positive)
                metrics["False Negatives (IoU=.50, conf>.50)"][image_name][region_key].append(false_negative)
                metrics["Precision (IoU=.50, conf>.50)"][image_name][region_key].append(precision_050)
                metrics["Recall (IoU=.50, conf>.50)"][image_name][region_key].append(recall_050)
                metrics["Accuracy (IoU=.50, conf>.50)"][image_name][region_key].append(acc_050)
                metrics["F1 Score (IoU=.50, conf>.50)"][image_name][region_key].append(f1_iou_050)
    metrics_end_time = time.time()
    metrics_elapsed_time = metrics_end_time - metrics_start_time
    logger.info("Finished calculating metrics. Took {} seconds.".format(metrics_elapsed_time))

    return metrics


    # for image_name in tqdm.tqdm(image_predictions.keys(), desc="Collecting metrics"):

    #     image_abs_boxes = annotations[image_name]["boxes"]
    #     image_classes = annotations[image_name]["classes"]
    #     image_status = annotations[image_name]["status"]
        
    #     if image_status == "completed_for_training" or image_status == "completed_for_testing":

    #         image_metrics[image_name] = {}

    #         pred_abs_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
    #         pred_classes = np.array(image_predictions[image_name]["pred_classes"])
    #         pred_scores = np.array(image_predictions[image_name]["pred_scores"])

    #         pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
    #                                                                image_abs_boxes, image_classes)

    #         image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
    #         image_metric_fn.add(pred_for_mAP, true_for_mAP)
    #         # pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
    #         coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


    #         # image_metrics[image_name]["Image PASCAL VOC mAP"] = float(pascal_voc_mAP) * 100
    #         image_metrics[image_name]["MS COCO mAP"] = float(coco_mAP) * 100



    # return image_metrics


def get_AP_vals(annotated_boxes, predicted_boxes, predicted_scores):
    

    print("AP: num_annotated_boxes: {}, num_predicted_boxes: {}".format(
        annotated_boxes.shape[0], predicted_boxes.shape[0]))

    NUM_BOXES_THRESH = 10000

    if (annotated_boxes.shape[0] * predicted_boxes.shape[0]) < (NUM_BOXES_THRESH * NUM_BOXES_THRESH):

        annotated_classes = np.zeros(shape=(annotated_boxes.shape[0]))
        predicted_classes = np.zeros(shape=(predicted_boxes.shape[0]))

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
            predicted_boxes, 
            predicted_classes, 
            predicted_scores,
            annotated_boxes,
            annotated_classes)

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        metric_fn.add(pred_for_mAP, true_for_mAP)

        ms_coco_mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        mAP_IoU_50 = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        mAP_IoU_75 = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        return {
            "AP (IoU=.50:.05:.95)": float(ms_coco_mAP) * 100,
            "AP (IoU=.50)": float(mAP_IoU_50) * 100,
            "AP (IoU=.75)": float(mAP_IoU_75) * 100
        }
    
    else:
        return {
            "AP (IoU=.50:.05:.95)": "unable_to_calculate",
            "AP (IoU=.50)": "unable_to_calculate",
            "AP (IoU=.75)": "unable_to_calculate"
        }
    


        # min_y_annotated = np.min(annotated_boxes[:, 0])
        # min_x_annotated = np.min(annotated_boxes[:, 1])
        # max_y_annotated = np.max(annotated_boxes[:, 2])
        # max_x_annotated = np.max(annotated_boxes[:, 3])

        # min_y_predicted = np.min(predicted_boxes[:, 0])
        # min_x_predicted = np.min(predicted_boxes[:, 1])
        # max_y_predicted = np.max(predicted_boxes[:, 2])
        # max_x_predicted = np.max(predicted_boxes[:, 3])

        # min_y = min(min_y_annotated, min_y_predicted)
        # min_x = min(min_x_annotated, min_x_predicted)
        # max_y = max(max_y_annotated, max_y_predicted)
        # max_x = max(max_x_annotated, max_x_predicted)

        # estimate_change = np.inf
        # delta = 0.01
        # ms_coco_mAP_vals = []
        # mAP_IoU_50_vals = []
        # mAP_IoU_75_vals = []
        # while len(ms_coco_mAP_vals) < 10: # estimate_change > delta:
        #     point_y = random.randrange(min_y, max_y)
        #     point_x = random.randrange(min_x, max_x)

        #     sample_min_y = point_y - 2
        #     sample_min_x = point_x - 2
        #     sample_max_y = point_y + 2
        #     sample_max_x = point_x + 2

        #     prev_sample_region = [sample_min_y, sample_min_x, sample_max_y, sample_max_x]
        #     sample_region = prev_sample_region
        #     contained_annotated = box_utils.get_contained_inds(annotated_boxes, [sample_region])
        #     contained_predicted = box_utils.get_contained_inds(predicted_boxes, [sample_region])
        #     matrix_size = contained_annotated.size * contained_predicted.size
        #     MAX_SAMPLE_MATRIX_SIZE = 1000 * 1000
        #     while matrix_size < MAX_SAMPLE_MATRIX_SIZE:
        #         cur_h = sample_max_y - sample_min_y
        #         sample_min_y = round(sample_min_y - (cur_h / 2))
        #         sample_max_y = round(sample_max_y + (cur_h / 2))

        #         cur_w = sample_max_x - sample_min_x
        #         sample_min_x = round(sample_min_x - (cur_w / 2))
        #         sample_max_x = round(sample_max_x + (cur_w / 2))

        #         prev_sample_region = sample_region
        #         sample_region = [sample_min_y, sample_min_x, sample_max_y, sample_max_x]
        #         print("sample_region", sample_region)
        #         contained_annotated = box_utils.get_contained_inds(annotated_boxes, [sample_region])
        #         contained_predicted = box_utils.get_contained_inds(predicted_boxes, [sample_region])
        #         matrix_size = contained_annotated.size * contained_predicted.size
        #         print("matrix_size", matrix_size)

        #     contained_annotated = box_utils.get_contained_inds(annotated_boxes, [prev_sample_region])
        #     contained_predicted = box_utils.get_contained_inds(predicted_boxes, [prev_sample_region])

        #     print("Calculating sample mAP with {} annotated boxes and {} predicted boxes".format(contained_annotated.size, contained_predicted.size))

        #     sample_annotated_boxes = annotated_boxes[contained_annotated]
        #     sample_predicted_boxes = predicted_boxes[contained_predicted]
        #     sample_predicted_scores = predicted_scores[contained_predicted]

        #     sample_annotated_classes = np.zeros(shape=(sample_annotated_boxes.shape[0]))
        #     sample_predicted_classes = np.zeros(shape=(sample_predicted_boxes.shape[0]))

        #     pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
        #         sample_predicted_boxes, 
        #         sample_predicted_classes, 
        #         sample_predicted_scores,
        #         sample_annotated_boxes,
        #         sample_annotated_classes)
        #     metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        #     metric_fn.add(pred_for_mAP, true_for_mAP)

        #     ms_coco_mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        #     mAP_IoU_50 = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        #     mAP_IoU_75 = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


        #     ms_coco_mAP_vals.append(ms_coco_mAP)
        #     mAP_IoU_50_vals.append(mAP_IoU_50)
        #     mAP_IoU_75_vals.append(mAP_IoU_75)

        #     print("ms_coco_mAP_vals", ms_coco_mAP_vals)
        #     print("mAP_IoU_50_vals", mAP_IoU_50_vals)
        #     print("mAP_IoU_75_vals", mAP_IoU_75_vals)



        # return {
        #     "AP (IoU=.50:.05:.95)": float(np.mean(ms_coco_mAP_vals)) * 100,
        #     "AP (IoU=.50)": float(np.mean(mAP_IoU_50_vals)) * 100,
        #     "AP (IoU=.75)": float(np.mean(mAP_IoU_75_vals)) * 100
        # }

def can_calculate_density(metadata, camera_specs):

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]

    # if metadata["is_ortho"] == "yes":
    #     return False

    # if (metadata["missing"]["latitude"] or metadata["missing"]["longitude"]) or metadata["camera_height"] == "":
    #     return False
    if metadata["camera_height"] == "":
        return False

    if make not in camera_specs:
        return False
    
    if model not in camera_specs[make]:
        return False

    return True

def calculate_area_m2(gsd, area_px):


    # area_m2 = (metadata["images"][image_name]["height_px"] * gsd) * (metadata["images"][image_name]["width_px"] * gsd)
    area_m2 = area_px * (gsd ** 2) #(area_height_px * gsd) * (area_width_px * gsd)

    return area_m2

def get_gsd(camera_specs, metadata):

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]
    camera_entry = camera_specs[make][model]

    gsd_h = (metadata["camera_height"] * camera_entry["sensor_height"]) / \
            (camera_entry["focal_length"] * camera_entry["image_height_px"]) #metadata["images"][image_name]["height_px"])

    gsd_w = (metadata["camera_height"] * camera_entry["sensor_width"]) / \
            (camera_entry["focal_length"] * camera_entry["image_width_px"]) # metadata["images"][image_name]["width_px"])

    gsd = min(gsd_h, gsd_w)

    return gsd

def create_spreadsheet(results_dir, regions_only=False): #username, farm_name, field_name, mission_date, result_uuid, download_uuid): #, annotation_version):

    path_pieces = results_dir.split("/")
    username = path_pieces[2]
    farm_name = path_pieces[4]
    field_name = path_pieces[5]
    mission_date = path_pieces[6]
    image_set_dir = os.path.join(*path_pieces[:len(path_pieces)-3])

    # image_set_dir = os.path.join("usr", "data", username, "image_sets",
    #                              farm_name, field_name, mission_date)

    # results_dir = os.path.join(image_set_dir, "model", "results", result_uuid)

    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)
    






    predictions_path = os.path.join(results_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path) #w3c_io.load_predictions(predictions_path, {"plant": 0})

    full_predictions_path = os.path.join(results_dir, "full_predictions.json")
    full_predictions = annotation_utils.load_predictions(full_predictions_path)


    # if annotation_version == "preserved":
    annotations_path = os.path.join(results_dir, "annotations.json")
    # excess_green_record_path = os.path.join(results_dir, "excess_green_record.json")
    vegetation_record_path = os.path.join(results_dir, "vegetation_record.json")
    metrics_path = os.path.join(results_dir, "metrics.json")
    metrics = json_io.load_json(metrics_path)
    
    
    tags_path = os.path.join(results_dir, "tags.json")
    # else:
    #     annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    #     # excess_green_record_path = os.path.join(image_set_dir, "excess_green", "record.json")
    #     vegetation_record_path = os.path.join(image_set_dir, "excess_green", "vegetation_record.json")


    annotations = annotation_utils.load_annotations(annotations_path) #w3c_io.load_annotations(annotations_path, {"plant": 0})
    # excess_green_record = json_io.load_json(excess_green_record_path)
    if os.path.exists(vegetation_record_path):
        vegetation_record = json_io.load_json(vegetation_record_path)
    else:
        vegetation_record = None
    # if os.path.exists(excess_green_record_path):
    #     excess_green_record = json_io.load_json(excess_green_record_path)
    # else:
    #     excess_green_record = None

    tags = json_io.load_json(tags_path)



    args = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
        "predictions": predictions,
        "full_predictions": full_predictions,
        "annotations": annotations,
        "metadata": metadata,
        "camera_specs": camera_specs,
        # "excess_green_record": excess_green_record,
        "vegetation_record": vegetation_record,
        "tags": tags
    }
    # if annotation_version == "preserved":
    updated_metrics = metrics
    # else:
    #     updated_metrics = collect_image_set_metrics(full_predictions, annotations)

    if not regions_only:
        images_df = create_images_sheet(args, updated_metrics)
    regions_df = create_regions_sheet(args, updated_metrics)
    stats_df = create_stats_sheet(args, regions_df)

    pandas.io.formats.excel.ExcelFormatter.header_style = None

    # out_dir = os.path.join(results_dir, "retrieval", download_uuid)
    # os.makedirs(out_dir)

    out_path = os.path.join(results_dir, "metrics.xlsx") #os.path.join(out_dir, "results.xlsx")
    # with pd.ExcelWriter(out_path) as writer:
    #     images_df.to_excel(writer, sheet_name="Images")

    sheet_name_to_df = {}

    if not regions_only:
        sheet_name_to_df["Images"] = images_df
    sheet_name_to_df["Regions"] = regions_df
    sheet_name_to_df["Stats"] = stats_df

    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    fmt = writer.book.add_format({"font_name": "Courier New"})

    for sheet_name in sheet_name_to_df.keys():

        df = sheet_name_to_df[sheet_name]

        df.to_excel(writer, index=False, sheet_name=sheet_name, na_rep='NA')  # send df to writer
        worksheet = writer.sheets[sheet_name]  # pull worksheet object

        worksheet.set_column('A:ZZ', None, fmt)
        worksheet.set_row(0, None, fmt)

        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            if series.size > 0:
                max_entry_size = series.astype(str).map(len).max()
            else:
                max_entry_size = 0
            max_len = max((
                max_entry_size,  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width


    # regions_df.to_excel(writer, index=False, sheet_name="Regions", na_rep='NA')  # send df to writer
    # worksheet = writer.sheets["Regions"]  # pull worksheet object

    # worksheet.set_column('A:Z', None, fmt)
    # worksheet.set_row(0, None, fmt)

    # for idx, col in enumerate(regions_df):  # loop through all columns
    #     series = regions_df[col]
    #     if series.size > 0:
    #         max_entry_size = series.astype(str).map(len).max()
    #     else:
    #         max_entry_size = 0
    #     max_len = max((
    #         max_entry_size,  # len of largest item
    #         len(str(series.name))  # len of column name/header
    #         )) + 1  # adding a little extra space
    #     worksheet.set_column(idx, idx, max_len)  # set column width



    writer.save()



    # df = pd.DataFrame(data=d, columns=columns)
    # df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))

    # out_dir = os.path.join(results_dir, "retrieval", download_uuid)
    # os.makedirs(out_dir)

    # out_path = os.path.join(out_dir, "results.csv")
    # df.to_csv(out_path, index=False)
    

    # metrics_out_path = os.path.join(out_dir, "metrics.json")
    # json_io.save_json(metrics_out_path, updated_metrics)





def create_images_sheet(args, updated_metrics):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    predictions = args["predictions"]
    # full_predictions = args["full_predictions"]
    annotations = args["annotations"]
    metadata = args["metadata"]
    camera_specs = args["camera_specs"]
    # excess_green_record = args["excess_green_record"]
    vegetation_record = args["vegetation_record"]


    include_density = can_calculate_density(metadata, camera_specs)

    num_classes = len(metadata["object_classes"])


    # defines the order of the columns
    columns = [
        "Username",
        "Farm Name",
        "Field Name",
        "Mission Date",
        "Image Name",
        "Regions of Interest",
        "Training Regions",
        "Test Regions",
        "Image Is Fully Annotated",
        "Source Of Annotations",
        # "Annotated Count",
        # "Predicted Count"
    ]
    if num_classes > 1:
        columns.extend([
            "Annotated Count (All Classes)",
            "Predicted Count (All Classes)"
        ])

    for object_class in metadata["object_classes"]:
        columns.append("Annotated Count (" + object_class + ")")
        columns.append("Predicted Count (" + object_class + ")")


    if include_density:
        if num_classes > 1:
            columns.extend([
                "Annotated Count Per Square Metre (All Classes)",
                "Predicted Count Per Square Metre (All Classes)"
            ])

        # columns.extend(["Annotated Count Per Square Metre", "Predicted Count Per Square Metre"])
        for object_class in metadata["object_classes"]:
            columns.append("Annotated Count Per Square Metre (" + object_class + ")")
            columns.append("Predicted Count Per Square Metre (" + object_class + ")")




    columns.append("Area (Pixels)")
    if num_classes > 1:
        columns.append("Mean of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Mean of Predicted Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Mean of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Mean of Predicted Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")")

    if include_density:
        columns.extend(["Area (Square Metres)"])
        if num_classes > 1:
            columns.append("Mean of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Mean of Predicted Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (All Classes)")


        for object_class in metadata["object_classes"]:
            columns.append("Mean of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Mean of Predicted Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")")
    # if excess_green_record is not None:
            
    if num_classes > 1:
        columns.append("Percent Count Error (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Percent Count Error (" + object_class + ")")

    if vegetation_record is not None:
        columns.extend([
            "Excess Green Threshold",
            "Vegetation Percentage"
            # "Percentage of Vegetation Belonging to Objects",
            # "Percentage of Vegetation Belonging to Non-Objects"
        ])
        if num_classes > 1:
            columns.extend([
                "Percentage of Vegetation Inside Object Boundaries",
                "Percentage of Vegetation Outside Object Boundaries"           
            ])
        for object_class in metadata["object_classes"]:
            columns.extend([
                "Percentage of Vegetation Inside " + object_class + " Boundaries",
                "Percentage of Vegetation Outside " + object_class + " Boundaries"    
            ])
    # columns.append("MS_COCO_mAP")
        

    metrics_lst = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)"  
    ]

    if num_classes > 1:
        for metric in metrics_lst:
            columns.append(metric + " (All Classes)")


    for object_class in metadata["object_classes"]:
        for metric in metrics_lst:
            columns.append(metric + " (" + object_class + ")")


    d = {}
    for c in columns:
        d[c] = []


    # new_metrics = {}

    for image_name in predictions.keys():

        # image_abs_boxes = annotations[image_name]["boxes"]
        regions_of_interest = annotations[image_name]["regions_of_interest"]
        training_regions = annotations[image_name]["training_regions"]
        test_regions = annotations[image_name]["test_regions"]

        # if annotations[image_name]["predictions_used_as_annotations"]:
        #     predictions_used_as_annotations = "yes"
        # else:
        #     predictions_used_as_annotations = "no"
        annotations_source = annotations[image_name]["source"]


        image_height_px = metadata["images"][image_name]["height_px"]
        image_width_px = metadata["images"][image_name]["width_px"]

        if annotation_utils.is_fully_annotated_for_training(annotations, image_name, image_width_px, image_height_px):
            fully_annotated = "yes: for fine-tuning"

        elif annotation_utils.is_fully_annotated_for_testing(annotations, image_name, image_width_px, image_height_px):
            fully_annotated = "yes: for testing"
            
        else:
            fully_annotated = "no"
        # image_status = annotations[image_name]["status"]

        # pred_image_abs_boxes = predictions[image_name]["boxes"]
        # pred_image_scores = predictions[image_name]["scores"]
        # sel_pred_image_abs_boxes = pred_image_abs_boxes[pred_image_scores > 0.50]
        # # print(pred_image_scores)

        # annotated_count = image_abs_boxes.shape[0]
        # predicted_count = sel_pred_image_abs_boxes.shape[0] #np.sum(pred_image_scores > 0.50)

        # if fully_annotated == "no":
        #     percent_count_error = "NA"
        # elif annotated_count > 0:
        #     percent_count_error = round(abs((predicted_count - annotated_count) / (annotated_count)) * 100, 2)
        # else:
        #     percent_count_error = "NA" #undefined"


        height_px = metadata["images"][image_name]["height_px"]
        width_px = metadata["images"][image_name]["width_px"]
        area_px = height_px * width_px

        d["Username"].append(username)
        d["Farm Name"].append(farm_name)
        d["Field Name"].append(field_name)
        d["Mission Date"].append(mission_date)
        d["Image Name"].append(image_name)
        d["Regions of Interest"].append(len(regions_of_interest))
        d["Training Regions"].append(len(training_regions))
        d["Test Regions"].append(len(test_regions))
        d["Image Is Fully Annotated"].append(fully_annotated)
        d["Source Of Annotations"].append(annotations_source)
        # if image_status == "unannotated":
        #     d["annotated_plant_count"].append("NA")
        # else:
        #     d["annotated_plant_count"].append(annotated_count)
        d["Area (Pixels)"].append(area_px)
        if include_density:
            gsd = get_gsd(camera_specs, metadata)
            area_m2 = calculate_area_m2(gsd, area_px)
            d["Area (Square Metres)"].append(round(area_m2, 8))

        if num_classes > 1:
            cls_tups = [(-1, "All Classes")]
            for i, object_class in enumerate(metadata["object_classes"]):
                cls_tups.append((i, object_class))
        else:
            cls_tups = [(0, metadata["object_classes"][0])]

        for cls_tup in cls_tups:
            cls_idx = cls_tup[0]
            object_class = cls_tup[1]

            annotated_boxes = []
            predicted_boxes = []
            for i in range(len(annotations[image_name]["boxes"])):
                if cls_idx == -1 or annotations[image_name]["classes"][i] == cls_idx:
                    annotated_boxes.append(annotations[image_name]["boxes"][i])

            for i in range(len(predictions[image_name]["boxes"])):
                if cls_idx == -1 or predictions[image_name]["classes"][i] == cls_idx:
                    if predictions[image_name]["scores"][i] > 0.50:
                        predicted_boxes.append(predictions[image_name]["boxes"][i])

            annotated_boxes = np.array(annotated_boxes)
            predicted_boxes = np.array(predicted_boxes)

            annotated_count = annotated_boxes.shape[0]
            predicted_count = predicted_boxes.shape[0]

            d["Annotated Count (" + object_class + ")"].append(annotated_count)
            d["Predicted Count (" + object_class + ")"].append(predicted_count)


            if annotated_count > 0:
                annotated_box_areas_px = box_utils.box_areas_np(annotated_boxes)
                mean_annotated_object_area_px = round(np.mean(annotated_box_areas_px), 8)
                stdev_annotated_object_area_px = round(np.std(annotated_box_areas_px), 8)
            else:
                mean_annotated_object_area_px = "NA"
                stdev_annotated_object_area_px = "NA"

            if predicted_count > 0:
                predicted_box_areas_px = box_utils.box_areas_np(predicted_boxes)
                mean_predicted_object_area_px = round(np.mean(predicted_box_areas_px), 8)
                stdev_predicted_object_area_px = round(np.std(predicted_box_areas_px), 8)
            else:
                mean_predicted_object_area_px = "NA"
                stdev_predicted_object_area_px = "NA"

            d["Mean of Annotated Object Areas (Pixels) (" + object_class + ")"].append(mean_annotated_object_area_px)
            d["Mean of Predicted Object Areas (Pixels) (" + object_class + ")"].append(mean_predicted_object_area_px)
            d["Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")"].append(stdev_annotated_object_area_px)
            d["Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")"].append(stdev_predicted_object_area_px)


            if include_density:
                gsd = get_gsd(camera_specs, metadata)
                area_m2 = calculate_area_m2(gsd, area_px)
                # if image_status == "unannotated":
                #     d["annotated_plant_count_per_square_metre"].append("NA")
                # else:
                if area_m2 > 0:
                    annotated_count_per_square_metre = round(annotated_count / area_m2, 8)
                    predicted_count_per_square_metre = round(predicted_count / area_m2, 8)
                else:
                    annotated_count_per_square_metre = "NA"
                    predicted_count_per_square_metre = "NA"
                d["Annotated Count Per Square Metre (" + object_class + ")"].append(annotated_count_per_square_metre)
                d["Predicted Count Per Square Metre (" + object_class + ")"].append(predicted_count_per_square_metre)
                # d["Area (Square Metres)"].append(round(area_m2, 8))

                if annotated_count > 0:
                    annotated_box_areas_m2 = calculate_area_m2(gsd, annotated_box_areas_px)
                    mean_annotated_object_area_m2 = round(np.mean(annotated_box_areas_m2), 8)
                    stdev_annotated_object_area_m2 = round(np.std(annotated_box_areas_m2), 8)
                else:
                    mean_annotated_object_area_m2 = "NA"
                    stdev_annotated_object_area_m2 = "NA"

                if predicted_count > 0:
                    predicted_box_areas_m2 = calculate_area_m2(gsd, predicted_box_areas_px)
                    mean_predicted_object_area_m2 = round(np.mean(predicted_box_areas_m2), 8)
                    stdev_predicted_object_area_m2 = round(np.std(predicted_box_areas_m2), 8)               
                else:
                    mean_predicted_object_area_m2 = "NA"
                    stdev_predicted_object_area_m2 = "NA"

                d["Mean of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(mean_annotated_object_area_m2)
                d["Mean of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(mean_predicted_object_area_m2)
                d["Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(stdev_annotated_object_area_m2)
                d["Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(stdev_predicted_object_area_m2)

            if fully_annotated == "no":
                percent_count_error = "NA"
            elif annotated_count > 0:
                percent_count_error = round(abs((predicted_count - annotated_count) / (annotated_count)) * 100, 2)
            else:
                percent_count_error = "NA" #undefined"

            d["Percent Count Error (" + object_class + ")"].append(percent_count_error)


            if fully_annotated == "no":
                for metric in metrics_lst:
                    d[metric + " (" + object_class + ")"].append("NA")
            elif fully_annotated == "yes: for fine-tuning":
                for metric in metrics_lst:
                    metric_val = updated_metrics[object_class][metric][image_name]["training_regions"][0]
                    if isinstance(metric_val, float):
                        metric_val = round(metric_val, 2)
                    d[metric + " (" + object_class + ")"].append(metric_val)
            else:
                region_index = -1
                for i in range(len(annotations[image_name]["test_regions"])):
                    region = annotations[image_name]["test_regions"][i]
                    if (region[0] == 0 and region[1] == 0) and (region[2] == image_height_px and region[3] == image_width_px):
                        region_index = i
                        break

                for metric in metrics_lst:
                    metric_val = updated_metrics[object_class][metric][image_name]["test_regions"][region_index]
                    if isinstance(metric_val, float):
                        metric_val = round(metric_val, 2)
                    d[metric + " (" + object_class + ")"].append(metric_val)


        if vegetation_record is not None:
            d["Excess Green Threshold"].append(vegetation_record[image_name]["sel_val"])
            vegetation_percentage = vegetation_record[image_name]["vegetation_percentage"]["image"]
            # obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"]["image"]
            # if vegetation_percentage == 0:
            #     obj_percentage = "NA"
            #     non_obj_percentage = "NA"
            # else:
            #     obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
            #     non_obj_percentage = round(100 - obj_percentage, 2)
            d["Vegetation Percentage"].append(vegetation_percentage)

            if num_classes > 1:
                obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"]["All Classes"]["image"]
                if vegetation_percentage == 0:
                    obj_percentage = "NA"
                    non_obj_percentage = "NA"
                else:
                    obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                    non_obj_percentage = round(100 - obj_percentage, 2)

                d["Percentage of Vegetation Inside Object Boundaries"].append(obj_percentage)
                d["Percentage of Vegetation Outside Object Boundaries"].append(non_obj_percentage) 

            for object_class in metadata["object_classes"]:

                obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"][object_class]["image"]
                if vegetation_percentage == 0:
                    obj_percentage = "NA"
                    non_obj_percentage = "NA"
                else:
                    obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                    non_obj_percentage = round(100 - obj_percentage, 2)

                d["Percentage of Vegetation Inside " + object_class + " Boundaries"].append(obj_percentage)
                d["Percentage of Vegetation Outside " + object_class + " Boundaries"].append(non_obj_percentage) 



        # d["Annotated Count"].append(annotated_count)
        # d["Predicted Count"].append(predicted_count)


        # height_px = metadata["images"][image_name]["height_px"]
        # width_px = metadata["images"][image_name]["width_px"]
        # area_px = height_px * width_px
        # d["Area (Pixels)"].append(area_px)

        # if annotated_count > 0:
        #     annotated_box_areas_px = box_utils.box_areas_np(image_abs_boxes)
        #     mean_annotated_object_area_px = round(np.mean(annotated_box_areas_px), 8)
        #     stdev_annotated_object_area_px = round(np.std(annotated_box_areas_px), 8)
        # else:
        #     mean_annotated_object_area_px = "NA"
        #     stdev_annotated_object_area_px = "NA"

        # if predicted_count > 0:
        #     predicted_box_areas_px = box_utils.box_areas_np(sel_pred_image_abs_boxes)
        #     mean_predicted_object_area_px = round(np.mean(predicted_box_areas_px), 8)
        #     stdev_predicted_object_area_px = round(np.std(predicted_box_areas_px), 8)
        # else:
        #     mean_predicted_object_area_px = "NA"
        #     stdev_predicted_object_area_px = "NA"

        # d["Mean of Annotated Object Areas (Pixels)"].append(mean_annotated_object_area_px)
        # d["Mean of Predicted Object Areas (Pixels)"].append(mean_predicted_object_area_px)
        # d["Std. Dev. of Annotated Object Areas (Pixels)"].append(stdev_annotated_object_area_px)
        # d["Std. Dev. of Predicted Object Areas (Pixels)"].append(stdev_predicted_object_area_px)


        # if include_density:
        #     gsd = get_gsd(camera_specs, metadata)
        #     area_m2 = calculate_area_m2(gsd, area_px)
        #     # if image_status == "unannotated":
        #     #     d["annotated_plant_count_per_square_metre"].append("NA")
        #     # else:
        #     if area_m2 > 0:
        #         annotated_count_per_square_metre = round(annotated_count / area_m2, 8)
        #         predicted_count_per_square_metre = round(predicted_count / area_m2, 8)
        #     else:
        #         annotated_count_per_square_metre = "NA"
        #         predicted_count_per_square_metre = "NA"
        #     d["Annotated Count Per Square Metre"].append(annotated_count_per_square_metre)
        #     d["Predicted Count Per Square Metre"].append(predicted_count_per_square_metre)
        #     d["Area (Square Metres)"].append(round(area_m2, 8))

        #     if annotated_count > 0:
        #         annotated_box_areas_m2 = calculate_area_m2(gsd, annotated_box_areas_px)
        #         mean_annotated_object_area_m2 = round(np.mean(annotated_box_areas_m2), 8)
        #         stdev_annotated_object_area_m2 = round(np.std(annotated_box_areas_m2), 8)
        #     else:
        #         mean_annotated_object_area_m2 = "NA"
        #         stdev_annotated_object_area_m2 = "NA"

        #     if predicted_count > 0:
        #         predicted_box_areas_m2 = calculate_area_m2(gsd, predicted_box_areas_px)
        #         mean_predicted_object_area_m2 = round(np.mean(predicted_box_areas_m2), 8)
        #         stdev_predicted_object_area_m2 = round(np.std(predicted_box_areas_m2), 8)               
        #     else:
        #         mean_predicted_object_area_m2 = "NA"
        #         stdev_predicted_object_area_m2 = "NA"

        #     d["Mean of Annotated Object Areas (Square Metres)"].append(mean_annotated_object_area_m2)
        #     d["Mean of Predicted Object Areas (Square Metres)"].append(mean_predicted_object_area_m2)
        #     d["Std. Dev. of Annotated Object Areas (Square Metres)"].append(stdev_annotated_object_area_m2)
        #     d["Std. Dev. of Predicted Object Areas (Square Metres)"].append(stdev_predicted_object_area_m2)




        # if excess_green_record is not None:
        # d["Percent Count Error"].append(percent_count_error)
        # if vegetation_record is not None:
        #     d["Excess Green Threshold"].append(vegetation_record[image_name]["sel_val"])
        #     vegetation_percentage = vegetation_record[image_name]["vegetation_percentage"]["image"]
        #     obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"]["image"]
        #     if vegetation_percentage == 0:
        #         obj_percentage = "NA"
        #         non_obj_percentage = "NA"
        #     else:
        #         obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
        #         non_obj_percentage = round(100 - obj_percentage, 2)
        #     d["Vegetation Percentage"].append(vegetation_percentage)
        #     d["Percentage of Vegetation Belonging to Objects"].append(obj_percentage)
        #     d["Percentage of Vegetation Belonging to Non-Objects"].append(non_obj_percentage)



        # if annotation_version == "preserved":
        #     if image_name in metrics:
        #         ms_coco_mAP = round(metrics[image_name]["MS COCO mAP"], 2)

        #         new_metrics[image_name] = {}
        #         new_metrics[image_name]["MS COCO mAP"] = round(metrics[image_name]["MS COCO mAP"], 2)
        #     else:
        #         ms_coco_mAP = "NA"

        # else:
        #     if image_status == "completed_for_training" or image_status == "completed_for_testing":
        #         ms_coco_mAP = round(calculate_MS_COCO_mAP_for_image(predictions, annotations, image_name), 2)


        #         new_metrics[image_name] = {}
        #         new_metrics[image_name]["MS COCO mAP"] = round(calculate_MS_COCO_mAP_for_image(predictions, annotations, image_name), 2)
        #     else:
        #         ms_coco_mAP = "NA"


        # d["MS_COCO_mAP"].append(ms_coco_mAP)
    # json_io.print_json(d)
    # print()
    # for col in columns:
    #     print(col, len(d[col]))

    df = pd.DataFrame(data=d, columns=columns)
    df.sort_values(by="Image Name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["Image Name"])))
    return df


def create_areas_spreadsheet(results_dir, regions_only=False):

    logger = logging.getLogger(__name__)

    path_pieces = results_dir.split("/")
    username = path_pieces[2]
    farm_name = path_pieces[4]
    field_name = path_pieces[5]
    mission_date = path_pieces[6]
    image_set_dir = os.path.join(*path_pieces[:len(path_pieces)-3])


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)
    

    predictions_path = os.path.join(results_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path) #w3c_io.load_predictions(predictions_path, {"plant": 0})

    annotations_path = os.path.join(results_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path) 

    # full_predictions_path = os.path.join(results_dir, "full_predictions.json")
    # full_predictions = annotation_utils.load_predictions(full_predictions_path)




    # metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    # metadata = json_io.load_json(metadata_path)

    if not can_calculate_density(metadata, camera_specs):
        logger.info("Cannot calculate voronoi areas (cannot calculate density).")
        return
    
    logger.info("Started collecting voronoi areas.")


    start_time = time.time()

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]
    camera_entry = camera_specs[make][model]

    image_w = metadata["images"][list(annotations.keys())[0]]["width_px"]
    image_h = metadata["images"][list(annotations.keys())[0]]["height_px"]

    gsd_h = (metadata["camera_height"] * camera_entry["sensor_height"]) / \
            (camera_entry["focal_length"] * image_h)

    gsd_w = (metadata["camera_height"] * camera_entry["sensor_width"]) / \
            (camera_entry["focal_length"] * image_w)

    gsd = min(gsd_h, gsd_w)

    # area_m2 = (metadata["images"][image_name]["height_px"] * gsd) * (metadata["images"][image_name]["width_px"] * gsd)
    
    

    
    object_entries = []
    voronoi_entries = []
    
    for image_name in predictions.keys():
        predicted_boxes = predictions[image_name]["boxes"]
        predicted_scores = predictions[image_name]["scores"]
        
        pred_mask = predicted_scores > 0.50
        sel_predicted_boxes = predicted_boxes[pred_mask]

        if (sel_predicted_boxes.size > 0):
            predicted_box_areas = (sel_predicted_boxes[:, 2] - sel_predicted_boxes[:, 0]) * (sel_predicted_boxes[:, 3] - sel_predicted_boxes[:, 1])
            predicted_box_areas_m2 = np.round(predicted_box_areas * (gsd ** 2), 8)
        else:
            predicted_box_areas_m2 = []

        d_object = {
            image_name: sorted(predicted_box_areas_m2)
        }



        if sel_predicted_boxes.shape[0] <= 3:
            d_voronoi = {
                image_name: []
            }
        else:
            try:
                predicted_centres = (sel_predicted_boxes[..., :2] + sel_predicted_boxes[..., 2:]) / 2.0
                xy_predicted_centres = np.stack([predicted_centres[:, 1], predicted_centres[:, 0]], axis=-1)

                vor = Voronoi(xy_predicted_centres)
                # fig = voronoi_plot_2d(vor)

                lines = [
                    shapely.geometry.LineString(vor.vertices[line])
                    for line in vor.ridge_vertices
                    if -1 not in line
                ]
                
                boundary = shapely.geometry.Polygon([(0, 0), (image_w, 0), (image_w, image_h), (0, image_h)])
                filtered_lines = []
                for line in lines:
                    if boundary.contains(line):
                        filtered_lines.append(line)

                areas_m2 = []
                # polygons = []
                for poly in shapely.ops.polygonize(filtered_lines):
                    # plt.scatter([x[0] for x in poly.exterior.coords], [x[1] for x in poly.exterior.coords], color="green")
                    # polygons.append(Polygon(poly.exterior.coords, fill=True, facecolor="green"))
                    area_px = poly.area
                    area_m2 = round(area_px * (gsd ** 2), 8)
                    areas_m2.append(area_m2)
                # p = PatchCollection(polygons, alpha=0.4)
                # fig.axes[0].add_collection(p)
                # plt.ylim((0, image_h))
                # plt.xlim((0, image_w))
                # plt.savefig(os.path.join(results_dir, "voronoi_plots", image_name + ":" + region_label + "_region_" + str(i) + ".svg"))

                # plt.close()

                d_voronoi = {
                    image_name: sorted(areas_m2)
                }
            except Exception as e:
                logger.info("Voronoi area calculation generated exception: {}".format(e))
                d_voronoi = {
                    image_name: []
                }


        object_entries.append(pd.DataFrame(d_object))
        voronoi_entries.append(pd.DataFrame(d_voronoi))

    if len(object_entries) > 0:
        object_images_df = pd.concat(object_entries, axis=1)
    else:
        object_images_df = pd.DataFrame()
    object_images_df = object_images_df.fillna('')

    if len(voronoi_entries) > 0:
        voronoi_images_df = pd.concat(voronoi_entries, axis=1)
    else:
        voronoi_images_df = pd.DataFrame()
    voronoi_images_df = voronoi_images_df.fillna('')




    object_entries = []
    voronoi_entries = []
    for image_name in predictions.keys():

        predicted_boxes = predictions[image_name]["boxes"]
        predicted_scores = predictions[image_name]["scores"]

        pred_mask = predicted_scores > 0.50
        sel_predicted_boxes = predicted_boxes[pred_mask]

        for region_type in ["regions_of_interest", "training_regions", "test_regions"]:
            if region_type == "regions_of_interest":
                region_label = "interest"
            elif region_type == "training_regions":
                region_label = "fine_tuning"
            else:
                region_label = "test"

            regions = annotations[image_name][region_type]

            for i, region in enumerate(regions):

                entry_name = image_name + ":" + region_label + "_" + str(i+1)

                predicted_centres = (sel_predicted_boxes[..., :2] + sel_predicted_boxes[..., 2:]) / 2.0
                # xy_predicted_centres = np.stack([predicted_centres[:, 1], predicted_centres[:, 0]], axis=-1)


                if region_type == "regions_of_interest":
                    predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])
                else:
                    predicted_inds = box_utils.get_contained_inds_for_points(predicted_centres, [region])
                
                region_predicted_boxes = sel_predicted_boxes[predicted_inds]


                if region_predicted_boxes.size > 0:
                    region_predicted_box_areas_px = (region_predicted_boxes[:, 2] - region_predicted_boxes[:, 0]) * (region_predicted_boxes[:, 3] - region_predicted_boxes[:, 1])
                    region_predicted_box_areas_m2 = np.round(region_predicted_box_areas_px * (gsd ** 2), 8)
                else:
                    region_predicted_box_areas_m2 = []
                d_object = {
                    entry_name: sorted(region_predicted_box_areas_m2)
                }


                if region_predicted_boxes.shape[0] <= 3:
                    d_voronoi = {
                        entry_name: []
                    }
                else:

                    try:

                        region_predicted_centres = (region_predicted_boxes[..., :2] + region_predicted_boxes[..., 2:]) / 2.0
                        xy_region_predicted_centres = np.stack([region_predicted_centres[:, 1], region_predicted_centres[:, 0]], axis=-1)

                        vor = Voronoi(xy_region_predicted_centres)

                        lines = [
                            shapely.geometry.LineString(vor.vertices[line])
                            for line in vor.ridge_vertices
                            if -1 not in line
                        ]

                        if region_type == "regions_of_interest":
                            boundary = shapely.geometry.Polygon([(x[1], x[0]) for x in region])
                        else:
                            boundary = shapely.geometry.Polygon([(region[1], region[0]), (region[3], region[0]), (region[3], region[2]), (region[1], region[2])])
                        
                        filtered_lines = []
                        for line in lines:
                            if boundary.contains(line):
                                filtered_lines.append(line)

                        areas_m2 = []
                        # polygons = []
                        for poly in shapely.ops.polygonize(filtered_lines):
                            # plt.scatter([x[0] for x in poly.exterior.coords], [x[1] for x in poly.exterior.coords], color="green")
                            # polygons.append(Polygon(poly.exterior.coords, fill=True, facecolor="green"))
                            area_px = poly.area
                            area_m2 = round(area_px * (gsd ** 2), 8)
                            areas_m2.append(area_m2)
                        # p = PatchCollection(polygons, alpha=0.4)
                        # fig.axes[0].add_collection(p)
                        # plt.ylim((0, image_h))
                        # plt.xlim((0, image_w))
                        # plt.savefig(os.path.join(results_dir, "voronoi_plots", image_name + ":" + region_label + "_region_" + str(i) + ".svg"))

                        # plt.close()

                        d_voronoi = {
                            entry_name: sorted(areas_m2)
                        }
                    except Exception as e:
                        logger.info("Voronoi area calculation generated exception: {}".format(e))
                        d_voronoi = {
                            entry_name: []
                        }


                object_entries.append(pd.DataFrame(d_object))
                voronoi_entries.append(pd.DataFrame(d_voronoi))



    if len(object_entries) > 0:
        object_regions_df = pd.concat(object_entries, axis=1)
    else:
        object_regions_df = pd.DataFrame()
    object_regions_df = object_regions_df.fillna('')


    if len(voronoi_entries) > 0:
        voronoi_regions_df = pd.concat(voronoi_entries, axis=1)
    else:
        voronoi_regions_df = pd.DataFrame()
    voronoi_regions_df = voronoi_regions_df.fillna('')




    out_path = os.path.join(results_dir, "areas.xlsx")

    if regions_only:
        sheet_name_to_df = {
            "Region Object Areas": object_regions_df,
            "Region Voronoi Areas": voronoi_regions_df
        }
    else:
        sheet_name_to_df = {
            "Image Object Areas": object_images_df,
            "Region Object Areas": object_regions_df,
            "Image Voronoi Areas": voronoi_images_df,
            "Region Voronoi Areas": voronoi_regions_df
            # "Stats": stats_df
        }
    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    fmt = writer.book.add_format({"font_name": "Courier New"})


    for sheet_name in sheet_name_to_df.keys():

        df = sheet_name_to_df[sheet_name]

        df.to_excel(writer, index=False, sheet_name=sheet_name, na_rep='NA')  # send df to writer
        worksheet = writer.sheets[sheet_name]  # pull worksheet object

        worksheet.set_column('A:ZZ', None, fmt)
        worksheet.set_row(0, None, fmt)

        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            if series.size > 0:
                max_entry_size = series.astype(str).map(len).max()
            else:
                max_entry_size = 0
            max_len = max((
                max_entry_size,  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width

    writer.save()

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)

    logger.info("Calculated Voronoi areas in {} seconds.".format(elapsed_time))


    

def create_regions_sheet(args, updated_metrics):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    predictions = args["predictions"]
    annotations = args["annotations"]
    metadata = args["metadata"]
    camera_specs = args["camera_specs"]
    vegetation_record = args["vegetation_record"]
    tags = args["tags"]

    num_classes = len(metadata["object_classes"])

    include_density = can_calculate_density(metadata, camera_specs)


    # defines the order of the columns
    columns = [
        "Username",
        "Farm Name",
        "Field Name",
        "Mission Date",
        "Image Name",
        "Region Name"]
    
    for tag_name in tags.keys():
        columns.append(tag_name)

    columns.extend(["Source Of Annotations (For Image)"])


    if num_classes > 1:
        columns.extend([
            "Annotated Count (All Classes)",
            "Predicted Count (All Classes)"
        ])

    for object_class in metadata["object_classes"]:
        columns.append("Annotated Count (" + object_class + ")")
        columns.append("Predicted Count (" + object_class + ")")


    if include_density:
        if num_classes > 1:
            columns.extend([
                "Annotated Count Per Square Metre (All Classes)",
                "Predicted Count Per Square Metre (All Classes)"
            ])

        # columns.extend(["Annotated Count Per Square Metre", "Predicted Count Per Square Metre"])
        for object_class in metadata["object_classes"]:
            columns.append("Annotated Count Per Square Metre (" + object_class + ")")
            columns.append("Predicted Count Per Square Metre (" + object_class + ")")



    columns.append("Area (Pixels)")
    if num_classes > 1:
        columns.append("Mean of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Mean of Predicted Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Mean of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Mean of Predicted Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")")

    if include_density:
        columns.extend(["Area (Square Metres)"])
        if num_classes > 1:
            columns.append("Mean of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Mean of Predicted Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (All Classes)")


        for object_class in metadata["object_classes"]:
            columns.append("Mean of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Mean of Predicted Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")")
    # if excess_green_record is not None:
            
    if num_classes > 1:
        columns.append("Percent Count Error (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Percent Count Error (" + object_class + ")")

    if vegetation_record is not None:
        columns.extend([
            "Excess Green Threshold",
            "Vegetation Percentage"
            # "Percentage of Vegetation Belonging to Objects",
            # "Percentage of Vegetation Belonging to Non-Objects"
        ])
        if num_classes > 1:
            columns.extend([
                "Percentage of Vegetation Inside Object Boundaries",
                "Percentage of Vegetation Outside Object Boundaries"           
            ])
        for object_class in metadata["object_classes"]:
            columns.extend([
                "Percentage of Vegetation Inside " + object_class + " Boundaries",
                "Percentage of Vegetation Outside " + object_class + " Boundaries"    
            ])





    metrics_lst = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)"  
    ]

    if num_classes > 1:
        for metric in metrics_lst:
            columns.append(metric + " (All Classes)")


    for object_class in metadata["object_classes"]:
        for metric in metrics_lst:
            columns.append(metric + " (" + object_class + ")")

    d = {}
    for c in columns:
        d[c] = []



    for image_name in predictions.keys():

        # annotated_boxes = annotations[image_name]["boxes"]
        # predicted_boxes = predictions[image_name]["boxes"]
        # predicted_scores = predictions[image_name]["scores"]
        # sel_predicted_boxes = predictions[image_name]["boxes"][predicted_scores > 0.50]

        annotations_source = annotations[image_name]["source"]
        # above_thresh_predicted_boxes = predicted_boxes[predicted_scores >= 0.50]

        # metrics["MS COCO mAP"][image_name] = {}
        # metrics["F1 Score (IoU=0.5)"][image_name] = {}
        # metrics["F1 Score (IoU=0.7)"][image_name] = {}
        # metrics["F1 Score (IoU=0.9)"][image_name] = {}

        for region_type in ["regions_of_interest", "training_regions", "test_regions"]:


            # metrics["MS COCO mAP"][image_name][region_type + "_regions"] = []
            # metrics["F1 Score (IoU=0.5)"][image_name][region_type + "_regions"] = []
            # metrics["F1 Score (IoU=0.7)"][image_name][region_type + "_regions"] = []
            # metrics["F1 Score (IoU=0.9)"][image_name][region_type + "_regions"] = []

            regions = annotations[image_name][region_type]

            for region_idx, region in enumerate(regions):
                if region_type == "regions_of_interest":
                    region_name = "interest_" + (str(region_idx+1))

                elif region_type == "training_regions":
                    region_name = "fine_tuning_" + (str(region_idx+1))
                else:
                    region_name = "test_" + (str(region_idx+1))

                if region_type == "regions_of_interest":
                    area_px = poly_utils.get_poly_area(region)
                else:
                    height_px = region[2] - region[0]
                    width_px = region[3] - region[1]
                    area_px = height_px * width_px
                

                d["Username"].append(username)
                d["Farm Name"].append(farm_name)
                d["Field Name"].append(field_name)
                d["Mission Date"].append(mission_date)
                d["Image Name"].append(image_name)
                d["Region Name"].append(region_name)


                nav_item = image_name + "/" + str(region_idx)
                for tag_name in tags.keys():
                    if region_type == "regions_of_interest" and nav_item in tags[tag_name]:
                        d[tag_name].append(tags[tag_name][nav_item])
                    else:
                        d[tag_name].append("NA")


                d["Source Of Annotations (For Image)"].append(annotations_source)


                d["Area (Pixels)"].append(area_px)
                if include_density:
                    gsd = get_gsd(camera_specs, metadata)
                    area_m2 = calculate_area_m2(gsd, area_px)
                    d["Area (Square Metres)"].append(round(area_m2, 8))




                if num_classes > 1:
                    cls_tups = [(-1, "All Classes")]
                    for i, object_class in enumerate(metadata["object_classes"]):
                        cls_tups.append((i, object_class))
                else:
                    cls_tups = [(0, metadata["object_classes"][0])]

                for cls_tup in cls_tups:
                    cls_idx = cls_tup[0]
                    object_class = cls_tup[1]

                    annotated_boxes = []
                    predicted_boxes = []
                    for i in range(len(annotations[image_name]["boxes"])):
                        if cls_idx == -1 or annotations[image_name]["classes"][i] == cls_idx:
                            annotated_boxes.append(annotations[image_name]["boxes"][i])

                    for i in range(len(predictions[image_name]["boxes"])):
                        if cls_idx == -1 or predictions[image_name]["classes"][i] == cls_idx:
                            if predictions[image_name]["scores"][i] > 0.50:
                                predicted_boxes.append(predictions[image_name]["boxes"][i])

                    annotated_boxes = np.array(annotated_boxes)
                    predicted_boxes = np.array(predicted_boxes)


                    annotated_centres = (annotated_boxes[..., :2] + annotated_boxes[..., 2:]) / 2.0
                    predicted_centres = (predicted_boxes[..., :2] + predicted_boxes[..., 2:]) / 2.0

                    if region_type == "regions_of_interest":
                        annotated_inds = poly_utils.get_contained_inds_for_points(annotated_centres, [region])
                        predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])
                        
                    else:
                        annotated_inds = box_utils.get_contained_inds_for_points(annotated_centres, [region])
                        predicted_inds = box_utils.get_contained_inds_for_points(predicted_centres, [region])

                    annotated_boxes = annotated_boxes[annotated_inds]
                    predicted_boxes = predicted_boxes[predicted_inds]


                    annotated_count = annotated_boxes.shape[0]
                    predicted_count = predicted_boxes.shape[0]

                    d["Annotated Count (" + object_class + ")"].append(annotated_count)
                    d["Predicted Count (" + object_class + ")"].append(predicted_count)



                    if annotated_count > 0:
                        annotated_box_areas_px = box_utils.box_areas_np(annotated_boxes)
                        mean_annotated_object_area_px = round(np.mean(annotated_box_areas_px), 8)
                        stdev_annotated_object_area_px = round(np.std(annotated_box_areas_px), 8)
                    else:
                        mean_annotated_object_area_px = "NA"
                        stdev_annotated_object_area_px = "NA"

                    if predicted_count > 0:
                        predicted_box_areas_px = box_utils.box_areas_np(predicted_boxes)
                        mean_predicted_object_area_px = round(np.mean(predicted_box_areas_px), 8)
                        stdev_predicted_object_area_px = round(np.std(predicted_box_areas_px), 8)
                    else:
                        mean_predicted_object_area_px = "NA"
                        stdev_predicted_object_area_px = "NA"

                    d["Mean of Annotated Object Areas (Pixels) (" + object_class + ")"].append(mean_annotated_object_area_px)
                    d["Mean of Predicted Object Areas (Pixels) (" + object_class + ")"].append(mean_predicted_object_area_px)
                    d["Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")"].append(stdev_annotated_object_area_px)
                    d["Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")"].append(stdev_predicted_object_area_px)


                    if include_density:
                        gsd = get_gsd(camera_specs, metadata)
                        area_m2 = calculate_area_m2(gsd, area_px)
                        # if image_status == "unannotated":
                        #     d["annotated_plant_count_per_square_metre"].append("NA")
                        # else:
                        if area_m2 > 0:
                            annotated_count_per_square_metre = round(annotated_count / area_m2, 8)
                            predicted_count_per_square_metre = round(predicted_count / area_m2, 8)
                        else:
                            annotated_count_per_square_metre = "NA"
                            predicted_count_per_square_metre = "NA"
                        d["Annotated Count Per Square Metre (" + object_class + ")"].append(annotated_count_per_square_metre)
                        d["Predicted Count Per Square Metre (" + object_class + ")"].append(predicted_count_per_square_metre)
                        # d["Area (Square Metres)"].append(round(area_m2, 8))

                        if annotated_count > 0:
                            annotated_box_areas_m2 = calculate_area_m2(gsd, annotated_box_areas_px)
                            mean_annotated_object_area_m2 = round(np.mean(annotated_box_areas_m2), 8)
                            stdev_annotated_object_area_m2 = round(np.std(annotated_box_areas_m2), 8)
                        else:
                            mean_annotated_object_area_m2 = "NA"
                            stdev_annotated_object_area_m2 = "NA"

                        if predicted_count > 0:
                            predicted_box_areas_m2 = calculate_area_m2(gsd, predicted_box_areas_px)
                            mean_predicted_object_area_m2 = round(np.mean(predicted_box_areas_m2), 8)
                            stdev_predicted_object_area_m2 = round(np.std(predicted_box_areas_m2), 8)               
                        else:
                            mean_predicted_object_area_m2 = "NA"
                            stdev_predicted_object_area_m2 = "NA"

                        d["Mean of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(mean_annotated_object_area_m2)
                        d["Mean of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(mean_predicted_object_area_m2)
                        d["Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(stdev_annotated_object_area_m2)
                        d["Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(stdev_predicted_object_area_m2)



                    if annotated_count > 0:
                        percent_count_error = round(abs((predicted_count - annotated_count) / (annotated_count)) * 100, 2)
                    else:
                        percent_count_error = "NA" #undefined"

                    d["Percent Count Error (" + object_class + ")"].append(percent_count_error)


                    for metric in metrics_lst:
                        metric_val = updated_metrics[object_class][metric][image_name][region_type][region_idx]

                        if isinstance(metric_val, float):
                            metric_val = round(metric_val, 2)
                        d[metric + " (" + object_class + ")"].append(metric_val)



                if vegetation_record is not None:

                    d["Excess Green Threshold"].append(vegetation_record[image_name]["sel_val"])
                    vegetation_percentage = vegetation_record[image_name]["vegetation_percentage"][region_type][region_idx]
                    d["Vegetation Percentage"].append(vegetation_percentage)

                    if num_classes > 1:
                        obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"]["All Classes"][region_type][region_idx]
                        if vegetation_percentage == 0:
                            obj_percentage = "NA"
                            non_obj_percentage = "NA"
                        else:
                            obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                            non_obj_percentage = round(100 - obj_percentage, 2)

                        d["Percentage of Vegetation Inside Object Boundaries"].append(obj_percentage)
                        d["Percentage of Vegetation Outside Object Boundaries"].append(non_obj_percentage) 

                    for object_class in metadata["object_classes"]:

                        obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"][object_class][region_type][region_idx]
                        if vegetation_percentage == 0:
                            obj_percentage = "NA"
                            non_obj_percentage = "NA"
                        else:
                            obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                            non_obj_percentage = round(100 - obj_percentage, 2)

                        d["Percentage of Vegetation Inside " + object_class + " Boundaries"].append(obj_percentage)
                        d["Percentage of Vegetation Outside " + object_class + " Boundaries"].append(non_obj_percentage) 


    # json_io.print_json(d)
    # print()
    # for col in columns:
    #     print(col, len(d[col]))

    df = pd.DataFrame(data=d, columns=columns)
    df.sort_values(by="Image Name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["Image Name"])))
    return df #, metrics



def create_stats_sheet(args, regions_df):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    # predictions = args["predictions"]
    # annotations = args["annotations"]
    metadata = args["metadata"]
    # camera_specs = args["camera_specs"]
    # vegetation_record = args["vegetation_record"]
    columns = [
        "Username",
        "Farm Name", 
        "Field Name", 
        "Mission Date", 
        "Region Type", 
        # "Pearson's r: Annotated Count v. Predicted Count",
        # "Pearson's r: Annotated Count v. Vegetation Percentage"

    ]
 
    averaged_metrics = [
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)",
        # "AP (IoU=.50:.05:.95)",
        # "AP (IoU=.50)",
        # "AP (IoU=.75)"
    ]

    num_classes = len(metadata["object_classes"])

    object_classes = []
    if num_classes > 1:
        object_classes.append("All Classes")
        columns.extend([
            "Mean Absolute Difference In Count (All Classes)",
            "Mean Squared Difference In Count (All Classes)"
        ])
        for metric in averaged_metrics:
            columns.append(metric + " (All Classes)")

    for object_class in metadata["object_classes"]:
        object_classes.append(object_class)
        columns.extend([
            "Mean Absolute Difference In Count (" + object_class + ")",
            "Mean Squared Difference In Count (" + object_class + ")"
        ])
        for metric in averaged_metrics:
            columns.append(metric + " (" + object_class + ")")

    # columns.extend(averaged_metrics)

    d = {}
    for c in columns:
        d[c] = []

    if len(regions_df.index) > 0:

        for region_type in ["regions_of_interest", "training_regions", "test_regions"]: #["interest", "training", "test"]:
            if region_type == "regions_of_interest":
                disp_region_type = "interest"
            elif region_type == "training_regions":
                disp_region_type = "fine_tuning"
            else:
                disp_region_type = "test"

            sub_df = regions_df[regions_df["Region Name"].str.contains(disp_region_type)]

            # print(sub_df)

            if len(sub_df) > 0:

                d["Username"].append(username)
                d["Farm Name"].append(farm_name)
                d["Field Name"].append(field_name)
                d["Mission Date"].append(mission_date)
                d["Region Type"].append(disp_region_type)


                for object_class in object_classes:
                    d["Mean Absolute Difference In Count (" + object_class + ")"].append(
                        round(float(np.mean(abs(sub_df["Annotated Count (" + object_class + ")"] - sub_df["Predicted Count (" + object_class + ")"]))), 2)
                    )

                    d["Mean Squared Difference In Count (" + object_class + ")"].append(
                        round(float(np.mean((sub_df["Annotated Count (" + object_class + ")"] - sub_df["Predicted Count (" + object_class + ")"]) ** 2)), 2)
                    )


                    
                    # # mean_abs_diff_in_count = round(np.mean(abs(sub_df["Annotated Count"] - sub_df["Predicted Count"])), 2)
                    # d["Mean Absolute Difference In Count"].append(
                    #     round(float(np.mean(abs(sub_df["Annotated Count"] - sub_df["Predicted Count"]))), 2)
                    # )
                    # # mean_squared_diff_in_count = round(np.mean((sub_df["Annotated Count"] - sub_df["Predicted Count"]) ** 2), 2)
                    # d["Mean Squared Difference In Count"].append(
                    #     round(float(np.mean((sub_df["Annotated Count"] - sub_df["Predicted Count"]) ** 2)), 2)
                    # )

                    # d["Pearson's r: Annotated Count v. Predicted Count"] = round(float(np.corrcoef(sub_df["Annotated Count"], sub_df["Predicted Count"])[0][1]), 2)
                    # d["Pearson's r: Annotated Count v. Vegetation Percentage"] = round(float(np.corrcoef(sub_df["Annotated Count"], sub_df["Vegetation Percentage"])[0][1]), 2)

                    
                    for metric in averaged_metrics:

                        # if region_type == "regions_of_interest":
                        #     metric_val = "NA"
                        # else:
                        # metric_val = updated_metrics[metric][image_name][region_type + "_regions"][i]
                        try:
                            # if isinstance(metric_val, float):
                            metric_val = round(float(np.mean(sub_df[metric + " (" + object_class + ")"])), 2)
                        except Exception:
                            metric_val = "unable_to_calculate"
                        
                        d[metric + " (" + object_class + ")"].append(metric_val)
                        # d[metric].append(round(float(np.mean(sub_df[metric])), 2))

    print(d)

    df = pd.DataFrame(data=d, columns=columns)
        # df.sort_values(by="Image Name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["Image Name"])))
    return df #, metrics

        # d["Annotated Count"]

        # for image_name in annotations.keys():

        #     annotated_boxes = annotations[image_name]["boxes"]
        #     predicted_boxes = predictions[image_name]["boxes"]
        #     predicted_scores = predictions[image_name]["scores"]



            
        #     for region in annotations[image_name][region_type]:









# def prepare_report(out_path, farm_name, field_name, mission_date, 
#                    image_predictions, annotations, excess_green_record, metrics):

#     logger = logging.getLogger(__name__)
#     logger.info("Preparing report")

#     #num_classes = len(config["arch"]["class_map"].keys())



#     d = {
#         "farm_name": [],
#         "field_name": [],
#         "mission_date": [],
#         "image_status": [],
#         "image_name": [],
#         "annotated_plant_count": [],
#         "predicted_plant_count": [],
#         "ground_cover_percentage": [],
#         "MS_COCO_mAP": [],
#         #"PASCAL_VOC_mAP": []
#     }

#     # for class_name in config["arch"]["class_map"].keys():
#     #     d["annotated_" + class_name + "_count"] = []
#     #     d["model_" + class_name + "_count"] = []


#     for image_name in tqdm.tqdm(image_predictions.keys(), desc="Collecting metrics"):

#         image_abs_boxes = annotations[image_name]["boxes"]
#         image_status = annotations[image_name]["status"]


#         d["farm_name"].append(farm_name)
#         d["field_name"].append(field_name)
#         d["mission_date"].append(mission_date)
#         d["image_status"].append(image_status)
#         d["image_name"].append(image_name)

#         if image_status == "unannotated":
#             annotated_count = "NA"
#         else:
#             annotated_count = image_abs_boxes.shape[0]
#         pred_abs_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
#         #pred_classes = np.array(image_predictions[image_name]["pred_classes"])
#         pred_scores = np.array(image_predictions[image_name]["pred_scores"])

#         mask = pred_scores >= 0.5

#         sel_pred_abs_boxes = pred_abs_boxes[mask]
#         #sel_pred_classes = pred_classes[mask]
#         sel_pred_scores = pred_scores[mask]

#         d["annotated_" + "plant" + "_count"].append(str(annotated_count))
#         d["predicted_" + "plant" + "_count"].append(str(sel_pred_abs_boxes.shape[0]))

#         d["ground_cover_percentage"].append("%.2f" % (excess_green_record[image_name]["ground_cover_percentage"]))

#         if image_name in metrics:
#             d["MS_COCO_mAP"].append(metrics[image_name]["Image MS COCO mAP"])
#         else:
#             d["MS_COCO_mAP"].append("NA")


#     pandas.io.formats.excel.ExcelFormatter.header_style = None
#     df = pd.DataFrame(data=d)
#     df.sort_values(by="image_name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["image_name"])))
#     df.to_csv(out_path, index=False)
    
#     # writer = pd.ExcelWriter(out_path, engine="xlsxwriter")

#     # df.to_excel(writer, index=False, sheet_name="Sheet1", na_rep='NA')  # send df to writer
#     # worksheet = writer.sheets["Sheet1"]  # pull worksheet object


#     # # for i, width in enumerate(get_col_widths(df)):
#     # #     worksheet.set_column(i, i, width)
#     # # for column in df:
#     # #     worksheet.column_dimensions[column].bestFit = True
#     # for idx, col in enumerate(df):  # loop through all columns
#     #     series = df[col]
#     #     max_len = max((
#     #         series.astype(str).map(len).max(),  # len of largest item
#     #         len(str(series.name))  # len of column name/header
#     #         )) # + 1  # adding a little extra space
#     #     worksheet.set_column(idx, idx, 0.9 * max_len)  # set column width
#     # writer.save()




# def collect_metrics(image_names, metrics, predictions, dataset, config,
#                     collect_patch_metrics=True, calculate_mAP=True):

#     logger = logging.getLogger(__name__)

#     num_classes = len(config["arch"]["class_map"].keys())

#     annotated_image_counts = {}
#     pred_image_counts = {}

#     class_map = config["arch"]["class_map"]
#     reverse_class_map =  config["arch"]["reverse_class_map"]

#     all_images_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes) #config.arch["num_classes"])

#     annotations = w3c_io.load_annotations(dataset.annotations_path, config["arch"]["class_map"])


#     # completed_images = w3c_io.get_completed_images(annotations)
#     # optimal_thresh_val, optimal_mean_abs_diff = calculate_optimal_score_threshold(annotations, predictions, completed_images)


    
#     point_metrics = metrics["point"]
#     boxplot_metrics = metrics["boxplot"]    
#     image_metrics = metrics["image"]
    
#     # point_metrics["true_optimal_score_threshold"] = {}
#     # point_metrics["true_optimal_score_threshold"]["threshold_value"] = optimal_thresh_val
#     # point_metrics["true_optimal_score_threshold"]["mean_absolute_difference"] = optimal_mean_abs_diff





#     annotated_image_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
#     pred_image_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}

#     #for image in dataset.completed_images:
#     for image_name in image_names: #predictions["image_predictions"].keys():

#         if annotations[image_name]["status"] == "completed":

#             #img_abs_boxes, img_classes = xml_io.load_boxes_and_classes(img.xml_path, img_set.class_map) #config.arch["class_map"])
#             image_abs_boxes = annotations[image_name]["boxes"]
#             image_classes = annotations[image_name]["classes"]
            
            
#             unique, counts = np.unique(image_classes, return_counts=True)
#             class_num_to_count = dict(zip(unique, counts))
#             cur_image_class_counts = {k: 0 for k in class_map} #config.arch["class_map"].keys()}
#             for class_num in class_num_to_count.keys():
#                 cur_image_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
#                 #cur_img_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

#             cur_image_pred_class_counts = predictions["image_predictions"][image_name]["pred_class_counts"]

#             for class_name in class_map.keys(): #config.arch["class_map"].keys():
#                 annotated_image_counts[class_name].append(cur_image_class_counts[class_name])
#                 pred_image_counts[class_name].append(cur_image_pred_class_counts[class_name])

#             #annotated_img_count = np.shape(img_abs_boxes)[0]

#             #pred_img_count = predictions["image_predictions"][img.img_name]["pred_count"]
            
#             if calculate_mAP:
#                 pred_abs_boxes = np.array(predictions["image_predictions"][image_name]["pred_image_abs_boxes"])
#                 pred_classes = np.array(predictions["image_predictions"][image_name]["pred_classes"])
#                 pred_scores = np.array(predictions["image_predictions"][image_name]["pred_scores"])

#                 pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
#                                                                     image_abs_boxes, image_classes)
#                 all_images_metric_fn.add(pred_for_mAP, true_for_mAP)


#                 image_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=num_classes)
#                 image_metric_fn.add(pred_for_mAP, true_for_mAP)
#                 pascal_voc_mAP = image_metric_fn.value(iou_thresholds=0.5)['mAP']
#                 coco_mAP = image_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']


#                 image_metrics[image_name]["Image PASCAL VOC mAP"] = float(pascal_voc_mAP) * 100
#                 image_metrics[image_name]["Image MS COCO mAP"] = float(coco_mAP) * 100

                
#                 #annotated_img_counts.append(annotated_img_count)
#             #pred_img_counts.append(pred_img_count)


#     #annotated_img_counts = np.array(annotated_img_counts)
#     #pred_img_counts = np.array(pred_img_counts)

#     # annotated_patch_counts = []
#     # pred_patch_counts = []

#     if collect_patch_metrics:
#         annotated_patch_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
#         pred_patch_counts = {k: [] for k in class_map.keys()} #config.arch["class_map"].keys()}
#         for patch_name, patch_pred in predictions["patch_predictions"].items():

#             if "patch_classes" in patch_pred:
#                 patch_classes = patch_pred["patch_classes"]
#                 unique, counts = np.unique(patch_classes, return_counts=True)
#                 class_num_to_count = dict(zip(unique, counts))
#                 cur_patch_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
#                 for class_num in class_num_to_count.keys():
#                     cur_patch_class_counts[reverse_class_map[class_num]] = class_num_to_count[class_num]
#                     #cur_patch_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_count[class_num]

#                 pred_patch_classes = patch_pred["pred_classes"]
#                 pred_unique, pred_counts = np.unique(pred_patch_classes, return_counts=True)
#                 class_num_to_pred_count = dict(zip(pred_unique, pred_counts))
#                 cur_patch_pred_class_counts = {k: 0 for k in class_map.keys()} #config.arch["class_map"].keys()}
#                 for class_num in class_num_to_pred_count.keys():
#                     cur_patch_pred_class_counts[reverse_class_map[class_num]] = class_num_to_pred_count[class_num]
#                     #cur_patch_pred_class_counts[config.arch["reverse_class_map"][class_num]] = class_num_to_pred_count[class_num]

#                 for class_name in class_map.keys(): #config.arch["class_map"].keys():
#                     annotated_patch_counts[class_name].append(cur_patch_class_counts[class_name])
#                     pred_patch_counts[class_name].append(cur_patch_pred_class_counts[class_name])


#     print("annotated_image_counts", annotated_image_counts)
#     total_obj_count = float(np.sum([np.sum(np.array(annotated_image_counts[class_name])) for class_name in annotated_image_counts.keys()]))

#     logger.info("Started calculating count metrics.")

#     # if "metrics" not in predictions:
#     #     predictions["metrics"] = {}

#     # if "point" not in predictions["metrics"]:
#     #     predictions["metrics"]["point"] = {}

#     # if "boxplot" not in predictions["metrics"]:
#     #     predictions["metrics"]["boxplot"] = {}




#     point_metrics["Image Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#     point_metrics["Image Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#     point_metrics["Image R Squared"] = {"Cross-Class Weighted Average": 0}
#     point_metrics["Image Non-Zero Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#     if collect_patch_metrics:
#         point_metrics["Patch Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#         point_metrics["Patch Mean Sq. Diff. in Count"] = {"Cross-Class Weighted Average": 0}
#         point_metrics["Patch R Squared"] = {"Cross-Class Weighted Average": 0}
#         point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"] = {"Cross-Class Weighted Average": 0}


#     boxplot_metrics["Difference in Count (Image)"] = {}
#     boxplot_metrics["Absolute Difference in Count (Image)"] = {} #{"Cross-Class Weighted Average": 0}
#     boxplot_metrics["Percent Difference in Count (Image)"] = {} #{"Cross-Class Weighted Average": 0}
#     #boxplot_metrics["Diff. in Count (Patch)"] = {}
#     #boxplot_metrics["Absolute Diff. in Count (Patch)"] = {} #{"Cross-Class Weighted Average": 0}
#     #boxplot_metrics["Percent Diff. in Count (Patch)"] = {} #{"Cross-Class Weighted Average": 0}
    
#     # boxplot_metrics["Confidence"] = {"Cross-Class Weighted Average": 0}
#     # boxplot_metrics["Box Area"] = {"Cross-Class Weighted Average": 0}
#     # boxplot_metrics["Inference Time (Per Image)"] = {"Cross-Class Weighted Average": 0}
#     # boxplot_metrics["Inference Time (Per Patch)"] = {"Cross-Class Weighted Average": 0}
    
#     #predictions["metrics"]["Patch Five Num. Summary Diff. in Count"] = {}
#     #predictions["metrics"]["Diff. in Count Occurrences"] = {}

#     for class_name in annotated_image_counts.keys():

#         image_class_annotated_count = np.array(annotated_image_counts[class_name])
#         image_class_pred_count = np.array(pred_image_counts[class_name])

#         point_metrics["Image Mean Abs. Diff. in Count"][class_name] = \
#                         float(np.mean(abs_DiC(image_class_annotated_count, image_class_pred_count)))
#         point_metrics["Image Mean Sq. Diff. in Count"][class_name] = \
#                         float(np.mean(squared_DiC(image_class_annotated_count, image_class_pred_count)))
#         point_metrics["Image R Squared"][class_name] = \
#                         r_squared(image_class_annotated_count, image_class_pred_count)
#         point_metrics["Image Non-Zero Mean Abs. Diff. in Count"][class_name] = \
#                         float(np.mean(nonzero_abs_DiC(image_class_annotated_count, image_class_pred_count)))


        



#         #predictions["metrics"]["Patch Five Num. Summary Diff. in Count"][class_name] = five_num_summary_DiC(patch_class_annotated_count, patch_class_pred_count)
#         #predictions["metrics"]["Diff. in Count Occurrences"][class_name] = DiC_occurrences(patch_class_annotated_count, patch_class_pred_count)
        
#         boxplot_metrics["Difference in Count (Image)"][class_name] = \
#                     boxplot_data(DiC(image_class_annotated_count, image_class_pred_count))

#         boxplot_metrics["Absolute Difference in Count (Image)"][class_name] = \
#                     boxplot_data(abs_DiC(image_class_annotated_count, image_class_pred_count))

#         boxplot_metrics["Percent Difference in Count (Image)"][class_name] = \
#                     boxplot_data(pct_DiC(image_class_annotated_count, image_class_pred_count))

#         # boxplot_metrics["Diff. in Count (Patch)"][class_name] = \
#         #             boxplot_data(DiC(patch_class_annotated_count, patch_class_pred_count))

#         # boxplot_metrics["Absolute Diff. in Count (Patch)"][class_name] = \
#         #             boxplot_data(abs_DiC(patch_class_annotated_count, patch_class_pred_count))

#         # boxplot_metrics["Percent Diff. in Count (Patch)"][class_name] = \
#         #             boxplot_data(pct_DiC(patch_class_annotated_count, patch_class_pred_count))



        
#         total_class_annotated_count = float(np.sum(image_class_annotated_count))

#         point_metrics["Image Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image Mean Abs. Diff. in Count"][class_name]
#         point_metrics["Image Mean Sq. Diff. in Count"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image Mean Sq. Diff. in Count"][class_name]
#         point_metrics["Image R Squared"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image R Squared"][class_name]
#         point_metrics["Image Non-Zero Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#             (total_class_annotated_count / total_obj_count) * point_metrics["Image Non-Zero Mean Abs. Diff. in Count"][class_name]



#         if collect_patch_metrics:

#             patch_class_annotated_count = np.array(annotated_patch_counts[class_name])
#             patch_class_pred_count = np.array(pred_patch_counts[class_name])

#             point_metrics["Patch Mean Abs. Diff. in Count"][class_name] = \
#                             float(np.mean(abs_DiC(patch_class_annotated_count, patch_class_pred_count)))
#             point_metrics["Patch Mean Sq. Diff. in Count"][class_name] = \
#                             float(np.mean(squared_DiC(patch_class_annotated_count, patch_class_pred_count)))
#             point_metrics["Patch R Squared"][class_name] = \
#                             r_squared(patch_class_annotated_count, patch_class_pred_count)
#             point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"][class_name] = \
#                             float(np.mean(nonzero_abs_DiC(patch_class_annotated_count, patch_class_pred_count)))

#             point_metrics["Patch Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch Mean Abs. Diff. in Count"][class_name]
#             point_metrics["Patch Mean Sq. Diff. in Count"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch Mean Sq. Diff. in Count"][class_name]
#             point_metrics["Patch R Squared"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch R Squared"][class_name]
#             point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"]["Cross-Class Weighted Average"] += \
#                 (total_class_annotated_count / total_obj_count) * point_metrics["Patch Non-Zero Mean Abs. Diff. in Count"][class_name]




#     logger.info("Finished calculating count metrics.")

#     if calculate_mAP:
#         logger.info("Started calculating mAP scores.")

#         pascal_voc_mAP = all_images_metric_fn.value(iou_thresholds=0.5)['mAP']
#         coco_mAP = all_images_metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

#         logger.info("Finished calculating mAP scores.")

#         point_metrics["Image PASCAL VOC mAP"] = {}
#         point_metrics["Image PASCAL VOC mAP"]["---"] = float(pascal_voc_mAP) * 100
#         point_metrics["Image MS COCO mAP"] = {}
#         point_metrics["Image MS COCO mAP"]["---"] = float(coco_mAP) * 100




def get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                              true_abs_boxes, true_classes):

    if pred_abs_boxes.size > 0:
        pred_abs_boxes = box_utils.swap_xy_np(pred_abs_boxes)
    else:
        pred_abs_boxes = np.reshape(pred_abs_boxes, (0, 4)) #np.expand_dims(pred_abs_boxes, axis=-1)
        
    pred_classes = np.expand_dims(pred_classes, axis=-1)
    pred_scores = np.expand_dims(pred_scores, axis=-1)
    pred = np.hstack([pred_abs_boxes, pred_classes, pred_scores])

    if true_abs_boxes.size > 0:
        true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    else:
        #true_abs_boxes = np.expand_dims(true_abs_boxes, axis=-1)
        true_abs_boxes = np.reshape(true_abs_boxes, (0, 4)) 
    #true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    true_classes = np.expand_dims(true_classes, axis=-1)
    difficult = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    crowd = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    true = np.hstack([true_abs_boxes, true_classes, difficult, crowd])  

    return pred, true





