import logging
import os
import numpy as np
import tqdm

from image_set import ImgSet
from io_utils import json_io
from models.common import box_utils, driver_utils, inference_metrics

def combine_img_predictions(ens_pred, predictions_lst):

    for predictions in tqdm.tqdm(predictions_lst, desc="Combining image predictions"):
        for img_name in predictions["image_predictions"].keys():
            if img_name not in ens_pred["image_predictions"]:
                ens_pred["image_predictions"][img_name] = {}
                ens_pred["image_predictions"][img_name]["all_pred_class_boxes"] = {}
                ens_pred["image_predictions"][img_name]["all_pred_class_scores"] = {}

            for class_name in predictions["image_predictions"][img_name]["pred_class_boxes"].keys():

                if class_name not in ens_pred["image_predictions"][img_name]["all_pred_class_boxes"]:
                    ens_pred["image_predictions"][img_name]["all_pred_class_boxes"][class_name] = []
                    ens_pred["image_predictions"][img_name]["all_pred_class_scores"][class_name] = []


                ens_pred["image_predictions"][img_name]["all_pred_class_boxes"][class_name].extend(
                        predictions["image_predictions"][img_name]["pred_class_boxes"][class_name])

                ens_pred["image_predictions"][img_name]["all_pred_class_scores"][class_name].extend(
                        predictions["image_predictions"][img_name]["pred_class_scores"][class_name])



def group_img_predictions(ens_pred, iou_thresh=0.5):


    for img_name in tqdm.tqdm(ens_pred["image_predictions"].keys(), desc="Grouping image predictions"):

        ens_pred["image_predictions"][img_name]["grouped_pred_class_boxes"] = {}
        ens_pred["image_predictions"][img_name]["grouped_pred_class_scores"] = {}

        for class_name in ens_pred["image_predictions"][img_name]["all_pred_class_boxes"].keys():

            ens_pred["image_predictions"][img_name]["grouped_pred_class_boxes"][class_name] = []
            ens_pred["image_predictions"][img_name]["grouped_pred_class_scores"][class_name] = []

            boxes = np.array(ens_pred["image_predictions"][img_name]["all_pred_class_boxes"][class_name])
            scores = np.array(ens_pred["image_predictions"][img_name]["all_pred_class_scores"][class_name])
            available = np.full(boxes.shape[0], True) #[False] * len(boxes)


            iou_mat = box_utils.compute_iou(boxes, boxes, box_format="corners_yx")
            
            for i in range(boxes.shape[0]):
                
                if available[i]:
                    group_mask = np.logical_and(iou_mat[i, :] > iou_thresh, available)
                    box_group = boxes[group_mask]
                    score_group = scores[group_mask]
                    available = np.logical_and(available, np.logical_not(group_mask))

                    ens_pred["image_predictions"][img_name]["grouped_pred_class_boxes"][class_name].append(box_group.tolist())
                    ens_pred["image_predictions"][img_name]["grouped_pred_class_scores"][class_name].append(score_group.tolist())



def _ensemble_predictions(predictions_lst, method, 
                          inter_group_iou_thresh, intra_group_iou_thresh):


    img_set = ImgSet(predictions_lst[0]["farm_name"],
                     predictions_lst[0]["field_name"],
                     predictions_lst[0]["mission_date"])
    dataset = img_set.datasets[predictions_lst[0]["dataset_name"]]

    num_methods = len(predictions_lst)

    ens_pred = driver_utils.create_predictions_skeleton(img_set, dataset)

    combine_img_predictions(ens_pred, predictions_lst)

    group_img_predictions(ens_pred, iou_thresh=inter_group_iou_thresh)

    for img_name in ens_pred["image_predictions"].keys():

        ens_pred["image_predictions"][img_name]["pred_class_boxes"] = {}
        ens_pred["image_predictions"][img_name]["pred_class_scores"] = {}
        ens_pred["image_predictions"][img_name]["pred_class_counts"] = {}
        ens_pred["image_predictions"][img_name]["singles_class_boxes"] = {}
        ens_pred["image_predictions"][img_name]["singles_class_scores"] = {}

        for class_name in ens_pred["image_predictions"][img_name]["grouped_pred_class_boxes"].keys():

            ens_pred["image_predictions"][img_name]["pred_class_boxes"][class_name] = []
            ens_pred["image_predictions"][img_name]["pred_class_scores"][class_name] = []
            ens_pred["image_predictions"][img_name]["singles_class_boxes"][class_name] = []
            ens_pred["image_predictions"][img_name]["singles_class_scores"][class_name] = []


            for i in range(len(ens_pred["image_predictions"][img_name]["grouped_pred_class_boxes"][class_name])):


                box_group = np.array(ens_pred["image_predictions"][img_name]["grouped_pred_class_boxes"][class_name][i])
                score_group = np.array(ens_pred["image_predictions"][img_name]["grouped_pred_class_scores"][class_name][i])

                group_size = score_group.size

                sel_boxes = np.array([])
                sel_scores = np.array([])

                if method == "unanimous":
                    if group_size == num_methods:
                        sel_boxes, sel_scores = \
                                box_utils.non_max_suppression(box_group, score_group, intra_group_iou_thresh)


                elif method == "consensus":
                    if group_size > (num_methods / 2):
                        sel_boxes, sel_scores = \
                                box_utils.non_max_suppression(box_group, score_group, intra_group_iou_thresh)


                elif method == "affirmative":
                    sel_boxes, sel_scores = \
                            box_utils.non_max_suppression(box_group, score_group, intra_group_iou_thresh)


                else:
                    raise RuntimeError("Unknown ensemble method: {}".format(method))

                ens_pred["image_predictions"][img_name]["pred_class_boxes"][class_name].extend(sel_boxes.tolist())
                ens_pred["image_predictions"][img_name]["pred_class_scores"][class_name].extend(sel_scores.tolist())


                if group_size == 1:
                    ens_pred["image_predictions"][img_name]["singles_class_boxes"][class_name].extend(box_group.tolist())
                    ens_pred["image_predictions"][img_name]["singles_class_scores"][class_name].extend(score_group.tolist())

            ens_pred["image_predictions"][img_name]["pred_class_counts"][class_name] = \
                len(ens_pred["image_predictions"][img_name]["pred_class_scores"][class_name])


    inference_metrics.collect_statistics(ens_pred, img_set, dataset)
    inference_metrics.collect_metrics(ens_pred, img_set, dataset, 
                                      collect_patch_metrics=False, calculate_mAP=False)
    return ens_pred




def ensemble_predictions(req_args):

    logger = logging.getLogger(__name__)


    ensemble_uuid = req_args["ensemble_uuid"]
    model_uuids = req_args["model_uuids"]
    prediction_dirnames = req_args["prediction_dirnames"]
    ensemble_method = req_args["ensemble_method"]
    inter_group_iou_thresh = req_args["inter_group_iou_thresh"]
    intra_group_iou_thresh = req_args["intra_group_iou_thresh"]

    predictions_lst = []

    for i, (model_uuid, prediction_dirname) in enumerate(zip(model_uuids, prediction_dirnames)):
        predictions_path = os.path.join("usr", "data", "models", model_uuid, "predictions", 
                                         prediction_dirname, "predictions.json")
        predictions = json_io.load_json(predictions_path)
        predictions_lst.append(predictions)

    ensemble_predictions = _ensemble_predictions(predictions_lst,
                                                 ensemble_method,
                                                 inter_group_iou_thresh,
                                                 intra_group_iou_thresh)


    ensemble_dir = os.path.join("usr", "data", "ensembles", ensemble_uuid)
    os.makedirs(ensemble_dir, exist_ok=True)
    ensemble_predictions_path = os.path.join(ensemble_dir, "predictions.json")
    #ensemble_request_path = os.path.join(ensemble_dir, "ensemble_config.json")

    #json_io.save_json(ensemble_request_path, req_args)
    json_io.save_json(ensemble_predictions_path, ensemble_predictions)

