import logging
import os
import numpy as np
import shutil


from io_utils import json_io
from models.common import box_utils
from lsnms import nms as lsnms_nms





def get_image_detections_any_overlap(patch_abs_boxes, patch_scores, patch_classes, patch_coords, 
                         region, overlap_px, trim=True): #, patch_border_buffer_percent=None): # buffer_pct=None):

    if patch_abs_boxes.size == 0:
        image_abs_boxes = np.array([], dtype=np.int32)
        image_scores = np.array([], dtype=np.float32)
        image_classes = np.array([], dtype=np.int32)

    else:
        patch_height = patch_coords[2] - patch_coords[0]
        patch_width = patch_coords[3] - patch_coords[1]


        image_abs_boxes = (np.array(patch_abs_boxes) + \
                           np.tile(patch_coords[:2], 2)).astype(np.int32)

        if trim:

            wiggle_room = 3

            accept_bottom = region[0] if patch_coords[0] == region[0] else patch_coords[0] + round(overlap_px / 2) - wiggle_room
            accept_left = region[1] if patch_coords[1] == region[1] else patch_coords[1] + round(overlap_px / 2) - wiggle_room
            accept_top = region[2] if patch_coords[2] >= region[2] else patch_coords[2] - round(overlap_px / 2) + wiggle_room
            accept_right = region[3] if patch_coords[3] >= region[3] else patch_coords[3] - round(overlap_px / 2) + wiggle_room


            box_centres = (image_abs_boxes[..., :2] + image_abs_boxes[..., 2:]) / 2.0

            mask = np.logical_and(
                np.logical_and(box_centres[:,0] >= accept_bottom, box_centres[:,0] < accept_top),
                np.logical_and(box_centres[:,1] >= accept_left, box_centres[:,1] < accept_right)
            )

            image_abs_boxes = image_abs_boxes[mask]
            image_scores = patch_scores[mask]
            image_classes = patch_classes[mask]

        else:
            image_scores = patch_scores
            image_classes = patch_classes


    return image_abs_boxes, image_scores, image_classes



def get_image_detections(patch_abs_boxes, patch_scores, patch_classes, patch_coords, region, trim=True):

    if patch_abs_boxes.size == 0:
        image_abs_boxes = np.array([], dtype=np.int32)
        image_scores = np.array([], dtype=np.float32)
        image_classes = np.array([], dtype=np.int32)

    else:
        patch_height = patch_coords[2] - patch_coords[0]
        patch_width = patch_coords[3] - patch_coords[1]

        image_abs_boxes = (np.array(patch_abs_boxes) + \
                           np.tile(patch_coords[:2], 2)).astype(np.int32)

        if trim:

            wiggle_room = 3

            accept_bottom = region[0] if patch_coords[0] == region[0] else patch_coords[0] + round(patch_height / 4) - wiggle_room
            accept_left = region[1] if patch_coords[1] == region[1] else patch_coords[1] + round(patch_width / 4) - wiggle_room
            accept_top = region[2] if patch_coords[2] >= region[2] else patch_coords[2] - round(patch_height / 4) + wiggle_room
            accept_right = region[3] if patch_coords[3] >= region[3] else patch_coords[3] - round(patch_width / 4) + wiggle_room


            box_centres = (image_abs_boxes[..., :2] + image_abs_boxes[..., 2:]) / 2.0

            mask = np.logical_and(
                np.logical_and(box_centres[:,0] >= accept_bottom, box_centres[:,0] < accept_top),
                np.logical_and(box_centres[:,1] >= accept_left, box_centres[:,1] < accept_right)
            )

            image_abs_boxes = image_abs_boxes[mask]
            image_scores = patch_scores[mask]
            image_classes = patch_classes[mask]

        else:
            image_scores = patch_scores
            image_classes = patch_classes

    return image_abs_boxes, image_scores, image_classes



def apply_nms_to_image_boxes(predictions, iou_thresh):

        for image_name in predictions.keys():
            if len(predictions[image_name]["boxes"]) > 0:
                pred_image_abs_boxes = np.array(predictions[image_name]["boxes"])
                pred_classes = np.array(predictions[image_name]["classes"])
                pred_scores = np.array(predictions[image_name]["scores"])

                nms_indices = lsnms_nms.nms(box_utils.swap_xy_np(pred_image_abs_boxes),
                                                    pred_scores,
                                                    class_ids=pred_classes,
                                                    iou_threshold=iou_thresh)

                nms_boxes = pred_image_abs_boxes[nms_indices]
                nms_scores = pred_scores[nms_indices]
                nms_classes = pred_classes[nms_indices]

            else:
                nms_boxes = np.array([])
                nms_scores = np.array([])
                nms_classes = np.array([])

            predictions[image_name]["boxes"] = nms_boxes.tolist()
            predictions[image_name]["classes"] = nms_classes.tolist()
            predictions[image_name]["scores"] = nms_scores.tolist()



def add_class_detections(image_predictions, config): #, opt_thresh_val):

    class_map = config["arch"]["class_map"]
    reverse_class_map = {v: k for k, v in class_map.items()}

    for image_name in image_predictions.keys():
        pred_boxes = np.array(image_predictions[image_name]["pred_image_abs_boxes"])
        pred_classes = np.array(image_predictions[image_name]["pred_classes"])
        pred_scores = np.array(image_predictions[image_name]["pred_scores"])
        unique = np.unique(pred_classes)

        pred_class_counts = {k: 0 for k in class_map.keys()}
        pred_class_boxes = {k: [] for k in class_map.keys()}
        pred_class_scores = {k: [] for k in class_map.keys()}
        #pred_opt_class_counts = {k: 0 for k in class_map.keys()}

        for class_num in unique:
            class_name = reverse_class_map[class_num]
            inds = np.where(pred_classes == class_num)[0]

            pred_class_counts[class_name] = int(pred_scores[inds].size)
            pred_class_boxes[class_name] = (pred_boxes[inds]).tolist()
            pred_class_scores[class_name] = (pred_scores[inds]).tolist()
            #pred_opt_class_counts[class_name] = int((np.where(pred_scores[inds] >= opt_thresh_val)[0]).size)

        
        # class_num_to_count = dict(zip(unique, counts))
        # #pred_class_counts = {k: 0 for k in config.arch["class_map"].keys()}
        # pred_class_counts = {k: 0 for k in class_map.keys()}
        # pred_class_boxes = {k: [] for k in class_map.keys()}
        # pred_class_scores = {k: [] for k in class_map.keys()}
        # for class_num in class_num_to_count.keys():
        #     class_name = reverse_class_map[class_num]
        #     #class_name = config.arch["reverse_class_map"][class_num]
        #     pred_class_counts[class_name] = int(class_num_to_count[class_num])
        #     pred_class_boxes[class_name] = (pred_boxes[class_num == pred_classes]).tolist()
        #     pred_class_scores[class_name] = (pred_scores[class_num == pred_classes]).tolist()


        image_predictions[image_name]["pred_class_counts"] = pred_class_counts
        image_predictions[image_name]["pred_class_boxes"] = pred_class_boxes
        image_predictions[image_name]["pred_class_scores"] = pred_class_scores
        #image_predictions[image_name]["pred_opt_class_counts"] = pred_opt_class_counts


