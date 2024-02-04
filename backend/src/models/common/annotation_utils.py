import math as m
import numpy as np

from models.common import poly_utils

from io_utils import json_io


def is_fully_annotated(annotations, image_name, image_w, image_h):
    return is_fully_annotated_for_fine_tuning(annotations, image_name, image_w, image_h) or is_fully_annotated_for_testing(annotations, image_name, image_w, image_h)


def is_fully_annotated_for_fine_tuning(annotations, image_name, image_w, image_h):
    image_area = image_w * image_h;
    for region in annotations[image_name]["fine_tuning_regions"]:

        reg_area = poly_utils.get_poly_area(region)
        if reg_area == image_area:
            return True
        
    return False

def is_fully_annotated_for_testing(annotations, image_name, image_w, image_h):

    image_area = image_w * image_h;
    for region in annotations[image_name]["test_regions"]:

        reg_area = poly_utils.get_poly_area(region)
        if reg_area == image_area:
            return True
        
    return False

def load_annotations(annotations_path):
    annotations = json_io.load_json(annotations_path)
    for image_name in annotations.keys():
        annotations[image_name]["boxes"] = np.array(annotations[image_name]["boxes"])
        annotations[image_name]["classes"] = np.array(annotations[image_name]["classes"])

    return annotations


def load_predictions(predictions_path):
    predictions = json_io.load_json(predictions_path)
    for image_name in predictions.keys():
        predictions[image_name]["boxes"] = np.array(predictions[image_name]["boxes"])
        predictions[image_name]["scores"] = np.array(predictions[image_name]["scores"])
        predictions[image_name]["classes"] = np.array(predictions[image_name]["classes"])

    return predictions


def save_annotations(annotations_path, annotations):
    save_annotations = {}
    for image_name in annotations.keys():
        save_annotations[image_name] = annotations[image_name]
        save_annotations[image_name]["boxes"] = annotations[image_name]["boxes"].tolist()
        save_annotations[image_name]["classes"] = annotations[image_name]["classes"].tolist()
    
    json_io.save_json(annotations_path, save_annotations)


def get_num_annotations(annotations, region_keys):
    num_annotations = 0
    for image_name in annotations.keys():
        boxes = annotations[image_name]["boxes"]
        regions = []
        for region_key in region_keys:
            regions.extend(annotations[image_name][region_key])


        centres = (boxes[..., :2] + boxes[..., 2:]) / 2.0
        inds = poly_utils.get_contained_inds_for_points(centres, regions)

        num_annotations += inds.size

    return num_annotations



def get_num_fine_tuning_regions(annotations):
    num_fine_tuning_regions = 0
    for image_name in annotations.keys():
        num_fine_tuning_regions += len(annotations[image_name]["fine_tuning_regions"])

    return num_fine_tuning_regions



def get_patch_size(annotations, region_keys):

    average_box_area = get_average_box_area(annotations, region_keys=region_keys, measure="mean")
    patch_size = average_box_area_to_patch_size(average_box_area)
    return patch_size


def average_box_area_to_patch_size(average_box_area):
    patch_area = average_box_area * (90000 / 2296)
    patch_size = round(m.sqrt(patch_area))
    patch_size = max(416, patch_size)
    return patch_size 


def get_average_box_area(annotations, region_keys, measure):
    return get_average_box_dim("area", annotations, region_keys, measure)

def get_average_box_height(annotations, region_keys, measure):
    return get_average_box_dim("height", annotations, region_keys, measure)

def get_average_box_width(annotations, region_keys, measure):
    return get_average_box_dim("width", annotations, region_keys, measure)    



def get_average_box_dim(dim, annotations, region_keys, measure):
    
    box_dims = []

    for image_name in annotations.keys():
        boxes = annotations[image_name]["boxes"]
        regions = []
        if region_keys is not None:
            for region_key in region_keys:
                regions.extend(annotations[image_name][region_key])
                
            centres = (boxes[..., :2] + boxes[..., 2:]) / 2.0
            inds = poly_utils.get_contained_inds_for_points(centres, regions)
            region_boxes = boxes[inds]
        else:
            region_boxes = boxes
        if region_boxes.size > 0:
            
            if dim == "area":
                img_box_dims = ((region_boxes[:, 3] - region_boxes[:, 1]) * (region_boxes[:, 2] - region_boxes[:, 0])).tolist()
            elif dim == "height":
                img_box_dims = (region_boxes[:, 2] - region_boxes[:, 0]).tolist()
            elif dim == "width":
                img_box_dims = (region_boxes[:, 3] - region_boxes[:, 1]).tolist()

            box_dims.extend(img_box_dims)

    if len(box_dims) == 0:
        raise RuntimeError("Empty box list")

            
    if measure == "mean":
        return np.mean(box_dims)
    elif measure == "median":
        return np.median(box_dims)
    elif measure == "std":
        return np.std(box_dims)
    else:
        raise RuntimeError("Unknown measure")
