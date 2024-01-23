import math as m
import numpy as np

from models.common import box_utils

from io_utils import json_io


# def get_completed_images(annotations, allow_empty=True):
#     return [image_name for image_name in annotations.keys() \
#             if annotations[image_name]["status"] == "completed_for_training" or annotations[image_name]["status"] == "completed_for_testing" and (allow_empty or annotations[image_name]["boxes"].size > 0)]


def is_fully_annotated(annotations, image_name, image_w, image_h):
    # print("checking {}. w: {}, h: {}, training_regions: {}, test_regions: {}".format(image_name, image_w, image_h, annotations[image_name]["training_regions"], annotations[image_name]["test_regions"]))
    return is_fully_annotated_for_training(annotations, image_name, image_w, image_h) or is_fully_annotated_for_testing(annotations, image_name, image_w, image_h)


def is_fully_annotated_for_training(annotations, image_name, image_w, image_h):
    if len(annotations[image_name]["training_regions"]) == 0:
        return False
    region = annotations[image_name]["training_regions"][0]
    return (region[0] == 0 and region[1] == 0) and (region[2] == image_h and region[3] == image_w)

def is_fully_annotated_for_testing(annotations, image_name, image_w, image_h):
    if len(annotations[image_name]["test_regions"]) == 0:
        return False
    region = annotations[image_name]["test_regions"][0]
    return (region[0] == 0 and region[1] == 0) and (region[2] == image_h and region[3] == image_w)

    # for i in range(len(annotations[image_name]["test_regions"])):
    #     region = annotations[image_name]["test_regions"][i]
    #     if (region[0] == 0 and region[1] == 0) and (region[2] == image_h and region[3] == image_w):
    #         return True
    # return False

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


def get_num_annotations(annotations, region_keys): #image_names):
    num_annotations = 0
    for image_name in annotations.keys():
        boxes = annotations[image_name]["boxes"]
        # image_mask = np.full(boxes.shape[0], False)
        regions = []
        for region_key in region_keys:
            regions.extend(annotations[image_name][region_key])

        inds = box_utils.get_contained_inds(boxes, regions)
        num_annotations += inds.size
        #      for region in annotations[image_name][region_key]:

        #         region_mask = np.logical_and(
        #             np.logical_and(boxes[:,1] < region[3], boxes[:,3] > region[1]),
        #             np.logical_and(boxes[:,0] < region[2], boxes[:,2] > region[0])
        #         )

        #         image_mask = np.logical_or(image_mask, region_mask)

        # num_annotations += image_mask.sum()


                # inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], region)
                # num_annotations += inds.size
        #if annotations[image_name]["status"] == "completed_for_training" or not require_completed_for_training:
        # boxes = annotations[image_name]["boxes"]
        # num_annotations += np.shape(boxes)[0]
    return num_annotations



def get_num_training_regions(annotations):
    num_training_regions = 0
    for image_name in annotations.keys():
        num_training_regions += len(annotations[image_name]["training_regions"])

    return num_training_regions


# def typical_box_hyp_to_patch_size(typical_box_hyp):

#     # patch_size = round(typical_box_hyp * (300 / 67.88))
#     patch_hyp = (424.26 - (67.88 - typical_box_hyp))
#     patch_size = round(patch_hyp / m.sqrt(2))
#     return patch_size 


def get_patch_size(annotations, region_keys):


    average_box_area = get_average_box_area(annotations, region_keys=region_keys, measure="mean")
    #typical_box_hyp = get_typical_box_hyp(annotations, allowed_statuses=["completed_for_training"], measure="mean")
    #(40000 / 288) (90000 / 2296) 


    #slope = (90000 - 40000) / (2296 - 288)
    #patch_area = slope * (median_box_area - 288) + 40000
    patch_size = average_box_area_to_patch_size(average_box_area)
    #patch_size = typical_box_hyp_to_patch_size(typical_box_hyp)
    
    # print("patch_size", patch_size)
    return patch_size


# def get_typical_box_hyp(annotations, image_names, measure="mean"):
#     box_hyps = []
#     for image_name in image_names: #annotations.keys():
#         # if annotations[image_name]["status"] in allowed_statuses:
#         boxes = annotations[image_name]["boxes"]
#         if boxes.size > 0:
#             img_box_hyps = (np.sqrt((boxes[:, 3] - boxes[:, 1]) ** 2 + (boxes[:, 2] - boxes[:, 0]) ** 2)).tolist()
#             box_hyps.extend(img_box_hyps)

#     if len(box_hyps) == 0:
#         raise RuntimeError("No annotations found.") 

#     if measure == "mean":
#         return np.mean(box_hyps)
#     elif measure == "median":
#         return np.median(box_hyps)
#     else:
#         raise RuntimeError("Unknown measure")
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
        # regions = annotations[image_name]["training_regions"] + annotations[image_name]["test_regions"]
            inds = box_utils.get_contained_inds(boxes, regions)
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

        # for region_key in region_keys:
        #     for region in annotations[image_name][region_key]:

                
        #         inds = box_utils.get_contained_inds(boxes, region)
        #         if inds.size > 0:
        #             # boxes = annotations[image_name][inds]
        #             region_boxes = boxes[inds]
        #             if dim == "area":
        #                 img_box_dims = ((region_boxes[:, 3] - region_boxes[:, 1]) * (region_boxes[:, 2] - region_boxes[:, 0])).tolist()
        #             elif dim == "height":
        #                 img_box_dims = (region_boxes[:, 2] - region_boxes[:, 0]).tolist()
        #             elif dim == "width":
        #                 img_box_dims = (region_boxes[:, 3] - region_boxes[:, 1]).tolist()

        #             box_dims.extend(img_box_dims)

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

            


    # for image_name in image_names:
    #     # if annotations[image_name]["status"] in allowed_statuses:
    #     boxes = annotations[image_name]["boxes"]
    #     if boxes.size > 0:
    #         if dim == "area":
    #             img_box_dims = ((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])).tolist()
    #         elif dim == "height":
    #             img_box_dims = (boxes[:, 2] - boxes[:, 0]).tolist()
    #         elif dim == "width":
    #             img_box_dims = (boxes[:, 3] - boxes[:, 1]).tolist()
            
    #         box_dims.extend(img_box_dims)

    # if len(box_dims) == 0:
    #     raise RuntimeError("No annotations found.") 

    # if measure == "mean":
    #     return np.mean(box_dims)
    # elif measure == "median":
    #     return np.median(box_dims)
    # else:
    #     raise RuntimeError("Unknown measure")