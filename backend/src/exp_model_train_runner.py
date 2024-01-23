import logging
import os
import glob
import shutil
import random
import urllib3
import tqdm
import time
import uuid
import math as m
import numpy as np
import cv2
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.common import annotation_utils, box_utils
from io_utils import json_io
from image_set import Image
import image_utils


import server
from lock_queue import LockQueue

import image_set_actions as isa

import ds_splits


ns_training_image_sets = [
{'username': 'new_split', 'farm_name': 'train_Saskatoon', 'field_name': 'train_Norheim1', 'mission_date': '2021-06-02'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_Serhienko9S', 'mission_date': '2022-06-07'},
{'username': 'new_split', 'farm_name': 'train_Biggar', 'field_name': 'train_Dennis5', 'mission_date': '2021-06-12'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_Serhienko9S', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'train_row_spacing', 'field_name': 'train_brown', 'mission_date': '2021-06-01'},
{'username': 'new_split', 'farm_name': 'train_Davidson', 'field_name': 'train_Stone11NE', 'mission_date': '2022-06-03'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_Lake', 'mission_date': '2021-06-09'},
{'username': 'new_split', 'farm_name': 'train_UNI', 'field_name': 'train_Vaderstad', 'mission_date': '2022-06-16'},
{'username': 'new_split', 'farm_name': 'train_UNI', 'field_name': 'train_Sutherland', 'mission_date': '2021-06-05'},
{'username': 'new_split', 'farm_name': 'train_Biggar', 'field_name': 'train_Dennis3', 'mission_date': '2021-06-12'},
{'username': 'new_split', 'farm_name': 'train_Biggar', 'field_name': 'train_Dennis3', 'mission_date': '2021-06-04'},
{'username': 'new_split', 'farm_name': 'train_Saskatoon', 'field_name': 'train_Norheim4', 'mission_date': '2022-05-24'},
{'username': 'new_split', 'farm_name': 'train_Biggar', 'field_name': 'train_Dennis1', 'mission_date': '2021-06-04'},
{'username': 'new_split', 'farm_name': 'train_UNI', 'field_name': 'train_Brown', 'mission_date': '2021-06-05'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_Serhienko10', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_HornerWest', 'mission_date': '2021-06-09'},
{'username': 'new_split', 'farm_name': 'train_row_spacing', 'field_name': 'train_brown', 'mission_date': '2021-06-08'},
{'username': 'new_split', 'farm_name': 'train_Biggar', 'field_name': 'train_Dennis2', 'mission_date': '2021-06-12'},
{'username': 'new_split', 'farm_name': 'train_row_spacing', 'field_name': 'train_nasser', 'mission_date': '2021-06-01'},
{'username': 'new_split', 'farm_name': 'train_MORSE', 'field_name': 'train_Dugout', 'mission_date': '2022-05-27'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_Serhienko15', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'train_UNI', 'field_name': 'train_LowN1', 'mission_date': '2021-06-07'},
{'username': 'new_split', 'farm_name': 'train_BlaineLake', 'field_name': 'train_Serhienko11', 'mission_date': '2022-06-07'},
{'username': 'new_split', 'farm_name': 'train_MORSE', 'field_name': 'train_Nasser', 'mission_date': '2022-05-27'},
{'username': 'new_split', 'farm_name': 'train_UNI', 'field_name': 'train_Dugout', 'mission_date': '2022-05-30'},
{'username': 'new_split', 'farm_name': 'train_Saskatoon', 'field_name': 'train_Norheim5', 'mission_date': '2022-05-24'},
{'username': 'new_split', 'farm_name': 'train_row_spacing', 'field_name': 'train_nasser2', 'mission_date': '2022-06-02'}
]





training_image_sets = [
    {
        "username": "kaylie",
        "farm_name": "row_spacing",
        "field_name": "nasser",
        "mission_date": "2021-06-01"
    },
    {
        "username": "kaylie",
        "farm_name": "row_spacing",
        "field_name": "brown",
        "mission_date": "2021-06-01"
    },
    {
        "username": "kaylie",
        "farm_name": "UNI",
        "field_name": "Dugout",
        "mission_date": "2022-05-30"
    },
    {
        "username": "kaylie",
        "farm_name": "MORSE",
        "field_name": "Dugout",
        "mission_date": "2022-05-27"
    },
    {
        "username": "kaylie",
        "farm_name": "UNI",
        "field_name": "Brown",
        "mission_date": "2021-06-05"
    },
    {
        "username": "kaylie",
        "farm_name": "UNI",
        "field_name": "Sutherland",
        "mission_date": "2021-06-05"
    },
    {
        "username": "kaylie",
        "farm_name": "row_spacing",
        "field_name": "nasser2",
        "mission_date": "2022-06-02"
    },
    {
        "username": "kaylie",
        "farm_name": "MORSE",
        "field_name": "Nasser",
        "mission_date": "2022-05-27"
    },
    {
        "username": "kaylie",
        "farm_name": "UNI",
        "field_name": "LowN2",
        "mission_date": "2021-06-07"
    },
    {
        "username": "kaylie",
        "farm_name": "Saskatoon",
        "field_name": "Norheim4",
        "mission_date": "2022-05-24"
    },
    {
        "username": "kaylie",
        "farm_name": "Saskatoon",
        "field_name": "Norheim5",
        "mission_date": "2022-05-24"
    },
    {
        "username": "kaylie",
        "farm_name": "Saskatoon",
        "field_name": "Norheim1",
        "mission_date": "2021-05-26"
    },
    {
        "username": "kaylie",
        "farm_name": "Saskatoon",
        "field_name": "Norheim2",
        "mission_date": "2021-05-26"
    },
    {
        "username": "kaylie",
        "farm_name": "Biggar",
        "field_name": "Dennis1",
        "mission_date": "2021-06-04"
    },
    {
        "username": "kaylie",
        "farm_name": "Biggar",
        "field_name": "Dennis3",
        "mission_date": "2021-06-04"
    },
    {
        "username": "kaylie",
        "farm_name": "BlaineLake",
        "field_name": "River",
        "mission_date": "2021-06-09"
    },
    {
        "username": "kaylie",
        "farm_name": "BlaineLake",
        "field_name": "Lake",
        "mission_date": "2021-06-09"
    },
    {
        "username": "kaylie",
        "farm_name": "BlaineLake",
        "field_name": "HornerWest",
        "mission_date": "2021-06-09"
    },
    {
        "username": "kaylie",
        "farm_name": "UNI",
        "field_name": "LowN1",
        "mission_date": "2021-06-07"
    },
    {
        "username": "kaylie",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9N",
        "mission_date": "2022-06-07"
    },
    {
        "username": "kaylie",
        "farm_name": "Saskatoon",
        "field_name": "Norheim1",
        "mission_date": "2021-06-02"
    },
    {
        "username": "kaylie",
        "farm_name": "row_spacing",
        "field_name": "brown",
        "mission_date": "2021-06-08"
    },
    {
        "username": "kaylie",
        "farm_name": "SaskatoonEast",
        "field_name": "Stevenson5NW",
        "mission_date": "2022-06-20"
    },
    {
        "username": "kaylie",
        "farm_name": "UNI",
        "field_name": "Vaderstad",
        "mission_date": "2022-06-16"
    },
    {
        "username": "kaylie",
        "farm_name": "Biggar",
        "field_name": "Dennis2",
        "mission_date": "2021-06-12"
    },
    {
        "username": "kaylie",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko10",
        "mission_date": "2022-06-14"
    },
    {
        "username": "kaylie",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-14"
    }
]

test_image_sets = [
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko11",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko15",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "SaskatoonEast",
        "field_name": "Stevenson5SW",
        "mission_date": "2022-06-13"
    },
    {
        "username": "eval",
        "farm_name": "Davidson",
        "field_name": "Stone11NE",
        "mission_date": "2022-06-03"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko12",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis3",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis5",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "CNH-DugoutROW",
        "mission_date": "2022-05-30"
    },
]



def get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent):
    patch_overlap_percent = 0
    overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

    image_names = list(annotations.keys())
    image_width = metadata["images"][image_names[0]]["width_px"]
    image_height = metadata["images"][image_names[0]]["height_px"]

    incr = patch_size - overlap_px
    w_covered = max(image_width - patch_size, 0)
    num_w_patches = m.ceil(w_covered / incr) + 1

    h_covered = max(image_height - patch_size, 0)
    num_h_patches = m.ceil(h_covered / incr) + 1

    num_patches = num_w_patches * num_h_patches

    return num_patches



def get_num_fully_annotated_images(annotations, image_w, image_h):
    num = 0
    for image_name in annotations.keys():
        if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
            num += 1
    return num








def get_num_patches_used_by_model(model_dir):

    log_path = os.path.join(model_dir, "log.json")
    log = json_io.load_json(log_path)
    total_num_patches = 0
    for image_set in log["image_sets"]:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
        if "patch_overlap_percent" in image_set:
            patch_overlap_percent = image_set["patch_overlap_percent"]
        else:
            patch_overlap_percent = 50

        if "taken_regions" in image_set:
            for image_name in image_set["taken_regions"].keys():
                for region in image_set["taken_regions"][image_name]:

                    region_width = region[3] - region[1]
                    region_height = region[2] - region[0]

                    overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

                    incr = patch_size - overlap_px
                    w_covered = max(region_width - patch_size, 0)
                    num_w_patches = m.ceil(w_covered / incr) + 1

                    h_covered = max(region_height - patch_size, 0)
                    num_h_patches = m.ceil(h_covered / incr) + 1

                    num_patches = num_w_patches * num_h_patches

                    total_num_patches += num_patches
        else:
            
            num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=patch_overlap_percent)
            
            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    total_num_patches += num_patches_per_image


    return total_num_patches
                    


def restore_annotations(training_image_sets):

    for image_set in training_image_sets:
        image_set_dir = os.path.join("usr", "data",
                            image_set["username"], "image_sets",
                            image_set["farm_name"],
                            image_set["field_name"],
                            image_set["mission_date"])
        preserved_annotations_path = os.path.join(image_set_dir, "annotations", "preserved_annotations_no_modification.json")
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        if os.path.exists(preserved_annotations_path):
            shutil.copy(preserved_annotations_path, annotations_path)



def annotation_removal_test(training_image_sets, fraction_to_remove, num_patches_to_take, prev_model_dir):

    model_name = "set_of_27_remove_" + str(fraction_to_remove) + "_" + str(num_patches_to_take) + "_patches_rep_0"
    existing_model_log_path = os.path.join(prev_model_dir, "log.json")
    existing_model_log = json_io.load_json(existing_model_log_path)

    # for i, image_set in enumerate(existing_model_log["image_sets"]):
    #     image_set_dir = os.path.join("usr", "data",
    #                                     image_set["username"], "image_sets",
    #                                     image_set["farm_name"],
    #                                     image_set["field_name"],
    #                                     image_set["mission_date"]) 
    #     annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    #     annotations = annotation_utils.load_annotations(annotations_path)

    #     average_box_area = annotation_utils.get_average_box_area(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
    #     patch_size = annotation_utils.average_box_area_to_patch_size(average_box_area)

    #     existing_model_log["image_sets"][i]["patch_size"] = patch_size


    for image_set in training_image_sets:

        image_set_dir = os.path.join("usr", "data",
                                    image_set["username"], "image_sets",
                                    image_set["farm_name"],
                                    image_set["field_name"],
                                    image_set["mission_date"])
        print("\tRemoving", image_set_dir)
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        preserved_annotations_path = os.path.join(image_set_dir, "annotations", "preserved_annotations_no_modification.json")
        shutil.copy(annotations_path, preserved_annotations_path)

        # metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        # metadata = json_io.load_json(metadata_path)

        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():

            boxes = annotations[image_name]["boxes"]
            if boxes.size > 0:
                keep_mask = np.random.rand(boxes.shape[0]) > fraction_to_remove
                kept_boxes = boxes[keep_mask]
                annotations[image_name]["boxes"] = kept_boxes

        annotation_utils.save_annotations(annotations_path, annotations)
        removed_annotations_path = os.path.join(image_set_dir, "annotations", "removed_annotations_" + str(fraction_to_remove) + ".json")
        shutil.copy(annotations_path, removed_annotations_path)

    # run_diverse_model(training_image_sets, 
    #                   "set_of_27_remove_" + str(fraction_to_remove) + "_" + str(num_patches_to_take) + "_patches_rep_0", 
    #                   num_patches_to_take, 
    #                   prev_model_dir, 
    #                   supplementary_weed_image_sets=None, 
    #                   run=True)
    

    run_from_existing_diverse_model(model_name, existing_model_log, run=True)


    for image_set in training_image_sets:
        image_set_dir = os.path.join("usr", "data",
                                    image_set["username"], "image_sets",
                                    image_set["farm_name"],
                                    image_set["field_name"],
                                    image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        preserved_annotations_path = os.path.join(image_set_dir, "annotations", "preserved_annotations_no_modification.json")
        shutil.copy(preserved_annotations_path, annotations_path)


def annotation_dilation_test(training_image_sets, dilation_sigmas, num_patches_to_take, prev_model_dir, rep_num): #taken_regions): #num_patches_to_take):


    # for i, image_set in enumerate(existing_model_log["image_sets"]):
    #     image_set_dir = os.path.join("usr", "data",
    #                                     image_set["username"], "image_sets",
    #                                     image_set["farm_name"],
    #                                     image_set["field_name"],
    #                                     image_set["mission_date"]) 
    #     annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    #     annotations = annotation_utils.load_annotations(annotations_path)

    #     average_box_area = annotation_utils.get_average_box_area(annotations, region_keys=["training_regions", "test_regions"], measure="mean")
    #     patch_size = annotation_utils.average_box_area_to_patch_size(average_box_area)

    #     existing_model_log["image_sets"][i]["patch_size"] = patch_size

    for dilation_sigma in dilation_sigmas:


        model_name = "set_of_27_uniformly_dilated_by_" + str(dilation_sigma) + "_" + str(num_patches_to_take) + "_patches_rep_" + str(rep_num)
        existing_model_log_path = os.path.join(prev_model_dir, "log.json")
        existing_model_log = json_io.load_json(existing_model_log_path)


        print("Dilation sigma: {}".format(dilation_sigma))
        for image_set in training_image_sets:

            image_set_dir = os.path.join("usr", "data",
                                        image_set["username"], "image_sets",
                                        image_set["farm_name"],
                                        image_set["field_name"],
                                        image_set["mission_date"])
            print("\tDilating", image_set_dir)
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            preserved_annotations_path = os.path.join(image_set_dir, "annotations", "preserved_annotations_no_modification.json")
            shutil.copy(annotations_path, preserved_annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            annotations = annotation_utils.load_annotations(annotations_path)
            for image_name in annotations.keys():
                image_h = metadata["images"][image_name]["height_px"]
                image_w = metadata["images"][image_name]["width_px"]
                for box in annotations[image_name]["boxes"]:
                    min_y_dilation = abs(round(random.uniform(0, dilation_sigma)))
                    min_x_dilation = abs(round(random.uniform(0, dilation_sigma)))
                    max_y_dilation = abs(round(random.uniform(0, dilation_sigma)))
                    max_x_dilation = abs(round(random.uniform(0, dilation_sigma)))

                    # min_y_dilation = abs(round(random.gauss(0, dilation_sigma)))
                    # min_x_dilation = abs(round(random.gauss(0, dilation_sigma)))
                    # max_y_dilation = abs(round(random.gauss(0, dilation_sigma)))
                    # max_x_dilation = abs(round(random.gauss(0, dilation_sigma)))


                    box[0] = max(0, box[0] - min_y_dilation) # min(image_h - 1, max(0, p_min_y)) #box[0] + random.randint(-(1) * perturbation_amount, perturbation_amount)))
                    box[1] = max(0, box[1] - min_x_dilation) #min(image_w - 1, max(0, p_min_x)) #box[1] + random.randint(-(1) * perturbation_amount, perturbation_amount)))
                    box[2] = min(image_h, box[2] + max_y_dilation) #max(box[0] + 1, min(image_h, p_max_y)) #box[2] + random.randint(-(1) * perturbation_amount, perturbation_amount)))
                    box[3] = min(image_w, box[3] + max_x_dilation) #max(box[1] + 1, min(image_w, p_max_x)) #box[3] + random.randint(-(1) * perturbation_amount, perturbation_amount)))


            annotation_utils.save_annotations(annotations_path, annotations)
            dilated_annotations_path = os.path.join(image_set_dir, "annotations", "dilated_annotations_" + str(dilation_sigma) + ".json")
            shutil.copy(annotations_path, dilated_annotations_path)

        # run_diverse_model(training_image_sets, "set_of_27_dilated_by_" + str(dilation_sigma) + "_" + str(num_patches_to_take) + "_patches_rep_0", num_patches_to_take, prev_model_dir=None)


        run_from_existing_diverse_model(model_name, existing_model_log, run=True)
        # run_diverse_model(training_image_sets, "set_of_27_dilated_by_" + str(dilation_sigma) + "_" + str(num_patches_to_take) + "_patches_rep_0", num_patches_to_take, prev_model_dir, supplementary_weed_image_sets=None, run=True)

        for image_set in training_image_sets:
            image_set_dir = os.path.join("usr", "data",
                                        image_set["username"], "image_sets",
                                        image_set["farm_name"],
                                        image_set["field_name"],
                                        image_set["mission_date"])

            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            preserved_annotations_path = os.path.join(image_set_dir, "annotations", "preserved_annotations_no_modification.json")
            shutil.copy(preserved_annotations_path, annotations_path)


def run_single_image_set_diverse_test(training_image_set, all_training_image_sets, num_patches_to_take, supplementary_weed_image_sets=None):

    patch_nums = get_patch_nums_of_sets(all_training_image_sets)

    training_set_str = training_image_set["username"] + " " + training_image_set["farm_name"] + " " + training_image_set["field_name"] + " " + training_image_set["mission_date"]
    model_name = training_image_set["farm_name"] + "_" + training_image_set["field_name"] + "_" + training_image_set["mission_date"]
    
    if patch_nums[training_set_str] < num_patches_to_take:
        print("Not training {} because there are not enough patches to take.".format(training_set_str))

    else:
        print("{} has enough patches (need {}, have {}). Training model.".format(training_set_str, num_patches_to_take, patch_nums[training_set_str]))
        run_diverse_model([training_image_set], model_name + "_" + str(num_patches_to_take) + "_patches_and_CottonWeedDet12_rep_1", num_patches_to_take, supplementary_weed_image_sets=supplementary_weed_image_sets, prev_model_dir=None)


    # full_model_name = "fixed_epoch_" + model_name + "_no_overlap"

    # # run_full_image_set_model([training_image_set], full_model_name)
    # run_diverse_model([training_image_set], "fixed_epoch_min_num_diverse_set_of_1_match_" + model_name + "_no_overlap", min_num, prev_model_dir=None)
    # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "available", "public", full_model_name)
    # num_patches_to_match = get_num_patches_used_by_model(model_dir_to_match)
    # run_diverse_model(all_training_image_sets, "fixed_epoch_diverse_set_of_27_match_" + model_name + "_no_overlap", num_patches_to_match, prev_model_dir=None)

def exg_zero_anno_replacement(training_image_sets, model_name, model_dir_to_match, run=True):

    match_log_path = os.path.join(model_dir_to_match, "log.json")
    match_log = json_io.load_json(match_log_path)

    zero_anno_candidates = []


    for image_set in tqdm.tqdm(training_image_sets, desc="Getting zero anno candidates"):
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
        image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                     farm_name, field_name, mission_date)
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        
        
        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        for image_name in annotations.keys():
                
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                image_w = metadata["images"][image_name]["width_px"]
                image_h = metadata["images"][image_name]["height_px"]

                image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
                image = Image(image_path)
                image_array = image.load_image_array()
                exg_array = image_utils.excess_green(image_array)



                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_coords])

                        if inds.size == 0:
                            exg_patch = exg_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
                            # sel_vals = exg_patch[exg_patch != -10]
                            score = np.sum(exg_patch ** 2) / exg_patch.size

                            zero_anno_candidates.append(
                                (image_set_str, image_name, patch_coords, score)
                            )
    print("number of zero anno candidates: {}".format(len(zero_anno_candidates)))
    zero_anno_candidates.sort(key=lambda x: x[3], reverse=True)
    print("10 best zero_anno_candidates: {}".format(zero_anno_candidates[:10]))
    print("10 worst zero anno candidates: {}".format(zero_anno_candidates[-10:]))
    zero_candidate_index = 0

    taken = {}
    for image_set in match_log["image_sets"]:

        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
        image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                     farm_name, field_name, mission_date)
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        for image_name in image_set["taken_regions"]:

            for patch_coords in image_set["taken_regions"][image_name]:

                inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_coords])
                if inds.size > 0:
                    if image_set_str not in taken:
                        taken[image_set_str] = {}
                    if image_name not in taken[image_set_str]:
                        taken[image_set_str][image_name] = []

                    taken[image_set_str][image_name].append(patch_coords)

                else:
                    c = zero_anno_candidates[zero_candidate_index]
                    zero_candidate_index += 1
                    repl_image_set_str = c[0]
                    repl_image_name = c[1]
                    repl_patch_coords = c[2]

                    if repl_image_set_str not in taken:
                        taken[repl_image_set_str] = {}
                    if repl_image_name not in taken[repl_image_set_str]:
                        taken[repl_image_set_str][repl_image_name] = []

                    taken[repl_image_set_str][repl_image_name].append(repl_patch_coords)
        
                    

    log = {}
    log["model_creator"] = "eval"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for image_set_str in taken:

        pieces = image_set_str.split(" ")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": taken[image_set_str],
            "patch_overlap_percent": 0
        })

    
    # if supplementary_weed_image_sets is not None:
    #     for image_set in supplementary_weed_image_sets:
    #         image_set["patch_overlap_percent"] = 0
    #         log["image_sets"].append(image_set)

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "eval", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()



def run_from_existing_diverse_model(model_name, existing_model_log, run=True):


    log = {}
    log["model_creator"] = "eval"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    # for key in s:
    #     pieces = key.split("/")
    #     log["image_sets"].append({
    #         "username": pieces[0],
    #         "farm_name": pieces[1],
    #         "field_name": pieces[2],
    #         "mission_date": pieces[3],
    #         "taken_regions": s[key],
    #         "patch_overlap_percent": 0
    #     })
    log["image_sets"] = existing_model_log["image_sets"]

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "eval", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()



def run_diverse_multi(training_image_sets, num_training_image_sets_per_model, num_patches, run=True):

    
    # num_reps = 10
    for rep_num in range(10, 20): #num_reps):

        s = {}

        model_name = "set_of_" + str(num_training_image_sets_per_model) + "_" + str(num_patches) + "_patches_rep_" + str(rep_num)

        model_training_image_sets = random.sample(training_image_sets, num_training_image_sets_per_model)

        candidates = []
        for image_set in model_training_image_sets:

            key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]


            image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                        image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            image_names = list(annotations.keys())
            image_w = metadata["images"][image_names[0]]["width_px"]
            image_h = metadata["images"][image_names[0]]["height_px"]

            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

                
            patch_overlap_percent = 0
            overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

            incr = patch_size - overlap_px
            w_covered = max(image_w - patch_size, 0)
            num_w_patches = m.ceil(w_covered / incr) + 1

            h_covered = max(image_h - patch_size, 0)
            num_h_patches = m.ceil(h_covered / incr) + 1

            for image_name in annotations.keys():
                image_w = metadata["images"][image_name]["width_px"]
                image_h = metadata["images"][image_name]["height_px"]
                if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                    for h_index in range(num_h_patches):
                        for w_index in range(num_w_patches):

                            patch_coords = [
                                patch_size * h_index,
                                patch_size * w_index,
                                min((patch_size * h_index) + patch_size, image_h),
                                min((patch_size * w_index) + patch_size, image_w)
                            ]

                            if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
                                pass
                            else:
                                candidates.append((key, image_name, patch_coords))

                    
        taken_candidates = random.sample(candidates, num_patches)

        for taken_candidate in taken_candidates:
            key = taken_candidate[0]
            if key not in s:
                s[key] = {}

            image_name = taken_candidate[1]
            if image_name not in s[key]:
                s[key][image_name] = []

            s[key][image_name].append(taken_candidate[2])


        log = {}
        log["model_creator"] = "eval"
        log["model_name"] = model_name
        log["model_object"] = "canola_seedling"
        log["public"] = "yes"
        log["image_sets"] = []

        for key in s:
            pieces = key.split("/")
            log["image_sets"].append({
                "username": pieces[0],
                "farm_name": pieces[1],
                "field_name": pieces[2],
                "mission_date": pieces[3],
                "taken_regions": s[key],
                "patch_overlap_percent": 0
            })

        log["submission_time"] = int(time.time())
        
        pending_model_path = os.path.join("usr", "data", "eval", "models", "pending", log["model_name"])

        os.makedirs(pending_model_path, exist_ok=False)
        
        log_path = os.path.join(pending_model_path, "log.json")
        json_io.save_json(log_path, log)

        if run:
            server.sch_ctx["baseline_queue"].enqueue(log)

            baseline_queue_size = server.sch_ctx["baseline_queue"].size()
            while baseline_queue_size > 0:
            
                log = server.sch_ctx["baseline_queue"].dequeue()
                re_enqueue = server.process_baseline(log)
                if re_enqueue:
                    server.sch_ctx["baseline_queue"].enqueue(log)
                baseline_queue_size = server.sch_ctx["baseline_queue"].size()

        


def run_ds_model(training_image_sets, model_name, num_patches_to_take, run=False):

    # num_patches_to_take = num_patches_to_match - num_patches_prev_taken

    # print("Num patches to match: {}".format(num_patches_to_match))
    # print("Num patches previously taken: {}".format(num_patches_prev_taken))
    # print("Num patches to take: {}".format(num_patches_to_take))
    s = {}

    candidates = []
    for image_set in training_image_sets:

        key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]


        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = 416 # annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        for image_name in annotations.keys():
            image_w = metadata["images"][image_name]["width_px"]
            image_h = metadata["images"][image_name]["height_px"]
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
                            pass
                        else:
                            candidates.append((key, image_name, patch_coords))

    print("taking {} patches from {} candidates".format(num_patches_to_take, len(candidates)))
    taken_candidates = random.sample(candidates, num_patches_to_take)

    for taken_candidate in taken_candidates:
        key = taken_candidate[0]
        if key not in s:
            s[key] = {}

        image_name = taken_candidate[1]
        if image_name not in s[key]:
            s[key][image_name] = []

        s[key][image_name].append(taken_candidate[2])

    log = {}
    log["model_creator"] = "ds_splits"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for key in s:
        pieces = key.split("/")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": s[key],
            "patch_overlap_percent": 0,
            "patch_size": 416
        })

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "ds_splits", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()





def run_diverse_model(training_image_sets, model_name, num_patches_to_match, prev_model_dir, supplementary_weed_image_sets=None, run=True):

    s = {}

    if prev_model_dir is not None:
        num_patches_prev_taken = get_num_patches_used_by_model(prev_model_dir)
        prev_log_path = os.path.join(prev_model_dir, "log.json")
        prev_log = json_io.load_json(prev_log_path)
        for image_set in prev_log["image_sets"]:
            key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]
            if key not in s:
                s[key] = {}

            s[key] = image_set["taken_regions"]


    else:
        num_patches_prev_taken = 0

    # num_patches_to_match = get_num_patches_used_by_model(model_dir_to_match)

    num_patches_to_take = num_patches_to_match - num_patches_prev_taken

    print("Num patches to match: {}".format(num_patches_to_match))
    print("Num patches previously taken: {}".format(num_patches_prev_taken))
    print("Num patches to take: {}".format(num_patches_to_take))


    candidates = []
    for image_set in training_image_sets:

        key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]


        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        for image_name in annotations.keys():
            image_w = metadata["images"][image_name]["width_px"]
            image_h = metadata["images"][image_name]["height_px"]
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
                            pass
                        else:
                            candidates.append((key, image_name, patch_coords))

                
    taken_candidates = random.sample(candidates, num_patches_to_take)

    for taken_candidate in taken_candidates:
        key = taken_candidate[0]
        if key not in s:
            s[key] = {}

        image_name = taken_candidate[1]
        if image_name not in s[key]:
            s[key][image_name] = []

        s[key][image_name].append(taken_candidate[2])

    log = {}
    log["model_creator"] = "new_split"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for key in s:
        pieces = key.split("/")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": s[key],
            "patch_overlap_percent": 0
        })

    
    if supplementary_weed_image_sets is not None:
        for image_set in supplementary_weed_image_sets:
            image_set["patch_overlap_percent"] = 0
            log["image_sets"].append(image_set)

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "new_split", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()













def run_ordered_selection_model(training_image_sets, model_name, num_patches_to_match, prev_model_dir, image_set_ordering, supplementary_weed_image_sets=None, run=True):

    s = {}

    if prev_model_dir is not None:
        num_patches_prev_taken = get_num_patches_used_by_model(prev_model_dir)
        prev_log_path = os.path.join(prev_model_dir, "log.json")
        prev_log = json_io.load_json(prev_log_path)
        for image_set in prev_log["image_sets"]:
            key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]
            if key not in s:
                s[key] = {}

            s[key] = image_set["taken_regions"]


    else:
        num_patches_prev_taken = 0

    # num_patches_to_match = get_num_patches_used_by_model(model_dir_to_match)

    num_patches_to_take = num_patches_to_match - num_patches_prev_taken

    print("Num patches to match: {}".format(num_patches_to_match))
    print("Num patches previously taken: {}".format(num_patches_prev_taken))
    print("Num patches to take: {}".format(num_patches_to_take))


    # candidates = []
    candidates = {}
    for image_set in training_image_sets:

        key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]
        candidates[key] = []

        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        for image_name in annotations.keys():
            image_w = metadata["images"][image_name]["width_px"]
            image_h = metadata["images"][image_name]["height_px"]
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
                            pass
                        else:
                            # candidates.append((key, image_name, patch_coords))
                            candidates[key].append((key, image_name, patch_coords))
    # num_newly_taken = 0
    cur_index = 0
    taken_candidates = []
    while num_patches_to_take > 0: #num_newly_taken < num_patches_to_take:
        cur_image_set = training_image_sets[image_set_ordering[cur_index]]
        key = cur_image_set["username"] + "/" + cur_image_set["farm_name"] + "/" + cur_image_set["field_name"] + "/" + cur_image_set["mission_date"]
        print("   ", key)
        image_set_taken_candidates = random.sample(candidates[key], min(num_patches_to_take, len(candidates[key])))
        taken_candidates.extend(image_set_taken_candidates)
        num_patches_to_take -= len(image_set_taken_candidates)
        print("took {} patches".format(len(image_set_taken_candidates)))
        cur_index += 1

                
    # taken_candidates = random.sample(candidates, num_patches_to_take)

    for taken_candidate in taken_candidates:
        key = taken_candidate[0]
        if key not in s:
            s[key] = {}

        image_name = taken_candidate[1]
        if image_name not in s[key]:
            s[key][image_name] = []

        s[key][image_name].append(taken_candidate[2])

    log = {}
    log["model_creator"] = "eval"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for key in s:
        pieces = key.split("/")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": s[key],
            "patch_overlap_percent": 0
        })

    
    if supplementary_weed_image_sets is not None:
        for image_set in supplementary_weed_image_sets:
            image_set["patch_overlap_percent"] = 0
            log["image_sets"].append(image_set)

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "eval", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()











def run_smaller_diverse_model(training_image_sets, model_name, num_patches_to_match, next_model_dir, supplementary_weed_image_sets=None, run=True):

    # s = {}

    all_next_taken_candidates = []

    if next_model_dir is not None:
        num_patches_next_taken = get_num_patches_used_by_model(next_model_dir)
        next_log_path = os.path.join(next_model_dir, "log.json")
        next_log = json_io.load_json(next_log_path)
        for image_set in next_log["image_sets"]:
            key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]
            # if key not in s:
            #     s[key] = {}

            # s[key] = image_set["taken_regions"]
            for image_name in image_set["taken_regions"].keys():
                for r in image_set["taken_regions"][image_name]:
                    all_next_taken_candidates.append([key, image_name, r])


    else:
        num_patches_next_taken = 0

    # num_patches_to_match = get_num_patches_used_by_model(model_dir_to_match)

    num_patches_to_remove = num_patches_next_taken - num_patches_to_match

    print("Num patches to match: {}".format(num_patches_to_match))
    print("Num patches next taken: {}".format(num_patches_next_taken))
    print("Num patches to remove: {}".format(num_patches_to_remove))

    all_next_taken_candidates = np.array(all_next_taken_candidates)
    indices = np.random.choice(np.arange(all_next_taken_candidates.shape[0]), replace=False,
                           size=num_patches_to_match) #int(myarray.size * 0.2))
    
    print("indices.size", indices.size)
    
    keep_mask = np.full(all_next_taken_candidates.shape[0], False)
    keep_mask[indices] = True
    
    taken_candidates = all_next_taken_candidates[indices]

    # for candidate in keep_candidates:





    # candidates = []
    # for image_set in training_image_sets:

    #     key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]


    #     image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
    #                                  image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

    #     annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    #     annotations = annotation_utils.load_annotations(annotations_path)

    #     metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    #     metadata = json_io.load_json(metadata_path)

    #     image_names = list(annotations.keys())
    #     image_w = metadata["images"][image_names[0]]["width_px"]
    #     image_h = metadata["images"][image_names[0]]["height_px"]

    #     patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
    #     patch_overlap_percent = 0
    #     overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

    #     incr = patch_size - overlap_px
    #     w_covered = max(image_w - patch_size, 0)
    #     num_w_patches = m.ceil(w_covered / incr) + 1

    #     h_covered = max(image_h - patch_size, 0)
    #     num_h_patches = m.ceil(h_covered / incr) + 1

    #     for image_name in annotations.keys():
    #         image_w = metadata["images"][image_name]["width_px"]
    #         image_h = metadata["images"][image_name]["height_px"]
    #         if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
    #             for h_index in range(num_h_patches):
    #                 for w_index in range(num_w_patches):

    #                     patch_coords = [
    #                         patch_size * h_index,
    #                         patch_size * w_index,
    #                         min((patch_size * h_index) + patch_size, image_h),
    #                         min((patch_size * w_index) + patch_size, image_w)
    #                     ]

    #                     if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
    #                         pass
    #                     else:
    #                         candidates.append((key, image_name, patch_coords))

                
    # taken_candidates = random.sample(candidates, num_patches_to_take)
    s = {}
    for taken_candidate in taken_candidates:
        key = taken_candidate[0]
        if key not in s:
            s[key] = {}

        image_name = taken_candidate[1]
        if image_name not in s[key]:
            s[key][image_name] = []

        s[key][image_name].append(taken_candidate[2])

    log = {}
    log["model_creator"] = "eval"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for key in s:
        pieces = key.split("/")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": s[key],
            "patch_overlap_percent": 0
        })

    
    if supplementary_weed_image_sets is not None:
        for image_set in supplementary_weed_image_sets:
            image_set["patch_overlap_percent"] = 0
            log["image_sets"].append(image_set)

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "eval", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()
















def run_diverse_model_2(training_image_sets, model_name, patch_sample_nums, run=True):

    s = {}

    # if prev_model_dir is not None:
    #     num_patches_prev_taken = get_num_patches_used_by_model(prev_model_dir)
    #     prev_log_path = os.path.join(prev_model_dir, "log.json")
    #     prev_log = json_io.load_json(prev_log_path)
    #     for image_set in prev_log["image_sets"]:
    #         key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]
    #         if key not in s:
    #             s[key] = {}

    #         s[key] = image_set["taken_regions"]


    # else:
    #     num_patches_prev_taken = 0

    # num_patches_to_match = get_num_patches_used_by_model(model_dir_to_match)

    # num_patches_to_take = num_patches_to_match - num_patches_prev_taken

    # print("Num patches to match: {}".format(num_patches_to_match))
    # print("Num patches previously taken: {}".format(num_patches_prev_taken))
    # print("Num patches to take: {}".format(num_patches_to_take))


    # candidates = {} #[]
    for i, image_set in enumerate(training_image_sets):
        candidates = []

        key = image_set["username"] + "/" + image_set["farm_name"] + "/" + image_set["field_name"] + "/" + image_set["mission_date"]


        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        for image_name in annotations.keys():
            image_w = metadata["images"][image_name]["width_px"]
            image_h = metadata["images"][image_name]["height_px"]
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]

                        # if key in s and image_name in s[key] and patch_coords in s[key][image_name]:
                        #     pass
                        # else:
                        candidates.append((image_name, patch_coords))

        taken_candidates = random.sample(candidates, patch_sample_nums[i])

        s[key] = {}
        for taken_candidate in taken_candidates:

            image_name = taken_candidate[0]
            if image_name not in s[key]:
                s[key][image_name] = []

            s[key][image_name].append(taken_candidate[1])



    log = {}
    log["model_creator"] = "eval"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["image_sets"] = []

    for key in s:
        pieces = key.split("/")
        log["image_sets"].append({
            "username": pieces[0],
            "farm_name": pieces[1],
            "field_name": pieces[2],
            "mission_date": pieces[3],
            "taken_regions": s[key],
            "patch_overlap_percent": 0
        })

    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "eval", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    if run:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()












def run_pending_model(model_dir):

    log_path = os.path.join(model_dir, "log.json")
    log = json_io.load_json(log_path)
    log["submission_time"] = int(time.time())
    json_io.save_json(log_path, log)


    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()


def run_full_image_set_model(training_image_sets, model_name):
    log = {}
    log["model_creator"] = "erik"
    # log["model_name"] = str(num_images_to_select) ##"set_of_" + str(len(org_image_sets)) + "_no_overlap"
    log["model_name"] = model_name #"random_images_" + str(iteration_number) #cur_num)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    # log["image_sets"] = image_sets
    log["image_sets"] = []

    for image_set in training_image_sets:
        image_set["patch_overlap_percent"] = 0
        log["image_sets"].append(image_set)


    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)


    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()





def run_weed_test(training_image_sets, weed_image_sets, model_name, num_weed_patches_to_add):

    log = {}
    log["model_creator"] = "erik"
    # log["model_name"] = str(num_images_to_select) ##"set_of_" + str(len(org_image_sets)) + "_no_overlap"
    log["model_name"] = model_name #"random_images_" + str(iteration_number) #cur_num)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    # log["image_sets"] = image_sets
    log["image_sets"] = []
    # for image_set in training_image_sets:
    #     image_set["patch_overlap_percent"] = 0
    #     log["image_sets"].append(image_set)
    


    for image_set in weed_image_sets:
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])
        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = 416 #annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        num_annotated = get_num_fully_annotated_images(annotations, image_w, image_h)

        num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)

        image_set["num_patches"] = num_patches_per_image * num_annotated


    capacities = []
    for image_set in weed_image_sets:
        capacities.append(image_set["num_patches"])
        # diverse_image_sets_patch_num += image_set["num_patches"]

    capacities = np.array(capacities)
    org_capacities = np.copy(capacities)
    num_to_match = num_weed_patches_to_add
    print("num_to_match", num_to_match)
    print("total capacity", np.sum(capacities))
    print("capacities", capacities)

    satisfied = False
    while not satisfied:

        min_capacity_this_iteration = min(capacities[capacities != 0])
        desired_num_to_take_this_iteration = num_to_match // np.sum(capacities != 0)

        num_actually_taken_this_iteration = min(min_capacity_this_iteration, desired_num_to_take_this_iteration)
        for i in range(capacities.size):
            if capacities[i] > 0:
                capacities[i] -= num_actually_taken_this_iteration
                num_to_match -= num_actually_taken_this_iteration
 
        if min_capacity_this_iteration >= desired_num_to_take_this_iteration:
            satisfied = True

    i = 0
    while num_to_match > 0:
        if capacities[i] > 0:
            capacities[i] -= 1
            num_to_match -= 1
        i += 1

    taken = org_capacities - capacities
    print("Taken: {}".format(taken))

    for i, image_set in enumerate(weed_image_sets):
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets",
                                     image_set["farm_name"], image_set["field_name"], image_set["mission_date"])

        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        image_names = list(annotations.keys())
        image_w = metadata["images"][image_names[0]]["width_px"]
        image_h = metadata["images"][image_names[0]]["height_px"]

        patch_size = 416 #annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

            
        patch_overlap_percent = 0
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(image_w - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(image_h - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        # num_patches = num_w_patches * num_h_patches

        annotated_image_names = []
        for image_name in annotations.keys():
            if annotation_utils.is_fully_annotated(annotations, image_name, image_w, image_h):
                annotated_image_names.append(image_name)
        
        num_taken = 0
        taken_regions = {}
        while num_taken < taken[i]:
            
            image_name = random.sample(annotated_image_names, 1)[0]
            
            w_index = random.randrange(0, num_w_patches)
            h_index = random.randrange(0, num_h_patches)

            patch_coords = [
                patch_size * h_index,
                patch_size * w_index,
                min((patch_size * h_index) + patch_size, image_h),
                min((patch_size * w_index) + patch_size, image_w)
            ]

            if image_name not in taken_regions:
                taken_regions[image_name] = []
                taken_regions[image_name].append(patch_coords)
                num_taken += 1
            elif patch_coords not in taken_regions[image_name]:
                taken_regions[image_name].append(patch_coords)
                num_taken += 1

        image_set["taken_regions"] = taken_regions
        image_set["patch_overlap_percent"] = 0

    for image_set in training_image_sets:
        image_set["patch_overlap_percent"] = 0

    log["image_sets"].extend(training_image_sets)
    log["image_sets"].extend(weed_image_sets)



    # for item in [item2]: #[item1, item2]:
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    # run_baseline(item)

    # if run_model:
    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()




def active_patch_selection_old(prev_active_model_log, rand_img_log_to_match, training_image_sets, iteration_number):
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_sets = []
    all_pred_images = []
    # image_set_str_to_image_shape = {}
    # image_set_str_to_patch_size = {}
    image_set_info = {}




    
    annotations_to_match = 0
    patches_to_match = 0
    # if rand_img_log_to_match is not None:
    for image_set in rand_img_log_to_match["image_sets"]:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
    
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)
        for image_name in image_set["taken_regions"].keys():
            annotations_to_match += len(annotations[image_name]["boxes"])
            patches_to_match += (num_patches_per_image)




















    taken_patches = {}

    if prev_active_model_log is None:
        all_patch_candidates = []
        for image_set in training_image_sets:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            image_set_dir = os.path.join("usr", "data", 
                                        username, "image_sets",
                                        farm_name,
                                        field_name,
                                        mission_date)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            # annotations = image_set_info[image_set_str]["annotations"]
            # image_shape = image_set_info[image_set_str]["image_shape"]
            image_height = metadata["images"][list(annotations.keys())[0]]["height_px"]
            image_width = metadata["images"][list(annotations.keys())[0]]["width_px"]
            # image_height = image_shape[0]
            # image_width = image_shape[1]
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            
            # patch_size = image_set_info[image_set_str]["patch_size"]

            incr = patch_size #- overlap_px
            w_covered = max(image_width - patch_size, 0)
            num_w_patches = m.ceil(w_covered / incr) + 1

            h_covered = max(image_height - patch_size, 0)
            num_h_patches = m.ceil(h_covered / incr) + 1


            image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
            print(image_set_str, patch_size)

            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    for i in range(0, num_w_patches):
                        for j in range(0, num_h_patches):
                            patch_min_y = (patch_size) * j
                            patch_min_x = (patch_size) * i

                            patch_max_y = min(patch_min_y + patch_size, image_height)
                            patch_max_x = min(patch_min_x + patch_size, image_width)

                            patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                            all_patch_candidates.append((image_set_str, image_name, patch_region))



        extra_patches = random.sample(all_patch_candidates, patches_to_match)
        for extra_patch in extra_patches:
            image_set_str = extra_patch[0]
            image_name = extra_patch[1]
            extra_patch_coords = extra_patch[2]
            if image_set_str not in taken_patches:
                taken_patches[image_set_str] = {}
            if image_name not in taken_patches[image_set_str]:
                taken_patches[image_set_str][image_name] = []
            taken_patches[image_set_str][image_name].append(extra_patch_coords)


    else:
        for image_set in training_image_sets:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            image_set_dir = os.path.join("usr", "data", 
                                        username, "image_sets",
                                        farm_name,
                                        field_name,
                                        mission_date)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            print("switching to model")
            model_dir = os.path.join(image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": prev_active_model_log["model_name"],
                "model_creator": prev_active_model_log["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(3)
                if not os.path.exists(switch_req_path):
                    switch_processed = True


        
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": [image_name for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "regions": [[[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]] for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "save_result": True,
                "regions_only": True,
                "calculate_vegetation_record": False,
                "results_name": "active_learning_iter_" + str(iteration_number),
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)

            # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
            # if len(result_dirs) > 1:
            #     raise RuntimeError("More than one result dir")
            # results_dir = result_dirs[0]


            results_dir = os.path.join(image_set_dir, "model", "results", request_uuid)
            predictions_path = os.path.join(results_dir, "predictions.json")
            predictions = json_io.load_json(predictions_path)

            image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date


            image_set_info[image_set_str] = {}
            image_set_info[image_set_str]["image_shape"] = [metadata["images"][list(annotations.keys())[0]]["height_px"], metadata["images"][list(annotations.keys())[0]]["width_px"]]
            
            
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            image_set_info[image_set_str]["patch_size"] = patch_size

            image_set_info[image_set_str]["annotations"] = annotations
            
            for image_name in predictions.keys():

                for i in range(len(predictions[image_name]["scores"])):
                    all_pred_boxes.append(predictions[image_name]["boxes"][i])
                    all_pred_scores.append(predictions[image_name]["scores"][i])
                    all_pred_image_sets.append(image_set_str)
                    all_pred_images.append(image_name)





        if prev_active_model_log is not None:
            for image_set in prev_active_model_log["image_sets"]:
                image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
                taken_patches[image_set_str] = image_set["taken_regions"]




        print(taken_patches.keys())
        
        for q_score in np.arange(0, 51, 1):
            print("q_score: {}".format(q_score))
            image_q_scores = (abs(0.5 -np.array(all_pred_scores)) * 100).astype(np.int64)
            # q_score_mask = image_q_scores == q_score
            q_score_inds = np.where(image_q_scores == q_score)[0]
            np.random.shuffle(q_score_inds)
            for q_score_ind in q_score_inds:
                candidate_box = np.array(all_pred_boxes[q_score_ind])
                centre = (candidate_box[..., :2] + candidate_box[..., 2:]) / 2.0
                candidate_image_set_str = all_pred_image_sets[q_score_ind]
                candidate_image_name = all_pred_images[q_score_ind]
                image_shape = image_set_info[candidate_image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[candidate_image_set_str]["patch_size"]


                patch_min_y = int((centre[0] // patch_size) * patch_size)
                patch_min_x = int((centre[1] // patch_size) * patch_size)
                patch_max_y = int(min(patch_min_y + patch_size, image_height))
                patch_max_x = int(min(patch_min_x + patch_size, image_width))
                patch = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                already_taken = False
                # if prev_active_model_log is not None:
                    # for image_set in prev_active_model_log["image_sets"]:
                        # if image_set["username"] == username and image_set["farm_name"] == farm_name and image_set["field_name"] == "field_name" and image_set["mission_date"] == "mission_date":
                if candidate_image_set_str not in taken_patches:
                    taken_patches[candidate_image_set_str] = {}
                if candidate_image_name not in taken_patches[candidate_image_set_str]:
                    taken_patches[candidate_image_set_str][candidate_image_name] = []
                if patch in taken_patches[candidate_image_set_str][candidate_image_name]:
                    already_taken = True
                # if not already_taken:
                #     if image_name not in taken_patches[image_set_str]:
                #         taken_patches[image_set_str][image_name] = []
                #     taken_patches[image_set_str][image_name].append(patch)

                cur_total = 0
                if not already_taken:
                    for image_set_str in taken_patches.keys():
                        annotations = image_set_info[image_set_str]["annotations"]
                        for image_name in taken_patches[image_set_str].keys():

                            if image_set_str == candidate_image_set_str and image_name == candidate_image_name:
                                l = taken_patches[image_set_str][image_name].copy() #annotations[candidate_image_name]["training_regions"].copy()
                                l.append(patch)
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                            else:
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                    
                    if cur_total <= annotations_to_match:
                        taken_patches[candidate_image_set_str][candidate_image_name].append(patch)

                    print(cur_total, annotations_to_match, len(taken_patches[candidate_image_set_str][candidate_image_name]))
                        # added_candidates.append((candidates[i][0], candidates[i][1]))
                    
                    # for image_name in image_names:
                    #     if image_name != candidates[i][0]:
                    #         if len(annotations[image_name]["training_regions"]) > 0:
                    #             cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                    # l = annotations[candidates[i][0]]["training_regions"].copy()
                    # l.append(candidates[i][1])
                    # cur_total += box_utils.get_contained_inds(annotations[candidates[i][0]]["boxes"], l).size
                    # if cur_total <= total_to_match:
                    #     annotations[candidates[i][0]]["training_regions"].append(candidates[i][1])
                    #     added_candidates.append((candidates[i][0], candidates[i][1]))
                if cur_total == annotations_to_match:
                    break
            if cur_total == annotations_to_match:
                break

        cur_num_patches = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                cur_num_patches += len(taken_patches[image_set_str][image_name])

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        num_to_add = patches_to_match - cur_num_patches
        if cur_num_patches < patches_to_match:
            all_patch_candidates = []
            for image_set_str in image_set_info.keys():
                annotations = image_set_info[image_set_str]["annotations"]
                # for image_name in taken_patches[image_set_str]:
                image_shape = image_set_info[image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[image_set_str]["patch_size"]

                incr = patch_size #- overlap_px
                w_covered = max(image_width - patch_size, 0)
                num_w_patches = m.ceil(w_covered / incr) + 1

                h_covered = max(image_height - patch_size, 0)
                num_h_patches = m.ceil(h_covered / incr) + 1



                for image_name in annotations.keys():

                    if len(annotations[image_name]["test_regions"]) > 0:
                        for i in range(0, num_w_patches):
                            for j in range(0, num_h_patches):
                                patch_min_y = (patch_size) * j
                                patch_min_x = (patch_size) * i

                                patch_max_y = min(patch_min_y + patch_size, image_height)
                                patch_max_x = min(patch_min_x + patch_size, image_width)

                                patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                                if image_set_str in taken_patches and image_name in taken_patches[image_set_str]:
                                    # if patch_region not in taken_patches[image_set_str][image_name]:
                                        # possible = True
                                        # all_patch_candidates.append(

                                        # )
                                    if patch_region not in taken_patches[image_set_str][image_name]: #patch_region not in annotations[image_name]["training_regions"]:
                                        num_without = box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                                        l = taken_patches[image_set_str][image_name].copy()
                                        l.append(patch_region)
                                        num_with = box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                        if num_without == num_with:
                                            all_patch_candidates.append((image_set_str, image_name, patch_region))


                                else:
                                    num_contained = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_region]).size
                                    if num_contained == 0:
                                        all_patch_candidates.append((
                                            image_set_str,
                                            image_name,
                                            patch_region
                                        ))

            print("picking {} from {} candidates".format(num_to_add, len(all_patch_candidates)))
            extra_patches = random.sample(all_patch_candidates, num_to_add)
            for extra_patch in extra_patches:
                image_set_str = extra_patch[0]
                image_name = extra_patch[1]
                extra_patch_coords = extra_patch[2]
                if image_set_str not in taken_patches:
                    taken_patches[image_set_str] = {}
                if image_name not in taken_patches[image_set_str]:
                    taken_patches[image_set_str][image_name] = []

                taken_patches[image_set_str][image_name].append(extra_patch_coords)

                
        elif cur_num_patches > patches_to_match:
            raise RuntimeError("too many patches -- not implemented")
                
            # random_image_set_str = random.choice

    if prev_active_model_log is not None:
        print("\n\n--- FINAL CHECK---\n\n")
        cur_num_patches = 0
        cur_num_annotations = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                annotations = image_set_info[image_set_str]["annotations"]
                cur_num_patches += len(taken_patches[image_set_str][image_name])
                cur_num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                # for patch_region in taken_patches[image_set_str][image_name]:
                #     if patch_region.shape[0] == 0:
                #         raise RuntimeError("BAD!")

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        if cur_num_patches != patches_to_match:
            raise RuntimeError("cur_num_patches != patches_to_match")


        print("cur_num_annotations", cur_num_annotations)
        print("annotations_to_match", annotations_to_match)
        if cur_num_annotations != annotations_to_match:
            raise RuntimeError("cur_num_annotations != annotations_to_match")

    log = {}
    log["image_sets"] = []
    for image_set_str in taken_patches.keys():
        pieces = image_set_str.split(" ")
        username = pieces[0]
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]

        log["image_sets"].append({
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "taken_regions": taken_patches[image_set_str]
        })


    log["model_creator"] = "erik"
    log["model_name"] = "selected_patches_" + str(iteration_number)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()




def active_patch_selection(training_image_sets, model_name, model_dir_to_match, prev_model_dir): #prev_active_model_log, rand_img_log_to_match, training_image_sets, iteration_number):
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_sets = []
    all_pred_images = []
    # image_set_str_to_image_shape = {}
    # image_set_str_to_patch_size = {}
    image_set_info = {}


    log_to_match_path = os.path.join(model_dir_to_match, "log.json")
    log_to_match = json_io.load_json(log_to_match_path)


    log = {}
    
    annotations_to_match = 0
    patches_to_match = 0
    # if rand_img_log_to_match is not None:
    for image_set in log_to_match["image_sets"]:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
    
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        # metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        # metadata = json_io.load_json(metadata_path)

        # patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        # num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)
        # for image_name in image_set["taken_regions"].keys():
        #     annotations_to_match += len(annotations[image_name]["boxes"])
        #     patches_to_match += (num_patches_per_image)


        for image_name in image_set["taken_regions"].keys():
            cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], image_set["taken_regions"][image_name]).size
            patches_to_match += len(image_set["taken_regions"][image_name])

    taken_patches = {}

    if prev_model_dir is None:
        log["image_sets"] = log_to_match["image_sets"]

    else:
        prev_active_model_log_path = os.path.join(prev_model_dir, "log.json")
        prev_active_model_log = json_io.load_json(prev_active_model_log_path)
        for image_set in training_image_sets:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            image_set_dir = os.path.join("usr", "data", 
                                        username, "image_sets",
                                        farm_name,
                                        field_name,
                                        mission_date)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            print("switching to model")
            model_dir = os.path.join(image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": prev_active_model_log["model_name"],
                "model_creator": prev_active_model_log["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(3)
                if not os.path.exists(switch_req_path):
                    switch_processed = True


        
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": [image_name for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "regions": [[[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]] for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "save_result": True,
                "regions_only": True,
                "calculate_vegetation_record": False,
                "results_name": "active_learning_eval",
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)

            # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
            # if len(result_dirs) > 1:
            #     raise RuntimeError("More than one result dir")
            # results_dir = result_dirs[0]


            results_dir = os.path.join(image_set_dir, "model", "results", request_uuid)
            predictions_path = os.path.join(results_dir, "predictions.json")
            predictions = json_io.load_json(predictions_path)

            image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date


            image_set_info[image_set_str] = {}
            image_set_info[image_set_str]["image_shape"] = [metadata["images"][list(annotations.keys())[0]]["height_px"], metadata["images"][list(annotations.keys())[0]]["width_px"]]
            
            
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
            image_set_info[image_set_str]["patch_size"] = patch_size

            image_set_info[image_set_str]["annotations"] = annotations
            
            for image_name in predictions.keys():

                for i in range(len(predictions[image_name]["scores"])):
                    all_pred_boxes.append(predictions[image_name]["boxes"][i])
                    all_pred_scores.append(predictions[image_name]["scores"][i])
                    all_pred_image_sets.append(image_set_str)
                    all_pred_images.append(image_name)





        # if prev_active_model_log is not None:
        for image_set in prev_active_model_log["image_sets"]:
            image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
            taken_patches[image_set_str] = image_set["taken_regions"]




        print(taken_patches.keys())
        
        for q_score in np.arange(0, 51, 1):
            print("q_score: {}".format(q_score))
            image_q_scores = (abs(0.5 -np.array(all_pred_scores)) * 100).astype(np.int64)
            # q_score_mask = image_q_scores == q_score
            q_score_inds = np.where(image_q_scores == q_score)[0]
            np.random.shuffle(q_score_inds)
            for q_score_ind in q_score_inds:
                candidate_box = np.array(all_pred_boxes[q_score_ind])
                centre = (candidate_box[..., :2] + candidate_box[..., 2:]) / 2.0
                candidate_image_set_str = all_pred_image_sets[q_score_ind]
                candidate_image_name = all_pred_images[q_score_ind]
                image_shape = image_set_info[candidate_image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[candidate_image_set_str]["patch_size"]


                patch_min_y = int((centre[0] // patch_size) * patch_size)
                patch_min_x = int((centre[1] // patch_size) * patch_size)
                patch_max_y = int(min(patch_min_y + patch_size, image_height))
                patch_max_x = int(min(patch_min_x + patch_size, image_width))
                patch = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                already_taken = False
                # if prev_active_model_log is not None:
                    # for image_set in prev_active_model_log["image_sets"]:
                        # if image_set["username"] == username and image_set["farm_name"] == farm_name and image_set["field_name"] == "field_name" and image_set["mission_date"] == "mission_date":
                if candidate_image_set_str not in taken_patches:
                    taken_patches[candidate_image_set_str] = {}
                if candidate_image_name not in taken_patches[candidate_image_set_str]:
                    taken_patches[candidate_image_set_str][candidate_image_name] = []
                if patch in taken_patches[candidate_image_set_str][candidate_image_name]:
                    already_taken = True
                # if not already_taken:
                #     if image_name not in taken_patches[image_set_str]:
                #         taken_patches[image_set_str][image_name] = []
                #     taken_patches[image_set_str][image_name].append(patch)

                cur_total = 0
                if not already_taken:
                    for image_set_str in taken_patches.keys():
                        annotations = image_set_info[image_set_str]["annotations"]
                        for image_name in taken_patches[image_set_str].keys():

                            if image_set_str == candidate_image_set_str and image_name == candidate_image_name:
                                l = taken_patches[image_set_str][image_name].copy() #annotations[candidate_image_name]["training_regions"].copy()
                                l.append(patch)
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                            else:
                                cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                    
                    if cur_total <= annotations_to_match:
                        taken_patches[candidate_image_set_str][candidate_image_name].append(patch)

                    print(cur_total, annotations_to_match, len(taken_patches[candidate_image_set_str][candidate_image_name]))
                        # added_candidates.append((candidates[i][0], candidates[i][1]))
                    
                    # for image_name in image_names:
                    #     if image_name != candidates[i][0]:
                    #         if len(annotations[image_name]["training_regions"]) > 0:
                    #             cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                    # l = annotations[candidates[i][0]]["training_regions"].copy()
                    # l.append(candidates[i][1])
                    # cur_total += box_utils.get_contained_inds(annotations[candidates[i][0]]["boxes"], l).size
                    # if cur_total <= total_to_match:
                    #     annotations[candidates[i][0]]["training_regions"].append(candidates[i][1])
                    #     added_candidates.append((candidates[i][0], candidates[i][1]))
                if cur_total == annotations_to_match:
                    break
            if cur_total == annotations_to_match:
                break

        cur_num_patches = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                cur_num_patches += len(taken_patches[image_set_str][image_name])

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        num_to_add = patches_to_match - cur_num_patches
        if cur_num_patches < patches_to_match:
            all_patch_candidates = []
            for image_set_str in image_set_info.keys():
                annotations = image_set_info[image_set_str]["annotations"]
                # for image_name in taken_patches[image_set_str]:
                image_shape = image_set_info[image_set_str]["image_shape"]
                image_height = image_shape[0]
                image_width = image_shape[1]
                patch_size = image_set_info[image_set_str]["patch_size"]

                incr = patch_size #- overlap_px
                w_covered = max(image_width - patch_size, 0)
                num_w_patches = m.ceil(w_covered / incr) + 1

                h_covered = max(image_height - patch_size, 0)
                num_h_patches = m.ceil(h_covered / incr) + 1



                for image_name in annotations.keys():

                    if len(annotations[image_name]["test_regions"]) > 0:
                        for i in range(0, num_w_patches):
                            for j in range(0, num_h_patches):
                                patch_min_y = (patch_size) * j
                                patch_min_x = (patch_size) * i

                                patch_max_y = min(patch_min_y + patch_size, image_height)
                                patch_max_x = min(patch_min_x + patch_size, image_width)

                                patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                                if image_set_str in taken_patches and image_name in taken_patches[image_set_str]:
                                    # if patch_region not in taken_patches[image_set_str][image_name]:
                                        # possible = True
                                        # all_patch_candidates.append(

                                        # )
                                    if patch_region not in taken_patches[image_set_str][image_name]: #patch_region not in annotations[image_name]["training_regions"]:
                                        num_without = box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                                        l = taken_patches[image_set_str][image_name].copy()
                                        l.append(patch_region)
                                        num_with = box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                        if num_without == num_with:
                                            all_patch_candidates.append((image_set_str, image_name, patch_region))


                                else:
                                    num_contained = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_region]).size
                                    if num_contained == 0:
                                        all_patch_candidates.append((
                                            image_set_str,
                                            image_name,
                                            patch_region
                                        ))

            print("picking {} from {} candidates".format(num_to_add, len(all_patch_candidates)))
            extra_patches = random.sample(all_patch_candidates, num_to_add)
            for extra_patch in extra_patches:
                image_set_str = extra_patch[0]
                image_name = extra_patch[1]
                extra_patch_coords = extra_patch[2]
                if image_set_str not in taken_patches:
                    taken_patches[image_set_str] = {}
                if image_name not in taken_patches[image_set_str]:
                    taken_patches[image_set_str][image_name] = []

                taken_patches[image_set_str][image_name].append(extra_patch_coords)

                
        elif cur_num_patches > patches_to_match:
            raise RuntimeError("too many patches -- not implemented")
                
            # random_image_set_str = random.choice


        print("\n\n--- FINAL CHECK---\n\n")
        cur_num_patches = 0
        cur_num_annotations = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                annotations = image_set_info[image_set_str]["annotations"]
                cur_num_patches += len(taken_patches[image_set_str][image_name])
                cur_num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                # for patch_region in taken_patches[image_set_str][image_name]:
                #     if patch_region.shape[0] == 0:
                #         raise RuntimeError("BAD!")

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        if cur_num_patches != patches_to_match:
            raise RuntimeError("cur_num_patches != patches_to_match")


        print("cur_num_annotations", cur_num_annotations)
        print("annotations_to_match", annotations_to_match)
        if cur_num_annotations != annotations_to_match:
            raise RuntimeError("cur_num_annotations != annotations_to_match")


        log["image_sets"] = []
        for image_set_str in taken_patches.keys():
            pieces = image_set_str.split(" ")
            username = pieces[0]
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]

            log["image_sets"].append({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "taken_regions": taken_patches[image_set_str]
            })


    log["model_creator"] = "erik"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()



def exg_active_patch_selection(training_image_sets, model_name, model_dir_to_match, prev_model_dir): #prev_active_model_log, rand_img_log_to_match, training_image_sets, iteration_number):
    # all_pred_boxes = []
    # all_pred_scores = []
    # all_pred_image_sets = []
    # all_pred_images = []
    # image_set_str_to_image_shape = {}
    # image_set_str_to_patch_size = {}
    # image_set_info = {}


    log_to_match_path = os.path.join(model_dir_to_match, "log.json")
    log_to_match = json_io.load_json(log_to_match_path)


    log = {}
    


    if prev_model_dir is None:
        log["image_sets"] = log_to_match["image_sets"]

    else:
        candidates = []
        taken_patches = {}

        patches_to_match = get_num_patches_used_by_model(model_dir_to_match) #0
        # for image_set in log_to_match["image_sets"]:
        #     username = image_set["username"]
        #     farm_name = image_set["farm_name"]
        #     field_name = image_set["field_name"]
        #     mission_date = image_set["mission_date"]
        #     image_set_dir = os.path.join("usr", "data", 
        #                                 username, "image_sets",
        #                                 farm_name,
        #                                 field_name,
        #                                 mission_date)
        
        #     # annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        #     # annotations = annotation_utils.load_annotations(annotations_path)

        #     for image_name in image_set["taken_regions"].keys():
        #         # cur_total += box_utils.get_contained_inds(annotations[image_name]["boxes"], image_set["taken_regions"][image_name]).size
        #         patches_to_match += len(image_set["taken_regions"][image_name])



        prev_active_model_log_path = os.path.join(prev_model_dir, "log.json")
        prev_active_model_log = json_io.load_json(prev_active_model_log_path)


        num_patches_prev_taken = 0
        for image_set in prev_active_model_log["image_sets"]:
            image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
            taken_patches[image_set_str] = image_set["taken_regions"]
            for image_name in image_set["taken_regions"]:
                num_patches_prev_taken += len(image_set["taken_regions"][image_name])


        for image_set in training_image_sets:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
            image_set_dir = os.path.join("usr", "data", 
                                        username, "image_sets",
                                        farm_name,
                                        field_name,
                                        mission_date)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)

            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)


            image_names = list(annotations.keys())
            image_w = metadata["images"][image_names[0]]["width_px"]
            image_h = metadata["images"][image_names[0]]["height_px"]

            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

                
            patch_overlap_percent = 0
            overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

            incr = patch_size - overlap_px
            w_covered = max(image_w - patch_size, 0)
            num_w_patches = m.ceil(w_covered / incr) + 1

            h_covered = max(image_h - patch_size, 0)
            num_h_patches = m.ceil(h_covered / incr) + 1


            print("switching to model")
            model_dir = os.path.join(image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": prev_active_model_log["model_name"],
                "model_creator": prev_active_model_log["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(3)
                if not os.path.exists(switch_req_path):
                    switch_processed = True


        
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": [image_name for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "regions": [[[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]] for image_name in annotations.keys() if len(annotations[image_name]["test_regions"]) > 0],
                "save_result": True,
                "regions_only": True,
                "calculate_vegetation_record": False,
                "results_name": "active_learning_eval",
                "results_message": ""
            }

            request_path = os.path.join(image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)

            # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
            # if len(result_dirs) > 1:
            #     raise RuntimeError("More than one result dir")
            # results_dir = result_dirs[0]


            results_dir = os.path.join(image_set_dir, "model", "results", request_uuid)
            predictions_path = os.path.join(results_dir, "predictions.json")
            predictions = json_io.load_json(predictions_path)


            for image_name in tqdm.tqdm(list(predictions.keys()), desc="Collecting excess green scores"):
                image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
                image = Image(image_path)
                image_array = image.load_image_array()
                exg_array = image_utils.excess_green(image_array)

                pred_boxes = np.array(predictions[image_name]["boxes"])
                pred_scores = np.array(predictions[image_name]["scores"])
                sel_pred_boxes = pred_boxes[pred_scores > 0.5]

                exg_array[exg_array < 0] = 0

                for pred_box in sel_pred_boxes:
                    exg_array[pred_box[0]:pred_box[2], pred_box[1]:pred_box[3]] = -10


                image_w = metadata["images"][image_name]["width_px"]
                image_h = metadata["images"][image_name]["height_px"]
                for h_index in range(num_h_patches):
                    for w_index in range(num_w_patches):

                        patch_coords = [
                            patch_size * h_index,
                            patch_size * w_index,
                            min((patch_size * h_index) + patch_size, image_h),
                            min((patch_size * w_index) + patch_size, image_w)
                        ]
                        if image_set_str not in taken_patches or image_name not in taken_patches[image_set_str] or patch_coords not in taken_patches[image_set_str][image_name]:
                            # if image_set_str in taken_patches and image_name in taken_patches[image_set_str] and patch_coords not in taken_patches[image_set_str][image_name]:

                            
                            # inds = box_utils.get_contained_inds(sel_pred_boxes, [patch_coords])
                            # patch_boxes = sel_pred_boxes[inds]
                            # for patch_box in patch_boxes:
                            #     exg_array[min(0, patch_box[0]-patch_coords[0]):min(0, patch_box[2]-patch_coords[0]),
                            #               min(0, patch_box[1]-patch_coords[1]):min(0, patch_box[3]-patch_coords[1])] = -10
                            
                            exg_patch = exg_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
                            sel_vals = exg_patch[exg_patch != -10]
                            score = np.sum(sel_vals ** 2) / sel_vals.size

                            
                            candidates.append(
                                (image_set_str, image_name, patch_coords, score)
                            )
                            
                        
        candidates.sort(key=lambda x: x[3], reverse=True)


        num_patches_to_take = patches_to_match - num_patches_prev_taken

        print("Patch count to match: {}".format(patches_to_match))
        print("Num patches previously taken: {}".format(num_patches_prev_taken))
        print("Num patches to take: {}".format(num_patches_to_take))
        print("Num candidates: {}".format(len(candidates)))

        taken_candidates = candidates[:num_patches_to_take]

        for candidate in taken_candidates:
            image_set_str = candidate[0]
            image_name = candidate[1]
            patch_coords = candidate[2]
            if image_set_str not in taken_patches:
                taken_patches[image_set_str] = {}
            if image_name not in taken_patches[image_set_str]:
                taken_patches[image_set_str][image_name] = []

                
            taken_patches[image_set_str][image_name].append(patch_coords)
            


        print("\n\n--- FINAL CHECK---\n\n")
        cur_num_patches = 0
        # cur_num_annotations = 0
        for image_set_str in taken_patches.keys():
            for image_name in taken_patches[image_set_str]:
                # annotations = image_set_info[image_set_str]["annotations"]
                cur_num_patches += len(taken_patches[image_set_str][image_name])
                # cur_num_annotations += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_patches[image_set_str][image_name]).size
                # for patch_region in taken_patches[image_set_str][image_name]:
                #     if patch_region.shape[0] == 0:
                #         raise RuntimeError("BAD!")

        print("cur_num_patches", cur_num_patches)
        print("patches_to_match", patches_to_match)
        if cur_num_patches != patches_to_match:
            raise RuntimeError("cur_num_patches != patches_to_match")


        # print("cur_num_annotations", cur_num_annotations)
        # print("annotations_to_match", annotations_to_match)
        # if cur_num_annotations != annotations_to_match:
        #     raise RuntimeError("cur_num_annotations != annotations_to_match")


        log["image_sets"] = []
        for image_set_str in taken_patches.keys():
            pieces = image_set_str.split(" ")
            username = pieces[0]
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]

            log["image_sets"].append({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "taken_regions": taken_patches[image_set_str],
                "patch_overlap_percent": 0
            })


    log["model_creator"] = "erik"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)

    if prev_model_dir is not None:
        candidates_dir = os.path.join(pending_model_path, "added_candidates")
        os.makedirs(candidates_dir)
        restructured_taken_candidates = {}
        for candidate in taken_candidates:
            image_set_str = candidate[0]
            image_name = candidate[1]
            patch_coords = candidate[2]
            score = candidate[3]
            if image_set_str not in restructured_taken_candidates:
                restructured_taken_candidates[image_set_str] = {}
            if image_name not in restructured_taken_candidates[image_set_str]:
                restructured_taken_candidates[image_set_str][image_name] = []

                
            restructured_taken_candidates[image_set_str][image_name].append((patch_coords, score))


        for image_set_str in restructured_taken_candidates:
            
            pieces = image_set_str.split(" ")
            username = pieces[0]
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]
            image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)


            for image_name in restructured_taken_candidates[image_set_str]:

                image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
                image = Image(image_path)
                image_array = image.load_image_array()

                for item in restructured_taken_candidates[image_set_str][image_name]:
                    patch_coords = item[0]
                    score = item[1]
                    patch_path = os.path.join(candidates_dir, str(score) + ".png")
                    image_patch = image_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
                    cv2.imwrite(patch_path, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))




    log_path = os.path.join(pending_model_path, "log.json")
    log["submission_time"] = int(time.time())
    json_io.save_json(log_path, log)

    server.sch_ctx["baseline_queue"].enqueue(log)

    baseline_queue_size = server.sch_ctx["baseline_queue"].size()
    print("baseline_queue_size", baseline_queue_size)
    while baseline_queue_size > 0:
    
        log = server.sch_ctx["baseline_queue"].dequeue()
        re_enqueue = server.process_baseline(log)
        if re_enqueue:
            server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()














def rand_img_selection(prev_log, training_image_sets, num_images_to_select, iteration_number, run_model):

    candidates = []
    for training_image_set in training_image_sets:
        username = training_image_set["username"]
        farm_name = training_image_set["farm_name"]
        field_name = training_image_set["field_name"]
        mission_date = training_image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                      username, "image_sets",
                                      farm_name,
                                      field_name,
                                      mission_date)
        

        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                image_id = username + " " + farm_name + " " + field_name + " " + mission_date + " " + image_name
                candidates.append(image_id)
    
    prev_taken = []
    if prev_log is not None:
        for image_set in prev_log["image_sets"]:
            username = image_set["username"]
            farm_name = image_set["farm_name"]
            field_name = image_set["field_name"]
            mission_date = image_set["mission_date"]
            for image_name in image_set["taken_regions"].keys():
                image_id = username + " " + farm_name + " " + field_name + " " + mission_date + " " + image_name
                prev_taken.append(image_id)

    random.shuffle(candidates)
    new_num_taken = 0
    new_taken = []
    # new_taken.extend(prev_taken)
    for i in range(len(candidates)):
        print("checking candidate", candidates[i])
        if candidates[i] not in prev_taken:
            print("\t\tadding candidate", candidates[i])
            new_taken.append(candidates[i])
            new_num_taken += 1
        if new_num_taken >= num_images_to_select:
            break

    # processed_taken = {}
    # for entry in new_taken:
    #     pieces = entry.split(" ")
    #     image_set_str = pieces[0] + " " + pieces[1] + " " + pieces[2] + " " + pieces[3]
    #     if image_set_str not in processed_taken:
    #         processed_taken[image_set_str] = []

    #     processed_taken[image_set_str].append(pieces[4])

    # region_coords = {}
    log = {}
    log["image_sets"] = []
    # if prev_log is not None:
    #     for image_set in prev_log["image_sets"]:
    #         log["image_sets"].append(image_set)
        # username = image_set["username"]
        # farm_name = image_set["farm_name"]
        # field_name = image_set["field_name"]
        # mission_date = image_set["mission_date"]
        # for image_name in image_set["taken_regions"].keys():



    # image_sets = []
    all_taken = []
    all_taken.extend(prev_taken)
    all_taken.extend(new_taken)
    for entry in all_taken: #processed_taken.keys():
        pieces = entry.split(" ")
        username = pieces[0]
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]
        image_name = pieces[4]
        image_set_dir = os.path.join("usr", "data", 
                                username, "image_sets",
                                farm_name,
                                field_name,
                                mission_date)
        
        # annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        # annotations = annotation_utils.load_annotations(annotations_path)
        # patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        # region_coords[image_set_str] = []
        image_height = metadata["images"][image_name]["height_px"]
        image_width = metadata["images"][image_name]["width_px"]
        # num_patches = 0
        # taken_regions = {}
        # for image_name in processed_taken[image_set_str]:
        #     # image_patch_coords = []
        #     image_height = metadata["images"][image_name]["height_px"]
        #     image_width = metadata["images"][image_name]["width_px"]

        #     # taken_regions[image_name] = [0, 0, image_height, image_width]

        #     # num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=0)

        #     # num_patches += num_patches_per_image


        #     # col_covered = False
        #     # patch_min_y = 0
        #     # while not col_covered:
        #     #     patch_max_y = patch_min_y + patch_size
        #     #     max_content_y = patch_max_y
        #     #     if patch_max_y >= image_height:
        #     #         max_content_y = image_height
        #     #         col_covered = True

        #     #     row_covered = False
        #     #     patch_min_x = 0
        #     #     while not row_covered:

        #     #         patch_max_x = patch_min_x + patch_size
        #     #         max_content_x = patch_max_x
        #     #         if patch_max_x >= image_width:
        #     #             max_content_x = image_width
        #     #             row_covered = True

                    
        #     #         patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

        #     #         image_patch_coords.append(patch_coords)

                
        #     #         patch_min_x += (patch_size)

        #     #     patch_min_y += (patch_size)
        
        #     # region_coords[image_set_str][image_name] = image_patch_coords
        exists = False
        for image_set in log["image_sets"]:

            if image_set["username"] == username and image_set["farm_name"] == farm_name and image_set["field_name"] == "field_name" and image_set["mission_date"] == "mission_date":
                # for image_name in processed_taken.keys():
                image_set["taken_regions"][image_name] = [[0, 0, image_height, image_width]]
                exists = True

        if not exists:
            log["image_sets"].append({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "taken_regions": {image_name: [[0, 0, image_height, image_width]]}, # for image_name in processed_taken.keys()}
                "patch_overlap_percent": 0
            })

            

        # image_sets.append({
        #     "username": username,
        #     "farm_name": farm_name,
        #     "field_name": field_name,
        #     "mission_date": mission_date,
        #     "taken": taken_regions,
        #     "num_patches": num_patches
        # })

    # if prev_log is None:
    #     prev_num = 0
    # else:
    #     prev_num = int(prev_log["model_name"].split("_")[-1])
    # cur_num = num_images_to_select + prev_num

    log["model_creator"] = "erik"
    # log["model_name"] = str(num_images_to_select) ##"set_of_" + str(len(org_image_sets)) + "_no_overlap"
    log["model_name"] = "random_images_" + str(iteration_number) #cur_num)
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    # log["image_sets"] = image_sets
    


    # for item in [item2]: #[item1, item2]:
    log["submission_time"] = int(time.time())
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)

    # run_baseline(item)

    if run_model:
        server.sch_ctx["baseline_queue"].enqueue(log)

        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        print("baseline_queue_size", baseline_queue_size)
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()




def run_random_images_models():
    models_dir = os.path.join("usr", "data", "erik", "models")
    prev_models = glob.glob(os.path.join(models_dir, "pending", "random_images_*"))
    sorted_prev_models = natsorted(prev_models)
    for prev_model in sorted_prev_models:
        log_path = os.path.join(prev_model, "log.json")
        log = json_io.load_json(log_path)
        log["submission_time"] = int(time.time())
        json_io.save_json(log_path, log)




        server.sch_ctx["baseline_queue"].enqueue(log)
        baseline_queue_size = server.sch_ctx["baseline_queue"].size()
        while baseline_queue_size > 0:
        
            log = server.sch_ctx["baseline_queue"].dequeue()
            re_enqueue = server.process_baseline(log)
            if re_enqueue:
                server.sch_ctx["baseline_queue"].enqueue(log)
            baseline_queue_size = server.sch_ctx["baseline_queue"].size()


def run_my_random_test(training_image_sets, run_models):
    models_dir = os.path.join("usr", "data", "erik", "models")
    for i in range(16):
        if i == 0:
            prev_log = None
        else:
            prev_models = glob.glob(os.path.join(models_dir, "available", "public", "random_images_*"))
            prev_models = glob.glob(os.path.join(models_dir, "pending", "random_images_*"))
            prev_counts = [(x, int(x.split("_")[-1])) for x in prev_models]
            prev_counts.sort(key=lambda x: x[1])
            prev_tup = prev_counts[-1]
            prev_model_path = prev_tup[0]
            prev_log_path = os.path.join(prev_model_path, "log.json")
            prev_log = json_io.load_json(prev_log_path)


        rand_img_selection(prev_log, training_image_sets, 10, i, run_model=run_models)

        
def run_my_active_test(training_image_sets):

    models_dir = os.path.join("usr", "data", "erik", "models")
    for i in range(16):
        if i == 0:
            prev_active_model_log = None
        else:
            prev_models = glob.glob(os.path.join(models_dir, "available", "public", "selected_patches_*"))
            prev_counts = [(x, int(x.split("_")[-1])) for x in prev_models]
            prev_counts.sort(key=lambda x: x[1])
            prev_tup = prev_counts[-1]
            prev_model_path = prev_tup[0]
            prev_log_path = os.path.join(prev_model_path, "log.json")
            prev_active_model_log = json_io.load_json(prev_log_path)


        rand_img_log_to_match_path = os.path.join(models_dir, "available", "public", "random_images_" + str(i), "log.json")
        rand_img_log_to_match_path = os.path.join(models_dir, "pending", "random_images_" + str(i), "log.json")
        rand_img_log_to_match = json_io.load_json(rand_img_log_to_match_path)

        
        active_patch_selection(prev_active_model_log, rand_img_log_to_match, training_image_sets, i)


def remove_previous_iter_results(training_image_sets):
    for image_set in training_image_sets:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
        # if len(result_dirs) != 1:
        #     raise RuntimeError("oh no")
        result_dir = os.path.join(image_set_dir, "model", "results")
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)

def remove_specific_iter_results(training_image_sets, result_name):
    for image_set in training_image_sets:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        # result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
        # if len(result_dirs) != 1:
        #     raise RuntimeError("oh no")
        result_dirs = glob.glob(os.path.join(image_set_dir, "model", "results", "*"))
        for result_dir in result_dirs:
            request_path = os.path.join(result_dir, "request.json")
            request = json_io.load_json(request_path)
            if request["results_name"] == result_name:
                shutil.rmtree(result_dir)



def get_patch_nums_of_sets(image_sets):
    # all_totals = []
    totals = {}
    for image_set in image_sets:
        total_num_patches = 0
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
        image_set_dir = os.path.join("usr", "data", 
                                    username, "image_sets",
                                    farm_name,
                                    field_name,
                                    mission_date)
        
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)

        try:
            patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])
        except Exception as e:
            patch_size = 416
        if "patch_overlap_percent" in image_set:
            patch_overlap_percent = image_set["patch_overlap_percent"]
        else:
            patch_overlap_percent = 50

        if "taken_regions" in image_set:
            for image_name in image_set["taken_regions"].keys():
                for region in image_set["taken_regions"][image_name]:

                    region_width = region[3] - region[1]
                    region_height = region[2] - region[0]

                    overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

                    incr = patch_size - overlap_px
                    w_covered = max(region_width - patch_size, 0)
                    num_w_patches = m.ceil(w_covered / incr) + 1

                    h_covered = max(region_height - patch_size, 0)
                    num_h_patches = m.ceil(h_covered / incr) + 1

                    num_patches = num_w_patches * num_h_patches

                    total_num_patches += num_patches #len(image_set["taken_regions"][image_name])
        else:
            # annotations = image_set_info[image_set_str]["annotations"]
            # image_shape = image_set_info[image_set_str]["image_shape"]
            # image_height = metadata["images"][list(annotations.keys())[0]]["height_px"]
            # image_width = metadata["images"][list(annotations.keys())[0]]["width_px"]
            # image_height = image_shape[0]
            # image_width = image_shape[1]
            
            num_patches_per_image = get_num_patches_per_image(annotations, metadata, patch_size, patch_overlap_percent=patch_overlap_percent)
            
            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    total_num_patches += num_patches_per_image


        print("{}: {} patches".format(image_set_str, total_num_patches))
        # all_totals.append(total_num_patches)
        totals[image_set_str] = total_num_patches


    # print("min: {}".format(np.min(all_totals)))
    # print("max: {}".format(np.max(all_totals)))

    return totals #np.min(all_totals)




def between_model(prev_model_dir, next_model_dir, num_to_take, model_name):

    prev_log_path = os.path.join(prev_model_dir, "log.json")
    prev_log = json_io.load_json(prev_log_path)

    next_log_path = os.path.join(next_model_dir, "log.json")
    next_log = json_io.load_json(next_log_path)

    next_taken = {}
    for image_set in next_log["image_sets"]:
        image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
        next_taken[image_set_str] = image_set["taken_regions"]

    prev_taken = {}
    tot_prev_taken = 0
    for image_set in prev_log["image_sets"]:
        image_set_str = image_set["username"] + " " + image_set["farm_name"] + " " + image_set["field_name"] + " " + image_set["mission_date"]
        prev_taken[image_set_str] = image_set["taken_regions"]
        for image_name in image_set["taken_regions"]:
            tot_prev_taken += len(image_set["taken_regions"][image_name])

    candidates = []
    for image_set_str in next_taken:
        for image_name in next_taken[image_set_str]:
            for region in next_taken[image_set_str][image_name]:
                if image_set_str not in prev_taken or image_name not in prev_taken[image_set_str] or region not in prev_taken[image_set_str][image_name]:
                    candidates.append((image_set_str, image_name, region))

    num_remaining_to_take = num_to_take - tot_prev_taken
    print("Previously took {}. Need to have: {}. Going to take {} additional regions.".format(
        tot_prev_taken, num_to_take, num_remaining_to_take
    ))

    picked = random.sample(candidates, num_remaining_to_take)

    taken = prev_taken
    for pick in picked:
        image_set_str = pick[0]
        if image_set_str not in taken:
            taken[image_set_str] = {}

        image_name = pick[1]
        if image_name not in taken[image_set_str]:
            taken[image_set_str][image_name] = []

        region = pick[2]
        taken[image_set_str][image_name].append(region)

    log = {}
    log["model_creator"] = "erik"
    log["model_name"] = model_name
    log["model_object"] = "canola_seedling"
    log["public"] = "yes"
    log["submission_time"] = int(time.time())
    log["image_sets"] = []
    for image_set_str in taken.keys():
        pieces = image_set_str.split(" ")
        username = pieces[0]
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]
        log["image_sets"].append({
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "taken_regions": taken[image_set_str],
            "patch_overlap_percent": 0
        })
    
    pending_model_path = os.path.join("usr", "data", "erik", "models", "pending", log["model_name"])

    os.makedirs(pending_model_path, exist_ok=False)
    
    log_path = os.path.join(pending_model_path, "log.json")
    json_io.save_json(log_path, log)
    




def copy_to_eval(image_sets):
    for image_set in image_sets:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        src_image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                     farm_name, field_name, mission_date)
        
        results = glob.glob(os.path.join(src_image_set_dir, "model", "results", "*"))
        print(results)
        
        # dst_image_set_dir = os.path.join("usr", "data", "eval", "image_sets",
        #                              farm_name, field_name, mission_date)
        
        # # os.makedirs(os.path.dirname(dst_image_set_dir), exist_ok=False)
        # shutil.copytree(src_image_set_dir, dst_image_set_dir)


def print_lex_sorted_image_sets(image_sets):

    values = []
    for image_set in image_sets:
        username = image_set["username"]
        farm_name = image_set["farm_name"]
        field_name = image_set["field_name"]
        mission_date = image_set["mission_date"]
        image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
        # image_set_strs.append(image_set_str)
        
        image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                     farm_name, field_name, mission_date)
        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        num_annotations = annotation_utils.get_num_annotations(annotations, ["test_regions"])
        num_test_images = 0
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                num_test_images += 1

        values.append((image_set_str, num_annotations, num_test_images))


    values = natsorted(values, key=lambda x: x[0])
    for v in values:
        image_set_str = v[0]
        num_annotations = v[1]
        num_images = v[2]
        pieces = image_set_str.split(" ")
        print(pieces[1] + " & " + pieces[2] + " & " + pieces[3] + 
              " & " + str(num_images) + " & " + str(num_annotations) + " \\\\\n\hline")


    total_num_images = np.sum([x[2] for x in values])
    total_num_annotations = np.sum([x[1] for x in values])

    print("total_num_images", total_num_images)
    print("total_num_annotations", total_num_annotations)




# def run_targeted_test(model_suffix, test_set, single_baselines):
#     results = compare_baseline.get_min_num_results([test_set], single_baselines, [])

#     test_set_str = list(results.keys())[0]
#     single_tuples = results[test_set_str]["single"]
#     diverse_tuples = results[test_set_str]["diverse"]

#     single_tuples.sort(key=lambda x: x[1])
#     # diverse_tuples.sort(key=lambda x: x[1])

#     for single_tuple in single_tuples:
#         print(single_tuple)

    
#     for i in range(1, len(single_tuples)):
#         model_name = "set_of_" + str(i) + model_suffix
#         targeted_training_image_sets = []
#         subset = single_tuples[:i]
#         for item in subset:
#             model_label = item[0]
#             pieces = model_label.split("/")
#             targeted_training_image_sets.append({
#                 "username": "kaylie",
#                 "farm_name": pieces[0],
#                 "field_name": pieces[1],
#                 "mission_date": pieces[2]
#             })

#         print("\n\n{}: running model.\n\n".format(i))

#         run_diverse_model(targeted_training_image_sets, model_name, 630, None, supplementary_weed_image_sets=None, run=True)

# def run_targeted_BlaineLake_Serhienko9S():
#     eval_single_630_baselines = [
#         "BlaineLake_River_2021-06-09_630_patches",
#         "BlaineLake_Lake_2021-06-09_630_patches",
#         "BlaineLake_HornerWest_2021-06-09_630_patches",
#         "UNI_LowN1_2021-06-07_630_patches",
#         "BlaineLake_Serhienko9N_2022-06-07_630_patches",
#         "row_spacing_nasser_2021-06-01_630_patches",
#         "Biggar_Dennis1_2021-06-04_630_patches",
#         "Biggar_Dennis3_2021-06-04_630_patches",
#         "MORSE_Dugout_2022-05-27_630_patches",
#         "MORSE_Nasser_2022-05-27_630_patches",
#         "row_spacing_brown_2021-06-01_630_patches",
#         "row_spacing_nasser2_2022-06-02_630_patches",
#         "Saskatoon_Norheim1_2021-05-26_630_patches",
#         "Saskatoon_Norheim2_2021-05-26_630_patches",
#         "Saskatoon_Norheim4_2022-05-24_630_patches",
#         "Saskatoon_Norheim5_2022-05-24_630_patches",
#         "UNI_Brown_2021-06-05_630_patches",
#         "UNI_Dugout_2022-05-30_630_patches",
#         "UNI_LowN2_2021-06-07_630_patches",
#         "UNI_Sutherland_2021-06-05_630_patches",
#         "Saskatoon_Norheim1_2021-06-02_630_patches",
#         "row_spacing_brown_2021-06-08_630_patches",
#         "SaskatoonEast_Stevenson5NW_2022-06-20_630_patches",
#         "UNI_Vaderstad_2022-06-16_630_patches",
#         "Biggar_Dennis2_2021-06-12_630_patches",
#         "BlaineLake_Serhienko10_2022-06-14_630_patches",
#         "BlaineLake_Serhienko9S_2022-06-14_630_patches"
#     ]
#     test_sets = [
#         {
#             "username": "eval",
#             "farm_name": "BlaineLake",
#             "field_name": "Serhienko9S",
#             "mission_date": "2022-06-07"
#         }
#     ]
#     single_baselines = []
#     for baseline in eval_single_630_baselines:

#         b_name = baseline[:len(baseline)-len("_630_patches")]
#         pieces = b_name.split("_")
#         label = "_".join(pieces[:-2]) + "/" + pieces[-2] + "/" + pieces[-1]


#         single_baselines.append({
#             "model_name": baseline,
#             "model_creator": "eval",
#             "model_label": label
#         })
    
#     model_suffix = "_target_BlaineLake_Serhienko9S_2022-06-07_630_patches_rep_0"
#     run_targeted_test(model_suffix, test_sets[0], single_baselines)



def dilation_plot(training_sets, sigma_vals, out_dir):
    # image_sets = {
    #     "training": training_sets,
    # }

    done = False
    # patches = {}

    image_set = random.choice(training_sets)
    # for key in image_sets.keys():
    #     patches[key] = []
        # for image_set in image_sets[key]:
    image_set_dir = os.path.join("usr", "data", image_set["username"], 
                                "image_sets", image_set["farm_name"], 
                                image_set["field_name"], image_set["mission_date"])
    
    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    
    annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)



    # max_contained = -1
    for image_name in annotations.keys():
        if len(annotations[image_name]["test_regions"]) > 0:
            image_w = metadata["images"][image_name]["width_px"]
            image_h = metadata["images"][image_name]["height_px"]

            patch_size = 416
            incr = patch_size #- overlap_px
            w_covered = max(image_w - patch_size, 0)
            num_w_patches = m.ceil(w_covered / incr) #+ 1

            h_covered = max(image_h - patch_size, 0)
            num_h_patches = m.ceil(h_covered / incr) #+ 1
            # candidates = []


            for i in range(0, num_w_patches):
                for j in range(0, num_h_patches):
                    patch_min_y = (patch_size) * j
                    patch_min_x = (patch_size) * i

                    patch_max_y = (patch_min_y + patch_size) #, image_height)
                    patch_max_x = (patch_min_x + patch_size) #, image_width)

                    patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                    num_contained = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_region]).size


                    if num_contained > 5:
                        chosen_patch_coords = patch_region
                        chosen_image = image_name
                        done = True
                        break

                    # if num_contained > max_contained:
                    #     max_contained = num_contained
                    #     chosen_patch_coords = patch_region
                    #     chosen_image = image_name

                if done:
                    break

        if done:
            break


            # print(max_contained)


    image_path = glob.glob(os.path.join(image_set_dir, "images", chosen_image + ".*"))[0]
    image = Image(image_path)
    image_array = image.load_image_array()

    patch = image_array[chosen_patch_coords[0]: chosen_patch_coords[2], chosen_patch_coords[1]: chosen_patch_coords[3]]

    # patches[key].append(patch)

    inds = box_utils.get_contained_inds(annotations[chosen_image]["boxes"], [chosen_patch_coords])
    sel_boxes = annotations[chosen_image]["boxes"][inds]
    sigma_to_boxes = {
        0: sel_boxes
    }
    for dilation_sigma in sigma_vals:

        boxes = np.copy(sel_boxes)

        for box in boxes:

            min_y_dilation = abs(round(random.gauss(0, dilation_sigma)))
            min_x_dilation = abs(round(random.gauss(0, dilation_sigma)))
            max_y_dilation = abs(round(random.gauss(0, dilation_sigma)))
            max_x_dilation = abs(round(random.gauss(0, dilation_sigma)))


            box[0] = max(0, box[0] - min_y_dilation)
            box[1] = max(0, box[1] - min_x_dilation)
            box[2] = min(image_h, box[2] + max_y_dilation)
            box[3] = min(image_w, box[3] + max_x_dilation)

        sigma_to_boxes[dilation_sigma] = boxes

    num_subplots = len(sigma_vals) + 1
    fig, axs = plt.subplots(1, num_subplots, figsize=(14, 6))

    for i in range(num_subplots):
        if i == 0:
            sigma = 0
        else:
            sigma = sigma_vals[i-1]

        boxes = sigma_to_boxes[sigma]
        print("boxes", boxes)


        out_array = np.copy(patch)

        # shapes = np.zeros_like(patch, np.uint8)
        for box in boxes:
            box[0] = box[0] - chosen_patch_coords[0]
            box[1] = box[1] - chosen_patch_coords[1]
            box[2] = box[2] - chosen_patch_coords[0]
            box[3] = box[3] - chosen_patch_coords[1]
            # cv2.rectangle(shapes, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), -1)
            cv2.rectangle(out_array, (max(box[1], 0), max(box[0], 0)),
                                 (min(box[3], out_array.shape[1]), min(box[2], out_array.shape[0])),
                                 (0, 132, 255), 2)
        # alpha = 0.25
        # mask = shapes.astype(bool)
        # out_array[mask] = cv2.addWeighted(patch, alpha, shapes, 1-alpha, 0)[mask]
        sigma_patch_path = os.path.join(out_dir, "sigma_" + str(sigma) + ".png")
        cv2.imwrite(sigma_patch_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))
        in_array = cv2.imread(sigma_patch_path, cv2.IMREAD_UNCHANGED)
        if in_array.ndim == 3:
            in_array = cv2.cvtColor(in_array, cv2.COLOR_BGR2RGB)

        axs[i].set_title("$\sigma = " + str(sigma) + "$") #"Training Samples", fontsize=50, fontweight=50)
        axs[i].imshow(in_array)
        axs[i].axis("off")


    plt.suptitle("Expanding Annotations By Varying Amounts")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dilated_annotations.png")
    plt.savefig(out_path)


def create_patch_sample_plot(training_sets, test_sets, out_dir):
    image_sets = {
        "training": training_sets,
        "test": test_sets,
    }

    patches = {}
    for key in image_sets.keys():
        patches[key] = []
        for image_set in image_sets[key]:
            image_set_dir = os.path.join("usr", "data", image_set["username"], 
                                        "image_sets", image_set["farm_name"], 
                                        image_set["field_name"], image_set["mission_date"])
            
            metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)
            
            annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
            annotations = annotation_utils.load_annotations(annotations_path)



            max_contained = -1
            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    image_w = metadata["images"][image_name]["width_px"]
                    image_h = metadata["images"][image_name]["height_px"]

                    patch_size = 416
                    incr = patch_size #- overlap_px
                    w_covered = max(image_w - patch_size, 0)
                    num_w_patches = m.ceil(w_covered / incr) #+ 1

                    h_covered = max(image_h - patch_size, 0)
                    num_h_patches = m.ceil(h_covered / incr) #+ 1
                    # candidates = []


                    for i in range(0, num_w_patches):
                        for j in range(0, num_h_patches):
                            patch_min_y = (patch_size) * j
                            patch_min_x = (patch_size) * i

                            patch_max_y = (patch_min_y + patch_size) #, image_height)
                            patch_max_x = (patch_min_x + patch_size) #, image_width)

                            patch_region = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                            num_contained = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_region]).size

                            if num_contained > max_contained:
                                max_contained = num_contained
                                chosen_patch_coords = patch_region
                                chosen_image = image_name

                            # all_patch_candidates.append((image_set_str, image_name, patch_region))



                    # if len(annotations[image_name]["boxes"]) > 0:
                    #     candidates.append(image_name)

            # print(image_set_dir, candidates)
            # sel_image_name = random.sample(candidates, 1)[0]


            # sel_ind = random.sample(np.arange(len(annotations[sel_image_name]["boxes"])).tolist(), 1)[0]
            # sel_box = annotations[sel_image_name]["boxes"][sel_ind]
            # box_centre_y = (sel_box[0] + sel_box[2]) / 2
            # box_centre_x = (sel_box[1] + sel_box[3]) / 2

            # patch_min_y = m.floor(box_centre_y / 416) * 416
            # patch_min_x = m.floor(box_centre_x / 416) * 416

            # patch_max_y = patch_min_y + 416
            # patch_max_x = patch_min_x + 416
            
            # if patch_max_y > image_h:
            #     patch_max_y = image_h
            #     patch_min_y = patch_max_y - 416
            # if patch_max_x > image_w:
            #     patch_max_x = image_w
            #     patch_min_x = patch_max_x - 416
            print(max_contained)


            image_path = glob.glob(os.path.join(image_set_dir, "images", chosen_image + ".*"))[0]
            image = Image(image_path)
            image_array = image.load_image_array()

            patch = image_array[chosen_patch_coords[0]: chosen_patch_coords[2], chosen_patch_coords[1]: chosen_patch_coords[3]] #patch_min_y:patch_max_y, patch_min_x:patch_max_x]
            # box_utils.get_contained_inds(annotations[image_name]["boxes"], [chosen_patch_coords])
            


            patches[key].append(patch)

    # 
    # f, axarr = plt.subplots(3, 9, figsize=(16, 8))

    # f = plt.figure(figsize=(16, 8))
    # f, axarr = = plt.subplots(2, 1, figsize=(16, 8))
    # gs1 = gridspec.GridSpec(4, 4)
    # gs1.update(wspace=0.025, hspace=0.05)
    # for i in range(27):
    #     print(i, i // 9, i % 9)
    #     # ax = axarr[i // 9, i % 9]
    #     ax = plt.subplot(gs1[i])
    #     axarr[i].imshow(patches["training"][i])
    #     axarr[i].axis("off")

    samples_path = os.path.join("eval_charts", "patch_samples")

    training_samples_dir = os.path.join(samples_path, "training")
    test_samples_dir = os.path.join(samples_path, "test")

    os.makedirs(training_samples_dir, exist_ok=True)
    os.makedirs(test_samples_dir, exist_ok=True)
    for i in range(len(patches["training"])):
        cv2.imwrite(os.path.join(training_samples_dir, str(i) + ".png"), 
                    cv2.cvtColor(patches["training"][i], cv2.COLOR_RGB2BGR))
    for i in range(len(patches["test"])):
        cv2.imwrite(os.path.join(test_samples_dir, str(i) + ".png"), 
                    cv2.cvtColor(patches["test"][i], cv2.COLOR_RGB2BGR))
        
    exit()


    # plt.subplots_adjust(wspace=0, hspace=0) #tight_layout()
    spacing_amount = 10
    training_image_mosaic = np.full((3*416 + 2*spacing_amount, 9*416 + 8*spacing_amount, 3), 255, dtype=np.uint8)

    for i in range(27):
        r_i = i // 9
        c_i = i % 9
        training_image_mosaic[r_i * 416 + (spacing_amount * r_i): (r_i * 416) + 416 + (spacing_amount * r_i), 
                              c_i * 416 + (spacing_amount * c_i): (c_i * 416) + 416 + (spacing_amount * c_i)] = patches["training"][i]


    test_image_mosaic = np.full((416, 9*416 + 8*spacing_amount, 3), 255, dtype=np.uint8)
    for i in range(9):
        # r_i = i // 9
        # c_i = i % 9
        test_image_mosaic[0: 416, i * 416 + (spacing_amount * i): (i * 416) + 416 + (spacing_amount * i)] = patches["test"][i]






    # fig = plt.figure(figsize=(16, 8))
    f, axarr = plt.subplots(2, 1, figsize=(36, 18), gridspec_kw={'height_ratios': [3, 1]})
    # hfont = {'fontname':'Helvetica'}
    axarr[0].set_title("Training Samples", fontsize=50, fontweight=50)
    axarr[0].imshow(training_image_mosaic)
    axarr[0].axis("off")

    axarr[1].set_title("Test Samples", fontsize=50, fontweight=50)
    axarr[1].imshow(test_image_mosaic)
    axarr[1].axis("off")

    # plt.suptitle("Image Set Sample Patches", fontsize=60)
    plt.tight_layout()

    # gs0 = gridspec.GridSpec(2, 1, figure=fig)

    # gs_training = gridspec.GridSpecFromSubplotSpec(3, 9, subplot_spec=gs0[0])
    # gs_test = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=gs0[0])


    # for i in range(27):
    #     ax = fig.add_subplot(gs_training[i])
    #     ax.imshow(patches["training"][i])
    #     ax.axis("off")

    # for i in range(9):
    #     ax = fig.add_subplot(gs_test[i])
    #     ax.imshow(patches["test"][i])
    #     ax.axis("off")



    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_patches.png")
    plt.savefig(out_path)

    
def rsync_from_z420(training_image_sets):


    for ims in training_image_sets:

        z420_path = "erik@z420:/home/erik/d1/plant_detection/plant_detection/src/usr/data/" + ims["username"] + "/image_sets/" + \
                        ims["farm_name"] + "/" + ims["field_name"] + "/" + ims["mission_date"] + "/annotations/annotations.json"
        
        asus_path = "/home/erik/Documents/plant_detection/plant_detection/src/usr/data/" + ims["username"] + "/image_sets/" + \
                        ims["farm_name"] + "/" + ims["field_name"] + "/" + ims["mission_date"] + "/annotations/annotations.json"

        cmd = "rsync -av -i " + z420_path + " " + asus_path

        os.system(cmd)




if __name__ == "__main__":

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()


    weed_image_sets = [
        {
            "username": "erik",
            "farm_name": "CottonWeedDet12",
            "field_name": "CottonWeedDet12",
            "mission_date": "2023-04-10"
        },
        # {
        #     "username": "erik",
        #     "farm_name": "blue_lupins",
        #     "field_name": "blue_lupins",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "turnipweed_in_wheat",
        #     "field_name": "turnipweed_in_wheat",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "chickpea_BFLY",
        #     "field_name": "chickpea_BFLY",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "chickpea_BFLYS",
        #     "field_name": "chickpea_BFLYS",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "20200827_wheat_BFLY",
        #     "field_name": "20200827_wheat_BFLY",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "20201014_wheat_BFLY",
        #     "field_name": "20201014_wheat_BFLY",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "wheat_BFLYS",
        #     "field_name": "wheat_BFLYS",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "radish_in_wheat",
        #     "field_name": "radish_in_wheat",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "narrabri_wheat",
        #     "field_name": "narrabri_wheat",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "broadleaf_weeds",
        #     "field_name": "broadleaf_weeds",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "sustainable_ag_2021",
        #     "field_name": "sustainable_ag_2021",
        #     "mission_date": "2023-04-11"
        # },
        # {
        #     "username": "erik",
        #     "farm_name": "wild_carrot_canola",
        #     "field_name": "wild_carrot_canola",
        #     "mission_date": "2023-04-11"
        # },      
    ]

    # remove_previous_iter_results(training_image_sets)

    # run_my_random_test(training_image_sets, run_models=False)
    # run_my_active_test(training_image_sets)

    # training_image_sets.extend(weed_image_sets)


    # run_weed_test(training_image_sets, "MORSE_Nasser_2022-05-27_and_WeedAI")

    # run_weed_test(training_image_sets, weed_image_sets, "MORSE_Nasser_2022-05-27_and_10000_weed", 10000)


    # run_full_image_set_model(training_image_sets, "fixed_epoch_set_of_12_no_overlap")



    # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "pending", "fixed_epoch_set_of_27_no_overlap")
    # prev_model_dir = os.path.join("usr", "data", "erik", "models", "pending", "fixed_epoch_diverse_set_of_27_match_18_no_overlap")

    # run_diverse_model(training_image_sets, "fixed_epoch_diverse_set_of_27_match_27_no_overlap", model_dir_to_match, prev_model_dir)


    # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_MORSE_Nasser_2022-05-27_no_overlap")

    # exg_active_patch_selection(training_image_sets, "fixed_epoch_exg_active_match_1_no_overlap", model_dir_to_match, prev_model_dir=None)


    # model_dir = os.path.join("usr", "data", "erik", "models", "pending", "fixed_epoch_diverse_set_of_27_match_12_no_overlap")
    # run_pending_model(model_dir)


    # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_diverse_set_of_27_match_3_no_overlap")

    # prev_model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_exg_active_match_1_no_overlap")

    # exg_active_patch_selection(training_image_sets, "fixed_epoch_exg_active_match_3_no_overlap", model_dir_to_match, prev_model_dir)


    
    # for i in range(15, 20):
    #     run_single_and_diverse_test(training_image_sets[i], training_image_sets)


    # min_num = get_min_patch_num_of_sets(training_image_sets)

    # model_name = training_image_set["farm_name"] + "_" + training_image_set["field_name"] + "_" + training_image_set["mission_date"]
    # # full_model_name = "fixed_epoch_min_num_" + model_name + "_no_overlap"

    # # run_full_image_set_model([training_image_set], full_model_name)
    # run_diverse_model([training_image_set], "fixed_epoch_min_num_diverse_and_CottonWeedDet_no_overlap", min_num, 
    # prev_model_dir=None, supplementary_weed_image_sets=weed_image_sets)
    # # model_dir_to_match = os.path.join("usr", "data", "erik", "models", "available", "public", full_model_name)
    # # run_diverse_model(all_training_image_sets, "fixed_epoch_min_num_diverse_set_of_27_match_" + model_name + "_no_overlap", min_num, prev_model_dir=None)



    # run_single_and_diverse_test(training_image_sets[26], training_image_sets)

    # patch_nums = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 38891]
    # for i in [5]: #range(len(patch_nums)):
    #     prev_model_dir = None
    #     model_name = "fixed_epoch_fixed_patch_num_" + str(patch_nums[i]) + "_no_overlap"
    #     model_dir = os.path.join("usr", "data", "erik", "models", "pending", model_name)
    #     # if i > 0:
    #     #     prev_model_name = "fixed_epoch_fixed_patch_num_" + str(patch_nums[i-1]) + "_no_overlap"
    #     #     prev_model_dir = os.path.join("usr", "data", "erik", "models", "pending", prev_model_name)
    #     # run_diverse_model(training_image_sets, model_name, patch_nums[i], prev_model_dir, supplementary_weed_image_sets=None, run=False)


    #     run_pending_model(model_dir)

    # prev_model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_fixed_patch_num_16000_no_overlap")
    # next_model_dir = os.path.join("usr", "data", "erik", "models", "available", "public", "fixed_epoch_fixed_patch_num_32000_no_overlap")
    # num_to_take = 24000
    # model_name = "fixed_epoch_fixed_patch_num_24000_no_overlap"
    # between_model(prev_model_dir, next_model_dir, num_to_take, model_name)

    # min_num = get_min_patch_num_of_sets(training_image_sets)

    # run_diverse_model(training_image_sets, "set_of_27_630_patches_rep_2", 630, prev_model_dir=None, supplementary_weed_image_sets=None, run=True)


    # patch_nums = get_patch_nums_of_sets(training_image_sets)
    # tot = 0
    # for image_set in training_image_sets:
    #     username = image_set["username"]
    #     farm_name = image_set["farm_name"]
    #     field_name = image_set["field_name"]
    #     mission_date = image_set["mission_date"]
    #     image_set_str = username + " " + farm_name + " " + field_name + " " + mission_date
    #     tot += patch_nums[image_set_str]
    #     # run_single_image_set_diverse_test(training_image_sets[i], training_image_sets, 1500)
    # print(tot)

    # run_single_image_set_diverse_test(training_image_sets[i], training_image_sets, 1500)
    # run_diverse_model(training_image_sets, "set_of_27_1500_patches_rep_0", 1500, prev_model_dir=None, supplementary_weed_image_sets=None, run=True)


    # perturbation_amounts = [10, 30, 50, 100]
    # annotation_perturbation_test(training_image_sets, perturbation_amounts, 16000)




    # for i in range(15, 16): #len(training_image_sets)):
    #     run_single_image_set_diverse_test(training_image_sets[i], 
    #         training_image_sets, 
    #         630, 
    #         supplementary_weed_image_sets=weed_image_sets)
        
    # for sz in [1000, 2000, 4000]: #16000]:
    #     run_diverse_model(training_image_sets, "set_of_27_" + str(sz) + "_patches_rep_3", sz, None, supplementary_weed_image_sets=None, run=True)

    # for sz in [1000, 2000, 4000]: #16000]:
    #     run_diverse_model(training_image_sets, "set_of_27_" + str(sz) + "_patches_rep_4", sz, None, supplementary_weed_image_sets=None, run=True)
    
    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_2")
    # annotation_dilation_test(training_image_sets, [28], 16000, matched_dir, 2)
    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_3")
    # annotation_dilation_test(training_image_sets, [28], 16000, matched_dir, 3)


    # run_diverse_multi(training_image_sets, 10, 3150, run=True)

    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_1")
    # annotation_dilation_test(training_image_sets, [30], 16000, matched_dir, 1)
    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_3")
    # annotation_dilation_test(training_image_sets, [18], 16000, matched_dir, 3)
    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_4")
    # annotation_dilation_test(training_image_sets, [30], 16000, matched_dir, 4)
    
    # rsync_from_z420(training_image_sets)
    # restore_annotations(training_image_sets)

    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_4")
    # annotation_dilation_test(training_image_sets, [24], 16000, matched_dir, 4)


    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_0")
    # annotation_dilation_test(training_image_sets, [26], 16000, matched_dir, 0)
    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_1")
    # annotation_dilation_test(training_image_sets, [26], 16000, matched_dir, 1)
    # matched_dir = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_16000_patches_rep_2")
    # annotation_dilation_test(training_image_sets, [26], 16000, matched_dir, 2)


    # run_smaller_diverse_model(training_image_sets, "row_spacing_brown_2021-06-01_250_patches_800_epochs_rep_0", 250, 
    #                           "usr/data/eval/models/available/public/row_spacing_brown_2021-06-01_630_patches_rep_0", 
    #                           supplementary_weed_image_sets=None, run=True)
    # run_smaller_diverse_model(training_image_sets, "row_spacing_brown_2021-06-01_250_patches_rep_1", 250, 
    #                           "usr/data/eval/models/available/public/row_spacing_brown_2021-06-01_630_patches_rep_1", 
    #                           supplementary_weed_image_sets=None, run=True)
    # run_smaller_diverse_model(training_image_sets, "row_spacing_brown_2021-06-01_250_patches_rep_2", 250, 
    #                           "usr/data/eval/models/available/public/row_spacing_brown_2021-06-01_630_patches_rep_2", 
    #                           supplementary_weed_image_sets=None, run=True)
    # run_smaller_diverse_model(training_image_sets, "row_spacing_brown_2021-06-01_250_patches_rep_3", 250, 
    #                           "usr/data/eval/models/available/public/row_spacing_brown_2021-06-01_630_patches_rep_3", 
    #                           supplementary_weed_image_sets=None, run=True)
    # run_smaller_diverse_model(training_image_sets, "row_spacing_brown_2021-06-01_250_patches_rep_4", 250, 
    #                           "usr/data/eval/models/available/public/row_spacing_brown_2021-06-01_630_patches_rep_4", 
    #                           supplementary_weed_image_sets=None, run=True)

    # sz = 2000
    # run_diverse_model(ns_training_image_sets, "ns_set_of_27_" + str(sz) + "_patches_rep_0", sz, None, supplementary_weed_image_sets=None, run=True)

    best_to_worst_ordering = [
        15, 19, 24, 12, 17, 11, 4, 20, 9, 22, 23, 10, 6, 5, 7, 2, 8, 18, 3, 14, 21, 25, 13, 1, 16, 0, 26
    ]
    # num_patches_to_match = 250
    rep_num = 1
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_250_patches_rep_" + str(rep_num), 250, None, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)
    
    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_250_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_500_patches_rep_" + str(rep_num), 500, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)
    
    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_500_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_1000_patches_rep_" + str(rep_num), 1000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)
    
    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_1000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_2000_patches_rep_" + str(rep_num), 2000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)
    
    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_2000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_4000_patches_rep_" + str(rep_num), 4000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)
    
    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_4000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_8000_patches_rep_" + str(rep_num), 8000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)

    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_8000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_16000_patches_rep_" + str(rep_num), 16000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)

    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_16000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_24000_patches_rep_" + str(rep_num), 24000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)

    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_24000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_32000_patches_rep_" + str(rep_num), 32000, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)

    # prev_model_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_32000_patches_rep_1")
    # run_ordered_selection_model(training_image_sets, "set_of_27_best_to_worst_38891_patches_rep_" + str(rep_num), 38891, prev_model_dir, best_to_worst_ordering, supplementary_weed_image_sets=None, run=False)


    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_250_patches_rep_1")
    # run_pending_model(m_dir)

    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_500_patches_rep_1")
    # run_pending_model(m_dir)

    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_1000_patches_rep_1")
    # run_pending_model(m_dir)

    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_2000_patches_rep_1")
    # run_pending_model(m_dir)

    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_4000_patches_rep_1")
    # run_pending_model(m_dir)

    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_8000_patches_rep_1")
    # run_pending_model(m_dir)

    # m_dir = os.path.join("usr", "data", "eval", "models", "pending", "set_of_27_best_to_worst_16000_patches_rep_1")
    # run_pending_model(m_dir)

    training_image_sets = ds_splits.get_split("EARLY", "TRAIN")
    num_patches = ds_splits.get_num_patches_for_split("EARLY", "TRAIN")
    # training_image_sets = ds_splits.get_split("HIWEED", "TRAIN")
    # num_patches = ds_splits.get_num_patches_for_split("HIWEED", "TRAIN")
    # training_image_sets = ds_splits.get_split("MANLOWEED", "TRAIN")
    # num_patches = ds_splits.get_num_patches_for_split("MANLOWEED", "TRAIN")



    print(training_image_sets)
    print("num_patches", num_patches)

    # num_patches = ds_splits.get_num_patches_for_split("LATE", "TRAIN")

    # print("num_patches", num_patches)

    for rep_num in range(1, 5):
        run_ds_model(training_image_sets, 
                    "EARLY_" + str(num_patches) + "_patches_rep_" + str(rep_num),  
                    num_patches, 
                    run=True)

    # m_dir = os.path.join("usr", "data", "ds_splits", "models", "pending", "EARLY_" + str(31500) + "_patches_rep_0")
    # run_pending_model(m_dir)


    # run_diverse_model(training_image_sets, "set_of_27_3906_patches_rep_3", 3906,
    #                     "usr/data/eval/models/available/public/set_of_27_2000_patches_rep_3", None, True)
    # run_diverse_model(training_image_sets, "set_of_27_3906_patches_rep_4", 3906,
    #                     "usr/data/eval/models/available/public/set_of_27_2000_patches_rep_4", None, True)


    
    # log_path = os.path.join(matched_dir, "log.json")
    # log = json_io.load_json(log_path)
    # annotation_dilation_test(training_image_sets, [10], 16000, matched_dir) #, taken_regions=log["taken_regions"]) #num_patches_to_take=16000)
    # annotation_dilation_test(training_image_sets, [12], 16000, matched_dir)
    # annotation_dilation_test(training_image_sets, [14], 16000, matched_dir)


    # annotation_removal_test(training_image_sets, 0.05, 16000, matched_dir)

    # dilation_plot(training_image_sets, [5, 10], "dilation_plot")
    # create_patch_sample_plot(training_image_sets, test_image_sets, "sample_patches")
    # model_name = "UNI_Dugout_2022-05-30_630_patches_BlaineLake_Serhienko9S_2022-06-14_630_patches_rep_1"
    # model_name = "BlaineLake_River_2021-06-09_630_patches_BlaineLake_Serhienko9S_2022-06-14_630_patches_rep_0"
    # model_name = "set_of_5_target_BlaineLake_Serhienko9S_2022-06-07_630_patches_rep_0"
    # targeted_training_image_sets = [
    #     training_image_sets[15],
    #     training_image_sets[19],
    #     training_image_sets[17],
    #     training_image_sets[12],
    #     training_image_sets[24]
    # ]
    # run_diverse_model(targeted_training_image_sets, model_name, 630, None, supplementary_weed_image_sets=None, run=True)
    # run_targeted_BlaineLake_Serhienko9S()
    
    # run_diverse_model_2([training_image_sets[15], training_image_sets[-1]], model_name, [630, 630], run=True)


    # for sz in [4000]: #[250, 500] :#, 1000, 2000, 4000]:
    #     model_dir_to_match = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_" + str(sz) + "_patches_rep_1")
    #     exg_zero_anno_replacement(training_image_sets, "set_of_27_exg_repl_" + str(sz) + "_patches_rep_1" , model_dir_to_match, run=True)

    # for sz in [2000]: #[250, 500] :#, 1000, 2000, 4000]:
    #     model_dir_to_match = os.path.join("usr", "data", "eval", "models", "available", "public", "set_of_27_" + str(sz) + "_patches_rep_2")
    #     exg_zero_anno_replacement(training_image_sets, "set_of_27_exg_repl_" + str(sz) + "_patches_rep_2" , model_dir_to_match, run=True)


    # copy_to_eval(training_image_sets)
    # print_lex_sorted_image_sets(training_image_sets)