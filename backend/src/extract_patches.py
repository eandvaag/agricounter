import logging
import os
import glob
import random
import math as m
import numpy as np
import cv2
from osgeo import gdal
from joblib import Parallel, delayed


from models.common import box_utils, annotation_utils
from io_utils import tf_record_io, json_io
from image_set import Image

DEFAULT_PATCH_SIZE = 300



def update_model_patch_size(image_set_dir, annotations, region_keys):

    status_path = os.path.join(image_set_dir, "model", "status.json")
    status = json_io.load_json(status_path)
    updated_patch_size = status["patch_size"]

    try:
        updated_patch_size = annotation_utils.get_patch_size(annotations, region_keys)
        status["patch_size"] = updated_patch_size
        json_io.save_json(status_path, status)
    except RuntimeError:
        pass

    return updated_patch_size




def update_training_patches(image_set_dir, annotations, updated_patch_size):

    images_dir = os.path.join(image_set_dir, "images")
    model_dir = os.path.join(image_set_dir, "model")
    patches_dir = os.path.join(model_dir, "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")


    patch_data = {}


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    is_ortho = metadata["is_ortho"] == "yes"

    for image_name in annotations.keys():

        if len(annotations[image_name]["fine_tuning_regions"]) > 0:
            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = Image(image_path)
            patch_records = extract_patch_records_from_image_tiled(
                image, 
                updated_patch_size,
                image_annotations=None,
                patch_overlap_percent=50,
                include_patch_arrays=False,
                regions=annotations[image_name]["fine_tuning_regions"],
                is_ortho=is_ortho,
                out_dir=patches_dir)

            patch_data[image_name] = patch_records

    json_io.save_json(patch_data_path, patch_data)






def write_annotated_patch_records(patch_records, patch_dir, includes_patch_arrays=True):

    if includes_patch_arrays:
        write_patches(patch_dir, patch_records)
    annotated_tf_records = tf_record_io.create_patch_tf_records(patch_records, patch_dir, is_annotated=True)
    annotated_patches_record_path = os.path.join(patch_dir, "annotated-patches-record.tfrec")
    tf_record_io.output_patch_tf_records(annotated_patches_record_path, annotated_tf_records)



def write_patches(out_dir, patch_records):
    Parallel(os.cpu_count())(
        delayed(cv2.imwrite)(os.path.join(out_dir, patch_record["patch_name"]), 
                             cv2.cvtColor(patch_record["patch"], cv2.COLOR_RGB2BGR)) for patch_record in patch_records)

    # for patch_record in patch_records:
    #     cv2.imwrite(os.path.join(out_dir, patch_record["patch_name"]),
    #                 cv2.cvtColor(patch_record["patch"], cv2.COLOR_RGB2BGR))


def add_annotations_to_patch_records(patch_records, image_annotations):
    annotation_boxes = image_annotations["boxes"]
    annotation_classes = image_annotations["classes"]

    for patch_record in patch_records:
        annotate_patch(patch_record, annotation_boxes, annotation_classes)



def extract_random_patches(image_path, num_patches, patch_size, is_ortho):

    image = Image(image_path)

    if is_ortho:
        ds = gdal.Open(image.image_path)
    else:
        image_array = image.load_image_array()

    w, h = image.get_wh()

    patches = []
    for _ in range(num_patches):

        min_y = random.randrange(0, h - patch_size)
        min_x = random.randrange(0, w - patch_size)

        max_y = min_y + patch_size
        max_x = min_x + patch_size

        patch_data = {}
        patch_data["patch_coords"] = [min_y, min_x, max_y, max_x]

        if is_ortho:
            # box_array = ds.ReadAsArray(box[1], box[0], (box[3]-box[1]), (box[2]-box[0]))
            patch_array = ds.ReadAsArray(min_x, min_y, (max_x-min_x), (max_y-min_y))
            patch_array = np.transpose(patch_array, (1, 2, 0))
        else:
            # box_array = image_array[box[0]:box[2], box[1]:box[3]]
            patch_array = image_array[min_y:max_y, min_x:max_x]

        patch_data["patch"] = patch_array
        patches.append(patch_data)

    return patches

def extract_box_image_areas(image_path, boxes):

    image = Image(image_path)
    image_array = image.load_image_array()
    patches = []
    for box in boxes:
        box_array = image_array[box[0]:box[2], box[1]:box[3]]
        resized = cv2.resize(box_array, (64, 64), interpolation=cv2.INTER_AREA)
        patch_data = {}
        patch_data["patch"] = resized #box_array



        patches.append(patch_data)
    return patches

def extract_box_patches(image_path, boxes, patch_size, is_ortho):

    image = Image(image_path)

    if is_ortho:
        ds = gdal.Open(image.image_path)
    else:
        image_array = image.load_image_array()

    # w, h = image.get_wh()

    patches = []
    for box in boxes:
        centre_y = round((box[0] + box[2]) / 2)
        centre_x = round((box[1] + box[3]) / 2)

        min_y = max(centre_y - round((patch_size / 2)), 0)
        min_x = max(centre_x - round((patch_size / 2)), 0)
        max_y = min_y + patch_size
        max_x = min_x + patch_size

        # min_y = max(box[0] - 20, 0)
        # min_x = max(box[1] - 20, 0)
        # max_y = min(box[2] + 20, h)
        # max_x = min(box[3] + 20, w)

        patch_data = {}
        # patch_data["image_name"] = image.image_name
        # patch_data["image_path"] = image.image_path
        # patch_data["patch_name"] = username + "-" + farm_name + "-" + field_name + "-" + mission_date + "-" + \
                                # image.image_name + "-" + str(patch_num).zfill(7) + ".png"
        patch_data["patch_coords"] = [min_y, min_x, max_y, max_x]
        # patch_data["patch_content_coords"] = [min_y, min_x, max_y, max_x]

        if is_ortho:
            # box_array = ds.ReadAsArray(box[1], box[0], (box[3]-box[1]), (box[2]-box[0]))
            box_array = ds.ReadAsArray(min_x, min_y, (max_x-min_x), (max_y-min_y))
            box_array = np.transpose(box_array, (1, 2, 0))
        else:
            # box_array = image_array[box[0]:box[2], box[1]:box[3]]
            box_array = image_array[min_y:max_y, min_x:max_x]

        patch_data["box"] = box
        patch_data["patch"] = box_array
        patches.append(patch_data)

    return patches




def extract_patch_records_from_image_tiled(image, 
                                           patch_size, 
                                           image_annotations=None,
                                           #    class_mapping=None,
                                           patch_overlap_percent=50, 
                                           include_patch_arrays=True, 
                                           regions="all",
                                           is_ortho=False,
                                           out_dir=None):

    if image_annotations is not None:
        # annotation_status = image_annotations["status"]
        annotation_boxes = image_annotations["boxes"]
        annotation_classes = image_annotations["classes"]

    image_patches = []

    # load_on_demand = True
    w, h = image.get_wh()
    if include_patch_arrays or out_dir is not None:
        # available_bytes = psutil.virtual_memory()[1]
        # if available_bytes > w * h * 3:
        #     logger.info("Can load image into memory!")
        if is_ortho: # load_on_demand:
            ds = gdal.Open(image.image_path)
        else:
            image_array = image.load_image_array()
            # h, w = image_array.shape[:2]
    else:
        w, h = image.get_wh()

    tile_size = patch_size
    overlap_px = int(m.floor(tile_size * (patch_overlap_percent / 100)))
    
    patch_num = 0

    image_path_pieces = image.image_path.split("/")
    username = image_path_pieces[-7]
    farm_name = image_path_pieces[-5]
    field_name = image_path_pieces[-4]
    mission_date = image_path_pieces[-3]

    if regions == "all":
        regions = [[0, 0, h, w]]


    for region in regions:

        col_covered = False
        patch_min_y = region[0] #0
        while not col_covered:
            patch_max_y = patch_min_y + tile_size
            max_content_y = patch_max_y
            if patch_max_y >= region[2]: # h:
                max_content_y = region[2] #h
                
                # patch_min_y = h - tile_size
                # patch_max_y = h
                col_covered = True

            row_covered = False
            patch_min_x = region[1] #0
            while not row_covered:

                patch_max_x = patch_min_x + tile_size
                max_content_x = patch_max_x
                if patch_max_x >= region[3]: #w:
                    max_content_x = region[3] #w
                    
                    # patch_min_x = w - tile_size
                    # patch_max_x = w
                    row_covered = True

                
                # patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                if (include_patch_arrays or out_dir is not None) and is_ortho: #load_on_demand:
                    image_array = ds.ReadAsArray(patch_min_x, patch_min_y, (max_content_x-patch_min_x), (max_content_y-patch_min_y))
                    image_array = np.transpose(image_array, (1, 2, 0))
                    # yoff=patch_min_y, xoff=patch_min_x, win_xsize=(max_content_x-patch_min_x), win_ysize=(max_content_y-patch_min_y))
                # print("patch_coords", patch_coords)

                patch_data = {}
                patch_data["image_name"] = image.image_name
                patch_data["image_path"] = image.image_path
                patch_data["patch_name"] = username + "-" + farm_name + "-" + field_name + "-" + mission_date + "-" + \
                                        image.image_name + "-" + str(patch_num).zfill(7) + ".png"
                patch_data["patch_coords"] = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]
                patch_data["patch_content_coords"] = [patch_min_y, patch_min_x, max_content_y, max_content_x]

                if include_patch_arrays or out_dir is not None:
                    patch_array = np.zeros(shape=(patch_size, patch_size, 3), dtype=np.uint8)
                    if is_ortho:
                        patch_array[0:(max_content_y-patch_min_y), 0:(max_content_x-patch_min_x)] = image_array
                    else:
                        patch_array[0:(max_content_y-patch_min_y), 0:(max_content_x-patch_min_x)] = image_array[patch_min_y:max_content_y, patch_min_x:max_content_x]
                    
                    # patch_array = image_array[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                    if out_dir is not None:
                        patch_data["patch_path"] = os.path.join(out_dir, patch_data["patch_name"])
                        cv2.imwrite(patch_data["patch_path"], 
                                    cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR))
                    if include_patch_arrays:
                        patch_data["patch"] = patch_array

                    # patch_data["patch"] = patch_array
                    # print("patch_array.shape", patch_array.shape)

                
                if image_annotations is not None:
                    annotate_patch(patch_data, annotation_boxes, annotation_classes) #, class_mapping=class_mapping)
                    # if is_ortho:
                    #     # FIX
                    #     pass

                    # else:
                    #     if (image_annotations["status"] == "completed_for_training" or image_annotations["status"] == "completed_for_testing"):
                    #         annotate_patch(patch_data, annotation_boxes) #, annotation_classes)
                image_patches.append(patch_data)
                patch_num += 1
                
                patch_min_x += (tile_size - overlap_px)

            patch_min_y += (tile_size - overlap_px)

    return image_patches








def annotate_patch(patch_data, gt_boxes, gt_classes): #, class_mapping=None): #, clip_coords=None):

    if gt_boxes.size == 0:
        patch_data["image_abs_boxes"] = []
        patch_data["patch_abs_boxes"] = []
        patch_data["patch_normalized_boxes"] = []
        patch_data["patch_classes"] = [] 

    else:

        #patch = patch_data["patch"]
        patch_coords = patch_data["patch_coords"]
        patch_content_coords = patch_data["patch_content_coords"]

        patch_size = patch_coords[2] - patch_coords[0]

        # centres = np.rint((gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.0).astype(np.int64)

        # print("now processing", patch_data["patch_name"])

        # if clip_coords is None:
        contained_inds = box_utils.get_contained_inds(gt_boxes, [patch_content_coords])
            #contained_inds = get_contained_inds(centres, patch_data["patch_coords"])
        # else:
            # contained_inds = get_contained_inds_2(gt_boxes, clip_coords)
            #contained_inds = get_contained_inds(centres, clip_coords)

        # print("num_contained_boxes", contained_inds.size)
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        # if clip_coords is None:
        
        image_abs_boxes, mask = box_utils.clip_boxes_and_get_small_visibility_mask(
            contained_boxes, patch_content_coords, min_visibility=0.15)

        # else:
        #     image_abs_boxes, mask = box_utils.clip_boxes_and_get_small_visibility_mask(
        #         contained_boxes, clip_coords, min_visibility=0.15)


        # ---- CLIP ----
        image_abs_boxes = image_abs_boxes[mask]
        contained_classes = contained_classes[mask]
        # ---- NO CLIP ----
        # image_abs_boxes = contained_boxes[mask]

        # contained_classes = contained_classes[mask]

        patch_abs_boxes = np.stack([image_abs_boxes[:,0] - patch_content_coords[0],
                                    image_abs_boxes[:,1] - patch_content_coords[1],
                                    image_abs_boxes[:,2] - patch_content_coords[0],
                                    image_abs_boxes[:,3] - patch_content_coords[1]], axis=-1)
        

        patch_normalized_boxes = patch_abs_boxes / patch_size

        remapped_classes = []
        remapped_image_abs_boxes = []
        remapped_patch_abs_boxes = []
        remapped_patch_normalized_boxes = []
        # if class_mapping is not None:
        #     for i, contained_class in enumerate(contained_classes):
        #         if contained_class in class_mapping:
        #             remapped_class = class_mapping[contained_class]
        #             remapped_classes.append(remapped_class)
        #             remapped_image_abs_boxes.append(image_abs_boxes[i])
        #             remapped_patch_abs_boxes.append(patch_abs_boxes[i])
        #             remapped_patch_normalized_boxes.append(patch_normalized_boxes[i])



        # else:
        #     remapped_classes = contained_classes.tolist()
        #     remapped_image_abs_boxes = image_abs_boxes.tolist()
        #     remapped_patch_abs_boxes = patch_abs_boxes.tolist()
        #     remapped_patch_normalized_boxes = patch_normalized_boxes.tolist()



        # patch_data["image_abs_boxes"] = remapped_image_abs_boxes
        # patch_data["patch_abs_boxes"] = remapped_patch_abs_boxes
        # patch_data["patch_normalized_boxes"] = remapped_patch_normalized_boxes
        # patch_data["patch_classes"] = remapped_classes

        patch_data["image_abs_boxes"] = image_abs_boxes.tolist()
        patch_data["patch_abs_boxes"] = patch_abs_boxes.tolist()
        patch_data["patch_normalized_boxes"] = patch_normalized_boxes.tolist()
        patch_data["patch_classes"] = contained_classes.tolist()
    return patch_data





def output_patch(patch, gt_boxes, pred_boxes, pred_classes, pred_scores, out_path):
    from models.common import model_vis

    out_array = model_vis.draw_boxes_on_image(patch,
                      pred_boxes,
                      pred_classes,
                      pred_scores,
                      class_map={"plant": 0},
                      gt_boxes=gt_boxes, #None,
                      patch_coords=None,
                      display_class=False,
                      display_score=False)
    cv2.imwrite(out_path, cv2.cvtColor(out_array, cv2.COLOR_RGB2BGR))