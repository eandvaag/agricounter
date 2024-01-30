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
    patches_dir = os.path.join(image_set_dir, "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")

    try:
        patch_data = json_io.load_json(patch_data_path)
    except Exception:
        patch_data = {
            "num_training_regions": 0,
            "patches": {}
        }


    num_training_regions = annotation_utils.get_num_training_regions(annotations)

    saved_num_training_regions = patch_data["num_training_regions"]

    apply_update = num_training_regions != saved_num_training_regions

    if apply_update:

        patch_data["patches"] = {}
        # N.B. 'num_training_regions' is updated after tf records have been written

        #  = {
        #     "num_training_regions": 0, # this is updated once the records are prepared  #num_training_regions,
        #     "patches": {}
        # }

        metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)
        is_ortho = metadata["is_ortho"] == "yes"

        for image_name in annotations.keys():

            if len(annotations[image_name]["training_regions"]) > 0:
                image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
                image = Image(image_path)
                patch_records = extract_patch_records_from_image_tiled(
                    image, 
                    updated_patch_size,
                    image_annotations=None,
                    patch_overlap_percent=50, #0, #50, #0, #50, #0, #50, 
                    include_patch_arrays=False,
                    regions=annotations[image_name]["training_regions"],
                    is_ortho=is_ortho,
                    out_dir=patches_dir)

                patch_data["patches"][image_name] = patch_records

    # patch_data["num_training_regions"] = num_training_regions
    json_io.save_json(patch_data_path, patch_data)

    return apply_update #changed_training_image_names

            

            



def update_patches(image_set_dir, annotations, image_names, updated_patch_size, update_thresh=0): #=None, image_status=None):
    
    logger = logging.getLogger(__name__)

    # if (image_names is None and image_status is None) or (image_names is not None and image_status is not None):
    #     raise RuntimeError("Only one of 'image_names' and 'image_status' should be None")

    changed_training_image_names = []

    images_dir = os.path.join(image_set_dir, "images")
    patches_dir = os.path.join(image_set_dir, "patches")
    patch_data_path = os.path.join(patches_dir, "patch_data.json")

    # read_time = int(time.time())

    # annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    # annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

    # if image_status is not None:
    #     image_names = []
    #     for image_name in annotations.keys():
    #         if annotations[image_name]["status"] == image_status:
    #             image_names.append(image_name)


    # num_annotations = w3c_io.get_num_annotations(annotations)

    # if num_annotations < 50:
        
    #     updated_patch_size = estimate_patch_size(image_set_dir, annotations)
    #     #updated_patch_size = 154 #300 # 154  #100 #300 #100 #400 #500 #300
    # else:
    #     try:
    #         updated_patch_size = w3c_io.get_patch_size(annotations)
    #     except RuntimeError:
    #         updated_patch_size = 300 #154 # 300 #100 #300 #100 #400 #500 #300
    #     # logger.info("Updated patch size: {}".format(updated_patch_size))

    # if os.path.exists(patch_data_path):
    #     patch_data = json_io.load_json(patch_data_path)
    # else:
    #     patch_data = {}

    
    # update_thresh = 10

    if os.path.exists(patch_data_path):
        patch_data = json_io.load_json(patch_data_path)
    else:
        patch_data = {}


    for image_name in image_names:
        update_image = False

        if image_name not in patch_data:
            update_image = True
            patch_data[image_name] = {}
        else:

            

            # if "update_time" not in patch_data[image_name] or "patches" not in patch_data[image_name]:
            #     update_image = True
            # else:
            #     update_time = patch_data[image_name]["update_time"]
            #     # print(annotations[image_name])
            #     if update_time < annotations[image_name]["update_time"]:
            #         update_image = True
            #     else:
            if len(patch_data[image_name]["patches"]) == 0:
                update_image = True
            else:
                sample_patch_coords = patch_data[image_name]["patches"][0]["patch_coords"]
                existing_patch_size = sample_patch_coords[2] - sample_patch_coords[0]
                abs_patch_size_diff = abs(existing_patch_size - updated_patch_size)
                if abs_patch_size_diff > update_thresh:
                    update_image = True


        if update_image:
            # print("updating", image_name)
            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = Image(image_path)
            patch_records = extract_patch_records_from_image_tiled(
                image, 
                updated_patch_size,
                image_annotations=None,
                patch_overlap_percent=50, 
                include_patch_arrays=False,
                out_dir=patches_dir)

            # logger.info("Writing patches for image {} (from: {})".format(image_name, image_set_dir))
            # write_patches(patches_dir, patch_records)

            # patch_records = extract_patch_records_from_image_tiled(
            #     image, 
            #     updated_patch_size,
            #     image_annotations=None,
            #     patch_overlap_percent=50, 
            #     include_patch_arrays=False)

            # annotations[]
            patch_data[image_name]["saved_status"] = annotations[image_name]["status"]
            patch_data[image_name]["patches"] = patch_records
            # if annotations_read_time is not None:
                # patch_data[image_name]["update_time"] = annotations_read_time
            if annotations[image_name]["status"] == "completed_for_training":
                changed_training_image_names.append(image_name)


        else:
            if annotations[image_name]["status"] == "completed_for_training" and \
                patch_data[image_name]["saved_status"] != "completed_for_training":

                patch_data[image_name]["saved_status"] = "completed_for_training"
                changed_training_image_names.append(image_name)


            
    json_io.save_json(patch_data_path, patch_data)

    return changed_training_image_names



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
        # annotate_patch(patch_record, annotation_boxes)



# def get_patch_coords_surrounding_box(box, patch_size, img_h, img_w):

#     logger = logging.getLogger(__name__)

#     box_y_min, box_x_min, box_y_max, box_x_max = box

#     box_h = box_y_max - box_y_min
#     box_w = box_x_max - box_x_min

#     if box_h > patch_size or box_w > patch_size:
#         logger.warning("Box exceeds size of patch. (box_w, box_h): ({}, {}). patch_size: {}.".format(
#             box_w, box_h, patch_size
#         ))

#     patch_y_min = random.randrange((box_y_min + box_h) - max(box_h, patch_size), box_y_min + 1)
#     patch_y_min = min(img_h - patch_size, max(0, patch_y_min))
#     patch_x_min = random.randrange((box_x_min + box_w) - max(box_w, patch_size), box_x_min + 1)
#     patch_x_min = min(img_w - patch_size, max(0, patch_x_min))
#     patch_y_max = patch_y_min + patch_size
#     patch_x_max = patch_x_min + patch_size

#     patch_coords = [patch_y_min, patch_x_min, patch_y_max, patch_x_max]

#     if patch_y_min < 0 or patch_x_min < 0 or patch_y_max > img_h or patch_x_max > img_w:
#         raise RuntimeError("Patch exceeds boundaries of the image.")

#     return patch_coords


def round_to_multiple(num, multiple):
    return multiple * round(num / multiple)

# def extract_patches_from_annotation_guides(image, patch_size, image_annotations, image_annotation_guides):

#     annotation_boxes = image_annotations["boxes"]
#     annotation_classes = image_annotations["classes"]
#     image_patches = []

#     image_array = image.load_image_array()
#     #image_h, image_w = image_array.shape[:2]

#     patch_num = 0

#     image_path_pieces = image.image_path.split("/")
#     farm_name = image_path_pieces[-5]
#     field_name = image_path_pieces[-4]
#     mission_date = image_path_pieces[-3]

#     patch_overlap_percent = 50
#     overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

#     for annotation_guide in image_annotation_guides:
#         #annotation_size = round(min(annotation_guide["width"], annotation_guide["height"]))


#         guide_coords = [
#             round(annotation_guide["py"]),
#             round(annotation_guide["px"]),
#             round(annotation_guide["py"]) + round(annotation_guide["height"]),
#             round(annotation_guide["px"]) + round(annotation_guide["width"])
#         ]
#         # print("guide_coords", guide_coords)

#         #guide_patch_array = image_array[guide_coords[0]: guide_coords[2],
#         #                                guide_coords[1]: guide_coords[3]]


#         patch_min_y = guide_coords[0]
   

#         col_covered = False
#         #patch_min_y = 0
#         while not col_covered:
#             patch_max_y = patch_min_y + patch_size #tile_size
#             if patch_max_y >= guide_coords[2]: #h:
#                 #patch_min_y = h - tile_size
#                 #patch_max_y = h
#                 col_covered = True


#             row_covered = False
#             #patch_min_x = 0
#             patch_min_x = guide_coords[1]
#             while not row_covered:

#                 patch_max_x = patch_min_x + patch_size #tile_size
#                 if patch_max_x >= guide_coords[3]: #w:
#                     #patch_min_x = w - tile_size
#                     #patch_max_x = w
#                     row_covered = True

#                 patch_data = {}
#                 patch_data["image_name"] = image.image_name
#                 patch_data["image_path"] = image.image_path
#                 patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
#                                             image.image_name + "-" + str(patch_num).zfill(7) + ".png"
#                 patch_data["patch_coords"] = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]
#                 patch_array = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
#                 content = image_array[patch_min_y: min(patch_max_y, guide_coords[2]), patch_min_x: min(patch_max_x, guide_coords[3])] 
#                 patch_array[0: content.shape[0], 0: content.shape[1], :] = content

#                 patch_data["patch"] = patch_array
            
#                 clip_coords = np.array([patch_min_y,
#                                         patch_min_x, 
#                                         min(patch_max_y, guide_coords[2]),
#                                         min(patch_max_x, guide_coords[3])])
#                 annotate_patch(patch_data, annotation_boxes, annotation_classes, clip_coords=clip_coords) #guide_coords) #, clip_coords=guide_coords)

#                 output_patch(patch_data["patch"], patch_data["patch_abs_boxes"], pred_boxes=[], pred_classes=[], pred_scores=[], 
#                             out_path=os.path.join("usr", "data", "tmp_patch_data_ex", patch_data["patch_name"]))

#                 image_patches.append(patch_data)
#                 patch_num += 1


#                 patch_min_x += (patch_size - overlap_px)

#             patch_min_y += (patch_size - overlap_px)

#         # patch_data = {}
#         # patch_data["image_name"] = image.image_name
#         # patch_data["image_path"] = image.image_path
#         # patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
#         #                                     image.image_name + "-" + str(patch_num).zfill(7) + ".png"


#         # patch_height = round_to_multiple(round(annotation_guide["height"]), patch_size)
#         # patch_width = round_to_multiple(round(annotation_guide["width"]), patch_size)

#         # patch_array = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)

#         # patch_array[:annotation_guide["height"], :annotation_guide["width"]] = guide_patch_array
#         # patch_data["patch_coords"] = [
#         #     guide_coords[0],
#         #     guide_coords[1],
#         #     guide_coords[0] + patch_height,
#         #     guide_coords[1] + patch_width
#         # ]

                    
#         # annotate_patch(patch_data, annotation_boxes, annotation_classes, clip_coords=guide_coords)
            

#         # # patch_data["patch"] = guide_patch_array
#         # # patch_data["patch_coords"] = guide_coords
#         # # annotate_patch(patch_data, annotation_boxes, annotation_classes)

#         # # print("annotation_guide size", annotation_size)
#         # # print("patch_size", patch_size)
#         # # if annotation_size >= patch_size:
#         # #     diff = annotation_size - patch_size
#         # #     inset = round(diff / 2)
#         # #     patch_coords = [guide_coords[0] + inset, 
#         # #                     guide_coords[1] + inset, 
#         # #                     guide_coords[0] + inset + patch_size, 
#         # #                     guide_coords[1] + inset + patch_size]
#         # #     patch_array = image_array[patch_coords[0]: patch_coords[2],
#         # #                               patch_coords[1]: patch_coords[3]]

#         # #     patch_data["patch"] = patch_array
#         # #     patch_data["patch_coords"] = patch_coords
#         # #     annotate_patch(patch_data, annotation_boxes, annotation_classes)

        
#         # # else:

#         # #     diff = patch_size - annotation_size
#         # #     inset = round(diff / 2)

#         # #     patch_array = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
#         # #     patch_array[inset: inset+annotation_size, inset:inset+annotation_size] = guide_patch_array
#         # #     patch_data["patch_coords"] = [
#         # #                     guide_coords[0] - inset,
#         # #                     guide_coords[1] - inset,
#         # #                     guide_coords[0] - inset + patch_size,
#         # #                     guide_coords[1] - inset + patch_size
#         # #     ]
#         # #     patch_data["patch"] = patch_array
            
#         # #     annotate_patch(patch_data, annotation_boxes, annotation_classes, clip_coords=guide_coords)


#         # output_patch(patch_data["patch"], patch_data["patch_abs_boxes"], pred_boxes=[], pred_classes=[], pred_scores=[], 
#         #               out_path=os.path.join("usr", "data", "tmp_patch_data_ex", patch_data["patch_name"]))



#         # image_patches.append(patch_data)
#         # patch_num += 1


#     return image_patches




# def extract_patch_records_surrounding_annotations(image, patch_size, image_annotations, include_patch_arrays=True):

#     annotation_boxes = image_annotations["boxes"]
#     annotation_classes = image_annotations["classes"]
#     image_patches = []

#     if include_patch_arrays:
#         image_array = image.load_image_array()
#         image_h, image_w = image_array.shape[:2]
#     else:
#         image_w, image_h = image.get_wh()

#     patch_num = 0

#     image_path_pieces = image.image_path.split("/")
#     farm_name = image_path_pieces[-5]
#     field_name = image_path_pieces[-4]
#     mission_date = image_path_pieces[-3]

#     for box in annotation_boxes:

#         patch_coords = get_patch_coords_surrounding_box(box, patch_size, image_h, image_w)
        
#         #patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

#         patch_data = {}
#         if include_patch_arrays:
#             #patch_array = image_array[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
#             patch_array = image_array[patch_coords[0]: patch_coords[2],
#                                         patch_coords[1]: patch_coords[3]]
#             patch_data["patch"] = patch_array
#         patch_data["image_name"] = image.image_name
#         patch_data["image_path"] = image.image_path
#         patch_data["patch_name"] = farm_name + "-" + field_name + "-" + mission_date + "-" + \
#                                     image.image_name + "-" + str(patch_num).zfill(7) + ".png"
#         patch_data["patch_coords"] = patch_coords
        
#         #if image_annotations is not None and \
#         #    (annotation_status == "completed_for_training" or annotation_status == "completed_for_testing"):
#         annotate_patch(patch_data, annotation_boxes, annotation_classes)
#         image_patches.append(patch_data)
#         patch_num += 1


#     return image_patches


# def extract_patch_records_from_ortho_tiled(image, 
#                                            patch_size, 
#                                            image_annotations=None, 
#                                            patch_overlap_percent=50, 
#                                            include_patch_arrays=True, 
#                                            regions="all", 
#                                            out_dir=None):

    

#     if is_ortho:
#         regions = annotations[ortho_region_type]



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