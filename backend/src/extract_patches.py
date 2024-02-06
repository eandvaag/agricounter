import os
import glob
import math as m
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import cv2
from osgeo import gdal


from models.common import box_utils, annotation_utils, poly_utils
from io_utils import json_io
from image_wrapper import ImageWrapper

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

    patch_data = {}


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    is_ortho = metadata["is_ortho"]

    for image_name in annotations.keys():

        if len(annotations[image_name]["fine_tuning_regions"]) > 0:
            image_path = glob.glob(os.path.join(images_dir, image_name + ".*"))[0]
            image = ImageWrapper(image_path)
            patch_records = extract_patch_records_from_image_tiled(
                image, 
                updated_patch_size,
                image_annotations=annotations[image_name],
                patch_overlap_percent=0,
                regions=annotations[image_name]["fine_tuning_regions"],
                is_ortho=is_ortho,
                out_dir=patches_dir)

            patch_data[image_name] = patch_records

    return patch_data




def extract_patch_records_from_image_tiled(image, 
                                           patch_size, 
                                           image_annotations,
                                           patch_overlap_percent, 
                                           regions,
                                           is_ortho,
                                           out_dir):


    image_patches = []


    if is_ortho:
        ds = gdal.Open(image.image_path)
    else:
        image_array = image.load_image_array()


    tile_size = patch_size
    overlap_px = int(m.floor(tile_size * (patch_overlap_percent / 100)))
    
    patch_num = 0

    image_path_pieces = image.image_path.split("/")
    username = image_path_pieces[-7]
    farm_name = image_path_pieces[-5]
    field_name = image_path_pieces[-4]
    mission_date = image_path_pieces[-3]


    for region in regions:


        region_bbox = poly_utils.get_poly_bbox(region)

        col_covered = False
        patch_min_y = region_bbox[0]
        while not col_covered:
            patch_max_y = patch_min_y + tile_size
            max_content_y = patch_max_y
            if patch_max_y >= region_bbox[2]:
                max_content_y = region_bbox[2]
                col_covered = True

            row_covered = False
            patch_min_x = region_bbox[1]
            while not row_covered:
                patch_max_x = patch_min_x + tile_size
                max_content_x = patch_max_x
                if patch_max_x >= region_bbox[3]:
                    max_content_x = region_bbox[3]
                    row_covered = True


                if is_ortho:
                    image_array = ds.ReadAsArray(patch_min_x, patch_min_y, (max_content_x-patch_min_x), (max_content_y-patch_min_y))
                    image_array = np.transpose(image_array, (1, 2, 0))


                patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]

                patch_poly = [
                    [patch_min_y, patch_min_x],
                    [patch_min_y, patch_max_x],
                    [patch_max_y, patch_max_x],
                    [patch_max_y, patch_min_x]
                ]
                intersects, intersect_regions = poly_utils.get_intersection_polys(region, patch_poly)

                if intersects:

                    patch_array = np.zeros(shape=(patch_size, patch_size, 3), dtype=np.uint8)
                    if is_ortho:
                        patch_array[0:(max_content_y-patch_min_y), 0:(max_content_x-patch_min_x)] = image_array
                    else:
                        patch_array[0:(max_content_y-patch_min_y), 0:(max_content_x-patch_min_x)] = image_array[patch_min_y:max_content_y, patch_min_x:max_content_x]
                    

                    tmp_img = PILImage.new("L", (patch_size, patch_size))
                    for intersect_region in intersect_regions:
                        polygon = []
                        for coord in intersect_region:
                            polygon.append((min(patch_size, max(0, round(coord[1] - patch_min_x))), 
                                            min(patch_size, max(0, round(coord[0] - patch_min_y)))))

                        if len(polygon) == 1:
                            PILImageDraw.Draw(tmp_img).point(polygon, fill=1)
                        else:
                            PILImageDraw.Draw(tmp_img).polygon(polygon, outline=1, fill=1)
                    mask = np.array(tmp_img) != 1
                    patch_array[mask] = [0, 0, 0]


                    patch_data = {}
                    patch_data["image_name"] = image.image_name
                    patch_data["image_path"] = image.image_path
                    patch_data["patch_name"] = username + "-" + farm_name + "-" + field_name + "-" + mission_date + "-" + \
                                            image.image_name + "-" + str(patch_num).zfill(7) + ".png"
                    patch_data["patch_coords"] = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]
                    patch_data["patch_path"] = os.path.join(out_dir, patch_data["patch_name"])


                    cv2.imwrite(patch_data["patch_path"], 
                                cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR))
                    

                    annotate_patch(patch_data, image_annotations, patch_coords, region)


                    image_patches.append(patch_data)
                    patch_num += 1


                patch_min_x += (tile_size - overlap_px)

            patch_min_y += (tile_size - overlap_px)

    return image_patches







def annotate_patch(patch_data, image_annotations, patch_coords, region, min_visibility=0.15):

    gt_boxes = image_annotations["boxes"]
    gt_classes = image_annotations["classes"]
        
    if gt_boxes.size == 0:
        patch_data["image_abs_boxes"] = []
        patch_data["patch_abs_boxes"] = []
        patch_data["patch_normalized_boxes"] = []
        patch_data["patch_classes"] = [] 

    else:

        contained_inds = box_utils.get_contained_inds(gt_boxes, [patch_coords])
        contained_boxes = gt_boxes[contained_inds]
        contained_classes = gt_classes[contained_inds]

        clipped_boxes = box_utils.clip_boxes_np(contained_boxes, patch_coords)

        mask = poly_utils.get_bbox_visibility_mask(contained_boxes, clipped_boxes, region, vis_thresh=min_visibility)

        image_abs_boxes = clipped_boxes[mask]
        patch_classes = contained_classes[mask]


        patch_abs_boxes = np.stack([image_abs_boxes[:,0] - patch_coords[0],
                                    image_abs_boxes[:,1] - patch_coords[1],
                                    image_abs_boxes[:,2] - patch_coords[0],
                                    image_abs_boxes[:,3] - patch_coords[1]], axis=-1)


        patch_size = patch_coords[2] - patch_coords[0]
        patch_normalized_boxes = patch_abs_boxes / patch_size


        patch_data["image_abs_boxes"] = image_abs_boxes.tolist()
        patch_data["patch_abs_boxes"] = patch_abs_boxes.tolist()
        patch_data["patch_normalized_boxes"] = patch_normalized_boxes.tolist()
        patch_data["patch_classes"] = patch_classes.tolist()

        
    return patch_data
