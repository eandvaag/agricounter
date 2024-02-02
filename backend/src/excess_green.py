import argparse
import os
import glob
import numpy as np
import cv2
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from osgeo import gdal

from joblib import Parallel, delayed


from image_set import Image
import image_utils
from io_utils import json_io
from models.common import box_utils, poly_utils

CHUNK_SIZE = 5000


def create_vegetation_record_for_orthomosaic(image_set_dir, excess_green_record, metadata, annotations, predictions):

    
    image_name = list(annotations.keys())[0]

    image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
    image = Image(image_path)
    w, h = image.get_wh()
    
    

    chunk_coords_lst = []
    for i in range(0, h, CHUNK_SIZE):
        for j in range(0, w, CHUNK_SIZE):
            chunk_coords_lst.append([i, j, min(i+CHUNK_SIZE, h), min(j+CHUNK_SIZE, w)])

    results = Parallel(int(os.cpu_count() / 3))(
        delayed(get_vegetation_percentages_for_chunk)(
            excess_green_record, metadata, annotations, predictions, image.image_path, chunk_coords) for chunk_coords in chunk_coords_lst)

    region_keys = ["regions_of_interest", "fine_tuning_regions", "test_regions"]
    cls_names = metadata["object_classes"] + ["All Classes"]

    vegetation_record = {}
    vegetation_record[image_name] = {}
    vegetation_record[image_name]["sel_val"] = excess_green_record[image_name]["sel_val"]
    vegetation_record[image_name]["vegetation_percentage"] = {}
    vegetation_record[image_name]["obj_vegetation_percentage"] = {}
    vegetation_record[image_name]["area_pixel_counts"] = {}
    vegetation_record[image_name]["vegetation_percentage"]["image"] = 0
    for cls_name in cls_names:
        vegetation_record[image_name]["obj_vegetation_percentage"][cls_name] = {}
        vegetation_record[image_name]["obj_vegetation_percentage"][cls_name]["image"] = 0
    # vegetation_record[image_name]["obj_vegetation_percentage"]["image"] = 0
    for region_key in region_keys:
        vegetation_record[image_name]["vegetation_percentage"][region_key] = []
        # vegetation_record[image_name]["obj_vegetation_percentage"][region_key] = []
        vegetation_record[image_name]["area_pixel_counts"][region_key] = []
        for cls_name in cls_names:
            vegetation_record[image_name]["obj_vegetation_percentage"][cls_name][region_key] = []

        # vegetation_record[image_name][region_key + "_coordinates"] = annotations[image_name][region_key]
        for i in range(len(annotations[image_name][region_key])):
            vegetation_record[image_name]["vegetation_percentage"][region_key].append(0)
            # vegetation_record[image_name]["obj_vegetation_percentage"][region_key].append(0)
            vegetation_record[image_name]["area_pixel_counts"][region_key].append(0)

            for cls_name in cls_names:
                vegetation_record[image_name]["obj_vegetation_percentage"][cls_name][region_key].append(0)



    for result in results:
        vegetation_record[image_name]["vegetation_percentage"]["image"] += result["vegetation_pixel_counts"]["chunk"]
        # vegetation_record[image_name]["obj_vegetation_percentage"]["image"] += result["obj_vegetation_pixel_counts"]["chunk"]
        for cls_name in cls_names:
            vegetation_record[image_name]["obj_vegetation_percentage"][cls_name]["image"] += result["obj_vegetation_pixel_counts"][cls_name]["chunk"]

        for region_key in region_keys:
            for i in range(len(annotations[image_name][region_key])):
                vegetation_record[image_name]["vegetation_percentage"][region_key][i] += result["vegetation_pixel_counts"][region_key][i]
                # vegetation_record[image_name]["obj_vegetation_percentage"][region_key][i] += result["obj_vegetation_pixel_counts"][region_key][i]
                vegetation_record[image_name]["area_pixel_counts"][region_key][i] += result["area_pixel_counts"][region_key][i]
                # vegetation_record[image_name][region_key][i] += result[region_key][i]

                for cls_name in cls_names:
                    vegetation_record[image_name]["obj_vegetation_percentage"][cls_name][region_key][i] += result["obj_vegetation_pixel_counts"][cls_name][region_key][i]
        
    # print("vegetation_record image pixel count", vegetation_record[image_name]["image"])
    vegetation_record[image_name]["vegetation_percentage"]["image"] = round(float((vegetation_record[image_name]["vegetation_percentage"]["image"] / (w * h)) * 100), 2)
    # vegetation_record[image_name]["obj_vegetation_percentage"]["image"] = round(float((vegetation_record[image_name]["obj_vegetation_percentage"]["image"] / (w * h)) * 100), 2)
    for cls_name in cls_names:
        vegetation_record[image_name]["obj_vegetation_percentage"][cls_name]["image"] = round(float((vegetation_record[image_name]["obj_vegetation_percentage"][cls_name]["image"] / (w * h)) * 100), 2)
    for region_key in region_keys:
        for i, region in enumerate(annotations[image_name][region_key]):
            # if region_key == "regions_of_interest":
            #     region_area = poly_utils.get_poly_area(region)
            # else:
            #     region_area = (region[2] - region[0]) * (region[3] - region[1])
            region_area = vegetation_record[image_name]["area_pixel_counts"][region_key][i]
            vegetation_record[image_name]["vegetation_percentage"][region_key][i] = round(float((vegetation_record[image_name]["vegetation_percentage"][region_key][i] / region_area) * 100), 2)
            # vegetation_record[image_name]["obj_vegetation_percentage"][region_key][i] = round(float((vegetation_record[image_name]["obj_vegetation_percentage"][region_key][i] / region_area) * 100), 2)
            for cls_name in cls_names:
                vegetation_record[image_name]["obj_vegetation_percentage"][cls_name][region_key][i] = round(float((vegetation_record[image_name]["obj_vegetation_percentage"][cls_name][region_key][i] / region_area) * 100), 2)


    return vegetation_record

def get_vegetation_percentages_for_chunk(excess_green_record, metadata, annotations, predictions, image_path, chunk_coords):
    ds = gdal.Open(image_path)
    chunk_w = chunk_coords[3]-chunk_coords[1]
    chunk_h = chunk_coords[2]-chunk_coords[0]
    chunk_array = ds.ReadAsArray(chunk_coords[1], 
                                 chunk_coords[0], 
                                 chunk_w, 
                                 chunk_h)

    chunk_array = np.transpose(chunk_array, (1, 2, 0))

    # print("chunk_array.shape", chunk_array.shape)
    exg_array = image_utils.excess_green(chunk_array)
    image_name = list(annotations.keys())[0]
    sel_val = excess_green_record[image_name]["sel_val"]
    chunk_vegetation_pixel_count = int(np.sum(exg_array > sel_val))



    # result = {

    #     "vegetation_percentage": {
    #         "chunk": chunk_vegetation_pixel_count,
    #         "regions_of_interest": [],
    #         "fine_tuning_regions": [],
    #         "test_regions": []
    #         # "tiles": []
    #     },
    #     "obj_vegetation_percentage": {
    #         "chunk": obj_chunk_vegetation_pixel_count,
    #         "regions_of_interest": [],
    #         "fine_tuning_regions": [],
    #         "test_regions": []
    #         # "tiles": []
    #     }
    # }


    result = {
        "vegetation_pixel_counts": {
            "chunk": chunk_vegetation_pixel_count,
            "regions_of_interest": [],
            "fine_tuning_regions": [],
            "test_regions": []
        },
        "obj_vegetation_pixel_counts": {},
        #     "chunk": obj_chunk_vegetation_pixel_count,
        #     "regions_of_interest": [],
        #     "fine_tuning_regions": [],
        #     "test_regions": []
        # },
        "area_pixel_counts": {
            "regions_of_interest": [],
            "fine_tuning_regions": [],
            "test_regions": []
        }
    }

    # tile_min_y = chunk_coords[0]
    # tile_min_x = chunk_coords[1]
    # num_y_tiles = m.ceil((chunk_coords[2]-chunk_coords[0]) / TILE_SIZE)
    # num_x_tiles = m.ceil((chunk_coords[3]-chunk_coords[1]) / TILE_SIZE)
    # for i in range(num_y_tiles):
    #     for j in range(num_x_tiles):
    #         # tile_coords = [
    #         #                chunk_coords[0] + (TILE_SIZE) * i,
    #         #                chunk_coords[1] + (TILE_SIZE) * j,
    #         #                min(chunk_coords[0] + (TILE_SIZE) * (i+1), chunk_coords[2]),
    #         #                min(chunk_coords[1] + (TILE_SIZE) * (j+1), chunk_coords[3])
    #         #                ]

    #         tile_exg_array = exg_array[i*(TILE_SIZE):(i+1)*TILE_SIZE, j*(TILE_SIZE):(j+1)*TILE_SIZE]
    #         tile_obj_exg_array = obj_exg_array[i*(TILE_SIZE):(i+1)*TILE_SIZE, j*(TILE_SIZE):(j+1)*TILE_SIZE]




    #         tile_vegetation_pixel_count = int(np.sum(tile_exg_array > sel_val))
    #         tile_obj_vegetation_pixel_count = int(np.sum(tile_obj_exg_array > sel_val))

    #         result["vegetation_percentage"]["tiles"].append(tile_vegetation_pixel_count)
    #         result["obj_vegetation_percentage"]["tiles"].append(tile_vegetation_pixel_count)




    # result = {
    #     "vegetation_percentage": {
    #         "chunk": chunk_vegetation_pixel_count,
    #         "fine_tuning_regions": [],
    #         "test_regions": []
    #     },
    #     "obj_vegetation_percentage": {
    #         "chunk": obj_chunk_vegetation_pixel_count,
    #         "fine_tuning_regions": [],
    #         "test_regions": []
    #     }

    #     # "chunk": chunk_vegetation_pixel_count,
    #     # "fine_tuning_regions": [],
    #     # "test_regions": []
    # }

    # print("chunk: {}, (v.c.: {} / {})".format(chunk_coords, chunk_vegetation_pixel_count, exg_array.size))


    for cls_ind, cls_name in enumerate(metadata["object_classes"] + ["All Classes"]):


        pred_mask = np.full((chunk_array.shape[0], chunk_array.shape[1]), True)

        image_boxes = np.array(predictions[image_name]["boxes"])
        image_scores = np.array(predictions[image_name]["scores"])
        image_classes = np.array(predictions[image_name]["classes"])



        if image_name in predictions:
            pred_boxes = image_boxes[image_scores > 0.50] 
            #np.array(predictions[image_name]["boxes"])[np.array(predictions[image_name]["scores"]) > 0.50]

            if cls_name == "All Classes":
                cls_mask = np.full(image_classes.size, True)
            else:
                cls_mask = image_classes == cls_ind

            sel_boxes = image_boxes[cls_mask]
            sel_scores = image_scores[cls_mask]

            pred_boxes = sel_boxes[sel_scores > 0.50]


        else:
            pred_boxes = np.array([])
        inds = box_utils.get_contained_inds(pred_boxes, [chunk_coords])
        pred_boxes = pred_boxes[inds]
        adj_pred_boxes = np.stack([
            np.maximum(pred_boxes[:, 0] - chunk_coords[0], 0),
            np.maximum(pred_boxes[:, 1] - chunk_coords[1], 0),
            np.maximum(pred_boxes[:, 2] - chunk_coords[0], 0),
            np.maximum(pred_boxes[:, 3] - chunk_coords[1], 0)
        ], axis=-1)
        for pred_box in adj_pred_boxes:
            pred_mask[pred_box[0]:pred_box[2], pred_box[1]:pred_box[3]] = False

        # obj_exg_array = exg_array[pred_mask]
        obj_exg_array = np.copy(exg_array)
        obj_exg_array[pred_mask] = -10000

        obj_chunk_vegetation_pixel_count = int(np.sum(obj_exg_array > sel_val))


        result["obj_vegetation_pixel_counts"][cls_name] = {
            "chunk": obj_chunk_vegetation_pixel_count,
            "regions_of_interest": [],
            "fine_tuning_regions": [],
            "test_regions": []            
        }

        region_keys = ["regions_of_interest", "fine_tuning_regions", "test_regions"]
        for region_key in region_keys:
            for region in annotations[image_name][region_key]:
                if region_key == "regions_of_interest":
                    chunk_poly = [[chunk_coords[0], chunk_coords[1]], [chunk_coords[2], chunk_coords[1]], [chunk_coords[2], chunk_coords[3]], [chunk_coords[0], chunk_coords[3]]]
                    intersects, intersect_regions = poly_utils.get_intersection_polys(region, chunk_poly)
                else:
                    intersects, intersect_region = box_utils.get_intersection_rect(region, chunk_coords)
                if not intersects:
                    result["vegetation_pixel_counts"][region_key].append(0)
                    result["obj_vegetation_pixel_counts"][cls_name][region_key].append(0)
                    result["area_pixel_counts"][region_key].append(0)
                else:
                    # index_coords = [
                    #     intersect_region[0] - chunk_coords[0],
                    #     intersect_region[2] - chunk_coords[0], 
                    #     intersect_region[1] - chunk_coords[1],
                    #     intersect_region[3] - chunk_coords[1]
                    # ]

                    if region_key == "regions_of_interest":
                        tmp_img = PILImage.new("L", (chunk_w, chunk_h))

                        # print("intersect_regions: {}".format(intersect_regions))
                        
                        for intersect_region in intersect_regions:
                            polygon = []
                            for coord in intersect_region:
                                polygon.append((min(chunk_w, max(0, round(coord[1] - chunk_coords[1]))), 
                                                min(chunk_h, max(0, round(coord[0] - chunk_coords[0])))))
                            # print("chunk_min_y: {}. chunk_min_x: {}, polygon: {}".format(chunk_coords[0], chunk_coords[1], polygon))
                            
                            # print("\tpolygon: {}".format(polygon))
                            if len(polygon) == 1:
                                PILImageDraw.Draw(tmp_img).point(polygon, fill=1)
                            else:
                                PILImageDraw.Draw(tmp_img).polygon(polygon, outline=1, fill=1)
                        mask = np.array(tmp_img) == 1
                        intersect_vals = exg_array[mask]
                        obj_intersect_vals = obj_exg_array[mask]
                    else:
                        intersect_vals = exg_array[intersect_region[0] - chunk_coords[0]:intersect_region[2] - chunk_coords[0], 
                                                intersect_region[1] - chunk_coords[1]:intersect_region[3] - chunk_coords[1]]
                    
                        obj_intersect_vals = obj_exg_array[intersect_region[0] - chunk_coords[0]:intersect_region[2] - chunk_coords[0], 
                                        intersect_region[1] - chunk_coords[1]:intersect_region[3] - chunk_coords[1]]

                    
                    # intersect_area = (intersect_region[2] - intersect_region[0]) * (intersect_region[3] - intersect_region[1])
                    vegetation_pixel_count = int(np.sum(intersect_vals > sel_val)) # / intersect_vals.size)
                    # print("intersect_region for {}-{}: {} ({}) (v. c. {} / {}) {}".format(
                    # region, chunk_coords, intersect_region, index_coords, vegetation_pixel_count, intersect_vals.size, exg_array.size))
                    # result["vegetation_percentage"][region_key].append(vegetation_pixel_count)


                    obj_vegetation_pixel_count = int(np.sum(obj_intersect_vals > sel_val))

                    # result["obj_vegetation_percentage"][region_key].append(obj_vegetation_pixel_count)


                    result["vegetation_pixel_counts"][region_key].append(vegetation_pixel_count)
                    result["obj_vegetation_pixel_counts"][cls_name][region_key].append(obj_vegetation_pixel_count)
                    result["area_pixel_counts"][region_key].append(intersect_vals.size)


    return result


def create_vegetation_record_for_image_set(image_set_dir, excess_green_record, metadata, annotations, predictions):
    # needs_update = []
    # for image_name in annotations.keys():
    #     if image_name not in vegetation_record or excess_green_record[image_name]["sel_val"] != vegetation_record[image_name]["sel_val"]:
    #         needs_update.append(image_name)
    #     elif not np.array_equal(np.array(vegetation_record[image_name]["fine_tuning_regions_coordinates"]), 
    #                             np.array(annotations[image_name]["fine_tuning_regions"])):
    #         needs_update.append(image_name)
    #     elif not np.array_equal(np.array(vegetation_record[image_name]["test_regions_coordinates"]), 
    #                             np.array(annotations[image_name]["test_regions"])):
    #         needs_update.append(image_name)

    image_names = list(annotations.keys()) 

    # num_threads = int(os.cpu_count() / 3)
    # for i in range(num_threads):
    #     worker = threading.Thread(name="worker_" + str(i), target=work)
    #     worker.start()

    results = Parallel(int(os.cpu_count() / 3))(
        delayed(get_vegetation_percentages_for_image)(image_set_dir, excess_green_record, metadata, annotations, predictions, image_name) for image_name in image_names)

    # print("new vegetation record results", results)
    vegetation_record = {}
    for result in results:
        image_name = result[0]
        vegetation_results = result[1]
        vegetation_record[image_name] = result[1] #{}
        # vegetation_record[image_name]["sel_val"] = excess_green_record[image_name]["sel_val"]
        # vegetation_record[image_name]["image"] = vegetation_results["image"]
        # vegetation_record[image_name]["fine_tuning_regions_coordinates"] = annotations[image_name]["fine_tuning_regions"]
        # vegetation_record[image_name]["fine_tuning_regions"] = vegetation_results["fine_tuning_regions"]
        # vegetation_record[image_name]["test_regions_coordinates"] = annotations[image_name]["test_regions"]
        # vegetation_record[image_name]["test_regions"] = vegetation_results["test_regions"]

    return vegetation_record



def get_vegetation_percentages_for_image(image_set_dir, excess_green_record, metadata, annotations, predictions, image_name):

    image_path = glob.glob(os.path.join(image_set_dir, "images", image_name + ".*"))[0]
    image = Image(image_path)
    image_array = image.load_image_array()
    exg_array = image_utils.excess_green(image_array)

    # pred_masks = {}
    # pred_mask = np.full((image_array.shape[0], image_array.shape[1]), True)
    
    # for cls_ind, cls_name in enumerate(metadata["object_classes"] + ["All Classes"]):
    
    sel_val = excess_green_record[image_name]["sel_val"]
    image_vegetation_percentage = round(float((np.sum(exg_array > sel_val) / exg_array.size) * 100), 2)

    # obj_exg_array = exg_array[pred_mask]

    result = {
        "sel_val": sel_val,
        "vegetation_percentage": {
            "image": image_vegetation_percentage,
            "regions_of_interest": [],
            "fine_tuning_regions": [],
            "test_regions": []
        },
        "obj_vegetation_percentage": {}
        # "obj_vegetation_percentage": {
        #     "image": obj_image_vegetation_percentage,
        #     "regions_of_interest": [],
        #     "fine_tuning_regions": [],
        #     "test_regions": []
        # }
    }
    for cls_ind, cls_name in enumerate(metadata["object_classes"] + ["All Classes"]):

        pred_mask = np.full((image_array.shape[0], image_array.shape[1]), True)
        if image_name in predictions:

            image_boxes = np.array(predictions[image_name]["boxes"])
            image_scores = np.array(predictions[image_name]["scores"])
            image_classes = np.array(predictions[image_name]["classes"])

            if cls_name == "All Classes":
                cls_mask = np.full(image_classes.size, True)
            else:
                cls_mask = image_classes == cls_ind

            sel_boxes = image_boxes[cls_mask]
            sel_scores = image_scores[cls_mask]

            pred_boxes = sel_boxes[sel_scores > 0.50]

            # pred_boxes = np.array(predictions[image_name]["boxes"])[np.array(predictions[image_name]["scores"]) > 0.50]
        else:
            pred_boxes = np.array([])

        for pred_box in pred_boxes:
            pred_mask[pred_box[0]:pred_box[2], pred_box[1]:pred_box[3]] = False
        
        
        obj_exg_array = np.copy(exg_array)
        obj_exg_array[pred_mask] = -10000


        obj_image_vegetation_percentage = round(float((np.sum(obj_exg_array > sel_val) / exg_array.size) * 100), 2)


        result["obj_vegetation_percentage"][cls_name] = {
            "image": obj_image_vegetation_percentage,
            "regions_of_interest": [],
            "fine_tuning_regions": [],
            "test_regions": []            
        }

        region_keys = ["regions_of_interest", "fine_tuning_regions", "test_regions"]
        for region_key in region_keys:
            for region in annotations[image_name][region_key]:

                if region_key == "regions_of_interest":
                    polygon = []
                    for coord in region:
                        polygon.append((round(coord[1]), round(coord[0])))
                    tmp_img = PILImage.new("L", (exg_array.shape[1], exg_array.shape[0]))
                    PILImageDraw.Draw(tmp_img).polygon(polygon, outline=1, fill=1)
                    mask = np.array(tmp_img) == 1
                    region_vals = exg_array[mask]
                    obj_region_vals = obj_exg_array[mask]
                else:
                    region_vals = exg_array[region[0]:region[2], region[1]:region[3]]
                    obj_region_vals = obj_exg_array[region[0]:region[2], region[1]:region[3]]

                vegetation_percentage = round(float((np.sum(region_vals > sel_val) / region_vals.size) * 100), 2)
                result["vegetation_percentage"][region_key].append(vegetation_percentage)
                
                obj_vegetation_percentage = round(float((np.sum(obj_region_vals > sel_val) / region_vals.size) * 100), 2)
                result["obj_vegetation_percentage"][cls_name][region_key].append(obj_vegetation_percentage)

    return (image_name, result)


def create_excess_green_for_image(image_set_dir, image_path):
    image_name_full = os.path.basename(image_path)
    image_name = image_name_full.split(".")[0]

    image = Image(image_path)
    #image_array = image.load_image_array().astype(np.int64)
    #exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    #exg_array = exg_array.astype(np.int64)
    exg_array = image_utils.excess_green(image)
    
    min_val = round(float(np.min(exg_array)), 2)
    max_val = round(float(np.max(exg_array)), 2)
    #sel_val = round(float(threshold_otsu(exg_array)), 2)
    sel_val = round((min_val + max_val) / 2, 2)
    percent_vegetation = round(float((np.sum(exg_array > sel_val) / exg_array.size) * 100), 2)

    exg_array = image_utils.scale_image(exg_array, -2, 2, 0, 255, np_type=np.uint8)

    # sel_val = determine_default_segmentation_value(exg_array, sel_val)
    


    

    


    cv2.imwrite(os.path.join(image_set_dir, "excess_green", image_name + ".png"), exg_array)


    #lockfile = lock.lock_acquire("excess_green_lock")

    # if os.path.exists(record_path):
    #     record = json_io.load_json(record_path)
    # else:
        
    # record[image_name] = {
    #     "min_val": min_val,
    #     "max_val": max_val,
    #     "sel_val": sel_val,
    #     "ground_cover_percentage": percent_vegetation
    # }

    return (image_name, {
        "min_val": min_val,
        "max_val": max_val,
        "sel_val": sel_val,
        "ground_cover_percentage": percent_vegetation
    })



def create_excess_green_for_image_set(image_set_dir):

    
    image_paths = glob.glob(os.path.join(image_set_dir, "images", "*"))

    record_path = os.path.join(image_set_dir, "excess_green", "record.json")
    record = {}

    rec_tups = Parallel(10)(
        delayed(create_excess_green_for_image)(image_set_dir, image_path) for image_path in image_paths)

    for rec_tup in rec_tups:
        record[rec_tup[0]] = rec_tup[1]

    json_io.save_json(record_path, record)


            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    #parser.add_argument("image_name", type=str)
    args = parser.parse_args()

    create_excess_green_for_image_set(args.image_set_dir) #, args.image_name)