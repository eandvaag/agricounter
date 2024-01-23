import os
import shutil
import glob
import math as m
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import time


RESULT_LIFETIME = 2 * 60 #2 * 3600

from sklearn.metrics import pairwise_distances
from models.common import annotation_utils, box_utils
# import jig

MAX_NUM_TILES = 75000


# from image_set import DataSet
from io_utils import json_io #, exif_io

def range_map(old_val, old_min, old_max, new_min, new_max):
    new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_val


# def similarity_map(username, farm_name, field_name, mission_date):

#     # farm_name = dataset.farm_name
#     # field_name = dataset.field_name
#     # mission_date = dataset.mission_date

    
#     image_set_dir =  os.path.join("usr", "data", username, "image_sets",
#                         farm_name, field_name, mission_date)

#     annotations_path = os.path.join(image_set_dir,
#                         "annotations", "annotations_w3c.json")
    
#     annotations = json_io.load_json(annotations_path)

#     annotated_features = []
#     unannotated_features = []
#     feature_counts = []
#     unannotated_image_names = []
#     for image_name in annotations.keys():
#         features_path = os.path.join(image_set_dir, "features", image_name + ".npy")
#         features = np.load(features_path)

#         if annotations[image_name]["status"] == "completed":
#             annotated_features.extend(features.tolist())
#         else:
#             unannotated_features.extend(features.tolist())
#             feature_counts.append(features.shape[0])
#             unannotated_image_names.append(image_name)

#     print("calculating pairwise distances")
#     distances = pairwise_distances(unannotated_features, annotated_features)
#     print("finished")
#     #print("feature_counts", feature_counts)
#     scores = []
#     prev_feature_sum = 0
#     for i, feature_count in enumerate(feature_counts):
#         sel = distances[prev_feature_sum:prev_feature_sum + feature_count, :]
#         sel_sum = np.sum(sel)
#         #print("adding {}, score: {}".format(unannotated_image_names[i], sel_sum))
#         #res[unannotated_image_names[i]] = sel_sum
#         scores.append(sel_sum)
#         prev_feature_sum = prev_feature_sum + feature_count

#     #json_io.print_json(res)
#     scores = np.array(scores)
#     inds = np.argsort(scores)
#     sorted_scores = scores[inds]
#     sorted_names = np.array(unannotated_image_names)[inds]
#     for i, (score, name) in enumerate(zip(sorted_scores, sorted_names)):
#         print("{}: Image: {} | Score: {}".format(i, name, score))


def create_plot(grid_z0, extent, vmin, vmax, cmap, out_path):
    plt.figure()
    plt.imshow(grid_z0.T, extent=extent, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    #out_path = os.path.join(out_dir, "annotated_map.svg")
    plt.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0)


def create_interpolation_map_for_ortho(username, farm_name, field_name, mission_date, 
                    predictions_path, out_dir, class_index, interpolation, tile_size):

    # annotations = annotation_utils.load_annotations(annotations_path)

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)


    if metadata["camera_height"] == "":# or metadata["missing"]["area_m2"]:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)


    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]

    if make not in camera_specs:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    if model not in camera_specs[make]:
        raise RuntimeError("Cannot compute map due to missing metadata.")



    camera_entry = camera_specs[make][model]
    sensor_height = camera_entry["sensor_height"]
    sensor_width = camera_entry["sensor_width"]
    focal_length = camera_entry["focal_length"]
    raw_image_height_px = camera_entry["image_height_px"]
    raw_image_width_px = camera_entry["image_width_px"]

    camera_height = metadata["camera_height"]

    predictions = json_io.load_json(predictions_path)
    image_name = list(metadata["images"].keys())[0]

    # raw_image_height_px = metadata["raw_image_height_px"]
    # raw_image_width_px = metadata["raw_image_width_px"]


    # image_name = list(annotations.keys())[0]

    image_height_px = metadata["images"][image_name]["height_px"]
    image_width_px = metadata["images"][image_name]["width_px"]


    gsd_h = (camera_height * sensor_height) / (focal_length * raw_image_height_px)
    gsd_w = (camera_height * sensor_width) / (focal_length * raw_image_width_px)

    gsd = min(gsd_h, gsd_w)


    # image_height_px = metadata["images"][image_name]["height_px"]
    # image_width_px = metadata["images"][image_name]["width_px"]

    image_height_m = image_height_px * gsd
    image_width_m = image_width_px * gsd


    px_per_m = round(1 / gsd)
    # print("px_per_m", px_per_m)

    # fully_annotated = annotation_utils.is_fully_annotated(annotations, image_name, image_width_px, image_height_px)
    # annotation_boxes = annotations[image_name]["boxes"]

    # if pred_path is not None:
    #     predictions = json_io.load_json(pred_path)
    if image_name in predictions:
        mask = np.array(predictions[image_name]["scores"]) > 0.50
        pred_boxes = np.array(predictions[image_name]["boxes"])[mask]
    else:
        pred_boxes = np.array([])

    annotated_values = []
    predicted_values = []
    all_points = []

    # TOTAL_NUM_POINTS = 10000
    # res = jig.jig(image_width_px, image_height_px, TOTAL_NUM_POINTS)
    # num_x_tiles = res["num_x_tiles"]
    # num_y_tiles = res["num_y_tiles"]
    tile_width_m = tile_size #1 #0.5
    tile_height_m = tile_size #1 #0.5

    num_x_tiles = round(image_width_m / tile_width_m)
    num_y_tiles = round(image_height_m / tile_height_m)



    tile_width = image_width_px / num_x_tiles
    tile_height = image_height_px / num_y_tiles

    if num_x_tiles * num_y_tiles > MAX_NUM_TILES:
        raise RuntimeError("Unable to create density map: too many tiles requested.")

    print("num_x_tiles", num_x_tiles)
    print("num_y_tiles", num_y_tiles)
    print("tile_x_width", tile_width)
    print("tile_y_width", tile_height)

    area_m2_per_tile = (tile_width * gsd) * (tile_height * gsd)
    box_centres = np.rint((pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0).astype(np.int64)
    # box centres (y, x) format

    y_regions = np.round(box_centres[:, 0] / tile_height).astype(np.int64)
    x_regions = np.round(box_centres[:, 1] / tile_width).astype(np.int64)

    for i in range(num_y_tiles):
        for j in range(num_x_tiles):

            region = [i * tile_height, j * tile_width, (i+1) * tile_height, (j+1) * tile_width]
            # print(region)

            # contained_box_centres = box_utils.get_contained_inds_for_points(box_centres, [region])
            # num_boxes_in_region = contained_box_centres.shape[0]
            y_mask = i == y_regions
            x_mask = j == x_regions
            mask = np.logical_and(y_mask, x_mask)
            num_boxes_in_region = np.sum(mask)
            val = num_boxes_in_region / area_m2_per_tile
            predicted_values.append(val)
            # if i == 0:
            #     point_y = round(region[0]) # + (tile_height / 4))
            # elif i == num_y_tiles - 1:
            #     point_y = round(region[2]) # - (tile_height / 4))
            # else:
            point_y = round(region[0] + (tile_height / 2))
            # if j == 0:
            #     point_x = round(region[1]) # + (tile_width / 4))
            # elif j == num_x_tiles - 1:
            #     point_x = round(region[3]) # - (tile_width / 4))
            # else:
            point_x = round(region[1] + (tile_width / 2))

            point = [point_x, point_y]
            all_points.append(point)
            if i == 0:
                add_point = [point_x, 0]
                predicted_values.append(val)
                all_points.append(add_point)
            if j == 0:
                add_point = [0, point_y]
                predicted_values.append(val)
                all_points.append(add_point)
            if i == num_y_tiles - 1:
                add_point = [point_x, round(region[2])]
                predicted_values.append(val)
                all_points.append(add_point)
            if j  == num_x_tiles - 1:
                add_point = [round(region[3]), point_y]
                predicted_values.append(val)
                all_points.append(add_point)

            if i == 0 and j == 0:
                add_point = [0, 0]
                predicted_values.append(val)
                all_points.append(add_point)
            if i == num_y_tiles - 1 and j == 0:
                add_point = [0, round(region[2])]
                predicted_values.append(val)
                all_points.append(add_point)
            if i == num_y_tiles - 1 and j == num_x_tiles - 1:
                add_point = [round(region[3]), round(region[2])]
                predicted_values.append(val)
                all_points.append(add_point)
            if i == 0 and j == num_x_tiles - 1:
                add_point = [round(region[3]), 0]
                predicted_values.append(val)
                all_points.append(add_point)
        

            # if i == 0 or j == 0 or i == num_y_tiles - 1 or j == num_x_tiles - 1:



            #     if i == 0:
            #         add_point_y = 0
            #     elif i == num_y_tiles - 1:
            #         add_point_y = image_height_px
            #     else:
            #         add_point_y = round(region[0] + (tile_height / 2))

            #     if j == 0:
            #         add_point_x = 0
            #     elif j == num_x_tiles - 1:
            #         add_point_x = image_width_px
            #     else:
            #         add_point_x = round(region[1] + (tile_width / 2))

            #     add_point = [add_point_x, add_point_y]
            #     all_points.append(add_point)
            #     predicted_values.append(val)
            # point = [round((region[1] + region[3]) / 2), round((region[0] + region[2]) / 2)]
            # point = [round(region[1]), round(region[0])]
            # print("\t{}".format(point))


    # print("predicted_values", predicted_values)
    # print("all_points", all_points)

            

    

    # # q = 0
    # STEP_SIZE = round(px_per_m / 4)
    # for i in range(0, image_height_px, STEP_SIZE):
    #     # r = 0
    #     for j in range(0, image_width_px, STEP_SIZE):
    #         region = [i, j, min(i + px_per_m, image_height_px), min(j + px_per_m, image_width_px)]
    #         # if fully_annotated:
    #         #     contained_boxes = box_utils.get_contained_inds(annotation_boxes, [region])
    #         #     annotated_values.append(contained_boxes.shape[0])

    #         # if pred_path is not None:
    #         contained_pred_boxes = box_utils.get_contained_inds(pred_boxes, [region])
    #         predicted_values.append(contained_pred_boxes.shape[0])

    #         all_points.append([j+round(STEP_SIZE/2), i+round(STEP_SIZE/2)])
    #         # all_points.append([r, q])
    #         # r += 1
    #     # q += 1

    all_points = np.array(all_points, dtype=np.float64)


    # print("all_points", all_points)
    # print("number of annotated values", len(annotated_values))
    # print("number of predicted values", len(predicted_values))
    # print("np.max(predicted_values)", np.max(predicted_values))
    # print("predicted_values", predicted_values)

    min_x = np.min(all_points[:,0])
    max_x = np.max(all_points[:,0])

    min_y = np.min(all_points[:,1])
    max_y = np.max(all_points[:,1])

    # print("min_x", min_x)
    # print("min_y", min_y)
    # print("max_x", max_x)
    # print("max_y", max_y)
    

    all_points[:,0] = range_map(all_points[:,0], min_x, max_x, 0, 1)
    all_points[:,1] = range_map(all_points[:,1], min_y, max_y, 0, 1)
    # print("all_points", all_points)


    # print("min_x", np.min(all_points[:,0]))
    # print("min_y", np.min(all_points[:,1]))
    # print("max_x", np.max(all_points[:,0]))
    # print("max_y", np.max(all_points[:,1]))
    


    all_grid_x, all_grid_y = np.mgrid[np.min(all_points[:,0]):np.max(all_points[:,0]):1000j, 
                                      np.max(all_points[:,1]):np.min(all_points[:,1]):1000j] #np.min(all_points[:,1]):np.max(all_points[:,1]):1000j]    



    # if fully_annotated:
    #     # completed_points[:,0] = range_map(completed_points[:,0], min_x, max_x, 0, 1)
    #     # completed_points[:,1] = range_map(completed_points[:,1], min_y, max_y, 0, 1)


    #     # grid_x, grid_y = np.mgrid[np.min(completed_points[:,0]):np.max(completed_points[:,0]):1000j, 
    #     #                         np.min(completed_points[:,1]):np.max(completed_points[:,1]):1000j]

    #     grid_z0 = griddata(all_points, annotated_values, (all_grid_x, all_grid_y), method=interpolation)




    #     extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
    #             np.min(all_points[:,1]), np.max(all_points[:,1]))





    predicted_values = np.array(predicted_values)

    # if completed_only:
    #     pred_grid_z0 = griddata(completed_points, predicted_values, (grid_x, grid_y), method=interpolation)

    #     pred_extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
    #                 np.min(completed_points[:,1]), np.max(completed_points[:,1]))
    # else:

    # print("np.max(predicted_values)", np.max(predicted_values))
    pred_grid_z0 = griddata(all_points, predicted_values, (all_grid_x, all_grid_y), method=interpolation)

    # print("pred_grid_z0", pred_grid_z0)
    # print("np.max(pred_grid_z0)", np.max(pred_grid_z0))


    pred_extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
                    np.min(all_points[:,1]), np.max(all_points[:,1]))

    # print("pred_extent", pred_extent)


    # if fully_annotated and (pred_path is not None):
    #     vmax = m.ceil(max(np.max(predicted_values), np.max(annotated_values)))
    # elif fully_annotated:
    #     vmax = m.ceil(np.max(annotated_values))
    # else:
    vmax = m.ceil(np.max(predicted_values))

    # if pred_path is None:
        
    # elif num_completed < 3:
    #     vmax = m.ceil(np.max(predicted_values))
    # else:
        

    vmin = 0

    colors = ["wheat", "forestgreen"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # if num_completed >= 3:
    # if fully_annotated:

    #     out_path = os.path.join(out_dir, map_download_uuid + "_annotated_map.svg")
    #     create_plot(grid_z0, extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)


    # if pred_path is not None:

    out_path = os.path.join(out_dir, interpolation + "_predicted_map.svg")
    create_plot(pred_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)

    min_max_rec = {
        "vmin": vmin,
        "vmax": vmax
    }
    min_max_rec_path = os.path.join(out_dir, interpolation + "_min_max_rec.json")
    json_io.save_json(min_max_rec_path, min_max_rec)


def create_interpolation_map(username, farm_name, field_name, mission_date, 
                             predictions_path, out_dir, class_index, interpolated_value, interpolation, 
                             tile_size, vegetation_record_path):

    # farm_name = dataset.farm_name
    # field_name = dataset.field_name
    # mission_date = dataset.mission_date

    #annotations_path = os.path.join("usr", "data", "image_sets",
    #                    farm_name, field_name, mission_date,
    #                    "annotations", "annotations_w3c.json")
    # annotations = json_io.load_json(annotations_path)



    # annotations = annotation_utils.load_annotations(annotations_path)

    #images_root = os.path.join("usr", "data", "image_sets",
    #                    farm_name, field_name, mission_date, "images")

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    if metadata["is_ortho"] == "yes":
        create_interpolation_map_for_ortho(username, farm_name, field_name, mission_date, 
                    predictions_path, out_dir, class_index, interpolation, tile_size)
    else:
        create_interpolation_map_for_image_set(username, farm_name, field_name, mission_date, 
                    predictions_path, out_dir, class_index, interpolated_value, interpolation, vegetation_record_path)


def create_interpolation_map_for_image_set(username, farm_name, field_name, mission_date, 
                                           predictions_path, out_dir, class_index, interpolated_value, interpolation, vegetation_record_path):

    # if pred_path is not None:
    predictions = json_io.load_json(predictions_path)

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    if vegetation_record_path:
        vegetation_record = json_io.load_json(vegetation_record_path)


    if (metadata["missing"]["latitude"] or metadata["missing"]["longitude"]) or metadata["camera_height"] == "":# or metadata["missing"]["area_m2"]:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)


    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]

    if make not in camera_specs:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    if model not in camera_specs[make]:
        raise RuntimeError("Cannot compute map due to missing metadata.")

    camera_entry = camera_specs[make][model]
    sensor_height = camera_entry["sensor_height"]
    sensor_width = camera_entry["sensor_width"]
    focal_length = camera_entry["focal_length"]

    camera_height = metadata["camera_height"]

    all_points = []
    predicted_values = []
    # completed_points = []
    # annotated_values = []
    # for image_name in annotations.keys():
    # for image_name in predictions.keys():
    for image_name in metadata["images"].keys():
        lon = metadata["images"][image_name]["longitude"]
        lat = metadata["images"][image_name]["latitude"]

        all_points.append([lon, lat])

        # status = annotations[image_name]["status"]

        gsd_h = (camera_height * sensor_height) / (focal_length * metadata["images"][image_name]["height_px"])
        gsd_w = (camera_height * sensor_width) / (focal_length * metadata["images"][image_name]["width_px"])

        gsd = min(gsd_h, gsd_w)

        image_height_px = metadata["images"][image_name]["height_px"]
        image_width_px = metadata["images"][image_name]["width_px"]

        image_height_m = image_height_px * gsd
        image_width_m = image_width_px * gsd

        area_m2 = image_width_m * image_height_m

        # fully_annotated = annotation_utils.is_fully_annotated(annotations, image_name, image_width_px, image_height_px)
        # print("is {} fully annotated? {}".format(image_name, fully_annotated))

        # if fully_annotated: #status == "completed_for_training" or status == "completed_for_testing":

        #     #print("image_name", image_name)
        #     annotated_value = annotations[image_name]["boxes"].shape[0] / area_m2
        #     completed_points.append([lon, lat])
        #     annotated_values.append(annotated_value)

        # if pred_path is not None:
            
        #     if not completed_only or fully_annotated: #(status == "completed_for_training" or status == "completed_for_testing"):  
        #         #predicted_value = len(predictions["image_predictions"][image_name]["pred_image_abs_boxes"]) / metadata["images"][image_name]["area_m2"]
        #         # v = 0
        #         # for annotation in predictions[image_name]["annotations"]:
        #         #     for b in annotation["body"]:
        #         #         if b["purpose"] == "score" and float(b["value"]) >= 0.5:
        #         #             v += 1
                
        #         #predicted_value = len(predictions[image_name]["annotations"]) 

        if interpolated_value == "obj_density":
            if image_name in predictions:
                if class_index == -1:
                    scores = np.array(predictions[image_name]["scores"])
                else:
                    class_mask = np.array(predictions[image_name]["classes"]) == class_index
                    scores = np.array(predictions[image_name]["scores"])[class_mask]
                predicted_value = np.sum(scores > 0.50) / area_m2
            else:
                predicted_value = 0
        else:
            perc_veg = vegetation_record[image_name]["vegetation_percentage"]["image"]

            if class_index == -1:
                class_name = "All Classes"
            else:
                class_name = metadata["object_classes"][class_index]

            perc_veg_obj = vegetation_record[image_name]["obj_vegetation_percentage"][class_name]["image"]
            perc_veg_non_obj = vegetation_record[image_name]["vegetation_percentage"]["image"] - vegetation_record[image_name]["obj_vegetation_percentage"][class_name]["image"]

            if interpolated_value == "perc_veg":
                predicted_value = perc_veg
            elif interpolated_value == "perc_veg_obj":
                predicted_value = perc_veg_obj
            elif interpolated_value == "perc_veg_non_obj":
                predicted_value = perc_veg_non_obj

        predicted_values.append(predicted_value)

    # print("predicted_values", predicted_values)
    # print("annotated_values", annotated_values)


    all_points = np.array(all_points)
    # completed_points = np.array(completed_points)
    # annotated_values = np.array(annotated_values)

    # num_completed = annotated_values.size 

    # print("num_completed", num_completed)
    if len(predicted_values) < 3:
        raise RuntimeError("Insufficient number of images for a map")
    # if num_completed < 3 and pred_path is None:
    #     raise RuntimeError("Unable to create an interpolated map with the provided arguments.")

    # if num_completed < 3 and completed_only:
    #     raise RuntimeError("If only fully-annotated images are to be used for the predicted map, at least three images must be fully annotated.")

    # if pred_path is not None:
        

    min_x = np.min(all_points[:,0])
    max_x = np.max(all_points[:,0])

    min_y = np.min(all_points[:,1])
    max_y = np.max(all_points[:,1])

    #points[:,0] = points[:,0] / max_x
    #points[:,1] = points[:,1] / max_y

    all_points[:,0] = range_map(all_points[:,0], min_x, max_x, 0, 1)
    all_points[:,1] = range_map(all_points[:,1], min_y, max_y, 0, 1)

    # if num_completed >= 3:
    #     completed_points[:,0] = range_map(completed_points[:,0], min_x, max_x, 0, 1)
    #     completed_points[:,1] = range_map(completed_points[:,1], min_y, max_y, 0, 1)


    #     grid_x, grid_y = np.mgrid[np.min(completed_points[:,0]):np.max(completed_points[:,0]):1000j, 
    #                             np.min(completed_points[:,1]):np.max(completed_points[:,1]):1000j]

    #     grid_z0 = griddata(completed_points, annotated_values, (grid_x, grid_y), method=interpolation)


    #     extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
    #             np.min(completed_points[:,1]), np.max(completed_points[:,1]))


    all_grid_x, all_grid_y = np.mgrid[np.min(all_points[:,0]):np.max(all_points[:,0]):1000j, 
                              np.min(all_points[:,1]):np.max(all_points[:,1]):1000j]                        


    # if pred_path is not None:

    predicted_values = np.array(predicted_values)

    # if completed_only:
    #     pred_grid_z0 = griddata(completed_points, predicted_values, (grid_x, grid_y), method=interpolation)

    #     pred_extent = (np.min(completed_points[:,0]), np.max(completed_points[:,0]),
    #                 np.min(completed_points[:,1]), np.max(completed_points[:,1]))
    # else:


    pred_grid_z0 = griddata(all_points, predicted_values, (all_grid_x, all_grid_y), method=interpolation)

    pred_extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
                    np.min(all_points[:,1]), np.max(all_points[:,1]))

    # print("np.max(predicted_values)", np.max(predicted_values))
    # print("pred_grid_z0", pred_grid_z0)
    # print("np.max(pred_grid_z0)", np.nanmax(pred_grid_z0))

    # if comparison_type == "side_by_side":

    # if pred_path is None:
    #     vmax = m.ceil(np.max(annotated_values))
    # elif num_completed < 3:
    vmax = m.ceil(np.max(predicted_values))
    # else:
    #     vmax = m.ceil(max(np.max(predicted_values), np.max(annotated_values)))

    vmin = 0

    colors = ["wheat", "forestgreen"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # if num_completed >= 3:

    #     out_path = os.path.join(out_dir, map_download_uuid + "_annotated_map.svg")
    #     create_plot(grid_z0, extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)


    # if pred_path is not None:

    out_path = os.path.join(out_dir, interpolation + "_predicted_map.svg")
    create_plot(pred_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)

    # elif comparison_type == "diff":

    #     diff_grid_z0 = pred_grid_z0 - grid_z0

    #     vmin = m.floor(np.nanmin(diff_grid_z0))
    #     vmax = m.ceil(np.nanmax(diff_grid_z0))
    #     vlim = max(abs(vmin), abs(vmax))
    #     vmin = -vlim
    #     vmax = vlim


    #     colors = ["royalblue", "oldlace", "tomato"] #"ghostwhite", "tomato"]
    #     #colors = ["royalblue", "whitesmoke", "tomato"]  #"tomato"]
    #     cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    #     out_path = os.path.join(out_dir, "difference_map.svg")
    #     create_plot(diff_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)


    min_max_rec = {
        "vmin": vmin,
        "vmax": vmax
    }
    min_max_rec_path = os.path.join(out_dir, interpolation + "_min_max_rec.json")
    json_io.save_json(min_max_rec_path, min_max_rec)



def remove_all_maps():
    image_set_root = os.path.join("usr", "data", "image_sets")
    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)
                maps_path = os.path.join(mission_path, "maps")
                if os.path.exists(maps_path):
                    shutil.rmtree(maps_path)


# def test():
#     ds = DataSet({"farm_name": "UNI", "field_name": "LowN1", "mission_date": "2021-06-07"})
#     predictions_path = os.path.join("usr", "data", "results", "UNI", "LowN1", "2021-06-07", 
#     "7a404942-dcd6-414c-a4aa-fcc39db4425b", "b4a13b7f-5e4d-4818-8c14-ea4095900fb3", "predictions.json")
#     predictions = json_io.load_json(predictions_path)
#     create_interpolation_map(ds, predictions)

def remove_old_maps(out_dir):
    if os.path.exists(out_dir):
        for f_path in glob.glob(os.path.join(out_dir, "*")):
            alive_time = time.time() - os.path.getmtime(f_path)
            if alive_time > RESULT_LIFETIME:
                os.remove(f_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str)
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    # parser.add_argument("annotations_path", type=str)
    parser.add_argument("predictions_path", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("class_index", type=int)    
    parser.add_argument("interpolated_value", type=str)
    # parser.add_argument("map_download_uuid", type=str)
    #parser.add_argument('-density', action='store_true')

    parser.add_argument("-nearest", action='store_true')
    # parser.add_argument('-completed_only', action='store_true')
    parser.add_argument("-tile_size", type=float)

    parser.add_argument("-vegetation_record_path", type=str)


    # parser.add_argument('-diff', action='store_true')
    args = parser.parse_args()
    

    # print("running map builder")
    # print(args.username)
    # print(args.farm_name)
    # print(args.field_name)
    # print(args.mission_date) 
    # print(args.annotations_path) 
    # print(args.out_dir)
    # print(args.map_download_uuid)
    # print(args.pred_path)  
    # dataset = DataSet({
    #     "farm_name": args.farm_name,
    #     "field_name": args.field_name,
    #     "mission_date": args.mission_date
    # })

    # if args.density:
    #     metric = "density"
    # else:
    #     metric = "count"

    if args.nearest:
        interpolation = "nearest"
    else:
        interpolation = "linear"

    if args.interpolated_value != "obj_density" and not args.vegetation_record_path:
        raise RuntimeError("Require vegetation record path")

    valid_values = ["obj_density", "perc_veg", "perc_veg_obj", "perc_veg_non_obj"]
    if args.interpolated_value not in valid_values:
        raise RuntimeError("Invalid interpolated value: {}".format(args.interpolated_value))

    # if args.diff:
    #     comparison_type = "diff"
    # else:
    # comparison_type = "side_by_side"
    # value = "plant_density"

    remove_old_maps(args.out_dir)


    create_interpolation_map(args.username,
                            args.farm_name,
                            args.field_name,
                            args.mission_date,
                            args.predictions_path, #annotations_path,
                            args.out_dir, 
                            args.class_index, 
                            args.interpolated_value,

                            # args.map_download_uuid,
                            interpolation=interpolation,
                            tile_size=args.tile_size,
                            vegetation_record_path=args.vegetation_record_path)
                            # completed_only=args.completed_only,
                            #pred_path=args.pred_path,
                            # value=value)
                            # comparison_type=comparison_type)