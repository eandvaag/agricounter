import os
import math as m
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

from models.common import annotation_utils


MAX_NUM_TILES = 75000


from io_utils import json_io

def range_map(old_val, old_min, old_max, new_min, new_max):
    new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_val


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

    plt.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0)


def create_interpolation_map_for_ortho(username, farm_name, field_name, mission_date, 
                    prediction_dir, out_dir, class_index, interpolation, tile_size):

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)


    if metadata["camera_height"] == "":
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

    predictions = annotation_utils.load_predictions_from_dir(prediction_dir)
    image_name = list(metadata["images"].keys())[0]

    image_height_px = metadata["images"][image_name]["height_px"]
    image_width_px = metadata["images"][image_name]["width_px"]


    gsd_h = (camera_height * sensor_height) / (focal_length * raw_image_height_px)
    gsd_w = (camera_height * sensor_width) / (focal_length * raw_image_width_px)

    gsd = min(gsd_h, gsd_w)

    image_height_m = image_height_px * gsd
    image_width_m = image_width_px * gsd


    if image_name in predictions:
        if class_index == -1:
            class_mask = np.full(predictions[image_name]["classes"].size, True)
        else:
            class_mask = predictions[image_name]["classes"] == class_index
        score_mask = predictions[image_name]["scores"] > 0.50
        mask = np.logical_and(class_mask, score_mask)
        pred_boxes = predictions[image_name]["boxes"][mask]
    else:
        pred_boxes = np.array([])

    predicted_values = []
    all_points = []

    tile_width_m = tile_size
    tile_height_m = tile_size

    num_x_tiles = round(image_width_m / tile_width_m)
    num_y_tiles = round(image_height_m / tile_height_m)



    tile_width = image_width_px / num_x_tiles
    tile_height = image_height_px / num_y_tiles

    if num_x_tiles * num_y_tiles > MAX_NUM_TILES:
        raise RuntimeError("Unable to create density map: too many tiles requested.")

    area_m2_per_tile = (tile_width * gsd) * (tile_height * gsd)
    box_centres = np.rint((pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0).astype(np.int64)

    y_regions = np.round(box_centres[:, 0] / tile_height).astype(np.int64)
    x_regions = np.round(box_centres[:, 1] / tile_width).astype(np.int64)

    for i in range(num_y_tiles):
        for j in range(num_x_tiles):

            region = [i * tile_height, j * tile_width, (i+1) * tile_height, (j+1) * tile_width]

            y_mask = i == y_regions
            x_mask = j == x_regions
            mask = np.logical_and(y_mask, x_mask)
            num_boxes_in_region = np.sum(mask)
            val = num_boxes_in_region / area_m2_per_tile
            predicted_values.append(val)
            point_y = round(region[0] + (tile_height / 2))
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
        

    all_points = np.array(all_points, dtype=np.float64)

    min_x = np.min(all_points[:,0])
    max_x = np.max(all_points[:,0])

    min_y = np.min(all_points[:,1])
    max_y = np.max(all_points[:,1])

    all_points[:,0] = range_map(all_points[:,0], min_x, max_x, 0, 1)
    all_points[:,1] = range_map(all_points[:,1], min_y, max_y, 0, 1)

    all_grid_x, all_grid_y = np.mgrid[np.min(all_points[:,0]):np.max(all_points[:,0]):1000j, 
                                      np.max(all_points[:,1]):np.min(all_points[:,1]):1000j]

    predicted_values = np.array(predicted_values)

    pred_grid_z0 = griddata(all_points, predicted_values, (all_grid_x, all_grid_y), method=interpolation)

    pred_extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
                    np.min(all_points[:,1]), np.max(all_points[:,1]))

    vmax = m.ceil(np.max(predicted_values))
    vmin = 0

    colors = ["wheat", "forestgreen"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, interpolation + "_predicted_map.svg")
    create_plot(pred_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)

    min_max_rec = {
        "vmin": vmin,
        "vmax": vmax
    }
    min_max_rec_path = os.path.join(out_dir, interpolation + "_min_max_rec.json")
    json_io.save_json(min_max_rec_path, min_max_rec)


def create_interpolation_map(username, farm_name, field_name, mission_date, 
                             prediction_dir, out_dir, class_index, interpolated_value, interpolation, 
                             tile_size, vegetation_record_path):

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    if metadata["is_ortho"]:
        create_interpolation_map_for_ortho(username, farm_name, field_name, mission_date, 
                    prediction_dir, out_dir, class_index, interpolation, tile_size)
    else:
        create_interpolation_map_for_image_set(username, farm_name, field_name, mission_date, 
                    prediction_dir, out_dir, class_index, interpolated_value, interpolation, vegetation_record_path)


def create_interpolation_map_for_image_set(username, farm_name, field_name, mission_date, 
                                           prediction_dir, out_dir, class_index, interpolated_value, interpolation, vegetation_record_path):

    predictions = annotation_utils.load_predictions_from_dir(prediction_dir)

    metadata_path = os.path.join("usr", "data", username, "image_sets",
                        farm_name, field_name, mission_date,
                        "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    if vegetation_record_path:
        vegetation_record = json_io.load_json(vegetation_record_path)


    if (metadata["missing"]["latitude"] or metadata["missing"]["longitude"]) or metadata["camera_height"] == "":
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
    for image_name in metadata["images"].keys():
        lon = metadata["images"][image_name]["longitude"]
        lat = metadata["images"][image_name]["latitude"]

        all_points.append([lon, lat])

        gsd_h = (camera_height * sensor_height) / (focal_length * metadata["images"][image_name]["height_px"])
        gsd_w = (camera_height * sensor_width) / (focal_length * metadata["images"][image_name]["width_px"])

        gsd = min(gsd_h, gsd_w)

        image_height_px = metadata["images"][image_name]["height_px"]
        image_width_px = metadata["images"][image_name]["width_px"]

        image_height_m = image_height_px * gsd
        image_width_m = image_width_px * gsd

        area_m2 = image_width_m * image_height_m


        if interpolated_value == "obj_density":
            if image_name in predictions:
                if class_index == -1:
                    scores = predictions[image_name]["scores"]
                else:
                    class_mask = predictions[image_name]["classes"] == class_index
                    scores = predictions[image_name]["scores"][class_mask]
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

    all_points = np.array(all_points)

    if len(predicted_values) < 3:
        raise RuntimeError("Insufficient number of images for a map")

    min_x = np.min(all_points[:,0])
    max_x = np.max(all_points[:,0])

    min_y = np.min(all_points[:,1])
    max_y = np.max(all_points[:,1])

    all_points[:,0] = range_map(all_points[:,0], min_x, max_x, 0, 1)
    all_points[:,1] = range_map(all_points[:,1], min_y, max_y, 0, 1)

    all_grid_x, all_grid_y = np.mgrid[np.min(all_points[:,0]):np.max(all_points[:,0]):1000j, 
                              np.min(all_points[:,1]):np.max(all_points[:,1]):1000j]                        


    predicted_values = np.array(predicted_values)

    pred_grid_z0 = griddata(all_points, predicted_values, (all_grid_x, all_grid_y), method=interpolation)

    pred_extent = (np.min(all_points[:,0]), np.max(all_points[:,0]),
                    np.min(all_points[:,1]), np.max(all_points[:,1]))

    vmax = m.ceil(np.max(predicted_values))
    vmin = 0

    colors = ["wheat", "forestgreen"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    out_path = os.path.join(out_dir, interpolation + "_predicted_map.svg")
    create_plot(pred_grid_z0, pred_extent, vmin=vmin, vmax=vmax, cmap=cmap, out_path=out_path)


    min_max_rec = {
        "vmin": vmin,
        "vmax": vmax
    }
    min_max_rec_path = os.path.join(out_dir, interpolation + "_min_max_rec.json")
    json_io.save_json(min_max_rec_path, min_max_rec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str)
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    parser.add_argument("prediction_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("class_index", type=int)    
    parser.add_argument("interpolated_value", type=str)
    parser.add_argument("-nearest", action='store_true')
    parser.add_argument("-tile_size", type=float)
    parser.add_argument("-vegetation_record_path", type=str)


    args = parser.parse_args()
    

    if args.nearest:
        interpolation = "nearest"
    else:
        interpolation = "linear"

    if args.interpolated_value != "obj_density" and not args.vegetation_record_path:
        raise RuntimeError("Require vegetation record path")

    valid_values = ["obj_density", "perc_veg", "perc_veg_obj", "perc_veg_non_obj"]
    if args.interpolated_value not in valid_values:
        raise RuntimeError("Invalid interpolated value: {}".format(args.interpolated_value))

    create_interpolation_map(args.username,
                            args.farm_name,
                            args.field_name,
                            args.mission_date,
                            args.prediction_dir,
                            args.out_dir, 
                            args.class_index, 
                            args.interpolated_value,
                            interpolation=interpolation,
                            tile_size=args.tile_size,
                            vegetation_record_path=args.vegetation_record_path)