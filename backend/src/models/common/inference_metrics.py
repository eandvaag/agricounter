import logging
import tqdm
import os
import time
import datetime
import math as m
import numpy as np
import tensorflow as tf
from mean_average_precision import MetricBuilder


from scipy.spatial import Voronoi
import shapely.geometry
import shapely.ops


import pandas as pd
import pandas.io.formats.excel
from natsort import index_natsorted


from models.common import box_utils, annotation_utils, poly_utils


from io_utils import json_io






def get_mAP_val(annotations, full_predictions, iou_thresh, assessment_images):
    
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for image_name in tqdm.tqdm(assessment_images):
        annotated_boxes = np.array(annotations[image_name]["boxes"])
        predicted_scores = np.array(full_predictions[image_name]["scores"])
        predicted_boxes = np.array(full_predictions[image_name]["boxes"])

        annotated_classes = np.zeros(shape=(annotated_boxes.shape[0]))
        predicted_classes = np.zeros(shape=(predicted_boxes.shape[0]))

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
            predicted_boxes, 
            predicted_classes, 
            predicted_scores,
            annotated_boxes,
            annotated_classes)

        metric_fn.add(pred_for_mAP, true_for_mAP)

    if iou_thresh == ".50:.05:.95":
        mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".50":
        mAP = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".75":
        mAP = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    elif iou_thresh == ".20":
        mAP = metric_fn.value(iou_thresholds=0.2, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    
    else:
        raise RuntimeError("Invalid IoU threshold: {}".format(iou_thresh))

    return mAP




def get_positives_and_negatives(annotated_boxes, predicted_boxes, iou_thresh):


    num_annotated = annotated_boxes.shape[0]
    num_predicted = predicted_boxes.shape[0]

    annotated_boxes = box_utils.swap_xy_np(annotated_boxes)
    predicted_boxes = box_utils.swap_xy_np(predicted_boxes)

    matches = np.full(num_predicted, -1)

    MAX_MAT_SIZE = 16000000
    STEP_SIZE = min(num_predicted, m.floor(MAX_MAT_SIZE / num_predicted))

    for i in range(0, num_predicted, STEP_SIZE):
        iou_mat = box_utils.compute_iou(
                    tf.convert_to_tensor(annotated_boxes, dtype=tf.float64),
                    tf.convert_to_tensor(predicted_boxes[i:i+STEP_SIZE, :], dtype=tf.float64), 
                    box_format="corners_xy").numpy()



        max_inds = np.argmax(iou_mat, axis=0)
        max_vals = np.take_along_axis(iou_mat, np.expand_dims(max_inds, axis=0), axis=0)[0]
        mask = max_vals >= iou_thresh
        matches[i:i+STEP_SIZE][mask] = max_inds[mask]





    matched_elements = matches[matches > -1]
    true_positive = np.sum(np.unique(matches) != -1)
    false_positive = np.sum(matches == -1) + (len(matched_elements) - len(np.unique(matched_elements)))
    false_negative = num_annotated - true_positive


    return int(true_positive), int(false_positive), int(false_negative)




def collect_image_set_metrics(predictions, annotations, metadata):


    logger = logging.getLogger(__name__)
    logger.info("Started collecting image set metrics.")


    metrics = {}
    metric_keys = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)"
    ]


    for obj_ind, obj_class in enumerate(metadata["object_classes"] + ["All Classes"]):
        metrics[obj_class] = {}

        for metric_key in metric_keys:
            metrics[obj_class][metric_key] = {}

            for image_name in predictions.keys():
                metrics[obj_class][metric_key][image_name] = {}

                for region_key in ["regions_of_interest", "fine_tuning_regions", "test_regions"]:
                    metrics[obj_class][metric_key][image_name][region_key] = []


    for obj_ind, obj_class in enumerate(metadata["object_classes"] + ["All Classes"]):
        for image_name in predictions.keys():

            anno_boxes = annotations[image_name]["boxes"]
            anno_classes = annotations[image_name]["classes"]
            pred_boxes = predictions[image_name]["boxes"]
            pred_scores = predictions[image_name]["scores"]
            pred_classes = predictions[image_name]["classes"]

            if obj_class != "All Classes":
                anno_class_mask = anno_classes == obj_ind
                anno_boxes = anno_boxes[anno_class_mask]

                pred_class_mask = pred_classes == obj_ind
                pred_boxes = pred_boxes[pred_class_mask]
                pred_scores = pred_scores[pred_class_mask]

            for region_key in ["regions_of_interest", "fine_tuning_regions", "test_regions"]:
            

                for region in annotations[image_name][region_key]:
                    annotated_centres = (anno_boxes[..., :2] + anno_boxes[..., 2:]) / 2.0
                    predicted_centres = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2.0
                    

                    annotated_inds = poly_utils.get_contained_inds_for_points(annotated_centres, [region])
                    predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])

                    region_anno_boxes = anno_boxes[annotated_inds]
                    region_pred_boxes = pred_boxes[predicted_inds]
                    region_pred_scores = pred_scores[predicted_inds]


                    sel_region_pred_boxes = region_pred_boxes[region_pred_scores > 0.50]
                    
                    num_predicted = sel_region_pred_boxes.shape[0]
                    num_annotated = region_anno_boxes.shape[0]

                    if num_predicted > 0:
                        if num_annotated > 0:
                            true_positive, false_positive, false_negative = get_positives_and_negatives(region_anno_boxes, sel_region_pred_boxes, 0.50)
                            print("anno: {}, pred: {}, tp: {}, fp: {}, fn: {}".format(
                                num_predicted, num_annotated, true_positive, false_positive, false_negative))
                            precision_050 = true_positive / (true_positive + false_positive)
                            recall_050 = true_positive / (true_positive + false_negative)
                            if precision_050 == 0 and recall_050 == 0:
                                f1_iou_050 = 0
                            else:
                                f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
                            acc_050 = true_positive / (true_positive + false_positive + false_negative)     
                        
                        else:
                            true_positive = 0
                            false_positive = num_predicted
                            false_negative = 0

                            precision_050 = 0.0
                            recall_050 = 0.0
                            f1_iou_050 = 0.0
                            acc_050 = 0.0
                    else:
                        if num_annotated > 0:
                            true_positive = 0
                            false_positive = 0
                            false_negative = num_annotated

                            precision_050 = 0.0
                            recall_050 = 0.0
                            f1_iou_050 = 0.0
                            acc_050 = 0.0
                        else:
                            true_positive = 0
                            false_positive = 0
                            false_negative = 0

                            precision_050 = 1.0
                            recall_050 = 1.0
                            f1_iou_050 = 1.0
                            acc_050 = 1.0

                    metrics[obj_class]["True Positives (IoU=.50, conf>.50)"][image_name][region_key].append(true_positive)
                    metrics[obj_class]["False Positives (IoU=.50, conf>.50)"][image_name][region_key].append(false_positive)
                    metrics[obj_class]["False Negatives (IoU=.50, conf>.50)"][image_name][region_key].append(false_negative)
                    metrics[obj_class]["Precision (IoU=.50, conf>.50)"][image_name][region_key].append(precision_050)
                    metrics[obj_class]["Recall (IoU=.50, conf>.50)"][image_name][region_key].append(recall_050)
                    metrics[obj_class]["Accuracy (IoU=.50, conf>.50)"][image_name][region_key].append(acc_050)
                    metrics[obj_class]["F1 Score (IoU=.50, conf>.50)"][image_name][region_key].append(f1_iou_050)


    return metrics





def get_AP_vals(annotated_boxes, predicted_boxes, predicted_scores):
    

    print("AP: num_annotated_boxes: {}, num_predicted_boxes: {}".format(
        annotated_boxes.shape[0], predicted_boxes.shape[0]))

    NUM_BOXES_THRESH = 10000

    if (annotated_boxes.shape[0] * predicted_boxes.shape[0]) < (NUM_BOXES_THRESH * NUM_BOXES_THRESH):

        annotated_classes = np.zeros(shape=(annotated_boxes.shape[0]))
        predicted_classes = np.zeros(shape=(predicted_boxes.shape[0]))

        pred_for_mAP, true_for_mAP = get_pred_and_true_for_mAP(
            predicted_boxes, 
            predicted_classes, 
            predicted_scores,
            annotated_boxes,
            annotated_classes)

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        metric_fn.add(pred_for_mAP, true_for_mAP)

        ms_coco_mAP = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        mAP_IoU_50 = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        mAP_IoU_75 = metric_fn.value(iou_thresholds=0.75, recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']

        return {
            "AP (IoU=.50:.05:.95)": float(ms_coco_mAP) * 100,
            "AP (IoU=.50)": float(mAP_IoU_50) * 100,
            "AP (IoU=.75)": float(mAP_IoU_75) * 100
        }
    
    else:
        return {
            "AP (IoU=.50:.05:.95)": "unable_to_calculate",
            "AP (IoU=.50)": "unable_to_calculate",
            "AP (IoU=.75)": "unable_to_calculate"
        }




def can_calculate_density(metadata, camera_specs):

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]

    if metadata["camera_height"] == "":
        return False

    if make not in camera_specs:
        return False
    
    if model not in camera_specs[make]:
        return False

    return True

def calculate_area_m2(gsd, area_px):

    area_m2 = area_px * (gsd ** 2)
    return area_m2

def get_gsd(camera_specs, metadata):

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]
    camera_entry = camera_specs[make][model]

    gsd_h = (metadata["camera_height"] * camera_entry["sensor_height"]) / \
            (camera_entry["focal_length"] * camera_entry["image_height_px"])

    gsd_w = (metadata["camera_height"] * camera_entry["sensor_width"]) / \
            (camera_entry["focal_length"] * camera_entry["image_width_px"])

    gsd = min(gsd_h, gsd_w)

    return gsd

def create_spreadsheet(job, regions_only=False):

    username = job["username"]
    farm_name = job["farm_name"]
    field_name = job["field_name"]
    mission_date = job["mission_date"]
    result_uuid = job["result_uuid"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                farm_name, field_name, mission_date)

    result_dir = os.path.join(image_set_dir, "model", "results", "available", result_uuid)


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)
    
    # predictions_path = os.path.join(result_dir, "predictions.json")
    # predictions = annotation_utils.load_predictions(predictions_path)

    predictions_dir = os.path.join(result_dir, "predictions")
    predictions = annotation_utils.load_predictions_from_dir(predictions_dir)

    full_predictions_path = os.path.join(result_dir, "full_predictions.json")
    full_predictions = annotation_utils.load_predictions(full_predictions_path)


    annotations_path = os.path.join(result_dir, "annotations.json")
    vegetation_record_path = os.path.join(result_dir, "vegetation_record.json")
    metrics_path = os.path.join(result_dir, "metrics.json")
    metrics = json_io.load_json(metrics_path)
    
    
    tags_path = os.path.join(result_dir, "tags.json")

    annotations = annotation_utils.load_annotations(annotations_path)

    if os.path.exists(vegetation_record_path):
        vegetation_record = json_io.load_json(vegetation_record_path)
    else:
        vegetation_record = None

    tags = json_io.load_json(tags_path)



    args = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
        "predictions": predictions,
        "full_predictions": full_predictions,
        "annotations": annotations,
        "metadata": metadata,
        "camera_specs": camera_specs,
        "vegetation_record": vegetation_record,
        "tags": tags
    }

    updated_metrics = metrics

    if not regions_only:
        images_df = create_images_sheet(args, updated_metrics)
    regions_df = create_regions_sheet(args, updated_metrics)
    stats_df = create_stats_sheet(args, regions_df)

    pandas.io.formats.excel.ExcelFormatter.header_style = None

    out_path = os.path.join(result_dir, "metrics.xlsx")

    sheet_name_to_df = {}

    if not regions_only:
        sheet_name_to_df["Images"] = images_df
    sheet_name_to_df["Regions"] = regions_df
    sheet_name_to_df["Stats"] = stats_df

    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    fmt = writer.book.add_format({"font_name": "Courier New"})

    for sheet_name in sheet_name_to_df.keys():

        df = sheet_name_to_df[sheet_name]

        df.to_excel(writer, index=False, sheet_name=sheet_name, na_rep='NA')  # send df to writer
        worksheet = writer.sheets[sheet_name]  # pull worksheet object

        worksheet.set_column('A:ZZ', None, fmt)
        worksheet.set_row(0, None, fmt)

        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            if series.size > 0:
                max_entry_size = series.astype(str).map(len).max()
            else:
                max_entry_size = 0
            max_len = max((
                max_entry_size,  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width


    writer.close()







def create_images_sheet(args, updated_metrics):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    predictions = args["predictions"]
    annotations = args["annotations"]
    metadata = args["metadata"]
    camera_specs = args["camera_specs"]
    vegetation_record = args["vegetation_record"]


    include_density = can_calculate_density(metadata, camera_specs)

    num_classes = len(metadata["object_classes"])

    columns = [
        "Username",
        "Farm Name",
        "Field Name",
        "Mission Date",
        "Image Name",
        "Regions of Interest",
        "Fine-Tuning Regions",
        "Test Regions",
        "Image Is Fully Annotated",
        "Source Of Annotations",
    ]
    if num_classes > 1:
        columns.extend([
            "Annotated Count (All Classes)",
            "Predicted Count (All Classes)"
        ])

    for object_class in metadata["object_classes"]:
        columns.append("Annotated Count (" + object_class + ")")
        columns.append("Predicted Count (" + object_class + ")")


    if include_density:
        if num_classes > 1:
            columns.extend([
                "Annotated Count Per Square Metre (All Classes)",
                "Predicted Count Per Square Metre (All Classes)"
            ])

        for object_class in metadata["object_classes"]:
            columns.append("Annotated Count Per Square Metre (" + object_class + ")")
            columns.append("Predicted Count Per Square Metre (" + object_class + ")")




    columns.append("Area (Pixels)")
    if num_classes > 1:
        columns.append("Mean of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Mean of Predicted Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Mean of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Mean of Predicted Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")")

    if include_density:
        columns.extend(["Area (Square Metres)"])
        if num_classes > 1:
            columns.append("Mean of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Mean of Predicted Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (All Classes)")


        for object_class in metadata["object_classes"]:
            columns.append("Mean of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Mean of Predicted Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")")


    if num_classes > 1:
        columns.append("Percent Count Error (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Percent Count Error (" + object_class + ")")

    if vegetation_record is not None:
        columns.extend([
            "Excess Green Threshold",
            "Vegetation Percentage"
        ])
        if num_classes > 1:
            columns.extend([
                "Percentage of Vegetation Inside Object Boundaries",
                "Percentage of Vegetation Outside Object Boundaries"           
            ])
        for object_class in metadata["object_classes"]:
            columns.extend([
                "Percentage of Vegetation Inside " + object_class + " Boundaries",
                "Percentage of Vegetation Outside " + object_class + " Boundaries"    
            ])
        

    metrics_lst = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)"  
    ]

    if num_classes > 1:
        for metric in metrics_lst:
            columns.append(metric + " (All Classes)")


    for object_class in metadata["object_classes"]:
        for metric in metrics_lst:
            columns.append(metric + " (" + object_class + ")")


    d = {}
    for c in columns:
        d[c] = []


    for image_name in predictions.keys():

        regions_of_interest = annotations[image_name]["regions_of_interest"]
        fine_tuning_regions = annotations[image_name]["fine_tuning_regions"]
        test_regions = annotations[image_name]["test_regions"]

        annotations_source = annotations[image_name]["source"]

        image_height_px = metadata["images"][image_name]["height_px"]
        image_width_px = metadata["images"][image_name]["width_px"]

        if annotation_utils.is_fully_annotated_for_fine_tuning(annotations, image_name, image_width_px, image_height_px):
            fully_annotated = "yes: for fine-tuning"

        elif annotation_utils.is_fully_annotated_for_testing(annotations, image_name, image_width_px, image_height_px):
            fully_annotated = "yes: for testing"
            
        else:
            fully_annotated = "no"


        height_px = metadata["images"][image_name]["height_px"]
        width_px = metadata["images"][image_name]["width_px"]
        area_px = height_px * width_px

        d["Username"].append(username)
        d["Farm Name"].append(farm_name)
        d["Field Name"].append(field_name)
        d["Mission Date"].append(mission_date)
        d["Image Name"].append(image_name)
        d["Regions of Interest"].append(len(regions_of_interest))
        d["Fine-Tuning Regions"].append(len(fine_tuning_regions))
        d["Test Regions"].append(len(test_regions))
        d["Image Is Fully Annotated"].append(fully_annotated)
        d["Source Of Annotations"].append(annotations_source)
        d["Area (Pixels)"].append(area_px)
        if include_density:
            gsd = get_gsd(camera_specs, metadata)
            area_m2 = calculate_area_m2(gsd, area_px)
            d["Area (Square Metres)"].append(round(area_m2, 8))

        if num_classes > 1:
            cls_tups = [(-1, "All Classes")]
            for i, object_class in enumerate(metadata["object_classes"]):
                cls_tups.append((i, object_class))
        else:
            cls_tups = [(0, metadata["object_classes"][0])]

        for cls_tup in cls_tups:
            cls_idx = cls_tup[0]
            object_class = cls_tup[1]

            annotated_boxes = []
            predicted_boxes = []
            for i in range(len(annotations[image_name]["boxes"])):
                if cls_idx == -1 or annotations[image_name]["classes"][i] == cls_idx:
                    annotated_boxes.append(annotations[image_name]["boxes"][i])

            for i in range(len(predictions[image_name]["boxes"])):
                if cls_idx == -1 or predictions[image_name]["classes"][i] == cls_idx:
                    if predictions[image_name]["scores"][i] > 0.50:
                        predicted_boxes.append(predictions[image_name]["boxes"][i])

            annotated_boxes = np.array(annotated_boxes)
            predicted_boxes = np.array(predicted_boxes)

            annotated_count = annotated_boxes.shape[0]
            predicted_count = predicted_boxes.shape[0]

            d["Annotated Count (" + object_class + ")"].append(annotated_count)
            d["Predicted Count (" + object_class + ")"].append(predicted_count)


            if annotated_count > 0:
                annotated_box_areas_px = box_utils.box_areas_np(annotated_boxes)
                mean_annotated_object_area_px = round(np.mean(annotated_box_areas_px), 8)
                stdev_annotated_object_area_px = round(np.std(annotated_box_areas_px), 8)
            else:
                mean_annotated_object_area_px = "NA"
                stdev_annotated_object_area_px = "NA"

            if predicted_count > 0:
                predicted_box_areas_px = box_utils.box_areas_np(predicted_boxes)
                mean_predicted_object_area_px = round(np.mean(predicted_box_areas_px), 8)
                stdev_predicted_object_area_px = round(np.std(predicted_box_areas_px), 8)
            else:
                mean_predicted_object_area_px = "NA"
                stdev_predicted_object_area_px = "NA"

            d["Mean of Annotated Object Areas (Pixels) (" + object_class + ")"].append(mean_annotated_object_area_px)
            d["Mean of Predicted Object Areas (Pixels) (" + object_class + ")"].append(mean_predicted_object_area_px)
            d["Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")"].append(stdev_annotated_object_area_px)
            d["Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")"].append(stdev_predicted_object_area_px)


            if include_density:
                gsd = get_gsd(camera_specs, metadata)
                area_m2 = calculate_area_m2(gsd, area_px)
                if area_m2 > 0:
                    annotated_count_per_square_metre = round(annotated_count / area_m2, 8)
                    predicted_count_per_square_metre = round(predicted_count / area_m2, 8)
                else:
                    annotated_count_per_square_metre = "NA"
                    predicted_count_per_square_metre = "NA"
                d["Annotated Count Per Square Metre (" + object_class + ")"].append(annotated_count_per_square_metre)
                d["Predicted Count Per Square Metre (" + object_class + ")"].append(predicted_count_per_square_metre)

                if annotated_count > 0:
                    annotated_box_areas_m2 = calculate_area_m2(gsd, annotated_box_areas_px)
                    mean_annotated_object_area_m2 = round(np.mean(annotated_box_areas_m2), 8)
                    stdev_annotated_object_area_m2 = round(np.std(annotated_box_areas_m2), 8)
                else:
                    mean_annotated_object_area_m2 = "NA"
                    stdev_annotated_object_area_m2 = "NA"

                if predicted_count > 0:
                    predicted_box_areas_m2 = calculate_area_m2(gsd, predicted_box_areas_px)
                    mean_predicted_object_area_m2 = round(np.mean(predicted_box_areas_m2), 8)
                    stdev_predicted_object_area_m2 = round(np.std(predicted_box_areas_m2), 8)               
                else:
                    mean_predicted_object_area_m2 = "NA"
                    stdev_predicted_object_area_m2 = "NA"

                d["Mean of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(mean_annotated_object_area_m2)
                d["Mean of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(mean_predicted_object_area_m2)
                d["Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(stdev_annotated_object_area_m2)
                d["Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(stdev_predicted_object_area_m2)

            if fully_annotated == "no":
                percent_count_error = "NA"
            elif annotated_count > 0:
                percent_count_error = round(abs((predicted_count - annotated_count) / (annotated_count)) * 100, 2)
            else:
                percent_count_error = "NA"

            d["Percent Count Error (" + object_class + ")"].append(percent_count_error)


            if fully_annotated == "no":
                for metric in metrics_lst:
                    d[metric + " (" + object_class + ")"].append("NA")
            elif fully_annotated == "yes: for fine-tuning":
                for metric in metrics_lst:
                    metric_val = updated_metrics[object_class][metric][image_name]["fine_tuning_regions"][0]
                    if isinstance(metric_val, float):
                        metric_val = round(metric_val, 2)
                    d[metric + " (" + object_class + ")"].append(metric_val)
            else:
                region_index = -1
                for i in range(len(annotations[image_name]["test_regions"])):
                    region = annotations[image_name]["test_regions"][i]
                    if (region[0] == 0 and region[1] == 0) and (region[2] == image_height_px and region[3] == image_width_px):
                        region_index = i
                        break

                for metric in metrics_lst:
                    metric_val = updated_metrics[object_class][metric][image_name]["test_regions"][region_index]
                    if isinstance(metric_val, float):
                        metric_val = round(metric_val, 2)
                    d[metric + " (" + object_class + ")"].append(metric_val)


        if vegetation_record is not None:
            d["Excess Green Threshold"].append(vegetation_record[image_name]["sel_val"])
            vegetation_percentage = vegetation_record[image_name]["vegetation_percentage"]["image"]
            d["Vegetation Percentage"].append(vegetation_percentage)

            if num_classes > 1:
                obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"]["All Classes"]["image"]
                if vegetation_percentage == 0:
                    obj_percentage = "NA"
                    non_obj_percentage = "NA"
                else:
                    obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                    non_obj_percentage = round(100 - obj_percentage, 2)

                d["Percentage of Vegetation Inside Object Boundaries"].append(obj_percentage)
                d["Percentage of Vegetation Outside Object Boundaries"].append(non_obj_percentage) 

            for object_class in metadata["object_classes"]:

                obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"][object_class]["image"]
                if vegetation_percentage == 0:
                    obj_percentage = "NA"
                    non_obj_percentage = "NA"
                else:
                    obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                    non_obj_percentage = round(100 - obj_percentage, 2)

                d["Percentage of Vegetation Inside " + object_class + " Boundaries"].append(obj_percentage)
                d["Percentage of Vegetation Outside " + object_class + " Boundaries"].append(non_obj_percentage) 


    df = pd.DataFrame(data=d, columns=columns)
    df.sort_values(by="Image Name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["Image Name"])))
    return df


def create_areas_spreadsheet(job, regions_only=False):

    logger = logging.getLogger(__name__)

    username = job["username"]
    farm_name = job["farm_name"]
    field_name = job["field_name"]
    mission_date = job["mission_date"]
    result_uuid = job["result_uuid"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets",
                                farm_name, field_name, mission_date)

    result_dir = os.path.join(image_set_dir, "model", "results", "available", result_uuid)


    metadata_path = os.path.join(image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    camera_specs_path = os.path.join("usr", "data", username, "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)

    # predictions_path = os.path.join(result_dir, "predictions.json")
    # predictions = annotation_utils.load_predictions(predictions_path)

    predictions_dir = os.path.join(result_dir, "predictions")
    predictions = annotation_utils.load_predictions_from_dir(predictions_dir)

    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path) 

    if not can_calculate_density(metadata, camera_specs):
        logger.info("Cannot calculate voronoi areas (cannot calculate density).")
        return
    
    logger.info("Started collecting voronoi areas.")


    start_time = time.time()

    make = metadata["camera_info"]["make"]
    model = metadata["camera_info"]["model"]
    camera_entry = camera_specs[make][model]

    image_w = metadata["images"][list(annotations.keys())[0]]["width_px"]
    image_h = metadata["images"][list(annotations.keys())[0]]["height_px"]

    gsd_h = (metadata["camera_height"] * camera_entry["sensor_height"]) / \
            (camera_entry["focal_length"] * image_h)

    gsd_w = (metadata["camera_height"] * camera_entry["sensor_width"]) / \
            (camera_entry["focal_length"] * image_w)

    gsd = min(gsd_h, gsd_w)

    
    object_entries = []
    voronoi_entries = []
    
    for image_name in predictions.keys():
        predicted_boxes = predictions[image_name]["boxes"]
        predicted_scores = predictions[image_name]["scores"]
        
        pred_mask = predicted_scores > 0.50
        sel_predicted_boxes = predicted_boxes[pred_mask]

        if (sel_predicted_boxes.size > 0):
            predicted_box_areas = (sel_predicted_boxes[:, 2] - sel_predicted_boxes[:, 0]) * (sel_predicted_boxes[:, 3] - sel_predicted_boxes[:, 1])
            predicted_box_areas_m2 = np.round(predicted_box_areas * (gsd ** 2), 8)
        else:
            predicted_box_areas_m2 = []

        d_object = {
            image_name: sorted(predicted_box_areas_m2)
        }



        if sel_predicted_boxes.shape[0] <= 3:
            d_voronoi = {
                image_name: []
            }
        else:
            try:
                predicted_centres = (sel_predicted_boxes[..., :2] + sel_predicted_boxes[..., 2:]) / 2.0
                xy_predicted_centres = np.stack([predicted_centres[:, 1], predicted_centres[:, 0]], axis=-1)

                vor = Voronoi(xy_predicted_centres)

                lines = [
                    shapely.geometry.LineString(vor.vertices[line])
                    for line in vor.ridge_vertices
                    if -1 not in line
                ]
                
                boundary = shapely.geometry.Polygon([(0, 0), (image_w, 0), (image_w, image_h), (0, image_h)])
                filtered_lines = []
                for line in lines:
                    if boundary.contains(line):
                        filtered_lines.append(line)

                areas_m2 = []
                for poly in shapely.ops.polygonize(filtered_lines):
                    area_px = poly.area
                    area_m2 = round(area_px * (gsd ** 2), 8)
                    areas_m2.append(area_m2)

                d_voronoi = {
                    image_name: sorted(areas_m2)
                }
            except Exception as e:
                logger.info("Voronoi area calculation generated exception: {}".format(e))
                d_voronoi = {
                    image_name: []
                }


        object_entries.append(pd.DataFrame(d_object))
        voronoi_entries.append(pd.DataFrame(d_voronoi))

    if len(object_entries) > 0:
        object_images_df = pd.concat(object_entries, axis=1)
    else:
        object_images_df = pd.DataFrame()
    object_images_df = object_images_df.fillna('')

    if len(voronoi_entries) > 0:
        voronoi_images_df = pd.concat(voronoi_entries, axis=1)
    else:
        voronoi_images_df = pd.DataFrame()
    voronoi_images_df = voronoi_images_df.fillna('')


    object_entries = []
    voronoi_entries = []
    for image_name in predictions.keys():

        predicted_boxes = predictions[image_name]["boxes"]
        predicted_scores = predictions[image_name]["scores"]

        pred_mask = predicted_scores > 0.50
        sel_predicted_boxes = predicted_boxes[pred_mask]

        for region_type in ["regions_of_interest", "fine_tuning_regions", "test_regions"]:
            if region_type == "regions_of_interest":
                region_label = "interest"
            elif region_type == "fine_tuning_regions":
                region_label = "fine_tuning"
            else:
                region_label = "test"

            regions = annotations[image_name][region_type]

            for i, region in enumerate(regions):

                entry_name = image_name + ":" + region_label + "_" + str(i+1)

                predicted_centres = (sel_predicted_boxes[..., :2] + sel_predicted_boxes[..., 2:]) / 2.0

                predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])

                region_predicted_boxes = sel_predicted_boxes[predicted_inds]


                if region_predicted_boxes.size > 0:
                    region_predicted_box_areas_px = (region_predicted_boxes[:, 2] - region_predicted_boxes[:, 0]) * (region_predicted_boxes[:, 3] - region_predicted_boxes[:, 1])
                    region_predicted_box_areas_m2 = np.round(region_predicted_box_areas_px * (gsd ** 2), 8)
                else:
                    region_predicted_box_areas_m2 = []
                d_object = {
                    entry_name: sorted(region_predicted_box_areas_m2)
                }


                if region_predicted_boxes.shape[0] <= 3:
                    d_voronoi = {
                        entry_name: []
                    }
                else:

                    try:

                        region_predicted_centres = (region_predicted_boxes[..., :2] + region_predicted_boxes[..., 2:]) / 2.0
                        xy_region_predicted_centres = np.stack([region_predicted_centres[:, 1], region_predicted_centres[:, 0]], axis=-1)

                        vor = Voronoi(xy_region_predicted_centres)

                        lines = [
                            shapely.geometry.LineString(vor.vertices[line])
                            for line in vor.ridge_vertices
                            if -1 not in line
                        ]

                        boundary = shapely.geometry.Polygon([(x[1], x[0]) for x in region])

                        filtered_lines = []
                        for line in lines:
                            if boundary.contains(line):
                                filtered_lines.append(line)

                        areas_m2 = []
                        for poly in shapely.ops.polygonize(filtered_lines):
                            area_px = poly.area
                            area_m2 = round(area_px * (gsd ** 2), 8)
                            areas_m2.append(area_m2)

                        d_voronoi = {
                            entry_name: sorted(areas_m2)
                        }
                    except Exception as e:
                        logger.info("Voronoi area calculation generated exception: {}".format(e))
                        d_voronoi = {
                            entry_name: []
                        }


                object_entries.append(pd.DataFrame(d_object))
                voronoi_entries.append(pd.DataFrame(d_voronoi))



    if len(object_entries) > 0:
        object_regions_df = pd.concat(object_entries, axis=1)
    else:
        object_regions_df = pd.DataFrame()
    object_regions_df = object_regions_df.fillna('')


    if len(voronoi_entries) > 0:
        voronoi_regions_df = pd.concat(voronoi_entries, axis=1)
    else:
        voronoi_regions_df = pd.DataFrame()
    voronoi_regions_df = voronoi_regions_df.fillna('')




    out_path = os.path.join(result_dir, "areas.xlsx")

    if regions_only:
        sheet_name_to_df = {
            "Region Object Areas": object_regions_df,
            "Region Voronoi Areas": voronoi_regions_df
        }
    else:
        sheet_name_to_df = {
            "Image Object Areas": object_images_df,
            "Region Object Areas": object_regions_df,
            "Image Voronoi Areas": voronoi_images_df,
            "Region Voronoi Areas": voronoi_regions_df
        }
    writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    fmt = writer.book.add_format({"font_name": "Courier New"})


    for sheet_name in sheet_name_to_df.keys():

        df = sheet_name_to_df[sheet_name]

        df.to_excel(writer, index=False, sheet_name=sheet_name, na_rep='NA')  # send df to writer
        worksheet = writer.sheets[sheet_name]  # pull worksheet object

        worksheet.set_column('A:ZZ', None, fmt)
        worksheet.set_row(0, None, fmt)

        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            if series.size > 0:
                max_entry_size = series.astype(str).map(len).max()
            else:
                max_entry_size = 0
            max_len = max((
                max_entry_size,  # len of largest item
                len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width

    writer.close()

    end_time = time.time()
    elapsed = str(datetime.timedelta(seconds=round(end_time - start_time)))

    logger.info("Calculated Voronoi areas. Time elapsed: {}.".format(elapsed))


    

def create_regions_sheet(args, updated_metrics):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    predictions = args["predictions"]
    annotations = args["annotations"]
    metadata = args["metadata"]
    camera_specs = args["camera_specs"]
    vegetation_record = args["vegetation_record"]
    tags = args["tags"]

    num_classes = len(metadata["object_classes"])

    include_density = can_calculate_density(metadata, camera_specs)


    columns = [
        "Username",
        "Farm Name",
        "Field Name",
        "Mission Date",
        "Image Name",
        "Region Name"]
    
    for tag_name in tags.keys():
        columns.append(tag_name)

    columns.extend(["Source Of Annotations (For Image)"])


    if num_classes > 1:
        columns.extend([
            "Annotated Count (All Classes)",
            "Predicted Count (All Classes)"
        ])

    for object_class in metadata["object_classes"]:
        columns.append("Annotated Count (" + object_class + ")")
        columns.append("Predicted Count (" + object_class + ")")


    if include_density:
        if num_classes > 1:
            columns.extend([
                "Annotated Count Per Square Metre (All Classes)",
                "Predicted Count Per Square Metre (All Classes)"
            ])

        for object_class in metadata["object_classes"]:
            columns.append("Annotated Count Per Square Metre (" + object_class + ")")
            columns.append("Predicted Count Per Square Metre (" + object_class + ")")



    columns.append("Area (Pixels)")
    if num_classes > 1:
        columns.append("Mean of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Mean of Predicted Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (All Classes)")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Mean of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Mean of Predicted Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")")
        columns.append("Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")")

    if include_density:
        columns.extend(["Area (Square Metres)"])
        if num_classes > 1:
            columns.append("Mean of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Mean of Predicted Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (All Classes)")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (All Classes)")


        for object_class in metadata["object_classes"]:
            columns.append("Mean of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Mean of Predicted Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")")
            columns.append("Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")")
            
    if num_classes > 1:
        columns.append("Percent Count Error (All Classes)")

    for object_class in metadata["object_classes"]:
        columns.append("Percent Count Error (" + object_class + ")")

    if vegetation_record is not None:
        columns.extend([
            "Excess Green Threshold",
            "Vegetation Percentage"
        ])
        if num_classes > 1:
            columns.extend([
                "Percentage of Vegetation Inside Object Boundaries",
                "Percentage of Vegetation Outside Object Boundaries"           
            ])
        for object_class in metadata["object_classes"]:
            columns.extend([
                "Percentage of Vegetation Inside " + object_class + " Boundaries",
                "Percentage of Vegetation Outside " + object_class + " Boundaries"    
            ])





    metrics_lst = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)"  
    ]

    if num_classes > 1:
        for metric in metrics_lst:
            columns.append(metric + " (All Classes)")


    for object_class in metadata["object_classes"]:
        for metric in metrics_lst:
            columns.append(metric + " (" + object_class + ")")

    d = {}
    for c in columns:
        d[c] = []



    for image_name in predictions.keys():

        annotations_source = annotations[image_name]["source"]

        for region_type in ["regions_of_interest", "fine_tuning_regions", "test_regions"]:

            regions = annotations[image_name][region_type]

            for region_idx, region in enumerate(regions):
                if region_type == "regions_of_interest":
                    region_name = "interest_" + (str(region_idx+1))

                elif region_type == "fine_tuning_regions":
                    region_name = "fine_tuning_" + (str(region_idx+1))
                else:
                    region_name = "test_" + (str(region_idx+1))

                area_px = poly_utils.get_poly_area(region)


                d["Username"].append(username)
                d["Farm Name"].append(farm_name)
                d["Field Name"].append(field_name)
                d["Mission Date"].append(mission_date)
                d["Image Name"].append(image_name)
                d["Region Name"].append(region_name)


                nav_item = image_name + "/" + str(region_idx)
                for tag_name in tags.keys():
                    if region_type == "regions_of_interest" and nav_item in tags[tag_name]:
                        d[tag_name].append(tags[tag_name][nav_item])
                    else:
                        d[tag_name].append("NA")


                d["Source Of Annotations (For Image)"].append(annotations_source)


                d["Area (Pixels)"].append(area_px)
                if include_density:
                    gsd = get_gsd(camera_specs, metadata)
                    area_m2 = calculate_area_m2(gsd, area_px)
                    d["Area (Square Metres)"].append(round(area_m2, 8))


                if num_classes > 1:
                    cls_tups = [(-1, "All Classes")]
                    for i, object_class in enumerate(metadata["object_classes"]):
                        cls_tups.append((i, object_class))
                else:
                    cls_tups = [(0, metadata["object_classes"][0])]

                for cls_tup in cls_tups:
                    cls_idx = cls_tup[0]
                    object_class = cls_tup[1]

                    annotated_boxes = []
                    predicted_boxes = []
                    for i in range(len(annotations[image_name]["boxes"])):
                        if cls_idx == -1 or annotations[image_name]["classes"][i] == cls_idx:
                            annotated_boxes.append(annotations[image_name]["boxes"][i])

                    for i in range(len(predictions[image_name]["boxes"])):
                        if cls_idx == -1 or predictions[image_name]["classes"][i] == cls_idx:
                            if predictions[image_name]["scores"][i] > 0.50:
                                predicted_boxes.append(predictions[image_name]["boxes"][i])

                    annotated_boxes = np.array(annotated_boxes)
                    predicted_boxes = np.array(predicted_boxes)


                    annotated_centres = (annotated_boxes[..., :2] + annotated_boxes[..., 2:]) / 2.0
                    predicted_centres = (predicted_boxes[..., :2] + predicted_boxes[..., 2:]) / 2.0

                    annotated_inds = poly_utils.get_contained_inds_for_points(annotated_centres, [region])
                    predicted_inds = poly_utils.get_contained_inds_for_points(predicted_centres, [region])

                    annotated_boxes = annotated_boxes[annotated_inds]
                    predicted_boxes = predicted_boxes[predicted_inds]


                    annotated_count = annotated_boxes.shape[0]
                    predicted_count = predicted_boxes.shape[0]

                    d["Annotated Count (" + object_class + ")"].append(annotated_count)
                    d["Predicted Count (" + object_class + ")"].append(predicted_count)


                    if annotated_count > 0:
                        annotated_box_areas_px = box_utils.box_areas_np(annotated_boxes)
                        mean_annotated_object_area_px = round(np.mean(annotated_box_areas_px), 8)
                        stdev_annotated_object_area_px = round(np.std(annotated_box_areas_px), 8)
                    else:
                        mean_annotated_object_area_px = "NA"
                        stdev_annotated_object_area_px = "NA"

                    if predicted_count > 0:
                        predicted_box_areas_px = box_utils.box_areas_np(predicted_boxes)
                        mean_predicted_object_area_px = round(np.mean(predicted_box_areas_px), 8)
                        stdev_predicted_object_area_px = round(np.std(predicted_box_areas_px), 8)
                    else:
                        mean_predicted_object_area_px = "NA"
                        stdev_predicted_object_area_px = "NA"

                    d["Mean of Annotated Object Areas (Pixels) (" + object_class + ")"].append(mean_annotated_object_area_px)
                    d["Mean of Predicted Object Areas (Pixels) (" + object_class + ")"].append(mean_predicted_object_area_px)
                    d["Std. Dev. of Annotated Object Areas (Pixels) (" + object_class + ")"].append(stdev_annotated_object_area_px)
                    d["Std. Dev. of Predicted Object Areas (Pixels) (" + object_class + ")"].append(stdev_predicted_object_area_px)


                    if include_density:
                        gsd = get_gsd(camera_specs, metadata)
                        area_m2 = calculate_area_m2(gsd, area_px)
                        if area_m2 > 0:
                            annotated_count_per_square_metre = round(annotated_count / area_m2, 8)
                            predicted_count_per_square_metre = round(predicted_count / area_m2, 8)
                        else:
                            annotated_count_per_square_metre = "NA"
                            predicted_count_per_square_metre = "NA"
                        d["Annotated Count Per Square Metre (" + object_class + ")"].append(annotated_count_per_square_metre)
                        d["Predicted Count Per Square Metre (" + object_class + ")"].append(predicted_count_per_square_metre)

                        if annotated_count > 0:
                            annotated_box_areas_m2 = calculate_area_m2(gsd, annotated_box_areas_px)
                            mean_annotated_object_area_m2 = round(np.mean(annotated_box_areas_m2), 8)
                            stdev_annotated_object_area_m2 = round(np.std(annotated_box_areas_m2), 8)
                        else:
                            mean_annotated_object_area_m2 = "NA"
                            stdev_annotated_object_area_m2 = "NA"

                        if predicted_count > 0:
                            predicted_box_areas_m2 = calculate_area_m2(gsd, predicted_box_areas_px)
                            mean_predicted_object_area_m2 = round(np.mean(predicted_box_areas_m2), 8)
                            stdev_predicted_object_area_m2 = round(np.std(predicted_box_areas_m2), 8)               
                        else:
                            mean_predicted_object_area_m2 = "NA"
                            stdev_predicted_object_area_m2 = "NA"

                        d["Mean of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(mean_annotated_object_area_m2)
                        d["Mean of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(mean_predicted_object_area_m2)
                        d["Std. Dev. of Annotated Object Areas (Square Metres) (" + object_class + ")"].append(stdev_annotated_object_area_m2)
                        d["Std. Dev. of Predicted Object Areas (Square Metres) (" + object_class + ")"].append(stdev_predicted_object_area_m2)



                    if annotated_count > 0:
                        percent_count_error = round(abs((predicted_count - annotated_count) / (annotated_count)) * 100, 2)
                    else:
                        percent_count_error = "NA"

                    d["Percent Count Error (" + object_class + ")"].append(percent_count_error)

                    for metric in metrics_lst:
                        metric_val = updated_metrics[object_class][metric][image_name][region_type][region_idx]

                        if isinstance(metric_val, float):
                            metric_val = round(metric_val, 2)
                        d[metric + " (" + object_class + ")"].append(metric_val)



                if vegetation_record is not None:

                    d["Excess Green Threshold"].append(vegetation_record[image_name]["sel_val"])
                    vegetation_percentage = vegetation_record[image_name]["vegetation_percentage"][region_type][region_idx]
                    d["Vegetation Percentage"].append(vegetation_percentage)

                    if num_classes > 1:
                        obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"]["All Classes"][region_type][region_idx]
                        if vegetation_percentage == 0:
                            obj_percentage = "NA"
                            non_obj_percentage = "NA"
                        else:
                            obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                            non_obj_percentage = round(100 - obj_percentage, 2)

                        d["Percentage of Vegetation Inside Object Boundaries"].append(obj_percentage)
                        d["Percentage of Vegetation Outside Object Boundaries"].append(non_obj_percentage) 

                    for object_class in metadata["object_classes"]:

                        obj_vegetation_percentage = vegetation_record[image_name]["obj_vegetation_percentage"][object_class][region_type][region_idx]
                        if vegetation_percentage == 0:
                            obj_percentage = "NA"
                            non_obj_percentage = "NA"
                        else:
                            obj_percentage = round((obj_vegetation_percentage / vegetation_percentage) * 100, 2)
                            non_obj_percentage = round(100 - obj_percentage, 2)

                        d["Percentage of Vegetation Inside " + object_class + " Boundaries"].append(obj_percentage)
                        d["Percentage of Vegetation Outside " + object_class + " Boundaries"].append(non_obj_percentage) 


    df = pd.DataFrame(data=d, columns=columns)
    df.sort_values(by="Image Name", inplace=True, key=lambda x: np.argsort(index_natsorted(df["Image Name"])))
    return df



def create_stats_sheet(args, regions_df):
    username = args["username"]
    farm_name = args["farm_name"]
    field_name = args["field_name"]
    mission_date = args["mission_date"]
    metadata = args["metadata"]
    columns = [
        "Username",
        "Farm Name", 
        "Field Name", 
        "Mission Date", 
        "Region Type", 
    ]
 
    averaged_metrics = [
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)",
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)",
    ]

    num_classes = len(metadata["object_classes"])

    object_classes = []
    if num_classes > 1:
        object_classes.append("All Classes")
        columns.extend([
            "Mean Absolute Difference In Count (All Classes)",
            "Mean Squared Difference In Count (All Classes)"
        ])
        for metric in averaged_metrics:
            columns.append(metric + " (All Classes)")

    for object_class in metadata["object_classes"]:
        object_classes.append(object_class)
        columns.extend([
            "Mean Absolute Difference In Count (" + object_class + ")",
            "Mean Squared Difference In Count (" + object_class + ")"
        ])
        for metric in averaged_metrics:
            columns.append(metric + " (" + object_class + ")")

    d = {}
    for c in columns:
        d[c] = []

    if len(regions_df.index) > 0:

        for region_type in ["regions_of_interest", "fine_tuning_regions", "test_regions"]:
            if region_type == "regions_of_interest":
                disp_region_type = "interest"
            elif region_type == "fine_tuning_regions":
                disp_region_type = "fine_tuning"
            else:
                disp_region_type = "test"

            sub_df = regions_df[regions_df["Region Name"].str.contains(disp_region_type)]

            if len(sub_df) > 0:

                d["Username"].append(username)
                d["Farm Name"].append(farm_name)
                d["Field Name"].append(field_name)
                d["Mission Date"].append(mission_date)
                d["Region Type"].append(disp_region_type)


                for object_class in object_classes:
                    d["Mean Absolute Difference In Count (" + object_class + ")"].append(
                        round(float(np.mean(abs(sub_df["Annotated Count (" + object_class + ")"] - sub_df["Predicted Count (" + object_class + ")"]))), 2)
                    )

                    d["Mean Squared Difference In Count (" + object_class + ")"].append(
                        round(float(np.mean((sub_df["Annotated Count (" + object_class + ")"] - sub_df["Predicted Count (" + object_class + ")"]) ** 2)), 2)
                    )
                    
                    for metric in averaged_metrics:

                        try:
                            metric_val = round(float(np.mean(sub_df[metric + " (" + object_class + ")"])), 2)
                        except Exception:
                            metric_val = "unable_to_calculate"
                        
                        d[metric + " (" + object_class + ")"].append(metric_val)

    print(d)

    df = pd.DataFrame(data=d, columns=columns)
    return df






def get_pred_and_true_for_mAP(pred_abs_boxes, pred_classes, pred_scores,
                              true_abs_boxes, true_classes):

    if pred_abs_boxes.size > 0:
        pred_abs_boxes = box_utils.swap_xy_np(pred_abs_boxes)
    else:
        pred_abs_boxes = np.reshape(pred_abs_boxes, (0, 4))
        
    pred_classes = np.expand_dims(pred_classes, axis=-1)
    pred_scores = np.expand_dims(pred_scores, axis=-1)
    pred = np.hstack([pred_abs_boxes, pred_classes, pred_scores])

    if true_abs_boxes.size > 0:
        true_abs_boxes = box_utils.swap_xy_np(true_abs_boxes)
    else:
        true_abs_boxes = np.reshape(true_abs_boxes, (0, 4)) 

    true_classes = np.expand_dims(true_classes, axis=-1)
    difficult = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    crowd = np.expand_dims(np.zeros(true_classes.size), axis=-1)
    true = np.hstack([true_abs_boxes, true_classes, difficult, crowd])  

    return pred, true
