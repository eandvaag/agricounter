import os
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from io_utils import json_io
from models.common import annotation_utils, inference_metrics
from exp_runner import get_mapping_for_test_set, eval_in_domain_test_sets, eval_test_sets, eval_GWHD_test_sets, eval_fixed_patch_num_baselines, my_plot_colors


def bar_count_chart(baseline_name, test_sets, out_dirname):
    mappings = {}

    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
    # print(mappings)

    # percent_count_errors = []
    # object_counts = []
    anno_counts = []
    pred_counts = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

        test_set_image_set_dir = os.path.join("usr", "data",
                                            test_set["username"], "image_sets",
                                            test_set["farm_name"],
                                            test_set["field_name"],
                                            test_set["mission_date"])
        
        model_name = baseline_name #+ "_rep_" + str(0) #rep_num)
        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])

        predictions_path = os.path.join(result_dir, "predictions.json")
        predictions = annotation_utils.load_predictions(predictions_path)
        annotations_path = os.path.join(result_dir, "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                # image_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, [image_name], iou_thresh=0.5)


                anno_count = annotations[image_name]["boxes"].shape[0]
                pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]


                anno_counts.append(anno_count)
                pred_counts.append(pred_count)

    anno_counts = np.array(anno_counts)
    pred_counts = np.array(pred_counts)

    abs_diffs = np.abs(anno_counts - pred_counts)
    inds = np.argsort(abs_diffs)

    anno_counts = anno_counts[inds]
    pred_counts = pred_counts[inds]

    abs_diffs = abs_diffs[inds]
    print(abs_diffs)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))


    # axs.set_title("Difference In Count Per $m^2$", fontsize=18) # Per Square Metre") #Annotated Object Count vs. Image-Based Accuracy")

    # X_axis = np.arange(len(anno_counts)) 
    # width = 0.35
    # axs.bar(X_axis, anno_counts, width, label='Annotated', color=my_plot_colors[0]) #(77/255, 195/255, 255/255), ) 
    # axs.bar(X_axis + width, pred_counts, width, label='Predicted', color=my_plot_colors[1]) #color=(255/255, 15/255, 155/255), ) 
    

    for i, (anno_count, pred_count) in enumerate(zip(anno_counts, pred_counts)):
        if pred_count > anno_count:
            c = my_plot_colors[0]
        else:
            c = my_plot_colors[1]
        print(anno_count - pred_count)
        if anno_count != pred_count:
            axs.plot([i, i], [anno_count, pred_count], color=c)
        else:
            axs.scatter([i], [anno_count], color="grey", marker="_", s=2)


    axs.set_xlim((0, len(anno_counts)))

    # axs.bar(X_axis, anno_counts, width, label='Annotated', color=my_plot_colors[0])
    # plt.xticks(X_axis, X) 
    # plt.xlabel("Groups") 
    # plt.ylabel("Number of Students") 
    # plt.title("Number of Students in each group") 
    # plt.legend() 
    # plt.show() 

    # axs.bar(np.arange(len(dids)), dids, color=colors) #my_plot_colors[0])
    # axs.set_xlabel("Annotated Object Count")
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.tick_params(axis='both', which='minor', labelsize=8)
    axs.set_ylabel("Plant Count", size=16)
    
    
    custom_lines = [Line2D([0], [0], color=my_plot_colors[0], lw=2),
                Line2D([0], [0], color=my_plot_colors[1], lw=2)]

    axs.legend(custom_lines, ["Overcounted", "Undercounted"], prop={'size': 14}, loc="upper left")
    # axs.set_ylabel("Difference In Count Per $m^2$ (Predicted - Annotated)", fontsize=14) #Percent Count Error") #Image-Based Accuracy")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    plt.margins(x=0.01, tight=True)
    plt.tight_layout()
    
    os.makedirs(out_dirname, exist_ok=True)
    out_path = os.path.join(out_dirname, "count_differences.png") #"count_comparison_bar_chart.svg")
    plt.savefig(out_path, dpi=600)
    


def density_diff_bar_chart(baseline_name, test_sets, out_dirname):
    mappings = {}

    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
    # print(mappings)

    # percent_count_errors = []
    # object_counts = []
    dids = []
    percent_count_errors = []
    anno_counts_for_percent_count_errors = []
    anno_counts_per_square_metre = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

        test_set_image_set_dir = os.path.join("usr", "data",
                                            test_set["username"], "image_sets",
                                            test_set["farm_name"],
                                            test_set["field_name"],
                                            test_set["mission_date"])

        metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)
        camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
        camera_specs = json_io.load_json(camera_specs_path)
    
        
        model_name = baseline_name #+ "_rep_" + str(0) #rep_num)
        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])

        predictions_path = os.path.join(result_dir, "predictions.json")
        predictions = annotation_utils.load_predictions(predictions_path)
        annotations_path = os.path.join(result_dir, "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                # image_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, [image_name], iou_thresh=0.5)


                anno_count = annotations[image_name]["boxes"].shape[0]
                pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                # abs_dic = abs(anno_count - pred_count)
                # abs_dics.append(abs_dic)

                height_px = metadata["images"][image_name]["height_px"]
                width_px = metadata["images"][image_name]["width_px"]
                area_px = height_px * width_px

                gsd = inference_metrics.get_gsd(camera_specs, metadata)
                area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                annotated_count_per_square_metre = anno_count / area_m2
                predicted_count_per_square_metre = pred_count / area_m2

                did = predicted_count_per_square_metre - annotated_count_per_square_metre
                dids.append(did)

                anno_counts_per_square_metre.append(annotated_count_per_square_metre)

                if anno_count > 0:
                    percent_count_error = (abs(anno_count - pred_count) / anno_count) * 100

                    percent_count_errors.append(percent_count_error)
                    anno_counts_for_percent_count_errors.append(anno_count)


    percent_count_errors = np.array(percent_count_errors)
    anno_counts_for_percent_count_errors = np.array(anno_counts_for_percent_count_errors)

    print("ave annotated number of plants per square metre: {}".format(np.mean(anno_counts_per_square_metre)))

    print("mean percent_count_error", np.mean(percent_count_errors))
    print("mean percent_count_error (anno > 5)", np.mean(percent_count_errors[anno_counts_for_percent_count_errors > 5]))
    print("mean percent_count_error (anno > 10)", np.mean(percent_count_errors[anno_counts_for_percent_count_errors > 10]))

    dids.sort()

    dids = np.array(dids)

    frac_below_1 = np.sum(np.abs(dids) < 1) / dids.size
    frac_below_5 = np.sum(np.abs(dids) < 5) / dids.size
    frac_below_10 = np.sum(np.abs(dids) < 10) / dids.size

    print("frac below 1: {}".format(frac_below_1))
    print("frac below 5: {}".format(frac_below_5))
    print("frac below 10: {}".format(frac_below_10))

    
    fig, axs = plt.subplots(1, 1, figsize=(14, 6))

    # axs.set_title("Difference In Count Per $m^2$", fontsize=18) # Per Square Metre") #Annotated Object Count vs. Image-Based Accuracy")

    colors = []
    for did in dids:
        if abs(did) < 5:
            colors.append(my_plot_colors[1])
        else:
            colors.append(my_plot_colors[0])

    axs.bar(np.arange(len(dids)), dids, color=colors) #my_plot_colors[0])
    # axs.set_xlabel("Annotated Object Count")
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.tick_params(axis='both', which='minor', labelsize=8)
    axs.set_ylabel("Difference In Count Per $m^2$ (Predicted - Annotated)", fontsize=14) #Percent Count Error") #Image-Based Accuracy")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    plt.tight_layout()
    
    os.makedirs(out_dirname, exist_ok=True)
    out_path = os.path.join(out_dirname, "differences_in_density.svg")
    plt.savefig(out_path)
    






def object_count_vs_image_based_accuracy(baseline_name, test_sets, out_dirname):
    mappings = {}

    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
    # print(mappings)

    image_accuracies = []
    object_counts = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

        test_set_image_set_dir = os.path.join("usr", "data",
                                            test_set["username"], "image_sets",
                                            test_set["farm_name"],
                                            test_set["field_name"],
                                            test_set["mission_date"])
        
        
        model_name = baseline_name #+ "_rep_" + str(0) #rep_num)
        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])

        predictions_path = os.path.join(result_dir, "predictions.json")
        predictions = annotation_utils.load_predictions(predictions_path)
        annotations_path = os.path.join(result_dir, "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                # image_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, [image_name], iou_thresh=0.5)

                anno_count = annotations[image_name]["boxes"].shape[0]
                pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]


                sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]
                anno_boxes = annotations[image_name]["boxes"]

                if pred_count > 0:
                    if anno_count > 0:
                        true_positive, false_positive, false_negative = inference_metrics.get_positives_and_negatives(anno_boxes, sel_pred_boxes, 0.50)
                        # print("anno: {}, pred: {}, tp: {}, fp: {}, fn: {}".format(
                        #     num_predicted, num_annotated, true_positive, false_positive, false_negative))
                        # precision_050 = true_positive / (true_positive + false_positive)
                        # recall_050 = true_positive / (true_positive + false_negative)
                        # if precision_050 == 0 and recall_050 == 0:
                        #     f1_iou_050 = 0
                        # else:
                        #     f1_iou_050 = (2 * precision_050 * recall_050) / (precision_050 + recall_050)
                        acc_050 = true_positive / (true_positive + false_positive + false_negative)
                        # true_positive, false_positive, false_negative = get_positives_and_negatives(annotated_boxes, sel_region_pred_boxes, 0.75)
                        # precision = true_positive / (true_positive + false_positive)
                        # recall = true_positive / (true_positive + false_negative)
                        # f1_iou_075 = (2 * precision * recall) / (precision + recall)                        

                        
                    
                    else:
                        true_positive = 0
                        false_positive = pred_count #num_predicted
                        false_negative = 0

                        # precision_050 = 0.0
                        # recall_050 = 0.0
                        # f1_iou_050 = 0.0
                        acc_050 = 0.0
                else:
                    if anno_count > 0:
                        true_positive = 0
                        false_positive = 0
                        false_negative = anno_count #num_annotated

                        # precision_050 = 0.0
                        # recall_050 = 0.0
                        # f1_iou_050 = 0.0
                        acc_050 = 0.0
                    else:
                        true_positive = 0
                        false_positive = 0
                        false_negative = 0

                        # precision_050 = 1.0
                        # recall_050 = 1.0
                        # f1_iou_050 = 1.0
                        acc_050 = 1.0


                image_accuracies.append(acc_050)
                object_counts.append(anno_count)

    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    axs.set_title("Annotated Object Count vs. Accuracy")

    axs.scatter(object_counts, image_accuracies, color=my_plot_colors[1])
    axs.set_xlabel("Annotated Object Count")
    axs.set_ylabel("Accuracy")

    plt.tight_layout()
    
    os.makedirs(out_dirname, exist_ok=True)
    out_path = os.path.join(out_dirname, "count_vs_accuracy.svg")
    plt.savefig(out_path)
    



def object_count_vs_percent_count_error(baseline_name, test_sets, out_dirname):
    mappings = {}

    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
    # print(mappings)

    percent_count_errors = []
    object_counts = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

        test_set_image_set_dir = os.path.join("usr", "data",
                                            test_set["username"], "image_sets",
                                            test_set["farm_name"],
                                            test_set["field_name"],
                                            test_set["mission_date"])
        
        
        model_name = baseline_name #+ "_rep_" + str(0) #rep_num)
        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])

        predictions_path = os.path.join(result_dir, "predictions.json")
        predictions = annotation_utils.load_predictions(predictions_path)
        annotations_path = os.path.join(result_dir, "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                # image_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, [image_name], iou_thresh=0.5)

                anno_count = annotations[image_name]["boxes"].shape[0]
                pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]


                sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]
                anno_boxes = annotations[image_name]["boxes"]

                if anno_count > 0:
                    percent_count_error = (abs(anno_count - pred_count) / anno_count) * 100


                percent_count_errors.append(percent_count_error)
                object_counts.append(anno_count)

    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    axs.set_title("Annotated Object Count vs. Image-Based Accuracy")

    axs.scatter(object_counts, percent_count_errors, color=my_plot_colors[1])
    axs.set_xlabel("Annotated Object Count")
    axs.set_ylabel("Percent Count Error") #Image-Based Accuracy")

    plt.tight_layout()
    
    os.makedirs(out_dirname, exist_ok=True)
    out_path = os.path.join(out_dirname, "count_vs_percent_count_error.svg")
    plt.savefig(out_path)
    



def id_ood_performance(id_test_sets, ood_test_sets, baseline_name, out_dirname, include_mAP_metrics=True):

    results = {}
    mappings = {}

    for test_set_type in [id_test_sets, ood_test_sets]:

        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
    print(mappings)

    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):
        if i == 0:
            type_label = "id"
        else:
            type_label = "ood"
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]

            test_set_image_set_dir = os.path.join("usr", "data",
                                                test_set["username"], "image_sets",
                                                test_set["farm_name"],
                                                test_set["field_name"],
                                                test_set["mission_date"])
            
            metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)
            camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
            camera_specs = json_io.load_json(camera_specs_path)


            # test_set_percent_count_errors = []


            reps_global_percent_count_error = []
            reps_mean_per_image_percent_count_error = []
            reps_median_per_image_percent_count_error = []
            reps_max_per_image_percent_count_error = []
            reps_mean_per_image_abs_dic = []
            reps_median_per_image_abs_dic = []
            reps_max_per_image_abs_dic = []
            reps_mean_per_image_abs_did = []
            reps_median_per_image_abs_did = []
            reps_max_per_image_abs_did = []
            reps_image_based_accuracy = []
            reps_instance_based_accuracy = []
            reps_mAP_05 = []
            reps_mAP_05_005_095 = []








            for rep_num in range(5):
                model_name = baseline_name + "_rep_" + str(rep_num)


                model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])


                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)
                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)
                assessment_images = []
                percent_count_errors = []
                abs_dics = []
                abs_dids = []
                tot_anno_count = 0
                tot_pred_count = 0
                for image_name in annotations.keys():
                    if len(annotations[image_name]["test_regions"]) > 0:
                        assessment_images.append(image_name)


                        anno_count = annotations[image_name]["boxes"].shape[0]
                        pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                        abs_dic = abs(anno_count - pred_count)
                        abs_dics.append(abs_dic)

                        height_px = metadata["images"][image_name]["height_px"]
                        width_px = metadata["images"][image_name]["width_px"]
                        area_px = height_px * width_px

                        gsd = inference_metrics.get_gsd(camera_specs, metadata)
                        area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                        annotated_count_per_square_metre = anno_count / area_m2
                        predicted_count_per_square_metre = pred_count / area_m2

                        abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                        abs_dids.append(abs_did)









                        if anno_count > 0:
                            percent_count_error = abs((anno_count - pred_count) / (anno_count)) * 100

                            percent_count_errors.append(percent_count_error)

                        tot_anno_count += anno_count
                        tot_pred_count += pred_count

                # rep_test_set_percent_count_error = np.mean(percent_count_errors)
                # test_set_percent_count_errors.append(percent_count_errors) #rep_test_set_percent_count_error)

                test_set_mean_per_image_percent_count_error = float(np.mean(percent_count_errors))
                test_set_median_per_image_percent_count_error = float(np.median(percent_count_errors))
                test_set_max_per_image_percent_count_error = float(np.max(percent_count_errors))

                test_set_mean_per_image_abs_dic = float(np.mean(abs_dics))
                test_set_median_per_image_abs_dic = float(np.median(abs_dics))
                test_set_max_per_image_abs_dic = float(np.max(abs_dics))

                test_set_mean_per_image_abs_did = float(np.mean(abs_dids))
                test_set_median_per_image_abs_did = float(np.median(abs_dids))
                test_set_max_per_image_abs_did = float(np.max(abs_dids))

                test_set_instance_based_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images, iou_thresh=0.5)

                excel_path = os.path.join(result_dir, "metrics.xlsx")
                df = pd.read_excel(excel_path, sheet_name=0)
                test_set_image_based_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                if include_mAP_metrics:
                    full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                    full_predictions = annotation_utils.load_predictions(full_predictions_path)
                    test_set_mAP_05 = float(inference_metrics.get_mAP_val(annotations, full_predictions, ".50", assessment_images))
                    test_set_mAP_05_005_095 = float(inference_metrics.get_mAP_val(annotations, full_predictions, ".50:.05:.95", assessment_images))


                test_set_global_percent_count_error = abs((tot_anno_count - tot_pred_count) / (tot_anno_count)) * 100
                # test_set_


                # test_set_percent_count_error = np.mean(test_set_percent_count_errors)

                reps_global_percent_count_error.append(test_set_global_percent_count_error)
                reps_mean_per_image_percent_count_error.append(test_set_mean_per_image_percent_count_error)
                reps_median_per_image_percent_count_error.append(test_set_median_per_image_percent_count_error)
                reps_max_per_image_percent_count_error.append(test_set_max_per_image_percent_count_error)
                reps_mean_per_image_abs_dic.append(test_set_mean_per_image_abs_dic)
                reps_median_per_image_abs_dic.append(test_set_median_per_image_abs_dic)
                reps_max_per_image_abs_dic.append(test_set_max_per_image_abs_dic)
                reps_mean_per_image_abs_did.append(test_set_mean_per_image_abs_did)
                reps_median_per_image_abs_did.append(test_set_median_per_image_abs_did)
                reps_max_per_image_abs_did.append(test_set_max_per_image_abs_did)
                reps_image_based_accuracy.append(test_set_image_based_accuracy)
                reps_instance_based_accuracy.append(test_set_instance_based_accuracy)

                if include_mAP_metrics:
                    reps_mAP_05.append(test_set_mAP_05)
                    reps_mAP_05_005_095.append(test_set_mAP_05_005_095)



            results[test_set_str] = {
                "type_label": type_label,
                "global_percent_count_error": float(np.mean(reps_global_percent_count_error)), #[0]
                "mean_per_image_percent_count_error": float(np.mean(reps_mean_per_image_percent_count_error)),
                "median_per_image_percent_count_error": float(np.mean(reps_median_per_image_percent_count_error)),
                "max_per_image_percent_count_error": float(np.mean(reps_max_per_image_percent_count_error)),
                "mean_per_image_abs_dic": float(np.mean(reps_mean_per_image_abs_dic)),
                "median_per_image_abs_dic": float(np.mean(reps_median_per_image_abs_dic)),
                "max_per_image_abs_dic": float(np.mean(reps_max_per_image_abs_dic)),
                "mean_per_image_abs_did": float(np.mean(reps_mean_per_image_abs_did)),
                "median_per_image_abs_did": float(np.mean(reps_median_per_image_abs_did)),
                "max_per_image_abs_did": float(np.mean(reps_max_per_image_abs_did)),

                "image_based_accuracy": float(np.mean(reps_image_based_accuracy)),
                "instance_based_accuracy": float(np.mean(reps_instance_based_accuracy)),

                "std_global_percent_count_error": float(np.std(reps_global_percent_count_error)), #[0]
                "std_mean_per_image_percent_count_error": float(np.std(reps_mean_per_image_percent_count_error)),
                "std_median_per_image_percent_count_error": float(np.std(reps_median_per_image_percent_count_error)),
                "std_max_per_image_percent_count_error": float(np.std(reps_max_per_image_percent_count_error)),
                "std_mean_per_image_abs_dic": float(np.std(reps_mean_per_image_abs_dic)),
                "std_median_per_image_abs_dic": float(np.std(reps_median_per_image_abs_dic)),
                "std_max_per_image_abs_dic": float(np.std(reps_max_per_image_abs_dic)),
                "std_mean_per_image_abs_did": float(np.std(reps_mean_per_image_abs_did)),
                "std_median_per_image_abs_did": float(np.std(reps_median_per_image_abs_did)),
                "std_max_per_image_abs_did": float(np.std(reps_max_per_image_abs_did)),

                "std_image_based_accuracy": float(np.std(reps_image_based_accuracy)),
                "std_instance_based_accuracy": float(np.std(reps_instance_based_accuracy)),

            }
            if include_mAP_metrics:
                results[test_set_str]["mAP_.50"] = float(np.mean(reps_mAP_05))
                results[test_set_str]["mAP_.50:.05:.95"] = float(np.mean(reps_mAP_05_005_095))

                results[test_set_str]["std_mAP_.50"] = float(np.std(reps_mAP_05))
                results[test_set_str]["std_mAP_.50:.05:.95"] = float(np.std(reps_mAP_05_005_095))



            # results[test_set_str] = {
            #     "type_label": type_label,
            #     "global_percent_count_error": test_set_global_percent_count_error, #[0]
            #     "mean_per_image_percent_count_error": test_set_mean_per_image_percent_count_error,
            #     "median_per_image_percent_count_error": test_set_median_per_image_percent_count_error,
            #     "max_per_image_percent_count_error": test_set_max_per_image_percent_count_error,
            #     "mean_per_image_abs_dic": test_set_mean_per_image_abs_dic,
            #     "median_per_image_abs_dic": test_set_median_per_image_abs_dic,
            #     "max_per_image_abs_dic": test_set_max_per_image_abs_dic,
            #     "mean_per_image_abs_did": test_set_mean_per_image_abs_did,
            #     "median_per_image_abs_did": test_set_median_per_image_abs_did,
            #     "max_per_image_abs_did": test_set_max_per_image_abs_did,

            #     "image_based_accuracy": test_set_image_based_accuracy,
            #     "instance_based_accuracy": test_set_instance_based_accuracy,
            # }
            # if include_mAP_metrics:
            #     results[test_set_str]["mAP_.50"] = test_set_mAP_05
            #     results[test_set_str]["mAP_.50:.05:.95"] = test_set_mAP_05_005_095
                
    json_io.print_json(results)

    os.makedirs(out_dirname, exist_ok=True)
    results_path = os.path.join(out_dirname, "results.json")
    json_io.save_json(results_path, results)



def plot_ood_results(out_dirname):
    results_path = os.path.join(out_dirname, "results.json")
    results = json_io.load_json(results_path)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # id_results = []
    ood_percent_count_errors = []
    ood_instance_based_accuracies = []
    ood_mAP_vals = []

    for test_set_str in results.keys():
        # if results[test_set_str]["type_label"] == "id":
        #     id_results.append(results[test_set_str]["percent_count_error"])
        # else:
        if results[test_set_str]["type_label"] == "ood":
            ood_mAP_vals.append(results[test_set_str]["mAP_.50"])
            ood_instance_based_accuracies.append(results[test_set_str]["instance_based_accuracy"])
            ood_percent_count_errors.append(results[test_set_str]["percent_count_error"])




    # ood_results.sort()

    axs[0].scatter(np.arange(len(ood_mAP_vals)), ood_mAP_vals, color=my_plot_colors[0])
    axs[1].scatter(np.arange(len(ood_instance_based_accuracies)), ood_instance_based_accuracies, color=my_plot_colors[0])
    axs[2].scatter(np.arange(len(ood_percent_count_errors)), ood_percent_count_errors, color=my_plot_colors[0])

    axs[0].set_title("Out-of-Domain mAP: .50")
    axs[1].set_title("Out-of-Domain Instance-Based Accuracy")
    axs[2].set_title("Out-of-Domain Percent Count Error")  


    plot_path = os.path.join(out_dirname, "plot.svg")
    plt.savefig(plot_path)


def plot_results(out_dirname):

    results_path = os.path.join(out_dirname, "results.json")
    results = json_io.load_json(results_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    id_results = []
    ood_results = []

    for test_set_str in results.keys():
        if results[test_set_str]["type_label"] == "id":
            id_results.append(results[test_set_str]["percent_count_error"])
        else:
            ood_results.append(results[test_set_str]["percent_count_error"])


    id_results.sort()
    ood_results.sort()

    # for i, id_result in enumerate(id_results):
    #     axs[0].scatter([i] * len(id_result), id_result, color=my_plot_colors[0])

    # for i, ood_result in enumerate(ood_results):
    #     axs[1].scatter([i] * len(ood_result), ood_result, color=my_plot_colors[0])

    axs[0].scatter(np.arange(len(id_results)), id_results, color=my_plot_colors[0])
    axs[1].scatter(np.arange(len(ood_results)), ood_results, color=my_plot_colors[1])

    # axs[0].set_ylim(0, 1)
    # axs[1].set_ylim(0, 1)

    axs[0].set_title("In-Domain Test Sets")
    axs[1].set_title("Out-of-Domain Test Sets")


    plot_path = os.path.join(out_dirname, "plot.svg")
    plt.savefig(plot_path)
    

def create_ood_tables(out_dirname):

    results_path = os.path.join(out_dirname, "results.json")
    results = json_io.load_json(results_path)

    include_mAP_metrics = True
    for test_set_str in results.keys():
        if "mAP_.50" not in results[test_set_str] or "mAP_.50:.05:.95" not in results[test_set_str]:
            include_mAP_metrics = False


    if include_mAP_metrics:
        print("\\begin{tabular}{p{0.11\linewidth} p{0.16\linewidth} p{0.11\linewidth} p{0.09\linewidth} p{0.09\linewidth} p{0.14\linewidth} p{0.14\linewidth}}")
        print("\hline")
        print("\\textbf{Farm Name} &")
        print("\\textbf{Field Name} &")
        print("\\textbf{Mission Date} &")
        print("\\textbf{AP\\newline (IOU=.50)} &")
        print("\\textbf{AP\\newline (IOU=\\newline .50:.05:.95)} &")
        print("\\textbf{Image-Based\\newline Accuracy\\newline (IOU=.50, conf$>$.50)} &")
        print("\\textbf{Instance-Based\\newline Accuracy\\newline (IOU=.50, conf$>$.50)} \\\\")
        print("\hline")

        test_set_strs = natsorted(list(results.keys()))

        for test_set_str in test_set_strs:

            pieces = test_set_str.split(" ")
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]

            print(farm_name + " & " + field_name + " & " + mission_date + " & " + 
                "{:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) \\\\".format(
                results[test_set_str]["mAP_.50"],
                results[test_set_str]["std_mAP_.50"],
                results[test_set_str]["mAP_.50:.05:.95"],
                results[test_set_str]["std_mAP_.50:.05:.95"],
                results[test_set_str]["image_based_accuracy"],
                results[test_set_str]["std_image_based_accuracy"],
                results[test_set_str]["instance_based_accuracy"],
                results[test_set_str]["std_instance_based_accuracy"],
                )
            )

        print("\hline")

        print("\\textbf{Average}" + " & & & " + 
            "\\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) \\\\".format(
            np.mean([results[x]["mAP_.50"] for x in results.keys()]),
            np.mean([results[x]["std_mAP_.50"] for x in results.keys()]),
            np.mean([results[x]["mAP_.50:.05:.95"] for x in results.keys()]),
            np.mean([results[x]["std_mAP_.50:.05:.95"] for x in results.keys()]),
            np.mean([results[x]["image_based_accuracy"] for x in results.keys()]),
            np.mean([results[x]["std_image_based_accuracy"] for x in results.keys()]),
            np.mean([results[x]["instance_based_accuracy"] for x in results.keys()]),
            np.mean([results[x]["std_instance_based_accuracy"] for x in results.keys()]),            
            )
        )

        print("\hline")

    else:
        print("\\begin{tabular}{p{0.11\linewidth} p{0.16\linewidth} p{0.11\linewidth} p{0.14\linewidth} p{0.14\linewidth}}")
        print("\hline")
        print("\\textbf{Farm Name} &")
        print("\\textbf{Field Name} &")
        print("\\textbf{Mission Date} &")
        print("\\textbf{Image-Based\\newline Accuracy\\newline (IOU=.50, conf$>$.50)} &")
        print("\\textbf{Instance-Based\\newline Accuracy\\newline (IOU=.50, conf$>$.50)} \\\\")
        print("\hline")

        test_set_strs = natsorted(list(results.keys()))

        for test_set_str in test_set_strs:

            pieces = test_set_str.split(" ")
            farm_name = pieces[1]
            field_name = pieces[2]
            mission_date = pieces[3]

            print(farm_name + " & " + field_name + " & " + mission_date + " & " + 
                "{:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) \\\\".format(
                results[test_set_str]["image_based_accuracy"],
                results[test_set_str]["std_image_based_accuracy"],
                results[test_set_str]["instance_based_accuracy"],
                results[test_set_str]["std_instance_based_accuracy"]
                )
            )
            
        print("\hline")

        print("\\textbf{Average}" + " & & & " + 
            "\\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}})\\\\".format(
            np.mean([results[x]["image_based_accuracy"] for x in results.keys()]),
            np.mean([results[x]["std_image_based_accuracy"] for x in results.keys()]),
            np.mean([results[x]["instance_based_accuracy"] for x in results.keys()]),
            np.mean([results[x]["std_instance_based_accuracy"] for x in results.keys()]),
            )
        )

        print("\hline")

    

    # print("\\begin{tabular}{p{0.11\linewidth} p{0.16\linewidth} p{0.11\linewidth} p{0.09\linewidth} p{0.09\linewidth} p{0.14\linewidth} p{0.14\linewidth}}")
    # print("\hline")
    # print("\\textbf{Farm Name} &")
    # print("\\textbf{Field Name} &")
    # print("\\textbf{Mission Date} &")
    # print("\\textbf{Mean Per-Image\\newline Percent Count\\newline Error} &")
    # print("\\textbf{Median Per-Image\\newline Percent Count\\newline Error} &")
    # print("\\textbf{Max Per-Image\\newline Percent Count\\newline Error} &")
    # print("\\textbf{Percent Count\\newline Error (Global\\newline Image Set\\newline Count)} \\\\")
    # print("\hline")


    # for test_set_str in test_set_strs:

    #     pieces = test_set_str.split(" ")
    #     farm_name = pieces[1]
    #     field_name = pieces[2]
    #     mission_date = pieces[3]

    #     print(farm_name + " & " + field_name + " & " + mission_date + " & " + 
    #           "{:0.2f} & {:0.2f} & {:0.2f} & {:0.2f} \\\\".format(
    #           results[test_set_str]["mean_per_image_percent_count_error"],
    #           results[test_set_str]["median_per_image_percent_count_error"],
    #           results[test_set_str]["max_per_image_percent_count_error"],
    #           results[test_set_str]["global_percent_count_error"]
    #           )
    #     )

    # print("\hline")

    # print("\\textbf{Average}" + " & & & " + 
    #       "\\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} \\\\".format(
    #     np.mean([results[x]["mean_per_image_percent_count_error"] for x in results.keys()]),
    #     np.mean([results[x]["median_per_image_percent_count_error"] for x in results.keys()]),
    #     np.mean([results[x]["max_per_image_percent_count_error"] for x in results.keys()]),
    #     np.mean([results[x]["global_percent_count_error"] for x in results.keys()])
    #       )
    # )

    # print("\hline")








    # print("\\begin{tabular}{p{0.11\linewidth} p{0.16\linewidth} p{0.11\linewidth} p{0.09\linewidth} p{0.14\linewidth} p{0.14\linewidth}}")
    # print("\hline")
    # print("\\textbf{Farm Name} &")
    # print("\\textbf{Field Name} &")
    # print("\\textbf{Mission Date} &")
    # print("\\textbf{Mean Per-Image\\newline Absolute Difference\\newline In Count} &")
    # print("\\textbf{Median Per-Image\\newline Absolute Difference\\newline In Count} &")
    # print("\\textbf{Max Per-Image\\newline Absolute Difference\\newline In Count} \\\\")
    # print("\hline")


    # for test_set_str in test_set_strs:

    #     pieces = test_set_str.split(" ")
    #     farm_name = pieces[1]
    #     field_name = pieces[2]
    #     mission_date = pieces[3]

    #     print(farm_name + " & " + field_name + " & " + mission_date + " & " + 
    #           "{:0.2f} & {:0.2f} & {:0.0f} \\\\".format(
    #           results[test_set_str]["mean_per_image_abs_dic"],
    #           results[test_set_str]["median_per_image_abs_dic"],
    #           results[test_set_str]["max_per_image_abs_dic"]
    #           )
    #     )

    # print("\hline")

    # print("\\textbf{Average}" + " & & & " + 
    #       "\\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} \\\\".format(
    #     np.mean([results[x]["mean_per_image_abs_dic"] for x in results.keys()]),
    #     np.mean([results[x]["median_per_image_abs_dic"] for x in results.keys()]),
    #     np.mean([results[x]["max_per_image_abs_dic"] for x in results.keys()]),
    #     # np.mean([results[x]["global_percent_count_error"] for x in results.keys()])
    #       )
    # )

    # print("\hline")




    # print("\\begin{tabular}{p{0.11\linewidth} p{0.16\linewidth} p{0.11\linewidth} p{0.09\linewidth} p{0.14\linewidth} p{0.14\linewidth}}")
    # print("\hline")
    # print("\\textbf{Farm Name} &")
    # print("\\textbf{Field Name} &")
    # print("\\textbf{Mission Date} &")
    # print("\\textbf{Mean Per-Image\\newline Absolute Difference\\newline In Density} &")
    # print("\\textbf{Median Per-Image\\newline Absolute Difference\\newline In Density} &")
    # print("\\textbf{Max Per-Image\\newline Absolute Difference\\newline In Density} \\\\")
    # print("\hline")


    # for test_set_str in test_set_strs:

    #     pieces = test_set_str.split(" ")
    #     farm_name = pieces[1]
    #     field_name = pieces[2]
    #     mission_date = pieces[3]

    #     print(farm_name + " & " + field_name + " & " + mission_date + " & " + 
    #           "{:0.2f} & {:0.2f} & {:0.2f} \\\\".format(
    #           results[test_set_str]["mean_per_image_abs_did"],
    #           results[test_set_str]["median_per_image_abs_did"],
    #           results[test_set_str]["max_per_image_abs_did"]
    #           )
    #     )

    # print("\hline")

    # print("\\textbf{Average}" + " & & & " + 
    #       "\\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} & \\textbf{{{:0.2f}}} \\\\".format(
    #     np.mean([results[x]["mean_per_image_abs_did"] for x in results.keys()]),
    #     np.mean([results[x]["median_per_image_abs_did"] for x in results.keys()]),
    #     np.mean([results[x]["max_per_image_abs_did"] for x in results.keys()]),
    #     # np.mean([results[x]["global_percent_count_error"] for x in results.keys()])
    #       )
    # )

    # print("\hline")




    print()
    print()
    print()
    print()


    print("\\begin{tabular}{p{0.11\linewidth} p{0.16\linewidth} p{0.11\linewidth} p{0.07\linewidth} p{0.07\linewidth} p{0.07\linewidth} p{0.07\linewidth} p{0.07\linewidth} p{0.07\linewidth}}")
    print("\hline")
    print("\\textbf{Farm Name} &")
    print("\\textbf{Field Name} &")
    print("\\textbf{Mission Date} &")
    print("\\multicolumn{3}{c}{\\textbf{Absolute Difference In Density}} &")
    print("\\multicolumn{3}{c}{\\textbf{Percent Count Error}} \\\\")
    print("\\multicolumn{3}{c}{} &")
    print("\\textbf{Mean} &")
    print("\\textbf{Median} &")
    print("\\textbf{Max} &")
    print("\\textbf{Mean} &")
    print("\\textbf{Median} &")
    print("\\textbf{Max} \\\\")



    # print("\\textbf{Median Per-Image\\newline Absolute Difference\\newline In Count} &")
    # print("\\textbf{Max Per-Image\\newline Absolute Difference\\newline In Count} \\\\")
    print("\hline")

    for test_set_str in test_set_strs:

        pieces = test_set_str.split(" ")
        farm_name = pieces[1]
        field_name = pieces[2]
        mission_date = pieces[3]

        print(farm_name + " & " + field_name + " & " + mission_date + " & " + 
              "{:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) & {:0.2f} ({:0.2f}) \\\\".format(
              results[test_set_str]["mean_per_image_abs_did"],
              results[test_set_str]["std_mean_per_image_abs_did"],
              results[test_set_str]["median_per_image_abs_did"],
              results[test_set_str]["std_median_per_image_abs_did"],
              results[test_set_str]["max_per_image_abs_did"],
              results[test_set_str]["std_max_per_image_abs_did"],
              results[test_set_str]["mean_per_image_percent_count_error"],
              results[test_set_str]["std_mean_per_image_percent_count_error"],
              results[test_set_str]["median_per_image_percent_count_error"],
              results[test_set_str]["std_median_per_image_percent_count_error"],
              results[test_set_str]["max_per_image_percent_count_error"],
              results[test_set_str]["std_max_per_image_percent_count_error"],


              )
        )
    print("\hline")

    print("\\textbf{Average}" + " & & & " + 
          "\\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}}) & \\textbf{{{:0.2f}}} (\\textbf{{{:0.2f}}})\\\\".format(
            np.mean([results[x]["mean_per_image_abs_did"] for x in results.keys()]),
            np.mean([results[x]["std_mean_per_image_abs_did"] for x in results.keys()]),
            np.mean([results[x]["median_per_image_abs_did"] for x in results.keys()]),
            np.mean([results[x]["std_median_per_image_abs_did"] for x in results.keys()]),
            np.mean([results[x]["max_per_image_abs_did"] for x in results.keys()]),
            np.mean([results[x]["std_max_per_image_abs_did"] for x in results.keys()]),
            np.mean([results[x]["mean_per_image_percent_count_error"] for x in results.keys()]),
            np.mean([results[x]["std_mean_per_image_percent_count_error"] for x in results.keys()]),
            np.mean([results[x]["median_per_image_percent_count_error"] for x in results.keys()]),
            np.mean([results[x]["std_median_per_image_percent_count_error"] for x in results.keys()]),
            np.mean([results[x]["max_per_image_percent_count_error"] for x in results.keys()]),
            np.mean([results[x]["std_max_per_image_percent_count_error"] for x in results.keys()]),


          )
    )



if __name__ == "__main__":

    baseline_name = "set_of_27_38891_patches_rep_0"
    out_dirname = os.path.join("eval_charts", "best_model_assessment")
    # id_ood_performance([], eval_test_sets, baseline_name, out_dirname)

    # # plot_ood_results(out_dirname)
    # create_ood_tables(out_dirname)


    # object_count_vs_percent_count_error(baseline_name, eval_test_sets, out_dirname)
    # object_count_vs_image_based_accuracy(baseline_name, eval_test_sets, out_dirname)
    # density_diff_bar_chart(baseline_name, eval_test_sets, out_dirname)
    bar_count_chart(baseline_name, eval_test_sets, out_dirname)

    # baseline_name = "GWHD_official_train_fixed_epoch_num"
    # out_dirname = os.path.join("eval_charts", "GWHD_model_assessment")
    # id_ood_performance([], eval_GWHD_test_sets, baseline_name, out_dirname, include_mAP_metrics=False)
    # create_ood_tables(out_dirname)


