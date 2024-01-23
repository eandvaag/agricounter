import os
import glob
import random
import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from io_utils import json_io
from models.common import driver_utils, box_utils, annotation_utils, inference_metrics
# import fine_tune_eval

from exp_runner import eval_test_sets, get_mapping_for_test_set, my_plot_colors


# def annotation_vs_dilated_annotation_results(test_sets, dilation_levels, out_dir):
#     results = {}
#     for dilation_level in dilation_levels:
#         print("dilation_level: {}".format(dilation_level))


#         rep_accuracies_01 = []
#         rep_accuracies_05 = []


#         for rep_num in range(5):
#             accuracies_01 = []
#             accuracies_05 = []

#             for test_set in test_sets:

#                 test_set_image_set_dir = os.path.join("usr", "data",
#                                                         test_set["username"], "image_sets",
#                                                         test_set["farm_name"],
#                                                         test_set["field_name"],
#                                                         test_set["mission_date"])
                

#                 annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
#                 annotations = annotation_utils.load_annotations(annotations_path)

#                 metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
#                 metadata = json_io.load_json(metadata_path)
                
#                 dilated_annotations = {}
#                 assessment_images = []
#                 for image_name in annotations.keys():
#                     if len(annotations[image_name]["test_regions"]) > 0:

#                         image_h = metadata["images"][image_name]["height_px"]
#                         image_w = metadata["images"][image_name]["width_px"]

#                         # for box in annotations[image_name]["boxes"]:

#                         dilated_annotations[image_name] = {}
#                         dilated_annotations[image_name]["boxes"] = []

#                         for box in annotations[image_name]["boxes"]:
#                             min_y_dilation = abs(round(random.uniform(0, dilation_level)))
#                             min_x_dilation = abs(round(random.uniform(0, dilation_level)))
#                             max_y_dilation = abs(round(random.uniform(0, dilation_level)))
#                             max_x_dilation = abs(round(random.uniform(0, dilation_level)))

#                             # min_y_dilation = abs(round(random.gauss(0, dilation_sigma)))
#                             # min_x_dilation = abs(round(random.gauss(0, dilation_sigma)))
#                             # max_y_dilation = abs(round(random.gauss(0, dilation_sigma)))
#                             # max_x_dilation = abs(round(random.gauss(0, dilation_sigma)))

#                             new_box = [
#                                 max(0, box[0] - min_y_dilation), # min(image_h - 1, max(0, p_min_y)) #box[0] + random.randint(-(1) * perturbation_amount, perturbation_amount)))
#                                 max(0, box[1] - min_x_dilation), #min(image_w - 1, max(0, p_min_x)) #box[1] + random.randint(-(1) * perturbation_amount, perturbation_amount)))
#                                 min(image_h, box[2] + max_y_dilation), #max(box[0] + 1, min(image_h, p_max_y)) #box[2] + random.randint(-(1) * perturbation_amount, perturbation_amount)))
#                                 min(image_w, box[3] + max_x_dilation)
#                             ]


#                             dilated_annotations[image_name]["boxes"].append(new_box)







#                         # boxes = np.copy(annotations[image_name]["boxes"])
#                         # if np.size(boxes) > 0:
#                         #     boxes[:, 0] = np.maximum(0, boxes[:, 0] - round(dilation_level / 2))
#                         #     boxes[:, 1] = np.maximum(0, boxes[:, 1] - round(dilation_level / 2))
#                         #     boxes[:, 2] = np.minimum(image_h, boxes[:, 2] + round(dilation_level / 2))
#                         #     boxes[:, 3] = np.minimum(image_w, boxes[:, 3] + round(dilation_level / 2))


#                         assessment_images.append(image_name)

#                 global_test_set_accuracy_01 = fine_tune_eval.get_global_accuracy_2(annotations, dilated_annotations, assessment_images, iou_thresh=0.1)
#                 print("acc", global_test_set_accuracy_01)
#                 global_test_set_accuracy_05 = fine_tune_eval.get_global_accuracy_2(annotations, dilated_annotations, assessment_images, iou_thresh=0.5)


#             accuracies_01.append(global_test_set_accuracy_01)
#             accuracies_05.append(global_test_set_accuracy_05)

#         rep_accuracy_01 = float(np.mean(accuracies_01))
#         rep_accuracy_05 = float(np.mean(accuracies_05))

#         rep_accuracies_01.append(rep_accuracy_01)
#         rep_accuracies_05.append(rep_accuracy_05)


#         results[dilation_level] = {
#             "accuracy_01": rep_accuracies_01, #float(np.mean(rep_accuracies_01)),
#             "accuracy_05": rep_accuracies_05 #float(np.mean(rep_accuracies_05))
#         }

                        
#         # mapping = get_mapping_for_test_set(test_set_image_set_dir)
#         # result_dir = os.path.join(test_set_image_set_dir, "model", "results", mapping[result_name])


#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     final_results_path = os.path.join(out_dir, "anno_vs_dilated_anno_results.json")
#     json_io.save_json(final_results_path, results)



def get_noise_results(result_name, result_dir, metadata, camera_specs):


    # results = {
    #     "trim": [],
    #     "no_trim": []
    # }
    # results = []

    # for trim_app_k in results.keys():

    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    predictions_path = os.path.join(result_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path)
    iou_threshs = [0.1, 0.5] #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("result_name", result_name)
    # patch_coords_0 = patch_predictions[list(patch_predictions.keys())[0]]["patch_coords"][0]
    # patch_size = patch_coords_0[2] - patch_coords_0[0]
    # print("patch_size", patch_size)
    # patch_overlap_percent = int(result_name.split("_")[2])
    # print("patch_overlap_percent", patch_overlap_percent)
    # overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))
    # print("overlap_px", overlap_px)

    # for iou_thresh in iou_threshs:
    #     print("iou_thresh: {}".format(iou_thresh))

    # predictions = {}

    # for image_name in patch_predictions.keys():

    #     predictions[image_name] = {
    #         "boxes": [],
    #         "scores": []
    #     }


    #     for i in range(len(patch_predictions[image_name]["patch_coords"])):

    #         pred_boxes = np.array(patch_predictions[image_name]["patch_boxes"][i])
    #         pred_classes = np.full(len(patch_predictions[image_name]["patch_scores"][i]), 1)
    #         pred_scores = np.array(patch_predictions[image_name]["patch_scores"][i])
    #         patch_coords = np.array(patch_predictions[image_name]["patch_coords"][i])
    #         if pred_boxes.size > 0:
    #             pred_boxes, pred_classes, pred_scores = box_utils.non_max_suppression_with_classes(
    #                 pred_boxes,
    #                 pred_classes,
    #                 pred_scores,
    #                 iou_thresh=iou_thresh)

    #             region = [0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]

    #             pred_image_abs_boxes, pred_image_scores = \
    #                 driver_utils.get_image_detections_any_overlap(pred_boxes, 
    #                                     pred_scores,
    #                                     patch_coords, 
    #                                     region,
    #                                     overlap_px,
    #                                     trim=(trim_app_k == "trim"))


    #             predictions[image_name]["boxes"].extend(pred_image_abs_boxes.tolist())
    #             predictions[image_name]["scores"].extend(pred_image_scores.tolist())

    # driver_utils.apply_nms_to_image_boxes(predictions, 
    #                                 iou_thresh=iou_thresh)



    # metrics = inference_metrics.collect_image_set_metrics(predictions, annotations)

    assessment_images = []
    abs_dics = []
    abs_dids = []
    for image_name in annotations.keys():
        if len(annotations[image_name]["test_regions"]) > 0:
            assessment_images.append(image_name)

            anno_count = annotations[image_name]["boxes"].shape[0]
            pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]
            abs_dics.append(abs(anno_count - pred_count))


            height_px = metadata["images"][image_name]["height_px"]
            width_px = metadata["images"][image_name]["width_px"]
            area_px = height_px * width_px

            gsd = inference_metrics.get_gsd(camera_specs, metadata)
            area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

            annotated_count_per_square_metre = anno_count / area_m2
            predicted_count_per_square_metre = pred_count / area_m2

            abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
            abs_dids.append(abs_did)




    global_test_set_accuracy_01 = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images, iou_thresh=0.1)
    global_test_set_accuracy_05 = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images, iou_thresh=0.5)

    # if noise_type == "uniform_dilation":
    #     shrunk_predictions = []
    #     for image_name in predictions.keys():
        
    #         predictions[image_name]["boxes"] - noise_amount / 2
    # global_test_set_mAP_05 = fine_tune_eval.get_AP(annotations, predictions, ".50", assessment_images)



    results = {
        "accuracy_01": global_test_set_accuracy_01,
        "accuracy_05": global_test_set_accuracy_05,
        "abs_dics": abs_dics,
        "abs_dids": abs_dids
        # "mAP_05": global_test_set_mAP_05
    }

    
    return results



def get_all_noise_results(noise_type, noise_amounts, test_sets, out_dir):

    all_noise_results = {}

    for noise_amount in noise_amounts:
        # all_dilation_results[dilation_amount] = []

        rep_mean_abs_dics = []
        rep_mean_abs_dids = []
        rep_accuracies_01 = []
        rep_accuracies_05 = []
        rep_mAPs_05 = []

        for rep_num in range(5): #5):

            if noise_amount == 0:
                result_name = "set_of_27_16000_patches_rep_" + str(rep_num)
            else:
                if noise_type == "uniform_dilation":
                    result_name = "set_of_27_uniformly_dilated_by_" + str(noise_amount) + "_16000_patches_rep_" + str(rep_num)
                elif noise_type == "removal":
                    result_name = "set_of_27_remove_" + str(noise_amount) + "_16000_patches_rep_" + str(rep_num)


            abs_dics = []
            accuracies_01 = []
            accuracies_05 = []
            mAPs_05 = []
            test_set_mean_abs_dics = []
            test_set_mean_abs_dids = []
            for test_set in test_sets:

                test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                
                metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
                metadata = json_io.load_json(metadata_path)

                camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
                camera_specs = json_io.load_json(camera_specs_path)


                mapping = get_mapping_for_test_set(test_set_image_set_dir)
                result_dir = os.path.join(test_set_image_set_dir, "model", "results", mapping[result_name])

                # metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
                # metadata = json_io.load_json(metadata_path)



                dilation_results = get_noise_results(result_name, result_dir, metadata, camera_specs) #noise_type, noise_amount)

                # all_dilation_results[dilation_amount].append(dilation_results)
                abs_dics.extend(dilation_results["abs_dics"])
                accuracies_01.append(dilation_results["accuracy_01"])
                accuracies_05.append(dilation_results["accuracy_05"])
                # mAPs_05.append(dilation_results["mAP_05"])

                test_set_mean_abs_dics.append(np.mean(dilation_results["abs_dics"]))
                test_set_mean_abs_dids.append(np.mean(dilation_results["abs_dids"]))
            
            rep_mean_abs_dic = float(np.mean(abs_dics))
            rep_accuracy_01 = float(np.mean(accuracies_01))
            rep_accuracy_05 = float(np.mean(accuracies_05))
            # rep_mAP_05 = float(np.mean(mAPs_05))

            rep_mean_abs_dic = float(np.mean(test_set_mean_abs_dics))
            rep_mean_abs_did = float(np.mean(test_set_mean_abs_dids))

            rep_mean_abs_dics.append(rep_mean_abs_dic)
            rep_mean_abs_dids.append(rep_mean_abs_did)
            rep_accuracies_01.append(rep_accuracy_01)
            rep_accuracies_05.append(rep_accuracy_05)
            # rep_mAPs_05.append(rep_mAP_05)




        all_noise_results[noise_amount] = {
            # "mean_abs_dic": float(np.mean(rep_mean_abs_dics)),
            # "mean_abs_dic_std": float(np.std(rep_mean_abs_dics)),

            # "accuracy_01": float(np.mean(rep_accuracies_01)),
            # "accuracy_01_std": float(np.std(rep_accuracies_01)),

            # "accuracy_05": float(np.mean(rep_accuracies_05)),
            # "accuracy_05_std": float(np.std(rep_accuracies_05)),


            "abs_dic": rep_mean_abs_dics,
            "abs_did": rep_mean_abs_dids,
            "accuracy_01": rep_accuracies_01,
            "accuracy_05": rep_accuracies_05,
            # "mAP_05": rep_mAPs_05

        }


    # results = {
    #     # "trim": {},
    #     # "no_trim": {}
    # }

    # for dilation_amount in all_dilation_results.keys():
    #     # results[dilation_amount] = {}
    #     for i in range(len(all_dilation_results[dilation_amount])):

    #         # for method in all_trimming_results[i].keys():
    #         for r in all_dilation_results[dilation_amount][i]:
    #             iou_thresh = r["iou_thresh"]
    #             if iou_thresh not in results:
    #                 results[iou_thresh] = {}
    #             if dilation_amount not in results[iou_thresh]:
    #                 results[iou_thresh][dilation_amount] = []
    #             results[iou_thresh][dilation_amount].append(r["accuracy"])


    #             # if iou_thresh not in results[dilation_amount]:
    #             #     results[dilation_amount][iou_thresh] = []
    #             # results[dilation_amount][iou_thresh].append(r["accuracy"])

    #         # for r in all_trimming_results[i][method]:
                
    #         #     iou_thresh = r["iou_thresh"]
    #         #     if iou_thresh not in results[method]:
    #         #         results[method][iou_thresh] = []
    #         #     results[method][iou_thresh].append(r["accuracy"])


    # # final_results = {
    # #     "trim": [],
    # #     "no_trim": []
    # # }
    # final_results = {}
    # # for method in results.keys():
    # for iou_thresh in results.keys():
    #     final_results[iou_thresh] = []
    #     for dilation_amount in results[iou_thresh].keys():
    #         final_results[iou_thresh].append((dilation_amount, float(np.mean(results[iou_thresh][dilation_amount]))))


    # for dilation_amount in results.keys():
    #     final_results[dilation_amount] = []
    #     for iou_thresh in results[dilation_amount].keys():
    #         final_results[dilation_amount].append((iou_thresh, float(np.mean(results[dilation_amount][iou_thresh]))))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_results_path = os.path.join(out_dir, noise_type + "_results.json")
    json_io.save_json(final_results_path, all_noise_results)








def create_dilation_anno_and_pred_plot(out_dir):

    pred_results_path = os.path.join(out_dir, "uniform_dilation_results.json")
    pred_results = json_io.load_json(pred_results_path)

    anno_results_path = os.path.join(out_dir, "anno_vs_dilated_anno_results.json")
    anno_results = json_io.load_json(anno_results_path)

    fig, axs = plt.subplots(1, 1, figsize=(6, 5))


    axs.plot([float(x) for x in pred_results.keys()], 
            [np.mean(pred_results[x]["accuracy_05"]) for x in pred_results.keys()], 
            color=my_plot_colors[0], label="IOU Threshold: 0.5")
    
    axs.plot([float(x) for x in pred_results.keys()], 
            [np.mean(pred_results[x]["accuracy_01"]) for x in pred_results.keys()], 
            color=my_plot_colors[1], label="IOU Threshold: 0.1")

    # for x in pred_results.keys():
    #     axs[0].scatter([float(x)] * len(pred_results[x]["accuracy_05"]), 
    #                [z for z in pred_results[x]["accuracy_05"]], 
    #                c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')



    axs.plot([float(x) for x in anno_results.keys()], 
            [np.mean(anno_results[x]["accuracy_05"]) for x in anno_results.keys()], 
            color=my_plot_colors[0], linestyle="dashed") #, label="IOU Threshold: 0.5")

    axs.plot([float(x) for x in anno_results.keys()], 
            [np.mean(anno_results[x]["accuracy_01"]) for x in anno_results.keys()], 
            color=my_plot_colors[1], linestyle="dashed")

    # for x in anno_results.keys():
    #     axs[0].scatter([float(x)] * len(anno_results[x]["accuracy_05"]), 
    #                [z for z in anno_results[x]["accuracy_05"]], 
    #                c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

    axs.legend()
    # plt.suptitle(title, size=16)

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    out_path = os.path.join(out_dir, "dilation_anno_and_pred.png") #".svg")
    plt.savefig(out_path, dpi=800)
    plt.close()

def create_abs_did_plot(out_dir):

    removal_results_path = os.path.join(out_dir, "removal", "removal_results.json")
    removal_results = json_io.load_json(removal_results_path)
    
    uniform_dilation_results_path = os.path.join(out_dir, "uniform_dilation", "uniform_dilation_results.json")
    uniform_dilation_results = json_io.load_json(uniform_dilation_results_path)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

 
    # if noise_type == "uniform_dilation":
    #     xlabel = "Maximum Dilation Amount (Pixels)"
    #     title = "Effect of Loosely Drawn Annotations on Predicted Count"
    # elif noise_type == "removal":
    #     xlabel = "Removal Percentage"
    #     title = "Effect of Missing Annotations on Predicted Count"


    title = "Effect of Annotation Noise on Predicted Counts"

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))



    # axs.plot([float(x) for x in final_results.keys()], 
    #         [np.mean(final_results[x]["accuracy_05"]) for x in final_results.keys()], 
    #         color=my_plot_colors[0], label="IOU Threshold: 0.5") #, linestyle=(0, (5, 5)))

    # for x in final_results.keys():
    #     axs.scatter([float(x)] * len(final_results[x]["accuracy_05"]), 
    #                [z for z in final_results[x]["accuracy_05"]], 
    #                c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')



    # axs.plot([float(x) for x in final_results.keys()], 
    #         [np.mean(final_results[x]["accuracy_01"]) for x in final_results.keys()], 
    #         color=my_plot_colors[1], label="IOU Threshold: 0.1") #, linestyle=(0, (5, 5)))

    # for x in final_results.keys():
    #     axs.scatter([float(x)] * len(final_results[x]["accuracy_01"]), 
    #                [z for z in final_results[x]["accuracy_01"]], 
    #                c=my_plot_colors[1], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

    # if noise_type == "removal":
    #     axs.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    #     axs.set_xticklabels(["0", "25", "50", "75", "100"])

    # axs[0].set_ylim(bottom=0, top=1.0)

    # # axs[1].set_title("Effect of Missing Annotations on Accuracy")
    # axs[0].set_ylabel("Instance-Based Accuracy")
    # axs[0].set_xlabel(xlabel)
    # axs[0].legend(loc="upper right")









    axs[0].plot([float(x) for x in uniform_dilation_results.keys()], 
            [np.mean(uniform_dilation_results[x]["abs_did"]) for x in uniform_dilation_results.keys()], 
            color=my_plot_colors[0]) #, linestyle=(0, (5, 5)))

    for x in uniform_dilation_results.keys():
        axs[0].scatter([float(x)] * len(uniform_dilation_results[x]["abs_did"]), 
                   [z for z in uniform_dilation_results[x]["abs_did"]], 
                   c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

    axs[0].set_title("Dilated Annotations") 
    axs[0].set_ylim(bottom=0)
    # axs[0].set_title("Effect of Missing Annotations on Mean Absolute Difference in Count")
    axs[0].set_ylabel("Mean Absolute Difference in Density") # Count")
    axs[0].set_xlabel("Maximum Dilation Amount (Pixels)")



    # if noise_type == "removal":

    axs[1].plot([float(x) for x in removal_results.keys()], 
            [np.mean(removal_results[x]["abs_did"]) for x in removal_results.keys()], 
            color=my_plot_colors[0]) #, linestyle=(0, (5, 5)))

    for x in removal_results.keys():
        axs[1].scatter([float(x)] * len(removal_results[x]["abs_did"]), 
                   [z for z in removal_results[x]["abs_did"]], 
                   c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

    axs[1].set_title("Missing Annotations") 
    axs[1].set_ylim(bottom=0)
    # axs[0].set_title("Effect of Missing Annotations on Mean Absolute Difference in Count")
    axs[1].set_ylabel("Mean Absolute Difference in Density") # Count")
    axs[1].set_xlabel("Removal Percentage")



    axs[1].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    axs[1].set_xticklabels(["0", "25", "50", "75", "100"])


    # out_path = os.path.join(out_dir, noise_type + "_mean_abs_dic.svg")
    # plt.savefig(out_path)
    # plt.close()

    # json_io.print_json(final_results)

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))



    # plt.tight_layout()




    # axs[2].plot([float(x) for x in final_results.keys()], 
    #         [np.mean(final_results[x]["mAP_05"]) for x in final_results.keys()], 
    #         color=my_plot_colors[0]) #, linestyle=(0, (5, 5)))

    # for x in final_results.keys():
    #     axs[2].scatter([float(x)] * len(final_results[x]["mAP_05"]), 
    #                [z for z in final_results[x]["mAP_05"]], 
    #                c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')
        
    # axs[2].set_ylim(bottom=0, top=1)
    # # axs[0].set_title("Effect of Missing Annotations on Mean Absolute Difference in Count")
    # axs[2].set_ylabel("mAP")
    # axs[2].set_xlabel(xlabel)

    axs[0].set_ylim(top=30)
    axs[1].set_ylim(top=30)



    plt.suptitle(title, size=16)
    # plt.text(0.5, 0.5, "DDDDDD")

    props = dict(boxstyle='round', facecolor='white', edgecolor="white") 
    plt.gcf().text(0.5, 0.92, "* lower values are better",  fontsize=12, style="italic", 
                   linespacing=1.8, verticalalignment='top', horizontalalignment="center", bbox=props)


    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.84,
                    wspace=0.4,
                    hspace=0.4)
    out_path = os.path.join(out_dir, "mean_abs_did_only" + ".svg") #".png") #".svg")
    plt.savefig(out_path) #, dpi=800)
    plt.close()

def create_aggregate_dilation_plot(noise_type, out_dir):



    color_lookup = {
        0.1: "#ff9600", 
        0.2: "#c89d00", 
        0.3: "#939e00", 
        0.4: "#62982d", 
        0.5: "#2f8f49",
        0.6: "#00835e",
        0.7: "#00756b",
        0.8: "#00676e",
        0.9: "#075767",
        # 18: "#2a4858"
        
    }
    final_results_path = os.path.join(out_dir, noise_type + "_results.json")
    final_results = json_io.load_json(final_results_path)

    # ax.set_facecolor('slategray')


    # for result_name in result_names:
    # for iou_thresh in final_results.keys():
    #     # marker = marker_lookup[int(result_name.split("_")[2])]
    #     # percent_overlap = int(result_name.split("_")[2])
    #     color = color_lookup[float(iou_thresh)]

    #     # final_results = json_io.load_json(os.path.join(out_dir, result_name + ".json"))
    #     # if result_name != "trimming_eval_0_percent_overlap":
    #     #     ax.plot([x[0] for x in final_results["trim"]], [x[1] for x in final_results["trim"]], color=color, linestyle="solid") #, label="With Box Pruning")
    #     #     # ax.scatter([x[0] for x in final_results["trim"]], [x[1] for x in final_results["trim"]], color=color, marker="^")

    #     ax.plot([x[0] for x in final_results[iou_thresh]], [x[1] for x in final_results[iou_thresh]], color=color, linestyle=(0, (5, 5))) #, label="Without Box Pruning")

    #     # ax.scatter([x[0] for x in final_results["no_trim"]], [x[1] for x in final_results["no_trim"]], color=color, marker="v", label=percent_overlap)
    #plt.legend()

    # ax.plot([int(x) for x in final_results.keys()], [final_results[x]["accuracy_05"] for x in final_results.keys()], color="orange", linestyle=(0, (5, 5)))
    # ax.plot([int(x) for x in final_results.keys()], [final_results[x]["accuracy_01"] for x in final_results.keys()], color="pink", linestyle=(0, (5, 5)))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

 
    if noise_type == "uniform_dilation":
        xlabel = "Maximum Dilation Amount (Pixels)"
        title = "Effect of Loosely Drawn Annotations"
    elif noise_type == "removal":
        xlabel = "Removal Percentage"
        title = "Effect of Missing Annotations"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))



    axs[0].plot([float(x) for x in final_results.keys()], 
            [np.mean(final_results[x]["accuracy_05"]) for x in final_results.keys()], 
            color=my_plot_colors[0], label="IOU Threshold: 0.5") #, linestyle=(0, (5, 5)))

    for x in final_results.keys():
        axs[0].scatter([float(x)] * len(final_results[x]["accuracy_05"]), 
                   [z for z in final_results[x]["accuracy_05"]], 
                   c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')



    axs[0].plot([float(x) for x in final_results.keys()], 
            [np.mean(final_results[x]["accuracy_01"]) for x in final_results.keys()], 
            color=my_plot_colors[1], label="IOU Threshold: 0.1") #, linestyle=(0, (5, 5)))

    for x in final_results.keys():
        axs[0].scatter([float(x)] * len(final_results[x]["accuracy_01"]), 
                   [z for z in final_results[x]["accuracy_01"]], 
                   c=my_plot_colors[1], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

    if noise_type == "removal":
        axs[0].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        axs[0].set_xticklabels(["0", "25", "50", "75", "100"])

    axs[0].set_ylim(bottom=0, top=1.0)

    # axs[1].set_title("Effect of Missing Annotations on Accuracy")
    axs[0].set_ylabel("Instance-Based Accuracy")
    axs[0].set_xlabel(xlabel)
    axs[0].legend(loc="upper right")









    axs[1].plot([float(x) for x in final_results.keys()], 
            [np.mean(final_results[x]["abs_did"]) for x in final_results.keys()], 
            color=my_plot_colors[0]) #, linestyle=(0, (5, 5)))

    for x in final_results.keys():
        axs[1].scatter([float(x)] * len(final_results[x]["abs_did"]), 
                   [z for z in final_results[x]["abs_did"]], 
                   c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')
        
    axs[1].set_ylim(bottom=0)
    # axs[0].set_title("Effect of Missing Annotations on Mean Absolute Difference in Count")
    axs[1].set_ylabel("Mean Absolute Difference in Density") # Count")
    axs[1].set_xlabel(xlabel)

    if noise_type == "removal":
        axs[1].set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        axs[1].set_xticklabels(["0", "25", "50", "75", "100"])


    # out_path = os.path.join(out_dir, noise_type + "_mean_abs_dic.svg")
    # plt.savefig(out_path)
    # plt.close()

    # json_io.print_json(final_results)

    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))



    # plt.tight_layout()




    # axs[2].plot([float(x) for x in final_results.keys()], 
    #         [np.mean(final_results[x]["mAP_05"]) for x in final_results.keys()], 
    #         color=my_plot_colors[0]) #, linestyle=(0, (5, 5)))

    # for x in final_results.keys():
    #     axs[2].scatter([float(x)] * len(final_results[x]["mAP_05"]), 
    #                [z for z in final_results[x]["mAP_05"]], 
    #                c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')
        
    # axs[2].set_ylim(bottom=0, top=1)
    # # axs[0].set_title("Effect of Missing Annotations on Mean Absolute Difference in Count")
    # axs[2].set_ylabel("mAP")
    # axs[2].set_xlabel(xlabel)




    plt.suptitle(title, size=16)

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    out_path = os.path.join(out_dir, noise_type + "mean_mean_abs_did" + ".svg") #".png") #".svg")
    plt.savefig(out_path) #, dpi=800)
    plt.close()

    # ax.fill_between([int(x) for x in final_results.keys()], 
    #                 [final_results[x]["mean_abs_dic"] - final_results[x]["mean_abs_dic_std"] for x in final_results.keys()], 
    #                 [final_results[x]["mean_abs_dic"] + final_results[x]["mean_abs_dic_std"] for x in final_results.keys()], 
    #                 edgecolor=my_plot_colors[0], facecolor=my_plot_colors[0], alpha=0.15)

    # ax.scatter([x[0]] * len(x[3]), x[3], c=my_plot_colors[0], marker="o", zorder=2, alpha=0.7, s=70, edgecolors='none')

    # ax.plot([int(x) for x in final_results.keys()], [final_results[x]["accuracy_05"] for x in final_results.keys()], color=my_plot_colors[0]) #, linestyle=(0, (5, 5)))


    # ax.fill_between([int(x) for x in final_results.keys()], 
    #                 [final_results[x]["accuracy_05"] - final_results[x]["accuracy_05_std"] for x in final_results.keys()], 
    #                 [final_results[x]["accuracy_05"] + final_results[x]["accuracy_05_std"] for x in final_results.keys()], 
    #                 edgecolor=my_plot_colors[0], facecolor=my_plot_colors[0], alpha=0.15)


    # ax.plot([int(x) for x in final_results.keys()], [final_results[x]["accuracy_01"] for x in final_results.keys()], color=my_plot_colors[1]) #, linestyle=(0, (5, 5)))


    # ax.fill_between([int(x) for x in final_results.keys()], 
    #                 [final_results[x]["accuracy_01"] - final_results[x]["accuracy_01_std"] for x in final_results.keys()], 
    #                 [final_results[x]["accuracy_01"] + final_results[x]["accuracy_01_std"] for x in final_results.keys()], 
    #                 edgecolor=my_plot_colors[1], facecolor=my_plot_colors[1], alpha=0.15)



    # legend_elements = [
    #     Line2D([0], [0], linestyle='solid', lw=1, c="black", label='With Box Pruning'),
    #     Line2D([0], [0], linestyle=(0, (5, 5)), lw=1, c="black", label='Without Box Pruning')
    # ]
    # first_legend = ax.legend(handles=legend_elements, handlelength=4, loc=(0.01, 0.01)) #"lower center")
    # plt.gca().add_artist(first_legend)

    # legend_elements = []
    # for i in [0, 10, 20, 30, 40, 50]:
    #     legend_elements.append(
    #         Patch(facecolor=color_lookup[i], label=str(i) + "%")
    #     )


    # legend_elements = [
    #      Patch(facecolor='orange', edgecolor='r', label='Color Patch'),
    #     Line2D([0], [0], linestyle='dashed', lw=4, label='Without Box Pruning')
    # ]
    # ax.set_title("Effect of Box Pruning On Accuracy")
    # ax.set_ylabel("Instance-Based Accuracy")
    # ax.set_xlabel("NMS IOU Threshold")
    # # ax.set_ylim([0, 1.0])
    # ax.legend(handles=legend_elements, title="Patch Overlap", loc=(0.01, 0.1)) #"lower left")













if __name__ == "__main__":
    noise_type = "removal" #"uniform_dilation" #"uniform_dilation" #"removal"

    out_dir = os.path.join("eval_charts", "noise") #, noise_type)

    if noise_type == "uniform_dilation":
        noise_amounts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    else:
        noise_amounts = [0, 0.05, 0.10, 0.15, 0.25, 0.5, 0.75, 0.9]

    # get_all_noise_results(noise_type, noise_amounts, eval_test_sets, out_dir)
    # create_aggregate_dilation_plot(noise_type, out_dir)

    # annotation_vs_dilated_annotation_results(eval_test_sets, noise_amounts, out_dir)

    
    # create_dilation_anno_and_pred_plot(out_dir)

    # create_aggregate_dilation_plot(noise_type, out_dir)

    create_abs_did_plot(out_dir)

