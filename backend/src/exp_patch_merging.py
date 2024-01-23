import os
import glob
import math as m
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


from io_utils import json_io
from models.common import driver_utils, box_utils, annotation_utils, inference_metrics
from exp_runner import my_plot_colors, eval_test_sets, eval_fixed_patch_num_baselines


def get_mapping_for_test_set(test_set_image_set_dir):

    mapping = {}
    results_dir = os.path.join(test_set_image_set_dir, "model", "results")
    for result_dir in glob.glob(os.path.join(results_dir, "*")):
        request_path = os.path.join(result_dir, "request.json")
        request = json_io.load_json(request_path)
        if request["results_name"] in mapping:
            raise RuntimeError("Duplicate result name: {}, {}".format(test_set_image_set_dir, request["results_name"]))
        mapping[request["results_name"]] = request["request_uuid"]
    return mapping


def get_trimming_results(result_name, result_dir, metadata):


    results = {
        "trim": [],
        "no_trim": []
    }

    for trim_app_k in results.keys():

        annotations_path = os.path.join(result_dir, "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        patch_predictions_path = os.path.join(result_dir, "patch_predictions.json")
        patch_predictions = json_io.load_json(patch_predictions_path)
        iou_threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        print("result_name", result_name)
        patch_coords_0 = patch_predictions[list(patch_predictions.keys())[0]]["patch_coords"][0]
        patch_size = patch_coords_0[2] - patch_coords_0[0]
        print("patch_size", patch_size)
        patch_overlap_percent = int(result_name.split("_")[2])
        print("patch_overlap_percent", patch_overlap_percent)
        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))
        print("overlap_px", overlap_px)

        for iou_thresh in iou_threshs:
            print("iou_thresh: {}".format(iou_thresh))

            predictions = {}

            for image_name in patch_predictions.keys():

                predictions[image_name] = {
                    "boxes": [],
                    "scores": []
                }


                for i in range(len(patch_predictions[image_name]["patch_coords"])):

                    pred_boxes = np.array(patch_predictions[image_name]["patch_boxes"][i])
                    pred_classes = np.full(len(patch_predictions[image_name]["patch_scores"][i]), 1)
                    pred_scores = np.array(patch_predictions[image_name]["patch_scores"][i])
                    patch_coords = np.array(patch_predictions[image_name]["patch_coords"][i])
                    if pred_boxes.size > 0:
                        pred_boxes, pred_classes, pred_scores = box_utils.non_max_suppression_with_classes(
                            pred_boxes,
                            pred_classes,
                            pred_scores,
                            iou_thresh=iou_thresh)

                        region = [0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]

                        pred_image_abs_boxes, pred_image_scores = \
                            driver_utils.get_image_detections_any_overlap(pred_boxes, 
                                                pred_scores,
                                                patch_coords, 
                                                region,
                                                overlap_px,
                                                trim=(trim_app_k == "trim"))


                        predictions[image_name]["boxes"].extend(pred_image_abs_boxes.tolist())
                        predictions[image_name]["scores"].extend(pred_image_scores.tolist())

            driver_utils.apply_nms_to_image_boxes(predictions, 
                                            iou_thresh=iou_thresh)



            # metrics = inference_metrics.collect_image_set_metrics(predictions, annotations)

            assessment_images = []
            for image_name in annotations.keys():
                if len(annotations[image_name]["test_regions"]) > 0:
                    assessment_images.append(image_name)

            global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images)
            # global_test_set_accuracy = fine_tune_eval.get_AP(annotations, predictions, ".20", assessment_images)


            results[trim_app_k].append({
                "iou_thresh": iou_thresh,
                "accuracy": global_test_set_accuracy
            })

    
    return results


def get_all_trimming_results(result_name, test_sets, out_dir):

    all_trimming_results = []

    for test_set in test_sets:
        test_set_image_set_dir = os.path.join("usr", "data",
                                                test_set["username"], "image_sets",
                                                test_set["farm_name"],
                                                test_set["field_name"],
                                                test_set["mission_date"])
        mapping = get_mapping_for_test_set(test_set_image_set_dir)
        result_dir = os.path.join(test_set_image_set_dir, "model", "results", mapping[result_name])

        metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)



        trimming_results = get_trimming_results(result_name, result_dir, metadata)

        all_trimming_results.append(trimming_results)
    results = {
        "trim": {},
        "no_trim": {}
    }

    for i in range(len(all_trimming_results)):

        for method in all_trimming_results[i].keys():

            for r in all_trimming_results[i][method]:
                
                iou_thresh = r["iou_thresh"]
                if iou_thresh not in results[method]:
                    results[method][iou_thresh] = []
                results[method][iou_thresh].append(r["accuracy"])


    final_results = {
        "trim": [],
        "no_trim": []
    }
    for method in results.keys():
        for iou_thresh in results[method].keys():
            final_results[method].append((iou_thresh, float(np.mean(results[method][iou_thresh]))))


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_results_path = os.path.join(out_dir, result_name + ".json")
    json_io.save_json(final_results_path, final_results)


    # plt.plot([x[0] for x in final_results["trim"]], [x[1] for x in final_results["trim"]], label="With Box Pruning")
    # plt.plot([x[0] for x in final_results["no_trim"]], [x[1] for x in final_results["no_trim"]], label="Without Box Pruning")

    # plt.legend()

    # out_path = os.path.join(out_dir, result_name + ".svg")
    # plt.savefig(out_path)



def create_trimming_plot(result_name, out_dir):
    final_results = json_io.load_json(os.path.join(out_dir, result_name + ".json"))

    plt.figure()

    plt.plot([x[0] for x in final_results["trim"]], [x[1] for x in final_results["trim"]], alpha=0.5, label="With Box Pruning")
    plt.plot([x[0] for x in final_results["no_trim"]], [x[1] for x in final_results["no_trim"]], alpha=0.5, label="Without Box Pruning")

    plt.legend()


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, result_name + ".svg")
    plt.savefig(out_path)





def create_aggregate_trimming_plot(result_names, out_dir):

    fig, ax = plt.subplots(figsize=(8, 7))

    # marker_lookup = {
    #     0: "o",
    #     10: "x",
    #     20: "+",
    #     30: "|",
    #     40: "v",
    #     50: "^"
    # }
    # color_lookup = {
    #     0: "orangered", 
    #     10: "royalblue", 
    #     20: "forestgreen", 
    #     30: "orange", 
    #     40: "mediumorchid",
    #     50: "hotpink"
    # }
    # color_lookup = {
    #     0: "orange", 
    #     10: "orangered", 
    #     20: "hotpink", 
    #     30: "mediumorchid", 
    #     40: "royalblue",
    #     50: "limegreen"
    # }


    # color_lookup = {
    #     0: "#f12e2e", 
    #     10: "#f9005f", 
    #     20: "#f00090", 
    #     30: "#d331bf", 
    #     40: "#9c56e6",
    #     50: "#2270ff"
    # }
    # color_lookup = {
    #     0: "#f12e2e", 
    #     10: "#ed6900", 
    #     20: "#da9600", 
    #     30: "#babe00", 
    #     40: "#8ae02a",
    #     50: "#00ff7e"
    # }
    # color_lookup = {
    #     0: "#126bff", 
    #     10: "#9753e9", 
    #     20: "#cf2ec7", 
    #     30: "#f0009d", 
    #     40: "#ff006f",
    #     50: "#ff0042"
    # }
    # color_lookup = {
    #     0: "#ff0042", 
    #     10: "#ff006f", 
    #     20: "#f0009d", 
    #     30: "#cf2ec7", 
    #     40: "#9753e9",
    #     50: "#126bff"
    # }
    color_lookup = {
        0: "#ff9600", 
        10: "#9d9e00", 
        20: "#45933f", 
        30: "#007e64", 
        40: "#00646e",
        50: "#2a4858"
    }


    # ax.set_facecolor('slategray')


    for result_name in result_names:
        # marker = marker_lookup[int(result_name.split("_")[2])]
        percent_overlap = int(result_name.split("_")[2])
        color = color_lookup[percent_overlap]

        final_results = json_io.load_json(os.path.join(out_dir, result_name + ".json"))
        if result_name != "trimming_eval_0_percent_overlap":
            ax.plot([x[0] for x in final_results["trim"]], [x[1] for x in final_results["trim"]], color=color, linestyle="solid") #, label="With Box Pruning")
            # ax.scatter([x[0] for x in final_results["trim"]], [x[1] for x in final_results["trim"]], color=color, marker="^")

        ax.plot([x[0] for x in final_results["no_trim"]], [x[1] for x in final_results["no_trim"]], color=color, linestyle=(0, (5, 5))) #, label="Without Box Pruning")

        # ax.scatter([x[0] for x in final_results["no_trim"]], [x[1] for x in final_results["no_trim"]], color=color, marker="v", label=percent_overlap)
    #plt.legend()


    legend_elements = [
        Line2D([0], [0], linestyle='solid', lw=1, c="black", label='With Box Pruning'),
        Line2D([0], [0], linestyle=(0, (5, 5)), lw=1, c="black", label='Without Box Pruning')
    ]
    first_legend = ax.legend(handles=legend_elements, handlelength=4, loc=(0.01, 0.01)) #"lower center")
    plt.gca().add_artist(first_legend)

    legend_elements = []
    for i in [0, 10, 20, 30, 40, 50]:
        legend_elements.append(
            Patch(facecolor=color_lookup[i], label=str(i) + "%")
        )
    # legend_elements = [
    #      Patch(facecolor='orange', edgecolor='r', label='Color Patch'),
    #     Line2D([0], [0], linestyle='dashed', lw=4, label='Without Box Pruning')
    # ]
    ax.set_title("Effect of Box Pruning On Accuracy")
    ax.set_ylabel("Instance-Based Accuracy")
    ax.set_xlabel("NMS IOU Threshold")
    # ax.set_ylim([0, 1.0])
    ax.legend(handles=legend_elements, title="Patch Overlap", loc=(0.01, 0.1)) #"lower left")
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, "aggregate_plot_grad.svg") #png")
    plt.savefig(out_path) #, dpi=600)





def create_patch_merging_plot(test_sets, baseline, merging_prefixes):


    mappings = {}
    # results = {
    #     "overall": {
    #         "single": [],
    #         "diverse": [],
    #         "single_improved": [],
    #         "diverse_improved": []
    #     }
    # }
    results = {}
    results["overall"] = {}
    # for merging_prefix in merging_prefixes:
    #     results["overall"][merging_prefix] = []
    for test_set in test_sets:
        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
        # results[test_set_str] = {
        #     "single": [],
        #     "diverse": [],
        #     "single_improved": [],
        #     "diverse_improved": []
        # }
        results[test_set_str] = {}
        # for merging_prefix in merging_prefixes:
        #     results[test_set_str][merging_prefix] = []

        test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
        mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    # for i in range(4):
        # if i == 0:
        #     baselines = single_baselines
        #     result_key = "single"
        # elif i == 1:
        #     baselines = diverse_baselines
        #     result_key = "diverse"
        # elif i == 2:
        #     baselines = single_baselines_improved
        #     result_key = "single_improved"
        # else:
        #     baselines = diverse_baselines_improved
        #     result_key = "diverse_improved"


    for merging_prefix in merging_prefixes:
        # for baseline in baselines:
        baseline_accuracies = []
        baseline_true_positives_lst = []
        baseline_false_positives_lst = []
        baseline_false_negatives_lst = []
        all_count_diffs = []
        for test_set in test_sets:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])

            # rep_accuracies = []
            # for rep_num in range(1):
            model_name = baseline["model_name"] + "_rep_" + str(0)
            model_dir = os.path.join(test_set_image_set_dir, "model", "results")
            result_dir = os.path.join(model_dir, mappings[test_set_str][merging_prefix + model_name])
            excel_path = os.path.join(result_dir, "metrics.xlsx")
            df = pd.read_excel(excel_path, sheet_name=0)
            rep_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
            baseline_true_positives = df["True Positives (IoU=.50, conf>.50)"].sum(skipna=True)
            baseline_false_positives = df["False Positives (IoU=.50, conf>.50)"].sum(skipna=True)
            baseline_false_negatives = df["False Negatives (IoU=.50, conf>.50)"].sum(skipna=True)
            sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
            annotated_counts = sub_df["Annotated Count"]
            predicted_counts = sub_df["Predicted Count"]
            count_diffs = (predicted_counts - annotated_counts).tolist()

            # rep_accuracies.append(rep_accuracy)

            baseline_accuracy = rep_accuracy #np.mean(rep_accuracies)
            # baseline_variance = np.std(rep_accuracies)
            baseline_accuracies.append(baseline_accuracy)
            baseline_true_positives_lst.append(baseline_true_positives)
            baseline_false_positives_lst.append(baseline_false_positives)
            baseline_false_negatives_lst.append(baseline_false_negatives)
            all_count_diffs.extend(count_diffs)
            
            results[test_set_str][merging_prefix] = [
                    baseline_accuracy,
                    baseline_true_positives,
                    baseline_false_positives,
                    baseline_false_negatives,
                    count_diffs
             ]

            # results[test_set_str][merging_prefix].append(baseline_accuracy)

                # (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], baseline_accuracy))



        # overall_baseline_accuracy = np.mean(baseline_accuracies)

        results["overall"][merging_prefix] = [
            np.mean(baseline_accuracies),
            np.sum(baseline_true_positives_lst),
            np.sum(baseline_false_positives_lst),
            np.sum(baseline_false_negatives_lst),
            all_count_diffs
        ]

        # results["overall"][merging_prefix].append(
        #     # (baseline["model_name"][:len(baseline["model_name"])-len("_630_patches")], 
        #     overall_baseline_accuracy) #,
        #     # np.min(baseline_accuracies),
        #     # np.max(baseline_accuracies)))


    # color_options = ["salmon", "royalblue", "forestgreen"]
    # colors = {}
    # for i, merging_prefix in enumerate(merging_prefixes):
    #     colors[merging_prefix] = color_options[i]

    label_lookup = {
        "no_overlap_": "0% Overlap + NMS",
        "no_prune_": "50% Overlap + NMS",
        "": "50% Overlap + Box Prune + NMS",
        # "alt_prune_": "50% overlap \n+ alt_prune \n+ NMS"
    }

    for test_set_str in ["overall"]:
        total_min_diff = 100000
        total_max_diff = -100000
        for merging_prefix in merging_prefixes:
            print(results[test_set_str][merging_prefix][4])
            min_diff = np.min(results[test_set_str][merging_prefix][4])
            max_diff = np.max(results[test_set_str][merging_prefix][4])
            if min_diff < total_min_diff:
                total_min_diff = min_diff
            if max_diff > total_max_diff:
                total_max_diff = max_diff
            # bars.append(results[test_set_str][merging_prefix][i])
            # labels.append(label_lookup[merging_prefix])
        end_point = max(abs(total_min_diff), abs(total_max_diff))
        fig, axs = plt.subplots(len(merging_prefixes), 1, figsize=(10, 8))
        fig.suptitle("Methods for Merging Patch Predictions: Effect on Predicted Object Count") # Difference in Count (Predicted - Annotated)")
        for i, merging_prefix in enumerate(merging_prefixes):
            counts, bins = np.histogram(results[test_set_str][merging_prefix][4], bins=2*end_point, range=(-end_point, end_point))
            axs[i].stairs(counts, bins, color=my_plot_colors[i], fill=True)
            # label=label_lookup[merging_prefix], 
            # axs[i].legend()
            props = dict(boxstyle="round", facecolor="white", alpha=0.5)
            textstr = label_lookup[merging_prefix]
            axs[i].text(0.985, 0.94, textstr, transform=axs[i].transAxes,
                verticalalignment='top', horizontalalignment="right", bbox=props, fontsize=12)
            # axs[i].set_title(textstr)

            # plt.locator_params(axis="y", nbins=4)

            axs[i].set_ylabel("Number of Test Images")
            if i == 2:
                axs[i].set_xlabel("Difference in Object Count (Predicted Minus Annotated)")

            yticks = ticker.MaxNLocator(4)
            axs[i].yaxis.set_major_locator(yticks)
        plt.tight_layout()
        out_path = os.path.join("eval_charts", "box_pruning_accuracy_demo", "count_differences.svg") #png")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path) #, dpi=200)




def dic_plot():

    baseline = {
        "model_name": eval_fixed_patch_num_baselines[-1]
    }

    create_patch_merging_plot(eval_test_sets, baseline,
                              ["no_overlap_", "no_prune_", ""])


def accuracy_plot():
    out_dir = os.path.join("eval_charts", "box_pruning_accuracy_demo")

    result_names = []
    for perc in [0, 10, 20, 30, 40, 50]:
        result_name = "trimming_eval_" + str(perc) + "_percent_overlap"

        # get_all_trimming_results(result_name, eval_test_sets, out_dir)
        result_names.append(result_name)
    create_aggregate_trimming_plot(result_names, out_dir)




if __name__ == "__main__":

    # accuracy_plot()
    dic_plot()


