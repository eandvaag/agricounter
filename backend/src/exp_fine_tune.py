
import os
import numpy as np
import matplotlib.pyplot as plt


from io_utils import json_io
from models.common import annotation_utils, inference_metrics, box_utils
from exp_runner import my_plot_colors, get_mapping_for_test_set, eval_test_sets
import exp_fine_tune_runner


def create_fine_tune_plot_averaged_from_records(baseline, test_sets):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    # fig, axs = plt.subplots(3, 2, figsize=(12, 15)) #18, 10)) #plt.subplots(2, 3, figsize=(18, 10))
    fig, axs = plt.subplots(2, 3, figsize=(16, 11)) #12, 15))

    letter_labels = ["A", "B", "C", "D", "E", "F"]

    for i, test_set in enumerate(test_sets):
        print(i)
        # if i == 0:
        #     continue
        test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    
        record_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_results.json")
        results = json_io.load_json(record_path)

        ax = axs[i // 3, i % 3]
        # ax = axs[i // 2, i % 2]

        ax.plot([x[0] for x in results["random_patches_second"]], [x[1] for x in results["random_patches_second"]], c=my_plot_colors[0], label="Random Region Selection", zorder=1)
        ax.plot([x[0] for x in results["selected_patches_first"]], [x[1] for x in results["selected_patches_first"]], c=my_plot_colors[1], label="Uncertainty-Based Region Selection", zorder=1)
    
        for x in results["random_patches_second"]:
            #print(x)
            ax.scatter([x[0]] * len(x[3]), x[3], c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

        for x in results["selected_patches_first"]:
            ax.scatter([x[0]] * len(x[3]), x[3], c=my_plot_colors[1], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')


        ax.axhline(y=results["pre_fine_tune_accuracy"], c="black", linestyle="dashdot", label="No Fine-Tuning")

        # axs[i // 3, i % 3].legend()
        ax.set_ylabel("Instance-Based Accuracy")
        ax.set_xlabel("Number of Annotations Used For Fine-Tuning") #Annotations")

        # props = dict(edgecolor="white", facecolor="white") #boxstyle='round', facecolor='white') #, alpha=0.5)
        props = dict(edgecolor="black", facecolor="white", alpha=1, pad=5.0)

        ax.text(0.04, 0.89, letter_labels[i], transform=ax.transAxes, fontsize=24,
                                bbox=props)

        ax.set_ylim([0.7, 1])

    handles, labels = axs[0, 0].get_legend_handles_labels() #axs[1, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=13) # loc='upper right', fontsize=11)
    # from matplotlib import rcParams
    # rcParams['axes.titlepad'] = 20 
    fig.suptitle("Targeted Annotation for Fine-Tuning: Instance-Based Accuracy", size=18) #, y=1.12)


    plt.tight_layout()
    plt.subplots_adjust(top=0.86) 
    out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", "averaged_global_all_results_2x3.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=800)


def create_fine_tune_plot_averaged_dic_from_records(baseline, test_sets):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    letter_labels = ["A", "B", "C", "D", "E", "F"]

    for i, test_set in enumerate(test_sets):
        print(i)
        # if i == 0:
        #     continue
        test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    
        record_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_results.json")
        results = json_io.load_json(record_path)

        # ax = axs[i // 3, i % 3]
        ax = axs[i // 2, i % 2]

        ax.plot([x[0] for x in results["random_patches_second"]], [x[4] for x in results["random_patches_second"]], c=my_plot_colors[0], label="Random Region Selection", zorder=1)
        ax.plot([x[0] for x in results["selected_patches_first"]], [x[4] for x in results["selected_patches_first"]], c=my_plot_colors[1], label="Uncertainty-Based Region Selection", zorder=1)
    
        for x in results["random_patches_second"]:
            #print(x)
            ax.scatter([x[0]] * len(x[6]), x[6], c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

        for x in results["selected_patches_first"]:
            ax.scatter([x[0]] * len(x[6]), x[6], c=my_plot_colors[1], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')


        ax.axhline(y=results["pre_fine_tune_mean_abs_dic"], c="black", linestyle="dashdot", label="No Fine-Tuning")

        # axs[i // 3, i % 3].legend()
        ax.set_ylabel("Mean Absolute Difference In Count")
        ax.set_xlabel("Number of Annotations Used For Fine-Tuning") #Annotations")

        # props = dict(edgecolor="white", facecolor="white", alpha=0) #boxstyle='round', facecolor='white') #, alpha=0.5)
        props = dict(edgecolor="black", facecolor="white", alpha=1, pad=5.0)

        ax.text(0.04, 0.89, letter_labels[i], transform=ax.transAxes, fontsize=24,
                                bbox=props)

        ax.set_ylim([0, 30])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=13)
    # from matplotlib import rcParams
    # rcParams['axes.titlepad'] = 20 
    fig.suptitle("Targeted Annotation for Fine-Tuning: Mean Average Difference in Count", size=18) #Results for Six Image Sets", size=18) #, y=1.12)


    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", "averaged_global_all_results_dic.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=800)





def create_fine_tune_plot_averaged_did_from_records(baseline, test_sets):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    letter_labels = ["A", "B", "C", "D", "E", "F"]

    for i, test_set in enumerate(test_sets):
        print(i)
        # if i == 0:
        #     continue
        test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    
        record_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_results.json")
        results = json_io.load_json(record_path)

        # ax = axs[i // 3, i % 3]
        ax = axs[i // 2, i % 2]

        ax.plot([x[0] for x in results["random_patches_second"]], [x[7] for x in results["random_patches_second"]], c=my_plot_colors[0], label="Random Region Selection", zorder=1)
        ax.plot([x[0] for x in results["selected_patches_first"]], [x[7] for x in results["selected_patches_first"]], c=my_plot_colors[1], label="Uncertainty-Based Region Selection", zorder=1)
    
        for x in results["random_patches_second"]:
            #print(x)
            ax.scatter([x[0]] * len(x[9]), x[9], c=my_plot_colors[0], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')

        for x in results["selected_patches_first"]:
            ax.scatter([x[0]] * len(x[9]), x[9], c=my_plot_colors[1], marker="o", zorder=2, alpha=0.3, s=20, edgecolors='none')


        ax.axhline(y=results["pre_fine_tune_mean_abs_did"], c="black", linestyle="dashdot", label="No Fine-Tuning")

        # axs[i // 3, i % 3].legend()
        ax.set_ylabel("Mean Absolute Difference In Density")
        ax.set_xlabel("Number of Annotations Used For Fine-Tuning") #Annotations")

        props = dict(edgecolor="black", facecolor="white", alpha=1, pad=5.0) # boxstyle='round') #, facecolor='white') #, alpha=0.5)

        ax.text(0.04, 0.89, letter_labels[i], transform=ax.transAxes, fontsize=24,
                                bbox=props)

        ax.set_ylim([0, 7])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=13)
    # from matplotlib import rcParams
    # rcParams['axes.titlepad'] = 20 
    fig.suptitle("Targeted Annotation for Fine-Tuning: Mean Average Difference in Density", size=18) #Results for Six Image Sets", size=18) #, y=1.12)


    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", "averaged_global_all_results_did.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=800)









def create_fine_tune_plot_record(baseline, test_set, methods, num_annotations_to_select_lst, num_dups):
    for num_annotations_to_select in num_annotations_to_select_lst:
        exp_fine_tune_runner.check_fine_tuning_models(baseline, test_set, num_dups, num_annotations_to_select)
    
    test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
    test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
    

    mapping = get_mapping_for_test_set(test_set_image_set_dir)
    # annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
    # annotations = annotation_utils.load_annotations(annotations_path)
    results = {}
    # labels = []

    metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)
    camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
    camera_specs = json_io.load_json(camera_specs_path)



    pre_fine_tune_result_name = baseline["model_name"] + "_pre_finetune" #+ str(num_images_to_select)
    result_uuid = mapping[pre_fine_tune_result_name]
    result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)

    predictions_path = os.path.join(result_dir, "predictions.json")
    predictions = annotation_utils.load_predictions(predictions_path)

    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    # accuracies = []
    # for image_name in annotations.keys():
    #     sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
    #     accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
    #     accuracies.append(accuracy)

    # pre_fine_tune_accuracy = np.mean(accuracies)

    pre_fine_tune_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, list(annotations.keys()))


    predicted_counts = [int(np.sum(predictions[image_name]["scores"] > 0.5)) for image_name in annotations.keys()]
    annotated_counts = [int(annotations[image_name]["boxes"].shape[0]) for image_name in annotations.keys()]
    pre_fine_tune_mean_abs_dic = np.mean(abs(np.array(predicted_counts) - np.array(annotated_counts)))


    abs_dids = []
    for image_name in annotations.keys():
        anno_count = annotations[image_name]["boxes"].shape[0]
        pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

        height_px = metadata["images"][image_name]["height_px"]
        width_px = metadata["images"][image_name]["width_px"]
        area_px = height_px * width_px

        gsd = inference_metrics.get_gsd(camera_specs, metadata)
        area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

        annotated_count_per_square_metre = anno_count / area_m2
        predicted_count_per_square_metre = pred_count / area_m2

        abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
        abs_dids.append(abs_did)

    pre_fine_tune_mean_abs_did = np.mean(abs_dids)



    # label_lookup = {
    #     "selected_patches_match_both": "selected_patches",
    #     "random_images": "random_images"
    # }
    results["pre_fine_tune_accuracy"] = pre_fine_tune_accuracy
    results["pre_fine_tune_mean_abs_dic"] = pre_fine_tune_mean_abs_dic
    results["pre_fine_tune_mean_abs_did"] = pre_fine_tune_mean_abs_did

    max_num_fine_tuning_boxes = 0
    for i, method in enumerate(methods):
        results[method] = []
        for j in range(len(num_annotations_to_select_lst)):

            dup_accuracies = []
            dup_mean_abs_dics = []
            dup_mean_abs_dids = []
            for dup_num in range(num_dups):

                result_name = baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select_lst[j]) + "_annotations_dup_" + str(dup_num)
                result_uuid = mapping[result_name]
                result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)

                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)

                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)


                num_fine_tuning_boxes = 0
                num_fine_tuning_regions = 0
                # accuracies = []
                for image_name in annotations.keys():
                    # sel_pred_boxes = predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.50]
                    # accuracy = fine_tune_eval.get_accuracy(annotations[image_name]["boxes"], sel_pred_boxes)
                    # accuracies.append(accuracy)

                    num_fine_tuning_boxes += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                    num_fine_tuning_regions += len(annotations[image_name]["training_regions"])


                if num_fine_tuning_boxes > max_num_fine_tuning_boxes:
                    max_num_fine_tuning_boxes = num_fine_tuning_boxes
                global_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, list(annotations.keys())) #assessment_images_lst)

                predicted_counts = [int(np.sum(predictions[image_name]["scores"] > 0.5)) for image_name in annotations.keys()]
                annotated_counts = [int(annotations[image_name]["boxes"].shape[0]) for image_name in annotations.keys()]
                mean_abs_dic = np.mean(abs(np.array(predicted_counts) - np.array(annotated_counts)))

                abs_dids = []
                for image_name in annotations.keys():
                    anno_count = annotations[image_name]["boxes"].shape[0]
                    pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                    height_px = metadata["images"][image_name]["height_px"]
                    width_px = metadata["images"][image_name]["width_px"]
                    area_px = height_px * width_px

                    gsd = inference_metrics.get_gsd(camera_specs, metadata)
                    area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                    annotated_count_per_square_metre = anno_count / area_m2
                    predicted_count_per_square_metre = pred_count / area_m2

                    abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                    abs_dids.append(abs_did)

                mean_abs_did = np.mean(abs_dids)


                test_set_accuracy = global_accuracy

                # test_set_accuracy = global_accuracy #np.mean(accuracy)
                # test_set_accuracy = np.mean(accuracies)
                dup_accuracies.append(float(test_set_accuracy))

                dup_mean_abs_dics.append(float(mean_abs_dic))

                dup_mean_abs_dids.append(float(mean_abs_did))
                # results[method].append((num_fine_tuning_boxes, test_set_accuracy))
                # results[method].append((num_fine_tuning_regions, test_set_accuracy))

            # results.append(np.mean(dup_accuracies))
            # labels.append(method)
            results[method].append((num_annotations_to_select_lst[j], 
                                    float(np.mean(dup_accuracies)), 
                                    float(np.std(dup_accuracies)),
                                    dup_accuracies,
                                    float(np.mean(dup_mean_abs_dics)),
                                    float(np.std(dup_mean_abs_dics)),
                                    dup_mean_abs_dics,
                                    float(np.mean(dup_mean_abs_dids)),
                                    float(np.std(dup_mean_abs_dids)),
                                    dup_mean_abs_dids
                                    ))

    print(results)

    results_out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_results.json")
    out_dir = os.path.dirname(results_out_path)
    os.makedirs(out_dir, exist_ok=True)
    json_io.save_json(results_out_path, results)


    # fig = plt.figure(figsize=(10, 10))
    # # ax = fig.add_axes([0.05, 0.05, 0.9, 0.9]) #[0.35, 0.15, 0.5, 0.7])
    # ax = fig.add_subplot(111)


    # # for i in range(len(results["random_patches_second"])):
    # #     # ax.plot([results["random_patches_second"][i][0], results["selected_patches_first"][i][0]], 
    # #     #         [results["random_patches_second"][i][1], results["selected_patches_first"][i][1]], c="black", zorder=1)
        
    # #     ax.plot([i, i], 
    # #             [results["random_patches_second"][i][1], results["selected_patches_first"][i][1]], c="black", zorder=1)
    # # for i, method in enumerate(list(results.keys())):
    # #     ax.scatter([x[0] for x in results[method]], [x[1] for x in results[method]], s=50, c=my_plot_colors[i], label=method, zorder=2)

    # # ax.scatter([x[0] for x in results["random_patches_second"]], [x[1] for x in results["random_patches_second"]], marker="_", c=my_plot_colors[0], label="random_patches_second", zorder=2)
    # # ax.scatter([x[0] for x in results["selected_patches_first"]], [x[1] for x in results["selected_patches_first"]], marker="_", c=my_plot_colors[1], label="selected_patches_first", zorder=2)
    
    # # for x in results["random_patches_second"]:
    # #     ax.plot([x[0], x[0]], [x[1]-x[2], x[1]+x[2]], c=my_plot_colors[0])
    

    # # for x in results["selected_patches_first"]:
    # #     ax.plot([x[0], x[0]], [x[1]-x[2], x[1]+x[2]], c=my_plot_colors[1])


    # ax.plot([x[0] for x in results["random_patches_second"]], [x[1] for x in results["random_patches_second"]], c=my_plot_colors[0], label="Random Patches", zorder=1)
    # ax.plot([x[0] for x in results["selected_patches_first"]], [x[1] for x in results["selected_patches_first"]], c=my_plot_colors[1], label="Selected Patches", zorder=1)
    
    # # ax.fill_between([x[0] for x in results["random_patches_second"]], 
    # #                 [x[1] - x[2] for x in results["random_patches_second"]], 
    # #                 [x[1] + x[2] for x in results["random_patches_second"]], edgecolor=my_plot_colors[0], facecolor=my_plot_colors[0], alpha=0.15)
    # # ax.fill_between([x[0] for x in results["selected_patches_first"]], 
    # #                 [x[1] - x[2] for x in results["selected_patches_first"]], 
    # #                 [x[1] + x[2] for x in results["selected_patches_first"]], edgecolor=my_plot_colors[1], facecolor=my_plot_colors[1], alpha=0.15)
    # for x in results["random_patches_second"]:
    #     print(x)
    #     ax.scatter([x[0]] * len(x[3]), x[3], c=my_plot_colors[0], marker="o", zorder=2, alpha=0.7, s=70, edgecolors='none')

    # for x in results["selected_patches_first"]:
    #     ax.scatter([x[0]] * len(x[3]), x[3], c=my_plot_colors[1], marker="o", zorder=2, alpha=0.7, s=70, edgecolors='none')

    # # ax.scatter([x[3] for x in results["random_patches_second"]], [x[4] for x in results["random_patches_second"]], s=50, c=my_plot_colors[0], label="random_patches_second", zorder=2)
    # # ax.scatter([x[3] for x in results["selected_patches_first"]], [x[4] for x in results["selected_patches_first"]], s=50, c=my_plot_colors[1], label="selected_patches_first", zorder=2)


    # # ax.plot([0, max_num_fine_tuning_boxes], [pre_fine_tune_accuracy, pre_fine_tune_accuracy], c="black", linestyle="dashed", label="No Fine-Tuning")

    # # ax.scatter(results, np.arange(len(labels))) #, color=colors) #, width=0.4)

    # # ax.set_yticks(np.arange(len(labels)))
    # # ax.set_yticklabels(labels)

    # plt.axhline(y=pre_fine_tune_accuracy, c="black", linestyle="dashdot", label="No Fine-Tuning")
    # ax.legend()
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("Number of Patches") #Annotations")

    # ax.set_ylim([0.7, 1])

    # plt.tight_layout()

    # out_path = os.path.join("eval_charts", "fine_tuning", "selected_first", test_set_str + "_" + baseline["model_name"] + "_averaged_global.svg")
    # out_dir = os.path.dirname(out_path)
    # os.makedirs(out_dir, exist_ok=True)
    # plt.savefig(out_path)


def create_plot():


    fine_tune_lookup_nums = {
        0: [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5560], 
        1: [250, 500, 750, 1000, 1250, 1500, 1710],
        2: [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 2977],
        3: [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3194],
        4: [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 4786], 
        5: [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5505]
    }

    # k = 1

    baselines = [{"model_name": "set_of_27_38891_patches_rep_0", "model_creator": "eval"}]

    methods = [
        "selected_patches_first",
        "random_patches_second",
    ]

    num_dups = 5

    # for k in [2]: #range(6):
    #     create_fine_tune_plot_record(baselines[0], 
    #                                 eval_test_sets[k], 
    #                                 methods, 
    #                                 fine_tune_lookup_nums[k],
    #                                 num_dups)
    
    create_fine_tune_plot_averaged_from_records(baselines[0], [eval_test_sets[i] for i in range(6)])
    # create_fine_tune_plot_averaged_dic_from_records(baselines[0], [eval_test_sets[i] for i in range(6)])
    # create_fine_tune_plot_averaged_did_from_records(baselines[0], [eval_test_sets[i] for i in range(6)])
    


if __name__ == "__main__":
    create_plot()