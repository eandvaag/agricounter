import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random

from io_utils import json_io
from models.common import inference_metrics, annotation_utils
from exp_runner import get_mapping_for_test_set, eval_in_domain_test_sets, \
                        eval_test_sets, eval_fixed_patch_num_baselines, eval_best_to_worst_baselines, \
                        my_plot_colors, ns_in_domain_test_sets, ns_test_sets, ns_fixed_patch_num_baselines



bad_id_test_sets = [
    'eval BlaineLake River 2021-06-09',
    'eval row_spacing brown 2021-06-08',
    'eval Biggar Dennis1 2021-06-04',
    'eval BlaineLake Serhienko9S 2022-06-14'
]


def create_individual_images_eval_size_plot_id_ood(id_test_sets, ood_test_sets, baseline_sets, out_dirname):

    print("num id test sets: {}".format(len(id_test_sets)))
    print("num ood test sets: {}".format(len(ood_test_sets)))
    # results = {}
    mappings = {}
    test_set_str_to_label = {}
    print("getting mappings")
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):

        if i == 0:
            label = "id"
        else:
            label = "ood"

        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            test_set_str_to_label[test_set_str] = label
    print("done getting mappings")

    if False: 
        #os.path.exists(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json")) and \
        # os.path.exists(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json")):
            
        image_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json"))
        instance_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"))
    else:
        image_based_test_set_results = {}
        # instance_based_test_set_results = {}

        json_io.print_json(mappings)
        for k in baseline_sets.keys():
            # results[k] = []
            baselines = baseline_sets[k]
            for baseline in baselines:
                model_name = baseline["model_name"]
                patch_num = baseline["patch_num"]

                print(model_name)

                for test_set in id_test_sets:

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    
                    print("\tid: {}".format(test_set_str))
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])                
                    
                    all_reps_id_test_set_accuracies = []
                    # id_global_test_set_accuracies = []
                    for rep_num in range(5):
                        print("\t\t{}".format(rep_num))
                        

                        

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)

                        id_test_set_accuracies = df[~df["Accuracy (IoU=.50, conf>.50)"].isna()]


                        all_reps_id_test_set_accuracies.append(id_test_set_accuracies)

                    # print(np.array(all_reps_id_test_set_accuracies).shape)


                    image_accs = np.mean(np.array(all_reps_id_test_set_accuracies), axis=0)

                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []
                    image_based_test_set_results[test_set_str].append(
                        (patch_num, image_accs)
                    )
                

                for test_set in ood_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

                    print("\tood: {}".format(test_set_str))
                    all_reps_ood_test_set_accuracies = []
                    for rep_num in range(5):
                        print("\t\t{}".format(rep_num))
                    

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)
                        # inds = df["Annotated Count"] > 10
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)


                        ood_test_set_accuracies = df[~df["Accuracy (IoU=.50, conf>.50)"].isna()]["Accuracy (IoU=.50, conf>.50)"]
                        print(ood_test_set_accuracies)
                        all_reps_ood_test_set_accuracies.append(ood_test_set_accuracies)
                    
                    print(np.array(all_reps_ood_test_set_accuracies).shape)
                    image_accs = np.mean(np.array(all_reps_ood_test_set_accuracies), axis=0)

                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []

                    image_based_test_set_results[test_set_str].append(
                        (patch_num, image_accs)
                    )


    # json_io.save_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json"), image_based_test_set_results)
    # json_io.save_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"), instance_based_test_set_results)

    # json_io.print_json(instance_based_test_set_results)
    # fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # for test_set_str in image_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         plot_color = my_plot_colors[0]
    #     else:
    #         plot_color = my_plot_colors[1]
    #     axs[0].plot(
    #         [x[0] for x in image_based_test_set_results[test_set_str]],
    #         [x[1] for x in image_based_test_set_results[test_set_str]],
    #         color=plot_color,
    #         alpha=0.5
    #     )

    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         plot_color = my_plot_colors[0]
    #     else:
    #         plot_color = my_plot_colors[1]
    #     axs[1].plot(
    #         [x[0] for x in instance_based_test_set_results[test_set_str]],
    #         [x[1] for x in instance_based_test_set_results[test_set_str]],
    #         color=plot_color,
    #         alpha=0.5
    #     )

    xticks = [0, 10000, 20000, 30000, 40000]
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 4))

    # largest_set_id_results = []
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         largest_set_id_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    id_label_added = False
    ood_label_added = False
    for test_set_str in image_based_test_set_results.keys(): #instance_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "id":
            plot_color = my_plot_colors[0]
            if not id_label_added:
                label = "In Domain"
                id_label_added = True
            else:
                label = None
        else:
            plot_color = my_plot_colors[1]
            if not ood_label_added:
                label = "Out Of Domain"
                ood_label_added = True
            else:
                label = None

        for image_index in range(len(image_based_test_set_results[test_set_str][0][1])):

            axs.plot(
                [x[0] for x in image_based_test_set_results[test_set_str]], # instance_based_test_set_results[test_set_str]],
                [x[1][image_index] for x in image_based_test_set_results[test_set_str]], # instance_based_test_set_results[test_set_str]],
                color=plot_color,
                alpha=0.3, #1.0, #0.5,
                label=label
            )

    # axs.set_ylim([0.3, 0.9])
    # axs.legend()

    axs.set_xlabel("Number of Training Patches")
    axs.set_ylabel("Instance-Based Accuracy")
    axs.set_xticks(xticks)

    # largest_set_id_results = []
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         largest_set_id_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    
    # largest_set_ood_results = []
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "ood":
    #         largest_set_ood_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    
    # largest_set_ood_results = []
    # for test_set_str in image_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "ood":
    #         largest_set_ood_results.append((test_set_str, image_based_test_set_results[test_set_str][-1][1]))
    

    # largest_set_id_results.sort(key=lambda x: x[1])
        
    # print("largest_set_id_results")
    # for v in largest_set_id_results:
    #     print(v)

    # print(largest_set_id_results)
    # largest_set_ood_results.sort(key=lambda x: x[1])
    # print("largest_set_ood_results")
    # for v in largest_set_ood_results:
    #     print(v)
    # print(largest_set_ood_results)

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.84)

    # plt.xscale("log")

    plt.suptitle("Effect of Training Set Size on Instance-Based Accuracy") #\n(Individual Image Sets)")


    out_path = os.path.join("eval_charts", out_dirname, "ood_individual_images.svg") #"id_ood_individual_image_sets.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)












def create_individual_image_sets_eval_size_plot_id_ood(id_test_sets, ood_test_sets, baseline_sets, out_dirname):

    print("num id test sets: {}".format(len(id_test_sets)))
    print("num ood test sets: {}".format(len(ood_test_sets)))
    # results = {}
    mappings = {}
    test_set_str_to_label = {}
    print("getting mappings")
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):

        if i == 0:
            label = "id"
        else:
            label = "ood"

        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            test_set_str_to_label[test_set_str] = label
    print("done getting mappings")

    if False: 
        #os.path.exists(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json")) and \
        # os.path.exists(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json")):
            
        image_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json"))
        instance_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"))
    else:
        image_based_test_set_results = {}
        instance_based_test_set_results = {}

        json_io.print_json(mappings)
        for k in baseline_sets.keys():
            # results[k] = []
            baselines = baseline_sets[k]
            for baseline in baselines:
                model_name = baseline["model_name"]
                patch_num = baseline["patch_num"]

                print(model_name)

                for test_set in id_test_sets:

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    
                    print("\tid: {}".format(test_set_str))
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])                
                    
                    id_test_set_accuracies = []
                    id_global_test_set_accuracies = []
                    for rep_num in range(1): #5):
                        print("\t\t{}".format(rep_num))
                        

                        

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)

                        # inds = df["Annotated Count"] > 10
                        # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        assessment_images = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                assessment_images.append(image_name)

                        id_global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                        # id_global_test_set_accuracy = 0

                        id_test_set_accuracies.append(id_test_set_accuracy)
                        id_global_test_set_accuracies.append(id_global_test_set_accuracy)


                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []
                    image_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(id_test_set_accuracies))
                    )
                    if test_set_str not in instance_based_test_set_results:
                        instance_based_test_set_results[test_set_str] = []
                    instance_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(id_global_test_set_accuracies))
                    )
                

                for test_set in ood_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

                    # print("\tood: {}".format(test_set_str))
                    ood_test_set_accuracies = []
                    ood_global_test_set_accuracies = []
                    for rep_num in range(1): #5):
                        print("\t\t{}".format(rep_num))
                    

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)
                        # inds = df["Annotated Count"] > 10
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                        

                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        assessment_images = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                assessment_images.append(image_name)

                        # full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                        # print("loading predictions")
                        # full_predictions = annotation_utils.load_predictions(full_predictions_path)
                        # print("loaded")
                        # for image_name in full_predictions.keys():
                        #     inds = full_predictions[image_name]["scores"] > 0.01
                        #     full_predictions[image_name]["scores"] = full_predictions[image_name]["scores"][inds]
                        #     full_predictions[image_name]["boxes"] = full_predictions[image_name]["boxes"][inds]
                        # test_set_mAP_05 = float(inference_metrics.get_mAP_val(annotations, full_predictions, ".50", assessment_images))

                        # ood_global_test_set_accuracy = test_set_mAP_05

                        # assessment_images = random.sample(assessment_images, 2)
                        # assessment_images = (np.array(assessment_images)[[0,3]]).tolist()

                        # ood_test_set_accuracy = df[df["Image Name"].astype(str).isin(assessment_images)]["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                        # print(assessment_images)
                        # print(df["Image Name"])
                        # print(ood_test_set_accuracy)

                        ood_global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                        

                        
                        # ood_global_test_set_accuracy = 0


                        ood_test_set_accuracies.append(ood_test_set_accuracy)
                        ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)


                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []

                    image_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(ood_test_set_accuracies))
                    )

                    if test_set_str not in instance_based_test_set_results:
                        instance_based_test_set_results[test_set_str] = []

                    instance_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(ood_global_test_set_accuracies))
                    )

    json_io.save_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results.json"), image_based_test_set_results)
    json_io.save_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"), instance_based_test_set_results)

    # json_io.print_json(instance_based_test_set_results)
    # fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # for test_set_str in image_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         plot_color = my_plot_colors[0]
    #     else:
    #         plot_color = my_plot_colors[1]
    #     axs[0].plot(
    #         [x[0] for x in image_based_test_set_results[test_set_str]],
    #         [x[1] for x in image_based_test_set_results[test_set_str]],
    #         color=plot_color,
    #         alpha=0.5
    #     )

    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         plot_color = my_plot_colors[0]
    #     else:
    #         plot_color = my_plot_colors[1]
    #     axs[1].plot(
    #         [x[0] for x in instance_based_test_set_results[test_set_str]],
    #         [x[1] for x in instance_based_test_set_results[test_set_str]],
    #         color=plot_color,
    #         alpha=0.5
    #     )

    xticks = [0, 10000, 20000, 30000, 40000]
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 3.6))

    # largest_set_id_results = []
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         largest_set_id_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    id_label_added = False
    ood_label_added = False
    for test_set_str in instance_based_test_set_results.keys(): #instance_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "id":
            plot_color = my_plot_colors[0]
            if not id_label_added:
                label = "In Domain"
                id_label_added = True
            else:
                label = None
        else:
            plot_color = my_plot_colors[1]
            if not ood_label_added:
                label = "Out Of Domain"
                ood_label_added = True
            else:
                label = None
        axs.plot(
            [x[0] for x in instance_based_test_set_results[test_set_str]], # instance_based_test_set_results[test_set_str]],
            [x[1] for x in instance_based_test_set_results[test_set_str]], # instance_based_test_set_results[test_set_str]],
            color=plot_color,
            alpha=0.5, #1.0, #0.5,
            label=label
        )

    # axs.set_ylim([0.3, 0.9])
    # axs.legend()

    axs.set_xlabel("Number of Training Patches")
    axs.set_ylabel("Accuracy") #mAP@0.5") #Instance-Based Accuracy")
    axs.set_xticks(xticks)

    # largest_set_id_results = []
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         largest_set_id_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    
    largest_set_ood_results = []
    for test_set_str in instance_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "ood":
            largest_set_ood_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    
    # largest_set_ood_results = []
    # for test_set_str in image_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "ood":
    #         largest_set_ood_results.append((test_set_str, image_based_test_set_results[test_set_str][-1][1]))
    

    # largest_set_id_results.sort(key=lambda x: x[1])
        
    # print("largest_set_id_results")
    # for v in largest_set_id_results:
    #     print(v)

    # print(largest_set_id_results)
    largest_set_ood_results.sort(key=lambda x: x[1])
    print("largest_set_ood_results")
    for v in largest_set_ood_results:
        print(v)
    print(largest_set_ood_results)

    legend_elements = [
        Line2D([], [], color=my_plot_colors[1], alpha=0.6, label="Test Image Set")
        # Patch(color=my_plot_colors[0], label="Test Image Set Model", linewidth=2.5),
        # Patch(color=my_plot_colors[1], label="Single Image Set Model", linewidth=2.5)
    ]

    legend_loc = "lower right"

    axs.legend(loc=legend_loc, 
                  handles=legend_elements, 
                  handlelength=0.50, #15, #05, 
                  edgecolor="black",
                  fontsize=12)


    # plt.tight_layout()
    # plt.subplots_adjust(top=0.84)

    # plt.xscale("log")
    # plt.subplots_adjust(top=0.985)
    plt.tight_layout()
    # plt.suptitle("Effect of Training Set Size on Accuracy") #mAP") #Instance-Based Accuracy") #\n(Individual Image Sets)")


    out_path = os.path.join("eval_charts", out_dirname, "ns_individual_image_sets.png") # "paper_plot_ood_individual_image_sets.png") #"id_ood_individual_image_sets.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)






def create_individual_image_sets_eval_size_plot_id_ood_abs_did(id_test_sets, ood_test_sets, baseline_sets, out_dirname):

    print("num id test sets: {}".format(len(id_test_sets)))
    print("num ood test sets: {}".format(len(ood_test_sets)))
    image_based_test_set_results = {}
    mappings = {}
    test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):

        if i == 0:
            label = "id"
        else:
            label = "ood"

        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            test_set_str_to_label[test_set_str] = label

    if False: #os.path.exists(os.path.join("eval_charts", out_dirname, "image_based_test_set_results_abs_dic.json")):
            
        image_based_test_set_results = json_io.load_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results_abs_dic.json"))
    else:
        image_based_test_set_results = {}
        instance_based_test_set_results = {}

        json_io.print_json(mappings)
        for k in baseline_sets.keys():
            # results[k] = []
            baselines = baseline_sets[k]
            for baseline in baselines:
                model_name = baseline["model_name"]
                patch_num = baseline["patch_num"]

                print(model_name)

                for test_set in id_test_sets:

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    
                    print("\tid: {}".format(test_set_str))
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])

                    metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
                    metadata = json_io.load_json(metadata_path)
                    camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
                    camera_specs = json_io.load_json(camera_specs_path)          
                    
                    id_test_set_abs_dids = []
                    # id_global_test_set_accuracies = []
                    for rep_num in range(5):
                        print("\t\t{}".format(rep_num))
                        

                        

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)

                        # inds = df["Annotated Count"] > 10
                        # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        # sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                        # annotated_counts = sub_df["Annotated Count"]
                        # predicted_counts = sub_df["Predicted Count"]
                        # count_diffs = (predicted_counts - annotated_counts).tolist()



                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        # assessment_images = []
                        # for image_name in annotations.keys():
                        #     if len(annotations[image_name]["test_regions"]) > 0:
                        #         assessment_images.append(image_name)

                        # id_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                        # id_global_test_set_accuracy = 0


                        abs_dids = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                # assessment_images.append(image_name)


                                anno_count = annotations[image_name]["boxes"].shape[0]
                                pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                                height_px = metadata["images"][image_name]["height_px"]
                                width_px = metadata["images"][image_name]["width_px"]
                                area_px = height_px * width_px

                                gsd = inference_metrics.get_gsd(camera_specs, metadata)
                                area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                                # print("area_m2", area_m2)

                                annotated_count_per_square_metre = anno_count / area_m2
                                predicted_count_per_square_metre = pred_count / area_m2

                                abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                                abs_dids.append(abs_did)




                        # id_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))
                        id_test_set_abs_dids.append(np.mean(abs_dids))

                        # id_test_set_accuracies.append(id_test_set_accuracy)
                        # id_global_test_set_accuracies.append(id_global_test_set_accuracy)


                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []
                    image_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(id_test_set_abs_dids))
                    )
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str] = []
                    # instance_based_test_set_results[test_set_str].append(
                    #     (patch_num, np.mean(id_global_test_set_accuracies))
                    # )
                

                for test_set in ood_test_sets:
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

                    print("\tood: {}".format(test_set_str))
                    ood_test_set_abs_dids = []
                    # ood_global_test_set_accuracies = []
                    for rep_num in range(5):
                        print("\t\t{}".format(rep_num))
                    

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)
                        # inds = df["Annotated Count"] > 10
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                        # sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                        # annotated_counts = sub_df["Annotated Count"]
                        # predicted_counts = sub_df["Predicted Count"]
                        # count_diffs = (predicted_counts - annotated_counts).tolist()



                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        # assessment_images = []
                        # for image_name in annotations.keys():
                        #     if len(annotations[image_name]["test_regions"]) > 0:
                        #         assessment_images.append(image_name)
                        # assessment_images = random.sample(assessment_images, 2)
                        # assessment_images = (np.array(assessment_images)[[0,3]]).tolist()



                        abs_dids = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                # assessment_images.append(image_name)


                                anno_count = annotations[image_name]["boxes"].shape[0]
                                pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                                height_px = metadata["images"][image_name]["height_px"]
                                width_px = metadata["images"][image_name]["width_px"]
                                area_px = height_px * width_px

                                gsd = inference_metrics.get_gsd(camera_specs, metadata)
                                area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                                print("area_m2", area_m2)

                                annotated_count_per_square_metre = anno_count / area_m2
                                predicted_count_per_square_metre = pred_count / area_m2

                                abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                                abs_dids.append(abs_did)


                        ood_test_set_abs_dids.append(np.mean(abs_dids))

                        # ood_test_set_accuracy = df[df["Image Name"].astype(str).isin(assessment_images)]["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                        # print(assessment_images)
                        # print(df["Image Name"])
                        # print(ood_test_set_accuracy)
                        # ood_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                        # ood_global_test_set_accuracy = 0

                        # ood_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                        # ood_test_set_accuracies.append(ood_test_set_accuracy)
                        # ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)


                    if test_set_str not in image_based_test_set_results:
                        image_based_test_set_results[test_set_str] = []

                    image_based_test_set_results[test_set_str].append(
                        (patch_num, np.mean(ood_test_set_abs_dids))
                    )

                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str] = []

                    # instance_based_test_set_results[test_set_str].append(
                    #     (patch_num, np.mean(ood_global_test_set_accuracies))
                    # )

    json_io.save_json(os.path.join("eval_charts", out_dirname, "image_based_test_set_results_abs_did.json"), image_based_test_set_results)
    # json_io.save_json(os.path.join("eval_charts", out_dirname, "instance_based_test_set_results.json"), instance_based_test_set_results)

    # json_io.print_json(instance_based_test_set_results)
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 3.6))
    for test_set_str in image_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "id":
            plot_color = my_plot_colors[0]
        else:
            plot_color = my_plot_colors[1]
        axs.plot(
            [x[0] for x in image_based_test_set_results[test_set_str]],
            [x[1] for x in image_based_test_set_results[test_set_str]],
            color=plot_color,
            alpha=0.5
        )

    axs.set_xlabel("Number of Training Patches")
    axs.set_ylabel("Mean Absolute Difference in Density")
    xticks = [0, 10000, 20000, 30000, 40000]
    axs.set_xticks(xticks)
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         plot_color = my_plot_colors[0]
    #     else:
    #         plot_color = my_plot_colors[1]
    #     axs[1].plot(
    #         [x[0] for x in instance_based_test_set_results[test_set_str]],
    #         [x[1] for x in instance_based_test_set_results[test_set_str]],
    #         color=plot_color,
    #         alpha=0.5
    #     )

    # largest_set_id_results = []
    # for test_set_str in instance_based_test_set_results.keys():
    #     if test_set_str_to_label[test_set_str] == "id":
    #         largest_set_id_results.append((test_set_str, instance_based_test_set_results[test_set_str][-1][1]))
    
    # largest_set_id_results.sort(key=lambda x: x[1])
        
    # print("largest_set_id_results")
    # for v in largest_set_id_results:
    #     print(v)
    # print(largest_set_id_results)

    # plt.xscale("log")

    largest_set_ood_results = []
    for test_set_str in image_based_test_set_results.keys():
        if test_set_str_to_label[test_set_str] == "ood":
            largest_set_ood_results.append((test_set_str, image_based_test_set_results[test_set_str][-1][1]))

    largest_set_ood_results.sort(key=lambda x: x[1])
    print("largest_set_ood_results")
    for v in largest_set_ood_results:
        print(v)
    print(largest_set_ood_results)

    legend_elements = [
        Line2D([], [], color=my_plot_colors[1], alpha=0.6, label="Test Image Set")
        # Patch(color=my_plot_colors[0], label="Test Image Set Model", linewidth=2.5),
        # Patch(color=my_plot_colors[1], label="Single Image Set Model", linewidth=2.5)
    ]

    legend_loc = "upper right"

    axs.legend(loc=legend_loc, 
                  handles=legend_elements, 
                  handlelength=0.50, #15, #05, 
                  edgecolor="black",
                  fontsize=12)

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.84)

    # plt.suptitle("Effect of Training Set Size on Count Accuracy")

    # plt.subplots_adjust(top=0.985)
    plt.tight_layout()
    

    out_path = os.path.join("eval_charts", out_dirname, "paper_plot_ood_individual_image_sets_abs_did.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)





def create_eval_size_plot_id_ood_percent_count_error(id_test_sets, ood_test_sets, baseline_sets, out_dirname):


    results = {} #[]
    mappings = {}
    # test_set_types = [("ood", ood_test_sets), ("id", id_test_sets)]
    # labels = {}
    # test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):
        print(i)

        # if i == 0:
        #     label = "id"
        # else:
        #     label = "ood"

        # test_set_label = test_set_type[0]
        # test_sets = test_set_type[1]
        # labels[test_set_label] = []
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            # labels[test_set_label].append(test_set_str)
            # test_set_str_to_label[test_set_str] = label


    json_io.print_json(mappings)
    for k in baseline_sets.keys():
        results[k] = []
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]
            
            # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
            ood_rep_abs_dics = []
            # ood_global_rep_accuracies = []
            id_rep_abs_dics = []
            # id_global_rep_accuracies = []
            for rep_num in range(5):

                print(rep_num)






                id_test_set_abs_dics = []
                # id_global_test_set_accuracies = []
                id_abs_dics = []

                for test_set in id_test_sets:
                    
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    # if test_set_str in bad_id_test_sets:
                    #     continue
                    
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    print("\t id", test_set_image_set_dir)
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)

                    # inds = df["Annotated Count"] > 10
                    # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                    sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    annotated_counts = sub_df["Annotated Count"]
                    predicted_counts = sub_df["Predicted Count"]
                    count_diffs = (predicted_counts - annotated_counts).tolist()

                    count_errors = []
                    for annotated_count, predicted_count in zip(annotated_counts, predicted_counts):
                        if annotated_count > 0:
                            count_errors.append(abs(annotated_count - predicted_count) / annotated_count)



                    # predictions_path = os.path.join(result_dir, "predictions.json")
                    # predictions = annotation_utils.load_predictions(predictions_path)
                    # annotations_path = os.path.join(result_dir, "annotations.json")
                    # annotations = annotation_utils.load_annotations(annotations_path)
                    # assessment_images = []
                    # for image_name in annotations.keys():
                    #     if len(annotations[image_name]["test_regions"]) > 0:
                    #         assessment_images.append(image_name)

                    # id_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")



                    # id_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    id_abs_dics.extend(count_errors) #np.abs(count_diffs).tolist())
                    # id_global_test_set_accuracies.append(id_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(id_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(id_global_test_set_accuracy)

                # id_rep_abs_dic = np.mean(id_test_set_abs_dics)
                # id_rep_abs_dics.append(id_rep_abs_dic)

                id_rep_abs_dics.append(np.mean(id_abs_dics))

                # id_global_rep_accuracy = np.mean(id_global_test_set_accuracies)
                # id_global_rep_accuracies.append(id_global_rep_accuracy)                










                # ood_test_set_accuracies = []
                ood_test_set_abs_dics = []
                # ood_global_test_set_accuracies = []
                ood_abs_dics = []

                for test_set in ood_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    print("\t ood:", test_set_image_set_dir)
                    

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    # inds = df["Annotated Count"] > 10
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                    sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    annotated_counts = sub_df["Annotated Count"]
                    predicted_counts = sub_df["Predicted Count"]
                    # count_diffs = (predicted_counts - annotated_counts).tolist()

                    count_errors = []
                    for annotated_count, predicted_count in zip(annotated_counts, predicted_counts):
                        if annotated_count > 0:
                            count_errors.append(abs(annotated_count - predicted_count) / annotated_count)



                    # predictions_path = os.path.join(result_dir, "predictions.json")
                    # predictions = annotation_utils.load_predictions(predictions_path)
                    # annotations_path = os.path.join(result_dir, "annotations.json")
                    # annotations = annotation_utils.load_annotations(annotations_path)
                    # assessment_images = []
                    # for image_name in annotations.keys():
                    #     if len(annotations[image_name]["test_regions"]) > 0:
                    #         assessment_images.append(image_name)

                    # ood_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    # ood_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    ood_abs_dics.extend(count_errors) #np.abs(count_diffs).tolist())
                    # ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(ood_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(ood_global_test_set_accuracy)


                # ood_rep_abs_dic = np.mean(ood_test_set_abs_dics)
                # ood_rep_abs_dics.append(ood_rep_abs_dic)
                ood_rep_abs_dics.append(np.mean(ood_abs_dics)) #ood_rep_abs_dic)


                # ood_rep_accuracy = np.mean(ood_test_set_accuracies)
                # ood_rep_accuracies.append(ood_rep_accuracy)
                # # print(model_name, rep_accuracy)


                # ood_global_rep_accuracy = np.mean(ood_global_test_set_accuracies)
                # ood_global_rep_accuracies.append(ood_global_rep_accuracy)                



            id_baseline_abs_dic = np.mean(id_rep_abs_dics) #None #np.mean(id_rep_abs_dics)
            id_baseline_stdev = np.std(id_rep_abs_dics) #None #np.std(id_rep_abs_dics)

            ood_baseline_abs_dic = np.mean(ood_rep_abs_dics)
            ood_baseline_stdev = np.std(ood_rep_abs_dics)
            # ood_baseline_accuracy = np.mean(ood_rep_accuracies)
            # ood_baseline_stdev = np.std(ood_rep_accuracies)


            # id_global_baseline_accuracy = np.mean(id_global_rep_accuracies)
            # id_global_baseline_stdev = np.std(id_global_rep_accuracies)

            # ood_global_baseline_accuracy = np.mean(ood_global_rep_accuracies)
            # ood_global_baseline_stdev = np.std(ood_global_rep_accuracies)

            # print(ood_baseline_accuracy, ood_global_baseline_accuracy)


            results[k].append((patch_num, 
                               id_baseline_abs_dic, 
                               id_baseline_stdev, 
                               ood_baseline_abs_dic, 
                               ood_baseline_stdev,
                               ))

    # fig = plt.figure(figsize=(8, 6))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    xticks = [0, 40000]


    for i, k in enumerate(list(results.keys())):
        # axs[0].scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
        # axs[0].scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

        axs.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], label="In Domain")
        axs.plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
        
        axs.fill_between([x[0] for x in results[k]], [x[1] - x[2] for x in results[k]], [x[1] + x[2] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
        axs.fill_between([x[0] for x in results[k]], [x[3] - x[4] for x in results[k]], [x[3] + x[4] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

    axs.set_xlabel("Number of Training Patches")
    axs.set_ylabel("Mean Percent Count Error") #Average Image Set Absolute Difference in Count")
    axs.set_ylim([0, None])
    axs.legend()

    # plt.suptitle("Effect of Training Set Size")

    # plt.legend()
    plt.tight_layout()

    out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_percent_count_error.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)



def create_eval_size_plot_id_ood_abs_did(id_test_sets, ood_test_sets, baseline_sets, out_dirname, remove_difficult_id_sets=False):


    results = {} #[]
    mappings = {}
    # test_set_types = [("ood", ood_test_sets), ("id", id_test_sets)]
    # labels = {}
    # test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):
        print(i)

        # if i == 0:
        #     label = "id"
        # else:
        #     label = "ood"

        # test_set_label = test_set_type[0]
        # test_sets = test_set_type[1]
        # labels[test_set_label] = []
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            # labels[test_set_label].append(test_set_str)
            # test_set_str_to_label[test_set_str] = label


    json_io.print_json(mappings)
    for k in baseline_sets.keys():
        results[k] = []
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]
            
            # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
            ood_rep_abs_dids = []
            # ood_global_rep_accuracies = []
            id_rep_abs_dids = []
            # id_global_rep_accuracies = []
            for rep_num in range(5):

                print(rep_num)






                id_test_set_abs_dics = []
                # id_global_test_set_accuracies = []
                id_abs_dics = []
                test_set_abs_dids = []

                for test_set in id_test_sets:
                    if test_set_str in bad_id_test_sets and remove_difficult_id_sets:
                        continue
                    
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    # if test_set_str in bad_id_test_sets:
                    #     continue
                    
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
                    metadata = json_io.load_json(metadata_path)
                    camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
                    camera_specs = json_io.load_json(camera_specs_path)
                    
                    print("\t id", test_set_image_set_dir)
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    # excel_path = os.path.join(result_dir, "metrics.xlsx")
                    # df = pd.read_excel(excel_path, sheet_name=0)

                    # # inds = df["Annotated Count"] > 10
                    # # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                    # sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    # annotated_counts = sub_df["Annotated Count"]
                    # predicted_counts = sub_df["Predicted Count"]
                    # count_diffs = (predicted_counts - annotated_counts).tolist()

                    # count_errors = []
                    # for annotated_count, predicted_count in zip(annotated_counts, predicted_counts):
                    #     if annotated_count > 0:
                    #         count_errors.append(abs(annotated_count - predicted_count) / annotated_count)



                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    abs_dids = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)


                            anno_count = annotations[image_name]["boxes"].shape[0]
                            pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                            height_px = metadata["images"][image_name]["height_px"]
                            width_px = metadata["images"][image_name]["width_px"]
                            area_px = height_px * width_px

                            gsd = inference_metrics.get_gsd(camera_specs, metadata)
                            area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                            print("area_m2", area_m2)

                            annotated_count_per_square_metre = anno_count / area_m2
                            predicted_count_per_square_metre = pred_count / area_m2

                            abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                            abs_dids.append(abs_did)


                    test_set_abs_did = np.mean(abs_dids)
                    test_set_abs_dids.append(test_set_abs_did)




                    # id_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")



                    # id_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    # id_abs_dics.extend(count_errors) #np.abs(count_diffs).tolist())
                    # id_global_test_set_accuracies.append(id_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(id_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(id_global_test_set_accuracy)

                # id_rep_abs_dic = np.mean(id_test_set_abs_dics)
                # id_rep_abs_dics.append(id_rep_abs_dic)

                # id_rep_abs_dics.append(np.mean(id_abs_dics))

                # id_global_rep_accuracy = np.mean(id_global_test_set_accuracies)
                # id_global_rep_accuracies.append(id_global_rep_accuracy)                


                id_rep_abs_did = np.mean(test_set_abs_dids)
                id_rep_abs_dids.append(id_rep_abs_did)








                # ood_test_set_accuracies = []
                ood_test_set_abs_dics = []
                # ood_global_test_set_accuracies = []
                ood_abs_dics = []

                for test_set in ood_test_sets:
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
                    
                    print("\t ood:", test_set_image_set_dir)
                    

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    # inds = df["Annotated Count"] > 10
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)



                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    abs_dids = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)


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


                    test_set_abs_did = np.mean(abs_dids)
                    test_set_abs_dids.append(test_set_abs_did)


                    # sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    # annotated_counts = sub_df["Annotated Count"]
                    # predicted_counts = sub_df["Predicted Count"]
                    # # count_diffs = (predicted_counts - annotated_counts).tolist()

                    # count_errors = []
                    # for annotated_count, predicted_count in zip(annotated_counts, predicted_counts):
                    #     if annotated_count > 0:
                    #         count_errors.append(abs(annotated_count - predicted_count) / annotated_count)



                    # # predictions_path = os.path.join(result_dir, "predictions.json")
                    # # predictions = annotation_utils.load_predictions(predictions_path)
                    # # annotations_path = os.path.join(result_dir, "annotations.json")
                    # # annotations = annotation_utils.load_annotations(annotations_path)
                    # # assessment_images = []
                    # # for image_name in annotations.keys():
                    # #     if len(annotations[image_name]["test_regions"]) > 0:
                    # #         assessment_images.append(image_name)

                    # # ood_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    # # ood_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    # ood_abs_dics.extend(count_errors) #np.abs(count_diffs).tolist())
                    # # ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)

                    # # if test_set_str not in image_based_test_set_results:
                    # #     image_based_test_set_results[test_set_str].append(ood_test_set_accuracy)
                    # # if test_set_str not in instance_based_test_set_results:
                    # #     instance_based_test_set_results[test_set_str].append(ood_global_test_set_accuracy)


                # ood_rep_abs_dic = np.mean(ood_test_set_abs_dics)
                # ood_rep_abs_dics.append(ood_rep_abs_dic)
                # ood_rep_abs_dics.append(np.mean(ood_abs_dics)) #ood_rep_abs_dic)



                ood_rep_abs_did = np.mean(test_set_abs_dids)
                ood_rep_abs_dids.append(ood_rep_abs_did)


                # ood_rep_accuracy = np.mean(ood_test_set_accuracies)
                # ood_rep_accuracies.append(ood_rep_accuracy)
                # # print(model_name, rep_accuracy)


                # ood_global_rep_accuracy = np.mean(ood_global_test_set_accuracies)
                # ood_global_rep_accuracies.append(ood_global_rep_accuracy)                



            id_baseline_abs_did = np.mean(id_rep_abs_dids) #None #np.mean(id_rep_abs_dics)
            id_baseline_stdev = np.std(id_rep_abs_dids) #None #np.std(id_rep_abs_dics)

            ood_baseline_abs_did = np.mean(ood_rep_abs_dids)
            ood_baseline_stdev = np.std(ood_rep_abs_dids)
            # ood_baseline_accuracy = np.mean(ood_rep_accuracies)
            # ood_baseline_stdev = np.std(ood_rep_accuracies)


            # id_global_baseline_accuracy = np.mean(id_global_rep_accuracies)
            # id_global_baseline_stdev = np.std(id_global_rep_accuracies)

            # ood_global_baseline_accuracy = np.mean(ood_global_rep_accuracies)
            # ood_global_baseline_stdev = np.std(ood_global_rep_accuracies)

            # print(ood_baseline_accuracy, ood_global_baseline_accuracy)


            results[k].append((patch_num, 
                               id_baseline_abs_did, 
                               id_baseline_stdev, 
                               ood_baseline_abs_did, 
                               ood_baseline_stdev,
                               ))

    # fig = plt.figure(figsize=(8, 6))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    xticks = [0, 40000]


    for i, k in enumerate(list(results.keys())):
        # axs[0].scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
        # axs[0].scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

        axs.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], label="In Domain")
        axs.plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
        
        axs.fill_between([x[0] for x in results[k]], [x[1] - x[2] for x in results[k]], [x[1] + x[2] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
        axs.fill_between([x[0] for x in results[k]], [x[3] - x[4] for x in results[k]], [x[3] + x[4] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

    axs.set_xlabel("Number of Training Patches")
    axs.set_ylabel("Mean Absolute Difference In Density") #Average Image Set Absolute Difference in Count")
    axs.set_ylim([0, None])
    axs.legend()

    plt.suptitle("Effect of Training Set Size")

    # plt.legend()
    plt.tight_layout()
    if remove_difficult_id_sets:
        out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_mean_mean_abs_did_no_difficult_id.svg")
    else:
        out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_mean_mean_abs_did.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)


def create_eval_size_plot_id_ood_abs_dic(id_test_sets, ood_test_sets, baseline_sets, out_dirname):


    results = {} #[]
    mappings = {}
    # test_set_types = [("ood", ood_test_sets), ("id", id_test_sets)]
    # labels = {}
    # test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):
        print(i)

        # if i == 0:
        #     label = "id"
        # else:
        #     label = "ood"

        # test_set_label = test_set_type[0]
        # test_sets = test_set_type[1]
        # labels[test_set_label] = []
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            # labels[test_set_label].append(test_set_str)
            # test_set_str_to_label[test_set_str] = label


    json_io.print_json(mappings)
    for k in baseline_sets.keys():
        results[k] = []
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]
            
            # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
            ood_rep_abs_dics = []
            # ood_global_rep_accuracies = []
            id_rep_abs_dics = []
            # id_global_rep_accuracies = []
            for rep_num in range(5):

                print(rep_num)






                id_test_set_abs_dics = []
                # id_global_test_set_accuracies = []
                id_abs_dics = []

                for test_set in id_test_sets:
                    
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    # if test_set_str in bad_id_test_sets:
                    #     continue
                    
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    print("\t id", test_set_image_set_dir)
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)

                    # inds = df["Annotated Count"] > 10
                    # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                    sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    annotated_counts = sub_df["Annotated Count"]
                    predicted_counts = sub_df["Predicted Count"]
                    count_diffs = (predicted_counts - annotated_counts).tolist()


                    # predictions_path = os.path.join(result_dir, "predictions.json")
                    # predictions = annotation_utils.load_predictions(predictions_path)
                    # annotations_path = os.path.join(result_dir, "annotations.json")
                    # annotations = annotation_utils.load_annotations(annotations_path)
                    # assessment_images = []
                    # for image_name in annotations.keys():
                    #     if len(annotations[image_name]["test_regions"]) > 0:
                    #         assessment_images.append(image_name)

                    # id_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")



                    # id_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    id_abs_dics.extend(np.abs(count_diffs).tolist())
                    # id_global_test_set_accuracies.append(id_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(id_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(id_global_test_set_accuracy)

                # id_rep_abs_dic = np.mean(id_test_set_abs_dics)
                # id_rep_abs_dics.append(id_rep_abs_dic)

                id_rep_abs_dics.append(np.mean(id_abs_dics))

                # id_global_rep_accuracy = np.mean(id_global_test_set_accuracies)
                # id_global_rep_accuracies.append(id_global_rep_accuracy)                










                # ood_test_set_accuracies = []
                ood_test_set_abs_dics = []
                # ood_global_test_set_accuracies = []
                ood_abs_dics = []

                for test_set in ood_test_sets:
                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    print("\t ood:", test_set_image_set_dir)
                    

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    # inds = df["Annotated Count"] > 10
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)


                    sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    annotated_counts = sub_df["Annotated Count"]
                    predicted_counts = sub_df["Predicted Count"]
                    count_diffs = (predicted_counts - annotated_counts).tolist()



                    # predictions_path = os.path.join(result_dir, "predictions.json")
                    # predictions = annotation_utils.load_predictions(predictions_path)
                    # annotations_path = os.path.join(result_dir, "annotations.json")
                    # annotations = annotation_utils.load_annotations(annotations_path)
                    # assessment_images = []
                    # for image_name in annotations.keys():
                    #     if len(annotations[image_name]["test_regions"]) > 0:
                    #         assessment_images.append(image_name)

                    # ood_global_test_set_accuracy = fine_tune_eval.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    # ood_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    ood_abs_dics.extend(np.abs(count_diffs).tolist())
                    # ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)

                    # if test_set_str not in image_based_test_set_results:
                    #     image_based_test_set_results[test_set_str].append(ood_test_set_accuracy)
                    # if test_set_str not in instance_based_test_set_results:
                    #     instance_based_test_set_results[test_set_str].append(ood_global_test_set_accuracy)


                # ood_rep_abs_dic = np.mean(ood_test_set_abs_dics)
                # ood_rep_abs_dics.append(ood_rep_abs_dic)
                ood_rep_abs_dics.append(np.mean(ood_abs_dics)) #ood_rep_abs_dic)


                # ood_rep_accuracy = np.mean(ood_test_set_accuracies)
                # ood_rep_accuracies.append(ood_rep_accuracy)
                # # print(model_name, rep_accuracy)


                # ood_global_rep_accuracy = np.mean(ood_global_test_set_accuracies)
                # ood_global_rep_accuracies.append(ood_global_rep_accuracy)                



            id_baseline_abs_dic = np.mean(id_rep_abs_dics) #None #np.mean(id_rep_abs_dics)
            id_baseline_stdev = np.std(id_rep_abs_dics) #None #np.std(id_rep_abs_dics)

            ood_baseline_abs_dic = np.mean(ood_rep_abs_dics)
            ood_baseline_stdev = np.std(ood_rep_abs_dics)
            # ood_baseline_accuracy = np.mean(ood_rep_accuracies)
            # ood_baseline_stdev = np.std(ood_rep_accuracies)


            # id_global_baseline_accuracy = np.mean(id_global_rep_accuracies)
            # id_global_baseline_stdev = np.std(id_global_rep_accuracies)

            # ood_global_baseline_accuracy = np.mean(ood_global_rep_accuracies)
            # ood_global_baseline_stdev = np.std(ood_global_rep_accuracies)

            # print(ood_baseline_accuracy, ood_global_baseline_accuracy)


            results[k].append((patch_num, 
                               id_baseline_abs_dic, 
                               id_baseline_stdev, 
                               ood_baseline_abs_dic, 
                               ood_baseline_stdev,
                               ))

    # fig = plt.figure(figsize=(8, 6))

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    xticks = [0, 40000]


    for i, k in enumerate(list(results.keys())):
        # axs[0].scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
        # axs[0].scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
        # for x in results[k]:
        #     axs[0].plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

        axs.plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], label="In Domain")
        axs.plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
        
        axs.fill_between([x[0] for x in results[k]], [x[1] - x[2] for x in results[k]], [x[1] + x[2] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
        axs.fill_between([x[0] for x in results[k]], [x[3] - x[4] for x in results[k]], [x[3] + x[4] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

    axs.set_xlabel("Number of Training Patches")
    axs.set_ylabel("Mean Absolute Difference In Count") #Average Image Set Absolute Difference in Count")
    axs.set_ylim([0, None])
    axs.legend()

    plt.suptitle("Effect of Training Set Size")

    # plt.legend()
    plt.tight_layout()

    out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_abs_dic.svg")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)



def create_eval_size_plot_id_ood(id_test_sets, ood_test_sets, baseline_sets, out_dirname, remove_difficult_id_sets=False, instance_only=False):


    results = {} #[]
    mappings = {}
    # test_set_types = [("ood", ood_test_sets), ("id", id_test_sets)]
    # labels = {}
    # test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):

        # if i == 0:
        #     label = "id"
        # else:
        #     label = "ood"

        # test_set_label = test_set_type[0]
        # test_sets = test_set_type[1]
        # labels[test_set_label] = []
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            # labels[test_set_label].append(test_set_str)
            # test_set_str_to_label[test_set_str] = label

    # image_based_test_set_results = {}
    # instance_based_test_set_results = {}






    json_io.print_json(mappings)

    # if 
    if remove_difficult_id_sets:
        results_path = os.path.join("eval_charts", out_dirname, "id_ood_size_results_no_difficult_id.json")
    else:
        results_path = os.path.join("eval_charts", out_dirname, "id_ood_size_results.json")

    if False: #os.path.exists(results_path):
        results = json_io.load_json(results_path)
    else:

        print("\n\ncalculating...\n\n")


        for k in baseline_sets.keys():
            results[k] = []
            baselines = baseline_sets[k]
            for baseline in baselines:
                model_name = baseline["model_name"]
                patch_num = baseline["patch_num"]
                
                # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
                ood_rep_accuracies = []
                ood_global_rep_accuracies = []
                id_rep_accuracies = []
                id_global_rep_accuracies = []
                for rep_num in range(0, 2):

                    id_test_set_accuracies = []
                    id_global_test_set_accuracies = []

                    for test_set in id_test_sets:
                        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                        if test_set_str in bad_id_test_sets and remove_difficult_id_sets:
                            continue
                        
                        test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
                        

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        print(model_name)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)

                        # inds = df["Annotated Count"] > 10
                        # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                        # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].loc[df["Accuracy (IoU=.50, conf>.50)"].first_valid_index()]
                        # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].loc[~df["Accuracy (IoU=.50, conf>.50)"].isnull()].iloc[1]



                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        assessment_images = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                assessment_images.append(image_name)

                        # assessment_images = assessment_images[:1]
                        # assessment_images = assessment_images[1:2]
                        # print(assessment_images)
                        id_global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")


                        # full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                        # full_predictions = annotation_utils.load_predictions(full_predictions_path)
                        # test_set_mAP_05 = float(inference_metrics.get_mAP_val(annotations, predictions, ".50", assessment_images))

                        # id_test_set_accuracy = test_set_mAP_05


                        id_test_set_accuracies.append(id_test_set_accuracy)
                        id_global_test_set_accuracies.append(id_global_test_set_accuracy)

                        # if test_set_str not in image_based_test_set_results:
                        #     image_based_test_set_results[test_set_str].append(id_test_set_accuracy)
                        # if test_set_str not in instance_based_test_set_results:
                        #     instance_based_test_set_results[test_set_str].append(id_global_test_set_accuracy)

                    id_rep_accuracy = np.mean(id_test_set_accuracies)
                    id_rep_accuracies.append(id_rep_accuracy)

                    id_global_rep_accuracy = np.mean(id_global_test_set_accuracies)
                    id_global_rep_accuracies.append(id_global_rep_accuracy)                


                    ood_test_set_accuracies = []
                    ood_global_test_set_accuracies = []

                    for test_set in ood_test_sets:
                        test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                        test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
                        

                        model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                        # print(model_name)
                        model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                        result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                        excel_path = os.path.join(result_dir, "metrics.xlsx")
                        df = pd.read_excel(excel_path, sheet_name=0)
                        # inds = df["Annotated Count"] > 10
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                        ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].loc[df["Accuracy (IoU=.50, conf>.50)"].first_valid_index()]
                        # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].loc[~df["Accuracy (IoU=.50, conf>.50)"].isnull()].iloc[1]


                        predictions_path = os.path.join(result_dir, "predictions.json")
                        predictions = annotation_utils.load_predictions(predictions_path)
                        annotations_path = os.path.join(result_dir, "annotations.json")
                        annotations = annotation_utils.load_annotations(annotations_path)
                        assessment_images = []
                        for image_name in annotations.keys():
                            if len(annotations[image_name]["test_regions"]) > 0:
                                assessment_images.append(image_name)

                        # import random
                        # assessment_images = random.sample(assessment_images, 2)


                        # full_predictions_path = os.path.join(result_dir, "full_predictions.json")
                        # full_predictions = annotation_utils.load_predictions(full_predictions_path)
                        # test_set_mAP_05 = float(inference_metrics.get_mAP_val(annotations, predictions, ".50", assessment_images))

                        # ood_test_set_accuracy = test_set_mAP_05

                        # assessment_images = assessment_images[0:2] #1:6]
                        assessment_images = random.sample(assessment_images, 2)
                        # print(assessment_images)
                        ood_global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                        ood_test_set_accuracies.append(ood_test_set_accuracy)
                        ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)

                        # if test_set_str not in image_based_test_set_results:
                        #     image_based_test_set_results[test_set_str].append(ood_test_set_accuracy)
                        # if test_set_str not in instance_based_test_set_results:
                        #     instance_based_test_set_results[test_set_str].append(ood_global_test_set_accuracy)



                    ood_rep_accuracy = np.mean(ood_test_set_accuracies)
                    ood_rep_accuracies.append(ood_rep_accuracy)
                    # print(model_name, rep_accuracy)


                    ood_global_rep_accuracy = np.mean(ood_global_test_set_accuracies)
                    ood_global_rep_accuracies.append(ood_global_rep_accuracy)                



                id_baseline_accuracy = float(np.mean(id_rep_accuracies))
                id_baseline_stdev = float(np.std(id_rep_accuracies))

                ood_baseline_accuracy = float(np.mean(ood_rep_accuracies))
                ood_baseline_stdev = float(np.std(ood_rep_accuracies))


                id_global_baseline_accuracy = float(np.mean(id_global_rep_accuracies))
                id_global_baseline_stdev = float(np.std(id_global_rep_accuracies))

                ood_global_baseline_accuracy = float(np.mean(ood_global_rep_accuracies))
                ood_global_baseline_stdev = float(np.std(ood_global_rep_accuracies))

                # print(ood_baseline_accuracy, ood_global_baseline_accuracy)


                results[k].append((patch_num, 
                                id_baseline_accuracy, 
                                id_baseline_stdev, 
                                ood_baseline_accuracy, 
                                ood_baseline_stdev,
                                id_global_baseline_accuracy,
                                id_global_baseline_stdev,
                                ood_global_baseline_accuracy,
                                ood_global_baseline_stdev
                                ))
                
        json_io.save_json(results_path, results)

    # fig = plt.figure(figsize=(8, 6))

    xticks = [0, 10000, 20000, 30000, 40000]

    if instance_only:
        fig, axs = plt.subplots(1, 1, figsize=(5.5, 4))
        # fig.legend(loc=(0.778, 0.94)) #0.95))

        for i, k in enumerate(list(results.keys())):
            # plt.scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
            # plt.scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

            # for x in results[k]:
            #     plt.plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
            # for x in results[k]:
            #     plt.plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

            axs.plot([x[0] for x in results[k]], [x[5] for x in results[k]], color=my_plot_colors[0], label="In Domain")
            axs.plot([x[0] for x in results[k]], [x[7] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
            
            axs.fill_between([x[0] for x in results[k]], [x[5] - x[6] for x in results[k]], [x[5] + x[6] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
            axs.fill_between([x[0] for x in results[k]], [x[7] - x[8] for x in results[k]], [x[7] + x[8] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

        axs.set_xlabel("Number of Training Patches")
        axs.set_ylabel("Instance-Based Accuracy")
        axs.set_ylim([0.3, 0.9]) #0.45, 0.8])
        axs.set_xticks(xticks)

        axs.legend(loc="lower right") #loc=(0.778, 0.94))

    else:

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))



        # results_path = json_io.save_json(results_path, results)


        for i, k in enumerate(list(results.keys())):
            # axs[0].scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
            # axs[0].scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

            # for x in results[k]:
            #     axs[0].plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
            # for x in results[k]:
            #     axs[0].plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

            axs[0].plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], label="In Domain")
            axs[0].plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
            
            axs[0].fill_between([x[0] for x in results[k]], [x[1] - x[2] for x in results[k]], [x[1] + x[2] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
            axs[0].fill_between([x[0] for x in results[k]], [x[3] - x[4] for x in results[k]], [x[3] + x[4] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

        axs[0].set_xlabel("Number of Training Patches")
        axs[0].set_ylabel("Image-Based Accuracy") #mAP@0.5") #"Image-Based Accuracy")
        # axs[0].set_ylim([0.45, 0.8])
        # axs[0].set_xscale("log")
        # axs[0].set_xticks(xticks)
        # axs[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        # axs[0].set_xticklabels(xticklabels)
        axs[0].set_xticks(xticks) #np.arange(len(xticks))) #[])
        # axs[0].set_xticklabels(xticks)


        # axs[0].
        fig.legend(loc=(0.815, 0.85)) #0.95))

        for i, k in enumerate(list(results.keys())):
            # plt.scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
            # plt.scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

            # for x in results[k]:
            #     plt.plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
            # for x in results[k]:
            #     plt.plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

            axs[1].plot([x[0] for x in results[k]], [x[5] for x in results[k]], color=my_plot_colors[0], label="In Domain")
            axs[1].plot([x[0] for x in results[k]], [x[7] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
            
            axs[1].fill_between([x[0] for x in results[k]], [x[5] - x[6] for x in results[k]], [x[5] + x[6] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
            axs[1].fill_between([x[0] for x in results[k]], [x[7] - x[8] for x in results[k]], [x[7] + x[8] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

        axs[1].set_xlabel("Number of Training Patches")
        axs[1].set_ylabel("Instance-Based Accuracy")
        # axs[1].set_ylim([0.45, 0.8])
        # axs[1].set_xscale("log")
        # axs[1].set_xticks(xticks)
        # axs[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        # plt.ylim([0.48, 0.87])
        # axs[1].legend()
        axs[1].set_xticks(xticks) #np.arange(len(xticks))) #[])
        # axs[1].set_xticklabels(xticks)


    plt.tight_layout()
    plt.subplots_adjust(top=0.84)
    if remove_difficult_id_sets:
        plt.suptitle("Effect of Training Set Size on Instance-Based Accuracy\n(Removed the Four Most Challenging In-Domain Test Sets)")
    else:
        plt.suptitle("Effect of Training Set Size on Test Performance")


    # plt.legend()


    if remove_difficult_id_sets:
        out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_no_difficult_id.png") #svg") #png")
    else:
        out_path = os.path.join("eval_charts", out_dirname, "rep_0_and_1") #"ns_id_and_ood.svg") #id_ood_training_set_size.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)






def compare_epochs_and_steps(id_test_sets, ood_test_sets, baseline_sets, out_dirname):


    pass



def create_final_combined_plot(id_test_sets, ood_test_sets, baseline_sets, out_dirname):


    results = {} #[]
    mappings = {}
    # test_set_types = [("ood", ood_test_sets), ("id", id_test_sets)]
    # labels = {}
    # test_set_str_to_label = {}
    for i, test_set_type in enumerate([id_test_sets, ood_test_sets]):
        print(i)

        # if i == 0:
        #     label = "id"
        # else:
        #     label = "ood"

        # test_set_label = test_set_type[0]
        # test_sets = test_set_type[1]
        # labels[test_set_label] = []
        for test_set in test_set_type:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])

            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)
            # labels[test_set_label].append(test_set_str)
            # test_set_str_to_label[test_set_str] = label


    json_io.print_json(mappings)
    for k in baseline_sets.keys():
        results[k] = []
        baselines = baseline_sets[k]
        for baseline in baselines:
            model_name = baseline["model_name"]
            patch_num = baseline["patch_num"]
            
            # patch_num = int((baseline["model_name"][len("set_of_27_"):]).split("_")[0])
            ood_rep_abs_dids = []
            ood_global_rep_accuracies = []
            id_rep_abs_dids = []
            id_global_rep_accuracies = []
            for rep_num in range(1):

                # if model_name == "set_of_27_250_patches":
                #     rep_num = 4

                print(rep_num)






                id_test_set_abs_dics = []
                id_global_test_set_accuracies = []
                id_abs_dics = []
                id_test_set_abs_dids = []

                for test_set in id_test_sets:

                    test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
                    # if test_set_str in bad_id_test_sets:
                    #     continue
                    
                    test_set_image_set_dir = os.path.join("usr", "data",
                                                        test_set["username"], "image_sets",
                                                        test_set["farm_name"],
                                                        test_set["field_name"],
                                                        test_set["mission_date"])
                    
                    metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
                    metadata = json_io.load_json(metadata_path)
                    camera_specs_path = os.path.join("usr", "data", test_set["username"], "cameras", "cameras.json")
                    camera_specs = json_io.load_json(camera_specs_path)
                    
                    print("\t id", test_set_image_set_dir)
                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    # excel_path = os.path.join(result_dir, "metrics.xlsx")
                    # df = pd.read_excel(excel_path, sheet_name=0)

                    # # inds = df["Annotated Count"] > 10
                    # # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # # id_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)

                    # sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    # annotated_counts = sub_df["Annotated Count"]
                    # predicted_counts = sub_df["Predicted Count"]
                    # count_diffs = (predicted_counts - annotated_counts).tolist()

                    # count_errors = []
                    # for annotated_count, predicted_count in zip(annotated_counts, predicted_counts):
                    #     if annotated_count > 0:
                    #         count_errors.append(abs(annotated_count - predicted_count) / annotated_count)



                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    abs_dids = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)


                            anno_count = annotations[image_name]["boxes"].shape[0]
                            pred_count = (predictions[image_name]["boxes"][predictions[image_name]["scores"] > 0.5]).shape[0]

                            height_px = metadata["images"][image_name]["height_px"]
                            width_px = metadata["images"][image_name]["width_px"]
                            area_px = height_px * width_px

                            gsd = inference_metrics.get_gsd(camera_specs, metadata)
                            area_m2 = inference_metrics.calculate_area_m2(gsd, area_px)

                            print("area_m2", area_m2)

                            annotated_count_per_square_metre = anno_count / area_m2
                            predicted_count_per_square_metre = pred_count / area_m2

                            abs_did = abs(annotated_count_per_square_metre - predicted_count_per_square_metre)
                            abs_dids.append(abs_did)


                    test_set_abs_did = np.mean(abs_dids)
                    id_test_set_abs_dids.append(test_set_abs_did)




                    id_global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")
                    id_global_test_set_accuracies.append(id_global_test_set_accuracy)



                id_global_rep_accuracy = np.mean(id_global_test_set_accuracies)
                id_global_rep_accuracies.append(id_global_rep_accuracy)                


                id_rep_abs_did = np.mean(id_test_set_abs_dids)
                id_rep_abs_dids.append(id_rep_abs_did)








                # ood_test_set_accuracies = []
                ood_test_set_abs_dics = []
                ood_global_test_set_accuracies = []
                ood_abs_dics = []
                ood_test_set_abs_dids = []

                for test_set in ood_test_sets:
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
                    
                    print("\t ood:", test_set_image_set_dir)
                    

                    model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    # print(model_name)
                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    excel_path = os.path.join(result_dir, "metrics.xlsx")
                    df = pd.read_excel(excel_path, sheet_name=0)
                    # inds = df["Annotated Count"] > 10
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"][inds].mean(skipna=True)
                    # ood_test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)



                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    abs_dids = []
                    for image_name in annotations.keys():
                        if len(annotations[image_name]["test_regions"]) > 0:
                            assessment_images.append(image_name)


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


                    test_set_abs_did = np.mean(abs_dids)
                    ood_test_set_abs_dids.append(test_set_abs_did)


                    # sub_df = df[df["Image Is Fully Annotated"] == "yes: for testing"]
                    # annotated_counts = sub_df["Annotated Count"]
                    # predicted_counts = sub_df["Predicted Count"]
                    # # count_diffs = (predicted_counts - annotated_counts).tolist()

                    # count_errors = []
                    # for annotated_count, predicted_count in zip(annotated_counts, predicted_counts):
                    #     if annotated_count > 0:
                    #         count_errors.append(abs(annotated_count - predicted_count) / annotated_count)



                    # # predictions_path = os.path.join(result_dir, "predictions.json")
                    # # predictions = annotation_utils.load_predictions(predictions_path)
                    # # annotations_path = os.path.join(result_dir, "annotations.json")
                    # # annotations = annotation_utils.load_annotations(annotations_path)
                    # # assessment_images = []
                    # # for image_name in annotations.keys():
                    # #     if len(annotations[image_name]["test_regions"]) > 0:
                    # #         assessment_images.append(image_name)

                    ood_global_test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images) # fine_tune_eval.get_AP(annotations, full_predictions, iou_thresh=".50:.05:.95")

                    # # ood_test_set_abs_dics.append(np.mean(np.abs(count_diffs)))

                    # ood_abs_dics.extend(count_errors) #np.abs(count_diffs).tolist())
                    ood_global_test_set_accuracies.append(ood_global_test_set_accuracy)

                    # # if test_set_str not in image_based_test_set_results:
                    # #     image_based_test_set_results[test_set_str].append(ood_test_set_accuracy)
                    # # if test_set_str not in instance_based_test_set_results:
                    # #     instance_based_test_set_results[test_set_str].append(ood_global_test_set_accuracy)


                # ood_rep_abs_dic = np.mean(ood_test_set_abs_dics)
                # ood_rep_abs_dics.append(ood_rep_abs_dic)
                # ood_rep_abs_dics.append(np.mean(ood_abs_dics)) #ood_rep_abs_dic)



                ood_rep_abs_did = np.mean(ood_test_set_abs_dids)
                ood_rep_abs_dids.append(ood_rep_abs_did)


                # ood_rep_accuracy = np.mean(ood_test_set_accuracies)
                # ood_rep_accuracies.append(ood_rep_accuracy)
                # # print(model_name, rep_accuracy)


                ood_global_rep_accuracy = np.mean(ood_global_test_set_accuracies)
                ood_global_rep_accuracies.append(ood_global_rep_accuracy)                



            id_baseline_abs_did = float(np.mean(id_rep_abs_dids)) #None #np.mean(id_rep_abs_dics)
            id_baseline_stdev = float(np.std(id_rep_abs_dids)) #None #np.std(id_rep_abs_dics)

            ood_baseline_abs_did = float(np.mean(ood_rep_abs_dids))
            ood_baseline_stdev = float(np.std(ood_rep_abs_dids))
            # ood_baseline_accuracy = np.mean(ood_rep_accuracies)
            # ood_baseline_stdev = np.std(ood_rep_accuracies)


            id_global_baseline_accuracy = float(np.mean(id_global_rep_accuracies))
            id_global_baseline_stdev = float(np.std(id_global_rep_accuracies))

            ood_global_baseline_accuracy = float(np.mean(ood_global_rep_accuracies))
            ood_global_baseline_stdev = float(np.std(ood_global_rep_accuracies))

            # print(ood_baseline_accuracy, ood_global_baseline_accuracy)


            results[k].append((patch_num, 
                               id_global_baseline_accuracy,
                               id_global_baseline_stdev,
                               ood_global_baseline_accuracy,
                               ood_global_baseline_stdev,
                               id_baseline_abs_did, 
                               id_baseline_stdev, 
                               ood_baseline_abs_did, 
                               ood_baseline_stdev
                               ))

    out_path = os.path.join("eval_charts", out_dirname, "SAVE_results.json")
    json_io.save_json(out_path, results)


    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    xticks = [0, 10000, 20000, 30000, 40000]

    for i, k in enumerate(list(results.keys())):

        axs[0].plot([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], label="In Domain")
        axs[0].plot([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
        
        axs[0].fill_between([x[0] for x in results[k]], [x[1] - x[2] for x in results[k]], [x[1] + x[2] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
        axs[0].fill_between([x[0] for x in results[k]], [x[3] - x[4] for x in results[k]], [x[3] + x[4] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

        axs[0].set_xlabel("Number of Training Patches")
        axs[0].set_ylabel("Instance-Based Accuracy")
        axs[0].set_ylim([0.45, 0.8])
        # axs[0].set_xscale("log")
        # axs[0].set_xticks(xticks)
        # axs[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        # axs[0].set_xticklabels(xticklabels)
        axs[0].set_xticks(xticks) #np.arange(len(xticks))) #[])
        # axs[0].set_xticklabels(xticks)


        # axs[0].
        fig.legend(loc=(0.815, 0.85)) #0.95))

        for i, k in enumerate(list(results.keys())):
            # plt.scatter([x[0] for x in results[k]], [x[1] for x in results[k]], color=my_plot_colors[0], marker="_", label="In Domain", zorder=2)
            # plt.scatter([x[0] for x in results[k]], [x[3] for x in results[k]], color=my_plot_colors[1], marker="_", label="Out Of Domain", zorder=2)

            # for x in results[k]:
            #     plt.plot([x[0], x[0]], [x[1] + x[2], x[1] - x[2]], color=my_plot_colors[0], linestyle="solid", linewidth=1, zorder=2)
            # for x in results[k]:
            #     plt.plot([x[0], x[0]], [x[3] + x[4], x[3] - x[4]], color=my_plot_colors[1], linestyle="solid", linewidth=1, zorder=2)

            axs[1].plot([x[0] for x in results[k]], [x[5] for x in results[k]], color=my_plot_colors[0], label="In Domain")
            axs[1].plot([x[0] for x in results[k]], [x[7] for x in results[k]], color=my_plot_colors[1], label="Out Of Domain")
            
            axs[1].fill_between([x[0] for x in results[k]], [x[5] - x[6] for x in results[k]], [x[5] + x[6] for x in results[k]], edgecolor=my_plot_colors[0], color=my_plot_colors[0], linewidth=1, facecolor=my_plot_colors[0], alpha=0.15)
            axs[1].fill_between([x[0] for x in results[k]], [x[7] - x[8] for x in results[k]], [x[7] + x[8] for x in results[k]], edgecolor=my_plot_colors[1], color=my_plot_colors[1], linewidth=1, facecolor=my_plot_colors[1], alpha=0.15)

        axs[1].set_xlabel("Number of Training Patches")
        axs[1].set_ylabel("Mean Absolute Difference in Density")
        # axs[1].set_ylim([0.45, 0.8])
        # axs[1].set_xscale("log")
        # axs[1].set_xticks(xticks)
        # axs[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        # plt.ylim([0.48, 0.87])
        # axs[1].legend()
        axs[1].set_ylim(bottom=0)
        axs[1].set_xticks(xticks) #np.arange(len(xticks))) #[])
        # axs[1].set_xticklabels(xticks)


    plt.tight_layout()
    plt.subplots_adjust(top=0.84)



    
    plt.suptitle("Effect of Training Set Size on Model Performance")


    # plt.legend()


    # if remove_difficult_id_sets:
    #     out_path = os.path.join("eval_charts", out_dirname, "id_ood_training_set_size_no_difficult_id.svg") #png")
    # else:
    out_path = os.path.join("eval_charts", out_dirname, "training_set_size_combined_plot.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=600)
















def my_size_plot():
    baseline_sets = {}

    baseline_sets["diverse"] = []
    for b in eval_fixed_patch_num_baselines:
        baseline_sets["diverse"].append({
            "model_name": b,
            "patch_num": int((b[len("set_of_27_"):]).split("_")[0])
        })


    # print("hello world")
    # create_eval_size_plot_id_ood(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size") #, remove_difficult_id_sets=True, instance_only=True)
    ns_baseline_sets = {}
    ns_baseline_sets["diverse"] = []
    for b in ns_fixed_patch_num_baselines:
        ns_baseline_sets["diverse"].append({
            "model_name": b,
            "patch_num": int((b[len("ns_set_of_27_"):]).split("_")[0])
        }) 



    # create_eval_size_plot_id_ood(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size", instance_only=True)
    # create_eval_size_plot_id_ood(ns_in_domain_test_sets, ns_test_sets, ns_baseline_sets, "id_ood_size")
    # create_eval_size_plot_id_ood(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size", remove_difficult_id_sets=True, instance_only=True)
    # create_eval_size_plot_id_ood_abs_dic(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size")
    # create_eval_size_plot_id_ood_percent_count_error(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size")
    # create_individual_image_sets_eval_size_plot_id_ood([], eval_test_sets, baseline_sets, "id_ood_size")
    # create_individual_image_sets_eval_size_plot_id_ood_abs_did([], eval_test_sets, baseline_sets, "id_ood_size")
    # create_individual_image_sets_eval_size_plot_id_ood(ns_in_domain_test_sets, ns_test_sets, ns_baseline_sets, "id_ood_size")
    # create_eval_size_plot_id_ood_abs_did(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size", remove_difficult_id_sets=True)
    # create_individual_images_eval_size_plot_id_ood([], eval_test_sets, baseline_sets, "id_ood_size")

    baseline_sets["best_to_worst"] = []
    for b in eval_best_to_worst_baselines:
        baseline_sets["best_to_worst"].append({
            "model_name": b,
            "patch_num": int((b[len("set_of_27_best_to_worst_"):]).split("_")[0])
        })
        

    create_eval_size_plot_id_ood([], eval_test_sets, baseline_sets, "best_to_worst_comparison", remove_difficult_id_sets=False, instance_only=True)

    # baseline_sets["diverse"].append({
    #     "model_name": "set_of_27_250_patches",
    #     "patch_num": 250
    # })

    # baseline_sets["diverse"].append({
    #     "model_name": "set_of_27_250_patches_match_steps",
    #     "patch_num": 5000
    # })



    # create_final_combined_plot(eval_in_domain_test_sets, eval_test_sets, baseline_sets, "id_ood_size_match_steps")


if __name__ == "__main__":
    my_size_plot()