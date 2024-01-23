import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

from io_utils import json_io
from models.common import annotation_utils, inference_metrics
from exp_runner import get_mapping_for_test_set, my_plot_colors
import ds_splits



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, vmin=0, vmax=1, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def continuum_plot(model1, model2, sets1, sets2, res_name, out_name):

    mappings = {}

    for sets in [sets1, sets2]:
        for test_set in sets["sets"]:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)

    all_sets = sets1["sets"] + sets2["sets"]

    results = {}

    for baseline in [model1, model2]:


        rep_accuracies = []
        rep_mean_abs_dics = []
        rep_mean_abs_dids = []


        # split_label = sets["split_label"]
        model_label = baseline["model_label"]


        for test_set in all_sets:
        

            # for test_set in sets["sets"]:

            for rep_num in range(1):

                model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                test_set_accuracies = []
                test_set_mean_abs_dics = []
                test_set_mean_abs_dids = []
                abs_dics = []
                # abs_dids = []
            

                # for test_set in sets["sets"]:
                # print(test_set)
                # abs_dics = []
                abs_dids = []
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


                model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                # rep_accuracies = []
                # for rep_num in range(3):
                # model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                # excel_path = os.path.join(result_dir, "metrics.xlsx")
                # df = pd.read_excel(excel_path, sheet_name=0)
                # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                # rep_accuracies.append(rep_accuracy)


                predictions_path = os.path.join(result_dir, "predictions.json")
                predictions = annotation_utils.load_predictions(predictions_path)
                annotations_path = os.path.join(result_dir, "annotations.json")
                annotations = annotation_utils.load_annotations(annotations_path)
                assessment_images = []
                image_accuracies = []
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

                        image_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, [image_name])
                        image_accuracies.append(image_accuracy)
                        
                test_set_accuracy = np.mean(image_accuracies) 


                test_set_str = test_set["farm_name"] + test_set["field_name"] + test_set["mission_date"]
                print(test_set_str)
                if model_label not in results:
                    results[model_label] = {}

                if test_set_str not in results[model_label]:
                    results[model_label][test_set_str] = []



                results[model_label][test_set_str].append(test_set_accuracy)



                # test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images)
                # test_set_accuracies.append(test_set_accuracy)

                # mean_abs_did = np.mean(abs_dids)
                # test_set_mean_abs_dids.append(mean_abs_did)

            # split_label = sets["split_label"]
            # model_label = baseline["model_label"]
            # print(split_label, model_label, abs_dics)

            # rep_mean_abs_dic = np.mean(abs_dics)
            # rep_mean_abs_dics.append(rep_mean_abs_dic)

            # rep_mean_abs_did = np.mean(test_set_mean_abs_dids)
            # rep_mean_abs_dids.append(rep_mean_abs_did)


            # rep_accuracy = np.mean(test_set_accuracies)
            # rep_accuracies.append(rep_accuracy)

            # if i == 1:
            # print(baseline["model_name"], rep_accuracies)




            # test_set_mean_abs_dic = float(np.mean(rep_mean_abs_dics))
            # test_set_mean_abs_dic_std = float(np.std(rep_mean_abs_dics))

            # test_set_mean_abs_did = float(np.mean(rep_mean_abs_dids))
            # test_set_mean_abs_did_std = float(np.std(rep_mean_abs_dids))

            # test_set_accuracy = float(np.mean(rep_accuracies))
            # test_set_accuracy_std = float(np.std(rep_accuracies))

            # results[test_set_str][model_label] = [
            #     rep_accuracy, 
            #     # rep_accuracy_std, 
            #     rep_mean_abs_dic,
            #     # rep_mean_abs_dic_std,
            #     rep_mean_abs_did,
            #     # rep_mean_abs_did_std                
            # ]
    
    json_io.print_json(results)

    # exit()

    model1_results = []
    model2_results = []
    for test_set in all_sets:
        test_set_str = test_set["farm_name"] + test_set["field_name"] + test_set["mission_date"]
        model1_results.append(results[model1["model_label"]][test_set_str])
        model2_results.append(results[model2["model_label"]][test_set_str])

    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(model1_results)), model1_results, my_plot_colors[0], label=model1["model_label"])
    ax.plot(np.arange(len(model2_results)), model2_results, my_plot_colors[1], label=model2["model_label"])

    plt.legend()

    ax.set_ylim(bottom=0, top=1)

    out_dir = os.path.join("eval_charts", "ds_splits")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, out_name + ".svg"))


def split_heatmap(model1, model2, sets1, sets2, res_name, out_name):

    mappings = {}

    for sets in [sets1, sets2]:
        for test_set in sets["sets"]:
            test_set_str = test_set["username"] + " " + test_set["farm_name"] + " " + test_set["field_name"] + " " + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
            mappings[test_set_str] = get_mapping_for_test_set(test_set_image_set_dir)


    results = {}

    for baseline in [model1, model2]:


        rep_accuracies = []
        rep_mean_abs_dics = []
        rep_mean_abs_dids = []

        for sets in [sets1, sets2]:

            split_label = sets["split_label"]
            model_label = baseline["model_label"]

            for rep_num in range(1):

                model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                
                test_set_accuracies = []
                test_set_mean_abs_dics = []
                test_set_mean_abs_dids = []
                abs_dics = []
                # abs_dids = []
                

                for test_set in sets["sets"]:
                    # print(test_set)
                    # abs_dics = []
                    abs_dids = []
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


                    model_dir = os.path.join(test_set_image_set_dir, "model", "results")

                    # rep_accuracies = []
                    # for rep_num in range(3):
                    # model_name = baseline["model_name"] + "_rep_" + str(rep_num)
                    
                    result_dir = os.path.join(model_dir, mappings[test_set_str][model_name])
                    # excel_path = os.path.join(result_dir, "metrics.xlsx")
                    # df = pd.read_excel(excel_path, sheet_name=0)
                    # test_set_accuracy = df["Accuracy (IoU=.50, conf>.50)"].mean(skipna=True)
                    # rep_accuracies.append(rep_accuracy)


                    predictions_path = os.path.join(result_dir, "predictions.json")
                    predictions = annotation_utils.load_predictions(predictions_path)
                    annotations_path = os.path.join(result_dir, "annotations.json")
                    annotations = annotation_utils.load_annotations(annotations_path)
                    assessment_images = []
                    image_accuracies = []
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

                            image_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, [image_name])
                            image_accuracies.append(image_accuracy)
                            
                    test_set_accuracy = np.mean(image_accuracies) 

                    # test_set_accuracy = inference_metrics.get_global_accuracy(annotations, predictions, assessment_images)
                    test_set_accuracies.append(test_set_accuracy)

                    mean_abs_did = np.mean(abs_dids)
                    test_set_mean_abs_dids.append(mean_abs_did)

                # split_label = sets["split_label"]
                # model_label = baseline["model_label"]
                print(split_label, model_label, abs_dics)

                rep_mean_abs_dic = np.mean(abs_dics)
                rep_mean_abs_dics.append(rep_mean_abs_dic)

                rep_mean_abs_did = np.mean(test_set_mean_abs_dids)
                rep_mean_abs_dids.append(rep_mean_abs_did)


                rep_accuracy = np.mean(test_set_accuracies)
                rep_accuracies.append(rep_accuracy)

            # if i == 1:
            # print(baseline["model_name"], rep_accuracies)




            baseline_mean_abs_dic = float(np.mean(rep_mean_abs_dics))
            baseline_mean_abs_dic_std = float(np.std(rep_mean_abs_dics))

            baseline_mean_abs_did = float(np.mean(rep_mean_abs_dids))
            baseline_mean_abs_did_std = float(np.std(rep_mean_abs_dids))


            baseline_accuracy = float(np.mean(rep_accuracies))
            baseline_accuracy_std = float(np.std(rep_accuracies))
        

            if split_label not in results:
                results[split_label] = {}

            results[split_label][model_label] = [
                baseline_accuracy, 
                baseline_accuracy_std, 
                baseline_mean_abs_dic,
                baseline_mean_abs_dic_std,
                baseline_mean_abs_did,
                baseline_mean_abs_did_std
            ]



            # results[result_key].append(
            #     (baseline["model_label"], 
            #     baseline_accuracy, 
            #     baseline_accuracy_std, 
            #     baseline_mean_abs_dic,
            #     baseline_mean_abs_dic_std,
            #     baseline_mean_abs_did,
            #     baseline_mean_abs_did_std
            #     ))



    for split_label in results.keys():
        print()
        print()
        print("-----")
        print(split_label)
        print()

        for model_label in results[split_label].keys():
            acc = results[split_label][model_label][0]

            print(model_label, acc)

    # arr = np.array([[0, 0], [0, 0]])
    print()
    print()
    print()
    arr = []
    for i, split in enumerate([sets1, sets2]):
        arr.append([])
        for k, model in enumerate([model1, model2]):
            split_label = split["split_label"]
            model_label = model["model_label"]
            r = results[split_label][model_label][0]
            print(i, k, r)
            # arr[i][k] = r
            # print(arr)
            arr[i].append(round(r, 2))

    arr = np.array(arr)
    print(arr)

    split_labels = [sets1["split_label"], sets2["split_label"]]
    model_labels = [model1["model_label"], model2["model_label"]]

    fig, ax = plt.subplots()

    im, cbar = heatmap(arr, split_labels, model_labels, ax=ax,
                    cmap="YlGn", cbarlabel="Accuracy")
    texts = annotate_heatmap(im) #, valfmt="{x:.1f} t")

    # fig.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots()
    # im = ax.imshow(arr, cmap="YlGn")

    # split_labels = [sets1["split_label"], sets2["split_label"]]
    # model_labels = [model1["model_label"], model2["model_label"]]

    # ax.set_yticks(np.arange(2), labels=split_labels)
    # ax.set_xticks(np.arange(2), labels=model_labels)

    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")

    # for i in range(len(split_labels)):
    #     for j in range(len(model_labels)):
    #         text = ax.text(j, i, arr[i, j],
    #                     ha="center", va="center", color="black")

    ax.set_title(res_name)
    fig.tight_layout()

    out_dir = os.path.join("eval_charts", "ds_splits")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, out_name + ".svg"))


if __name__ == "__main__":


    # model1 = {
    #     "model_name": "EARLY_31500_patches",
    #     "model_label": "EARLY_MODEL"
    # }

    # model2 = {
    #     "model_name": "LATE_38934_patches",
    #     "model_label": "LATE_MODEL"
    # }

    # sets1 = {
    #     "sets": ds_splits.get_split("EARLY", "ID"),
    #     "split_label": "EARLY_SETS"
    # }

    # sets2 = {
    #     "sets": ds_splits.get_split("LATE", "ID"),
    #     "split_label": "LATE_SETS"
    # }


    # split_heatmap(model1, model2, sets1, sets2, "EARLY-LATE split", "EARLY-LATE_heatmap")

    # continuum_plot(model1, model2, sets1, sets2, "EARLY-LATE split", "EARLY-LATE_continuum")


    # model1 = {
    #     "model_name": "RAND1_39312_patches",
    #     "model_label": "RAND1_MODEL"
    # }

    # model2 = {
    #     "model_name": "RAND2_31122_patches",
    #     "model_label": "RAND2_MODEL"
    # }

    # sets1 = {
    #     "sets": ds_splits.get_split("RAND1", "ID"),
    #     "split_label": "RAND1_SETS"
    # }

    # sets2 = {
    #     "sets": ds_splits.get_split("RAND2", "ID"),
    #     "split_label": "RAND2_SETS"
    # }


    # split_heatmap(model1, model2, sets1, sets2, "RAND1-RAND2 split", "RAND1-RAND2_heatmap")

    # continuum_plot(model1, model2, sets1, sets2, "RAND1-RAND2 split", "RAND1-RAND2_continuum")



    # model1 = {
    #     "model_name": "LOWEED_47250_patches",
    #     "model_label": "LOWEED_MODEL"
    # }

    # model2 = {
    #     "model_name": "HIWEED_23184_patches",
    #     "model_label": "HIWEED_MODEL"
    # }

    # sets1 = {
    #     "sets": ds_splits.get_split("LOWEED", "ID"),
    #     "split_label": "LOWEED_SETS"
    # }

    # sets2 = {
    #     "sets": ds_splits.get_split("HIWEED", "ID"),
    #     "split_label": "HIWEED_SETS"
    # }


    # split_heatmap(model1, model2, sets1, sets2, "LOWEED-HIWEED split", "LOWEED-HIWEED_heatmap")

    # continuum_plot(model1, model2, sets1, sets2, "LOWEED-HIWEED split", "LOWEED-HIWEED_continuum")



    model1 = {
        "model_name": "MANLOWEED_25578_patches",
        "model_label": "MANLOWEED_MODEL"
    }

    model2 = {
        "model_name": "MANHIWEED_44856_patches",
        "model_label": "MANHIWEED_MODEL"
    }

    sets1 = {
        "sets": ds_splits.get_split("MANLOWEED", "ID"),
        "split_label": "MANLOWEED_SETS"
    }

    sets2 = {
        "sets": ds_splits.get_split("MANHIWEED", "ID"),
        "split_label": "MANHIWEED_SETS"
    }


    split_heatmap(model1, model2, sets1, sets2, "MANLOWEED-MANHIWEED split", "MANLOWEED-MANHIWEED_heatmap")

    continuum_plot(model1, model2, sets1, sets2, "MANLOWEED-MANHIWEED split", "MANLOWEED-MANHIWEED_continuum")