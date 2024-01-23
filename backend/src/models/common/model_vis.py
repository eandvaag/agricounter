import numpy as np
#import os
#import shutil
#import tqdm
import cv2
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#import plotly.graph_objects as go
#import plotly.express as px

#from io_utils import json_io

#import models.common.decode_predictions as decode_predictions


def draw_boxes_on_image(image_array,
                        pred_boxes,
                        pred_classes,
                        pred_scores,
                        class_map,
                        gt_boxes=None,
                        patch_coords=None,
                        display_class=True,
                        display_score=True):

    if display_class:
        rev_class_map = dict([(v, k) for k, v in class_map.items()])

    out_array = np.copy(image_array)

    if gt_boxes is not None:
        shapes = np.zeros_like(image_array, np.uint8)
        for gt_box in gt_boxes:
            cv2.rectangle(shapes, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (255, 0, 0), -1)
        alpha = 0.25
        mask = shapes.astype(bool)
        out_array[mask] = cv2.addWeighted(image_array, alpha, shapes, 1-alpha, 0)[mask]


    if patch_coords is not None:
        for patch_coord in patch_coords:
            cv2.rectangle(out_array, (patch_coord[1], patch_coord[0]), (patch_coord[3], patch_coord[2]), (255, 0, 255), 1)
        #    cv2.rectangle(out_array, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (255, 0, 0), 1)

    for pred_box, pred_class, pred_score in zip(pred_boxes, pred_classes, pred_scores):

        cv2.rectangle(out_array, (max(pred_box[1], 0), max(pred_box[0], 0)),
                                 (min(pred_box[3], out_array.shape[1]), min(pred_box[2], out_array.shape[0])),
                                 (0, 255, 0), 1)

        if display_class or display_score:
            if display_class and display_score:
                label = rev_class_map[pred_class] + ": " + str(round(pred_score, 2))
            elif display_class:
                label = rev_class_map[pred_class]
            elif display_score:
                label = str(round(pred_score, 2))

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_array, (pred_box[1], int(pred_box[0] - text_h)), (int(pred_box[1] + text_w), pred_box[0]), (0, 255, 0), -1)
            cv2.putText(out_array, label, (pred_box[1], pred_box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out_array






# def output_patch_predictions(out_dir, patch_predictions_lst, class_map, patch_paths):

#     for patch_path in patch_paths:

#         patch_arrays = []

#         for i, patch_predictions in enumerate(patch_predictions_lst):

#             patch_pred = patch_predictions[patch_path]

#             patch_array = cv2.imread(patch_path)

#             gt_boxes = patch_pred["patch_abs_boxes"] if "patch_abs_boxes" in patch_pred else None

#             patch_array = draw_boxes_on_img(patch_array,
#                                            patch_pred["pred_patch_abs_boxes"],
#                                            patch_pred["pred_classes"], 
#                                            patch_pred["pred_scores"],
#                                            class_map,
#                                            gt_boxes=gt_boxes,
#                                            display_class=True,
#                                            display_score=True)

#             patch_array = cv2.cvtColor(patch_array, cv2.COLOR_BGR2RGB)

#             patch_arrays.append(patch_array)

#         patch_arrays = np.stack(patch_arrays, axis=0)

#         fig = px.imshow(patch_arrays, animation_frame=0, binary_string=True,
#                         labels=dict(animation_frame="slice"), binary_compression_level=5)

#         out_name = os.path.basename(patch_path)[:-4] + "_predictions.html"

#         fig.write_html(os.path.join(out_dir, out_name))






    # for patch_path, patch_pred in tqdm.tqdm(patch_predictions, desc="Outputting patch predictions"):

    #     if patch_paths is not None and pred["patch_path"] not in patch_paths:
    #         continue

    #     patch_array = cv2.imread(pred["patch_path"])

    #     gt_boxes = pred["patch_abs_boxes"] if is_annotated else None

    #     pred_patch = draw_boxes_on_img(patch_array, 
    #                                    pred["pred_patch_abs_boxes"],
    #                                    pred["pred_classes"], 
    #                                    pred["pred_scores"],
    #                                    class_map,
    #                                    gt_boxes=gt_boxes,
    #                                    display_class=True,#False,
    #                                    display_score=True)#False)


    #     pred_patch_name = os.path.basename(pred["patch_path"])[:-4] + "_predictions.png"

    #     cv2.imwrite(os.path.join(out_dir, pred_patch_name), pred_patch)





def output_image_predictions(out_dir, img_predictions_lst, class_map, img_paths):

    for img_path in img_paths:

        img_arrays = []

        for i, img_predictions in enumerate(img_predictions_lst):

            img_pred = img_predictions[img_path]

            img_array = cv2.imread(img_path)

            gt_boxes = img_pred["img_abs_boxes"] if "img_abs_boxes" in img_pred else None

            img_array = draw_boxes_on_img(img_array,
                                           img_pred["nms_pred_img_abs_boxes"],
                                           img_pred["nms_pred_classes"], 
                                           img_pred["nms_pred_scores"],
                                           class_map,
                                           gt_boxes=gt_boxes,
                                           patch_coords=img_pred["patch_coords"],
                                           display_class=True,
                                           display_score=True)

            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            img_arrays.append(img_array)

        img_arrays = np.stack(img_arrays, axis=0)

        fig = px.imshow(img_arrays, animation_frame=0, binary_string=True,
                        labels=dict(animation_frame="slice"), binary_compression_level=9)

        out_name = os.path.basename(img_path).split(".")[0] + "_predictions.html"

        fig.write_html(os.path.join(out_dir, out_name))



    # for img_path in tqdm.tqdm(img_predictions.keys(), "Outputting image predictions"):

    #     if img_paths is not None and img_path not in img_paths:
    #         continue

    #     img_array = cv2.imread(img_path)

    #     gt_boxes = img_predictions[img_path]["img_abs_boxes"] if is_annotated else None

    #     pred_img = draw_boxes_on_img(img_array,
    #                                  img_predictions[img_path]["nms_pred_img_abs_boxes"],
    #                                  img_predictions[img_path]["nms_pred_classes"],
    #                                  img_predictions[img_path]["nms_pred_scores"],
    #                                  class_map,
    #                                  gt_boxes=gt_boxes,
    #                                  patch_coords=img_predictions[img_path]["patch_coords"],
    #                                  display_class=False,
    #                                  display_score=False)
    #     pred_img_name = os.path.basename(img_path)[:-4] + "_predictions.png"

    #     cv2.imwrite(os.path.join(out_dir, pred_img_name), pred_img)






# def output_patch_count_diff_histogram(model, patch_dir, settings):

#     model_eval.evaluate(model, patch_dir, settings)

#     pred_dir = os.path.join(model.config.model_dir, os.path.basename(patch_dir))
#     pred_stats = json_io.load_json(os.path.join(pred_dir, "prediction_stats.json"))

#     out_dir = os.path.join(pred_dir, "prediction_stats_vis")
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     save_path = os.path.join(out_dir, "patch_plant_count_diff_histogram.svg")

#     patch_paths = pred_stats["patch_results"].keys()
#     data = [pred_stats["patch_results"][patch_path]["pred_minus_actual"] for patch_path in patch_paths]

#     plt.figure()
#     plt.hist(data, color="palegreen")
#     plt.title("Patch Plant Counts: Predicted Minus Actual")
#     plt.xlabel("Count Difference")
#     plt.ylabel("Number of patches")
#     plt.savefig(save_path)
#     plt.close()




# def loss_plot(out_dir, model_instance_name, model_dir):

#     loss_record_path = os.path.join(model_dir, "loss_record.json")
#     save_path = os.path.join(out_dir, "loss_plot.svg")

#     loss_record = json_io.load_json(loss_record_path)

#     train_loss_values = [loss_record[epoch]["train_loss"] for epoch in loss_record.keys()]
#     val_loss_values = [loss_record[epoch]["val_loss"] for epoch in loss_record.keys()]
#     epochs = [int(epoch) for epoch in loss_record.keys()]

#     plt.figure()
#     plt.plot(epochs, train_loss_values, "steelblue", label="Training")
#     plt.plot(epochs, val_loss_values, "tomato", label="Validation")
#     plt.title(model_instance_name + " Loss Values")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig(save_path)
#     plt.close()



# def image_counts(out_dir, model_instance_names, pred_stats_lst):

#     data = []

#     for i, (model_instance_name, pred_stats) in enumerate(zip(model_instance_names, pred_stats_lst)):

#         img_paths = sorted(pred_stats["image_results"].keys())
#         img_names = [os.path.basename(img_path)[:-4] for img_path in img_paths]
#         pred_counts = [pred_stats["image_results"][img_path]["num_pred"] for img_path in img_paths]


#         if i == 0 and pred_stats["is_annotated"]:
#             gt_counts = [pred_stats["image_results"][img_path]["num_actual"] for img_path in img_paths]
#             data.append(go.Bar(name="Ground Truth", x=img_names, y=gt_counts))

#         data.append(go.Bar(name=model_instance_name, x=img_names, y=pred_counts))

#     fig = go.Figure(data=data)
#     fig.update_layout(
#         barmode='group',
#         title="Ground Truth and Predicted Counts",
#         xaxis_title="Image",
#         yaxis_title="Count",
#         font=dict(
#             family="verdana",
#             size=18,
#             color="darkslategray"
#         )
#     )
#     out_path = os.path.join(out_dir, "image_counts.html")
#     fig.write_html(out_path)


# def time_vs_mAP(out_dir, model_instance_names, pred_lst, pred_stats_lst, metric="coco_score"):

#     fig = go.Figure()
#     for (model_instance_name, pred, pred_stats) in zip(model_instance_names, pred_lst, pred_stats_lst):

#         per_patch_inference_time = pred["per_patch_inference_time"]
#         score = pred_stats["img_summary_results"]["coco_score"]

#         fig.add_trace(go.Scatter(x=[per_patch_inference_time], y=[score],
#                                  mode="markers", marker=dict(size=5), name=model_instance_name))


#     fig.update_layout(
#         title="Inference Time vs COCO mAP",
#         xaxis_title="Per-Patch Inference Time",
#         yaxis_title="COCO mAP",
#         font=dict(
#             family="verdana",
#             size=18,
#             color="darkslategray"
#         )
#     )

#     out_path = os.path.join(out_dir, "inference_time_vs_mAP.html")
#     fig.write_html(out_path)