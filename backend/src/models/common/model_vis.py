import numpy as np
import cv2


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
