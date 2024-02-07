import numpy as np
import tensorflow as tf
    
def get_intersection_rect(box_1, box_2):


    intersects = np.logical_and(
                        np.logical_and(box_1[1] < box_2[3], box_1[3] > box_2[1]),
                        np.logical_and(box_1[0] < box_2[2], box_1[2] > box_2[0])
                      )

    if intersects:

        intersection_rect = [
            max(box_1[0], box_2[0]),
            max(box_1[1], box_2[1]),
            min(box_1[2], box_2[2]),
            min(box_1[3], box_2[3])
        ]

        return True, intersection_rect
    else:
        return False, None


def swap_xy_np(boxes):
    if boxes.size == 0:
        return boxes
    else:
        return np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def swap_xy_tf(boxes):
    """Swaps the order of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    if tf.size(boxes) == 0:
        return boxes
    else:
        return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh_tf(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners_tf(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def convert_to_openseadragon_format(boxes, img_width, img_height):
    """
        input boxes are in min_y, min_x, max_y, max_x format
    """

    min_y = boxes[..., 0] / img_height
    min_x = boxes[..., 1] / img_width
    h = (boxes[..., 2] - boxes[..., 0]) / img_height
    w = (boxes[..., 3] - boxes[..., 1]) / img_width

    return np.stack([min_x, min_y, w, h], axis=-1)


def box_areas_np(boxes):
    """
        boxes: min_y, min_x, max_y, max_x format
    """

    return (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])

def box_visibilities_np(boxes, clipped_boxes):

    box_areas = box_areas_np(boxes)
    clipped_box_areas = box_areas_np(clipped_boxes)
    visibilities = np.divide(clipped_box_areas, box_areas, out=np.zeros_like(clipped_box_areas, dtype="float64"), where=box_areas!=0)
    return visibilities

def clip_boxes_and_get_small_visibility_mask(boxes, patch_coords, min_visibility):

    clipped_boxes = clip_boxes_np(boxes, patch_coords)
    box_visibilities = box_visibilities_np(boxes, clipped_boxes)
    mask = box_visibilities >= min_visibility
    return clipped_boxes, mask


def get_edge_boxes_mask(boxes, patch_shape):

    mask = np.logical_or(np.logical_or(boxes[:, 0] <= 0, boxes[:, 1] <= 0), 
                  np.logical_or(boxes[:, 2] >= patch_shape[0]-1, boxes[:, 3] >= patch_shape[1]-1))
    return mask



# Note this doesn't give 'correct' behaviour if the boxes are not at least 
# partially contained within the patch_coords
def clip_boxes_np(boxes, patch_coords):
    """
        boxes: min_y, min_x, max_y, max_x format
    """
    boxes = np.concatenate([np.maximum(boxes[:, :2], [patch_coords[0], patch_coords[1]]),
                            np.minimum(boxes[:, 2:], [patch_coords[2], patch_coords[3]])], axis=-1)
    return boxes

def non_max_suppression_with_classes(boxes, classes, scores, iou_thresh):

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)
    sel_boxes = tf.gather(boxes, sel_indices).numpy()
    sel_classes = tf.gather(classes, sel_indices).numpy()
    sel_scores = tf.gather(scores, sel_indices).numpy()
    
    return sel_boxes, sel_classes, sel_scores

def non_max_suppression(boxes, scores, iou_thresh):
    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)
    return sel_indices.numpy()


def non_max_suppression_indices(boxes, scores, iou_thresh):

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)

    return (sel_indices).numpy()


def compute_iou(boxes1, boxes2, box_format="xywh"):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

      box_format: "xywh" or "corners_xy" or "corner_yx"
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """



    if box_format == "xywh":
        boxes1_corners = convert_to_corners_tf(boxes1)
        boxes2_corners = convert_to_corners_tf(boxes2)
    elif box_format == "corners_yx":
        boxes1_corners = swap_xy_tf(boxes1)
        boxes2_corners = swap_xy_tf(boxes2)
    elif box_format == "corners_xy":
        boxes1_corners = boxes1
        boxes2_corners = boxes2
    else:
        raise RuntimeError("Unrecognized box format")

    boxes1_area = (boxes1_corners[:,2] - boxes1_corners[:,0]) * (boxes1_corners[:,3] - boxes1_corners[:,1])
    boxes2_area = (boxes2_corners[:,2] - boxes2_corners[:,0]) * (boxes2_corners[:,3] - boxes2_corners[:,1])

    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    res = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


    return res




def compute_iou_np(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:

      box_format: [min_x, min_y, max_x, max_y]
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """

    boxes1_corners = boxes1
    boxes2_corners = boxes2

    boxes1_area = (boxes1_corners[:,2] - boxes1_corners[:,0]) * (boxes1_corners[:,3] - boxes1_corners[:,1])
    boxes2_area = (boxes2_corners[:,2] - boxes2_corners[:,0]) * (boxes2_corners[:,3] - boxes2_corners[:,1])

    lu = np.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = np.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    union_area = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )

    res = np.clip(intersection_area / union_area, 0.0, 1.0)

    return res


def get_contained_inds_for_points(points, regions):
    if points.size == 0:
        return np.array([], dtype=np.int64)

    mask = np.full(points.shape[0], False)

    for region in regions:


        region_mask = np.logical_and(
                        np.logical_and(points[:,0] > region[0], points[:,0] < region[2]),
                        np.logical_and(points[:,1] > region[1], points[:,1] < region[3]))
        mask = np.logical_or(mask, region_mask)

    return np.where(mask)[0]



def get_fully_contained_inds(boxes, regions):

    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    mask = np.full(boxes.shape[0], False)

    for region in regions:
        region_mask = np.logical_and(
                        np.logical_and(boxes[:,3] <= region[3], boxes[:,1] >= region[1]),
                        np.logical_and(boxes[:,2] <= region[2], boxes[:,0] >= region[0])
                      )
        mask = np.logical_or(mask, region_mask)
        
    return np.where(mask)[0]

def get_contained_inds(boxes, regions):

    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    mask = np.full(boxes.shape[0], False)

    for region in regions:
        region_mask = np.logical_and(
                        np.logical_and(boxes[:,1] < region[3], boxes[:,3] > region[1]),
                        np.logical_and(boxes[:,0] < region[2], boxes[:,2] > region[0])
                      )
        mask = np.logical_or(mask, region_mask)
        
    return np.where(mask)[0]