import math as m
import numpy as np
import tensorflow as tf

# from shapely.geometry import Polygon


# import time


    
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


    # p_1 = Polygon([
    #     [box_1[1], box_1[0]],
    #     [box_1[1], box_1[2]],
    #     [box_1[3], box_1[2]],
    #     [box_1[3], box_1[0]]
    # ])

    # p_2 = Polygon([
    #     [box_2[1], box_2[0]],
    #     [box_2[1], box_2[2]],
    #     [box_2[3], box_2[2]],
    #     [box_2[3], box_2[0]]
    # ])

    # inter_poly = p_1.intersection(p_2)

    # if inter_poly.geom_type != "Polygon":
    #     return False, None

    # inter_bounds = inter_poly.bounds

    # if len(inter_bounds) != 4:
    #     return False, None

    # inter_box = [
    #     int(inter_bounds[1]),
    #     int(inter_bounds[0]),
    #     int(inter_bounds[3]),
    #     int(inter_bounds[2])
    # ]

    # return True, inter_box



def swap_xy_np(boxes):

    return np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def swap_xy_tf(boxes):
    """Swaps the order of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
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
    #if box_fmt == "xywh":
    #    boxes = convert_to_corners_tf(boxes)
    mask = np.logical_or(np.logical_or(boxes[:, 0] <= 0, boxes[:, 1] <= 0), 
                  np.logical_or(boxes[:, 2] >= patch_shape[0]-1, boxes[:, 3] >= patch_shape[1]-1))
    #edge_boxes = boxes[mask]
    #non_edge_boxes = boxes[np.logical_not(mask)]
    return mask #edge_boxes, non_edge_boxes



# def get_normalized_patch_wh(training_dataset):

#     BOX_PATCH_RATIO = 0.02

#     mean_box_area = training_dataset.get_mean_box_area()
#     patch_area = mean_box_area / BOX_PATCH_RATIO

#     patch_wh = round(m.sqrt(patch_area))

#     return patch_wh


# def clip_boxes_np(boxes, img_width, img_height):
#     """
#         boxes: min_y, min_x, max_y, max_x format
#     """

#     boxes = np.concatenate([np.maximum(boxes[:, :2], [0, 0]),
#                             np.minimum(boxes[:, 2:], [img_height-1, img_width-1])], axis=-1)
#     return boxes


def clip_boxes_np(boxes, patch_coords):
    """
        boxes: min_y, min_x, max_y, max_x format
    """

    #boxes = np.stack([np.maximum(boxes[:,0], patch_coords[0]),
    #                           np.maximum(boxes[:,1], patch_coords[1]),
    #                           np.minimum(boxes[:,2], patch_coords[2]-1),
    #                           np.minimum(boxes[:,3], patch_coords[3]-1)], axis=-1)

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
    # print("boxes.dtype", boxes.dtype)

    sel_indices = tf.image.non_max_suppression(boxes, scores, boxes.shape[0], iou_thresh)

    return sel_indices.numpy()
    # sel_boxes = tf.gather(boxes, sel_indices).numpy()
    # sel_scores = tf.gather(scores, sel_indices).numpy()

    # print("sel_boxes.dtype", sel_boxes.dtype)

    # return sel_boxes, sel_scores


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
        #boxes1_area = boxes1[:, 2] * boxes1[:, 3]
        #boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    elif box_format == "corners_yx":
        boxes1_corners = swap_xy_tf(boxes1) #boxes1 #swap_xy_tf(boxes1) #tf.convert_to_tensor(boxes1, dtype=tf.float32))
        boxes2_corners = swap_xy_tf(boxes2) #boxes2 #swap_xy_tf(boxes2) #tf.convert_to_tensor(boxes2, dtype=tf.float32))
    elif box_format == "corners_xy":
        boxes1_corners = boxes1
        boxes2_corners = boxes2
    else:
        raise RuntimeError("Unrecognized box format")

    boxes1_area = (boxes1_corners[:,2] - boxes1_corners[:,0]) * (boxes1_corners[:,3] - boxes1_corners[:,1])
    boxes2_area = (boxes2_corners[:,2] - boxes2_corners[:,0]) * (boxes2_corners[:,3] - boxes2_corners[:,1])

    # print("boxes1_corners", boxes1_corners)
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
    # print("---")
    # print(intersection_area)
    # print("---")
    # print(union_area)
    res = np.clip(intersection_area / union_area, 0.0, 1.0)
    # print("---")
    # print(res)
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


# """
#     Non-max Suppression Algorithm
#     @param list  Object candidate bounding boxes
#     @param list  Confidence score of bounding boxes
#     @param float IoU threshold
#     @return Rest boxes after nms operation
# """
# def stolen_nms(bounding_boxes, confidence_score, threshold):
#     # If no bounding boxes, return empty list
#     if len(bounding_boxes) == 0:
#         return [], []

#     # Bounding boxes
#     boxes = np.array(bounding_boxes)

#     print("First pass")
#     # iou_mat = compute_iou_np(bounding_boxes, bounding_boxes)
#     keep_indices = []
#     CHUNK_SIZE = 100
#     keeping = 0

#     region_boxes = []
#     max_box_y = np.max(boxes[:,2])
#     max_box_x = np.max(boxes[:,3])
#     print("Started getting region boxes")
#     for i in range(0, max_box_y, CHUNK_SIZE):
#         for j in range(0, max_box_x, CHUNK_SIZE):
#             get_contained_inds(boxes, [[i, j, i+CHUNK_SIZE, j+CHUNK_SIZE]])

#     print("Done getting region boxes")
#     exit()

#     for i in range(0, boxes.shape[0], CHUNK_SIZE):

        
#         # if i % 100000:
#         print(i / boxes.shape[0])
#         iou_mat = compute_iou_np(boxes[i:i+CHUNK_SIZE], boxes)
#         keep_mask = np.all(iou_mat < threshold, axis=1)
#         # keep_indices.append(i)
#         keeping += np.sum(keep_mask)

#     print("First pass complete: Keeping {} / {}".format(len(keeping), boxes.shape[0]))

#     # coordinates of bounding boxes
#     start_x = boxes[:, 0]
#     start_y = boxes[:, 1]
#     end_x = boxes[:, 2]
#     end_y = boxes[:, 3]

#     # Confidence scores of bounding boxes
#     score = np.array(confidence_score)

#     # Picked bounding boxes
#     picked_boxes = []
#     picked_score = []

#     # Compute areas of bounding boxes
#     areas = (end_x - start_x + 1) * (end_y - start_y + 1)

#     # Sort by confidence score of bounding boxes
#     order = np.argsort(score)

#     # Iterate bounding boxes

#     print("applying nms. order.size: {}".format(order.size))
#     org_order_size = order.size
#     percent_complete = 0
#     i = 0
#     while order.size > 0:

#         if i % 10 == 0:
#             print("i: {}, order.size: {}".format(i, order.size))
#         prev_percent_complete = percent_complete
#         percent_complete = ((org_order_size - order.size) / org_order_size) * 100
#         if m.floor(percent_complete) > m.floor(prev_percent_complete):
#             print("nms: {} percent complete.".format(m.floor(percent_complete)))


#         # The index of largest confidence score
#         index = order[-1]

#         # Pick the bounding box with largest confidence score
#         picked_boxes.append(bounding_boxes[index])
#         picked_score.append(confidence_score[index])

#         # Compute ordinates of intersection-over-union(IOU)
#         x1 = np.maximum(start_x[index], start_x[order[:-1]])
#         x2 = np.minimum(end_x[index], end_x[order[:-1]])
#         y1 = np.maximum(start_y[index], start_y[order[:-1]])
#         y2 = np.minimum(end_y[index], end_y[order[:-1]])

#         # Compute areas of intersection-over-union
#         w = np.maximum(0.0, x2 - x1 + 1)
#         h = np.maximum(0.0, y2 - y1 + 1)
#         intersection = w * h

#         # Compute the ratio between intersection and union
#         ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

#         left = np.where(ratio < threshold)
#         order = order[left]

#         i += 1

#     return picked_boxes, picked_score






        



# """
#     Non-max Suppression Algorithm
#     @param list  Object candidate bounding boxes
#     @param list  Confidence score of bounding boxes
#     @param float IoU threshold
#     @return Rest boxes after nms operation
# """
# def reworked_nms(pred_boxes, pred_scores, iou_threshold):
#     # If no bounding boxes, return empty list
#     if pred_boxes.size == 0:
#         return np.array([]), np.array([])

#     # Bounding boxes
#     # boxes = np.array(bounding_boxes)

#     # coordinates of bounding boxes
#     start_y = pred_boxes[:, 0]
#     start_x = pred_boxes[:, 1]
#     end_y = pred_boxes[:, 2]
#     end_x = pred_boxes[:, 3]

#     # Confidence scores of bounding boxes
#     # score = np.array(confidence_score)

#     # Picked bounding boxes
#     picked_boxes = []
#     picked_scores = []

#     # Compute areas of bounding boxes
#     areas = (end_x - start_x + 1) * (end_y - start_y + 1)

#     # Sort by confidence score of bounding boxes
#     order = np.argsort(pred_scores)

#     print("pred_scores.size", pred_scores.size)
#     print("pred_boxes.shape[0]", pred_boxes.shape[0])

#     # Iterate bounding boxes

#     print("applying nms. order.size: {}".format(order.size))
#     org_order_size = order.size
#     percent_complete = 0
#     i = 0
#     remaining_boxes = pred_boxes
#     remaining_scores = pred_scores
#     remaining_inds = np.arange(0, order.size)
#     still_kept_mask = np.full(order.size, True)
#     while order.size > 0:
#     # while boxes_left:

#         if i % 10 == 0:
#             print("i: {}, order.size: {}".format(i, order.size))
#         prev_percent_complete = percent_complete
#         percent_complete = ((org_order_size - order.size) / org_order_size) * 100
#         if m.floor(percent_complete) > m.floor(prev_percent_complete):
#             print("nms: {} percent complete.".format(m.floor(percent_complete)))


#         # The index of largest confidence score
#         index = order[-1]

#         # Pick the bounding box with largest confidence score
#         picked_box = remaining_boxes[index]
#         picked_score = remaining_scores[index]
#         picked_boxes.append(picked_box)
#         picked_scores.append(picked_score)


#         # start_time = time.time()

#         # involved_mask = np.full(order.size, False)
#         involved_mask = np.logical_and(
#                     np.logical_and(remaining_boxes[:,1] < picked_box[3], remaining_boxes[:,3] > picked_box[1]),
#                     np.logical_and(remaining_boxes[:,0] < picked_box[2], remaining_boxes[:,2] > picked_box[0])
#                 )



#         # involved_inds = get_contained_inds(bounding_boxes, bounding_boxes[index])
#         involved_mask[index] = False
        

#         # end_time = time.time()
#         # elapsed_time = end_time - start_time
#         # print("Time to figure out which boxes are involved", elapsed_time)

#         # np.delete()
#         # involved_boxes = bounding_boxes[intersection_inds]

#         # start_time = time.time()
#         # Compute ordinates of intersection-over-union(IOU)
#         x1 = np.maximum(start_x[still_kept_mask][index], start_x[still_kept_mask][involved_mask])
#         x2 = np.minimum(end_x[still_kept_mask][index], end_x[still_kept_mask][involved_mask])
#         y1 = np.maximum(start_y[still_kept_mask][index], start_y[still_kept_mask][involved_mask])
#         y2 = np.minimum(end_y[still_kept_mask][index], end_y[still_kept_mask][involved_mask])

#         # Compute areas of intersection-over-union
#         w = np.maximum(0.0, x2 - x1)
#         h = np.maximum(0.0, y2 - y1)
#         intersection = w * h

#         # Compute the ratio between intersection and union
#         ratio = intersection / (areas[still_kept_mask][index] + areas[still_kept_mask][involved_mask] - intersection)

#         # left = np.where(ratio < threshold)

#         involved_and_keep = ratio < iou_threshold


#         # end_time = time.time()
#         # elapsed_time = end_time - start_time
#         # print("Time to figure out which involved boxes are kept", elapsed_time)

#         # keep_for_next_round = np.full(order.size, False)

#         # keep_for_next_round[np.logical_not(involved_mask)] = True
#         # keep_for_next_round[np.where(involved_mask)[0][involved_and_keep]] = True
#         # keep_for_next_round[index] = False

#         # start_time = time.time()

#         keep_for_next_round = np.full(order.size, False)
#         keep_for_next_round[np.logical_not(involved_mask)] = True
#         keep_for_next_round[np.where(involved_mask)[0][involved_and_keep]] = True
#         keep_for_next_round[index] = False



#         # keep_mask = np.full(bounding_boxes.shape[0], True)
#         # keep_mask[index] = False
        
#         # keep_mask[involved_inds] = False
#         # keep_mask[involved_inds][left] = True


#         # additional_keep_inds = involved_inds[left]

#         # order = order[left]


#         # print("np.all(keep_for_next_round)", np.all(keep_for_next_round))
#         # print("np.any(keep_for_next_round)", np.any(keep_for_next_round))

#         order = order[keep_for_next_round]
#         remaining_boxes = remaining_boxes[keep_for_next_round]
#         remaining_scores = remaining_scores[keep_for_next_round]


#         still_kept_mask[remaining_inds[np.logical_not(keep_for_next_round)]] = False

#         remaining_inds = remaining_inds[keep_for_next_round]

#         # end_time = time.time()
#         # elapsed_time = end_time - start_time
#         # print("Time to do array management", elapsed_time)

#         # still_kept_mask = np.logical_and(still_kept_mask, keep_for_next_round)


#         i += 1

#     print("picked_boxes.shape[0]", np.array(picked_boxes).shape[0])

#     return np.array(picked_boxes), np.array(picked_scores)




# def non_max_suppression_fast(boxes, overlapThresh):
# 	# if there are no boxes, return an empty list
# 	if len(boxes) == 0:
# 		return []
# 	# if the bounding boxes integers, convert them to floats --
# 	# this is important since we'll be doing a bunch of divisions
# 	if boxes.dtype.kind == "i":
# 		boxes = boxes.astype("float")
# 	# initialize the list of picked indexes	
# 	pick = []
# 	# grab the coordinates of the bounding boxes
# 	x1 = boxes[:,0]
# 	y1 = boxes[:,1]
# 	x2 = boxes[:,2]
# 	y2 = boxes[:,3]
# 	# compute the area of the bounding boxes and sort the bounding
# 	# boxes by the bottom-right y-coordinate of the bounding box
# 	area = (x2 - x1 + 1) * (y2 - y1 + 1)
# 	idxs = np.argsort(y2)
# 	# keep looping while some indexes still remain in the indexes
# 	# list
# 	while len(idxs) > 0:
# 		# grab the last index in the indexes list and add the
# 		# index value to the list of picked indexes
# 		last = len(idxs) - 1
# 		i = idxs[last]
# 		pick.append(i)
# 		# find the largest (x, y) coordinates for the start of
# 		# the bounding box and the smallest (x, y) coordinates
# 		# for the end of the bounding box
# 		xx1 = np.maximum(x1[i], x1[idxs[:last]])
# 		yy1 = np.maximum(y1[i], y1[idxs[:last]])
# 		xx2 = np.minimum(x2[i], x2[idxs[:last]])
# 		yy2 = np.minimum(y2[i], y2[idxs[:last]])
# 		# compute the width and height of the bounding box
# 		w = np.maximum(0, xx2 - xx1 + 1)
# 		h = np.maximum(0, yy2 - yy1 + 1)
# 		# compute the ratio of overlap
# 		overlap = (w * h) / area[idxs[:last]]
# 		# delete all indexes from the index list that have
# 		idxs = np.delete(idxs, np.concatenate(([last],
# 			np.where(overlap > overlapThresh)[0])))
# 	# return only the bounding boxes that were picked using the
# 	# integer data type
# 	return boxes[pick].astype("int")