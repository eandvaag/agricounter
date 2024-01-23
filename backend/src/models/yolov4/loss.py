import tensorflow as tf
import numpy as np

from models.yolov4.iou import bbox_iou, bbox_ciou
from models.yolov4.encode import Decoder

class YOLOv4Loss:
    """

        GIoU: Generalized IoU
        DIoU: Distance-IoU -- incorporates the normalized distance between the predicted box and the target box
        CIoU: Complete IoU -- incorporates 3 geometric factors: overlap area, central point distance, and aspect ratio


        l2 loss is a standard loss function for object detectors.

        IoU loss improves on l2 loss by considering the box as a single unit.
        IoU loss = 1 - IoU

        IoU loss only works when the bounding boxes have overlap, and does not provide any gradient for
        non-overlapping cases. Thus, Generalized IoU (GIoU) adds an additional penalty term:


        GIoU loss = 1 - IoU + ( | C - Union(B, Bgt) | / | C | ),

        where C is the smallest box covering B and Bgt. ( | X | means "the area of X".)
    
        A downside of GIoU loss is that it tends to increase the size of the predicted box so that it
        will overlap with the target, rather than directly minimizing the normalized distance of central points.
        It also degrades to IoU loss when one box encloses the other box. DIoU loss addresses these issues, leading
        to faster convergence:

        DIoU loss = 1 - IoU + (p^2(b, bgt) / c^2)


        The last term is a measure of the central point distance (the distance between the centre
        of the ground truth box and the predicted box).
        b and bgt are the centre points of the predicted box (B) and ground truth box (Bgt).
        p() is Euclidean distance
        c is the diagonal length of the smallest enclosing box covering the two boxes
        While GIoU loss aims to reduce the area of C - Union(B, Bgt), the penalty term of DIoU loss
        directly minimizes the distance between two central points.
    

        CIoU loss adds an additional term to DIoU loss. This term compares the aspect ratios of the
        ground truth box and the predicted box.


        CIoU loss = 1 - IoU + (p^2(b, bgt) / c^2) + (alpha * v)


        alpha is a positive trade-off parameter, defined as:
    
            alpha = v / ((1 - IoU) + v) ,

        by which the overlap area factor is given higher priority for regression, especially for
        non-overlapping cases.

        v measures the consistency of the aspect ratio, defined as:

            v = (4 / pi^2) * (arctan(wgt/ hgt) - arctan(w / h))^2


    """

    def __init__(self, config):

        self.num_classes = config["arch"]["num_classes"]
        self.strides = config["arch"]["strides"]
        self.iou_loss_thresh = config["arch"]["iou_loss_thresh"]
        self.anchors_per_scale = config["arch"]["anchors_per_scale"]
        self.num_scales = config["arch"]["num_scales"]
        self.decoder = Decoder(config)

    @tf.function
    def __call__(self, batch_labels, conv):


        pred = self.decoder(conv)
        ciou_loss = 0
        obj_loss = 0
        prob_loss = 0
        for i in range(self.num_scales):
            label = batch_labels[i][0]
            bboxes = batch_labels[i][1]
            loss_items = self._calculate_loss_for_scale(label, bboxes, conv[i], pred[i], self.strides[i])
            ciou_loss += loss_items[0]
            obj_loss += loss_items[1]
            prob_loss += loss_items[2]

        # label = batch_labels[0][0]
        # bboxes = batch_labels[0][1]
        # loss_items = self._calculate_loss_for_scale(label, bboxes, conv[0], pred[0], self.strides[0])
        # ciou_loss += loss_items[0]
        # obj_loss += loss_items[1]
        # prob_loss += loss_items[2]

        # label = batch_labels[1][0]
        # bboxes = batch_labels[1][1]
        # loss_items = self._calculate_loss_for_scale(label, bboxes, conv[1], pred[1], self.strides[1])
        # ciou_loss += loss_items[0]
        # obj_loss += loss_items[1]
        # prob_loss += loss_items[2]


        # label = batch_labels[2][0]
        # bboxes = batch_labels[2][1]
        # loss_items = self._calculate_loss_for_scale(label, bboxes, conv[2], pred[2], self.strides[2])
        # ciou_loss += loss_items[0]
        # obj_loss += loss_items[1]
        # prob_loss += loss_items[2]


        loss_value = ciou_loss + obj_loss + prob_loss
        return loss_value


        # if i == 0:
        #     print("tf.shape(y_pred)", tf.shape(y_pred))
        #     print("y_pred", y_pred)

        #     print("tf.shape(label)", tf.shape(label))
        #     print("label", label)
        #     print("label all_zeros?", not np.any(label))

        #     print("tf.shape(bboxes)", tf.shape(bboxes))
        #     print("bboxes", bboxes)

        #print("tf.shape(conv)", tf.shape(conv))
        #print("tf.shape(pred)", tf.shape(pred))

    #@tf.function
    def _calculate_loss_for_scale(self, label, bboxes, conv, pred, stride):
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = output_size * tf.cast(stride, dtype=tf.int32)


        # at this point, both predicted and ground truth boxes are in xywh format
        # where x, y, w, and h are all absolute coordinates within the input image shape

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]
        #pred_prob = pred[:, :, :, :, 5: ]

        conv = tf.reshape(conv, (batch_size, output_size, output_size, self.anchors_per_scale, 5 + self.num_classes))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        # respond_bbox: if the bounding box prior has been assigned to a ground truth object,
        # the value will be 1.0. Otherwise, the value is 0.
        # If a bounding box prior is not assigned to a ground truth object, it incurs no loss
        # for coordinate or class predictions, only objectness.


        ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)


        # (??) the smaller the size of the bounding box, the larger the value of bbox_loss_scale,
        # which can weaken the influence of the size of the bounding box on the loss value
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)


        # the larger the CIoU value between the two bounding boxes, the smaller the loss value of CIoU
        # (?? I cannot find this concept of bbox_loss_scale in any paper)
        ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)

        #iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        iou = bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])

        # (??) Find the predicted box with the largest IoU value from the ground truth box
        # (Possibly: for each predicted box, find the ground truth box with the largest IoU)
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest IoU is less than the threshold, then it is considered that the predicted box
        # does not contain objects, and it is a background box
        # to be a 'background box', a predicted box must correspond with an anchor that has not been assigned to an object 
        #(i.e., respond_bbox entry == 0), AND its maximum IoU with any ground truth box must be less than iou_loss_thresh.  
        # if a predicted box corresponds with an anchor that has not been assigned to an object (i.e., respond_bbox entry == 0),
        # but its maximum IoU with a ground truth box is greater than iou_loss_thresh, then it is ignored in the loss.
        # "If the bounding box prior is not the best but does overlap a ground truth object by more
        # than some threshold we ignore the prediction, following [17]. We use a threshold of 0.5.""


        # only anchors that are considered "positives" (respond_bbox == 1) contribute to the 
        # ciou_loss (box regression loss) and prob_loss (classification loss). Unlike RetinaNet, YOLOv4
        # can assign multiple anchors to the same object. With RetinaNet, only the the anchor with the 
        # greatest IoU with the object is assigned to the object. With YOLOv4, all anchors 
        # (from a possible total of num_scales * anchors_per_scale) with IoU greater than 0.3 are considered 
        # "positives" (this is done in the encoding).

        # For the confidence / objectness loss, both "positive" and "background" anchors contribute to the loss.
        # Only "ignored" boxes do not contribue to the confidence loss.

        # an apparent difference is that "background" anchors are determined in YOLOv4 using IoU between
        # *predicted* and ground truth boxes. In RetinaNet, we look at IoU between *anchors* and ground truth boxes.
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)


        # Focal loss: by modifying the standard cross-entropy loss function, the weight of samples that
        # can be classified well is reduced.

        # conf_focal is the modulating factor for the focal loss.
        # here, gamma is 2, and alpha seems to be missing
        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        eps = 1e-15
        # conf_loss = conf_focal * (
        #                 respond_bbox * -(respond_bbox * tf.math.log(tf.clip_by_value(pred_conf, eps, 1.0)))
        #                 +
        #                 respond_bgd * -(respond_bgd * tf.math.log(tf.clip_by_value((1 - pred_conf), eps, 1.0)))
        #                 )

        # essentially: compute cross-entropy loss for all anchors, but only consider contributions from 
        # "positive" and "background" anchors. Do not consider contributions from ignored anchors
        conf_loss = conf_focal * (
                        respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                        +
                        respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                        )
        # ^ should the second labels be respond_bbox, or respond_bgd ?


        # this is just categorical cross-entropy loss. Both label_prod and pred_prob have values that vary between
        # 0 and 1. label_prod is the smoothed one-hot encoded class vector. pred_prob has been passed through
        # the sigmoid function in decode(), so it also has values between 0 and 1.
        # only anchors that are considered to have an object in them (respond_bbox == 1) contribute to the prob_loss. 
        # the clipping is just for stability.
        #prob_loss = respond_bbox * -(label_prob * tf.math.log(tf.clip_by_value(pred_prob, eps, 1.0))
        #                            +
        #                            (1 - label_prob) * tf.math.log(tf.clip_by_value((1 - pred_prob), eps, 1.0)))


        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        
        save_ciou_loss = ciou_loss
        save_conf_loss = conf_loss
        save_prob_loss = prob_loss

        #tf.print("ciou_loss", ciou_loss)
        #tf.print("conf_loss", conf_loss)
        #tf.print("prob_loss", prob_loss)

        ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        #tf.print("ciou_loss", ciou_loss)
        #tf.print("conf_loss", conf_loss)
        #tf.print("prob_loss", prob_loss)
        
        # if tf.math.is_nan(ciou_loss) or tf.math.is_nan(conf_loss) or tf.math.is_nan(prob_loss):

        #     print("NaN loss has occurred")

        #     print("pred", pred)

        #     print("ciou_loss", save_ciou_loss)
        #     print("conf_loss", save_conf_loss)
        #     print("prob_loss", save_prob_loss)

        #     print("final ciou_loss", ciou_loss)
        #     print("final conf_loss", conf_loss)
        #     print("final prob_loss", prob_loss)

        #     exit()

        return ciou_loss, conf_loss, prob_loss


