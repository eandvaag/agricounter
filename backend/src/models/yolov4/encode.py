import tensorflow as tf
import numpy as np

from models.yolov4.iou import bbox_iou_np


class LabelEncoder:


    def __init__(self, config):
        super(LabelEncoder, self).__init__()
        self.strides = config["arch"]["strides"]
        self.num_classes = config["arch"]["num_classes"]
        self.anchors = config["arch"]["anchors"]
        self.anchors_per_scale = config["arch"]["anchors_per_scale"]
        self.input_image_shape = config["arch"]["input_image_shape"]
        assert self.input_image_shape[0] == self.input_image_shape[1]
        self.output_dim = self.input_image_shape[0] // self.strides # an nx1 array, n == num_scales
        self.max_detections_per_scale = config["arch"]["max_detections_per_scale"]
        self.num_scales = config["arch"]["num_scales"]
        #self.batch_size = config.training["active"]["batch_size"]



    def _encode_sample(self, gt_boxes, cls_ids):
        """
            gt_boxes: xywh abs pixel format
        """
        # At each level of the pyramid, there are self.train_output_sizes[i] * self.train_output_sizes[i] locations
        # for anchor boxes. At each location there are self.anchors_per_scale anchor_boxes. Each encoded box is
        # represented with 5 + self.num_classes values (4 for box coordinates, 1 for 'objectness', and 
        # self.num_classes for the smooth one-hot class vector)
        label = [np.zeros((self.output_dim[i], self.output_dim[i], self.anchors_per_scale, 
                 5 + self.num_classes)) for i in range(self.num_scales)]


        # contains the bboxes that have been assigned to each level of the pyramid
        bboxes_xywh = [np.zeros((self.max_detections_per_scale, 4)) for _ in range(self.num_scales)]

        # bbox count records the number of bboxes at each level of the pyramid
        bbox_count = np.zeros((3,))

        for (bbox_xywh, cls_id) in zip(gt_boxes, cls_ids):
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[cls_id] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            delta = 0.01
            smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

            # an nx4 array -- each row contains the scaled xywh box coordinates for that scale of the pyramid
            # these are 'pixel coordinates' (though they are floats)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            
            #print("bbox_xywh_scaled", bbox_xywh_scaled)

            iou_lst = []
            exist_positive = False
            for i in range(self.num_scales):

                # anchors xywh is nx4
                anchors_xywh = np.zeros((self.anchors_per_scale, 4))
                # round down the scaled xy centre of the box to the nearest integer, then add 0.5 to those rounded values
                # this will be the xy position of the closest anchor at that level of pyramid
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                # assign wh of anchors for this scale. N.B.: the anchors have already been divided by the appropriate stride.
                # So we can directly compare anchors[i] with bbox_xywh_scaled[i].
                anchors_xywh[:, 2:4] = self.anchors[i]

                
                iou_scale = bbox_iou_np(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)

                iou_lst.append(iou_scale)
                iou_mask = iou_scale > 0.3

                # do any of the closest anchors at this scale have an iou with the gt box that is greater than 0.3 ?
                if np.any(iou_mask):
                    #xind, yind -- the index of the cell responsible for this box (at this level of the pyramid)
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)


                    # set labels for all anchors in the xind, yind cell at level i that have IoU threshold > 0.3 
                    # (each assign will be multiple assigns if more than one anchor has IoU threshold > 0.3)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] %  self.max_detections_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True
                    
                    #print("found a positive match (i: {})".format(i))

            if not exist_positive:
                # 3 scales, 3 anchors per scale --> pick the best anchor
                best_anchor_ind = np.argmax(np.array(iou_lst).reshape(-1), axis=-1)
                best_scale = int(best_anchor_ind / self.anchors_per_scale)
                best_anchor = int(best_anchor_ind % self.anchors_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_scale, 0:2]).astype(np.int32)

                label[best_scale][yind, xind, best_anchor, :] = 0
                label[best_scale][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_scale][yind, xind, best_anchor, 4:5] = 1.0
                label[best_scale][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_scale] % self.max_detections_per_scale)
                bboxes_xywh[best_scale][bbox_ind, :4] = bbox_xywh
                bbox_count[best_scale] += 1


        return [label, bboxes_xywh]

        #label_sbbox, label_mbbox, label_lbbox = label
        #sbboxes, mbboxes, lbboxes = bboxes_xywh
        #return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


    def encode_batch(self, batch_images, batch_gt_boxes, batch_cls_ids):
        # print("batch_gt_boxes", batch_gt_boxes)
        # print("batch_cls_ids", batch_cls_ids)

        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        batch_labels = []
        for i in range(self.num_scales):
            batch_labels.append(np.zeros((batch_size, self.output_dim[i], self.output_dim[i],
                                      self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32))

        #batch_label_sbbox = np.zeros((batch_size, self.output_dim[0], self.output_dim[0],
        #                              self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)
        #batch_label_mbbox = np.zeros((batch_size, self.output_dim[1], self.output_dim[1],
        #                              self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)
        #batch_label_lbbox = np.zeros((batch_size, self.output_dim[2], self.output_dim[2],
        #                              self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)

        batch_boxes = []
        for _ in range(self.num_scales):
            batch_boxes.append(np.zeros((batch_size, self.max_detections_per_scale, 4), dtype=np.float32))

        #batch_sbboxes = np.zeros((batch_size, self.max_detections_per_scale, 4), dtype=np.float32)
        #batch_mbboxes = np.zeros((batch_size, self.max_detections_per_scale, 4), dtype=np.float32)
        #batch_lbboxes = np.zeros((batch_size, self.max_detections_per_scale, 4), dtype=np.float32)


        for i, (gt_boxes, cls_ids) in enumerate(zip(batch_gt_boxes, batch_cls_ids)):

            #label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self._encode_sample(gt_boxes, cls_ids)
            labels, boxes = self._encode_sample(gt_boxes, cls_ids)

            for j in range(len(labels)):
                batch_labels[j][i, :, :, :, :] = labels[j]
                batch_boxes[j][i, :, :] = boxes[j]

            #batch_label_sbbox[i, :, :, :, :] = label_sbbox
            #batch_label_mbbox[i, :, :, :, :] = label_mbbox
            #batch_label_lbbox[i, :, :, :, :] = label_lbbox
            #batch_sbboxes[i, :, :] = sbboxes
            #batch_mbboxes[i, :, :] = mbboxes
            #batch_lbboxes[i, :, :] = lbboxes  

        #batch_small_target = batch_label_sbbox, batch_sbboxes
        #batch_medium_target = batch_label_mbbox, batch_mbboxes
        #batch_large_target = batch_label_lbbox, batch_lbboxes

        batch_targets = []
        for batch_label, batch_box in zip(batch_labels, batch_boxes):
            batch_targets.append((batch_label, batch_box))

        #print("batch_label_sbbox", batch_label_sbbox)
        #print("batch_sbboxes", batch_sbboxes)

        return batch_images, batch_targets #(batch_small_target, batch_medium_target, batch_large_target)          




class Decoder:

    def __init__(self, config): #, **kwargs):
        #super(Decoder, self).__init__(**kwargs)
        self.num_classes = config["arch"]["num_classes"]
        self.xy_scales = config["arch"]["xy_scales"]
        self.strides = config["arch"]["strides"]
        self.anchors = config["arch"]["anchors"]
        self.anchors_per_scale = config["arch"]["anchors_per_scale"]


    def __call__(self, conv_outputs):

        decoded_fm = []
        for i, conv in enumerate(conv_outputs):
            #print("conv", conv)
            conv_shape = tf.shape(conv)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]

            # last dimension is box predictions (4) + objectness (1) + conditional_class_probabilities (`num_classes`)
            # basically this line just reshapes the volume so that there is an axis for the anchors

            # total number of output values for this level of the pyramid is output_size * output_size * anchors per cell * (5 + num_classes)
            # We predict class and objectness for every anchor box. (4 regression values, 1 objectness value, `num_class` conditional class probabilities)
            conv_output = tf.reshape(conv, (batch_size, output_size, output_size, self.anchors_per_scale, 5 + self.num_classes))

            conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, self.num_classes), axis=-1)

            x = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=0), [output_size, 1])
            y = tf.tile(tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=1), [1, output_size])
            xy_grid = tf.expand_dims(tf.stack([x, y], axis=-1), axis=2)     # [gx, gy, 1, 2]

            xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)

            # xy_grid contains the offsets of each cell for this level of the pyramid
            # ignoring batch size, each offset will appear 3 times, corresponding to the 3 anchors

            # let tx, ty, tw, th be the coordinates predicted for the bounding box

            
            # if the cell is offset from the top left corner of the image by cx, cy
            # bx = sigmoid(tx) + cx
            # by = sigmoid(ty) + cy
            # before multiplying by strides[i] we have the predicted absolute x, y coordinates within the feature map
            # after multiplying by strides[i], we have the predicted absolute x, y coordinates for the original input image
            pred_xy = ((tf.sigmoid(conv_raw_dxdy) * self.xy_scales[i]) - 0.5 * (self.xy_scales[i] - 1) + xy_grid) * self.strides[i]
           
            # if the bounding box prior has width pw and height ph, then the predictions correspond to
            # bw = pw * exp(tw)
            # bh = ph * exp(th)
            pred_wh = tf.exp(conv_raw_dwdh) * self.anchors[i]

            # max_scale == size of network input. The predicted box cannot be larger than the input image size
            max_scale = tf.cast(output_size * self.strides[i], dtype=tf.float32)
            pred_wh = tf.clip_by_value(pred_wh, clip_value_min=0.0, clip_value_max=max_scale)
            pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

            # We define confidence as Pr(Object) * IoU(truth, pred). There is one confidence value per predicted box.
            pred_conf = tf.sigmoid(conv_raw_conf)

            # Pr(Class_i | Object). For each predicted box, one conditional probability is predicted per class.
            pred_prob = tf.sigmoid(conv_raw_prob)

            decoded_fm.append(tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1))
        return decoded_fm





