import numpy as np
import tensorflow as tf


from models.centernet.gaussian import gaussian_radius, draw_umich_gaussian



def gather_feat(feat, idx):
    """
        Gathers values from n-channel feature maps given at raster-order index idx.
        feat: 4d array. Shape is (batch_size, feature_map_height, feature_map_width, feature_map_num_channels).
        idx:  2d array. Shape is (batch_size, max_detections). 
              Each entry of idx contains a raster order index into the respective image.

        This function is used to retrieve values from the offset and size feature maps.
    """
    feat = tf.reshape(feat, shape=(feat.shape[0], -1, feat.shape[-1]))
    idx = tf.cast(idx, dtype=tf.int32)
    feat = tf.gather(params=feat, indices=idx, batch_dims=1)
    return feat




class LabelEncoder:


    def __init__(self, config):
        self.num_classes = config.arch["num_classes"]
        self.max_detections = config.arch["max_detections"]
        self.downsampling_ratio = config.arch["downsampling_ratio"]


    def _encode_sample(self, features_shape, gt_boxes, cls_ids):

        hm = np.zeros(shape=(features_shape[0], features_shape[1], self.num_classes), dtype=np.float32)
        reg = np.zeros(shape=(self.max_detections, 2), dtype=np.float32)
        wh = np.zeros(shape=(self.max_detections, 2), dtype=np.float32)
        reg_mask = np.zeros(shape=(self.max_detections), dtype=np.float32)
        ind = np.zeros(shape=(self.max_detections), dtype=np.float32)

        true_inds = np.where(cls_ids != -1)[0]
        gt_boxes = gt_boxes.numpy()[true_inds]
        cls_ids = cls_ids.numpy()[true_inds]
        for i, (gt_box, cls_id) in enumerate(zip(gt_boxes, cls_ids)):

            gt_box = gt_box / self.downsampling_ratio
            x_min, y_min, x_max, y_max = gt_box
            cls_id = cls_id.astype(np.int32)

            h, w = int(y_max - y_min), int(x_max - x_min)
            radius = max(0, int(gaussian_radius((h, w))))
            x_centre, y_centre = (x_min + x_max) / 2, (y_min + y_max) / 2
            centre_point = np.array([x_centre, y_centre], dtype=np.float32)
            centre_point_int = centre_point.astype(np.int32)

            draw_umich_gaussian(hm[:,:,cls_id], centre_point_int, radius)

            reg[i] = centre_point - centre_point_int
            wh[i] = 1. * w, 1. * h
            reg_mask[i] = 1
            ind[i] = centre_point_int[1] * features_shape[1] + centre_point_int[0]

        return hm, reg, wh, reg_mask, ind


    def encode_batch(self, batch_images, batch_gt_boxes, batch_cls_ids):

        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]
        features_shape = np.array(images_shape[1:3], dtype=np.int32) // self.downsampling_ratio

        batch_hm = np.zeros(shape=(batch_size, features_shape[0], features_shape[1], self.num_classes), dtype=np.float32)
        batch_reg = np.zeros(shape=(batch_size, self.max_detections, 2), dtype=np.float32)
        batch_wh = np.zeros(shape=(batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg_mask = np.zeros(shape=(batch_size, self.max_detections), dtype=np.float32)
        batch_ind = np.zeros(shape=(batch_size, self.max_detections), dtype=np.float32)        



        for i in range(batch_size):

            hm, reg, wh, reg_mask, ind = self._encode_sample(features_shape, batch_gt_boxes[i], batch_cls_ids[i])

            batch_hm[i,:,:,:] = hm
            batch_reg[i,:,:] = reg
            batch_wh[i,:,:] = wh
            batch_reg_mask[i,:] = reg_mask
            batch_ind[i,:] = ind

        return batch_images, [batch_hm, batch_reg, batch_wh, batch_reg_mask, batch_ind]





class Decoder:
    def __init__(self, config, **kwargs):

        super(Decoder, self).__init__(**kwargs)
        self.max_detections = config.arch["max_detections"]
        self.num_classes = config.arch["num_classes"]
        self.score_thresh = config.inference["active"]["score_thresh"]
        self.downsampling_ratio = config.arch["downsampling_ratio"]

    def __call__(self, predictions):

        heatmap, reg, wh = tf.split(value=predictions, num_or_size_splits=[self.num_classes, 2, 2], axis=-1)

        heatmap = tf.math.sigmoid(heatmap)
        batch_size = heatmap.shape[0]

        heatmap = self._heatmap_peaks(heatmap)
        top_k_scores, top_k_inds, top_k_cls_ids, top_k_ys, top_k_xs = self._top_scores(scores=heatmap)
        
        # add offsets to box centres
        if reg is not None:
            reg = gather_feat(feat=reg, idx=top_k_inds)
            top_k_xs = tf.reshape(top_k_xs, shape=(batch_size, self.max_detections, 1)) + reg[:,:,0:1]
            top_k_ys = tf.reshape(top_k_ys, shape=(batch_size, self.max_detections, 1)) + reg[:,:,1:2]
        else:
            top_k_xs = tf.reshape(top_k_xs, shape=(batch_size, self.max_detections, 1)) + 0.5
            top_k_ys = tf.reshape(top_k_ys, shape=(batch_size, self.max_detections, 1)) + 0.5


        cls_ids = tf.cast(tf.reshape(top_k_cls_ids, shape=(batch_size, self.max_detections)), dtype=tf.float32)
        scores = tf.reshape(top_k_scores, shape=(batch_size, self.max_detections))
        wh = gather_feat(feat=wh, idx=top_k_inds)
        boxes = tf.concat(values=[top_k_xs - wh[..., 0:1] / 2,
                                  top_k_ys - wh[..., 1:2] / 2,
                                  top_k_xs + wh[..., 0:1] / 2,
                                  top_k_ys + wh[..., 1:2] / 2], axis=2)

        boxes = tf.stack([
            (boxes[:, :, 0] * self.downsampling_ratio),
            (boxes[:, :, 1] * self.downsampling_ratio),
            (boxes[:, :, 2] * self.downsampling_ratio),
            (boxes[:, :, 3] * self.downsampling_ratio)
         
        ], axis=-1)


        score_mask = tf.cast(scores >= self.score_thresh, tf.int32)
        num_detections = tf.reduce_sum(score_mask, axis=1)


        return [num_detections, boxes, scores, cls_ids]



    def _heatmap_peaks(self, heatmap, pool_size=3):
        heatmap_max = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=1, padding="same")(heatmap)
        keep = tf.cast(tf.equal(heatmap, heatmap_max), tf.float32)
        return heatmap_max * keep

    def _top_scores(self, scores):

        scores_shape = tf.cast(tf.shape(scores), dtype=tf.int32)
        batch_size = scores_shape[0]
        height = scores_shape[1]
        width = scores_shape[2]
        num_classes = scores_shape[3]


        scores = tf.reshape(scores, shape=(batch_size, -1))

        # top_k_inds is of shape (batch_size, self.max_detections)
        top_k_scores, top_k_inds = tf.math.top_k(input=scores, k=self.max_detections, sorted=True)


        top_k_cls_ids = top_k_inds % num_classes

        # get the column of each top k score within its respective image
        top_k_xs = tf.cast((top_k_inds // num_classes) % width, tf.float32)

        # get the rows
        top_k_ys = tf.cast((top_k_inds // num_classes) // width, tf.float32)

        # get raster order index of each top k score within its respective image
        top_k_inds = tf.cast(top_k_ys * tf.cast(width, tf.float32) + top_k_xs, tf.int32)    

        return top_k_scores, top_k_inds, top_k_cls_ids, top_k_ys, top_k_xs