import tensorflow as tf

from models.centernet.encode import gather_feat

class CenterNetHeatMapLoss:#(tf.losses.Loss):

    def __init__(self, alpha=2, beta=4):

        #super(CenterNetHeatMapLoss, self).__init__(reduction="none", name="CenterNetHeatMapLoss")
        self._alpha = alpha
        self._beta = beta

    def __call__(self, y_true, y_pred):

        # TODO: add log-loss clipping
        # eps = 1e-15
        # y_pred = tf.max(eps, tf.min(1 - eps, y_pred))

        pos_mask = tf.cast(tf.math.equal(y_true, 1), dtype=tf.float32)
        pos_loss = tf.math.pow(1 - y_pred, self._alpha) * tf.math.log(y_pred) * pos_mask
        pos_loss = tf.math.reduce_sum(pos_loss)

        neg_mask = tf.cast(tf.math.less(y_true, 1), dtype=tf.float32)
        neg_loss = tf.math.pow(1 - y_true, self._beta) * tf.math.pow(y_pred, self._alpha) * tf.math.log(1 - y_pred) * neg_mask
        neg_loss = tf.math.reduce_sum(neg_loss)
        
        num_pos = tf.math.reduce_sum(pos_mask)
        loss = 0
        if num_pos == 0:
            #loss = (-1.0) * neg_loss
            loss = loss - neg_loss
        else:
            #loss = (-1.0 / num_pos) * (pos_loss + neg_loss)
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class CenterNetL1Loss:#(tf.losses.Loss):

    #def __init__(self):
    #
    #    #super(CenterNetRegressionLoss, self).__init__(reduction="", name="CenterNetRegressionLoss")

    def __call__(self, y_true, y_pred, y_true_reg_mask, y_true_indices):

        y_pred = gather_feat(y_pred, y_true_indices)

        mask = tf.tile(tf.expand_dims(y_true_reg_mask, axis=-1), tf.constant([1, 1, 2], dtype=tf.int32))
        loss = tf.math.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
        reg_loss = loss / (tf.math.reduce_sum(mask) + 1e-4)
        return reg_loss





class CenterNetLoss:#(tf.losses.Loss):

    def __init__(self, config):
        #super(CenterNetLoss, self).__init__(reduction="", name="CenterNetLoss")
        self.num_classes = config.arch["num_classes"]

        self.heatmap_loss_obj = CenterNetHeatMapLoss()
        self.offset_loss_obj = CenterNetL1Loss()
        self.size_loss_obj = CenterNetL1Loss()

        self.heatmap_loss_weight = config.arch["heatmap_loss_weight"]
        self.offset_loss_weight = config.arch["offset_loss_weight"]
        self.size_loss_weight = config.arch["size_loss_weight"]



    def __call__(self, y_true, y_pred):

        y_true_hm, y_true_reg, y_true_wh, y_true_reg_mask, y_true_indices = y_true

        y_pred_hm, y_pred_reg, y_pred_wh = tf.split(value=y_pred, num_or_size_splits=[self.num_classes, 2, 2], axis=-1)
        y_pred_hm = tf.clip_by_value(t=tf.math.sigmoid(y_pred_hm), clip_value_min=1e-4, clip_value_max=1.0 - 1e-4)
        
        heatmap_loss = self.heatmap_loss_obj(y_true_hm, y_pred_hm)
        offset_loss = self.offset_loss_obj(y_true_reg, y_pred_reg, y_true_reg_mask, y_true_indices)
        size_loss = self.size_loss_obj(y_true_wh, y_pred_wh, y_true_reg_mask, y_true_indices)

        loss = self.heatmap_loss_weight * heatmap_loss + \
               self.offset_loss_weight * offset_loss + \
               self.size_loss_weight * size_loss

        return loss