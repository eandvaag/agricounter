import tensorflow as tf




class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        # self._delta = 1
        #
        # if |x| < self._delta:
        #   smoothL1Loss(x) = 0.5 * x**2
        # else:
        #   smoothL1Loss(x) = |x| - 0.5

        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)

        # gamma controls the strength of the modulating factor of the focal loss.
        # The modulating factor reduces the contribution of easy examples.
        # This is important for one-shot detectors which generate a very large
        # number of proposals (i.e., a large number of easy negatives).
        # alpha is used to address class-imbalance. alpha balances the importance
        # of positive / negative examples, whereas gamma balances the importance
        # of easy / hard examples.
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, config):
        # reduction "auto": loss values will be computed for each item in the batch (in parallel), then
        # the average loss is returned (losses are summed, then the sum is divided by the batch size)
        #super(RetinaNetLoss, self).__init__(reduction="sum_over_batch_size", name="RetinaNetLoss")
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self.clf_loss = RetinaNetClassificationLoss(config.arch["alpha"], config.arch["gamma"])
        self.box_loss = RetinaNetBoxLoss(config.arch["delta"])
        self.num_classes = config.arch["num_classes"]

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        # one-hot encoding -- "background" class (-1) will be set to all zeroes (ignore "class" (-2) 
        # is also encoded as all zeroes, we will later ignore these entries)
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self.num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self.clf_loss(cls_labels, cls_predictions)
        box_loss = self.box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        #print("0 clf_loss", clf_loss)
        #print("0 box_loss", box_loss)

        # losses are normalized by the number of anchors assigned to a ground truth box   
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        #print("normalizer", normalizer)
        

        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        #clf_loss = tf.reduce_mean(clf_loss, axis=-1)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        #box_loss = tf.reduce_mean(box_loss, axis=-1)

        #print("1 clf_loss", clf_loss)
        #print("1 box_loss", box_loss)

        loss = clf_loss + box_loss
        #print("loss", loss)
        return loss