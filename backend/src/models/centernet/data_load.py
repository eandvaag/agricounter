from abc import ABC
import numpy as np
import tensorflow as tf
import cv2

import models.common.box_utils as box_utils
import models.common.data_augment as data_augment

from models.centernet.encode import LabelEncoder

from io_utils import tf_record_io



class DataLoader(ABC):


    def __init__(self, tf_record_paths, config):
        self.tf_record_paths = tf_record_paths
        self.input_img_shape = config.arch["input_img_shape"]

    def _img_preprocess(self, img):
        ratio = np.array(img.shape[:2]) / np.array(self.input_img_shape[:2])
        img = tf.image.resize(images=img, size=self.input_img_shape[:2])
        return img, ratio


class InferenceDataLoader(DataLoader):

    def __init__(self, tf_record_paths, config):
        super().__init__(tf_record_paths, config)
        self.batch_size = config.inference["active"]["batch_size"]

    def create_dataset(self):
        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)
        dataset_size = np.sum([1 for _ in dataset])
        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset, dataset_size

    def read_batch_data(self, batch_data, is_annotated):

        batch_imgs = []
        batch_info = []
        batch_ratios = []
        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=is_annotated)
            img, ratio = self._preprocess(sample)
            batch_imgs.append(img)
            batch_ratios.append(ratio)
            batch_info.append(sample)

        batch_imgs = tf.stack(batch_imgs, axis=0)
        return batch_imgs, batch_ratios, batch_info


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())
        img = tf.cast(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), dtype=tf.float32)

        #img = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img, channels=3, dtype=tf.dtypes.float32)

        img, ratio = self._img_preprocess(img)
        return img, ratio


class TrainDataLoader(DataLoader):

    def __init__(self, tf_record_paths, config, shuffle, augment):

        super().__init__(tf_record_paths, config)
        self.batch_size = config.training["active"]["batch_size"]
        self.max_detections = config.arch["max_detections"]
        self.label_encoder = LabelEncoder(config)
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config.training["active"]["data_augmentations"]
        #self.pct_of_training_set_used = config.pct_of_training_set_used


    def create_batched_dataset(self, take_percent=100):

        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)
        #print("(1) DATASET size is ", np.sum([1 for _ in dataset]))
        #dataset = dataset[:self.pct_of_training_set_used]


        #print("(3) DATASET size is ", np.sum([1 for _ in dataset]))
        #exit()
        dataset_size = np.sum([1 for _ in dataset])
        if self.shuffle:
            dataset = dataset.shuffle(dataset_size, reshuffle_each_iteration=True)

        dataset = dataset.take(dataset_size * (take_percent / 100))
        dataset_size = np.sum([1 for _ in dataset])

        dataset = dataset.batch(batch_size=self.batch_size)

        autotune = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(autotune)

        return dataset, dataset_size


    def read_batch_data(self, batch_data):

        batch_imgs = []
        batch_boxes = []
        batch_classes = []

        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            img, boxes, classes = self._preprocess(sample)
            batch_imgs.append(img)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

        batch_imgs = tf.stack(values=batch_imgs, axis=0)
        batch_boxes = tf.stack(batch_boxes, axis=0)
        batch_classes = tf.stack(batch_classes, axis=0)

        return self.label_encoder.encode_batch(batch_imgs, batch_boxes, batch_classes)


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())

        img = (cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
        #img = tf.io.read_file(filename=img_path)
        #img = tf.image.decode_image(contents=img, channels=3, dtype=tf.dtypes.float32)

        h, w = img.shape[:2]
        #boxes = tf.cast(box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_abs_boxes"]), shape=(-1, 4))), tf.float32)
        boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
        classes = tf.sparse.to_dense(sample["patch_classes"]).numpy().astype(np.float32)

        if self.augment:
            img, boxes, classes = data_augment.apply_augmentations(self.data_augmentations, img, boxes, classes)

        img = tf.convert_to_tensor(img, dtype=tf.float32)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.float32)


        img, _ = self._img_preprocess(img)
        boxes = self._box_preprocess(boxes)

        num_boxes = boxes.shape[0]
        num_pad_boxes = self.max_detections - num_boxes

        pad_boxes = np.zeros((num_pad_boxes, 4))
        pad_classes = np.full(num_pad_boxes, -1)

        boxes = np.vstack([boxes, pad_boxes]).astype(np.float32)
        classes = np.concatenate([classes, pad_classes]).astype(np.float32)

        return img, boxes, classes


    def _box_preprocess(self, boxes):

        #resize_ratio = [self.input_img_shape[0] / h, self.input_img_shape[1] / w]

        boxes = tf.math.round(
            tf.stack([
                boxes[:, 0] * self.input_img_shape[1], #resize_ratio[1],
                boxes[:, 1] * self.input_img_shape[0], #resize_ratio[0],
                boxes[:, 2] * self.input_img_shape[1], #resize_ratio[1],
                boxes[:, 3] * self.input_img_shape[0], #resize_ratio[0]

            ], axis=-1)
        )

        return boxes




