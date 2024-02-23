from abc import ABC
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage

import models.common.box_utils as box_utils
import models.common.data_augment as data_augment

from models.yolov4.encode import LabelEncoder

from io_utils import tf_record_io


MAX_MEMORY_LIMIT = 85.0


class DataLoader(ABC):


    def __init__(self, tf_record_paths, config):
        self.tf_record_paths = tf_record_paths
        self.input_image_shape = config["arch"]["input_image_shape"]

    def _image_preprocess(self, image):
        ratio = np.array(image.shape[:2]) / np.array(self.input_image_shape[:2])
        image = tf.image.resize(images=image, size=self.input_image_shape[:2])
        #print("image shape", tf.shape(image))
        return image, ratio

    def _box_preprocess(self, boxes):

        #resize_ratio = [self.input_image_shape[0] / h, self.input_image_shape[1] / w]

        boxes = tf.math.round(
            tf.stack([
                boxes[:, 0] * self.input_image_shape[1], #resize_ratio[1],
                boxes[:, 1] * self.input_image_shape[0], #resize_ratio[0],
                boxes[:, 2] * self.input_image_shape[1], #resize_ratio[1],
                boxes[:, 3] * self.input_image_shape[0], #resize_ratio[0]

            ], axis=-1)
        )
        boxes = box_utils.convert_to_xywh_tf(boxes)
        return boxes




    def get_model_input_shape(self):
        return self.input_image_shape
    



class InferenceDataLoader:

    def __init__(self, patch_records, config):
        self.input_image_shape = config["arch"]["input_image_shape"]
        self.batch_size = config["inference"]["batch_size"]
        self.patch_records = patch_records

    def get_model_input_shape(self):
        return self.input_image_shape

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(self.patch_records)))
        dataset = dataset.batch(batch_size=self.batch_size)
        autotune = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(autotune)
        return dataset

    def read_batch_data(self, batch_data):
        batch_images = []
        batch_ratios = []
        batch_indices = []

        for index in batch_data:
            sample_record = self.patch_records[index.numpy()]
            image, ratio = self._preprocess(sample_record)
            batch_images.append(image)
            batch_ratios.append(ratio)
            batch_indices.append(index)

        batch_images = tf.stack(batch_images, axis=0)
        return batch_images, batch_ratios, batch_indices


    def _image_preprocess(self, image):
        ratio = np.array(image.shape[:2]) / np.array(self.input_image_shape[:2])
        image = tf.image.resize(images=image, size=self.input_image_shape[:2])
        #print("image shape", tf.shape(image))
        return image, ratio

    def _preprocess(self, sample_record):
        # print("Sample record", sample_record)
        if "patch" in sample_record:
            patch = sample_record["patch"]
        else:
            patch = (np.array(PILImage.open(sample_record["patch_path"]))).astype(np.uint8)

        image = tf.cast(patch, dtype=tf.float32)

        image, ratio = self._image_preprocess(image)
        return image, ratio




class TrainDataLoader(DataLoader):

    def __init__(self, tf_record_paths, config, shuffle, augment):

        super().__init__(tf_record_paths, config)
        self.batch_size = config["training"]["active"]["batch_size"]
        #self.max_detections = config.arch["max_detections"]
        self.label_encoder = LabelEncoder(config)
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config["training"]["active"]["data_augmentations"]
        #self.pct_of_training_set_used = config.pct_of_training_set_used


    def create_batched_dataset(self):

        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)

        dataset_size = np.sum([1 for _ in dataset])
        if self.shuffle:
            dataset = dataset.shuffle(dataset_size, reshuffle_each_iteration=True)

        num_images = np.sum([1 for _ in dataset])

        dataset = dataset.batch(batch_size=self.batch_size)

        autotune = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(autotune)

        # for i, batch_data in enumerate(dataset):

        #     for tf_sample in batch_data:
        #         sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
        #         image_path = bytes.decode((sample["patch_path"]).numpy())
        #         boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
        #         print("sample: {} {}".format(image_path, boxes))
        #     if i == 0:
        #         break

        return dataset, num_images


    def read_batch_data(self, batch_data):

        batch_images = []
        batch_boxes = []
        batch_classes = []

        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            image, boxes, classes = self._preprocess(sample)
            batch_images.append(image)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

        batch_images = tf.stack(values=batch_images, axis=0)
        #batch_boxes = tf.stack(batch_boxes, axis=0)
        #batch_classes = tf.stack(batch_classes, axis=0)

        return self.label_encoder.encode_batch(batch_images, batch_boxes, batch_classes)
        

    def _preprocess(self, sample):
        image_path = bytes.decode((sample["patch_path"]).numpy())
        image = (np.array(PILImage.open(image_path))).astype(np.uint8)
        #image = tf.io.read_file(filename=image_path)
        #image = tf.image.decode_image(contents=image, channels=3, dtype=tf.dtypes.float32)

        h, w = image.shape[:2]
        #boxes = tf.cast(box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_abs_boxes"]), shape=(-1, 4))), tf.float32)
        boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
        
        # classes = np.zeros(shape=(tf.shape(boxes)[0]))
        # classes = tf.sparse.to_dense(sample["patch_classes"]).numpy().astype(np.float32)
        classes = tf.sparse.to_dense(sample["patch_classes"]).numpy()

        if self.augment:
            image, boxes, classes = data_augment.apply_augmentations(self.data_augmentations, image, boxes, classes)


        # remove boxes with width or height less than 1 pixel
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        mask = np.logical_and(w > (1 / 416), h > (1 / 416))
        boxes = boxes[mask]
        classes = classes[mask]


        image = tf.convert_to_tensor(image, dtype=tf.float32)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.uint8) #tf.float32)


        image, _ = self._image_preprocess(image)
        boxes = self._box_preprocess(boxes)

        #num_boxes = boxes.shape[0]
        #num_pad_boxes = self.max_detections - num_boxes

        #pad_boxes = np.zeros((num_pad_boxes, 4))
        #pad_classes = np.full(num_pad_boxes, -1)

        #boxes = np.vstack([boxes, pad_boxes]).astype(np.float32)
        #classes = np.concatenate([classes, pad_classes]).astype(np.uint8) #float32)

        return image, boxes, classes

