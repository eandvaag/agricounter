from abc import ABC
import numpy as np
import tensorflow as tf
import cv2

import models.common.box_utils as box_utils
import models.common.data_augment as data_augment

from models.retinanet.encode import LabelEncoder

from io_utils import tf_record_io



# def preprocess_data(sample, augment):
#     image = sample["image"]
#     bbox = box_utils.swap_xy_tf(sample["objects"]["bbox"])
#     class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

#     if augment:
#         image, bbox = random_flip_horizontal(image, bbox)

#     image, image_shape, _ = resize_and_pad_image(image, jitter=None)
    
#     bbox = tf.stack(
#         [
#             bbox[:, 0] * image_shape[1],
#             bbox[:, 1] * image_shape[0],
#             bbox[:, 2] * image_shape[1],
#             bbox[:, 3] * image_shape[0],
#         ],
#         axis=-1,
#     )
#     bbox = box_utils.convert_to_xywh_tf(bbox)
#     return image, bbox, class_id


# def prepare_dataset_for_training(dataset, config, shuffle, augment):

#     autotune = tf.data.experimental.AUTOTUNE
#     label_encoder = LabelEncoder()

#     dataset = dataset.map(lambda x: preprocess_data(x, augment), num_parallel_calls=autotune)
#     if shuffle:
#         dataset = dataset.shuffle(8 * config.train_batch_size)
#     dataset = dataset.padded_batch(batch_size=config.train_batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
#     dataset = dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
#     dataset = dataset.apply(tf.data.experimental.ignore_errors())
#     dataset = dataset.prefetch(autotune)
#     return dataset



class DataLoader(ABC):

    def __init__(self, tf_record_paths, config):
        super().__init__()
        self.tf_record_paths = tf_record_paths
        #self.input_img_min_side = config.input_img_min_side
        #self.input_img_max_side = config.input_img_max_side
        self.input_img_shape = config.arch["input_img_shape"]
        self.smallest_fmap_stride = 128.0



    def get_model_input_shape(self):

        stride = self.smallest_fmap_stride
        padded_img_shape = tf.cast(tf.math.ceil(np.array(self.input_img_shape[:2]) / stride) * stride, dtype=tf.int32)
        return [*padded_img_shape, self.input_img_shape[2]]

    def _img_preprocess(self, img):

        stride = tf.cast(self.smallest_fmap_stride, dtype=tf.float32)
        patch_wh = tf.cast(tf.shape(img)[:2], dtype=tf.float32)

        img = tf.image.resize(img, self.input_img_shape[:2]) #tf.cast(self.input_img_shape[:2], dtype=tf.int32))
        resized_img_wh = tf.cast(tf.shape(img)[:2], dtype=tf.float32)#, dtype=tf.int32)
        
        padded_img_wh = tf.cast(tf.math.ceil(resized_img_wh / stride) * stride, dtype=tf.int32)
        img = tf.image.pad_to_bounding_box(img, 0, 0, padded_img_wh[0], padded_img_wh[1])   

        
        ratio = patch_wh / resized_img_wh

        #print("patch_wh", patch_wh)
        #print("padded_img_wh", padded_img_wh)
        #print("ratio", ratio)

        return img, resized_img_wh, ratio

    # def _img_preprocess(self, img):

    #     min_side = self.input_img_min_side
    #     max_side = self.input_img_max_side
    #     stride = self.smallest_fmap_stride


    #     img_shape = tf.cast(tf.shape(img)[:2], dtype=tf.float32)

    #     ratio = min_side / tf.reduce_min(img_shape)
    #     if ratio * tf.reduce_max(img_shape) > max_side:
    #     #    ratio = max_side / tf.reduce_max(img_shape)

    #     img_shape = ratio * img_shape
    #     img = tf.image.resize(img, tf.cast(img_shape, dtype=tf.int32))
    #     padded_img_shape = tf.cast(tf.math.ceil(img_shape / stride) * stride, dtype=tf.int32)
    #     img = tf.image.pad_to_bounding_box(img, 0, 0, padded_img_shape[0], padded_img_shape[1])

    #     ratio = [1.0 / ratio, 1.0 / ratio]

    #     return img, img_shape, ratio

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

        #img_raw = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        img, _, ratio = self._img_preprocess(img)
        return img, ratio


class MixedDataLoader(DataLoader):

    def __init__(self, tf_record_paths_obj, tf_record_paths_bg, config, shuffle, augment):

        super().__init__(tf_record_paths_obj + tf_record_paths_bg, config)
        self.tf_record_paths_obj = tf_record_paths_obj
        self.tf_record_paths_bg = tf_record_paths_bg
        self.obj_batch_size = config.training["active"]["batch_size_obj"]
        self.bg_batch_size = config.training["active"]["batch_size_bg"]
        self.label_encoder = LabelEncoder()
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config.training["active"]["data_augmentations"]

    def create_batched_dataset(self, take_percent=100):

        def concat(*tensor_list):
            return tf.concat(tensor_list, axis=0)

        dataset_obj = tf.data.TFRecordDataset(filenames=self.tf_record_paths_obj)
        dataset_bg = tf.data.TFRecordDataset(filenames=self.tf_record_paths_bg)

        dataset_obj_size = np.sum([1 for _ in dataset_obj])
        dataset_bg_size = np.sum([1 for _ in dataset_bg])
        
        if self.shuffle:
            dataset_obj = dataset_obj.shuffle(dataset_obj_size, reshuffle_each_iteration=True)
            dataset_bg = dataset_bg.shuffle(dataset_bg_size, reshuffle_each_iteration=True)


        dataset_obj = dataset_obj.take(dataset_obj_size * (take_percent / 100))
        dataset_bg = dataset_bg.take(dataset_bg_size * (take_percent / 100))

        num_obj_images = np.sum([1 for _ in dataset_obj])
        num_bg_images = np.sum([1 for _ in dataset_bg])

        num_images = num_obj_images + num_bg_images


        dataset_obj = dataset_obj.batch(batch_size=self.obj_batch_size)
        dataset_bg = dataset_bg.batch(batch_size=self.bg_batch_size)

        zipped_dataset = tf.data.Dataset.zip((dataset_obj, dataset_bg))
        mixed_dataset = zipped_dataset.map(concat)

        autotune = tf.data.experimental.AUTOTUNE
        mixed_dataset = mixed_dataset.prefetch(autotune)

        return mixed_dataset, num_images


    def read_batch_data(self, batch_data):

        batch_imgs = []
        batch_boxes = []
        batch_classes = []

        most_boxes = 0 #1 #1?

        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            img, boxes, classes = self._preprocess(sample)
            num_boxes = boxes.shape[0]
            batch_imgs.append(img)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

            if num_boxes > most_boxes:
                most_boxes = num_boxes

        padded_batch_boxes = []
        padded_batch_classes = []
        for boxes, classes in zip(batch_boxes, batch_classes):
            padded_boxes = tf.pad(boxes, [[0, most_boxes - boxes.shape[0]], [0, 0]], "CONSTANT", constant_values=1e-8)
            padded_classes = tf.pad(classes, [[0, most_boxes - boxes.shape[0]]], "CONSTANT", constant_values=-1)
            padded_batch_boxes.append(padded_boxes)
            padded_batch_classes.append(padded_classes)

        batch_imgs = tf.stack(values=batch_imgs, axis=0)
        padded_batch_boxes = tf.stack(padded_batch_boxes, axis=0)
        padded_batch_classes = tf.stack(padded_batch_classes, axis=0)

        return self.label_encoder.encode_batch(batch_imgs, padded_batch_boxes, padded_batch_classes)





    def _box_preprocess(self, resized_img_wh, boxes):
        boxes = tf.stack(
            [
                boxes[:, 0] * resized_img_wh[1],
                boxes[:, 1] * resized_img_wh[0],
                boxes[:, 2] * resized_img_wh[1],
                boxes[:, 3] * resized_img_wh[0]
            ],
            axis=-1
        )
        boxes = box_utils.convert_to_xywh_tf(boxes)
        return boxes


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())
        #img_raw = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        img = (cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
        boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
        classes = tf.sparse.to_dense(sample["patch_classes"]).numpy().astype(np.float32)

        # print("before augment:")
        # print("img.dtype", img.dtype)
        #print("boxes", boxes)
        #print("classes", classes)

        if self.augment:
            img, boxes, classes = data_augment.apply_augmentations(self.data_augmentations, img, boxes, classes)


        # print("after augment:")
        # print("img.dtype", img.dtype)
        #print("boxes", boxes)
        #print("classes", classes)



        img = tf.convert_to_tensor(img, dtype=tf.float32)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.float32)

        img, resized_img_wh, ratio = self._img_preprocess(img)
        boxes = self._box_preprocess(resized_img_wh, boxes)
        #classes = tf.cast(classes, dtype=tf.int32)

        return img, boxes, classes




class TrainDataLoader(DataLoader):

    def __init__(self, tf_record_paths, config, shuffle, augment):
        super().__init__(tf_record_paths, config)
        self.batch_size = config.training["active"]["batch_size"]
        self.label_encoder = LabelEncoder()
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config.training["active"]["data_augmentations"]


    def create_batched_dataset(self, take_percent=100):

        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)
        
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

        most_boxes = 1 #1?

        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            img, boxes, classes = self._preprocess(sample)
            num_boxes = boxes.shape[0]
            batch_imgs.append(img)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

            if num_boxes > most_boxes:
                most_boxes = num_boxes

        padded_batch_boxes = []
        padded_batch_classes = []
        for boxes, classes in zip(batch_boxes, batch_classes):
            padded_boxes = tf.pad(boxes, [[0, most_boxes - boxes.shape[0]], [0, 0]], "CONSTANT", constant_values=1e-8)
            padded_classes = tf.pad(classes, [[0, most_boxes - boxes.shape[0]]], "CONSTANT", constant_values=-1)
            padded_batch_boxes.append(padded_boxes)
            padded_batch_classes.append(padded_classes)

        batch_imgs = tf.stack(values=batch_imgs, axis=0)
        padded_batch_boxes = tf.stack(padded_batch_boxes, axis=0)
        padded_batch_classes = tf.stack(padded_batch_classes, axis=0)

        return self.label_encoder.encode_batch(batch_imgs, padded_batch_boxes, padded_batch_classes)





    def _box_preprocess(self, resized_img_wh, boxes):
        boxes = tf.stack(
            [
                boxes[:, 0] * resized_img_wh[1],
                boxes[:, 1] * resized_img_wh[0],
                boxes[:, 2] * resized_img_wh[1],
                boxes[:, 3] * resized_img_wh[0]
            ],
            axis=-1
        )
        boxes = box_utils.convert_to_xywh_tf(boxes)
        return boxes


    def _preprocess(self, sample):
        img_path = bytes.decode((sample["patch_path"]).numpy())
        #img_raw = tf.io.read_file(filename=img_path)
        #img = tf.io.decode_image(contents=img_raw, channels=self.input_img_channels, dtype=tf.float32, expand_animations=False)
        img = (cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
        boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
        classes = tf.sparse.to_dense(sample["patch_classes"]).numpy().astype(np.float32)

        # print("before augment:")
        # print("img.dtype", img.dtype)
        #print("boxes", boxes)
        #print("classes", classes)

        if self.augment:
            img, boxes, classes = data_augment.apply_augmentations(self.data_augmentations, img, boxes, classes)


        # print("after augment:")
        # print("img.dtype", img.dtype)
        #print("boxes", boxes)
        #print("classes", classes)



        img = tf.convert_to_tensor(img, dtype=tf.float32)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.float32)

        img, resized_img_wh, ratio = self._img_preprocess(img)
        boxes = self._box_preprocess(resized_img_wh, boxes)
        #classes = tf.cast(classes, dtype=tf.int32)

        return img, boxes, classes

