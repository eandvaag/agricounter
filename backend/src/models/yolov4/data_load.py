from abc import ABC
import math as m
import numpy as np
import tensorflow as tf
import cv2
import psutil
import tqdm

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
            patch = cv2.cvtColor(cv2.imread(sample_record["patch_path"]), cv2.COLOR_BGR2RGB)

        image = tf.cast(patch, dtype=tf.float32)

        image, ratio = self._image_preprocess(image)
        return image, ratio



class InferenceDataLoaderOrg(DataLoader):

    def __init__(self, tf_record_paths, config, mask_border_boxes=False):
        super().__init__(tf_record_paths, config)
        self.batch_size = config["inference"]["batch_size"]
        self.mask_border_boxes = mask_border_boxes

    def create_dataset(self):
        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)
        dataset_size = np.sum([1 for _ in dataset])
        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset, dataset_size

    def read_batch_data(self, batch_data, is_annotated):

        batch_images = []
        batch_info = []
        batch_ratios = []
        for tf_sample in batch_data:
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=is_annotated)
            image, ratio = self._preprocess(sample)
            batch_images.append(image)
            batch_ratios.append(ratio)
            batch_info.append(sample)

        batch_images = tf.stack(batch_images, axis=0)
        return batch_images, batch_ratios, batch_info


    def _preprocess(self, sample):
        image_path = bytes.decode((sample["patch_path"]).numpy())

        #print("loading image from path", image_path)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # image = data_augment.apply_inference_transform([image], "CLAHE")[0]

        #image = tf.io.read_file(filename=image_path)
        #image = tf.io.decode_image(contents=image, channels=3, dtype=tf.dtypes.float32)
        
        if self.mask_border_boxes:
            #boxes = (tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4))).numpy().astype(np.float32)
            patch_abs_boxes = tf.reshape(tf.sparse.to_dense(sample["patch_abs_boxes"]), shape=(-1, 4)).numpy().astype(np.int64)
            edge_boxes_mask = box_utils.get_edge_boxes_mask(patch_abs_boxes, image.shape[:2])
            edge_boxes = patch_abs_boxes[edge_boxes_mask]
            non_edge_boxes = patch_abs_boxes[np.logical_not(edge_boxes_mask)]

            mask = np.full(image.shape[:2], False)
            for box in edge_boxes:
                mask[box[0]:box[2]+1, box[1]:box[3]+1] = True
            for box in non_edge_boxes:
                mask[box[0]:box[2]+1, box[1]:box[3]+1] = False


            image[mask] = [0, 0, 0]
            #sample["patch_normalized_boxes"] = sample["patch_normalized_boxes"][edge_boxes_mask]
            sample["masked_box_count"] = non_edge_boxes.shape[0]
            



            
            #classes = tf.sparse.to_dense(sample["patch_classes"]).numpy().astype(np.uint8)

            #image = tf.convert_to_tensor(image, dtype=tf.float32)
            #boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
            #classes = tf.convert_to_tensor(classes, dtype=tf.uint8) #tf.float32)


            # image, _ = self._image_preprocess(image)
            #boxes = self._box_preprocess(boxes)
            
        image = tf.cast(image, dtype=tf.float32)

        image, ratio = self._image_preprocess(image)
        return image, ratio


def get_data_loader(tf_record_paths, config, shuffle, augment):

    dataset = tf.data.TFRecordDataset(filenames=tf_record_paths)

    dataset_size = np.sum([1 for _ in dataset])

    input_image_shape = config["arch"]["input_image_shape"]

    num_bytes = dataset_size * input_image_shape[0] * input_image_shape[1] * 3

    available_bytes = psutil.virtual_memory()[1]

    THRESH = 0.6

    if num_bytes / available_bytes < THRESH:
        return (True, PreLoadedTrainDataLoader(tf_record_paths, config, shuffle=shuffle, augment=augment))
    else:
        return (False, TrainDataLoader(tf_record_paths, config, shuffle=shuffle, augment=augment))



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


    def create_batched_dataset(self, take_percent=100):

        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)

        dataset_size = np.sum([1 for _ in dataset])
        if self.shuffle:
            dataset = dataset.shuffle(dataset_size, reshuffle_each_iteration=True)

        dataset = dataset.take(dataset_size * (take_percent / 100))
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

        image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
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







class PreLoadedTrainDataLoader(DataLoader):

    def __init__(self, tf_record_paths, config, shuffle, augment):

        super().__init__(tf_record_paths, config)
        self.batch_size = config["training"]["active"]["batch_size"]
        #self.max_detections = config.arch["max_detections"]
        self.label_encoder = LabelEncoder(config)
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config["training"]["active"]["data_augmentations"]
        #self.pct_of_training_set_used = config.pct_of_training_set_used
        self.dataset_data = {}

    def create_batched_dataset(self, take_percent=100):

        dataset = tf.data.TFRecordDataset(filenames=self.tf_record_paths)


        for i, tf_sample in enumerate(tqdm.tqdm(dataset, "Creating batched dataset")):
            sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            image_path = bytes.decode((sample["patch_path"]).numpy())

            #print("loading image from path", image_path)
            image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
            boxes = (box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_normalized_boxes"]), shape=(-1, 4)))).numpy().astype(np.float32)
            classes = tf.sparse.to_dense(sample["patch_classes"]).numpy().astype(np.uint8)
            
            
            #image = tf.convert_to_tensor(image, dtype=tf.float32)
            #boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
            #classes = tf.convert_to_tensor(classes, dtype=tf.uint8) #tf.float32)
            #image, _ = self._image_preprocess(image)
            image = tf.image.resize(images=image, size=self.input_image_shape[:2]).numpy().astype(np.uint8)
            #boxes = self._box_preprocess(boxes)

            self.dataset_data[i] = {
                "image": image,
                "boxes": boxes,
                "classes": classes
            }

            if psutil.virtual_memory()[2] > MAX_MEMORY_LIMIT:
                raise RuntimeError("Max memory limit exceeded while loading patches.")
        
        #dataset = tf.data.Dataset.from_tensor_slices(np.arange())


        dataset_size = np.sum([1 for _ in dataset])
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(dataset_size))
        if self.shuffle:
            dataset = dataset.shuffle(dataset_size, reshuffle_each_iteration=True)

        dataset = dataset.take(dataset_size * (take_percent / 100))
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
            #sample = tf_record_io.parse_sample_from_tf_record(tf_sample, is_annotated=True)
            image, boxes, classes = self._preprocess(self.dataset_data[tf_sample.numpy()])
            batch_images.append(image)
            batch_boxes.append(boxes)
            batch_classes.append(classes)

        batch_images = tf.stack(values=batch_images, axis=0)
        #batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        #batch_boxes = tf.stack(batch_boxes, axis=0)
        #batch_boxes = tf.convert_to_tensor(batch_boxes, dtype=tf.float32)
        #batch_boxes = self._batch_box_preprocess(batch_boxes)
        #batch_classes = tf.convert_to_tensor(batch_classes, dtype=tf.uint8)



        return self.label_encoder.encode_batch(batch_images, batch_boxes, batch_classes)

    
    def _preprocess(self, data):
        #image_path = bytes.decode((sample["patch_path"]).numpy())

        #data = self.dataset_data[sample_num]
        image = data["image"]
        boxes = data["boxes"]
        classes = data["classes"]

        # #image = (cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)).astype(np.uint8)
        # #image = self.dataset_images[image_path]
        # #image = tf.io.read_file(filename=image_path)
        # #image = tf.image.decode_image(contents=image, channels=3, dtype=tf.dtypes.float32)

        # h, w = image.shape[:2]
        # #boxes = tf.cast(box_utils.swap_xy_tf(tf.reshape(tf.sparse.to_dense(sample["patch_abs_boxes"]), shape=(-1, 4))), tf.float32)


        if self.augment:
            image, boxes, classes = data_augment.apply_augmentations(self.data_augmentations, image, boxes, classes)


        #print("image.dtype", image.dtype)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.uint8) #tf.float32)


        # image, _ = self._image_preprocess(image)
        boxes = self._box_preprocess(boxes)

        # #num_boxes = boxes.shape[0]
        # #num_pad_boxes = self.max_detections - num_boxes

        # #pad_boxes = np.zeros((num_pad_boxes, 4))
        # #pad_classes = np.full(num_pad_boxes, -1)

        # #boxes = np.vstack([boxes, pad_boxes]).astype(np.float32)
        # #classes = np.concatenate([classes, pad_classes]).astype(np.uint8) #float32)

        return image, boxes, classes

    def _batch_box_preprocess(self, boxes):

        #resize_ratio = [self.input_image_shape[0] / h, self.input_image_shape[1] / w]

        boxes = tf.math.round(
            tf.stack([
                boxes[..., 0] * self.input_image_shape[1], #resize_ratio[1],
                boxes[..., 1] * self.input_image_shape[0], #resize_ratio[0],
                boxes[..., 2] * self.input_image_shape[1], #resize_ratio[1],
                boxes[..., 3] * self.input_image_shape[0], #resize_ratio[0]

            ], axis=-1)
        )
        boxes = box_utils.convert_to_xywh_tf(boxes)
        #boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        return boxes





class SplitDataLoader(TrainDataLoader):

    def __init__(self, tf_record_paths_obj, tf_record_paths_bg, config, shuffle, augment):

        super().__init__(tf_record_paths_obj + tf_record_paths_bg, config, shuffle, augment)
        self.tf_record_paths_obj = tf_record_paths_obj
        self.tf_record_paths_bg = tf_record_paths_bg
        self.pct_with_obj = config["training"]["active"]["data_loader"]["percent_of_batch_with_objects"]
        self.label_encoder = LabelEncoder(config)
        self.shuffle = shuffle
        self.augment = augment
        self.data_augmentations = config["training"]["active"]["data_augmentations"]

    def create_batched_dataset(self, take_percent=100):

        num_obj_patches = m.ceil(self.batch_size * (self.pct_with_obj / 100))
        num_bg_patches = self.batch_size - num_obj_patches


        # https://stackoverflow.com/questions/48272035/tensorflow-how-to-generate-unbalanced-combined-data-sets
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


        dataset_obj = dataset_obj.batch(batch_size=num_obj_patches)
        dataset_bg = dataset_bg.batch(batch_size=num_bg_patches)

        zipped_dataset = tf.data.Dataset.zip((dataset_obj, dataset_bg))
        split_dataset = zipped_dataset.map(concat)

        autotune = tf.data.experimental.AUTOTUNE
        split_dataset = split_dataset.prefetch(autotune)

        return split_dataset, num_images
