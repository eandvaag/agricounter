import logging
import os
import shutil
import glob
import tqdm
import imagesize
import numpy as np
import cv2
import json
import requests

from io_utils import json_io, xml_io, w3c_io, exif_io
from models.common import box_utils




class ImgSet(object):
    def __init__(self):
        pass

# class ImgSet(object):

#     def __init__(self, farm_name, field_name, mission_date):

#         usr_data_root = os.path.join("usr", "data")

#         self.farm_name = farm_name
#         self.field_name = field_name
#         self.mission_date = mission_date
#         self.root_dir = os.path.join(usr_data_root, "image_sets", farm_name, field_name, mission_date)
#         self.img_dir = os.path.join(self.root_dir, "images")
#         self.patch_dir = os.path.join(self.root_dir, "patches")
#         self.dzi_dir = os.path.join(self.root_dir, "dzi_images")

#         img_set_data_path = os.path.join(self.root_dir, "image_set_data.json")
#         img_set_data = json_io.load_json(img_set_data_path)
        
#         self.class_map = img_set_data["class_map"]
#         self.reverse_class_map = {v: k for k, v in self.class_map.items()}
#         self.num_classes = img_set_data["num_classes"]
        
#         self.datasets = {
#             "training": DataSet(img_set_data["training_image_paths"], self.class_map, "training"),
#             "validation": DataSet(img_set_data["validation_image_paths"], self.class_map, "validation"),
#             "test": DataSet(img_set_data["test_image_paths"], self.class_map, "test"),
#             "all": DataSet(img_set_data["all_image_paths"], self.class_map, "all")

#         }

        # if (not self.datasets["training"].is_annotated) or \
        #    (not self.datasets["validation"].is_annotated) or \
        #    (not self.datasets["test"].is_annotated):

        #    raise RuntimeError("Training, validation, and test datasets must only contain annotated images.")


        #img_width, img_height = imagesize.get(img_set_info["training_image_paths"][0])
        #self.img_width = img_width
        #self.img_height = img_height
        #self.camera_metadata = img_set_info["camera_metadata"]

    # def get_box_counts(self):

    #     box_counts = {}
    #     for dataset_name, dataset in self.datasets.items():
    #         box_counts[dataset_name] = dataset.get_box_counts(self.class_map)
    #     return box_counts

# class ImageSet(object):
#     def __init__(self, imageset_conf):

#         usr_data_root = os.path.join("usr", "data")

#         self.farm_name = imageset_conf["farm_name"]
#         self.field_name = imageset_conf["field_name"]
#         self.mission_date = imageset_conf["mission_date"]
#         #self.patch_extraction_params = imageset_conf["patch_extraction_params"]

#         self.image_set_root = os.path.join(usr_data_root, "image_sets", 
#                                       self.farm_name, self.field_name, self.mission_date)
        
#         self.annotations_path = os.path.join(self.image_set_root, "annotations", "annotations_w3c.json")
#         self.images_root = os.path.join(self.image_set_root, "images")

#         template = {
#             "farm_name": self.farm_name,
#             "field_name": self.field_name,
#             "mission_date": self.mission_date,
#             #"patch_extraction_params": self.patch_extraction_params
#         }

#         #template["image_names"] = imageset_conf["training_image_names"]
#         self.training_dataset = DataSet(template, selected_image_names=imageset_conf["training_image_names"])
#         #template["image_names"] = imageset_conf["validation_image_names"]
#         self.validation_dataset = DataSet(template, selected_image_names=imageset_conf["validation_image_names"])
#         #template["image_names"] = imageset_conf["test_image_names"]
#         self.test_dataset = DataSet(template, selected_image_names=imageset_conf["test_image_names"])

#         self.all_dataset = DataSet(template) #, all_images=True)        


class DataSet(object):
    def __init__(self, dataset_conf, selected_image_names=[]): #, all_images=False):

        usr_data_root = os.path.join("usr", "data")

        self.farm_name = dataset_conf["farm_name"]
        self.field_name = dataset_conf["field_name"]
        self.mission_date = dataset_conf["mission_date"]
        self.image_set_name = self.farm_name + "-" + self.field_name + "-" + self.mission_date
        #self.patch_extraction_params = dataset_conf["patch_extraction_params"]

        self.image_set_root = os.path.join(usr_data_root, "image_sets", 
                                      self.farm_name, self.field_name, self.mission_date)
        
        self.annotations_path = os.path.join(self.image_set_root, "annotations", "annotations_w3c.json")
        self.images_root = os.path.join(self.image_set_root, "images")

        annotations = w3c_io.load_annotations(self.annotations_path, {"plant": 0})

        self.image_names = []
        for f in glob.glob(os.path.join(self.images_root, "*")):
            self.image_names.append(os.path.basename(f).split(".")[0])
        self.selected_image_names = selected_image_names


        self.images = []
        self.completed_images = []
        self.nonempty_completed_images = []
        self.selected_images = []
        for image_name in self.image_names:
            #print("adding", image_name)
            full_path = glob.glob(os.path.join(self.images_root, image_name + ".*"))[0]
            #print("full_path", full_path)
            image = Image(full_path)
            self.images.append(image)
            if image_name in selected_image_names:
                self.selected_images.append(image)
            if annotations[image_name]["status"] == "completed":
                self.completed_images.append(image)
                if annotations[image_name]["boxes"].size > 0:
                    self.nonempty_completed_images.append(image)





# class DataSet(object):

#     def __init__(self, img_paths, class_map, name):

#         self.imgs = []
#         self.class_map = class_map

#         # self.is_annotated = True
#         for img_path in img_paths:
#             img = Img(img_path)
#             # if not img.is_annotated:
#             #     self.is_annotated = False
#             self.imgs.append(img)
            
#         self.name = name

#     # def get_box_counts(self):
#     #     xml_paths = [img.xml_path for img in self.imgs if img.is_annotated]
#     #     box_counts = xml_io.get_box_counts(xml_paths, self.class_map)
#     #     return box_counts


#     # def get_mean_box_area(self):
#     #     box_areas = []

#     #     for img in self.imgs:
#     #         boxes, _ = xml_io.load_boxes_and_classes(img.xml_path, self.class_map)
#     #         box_areas.extend(box_utils.box_areas_np(boxes).tolist())

#     #     return np.mean(box_areas)

class Image(object):

    def __init__(self, image_path):

        self.image_name = os.path.basename(image_path).split(".")[0]
        self.image_path = image_path


    def load_image_array(self):
        logger = logging.getLogger(__name__)

        image_array = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if image_array.ndim == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return image_array #cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)

        

    def get_wh(self):
        w, h = imagesize.get(self.image_path)
        return w, h


    def get_metadata(self):
        return exif_io.get_exif_metadata(self.image_path)


    def get_height_m(self, metadata):

        gps_altitude = metadata["EXIF:GPSAltitude"]
        gps_latitude = metadata["EXIF:GPSLatitude"]
        gps_longitude = metadata["EXIF:GPSLongitude"]
        gps_latitude_ref = metadata["EXIF:GPSLatitudeRef"]
        gps_longitude_ref = metadata["EXIF:GPSLongitudeRef"]

        if gps_latitude_ref == "S":
            gps_latitude *= -1.0
        if gps_longitude_ref == "W":
            gps_longitude *= -1.0



        request = "https://api.open-elevation.com/api/v1/lookup?locations=" + str(gps_latitude) + "," + str(gps_longitude)
        # print("Request", request)
        res = requests.get(request) #52.36134444444445,-107.94438333333333")

        ground_elevation = float(res.json()["results"][0]["elevation"])
        
        # print("GPS Altitude: {}, ground_elevation: {}".format(gps_altitude, ground_elevation))
        height_m = (gps_altitude - ground_elevation) #* 1000

        return height_m

    def get_gsd(self, metadata, username, camera_height):
        # try:
            
            # print(json.dumps(metadata, indent=4, sort_keys=True))

        make = metadata["EXIF:Make"]
        model = metadata["EXIF:Model"]

        cameras = json_io.load_json(os.path.join("usr", "data", username, "cameras", "cameras.json"))

        specs = cameras[make][model]
        sensor_width = float(specs["sensor_width"]) #[:-2])
        sensor_height = float(specs["sensor_height"]) #[:-2])
        focal_length = float(specs["focal_length"]) #[:-2])

        #camera_height = self.get_height_m(metadata)

        # camera_height = camera_metadata["camera_height_m"]
        # sensor_height = camera_metadata["sensor_height_mm"] * 1000
        # sensor_width = camera_metadata["sensor_width_mm"] * 1000
        # focal_length = camera_metadata["focal_length_mm"] * 1000
        # print("camera_height: {}, sensor_height: {}, sensor_width: {}, focal_length: {}".format(
        #     camera_height, sensor_height, sensor_width, focal_length
        # ))



        image_width, image_height = self.get_wh()
        # print("image_width: {}, image_height: {}".format(image_width, image_height))



        gsd_h = (camera_height * sensor_height) / (focal_length * image_height)
        gsd_w = (camera_height * sensor_width) / (focal_length * image_width)

        # print("gsd_h: {}, gsd_w: {}".format(gsd_h, gsd_w))
        gsd = min(gsd_h, gsd_w)

        return gsd
            
        # except:
        #     raise RuntimeError("Missing EXIF data needed to determine GSD.")


    def get_area_m2(self, metadata, username, camera_height):
        # try:
        gsd = self.get_gsd(metadata, username, camera_height)
        image_width, image_height = self.get_wh()
        image_width_m = image_width * gsd
        image_height_m = image_height * gsd
        #print("Dw: {}, Dh: {}".format(image_width_m, image_height_m))
        area_m2 = image_width_m * image_height_m
        return area_m2
        # except:
        #     raise RuntimeError("Missing EXIF data needed to determine image area.")

