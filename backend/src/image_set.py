import logging
import os
import imagesize
import cv2
import requests

from io_utils import json_io, exif_io






class Image(object):

    def __init__(self, image_path):

        self.image_name = os.path.basename(image_path).split(".")[0]
        self.image_path = image_path


    def load_image_array(self):
        logger = logging.getLogger(__name__)

        image_array = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if image_array.ndim == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return image_array

        

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
        res = requests.get(request)

        ground_elevation = float(res.json()["results"][0]["elevation"])
        
        height_m = (gps_altitude - ground_elevation)

        return height_m

    def get_gsd(self, metadata, username, camera_height):

        make = metadata["EXIF:Make"]
        model = metadata["EXIF:Model"]

        cameras = json_io.load_json(os.path.join("usr", "data", username, "cameras", "cameras.json"))

        specs = cameras[make][model]
        sensor_width = float(specs["sensor_width"])
        sensor_height = float(specs["sensor_height"])
        focal_length = float(specs["focal_length"])

        image_width, image_height = self.get_wh()


        gsd_h = (camera_height * sensor_height) / (focal_length * image_height)
        gsd_w = (camera_height * sensor_width) / (focal_length * image_width)

        gsd = min(gsd_h, gsd_w)

        return gsd
    


    def get_area_m2(self, metadata, username, camera_height):
        gsd = self.get_gsd(metadata, username, camera_height)
        image_width, image_height = self.get_wh()
        image_width_m = image_width * gsd
        image_height_m = image_height * gsd
        area_m2 = image_width_m * image_height_m
        return area_m2

