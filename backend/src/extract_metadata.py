import os
import glob
import tqdm
import natsort

from image_wrapper import ImageWrapper
from io_utils import json_io



MULTIPLE_CAMERA_TYPES_MESSAGE = "The images in the image set were captured by several different camera types. This is not allowed."



def extract_metadata(config):

    username = config["username"]
    farm_name = config["farm_name"]
    field_name = config["field_name"]
    mission_date = config["mission_date"]
    object_classes = natsort.natsorted(config["object_classes"])
    camera_height = config["camera_height"]
    is_public = config["is_public"]
    is_ortho = config["is_ortho"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", 
                                 farm_name, field_name, mission_date)



    images_dir = os.path.join(image_set_dir, "images")
    metadata_dir = os.path.join(image_set_dir, "metadata")

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_path = os.path.join(metadata_dir, "metadata.json")

    if os.path.exists(metadata_path):
        raise RuntimeError("Existing metadata file found.")


    image_set_metadata = {
        "camera_height": camera_height,
        "images": {},
        "missing": {
            "latitude": False,
            "longitude": False
        },
        "is_public": is_public,
        "is_ortho": is_ortho,
        "object_classes": object_classes
    }


    image_num = 0
    for image_path in tqdm.tqdm(glob.glob(os.path.join(images_dir, "*")), desc="Extracting metadata"):

        image_name = os.path.basename(image_path).split(".")[0]

        image = ImageWrapper(image_path)

        md = image.get_metadata()

        image_width, image_height = image.get_wh()

        if "EXIF:Make" in md:
            make = md["EXIF:Make"]
        else:
            make = ""
        if "EXIF:Model" in md:
            model = md["EXIF:Model"]
        else:
            model = ""

        if image_num == 0:
            camera_info = {
                "make": make,
                "model": model
            }
            image_set_metadata["camera_info"] = camera_info

        else:

            if make != image_set_metadata["camera_info"]["make"]:
                raise RuntimeError(MULTIPLE_CAMERA_TYPES_MESSAGE)
            
            if model != image_set_metadata["camera_info"]["model"]:
                raise RuntimeError(MULTIPLE_CAMERA_TYPES_MESSAGE)


        if "EXIF:GPSLatitude" in md and "EXIF:GPSLatitudeRef" in md:
            gps_latitude = md["EXIF:GPSLatitude"]
            gps_latitude_ref = md["EXIF:GPSLatitudeRef"]
            if gps_latitude_ref == "S":
                gps_latitude *= -1.0
        else:
            gps_latitude = "unknown"
            image_set_metadata["missing"]["latitude"] = True

        if "EXIF:GPSLongitude" in md and "EXIF:GPSLongitudeRef" in md:
            gps_longitude = md["EXIF:GPSLongitude"]
            gps_longitude_ref = md["EXIF:GPSLongitudeRef"]
            if gps_longitude_ref == "W":
                gps_longitude *= -1.0
        else:
            gps_longitude = "unknown"
            image_set_metadata["missing"]["longitude"] = True


        image_set_metadata["images"][image_name] = {
            "latitude": gps_latitude,
            "longitude": gps_longitude,
            "width_px": image_width,
            "height_px": image_height
        }

        image_num += 1



    json_io.save_json(metadata_path, image_set_metadata)

