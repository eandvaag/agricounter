import logging
import os
import glob
from re import L
import tqdm
import argparse
# import time

from image_set import Image
from io_utils import json_io


# def add_camera_specs(image_set_dir):

#     images_dir = os.path.join(image_set_dir, "images")
#     image_paths = glob.glob(os.path.join(images_dir, "*"))
#     # attempts = 0
#     # while len(image_paths) == 0 and attempts < 10:
#     #     time.sleep(5)
#     #     attempts += 1
#     #     print("sleeping")
    

#     print("{}: Make: {} | Model: {}".format(image_set_dir, make, model))
#     metadata_dir = os.path.join(image_set_dir, "metadata")
#     if not os.path.exists(metadata_dir):
#         os.makedirs(metadata_dir)
#     camera_path = os.path.join(image_set_dir, "metadata", "camera.json")
#     json_io.save_json(camera_path, camera_info)


def extract_metadata(image_set_dir, camera_height=None):

    logger = logging.getLogger(__name__)

    images_dir = os.path.join(image_set_dir, "images")
    metadata_dir = os.path.join(image_set_dir, "metadata")

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    metadata_path = os.path.join(metadata_dir, "metadata.json")

    if os.path.exists(metadata_path):
        raise RuntimeError("Existing metadata file found.")

    if camera_height is None:
        camera_height = ""

    image_set_metadata = {
        "camera_height": camera_height,
        "images": {},
        "missing": {
            # "area_m2": False,
            # "height_m": False,
            "latitude": False,
            "longitude": False
        }
    }

    # image_path = glob.glob(os.path.join(images_dir, "*"))[0]

    # image = Image(image_path)
    # md = image.get_metadata()


    

    image_set_path_pieces = image_set_dir.split("/")
    username = image_set_path_pieces[2]



    image_num = 0
    for image_path in tqdm.tqdm(glob.glob(os.path.join(images_dir, "*")), desc="Extracting metadata"):

        image_name = os.path.basename(image_path).split(".")[0]

        image = Image(image_path)

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
                # raise RuntimeError("Image set contains multiple camera makes. Established make: {}. Conflicting make: {}".format(
                #     image_set_metadata["camera_info"]["make"], make
                # ))
                exit(1)
                # logger.warning("Conflicting camera makes within image set. Established make: {}. Conflicting make: {}".format(
                #     image_set_metadata["camera_info"]["make"], make
                # ))
            if model != image_set_metadata["camera_info"]["model"]:
                # raise RuntimeError("Image set contains multiple camera models. Established model: {}. Conflicting model: {}".format(
                #     image_set_metadata["camera_info"]["model"], model
                # ))
                exit(2)
                # logger.warning("Conflicting camera models within image set. Established model: {}. Conflicting model: {}".format(
                #     image_set_metadata["camera_info"]["model"], model
                # ))




        # if camera_height is None:
        #     area_m2 = "unknown"
        #     image_set_metadata["missing"]["area_m2"] = True
        # else:
        #     try:
        #         area_m2 = image.get_area_m2(md, username, camera_height)
        #     except:
        #         area_m2 = "unknown"
        #         image_set_metadata["missing"]["area_m2"] = True

        # try:
        #     height_m = image.get_height_m(md)
        # except:
        #     height_m = "unknown"
        #     image_set_metadata["missing"]["height_m"]  = True


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
            #"height_m": height_m,
            # "area_m2": area_m2,
            "width_px": image_width,
            "height_px": image_height
        }

        image_num += 1




    

    #print("writing metadata for {}. metadata_missing? {}".format(image_set_dir, image_set_metadata["missing"]))
    #json_io.print_json(image_set_metadata)
    json_io.save_json(metadata_path, image_set_metadata)


def extract_metadata_for_all_image_sets():
    image_set_root = os.path.join("usr", "data", "image_sets")

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                metadata_dir = os.path.join(mission_path, "metadata")
                #if not os.path.exists(metadata_dir):
                #    print("Missing: {} {} {}".format(farm_name, field_name, mission_date))

                extract_metadata(mission_path, camera_height=2)



def tmp_create_lock_files():
    
    image_set_root = os.path.join("usr", "data", "image_sets")

    for farm_path in glob.glob(os.path.join(image_set_root, "*")):
        farm_name = os.path.basename(farm_path)
        for field_path in glob.glob(os.path.join(farm_path, "*")):
            field_name = os.path.basename(field_path)
            for mission_path in glob.glob(os.path.join(field_path, "*")):
                mission_date = os.path.basename(mission_path)

                lock_path = os.path.join(mission_path, "annotations", "lock.json")
                if not os.path.exists(lock_path):
                    json_io.save_json(lock_path, {"last_refresh": 0})
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    parser.add_argument("--camera_height", type=float)

    args = parser.parse_args()
    image_set_dir = args.image_set_dir
    camera_height = args.camera_height

    # print("camera_height: {}".format(camera_height))

    # add_camera_specs(image_set_dir)
    # add_camera_specs_for_all_image_sets()

    extract_metadata(image_set_dir, camera_height=camera_height)
