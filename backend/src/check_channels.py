import os
import glob
import argparse
import shutil
import subprocess

import imagesize
from osgeo import gdal
import pyvips

# import time

accepted_ftypes = ["JPEG", "PNG", "TIFF", "Big TIFF"]


ftype_str_to_ext = {
    "JPEG": "jpg",
    "PNG": "png",
    "TIFF": "tif",
    "Big TIFF": "tif"
}

accepted_ftype_strs_for_extension = {
    "jpg": ["JPEG"],
    "jpeg": ["JPEG"],
    "JPG": ["JPEG"],
    "JPEG": ["JPEG"],
    "png": ["PNG"],
    "PNG": ["PNG"],
    "tif": ["TIFF", "Big TIFF"],
    "tiff": ["TIFF", "Big TIFF"],
    "TIF": ["TIFF", "Big TIFF"],
    "TIFF": ["TIFF", "Big TIFF"]
}

def check_channels(image_set_dir, is_ortho):

    # time.sleep(10)

    image_list = glob.glob(os.path.join(image_set_dir, "images", "*"))

    for image_path in image_list:
        image_name = os.path.basename(image_path)
        if "." in image_name:
            image_name_split = image_name.split(".")
            extensionless_name = image_name_split[0]
            extension = image_name_split[1]
        else:
            extension = None

        out = subprocess.check_output(["file", image_path]).decode("utf-8")
        ftype_str = out[len(image_path)+2: ]

        if extension is None:
            chosen_ftype_str = None
            for ftype in accepted_ftypes:
                if ftype_str.startswith(ftype):
                    chosen_ftype_str = ftype
            if chosen_ftype_str is None:
                exit(1)
            else:
                image_name_with_extension = image_name + "." + ftype_str_to_ext[chosen_ftype_str]
                new_image_path = os.path.join(image_set_dir, "images", image_name_with_extension)
                shutil.move(image_path, new_image_path)

        else:
            
            accepted = False
            for accepted_ftype_str in accepted_ftype_strs_for_extension[extension]:
                if ftype_str.startswith(accepted_ftype_str):
                    accepted = True

            if not accepted:
                exit(2)

    image_list = glob.glob(os.path.join(image_set_dir, "images", "*"))

    for image_path in image_list:


        # image_path = old_image_path + ".png"
        # shutil.copy(old_image_path, image_path)
        # image_path = old_image_path
        # subprocess.run(["cp", old_image_path, image_path])
        # os.system("cp " + old_image_path + " " + image_path)
        # os.system("cat < " + old_image_path + " >" + image_path)


        # w, h = imagesize.get(image_path)
        # if w < 1 or h < 1:
        #     exit(-10)

        image = pyvips.Image.new_from_file(image_path, access="sequential")

        if image.hasalpha() == 1:
            image = image.flatten()
            image_name = os.path.basename(image_path)
            image_name_split = image_name.split(".")
            extensionless_name = image_name_split[0]
            extension = image_name_split[1]
            # if extension in ["tif", "TIF", "tiff", "TIFF"] and is_ortho:
            #     kwargs = [compression=lzw]
            #     new_name = extensionless_name + "_vips_no_alpha." + extension + "[compression=lzw, bigtiff]"
            # else:
            new_name = extensionless_name + "_vips_no_alpha." + extension
            
            vips_out_path = os.path.join(image_set_dir, "images", new_name)
            if extension in ["tif", "TIF", "tiff", "TIFF"] and is_ortho:
                image.write_to_file(vips_out_path, compression="lzw", bigtiff=True)
            else:
                image.write_to_file(vips_out_path)
            shutil.move(vips_out_path, image_path)

        # ds = gdal.Open(image_path)

        # image_array = ds.ReadAsArray(0, 0, 1, 1)

        if image.bands != 3: #image_array.shape[0] != 3:
            exit(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_set_dir", type=str)
    parser.add_argument("is_ortho", type=str)
    args = parser.parse_args()

    is_ortho = args.is_ortho == "yes"
    check_channels(args.image_set_dir, is_ortho)