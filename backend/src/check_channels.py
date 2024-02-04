import os
import glob
import shutil
import subprocess
import pyvips


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


BAD_IMAGE_TYPE_MESSAGE = "At least one file is not an accepted image type. Accepted image types are JPEGs, PNGs, and TIFFs. Extensions are optional, but must match the underlying file type if they are included."
BAD_EXTENSION_MESSAGE = "At least one file's extension does not match the true underlying image type. Accepted image types are JPEGs, PNGs, and TIFFs. Extensions are optional, but must match the underlying file type if they are included.";
BAD_CHANNEL_COUNT_MESSAGE = "At least one image contains an invalid number of channels. Only RGB images can be uploaded (with optional alpha channel)."

def check_channels(image_set_dir, is_ortho):

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
                raise RuntimeError(BAD_IMAGE_TYPE_MESSAGE)
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
                raise RuntimeError(BAD_EXTENSION_MESSAGE)

    image_list = glob.glob(os.path.join(image_set_dir, "images", "*"))

    for image_path in image_list:

        image = pyvips.Image.new_from_file(image_path, access="sequential")

        if image.hasalpha() == 1:
            image = image.flatten()
            image_name = os.path.basename(image_path)
            image_name_split = image_name.split(".")
            extensionless_name = image_name_split[0]
            extension = image_name_split[1]
            new_name = extensionless_name + "_vips_no_alpha." + extension
            
            vips_out_path = os.path.join(image_set_dir, "images", new_name)
            if extension in ["tif", "TIF", "tiff", "TIFF"] and is_ortho:
                image.write_to_file(vips_out_path, compression="lzw", bigtiff=True)
            else:
                image.write_to_file(vips_out_path)
            shutil.move(vips_out_path, image_path)

        if image.bands != 3:
            raise RuntimeError(BAD_CHANNEL_COUNT_MESSAGE)
        



    
