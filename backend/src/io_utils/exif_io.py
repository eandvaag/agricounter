
import exiftool
import json

EXIF_TAG_ERROR = "ExifTool:Error"




def get_exif_metadata(file_path, tags_to_grab=None, raise_on_tags_missing=True):
    with exiftool.ExifTool() as et:
        try:
            if tags_to_grab:
                metadata = et.get_tags(tags_to_grab, file_path)
                if raise_on_tags_missing and not set(tags_to_grab).issubset(metadata.keys()):
                    raise RuntimeError("Tags {} not all found within exif for file {}".format(
                                        tags_to_grab, file_path))
            else:
                metadata = et.get_metadata(file_path)
        except json.decoder.JSONDecodeError:
            raise FileNotFoundError(file_path)
    if EXIF_TAG_ERROR in metadata:
        raise RuntimeError(
            "file at path {} is not a valid image (exiftool did not recognize it).".format(file_path))

    return metadata

