import numpy as np

def scale_image(img, old_min, old_max, new_min, new_max, rint=False, np_type=None):
    """
    :param img: numpy array
    :param old_min: scalar old minimum pixel value
    :param old_max: scalar old maximum pixel value.
    :param new_min: scalar new minimum pixel value
    :param new_max: scalar new maximum pixel value.
    :param rint: Should the resulting image be rounded to the nearest int values? Does not convert dtype.
    :param np_type: Optional new np datatype for the array, e.g. np.uint16. If none, keep current type.
    :raises DivideByZerror error  if old_max  - old_min = 0
    :return: scaled copy of img.
    """
    # equivalent to:
    # img = (new_max - new_min) * (img - old_min) / (old_max - old_min) + new_min
    # see https://stats.stackexchange.com/a/70808/71483 and its comments.

    a = (new_max - new_min) / (old_max - old_min)
    b = new_min - a * old_min
    # This autoconverts to float64, preventing over-/under-flow in most cases.
    img = a * img + b
    if rint:
        img = np.rint(img)
    if np_type:
        img = img.astype(np_type)
    return img



def excess_green(image_array):
    image_array = np.float32(image_array) / 255
    exg_array = (2 * image_array[:,:,1]) - image_array[:,:,0] - image_array[:,:,2]
    return exg_array

