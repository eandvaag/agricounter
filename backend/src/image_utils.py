import numpy as np
import scipy.misc
from PIL import Image

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


# def global_contrast_normalization(image_array):

#     frac = 1 / (image_array.shape[0] * image_array.shape[1] * image_array.shape[2])
#     x_bar = frac * np.sum(image_array)

#     s = 1.0
#     lamb = 0
#     denom = np.sqrt(frac * (np.sum(image_array - x_bar) ** 2))
#     epsilon = 10 ** -8
#     normalized_image_array = (s) * ((image_array - x_bar) / max(epsilon, denom))

#     return normalized_image_array


def global_contrast_normalization(image_array, s=1.0, lmda=0.0, epsilon=10^-8): #filename, s, lmda, epsilon):
    X = image_array #np.array(Image.open(filename))

    # replacement for the loop
    X_average = np.mean(X)
    #print('Mean: ', X_average)
    X = X - X_average

    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(X**2))

    X = s * X / max(contrast, epsilon)

    #print(np.min(X))
    #print(np.max(X))

    X = scale_image(X, np.min(X), np.max(X), 0, 255, np_type=np.uint8)
    return X

    ## scipy can handle it
    #scipy.misc.imsave('result.jpg', X)





