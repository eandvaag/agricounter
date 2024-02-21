import os
import imagesize
# import cv2
import numpy as np
from PIL import Image as PILImage

from io_utils import exif_io



class ImageWrapper(object):

    def __init__(self, image_path):
        self.image_name = os.path.basename(image_path).split(".")[0]
        self.image_path = image_path


    def load_image_array(self):
        image_array = (np.array(PILImage.open(self.image_path))).astype(np.uint8)
        # image_array = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        # if image_array.ndim == 3:
        #     image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return image_array

        

    def get_wh(self):
        w, h = imagesize.get(self.image_path)
        return w, h


    def get_metadata(self):
        return exif_io.get_exif_metadata(self.image_path)

