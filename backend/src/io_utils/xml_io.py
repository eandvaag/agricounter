from xml.etree import ElementTree as ET
import numpy as np

from models.common import box_utils

def load_boxes_and_classes(xml_path, class_map):

    boxes = []
    classes = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        obj_class = obj.find("name").text
        bnd_box = obj.find("bndbox")
        x_min = int(bnd_box.find("xmin").text)
        y_min = int(bnd_box.find("ymin").text)
        x_max = int(bnd_box.find("xmax").text)
        y_max = int(bnd_box.find("ymax").text)

        box = [y_min, x_min, y_max, x_max]
        boxes.append(box)
        classes.append(class_map[obj_class])

    return np.array(boxes), np.array(classes)

def get_box_counts(xml_paths, class_map):

    box_counts = {k: 0 for k in class_map.keys()}

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            obj_class = obj.find("name").text
            box_counts[obj_class] += 1

    return box_counts


def create_class_map(xml_paths):

    class_map = {}
    class_num = 0

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            obj_class = obj.find("name").text

            if obj_class not in class_map:
                class_map[obj_class] = class_num
                class_num += 1

    return class_map


