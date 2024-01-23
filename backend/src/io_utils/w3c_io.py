import uuid
import os
import glob

import math as m
import numpy as np

from io_utils import xml_io, json_io


def convert_json_annotations(annotations, class_map):


    
    prefix_len = len("xywh=pixel:")

    #boxes = {}
    #classes = {}
    ret_annotations = {}

    for image_name in annotations.keys():
        ret_annotations[image_name] = {
            "status": annotations[image_name]["status"],
            #"available_for_training": annotations[image_name]["available_for_training"],
            "boxes": [],
            "classes": [],
            # "update_time": annotations[image_name]["update_time"]
        }
        #boxes[image_name] = []
        #classes[image_name] = []
        for annotation in annotations[image_name]["annotations"]:
            px_str = annotation["target"]["selector"]["value"]
            cls_str = annotation["body"][0]["value"]
            xywh = [int(round(float(x))) for x in px_str[prefix_len:].split(",")]

            min_y = xywh[1]
            min_x = xywh[0]
            max_y = min_y + xywh[3]
            max_x = min_x + xywh[2]
            ret_annotations[image_name]["boxes"].append([min_y, min_x, max_y, max_x])
            ret_annotations[image_name]["classes"].append(class_map[cls_str])



        ret_annotations[image_name]["boxes"] = np.array(ret_annotations[image_name]["boxes"])
        ret_annotations[image_name]["classes"] = np.array(ret_annotations[image_name]["classes"])

    return ret_annotations #boxes, classes


def load_annotations(annotations_path, class_map):

    annotations = json_io.load_json(annotations_path)
    return convert_json_annotations(annotations, class_map)


def load_predictions(predictions_path, class_map):

    prefix_len = len("xywh=pixel:")

    ret_predictions = {}

    predictions = json_io.load_json(predictions_path)
    for image_name in predictions.keys():
        ret_predictions[image_name] = {
            "boxes": [],
            "classes": [],
            "scores": []
        }

        for annotation in predictions[image_name]["annotations"]:

            pred_class = class_map[annotation["body"][0]["value"]]
            pred_score = float(annotation["body"][1]["value"])

            px_str = annotation["target"]["selector"]["value"]
            xywh = [int(round(float(x))) for x in px_str[prefix_len:].split(",")]

            min_y = xywh[1]
            min_x = xywh[0]
            max_y = min_y + xywh[3]
            max_x = min_x + xywh[2]
            ret_predictions[image_name]["boxes"].append([min_y, min_x, max_y, max_x])
            ret_predictions[image_name]["classes"].append(pred_class)
            ret_predictions[image_name]["scores"].append(pred_score)



        ret_predictions[image_name]["boxes"] = np.array(ret_predictions[image_name]["boxes"])
        ret_predictions[image_name]["classes"] = np.array(ret_predictions[image_name]["classes"])
        ret_predictions[image_name]["scores"] = np.array(ret_predictions[image_name]["scores"])

    return ret_predictions
            
def save_predictions(predictions_path, image_predictions, config):

    annotations = {}
    reverse_class_map = {v: k for k, v in config["arch"]["class_map"].items()}

    for image_name in image_predictions.keys():

        annotations[image_name] = {}
        annotations[image_name]["annotations"] = []

        for i in range(len(image_predictions[image_name]["pred_image_abs_boxes"])):

            annotation_uuid = str(uuid.uuid4())

            pred_class_num = image_predictions[image_name]["pred_classes"][i]
            pred_score = image_predictions[image_name]["pred_scores"][i]

            min_y = image_predictions[image_name]["pred_image_abs_boxes"][i][0]
            min_x = image_predictions[image_name]["pred_image_abs_boxes"][i][1]
            max_y = image_predictions[image_name]["pred_image_abs_boxes"][i][2]
            max_x = image_predictions[image_name]["pred_image_abs_boxes"][i][3]

            h = max_y - min_y
            w = max_x - min_x

            box_str = ",".join([str(min_x), str(min_y), str(w), str(h)])

            w3c_annotation = {
                "type": "Annotation",
                "body": [{
                    "type": "TextualBody",
                    "purpose": "class",
                    "value": reverse_class_map[pred_class_num]
                },
                {
                    "type": "TextualBody",
                    "purpose": "score",
                    "value": "%.2f" % pred_score
                }],
                "target": {
                    "source": "",
                    "selector": {
                        "type": "FragmentSelector",
                        "conformsTo": "http://www.w3.org/TR/media-frags/",
                        "value": "xywh=pixel:" + box_str
                    }
                },
                "@context": "http://www.w3.org/ns/anno.jsonld",
                "id": annotation_uuid
            }


            annotations[image_name]["annotations"].append(w3c_annotation)


    json_io.save_json(predictions_path, annotations)




def convert_xml_files_to_w3c(xml_dir, class_map):

    xml_files = os.listdir(xml_dir)

    reverse_class_map = {v:k for k,v in class_map.items()}

    res = {}

    for xml_file in xml_files:

        extensionless_name = os.path.basename(xml_file)[0]
        res[extensionless_name] = []

        xml_path = os.path.join(xml_dir, xml_file)

        boxes, classes = xml_io.load_boxes_and_classes(xml_path, class_map)

        for (box, cls_num) in zip(boxes, classes):

            if reverse_class_map[cls_num] == "plant":
                annotation_uuid = str(uuid.uuid4())

                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                box_str = ",".join([str(box[1]), str(box[0]), str(box_w), str(box_h)])

                w3c_annotation = {
                    "type": "Annotation",
                    "body": [{
                        "type": "TextualBody",
                        "purpose": "class",
                        "value": reverse_class_map[cls_num]
                    }],
                    "target": {
                        "source": "",
                        "selector": {
                            "type": "FragmentSelector",
                            "conformsTo": "http://www.w3.org/TR/media-frags/",
                            "value": "xywh=pixel:" + box_str
                        }
                    },
                    "@context": "http://www.w3.org/ns/anno.jsonld",
                    "id": annotation_uuid
                }

                res[extensionless_name].append(w3c_annotation)
            else:
                print("skipping weed")

    return res


def get_annotation_stats(username):
    num_annotations = 0
    num_completed_images = 0
    num_image_sets = 0
    #image_set_root = os.path.join("usr", "data", "image_sets")
    for username_path in glob.glob(os.path.join("usr", "data", "*")):
        if os.path.basename(username_path) == username:
            for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
                farm_name = os.path.basename(farm_path)
                for field_path in glob.glob(os.path.join(farm_path, "*")):
                    field_name = os.path.basename(field_path)
                    for mission_path in glob.glob(os.path.join(field_path, "*")):
                        num_image_sets += 1
                        mission_date = os.path.basename(mission_path)
                        annotations_path = os.path.join(mission_path, "annotations", "annotations_w3c.json")
                        annotations = load_annotations(annotations_path, {"plant": 0})
                        completed_images = get_completed_images(annotations)
                        num_completed_images += len(completed_images)
                        for image_name in annotations.keys(): #completed_images:
                            num_annotations += annotations[image_name]["boxes"].shape[0]

    return {
        "num_image_sets": num_image_sets,
        "num_completed_images": num_completed_images,
        "num_annotations": num_annotations
    }

