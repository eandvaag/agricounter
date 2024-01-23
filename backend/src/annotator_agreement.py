import os
import math as m
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


import krippendorff
from exp_runner import my_plot_colors
from io_utils import json_io
from models.common import annotation_utils, box_utils


def get_krippendorff(annotation_files, image_names):
    data = []
    for annotation_file in annotation_files:
        l = []
        for image_name in image_names:
            l.append((annotation_file[image_name]["boxes"]).shape[0])

        data.append(l)

    data = np.array(data)
    return krippendorff.krippendorff_alpha(data, metric=krippendorff.ratio_metric)


def get_krippendorff_patches(annotation_files, image_names):

    image_width = 5472
    image_height = 3648
    patch_size = 416
    data = []
    for annotation_file in annotation_files:
        l = []
        for image_name in image_names:
            # print(image_name)

            col_covered = False
            patch_min_y = 0
            while not col_covered:
                patch_max_y = patch_min_y + patch_size
                max_content_y = patch_max_y
                if patch_max_y >= image_height:
                    max_content_y = image_height
                    col_covered = True

                row_covered = False
                patch_min_x = 0
                while not row_covered:

                    patch_max_x = patch_min_x + patch_size
                    max_content_x = patch_max_x
                    if patch_max_x >= image_width:
                        max_content_x = image_width
                        row_covered = True

                    
                    patch_coords = [patch_min_y, patch_min_x, patch_max_y, patch_max_x]
                    # print("\t{}".format(patch_coords))
                    inds = box_utils.get_contained_inds(annotation_file[image_name]["boxes"], [patch_coords])
                    l.append(inds.size)


                    patch_min_x += (patch_size) # - overlap_px)

                patch_min_y += (patch_size) # - overlap_px)                    


            # l.append((annotation_file[image_name]["boxes"]).shape[0])

        data.append(l)

    data = np.array(data)
    # print(data)
    return krippendorff.krippendorff_alpha(data, metric=krippendorff.interval_metric) #ratio_metric)



def get_gtc(boxes):

    # iou_mat = box_utils.compute_iou(boxes, boxes, box_format="corners_yx")

    boxes1_corners = boxes
    boxes2_corners = boxes

    boxes1_area = (boxes1_corners[:,2] - boxes1_corners[:,0]) * (boxes1_corners[:,3] - boxes1_corners[:,1])
    boxes2_area = (boxes2_corners[:,2] - boxes2_corners[:,0]) * (boxes2_corners[:,3] - boxes2_corners[:,1])

    lu = np.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = np.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    union_area = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )

    union_sum = 0
    intersection_sum = 0
    for i in range(union_area.shape[0]):
        for j in range(i, union_area.shape[1]):
            if i == j:
                continue

            union_sum += union_area[i][j]
            intersection_sum += intersection_area[i][j]

    gtc = intersection_sum / union_sum
    return gtc


def get_dbscan(annotation_files, image_names):

    majority = 3 #m.ceil(len(annotation_files) / 2)
    print("majority", majority)

    num_noise = 0
    num_boxes = 0
    gtcs = []
    for image_name in image_names:
        X = []
        all_boxes = []
        for annotation_file in annotation_files:
            boxes = annotation_file[image_name]["boxes"]
            centres = (boxes[..., :2] + boxes[..., 2:]) / 2.0
            X.extend(centres.tolist())
            all_boxes.extend(boxes.tolist())


        X = np.array(X)
        all_boxes = np.array(all_boxes)

        db = DBSCAN(eps=5, min_samples=majority).fit(X)
        labels = db.labels_

        num_noise_in_image = list(labels).count(-1)
        num_boxes_in_image = len(X)

        num_noise += num_noise_in_image
        num_boxes += num_boxes_in_image

        # fraction_belonging_to_cluster = (num_total - num_noise) / num_total

        for label in set(labels):
            if label == -1:
                continue
            
            inds = np.where(labels == label)[0]
            cluster_boxes = all_boxes[inds]
            gtc = get_gtc(cluster_boxes)
            gtcs.append(gtc)

            
    fraction_belonging_to_cluster = (num_boxes - num_noise) / num_boxes
    ave_gtc = np.mean(gtcs)

    return fraction_belonging_to_cluster, ave_gtc

def print_counts(annotation_files, image_names):
    for image_name in image_names:
        box_counts = []
        for annotation_file in annotation_files:
            box_count = annotation_file[image_name]["boxes"].shape[0]
            box_counts.append(box_count)
        print("{}\t\t: {}".format(image_name, box_counts))



def count_chart(anno_files, out_dir):
    lookup = {
        "BlaineLake/BlaineLake3/2021-06-09": ["8", "51"],
        "Uni/Uni9/2022-05-30": ["UNI_CNH_May30_318", "UNI_CNH_May30_219"]
    }
    color_lookup = {
        "BlaineLake/BlaineLake3/2021-06-09": my_plot_colors[0],
        "Uni/Uni9/2022-05-30": my_plot_colors[1]
    }
    label_lookup = {
        "8": "BlaineLake/BlaineLake3/2021-06-09", 
        "51": None, 
        "UNI_CNH_May30_318": "Uni/Uni9/2022-05-30", 
        "UNI_CNH_May30_219": None
    }

    # box_counts = {}
    data = []
    for image_set in lookup.keys():
        for image_name in lookup[image_set]:
            anno_counts = []
            for anno_file in anno_files:
                anno_count = anno_file[image_name]["boxes"].shape[0]
                anno_counts.append(anno_count)
            data.append((image_set, image_name, anno_counts))
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i in range(len(data)):
        ax.scatter(data[i][2], [i] * len(data[i][2]), label=label_lookup[data[i][1]],
                   color=color_lookup[data[i][0]],
                    marker="o", alpha=0.4, edgecolors='none')
    yticks = ["Test Image 1", "Test Image 2", "Test Image 1", "Test Image 2"]
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_ylim([-1, 3.4])
    # ax.set_xlabel("Annotated Counts")
    plt.title("Annotated Object Counts For Two In-Domain Test Sets (4 Annotators) ", pad=30)
    ax.set_xlabel("Annotated Count")
    # ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlim(left=0)
    plt.gca().invert_yaxis()
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.84)


    out_path = os.path.join(out_dir, "annotator_counts.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=600)

    

if __name__ == '__main__': 
    # print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    # data = (
    #     "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3", # coder A
    #     "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *", # coder B
    #     "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4", # coder C
    # )

    # missing = '*' # indicator for missing values
    # array = [d.split() for d in data]  # convert to 2D list of string items
    
    # print("nominal metric: %.3f" % krippendorff_alpha(array, nominal_metric, missing_items=missing))
    # print("interval metric: %.3f" % krippendorff_alpha(array, interval_metric, missing_items=missing))


    annotator_dir = "annotator_agreement"
    annotator_names = ["erik", "tim", "kathy", "ian"]

    # for annotator_name in annotator_names:
    #     path_1 = os.path.join("annotator_agreement", annotator_name + "_BlaineLake_annotations.json")
    #     path_2 = os.path.join("annotator_agreement", annotator_name + "_UNI_annotations.json")

    #     log_1 = json_io.load_json(path_1)
    #     log_2 = json_io.load_json(path_2)


    #     annotations = {}
    #     image_names = ["8", "51"]
    #     for image_name in image_names:
    #         annotations[image_name] = log_1[image_name]


    #     image_names = ["UNI_CNH_May30_318", "UNI_CNH_May30_219"]
    #     for image_name in image_names:
    #         annotations[image_name] = log_2[image_name]

        
    #     out_path = os.path.join("annotator_agreement", annotator_name + ".json")
    #     json_io.save_json(out_path, annotations)

    # exit()

    annotation_files = []
    for annotator_name in annotator_names:
        annotations_path = os.path.join(annotator_dir, annotator_name + ".json")
        annotation_file = annotation_utils.load_annotations(annotations_path)
        annotation_files.append(annotation_file)



    image_names = ["8", "51"]
    alpha = get_krippendorff(annotation_files, image_names)
    alpha_patches = get_krippendorff_patches(annotation_files, image_names)
    frac_in_cluster, ave_gtc = get_dbscan(annotation_files, image_names)
    print("BlaineLake River")
    print_counts(annotation_files, image_names)
    print("\tKrippendorff Alpha: {}".format(alpha))
    print("\tKrippendorff Alpha (Patches): {}".format(alpha_patches))
    print("\tFraction in Cluster: {}".format(frac_in_cluster))
    print("\tAve. GTC: {}".format(ave_gtc))
    print()
    print()
    
    image_names = ["UNI_CNH_May30_318", "UNI_CNH_May30_219"]
    alpha = get_krippendorff(annotation_files, image_names)
    alpha_patches = get_krippendorff_patches(annotation_files, image_names)
    frac_in_cluster, ave_gtc = get_dbscan(annotation_files, image_names)
    print("UNI Dugout")
    print_counts(annotation_files, image_names)
    print("\tKrippendorff Alpha: {}".format(alpha))
    print("\tKrippendorff Alpha (Patches): {}".format(alpha_patches))
    print("\tFraction in Cluster: {}".format(frac_in_cluster))
    print("\tAve. GTC: {}".format(ave_gtc))


    out_dir = os.path.join("eval_charts", "annotator_agreement")
    count_chart(annotation_files, out_dir)
