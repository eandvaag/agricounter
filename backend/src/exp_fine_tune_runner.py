import os
import uuid
import random
import time
import math as m
import numpy as np
import glob



from io_utils import json_io
from models.common import annotation_utils, box_utils
import image_set_actions as isa
from image_set import Image
import image_utils





def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def select_fine_tuning_data(test_set_image_set_dir, method, annotations, predictions, metadata, baseline_matching_info, patch_size, num_annotations_to_select):




    patch_overlap_percent = 0
    overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

    image_names = list(annotations.keys())
    image_width = metadata["images"][image_names[0]]["width_px"]
    image_height = metadata["images"][image_names[0]]["height_px"]

    incr = patch_size - overlap_px
    w_covered = max(image_width - patch_size, 0)
    num_w_patches = m.ceil(w_covered / incr) + 1

    h_covered = max(image_height - patch_size, 0)
    num_h_patches = m.ceil(h_covered / incr) + 1

    # num_patches = num_w_patches * num_h_patches

    patch_candidates = []
    for image_name in image_names:
        # image_path = glob.glob(os.path.join(test_set_image_set_dir, "images", image_name + ".*"))[0]
        # image = Image(image_path)
        # image_array = image.load_image_array()
        # exg_array = image_utils.excess_green(image_array)

        for i in range(num_h_patches):
            for j in range(num_w_patches):
                patch_coords = [
                    i * patch_size,
                    j * patch_size,
                    min(image_height, (i+1) * patch_size),
                    min(image_width, (j+1) * patch_size)
                ]

                
                if predictions[image_name]["boxes"].size == 0:
                    patch_value = 1000000 + random.random()

                else:

                    contained_inds = box_utils.get_contained_inds(predictions[image_name]["boxes"], [patch_coords])
                    # contained_annotation_inds = box_utils.get_contained_inds(annotations[image_name]["boxes"], [patch_coords])
                    contained_boxes = predictions[image_name]["boxes"][contained_inds]
                    contained_scores = predictions[image_name]["scores"][contained_inds]
                    clipped_boxes = box_utils.clip_boxes_np(contained_boxes, patch_coords)
                    visibilities = box_utils.box_visibilities_np(contained_boxes, clipped_boxes)
                    # score = np.abs(contained_scores - 0.50) * visibilities

                    # patch_value = 
                    # patch_value = 0
                    # for k in range(contained_scores.size):
                    #     if contained_scores[k] <= 0.5:
                    #         patch_value += (4 * contained_scores[k] + (-1))
                    #     else:
                    #         patch_value += (-4 * (contained_scores[k]) + 3)


                    if contained_scores.size > 0:
                        dists = []
                        for k in range(contained_scores.size):
                            dist = abs(0.5 - contained_scores[k])
                            dists.append(dist)
                        patch_value = np.mean(dists)
                    else:
                        patch_value = 1000000 + random.random()

                    # patch_value = 0
                    # for k in range(contained_scores.size):
                    #     patch_value += gaussian(contained_scores[k], 0.5, 0.167) * visibilities[k]

                    # exg_patch = exg_array[patch_coords[0]:patch_coords[2], patch_coords[1]:patch_coords[3]]
                    # exg_value = np.sum(exg_patch ** 2) / exg_patch.size

                patch_candidates.append([image_name, patch_coords, patch_value]) #, exg_value]) #, contained_annotation_inds.size])






    taken_regions = {}


    print("METHOD", method)

    if method == "random_images":
        candidates = list(annotations.keys())
        selected_images = random.sample(candidates, 5)
        for image_name in selected_images:
            taken_regions[image_name] = []
        for candidate in patch_candidates:
            if candidate[0] in selected_images:
                taken_regions[candidate[0]].append(candidate[1])


    # elif method == "random_patches_match_patch_num":
    #     num_patches_to_match = baseline_matching_info["num_patches"]
    #     selected_candidates = random.sample(patch_candidates, num_patches_to_match)
    #     for candidate in selected_candidates:
    #         image_name = candidate[0]
    #         patch_coords = candidate[1]
    #         if image_name not in taken_regions:
    #             taken_regions[image_name] = []
    #         taken_regions[image_name].append(patch_coords)


    elif method == "random_patches":
        print("method is random_patches")
        print("num_annotations_to_select", num_annotations_to_select)
        patch_candidate_inds = np.arange(0, len(patch_candidates))
        random.shuffle(patch_candidate_inds)
        # num_annotations_to_take = num_annotations_to_select #method.split("_")[2]
        i = 0
        while True:

            sel_ind = patch_candidate_inds[i]
            candidate = patch_candidates[sel_ind]
            i += 1
            image_name = candidate[0]
            patch_coords = candidate[1]
            if image_name not in taken_regions:
                taken_regions[image_name] = []
            taken_regions[image_name].append(patch_coords)

            new_annotation_count = 0
            for image_name in annotations.keys():
                if image_name in taken_regions:
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size

            print(i, new_annotation_count)
            if new_annotation_count >= num_annotations_to_select:
                break


        print("Selected {} patches and {} annotations. Wanted {} annotations".format(i, new_annotation_count, num_annotations_to_select)) 
        



    # elif method == "random_patches_match_annotation_num":
    #     num_annotations_to_match = baseline_matching_info["num_annotations"]
    #     num_annotations = 0
    #     candidate_inds = np.arange(len(patch_candidates)).tolist()
    #     # while num_annotations < num_annotations_to_match:
    #     while True:
    #         sel_ind = random.choice(candidate_inds)
    #         candidate = patch_candidates[sel_ind]
    #         del candidate_inds[sel_ind]

    #         image_name = candidate[0]
    #         patch_coords = candidate[1]
    #         candidate_num_annotations = candidate[3]
    #         num_after = num_annotations + candidate_num_annotations
    #         if abs(num_after - num_annotations_to_match) > abs(num_annotations - num_annotations_to_match):
    #             break


    #         if image_name not in taken_regions:
    #             taken_regions[image_name] = []
    #         taken_regions[image_name].append(patch_coords)
    #         num_annotations += candidate_num_annotations
            
    #         if num_annotations >= num_annotations_to_match:
    #             break



    elif method == "selected_patches_match_patch_num":
        num_patches_to_match = baseline_matching_info["num_patches"]
        sorted_candidates = sorted(patch_candidates, key=lambda x: x[2], reverse=True)
        selected_candidates = sorted_candidates[:num_patches_to_match]
        for candidate in selected_candidates:
            image_name = candidate[0]
            patch_coords = candidate[1]

            if image_name not in taken_regions:
                taken_regions[image_name] = []
            taken_regions[image_name].append(patch_coords)
        # selected_candidates = random.sample(patch_candidates, num_patches_to_match)

    elif method == "selected_patches_match_annotation_num":
        num_annotations_to_match = baseline_matching_info["num_annotations"]
        sorted_candidates = sorted(patch_candidates, key=lambda x: x[2], reverse=True)
        # selected_candidates = sorted_candidates[:num_patches_to_match]
        cur_annotation_count = 0
        i = 0
        # for i in range(len(patch_candidates)):
        # while num_annotations < num_annotations_to_match:
        while True:
            candidate = sorted_candidates[i]
            s_image_name = candidate[0]
            patch_coords = candidate[1]
            # candidate_num_annotations = candidate[3]

            if s_image_name not in taken_regions:
                taken_regions[s_image_name] = []
            taken_regions[s_image_name].append(patch_coords)


            new_annotation_count = 0
            for image_name in annotations.keys():
                if image_name in taken_regions:
                # if image_name == s_image_name:
                #     new_lst = taken_regions[s_image_name].copy()
                #     new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], new_lst).size
                # else:
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                
            
            # num_after = num_annotations + candidate_num_annotations
            if abs(new_annotation_count - num_annotations_to_match) > abs(cur_annotation_count - num_annotations_to_match):
                del taken_regions[s_image_name][-1]
                if len(taken_regions[s_image_name]) == 0:
                    del taken_regions[s_image_name]
                break

            # num_annotations += candidate_num_annotations
            i += 1
            cur_annotation_count = new_annotation_count

            if cur_annotation_count >= num_annotations_to_match:
                break


    # elif method == "correct_bad_detections":
    #     num_annotations_to_match = baseline_matching_info["num_annotations"]
    #     # num_patches_to_match = baseline_matching_info["num_patches"]
    #     num_full_patches_to_match = baseline_matching_info["num_full_patches"]
    #     num_partial_patches_to_match = baseline_matching_info["num_partial_patches"]



    elif method == "selected_patches":
        num_annotations_to_match = baseline_matching_info["num_annotations"]
        # num_patches_to_match = baseline_matching_info["num_patches"]
        num_full_patches_to_match = baseline_matching_info["num_full_patches"]
        num_partial_patches_to_match = baseline_matching_info["num_partial_patches"]
        sorted_candidates = sorted(patch_candidates, key=lambda x: x[2], reverse=True)

        print("There are {} candidates to pick from".format(len(sorted_candidates)))
        cur_annotation_count = 0
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        i = 0
        while True:
            candidate = sorted_candidates[i]
            i += 1
            s_image_name = candidate[0]
            patch_coords = candidate[1]

            is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size

            if is_full_patch and cur_full_patch_count >= num_full_patches_to_match:
                continue
            if not is_full_patch and cur_partial_patch_count >= num_partial_patches_to_match:
                continue


            if s_image_name not in taken_regions:
                taken_regions[s_image_name] = []
            
            taken_regions[s_image_name].append(patch_coords)
            if is_full_patch:
                cur_full_patch_count += 1
            else:
                cur_partial_patch_count += 1


            new_annotation_count = 0
            for image_name in annotations.keys():
                if image_name in taken_regions:
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                
            if new_annotation_count > num_annotations_to_match: #abs(new_annotation_count - num_annotations_to_match) > abs(cur_annotation_count - num_annotations_to_match):
                del taken_regions[s_image_name][-1]
                if len(taken_regions[s_image_name]) == 0:
                    del taken_regions[s_image_name]
                print("Breaking because number of annotations would be exceeded")
                break

 
            cur_annotation_count = new_annotation_count

            if cur_annotation_count >= num_annotations_to_match:
                print("Breaking because number of annotations has been matched")
                break

            # if i >= num_patches_to_match:
            #     break
            if cur_full_patch_count >= num_full_patches_to_match and cur_partial_patch_count >= num_partial_patches_to_match:
                print("Breaking because number of patches has been matched")
                break

            # if cur_partial_patch_count >= num_partial_patches_to_match:
            #     print("Breaking because number of partial patches has been matched")
            #     break


        cur_annotation_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                for patch_coords in taken_regions[image_name]:
                    patch_h = patch_coords[2] - patch_coords[0]
                    patch_w = patch_coords[3] - patch_coords[1]
                    if patch_h == patch_size and patch_w == patch_size:
                        cur_full_patch_count += 1
                    else:
                        cur_partial_patch_count += 1
                # cur_patch_count += len(taken_regions[image_name])

        if cur_partial_patch_count > num_partial_patches_to_match or cur_full_patch_count > num_full_patches_to_match:
            print("Have too many patches...")
            exit() 



        # print("Have {}/{} full patches and {}/{} partial patches".format(
        #     cur_full_patch_count, num_full_patches_to_match, cur_partial_patch_count, num_partial_patches_to_match))


        # sorted_candidates = sorted(patch_candidates, key=lambda x: x[3], reverse=True)
        # print("There are {} candidates to pick from".format(len(sorted_candidates)))

        # i = 0
        # while True:
        #     if i >= len(sorted_candidates):
        #         print("ran out of candidates.")
        #         break
        #     candidate = sorted_candidates[i]
        #     i += 1
        #     s_image_name = candidate[0]
        #     patch_coords = candidate[1]

        #     is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size
        #     skip = False
        #     if is_full_patch:
        #         if cur_full_patch_count >= num_full_patches_to_match:
        #             skip = True
        #     else:
        #         if cur_partial_patch_count >= num_partial_patches_to_match:
        #             skip = True

        #     if not skip and (s_image_name not in taken_regions or patch_coords not in taken_regions[s_image_name]):

        #         new_annotation_count = 0
        #         for image_name in annotations.keys():
        #             if image_name == s_image_name:
        #                 l = [patch_coords]
        #                 if image_name in taken_regions:
        #                     l.extend((taken_regions[image_name]).copy())
        #                 new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
        #             else:
        #                 if image_name in taken_regions:
        #                     new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                
        #         print(cur_annotation_count, new_annotation_count)
        #         if new_annotation_count == cur_annotation_count:
        #             if s_image_name not in taken_regions:
        #                 taken_regions[s_image_name] = []
        #             taken_regions[s_image_name].append(patch_coords)


        #             if is_full_patch:
        #                 cur_full_patch_count += 1
        #             else:
        #                 cur_partial_patch_count += 1


        #         if cur_full_patch_count >= num_full_patches_to_match and cur_partial_patch_count >= num_partial_patches_to_match:
        #             break


        print("Check:")

        cur_annotation_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                for patch_coords in taken_regions[image_name]:
                    patch_h = patch_coords[2] - patch_coords[0]
                    patch_w = patch_coords[3] - patch_coords[1]
                    if patch_h == patch_size and patch_w == patch_size:
                        cur_full_patch_count += 1
                    else:
                        cur_partial_patch_count += 1

                # cur_patch_count += len(taken_regions[image_name])

        print("Annotation count: {} (want {})".format(cur_annotation_count, num_annotations_to_match))
        print("Full patch count: {} (want {})".format(cur_full_patch_count, num_full_patches_to_match))
        print("Partial patch count: {} (want {})".format(cur_partial_patch_count, num_partial_patches_to_match))

    elif method == "selected_patches_first":


        print("method is selected_patches_first")
        print("num_annotations_to_select", num_annotations_to_select)
        # patch_candidate_inds = np.arange(0, len(patch_candidates))
        # random.shuffle(patch_candidate_inds)
        sorted_candidates = sorted(patch_candidates, key=lambda x: x[2])
        # num_annotations_to_take = num_annotations_to_select #method.split("_")[2]
        i = 0
        while True:
            candidate = sorted_candidates[i]
            i += 1
            image_name = candidate[0]
            patch_coords = candidate[1]
            if image_name not in taken_regions:
                taken_regions[image_name] = []
            taken_regions[image_name].append(patch_coords)

            new_annotation_count = 0
            for image_name in annotations.keys():
                if image_name in taken_regions:
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size

            # print(i, new_annotation_count)
            if new_annotation_count >= num_annotations_to_select:
                break


        print("Selected {} patches and {} annotations. Wanted {} annotations".format(i, new_annotation_count, num_annotations_to_select)) 
        
    elif method == "random_patches_second":
        print("method is random_patches_second")

        num_annotations_to_match = baseline_matching_info["num_annotations"]
        num_full_patches_to_match = baseline_matching_info["num_full_patches"]
        num_partial_patches_to_match = baseline_matching_info["num_partial_patches"]

        patch_candidate_inds = np.arange(0, len(patch_candidates))
        random.shuffle(patch_candidates)
        random.shuffle(patch_candidate_inds)
        # num_annotations_to_take = num_annotations_to_select #method.split("_")[2]
        taken_candidates = []
        cur_annotation_count = 0
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        i = 0
        while True:

            sel_ind = patch_candidate_inds[i]
            candidate = patch_candidates[sel_ind]
            i += 1
            s_image_name = candidate[0]
            patch_coords = candidate[1]

            is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size

            if is_full_patch and cur_full_patch_count >= num_full_patches_to_match:
                continue
            if not is_full_patch and cur_partial_patch_count >= num_partial_patches_to_match:
                continue


            taken_candidates.append(candidate)

            if s_image_name not in taken_regions:
                taken_regions[s_image_name] = []
            
            taken_regions[s_image_name].append(patch_coords)
            if is_full_patch:
                cur_full_patch_count += 1
            else:
                cur_partial_patch_count += 1


            new_annotation_count = 0
            for image_name in annotations.keys():
                if image_name in taken_regions:
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                
            # if new_annotation_count > num_annotations_to_match:
            #     del taken_regions[s_image_name][-1]
            #     if len(taken_regions[s_image_name]) == 0:
            #         del taken_regions[s_image_name]
            #     print("Breaking because number of annotations would be exceeded")
            #     break

 
            cur_annotation_count = new_annotation_count

            if cur_annotation_count >= num_annotations_to_match:
                print("Breaking because number of annotations has been matched")
                break

            if cur_full_patch_count >= num_full_patches_to_match and cur_partial_patch_count >= num_partial_patches_to_match:
                print("Breaking because number of patches has been matched")
                break

    

        cur_annotation_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                for patch_coords in taken_regions[image_name]:
                    patch_h = patch_coords[2] - patch_coords[0]
                    patch_w = patch_coords[3] - patch_coords[1]
                    if patch_h == patch_size and patch_w == patch_size:
                        cur_full_patch_count += 1
                    else:
                        cur_partial_patch_count += 1


        print("After initial random selection step, have {} annotations, {} full patches, and {} partial_patches".format(
            cur_annotation_count, cur_full_patch_count, cur_partial_patch_count
        ))

        if cur_annotation_count < num_annotations_to_match:

            # PRE CHANGE COMMIT ID : ff0e3cfaeb8057ba2e1cfd5808eff64d243bf5aa
            current_annotation_counts_per_image = {}
            for image_name in taken_regions.keys():
                current_annotation_counts_per_image[image_name] = box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size


            print("Not enough annotations. Performing substitutions...")
            while True:

                possibly_remove_ind = random.randrange(len(taken_candidates))
                possibly_remove_candidate = taken_candidates[possibly_remove_ind]

                s_image_name = possibly_remove_candidate[0]
                patch_coords = possibly_remove_candidate[1]

                is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size

                # print("getting annotation_count_before")
                annotation_count_before = 0
                # for image_name in taken_regions.keys():
                #     # if image_name == s_image_name:
                #     #     l = taken_regions[image_name].copy()
                #     #     ind = l.index(patch_coords)
                #     #     del l[ind]
                #     #     annotation_count_before += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                #     # else:
                #     annotation_count_before += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size

                for image_name in current_annotation_counts_per_image.keys():
                    annotation_count_before += current_annotation_counts_per_image[image_name]


                print(annotation_count_before)

                done = False
                for substitute_candidate in patch_candidates:
                    if substitute_candidate not in taken_candidates:
                        substitute_image_name = substitute_candidate[0]
                        substitute_patch_coords = substitute_candidate[1]
                        substitute_is_full_patch = substitute_patch_coords[2] - substitute_patch_coords[0] == patch_size and substitute_patch_coords[3] - substitute_patch_coords[1] == patch_size
                        if is_full_patch == substitute_is_full_patch:
                            annotation_count_after = 0
                            for image_name in taken_regions.keys():
                                if image_name == substitute_image_name or image_name == s_image_name:
                                    l = taken_regions[image_name].copy()
                                    if image_name == substitute_image_name:
                                        l.append(substitute_patch_coords)
                                        # annotation_count_after += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                    if image_name == s_image_name:
                                        # l = taken_regions[image_name].copy()
                                        ind = l.index(patch_coords)
                                        del l[ind]
                                    annotation_count_after += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                else:
                                    # annotation_count_after += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                                    annotation_count_after += current_annotation_counts_per_image[image_name]
                            
                            
                            if annotation_count_after > annotation_count_before:
                                print("\t{} {}".format(annotation_count_before, annotation_count_after))
                                del_index = taken_regions[s_image_name].index(patch_coords)
                                del taken_regions[s_image_name][del_index]

                                if substitute_image_name not in taken_regions:
                                    taken_regions[substitute_image_name] = []
                                taken_regions[substitute_image_name].append(substitute_patch_coords)

                                if len(taken_regions[s_image_name]) == 0:
                                    del taken_regions[s_image_name]

                                del_index = taken_candidates.index(possibly_remove_candidate)
                                del taken_candidates[del_index]
                                taken_candidates.append(substitute_candidate)
                                
                                if s_image_name in taken_regions:
                                    current_annotation_counts_per_image[s_image_name] = box_utils.get_contained_inds(annotations[s_image_name]["boxes"], taken_regions[s_image_name]).size
                                else:
                                    current_annotation_counts_per_image[s_image_name] = 0
                                current_annotation_counts_per_image[substitute_image_name] = box_utils.get_contained_inds(annotations[substitute_image_name]["boxes"], taken_regions[substitute_image_name]).size

                                done = True
                                break
                            if done:
                                break
                        if done:
                            break
                    if done:
                        break

                if not done:
                    print("found none")

                new_annotation_count = 0
                for image_name in taken_regions.keys():
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                
                if new_annotation_count >= num_annotations_to_match:
                    break


        elif cur_full_patch_count < num_full_patches_to_match or cur_partial_patch_count < num_partial_patches_to_match:
            print("Not enough patches ... adding more.")

            while True:
                possibly_take_ind = random.randrange(len(patch_candidates))
                candidate = patch_candidates[possibly_take_ind]
                s_image_name = candidate[0]
                patch_coords = candidate[1]

                if candidate in taken_candidates:
                    continue

                is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size

                if is_full_patch and cur_full_patch_count >= num_full_patches_to_match:
                    continue
                if not is_full_patch and cur_partial_patch_count >= num_partial_patches_to_match:
                    continue

                new_annotation_count = 0
                for image_name in annotations.keys():
                    if image_name == s_image_name:
                        if image_name in taken_regions:
                            l = taken_regions[image_name].copy()
                        else:
                            l = []
                        l.append(patch_coords)
                        new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
                    else:
                        if image_name in taken_regions:
                            new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                        

                if cur_annotation_count == new_annotation_count:
                    if s_image_name not in taken_regions:
                        taken_regions[s_image_name] = []
                    taken_regions[s_image_name].append(patch_coords)
                    taken_candidates.append(candidate)
                    if is_full_patch:
                        cur_full_patch_count += 1
                    else:
                        cur_partial_patch_count += 1

                if cur_full_patch_count >= num_full_patches_to_match and cur_partial_patch_count >= num_partial_patches_to_match:
                    break




        cur_annotation_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                for patch_coords in taken_regions[image_name]:
                    patch_h = patch_coords[2] - patch_coords[0]
                    patch_w = patch_coords[3] - patch_coords[1]
                    if patch_h == patch_size and patch_w == patch_size:
                        cur_full_patch_count += 1
                    else:
                        cur_partial_patch_count += 1



        print("Final totals: {}/{} annotations, {}/{} full patches, {}/{} partial patches".format(
            cur_annotation_count, num_annotations_to_match, cur_full_patch_count, num_full_patches_to_match,
            cur_partial_patch_count, num_partial_patches_to_match
        ))




    elif method == "selected_patches_unfair_dist_score":
        num_annotations_to_match = baseline_matching_info["num_annotations"]
        num_full_patches_to_match = baseline_matching_info["num_full_patches"]
        num_partial_patches_to_match = baseline_matching_info["num_partial_patches"]
        sorted_candidates = sorted(patch_candidates, key=lambda x: x[2]) #, reverse=True)

        print("There are {} candidates to pick from".format(len(sorted_candidates)))
        cur_annotation_count = 0
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        # taken_candidates = []
        i = 0
        while True:
            candidate = sorted_candidates[i]
            i += 1
            s_image_name = candidate[0]
            patch_coords = candidate[1]

            is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size

            if is_full_patch and cur_full_patch_count >= num_full_patches_to_match:
                continue
            if not is_full_patch and cur_partial_patch_count >= num_partial_patches_to_match:
                continue


            if s_image_name not in taken_regions:
                taken_regions[s_image_name] = []
            
            taken_regions[s_image_name].append(patch_coords)
            if is_full_patch:
                cur_full_patch_count += 1
            else:
                cur_partial_patch_count += 1


            new_annotation_count = 0
            for image_name in annotations.keys():
                if image_name in taken_regions:
                    new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                
            if new_annotation_count > num_annotations_to_match:
                del taken_regions[s_image_name][-1]
                if len(taken_regions[s_image_name]) == 0:
                    del taken_regions[s_image_name]
                print("Breaking because number of annotations would be exceeded")
                break

 
            cur_annotation_count = new_annotation_count

            if cur_annotation_count >= num_annotations_to_match:
                print("Breaking because number of annotations has been matched")
                break

            if cur_full_patch_count >= num_full_patches_to_match and cur_partial_patch_count >= num_partial_patches_to_match:
                print("Breaking because number of patches has been matched") #Got enough patches, but not enough annotations")
                #. Num annotations {}/{}. Num full patches {}/{}. Num partial patches {}/{}.".format(cur_annotation_count, num_annotations_to_match,
                #                                                       cur_full_patch_count, num_full_patches_to_match,
                #                                                       cur_partial_patch_count, num_partial_patches_to_match))
                # exit()
                break
                # print("Breaking because number of patches has been matched")
                # break


        cur_annotation_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        cur_full_patch_count = 0
        cur_partial_patch_count = 0
        for image_name in annotations.keys():
            if image_name in taken_regions:
                for patch_coords in taken_regions[image_name]:
                    patch_h = patch_coords[2] - patch_coords[0]
                    patch_w = patch_coords[3] - patch_coords[1]
                    if patch_h == patch_size and patch_w == patch_size:
                        cur_full_patch_count += 1
                    else:
                        cur_partial_patch_count += 1


        print("After initial selection step, have {} annotations, {} full patches, and {} partial_patches".format(
            cur_annotation_count, cur_full_patch_count, cur_partial_patch_count
        ))

        # if cur_annotation_count < num_annotations_to_match:

        #     while True:


        # elif cur_full_patch_count < num_full_patches_to_match or cur_partial_patch_count < num_partial_patches_to_match:


        #     while True:
        #         candidate = sorted_candidates[i]
        #         i += 1
        #         s_image_name = candidate[0]
        #         patch_coords = candidate[1]

        #         is_full_patch = patch_coords[2] - patch_coords[0] == patch_size and patch_coords[3] - patch_coords[1] == patch_size

        #         if is_full_patch and cur_full_patch_count >= num_full_patches_to_match:
        #             continue
        #         if not is_full_patch and cur_partial_patch_count >= num_partial_patches_to_match:
        #             continue

        #         new_annotation_count = 0
        #         for image_name in annotations.keys():
        #             if image_name == s_image_name:
        #                 if image_name in taken_regions:
        #                     l = taken_regions[image_name].copy()
        #                 else:
        #                     l = []
        #                 l.append(patch_coords)
        #                 new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], l).size
        #             else:
        #                 if image_name in taken_regions:
        #                     new_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
                        

        #         if cur_annotation_count == new_annotation_count:
        #             taken_regions[s_image_name].append(patch_coords)
        #             if is_full_patch:
        #                 cur_full_patch_count += 1
        #             else:
        #                 cur_partial_patch_count += 1

        #         if cur_full_patch_count >= num_full_patches_to_match and cur_partial_patch_count >= num_partial_patches_to_match:
        #             break

        # cur_annotation_count = 0
        # for image_name in annotations.keys():
        #     if image_name in taken_regions:
        #         cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        # cur_full_patch_count = 0
        # cur_partial_patch_count = 0
        # for image_name in annotations.keys():
        #     if image_name in taken_regions:
        #         for patch_coords in taken_regions[image_name]:
        #             patch_h = patch_coords[2] - patch_coords[0]
        #             patch_w = patch_coords[3] - patch_coords[1]
        #             if patch_h == patch_size and patch_w == patch_size:
        #                 cur_full_patch_count += 1
        #             else:
        #                 cur_partial_patch_count += 1
        
        # print("Annotation count: {} (want {})".format(cur_annotation_count, num_annotations_to_match))
        # print("Full patch count: {} (want {})".format(cur_full_patch_count, num_full_patches_to_match))
        # print("Partial patch count: {} (want {})".format(cur_partial_patch_count, num_partial_patches_to_match))

        if cur_full_patch_count > num_full_patches_to_match:
            print("Have too many full patches...")
            exit()

        if cur_partial_patch_count > num_partial_patches_to_match:
            print("Have too many partial patches...")
            exit()

        if cur_annotation_count > num_annotations_to_match:
            print("Have too many annotations...")
            exit()


        # print("Check:")

        # cur_annotation_count = 0
        # for image_name in annotations.keys():
        #     if image_name in taken_regions:
        #         cur_annotation_count += box_utils.get_contained_inds(annotations[image_name]["boxes"], taken_regions[image_name]).size
        # cur_full_patch_count = 0
        # cur_partial_patch_count = 0
        # for image_name in annotations.keys():
        #     if image_name in taken_regions:
        #         for patch_coords in taken_regions[image_name]:
        #             patch_h = patch_coords[2] - patch_coords[0]
        #             patch_w = patch_coords[3] - patch_coords[1]
        #             if patch_h == patch_size and patch_w == patch_size:
        #                 cur_full_patch_count += 1
        #             else:
        #                 cur_partial_patch_count += 1









    return taken_regions
    



def get_mapping_for_test_set(test_set_image_set_dir):

    mapping = {}
    results_dir = os.path.join(test_set_image_set_dir, "model", "results")
    for result_dir in glob.glob(os.path.join(results_dir, "*")):
        request_path = os.path.join(result_dir, "request.json")
        request = json_io.load_json(request_path)
        mapping[request["results_name"]] = request["request_uuid"]
    return mapping


def possibly_update_baseline_matching_info(baseline, test_set_image_set_dir, num_annotations_to_select, dup_num):
    baseline_matching_info = None
    mapping = get_mapping_for_test_set(test_set_image_set_dir)
    # random_images_name = baseline["model_name"] + "_post_finetune_random_images" + "_" + str(num_images_to_select) + "_dup_" + str(dup_num)
    # random_patches_name = baseline["model_name"] + "_post_finetune_random_patches" + "_" + str(num_annotations_to_select) + "_annotations_dup_" + str(dup_num)
    random_patches_name = baseline["model_name"] + "_post_finetune_selected_patches_first" + "_" + str(num_annotations_to_select) + "_annotations_dup_" + str(dup_num)
    
    if random_patches_name in mapping:
        print("\nupdating baseline matching info\n")
        result_uuid = mapping[random_patches_name]
        result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)
        annotations_path = os.path.join(result_dir, "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

        baseline_matching_info = {}
        baseline_matching_info["num_patches"] = 0
        baseline_matching_info["num_full_patches"] = 0
        baseline_matching_info["num_partial_patches"] = 0
        baseline_matching_info["num_annotations"] = 0
        for image_name in annotations.keys():
            if len(annotations[image_name]["training_regions"]) > 0:
                for region in annotations[image_name]["training_regions"]:
                    if region[2] - region[0] == patch_size and region[3] - region[1] == patch_size:
                        baseline_matching_info["num_full_patches"] += 1
                    else:
                        baseline_matching_info["num_partial_patches"] += 1

                if len(annotations[image_name]["training_regions"]) != len(np.unique(annotations[image_name]["training_regions"], axis=0)):
                    raise RuntimeError("Problem -- duplicate training region")

                baseline_matching_info["num_patches"] += len(annotations[image_name]["training_regions"])
                baseline_matching_info["num_annotations"] += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size
                #len(annotations[image_name]["boxes"])
    else:
        print("\nnot updating baseline matching info\n")

    return baseline_matching_info




def get_anno_stats_for_model(mapping, test_set_image_set_dir, model_name):

    result_uuid = mapping[model_name]
    result_dir = os.path.join(test_set_image_set_dir, "model", "results", result_uuid)
    annotations_path = os.path.join(result_dir, "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)

    patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])

    baseline_matching_info = {}
    baseline_matching_info["num_patches"] = 0
    baseline_matching_info["num_full_patches"] = 0
    baseline_matching_info["num_partial_patches"] = 0
    baseline_matching_info["num_annotations"] = 0
    for image_name in annotations.keys():
        if len(annotations[image_name]["training_regions"]) > 0:
            for region in annotations[image_name]["training_regions"]:
                if region[2] - region[0] == patch_size and region[3] - region[1] == patch_size:
                    baseline_matching_info["num_full_patches"] += 1
                else:
                    baseline_matching_info["num_partial_patches"] += 1

            baseline_matching_info["num_patches"] += len(annotations[image_name]["training_regions"])
            baseline_matching_info["num_annotations"] += box_utils.get_contained_inds(annotations[image_name]["boxes"], annotations[image_name]["training_regions"]).size

    return baseline_matching_info

def check_fine_tuning_models(baseline, test_set, num_dups, num_annotations_to_select):

    test_set_image_set_dir = os.path.join("usr", "data",
                                                test_set["username"], "image_sets",
                                                test_set["farm_name"],
                                                test_set["field_name"],
                                                test_set["mission_date"])

    mapping = get_mapping_for_test_set(test_set_image_set_dir)

    methods = [
        "selected_patches_first",
        "random_patches_second"
    ]

    
    for dup_num in range(0, num_dups+0):
        dup_num_results = []
        for method in methods:
            res_name = baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select) + "_annotations_dup_" + str(dup_num)
    
            anno_stats = get_anno_stats_for_model(mapping, test_set_image_set_dir, res_name)
            print()
            print(res_name)
            print(anno_stats)
            dup_num_results.append(anno_stats)

        for k in ["num_annotations", "num_partial_patches", "num_full_patches"]:
            if dup_num_results[0][k] > dup_num_results[1][k]:
                raise RuntimeError("Problem")


        print()
        print("---")
        print()



def eval_fine_tune_test(server, test_set, baseline, methods, num_annotations_to_select, num_dups):




    
    test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
        
    res_names = [] #[baseline["model_name"] + "_pre_finetune"]
    for method in methods:
        for dup_num in range(1, num_dups+0):
            res_name = baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select) + "_annotations_dup_" + str(dup_num)
            res_names.append(res_name)

    mapping = get_mapping_for_test_set(test_set_image_set_dir)
    for res_name in res_names:
        if res_name in mapping:
            raise RuntimeError("Running fine-tune experiment would cause duplicate result name: {}".format(res_name))

    metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
    metadata = json_io.load_json(metadata_path)

    annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
    annotations = annotation_utils.load_annotations(annotations_path)
    for image_name in annotations.keys():
        annotations[image_name]["training_regions"] = []
        annotations[image_name]["test_regions"] = [
            [0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]
        ]
    annotation_utils.save_annotations(annotations_path, annotations)
    annotations = annotation_utils.load_annotations(annotations_path)
    patch_size = annotation_utils.get_patch_size(annotations, ["training_regions", "test_regions"])



    print("switching to model")
    model_dir = os.path.join(test_set_image_set_dir, "model")
    switch_req_path = os.path.join(model_dir, "switch_request.json")
    switch_req = {
        "model_name": baseline["model_name"],
        "model_creator": baseline["model_creator"]
    }
    json_io.save_json(switch_req_path, switch_req)

    item = {
        "username": test_set["username"],
        "farm_name": test_set["farm_name"],
        "field_name": test_set["field_name"],
        "mission_date": test_set["mission_date"]
    }


    switch_processed = False
    isa.process_switch(item)
    while not switch_processed:
        print("Waiting for process switch")
        time.sleep(1)
        if not os.path.exists(switch_req_path):
            switch_processed = True


    
    

    annotations = annotation_utils.load_annotations(annotations_path)
    regions = []
    image_names = list(annotations.keys())
    for image_name in annotations.keys():
        regions.append([[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]])


    
    pre_finetune_request_uuid = str(uuid.uuid4())
    pre_finetune_result_name = baseline["model_name"] + "_pre_finetune"
    request = {
        "request_uuid": pre_finetune_request_uuid,
        "start_time": int(time.time()),
        "image_names": image_names,
        "regions": regions,
        "save_result": True,
        "results_name": pre_finetune_result_name, #+ str(num_images_to_select),
        "results_message": ""
    }
    # mapping = get_mapping_for_test_set(test_set_image_set_dir)
    # request_uuid = mapping[baseline["model_name"] + "_pre_finetune_selected_patches_match_both" + "_" + str(num_images_to_select)]

    request_path = os.path.join(test_set_image_set_dir, "model", "prediction", 
                                "image_set_requests", "pending", pre_finetune_request_uuid + ".json")


    if pre_finetune_result_name not in mapping:
        json_io.save_json(request_path, request)
        print("running pre finetune process_predict")
        server.process_predict(item)

    # initial = True



    


    for method in methods:


        for dup_num in range(1, num_dups+0):



            # if method == "random_images":
            #     baseline_matching_info = None
            # else:
            baseline_matching_info = possibly_update_baseline_matching_info(baseline, test_set_image_set_dir, num_annotations_to_select, dup_num)

            print("baseline_matching_info", baseline_matching_info)

            # if initial:
            #     initial = False
            # else:
            mapping = get_mapping_for_test_set(test_set_image_set_dir)
            pre_finetune_request_uuid = mapping[baseline["model_name"] + "_pre_finetune"]  #+ str(num_images_to_select)]


            annotations = annotation_utils.load_annotations(annotations_path)
            result_dir = os.path.join(test_set_image_set_dir, "model", "results", pre_finetune_request_uuid)
            predictions_path = os.path.join(result_dir, "predictions.json")
            predictions = annotation_utils.load_predictions(predictions_path)

            taken_regions = select_fine_tuning_data(test_set_image_set_dir, method, annotations, predictions, metadata, baseline_matching_info, patch_size, num_annotations_to_select)


            for image_name in taken_regions.keys():
                annotations[image_name]["training_regions"] = taken_regions[image_name]
                annotations[image_name]["test_regions"] = []

            annotation_utils.save_annotations(annotations_path, annotations)

            # if method == "random_images":
            #     baseline_matching_info = {}
                
            #     num_selected_patches = 0
            #     for image_name in taken_regions.keys():
            #         num_selected_patches += len(taken_regions[image_name])

            #     baseline_matching_info["num_patches"] = num_selected_patches




            server.sch_ctx["training_queue"].enqueue(test_set)
                
            train_queue_size = server.sch_ctx["training_queue"].size()
            print("train_queue_size", train_queue_size)
            while train_queue_size > 0:
                item = server.sch_ctx["training_queue"].dequeue()
                print("running process_train")


                re_enqueue = server.process_train(item)


                if re_enqueue:
                    server.sch_ctx["training_queue"].enqueue(item)
                train_queue_size = server.sch_ctx["training_queue"].size()

        
            
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": image_names,
                "regions": regions,
                "save_result": True,
                "results_name": baseline["model_name"] + "_post_finetune_" + method + "_" + str(num_annotations_to_select) + "_annotations_dup_" + str(dup_num),
                "results_message": ""
            }

            request_path = os.path.join(test_set_image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running post finetune process_predict")
            server.process_predict(item)

        

            annotations = annotation_utils.load_annotations(annotations_path)
            for image_name in annotations.keys():
                annotations[image_name]["training_regions"] = []
                annotations[image_name]["test_regions"] = [
                    [0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]
                ]
            annotation_utils.save_annotations(annotations_path, annotations)



            print("switching to model")
            model_dir = os.path.join(test_set_image_set_dir, "model")
            switch_req_path = os.path.join(model_dir, "switch_request.json")
            switch_req = {
                "model_name": baseline["model_name"],
                "model_creator": baseline["model_creator"]
            }
            json_io.save_json(switch_req_path, switch_req)

            item = {
                "username": test_set["username"],
                "farm_name": test_set["farm_name"],
                "field_name": test_set["field_name"],
                "mission_date": test_set["mission_date"]
            }

            switch_processed = False
            isa.process_switch(item)
            while not switch_processed:
                print("Waiting for process switch")
                time.sleep(1)
                if not os.path.exists(switch_req_path):
                    switch_processed = True