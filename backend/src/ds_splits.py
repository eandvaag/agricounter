import os
import math as m

from models.common import annotation_utils

LOWEED = [
'Davidson Stone11NE 2022-06-03',
'row_spacing nasser 2021-06-01',
'SaskatoonEast Stevenson5SW 2022-06-13',
'Biggar Dennis1 2021-06-04',
'Saskatoon Norheim1 2021-05-26',
'BlaineLake Serhienko11 2022-06-07',
'row_spacing brown 2021-06-01',
'UNI Brown 2021-06-05',
'row_spacing nasser2 2022-06-02',
'UNI Sutherland 2021-06-05',
'Biggar Dennis3 2021-06-04',
'Saskatoon Norheim2 2021-05-26',
'Saskatoon Norheim5 2022-05-24',
'Saskatoon Norheim4 2022-05-24',
'UNI LowN2 2021-06-07',
'BlaineLake Serhienko9S 2022-06-07',
'BlaineLake Serhienko12 2022-06-14',
'BlaineLake Serhienko15 2022-06-14'
]


HIWEED = [
'UNI Dugout 2022-05-30',
'UNI CNH-DugoutROW 2022-05-30',
'BlaineLake Serhienko10 2022-06-14',
'Saskatoon Norheim1 2021-06-02',
'MORSE Nasser 2022-05-27',
'Biggar Dennis5 2021-06-12',
'MORSE Dugout 2022-05-27',
'Biggar Dennis3 2021-06-12',
'UNI LowN1 2021-06-07',
'row_spacing brown 2021-06-08',
'BlaineLake Serhienko9N 2022-06-07',
'BlaineLake Lake 2021-06-09',
'SaskatoonEast Stevenson5NW 2022-06-20',
'UNI Vaderstad 2022-06-16',
'Biggar Dennis2 2021-06-12',
'BlaineLake River 2021-06-09',
'BlaineLake Serhienko9S 2022-06-14',
'BlaineLake HornerWest 2021-06-09'
]

uneq_manual_LOWEED = [

'Davidson Stone11NE 2022-06-03',
'row_spacing nasser 2021-06-01',
'UNI Dugout 2022-05-30',
'MORSE Nasser 2022-05-27',
'MORSE Dugout 2022-05-27',

'row_spacing brown 2021-06-08',
'Biggar Dennis1 2021-06-04',
'Biggar Dennis3 2021-06-12',

'row_spacing brown 2021-06-01',
'row_spacing nasser2 2022-06-02',
'UNI CNH-DugoutROW 2022-05-30',
'SaskatoonEast Stevenson5NW 2022-06-20',
]


uneq_manual_HIWEED = [

'UNI Vaderstad 2022-06-16',
'UNI LowN2 2021-06-07',
'UNI Sutherland 2021-06-05',
'UNI Brown 2021-06-05',
'Biggar Dennis3 2021-06-04',
'BlaineLake Serhienko10 2022-06-14',
'BlaineLake Serhienko9N 2022-06-07',
'Saskatoon Norheim1 2021-06-02',
'Saskatoon Norheim5 2022-05-24',
'Saskatoon Norheim2 2021-05-26',
'Saskatoon Norheim4 2022-05-24',
'Biggar Dennis5 2021-06-12',
'Biggar Dennis2 2021-06-12',
'UNI LowN1 2021-06-07',


'BlaineLake Lake 2021-06-09',
'BlaineLake River 2021-06-09',
'BlaineLake Serhienko9S 2022-06-14',
'BlaineLake Serhienko15 2022-06-14'
'BlaineLake Serhienko11 2022-06-07',    
'Saskatoon Norheim1 2021-05-26',
'SaskatoonEast Stevenson5SW 2022-06-13',
'BlaineLake Serhienko9S 2022-06-07',
'BlaineLake Serhienko12 2022-06-14',
'BlaineLake HornerWest 2021-06-09'
]


manual_LOWEED = [

'Davidson Stone11NE 2022-06-03',
'row_spacing nasser 2021-06-01',
'UNI Dugout 2022-05-30',
'MORSE Nasser 2022-05-27',
'MORSE Dugout 2022-05-27',

'row_spacing brown 2021-06-08',
'Biggar Dennis1 2021-06-04',
'Biggar Dennis3 2021-06-12',

'row_spacing brown 2021-06-01',
'row_spacing nasser2 2022-06-02',
'UNI CNH-DugoutROW 2022-05-30',
'SaskatoonEast Stevenson5NW 2022-06-20',

'UNI Vaderstad 2022-06-16',
'UNI LowN2 2021-06-07',
'UNI Sutherland 2021-06-05',
'UNI Brown 2021-06-05',
'Biggar Dennis3 2021-06-04',
'BlaineLake Serhienko10 2022-06-14'

]


manual_HIWEED = [

'BlaineLake Serhienko9N 2022-06-07',
'Saskatoon Norheim1 2021-06-02',
'Saskatoon Norheim5 2022-05-24',
'Saskatoon Norheim2 2021-05-26',
'Saskatoon Norheim4 2022-05-24',
'Biggar Dennis5 2021-06-12',
'Biggar Dennis2 2021-06-12',
'UNI LowN1 2021-06-07',
'BlaineLake Lake 2021-06-09',
'BlaineLake River 2021-06-09',
'BlaineLake Serhienko9S 2022-06-14',
'BlaineLake Serhienko15 2022-06-14',
'BlaineLake Serhienko11 2022-06-07',    
'Saskatoon Norheim1 2021-05-26',
'SaskatoonEast Stevenson5SW 2022-06-13',
'BlaineLake Serhienko9S 2022-06-07',
'BlaineLake Serhienko12 2022-06-14',
'BlaineLake HornerWest 2021-06-09'
]





EARLY = [
'Biggar Dennis1 2021-06-04', 
'row_spacing nasser 2021-06-01', 
'row_spacing brown 2021-06-01', 
'BlaineLake Serhienko11 2022-06-07',
'Biggar Dennis3 2021-06-04', 
'Davidson Stone11NE 2022-06-03', 
'UNI CNH-DugoutROW 2022-05-30', 
'MORSE Nasser 2022-05-27', 
'BlaineLake Serhienko15 2022-06-14', 
'MORSE Dugout 2022-05-27', 
'UNI Dugout 2022-05-30', 
'UNI Sutherland 2021-06-05', 
'UNI Brown 2021-06-05', 
'Saskatoon Norheim4 2022-05-24', 
'row_spacing nasser2 2022-06-02', 
'Saskatoon Norheim5 2022-05-24', 
'BlaineLake Lake 2021-06-09', 
'BlaineLake River 2021-06-09'
]


LATE = [
'BlaineLake HornerWest 2021-06-09', 
'BlaineLake Serhienko9S 2022-06-07', 
'SaskatoonEast Stevenson5SW 2022-06-13', 
'Saskatoon Norheim2 2021-05-26', 
'Saskatoon Norheim1 2021-05-26', 
'BlaineLake Serhienko9N 2022-06-07', 
'BlaineLake Serhienko12 2022-06-14', 
'UNI LowN1 2021-06-07', 
'UNI LowN2 2021-06-07', 
'Saskatoon Norheim1 2021-06-02', 
'Biggar Dennis5 2021-06-12', 
'Biggar Dennis3 2021-06-12', 
'Biggar Dennis2 2021-06-12', 
'row_spacing brown 2021-06-08', 
'SaskatoonEast Stevenson5NW 2022-06-20', 
'BlaineLake Serhienko10 2022-06-14', 
'UNI Vaderstad 2022-06-16', 
'BlaineLake Serhienko9S 2022-06-14'
]


RAND1 = [
'UNI CNH-DugoutROW 2022-05-30',
'UNI LowN2 2021-06-07',
'BlaineLake Serhienko11 2022-06-07',
'BlaineLake HornerWest 2021-06-09',
'Saskatoon Norheim5 2022-05-24',
'Biggar Dennis3 2021-06-12',
'BlaineLake Serhienko9S 2022-06-14',
'BlaineLake Serhienko9N 2022-06-07',
'BlaineLake Lake 2021-06-09',
'row_spacing nasser2 2022-06-02',
'BlaineLake River 2021-06-09',
'SaskatoonEast Stevenson5SW 2022-06-13',
'row_spacing nasser 2021-06-01',
'row_spacing brown 2021-06-08',
'BlaineLake Serhienko12 2022-06-14',
'Saskatoon Norheim4 2022-05-24',
'Davidson Stone11NE 2022-06-03',
'Biggar Dennis2 2021-06-12'
]


RAND2 = [
'row_spacing brown 2021-06-01',
'SaskatoonEast Stevenson5NW 2022-06-20',
'Saskatoon Norheim1 2021-06-02',
'BlaineLake Serhienko15 2022-06-14',
'BlaineLake Serhienko10 2022-06-14',
'Biggar Dennis5 2021-06-12',
'Biggar Dennis1 2021-06-04',
'UNI Sutherland 2021-06-05',
'MORSE Dugout 2022-05-27',
'MORSE Nasser 2022-05-27',
'UNI Brown 2021-06-05',
'Biggar Dennis3 2021-06-04',
'Saskatoon Norheim2 2021-05-26',
'UNI LowN1 2021-06-07',
'BlaineLake Serhienko9S 2022-06-07',
'UNI Dugout 2022-05-30',
'UNI Vaderstad 2022-06-16',
'Saskatoon Norheim1 2021-05-26'
]


split_lookup = {
    "LOWEED": LOWEED,
    "HIWEED": HIWEED,
    "EARLY": EARLY,
    "LATE": LATE,
    "RAND1": RAND1,
    "RAND2": RAND2,
    "MANLOWEED": manual_LOWEED,
    "MANHIWEED": manual_HIWEED
}

def get_split(split_name, prefix):

    lst = split_lookup[split_name]

    image_sets = []
    for s in lst:
        items = s.split(" ")
        image_set = {
            "username": "ds_splits",
            "farm_name": prefix + "_" + items[0],
            "field_name": prefix + "_" + items[1],
            "mission_date": items[2]
        }
        image_sets.append(image_set)

    return image_sets


def get_num_patches_for_split(split_name, prefix):

    image_sets = get_split(split_name, prefix)

    total_num_patches = 0
    for image_set in image_sets:
        image_set_dir = os.path.join("usr", "data", image_set["username"], "image_sets", 
                                image_set["farm_name"], 
                                image_set["field_name"], 
                                image_set["mission_date"])


        annotations_path = os.path.join(image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        num_annotated = 0
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                num_annotated += 1

        width = 5472
        height = 3648
        patch_size = 416

        patch_overlap_percent = 0

        overlap_px = int(m.floor(patch_size * (patch_overlap_percent / 100)))

        incr = patch_size - overlap_px
        w_covered = max(width - patch_size, 0)
        num_w_patches = m.ceil(w_covered / incr) + 1

        h_covered = max(height - patch_size, 0)
        num_h_patches = m.ceil(h_covered / incr) + 1

        num_patches = num_w_patches * num_h_patches

        total_num_patches += (num_patches * num_annotated)

    return total_num_patches