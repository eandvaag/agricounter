import os
import glob
import time
import uuid
import logging
import urllib3

from io_utils import json_io
from models.common import annotation_utils
import image_set_actions as isa
import server
from lock_queue import LockQueue
import ds_splits

my_plot_colors = ["orangered", "royalblue", "forestgreen", "orange", "mediumorchid"]



ds_full_sets = [
{'username': 'ds_splits', 'farm_name': 'FULL_Saskatoon', 'field_name': 'FULL_Norheim1', 'mission_date': '2021-06-02'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko9S', 'mission_date': '2022-06-07'},
{'username': 'ds_splits', 'farm_name': 'FULL_Biggar', 'field_name': 'FULL_Dennis5', 'mission_date': '2021-06-12'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko9S', 'mission_date': '2022-06-14'},
{'username': 'ds_splits', 'farm_name': 'FULL_row_spacing', 'field_name': 'FULL_brown', 'mission_date': '2021-06-01'},
{'username': 'ds_splits', 'farm_name': 'FULL_Davidson', 'field_name': 'FULL_Stone11NE', 'mission_date': '2022-06-03'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Lake', 'mission_date': '2021-06-09'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_Vaderstad', 'mission_date': '2022-06-16'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_Sutherland', 'mission_date': '2021-06-05'},
{'username': 'ds_splits', 'farm_name': 'FULL_Biggar', 'field_name': 'FULL_Dennis3', 'mission_date': '2021-06-12'},
{'username': 'ds_splits', 'farm_name': 'FULL_Biggar', 'field_name': 'FULL_Dennis3', 'mission_date': '2021-06-04'},
{'username': 'ds_splits', 'farm_name': 'FULL_Saskatoon', 'field_name': 'FULL_Norheim4', 'mission_date': '2022-05-24'},
{'username': 'ds_splits', 'farm_name': 'FULL_Biggar', 'field_name': 'FULL_Dennis1', 'mission_date': '2021-06-04'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_Brown', 'mission_date': '2021-06-05'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko10', 'mission_date': '2022-06-14'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_HornerWest', 'mission_date': '2021-06-09'},
{'username': 'ds_splits', 'farm_name': 'FULL_row_spacing', 'field_name': 'FULL_brown', 'mission_date': '2021-06-08'},
{'username': 'ds_splits', 'farm_name': 'FULL_Biggar', 'field_name': 'FULL_Dennis2', 'mission_date': '2021-06-12'},
{'username': 'ds_splits', 'farm_name': 'FULL_row_spacing', 'field_name': 'FULL_nasser', 'mission_date': '2021-06-01'},
{'username': 'ds_splits', 'farm_name': 'FULL_MORSE', 'field_name': 'FULL_Dugout', 'mission_date': '2022-05-27'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko15', 'mission_date': '2022-06-14'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_LowN1', 'mission_date': '2021-06-07'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko11', 'mission_date': '2022-06-07'},
{'username': 'ds_splits', 'farm_name': 'FULL_MORSE', 'field_name': 'FULL_Nasser', 'mission_date': '2022-05-27'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_Dugout', 'mission_date': '2022-05-30'},
{'username': 'ds_splits', 'farm_name': 'FULL_Saskatoon', 'field_name': 'FULL_Norheim5', 'mission_date': '2022-05-24'},
{'username': 'ds_splits', 'farm_name': 'FULL_row_spacing', 'field_name': 'FULL_nasser2', 'mission_date': '2022-06-02'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_LowN2', 'mission_date': '2021-06-07'},
{'username': 'ds_splits', 'farm_name': 'FULL_Saskatoon', 'field_name': 'FULL_Norheim1', 'mission_date': '2021-05-26'},
{'username': 'ds_splits', 'farm_name': 'FULL_Saskatoon', 'field_name': 'FULL_Norheim2', 'mission_date': '2021-05-26'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_River', 'mission_date': '2021-06-09'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko9N', 'mission_date': '2022-06-07'},
{'username': 'ds_splits', 'farm_name': 'FULL_SaskatoonEast', 'field_name': 'FULL_Stevenson5NW', 'mission_date': '2022-06-20'},
{'username': 'ds_splits', 'farm_name': 'FULL_SaskatoonEast', 'field_name': 'FULL_Stevenson5SW', 'mission_date': '2022-06-13'},
{'username': 'ds_splits', 'farm_name': 'FULL_BlaineLake', 'field_name': 'FULL_Serhienko12', 'mission_date': '2022-06-14'},
{'username': 'ds_splits', 'farm_name': 'FULL_UNI', 'field_name': 'FULL_CNH-DugoutROW', 'mission_date': '2022-05-30'}

]


ns_in_domain_test_sets = [
{'username': 'new_split', 'farm_name': 'id_Saskatoon', 'field_name': 'id_Norheim1', 'mission_date': '2021-06-02'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_Serhienko9S', 'mission_date': '2022-06-07'},
{'username': 'new_split', 'farm_name': 'id_Biggar', 'field_name': 'id_Dennis5', 'mission_date': '2021-06-12'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_Serhienko9S', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'id_row_spacing', 'field_name': 'id_brown', 'mission_date': '2021-06-01'},
{'username': 'new_split', 'farm_name': 'id_Davidson', 'field_name': 'id_Stone11NE', 'mission_date': '2022-06-03'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_Lake', 'mission_date': '2021-06-09'},
{'username': 'new_split', 'farm_name': 'id_UNI', 'field_name': 'id_Vaderstad', 'mission_date': '2022-06-16'},
{'username': 'new_split', 'farm_name': 'id_UNI', 'field_name': 'id_Sutherland', 'mission_date': '2021-06-05'},
{'username': 'new_split', 'farm_name': 'id_Biggar', 'field_name': 'id_Dennis3', 'mission_date': '2021-06-12'},
{'username': 'new_split', 'farm_name': 'id_Biggar', 'field_name': 'id_Dennis3', 'mission_date': '2021-06-04'},
{'username': 'new_split', 'farm_name': 'id_Saskatoon', 'field_name': 'id_Norheim4', 'mission_date': '2022-05-24'},
{'username': 'new_split', 'farm_name': 'id_Biggar', 'field_name': 'id_Dennis1', 'mission_date': '2021-06-04'},
{'username': 'new_split', 'farm_name': 'id_UNI', 'field_name': 'id_Brown', 'mission_date': '2021-06-05'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_Serhienko10', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_HornerWest', 'mission_date': '2021-06-09'},
{'username': 'new_split', 'farm_name': 'id_row_spacing', 'field_name': 'id_brown', 'mission_date': '2021-06-08'},
{'username': 'new_split', 'farm_name': 'id_Biggar', 'field_name': 'id_Dennis2', 'mission_date': '2021-06-12'},
{'username': 'new_split', 'farm_name': 'id_row_spacing', 'field_name': 'id_nasser', 'mission_date': '2021-06-01'},
{'username': 'new_split', 'farm_name': 'id_MORSE', 'field_name': 'id_Dugout', 'mission_date': '2022-05-27'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_Serhienko15', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'id_UNI', 'field_name': 'id_LowN1', 'mission_date': '2021-06-07'},
{'username': 'new_split', 'farm_name': 'id_BlaineLake', 'field_name': 'id_Serhienko11', 'mission_date': '2022-06-07'},
{'username': 'new_split', 'farm_name': 'id_MORSE', 'field_name': 'id_Nasser', 'mission_date': '2022-05-27'},
{'username': 'new_split', 'farm_name': 'id_UNI', 'field_name': 'id_Dugout', 'mission_date': '2022-05-30'},
{'username': 'new_split', 'farm_name': 'id_Saskatoon', 'field_name': 'id_Norheim5', 'mission_date': '2022-05-24'},
{'username': 'new_split', 'farm_name': 'id_row_spacing', 'field_name': 'id_nasser2', 'mission_date': '2022-06-02'}
]

ns_test_sets = [
{'username': 'new_split', 'farm_name': 'test_UNI', 'field_name': 'test_LowN2', 'mission_date': '2021-06-07'},
{'username': 'new_split', 'farm_name': 'test_Saskatoon', 'field_name': 'test_Norheim1', 'mission_date': '2021-05-26'},
{'username': 'new_split', 'farm_name': 'test_Saskatoon', 'field_name': 'test_Norheim2', 'mission_date': '2021-05-26'},
{'username': 'new_split', 'farm_name': 'test_BlaineLake', 'field_name': 'test_River', 'mission_date': '2021-06-09'},
{'username': 'new_split', 'farm_name': 'test_BlaineLake', 'field_name': 'test_Serhienko9N', 'mission_date': '2022-06-07'},
{'username': 'new_split', 'farm_name': 'test_SaskatoonEast', 'field_name': 'test_Stevenson5NW', 'mission_date': '2022-06-20'},
{'username': 'new_split', 'farm_name': 'test_SaskatoonEast', 'field_name': 'test_Stevenson5SW', 'mission_date': '2022-06-13'},
{'username': 'new_split', 'farm_name': 'test_BlaineLake', 'field_name': 'test_Serhienko12', 'mission_date': '2022-06-14'},
{'username': 'new_split', 'farm_name': 'test_UNI', 'field_name': 'test_CNH-DugoutROW', 'mission_date': '2022-05-30'}
]


eval_in_domain_test_sets = [
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "nasser",
        "mission_date": "2021-06-01"
    },
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "brown",
        "mission_date": "2021-06-01"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Dugout",
        "mission_date": "2022-05-30"
    },
    {
        "username": "eval",
        "farm_name": "MORSE",
        "field_name": "Dugout",
        "mission_date": "2022-05-27"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Brown",
        "mission_date": "2021-06-05"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Sutherland",
        "mission_date": "2021-06-05"
    },
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "nasser2",
        "mission_date": "2022-06-02"
    },
    {
        "username": "eval",
        "farm_name": "MORSE",
        "field_name": "Nasser",
        "mission_date": "2022-05-27"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "LowN2",
        "mission_date": "2021-06-07"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim4",
        "mission_date": "2022-05-24"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim5",
        "mission_date": "2022-05-24"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim1",
        "mission_date": "2021-05-26"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim2",
        "mission_date": "2021-05-26"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis1",
        "mission_date": "2021-06-04"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis3",
        "mission_date": "2021-06-04"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "River",
        "mission_date": "2021-06-09"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Lake",
        "mission_date": "2021-06-09"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "HornerWest",
        "mission_date": "2021-06-09"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "LowN1",
        "mission_date": "2021-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9N",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "Saskatoon",
        "field_name": "Norheim1",
        "mission_date": "2021-06-02"
    },
    {
        "username": "eval",
        "farm_name": "row_spacing",
        "field_name": "brown",
        "mission_date": "2021-06-08"
    },
    {
        "username": "eval",
        "farm_name": "SaskatoonEast",
        "field_name": "Stevenson5NW",
        "mission_date": "2022-06-20"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "Vaderstad",
        "mission_date": "2022-06-16"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis2",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko10",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-14"
    }
]

eval_test_sets = [
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko9S",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko11",
        "mission_date": "2022-06-07"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko15",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "SaskatoonEast",
        "field_name": "Stevenson5SW",
        "mission_date": "2022-06-13"
    },
    {
        "username": "eval",
        "farm_name": "Davidson",
        "field_name": "Stone11NE",
        "mission_date": "2022-06-03"
    },
    {
        "username": "eval",
        "farm_name": "BlaineLake",
        "field_name": "Serhienko12",
        "mission_date": "2022-06-14"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis3",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "Biggar",
        "field_name": "Dennis5",
        "mission_date": "2021-06-12"
    },
    {
        "username": "eval",
        "farm_name": "UNI",
        "field_name": "CNH-DugoutROW",
        "mission_date": "2022-05-30"
    },
]

eval_GWHD_test_sets = [
    {
        "username": "eval",
        "farm_name": "CIMMYT_1",
        "field_name": "CIMMYT_1",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "CIMMYT_2",
        "field_name": "CIMMYT_2",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "CIMMYT_3",
        "field_name": "CIMMYT_3",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "KSU_1",
        "field_name": "KSU_1",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "KSU_2",
        "field_name": "KSU_2",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "KSU_3",
        "field_name": "KSU_3",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "KSU_4",
        "field_name": "KSU_4",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "Terraref_1",
        "field_name": "Terraref_1",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "Terraref_2",
        "field_name": "Terraref_2",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_1",
        "field_name": "UQ_1",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_2",
        "field_name": "UQ_2",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_3",
        "field_name": "UQ_3",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_4",
        "field_name": "UQ_4",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_5",
        "field_name": "UQ_5",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_6",
        "field_name": "UQ_6",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_7",
        "field_name": "UQ_7",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_8",
        "field_name": "UQ_8",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_9",
        "field_name": "UQ_9",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_10",
        "field_name": "UQ_10",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "UQ_11",
        "field_name": "UQ_11",
        "mission_date": "2023-08-29"
    },
    {
        "username": "eval",
        "farm_name": "Usask_1",
        "field_name": "Usask_1",
        "mission_date": "2023-08-29"
    }
]


eval_fixed_patch_num_baselines = [
    "set_of_27_250_patches",
    "set_of_27_500_patches",
    "set_of_27_1000_patches",
    "set_of_27_2000_patches",
    "set_of_27_4000_patches",
    "set_of_27_8000_patches",
    "set_of_27_16000_patches",
    "set_of_27_24000_patches",
    "set_of_27_32000_patches",
    "set_of_27_38891_patches"
]

ns_fixed_patch_num_baselines = [
    "ns_set_of_27_500_patches",
    "ns_set_of_27_2000_patches",
    "ns_set_of_27_8000_patches"
]

eval_single_630_baselines = [
    "BlaineLake_River_2021-06-09_630_patches",
    "BlaineLake_Lake_2021-06-09_630_patches",
    "BlaineLake_HornerWest_2021-06-09_630_patches",
    "UNI_LowN1_2021-06-07_630_patches",
    "BlaineLake_Serhienko9N_2022-06-07_630_patches",
    "row_spacing_nasser_2021-06-01_630_patches",
    "Biggar_Dennis1_2021-06-04_630_patches",
    "Biggar_Dennis3_2021-06-04_630_patches",
    "MORSE_Dugout_2022-05-27_630_patches",
    "MORSE_Nasser_2022-05-27_630_patches",
    "row_spacing_brown_2021-06-01_630_patches",
    "row_spacing_nasser2_2022-06-02_630_patches",
    "Saskatoon_Norheim1_2021-05-26_630_patches",
    "Saskatoon_Norheim2_2021-05-26_630_patches",
    "Saskatoon_Norheim4_2022-05-24_630_patches",
    "Saskatoon_Norheim5_2022-05-24_630_patches",
    "UNI_Brown_2021-06-05_630_patches",
    "UNI_Dugout_2022-05-30_630_patches",
    "UNI_LowN2_2021-06-07_630_patches",
    "UNI_Sutherland_2021-06-05_630_patches",
    "Saskatoon_Norheim1_2021-06-02_630_patches",
    "row_spacing_brown_2021-06-08_630_patches",
    "SaskatoonEast_Stevenson5NW_2022-06-20_630_patches",
    "UNI_Vaderstad_2022-06-16_630_patches",
    "Biggar_Dennis2_2021-06-12_630_patches",
    "BlaineLake_Serhienko10_2022-06-14_630_patches",
    "BlaineLake_Serhienko9S_2022-06-14_630_patches"
]

eval_diverse_630_baselines = [
    "set_of_27_630_patches"
]



eval_best_to_worst_baselines = [
    "set_of_27_best_to_worst_250_patches",
    "set_of_27_best_to_worst_500_patches",
    "set_of_27_best_to_worst_1000_patches",
    "set_of_27_best_to_worst_2000_patches",
    "set_of_27_best_to_worst_4000_patches",
    "set_of_27_best_to_worst_8000_patches",
]









def get_mapping_for_test_set(test_set_image_set_dir):

    mapping = {}
    print(test_set_image_set_dir)
    results_dir = os.path.join(test_set_image_set_dir, "model", "results")
    for result_dir in glob.glob(os.path.join(results_dir, "*")):
        request_path = os.path.join(result_dir, "request.json")
        request = json_io.load_json(request_path)
        if request["results_name"] in mapping:
            raise RuntimeError("Duplicate result name: {}, {}".format(test_set_image_set_dir, request["results_name"]))
        mapping[request["results_name"]] = request["request_uuid"]
    return mapping



def predict_on_test_sets(test_sets, baselines):

    for test_set in test_sets:

        test_set_image_set_dir = os.path.join("usr", "data",
                                                    test_set["username"], "image_sets",
                                                    test_set["farm_name"],
                                                    test_set["field_name"],
                                                    test_set["mission_date"])
        
        print("\n\nProcessing test_set: {}\n\n".format(test_set_image_set_dir))
        

        annotations_path = os.path.join(test_set_image_set_dir, "annotations", "annotations.json")
        annotations = annotation_utils.load_annotations(annotations_path)

        image_names = []
        for image_name in annotations.keys():
            if len(annotations[image_name]["test_regions"]) > 0:
                image_names.append(image_name)

        metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
        metadata = json_io.load_json(metadata_path)


        regions = []
        for image_name in image_names:
            regions.append([[0, 0, metadata["images"][image_name]["height_px"], metadata["images"][image_name]["width_px"]]])

        for baseline in baselines:

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

            
            request_uuid = str(uuid.uuid4())
            request = {
                "request_uuid": request_uuid,
                "start_time": int(time.time()),
                "image_names": image_names,
                "regions": regions,
                "save_result": True,
                "results_name": baseline["model_name"], # "trimming_eval_50_percent_overlap", #baseline["model_name"], # "no_prune_" + baseline["model_name"],
                "results_message": ""
            }

            request_path = os.path.join(test_set_image_set_dir, "model", "prediction", 
                                        "image_set_requests", "pending", request_uuid + ".json")

            json_io.save_json(request_path, request)
            print("running process_predict")
            server.process_predict(item)




def run():

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.basicConfig(level=logging.INFO)

    server.sch_ctx["switch_queue"] = LockQueue()
    server.sch_ctx["auto_select_queue"] = LockQueue()
    server.sch_ctx["prediction_queue"] = LockQueue()
    server.sch_ctx["training_queue"] = LockQueue()
    server.sch_ctx["baseline_queue"] = LockQueue()


    # div_baselines = [
    #     "set_of_5_3150_patches",
    #     "set_of_10_3150_patches",
    #     "set_of_15_3150_patches",
    #     "set_of_20_3150_patches",
    #     "set_of_25_3150_patches",
    # ]

    # b = [
    #     "EARLY_31500_patches",
    #     "LATE_38934_patches"
    # ]

    # b = [
    #     "RAND1_39312_patches",
    #     "RAND2_31122_patches"
    # ]

    # b = [
    #     "LOWEED_47250_patches",
    #     "HIWEED_23184_patches"
    # ]

    b = [
        # "MANLOWEED_25578_patches",
        "MANHIWEED_44856_patches"
    ]

    test_baselines = []
    for baseline in b:
        for rep_num in range(1):
            test_baselines.append({
                "model_name": baseline + "_rep_" + str(rep_num),
                "model_creator": "ds_splits"
            })

    # id_image_sets = ds_splits.get_split("EARLY", "ID")
    # id_image_sets = ds_splits.get_split("RAND1", "ID")
    id_image_sets = ds_splits.get_split("LOWEED", "ID")
    predict_on_test_sets(id_image_sets, test_baselines)

    # id_image_sets = ds_splits.get_split("LATE", "ID")
    # id_image_sets = ds_splits.get_split("RAND2", "ID")
    id_image_sets = ds_splits.get_split("HIWEED", "ID")
    predict_on_test_sets(id_image_sets, test_baselines)

    exit()

    test_baselines = []
    for baseline in eval_best_to_worst_baselines: #div_baselines:
        for rep_num in range(1, 2): #10, 20):
            test_baselines.append({
                "model_name": baseline + "_rep_" + str(rep_num),
                "model_creator": "eval"
            })
    # test_baselines = [
    #     {
    #         "model_name": "ns_set_of_27_8000_patches_rep_0",
    #         "model_creator": "new_split"
    #     }
    # ]

    predict_on_test_sets(eval_test_sets, test_baselines)
    # predict_on_test_sets(in_domain_test_sets, test_baselines)


    # test_baselines = [
    #     {
    #         "model_name": "GWHD_official_train_3_epochs_no_patches", #_fixed_epoch_num",
    #         "model_creator": "eval"
    #     }
    # ]
    # predict_on_test_sets(eval_GWHD_test_sets, test_baselines)


if __name__ == "__main__":
    run()
