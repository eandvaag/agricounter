# import rasterio
import numpy as np
# import osmnx as ox
# import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.lines import Line2D

import os

from io_utils import json_io
from exp_runner import eval_test_sets, eval_in_domain_test_sets


my_plot_colors = ["orangered", "royalblue", "forestgreen", "orange", "mediumorchid"]


def show_image_set_locations(id_image_sets, ood_image_sets, our_dirname):
    locs = [] #{}
    for i, image_set_type in enumerate([id_image_sets, ood_image_sets]):
        # if i == 0:
        #     label = "Training"
        # else:
        #     label = "Test"
        # locs[label] = []

        for test_set in image_set_type:
            test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
            
            metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            lons = []
            lats = []
            for image_name in metadata["images"]:
                lons.append(metadata["images"][image_name]["longitude"])
                lats.append(metadata["images"][image_name]["latitude"])

            mean_lon = np.mean(lons)
            mean_lat = np.mean(lats)
            locs.append((mean_lon, mean_lat))

    print("reading provinces file")
    provinces = gpd.read_file("provinces")  # note that I point to the shapefile "directory" containg all the individual files
    # print("read")
    # sk = provinces.iloc[[7]].to_crs("EPSG:4326")    # I'll explain this later, I'm converting to a different coordinate reference system
    sk = provinces.iloc[[7]].to_crs({'proj':'longlat', 'ellps':'WGS84', 'datum':'WGS84'}) #to_crs("EPSG:4326")
    # print("to_crs done")
    # province = "Saskatchewan"
    # sk = provinces.query("PRENAME == @province").copy()
    # fig = plt.figure()

    f, ax = plt.subplots(figsize=(10, 10))
    sk.plot(edgecolor="black", facecolor="#b8fca7", ax=ax)
    # plt.title("Image Acquisition Locations in Saskatchewan")
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    # plt.facecolor("lightgrey")
    # plt.rcParams['figure.facecolor'] = 'lightblue'
    # fig.patch.set_facecolor('red')
    # .set_facecolor("#dae3f5") #lightgrey")



    axins = zoomed_inset_axes(ax, 4, loc=1)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="white", ec="red")

    # axins.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off
    # axins.tick_params(
    # axis='y',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off

    axins.set_xticks([])
    axins.set_yticks([])

    axins.spines['bottom'].set_color('red')
    axins.spines['top'].set_color('red') 
    axins.spines['right'].set_color('red')
    axins.spines['left'].set_color('red')

    ax.set_xlim([-111, -92])


    # plt.setp(axins.get_xticklabels(), visible=False)
    # plt.setp(axins.get_yticklabels(), visible=False)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    # i = 0
    # for label in locs.keys():
    axins.scatter([x[0] for x in locs], [x[1] for x in locs], label="Acquisition Location", c="black", marker="x", alpha=0.5)
    # ax.scatter([x[0] for x in locs], [x[1] for x in locs], label="Acquisition Location", c="black", marker="x", alpha=0.5)

    # i += 1

    cities = {
        "Saskatoon": [52.139722, -106.686111],
        "Regina": [50.454722, -104.606667]
    }


    for city in cities.keys():
        # axins.scatter([cities[city][1]], [cities[city][0]], marker="*", c="red")
        # axins.annotate(city, (cities[city][1], cities[city][0]+0.05), ha="right")

        ax.scatter([cities[city][1]], [cities[city][0]], marker="*", c="red")
        ax.annotate(city, (cities[city][1], cities[city][0]+0.1), ha="center")       

    axins.scatter([cities["Saskatoon"][1]], [cities["Saskatoon"][0]], marker="*", c="red")
    axins.annotate("Saskatoon", (cities["Saskatoon"][1], cities["Saskatoon"][0]+0.03), ha="right")

    # plt.legend()
    # ax.legend()
    axins.legend(handles=[Line2D([0], [0], marker='x', color='w', markeredgecolor="black", alpha=0.5, label='Acquisition Location',
                          )],
                          loc="lower center", 
                          framealpha=1.0,
                          facecolor="white",
                          edgecolor="black",
                          bbox_to_anchor=(0.5, -0.07))

    # plt.ylim([49, 54])
    # plt.ylim([51, 53])
    # plt.xlim([-108.5, -106])

    # plt.xlim([-111, -96])

    # plt.scatter([-106], [52], marker="*", c="red")
    # plt.annotate("Saskatoon", (-106-0.1, 52))
    plt.tight_layout()

    out_path = os.path.join("eval_charts", our_dirname, "locations_inset.svg") #png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path) #, dpi=800)




def show_image_set_locations_plain(id_image_sets, ood_image_sets, our_dirname):
    locs = [] #{}
    for i, image_set_type in enumerate([id_image_sets, ood_image_sets]):
        # if i == 0:
        #     label = "Training"
        # else:
        #     label = "Test"
        # locs[label] = []

        for test_set in image_set_type:
            test_set_str = test_set["username"] + ":" + test_set["farm_name"] + ":" + test_set["field_name"] + ":" + test_set["mission_date"]
            test_set_image_set_dir = os.path.join("usr", "data",
                                                            test_set["username"], "image_sets",
                                                            test_set["farm_name"],
                                                            test_set["field_name"],
                                                            test_set["mission_date"])
            
            metadata_path = os.path.join(test_set_image_set_dir, "metadata", "metadata.json")
            metadata = json_io.load_json(metadata_path)

            lons = []
            lats = []
            for image_name in metadata["images"]:
                lons.append(metadata["images"][image_name]["longitude"])
                lats.append(metadata["images"][image_name]["latitude"])

            mean_lon = np.mean(lons)
            mean_lat = np.mean(lats)
            locs.append((mean_lon, mean_lat))

    print("reading provinces file")
    provinces = gpd.read_file("provinces")  # note that I point to the shapefile "directory" containg all the individual files
    # print("read")
    sk = provinces.iloc[[7]].to_crs("EPSG:4326")    # I'll explain this later, I'm converting to a different coordinate reference system
    # print("to_crs done")
    # province = "Saskatchewan"
    # sk = provinces.query("PRENAME == @province").copy()
    # fig = plt.figure()

    p = sk.plot(edgecolor="black", facecolor="white", figsize=(5, 10))
    plt.title("Image Acquisition Locations in Saskatchewan")
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    # plt.facecolor("lightgrey")
    # plt.rcParams['figure.facecolor'] = 'lightblue'
    # fig.patch.set_facecolor('red')
    p.set_facecolor("#dae3f5") #lightgrey")



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    # i = 0
    # for label in locs.keys():
    plt.scatter([x[0] for x in locs], [x[1] for x in locs], label="Acquisition Location", c="black", marker="x", alpha=0.5)
    # i += 1

    cities = {
        "Saskatoon": [52.139722, -106.686111],
        "Regina": [50.454722, -104.606667]
    }


    for city in cities.keys():
        plt.scatter([cities[city][1]], [cities[city][0]], marker="*", c="red")
        plt.annotate(city, (cities[city][1], cities[city][0]+0.1))

    plt.legend()

    # plt.ylim([49, 54])
    # plt.ylim([51, 53])
    # plt.xlim([-108.5, -106])

    # plt.scatter([-106], [52], marker="*", c="red")
    # plt.annotate("Saskatoon", (-106-0.1, 52))
    plt.tight_layout()

    out_path = os.path.join("eval_charts", our_dirname, "locations.png")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=800)


if __name__ == "__main__":

    out_dirname = "locs"

    show_image_set_locations(eval_in_domain_test_sets, eval_test_sets, out_dirname)