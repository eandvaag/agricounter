import logging
from shapely import Point, Polygon
import numpy as np


def get_contained_inds_for_points(points, regions):

    shp_points = []
    for point in points:
        shp_points.append(Point(point))

    contains = np.full(len(points), False)
    
    for region in regions:
        shp_region = Polygon(region)
        for i, shp_point in enumerate(shp_points):
            if shp_region.contains(shp_point):
                contains[i] = True
            
    return np.where(contains)[0]



def get_intersection_polys(a, b):

    logger = logging.getLogger(__name__)

    p_a = Polygon(a)
    p_b = Polygon(b)
    r = p_a.intersection(p_b, grid_size=1)

    # print(r.geom_type)
    single_types = ["Point", "LineString", "Polygon"]
    multi_types = ["MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"]

    if r.geom_type in single_types:
        geoms = [r]
    elif r.geom_type in multi_types:
        geoms = list(r.geoms)
    else:
        logger.error("Unknown geometry type returned by shapely intersection: {}".format(r.geom_type))
        geoms = []

    # print("geoms", geoms)
    intersect_regions = []
    for geom in geoms:
        if geom.geom_type == "Polygon":
            if len(list(geom.exterior.coords)) > 0:
                intersect_regions.append(list(geom.exterior.coords)[:-1])
        elif geom.geom_type == "Point":
            intersect_regions.append(list(geom.coords))
        elif geom.geom_type == "LineString":
            intersect_regions.append(list(geom.coords))

    # if r.geom_type == "Polygon":
    #     intersect_regions.append(list(r.exterior.coords)[:-1])
    # elif r.geom_type == "Point":
    #     intersect_regions.append(list(r.coords))
    # elif r.geom_type == "LineString":
    #     intersect_regions.append(list(r.coords))
    # elif r.geom_type == "GeometryCollection":
    #     for geom in list(r.geoms):

    #         intersect_regions.append(list(geom.coords))

    # intersects = len(coords) > 0
    intersects = len(intersect_regions) > 0
    return intersects, intersect_regions
    

    # intersects = r.geom_type == "Polygon"
    # print("r.geom_type", r.geom_type)
    # if intersects:
    #     coords = list(r.exterior.coords)[:-1]
    # else:
    #     coords = []
    # return intersects, coords


def get_poly_area(p):
    return Polygon(p).area


def get_poly_bbox(p):
    p_arr = np.array(p)
    return [int(np.min(p_arr[:, 0])), int(np.min(p_arr[:, 1])), int(np.max(p_arr[:, 0])), int(np.max(p_arr[:, 1]))]