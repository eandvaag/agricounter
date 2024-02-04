import logging
from shapely import Point, Polygon
import numpy as np

from models.common import box_utils



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

    single_types = ["Point", "LineString", "Polygon"]
    multi_types = ["MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"]

    if r.geom_type in single_types:
        geoms = [r]
    elif r.geom_type in multi_types:
        geoms = list(r.geoms)
    else:
        logger.error("Unknown geometry type returned by shapely intersection: {}".format(r.geom_type))
        geoms = []

    intersect_regions = []
    for geom in geoms:
        if geom.geom_type == "Polygon":
            if len(list(geom.exterior.coords)) > 0:
                intersect_regions.append(list(geom.exterior.coords)[:-1])
        elif geom.geom_type == "Point":
            intersect_regions.append(list(geom.coords))
        elif geom.geom_type == "LineString":
            intersect_regions.append(list(geom.coords))

    intersects = len(intersect_regions) > 0
    return intersects, intersect_regions

def get_poly_area(p):
    return Polygon(p).area


def get_poly_bbox(p):
    p_arr = np.array(p)
    return [int(np.min(p_arr[:, 0])), int(np.min(p_arr[:, 1])), int(np.max(p_arr[:, 0])), int(np.max(p_arr[:, 1]))]







def get_bbox_visibility_mask(boxes, patch_clipped_boxes, region, vis_thresh):

    visibilities = []

    box_areas = box_utils.box_areas_np(boxes)

    for i in range(boxes.shape[0]):

        pcb = patch_clipped_boxes[i]

        poly_pcb =  [
            [pcb[0], pcb[1]],
            [pcb[0], pcb[3]],
            [pcb[2], pcb[3]],
            [pcb[2], pcb[1]]
        ]
        


        p_a = Polygon(poly_pcb)
        p_b = Polygon(region)
        r = p_a.intersection(p_b, grid_size=1)

        # area of the twice-clipped box divided by original box area
        visibility = r.area / box_areas[i]
        visibilities.append(visibility)

    mask = np.array(visibilities) > vis_thresh

    return mask


