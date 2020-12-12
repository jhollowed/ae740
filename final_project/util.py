import numpy as np
from shapely import geometry
from sympy import Plane, Point3D
import networkx as nx



# ==================================================================================================
# ==================================================================================================



def z_cross_product(v1,v2,v3):
    '''
    Computes the cross product of two vectors sharing a point

    Parameters
    ----------
    v1, v2, v3 : float
        vector coordinates
    
    Returns
    float
        The cross poduct
    '''
    return (v1[0]-v2[0]) * (v2[1]-v3[1]) - (v1[1]-v2[1]) * (v2[0]-v3[0])


# --------------------------------------------------------------------------------------------------


def is_convex(poly):
    '''
    Check if a polygon is convex, assuming it is a simple polygon

    Parameters
    ----------
    poly : shapely Polygon object
        The input polygon

    Returns
    -------
    bool
        Whether or not the returned shape is everywhere convex
    '''

    vertices = np.array(poly.exterior.coords.xy).T
    if len(vertices)<4:
        return True
    signs= [z_cross_product(v1,v2,v3) > 0 for v1,v2,v3 in zip(vertices[2:],vertices[1:],vertices)]
    return all(signs) or not any(signs)


# --------------------------------------------------------------------------------------


def normalize_footprint(footprint):
    '''
    Normalizes a footprint, setting its long edge to be found in the x-dimension, 
    and scaling that long edge to be of length 100 (meant to be interpreted as inches,
    matching the length of the Xmid).

    Parameters
    ----------
    footprint : Shapely Polygon object
        the footprint to normalzie

    Returns
    -------
    normalized_footprint : Shapely Polygon object
        the normalized footprint
    '''
    
    stakes_x, stakes_y = np.array(footprint.exterior.coords.xy).T[:-1].T
    stakes = np.vstack([stakes_x, stakes_y])
    
    origin_offset_x = np.mean([np.max(stakes_x), np.min(stakes_x)])
    origin_offset_y = np.mean([np.max(stakes_y), np.min(stakes_y)])
    stakes[0] -= origin_offset_x
    stakes[1] -= origin_offset_y

    # move longest side to x dimension, if currently oriented in y
    x_width = np.max(stakes[0]) - np.min(stakes[0]) 
    y_width = np.max(stakes[1]) - np.min(stakes[1])
    if(y_width >  x_width):
        stakes = stakes[::-1]
        stakes /= y_width
    else:
        stakes /= x_width
    stakes *= 100

    # order vertices by radial position
    theta = np.arctan2(stakes[1], stakes[0])
    theta_sorter = np.argsort(theta)
    stakes_out = stakes.T[theta_sorter]
    
    footprint_transformed = geometry.Polygon(stakes_out)
    return footprint_transformed
