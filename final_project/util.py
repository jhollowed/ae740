import numpy as np
from shapely import geometry
from sympy import Plane, Point3D
import networkx as nx

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


