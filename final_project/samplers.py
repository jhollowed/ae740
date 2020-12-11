import pdb
import numpy as np
from shapely import geometry
import geopandas as gpd
import matplotlib.pyplot as plt
import util


# =====================================================================================
# =====================================================================================


def sample_bilateral_poles(footprint, sym_axis=None):
    '''
    Distribute a pair of poles with bilateral symmetry within a given footprint

    Parameters
    ----------
    footprint : shapely Polygon object
        the input footprint
    
    Returns
    -------
    footprint : a shapely Polygon object
        the resulting footprint
    '''
    
    # find axis of symmetry
    if(sym_axis is None):
        x, y = np.array(footprint.exterior.coords.xy).T[:-1].T
        if(np.sum(x) > np.sum(y)): sym_axis = 'y'
        else: sym_axis = 'x'
    if(sym_axis == 'x'): pole2_signs = [-1, 1]
    elif(sym_axis == 'y'): pole2_signs = [1, -1]


    # place poles by reflection, rejection
    minx, _, maxx, maxy = footprint.bounds
    while True:
        pole1 = geometry.Point(np.random.uniform(minx, maxx), np.random.uniform(0, maxy))
        pole2 = geometry.Point(pole2_signs[0] * pole1.x, pole2_signs[1] * pole1.y)
        if footprint.contains(pole1) and footprint.contains(pole2):
            break
    return [pole1, pole2] 


# --------------------------------------------------------------------------------------


def sample_biradial_poles(footprint):
    '''
    Distribute a pair of poles with bilateral symmetry within a given footprint

    Parameters
    ----------
    footprint : shapely Polygon object
        the input footprint
    
    Returns
    -------
    footprint : a shapely Polygon object
        the resulting footprint
    '''

    minx, miny, maxx, maxy = footprint.bounds
    while True:
        pole1 = geometry.Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        pole2 = geometry.Point(-pole1.x, -pole1.y)
        if footprint.contains(pole1) and footprint.contains(pole2):
            break
    return [pole1, pole2] 


# --------------------------------------------------------------------------------------


def sample_bilateral_footprint(nstakes):
    '''
    Generate a random convex polygon with bilateral symmetry

    Parameters
    ----------
    nstakes : int
        number of polygon vertices
    
    Returns
    -------
    footprint : a shapely Polygon object
        the resulting footprint
    '''

    assert(nstakes % 2 == 0), 'nstakes must be divisible by 2'

    # first draw an asymmetric footprint, obtaining one with nstakes/2 vertices 
    # in the upper-half plane, and enforced convexity, via rejection
   
    footprint_accepted = False
    while not footprint_accepted:
        
        nstakes_upper = 0
        while nstakes_upper != nstakes/2:        
            
            trial_footprint = sample_asymmetric_footprint(nstakes)
            stakes_x, stakes_y = np.array(trial_footprint.exterior.coords.xy).T[:-1].T
            
            # check number of stakes for symmetry
            upper_mask = stakes_y > 0
            nstakes_upper = np.sum(upper_mask)

        # reflect about y=0
        stakes_x = np.hstack([stakes_x[upper_mask], stakes_x[upper_mask][::-1]]) 
        stakes_y = np.hstack([stakes_y[upper_mask], -stakes_y[upper_mask][::-1]])
        stakes = np.vstack([stakes_x, stakes_y])
        
        footprint = normalize_footprint(geometry.Polygon(stakes.T))
        footprint_accepted = util.is_convex(footprint)
    
    return footprint


# --------------------------------------------------------------------------------------


def sample_biradial_footprint(nstakes):
    '''
    Generate a random convex polygon with D_2 biradial symmetry

    Parameters
    ----------
    nstakes : int
        number of polygon vertices
    
    Returns
    -------
    footprint : a shapely Polygon object
        the resulting footprint
    '''

    assert(nstakes % 4 == 0), 'nstakes must be divisible by 4'

    # first draw an asymmetric footprint, obtaining one with nstakes/2 vertices 
    # in the upper-half plane, and enforced convexity, via rejection
   
    footprint_accepted = False
    while not footprint_accepted:
        
        nstakes_quad1 = 0
        while nstakes_quad1 != nstakes/4:        
            
            trial_footprint = sample_asymmetric_footprint(nstakes)
            stakes_x, stakes_y = np.array(trial_footprint.exterior.coords.xy).T[:-1].T
            
            # check number of stakes for symmetry
            quad_mask = np.logical_and(stakes_y > 0, stakes_x > 0)
            nstakes_quad1 = np.sum( quad_mask )

        # reflect about y=0 and x=0 
        stakes_x = np.hstack([stakes_x[quad_mask], -stakes_x[quad_mask][::-1], 
                              -stakes_x[quad_mask], stakes_x[quad_mask][::-1]]) 
        stakes_y = np.hstack([stakes_y[quad_mask], stakes_y[quad_mask][::-1],
                              -stakes_y[quad_mask], -stakes_y[quad_mask][::-1]])
        stakes = np.vstack([stakes_x, stakes_y])

        footprint = normalize_footprint(geometry.Polygon(stakes.T))
        footprint_accepted = util.is_convex(footprint)
    
    return footprint
    

# --------------------------------------------------------------------------------------


def sample_asymmetric_footprint(nstakes):
    '''
    Generate a random convex polygon in the algorithm of Sander Verdonschot
    (http://cglab.ca/~sander/misc/ConvexGeneration/convex.html)

    Parameters
    ----------
    nstakes : int
        number of polygon vertices
    
    Returns
    -------
    footprint : a shapely Polygon object
        the resulting footprint
    '''
    
    # generate two lists of uniform random cartesian coordinates, and sort
    stakes = np.random.rand(nstakes, 2)
    x, y = np.sort(stakes.T[0]), np.sort(stakes.T[1])
    
    # isolate extreme points, and randomly divide the interior points into two chains
    x_ex = np.array([x[0], x[1]])
    y_ex = np.array([y[0], y[1]])
    
    x_chain_mask = np.random.rand(nstakes - 2) > 0.5
    x_chain1 = np.hstack([x_ex[0], x[1:-1][x_chain_mask], x_ex[1]])
    x_chain2 = np.hstack([x_ex[0], x[1:-1][~x_chain_mask], x_ex[1]])
    
    y_chain_mask = np.random.rand(nstakes - 2) > 0.5
    y_chain1 = np.hstack([y_ex[0], y[1:-1][y_chain_mask], y_ex[1]])
    y_chain2 = np.hstack([y_ex[0], y[1:-1][~y_chain_mask], y_ex[1]])
    
    # extract vector components
    x_vec = np.hstack([np.diff(x_chain1), np.diff(x_chain2)])
    y_vec = np.hstack([np.diff(y_chain1), np.diff(y_chain2)])

    # randomly pair them up and form vectors
    np.random.shuffle(x_vec)
    vectors = np.array([x_vec, y_vec]).T

    # sort by angle
    theta = np.arctan2(vectors.T[1], vectors.T[0])
    theta_sorter = np.argsort(theta)
    
    # lay end to end to build polygon
    stakes_x = np.cumsum(x_vec[theta_sorter])
    stakes_y = np.cumsum(y_vec[theta_sorter])
    stakes = np.vstack([stakes_x, stakes_y])

    # build polygon object
    footprint = normalize_footprint(geometry.Polygon(stakes.T))
    
    return footprint


# --------------------------------------------------------------------------------------


def normalize_footprint(footprint):
    
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




# =====================================================================================
    

if __name__ == '__main__':
    for i in range(10):
        #s = sample_bilateral_footprint()
        s = sample_biradial_footprint(8)
        x,y = s.exterior.xy
        plt.plot(x, y, '--ok')
        plt.show()
        #if(len(yy) != 9): pdb.set_trace()
