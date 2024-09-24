import pdb
import numpy as np
from shapely import geometry
import geopandas as gpd
import matplotlib.pyplot as plt
import util
import shelter



# =====================================================================================
# =====================================================================================



# ============================ INDIVIDUAL SAMPLERS ====================================


def sample_bilateral_poles(footprint, sym_axis=None):
    '''
    Distribute a pair of poles with bilateral symmetry within a given footprint

    Parameters
    ----------
    footprint : Shapely Polygon object
        the input footprint
    
    Returns
    -------
    footprint : a Shapely Polygon object
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
    footprint : Shapely Polygon object
        the input footprint
    
    Returns
    -------
    footprint : a Shapely Polygon object
        the resulting footprint
    '''

    minx, miny, maxx, maxy = footprint.bounds
    while True:
        pole1 = geometry.Point(np.random.uniform(0, maxx), np.random.uniform(0, maxy))
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
    footprint : a Shapely Polygon object
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
        
        footprint = util.normalize_footprint(geometry.Polygon(stakes.T))
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
    footprint : a Shapely Polygon object
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

        footprint = util.normalize_footprint(geometry.Polygon(stakes.T))
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
    footprint : a Shapely Polygon object
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
    footprint = util.normalize_footprint(geometry.Polygon(stakes.T))
    
    return footprint



# ============================ ENSEMBLE SAMPLERS ====================================



def random_sampling(nstakes, N, symmetry='biradial', normed=True):
    '''
    Randomly sample shelters in normalized parameter space

    Parameters
    ----------
    nstakes : int
        Sample shelters with this number of total stakes
    N : int
        Number of shelters to sample
    normed : bool
        Whether or not to normalize the free parameters with respect to the footprint size
        of each sample
    '''

    # number of free parameters
    if(symmetry == 'biradial'):
        assert(nstakes % 4 == 0), 'nstakes must be divisible by 4'
        free_stakes = int((nstakes/4))
    if(symmetry == 'bilateral'): 
        assert(nstakes % 2 == 0), 'nstakes must be divisible by 2'
        free_stakes = int((nstakes/2))
    free_poles = 1
    
    nparam = (free_stakes + free_poles) * 2  
    free_params = np.zeros((N, int(nparam/2), 2))
    
    sampler = shelter.shelter(nstakes, symmetry=symmetry)
    ve = np.zeros(N)
    wp = np.zeros(N)
    
    # sample shelters, compute volumetric efficiency and weather performance
    for i in range(N):
        if(i%100 == 0): print('drawing sample {}/{}'.format(i, N))
        sampler.sample_footprint()
        sampler.sample_poles()
        sampler.pitch()
        this_params = sampler.get_free_params(normed = normed)
        free_params[i] = np.vstack(this_params)
        ve[i] = shelter.compute_volumetric_efficiency(sampler)
        wp[i] = shelter.compute_weather_performance(sampler)

    # compute quality metric
    quality = (ve/np.max(ve)) * (wp/np.max(wp))
    shelter_quality_metrics = np.array([ve, wp, quality])
    
    # return sample inputs, outputs
    return free_params, shelter_quality_metrics

# --------------------------------------------------------------------------------------

def latin_hypercube_sampling(nstakes, N, symmetry='biradial', normed=True):
    
    # number of free parameters
    if(symmetry == 'biradial'):
        assert(nstakes % 4 == 0), 'nstakes must be divisible by 4'
        free_stakes = int((nstakes/4))
    if(symmetry == 'bilateral'): 
        assert(nstakes % 2 == 0), 'nstakes must be divisible by 2'
        free_stakes = int((nstakes/2))
    free_poles = 1
    
    nparam = (free_stakes + free_poles) * 2  
    free_params = np.zeros((N, int(nparam/2), 2))
    
    sampler = shelter.shelter(nstakes, symmetry=symmetry)
    ve = np.zeros(N)
    wp = np.zeros(N)
    
    # sample shelters, compute volumetric efficiency and weather performance
    
    # latin hypercube sampling
    scaled_samples = np.array(lhsmdu.sample(free_params, N)).T
    stakes = np.hstack([np.ones(N).reshape(N,1), scaled_samples[:, :-2]])
    poles = scaled_samples[:, -2:]
    
    # get quality metrics per sample point
    for i in range(N):
        if(i%100 == 0): print('drawing sample {}/{}'.format(i, N))
        
        # extract stake, pole positions, add in constrained stake_x as 1 (will be normed)
        stakes = stakes[i].reshape((nstakes/4, 2))
        pole = poles[i]
        
        # get sample outputs
        sampler.choose_footprint(stakes, normalize=True)
        sampler.choose_stakes(pole)
        sampler.pitch()
        this_params = sampler.get_free_params(normed = normed)
        free_params[i] = np.vstack(this_params)
        ve[i] = shelter.compute_volumetric_efficiency(sampler)
        wp[i] = shelter.compute_weather_performance(sampler)
    
    # compute quality metric
    quality = (ve/np.max(ve)) * (wp/np.max(wp))
    
    # return sample inputs, outputs
    return free_params, quality










'''
def random_sampling(nstakes, N, symmetry = 'biradial'):
    
    Randomly sample shelters in normalized parameter space

    Parameters
    ----------
    nstakes : int
        Sample shelters with this number of total stakes
    N : int
        Number of shelters to sample
    

    if(symmetry == 'biradial'):
        assert(nstakes % 4 == 0), 'nstakes must be divisible by 4'
        free_stakes = int((nstakes/4))
    if(symmetry == 'bilateral'): 
        assert(nstakes % 2 == 0), 'nstakes must be divisible by 2'
        free_stakes = int((nstakes/2))
    
    g = shelter.shelter(nstakes, symmetry=symmetry)
    sx = np.zeros((N, free_stakes))
    sy = np.zeros((N, free_stakes))
    px = np.zeros(N)
    py = np.zeros(N)
    ve = np.zeros(N)
    wp = np.zeros(N)
    pp = np.zeros(N)

    for i in range(N):
        if(i%100 == 0): print(i)
        g.sample_footprint()
        g.sample_poles()
        g.pitch()
        fp = g.get_free_params()
        sx[i] = fp[0][:,0]
        sy[i]= fp[0][:,1] / sx[i]
        px[i]= fp[1][0] / sx[i]
        py[i]= fp[1][1] / (sy[i] * sx[i])
        ve[i] = shelter.compute_volumetric_efficiency(g)
        wp[i] = shelter.compute_weather_performance(g)
    pp = (ve/np.max(ve)) * (wp/np.max(wp))
    
    f = plt.figure(figsize=(10,8))
    ax4 = f.add_subplot(221, projection='3d')
    ax5 = f.add_subplot(222, projection='3d')
    ax6 = f.add_subplot(223, projection='3d')
    
    sctr4 = ax4.scatter(px, py, sy[:,0], c=ve, marker='o', cmap=plt.cm.viridis, alpha=0.5)
    sctr5 = ax5.scatter(px, py, sy[:,0], c=wp, marker='o', cmap=plt.cm.viridis, alpha=0.5)
    sctr6 = ax6.scatter(px, py, sy[:,0], c=pp, marker='o', cmap=plt.cm.viridis, alpha=0.5)

    scatters = [sctr4, sctr5, sctr6]
    metrics = [r'$\epsilon_V$', r'$P_W$', r'$\epsilon_VP_W$']
    axes = [ax4, ax5, ax6]
    xmid_s = 60
    xmid_lw = 2
    
    for i in range(len(axes)):
        axes[i].scatter([xmid1p.norm_px], [xmid1p.norm_py], [xmid1p.norm_y], color='r', marker='x', s=xmid_s, lw=xmid_lw)
        axes[i].scatter([xmid2p.norm_px], [xmid2p.norm_py], [xmid2p,norm_y], color='r', marker='^', s=xmid_s, lw=xmid_lw)
        axes[i].set_xlabel(r'$p_x / s_x$', fontsize=14)
        axes[i].set_ylabel(r'$p_y / s_y$', fontsize=14)
        axes[i].set_zlabel(r'$s_y / s_x$', fontsize=14)
        cbar = f.colorbar(scatters[i], ax=axes[i])
        cbar.set_label(metrics[i], fontsize=14)
    plt.tight_layout()
    plt.show()
'''


# =====================================================================================
# =====================================================================================
