import numpy as np
import samplers
import matplotlib.pyplot as plt
import astropy.units as u
import util
import pdb
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay, ConvexHull
from shapely import geometry



# ============================================================================
# ============================================================================



class shelter:
    def __init__(self, nstakes, pole_height = 45, symmetry='bilateral'):
        '''
        Constructs a shelter object, which contain information about the shelter
        footprint, poles, and facilitaties visualization. Can be passed to quality 
        matric calculation functions  once pitched.

        Parameters
        ----------
        nstakes : int
            The number of stakes to use
        pole_height : float
            The height of the two poles, in inches
        symmetry : string
            The kind of symmtry to enforce, either bilateral or biradial
        '''
        self.nstakes = nstakes
        self.pole_height = pole_height
        self.footprint = None
        self.stakes = None
        self.poles = None
        self.walls = None

        self.symmetry = symmetry
        self.symidx = ['bilateral', 'biradial'].index(symmetry)

    # -----------------------------------------------------

    def sample_footprint(self):
        '''
        Samples a random footprint given the number of stakes, imposing the objects 
        assigned symmetry property and stores it as a class attribute in the form of 
        a Shapely Polygon object.
        '''

        fsamplers = [samplers.sample_bilateral_footprint, samplers.sample_biradial_footprint]
        footprint_sampler = fsamplers[self.symidx]
        self.footprint = footprint_sampler(self.nstakes)

    # -----------------------------------------------------
    
    def sample_poles(self):
        '''
        Samples random positions for the two poles, imposing the obects assigned symmetry 
        property, and stores them as a class attribute in the form of a list of 
        Shapely Point objects.
        '''
        
        assert self.footprint is not None, 'must first call sample_footprint() or choose_footprint()'
        
        psamplers = [samplers.sample_bilateral_poles, samplers.sample_biradial_poles]
        pole_sampler = psamplers[self.symidx]
        self.poles = pole_sampler(self.footprint)

    # -----------------------------------------------------
 
    def choose_footprint(self, stake_points, normalize=True):
        '''
        Generates a footprint polygon based on input stake points. The number of points passed 
        should reflect the assigned symmetry of the object, and the total number of stakes. The
        result is stored as a class attribute in the form of a Shapely Polygon object.

        Parameters
        ----------
        stake_points : array of shape (N,2)
            The x,y position pairs of each stake, where N = nstakes/s, where s is a factor implied
            by the symmetry of the shelter; s=2 for bilateral, and s=4 for biradial.
        normalize : bool
            Whether or not to normalize the resulting footprint, which sets its long side to match 
            that of the XMid, at 100 inches.
        '''

        # check inputs
        required_num_points = [self.nstakes/2, self.nstakes/4][self.symidx]
        assert(len(stake_points) == required_num_points), \
               'must specify {} stake points for {} symmetry with nstakes={}'.format(required_num_points, 
                                                                                     self.symmetry,
                                                                                     self.nstakes)
        # extract components
        stakes_x = stake_points[:,0]
        stakes_y = stake_points[:,1]

        if(self.symmetry == 'bilateral'):
            # reflect about y=0
            stakes_x = np.hstack([stakes_x, stakes_x]) 
            stakes_y = np.hstack([stakes_y, -stakes_y])
            stakes = np.vstack([stakes_x, stakes_y])    
        
        elif(self.symmetry == 'biradial'):
            # reflect about x=0 and y=0
            stakes_x = np.hstack(np.array([1, 1, -1, -1]) * stakes_x) 
            stakes_y = np.hstack(np.array([1, -1, 1, -1]) * stakes_y)
            stakes = np.vstack([stakes_x, stakes_y]).T
           
        # sort by angular position
        theta = np.arctan2(stakes[:,1], stakes[:,0])
        theta_sorter = np.argsort(theta)
        stakes = stakes[theta_sorter]
        
        # make footprint
        self.footprint = geometry.Polygon(stakes)
        if(normalize): self.footprint = util.normalize_footprint(geometry.Polygon(stakes))
        
        # check convex
        footprint_accepted = util.is_convex(self.footprint)
        assert footprint_accepted, 'footprint is not convex!'

    # -----------------------------------------------------
    
    def choose_poles(self, pole_point):
        '''
        Generates a pair of poles based on an input position, with the other being inferred from
        the assigned symmetry of the shelter. The result is stored as a class attribute in the form
        of a list of Shapely Point objects.

        Parameters
        ----------
        pole_point : vector of length 2
            the x,y position of the pole. Must lie within the footprint.
        '''
        
        assert self.footprint is not None, 'must first call sample_footprint() or choose_footprint()'
        assert(np.shape(pole_point) == (2,)), 'pole point must by a 2x1 vector'
        
        if(self.symmetry == 'bilateral'):
            # reflect about y=0
            pole_pos = np.vstack([pole_point, pole_point * [1, -1]])
        
        elif(self.symmetry == 'biradial'):
            # reflect about x=0 and y=0
            pole_pos = np.vstack([pole_point, pole_point * -1])

        poles = [geometry.Point(pole_pos[i]) for i in range(2)]

        # check that both poles lie inside footprint
        assert self.footprint.contains(poles[0]) and self.footprint.contains(poles[1]),\
                                                'poles not contained within footprint' 
        self.poles = poles 

    # -----------------------------------------------------
    
    def pitch(self):
        '''
        Compute the convex hull of the pointset consisting of all stakes and poles. The 
        result is stored in a class attribute self.walls as the return type of scipy.spatial.ConvexHull
        '''
        
        assert self.poles is not None, 'must first generate footprint and pole placement'
        
        stakes = np.array(self.footprint.exterior.coords.xy).T[:-1]
        poles = np.vstack([np.array(p.xy).T[0] for p in self.poles])
        heights = np.hstack([np.zeros(len(stakes)), np.ones(2) * self.pole_height])
        self.wall_points = np.hstack([np.vstack([stakes, poles]), heights[:, np.newaxis]])

        # do Delaunay tesselation
        #self.walls = Delaunay(self.wall_points)
        self.walls = ConvexHull(self.wall_points)
    
    # -----------------------------------------------------    

    def get_free_params(self, normed = False):
        '''
        Return all of the "free parameters" of this shelter, which is the position of stakes and 
        poles implied by the shelter's symmetry. If bilateral, then this is everything in the 
        upper half-plane. If biradial, then everything in the first quadrant.

        Parameters
        ----------
        normed : bool
            Whether or not to normalize the free parameters with respect to the footprint size

        Returns
        -------
        list of length 2, the pole positions as a (N,2) array, and pole positions as a (2,) array
        '''
        
        stakes = np.array(self.footprint.exterior.coords.xy).T[:-1]
        poles = np.vstack([np.array(p.xy).T[0] for p in self.poles])
        
        if(self.symmetry == 'bilateral'):
            pass
        if(self.symmetry == 'biradial'):
            stake_mask = np.logical_and(stakes[:,0] > 0, stakes[:,1] > 0)
            pole_mask = np.logical_and(poles[:,0] > 0, poles[:,1] > 0)
        if(normed):
            extent = np.max(stakes, axis=0)
            return stakes[stake_mask] / extent[0], poles[pole_mask][0] / extent
        else:
           return stakes[stake_mask], poles[pole_mask][0]
        

     
    # -----------------------------------------------------    
    
    def vis_layout(self, ion=False):

        assert self.walls is not None, 'must first call pitch()'
      
        if(ion): plt.ion()
        f = plt.figure(figsize=(5, 8))
        ax = f.add_subplot(211, projection='3d')
        ax.plot_trisurf(self.wall_points[:,0], self.wall_points[:,1], self.wall_points[:,2],
                        triangles = self.walls.simplices, 
                        antialiased=True, color=np.array([184, 192, 223])/255)
        ax.auto_scale_xyz([-50, 50], [-50, 50], [0, 100])
       
        stakes = np.array(self.footprint.exterior.coords.xy)
        ax2 = f.add_subplot(212)
        ax2.plot(stakes[0], stakes[1], '-xk')
        ax2.plot(self.wall_points[-2:,0], self.wall_points[-2:,1], 'ok')
        ax2.set_xlim([-70, 70])
        ax2.set_ylim([-70, 70])    
        if(not ion): plt.show()
     

# ================================ QUALITY METRICS ============================================


def compute_weather_performance(shelter):
    '''
    Compute the weather performance coefficient, with consideration to both percipitation
    and wind influences

    Parameters
    ----------
    shelter : a shelter object
        a shelter object, as constructed in the above class, which has been pitched

    Returns
    -------
    weather_performance : float
        the weather performance as a float
    '''
    
    assert(shelter.walls is not None), 'shelter must first be pitched'
    
    response_func = lambda theta:  0.5 * np.sin(2*theta) 
    panels = shelter.wall_points[shelter.walls.simplices]
    
    # remove floor panels
    floor_mask = np.sum(panels[:,:,-1], axis=1) != 0
    panels = panels[floor_mask] 

    wall_angles = np.zeros(len(panels))
    for i in range(len(panels)):    
        # compute angle of panel plane relative to the ground
        ab = panels[i][0] - panels[i][1]
        ac = panels[i][0] - panels[i][2]
        cross = np.cross(ab, ac)
        z = cross[-1]
        r = np.linalg.norm(cross)
        wall_angles[i] = np.pi/2 - np.arcsin(np.abs(z)/r)
    
    performance = response_func(wall_angles)
    return np.mean(performance)


# --------------------------------------------------------------------------------------------


def compute_volumetric_efficiency(shelter):
    '''
    Computes the ratio of volume to surface area

    Parameters
    ----------
    shelter : a shelter object
        a shelter object, as constructed in the above class, which has been pitched

    Returns
    -------
    volumetric_efficiency : float
        the volumetric efficiency
    '''

    assert(shelter.walls is not None), 'shelter must first be pitched'

    panels = shelter.wall_points[shelter.walls.simplices]
    
    # remove floor panels
    floor_mask = np.sum(panels[:,:,-1], axis=1) != 0
    panels = panels[floor_mask]
    
    shelter.volume = 0
    shelter.surface_area = 0 
    for i in range(len(panels)):    
        # compute surface area of panel
        ab = panels[i][0] - panels[i][1]
        ac = panels[i][0] - panels[i][2]
        shelter.surface_area += np.linalg.norm( np.cross(ab, ac)  ) / 2

        # and volume under panel
        ab_base = np.hstack([ab[:-1], 0])
        ac_base = np.hstack([ac[:-1], 0])
        base_area = np.linalg.norm( np.cross(ab_base, ac_base)  ) / 2
        shelter.volume += np.sum((panels[i][:,-1])) * (base_area/3)

    # scale to cubic feet
    u.imperial.enable()
    ss = (shelter.surface_area * u.imperial.inch**2).to(u.imperial.ft**2)
    vv = (shelter.volume * u.imperial.inch**3).to(u.imperial.ft**3)

    volumetric_efficiency = (vv / ss).value
    return volumetric_efficiency



# ================================ DEFINED MODELS ============================================



class defined_model:
    def __init__(self, stakes, pole):
        '''
        Class for storing summary properties of market models when the entire shelter 
        class is not needed (useful for storing positions of models in parameter space).
        Intended for use only with biradial models.

        Parameters
        ----------
        stakes : (N,2) array
            The stake positions of the model, assumed to be in the first quandrant, 
            with the origin at the tent center.
        poles: (2,) array
            Thd position of one pole of the model, assumed to be in the first
            quadrant, with the origin at the tent center.
        '''
        
        self.x = stakes[:,0]
        self.y = stakes[:,1] 
        self.widthx = np.max(self.x) * 2
        self.widthy = np.max(self.y) * 2
        width = max([self.widthx, self.widthy])
        self.norm_x = self.x / (width / 2)
        self.norm_y = self.y / (width / 2)
        
        self.px = pole[0]
        self.py = pole[1]
        self.norm_px = self.px / (self.widthx / 2)
        self.norm_py = self.py / (self.widthy / 2)

xmid2p = defined_model(np.array([[100/2, (100/103 * 88)/2]]), [25, 18])
xmid1p = defined_model(np.array([[100/2, 67/2]]), [25, 8.5])

        
# ============================================================================
# ============================================================================
