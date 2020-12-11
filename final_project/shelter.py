import numpy as np
import samplers
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay, ConvexHull
import astropy.units as u
from shapely import geometry
import util

# ============================================================================
# ============================================================================



class shelter:
    def __init__(self, nstakes, pole_height = 45, symmetry='bilateral'):
        '''
        '''
        self.nstakes = nstakes
        self.pole_height = pole_height
        self.footprint = None
        self.poles = None
        self.walls = None

        self.symmetry = symmetry
        self.symidx = ['bilateral', 'biradial'].index(symmetry)

    # -----------------------------------------------------

    def sample_footprint(self):

        fsamplers = [samplers.sample_bilateral_footprint, samplers.sample_biradial_footprint]
        footprint_sampler = fsamplers[self.symidx]
        self.footprint = footprint_sampler(self.nstakes)

    # -----------------------------------------------------
    
    def sample_poles(self):
        
        assert self.footprint is not None, 'must first call sample_footprint() or choose_footprint()'
        
        psamplers = [samplers.sample_bilateral_poles, samplers.sample_biradial_poles]
        pole_sampler = psamplers[self.symidx]
        self.poles = pole_sampler(self.footprint)

    # -----------------------------------------------------
 
    def choose_footprint(self, stake_points, normalize=False):

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
        if(normalize): self.footprint = normalize_footprint(geometry.Polygon(stakes.T))
        
        # check convex
        footprint_accepted = util.is_convex(self.footprint)
        assert footprint_accepted, 'footprint is not convex!'

    # -----------------------------------------------------
    
    def choose_poles(self, pole_point):
        
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
        
        assert self.poles is not None, 'must first generate footprint and pole placement'
        
        stakes = np.array(self.footprint.exterior.coords.xy).T[:-1]
        poles = np.vstack([np.array(p.xy).T[0] for p in self.poles])
        heights = np.hstack([np.zeros(len(stakes)), np.ones(2) * self.pole_height])
        self.wall_points = np.hstack([np.vstack([stakes, poles]), heights[:, np.newaxis]])

        # do Delaunay tesselation
        #self.walls = Delaunay(self.wall_points)
        self.walls = ConvexHull(self.wall_points)
    
    # -----------------------------------------------------    
    
    def compute_volumetric_efficiency(self):
        '''
        '''
        panels = self.wall_points[self.walls.simplices]
        self.volume = 0
        self.surface_area = 0
        
        for i in range(len(panels)):    
            # skip if this is a floor panel
            is_floor = np.sum(panels[i][:,-1]) == 0
            if(is_floor): continue

            # compute surface area of panel
            ab = panels[i][0] - panels[i][1]
            ac = panels[i][0] - panels[i][2]
            self.surface_area += np.linalg.norm( np.cross(ab, ac)  ) / 2

            # and volume under panel
            ab_base = np.hstack([ab[:-1], 0])
            ac_base = np.hstack([ac[:-1], 0])
            base_area = np.linalg.norm( np.cross(ab_base, ac_base)  ) / 2
            self.volume += np.sum((panels[i][:,-1])) * (base_area/3)

        # scale to cubic feet
        u.imperial.enable()
        ss = (self.surface_area * u.imperial.inch**2).to(u.imperial.ft**2)
        vv = (self.volume * u.imperial.inch**3).to(u.imperial.ft**3)

        volumetric_efficiency = (vv / ss).value
        return volumetric_efficiency
    
    # -----------------------------------------------------    
    
    def vis_layout(self):

        assert self.walls is not None, 'must first call pitch()'
      
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
        plt.show()
     
    
    # -----------------------------------------------------    

    def compute_collpase_threshold():
        '''
        '''
        return


        

# ============================================================================
# ============================================================================

# use shalpey for all this


# could do...
# generate random polygon
# enforce symmetry by taking upper half-plane and reflecting...
# if not N/2 points in half-plane, reject, sample again (shouldn't take too long)
# likewise for 2-fold symmetry, but for N/4
# via this method:
# https://cglab.ca/~sander/misc/ConvexGeneration/convex.html

# place poles...
# for bilateral: place randomly in upper half-plane
# for biradia: place randomly in 1st quadrant
# via this method:
# choose random unfiorm over bounding box, reject until found

# or, much better:
# inputs...
# pole angle (0-pi/2), distance (within footprint)
# x-symmery coeff (0-1)
# y-symmetry coeff (0-1)
# rectangular coeff (0-1)

# should 2-d positions of _all_ stakes and poles be parameters? Or fewer global parameters?
# kernels would be v complicated then....

# params: 
# θ, r for stakes (N/2 total params for biradial, N for bilateral)
# then just rejection sampling for convex...?
# θ, r for one pole (2 total params), other one follows symmetry properties (either reflects once, or twice)


# 2 ways of sampling parameter space:
# grid over parameter space, reject footprints that aren't convex, or that dont contain poles?
# OR
# generate random convex polygons; place poles randomly. Sample set will no be evenly discretized, but should still homogenously fill parameter space for large N
