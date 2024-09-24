import pdb
import lhsmdu
import shelter
import kernels
import samplers
import numpy as np
import gaussianProcess as GP
import matplotlib.pyplot as plt
from shelter import xmid2p, xmid1p
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')
    


# =============================================================================================
# =============================================================================================



class shelterEmulator:
    def __init__(self, nstakes, normalized=True):
        '''
        Constructs an M-dimensional emulator of trekking pole shelter quality metrics, 
        using gaussian process regression to estimate the mean and covaraince of prediciton
        sites, and monte carlo methods to obtain confidence intervals. This class is built
        particularly for shelters which exhibit reflection symmetry in 2-dimensions, 
        elsewhere in this repository 'called biradial' shelters.

        Parameters
        ----------
        nstakes : int
            Number of stakes for the shelter models that will be emulated
        normalized : bool
            whether or not to expect all shelters to be normalized in the x-dimension
            by util.normalize_footprint
        '''

        assert(nstakes % 4 == 0), 'number of stakes must be divisible by 4 for biradial models'
        # compute number of dimensions for free-parameter space
        # +2 DOF per each stakes, and one pole (remainder set by symmetry)
        # -1 for the normalization condition on the footprint width
        self.M = int(2 * (nstakes / 4 + 1) - 1)
        
        self.normalized = normalized
        self.nstakes = nstakes
        
        # define data positions and quality metrics
        # one data point on parameter space will take the form [sy, sx, sy, sx, sy,.... px, py]
        # where sy,x are cartesian positions of stakes, and px,y are that of the pole data shall 
        # be sorted such that the normalized position sx' is removed after obtaining from 
        # shelter.get_free_params 
        self.x = None
        self.quality = None
        self.volumetric_efficiency = None
        self.weather_performance = None
        
        # define GP grid, posterior, covariance
        self.prediction_grid = None
        self.gp_posterior = None
        self.gp_cov = None

    # ----------------------------------------------------------------------------------------

    def sample_space(self, N, method='random'):
        '''
        Samples parameter space; each sample is a shelter with a particular geometry, and each
        sample output is the shelter quality metric.

        Parameters
        ----------
        N : int
            The number of samples to draw
        method : string
            The sampling method to use
        '''

        if(method == 'random'):
            # randomly sample shelter
            all_x, q = samplers.random_sampling(self.nstakes, N, normed=True)
        elif(method == 'hypercube'):
            # sample shelters on latin hypercube
            all_x, q = samplers.random_sampling(self.nstakes, N, normed=True) 
        self.noise = np.zeros(N)
        self.volumetric_efficiency, self.weather_performance, self.quality = q[0], q[1], q[2]
        
        # transform into format described in constructor
        self.x = np.zeros((N, self.M))
        for i in range(N):
            this_x = all_x[i]
            
            # find the stake with largest x-value for each sample (this one is normalized,
            # and thus needs to be removed from the free parameters)
            norm_stake = np.argmax(all_x[i,:-1,0])
            
            # move stake with largest x-value to the front of the array
            this_x[[0, norm_stake]] = this_x[[norm_stake, 0]]
            
            # clip normalized stake and add this point in parameter space to class attribute
            self.x[i] = np.ravel(all_x[i])[1:]
            
    # ----------------------------------------------------------------------------------------
        
    def build_gp_grid(self, n, dim_limits=None, normed=True):
        '''
        Builds the M-dimensional prediciton grid over which the emulator will return results

        Parameters
        ----------
        n : int
            number of grid points in each dimension; grid will be composed of (ndim^M) total points
        dim_limits : (M,2) array, optional
            grid limits on each dimension
        '''
        
        if(normed): scale = 1
        else: scale = 50

        if(dim_limits is None and self.normalized == True):
            dim_limits = np.vstack([np.zeros(self.M), np.ones(self.M)]).T * scale

        numpoints = int(n ** self.M)
        arrays = np.zeros((self.M, n))
        
        print('building GP grid with {} points'.format(numpoints))
        for i in range(self.M):
            arrays[i] = np.linspace(int(dim_limits[i][0]), int(dim_limits[i][1]), n)
       
        grid_dims = np.meshgrid(*arrays)
        self.prediction_grid = np.array([dim.flatten() for dim in grid_dims]).T
    
    # ----------------------------------------------------------------------------------------

    def view_grid(self):
        
        f = plt.figure(figsize=(10,8))
        ax4 = f.add_subplot(221, projection='3d')
        ax5 = f.add_subplot(222, projection='3d')
        ax6 = f.add_subplot(223, projection='3d')
               
        sctr4 = ax4.scatter(*self.x[:,::-1].T, c=self.volumetric_efficiency, 
                                  marker='o', cmap=plt.cm.viridis, alpha=0.5)
        sctr5 = ax5.scatter(*self.x[:,::-1].T, c=self.weather_performance, 
                                  marker='o', cmap=plt.cm.viridis, alpha=0.5)
        sctr6 = ax6.scatter(*self.x[:,::-1].T, c=self.quality, 
                                  marker='o', cmap=plt.cm.viridis, alpha=0.5)

        scatters = [sctr4, sctr5, sctr6]
        metrics = [r'$\epsilon_V$', r'$P_W$', r'$\epsilon_VP_W$']
        axes = [ax4, ax5, ax6]
        xmid_s = 60
        xmid_lw = 2
        
        for i in range(len(axes)):
            axes[i].scatter([xmid1p.norm_px], [xmid1p.norm_py], [xmid1p.norm_y], 
                            color='r', marker='x', s=xmid_s, lw=xmid_lw)
            axes[i].scatter([xmid2p.norm_px], [xmid2p.norm_py], [xmid2p.norm_y], 
                            color='r', marker='^', s=xmid_s, lw=xmid_lw)
            axes[i].set_xlabel(r'$p_x / s_x$', fontsize=14)
            axes[i].set_ylabel(r'$p_y / s_y$', fontsize=14)
            axes[i].set_zlabel(r'$s_y / s_x$', fontsize=14)
            cbar = f.colorbar(scatters[i], ax=axes[i])
            cbar.set_label(metrics[i], fontsize=14)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------------------------

    def run_gp_regression(self, tau=None, l=None, vis_result=False, plot_sfx=''):
        '''
        Runs a Gaussian Process regression in the free parameter space with data sampled from 
        self.sample_space()

        Parameters
        ----------
        tau : (M,) float array
            variance parameter for the Gaussian kernel
        l : (M,) float array
            scale parameter for the Gaussian kernel
        ''' 
       
        assert(self.x is not None), 'Must fist sample parameter space'
        
        # define kernel and prior functions
        # use these values for the kernal params if not passed; informed by runs with 
        # 4 stakes, biradial symmetry, and normalized parameters
        if(tau is None):
            tau = np.ones(self.M) * 1
        if(l is None):
            l = np.ones(self.M) * .1
        self.kernel = lambda x,xp: kernels.sqExpNd(x, xp, tau=tau, l=l)
        self.flat_prior = lambda x: np.zeros((x.shape[0]))
       
        print('running GP regression with hyperparameters tau={}, l={}'.format(tau, l))
        gp_result = GP.run_gp_regression(self.flat_prior, self.kernel, self.x[:,::-1], self.quality,
                                         self.prediction_grid, self.noise, plot=vis_result, plot_sfx=plot_sfx)
        self.gp_posterior = gp_result[0]
        self.gp_cov = gp_result[1]
        pdb.set_trace()
    
    # ----------------------------------------------------------------------------------------
    
    def mc_confidence():
        return



# ============================================================================================
# ============================================================================================



if __name__ == '__main__':
    ff = shelterEmulator(4)
    ff.build_gp_grid(10)
    ff.sample_space(1000, method='hypercube')
    ff.view_grid()
    ff.run_gp_regression()
    pdb.set_trace()
      



    
def random_sampling(nstakes, N, symmetry = 'biradial'):

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
   
if __name__ == '__main__':
    random_sampling(4, 1000)
