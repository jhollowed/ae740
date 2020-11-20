import pdb
import sys
import seaborn
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import kernels
import gaussianProcess as GP
from mpl_toolkits import mplot3d

sys.path.append('{}/../project1'.format(pathlib.Path(__file__).parent.absolute()))
import monteCarlo as mc


# =============================================================================================
# =============================================================================================


class smellScape:
    def __init__(self, init_data):

        # read in the initial data
        print('reading initial data from {}'.format(init_data))
        idata = np.genfromtxt(init_data, delimiter=' ', names=True)
        self.x = idata['x']
        self.y = idata['y']
        self.sensor = idata['sensor']
        self.noise = np.zeros(len(self.sensor))

        # define field of play
        self.xlim = [-1, 1]
        self.ylim = [-1, 1]
        
        # define Gp grid, posterior, covaraince
        self.prediction_grid = None
        self.gp_posterior = None
        self.gp_cov = None


    def build_gp_grid(self, nside):
       
        print('building GP grid with {} points'.format(nside**2))
        xgrid = np.linspace(self.xlim[0], self.xlim[1], nside)
        ygrid = np.linspace(self.ylim[0], self.ylim[1], nside)
        xygrid = np.meshgrid(xgrid, ygrid)
        self.prediction_grid = np.array([np.ravel(xygrid[0]), np.ravel(xygrid[1])]).T

        self.gp_posterior = np.zeros(len(self.prediction_grid[0]))
        self.gp_cov = np.zeros(len(self.prediction_grid[0]))
        
        self.flat_prior = lambda x: np.zeros((x.shape[0])) 
       

    def generate_data_request(self, nx, ny, xlim, ylim, suffix=None):
        
        x_request = np.linspace(xlim[0], xlim[1], nx)
        y_request = np.linspace(ylim[0], ylim[1], ny)
        xy_request = np.meshgrid(x_request, y_request)
        pairs = np.array([np.ravel(xy_request[0]), np.ravel(xy_request[1])]).T
        np.savetxt('request{}.dat'.format(suffix), pairs, fmt='%.4f')
   

    def add_data(self, data):
        
        # read in the new data
        print('reading additional data from {}'.format(data))
        data = np.genfromtxt(data, delimiter=' ', names=True)
        self.x = np.hstack([self.x, data['x']])
        self.y = np.hstack([self.y, data['y']])
        self.sensor = np.hstack([self.sensor, data['sensor']])
        self.noise = np.zeros(len(self.sensor))
        

    def run_gp_regression(self, tau, l, vis_result=False, plot_sfx=''):     

        if(self.prediction_grid is None):
            raise RuntimeError('build_gp_grid() must be called before GP regressions can be run')
        
        self.kernel = lambda x,xp : kernels.sqExpNd(x, xp, tau=tau, l=l)

        print('running GP regression with hyperparameters tau={}, l={}'.format(tau, l))
        input_points = np.array([self.x, self.y]).T
        gp_result = GP.run_gp_regression(self.flat_prior, self.kernel, input_points, self.sensor, 
                                         self.prediction_grid, self.noise, plot=vis_result, plot_sfx=plot_sfx)
        self.gp_posterior = gp_result[0]
        self.gp_cov = gp_result[1]


    def mc_confidence(self, A, nsamples, use_flat_prior=False):
        
        if(use_flat_prior):
            # for debugging only
            prior = self.flat_prior(self.prediction_grid)
            outsfx = ' on flat prior'
        else:
            # use regression posterior as prior input to GP function sampler
            prior = self.gp_posterior
            outsfx = ''

        # samples N GP functions defined on this grid
        sampler = lambda N: \
                  GP.sample_functions(self.prediction_grid, N, prior, self.kernel)
        # evaluator finds the argmax of each sampled function, assigns 0 or 1 based upon whether or 
        # not it is within the set A
        evaluator = lambda fs:\
                    np.logical_and(
                        np.logical_and( self.prediction_grid[np.argmax(fs, axis=1)].T[0] >= A[0][0], 
                                        self.prediction_grid[np.argmax(fs, axis=1)].T[0] <= A[0][1] ), 
                        np.logical_and( self.prediction_grid[np.argmax(fs, axis=1)].T[1] >= A[1][0], 
                                        self.prediction_grid[np.argmax(fs, axis=1)].T[1] <= A[1][1] )   
                                 ) 
        # run monte carlo
        estimate, _, _ = mc.monte_carlo(nsamples, sampler, evaluator)
        print('Monte Carlo estimate of confidence interval of {:.2f}% for region {}{}'.format(
               estimate*100, A, outsfx))
   

# =====================================================================================================

        
def plot_sampled_functions(nrows=2, nside=None, grid=None, mean_func=None, kernel=None, sfx=None):
    '''
    Visualize sampled GP functions.

    Parameters
    ----------
    nrows : int
        number of rows of 4 sample functions to plot (3*nrows total functions)
    nside : int
        number of sample points in x and y space per-function (nside^2 total evaluations per sampled function) 
    grid : (N,N) array
        grid attribute of smellScape instance object; not needed if nside is passed, and vice verse
    '''
    
    if(mean_func is None):
        mean_func = lambda x: np.zeros((x.shape[0])) 
    if(kernel is None):
        kernel = lambda x,xp : kernels.sqExpNd(x, xp, tau=[1, 0.7], l=[0.5, 0.8])
    if(grid is None and nside is not None):
        x = np.linspace(-1, 1, nside)
        y = np.linspace(-1, 1, nside)
        xy = np.meshgrid(x, y)
        grid = np.vstack([np.ravel(xy[0]), np.ravel(xy[1])]).T
    elif(grid is not None and nside is None):
        nside = int(np.sqrt(len(grid)))
        xy = np.array([grid.T[0].reshape(nside, nside), grid.T[1].reshape(nside, nside)])
    else:
        raise RuntimeError('Either grid or nside must be passed, not both')
   
    print('visualizing sampled functions')
    ss = GP.sample_functions(grid, nrows*4, mean_func, kernel)

    f = plt.figure(figsize=(10, 10*nrows/4))
    for i in range(nrows*4):
        ax = f.add_subplot(nrows,4,i+1, projection='3d')
        ax.plot_surface(xy[0], xy[1], ss[i].reshape(nside, nside), cmap='viridis', edgecolor='none')
        ax.dist = 10.6
    plt.tight_layout()
    plt.savefig('sampled_functions{}.png'.format(sfx), dpi=300)


# =====================================================================================================
# =====================================================================================================


# driver
if __name__ == '__main__':

    # create smellScapre object, define data and prediction grid, run gp regression
    s = smellScape('initialReadings.csv')
    s.build_gp_grid(50)
    #s.add_data('sensor_request1.dat')
    s.run_gp_regression(tau=[0.5, 0.8], l=[0.4, 0.4], vis_result=True, plot_sfx='_initial')
    
    # inspect sample functions for flat prior and gpr posterior
    plot_sampled_functions(2, grid=s.prediction_grid, mean_func=s.gp_posterior, sfx='_gpr')
    plot_sampled_functions(2, grid=s.prediction_grid, mean_func=s.flat_prior(s.prediction_grid), sfx='_flatPrior')
    
    # evaluate MC estaimte of confidence interval
    s.mc_confidence([[-1, 1], [-1, 1]], 10000)
    s.mc_confidence([[-1, 1], [-1, 0]], 10000)
    s.mc_confidence([[0, 1], [-1, 0]], 10000)
    s.mc_confidence([[-1, 0], [-1, 0]], 10000)
    
    s.mc_confidence([[0, 0.5], [-0.5, 0]], 10000)
    s.mc_confidence([[0.5, 1], [-0.5, 0]], 10000)
    s.mc_confidence([[0, 0.5], [-1, -0.5]], 10000)
    s.mc_confidence([[0.5, 1], [-1, -0.5]], 10000)
    
    # evaluate MC estaimte of confidence interval on flat prior (sanity check)
    s.mc_confidence([[-1, 1], [-1, 1]], 1000, use_flat_prior=True)
    s.mc_confidence([[-1, 1], [-1, 0]], 1000, use_flat_prior=True)
    s.mc_confidence([[0, 1], [-1, 0]], 1000, use_flat_prior=True)
    s.mc_confidence([[-1, 0], [-1, 0]], 1000, use_flat_prior=True) 
