import pdb
import numpy as np
import matplotlib.pyplot as plt
import randomVariables as rv
import monteCarlo as mc
import sampleSchemes as ss
from scipy.stats import norm

from matplotlib import gridspec
import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')


# ========================================================================================

class geometric_brownian_motion:
    def __init__(self, Y0, mu, sigma, T=1):
        '''
        Performs a geometric Brownian motion

        Parameters
        ----------
        Y0 : float
            the initial condition
        mu : float
            drift scale parameter
        sigma : float
            diffusion scale parameter
        T : float, optional
            total time; defaults to 1
        '''
        self.Y0 = Y0
        self.mu = mu
        self.sigma = sigma
        self.T = T


    # ----------------------------------------------------------------------------------------


    def walk_old(self, dt):
        '''
        Performs a geometric Brownian motion

        Parameters
        ----------
        dt : int
            size of the timestep
        
        Returns
        -------
        bmotion : 2 float arrays
            the Brownian record, and assocaited Weiner process W
        '''
        
        # number of samples from T, dt
        N = int(np.ceil(self.T / dt)) + 1
        
        # define Gaussian RV and sample via inverse transform sampling
        W = rv.Gaussian(mu=0, sigma=dt)
        sampler = ss.InverseTransformSampler(W).sample
        dW = sampler(N)

        # define drift and diffusion terms
        b = lambda t, Y: self.mu * Y
        h = lambda t, Y: self.sigma * Y

        # build Yn
        Y = np.zeros(N)
        Y[0] = self.Y0
        for n in range(N-1):
            Y[n+1] = Y[n] + b(dt*n, Y[n])*dt + h(dt*n, Y[n])*dW[n]
        
        return Y, dW
    
    
    # ----------------------------------------------------------------------------------------
    
    
    def walk(self, dt, nsamples=1):
        '''
        Performs a geometric Brownian motion

        Parameters
        ----------
        dt : int
            size of the timestep
        
        Returns
        -------
        bmotion : 2 float arrays
            the Brownian record, and assocaited Weiner process W
        '''
        
        # number of samples from T, dt
        N = int(np.ceil(self.T / dt)) + 1
        
        # define Gaussian RV and sample via inverse transform sampling
        W = rv.Gaussian(mu=0, sigma=dt)
        sampler = ss.InverseTransformSampler(W).sample
        dW = sampler((nsamples, N))

        # define drift and diffusion terms
        b = lambda t, Y: self.mu * Y
        h = lambda t, Y: self.sigma * Y

        # build Yn
        Y = np.zeros((nsamples, N))
        Y[:, 0] = self.Y0
        for n in range(N-1):
            Y[:,n+1] = Y[:,n] + b(dt*n, Y[:,n])*dt + h(dt*n, Y[:,n])*dW[:,n]

        return Y, dW


    # ----------------------------------------------------------------------------------------


    def fine_to_coarse(self, dt_coarse, M, brownian_fine=None, nsamples=None):
        '''
        Extract a sample of a coarse brownian motion simulation from a finer simulation

        Parameters
        ----------
        dt : int
            size of the coarse timestep
        brownian_fine : float array
            the fine-scale Brownian record
        M : float
            coarsening factor
        nsamples : int
            number of samples; only needs to be passed if brownian_fine is None, 
            otherwise will match the input fine sample set
        '''

        if(brownian_fine is None):
            # get fine motion
            brownian_fine, _ = self.walk(dt_coarse / M, nsamples)
            return_fine = True
        else:
            nsamples = len(brownian_fine)
            return_fine = False

        #subsample fine motion
        N = int(np.ceil(self.T/dt_coarse))+1
        brownian_coarse = np.zeros((nsamples, N))
        brownian_coarse[:, 0] = brownian_fine[:, 0]
      
        for i in range(1, N):
            delta = brownian_fine[:, int(i * M)] - brownian_fine[:, int((i-1) * M)]
            brownian_coarse[:, int(i)] = brownian_coarse[:, int(i-1)] + delta
        
        if(return_fine):
            return [brownian_coarse, brownian_fine]
        else:
            return brownian_coarse
    
    
    # ----------------------------------------------------------------------------------------

    
    def time_vector(self, dt):
        '''
        Builds a time vector for a given timestep dt

        Parameters
        ----------
        dt : int
            size of the timestep
        ''' 
        return np.arange(0, self.T + dt, dt)
        


# ========================================================================================


class problem4:
    def __init__(self, dt):
        self.dt = 1e-3
        self.Y0 = 1
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1
        self.t = np.arange(0,self.T,self.dt)
        return
    

    def part1(self):
        
        # build geometric brownian motion
        dt, Y0, mu, sigma, t, T = self.dt, self.Y0, self.mu, self.sigma, self.t, self.T
        gbm = geometric_brownian_motion(Y0, mu, sigma, T)

        # visualize a few walks
        print('taking a few brownian motion samples...')
        f = plt.figure()
        ax = f.add_subplot(111)
        for i in range(15):
            Y, dW = gbm.walk(dt)
            ax.plot(t, Y, '-k', alpha=0.33)
        ax.set_xlabel(r'$t$', fontsize=14)
        ax.set_ylabel(r'$Y_t$', fontsize=14)
        plt.savefig('geom_walks.png', dpi=300)

        # verify mean and variance via Monte Carlo. walk[0] is Y(t).
        nsamples = int(1e5)
        print('Doing MC estimate of mean of Y(1) with {} samples'.format(nsamples))
        
        sampler = lambda N: np.array([gbm.walk(dt) for i in range(N)])
        evaluator = lambda walks: np.array([w[0] for w in walks]).T[-1]
        
        estimate, samples, evaluations = mc.monte_carlo(nsamples, sampler, evaluator)
        Y_samples = np.array([s[0] for s in samples])
        dW_samples = np.array([s[1] for s in samples])
        
        # get MC 95% confidence
        z95 = 1.96
        var = np.var(evaluations)
        std = np.sqrt(var/len(evaluations))
        uncert = z95 * np.sqrt(var/len(evaluations))
        print('MC estimate: E(Y(1)) = {} +- {}'.format(estimate, uncert))

        # get analytic expectation distribution Y(1). Wt1 is W(t=1); Yt1_analytic is Y(t=1)
        Wt1 = np.sum(dW_samples, axis=1)
        Yt1_analytic = Y0 * np.exp((mu - sigma**2/2) + sigma*Wt1)
        analytic_mean = np.mean(Yt1_analytic)
        analytic_var = np.var(Yt1_analytic)
        analytic_err = z95 * np.sqrt(analytic_var/len(Wt1))
        print('analytic result: {} +- {}'.format(analytic_mean, analytic_err))
      
        # visualize mc estimate
        print('plotting MC result')
        gs = gridspec.GridSpec(1,3)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(gs[0, 0:2])
        ax2 = f.add_subplot(gs[0, 2])
        
        # --- plot paths
        nsamples_to_plot = 100
        ax.plot(t, Y_samples[0:nsamples_to_plot, :].T, '-k', alpha=0.25)
        ax.set_xlim(0, max(t))

        # --- plot Y(1) histogram
        ax2.hist(evaluations, bins=100, orientation='horizontal', lw=0, 
                 color ='k', alpha=0.5, density=True, label=r'$\mathrm{evaluations}$')
        xlim = ax2.get_xlim()
        ax2.plot(xlim, [estimate, estimate], '--r', label=r'$\mathrm{estimate}$')

        # --- plot mean estaimte w/ confidence interval
        ax2.fill_between(xlim, np.array([estimate, estimate]) - uncert, 
                               np.array([estimate, estimate]) + uncert,
                               color='r', alpha=0.5, label=r'$95\%\mathrm{\>\>confidence}$')
     
        # --- format
        ax2.set_ylim(ax.get_ylim())
        ax2.legend(fontsize=14)

        ax.set_xlabel(r'$t$', fontsize=16)
        ax.set_ylabel(r'$Y_t$', fontsize=16)
        ax2.set_xlabel(r'$\mathrm{pdf}$', fontsize=16)
        ax2.set_ylabel(r'$Y(1)$', fontsize=16)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax.set_title(r'${}\mathrm{{\>samples\>with\>}}dt={}$'.format(nsamples, dt), fontsize=14)
        plt.tight_layout()
        plt.savefig('geom_walks_mc.png', dpi=300)
         
        
    def part2(self):
        
        # --- define coarse Y with timestep inflated by factor of 4
        dt = 0.05
        dt_coarse = 4*dt
        gbm = geometric_brownian_motion(self.Y0, self.mu, self.sigma, self.T)
        [Y_coarse, Y_fine] = gbm.fine_to_coarse(dt_coarse, M=4)
        [t_coarse, t_fine] = [gbm.time_vector(dt_coarse), gbm.time_vector(dt)]
       
        # --- plot results
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(t_fine, Y_fine, '-ok', label=r'$\Delta t$')
        ax.plot(t_coarse, Y_coarse, '--xr', label=r'$4\Delta t$')
        ax.legend()
        ax.set_xlabel(r'$t$', fontsize=14)
        ax.set_ylabel(r'$Y_t$', fontsize=14)
        plt.savefig('figs/shared_realz.png', dpi=300) 
        return
    

    def part3(self):
       
        dt = np.array([4**(-2), 4**(-3), 4**(-4), 4**(-5)])
        nsamples = int(1e3)
        
        gbm = geometric_brownian_motion(self.Y0, self.mu, self.sigma, self.T)
        means, var, var_discrep = mc.multilevel_montecarlo_sde(gbm, dt, nsamples)
        print('MLMC estimate of E[Y(1)] = {}'.format(means))
        
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot([2, 3, 4, 5], var, '-ok')
        ax.set_xlabel(r'$-\log_4(\Delta t)$', fontsize=14)
        ax.set_ylabel(r'$\mathrm{Var}[Y(1)]$', fontsize=14)
        plt.savefig('figs/mlmc_var.png', dpi=300)


    def part4(self):
        return


# ========================================================================================


if __name__ == '__main__':
    p4 = problem4(dt = 0.001)
    p4.part3()

