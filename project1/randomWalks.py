import pdb
import numpy as np
import matplotlib.pyplot as plt
import sampleSchemes as ss
import randomVariables as rv
import monteCarlo as mc
from scipy.special import comb

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

# ================================================================================
# ================================================================================

def bernoulli_walk_1d(N, T=1):
    '''
    Simulates a single path of one-dimensional Bernoulli walk

    Parameters
    ----------
    N : int
        Number of steps

    Returns
    -------
    path : float array
        The resulting Bernoulli motion path
    '''

    # define uniform sampled RV
    X = np.random.rand(N)
    X[X <= 0.5] = -1
    X[X > 0.5] = 1

    # define the Brownian record
    S = np.cumsum(np.hstack([[0], X])) 
    return S


# --------------------------------------------------------------------------------


def gaussian_motion_1d(N):
    '''
    Simulates a single path of one-dimensional gaussian motion

    Parameters
    ----------
    N : int
        Number of steps

    Returns
    -------
    path : float array
        The resulting Brownian motion path
    '''

    # define Gaussian RV and sample via inverse transform sampling
    X = rv.Gaussian()
    sampler = ss.InverseTransformSampler(X).sample
    Xj = sampler(N)

    # define the record
    S = np.cumsum(np.hstack([[0], Xj])) 
    return S


# --------------------------------------------------------------------------------

 
class problem3_1:
    
    def __init__(self, N):
        
        self.N = N
        self.p = 0.5

        # analytic probability
        self.P = lambda k: np.sum([comb(self.N, k+i) * self.p**self.N for i in range(self.N-k)])
   

    def part_a(self):

        # ------------ part a -------------------
        f = plt.figure()
        ax = f.add_subplot(111)

        print('Simulating 1d Bernoulli walk for N={}'.format(self.N))
        for i in range(15):
            S = bernoulli_walk_1d(self.N)
            ax.plot(np.linspace(0, 1, self.N+1), S, alpha=0.3, color='k')
        ax.set_xlabel(r'$t$', fontsize=14)
        ax.set_ylabel(r'$W_t$', fontsize=14)
        plt.savefig('1dwalk.png', dpi=300)


    def part_b(self):

        # ------------ part b -------------------
        trials = int(1e5)
        steps = np.linspace(0, 1, self.N+1)
            
        # plotting too slow for 1e5...
        if(trials <= 1e4):
            gs = gridspec.GridSpec(1,3)
            f = plt.figure()
            ax = f.add_subplot(gs[0, 0:2])

        print('Simulating {} 1d Bernoulli walks for N={}'.format(trials, self.N))
        S = np.zeros(trials)
        for i in range(trials):
            if(i%1000 == 0): print('trial {}/{}'.format(i, trials))
            walk = bernoulli_walk_1d(self.N)
            S[i] = walk[-1]
            if(trials <= 1e4):
                ax.plot(steps, walk, alpha=0.1, color='k')
        
        if(trials <= 1e4):
            ax2 = f.add_subplot(gs[0, 2])
            ax2.hist(S, bins=100, orientation='horizontal', lw=0, color ='k', alpha=0.5, density=True)
            ax.set_xlabel(r'$t$', fontsize=14)
            ax.set_ylabel(r'$W_t$', fontsize=14)
            ax2.set_xlabel(r'$\mathrm{pdf}$', fontsize=14)
            ax2.set_ylabel(r'$S_{100}$', fontsize=14)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax.set_title(r'$10^4\mathrm{\>trials\>with\>}N=100$', fontsize=14)
            plt.savefig('1dwalk_MC.png', dpi=300)
    
        # compute probability that S > 10 via monte carlo
        sampler = lambda nsamples: [np.diff(bernoulli_walk_1d(100)) for i in range(nsamples)]
        evaluator = lambda X: np.array(np.sum(X, axis=1) > 10, dtype=int)
        estimate, _, evaluations = mc.monte_carlo(trials, sampler, evaluator)
        print('MC estimate: P(s>10) = {}'.format(estimate))
    
    
    def part_d(self):
        
        # ------------ part d -------------------
        P10 = self.P(56)
        P55 = self.P(78)
        print('Exact result for P(S > 10) = {}'.format(P10))
        print('Exact result for P(S > 55) = {}'.format(P55))


    def part_ei(self):
     
        # ------------ part e.i -------------------
        z95 = 1.96
        var = np.var(evaluations)
        std = np.sqrt(var/len(evaluations))
        uncert = z95 * np.sqrt(var/len(evaluations))
        print('MC estimate: P(s>10) = {} +- {}'.format(estimate, std))

    
    def part_eii(self):
    
        # ------------ part e.ii -------------------
        nMC = 1000
        estimates = np.zeros(nMC)
        err = np.zeros(nMC)
        includes_truth = np.zeros(nMC)
        for i in range(1000):
            if(i%10==0): print('running trial {}/{}'.format(i, nMC))
            estimates[i], _, evaluations = mc.monte_carlo(trials, sampler, evaluator)
            err[i] = z95 * np.sqrt(np.var(evaluations) / len(evaluations))
            includes_truth[i] = int(estimates[i] - err[i] < self.P(56)) *\
                                int(estimates[i] + err[i] > self.P(56))
        
        frac_includes_truth = sum(includes_truth)/len(includes_truth)
        print('{}% of estiamtes include the truth within 95% confidence interval'.format(
               frac_includes_truth*100))
        
        # plot a 50 of these samples up
        f = plt.figure(figsize=(8, 3))
        ax = f.add_subplot(111)
        ax.plot([0, 50], np.ones(2) * self.P(56), '--k', lw=1.75, label='truth')
        ax.errorbar(np.arange(50), estimates[:50], yerr=err, fmt='sr', label=r'$\mathrm{MC\>estaimtes}$')
        ax.legend(fontsize=14)
        ax.set_xlabel(r'$\mathrm{trial}$', fontsize=14)
        ax.set_ylabel(r'$P(S_{100}>10)$', fontsize=14)
        plt.savefig('MC_errorbars.png', dpi=300)


    def part_eiii(self):
     
        # ------------ part e.iii -------------------
        
        # fine parameters for pretty plots
        #nMC = 50
        #downsampleM = 1
        #trials = int(1e4)
        #lower_xlim = 1
        #plot = True
        
        # parameters for accurate results
        nMC = 1000
        downsampleM = 1000
        trials = int(1e5)
        lower_xlim = 100
        plot = False
        
        M = np.arange(trials, dtype=int)
        M_coarse = M[1::downsampleM]
        running_estimations = np.zeros((nMC, len(M_coarse)))
        true_prob = self.P(56)
             
        for j in range(nMC):
            
            # compute probability that S > 10 via monte carlo as a running estimate with M
            print('performing MC estimation {}/{} for N={} with {} trials'.format(
                   j, nMC, self.N, trials))
            sampler = lambda nsamples: [np.diff(bernoulli_walk_1d(100)) for i in range(nsamples)]
            evaluator = lambda X: np.array(np.sum(X, axis=1) > 10, dtype=int)
            _, samples, evaluations = mc.monte_carlo(trials, sampler, evaluator)

            for i in range(len(M_coarse)):
                #if(i%1000 == 0): 
                #    print('computing running estimate at M={}/{}'.format(M[i], trials))
                subevaluations = evaluations[0:M_coarse[i]+1]
                running_estimations[j][i] = np.sum(subevaluations, axis=0) / len(subevaluations)
                running_estimations[j][i] -= true_prob
            
        # compute envelopes at 95%
        upperq = np.quantile(running_estimations, q=0.95, axis=0)
        lowerq = np.quantile(running_estimations, q=0.05, axis=0)
        print('upper envelope at x={}: {}'.format(trials, upperq[-1]))
        print('lower envelope at x={}: {}'.format(trials, lowerq[-1]))

        if(plot):
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(M_coarse, running_estimations.T, '-k', alpha=0.25)
            ax.plot([0, 0], [0, 0], '-k', label=r'$\mathrm{MC\>running\>estimates}$')
            ax.plot(M_coarse, upperq, '-r', lw=2, label=r'$95\%\>\mathrm{quantiles}$')
            ax.plot(M_coarse, lowerq, '-r')
            ax.tick_params(labelright=True)
            
            ax.set_xlabel(r'$M$', fontsize=14)
            ax.set_ylabel(r'$P(S_{100} > 10) - P_\mathrm{true}(S_{100} > 10)$', fontsize=14)
            plt.legend()
            ax.set_ylim([ax.get_ylim()[0], -ax.get_ylim()[0]])
            ax.set_ylim([-0.05, 0.05])
            ax.set_xlim([lower_xlim, trials])
            ax.set_xscale('log')
            plt.savefig('running_estimation_downs{}_trials{}_lx{}.png'.format(
                         downsampleM, trials, lower_xlim), dpi=300)
        

# --------------------------------------------------------------------------------
 
 
class problem3_2:

    def __init__(self):
        
        # define sampler for 3d walk; components and total distance
        self.sampler_1d = lambda N: np.array([gaussian_motion_1d(100) for i in range(N)])
        self.sampler_3d = lambda N: np.array([[gaussian_motion_1d(100) for i in range(3)] 
                                                                       for j in range(N)])
    
    def part_a(self, nwalks=3):
    
        # plot nwalks sample paths in 3d
        print('Taking {} random walks'.format(nwalks))
        samples = self.sampler_3d(nwalks)
        
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        
        for i in range(len(samples)):
            walk = samples[i]
            x, y, z = walk[0], walk[1], walk[2]
            ax.plot(x, y, z, '-', color=plt.cm.plasma(0.25 + 0.2*i))
        
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$y$', fontsize=14)
        ax.set_zlabel(r'$z$', fontsize=14)
        plt.savefig('3d_gaussian_walk.png', dpi=300)
        return
    
    
    def part_b(self):
    
        # monte carlo method to find P(|S|>10)
        # X[i].T[-1] gets x,y,z for the final position of the i'th trial
        print('Computing Monte Carlo estimate for P(|S| > 10)')
        evaluator = lambda X: [np.linalg.norm(X[i].T[-1]) > 10 for i in range(len(X))]
        estimate, samples, evaluations = mc.monte_carlo(int(1e5), self.sampler_3d, evaluator)

        # report errors
        z95 = 1.96
        var = np.var(evaluations)
        std = np.sqrt(var/len(evaluations))
        uncert = z95 * np.sqrt(var/len(evaluations))
        print('MC estimate: P(|S| > 10) = {} +- {}'.format(estimate, std)) 
        
        # visualize first 50 walks in x-y plane
        f = plt.figure(figsize=(6,6))
        ax = f.add_subplot(111)
        ax.plot(samples[:100,0,:].T, samples[:100,1,:].T, '-k', alpha=0.25)
        ax.set_xlim([min(ax.get_ylim()[0], ax.get_xlim()[0]),
                     max(ax.get_ylim()[1], ax.get_xlim()[1])])
        ax.set_ylim(ax.get_xlim())
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_xlabel(r'$y$', fontsize=14)
        plt.savefig('MC_3d_gaussian_walks.png', dpi=300)
        
            
    def part_c(self):
        return 
        
    def part_d(self):
        return
    



# --------------------------------------------------------------------------------
    

if __name__ == '__main__':
    
    #p31 = problem3_1(N = 100)
    #p31.part_a()
    #p31.part_b()
    #p31.part_c()
    #p31.part_d()
    #p31.part_ei()
    #p31.part_eii()
    #p31.part_eiii()
    
    p32 = problem3_2()
    p32.part_a()
    p32.part_b()
