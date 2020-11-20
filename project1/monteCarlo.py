
import pdb
import numpy as np
import sampleSchemes as ss
import randomVariables as rv
import matplotlib.pyplot as plt

import pathlib
wd = pathlib.Path(__file__).parent.absolute()

import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
cm = plt.cm.plasma


# ===============================================================


def monte_carlo(n, sampler, evaluator):
    '''
    Performs Monte Carlo sampling for a general sampler and evaluator

    Parameters
    ----------
    n : int
        The number of samples to draw
    sampler : function handle
        The sampler function for the random variable X
    gevaluator : function handle
        Function implementing the evaluator (takes samples as inputs)
    '''

    samples = sampler(n)
    evaluations = evaluator(samples)
    estimate = np.sum(evaluations, axis=0) / n

    return estimate, samples, evaluations


# ------------------------------------------------------------------


def monte_carlo_expectation(n, sampler, g, sampler_kwargs=None, plot=False, pdf=None):
    '''
    Numerically estimates the expectation value of a general probability 
    distribution via the Monte Carlo method. This is just version of the above function
    with plotting capabilities specific to the warmpup exercise.

    Parameters
    ----------
    sampler : function handle
        The sampler function for the random variable X
    gx : function handle
        Function implementing the pdf
    n : int
        The number of samples to draw from uniform
    sampler_kwargs : dict, optional
        optional keyword arguments to pass to the sampler, as a dictionary. Defaults to None
    plot : bool, optional
        Whether or not to plot the new random variable Y=g(X). Defaults to False
    pdf : function handle, optional 
        function handle to probability_density method of a randomVariable object instance, 
        implementing underlying limiting distribution. Only used for plotting. Defaults to None, 
        in which case only a histogram of the samples is plotted for plot=True
    '''

    samples = np.sort(sampler(n))
    gx = g(samples)
    mean = np.sum(gx) / len(gx)

    if(plot):
        std_samples = np.linspace(min(samples), max(samples), 1000)
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.hist(samples, bins=100, density=True, label=r'$\mathrm{sampled}\>X$', color=cm(0.75))
        if(pdf is not None):
            ax.plot(std_samples, pdf(std_samples), '-k', label=r'$f_X(x)$')
        ax.legend(fontsize=12)
        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$g(X)$', fontsize=14)
        plt.show()

    return mean, samples


# ------------------------------------------------------------------


def twolevel_montecarlo_sde(gbm, evaluator, nsamples, dt_fine, M):
    '''
    Multilevel Monte Carlo for geometric Brownian motion SDE

    Parameters
    ----------
    gbm : geometric_brownian_motion instance object
    evaluator : function handle
        The evaluating function
    nsamples : (m, ) array
        Number of samples per Monte Carlo level (m levels)
    M : (m, ) array
        coarsening factor per Monte Carlo level (m levels)
    '''

    dt_coarse = dt_fine * M
    T = gbm.T
    
    # --- level 0; only coarse grid evaluations are used
    bmotion_level_0 = gbm.walk(dt_coarse, nsamples)
    samples_level_0 = evaluator(bmotion_level_0, dt_coarse, T)

    # --- level 1
    # --- First generate fine and coarse samples
    bmotion_level_1_fine = sampler(nsamples[1])
    bmotion_level_1_coarse = gbm.fine_to_coarse(dt_coarse[0], M[0], level_1_fine)

    # --- Now evaluate the samples
    samples_level1_fine = evaluator(level_1_fine, dt_fine, T)
    samples_level1_coarse = evaluator(level_1_coarse, dt_fine, T)

    # --- subtract
    samples_delta_level_1 = samples_level_1_fine - samples_level_1_coarse
    
    # --- combine levels
    est_mean = np.mean(samples_level_0) + np.mean(samples_delta_level_1)
    est_var_levels = (np.var(samples_level_0), np.ar(samples_level_1_fine), np.var(samples_delta_level_1))
    
    return est_mean, est_var_levels


# ------------------------------------------------------------------


def multilevel_montecarlo_sde(gbm, dt, nsamples):
    '''
    Multilevel Monte Carlo for the expectation of Y(t=1) for the geometric Brownian motion SDE

    Parameters
    ----------
    gbm : geometric_brownian_motion instance object
    nsamples : int
        Number of samples per Monte Carlo level
    dt : float array
        timestep sizes per level, in decreasing order
    '''
    
    T = gbm.T
    n_levels = len(dt)
    sampler = lambda dt: gbm.walk(dt, nsamples)[0]

    # dt[-1] is dt_fine, decreasing monotonically
    # M is the coarsening factor between each level
    dt_fine = dt[0]
    M = np.hstack([[1], [dt[i]/dt[i+1] for i in range(len(dt)-1)] ])
    level_means = np.zeros(n_levels)
    level_vars = np.zeros(n_levels)
    discrepancy_vars = np.zeros(n_levels)
   
    for l in range(len(M)):

        if(l == 0):
            # --- level 0; only coarsest grid evaluations are used
            # --- [:,-1] gets last position of all motion samples (Y(1))
            samples_level_i = sampler(dt[0])[:,-1]
            
            # --- take mean and var; [:,-1] gets last position of all motion samples (Y(1))
            level_means[l] = np.mean(samples_level_i)
            level_vars[l] = np.var(samples_level_i)
            discrepancy_vars[l] = 0
        
        else:
            # --- level i
            # --- First generate fine and coarse samples
            samples_level_l_fine = sampler(dt[l])
            samples_level_l_coarse = gbm.fine_to_coarse(dt[l-1], M[l], samples_level_l_fine)[:,-1]
            samples_level_l_fine = samples_level_l_fine[:,-1]

            # --- subtract
            samples_delta_level_l = samples_level_l_fine - samples_level_l_coarse
           
            # --- take mean and var
            level_means[l] = np.mean(samples_delta_level_l)
            level_vars[l] = np.var(samples_level_l_fine)
            discrepancy_vars[l] = np.var(samples_delta_level_l)
    
    # --- combine levels
    est_mean = np.sum(level_means)
    est_var_levels = level_vars
        
    return est_mean, est_var_levels, discrepancy_vars 
