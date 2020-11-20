
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


def multilevel_montecarlo_sde()
