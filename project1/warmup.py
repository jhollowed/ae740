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


def monte_carlo(n, sampler, g, sampler_kwargs=None, plot=False, pdf=None):
    '''
    Numerically estimates the expectation value of a general proabbility 
    distribution via the Monte Carlo method.

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
    #pdb.set_trace()

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
 

# ---------------------------------------------------------------


def mc_inverse_sampling():

    # define Pareto random variable, and g(x)
    X = rv.Pareto(alpha=3/2)
    g = lambda x: x
    true_mean = X.mean
    
    # declare plot
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$N$', fontsize=14)
    ax.set_ylabel(r'$\mathbb{E}[X]$', fontsize=14)
        
    print('running MC with inverse transform sampler') 
    
    # define n, mean array, and sampler
    n = np.logspace(1, 6, 100, dtype=int)
    mean = np.zeros(len(n)) 
    sampler = ss.InverseTransformSampler(X).sample

    # perform MC procedure for each value of n
    for j in range(len(n)):
        if(j % 20 == 0): print('running n {}/{}'.format(j+1, len(n)))
        [mean[j], samples] = monte_carlo(n[j], sampler, g, sampler_kwargs={'plot':True}, 
                                         plot=False, pdf=X.probability_density) 
    # plot
    ax.plot(n, mean, '-', color=cm(0.33)) 
    ax.plot([min(n), max(n)], [true_mean, true_mean], '--k', label=r'$\mathrm{true\>mean}$')
    ax.set_xscale('log')
    ax.set_ylim([1, 5])
    ax.legend(fontsize=14)
    plt.savefig('{}/convegence_inverse_transform_sampling.png'.format(wd), dpi=300)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------


def mc_rejection_sampling():

    # define Pareto random variable, and g(x)
    X = rv.Pareto(alpha=3/2)
    g = lambda x: x
    true_mean = X.mean
    
    # declare b's (upper bound of rejection sampling)
    all_b = [10, 20, 40, 80, 160]
   
    # declare plot
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$N$', fontsize=14)
    ax.set_ylabel(r'$\mathbb{E}[X]$', fontsize=14)
        
    for i in range(len(all_b)):
        print('running MC with rejection sampler for b={}'.format(all_b[i])) 
        
        # define n, mean array, and sampler
        n = np.logspace(1, 5, 100, dtype=int)
        mean = np.zeros(len(n)) 
        a, b = 0, all_b[i]
        sampler = ss.BoundedRejectionSampler(X, a, all_b[i]).sample

        # perform MC procedure for each value of n
        for j in range(len(n)):
            if(j % 20 == 0): print('running n {}/{}'.format(j+1, len(n)))
            [mean[j], samples] = monte_carlo(n[j], sampler, g, sampler_kwargs={'plot':True}, 
                                             plot=False, pdf=X.probability_density)
            n[j] = len(samples)
        
        # since we used rejection sampling, and therefore don't know exactly what the sample
        # sizes ended up being, resort by the returned sample sizes, and purge any duplicates
        sorter = np.argsort(n)
        unique_mask = np.unique(n[sorter], return_index=True)[-1]
        n = n[sorter][unique_mask]
        mean = mean[sorter][unique_mask]

        # plot
        ax.plot(n, mean, '-', color=cm(i/len(all_b)), label=r'$b={}$'.format(b))
        
    ax.plot([min(n), max(n)], [true_mean, true_mean], '--k', label=r'$\mathrm{true\>mean}$')
    ax.set_xscale('log')
    ax.set_ylim([1, 5])
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('{}/convegence_rejection_sampling.png'.format(wd), dpi=300)
    plt.show()


# ---------------------------------------------------------------
 
 
if __name__ == '__main__':
    #mc_inverse_sampling()
    mc_rejection_sampling()
