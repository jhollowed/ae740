tmport pdb
import numpy as np
import sampleSchemes as ss
import randomVariables as rv
import monteCarlo as mc
import matplotlib.pyplot as plt

import pathlib
wd = pathlib.Path(__file__).parent.absolute()

import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
cm = plt.cm.plasma


# ===============================================================


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
        [mean[j], samples] = mc.monte_carlo_expectation(n[j], sampler, g, sampler_kwargs={'plot':True}, 
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
            [mean[j], samples] = mc.monte_carlo_expectation(n[j], sampler, g, 
                                                            sampler_kwargs={'plot':True}, 
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
