import pdb
import warnings
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')


# ================================================================================
# ================================================================================


class BoundedRejectionSampler:
    def __init__(self, rv, a, b):
        '''
        Constructs a rejection sampler with bounded support, initialized for a particular pdf

        Parameters
        ----------
        rv : randomVariables instance object
            and instance of one of the random variable classes offered in randomVariables, 
            having a probability_density method
        a : float
            The lower bound of y
        b : float
            The upper bound of y

        Attributes
        ----------
        a : float
            The lower bound of y
        b : float
            The upper bound of y 
        pdf
            function handle for probability_density method of input randomVariable instance
        '''
        self._rv = rv
        self.pdf = rv.probability_density
        self.a = a
        self.b = b
        
        # If the max of the distribution is know a priori (i.e. the input object rv has a 
        # valid rv.max attribute), then define a scaling for n by the analytic acceptance 
        # probability of the method, to target a number of returned samples which is near 
        # the caller request.
        if hasattr(self._rv, 'max'):
            self._m = self.pdf([self._rv.max])
            self._n_scaling = self._m*(self.b-self.a)/1
        else:
            self._m = None
            self._n_scaling = 1
        
    # ---------------------------------------------------------------------
        
    def sample(self, n, **kwargs):
        '''
        Samples from the pdf via bounded rejection sampling

        Parameters
        ----------
        n : int
            The number of samples to draw
        
        kwargs
        ------
        plot : bool
            Whether or not to plot all cadidate points, color
            coded by rejection or acceptance

        Returns
        -------
        x : float array
            The samples x
        '''
       
        # decalre keywords args
        plot = False
        for key, value in kwargs.items():
            if(key == 'plot'): plot = bool(value)
            else:  warnings.warn('Unknown keyword argument {}; ignoring'.format(key))

        # scale n
        n = int(self._n_scaling * n)
        
        # sample uniform RV's
        # if m was not computed at object initialization, estimate it now by the
        # current samping
        y = np.random.uniform(low=self.a, high=self.b, size=n)
        fXy = self.pdf(y)
        if(self._m is None): self._m = np.max(fXy)
        u = np.random.uniform(low=0, high=self._m, size=n)

        # marginalize over u
        mask = np.logical_and(0 <= u, u <= fXy, dtype=bool)
        x = y[mask]
        
        # plot
        if(plot):
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(y[~mask], u[~mask], '.r', label=r'$\mathrm{rejected}$')
            ax.plot(y[mask], u[mask], '.b', label=r'$\mathrm{accepted}$')
            
            xx = np.linspace(self.a, self.b, 1000)
            ax.plot(xx, self.pdf(xx), '--k', label=r'$f_X(x)$')
            
            ax.legend(fontsize=14)
            ax.set_xlabel('y', fontsize=14)
            ax.set_xlabel('u', fontsize=14)
            ax.set_title('n = {}'.format(n), fontsize=16)
            plt.show()
        
        return x


# ================================================================================
# ================================================================================


class InverseTransformSampler:
    def __init__(self, rv):
        '''
        Constructs an inverse transform sampler, initialized for a particular random
        variable with known CDF.

        Parameters
        ----------
        rv : randomVariables instance object
            and instance of one of the random variable classes offered in randomVariables, 
            having a probability_density method

        Attributes
        ----------
        pdf
            function handle for probability_density method of input randomVariable instance
        cdf
            function handle for cumulative_density method of input randomVariable instance
        '''
        self._rv = rv
        self.pdf = rv.probability_density
        self.icdf = rv.inverse_cumulative_density
        
    # ---------------------------------------------------------------------
        
    def sample(self, n, **kwargs):
        '''
        Samples from the pdf

        Parameters
        ----------
        n : int
            The number of samples to draw
        
        kwargs
        ------
        plot : bool
            Whether or not to plot all cadidate points, color
            coded by rejection or acceptance

        Returns
        -------
        x : float array
            The samples x
        '''
        u = np.random.uniform(low=0, high=1, size=n)
        x = self.icdf(u)
        return x

