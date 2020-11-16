import numpy as np
import pdb

'''
For all functions included here:

Parameters
----------
x: (M, N) array of multiple inputs
xp: float

Returns
-------
cov (M,) -- Covarance between each input at x, and the function values at x
'''


# --------------------------------------------------------------------------------------------


def sqExpNd(x, xp, tau, l):
    '''
    Squared exponential kernel (N-dimensional)
    ''' 
    N = len(xp)
    kfunc = lambda x,xp,tau,l: tau**2 * np.exp(-1/2 * (x-xp)**2 / l**2)
    cov = np.prod([kfunc(x.T[i], xp[i], tau[i], l[i]) for i in range(N)], axis=0)
    return cov


# --------------------------------------------------------------------------------------------


def periodicNd(x, xp, tau=1, l=1.0, p=0.4):
    '''
    Periodic kernel (N-dimensional)
    '''

    N = len(x)
    kfunc = lambda x,xp: tau**2 * np.exp (-2 * np.sin(np.pi*np.abs(x - xp) / p)**2 / l**2)
    cov = np.prod([kfunc(x[i], xp[i]) for i in range(N)])
    return cov


# --------------------------------------------------------------------------------------------


def polyNd(x, xp, c=1, d=3):
    '''
    Polynomial kernel (N-dimensional)
    '''

    N = len(x)
    kfunc = lambda x,xp: (x * xp + c)**d
    cov = np.prod([kfunc(x[i], xp[i]) for i in range(N)])
    return cov
