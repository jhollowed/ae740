import pdb
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable



# ============================================================================
# ============================================================================



def build_covariance(x, xp, kern):
    '''
    Builds a covariance matrix

    Parameters
    ----------
    x: (N) array of inputs
    xp: (M) array of primed inputs
    kern: a function mapping inputs to covariance
    
    Returns
    -------
    cov: (N, M) covariance matrix between x, xp
    '''
   
    out = np.zeros((x.shape[0], xp.shape[0]))
    for i in range(xp.shape[0]):
        out[:, i] = kern(x, xp[i])
    return out


# -----------------------------------------------------------


def sample_functions(x, nsamples, mean_func, kern):
    '''
    Generate samples of functions from a Gaussian Process

    Parameters
    ----------
    x : (N,) vector
        locations at which to obtain samples
    nsamples : int
        number of funtions to sample
    mean_func : function hanle
        function to evaluate the prior at each location
    kern : function handle
        covariance kernel function

    Returns
    -------
    out : (N, nsamples) 2d array
        array of sampled function evaluations, per position x 
    '''

    if callable(mean_func):
        # mean prior was a callable
        prior = np.tile(mean_func(x)[:, np.newaxis], (1, nsamples))
    else:
        # mean prior was given as evaluated on the sample locations x
        prior = np.tile(mean_func[:, np.newaxis], (1, nsamples))

    cov = build_covariance(x, x, kern)
    u, s, v = np.linalg.svd(cov)
    sqrtcov = np.dot(u, np.sqrt(np.diag(s)))
    out = prior + np.dot(sqrtcov, np.random.randn(x.shape[0], nsamples))
    return out.T


# -----------------------------------------------------------


def gpr_Md(xin, zin, xpred, noise_var, mean_func, kernel):
    '''
    Gaussian process regression algorithm in M dimensions, with 
    Ni training inputs, and Np prediction outpits

    Parameters
    ----------
    xin : (Ni,M) array
        M-dimensional training inputs
    zin : (N,) array
        training outputs
    xpred: (Np,M) array
        locations at which to make predictions
    noise_var : (Np,)
        noise at every training output
    kernel : instance object from kernels.py
        covaraince kernel to use
    '''
    
    cov = build_covariance(xin, xin, kernel)
    u, s, v = np.linalg.svd(cov)
    sqrtcov = np.dot(u, np.sqrt(np.diag(s)))

    invcov = np.linalg.pinv(cov + np.diag(noise_var))
    
    vec_pred = build_covariance(xpred, xin, kernel)
    pred_mean = mean_func(xpred) + np.dot(vec_pred, np.dot(invcov, zin - mean_func(xin)))
    
    cov_predict_pre = build_covariance(xpred, xpred, kernel)
    cov_predict_up = np.dot(vec_pred, np.dot(invcov, vec_pred.T))
    pred_cov = cov_predict_pre - cov_predict_up
    
    return pred_mean, pred_cov


# -----------------------------------------------------------


def plot_function_gp_2d(axis, grid_var, xdata, zdata, xpred, title, label, 
                        cm=plt.cm.viridis, colorbar=True, cbar_label='sensor'):
    
    pixels = int(np.sqrt(len(xpred.T[0])))
    grid_var = grid_var.reshape(pixels, pixels)
    
    im = axis.imshow(grid_var[::-1], cmap=cm, aspect='equal', interpolation='nearest', 
                extent=[-1, 1, -1, 1], label=label, vmin=0)
    
    norm_marker_area = (zdata/min(zdata))**10 * 10
    axis.scatter(xdata.T[0], xdata.T[1], c='k', marker='o', s=norm_marker_area, linewidths=1, edgecolors='w', 
                 label='data', cmap=plt.cm.viridis, alpha=0.75)
    axis.set_xlabel('x', fontsize=14)
    axis.set_ylabel('y', fontsize=14)
    
    if(colorbar):
        cbar = plt.gcf().colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=14)
    
    axis.legend()
    axis.set_title(title, fontsize=14)


# -----------------------------------------------------------


def plot_function_gp_2Nd(grid_var, xdata, zdata, xpred, title, label, 
                         cm=plt.cm.viridis, colorbar=True, cbar_label='sensor'):
    '''
    WIP
    '''
    fig = plt.figure()
    M = len(xdata[0])
    num_axes = comb(M, 2) 
        
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=1.5)
    ax1 = fig.add_subplot(gs[0, :2], )
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 1:3])
    
    plot_function_gp(ax1, prior_mean, xdata, ydata, xspace, 'Prior', 'Prior mean', colorbar=False)
    plot_function_gp(ax2, mean_predict, xdata, ydata, xspace, 'Posterior', 'Posterior mean')
    
    pred_2sig = 2 * np.sqrt(np.diag(cov_predict))
    plot_function_gp(ax3, pred_2sig, xdata, ydata, xspace, 'Covaraince', 'Posterior Covariance', 
                     cm = plt.cm.plasma, cbar_label=r'$2\sigma$') 
    plt.savefig('gp_regression_p{}{}_{}.png'.format(plot_sfx), dpi=300)
    
    

# -----------------------------------------------------------


def run_gp_regression(mean_func, kernel, xdata, ydata, xspace, noise_cov=1e-1, 
                      plot=False, dim_labels=None, plot_sfx=''):
    '''
    Runs a M-dimensional GP regressoin

    Parameters
    ----------
    mean_func : function handle
        Function implementing the prior
    kernel : function handle
        Function implementing the kernel
    xdata : (Ni, M) array
        Ni data inputs in M dimensions
    ydata : (Ni,) array
        Ni data outputs
    xspace : (Np, M) array
        Np prediction points in M dimensions
    noise_cov : float
        Noise to add to the covariance
    plot : bool
        Whether or not to visualize the output GP result in projected parameter planes.
        If True, each plane of parameter space will occupy a 2d colormap in a triangle plot
    dim_labels : (M,) string array
        Names of the M dimensions for aces labelling
    plot_sfx : string
        String to append to the end of the figure image filename

    Return
    ------
    mean_predict : (Np,) array
        regression prediction at each prediciton site of xspace
    mean_cov : (Np,) array
        regression covaraince at each prediciton site of xspace
    '''

    prior_mean = mean_func(xspace)
    prior_cov = build_covariance(xspace, xspace, kernel) 
    mean_predict, cov_predict = gpr_Md(xdata, ydata, xspace, noise_cov * np.ones((xdata.shape[0])), 
                                       mean_func, kernel)
    if(plot):
        plot_funciton_gp_Nd(mean_predict, xdata, ydata, xspace, 'Posterior', 'Posterior mean')
    
    return mean_predict, cov_predict
