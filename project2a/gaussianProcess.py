import pdb
import numpy as np
import matplotlib.pyplot as plt


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
   
    pdb.set_trace()
    out = np.zeros((x.shape[0], xp.shape[0]))
    for i in range(xp.shape[0]):
        out[:, i] = kern(x, xp[i])
    return out


# -----------------------------------------------------------


def gpr2d(xin, zin, xpred, noise_var, mean_func, kernel):
    '''
    Gaussian process regression algorithm in two dimensions

    Parameters
    ----------
    xin : (N,2) array
        2-dimensional training inputs
    zin : (N,) array
        training outputs
    xpred: (M,2) array
        locations at which to make predictions
    noise_var : (N,)
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


def plot_function_gp(axis, pred_mean, predict_cov, xdata, ydata, xpred, title, label):
    
    axis.scatter(xdata.T[0], xdata.T[1], c=ydata, marker='^', s=200, linewidths=1, edgecolors='k', 
                 label='data', cmap=plt.cm.viridis)
    pixels = int(np.sqrt(len(xpred.T[0])))
    pred_mean = pred_mean.reshape(pixels, pixels)
    axis.imshow(pred_mean[::-1], cmap=plt.cm.viridis, aspect='equal', interpolation='nearest', 
                extent=[-1, 1, -1, 1], label=label)
    
    #axis.scatter(xpred[0], xpred[1], c=pred_mean, label=label)
    #pred_fstd = np.sqrt(np.diag(predict_cov))
    #axis.fill_between(xpred, pred_mean - 2*pred_fstd, pred_mean+2*pred_fstd, 
    #                  lw=3, label=r'2$\sigma$', color='red', alpha=0.2)

    axis.legend()
    axis.set_title(title, fontsize=14)
    

# -----------------------------------------------------------


def run_gp_regression(mean_func, kernel, xdata, ydata, xspace, noise_cov=1e-1):

    prior_mean = mean_func(xspace)
    prior_cov = build_covariance(xspace, xspace, kernel)
    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    plot_function_gp(axis[0], prior_mean, prior_cov, xdata, ydata, xspace, 
                     'Prior', 'Prior mean')
    
    mean_predict, cov_predict = gpr2d(xdata, ydata, xspace, noise_cov * np.ones((xdata.shape[0])), 
                                      mean_func, kernel)
    if(plot):
        plot_function_gp(axis[1], mean_predict, cov_predict, xdata, ydata, xspace, 
                     'Posterior', 'Posterior mean')
    plt.show()    
