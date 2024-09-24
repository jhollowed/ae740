import shelter
import pdb
import numpy as np
import sys
import pathlib
import matplotlib.pyplot as plt
from shelter import xmid2p, xmid1p, zpacks_duplex

sys.path.append('{}/../project1'.format(pathlib.Path(__file__).parent.absolute()))
sys.path.append('{}/../project2'.format(pathlib.Path(__file__).parent.absolute()))
import monteCarlo


def random_sampling(nstakes, N, symmetry = 'biradial'):

    if(symmetry == 'biradial'):
        assert(nstakes % 4 == 0), 'nstakes must be divisible by 4'
        free_stakes = int((nstakes/4))
    if(symmetry == 'bilateral'): 
        assert(nstakes % 2 == 0), 'nstakes must be divisible by 2'
        free_stakes = int((nstakes/2))
    
    g = shelter.shelter(nstakes, symmetry=symmetry)
    sx = np.zeros((N, free_stakes))
    sy = np.zeros((N, free_stakes))
    px = np.zeros(N)
    py = np.zeros(N)
    ve = np.zeros(N)
    wp = np.zeros(N)
    pp = np.zeros(N)

    for i in range(N):
        if(i%100 == 0): print(i)
        g.sample_footprint()
        g.sample_poles()
        g.pitch()
        fp = g.get_free_params()
        sx[i] = fp[0][:,0]
        sy[i]= fp[0][:,1] / sx[i]
        px[i]= fp[1][0] / sx[i]
        py[i]= fp[1][1] / (sy[i] * sx[i])
        ve[i] = shelter.compute_volumetric_efficiency(g)
        wp[i] = shelter.compute_weather_performance(g)
    pp = (ve/np.max(ve)) * (wp/np.max(wp))

    # plot in 3D
    f = plt.figure(figsize=(10,8))
    ax4 = f.add_subplot(221, projection='3d')
    ax5 = f.add_subplot(222, projection='3d')
    ax6 = f.add_subplot(223, projection='3d')
   
    sctr4 = ax4.scatter(px, py, sy[:,0], c=ve, marker='o', cmap=plt.cm.viridis, alpha=0.5)
    sctr5 = ax5.scatter(px, py, sy[:,0], c=wp, marker='o', cmap=plt.cm.viridis, alpha=0.5)
    sctr6 = ax6.scatter(px, py, sy[:,0], c=pp, marker='o', cmap=plt.cm.viridis, alpha=0.5)

    scatters = [sctr4, sctr5, sctr6]
    metrics = [r'$\epsilon_V$', r'$P_W$', r'$\epsilon_VP_W$']
    axes = [ax4, ax5, ax6]
    xmid_s = 60
    xmid_lw = 2
    
    for i in range(len(axes)):
        axes[i].scatter([xmid1p.norm_px], [xmid1p.norm_py], [xmid1p.norm_y], color='r', marker='x', s=xmid_s, lw=xmid_lw)
        axes[i].scatter([xmid2p.norm_px], [xmid2p.norm_py], [xmid2p.norm_y], color='r', marker='^', s=xmid_s, lw=xmid_lw)
        axes[i].scatter([zpacks_duplex.norm_px], [zpacks_duplex.norm_py], [np.max(zpacks_duplex.norm_y)], color='m', marker='^', s=xmid_s, lw=xmid_lw)
        axes[i].set_xlabel(r'$p_x / s_x$', fontsize=14)
        axes[i].set_ylabel(r'$p_y / s_y$', fontsize=14)
        axes[i].set_zlabel(r'$s_y / s_x$', fontsize=14)
        cbar = f.colorbar(scatters[i], ax=axes[i])
        cbar.set_label(metrics[i], fontsize=14)
    plt.tight_layout()
    plt.show()
   
if __name__ == '__main__':
    random_sampling(4, 1000)


for i in range(100):
    g = shelter.shelter(8, symmetry='biradial')
    g.sample_footprint()
    g.sample_poles()
    g.pitch()
    ve = shelter.compute_volumetric_efficiency(g)
    wp = shelter.compute_weather_performance(g)
    pp = ve * wp
    print('{}, {}, {}'.format(ve, wp, pp))
    g.vis_layout(ion=False)

    #g.choose_footprint(np.array([[50, 50]]))
    #g.choose_poles(np.array([49.99, 49.99]))
    #g.pitch()
    #ve = shelter.compute_volumetric_efficiency(g)
    #wp = shelter.compute_weather_performance(g)
    #pp = ve * wp
    #print('{}, {}, {}'.format(ve, wp, pp))
    #g.vis_layout()    
