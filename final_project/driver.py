from shelter import shelter 
import pdb
import numpy as np

for i in range(100):
    g = shelter(4, symmetry='biradial')
    g.sample_footprint()
    g.sample_poles()
    g.pitch()
    print(g.compute_volumetric_efficiency())
    g.vis_layout()

    g.choose_footprint(np.array([[50, 34]]))
    g.choose_poles(np.array([35, 20]))
    g.pitch()
    print(g.compute_volumetric_efficiency())
    g.vis_layout()
    
