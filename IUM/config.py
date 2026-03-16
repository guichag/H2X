"""IUM runs config file"""

import numpy as np


theta_crit = 0.128   # volumetric soil moisture content at critical point (m3 m-3)

zmax = 40000.   # max height of UM [m]

layer_midpoints = np.array([0.05 , 0.225, 0.675, 2.])
layer_depths = np.array([0.1, 0.25, 0.65, 2.])


longitude_dim = 200
latitude_dim = 200
layer_dim = 4
