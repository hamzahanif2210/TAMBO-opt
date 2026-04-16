# import h5py

# with h5py.File('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5', 'r') as f:
#     def print_all(name, obj):
#         print(name, "→", type(obj).__name__)
    
#     f.visititems(print_all)

import h5py
import numpy as np

with h5py.File('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5', "r") as f:
    pc = f["target/point_clouds"][:10]   # array of variable-length flat float32
    npts = f["target/num_points"][:10]   # int32, real hits per shower
    print(pc)
    print(npts)
