# import h5py

# with h5py.File('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5', 'r') as f:
#     def print_all(name, obj):
#         print(name, "→", type(obj).__name__)
    
#     f.visititems(print_all)

# import h5py
# import numpy as np

# with h5py.File('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5', "r") as f:
#     pc = f["target/point_clouds"][:10]   # array of variable-length flat float32
#     npts = f["target/num_points"][:10]   # int32, real hits per shower
#     F = f["target"].attrs["num_features"]
#     print(pc)

# # Unpack vlen -> list of (num_points, F) arrays
# showers = [pc[i].reshape(npts[i], F) for i in range(len(pc))]

import h5py

path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5"

items = []

def collect(name, obj):
    if isinstance(obj, h5py.Dataset):
        if obj.dtype.kind == "O":  # vlen/object dtype
            storage = obj.id.get_storage_size()   # actual bytes on disk
            logical = None
        else:
            storage = obj.id.get_storage_size()   # actual bytes on disk
            logical = obj.size * obj.dtype.itemsize  # raw uncompressed size in memory

        items.append({
            "name": name,
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "storage_mb": storage / 1024**2,
            "logical_mb": None if logical is None else logical / 1024**2,
            "compression": obj.compression,
        })

with h5py.File(path, "r") as f:
    f.visititems(collect)

items.sort(key=lambda x: x["storage_mb"], reverse=True)

for x in items:
    logical_str = "vlen/object" if x["logical_mb"] is None else f"{x['logical_mb']:.2f} MB"
    print(
        f"{x['name']:<35} "
        f"shape={str(x['shape']):<20} "
        f"dtype={x['dtype']:<15} "
        f"on_disk={x['storage_mb']:8.2f} MB   "
        f"logical={logical_str:<12} "
        f"compression={x['compression']}"
    )