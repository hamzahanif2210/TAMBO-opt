import h5py
import numpy as np
import os
from tqdm import tqdm

path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons.h5"
out_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced_v2.h5"

if os.path.exists(out_path):
    os.remove(out_path)
    print("Removed existing file")

CHUNK_SIZE = 10000  # process 10k rows at a time

with h5py.File(path, 'r') as f:
    pdg = f['pdg'][:]

    idx_0 = np.where(pdg == 0)[0]
    idx_1 = np.where(pdg == 1)[0]

    rng = np.random.default_rng(42)
    idx_0_sampled = rng.choice(idx_0, size=65000, replace=False)
    idx_1_sampled = rng.choice(idx_1, size=65000, replace=False)

    idx_combined = np.concatenate([idx_0_sampled, idx_1_sampled])
    idx_sorted = np.sort(idx_combined)
    rng2 = np.random.default_rng(99)
    shuffle_order = rng2.permutation(len(idx_sorted))

    # Apply shuffle to the sorted indices directly
    # so we only need one read pass per dataset
    idx_final = idx_sorted[shuffle_order]  # final order, but unsorted = slow for h5py
    # re-sort for h5py reading, track where each ends up
    argsort = np.argsort(idx_final)
    idx_read_order = idx_final[argsort]         # sorted for h5py
    restore_order = np.argsort(argsort)         # to restore shuffle after reading

    total = len(idx_read_order)

    datasets = ['showers', 'directions', 'energies', 'pdg', 'actual_pdg', 'shower_ids', 'num_points']

    with h5py.File(out_path, 'w') as out:
        # Pre-create datasets with correct shape and dtype
        print("Creating output datasets...")
        for name in datasets:
            ds = f[name]
            shape = (total,) + ds.shape[1:]
            out.create_dataset(name, shape=shape, dtype=ds.dtype, chunks=True)
        out.create_dataset('shape', data=np.array([total, 6000, 5], dtype=np.int64))

        # Process each dataset in chunks
        for name in datasets:
            print(f"\nProcessing '{name}'...")
            ds = f[name]

            # Read in chunks
            all_data = []
            for start in tqdm(range(0, total, CHUNK_SIZE), desc=f"  Reading '{name}'", unit="chunk"):
                end = min(start + CHUNK_SIZE, total)
                chunk_idx = idx_read_order[start:end]
                all_data.append(ds[chunk_idx])

            # Concatenate, restore shuffle order, write
            print(f"  Concatenating & shuffling '{name}'...")
            all_data = np.concatenate(all_data, axis=0)
            all_data = all_data[restore_order]

            print(f"  Writing '{name}'...")
            out[name][:] = all_data
            del all_data  # free RAM immediately

print(f"\nDone! {total} entries saved.")
print(f"Shape dataset: [{total}, 2000, 5]")