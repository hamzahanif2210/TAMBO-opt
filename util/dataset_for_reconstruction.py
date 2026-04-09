#!/usr/bin/env python3
"""
dataset_for_reconstruction.py

Takes three HDF5 files (electron, muon, photon) and creates a combined HDF5
file for training the reconstruction model.

From each file it extracts:
  - directions  (N, 3)
  - pdg         (N,)

And calculates per-layer observables:
  - num_points_per_layer  (N, num_layers)
  - energy_per_layer      (N, num_layers)
  - time_per_layer        (N, num_layers)   [average time per layer; time = feature index 4]

The output file stores these with particle-type suffixes
(e.g. num_points_per_layer_electron) so that the reconstruction model
can condition on all particle-type observables simultaneously.
For a given shower, only the columns corresponding to its particle type
are non-zero; the rest are zero-padded.
"""

import argparse
import os

import h5py
import numpy as np


# ── Particle-type mapping ────────────────────────────────────────────────────
PARTICLE_TYPES = ["electron", "muon", "photon"]


# ── Per-layer observable computation ─────────────────────────────────────────

def calc_num_points_per_layer(h5_showers, start: int, stop: int, num_layers: int) -> np.ndarray:
    """Number of hits (energy > 0) per layer."""
    num_showers = stop - start
    result = np.zeros((num_showers, num_layers), dtype=np.int32)
    for i, gi in enumerate(range(start, stop)):
        pts = np.array(h5_showers[gi]).reshape(-1, 5)
        layer = np.clip((pts[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
        mask = (pts[:, 3] > 0).astype(np.int32)
        np.add.at(result[i], layer, mask)
    return result


def calc_energy_per_layer(h5_showers, start: int, stop: int, num_layers: int) -> np.ndarray:
    """Total energy per layer."""
    num_showers = stop - start
    result = np.zeros((num_showers, num_layers), dtype=np.float32)
    for i, gi in enumerate(range(start, stop)):
        pts = np.array(h5_showers[gi]).reshape(-1, 5)
        layer = np.clip((pts[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
        energies = pts[:, 3] * (pts[:, 3] > 0).astype(np.float32)
        np.add.at(result[i], layer, energies)
    return result


def calc_time_per_layer(h5_showers, start: int, stop: int, num_layers: int) -> np.ndarray:
    """Average time (feature index 4) per layer."""
    num_showers = stop - start
    time_sum = np.zeros((num_showers, num_layers), dtype=np.float64)
    count = np.zeros((num_showers, num_layers), dtype=np.int32)
    for i, gi in enumerate(range(start, stop)):
        pts = np.array(h5_showers[gi]).reshape(-1, 5)
        layer = np.clip((pts[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
        mask = pts[:, 3] > 0
        np.add.at(time_sum[i], layer[mask], pts[mask, 4])
        np.add.at(count[i], layer[mask], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(count > 0, time_sum / count, 0.0)
    return avg.astype(np.float32)


# ── Label utilities ──────────────────────────────────────────────────────────

def create_label_list(pdg_arrays: list[np.ndarray]) -> list[int]:
    """Deterministic label list sorted by (|pdg|, -pdg)."""
    all_pdg = np.concatenate(pdg_arrays)
    unique = np.unique(all_pdg).tolist()
    unique.sort(key=lambda x: (abs(int(x)), -int(x)))
    return [int(x) for x in unique]


def pdg_to_label(pdg: np.ndarray, label_list: list[int]) -> np.ndarray:
    lmap = {v: i for i, v in enumerate(label_list)}
    return np.array([lmap[int(x)] for x in pdg], dtype=np.int32)


# ── Process one source file ──────────────────────────────────────────────────

def extract_from_file(
    path: str,
    num_layers: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Return directions, pdg, and per-layer observables for one particle file."""
    with h5py.File(path, "r") as f:
        N = f["pdg"].shape[0]
        directions = f["directions"][:].astype(np.float32)
        pdg = f["pdg"][:].astype(np.int32)

        num_points_all = np.zeros((N, num_layers), dtype=np.int32)
        energy_all = np.zeros((N, num_layers), dtype=np.float32)
        time_all = np.zeros((N, num_layers), dtype=np.float32)

        for start in range(0, N, chunk_size):
            stop = min(N, start + chunk_size)
            num_points_all[start:stop] = calc_num_points_per_layer(
                f["showers"], start, stop, num_layers
            )
            energy_all[start:stop] = calc_energy_per_layer(
                f["showers"], start, stop, num_layers
            )
            time_all[start:stop] = calc_time_per_layer(
                f["showers"], start, stop, num_layers
            )
            if stop % (chunk_size * 10) == 0 or stop == N:
                print(f"  Processed {stop}/{N}")

    return {
        "directions": directions,
        "pdg": pdg,
        "num_points_per_layer": num_points_all,
        "energy_per_layer": energy_all,
        "time_per_layer": time_all,
    }


# ── Main combination logic ───────────────────────────────────────────────────

def combine_files(
    electron_path: str,
    muon_path: str,
    photon_path: str,
    output_path: str,
    num_layers: int = 24,
    chunk_size: int = 5000,
    overwrite: bool = False,
) -> None:
    """Read three particle-type files, compute observables, write combined dataset."""
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace."
        )

    paths = {
        "electron": electron_path,
        "muon": muon_path,
        "photon": photon_path,
    }

    data: dict[str, dict[str, np.ndarray]] = {}
    for ptype, path in paths.items():
        print(f"\nExtracting {ptype} from {path}")
        data[ptype] = extract_from_file(path, num_layers, chunk_size)

    # Collect all PDGs for a unified label list
    pdg_arrays = [data[p]["pdg"] for p in PARTICLE_TYPES]
    label_list = create_label_list(pdg_arrays)
    print(f"\nLabel list (pdg -> label): {dict(zip(label_list, range(len(label_list))))}")

    # Total number of samples
    N = sum(data[p]["pdg"].shape[0] for p in PARTICLE_TYPES)
    print(f"Total samples: {N}")

    # Shuffle index
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)

    # Stack arrays
    directions = np.concatenate([data[p]["directions"] for p in PARTICLE_TYPES], axis=0)
    pdg_raw = np.concatenate([data[p]["pdg"] for p in PARTICLE_TYPES], axis=0)
    labels = pdg_to_label(pdg_raw, label_list)

    # Build per-particle-type condition arrays (zero-padded)
    cond_arrays: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for ptype in PARTICLE_TYPES:
        n_p = data[ptype]["pdg"].shape[0]
        cond_arrays[ptype] = {}
        for obs_key in ["num_points_per_layer", "energy_per_layer", "time_per_layer"]:
            arr = np.zeros((N, num_layers), dtype=np.float32)
            arr[offset : offset + n_p] = data[ptype][obs_key]
            cond_arrays[ptype][obs_key] = arr
        offset += n_p

    # Write output
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as hout:
        hout.attrs["label_list"] = np.array(label_list, dtype=np.int32)
        hout.attrs["num_layers"] = np.int32(num_layers)
        hout.attrs["particle_types"] = PARTICLE_TYPES

        # Targets
        hout.create_dataset("directions", data=directions[perm], compression="gzip")
        hout.create_dataset("pdg", data=pdg_raw[perm], compression="gzip")
        hout.create_dataset("labels", data=labels[perm], compression="gzip")

        # Per-particle-type condition features
        for ptype in PARTICLE_TYPES:
            for obs_key in ["num_points_per_layer", "energy_per_layer", "time_per_layer"]:
                ds_name = f"{obs_key}_{ptype}"
                hout.create_dataset(
                    ds_name,
                    data=cond_arrays[ptype][obs_key][perm],
                    compression="gzip",
                )

    print(f"\nWritten: {output_path}")
    with h5py.File(output_path, "r") as hout:
        print("Datasets:")
        for name in sorted(hout.keys()):
            print(f"  {name}: {hout[name].shape} {hout[name].dtype}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine electron/muon/photon HDF5 files into a reconstruction training dataset."
    )
    parser.add_argument("--electron", required=True, help="Path to electron HDF5 file.")
    parser.add_argument("--muon", required=True, help="Path to muon HDF5 file.")
    parser.add_argument("--photon", required=True, help="Path to photon HDF5 file.")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file.")
    parser.add_argument("--num-layers", type=int, default=24, help="Number of calorimeter layers (default: 24).")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for processing (default: 5000).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = parser.parse_args()

    combine_files(
        electron_path=args.electron,
        muon_path=args.muon,
        photon_path=args.photon,
        output_path=args.output,
        num_layers=args.num_layers,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
