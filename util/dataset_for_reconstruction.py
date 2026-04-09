#!/usr/bin/env python3
"""
dataset_for_reconstruction_mp.py

Multiprocessing version of dataset_for_reconstruction.py.

Two levels of parallelism:
  1. File-level  – electron, muon, photon files are extracted concurrently
                   (each worker opens its own h5py handle).
  2. Chunk-level – within each file, shower chunks are dispatched to a pool;
                   each worker computes all three observables in a single pass,
                   avoiding three separate loops over the same data.

Takes three HDF5 files (electron, muon, photon) and creates a combined HDF5
file for training the reconstruction model.

From each file it extracts:
  - directions  (N, 3)
  - pdg         (N,)
  - energies    (N,)   [primary particle energy]

And calculates per-layer observables:
  - num_points_per_layer  (N, num_layers)
  - energy_per_layer      (N, num_layers)
  - time_per_layer        (N, num_layers)   [average time per layer; time = feature index 4]

The output file is grouped by direction AND energy. For each unique
(direction, energy) pair, only the particle types that actually exist in the
input files have non-zero observables; missing particle types are zero.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np


# ── Particle-type mapping ────────────────────────────────────────────────────
PARTICLE_TYPES = ["electron", "muon", "photon"]


# ── Per-shower worker (runs in a subprocess) ─────────────────────────────────

def _process_chunk(args: tuple) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function: opens the HDF5 file itself and reads only its slice,
    avoiding the need to pre-load all shower data into the main process.

    Parameters
    ----------
    args : (file_path, chunk_start, chunk_stop, num_layers)

    Returns
    -------
    (chunk_start, num_points, energy, time)  – arrays shaped (chunk_len, num_layers)
    """
    file_path, chunk_start, chunk_stop, num_layers = args
    n = chunk_stop - chunk_start

    num_points = np.zeros((n, num_layers), dtype=np.int32)
    energy_arr = np.zeros((n, num_layers), dtype=np.float32)
    time_sum   = np.zeros((n, num_layers), dtype=np.float64)
    count      = np.zeros((n, num_layers), dtype=np.int32)

    with h5py.File(file_path, "r") as f:
        for i, gi in enumerate(range(chunk_start, chunk_stop)):
            pts   = np.array(f["showers"][gi]).reshape(-1, 5)
            layer = np.clip((pts[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
            pos   = pts[:, 3] > 0

            np.add.at(num_points[i], layer, pos.astype(np.int32))

            e = pts[:, 3] * pos.astype(np.float32)
            np.add.at(energy_arr[i], layer, e)

            np.add.at(time_sum[i], layer[pos], pts[pos, 4])
            np.add.at(count[i],    layer[pos], 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        time_avg = np.where(count > 0, time_sum / count, 0.0).astype(np.float32)

    return chunk_start, num_points, energy_arr, time_avg


# ── Per-layer observable computation ─────────────────────────────────────────
# These thin wrappers are kept for API compatibility but are no longer called
# directly in the hot path; _process_chunk handles everything in one pass.

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


# ── Process one source file (chunk-level MP) ─────────────────────────────────

def _extract_one_file(args: tuple) -> tuple[str, dict[str, np.ndarray]]:
    """
    Top-level worker for file-level parallelism.
    Opens its own h5py handle and uses an inner ProcessPoolExecutor for chunks.
    Returns (ptype, data_dict).
    """
    ptype, path, num_layers, chunk_size, num_workers = args
    result = extract_from_file(path, num_layers, chunk_size, num_workers=num_workers)
    return ptype, result


def extract_from_file(
    path: str,
    num_layers: int,
    chunk_size: int,
    num_workers: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Return directions, pdg, energies, and per-layer observables for one
    particle file.  Shower chunks are processed in parallel.
    """
    with h5py.File(path, "r") as f:
        N          = f["pdg"].shape[0]
        directions = f["directions"][:].astype(np.float32)
        pdg        = f["pdg"][:].astype(np.int32)
        energies   = f["energies"][:].astype(np.float32)

    # Build chunk argument list — workers open the file themselves.
    chunks = [
        (path, start, min(N, start + chunk_size), num_layers)
        for start in range(0, N, chunk_size)
    ]
    print(f"  Dispatching {len(chunks)} chunks ({N} showers) to {num_workers or 'all'} workers...")

    num_points_all = np.zeros((N, num_layers), dtype=np.int32)
    energy_all     = np.zeros((N, num_layers), dtype=np.float32)
    time_all       = np.zeros((N, num_layers), dtype=np.float32)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_chunk, c): c[1] for c in chunks}
        completed = 0
        for fut in as_completed(futures):
            chunk_start, num_pts, energy, time_avg = fut.result()
            chunk_len = num_pts.shape[0]
            num_points_all[chunk_start : chunk_start + chunk_len] = num_pts
            energy_all    [chunk_start : chunk_start + chunk_len] = energy
            time_all      [chunk_start : chunk_start + chunk_len] = time_avg
            completed += chunk_len
            if completed % (chunk_size * 10) == 0 or completed >= N:
                print(f"  Processed {completed}/{N}")

    return {
        "directions":           directions,
        "pdg":                  pdg,
        "energies":             energies,
        "num_points_per_layer": num_points_all,
        "energy_per_layer":     energy_all,
        "time_per_layer":       time_all,
    }


# ── Align by direction + energy ──────────────────────────────────────────────

def directions_to_key(directions: np.ndarray) -> list[tuple]:
    """Convert direction vectors to hashable tuple keys for matching."""
    return [tuple(d.tolist()) for d in directions]


def align_by_direction(
    data: dict[str, dict[str, np.ndarray]],
    num_layers: int,
) -> dict[str, np.ndarray]:
    """
    For each unique (direction, energy, pdg) triple, emit ONE output row with
    observables from all three particle types side by side. Missing particle
    types get zeros. Direction and energy are taken from whichever particle
    type is present for that key.
    """
    # Build a lookup: (direction_key, energy, pdg) -> {ptype: row_index}
    dir_to_rows: dict[tuple, dict[str, int]] = {}
    for ptype in PARTICLE_TYPES:
        dir_keys = directions_to_key(data[ptype]["directions"])
        energies = data[ptype]["energies"]
        pdgs     = data[ptype]["pdg"]
        for row_idx, (dir_key, energy, pdg) in enumerate(zip(dir_keys, energies, pdgs)):
            key = (dir_key, float(energy), int(pdg))
            if key not in dir_to_rows:
                dir_to_rows[key] = {}
            dir_to_rows[key][ptype] = row_idx

    out_directions  = []
    out_pdg         = []
    out_energies    = []
    out_observables = {
        ptype: {obs: [] for obs in ["num_points_per_layer", "energy_per_layer", "time_per_layer"]}
        for ptype in PARTICLE_TYPES
    }

    for (dir_key, energy_val, pdg_value), ptype_rows in dir_to_rows.items():
        # Use direction/energy from the first available particle type
        ref_ptype = next(iter(ptype_rows))
        ref_idx   = ptype_rows[ref_ptype]
        out_directions.append(data[ref_ptype]["directions"][ref_idx])
        out_pdg.append(pdg_value)
        out_energies.append(data[ref_ptype]["energies"][ref_idx])

        # One row: fill each particle type's observables (zeros if missing)
        for ptype in PARTICLE_TYPES:
            for obs in ["num_points_per_layer", "energy_per_layer", "time_per_layer"]:
                if ptype in ptype_rows:
                    obs_row = data[ptype][obs][ptype_rows[ptype]]
                else:
                    obs_row = np.zeros(num_layers, dtype=np.float32)
                out_observables[ptype][obs].append(obs_row)

    # Stack into arrays
    result: dict[str, np.ndarray] = {
        "directions": np.stack(out_directions).astype(np.float32),
        "pdg":        np.array(out_pdg,      dtype=np.int32),
        "energies":   np.array(out_energies, dtype=np.float32),
    }
    for ptype in PARTICLE_TYPES:
        for obs in ["num_points_per_layer", "energy_per_layer", "time_per_layer"]:
            result[f"{obs}_{ptype}"] = np.stack(out_observables[ptype][obs]).astype(np.float32)

    return result


# ── Main combination logic ───────────────────────────────────────────────────

def combine_files(
    electron_path: str,
    muon_path: str,
    photon_path: str,
    output_path: str,
    num_layers: int = 24,
    chunk_size: int = 5000,
    overwrite: bool = False,
    num_workers: int | None = None,
) -> None:
    """
    Read three particle-type files, compute observables, write combined dataset.

    Parameters
    ----------
    num_workers : int or None
        Number of worker processes for chunk-level parallelism within each
        file.  None → use all available CPUs.  The three files are always
        extracted concurrently (file-level parallelism), so the effective
        process count is up to 3 × num_workers during extraction.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace."
        )

    paths = {
        "electron": electron_path,
        "muon":     muon_path,
        "photon":   photon_path,
    }

    # ── File-level parallelism: extract all three files concurrently ──────────
    data: dict[str, dict[str, np.ndarray]] = {}
    file_args = [
        (ptype, path, num_layers, chunk_size, num_workers)
        for ptype, path in paths.items()
    ]

    print("Extracting all particle files in parallel...")
    with ProcessPoolExecutor(max_workers=len(PARTICLE_TYPES)) as pool:
        futures = {pool.submit(_extract_one_file, arg): arg[0] for arg in file_args}
        for fut in as_completed(futures):
            ptype, result = fut.result()
            data[ptype] = result
            print(f"  Finished extracting {ptype}")

    # ── Align all data by direction + energy ──────────────────────────────────
    print("\nAligning by direction, energy, and pdg...")
    aligned = align_by_direction(data, num_layers)

    N = aligned["directions"].shape[0]
    print(f"Total aligned samples: {N}")

    # Label list from aligned pdg
    label_list = create_label_list([aligned["pdg"]])
    print(f"Label list (pdg -> label): {dict(zip(label_list, range(len(label_list))))}")
    labels = pdg_to_label(aligned["pdg"], label_list)

    # Shuffle
    rng  = np.random.default_rng(42)
    perm = rng.permutation(N)

    # Write output
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as hout:
        hout.attrs["label_list"]     = np.array(label_list, dtype=np.int32)
        hout.attrs["num_layers"]     = np.int32(num_layers)
        hout.attrs["particle_types"] = PARTICLE_TYPES

        # Targets
        hout.create_dataset("directions", data=aligned["directions"][perm], compression="gzip")
        hout.create_dataset("pdg",        data=aligned["pdg"][perm],        compression="gzip")
        hout.create_dataset("labels",     data=labels[perm],                compression="gzip")
        hout.create_dataset("energies",   data=aligned["energies"][perm],   compression="gzip")

        # Per-particle-type condition features
        for ptype in PARTICLE_TYPES:
            for obs in ["num_points_per_layer", "energy_per_layer", "time_per_layer"]:
                ds_name = f"{obs}_{ptype}"
                hout.create_dataset(
                    ds_name,
                    data=aligned[ds_name][perm],
                    compression="gzip",
                )

    print(f"\nWritten: {output_path}")
    with h5py.File(output_path, "r") as hout:
        print("Datasets:")
        for name in sorted(hout.keys()):
            print(f"  {name}: {hout[name].shape} {hout[name].dtype}")


# ── Config ───────────────────────────────────────────────────────────────────

ELECTRON_PATH = "/path/to/electron.h5"
MUON_PATH     = "/path/to/muon.h5"
PHOTON_PATH   = "/path/to/photon.h5"
OUTPUT_PATH   = "/path/to/output.h5"
NUM_LAYERS    = 24
CHUNK_SIZE    = 5000
OVERWRITE     = True
NUM_WORKERS   = None   # None → all CPUs; set e.g. 16 to cap worker count


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build reconstruction dataset from particle HDF5 files.")
    parser.add_argument("--electron-path", default=ELECTRON_PATH)
    parser.add_argument("--muon-path",     default=MUON_PATH)
    parser.add_argument("--photon-path",   default=PHOTON_PATH)
    parser.add_argument("--output-path",   default=OUTPUT_PATH)
    parser.add_argument("--num-layers",    type=int, default=NUM_LAYERS)
    parser.add_argument("--chunk-size",    type=int, default=CHUNK_SIZE)
    parser.add_argument("--num-workers",   type=int, default=NUM_WORKERS,
                        help="Worker processes per file (None → all CPUs).")
    parser.add_argument("--overwrite",     action="store_true", default=OVERWRITE)
    args = parser.parse_args()

    combine_files(
        electron_path=args.electron_path,
        muon_path=args.muon_path,
        photon_path=args.photon_path,
        output_path=args.output_path,
        num_layers=args.num_layers,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
    )