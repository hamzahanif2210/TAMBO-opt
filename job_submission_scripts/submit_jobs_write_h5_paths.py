#!/usr/bin/env python3

import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE = Path("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training")

PDG_FOLDERS = ["pdg_-11", "pdg_11", "pdg_111", "pdg_-211", "pdg_211"]
PARTICLES   = ["electrons", "muons", "photons"]

N_WORKERS   = cpu_count()


def find_h5_in_run_dir(run_dir):
    """
    Given a leaf run directory, return all h5 file paths found inside it.
    Called in parallel across all 700k run dirs.
    """
    results = {p: [] for p in PARTICLES}
    try:
        for entry in os.scandir(run_dir):
            if entry.name.endswith(".h5") and entry.is_file():
                for particle in PARTICLES:
                    if entry.name.endswith(f"_{particle}.h5"):
                        results[particle].append(entry.path)
                        break
    except PermissionError:
        pass
    return results


def collect_run_dirs(pdg_dir):
    """
    Walk pdg_dir two levels deep (energy_* / run_*) to collect all leaf run dirs.
    Uses os.scandir for speed instead of rglob.
    """
    run_dirs = []
    try:
        for energy_entry in os.scandir(pdg_dir):
            if not energy_entry.is_dir():
                continue
            try:
                for run_entry in os.scandir(energy_entry.path):
                    if run_entry.is_dir():
                        run_dirs.append(run_entry.path)
            except PermissionError:
                pass
    except PermissionError:
        pass
    return run_dirs


def process_pdg(pdg):
    pdg_dir = BASE / pdg

    if not pdg_dir.is_dir():
        print(f"WARNING: Directory not found: {pdg_dir}")
        return

    print(f"[{pdg}] Collecting run directories...")
    run_dirs = collect_run_dirs(pdg_dir)
    print(f"[{pdg}] Found {len(run_dirs):,} run dirs — scanning in parallel with {N_WORKERS} workers...")

    # Parallel scan across all run dirs
    with Pool(processes=N_WORKERS) as pool:
        all_results = pool.map(find_h5_in_run_dir, run_dirs, chunksize=500)

    # Merge results per particle
    merged = {p: [] for p in PARTICLES}
    for result in all_results:
        for particle in PARTICLES:
            merged[particle].extend(result[particle])

    # Write one txt file per particle
    for particle in PARTICLES:
        h5_files = sorted(merged[particle])
        outfile = BASE / f"{pdg}_{particle}.txt"
        with open(outfile, "w") as f:
            f.write("\n".join(h5_files))
            if h5_files:
                f.write("\n")
        print(f"[{pdg}] Written: {outfile}  ({len(h5_files):,} files)")


if __name__ == "__main__":
    print(f"Using {N_WORKERS} workers\n")
    for pdg in PDG_FOLDERS:
        process_pdg(pdg)
    print("\nDone! 15 txt files written to:", BASE)