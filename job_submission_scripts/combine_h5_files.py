#!/usr/bin/env python3
"""
Combine h5 files listed in txt files by particle type (electrons, muons, photons).

Reads paths from all pdg_*_<particle>.txt files in BASE_DIR and combines them
into a single merged h5 per particle type.

Outputs (in BASE_DIR):
  - combined_electrons.h5         (train set, balanced by actual_pdg)
  - combined_electrons_test.h5    (remainder, if --save-test-also / --save-test-only)
  - combined_muons.h5
  - combined_photons.h5

Datasets written:
  - showers     : (N,) variable-length float32
  - directions  : (N, 3) float32
  - energies    : (N, 1) float32
  - pdg         : (N,) int32
  - actual_pdg  : (N,) int32
  - num_points  : (N,) int32
  - shower_ids  : (N,) int32  (reassigned sequentially 0..N-1)

PDG balancing (--train-limit, default 130000):
  Group A (charged pions) : actual_pdg in {211, -211}  → up to train_limit // 2 showers
  Group B (EM)            : actual_pdg in {11, -11, 111} → up to train_limit // 2 showers
  Remaining showers are written to combined_<particle>_test.h5 when
  --save-test-also or --save-test-only is passed.

Usage:
  python combine_by_particle.py --particle electrons
  python combine_by_particle.py --particle electrons --train-limit 130000 --save-test-also
  python combine_by_particle.py --particle electrons --train-limit 130000 --save-test-only
  python combine_by_particle.py --particle all
  python /n/home04/hhanif/TAMBO-opt/job_submission_scripts/combine_h5_files.py --particle electrons  --train-limit 130000 --save-test-also --slurm
"""

import os
import glob
import argparse
import subprocess
import textwrap
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import h5py
from tqdm.auto import tqdm
import random

BASE_DIR  = Path("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files")
PARTICLES = ["electrons", "muons", "photons"]

# PDG groups for balanced train/test splitting
PDG_GROUP_PION = {211, -211}          # charged pions  → 65k
PDG_GROUP_EM   = {11, -11, 111}       # e± / π⁰        → 65k
DEFAULT_TRAIN_LIMIT = 130_000         # 65k per group

# Path to this script (used when generating Slurm job scripts)
THIS_SCRIPT = Path(__file__).resolve()

# Slurm / environment settings
SLURM_PARTITION   = "arguelles_delgado,shared,sapphire"
SLURM_MEM         = "64G"
SLURM_TIME        = "0-24:00"
SLURM_CPUS        = 48
CONDA_ENV         = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/"
LOG_DIR           = BASE_DIR / "slurm_logs"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_chunk_files(particle: str) -> list:
    """
    Glob all chunk H5 files for a given particle type directly from PDG subdirs.
    Structure: BASE_DIR/pdg_<N>/chunk_<XXXX>_<particle>.h5
    """
    pattern = str(BASE_DIR / f"pdg_*" / f"chunk_*_{particle}.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No chunk h5 files found matching: {pattern}\n"
            f"Expected structure: {BASE_DIR}/pdg_<N>/chunk_<XXXX>_{particle}.h5"
        )
    return files


def count_entries(path: str) -> int:
    try:
        with h5py.File(path, "r") as hf:
            return int(hf["directions"].shape[0])
    except Exception:
        return 0


def count_entries_mp(paths: list) -> list:
    with Pool(processes=min(cpu_count(), len(paths))) as pool:
        return pool.map(count_entries, paths, chunksize=500)


# ── Slurm submission ──────────────────────────────────────────────────────────

def submit_slurm_jobs(particles: list, train_limit: int, save_test_also: bool, save_test_only: bool):
    """
    For --particle all --slurm: generate and submit one Slurm batch script
    per particle type so they run in parallel on the cluster.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    job_ids = []
    for particle in particles:
        job_name    = f"combine_{particle}"
        log_out     = LOG_DIR / f"{job_name}_%j.out"
        log_err     = LOG_DIR / f"{job_name}_%j.err"
        script_path = LOG_DIR / f"submit_{particle}.sh"

        py_cmd = f"python {THIS_SCRIPT} --particle {particle} --train-limit {train_limit}"
        if save_test_also:
            py_cmd += " --save-test-also"
        if save_test_only:
            py_cmd += " --save-test-only"

        batch_script = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --partition={SLURM_PARTITION}\n"
            f"#SBATCH --mem={SLURM_MEM}\n"
            f"#SBATCH --time={SLURM_TIME}\n"
            f"#SBATCH --cpus-per-task={SLURM_CPUS}\n"
            f"#SBATCH --output={log_out}\n"
            f"#SBATCH --error={log_err}\n"
            f"\n"
            f"module load python\n"
            f'eval \"$(mamba shell hook --shell bash)\"\n'
            f"mamba config set changeps1 False\n"
            f"mamba activate {CONDA_ENV}\n"
            f"\n"
            f"{py_cmd}\n"
        )

        script_path.write_text(batch_script)
        script_path.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"  Submitted {particle:<12} → job {job_id}  (log: {LOG_DIR}/{job_name}_<jobid>.out)")
        else:
            print(f"  ERROR submitting {particle}: {result.stderr.strip()}")

    if job_ids:
        print(f"\n{len(job_ids)}/{len(particles)} jobs submitted successfully.")
        print(f"Monitor with:  squeue -j {','.join(job_ids)}")
    return job_ids


# ── Core combine ──────────────────────────────────────────────────────────────

def _read_all_actual_pdg(all_h5_paths: list, counts: list) -> list:
    """
    First pass: read actual_pdg from every source file and return a flat list of
    (file_index, row_within_file) tuples, one per shower.
    """
    index = []   # list of (file_idx, local_row, actual_pdg_val)
    for file_idx, (path, n) in enumerate(zip(all_h5_paths, counts)):
        if n == 0:
            continue
        try:
            with h5py.File(path, "r") as hf:
                keys = set(hf.keys())
                if "actual_pdg" in keys:
                    apdg = hf["actual_pdg"][:].astype(np.int32)
                else:
                    apdg = hf["pdg"][:].astype(np.int32)
            for row, val in enumerate(apdg):
                index.append((file_idx, row, int(val)))
        except Exception as e:
            print(f"  WARNING: could not read actual_pdg from {os.path.basename(path)}: {e}")
    return index


def _build_train_test_indices(
    index: list,
    train_limit: int,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Split flat index into train and test index lists.

    Train set:
      - up to train_limit // 2 showers from PDG_GROUP_PION  (±211)
      - up to train_limit // 2 showers from PDG_GROUP_EM    (±11, 111)
      Randomly sampled if more are available; reproducible via seed.

    Test set:
      - all remaining showers not selected for train.
    """
    rng = random.Random(seed)

    pion_idx = [(fi, row) for fi, row, v in index if v in PDG_GROUP_PION]
    em_idx   = [(fi, row) for fi, row, v in index if v in PDG_GROUP_EM]
    other_idx = [(fi, row) for fi, row, v in index
                 if v not in PDG_GROUP_PION and v not in PDG_GROUP_EM]

    per_group = train_limit // 2

    rng.shuffle(pion_idx)
    rng.shuffle(em_idx)

    train_pion = pion_idx[:per_group]
    train_em   = em_idx[:per_group]
    test_pion  = pion_idx[per_group:]
    test_em    = em_idx[per_group:]

    train_set = set(map(tuple, train_pion + train_em))

    # preserve original file order for sequential reads
    all_pairs   = [(fi, row) for fi, row, _ in index]
    train_index = [p for p in all_pairs if p in train_set]
    test_index  = [p for p in all_pairs if p not in train_set] + other_idx

    print(f"\n  PDG group sizes (all data):")
    print(f"    ±211  (pion) : {len(pion_idx):>10,}")
    print(f"    ±11/111 (EM) : {len(em_idx):>10,}")
    print(f"    other        : {len(other_idx):>10,}")
    print(f"\n  Train set      : {len(train_index):>10,}  "
          f"({len(train_pion):,} pion + {len(train_em):,} EM)")
    print(f"  Test  set      : {len(test_index):>10,}")

    return train_index, test_index


def _create_datasets(hf, n: int, chunk_n: int) -> dict:
    """Create all standard datasets in an open H5 file and return them by name."""
    vlen_dt = h5py.vlen_dtype(np.dtype("float32"))
    return {
        "directions": hf.create_dataset("directions", shape=(n, 3), maxshape=(None, 3),
                                         dtype=np.float32, compression="gzip",
                                         compression_opts=4, chunks=(chunk_n, 3)),
        "energies":   hf.create_dataset("energies",   shape=(n, 1), maxshape=(None, 1),
                                         dtype=np.float32, compression="gzip",
                                         compression_opts=4, chunks=(chunk_n, 1)),
        "pdg":        hf.create_dataset("pdg",        shape=(n,),   maxshape=(None,),
                                         dtype=np.int32,   compression="gzip",
                                         compression_opts=4, chunks=(chunk_n,)),
        "actual_pdg": hf.create_dataset("actual_pdg", shape=(n,),   maxshape=(None,),
                                         dtype=np.int32,   compression="gzip",
                                         compression_opts=4, chunks=(chunk_n,)),
        "shower_ids": hf.create_dataset("shower_ids", shape=(n,),   maxshape=(None,),
                                         dtype=np.int32,   compression="gzip",
                                         compression_opts=4, chunks=(chunk_n,)),
        "num_points": hf.create_dataset("num_points", shape=(n,),   maxshape=(None,),
                                         dtype=np.int32,   compression="gzip",
                                         compression_opts=4, chunks=(chunk_n,)),
        "showers":    hf.create_dataset("showers",    shape=(n,),   maxshape=(None,),
                                         dtype=vlen_dt),
    }


def _write_selected(
    all_h5_paths: list,
    counts: list,
    selected_index: list,
    out_path: Path,
    particle: str,
    label: str,
    n_source_files: int,
) -> int:
    """
    Write the showers identified by selected_index (list of (file_idx, row) pairs)
    into out_path. Returns number of showers written.
    """
    n_total = len(selected_index)
    if n_total == 0:
        print(f"  No showers for {label}, skipping.")
        return 0

    chunk_n = min(4096, n_total)

    # Build a lookup: file_idx -> sorted list of local rows needed
    from collections import defaultdict
    rows_needed: dict[int, list] = defaultdict(list)
    for fi, row in selected_index:
        rows_needed[fi].append(row)
    for fi in rows_needed:
        rows_needed[fi].sort()

    # Precompute global write position for each (fi, row) pair
    # We iterate in the order of selected_index to preserve file ordering
    write_order = selected_index  # already in file order from _build_train_test_indices

    print(f"\n  Writing {label}: {out_path.name}  ({n_total:,} showers)")

    global_idx = 0
    n_skipped  = 0

    with h5py.File(out_path, "w") as hout:
        ds = _create_datasets(hout, n_total, chunk_n)
        hout.attrs["particle"]       = particle
        hout.attrs["n_simulations"]  = n_total
        hout.attrs["n_source_files"] = n_source_files
        hout.attrs["split"]          = label

        # Group consecutive rows from same file for efficient reading
        from itertools import groupby
        for fi, group in tqdm(
            groupby(write_order, key=lambda x: x[0]),
            desc=f"  {label}",
            unit="file",
        ):
            rows = [row for _, row in group]
            path = all_h5_paths[fi]
            try:
                with h5py.File(path, "r") as hf:
                    keys     = set(hf.keys())
                    npts_key = "num_points" if "num_points" in keys else "num_of_points"

                    rows_arr = np.array(rows, dtype=np.intp)
                    n        = len(rows_arr)
                    sl       = slice(global_idx, global_idx + n)

                    ds["directions"][sl] = hf["directions"][rows_arr]

                    en = hf["energies"][rows_arr]
                    ds["energies"][sl]   = en.reshape(-1, 1) if en.ndim == 1 else en

                    ds["pdg"][sl]        = hf["pdg"][rows_arr]
                    ds["num_points"][sl] = hf[npts_key][rows_arr].astype(np.int32)

                    if "actual_pdg" in keys:
                        ds["actual_pdg"][sl] = hf["actual_pdg"][rows_arr]
                    else:
                        ds["actual_pdg"][sl] = hf["pdg"][rows_arr]

                    ds["shower_ids"][sl] = np.arange(global_idx, global_idx + n, dtype=np.int32)

                    shw_data = hf["showers"][:]
                    for i, row in enumerate(rows_arr):
                        ds["showers"][global_idx + i] = shw_data[row]

                    global_idx += n

            except Exception as e:
                print(f"\n  WARNING: Error reading {os.path.basename(path)}: {e}, skipping.")
                n_skipped += 1
                continue

        if global_idx != n_total:
            hout.attrs["n_simulations"] = global_idx
            for k, shape in [("directions", (global_idx, 3)), ("energies", (global_idx, 1)),
                              ("pdg", (global_idx,)), ("actual_pdg", (global_idx,)),
                              ("shower_ids", (global_idx,)), ("num_points", (global_idx,)),
                              ("showers", (global_idx,))]:
                hout[k].resize(shape)

        # read nmax and n_feat from ALL source chunk files, take the max
        nmax_val, n_feat_val = -1, -1
        for path in all_h5_paths:
            try:
                with h5py.File(path, "r") as hf:
                    nmax_val   = max(nmax_val,   int(hf.attrs.get("nmax",   -1)))
                    n_feat_val = max(n_feat_val, int(hf.attrs.get("n_feat", -1)))
            except Exception:
                continue

        hout.create_dataset(
            "shape",
            data=np.array([global_idx, nmax_val, n_feat_val], dtype=np.int32),
            dtype=np.int32,
        )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Done: {out_path.name}  ({global_idx:,} showers, {size_mb:.1f} MB)")
    print(f"  shape dataset  : [{global_idx}, {nmax_val}, {n_feat_val}]")
    return global_idx


def combine_particle(particle: str, train_limit: int = DEFAULT_TRAIN_LIMIT,
                     save_test_also: bool = False, save_test_only: bool = False):
    print(f"\n{'='*60}")
    print(f"  Combining: {particle}  (train_limit={train_limit:,})")
    print(f"{'='*60}")

    all_h5_paths = get_chunk_files(particle)

    # print breakdown per PDG subdir
    from collections import defaultdict
    per_pdg: dict = defaultdict(int)
    for p in all_h5_paths:
        pdg_dir = Path(p).parent.name   # e.g. pdg_-11
        per_pdg[pdg_dir] += 1
    print(f"\nChunk files found ({len(all_h5_paths):,} total):")
    for pdg_dir, n in sorted(per_pdg.items()):
        print(f"  {pdg_dir}: {n} files")

    print(f"\nTotal h5 files        : {len(all_h5_paths):,}")

    print("Counting total showers (parallel)...")
    counts = count_entries_mp(all_h5_paths)
    total_n = sum(counts)
    print(f"Total showers         : {total_n:,}")

    if total_n == 0:
        print(f"  ERROR: No showers found for {particle}, skipping.")
        return

    # ── First pass: read all actual_pdg to build balanced train/test split ──
    print("\nFirst pass: reading actual_pdg values for all showers...")
    flat_index = _read_all_actual_pdg(all_h5_paths, counts)
    train_index, test_index = _build_train_test_indices(flat_index, train_limit)

    # ── Second pass: write selected showers ──
    if not save_test_only:
        out_path = BASE_DIR / f"combined_{particle}.h5"
        _write_selected(all_h5_paths, counts, train_index, out_path,
                        particle, label="train", n_source_files=len(all_h5_paths))
        with h5py.File(out_path, "r") as hf:
            print(f"\nDataset shapes in {out_path.name}:")
            for k in ["showers", "directions", "energies", "pdg", "actual_pdg", "shower_ids", "num_points", "shape"]:
                print(f"  {k:16s}: {hf[k].shape}")

    if save_test_also or save_test_only:
        test_path = BASE_DIR / f"combined_{particle}_test.h5"
        _write_selected(all_h5_paths, counts, test_index, test_path,
                        particle, label="test", n_source_files=len(all_h5_paths))
        with h5py.File(test_path, "r") as hf:
            print(f"\nDataset shapes in {test_path.name}:")
            for k in ["showers", "directions", "energies", "pdg", "actual_pdg", "shower_ids", "num_points", "shape"]:
                print(f"  {k:16s}: {hf[k].shape}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine h5 files by particle type from txt file lists."
    )
    parser.add_argument(
        "--particle",
        choices=["electrons", "muons", "photons", "all"],
        required=True,
        help="Which particle type to combine. 'all' runs all three separately."
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        default=False,
        help=(
            "Only valid with --particle all. "
            "Submit one separate Slurm job per particle instead of running locally."
        ),
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=DEFAULT_TRAIN_LIMIT,
        help=f"Total number of showers in the train file. Split evenly between "
             f"pion group (±211) and EM group (±11/111). Default: {DEFAULT_TRAIN_LIMIT:,}."
    )
    parser.add_argument(
        "--save-test-also",
        action="store_true",
        default=False,
        help="Write remaining showers (not in train set) to combined_<particle>_test.h5 "
             "in addition to the train file."
    )
    parser.add_argument(
        "--save-test-only",
        action="store_true",
        default=False,
        help="Write ONLY the test file (combined_<particle>_test.h5), skip train file."
    )
    args = parser.parse_args()

    if args.save_test_also and args.save_test_only:
        parser.error("--save-test-also and --save-test-only are mutually exclusive.")

    if args.slurm:
        # ── Slurm path: submit one job per target particle ──
        targets = PARTICLES if args.particle == "all" else [args.particle]
        print("Submitting Slurm jobs for:", ", ".join(targets))
        print(f"  Partition  : {SLURM_PARTITION}")
        print(f"  Memory     : {SLURM_MEM}")
        print(f"  Time       : {SLURM_TIME}")
        print(f"  CPUs/task  : {SLURM_CPUS}")
        print(f"  Conda env  : {CONDA_ENV}")
        print(f"  Log dir    : {LOG_DIR}\n")
        submit_slurm_jobs(
            particles      = targets,
            train_limit    = args.train_limit,
            save_test_also = args.save_test_also,
            save_test_only = args.save_test_only,
        )
    else:
        # ── Local path: run sequentially ──
        targets = PARTICLES if args.particle == "all" else [args.particle]
        for particle in targets:
            combine_particle(
                particle,
                train_limit    = args.train_limit,
                save_test_also = args.save_test_also,
                save_test_only = args.save_test_only,
            )
        print("\n\nAll done!")


if __name__ == "__main__":
    main()