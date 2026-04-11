#!/usr/bin/env python3
"""
Combine h5 files listed in txt files by particle type (electrons, muons, photons).

Reads paths from all pdg_*_<particle>.txt files in BASE_DIR and combines them
into a single merged h5 per particle type.

Outputs (in BASE_DIR):
  - combined_electrons.h5
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

Usage:
  python combine_by_particle.py --particle electrons
  python combine_by_particle.py --particle muons
  python combine_by_particle.py --particle photons
  python combine_by_particle.py --particle all          # runs locally (all 3 sequentially)
  python /n/home04/hhanif/tam/job_submission_scripts/combine_h5_files.py --particle all --slurm  # submits 3 separate Slurm jobs
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

BASE_DIR  = Path("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training")
PARTICLES = ["electrons", "muons", "photons"]

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

def read_txt(txt_path: Path) -> list:
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def get_txt_files(particle: str) -> list:
    pattern = str(BASE_DIR / f"pdg_*_{particle}.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No txt files found matching: {pattern}")
    return [Path(f) for f in files]


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

def submit_slurm_jobs():
    """
    For --particle all --slurm: generate and submit one Slurm batch script
    per particle type so they run in parallel on the cluster.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    job_ids = []
    for particle in PARTICLES:
        job_name   = f"combine_{particle}"
        log_out    = LOG_DIR / f"{job_name}_%j.out"
        log_err    = LOG_DIR / f"{job_name}_%j.err"
        script_path = LOG_DIR / f"submit_{particle}.sh"

        batch_script = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --partition={SLURM_PARTITION}
            #SBATCH --mem={SLURM_MEM}
            #SBATCH --time={SLURM_TIME}
            #SBATCH --cpus-per-task={SLURM_CPUS}
            #SBATCH --output={log_out}
            #SBATCH --error={log_err}

            module load python
            eval "$(mamba shell hook --shell bash)"
            mamba config set changeps1 False
            mamba activate {CONDA_ENV}

            python {THIS_SCRIPT} --particle {particle}
        """)

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
        print(f"\n{len(job_ids)}/{len(PARTICLES)} jobs submitted successfully.")
        print(f"Monitor with:  squeue -j {','.join(job_ids)}")
    return job_ids


# ── Core combine ──────────────────────────────────────────────────────────────

def combine_particle(particle: str):
    print(f"\n{'='*60}")
    print(f"  Combining: {particle}")
    print(f"{'='*60}")

    txt_files = get_txt_files(particle)
    print(f"\nTxt files found ({len(txt_files)}):")

    all_h5_paths = []
    for txt in txt_files:
        paths = read_txt(txt)
        print(f"  {txt.name}: {len(paths)} paths")
        all_h5_paths.extend(paths)

    print(f"\nTotal h5 files        : {len(all_h5_paths):,}")

    print("Counting total showers (parallel)...")
    counts = count_entries_mp(all_h5_paths)
    total_n = sum(counts)
    print(f"Total showers         : {total_n:,}")

    if total_n == 0:
        print(f"  ERROR: No showers found for {particle}, skipping.")
        return

    out_path = BASE_DIR / f"combined_{particle}.h5"
    print(f"Output                : {out_path}")

    vlen_dt = h5py.vlen_dtype(np.dtype("float32"))
    chunk_n = min(4096, total_n)

    with h5py.File(out_path, "w") as hout:
        ds_dir  = hout.create_dataset("directions", shape=(total_n, 3), dtype=np.float32,
                                      compression="gzip", compression_opts=4,
                                      chunks=(chunk_n, 3))
        ds_en   = hout.create_dataset("energies",   shape=(total_n, 1), dtype=np.float32,
                                      compression="gzip", compression_opts=4,
                                      chunks=(chunk_n, 1))
        ds_pdg  = hout.create_dataset("pdg",        shape=(total_n,),   dtype=np.int32,
                                      compression="gzip", compression_opts=4,
                                      chunks=(chunk_n,))
        ds_apdg = hout.create_dataset("actual_pdg", shape=(total_n,),   dtype=np.int32,
                                      compression="gzip", compression_opts=4,
                                      chunks=(chunk_n,))
        ds_ids  = hout.create_dataset("shower_ids", shape=(total_n,),   dtype=np.int32,
                                      compression="gzip", compression_opts=4,
                                      chunks=(chunk_n,))
        ds_npts = hout.create_dataset("num_points", shape=(total_n,),   dtype=np.int32,
                                      compression="gzip", compression_opts=4,
                                      chunks=(chunk_n,))
        ds_shw  = hout.create_dataset("showers",    shape=(total_n,),   dtype=vlen_dt)

        hout.attrs["particle"]       = particle
        hout.attrs["n_simulations"]  = total_n
        hout.attrs["n_source_files"] = len(all_h5_paths)

        global_idx = 0
        n_skipped  = 0

        for path, n_expected in tqdm(zip(all_h5_paths, counts),
                                     total=len(all_h5_paths),
                                     desc=f"Merging {particle}",
                                     unit="file"):
            if n_expected == 0:
                n_skipped += 1
                continue
            try:
                with h5py.File(path, "r") as hf:
                    keys     = set(hf.keys())
                    npts_key = "num_points" if "num_points" in keys else "num_of_points"
                    n        = int(hf["directions"].shape[0])
                    sl       = slice(global_idx, global_idx + n)

                    # ── Read entire arrays at once (vectorized, no row loop) ──
                    ds_dir[sl]  = hf["directions"][:]
                    ds_en[sl]   = hf["energies"][:]
                    ds_pdg[sl]  = hf["pdg"][:]
                    ds_npts[sl] = hf[npts_key][:]

                    # actual_pdg: use dataset if present, else copy from pdg
                    if "actual_pdg" in keys:
                        ds_apdg[sl] = hf["actual_pdg"][:]
                    else:
                        ds_apdg[sl] = hf["pdg"][:]

                    # shower_ids always reassigned sequentially
                    ds_ids[sl]  = np.arange(global_idx, global_idx + n, dtype=np.int32)

                    # showers is vlen — must be written entry by entry
                    shw = hf["showers"]
                    for i in range(n):
                        ds_shw[global_idx + i] = shw[i]

                    global_idx += n

            except Exception as e:
                print(f"\n  WARNING: Error reading {os.path.basename(path)}: {e}, skipping.")
                n_skipped += 1
                continue

        if global_idx != total_n:
            print(f"\n  NOTE: Expected {total_n}, wrote {global_idx} ({n_skipped} files skipped).")
            hout.attrs["n_simulations"] = global_idx

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nDone: {out_path}  ({global_idx:,} showers, {size_mb:.1f} MB)")

    with h5py.File(out_path, "r") as hf:
        print(f"\nDataset shapes in {out_path.name}:")
        for k in ["showers", "directions", "energies", "pdg", "actual_pdg", "shower_ids", "num_points"]:
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
    args = parser.parse_args()

    if args.slurm and args.particle != "all":
        parser.error("--slurm is only valid when --particle all is specified.")

    if args.slurm:
        # ── Slurm path: submit 3 independent jobs ──
        print("Submitting separate Slurm jobs for each particle type...")
        print(f"  Partition : {SLURM_PARTITION}")
        print(f"  Memory    : {SLURM_MEM}")
        print(f"  Time      : {SLURM_TIME}")
        print(f"  CPUs/task : {SLURM_CPUS}")
        print(f"  Conda env : {CONDA_ENV}")
        print(f"  Log dir   : {LOG_DIR}\n")
        submit_slurm_jobs()
    else:
        # ── Local path: run sequentially ──
        targets = PARTICLES if args.particle == "all" else [args.particle]
        for particle in targets:
            combine_particle(particle)
        print("\n\nAll done!")


if __name__ == "__main__":
    main()