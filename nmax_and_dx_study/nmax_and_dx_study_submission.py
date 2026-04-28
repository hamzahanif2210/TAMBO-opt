#!/usr/bin/env python3
"""
Submit a SLURM job array to run postprocess_showers_csv.py on multiple PDGs.

Plan
----
For each incident PDG in --pdgs (default: 111 -11 11 -211 211):
    1. Read its registry file:  <registry-dir>/registry_pdg_<PDG>.txt
    2. Draw a random subsample of size --total-per-pdg (without replacement,
       seeded for reproducibility).
    3. Split that subsample into --chunks-per-pdg sublists of equal size and
       write them as per-chunk path-list files into <out>/_chunks/.

Then submit ONE SLURM array of size (n_pdgs * chunks_per_pdg) where each
task processes one (pdg, chunk) pair via postprocess_showers_csv.py.

This guarantees:
    - Exactly --total-per-pdg unique showers per PDG (no overlap across
      chunks of the same PDG).
    - Reproducible draws given --seed.
    - Independent CSV per (pdg, chunk) under <out>/pdg_<PDG>/.

Usage
-----
python /n/home04/hhanif/TAMBO-opt/nmax_and_dx_study/nmax_and_dx_study_submission.py \
    --registry-dir /n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_samples_for_training/  \
    --output-dir   /n/home04/hhanif/TAMBO-opt/results/nmax_study \
    --total-per-pdg 10000 --chunks-per-pdg 20 \
    --dx-sweep 5 8 10 --nmax-sweep 2000 4000 6000 10000 14000 18000 20000 --seed 42 \
    --worker-script /n/home04/hhanif/TAMBO-opt/nmax_and_dx_study/nmax_and_dx_study.py

Add --dry-run to just write chunk files and emit the sbatch script without
submitting it.

After all jobs finish, concatenate per-pdg CSVs with:
    cd <out>/pdg_<PDG> && head -n 1 chunk_0000_energy_ignored.csv > all.csv \\
        && tail -n +2 -q chunk_*_energy_ignored.csv >> all.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np


DEFAULT_PDGS = [111, -11, 11, -211, 211]


def read_registry(path: Path) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def prepare_chunks(
    registry_dir: Path,
    out_dir: Path,
    pdgs: list[int],
    total_per_pdg: int,
    chunks_per_pdg: int,
    seed: int,
) -> dict[int, list[Path]]:
    """
    For each PDG: draw `total_per_pdg` random unique paths from the registry,
    split into `chunks_per_pdg` files, return {pdg: [chunk_file_paths]}.

    Per-PDG seed = master_seed XOR pdg, so different runs with the same
    master seed are reproducible AND different PDGs get independent draws.
    """
    chunk_dir = out_dir / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    plan: dict[int, list[Path]] = {}
    for pdg in pdgs:
        reg_path = registry_dir / f"registry_pdg_{pdg}.txt"
        if not reg_path.is_file():
            print(f"ERROR: registry not found: {reg_path}", file=sys.stderr)
            sys.exit(1)

        all_paths = read_registry(reg_path)
        if len(all_paths) < total_per_pdg:
            print(f"WARNING [pdg={pdg}]: registry has {len(all_paths)} entries "
                  f"but --total-per-pdg={total_per_pdg}. Using all of them.",
                  file=sys.stderr)
            n_take = len(all_paths)
        else:
            n_take = total_per_pdg

        # Per-PDG seed: deterministic but distinct per pdg.
        # Use unsigned 32-bit space; XOR on a positive recoding of pdg.
        pdg_seed = (int(seed) ^ (abs(int(pdg)) * 1_000_003)) & 0xFFFFFFFF
        rng = np.random.default_rng(pdg_seed)

        idx = rng.choice(len(all_paths), size=n_take, replace=False)
        idx.sort()
        sample = [all_paths[i] for i in idx]

        # Split as evenly as possible into chunks_per_pdg parts.
        # np.array_split handles uneven sizes (early chunks get +1 entry).
        splits = np.array_split(np.arange(len(sample)), chunks_per_pdg)
        chunk_files: list[Path] = []
        for chunk_id, idx_block in enumerate(splits):
            cf = chunk_dir / f"pdg_{pdg}_chunk_{chunk_id:04d}.txt"
            with cf.open("w") as f:
                f.write(f"# pdg={pdg}  chunk_id={chunk_id}  "
                        f"n={len(idx_block)}  seed={pdg_seed}\n")
                for k in idx_block:
                    f.write(sample[int(k)] + "\n")
            chunk_files.append(cf)

        plan[pdg] = chunk_files
        print(f"  pdg={pdg:>4}: registry={len(all_paths)}, "
              f"sampled={n_take}, chunks={len(chunk_files)} "
              f"(sizes: {[len(b) for b in splits]})")
    return plan


def build_index_table(plan: dict[int, list[Path]]) -> list[tuple[int, int, str]]:
    """Flat list of (task_id, pdg, chunk_id, chunk_file_path) entries."""
    rows = []
    for pdg, chunk_files in plan.items():
        for chunk_id, cf in enumerate(chunk_files):
            rows.append((pdg, chunk_id, str(cf)))
    return rows


SBATCH_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=energy_ignored
#SBATCH --output={log_dir}/slurm_%A_%a.out
#SBATCH --error={log_dir}/slurm_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{last_idx}
#SBATCH -p serial_requeue

set -euo pipefail

# ---- environment ----
module load python/3.12.11-fasrc01
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

# ---- task → (pdg, chunk_id, chunk_file) ----
INDEX_FILE="{index_file}"
read -r PDG CHUNK_ID CHUNK_FILE < <(awk -v t=$SLURM_ARRAY_TASK_ID '$1==t {{print $2, $3, $4}}' "$INDEX_FILE")

if [[ -z "${{PDG:-}}" ]]; then
    echo "ERROR: no row in $INDEX_FILE for task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi

echo "host       : $(hostname)"
echo "task id    : $SLURM_ARRAY_TASK_ID"
echo "pdg        : $PDG"
echo "chunk_id   : $CHUNK_ID"
echo "chunk_file : $CHUNK_FILE"
echo "started    : $(date -Iseconds)"

# ---- run the postprocessor ----
# NOTE: --random is NOT passed: the chunk files were prebuilt by the
# submitter with a globally non-overlapping random sample per PDG.
python "{worker_script}" \
    --chunk-list   "$CHUNK_FILE" \
    --incident-pdg "$PDG" \
    --chunk-id     "$CHUNK_ID" \
    --output-dir   "{output_dir}" \
    --dx-sweep     {dx_sweep} \
    --nmax-sweep   {nmax_sweep}

echo "finished   : $(date -Iseconds)"
"""


def write_index_file(rows, path: Path):
    with path.open("w") as f:
        f.write("# task_id pdg chunk_id chunk_file\n")
        for task_id, (pdg, chunk_id, cf) in enumerate(rows):
            f.write(f"{task_id} {pdg} {chunk_id} {cf}\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--registry-dir", required=True, type=Path,
                   help="Directory containing registry_pdg_<PDG>.txt files.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output root. Per-pdg CSVs go to <out>/pdg_<PDG>/.")
    p.add_argument("--worker-script", type=Path,
                   default=Path(__file__).resolve().parent / "postprocess_showers_csv.py",
                   help="Path to postprocess_showers_csv.py.")
    p.add_argument("--pdgs", type=int, nargs="+", default=DEFAULT_PDGS,
                   help="Incident PDG codes to process.")
    p.add_argument("--total-per-pdg", type=int, default=10000,
                   help="Number of unique showers to use per PDG (random subsample).")
    p.add_argument("--chunks-per-pdg", type=int, default=5,
                   help="Split each PDG's sample into this many SLURM tasks.")
    p.add_argument("--dx-sweep", type=float, nargs="+", default=[5, 10, 20],
                   help="dx values passed to the worker script.")
    p.add_argument("--nmax-sweep", type=int, nargs="+", default=[1000, 2000, 4000],
                   help="Nmax values passed to the worker script.")
    p.add_argument("--seed", type=int, default=42,
                   help="Master seed for random subsampling.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build chunk files and sbatch script but do not submit.")
    args = p.parse_args()

    # Resolve & validate paths.
    args.registry_dir = args.registry_dir.resolve()
    args.output_dir   = args.output_dir.resolve()
    args.worker_script = args.worker_script.resolve()
    if not args.worker_script.is_file():
        print(f"ERROR: worker script not found: {args.worker_script}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Registry dir   : {args.registry_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Worker script  : {args.worker_script}")
    print(f"PDGs           : {args.pdgs}")
    print(f"Total per pdg  : {args.total_per_pdg}")
    print(f"Chunks per pdg : {args.chunks_per_pdg}")
    print(f"dx sweep       : {args.dx_sweep}")
    print(f"Nmax sweep     : {args.nmax_sweep}")
    print(f"Seed           : {args.seed}")
    print()

    print("Preparing chunk files (random sample, no overlap within PDG)...")
    plan = prepare_chunks(
        registry_dir   = args.registry_dir,
        out_dir        = args.output_dir,
        pdgs           = args.pdgs,
        total_per_pdg  = args.total_per_pdg,
        chunks_per_pdg = args.chunks_per_pdg,
        seed           = args.seed,
    )

    rows = build_index_table(plan)
    n_tasks = len(rows)
    if n_tasks == 0:
        print("ERROR: no tasks to submit.", file=sys.stderr)
        sys.exit(1)

    index_file = args.output_dir / "_chunks" / "task_index.txt"
    write_index_file(rows, index_file)

    sbatch_path = args.output_dir / "_chunks" / "submit_array.sbatch"
    sbatch_text = SBATCH_TEMPLATE.format(
        log_dir       = log_dir,
        last_idx      = n_tasks - 1,
        index_file    = index_file,
        worker_script = args.worker_script,
        output_dir    = args.output_dir,
        dx_sweep      = " ".join(str(v) for v in args.dx_sweep),
        nmax_sweep    = " ".join(str(v) for v in args.nmax_sweep),
    )
    sbatch_path.write_text(sbatch_text)
    sbatch_path.chmod(0o755)

    print()
    print(f"Tasks          : {n_tasks}  (array 0-{n_tasks-1})")
    print(f"Index file     : {index_file}")
    print(f"Sbatch script  : {sbatch_path}")
    print(f"Logs           : {log_dir}/slurm_%A_%a.{{out,err}}")

    if args.dry_run:
        print("\n--dry-run: not submitting. Inspect the sbatch script and run:")
        print(f"  sbatch {sbatch_path}")
        return

    if shutil.which("sbatch") is None:
        print("\nERROR: 'sbatch' not found on PATH. Re-run with --dry-run on a "
              "login/submit node, or run on a node that has SLURM.", file=sys.stderr)
        sys.exit(1)

    print("\nSubmitting...")
    r = subprocess.run(["sbatch", str(sbatch_path)],
                       capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="", file=sys.stderr)
        sys.exit(r.returncode)

    print()
    print("Done. Monitor with:")
    print(f"  squeue -u $USER")
    print(f"  tail -f {log_dir}/slurm_*_0.out")


if __name__ == "__main__":
    main()