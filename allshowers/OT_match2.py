import argparse
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import ot
import showerdata
import torch
import yaml

from allshowers import preprocessing


'''
python /n/home04/hhanif/AllShowers/allshowers/OT_match2.py /n/home04/hhanif/AllShowers/conf/allshowers_muons.yaml --with-time --heavy-files --num-jobs 100
python /n/home04/hhanif/AllShowers/allshowers/OT_match2.py /n/home04/hhanif/AllShowers/conf/allshowers_muons.yaml --with-time --merge --num-jobs 200
'''
start = time.time()
batch_type = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]
]

# ──────────────────────────────────────────────
# Slurm config — edit paths here
# ──────────────────────────────────────────────
SLURM_HEADER = """\
#!/bin/bash
#SBATCH --job-name=ot_heavy
#SBATCH --mem=100G
#SBATCH --cpus-per-task=36
#SBATCH --time=3:00:00
#SBATCH -p serial_requeue
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/ot_heavy_%A_%a.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/ot_heavy_%A_%a.err
#SBATCH --array=0-{num_jobs_minus_1}
export HDF5_USE_FILE_LOCKING=FALSE

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/
"""

SLURM_SINGLE_HEADER = """\
#!/bin/bash
#SBATCH --job-name=ot_full
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH -p serial_requeue
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/ot_full_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/ot_full_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/
"""


def print_time(*args, **kwargs) -> None:
    elapsed = time.time() - start
    print(f"[{elapsed: 5.2f}s]", *args, **kwargs)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match noise to points using OT and save it to the file. "
            "The mapping is done for each shower and each layer separately."
        )
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--resubmit-failed",
        action="store_true",
        default=False,
        help=(
            "Check which sidecar files are missing and resubmit only those Slurm jobs. "
            "Use after a partial --heavy-files run where some jobs failed."
        ),
    )
    parser.add_argument(
        "--with-time",
        action="store_true",
        default=False,
        help=(
            "Include time as a 4th point feature (x, y, e, t) when computing OT. "
            "Requires 'samples_time_trafo' in the config and a 5-column data file. "
            "Without this flag, the original 3-feature mode (x, y, e) is used."
        ),
    )
    parser.add_argument(
        "--num-events", type=int, default=None,
        help=(
            "Number of events/showers to process. If not set, processes the "
            "entire file. Use this for large files to limit memory and compute."
        ),
    )

    # ── Slurm submission mode ──────────────────────────────────────────────
    parser.add_argument(
        "--with-slurm",
        action="store_true",
        default=False,
        help=(
            "Submit a single Slurm job that processes the entire file at once "
            "(uses the full-job header with 200G memory)."
        ),
    )

    # ── Heavy-files array mode ─────────────────────────────────────────────
    parser.add_argument(
        "--heavy-files",
        action="store_true",
        default=False,
        help=(
            "Split the dataset into Slurm array jobs. "
            "Each job processes its own shower slice and writes to a sidecar file."
        ),
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=100,
        help="Number of Slurm array jobs (only used with --heavy-files). Default: 100.",
    )

    # ── Merge sidecars after array jobs finish ──────────────────────────
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help=(
            "Merge per-worker sidecar HDF5 files into the main data file. "
            "Runs automatically after array jobs complete."
        ),
    )

    parser.add_argument(
        "--preprocessed-dir", type=str, default=None,
        help=(
            "Path to directory with preprocessed .pt shard files "
            "(created by preprocess_shards.py). When set, skips raw H5 "
            "loading and transformation — reads already-transformed data "
            "from shards, runs OT matching, and saves noise back into "
            "the shard files."
        ),
    )

    # ── Internal use: array worker arguments (set automatically by Slurm) ──
    parser.add_argument("--start", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--end",   type=int, default=None, help=argparse.SUPPRESS)

    return parser.parse_args(args)


# ══════════════════════════════════════════════════════════════════════════════
# Slurm submission helpers
# ══════════════════════════════════════════════════════════════════════════════

def submit_single_slurm_job(config_file: str, extra_flags: list[str]) -> None:
    """Submit one Slurm job that runs the full file (--with-slurm path)."""
    script_path = os.path.abspath(__file__)
    flags = " ".join(extra_flags)
    script_body = (
        SLURM_SINGLE_HEADER
        + f"\npython {script_path} {config_file} {flags}\n"
    )
    _submit_script(script_body, label="single")


def submit_array_slurm_jobs(
    config_file: str,
    num_showers: int,
    num_jobs: int,
    with_time: bool,
) -> None:
    """Submit a Slurm array that splits showers across num_jobs workers,
    followed by a merge job that combines sidecars into the main file."""
    showers_per_job = -(-num_showers // num_jobs)  # ceiling division
    script_path = os.path.abspath(__file__)

    time_flag = "--with-time" if with_time else ""

    worker_cmd = (
        f"START=$(( SLURM_ARRAY_TASK_ID * {showers_per_job} ))\n"
        f"END=$(( START + {showers_per_job} ))\n"
        f"END=$(( END < {num_showers} ? END : {num_showers} ))\n"
        f"python {script_path} {config_file} {time_flag} --start $START --end $END\n"
    )

    header = SLURM_HEADER.format(num_jobs_minus_1=num_jobs - 1)
    script_body = header + "\n" + worker_cmd

    print_time(
        f"Array config: {num_jobs} jobs, "
        f"{showers_per_job} showers/job, "
        f"total {num_showers} showers"
    )
    array_job_id = _submit_script(script_body, label="array")

    # Submit a merge job that runs after all array tasks complete
    merge_cmd = (
        f"python {script_path} {config_file} {time_flag} "
        f"--merge --num-jobs {num_jobs}\n"
    )
    merge_header = SLURM_SINGLE_HEADER
    merge_body = merge_header + "\n" + merge_cmd
    if array_job_id:
        _submit_script(merge_body, label="merge", dependency=f"afterok:{array_job_id}")
    else:
        print_time("WARNING: could not parse array job ID — submit merge job manually:")
        print_time(f"  python {script_path} {config_file} {time_flag} --merge --num-jobs {num_jobs}")

def resubmit_failed_jobs(
    config_file: str,
    num_showers: int,
    num_jobs: int,
    with_time: bool,
) -> None:
    """Find missing sidecars and resubmit only those jobs as a new array."""
    with open(config_file) as f:
        config = yaml.safe_load(f)
    data_file = config["data"]["path"]

    showers_per_job = -(-num_showers // num_jobs)
    script_path = os.path.abspath(__file__)
    time_flag = "--with-time" if with_time else ""

    # Find which job IDs are missing their sidecar
    failed_job_ids = []
    for job_id in range(num_jobs):
        s = job_id * showers_per_job
        e = min(s + showers_per_job, num_showers)
        if s >= num_showers:
            break
        sidecar = _sidecar_path(data_file, s, e)
        if not os.path.exists(sidecar):
            failed_job_ids.append((job_id, s, e))

    if not failed_job_ids:
        print_time("No missing sidecars found — all jobs completed successfully.")
        return

    print_time(f"Found {len(failed_job_ids)} missing sidecars:")
    for job_id, s, e in failed_job_ids:
        print_time(f"  job {job_id}: [{s}, {e})")
    sys.stdout.flush()

    # Submit one job per missing sidecar
    job_ids = []
    for job_id, s, e in failed_job_ids:
        worker_cmd = (
            f"python {script_path} {config_file} {time_flag} "
            f"--start {s} --end {e}\n"
        )
        script_body = SLURM_HEADER.format(num_jobs_minus_1=0) + "\n" + worker_cmd
        # Replace array directive since these are individual jobs
        script_body = script_body.replace(
            "#SBATCH --array=0-0\n", ""
        ).replace(
            "#SBATCH --job-name=ot_heavy\n",
            f"#SBATCH --job-name=ot_retry_{s}_{e}\n"
        )
        jid = _submit_script(script_body, label=f"retry_{s}_{e}")
        if jid:
            job_ids.append(jid)

    # Submit merge job dependent on all retry jobs finishing
    if job_ids:
        dependency = "afterok:" + ":".join(job_ids)
        merge_cmd = (
            f"python {script_path} {config_file} {time_flag} "
            f"--merge --num-jobs {num_jobs}\n"
        )
        merge_body = SLURM_SINGLE_HEADER + "\n" + merge_cmd
        _submit_script(merge_body, label="merge_retry", dependency=dependency)
    else:
        print_time("WARNING: could not parse any job IDs — submit merge job manually.")

def _submit_script(
    script_body: str, label: str, dependency: str | None = None,
) -> str | None:
    """Submit a Slurm script and return the job ID (or None on failure to parse)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix=f"ot_{label}_"
    ) as f:
        f.write(script_body)
        tmp_path = f.name

    cmd = ["sbatch"]
    if dependency:
        cmd += [f"--dependency={dependency}"]
    cmd.append(tmp_path)

    print_time(f"Submitting Slurm script: {tmp_path}")
    print("─" * 60)
    print(script_body)
    print("─" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print_time("Submitted:", result.stdout.strip())
    else:
        print_time("sbatch failed:", result.stderr.strip())
        sys.exit(1)
    os.unlink(tmp_path)

    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else None


# ══════════════════════════════════════════════════════════════════════════════
# HDF5 target writer — no padding, exact num_points per shower
# ══════════════════════════════════════════════════════════════════════════════

def init_target_dataset(path: str, num_showers: int, F: int, key: str = "target") -> None:
    """
    Create the target group in the HDF5 file if it doesn't already exist.
    Uses variable-length float32 storage so no padding is needed.
    """
    with h5py.File(path, "a") as f:
        if key in f:
            print_time(f"Target dataset '{key}' already exists — skipping creation.")
            return
        grp = f.create_group(key)
        vlen_dtype = h5py.vlen_dtype(np.dtype("float32"))
        grp.create_dataset(
            "point_clouds",
            shape=(num_showers,),
            dtype=vlen_dtype,
        )
        grp.create_dataset(
            "num_points",
            shape=(num_showers,),
            dtype=np.int32,
        )
        grp.attrs["num_features"] = F
    print_time(f"Target dataset '{key}' created (vlen, no padding, F={F}).")


def save_target_batch_exact(
    noise: npt.NDArray[np.float32],    # shape [batch, max_points, F]
    num_points: npt.NDArray[np.int32], # shape [batch]
    path: str,
    start: int,
    key: str = "target",
) -> None:
    """
    Write a batch of noise arrays to the HDF5 file at position [start:start+batch].
    Only the first num_points[i] rows of noise[i] are stored — no padding zeros.
    """
    F = noise.shape[2]
    point_clouds = [
        noise[i, :num_points[i], :].flatten().astype(np.float32)
        for i in range(len(noise))
    ]

    with h5py.File(path, "a") as f:
        if key not in f:
            raise KeyError(
                f"Target group '{key}' not found. "
                "Run init_target_dataset() before saving batches."
            )
        pc_ds  = f[f"{key}/point_clouds"]
        npt_ds = f[f"{key}/num_points"]

        end = start + len(point_clouds)
        pc_ds[start:end]  = point_clouds
        npt_ds[start:end] = num_points


# ══════════════════════════════════════════════════════════════════════════════
# Original preprocessing / OT logic (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class PreProcessor:
    def __init__(self, config_file: str, with_time: bool = False) -> None:
        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.with_time = with_time
        self.num_features = 4 if with_time else 3

        self.samples_energy_trafo = preprocessing.compose(
            transformation=config["data"]["samples_energy_trafo"],
        )
        self.samples_coordinate_trafo = preprocessing.compose(
            transformation=config["data"]["samples_coordinate_trafo"],
        )

        if self.with_time:
            if "samples_time_trafo" not in config["data"]:
                raise KeyError(
                    "'--with-time' was set but 'samples_time_trafo' is missing from "
                    "the config file's 'data' section."
                )
            self.samples_time_trafo = preprocessing.compose(
                transformation=config["data"]["samples_time_trafo"],
            )
        else:
            self.samples_time_trafo = None

        self.file_path, showers, self.data_shape = self.__get_data(config)
        showers = torch.from_numpy(showers)

        mask = showers[:, :, 3] > 0.0

        self.samples_coordinate_trafo.to(showers.dtype)
        self.samples_energy_trafo.to(showers.dtype)

        self.samples_coordinate_trafo.fit(
            x=showers[:, :, :2],
            mask=mask[:, :, None].repeat(1, 1, 2),
        )
        self.samples_energy_trafo.fit(
            x=showers[:, :, 3],
            mask=mask,
        )

        if self.with_time:
            self.samples_time_trafo.to(showers.dtype)
            self.samples_time_trafo.fit(
                x=showers[:, :, 4],
                mask=mask,
            )

        layer = (showers[:, :, 2] + 0.5).to(torch.int64)
        self.num_layers = int(torch.max(layer).item() + 1)

    def __get_data(
        self, config: dict[str, Any]
    ) -> tuple[str, npt.NDArray[np.float32], tuple[int, ...]]:
        data_shape = showerdata.get_file_shape(config["data"]["path"])
        showers = showerdata.load(
            path=config["data"]["path"],
            stop=8000,
        )
        num_cols = 5 if self.with_time else 4
        return config["data"]["path"], showers.points[:, :, :num_cols], data_shape

    def __call__(
        self,
        x: npt.NDArray[np.float32],
    ) -> batch_type:
        x_tensor = torch.from_numpy(x)
        mask = x_tensor[:, 3] > 0.0

        x_tensor[:, :2] = self.samples_coordinate_trafo(
            x_tensor[:, :2].permute(0, 2, 1)
        ).permute(0, 2, 1)
        x_tensor[:, 3] = self.samples_energy_trafo(x_tensor[:, 3])

        layer = (x_tensor[:, 2] + 0.5).to(torch.int64)

        if self.with_time:
            x_tensor[:, 4] = self.samples_time_trafo(x_tensor[:, 4])
            x_tensor = x_tensor[:, [0, 1, 3, 4]]
        else:
            x_tensor = x_tensor[:, [0, 1, 3]]

        return x_tensor.numpy(), mask.numpy(), layer.numpy()


class DataLoader(Iterable[npt.NDArray[np.float32]]):
    def __init__(self, data_file: str, batch_size: int, start: int = 0, end: int | None = None) -> None:
        self.file_name = data_file
        self.batch_size = batch_size
        self.start = start
        self.end = end

    def __iter__(self) -> Iterator[npt.NDArray[np.float32]]:
        with showerdata.ShowerDataFile(self.file_name, "r") as file:
            total = self.end if self.end is not None else len(file)
            for batch_start in range(self.start, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                samples = file[batch_start:batch_end].points
                yield samples.transpose(0, 2, 1)


class NoiseMatcher:
    def __init__(self, pre_processor: PreProcessor) -> None:
        self.__num_layers = pre_processor.num_layers
        self.__num_features = pre_processor.num_features
        self.pre_processor = pre_processor

    def __call__(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        points, mask, layer = self.pre_processor(samples)
        F = self.__num_features
        noise = np.random.randn(points.shape[0], F, points.shape[2])

        for i in range(self.__num_layers):
            mask_local = np.expand_dims(np.logical_and(mask, layer == i), 1)
            for j in range(len(points)):
                points_j = (
                    points[j].T[mask_local[j].repeat(F).reshape(-1, F)]
                    .reshape(-1, F)
                )
                noise_j = (
                    noise[j].T[mask_local[j].repeat(F).reshape(-1, F)]
                    .reshape(-1, F)
                )
                if len(points_j) > 1:
                    N = len(points_j)
                    assert len(noise_j) == N

                    M = np.sqrt(
                        np.sum(
                            (points_j[:, None, :] - noise_j[None, :, :]) ** 2, axis=-1
                        )
                    )
                    wa = np.ones(N) / N
                    wb = np.ones(N) / N
                    T = ot.emd(wa, wb, M, numItermax=1_000_000)
                    noise_j = N * (T @ noise_j)

                    noise[j].T[mask_local[j].repeat(F).reshape(-1, F)] = (
                        noise_j.flatten()
                    )

        noise[(~mask[:, None, :]).repeat(F, axis=1)] = 0.0
        return noise.astype(np.float32, copy=False)


# ══════════════════════════════════════════════════════════════════════════════
# Processing — two paths: full file vs slice (heavy-files worker)
# ══════════════════════════════════════════════════════════════════════════════

def process_full_file(
    data_file: str,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    batch_size: int = 128,
    num_events: int | None = None,
) -> None:
    """Original single-job path — saves with showerdata.save_target."""
    F = pre_processor.num_features
    total_events = num_events if num_events is not None else data_shape[0]
    num_batches = -(-total_events // batch_size)
    print_time("batch size:", batch_size)
    print_time(f"total events: {total_events} (file has {data_shape[0]})")
    print_time("number of batches:", num_batches)
    print_time(f"num features: {F}  {'(x, y, e, t)' if F == 4 else '(x, y, e)'}")
    sys.stdout.flush()

    noise_matcher = NoiseMatcher(pre_processor)
    noise = np.empty((total_events, F, data_shape[1]), dtype=np.float32)
    print_time(f"NoiseMatcher initialized. (noise shape={noise.shape})")
    sys.stdout.flush()

    num_processes = n - 1 if (n := os.cpu_count()) else 1
    with multiprocessing.Pool(num_processes) as pool:
        for i, batch in enumerate(
            pool.imap(
                noise_matcher,
                DataLoader(data_file, batch_size, start=0, end=total_events),
            )
        ):
            noise[i * batch_size : i * batch_size + len(batch)] = batch

    print_time("All batches processed.")
    sys.stdout.flush()

    noise = noise.transpose(0, 2, 1)
    showerdata.save_target(noise, data_file, overwrite=True)
    print_time(f"Noise saved successfully to {data_file} (shape={noise.shape}).")
    sys.stdout.flush()


def _sidecar_path(data_file: str, slice_start: int, slice_end: int) -> str:
    """Return the path for a per-worker sidecar HDF5 file."""
    base, ext = os.path.splitext(data_file)
    return f"{base}_target_{slice_start}_{slice_end}{ext}"


def process_slice(
    data_file: str,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    slice_start: int,
    slice_end: int,
    batch_size: int = 32,
    key: str = "target",
) -> None:
    """
    Heavy-files worker path.
    Processes showers [slice_start, slice_end) and writes noise to a
    per-worker sidecar HDF5 file to avoid concurrent-write corruption.
    """
    F = pre_processor.num_features
    slice_len = slice_end - slice_start
    num_batches = -(-slice_len // batch_size)

    print_time(f"Worker slice: [{slice_start}, {slice_end})  ({slice_len} showers)")
    print_time(f"num features: {F}  {'(x, y, e, t)' if F == 4 else '(x, y, e)'}")
    print_time(f"num batches: {num_batches}")
    sys.stdout.flush()

    noise_matcher = NoiseMatcher(pre_processor)
    max_points = data_shape[1]

    # shape: [slice_len, max_points, F]
    noise_full = np.zeros((slice_len, max_points, F), dtype=np.float32)

    num_processes = max(1, (os.cpu_count() or 1) - 1)
    loader = DataLoader(data_file, batch_size, start=slice_start, end=slice_end)

    with multiprocessing.Pool(num_processes) as pool:
        for i, batch_noise in enumerate(pool.imap(noise_matcher, loader)):
            # batch_noise: [batch, F, max_points]
            local_start = i * batch_size
            local_end   = local_start + batch_noise.shape[0]
            # transpose to [batch, max_points, F] before storing
            noise_full[local_start:local_end] = batch_noise.transpose(0, 2, 1)
            print_time(f"  batch {i+1}/{num_batches} done")
            sys.stdout.flush()

    print_time("All batches processed. Computing num_points and writing to sidecar...")
    sys.stdout.flush()

    with showerdata.ShowerDataFile(data_file, "r") as sf:
        src_showers = sf[slice_start:slice_end]
        num_points = src_showers._num_points.astype(np.int32)

    sidecar = _sidecar_path(data_file, slice_start, slice_end)
    init_target_dataset(sidecar, slice_len, F, key=key)
    save_target_batch_exact(
        noise=noise_full,
        num_points=num_points,
        path=sidecar,
        start=0,
        key=key,
    )

    print_time(
        f"Slice [{slice_start}, {slice_end}) written to sidecar '{sidecar}'."
    )
    sys.stdout.flush()


def merge_sidecars(
    data_file: str,
    num_showers: int,
    num_jobs: int,
    F: int,
    key: str = "target",
) -> None:
    """
    Merge per-worker sidecar HDF5 files into the main data file.
    Processes one slice at a time to avoid OOM. Sidecars are deleted
    only after all slices are successfully written.
    """
    showers_per_job = -(-num_showers // num_jobs)

    # Pass 1: find global max_points across all sidecars
    max_points = 0
    for job_id in range(num_jobs):
        s = job_id * showers_per_job
        e = min(s + showers_per_job, num_showers)
        if s >= num_showers:
            break
        sidecar = _sidecar_path(data_file, s, e)
        if not os.path.exists(sidecar):
            print_time(f"WARNING: sidecar {sidecar} not found — skipping in pass 1")
            continue
        with h5py.File(sidecar, "r") as sf:
            npts = sf[f"{key}/num_points"][:]
            max_points = max(max_points, int(npts.max()))

    if max_points == 0:
        raise RuntimeError("No sidecar files found — cannot determine max_points.")

    print_time(f"Global max_points={max_points}.")
    sys.stdout.flush()

    # Delete existing target group so we can recreate it cleanly
    with h5py.File(data_file, "a") as hf:
        if key in hf:
            del hf[key]
            print_time(f"Deleted old '{key}' group from main file.")

    # Create target dataset upfront with correct shape
    shape = (num_showers, max_points, F)
    print_time(f"Creating target dataset with shape {shape}...")
    showerdata.add_target_dataset(data_file, shape=shape, key=key, exists_ok=False)
    sys.stdout.flush()

    # Pass 2: merge sidecar by sidecar — only one slice in memory at a time
    sidecars_to_delete = []
    for job_id in range(num_jobs):
        s = job_id * showers_per_job
        e = min(s + showers_per_job, num_showers)
        if s >= num_showers:
            break

        sidecar = _sidecar_path(data_file, s, e)
        if not os.path.exists(sidecar):
            print_time(f"WARNING: sidecar {sidecar} not found — skipping")
            continue

        slice_len = e - s
        noise_slice = np.zeros((slice_len, max_points, F), dtype=np.float32)

        with h5py.File(sidecar, "r") as sf:
            pc_raw = sf[f"{key}/point_clouds"]
            npts   = sf[f"{key}/num_points"][:]
            for i in range(slice_len):
                n = npts[i]
                noise_slice[i, :n, :] = pc_raw[i].reshape(n, F)

        showerdata.save_target_batch(noise_slice, data_file, npts, start=s, key=key)
        print_time(f"Merged sidecar [{s}, {e})")
        sidecars_to_delete.append(sidecar)
        sys.stdout.flush()

    # Only delete sidecars after all writes succeed
    print_time("All sidecars merged successfully. Deleting sidecars...")
    for sidecar in sidecars_to_delete:
        os.remove(sidecar)
        print_time(f"Deleted {sidecar}")

    print_time("Done.")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessed-shards processing path
# ══════════════════════════════════════════════════════════════════════════════

def match_noise_batch(
    points: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    layer: npt.NDArray[np.int64],
    num_layers: int,
    num_features: int,
) -> npt.NDArray[np.float32]:
    """Run OT noise matching on a batch of already-transformed data.

    Parameters
    ----------
    points : [batch, F, max_points]
    mask   : [batch, max_points]
    layer  : [batch, max_points]

    Returns
    -------
    noise  : [batch, F, max_points]
    """
    F = num_features
    noise = np.random.randn(points.shape[0], F, points.shape[2])

    for i in range(num_layers):
        mask_local = np.expand_dims(np.logical_and(mask, layer == i), 1)
        for j in range(len(points)):
            points_j = (
                points[j].T[mask_local[j].repeat(F).reshape(-1, F)]
                .reshape(-1, F)
            )
            noise_j = (
                noise[j].T[mask_local[j].repeat(F).reshape(-1, F)]
                .reshape(-1, F)
            )
            if len(points_j) > 1:
                N = len(points_j)
                assert len(noise_j) == N

                M = np.sqrt(
                    np.sum(
                        (points_j[:, None, :] - noise_j[None, :, :]) ** 2, axis=-1
                    )
                )
                wa = np.ones(N) / N
                wb = np.ones(N) / N
                T = ot.emd(wa, wb, M, numItermax=1_000_000)
                noise_j = N * (T @ noise_j)

                noise[j].T[mask_local[j].repeat(F).reshape(-1, F)] = (
                    noise_j.flatten()
                )

    noise[(~mask[:, None, :]).repeat(F, axis=1)] = 0.0
    return noise.astype(np.float32, copy=False)


class ShardNoiseMatcher:
    """Callable wrapper for multiprocessing — takes pre-transformed batch tuple."""

    def __init__(self, num_layers: int, num_features: int) -> None:
        self.num_layers = num_layers
        self.num_features = num_features

    def __call__(
        self,
        batch: tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    ) -> npt.NDArray[np.float32]:
        points, mask, layer = batch
        return match_noise_batch(
            points, mask, layer, self.num_layers, self.num_features,
        )


def _iter_shard_batches(
    shard_path: str,
    batch_size: int,
) -> Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]:
    """Yield (points, mask, layer) batches from a single .pt shard."""
    data = torch.load(shard_path, weights_only=False)
    x = data["x"].numpy()           # [N, max_points, F]
    mask = data["mask"].numpy()      # [N, max_points, 1]
    layer = data["layer"].numpy()    # [N, max_points, 1]
    n = len(x)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield (
            x[s:e].transpose(0, 2, 1),     # → [batch, F, max_points]
            mask[s:e].squeeze(-1),          # → [batch, max_points]
            layer[s:e].squeeze(-1),         # → [batch, max_points]
        )


def process_shards(
    preprocessed_dir: str,
    prefix: str,
    batch_size: int = 4,
) -> None:
    """Run OT matching on preprocessed .pt shards and save noise back."""
    shard_files = sorted(
        f for f in os.listdir(preprocessed_dir)
        if f.startswith(prefix + "_") and f.endswith(".pt")
    )
    if not shard_files:
        raise FileNotFoundError(
            f"No shard files matching '{prefix}_*.pt' in {preprocessed_dir}"
        )

    first = torch.load(
        os.path.join(preprocessed_dir, shard_files[0]), weights_only=False,
    )
    num_features = first["x"].shape[-1]
    num_layers = int(first["layer"].max().item()) + 1
    del first

    print_time(f"Shards: {len(shard_files)} files ({prefix}_*.pt)")
    print_time(f"num_features: {num_features}, num_layers: {num_layers}")
    sys.stdout.flush()

    noise_matcher = ShardNoiseMatcher(num_layers, num_features)
    num_processes = max(1, (os.cpu_count() or 1) - 1)

    for shard_file in shard_files:
        shard_path = os.path.join(preprocessed_dir, shard_file)
        print_time(f"Processing {shard_file}...")
        sys.stdout.flush()

        noise_parts = []
        with multiprocessing.Pool(num_processes) as pool:
            for batch_noise in pool.imap(
                noise_matcher,
                _iter_shard_batches(shard_path, batch_size),
            ):
                noise_parts.append(batch_noise.transpose(0, 2, 1))

        noise = np.concatenate(noise_parts, axis=0)
        noise_tensor = torch.from_numpy(noise)

        data = torch.load(shard_path, weights_only=False)
        data["noise"] = noise_tensor
        torch.save(data, shard_path)

        print_time(f"  Saved noise to {shard_file} ({len(noise)} events)")
        sys.stdout.flush()

    print_time("All shards processed.")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def main(args: list[str] | None = None):
    torch.set_num_threads(1)

    parsed_args = parse_args(args)
    print_time("Parsed arguments:", parsed_args)
    sys.stdout.flush()

    # ── 0. --preprocessed-dir: process .pt shards directly ──────────────
    if parsed_args.preprocessed_dir is not None:
        print_time(f"Mode: preprocessed shards from {parsed_args.preprocessed_dir}")
        sys.stdout.flush()

        for prefix in ("train", "val"):
            shard_check = [
                f for f in os.listdir(parsed_args.preprocessed_dir)
                if f.startswith(prefix + "_") and f.endswith(".pt")
            ]
            if shard_check:
                print_time(f"\n--- Processing {prefix} shards ---")
                sys.stdout.flush()
                process_shards(
                    parsed_args.preprocessed_dir,
                    prefix=prefix,
                    batch_size=4,
                )
            else:
                print_time(f"No {prefix} shards found, skipping.")
        return

    # ── 1. --with-slurm: submit one big job and exit ───────────────────────
    if parsed_args.with_slurm:
        extra = []
        if parsed_args.with_time:
            extra.append("--with-time")
        submit_single_slurm_job(parsed_args.file, extra)
        return

    # ── Resolve num_events for all paths ────────────────────────────────
    with open(parsed_args.file) as f:
        config = yaml.safe_load(f)
    data_shape = showerdata.get_file_shape(config["data"]["path"])
    num_showers = data_shape[0]

    if parsed_args.num_events is not None:
        if parsed_args.num_events > num_showers:
            print_time(
                f"WARNING: --num-events {parsed_args.num_events} exceeds file size "
                f"{num_showers}, clamping."
            )
        num_showers = min(parsed_args.num_events, num_showers)

    print_time(f"Events in file: {data_shape[0]}")
    print_time(f"Events to process: {num_showers}")
    sys.stdout.flush()

    # ── 2. --heavy-files: submit array jobs + merge job, then exit ──────
    if parsed_args.heavy_files:
        submit_array_slurm_jobs(
            config_file=parsed_args.file,
            num_showers=num_showers,
            num_jobs=parsed_args.num_jobs,
            with_time=parsed_args.with_time,
        )
        return

    # ── 2b. --merge: combine sidecar files after array jobs ──────────────
    if parsed_args.merge:
        F = 4 if parsed_args.with_time else 3

        merge_sidecars(
            data_file=config["data"]["path"],
            num_showers=num_showers,
            num_jobs=parsed_args.num_jobs,
            F=F,
            key="target",
        )
        return

    # ── 2c. --resubmit-failed: resubmit missing sidecar jobs ─────────────
    if parsed_args.resubmit_failed:
        resubmit_failed_jobs(
            config_file=parsed_args.file,
            num_showers=num_showers,
            num_jobs=parsed_args.num_jobs,
            with_time=parsed_args.with_time,
        )
        return
    # ── 3. Worker mode: --start and --end set by Slurm array ──────────────
    if parsed_args.start is not None and parsed_args.end is not None:
        print_time(
            f"Mode: heavy-files worker  "
            f"[{parsed_args.start}, {parsed_args.end})  "
            f"{'with time' if parsed_args.with_time else 'no time'}"
        )
        sys.stdout.flush()

        pre_processor = PreProcessor(parsed_args.file, with_time=parsed_args.with_time)
        print_time("PreProcessor initialised.")
        sys.stdout.flush()

        process_slice(
            data_file=pre_processor.file_path,
            data_shape=pre_processor.data_shape,
            pre_processor=pre_processor,
            slice_start=parsed_args.start,
            slice_end=min(parsed_args.end, pre_processor.data_shape[0]),
            batch_size=4,
            key="target",
        )
        return

    # ── 4. Default: run everything locally (original behaviour) ───────────
    print_time(
        f"Mode: {'with time (x, y, e, t)' if parsed_args.with_time else 'original (x, y, e)'}"
    )
    sys.stdout.flush()

    pre_processor = PreProcessor(parsed_args.file, with_time=parsed_args.with_time)
    print_time("PreProcessor initialised.")
    sys.stdout.flush()

    process_full_file(
        data_file=pre_processor.file_path,
        data_shape=pre_processor.data_shape,
        pre_processor=pre_processor,
        batch_size=4,
        num_events=num_showers,
    )


if __name__ == "__main__":
    main()