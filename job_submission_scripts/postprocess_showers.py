#!/usr/bin/env python3
"""
Post-process a CHUNK of combined hit-level Parquet files into 3 HDF5 files.

Consumes the output of preprocess_showers.py:
    - Per-shower combined parquet: <shower_id>_hits.parquet  (hit-level rows)
    - Per-shower event CSV:        <shower_id>_event.csv     (1 metadata row)
    - A per-PDG registry file:     registry_pdg_<PDG>.txt    (list of parquet paths)

For one chunk (subset of parquet paths), it produces 3 HDF5 files in an
output subdirectory named after the incident PDG (e.g. ``pdg_11/``):

    <out>/pdg_<PDG>/chunk_<XXXX>_electrons.h5
    <out>/pdg_<PDG>/chunk_<XXXX>_muons.h5
    <out>/pdg_<PDG>/chunk_<XXXX>_photons.h5

Per H5 file, datasets follow the original shower schema:
    showers     vlen float32  (N_showers,)      flat → reshape(num_points, n_feat)
                                                columns: x, y, plane_idx, energy, time
    directions  float32       (N_showers, 3)    [nx, ny, nz] from event CSV
    energies    float32       (N_showers,)      primary kinetic energy
    pdg         int32         (N_showers,)      class: 0=e±/γ/π⁰, 1=π±
    actual_pdg  int32         (N_showers,)      true primary PDG ID
    shape       int64         (3,)              [N_showers, nmax, n_feat]
    num_points  int64         (N_showers,)      actual clustered points per shower

Pipeline per shower:
    1. Read hits parquet (only columns needed)
    2. Drop near-duplicate hits (time/energy/xy tolerance — optional)
    3. Split hits by secondary type (e±, μ±, γ)
    4. Cluster per plane into (dx × dx) cells (sum energy, earliest time)
    5. Truncate to top-nmax highest-energy cells
    6. Sort by plane_idx
    7. Append into per-particle H5 file (growing datasets)

Usage (single node, processes one chunk = many parquets):

    python postprocess_showers.py \\
        --chunk-list /path/to/chunk_0000.txt \\
        --incident-pdg 11 \\
        --chunk-id 0 \\
        --output-dir /path/to/output \\
        --electrons-dx 10.0 --muons-dx 10.0 --photons-dx 10.0 \\
        --electrons-nmax 4000 --muons-nmax 28000 --photons-nmax 8000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from typing import Optional


# =============================================================================
# Particle type configuration (secondary particles)
#
# NOTE: preprocess_showers.py applies PDG_REMAP that collapses sign:
#     {11: 11, -11: 11, 13: 13, -13: 13, 22: 22}
# so the `pdg` column in the input parquets only ever contains {11, 13, 22}.
# We therefore match only the positive value per species.
# =============================================================================

PARTICLE_CONFIGS = {
    "electrons": {
        "pdg_values":   [11],           # e± (sign-collapsed upstream)
        "label":        "e± (pdg 11)",
        "default_nmax": 4000,
        "default_dx":   10.0,
        "h5_suffix":    "electrons",
    },
    "muons": {
        "pdg_values":   [13],           # μ± (sign-collapsed upstream)
        "label":        "μ± (pdg 13)",
        "default_nmax": 28000,
        "default_dx":   10.0,
        "h5_suffix":    "muons",
    },
    "photons": {
        "pdg_values":   [22],
        "label":        "γ (pdg 22)",
        "default_nmax": 8000,
        "default_dx":   10.0,
        "h5_suffix":    "photons",
    },
}


# =============================================================================
# Lazy imports
# =============================================================================

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import h5py
    return np, pd, pa, pq, pc, h5py


# =============================================================================
# Helpers
# =============================================================================

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def _pdg_class(pdg_primary: float) -> int:
    """Binary class label: 0 = e±/γ/π⁰, 1 = π±."""
    try:
        return 1 if abs(int(round(float(pdg_primary)))) == 211 else 0
    except Exception:
        return 0


# =============================================================================
# Event-CSV reader
# =============================================================================

def _event_csv_path_for(hits_parquet_path: str) -> str:
    """<shower>_hits.parquet  →  <shower>_event.csv"""
    base = hits_parquet_path
    if base.endswith("_hits.parquet"):
        return base[: -len("_hits.parquet")] + "_event.csv"
    # fallback: replace last .parquet with _event.csv
    root, _ = os.path.splitext(base)
    return root + "_event.csv"


def _load_event_row(csv_path: str) -> Optional[dict]:
    """Read the single-row event CSV and return it as a dict (or None)."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception:
        return None
    return None


def _float_or(default: float, val) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _extract_event_meta(csv_path: str) -> dict:
    """
    Pull the fields we need from <shower>_event.csv.
    Falls back gracefully if file is missing/malformed.
    """
    row = _load_event_row(csv_path) or {}
    return {
        "shower_id":        str(row.get("shower_id", "")),
        "incident_energy":  _float_or(0.0, row.get("incident_energy")),
        "direction_x":      _float_or(0.0, row.get("direction_x")),
        "direction_y":      _float_or(0.0, row.get("direction_y")),
        "direction_z":      _float_or(1.0, row.get("direction_z")),
        "incident_pdg":     _float_or(0.0, row.get("incident_pdg")),
        "incident_class_id": _float_or(0.0, row.get("incident_class_id")),
    }


# =============================================================================
# Duplicate removal
# =============================================================================

def _drop_near_duplicates(df, np, pd,
                          time_tol: float, energy_rel_tol: float, xy_tol: float,
                          verbose: bool = False):
    """
    Drop near-duplicate hits that arise when the same particle is recorded
    twice on a plane (tiny jitter in time / energy / xy).

    Strategy: sort by (plane_index, pdg, time, energy, x, y), then flag
    consecutive rows where ALL of {time, energy (relative), x, y} are within
    their respective tolerances AND the same plane+pdg.

    Returns (deduped_df, n_removed).
    """
    if len(df) == 0:
        return df, 0

    # Sort so true near-duplicates become adjacent
    sort_cols = ["plane_index", "pdg", "time", "kinetic_energy", "x", "y"]
    df_sorted = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    n = len(df_sorted)
    if n < 2:
        return df_sorted, 0

    t   = df_sorted["time"].to_numpy(dtype=np.float64)
    e   = df_sorted["kinetic_energy"].to_numpy(dtype=np.float64)
    xx  = df_sorted["x"].to_numpy(dtype=np.float64)
    yy  = df_sorted["y"].to_numpy(dtype=np.float64)
    pl  = df_sorted["plane_index"].to_numpy()
    pdg = df_sorted["pdg"].to_numpy()

    dt  = np.abs(np.diff(t))
    de  = np.abs(np.diff(e))
    # relative energy diff, guarding against zeros
    emag = np.maximum(np.abs(e[:-1]), np.abs(e[1:]))
    emag = np.where(emag > 0, emag, 1.0)
    de_rel = de / emag
    dx  = np.abs(np.diff(xx))
    dy  = np.abs(np.diff(yy))
    same_plane = (pl[1:] == pl[:-1])
    same_pdg   = (pdg[1:] == pdg[:-1])

    close = (
        same_plane
        & same_pdg
        & (dt    <= time_tol)
        & (de_rel <= energy_rel_tol)
        & (dx    <= xy_tol)
        & (dy    <= xy_tol)
    )

    # mark row i+1 as duplicate when close[i] is True
    dup_mask = np.zeros(n, dtype=bool)
    dup_mask[1:] = close

    n_removed = int(dup_mask.sum())
    if n_removed == 0:
        return df_sorted, 0

    if verbose and n_removed > 0:
        print(f"      dedup: removed {n_removed}/{n} near-duplicate hits "
              f"(time<{time_tol:.1e}s, dE/E<{energy_rel_tol:.1e}, dxy<{xy_tol:.1e}m)")

    return df_sorted.loc[~dup_mask].reset_index(drop=True), n_removed


# =============================================================================
# Clustering (adapted from original postprocess_showers)
# =============================================================================

def _cluster_shower_part(xy, e, cell_size, shift, np, t=None):
    """Bin particles into (cell_size × cell_size) cells on one plane."""
    if len(xy) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,),   dtype=np.float32),
            None if t is None else np.empty((0,), dtype=np.float32),
        )

    xy = xy.copy().astype(np.float32)
    xy += shift
    xy /= cell_size
    xy_idx = np.floor(xy).astype(np.int32)

    x_min = int(xy_idx[:, 0].min())
    y_min = int(xy_idx[:, 1].min())
    stride = int(xy_idx[:, 0].max()) - x_min + 2
    keys = (xy_idx[:, 0].astype(np.int64) - x_min) * stride + \
           (xy_idx[:, 1].astype(np.int64) - y_min)

    unique_keys, inverse_idx = np.unique(keys, return_inverse=True)
    unique_x = (unique_keys // stride).astype(np.int32) + x_min
    unique_y = (unique_keys  % stride).astype(np.int32) + y_min

    # summed energy
    e_clustered = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(e_clustered, inverse_idx, e)

    # earliest time
    if t is not None:
        t_clustered = np.full(len(unique_keys), np.inf, dtype=np.float32)
        np.minimum.at(t_clustered, inverse_idx, t)
    else:
        t_clustered = None

    xy_clustered = np.column_stack([unique_x, unique_y]).astype(np.float32)
    xy_clustered += 0.5
    xy_clustered *= cell_size
    xy_clustered -= shift
    return xy_clustered, e_clustered, t_clustered


def _cluster_hits_for_type(df_sub, cell_size, np, include_time):
    """
    df_sub:  one secondary-type subset, columns: x, y, plane_index, kinetic_energy, time
    Returns: ndarray (n_pts, n_feat)  with columns x, y, plane_idx, energy [, time]
    """
    n_feat = 5 if include_time else 4
    if len(df_sub) == 0:
        return np.empty((0, n_feat), dtype=np.float32)

    pos   = df_sub[["x", "y", "plane_index"]].to_numpy(dtype=np.float32)
    e_all = np.maximum(df_sub["kinetic_energy"].to_numpy(dtype=np.float32), 0.0)
    t_all = (df_sub["time"].to_numpy(dtype=np.float32)
             if include_time and "time" in df_sub.columns else None)
    plane = pos[:, 2].astype(np.int32)

    parts = []
    for p_val in np.unique(plane):
        mask = plane == p_val
        xy_p = pos[mask, :2]
        e_p  = e_all[mask]
        t_p  = t_all[mask] if t_all is not None else None
        shift = np.array([0.0, 0.0], dtype=np.float32)  # deterministic

        xy_c, e_c, t_c = _cluster_shower_part(xy_p, e_p, cell_size, shift, np, t=t_p)
        if len(xy_c) == 0:
            continue

        block_cols = [
            xy_c[:, 0],
            xy_c[:, 1],
            np.full(len(xy_c), p_val, dtype=np.float32),
            e_c,
        ]
        if include_time:
            block_cols.append(t_c)
        parts.append(np.column_stack(block_cols).astype(np.float32))

    if not parts:
        return np.empty((0, n_feat), dtype=np.float32)

    return np.concatenate(parts, axis=0)


def _truncate_and_sort(arr, nmax, np):
    """Keep top-nmax by energy (col 3), then sort by plane_idx (col 2)."""
    if len(arr) == 0:
        return arr
    if len(arr) > nmax:
        top_idx = arr[:, 3].argpartition(-nmax)[-nmax:]
        arr = arr[top_idx]
    # sort by plane_idx ascending
    order = np.argsort(arr[:, 2], kind="mergesort")
    return arr[order]


# =============================================================================
# HDF5 writer (appending, growing datasets)
# =============================================================================

class ShowerH5Writer:
    """Appends one-shower-at-a-time into an H5 file with resizable datasets."""

    def __init__(self, path: str, nmax: int, n_feat: int, np, h5py,
                 extra_attrs: Optional[dict] = None):
        self.path   = path
        self.nmax   = int(nmax)
        self.n_feat = int(n_feat)
        self.np     = np
        self.h5py   = h5py

        self.f = h5py.File(path, "w")
        self.f.attrs["created"] = datetime.now().isoformat()
        self.f.attrs["nmax"]    = self.nmax
        self.f.attrs["n_feat"]  = self.n_feat
        if extra_attrs:
            for k, v in extra_attrs.items():
                try:
                    self.f.attrs[k] = v
                except Exception:
                    self.f.attrs[k] = str(v)

        dt_vlen = h5py.vlen_dtype(np.dtype("float32"))
        self.ds_showers = self.f.create_dataset(
            "showers", shape=(0,), maxshape=(None,), dtype=dt_vlen)
        self.ds_directions = self.f.create_dataset(
            "directions", shape=(0, 3), maxshape=(None, 3), dtype=np.float32)
        self.ds_energies = self.f.create_dataset(
            "energies", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        self.ds_pdg = self.f.create_dataset(
            "pdg", shape=(0,), maxshape=(None,), dtype=np.int32)
        self.ds_actual_pdg = self.f.create_dataset(
            "actual_pdg", shape=(0,), maxshape=(None,), dtype=np.int32)
        self.ds_num_points = self.f.create_dataset(
            "num_points", shape=(0,), maxshape=(None,), dtype=np.int32)
        # shower_id as fixed-length bytes for easy traceability
        self.ds_shower_id = self.f.create_dataset(
            "shower_id", shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"))

        self.count = 0

    def append(self, arr, direction, energy, pdg_class, actual_pdg,
               shower_id=""):
        """Append one shower. arr shape: (n_pts, n_feat)."""
        np = self.np
        n_pts = int(arr.shape[0])
        flat  = arr.astype(np.float32, copy=False).ravel()

        new_n = self.count + 1
        self.ds_showers.resize((new_n,))
        self.ds_directions.resize((new_n, 3))
        self.ds_energies.resize((new_n, 1))
        self.ds_pdg.resize((new_n,))
        self.ds_actual_pdg.resize((new_n,))
        self.ds_num_points.resize((new_n,))
        self.ds_shower_id.resize((new_n,))

        self.ds_showers[self.count]     = flat
        self.ds_directions[self.count]  = np.asarray(direction, dtype=np.float32)[:3]
        self.ds_energies[self.count, 0] = np.float32(energy)
        self.ds_pdg[self.count]         = np.int32(pdg_class)
        self.ds_actual_pdg[self.count]  = np.int32(actual_pdg)
        self.ds_num_points[self.count]  = np.int32(n_pts)
        self.ds_shower_id[self.count]   = str(shower_id)

        self.count = new_n

    def close(self):
        # final shape dataset for convenience: [N_showers, nmax, n_feat]
        import numpy as np
        if "shape" in self.f:
            del self.f["shape"]
        self.f.create_dataset(
            "shape",
            data=np.array([self.count, self.nmax, self.n_feat], dtype=np.int64),
            dtype=np.int64,
        )
        self.f.attrs["n_showers"] = self.count
        self.f.close()


# =============================================================================
# Core: process one chunk (a list of parquet paths)
# =============================================================================

def process_chunk(
    chunk_paths: list[str],
    output_dir: str,
    incident_pdg: int,
    chunk_id: int,
    nmax_per_particle: dict,
    dx_per_particle: dict,
    include_time: bool = True,
    dedup_time_tol: float = 1e-15,
    dedup_energy_rel_tol: float = 1e-6,
    dedup_xy_tol: float = 1e-3,
    do_dedup: bool = True,
    particles: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """Process a list of parquet files → H5 files in output_dir/pdg_<N>/.

    particles: subset of {"electrons","muons","photons"}. If None or empty,
    all three are written.
    """
    np, pd, pa, pq, pc, h5py = _lazy_imports()

    # normalise + validate the particle selection
    active = list(PARTICLE_CONFIGS.keys())
    if particles:
        bad = [p for p in particles if p not in PARTICLE_CONFIGS]
        if bad:
            raise ValueError(f"Unknown particle types: {bad}. "
                             f"Choose from {list(PARTICLE_CONFIGS)}.")
        active = [p for p in PARTICLE_CONFIGS if p in particles]
        if not active:
            raise ValueError("Empty particle selection after filtering.")

    subdir = os.path.join(output_dir, f"pdg_{incident_pdg}")
    os.makedirs(subdir, exist_ok=True)

    result = {
        "incident_pdg":   incident_pdg,
        "chunk_id":       chunk_id,
        "n_parquets":     len(chunk_paths),
        "n_processed":    0,
        "n_skipped":      0,
        "n_dedup_removed": 0,
        "particles":      active,
        "h5_files":       [],
        "h5_total_bytes": 0,
        "status":         "ok",
        "message":        "",
        "elapsed_s":      0.0,
    }
    t_start = time.time()

    n_feat = 5 if include_time else 4

    # --- open H5 writers (one per SELECTED secondary type) ---
    writers = {}
    for pkey in active:
        cfg = PARTICLE_CONFIGS[pkey]
        nmax = int(nmax_per_particle.get(pkey, cfg["default_nmax"]))
        dx   = float(dx_per_particle.get(pkey, cfg["default_dx"]))
        h5_path = os.path.join(
            subdir, f"chunk_{chunk_id:04d}_{cfg['h5_suffix']}.h5")
        writers[pkey] = {
            "writer": ShowerH5Writer(
                path=h5_path, nmax=nmax, n_feat=n_feat, np=np, h5py=h5py,
                extra_attrs={
                    "dx":            dx,
                    "nmax":          nmax,
                    "include_time":  include_time,
                    "incident_pdg":  incident_pdg,
                    "chunk_id":      chunk_id,
                    "particle_type": pkey,
                },
            ),
            "nmax":   nmax,
            "dx":     dx,
            "h5_path": h5_path,
        }

    required_cols = ["x", "y", "pdg", "time", "kinetic_energy", "plane_index"]

    # --- iterate over parquet files ---
    for i, pq_path in enumerate(chunk_paths):
        if not os.path.exists(pq_path):
            result["n_skipped"] += 1
            if verbose:
                print(f"  [{i+1}/{len(chunk_paths)}] MISSING: {pq_path}")
            continue

        try:
            tbl = pq.read_table(pq_path, columns=required_cols)
        except Exception as e:
            result["n_skipped"] += 1
            if verbose:
                print(f"  [{i+1}/{len(chunk_paths)}] READ ERROR {pq_path}: {e}")
            continue

        if len(tbl) == 0:
            result["n_skipped"] += 1
            continue

        # --- event metadata from sibling CSV ---
        event_csv = _event_csv_path_for(pq_path)
        meta      = _extract_event_meta(event_csv)
        actual_pdg_int = int(round(meta["incident_pdg"])) if meta["incident_pdg"] else incident_pdg
        pdg_class_int  = _pdg_class(meta["incident_pdg"] or actual_pdg_int)
        direction      = np.array(
            [meta["direction_x"], meta["direction_y"], meta["direction_z"]],
            dtype=np.float32,
        )
        p_energy  = float(meta["incident_energy"])
        shower_id = meta["shower_id"] or os.path.basename(pq_path).replace("_hits.parquet", "")

        df = tbl.to_pandas()

        # --- dedup ---
        if do_dedup:
            df, n_rm = _drop_near_duplicates(
                df, np, pd,
                time_tol=dedup_time_tol,
                energy_rel_tol=dedup_energy_rel_tol,
                xy_tol=dedup_xy_tol,
                verbose=(verbose and i < 3),  # only log first few
            )
            result["n_dedup_removed"] += n_rm

        # --- split by secondary type and cluster (only for selected particles) ---
        pdg_arr = df["pdg"].to_numpy()
        for pkey in active:
            cfg  = PARTICLE_CONFIGS[pkey]
            mask = np.isin(pdg_arr, cfg["pdg_values"])
            df_sub = df[mask]

            arr = _cluster_hits_for_type(
                df_sub,
                cell_size=writers[pkey]["dx"],
                np=np,
                include_time=include_time,
            )
            arr = _truncate_and_sort(arr, writers[pkey]["nmax"], np)

            writers[pkey]["writer"].append(
                arr=arr,
                direction=direction,
                energy=p_energy,
                pdg_class=pdg_class_int,
                actual_pdg=actual_pdg_int,
                shower_id=shower_id,
            )

        result["n_processed"] += 1

        if verbose and ((i + 1) % 20 == 0 or i + 1 == len(chunk_paths)):
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  [{i+1}/{len(chunk_paths)}] processed "
                  f"({rate:.2f} showers/s, dedup_removed={result['n_dedup_removed']})")

    # --- close all writers ---
    for pkey, w in writers.items():
        w["writer"].close()
        size = os.path.getsize(w["h5_path"])
        result["h5_total_bytes"] += size
        result["h5_files"].append(w["h5_path"])
        if verbose:
            print(f"  wrote {os.path.basename(w['h5_path'])}: "
                  f"{w['writer'].count} showers, {_human_bytes(size)}")

    result["elapsed_s"] = time.time() - t_start
    if result["n_processed"] == 0:
        result["status"]  = "skipped"
        result["message"] = "no parquets processed"
    return result


# =============================================================================
# CLI
# =============================================================================

def _read_chunk_list(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Post-process a CHUNK of combined hit parquets → 3 H5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunk-list", required=True,
                        help="Text file with one parquet path per line.")
    parser.add_argument("--incident-pdg", type=int, required=True,
                        help="Incident PDG this chunk belongs to (used for subdir name).")
    parser.add_argument("--chunk-id", type=int, required=True,
                        help="Zero-padded chunk id (used in output filenames).")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir. H5 files go to <out>/pdg_<PDG>/.")

    # per-particle clustering
    parser.add_argument("--electrons-dx",   type=float,
                        default=PARTICLE_CONFIGS["electrons"]["default_dx"])
    parser.add_argument("--muons-dx",       type=float,
                        default=PARTICLE_CONFIGS["muons"]["default_dx"])
    parser.add_argument("--photons-dx",     type=float,
                        default=PARTICLE_CONFIGS["photons"]["default_dx"])
    parser.add_argument("--electrons-nmax", type=int,
                        default=PARTICLE_CONFIGS["electrons"]["default_nmax"])
    parser.add_argument("--muons-nmax",     type=int,
                        default=PARTICLE_CONFIGS["muons"]["default_nmax"])
    parser.add_argument("--photons-nmax",   type=int,
                        default=PARTICLE_CONFIGS["photons"]["default_nmax"])

    parser.add_argument("--no-time", action="store_true",
                        help="Exclude time feature (store 4 features instead of 5).")

    parser.add_argument("--particles", nargs="+", default=None,
                        choices=list(PARTICLE_CONFIGS.keys()),
                        help="Subset of secondary types to write H5 files for. "
                             "Choose any of: electrons muons photons. "
                             "Omit to process all three.")

    # dedup knobs
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable near-duplicate removal.")
    parser.add_argument("--dedup-time-tol", type=float, default=1e-15,
                        help="Time tolerance (seconds) for near-duplicate match.")
    parser.add_argument("--dedup-energy-rel-tol", type=float, default=1e-6,
                        help="Relative energy tolerance for near-duplicate match.")
    parser.add_argument("--dedup-xy-tol", type=float, default=1e-3,
                        help="xy tolerance (metres) for near-duplicate match.")

    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.chunk_list):
        print(f"ERROR: chunk-list not found: {args.chunk_list}", file=sys.stderr)
        sys.exit(1)

    chunk_paths = _read_chunk_list(args.chunk_list)
    if not chunk_paths:
        print(f"ERROR: chunk-list is empty: {args.chunk_list}", file=sys.stderr)
        sys.exit(1)

    nmax_per_particle = {
        "electrons": args.electrons_nmax,
        "muons":     args.muons_nmax,
        "photons":   args.photons_nmax,
    }
    dx_per_particle = {
        "electrons": args.electrons_dx,
        "muons":     args.muons_dx,
        "photons":   args.photons_dx,
    }

    print(f"Chunk list    : {args.chunk_list}")
    print(f"N parquets    : {len(chunk_paths)}")
    print(f"Incident PDG  : {args.incident_pdg}")
    print(f"Chunk id      : {args.chunk_id:04d}")
    print(f"Output dir    : {args.output_dir}/pdg_{args.incident_pdg}/")
    print(f"Particles     : {args.particles if args.particles else 'electrons muons photons (all)'}")
    print(f"Dedup         : {'OFF' if args.no_dedup else 'ON'} "
          f"(time<{args.dedup_time_tol:.1e}s, dE/E<{args.dedup_energy_rel_tol:.1e}, "
          f"dxy<{args.dedup_xy_tol:.1e}m)")
    print()

    result = process_chunk(
        chunk_paths       = chunk_paths,
        output_dir        = args.output_dir,
        incident_pdg      = args.incident_pdg,
        chunk_id          = args.chunk_id,
        nmax_per_particle = nmax_per_particle,
        dx_per_particle   = dx_per_particle,
        include_time      = not args.no_time,
        dedup_time_tol    = args.dedup_time_tol,
        dedup_energy_rel_tol = args.dedup_energy_rel_tol,
        dedup_xy_tol      = args.dedup_xy_tol,
        particles         = args.particles,
        do_dedup          = not args.no_dedup,
        verbose           = not args.quiet,
    )

    print()
    print(f"Status        : {result['status']}")
    print(f"Processed     : {result['n_processed']}/{result['n_parquets']} showers")
    print(f"Skipped       : {result['n_skipped']}")
    print(f"Dedup removed : {result['n_dedup_removed']} hits (total across all showers)")
    print(f"H5 written    : {_human_bytes(result['h5_total_bytes'])} "
          f"({len(result['h5_files'])} files)")
    print(f"Elapsed       : {result['elapsed_s']:.1f} s")
    if result["message"]:
        print(f"Detail        : {result['message']}")

    sys.exit(0 if result["status"] in ("ok", "partial") else 1)


if __name__ == "__main__":
    main()