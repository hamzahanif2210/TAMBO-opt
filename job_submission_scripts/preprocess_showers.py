#!/usr/bin/env python3
"""
Post-process one TAMBO simulation directory into:

1. ONE combined Parquet file
   - hit-level rows (one row per particle hit)
   - includes shower_id
   - event-level metadata stored in Parquet file metadata (not repeated per row)

Example:
    python preprocess_showers_split.py \
        --sim-dir /path/to/5092721_2

Assumptions / behavior:
- shower_id is taken from the base folder name of sim_dir
  e.g. /.../5092721_2  ->  shower_id = "5092721_2"
- Reads all particles* folders across planes
- Keeps only e±, μ±, γ hits
- Converts local plane coordinates -> global x,y,z
- Deletes ALL particles.parquet files in sim_dir and subdirectories before writing output
- Saves one combined hit-level parquet with event metadata embedded

python /n/home04/hhanif/TAMBO-opt/job_submission_scripts/submit_jobs.py --jobs 3 --sims-per-job 2 --pdg -211 --gamma 1.5 --submit-sleep 0.3 --postprocess
"""

import argparse
import math
import os
import re
import sys
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

ALL_PDG_VALUES = [11, -11, 13, -13, 22]
PDG_REMAP = {11: 11, -11: 11, 13: 13, -13: 13, 22: 22}

HIT_COLUMNS = [
    "shower_id",
    "x",
    "y",
    "z",
    # "nx",
    # "ny",
    # "nz",
    "pdg",
    "time",
    "kinetic_energy",
    "weight",
    "plane_index",
    "z_depth",
]

EVENT_COLUMNS = [
    "shower_id",
    "incident_energy",
    "incident_zenith",
    "incident_azimuth",
    "incident_class_id",
    "incident_x",
    "incident_y",
    "incident_z",
    "direction_x",
    "direction_y",
    "direction_z",
    "incident_pdg",
    "z_depth_start",
    "z_depth_step",
    "created",
    "sim_dir",
]

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


# =============================================================================
# Lazy imports
# =============================================================================

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import yaml
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    return np, pd, yaml, pa, pq, pc


# =============================================================================
# Small helpers
# =============================================================================

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def _parse_args_string(args_str: str) -> dict:
    import shlex

    tokens = shlex.split(args_str) if isinstance(args_str, str) else []
    arg_dict = {}
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            if "=" in tok[2:]:
                key, val = tok[2:].split("=", 1)
                arg_dict[key] = val
            else:
                key = tok[2:]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    arg_dict[key] = tokens[i + 1]
                    i += 1
                else:
                    arg_dict[key] = True
        i += 1

    return arg_dict


def _regex_flag_float(args_str, names):
    if not isinstance(args_str, str):
        return None

    for name in names:
        m = re.search(rf"--{re.escape(name)}(?:\s*=\s*|\s+)\s*({_NUM})", args_str)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    return None


def _to_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _get_float_flag(args_str, arg_dict, keys, default=0.0):
    for k in keys:
        if k in arg_dict:
            v = _to_float(arg_dict[k], None)
            if v is not None:
                return v

    v = _regex_flag_float(args_str, keys)
    return default if v is None else v


def _radians_sin_cos(angle: float):
    """Auto-convert degrees -> radians if value looks like degrees."""
    if math.isfinite(angle) and abs(angle) > 2 * math.pi + 1e-6:
        angle = math.radians(angle)
    return math.sin(angle), math.cos(angle)


def _build_direction(zenith: float, azimuth: float, np):
    """
    Unit direction vector using:
        nx = sin(theta) * cos(phi)
        ny = sin(theta) * sin(phi)
        nz = cos(theta)
    """
    sin_zen, cos_zen = _radians_sin_cos(zenith)
    sin_azi, cos_azi = _radians_sin_cos(azimuth)
    return np.array(
        [sin_zen * cos_azi, sin_zen * sin_azi, cos_zen],
        dtype=np.float32,
    )


def _incident_class_id(pdg_primary: float) -> int:
    """
    Binary class label:
      0 = e±/γ/π0-like
      1 = π±
    """
    return 1 if abs(int(round(pdg_primary))) == 211 else 0


def _get_plane_index(folder_name: str) -> int:
    """
    particles1  -> plane 0
    particles2  -> plane 1
    ...
    particles   -> plane 23  (last plane, no number suffix)
    """
    if folder_name == "particles":
        return 23

    m = re.match(r"^particles(\d+)$", folder_name)
    if m:
        return int(m.group(1)) - 1

    return -1


def _apply_pdg_mask(tbl, pdg_values, pc):
    masks = [pc.equal(tbl["pdg"], v) for v in pdg_values]
    combined = masks[0]
    for m in masks[1:]:
        combined = pc.or_(combined, m)
    return tbl.filter(combined)


def _read_yaml_if_exists(path):
    _, _, yaml, _, _, _ = _lazy_imports()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _read_plane_normal_from_cfg(cfg: dict) -> tuple[float, float, float]:
    """
    Tries to read:
        plane:
          normal: [nx, ny, nz]
    """
    try:
        normal = cfg["plane"]["normal"]
        return float(normal[0]), float(normal[1]), float(normal[2])
    except Exception:
        return 0.0, 0.0, 0.0


def _get_shower_id(sim_dir: str) -> str:
    """Use base folder name as shower_id, e.g. .../5092721_2 -> 5092721_2"""
    return os.path.basename(os.path.normpath(sim_dir))


# =============================================================================
# Raw parquet cleanup
# =============================================================================

def _remove_all_raw_parquets(sim_dir: str, verbose: bool = True) -> tuple[int, int]:
    """
    Walk sim_dir recursively and delete every particles.parquet file found.
    Returns (files_removed, bytes_removed).
    """
    files_removed = 0
    bytes_removed = 0

    for dirpath, _dirnames, filenames in os.walk(sim_dir):
        for fname in filenames:
            if fname == "particles.parquet":
                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                    os.remove(full_path)
                    bytes_removed += size
                    files_removed += 1
                    if verbose:
                        print(f"  deleted  : {os.path.relpath(full_path, sim_dir)}")
                except OSError as e:
                    if verbose:
                        print(f"  WARNING  : could not delete {full_path}: {e}")

    return files_removed, bytes_removed


# =============================================================================
# PDG registry helpers
# =============================================================================

# Maps every supported PDG value to its canonical registry filename.
_PDG_REGISTRY_FILES = {
    -11:  "registry_pdg_-11.txt",
     11:  "registry_pdg_11.txt",
    111:  "registry_pdg_111.txt",
    211:  "registry_pdg_211.txt",
   -211:  "registry_pdg_-211.txt",
}


def _get_registry_path(base_output_dir: str, incident_pdg: int) -> str | None:
    """
    Return the path to the per-PDG registry .txt file that lives in
    BASE_OUTPUT_DIR.  Returns None if the PDG is not in the known set.
    """
    fname = _PDG_REGISTRY_FILES.get(incident_pdg)
    if fname is None:
        return None
    return os.path.join(base_output_dir, fname)


def _append_to_registry(registry_path: str, parquet_path: str) -> None:
    """
    Append *parquet_path* as a new line to the registry file, using an
    exclusive fcntl lock so that parallel SLURM tasks writing to the same
    file do not corrupt it.
    """
    import fcntl

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(parquet_path + "\n")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _write_event_csv(csv_path: str, incident_meta: dict,
                     incident_x: float, incident_y: float, incident_z: float,
                     direction: "np.ndarray",
                     z_depth_start: float, z_depth_step: float,
                     shower_id: str, sim_dir: str) -> None:
    """
    Write a single-row CSV whose columns are exactly EVENT_COLUMNS.
    This makes the event-level metadata available as a flat file alongside
    the hit-level parquet (the parquet only stores these in its key-value
    schema metadata, not as row columns).
    """
    import csv

    row = {
        "shower_id":         shower_id,
        "incident_energy":   incident_meta["incident_energy"],
        "incident_zenith":   incident_meta["incident_zenith"],
        "incident_azimuth":  incident_meta["incident_azimuth"],
        "incident_class_id": incident_meta["incident_class_id"],
        "incident_x":        incident_x,
        "incident_y":        incident_y,
        "incident_z":        incident_z,
        "direction_x":       float(direction[0]),
        "direction_y":       float(direction[1]),
        "direction_z":       float(direction[2]),
        "incident_pdg":      incident_meta["incident_pdg"],
        "z_depth_start":     z_depth_start,
        "z_depth_step":      z_depth_step,
        "created":           datetime.now().isoformat(),
        "sim_dir":           sim_dir,
    }

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Metadata readers
# =============================================================================

def _find_best_global_config(sim_dir: str) -> dict:
    """
    Try several likely locations for the run-level config.
    Priority:
      1) sim_dir/config.yaml
      2) sim_dir/profile/config.yaml
      3) sim_dir/primary/config.yaml
    """
    candidates = [
        os.path.join(sim_dir, "config.yaml"),
        os.path.join(sim_dir, "profile", "config.yaml"),
        os.path.join(sim_dir, "primary", "config.yaml"),
    ]

    for path in candidates:
        cfg = _read_yaml_if_exists(path)
        if cfg:
            return cfg

    return {}


def _find_plane_normal(sim_dir: str) -> tuple[float, float, float]:
    """
    Try likely locations for plane normal.
    """
    candidates = [
        os.path.join(sim_dir, "primary", "config.yaml"),
        os.path.join(sim_dir, "profile", "config.yaml"),
        os.path.join(sim_dir, "config.yaml"),
    ]

    for path in candidates:
        cfg = _read_yaml_if_exists(path)
        if cfg:
            x, y, z = _read_plane_normal_from_cfg(cfg)
            if (x, y, z) != (0.0, 0.0, 0.0):
                return x, y, z

    return 0.0, 0.0, 0.0


def _extract_incident_metadata(sim_dir: str, fallback_cfg: dict | None = None) -> dict:
    """
    Read incident-level metadata from the best available config args string.
    """
    global_cfg = _find_best_global_config(sim_dir)
    args_str = global_cfg.get("args", "")

    if not args_str and fallback_cfg:
        args_str = fallback_cfg.get("args", "")

    arg_dict = _parse_args_string(args_str)

    incident_energy  = _get_float_flag(args_str, arg_dict, ["energy"],  default=0.0)
    incident_zenith  = _get_float_flag(args_str, arg_dict, ["zenith"],  default=0.0)
    incident_azimuth = _get_float_flag(args_str, arg_dict, ["azimuth"], default=0.0)
    incident_pdg     = int(round(_get_float_flag(args_str, arg_dict, ["pdg"], default=0.0)))

    return {
        "incident_energy":   float(incident_energy),
        "incident_zenith":   float(incident_zenith),
        "incident_azimuth":  float(incident_azimuth),
        "incident_pdg":      int(incident_pdg),
        "incident_class_id": int(_incident_class_id(incident_pdg)),
    }


# =============================================================================
# Per-plane processing
# =============================================================================

def _process_particle_folder(
    folder_path,
    plane_idx,
    sim_dir,
    shower_id,
    pdg_values,
    z_depth_start,
    z_depth_step,
    np, pa, pq, pc, yaml,
):
    """
    Read one particles* folder and return a hit-level Arrow table.
    """
    parquet_path = os.path.join(folder_path, "particles.parquet")
    config_path  = os.path.join(folder_path, "config.yaml")

    if not os.path.exists(parquet_path):
        return None, None, "missing_parquet"
    if not os.path.exists(config_path):
        return None, None, "missing_config"

    required_cols = ["x", "y", "pdg", "time", "kinetic_energy", "nx", "ny", "nz", "weight"]

    try:
        tbl = pq.read_table(parquet_path, columns=required_cols)
    except Exception as e:
        return None, None, f"read_error: {e}"

    if len(tbl) == 0:
        return None, None, "empty_data"

    missing = set(required_cols) - set(tbl.schema.names)
    if missing:
        return None, None, f"missing_columns: {missing}"

    tbl = _apply_pdg_mask(tbl, pdg_values, pc)
    if len(tbl) == 0:
        return None, None, "no_valid_pdg"

    try:
        with open(config_path) as f:
            particles_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return None, None, f"bad_config: {e}"

    # Geometry for local -> global transform
    # Rotation matrix rows are the plane's basis vectors (xhat, yhat, zhat).
    # Each hit has local coords (x_loc, y_loc, 0) since it lies on the plane.
    # Global position = mat.T @ [x, y, 0] + center
    try:
        center = np.array(particles_cfg["plane"]["center"], dtype=np.float64)
        zhat   = np.array(particles_cfg["plane"]["normal"], dtype=np.float64)
        xhat   = np.array(particles_cfg["x-axis"],          dtype=np.float64)
        yhat   = np.array(particles_cfg["y-axis"],          dtype=np.float64)
    except Exception as e:
        return None, particles_cfg, f"bad_geometry: {e}"

    # mat shape: (3, 3) — rows are xhat, yhat, zhat
    mat   = np.array([xhat, yhat, zhat], dtype=np.float64)

    x_loc = np.asarray(tbl["x"].to_pylist(), dtype=np.float64)
    y_loc = np.asarray(tbl["y"].to_pylist(), dtype=np.float64)
    n     = len(x_loc)

    # local_coords shape: (3, N) — z component is 0 (hits lie on the plane)
    local_coords = np.zeros((3, n), dtype=np.float64)
    local_coords[0] = x_loc
    local_coords[1] = y_loc

    # mat.T @ local_coords: (3, 3).T @ (3, N) = (3, N), then broadcast center
    xyz = mat.T @ local_coords + center[:, None]  # shape (3, N)

    raw_pdg  = np.asarray(tbl["pdg"].to_pylist(), dtype=np.int32)
    remapped = np.array([PDG_REMAP.get(int(v), abs(int(v))) for v in raw_pdg], dtype=np.int32)

    z_depth_val = float(z_depth_start + plane_idx * z_depth_step)

    hit_table = pa.table({
        "shower_id":      pa.array([shower_id] * n,                           type=pa.string()),
        "x":              pa.array(xyz[0],                                     type=pa.float64()),
        "y":              pa.array(xyz[1],                                     type=pa.float64()),
        "z":              pa.array(xyz[2],                                     type=pa.float64()),
        "nx":             tbl["nx"],
        "ny":             tbl["ny"],
        "nz":             tbl["nz"],
        "pdg":            pa.array(remapped,                                   type=pa.int32()),
        "time":           tbl["time"],
        "kinetic_energy": tbl["kinetic_energy"],
        "weight":         tbl["weight"],
        "plane_index":    pa.array(np.full(n, plane_idx,   dtype=np.int32),   type=pa.int32()),
        "z_depth":        pa.array(np.full(n, z_depth_val, dtype=np.float64), type=pa.float64()),
    })

    return hit_table, particles_cfg, "ok"


# =============================================================================
# Main processing
# =============================================================================

def postprocess_simulation(
    sim_dir: str,
    z_depth_start: float = 500.0,
    z_depth_step: float  = 500.0,
    verbose: bool = True,
    base_output_dir: str | None = None,
) -> dict:
    np, pd, yaml, pa, pq, pc = _lazy_imports()

    result = {
        "status":              "ok",
        "message":             "",
        "shower_id":           _get_shower_id(sim_dir),
        "input_bytes_removed": 0,
        "hits_out_bytes":      0,
        "hits_out_file":       None,
        "event_csv_file":      None,
        "registry_file":       None,
    }

    shower_id = result["shower_id"]

    if not os.path.isdir(sim_dir):
        result["status"]  = "error"
        result["message"] = f"sim_dir does not exist: {sim_dir}"
        return result

    if not os.path.exists(os.path.join(sim_dir, "summary.yaml")):
        result["status"]  = "skipped"
        result["message"] = "no summary.yaml"
        return result

    # ------------------------------------------------------------------
    # Discover particles* folders
    # ------------------------------------------------------------------
    particle_folders = []
    try:
        for item in os.listdir(sim_dir):
            full_path = os.path.join(sim_dir, item)
            if os.path.isdir(full_path) and re.match(r"^particles(\d+)?$", item):
                pidx = _get_plane_index(item)
                if pidx >= 0:
                    particle_folders.append((pidx, full_path))
    except OSError as e:
        result["status"]  = "error"
        result["message"] = f"cannot list sim_dir: {e}"
        return result

    if not particle_folders:
        result["status"]  = "skipped"
        result["message"] = "no particles* folders found"
        return result

    particle_folders.sort(key=lambda x: x[0])

    # Count raw bytes that will be removed
    for _, folder_path in particle_folders:
        pq_path = os.path.join(folder_path, "particles.parquet")
        if os.path.exists(pq_path):
            result["input_bytes_removed"] += os.path.getsize(pq_path)

    sim_name      = os.path.basename(os.path.normpath(sim_dir))
    hits_out_path = os.path.join(sim_dir, f"{sim_name}_hits.parquet")

    # Read plane normal once (used as incident_x/y/z)
    incident_x, incident_y, incident_z = _find_plane_normal(sim_dir)

    # ------------------------------------------------------------------
    # Process each plane
    # ------------------------------------------------------------------
    tables      = []
    fallback_cfg = None

    for pidx, folder_path in particle_folders:
        table, cfg, status = _process_particle_folder(
            folder_path   = folder_path,
            plane_idx     = pidx,
            sim_dir       = sim_dir,
            shower_id     = shower_id,
            pdg_values    = ALL_PDG_VALUES,
            z_depth_start = z_depth_start,
            z_depth_step  = z_depth_step,
            np=np, pa=pa, pq=pq, pc=pc, yaml=yaml,
        )

        if cfg and fallback_cfg is None:
            fallback_cfg = cfg

        if status == "ok" and table is not None:
            tables.append(table)

        if verbose:
            z_val = z_depth_start + pidx * z_depth_step
            label = os.path.basename(folder_path)
            n_str = f"hits={len(table):>7,d}" if (status == "ok" and table is not None) else status
            print(f"  plane {pidx:>2d} ({label:>11s})  z={z_val:>6.0f} m  {n_str}")

    if not tables:
        result["status"]  = "skipped"
        result["message"] = "no valid data across all planes"
        return result

    # ------------------------------------------------------------------
    # Incident metadata
    # ------------------------------------------------------------------
    incident_meta = _extract_incident_metadata(sim_dir, fallback_cfg=fallback_cfg)
    direction = _build_direction(
        incident_meta["incident_zenith"],
        incident_meta["incident_azimuth"],
        np,
    )

    # ------------------------------------------------------------------
    # Remove ALL particles.parquet files recursively BEFORE writing output
    # ------------------------------------------------------------------
    if verbose:
        print("\n  Removing raw parquet files...")
    files_removed, bytes_removed = _remove_all_raw_parquets(sim_dir, verbose=verbose)
    result["input_bytes_removed"] = bytes_removed  # update with actual recursive total
    if verbose:
        print(f"  removed {files_removed} file(s) ({_human_bytes(bytes_removed)})")

    # ------------------------------------------------------------------
    # Combine hit tables and write single output parquet
    # ------------------------------------------------------------------
    combined_hits = (
        pa.concat_tables(tables)
        .to_pandas()
        .sort_values(["plane_index"], ascending=True)
        .reset_index(drop=True)
    )

    n_hits = len(combined_hits)

    try:
        hits_df = combined_hits[HIT_COLUMNS].copy()

        # Dtype optimisation
        hits_df["x"]              = hits_df["x"].astype("float32")
        hits_df["y"]              = hits_df["y"].astype("float32")
        hits_df["z"]              = hits_df["z"].astype("float32")
        hits_df["time"]           = hits_df["time"].astype("float32")
        hits_df["kinetic_energy"] = hits_df["kinetic_energy"].astype("float32")
        hits_df["z_depth"]        = hits_df["z_depth"].astype("float32")
        hits_df["pdg"]            = hits_df["pdg"].astype("int32")
        hits_df["plane_index"]    = hits_df["plane_index"].astype("int32")

        # Embed all event-level metadata into the file's key-value metadata
        zen = incident_meta["incident_zenith"]
        azi = incident_meta["incident_azimuth"]

        file_meta = {
            b"shower_id":         shower_id.encode(),
            b"incident_energy":   str(incident_meta["incident_energy"]).encode(),
            b"incident_zenith":   str(zen).encode(),
            b"incident_azimuth":  str(azi).encode(),
            b"incident_class_id": str(incident_meta["incident_class_id"]).encode(),
            b"incident_x":        str(incident_x).encode(),
            b"incident_y":        str(incident_y).encode(),
            b"incident_z":        str(incident_z).encode(),
            b"direction_x":       str(float(direction[0])).encode(),
            b"direction_y":       str(float(direction[1])).encode(),
            b"direction_z":       str(float(direction[2])).encode(),
            b"incident_pdg":      str(incident_meta["incident_pdg"]).encode(),
            b"z_depth_start":     str(z_depth_start).encode(),
            b"z_depth_step":      str(z_depth_step).encode(),
            b"created":           datetime.now().isoformat().encode(),
            b"sim_dir":           sim_dir.encode(),
        }

        hits_table = pa.Table.from_pandas(hits_df, preserve_index=False)
        existing   = hits_table.schema.metadata or {}
        hits_table = hits_table.replace_schema_metadata({**existing, **file_meta})

        pq.write_table(hits_table, hits_out_path, compression="zstd",compression_level=10)

        result["hits_out_file"]  = hits_out_path
        result["hits_out_bytes"] = os.path.getsize(hits_out_path)

        # ------------------------------------------------------------------
        # Write EVENT_COLUMNS as a single-row CSV next to the parquet.
        # (The parquet stores these only in schema key-value metadata, not
        #  as row-level columns, so the CSV makes them readily accessible.)
        # ------------------------------------------------------------------
        event_csv_path = hits_out_path.replace("_hits.parquet", "_event.csv")
        try:
            _write_event_csv(
                csv_path       = event_csv_path,
                incident_meta  = incident_meta,
                incident_x     = incident_x,
                incident_y     = incident_y,
                incident_z     = incident_z,
                direction      = direction,
                z_depth_start  = z_depth_start,
                z_depth_step   = z_depth_step,
                shower_id      = shower_id,
                sim_dir        = sim_dir,
            )
            result["event_csv_file"] = event_csv_path
            if verbose:
                print(f"  event CSV        : {os.path.basename(event_csv_path)}")
        except Exception as e:
            if verbose:
                print(f"  WARNING: could not write event CSV: {e}")

        # ------------------------------------------------------------------
        # Append the parquet path to the per-PDG registry in BASE_OUTPUT_DIR.
        # Uses fcntl locking so parallel SLURM tasks are safe.
        # ------------------------------------------------------------------
        if base_output_dir is not None:
            registry_path = _get_registry_path(base_output_dir, incident_meta["incident_pdg"])
            if registry_path is not None:
                try:
                    _append_to_registry(registry_path, hits_out_path)
                    result["registry_file"] = registry_path
                    if verbose:
                        print(f"  registry         : {os.path.basename(registry_path)}")
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: could not update registry: {e}")

    except Exception as e:
        result["status"]  = "error"
        result["message"] = f"failed writing hit parquet: {e}"
        return result

    if verbose:
        zen_deg = math.degrees(zen) if abs(zen) <= 2 * math.pi + 1e-6 else zen
        azi_deg = math.degrees(azi) if abs(azi) <= 2 * math.pi + 1e-6 else azi

        print("\n  ── Summary ──────────────────────────────────────────────────")
        print(f"  shower_id        : {shower_id}")
        print(f"  hit file         : {os.path.basename(hits_out_path)}")
        print(f"  hits             : {n_hits:,}")
        print(f"  incident_pdg     : {incident_meta['incident_pdg']}")
        print(f"  incident_class_id: {incident_meta['incident_class_id']}")
        print(f"  incident_energy  : {incident_meta['incident_energy']:.3e} GeV")
        print(f"  incident_zenith  : {zen_deg:.2f}°")
        print(f"  incident_azimuth : {azi_deg:.2f}°")
        print(f"  plane normal     : [{incident_x:.3f}, {incident_y:.3f}, {incident_z:.3f}]")
        print(f"  direction vec    : [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
        print(f"  hits size        : {_human_bytes(result['hits_out_bytes'])}")
        print(f"  removed raw      : {_human_bytes(result['input_bytes_removed'])}")
        print("  ─────────────────────────────────────────────────────────────")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process TAMBO shower output into a single hit-level Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim-dir",
        required=True,
        help="Path to one simulation directory, e.g. .../5092721_2",
    )
    parser.add_argument(
        "--z-depth-start",
        type=float,
        default=500.0,
        help="z_depth for plane 0 / particles1 (m)",
    )
    parser.add_argument(
        "--z-depth-step",
        type=float,
        default=500.0,
        help="z_depth spacing between planes (m)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging",
    )
    parser.add_argument(
        "--base-output-dir",
        default=None,
        help=(
            "Root output directory (same as BASE_OUTPUT_DIR in submit_jobs.py). "
            "When provided, the path of the written parquet is appended to a "
            "per-PDG registry file, e.g. registry_pdg_-11.txt, inside this dir."
        ),
    )

    args    = parser.parse_args()
    sim_dir = os.path.normpath(args.sim_dir)

    if not os.path.isdir(sim_dir):
        print(f"ERROR: sim_dir does not exist: {sim_dir}", file=sys.stderr)
        sys.exit(1)

    shower_id = _get_shower_id(sim_dir)

    print(f"Processing : {sim_dir}")
    print(f"shower_id  : {shower_id}")
    print(
        f"z_depth    : plane 0 = {args.z_depth_start:.0f} m, "
        f"step = {args.z_depth_step:.0f} m"
    )
    print(f"hit cols   : {HIT_COLUMNS}")
    print()

    result = postprocess_simulation(
        sim_dir         = sim_dir,
        z_depth_start   = args.z_depth_start,
        z_depth_step    = args.z_depth_step,
        verbose         = not args.quiet,
        base_output_dir = args.base_output_dir,
    )

    print(f"\nStatus      : {result['status']}")
    print(f"Shower ID   : {result['shower_id']}")
    print(f"Input freed : {_human_bytes(result['input_bytes_removed'])}")
    print(f"Hit output  : {_human_bytes(result['hits_out_bytes'])}")

    if result["hits_out_file"]:
        print(f"Hits file   : {result['hits_out_file']}")
    if result["event_csv_file"]:
        print(f"Event CSV   : {result['event_csv_file']}")
    if result["registry_file"]:
        print(f"Registry    : {result['registry_file']}")
    if result["message"]:
        print(f"Detail      : {result['message']}")

    sys.exit(0 if result["status"] in ("ok", "partial") else 1)


if __name__ == "__main__":
    main()