#!/usr/bin/env python3
"""
Post-process one TAMBO simulation directory: parquet → HDF5.

Called automatically by submit_jobs.py after each simulation, or manually:

  python preprocess_showers.py --sim-dir /path/to/sim_dir \
      --electrons-dx 10.0 --muons-dx 10.0 --photons-dx 10.0 \
      --electrons-nmax 2048 --muons-nmax 28032 --photons-nmax 6016

For each of the three secondary particle types (electrons, muons, photons):
  1. Reads all particles* sub-folders
  2. Applies PDG filter, global coordinate transform, 90th-pct radius pre-filter
  3. Clusters hits into (dx × dx) cells per plane (independent per plane)
  4. Truncates to the top-nmax highest-energy clustered points
  5. Writes one HDF5 file with datasets:
       showers     vlen float32 (1,)   flat → reshape(num_points, n_feat)
                                       columns: x, y, z(plane 0-23), energy, time
       directions  float32      (1,3)  [nx, ny, nz] primary particle direction
       energies    float32      (1,)   primary kinetic energy
       pdg         int32        (1,)   class: 0=e±/γ/π⁰  1=π±
       actual_pdg  int32        (1,)   true primary PDG ID
       shape       int64        (3,)   [1, nmax, n_feat]
       num_points  int64        (1,)   actual clustered points stored

After all three H5 files are written the parquet files are deleted.
"""

import argparse
import math
import os
import re
import sys
from datetime import datetime


# =============================================================================
# ── Particle type configuration ───────────────────────────────────────────────
# =============================================================================

PARTICLE_CONFIGS = {
    "electrons": {
        "pdg_values":   [11, -11],
        "label":        "e± (pdg ±11)",
        "default_nmax": 4000,
        "default_dx":   10.0,
        "h5_suffix":    "electrons",
    },
    "muons": {
        "pdg_values":   [13, -13],
        "label":        "μ± (pdg ±13)",
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
# ── Imports (lazy — only loaded when actually processing) ─────────────────────
# =============================================================================

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import yaml
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import h5py
    return np, pd, yaml, pa, pq, pc, h5py


# =============================================================================
# ── Config / geometry helpers ─────────────────────────────────────────────────
# =============================================================================

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _parse_args_string(args_str: str) -> dict:
    import shlex
    tokens, arg_dict, i = shlex.split(args_str) if isinstance(args_str, str) else [], {}, 0
    tokens = shlex.split(args_str) if isinstance(args_str, str) else []
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


# =============================================================================
# ── Plane index ───────────────────────────────────────────────────────────────
# =============================================================================

def _get_plane_index(folder_name: str) -> int:
    """
    particles  → plane 23  (the single no-suffix folder = deepest plane)
    particles1 → plane 0
    particles2 → plane 1
    ...
    """
    if folder_name == "particles":
        return 23
    m = re.match(r"^particles(\d+)$", folder_name)
    if m:
        return int(m.group(1)) - 1
    return -1


# =============================================================================
# ── PDG mask ──────────────────────────────────────────────────────────────────
# =============================================================================

def _apply_pdg_mask(tbl, pdg_values, pc):
    masks    = [pc.equal(tbl["pdg"], v) for v in pdg_values]
    combined = masks[0]
    for m in masks[1:]:
        combined = pc.or_(combined, m)
    return tbl.filter(combined)


# =============================================================================
# ── Spatial clustering ────────────────────────────────────────────────────────
# =============================================================================

def _cluster_shower_part(xy, e, cell_size, shift, np, t=None):
    """
    Bin particles into (cell_size × cell_size) grid cells for one detector plane.

      - Energy  : summed per cell  (np.add.at)
      - Time    : earliest arrival per cell  (np.minimum.at)
      - Position: cell centre in metres

    Returns (xy_clustered, e_clustered, t_clustered | None).
    """
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

    x_min, y_min = int(xy_idx[:, 0].min()), int(xy_idx[:, 1].min())
    stride = int(xy_idx[:, 0].max()) - x_min + 2   # +1 for range, +1 for stride
    keys   = (xy_idx[:, 0].astype(np.int64) - x_min) * stride + \
             (xy_idx[:, 1].astype(np.int64) - y_min)

    unique_keys, inverse_idx = np.unique(keys, return_inverse=True)
    unique_x = (unique_keys // stride).astype(np.int32) + x_min
    unique_y = (unique_keys  % stride).astype(np.int32) + y_min

    # Sum energies
    e_clustered = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(e_clustered, inverse_idx, e)

    # Earliest arrival time
    if t is not None:
        t_clustered = np.full(len(unique_keys), np.inf, dtype=np.float32)
        np.minimum.at(t_clustered, inverse_idx, t)
    else:
        t_clustered = None

    # Re-centre positions back to metres
    xy_clustered = np.column_stack([unique_x, unique_y]).astype(np.float32)
    xy_clustered += 0.5          # corner → centre (cell-index units)
    xy_clustered *= cell_size    # → metres
    xy_clustered -= shift        # undo random shift
    return xy_clustered, e_clustered, t_clustered


def _cluster_hits(df, cell_size, np, random_shift=False, include_time=True):
    """
    Cluster all particles in *df* per detector plane independently.

    Returns a DataFrame with columns:
        x, y, plane_idx (z = 0-23), energy, [time], p_energy
    """
    if len(df) == 0:
        cols = ["x", "y", "plane_idx", "energy", "p_energy"]
        if include_time:
            cols.insert(4, "time")
        import pandas as pd
        return pd.DataFrame(columns=cols)

    pos_all      = df[["x", "y", "plane_index"]].to_numpy(dtype=np.float32)
    e_all        = np.maximum(df["kinetic_energy"].to_numpy(dtype=np.float32), 0.0)
    t_all        = df["time"].to_numpy(dtype=np.float32) \
                   if include_time and "time" in df.columns else None
    p_energy_val = float(df["p_energy"].iloc[0]) if "p_energy" in df.columns else 0.0
    plane_col    = pos_all[:, 2].astype(np.int32)

    parts_pos, parts_e, parts_t = [], [], []

    for plane in np.unique(plane_col):
        mask = plane_col == plane
        xy_p = pos_all[mask, :2]
        e_p  = e_all[mask]
        t_p  = t_all[mask] if t_all is not None else None

        # Random shift disabled for post-processing (deterministic output).
        # Enable only during training-time augmentation in the dataloader.
        shift = (
            (np.random.rand(2).astype(np.float32) * cell_size) - cell_size / 2
            if random_shift else np.array([0.0, 0.0], dtype=np.float32)
        )

        xy_c, ec, tc = _cluster_shower_part(xy_p, e_p, cell_size, shift, np, t=t_p)
        parts_pos.append(np.column_stack([xy_c, np.full(len(xy_c), plane, dtype=np.float32)]))
        parts_e.append(ec)
        if t_all is not None:
            parts_t.append(tc)

    if not parts_pos:
        import pandas as pd
        cols = ["x", "y", "plane_idx", "energy", "p_energy"]
        if include_time:
            cols.insert(4, "time")
        return pd.DataFrame(columns=cols)

    import pandas as pd
    pos_c = np.concatenate(parts_pos, axis=0)
    e_c   = np.concatenate(parts_e,   axis=0)
    out   = {
        "x":         pos_c[:, 0],
        "y":         pos_c[:, 1],
        "plane_idx": pos_c[:, 2].astype(np.int32),
        "energy":    e_c,
        "p_energy":  np.full(len(e_c), p_energy_val, dtype=np.float32),
    }
    if t_all is not None and parts_t:
        out["time"] = np.concatenate(parts_t, axis=0)
    return pd.DataFrame(out)


# =============================================================================
# ── Per-folder reader ─────────────────────────────────────────────────────────
# =============================================================================

def _process_particle_folder(folder_path, plane_idx, run_dir, pdg_values,
                              np, pa, pq, pc, yaml):
    """
    Read one particles* folder:
      1. Load parquet, apply PDG filter
      2. Read geometry from config.yaml
      3. Transform local (x,y) → global (x,y,z) using plane rotation matrix
      4. Apply 90th-percentile radius pre-filter (removes distant outliers)
      5. Return a PyArrow table with global coords + metadata columns

    Returns: (table, p_energy, zenith, azimuth, pdg_primary, status_str)
    """
    parquet_path     = os.path.join(folder_path, "particles.parquet")
    config_path      = os.path.join(folder_path, "config.yaml")
    base_config_path = os.path.join(run_dir,     "config.yaml")

    if not os.path.exists(parquet_path):
        return None, None, None, None, None, "missing_parquet"
    if not os.path.exists(config_path):
        return None, None, None, None, None, "missing_config"

    _REQUIRED = ["x", "y", "pdg", "time", "kinetic_energy"]
    try:
        tbl = pq.read_table(parquet_path, columns=_REQUIRED)
    except Exception as e:
        return None, None, None, None, None, f"read_error: {e}"

    if len(tbl) == 0:
        return None, None, None, None, None, "empty_data"

    missing = set(_REQUIRED) - set(tbl.schema.names)
    if missing:
        return None, None, None, None, None, f"missing_columns: {missing}"

    tbl = _apply_pdg_mask(tbl, pdg_values, pc)
    if len(tbl) == 0:
        return None, None, None, None, None, "no_valid_pdg"

    # ── Read configs ──────────────────────────────────────────────────────────
    try:
        with open(config_path) as f:
            particles_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        return None, None, None, None, None, f"bad_config: {e}"

    base_cfg = {}
    try:
        if os.path.exists(base_config_path):
            with open(base_config_path) as f:
                base_cfg = yaml.safe_load(f) or {}
    except Exception:
        pass

    # Primary particle metadata comes from the base config args string
    args_str    = base_cfg.get("args") or particles_cfg.get("args") or ""
    arg_dict    = _parse_args_string(args_str)
    p_energy    = _get_float_flag(args_str, arg_dict, ["energy"],  default=0.0)
    zenith      = _get_float_flag(args_str, arg_dict, ["zenith"],  default=0.0)
    azimuth     = _get_float_flag(args_str, arg_dict, ["azimuth"], default=0.0)
    pdg_primary = _get_float_flag(args_str, arg_dict, ["pdg"],     default=0.0)

    # ── Plane geometry ────────────────────────────────────────────────────────
    try:
        center = np.array(particles_cfg["plane"]["center"])
        zhat   = np.array(particles_cfg["plane"]["normal"])
        xhat   = np.array(particles_cfg["x-axis"])
        yhat   = np.array(particles_cfg["y-axis"])
    except Exception as e:
        return None, None, None, None, None, f"bad_geometry: {e}"

    # ── Local → global coordinate transform ──────────────────────────────────
    # Particles are stored in 2D local plane coords (x_local, y_local).
    # Global pos = center  +  x_local * xhat  +  y_local * yhat
    mat   = np.array([xhat, yhat, zhat])   # (3,3) rotation matrix
    x_loc = np.asarray(tbl["x"].to_pylist(), dtype=np.float64)
    y_loc = np.asarray(tbl["y"].to_pylist(), dtype=np.float64)
    local = np.column_stack([x_loc, y_loc, np.zeros_like(x_loc)])
    xyz   = center + local @ mat.T

    # ── 90th-percentile radius pre-filter ────────────────────────────────────
    r              = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    percentile_val = np.percentile(r, 90)
    mask_r         = r <= percentile_val
    if mask_r.sum() == 0:
        return None, None, None, None, None, "empty_after_radius_filter"

    xyz = xyz[mask_r]
    tbl = tbl.filter(pa.array(mask_r))

    N = len(tbl)
    new_table = pa.table({
        "x":              pa.array(xyz[:, 0], type=pa.float64()),
        "y":              pa.array(xyz[:, 1], type=pa.float64()),
        "z":              pa.array(xyz[:, 2], type=pa.float64()),
        "pdg":            tbl["pdg"],
        "time":           tbl["time"],
        "kinetic_energy": tbl["kinetic_energy"],
        "plane_index":    pa.array(np.full(N, plane_idx, dtype=np.int32)),
        "p_energy":       pa.array(np.full(N, p_energy,  dtype=np.float64)),
    })
    return new_table, p_energy, zenith, azimuth, pdg_primary, "ok"


# =============================================================================
# ── Direction / PDG helpers ───────────────────────────────────────────────────
# =============================================================================

def _radians_sin_cos(angle: float):
    """Auto-convert degrees → radians if |angle| > 2π, then return (sin, cos)."""
    if math.isfinite(angle) and abs(angle) > 2 * math.pi + 1e-6:
        angle = math.radians(angle)
    return math.sin(angle), math.cos(angle)


def _build_direction(zenith: float, azimuth: float, np):
    """
    Unit direction vector matching CORSIKA convention:
        nx = sin(θ) · cos(φ)
        ny = sin(θ) · sin(φ)
        nz = cos(θ)
    """
    sin_zen, cos_zen = _radians_sin_cos(zenith)
    sin_azi, cos_azi = _radians_sin_cos(azimuth)
    return np.array([sin_zen * cos_azi, sin_zen * sin_azi, cos_zen], dtype=np.float32)


def _pdg_class(pdg_primary: float) -> int:
    """Binary class label: 0 = e±/π⁰,  1 = π±."""
    return 1 if abs(int(round(pdg_primary))) == 211 else 0


# =============================================================================
# ── File-size helpers ─────────────────────────────────────────────────────────
# =============================================================================

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# =============================================================================
# ── Core: process one simulation directory ────────────────────────────────────
# =============================================================================

def postprocess_simulation(
    sim_dir: str,
    nmax_per_particle: dict,
    dx_per_particle: dict,
    include_time: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Process one simulation directory → three HDF5 files + delete parquet files.

    Returns dict with keys:
        parquet_total_bytes, h5_total_bytes, h5_files, status, message
    """
    np, pd, yaml, pa, pq, pc, h5py = _lazy_imports()

    result = {
        "parquet_total_bytes": 0,
        "h5_total_bytes":      0,
        "h5_files":            [],
        "status":              "ok",
        "message":             "",
    }

    # ── Guard: summary.yaml must exist ───────────────────────────────────────
    if not os.path.exists(os.path.join(sim_dir, "summary.yaml")):
        result["status"]  = "skipped"
        result["message"] = "no summary.yaml"
        return result

    # ── Discover particles* folders ───────────────────────────────────────────
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

    # ── Measure parquet size before touching anything ─────────────────────────
    parquet_files_found = []
    for _, folder_path in particle_folders:
        pq_path = os.path.join(folder_path, "particles.parquet")
        if os.path.exists(pq_path):
            parquet_files_found.append(pq_path)
            result["parquet_total_bytes"] += os.path.getsize(pq_path)

    if not parquet_files_found:
        result["status"]  = "skipped"
        result["message"] = "no parquet files found"
        return result

    sim_name       = os.path.basename(os.path.normpath(sim_dir))
    any_h5_written = False
    errors         = []

    # ── Process each particle type ────────────────────────────────────────────
    for filter_key, cfg in PARTICLE_CONFIGS.items():
        pdg_values = cfg["pdg_values"]
        nmax       = nmax_per_particle.get(filter_key, cfg["default_nmax"])
        dx         = dx_per_particle.get(filter_key,   cfg["default_dx"])
        h5_path    = os.path.join(sim_dir, f"{sim_name}_{cfg['h5_suffix']}.h5")

        # Collect tables from all plane folders
        tables      = []
        p_energy    = 0.0
        zenith      = 0.0
        azimuth     = 0.0
        pdg_primary = 0.0
        meta_found  = False

        for pidx, folder_path in particle_folders:
            table, pe, zen, azi, pdg_p, status = _process_particle_folder(
                folder_path, pidx, sim_dir, pdg_values, np, pa, pq, pc, yaml
            )
            if status == "ok" and table is not None:
                tables.append(table)
                if not meta_found:
                    p_energy    = pe    if pe    is not None else 0.0
                    zenith      = zen   if zen   is not None else 0.0
                    azimuth     = azi   if azi   is not None else 0.0
                    pdg_primary = pdg_p if pdg_p is not None else 0.0
                    meta_found  = True

        if not tables:
            if verbose:
                print(f"    [{filter_key}] no valid data, skipping.")
            continue

        # Merge all planes → one DataFrame
        df = pa.concat_tables(tables).to_pandas()

        # Cluster (independent per plane, deterministic shift=0)
        points = _cluster_hits(df, cell_size=dx, np=np,
                               random_shift=False, include_time=include_time)
        if len(points) == 0:
            if verbose:
                print(f"    [{filter_key}] empty after clustering, skipping.")
            continue

        # Keep only top-nmax highest-energy cells
        if len(points) > nmax:
            top_idx = points["energy"].to_numpy().argpartition(-nmax)[-nmax:]
            points  = points.iloc[top_idx].reset_index(drop=True)

        # Sort by plane_idx (z order: plane 0 → plane 23) so the model sees
        # points in detector depth order regardless of energy truncation order.
        points = (
            points
            .sort_values("plane_idx", ascending=True)
            .reset_index(drop=True)
        )

        # Build flat showers array: (n_pts, 5) → x, y, plane_idx, energy, time
        shower_cols = ["x", "y", "plane_idx", "energy"]
        if include_time and "time" in points.columns:
            shower_cols.append("time")
        showers_arr = points[shower_cols].to_numpy(dtype=np.float32)
        n_pts  = showers_arr.shape[0]
        n_feat = showers_arr.shape[1]
        shower_flat = showers_arr.flatten()

        # Primary particle direction and class
        direction      = _build_direction(zenith, azimuth, np)   # float32 (3,)
        actual_pdg_int = int(round(pdg_primary))
        pdg_class_int  = _pdg_class(pdg_primary)

        # ── Write HDF5 ────────────────────────────────────────────────────────
        try:
            with h5py.File(h5_path, "w") as hf:
                # Attrs
                hf.attrs["sim_dir"]      = sim_dir
                hf.attrs["filter"]       = filter_key
                hf.attrs["dx"]           = dx
                hf.attrs["nmax"]         = nmax
                hf.attrs["include_time"] = include_time
                hf.attrs["created"]      = datetime.now().isoformat()

                # shape: [n_sims=1, nmax, n_feat]
                hf.create_dataset("shape",
                    data=np.array([1, nmax, n_feat], dtype=np.int64), dtype=np.int64)

                # num_points: actual clustered points stored
                hf.create_dataset("num_points",
                    data=np.array([n_pts], dtype=np.int64), dtype=np.int64)

                # showers: vlen float32, one entry per sim
                # element shape = (n_pts * n_feat,) — reshape with num_points & shape[2]
                dt_vlen    = h5py.vlen_dtype(np.dtype("float32"))
                ds_showers = hf.create_dataset("showers", (1,), dtype=dt_vlen)
                ds_showers[0] = shower_flat

                # directions: (1, 3) float32
                hf.create_dataset("directions",
                    data=direction.reshape(1, 3), dtype=np.float32)

                # energies: (1,) float32
                hf.create_dataset("energies",
                    data=np.array([p_energy], dtype=np.float32), dtype=np.float32)

                # pdg: (1,) int32  — class label
                hf.create_dataset("pdg",
                    data=np.array([pdg_class_int], dtype=np.int32), dtype=np.int32)

                # actual_pdg: (1,) int32  — true primary PDG
                hf.create_dataset("actual_pdg",
                    data=np.array([actual_pdg_int], dtype=np.int32), dtype=np.int32)

            h5_size = os.path.getsize(h5_path)
            result["h5_total_bytes"] += h5_size
            result["h5_files"].append(h5_path)
            any_h5_written = True

            if verbose:
                print(
                    f"    [{filter_key:9s}]  {n_pts:>6d}/{nmax} pts  "
                    f"feats={n_feat}  dx={dx}m  "
                    f"dir=[{direction[0]:.3f},{direction[1]:.3f},{direction[2]:.3f}]  "
                    f"E={p_energy:.3e}  pdg={actual_pdg_int}(cls={pdg_class_int})  "
                    f"→ {os.path.basename(h5_path)} ({_human_bytes(h5_size)})"
                )
        except Exception as e:
            errors.append(f"{filter_key}: h5 write error: {e}")
            if verbose:
                print(f"    [{filter_key}] ERROR writing h5: {e}")

    # ── Delete parquet files ──────────────────────────────────────────────────
    if any_h5_written:
        for pq_path in parquet_files_found:
            try:
                os.remove(pq_path)
            except OSError as e:
                if verbose:
                    print(f"    WARNING: could not delete {pq_path}: {e}")
        if verbose:
            print(f"    Removed {len(parquet_files_found)} parquet file(s)  "
                  f"(freed {_human_bytes(result['parquet_total_bytes'])})")
    else:
        result["status"]  = "skipped"
        result["message"] = "no h5 files written (all particle types empty)"

    if errors:
        result["status"]  = "partial"
        result["message"] = "; ".join(errors)

    return result


# =============================================================================
# ── CLI ───────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process one TAMBO simulation directory: parquet → HDF5."
    )
    parser.add_argument("--sim-dir", required=True,
                        help="Path to simulation directory (must contain summary.yaml).")

    parser.add_argument("--electrons-dx",    type=float,
                        default=PARTICLE_CONFIGS["electrons"]["default_dx"], metavar="M")
    parser.add_argument("--muons-dx",        type=float,
                        default=PARTICLE_CONFIGS["muons"]["default_dx"],     metavar="M")
    parser.add_argument("--photons-dx",      type=float,
                        default=PARTICLE_CONFIGS["photons"]["default_dx"],   metavar="M")

    parser.add_argument("--electrons-nmax",  type=int,
                        default=PARTICLE_CONFIGS["electrons"]["default_nmax"], metavar="N")
    parser.add_argument("--muons-nmax",      type=int,
                        default=PARTICLE_CONFIGS["muons"]["default_nmax"],     metavar="N")
    parser.add_argument("--photons-nmax",    type=int,
                        default=PARTICLE_CONFIGS["photons"]["default_nmax"],   metavar="N")

    parser.add_argument("--no-time",  action="store_true",
                        help="Exclude time feature (store 4 features instead of 5).")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-particle verbose output.")

    args = parser.parse_args()

    sim_dir = os.path.normpath(args.sim_dir)
    if not os.path.isdir(sim_dir):
        print(f"ERROR: sim_dir does not exist: {sim_dir}", file=sys.stderr)
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

    print(f"Processing: {sim_dir}")
    result = postprocess_simulation(
        sim_dir           = sim_dir,
        nmax_per_particle = nmax_per_particle,
        dx_per_particle   = dx_per_particle,
        include_time      = not args.no_time,
        verbose           = not args.quiet,
    )

    print(f"Status  : {result['status']}")
    print(f"Parquet : {_human_bytes(result['parquet_total_bytes'])} removed")
    print(f"HDF5    : {_human_bytes(result['h5_total_bytes'])} written "
          f"({len(result['h5_files'])} files)")
    if result["message"]:
        print(f"Detail  : {result['message']}")

    sys.exit(0 if result["status"] in ("ok", "partial") else 1)


if __name__ == "__main__":
    main()