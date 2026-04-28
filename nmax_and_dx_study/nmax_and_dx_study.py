#!/usr/bin/env python3
"""
Post-process a CHUNK of combined hit-level Parquet files into ONE CSV file
with energy-ignored statistics for an Nmax sweep.

Adapted from postprocess_showers.py — same parquet input, same dedup, same
clustering — but instead of writing H5 files, we record, for each shower and
each (particle_type, dx, nmax) combination, how much shower energy is lost
when only the top-Nmax highest-energy clustered cells are kept.

Energy-ignored definition (per shower, per particle_type, per (dx, nmax)):
    1. Cluster hits into (dx × dx) cells per plane (sum energy).
    2. Sort clustered cells by energy (highest → lowest).
    3. Keep the first Nmax cells; discard the rest.
    4. energy_ignored_frac = (total_E - retained_E) / total_E
       If n_clustered_cells <= Nmax → energy_ignored_frac = 0
       If total_E == 0              → energy_ignored_frac = 0

Output CSV (one big CSV per chunk) — written to <out>/pdg_<PDG>/:
    chunk_<XXXX>_energy_ignored.csv

Columns:
    shower_id, incident_pdg, actual_pdg, incident_energy,
    particle_type, dx, nmax,
    total_energy, retained_energy, energy_ignored_frac,
    n_clustered_cells

For a single (shower × particle_type × dx) we emit ONE row per nmax in the
sweep, so the file grows as
    n_showers × n_particle_types × n_dx × n_nmax  rows.

Usage:
python postprocess_showers_csv.py \
    --chunk-list registry_pdg_11.txt \
    --incident-pdg 11 \
    --chunk-id 0 \
    --output-dir /path/to/output \
    --max-showers 10000 --random --seed 42 \
    --dx-sweep 5 10 20 \
    --nmax-sweep 1000 2000 4000

By default --nmax-sweep is "1000 2000 4000" (matches the reference plot).
By default each --<particle>-dx-sweep is just "10".

If --max-showers is given:
    - WITHOUT --random  → take the first N entries (deterministic).
    - WITH    --random  → take a uniform random sample of size N (use --seed
                          for reproducibility).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Optional


# =============================================================================
# Particle type configuration (secondary particles)
#
# preprocess_showers.py applies PDG_REMAP that collapses sign:
#     {11: 11, -11: 11, 13: 13, -13: 13, 22: 22}
# so the `pdg` column in the input parquets only ever contains {11, 13, 22}.
# =============================================================================

PARTICLE_CONFIGS = {
    "electrons": {
        "pdg_values": [11],
        "label":      "e± (pdg 11)",
        "default_dx_sweep": [10.0],
    },
    "muons": {
        "pdg_values": [13],
        "label":      "μ± (pdg 13)",
        "default_dx_sweep": [10.0],
    },
    "photons": {
        "pdg_values": [22],
        "label":      "γ (pdg 22)",
        "default_dx_sweep": [10.0],
    },
}

DEFAULT_NMAX_SWEEP = [1000, 2000, 4000]


# =============================================================================
# Lazy imports
# =============================================================================

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    return np, pd, pq


# =============================================================================
# Helpers (event CSV reader + small utils)
# =============================================================================

def _event_csv_path_for(hits_parquet_path: str) -> str:
    """<shower>_hits.parquet  →  <shower>_event.csv"""
    base = hits_parquet_path
    if base.endswith("_hits.parquet"):
        return base[: -len("_hits.parquet")] + "_event.csv"
    root, _ = os.path.splitext(base)
    return root + "_event.csv"


def _load_event_row(csv_path: str) -> Optional[dict]:
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
    row = _load_event_row(csv_path) or {}
    return {
        "shower_id":       str(row.get("shower_id", "")),
        "incident_energy": _float_or(0.0, row.get("incident_energy")),
        "incident_pdg":    _float_or(0.0, row.get("incident_pdg")),
    }


# =============================================================================
# Duplicate removal — IDENTICAL to original script
# =============================================================================

def _drop_near_duplicates(df, np, pd,
                          time_tol: float, energy_rel_tol: float, xy_tol: float,
                          verbose: bool = False):
    """
    Drop near-duplicate hits that arise when the same particle is recorded
    twice on a plane. Sort by (plane, pdg, time, energy, x, y), then flag
    consecutive rows where ALL of {time, energy (relative), x, y} are within
    tolerance AND the same plane+pdg.
    Returns (deduped_df, n_removed).
    """
    if len(df) == 0:
        return df, 0

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

    dup_mask = np.zeros(n, dtype=bool)
    dup_mask[1:] = close

    n_removed = int(dup_mask.sum())
    if n_removed == 0:
        return df_sorted, 0

    if verbose:
        print(f"      dedup: removed {n_removed}/{n} near-duplicate hits")

    return df_sorted.loc[~dup_mask].reset_index(drop=True), n_removed


# =============================================================================
# Clustering — IDENTICAL to original script
# =============================================================================

def _cluster_shower_part(xy, e, cell_size, shift, np):
    """Bin particles into (cell_size × cell_size) cells on one plane.
       Returns (xy_clustered, e_clustered)."""
    if len(xy) == 0:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0,),   dtype=np.float32))

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

    e_clustered = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(e_clustered, inverse_idx, e)

    xy_clustered = np.column_stack([unique_x, unique_y]).astype(np.float32)
    xy_clustered += 0.5
    xy_clustered *= cell_size
    xy_clustered -= shift
    return xy_clustered, e_clustered


def _cluster_energies_for_type(df_sub, cell_size, np):
    """
    Cluster hits of one secondary type into (dx × dx) cells per plane,
    summing energy. Returns a 1-D ndarray of clustered cell energies
    (positions are not needed for the energy-ignored calculation).
    """
    if len(df_sub) == 0:
        return np.empty((0,), dtype=np.float32)

    pos   = df_sub[["x", "y", "plane_index"]].to_numpy(dtype=np.float32)
    e_all = np.maximum(df_sub["kinetic_energy"].to_numpy(dtype=np.float32), 0.0)
    plane = pos[:, 2].astype(np.int32)

    e_parts = []
    for p_val in np.unique(plane):
        mask = plane == p_val
        xy_p = pos[mask, :2]
        e_p  = e_all[mask]
        shift = np.array([0.0, 0.0], dtype=np.float32)

        _, e_c = _cluster_shower_part(xy_p, e_p, cell_size, shift, np)
        if len(e_c) > 0:
            e_parts.append(e_c)

    if not e_parts:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(e_parts, axis=0)


# =============================================================================
# Energy-ignored computation
# =============================================================================

def _energy_ignored_for_nmax_sweep(cell_energies, nmax_list, np):
    """
    Given a 1-D array of clustered cell energies (any order) and a list of
    Nmax values, return a list of dicts, one per nmax, with:
        nmax, total_energy, retained_energy, energy_ignored_frac,
        n_clustered_cells

    Sorts cells by energy descending ONCE, then for each Nmax takes the
    cumulative sum of the top-Nmax entries.
    """
    n_cells = int(cell_energies.shape[0])
    total_E = float(cell_energies.sum()) if n_cells > 0 else 0.0

    if n_cells == 0:
        # No cells at all → nothing to ignore. Mark retained=0, ignored=0.
        return [
            {
                "nmax": int(nmax),
                "total_energy": 0.0,
                "retained_energy": 0.0,
                "energy_ignored_frac": 0.0,
                "n_clustered_cells": 0,
            }
            for nmax in nmax_list
        ]

    # sort descending — this is the "keep highest-energy cells" step
    sorted_desc = np.sort(cell_energies)[::-1].astype(np.float64)
    cumE = np.cumsum(sorted_desc)  # cumE[k-1] = sum of top-k cells

    out = []
    for nmax in nmax_list:
        nmax_i = int(nmax)
        if n_cells <= nmax_i:
            # Spec: shower has fewer than Nmax cells → no energy ignored.
            retained = total_E
            ignored  = 0.0
        else:
            retained = float(cumE[nmax_i - 1])
            ignored  = (total_E - retained) / total_E if total_E > 0 else 0.0

        out.append({
            "nmax": nmax_i,
            "total_energy": total_E,
            "retained_energy": retained,
            "energy_ignored_frac": ignored,
            "n_clustered_cells": n_cells,
        })
    return out


# =============================================================================
# Core: process one chunk → ONE big CSV
# =============================================================================

CSV_COLUMNS = [
    "shower_id",
    "incident_pdg",
    "actual_pdg",
    "incident_energy",
    "particle_type",
    "dx",
    "nmax",
    "total_energy",
    "retained_energy",
    "energy_ignored_frac",
    "n_clustered_cells",
]


def process_chunk(
    chunk_paths: list[str],
    output_dir: str,
    incident_pdg: int,
    chunk_id: int,
    dx_sweep_per_particle: dict,
    nmax_sweep: list[int],
    particles: Optional[list[str]] = None,
    max_showers: Optional[int] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
    do_dedup: bool = True,
    dedup_time_tol: float = 1e-15,
    dedup_energy_rel_tol: float = 1e-6,
    dedup_xy_tol: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Process a list of parquet files → one big CSV in output_dir/pdg_<N>/."""
    np, pd, pq = _lazy_imports()

    active = list(PARTICLE_CONFIGS.keys())
    if particles:
        bad = [p for p in particles if p not in PARTICLE_CONFIGS]
        if bad:
            raise ValueError(f"Unknown particle types: {bad}. "
                             f"Choose from {list(PARTICLE_CONFIGS)}.")
        active = [p for p in PARTICLE_CONFIGS if p in particles]
        if not active:
            raise ValueError("Empty particle selection after filtering.")

    if not nmax_sweep:
        raise ValueError("nmax_sweep must contain at least one value.")
    nmax_sweep = sorted({int(n) for n in nmax_sweep})

    # Validate / normalise per-particle dx sweeps for the active set.
    dx_sweep_resolved: dict[str, list[float]] = {}
    for pkey in active:
        cfg = PARTICLE_CONFIGS[pkey]
        sweep = dx_sweep_per_particle.get(pkey) or cfg["default_dx_sweep"]
        sweep = sorted({float(v) for v in sweep})
        if not sweep:
            raise ValueError(f"Empty dx sweep for particle '{pkey}'.")
        if any(v <= 0 for v in sweep):
            raise ValueError(f"dx values must be positive for '{pkey}': {sweep}")
        dx_sweep_resolved[pkey] = sweep

    subdir = os.path.join(output_dir, f"pdg_{incident_pdg}")
    os.makedirs(subdir, exist_ok=True)
    csv_path = os.path.join(subdir, f"chunk_{chunk_id:04d}_energy_ignored.csv")

    result = {
        "incident_pdg":     incident_pdg,
        "chunk_id":         chunk_id,
        "n_parquets":       len(chunk_paths),
        "n_processed":      0,
        "n_skipped":        0,
        "n_dedup_removed":  0,
        "particles":        active,
        "dx_sweep":         dx_sweep_resolved,
        "nmax_sweep":       nmax_sweep,
        "random_sample":    bool(random_sample),
        "seed":             seed,
        "csv_path":         csv_path,
        "n_rows_written":   0,
        "status":           "ok",
        "message":          "",
        "elapsed_s":        0.0,
    }
    t_start = time.time()
    required_cols = ["x", "y", "pdg", "time", "kinetic_energy", "plane_index"]

    # --- pick which paths to use --------------------------------------------
    # Default: all paths in order. With --max-showers and --random, draw a
    # uniform random sample without replacement (seeded for reproducibility).
    if max_showers is not None and max_showers < len(chunk_paths):
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(chunk_paths), size=max_showers, replace=False)
            idx.sort()  # process in input order so log timing is monotonic
            paths_to_use = [chunk_paths[i] for i in idx]
        else:
            paths_to_use = chunk_paths[:max_showers]
    else:
        paths_to_use = list(chunk_paths)
    result["n_selected"] = len(paths_to_use)

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, pq_path in enumerate(paths_to_use):
            if not os.path.exists(pq_path):
                result["n_skipped"] += 1
                if verbose:
                    print(f"  [{i+1}/{len(paths_to_use)}] MISSING: {pq_path}")
                continue

            try:
                tbl = pq.read_table(pq_path, columns=required_cols)
            except Exception as e:
                result["n_skipped"] += 1
                if verbose:
                    print(f"  [{i+1}/{len(paths_to_use)}] READ ERROR {pq_path}: {e}")
                continue

            if len(tbl) == 0:
                result["n_skipped"] += 1
                continue

            # event metadata
            event_csv = _event_csv_path_for(pq_path)
            meta = _extract_event_meta(event_csv)
            actual_pdg_int = (int(round(meta["incident_pdg"]))
                              if meta["incident_pdg"] else incident_pdg)
            p_energy = float(meta["incident_energy"])
            shower_id = (meta["shower_id"]
                         or os.path.basename(pq_path).replace("_hits.parquet", ""))

            df = tbl.to_pandas()

            if do_dedup:
                df, n_rm = _drop_near_duplicates(
                    df, np, pd,
                    time_tol=dedup_time_tol,
                    energy_rel_tol=dedup_energy_rel_tol,
                    xy_tol=dedup_xy_tol,
                    verbose=(verbose and i < 3),
                )
                result["n_dedup_removed"] += n_rm

            pdg_arr = df["pdg"].to_numpy()

            for pkey in active:
                cfg  = PARTICLE_CONFIGS[pkey]
                mask = np.isin(pdg_arr, cfg["pdg_values"])
                df_sub = df[mask]

                for dx in dx_sweep_resolved[pkey]:
                    cell_E = _cluster_energies_for_type(df_sub, cell_size=dx, np=np)
                    rows = _energy_ignored_for_nmax_sweep(cell_E, nmax_sweep, np)

                    for r in rows:
                        writer.writerow({
                            "shower_id":           shower_id,
                            "incident_pdg":        incident_pdg,
                            "actual_pdg":          actual_pdg_int,
                            "incident_energy":     p_energy,
                            "particle_type":       pkey,
                            "dx":                  dx,
                            "nmax":                r["nmax"],
                            "total_energy":        r["total_energy"],
                            "retained_energy":     r["retained_energy"],
                            "energy_ignored_frac": r["energy_ignored_frac"],
                            "n_clustered_cells":   r["n_clustered_cells"],
                        })
                        result["n_rows_written"] += 1

            result["n_processed"] += 1

            if verbose and ((i + 1) % 50 == 0 or i + 1 == len(paths_to_use)):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  [{i+1}/{len(paths_to_use)}] processed "
                      f"({rate:.2f} showers/s, dedup_removed={result['n_dedup_removed']})")

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
        description="Post-process a CHUNK of combined hit parquets → "
                    "ONE CSV with energy-ignored statistics for an Nmax sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunk-list", required=True,
                        help="Text file with one parquet path per line.")
    parser.add_argument("--incident-pdg", type=int, required=True,
                        help="Incident PDG this chunk belongs to (used for subdir name).")
    parser.add_argument("--chunk-id", type=int, required=True,
                        help="Zero-padded chunk id (used in output filename).")
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir. CSV goes to <out>/pdg_<PDG>/.")

    # per-particle cell-size SWEEP (one or more values; nargs="+")
    parser.add_argument("--electrons-dx-sweep", type=float, nargs="+",
                        default=PARTICLE_CONFIGS["electrons"]["default_dx_sweep"],
                        help="Cell sizes (dx) to evaluate for electrons.")
    parser.add_argument("--muons-dx-sweep", type=float, nargs="+",
                        default=PARTICLE_CONFIGS["muons"]["default_dx_sweep"],
                        help="Cell sizes (dx) to evaluate for muons.")
    parser.add_argument("--photons-dx-sweep", type=float, nargs="+",
                        default=PARTICLE_CONFIGS["photons"]["default_dx_sweep"],
                        help="Cell sizes (dx) to evaluate for photons.")
    # Convenience: apply the same dx sweep to all three particle types.
    parser.add_argument("--dx-sweep", type=float, nargs="+", default=None,
                        help="Shortcut: apply this dx sweep to e/μ/γ "
                             "(overrides the per-particle --*-dx-sweep flags).")

    # Nmax sweep
    parser.add_argument("--nmax-sweep", type=int, nargs="+",
                        default=DEFAULT_NMAX_SWEEP,
                        help="Nmax values to evaluate (default matches reference plot).")

    parser.add_argument("--particles", nargs="+", default=None,
                        choices=list(PARTICLE_CONFIGS.keys()),
                        help="Subset of secondary types. Default: all three.")

    parser.add_argument("--max-showers", type=int, default=None,
                        help="Cap on number of showers consumed from chunk-list "
                             "(useful for the 10k-per-pdg study).")
    parser.add_argument("--random", action="store_true",
                        help="With --max-showers: take a uniform RANDOM sample "
                             "(without replacement) instead of the first N.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for --random sampling (use to reproduce a draw).")

    # dedup knobs (kept ON with the same defaults as the original script)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable near-duplicate removal.")
    parser.add_argument("--dedup-time-tol", type=float, default=1e-15)
    parser.add_argument("--dedup-energy-rel-tol", type=float, default=1e-6)
    parser.add_argument("--dedup-xy-tol", type=float, default=1e-3)

    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.chunk_list):
        print(f"ERROR: chunk-list not found: {args.chunk_list}", file=sys.stderr)
        sys.exit(1)

    chunk_paths = _read_chunk_list(args.chunk_list)
    if not chunk_paths:
        print(f"ERROR: chunk-list is empty: {args.chunk_list}", file=sys.stderr)
        sys.exit(1)

    if args.dx_sweep:
        dx_sweep_per_particle = {
            "electrons": list(args.dx_sweep),
            "muons":     list(args.dx_sweep),
            "photons":   list(args.dx_sweep),
        }
    else:
        dx_sweep_per_particle = {
            "electrons": list(args.electrons_dx_sweep),
            "muons":     list(args.muons_dx_sweep),
            "photons":   list(args.photons_dx_sweep),
        }

    if args.random and args.max_showers is None:
        print("WARNING: --random has no effect without --max-showers; ignored.",
              file=sys.stderr)

    print(f"Chunk list    : {args.chunk_list}")
    print(f"N parquets    : {len(chunk_paths)} "
          f"(cap: {args.max_showers if args.max_showers else 'none'}"
          f"{', RANDOM' if args.random and args.max_showers else ''}"
          f"{f', seed={args.seed}' if args.random and args.max_showers and args.seed is not None else ''})")
    print(f"Incident PDG  : {args.incident_pdg}")
    print(f"Chunk id      : {args.chunk_id:04d}")
    print(f"Output dir    : {args.output_dir}/pdg_{args.incident_pdg}/")
    print(f"Particles     : {args.particles if args.particles else 'electrons muons photons'}")
    print(f"dx sweeps     : "
          f"e={dx_sweep_per_particle['electrons']} "
          f"μ={dx_sweep_per_particle['muons']} "
          f"γ={dx_sweep_per_particle['photons']}")
    print(f"Nmax sweep    : {args.nmax_sweep}")
    print(f"Dedup         : {'OFF' if args.no_dedup else 'ON'} "
          f"(time<{args.dedup_time_tol:.1e}s, dE/E<{args.dedup_energy_rel_tol:.1e}, "
          f"dxy<{args.dedup_xy_tol:.1e}m)")
    print()

    result = process_chunk(
        chunk_paths       = chunk_paths,
        output_dir        = args.output_dir,
        incident_pdg      = args.incident_pdg,
        chunk_id          = args.chunk_id,
        dx_sweep_per_particle = dx_sweep_per_particle,
        nmax_sweep        = args.nmax_sweep,
        particles         = args.particles,
        max_showers       = args.max_showers,
        random_sample     = args.random,
        seed              = args.seed,
        do_dedup          = not args.no_dedup,
        dedup_time_tol    = args.dedup_time_tol,
        dedup_energy_rel_tol = args.dedup_energy_rel_tol,
        dedup_xy_tol      = args.dedup_xy_tol,
        verbose           = not args.quiet,
    )

    print()
    print(f"Status        : {result['status']}")
    print(f"Selected      : {result.get('n_selected', result['n_parquets'])}/{result['n_parquets']} "
          f"showers from chunk-list")
    print(f"Processed     : {result['n_processed']} showers")
    print(f"Skipped       : {result['n_skipped']}")
    print(f"Dedup removed : {result['n_dedup_removed']} hits")
    print(f"CSV rows      : {result['n_rows_written']}")
    print(f"CSV path      : {result['csv_path']}")
    print(f"Elapsed       : {result['elapsed_s']:.1f} s")
    if result["message"]:
        print(f"Detail        : {result['message']}")

    sys.exit(0 if result["status"] in ("ok", "partial") else 1)


if __name__ == "__main__":
    main()