#!/usr/bin/env python3
"""
Inspect (and optionally remove) showers that fail the --num-layer-cond isolation
filter from an already-merged HDF5 file.

Isolation filter (same logic as allshowers_merge.py):
  1. Only 1 unique layer (all hits in same layer)
  2. Any layer has no neighbouring occupied layer within ±half (num_layer_cond // 2)

Modes
-----
  Inspect only (default):
      Print a summary of how many showers fail the condition.

  Remove (--remove):
      Write a new HDF5 (--output, required) containing only the passing showers.
      All datasets are re-written; shower_ids are reassigned 0..N_pass-1.
      The original file is NEVER modified.
  python /n/home04/hhanif/AllShowers/util/check_isolation.py \
      --input  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5 \
      --num-layer-cond 8 \
      --chunk-size 5000 \
      --with-time \
      --remove \
      --output  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced_isolations_removed.h5


Datasets handled
----------------
  Required : showers, directions, energies, pdg, shower_ids, num_points
  Optional : shape   (recomputed and written in the output)
  Any other dataset present in the input is copied verbatim if its first
  dimension == N; otherwise copied as-is unchanged.
  Handles any dataset shape: (N,), (N,3), (N,1), (N,M,K), vlen, etc.

Usage examples
--------------
  # Inspect only
  python filter_showers.py \\
      --input  merged_all_showers.h5 \\
      --num-layer-cond 8

  # Inspect + remove bad showers -> new file
  python filter_showers.py \\
      --input  merged_all_showers.h5 \\
      --num-layer-cond 8 \\
      --remove \\
      --output merged_all_showers_filtered.h5

  # 5-field showers [x, y, layer, energy, time]
  python filter_showers.py \\
      --input  merged_with_time.h5 \\
      --num-layer-cond 8 \\
      --with-time \\
      --remove \\
      --output merged_with_time_filtered.h5

  # Adjust chunk size (default 512) to trade RAM vs speed
  python filter_showers.py \\
      --input  big_file.h5 \\
      --num-layer-cond 8 \\
      --chunk-size 1024 \\
      --remove \\
      --output big_file_filtered.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Isolation filter (identical logic to allshowers_merge.py)
# ---------------------------------------------------------------------------

def is_bad_shower(arr: np.ndarray, half: int, n_fields: int = 4) -> tuple[bool, str]:
    """
    Return (True, reason) if shower should be removed, else (False, "").

    arr      : flat float32 array  [x, y, layer, energy, (time), x, y, ...]
    half     : neighbourhood half-window  (= num_layer_cond // 2)
    n_fields : 4 (default) or 5 (--with-time)
    """
    pts = arr.reshape(-1, n_fields)
    mask = pts[:, 3] > 0                      # energy index is always 3
    layers = (pts[:, 2] + 0.1).astype(int)   # layer index is always 2
    valid_layers = layers[mask]

    if len(valid_layers) == 0:
        return True, "empty (no hits with energy > 0)"

    unique_layers = sorted(set(valid_layers.tolist()))

    # condition 1: only 1 unique layer
    if len(unique_layers) == 1:
        return True, f"single unique layer {unique_layers[0]}"

    # condition 2: any layer has no neighbour within +/-half
    isolated = [
        l for l in unique_layers
        if not any(0 < abs(l - other) <= half for other in unique_layers)
    ]
    if isolated:
        return True, f"isolated layer(s) {isolated} among {unique_layers}"

    return False, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORE_DATASETS = {"showers", "directions", "energies", "pdg", "shower_ids", "num_points"}


def iter_chunks(total: int, chunk_size: int):
    """Yield (start, end) half-open pairs that tile [0, total)."""
    start = 0
    while start < total:
        yield start, min(start + chunk_size, total)
        start += chunk_size


def is_vlen(ds: h5py.Dataset) -> bool:
    return h5py.check_vlen_dtype(ds.dtype) is not None


def create_output_dataset(hout: h5py.File, name: str,
                          src_ds: h5py.Dataset, N_good: int,
                          vlen_dt) -> h5py.Dataset:
    """
    Create a destination dataset whose first dim is N_good, mirroring
    src_ds in dtype, remaining shape, and compression.

    Correctly handles:
      - vlen (variable-length) dtype  -> shape (N_good,)
      - (N,)                          -> (N_good,)
      - (N, k)                        -> (N_good, k)
      - (N, k, m)                     -> (N_good, k, m)
      etc.
    """
    chunk_n = min(1000, max(1, N_good))

    if is_vlen(src_ds):
        return hout.create_dataset(name, shape=(N_good,), dtype=vlen_dt)

    rest        = src_ds.shape[1:]              # everything after the N dim
    out_shape   = (N_good,) + rest
    chunk_shape = (chunk_n,)  + rest

    kwargs = dict(shape=out_shape, dtype=src_ds.dtype)
    if src_ds.compression:
        kwargs["compression"] = src_ds.compression
        kwargs["chunks"]      = chunk_shape

    return hout.create_dataset(name, **kwargs)


# ---------------------------------------------------------------------------
# Pass 1 – inspect
# ---------------------------------------------------------------------------

def inspect(input_path: str, half: int,
            n_fields: int, chunk_size: int) -> np.ndarray:
    """
    Stream through the file, apply the isolation filter, return a boolean
    good_mask of length N (True = keep).
    """
    print(f"\nInspecting  : {input_path}")
    print(f"Filter      : +/-{half} layers  (num_layer_cond={half * 2})")
    print(f"Fields/point: {n_fields}  "
          f"({'[x,y,layer,energy,time]' if n_fields == 5 else '[x,y,layer,energy]'})")

    with h5py.File(input_path, "r") as hf:
        N = int(hf["directions"].shape[0])
        print(f"Total showers in file: {N:,}\n")

        good_mask     = np.ones(N, dtype=bool)
        reason_counts: dict[str, int] = {}
        shw_ds        = hf["showers"]

        for start, end in tqdm(list(iter_chunks(N, chunk_size)),
                               desc="Scanning chunks", unit="chunk"):
            for i in range(start, end):
                bad, reason = is_bad_shower(shw_ds[i], half, n_fields)
                if bad:
                    good_mask[i] = False
                    if   reason.startswith("empty"):   cat = "empty"
                    elif reason.startswith("single"):  cat = "single unique layer"
                    else:                              cat = "isolated layer(s)"
                    reason_counts[cat] = reason_counts.get(cat, 0) + 1

    n_bad  = int(np.sum(~good_mask))
    n_good = N - n_bad

    print("\n=== Inspection Summary ===")
    print(f"  Total showers         : {N:>10,}")
    print(f"  PASS (good)           : {n_good:>10,}  ({100 * n_good / N:.2f}%)")
    print(f"  FAIL (bad)            : {n_bad:>10,}  ({100 * n_bad  / N:.2f}%)")
    if reason_counts:
        print("\n  Failure breakdown:")
        for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<30s}: {cnt:,}")

    return good_mask


# ---------------------------------------------------------------------------
# Pass 2 – write filtered output
# ---------------------------------------------------------------------------

def write_filtered(input_path: str, output_path: str,
                   good_mask: np.ndarray, n_fields: int, chunk_size: int):
    """
    Write a new HDF5 with only the rows where good_mask is True.

    Strategy per dataset type
    -------------------------
    vlen datasets (e.g. 'showers'):
        Must be read and written one row at a time – h5py requires it.

    Fixed-shape datasets whose first dim == N (e.g. directions, energies …):
        Read in batches for speed.
        We read the bounding slice [lo:hi] from disk and fancy-index within
        it to get exactly the good rows – avoids one HDF5 call per row.

        Why a bounding slice and not direct fancy indexing?
        h5py supports fancy (non-contiguous) indexing BUT it issues one
        low-level read per selected element when indices are non-contiguous,
        which is slower than reading a contiguous block and slicing in NumPy.
        For large chunks the bounding-slice approach is significantly faster.

    Datasets whose first dim != N (or the 'shape' dataset):
        'shape'   -> recomputed from scratch.
        everything else -> copied verbatim (hin.copy).

    shower_ids:
        Always reassigned sequentially (0 … N_good-1) regardless of dtype
        or shape – so (N,) and (N,1) are both handled.
    """
    good_indices = np.where(good_mask)[0]
    N_good       = len(good_indices)
    N_orig       = len(good_mask)

    print(f"\nWriting filtered file : {output_path}")
    print(f"Showers to write      : {N_good:,}  (dropped {N_orig - N_good:,})")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    vlen_dt = h5py.vlen_dtype(np.dtype("float32"))

    with h5py.File(input_path, "r") as hin, \
         h5py.File(output_path, "w") as hout:

        # -- root attributes ------------------------------------------------
        for k, v in hin.attrs.items():
            hout.attrs[k] = v
        hout.attrs["n_simulations"] = N_good
        hout.attrs["n_original"]    = N_orig
        hout.attrs["n_removed"]     = N_orig - N_good

        # -- classify datasets ----------------------------------------------
        #   vlen_keys  : variable-length, first dim == N  -> row-by-row
        #   fixed_keys : fixed dtype,     first dim == N  -> batched
        #   skip_keys  : 'shape' or first dim != N        -> recompute / copy
        vlen_keys, fixed_keys, skip_keys = [], [], []

        for k in hin.keys():
            if k == "shape":
                skip_keys.append(k)
                continue
            # Skip HDF5 Groups — only handle Datasets
            if not isinstance(hin[k], h5py.Dataset):
                skip_keys.append(k)
                continue
            ds = hin[k]
            if len(ds.shape) == 0 or ds.shape[0] != N_orig:
                skip_keys.append(k)
                continue
            (vlen_keys if is_vlen(ds) else fixed_keys).append(k)

        print(f"\nDataset classification:")
        print(f"  variable-length (row-by-row) : {vlen_keys}")
        print(f"  fixed-shape     (batched)    : {fixed_keys}")
        print(f"  skipped / recomputed         : {skip_keys}")

        # -- pre-create output datasets -------------------------------------
        out_ds = {}
        for k in vlen_keys + fixed_keys:
            out_ds[k] = create_output_dataset(hout, k, hin[k], N_good, vlen_dt)

        # -- stream and write -----------------------------------------------
        actual_max_pts = 0
        out_i          = 0

        for chunk_start, chunk_end in tqdm(
                list(iter_chunks(N_good, chunk_size)),
                desc="Writing chunks", unit="chunk"):

            idx_chunk = good_indices[chunk_start:chunk_end]   # source indices
            chunk_len = len(idx_chunk)
            out_slice = slice(out_i, out_i + chunk_len)

            # batched writes for fixed-shape datasets
            if fixed_keys:
                lo = int(idx_chunk[0])
                hi = int(idx_chunk[-1]) + 1
                local = idx_chunk - lo              # relative offsets in block

                for k in fixed_keys:
                    block    = hin[k][lo:hi]        # contiguous disk read
                    selected = block[local]         # numpy fancy index (in RAM)

                    if k == "shower_ids":
                        # reassign sequentially; preserve original ndim
                        new_ids = np.arange(out_i, out_i + chunk_len,
                                            dtype=hin[k].dtype)
                        if hin[k].ndim == 2:        # shape was (N, 1)
                            new_ids = new_ids[:, None]
                        selected = new_ids

                    out_ds[k][out_slice] = selected

            # row-by-row writes for vlen datasets
            for rel_i, src_i in enumerate(idx_chunk):
                g_i = out_i + rel_i
                for k in vlen_keys:
                    arr = hin[k][src_i]
                    out_ds[k][g_i] = arr
                    if k == "showers":
                        actual_max_pts = max(actual_max_pts, len(arr) // n_fields)

            out_i += chunk_len

        # -- recompute 'shape' ----------------------------------------------
        hout.create_dataset(
            "shape",
            data=np.array([N_good, actual_max_pts, n_fields], dtype=np.int32),
        )

        # -- verbatim copy for non-N datasets (excluding 'shape') -----------
        for k in skip_keys:
            if k == "shape":
                continue
            hin.copy(k, hout)
            print(f"  Copied verbatim : {k}  shape={hin[k].shape}")

    out_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nDone.")
    print(f"  Output size           : {out_mb:.1f} MB")
    print(f"  Output shape dataset  : [{N_good}, {actual_max_pts}, {n_fields}]")

    print(f"\nOutput dataset shapes:")
    with h5py.File(output_path, "r") as hf:
        for k in hf.keys():
            print(f"  {k:<20s}: {hf[k].shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect (and optionally remove) showers that fail the "
            "--num-layer-cond isolation filter from a merged HDF5 file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                        help="Path to the merged HDF5 file.")
    parser.add_argument("--num-layer-cond", type=int, required=True,
                        help="Layer-window size. Showers with any layer isolated "
                             "beyond +/-(num_layer_cond//2) are flagged. "
                             "E.g. 8 -> +/-4 layers.")
    parser.add_argument("--remove", action="store_true",
                        help="Remove bad showers and write a new file to --output.")
    parser.add_argument("--output", default=None,
                        help="Output HDF5 path (required when --remove is set).")
    parser.add_argument("--with-time", action="store_true",
                        help="Showers have 5 fields/point [x,y,layer,energy,time].")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Showers per chunk (default 512). "
                             "Larger = faster but more RAM.")
    args = parser.parse_args()

    if args.remove and args.output is None:
        parser.error("--output is required when --remove is set.")
    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: input file not found: {args.input}")

    half     = args.num_layer_cond // 2
    n_fields = 5 if args.with_time else 4

    good_mask = inspect(args.input, half, n_fields, args.chunk_size)

    if args.remove:
        if os.path.abspath(args.output) == os.path.abspath(args.input):
            sys.exit("ERROR: --output must differ from --input.")
        write_filtered(args.input, args.output, good_mask, n_fields, args.chunk_size)
    else:
        print("\n(Run with --remove --output <path> to write a filtered file.)")


if __name__ == "__main__":
    main()