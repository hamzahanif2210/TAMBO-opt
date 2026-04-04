"""Match noise to points using OT and save it to the file.

Supports two input modes:
  1. Raw H5 file (original): reads from H5, applies transformations, then OT.
  2. Preprocessed .pt shards (--preprocessed-dir): data is already transformed,
     just does OT matching and saves noise back into the shard files.

Usage
-----
    # From raw H5 (original)
    python -m allshowers.OT_match conf/allshowers_photons.yaml --with-time

    # From preprocessed .pt shards
    python -m allshowers.OT_match conf/allshowers_photons.yaml --with-time \
        --preprocessed-dir /path/to/preprocessed_photons
"""

import argparse
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import ot
import showerdata
import torch
import yaml

from allshowers import preprocessing

start = time.time()
batch_type = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]
]


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
            "entire file. Use this for large files to limit memory usage."
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
    return parser.parse_args(args)


# ══════════════════════════════════════════════════════════════════════════════
# Original H5-based classes (unchanged)
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

        # Mask is based on energy (col 3) in both modes
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
        # Load 5 cols when using time, otherwise original 4
        num_cols = 5 if self.with_time else 4
        return config["data"]["path"], showers.points[:, :, :num_cols], data_shape

    def __call__(
        self,
        x: npt.NDArray[np.float32],
    ) -> batch_type:
        # x shape: [batch, 4 or 5, points]  — transposed by DataLoader
        x_tensor = torch.from_numpy(x)

        # Mask on energy (row 3 in transposed layout) — same in both modes
        mask = x_tensor[:, 3] > 0.0

        # Transform x, y (cols 0, 1)
        x_tensor[:, :2] = self.samples_coordinate_trafo(
            x_tensor[:, :2].permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Transform e (col 3)
        x_tensor[:, 3] = self.samples_energy_trafo(x_tensor[:, 3])

        # Extract layer from z (col 2) before dropping it
        layer = (x_tensor[:, 2] + 0.5).to(torch.int64)

        if self.with_time:
            # Transform t (col 4)
            x_tensor[:, 4] = self.samples_time_trafo(x_tensor[:, 4])
            # Drop z: keep x, y, e, t → [batch, 4, points]
            x_tensor = x_tensor[:, [0, 1, 3, 4]]
        else:
            # Original: drop z, keep x, y, e → [batch, 3, points]
            x_tensor = x_tensor[:, [0, 1, 3]]

        return x_tensor.numpy(), mask.numpy(), layer.numpy()


class DataLoader(Iterable[npt.NDArray[np.float32]]):
    def __init__(
        self, data_file: str, batch_size: int,
        start: int = 0, end: int | None = None,
    ) -> None:
        self.file_name = data_file
        self.batch_size = batch_size
        self.start = start
        self.end = end

    def __iter__(self) -> Iterator[npt.NDArray[np.float32]]:
        with showerdata.ShowerDataFile(self.file_name, "r") as file:
            total = self.end if self.end is not None else len(file)
            for start in range(self.start, total, self.batch_size):
                end = min(start + self.batch_size, total)
                samples = file[start:end].points
                # Transpose: [batch, points, N] → [batch, N, points]
                yield samples.transpose(0, 2, 1)


# ══════════════════════════════════════════════════════════════════════════════
# OT noise matching (shared by both paths)
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


class NoiseMatcher:
    """Callable wrapper for multiprocessing — takes raw H5 samples."""

    def __init__(self, pre_processor: PreProcessor) -> None:
        self.__num_layers = pre_processor.num_layers
        self.__num_features = pre_processor.num_features
        self.pre_processor = pre_processor

    def __call__(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        points, mask, layer = self.pre_processor(samples)
        return match_noise_batch(
            points, mask, layer, self.__num_layers, self.__num_features,
        )


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


# ══════════════════════════════════════════════════════════════════════════════
# Original H5 processing path
# ══════════════════════════════════════════════════════════════════════════════

def process_file(
    data_file,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    batch_size: int = 128,
    num_events: int | None = None,
) -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessed-shards processing path
# ══════════════════════════════════════════════════════════════════════════════

def _iter_shard_batches(
    shard_path: str,
    batch_size: int,
) -> Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray]]:
    """Yield (points, mask, layer) batches from a single .pt shard.

    points : [batch, F, max_points]
    mask   : [batch, max_points]
    layer  : [batch, max_points]
    """
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
    batch_size: int = 128,
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

    # Read first shard to get num_layers and num_features
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

        # Collect all noise for this shard
        noise_parts = []
        with multiprocessing.Pool(num_processes) as pool:
            for batch_noise in pool.imap(
                noise_matcher,
                _iter_shard_batches(shard_path, batch_size),
            ):
                # batch_noise: [batch, F, max_points] → [batch, max_points, F]
                noise_parts.append(batch_noise.transpose(0, 2, 1))

        noise = np.concatenate(noise_parts, axis=0)
        noise_tensor = torch.from_numpy(noise)

        # Load shard, add noise, save back
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
    print_time("Parsing arguments:", parsed_args)
    sys.stdout.flush()

    # ── Preprocessed-shards path ──────────────────────────────────────────
    if parsed_args.preprocessed_dir is not None:
        print_time(f"Mode: preprocessed shards from {parsed_args.preprocessed_dir}")
        sys.stdout.flush()

        # Process both train and val shards
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
                    batch_size=128,
                )
            else:
                print_time(f"No {prefix} shards found, skipping.")
        return

    # ── Original H5 path ─────────────────────────────────────────────────
    print_time(
        f"Mode: {'with time (x, y, e, t)' if parsed_args.with_time else 'original (x, y, e)'}"
    )
    sys.stdout.flush()

    pre_processor = PreProcessor(parsed_args.file, with_time=parsed_args.with_time)
    print_time("PreProcessor initialized.")
    sys.stdout.flush()

    num_events = parsed_args.num_events
    if num_events is not None and num_events > pre_processor.data_shape[0]:
        print_time(
            f"WARNING: --num-events {num_events} exceeds file size "
            f"{pre_processor.data_shape[0]}, clamping."
        )
        num_events = pre_processor.data_shape[0]

    print_time("Processing file")
    sys.stdout.flush()
    process_file(
        data_file=pre_processor.file_path,
        data_shape=pre_processor.data_shape,
        pre_processor=pre_processor,
        batch_size=128,
        num_events=num_events,
    )


if __name__ == "__main__":
    main()
