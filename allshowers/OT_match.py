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
        "--continuous-z",
        action="store_true",
        default=False,
        help=(
            "Treat Z as a continuous coordinate instead of a discrete layer index. "
            "Z is kept as a feature and OT matching is done globally, not per-layer. "
            "Features become (x, y, z, e[, t]) instead of (x, y, e[, t])."
        ),
    )
    return parser.parse_args(args)


class PreProcessor:
    def __init__(
        self, config_file: str, with_time: bool = False, continuous_z: bool = False
    ) -> None:
        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.with_time = with_time
        self.continuous_z = continuous_z
        # Feature count: base (x,y,e)=3, +z if continuous, +t if with_time
        self.num_features = 3 + int(continuous_z) + int(with_time)

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

        num_coord = 3 if continuous_z else 2
        self.samples_coordinate_trafo.fit(
            x=showers[:, :, :num_coord],
            mask=mask[:, :, None].repeat(1, 1, num_coord),
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

        # Extract layer from z (col 2) before any transformation
        layer = (x_tensor[:, 2] + 0.5).to(torch.int64)

        if self.continuous_z:
            # Transform x, y, z (cols 0, 1, 2) together
            x_tensor[:, :3] = self.samples_coordinate_trafo(
                x_tensor[:, :3].permute(0, 2, 1)
            ).permute(0, 2, 1)
        else:
            # Transform x, y (cols 0, 1) only
            x_tensor[:, :2] = self.samples_coordinate_trafo(
                x_tensor[:, :2].permute(0, 2, 1)
            ).permute(0, 2, 1)

        # Transform e (col 3)
        x_tensor[:, 3] = self.samples_energy_trafo(x_tensor[:, 3])

        if self.continuous_z:
            if self.with_time:
                x_tensor[:, 4] = self.samples_time_trafo(x_tensor[:, 4])
                # Keep x, y, z, e, t → [batch, 5, points]
                x_tensor = x_tensor[:, [0, 1, 2, 3, 4]]
            else:
                # Keep x, y, z, e → [batch, 4, points]
                x_tensor = x_tensor[:, [0, 1, 2, 3]]
        else:
            if self.with_time:
                x_tensor[:, 4] = self.samples_time_trafo(x_tensor[:, 4])
                # Drop z: keep x, y, e, t → [batch, 4, points]
                x_tensor = x_tensor[:, [0, 1, 3, 4]]
            else:
                # Original: drop z, keep x, y, e → [batch, 3, points]
                x_tensor = x_tensor[:, [0, 1, 3]]

        return x_tensor.numpy(), mask.numpy(), layer.numpy()


class DataLoader(Iterable[npt.NDArray[np.float32]]):
    def __init__(self, data_file: str, batch_size: int) -> None:
        self.file_name = data_file
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[npt.NDArray[np.float32]]:
        with showerdata.ShowerDataFile(self.file_name, "r") as file:
            for start in range(0, len(file), self.batch_size):
                end = min(start + self.batch_size, len(file))
                samples = file[start:end].points
                # Transpose: [batch, points, N] → [batch, N, points]
                yield samples.transpose(0, 2, 1)


class NoiseMatcher:
    def __init__(self, pre_processor: PreProcessor) -> None:
        self.__num_layers = pre_processor.num_layers
        self.__num_features = pre_processor.num_features
        self.__continuous_z = pre_processor.continuous_z
        self.pre_processor = pre_processor

    def _match_subset(
        self,
        points: npt.NDArray[np.float32],
        noise: npt.NDArray[np.float32],
        mask_local: npt.NDArray[np.bool_],
        F: int,
        j: int,
    ) -> None:
        """Run OT matching for a subset of points defined by mask_local."""
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

    def __call__(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        points, mask, layer = self.pre_processor(samples)
        # points shape: [batch, F, points]

        F = self.__num_features
        noise = np.random.randn(points.shape[0], F, points.shape[2])

        if self.__continuous_z:
            # Global matching: match all points at once (not per-layer)
            for j in range(len(points)):
                mask_local = np.expand_dims(mask[j], 0)  # [1, points]
                self._match_subset(points, noise, mask_local, F, j)
        else:
            # Per-layer matching (original)
            for i in range(self.__num_layers):
                mask_local = np.expand_dims(np.logical_and(mask, layer == i), 1)
                for j in range(len(points)):
                    self._match_subset(points, noise, mask_local, F, j)

        # Zero out padding positions
        noise[(~mask[:, None, :]).repeat(F, axis=1)] = 0.0
        return noise.astype(np.float32, copy=False)


def process_file(
    data_file,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    batch_size: int = 128,
) -> None:
    F = pre_processor.num_features
    num_batches = -(-data_shape[0] // batch_size)
    print_time("batch size:", batch_size)
    print_time("number of batches:", num_batches)
    print_time(f"num features: {F}  {'(x, y, e, t)' if F == 4 else '(x, y, e)'}")
    sys.stdout.flush()

    noise_matcher = NoiseMatcher(pre_processor)
    # Noise buffer: [N, F, points]
    noise = np.empty((data_shape[0], F, data_shape[1]), dtype=np.float32)
    print_time(f"NoiseMatcher initialized. (noise shape={noise.shape})")
    sys.stdout.flush()

    num_processes = n - 1 if (n := os.cpu_count()) else 1
    with multiprocessing.Pool(num_processes) as pool:
        for i, batch in enumerate(
            pool.imap(
                noise_matcher,
                DataLoader(data_file, batch_size),
            )
        ):
            noise[i * batch_size : i * batch_size + len(batch)] = batch
    print_time("All batches processed.")
    sys.stdout.flush()

    # Transpose back: [N, F, points] → [N, points, F] for saving
    noise = noise.transpose(0, 2, 1)
    showerdata.save_target(noise, data_file, overwrite=True)

    print_time(f"Noise saved successfully to {data_file} (shape={noise.shape}).")
    sys.stdout.flush()


@torch.inference_mode()
def main(args: list[str] | None = None):
    torch.set_num_threads(1)

    parsed_args = parse_args(args)
    print_time("Parsing arguments:", parsed_args)
    z_str = "z, " if parsed_args.continuous_z else ""
    t_str = ", t" if parsed_args.with_time else ""
    print_time(f"Mode: (x, y, {z_str}e{t_str})")
    print_time(f"continuous_z: {parsed_args.continuous_z}")
    sys.stdout.flush()

    pre_processor = PreProcessor(
        parsed_args.file,
        with_time=parsed_args.with_time,
        continuous_z=parsed_args.continuous_z,
    )
    print_time("PreProcessor initialized.")
    sys.stdout.flush()

    print_time("Processing file")
    sys.stdout.flush()
    process_file(
        data_file=pre_processor.file_path,
        data_shape=pre_processor.data_shape,
        pre_processor=pre_processor,
        batch_size=128,
    )


if __name__ == "__main__":
    main()