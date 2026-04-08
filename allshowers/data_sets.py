import os
import time
import warnings
from collections.abc import Callable
from typing import TypedDict

import showerdata
import torch
from torch import Tensor

from allshowers.data_loader import (
    ChunkedDataLoader,
    DataLoader,
    DictDataSet,
    ModelInputDict,
)
from allshowers.preprocessing import Identity, Transformation, compose

__all__ = ["create_label_list", "to_label_tensor", "get_data_loaders"]


class ShowerDict(TypedDict):
    shower: Tensor
    energy: Tensor
    direction: Tensor
    pdg: Tensor
    noise: Tensor | None


def batched_histogram(
    data: torch.Tensor, mask: torch.Tensor, num_bins: int = -1
) -> torch.Tensor:
    if num_bins < 0:
        num_bins = int(torch.max(data[mask]).item()) + 1
    histograms = torch.zeros(size=(data.shape[0], num_bins), dtype=torch.int32)
    ones = torch.zeros(size=data.shape, dtype=histograms.dtype)
    ones[mask] = 1
    histograms.scatter_add_(1, data, ones)
    return histograms


@torch.no_grad()
def initialise_trafos(
    energies: Tensor,
    showers: Tensor,
    mask: Tensor,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    samples_time_trafo: Transformation | None = None,   # ADD: optional time trafo
    *,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    if trafos_file is None and world_size > 1:
        raise ValueError(
            "If using distributed training, a trafos_file must be provided to save and load the transformations."
        )
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
    if rank != 0:
        torch.distributed.barrier(device_ids=[local_rank])
    if os.path.isfile(trafos_file):
        if world_size > 1 and rank == 0:
            torch.distributed.barrier(device_ids=[local_rank])
        parameters = torch.load(trafos_file, weights_only=True)
        samples_energy_trafo.load_state_dict(parameters["samples_energy_trafo"])
        samples_coordinate_trafo.load_state_dict(parameters["samples_coordinate_trafo"])
        cond_trafo.load_state_dict(parameters["cond_trafo"])
        # Load time trafo state if present in the saved file
        if samples_time_trafo is not None and "samples_time_trafo" in parameters:
            samples_time_trafo.load_state_dict(parameters["samples_time_trafo"])
        print(f"[rank {rank}] Loaded transformations from {trafos_file}")
    else:
        if rank != 0:
            raise RuntimeError(
                "Initialization of transformations is only allowed for rank 0"
            )
        energies_l = energies[:100_000]
        showers_l = showers[:100_000]
        mask_l = mask[:100_000]
        cond_trafo.fit(energies_l)
        samples_coordinate_trafo.fit(showers_l[:, :, :2], mask_l)
        samples_energy_trafo.fit(showers_l[:, :, 3], mask_l.squeeze())
        # Fit time trafo on col 4 if provided
        if samples_time_trafo is not None:
            samples_time_trafo.fit(showers_l[:, :, 4], mask_l.squeeze())
        if trafos_file:
            parameters = {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
            }
            # Save time trafo state alongside the others
            if samples_time_trafo is not None:
                parameters["samples_time_trafo"] = samples_time_trafo.state_dict()
            torch.save(parameters, trafos_file)
            print(f"[rank {rank}] Saved transformations to {trafos_file}")
        if world_size > 1:
            time.sleep(5)  # make sure file is on network drive
            torch.distributed.barrier(device_ids=[local_rank])


def load_data(
    path: str,
    *,
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    max_num_points: int | None = None,
    with_time: bool = False,    # ADD: controls whether col 4 (time) is kept
) -> ShowerDict:
    showers = showerdata.load(
        path,
        start,
        stop,
        max_points=max_num_points,
    )
    if return_noise:
        import numpy as np

        noise, _ = showerdata.load_target(path, "target", start=start, stop=stop)
        target_pts = showers.points.shape[1]
        n_pts = noise.shape[1]
        if n_pts > target_pts:
            noise = noise[:, :target_pts, :]
        elif n_pts < target_pts:
            pad = np.zeros(
                (noise.shape[0], target_pts - n_pts, noise.shape[2]),
                dtype=noise.dtype,
            )
            noise = np.concatenate([noise, pad], axis=1)
    else:
        noise = None

    if with_time:
        # Keep all 5 columns: x, y, z, e, t
        if showers.points.shape[2] < 5:
            raise ValueError(
                f"with_time=True requires data with 5 columns (x, y, z, e, t), "
                f"but file has shape {showers.points.shape}."
            )
        # No truncation — keep col 4 (time)
    else:
        # Original behaviour: drop col 4 if present
        if showers.points.shape[2] == 5:
            showers.points = showers.points[:, :, :4]

    data = ShowerDict(
        shower=torch.from_numpy(showers.points),
        energy=torch.from_numpy(showers.energies),
        direction=torch.from_numpy(showers.directions),
        pdg=torch.from_numpy(showers.pdg),
        noise=torch.from_numpy(noise) if noise is not None else None,
    )

    return data


@torch.no_grad()
def create_label_list(
    pdg: torch.Tensor,
) -> list[int]:
    unique_pdg = pdg.unique().tolist()
    unique_pdg.sort(key=lambda x: (abs(x), -x))
    return unique_pdg


@torch.no_grad()
def to_label_tensor(
    pdg: torch.Tensor | None,
    label_list: list[int] | None = None,
) -> torch.Tensor | None:
    if pdg is None:
        return None
    if label_list is None:
        label_list = create_label_list(pdg)
    if max(pdg.shape, default=1) != pdg.numel():
        raise ValueError("pdg must be a 1D tensor.")
    pdg = pdg.view(-1)
    label_tensor = torch.zeros(pdg.shape[0], dtype=torch.int64)
    for i, label in enumerate(label_list):
        label_tensor[pdg == label] = i
    return label_tensor


@torch.no_grad()
def load_and_prepare(
    path: str,
    *,
    samples_energy_trafo: Transformation = Identity(),
    samples_coordinate_trafo: Transformation = Identity(),
    cond_trafo: Transformation = Identity(),
    samples_time_trafo: Transformation | None = None,   # ADD: None = original mode
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    return_direction: bool = False,
    max_num_points: int | None = None,
    num_layers: int = -1,
    do_initialise_trafos: bool = True,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> ModelInputDict:
    with_time = samples_time_trafo is not None

    data = load_data(
        path,
        start=start,
        stop=stop,
        return_noise=return_noise,
        max_num_points=max_num_points,
        with_time=with_time,
    )

    # Mask is always based on energy (col 3) regardless of time
    mask = data["shower"][:, :, [3]] > 0

    if do_initialise_trafos:
        initialise_trafos(
            data["energy"],
            data["shower"],
            mask,
            samples_energy_trafo,
            samples_coordinate_trafo,
            cond_trafo,
            samples_time_trafo,            # passed through; None in original mode
            trafos_file=trafos_file,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    energy = cond_trafo(data["energy"])

    if with_time:
        # 4 features: x, y, e, t   (z/layer stored separately)
        x = torch.concat(
            [
                samples_coordinate_trafo(data["shower"][:, :, :2]),     # x, y
                samples_energy_trafo(data["shower"][:, :, [3]]),         # e
                samples_time_trafo(data["shower"][:, :, [4]]),           # t
            ],
            dim=-1,
        )
        x[~mask.repeat(1, 1, 4)] = 0.0
    else:
        # Original: 3 features: x, y, e
        x = torch.concat(
            [
                samples_coordinate_trafo(data["shower"][:, :, :2]),
                samples_energy_trafo(data["shower"][:, :, [3]]),
            ],
            dim=-1,
        )
        x[~mask.repeat(1, 1, 3)] = 0.0

    layer = (data["shower"][:, :, [2]] + 0.1).long()
    num_points = batched_histogram(
        data=layer.squeeze(dim=-1),
        mask=mask.squeeze(dim=-1),
        num_bins=num_layers,
    )
    label = to_label_tensor(data["pdg"])

    if return_direction:
        cond = torch.concat([energy, data["direction"]], dim=-1)
    else:
        cond = energy

    return ModelInputDict(
        x=x,
        cond=cond,
        num_points=num_points,
        layer=layer,
        mask=mask,
        label=label if label is not None else torch.zeros(0, dtype=torch.int64),
        noise=data["noise"],
    )


def _init_trafos_from_sample(
    config_dataset: dict,
    trafos_file: str,
    rank: int,
    world_size: int,
    local_rank: int,
    sample_size: int = 100_000,
) -> None:
    """Fit transformations on a small sample without loading the full dataset.

    Only used in chunked mode to initialise trafos before streaming begins.
    Loads at most ``sample_size`` samples from the beginning of the file.
    """
    load_and_prepare(
        **config_dataset,
        start=0,
        stop=sample_size,
        trafos_file=trafos_file,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )


def get_data_loaders(
    config_dataset: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[DataLoader | ChunkedDataLoader, DataLoader | ChunkedDataLoader, dict[str, Transformation]]:
    config_dataset = config_dataset.copy()
    data_len = showerdata.get_file_shape(config_dataset["path"])[0]
    if "stop" in config_dataset:
        data_len = min(data_len, config_dataset["stop"])
        del config_dataset["stop"]
    if "val_len" in config_dataset:
        val_len = config_dataset.pop("val_len")
        if val_len > data_len // 2:
            warnings.warn(
                f"val_len {val_len} is larger than 50% of data length {data_len // 2},"
                f" reducing to {data_len // 2}.",
                UserWarning,
            )
            val_len = min(val_len, data_len // 2)
    else:
        val_len = data_len // 10
    split = data_len - val_len

    # Extract chunk_size if present — triggers chunked loading mode
    chunk_size = config_dataset.pop("chunk_size", None)

    if "samples_energy_trafo" in config_dataset:
        config_dataset["samples_energy_trafo"] = compose(
            config_dataset["samples_energy_trafo"]
        )
    if "samples_coordinate_trafo" in config_dataset:
        config_dataset["samples_coordinate_trafo"] = compose(
            config_dataset["samples_coordinate_trafo"]
        )
    if "cond_trafo" in config_dataset:
        config_dataset["cond_trafo"] = compose(config_dataset["cond_trafo"])
    # Wire up time trafo from config if present; otherwise stays absent (original mode)
    if "samples_time_trafo" in config_dataset:
        config_dataset["samples_time_trafo"] = compose(
            config_dataset["samples_time_trafo"]
        )

    start = rank * (split // world_size)
    stop = (rank + 1) * (split // world_size)

    if chunk_size is not None:
        # ---- Chunked loading mode for large datasets ----
        # 1) Fit transformations on a small sample first
        _init_trafos_from_sample(
            config_dataset,
            trafos_file,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

        # 2) Build a load function that loads & transforms a [start, stop) slice
        def _make_load_fn(
            cfg: dict, offset: int, tf: str
        ) -> Callable[[int, int], ModelInputDict]:
            def load_fn(chunk_start: int, chunk_stop: int) -> ModelInputDict:
                return load_and_prepare(
                    **cfg,
                    start=offset + chunk_start,
                    stop=offset + chunk_stop,
                    trafos_file=tf,
                    do_initialise_trafos=False,
                )
            return load_fn

        total_train = stop - start
        loader_train = ChunkedDataLoader(
            load_fn=_make_load_fn(config_dataset, start, trafos_file),
            total_samples=total_train,
            chunk_size=chunk_size,
            batch_size=batch_size,
            drop_last=total_train > batch_size,
            shuffle=True,
        )

        if rank == 0:
            loader_test = ChunkedDataLoader(
                load_fn=_make_load_fn(config_dataset, split, trafos_file),
                total_samples=val_len,
                chunk_size=chunk_size,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )
        else:
            loader_test = DataLoader(
                data_set=DictDataSet(
                    ModelInputDict(
                        x=torch.empty(0, 0, 0),
                        cond=torch.empty(0, 0),
                        num_points=torch.empty(0, 0, dtype=torch.int64),
                        layer=torch.empty(0, 0, dtype=torch.int64),
                        mask=torch.empty(0, 0, dtype=torch.bool),
                        label=torch.empty(0, 0, dtype=torch.int64),
                        noise=None,
                    )
                ),
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )
    else:
        # ---- Original in-memory loading mode ----
        data_train = DictDataSet(
            load_and_prepare(
                **config_dataset,
                start=start,
                stop=stop,
                trafos_file=trafos_file,
                world_size=world_size,
                rank=rank,
                local_rank=local_rank,
            )
        )
        loader_train = DataLoader(
            data_set=data_train,
            batch_size=batch_size,
            drop_last=(stop - start) > batch_size,
            shuffle=True,
        )
        if rank == 0:
            data_test = DictDataSet(
                load_and_prepare(
                    **config_dataset,
                    start=split,
                    stop=data_len,
                    trafos_file=trafos_file,
                    do_initialise_trafos=False,
                )
            )
            loader_test = DataLoader(
                data_set=data_test, batch_size=batch_size, drop_last=False, shuffle=False
            )
        else:
            loader_test = DataLoader(
                data_set=DictDataSet(
                    ModelInputDict(
                        x=torch.empty(0, 0, 0),
                        cond=torch.empty(0, 0),
                        num_points=torch.empty(0, 0, dtype=torch.int64),
                        layer=torch.empty(0, 0, dtype=torch.int64),
                        mask=torch.empty(0, 0, dtype=torch.bool),
                        label=torch.empty(0, 0, dtype=torch.int64),
                        noise=None,
                    )
                ),
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )

    trafos = {
        "samples_energy_trafo": config_dataset.get("samples_energy_trafo", Identity()),
        "samples_coordinate_trafo": config_dataset.get(
            "samples_coordinate_trafo", Identity()
        ),
        "cond_trafo": config_dataset.get("cond_trafo", Identity()),
        # Included only when present; generator.py can check with .get()
        **({
            "samples_time_trafo": config_dataset["samples_time_trafo"]
        } if "samples_time_trafo" in config_dataset else {}),
    }
    return loader_train, loader_test, trafos