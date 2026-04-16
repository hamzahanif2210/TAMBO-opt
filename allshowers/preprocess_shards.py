"""Offline preprocessing: read raw H5, apply transformations, write .pt shards.

Usage
-----
    python -m allshowers.preprocess_shards conf/allshowers_photons.yaml \\
        --output-dir /path/to/preprocessed_photons \\
        --train-events 700000 \\
        --val-events 30000 \\
        --shard-size 70000

python /n/home04/hhanif/AllShowers/allshowers/preprocess_shards.py /n/home04/hhanif/AllShowers/conf/AllShowers_transformation/allshowers_electrons.yaml \
   --output-dir  /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/preprocessed_electrons \
        --train-events 700000 \
        --val-events 30000 \
        --shard-size 30000

This reads the H5 file, lets you specify exactly how many events to use
for training and validation, transforms them, and writes .pt shards:

    /path/to/preprocessed_photons/
        train_000.pt   # first 70k training samples (preprocessed)
        train_001.pt
        ...
        val_000.pt     # validation samples
        trafos.pt      # fitted transformations (for generation)
        meta.pt        # metadata (shard sizes, num_layers, etc.)

Then in the YAML config, replace the data section with:
    data:
      preprocessed_dir: /path/to/preprocessed_photons
"""

import argparse
import os
import sys

import showerdata
import torch
import yaml

from allshowers.data_sets import (
    batched_histogram,
    initialise_trafos,
    load_data,
    to_label_tensor,
)
from allshowers.preprocessing import Identity, compose


def preprocess_chunk(
    path: str,
    start: int,
    stop: int,
    *,
    samples_energy_trafo,
    samples_coordinate_trafo,
    cond_trafo,
    samples_time_trafo=None,
    return_noise: bool = False,
    return_direction: bool = False,
    max_num_points: int | None = None,
    num_layers: int = -1,
):
    """Load a chunk from the H5 file and return preprocessed tensors."""
    with_time = samples_time_trafo is not None

    data = load_data(
        path,
        start=start,
        stop=stop,
        return_noise=return_noise,
        max_num_points=max_num_points,
        with_time=with_time,
    )

    mask = data["shower"][:, :, [3]] > 0
    energy = cond_trafo(data["energy"])

    if with_time:
        x = torch.cat(
            [
                samples_coordinate_trafo(data["shower"][:, :, :2]),
                samples_energy_trafo(data["shower"][:, :, [3]]),
                samples_time_trafo(data["shower"][:, :, [4]]),
            ],
            dim=-1,
        )
        x[~mask.repeat(1, 1, 4)] = 0.0
    else:
        x = torch.cat(
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
        cond = torch.cat([energy, data["direction"]], dim=-1)
    else:
        cond = energy

    return {
        "x": x,
        "cond": cond,
        "num_points": num_points,
        "layer": layer,
        "mask": mask,
        "label": label if label is not None else torch.zeros(len(x), dtype=torch.int64),
        "noise": data["noise"],
    }


def write_shards(
    path: str,
    start: int,
    stop: int,
    output_dir: str,
    prefix: str,
    shard_size: int,
    **kwargs,
) -> list[str]:
    """Process a data range in chunks and write .pt shard files."""
    shard_files = []
    shard_idx = 0
    total = stop - start

    for chunk_start in range(0, total, shard_size):
        chunk_end = min(chunk_start + shard_size, total)
        abs_start = start + chunk_start
        abs_stop = start + chunk_end
        n = abs_stop - abs_start

        print(f"  {prefix}_{shard_idx:03d}: samples [{abs_start}, {abs_stop}) ({n} samples)")
        sys.stdout.flush()

        data = preprocess_chunk(path, abs_start, abs_stop, **kwargs)
        shard_file = os.path.join(output_dir, f"{prefix}_{shard_idx:03d}.pt")
        torch.save(data, shard_file)
        shard_files.append(shard_file)
        shard_idx += 1

        # Free memory
        del data

    return shard_files


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess H5 data into .pt shards for fast training"
    )
    parser.add_argument("param_file", type=str, help="YAML config file")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write preprocessed shards",
    )
    parser.add_argument(
        "--train-events", type=int, required=True,
        help="Number of events/showers to use for training",
    )
    parser.add_argument(
        "--val-events", type=int, required=True,
        help="Number of events/showers to use for validation",
    )
    parser.add_argument(
        "--shard-size", type=int, default=70_000,
        help="Number of samples per shard file (default: 70000)",
    )
    parser.add_argument(
        "--fit-samples", type=int, default=100_000,
        help="Number of samples to use for fitting transformations (default: 100000)",
    )
    parsed = parser.parse_args(args)

    with open(parsed.param_file) as f:
        conf = yaml.safe_load(f)

    data_conf = conf["data"].copy()
    path = data_conf["path"]
    num_layers = conf["model"].get("num_layers", -1)

    # --- Parse trafos from config ---
    se_trafo = compose(data_conf["samples_energy_trafo"]) if "samples_energy_trafo" in data_conf else Identity()
    sc_trafo = compose(data_conf["samples_coordinate_trafo"]) if "samples_coordinate_trafo" in data_conf else Identity()
    c_trafo = compose(data_conf["cond_trafo"]) if "cond_trafo" in data_conf else Identity()
    st_trafo = compose(data_conf["samples_time_trafo"]) if "samples_time_trafo" in data_conf else None

    return_noise = data_conf.get("return_noise", False)
    return_direction = data_conf.get("return_direction", False)
    max_num_points = data_conf.get("max_num_points", None)

    # --- Validate event counts against file ---
    file_len = showerdata.get_file_shape(path)[0]
    total_requested = parsed.train_events + parsed.val_events

    print(f"Data file:        {path}")
    print(f"Events in file:   {file_len}")
    print(f"Train events:     {parsed.train_events}")
    print(f"Val events:       {parsed.val_events}")
    print(f"Total requested:  {total_requested}")
    print(f"Shard size:       {parsed.shard_size}")
    print(f"Output dir:       {parsed.output_dir}")
    print()

    if total_requested > file_len:
        print(
            f"ERROR: Requested {total_requested} events "
            f"(train={parsed.train_events} + val={parsed.val_events}) "
            f"but file only has {file_len} events."
        )
        sys.exit(1)

    train_start = 0
    train_stop = parsed.train_events
    val_start = parsed.train_events
    val_stop = parsed.train_events + parsed.val_events

    print(f"Train range: [{train_start}, {train_stop})")
    print(f"Val range:   [{val_start}, {val_stop})")
    print()
    sys.stdout.flush()

    # --- Fit transformations ---
    os.makedirs(parsed.output_dir, exist_ok=True)
    trafos_file = os.path.join(parsed.output_dir, "trafos.pt")

    fit_stop = min(parsed.fit_samples, train_stop)
    print(f"Fitting transformations on first {fit_stop} events...")
    sys.stdout.flush()
    with_time = st_trafo is not None
    fit_data = load_data(
        path,
        start=0,
        stop=fit_stop,
        return_noise=False,
        max_num_points=max_num_points,
        with_time=with_time,
    )
    mask = fit_data["shower"][:, :, [3]] > 0
    initialise_trafos(
        fit_data["energy"],
        fit_data["shower"],
        mask,
        se_trafo,
        sc_trafo,
        c_trafo,
        st_trafo,
        trafos_file=trafos_file,
    )
    del fit_data, mask
    print(f"Saved trafos to {trafos_file}")
    print()
    sys.stdout.flush()

    common_kwargs = dict(
        samples_energy_trafo=se_trafo,
        samples_coordinate_trafo=sc_trafo,
        cond_trafo=c_trafo,
        samples_time_trafo=st_trafo,
        return_noise=return_noise,
        return_direction=return_direction,
        max_num_points=max_num_points,
        num_layers=num_layers,
    )

    # --- Write training shards ---
    print(f"Writing training shards ({parsed.train_events} events)...")
    sys.stdout.flush()
    train_shards = write_shards(
        path, train_start, train_stop,
        parsed.output_dir, "train", parsed.shard_size,
        **common_kwargs,
    )
    print(f"Wrote {len(train_shards)} training shards")
    print()
    sys.stdout.flush()

    # --- Write validation shards ---
    print(f"Writing validation shards ({parsed.val_events} events)...")
    sys.stdout.flush()
    val_shards = write_shards(
        path, val_start, val_stop,
        parsed.output_dir, "val", parsed.shard_size,
        **common_kwargs,
    )
    print(f"Wrote {len(val_shards)} validation shards")
    print()
    sys.stdout.flush()

    # --- Write metadata ---
    meta = {
        "file_len": file_len,
        "train_events": parsed.train_events,
        "val_events": parsed.val_events,
        "shard_size": parsed.shard_size,
        "num_train_shards": len(train_shards),
        "num_val_shards": len(val_shards),
        "num_layers": num_layers,
        "return_noise": return_noise,
        "return_direction": return_direction,
        "max_num_points": max_num_points,
        "source_path": path,
    }
    meta_file = os.path.join(parsed.output_dir, "meta.pt")
    torch.save(meta, meta_file)
    print(f"Saved metadata to {meta_file}")
    print("Done!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
