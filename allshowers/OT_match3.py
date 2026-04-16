import argparse
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from typing import Iterator

import numpy as np
import numpy.typing as npt
import ot
import torch

'''
python /n/home04/hhanif/AllShowers/allshowers/OT_match3.py --preprocessed-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/preprocessed_electrons/ --submit-slurm
'''



START_TIME = time.time()


def log(*args) -> None:
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:7.2f}s]", *args, flush=True)


SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name=ot_shards
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -p serial_requeue
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/ot_shards_%A_%a.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/ot_shards_%A_%a.err
#SBATCH --array=0-{max_index}

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OT matching on preprocessed .pt shards. "
            "One Slurm array task processes exactly one shard and saves noise back into it."
        )
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        required=True,
        help="Directory containing .pt shard files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size inside each shard.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Multiprocessing workers used within each shard job.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Index of shard to process from the sorted .pt file list.",
    )
    parser.add_argument(
        "--submit-slurm",
        action="store_true",
        help="Submit one Slurm array task per shard and exit.",
    )
    return parser.parse_args()


def list_shards(preprocessed_dir: str) -> list[str]:
    shard_files = sorted(
        f for f in os.listdir(preprocessed_dir)
        if f.endswith(".pt") and (f.startswith("train_") or f.startswith("val_"))
    )
    if not shard_files:
        raise FileNotFoundError(
            "No train_/val_ .pt shard files found in "
            f"{preprocessed_dir}"
        )
    return shard_files


def iter_shard_batches(
    shard_path: str,
    batch_size: int,
) -> Iterator[tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]]]:
    data = torch.load(shard_path, weights_only=False)

    if "x" not in data or "mask" not in data or "layer" not in data:
        raise KeyError(f"{shard_path} must contain keys: x, mask, layer")

    x = data["x"].cpu().numpy()          # [N, P, F]
    mask = data["mask"].cpu().numpy()    # [N, P, 1] or [N, P]
    layer = data["layer"].cpu().numpy()  # [N, P, 1] or [N, P]

    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    if layer.ndim == 3:
        layer = layer.squeeze(-1)

    n_events = x.shape[0]
    for start in range(0, n_events, batch_size):
        end = min(start + batch_size, n_events)
        yield (
            x[start:end].transpose(0, 2, 1).astype(np.float32, copy=False),  # [B, F, P]
            mask[start:end].astype(np.bool_, copy=False),                    # [B, P]
            layer[start:end].astype(np.int64, copy=False),                   # [B, P]
        )


def match_noise_batch(
    points: npt.NDArray[np.float32],   # [B, F, P]
    mask: npt.NDArray[np.bool_],       # [B, P]
    layer: npt.NDArray[np.int64],      # [B, P]
    num_layers: int,
    num_features: int,
) -> npt.NDArray[np.float32]:
    noise = np.random.randn(points.shape[0], num_features, points.shape[2]).astype(np.float32)

    for layer_id in range(num_layers):
        mask_local = np.expand_dims(np.logical_and(mask, layer == layer_id), axis=1)  # [B,1,P]

        for j in range(points.shape[0]):
            selector = mask_local[j].repeat(num_features, axis=0).T.reshape(-1, num_features)

            points_j = points[j].T[selector].reshape(-1, num_features)
            noise_j = noise[j].T[selector].reshape(-1, num_features)

            if len(points_j) <= 1:
                continue

            n = len(points_j)
            cost = np.sqrt(
                np.sum((points_j[:, None, :] - noise_j[None, :, :]) ** 2, axis=-1)
            )

            wa = np.ones(n, dtype=np.float64) / n
            wb = np.ones(n, dtype=np.float64) / n

            transport = ot.emd(wa, wb, cost, numItermax=1_000_000)
            matched_noise = n * (transport @ noise_j)

            noise[j].T[selector] = matched_noise.flatten()

    noise[(~mask[:, None, :]).repeat(num_features, axis=1)] = 0.0
    return noise.astype(np.float32, copy=False)


class ShardNoiseMatcher:
    def __init__(self, num_layers: int, num_features: int) -> None:
        self.num_layers = num_layers
        self.num_features = num_features

    def __call__(
        self,
        batch: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]],
    ) -> npt.NDArray[np.float32]:
        points, mask, layer = batch
        return match_noise_batch(
            points=points,
            mask=mask,
            layer=layer,
            num_layers=self.num_layers,
            num_features=self.num_features,
        )


def process_one_shard(
    shard_path: str,
    batch_size: int,
    num_workers: int,
) -> None:
    log(f"Processing shard: {shard_path}")

    meta = torch.load(shard_path, weights_only=False)
    if "x" not in meta or "mask" not in meta or "layer" not in meta:
        raise KeyError(f"{shard_path} must contain keys: x, mask, layer")

    num_events = int(meta["x"].shape[0])
    num_features = int(meta["x"].shape[-1])
    num_layers = int(meta["layer"].max().item()) + 1
    del meta

    log(f"num_events   = {num_events}")
    log(f"num_features = {num_features}")
    log(f"num_layers   = {num_layers}")
    log(f"batch_size   = {batch_size}")
    log(f"num_workers  = {num_workers}")

    matcher = ShardNoiseMatcher(num_layers=num_layers, num_features=num_features)
    noise_parts: list[npt.NDArray[np.float32]] = []

    with mp.Pool(processes=num_workers) as pool:
        for batch_id, batch_noise in enumerate(pool.imap(matcher, iter_shard_batches(shard_path, batch_size)), start=1):
            noise_parts.append(batch_noise.transpose(0, 2, 1))  # [B, P, F]
            log(f"finished batch {batch_id}")

    if not noise_parts:
        raise RuntimeError(f"No data processed for shard: {shard_path}")

    noise = np.concatenate(noise_parts, axis=0)
    noise_tensor = torch.from_numpy(noise)

    log("Saving noise back into shard...")
    data = torch.load(shard_path, weights_only=False)
    data["noise"] = noise_tensor
    torch.save(data, shard_path)

    log(f"Saved noise to {shard_path}")


def submit_slurm_array(
    preprocessed_dir: str,
    batch_size: int,
    num_workers: int,
) -> None:
    shard_files = list_shards(preprocessed_dir)
    max_index = len(shard_files) - 1
    script_path = os.path.abspath(__file__)

    command = (
        f"python {script_path} "
        f"--preprocessed-dir {preprocessed_dir} "
        f"--batch-size {batch_size} "
        f"--num-workers {num_workers} "
        f"--shard-index $SLURM_ARRAY_TASK_ID\n"
    )

    script = SLURM_HEADER.format(max_index=max_index) + "\n" + command

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, prefix="ot_shards_") as f:
        f.write(script)
        tmp_script = f.name

    log(f"Submitting {len(shard_files)} shard jobs...")
    print("-" * 80)
    print(script)
    print("-" * 80)

    result = subprocess.run(["sbatch", tmp_script], capture_output=True, text=True)
    os.unlink(tmp_script)

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    log(result.stdout.strip())


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()

    log("Arguments:", args)

    shard_files = list_shards(args.preprocessed_dir)
    log(f"Found {len(shard_files)} shard(s)")

    if args.submit_slurm:
        submit_slurm_array(
            preprocessed_dir=args.preprocessed_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return

    if args.shard_index is None:
        raise ValueError("Use --submit-slurm to submit all shards, or provide --shard-index to process one shard.")

    if args.shard_index < 0 or args.shard_index >= len(shard_files):
        raise IndexError(
            f"--shard-index {args.shard_index} out of range. Valid range: 0 to {len(shard_files) - 1}"
        )

    shard_path = os.path.join(args.preprocessed_dir, shard_files[args.shard_index])
    log(f"Selected shard index {args.shard_index}: {os.path.basename(shard_path)}")

    process_one_shard(
        shard_path=shard_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
