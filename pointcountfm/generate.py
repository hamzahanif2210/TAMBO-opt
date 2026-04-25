import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from pointcountfm import flow_matching, models
from pointcountfm.preprocessing import Transformation

'''
python /n/home04/hhanif/TAMBO-opt/pointcountfm/generate.py \
    --run-dir /n/home04/hhanif/TAMBO-opt/results/20260416_050139_Electron-PointCountFM \
    --num-samples 10000 \
    --pdg-codes 0 1

Output is written to <run-dir>/generated_00.h5, generated_01.h5, ...
'''

# ── Physics defaults ──────────────────────────────────────────────────────────
E_MIN        = 1e5   # GeV
E_MAX        = 1e8   # GeV
ZENITH_MIN   = 60.0  # degrees
ZENITH_MAX   = 100.0 # degrees
AZIMUTH_MIN  = 0.0   # degrees
AZIMUTH_MAX  = 360.0 # degrees


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _deg_to_rad(angle: float) -> float:
    """Convert degrees → radians if |angle| > 2π, otherwise pass through."""
    if math.isfinite(angle) and abs(angle) > 2 * math.pi + 1e-6:
        return math.radians(angle)
    return angle


def build_direction_vector(zenith: float, azimuth: float) -> np.ndarray:
    """Return a CORSIKA-convention unit direction vector from zenith and azimuth.

        nx = sin(θ) · cos(φ)
        ny = sin(θ) · sin(φ)
        nz = cos(θ)
    """
    theta = _deg_to_rad(zenith)
    phi   = _deg_to_rad(azimuth)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    sin_p, cos_p = math.sin(phi),   math.cos(phi)
    return np.array([sin_t * cos_p, sin_t * sin_p, cos_t], dtype=np.float32)


def sample_conditions(
    num_samples: int,
    pdg_codes: list[int],
    e_min: float,
    e_max: float,
    zenith_min: float,
    zenith_max: float,
    azimuth_min: float,
    azimuth_max: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw num_samples random (energy, label, direction) tuples.

    Returns
    -------
    conditions : Tensor [num_samples, 1 + num_classes + 3]  (float32, ready for model)
    energies   : ndarray [num_samples, 1]  (GeV, raw)
    labels     : ndarray [num_samples]     (int, class index)
    directions : ndarray [num_samples, 3]  (unit vectors)
    """
    num_classes = len(pdg_codes)

    # Log-uniform energy sampling
    log_e = rng.uniform(math.log10(e_min), math.log10(e_max), size=(num_samples, 1))
    energies = (10.0 ** log_e).astype(np.float32)

    # Uniform class labels
    labels = rng.integers(0, num_classes, size=num_samples)

    # Uniform zenith / azimuth
    zeniths  = rng.uniform(zenith_min,  zenith_max,  size=num_samples)
    azimuths = rng.uniform(azimuth_min, azimuth_max, size=num_samples)
    directions = np.stack(
        [build_direction_vector(z, a) for z, a in zip(zeniths, azimuths)],
        axis=0,
    )  # [num_samples, 3]

    labels_t = torch.from_numpy(labels.astype(np.int64))
    conditions = torch.concatenate(
        (
            torch.from_numpy(energies),
            torch.nn.functional.one_hot(labels_t, num_classes=num_classes).to(torch.float32),
            torch.from_numpy(directions),
        ),
        dim=1,
    )

    return conditions, energies, labels, directions


# ── Model helpers (shared with inference_cond_file.py) ────────────────────────

def build_model(config: dict) -> flow_matching.CNF:
    model_config = config["model"].copy()
    flow_config  = model_config.pop("flow") if "flow" in model_config else {}
    model_name   = model_config.pop("name")
    model_class  = getattr(models, model_name)
    return flow_matching.CNF(model_class(**model_config), **flow_config)


class Sampler(nn.Module):
    def __init__(
        self,
        model: flow_matching.CNF,
        transform_inc: Transformation,
        transform_num_points: Transformation,
        dim_data: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform_inc = transform_inc
        self.transform_num_points = transform_num_points
        self.dim_data = dim_data
        self.steps = steps

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        condition = torch.clone(condition)
        condition[:, :1] = self.transform_inc(condition[:, :1])
        samples = self.model.sample(
            (condition.shape[0], self.dim_data), self.steps, condition=condition
        )
        return self.transform_num_points.inverse(samples)


def next_output_path(run_dir: str) -> str:
    i = 0
    while True:
        path = os.path.join(run_dir, f"generated_{i:02d}.h5")
        if not os.path.exists(path):
            return path
        i += 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate num_points_per_layer predictions from random conditions."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training result directory (conf.yaml, weights/best.pt, preprocessing/trafos.pt).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of random showers to generate.",
    )
    parser.add_argument(
        "--pdg-codes",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Ordered list of PDG class labels (default: 0 1).",
    )
    parser.add_argument(
        "--e-min",   type=float, default=E_MIN,       help=f"Min energy in GeV (default: {E_MIN:.0e})")
    parser.add_argument(
        "--e-max",   type=float, default=E_MAX,       help=f"Max energy in GeV (default: {E_MAX:.0e})")
    parser.add_argument(
        "--zenith-min",  type=float, default=ZENITH_MIN,  help=f"Min zenith in degrees (default: {ZENITH_MIN})")
    parser.add_argument(
        "--zenith-max",  type=float, default=ZENITH_MAX,  help=f"Max zenith in degrees (default: {ZENITH_MAX})")
    parser.add_argument(
        "--azimuth-min", type=float, default=AZIMUTH_MIN, help=f"Min azimuth in degrees (default: {AZIMUTH_MIN})")
    parser.add_argument(
        "--azimuth-max", type=float, default=AZIMUTH_MAX, help=f"Max azimuth in degrees (default: {AZIMUTH_MAX})")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="ODE integration steps (default: value from training config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible condition sampling.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="Explicit output path. If omitted, saves to <run-dir>/generated_NN.h5.",
    )
    return parser.parse_args(args)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.set_num_threads(1)

    run_dir = args.run_dir
    output_file = args.output if args.output else next_output_path(run_dir)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Output will be written to: {output_file}")

    # ------------------------------------------------------------------ config
    config_path = os.path.join(run_dir, "conf.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    steps = args.steps if args.steps is not None else config["training"].get("steps", 50)

    # ------------------------------------------------------------------ model
    print("Building model from config...")
    model = build_model(config)

    best_ckpt = os.path.join(run_dir, "weights", "best.pt")
    last_ckpt = os.path.join(run_dir, "weights", "last.pt")
    checkpoint_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
    print(f"Loading checkpoint from {checkpoint_path}...")
    start = time.time()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(torch.float32)
    print(f"Checkpoint loaded in {(time.time() - start) * 1000.0:.1f}ms")

    # --------------------------------------------------------------- trafos
    print("Loading transformations...")
    trafos_path = os.path.join(run_dir, "preprocessing", "trafos.pt")
    trafos = torch.load(trafos_path, map_location="cpu", weights_only=False)
    transform_inc: Transformation        = trafos["transform_inc"].to(torch.float32)
    transform_num_points: Transformation = trafos["transform_num_points"].to(torch.float32)

    network = model.network
    dim_data: int = (
        network.output.out_features if hasattr(network, "output")
        else network.network[-1].out_features
    )

    # ----------------------------------------------------------------- sampler
    sampler = Sampler(
        model=model,
        transform_inc=transform_inc,
        transform_num_points=transform_num_points,
        dim_data=dim_data,
        steps=steps,
    )

    # Trace on a single-sample example
    example = torch.concatenate(
        (
            torch.tensor([[50.0]]),
            torch.nn.functional.one_hot(
                torch.tensor([0]), num_classes=len(args.pdg_codes)
            ).to(torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ),
        dim=1,
    )
    print("Tracing inference function...")
    start = time.time()
    with torch.inference_mode():
        sampler_traced = torch.jit.trace(sampler, example, check_trace=False)
        sampler_traced(example)  # warm-up
        sampler_traced(example)
    print(f"Traced and warmed up in {(time.time() - start) * 1000.0:.1f}ms")

    # --------------------------------------------------------- sample conditions
    print(f"Sampling {args.num_samples} random conditions...")
    rng = np.random.default_rng(args.seed)
    conditions, energies, labels, directions = sample_conditions(
        num_samples  = args.num_samples,
        pdg_codes    = args.pdg_codes,
        e_min        = args.e_min,
        e_max        = args.e_max,
        zenith_min   = args.zenith_min,
        zenith_max   = args.zenith_max,
        azimuth_min  = args.azimuth_min,
        azimuth_max  = args.azimuth_max,
        rng          = rng,
    )
    print(f"  energies:   [{energies.min():.2e}, {energies.max():.2e}] GeV")
    print(f"  labels:     {np.bincount(labels).tolist()}  (counts per class)")
    print(f"  directions: shape {directions.shape}")

    # ----------------------------------------------------------------- infer
    print("Running inference...")
    start = time.time()
    with torch.inference_mode():
        results = sampler_traced(conditions)
    elapsed = time.time() - start
    print(f"Inference done in {elapsed:.2f}s  "
          f"({elapsed / args.num_samples * 1000.0:.2f} ms/sample)")

    results_np = (torch.clamp(results, min=0.0) + 0.5).to(torch.int32).numpy()

    # ----------------------------------------------------------------- save
    import h5py
    print(f"Saving to {output_file} ...")
    start = time.time()
    with h5py.File(output_file, "w") as hf:
        hf.create_dataset("energies",   data=energies,    compression="gzip")
        hf.create_dataset("labels",     data=labels.astype(np.int32), compression="gzip")
        hf.create_dataset("directions", data=directions,  compression="gzip")
        obs = hf.create_group("observables")
        obs.create_dataset("num_points_per_layer", data=results_np, compression="gzip")

        # Store generation metadata as attributes on the root group
        hf.attrs["num_samples"]  = args.num_samples
        hf.attrs["e_min"]        = args.e_min
        hf.attrs["e_max"]        = args.e_max
        hf.attrs["zenith_min"]   = args.zenith_min
        hf.attrs["zenith_max"]   = args.zenith_max
        hf.attrs["azimuth_min"]  = args.azimuth_min
        hf.attrs["azimuth_max"]  = args.azimuth_max
        hf.attrs["pdg_codes"]    = args.pdg_codes
        hf.attrs["run_dir"]      = run_dir
        hf.attrs["steps"]        = steps
        if args.seed is not None:
            hf.attrs["seed"] = args.seed

    print(f"Saved in {(time.time() - start) * 1000.0:.1f}ms")
    print(f"\nDone.  Output: {output_file}")


if __name__ == "__main__":
    main()