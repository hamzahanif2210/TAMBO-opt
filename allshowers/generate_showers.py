#!/usr/bin/env python3
"""
generate_showers.py
-------------------
Two-stage TAMBO shower generation pipeline.

Stage 1 – PointCountFM  (compiled TorchScript model)
    Predicts the number of secondary particles per detector layer for each
    primary particle.

Stage 2 – AllShowers CNF-Transformer
    Predicts the per-hit features (x, y, layer, energy, time) for every
    secondary particle in each shower.

Both stages share the same primary-particle inputs, which are sampled randomly
inside this script:

    energies    (N, 1)  float32   log-uniform in [E_MIN, E_MAX] GeV
    directions  (N, 3)  float32   CORSIKA unit vector (sin θ cos φ, sin θ sin φ, cos θ)
    labels      (N,)    int64     particle class: 0 or 1

Checkpoint locations (defaults point to the local checkpoints/ directory):
    --point-count-model   path to compiled PointCountFM TorchScript dir
    --allshowers-run-dir  path to AllShowers run dir (conf.yaml, weights/, preprocessing/)

Usage example
-------------
    python generate_showers.py \\
        --num-samples 1000 \\
        --output my_showers.h5 \\
        --num-timesteps 16 \\
        --device cuda:0
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default checkpoint paths (relative to this file)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POINT_COUNT_MODEL = os.path.join(_SCRIPT_DIR, "checkpoints", "num_of_point_clouds_dequantize_compiled.pt")
_DEFAULT_ALLSHOWERS_RUN_DIR = os.path.join(_SCRIPT_DIR, "checkpoints", "all_showers")

# ---------------------------------------------------------------------------
# Primary particle sampling parameters
# ---------------------------------------------------------------------------
E_MIN        = 1e5   # GeV
E_MAX        = 1e8   # GeV
ZENITH_MIN   = 60.0  # degrees
ZENITH_MAX   = 100.0 # degrees
AZIMUTH_MIN  = 0.0   # degrees
AZIMUTH_MAX  = 360.0 # degrees
NUM_CLASSES  = 2     # particle labels: 0 and 1


# ---------------------------------------------------------------------------
# Helpers: direction vector
# ---------------------------------------------------------------------------

def _deg_to_rad(angle: float) -> float:
    """Convert degrees → radians if |angle| > 2π, otherwise pass through."""
    if math.isfinite(angle) and abs(angle) > 2 * math.pi + 1e-6:
        return math.radians(angle)
    return angle


def build_direction_vector(zenith: float, azimuth: float) -> np.ndarray:
    """Return a CORSIKA-convention unit direction vector from zenith and azimuth.

    Both angles are in degrees (auto-converted to radians internally).

        nx = sin(θ) · cos(φ)
        ny = sin(θ) · sin(φ)
        nz = cos(θ)
    """
    theta = _deg_to_rad(zenith)
    phi   = _deg_to_rad(azimuth)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    sin_p, cos_p = math.sin(phi),   math.cos(phi)
    return np.array([sin_t * cos_p, sin_t * sin_p, cos_t], dtype=np.float32)


# ---------------------------------------------------------------------------
# Primary particle sampling
# ---------------------------------------------------------------------------

def sample_primary_particles(
    n: int,
    e_min: float = E_MIN,
    e_max: float = E_MAX,
    zenith_min: float = ZENITH_MIN,
    zenith_max: float = ZENITH_MAX,
    azimuth_min: float = AZIMUTH_MIN,
    azimuth_max: float = AZIMUTH_MAX,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample N random primary cosmic-ray shower parameters.

    Parameters
    ----------
    n          : Number of primary particles.
    e_min/max  : Energy range [GeV] (log-uniform sampling).
    zenith_min/max : Zenith angle range [degrees].
    azimuth_min/max : Azimuth angle range [degrees].
    seed       : Optional RNG seed for reproducibility.

    Returns
    -------
    dict with:
        energies    (N, 1)  float32  – primary energies in GeV
        directions  (N, 3)  float32  – CORSIKA unit direction vectors
        labels      (N,)    int64    – particle class (0 or 1)
    """
    rng = np.random.default_rng(seed)

    # Log-uniform energy sampling
    log_energies = rng.uniform(np.log10(e_min), np.log10(e_max), size=n)
    energies = (10.0 ** log_energies).astype(np.float32).reshape(n, 1)

    # Random azimuth and zenith in configured range
    azimuths = rng.uniform(azimuth_min, azimuth_max, size=n)
    zeniths  = rng.uniform(zenith_min, zenith_max, size=n)
    directions = np.stack(
        [build_direction_vector(float(z), float(a)) for z, a in zip(zeniths, azimuths)],
        axis=0,
    )  # (N, 3)

    # Particle class labels: 0 or 1
    labels = rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)

    return {
        "energies":   torch.from_numpy(energies),
        "directions": torch.from_numpy(directions),
        "labels":     torch.from_numpy(labels),
    }


# ---------------------------------------------------------------------------
# Stage 1 – PointCountFM
# ---------------------------------------------------------------------------

def run_point_count_fm(
    model_path: str,
    energies: torch.Tensor,    # (N, 1) float32
    directions: torch.Tensor,  # (N, 3) float32
    labels: torch.Tensor,      # (N,)   int64
    device: str = "cpu",
) -> torch.Tensor:
    """Run the PointCountFM model to predict num_points per detector layer.

    The model condition is built as:
        [energy (1) | one_hot(label, 2) | direction (3)]  →  shape (N, 6)

    Returns
    -------
    num_points : (N, num_layers) int32  — non-negative integer counts per layer
    """
    print(f"  Loading PointCountFM from {model_path} ...")
    t0 = time.perf_counter()
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print(f"  Loaded in {(time.perf_counter() - t0)*1e3:.1f} ms")

    one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).to(torch.float32)
    conditions = torch.cat(
        [energies.to(torch.float32), one_hot_labels, directions.to(torch.float32)],
        dim=1,
    ).to(device)  # (N, 6)

    inference = torch.jit.script(model.to(torch.float32))

    print(f"  Running inference on {conditions.shape[0]} samples ...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        raw = inference(conditions)
    print(f"  Inference done in {time.perf_counter() - t0:.2f} s")

    num_points = (torch.clamp(raw, min=0.0) + 0.5).to(torch.int32)  # (N, num_layers)
    print(f"  Mean total hits predicted per shower: {num_points.sum(1).float().mean():.1f}")
    return num_points


# ---------------------------------------------------------------------------
# Stage 2 – AllShowers CNF-Transformer
# ---------------------------------------------------------------------------

def run_allshowers(
    run_dir: str,
    energies: torch.Tensor,    # (N, 1)          float32
    directions: torch.Tensor,  # (N, 3)          float32
    labels: torch.Tensor,      # (N,)             int64
    num_points: torch.Tensor,  # (N, num_layers)  int32
    num_timesteps: int = 16,
    batch_size: int = 128,
    solver: str = "midpoint",
    device: str = "cpu",
) -> torch.Tensor:
    """Run the AllShowers CNF-Transformer to generate shower point clouds.

    Returns
    -------
    samples : (N, max_points, 5) float32
              Columns: x, y, z(layer_index), e, t
    """
    # Make sure the package root (parent of allshowers/) is importable
    pkg_root = os.path.dirname(_SCRIPT_DIR)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from allshowers.generator import Generator, generate  # type: ignore

    print(f"  Loading AllShowers generator from {run_dir} ...")
    t0 = time.perf_counter()
    generator = Generator(
        run_dir=run_dir,
        num_timesteps=num_timesteps,
        compile=("cuda" in device.lower()),
        solver=solver,
    )
    generator.eval()
    generator = generator.to(device)
    print(f"  Loaded in {time.perf_counter() - t0:.2f} s")
    print(f"  Time mode: {'ON (x,y,e,t)' if generator.with_time else 'OFF (x,y,e)'}")

    print(f"  Generating {energies.shape[0]} showers "
          f"({num_timesteps} ODE steps, solver={solver}, device={device}) ...")
    t0 = time.perf_counter()
    samples = generate(
        generator=generator,
        energies=energies,
        num_points=num_points,
        angles=directions,     # (N, 3) – 3D unit direction vectors
        batch_size=batch_size,
        device=device,
        labels=labels,
    )
    print(f"  Generation done in {time.perf_counter() - t0:.2f} s")
    return samples  # (N, max_points, 4 or 5)


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_output(
    path: str,
    samples: torch.Tensor,
    energies: torch.Tensor,
    directions: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Save generated showers to an HDF5 file via showerdata.Showers."""
    import showerdata  # type: ignore

    showers = showerdata.Showers(
        points=samples.detach().numpy(),
        energies=energies.detach().numpy(),
        directions=directions.detach().numpy(),
        pdg=labels.detach().numpy().astype(np.int32),
    )
    showers.save(path, overwrite=True)
    print(f"  Saved {len(energies)} showers → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage TAMBO shower generator (PointCountFM → AllShowers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=100,
        help="Number of primary particles (showers) to generate.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="generated_showers.h5",
        help="Output HDF5 file path (written to the current directory).",
    )
    parser.add_argument(
        "--point-count-model", type=str, default=_DEFAULT_POINT_COUNT_MODEL,
        help="Path to the compiled PointCountFM TorchScript model directory.",
    )
    parser.add_argument(
        "--allshowers-run-dir", type=str, default=_DEFAULT_ALLSHOWERS_RUN_DIR,
        help="Path to the AllShowers run directory (must contain conf.yaml, "
             "weights/best.pt, preprocessing/trafos.pt).",
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=16,
        help="ODE integration steps for AllShowers (16 = fast, 200 = accurate).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for AllShowers Stage 2 generation.",
    )
    parser.add_argument(
        "--solver", type=str, default="midpoint", choices=["heun", "midpoint"],
        help="ODE solver for AllShowers.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device (e.g. cuda:0, cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible primary particle sampling.",
    )
    parser.add_argument(
        "--e-min", type=float, default=E_MIN,
        help="Minimum primary energy [GeV].",
    )
    parser.add_argument(
        "--e-max", type=float, default=E_MAX,
        help="Maximum primary energy [GeV].",
    )
    parser.add_argument(
        "--zenith-min", type=float, default=ZENITH_MIN,
        help="Minimum zenith angle [degrees].",
    )
    parser.add_argument(
        "--zenith-max", type=float, default=ZENITH_MAX,
        help="Maximum zenith angle [degrees].",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = get_args(args)
    t_start = time.perf_counter()

    # --- Device selection ---------------------------------------------------
    if parsed.device:
        device = parsed.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("=" * 60)
    print("TAMBO Two-Stage Shower Generator")
    print("=" * 60)
    print(f"  Device          : {device}")
    print(f"  Num samples     : {parsed.num_samples}")
    print(f"  Energy range    : {parsed.e_min:.1e} – {parsed.e_max:.1e} GeV")
    print(f"  Zenith range    : {parsed.zenith_min}° – {parsed.zenith_max}°")
    print(f"  Particle labels : {list(range(NUM_CLASSES))}")
    print(f"  ODE timesteps   : {parsed.num_timesteps}")
    print(f"  Solver          : {parsed.solver}")
    print(f"  Seed            : {parsed.seed}")
    print(f"  Output          : {os.path.abspath(parsed.output)}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Sample primary particle parameters
    # -----------------------------------------------------------------------
    print("[1/3] Sampling primary particle parameters ...")
    primary = sample_primary_particles(
        n=parsed.num_samples,
        e_min=parsed.e_min,
        e_max=parsed.e_max,
        zenith_min=parsed.zenith_min,
        zenith_max=parsed.zenith_max,
        seed=parsed.seed,
    )
    print(f"  Energy:    min={primary['energies'].min():.2e}  "
          f"max={primary['energies'].max():.2e}  "
          f"mean={primary['energies'].mean():.2e} GeV")
    print(f"  Labels:    {dict(zip(*np.unique(primary['labels'].numpy(), return_counts=True)))}")

    # -----------------------------------------------------------------------
    # Step 2: PointCountFM → num_points_per_layer
    # -----------------------------------------------------------------------
    print("\n[2/3] Stage 1 – PointCountFM ...")
    num_points = run_point_count_fm(
        model_path=parsed.point_count_model,
        energies=primary["energies"],
        directions=primary["directions"],
        labels=primary["labels"],
    )

    # -----------------------------------------------------------------------
    # Step 3: AllShowers → shower point clouds
    # -----------------------------------------------------------------------
    print("\n[3/3] Stage 2 – AllShowers CNF-Transformer ...")
    samples = run_allshowers(
        run_dir=parsed.allshowers_run_dir,
        energies=primary["energies"],
        directions=primary["directions"],
        labels=primary["labels"],
        num_points=num_points,
        num_timesteps=parsed.num_timesteps,
        batch_size=parsed.batch_size,
        solver=parsed.solver,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print("\nSaving output ...")
    save_output(
        path=parsed.output,
        samples=samples,
        energies=primary["energies"],
        directions=primary["directions"],
        labels=primary["labels"],
    )

    print()
    print(f"Total time: {time.perf_counter() - t_start:.1f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
