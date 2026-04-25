#!/usr/bin/env python3
"""
generate_showers2.py
--------------------
Two-stage TAMBO shower generation pipeline.

Stage 1 – PointCountFM  (compiled TorchScript file  -OR-  run directory)
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

--point-count-model accepts either:
    (a) a compiled TorchScript .pt file  (legacy behaviour), or
    (b) a run directory that contains:
            conf.yaml
            weights/best.pt          ← training snapshot with "model" key
            preprocessing/trafos.pt  ← dict {"num_points": Sequence, "inc": Sequence}

Checkpoint locations (defaults point to the local checkpoints/ directory):
    --point-count-model   path to compiled .pt file  OR  run directory
    --allshowers-run-dir  path to AllShowers run dir (conf.yaml, weights/, preprocessing/)

Usage example
-------------
    python generate_showers2.py \\
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
# Stage 1 – shared helpers
# ---------------------------------------------------------------------------

def _ensure_pkg_root_on_path() -> None:
    """Add /n/home04/hhanif/TAMBO-opt (parent of allshowers/) to sys.path."""
    pkg_root = os.path.dirname(_SCRIPT_DIR)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _build_point_count_conditions(
    energies: torch.Tensor,
    directions: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Build the (N, 6) condition tensor: [energy | one_hot(label, 2) | direction]."""
    one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).to(torch.float32)
    return torch.cat(
        [energies.to(torch.float32), one_hot_labels, directions.to(torch.float32)],
        dim=1,
    ).to(device)  # (N, 6)


def _postprocess_raw_counts(raw: torch.Tensor) -> torch.Tensor:
    """Clamp and round raw (inverse-transformed) output to non-negative integers."""
    return (torch.clamp(raw, min=0.0) + 0.5).to(torch.int32)  # (N, num_layers)


# ---------------------------------------------------------------------------
# Stage 1a – compiled TorchScript file (legacy)
# ---------------------------------------------------------------------------

def _run_point_count_compiled(
    model_path: str,
    conditions: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Run PointCountFM from a compiled TorchScript .pt file."""
    print(f"  Loading compiled PointCountFM from {model_path} ...")
    t0 = time.perf_counter()
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print(f"  Loaded in {(time.perf_counter() - t0)*1e3:.1f} ms")

    inference = torch.jit.script(model.to(torch.float32))

    print(f"  Running inference on {conditions.shape[0]} samples ...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        raw = inference(conditions)
    print(f"  Inference done in {time.perf_counter() - t0:.2f} s")
    return raw


# ---------------------------------------------------------------------------
# Stage 1b – run directory (conf.yaml + weights/best.pt + preprocessing/trafos.pt)
# ---------------------------------------------------------------------------

def _build_cnf_from_config(cfg: dict) -> "torch.nn.Module":
    """Instantiate CNF(FullyConnected(...)) from a parsed conf.yaml dict.

    The checkpoint keys look like:
        frequencies          ← nn.Buffer on CNF
        network.network.0.*  ← CNF.network = FullyConnected, FullyConnected.network = Sequential

    So the correct hierarchy is:  CNF  →  self.network = FullyConnected(...)
    """
    _ensure_pkg_root_on_path()

    from pointcountfm.models import FullyConnected       # type: ignore
    from pointcountfm.flow_matching import CNF           # type: ignore

    model_cfg = cfg["model"]
    name = model_cfg["name"]

    if name == "FullyConnected":
        network = FullyConnected(
            dim_input=model_cfg["dim_input"],
            dim_condition=model_cfg["dim_condition"],
            dim_time=model_cfg["dim_time"],
            hidden_dims=model_cfg["hidden_dims"],
        )
        # dim_time = 2 * frequencies  (cos + sin for each frequency)
        frequencies = model_cfg["dim_time"] // 2
        return CNF(network=network, frequencies=frequencies)

    raise ValueError(
        f"Unknown PointCountFM model name '{name}'. "
        "Extend _build_cnf_from_config() to handle it."
    )


def _run_point_count_from_run_dir(
    run_dir: str,
    conditions: torch.Tensor,
    num_timesteps: int,
    device: str,
) -> torch.Tensor:
    """Run PointCountFM by loading conf.yaml + weights/best.pt + preprocessing/trafos.pt.

    Parameters
    ----------
    run_dir       : Path to the PointCountFM result directory.
    conditions    : (N, 6) condition tensor already on ``device``.
    num_timesteps : ODE integration steps for the CNF decoder.
    device        : Torch device string.

    Returns
    -------
    raw_inv : (N, num_layers) float32 — counts in original (count) space,
              before rounding to integers.
    """
    import yaml  # PyYAML

    _ensure_pkg_root_on_path()

    conf_path    = os.path.join(run_dir, "conf.yaml")
    weights_path = os.path.join(run_dir, "weights", "best.pt")
    trafos_path  = os.path.join(run_dir, "preprocessing", "trafos.pt")

    for p in (conf_path, weights_path, trafos_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected file not found in PointCountFM run directory:\n  {p}"
            )

    # ── Config ──────────────────────────────────────────────────────────────
    print(f"  Loading conf      : {conf_path}")
    with open(conf_path) as fh:
        cfg = yaml.safe_load(fh)

    num_layers = cfg["model"]["dim_input"]   # e.g. 24
    print(f"  num_layers (dim_input) : {num_layers}")
    print(f"  num_classes (config)   : {cfg['data'].get('num_classes', NUM_CLASSES)}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"  Building CNF(FullyConnected) ...")
    model = _build_cnf_from_config(cfg)

    print(f"  Loading weights   : {weights_path}")
    t0 = time.perf_counter()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    # Training snapshot format: {"model": state_dict, "optimizer": ..., ...}
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print(f"  Weights loaded in {(time.perf_counter() - t0)*1e3:.1f} ms")

    # ── Preprocessing transforms ─────────────────────────────────────────────
    print(f"  Loading trafos    : {trafos_path}")
    trafos = torch.load(trafos_path, map_location="cpu", weights_only=False)
    # trafos is expected to be a dict: {"num_points": Sequence(...), "inc": Sequence(...)}
    # where Sequence comes from pointcountfm.preprocessing and has an .inverse() method.
    if isinstance(trafos, dict):
        num_points_trafo = trafos["transform_num_points"]
    else:
        # Fallback: trafos is the Sequence directly
        num_points_trafo = trafos
    print(f"  Trafo type        : {type(num_points_trafo).__name__}")

    # ── Inference via CNF.sample (ODE decode from noise → counts) ────────────
    n = conditions.shape[0]
    print(f"  Sampling {n} showers via CNF ({num_timesteps} ODE steps) ...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        raw = model.sample(
            shape=(n, num_layers),
            steps=num_timesteps,
            condition=conditions,
        )                                    # (N, num_layers) in normalised space
    print(f"  Inference done in {time.perf_counter() - t0:.2f} s")

    # ── Inverse transform: normalised → original count space ─────────────────
    raw_inv = num_points_trafo.inverse(raw.cpu())   # Sequence.inverse() handles ordering

    return raw_inv.to(device)


# ---------------------------------------------------------------------------
# Stage 1 – public entry point
# ---------------------------------------------------------------------------

def run_point_count_fm(
    model_path: str,
    energies: torch.Tensor,    # (N, 1) float32
    directions: torch.Tensor,  # (N, 3) float32
    labels: torch.Tensor,      # (N,)   int64
    num_timesteps: int = 16,
    device: str = "cpu",
) -> torch.Tensor:
    """Run the PointCountFM model to predict num_points per detector layer.

    ``model_path`` can be either:
        • a compiled TorchScript ``.pt`` file  (legacy), or
        • a run directory containing ``conf.yaml``, ``weights/best.pt``,
          and ``preprocessing/trafos.pt``.

    The condition tensor fed to the model is:
        [energy (1) | one_hot(label, 2) | direction (3)]  →  shape (N, 6)

    Returns
    -------
    num_points : (N, num_layers) int32  — non-negative integer counts per layer
    """
    conditions = _build_point_count_conditions(energies, directions, labels, device)

    if os.path.isdir(model_path):
        print(f"  PointCountFM mode  : run directory")
        raw = _run_point_count_from_run_dir(model_path, conditions, num_timesteps, device)
    else:
        print(f"  PointCountFM mode  : compiled TorchScript")
        raw = _run_point_count_compiled(model_path, conditions, device)

    num_points = _postprocess_raw_counts(raw)
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
    samples : (N, max_points, 4 or 5) float32
              Columns: x, y, z(layer_index), e[, t]
    """
    _ensure_pkg_root_on_path()

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
        angles=directions,
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
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--point-count-model", type=str, default=_DEFAULT_POINT_COUNT_MODEL,
        help=(
            "PointCountFM checkpoint. Accepts either:\n"
            "  (a) a compiled TorchScript .pt file  (legacy), or\n"
            "  (b) a run directory containing conf.yaml, weights/best.pt,\n"
            "      and preprocessing/trafos.pt."
        ),
    )
    parser.add_argument(
        "--allshowers-run-dir", type=str, default=_DEFAULT_ALLSHOWERS_RUN_DIR,
        help="Path to the AllShowers run directory (conf.yaml, weights/best.pt, preprocessing/trafos.pt).",
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=16,
        help="ODE integration steps for both PointCountFM and AllShowers (16 = fast, 200 = accurate).",
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
        num_timesteps=parsed.num_timesteps,
        device=device,
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