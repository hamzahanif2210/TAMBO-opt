#!/usr/bin/env python3
"""
optimize_detector_layout.py
---------------------------
Detector layout optimization using the TAMBO ML surrogate pipeline.

Pipeline per optimization step:
  1. Sample primary particles (energy, direction, label).
  2. PointCountFM  → predict num_points per detector layer.
  3. AllShowers    → generate shower point clouds (x, y, layer, energy, time).
  4. Bridge        → extract per-layer condition features from the point clouds
                     (energy_per_layer, num_points_per_layer, time_per_layer).
  5. Reconstruction compiled model → predict (direction, pdg, energy).
  6. Compute utility from reconstruction quality → backprop through detector
     positions → SGD update.

The detector positions (x, y) are learnable parameters.  Every few epochs the
reconstruction model can be fine-tuned on freshly generated data so it stays
accurate for the current layout.

Usage
-----
    # From HDF5 geometry (TAMBOSim native format):
    python optimize_detector_layout.py \
        --point-count-model  allshowers/checkpoints/num_of_point_clouds_dequantize_compiled.pt \
        --allshowers-run-dir allshowers/checkpoints/all_showers \
        --reconstruction-model results/reconstruction/compiled.pt \
        --geometry resources/basic_geometry.h5:colca_valley_30000 \
        --detector-key detector1 \
        --num-epochs 100 \
        --device cuda:0

    # From plain text layout file:
    python optimize_detector_layout.py \
        --point-count-model  allshowers/checkpoints/num_of_point_clouds_dequantize_compiled.pt \
        --allshowers-run-dir allshowers/checkpoints/all_showers \
        --reconstruction-model results/reconstruction/compiled.pt \
        --layout initial_layout.txt \
        --num-epochs 100 \
        --device cuda:0
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from allshowers.generate_showers import (
    build_direction_vector,
    run_allshowers,
    run_point_count_fm,
    sample_primary_particles,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_POINT_COUNT_MODEL = os.path.join(
    _SCRIPT_DIR, "allshowers", "checkpoints",
    "num_of_point_clouds_dequantize_compiled.pt",
)
_DEFAULT_ALLSHOWERS_RUN_DIR = os.path.join(
    _SCRIPT_DIR, "allshowers", "checkpoints", "all_showers",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 24
TANK_RADIUS = 50.0  # metres – minimum half-distance between detectors

# SWGO site triangle vertices
SITE_A = torch.tensor([-3800.0, 1500.0])
SITE_B = torch.tensor([1200.0, 1500.0])
SITE_C = torch.tensor([1200.0, -4100.0])


# ═══════════════════════════════════════════════════════════════════════════
# Load detector layout from HDF5 geometry file
# ═══════════════════════════════════════════════════════════════════════════

def load_layout_from_h5(geometry: str, detector_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load detector (x, y) positions in local ENU metres from an HDF5 geometry file.

    Parameters
    ----------
    geometry : str
        Path and group in the form ``"path/to/basic_geometry.h5:group_name"``
        (e.g. ``"resources/basic_geometry.h5:colca_valley_30000"``).
    detector_key : str
        Dataset name within the group that holds triangle indices for the
        detector region (e.g. ``"detector1"``).

    Returns
    -------
    x_enu, y_enu : (N,) torch.Tensor
        East and North coordinates in metres relative to the site origin.
    """
    if ":" not in geometry:
        raise ValueError("--geometry must be 'path/to/file.h5:group_name'")
    h5_path, group_name = geometry.split(":", 1)

    with h5py.File(h5_path, "r") as f:
        grp = f[group_name]
        # Location is stored as [longitude, latitude] in degrees
        lon_deg, lat_deg = grp["location"][:]
        # Vertices: shape (3, N_verts) — ECEF metres
        verts = grp["vertices"][:]          # (3, N_verts)
        # Faces: shape (3, N_tri) — 1-based vertex indices (Julia convention)
        faces = grp["faces"][:]             # (3, N_tri)
        # Detector region: 1-based triangle indices
        det_tri_idx = grp[detector_key][:] # (N_det,)

    # Convert 1-based Julia indices to 0-based
    tri_idx = det_tri_idx - 1                          # (N_det,)
    face_verts = faces[:, tri_idx] - 1                 # (3, N_det), 0-based vertex ids

    # Triangle centroids in ECEF
    v0 = verts[:, face_verts[0]]   # (3, N_det)
    v1 = verts[:, face_verts[1]]
    v2 = verts[:, face_verts[2]]
    centroids_ecef = (v0 + v1 + v2) / 3.0             # (3, N_det)

    # ECEF → local ENU using site location as the reference point
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # Reference ECEF point: project origin onto the sphere at the same radius
    # Use the mean radius of the detector centroids as R
    R = float(np.linalg.norm(centroids_ecef, axis=0).mean())
    X0 = R * cos_lat * cos_lon
    Y0 = R * cos_lat * sin_lon
    Z0 = R * sin_lat

    dx = centroids_ecef[0] - X0
    dy = centroids_ecef[1] - Y0
    dz = centroids_ecef[2] - Z0

    # ENU rotation
    east  = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

    x_enu = torch.tensor(east,  dtype=torch.float32)
    y_enu = torch.tensor(north, dtype=torch.float32)
    return x_enu, y_enu


# ═══════════════════════════════════════════════════════════════════════════
# Learnable detector positions
# ═══════════════════════════════════════════════════════════════════════════

class LearnableXY(nn.Module):
    """Wraps (x, y) detector coordinates as learnable parameters."""

    def __init__(self, x_init: torch.Tensor, y_init: torch.Tensor) -> None:
        super().__init__()
        self.x = nn.Parameter(x_init.clone())
        self.y = nn.Parameter(y_init.clone())

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y


# ═══════════════════════════════════════════════════════════════════════════
# Geometric constraints
# ═══════════════════════════════════════════════════════════════════════════

def barycentric_coords(
    P: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Barycentric coordinates of points P w.r.t. triangle ABC."""
    v0 = C - A
    v1 = B - A
    v2 = P - A
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return u, v


def project_to_triangle(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project detector positions inside the SWGO site triangle."""
    A = SITE_A.to(x.device)
    B = SITE_B.to(x.device)
    C = SITE_C.to(x.device)
    P = torch.stack([x, y], dim=1)
    u, v = barycentric_coords(P, A, B, C)
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    u_c = torch.clamp(u, 0.0, 1.0)
    v_c = torch.clamp(v, 0.0, 1.0)
    over = u_c + v_c > 1.0
    u_c[over] = u_c[over] / (u_c[over] + v_c[over])
    v_c[over] = v_c[over] / (u_c[over] + v_c[over])
    v0 = C - A
    v1 = B - A
    P_proj = A + u_c.unsqueeze(1) * v0 + v_c.unsqueeze(1) * v1
    final = torch.where(inside.unsqueeze(1), P, P_proj)
    return final[:, 0], final[:, 1]


def push_apart(module: LearnableXY, min_dist: float = 2 * TANK_RADIUS) -> None:
    """Enforce minimum distance between all detector pairs (in-place)."""
    x, y = module()
    coords = torch.stack([x, y], dim=1)
    with torch.no_grad():
        for i in range(coords.shape[0]):
            diffs = coords[i] - coords
            dists = torch.norm(diffs, dim=1)
            mask = (dists < min_dist) & (dists > 0)
            for j in torch.where(mask)[0]:
                direction = diffs[j] / dists[j]
                displacement = 0.5 * (min_dist - dists[j]) * direction
                coords[i] += displacement
                coords[j] -= displacement
        module.x.data.copy_(coords[:, 0])
        module.y.data.copy_(coords[:, 1])


# ═══════════════════════════════════════════════════════════════════════════
# Bridge: shower point clouds → per-layer reconstruction conditions
# ═══════════════════════════════════════════════════════════════════════════

def showers_to_condition(
    samples: torch.Tensor,
    num_layers: int = NUM_LAYERS,
) -> torch.Tensor:
    """Convert AllShowers output to reconstruction condition features.

    Parameters
    ----------
    samples : (N, max_points, 5)  — columns: x, y, z(layer_idx), energy, time

    Returns
    -------
    condition : (N, 3 * num_layers) — [energy_per_layer | num_points_per_layer | time_per_layer]
                Each block is (N, num_layers).
    """
    N, max_pts, _ = samples.shape
    device = samples.device

    energy_per_layer = torch.zeros(N, num_layers, device=device)
    num_points_per_layer = torch.zeros(N, num_layers, device=device)
    time_sum_per_layer = torch.zeros(N, num_layers, device=device)
    count_per_layer = torch.zeros(N, num_layers, device=device)

    layer_idx = samples[:, :, 2].long().clamp(0, num_layers - 1)  # (N, max_pts)
    hit_energy = samples[:, :, 3]                                  # (N, max_pts)
    hit_time = samples[:, :, 4]                                    # (N, max_pts)
    active = hit_energy > 0                                        # (N, max_pts)

    # Batch-index for scatter_add
    batch_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(layer_idx)

    # Flat indices for (batch, layer)
    flat_idx = batch_idx * num_layers + layer_idx  # (N, max_pts)

    # Energy per layer
    energy_flat = torch.zeros(N * num_layers, device=device)
    energy_flat.scatter_add_(0, flat_idx.reshape(-1), (hit_energy * active.float()).reshape(-1))
    energy_per_layer = energy_flat.reshape(N, num_layers)

    # Num points per layer
    count_flat = torch.zeros(N * num_layers, device=device)
    count_flat.scatter_add_(0, flat_idx.reshape(-1), active.float().reshape(-1))
    num_points_per_layer = count_flat.reshape(N, num_layers)

    # Average time per layer
    time_flat = torch.zeros(N * num_layers, device=device)
    time_flat.scatter_add_(0, flat_idx.reshape(-1), (hit_time * active.float()).reshape(-1))
    time_sum = time_flat.reshape(N, num_layers)
    time_per_layer = torch.where(
        num_points_per_layer > 0,
        time_sum / num_points_per_layer,
        torch.zeros_like(time_sum),
    )

    # Concatenate in the same order the reconstruction model expects:
    # energy_per_layer (24) | num_points_per_layer (24) | time_per_layer (24) = 72
    condition = torch.cat([energy_per_layer, num_points_per_layer, time_per_layer], dim=1)
    return condition


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def angular_error(
    pred_dirs: torch.Tensor,   # (N, 3)
    true_dirs: torch.Tensor,   # (N, 3)
) -> torch.Tensor:
    """Mean angular error in degrees between predicted and true directions."""
    pred_n = F.normalize(pred_dirs, dim=1)
    true_n = F.normalize(true_dirs, dim=1)
    cos_angle = (pred_n * true_n).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / math.pi)  # (N,)


def energy_resolution(
    pred_e: torch.Tensor,   # (N,)
    true_e: torch.Tensor,   # (N,)
) -> torch.Tensor:
    """Absolute relative energy residual: |pred - true| / (true + eps)."""
    return torch.abs(pred_e - true_e) / (true_e.abs() + 1e-12)


def classification_accuracy(
    pred_pdg: torch.Tensor,  # (N,) continuous
    true_labels: torch.Tensor,  # (N,) int
    num_classes: int = 2,
) -> torch.Tensor:
    """Fraction of correctly classified particles (differentiable proxy)."""
    pred_cls = pred_pdg.floor().long().clamp(0, num_classes - 1)
    return (pred_cls == true_labels).float().mean()


def reconstructability_score(num_points_per_layer: torch.Tensor) -> torch.Tensor:
    """Score [0, 1] indicating how reconstructable each event is.

    Events with more active layers and more hits are more reconstructable.
    """
    active_layers = (num_points_per_layer > 0).float().sum(dim=1)
    total_hits = num_points_per_layer.sum(dim=1)
    # Normalize: active_layers / num_layers ∈ [0,1], log(1+hits) / scale
    layer_score = active_layers / num_points_per_layer.shape[1]
    hit_score = torch.log1p(total_hits) / torch.log1p(total_hits.max() + 1e-6)
    return 0.5 * (layer_score + hit_score)


def compute_utility(
    pred_dirs: torch.Tensor,      # (N, 3)
    pred_pdg: torch.Tensor,       # (N,)
    pred_energy: torch.Tensor,    # (N,)
    true_dirs: torch.Tensor,      # (N, 3)
    true_labels: torch.Tensor,    # (N,)
    true_energy: torch.Tensor,    # (N,)
    r_score: torch.Tensor,        # (N,)
) -> torch.Tensor:
    """Compute the total utility (to be maximized).

    U = w_angle * U_angle + w_energy * U_energy + w_recon * U_recon

    Where each utility component rewards better reconstruction, weighted by
    the reconstructability score of each event.
    """
    # Angular utility: negative mean angular error weighted by r_score
    ang_err = angular_error(pred_dirs, true_dirs)            # (N,) degrees
    U_angle = -(ang_err * r_score).mean()

    # Energy utility: negative mean relative residual weighted by r_score
    e_res = energy_resolution(pred_energy, true_energy)      # (N,)
    U_energy = -(e_res * r_score).mean()

    # Reconstructability utility: reward high average reconstructability
    U_recon = r_score.mean()

    # Combined utility (signs chosen so that maximizing U improves layout)
    U = 1e-2 * U_angle + U_energy + U_recon
    return U


# ═══════════════════════════════════════════════════════════════════════════
# Full forward pass: primaries → showers → reconstruction predictions
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    point_count_model_path: str,
    allshowers_run_dir: str,
    reconstruction_model: torch.jit.ScriptModule,
    primary: dict[str, torch.Tensor],
    num_timesteps: int = 16,
    batch_size: int = 128,
    solver: str = "midpoint",
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Run the full TAMBO surrogate pipeline.

    Returns dict with keys: predictions (N,5), condition (N,72),
    samples (N, max_pts, 5).
    """
    energies = primary["energies"]
    directions = primary["directions"]
    labels = primary["labels"]

    # Stage 1: PointCountFM
    num_points = run_point_count_fm(
        model_path=point_count_model_path,
        energies=energies,
        directions=directions,
        labels=labels,
        device=device,
    )

    # Stage 2: AllShowers
    samples = run_allshowers(
        run_dir=allshowers_run_dir,
        energies=energies,
        directions=directions,
        labels=labels,
        num_points=num_points,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        solver=solver,
        device=device,
    )

    # Bridge: extract per-layer features for reconstruction
    condition = showers_to_condition(samples, num_layers=NUM_LAYERS)

    # Stage 3: Reconstruction
    with torch.inference_mode():
        predictions = reconstruction_model(condition.to(torch.float32).cpu())
    # predictions shape: (N, 5) = [dx, dy, dz, pdg, energy]

    return {
        "predictions": predictions,
        "condition": condition,
        "samples": samples,
        "num_points": num_points,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Optimization loop
# ═══════════════════════════════════════════════════════════════════════════

def optimize(
    point_count_model_path: str,
    allshowers_run_dir: str,
    reconstruction_model_path: str,
    layout_path: str | None,
    geometry: str | None,
    detector_key: str,
    num_detectors: int,
    num_epochs: int,
    samples_per_epoch: int,
    num_timesteps: int,
    batch_size: int,
    solver: str,
    lr: float,
    momentum: float,
    max_grad_norm: float,
    finetune_interval: int,
    output_dir: str,
    device: str,
    seed: int | None,
) -> None:
    """Main optimization loop."""
    os.makedirs(output_dir, exist_ok=True)
    layouts_dir = os.path.join(output_dir, "layouts")
    os.makedirs(layouts_dir, exist_ok=True)

    # --- Load reconstruction model ---
    print("Loading reconstruction model ...")
    recon_model = torch.jit.load(reconstruction_model_path, map_location="cpu")
    recon_model.eval()

    # --- Initialize detector positions ---
    if geometry:
        x_init, y_init = load_layout_from_h5(geometry, detector_key)
        num_detectors = len(x_init)
        print(f"Loaded layout from HDF5 '{geometry}' key='{detector_key}' "
              f"({num_detectors} detectors)")
        print(f"  ENU x range: [{x_init.min():.1f}, {x_init.max():.1f}] m")
        print(f"  ENU y range: [{y_init.min():.1f}, {y_init.max():.1f}] m")
    elif layout_path and os.path.exists(layout_path):
        data = np.loadtxt(layout_path)
        x_init = torch.tensor(data[:, 0], dtype=torch.float32)
        y_init = torch.tensor(data[:, 1], dtype=torch.float32)
        num_detectors = len(x_init)
        print(f"Loaded layout from {layout_path} ({num_detectors} detectors)")
    else:
        # Random initialization inside the site triangle
        rng = np.random.default_rng(seed)
        x_init = torch.tensor(
            rng.uniform(SITE_A[0].item(), SITE_B[0].item(), num_detectors),
            dtype=torch.float32,
        )
        y_init = torch.tensor(
            rng.uniform(SITE_C[1].item(), SITE_A[1].item(), num_detectors),
            dtype=torch.float32,
        )
        x_init, y_init = project_to_triangle(x_init, y_init)
        print(f"Initialized {num_detectors} random detector positions")

    xy_module = LearnableXY(x_init, y_init)
    optimizer = torch.optim.SGD(xy_module.parameters(), lr=lr, momentum=momentum)

    # --- Check for checkpoint ---
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    start_epoch = 0
    utility_history = []

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        xy_module.x.data.copy_(ckpt["x"])
        xy_module.y.data.copy_(ckpt["y"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        utility_history = ckpt.get("utility_history", torch.tensor([])).tolist()
        print(f"Resumed from epoch {start_epoch}")

    # --- Optimization loop ---
    print("\n" + "=" * 70)
    print("DETECTOR LAYOUT OPTIMIZATION")
    print("=" * 70)
    print(f"  Detectors       : {num_detectors}")
    print(f"  Epochs          : {start_epoch} → {start_epoch + num_epochs}")
    print(f"  Samples/epoch   : {samples_per_epoch}")
    print(f"  ODE timesteps   : {num_timesteps}")
    print(f"  Solver          : {solver}")
    print(f"  LR              : {lr}")
    print(f"  Device          : {device}")
    print(f"  Output          : {output_dir}")
    print()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        t0 = time.perf_counter()

        # 1. Sample primary particles
        primary = sample_primary_particles(
            n=samples_per_epoch,
            seed=seed + epoch if seed is not None else None,
        )

        # 2-4. Run full pipeline
        result = run_full_pipeline(
            point_count_model_path=point_count_model_path,
            allshowers_run_dir=allshowers_run_dir,
            reconstruction_model=recon_model,
            primary=primary,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            solver=solver,
            device=device,
        )

        predictions = result["predictions"]  # (N, 5)
        condition = result["condition"]       # (N, 72)

        pred_dirs = predictions[:, :3]
        pred_pdg = predictions[:, 3]
        pred_energy = predictions[:, 4]

        true_dirs = primary["directions"]
        true_labels = primary["labels"]
        true_energy = primary["energies"].squeeze(1)

        # Reconstructability from the generated num_points
        num_points_per_layer_cond = condition[:, NUM_LAYERS:2 * NUM_LAYERS]
        r_score = reconstructability_score(num_points_per_layer_cond)

        # 5. Compute utility
        U = compute_utility(
            pred_dirs=pred_dirs,
            pred_pdg=pred_pdg,
            pred_energy=pred_energy,
            true_dirs=true_dirs,
            true_labels=true_labels,
            true_energy=true_energy,
            r_score=r_score,
        )

        # 6. Gradient step (minimize -U)
        loss = -U
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Enforce constraints
        with torch.no_grad():
            push_apart(xy_module)
            x_proj, y_proj = project_to_triangle(
                xy_module.x.data, xy_module.y.data
            )
            xy_module.x.data.copy_(x_proj)
            xy_module.y.data.copy_(y_proj)

        utility_history.append(U.item())
        dt = time.perf_counter() - t0

        # Log
        ang_err = angular_error(pred_dirs, true_dirs).mean().item()
        e_res = energy_resolution(pred_energy, true_energy).mean().item()
        cls_acc = classification_accuracy(pred_pdg, true_labels).item()

        print(
            f"Epoch {epoch:4d} | U={U.item():+8.2f} | "
            f"AngErr={ang_err:6.2f}° | ERes={e_res:.4f} | "
            f"ClsAcc={cls_acc:.2%} | RScore={r_score.mean():.3f} | "
            f"{dt:.1f}s"
        )

        # Save layout
        x_np = xy_module.x.detach().cpu().numpy()
        y_np = xy_module.y.detach().cpu().numpy()
        np.savetxt(
            os.path.join(layouts_dir, f"layout_{epoch:04d}.txt"),
            np.column_stack((x_np, y_np)),
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "x": xy_module.x.data.cpu(),
                "y": xy_module.y.data.cpu(),
                "optimizer_state_dict": optimizer.state_dict(),
                "utility_history": torch.tensor(utility_history),
                "loss": loss.item(),
            },
            checkpoint_path,
        )

        # --- Optional fine-tuning of reconstruction model ---
        if finetune_interval > 0 and (epoch + 1) % finetune_interval == 0:
            print(f"  [Fine-tune] Generating {samples_per_epoch * 5} samples "
                  f"with current layout ...")
            # NOTE: Full fine-tuning requires retraining the reconstruction
            # model. This is a placeholder for the integration point.
            # In production, you would:
            #   1. Generate a large dataset with the current layout
            #   2. Retrain or fine-tune the reconstruction model
            #   3. Re-compile it to TorchScript
            print(f"  [Fine-tune] Skipped (not yet integrated)")

    # Save final layout
    final_path = os.path.join(output_dir, "final_layout.txt")
    np.savetxt(
        final_path,
        np.column_stack((
            xy_module.x.detach().cpu().numpy(),
            xy_module.y.detach().cpu().numpy(),
        )),
    )
    print(f"\nFinal layout saved → {final_path}")

    # Save utility history
    np.savetxt(
        os.path.join(output_dir, "utility_history.txt"),
        np.array(utility_history),
    )

    print(f"Optimization complete. {len(utility_history)} epochs total.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def get_args(args: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize TAMBO detector layout using ML surrogate pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    p.add_argument(
        "--point-count-model", type=str,
        default=_DEFAULT_POINT_COUNT_MODEL,
        help="Path to compiled PointCountFM TorchScript model.",
    )
    p.add_argument(
        "--allshowers-run-dir", type=str,
        default=_DEFAULT_ALLSHOWERS_RUN_DIR,
        help="Path to AllShowers run dir (conf.yaml, weights/, preprocessing/).",
    )
    p.add_argument(
        "--reconstruction-model", type=str, required=True,
        help="Path to compiled reconstruction TorchScript model (compiled.pt).",
    )

    # Layout
    layout_grp = p.add_mutually_exclusive_group()
    layout_grp.add_argument(
        "--geometry", type=str, default=None,
        metavar="H5FILE:GROUP",
        help="Load initial detector positions from an HDF5 geometry file, e.g. "
             "'resources/basic_geometry.h5:colca_valley_30000'. "
             "Triangle centroids for --detector-key are converted to local ENU (m).",
    )
    layout_grp.add_argument(
        "--layout", type=str, default=None,
        help="Path to initial layout text file (x, y columns). "
             "If neither --geometry nor --layout is provided, random positions are used.",
    )
    p.add_argument(
        "--detector-key", type=str, default="detector1",
        help="Dataset name within the HDF5 group that holds detector triangle indices "
             "(used with --geometry).",
    )
    p.add_argument(
        "--num-detectors", type=int, default=100,
        help="Number of detectors (used only if neither --geometry nor --layout is provided).",
    )

    # Optimization
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--samples-per-epoch", type=int, default=200,
                   help="Number of primary particles per optimization step.")
    p.add_argument("--lr", type=float, default=5.0,
                   help="SGD learning rate for detector positions.")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--finetune-interval", type=int, default=0,
                   help="Fine-tune reconstruction every N epochs (0=disabled).")

    # Generation
    p.add_argument("--num-timesteps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--solver", type=str, default="midpoint",
                   choices=["heun", "midpoint"])

    # I/O
    p.add_argument("--output-dir", type=str, default="optimization_output")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = get_args(args)

    device = parsed.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    optimize(
        point_count_model_path=parsed.point_count_model,
        allshowers_run_dir=parsed.allshowers_run_dir,
        reconstruction_model_path=parsed.reconstruction_model,
        layout_path=parsed.layout,
        geometry=parsed.geometry,
        detector_key=parsed.detector_key,
        num_detectors=parsed.num_detectors,
        num_epochs=parsed.num_epochs,
        samples_per_epoch=parsed.samples_per_epoch,
        num_timesteps=parsed.num_timesteps,
        batch_size=parsed.batch_size,
        solver=parsed.solver,
        lr=parsed.lr,
        momentum=parsed.momentum,
        max_grad_norm=parsed.max_grad_norm,
        finetune_interval=parsed.finetune_interval,
        output_dir=parsed.output_dir,
        device=device,
        seed=parsed.seed,
    )


if __name__ == "__main__":
    main()
