#!/usr/bin/env python3
"""
Inference script for the reconstruction flow-matching model.

Loads a compiled (TorchScript) reconstruction model and a combined HDF5
dataset, runs inference on a pool of showers, and produces comparison plots
of predicted vs ground-truth directions, PDG labels, and energies.

Edit the CONFIG block below, then run:  python inference_reconstruction.py
"""

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    # Combined reconstruction HDF5 file
    "input_file": (
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif"
        "/tambo_simulations_for_training/reconstruction_combined.h5"
    ),

    # Models to compare: list of {path, label} dicts.
    "models": [
        {
            "path": "/path/to/results/compiled.pt",
            "label": "FullyConnected",
        },
    ],

    # How many showers to sample for plots (from the end of the file).
    "num_samples": 10_000,

    # Condition feature keys (must match training config).
    "condition_features": [
        "energy_per_layer_electron",
        "num_points_per_layer_electron",
        "time_per_layer_electron",
    ],

    # PDG label list: maps label index -> display name.
    "pdg_names": {0: "electron", 1: "muon"},

    # Where to write output plots.
    "plot_dir": "/n/home04/hhanif/AllShowers/plots",

    # Random seed for reproducibility.
    "seed": 42,
}
# ─────────────────────────────────────────────────────────────────────────────

import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ground_truth(
    input_file: str,
    condition_features: list[str],
    start: int,
    end: int | None,
) -> dict[str, np.ndarray]:
    """Load ground-truth targets and conditions from the reconstruction HDF5."""
    with h5py.File(input_file, "r") as f:
        directions = f["directions"][start:end].astype(np.float32)
        labels = f["labels"][start:end].astype(np.int32)
        energies = f["energies"][start:end].astype(np.float32)

        cond_parts = []
        for feat in condition_features:
            cond_parts.append(f[feat][start:end].astype(np.float32))

    condition = np.concatenate(cond_parts, axis=-1)
    return {
        "directions": directions,
        "labels": labels,
        "energies": energies.squeeze(),
        "condition": condition,
    }


# ── Model helpers ────────────────────────────────────────────────────────────

def load_model(path: str) -> torch.jit.ScriptModule:
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def run_inference(
    model: torch.jit.ScriptModule,
    condition: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    """Run batched inference and return (N, 5) numpy array."""
    results = []
    for i in range(0, condition.shape[0], batch_size):
        batch = condition[i : i + batch_size]
        out = model(batch)
        results.append(out.cpu())
    return torch.cat(results, dim=0).numpy()


# ── Plot helpers ─────────────────────────────────────────────────────────────

_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]


def plot_direction_components(
    gt_dirs: np.ndarray,
    pred_dirs_list: list[np.ndarray],
    labels: list[str],
    outdir: Path,
) -> None:
    """Histogram comparison of direction components (dx, dy, dz)."""
    comp_names = ["dx", "dy", "dz"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, (ax, name) in enumerate(zip(axes, comp_names)):
        ax.hist(gt_dirs[:, j], bins=80, density=True, alpha=0.5,
                color="black", label="Ground truth")
        for m_idx, (pred, label) in enumerate(zip(pred_dirs_list, labels)):
            ax.hist(pred[:, j], bins=80, density=True, alpha=0.4,
                    color=_COLORS[m_idx % len(_COLORS)],
                    label=f"ML – {label}", histtype="step", linewidth=2)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Direction component: {name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "directions.png", dpi=200)
    plt.close(fig)


def plot_angular_error(
    gt_dirs: np.ndarray,
    pred_dirs_list: list[np.ndarray],
    labels: list[str],
    outdir: Path,
) -> None:
    """Histogram of angular error (degrees) between predicted and GT directions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m_idx, (pred, label) in enumerate(zip(pred_dirs_list, labels)):
        # Normalize both
        gt_norm = gt_dirs / (np.linalg.norm(gt_dirs, axis=1, keepdims=True) + 1e-12)
        pr_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
        cos_angle = np.clip(np.sum(gt_norm * pr_norm, axis=1), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        median = np.median(angle_deg)
        ax.hist(angle_deg, bins=100, density=True, alpha=0.5,
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label} (median={median:.2f}°)")
    ax.set_xlabel("Angular error (degrees)")
    ax.set_ylabel("Density")
    ax.set_title("Angular error between predicted and GT directions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "angular_error.png", dpi=200)
    plt.close(fig)


def plot_pdg_classification(
    gt_labels: np.ndarray,
    pred_pdg_list: list[np.ndarray],
    model_labels: list[str],
    pdg_names: dict[int, str],
    outdir: Path,
) -> None:
    """Bar chart of PDG classification accuracy per class."""
    unique_labels = np.unique(gt_labels)
    n_classes = len(unique_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.8 / max(len(pred_pdg_list), 1)

    for m_idx, (pred, label) in enumerate(zip(pred_pdg_list, model_labels)):
        pred_int = np.floor(pred).astype(np.int32).squeeze()
        pred_int = np.clip(pred_int, unique_labels.min(), unique_labels.max())
        accs = []
        for cls in unique_labels:
            mask = gt_labels == cls
            acc = np.mean(pred_int[mask] == cls) if mask.sum() > 0 else 0.0
            accs.append(acc)
        x = np.arange(n_classes) + m_idx * bar_width
        bars = ax.bar(x, accs, bar_width, label=f"ML – {label}",
                       color=_COLORS[m_idx % len(_COLORS)], alpha=0.7)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.1%}", ha="center", va="bottom", fontsize=8)

    overall_accs = []
    for pred in pred_pdg_list:
        pred_int = np.floor(pred).astype(np.int32).squeeze()
        pred_int = np.clip(pred_int, unique_labels.min(), unique_labels.max())
        overall_accs.append(np.mean(pred_int == gt_labels))

    tick_labels = [pdg_names.get(int(c), str(c)) for c in unique_labels]
    ax.set_xticks(np.arange(n_classes) + bar_width * (len(pred_pdg_list) - 1) / 2)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Accuracy")
    title = "PDG classification accuracy"
    if len(overall_accs) == 1:
        title += f"  (overall: {overall_accs[0]:.1%})"
    ax.set_title(title)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "pdg_accuracy.png", dpi=200)
    plt.close(fig)


def plot_pdg_distribution(
    gt_labels: np.ndarray,
    pred_pdg_list: list[np.ndarray],
    model_labels: list[str],
    pdg_names: dict[int, str],
    outdir: Path,
) -> None:
    """Histogram of raw predicted PDG values (continuous) vs GT."""
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = np.unique(gt_labels)
    ax.hist(gt_labels, bins=np.arange(unique_labels.min() - 0.5,
            unique_labels.max() + 1.5, 1), density=True, alpha=0.5,
            color="black", label="Ground truth")
    for m_idx, (pred, label) in enumerate(zip(pred_pdg_list, model_labels)):
        ax.hist(pred.squeeze(), bins=80, density=True, alpha=0.4,
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label} (continuous)", histtype="step", linewidth=2)
    ax.set_xlabel("PDG label value")
    ax.set_ylabel("Density")
    ax.set_title("PDG label distribution (GT discrete vs ML continuous)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "pdg_distribution.png", dpi=200)
    plt.close(fig)


def plot_energy(
    gt_energy: np.ndarray,
    pred_energy_list: list[np.ndarray],
    model_labels: list[str],
    outdir: Path,
) -> None:
    """Energy comparison: histogram + scatter + residual."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Histogram
    ax = axes[0]
    ax.hist(gt_energy, bins=80, density=True, alpha=0.5,
            color="black", label="Ground truth")
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        ax.hist(pred.squeeze(), bins=80, density=True, alpha=0.4,
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label}", histtype="step", linewidth=2)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title("Energy distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scatter: predicted vs true
    ax = axes[1]
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        ax.scatter(gt_energy, pred.squeeze(), s=1, alpha=0.2,
                   color=_COLORS[m_idx % len(_COLORS)], label=f"ML – {label}")
    emin = min(gt_energy.min(), min(p.min() for p in pred_energy_list))
    emax = max(gt_energy.max(), max(p.max() for p in pred_energy_list))
    ax.plot([emin, emax], [emin, emax], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("True energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Energy: predicted vs true")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Relative residual
    ax = axes[2]
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        residual = (pred.squeeze() - gt_energy) / (gt_energy + 1e-12)
        median_res = np.median(np.abs(residual))
        ax.hist(residual, bins=100, density=True, alpha=0.5,
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label} (med|res|={median_res:.3f})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("(pred - true) / true")
    ax.set_ylabel("Density")
    ax.set_title("Energy relative residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "energy.png", dpi=200)
    plt.close(fig)


def plot_energy_per_class(
    gt_energy: np.ndarray,
    gt_labels: np.ndarray,
    pred_energy_list: list[np.ndarray],
    model_labels: list[str],
    pdg_names: dict[int, str],
    outdir: Path,
) -> None:
    """Energy scatter per PDG class."""
    unique_labels = np.unique(gt_labels)
    n_classes = len(unique_labels)
    n_models = len(pred_energy_list)

    fig, axes = plt.subplots(n_models, n_classes,
                             figsize=(7 * n_classes, 5 * n_models), squeeze=False)
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        for c_idx, cls in enumerate(unique_labels):
            ax = axes[m_idx, c_idx]
            mask = gt_labels == cls
            cls_name = pdg_names.get(int(cls), str(cls))
            ax.scatter(gt_energy[mask], pred[mask].squeeze(), s=1, alpha=0.3,
                       color=_COLORS[m_idx % len(_COLORS)])
            emin = gt_energy[mask].min()
            emax = gt_energy[mask].max()
            ax.plot([emin, emax], [emin, emax], "k--", linewidth=1)
            ax.set_xlabel("True energy")
            ax.set_ylabel("Predicted energy")
            ax.set_title(f"{label} – {cls_name} (N={mask.sum()})")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "energy_per_class.png", dpi=200)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_file = CONFIG["input_file"]
    model_specs = [(m["path"], m["label"]) for m in CONFIG["models"]]
    condition_features = CONFIG["condition_features"]
    pdg_names = CONFIG["pdg_names"]
    num_samples = CONFIG["num_samples"]
    plot_dir = CONFIG["plot_dir"]
    seed = CONFIG["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    print(f"Models to compare ({len(model_specs)}):")
    for path, label in model_specs:
        print(f"  [{label}]  {path}")

    # ── Load ground truth ────────────────────────────────────────────────────
    print(f"\nLoading ground truth (last {num_samples} samples) ...")
    t0 = time.time()
    gt = load_ground_truth(input_file, condition_features,
                           start=-num_samples, end=None)
    print(f"  Loaded {gt['directions'].shape[0]} samples in "
          f"{(time.time() - t0) * 1000:.0f} ms")
    print(f"  Condition shape: {gt['condition'].shape}")
    print(f"  Label distribution: "
          f"{dict(zip(*np.unique(gt['labels'], return_counts=True)))}")

    condition_t = torch.from_numpy(gt["condition"]).to(torch.float32)

    # ── Run inference for each model ─────────────────────────────────────────
    all_preds: list[np.ndarray] = []
    all_labels: list[str] = []

    for path, label in model_specs:
        print(f"\nLoading model [{label}] ...")
        t0 = time.time()
        model = load_model(path)
        print(f"  Loaded in {(time.time() - t0) * 1000:.0f} ms")

        print(f"  Running inference ...")
        t0 = time.time()
        pred = run_inference(model, condition_t)
        dt = time.time() - t0
        n = pred.shape[0]
        print(f"  Done: {n} samples in {dt:.2f} s "
              f"({dt / n * 1000:.3f} ms/sample)")
        print(f"  Output shape: {pred.shape}")

        all_preds.append(pred)
        all_labels.append(label)

    # ── Extract components ───────────────────────────────────────────────────
    gt_dirs = gt["directions"]
    gt_labels_arr = gt["labels"]
    gt_energy = gt["energies"]

    pred_dirs_list = [p[:, :3] for p in all_preds]
    pred_pdg_list = [p[:, 3:4] for p in all_preds]
    pred_energy_list = [p[:, 4:] for p in all_preds]

    # ── Create output directory ──────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(plot_dir) / f"reconstruction_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    plot_direction_components(gt_dirs, pred_dirs_list, all_labels, outdir)
    print("  Saved: directions.png")

    plot_angular_error(gt_dirs, pred_dirs_list, all_labels, outdir)
    print("  Saved: angular_error.png")

    plot_pdg_classification(gt_labels_arr, pred_pdg_list, all_labels,
                            pdg_names, outdir)
    print("  Saved: pdg_accuracy.png")

    plot_pdg_distribution(gt_labels_arr, pred_pdg_list, all_labels,
                          pdg_names, outdir)
    print("  Saved: pdg_distribution.png")

    plot_energy(gt_energy, pred_energy_list, all_labels, outdir)
    print("  Saved: energy.png")

    plot_energy_per_class(gt_energy, gt_labels_arr, pred_energy_list,
                          all_labels, pdg_names, outdir)
    print("  Saved: energy_per_class.png")

    print(f"\nDone. All plots saved to: {outdir}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
