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
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/reconstruction_for_test.h5"
    ),

    # Models to compare: list of {path, label} dicts.
    "models": [
        # {
        #     "path": "/n/home04/hhanif/AllShowers/results/20260405_025922_Reconstruction-FM/compiled.pt",
        #     "label": "ML-FullyConnected",
        # },
        {
            "path": "/n/home04/hhanif/AllShowers/results/20260405_051159_Reconstruction-FM/compiled.pt",
            "label": "ML-FullyConnected",
        },

    ],

    # How many showers to sample for plots (from the end of the file).
    "num_samples": 100,

    # Condition feature keys (must match training config).
    "condition_features": [
        "energy_per_layer_electron",
        "num_points_per_layer_electron",
        "time_per_layer_electron",
    ],

    # PDG label list: maps label index -> display name.
    "pdg_names": {0: "Electron and Neutral Pion", 1: "Charged Pions"},

    # Where to write output plots.
    "plot_dir": "/n/home04/hhanif/AllShowers/plots",

    # Random seed for reproducibility.
    "seed": 42,

    # ── Per-sample plot settings ──────────────────────────────────────────
    "num_per_sample_plots": 20,
    "per_sample_grid_cols": 5,
}
# ─────────────────────────────────────────────────────────────────────────────

import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


# ── Shared utilities ─────────────────────────────────────────────────────────

_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

_COMP_NAMES = ["dx", "dy", "dz"]


def _angular_error_deg(gt: np.ndarray, pred: np.ndarray) -> float:
    """Scalar angular error in degrees between two (3,) unit vectors."""
    gt_n = gt / (np.linalg.norm(gt) + 1e-12)
    pr_n = pred / (np.linalg.norm(pred) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(gt_n, pr_n), -1.0, 1.0))))


def _energy_residual(gt_e: float, pred_e: float) -> float:
    return (pred_e - gt_e) / (gt_e + 1e-12)


# ── Aggregate plot helpers ───────────────────────────────────────────────────

def plot_direction_components(
    gt_dirs: np.ndarray,
    pred_dirs_list: list[np.ndarray],
    labels: list[str],
    outdir: Path,
) -> None:
    """Histogram comparison of direction components (dx, dy, dz)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, (ax, name) in enumerate(zip(axes, _COMP_NAMES)):
        ax.hist(
            gt_dirs[:, j], bins=80, density=True, alpha=0.5,
            color="black", label="Ground truth"
        )
        for m_idx, (pred, label) in enumerate(zip(pred_dirs_list, labels)):
            ax.hist(
                pred[:, j], bins=80, density=True, alpha=0.4,
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label}", histtype="step", linewidth=2
            )
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Direction component: {name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "directions.png", dpi=200)
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
        bars = ax.bar(
            x, accs, bar_width, label=f"ML – {label}",
            color=_COLORS[m_idx % len(_COLORS)], alpha=0.7
        )
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=8
            )

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
    ax.hist(
        gt_labels,
        bins=np.arange(unique_labels.min() - 0.5, unique_labels.max() + 1.5, 1),
        density=True,
        alpha=0.5,
        color="black",
        label="Ground truth",
    )
    for m_idx, (pred, label) in enumerate(zip(pred_pdg_list, model_labels)):
        ax.hist(
            pred.squeeze(), bins=80, density=True, alpha=0.4,
            color=_COLORS[m_idx % len(_COLORS)],
            label=f"ML – {label} (continuous)", histtype="step", linewidth=2
        )
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

    ax = axes[0]
    ax.hist(gt_energy, bins=80, density=True, alpha=0.5, color="black", label="Ground truth")
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        ax.hist(
            pred.squeeze(), bins=80, density=True, alpha=0.4,
            color=_COLORS[m_idx % len(_COLORS)],
            label=f"ML – {label}", histtype="step", linewidth=2
        )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title("Energy distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        ax.scatter(
            gt_energy, pred.squeeze(), s=1, alpha=0.2,
            color=_COLORS[m_idx % len(_COLORS)], label=f"ML – {label}"
        )
    emin = min(gt_energy.min(), min(p.min() for p in pred_energy_list))
    emax = max(gt_energy.max(), max(p.max() for p in pred_energy_list))
    ax.plot([emin, emax], [emin, emax], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("True energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Energy: predicted vs true")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        residual = (pred.squeeze() - gt_energy) / (gt_energy + 1e-12)
        median_res = np.median(np.abs(residual))
        ax.hist(
            residual, bins=100, density=True, alpha=0.5,
            color=_COLORS[m_idx % len(_COLORS)],
            label=f"ML – {label} (med|res|={median_res:.3f})"
        )
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

    fig, axes = plt.subplots(
        n_models, n_classes,
        figsize=(7 * n_classes, 5 * n_models),
        squeeze=False,
    )
    for m_idx, (pred, label) in enumerate(zip(pred_energy_list, model_labels)):
        for c_idx, cls in enumerate(unique_labels):
            ax = axes[m_idx, c_idx]
            mask = gt_labels == cls
            cls_name = pdg_names.get(int(cls), str(cls))
            ax.scatter(
                gt_energy[mask], pred[mask].squeeze(), s=1, alpha=0.3,
                color=_COLORS[m_idx % len(_COLORS)]
            )
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


# ── Per-sample plots ─────────────────────────────────────────────────────────

def _draw_direction_arrow_3d(ax, direction: np.ndarray, color: str, label: str) -> None:
    """Draw a unit direction arrow from the origin in a 3-D axes."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    ax.quiver(
        0, 0, 0, d[0], d[1], d[2],
        length=1.0, normalize=True, color=color,
        linewidth=2, label=label, arrow_length_ratio=0.25
    )


def plot_single_sample(
    sample_idx: int,
    gt_dir: np.ndarray,
    gt_label: int,
    gt_energy: float,
    pred_dirs: list[np.ndarray],
    pred_pdgs: list[float],
    pred_energies: list[float],
    model_labels: list[str],
    pdg_names: dict[int, str],
    outdir: Path,
) -> None:
    """
    Clean paper-style 2x2 figure per shower:
      - 3D direction comparison
      - Direction components
      - PDG prediction
      - Energy prediction
    """
    n_models = len(model_labels)
    colors_m = [_COLORS[i % len(_COLORS)] for i in range(n_models)]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Sample #{sample_idx}  ",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # ── (0,0): 3D direction arrows ───────────────────────────────────────
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_direction_arrow_3d(ax3d, gt_dir, "black", "Ground truth")
    for m_idx, (pd, ml) in enumerate(zip(pred_dirs, model_labels)):
        _draw_direction_arrow_3d(ax3d, pd, colors_m[m_idx], ml)

    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-1, 1)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Direction (3D)", fontsize=12)
    ax3d.legend(fontsize=9, loc="upper left", framealpha=0.9)

    # ── (0,1): Direction components ──────────────────────────────────────
    ax_comp = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(3)
    n_groups = n_models + 1
    bar_w = 0.8 / n_groups

    all_series = [("GT", gt_dir, "black")] + [
        (ml, pd, colors_m[m]) for m, (pd, ml) in enumerate(zip(pred_dirs, model_labels))
    ]

    for s_idx, (lbl, vals, col) in enumerate(all_series):
        offsets = x_pos + (s_idx - n_groups / 2 + 0.5) * bar_w
        bars = ax_comp.bar(offsets, vals, bar_w, label=lbl, color=col, alpha=0.8)
        ax_comp.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels([r"$d_x$", r"$d_y$", r"$d_z$"], fontsize=11)
    ax_comp.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_comp.set_ylabel("Component value", fontsize=11)
    ax_comp.set_title("Direction components", fontsize=12)
    ax_comp.legend(fontsize=9)
    ax_comp.grid(True, alpha=0.3, axis="y")

    # ── (1,0): PDG prediction ────────────────────────────────────────────
    ax_pdg = fig.add_subplot(gs[1, 0])

    class_keys = sorted(pdg_names.keys())
    ymin = min(class_keys) - 0.4
    ymax = max(class_keys) + 0.4
    ax_pdg.set_ylim(ymin, ymax)

    for cls in class_keys:
        ax_pdg.axhline(cls, color="lightgray", linestyle="--", linewidth=1.2, zorder=0)

    x_labels = ["GT"] + model_labels
    x_pos = np.arange(len(x_labels))

    ax_pdg.scatter(
        x_pos[0], gt_label,
        s=220, marker="D", color="black", edgecolor="white", linewidth=1.2,
        zorder=3, label="Ground truth"
    )


    for i, (raw_pdg, ml, col) in enumerate(zip(pred_pdgs, model_labels, colors_m), start=1):
        pred_cls = int(np.clip(np.floor(raw_pdg), min(class_keys), max(class_keys)))
        correct = pred_cls == gt_label

        ax_pdg.scatter(
            x_pos[i], pred_cls,
            s=240, marker="o", color=col, edgecolor="white", linewidth=1.5, zorder=3
        )





    ax_pdg.set_xticks(x_pos)
    ax_pdg.set_xticklabels(x_labels, fontsize=10, rotation=15)
    ax_pdg.set_yticks(class_keys)
    ax_pdg.set_yticklabels([pdg_names[k] for k in class_keys], fontsize=11)
    ax_pdg.set_ylabel("Class", fontsize=11)
    ax_pdg.set_title("Particle ID prediction", fontsize=12)
    ax_pdg.grid(False)

    # ── (1,1): Energy prediction ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])

    all_e_vals = [gt_energy] + [float(pe) for pe in pred_energies]
    all_e_labels = ["GT"] + model_labels
    all_e_colors = ["black"] + colors_m

    bars = ax_e.bar(
        all_e_labels, all_e_vals,
        color=all_e_colors, alpha=0.88, edgecolor="white", width=0.62
    )

    e_min = min(all_e_vals)
    e_max = max(all_e_vals)
    span = max(e_max - e_min, 1e-6)
    ax_e.set_ylim(0, e_max * 1.6)
    
    for bar, val in zip(bars, all_e_vals):
        ax_e.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03 * span + 0.02 * max(abs(e_max), 1.0),
            f"{val:.4g}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    residual_lines = []
    for ml, pe, col in zip(model_labels, pred_energies, colors_m):
        res = _energy_residual(gt_energy, float(pe))
        residual_lines.append(f"{ml}:  ΔE/E = {res:+.3f}")

    ax_e.text(
        0.03, 0.97,
        "\n".join(residual_lines),
        transform=ax_e.transAxes,
        ha="left", va="top", fontsize=9, fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="whitesmoke",
            edgecolor="lightgray",
            alpha=0.95,
        ),
    )

    ax_e.set_ylabel("Energy [GeV]", fontsize=11)
    ax_e.set_title("Energy prediction", fontsize=12)
    ax_e.tick_params(axis="x", labelsize=10, rotation=15)
    ax_e.grid(True, alpha=0.3, axis="y")

    fig.set_constrained_layout(True)
    fig.savefig(outdir / f"sample_{sample_idx:04d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample(
    gt_dirs: np.ndarray,
    gt_labels: np.ndarray,
    gt_energies: np.ndarray,
    all_preds: list[np.ndarray],
    all_labels: list[str],
    pdg_names: dict[int, str],
    outdir: Path,
    num_plots: int,
) -> None:
    """
    Generate one summary figure per shower and save to outdir/per_sample/.
    Also generates a grid overview page showing all thumbnails in one image.
    """
    n = min(num_plots, gt_dirs.shape[0])
    per_sample_dir = outdir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating {n} per-sample plots → {per_sample_dir}")
    for i in range(n):
        pred_dirs_i = [p[i, :3] for p in all_preds]
        pred_pdgs_i = [float(p[i, 3]) for p in all_preds]
        pred_energ_i = [float(p[i, 4]) for p in all_preds]

        plot_single_sample(
            sample_idx=i,
            gt_dir=gt_dirs[i],
            gt_label=int(gt_labels[i]),
            gt_energy=float(gt_energies[i]),
            pred_dirs=pred_dirs_i,
            pred_pdgs=pred_pdgs_i,
            pred_energies=pred_energ_i,
            model_labels=all_labels,
            pdg_names=pdg_names,
            outdir=per_sample_dir,
        )
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"    {i + 1}/{n} done")

    _plot_per_sample_grid(
        gt_dirs=gt_dirs[:n],
        gt_labels=gt_labels[:n],
        gt_energies=gt_energies[:n],
        all_preds=all_preds,
        all_labels=all_labels,
        pdg_names=pdg_names,
        outdir=outdir,
        n=n,
        cols=CONFIG["per_sample_grid_cols"],
    )


def _plot_per_sample_grid(
    gt_dirs: np.ndarray,
    gt_labels: np.ndarray,
    gt_energies: np.ndarray,
    all_preds: list[np.ndarray],
    all_labels: list[str],
    pdg_names: dict[int, str],
    outdir: Path,
    n: int,
    cols: int,
) -> None:
    """
    Compact grid: one mini-panel per shower showing angular error and energy
    residual for each model, colour-coded by accuracy.

    Saves to outdir/per_sample_overview.png.
    """
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(n):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        gt_cls_name = pdg_names.get(int(gt_labels[i]), str(gt_labels[i]))
        ax.set_title(f"#{i}  {gt_cls_name}  E={gt_energies[i]:.3g}", fontsize=7, fontweight="bold")
        ax.axis("off")

        lines = []
        for m_idx, ml in enumerate(all_labels):
            ang = _angular_error_deg(gt_dirs[i], all_preds[m_idx][i, :3])
            res = _energy_residual(gt_energies[i], float(all_preds[m_idx][i, 4]))
            pred_cls = int(np.clip(np.floor(all_preds[m_idx][i, 3]), 0, max(pdg_names)))
            ok = "✓" if pred_cls == int(gt_labels[i]) else "✗"
            lines.append(f"{ml}: Δθ={ang:.1f}° {ok}  ΔE={res:+.2f}")

        y = 0.82
        for ln in lines:
            ax.text(
                0.05, y, ln, transform=ax.transAxes,
                fontsize=6.5, fontfamily="monospace", color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="lightyellow",
                    edgecolor="lightgrey",
                    alpha=0.85,
                ),
            )
            y -= 0.25

        worst_ang = max(
            _angular_error_deg(gt_dirs[i], all_preds[m_idx][i, :3])
            for m_idx in range(len(all_labels))
        )
        border_col = "green" if worst_ang < 5 else "orange" if worst_ang < 20 else "red"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.5)

    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Per-sample overview  (border: green <5°, orange <20°, red ≥20°)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outdir / "per_sample_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: per_sample_overview.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_file = CONFIG["input_file"]
    model_specs = [(m["path"], m["label"]) for m in CONFIG["models"]]
    condition_features = CONFIG["condition_features"]
    pdg_names = CONFIG["pdg_names"]
    num_samples = CONFIG["num_samples"]
    plot_dir = CONFIG["plot_dir"]
    seed = CONFIG["seed"]
    num_per_sample = CONFIG.get("num_per_sample_plots", 20)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    print(f"Models to compare ({len(model_specs)}):")
    for path, label in model_specs:
        print(f"  [{label}]  {path}")

    print(f"\nLoading ground truth (last {num_samples} samples) ...")
    t0 = time.time()
    gt = load_ground_truth(input_file, condition_features, start=-num_samples, end=None)
    print(f"  Loaded {gt['directions'].shape[0]} samples in {(time.time() - t0) * 1000:.0f} ms")
    print(f"  Condition shape: {gt['condition'].shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(gt['labels'], return_counts=True)))}")

    condition_t = torch.from_numpy(gt["condition"]).to(torch.float32)

    all_preds: list[np.ndarray] = []
    all_labels: list[str] = []

    for path, label in model_specs:
        print(f"\nLoading model [{label}] ...")
        t0 = time.time()
        model = load_model(path)
        print(f"  Loaded in {(time.time() - t0) * 1000:.0f} ms")

        print("  Running inference ...")
        t0 = time.time()
        pred = run_inference(model, condition_t)
        dt = time.time() - t0
        n = pred.shape[0]
        print(f"  Done: {n} samples in {dt:.2f} s ({dt / n * 1000:.3f} ms/sample)")
        print(f"  Output shape: {pred.shape}")

        all_preds.append(pred)
        all_labels.append(label)

    gt_dirs = gt["directions"]
    gt_labels_arr = gt["labels"]
    gt_energy = gt["energies"]

    pred_dirs_list = [p[:, :3] for p in all_preds]
    pred_pdg_list = [p[:, 3:4] for p in all_preds]
    pred_energy_list = [p[:, 4:] for p in all_preds]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(plot_dir) / f"reconstruction_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # print("\nGenerating aggregate plots ...")

    # plot_direction_components(gt_dirs, pred_dirs_list, all_labels, outdir)
    # print("  Saved: directions.png")

    # plot_pdg_classification(gt_labels_arr, pred_pdg_list, all_labels, pdg_names, outdir)
    # print("  Saved: pdg_accuracy.png")

    # plot_pdg_distribution(gt_labels_arr, pred_pdg_list, all_labels, pdg_names, outdir)
    # print("  Saved: pdg_distribution.png")

    # plot_energy(gt_energy, pred_energy_list, all_labels, outdir)
    # print("  Saved: energy.png")

    # plot_energy_per_class(gt_energy, gt_labels_arr, pred_energy_list, all_labels, pdg_names, outdir)
    # print("  Saved: energy_per_class.png")

    print(f"\nGenerating per-sample plots (num_per_sample_plots={num_per_sample}) ...")
    plot_per_sample(
        gt_dirs=gt_dirs,
        gt_labels=gt_labels_arr,
        gt_energies=gt_energy,
        all_preds=all_preds,
        all_labels=all_labels,
        pdg_names=pdg_names,
        outdir=outdir,
        num_plots=num_per_sample,
    )

    print(f"\nDone. All plots saved to: {outdir}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()