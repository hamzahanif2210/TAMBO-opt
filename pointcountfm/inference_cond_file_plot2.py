#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────────────
# Configuration — edit this block, then just run:  python inference_cond_file_plot.py
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Input H5 file
    "input_file": (
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif"
        "/tambo_simulations_for_training/combined_electrons_balanced-test-file.h5"
    ),

    # Models to compare: list of {path, label} dicts.
    "models": [
        {
            "path": (
                "/n/home04/hhanif/AllShowers/results"
                "/20260402_235544_Electron-PointCountFM/compiled.pt"
            ),
            "label": "ML - Concquat Squash",
        },
        {
            "path": (
                "/n/home04/hhanif/AllShowers/results"
                "/20260402_234941_Electron-PointCountFM/compiled.pt"
            ),
            "label": "ML - Fully Connected",
        },
        {
            "path": "/n/home04/hhanif/AllShowers/results/20260403_013247_Electron-PointCountFM/compiled.pt",
            "label": "NL - Fully Connected with dequantized",
        },
    ],

    # How many showers to randomly sample PER PDG CLASS for per-shower plots.
    # Total plots = showers_per_class * len(pdg_codes)
    "showers_per_class": 2000,

    # Pool to draw from: the script reads this many showers from the start of
    # the file to build the random sample.  Set to -1 to use the whole file.
    "pool_stop": 5000,

    # Detector geometry
    "num_layers": 24,

    # PDG codes that define the one-hot class ordering.
    # 0 = e±/γ/π⁰,  1 = π±
    "pdg_codes": [0, 1],

    # Human-readable names for each PDG class (same order as pdg_codes).
    "pdg_names": {
        0: "e±/π⁰",
        1: "π±",
    },

    # Name of the variable-length showers dataset inside the H5 file.
    "showers_dset": "showers",

    # Where to write output plots.
    "plot_dir": "/n/home04/hhanif/AllShowers/plots",

    # Random seed for reproducible shower sampling.
    "seed": 42,
}
# ──────────────────────────────────────────────────────────────────────────────

import time
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import showerdata


# ── PDG helpers ───────────────────────────────────────────────────────────────

def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def calc_num_points_per_layer_h5(h5_dataset, indices: np.ndarray, num_layers: int) -> np.ndarray:
    """
    Count number of hits per layer for an arbitrary list of global shower indices.
    layer index = (z + 0.1).astype(int32)
    """
    points_per_layer = np.zeros((len(indices), num_layers), dtype=np.int32)
    for i, global_i in enumerate(indices):
        shower = np.array(h5_dataset[global_i])
        if shower.size == 0:
            continue
        points = shower.reshape(-1, 5)  # x, y, z, e, pdg (5 cols)
        layer_idx = np.clip((points[:, 2] + 0.1).astype(np.int32), 0, num_layers - 1)
        mask = (points[:, 3] > 0).astype(np.int32)
        np.add.at(points_per_layer[i], layer_idx, mask)
    return points_per_layer


# ── Model helpers ─────────────────────────────────────────────────────────────

class InferenceModel(torch.nn.Module):
    def __init__(self, model: torch.jit.ScriptModule):
        super().__init__()
        self.model = model

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(conditions.size(0)):
            results.append(self.model(conditions[[i]]))
        return torch.cat(results, dim=0)


def load_model(path: str) -> InferenceModel:
    model = torch.jit.load(path)
    model.eval()
    inference = InferenceModel(model).to(torch.float32)
    inference = torch.jit.script(inference)
    return inference


def run_inference(inference: InferenceModel, conditions: torch.Tensor,
                  num_layers: int, n: int) -> np.ndarray:
    ml = inference(conditions)
    ml = (torch.clamp(ml, min=0.0) + 0.5).to(torch.int32).cpu().numpy()
    if ml.ndim != 2 or ml.shape[1] != num_layers:
        raise ValueError(f"ML output has shape {ml.shape}, expected ({n}, {num_layers})")
    return ml


# ── Colour / marker cycle ─────────────────────────────────────────────────────
_COLORS  = ["tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
_MARKERS = ["o", "^", "D", "v", "s", "P", "*", "X"]

GT_COLOR  = "black"
GT_MARKER = "s"


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save_per_shower_plot(global_idx: int, gt_i: np.ndarray,
                          all_ml_i: list, all_labels: list,
                          num_layers: int, pdg_name: str,
                          out_path: Path) -> None:
    x = np.arange(num_layers, dtype=np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Left: counts ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, gt_i, linestyle="None", marker=GT_MARKER, color=GT_COLOR,
            label="Ground truth (CORSIKA)", zorder=10)
    for m_idx, (ml_i, label) in enumerate(zip(all_ml_i, all_labels)):
        ax.plot(x, ml_i, linestyle="None",
                marker=_MARKERS[m_idx % len(_MARKERS)],
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label}")
    ax.set_xlabel("layer_idx")
    ax.set_ylabel("num_points")
    ax.set_title(f"Shower {global_idx} [{pdg_name}]: num points per layer")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: ratio ML / GT ──────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(1.0, color=GT_COLOR, linestyle="--", linewidth=1.5,
               label="Perfect (ratio = 1)")
    safe_gt = np.where(gt_i > 0, gt_i.astype(float), np.nan)
    for m_idx, (ml_i, label) in enumerate(zip(all_ml_i, all_labels)):
        ratio = ml_i.astype(float) / safe_gt
        ax.plot(x, ratio, linestyle="-",
                marker=_MARKERS[m_idx % len(_MARKERS)],
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label}")
    ax.set_xlabel("layer_idx")
    ax.set_ylabel("ML / GT")
    ax.set_title(f"Shower {global_idx} [{pdg_name}]: ratio ML / GT per layer")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_average_plot(gt: np.ndarray, all_ml: list, all_labels: list,
                       num_layers: int, n: int, title_suffix: str,
                       out_path: Path) -> None:
    x = np.arange(num_layers, dtype=np.int32)
    gt_mean = gt.mean(axis=0)
    gt_std  = gt.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean ± std
    ax = axes[0]
    ax.errorbar(x, gt_mean, yerr=gt_std, fmt=GT_MARKER, color=GT_COLOR,
                capsize=3, label="Ground truth (CORSIKA)", zorder=10)
    for m_idx, (ml, label) in enumerate(zip(all_ml, all_labels)):
        ml_mean = ml.mean(axis=0)
        ml_std  = ml.std(axis=0)
        ax.errorbar(x, ml_mean, yerr=ml_std,
                    fmt=_MARKERS[m_idx % len(_MARKERS)],
                    color=_COLORS[m_idx % len(_COLORS)],
                    capsize=3, label=f"ML – {label}")
    ax.set_xlabel("layer_idx")
    ax.set_ylabel("mean num_points")
    ax.set_title(f"Average num points per layer{title_suffix} (N={n})\nmean ± 1σ")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: ratio ML mean / GT mean
    ax = axes[1]
    safe_gt = np.where(gt_mean > 0, gt_mean, np.nan)
    ax.axhline(1.0, color=GT_COLOR, linestyle="--", linewidth=1.5,
               label="Perfect (ratio = 1)")
    for m_idx, (ml, label) in enumerate(zip(all_ml, all_labels)):
        ml_mean = ml.mean(axis=0)
        ratio = ml_mean / safe_gt
        ax.plot(x, ratio, linestyle="-",
                marker=_MARKERS[m_idx % len(_MARKERS)],
                color=_COLORS[m_idx % len(_COLORS)],
                label=f"ML – {label}")
    ax.set_xlabel("layer_idx")
    ax.set_ylabel("ML mean / GT mean")
    ax.set_title(f"Average ratio ML / GT per layer{title_suffix} (N={n})")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_file       = CONFIG["input_file"]
    model_specs      = [(m["path"], m["label"]) for m in CONFIG["models"]]
    pdg_codes        = CONFIG["pdg_codes"]
    pdg_names        = CONFIG["pdg_names"]
    num_layers       = CONFIG["num_layers"]
    showers_dset     = CONFIG["showers_dset"]
    plot_dir         = CONFIG["plot_dir"]
    showers_per_class = CONFIG["showers_per_class"]
    pool_stop        = CONFIG["pool_stop"]
    seed             = CONFIG["seed"]

    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)

    print(f"Models to compare ({len(model_specs)}):")
    for path, label in model_specs:
        print(f"  [{label}]  {path}")

    # ── Resolve pool range ────────────────────────────────────────────────────
    file_len = showerdata.get_file_length(input_file)
    pool_end = file_len if pool_stop == -1 else min(pool_stop, file_len)
    pool_start = 0
    print(f"\nPool: showers [{pool_start}, {pool_end})  (file has {file_len} showers)")

    # ── Load conditioning data for the whole pool ─────────────────────────────
    print(f"Loading conditioning data for pool ...")
    t0 = time.time()
    cond_data = showerdata.load_inc_particles(input_file, start=pool_start, stop=pool_end)
    pool_labels_np = to_labels(torch.from_numpy(cond_data.pdg), pdg_codes).numpy()
    print(f"  Done in {(time.time() - t0) * 1000.0:.1f} ms")

    # Check all PDGs are covered
    missing_mask = pool_labels_np < 0
    if missing_mask.any():
        bad_pdgs = np.unique(cond_data.pdg[missing_mask])
        raise ValueError(
            f"PDGs not in CONFIG['pdg_codes']: {bad_pdgs.tolist()}. "
            "Add them to pdg_codes (and pdg_names)."
        )

    # ── Sample showers_per_class indices per PDG class ────────────────────────
    sampled_global: dict[int, np.ndarray] = {}   # pdg_code -> global indices
    sampled_pool:   dict[int, np.ndarray] = {}   # pdg_code -> pool-relative indices

    for pdg in pdg_codes:
        pdg_label  = pdg_codes.index(pdg)
        pool_idx   = np.where(pool_labels_np == pdg_label)[0]
        if len(pool_idx) == 0:
            raise ValueError(f"No showers with PDG code {pdg} in pool.")
        chosen = rng.choice(pool_idx, size=min(showers_per_class, len(pool_idx)),
                            replace=False)
        chosen.sort()
        sampled_pool[pdg]   = chosen
        sampled_global[pdg] = chosen + pool_start   # offset to file indices

    all_sampled_pool   = np.concatenate([sampled_pool[p]   for p in pdg_codes])
    all_sampled_global = np.concatenate([sampled_global[p] for p in pdg_codes])

    # Build conditions for sampled showers only
    pool_labels_t = torch.from_numpy(pool_labels_np[all_sampled_pool])
    conditions = torch.concatenate(
        (
            torch.from_numpy(cond_data.energies[all_sampled_pool]).to(torch.float32),
            torch.nn.functional.one_hot(pool_labels_t,
                                        num_classes=len(pdg_codes)).to(torch.float32),
            torch.from_numpy(cond_data.directions[all_sampled_pool]).to(torch.float32),
        ),
        dim=1,
    )
    n_total = len(all_sampled_pool)

    # ── ML inference ─────────────────────────────────────────────────────────
    all_ml: list[np.ndarray] = []
    all_labels: list[str]    = []

    for path, label in model_specs:
        print(f"\nLoading model [{label}] ...")
        t0 = time.time()
        inference = load_model(path)
        print(f"  Ready in {(time.time() - t0) * 1000.0:.1f} ms")

        print(f"  Running inference [{label}] ...")
        t0 = time.time()
        ml = run_inference(inference, conditions, num_layers, n_total)
        dt = time.time() - t0
        print(f"  Done in {dt:.2f} s ({(dt / n_total) * 1000.0:.3f} ms / shower)")

        all_ml.append(ml)
        all_labels.append(label)

    # ── Ground truth ─────────────────────────────────────────────────────────
    print("\nComputing ground truth ...")
    t0 = time.time()
    with h5py.File(input_file, "r") as hf:
        if showers_dset not in hf:
            raise KeyError(f"Dataset '{showers_dset}' not found. "
                           f"Keys: {list(hf.keys())}")
        gt_all = calc_num_points_per_layer_h5(hf[showers_dset],
                                               all_sampled_global, num_layers)
    print(f"  Done in {(time.time() - t0):.2f} s")

    # ── Output dirs ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(plot_dir) / f"compare_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Per-shower plots (per PDG class sub-folder) ───────────────────────────
    offset = 0
    for pdg in pdg_codes:
        pdg_name  = pdg_names.get(pdg, str(pdg))
        n_cls     = len(sampled_pool[pdg])
        cls_dir   = outdir / f"per_shower_pdg{pdg}_{pdg_name.replace('/', '_')}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving {n_cls} per-shower plots for PDG {pdg} [{pdg_name}] ...")

        for i in range(n_cls):
            flat_i     = offset + i
            global_idx = int(all_sampled_global[flat_i])
            gt_i       = gt_all[flat_i]
            ml_i_list  = [ml[flat_i] for ml in all_ml]

            out_path = cls_dir / f"shower_{global_idx:07d}.png"
            _save_per_shower_plot(
                global_idx, gt_i, ml_i_list, all_labels,
                num_layers, pdg_name, out_path,
            )

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_cls} saved ...")

        offset += n_cls

    # ── Average plots: one per PDG class + one overall ────────────────────────
    print("\nSaving average plots ...")

    offset = 0
    for pdg in pdg_codes:
        pdg_name = pdg_names.get(pdg, str(pdg))
        n_cls    = len(sampled_pool[pdg])
        sl       = slice(offset, offset + n_cls)

        gt_cls  = gt_all[sl]
        ml_cls  = [ml[sl] for ml in all_ml]

        avg_path = outdir / f"average_pdg{pdg}_{pdg_name.replace('/', '_')}.png"
        _save_average_plot(gt_cls, ml_cls, all_labels, num_layers,
                           n_cls, f" [{pdg_name}]", avg_path)
        print(f"  Saved: {avg_path}")
        offset += n_cls

    # Overall average (all classes combined)
    avg_all_path = outdir / "average_all_classes.png"
    _save_average_plot(gt_all, all_ml, all_labels, num_layers,
                       n_total, " [all classes]", avg_all_path)
    print(f"  Saved: {avg_all_path}")

    print(f"\nDone.  Output root: {outdir}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()