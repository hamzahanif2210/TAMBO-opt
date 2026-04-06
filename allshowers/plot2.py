import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse


# =============================================================================
# Config
# =============================================================================

ml_file = "/n/home04/hhanif/AllShowers/results/20260404_205538_Electron-Allshower/samples01.h5"
simulated_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced-test-file.h5"

NUM_LAYERS = 24
US = 1e6  # seconds -> microseconds

CLASS_NAMES = {
    0: r"$e^\pm/\gamma/\pi^0$",
    1: r"$\pi^\pm$",
}

R_MIN = 0.0
R_MAX = 200.0
N_R_BINS = 60

SIM_COLOR = "black"
ML_COLOR = "blue"
RATIO_COLOR = "blue"


# =============================================================================
# Loading
# =============================================================================

def load_all(path):
    print(f"  Reading {path} ...")
    with h5py.File(path, "r") as f:
        pdg = f["pdg"][:]
        shape = f["shape"][:]
        raw_flat = f["showers"][:]

    n, max_pts, ncols = int(shape[0]), int(shape[1]), int(shape[2])
    print(f"  shape = ({n}, {max_pts}, {ncols})")

    if ncols < 4:
        raise ValueError(f"Expected ≥4 columns [x,y,z,e], got ncols={ncols}")

    if raw_flat.dtype == object:
        pts = np.zeros((n, max_pts, ncols), dtype=np.float32)
        for i, flat in enumerate(raw_flat):
            arr = np.asarray(flat, dtype=np.float32).reshape(-1, ncols)
            pts[i, :len(arr)] = arr
    else:
        pts = raw_flat.astype(np.float32).reshape(n, max_pts, ncols)

    energies = None
    directions = None
    with h5py.File(path, "r") as f:
        if "energies" in f:
            energies = f["energies"][:]
        if "directions" in f:
            directions = f["directions"][:]

    return pdg, pts, ncols, directions, energies


# =============================================================================
# Shower observables
# =============================================================================

def _layer_indices(pts, num_layers):
    return np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)


def compute_energy_per_layer(pts, num_layers=NUM_LAYERS):
    n, _ = pts.shape[:2]
    layer_idx = _layer_indices(pts, num_layers)
    energy = pts[..., 3].astype(np.float64)
    shower_offset = (np.arange(n) * num_layers)[:, None]
    flat_idx = (layer_idx + shower_offset).ravel()

    out = np.bincount(
        flat_idx,
        weights=energy.ravel(),
        minlength=n * num_layers
    ).reshape(n, num_layers)

    return out.astype(np.float32)


def compute_avg_time_per_layer(pts, num_layers=NUM_LAYERS):
    if pts.shape[2] < 5:
        raise ValueError(f"ncols={pts.shape[2]}: no time column present")

    n, _ = pts.shape[:2]
    mask = pts[..., 3] > 0
    t_zeroed = np.where(mask, pts[..., 4], 0.0)
    count = mask.astype(np.float64)

    layer_idx = _layer_indices(pts, num_layers)
    shower_offset = (np.arange(n) * num_layers)[:, None]
    flat_idx = (layer_idx + shower_offset).ravel()
    total = n * num_layers

    t_sum = np.bincount(
        flat_idx,
        weights=t_zeroed.ravel(),
        minlength=total
    ).reshape(n, num_layers)

    c_sum = np.bincount(
        flat_idx,
        weights=count.ravel(),
        minlength=total
    ).reshape(n, num_layers)

    return t_sum / np.clip(c_sum, 1, None)


def calc_energy_per_radial_bin_like_observables(pts, bin_edges):
    """
    Match the logic of calc_energy_per_radial_bin(...) from observables.py:
      - radial distance = sqrt(x^2 + y^2)
      - bin assignment via np.digitize(...)-1
      - energies thresholded with energy > 0
      - overflow bins are zeroed
    """
    bin_edges = np.asarray(bin_edges, dtype=np.float32)
    n = pts.shape[0]
    num_bins = len(bin_edges) - 1

    radial_distances = np.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
    bin_indices = np.digitize(radial_distances, bins=bin_edges) - 1

    energy_per_radial_bin = np.zeros((n, num_bins), dtype=np.float32)

    shower_indices = (
        np.arange(n).reshape(-1, 1).repeat(pts.shape[1], axis=1)
    )

    energies = pts[..., 3].astype(np.float32) * (pts[..., 3] > 0).astype(np.float32)

    overflow = (bin_indices == num_bins)
    energies = energies.copy()
    bin_indices = bin_indices.copy()
    energies[overflow] = 0.0
    bin_indices[overflow] = 0

    # safety for underflow, though with non-negative radii it normally won't happen
    underflow = bin_indices < 0
    energies[underflow] = 0.0
    bin_indices[underflow] = 0

    np.add.at(energy_per_radial_bin, (shower_indices, bin_indices), energies)
    return energy_per_radial_bin


def compute_cell_energy_spectrum_with_errors(pts, bins):
    mask = pts[..., 3] > 0
    e_flat = pts[..., 3][mask]
    counts, _ = np.histogram(e_flat, bins=bins)
    errors = np.sqrt(counts.astype(np.float64))
    return counts.astype(np.float64), errors


# =============================================================================
# Stats helpers
# =============================================================================

def mean_and_sem(per_shower_matrix):
    mean = per_shower_matrix.mean(axis=0)
    n = per_shower_matrix.shape[0]
    if n > 1:
        sem = per_shower_matrix.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def ratio_and_error(num, num_err, den, den_err):
    ratio = np.full_like(num, np.nan, dtype=np.float64)
    ratio_err = np.full_like(num, np.nan, dtype=np.float64)

    valid = (den > 0) & np.isfinite(den) & np.isfinite(num)
    ratio[valid] = num[valid] / den[valid]

    valid_err = valid & (num > 0)
    if np.any(valid_err):
        rel_num = np.zeros_like(num, dtype=np.float64)
        rel_den = np.zeros_like(den, dtype=np.float64)
        rel_num[valid_err] = num_err[valid_err] / num[valid_err]
        rel_den[valid_err] = den_err[valid_err] / den[valid_err]
        ratio_err[valid_err] = ratio[valid_err] * np.sqrt(
            rel_num[valid_err] ** 2 + rel_den[valid_err] ** 2
        )

    valid_zero_num = valid & (num == 0)
    ratio[valid_zero_num] = 0.0
    ratio_err[valid_zero_num] = 0.0

    return ratio, ratio_err


# =============================================================================
# Misc helpers
# =============================================================================

def mask_for_class(pdg_arr, class_value):
    if class_value is None:
        return np.ones(len(pdg_arr), dtype=bool)
    return pdg_arr == class_value


def subsample_indices(idx, n_samples, rng=None):
    if n_samples is None or n_samples >= len(idx):
        return idx
    if rng is None:
        rng = np.random.default_rng()
    chosen = rng.choice(len(idx), size=n_samples, replace=False)
    chosen.sort()
    return idx[chosen]


# =============================================================================
# Plot helpers
# =============================================================================

def _stairs_values(ax, edges, values, label=None, color=None, linewidth=1.8):
    kwargs = {"linewidth": linewidth}
    if label is not None:
        kwargs["label"] = label
    if color is not None:
        kwargs["color"] = color
    return ax.stairs(values, edges, **kwargs)


def _step_fill(ax, edges, lower, upper, color, alpha=0.22, logx=False):
    if logx:
        x = np.sqrt(edges[:-1] * edges[1:])
    else:
        x = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(
        x, lower, upper,
        step="mid",
        color=color,
        alpha=alpha,
        linewidth=0
    )


def _stairs_with_band(ax, edges, values, errors, label, color, alpha_fill=0.22, logx=False, logy=False):
    floor = 1e-12 if logy else 0.0
    lower = np.clip(values - errors, floor, None)
    upper = values + errors
    _stairs_values(ax, edges, values, label=label, color=color, linewidth=1.8)
    _step_fill(ax, edges, lower, upper, color=color, alpha=alpha_fill, logx=logx)


def _stairs_ratio_with_band(ax, edges, ratio, ratio_err, color=RATIO_COLOR, alpha_fill=0.22, logx=False):
    lower = ratio - np.nan_to_num(ratio_err, nan=0.0)
    upper = ratio + np.nan_to_num(ratio_err, nan=0.0)

    valid = np.isfinite(ratio)
    r = np.where(valid, ratio, np.nan)
    lo = np.where(valid, lower, np.nan)
    hi = np.where(valid, upper, np.nan)

    _stairs_values(ax, edges, r, color=color, linewidth=1.6)
    _step_fill(ax, edges, lo, hi, color=color, alpha=alpha_fill, logx=logx)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)


def style_ratio_axis(ax, xlabel, xlim=None, xticks=None, logx=False, ylim=(0.5, 1.5)):
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if logx:
        ax.set_xscale("log")
    ax.set_ylabel("ML / Sim", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(True, lw=0.4)
    ax.tick_params(labelsize=7)
    ax.set_ylim(*ylim)


# =============================================================================
# Plotting one panel pair
# =============================================================================

def plot_main_and_ratio(ax_main, ax_ratio, edges, sim_vals, sim_err, ml_vals, ml_err,
                        sim_label, ml_label, ylabel, xlabel, title,
                        sim_color=SIM_COLOR, ml_color=ML_COLOR,
                        xlim=None, xticks=None, logx=False, logy=False,
                        ratio_ylim=(0.5, 1.5)):
    _stairs_with_band(ax_main, edges, sim_vals, sim_err, sim_label,
                      color=sim_color, alpha_fill=0.20, logx=logx, logy=logy)
    _stairs_with_band(ax_main, edges, ml_vals, ml_err, ml_label,
                      color=ml_color, alpha_fill=0.20, logx=logx, logy=logy)

    if xlim is not None:
        ax_main.set_xlim(*xlim)
    if xticks is not None:
        ax_main.set_xticks(xticks)
    if logx:
        ax_main.set_xscale("log")
    if logy:
        ax_main.set_yscale("log")

    ax_main.grid(True, lw=0.4)
    ax_main.legend(fontsize=7)
    ax_main.set_title(title, fontsize=9)
    ax_main.set_ylabel(ylabel, fontsize=8)
    ax_main.tick_params(labelsize=7)

    ratio, ratio_err = ratio_and_error(ml_vals, ml_err, sim_vals, sim_err)
    _stairs_ratio_with_band(ax_ratio, edges, ratio, ratio_err,
                            color=RATIO_COLOR, alpha_fill=0.20, logx=logx)
    style_ratio_axis(ax_ratio, xlabel=xlabel, xlim=xlim, xticks=xticks,
                     logx=logx, ylim=ratio_ylim)


# =============================================================================
# Plotting
# =============================================================================

def plot_row(main_axes, ratio_axes, n, s_pts_sel, m_pts_sel, r_bins, num_layers=NUM_LAYERS):
    sim_label = f"Simulated ({n})"
    ml_label = f"ML ({n})"

    layer_edges = np.arange(0.5, num_layers + 1.5, 1.0)
    layer_xlim = (0.5, num_layers + 0.5)
    layer_xticks = np.arange(1, num_layers + 1, 4)

    # ------------------------------------------------------------------ #
    # 0. Average time per layer -> SEM, ratio 0.8 to 1.2
    # ------------------------------------------------------------------ #
    axm = main_axes[0]
    axr = ratio_axes[0]
    has_time = (s_pts_sel.shape[2] >= 5 and m_pts_sel.shape[2] >= 5)

    if has_time:
        s_t_layer = compute_avg_time_per_layer(s_pts_sel, num_layers) * US
        m_t_layer = compute_avg_time_per_layer(m_pts_sel, num_layers) * US

        s_mean, s_sem = mean_and_sem(s_t_layer)
        m_mean, m_sem = mean_and_sem(m_t_layer)

        plot_main_and_ratio(
            axm, axr, layer_edges,
            s_mean, s_sem, m_mean, m_sem,
            sim_label, ml_label,
            ylabel=r"Mean $t$ [$\mu$s]",
            xlabel="Layer",
            title="Avg Time per Layer",
            xlim=layer_xlim,
            xticks=layer_xticks,
            logx=False,
            logy=False,
            ratio_ylim=(0.95, 1.05),
        )
    else:
        axm.text(0.5, 0.5, "No time column", ha="center", va="center",
                 transform=axm.transAxes, fontsize=8)
        axm.set_title("Avg Time per Layer", fontsize=9)
        axm.axis("off")
        axr.axis("off")

    # ------------------------------------------------------------------ #
    # 1. Longitudinal energy profile -> mean ± SEM, not normalized
    # ------------------------------------------------------------------ #
    axm = main_axes[1]
    axr = ratio_axes[1]
    s_e_layer = compute_energy_per_layer(s_pts_sel, num_layers)
    m_e_layer = compute_energy_per_layer(m_pts_sel, num_layers)

    s_mean, s_sem = mean_and_sem(s_e_layer)
    m_mean, m_sem = mean_and_sem(m_e_layer)

    plot_main_and_ratio(
        axm, axr, layer_edges,
        s_mean, s_sem, m_mean, m_sem,
        sim_label, ml_label,
        ylabel="Mean Energy",
        xlabel="Layer",
        title="Longitudinal Energy Profile",
        xlim=layer_xlim,
        xticks=layer_xticks,
        logx=False,
        logy=False,
        ratio_ylim=(0.5, 1.5),
    )

    # ------------------------------------------------------------------ #
    # 2. Radial energy profile -> use calc_energy_per_radial_bin-style logic
    # ------------------------------------------------------------------ #
    axm = main_axes[2]
    axr = ratio_axes[2]

    s_r_mat = calc_energy_per_radial_bin_like_observables(s_pts_sel, r_bins)
    m_r_mat = calc_energy_per_radial_bin_like_observables(m_pts_sel, r_bins)

    s_mean, s_sem = mean_and_sem(s_r_mat)
    m_mean, m_sem = mean_and_sem(m_r_mat)

    plot_main_and_ratio(
        axm, axr, r_bins,
        s_mean, s_sem, m_mean, m_sem,
        sim_label, ml_label,
        ylabel="Mean energy / bin",
        xlabel="Radius [m]",
        title="Radial Energy Profile",
        xlim=(r_bins[0], r_bins[-1]),
        xticks=None,
        logx=False,
        logy=False,
        ratio_ylim=(0.5, 1.5),
    )

    # ------------------------------------------------------------------ #
    # 3. Cell energy spectrum -> counts ± Poisson
    # ------------------------------------------------------------------ #
    axm = main_axes[3]
    axr = ratio_axes[3]
    s_e_all = s_pts_sel[..., 3][s_pts_sel[..., 3] > 0]
    m_e_all = m_pts_sel[..., 3][m_pts_sel[..., 3] > 0]

    if len(s_e_all) > 0 and len(m_e_all) > 0:
        e_min = max(min(s_e_all.min(), m_e_all.min()), 1e-6)
        e_max = max(s_e_all.max(), m_e_all.max())
        if e_max <= e_min:
            e_max = e_min * 10.0

        bins_e = np.logspace(np.log10(e_min), np.log10(e_max), 80)

        s_counts, s_err = compute_cell_energy_spectrum_with_errors(s_pts_sel, bins_e)
        m_counts, m_err = compute_cell_energy_spectrum_with_errors(m_pts_sel, bins_e)

        _stairs_values(axm, bins_e, s_counts, label=sim_label, color=SIM_COLOR, linewidth=1.8)
        _step_fill(axm, bins_e, np.clip(s_counts - s_err, 1e-12, None), s_counts + s_err,
                   color=SIM_COLOR, alpha=0.20, logx=True)
        _stairs_values(axm, bins_e, m_counts, label=ml_label, color=ML_COLOR, linewidth=1.8)
        _step_fill(axm, bins_e, np.clip(m_counts - m_err, 1e-12, None), m_counts + m_err,
                   color=ML_COLOR, alpha=0.20, logx=True)

        axm.set_xscale("log")
        axm.set_yscale("log")
        axm.grid(True, lw=0.4)
        axm.legend(fontsize=7)
        axm.set_title("Cell Energy Spectrum", fontsize=9)
        axm.set_ylabel("Number of cells", fontsize=8)
        axm.tick_params(labelsize=7)

        ratio, ratio_err = ratio_and_error(m_counts, m_err, s_counts, s_err)
        _stairs_ratio_with_band(axr, bins_e, ratio, ratio_err,
                                color=RATIO_COLOR, alpha_fill=0.20, logx=True)
        style_ratio_axis(axr, xlabel="Cell energy", logx=True, ylim=(0.5, 1.5))
    else:
        axm.axis("off")
        axr.axis("off")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot shower observables comparing ML vs simulated samples with ratio panels."
    )
    parser.add_argument(
        "--n-samples",
        type=str,
        default="all",
        help="Number of samples per row or 'all'. Default: all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_samples.strip().lower() == "all":
        n_samples_requested = None
        sample_label = "all"
    else:
        try:
            n_samples_requested = int(args.n_samples)
            if n_samples_requested <= 0:
                raise ValueError
            sample_label = str(n_samples_requested)
        except ValueError:
            raise ValueError(
                f"--n-samples must be a positive integer or 'all', got: {args.n_samples!r}"
            )

    rng = np.random.default_rng(args.seed)

    print("\nLoading Simulated file...")
    s_pdg, s_pts, s_ncols, s_dir, s_energy = load_all(simulated_file)

    print("Loading ML file...")
    m_pdg, m_pts, m_ncols, m_dir, m_energy = load_all(ml_file)

    if len(s_pdg) != len(m_pdg):
        raise ValueError(
            f"Row count mismatch: simulated={len(s_pdg)}, ML={len(m_pdg)}. "
            "Files must be in the same sorted order."
        )
    print(f"\nTotal samples: {len(s_pdg)}")

    r_bins = np.linspace(R_MIN, R_MAX, N_R_BINS + 1)

    row_configs = [
        ("All", None),
        (f"Class 0: {CLASS_NAMES[0]}", 0),
        (f"Class 1: {CLASS_NAMES[1]}", 1),
    ]

    nrows = len(row_configs)

    fig = plt.figure(figsize=(4 * 4.4, nrows * 4.8))
    outer = fig.add_gridspec(nrows=nrows, ncols=4, wspace=0.28, hspace=0.42)

    for row_i, (row_label, class_val) in enumerate(row_configs):
        idx_full = np.where(mask_for_class(s_pdg, class_val))[0]
        idx = subsample_indices(idx_full, n_samples_requested, rng=rng)
        n = len(idx)

        main_axes = []
        ratio_axes = []

        for col in range(4):
            sub = outer[row_i, col].subgridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.06)
            ax_main = fig.add_subplot(sub[0, 0])
            ax_ratio = fig.add_subplot(sub[1, 0], sharex=ax_main)
            plt.setp(ax_main.get_xticklabels(), visible=False)
            main_axes.append(ax_main)
            ratio_axes.append(ax_ratio)

        if n == 0:
            for ax in main_axes + ratio_axes:
                ax.axis("off")
            continue

        plot_row(
            main_axes,
            ratio_axes,
            n,
            s_pts[idx],
            m_pts[idx],
            r_bins,
            num_layers=NUM_LAYERS,
        )

        main_axes[0].annotate(
            f"{row_label} — samples: {n}"
            + (f" (of {len(idx_full)})" if n < len(idx_full) else ""),
            xy=(0, 1.20),
            xycoords="axes fraction",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.suptitle(
        f"Shower Observables with Ratio Panels  [n_samples={sample_label}]",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"shower_observables_ratio_n{sample_label}_radial_fix.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()