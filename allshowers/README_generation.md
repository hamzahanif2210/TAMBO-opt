# TAMBO Shower Generation — Complete Guide

This document explains the **two-stage shower generation pipeline**, the
checkpoint locations, all inputs and outputs, and how to run everything from a
single script.

---

## Pipeline Overview

```
Primary particle (E, direction, label)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1 – PointCountFM         │  checkpoint: checkpoints/compiled/
│  (TorchScript compiled model)   │
│                                 │
│  Input:  energy (1) + one_hot   │
│          label (2) + direction  │
│          (3)  →  (N, 6)         │
│                                 │
│  Output: num_points_per_layer   │
│          (N, 24)  int32         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  Stage 2 – AllShowers CNF       │  checkpoint: checkpoints/all_showers/
│  (Transformer flow model)       │
│                                 │
│  Input:  energy (1) + direction │
│          (3) + num_points (24)  │
│          + label (int)          │
│                                 │
│  Output: shower point cloud     │
│          (N, max_pts, 5)        │
│          x, y, layer, E, t      │
└─────────────────────────────────┘
        │
        ▼
  generated_showers.h5
```

---

## Checkpoint Locations

Both model checkpoints live under:
```
TAMBO-opt/allshowers/checkpoints/
```

| Path | Contents | Used by |
|---|---|---|
| `checkpoints/compiled/` | TorchScript compiled PointCountFM model | Stage 1 |
| `checkpoints/all_showers/` | AllShowers run dir (conf.yaml, weights/best.pt, preprocessing/trafos.pt) | Stage 2 |

The `all_showers/` directory has the same structure as any run directory
produced by `train.py`:

```
checkpoints/all_showers/
├── conf.yaml                ← model + data config
├── weights/
│   └── best.pt              ← weights at lowest validation loss  ← used for generation
├── checkpoints/
│   └── last.pt              ← full training state (resume training only)
├── preprocessing/
│   └── trafos.pt            ← fitted preprocessing transforms
├── data/losses.txt
└── plots/
```

---

## Primary Particle Inputs (randomly sampled by `generate_showers.py`)

| Parameter | Range | Distribution |
|---|---|---|
| Energy | 1×10⁵ – 1×10⁸ GeV | Log-uniform |
| Zenith angle θ | 60° – 100° | Uniform |
| Azimuth angle φ | 0° – 360° | Uniform |
| Particle label | 0 or 1 | Uniform |

The direction is converted to a 3D CORSIKA unit vector:
```
nx = sin(θ) · cos(φ)
ny = sin(θ) · sin(φ)
nz = cos(θ)
```
This `(N, 3)` direction vector is fed to **both** models.

---

## Model Inputs & Outputs

### Stage 1 — PointCountFM

| Tensor | Shape | dtype | Description |
|---|---|---|---|
| `energies` | `(N, 1)` | float32 | Primary particle energy |
| `one_hot(label, 2)` | `(N, 2)` | float32 | Particle class (0 or 1 → one-hot) |
| `directions` | `(N, 3)` | float32 | CORSIKA unit direction vector |
| **condition** (concat) | `(N, 6)` | float32 | Full model input |

**Output**: `num_points_per_layer` — `(N, 24)` int32, number of secondary
particles in each of the 24 detector layers.

### Stage 2 — AllShowers CNF-Transformer

| Tensor | Shape | dtype | Description |
|---|---|---|---|
| `energies` | `(N, 1)` | float32 | Primary particle energy |
| `directions` | `(N, 3)` | float32 | CORSIKA unit direction vector |
| `labels` | `(N,)` | int64 | Particle class integer (0 or 1) |
| `num_points` | `(N, 24)` | int32 | From Stage 1 output |

**Output**: shower point cloud — `(N, 2048, 5)` float32

| Column | Feature |
|---|---|
| 0 | x position |
| 1 | y position |
| 2 | layer index (0–23) |
| 3 | hit energy |
| 4 | hit time |

Rows beyond the real hit count for each shower are zero-padded.
Valid hits = `sum(num_points_per_layer[i])` for shower `i`.

---

## Running Generation — Single Combined Script

### Default run (uses local checkpoints):

```bash
cd /n/home05/zdimitrov/tambo/TAMBO-opt/allshowers

python generate_showers.py \
    --num-samples 10 \
    --output my_showers.h5 \
    --num-timesteps 16 \
    --device cuda:0
```

Output is written to the **current working directory** as `my_showers.h5`.

### All CLI options:

```
usage: generate_showers.py [-h] [-n NUM_SAMPLES] [-o OUTPUT]
                           [--point-count-model PATH]
                           [--allshowers-run-dir PATH]
                           [--num-timesteps N] [--batch-size N]
                           [--solver {heun,midpoint}]
                           [--device DEVICE] [--seed SEED]
                           [--e-min FLOAT] [--e-max FLOAT]
                           [--zenith-min FLOAT] [--zenith-max FLOAT]
```

| Flag | Default | Description |
|---|---|---|
| `-n / --num-samples` | 100 | Number of showers to generate |
| `-o / --output` | `generated_showers.h5` | Output HDF5 file path |
| `--point-count-model` | `checkpoints/compiled/` | PointCountFM TorchScript model dir |
| `--allshowers-run-dir` | `checkpoints/all_showers/` | AllShowers run dir |
| `--num-timesteps` | 16 | ODE integration steps (16 = fast, 200 = accurate) |
| `--batch-size` | 128 | AllShowers batch size |
| `--solver` | `midpoint` | ODE solver: `midpoint` or `heun` |
| `--device` | auto | Compute device (e.g. `cuda:0`, `cpu`) |
| `--seed` | None | Random seed for reproducibility |
| `--e-min` | 1e5 | Minimum primary energy [GeV] |
| `--e-max` | 1e8 | Maximum primary energy [GeV] |
| `--zenith-min` | 60 | Minimum zenith angle [degrees] |
| `--zenith-max` | 100 | Maximum zenith angle [degrees] |

---

## Output File Format

The output `generated_showers.h5` is saved via `showerdata.Showers` and contains:

| Array | Shape | Description |
|---|---|---|
| `showers` | variable-length | Raw per-hit data (x, y, layer, E, t) |
| `energies` | `(N, 1)` | Primary particle energies |
| `directions` | `(N, 3)` | 3D direction unit vectors |
| `pdg` | `(N,)` | Particle labels (0 or 1) |
| `observables/energy_per_layer` | `(N, 24)` | Total energy deposited per layer |
| `observables/energy_per_radial_bin` | `(N, 200)` | Radial energy profile |
| `observables/center_of_energy` | `(N, 3)` | Energy-weighted shower centre |

This file can be fed directly into `plot.ipynb` for comparison against CORSIKA.

---

## Running Generation from a Specific Training Run

If you want to use a model from a specific training run (e.g. the latest run
from hhanif's results) rather than the local checkpoint:

```bash
python generate_showers.py \
    --num-samples 1000 \
    --allshowers-run-dir /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/AllShowers/results/20260307_101304_CNF-Transformer \
    --output my_showers.h5 \
    --device cuda:0
```

> **Note on `best.pt` vs `last.pt`**: `generate_showers.py` loads
> `weights/best.pt` (lowest validation loss epoch). To generate from the
> absolute last training epoch, first extract the weights:
> ```python
> import torch
> ckpt = torch.load("checkpoints/all_showers/checkpoints/last.pt", weights_only=True)
> torch.save(ckpt["flow"], "checkpoints/all_showers/weights/best.pt")
> ```

---

## Architecture Summary

| Component | Details |
|---|---|
| Stage 1 model | TorchScript compiled flow matching model (PointCountFM) |
| Stage 2 model | CNF-Transformer (`dim_inputs=[4,6,4]`, 24 layers, 4 heads, 4 blocks) |
| Stage 2 features | x, y, energy, time per hit (`dim_inputs[0]=4`) |
| Condition dim | 6 (Fourier time embedding) + 4 (log-energy + 3D direction) |
| Particle classes | 2 (labels 0 and 1) |
| Max hits per shower | 2048 (from `max_num_points` in conf) |
| Preprocessing | Log + StandardScaler for energy and time; StandardScaler for coordinates |
