#!/usr/bin/env python3
"""
Two-stage generation pipeline
==============================
Stage 1 — PointCountFM  : random conditions → num_points_per_layer
Stage 2 — AllShowers    : (conditions + num_points_per_layer) → full shower point clouds

Usage
-----
python generate_pipeline.py \
    --pcfm-run-dir  /n/home04/hhanif/TAMBO-opt/results/20260416_050139_Electron-PointCountFM \
    --as-run-dir    /n/home04/hhanif/AllShowers/results/20260402_150113_CNF-Transformer \
    --num-samples   10000 \
    --pdg-codes     0 1 \
    --device        cuda:0

Output is saved to <as-run-dir>/samples_pipeline_NN.h5 (auto-incremented).
"""

import argparse
import math
import os
import platform
import sys
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor

# ── PointCountFM imports ──────────────────────────────────────────────────────
from pointcountfm import flow_matching as pcfm_fm
from pointcountfm import models as pcfm_models
from pointcountfm.preprocessing import Transformation as PCFMTransformation

# ── AllShowers imports ────────────────────────────────────────────────────────
import showerdata
from allshowers import flow_matching as as_fm
from allshowers import transformer
from allshowers.data_sets import to_label_tensor
from allshowers.preprocessing import compose as as_compose

pipeline_start = time.perf_counter()


# ══════════════════════════════════════════════════════════════════════════════
# Physics helpers
# ══════════════════════════════════════════════════════════════════════════════

E_MIN        = 1e5    # GeV
E_MAX        = 1e8    # GeV
ZENITH_MIN   = 60.0   # degrees
ZENITH_MAX   = 100.0  # degrees
AZIMUTH_MIN  = 0.0    # degrees
AZIMUTH_MAX  = 360.0  # degrees


def _deg_to_rad(angle: float) -> float:
    if math.isfinite(angle) and abs(angle) > 2 * math.pi + 1e-6:
        return math.radians(angle)
    return angle


def build_direction_vector(zenith: float, azimuth: float) -> np.ndarray:
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
) -> tuple[Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    conditions : Tensor  [N, 1 + num_classes + 3]  (float32, for PointCountFM)
    energies   : ndarray [N, 1]   GeV
    labels     : ndarray [N]      class index
    directions : ndarray [N, 3]   unit vectors
    """
    num_classes = len(pdg_codes)
    log_e      = rng.uniform(math.log10(e_min), math.log10(e_max), size=(num_samples, 1))
    energies   = (10.0 ** log_e).astype(np.float32)
    labels     = rng.integers(0, num_classes, size=num_samples)
    zeniths    = rng.uniform(zenith_min,  zenith_max,  size=num_samples)
    azimuths   = rng.uniform(azimuth_min, azimuth_max, size=num_samples)
    directions = np.stack(
        [build_direction_vector(z, a) for z, a in zip(zeniths, azimuths)], axis=0
    )
    labels_t   = torch.from_numpy(labels.astype(np.int64))
    conditions = torch.cat(
        (
            torch.from_numpy(energies),
            torch.nn.functional.one_hot(labels_t, num_classes=num_classes).float(),
            torch.from_numpy(directions),
        ),
        dim=1,
    )
    return conditions, energies, labels, directions


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — PointCountFM
# ══════════════════════════════════════════════════════════════════════════════

def _pcfm_build_model(config: dict) -> pcfm_fm.CNF:
    model_config = config["model"].copy()
    flow_config  = model_config.pop("flow") if "flow" in model_config else {}
    model_name   = model_config.pop("name")
    model_class  = getattr(pcfm_models, model_name)
    return pcfm_fm.CNF(model_class(**model_config), **flow_config)


class PCFMSampler(nn.Module):
    def __init__(
        self,
        model: pcfm_fm.CNF,
        transform_inc: PCFMTransformation,
        transform_num_points: PCFMTransformation,
        dim_data: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform_inc = transform_inc
        self.transform_num_points = transform_num_points
        self.dim_data = dim_data
        self.steps = steps

    def forward(self, condition: Tensor) -> Tensor:
        condition = torch.clone(condition)
        condition[:, :1] = self.transform_inc(condition[:, :1])
        samples = self.model.sample(
            (condition.shape[0], self.dim_data), self.steps, condition=condition
        )
        return self.transform_num_points.inverse(samples)


def load_pcfm(run_dir: str, steps: int | None, pdg_codes: list[int]) -> PCFMSampler:
    config_path    = os.path.join(run_dir, "conf.yaml")
    best_ckpt      = os.path.join(run_dir, "weights", "best.pt")
    last_ckpt      = os.path.join(run_dir, "weights", "last.pt")
    checkpoint_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
    trafos_path    = os.path.join(run_dir, "preprocessing", "trafos.pt")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if steps is None:
        steps = config["training"].get("steps", 50)

    print(f"  [PCFM] Building model...")
    model = _pcfm_build_model(config)
    print(f"  [PCFM] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval().to(torch.float32)

    print(f"  [PCFM] Loading trafos: {trafos_path}")
    trafos = torch.load(trafos_path, map_location="cpu", weights_only=False)
    transform_inc         = trafos["transform_inc"].to(torch.float32)
    transform_num_points  = trafos["transform_num_points"].to(torch.float32)

    network = model.network
    dim_data: int = (
        network.output.out_features if hasattr(network, "output")
        else network.network[-1].out_features
    )

    sampler = PCFMSampler(
        model=model,
        transform_inc=transform_inc,
        transform_num_points=transform_num_points,
        dim_data=dim_data,
        steps=steps,
    )

    # Trace on a single example
    example = torch.cat(
        (
            torch.tensor([[50.0]]),
            torch.nn.functional.one_hot(
                torch.tensor([0]), num_classes=len(pdg_codes)
            ).float(),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ),
        dim=1,
    )
    print(f"  [PCFM] Tracing + warming up...")
    with torch.inference_mode():
        sampler_traced = torch.jit.trace(sampler, example, check_trace=False)
        sampler_traced(example)
        sampler_traced(example)
    return sampler_traced


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — AllShowers Generator (lifted from generator.py)
# ══════════════════════════════════════════════════════════════════════════════

class AllShowersGenerator(nn.Module):
    def __init__(
        self,
        run_dir: str,
        num_timesteps: int = 200,
        do_compile: bool = False,
        solver: str = "heun",
        resize_factor: float = 1.0,
    ) -> None:
        super().__init__()
        run_params_file = os.path.join(run_dir, "conf.yaml")
        state_dict_file = os.path.join(run_dir, "weights/best.pt")
        if not os.path.exists(state_dict_file):
            state_dict_file = os.path.join(run_dir, "weights/best-all.pt")
        trafo_file = os.path.join(run_dir, "preprocessing/trafos.pt")
        if not os.path.exists(trafo_file):
            trafo_file = os.path.join(run_dir, "preprocessing/trafos-all.pt")

        self.num_timesteps = num_timesteps
        self.do_compile    = do_compile
        self.resize_factor = resize_factor

        with open(run_params_file) as f:
            run_params = yaml.load(f, Loader=yaml.FullLoader)

        self._init_model(run_params["model"], state_dict_file, solver=solver)
        self._init_trafo(run_params["data"], trafo_file)
        self.to(torch.get_default_dtype())

        self.feature_last   = run_params["data"].get("feature_last", False)
        self.num_layers     = run_params["model"].get("num_layers", None)
        self.max_points     = run_params["data"].get("max_num_points", 6016)
        self.expects_angles = run_params["model"]["dim_inputs"][-1] > 1
        self.with_time      = run_params["model"]["dim_inputs"][0] == 4

    def _init_model(self, params: dict[str, Any], state_file: str, solver: str) -> None:
        flow_config = params.pop("flow_config") if "flow_config" in params else {}
        flow_config["solver"] = solver
        network    = transformer.Transformer(**params)
        state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        trained_compiled = any("_orig_mod." in k for k in state_dict)
        if trained_compiled and not self.do_compile:
            for k in list(state_dict):
                state_dict[k.replace("_orig_mod.", "")] = state_dict.pop(k)
        elif not trained_compiled and self.do_compile:
            for k in list(state_dict):
                if "network." in k:
                    state_dict[k.replace("network.", "network._orig_mod.")] = state_dict.pop(k)
        if self.do_compile:
            network = torch.compile(network)
        self.flow = as_fm.CNF(network, **flow_config)
        self.flow.load_state_dict(state_dict)

    def _init_trafo(self, params: dict[str, Any], trafo_file: str) -> None:
        self.samples_energy_trafo      = as_compose(params.get("samples_energy_trafo"))
        self.samples_coordinate_trafo  = as_compose(params.get("samples_coordinate_trafo"))
        self.cond_trafo                = as_compose(params.get("cond_trafo"))
        self.samples_time_trafo = (
            as_compose(params.get("samples_time_trafo"))
            if params.get("samples_time_trafo") is not None
            else None
        )
        state = torch.load(trafo_file, map_location="cpu", weights_only=True)
        self.samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(state["cond_trafo"])
        if self.samples_time_trafo is not None and "samples_time_trafo" in state:
            self.samples_time_trafo.load_state_dict(state["samples_time_trafo"])

    def forward(
        self,
        energies: Tensor,
        num_points: Tensor,
        angles: Tensor,
        label: Tensor | None = None,
    ) -> Tensor:
        if self.expects_angles:
            condition = torch.cat(
                [self.cond_trafo(energies * self.resize_factor), angles], dim=-1
            )
        else:
            condition = self.cond_trafo(energies)

        layer = torch.zeros((condition.shape[0], self.max_points, 1), dtype=torch.int32)
        mask  = torch.zeros((condition.shape[0], self.max_points, 1), dtype=torch.bool)
        for i in range(condition.shape[0]):
            total_points = torch.sum(num_points[i])
            layer_i = torch.repeat_interleave(num_points[i])
            if total_points > self.max_points:
                warnings.warn(
                    f"num points {total_points} exceeds max points {self.max_points}, truncating"
                )
                total_points = self.max_points
                layer_i = layer_i[: self.max_points]
            layer[i, :total_points, 0] = layer_i
            mask[i,  :total_points, 0] = True
        layer = layer.to(condition.device)
        mask  = mask.to(condition.device)

        if self.with_time:
            raw = self.flow.sample(
                shape=(condition.shape[0], self.max_points, 4),
                num_timesteps=self.num_timesteps,
                cond=condition,
                num_points=num_points,
                layer=layer,
                mask=mask,
                label=label,
            )
            out = torch.zeros((condition.shape[0], self.max_points, 5), device=raw.device)
            out[:, :, :2] = self.samples_coordinate_trafo.inverse(raw[:, :, :2])
            out[:, :, 2]  = layer.squeeze(2)
            out[:, :, 3]  = self.samples_energy_trafo.inverse(raw[:, :, 2])
            out[:, :, 4]  = self.samples_time_trafo.inverse(raw[:, :, 3])
            out[~mask.repeat(1, 1, 5)] = 0
        else:
            raw = self.flow.sample(
                shape=(condition.shape[0], self.max_points, 3),
                num_timesteps=self.num_timesteps,
                cond=condition,
                num_points=num_points,
                layer=layer,
                mask=mask,
                label=label,
            )
            out = torch.zeros((condition.shape[0], self.max_points, 4), device=raw.device)
            out[:, :, :2] = self.samples_coordinate_trafo.inverse(raw[:, :, :2])
            out[:, :, 2]  = layer.squeeze(2)
            out[:, :, 3]  = self.samples_energy_trafo.inverse(raw[:, :, 2])
            out[~mask.repeat(1, 1, 4)] = 0
        return out


def run_allshowers(
    generator: AllShowersGenerator,
    energies: Tensor,
    num_points: Tensor,
    angles: Tensor,
    labels: Tensor | None,
    batch_size: int,
    device: str | torch.device,
) -> Tensor:
    generator = generator.to(device).eval()
    split_e   = torch.split(energies,    batch_size)
    split_np  = torch.split(num_points,  batch_size)
    split_ang = torch.split(angles,      batch_size)
    split_lbl = torch.split(labels, batch_size) if labels is not None else [None] * len(split_e)

    samples = []
    for i, (e, np_, ang, lbl) in enumerate(zip(split_e, split_np, split_ang, split_lbl)):
        elapsed = int(time.perf_counter() - pipeline_start)
        print(f"  [{elapsed:6d}s] [AllShowers] batch {i:3d} / {len(split_e)}")
        sys.stdout.flush()
        e, np_, ang = e.to(device), np_.to(device), ang.to(device)
        if lbl is not None:
            lbl = lbl.to(device)
        samples.append(generator(e, np_, ang, lbl).cpu())
    return torch.cat(samples)


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def next_output_path(run_dir: str) -> str:
    i = 0
    while True:
        path = os.path.join(run_dir, f"samples_pipeline_{i:02d}.h5")
        if not os.path.exists(path):
            return path
        i += 1


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PointCountFM → AllShowers two-stage pipeline with on-the-fly conditions."
    )
    p.add_argument("--pcfm-run-dir", required=True,
                   help="PointCountFM result directory (conf.yaml / weights / preprocessing).")
    p.add_argument("--as-run-dir",   required=True,
                   help="AllShowers result directory. Output is also saved here.")
    p.add_argument("--num-samples",  type=int, required=True,
                   help="Number of showers to generate.")
    p.add_argument("--pdg-codes",    type=int, nargs="+", default=[0, 1],
                   help="Ordered PDG class labels (default: 0 1).")
    p.add_argument("--e-min",        type=float, default=E_MIN)
    p.add_argument("--e-max",        type=float, default=E_MAX)
    p.add_argument("--zenith-min",   type=float, default=ZENITH_MIN)
    p.add_argument("--zenith-max",   type=float, default=ZENITH_MAX)
    p.add_argument("--azimuth-min",  type=float, default=AZIMUTH_MIN)
    p.add_argument("--azimuth-max",  type=float, default=AZIMUTH_MAX)
    p.add_argument("--pcfm-steps",   type=int, default=None,
                   help="PointCountFM ODE steps (default: from training config).")
    p.add_argument("--num-timesteps", type=int, default=200,
                   help="AllShowers ODE timesteps (default: 200).")
    p.add_argument("--batch-size",   type=int, default=64,
                   help="Batch size for AllShowers inference (default: 64).")
    p.add_argument("--solver",       type=str, default="heun",
                   help="AllShowers ODE solver (default: heun).")
    p.add_argument("--rescale-factor", type=float, default=1.0,
                   help="Energy rescale factor for AllShowers (default: 1.0).")
    p.add_argument("--device",       type=str, default=None,
                   help="Compute device (default: auto-detect).")
    p.add_argument("--dtype",        type=str, default="float32",
                   choices=["float16", "float32", "float64"])
    p.add_argument("--seed",         type=int, default=None,
                   help="Random seed for condition sampling.")
    p.add_argument("-o", "--output", type=str, default="",
                   help="Explicit output path (default: <as-run-dir>/samples_pipeline_NN.h5).")
    return p.parse_args(args)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # ── Device / dtype ────────────────────────────────────────────────────────
    dtypes = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
    dtype  = dtypes[args.dtype]
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if "cuda" in device.lower():
        print("device:", torch.cuda.get_device_name(torch.device(device)))
    else:
        print("device:", platform.processor())

    # ── Output path ───────────────────────────────────────────────────────────
    output_file = args.output if args.output else next_output_path(args.as_run_dir)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Output will be written to: {output_file}\n")

    # ── Stage 1: load PointCountFM ────────────────────────────────────────────
    t0 = time.perf_counter()
    print("=" * 60)
    print("Loading PointCountFM (Stage 1)...")
    print("=" * 60)
    pcfm_sampler = load_pcfm(args.pcfm_run_dir, args.pcfm_steps, args.pdg_codes)
    print(f"  Ready in {(time.perf_counter() - t0):.1f}s\n")

    # ── Stage 2: load AllShowers ──────────────────────────────────────────────
    t0 = time.perf_counter()
    print("=" * 60)
    print("Loading AllShowers (Stage 2)...")
    print("=" * 60)
    as_generator = AllShowersGenerator(
        run_dir       = args.as_run_dir,
        num_timesteps = args.num_timesteps,
        do_compile    = False,  # compile causes OOM on small GPUs; enable manually if needed
        solver        = args.solver,
        resize_factor = args.rescale_factor,
    )
    print(f"  Time mode: {'ON (x,y,e,t)' if as_generator.with_time else 'OFF (x,y,e)'}")
    print(f"  Ready in {(time.perf_counter() - t0):.1f}s\n")

    # ── Sample random conditions ──────────────────────────────────────────────
    print("=" * 60)
    print(f"Sampling {args.num_samples} random conditions...")
    print("=" * 60)
    rng = np.random.default_rng(args.seed)
    conditions, energies, labels_np, directions = sample_conditions(
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
    energies_t   = torch.from_numpy(energies).to(dtype)       # [N, 1]
    directions_t = torch.from_numpy(directions).to(dtype)     # [N, 3]
    labels_t  = torch.from_numpy(labels_np.astype(np.int64))  # [N]
    # to_label_tensor maps class indices → the label tensor AllShowers expects ([N] int64)
    as_labels = to_label_tensor(pdg=labels_t, label_list=list(range(len(args.pdg_codes))))
    print(f"  energies:   [{energies.min():.2e}, {energies.max():.2e}] GeV")
    print(f"  labels:     {np.bincount(labels_np).tolist()}  (counts per class)\n")

    # ── Stage 1: PointCountFM → num_points_per_layer ─────────────────────────
    print("=" * 60)
    print("Stage 1 — PointCountFM inference...")
    print("=" * 60)
    t0 = time.perf_counter()
    pcfm_results = pcfm_sampler(conditions)
    num_points = (torch.clamp(pcfm_results, min=0.0) + 0.5).to(torch.int32)  # [N, num_layers]
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.2f}s  ({dt / args.num_samples * 1000:.2f} ms/sample)")
    print(f"  num_points shape: {num_points.shape}")
    print(f"  total hits/shower (mean): {num_points.sum(dim=1).float().mean():.1f}\n")

    # ── Stage 2: AllShowers → full shower point clouds ───────────────────────
    print("=" * 60)
    print("Stage 2 — AllShowers inference...")
    print("=" * 60)
    t0 = time.perf_counter()
    samples = run_allshowers(
        generator  = as_generator,
        energies   = energies_t,
        num_points = num_points,
        angles     = directions_t,
        labels     = as_labels,
        batch_size = args.batch_size,
        device     = device,
    )
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.2f}s  ({dt / args.num_samples * 1000:.2f} ms/sample)\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"Saving to {output_file} ...")
    showers = showerdata.Showers(
        points     = samples.numpy(),
        energies   = energies,
        directions = directions,
        pdg        = labels_np.astype(np.int32),
    )
    showers.save(output_file)

    # Also save num_points_per_layer into the same file
    showerdata.observables.save_observables_to_file(
        output_file,
        {"num_points_per_layer": num_points.numpy()},
        overwrite=True,
    )

    # Save a matching YAML with all run parameters
    yaml_path = output_file.replace(".h5", ".yaml")
    meta = vars(args).copy()
    meta["pcfm_steps_used"] = int(pcfm_sampler.steps) if hasattr(pcfm_sampler, "steps") else None
    with open(yaml_path, "w") as f:
        yaml.dump(meta, f)

    total = int(time.perf_counter() - pipeline_start)
    print(f"Saved to {output_file}")
    print(f"Done. Total wall time: {total}s")


if __name__ == "__main__":
    main()