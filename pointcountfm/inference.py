import argparse
import os
import time
from collections.abc import Iterable

import showerdata
import torch
import torch.nn as nn
import yaml

from pointcountfm import flow_matching, models
from pointcountfm.preprocessing import Transformation

'''
python /n/home04/hhanif/TAMBO-opt/pointcountfm/inference.py     /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced-test-file.h5     --run-dir /n/home04/hhanif/TAMBO-opt/results/20260416_050139_Electron-PointCountFM  --pdg-codes 0 1

# Expected directory layout under --run-dir:
#   conf.yaml
#   weights/best.pt   (or last.pt if best.pt is absent)
#   preprocessing/trafos.pt
#
# Output is written to --run-dir/output_00.h5, output_01.h5, ... (auto-incremented).
# Use -o / --output to override with an explicit path instead.
'''


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inference on a model")
    parser.add_argument("input_file", type=str, help="Input file with conditioning data")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Training result directory. Must contain conf.yaml, "
            "weights/best.pt (or weights/last.pt), and preprocessing/trafos.pt. "
            "Output is saved here as output_00.h5, output_01.h5, ... unless -o is given."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of ODE integration steps. Defaults to the value in the training config.",
    )
    parser.add_argument(
        "--pdg-codes",
        type=int,
        nargs="+",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        help="List of PDG codes corresponding to particle classes.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help=(
            "Explicit output file path. If omitted, output is written to "
            "<run-dir>/output_NN.h5 with NN auto-incremented."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Whether to overwrite the output file if it exists. "
            "If input and output files are the same, this flag "
            "determines whether to overwrite the dataset within the "
            "file if it exists."
        ),
    )
    return parser.parse_args(args)


def next_output_path(run_dir: str) -> str:
    """Return <run_dir>/output_NN.h5 where NN is the lowest free index."""
    i = 0
    while True:
        path = os.path.join(run_dir, f"output_{i:02d}.h5")
        if not os.path.exists(path):
            return path
        i += 1


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def build_model(config: dict) -> flow_matching.CNF:
    """Reconstruct a CNF model from a training config dict."""
    model_config = config["model"].copy()
    flow_config = model_config.pop("flow") if "flow" in model_config else {}
    model_name = model_config.pop("name")
    model_class = getattr(models, model_name)
    network = model_class(**model_config)
    return flow_matching.CNF(network, **flow_config)


class Sampler(nn.Module):
    """Mirrors the Sampler used in Trainer.compile() so inference is identical."""

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


def main():
    args = parse_args()
    torch.set_num_threads(1)

    # ------------------------------------------------------------------ paths
    run_dir = args.run_dir
    config_path = os.path.join(run_dir, "conf.yaml")
    best_ckpt = os.path.join(run_dir, "weights", "best.pt")
    last_ckpt = os.path.join(run_dir, "weights", "last.pt")
    checkpoint_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt
    trafos_path = os.path.join(run_dir, "preprocessing", "trafos.pt")

    # Resolve output path before doing any heavy work so we fail fast.
    if args.output:
        output_file = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    else:
        output_file = next_output_path(run_dir)
    print(f"Output will be written to: {output_file}")

    # ------------------------------------------------------------------ config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    steps = args.steps if args.steps is not None else config["training"].get("steps", 50)

    # ------------------------------------------------------------------ model
    print("Building model from config...")
    model = build_model(config)

    print(f"Loading checkpoint from {checkpoint_path}...")
    start = time.time()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(torch.float32)
    print(f"Checkpoint loaded in {(time.time() - start) * 1000.0:.1f}ms")

    # --------------------------------------------------------------- trafos
    print("Loading transformations...")
    trafos = torch.load(trafos_path, map_location="cpu", weights_only=False)
    transform_inc: Transformation = trafos["transform_inc"]
    transform_num_points: Transformation = trafos["transform_num_points"]
    transform_inc = transform_inc.to(torch.float32)
    transform_num_points = transform_num_points.to(torch.float32)

    # Infer dim_data from the model's output layer.
    # ConcatSquash has model.network.output; FullyConnected has model.network.network[-1].
    network = model.network
    if hasattr(network, "output"):          # ConcatSquash
        dim_data: int = network.output.out_features
    else:                                   # FullyConnected
        dim_data = network.network[-1].out_features

    # ----------------------------------------------------------------- sampler
    sampler = Sampler(
        model=model,
        transform_inc=transform_inc,
        transform_num_points=transform_num_points,
        dim_data=dim_data,
        steps=steps,
    )

    # Build example conditions for tracing / warm-up.
    example_conditions = torch.concatenate(
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
        sampler_traced = torch.jit.trace(sampler, example_conditions, check_trace=False)
    print(f"Inference function traced in {(time.time() - start) * 1000.0:.1f}ms")

    # --------------------------------------------------------------- warm-up
    print("Warming up...")
    start = time.time()
    with torch.inference_mode():
        sampler_traced(example_conditions)
        sampler_traced(example_conditions)
    print(f"Warmup done in {(time.time() - start) * 1000.0:.1f}ms")

    # ---------------------------------------------------------- load cond data
    print("Loading conditioning data...")
    start = time.time()
    input_len = showerdata.get_file_length(args.input_file)
    cond_data = showerdata.load_inc_particles(args.input_file, start=input_len - 50000)
    conditions = torch.concatenate(
        (
            torch.from_numpy(cond_data.energies).to(torch.float32),
            torch.nn.functional.one_hot(
                to_labels(torch.from_numpy(cond_data.pdg), args.pdg_codes),
                num_classes=len(args.pdg_codes),
            ).to(torch.float32),
            torch.from_numpy(cond_data.directions).to(torch.float32),
        ),
        dim=1,
    )
    print(f"Conditioning data loaded in {(time.time() - start) * 1000.0:.1f}ms")

    # ---------------------------------------------------------------- infer
    print("Running inference...")
    start = time.time()
    with torch.inference_mode():
        results = sampler_traced(conditions)
    elapsed = time.time() - start
    print(f"Inference done in {elapsed:.2f}s")
    print(f"Average inference time: {elapsed / len(conditions) * 1000.0:.1f}ms per sample")

    # ----------------------------------------------------------------- save
    print(f"Saving results to {output_file} ...")
    results = (torch.clamp(results, min=0.0) + 0.5).to(torch.int32)
    start = time.time()
    if output_file != args.input_file:
        showerdata.save(cond_data, output_file, overwrite=args.overwrite)
    showerdata.observables.save_observables_to_file(
        output_file,
        {"num_points_per_layer": results.numpy()},
        overwrite=args.overwrite,
    )
    print(f"Results saved in {(time.time() - start) * 1000.0:.1f}ms")


if __name__ == "__main__":
    main()