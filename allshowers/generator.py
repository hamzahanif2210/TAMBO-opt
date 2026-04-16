'''

python /n/home04/hhanif/AllShowers/allshowers/generator.py \
  --run-dir /n/home04/hhanif/AllShowers/results/20260305_141348_CNF-Transformer \
  --num-samples 1000 \
  --num-timesteps 16 \
  --device cuda:0 \
  --solver midpoint \
  --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v5/merged_all_showers_test_data_with_num_points.h5 \
    --pdgs 11 211 -11 111 -211

# With time (model trained with samples_time_trafo in config):
python /n/home04/hhanif/AllShowers/allshowers/generator.py \
  --run-dir /n/home04/hhanif/AllShowers/results/20260402_150113_CNF-Transformer \
  --num-samples 10000 \
  --num-timesteps 16 \
  --device cuda:0 \
  --solver midpoint \
  --cond_file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced-test-file_data_with_num_points.h5 \
    --pdgs 0 1

'''

import argparse
import gc
import os
import platform
import sys
import time
import warnings
from typing import Any

# Add parent directory to path so 'allshowers' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import showerdata
import torch
import yaml
from torch import Tensor, nn

from allshowers import flow_matching as fm
from allshowers import transformer
from allshowers.data_sets import to_label_tensor
from allshowers.preprocessing import compose

start = time.perf_counter()


class Generator(nn.Module):
    def __init__(
        self,
        run_dir: str,
        num_timesteps: int = 200,
        compile: bool = False,
        solver: str = "heun",
        resize_factor: float = 1.0,
    ) -> None:
        super().__init__()

        run_params_file = os.path.join(run_dir, "conf.yaml")
        state_dict_file = os.path.join(run_dir, "weights/best.pt")
        if not os.path.exists(run_params_file):
            state_dict_file = os.path.join(run_dir, "weights/best-all.pt")
        trafo_file = os.path.join(run_dir, "preprocessing/trafos.pt")
        if not os.path.exists(trafo_file):
            trafo_file = os.path.join(run_dir, "preprocessing/trafos-all.pt")
        self.result_dir = run_dir
        self.num_timesteps = num_timesteps
        self.do_compile = compile
        self.resize_factor = resize_factor

        with open(run_params_file) as f:
            run_params = yaml.load(f, Loader=yaml.FullLoader)

        self.__init_model(run_params["model"], state_dict_file, solver=solver)
        self.__init_trafo(run_params["data"], trafo_file)
        self.to(torch.get_default_dtype())
        self.feature_last = run_params["data"].get("feature_last", False)
        self.num_layers = run_params["model"].get("num_layers", None)
        self.max_points = run_params["data"].get("max_num_points", 6016)
        self.expects_angles = run_params["model"]["dim_inputs"][-1] > 1

        # Auto-detect time mode from config — no CLI flag needed.
        # If the model was trained with samples_time_trafo, dim_inputs[0] == 4.
        self.with_time = run_params["model"]["dim_inputs"][0] == 4

    def __init_model(
        self, params: dict[str, Any], state_file: str, solver: str = "heun"
    ) -> None:
        flow_config = params.pop("flow_config") if "flow_config" in params else {}
        flow_config["solver"] = solver
        network = transformer.Transformer(**params)
        state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        trained_compiled = any("_orig_mod." in key for key in state_dict)
        if trained_compiled and not self.do_compile:
            for k in list(state_dict.keys()):
                if "_orig_mod." in k:
                    new_k = k.replace("_orig_mod.", "")
                    state_dict[new_k] = state_dict.pop(k)
        elif not trained_compiled and self.do_compile:
            for k in list(state_dict.keys()):
                if "network." in k:
                    new_k = k.replace("network.", "network._orig_mod.")
                    state_dict[new_k] = state_dict.pop(k)
        if self.do_compile:
            network = torch.compile(network)
        self.flow = fm.CNF(network, **flow_config)  # type: ignore
        self.flow.load_state_dict(state_dict)

    def __init_trafo(self, params: dict[str, Any], trafo_file: str) -> None:
        self.samples_energy_trafo = compose(params.get("samples_energy_trafo"))
        self.samples_coordinate_trafo = compose(params.get("samples_coordinate_trafo"))
        self.cond_trafo = compose(params.get("cond_trafo"))

        # Time trafo — only present when model was trained with time
        if params.get("samples_time_trafo") is not None:
            self.samples_time_trafo = compose(params.get("samples_time_trafo"))
        else:
            self.samples_time_trafo = None

        state = torch.load(trafo_file, map_location="cpu", weights_only=True)
        self.samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(state["cond_trafo"])

        # Load time trafo state if saved in the trafos file
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
            condition = torch.concatenate(
                [self.cond_trafo(energies * self.resize_factor), angles], dim=-1
            )
        else:
            condition = self.cond_trafo(energies)

        layer = torch.zeros((condition.shape[0], self.max_points, 1), dtype=torch.int32)
        mask = torch.zeros((condition.shape[0], self.max_points, 1), dtype=torch.bool)
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
            mask[i, :total_points, 0] = True
        layer = layer.to(condition.device)
        mask = mask.to(condition.device)

        if self.with_time:
            # Sample 4 features: x, y, e, t
            raw_samples = self.flow.sample(
                shape=(condition.shape[0], self.max_points, 4),
                num_timesteps=self.num_timesteps,
                cond=condition,
                num_points=num_points,
                layer=layer,
                mask=mask,
                label=label,
            )
            # Reconstruct 5-column output: x, y, z(layer), e, t
            samples = torch.zeros(
                (condition.shape[0], self.max_points, 5), device=raw_samples.device
            )
            samples[:, :, :2] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :2])
            samples[:, :, 2]  = layer.squeeze(2)
            samples[:, :, 3]  = self.samples_energy_trafo.inverse(raw_samples[:, :, 2])
            samples[:, :, 4]  = self.samples_time_trafo.inverse(raw_samples[:, :, 3])
            samples[~mask.repeat(1, 1, 5)] = 0
        else:
            # Original: sample 3 features: x, y, e
            raw_samples = self.flow.sample(
                shape=(condition.shape[0], self.max_points, 3),
                num_timesteps=self.num_timesteps,
                cond=condition,
                num_points=num_points,
                layer=layer,
                mask=mask,
                label=label,
            )
            # Reconstruct 4-column output: x, y, z(layer), e
            samples = torch.zeros(
                (condition.shape[0], self.max_points, 4), device=raw_samples.device
            )
            samples[:, :, :2] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :2])
            samples[:, :, 2]  = layer.squeeze(2)
            samples[:, :, 3]  = self.samples_energy_trafo.inverse(raw_samples[:, :, 2])
            samples[~mask.repeat(1, 1, 4)] = 0

        return samples


def print_time(text):
    now = time.perf_counter()
    print(f"[{int(now - start):6d}s]: {text}")
    sys.stdout.flush()


def generate(
    generator: Generator,
    energies: Tensor,
    num_points: Tensor,
    angles: Tensor,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
    labels: Tensor | None = None,
) -> Tensor:
    if batch_size is None:
        batch_size = energies.shape[0]
    split_energies = torch.split(energies, batch_size, dim=0)
    split_num_points = torch.split(num_points, batch_size, dim=0)
    split_angles = torch.split(angles, batch_size, dim=0)
    if labels is not None:
        split_labels = torch.split(labels, batch_size, dim=0)
    else:
        split_labels = [None] * len(split_energies)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for i, batch in enumerate(
        zip(split_energies, split_num_points, split_angles, split_labels)
    ):
        print_time(f"start batch {i:3d}")
        batch = [e.to(device) if e is not None else None for e in batch]
        samples_l = generator(*batch).detach().cpu()
        samples.append(samples_l)
        
    samples = torch.cat(samples)
    print_time("generation done")
    return samples


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new samples")
    parser.add_argument(
        "--run-dir",
        help="directory that contains the model's weights and where the generated samples should be saved",
    )
    parser.add_argument(
        "--cond_file",
        help="file with the conditioning information (e.g. energies, number of points)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=1,
        type=int,
        help="number of samples to generate. default: 1",
    )
    parser.add_argument(
        "-b", "--batch-size", default=1024, type=int, help="default: 1024"
    )
    parser.add_argument("-t", "--num-threads", default=None, type=int)
    parser.add_argument("-d", "--device", default=None, help="device for computations")
    parser.add_argument(
        "--num-timesteps",
        default=200,
        type=int,
        help="number of timesteps for the ODE solver. default: 200",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="data type for the generated samples. default: float32",
    )
    parser.add_argument(
        "-r",
        "--rescale-factor",
        default=1.0,
        type=float,
        help="energy rescale factor applied during generation. default: 1.0",
    )
    parser.add_argument(
        "--solver",
        default="heun",
        type=str,
        help="ODE solver to use during generation. default: heun",
    )
    parser.add_argument(
        "--pdgs",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        nargs="+",
        type=int,
        help="list of pdg codes for the labels. default: [11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212]",
    )
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    parsed_args.pdgs.sort(key=lambda x: (abs(x), -x))
    print_time("start main")
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if parsed_args.dtype not in dtypes:
        raise ValueError(f"invalid dtype: {parsed_args.dtype}")
    dtype = dtypes[parsed_args.dtype]
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)
    print(yaml.dump(vars(parsed_args)), end="")
    if parsed_args.device:
        device = parsed_args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_default_device(device)
    if "cuda" in device.lower():
        print("device:", torch.cuda.get_device_name(torch.device(device)))
    elif device.lower() == "cpu":
        print("device:", platform.processor())
    print("num threads:", torch.get_num_threads())
    sys.stdout.flush()

    generator = Generator(
        run_dir=parsed_args.run_dir,
        num_timesteps=parsed_args.num_timesteps,
        compile=("cuda" in device.lower()),
        solver=parsed_args.solver,
        resize_factor=parsed_args.rescale_factor,
    )

    print_time(f"time mode: {'ON (x,y,e,t)' if generator.with_time else 'OFF (x,y,e)'}")

    cond_data = showerdata.observables.read_observables_from_file(
        parsed_args.cond_file,
        observables=[
            "incident_energies",
            "incident_pdg",
            "incident_directions",
            "num_points_per_layer_corsika",
        ],
        start=-parsed_args.num_samples,
    )
    energies = torch.from_numpy(cond_data["incident_energies"])
    num_points = torch.from_numpy(cond_data["num_points_per_layer_corsika"])
    angle = torch.from_numpy(cond_data["incident_directions"])
    pdg = torch.from_numpy(cond_data["incident_pdg"])
    labels = to_label_tensor(
        pdg=pdg,
        label_list=parsed_args.pdgs,
    )

    energies = energies.to(dtype, copy=False)

    generator.eval()
    generator = generator.to(device)

    samples = generate(
        generator,
        energies,
        num_points,
        angle,
        parsed_args.batch_size,
        device,
        labels,
    )
    showers = showerdata.Showers(
        points=samples.numpy(),
        energies=energies.numpy(),
        directions=angle.numpy(),
        pdg=pdg.numpy(),
    )

    for i in range(100):
        name = f"samples{i:02d}"
        file_path = os.path.join(parsed_args.run_dir, name + ".h5")
        if not os.path.exists(file_path):
            break
    else:
        raise RuntimeError("no free sample file name found")

    showers.save(file_path)
    with open(os.path.join(parsed_args.run_dir, name + ".yaml"), "w") as f:
        yaml.dump(vars(parsed_args), f)

    print(f"saved to {file_path}")
    print_time("all done")


if __name__ == "__main__":
    main()