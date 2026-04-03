#!/usr/bin/env python3
"""
mkresultdir_pointcloudfm.py

Creates a result directory for the pointcloud FM trainer, writes:
  - conf.yaml   (copied params for this run)
  - run.sh      (sbatch script)
  - script.sh   (worker script run directly via bash)

Single-GPU trainer — no srun, no distributed setup needed.

SLURM partitions:
  -p gpu                          (A100)
  -p gpu_h200                     (H200)
  -p arguelles_delgado_gpu_mixed  (A100 80GB GRES)

Example:
python /n/home04/hhanif/AllShowers/mkresultdir_pointcloudfm.py /n/home04/hhanif/AllShowers/conf/pointcloudfm.yaml \
  -p gpu --mem 50G --cpus-per-task 2 --time 02:00:00 -r

python /n/home04/hhanif/AllShowers/mkresultdir_pointcloudfm.py /n/home04/hhanif/AllShowers/conf/pointcloudfm.yaml \
  -p arguelles_delgado_gpu_mixed --mem 100G --cpus-per-task 2 --time 00:20:00 -r

python /n/home04/hhanif/AllShowers/mkresultdir_pointcloudfm.py /n/home04/hhanif/AllShowers/conf/pointcloudfm.yaml \
  -p gpu_requeue --mem 200G --cpus-per-task 1 --time 24:00:00 -r

"""

import argparse
import os
from pathlib import Path

import yaml

from pointcountfm.trainer import setup_result_path


JOB_SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={name:s}
#SBATCH --mem={mem:s}
#SBATCH --cpus-per-task={cpus_per_task:d}
#SBATCH --time={time_limit:s}
#SBATCH -p {partition:s}
{gres_line:s}
#SBATCH --ntasks=1
#SBATCH --output={result_path:s}/log/train_%j.out
#SBATCH --error={result_path:s}/log/train_%j.err
{mail_lines:s}

echo "job id: $SLURM_JOB_ID"
echo "node list: $SLURM_JOB_NODELIST"
echo ""

bash {result_path:s}/script.sh
"""


WORKER_SCRIPT_TEMPLATE = """\
#!/bin/env bash
set -euo pipefail

cd {repo_path:s}

# ===== your cluster environment =====
module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate {mamba_env:s}
# ====================================

# Threading
num_cpus=$(nproc --all)
export OMP_NUM_THREADS=$num_cpus
if [ "$OMP_NUM_THREADS" -lt 1 ]; then
  export OMP_NUM_THREADS=1
fi

echo "node:        $(uname -n)"
echo "num CPUs:    $num_cpus"
nvidia-smi -L || true
grep MemTotal /proc/meminfo || true

echo ""
echo "config file: {config_rel:s}"
echo "start time: $(date)"
echo ""

python -m pointcountfm.trainer {config_rel:s}
"""


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create result directory + Slurm scripts for pointcloud FM trainer, optionally submit with sbatch."
    )
    p.add_argument("param_file", help="YAML parameter file (input).")
    p.add_argument(
        "-r", "--run", action="store_true",
        help="Submit the job via sbatch after creating scripts."
    )

    p.add_argument(
        "-p",
        "--partition",
        choices=["gpu", "gpu_requeue", "gpu_h200", "arguelles_delgado_gpu_mixed"],
        default="gpu",
        help='SLURM partition: "gpu" (A100), "gpu_h200" (H200), or "arguelles_delgado_gpu_mixed" (A100 80GB GRES).',
    )

    p.add_argument("--mem", type=str, default="300G", help='Memory request. Default: "300G"')
    p.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task. Default: 4")
    p.add_argument("--time", type=str, default="2-00:00:00", help='Time limit. Default: "2-00:00:00"')

    p.add_argument(
        "--mamba-env",
        type=str,
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/",
        help="Full path to the mamba environment to activate.",
    )

    p.add_argument(
        "--mail",
        type=str,
        default="",
        help="Email address for Slurm notifications (END,FAIL). Leave empty to disable.",
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = get_args()

    # Load params yaml
    with open(args.param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # Create result_path using pointcountfm's helper
    params["result_path"] = setup_result_path(params["run_name"], args.param_file)
    result_path = Path(params["result_path"])

    # Create subdirs
    for d in ["checkpoints", "weights", "plots", "log", "preprocessing", "data"]:
        ensure_dir(result_path / d)

    # Write conf.yaml into result dir
    conf_file = result_path / "conf.yaml"
    with open(conf_file, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

    # Paths for scripts
    run_file = result_path / "run.sh"
    worker_file = result_path / "script.sh"

    repo_path = Path(__file__).resolve().parent
    config_rel = os.path.relpath(str(conf_file), str(repo_path))

    # Mail lines
    mail_lines = ""
    if args.mail.strip():
        mail_lines = (
            "#SBATCH --mail-type=END,FAIL\n"
            f"#SBATCH --mail-user={args.mail.strip()}\n"
        )

    # GRES line: special case for the mixed partition
    if args.partition == "arguelles_delgado_gpu_mixed":
        gres_line = "#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1"
    else:
        gres_line = "#SBATCH --gres=gpu:1"

    # Write run.sh (sbatch)
    job_script = JOB_SCRIPT_TEMPLATE.format(
        name=params.get("run_name", "pointcloudfm"),
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        time_limit=args.time,
        partition=args.partition,
        gres_line=gres_line,
        result_path=str(result_path),
        mail_lines=mail_lines.rstrip("\n"),
    )

    with open(run_file, "w") as f:
        f.write(job_script + "\n")
    os.chmod(run_file, 0o750)

    # Write script.sh (worker)
    worker_script = WORKER_SCRIPT_TEMPLATE.format(
        repo_path=str(repo_path),
        mamba_env=args.mamba_env,
        config_rel=config_rel,
    )

    with open(worker_file, "w") as f:
        f.write(worker_script + "\n")
    os.chmod(worker_file, 0o750)

    # Print and optionally submit
    cmd = f"sbatch {run_file}"
    print(cmd)
    if args.run:
        print(os.popen(cmd).read())


if __name__ == "__main__":
    main()
