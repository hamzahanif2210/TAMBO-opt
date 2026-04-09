#!/usr/bin/env python3
"""
mkresultdir.py

Creates a result directory, writes:
  - conf.yaml   (copied params for this run)
  - run.sh      (sbatch script)
  - script.sh   (worker script run via srun)

Launch strategy (no torchrun / no c10d TCPStore):
  - srun launches ONE task PER GPU across all nodes
  - each task is one Python process bound to its GPU via SLURM_LOCALID
  - PyTorch distributed is initialized with init_method="env://"
    using MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK set from SLURM env vars
  - NO torchrun, NO c10d rendezvous server, NO open listening port needed
  - MASTER_ADDR is resolved to IPv4 to avoid errno-97 IPv6 issues

SLURM partitions:
  -p gpu                          (A100)
  -p gpu_h200                     (H200)
  -p arguelles_delgado_gpu_mixed  (A100 80GB GRES)

Example:
python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/transformer.yaml \
  -p gpu -g 1 -n 4 --mem 50G --cpus-per-task 2 --time 02:00:00 -r

python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/transformer.yaml \
  -p arguelles_delgado_gpu_mixed -g 1 -n 1 --mem 100G --cpus-per-task 2 --time 00:20:00 -r


python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/transformer_time.yaml \
  -p gpu_requeue -g 1 -n 4 --mem 200G --cpus-per-task 1 --time 24:00:00 -r

python /n/home04/hhanif/AllShowers/mkresultdir.py /n/home04/hhanif/AllShowers/conf/allshowers/allshowers_electrons.yaml\
  -p gpu_requeue -g 1 -n 1 --mem 200G --cpus-per-task 1 --time 24:00:00 -r

"""

import argparse
import os
from pathlib import Path

import yaml

from allshowers import util


JOB_SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={name:s}
#SBATCH --mem={mem:s}
#SBATCH --cpus-per-task={cpus_per_task:d}
#SBATCH --time={time_limit:s}
#SBATCH -p {partition:s}
{gres_line:s}
#SBATCH --nodes={num_nodes:d}
#SBATCH --constraint=a100
#SBATCH --output={result_path:s}/log/train_%j.out
#SBATCH --error={result_path:s}/log/train_%j.err
{mail_lines:s}

echo "job id: $SLURM_JOB_ID"
echo "node list: $SLURM_JOB_NODELIST"
echo ""

# Launch one task per GPU across all nodes.
# script.sh is a plain python process - no torchrun, no rendezvous port needed.
srun \
  --nodes={num_nodes:d} \
  --ntasks-per-node={num_gpus:d} \
  --kill-on-bad-exit=1 \
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

# ---------------------------------------------------------------------------
# Distributed setup via SLURM env vars - no torchrun / no c10d TCPStore.
# PyTorch uses init_method="env://" which reads these variables directly.
# This avoids any listening port / firewall issues entirely.
# ---------------------------------------------------------------------------

# Resolve master hostname to IPv4 to avoid errno-97 (AF not supported)
_MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(python3 -c "
import socket
print(socket.getaddrinfo('$_MASTER_HOSTNAME', None, socket.AF_INET)[0][4][0])
")

export MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 10000) ))
export WORLD_SIZE=$SLURM_NTASKS      # total processes = nodes * gpus_per_node
export RANK=$SLURM_PROCID            # global rank of this process
export LOCAL_RANK=$SLURM_LOCALID     # local rank within this node (= GPU index)

# Disable IPv6 everywhere just in case
export GLOO_USE_IPV6=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export GLOO_SOCKET_IFNAME=^lo,docker0

# Threading
num_cpus=$(nproc --all)
num_gpus=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=$(( num_cpus / num_gpus ))
if [ "$OMP_NUM_THREADS" -lt 1 ]; then
  export OMP_NUM_THREADS=1
fi

echo "node:        $(uname -n)"
echo "rank:        $RANK / $WORLD_SIZE"
echo "local_rank:  $LOCAL_RANK"
echo "master:      $MASTER_ADDR:$MASTER_PORT  (resolved from $_MASTER_HOSTNAME)"
echo "num CPUs:    $num_cpus"
echo "num GPUs:    $num_gpus"
grep MemTotal /proc/meminfo || true

echo ""
echo "config file: {config_rel:s}"
echo "start time: $(date)"
echo ""

python allshowers/train.py --ddp {config_rel:s}
"""


# ---------------------------------------------------------------------------
# NOTE for train.py:
#   Replace any torchrun-style init with:
#
#     import os, torch, torch.distributed as dist
#
#     dist.init_process_group(backend="nccl", init_method="env://")
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#
#   That's it - no changes needed to the rest of your training logic.
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create result directory + Slurm scripts, optionally submit with sbatch."
    )
    p.add_argument("param_file", help="YAML parameter file (input).")
    p.add_argument(
        "-r", "--run", action="store_true",
        help="Submit the job via sbatch after creating scripts."
    )

    p.add_argument(
        "-p",
        "--partition",
        choices=["gpu", "gpu_requeue","gpu_h200", "arguelles_delgado_gpu_mixed"],
        default="gpu",
        help='SLURM partition: "gpu" (A100), "gpu_h200" (H200), or "arguelles_delgado_gpu_mixed" (A100 80GB GRES).',
    )

    p.add_argument("-g", "--num_gpu", type=int, default=1, help="GPUs per node. Default: 1")
    p.add_argument("-n", "--num_nodes", type=int, default=1, help="Number of nodes. Default: 1")

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

    # Create result_path using your project's helper
    params["result_path"] = util.setup_result_path(params["run_name"], args.param_file)
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
        gres_line = f"#SBATCH --gres=gpu:{args.num_gpu}"

    # Write run.sh (sbatch)
    job_script = JOB_SCRIPT_TEMPLATE.format(
        name=params.get("run_name", "allshowers"),
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        time_limit=args.time,
        partition=args.partition,
        gres_line=gres_line,
        num_nodes=args.num_nodes,
        num_gpus=args.num_gpu,
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