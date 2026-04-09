#!/bin/bash
#SBATCH --job-name=ot_full
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=100
#SBATCH --time=6:00:00
#SBATCH -p serial_requeue
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/ot_full_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/ot_full_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/



python /n/home04/hhanif/AllShowers/allshowers/OT_match.py /n/home04/hhanif/AllShowers/conf/allshowers/allshowers_muons.yaml --with-time