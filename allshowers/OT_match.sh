#!/bin/bash
#SBATCH --job-name=ot_full
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --time=6:00:00
#SBATCH -p serial_requeue
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/ot_full_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/ot_full_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/



python /n/home04/hhanif/TAMBO-opt/allshowers/OT_match.py /n/home04/hhanif/TAMBO-opt/conf/allshowers_electrons.yaml --with-time