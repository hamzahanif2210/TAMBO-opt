#!/bin/bash
#SBATCH --job-name=dataset_Reconstruction
#SBATCH --mem=500G
#SBATCH --cpus-per-task=90 
#SBATCH --time=1:00:00
#SBATCH -p serial_requeue
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/dataset_reconstruction_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/dataset_reconstruction_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/


python /n/home04/hhanif/AllShowers/util/dataset_for_reconstruction.py --electron-path /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_electrons_balanced.h5 --muon-path /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_muons_balanced.h5 --photon-path /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/combined_photons_balanced.h5 --num-workers 30 --output-path /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/reconstruction_for_training.h5 --chunk-size 5000 --overwrite 