#!/usr/bin/env bash
#SBATCH --job-name=t_2c_k0p7
#SBATCH --output=t_2c_k0p7.log
#SBATCH --account=ecrpstats
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=harrison.goldwyn@nrel.gov
#SBATCH --ntasks-per-node=1

module purge
module load conda

conda activate torch
python stage_2c_kappa0p7.py
