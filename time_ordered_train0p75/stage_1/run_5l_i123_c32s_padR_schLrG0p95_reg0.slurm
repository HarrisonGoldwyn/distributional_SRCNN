#!/usr/bin/env bash
#SBATCH --job-name=s75_mse
#SBATCH --output=s75_mse.log
#SBATCH --account=ecrpstats
#SBATCH --time=02:00:00
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
python mse_5l_i123_c32s_padR_schLrG0p95_reg0.py
