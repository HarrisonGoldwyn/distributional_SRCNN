#!/usr/bin/env bash
#SBATCH --job-name=s_2b_k4p5
#SBATCH --output=s_2b_k4p5.log
#SBATCH --account=ecrpstats
#SBATCH --time=1:00:00
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
python stage_2b_kappa4p5.py