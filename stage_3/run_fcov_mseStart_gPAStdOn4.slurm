#!/usr/bin/env bash
#SBATCH --job-name=S3_On4
#SBATCH --output=S3_On4.log
#SBATCH --account=ecrpstats
#SBATCH --time=10:00:00
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
python fcov_GL_mseStart_gPAStdOn4.py
