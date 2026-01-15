#!/usr/bin/env bash
#SBATCH --job-name=sS2_On0p8
#SBATCH --output=sS2_On0p8.log
#SBATCH --account=ecrpstats
#SBATCH --time=01:00:00
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
python parCov_fitting_fourier_allImages_mseStart_globalPrior_anal_empPriorStdOn0p8.py