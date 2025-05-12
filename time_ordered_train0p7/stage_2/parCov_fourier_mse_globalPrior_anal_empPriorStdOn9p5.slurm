#!/usr/bin/env bash
#SBATCH --job-name=toS2_On9p5
#SBATCH --output=toS2_On9p5.log
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
python parCov_fitting_fourier_allImages_mseStart_globalPrior_anal_empPriorStdOn9p5.py