#!/usr/bin/env bash
#SBATCH --job-name=cn_774
#SBATCH --output=cn_774.log
#SBATCH --account=ecrpstats
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=harrison.goldwyn@nrel.gov
#SBATCH --ntasks-per-node=1

module purge
module load conda

conda activate torch
python compute_noise_singleImg.py 774
