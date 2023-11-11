#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139
#SBATCH -t 120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name="calib_iter"
#SBATCH --error=log/calib_iter.%j.err
#SBATCH --output=log/calib_iter.%j.out


module purge all
source /projects/b1139/environments/pytorch-1.11-emodpy-py39/bin/activate

cd /projects/b1139/basel-hackathon-2023/simulations
python run_calib.py